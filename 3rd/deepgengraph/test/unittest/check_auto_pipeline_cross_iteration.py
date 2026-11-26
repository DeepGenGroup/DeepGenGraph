#!/usr/bin/env python3
"""Early marked unrolling, mixed Q/K/V streams and cross-iteration contracts."""
import re
import subprocess
from check_auto_pipeline_v_stream import fixture as v_fixture, TOOL


def fixture():
    text = v_fixture().replace('%other: memref<128x32xf16, 1>',
                              '%other: memref<128x128xf16, 1>, %q: memref<32x128xf16, 1>')
    inner = '''
%qk = affine.for %kk = 0 to 128 step 32 iter_args(%c = %acc) -> (memref<32x32xf32>) {
  %qs = frisk.buffer_view %q[0, %kk], ranges = [32, 32] : memref<32x128xf16, 1> -> memref<32x32xf16, 1>
  %qb = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x32xf16, 3>
  %qt = frisk.async_copy_gm_to_shm %qs to %qb {alignment = 16 : i64} : memref<32x32xf16, 1>, memref<32x32xf16, 3> -> !frisk.async_task
  frisk.wait_buffer_ready %qt : !frisk.async_task
  %ks = frisk.buffer_view %other[(%bn * 32), %kk], ranges = [32, 32] : memref<128x128xf16, 1> -> memref<32x32xf16, 1>
  %kb = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x32xf16, 3>
  %kt = frisk.async_copy_gm_to_shm %ks to %kb {alignment = 16 : i64} : memref<32x32xf16, 1>, memref<32x32xf16, 3> -> !frisk.async_task
  frisk.wait_buffer_ready %kt : !frisk.async_task
  %r = frisk.gemm(%qb, %kb, %c) {transA = false, transB = false} : memref<32x32xf16, 3>, memref<32x32xf16, 3>, memref<32x32xf32> -> memref<32x32xf32>
  affine.yield %r : memref<32x32xf32>
} {frisk.loopUnrollFull = true}
'''
    return text.replace('affine.for %bn = 0 to 4 {', 'affine.for %bn = 0 to 4 {\n' + inner)


def run(text, options='double-buffer=true cross-iteration=true'):
    p = subprocess.run([str(TOOL), '-', '--allow-unregistered-dialect',
                        '--frisk-auto-pipeline=' + options, '--verify-each'],
                       input=text, text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


def main():
    count = 0

    def check(source, cross, loops=1, v=4, options='double-buffer=true cross-iteration=true'):
        nonlocal count
        out = run(source, options)
        assert out.count('affine.for') == loops, out
        assert out.count('frisk.cross_iteration_plan') == cross, out
        assert out.count('frisk.cross_iteration_issue') == cross, out
        assert out.count('frisk.cross_iteration_anchor') == cross, out
        assert out.count('frisk.shared_pipeline_plan') == v, out
        assert run(out, options) == out, 'non-idempotent early planning'
        if cross:
            issues = [line for line in out.splitlines() if 'frisk.cross_iteration_issue' in line]
            anchors = [line for line in out.splitlines() if 'frisk.cross_iteration_anchor' in line]
            assert all('frisk.async_copy_gm_to_shm' in line for line in issues + anchors)
            assert 'frisk.shared_pipeline_plan' not in issues[0], 'selected V instead of K0'
            if v:
                assert 'array<i64: 3, 4, 1, 2>' in anchors[0], 'wrong tail readiness'
        count += 1
        return out

    base = fixture()
    out = check(base, 1)
    assert out.count('frisk.async_copy_gm_to_shm') == 12
    assert out.count('frisk.gemm') == 8
    # SSA carries the QK accumulator through all four expanded GEMMs.
    qk = re.findall(r'(%\w+) = frisk.gemm\([^,]+, [^,]+, (%\w+)\)', out)[:4]
    assert len(qk) == 4 and all(qk[i][1] == qk[i-1][0] for i in range(1, 4)), out
    check(base, 0, options='double-buffer=true cross-iteration=false')
    check(base, 1, v=0, options='double-buffer=false cross-iteration=true')
    check(base, 0, loops=2, v=0, options='double-buffer=false cross-iteration=false')
    check(base.replace('frisk.loopUnrollFull = true', 'frisk.loopUnrollFull = false'), 0, loops=2)
    check(base.replace('0 to 128 step 32', '0 to 128 step 1'), 0, loops=2)
    check(base.replace('frisk.loopUnrollFull = true', 'frisk.loopUnrollFull = true, frisk.pipeline_hint = 1 : i64'), 0, loops=2)
    check(base.replace('frisk.loopUnrollFull = true', 'frisk.loopUnrollFull = true, pipeline.scheduled'), 0, loops=2)
    check(base.replace('affine.for %bn = 0 to 4', 'affine.for %bn = 0 to 0'), 1)
    check(base.replace('affine.for %bn = 0 to 4', 'affine.for %bn = 0 to %tid'), 0, v=0)
    check(base.replace('%other[(%bn * 32), %kk]', '%other[0, %kk]'), 0)
    check(base.replace('%src0 =', '"test.unknown"() : () -> ()\n%src0 ='), 0)
    check(base.replace('%src0 =', '%z = arith.constant 0.0 : f16\nmemref.store %z, %other[%zero, %zero] : memref<128x128xf16, 1>\n%src0 ='), 0)
    check(base.replace('%pv3 = frisk.gemm(%a, %buf3, %acc)', '%pv3 = frisk.gemm(%buf3, %a, %acc)'), 0, v=3)
    legacy = base.replace('%qk = affine.for', '%init = frisk.cast %acc : memref<32x32xf32> -> vector<32x32xf32>\n%qk = affine.for')
    legacy = legacy.replace('iter_args(%c = %acc) -> (memref<32x32xf32>)',
                            'iter_args(%cv = %init) -> (vector<32x32xf32>)')
    legacy = legacy.replace('%r = frisk.gemm', '%c = frisk.cast %cv : vector<32x32xf32> -> memref<32x32xf32>\n%r = frisk.gemm')
    legacy = legacy.replace('affine.yield %r : memref<32x32xf32>',
                            'frisk.copy %r to %cv [affine_map<() -> (2)>] : memref<32x32xf32>, vector<32x32xf32>\naffine.yield %cv : vector<32x32xf32>')
    out = check(legacy, 1)
    assert 'frisk.copy' not in out and 'affine.yield' not in out
    check(legacy.replace('affine_map<() -> (2)>', 'affine_map<() -> (0)>'), 0, loops=2)
    print(f'PASS {count} early cross-iteration cases: unrolling, recurrence, mixed streams, ownership and rejection')


if __name__ == '__main__':
    main()
