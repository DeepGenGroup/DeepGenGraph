#!/usr/bin/env python3
"""Operand-independent retention legality, resource-budget and scope regressions."""
from pathlib import Path
import re
import subprocess
import sys

TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'build/test/ThreadTilingTest'


def fixture(name='operand_b', upper='4', source='%b', extra='', before='', n=16):
    reads = '\n'.join(f'    %x{i} = memref.load %tile[%tid, %c{i}] : memref<256x{n}xf16, 3>' for i in range(n))
    constants = '\n'.join(f'  %c{i} = arith.constant {i} : index' for i in range(n))
    values = ', '.join(f'%x{i}' for i in range(n))
    return f'''module {{
func.func @{name}(%a: memref<256x{n}xf16, 1>, %b: memref<256x{n}xf16, 1>, %n: index) -> vector<{n}xf16> attributes {{thread_num = 256 : i32}} {{
  {constants}
  %tid = gpu.thread_id x
  %zero = arith.constant dense<0.0> : vector<{n}xf16>
  %scale = arith.constant dense<0.5> : vector<{n}xf16>
  {before}
  %result = affine.for %i = 0 to {upper} iter_args(%acc = %zero) -> vector<{n}xf16> {{
    %tile = memref.alloc() alignment = 16 : memref<256x{n}xf16, 3>
    frisk.async_copy_gm_to_shm {source} to %tile {{alignment = 16 : i64}} : memref<256x{n}xf16, 1>, memref<256x{n}xf16, 3>
    frisk.wait_buffer_ready %tile : memref<256x{n}xf16, 3>
{reads}
    %v = vector.from_elements {values} : vector<{n}xf16>
    %scaled = arith.mulf %v, %scale : vector<{n}xf16>
    {extra}
    %next = arith.addf %acc, %scaled : vector<{n}xf16>
    affine.yield %next : vector<{n}xf16>
  }}
  return %result : vector<{n}xf16>
}}
}}
'''


def run(text, words=8):
    p = subprocess.run([str(TOOL), f'--frisk-retain-invariant-fragments=max-words={words}'],
                       input=text, text=True, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout


def retained(text, words):
    out = run(text, words)
    before, body = out.split('affine.for', 1)
    assert 'frisk.async_copy_gm_to_shm' in before, out
    assert 'frisk.wait_buffer_ready' in before, out
    assert 'arith.mulf' in before, out
    assert 'frisk.async_copy_gm_to_shm' not in body, out
    assert f'frisk.retained_words = {words} : i32' in out, out
    assert 'arith.addf' in body, out
    assert run(out, words) == out, 'not idempotent'
    return out


def rejected(text, words=8):
    out = run(text, words)
    assert 'frisk.retained_words' not in out, out
    assert 'frisk.async_copy_gm_to_shm' in out.split('affine.for', 1)[1], out


def main():
    cases = 0
    for name, source, n in [('operand_b', '%b', 16), ('renamed_a', '%a', 8), ('wide_operand', '%b', 32)]:
        retained(fixture(name=name, source=source, n=n), n//2)
        cases += 1
    rejected(fixture(), 7)
    rejected(fixture(), 0)
    rejected(fixture(upper='0'))
    rejected(fixture(upper='1'))
    rejected(fixture(upper='%n'))
    rejected(fixture(extra='memref.store %x0, %b[%c0, %c0] : memref<256x16xf16, 1>'))
    # Distinct arguments have no implicit noalias contract.
    rejected(fixture(extra='memref.store %x0, %a[%c0, %c0] : memref<256x16xf16, 1>'))
    rejected(fixture(extra='memref.store %x0, %tile[%c0, %c0] : memref<256x16xf16, 3>'))
    rejected(fixture(extra='%alias = memref.cast %tile : memref<256x16xf16, 3> to memref<?x16xf16, 3>'))
    rejected(fixture().replace('%tile[%tid, %c0]', '%tile[%i, %c0]'))
    # Readiness must precede every reader, and only one producer is permitted.
    wait = '    frisk.wait_buffer_ready %tile : memref<256x16xf16, 3>\n'
    rejected(fixture().replace(wait, '').replace('    %scaled =', wait+'    %scaled ='))
    copy = '    frisk.async_copy_gm_to_shm %b to %tile {alignment = 16 : i64} : memref<256x16xf16, 1>, memref<256x16xf16, 3>\n'
    rejected(fixture().replace(copy, copy+copy))
    # A thread-dependent trip count cannot justify collective prologue motion.
    rejected(fixture(upper='%bound', before='%bound = arith.addi %tid, %c1 : index'))
    # Known block ranges can prove a dynamic uniform loop nonempty.
    retained(fixture(upper='%bound', before='%bid = gpu.block_id x {range = 32 : index}\n  %bound = arith.addi %bid, %c1 : index'), 8)
    cases += 14
    # A region's implicit captures must be analyzed, not just its direct operands.
    region = fixture()
    begin, end = region.index('    %x0 ='), region.index('    %scaled =')
    read_body = region[begin:end].replace('%v =', '%packed =')
    region = region[:begin] + '    %v = frisk.fragment "load" -> (vector<16xf16>) {\n' + read_body + '      frisk.fragment_yield %packed : vector<16xf16>\n    }\n' + region[end:]
    retained(region, 8)
    rejected(region.replace('%tile[%tid, %c0]', '%tile[%i, %c0]'))
    # Two arbitrary operands share one budget; do not spend max-words per copy.
    two = fixture()
    begin, end = two.index('    %tile ='), two.index('    %next =')
    second = re.sub(r'%(tile|x\d+|v|scaled)\b', lambda m: m[0]+'_b', two[begin:end]).replace('%b to', '%a to')
    two = two[:end] + second + two[end:]
    two = two.replace('    %next = arith.addf %acc, %scaled', '    %partial = arith.addf %acc, %scaled : vector<16xf16>\n    %next = arith.addf %partial, %scaled_b')
    for budget, count in [(8,1), (16,2)]:
        out = run(two, budget)
        before, body = out.split('affine.for', 1)
        assert before.count('frisk.async_copy_gm_to_shm') == count
        assert body.count('frisk.async_copy_gm_to_shm') == 2-count
        assert f'frisk.retained_words = {budget} : i32' in out
    unknown = fixture(extra='func.call @opaque() : () -> ()').replace('module {', 'module {\nfunc.func private @opaque()')
    rejected(unknown)
    cases += 5
    print(f'PASS {cases} retention cases: arbitrary operand, size/budget, idempotence, alias/write, readiness, zero-trip, uniformity')


if __name__ == '__main__':
    main()
