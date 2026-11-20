#!/usr/bin/env python3
"""QK load motion, exact SSA graphs, grouped guards and rejection boundaries."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

from check_pv_interleave import ROOT, LLVM, fixture as pv_fixture, ids, result_graph


def fixture(chains=2, steps=8, warp=False, views=False, distinct=False):
    vt = 'vector<4xf32>'
    lines = ['module {', 'func.func private @unknown()',
             'func.func @kernel(%sm: memref<4096xf16, 3>, %out: memref<4xf32>, '
             f'%a: vector<4xf16>, %seed: {vt}, %idx: index) -> (' +
             ', '.join([vt] * chains) + ') {', '%i = arith.constant 0 : index',
             '%zero = arith.constant 0.0 : f32', 'gpu.barrier']
    for c in range(chains):
        for k in range(steps):
            packet = f'{c}_{k}' if distinct else str(k)
            if c == 0 or distinct:
                lines += [f'%r{packet} = frisk.lds_load %sm[%idx] {{alignment = 8 : i64, byte_offset = {(c*steps+k)*8} : i64}} : memref<4096xf16, 3> -> vector<4xf16>']
                if views:
                    for j in range(4):
                        lines += [f'%x{packet}_{j} = vector.extract %r{packet}[{j}] : f16 from vector<4xf16>']
                    lines += [f'%b{packet} = vector.from_elements ' + ', '.join(f'%x{packet}_{j}' for j in range(4)) + ' : vector<4xf16>']
            b = f'%b{packet}' if views else f'%r{packet}'
            prev = '%seed' if k == 0 else f'%s{c}_{k-1}'
            tag = f'test.id = "{c}:{k}"'
            if warp:
                lines += [f'%s{c}_{k} = frisk.warp_mma_rr %a, {b}, {prev} {{inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", inst_constraints = "=v,v,v,0", {tag}}} : vector<4xf16>, vector<4xf16>, {vt} -> {vt}']
            else:
                lines += [f'%s{c}_{k} = llvm.inline_asm has_side_effects convergent {{{tag}}} "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", "=v,v,v,0" %a, {b}, {prev} : (vector<4xf16>, vector<4xf16>, {vt}) -> {vt}']
            if c == 0 and k == 0:
                # A float select previously split the paired-LDS matching region.
                lines += ['%mask = arith.cmpi eq, %idx, %i : index',
                          '%masked = arith.select %mask, %zero, %zero : f32',
                          '// boundary']
    lines += ['gpu.barrier', 'return ' + ', '.join(f'%s{c}_{steps-1}' for c in range(chains)) +
              ' : ' + ', '.join([vt]*chains), '}', '}']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, default=ROOT/'build/test/ThreadTilingTest')
    args = parser.parse_args()
    cases = 0

    def run(source, *passes, ok=True):
        p = subprocess.run([str(args.tool), '--verify-each', *passes], input=source,
                           text=True, capture_output=True, timeout=120)
        assert (p.returncode == 0) == ok, p.stderr
        return p.stdout if ok else p.stderr

    def transform(source, width=2, guards=2, words=32, mode='qk'):
        return run(source, f'--frisk-thread-interleave-{mode}=width={width} guard-group={guards} prefetch-words={words}')

    for warp in (False, True):
        for views in (False, True):
            source = fixture(warp=warp, views=views)
            graph = result_graph(run(source))
            for width in (0, 1, 2):
                for guards in (-1, 0, 1, 2, 3, 4, 32):
                    out = transform(source, width, guards)
                    assert result_graph(out) == graph, 'changed MMA arithmetic graph'
                    w = width or 1
                    expected = [f'{c}:{k}' for g in range(0, 2, w)
                                for k in range(8) for c in range(g, g+w)]
                    assert ids(out) == expected
                    count = 0 if not width or not guards else 32 if guards == -1 else (16+guards-1)//guards
                    assert out.count('llvm.call @llvm.amdgcn.sched.barrier') == count
                    if width:
                        first = out.index('test.id')
                        assert 'frisk.lds_load' not in out[first:], 'late K packet'
                        if guards > 0:
                            chunks = out.split('llvm.call @llvm.amdgcn.sched.barrier')
                            assert [len(ids(x)) for x in chunks[:-1]] == [
                                min(guards, 16-i) for i in range(0, 16, guards)]
                    assert out.count('gpu.barrier') == 2
                    assert len(ids(out.split('gpu.barrier')[0])) == 0
                    assert len(ids(out.split('gpu.barrier')[1])) == 16
                    assert transform(out, width, guards) == out, 'replay changed schedule'
                    cleaned = run(out, '--canonicalize', '--cse')
                    assert ids(cleaned) == expected
                    assert cleaned.count('llvm.call @llvm.amdgcn.sched.barrier') == count
                    cases += 1

    # More than one pair, different packet sets, full vs partial guard groups.
    for mode, source in [('qk', fixture(chains=4, steps=3, distinct=True)),
                         ('pv', pv_fixture(steps=3))]:
        for width in (2, 4):
            for guards in (0, 2, 5):
                out = transform(source, width, guards, mode=mode)
                assert result_graph(out) == result_graph(run(source))
                assert ids(out) == [f'{c}:{k}' for g in range(0, 4, width)
                                    for k in range(3) for c in range(g, g+width)]
                assert out.count('llvm.call @llvm.amdgcn.sched.barrier') == (0 if not guards else (12+guards-1)//guards)
                assert transform(out, width, guards, mode=mode) == out
                cases += 1

    base = fixture()
    bad = {
        'barrier': base.replace('// boundary', 'gpu.barrier'),
        'call': base.replace('// boundary', 'func.call @unknown() : () -> ()'),
        'store': base.replace('// boundary', 'memref.store %masked, %out[%i] : memref<4xf32>'),
        'unrelated load': base.replace('// boundary', '%extra = memref.load %out[%i] : memref<4xf32>'),
        'existing guard': base.replace('module {', 'module {\nllvm.func @llvm.amdgcn.sched.barrier(i32)').replace(
            '// boundary', '%g = arith.constant 0 : i32\nllvm.call @llvm.amdgcn.sched.barrier(%g) : (i32) -> ()'),
        'nested control': base.replace('// boundary', 'scf.if %mask {\nscf.yield\n}'),
        'weak alignment': base.replace('alignment = 8', 'alignment = 4'),
        'protected pair': base.replace('byte_offset = 8 : i64', 'byte_offset = 8 : i64, frisk.k_lds_pair = 0 : i64'),
        'bad asm': base.replace('"=v,v,v,0"', '"=v,v,v,0,~{memory}"'),
        'single chain': fixture(chains=1),
        'single step': fixture(steps=1),
        'nondivisible': fixture(chains=3),
        'optnone': base.replace(') {\n', ') attributes {llvm.optnone} {\n'),
        'strictfp': base.replace(') {\n', ') attributes {llvm.strictfp} {\n'),
    }
    # Change only the second chain's seed; the resulting chain exceeds depth 8
    # and has no independent partner. No partially scheduled first chain.
    bad['cross chain'] = '\n'.join(
        line.replace('%a, %r0, %seed', '%a, %r0, %s0_7') if 'test.id = "1:0"' in line else line
        for line in base.splitlines())
    async_base = base.replace('%idx: index)',
        '%idx: index, %global: memref<64x64xf16, 1>, %stage: memref<64x64xf16, 3>, %task: !frisk.async_task)')
    bad['DMA issue'] = async_base.replace('// boundary',
        '%newtask = frisk.async_copy_gm_to_shm %global to %stage : memref<64x64xf16, 1>, memref<64x64xf16, 3> -> !frisk.async_task')
    bad['readiness wait'] = async_base.replace('// boundary',
        'frisk.wait_buffer_ready %task : !frisk.async_task')
    bad['mixed packet view'] = fixture(views=True).replace(
        '%b1 = vector.from_elements %x1_0', '%b1 = vector.from_elements %x0_0')
    # Address depends on an earlier MMA: moving the read above it is illegal.
    bad['dependent address'] = base.replace('// boundary',
        '%lane = vector.extract %s0_0[0] : f32 from vector<4xf32>\n'
        '%int = arith.fptosi %lane : f32 to i32\n%addr = arith.index_cast %int : i32 to index').replace(
        '%r1 = frisk.lds_load %sm[%idx]', '%r1 = frisk.lds_load %sm[%addr]')
    for name, source in bad.items():
        try:
            assert transform(source) == run(source)
        except AssertionError as error:
            raise AssertionError(name) from error
        cases += 1
    for words in (0, 15):
        assert transform(base, words=words) == run(base), 'operand budget ignored'
        cases += 1
    assert len(ids(transform(base, words=16))) == 16
    for mode in ('pv', 'qk'):
        for guards in (-2, 33):
            assert 'guard-group' in run(base, f'--frisk-thread-interleave-{mode}=width=2 guard-group={guards}', ok=False)
            cases += 1
    for width in (-1, 3, 8):
        run(base, f'--frisk-thread-interleave-qk=width={width}', ok=False)
        cases += 1

    # Source kinds do not accidentally match each other's regions.
    assert transform(base, mode='pv') == run(base)
    assert transform(pv_fixture(), mode='qk') == run(pv_fixture())
    cases += 2
    with tempfile.TemporaryDirectory(prefix='thread-mma-') as tmp:
        tmp = Path(tmp)
        for mode, source, guards in [('qk', base, 2), ('qk', base, 0), ('pv', pv_fixture(), 3)]:
            out = transform(source, guards=guards, mode=mode)
            lowered = run(out, '--test-async-to-llvm')
            p = subprocess.run([str(LLVM/'mlir-translate'), '--mlir-to-llvmir'],
                               input=lowered, text=True, capture_output=True)
            assert p.returncode == 0, p.stderr
            (tmp/'kernel.ll').write_text(p.stdout)
            p = subprocess.run([str(LLVM/'opt'), '-S', '-passes=default<O3>,verify',
                                str(tmp/'kernel.ll'), '-o', str(tmp/'kernel.o3.ll')], capture_output=True, text=True)
            assert p.returncode == 0, p.stderr
            optimized = (tmp/'kernel.o3.ll').read_text()
            assert optimized.count('call void @llvm.amdgcn.sched.barrier(i32 0)') == out.count('llvm.call @llvm.amdgcn.sched.barrier')
            if guards:
                assert 'memory(inaccessiblemem: readwrite)' in optimized
            cases += 1
        for option in ('--qk-interleave=3', '--pv-guard-group=-2', '--qk-guard-group=33', '--qk-prefetch-words=-1'):
            p = subprocess.run([str(args.tool.parent/'MyTest'), str(ROOT/'test/test_input_shared_d.mlir'),
                                '0', str(tmp/'unused.ll'), option], capture_output=True, text=True, timeout=30)
            assert p.returncode and 'invalid MMA schedule option' in p.stderr
            cases += 1
    print(f'PASS {cases} MMA scheduling cases (QK, PV groups, SSA, effects, budget, replay, LLVM/O3)')


if __name__ == '__main__':
    main()
