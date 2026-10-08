#!/usr/bin/env python3
"""Check preserved fragment boundaries, lookahead legality and CPU semantics."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from check_gemm_pv_addresses import Addresses, split_args

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / 'build/test/ThreadTilingTest'
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')


def run(command):
    p = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, f'{command}\n{p.stderr[-6000:]}\n{p.stdout[-2000:]}'
    return p.stdout


def fixture(name, lower=0, upper=4, step=1, extra='', address='%i'):
    return f'''
func.func @{name}(%a: memref<8xf32>, %base: index) -> f32 {{
  %zero = arith.constant 0.0 : f32
  %out = affine.for %i = {lower} to {upper} step {step} iter_args(%acc = %zero) -> f32 {{
    %index = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%i)[%base]
    %v = frisk.fragment "load" -> (vector<1xf32>) {{
      %x = memref.load %a[{address}] : memref<8xf32>
      %p = vector.broadcast %x : f32 to vector<1xf32>
      frisk.fragment_yield %p : vector<1xf32>
    }}
    {extra}
    %sum = frisk.fragment "compute" -> (f32) {{
      %x = vector.extract %v[0] : f32 from vector<1xf32>
      %y = arith.addf %acc, %x : f32
      frisk.fragment_yield %y : f32
    }}
    affine.yield %sum : f32
  }} {{frisk.fragment_loop}}
  return %out : f32
}}
'''


FUSED = r'''func.func @fused(%a: memref<8xf32>, %base: index) -> f32 {
  %z = arith.constant dense<0.0> : vector<4xf32>
  %zero = arith.constant 0.0 : f32
  %tile = affine.for %i = 0 to 4 iter_args(%v = %z) -> vector<4xf32> {
    %x = frisk.fragment "load" -> (f32) {
      %r = memref.load %a[%i] : memref<8xf32>
      frisk.fragment_yield %r : f32
    }
    %r = vector.insert %x, %v[%i] : f32 into vector<4xf32>
    affine.yield %r : vector<4xf32>
  } {frisk.fragment_loop}
  %sum = affine.for %i = 0 to 4 iter_args(%acc = %zero) -> f32 {
    %r = frisk.fragment "compute" -> (f32) {
      %x = vector.extract %tile[%i] : f32 from vector<4xf32>
      %r = arith.addf %acc, %x : f32
      frisk.fragment_yield %r : f32
    }
    affine.yield %r : f32
  } {frisk.fragment_loop}
  return %sum : f32
}
'''


def check_mma_addresses(text):
    # Trace the emitted LLVM operands back to physical byte addresses, keeping
    # every lane/register distinct. This catches wrong MN/K fragment selection
    # even when ordinary numerical fixtures contain equal values.
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))

    class ByteAddresses(Addresses):
        def value(self, name):
            if name not in self.cache and name in self.definitions:
                opcode, body = self.definitions[name].split(' ', 1)
                if opcode == 'getelementptr':
                    ty, pointer, index = split_args(body)
                    buffer, offset = self.typed(pointer)
                    size = {'i8': 1, 'half': 2, 'float': 4}[ty.split()[-1]]
                    self.cache[name] = (buffer, offset + size * self.typed(index))
                elif opcode == 'fmul':
                    # Q scaling preserves the source coordinates.
                    self.cache[name] = self.typed(split_args(body)[0])
            return super().value(name)

    tid = next(n for n, e in definitions.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    mma = [n for n, e in definitions.items() if 'asm sideeffect "' in e and 'v_mmac_' in e]
    operands = [split_args(re.search(r'"\((.*)\)', definitions[n]).group(1)) for n in mma]
    assert len(mma) == 64  # QK: 4 MN * 8 K; PV: 16 MN * 2 K.
    for start, k_count, n_count, n_width, k_width in [(0,8,2,32,128), (32,2,8,128,32)]:
        evaluator = ByteAddresses(definitions, {tid: 0})
        a_base = evaluator.typed(operands[start][0])[0]
        b_base = evaluator.typed(operands[start][1])[0]
        for i in range(32):
            k = i % k_count
            expected = '<4 x float> ' + ('zeroinitializer' if k == 0 else mma[start+i-1])
            assert operands[start+i][2] == expected, (i, operands[start+i])
        for thread in range(128):
            evaluator = ByteAddresses(definitions, {tid: thread})
            lane = thread % 64
            for i in range(32):
                tile, k = divmod(i, k_count)
                row = tile // n_count * 32 + thread // 64 * 16 + lane % 16
                col = tile % n_count * 16 + lane % 16
                kk = k * 16 + lane // 16 * 4
                a = evaluator.typed(operands[start+i][0])
                b = evaluator.typed(operands[start+i][1])
                expected_a = [(a_base[0], a_base[1] + 2*(row*k_width+kk+r)) for r in range(4)]
                expected_b = [(b_base[0], b_base[1] + 2*((kk+r)*n_width+col)) for r in range(4)]
                assert a == expected_a, (thread, i, a, expected_a)
                assert b == expected_b, (thread, i, b, expected_b)
    print('PASS attention: 65536 QK/PV operand addresses and 64 accumulator inputs')


def main():
    with tempfile.TemporaryDirectory(prefix='fragment-schedule-') as directory:
        directory = Path(directory)
        cases = {
            'ordinary': (fixture('ordinary'), True, 10.0),
            'fused': (FUSED, True, 10.0),
            'nonzero_step': (fixture('nonzero_step', 1, 8, 2), True, 20.0),
            'symbolic': (fixture('symbolic', address='%index'), True, 10.0),
            'one': (fixture('one', 0, 1), False, 1.0),
            'empty': (fixture('empty', 0, 0), False, 0.0),
            'barrier': (fixture('barrier', extra='gpu.barrier'), False, None),
            'write': (fixture('write', extra='memref.store %zero, %a[%i] : memref<8xf32>'), False, None),
            'carried_address': (fixture('carried_address').replace(
                '%index = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%i)[%base]',
                '%integer = arith.fptosi %acc : f32 to i32\n    %index = arith.index_cast %integer : i32 to index').replace('%a[%i]', '%a[%index]'), False, None),
        }
        cpu, expected = [], []
        for name, (text, should_pipeline, answer) in cases.items():
            src, dst = directory / f'{name}.mlir', directory / f'{name}.out.mlir'
            src.write_text(text)
            run([TOOL, src, '--frisk-frag-ir-reorder', '--verify-each', '-o', dst])
            ir = dst.read_text()
            assert ('frisk.pipelined' in ir) == should_pipeline, (name, ir)
            if name == 'fused':
                assert 'frisk.fused' in ir, ir
            if should_pipeline:
                assert '"prologue"' in ir and '"prefetch"' in ir
                assert ir.index('"prefetch"') < ir.index('frisk.fragment "compute"')
                # The steady loop ends before the final valid index, so no
                # out-of-bounds speculative read is emitted for the epilogue.
                assert ir.count('frisk.fragment "load"') == 2, ir
                assert ir.count('frisk.fragment "compute"') == 2, ir
            twice = directory / f'{name}.twice.mlir'
            run([TOOL, dst, '--frisk-frag-ir-reorder', '-o', twice])
            assert ir == twice.read_text(), (name, 'non-idempotent schedule')
            if answer is not None:
                cpu.append(text)
                expected.append((name, answer))
            print('PASS schedule', name)
        main_ir = '''func.func @main() -> i32 {
  %a = memref.alloca() : memref<8xf32>
  %base = arith.constant 0 : index
  %data = arith.constant dense<[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]> : vector<8xf32>
  vector.store %data, %a[%base] : memref<8xf32>, vector<8xf32>
  %ok0 = arith.constant true
'''
        for n, (name, answer) in enumerate(expected, 1):
            main_ir += f'''  %r{n} = func.call @{name}(%a, %base) : (memref<8xf32>, index) -> f32
  %e{n} = arith.constant {answer} : f32
  %c{n} = arith.cmpf oeq, %r{n}, %e{n} : f32
  %ok{n} = arith.andi %ok{n-1}, %c{n} : i1
'''
        main_ir += f'''  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %status = arith.select %ok{len(expected)}, %zero, %one : i32
  return %status : i32
}}
'''
        source = directory / 'cpu.mlir'
        source.write_text('\n'.join(cpu) + main_ir)
        for mode in ['baseline', 'scheduled']:
            plain, low, ll = [directory / (mode + suffix) for suffix in ['.mlir', '.llvm.mlir', '.ll']]
            passes = ['--frisk-frag-ir-reorder'] if mode == 'scheduled' else []
            run([TOOL, source, *passes, '--lower-frisk-fragments', '-o', plain])
            assert 'frisk.fragment ' not in plain.read_text()
            run([LLVM/'mlir-opt', plain, '--lower-affine', '--convert-scf-to-cf',
                 '--convert-vector-to-llvm', '--finalize-memref-to-llvm', '--convert-arith-to-llvm',
                 '--convert-func-to-llvm', '--convert-cf-to-llvm', '--reconcile-unrealized-casts', '-o', low])
            run([LLVM/'mlir-translate', '--mlir-to-llvmir', low, '-o', ll])
            run([LLVM/'lli', ll])
            print('PASS CPU', mode)
        base = HERE / 'test_friskBaseDebug.mlir'
        tile, finalized, scheduled, late = [directory / (x+'.mlir') for x in ['tile','final','scheduled','late']]
        run([TOOL, base, '--convert-friskbase-to-thread', '-o', tile])
        run([TOOL, tile, '--finalize-thread-tiling', '-o', finalized])
        for file in [tile, finalized]:
            text = file.read_text()
            assert text.count('frisk.warp_mma_rr') == 2, file
            assert text.count('iterLabel = "gemm_k"') == 2, file
            assert text.count('iterLabel = "gemm_mn"') == 2, file
            assert 'frisk.shm_pool' not in text, 'reuse must wait for scheduling'
        run([TOOL, finalized, '--frisk-frag-ir-reorder', '-o', scheduled])
        text = scheduled.read_text()
        assert 'frisk.pipelined' in text and 'frisk.fused' in text
        assert text.count('"prefetch"') >= 2, 'GEMM operands should be pipelined'
        run([TOOL, scheduled, '--lower-frisk-fragments', '-o', late])
        assert not re.search(r'frisk\.fragment(?:\s|_yield\b)', late.read_text()), 'all fragment regions must be lowered'
        print('PASS attention: preserved MN/K, scheduled prefetch, late lowering')

        # Exercise the production ordering as well: affine LICM must only run
        # after fragment regions have been inlined, because this LLVM version
        # does not inspect captures in unknown region operations.
        full_tool = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else TOOL.with_name('MyTest')
        result = subprocess.run([str(full_tool), str(HERE/'test_input.mlir'), '0'],
                                cwd=directory, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=120)
        assert result.returncode == 0, result.stdout[-8000:]
        log = result.stdout
        scheduled_ir = log.split('---------- after createFinalizeThreadTilingPass ---------', 1)[1].split(
            '---- after lower-frisk-fragments -----', 1)[0]
        assert 'frisk.pipelined' in scheduled_ir and '"prefetch"' in scheduled_ir
        optimized = log.split('---- after createIRDeepOptimizePass -----', 1)[1].split(
            '---- after affine-scalrep -----', 1)[0]
        assert not re.search(r'frisk\.fragment(?:\s|_yield\b)', optimized)
        for line in optimized.splitlines():
            if re.search(r'vector\.(insert|extract)\s', line):
                assert '%' not in line.split('[', 1)[1].split(']', 1)[0], line
        ll = directory/'finalLLVMText.ll'
        assert ll.exists()
        for line in ll.read_text().splitlines():
            if 'extractelement ' in line or 'insertelement ' in line:
                assert not re.search(r', i(?:32|64) %[^ ,]+(?:, !dbg.*)?$', line), line
        run([LLVM/'llvm-as', ll, '-o', directory/'full.bc'])
        check_mma_addresses(ll.read_text())
        print('PASS full attention: LLVM export and static vector indices')


if __name__ == '__main__':
    main()
