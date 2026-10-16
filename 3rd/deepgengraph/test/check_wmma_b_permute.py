#!/usr/bin/env python3
"""Execute generated LowerInfo maps against the wave64 MMAC contract on CPU.

Usage: check_wmma_b_permute.py build/test/ThreadTilingTest
Nonuniform A/B values catch column permutations, lane ownership, K fragments,
and duplicate/missing output elements. This is not a GPU performance test.
"""
import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOL = Path(sys.argv[1]).resolve()


def run(directory, name, source, mode, *extra):
    src, dst = directory / (name + '.mlir'), directory / (name + '.out.mlir')
    src.write_text(source)
    # This interpreter consumes explicit to_threadTile operand maps. Collective
    # matrix reads replace that boundary and are executed by the matrix-load
    # and exported LLVM numerical tests; retain the ordinary-read control here.
    p = subprocess.run([str(TOOL), str(src),
                        '--convert-friskbase-to-thread=matrix-loads=false wmma-layout=' + mode,
                        *extra, '--verify-each', '-o', str(dst)],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, name + '\n' + p.stderr[-6000:]
    return dst.read_text(), p.stdout


def gemm(m, n, k, threads, chain=True):
    a, b, c = f'memref<{m}x{k}xf16, 3>', f'memref<{k}x{n}xf16, 3>', f'memref<{m}x{n}xf32>'
    p, v, o = f'memref<{m}x{n}xf16, 3>', f'memref<{n}x64xf16, 3>', f'memref<{m}x64xf32>'
    args, results = (f', %v: {v}', f'({c}, {o})') if chain else ('', c)
    suffix = f'''%p = frisk.cast %c : {c} -> {p}
      %o = frisk.gemm(%p, %v) {{transA = false, transB = false}} : {p}, {v} -> {o}
      return %c, %o : {c}, {o}''' if chain else f'return %c : {c}'
    return f'''func.func @gemm(%a: {a}, %b: {b}{args}) -> {results}
      attributes {{thread_num = {threads} : i32}} {{
      %c = frisk.gemm(%a, %b) {{transA = false, transB = false}} : {a}, {b} -> {c}
      {suffix}
    }}'''


class Layout:
    def __init__(self, line, ir):
        self.attrs = dict((name, tuple(map(int, values.split(',')))) for name, values in
                         re.findall(r'(\w+) = array<i64: ([^>]+)>', line))
        maps = dict(re.findall(r'^(#\w+) = (affine_map<.*>)$', ir, re.M))
        name = re.search(r'tile_layout = (#[\w]+|affine_map<.*?>)', line)[1]
        text = maps.get(name, name)
        results = re.search(r'-> \((.*)\)>', text)[1].split(',')
        self.code = []
        for result in results:
            tree = ast.parse(result.strip().replace('floordiv', '//').replace('mod', '%'), mode='eval')
            assert all(isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                ast.Name, ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult,
                ast.FloorDiv, ast.Mod, ast.USub)) for node in ast.walk(tree))
            self.code.append(compile(tree, '<generated tile_layout>', 'eval'))

    def coord(self, tid, row, col):
        w, r, u = [self.attrs[key] for key in ('thread_widths', 'warp_repeat', 'warp_inst_unroll')]
        br, iu, wr, reg = [], [], [], []
        for d, value in enumerate((row, col)):
            br.append(value // (w[d] * r[d] * u[d]))
            iu.append(value // (w[d] * r[d]) % u[d])
            wr.append(value // w[d] % r[d])
            reg.append(value % w[d])
        def flatten(values, shape, order):
            return values[order[0]] + values[order[1]] * shape[order[0]]
        args = [tid, *br, *iu,
                flatten(wr, r, self.attrs['warp_repeat_order']),
                flatten(reg, w, self.attrs['thread_creg_order'])]
        env = {f'd{i}': v for i, v in enumerate(args)}
        return tuple(eval(c, {'__builtins__': {}}, env) for c in self.code)


def execute_mmac(ir, m, n, k, threads):
    lines = ir.splitlines()
    a_line = next(s for s in lines if 'frisk.to_threadTile %arg0 ' in s)
    b_line = next(s for s in lines if 'frisk.to_threadTile %arg1 ' in s)
    c_line = next(s for s in lines if 'frisk.from_threadTile' in s)
    assert 'logical_transform' not in a_line
    assert 'logical_transform' in b_line and 'logical_transform' in c_line
    la, lb, lc = [Layout(s, ir) for s in (a_line, b_line, c_line)]
    rng = np.random.default_rng(m + n + k + threads)
    a = rng.normal(size=(m, k)).astype(np.float16).astype(np.float32)
    b = rng.normal(size=(k, n)).astype(np.float16).astype(np.float32)
    out, seen = np.zeros((m, n), np.float32), np.zeros((m, n), np.int32)
    bm, bn = lc.attrs['block_layout']
    for wave in range(threads // 64):
        for mi in range(m // (16 * bm)):
            for ni in range(n // (16 * bn)):
                acc = np.zeros((16, 16), np.float32)
                for ki in range(k // 16):
                    aa, bb = np.empty((16, 16), np.float32), np.empty((16, 16), np.float32)
                    for lane in range(64):
                        tid = wave * 64 + lane
                        for reg in range(4):
                            aa[lane % 16, lane // 16 * 4 + reg] = a[la.coord(tid, mi, ki * 4 + reg)]
                            bb[lane // 16 * 4 + reg, lane % 16] = b[lb.coord(tid, ki * 4 + reg, ni)]
                    acc += aa @ bb
                for lane in range(64):
                    columns = []
                    for reg in range(4):
                        dst = lc.coord(wave * 64 + lane, mi, ni * 4 + reg)
                        out[dst] = acc[lane % 16, lane // 16 + 4 * reg]
                        seen[dst] += 1
                        columns.append(dst[1])
                    assert columns == list(range(columns[0], columns[0] + 4))
    assert np.all(seen == 1), 'C layout is not a bijection'
    np.testing.assert_allclose(out, a @ b, rtol=2e-5, atol=2e-5)
    # Check the explicit column sequence from the supplied diagram as well.
    assert [lb.coord(j, 0, 0)[1] for j in range(16)] == [
        0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51]
    return out.astype(np.float16).astype(np.float32)


def execute_pv(ir, m, k, threads, p):
    lines = ir.splitlines()
    inputs = [s for s in lines if 'frisk.to_threadTile' in s]
    outputs = [s for s in lines if 'frisk.from_threadTile' in s]
    producer, lc = Layout(outputs[0], ir), Layout(outputs[-1], ir)
    direct = 'logical_transform' in inputs[-1]
    assert direct == (producer.attrs['block_layout'][1] == 1)
    if not direct:
        assert 'memref.alloc' in ir, 'incompatible wave tiling must keep its exchange'
        return
    la, lb = [Layout(s, ir) for s in inputs[-2:]]
    assert 'memref.alloc' not in ir, 'P unexpectedly staged through LDS'
    rng = np.random.default_rng(m + k + threads)
    v = rng.normal(size=(k, 64)).astype(np.float16).astype(np.float32)
    out, seen = np.zeros((m, 64), np.float32), np.zeros((m, 64), np.int32)
    bm, bn = lc.attrs['block_layout']
    for wave in range(threads // 64):
        for mi in range(m // (16 * bm)):
            for ni in range(64 // (16 * bn)):
                acc = np.zeros((16, 16), np.float32)
                for ki in range(k // 16):
                    aa, bb = np.empty((16, 16), np.float32), np.empty((16, 16), np.float32)
                    for lane in range(64):
                        tid = wave * 64 + lane
                        for reg in range(4):
                            coord = la.coord(tid, mi, ki * 4 + reg)
                            if direct:
                                assert coord == producer.coord(tid, mi, ki * 4 + reg)
                            aa[lane % 16, lane // 16 * 4 + reg] = p[coord]
                            bb[lane // 16 * 4 + reg, lane % 16] = v[lb.coord(tid, ki * 4 + reg, ni)]
                    acc += aa @ bb
                # Terminal C uses the separately tested contiguous-C shuffle.
                for lane in range(64):
                    for reg in range(4):
                        dst = lc.coord(wave * 64 + lane, mi, ni * 4 + reg)
                        out[dst] = acc[lane % 16, lane // 16 * 4 + reg]
                        seen[dst] += 1
    assert np.all(seen == 1), 'PV output has duplicate/missing coordinates'
    np.testing.assert_allclose(out, p @ v, rtol=2e-5, atol=2e-4)


def main():
    with tempfile.TemporaryDirectory(prefix='wmma-b-permute-') as tmp:
        directory = Path(tmp)
        for dims in [(16, 64, 16, 64), (64, 128, 32, 128),
                     (128, 128, 32, 128), (32, 192, 64, 64),
                     (64, 128, 32, 256), (128, 192, 32, 512)]:
            source = gemm(*dims)
            name = '-'.join(map(str, dims))
            ir, log = run(directory, name, source, 'b-permute')
            assert log.count('frisk.wmma_layout_applied = "b-permute"') == 1
            # Terminal C retains its old layout. Its B may share A's K
            # permutation, but must not acquire the producer's N permutation.
            terminal_c = [s for s in ir.splitlines() if 'frisk.from_threadTile' in s][-1]
            assert 'logical_transform' not in terminal_c
            assert 'gpu.shuffle' in ir
            p = execute_mmac(ir, *dims)
            execute_pv(ir, dims[0], dims[1], dims[3], p)
            again, _ = run(directory, name + '-again', ir, 'b-permute')
            assert again == ir, 'tiling must be idempotent'
            print('PASS numerical QK + PV / paired K / direct P or wave-layout fallback:', dims)
        source = gemm(32, 32, 16, 64)
        fallback, log = run(directory, 'fallback', source, 'b-permute')
        assert 'shuffle-fallback' in log and 'logical_transform' not in fallback
        assert 'gpu.shuffle' in fallback
        shared, _ = run(directory, 'shared', source, 'shared')
        shuffle, _ = run(directory, 'shuffle', source, 'shuffle')
        assert 'memref.alloc' in shared and 'gpu.shuffle' in shared
        assert 'gpu.shuffle' in shuffle and 'memref.alloc' not in shuffle
        plain = gemm(32, 64, 16, 64, chain=False)
        outputs = [run(directory, 'plain-' + mode, plain, mode)[0]
                   for mode in ('shared', 'shuffle', 'b-permute')]
        # The kernel summary is taken from the first entry of an unordered
        # LowerInfo map (A/C or B); compare all actual per-value layouts and
        # operations, without depending on that representative's lane order.
        outputs = [re.sub(r'(?m)^(\s*func.func[^\n]*)warp_layout = array<i64: [^>]+>',
                          r'\1warp_layout = <summary>', text) for text in outputs]
        assert outputs[0] == outputs[1] == outputs[2]
        b_only = '''func.func @b_only(%a: memref<32x16xf16, 3>, %b: memref<16x64xf16, 3>,
            %x: memref<64x32xf16, 3>) -> memref<64x64xf32> attributes {thread_num = 64 : i32} {
          %g = frisk.gemm(%a, %b) {transA = false, transB = false} : memref<32x16xf16, 3>, memref<16x64xf16, 3> -> memref<32x64xf32>
          %p = frisk.cast %g : memref<32x64xf32> -> memref<32x64xf16, 3>
          %o = frisk.gemm(%x, %p) {transA = false, transB = false} : memref<64x32xf16, 3>, memref<32x64xf16, 3> -> memref<64x64xf32>
          return %o : memref<64x64xf32>
        }'''
        b_ir, b_log = run(directory, 'b-only', b_only, 'b-permute')
        assert 'logical_transform' not in b_ir and 'frisk.b_permute' not in b_log
        # A computed B is a register value too. It cannot adopt a different K
        # ownership merely by changing its read coordinates.
        source = gemm(32, 64, 16, 64)
        vtype = 'memref<64x64xf16, 3>'
        source = source.replace('%o = frisk.gemm(%p, %v)',
            f'%vv = frisk.add %v, %v : {vtype}, {vtype} -> {vtype}\n'
            '      %o = frisk.gemm(%p, %vv)')
        computed, _ = run(directory, 'computed-b', source, 'b-permute')
        assert 'memref.alloc' in computed, 'computed B must keep the safe layout conversion'
        print('PASS P-only selection, terminal/unrelated GEMMs unchanged, 3.1/3.2 and 3.3 fallback')


if __name__ == '__main__':
    main()
