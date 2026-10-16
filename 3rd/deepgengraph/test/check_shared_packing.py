#!/usr/bin/env python3
"""Generic LDS packing: scalar semantics, conservative fallback and GEMM lowering.

CPU execution verifies the address permutation after removing only address space
3. GPU instruction throughput / bank conflicts still require device profiling.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent/'build/test/ThreadTilingTest'
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')


def run(args, source=None):
    p = subprocess.run(list(map(str, args)), input=source, capture_output=True,
                       text=True, timeout=180)
    assert p.returncode == 0, p.stderr[-8000:]
    return p.stdout


def fixture(rows, cols, pack, axis=0):
    ty = f'memref<{rows}x{cols}xf32, 3>'
    text = f'''func.func @read_group(%group: index, %col: index) -> f32 {{
  %a = memref.alloc() alignment = 16 {{frisk.shared_pack = {pack} : i64}} : {ty}
  affine.for %r = 0 to {rows} {{
    affine.for %c = 0 to {cols} {{
      %idx = affine.apply affine_map<(d0,d1) -> (d0 * 37 + d1 * 3)>(%r, %c)
      %i = arith.index_cast %idx : index to i32
      %v = arith.sitofp %i : i32 to f32
      affine.store %v, %a[%r,%c] : {ty}
    }}
  }}
'''
    for i in range(pack):
        text += f'''  %r{i} = affine.apply affine_map<(d0) -> (d0 * {pack} + {i})>(%group)
  %v{i} = memref.load %a[%r{i},%col] : {ty}
'''
    text += '  %z = arith.constant 0.0 : f32\n'
    prev = '%z'
    for i in range(pack):
        text += f'  %s{i} = arith.addf {prev}, %v{i} : f32\n'
        prev = f'%s{i}'
    # A second consumer observes an intervening write. Coalescing must not
    # hoist it across that write, nor merge the two observations of a location.
    text += f'''  memref.store %z, %a[%r0,%col] : {ty}
  %new = affine.load %a[%r0,%col] : {ty}
  %out = arith.addf {prev}, %new : f32
  memref.dealloc %a : {ty}
  return %out : f32
}}
func.func @main() -> i32 {{
  %zero = arith.constant 0.0 : f32
  %sum = affine.for %g = 0 to {rows//pack} iter_args(%acc = %zero) -> f32 {{
    %inner = affine.for %c = 0 to {cols} iter_args(%s = %acc) -> f32 {{
      %v = func.call @read_group(%g,%c) : (index,index) -> f32
      %wantidx = affine.apply affine_map<(d0,d1) -> (d0 * {pack*pack*37} + d1 * {pack*3} + {pack*(pack-1)//2*37})>(%g,%c)
      %wanti = arith.index_cast %wantidx : index to i32
      %want = arith.sitofp %wanti : i32 to f32
      %ok = arith.cmpf oeq, %v, %want : f32
      %bad = arith.constant -1.0e9 : f32
      %checked = arith.select %ok, %v, %bad : f32
      %n = arith.addf %s, %checked : f32
      affine.yield %n : f32
    }}
    affine.yield %inner : f32
  }}
  %expected = arith.constant {float(sum(r*37+c*3 for r in range(rows) for c in range(cols)))} : f32
  %ok = arith.cmpf oeq, %sum, %expected : f32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %status = arith.select %ok, %c0, %c1 : i32
  return %status : i32
}}
'''
    if axis == 1:
        text = text.replace(ty, f'memref<{cols}x{rows}xf32, 3>')
        text = text.replace(f'frisk.shared_pack = {pack} : i64',
                            f'frisk.shared_pack = {pack} : i64, frisk.shared_pack_axis = 1 : i64')
        text = re.sub(r'%a\[([^,]+),([^\]]+)\]', r'%a[\2,\1]', text)
    return text


def cpu(text, directory, name):
    src, low, ll = [directory/(name+suffix) for suffix in ('.mlir', '.low.mlir', '.ll')]
    src.write_text(text.replace(', 3>', '>'))
    run([LLVM/'mlir-opt', src, '--lower-affine', '--convert-scf-to-cf',
         '--convert-vector-to-llvm', '--finalize-memref-to-llvm',
         '--convert-arith-to-llvm', '--convert-func-to-llvm', '--convert-cf-to-llvm',
         '--reconcile-unrealized-casts', '-o', low])
    run([LLVM/'mlir-translate', '--mlir-to-llvmir', low, '-o', ll])
    run([LLVM/'lli', ll])


def gemm(m, n, k, threads, extra=''):
    a, b, c = f'memref<{m}x{k}xf16, 3>', f'memref<{k}x{n}xf16, 3>', f'memref<{m}x{n}xf32>'
    return f'''func.func @matmul(%a: {a}, %global: memref<{k}x{n}xf16, 1>, %out: memref<{m}x{n}xf32, 1>) attributes {{thread_num = {threads} : i32}} {{
  %b = frisk.alloc_buffer {{scope = "shared", alignment = 16}} -> {b}
  frisk.copy %global to %b [affine_map<() -> (2)>] : memref<{k}x{n}xf16, 1>, {b}
  {extra}
  %c = frisk.gemm(%a, %b) {{transA = false, transB = false}} : {a}, {b} -> {c}
  frisk.copy %c to %out [affine_map<() -> (2)>] : {c}, memref<{m}x{n}xf32, 1>
  return
}}'''


def main():
    with tempfile.TemporaryDirectory(prefix='shared-pack-') as tmp:
        work = Path(tmp)
        for cols, rows, stride, pair in [(32,7,4,16), (64,11,2,8), (64,16,8,16)]:
            source = fixture(cols, rows, 4, axis=1).replace(
                'frisk.shared_pack_axis = 1 : i64',
                f'frisk.shared_pack_axis = 1 : i64, frisk.shared_store_stride = {stride} : i64, frisk.shared_store_pair = {pair} : i64')
            packed = run([TOOL, '--pack-shared-memory', '--verify-each'], source)
            assert f'frisk.packed_store_stride = {stride}' in packed
            cpu(source, work, f'store-layout-{stride}-before')
            cpu(packed, work, f'store-layout-{stride}-after')
            assert run([TOOL, '--pack-shared-memory'], packed) == packed
            invalid = source.replace(f'frisk.shared_store_pair = {pair}', 'frisk.shared_store_pair = 3')
            assert 'frisk.packed_store_stride' not in run([TOOL, '--pack-shared-memory'], invalid)
            print(f'PASS CPU producer/consumer packing {rows}x{cols}, stride={stride}, pair={pair}')
        for rows, cols, pack in [(12,7,2), (16,11,4)]:
            source = fixture(rows, cols, pack)
            packed = run([TOOL, '--pack-shared-memory', '--verify-each'], source)
            assert 'frisk.packed_width' in packed and 'vector.load' in packed
            assert f'memref<{rows*cols}xf32, 3>' in packed
            assert run([TOOL, '--pack-shared-memory'], packed) == packed
            cpu(source, work, f'base-{pack}')
            cpu(packed, work, f'packed-{pack}')
            print(f'PASS CPU {rows}x{cols}, pack={pack}, every coordinate and multiple consumers')

        for rows, cols, pack in [(12,7,2), (16,11,4), (32,64,4)]:
            source = fixture(rows, cols, pack, axis=1)
            packed = run([TOOL, '--pack-shared-memory', '--verify-each'], source)
            assert 'frisk.packed_axis = 1' in packed and 'vector.load' in packed
            assert run([TOOL, '--pack-shared-memory'], packed) == packed
            cpu(source, work, f'column-base-{rows}-{cols}')
            cpu(packed, work, f'column-packed-{rows}-{cols}')
        for axis in (-1, 2):
            source = fixture(16,11,4,axis=1).replace('shared_pack_axis = 1', f'shared_pack_axis = {axis}')
            assert 'frisk.packed_width' not in run([TOOL, '--pack-shared-memory'], source)
        source = fixture(16,11,4,axis=1)
        for name, text in [
            ('column_tail', source.replace('11x16xf32', '11x15xf32')),
            ('column_pair', source.replace('frisk.shared_pack_axis = 1 : i64',
                'frisk.shared_pack_axis = 1 : i64, frisk.shared_pair_stride = 4 : i64')),
            ('column_view', source.replace('  %z =',
                '  %view = memref.cast %a : memref<11x16xf32, 3> to memref<?x16xf32, 3>\n  %z =')),
            ('column_vector', source.replace('  %z =',
                '  %extra = vector.load %a[%col,%r0] : memref<11x16xf32, 3>, vector<2xf32>\n  %z =')),
        ]:
            out = run([TOOL, '--pack-shared-memory', '--verify-each'], text)
            assert 'frisk.packed_width' not in out, name
        print('PASS CPU column packing: rectangular shapes, all coordinates, reads after writes, invalid axis fallback')

        source = fixture(16,11,4)
        for name, text in [
            ('nondivisible', source.replace('16x11', '15x11')),
            ('unmarked', source.replace('frisk.shared_pack = 4 : i64', '')),
            ('global', source.replace(', 3>', ', 1>')),
            ('escaped', 'func.func private @escape(memref<16x11xf32, 3>)\n' + source.replace(
                '  %z =', '  func.call @escape(%a) : (memref<16x11xf32, 3>) -> ()\n  %z =')),
            ('view', source.replace('  %z =',
                '  %view = memref.cast %a : memref<16x11xf32, 3> to memref<?x11xf32, 3>\n  %z =')),
        ]:
            out = run([TOOL, '--pack-shared-memory', '--verify-each'], text)
            assert 'frisk.packed_width' not in out, name
            print('PASS fallback', name)

        # Two same-lane K fragments, separated by one instruction's K span,
        # occupy one 128-bit packet. All scalar writers/consumers must agree.
        source = fixture(16,11,2).replace('frisk.shared_pack = 2 : i64',
            'frisk.shared_pack = 2 : i64, frisk.shared_pair_stride = 4 : i64')
        packed = run([TOOL, '--pack-shared-memory', '--verify-each'], source)
        assert 'frisk.packed_pair_stride = 4' in packed
        cpu(source, work, 'paired-base')
        cpu(packed, work, 'paired-packed')
        for stride in (0, 1, 3, 32):
            invalid = source.replace('frisk.shared_pair_stride = 4',
                                     f'frisk.shared_pair_stride = {stride}')
            out = run([TOOL, '--pack-shared-memory', '--verify-each'], invalid)
            assert 'frisk.packed_width' not in out, stride
        print('PASS CPU paired K layout, all coordinates and invalid stride fallback')
        for cols in (16, 32):
            source = fixture(16,cols,2).replace('frisk.shared_pack = 2 : i64',
                'frisk.shared_pack = 2 : i64, frisk.shared_pair_stride = 4 : i64')
            packed = run([TOOL, '--pack-shared-memory', '--verify-each'], source)
            cpu(source, work, f'swizzled-base-{cols}')
            cpu(packed, work, f'swizzled-packed-{cols}')
        print('PASS CPU packet column permutation, every coordinate including group boundaries')

        # Exercise scalar/vector LDS packing separately from matrix reads,
        # whose packet contract is covered by check_gemm_matrix_loads.py.
        passes = ['--convert-friskbase-to-thread=matrix-loads=false', '--finalize-thread-tiling',
                  '--pack-shared-memory', '--verify-each']
        for m,n,k,threads in [(32,32,16,64), (64,48,32,128), (32,128,32,64)]:
            source = gemm(m,n,k,threads)
            # The lowering's analysis diagnostics precede the actual module.
            path = work/'gemm.mlir'; path.write_text(source)
            out = work/'gemm.out.mlir'
            run([TOOL, path, *passes, '-o', out])
            text = out.read_text()
            assert 'frisk.packed_width = 4' in text
            assert re.search(r'vector.load .*memref<\d+xf16, 3>, vector<(4|8)xf16>', text)
            store_width = 4 if k >= 32 else 8
            assert re.search(r'vector.store .*memref<\d+xf16, 3>, vector<'+str(store_width)+r'xf16>', text)
            global_widths = [int(w) for w in re.findall(
                r'vector.load[^\n]*memref<[^\n]*, 1>, vector<(\d+)xf16>', text)]
            assert global_widths and all(w <= 8 for w in global_widths), global_widths
            run([TOOL, path, '--convert-friskbase-to-thread=pack-shared-operands=false matrix-loads=false',
                 *passes[1:], '-o', out])
            assert 'frisk.packed_width' not in out.read_text()
            print(f'PASS standalone GEMM {m}x{n}x{k}, {threads} threads, wide reads/writes and disable option')

        # Automatic computed-producer planning, independent of attention.
        # Two matmuls with a numeric cast exercise distinct M/N/K and thread counts.
        for m,k,n,outcols,threads in [(32,32,32,64,64), (64,64,32,32,128),
                                      (32,64,64,32,128), (32,32,16,32,64)]:
            a, b = f'memref<{m}x{k}xf16, 3>', f'memref<{k}x{n}xf16, 3>'
            c, p = f'memref<{m}x{n}xf32>', f'memref<{m}x{n}xf16, 3>'
            rhs = f'memref<{n}x{outcols}xf16, 3>'
            r, dst = f'memref<{m}x{outcols}xf32>', f'memref<{m}x{outcols}xf32, 1>'
            source = f'''func.func @matmul_cast_matmul(%a: {a}, %b: {b}, %rhs: {rhs}, %out: {dst}) attributes {{thread_num = {threads} : i32}} {{
  %c = frisk.gemm(%a, %b) {{transA = false, transB = false}} : {a}, {b} -> {c}
  %value = frisk.cast %c : {c} -> {p}
  %storage = frisk.alloc_buffer {{scope = "shared", alignment = 16}} -> {p}
  frisk.copy %value to %storage [affine_map<() -> (2)>] : {p}, {p}
  %r = frisk.gemm(%storage, %rhs) {{transA = false, transB = false}} : {p}, {rhs} -> {r}
  frisk.copy %r to %out [affine_map<() -> (2)>] : {r}, {dst}
  return
}}'''
            path.write_text(source)
            run([TOOL, path, *passes, '-o', out])
            text = out.read_text()
            # Default contiguous C uses ordinary column packing. Strided
            # producer packing is only needed with the legacy C layout.
            assert 'frisk.packed_axis = 1' in text, text[-4000:]
            assert text.count('memref.alloc()') == 1, 'Packing must not introduce scratch storage'
            assert text.count('gpu.barrier') == 1, 'Preserve the publication barrier'
            # An external observer must retain the original physical layout.
            escaped = f'func.func private @escape({p})\n' + source.replace(
                '  %r =', f'  func.call @escape(%storage) : ({p}) -> ()\n  %r =')
            path.write_text(escaped)
            run([TOOL, path, *passes, '-o', out])
            assert 'frisk.packed_width' not in out.read_text()
            print(f'PASS computed GEMM/cast/GEMM {m}x{k}x{n}x{outcols}, {threads} threads: automatic planning and fallback')

        # Standalone matmul exercises both A and B packing, with global and
        # shared producers, independent of attention or fixed tile dimensions.
        for m,n,k,threads in [(32,32,16,64), (64,48,32,128), (32,128,64,64)]:
            for space in (1, 3):
                a = f'memref<{m}x{k}xf16, 3>'
                src = f'memref<{m}x{k}xf16, {space}>'
                source = gemm(m,n,k,threads).replace(f'%a: {a}', f'%src: {src}')
                source = source.replace('  %b =', f'''  %a = frisk.alloc_buffer {{scope = "shared", alignment = 16}} -> {a}
  frisk.copy %src to %a [affine_map<() -> (2)>] : {src}, {a}
  %b =''', 1)
                path.write_text(source)
                run([TOOL, path, *passes, '-o', out])
                text = out.read_text()
                assert 'frisk.packed_axis = 1' in text, text[-4000:]
                assert 'frisk.packed_axis = 0' in text
                # A's global loads remain 128-bit even though LDS packets are 64-bit.
                if space == 1:
                    assert re.search(r'vector.load[^\n]*, 1>, vector<8xf16>', text)
                run([TOOL, path, '--convert-friskbase-to-thread=pack-shared-operands=false matrix-loads=false',
                     *passes[1:], '-o', out])
                assert 'frisk.packed_axis' not in out.read_text()
                # An unknown consumer must prevent any change to A's storage.
                escaped = f'func.func private @escape({a})\n' + source.replace(
                    '  %b =', f'  func.call @escape(%a) : ({a}) -> ()\n  %b =', 1)
                path.write_text(escaped)
                run([TOOL, path, *passes, '-o', out])
                assert 'frisk.packed_axis = 1' not in out.read_text()
            print(f'PASS standalone GEMM A/B {m}x{n}x{k}: global/shared producers, disable and escape fallback')

        a = 'memref<32x32xf16, 3>'
        c = 'memref<32x32xf32>'
        source = gemm(32,32,32,64).replace(f'%a: {a}', f'%src: {a}')
        source = source.replace('  %b =', f'''  %a = frisk.alloc_buffer {{scope = "shared", alignment = 16}} -> {a}
  frisk.copy %src to %a [affine_map<() -> (2)>] : {a}, {a}
  %b =''', 1)
        for name, operands in [('same', '%a, %b'), ('conflicting', '%b, %a')]:
            two = source.replace('  frisk.copy %c to', f'''  %c2 = frisk.gemm({operands}) {{transA = false, transB = false}} : {a}, {a} -> {c}
  %sum = frisk.add %c, %c2 : {c}, {c} -> {c}
  frisk.copy %sum to''')
            path.write_text(two)
            run([TOOL, path, *passes, '-o', out])
            assert out.read_text().count('frisk.packed_axis = 1') == (1 if name == 'same' else 0)
        print('PASS A with multiple consumers: compatible layouts share packing; conflicting layouts fall back')

        a, b, c = 'memref<32x16xf16, 3>', 'memref<16x32xf16, 3>', 'memref<32x32xf32>'
        two = gemm(32,32,16,64, f'%c2 = frisk.gemm(%a, %b) {{transA = false, transB = false}} : {a}, {b} -> {c}')
        two = two.replace('  frisk.copy %c to',
                          f'  %sum = frisk.add %c, %c2 : {c}, {c} -> {c}\n  frisk.copy %sum to')
        path.write_text(two)
        run([TOOL, path, *passes, '-o', out])
        assert out.read_text().count('frisk.packed_width = 4') == 1
        escaped = f'func.func private @escape({b})\n' + gemm(
            32,32,16,64, f'func.call @escape(%b) : ({b}) -> ()')
        path.write_text(escaped)
        run([TOOL, path, *passes, '-o', out])
        assert 'frisk.packed_width' not in out.read_text()
        print('PASS shared B with two GEMM consumers; unknown consumer disables automatic packing')


if __name__ == '__main__':
    main()
