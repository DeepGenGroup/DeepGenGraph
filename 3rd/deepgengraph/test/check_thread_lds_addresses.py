#!/usr/bin/env python3
"""Thread LDS proofs, replay, actual LLVM lowering and CPU address equivalence."""
import ctypes
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'build/test/ThreadTilingTest'
LLVM = Path('/data2/xsl/install/bin')
PASS = '--frisk-canonicalize-lds-addresses'


def run(args, source=None, ok=True):
    p = subprocess.run(list(map(str, args)), input=source, text=True,
                       capture_output=True, timeout=180)
    assert (p.returncode == 0) == ok, p.stderr
    return p.stdout


def transform(source):
    return run([TOOL, '--verify-each', PASS], source)


def fixture(count=64, size=64):
    return f'''module {{
      func.func @reads(%out: memref<2xf32, 1>) attributes {{thread_num = {count} : i32}} {{
        %s = memref.alloc() alignment = 16 : memref<16384xf32, 3>
        %t = gpu.thread_id x
        %c = arith.constant {size} : index
        %d = arith.constant {size * 2} : index
        %two = arith.constant 2 : index
        %zero = arith.constant 0 : index
        %a = arith.xori %t, %c : index
        %b = arith.addi %t, %c : index
        %c1 = arith.xori %b, %d : index
        %i = arith.muli %a, %two : index
        %j = arith.muli %c1, %two : index
        %x = vector.load %s[%i] : memref<16384xf32, 3>, vector<2xf32>
        %y = vector.load %s[%j] : memref<16384xf32, 3>, vector<2xf32>
        %sum = arith.addf %x, %y : vector<2xf32>
        vector.store %sum, %out[%zero] : memref<2xf32, 1>, vector<2xf32>
        return
      }}
    }}'''


def export(source):
    low = run([TOOL, '--verify-each', '--test-async-to-llvm'], source)
    assert 'frisk.lds_load' not in low
    ll = run([LLVM / 'mlir-translate', '--mlir-to-llvmir'], low)
    run([LLVM / 'opt', '-passes=verify', '-disable-output'], ll)
    return ll


for count in (1, 32, 64, 192, 256, 512, 1024):
    size = 1 << (count - 1).bit_length()
    result = transform(fixture(count, size))
    assert result.count('frisk.lds_load') == 2, result
    assert f'byte_offset = {size * 16} : i64' in result, result
    assert result.count('alignment = 8 : i64') == 2 or count == 1, result
    assert transform(result) == result, 'pass must be idempotent after serialization'
    ll = export(result)
    assert re.search(r'getelementptr i8, ptr addrspace\(3\) %\w+, i(?:32|64) ' + str(size * 16), ll), ll
    assert len(re.findall(r'load <2 x float>.*align (?:8|16)', ll)) == 2, ll
print('PASS: queried launch ranges 1..1024, nonlinear differences, alignment, replay and LLVM GEPs')

for attr in ('', 'thread_num = 0 : i32', 'thread_num = -1 : i32',
             'thread_num = 1025 : i32', 'thread_num = "64"'):
    source = re.sub(r'thread_num = 64 : i32', attr, fixture())
    unknown = transform(source)
    assert 'byte_offset = 1024' not in unknown, (attr, unknown)
# A wider valid domain contains a counterexample: alignment may still improve,
# but the two reads must retain separate anchors.
wider = transform(fixture(65, 64))
assert 'byte_offset = 1024' not in wider, wider
print('PASS: absent/malformed launch contracts and full-domain counterexamples')

symbolic = '''module {
  func.func @symbolic(%s: memref<8192xf32, 3>, %i: index) -> (f32, f32) {
    %x = affine.load %s[%i + 16] : memref<8192xf32, 3>
    %y = affine.load %s[%i] : memref<8192xf32, 3>
    return %x, %y : f32, f32
  }
}'''
result = transform(symbolic)
assert 'byte_offset = -64 : i64' in result, result
assert transform(result) == result
export(result)
unrelated = symbolic.replace('%i: index', '%other: memref<8192xf32, 3>, %i: index')
unrelated = unrelated.replace('%y = affine.load %s', '%y = affine.load %other')
assert 'frisk.lds_load' not in transform(unrelated)
branches = '''module {
  func.func @branches(%s: memref<8192xf32, 3>, %i: index, %cond: i1) -> f32 {
    %r = scf.if %cond -> f32 {
      %a = affine.load %s[%i] : memref<8192xf32, 3>
      scf.yield %a : f32
    } else {
      %b = affine.load %s[%i + 16] : memref<8192xf32, 3>
      scf.yield %b : f32
    }
    return %r : f32
  }
}'''
assert 'frisk.lds_load' not in transform(branches)
assert 'frisk.lds_load' not in transform(symbolic.replace(', 3>', ', 1>'))
assert 'frisk.lds_load' not in transform(symbolic.replace(' : memref<8192xf32, 3>\n',
                                                       ' {volatile} : memref<8192xf32, 3>\n'))
print('PASS: symbolic/negative differences, sibling dominance, unrelated/global/unsupported reads')

nt = transform(fixture().replace('%y = vector.load %s[%j]',
                                '%y = vector.load %s[%j] nontemporal = true'))
assert 'nontemporal = true' in nt
assert '!nontemporal' in export(nt)
run([TOOL, '--verify-each'], nt.replace('alignment = 8 : i64', 'alignment = 3 : i64'), ok=False)
run([TOOL, '--verify-each'], nt.replace('memref<16384xf32, 3>', 'memref<16384xf32, 1>'), ok=False)
print('PASS: nontemporal propagation and canonical-load verifier rejects invalid contracts')

# Pool byte shifts must weaken alignment even when allocation alignment is high.
pool = fixture().replace('%s = memref.alloc() alignment = 16 : memref<16384xf32, 3>', '''
    %pool = memref.alloc() alignment = 64 : memref<65540xi8, 3>
    %shift = arith.constant 4 : index
    %s = memref.view %pool[%shift][] : memref<65540xi8, 3> to memref<16384xf32, 3>''')
result = transform(pool)
assert result.count('alignment = 4 : i64') == 2, result
export(result)
print('PASS: physical pool/view shift included in alignment proof')

global_pool = pool.replace('module {', '''module {
  memref.global "private" @shared : memref<65540xi8, 3> = uninitialized alignment = 64''', 1)
global_pool = global_pool.replace('%pool = memref.alloc() alignment = 64 : memref<65540xi8, 3>',
                                  '%pool = memref.get_global @shared : memref<65540xi8, 3>')
result = transform(global_pool)
assert result.count('alignment = 4 : i64') == 2, result
export(result)
print('PASS: global-backed shared pool follows the same alignment contract')

# Exercise real lowering on CPU with thread IDs supplied by an ordinary argument.
# Strip only the GPU address-space spelling after LLVM conversion. The load/GEP
# instructions, their offsets and alignment are exactly those emitted above.
cpu = '''module {
  func.func @probe(%s: memref<4096xf32, 3>, %thread: i32, %bias: index, %v: f32) -> f32
      attributes {thread_num = 64 : i32} {
    %t = gpu.thread_id x
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %a = arith.xori %t, %c64 : index
    %b = arith.addi %t, %c64 : index
    %c = arith.xori %b, %c128 : index
    %i = arith.addi %a, %bias : index
    %j = arith.addi %c, %bias : index
    %x = memref.load %s[%i] : memref<4096xf32, 3>
    %y = memref.load %s[%j] : memref<4096xf32, 3>
    memref.store %v, %s[%i] : memref<4096xf32, 3>
    %z = memref.load %s[%i] : memref<4096xf32, 3>
    %sum = arith.addf %x, %y : f32
    %r = arith.addf %sum, %z : f32
    return %r : f32
  }
}'''
optimized = transform(cpu)
assert 'byte_offset = 512 : i64' in optimized, optimized
with tempfile.TemporaryDirectory(prefix='thread-lds-') as directory:
    directory = Path(directory)
    functions = []
    for name, source in (('baseline', cpu), ('optimized', optimized)):
        source = re.sub(r'attributes\s*\{thread_num = 64 : i32\}', '', source)
        # SSA names are renamed by the printer; the thread argument is the second argument.
        thread = re.search(r'(%\w+): i32', source)[1]
        source = source.replace('gpu.thread_id x', f'arith.index_cast {thread} : i32 to index')
        ll = export(source).replace('addrspace(3)', '')
        obj, lib = directory / (name + '.o'), directory / (name + '.so')
        run([LLVM / 'llc', '-filetype=obj', '-relocation-model=pic', '-o', obj], ll)
        run(['cc', '-shared', obj, '-o', lib])
        library = ctypes.CDLL(str(lib))
        fn = library.probe
        fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int64, ctypes.c_float]
        fn.restype = ctypes.c_float
        functions.append((library, fn))
    for tid in range(64):
        for bias in (0, 7, 256):
            outputs = []
            for _, fn in functions:
                data = (ctypes.c_float * 4096)(*[float(i) for i in range(4096)])
                outputs.append((fn(data, tid, bias, -7.0), list(data)))
            assert outputs[0] == outputs[1], (tid, bias)
            expected = float((tid ^ 64) + bias + ((tid + 64) ^ 128) + bias - 7)
            assert outputs[1][0] == expected, (tid, bias, outputs[1][0], expected)
print('PASS: 192 CPU baseline/candidate cases, dynamic common origin and intervening writes')

# Target code generation is independent of the old LLVM canonicalizer.
ll = export(transform(fixture()))
o3 = run([LLVM / 'opt', '-S', '-passes=default<O3>'], ll)
assembly = run([LLVM / 'llc', '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx90a', '-O3', '-o', '-'], o3)
assert re.search(r'ds_read2(?:st64)?_b64', assembly), assembly
lifetimes = run([ROOT / 'build/test/LegacyLLVMExportTest', '--optimize-register-lifetimes', '-'], ll)
o3 = run([LLVM / 'opt', '-S', '-passes=default<O3>,verify'], lifetimes)
assembly = run([LLVM / 'llc', '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx90a', '-O3', '-o', '-'], o3)
assert re.search(r'ds_read2(?:st64)?_b64', assembly), assembly
print('PASS: paired b64 reads survive lifetime optimization and AMDGPU codegen without LLVM address canonicalization')
