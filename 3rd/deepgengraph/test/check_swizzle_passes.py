#!/usr/bin/env python3
"""CPU proofs for packet swizzles, grouped workgroups and constant LDS GEPs."""
import re
import subprocess
import tempfile
from pathlib import Path
import check_shared_packing as packing
import check_shared_buffer_aliases as alias_tests

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / 'build/test/ThreadTilingTest'
EXPORT = HERE.parent / 'build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')

def run(args, source=None, ok=True):
    p = subprocess.run(list(map(str, args)), input=source, text=True,
                       capture_output=True, timeout=180)
    assert (p.returncode == 0) == ok, p.stderr[-6000:]
    return p.stdout if ok else p.stderr

def order_fixture(q, h, group, reverse, qaxis):
    sx, sy = (q, h) if qaxis == 'x' else (h, q)
    query, head = ('%x', '%y') if qaxis == 'x' else ('%y', '%x')
    source = f'''func.func @map(%px: index, %py: index) -> index {{
      %x = gpu.block_id x {{range = {sx} : index}}
      %y = gpu.block_id y {{range = {sy} : index}}
      %heads = arith.constant {h} : index
      %q = arith.muli {query}, %heads : index
      %r = arith.addi %q, {head} : index
      return %r : index
    }}'''
    option = f'group-size={group} reverse={str(reverse).lower()} query-axis={qaxis} head-axis={"y" if qaxis == "x" else "x"}'
    result = run([TOOL, f'--frisk-reorder-workgroups={option}', '--verify-each'], source)
    twice = run([TOOL, f'--frisk-reorder-workgroups={option}'], result)
    assert twice == result
    if group:
        run([TOOL, f'--frisk-reorder-workgroups={option} group-size={group+1}'], result, ok=False)
    for name, axis in re.findall(r'(%\w+) = gpu.block_id ([xy])[^\n]*', result):
        result = re.sub(r'^.*' + re.escape(name) + r' = gpu.block_id[^\n]*\n', '', result, flags=re.M)
        result = re.sub(re.escape(name) + r'\b', ('%arg0' if axis == 'x' else '%arg1'), result)
    main = ['func.func @main() -> i32 {', '  %good = arith.constant true']
    seen = set()
    previous = '%good'
    for y in range(sy):
        for x in range(sx):
            t = y * sx + x
            if group:
                g = min(group, q)
                start = t // (g * h) * g
                width = min(g, q - start)
                within = t % (g * h)
                logical_q, logical_h = start + within % width, within // width
                if reverse: logical_q = q - 1 - logical_q
            else:
                logical_q, logical_h = (x, y) if qaxis == 'x' else (y, x)
            assert (logical_q, logical_h) not in seen
            seen.add((logical_q, logical_h))
            main += [f'  %x{t} = arith.constant {x} : index',
                     f'  %y{t} = arith.constant {y} : index',
                     f'  %want{t} = arith.constant {logical_q*h+logical_h} : index',
                     f'  %got{t} = func.call @map(%x{t}, %y{t}) : (index,index) -> index',
                     f'  %eq{t} = arith.cmpi eq, %got{t}, %want{t} : index',
                     f'  %ok{t} = arith.andi {previous}, %eq{t} : i1']
            previous = f'%ok{t}'
    main += ['  %zero = arith.constant 0 : i32', '  %one = arith.constant 1 : i32',
             f'  %result = arith.select {previous}, %zero, %one : i32', '  return %result : i32', '}']
    return result.rsplit('}', 1)[0] + '\n'.join(main) + '\n}\n'

def main():
    with tempfile.TemporaryDirectory(prefix='swizzle-passes-') as directory:
        directory = Path(directory)
        for rows, cols, axis in [(32,64,0), (48,32,1), (16,128,0)]:
            source = packing.fixture(rows, cols, 4, axis).replace(
                'frisk.shared_pack = 4 : i64',
                'frisk.shared_swizzle_src = 8 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_pack = 4 : i64')
            result = run([TOOL, '--pack-shared-memory', '--verify-each'], source)
            assert 'vector.load' in result, result[:1800]
            packing.cpu(result, directory, f'swizzle-{rows}-{cols}-{axis}')
            assert run([TOOL, '--pack-shared-memory'], result) == result
            run([TOOL, '--pack-shared-memory'], source.replace('swizzle_dst = 6', 'swizzle_dst = 3'), ok=False)
        print('PASS CPU shared writes/reads, both axes, unequal shapes, vector coalescing, idempotence and invalid swizzle')
        for axis in (0,1):
            source = alias_tests.fixture(16,32,axis).replace(
                'frisk.shared_pack = 2 : i64',
                'frisk.shared_swizzle_src = 8 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_pack = 2 : i64')
            result = run([TOOL, '--pack-shared-memory', '--verify-each'], source)
            packing.cpu(result, directory, f'swizzle-alias-{axis}')
            conflict = source.replace('shared_swizzle_dst = 6', 'shared_swizzle_dst = 5', 1)
            run([TOOL, '--pack-shared-memory'], conflict, ok=False)
        print('PASS CPU swizzle through selected/loop-carried aliases; conflicting contracts rejected')
        minimal = packing.gemm(32,32,16,128)
        for option in ('swizzle-k=7:3 swizzle-k-arg=0', 'swizzle-k=on',
                       'swizzle-k=on swizzle-k-arg=0'):
            run([TOOL, '--convert-friskbase-to-thread=' + option], minimal, ok=False)
        print('PASS invalid subpacket swizzle, missing role binding, and unmatched producer rejected')

        for q, h, g in [(13,3,8), (5,2,8), (16,3,8), (1,7,1), (9,1,4), (7,3,0)]:
            for reverse in (False, True):
                for axis in ('x','y'):
                    result = order_fixture(q,h,g,reverse,axis)
                    packing.cpu(result, directory, f'order-{q}-{h}-{g}-{reverse}-{axis}')
        print('PASS CPU all workgroups: tail, oversized group, uneven axes, reverse, identity, idempotence')
        fixture = '''target datalayout = "e-p:64:64-p3:32:32"
define float @reads(ptr addrspace(3) %base, i32 %i, ptr addrspace(3) %other) {
  %a = xor i32 %i, 64
  %p = getelementptr float, ptr addrspace(3) %base, i32 %a
  %b = add i32 %a, 32
  %q = getelementptr float, ptr addrspace(3) %base, i32 %b
  %x = load float, ptr addrspace(3) %p, align 4
  %y = load float, ptr addrspace(3) %q, align 4
  %v = load volatile float, ptr addrspace(3) %q, align 4
  %z = load float, ptr addrspace(3) %other, align 4
  %s = fadd float %x, %y
  %t = fadd float %s, %v
  %r = fadd float %t, %z
  ret float %r
}
'''
        source = directory/'geps.ll'; source.write_text(fixture)
        result = run([EXPORT, '--canonicalize-lds-geps', source])
        assert 'getelementptr i8, ptr addrspace(3) %p, i32 128' in result, result
        assert 'load volatile float, ptr addrspace(3) %q' in result
        assert 'load float, ptr addrspace(3) %other' in result
        source.write_text(result)
        run([LLVM/'opt', '-passes=verify', '-disable-output', source])
        print('PASS LLVM constant-difference proof, volatile/unrelated exclusions and verifier')
        nonlinear = '''target datalayout = "e-p:64:64-p3:32:32"
@sm = addrspace(3) global [4096 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @nonlinear(ptr addrspace(1) %out) {
  %t = call i32 @llvm.amdgcn.workitem.id.x(), !range !0
  %a = xor i32 %t, 64
  %i = shl i32 %a, 3
  %p = getelementptr i8, ptr addrspace(3) @sm, i32 %i
  %b = add i32 %t, 64
  %c = xor i32 %b, 128
  %j = shl i32 %c, 3
  %q = getelementptr i8, ptr addrspace(3) @sm, i32 %j
  %x = load float, ptr addrspace(3) %p, align 2
  %y = load float, ptr addrspace(3) %q, align 2
  %s = fadd float %x, %y
  store float %s, ptr addrspace(1) %out
  ret void
}
!0 = !{i32 0, i32 64}
'''
        source.write_text(nonlinear)
        result = run([EXPORT, '--canonicalize-lds-geps', source])
        assert 'getelementptr i8, ptr addrspace(3) %p, i32 1024' in result
        assert result.count(', align 8') == 2
        source.write_text(nonlinear.replace(', !range !0', ''))
        unknown = run([EXPORT, '--canonicalize-lds-geps', source])
        assert 'lds.const.gep' not in unknown
        print('PASS exact nonlinear address proof and alignment; wider thread domain with counterexamples left unchanged')


if __name__ == '__main__': main()
