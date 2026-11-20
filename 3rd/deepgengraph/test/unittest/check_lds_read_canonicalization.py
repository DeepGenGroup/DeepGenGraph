#!/usr/bin/env python3
"""Role-independent LDS pointers, conservative exclusions and dominance checks."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT/'3rd/deepgengraph/build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')


def transform(source):
    result = subprocess.run([str(TOOL), '--canonicalize-lds-geps-only', '-'],
        input=source, text=True, capture_output=True, check=True).stdout
    subprocess.run([str(LLVM/'opt'), '-passes=default<O3>,verify', '-disable-output', '-'],
        input=result, text=True, capture_output=True, check=True)
    return result


def alignment(text, name):
    return int(re.search(r'%' + name + r' = load .*?, align (\d+)', text)[1])


def main():
    source = '''target datalayout = "e-p:64:64-p3:32:32"
define float @arbitrary_buffers(ptr addrspace(3) align 16 %ring,
    ptr addrspace(3) align 32 %scratch, ptr addrspace(3) %unknown,
    ptr addrspace(1) %global, i32 %index) {
  %offset = shl i32 %index, 4
  %p = getelementptr i8, ptr addrspace(3) %ring, i32 %offset
  %q = getelementptr i8, ptr addrspace(3) %p, i32 8
  %r = getelementptr i8, ptr addrspace(3) %scratch, i32 %offset
  %s = getelementptr i8, ptr addrspace(3) %r, i32 -8
  %a = load float, ptr addrspace(3) %p, align 1
  %b = load <2 x float>, ptr addrspace(3) %q, align 1
  %c = load <4 x half>, ptr addrspace(3) %r, align 1
  %d = load i16, ptr addrspace(3) %s, align 1
  %v = load volatile float, ptr addrspace(3) %q, align 1
  %atom = load atomic i32, ptr addrspace(3) %p monotonic, align 4
  %g = load float, ptr addrspace(1) %global, align 1
  %u = load float, ptr addrspace(3) %unknown, align 1
  call void asm sideeffect "", "v,v,v,v,v,v,v,v"(float %a, <2 x float> %b,
       <4 x half> %c, i16 %d, float %v, i32 %atom, float %g, float %u)
  ret float %a
}
'''
    result = transform(source)
    assert [alignment(result, x) for x in 'abcd'] == [16, 8, 16, 8], result
    assert [alignment(result, x) for x in ('v','atom','g','u')] == [1,4,1,1]
    assert result.count('load ') == 8
    assert 'ptr addrspace(3) %scratch' in result and 'ptr addrspace(3) %unknown' in result
    assert transform(result) == result, 'pass must be idempotent'
    print('PASS: arbitrary pointer arguments, dynamic offsets, negative offsets and mixed load types')
    print('PASS: volatile/atomic/global/unknown alignment unchanged; idempotence; LLVM verifier')
    for size, shift in ((32,2), (64,3), (128,4)):
        source = f'''target datalayout = "e-p:64:64-p3:32:32"
@workspace = addrspace(3) global [8192 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
define float @nonlinear() {{
  %t = call i32 @llvm.amdgcn.workitem.id.x(), !range !0
  %a = xor i32 %t, {size}
  %b = add i32 %t, {size}
  %c = xor i32 %b, {size*2}
  %i = shl i32 %a, {shift}
  %j = shl i32 %c, {shift}
  %p = getelementptr i8, ptr addrspace(3) @workspace, i32 %i
  %q = getelementptr i8, ptr addrspace(3) @workspace, i32 %j
  %x = load float, ptr addrspace(3) %p, align 1
  %y = load float, ptr addrspace(3) %q, align 1
  %sum = fadd float %x, %y
  ret float %sum
}}
!0 = !{{i32 0, i32 {size}}}
'''
        result = transform(source)
        # (t + size) XOR (2*size) minus (t XOR size) = 2*size.
        assert f'getelementptr i8, ptr addrspace(3) %p, i32 {2*size*(1<<shift)}' in result
        assert alignment(result,'x') == alignment(result,'y') == 1 << shift
        unknown = transform(source.replace(', !range !0',''))
        assert 'lds.const.gep' not in unknown
    print('PASS: nonlinear layouts at 32/64/128 threads; unknown domain does not prove false offsets')
    source = '''target datalayout = "e-p:64:64-p3:32:32"
define float @branches(ptr addrspace(3) %base, i32 %i, i1 %flag) {
entry:
  br i1 %flag, label %left, label %right
left:
  %p = getelementptr float, ptr addrspace(3) %base, i32 %i
  %x = load float, ptr addrspace(3) %p, align 4
  ret float %x
right:
  %j = add i32 %i, 16
  %q = getelementptr float, ptr addrspace(3) %base, i32 %j
  %y = load float, ptr addrspace(3) %q, align 4
  ret float %y
}
'''
    assert 'lds.const.gep' not in transform(source)
    print('PASS: an address in a sibling block is not reused without dominance')


if __name__ == '__main__':
    main()
