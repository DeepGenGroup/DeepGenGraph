#!/usr/bin/env python3
"""Legality, downstream O3 survival and legacy codegen for lifetime scheduling."""
import ctypes.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from check_legacy_llvm_export import load_parser


def run(args, text=None):
    result = subprocess.run(list(map(str, args)), input=text, text=True,
                            capture_output=True, timeout=180)
    assert result.returncode == 0, result.stderr
    return result.stdout


def body(text, name):
    return re.search(r'define [^@]*@' + name + r'\(.*?\n\}', text, re.S)[0]


def normalize_metadata_ids(text):
    # Metadata slot numbers may change when LLVM parses and prints the IR.
    slots = {slot: str(index) for index, slot in enumerate(
        re.findall(r'^!(\d+) =', text, re.M))}
    return re.sub(r'!(\d+)\b', lambda match: '!' + slots[match[1]], text)


def main():
    tool = Path(sys.argv[1]).resolve()
    llvm = Path('/data2/xsl/install/bin')
    source = '''target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
declare float @llvm.exp2.f32(float)
declare void @opaque()
define amdgpu_kernel void @address(ptr addrspace(1) %out, ptr addrspace(1) %indices, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idx = mul nuw i32 %tid, 7
  %wide = zext i32 %idx to i64
  %p = getelementptr float, ptr addrspace(1) %out, i64 %wide
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @opaque()
  %loaded = load volatile i32, ptr addrspace(1) %indices
  %q = getelementptr float, ptr addrspace(1) %p, i32 %loaded
  store volatile float 1.0, ptr addrspace(1) %q
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %n
  br i1 %more, label %loop, label %exit
exit:
  ret void
}
define amdgpu_kernel void @uniform(ptr addrspace(1) %out, i64 %i) {
  %p = getelementptr float, ptr addrspace(1) %out, i64 %i
  store float 1.0, ptr addrspace(1) %p
  ret void
}
define half @scalar(float %x) {
  %e = call float @llvm.exp2.f32(float %x)
  call void @opaque()
  %h = fptrunc float %e to half
  ret half %h
}
declare i32 @llvm.amdgcn.workitem.id.y()
declare i32 @llvm.amdgcn.workitem.id.z()
define amdgpu_kernel void @axes(ptr addrspace(1) %out, i1 %cond) "amdgpu-flat-work-group-size"="256,256" !reqd_work_group_size !0 {
entry:
  %y = call i32 @llvm.amdgcn.workitem.id.y()
  %z = call i32 @llvm.amdgcn.workitem.id.z()
  %stride = mul i32 %z, 8
  %index = add i32 %stride, %y
  %p = getelementptr float, ptr addrspace(1) %out, i32 %index
  br i1 %cond, label %write, label %exit
write:
  store float 2.0, ptr addrspace(1) %p
  br label %exit
exit:
  ret void
}
!0 = !{i32 32, i32 4, i32 2}
'''
    # Different vector widths and unrelated intervening work. The sum's
    # association must survive exactly, and casts should precede opaque().
    for width in (2, 4):
        source += f'''define <{width} x half> @math{width}(float %x, ptr %sum) {{
  %e = call float @llvm.exp2.f32(float %x)
  %v = insertelement <{width} x float> poison, float %e, i32 0
  %s = fadd float %e, 1.0
  %t = fadd float %s, 2.0
  call void @opaque()
  %h = fptrunc <{width} x float> %v to <{width} x half>
  store float %t, ptr %sum
  ret <{width} x half> %h
}}
'''
    source += '''define <2 x half> @strict(float %x) strictfp {
  %e = call float @llvm.exp2.f32(float %x)
  %v = insertelement <2 x float> poison, float %e, i32 0
  %h = fptrunc <2 x float> %v to <2 x half>
  ret <2 x half> %h
}
declare float @unknown(float)
define <2 x half> @unknown_math(float %x) {
  %e = call float @unknown(float %x)
  %v = insertelement <2 x float> poison, float %e, i32 0
  %h = fptrunc <2 x float> %v to <2 x half>
  ret <2 x half> %h
}
'''
    output = run([tool, '--optimize-register-lifetimes', '-'], source)
    run([llvm / 'opt', '-passes=verify', '-disable-output', '-'], output)
    repeated = run([tool, '--optimize-register-lifetimes', '-'], output)
    assert normalize_metadata_ids(repeated) == normalize_metadata_ids(output)
    address = body(output, 'address')
    assert 'address.local' not in address.split('loop:')[0]
    assert address.count('load volatile') == 1 and address.count('store volatile') == 1
    assert address.index('call void @opaque') < address.index('load volatile') < address.index('address.thread')
    assert '%loaded' in address and '%next = add i32 %i, 1' in address
    for name in ('uniform', 'strict', 'unknown_math'):
        assert 'asm sideeffect' not in body(output, name), name
    scalar = body(output, 'scalar')
    assert scalar.count('asm sideeffect "", "=v,0"') == 2
    assert scalar.index('fptrunc') < scalar.index('call void @opaque')
    assert body(output, 'axes').count('!range') == 4  # intrinsic and local identity
    assert '!{i32 0, i32 4}' in output and '!{i32 0, i32 2}' in output
    for width in (2, 4):
        section = body(output, f'math{width}')
        assert section.count('asm sideeffect "", "=v,0"') == 2
        assert section.index('fptrunc') < section.index('call void @opaque')
        assert '%s = fadd float %e, 1.000000e+00' in section
        assert '%t = fadd float %s, 2.000000e+00' in section
    cpu = run([tool, '--optimize-register-lifetimes', '-'],
              source.replace('amdgcn-amd-amdhsa', 'x86_64-unknown-linux-gnu'))
    assert 'asm sideeffect' not in cpu
    optimized = run([llvm / 'opt', '-S', '-passes=default<O3>', '-'], output)
    loop = body(optimized, 'address').split('loop:')[1].split('exit:')[0]
    assert 'asm sideeffect "", "=v,0"' in loop and 'getelementptr' in loop
    print('PASS: local addresses survive O3; volatile loads/PHIs/control flow preserved; uniform, strictfp and unknown-call fallbacks; idempotence')

    legacy = run([tool, '--legalize-text', '-'], output)
    failed, error = load_parser(ctypes.util.find_library('LLVM-15'))(legacy)
    assert not failed, error
    # Real backend acceptance of the tied register constraints at O3. External
    # opaque calls prevent deletion; this is gfx90a legality, not gfx936 timing.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'test.ll'
        path.write_text(legacy)
        assembly = Path(directory) / 'test.s'
        run([sys.executable, Path(__file__).with_name('check_legacy_llvm_export.py'),
             '--codegen', path, assembly, ctypes.util.find_library('LLVM-15'), 3])
        assert 'v_exp_f32' in assembly.read_text()
    print('PASS: exact FP operand trees and early packing at multiple widths; LLVM 15 parsing and gfx90a O3 codegen')


if __name__ == '__main__':
    main()
