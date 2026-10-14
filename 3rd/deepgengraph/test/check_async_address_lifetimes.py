#!/usr/bin/env python3
"""DMA address rematerialization must preserve convergent calls and ordering."""
import re
import subprocess
import sys
from pathlib import Path
from check_register_lifetimes import normalize_metadata_ids

tool = Path(sys.argv[1]).resolve()
llvm = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')
source = '''target triple = "amdgcn-amd-amdhsa"
@lds = addrspace(3) global [8192 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.readfirstlane.i32(i32)
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1), ptr addrspace(3), i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg)
declare void @llvm.amdgcn.s.barrier()
declare void @opaque()
define amdgpu_kernel void @copy(ptr addrspace(1) %src, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %wide = zext i32 %tid to i64
  %offset = mul i64 %wide, 4
  %global = getelementptr i8, ptr addrspace(1) %src, i64 %offset
  %wave = and i32 %tid, -64
  %bytes = mul i32 %wave, 4
  %local = getelementptr i8, ptr addrspace(3) @lds, i32 %bytes
  %address = ptrtoint ptr addrspace(3) %local to i32
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @opaque()
  %uniform = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address)
  %dst = inttoptr i32 %uniform to ptr addrspace(3)
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %global, ptr addrspace(3) %dst, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  call void @llvm.amdgcn.s.barrier()
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %n
  br i1 %more, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
'''


def run(args, text):
    p = subprocess.run(list(map(str, args)), input=text, text=True,
                       capture_output=True, check=True, timeout=120)
    return p.stdout


output = run([tool, '--optimize-register-lifetimes', '-'], source)
assert normalize_metadata_ids(run([tool, '--optimize-register-lifetimes', '-'], output)) == normalize_metadata_ids(output)
for ir in (output, run([llvm/'opt', '-passes=default<O3>', '-S', '-o', '-'], output)):
    run([llvm/'opt', '-passes=verify', '-disable-output'], ir)
    entry, loop = ir.split('loop:', 1)
    assert not re.search(r' = (getelementptr|ptrtoint)', entry)
    calls = [m[1] for m in re.finditer(r'call [^\n@]+ @llvm.amdgcn.([\w.]+)\(', loop)]
    calls = ['readfirstlane' if c.startswith('readfirstlane.') else c for c in calls]
    assert calls == ['readfirstlane', 'global.load.async.lds', 'asyncmark',
                     'wait.asyncmark', 's.barrier'], calls
    assert loop.count('address.thread') > 1
print('PASS DMA source and wave-base addresses stay local through O3; convergent calls/order and idempotence preserved')
