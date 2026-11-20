#!/usr/bin/env python3
"""Collective matrix reads retain order while their addresses move into loops."""
import sys
from pathlib import Path
from check_register_lifetimes import run, body


def main():
    tool = Path(sys.argv[1]).resolve()
    llvm = Path('/data2/xsl/install/bin')
    matrix = '''target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) convergent
define <4 x i32> @matrix(ptr addrspace(3) %base, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %offset = shl i32 %tid, 4
  %ptr = getelementptr i8, ptr addrspace(3) %base, i32 %offset
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  call void @llvm.amdgcn.s.barrier()
  %read = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %ptr, i32 0)
  %next = add i32 %i, 1
  %again = icmp ult i32 %next, %n
  br i1 %again, label %loop, label %exit
exit:
  ret <4 x i32> %read
}
'''
    matrix = run([tool, '--optimize-register-lifetimes', '-'], matrix)
    run([llvm / 'opt', '-passes=verify', '-disable-output', '-'], matrix)
    section = body(matrix, 'matrix').split('loop:')[1]
    assert section.count('call <4 x i32> @llvm.hcu.ds.read') == 1
    assert section.count('call void @llvm.amdgcn.s.barrier') == 1
    assert section.index('s.barrier') < section.index('address.thread') < section.index('ds.read')
    assert 'ptr addrspace(3) %address.local' in section
    print('PASS: collective matrix-read addresses rematerialize locally without duplicating or moving the read')
    optimized = run([llvm / 'opt', '-S', '-passes=default<O3>', '-'], matrix)
    section = body(optimized, 'matrix').split('loop:')[1]
    assert 'asm sideeffect "", "=v,0"' in section and 'getelementptr' in section
    assert section.count('call <4 x i32> @llvm.hcu.ds.read') == 1
    print('PASS: local matrix addresses survive downstream O3')


if __name__ == '__main__':
    main()
