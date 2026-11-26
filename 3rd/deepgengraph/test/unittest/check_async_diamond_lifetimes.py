#!/usr/bin/env python3
"""Reuse SSA addresses across planned DMA branches, never memory contents."""
import re
import sys
from pathlib import Path

from check_register_lifetimes import run, body, normalize_metadata_ids

LLVM = Path('/data2/xsl/install/bin')
COPY = 'call void @__frisk_async_load_lds(ptr addrspace(1) %src, ptr addrspace(3) %dst, i32 8, i32 0, i32 0)'


def source(effect=COPY, marker=True, reverse=False, forwarding=False,
           extra_pred=False, different=False):
    entry = '''entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %offset = shl i32 %tid, 4
  %ptr = getelementptr i8, ptr addrspace(3) %base, i32 %offset
  %other = getelementptr i8, ptr addrspace(3) %ptr, i32 256
  br label %head
'''
    head = '''head:
  %i = phi i32 [0, %entry], [%next, %join]
  call void @llvm.amdgcn.s.barrier()
  %first = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %ptr, i32 0)
  br i1 %take, label %issue, label %join
'''
    issue = 'issue:\n  ' + effect + '\n  call void @llvm.amdgcn.asyncmark()\n  br label %join\n'
    join = '''join:
  %second = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %ptr, i32 0)
  %sum = add <4 x i32> %first, %second
  store volatile <4 x i32> %sum, ptr addrspace(1) %src
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %n
  br i1 %more, label %head, label %exit
exit:
  ret void
'''
    if different:
        join = join.replace('addrspace(3) %ptr', 'addrspace(3) %other')
    if forwarding:
        head = head.replace('[%next, %join]', '[%next, %continue]')
        join = join.replace('join:\n', 'join:\n  br label %continue\ncontinue:\n')
    if extra_pred:
        # The join is no longer dominated by head. Its live values need PHIs.
        entry = entry.replace('br label %head', 'br i1 %bypass, label %head, label %join')
        join = join.replace('join:\n', 'join:\n  %seed = phi <4 x i32> [zeroinitializer, %entry], [%first, %head], [%first, %issue]\n  %iv = phi i32 [0, %entry], [%i, %head], [%i, %issue]\n')
        join = join.replace('%sum = add <4 x i32> %first', '%sum = add <4 x i32> %seed').replace('%next = add i32 %i', '%next = add i32 %iv')
    attrs = ' "frisk.cross_bn_k0"="shared-plan-v1"' if marker else ''
    return '''target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16)
declare void @__frisk_async_load_lds(ptr addrspace(1), ptr addrspace(3), i32, i32, i32) convergent
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1), ptr addrspace(3), i32, i32, i32)
declare void @unknown()
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) convergent
define amdgpu_kernel void @kernel(ptr addrspace(1) %src, ptr addrspace(3) %dst, ptr addrspace(3) %base, i32 %n, i1 %take, i1 %bypass)''' + attrs + ' {\n' + entry + (join + issue + head if reverse else head + issue + join) + '}\n'


def main():
    tool = Path(sys.argv[1]).resolve()
    cases = 0
    for reverse in (False, True):
        for forwarding in (False, True):
            for copy in (COPY, COPY.replace('@__frisk_async_load_lds', '@llvm.amdgcn.global.load.async.lds').replace('i32 8,', 'i32 16,')):
                original = source(copy, reverse=reverse, forwarding=forwarding)
                result = run([tool, '--optimize-register-lifetimes', '-'], original)
                run([LLVM/'opt', '-passes=verify', '-disable-output', '-'], result)
                kernel = body(result, 'kernel')
                assert len(re.findall(r'%address.thread\w* = ', kernel)) == 1, kernel
                ptrs = re.findall(r'call <4 x i32> @llvm.hcu.ds.read[^\n]*ptr addrspace\(3\) (%[-\w.$]+)', kernel)
                assert len(ptrs) == 2 and ptrs[0] == ptrs[1], kernel
                issue = kernel.split('issue:', 1)[1].split('\n\n', 1)[0]
                assert 'call void @' in issue and 'asyncmark()' in issue
                assert 'ds.read' not in issue and 'address.thread' not in issue
                assert kernel.count('call void @llvm.amdgcn.s.barrier') == 1
                optimized = run([LLVM/'opt', '-S', '-passes=default<O3>,verify', '-'], result)
                assert len(re.findall(r'%address.thread\w* = ', body(optimized, 'kernel'))) == 1
                replay = run([tool, '--optimize-register-lifetimes', '-'], result)
                assert normalize_metadata_ids(replay) == normalize_metadata_ids(result)
                cases += 1
    # The attribute alone never authorizes reuse through arbitrary effects,
    # a wait/barrier, a malformed bridge, or a join with a third predecessor.
    blocked = [source(marker=False), source(extra_pred=True)]
    for effect in ('call void @unknown()', 'call void @llvm.amdgcn.s.barrier()',
                   'call void @llvm.amdgcn.wait.asyncmark(i16 0)',
                   'store volatile i32 0, ptr addrspace(3) %dst',
                   '%loaded = load volatile i32, ptr addrspace(3) %dst',
                   'call void asm sideeffect "", ""()', COPY.replace('i32 8,', 'i32 7,')):
        blocked.append(source(effect))
    for original in blocked:
        result = run([tool, '--optimize-register-lifetimes', '-'], original)
        run([LLVM/'opt', '-passes=default<O3>,verify', '-disable-output', '-'], result)
        assert len(re.findall(r'%address.thread\w* = ', body(result, 'kernel'))) == 2
        cases += 1
    result = run([tool, '--optimize-register-lifetimes', '-'], source(different=True))
    ptrs = re.findall(r'call <4 x i32> @llvm.hcu.ds.read[^\n]*ptr addrspace\(3\) (%[-\w.$]+)', result)
    assert len(ptrs) == 2 and ptrs[0] != ptrs[1]
    run([LLVM/'opt', '-passes=default<O3>,verify', '-disable-output', '-'], result)
    print(f'PASS {cases + 1} async-diamond cases: address reuse, effects, dominance, loop backedge, O3 and replay')


if __name__ == '__main__':
    main()
