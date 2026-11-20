#!/usr/bin/env python3
"""Address reuse across pure optional computations, without crossing effects."""
import re
import sys
from pathlib import Path
from check_register_lifetimes import run, body

LLVM = Path('/data2/xsl/install/bin')
TOOL = Path(sys.argv[1]).resolve()


def source(effect='', reverse=False, swapped=False, extra_pred=False, full=False, forwarding=False):
    entry = '''entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %offset = mul i32 %tid, 7
  %ptr = getelementptr float, ptr addrspace(1) %base, i32 %offset
  br label %head
'''
    if extra_pred:
        entry = entry.replace('br label %head', 'br i1 %other, label %head, label %join')
    head = '''head:
  %first = load volatile float, ptr addrspace(1) %ptr
  br i1 %take, label %slow, label %join
'''
    if swapped:
        head = head.replace('label %slow, label %join', 'label %join, label %slow')
    slow = '''slow:
  EFFECT
  %scaled = call float @llvm.exp2.f32(float %first)
  br label %join
'''.replace('EFFECT', effect)
    if '%collect =' in effect:
        slow = slow.replace('@llvm.exp2.f32(float %first)', '@llvm.exp2.f32(float %collect)')
    join = '''join:
  %state = phi float [ %first, %head ], [ %scaled, %slow ]EXTRA
  %second = load volatile float, ptr addrspace(1) %ptr
  %sum = fadd float %state, %second
  store volatile float %sum, ptr addrspace(1) %base
  ret void
'''.replace('EXTRA', ', [ 0.0, %entry ]' if extra_pred else '')
    if full:
        head = head.replace('label %join', 'label %fast')
        join = join.replace('[ %first, %head ]', '[ %first, %fast ]')
        slow += 'fast:\n  br label %join\n'
    if forwarding:
        join = join.replace('  %second = ', '  br label %continuation\ncontinuation:\n  %second = ')
    return '''target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
declare float @llvm.exp2.f32(float)
declare void @llvm.amdgcn.s.barrier()
declare float @collective(float) convergent memory(none) nounwind willreturn
@shared = addrspace(3) global float 0.0
define amdgpu_kernel void @kernel(ptr addrspace(1) %base, i1 %take, i1 %other) {
''' + entry + (join + slow + head if reverse else head + slow + join) + '}\n'


cases = 0
for shape in ('triangle', 'diamond', 'yield'):
    for reverse in (False, True):
        for swapped in (False, True):
            for effect in ('', '%unused = load volatile float, ptr addrspace(1) %base',
                           'store volatile float 0.0, ptr addrspace(1) %base',
                           'call void @llvm.amdgcn.s.barrier()',
                           '%collect = call float @collective(float %first)',
                           'call void asm sideeffect "", ""()'):
                original = source(effect, reverse, swapped, full=shape != 'triangle', forwarding=shape == 'yield')
                result = run([TOOL, '--optimize-register-lifetimes', '-'], original)
                run([LLVM/'opt', '-passes=verify', '-disable-output', '-'], result)
                kernel = body(result, 'kernel')
                anchors = re.findall(r'%address.thread\w* = ', kernel)
                assert len(anchors) == (2 if effect else 1), (effect, reverse, swapped, kernel)
                assert kernel.count('load volatile float') == original.count('load volatile float')
                assert kernel.count('store volatile float') == original.count('store volatile float')
                # Both loads must use the same materialized address only for the
                # pure two-predecessor join; memory accesses stay in their blocks.
                ptrs = re.findall(r'load volatile float, ptr addrspace\(1\) (%address.local\w*)', kernel)
                assert len(ptrs) == 2 and (ptrs[0] == ptrs[1]) == (not effect), kernel
                run([LLVM/'opt', '-passes=default<O3>,verify', '-disable-output', '-'], result)
                cases += 1
result = run([TOOL, '--optimize-register-lifetimes', '-'], source(extra_pred=True))
run([LLVM/'opt', '-passes=verify', '-disable-output', '-'], result)
assert len(re.findall(r'%address.thread\w* = ', result)) == 2
print(f'PASS {cases + 1} pure-diamond address lifetime cases (effects, convergence, dominance, block order, O3)')
