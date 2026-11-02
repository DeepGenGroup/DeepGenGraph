; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_lds_prefetch_pass/pair_probe.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_lds_prefetch_pass/pair_probe.ll"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@shm_0 = local_unnamed_addr addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair0(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 128
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair1(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 4096
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 4224
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair2(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %1 = trunc i64 %.idx to i32
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %1
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %2 = trunc i64 %.idx1410 to i32
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i32 %2
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %3 = trunc nuw nsw i64 %address.local546 to i32
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i32 %3
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %4 = trunc nsw i64 %address.local550 to i32
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i32 %4
  %address.local553 = and i64 %address.local542, 12
  %5 = trunc nuw nsw i64 %address.local553 to i32
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i32 %5
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %6 = trunc nuw nsw i64 %address.local556 to i32
  %address.local559.idx1 = mul nuw nsw i32 %6, 2032
  %address.local559 = getelementptr i8, ptr addrspace(3) %ssa143, i32 %address.local559.idx1
  %ssa144 = add nsw i64 %.idx1410, %.idx
  %.idx1 = shl nuw nsw i64 %address.local546, 1
  %ssa145 = or disjoint i64 %ssa144, %.idx1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa148 = add nsw i64 %ssa145, %address.local559.idx
  %ssa146 = or disjoint i64 %address.local550, %address.local553
  %ssa147 = shl nsw i64 %ssa146, 1
  %ssa149 = add nsw i64 %ssa148, %ssa147
  %ssa150 = lshr i64 %ssa149, 6
  %ssa151 = and i64 %ssa150, 8
  %ssa152 = xor i64 %ssa151, %ssa149
  %ssa153 = sub i64 %ssa152, %ssa149
  %7 = trunc i64 %ssa153 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %7
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 128
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 1024
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %8 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair3(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %1 = trunc i64 %.idx to i32
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %1
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %2 = trunc i64 %.idx1410 to i32
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i32 %2
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %3 = trunc nuw nsw i64 %address.local546 to i32
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i32 %3
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %4 = trunc nsw i64 %address.local550 to i32
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i32 %4
  %address.local553 = and i64 %address.local542, 12
  %5 = trunc nuw nsw i64 %address.local553 to i32
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i32 %5
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %6 = trunc nuw nsw i64 %address.local556 to i32
  %address.local559.idx1 = mul nuw nsw i32 %6, 2032
  %address.local559 = getelementptr i8, ptr addrspace(3) %ssa143, i32 %address.local559.idx1
  %ssa144 = add nsw i64 %.idx1410, %.idx
  %.idx1 = shl nuw nsw i64 %address.local546, 1
  %ssa145 = or disjoint i64 %ssa144, %.idx1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa148 = add nsw i64 %ssa145, %address.local559.idx
  %ssa146 = or disjoint i64 %address.local550, %address.local553
  %ssa147 = shl nsw i64 %ssa146, 1
  %ssa149 = add nsw i64 %ssa148, %ssa147
  %ssa150 = lshr i64 %ssa149, 6
  %ssa151 = and i64 %ssa150, 8
  %ssa152 = xor i64 %ssa151, %ssa149
  %ssa153 = sub i64 %ssa152, %ssa149
  %7 = trunc i64 %ssa153 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %7
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 4224
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 5120
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %8 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair4(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 1024
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 1152
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair5(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 5120
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 5248
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair6(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %1 = trunc i64 %.idx to i32
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %1
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %2 = trunc i64 %.idx1410 to i32
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i32 %2
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %3 = trunc nuw nsw i64 %address.local546 to i32
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i32 %3
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %4 = trunc nsw i64 %address.local550 to i32
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i32 %4
  %address.local553 = and i64 %address.local542, 12
  %5 = trunc nuw nsw i64 %address.local553 to i32
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i32 %5
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %6 = trunc nuw nsw i64 %address.local556 to i32
  %address.local559.idx1 = mul nuw nsw i32 %6, 2032
  %address.local559 = getelementptr i8, ptr addrspace(3) %ssa143, i32 %address.local559.idx1
  %ssa144 = add nsw i64 %.idx1410, %.idx
  %.idx1 = shl nuw nsw i64 %address.local546, 1
  %ssa145 = or disjoint i64 %ssa144, %.idx1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa148 = add nsw i64 %ssa145, %address.local559.idx
  %ssa146 = or disjoint i64 %address.local550, %address.local553
  %ssa147 = shl nsw i64 %ssa146, 1
  %ssa149 = add nsw i64 %ssa148, %ssa147
  %ssa150 = lshr i64 %ssa149, 6
  %ssa151 = and i64 %ssa150, 8
  %ssa152 = xor i64 %ssa151, %ssa149
  %ssa153 = sub i64 %ssa152, %ssa149
  %7 = trunc i64 %ssa153 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %7
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 1152
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 5248
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %8 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair7(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %1 = trunc i64 %.idx to i32
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %1
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %2 = trunc i64 %.idx1410 to i32
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i32 %2
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %3 = trunc nuw nsw i64 %address.local546 to i32
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i32 %3
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %4 = trunc nsw i64 %address.local550 to i32
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i32 %4
  %address.local553 = and i64 %address.local542, 12
  %5 = trunc nuw nsw i64 %address.local553 to i32
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i32 %5
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %6 = trunc nuw nsw i64 %address.local556 to i32
  %address.local559.idx1 = mul nuw nsw i32 %6, 2032
  %address.local559 = getelementptr i8, ptr addrspace(3) %ssa143, i32 %address.local559.idx1
  %ssa144 = add nsw i64 %.idx1410, %.idx
  %.idx1 = shl nuw nsw i64 %address.local546, 1
  %ssa145 = or disjoint i64 %ssa144, %.idx1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa148 = add nsw i64 %ssa145, %address.local559.idx
  %ssa146 = or disjoint i64 %address.local550, %address.local553
  %ssa147 = shl nsw i64 %ssa146, 1
  %ssa149 = add nsw i64 %ssa148, %ssa147
  %ssa150 = lshr i64 %ssa149, 6
  %ssa151 = and i64 %ssa150, 8
  %ssa152 = xor i64 %ssa151, %ssa149
  %ssa153 = sub i64 %ssa152, %ssa149
  %7 = trunc i64 %ssa153 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %7
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 8192
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 8320
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %8 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair8(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %1 = trunc i64 %.idx to i32
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %1
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %2 = trunc i64 %.idx1410 to i32
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i32 %2
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %3 = trunc nuw nsw i64 %address.local546 to i32
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i32 %3
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %4 = trunc nsw i64 %address.local550 to i32
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i32 %4
  %address.local553 = and i64 %address.local542, 12
  %5 = trunc nuw nsw i64 %address.local553 to i32
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i32 %5
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %6 = trunc nuw nsw i64 %address.local556 to i32
  %address.local559.idx1 = mul nuw nsw i32 %6, 2032
  %address.local559 = getelementptr i8, ptr addrspace(3) %ssa143, i32 %address.local559.idx1
  %ssa144 = add nsw i64 %.idx1410, %.idx
  %.idx1 = shl nuw nsw i64 %address.local546, 1
  %ssa145 = or disjoint i64 %ssa144, %.idx1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa148 = add nsw i64 %ssa145, %address.local559.idx
  %ssa146 = or disjoint i64 %address.local550, %address.local553
  %ssa147 = shl nsw i64 %ssa146, 1
  %ssa149 = add nsw i64 %ssa148, %ssa147
  %ssa150 = lshr i64 %ssa149, 6
  %ssa151 = and i64 %ssa150, 8
  %ssa152 = xor i64 %ssa151, %ssa149
  %ssa153 = sub i64 %ssa152, %ssa149
  %7 = trunc i64 %ssa153 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %7
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 12288
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 12416
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %8 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair9(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 8192
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 8320
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair10(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 12288
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 12416
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair11(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %1 = trunc i64 %.idx to i32
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %1
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %2 = trunc i64 %.idx1410 to i32
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i32 %2
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %3 = trunc nuw nsw i64 %address.local546 to i32
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i32 %3
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %4 = trunc nsw i64 %address.local550 to i32
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i32 %4
  %address.local553 = and i64 %address.local542, 12
  %5 = trunc nuw nsw i64 %address.local553 to i32
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i32 %5
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %6 = trunc nuw nsw i64 %address.local556 to i32
  %address.local559.idx1 = mul nuw nsw i32 %6, 2032
  %address.local559 = getelementptr i8, ptr addrspace(3) %ssa143, i32 %address.local559.idx1
  %ssa144 = add nsw i64 %.idx1410, %.idx
  %.idx1 = shl nuw nsw i64 %address.local546, 1
  %ssa145 = or disjoint i64 %ssa144, %.idx1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa148 = add nsw i64 %ssa145, %address.local559.idx
  %ssa146 = or disjoint i64 %address.local550, %address.local553
  %ssa147 = shl nsw i64 %ssa146, 1
  %ssa149 = add nsw i64 %ssa148, %ssa147
  %ssa150 = lshr i64 %ssa149, 6
  %ssa151 = and i64 %ssa150, 8
  %ssa152 = xor i64 %ssa151, %ssa149
  %ssa153 = sub i64 %ssa152, %ssa149
  %7 = trunc i64 %ssa153 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %7
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 9216
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 9344
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %8 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair12(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %1 = trunc i64 %.idx to i32
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %1
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %2 = trunc i64 %.idx1410 to i32
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i32 %2
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %3 = trunc nuw nsw i64 %address.local546 to i32
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i32 %3
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %4 = trunc nsw i64 %address.local550 to i32
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i32 %4
  %address.local553 = and i64 %address.local542, 12
  %5 = trunc nuw nsw i64 %address.local553 to i32
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i32 %5
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %6 = trunc nuw nsw i64 %address.local556 to i32
  %address.local559.idx1 = mul nuw nsw i32 %6, 2032
  %address.local559 = getelementptr i8, ptr addrspace(3) %ssa143, i32 %address.local559.idx1
  %ssa144 = add nsw i64 %.idx1410, %.idx
  %.idx1 = shl nuw nsw i64 %address.local546, 1
  %ssa145 = or disjoint i64 %ssa144, %.idx1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa148 = add nsw i64 %ssa145, %address.local559.idx
  %ssa146 = or disjoint i64 %address.local550, %address.local553
  %ssa147 = shl nsw i64 %ssa146, 1
  %ssa149 = add nsw i64 %ssa148, %ssa147
  %ssa150 = lshr i64 %ssa149, 6
  %ssa151 = and i64 %ssa150, 8
  %ssa152 = xor i64 %ssa151, %ssa149
  %ssa153 = sub i64 %ssa152, %ssa149
  %7 = trunc i64 %ssa153 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %7
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 13312
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 13440
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %8 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair13(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 9216
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 9344
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair14(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 13312
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 13440
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair15(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %1 = trunc i64 %.idx to i32
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %1
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %2 = trunc i64 %.idx1410 to i32
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i32 %2
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %3 = trunc nuw nsw i64 %address.local546 to i32
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i32 %3
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %4 = trunc nsw i64 %address.local550 to i32
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i32 %4
  %address.local553 = and i64 %address.local542, 12
  %5 = trunc nuw nsw i64 %address.local553 to i32
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i32 %5
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %6 = trunc nuw nsw i64 %address.local556 to i32
  %address.local559.idx1 = mul nuw nsw i32 %6, 2032
  %address.local559 = getelementptr i8, ptr addrspace(3) %ssa143, i32 %address.local559.idx1
  %ssa144 = add nsw i64 %.idx1410, %.idx
  %.idx1 = shl nuw nsw i64 %address.local546, 1
  %ssa145 = or disjoint i64 %ssa144, %.idx1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa148 = add nsw i64 %ssa145, %address.local559.idx
  %ssa146 = or disjoint i64 %address.local550, %address.local553
  %ssa147 = shl nsw i64 %ssa146, 1
  %ssa149 = add nsw i64 %ssa148, %ssa147
  %ssa150 = lshr i64 %ssa149, 6
  %ssa151 = and i64 %ssa150, 8
  %ssa152 = xor i64 %ssa151, %ssa149
  %ssa153 = sub i64 %ssa152, %ssa149
  %7 = trunc i64 %ssa153 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %7
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 128
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %8 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair16(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %1 = trunc i64 %.idx to i32
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %1
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %2 = trunc i64 %.idx1410 to i32
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i32 %2
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %3 = trunc nuw nsw i64 %address.local546 to i32
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i32 %3
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %4 = trunc nsw i64 %address.local550 to i32
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i32 %4
  %address.local553 = and i64 %address.local542, 12
  %5 = trunc nuw nsw i64 %address.local553 to i32
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i32 %5
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %6 = trunc nuw nsw i64 %address.local556 to i32
  %address.local559.idx1 = mul nuw nsw i32 %6, 2032
  %address.local559 = getelementptr i8, ptr addrspace(3) %ssa143, i32 %address.local559.idx1
  %ssa144 = add nsw i64 %.idx1410, %.idx
  %.idx1 = shl nuw nsw i64 %address.local546, 1
  %ssa145 = or disjoint i64 %ssa144, %.idx1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa148 = add nsw i64 %ssa145, %address.local559.idx
  %ssa146 = or disjoint i64 %address.local550, %address.local553
  %ssa147 = shl nsw i64 %ssa146, 1
  %ssa149 = add nsw i64 %ssa148, %ssa147
  %ssa150 = lshr i64 %ssa149, 6
  %ssa151 = and i64 %ssa150, 8
  %ssa152 = xor i64 %ssa151, %ssa149
  %ssa153 = sub i64 %ssa152, %ssa149
  %7 = trunc i64 %ssa153 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %7
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 4096
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 4224
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %8 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair17(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 128
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair18(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 4096
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 4224
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair19(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %1 = trunc i64 %.idx to i32
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %1
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %2 = trunc i64 %.idx1410 to i32
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i32 %2
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %3 = trunc nuw nsw i64 %address.local546 to i32
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i32 %3
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %4 = trunc nsw i64 %address.local550 to i32
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i32 %4
  %address.local553 = and i64 %address.local542, 12
  %5 = trunc nuw nsw i64 %address.local553 to i32
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i32 %5
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %6 = trunc nuw nsw i64 %address.local556 to i32
  %address.local559.idx1 = mul nuw nsw i32 %6, 2032
  %address.local559 = getelementptr i8, ptr addrspace(3) %ssa143, i32 %address.local559.idx1
  %ssa144 = add nsw i64 %.idx1410, %.idx
  %.idx1 = shl nuw nsw i64 %address.local546, 1
  %ssa145 = or disjoint i64 %ssa144, %.idx1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa148 = add nsw i64 %ssa145, %address.local559.idx
  %ssa146 = or disjoint i64 %address.local550, %address.local553
  %ssa147 = shl nsw i64 %ssa146, 1
  %ssa149 = add nsw i64 %ssa148, %ssa147
  %ssa150 = lshr i64 %ssa149, 6
  %ssa151 = and i64 %ssa150, 8
  %ssa152 = xor i64 %ssa151, %ssa149
  %ssa153 = sub i64 %ssa152, %ssa149
  %7 = trunc i64 %ssa153 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %7
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 1024
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 1152
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %8 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair20(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %1 = trunc i64 %.idx to i32
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %1
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %2 = trunc i64 %.idx1410 to i32
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i32 %2
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %3 = trunc nuw nsw i64 %address.local546 to i32
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i32 %3
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %4 = trunc nsw i64 %address.local550 to i32
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i32 %4
  %address.local553 = and i64 %address.local542, 12
  %5 = trunc nuw nsw i64 %address.local553 to i32
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i32 %5
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %6 = trunc nuw nsw i64 %address.local556 to i32
  %address.local559.idx1 = mul nuw nsw i32 %6, 2032
  %address.local559 = getelementptr i8, ptr addrspace(3) %ssa143, i32 %address.local559.idx1
  %ssa144 = add nsw i64 %.idx1410, %.idx
  %.idx1 = shl nuw nsw i64 %address.local546, 1
  %ssa145 = or disjoint i64 %ssa144, %.idx1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa148 = add nsw i64 %ssa145, %address.local559.idx
  %ssa146 = or disjoint i64 %address.local550, %address.local553
  %ssa147 = shl nsw i64 %ssa146, 1
  %ssa149 = add nsw i64 %ssa148, %ssa147
  %ssa150 = lshr i64 %ssa149, 6
  %ssa151 = and i64 %ssa150, 8
  %ssa152 = xor i64 %ssa151, %ssa149
  %ssa153 = sub i64 %ssa152, %ssa149
  %7 = trunc i64 %ssa153 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %7
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 5120
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 5248
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %8 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair21(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 1024
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 1152
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @pair22(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #0 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %ssa94 = and i32 %kdma.tid, 1
  %ssa95 = add nuw nsw i32 %ssa94, 1
  %ssa92 = shl nuw nsw i32 %kdma.tid, 2
  %ssa93 = and i32 %ssa92, 3084
  %narrow = or disjoint i32 %ssa95, %ssa93
  %ssa96 = lshr i32 %narrow, 1
  %ssa97 = zext nneg i32 %ssa96 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa97) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %1 = trunc i64 %address.local584 to i32
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %1
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %2 = trunc nuw nsw i64 %address.local553 to i32
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i32 %2
  %address.local555 = lshr i64 %address.local499, 5
  %address.local556 = and i64 %address.local555, 1
  %3 = trunc nuw nsw i64 %address.local556 to i32
  %ssa159.idx = mul nuw nsw i32 %3, 2032
  %ssa159 = getelementptr i8, ptr addrspace(3) %ssa158, i32 %ssa159.idx
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i32 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %4 = trunc i64 %ssa166 to i32
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %4
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 5120
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i32 5248
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8
  %probe.off = shl nuw nsw i32 %kdma.tid, 4
  %5 = zext nneg i32 %probe.off to i64
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i64 %5
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i64 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none) "amdgpu-flat-work-group-size"="256,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nounwind memory(none) }
