target triple = "amdgcn-amd-amdhsa"
@shm_0 = addrspace(3) global [16384 x i8] undef, align 16
define amdgpu_kernel void @pair0(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair1(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair2(ptr addrspace(1) %probe.out) #1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i64 %address.local550
  %address.local553 = and i64 %address.local542, 12
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %ssa143, i64 %address.local557
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
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %ssa153
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair3(ptr addrspace(1) %probe.out) #1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i64 %address.local550
  %address.local553 = and i64 %address.local542, 12
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %ssa143, i64 %address.local557
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
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %ssa153
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair4(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair5(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair6(ptr addrspace(1) %probe.out) #1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i64 %address.local550
  %address.local553 = and i64 %address.local542, 12
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %ssa143, i64 %address.local557
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
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %ssa153
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair7(ptr addrspace(1) %probe.out) #1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i64 %address.local550
  %address.local553 = and i64 %address.local542, 12
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %ssa143, i64 %address.local557
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
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %ssa153
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair8(ptr addrspace(1) %probe.out) #1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i64 %address.local550
  %address.local553 = and i64 %address.local542, 12
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %ssa143, i64 %address.local557
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
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %ssa153
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair9(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair10(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair11(ptr addrspace(1) %probe.out) #1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i64 %address.local550
  %address.local553 = and i64 %address.local542, 12
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %ssa143, i64 %address.local557
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
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %ssa153
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair12(ptr addrspace(1) %probe.out) #1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i64 %address.local550
  %address.local553 = and i64 %address.local542, 12
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %ssa143, i64 %address.local557
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
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %ssa153
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair13(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair14(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair15(ptr addrspace(1) %probe.out) #1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i64 %address.local550
  %address.local553 = and i64 %address.local542, 12
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %ssa143, i64 %address.local557
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
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %ssa153
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair16(ptr addrspace(1) %probe.out) #1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i64 %address.local550
  %address.local553 = and i64 %address.local542, 12
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %ssa143, i64 %address.local557
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
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %ssa153
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair17(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair18(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair19(ptr addrspace(1) %probe.out) #1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i64 %address.local550
  %address.local553 = and i64 %address.local542, 12
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %ssa143, i64 %address.local557
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
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %ssa153
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair20(ptr addrspace(1) %probe.out) #1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %ssa139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local542 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local542, -1280
  %ssa140 = getelementptr i8, ptr addrspace(3) %ssa139, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %ssa141 = getelementptr [2 x i8], ptr addrspace(3) %ssa140, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %ssa142 = getelementptr [2 x i8], ptr addrspace(3) %ssa141, i64 %address.local550
  %address.local553 = and i64 %address.local542, 12
  %ssa143 = getelementptr [2 x i8], ptr addrspace(3) %ssa142, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %ssa143, i64 %address.local557
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
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %ssa153
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair21(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
define amdgpu_kernel void @pair22(ptr addrspace(1) %probe.out) #1 {
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
  %ssa157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local542 = lshr i64 %address.local499, 2
  %address.local553 = and i64 %address.local542, 12
  %ssa158 = getelementptr [2 x i8], ptr addrspace(3) %ssa157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %ssa159 = getelementptr [2 x i8], ptr addrspace(3) %ssa158, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %ssa159, i64 64
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %ssa161 = add nuw nsw i64 %address.local559.idx, 64
  %.idx47 = add i64 %address.local584, %address.local553
  %ssa160 = shl i64 %.idx47, 1
  %ssa162 = add i64 %ssa161, %ssa160
  %ssa163 = lshr i64 %ssa162, 6
  %ssa164 = and i64 %ssa163, 8
  %ssa165 = xor i64 %ssa164, %ssa162
  %ssa166 = sub i64 %ssa165, %ssa162
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %ssa166
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248
  %probe.a = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8
  %probe.b = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8
  %probe.off = shl i32 %kdma.tid, 4
  %probe.dst = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.dst, i32 8
  store <4 x half> %probe.a, ptr addrspace(1) %probe.dst, align 8
  store <4 x half> %probe.b, ptr addrspace(1) %probe.dst2, align 8
  ret void
}
declare i32 @llvm.amdgcn.workitem.id.x() #0
attributes #0 = { nounwind memory(none) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" }
attributes #2 = { convergent nounwind memory(none) }
