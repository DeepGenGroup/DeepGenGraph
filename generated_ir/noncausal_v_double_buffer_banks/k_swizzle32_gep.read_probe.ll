target triple = "amdgcn-amd-amdhsa"
@shm_0 = addrspace(3) global [16384 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x() #0
define amdgpu_kernel void @probe(ptr addrspace(1) %probe.out) #1 {
  %grp.n25 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %grp.n25) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul nuw nsw i64 %address.local499, 320
  %grp.n154 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %address.local508 = lshr i64 %address.local499, 2
  %.idx1410 = mul nsw i64 %address.local508, -1280
  %grp.n155 = getelementptr i8, ptr addrspace(3) %grp.n154, i64 %.idx1410
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %grp.n156 = getelementptr [2 x i8], ptr addrspace(3) %grp.n155, i64 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %grp.n157 = getelementptr [2 x i8], ptr addrspace(3) %grp.n156, i64 %address.local550
  %address.local553 = and i64 %address.local508, 12
  %grp.n158 = getelementptr [2 x i8], ptr addrspace(3) %grp.n157, i64 %address.local553
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %grp.n158, i64 %address.local557
  %bank.read0.addr = ptrtoint ptr addrspace(3) %address.local559 to i32
  %bank.read0.shift = lshr i32 %bank.read0.addr, 6
  %bank.read0.mask = and i32 %bank.read0.shift, 8
  %bank.read0.xor = xor i32 %bank.read0.addr, %bank.read0.mask
  %bank.read0.ptr = inttoptr i32 %bank.read0.xor to ptr addrspace(3)
  %bank.read1.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 4096
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %grp.n115 = and i32 %grp.n25, 1
  %grp.n1161 = add nuw nsw i32 %grp.n115, 1
  %probe.ssa16 = shl nuw nsw i32 %grp.n25, 2
  %probe.ssa17 = and i32 %probe.ssa16, 12
  %narrow = or disjoint i32 %grp.n1161, %probe.ssa17
  %probe.ssa18 = lshr i32 %narrow, 1
  %grp.n118 = zext nneg i32 %probe.ssa18 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %grp.n118) #2
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %grp.n162 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %grp.n163 = getelementptr [2 x i8], ptr addrspace(3) %grp.n162, i64 %address.local553
  %grp.n164 = getelementptr [2 x i8], ptr addrspace(3) %grp.n163, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 64
  %bank.read2.addr = ptrtoint ptr addrspace(3) %address.local588 to i32
  %bank.read2.shift = lshr i32 %bank.read2.addr, 6
  %bank.read2.mask = and i32 %bank.read2.shift, 8
  %bank.read2.xor = xor i32 %bank.read2.addr, %bank.read2.mask
  %bank.read2.ptr = inttoptr i32 %bank.read2.xor to ptr addrspace(3)
  %bank.read3.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 128
  %bank.read4.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 4096
  %bank.read5.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 4224
  %bank.read6.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 128
  %bank.read7.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 1024
  %bank.read8.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 4224
  %bank.read9.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 1152
  %bank.read10.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 5120
  %bank.read11.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 5248
  %bank.read12.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 1024
  %bank.read13.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 1152
  %bank.read14.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 5120
  %bank.read15.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 5248
  %bank.read16.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 8192
  %bank.read17.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 8320
  %bank.read18.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 12288
  %bank.read19.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 12416
  %bank.read20.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 8192
  %bank.read21.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 8320
  %bank.read22.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 12288
  %bank.read23.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 12416
  %bank.read24.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 9216
  %bank.read25.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 9344
  %bank.read26.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 13312
  %bank.read27.ptr = getelementptr i8, ptr addrspace(3) %bank.read0.ptr, i64 13440
  %bank.read28.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 9216
  %bank.read29.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 9344
  %bank.read30.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 13312
  %bank.read31.ptr = getelementptr i8, ptr addrspace(3) %bank.read2.ptr, i64 13440
  %probe.value0 = load <4 x half>, ptr addrspace(3) %bank.read0.ptr, align 8
  %probe.value1 = load <4 x half>, ptr addrspace(3) %bank.read1.ptr, align 8
  %probe.value2 = load <4 x half>, ptr addrspace(3) %bank.read2.ptr, align 8
  %probe.value3 = load <4 x half>, ptr addrspace(3) %bank.read3.ptr, align 8
  %probe.value4 = load <4 x half>, ptr addrspace(3) %bank.read4.ptr, align 8
  %probe.value5 = load <4 x half>, ptr addrspace(3) %bank.read5.ptr, align 8
  %probe.value6 = load <4 x half>, ptr addrspace(3) %bank.read6.ptr, align 8
  %probe.value7 = load <4 x half>, ptr addrspace(3) %bank.read7.ptr, align 8
  %probe.value8 = load <4 x half>, ptr addrspace(3) %bank.read8.ptr, align 8
  %probe.value9 = load <4 x half>, ptr addrspace(3) %bank.read9.ptr, align 8
  %probe.value10 = load <4 x half>, ptr addrspace(3) %bank.read10.ptr, align 8
  %probe.value11 = load <4 x half>, ptr addrspace(3) %bank.read11.ptr, align 8
  %probe.value12 = load <4 x half>, ptr addrspace(3) %bank.read12.ptr, align 8
  %probe.value13 = load <4 x half>, ptr addrspace(3) %bank.read13.ptr, align 8
  %probe.value14 = load <4 x half>, ptr addrspace(3) %bank.read14.ptr, align 8
  %probe.value15 = load <4 x half>, ptr addrspace(3) %bank.read15.ptr, align 8
  %probe.value16 = load <4 x half>, ptr addrspace(3) %bank.read16.ptr, align 8
  %probe.value17 = load <4 x half>, ptr addrspace(3) %bank.read17.ptr, align 8
  %probe.value18 = load <4 x half>, ptr addrspace(3) %bank.read18.ptr, align 8
  %probe.value19 = load <4 x half>, ptr addrspace(3) %bank.read19.ptr, align 8
  %probe.value20 = load <4 x half>, ptr addrspace(3) %bank.read20.ptr, align 8
  %probe.value21 = load <4 x half>, ptr addrspace(3) %bank.read21.ptr, align 8
  %probe.value22 = load <4 x half>, ptr addrspace(3) %bank.read22.ptr, align 8
  %probe.value23 = load <4 x half>, ptr addrspace(3) %bank.read23.ptr, align 8
  %probe.value24 = load <4 x half>, ptr addrspace(3) %bank.read24.ptr, align 8
  %probe.value25 = load <4 x half>, ptr addrspace(3) %bank.read25.ptr, align 8
  %probe.value26 = load <4 x half>, ptr addrspace(3) %bank.read26.ptr, align 8
  %probe.value27 = load <4 x half>, ptr addrspace(3) %bank.read27.ptr, align 8
  %probe.value28 = load <4 x half>, ptr addrspace(3) %bank.read28.ptr, align 8
  %probe.value29 = load <4 x half>, ptr addrspace(3) %bank.read29.ptr, align 8
  %probe.value30 = load <4 x half>, ptr addrspace(3) %bank.read30.ptr, align 8
  %probe.value31 = load <4 x half>, ptr addrspace(3) %bank.read31.ptr, align 8
  %probe.off = mul i32 %grp.n25, 256
  %probe.base = getelementptr i8, ptr addrspace(1) %probe.out, i32 %probe.off
  %probe.dst0 = getelementptr i8, ptr addrspace(1) %probe.base, i32 0
  store <4 x half> %probe.value0, ptr addrspace(1) %probe.dst0, align 8
  %probe.dst1 = getelementptr i8, ptr addrspace(1) %probe.base, i32 8
  store <4 x half> %probe.value1, ptr addrspace(1) %probe.dst1, align 8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.base, i32 16
  store <4 x half> %probe.value2, ptr addrspace(1) %probe.dst2, align 8
  %probe.dst3 = getelementptr i8, ptr addrspace(1) %probe.base, i32 24
  store <4 x half> %probe.value3, ptr addrspace(1) %probe.dst3, align 8
  %probe.dst4 = getelementptr i8, ptr addrspace(1) %probe.base, i32 32
  store <4 x half> %probe.value4, ptr addrspace(1) %probe.dst4, align 8
  %probe.dst5 = getelementptr i8, ptr addrspace(1) %probe.base, i32 40
  store <4 x half> %probe.value5, ptr addrspace(1) %probe.dst5, align 8
  %probe.dst6 = getelementptr i8, ptr addrspace(1) %probe.base, i32 48
  store <4 x half> %probe.value6, ptr addrspace(1) %probe.dst6, align 8
  %probe.dst7 = getelementptr i8, ptr addrspace(1) %probe.base, i32 56
  store <4 x half> %probe.value7, ptr addrspace(1) %probe.dst7, align 8
  %probe.dst8 = getelementptr i8, ptr addrspace(1) %probe.base, i32 64
  store <4 x half> %probe.value8, ptr addrspace(1) %probe.dst8, align 8
  %probe.dst9 = getelementptr i8, ptr addrspace(1) %probe.base, i32 72
  store <4 x half> %probe.value9, ptr addrspace(1) %probe.dst9, align 8
  %probe.dst10 = getelementptr i8, ptr addrspace(1) %probe.base, i32 80
  store <4 x half> %probe.value10, ptr addrspace(1) %probe.dst10, align 8
  %probe.dst11 = getelementptr i8, ptr addrspace(1) %probe.base, i32 88
  store <4 x half> %probe.value11, ptr addrspace(1) %probe.dst11, align 8
  %probe.dst12 = getelementptr i8, ptr addrspace(1) %probe.base, i32 96
  store <4 x half> %probe.value12, ptr addrspace(1) %probe.dst12, align 8
  %probe.dst13 = getelementptr i8, ptr addrspace(1) %probe.base, i32 104
  store <4 x half> %probe.value13, ptr addrspace(1) %probe.dst13, align 8
  %probe.dst14 = getelementptr i8, ptr addrspace(1) %probe.base, i32 112
  store <4 x half> %probe.value14, ptr addrspace(1) %probe.dst14, align 8
  %probe.dst15 = getelementptr i8, ptr addrspace(1) %probe.base, i32 120
  store <4 x half> %probe.value15, ptr addrspace(1) %probe.dst15, align 8
  %probe.dst16 = getelementptr i8, ptr addrspace(1) %probe.base, i32 128
  store <4 x half> %probe.value16, ptr addrspace(1) %probe.dst16, align 8
  %probe.dst17 = getelementptr i8, ptr addrspace(1) %probe.base, i32 136
  store <4 x half> %probe.value17, ptr addrspace(1) %probe.dst17, align 8
  %probe.dst18 = getelementptr i8, ptr addrspace(1) %probe.base, i32 144
  store <4 x half> %probe.value18, ptr addrspace(1) %probe.dst18, align 8
  %probe.dst19 = getelementptr i8, ptr addrspace(1) %probe.base, i32 152
  store <4 x half> %probe.value19, ptr addrspace(1) %probe.dst19, align 8
  %probe.dst20 = getelementptr i8, ptr addrspace(1) %probe.base, i32 160
  store <4 x half> %probe.value20, ptr addrspace(1) %probe.dst20, align 8
  %probe.dst21 = getelementptr i8, ptr addrspace(1) %probe.base, i32 168
  store <4 x half> %probe.value21, ptr addrspace(1) %probe.dst21, align 8
  %probe.dst22 = getelementptr i8, ptr addrspace(1) %probe.base, i32 176
  store <4 x half> %probe.value22, ptr addrspace(1) %probe.dst22, align 8
  %probe.dst23 = getelementptr i8, ptr addrspace(1) %probe.base, i32 184
  store <4 x half> %probe.value23, ptr addrspace(1) %probe.dst23, align 8
  %probe.dst24 = getelementptr i8, ptr addrspace(1) %probe.base, i32 192
  store <4 x half> %probe.value24, ptr addrspace(1) %probe.dst24, align 8
  %probe.dst25 = getelementptr i8, ptr addrspace(1) %probe.base, i32 200
  store <4 x half> %probe.value25, ptr addrspace(1) %probe.dst25, align 8
  %probe.dst26 = getelementptr i8, ptr addrspace(1) %probe.base, i32 208
  store <4 x half> %probe.value26, ptr addrspace(1) %probe.dst26, align 8
  %probe.dst27 = getelementptr i8, ptr addrspace(1) %probe.base, i32 216
  store <4 x half> %probe.value27, ptr addrspace(1) %probe.dst27, align 8
  %probe.dst28 = getelementptr i8, ptr addrspace(1) %probe.base, i32 224
  store <4 x half> %probe.value28, ptr addrspace(1) %probe.dst28, align 8
  %probe.dst29 = getelementptr i8, ptr addrspace(1) %probe.base, i32 232
  store <4 x half> %probe.value29, ptr addrspace(1) %probe.dst29, align 8
  %probe.dst30 = getelementptr i8, ptr addrspace(1) %probe.base, i32 240
  store <4 x half> %probe.value30, ptr addrspace(1) %probe.dst30, align 8
  %probe.dst31 = getelementptr i8, ptr addrspace(1) %probe.base, i32 248
  store <4 x half> %probe.value31, ptr addrspace(1) %probe.dst31, align 8
  ret void
}
attributes #0 = { nounwind memory(none) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" }
attributes #2 = { convergent nounwind memory(none) }
