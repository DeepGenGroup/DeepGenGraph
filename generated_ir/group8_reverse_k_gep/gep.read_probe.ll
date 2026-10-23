target triple = "amdgcn-amd-amdhsa"
@shm_0 = addrspace(3) global [16384 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x() #0
define amdgpu_kernel void @probe(ptr addrspace(1) %out) #1 {
  %probe.ssa24 = call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %probe.ssa24) #2
  %address.local179 = lshr i32 %address.thread, 2
  %address.local180 = and i32 %address.local179, 4
  %address.local181 = shl nuw nsw i32 %address.thread, 1
  %address.local182 = and i32 %address.local181, 24
  %address.local183 = or disjoint i32 %address.local180, %address.local182
  %address.local184 = shl nuw nsw i32 %address.thread, 5
  %address.local185 = and i32 %address.local184, 1024
  %address.local186 = or disjoint i32 %address.local183, %address.local185
  %address.local187 = shl nuw nsw i32 %address.thread, 7
  %address.local188 = and i32 %address.local187, 384
  %address.local189 = or disjoint i32 %address.local186, %address.local188
  %address.local190 = zext nneg i32 %address.local189 to i64
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local190
  %ksw.read0.address = ptrtoint ptr addrspace(3) %address.local191 to i32
  %ksw.read.bit = and i32 %probe.ssa24, 1
  %ksw.read.mask = shl i32 %ksw.read.bit, 6
  %ksw.read0.xor = xor i32 %ksw.read0.address, %ksw.read.mask
  %ksw.read0.pointer = inttoptr i32 %ksw.read0.xor to ptr addrspace(3)
  %ksw.read1.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 4096
  %kgep.base1.bits = xor i32 %ksw.read0.xor, 64
  %kgep.base1 = inttoptr i32 %kgep.base1.bits to ptr addrspace(3)
  %ksw.read2.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 0
  %ksw.read3.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 4096
  %ksw.read4.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 128
  %ksw.read5.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 4224
  %ksw.read6.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 128
  %ksw.read7.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 4224
  %ksw.read8.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 1024
  %ksw.read9.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 5120
  %ksw.read10.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 1024
  %ksw.read11.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 5120
  %ksw.read12.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 1152
  %ksw.read13.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 5248
  %ksw.read14.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 1152
  %ksw.read15.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 5248
  %probe.value0 = load <4 x half>, ptr addrspace(3) %ksw.read0.pointer, align 8
  %probe.value1 = load <4 x half>, ptr addrspace(3) %ksw.read1.pointer, align 8
  %probe.value2 = load <4 x half>, ptr addrspace(3) %ksw.read2.pointer, align 8
  %probe.value3 = load <4 x half>, ptr addrspace(3) %ksw.read3.pointer, align 8
  %probe.value4 = load <4 x half>, ptr addrspace(3) %ksw.read4.pointer, align 8
  %probe.value5 = load <4 x half>, ptr addrspace(3) %ksw.read5.pointer, align 8
  %probe.value6 = load <4 x half>, ptr addrspace(3) %ksw.read6.pointer, align 8
  %probe.value7 = load <4 x half>, ptr addrspace(3) %ksw.read7.pointer, align 8
  %probe.value8 = load <4 x half>, ptr addrspace(3) %ksw.read8.pointer, align 8
  %probe.value9 = load <4 x half>, ptr addrspace(3) %ksw.read9.pointer, align 8
  %probe.value10 = load <4 x half>, ptr addrspace(3) %ksw.read10.pointer, align 8
  %probe.value11 = load <4 x half>, ptr addrspace(3) %ksw.read11.pointer, align 8
  %probe.value12 = load <4 x half>, ptr addrspace(3) %ksw.read12.pointer, align 8
  %probe.value13 = load <4 x half>, ptr addrspace(3) %ksw.read13.pointer, align 8
  %probe.value14 = load <4 x half>, ptr addrspace(3) %ksw.read14.pointer, align 8
  %probe.value15 = load <4 x half>, ptr addrspace(3) %ksw.read15.pointer, align 8
  %probe.output.offset = shl i32 %probe.ssa24, 7
  %probe.output.base = getelementptr i8, ptr addrspace(1) %out, i32 %probe.output.offset
  %probe.out0 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 0
  store <4 x half> %probe.value0, ptr addrspace(1) %probe.out0, align 8
  %probe.out1 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 8
  store <4 x half> %probe.value1, ptr addrspace(1) %probe.out1, align 8
  %probe.out2 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 16
  store <4 x half> %probe.value2, ptr addrspace(1) %probe.out2, align 8
  %probe.out3 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 24
  store <4 x half> %probe.value3, ptr addrspace(1) %probe.out3, align 8
  %probe.out4 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 32
  store <4 x half> %probe.value4, ptr addrspace(1) %probe.out4, align 8
  %probe.out5 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 40
  store <4 x half> %probe.value5, ptr addrspace(1) %probe.out5, align 8
  %probe.out6 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 48
  store <4 x half> %probe.value6, ptr addrspace(1) %probe.out6, align 8
  %probe.out7 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 56
  store <4 x half> %probe.value7, ptr addrspace(1) %probe.out7, align 8
  %probe.out8 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 64
  store <4 x half> %probe.value8, ptr addrspace(1) %probe.out8, align 8
  %probe.out9 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 72
  store <4 x half> %probe.value9, ptr addrspace(1) %probe.out9, align 8
  %probe.out10 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 80
  store <4 x half> %probe.value10, ptr addrspace(1) %probe.out10, align 8
  %probe.out11 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 88
  store <4 x half> %probe.value11, ptr addrspace(1) %probe.out11, align 8
  %probe.out12 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 96
  store <4 x half> %probe.value12, ptr addrspace(1) %probe.out12, align 8
  %probe.out13 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 104
  store <4 x half> %probe.value13, ptr addrspace(1) %probe.out13, align 8
  %probe.out14 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 112
  store <4 x half> %probe.value14, ptr addrspace(1) %probe.out14, align 8
  %probe.out15 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 120
  store <4 x half> %probe.value15, ptr addrspace(1) %probe.out15, align 8
  ret void
}
attributes #0 = { nounwind memory(none) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" }
attributes #2 = { convergent nounwind memory(none) }
