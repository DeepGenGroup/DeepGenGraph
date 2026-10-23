target triple = "amdgcn-amd-amdhsa"
@shm_0 = addrspace(3) global [16384 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x() #0
define amdgpu_kernel void @probe(ptr addrspace(1) %out) #1 {
  %ssa34 = call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %ssa34) #2
  %address.local = zext nneg i32 %address.thread to i64
  %address.local406 = mul nuw nsw i64 %address.local, 160
  %address.local407 = lshr i64 %address.local, 2
  %address.local408 = mul nsw i64 %address.local407, -640
  %address.local409 = add nsw i64 %address.local406, %address.local408
  %address.local410 = shl nuw nsw i64 %address.local, 1
  %address.local411 = and i64 %address.local410, 24
  %address.local412 = or disjoint i64 %address.local409, %address.local411
  %address.local413 = shl i64 %address.local, 62
  %address.local414 = ashr exact i64 %address.local413, 57
  %address.local415 = and i64 %address.local414, -64
  %address.local416 = add nsw i64 %address.local412, %address.local415
  %address.local417 = lshr i64 %address.local, 2
  %address.local418 = and i64 %address.local417, 12
  %address.local419 = add nsw i64 %address.local416, %address.local418
  %address.local420 = and i64 %address.local, 63
  %address.local421 = lshr i64 %address.local420, 5
  %address.local422 = mul nuw nsw i64 %address.local421, 1016
  %address.local423 = add i64 %address.local419, %address.local422
  %address.local424 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local423
  %lds.const.gep1337 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4096
  %address.local435 = shl nuw nsw i32 %address.thread, 1
  %address.local436 = and i32 %address.local435, 24
  %address.local437 = shl nuw nsw i32 %address.thread, 8
  %address.local438 = and i32 %address.local437, 768
  %address.local439 = or disjoint i32 %address.local436, %address.local438
  %address.local440 = shl nuw nsw i32 %address.thread, 5
  %address.local441 = and i32 %address.local440, 96
  %address.local442 = or disjoint i32 %address.local439, %address.local441
  %address.local443 = zext nneg i32 %address.local442 to i64
  %ssa35 = zext nneg i32 %ssa34 to i64
  %ssa43 = shl nuw nsw i64 %ssa35, 2
  %ssa42 = lshr i64 %ssa35, 2
  %ssa44 = mul nsw i64 %ssa42, -16
  %ssa45 = add nsw i64 %ssa43, %ssa44
  %ssa46 = and i32 %ssa34, 1
  %index.compact279 = zext nneg i32 %ssa46 to i64
  %ssa47 = or disjoint i64 %ssa45, %index.compact279
  %ssa48 = add nuw nsw i64 %ssa47, 1
  %ssa49 = sdiv i64 %ssa48, 2
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %ssa49) #2
  %address.local444 = lshr i32 %address.thread, 1
  %address.local445 = and i32 %address.local444, 1
  %address.local446 = zext nneg i32 %address.local445 to i64
  %address.local447 = add nsw i64 %address.base, %address.local446
  %address.local448 = shl i64 %address.local447, 6
  %address.local449 = sub i64 %address.local443, %address.local448
  %address.local450 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local449
  %address.local451 = getelementptr [2 x i8], ptr addrspace(3) %address.local450, i64 %address.local418
  %address.local452 = getelementptr [2 x i8], ptr addrspace(3) %address.local451, i64 %address.local422
  %address.local453 = getelementptr i8, ptr addrspace(3) %address.local452, i64 64
  %lds.const.gep1338 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4096
  %lds.const.gep1339 = getelementptr i8, ptr addrspace(3) %address.local424, i64 128
  %lds.const.gep1340 = getelementptr i8, ptr addrspace(3) %address.local424, i64 4224
  %lds.const.gep1341 = getelementptr i8, ptr addrspace(3) %address.local453, i64 128
  %lds.const.gep1342 = getelementptr i8, ptr addrspace(3) %address.local453, i64 4224
  %lds.const.gep1343 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1024
  %lds.const.gep1344 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5120
  %lds.const.gep1345 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1024
  %lds.const.gep1346 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5120
  %lds.const.gep1347 = getelementptr i8, ptr addrspace(3) %address.local424, i64 1152
  %lds.const.gep1348 = getelementptr i8, ptr addrspace(3) %address.local424, i64 5248
  %lds.const.gep1349 = getelementptr i8, ptr addrspace(3) %address.local453, i64 1152
  %lds.const.gep1350 = getelementptr i8, ptr addrspace(3) %address.local453, i64 5248
  %probe.value0 = load <4 x half>, ptr addrspace(3) %address.local424, align 8
  %probe.value1 = load <4 x half>, ptr addrspace(3) %lds.const.gep1337, align 8
  %probe.value2 = load <4 x half>, ptr addrspace(3) %address.local453, align 8
  %probe.value3 = load <4 x half>, ptr addrspace(3) %lds.const.gep1338, align 8
  %probe.value4 = load <4 x half>, ptr addrspace(3) %lds.const.gep1339, align 8
  %probe.value5 = load <4 x half>, ptr addrspace(3) %lds.const.gep1340, align 8
  %probe.value6 = load <4 x half>, ptr addrspace(3) %lds.const.gep1341, align 8
  %probe.value7 = load <4 x half>, ptr addrspace(3) %lds.const.gep1342, align 8
  %probe.value8 = load <4 x half>, ptr addrspace(3) %lds.const.gep1343, align 8
  %probe.value9 = load <4 x half>, ptr addrspace(3) %lds.const.gep1344, align 8
  %probe.value10 = load <4 x half>, ptr addrspace(3) %lds.const.gep1345, align 8
  %probe.value11 = load <4 x half>, ptr addrspace(3) %lds.const.gep1346, align 8
  %probe.value12 = load <4 x half>, ptr addrspace(3) %lds.const.gep1347, align 8
  %probe.value13 = load <4 x half>, ptr addrspace(3) %lds.const.gep1348, align 8
  %probe.value14 = load <4 x half>, ptr addrspace(3) %lds.const.gep1349, align 8
  %probe.value15 = load <4 x half>, ptr addrspace(3) %lds.const.gep1350, align 8
  %probe.off = shl i32 %ssa34, 7
  %probe.base = getelementptr i8, ptr addrspace(1) %out, i32 %probe.off
  %probe.out0 = getelementptr i8, ptr addrspace(1) %probe.base, i32 0
  store <4 x half> %probe.value0, ptr addrspace(1) %probe.out0, align 8
  %probe.out1 = getelementptr i8, ptr addrspace(1) %probe.base, i32 8
  store <4 x half> %probe.value1, ptr addrspace(1) %probe.out1, align 8
  %probe.out2 = getelementptr i8, ptr addrspace(1) %probe.base, i32 16
  store <4 x half> %probe.value2, ptr addrspace(1) %probe.out2, align 8
  %probe.out3 = getelementptr i8, ptr addrspace(1) %probe.base, i32 24
  store <4 x half> %probe.value3, ptr addrspace(1) %probe.out3, align 8
  %probe.out4 = getelementptr i8, ptr addrspace(1) %probe.base, i32 32
  store <4 x half> %probe.value4, ptr addrspace(1) %probe.out4, align 8
  %probe.out5 = getelementptr i8, ptr addrspace(1) %probe.base, i32 40
  store <4 x half> %probe.value5, ptr addrspace(1) %probe.out5, align 8
  %probe.out6 = getelementptr i8, ptr addrspace(1) %probe.base, i32 48
  store <4 x half> %probe.value6, ptr addrspace(1) %probe.out6, align 8
  %probe.out7 = getelementptr i8, ptr addrspace(1) %probe.base, i32 56
  store <4 x half> %probe.value7, ptr addrspace(1) %probe.out7, align 8
  %probe.out8 = getelementptr i8, ptr addrspace(1) %probe.base, i32 64
  store <4 x half> %probe.value8, ptr addrspace(1) %probe.out8, align 8
  %probe.out9 = getelementptr i8, ptr addrspace(1) %probe.base, i32 72
  store <4 x half> %probe.value9, ptr addrspace(1) %probe.out9, align 8
  %probe.out10 = getelementptr i8, ptr addrspace(1) %probe.base, i32 80
  store <4 x half> %probe.value10, ptr addrspace(1) %probe.out10, align 8
  %probe.out11 = getelementptr i8, ptr addrspace(1) %probe.base, i32 88
  store <4 x half> %probe.value11, ptr addrspace(1) %probe.out11, align 8
  %probe.out12 = getelementptr i8, ptr addrspace(1) %probe.base, i32 96
  store <4 x half> %probe.value12, ptr addrspace(1) %probe.out12, align 8
  %probe.out13 = getelementptr i8, ptr addrspace(1) %probe.base, i32 104
  store <4 x half> %probe.value13, ptr addrspace(1) %probe.out13, align 8
  %probe.out14 = getelementptr i8, ptr addrspace(1) %probe.base, i32 112
  store <4 x half> %probe.value14, ptr addrspace(1) %probe.out14, align 8
  %probe.out15 = getelementptr i8, ptr addrspace(1) %probe.base, i32 120
  store <4 x half> %probe.value15, ptr addrspace(1) %probe.out15, align 8
  ret void
}
attributes #0 = { nounwind memory(none) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" }
attributes #2 = { convergent nounwind memory(none) }
