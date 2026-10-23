; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/swizzle_pass/k.read_probe.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/swizzle_pass/k.read_probe.ll"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@shm_0 = local_unnamed_addr addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @probe(ptr addrspace(1) nofree writeonly captures(none) %out) local_unnamed_addr #1 {
  %ssa34 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %ssa34) #2
  %address.local = zext nneg i32 %address.thread to i64
  %address.local406 = mul nuw nsw i64 %address.local, 160
  %address.local407 = lshr i64 %address.local, 2
  %address.local408 = mul nsw i64 %address.local407, -640
  %address.local410 = shl nuw nsw i64 %address.local, 1
  %address.local411 = and i64 %address.local410, 24
  %address.local413 = shl i64 %address.local, 62
  %address.local414 = ashr exact i64 %address.local413, 57
  %address.local415 = and i64 %address.local414, 4294967232
  %address.local418 = and i64 %address.local407, 12
  %1 = and i64 %address.local, 32
  %.not = icmp eq i64 %1, 0
  %address.local422 = select i1 %.not, i64 0, i64 1016
  %address.local412 = add nsw i64 %address.local408, %address.local406
  %address.local409 = or disjoint i64 %address.local412, %address.local411
  %address.local416 = add nsw i64 %address.local409, %address.local418
  %address.local419 = add nsw i64 %address.local416, %address.local415
  %address.local423 = add nsw i64 %address.local419, %address.local422
  %2 = trunc i64 %address.local423 to i32
  %address.local424 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i32 8192), i32 %2
  %lds.const.gep1337 = getelementptr i8, ptr addrspace(3) %address.local424, i32 4096
  %address.local435 = shl nuw nsw i32 %address.thread, 1
  %address.local436 = and i32 %address.local435, 24
  %address.local437 = shl nuw nsw i32 %address.thread, 8
  %address.local438 = and i32 %address.local437, 768
  %address.local440 = shl nuw nsw i32 %address.thread, 5
  %address.local441 = and i32 %address.local440, 96
  %3 = shl nuw nsw i32 %ssa34, 2
  %4 = and i32 %3, 12
  %ssa46 = and i32 %ssa34, 1
  %ssa472 = add nuw nsw i32 %ssa46, 1
  %narrow = or disjoint i32 %ssa472, %4
  %5 = lshr i32 %narrow, 1
  %ssa4913 = zext nneg i32 %5 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %ssa4913) #2
  %6 = shl i32 %address.thread, 30
  %7 = ashr i32 %6, 31
  %8 = trunc i64 %address.base to i32
  %address.local447.tr.neg = sub i32 %7, %8
  %9 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i32 8192), i32 %address.local436
  %10 = getelementptr [2 x i8], ptr addrspace(3) %9, i32 %address.local438
  %11 = getelementptr [2 x i8], ptr addrspace(3) %10, i32 %address.local441
  %address.local450.idx = shl i32 %address.local447.tr.neg, 7
  %address.local450 = getelementptr i8, ptr addrspace(3) %11, i32 %address.local450.idx
  %12 = trunc nuw nsw i64 %address.local418 to i32
  %address.local451 = getelementptr [2 x i8], ptr addrspace(3) %address.local450, i32 %12
  %13 = trunc nuw nsw i64 %address.local422 to i32
  %address.local452 = getelementptr [2 x i8], ptr addrspace(3) %address.local451, i32 %13
  %address.local453 = getelementptr i8, ptr addrspace(3) %address.local452, i32 64
  %lds.const.gep1338 = getelementptr i8, ptr addrspace(3) %address.local452, i32 4160
  %lds.const.gep1339 = getelementptr i8, ptr addrspace(3) %address.local424, i32 128
  %lds.const.gep1340 = getelementptr i8, ptr addrspace(3) %address.local424, i32 4224
  %lds.const.gep1341 = getelementptr i8, ptr addrspace(3) %address.local452, i32 192
  %lds.const.gep1342 = getelementptr i8, ptr addrspace(3) %address.local452, i32 4288
  %lds.const.gep1343 = getelementptr i8, ptr addrspace(3) %address.local424, i32 1024
  %lds.const.gep1344 = getelementptr i8, ptr addrspace(3) %address.local424, i32 5120
  %lds.const.gep1345 = getelementptr i8, ptr addrspace(3) %address.local452, i32 1088
  %lds.const.gep1346 = getelementptr i8, ptr addrspace(3) %address.local452, i32 5184
  %lds.const.gep1347 = getelementptr i8, ptr addrspace(3) %address.local424, i32 1152
  %lds.const.gep1348 = getelementptr i8, ptr addrspace(3) %address.local424, i32 5248
  %lds.const.gep1349 = getelementptr i8, ptr addrspace(3) %address.local452, i32 1216
  %lds.const.gep1350 = getelementptr i8, ptr addrspace(3) %address.local452, i32 5312
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
  %probe.off = shl nuw nsw i32 %ssa34, 7
  %14 = zext nneg i32 %probe.off to i64
  %probe.base = getelementptr i8, ptr addrspace(1) %out, i64 %14
  store <4 x half> %probe.value0, ptr addrspace(1) %probe.base, align 8
  %probe.out1 = getelementptr i8, ptr addrspace(1) %probe.base, i64 8
  store <4 x half> %probe.value1, ptr addrspace(1) %probe.out1, align 8
  %probe.out2 = getelementptr i8, ptr addrspace(1) %probe.base, i64 16
  store <4 x half> %probe.value2, ptr addrspace(1) %probe.out2, align 8
  %probe.out3 = getelementptr i8, ptr addrspace(1) %probe.base, i64 24
  store <4 x half> %probe.value3, ptr addrspace(1) %probe.out3, align 8
  %probe.out4 = getelementptr i8, ptr addrspace(1) %probe.base, i64 32
  store <4 x half> %probe.value4, ptr addrspace(1) %probe.out4, align 8
  %probe.out5 = getelementptr i8, ptr addrspace(1) %probe.base, i64 40
  store <4 x half> %probe.value5, ptr addrspace(1) %probe.out5, align 8
  %probe.out6 = getelementptr i8, ptr addrspace(1) %probe.base, i64 48
  store <4 x half> %probe.value6, ptr addrspace(1) %probe.out6, align 8
  %probe.out7 = getelementptr i8, ptr addrspace(1) %probe.base, i64 56
  store <4 x half> %probe.value7, ptr addrspace(1) %probe.out7, align 8
  %probe.out8 = getelementptr i8, ptr addrspace(1) %probe.base, i64 64
  store <4 x half> %probe.value8, ptr addrspace(1) %probe.out8, align 8
  %probe.out9 = getelementptr i8, ptr addrspace(1) %probe.base, i64 72
  store <4 x half> %probe.value9, ptr addrspace(1) %probe.out9, align 8
  %probe.out10 = getelementptr i8, ptr addrspace(1) %probe.base, i64 80
  store <4 x half> %probe.value10, ptr addrspace(1) %probe.out10, align 8
  %probe.out11 = getelementptr i8, ptr addrspace(1) %probe.base, i64 88
  store <4 x half> %probe.value11, ptr addrspace(1) %probe.out11, align 8
  %probe.out12 = getelementptr i8, ptr addrspace(1) %probe.base, i64 96
  store <4 x half> %probe.value12, ptr addrspace(1) %probe.out12, align 8
  %probe.out13 = getelementptr i8, ptr addrspace(1) %probe.base, i64 104
  store <4 x half> %probe.value13, ptr addrspace(1) %probe.out13, align 8
  %probe.out14 = getelementptr i8, ptr addrspace(1) %probe.base, i64 112
  store <4 x half> %probe.value14, ptr addrspace(1) %probe.out14, align 8
  %probe.out15 = getelementptr i8, ptr addrspace(1) %probe.base, i64 120
  store <4 x half> %probe.value15, ptr addrspace(1) %probe.out15, align 8
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none) "amdgpu-flat-work-group-size"="256,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" }
attributes #2 = { convergent nounwind memory(none) }
