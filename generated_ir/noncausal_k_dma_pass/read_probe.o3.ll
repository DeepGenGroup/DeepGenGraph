; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/read_probe.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/read_probe.ll"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @probe(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #1 {
  %grp.n25 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %grp.n25) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul i32 %address.thread498, 320
  %grp.n154 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %.idx
  %address.local508 = lshr i32 %address.thread498, 2
  %1 = mul i32 %address.local508, -1280
  %grp.n155 = getelementptr i8, ptr addrspace(3) %grp.n154, i32 %1
  %address.local545 = shl i32 %address.thread498, 1
  %address.local546 = and i32 %address.local545, 24
  %grp.n156 = getelementptr [2 x i8], ptr addrspace(3) %grp.n155, i32 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %2 = trunc nsw i64 %address.local549 to i32
  %3 = and i32 %2, -64
  %grp.n157 = getelementptr [2 x i8], ptr addrspace(3) %grp.n156, i32 %3
  %4 = and i32 %address.local508, 12
  %grp.n158 = getelementptr [2 x i8], ptr addrspace(3) %grp.n157, i32 %4
  %5 = and i64 %address.local499, 32
  %.not = icmp eq i64 %5, 0
  %6 = select i1 %.not, i32 0, i32 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %grp.n158, i32 %6
  %7 = ptrtoint ptr addrspace(3) %address.local559 to i32
  %ssa39 = zext i32 %7 to i64
  %ssa40 = sub nsw i64 %ssa39, ptrtoint (ptr addrspace(3) @shm_0 to i64)
  %ssa41 = lshr i64 %ssa40, 6
  %ssa42 = and i64 %ssa41, 8
  %ssa43 = xor i64 %ssa42, %ssa40
  %ssa44 = sub nsw i64 %ssa43, %ssa40
  %8 = trunc i64 %ssa44 to i32
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i32 %8
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 4096
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %grp.n115 = and i32 %grp.n25, 1
  %grp.n1161 = add nuw nsw i32 %grp.n115, 1
  %ssa18 = shl nuw nsw i32 %grp.n25, 2
  %ssa19 = and i32 %ssa18, 12
  %narrow = or disjoint i32 %grp.n1161, %ssa19
  %ssa20 = lshr i32 %narrow, 1
  %grp.n118 = zext nneg i32 %ssa20 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %grp.n118) #2
  %9 = shl i32 %address.thread498, 30
  %10 = ashr i32 %9, 31
  %11 = trunc i64 %address.base to i32
  %address.local582.tr.neg = sub i32 %10, %11
  %12 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %address.local573
  %13 = getelementptr [2 x i8], ptr addrspace(3) %12, i32 %address.local576
  %14 = getelementptr [2 x i8], ptr addrspace(3) %13, i32 %address.local546
  %grp.n162.idx = shl i32 %address.local582.tr.neg, 7
  %grp.n162 = getelementptr i8, ptr addrspace(3) %14, i32 %grp.n162.idx
  %grp.n163 = getelementptr [2 x i8], ptr addrspace(3) %grp.n162, i32 %4
  %grp.n164 = getelementptr [2 x i8], ptr addrspace(3) %grp.n163, i32 %6
  %address.local588 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 64
  %15 = ptrtoint ptr addrspace(3) %address.local588 to i32
  %ssa45 = zext i32 %15 to i64
  %ssa46 = sub nsw i64 %ssa45, ptrtoint (ptr addrspace(3) @shm_0 to i64)
  %ssa47 = lshr i64 %ssa46, 6
  %ssa48 = and i64 %ssa47, 8
  %ssa49 = xor i64 %ssa48, %ssa46
  %ssa50 = sub nsw i64 %ssa49, %ssa46
  %16 = trunc i64 %ssa50 to i32
  %kdma.swizzled262 = getelementptr i8, ptr addrspace(3) %address.local588, i32 %16
  %kdma.gep263 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 128
  %kdma.gep264 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 4096
  %kdma.gep265 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 4224
  %kdma.gep266 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 128
  %kdma.gep267 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 1024
  %kdma.gep268 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 4224
  %kdma.gep269 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 1152
  %kdma.gep270 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 5120
  %kdma.gep271 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 5248
  %kdma.gep272 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 1024
  %kdma.gep273 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 1152
  %kdma.gep274 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 5120
  %kdma.gep275 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 5248
  %kdma.gep276 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 8192
  %kdma.gep277 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 8320
  %kdma.gep278 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 12288
  %kdma.gep279 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 12416
  %kdma.gep280 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 8192
  %kdma.gep281 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 8320
  %kdma.gep282 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 12288
  %kdma.gep283 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 12416
  %kdma.gep284 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 9216
  %kdma.gep285 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 9344
  %kdma.gep286 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 13312
  %kdma.gep287 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i32 13440
  %kdma.gep288 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 9216
  %kdma.gep289 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 9344
  %kdma.gep290 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 13312
  %kdma.gep291 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i32 13440
  %probe.value0 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8
  %probe.value1 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8
  %probe.value2 = load <4 x half>, ptr addrspace(3) %kdma.swizzled262, align 8
  %probe.value3 = load <4 x half>, ptr addrspace(3) %kdma.gep263, align 8
  %probe.value4 = load <4 x half>, ptr addrspace(3) %kdma.gep264, align 8
  %probe.value5 = load <4 x half>, ptr addrspace(3) %kdma.gep265, align 8
  %probe.value6 = load <4 x half>, ptr addrspace(3) %kdma.gep266, align 8
  %probe.value7 = load <4 x half>, ptr addrspace(3) %kdma.gep267, align 8
  %probe.value8 = load <4 x half>, ptr addrspace(3) %kdma.gep268, align 8
  %probe.value9 = load <4 x half>, ptr addrspace(3) %kdma.gep269, align 8
  %probe.value10 = load <4 x half>, ptr addrspace(3) %kdma.gep270, align 8
  %probe.value11 = load <4 x half>, ptr addrspace(3) %kdma.gep271, align 8
  %probe.value12 = load <4 x half>, ptr addrspace(3) %kdma.gep272, align 8
  %probe.value13 = load <4 x half>, ptr addrspace(3) %kdma.gep273, align 8
  %probe.value14 = load <4 x half>, ptr addrspace(3) %kdma.gep274, align 8
  %probe.value15 = load <4 x half>, ptr addrspace(3) %kdma.gep275, align 8
  %probe.value16 = load <4 x half>, ptr addrspace(3) %kdma.gep276, align 8
  %probe.value17 = load <4 x half>, ptr addrspace(3) %kdma.gep277, align 8
  %probe.value18 = load <4 x half>, ptr addrspace(3) %kdma.gep278, align 8
  %probe.value19 = load <4 x half>, ptr addrspace(3) %kdma.gep279, align 8
  %probe.value20 = load <4 x half>, ptr addrspace(3) %kdma.gep280, align 8
  %probe.value21 = load <4 x half>, ptr addrspace(3) %kdma.gep281, align 8
  %probe.value22 = load <4 x half>, ptr addrspace(3) %kdma.gep282, align 8
  %probe.value23 = load <4 x half>, ptr addrspace(3) %kdma.gep283, align 8
  %probe.value24 = load <4 x half>, ptr addrspace(3) %kdma.gep284, align 8
  %probe.value25 = load <4 x half>, ptr addrspace(3) %kdma.gep285, align 8
  %probe.value26 = load <4 x half>, ptr addrspace(3) %kdma.gep286, align 8
  %probe.value27 = load <4 x half>, ptr addrspace(3) %kdma.gep287, align 8
  %probe.value28 = load <4 x half>, ptr addrspace(3) %kdma.gep288, align 8
  %probe.value29 = load <4 x half>, ptr addrspace(3) %kdma.gep289, align 8
  %probe.value30 = load <4 x half>, ptr addrspace(3) %kdma.gep290, align 8
  %probe.value31 = load <4 x half>, ptr addrspace(3) %kdma.gep291, align 8
  %probe.off = shl nuw nsw i32 %grp.n25, 8
  %17 = zext nneg i32 %probe.off to i64
  %probe.base = getelementptr i8, ptr addrspace(1) %probe.out, i64 %17
  store <4 x half> %probe.value0, ptr addrspace(1) %probe.base, align 8
  %probe.dst1 = getelementptr i8, ptr addrspace(1) %probe.base, i64 8
  store <4 x half> %probe.value1, ptr addrspace(1) %probe.dst1, align 8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.base, i64 16
  store <4 x half> %probe.value2, ptr addrspace(1) %probe.dst2, align 8
  %probe.dst3 = getelementptr i8, ptr addrspace(1) %probe.base, i64 24
  store <4 x half> %probe.value3, ptr addrspace(1) %probe.dst3, align 8
  %probe.dst4 = getelementptr i8, ptr addrspace(1) %probe.base, i64 32
  store <4 x half> %probe.value4, ptr addrspace(1) %probe.dst4, align 8
  %probe.dst5 = getelementptr i8, ptr addrspace(1) %probe.base, i64 40
  store <4 x half> %probe.value5, ptr addrspace(1) %probe.dst5, align 8
  %probe.dst6 = getelementptr i8, ptr addrspace(1) %probe.base, i64 48
  store <4 x half> %probe.value6, ptr addrspace(1) %probe.dst6, align 8
  %probe.dst7 = getelementptr i8, ptr addrspace(1) %probe.base, i64 56
  store <4 x half> %probe.value7, ptr addrspace(1) %probe.dst7, align 8
  %probe.dst8 = getelementptr i8, ptr addrspace(1) %probe.base, i64 64
  store <4 x half> %probe.value8, ptr addrspace(1) %probe.dst8, align 8
  %probe.dst9 = getelementptr i8, ptr addrspace(1) %probe.base, i64 72
  store <4 x half> %probe.value9, ptr addrspace(1) %probe.dst9, align 8
  %probe.dst10 = getelementptr i8, ptr addrspace(1) %probe.base, i64 80
  store <4 x half> %probe.value10, ptr addrspace(1) %probe.dst10, align 8
  %probe.dst11 = getelementptr i8, ptr addrspace(1) %probe.base, i64 88
  store <4 x half> %probe.value11, ptr addrspace(1) %probe.dst11, align 8
  %probe.dst12 = getelementptr i8, ptr addrspace(1) %probe.base, i64 96
  store <4 x half> %probe.value12, ptr addrspace(1) %probe.dst12, align 8
  %probe.dst13 = getelementptr i8, ptr addrspace(1) %probe.base, i64 104
  store <4 x half> %probe.value13, ptr addrspace(1) %probe.dst13, align 8
  %probe.dst14 = getelementptr i8, ptr addrspace(1) %probe.base, i64 112
  store <4 x half> %probe.value14, ptr addrspace(1) %probe.dst14, align 8
  %probe.dst15 = getelementptr i8, ptr addrspace(1) %probe.base, i64 120
  store <4 x half> %probe.value15, ptr addrspace(1) %probe.dst15, align 8
  %probe.dst16 = getelementptr i8, ptr addrspace(1) %probe.base, i64 128
  store <4 x half> %probe.value16, ptr addrspace(1) %probe.dst16, align 8
  %probe.dst17 = getelementptr i8, ptr addrspace(1) %probe.base, i64 136
  store <4 x half> %probe.value17, ptr addrspace(1) %probe.dst17, align 8
  %probe.dst18 = getelementptr i8, ptr addrspace(1) %probe.base, i64 144
  store <4 x half> %probe.value18, ptr addrspace(1) %probe.dst18, align 8
  %probe.dst19 = getelementptr i8, ptr addrspace(1) %probe.base, i64 152
  store <4 x half> %probe.value19, ptr addrspace(1) %probe.dst19, align 8
  %probe.dst20 = getelementptr i8, ptr addrspace(1) %probe.base, i64 160
  store <4 x half> %probe.value20, ptr addrspace(1) %probe.dst20, align 8
  %probe.dst21 = getelementptr i8, ptr addrspace(1) %probe.base, i64 168
  store <4 x half> %probe.value21, ptr addrspace(1) %probe.dst21, align 8
  %probe.dst22 = getelementptr i8, ptr addrspace(1) %probe.base, i64 176
  store <4 x half> %probe.value22, ptr addrspace(1) %probe.dst22, align 8
  %probe.dst23 = getelementptr i8, ptr addrspace(1) %probe.base, i64 184
  store <4 x half> %probe.value23, ptr addrspace(1) %probe.dst23, align 8
  %probe.dst24 = getelementptr i8, ptr addrspace(1) %probe.base, i64 192
  store <4 x half> %probe.value24, ptr addrspace(1) %probe.dst24, align 8
  %probe.dst25 = getelementptr i8, ptr addrspace(1) %probe.base, i64 200
  store <4 x half> %probe.value25, ptr addrspace(1) %probe.dst25, align 8
  %probe.dst26 = getelementptr i8, ptr addrspace(1) %probe.base, i64 208
  store <4 x half> %probe.value26, ptr addrspace(1) %probe.dst26, align 8
  %probe.dst27 = getelementptr i8, ptr addrspace(1) %probe.base, i64 216
  store <4 x half> %probe.value27, ptr addrspace(1) %probe.dst27, align 8
  %probe.dst28 = getelementptr i8, ptr addrspace(1) %probe.base, i64 224
  store <4 x half> %probe.value28, ptr addrspace(1) %probe.dst28, align 8
  %probe.dst29 = getelementptr i8, ptr addrspace(1) %probe.base, i64 232
  store <4 x half> %probe.value29, ptr addrspace(1) %probe.dst29, align 8
  %probe.dst30 = getelementptr i8, ptr addrspace(1) %probe.base, i64 240
  store <4 x half> %probe.value30, ptr addrspace(1) %probe.dst30, align 8
  %probe.dst31 = getelementptr i8, ptr addrspace(1) %probe.base, i64 248
  store <4 x half> %probe.value31, ptr addrspace(1) %probe.dst31, align 8
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nounwind memory(read, argmem: write, inaccessiblemem: none, target_mem: none) "amdgpu-flat-work-group-size"="256,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" }
attributes #2 = { convergent nounwind memory(none) }
