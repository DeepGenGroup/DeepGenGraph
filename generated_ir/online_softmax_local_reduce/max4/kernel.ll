; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
  %v5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %v6 = sext i32 %v5 to i64, !dbg !5
  %v7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %v8 = sext i32 %v7 to i64, !dbg !5
  %v9 = shl nsw i64 %v8, 5, !dbg !5
  %v10 = add nsw i64 %v9, %v6, !dbg !5
  %v11 = lshr i64 %v10, 5, !dbg !5
  %v12 = and i64 %v11, 576460752303423484, !dbg !5
  %v13 = and i64 %v6, 3, !dbg !5
  %v14 = or disjoint i64 %v12, %v13, !dbg !5
  %v15 = sub nsw i64 31, %v14, !dbg !5
  %v16 = lshr i64 %v10, 2, !dbg !5
  %v17 = icmp ugt i64 %v10, 1023, !dbg !8
  %v18 = add nsw i64 %v14, -32, !dbg !8
  %v19 = select i1 %v17, i64 %v18, i64 %v15, !dbg !8
  %v20 = sdiv i64 %v19, 32, !dbg !8
  %v21 = sext i1 %v17 to i64, !dbg !8
  %v22 = xor i64 %v20, %v21, !dbg !8
  %v23 = add nsw i64 %v16, %v22, !dbg !8
  %v24 = and i64 %v23, 31, !dbg !8
  %v25 = shl nsw i64 %v15, 7, !dbg !8
  %v26 = mul nsw i64 %v22, -4096, !dbg !8
  %v27 = add i64 %v25, %v26, !dbg !8
  %v28 = shl nuw nsw i64 %v24, 19, !dbg !8
  %v29 = shl nsw i64 %v27, 7, !dbg !8
  %v30 = add i64 %v28, %v29, !dbg !8
  %v31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !39
  %v32 = zext nneg i32 %v31 to i64, !dbg !8
  %v33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %v30, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !8, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local339 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local340 = and i64 %address.local339, 1536, !dbg !8
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !8
  %address.local342 = ptrtoint ptr addrspace(3) %address.local341 to i64, !dbg !8
  %address.local343 = trunc i64 %address.local342 to i32, !dbg !8
  %v34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !8
  %v35 = zext i32 %v34 to i64, !dbg !8
  %v36 = inttoptr i64 %v35 to ptr addrspace(3), !dbg !8
  %address.local344 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local345 = and i64 %address.local344, 16256, !dbg !8
  %address.local346 = lshr i64 %address.local, 4, !dbg !8
  %address.local347 = and i64 %address.local346, 8, !dbg !8
  %address.local348 = or disjoint i64 %address.local345, %address.local347, !dbg !8
  %address.local349 = getelementptr [2 x i8], ptr addrspace(1) %v33, i64 %address.local348, !dbg !8
  %v37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v38 = ptrtoint ptr addrspace(1) %address.local349 to i64, !dbg !8
  %v39 = sub i64 %v38, %v37, !dbg !8
  %dma.buffer.offset1612 = trunc i64 %v39 to i32, !dbg !8
  %v40 = trunc i64 %v37 to i32, !dbg !8
  %v41 = insertelement <4 x i32> poison, i32 %v40, i64 0, !dbg !8
  %v42 = lshr i64 %v37, 32, !dbg !8
  %v43 = trunc i64 %v42 to i32, !dbg !8
  %v44 = insertelement <4 x i32> %v41, i32 %v43, i64 1, !dbg !8
  %v45 = insertelement <4 x i32> %v44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1613 = insertelement <4 x i32> %v45, i32 131072, i64 3, !dbg !8
  %v46 = ptrtoint ptr addrspace(3) %v36 to i32, !dbg !8
  %dma.buffer.lds1614 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v46), !dbg !8
  %v47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1612, <4 x i32> %dma.buffer.rsrc1613) #10, !dbg !8
  %address.local350 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local351 = shl nuw nsw i64 %address.local350, 3, !dbg !8
  %address.local352 = and i64 %address.local351, 3584, !dbg !8
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !8
  %address.local354 = ptrtoint ptr addrspace(3) %address.local353 to i64, !dbg !8
  %address.local355 = trunc i64 %address.local354 to i32, !dbg !8
  %v48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !8
  %v49 = zext i32 %v48 to i64, !dbg !8
  %v50 = inttoptr i64 %v49 to ptr addrspace(3), !dbg !8
  %address.local356 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local357 = and i64 %address.local356, 16256, !dbg !8
  %address.local358 = lshr i64 %address.local350, 4, !dbg !8
  %address.local359 = and i64 %address.local358, 24, !dbg !8
  %address.local360 = or disjoint i64 %address.local357, %address.local359, !dbg !8
  %address.local361 = getelementptr [2 x i8], ptr addrspace(1) %v33, i64 %address.local360, !dbg !8
  %v51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v52 = ptrtoint ptr addrspace(1) %address.local361 to i64, !dbg !8
  %v53 = sub i64 %v52, %v51, !dbg !8
  %dma.buffer.offset1609 = trunc i64 %v53 to i32, !dbg !8
  %v54 = trunc i64 %v51 to i32, !dbg !8
  %v55 = insertelement <4 x i32> poison, i32 %v54, i64 0, !dbg !8
  %v56 = lshr i64 %v51, 32, !dbg !8
  %v57 = trunc i64 %v56 to i32, !dbg !8
  %v58 = insertelement <4 x i32> %v55, i32 %v57, i64 1, !dbg !8
  %v59 = insertelement <4 x i32> %v58, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1610 = insertelement <4 x i32> %v59, i32 131072, i64 3, !dbg !8
  %v60 = ptrtoint ptr addrspace(3) %v50 to i32, !dbg !8
  %dma.buffer.lds1611 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v60), !dbg !8
  %v61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1609, <4 x i32> %dma.buffer.rsrc1610) #10, !dbg !8
  %v62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %v63 = sext i32 %v62 to i64, !dbg !5
  %v64 = shl i64 %v15, 7, !dbg !10
  %v65 = lshr i64 %v32, 6, !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %v66 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %v30, !dbg !8
  %v67 = getelementptr i8, ptr addrspace(1) %v66, i64 64, !dbg !8
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !8
  %address.local363 = ptrtoint ptr addrspace(3) %address.local362 to i64, !dbg !8
  %address.local364 = trunc i64 %address.local363 to i32, !dbg !8
  %v68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !8
  %v69 = zext i32 %v68 to i64, !dbg !8
  %v70 = inttoptr i64 %v69 to ptr addrspace(3), !dbg !8
  %address.local365 = getelementptr [2 x i8], ptr addrspace(1) %v67, i64 %address.local348, !dbg !8
  %v71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v72 = ptrtoint ptr addrspace(1) %address.local365 to i64, !dbg !8
  %v73 = sub i64 %v72, %v71, !dbg !8
  %dma.buffer.offset1606 = trunc i64 %v73 to i32, !dbg !8
  %v74 = trunc i64 %v71 to i32, !dbg !8
  %v75 = insertelement <4 x i32> poison, i32 %v74, i64 0, !dbg !8
  %v76 = lshr i64 %v71, 32, !dbg !8
  %v77 = trunc i64 %v76 to i32, !dbg !8
  %v78 = insertelement <4 x i32> %v75, i32 %v77, i64 1, !dbg !8
  %v79 = insertelement <4 x i32> %v78, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1607 = insertelement <4 x i32> %v79, i32 131072, i64 3, !dbg !8
  %v80 = ptrtoint ptr addrspace(3) %v70 to i32, !dbg !8
  %dma.buffer.lds1608 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v80), !dbg !8
  %v81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1606, <4 x i32> %dma.buffer.rsrc1607) #10, !dbg !8
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !8
  %address.local367 = ptrtoint ptr addrspace(3) %address.local366 to i64, !dbg !8
  %address.local368 = trunc i64 %address.local367 to i32, !dbg !8
  %v82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !8
  %v83 = zext i32 %v82 to i64, !dbg !8
  %v84 = inttoptr i64 %v83 to ptr addrspace(3), !dbg !8
  %address.local369 = getelementptr [2 x i8], ptr addrspace(1) %v67, i64 %address.local360, !dbg !8
  %v85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v86 = ptrtoint ptr addrspace(1) %address.local369 to i64, !dbg !8
  %v87 = sub i64 %v86, %v85, !dbg !8
  %dma.buffer.offset1603 = trunc i64 %v87 to i32, !dbg !8
  %v88 = trunc i64 %v85 to i32, !dbg !8
  %v89 = insertelement <4 x i32> poison, i32 %v88, i64 0, !dbg !8
  %v90 = lshr i64 %v85, 32, !dbg !8
  %v91 = trunc i64 %v90 to i32, !dbg !8
  %v92 = insertelement <4 x i32> %v89, i32 %v91, i64 1, !dbg !8
  %v93 = insertelement <4 x i32> %v92, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1604 = insertelement <4 x i32> %v93, i32 131072, i64 3, !dbg !8
  %v94 = ptrtoint ptr addrspace(3) %v84 to i32, !dbg !8
  %dma.buffer.lds1605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v94), !dbg !8
  %v95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1603, <4 x i32> %dma.buffer.rsrc1604) #10, !dbg !8
  %address.local370 = lshr i32 %address.thread, 2, !dbg !11
  %address.local371 = and i32 %address.local370, 4, !dbg !11
  %address.local372 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local373 = and i32 %address.local372, 384, !dbg !11
  %address.local374 = or disjoint i32 %address.local371, %address.local373, !dbg !11
  %address.local375 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local376 = and i32 %address.local375, 120, !dbg !11
  %address.local377 = or disjoint i32 %address.local374, %address.local376, !dbg !11
  %address.local378 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local379 = and i32 %address.local378, 1024, !dbg !11
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !11
  %address.local381 = zext nneg i32 %address.local380 to i64, !dbg !11
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !11
  %v96 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !11
  %v97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %v98 = fmul <4 x half> %v96, splat (half 0xH3015), !dbg !11
  %lds.const.gep1413 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !11
  %v99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !11
  %v100 = fmul <4 x half> %v97, splat (half 0xH3015), !dbg !11
  %lds.const.gep1414 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !11
  %v101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !11
  %v102 = fmul <4 x half> %v99, splat (half 0xH3015), !dbg !11
  %v103 = fmul <4 x half> %v101, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %v104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %v30, !dbg !8
  %v105 = getelementptr i8, ptr addrspace(1) %v104, i64 128, !dbg !8
  %v106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !8
  %v107 = zext i32 %v106 to i64, !dbg !8
  %v108 = inttoptr i64 %v107 to ptr addrspace(3), !dbg !8
  %address.local425 = getelementptr [2 x i8], ptr addrspace(1) %v105, i64 %address.local348, !dbg !8
  %v109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v110 = ptrtoint ptr addrspace(1) %address.local425 to i64, !dbg !8
  %v111 = sub i64 %v110, %v109, !dbg !8
  %dma.buffer.offset1600 = trunc i64 %v111 to i32, !dbg !8
  %v112 = trunc i64 %v109 to i32, !dbg !8
  %v113 = insertelement <4 x i32> poison, i32 %v112, i64 0, !dbg !8
  %v114 = lshr i64 %v109, 32, !dbg !8
  %v115 = trunc i64 %v114 to i32, !dbg !8
  %v116 = insertelement <4 x i32> %v113, i32 %v115, i64 1, !dbg !8
  %v117 = insertelement <4 x i32> %v116, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1601 = insertelement <4 x i32> %v117, i32 131072, i64 3, !dbg !8
  %v118 = ptrtoint ptr addrspace(3) %v108 to i32, !dbg !8
  %dma.buffer.lds1602 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v118), !dbg !8
  %v119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1602, i32 %dma.buffer.offset1600, <4 x i32> %dma.buffer.rsrc1601) #10, !dbg !8
  %v120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !8
  %v121 = zext i32 %v120 to i64, !dbg !8
  %v122 = inttoptr i64 %v121 to ptr addrspace(3), !dbg !8
  %address.local426 = getelementptr [2 x i8], ptr addrspace(1) %v105, i64 %address.local360, !dbg !8
  %v123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v124 = ptrtoint ptr addrspace(1) %address.local426 to i64, !dbg !8
  %v125 = sub i64 %v124, %v123, !dbg !8
  %dma.buffer.offset1597 = trunc i64 %v125 to i32, !dbg !8
  %v126 = trunc i64 %v123 to i32, !dbg !8
  %v127 = insertelement <4 x i32> poison, i32 %v126, i64 0, !dbg !8
  %v128 = lshr i64 %v123, 32, !dbg !8
  %v129 = trunc i64 %v128 to i32, !dbg !8
  %v130 = insertelement <4 x i32> %v127, i32 %v129, i64 1, !dbg !8
  %v131 = insertelement <4 x i32> %v130, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1598 = insertelement <4 x i32> %v131, i32 131072, i64 3, !dbg !8
  %v132 = ptrtoint ptr addrspace(3) %v122 to i32, !dbg !8
  %dma.buffer.lds1599 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v132), !dbg !8
  %v133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1599, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1598) #10, !dbg !8
  %lds.const.gep1415 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !11
  %v134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !11
  %lds.const.gep1416 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !11
  %v135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !11
  %lds.const.gep1417 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !11
  %v136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !11
  %v137 = fmul <4 x half> %v134, splat (half 0xH3015), !dbg !11
  %lds.const.gep1418 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !11
  %v138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !11
  %v139 = fmul <4 x half> %v135, splat (half 0xH3015), !dbg !11
  %v140 = fmul <4 x half> %v136, splat (half 0xH3015), !dbg !11
  %v141 = fmul <4 x half> %v138, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %v142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %v30, !dbg !8
  %v143 = getelementptr i8, ptr addrspace(1) %v142, i64 192, !dbg !8
  %v144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !8
  %v145 = zext i32 %v144 to i64, !dbg !8
  %v146 = inttoptr i64 %v145 to ptr addrspace(3), !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(1) %v143, i64 %address.local348, !dbg !8
  %v147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v148 = ptrtoint ptr addrspace(1) %address.local431 to i64, !dbg !8
  %v149 = sub i64 %v148, %v147, !dbg !8
  %dma.buffer.offset1594 = trunc i64 %v149 to i32, !dbg !8
  %v150 = trunc i64 %v147 to i32, !dbg !8
  %v151 = insertelement <4 x i32> poison, i32 %v150, i64 0, !dbg !8
  %v152 = lshr i64 %v147, 32, !dbg !8
  %v153 = trunc i64 %v152 to i32, !dbg !8
  %v154 = insertelement <4 x i32> %v151, i32 %v153, i64 1, !dbg !8
  %v155 = insertelement <4 x i32> %v154, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1595 = insertelement <4 x i32> %v155, i32 131072, i64 3, !dbg !8
  %v156 = ptrtoint ptr addrspace(3) %v146 to i32, !dbg !8
  %dma.buffer.lds1596 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v156), !dbg !8
  %v157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1596, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1595) #10, !dbg !8
  %v158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !8
  %v159 = zext i32 %v158 to i64, !dbg !8
  %v160 = inttoptr i64 %v159 to ptr addrspace(3), !dbg !8
  %address.local432 = getelementptr [2 x i8], ptr addrspace(1) %v143, i64 %address.local360, !dbg !8
  %v161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v162 = ptrtoint ptr addrspace(1) %address.local432 to i64, !dbg !8
  %v163 = sub i64 %v162, %v161, !dbg !8
  %dma.buffer.offset1591 = trunc i64 %v163 to i32, !dbg !8
  %v164 = trunc i64 %v161 to i32, !dbg !8
  %v165 = insertelement <4 x i32> poison, i32 %v164, i64 0, !dbg !8
  %v166 = lshr i64 %v161, 32, !dbg !8
  %v167 = trunc i64 %v166 to i32, !dbg !8
  %v168 = insertelement <4 x i32> %v165, i32 %v167, i64 1, !dbg !8
  %v169 = insertelement <4 x i32> %v168, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1592 = insertelement <4 x i32> %v169, i32 131072, i64 3, !dbg !8
  %v170 = ptrtoint ptr addrspace(3) %v160 to i32, !dbg !8
  %dma.buffer.lds1593 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v170), !dbg !8
  %v171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1592) #10, !dbg !8
  %v172 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !11
  %lds.const.gep1419 = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !11
  %v173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1419, align 8, !dbg !11
  %lds.const.gep1420 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !11
  %v174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1420, align 8, !dbg !11
  %v175 = fmul <4 x half> %v172, splat (half 0xH3015), !dbg !11
  %lds.const.gep1421 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !11
  %v176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1421, align 8, !dbg !11
  %v177 = fmul <4 x half> %v173, splat (half 0xH3015), !dbg !11
  %v178 = fmul <4 x half> %v174, splat (half 0xH3015), !dbg !11
  %v179 = fmul <4 x half> %v176, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1422 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !11
  %v180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1422, align 8, !dbg !11
  %lds.const.gep1423 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !11
  %v181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1423, align 8, !dbg !11
  %lds.const.gep1424 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !11
  %v182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1424, align 8, !dbg !11
  %v183 = fmul <4 x half> %v180, splat (half 0xH3015), !dbg !11
  %lds.const.gep1425 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !11
  %v184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1425, align 8, !dbg !11
  %v185 = fmul <4 x half> %v181, splat (half 0xH3015), !dbg !11
  %v186 = fmul <4 x half> %v182, splat (half 0xH3015), !dbg !11
  %v187 = fmul <4 x half> %v184, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %v188 = or disjoint i64 %v64, 127, !dbg !13
  %v189 = add i64 %v64, 128, !dbg !13
  %v190 = icmp uge i64 %v189, %v188, !dbg !13
  %v191 = sub nsw i64 32, %v14, !dbg !15
  %v192 = shl i64 %v16, 7, !dbg !16
  %v193 = and i64 %v192, 3968, !dbg !16
  %v194 = shl nsw i64 %v63, 12, !dbg !16
  %v195 = or disjoint i64 %v193, %v194, !dbg !16
  %v196 = lshr i64 %v32, 2, !dbg !17
  %v197 = shl nuw nsw i64 %v32, 2, !dbg !17
  %v198 = mul nsw i64 %v196, -16, !dbg !17
  %v199 = add nsw i64 %v197, %v198, !dbg !17
  %v200 = and i32 %v31, 1, !dbg !17
  %index.compact279 = zext nneg i32 %v200 to i64, !dbg !17
  %v201 = or disjoint i64 %v199, %index.compact279, !dbg !17
  %v202 = add nuw nsw i64 %v201, 1, !dbg !17
  %v203 = sdiv i64 %v202, 2, !dbg !17
  br label %v204, !dbg !7

v204:                                              ; preds = %v848, %4
  %v205 = phi i64 [ %v1779, %v848 ], [ 0, %4 ], !dbg !7
  %v206 = phi [2 x <8 x float>] [ %v1764, %v848 ], [ zeroinitializer, %4 ], !dbg !7
  %v207 = phi [2 x <1 x float>] [ %v1370, %v848 ], [ zeroinitializer, %4 ], !dbg !7
  %v208 = phi [2 x <1 x float>] [ %v1766, %v848 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %v209 = phi [2 x <8 x float>] [ %v1770, %v848 ], [ zeroinitializer, %4 ], !dbg !7
  %v210 = phi [2 x <8 x float>] [ %v1774, %v848 ], [ zeroinitializer, %4 ], !dbg !7
  %v211 = phi [2 x <8 x float>] [ %v1778, %v848 ], [ zeroinitializer, %4 ], !dbg !7
  %v212 = icmp slt i64 %v205, %v191, !dbg !7
  br i1 %v212, label %v213, label %v1780, !dbg !7

v213:                                              ; preds = %v204
  %v214 = shl nsw i64 %v205, 7, !dbg !7
  %v215 = lshr i64 %v205, 12, !dbg !16
  %v216 = add nuw nsw i64 %v215, %v16, !dbg !16
  %v217 = lshr i64 %v205, 5, !dbg !16
  %v218 = and i64 %v217, 127, !dbg !16
  %v219 = shl i64 %v216, 19, !dbg !16
  %v220 = and i64 %v219, 16252928, !dbg !16
  %v221 = shl i64 %v217, 19, !dbg !16
  %v222 = shl i64 %v205, 14, !dbg !16
  %v223 = sub i64 %v222, %v221, !dbg !16
  %v224 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %v220, !dbg !16
  %v225 = getelementptr [2 x i8], ptr addrspace(1) %v224, i64 %v218, !dbg !16
  %v226 = getelementptr [2 x i8], ptr addrspace(1) %v225, i64 %v223, !dbg !16
  %address.thread433 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local434 = zext nneg i32 %address.thread433 to i64, !dbg !8
  %address.local435 = shl nuw nsw i64 %address.local434, 3, !dbg !8
  %address.local436 = and i64 %address.local435, 1536, !dbg !8
  %address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local436, !dbg !8
  %address.local438 = ptrtoint ptr addrspace(3) %address.local437 to i64, !dbg !8
  %address.local439 = trunc i64 %address.local438 to i32, !dbg !8
  %v227 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !16
  %v228 = zext i32 %v227 to i64, !dbg !16
  %v229 = inttoptr i64 %v228 to ptr addrspace(3), !dbg !16
  %address.local440 = lshr i64 %address.local434, 4, !dbg !16
  %address.local441 = and i64 %address.local440, 8, !dbg !16
  %address.local442 = and i64 %address.local434, 123, !dbg !16
  %address.local443 = lshr i64 %address.local434, 2, !dbg !16
  %address.local444 = and i64 %address.local443, 4, !dbg !16
  %address.local445 = add nuw nsw i64 %address.local444, %address.local434, !dbg !16
  %address.local446 = and i64 %address.local445, 4, !dbg !16
  %address.local447 = or disjoint i64 %address.local442, %address.local446, !dbg !16
  %address.local448 = shl nuw nsw i64 %address.local447, 7, !dbg !16
  %address.local449 = or disjoint i64 %address.local441, %address.local448, !dbg !16
  %address.local450 = getelementptr [2 x i8], ptr addrspace(1) %v226, i64 %address.local449, !dbg !16
  %v230 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v231 = ptrtoint ptr addrspace(1) %address.local450 to i64, !dbg !16
  %v232 = sub i64 %v231, %v230, !dbg !16
  %dma.buffer.offset1588 = trunc i64 %v232 to i32, !dbg !16
  %v233 = trunc i64 %v230 to i32, !dbg !16
  %v234 = insertelement <4 x i32> poison, i32 %v233, i64 0, !dbg !16
  %v235 = lshr i64 %v230, 32, !dbg !16
  %v236 = trunc i64 %v235 to i32, !dbg !16
  %v237 = insertelement <4 x i32> %v234, i32 %v236, i64 1, !dbg !16
  %v238 = insertelement <4 x i32> %v237, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1589 = insertelement <4 x i32> %v238, i32 131072, i64 3, !dbg !16
  %v239 = ptrtoint ptr addrspace(3) %v229 to i32, !dbg !16
  %dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v239), !dbg !16
  %v240 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #10, !dbg !16
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !8
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !8
  %address.local453 = and i64 %address.local452, 3584, !dbg !8
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !8
  %address.local455 = ptrtoint ptr addrspace(3) %address.local454 to i64, !dbg !8
  %address.local456 = trunc i64 %address.local455 to i32, !dbg !8
  %v241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !16
  %v242 = zext i32 %v241 to i64, !dbg !16
  %v243 = inttoptr i64 %v242 to ptr addrspace(3), !dbg !16
  %address.local457 = lshr i64 %address.local451, 4, !dbg !16
  %address.local458 = and i64 %address.local457, 24, !dbg !16
  %address.local459 = and i64 %address.local434, 123, !dbg !16
  %address.local460 = lshr i64 %address.local434, 2, !dbg !16
  %address.local461 = and i64 %address.local460, 4, !dbg !16
  %address.local462 = add nuw nsw i64 %address.local461, %address.local434, !dbg !16
  %address.local463 = and i64 %address.local462, 4, !dbg !16
  %address.local464 = or disjoint i64 %address.local459, %address.local463, !dbg !16
  %address.local465 = shl nuw nsw i64 %address.local464, 7, !dbg !16
  %address.local466 = or disjoint i64 %address.local458, %address.local465, !dbg !16
  %address.local467 = getelementptr [2 x i8], ptr addrspace(1) %v226, i64 %address.local466, !dbg !16
  %v244 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v245 = ptrtoint ptr addrspace(1) %address.local467 to i64, !dbg !16
  %v246 = sub i64 %v245, %v244, !dbg !16
  %dma.buffer.offset1585 = trunc i64 %v246 to i32, !dbg !16
  %v247 = trunc i64 %v244 to i32, !dbg !16
  %v248 = insertelement <4 x i32> poison, i32 %v247, i64 0, !dbg !16
  %v249 = lshr i64 %v244, 32, !dbg !16
  %v250 = trunc i64 %v249 to i32, !dbg !16
  %v251 = insertelement <4 x i32> %v248, i32 %v250, i64 1, !dbg !16
  %v252 = insertelement <4 x i32> %v251, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1586 = insertelement <4 x i32> %v252, i32 131072, i64 3, !dbg !16
  %v253 = ptrtoint ptr addrspace(3) %v243 to i32, !dbg !16
  %dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v253), !dbg !16
  %v254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1586) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %v255 = add nuw i64 %v205, 1024, !dbg !16
  %v256 = icmp slt i64 %v255, 0, !dbg !16
  %v257 = sub nuw i64 -1025, %v205, !dbg !16
  %v258 = select i1 %v256, i64 %v257, i64 %v255, !dbg !16
  %v259 = sdiv i64 %v258, 4096, !dbg !16
  %.lobit37 = ashr i64 %v255, 63, !dbg !16
  %v260 = xor i64 %v259, %.lobit37, !dbg !16
  %v261 = add nsw i64 %v260, %v16, !dbg !16
  %v262 = add nsw i64 %v217, %v195, !dbg !16
  %v263 = add nsw i64 %v262, 32, !dbg !16
  %v264 = icmp slt i64 %v262, -32, !dbg !16
  %v265 = sub nsw i64 -33, %v262, !dbg !16
  %v266 = select i1 %v264, i64 %v265, i64 %v263, !dbg !16
  %v267 = sdiv i64 %v266, 128, !dbg !16
  %v268 = sext i1 %v264 to i64, !dbg !16
  %v269 = xor i64 %v267, %v268, !dbg !16
  %v270 = shl nsw i64 %v269, 7, !dbg !16
  %v271 = sub i64 %v195, %v270, !dbg !16
  %v272 = shl i64 %v261, 19, !dbg !16
  %v273 = and i64 %v272, 16252928, !dbg !16
  %v274 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %v273, !dbg !16
  %v275 = getelementptr [2 x i8], ptr addrspace(1) %v274, i64 %v217, !dbg !16
  %v276 = getelementptr [2 x i8], ptr addrspace(1) %v275, i64 %v271, !dbg !16
  %v277 = getelementptr i8, ptr addrspace(1) %v276, i64 64, !dbg !16
  %v278 = getelementptr [2 x i8], ptr addrspace(1) %v277, i64 %v223, !dbg !16
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !8
  %address.local469 = ptrtoint ptr addrspace(3) %address.local468 to i64, !dbg !8
  %address.local470 = trunc i64 %address.local469 to i32, !dbg !8
  %v279 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !16
  %v280 = zext i32 %v279 to i64, !dbg !16
  %v281 = inttoptr i64 %v280 to ptr addrspace(3), !dbg !16
  %address.local471 = getelementptr [2 x i8], ptr addrspace(1) %v278, i64 %address.local449, !dbg !16
  %v282 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v283 = ptrtoint ptr addrspace(1) %address.local471 to i64, !dbg !16
  %v284 = sub i64 %v283, %v282, !dbg !16
  %dma.buffer.offset1582 = trunc i64 %v284 to i32, !dbg !16
  %v285 = trunc i64 %v282 to i32, !dbg !16
  %v286 = insertelement <4 x i32> poison, i32 %v285, i64 0, !dbg !16
  %v287 = lshr i64 %v282, 32, !dbg !16
  %v288 = trunc i64 %v287 to i32, !dbg !16
  %v289 = insertelement <4 x i32> %v286, i32 %v288, i64 1, !dbg !16
  %v290 = insertelement <4 x i32> %v289, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1583 = insertelement <4 x i32> %v290, i32 131072, i64 3, !dbg !16
  %v291 = ptrtoint ptr addrspace(3) %v281 to i32, !dbg !16
  %dma.buffer.lds1584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v291), !dbg !16
  %v292 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1583) #10, !dbg !16
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !8
  %address.local473 = ptrtoint ptr addrspace(3) %address.local472 to i64, !dbg !8
  %address.local474 = trunc i64 %address.local473 to i32, !dbg !8
  %v293 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !16
  %v294 = zext i32 %v293 to i64, !dbg !16
  %v295 = inttoptr i64 %v294 to ptr addrspace(3), !dbg !16
  %address.local475 = getelementptr [2 x i8], ptr addrspace(1) %v278, i64 %address.local466, !dbg !16
  %v296 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v297 = ptrtoint ptr addrspace(1) %address.local475 to i64, !dbg !16
  %v298 = sub i64 %v297, %v296, !dbg !16
  %dma.buffer.offset1579 = trunc i64 %v298 to i32, !dbg !16
  %v299 = trunc i64 %v296 to i32, !dbg !16
  %v300 = insertelement <4 x i32> poison, i32 %v299, i64 0, !dbg !16
  %v301 = lshr i64 %v296, 32, !dbg !16
  %v302 = trunc i64 %v301 to i32, !dbg !16
  %v303 = insertelement <4 x i32> %v300, i32 %v302, i64 1, !dbg !16
  %v304 = insertelement <4 x i32> %v303, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1580 = insertelement <4 x i32> %v304, i32 131072, i64 3, !dbg !16
  %v305 = ptrtoint ptr addrspace(3) %v295 to i32, !dbg !16
  %dma.buffer.lds1581 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v305), !dbg !16
  %v306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1580) #10, !dbg !16
  %address.local476 = mul nuw nsw i64 %address.local434, 160, !dbg !17
  %address.local477 = lshr i64 %address.local434, 2, !dbg !17
  %address.local478 = mul nsw i64 %address.local477, -640, !dbg !17
  %address.local479 = add nsw i64 %address.local476, %address.local478, !dbg !17
  %address.local480 = shl nuw nsw i64 %address.local434, 1, !dbg !17
  %address.local481 = and i64 %address.local480, 24, !dbg !17
  %address.local482 = or disjoint i64 %address.local479, %address.local481, !dbg !17
  %address.local483 = shl i64 %address.local434, 62, !dbg !17
  %address.local484 = ashr exact i64 %address.local483, 57, !dbg !17
  %address.local485 = and i64 %address.local484, -64, !dbg !17
  %address.local486 = add nsw i64 %address.local482, %address.local485, !dbg !17
  %address.local487 = lshr i64 %address.local434, 2, !dbg !11
  %address.local488 = and i64 %address.local487, 12, !dbg !11
  %address.local489 = add nsw i64 %address.local486, %address.local488, !dbg !17
  %address.local490 = and i64 %address.local434, 63, !dbg !11
  %address.local491 = lshr i64 %address.local490, 5, !dbg !11
  %address.local492 = mul nuw nsw i64 %address.local491, 1016, !dbg !11
  %address.local493 = add i64 %address.local489, %address.local492, !dbg !17
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local493, !dbg !17
  %v307 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %v308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v307, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %v309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !17
  %address.local505 = shl nuw nsw i32 %address.thread433, 1, !dbg !17
  %address.local506 = and i32 %address.local505, 24, !dbg !17
  %address.local507 = shl nuw nsw i32 %address.thread433, 8, !dbg !17
  %address.local508 = and i32 %address.local507, 768, !dbg !17
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !17
  %address.local510 = shl nuw nsw i32 %address.thread433, 5, !dbg !17
  %address.local511 = and i32 %address.local510, 96, !dbg !17
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !17
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %v203) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local514 = lshr i32 %address.thread433, 1, !dbg !17
  %address.local515 = and i32 %address.local514, 1, !dbg !17
  %address.local516 = zext nneg i32 %address.local515 to i64, !dbg !17
  %address.local517 = add nsw i64 %address.base, %address.local516, !dbg !17
  %address.local518 = shl i64 %address.local517, 6, !dbg !17
  %address.local519 = sub i64 %address.local513, %address.local518, !dbg !17
  %address.local520 = add i64 %address.local519, %address.local488, !dbg !17
  %address.local521 = add i64 %address.local520, %address.local492, !dbg !17
  %address.local522 = add i64 %address.local521, 32, !dbg !17
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local522, !dbg !17
  %v310 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %v311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v309, <4 x float> %v308) #9, !dbg !17
  %v312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v310, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %v313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !17
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %v314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !17
  %v315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v313, <4 x float> %v312) #9, !dbg !17
  %v316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v314, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %v317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !17
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %v318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !17
  %v319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v317, <4 x float> %v316) #9, !dbg !17
  %v320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v318, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %v321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !17
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %v322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !17
  %v323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v321, <4 x float> %v320) #9, !dbg !17
  %v324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v322, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %v325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !17
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %v326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !17
  %v327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v325, <4 x float> %v324) #9, !dbg !17
  %v328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v326, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %v329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !17
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %v330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !17
  %v331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v329, <4 x float> %v328) #9, !dbg !17
  %v332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v330, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %v333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !17
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %v334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !17
  %v335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v333, <4 x float> %v332) #9, !dbg !17
  %v336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v334, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %v337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !17
  %v338 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %v339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v337, <4 x float> %v336) #9, !dbg !17
  %v340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v338, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %v341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !17
  %v342 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %v343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v341, <4 x float> %v340) #9, !dbg !17
  %v344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v342, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %v345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !17
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %v346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !17
  %v347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v345, <4 x float> %v344) #9, !dbg !17
  %v348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v346, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1443 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %v349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1443, align 8, !dbg !17
  %lds.const.gep1444 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %v350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1444, align 8, !dbg !17
  %v351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v349, <4 x float> %v348) #9, !dbg !17
  %v352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v350, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1445 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %v353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1445, align 8, !dbg !17
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %v354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !17
  %v355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v353, <4 x float> %v352) #9, !dbg !17
  %v356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v354, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %v357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !17
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %v358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !17
  %v359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v357, <4 x float> %v356) #9, !dbg !17
  %v360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v358, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %v361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !17
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %v362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !17
  %v363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v361, <4 x float> %v360) #9, !dbg !17
  %v364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v362, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %v365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !17
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %v366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !17
  %v367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v365, <4 x float> %v364) #9, !dbg !17
  %v368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v366, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %v369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !17
  %v370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v369, <4 x float> %v368) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %v371 = add nuw i64 %v205, 2048, !dbg !16
  %v372 = icmp slt i64 %v371, 0, !dbg !16
  %v373 = sub nuw i64 -2049, %v205, !dbg !16
  %v374 = select i1 %v372, i64 %v373, i64 %v371, !dbg !16
  %v375 = sdiv i64 %v374, 4096, !dbg !16
  %.lobit38 = ashr i64 %v371, 63, !dbg !16
  %v376 = xor i64 %v375, %.lobit38, !dbg !16
  %v377 = add nsw i64 %v376, %v16, !dbg !16
  %v378 = add nsw i64 %v262, 64, !dbg !16
  %v379 = icmp slt i64 %v262, -64, !dbg !16
  %v380 = sub nsw i64 -65, %v262, !dbg !16
  %v381 = select i1 %v379, i64 %v380, i64 %v378, !dbg !16
  %v382 = sdiv i64 %v381, 128, !dbg !16
  %v383 = sext i1 %v379 to i64, !dbg !16
  %v384 = xor i64 %v382, %v383, !dbg !16
  %v385 = shl nsw i64 %v384, 7, !dbg !16
  %v386 = sub i64 %v195, %v385, !dbg !16
  %v387 = shl i64 %v377, 19, !dbg !16
  %v388 = and i64 %v387, 16252928, !dbg !16
  %v389 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %v388, !dbg !16
  %v390 = getelementptr [2 x i8], ptr addrspace(1) %v389, i64 %v217, !dbg !16
  %v391 = getelementptr [2 x i8], ptr addrspace(1) %v390, i64 %v386, !dbg !16
  %v392 = getelementptr i8, ptr addrspace(1) %v391, i64 128, !dbg !16
  %v393 = getelementptr [2 x i8], ptr addrspace(1) %v392, i64 %v223, !dbg !16
  %v394 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !16
  %v395 = zext i32 %v394 to i64, !dbg !16
  %v396 = inttoptr i64 %v395 to ptr addrspace(3), !dbg !16
  %address.local832 = getelementptr [2 x i8], ptr addrspace(1) %v393, i64 %address.local449, !dbg !16
  %v397 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v398 = ptrtoint ptr addrspace(1) %address.local832 to i64, !dbg !16
  %v399 = sub i64 %v398, %v397, !dbg !16
  %dma.buffer.offset1576 = trunc i64 %v399 to i32, !dbg !16
  %v400 = trunc i64 %v397 to i32, !dbg !16
  %v401 = insertelement <4 x i32> poison, i32 %v400, i64 0, !dbg !16
  %v402 = lshr i64 %v397, 32, !dbg !16
  %v403 = trunc i64 %v402 to i32, !dbg !16
  %v404 = insertelement <4 x i32> %v401, i32 %v403, i64 1, !dbg !16
  %v405 = insertelement <4 x i32> %v404, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1577 = insertelement <4 x i32> %v405, i32 131072, i64 3, !dbg !16
  %v406 = ptrtoint ptr addrspace(3) %v396 to i32, !dbg !16
  %dma.buffer.lds1578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v406), !dbg !16
  %v407 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1577) #10, !dbg !16
  %v408 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !16
  %v409 = zext i32 %v408 to i64, !dbg !16
  %v410 = inttoptr i64 %v409 to ptr addrspace(3), !dbg !16
  %address.local833 = getelementptr [2 x i8], ptr addrspace(1) %v393, i64 %address.local466, !dbg !16
  %v411 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v412 = ptrtoint ptr addrspace(1) %address.local833 to i64, !dbg !16
  %v413 = sub i64 %v412, %v411, !dbg !16
  %dma.buffer.offset1573 = trunc i64 %v413 to i32, !dbg !16
  %v414 = trunc i64 %v411 to i32, !dbg !16
  %v415 = insertelement <4 x i32> poison, i32 %v414, i64 0, !dbg !16
  %v416 = lshr i64 %v411, 32, !dbg !16
  %v417 = trunc i64 %v416 to i32, !dbg !16
  %v418 = insertelement <4 x i32> %v415, i32 %v417, i64 1, !dbg !16
  %v419 = insertelement <4 x i32> %v418, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1574 = insertelement <4 x i32> %v419, i32 131072, i64 3, !dbg !16
  %v420 = ptrtoint ptr addrspace(3) %v410 to i32, !dbg !16
  %dma.buffer.lds1575 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v420), !dbg !16
  %v421 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1575, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1574) #10, !dbg !16
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %v422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !17
  %v423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v422, <4 x float> %v311) #9, !dbg !17
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %v424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !17
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %v425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !17
  %v426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v424, <4 x float> %v423) #9, !dbg !17
  %v427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v425, <4 x float> %v315) #9, !dbg !17
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %v428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !17
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %v429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !17
  %v430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v428, <4 x float> %v427) #9, !dbg !17
  %v431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v429, <4 x float> %v319) #9, !dbg !17
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %v432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !17
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %v433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !17
  %v434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v432, <4 x float> %v431) #9, !dbg !17
  %v435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v433, <4 x float> %v323) #9, !dbg !17
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %v436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !17
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %v437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !17
  %v438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v436, <4 x float> %v435) #9, !dbg !17
  %v439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v437, <4 x float> %v327) #9, !dbg !17
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %v440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !17
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %v441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !17
  %v442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v440, <4 x float> %v439) #9, !dbg !17
  %v443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v441, <4 x float> %v331) #9, !dbg !17
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %v444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !17
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %v445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !17
  %v446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v444, <4 x float> %v443) #9, !dbg !17
  %v447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v445, <4 x float> %v335) #9, !dbg !17
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %v448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !17
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %v449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !17
  %v450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v448, <4 x float> %v447) #9, !dbg !17
  %v451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v449, <4 x float> %v339) #9, !dbg !17
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %v452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !17
  %lds.const.gep1470 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %v453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1470, align 8, !dbg !17
  %v454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v452, <4 x float> %v451) #9, !dbg !17
  %v455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v453, <4 x float> %v343) #9, !dbg !17
  %lds.const.gep1471 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %v456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1471, align 8, !dbg !17
  %lds.const.gep1472 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %v457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1472, align 8, !dbg !17
  %v458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v456, <4 x float> %v455) #9, !dbg !17
  %v459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v457, <4 x float> %v347) #9, !dbg !17
  %lds.const.gep1473 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %v460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1473, align 8, !dbg !17
  %lds.const.gep1474 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %v461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1474, align 8, !dbg !17
  %v462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v460, <4 x float> %v459) #9, !dbg !17
  %v463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v461, <4 x float> %v351) #9, !dbg !17
  %lds.const.gep1475 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %v464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1475, align 8, !dbg !17
  %lds.const.gep1476 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %v465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1476, align 8, !dbg !17
  %v466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v464, <4 x float> %v463) #9, !dbg !17
  %v467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v465, <4 x float> %v355) #9, !dbg !17
  %lds.const.gep1477 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %v468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1477, align 8, !dbg !17
  %lds.const.gep1478 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %v469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !17
  %v470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v468, <4 x float> %v467) #9, !dbg !17
  %v471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v469, <4 x float> %v359) #9, !dbg !17
  %lds.const.gep1479 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %v472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !17
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %v473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !17
  %v474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v472, <4 x float> %v471) #9, !dbg !17
  %v475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v473, <4 x float> %v363) #9, !dbg !17
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %v476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !17
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %v477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !17
  %v478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v476, <4 x float> %v475) #9, !dbg !17
  %v479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v477, <4 x float> %v367) #9, !dbg !17
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %v480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !17
  %lds.const.gep1484 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %v481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1484, align 8, !dbg !17
  %v482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v480, <4 x float> %v479) #9, !dbg !17
  %v483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v481, <4 x float> %v370) #9, !dbg !17
  %lds.const.gep1485 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %v484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1485, align 8, !dbg !17
  %v485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v484, <4 x float> %v483) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %v486 = add nuw i64 %v205, 3072, !dbg !16
  %v487 = icmp slt i64 %v486, 0, !dbg !16
  %v488 = sub nuw i64 -3073, %v205, !dbg !16
  %v489 = select i1 %v487, i64 %v488, i64 %v486, !dbg !16
  %v490 = sdiv i64 %v489, 4096, !dbg !16
  %.lobit39 = ashr i64 %v486, 63, !dbg !16
  %v491 = xor i64 %v490, %.lobit39, !dbg !16
  %v492 = add nsw i64 %v491, %v16, !dbg !16
  %v493 = add nsw i64 %v262, 96, !dbg !16
  %v494 = icmp slt i64 %v262, -96, !dbg !16
  %v495 = sub nsw i64 -97, %v262, !dbg !16
  %v496 = select i1 %v494, i64 %v495, i64 %v493, !dbg !16
  %v497 = sdiv i64 %v496, 128, !dbg !16
  %v498 = sext i1 %v494 to i64, !dbg !16
  %v499 = xor i64 %v497, %v498, !dbg !16
  %v500 = shl nsw i64 %v499, 7, !dbg !16
  %v501 = sub i64 %v195, %v500, !dbg !16
  %v502 = shl i64 %v492, 19, !dbg !16
  %v503 = and i64 %v502, 16252928, !dbg !16
  %v504 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %v503, !dbg !16
  %v505 = getelementptr [2 x i8], ptr addrspace(1) %v504, i64 %v217, !dbg !16
  %v506 = getelementptr [2 x i8], ptr addrspace(1) %v505, i64 %v501, !dbg !16
  %v507 = getelementptr i8, ptr addrspace(1) %v506, i64 192, !dbg !16
  %v508 = getelementptr [2 x i8], ptr addrspace(1) %v507, i64 %v223, !dbg !16
  %v509 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !16
  %v510 = zext i32 %v509 to i64, !dbg !16
  %v511 = inttoptr i64 %v510 to ptr addrspace(3), !dbg !16
  %address.local850 = getelementptr [2 x i8], ptr addrspace(1) %v508, i64 %address.local449, !dbg !16
  %v512 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v513 = ptrtoint ptr addrspace(1) %address.local850 to i64, !dbg !16
  %v514 = sub i64 %v513, %v512, !dbg !16
  %dma.buffer.offset1570 = trunc i64 %v514 to i32, !dbg !16
  %v515 = trunc i64 %v512 to i32, !dbg !16
  %v516 = insertelement <4 x i32> poison, i32 %v515, i64 0, !dbg !16
  %v517 = lshr i64 %v512, 32, !dbg !16
  %v518 = trunc i64 %v517 to i32, !dbg !16
  %v519 = insertelement <4 x i32> %v516, i32 %v518, i64 1, !dbg !16
  %v520 = insertelement <4 x i32> %v519, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1571 = insertelement <4 x i32> %v520, i32 131072, i64 3, !dbg !16
  %v521 = ptrtoint ptr addrspace(3) %v511 to i32, !dbg !16
  %dma.buffer.lds1572 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v521), !dbg !16
  %v522 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1572, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1571) #10, !dbg !16
  %v523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !16
  %v524 = zext i32 %v523 to i64, !dbg !16
  %v525 = inttoptr i64 %v524 to ptr addrspace(3), !dbg !16
  %address.local851 = getelementptr [2 x i8], ptr addrspace(1) %v508, i64 %address.local466, !dbg !16
  %v526 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v527 = ptrtoint ptr addrspace(1) %address.local851 to i64, !dbg !16
  %v528 = sub i64 %v527, %v526, !dbg !16
  %dma.buffer.offset1567 = trunc i64 %v528 to i32, !dbg !16
  %v529 = trunc i64 %v526 to i32, !dbg !16
  %v530 = insertelement <4 x i32> poison, i32 %v529, i64 0, !dbg !16
  %v531 = lshr i64 %v526, 32, !dbg !16
  %v532 = trunc i64 %v531 to i32, !dbg !16
  %v533 = insertelement <4 x i32> %v530, i32 %v532, i64 1, !dbg !16
  %v534 = insertelement <4 x i32> %v533, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1568 = insertelement <4 x i32> %v534, i32 131072, i64 3, !dbg !16
  %v535 = ptrtoint ptr addrspace(3) %v525 to i32, !dbg !16
  %dma.buffer.lds1569 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v535), !dbg !16
  %v536 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1569, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1568) #10, !dbg !16
  %v537 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %v538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v537, <4 x float> %v426) #9, !dbg !17
  %lds.const.gep1486 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %v539 = load <4 x half>, ptr addrspace(3) %lds.const.gep1486, align 8, !dbg !17
  %v540 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %v541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v539, <4 x float> %v538) #9, !dbg !17
  %v542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v540, <4 x float> %v430) #9, !dbg !17
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %v543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !17
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %v544 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !17
  %v545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v543, <4 x float> %v542) #9, !dbg !17
  %v546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v544, <4 x float> %v434) #9, !dbg !17
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %v547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !17
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %v548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !17
  %v549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v547, <4 x float> %v546) #9, !dbg !17
  %v550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v548, <4 x float> %v438) #9, !dbg !17
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %v551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !17
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %v552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !17
  %v553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v551, <4 x float> %v550) #9, !dbg !17
  %v554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v552, <4 x float> %v442) #9, !dbg !17
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %v555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !17
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %v556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !17
  %v557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v555, <4 x float> %v554) #9, !dbg !17
  %v558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v556, <4 x float> %v446) #9, !dbg !17
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %v559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !17
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %v560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !17
  %v561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v559, <4 x float> %v558) #9, !dbg !17
  %v562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v560, <4 x float> %v450) #9, !dbg !17
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %v563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !17
  %lds.const.gep1498 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %v564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !17
  %v565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v563, <4 x float> %v562) #9, !dbg !17
  %v566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v564, <4 x float> %v454) #9, !dbg !17
  %lds.const.gep1499 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %v567 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !17
  %v568 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %v569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v567, <4 x float> %v566) #9, !dbg !17
  %v570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v568, <4 x float> %v458) #9, !dbg !17
  %lds.const.gep1500 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %v571 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !17
  %v572 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %v573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v571, <4 x float> %v570) #9, !dbg !17
  %v574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v572, <4 x float> %v462) #9, !dbg !17
  %lds.const.gep1501 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %v575 = load <4 x half>, ptr addrspace(3) %lds.const.gep1501, align 8, !dbg !17
  %lds.const.gep1502 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %v576 = load <4 x half>, ptr addrspace(3) %lds.const.gep1502, align 8, !dbg !17
  %v577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v575, <4 x float> %v574) #9, !dbg !17
  %v578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v576, <4 x float> %v466) #9, !dbg !17
  %lds.const.gep1503 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %v579 = load <4 x half>, ptr addrspace(3) %lds.const.gep1503, align 8, !dbg !17
  %lds.const.gep1504 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %v580 = load <4 x half>, ptr addrspace(3) %lds.const.gep1504, align 8, !dbg !17
  %v581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v579, <4 x float> %v578) #9, !dbg !17
  %v582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v580, <4 x float> %v470) #9, !dbg !17
  %lds.const.gep1505 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %v583 = load <4 x half>, ptr addrspace(3) %lds.const.gep1505, align 8, !dbg !17
  %lds.const.gep1506 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %v584 = load <4 x half>, ptr addrspace(3) %lds.const.gep1506, align 8, !dbg !17
  %v585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v583, <4 x float> %v582) #9, !dbg !17
  %v586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v584, <4 x float> %v474) #9, !dbg !17
  %lds.const.gep1507 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %v587 = load <4 x half>, ptr addrspace(3) %lds.const.gep1507, align 8, !dbg !17
  %lds.const.gep1508 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %v588 = load <4 x half>, ptr addrspace(3) %lds.const.gep1508, align 8, !dbg !17
  %v589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v587, <4 x float> %v586) #9, !dbg !17
  %v590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v588, <4 x float> %v478) #9, !dbg !17
  %lds.const.gep1509 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %v591 = load <4 x half>, ptr addrspace(3) %lds.const.gep1509, align 8, !dbg !17
  %lds.const.gep1510 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %v592 = load <4 x half>, ptr addrspace(3) %lds.const.gep1510, align 8, !dbg !17
  %v593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v591, <4 x float> %v590) #9, !dbg !17
  %v594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v592, <4 x float> %v482) #9, !dbg !17
  %lds.const.gep1511 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %v595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1511, align 8, !dbg !17
  %lds.const.gep1512 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %v596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !17
  %v597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v595, <4 x float> %v594) #9, !dbg !17
  %v598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v596, <4 x float> %v485) #9, !dbg !17
  %lds.const.gep1513 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %v599 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !17
  %v600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v599, <4 x float> %v598) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %v601 = add nuw nsw i64 %v217, %v16, !dbg !18
  %v602 = shl i64 %v601, 19, !dbg !18
  %v603 = and i64 %v602, 16252928, !dbg !18
  %v604 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %v603, !dbg !18
  %v605 = getelementptr [2 x i8], ptr addrspace(1) %v604, i64 %v223, !dbg !18
  %v606 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !18
  %v607 = zext i32 %v606 to i64, !dbg !18
  %v608 = inttoptr i64 %v607 to ptr addrspace(3), !dbg !18
  %address.local852 = shl nuw nsw i64 %address.local434, 5, !dbg !18
  %address.local853 = and i64 %address.local852, 8064, !dbg !18
  %address.local854 = shl nuw nsw i64 %address.local434, 3, !dbg !18
  %address.local855 = and i64 %address.local854, 24, !dbg !18
  %address.local856 = or disjoint i64 %address.local853, %address.local855, !dbg !18
  %address.local857 = getelementptr [2 x i8], ptr addrspace(1) %v605, i64 %address.local856, !dbg !18
  %v609 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v610 = ptrtoint ptr addrspace(1) %address.local857 to i64, !dbg !18
  %v611 = sub i64 %v610, %v609, !dbg !18
  %dma.buffer.offset1564 = trunc i64 %v611 to i32, !dbg !18
  %v612 = trunc i64 %v609 to i32, !dbg !18
  %v613 = insertelement <4 x i32> poison, i32 %v612, i64 0, !dbg !18
  %v614 = lshr i64 %v609, 32, !dbg !18
  %v615 = trunc i64 %v614 to i32, !dbg !18
  %v616 = insertelement <4 x i32> %v613, i32 %v615, i64 1, !dbg !18
  %v617 = insertelement <4 x i32> %v616, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1565 = insertelement <4 x i32> %v617, i32 131072, i64 3, !dbg !18
  %v618 = ptrtoint ptr addrspace(3) %v608 to i32, !dbg !18
  %dma.buffer.lds1566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v618), !dbg !18
  %v619 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1566, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #10, !dbg !18
  %v620 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !18
  %v621 = zext i32 %v620 to i64, !dbg !18
  %v622 = inttoptr i64 %v621 to ptr addrspace(3), !dbg !18
  %address.local858 = shl nuw nsw i64 %address.local451, 5, !dbg !18
  %address.local859 = and i64 %address.local858, 16256, !dbg !18
  %address.local860 = shl nuw nsw i64 %address.local434, 3, !dbg !18
  %address.local861 = and i64 %address.local860, 24, !dbg !18
  %address.local862 = or disjoint i64 %address.local859, %address.local861, !dbg !18
  %address.local863 = getelementptr [2 x i8], ptr addrspace(1) %v605, i64 %address.local862, !dbg !18
  %v623 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v624 = ptrtoint ptr addrspace(1) %address.local863 to i64, !dbg !18
  %v625 = sub i64 %v624, %v623, !dbg !18
  %dma.buffer.offset1561 = trunc i64 %v625 to i32, !dbg !18
  %v626 = trunc i64 %v623 to i32, !dbg !18
  %v627 = insertelement <4 x i32> poison, i32 %v626, i64 0, !dbg !18
  %v628 = lshr i64 %v623, 32, !dbg !18
  %v629 = trunc i64 %v628 to i32, !dbg !18
  %v630 = insertelement <4 x i32> %v627, i32 %v629, i64 1, !dbg !18
  %v631 = insertelement <4 x i32> %v630, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1562 = insertelement <4 x i32> %v631, i32 131072, i64 3, !dbg !18
  %v632 = ptrtoint ptr addrspace(3) %v622 to i32, !dbg !18
  %dma.buffer.lds1563 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v632), !dbg !18
  %v633 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1563, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1562) #10, !dbg !18
  %lds.const.gep1514 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %v634 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !17
  %v635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v634, <4 x float> %v541) #9, !dbg !17
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %v636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !17
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %v637 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !17
  %v638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v636, <4 x float> %v635) #9, !dbg !17
  %v639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v637, <4 x float> %v545) #9, !dbg !17
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %v640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !17
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %v641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !17
  %v642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v640, <4 x float> %v639) #9, !dbg !17
  %v643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v641, <4 x float> %v549) #9, !dbg !17
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %v644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !17
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %v645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !17
  %v646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v644, <4 x float> %v643) #9, !dbg !17
  %v647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v645, <4 x float> %v553) #9, !dbg !17
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %v648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !17
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %v649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !17
  %v650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v648, <4 x float> %v647) #9, !dbg !17
  %v651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v649, <4 x float> %v557) #9, !dbg !17
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %v652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !17
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %v653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !17
  %v654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v652, <4 x float> %v651) #9, !dbg !17
  %v655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v653, <4 x float> %v561) #9, !dbg !17
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %v656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !17
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %v657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !17
  %v658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v656, <4 x float> %v655) #9, !dbg !17
  %v659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v657, <4 x float> %v565) #9, !dbg !17
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %v660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !17
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %v661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !17
  %v662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v660, <4 x float> %v659) #9, !dbg !17
  %v663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v661, <4 x float> %v569) #9, !dbg !17
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %v664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !17
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %v665 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !17
  %v666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v664, <4 x float> %v663) #9, !dbg !17
  %v667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v665, <4 x float> %v573) #9, !dbg !17
  %lds.const.gep1531 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %v668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1531, align 8, !dbg !17
  %lds.const.gep1532 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %v669 = load <4 x half>, ptr addrspace(3) %lds.const.gep1532, align 8, !dbg !17
  %v670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v668, <4 x float> %v667) #9, !dbg !17
  %v671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v669, <4 x float> %v577) #9, !dbg !17
  %lds.const.gep1533 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %v672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1533, align 8, !dbg !17
  %lds.const.gep1534 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %v673 = load <4 x half>, ptr addrspace(3) %lds.const.gep1534, align 8, !dbg !17
  %v674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v672, <4 x float> %v671) #9, !dbg !17
  %v675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v673, <4 x float> %v581) #9, !dbg !17
  %lds.const.gep1535 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %v676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1535, align 8, !dbg !17
  %lds.const.gep1536 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %v677 = load <4 x half>, ptr addrspace(3) %lds.const.gep1536, align 8, !dbg !17
  %v678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v676, <4 x float> %v675) #9, !dbg !17
  %v679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v677, <4 x float> %v585) #9, !dbg !17
  %lds.const.gep1537 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %v680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1537, align 8, !dbg !17
  %lds.const.gep1538 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %v681 = load <4 x half>, ptr addrspace(3) %lds.const.gep1538, align 8, !dbg !17
  %v682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v680, <4 x float> %v679) #9, !dbg !17
  %v683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v681, <4 x float> %v589) #9, !dbg !17
  %lds.const.gep1539 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %v684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1539, align 8, !dbg !17
  %lds.const.gep1540 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %v685 = load <4 x half>, ptr addrspace(3) %lds.const.gep1540, align 8, !dbg !17
  %v686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v684, <4 x float> %v683) #9, !dbg !17
  %v687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v685, <4 x float> %v593) #9, !dbg !17
  %lds.const.gep1541 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %v688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1541, align 8, !dbg !17
  %lds.const.gep1542 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %v689 = load <4 x half>, ptr addrspace(3) %lds.const.gep1542, align 8, !dbg !17
  %v690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v688, <4 x float> %v687) #9, !dbg !17
  %v691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v689, <4 x float> %v597) #9, !dbg !17
  %lds.const.gep1543 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %v692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1543, align 8, !dbg !17
  %lds.const.gep1544 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %v693 = load <4 x half>, ptr addrspace(3) %lds.const.gep1544, align 8, !dbg !17
  %v694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v692, <4 x float> %v691) #9, !dbg !17
  %v695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v693, <4 x float> %v600) #9, !dbg !17
  %lds.const.gep1545 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %v696 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !17
  %v697 = or disjoint i64 %v214, 127, !dbg !13
  %v698 = icmp uge i64 %v64, %v697, !dbg !13
  %v699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v696, <4 x float> %v695) #9, !dbg !17
  %v700 = and i1 %v190, %v698, !dbg !13
  br i1 %v700, label %v701, label %v702, !dbg !13

v701:                                              ; preds = %v213
  br label %v846, !dbg !13

v702:                                              ; preds = %v213
  %v703 = add i64 %v64, %v32, !dbg !13
  %v704 = shl nuw nsw i64 %v65, 4, !dbg !13
  %v705 = add i64 %v703, %v704, !dbg !13
  %v706 = and i32 %v31, 240, !dbg !13
  %index.compact = zext nneg i32 %v706 to i64, !dbg !13
  %v707 = sub i64 %v705, %index.compact, !dbg !13
  %v708 = and i64 %v32, 48, !dbg !13
  %v709 = or disjoint i64 %v214, %v708, !dbg !13
  %v710 = add i64 %v707, 1, !dbg !23
  %v711 = lshr i64 %v710, 32, !dbg !24
  %compare.high = trunc i64 %v711 to i32, !dbg !24
  %v712 = lshr i64 %v709, 32, !dbg !24
  %compare.high1157 = trunc i64 %v712 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !24
  %compare.low.base = trunc i64 %v710 to i32, !dbg !24
  %compare.low.base1158 = trunc i64 %v709 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1158, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %v713 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1159 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result1160 = select i1 %compare.high.equal, i1 %compare.low1159, i1 %compare.high.order, !dbg !24
  %v714 = select i1 %compare.result1160, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1161 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1162 = or i32 %compare.local.base1161, 2, !dbg !24
  %compare.low1163 = icmp ugt i32 %compare.low.base, %compare.local1162, !dbg !24
  %compare.result1164 = select i1 %compare.high.equal, i1 %compare.low1163, i1 %compare.high.order, !dbg !24
  %v715 = select i1 %compare.result1164, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1165 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1166 = or i32 %compare.local.base1165, 3, !dbg !24
  %compare.low1167 = icmp ugt i32 %compare.low.base, %compare.local1166, !dbg !24
  %compare.result1168 = select i1 %compare.high.equal, i1 %compare.low1167, i1 %compare.high.order, !dbg !24
  %v716 = select i1 %compare.result1168, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1169 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1170 = or i32 %compare.local.base1169, 4, !dbg !24
  %compare.low1171 = icmp ugt i32 %compare.low.base, %compare.local1170, !dbg !24
  %compare.result1172 = select i1 %compare.high.equal, i1 %compare.low1171, i1 %compare.high.order, !dbg !24
  %v717 = select i1 %compare.result1172, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1173 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1174 = or i32 %compare.local.base1173, 5, !dbg !24
  %compare.low1175 = icmp ugt i32 %compare.low.base, %compare.local1174, !dbg !24
  %compare.result1176 = select i1 %compare.high.equal, i1 %compare.low1175, i1 %compare.high.order, !dbg !24
  %v718 = select i1 %compare.result1176, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1177 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1178 = or i32 %compare.local.base1177, 6, !dbg !24
  %compare.low1179 = icmp ugt i32 %compare.low.base, %compare.local1178, !dbg !24
  %compare.result1180 = select i1 %compare.high.equal, i1 %compare.low1179, i1 %compare.high.order, !dbg !24
  %v719 = select i1 %compare.result1180, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1181 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1182 = or i32 %compare.local.base1181, 7, !dbg !24
  %compare.low1183 = icmp ugt i32 %compare.low.base, %compare.local1182, !dbg !24
  %compare.result1184 = select i1 %compare.high.equal, i1 %compare.low1183, i1 %compare.high.order, !dbg !24
  %v720 = select i1 %compare.result1184, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1185 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1186 = or i32 %compare.local.base1185, 8, !dbg !24
  %compare.low1187 = icmp ugt i32 %compare.low.base, %compare.local1186, !dbg !24
  %compare.result1188 = select i1 %compare.high.equal, i1 %compare.low1187, i1 %compare.high.order, !dbg !24
  %v721 = select i1 %compare.result1188, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1189 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1190 = or i32 %compare.local.base1189, 9, !dbg !24
  %compare.low1191 = icmp ugt i32 %compare.low.base, %compare.local1190, !dbg !24
  %compare.result1192 = select i1 %compare.high.equal, i1 %compare.low1191, i1 %compare.high.order, !dbg !24
  %v722 = select i1 %compare.result1192, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1193 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1194 = or i32 %compare.local.base1193, 10, !dbg !24
  %compare.low1195 = icmp ugt i32 %compare.low.base, %compare.local1194, !dbg !24
  %compare.result1196 = select i1 %compare.high.equal, i1 %compare.low1195, i1 %compare.high.order, !dbg !24
  %v723 = select i1 %compare.result1196, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1197 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1198 = or i32 %compare.local.base1197, 11, !dbg !24
  %compare.low1199 = icmp ugt i32 %compare.low.base, %compare.local1198, !dbg !24
  %compare.result1200 = select i1 %compare.high.equal, i1 %compare.low1199, i1 %compare.high.order, !dbg !24
  %v724 = select i1 %compare.result1200, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1201 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1202 = or i32 %compare.local.base1201, 12, !dbg !24
  %compare.low1203 = icmp ugt i32 %compare.low.base, %compare.local1202, !dbg !24
  %compare.result1204 = select i1 %compare.high.equal, i1 %compare.low1203, i1 %compare.high.order, !dbg !24
  %v725 = select i1 %compare.result1204, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1205 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1206 = or i32 %compare.local.base1205, 13, !dbg !24
  %compare.low1207 = icmp ugt i32 %compare.low.base, %compare.local1206, !dbg !24
  %compare.result1208 = select i1 %compare.high.equal, i1 %compare.low1207, i1 %compare.high.order, !dbg !24
  %v726 = select i1 %compare.result1208, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1209 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1210 = or i32 %compare.local.base1209, 14, !dbg !24
  %compare.low1211 = icmp ugt i32 %compare.low.base, %compare.local1210, !dbg !24
  %compare.result1212 = select i1 %compare.high.equal, i1 %compare.low1211, i1 %compare.high.order, !dbg !24
  %v727 = select i1 %compare.result1212, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1213 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1214 = or i32 %compare.local.base1213, 15, !dbg !24
  %compare.low1215 = icmp ugt i32 %compare.low.base, %compare.local1214, !dbg !24
  %compare.result1216 = select i1 %compare.high.equal, i1 %compare.low1215, i1 %compare.high.order, !dbg !24
  %v728 = select i1 %compare.result1216, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1217 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1218 = or i32 %compare.local.base1217, 64, !dbg !24
  %compare.low1219 = icmp ugt i32 %compare.low.base, %compare.local1218, !dbg !24
  %compare.result1220 = select i1 %compare.high.equal, i1 %compare.low1219, i1 %compare.high.order, !dbg !24
  %v729 = select i1 %compare.result1220, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1221 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1222 = or i32 %compare.local.base1221, 65, !dbg !24
  %compare.low1223 = icmp ugt i32 %compare.low.base, %compare.local1222, !dbg !24
  %compare.result1224 = select i1 %compare.high.equal, i1 %compare.low1223, i1 %compare.high.order, !dbg !24
  %v730 = select i1 %compare.result1224, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1225 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1226 = or i32 %compare.local.base1225, 66, !dbg !24
  %compare.low1227 = icmp ugt i32 %compare.low.base, %compare.local1226, !dbg !24
  %compare.result1228 = select i1 %compare.high.equal, i1 %compare.low1227, i1 %compare.high.order, !dbg !24
  %v731 = select i1 %compare.result1228, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1229 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1230 = or i32 %compare.local.base1229, 67, !dbg !24
  %compare.low1231 = icmp ugt i32 %compare.low.base, %compare.local1230, !dbg !24
  %compare.result1232 = select i1 %compare.high.equal, i1 %compare.low1231, i1 %compare.high.order, !dbg !24
  %v732 = select i1 %compare.result1232, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1233 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1234 = or i32 %compare.local.base1233, 68, !dbg !24
  %compare.low1235 = icmp ugt i32 %compare.low.base, %compare.local1234, !dbg !24
  %compare.result1236 = select i1 %compare.high.equal, i1 %compare.low1235, i1 %compare.high.order, !dbg !24
  %v733 = select i1 %compare.result1236, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1237 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1238 = or i32 %compare.local.base1237, 69, !dbg !24
  %compare.low1239 = icmp ugt i32 %compare.low.base, %compare.local1238, !dbg !24
  %compare.result1240 = select i1 %compare.high.equal, i1 %compare.low1239, i1 %compare.high.order, !dbg !24
  %v734 = select i1 %compare.result1240, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1241 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1242 = or i32 %compare.local.base1241, 70, !dbg !24
  %compare.low1243 = icmp ugt i32 %compare.low.base, %compare.local1242, !dbg !24
  %compare.result1244 = select i1 %compare.high.equal, i1 %compare.low1243, i1 %compare.high.order, !dbg !24
  %v735 = select i1 %compare.result1244, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1245 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1246 = or i32 %compare.local.base1245, 71, !dbg !24
  %compare.low1247 = icmp ugt i32 %compare.low.base, %compare.local1246, !dbg !24
  %compare.result1248 = select i1 %compare.high.equal, i1 %compare.low1247, i1 %compare.high.order, !dbg !24
  %v736 = select i1 %compare.result1248, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1249 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1250 = or i32 %compare.local.base1249, 72, !dbg !24
  %compare.low1251 = icmp ugt i32 %compare.low.base, %compare.local1250, !dbg !24
  %compare.result1252 = select i1 %compare.high.equal, i1 %compare.low1251, i1 %compare.high.order, !dbg !24
  %v737 = select i1 %compare.result1252, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1253 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1254 = or i32 %compare.local.base1253, 73, !dbg !24
  %compare.low1255 = icmp ugt i32 %compare.low.base, %compare.local1254, !dbg !24
  %compare.result1256 = select i1 %compare.high.equal, i1 %compare.low1255, i1 %compare.high.order, !dbg !24
  %v738 = select i1 %compare.result1256, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1257 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1258 = or i32 %compare.local.base1257, 74, !dbg !24
  %compare.low1259 = icmp ugt i32 %compare.low.base, %compare.local1258, !dbg !24
  %compare.result1260 = select i1 %compare.high.equal, i1 %compare.low1259, i1 %compare.high.order, !dbg !24
  %v739 = select i1 %compare.result1260, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1261 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1262 = or i32 %compare.local.base1261, 75, !dbg !24
  %compare.low1263 = icmp ugt i32 %compare.low.base, %compare.local1262, !dbg !24
  %compare.result1264 = select i1 %compare.high.equal, i1 %compare.low1263, i1 %compare.high.order, !dbg !24
  %v740 = select i1 %compare.result1264, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1265 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1266 = or i32 %compare.local.base1265, 76, !dbg !24
  %compare.low1267 = icmp ugt i32 %compare.low.base, %compare.local1266, !dbg !24
  %compare.result1268 = select i1 %compare.high.equal, i1 %compare.low1267, i1 %compare.high.order, !dbg !24
  %v741 = select i1 %compare.result1268, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1269 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1270 = or i32 %compare.local.base1269, 77, !dbg !24
  %compare.low1271 = icmp ugt i32 %compare.low.base, %compare.local1270, !dbg !24
  %compare.result1272 = select i1 %compare.high.equal, i1 %compare.low1271, i1 %compare.high.order, !dbg !24
  %v742 = select i1 %compare.result1272, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1273 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1274 = or i32 %compare.local.base1273, 78, !dbg !24
  %compare.low1275 = icmp ugt i32 %compare.low.base, %compare.local1274, !dbg !24
  %compare.result1276 = select i1 %compare.high.equal, i1 %compare.low1275, i1 %compare.high.order, !dbg !24
  %v743 = select i1 %compare.result1276, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1277 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1278 = or i32 %compare.local.base1277, 79, !dbg !24
  %compare.low1279 = icmp ugt i32 %compare.low.base, %compare.local1278, !dbg !24
  %compare.result1280 = select i1 %compare.high.equal, i1 %compare.low1279, i1 %compare.high.order, !dbg !24
  %v744 = select i1 %compare.result1280, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %v745 = add i64 %v707, 65, !dbg !23
  %v746 = lshr i64 %v745, 32, !dbg !24
  %compare.high1281 = trunc i64 %v746 to i32, !dbg !24
  %v747 = lshr i64 %v709, 32, !dbg !24
  %compare.high1282 = trunc i64 %v747 to i32, !dbg !24
  %compare.high.equal1283 = icmp eq i32 %compare.high1281, %compare.high1282, !dbg !24
  %compare.high.order1284 = icmp ugt i32 %compare.high1281, %compare.high1282, !dbg !24
  %compare.low.base1285 = trunc i64 %v745 to i32, !dbg !24
  %compare.low.base1286 = trunc i64 %v709 to i32, !dbg !24
  %compare.low1287 = icmp ugt i32 %compare.low.base1285, %compare.low.base1286, !dbg !24
  %compare.result1288 = select i1 %compare.high.equal1283, i1 %compare.low1287, i1 %compare.high.order1284, !dbg !24
  %v748 = select i1 %compare.result1288, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1289 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1290 = or i32 %compare.local.base1289, 1, !dbg !24
  %compare.low1291 = icmp ugt i32 %compare.low.base1285, %compare.local1290, !dbg !24
  %compare.result1292 = select i1 %compare.high.equal1283, i1 %compare.low1291, i1 %compare.high.order1284, !dbg !24
  %v749 = select i1 %compare.result1292, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1294 = or i32 %compare.local.base1293, 2, !dbg !24
  %compare.low1295 = icmp ugt i32 %compare.low.base1285, %compare.local1294, !dbg !24
  %compare.result1296 = select i1 %compare.high.equal1283, i1 %compare.low1295, i1 %compare.high.order1284, !dbg !24
  %v750 = select i1 %compare.result1296, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1298 = or i32 %compare.local.base1297, 3, !dbg !24
  %compare.low1299 = icmp ugt i32 %compare.low.base1285, %compare.local1298, !dbg !24
  %compare.result1300 = select i1 %compare.high.equal1283, i1 %compare.low1299, i1 %compare.high.order1284, !dbg !24
  %v751 = select i1 %compare.result1300, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1302 = or i32 %compare.local.base1301, 4, !dbg !24
  %compare.low1303 = icmp ugt i32 %compare.low.base1285, %compare.local1302, !dbg !24
  %compare.result1304 = select i1 %compare.high.equal1283, i1 %compare.low1303, i1 %compare.high.order1284, !dbg !24
  %v752 = select i1 %compare.result1304, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1306 = or i32 %compare.local.base1305, 5, !dbg !24
  %compare.low1307 = icmp ugt i32 %compare.low.base1285, %compare.local1306, !dbg !24
  %compare.result1308 = select i1 %compare.high.equal1283, i1 %compare.low1307, i1 %compare.high.order1284, !dbg !24
  %v753 = select i1 %compare.result1308, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1310 = or i32 %compare.local.base1309, 6, !dbg !24
  %compare.low1311 = icmp ugt i32 %compare.low.base1285, %compare.local1310, !dbg !24
  %compare.result1312 = select i1 %compare.high.equal1283, i1 %compare.low1311, i1 %compare.high.order1284, !dbg !24
  %v754 = select i1 %compare.result1312, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1314 = or i32 %compare.local.base1313, 7, !dbg !24
  %compare.low1315 = icmp ugt i32 %compare.low.base1285, %compare.local1314, !dbg !24
  %compare.result1316 = select i1 %compare.high.equal1283, i1 %compare.low1315, i1 %compare.high.order1284, !dbg !24
  %v755 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 8, !dbg !24
  %compare.low1319 = icmp ugt i32 %compare.low.base1285, %compare.local1318, !dbg !24
  %compare.result1320 = select i1 %compare.high.equal1283, i1 %compare.low1319, i1 %compare.high.order1284, !dbg !24
  %v756 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 9, !dbg !24
  %compare.low1323 = icmp ugt i32 %compare.low.base1285, %compare.local1322, !dbg !24
  %compare.result1324 = select i1 %compare.high.equal1283, i1 %compare.low1323, i1 %compare.high.order1284, !dbg !24
  %v757 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 10, !dbg !24
  %compare.low1327 = icmp ugt i32 %compare.low.base1285, %compare.local1326, !dbg !24
  %compare.result1328 = select i1 %compare.high.equal1283, i1 %compare.low1327, i1 %compare.high.order1284, !dbg !24
  %v758 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 11, !dbg !24
  %compare.low1331 = icmp ugt i32 %compare.low.base1285, %compare.local1330, !dbg !24
  %compare.result1332 = select i1 %compare.high.equal1283, i1 %compare.low1331, i1 %compare.high.order1284, !dbg !24
  %v759 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 12, !dbg !24
  %compare.low1335 = icmp ugt i32 %compare.low.base1285, %compare.local1334, !dbg !24
  %compare.result1336 = select i1 %compare.high.equal1283, i1 %compare.low1335, i1 %compare.high.order1284, !dbg !24
  %v760 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 13, !dbg !24
  %compare.low1339 = icmp ugt i32 %compare.low.base1285, %compare.local1338, !dbg !24
  %compare.result1340 = select i1 %compare.high.equal1283, i1 %compare.low1339, i1 %compare.high.order1284, !dbg !24
  %v761 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 14, !dbg !24
  %compare.low1343 = icmp ugt i32 %compare.low.base1285, %compare.local1342, !dbg !24
  %compare.result1344 = select i1 %compare.high.equal1283, i1 %compare.low1343, i1 %compare.high.order1284, !dbg !24
  %v762 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 15, !dbg !24
  %compare.low1347 = icmp ugt i32 %compare.low.base1285, %compare.local1346, !dbg !24
  %compare.result1348 = select i1 %compare.high.equal1283, i1 %compare.low1347, i1 %compare.high.order1284, !dbg !24
  %v763 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 64, !dbg !24
  %compare.low1351 = icmp ugt i32 %compare.low.base1285, %compare.local1350, !dbg !24
  %compare.result1352 = select i1 %compare.high.equal1283, i1 %compare.low1351, i1 %compare.high.order1284, !dbg !24
  %v764 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 65, !dbg !24
  %compare.low1355 = icmp ugt i32 %compare.low.base1285, %compare.local1354, !dbg !24
  %compare.result1356 = select i1 %compare.high.equal1283, i1 %compare.low1355, i1 %compare.high.order1284, !dbg !24
  %v765 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 66, !dbg !24
  %compare.low1359 = icmp ugt i32 %compare.low.base1285, %compare.local1358, !dbg !24
  %compare.result1360 = select i1 %compare.high.equal1283, i1 %compare.low1359, i1 %compare.high.order1284, !dbg !24
  %v766 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 67, !dbg !24
  %compare.low1363 = icmp ugt i32 %compare.low.base1285, %compare.local1362, !dbg !24
  %compare.result1364 = select i1 %compare.high.equal1283, i1 %compare.low1363, i1 %compare.high.order1284, !dbg !24
  %v767 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 68, !dbg !24
  %compare.low1367 = icmp ugt i32 %compare.low.base1285, %compare.local1366, !dbg !24
  %compare.result1368 = select i1 %compare.high.equal1283, i1 %compare.low1367, i1 %compare.high.order1284, !dbg !24
  %v768 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 69, !dbg !24
  %compare.low1371 = icmp ugt i32 %compare.low.base1285, %compare.local1370, !dbg !24
  %compare.result1372 = select i1 %compare.high.equal1283, i1 %compare.low1371, i1 %compare.high.order1284, !dbg !24
  %v769 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 70, !dbg !24
  %compare.low1375 = icmp ugt i32 %compare.low.base1285, %compare.local1374, !dbg !24
  %compare.result1376 = select i1 %compare.high.equal1283, i1 %compare.low1375, i1 %compare.high.order1284, !dbg !24
  %v770 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 71, !dbg !24
  %compare.low1379 = icmp ugt i32 %compare.low.base1285, %compare.local1378, !dbg !24
  %compare.result1380 = select i1 %compare.high.equal1283, i1 %compare.low1379, i1 %compare.high.order1284, !dbg !24
  %v771 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 72, !dbg !24
  %compare.low1383 = icmp ugt i32 %compare.low.base1285, %compare.local1382, !dbg !24
  %compare.result1384 = select i1 %compare.high.equal1283, i1 %compare.low1383, i1 %compare.high.order1284, !dbg !24
  %v772 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 73, !dbg !24
  %compare.low1387 = icmp ugt i32 %compare.low.base1285, %compare.local1386, !dbg !24
  %compare.result1388 = select i1 %compare.high.equal1283, i1 %compare.low1387, i1 %compare.high.order1284, !dbg !24
  %v773 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 74, !dbg !24
  %compare.low1391 = icmp ugt i32 %compare.low.base1285, %compare.local1390, !dbg !24
  %compare.result1392 = select i1 %compare.high.equal1283, i1 %compare.low1391, i1 %compare.high.order1284, !dbg !24
  %v774 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1393 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1394 = or i32 %compare.local.base1393, 75, !dbg !24
  %compare.low1395 = icmp ugt i32 %compare.low.base1285, %compare.local1394, !dbg !24
  %compare.result1396 = select i1 %compare.high.equal1283, i1 %compare.low1395, i1 %compare.high.order1284, !dbg !24
  %v775 = select i1 %compare.result1396, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1398 = or i32 %compare.local.base1397, 76, !dbg !24
  %compare.low1399 = icmp ugt i32 %compare.low.base1285, %compare.local1398, !dbg !24
  %compare.result1400 = select i1 %compare.high.equal1283, i1 %compare.low1399, i1 %compare.high.order1284, !dbg !24
  %v776 = select i1 %compare.result1400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1402 = or i32 %compare.local.base1401, 77, !dbg !24
  %compare.low1403 = icmp ugt i32 %compare.low.base1285, %compare.local1402, !dbg !24
  %compare.result1404 = select i1 %compare.high.equal1283, i1 %compare.low1403, i1 %compare.high.order1284, !dbg !24
  %v777 = select i1 %compare.result1404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1406 = or i32 %compare.local.base1405, 78, !dbg !24
  %compare.low1407 = icmp ugt i32 %compare.low.base1285, %compare.local1406, !dbg !24
  %compare.result1408 = select i1 %compare.high.equal1283, i1 %compare.low1407, i1 %compare.high.order1284, !dbg !24
  %v778 = select i1 %compare.result1408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1410 = or i32 %compare.local.base1409, 79, !dbg !24
  %compare.low1411 = icmp ugt i32 %compare.low.base1285, %compare.local1410, !dbg !24
  %compare.result1412 = select i1 %compare.high.equal1283, i1 %compare.low1411, i1 %compare.high.order1284, !dbg !24
  %v779 = select i1 %compare.result1412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %v780 = insertelement <32 x float> poison, float %v713, i64 0, !dbg !13
  %v781 = insertelement <32 x float> %v780, float %v714, i64 1, !dbg !13
  %v782 = insertelement <32 x float> %v781, float %v715, i64 2, !dbg !13
  %v783 = insertelement <32 x float> %v782, float %v716, i64 3, !dbg !13
  %v784 = insertelement <32 x float> %v783, float %v717, i64 4, !dbg !13
  %v785 = insertelement <32 x float> %v784, float %v718, i64 5, !dbg !13
  %v786 = insertelement <32 x float> %v785, float %v719, i64 6, !dbg !13
  %v787 = insertelement <32 x float> %v786, float %v720, i64 7, !dbg !13
  %v788 = insertelement <32 x float> %v787, float %v721, i64 8, !dbg !13
  %v789 = insertelement <32 x float> %v788, float %v722, i64 9, !dbg !13
  %v790 = insertelement <32 x float> %v789, float %v723, i64 10, !dbg !13
  %v791 = insertelement <32 x float> %v790, float %v724, i64 11, !dbg !13
  %v792 = insertelement <32 x float> %v791, float %v725, i64 12, !dbg !13
  %v793 = insertelement <32 x float> %v792, float %v726, i64 13, !dbg !13
  %v794 = insertelement <32 x float> %v793, float %v727, i64 14, !dbg !13
  %v795 = insertelement <32 x float> %v794, float %v728, i64 15, !dbg !13
  %v796 = insertelement <32 x float> %v795, float %v729, i64 16, !dbg !13
  %v797 = insertelement <32 x float> %v796, float %v730, i64 17, !dbg !13
  %v798 = insertelement <32 x float> %v797, float %v731, i64 18, !dbg !13
  %v799 = insertelement <32 x float> %v798, float %v732, i64 19, !dbg !13
  %v800 = insertelement <32 x float> %v799, float %v733, i64 20, !dbg !13
  %v801 = insertelement <32 x float> %v800, float %v734, i64 21, !dbg !13
  %v802 = insertelement <32 x float> %v801, float %v735, i64 22, !dbg !13
  %v803 = insertelement <32 x float> %v802, float %v736, i64 23, !dbg !13
  %v804 = insertelement <32 x float> %v803, float %v737, i64 24, !dbg !13
  %v805 = insertelement <32 x float> %v804, float %v738, i64 25, !dbg !13
  %v806 = insertelement <32 x float> %v805, float %v739, i64 26, !dbg !13
  %v807 = insertelement <32 x float> %v806, float %v740, i64 27, !dbg !13
  %v808 = insertelement <32 x float> %v807, float %v741, i64 28, !dbg !13
  %v809 = insertelement <32 x float> %v808, float %v742, i64 29, !dbg !13
  %v810 = insertelement <32 x float> %v809, float %v743, i64 30, !dbg !13
  %v811 = insertelement <32 x float> %v810, float %v744, i64 31, !dbg !13
  %v812 = insertvalue [2 x <32 x float>] poison, <32 x float> %v811, 0, !dbg !13
  %v813 = insertelement <32 x float> poison, float %v748, i64 0, !dbg !13
  %v814 = insertelement <32 x float> %v813, float %v749, i64 1, !dbg !13
  %v815 = insertelement <32 x float> %v814, float %v750, i64 2, !dbg !13
  %v816 = insertelement <32 x float> %v815, float %v751, i64 3, !dbg !13
  %v817 = insertelement <32 x float> %v816, float %v752, i64 4, !dbg !13
  %v818 = insertelement <32 x float> %v817, float %v753, i64 5, !dbg !13
  %v819 = insertelement <32 x float> %v818, float %v754, i64 6, !dbg !13
  %v820 = insertelement <32 x float> %v819, float %v755, i64 7, !dbg !13
  %v821 = insertelement <32 x float> %v820, float %v756, i64 8, !dbg !13
  %v822 = insertelement <32 x float> %v821, float %v757, i64 9, !dbg !13
  %v823 = insertelement <32 x float> %v822, float %v758, i64 10, !dbg !13
  %v824 = insertelement <32 x float> %v823, float %v759, i64 11, !dbg !13
  %v825 = insertelement <32 x float> %v824, float %v760, i64 12, !dbg !13
  %v826 = insertelement <32 x float> %v825, float %v761, i64 13, !dbg !13
  %v827 = insertelement <32 x float> %v826, float %v762, i64 14, !dbg !13
  %v828 = insertelement <32 x float> %v827, float %v763, i64 15, !dbg !13
  %v829 = insertelement <32 x float> %v828, float %v764, i64 16, !dbg !13
  %v830 = insertelement <32 x float> %v829, float %v765, i64 17, !dbg !13
  %v831 = insertelement <32 x float> %v830, float %v766, i64 18, !dbg !13
  %v832 = insertelement <32 x float> %v831, float %v767, i64 19, !dbg !13
  %v833 = insertelement <32 x float> %v832, float %v768, i64 20, !dbg !13
  %v834 = insertelement <32 x float> %v833, float %v769, i64 21, !dbg !13
  %v835 = insertelement <32 x float> %v834, float %v770, i64 22, !dbg !13
  %v836 = insertelement <32 x float> %v835, float %v771, i64 23, !dbg !13
  %v837 = insertelement <32 x float> %v836, float %v772, i64 24, !dbg !13
  %v838 = insertelement <32 x float> %v837, float %v773, i64 25, !dbg !13
  %v839 = insertelement <32 x float> %v838, float %v774, i64 26, !dbg !13
  %v840 = insertelement <32 x float> %v839, float %v775, i64 27, !dbg !13
  %v841 = insertelement <32 x float> %v840, float %v776, i64 28, !dbg !13
  %v842 = insertelement <32 x float> %v841, float %v777, i64 29, !dbg !13
  %v843 = insertelement <32 x float> %v842, float %v778, i64 30, !dbg !13
  %v844 = insertelement <32 x float> %v843, float %v779, i64 31, !dbg !13
  %v845 = insertvalue [2 x <32 x float>] %v812, <32 x float> %v844, 1, !dbg !13
  %v849 = extractvalue [2 x <32 x float>] %v845, 0, !dbg !26
  %v850 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %v851.masked = fadd <4 x float> %v638, %v850, !dbg !26
  %v856 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %v857.masked = fadd <4 x float> %v642, %v856, !dbg !26
  %v862 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %v863.masked = fadd <4 x float> %v646, %v862, !dbg !26
  %v868 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %v869.masked = fadd <4 x float> %v650, %v868, !dbg !26
  %v874 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %v875.masked = fadd <4 x float> %v654, %v874, !dbg !26
  %v880 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %v881.masked = fadd <4 x float> %v658, %v880, !dbg !26
  %v886 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %v887.masked = fadd <4 x float> %v662, %v886, !dbg !26
  %v892 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %v893.masked = fadd <4 x float> %v666, %v892, !dbg !26
  %v898 = extractvalue [2 x <32 x float>] %v845, 1, !dbg !26
  %v899 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %v900.masked = fadd <4 x float> %v670, %v899, !dbg !26
  %v905 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %v906.masked = fadd <4 x float> %v674, %v905, !dbg !26
  %v911 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %v912.masked = fadd <4 x float> %v678, %v911, !dbg !26
  %v917 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %v918.masked = fadd <4 x float> %v682, %v917, !dbg !26
  %v923 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %v924.masked = fadd <4 x float> %v686, %v923, !dbg !26
  %v929 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %v930.masked = fadd <4 x float> %v690, %v929, !dbg !26
  %v935 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %v936.masked = fadd <4 x float> %v694, %v935, !dbg !26
  %v941 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %v942.masked = fadd <4 x float> %v699, %v941, !dbg !26
  br label %v846, !dbg !13

v846:                                              ; preds = %v701, %v702
  %v851 = phi <4 x float> [ %v851.masked, %v702 ], [ %v638, %v701 ], !dbg !26
  %v857 = phi <4 x float> [ %v857.masked, %v702 ], [ %v642, %v701 ], !dbg !26
  %v863 = phi <4 x float> [ %v863.masked, %v702 ], [ %v646, %v701 ], !dbg !26
  %v869 = phi <4 x float> [ %v869.masked, %v702 ], [ %v650, %v701 ], !dbg !26
  %v875 = phi <4 x float> [ %v875.masked, %v702 ], [ %v654, %v701 ], !dbg !26
  %v881 = phi <4 x float> [ %v881.masked, %v702 ], [ %v658, %v701 ], !dbg !26
  %v887 = phi <4 x float> [ %v887.masked, %v702 ], [ %v662, %v701 ], !dbg !26
  %v893 = phi <4 x float> [ %v893.masked, %v702 ], [ %v666, %v701 ], !dbg !26
  %v900 = phi <4 x float> [ %v900.masked, %v702 ], [ %v670, %v701 ], !dbg !26
  %v906 = phi <4 x float> [ %v906.masked, %v702 ], [ %v674, %v701 ], !dbg !26
  %v912 = phi <4 x float> [ %v912.masked, %v702 ], [ %v678, %v701 ], !dbg !26
  %v918 = phi <4 x float> [ %v918.masked, %v702 ], [ %v682, %v701 ], !dbg !26
  %v924 = phi <4 x float> [ %v924.masked, %v702 ], [ %v686, %v701 ], !dbg !26
  %v930 = phi <4 x float> [ %v930.masked, %v702 ], [ %v690, %v701 ], !dbg !26
  %v936 = phi <4 x float> [ %v936.masked, %v702 ], [ %v694, %v701 ], !dbg !26
  %v942 = phi <4 x float> [ %v942.masked, %v702 ], [ %v699, %v701 ], !dbg !26
  br label %v848, !dbg !13

v848:                                              ; preds = %v846
  %v852 = extractelement <4 x float> %v851, i64 0, !dbg !26
  %v853 = extractelement <4 x float> %v851, i64 1, !dbg !26
  %v854 = extractelement <4 x float> %v851, i64 2, !dbg !26
  %v855 = extractelement <4 x float> %v851, i64 3, !dbg !26
  %v858 = extractelement <4 x float> %v857, i64 0, !dbg !26
  %v859 = extractelement <4 x float> %v857, i64 1, !dbg !26
  %v860 = extractelement <4 x float> %v857, i64 2, !dbg !26
  %v861 = extractelement <4 x float> %v857, i64 3, !dbg !26
  %v864 = extractelement <4 x float> %v863, i64 0, !dbg !26
  %v865 = extractelement <4 x float> %v863, i64 1, !dbg !26
  %v866 = extractelement <4 x float> %v863, i64 2, !dbg !26
  %v867 = extractelement <4 x float> %v863, i64 3, !dbg !26
  %v870 = extractelement <4 x float> %v869, i64 0, !dbg !26
  %v871 = extractelement <4 x float> %v869, i64 1, !dbg !26
  %v872 = extractelement <4 x float> %v869, i64 2, !dbg !26
  %v873 = extractelement <4 x float> %v869, i64 3, !dbg !26
  %v876 = extractelement <4 x float> %v875, i64 0, !dbg !26
  %v877 = extractelement <4 x float> %v875, i64 1, !dbg !26
  %v878 = extractelement <4 x float> %v875, i64 2, !dbg !26
  %v879 = extractelement <4 x float> %v875, i64 3, !dbg !26
  %v882 = extractelement <4 x float> %v881, i64 0, !dbg !26
  %v883 = extractelement <4 x float> %v881, i64 1, !dbg !26
  %v884 = extractelement <4 x float> %v881, i64 2, !dbg !26
  %v885 = extractelement <4 x float> %v881, i64 3, !dbg !26
  %v888 = extractelement <4 x float> %v887, i64 0, !dbg !26
  %v889 = extractelement <4 x float> %v887, i64 1, !dbg !26
  %v890 = extractelement <4 x float> %v887, i64 2, !dbg !26
  %v891 = extractelement <4 x float> %v887, i64 3, !dbg !26
  %v894 = extractelement <4 x float> %v893, i64 0, !dbg !26
  %v895 = extractelement <4 x float> %v893, i64 1, !dbg !26
  %v896 = extractelement <4 x float> %v893, i64 2, !dbg !26
  %v897 = extractelement <4 x float> %v893, i64 3, !dbg !26
  %v901 = extractelement <4 x float> %v900, i64 0, !dbg !26
  %v902 = extractelement <4 x float> %v900, i64 1, !dbg !26
  %v903 = extractelement <4 x float> %v900, i64 2, !dbg !26
  %v904 = extractelement <4 x float> %v900, i64 3, !dbg !26
  %v907 = extractelement <4 x float> %v906, i64 0, !dbg !26
  %v908 = extractelement <4 x float> %v906, i64 1, !dbg !26
  %v909 = extractelement <4 x float> %v906, i64 2, !dbg !26
  %v910 = extractelement <4 x float> %v906, i64 3, !dbg !26
  %v913 = extractelement <4 x float> %v912, i64 0, !dbg !26
  %v914 = extractelement <4 x float> %v912, i64 1, !dbg !26
  %v915 = extractelement <4 x float> %v912, i64 2, !dbg !26
  %v916 = extractelement <4 x float> %v912, i64 3, !dbg !26
  %v919 = extractelement <4 x float> %v918, i64 0, !dbg !26
  %v920 = extractelement <4 x float> %v918, i64 1, !dbg !26
  %v921 = extractelement <4 x float> %v918, i64 2, !dbg !26
  %v922 = extractelement <4 x float> %v918, i64 3, !dbg !26
  %v925 = extractelement <4 x float> %v924, i64 0, !dbg !26
  %v926 = extractelement <4 x float> %v924, i64 1, !dbg !26
  %v927 = extractelement <4 x float> %v924, i64 2, !dbg !26
  %v928 = extractelement <4 x float> %v924, i64 3, !dbg !26
  %v931 = extractelement <4 x float> %v930, i64 0, !dbg !26
  %v932 = extractelement <4 x float> %v930, i64 1, !dbg !26
  %v933 = extractelement <4 x float> %v930, i64 2, !dbg !26
  %v934 = extractelement <4 x float> %v930, i64 3, !dbg !26
  %v937 = extractelement <4 x float> %v936, i64 0, !dbg !26
  %v938 = extractelement <4 x float> %v936, i64 1, !dbg !26
  %v939 = extractelement <4 x float> %v936, i64 2, !dbg !26
  %v940 = extractelement <4 x float> %v936, i64 3, !dbg !26
  %v943 = extractelement <4 x float> %v942, i64 0, !dbg !26
  %v944 = extractelement <4 x float> %v942, i64 1, !dbg !26
  %v945 = extractelement <4 x float> %v942, i64 2, !dbg !26
  %v946 = extractelement <4 x float> %v942, i64 3, !dbg !26
  %local.max.r0.seed = call float @llvm.maxnum.f32(float %v852, float 0xFFF0000000000000), !dbg !27
  %local.max.r0.p4 = call float @llvm.maxnum.f32(float %local.max.r0.seed, float %v858), !dbg !27
  %local.max.r0.p5 = call float @llvm.maxnum.f32(float %v853, float %v859), !dbg !27
  %local.max.r0.p6 = call float @llvm.maxnum.f32(float %v854, float %v860), !dbg !27
  %local.max.r0.p7 = call float @llvm.maxnum.f32(float %v855, float %v861), !dbg !27
  %local.max.r0.p8 = call float @llvm.maxnum.f32(float %local.max.r0.p4, float %v864), !dbg !27
  %local.max.r0.p9 = call float @llvm.maxnum.f32(float %local.max.r0.p5, float %v865), !dbg !27
  %local.max.r0.p10 = call float @llvm.maxnum.f32(float %local.max.r0.p6, float %v866), !dbg !27
  %local.max.r0.p11 = call float @llvm.maxnum.f32(float %local.max.r0.p7, float %v867), !dbg !27
  %local.max.r0.p12 = call float @llvm.maxnum.f32(float %local.max.r0.p8, float %v870), !dbg !27
  %local.max.r0.p13 = call float @llvm.maxnum.f32(float %local.max.r0.p9, float %v871), !dbg !27
  %local.max.r0.p14 = call float @llvm.maxnum.f32(float %local.max.r0.p10, float %v872), !dbg !27
  %local.max.r0.p15 = call float @llvm.maxnum.f32(float %local.max.r0.p11, float %v873), !dbg !27
  %local.max.r0.p16 = call float @llvm.maxnum.f32(float %local.max.r0.p12, float %v876), !dbg !27
  %local.max.r0.p17 = call float @llvm.maxnum.f32(float %local.max.r0.p13, float %v877), !dbg !27
  %local.max.r0.p18 = call float @llvm.maxnum.f32(float %local.max.r0.p14, float %v878), !dbg !27
  %local.max.r0.p19 = call float @llvm.maxnum.f32(float %local.max.r0.p15, float %v879), !dbg !27
  %local.max.r0.p20 = call float @llvm.maxnum.f32(float %local.max.r0.p16, float %v882), !dbg !27
  %local.max.r0.p21 = call float @llvm.maxnum.f32(float %local.max.r0.p17, float %v883), !dbg !27
  %local.max.r0.p22 = call float @llvm.maxnum.f32(float %local.max.r0.p18, float %v884), !dbg !27
  %local.max.r0.p23 = call float @llvm.maxnum.f32(float %local.max.r0.p19, float %v885), !dbg !27
  %local.max.r0.p24 = call float @llvm.maxnum.f32(float %local.max.r0.p20, float %v888), !dbg !27
  %local.max.r0.p25 = call float @llvm.maxnum.f32(float %local.max.r0.p21, float %v889), !dbg !27
  %local.max.r0.p26 = call float @llvm.maxnum.f32(float %local.max.r0.p22, float %v890), !dbg !27
  %local.max.r0.p27 = call float @llvm.maxnum.f32(float %local.max.r0.p23, float %v891), !dbg !27
  %local.max.r0.p28 = call float @llvm.maxnum.f32(float %local.max.r0.p24, float %v894), !dbg !27
  %local.max.r0.p29 = call float @llvm.maxnum.f32(float %local.max.r0.p25, float %v895), !dbg !27
  %local.max.r0.p30 = call float @llvm.maxnum.f32(float %local.max.r0.p26, float %v896), !dbg !27
  %local.max.r0.p31 = call float @llvm.maxnum.f32(float %local.max.r0.p27, float %v897), !dbg !27
  %local.max.r0.merge01 = call float @llvm.maxnum.f32(float %local.max.r0.p28, float %local.max.r0.p29), !dbg !27
  %local.max.r0.merge23 = call float @llvm.maxnum.f32(float %local.max.r0.p30, float %local.max.r0.p31), !dbg !27
  %v978 = call float @llvm.maxnum.f32(float %local.max.r0.merge01, float %local.max.r0.merge23), !dbg !27
  %v979 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %v980 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %v979), !dbg !27
  %v981 = add nuw nsw i32 %v980, 64, !dbg !27
  %v982 = xor i32 %v980, 16, !dbg !27
  %v983 = and i32 %v981, 192, !dbg !27
  %v984 = icmp samesign ult i32 %v982, %v983, !dbg !27
  %v985 = select i1 %v984, i32 %v982, i32 %v980, !dbg !27
  %v986 = shl nuw nsw i32 %v985, 2, !dbg !27
  %v987 = bitcast float %v978 to i32, !dbg !27
  %v988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v987), !dbg !27
  %v989 = bitcast i32 %v988 to float, !dbg !27
  %v990 = call float @llvm.maxnum.f32(float %v978, float %v989), !dbg !27
  %v991 = xor i32 %v980, 32, !dbg !27
  %v992 = icmp samesign ult i32 %v991, %v983, !dbg !27
  %v993 = select i1 %v992, i32 %v991, i32 %v980, !dbg !27
  %v994 = shl nuw nsw i32 %v993, 2, !dbg !27
  %v995 = bitcast float %v990 to i32, !dbg !27
  %v996 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v995), !dbg !27
  %v997 = bitcast i32 %v996 to float, !dbg !27
  %v998 = call float @llvm.maxnum.f32(float %v990, float %v997), !dbg !27
  %v999 = extractvalue [2 x <1 x float>] %v208, 0, !dbg !27
  %v1000 = insertelement <1 x float> poison, float %v998, i64 0, !dbg !27
  %v1001 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %v1000, <1 x float> %v999), !dbg !27
  %local.max.r1.seed = call float @llvm.maxnum.f32(float %v901, float 0xFFF0000000000000), !dbg !27
  %local.max.r1.p4 = call float @llvm.maxnum.f32(float %local.max.r1.seed, float %v907), !dbg !27
  %local.max.r1.p5 = call float @llvm.maxnum.f32(float %v902, float %v908), !dbg !27
  %local.max.r1.p6 = call float @llvm.maxnum.f32(float %v903, float %v909), !dbg !27
  %local.max.r1.p7 = call float @llvm.maxnum.f32(float %v904, float %v910), !dbg !27
  %local.max.r1.p8 = call float @llvm.maxnum.f32(float %local.max.r1.p4, float %v913), !dbg !27
  %local.max.r1.p9 = call float @llvm.maxnum.f32(float %local.max.r1.p5, float %v914), !dbg !27
  %local.max.r1.p10 = call float @llvm.maxnum.f32(float %local.max.r1.p6, float %v915), !dbg !27
  %local.max.r1.p11 = call float @llvm.maxnum.f32(float %local.max.r1.p7, float %v916), !dbg !27
  %local.max.r1.p12 = call float @llvm.maxnum.f32(float %local.max.r1.p8, float %v919), !dbg !27
  %local.max.r1.p13 = call float @llvm.maxnum.f32(float %local.max.r1.p9, float %v920), !dbg !27
  %local.max.r1.p14 = call float @llvm.maxnum.f32(float %local.max.r1.p10, float %v921), !dbg !27
  %local.max.r1.p15 = call float @llvm.maxnum.f32(float %local.max.r1.p11, float %v922), !dbg !27
  %local.max.r1.p16 = call float @llvm.maxnum.f32(float %local.max.r1.p12, float %v925), !dbg !27
  %local.max.r1.p17 = call float @llvm.maxnum.f32(float %local.max.r1.p13, float %v926), !dbg !27
  %local.max.r1.p18 = call float @llvm.maxnum.f32(float %local.max.r1.p14, float %v927), !dbg !27
  %local.max.r1.p19 = call float @llvm.maxnum.f32(float %local.max.r1.p15, float %v928), !dbg !27
  %local.max.r1.p20 = call float @llvm.maxnum.f32(float %local.max.r1.p16, float %v931), !dbg !27
  %local.max.r1.p21 = call float @llvm.maxnum.f32(float %local.max.r1.p17, float %v932), !dbg !27
  %local.max.r1.p22 = call float @llvm.maxnum.f32(float %local.max.r1.p18, float %v933), !dbg !27
  %local.max.r1.p23 = call float @llvm.maxnum.f32(float %local.max.r1.p19, float %v934), !dbg !27
  %local.max.r1.p24 = call float @llvm.maxnum.f32(float %local.max.r1.p20, float %v937), !dbg !27
  %local.max.r1.p25 = call float @llvm.maxnum.f32(float %local.max.r1.p21, float %v938), !dbg !27
  %local.max.r1.p26 = call float @llvm.maxnum.f32(float %local.max.r1.p22, float %v939), !dbg !27
  %local.max.r1.p27 = call float @llvm.maxnum.f32(float %local.max.r1.p23, float %v940), !dbg !27
  %local.max.r1.p28 = call float @llvm.maxnum.f32(float %local.max.r1.p24, float %v943), !dbg !27
  %local.max.r1.p29 = call float @llvm.maxnum.f32(float %local.max.r1.p25, float %v944), !dbg !27
  %local.max.r1.p30 = call float @llvm.maxnum.f32(float %local.max.r1.p26, float %v945), !dbg !27
  %local.max.r1.p31 = call float @llvm.maxnum.f32(float %local.max.r1.p27, float %v946), !dbg !27
  %local.max.r1.merge01 = call float @llvm.maxnum.f32(float %local.max.r1.p28, float %local.max.r1.p29), !dbg !27
  %local.max.r1.merge23 = call float @llvm.maxnum.f32(float %local.max.r1.p30, float %local.max.r1.p31), !dbg !27
  %v1033 = call float @llvm.maxnum.f32(float %local.max.r1.merge01, float %local.max.r1.merge23), !dbg !27
  %v1034 = bitcast float %v1033 to i32, !dbg !27
  %v1035 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v1034), !dbg !27
  %v1036 = bitcast i32 %v1035 to float, !dbg !27
  %v1037 = call float @llvm.maxnum.f32(float %v1033, float %v1036), !dbg !27
  %v1038 = bitcast float %v1037 to i32, !dbg !27
  %v1039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v1038), !dbg !27
  %v1040 = bitcast i32 %v1039 to float, !dbg !27
  %v1041 = call float @llvm.maxnum.f32(float %v1037, float %v1040), !dbg !27
  %v1042 = extractvalue [2 x <1 x float>] %v208, 1, !dbg !27
  %v1043 = insertelement <1 x float> poison, float %v1041, i64 0, !dbg !27
  %v1044 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %v1043, <1 x float> %v1042), !dbg !27
  %v1045 = shufflevector <1 x float> %v1001, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %v1046 = fsub <4 x float> %v851, %v1045, !dbg !28
  %v1047 = fsub <4 x float> %v857, %v1045, !dbg !28
  %v1048 = fsub <4 x float> %v863, %v1045, !dbg !28
  %v1049 = fsub <4 x float> %v869, %v1045, !dbg !28
  %v1050 = fsub <4 x float> %v875, %v1045, !dbg !28
  %v1051 = fsub <4 x float> %v881, %v1045, !dbg !28
  %v1052 = fsub <4 x float> %v887, %v1045, !dbg !28
  %v1053 = fsub <4 x float> %v893, %v1045, !dbg !28
  %v1054 = shufflevector <1 x float> %v1044, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %v1055 = fsub <4 x float> %v900, %v1054, !dbg !28
  %v1056 = fsub <4 x float> %v906, %v1054, !dbg !28
  %v1057 = fsub <4 x float> %v912, %v1054, !dbg !28
  %v1058 = fsub <4 x float> %v918, %v1054, !dbg !28
  %v1059 = fsub <4 x float> %v924, %v1054, !dbg !28
  %v1060 = fsub <4 x float> %v930, %v1054, !dbg !28
  %v1061 = fsub <4 x float> %v936, %v1054, !dbg !28
  %v1062 = fsub <4 x float> %v942, %v1054, !dbg !28
  %v1063 = fsub <1 x float> %v999, %v1001, !dbg !29
  %v1064 = extractelement <1 x float> %v1063, i64 0, !dbg !30
  %v1065 = call float @llvm.amdgcn.exp2.f32(float %v1064), !dbg !30
  %v1066 = insertelement <4 x float> poison, float %v1065, i64 0, !dbg !35
  %v1067 = shufflevector <4 x float> %v1066, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %v1068 = extractvalue [2 x <1 x float>] %v207, 0, !dbg !31
  %v1069 = extractelement <1 x float> %v1068, i64 0, !dbg !31
  %v1070 = fmul float %v1069, %v1065, !dbg !31
  %v1071 = fsub <1 x float> %v1042, %v1044, !dbg !29
  %v1072 = extractelement <1 x float> %v1071, i64 0, !dbg !30
  %v1073 = call float @llvm.amdgcn.exp2.f32(float %v1072), !dbg !30
  %v1074 = insertelement <4 x float> poison, float %v1073, i64 0, !dbg !35
  %v1075 = shufflevector <4 x float> %v1074, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %v1076 = extractvalue [2 x <1 x float>] %v207, 1, !dbg !31
  %v1077 = extractelement <1 x float> %v1076, i64 0, !dbg !31
  %v1078 = fmul float %v1077, %v1073, !dbg !31
  %v1079 = extractelement <4 x float> %v1046, i64 0, !dbg !32
  %v1080 = call float @llvm.amdgcn.exp2.f32(float %v1079), !dbg !32
  %v1081 = insertelement <4 x float> poison, float %v1080, i64 0, !dbg !34
  %v1082 = extractelement <4 x float> %v1046, i64 1, !dbg !32
  %v1083 = call float @llvm.amdgcn.exp2.f32(float %v1082), !dbg !32
  %v1084 = insertelement <4 x float> %v1081, float %v1083, i64 1, !dbg !34
  %v1085 = fadd float %v1080, %v1083, !dbg !33
  %v1086 = extractelement <4 x float> %v1046, i64 2, !dbg !32
  %v1087 = call float @llvm.amdgcn.exp2.f32(float %v1086), !dbg !32
  %v1088 = insertelement <4 x float> %v1084, float %v1087, i64 2, !dbg !34
  %v1089 = fadd float %v1085, %v1087, !dbg !33
  %v1090 = extractelement <4 x float> %v1046, i64 3, !dbg !32
  %v1091 = call float @llvm.amdgcn.exp2.f32(float %v1090), !dbg !32
  %v1092 = insertelement <4 x float> %v1088, float %v1091, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1092) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1093 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1093) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1094 = fadd float %v1089, %v1091, !dbg !33
  %v1095 = extractelement <4 x float> %v1047, i64 0, !dbg !32
  %v1096 = call float @llvm.amdgcn.exp2.f32(float %v1095), !dbg !32
  %v1097 = insertelement <4 x float> poison, float %v1096, i64 0, !dbg !34
  %v1098 = fadd float %v1094, %v1096, !dbg !33
  %v1099 = extractelement <4 x float> %v1047, i64 1, !dbg !32
  %v1100 = call float @llvm.amdgcn.exp2.f32(float %v1099), !dbg !32
  %v1101 = insertelement <4 x float> %v1097, float %v1100, i64 1, !dbg !34
  %v1102 = fadd float %v1098, %v1100, !dbg !33
  %v1103 = extractelement <4 x float> %v1047, i64 2, !dbg !32
  %v1104 = call float @llvm.amdgcn.exp2.f32(float %v1103), !dbg !32
  %v1105 = insertelement <4 x float> %v1101, float %v1104, i64 2, !dbg !34
  %v1106 = fadd float %v1102, %v1104, !dbg !33
  %v1107 = extractelement <4 x float> %v1047, i64 3, !dbg !32
  %v1108 = call float @llvm.amdgcn.exp2.f32(float %v1107), !dbg !32
  %v1109 = insertelement <4 x float> %v1105, float %v1108, i64 3, !dbg !34
  %math.result309 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1109) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1110 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !34
  %math.packed310 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1110) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1111 = fadd float %v1106, %v1108, !dbg !33
  %v1112 = extractelement <4 x float> %v1048, i64 0, !dbg !32
  %v1113 = call float @llvm.amdgcn.exp2.f32(float %v1112), !dbg !32
  %v1114 = insertelement <4 x float> poison, float %v1113, i64 0, !dbg !34
  %v1115 = fadd float %v1111, %v1113, !dbg !33
  %v1116 = extractelement <4 x float> %v1048, i64 1, !dbg !32
  %v1117 = call float @llvm.amdgcn.exp2.f32(float %v1116), !dbg !32
  %v1118 = insertelement <4 x float> %v1114, float %v1117, i64 1, !dbg !34
  %v1119 = fadd float %v1115, %v1117, !dbg !33
  %v1120 = extractelement <4 x float> %v1048, i64 2, !dbg !32
  %v1121 = call float @llvm.amdgcn.exp2.f32(float %v1120), !dbg !32
  %v1122 = insertelement <4 x float> %v1118, float %v1121, i64 2, !dbg !34
  %v1123 = fadd float %v1119, %v1121, !dbg !33
  %v1124 = extractelement <4 x float> %v1048, i64 3, !dbg !32
  %v1125 = call float @llvm.amdgcn.exp2.f32(float %v1124), !dbg !32
  %v1126 = insertelement <4 x float> %v1122, float %v1125, i64 3, !dbg !34
  %math.result311 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1126) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1127 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !34
  %math.packed312 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1127) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1128 = fadd float %v1123, %v1125, !dbg !33
  %v1129 = extractelement <4 x float> %v1049, i64 0, !dbg !32
  %v1130 = call float @llvm.amdgcn.exp2.f32(float %v1129), !dbg !32
  %v1131 = insertelement <4 x float> poison, float %v1130, i64 0, !dbg !34
  %v1132 = fadd float %v1128, %v1130, !dbg !33
  %v1133 = extractelement <4 x float> %v1049, i64 1, !dbg !32
  %v1134 = call float @llvm.amdgcn.exp2.f32(float %v1133), !dbg !32
  %v1135 = insertelement <4 x float> %v1131, float %v1134, i64 1, !dbg !34
  %v1136 = fadd float %v1132, %v1134, !dbg !33
  %v1137 = extractelement <4 x float> %v1049, i64 2, !dbg !32
  %v1138 = call float @llvm.amdgcn.exp2.f32(float %v1137), !dbg !32
  %v1139 = insertelement <4 x float> %v1135, float %v1138, i64 2, !dbg !34
  %v1140 = fadd float %v1136, %v1138, !dbg !33
  %v1141 = extractelement <4 x float> %v1049, i64 3, !dbg !32
  %v1142 = call float @llvm.amdgcn.exp2.f32(float %v1141), !dbg !32
  %v1143 = insertelement <4 x float> %v1139, float %v1142, i64 3, !dbg !34
  %math.result313 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1143) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1144 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !34
  %math.packed314 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1144) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1145 = fadd float %v1140, %v1142, !dbg !33
  %v1146 = extractelement <4 x float> %v1050, i64 0, !dbg !32
  %v1147 = call float @llvm.amdgcn.exp2.f32(float %v1146), !dbg !32
  %v1148 = insertelement <4 x float> poison, float %v1147, i64 0, !dbg !34
  %v1149 = fadd float %v1145, %v1147, !dbg !33
  %v1150 = extractelement <4 x float> %v1050, i64 1, !dbg !32
  %v1151 = call float @llvm.amdgcn.exp2.f32(float %v1150), !dbg !32
  %v1152 = insertelement <4 x float> %v1148, float %v1151, i64 1, !dbg !34
  %v1153 = fadd float %v1149, %v1151, !dbg !33
  %v1154 = extractelement <4 x float> %v1050, i64 2, !dbg !32
  %v1155 = call float @llvm.amdgcn.exp2.f32(float %v1154), !dbg !32
  %v1156 = insertelement <4 x float> %v1152, float %v1155, i64 2, !dbg !34
  %v1157 = fadd float %v1153, %v1155, !dbg !33
  %v1158 = extractelement <4 x float> %v1050, i64 3, !dbg !32
  %v1159 = call float @llvm.amdgcn.exp2.f32(float %v1158), !dbg !32
  %v1160 = insertelement <4 x float> %v1156, float %v1159, i64 3, !dbg !34
  %math.result315 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1160) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1161 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !34
  %math.packed316 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1161) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1162 = fadd float %v1157, %v1159, !dbg !33
  %v1163 = extractelement <4 x float> %v1051, i64 0, !dbg !32
  %v1164 = call float @llvm.amdgcn.exp2.f32(float %v1163), !dbg !32
  %v1165 = insertelement <4 x float> poison, float %v1164, i64 0, !dbg !34
  %v1166 = fadd float %v1162, %v1164, !dbg !33
  %v1167 = extractelement <4 x float> %v1051, i64 1, !dbg !32
  %v1168 = call float @llvm.amdgcn.exp2.f32(float %v1167), !dbg !32
  %v1169 = insertelement <4 x float> %v1165, float %v1168, i64 1, !dbg !34
  %v1170 = fadd float %v1166, %v1168, !dbg !33
  %v1171 = extractelement <4 x float> %v1051, i64 2, !dbg !32
  %v1172 = call float @llvm.amdgcn.exp2.f32(float %v1171), !dbg !32
  %v1173 = insertelement <4 x float> %v1169, float %v1172, i64 2, !dbg !34
  %v1174 = fadd float %v1170, %v1172, !dbg !33
  %v1175 = extractelement <4 x float> %v1051, i64 3, !dbg !32
  %v1176 = call float @llvm.amdgcn.exp2.f32(float %v1175), !dbg !32
  %v1177 = insertelement <4 x float> %v1173, float %v1176, i64 3, !dbg !34
  %math.result317 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1177) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1178 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !34
  %math.packed318 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1178) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1179 = fadd float %v1174, %v1176, !dbg !33
  %v1180 = extractelement <4 x float> %v1052, i64 0, !dbg !32
  %v1181 = call float @llvm.amdgcn.exp2.f32(float %v1180), !dbg !32
  %v1182 = insertelement <4 x float> poison, float %v1181, i64 0, !dbg !34
  %v1183 = fadd float %v1179, %v1181, !dbg !33
  %v1184 = extractelement <4 x float> %v1052, i64 1, !dbg !32
  %v1185 = call float @llvm.amdgcn.exp2.f32(float %v1184), !dbg !32
  %v1186 = insertelement <4 x float> %v1182, float %v1185, i64 1, !dbg !34
  %v1187 = fadd float %v1183, %v1185, !dbg !33
  %v1188 = extractelement <4 x float> %v1052, i64 2, !dbg !32
  %v1189 = call float @llvm.amdgcn.exp2.f32(float %v1188), !dbg !32
  %v1190 = insertelement <4 x float> %v1186, float %v1189, i64 2, !dbg !34
  %v1191 = fadd float %v1187, %v1189, !dbg !33
  %v1192 = extractelement <4 x float> %v1052, i64 3, !dbg !32
  %v1193 = call float @llvm.amdgcn.exp2.f32(float %v1192), !dbg !32
  %v1194 = insertelement <4 x float> %v1190, float %v1193, i64 3, !dbg !34
  %math.result319 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1194) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1195 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !34
  %math.packed320 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1195) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1196 = fadd float %v1191, %v1193, !dbg !33
  %v1197 = extractelement <4 x float> %v1053, i64 0, !dbg !32
  %v1198 = call float @llvm.amdgcn.exp2.f32(float %v1197), !dbg !32
  %v1199 = insertelement <4 x float> poison, float %v1198, i64 0, !dbg !34
  %v1200 = fadd float %v1196, %v1198, !dbg !33
  %v1201 = extractelement <4 x float> %v1053, i64 1, !dbg !32
  %v1202 = call float @llvm.amdgcn.exp2.f32(float %v1201), !dbg !32
  %v1203 = insertelement <4 x float> %v1199, float %v1202, i64 1, !dbg !34
  %v1204 = fadd float %v1200, %v1202, !dbg !33
  %v1205 = extractelement <4 x float> %v1053, i64 2, !dbg !32
  %v1206 = call float @llvm.amdgcn.exp2.f32(float %v1205), !dbg !32
  %v1207 = insertelement <4 x float> %v1203, float %v1206, i64 2, !dbg !34
  %v1208 = fadd float %v1204, %v1206, !dbg !33
  %v1209 = extractelement <4 x float> %v1053, i64 3, !dbg !32
  %v1210 = call float @llvm.amdgcn.exp2.f32(float %v1209), !dbg !32
  %v1211 = insertelement <4 x float> %v1207, float %v1210, i64 3, !dbg !34
  %math.result321 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1211) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1212 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !34
  %math.packed322 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1212) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1213 = fadd float %v1208, %v1210, !dbg !33
  %v1214 = extractelement <4 x float> %v1055, i64 0, !dbg !32
  %v1215 = call float @llvm.amdgcn.exp2.f32(float %v1214), !dbg !32
  %v1216 = insertelement <4 x float> poison, float %v1215, i64 0, !dbg !34
  %v1217 = extractelement <4 x float> %v1055, i64 1, !dbg !32
  %v1218 = call float @llvm.amdgcn.exp2.f32(float %v1217), !dbg !32
  %v1219 = insertelement <4 x float> %v1216, float %v1218, i64 1, !dbg !34
  %v1220 = fadd float %v1215, %v1218, !dbg !33
  %v1221 = extractelement <4 x float> %v1055, i64 2, !dbg !32
  %v1222 = call float @llvm.amdgcn.exp2.f32(float %v1221), !dbg !32
  %v1223 = insertelement <4 x float> %v1219, float %v1222, i64 2, !dbg !34
  %v1224 = fadd float %v1220, %v1222, !dbg !33
  %v1225 = extractelement <4 x float> %v1055, i64 3, !dbg !32
  %v1226 = call float @llvm.amdgcn.exp2.f32(float %v1225), !dbg !32
  %v1227 = insertelement <4 x float> %v1223, float %v1226, i64 3, !dbg !34
  %math.result323 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1227) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1228 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !34
  %math.packed324 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1228) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1229 = fadd float %v1224, %v1226, !dbg !33
  %v1230 = extractelement <4 x float> %v1056, i64 0, !dbg !32
  %v1231 = call float @llvm.amdgcn.exp2.f32(float %v1230), !dbg !32
  %v1232 = insertelement <4 x float> poison, float %v1231, i64 0, !dbg !34
  %v1233 = fadd float %v1229, %v1231, !dbg !33
  %v1234 = extractelement <4 x float> %v1056, i64 1, !dbg !32
  %v1235 = call float @llvm.amdgcn.exp2.f32(float %v1234), !dbg !32
  %v1236 = insertelement <4 x float> %v1232, float %v1235, i64 1, !dbg !34
  %v1237 = fadd float %v1233, %v1235, !dbg !33
  %v1238 = extractelement <4 x float> %v1056, i64 2, !dbg !32
  %v1239 = call float @llvm.amdgcn.exp2.f32(float %v1238), !dbg !32
  %v1240 = insertelement <4 x float> %v1236, float %v1239, i64 2, !dbg !34
  %v1241 = fadd float %v1237, %v1239, !dbg !33
  %v1242 = extractelement <4 x float> %v1056, i64 3, !dbg !32
  %v1243 = call float @llvm.amdgcn.exp2.f32(float %v1242), !dbg !32
  %v1244 = insertelement <4 x float> %v1240, float %v1243, i64 3, !dbg !34
  %math.result325 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1244) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1245 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !34
  %math.packed326 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1245) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1246 = fadd float %v1241, %v1243, !dbg !33
  %v1247 = extractelement <4 x float> %v1057, i64 0, !dbg !32
  %v1248 = call float @llvm.amdgcn.exp2.f32(float %v1247), !dbg !32
  %v1249 = insertelement <4 x float> poison, float %v1248, i64 0, !dbg !34
  %v1250 = fadd float %v1246, %v1248, !dbg !33
  %v1251 = extractelement <4 x float> %v1057, i64 1, !dbg !32
  %v1252 = call float @llvm.amdgcn.exp2.f32(float %v1251), !dbg !32
  %v1253 = insertelement <4 x float> %v1249, float %v1252, i64 1, !dbg !34
  %v1254 = fadd float %v1250, %v1252, !dbg !33
  %v1255 = extractelement <4 x float> %v1057, i64 2, !dbg !32
  %v1256 = call float @llvm.amdgcn.exp2.f32(float %v1255), !dbg !32
  %v1257 = insertelement <4 x float> %v1253, float %v1256, i64 2, !dbg !34
  %v1258 = fadd float %v1254, %v1256, !dbg !33
  %v1259 = extractelement <4 x float> %v1057, i64 3, !dbg !32
  %v1260 = call float @llvm.amdgcn.exp2.f32(float %v1259), !dbg !32
  %v1261 = insertelement <4 x float> %v1257, float %v1260, i64 3, !dbg !34
  %math.result327 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1261) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1262 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !34
  %math.packed328 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1262) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1263 = fadd float %v1258, %v1260, !dbg !33
  %v1264 = extractelement <4 x float> %v1058, i64 0, !dbg !32
  %v1265 = call float @llvm.amdgcn.exp2.f32(float %v1264), !dbg !32
  %v1266 = insertelement <4 x float> poison, float %v1265, i64 0, !dbg !34
  %v1267 = fadd float %v1263, %v1265, !dbg !33
  %v1268 = extractelement <4 x float> %v1058, i64 1, !dbg !32
  %v1269 = call float @llvm.amdgcn.exp2.f32(float %v1268), !dbg !32
  %v1270 = insertelement <4 x float> %v1266, float %v1269, i64 1, !dbg !34
  %v1271 = fadd float %v1267, %v1269, !dbg !33
  %v1272 = extractelement <4 x float> %v1058, i64 2, !dbg !32
  %v1273 = call float @llvm.amdgcn.exp2.f32(float %v1272), !dbg !32
  %v1274 = insertelement <4 x float> %v1270, float %v1273, i64 2, !dbg !34
  %v1275 = fadd float %v1271, %v1273, !dbg !33
  %v1276 = extractelement <4 x float> %v1058, i64 3, !dbg !32
  %v1277 = call float @llvm.amdgcn.exp2.f32(float %v1276), !dbg !32
  %v1278 = insertelement <4 x float> %v1274, float %v1277, i64 3, !dbg !34
  %math.result329 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1278) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1279 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !34
  %math.packed330 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1279) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1280 = fadd float %v1275, %v1277, !dbg !33
  %v1281 = extractelement <4 x float> %v1059, i64 0, !dbg !32
  %v1282 = call float @llvm.amdgcn.exp2.f32(float %v1281), !dbg !32
  %v1283 = insertelement <4 x float> poison, float %v1282, i64 0, !dbg !34
  %v1284 = fadd float %v1280, %v1282, !dbg !33
  %v1285 = extractelement <4 x float> %v1059, i64 1, !dbg !32
  %v1286 = call float @llvm.amdgcn.exp2.f32(float %v1285), !dbg !32
  %v1287 = insertelement <4 x float> %v1283, float %v1286, i64 1, !dbg !34
  %v1288 = fadd float %v1284, %v1286, !dbg !33
  %v1289 = extractelement <4 x float> %v1059, i64 2, !dbg !32
  %v1290 = call float @llvm.amdgcn.exp2.f32(float %v1289), !dbg !32
  %v1291 = insertelement <4 x float> %v1287, float %v1290, i64 2, !dbg !34
  %v1292 = fadd float %v1288, %v1290, !dbg !33
  %v1293 = extractelement <4 x float> %v1059, i64 3, !dbg !32
  %v1294 = call float @llvm.amdgcn.exp2.f32(float %v1293), !dbg !32
  %v1295 = insertelement <4 x float> %v1291, float %v1294, i64 3, !dbg !34
  %math.result331 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1295) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1296 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !34
  %math.packed332 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1296) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1297 = fadd float %v1292, %v1294, !dbg !33
  %v1298 = extractelement <4 x float> %v1060, i64 0, !dbg !32
  %v1299 = call float @llvm.amdgcn.exp2.f32(float %v1298), !dbg !32
  %v1300 = insertelement <4 x float> poison, float %v1299, i64 0, !dbg !34
  %v1301 = fadd float %v1297, %v1299, !dbg !33
  %v1302 = extractelement <4 x float> %v1060, i64 1, !dbg !32
  %v1303 = call float @llvm.amdgcn.exp2.f32(float %v1302), !dbg !32
  %v1304 = insertelement <4 x float> %v1300, float %v1303, i64 1, !dbg !34
  %v1305 = fadd float %v1301, %v1303, !dbg !33
  %v1306 = extractelement <4 x float> %v1060, i64 2, !dbg !32
  %v1307 = call float @llvm.amdgcn.exp2.f32(float %v1306), !dbg !32
  %v1308 = insertelement <4 x float> %v1304, float %v1307, i64 2, !dbg !34
  %v1309 = fadd float %v1305, %v1307, !dbg !33
  %v1310 = extractelement <4 x float> %v1060, i64 3, !dbg !32
  %v1311 = call float @llvm.amdgcn.exp2.f32(float %v1310), !dbg !32
  %v1312 = insertelement <4 x float> %v1308, float %v1311, i64 3, !dbg !34
  %math.result333 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1312) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1313 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !34
  %math.packed334 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1313) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1314 = fadd float %v1309, %v1311, !dbg !33
  %v1315 = extractelement <4 x float> %v1061, i64 0, !dbg !32
  %v1316 = call float @llvm.amdgcn.exp2.f32(float %v1315), !dbg !32
  %v1317 = insertelement <4 x float> poison, float %v1316, i64 0, !dbg !34
  %v1318 = fadd float %v1314, %v1316, !dbg !33
  %v1319 = extractelement <4 x float> %v1061, i64 1, !dbg !32
  %v1320 = call float @llvm.amdgcn.exp2.f32(float %v1319), !dbg !32
  %v1321 = insertelement <4 x float> %v1317, float %v1320, i64 1, !dbg !34
  %v1322 = fadd float %v1318, %v1320, !dbg !33
  %v1323 = extractelement <4 x float> %v1061, i64 2, !dbg !32
  %v1324 = call float @llvm.amdgcn.exp2.f32(float %v1323), !dbg !32
  %v1325 = insertelement <4 x float> %v1321, float %v1324, i64 2, !dbg !34
  %v1326 = fadd float %v1322, %v1324, !dbg !33
  %v1327 = extractelement <4 x float> %v1061, i64 3, !dbg !32
  %v1328 = call float @llvm.amdgcn.exp2.f32(float %v1327), !dbg !32
  %v1329 = insertelement <4 x float> %v1325, float %v1328, i64 3, !dbg !34
  %math.result335 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1329) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1330 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !34
  %math.packed336 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1330) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1331 = fadd float %v1326, %v1328, !dbg !33
  %v1332 = extractelement <4 x float> %v1062, i64 0, !dbg !32
  %v1333 = call float @llvm.amdgcn.exp2.f32(float %v1332), !dbg !32
  %v1334 = insertelement <4 x float> poison, float %v1333, i64 0, !dbg !34
  %v1335 = fadd float %v1331, %v1333, !dbg !33
  %v1336 = extractelement <4 x float> %v1062, i64 1, !dbg !32
  %v1337 = call float @llvm.amdgcn.exp2.f32(float %v1336), !dbg !32
  %v1338 = insertelement <4 x float> %v1334, float %v1337, i64 1, !dbg !34
  %v1339 = fadd float %v1335, %v1337, !dbg !33
  %v1340 = extractelement <4 x float> %v1062, i64 2, !dbg !32
  %v1341 = call float @llvm.amdgcn.exp2.f32(float %v1340), !dbg !32
  %v1342 = insertelement <4 x float> %v1338, float %v1341, i64 2, !dbg !34
  %v1343 = fadd float %v1339, %v1341, !dbg !33
  %v1344 = extractelement <4 x float> %v1062, i64 3, !dbg !32
  %v1345 = call float @llvm.amdgcn.exp2.f32(float %v1344), !dbg !32
  %v1346 = insertelement <4 x float> %v1342, float %v1345, i64 3, !dbg !34
  %math.result337 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1346) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1347 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !34
  %math.packed338 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1347) #9, !dbg !33, !frisk.register_lifetimes !1
  %v1348 = fadd float %v1343, %v1345, !dbg !33
  %v1349 = bitcast float %v1213 to i32, !dbg !33
  %v1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v1349), !dbg !33
  %v1351 = bitcast i32 %v1350 to float, !dbg !33
  %v1352 = fadd float %v1213, %v1351, !dbg !33
  %v1353 = bitcast float %v1352 to i32, !dbg !33
  %v1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v1353), !dbg !33
  %v1355 = bitcast i32 %v1354 to float, !dbg !33
  %v1356 = fadd float %v1352, %v1355, !dbg !33
  %v1357 = fadd float %v1356, %v1070, !dbg !33
  %v1358 = insertelement <1 x float> poison, float %v1357, i64 0, !dbg !7
  %v1359 = insertvalue [2 x <1 x float>] poison, <1 x float> %v1358, 0, !dbg !7
  %v1360 = bitcast float %v1348 to i32, !dbg !33
  %v1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v1360), !dbg !33
  %v1362 = bitcast i32 %v1361 to float, !dbg !33
  %v1363 = fadd float %v1348, %v1362, !dbg !33
  %v1364 = bitcast float %v1363 to i32, !dbg !33
  %v1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v1364), !dbg !33
  %v1366 = bitcast i32 %v1365 to float, !dbg !33
  %v1367 = fadd float %v1363, %v1366, !dbg !33
  %v1368 = fadd float %v1367, %v1078, !dbg !33
  %v1369 = insertelement <1 x float> poison, float %v1368, i64 0, !dbg !7
  %v1370 = insertvalue [2 x <1 x float>] %v1359, <1 x float> %v1369, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread864 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !14, !range !39, !frisk.register_lifetimes !1
  %address.local865 = shl nuw nsw i32 %address.thread864, 3, !dbg !14
  %address.local866 = and i32 %address.local865, 120, !dbg !14
  %address.local867 = shl nuw nsw i32 %address.thread864, 5, !dbg !14
  %address.local868 = and i32 %address.local867, 1536, !dbg !14
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !14
  %address.local870 = zext nneg i32 %address.local869 to i64, !dbg !14
  %address.local871 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !14
  %v1371 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %v1372 = bitcast <4 x i32> %v1371 to <8 x half>, !dbg !14
  %address.local872 = zext nneg i32 %address.thread864 to i64, !dbg !8
  %address.local873 = and i64 %address.local872, 60, !dbg !14
  %address.local874 = add nuw nsw i64 %address.local873, %address.local872, !dbg !14
  %address.local875 = shl nuw nsw i64 %address.local874, 3, !dbg !14
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local877 = and i64 %address.local872, 63, !dbg !11
  %address.local878 = lshr i64 %address.local877, 4, !dbg !14
  %address.local879 = mul nuw nsw i64 %address.local878, 384, !dbg !14
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) %address.local876, i64 %address.local879, !dbg !14
  %address.local881 = lshr i64 %address.local872, 2, !dbg !17
  %address.local882 = mul nsw i64 %address.local881, -32, !dbg !14
  %address.local883 = getelementptr [2 x i8], ptr addrspace(3) %address.local880, i64 %address.local882, !dbg !14
  %address.local884 = getelementptr i8, ptr addrspace(3) %address.local883, i64 256, !dbg !14
  %v1373 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %v1374 = bitcast <4 x i32> %v1373 to <8 x half>, !dbg !14
  %address.local885 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local886 = getelementptr [2 x i8], ptr addrspace(3) %address.local885, i64 %address.local879, !dbg !14
  %address.local887 = getelementptr [2 x i8], ptr addrspace(3) %address.local886, i64 %address.local882, !dbg !14
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local887, i64 512, !dbg !14
  %v1375 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %v1376 = bitcast <4 x i32> %v1375 to <8 x half>, !dbg !14
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local890 = getelementptr [2 x i8], ptr addrspace(3) %address.local889, i64 %address.local879, !dbg !14
  %address.local891 = getelementptr [2 x i8], ptr addrspace(3) %address.local890, i64 %address.local882, !dbg !14
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local891, i64 768, !dbg !14
  %v1377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %v1378 = bitcast <4 x i32> %v1377 to <8 x half>, !dbg !14
  %address.local893 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !14
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local893, i64 4096, !dbg !14
  %v1379 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %v1380 = bitcast <4 x i32> %v1379 to <8 x half>, !dbg !14
  %address.local895 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local896 = getelementptr [2 x i8], ptr addrspace(3) %address.local895, i64 %address.local879, !dbg !14
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) %address.local896, i64 %address.local882, !dbg !14
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local897, i64 4352, !dbg !14
  %v1381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %v1382 = bitcast <4 x i32> %v1381 to <8 x half>, !dbg !14
  %address.local899 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local900 = getelementptr [2 x i8], ptr addrspace(3) %address.local899, i64 %address.local879, !dbg !14
  %address.local901 = getelementptr [2 x i8], ptr addrspace(3) %address.local900, i64 %address.local882, !dbg !14
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local901, i64 4608, !dbg !14
  %v1383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %v1384 = bitcast <4 x i32> %v1383 to <8 x half>, !dbg !14
  %address.local903 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) %address.local903, i64 %address.local879, !dbg !14
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) %address.local904, i64 %address.local882, !dbg !14
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 4864, !dbg !14
  %v1385 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %v1386 = bitcast <4 x i32> %v1385 to <8 x half>, !dbg !14
  %v1387 = add nuw nsw i64 %v217, %v16, !dbg !18
  %v1388 = shl i64 %v1387, 19, !dbg !18
  %v1389 = and i64 %v1388, 16252928, !dbg !18
  %v1390 = add i64 %v1389, %v223, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1391 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %v1390, !dbg !18
  %v1392 = getelementptr i8, ptr addrspace(1) %v1391, i64 64, !dbg !18
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !8
  %address.local908 = and i64 %address.local907, 1536, !dbg !8
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !8
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !8
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !8
  %v1393 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %v1394 = zext i32 %v1393 to i64, !dbg !18
  %v1395 = inttoptr i64 %v1394 to ptr addrspace(3), !dbg !18
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !18
  %address.local913 = and i64 %address.local912, 8064, !dbg !18
  %address.local914 = shl nuw nsw i64 %address.local872, 3, !dbg !18
  %address.local915 = and i64 %address.local914, 24, !dbg !18
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !18
  %address.local917 = getelementptr [2 x i8], ptr addrspace(1) %v1392, i64 %address.local916, !dbg !18
  %v1396 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1397 = ptrtoint ptr addrspace(1) %address.local917 to i64, !dbg !18
  %v1398 = sub i64 %v1397, %v1396, !dbg !18
  %dma.buffer.offset1558 = trunc i64 %v1398 to i32, !dbg !18
  %v1399 = trunc i64 %v1396 to i32, !dbg !18
  %v1400 = insertelement <4 x i32> poison, i32 %v1399, i64 0, !dbg !18
  %v1401 = lshr i64 %v1396, 32, !dbg !18
  %v1402 = trunc i64 %v1401 to i32, !dbg !18
  %v1403 = insertelement <4 x i32> %v1400, i32 %v1402, i64 1, !dbg !18
  %v1404 = insertelement <4 x i32> %v1403, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1559 = insertelement <4 x i32> %v1404, i32 131072, i64 3, !dbg !18
  %v1405 = ptrtoint ptr addrspace(3) %v1395 to i32, !dbg !18
  %dma.buffer.lds1560 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1405), !dbg !18
  %v1406 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1559) #10, !dbg !18
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !8
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !8
  %address.local920 = and i64 %address.local919, 3584, !dbg !8
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !8
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !8
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !8
  %v1407 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %v1408 = zext i32 %v1407 to i64, !dbg !18
  %v1409 = inttoptr i64 %v1408 to ptr addrspace(3), !dbg !18
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !18
  %address.local925 = and i64 %address.local924, 16256, !dbg !18
  %address.local926 = shl nuw nsw i64 %address.local872, 3, !dbg !18
  %address.local927 = and i64 %address.local926, 24, !dbg !18
  %address.local928 = or disjoint i64 %address.local925, %address.local927, !dbg !18
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %v1392, i64 %address.local928, !dbg !18
  %v1410 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1411 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !18
  %v1412 = sub i64 %v1411, %v1410, !dbg !18
  %dma.buffer.offset1555 = trunc i64 %v1412 to i32, !dbg !18
  %v1413 = trunc i64 %v1410 to i32, !dbg !18
  %v1414 = insertelement <4 x i32> poison, i32 %v1413, i64 0, !dbg !18
  %v1415 = lshr i64 %v1410, 32, !dbg !18
  %v1416 = trunc i64 %v1415 to i32, !dbg !18
  %v1417 = insertelement <4 x i32> %v1414, i32 %v1416, i64 1, !dbg !18
  %v1418 = insertelement <4 x i32> %v1417, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1556 = insertelement <4 x i32> %v1418, i32 131072, i64 3, !dbg !18
  %v1419 = ptrtoint ptr addrspace(3) %v1409 to i32, !dbg !18
  %dma.buffer.lds1557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1419), !dbg !18
  %v1420 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1556) #10, !dbg !18
  %v1421 = shufflevector <8 x half> %v1372, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1422 = extractvalue [2 x <8 x float>] %v206, 0, !dbg !35
  %v1423 = shufflevector <8 x float> %v1422, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1424 = fmul <4 x float> %v1423, %v1067, !dbg !35
  %v1425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1421, <4 x float> %v1424) #9, !dbg !14
  %v1426 = shufflevector <8 x half> %v1374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1426, <4 x float> %v1425) #9, !dbg !14
  %v1428 = shufflevector <8 x half> %v1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1428, <4 x float> %v1427) #9, !dbg !14
  %v1430 = shufflevector <8 x half> %v1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1430, <4 x float> %v1429) #9, !dbg !14
  %v1432 = shufflevector <8 x half> %v1380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1432, <4 x float> %v1431) #9, !dbg !14
  %v1434 = shufflevector <8 x half> %v1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1434, <4 x float> %v1433) #9, !dbg !14
  %v1436 = shufflevector <8 x half> %v1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1436, <4 x float> %v1435) #9, !dbg !14
  %v1438 = shufflevector <8 x half> %v1386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1438, <4 x float> %v1437) #9, !dbg !14
  %v1440 = shufflevector <8 x half> %v1372, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1441 = shufflevector <8 x float> %v1422, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1442 = fmul <4 x float> %v1441, %v1067, !dbg !35
  %v1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1440, <4 x float> %v1442) #9, !dbg !14
  %v1444 = shufflevector <8 x half> %v1374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1444, <4 x float> %v1443) #9, !dbg !14
  %v1446 = shufflevector <8 x half> %v1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1446, <4 x float> %v1445) #9, !dbg !14
  %v1448 = shufflevector <8 x half> %v1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1448, <4 x float> %v1447) #9, !dbg !14
  %v1450 = shufflevector <8 x half> %v1380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1450, <4 x float> %v1449) #9, !dbg !14
  %v1452 = shufflevector <8 x half> %v1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1452, <4 x float> %v1451) #9, !dbg !14
  %v1454 = shufflevector <8 x half> %v1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1454, <4 x float> %v1453) #9, !dbg !14
  %v1456 = shufflevector <8 x half> %v1386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1456, <4 x float> %v1455) #9, !dbg !14
  %v1458 = extractvalue [2 x <8 x float>] %v206, 1, !dbg !35
  %v1459 = shufflevector <8 x float> %v1458, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1460 = fmul <4 x float> %v1459, %v1075, !dbg !35
  %v1461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1421, <4 x float> %v1460) #9, !dbg !14
  %v1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1426, <4 x float> %v1461) #9, !dbg !14
  %v1463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1428, <4 x float> %v1462) #9, !dbg !14
  %v1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1430, <4 x float> %v1463) #9, !dbg !14
  %v1465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1432, <4 x float> %v1464) #9, !dbg !14
  %v1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1434, <4 x float> %v1465) #9, !dbg !14
  %v1467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1436, <4 x float> %v1466) #9, !dbg !14
  %v1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1438, <4 x float> %v1467) #9, !dbg !14
  %v1469 = shufflevector <8 x float> %v1458, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1470 = fmul <4 x float> %v1469, %v1075, !dbg !35
  %v1471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1440, <4 x float> %v1470) #9, !dbg !14
  %v1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1444, <4 x float> %v1471) #9, !dbg !14
  %v1473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1446, <4 x float> %v1472) #9, !dbg !14
  %v1474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1448, <4 x float> %v1473) #9, !dbg !14
  %v1475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1450, <4 x float> %v1474) #9, !dbg !14
  %v1476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1452, <4 x float> %v1475) #9, !dbg !14
  %v1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1454, <4 x float> %v1476) #9, !dbg !14
  %v1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1456, <4 x float> %v1477) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1479 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %v1480 = bitcast <4 x i32> %v1479 to <8 x half>, !dbg !14
  %v1481 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %v1482 = bitcast <4 x i32> %v1481 to <8 x half>, !dbg !14
  %v1483 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %v1484 = bitcast <4 x i32> %v1483 to <8 x half>, !dbg !14
  %v1485 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %v1486 = bitcast <4 x i32> %v1485 to <8 x half>, !dbg !14
  %v1487 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %v1488 = bitcast <4 x i32> %v1487 to <8 x half>, !dbg !14
  %v1489 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %v1490 = bitcast <4 x i32> %v1489 to <8 x half>, !dbg !14
  %v1491 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %v1492 = bitcast <4 x i32> %v1491 to <8 x half>, !dbg !14
  %v1493 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %v1494 = bitcast <4 x i32> %v1493 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1495 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %v1390, !dbg !18
  %v1496 = getelementptr i8, ptr addrspace(1) %v1495, i64 128, !dbg !18
  %v1497 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %v1498 = zext i32 %v1497 to i64, !dbg !18
  %v1499 = inttoptr i64 %v1498 to ptr addrspace(3), !dbg !18
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %v1496, i64 %address.local916, !dbg !18
  %v1500 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1501 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !18
  %v1502 = sub i64 %v1501, %v1500, !dbg !18
  %dma.buffer.offset1552 = trunc i64 %v1502 to i32, !dbg !18
  %v1503 = trunc i64 %v1500 to i32, !dbg !18
  %v1504 = insertelement <4 x i32> poison, i32 %v1503, i64 0, !dbg !18
  %v1505 = lshr i64 %v1500, 32, !dbg !18
  %v1506 = trunc i64 %v1505 to i32, !dbg !18
  %v1507 = insertelement <4 x i32> %v1504, i32 %v1506, i64 1, !dbg !18
  %v1508 = insertelement <4 x i32> %v1507, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1553 = insertelement <4 x i32> %v1508, i32 131072, i64 3, !dbg !18
  %v1509 = ptrtoint ptr addrspace(3) %v1499 to i32, !dbg !18
  %dma.buffer.lds1554 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1509), !dbg !18
  %v1510 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1554, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1553) #10, !dbg !18
  %v1511 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %v1512 = zext i32 %v1511 to i64, !dbg !18
  %v1513 = inttoptr i64 %v1512 to ptr addrspace(3), !dbg !18
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %v1496, i64 %address.local928, !dbg !18
  %v1514 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1515 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !18
  %v1516 = sub i64 %v1515, %v1514, !dbg !18
  %dma.buffer.offset1549 = trunc i64 %v1516 to i32, !dbg !18
  %v1517 = trunc i64 %v1514 to i32, !dbg !18
  %v1518 = insertelement <4 x i32> poison, i32 %v1517, i64 0, !dbg !18
  %v1519 = lshr i64 %v1514, 32, !dbg !18
  %v1520 = trunc i64 %v1519 to i32, !dbg !18
  %v1521 = insertelement <4 x i32> %v1518, i32 %v1520, i64 1, !dbg !18
  %v1522 = insertelement <4 x i32> %v1521, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1550 = insertelement <4 x i32> %v1522, i32 131072, i64 3, !dbg !18
  %v1523 = ptrtoint ptr addrspace(3) %v1513 to i32, !dbg !18
  %dma.buffer.lds1551 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1523), !dbg !18
  %v1524 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1551, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1550) #10, !dbg !18
  %v1525 = shufflevector <8 x half> %v1480, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1526 = extractvalue [2 x <8 x float>] %v209, 0, !dbg !35
  %v1527 = shufflevector <8 x float> %v1526, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1528 = fmul <4 x float> %v1527, %v1067, !dbg !35
  %v1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1525, <4 x float> %v1528) #9, !dbg !14
  %v1530 = shufflevector <8 x half> %v1482, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1530, <4 x float> %v1529) #9, !dbg !14
  %v1532 = shufflevector <8 x half> %v1484, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1532, <4 x float> %v1531) #9, !dbg !14
  %v1534 = shufflevector <8 x half> %v1486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1534, <4 x float> %v1533) #9, !dbg !14
  %v1536 = shufflevector <8 x half> %v1488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1536, <4 x float> %v1535) #9, !dbg !14
  %v1538 = shufflevector <8 x half> %v1490, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1538, <4 x float> %v1537) #9, !dbg !14
  %v1540 = shufflevector <8 x half> %v1492, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1540, <4 x float> %v1539) #9, !dbg !14
  %v1542 = shufflevector <8 x half> %v1494, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1542, <4 x float> %v1541) #9, !dbg !14
  %v1544 = shufflevector <8 x half> %v1480, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1545 = shufflevector <8 x float> %v1526, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1546 = fmul <4 x float> %v1545, %v1067, !dbg !35
  %v1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1544, <4 x float> %v1546) #9, !dbg !14
  %v1548 = shufflevector <8 x half> %v1482, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1548, <4 x float> %v1547) #9, !dbg !14
  %v1550 = shufflevector <8 x half> %v1484, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1550, <4 x float> %v1549) #9, !dbg !14
  %v1552 = shufflevector <8 x half> %v1486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1552, <4 x float> %v1551) #9, !dbg !14
  %v1554 = shufflevector <8 x half> %v1488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1554, <4 x float> %v1553) #9, !dbg !14
  %v1556 = shufflevector <8 x half> %v1490, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1556, <4 x float> %v1555) #9, !dbg !14
  %v1558 = shufflevector <8 x half> %v1492, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1558, <4 x float> %v1557) #9, !dbg !14
  %v1560 = shufflevector <8 x half> %v1494, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1560, <4 x float> %v1559) #9, !dbg !14
  %v1562 = extractvalue [2 x <8 x float>] %v209, 1, !dbg !35
  %v1563 = shufflevector <8 x float> %v1562, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1564 = fmul <4 x float> %v1563, %v1075, !dbg !35
  %v1565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1525, <4 x float> %v1564) #9, !dbg !14
  %v1566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1530, <4 x float> %v1565) #9, !dbg !14
  %v1567 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1532, <4 x float> %v1566) #9, !dbg !14
  %v1568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1534, <4 x float> %v1567) #9, !dbg !14
  %v1569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1536, <4 x float> %v1568) #9, !dbg !14
  %v1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1538, <4 x float> %v1569) #9, !dbg !14
  %v1571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1540, <4 x float> %v1570) #9, !dbg !14
  %v1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1542, <4 x float> %v1571) #9, !dbg !14
  %v1573 = shufflevector <8 x float> %v1562, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1574 = fmul <4 x float> %v1573, %v1075, !dbg !35
  %v1575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1544, <4 x float> %v1574) #9, !dbg !14
  %v1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1548, <4 x float> %v1575) #9, !dbg !14
  %v1577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1550, <4 x float> %v1576) #9, !dbg !14
  %v1578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1552, <4 x float> %v1577) #9, !dbg !14
  %v1579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1554, <4 x float> %v1578) #9, !dbg !14
  %v1580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1556, <4 x float> %v1579) #9, !dbg !14
  %v1581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1558, <4 x float> %v1580) #9, !dbg !14
  %v1582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1560, <4 x float> %v1581) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1583 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %v1584 = bitcast <4 x i32> %v1583 to <8 x half>, !dbg !14
  %v1585 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %v1586 = bitcast <4 x i32> %v1585 to <8 x half>, !dbg !14
  %v1587 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %v1588 = bitcast <4 x i32> %v1587 to <8 x half>, !dbg !14
  %v1589 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %v1590 = bitcast <4 x i32> %v1589 to <8 x half>, !dbg !14
  %v1591 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %v1592 = bitcast <4 x i32> %v1591 to <8 x half>, !dbg !14
  %v1593 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %v1594 = bitcast <4 x i32> %v1593 to <8 x half>, !dbg !14
  %v1595 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %v1596 = bitcast <4 x i32> %v1595 to <8 x half>, !dbg !14
  %v1597 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %v1598 = bitcast <4 x i32> %v1597 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1599 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %v1390, !dbg !18
  %v1600 = getelementptr i8, ptr addrspace(1) %v1599, i64 192, !dbg !18
  %v1601 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %v1602 = zext i32 %v1601 to i64, !dbg !18
  %v1603 = inttoptr i64 %v1602 to ptr addrspace(3), !dbg !18
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %v1600, i64 %address.local916, !dbg !18
  %v1604 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1605 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !18
  %v1606 = sub i64 %v1605, %v1604, !dbg !18
  %dma.buffer.offset1546 = trunc i64 %v1606 to i32, !dbg !18
  %v1607 = trunc i64 %v1604 to i32, !dbg !18
  %v1608 = insertelement <4 x i32> poison, i32 %v1607, i64 0, !dbg !18
  %v1609 = lshr i64 %v1604, 32, !dbg !18
  %v1610 = trunc i64 %v1609 to i32, !dbg !18
  %v1611 = insertelement <4 x i32> %v1608, i32 %v1610, i64 1, !dbg !18
  %v1612 = insertelement <4 x i32> %v1611, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1547 = insertelement <4 x i32> %v1612, i32 131072, i64 3, !dbg !18
  %v1613 = ptrtoint ptr addrspace(3) %v1603 to i32, !dbg !18
  %dma.buffer.lds1548 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1613), !dbg !18
  %v1614 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1548, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1547) #10, !dbg !18
  %v1615 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %v1616 = zext i32 %v1615 to i64, !dbg !18
  %v1617 = inttoptr i64 %v1616 to ptr addrspace(3), !dbg !18
  %address.local933 = getelementptr [2 x i8], ptr addrspace(1) %v1600, i64 %address.local928, !dbg !18
  %v1618 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1619 = ptrtoint ptr addrspace(1) %address.local933 to i64, !dbg !18
  %v1620 = sub i64 %v1619, %v1618, !dbg !18
  %dma.buffer.offset = trunc i64 %v1620 to i32, !dbg !18
  %v1621 = trunc i64 %v1618 to i32, !dbg !18
  %v1622 = insertelement <4 x i32> poison, i32 %v1621, i64 0, !dbg !18
  %v1623 = lshr i64 %v1618, 32, !dbg !18
  %v1624 = trunc i64 %v1623 to i32, !dbg !18
  %v1625 = insertelement <4 x i32> %v1622, i32 %v1624, i64 1, !dbg !18
  %v1626 = insertelement <4 x i32> %v1625, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %v1626, i32 131072, i64 3, !dbg !18
  %v1627 = ptrtoint ptr addrspace(3) %v1617 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1627), !dbg !18
  %v1628 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %v1629 = shufflevector <8 x half> %v1584, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1630 = extractvalue [2 x <8 x float>] %v210, 0, !dbg !35
  %v1631 = shufflevector <8 x float> %v1630, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1632 = fmul <4 x float> %v1631, %v1067, !dbg !35
  %v1633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1629, <4 x float> %v1632) #9, !dbg !14
  %v1634 = shufflevector <8 x half> %v1586, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1634, <4 x float> %v1633) #9, !dbg !14
  %v1636 = shufflevector <8 x half> %v1588, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1636, <4 x float> %v1635) #9, !dbg !14
  %v1638 = shufflevector <8 x half> %v1590, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1638, <4 x float> %v1637) #9, !dbg !14
  %v1640 = shufflevector <8 x half> %v1592, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1640, <4 x float> %v1639) #9, !dbg !14
  %v1642 = shufflevector <8 x half> %v1594, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1642, <4 x float> %v1641) #9, !dbg !14
  %v1644 = shufflevector <8 x half> %v1596, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1644, <4 x float> %v1643) #9, !dbg !14
  %v1646 = shufflevector <8 x half> %v1598, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1646, <4 x float> %v1645) #9, !dbg !14
  %v1648 = shufflevector <8 x half> %v1584, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1649 = shufflevector <8 x float> %v1630, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1650 = fmul <4 x float> %v1649, %v1067, !dbg !35
  %v1651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1648, <4 x float> %v1650) #9, !dbg !14
  %v1652 = shufflevector <8 x half> %v1586, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1652, <4 x float> %v1651) #9, !dbg !14
  %v1654 = shufflevector <8 x half> %v1588, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1654, <4 x float> %v1653) #9, !dbg !14
  %v1656 = shufflevector <8 x half> %v1590, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1656, <4 x float> %v1655) #9, !dbg !14
  %v1658 = shufflevector <8 x half> %v1592, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1658, <4 x float> %v1657) #9, !dbg !14
  %v1660 = shufflevector <8 x half> %v1594, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1660, <4 x float> %v1659) #9, !dbg !14
  %v1662 = shufflevector <8 x half> %v1596, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1662, <4 x float> %v1661) #9, !dbg !14
  %v1664 = shufflevector <8 x half> %v1598, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1664, <4 x float> %v1663) #9, !dbg !14
  %v1666 = extractvalue [2 x <8 x float>] %v210, 1, !dbg !35
  %v1667 = shufflevector <8 x float> %v1666, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1668 = fmul <4 x float> %v1667, %v1075, !dbg !35
  %v1669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1629, <4 x float> %v1668) #9, !dbg !14
  %v1670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1634, <4 x float> %v1669) #9, !dbg !14
  %v1671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1636, <4 x float> %v1670) #9, !dbg !14
  %v1672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1638, <4 x float> %v1671) #9, !dbg !14
  %v1673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1640, <4 x float> %v1672) #9, !dbg !14
  %v1674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1642, <4 x float> %v1673) #9, !dbg !14
  %v1675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1644, <4 x float> %v1674) #9, !dbg !14
  %v1676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1646, <4 x float> %v1675) #9, !dbg !14
  %v1677 = shufflevector <8 x float> %v1666, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1678 = fmul <4 x float> %v1677, %v1075, !dbg !35
  %v1679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1648, <4 x float> %v1678) #9, !dbg !14
  %v1680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1652, <4 x float> %v1679) #9, !dbg !14
  %v1681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1654, <4 x float> %v1680) #9, !dbg !14
  %v1682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1656, <4 x float> %v1681) #9, !dbg !14
  %v1683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1658, <4 x float> %v1682) #9, !dbg !14
  %v1684 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1660, <4 x float> %v1683) #9, !dbg !14
  %v1685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1662, <4 x float> %v1684) #9, !dbg !14
  %v1686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1664, <4 x float> %v1685) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1687 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %v1688 = bitcast <4 x i32> %v1687 to <8 x half>, !dbg !14
  %v1689 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %v1690 = bitcast <4 x i32> %v1689 to <8 x half>, !dbg !14
  %v1691 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %v1692 = bitcast <4 x i32> %v1691 to <8 x half>, !dbg !14
  %v1693 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %v1694 = bitcast <4 x i32> %v1693 to <8 x half>, !dbg !14
  %v1695 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %v1696 = bitcast <4 x i32> %v1695 to <8 x half>, !dbg !14
  %v1697 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %v1698 = bitcast <4 x i32> %v1697 to <8 x half>, !dbg !14
  %v1699 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %v1700 = bitcast <4 x i32> %v1699 to <8 x half>, !dbg !14
  %v1701 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %v1702 = bitcast <4 x i32> %v1701 to <8 x half>, !dbg !14
  %v1703 = shufflevector <8 x half> %v1688, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1704 = extractvalue [2 x <8 x float>] %v211, 0, !dbg !35
  %v1705 = shufflevector <8 x float> %v1704, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1706 = fmul <4 x float> %v1705, %v1067, !dbg !35
  %v1707 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1703, <4 x float> %v1706) #9, !dbg !14
  %v1708 = shufflevector <8 x half> %v1690, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1708, <4 x float> %v1707) #9, !dbg !14
  %v1710 = shufflevector <8 x half> %v1692, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1710, <4 x float> %v1709) #9, !dbg !14
  %v1712 = shufflevector <8 x half> %v1694, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1712, <4 x float> %v1711) #9, !dbg !14
  %v1714 = shufflevector <8 x half> %v1696, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1715 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1714, <4 x float> %v1713) #9, !dbg !14
  %v1716 = shufflevector <8 x half> %v1698, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1717 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1716, <4 x float> %v1715) #9, !dbg !14
  %v1718 = shufflevector <8 x half> %v1700, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1719 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1718, <4 x float> %v1717) #9, !dbg !14
  %v1720 = shufflevector <8 x half> %v1702, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1720, <4 x float> %v1719) #9, !dbg !14
  %v1722 = shufflevector <8 x half> %v1688, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1723 = shufflevector <8 x float> %v1704, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1724 = fmul <4 x float> %v1723, %v1067, !dbg !35
  %v1725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1722, <4 x float> %v1724) #9, !dbg !14
  %v1726 = shufflevector <8 x half> %v1690, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1726, <4 x float> %v1725) #9, !dbg !14
  %v1728 = shufflevector <8 x half> %v1692, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1728, <4 x float> %v1727) #9, !dbg !14
  %v1730 = shufflevector <8 x half> %v1694, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1730, <4 x float> %v1729) #9, !dbg !14
  %v1732 = shufflevector <8 x half> %v1696, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1733 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1732, <4 x float> %v1731) #9, !dbg !14
  %v1734 = shufflevector <8 x half> %v1698, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1735 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1734, <4 x float> %v1733) #9, !dbg !14
  %v1736 = shufflevector <8 x half> %v1700, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1736, <4 x float> %v1735) #9, !dbg !14
  %v1738 = shufflevector <8 x half> %v1702, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1738, <4 x float> %v1737) #9, !dbg !14
  %v1740 = extractvalue [2 x <8 x float>] %v211, 1, !dbg !35
  %v1741 = shufflevector <8 x float> %v1740, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1742 = fmul <4 x float> %v1741, %v1075, !dbg !35
  %v1743 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1703, <4 x float> %v1742) #9, !dbg !14
  %v1744 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1708, <4 x float> %v1743) #9, !dbg !14
  %v1745 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1710, <4 x float> %v1744) #9, !dbg !14
  %v1746 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1712, <4 x float> %v1745) #9, !dbg !14
  %v1747 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1714, <4 x float> %v1746) #9, !dbg !14
  %v1748 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1716, <4 x float> %v1747) #9, !dbg !14
  %v1749 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1718, <4 x float> %v1748) #9, !dbg !14
  %v1750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1720, <4 x float> %v1749) #9, !dbg !14
  %v1751 = shufflevector <8 x float> %v1740, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1752 = fmul <4 x float> %v1751, %v1075, !dbg !35
  %v1753 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1722, <4 x float> %v1752) #9, !dbg !14
  %v1754 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1726, <4 x float> %v1753) #9, !dbg !14
  %v1755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1728, <4 x float> %v1754) #9, !dbg !14
  %v1756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1730, <4 x float> %v1755) #9, !dbg !14
  %v1757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1732, <4 x float> %v1756) #9, !dbg !14
  %v1758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1734, <4 x float> %v1757) #9, !dbg !14
  %v1759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1736, <4 x float> %v1758) #9, !dbg !14
  %v1760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1738, <4 x float> %v1759) #9, !dbg !14
  %v1761 = shufflevector <4 x float> %v1439, <4 x float> %v1457, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1762 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1761, 0, !dbg !7
  %v1763 = shufflevector <4 x float> %v1468, <4 x float> %v1478, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1764 = insertvalue [2 x <8 x float>] %v1762, <8 x float> %v1763, 1, !dbg !7
  %v1765 = insertvalue [2 x <1 x float>] poison, <1 x float> %v1001, 0, !dbg !7
  %v1766 = insertvalue [2 x <1 x float>] %v1765, <1 x float> %v1044, 1, !dbg !7
  %v1767 = shufflevector <4 x float> %v1543, <4 x float> %v1561, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1768 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1767, 0, !dbg !7
  %v1769 = shufflevector <4 x float> %v1572, <4 x float> %v1582, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1770 = insertvalue [2 x <8 x float>] %v1768, <8 x float> %v1769, 1, !dbg !7
  %v1771 = shufflevector <4 x float> %v1647, <4 x float> %v1665, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1772 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1771, 0, !dbg !7
  %v1773 = shufflevector <4 x float> %v1676, <4 x float> %v1686, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1774 = insertvalue [2 x <8 x float>] %v1772, <8 x float> %v1773, 1, !dbg !7
  %v1775 = shufflevector <4 x float> %v1721, <4 x float> %v1739, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1776 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1775, 0, !dbg !7
  %v1777 = shufflevector <4 x float> %v1750, <4 x float> %v1760, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1778 = insertvalue [2 x <8 x float>] %v1776, <8 x float> %v1777, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %v1779 = add nuw nsw i64 %v205, 1, !dbg !7
  br label %v204, !dbg !7

v1780:                                             ; preds = %v204
  %v1781 = and i32 %v31, 16, !dbg !7
  %.not = icmp eq i32 %v1781, 0, !dbg !7
  %v1782 = and i32 %v31, 32, !dbg !7
  %.not36 = icmp eq i32 %v1782, 0, !dbg !7
  %v1783 = extractvalue [2 x <8 x float>] %v211, 0, !dbg !7
  %v1784 = extractelement <8 x float> %v1783, i64 0, !dbg !7
  %v1785 = extractelement <8 x float> %v1783, i64 1, !dbg !7
  %v1786 = extractelement <8 x float> %v1783, i64 2, !dbg !7
  %v1787 = extractelement <8 x float> %v1783, i64 3, !dbg !7
  %v1788 = select i1 %.not, float %v1785, float %v1784, !dbg !7
  %v1789 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %v1790 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %v1789), !dbg !7
  %v1791 = add nuw nsw i32 %v1790, 64, !dbg !7
  %v1792 = xor i32 %v1790, 16, !dbg !7
  %v1793 = and i32 %v1791, 192, !dbg !7
  %v1794 = icmp samesign ult i32 %v1792, %v1793, !dbg !7
  %v1795 = select i1 %v1794, i32 %v1792, i32 %v1790, !dbg !7
  %v1796 = shl nuw nsw i32 %v1795, 2, !dbg !7
  %v1797 = bitcast float %v1788 to i32, !dbg !7
  %v1798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1797), !dbg !7
  %v1799 = bitcast i32 %v1798 to float, !dbg !7
  %v1800 = select i1 %.not, float %v1784, float %v1799, !dbg !7
  %v1801 = select i1 %.not, float %v1799, float %v1785, !dbg !7
  %v1802 = select i1 %.not, float %v1787, float %v1786, !dbg !7
  %v1803 = bitcast float %v1802 to i32, !dbg !7
  %v1804 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1803), !dbg !7
  %v1805 = bitcast i32 %v1804 to float, !dbg !7
  %v1806 = select i1 %.not, float %v1786, float %v1805, !dbg !7
  %v1807 = select i1 %.not, float %v1805, float %v1787, !dbg !7
  %v1808 = select i1 %.not36, float %v1806, float %v1800, !dbg !7
  %v1809 = xor i32 %v1790, 32, !dbg !7
  %v1810 = icmp samesign ult i32 %v1809, %v1793, !dbg !7
  %v1811 = select i1 %v1810, i32 %v1809, i32 %v1790, !dbg !7
  %v1812 = shl nuw nsw i32 %v1811, 2, !dbg !7
  %v1813 = bitcast float %v1808 to i32, !dbg !7
  %v1814 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1813), !dbg !7
  %v1815 = bitcast i32 %v1814 to float, !dbg !7
  %v1816 = select i1 %.not36, float %v1800, float %v1815, !dbg !7
  %v1817 = select i1 %.not36, float %v1815, float %v1806, !dbg !7
  %v1818 = select i1 %.not36, float %v1807, float %v1801, !dbg !7
  %v1819 = bitcast float %v1818 to i32, !dbg !7
  %v1820 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1819), !dbg !7
  %v1821 = bitcast i32 %v1820 to float, !dbg !7
  %v1822 = select i1 %.not36, float %v1801, float %v1821, !dbg !7
  %v1823 = select i1 %.not36, float %v1821, float %v1807, !dbg !7
  %v1824 = extractvalue [2 x <8 x float>] %v210, 0, !dbg !7
  %v1825 = extractelement <8 x float> %v1824, i64 0, !dbg !7
  %v1826 = extractelement <8 x float> %v1824, i64 1, !dbg !7
  %v1827 = extractelement <8 x float> %v1824, i64 2, !dbg !7
  %v1828 = extractelement <8 x float> %v1824, i64 3, !dbg !7
  %v1829 = select i1 %.not, float %v1826, float %v1825, !dbg !7
  %v1830 = bitcast float %v1829 to i32, !dbg !7
  %v1831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1830), !dbg !7
  %v1832 = bitcast i32 %v1831 to float, !dbg !7
  %v1833 = select i1 %.not, float %v1825, float %v1832, !dbg !7
  %v1834 = select i1 %.not, float %v1832, float %v1826, !dbg !7
  %v1835 = select i1 %.not, float %v1828, float %v1827, !dbg !7
  %v1836 = bitcast float %v1835 to i32, !dbg !7
  %v1837 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1836), !dbg !7
  %v1838 = bitcast i32 %v1837 to float, !dbg !7
  %v1839 = select i1 %.not, float %v1827, float %v1838, !dbg !7
  %v1840 = select i1 %.not, float %v1838, float %v1828, !dbg !7
  %v1841 = select i1 %.not36, float %v1839, float %v1833, !dbg !7
  %v1842 = bitcast float %v1841 to i32, !dbg !7
  %v1843 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1842), !dbg !7
  %v1844 = bitcast i32 %v1843 to float, !dbg !7
  %v1845 = select i1 %.not36, float %v1833, float %v1844, !dbg !7
  %v1846 = select i1 %.not36, float %v1844, float %v1839, !dbg !7
  %v1847 = select i1 %.not36, float %v1840, float %v1834, !dbg !7
  %v1848 = bitcast float %v1847 to i32, !dbg !7
  %v1849 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1848), !dbg !7
  %v1850 = bitcast i32 %v1849 to float, !dbg !7
  %v1851 = select i1 %.not36, float %v1834, float %v1850, !dbg !7
  %v1852 = select i1 %.not36, float %v1850, float %v1840, !dbg !7
  %v1853 = extractvalue [2 x <8 x float>] %v209, 0, !dbg !7
  %v1854 = extractelement <8 x float> %v1853, i64 0, !dbg !7
  %v1855 = extractelement <8 x float> %v1853, i64 1, !dbg !7
  %v1856 = extractelement <8 x float> %v1853, i64 2, !dbg !7
  %v1857 = extractelement <8 x float> %v1853, i64 3, !dbg !7
  %v1858 = select i1 %.not, float %v1855, float %v1854, !dbg !7
  %v1859 = bitcast float %v1858 to i32, !dbg !7
  %v1860 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1859), !dbg !7
  %v1861 = bitcast i32 %v1860 to float, !dbg !7
  %v1862 = select i1 %.not, float %v1854, float %v1861, !dbg !7
  %v1863 = select i1 %.not, float %v1861, float %v1855, !dbg !7
  %v1864 = select i1 %.not, float %v1857, float %v1856, !dbg !7
  %v1865 = bitcast float %v1864 to i32, !dbg !7
  %v1866 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1865), !dbg !7
  %v1867 = bitcast i32 %v1866 to float, !dbg !7
  %v1868 = select i1 %.not, float %v1856, float %v1867, !dbg !7
  %v1869 = select i1 %.not, float %v1867, float %v1857, !dbg !7
  %v1870 = select i1 %.not36, float %v1868, float %v1862, !dbg !7
  %v1871 = bitcast float %v1870 to i32, !dbg !7
  %v1872 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1871), !dbg !7
  %v1873 = bitcast i32 %v1872 to float, !dbg !7
  %v1874 = select i1 %.not36, float %v1862, float %v1873, !dbg !7
  %v1875 = select i1 %.not36, float %v1873, float %v1868, !dbg !7
  %v1876 = select i1 %.not36, float %v1869, float %v1863, !dbg !7
  %v1877 = bitcast float %v1876 to i32, !dbg !7
  %v1878 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1877), !dbg !7
  %v1879 = bitcast i32 %v1878 to float, !dbg !7
  %v1880 = select i1 %.not36, float %v1863, float %v1879, !dbg !7
  %v1881 = select i1 %.not36, float %v1879, float %v1869, !dbg !7
  %v1882 = extractvalue [2 x <8 x float>] %v206, 0, !dbg !7
  %v1883 = extractelement <8 x float> %v1882, i64 0, !dbg !7
  %v1884 = extractelement <8 x float> %v1882, i64 1, !dbg !7
  %v1885 = extractelement <8 x float> %v1882, i64 2, !dbg !7
  %v1886 = extractelement <8 x float> %v1882, i64 3, !dbg !7
  %v1887 = select i1 %.not, float %v1884, float %v1883, !dbg !7
  %v1888 = bitcast float %v1887 to i32, !dbg !7
  %v1889 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1888), !dbg !7
  %v1890 = bitcast i32 %v1889 to float, !dbg !7
  %v1891 = select i1 %.not, float %v1883, float %v1890, !dbg !7
  %v1892 = select i1 %.not, float %v1890, float %v1884, !dbg !7
  %v1893 = select i1 %.not, float %v1886, float %v1885, !dbg !7
  %v1894 = bitcast float %v1893 to i32, !dbg !7
  %v1895 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1894), !dbg !7
  %v1896 = bitcast i32 %v1895 to float, !dbg !7
  %v1897 = select i1 %.not, float %v1885, float %v1896, !dbg !7
  %v1898 = select i1 %.not, float %v1896, float %v1886, !dbg !7
  %v1899 = select i1 %.not36, float %v1897, float %v1891, !dbg !7
  %v1900 = bitcast float %v1899 to i32, !dbg !7
  %v1901 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1900), !dbg !7
  %v1902 = select i1 %.not36, float %v1898, float %v1892, !dbg !7
  %v1903 = bitcast float %v1902 to i32, !dbg !7
  %v1904 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1903), !dbg !7
  %v1905 = extractvalue [2 x <1 x float>] %v207, 0, !dbg !19
  %v1906 = shufflevector <1 x float> %v1905, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %v1907 = extractelement <8 x float> %v1783, i64 4, !dbg !7
  %v1908 = extractelement <8 x float> %v1783, i64 5, !dbg !7
  %v1909 = extractelement <8 x float> %v1783, i64 6, !dbg !7
  %v1910 = extractelement <8 x float> %v1783, i64 7, !dbg !7
  %v1911 = select i1 %.not, float %v1908, float %v1907, !dbg !7
  %v1912 = bitcast float %v1911 to i32, !dbg !7
  %v1913 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1912), !dbg !7
  %v1914 = bitcast i32 %v1913 to float, !dbg !7
  %v1915 = select i1 %.not, float %v1907, float %v1914, !dbg !7
  %v1916 = select i1 %.not, float %v1914, float %v1908, !dbg !7
  %v1917 = select i1 %.not, float %v1910, float %v1909, !dbg !7
  %v1918 = bitcast float %v1917 to i32, !dbg !7
  %v1919 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1918), !dbg !7
  %v1920 = bitcast i32 %v1919 to float, !dbg !7
  %v1921 = select i1 %.not, float %v1909, float %v1920, !dbg !7
  %v1922 = select i1 %.not, float %v1920, float %v1910, !dbg !7
  %v1923 = select i1 %.not36, float %v1921, float %v1915, !dbg !7
  %v1924 = bitcast float %v1923 to i32, !dbg !7
  %v1925 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1924), !dbg !7
  %v1926 = bitcast i32 %v1925 to float, !dbg !7
  %v1927 = select i1 %.not36, float %v1915, float %v1926, !dbg !7
  %v1928 = select i1 %.not36, float %v1926, float %v1921, !dbg !7
  %v1929 = select i1 %.not36, float %v1922, float %v1916, !dbg !7
  %v1930 = bitcast float %v1929 to i32, !dbg !7
  %v1931 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1930), !dbg !7
  %v1932 = bitcast i32 %v1931 to float, !dbg !7
  %v1933 = select i1 %.not36, float %v1916, float %v1932, !dbg !7
  %v1934 = select i1 %.not36, float %v1932, float %v1922, !dbg !7
  %v1935 = extractelement <8 x float> %v1824, i64 4, !dbg !7
  %v1936 = extractelement <8 x float> %v1824, i64 5, !dbg !7
  %v1937 = extractelement <8 x float> %v1824, i64 6, !dbg !7
  %v1938 = extractelement <8 x float> %v1824, i64 7, !dbg !7
  %v1939 = select i1 %.not, float %v1936, float %v1935, !dbg !7
  %v1940 = bitcast float %v1939 to i32, !dbg !7
  %v1941 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1940), !dbg !7
  %v1942 = bitcast i32 %v1941 to float, !dbg !7
  %v1943 = select i1 %.not, float %v1935, float %v1942, !dbg !7
  %v1944 = select i1 %.not, float %v1942, float %v1936, !dbg !7
  %v1945 = select i1 %.not, float %v1938, float %v1937, !dbg !7
  %v1946 = bitcast float %v1945 to i32, !dbg !7
  %v1947 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1946), !dbg !7
  %v1948 = bitcast i32 %v1947 to float, !dbg !7
  %v1949 = select i1 %.not, float %v1937, float %v1948, !dbg !7
  %v1950 = select i1 %.not, float %v1948, float %v1938, !dbg !7
  %v1951 = select i1 %.not36, float %v1949, float %v1943, !dbg !7
  %v1952 = bitcast float %v1951 to i32, !dbg !7
  %v1953 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1952), !dbg !7
  %v1954 = bitcast i32 %v1953 to float, !dbg !7
  %v1955 = select i1 %.not36, float %v1943, float %v1954, !dbg !7
  %v1956 = select i1 %.not36, float %v1954, float %v1949, !dbg !7
  %v1957 = select i1 %.not36, float %v1950, float %v1944, !dbg !7
  %v1958 = bitcast float %v1957 to i32, !dbg !7
  %v1959 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1958), !dbg !7
  %v1960 = bitcast i32 %v1959 to float, !dbg !7
  %v1961 = select i1 %.not36, float %v1944, float %v1960, !dbg !7
  %v1962 = select i1 %.not36, float %v1960, float %v1950, !dbg !7
  %v1963 = extractelement <8 x float> %v1853, i64 4, !dbg !7
  %v1964 = extractelement <8 x float> %v1853, i64 5, !dbg !7
  %v1965 = extractelement <8 x float> %v1853, i64 6, !dbg !7
  %v1966 = extractelement <8 x float> %v1853, i64 7, !dbg !7
  %v1967 = select i1 %.not, float %v1964, float %v1963, !dbg !7
  %v1968 = bitcast float %v1967 to i32, !dbg !7
  %v1969 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1968), !dbg !7
  %v1970 = bitcast i32 %v1969 to float, !dbg !7
  %v1971 = select i1 %.not, float %v1963, float %v1970, !dbg !7
  %v1972 = select i1 %.not, float %v1970, float %v1964, !dbg !7
  %v1973 = select i1 %.not, float %v1966, float %v1965, !dbg !7
  %v1974 = bitcast float %v1973 to i32, !dbg !7
  %v1975 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1974), !dbg !7
  %v1976 = bitcast i32 %v1975 to float, !dbg !7
  %v1977 = select i1 %.not, float %v1965, float %v1976, !dbg !7
  %v1978 = select i1 %.not, float %v1976, float %v1966, !dbg !7
  %v1979 = select i1 %.not36, float %v1977, float %v1971, !dbg !7
  %v1980 = bitcast float %v1979 to i32, !dbg !7
  %v1981 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1980), !dbg !7
  %v1982 = bitcast i32 %v1981 to float, !dbg !7
  %v1983 = select i1 %.not36, float %v1971, float %v1982, !dbg !7
  %v1984 = select i1 %.not36, float %v1982, float %v1977, !dbg !7
  %v1985 = select i1 %.not36, float %v1978, float %v1972, !dbg !7
  %v1986 = bitcast float %v1985 to i32, !dbg !7
  %v1987 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1986), !dbg !7
  %v1988 = bitcast i32 %v1987 to float, !dbg !7
  %v1989 = select i1 %.not36, float %v1972, float %v1988, !dbg !7
  %v1990 = select i1 %.not36, float %v1988, float %v1978, !dbg !7
  %v1991 = extractelement <8 x float> %v1882, i64 4, !dbg !7
  %v1992 = extractelement <8 x float> %v1882, i64 5, !dbg !7
  %v1993 = extractelement <8 x float> %v1882, i64 6, !dbg !7
  %v1994 = extractelement <8 x float> %v1882, i64 7, !dbg !7
  %v1995 = select i1 %.not, float %v1992, float %v1991, !dbg !7
  %v1996 = bitcast float %v1995 to i32, !dbg !7
  %v1997 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1996), !dbg !7
  %v1998 = bitcast i32 %v1997 to float, !dbg !7
  %v1999 = select i1 %.not, float %v1991, float %v1998, !dbg !7
  %v2000 = select i1 %.not, float %v1998, float %v1992, !dbg !7
  %v2001 = select i1 %.not, float %v1994, float %v1993, !dbg !7
  %v2002 = bitcast float %v2001 to i32, !dbg !7
  %v2003 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2002), !dbg !7
  %v2004 = bitcast i32 %v2003 to float, !dbg !7
  %v2005 = select i1 %.not, float %v1993, float %v2004, !dbg !7
  %v2006 = select i1 %.not, float %v2004, float %v1994, !dbg !7
  %v2007 = select i1 %.not36, float %v2005, float %v1999, !dbg !7
  %v2008 = bitcast float %v2007 to i32, !dbg !7
  %v2009 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2008), !dbg !7
  %v2010 = select i1 %.not36, float %v2006, float %v2000, !dbg !7
  %v2011 = bitcast float %v2010 to i32, !dbg !7
  %v2012 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2011), !dbg !7
  %v2013 = extractvalue [2 x <8 x float>] %v211, 1, !dbg !7
  %v2014 = extractelement <8 x float> %v2013, i64 0, !dbg !7
  %v2015 = extractelement <8 x float> %v2013, i64 1, !dbg !7
  %v2016 = extractelement <8 x float> %v2013, i64 2, !dbg !7
  %v2017 = extractelement <8 x float> %v2013, i64 3, !dbg !7
  %v2018 = select i1 %.not, float %v2015, float %v2014, !dbg !7
  %v2019 = bitcast float %v2018 to i32, !dbg !7
  %v2020 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2019), !dbg !7
  %v2021 = bitcast i32 %v2020 to float, !dbg !7
  %v2022 = select i1 %.not, float %v2014, float %v2021, !dbg !7
  %v2023 = select i1 %.not, float %v2021, float %v2015, !dbg !7
  %v2024 = select i1 %.not, float %v2017, float %v2016, !dbg !7
  %v2025 = bitcast float %v2024 to i32, !dbg !7
  %v2026 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2025), !dbg !7
  %v2027 = bitcast i32 %v2026 to float, !dbg !7
  %v2028 = select i1 %.not, float %v2016, float %v2027, !dbg !7
  %v2029 = select i1 %.not, float %v2027, float %v2017, !dbg !7
  %v2030 = select i1 %.not36, float %v2028, float %v2022, !dbg !7
  %v2031 = bitcast float %v2030 to i32, !dbg !7
  %v2032 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2031), !dbg !7
  %v2033 = bitcast i32 %v2032 to float, !dbg !7
  %v2034 = select i1 %.not36, float %v2022, float %v2033, !dbg !7
  %v2035 = select i1 %.not36, float %v2033, float %v2028, !dbg !7
  %v2036 = select i1 %.not36, float %v2029, float %v2023, !dbg !7
  %v2037 = bitcast float %v2036 to i32, !dbg !7
  %v2038 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2037), !dbg !7
  %v2039 = bitcast i32 %v2038 to float, !dbg !7
  %v2040 = select i1 %.not36, float %v2023, float %v2039, !dbg !7
  %v2041 = select i1 %.not36, float %v2039, float %v2029, !dbg !7
  %v2042 = extractvalue [2 x <8 x float>] %v210, 1, !dbg !7
  %v2043 = extractelement <8 x float> %v2042, i64 0, !dbg !7
  %v2044 = extractelement <8 x float> %v2042, i64 1, !dbg !7
  %v2045 = extractelement <8 x float> %v2042, i64 2, !dbg !7
  %v2046 = extractelement <8 x float> %v2042, i64 3, !dbg !7
  %v2047 = select i1 %.not, float %v2044, float %v2043, !dbg !7
  %v2048 = bitcast float %v2047 to i32, !dbg !7
  %v2049 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2048), !dbg !7
  %v2050 = bitcast i32 %v2049 to float, !dbg !7
  %v2051 = select i1 %.not, float %v2043, float %v2050, !dbg !7
  %v2052 = select i1 %.not, float %v2050, float %v2044, !dbg !7
  %v2053 = select i1 %.not, float %v2046, float %v2045, !dbg !7
  %v2054 = bitcast float %v2053 to i32, !dbg !7
  %v2055 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2054), !dbg !7
  %v2056 = bitcast i32 %v2055 to float, !dbg !7
  %v2057 = select i1 %.not, float %v2045, float %v2056, !dbg !7
  %v2058 = select i1 %.not, float %v2056, float %v2046, !dbg !7
  %v2059 = select i1 %.not36, float %v2057, float %v2051, !dbg !7
  %v2060 = bitcast float %v2059 to i32, !dbg !7
  %v2061 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2060), !dbg !7
  %v2062 = bitcast i32 %v2061 to float, !dbg !7
  %v2063 = select i1 %.not36, float %v2051, float %v2062, !dbg !7
  %v2064 = select i1 %.not36, float %v2062, float %v2057, !dbg !7
  %v2065 = select i1 %.not36, float %v2058, float %v2052, !dbg !7
  %v2066 = bitcast float %v2065 to i32, !dbg !7
  %v2067 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2066), !dbg !7
  %v2068 = bitcast i32 %v2067 to float, !dbg !7
  %v2069 = select i1 %.not36, float %v2052, float %v2068, !dbg !7
  %v2070 = select i1 %.not36, float %v2068, float %v2058, !dbg !7
  %v2071 = extractvalue [2 x <8 x float>] %v209, 1, !dbg !7
  %v2072 = extractelement <8 x float> %v2071, i64 0, !dbg !7
  %v2073 = extractelement <8 x float> %v2071, i64 1, !dbg !7
  %v2074 = extractelement <8 x float> %v2071, i64 2, !dbg !7
  %v2075 = extractelement <8 x float> %v2071, i64 3, !dbg !7
  %v2076 = select i1 %.not, float %v2073, float %v2072, !dbg !7
  %v2077 = bitcast float %v2076 to i32, !dbg !7
  %v2078 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2077), !dbg !7
  %v2079 = bitcast i32 %v2078 to float, !dbg !7
  %v2080 = select i1 %.not, float %v2072, float %v2079, !dbg !7
  %v2081 = select i1 %.not, float %v2079, float %v2073, !dbg !7
  %v2082 = select i1 %.not, float %v2075, float %v2074, !dbg !7
  %v2083 = bitcast float %v2082 to i32, !dbg !7
  %v2084 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2083), !dbg !7
  %v2085 = bitcast i32 %v2084 to float, !dbg !7
  %v2086 = select i1 %.not, float %v2074, float %v2085, !dbg !7
  %v2087 = select i1 %.not, float %v2085, float %v2075, !dbg !7
  %v2088 = select i1 %.not36, float %v2086, float %v2080, !dbg !7
  %v2089 = bitcast float %v2088 to i32, !dbg !7
  %v2090 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2089), !dbg !7
  %v2091 = bitcast i32 %v2090 to float, !dbg !7
  %v2092 = select i1 %.not36, float %v2080, float %v2091, !dbg !7
  %v2093 = select i1 %.not36, float %v2091, float %v2086, !dbg !7
  %v2094 = select i1 %.not36, float %v2087, float %v2081, !dbg !7
  %v2095 = bitcast float %v2094 to i32, !dbg !7
  %v2096 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2095), !dbg !7
  %v2097 = bitcast i32 %v2096 to float, !dbg !7
  %v2098 = select i1 %.not36, float %v2081, float %v2097, !dbg !7
  %v2099 = select i1 %.not36, float %v2097, float %v2087, !dbg !7
  %v2100 = extractvalue [2 x <8 x float>] %v206, 1, !dbg !7
  %v2101 = extractelement <8 x float> %v2100, i64 0, !dbg !7
  %v2102 = extractelement <8 x float> %v2100, i64 1, !dbg !7
  %v2103 = extractelement <8 x float> %v2100, i64 2, !dbg !7
  %v2104 = extractelement <8 x float> %v2100, i64 3, !dbg !7
  %v2105 = select i1 %.not, float %v2102, float %v2101, !dbg !7
  %v2106 = bitcast float %v2105 to i32, !dbg !7
  %v2107 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2106), !dbg !7
  %v2108 = bitcast i32 %v2107 to float, !dbg !7
  %v2109 = select i1 %.not, float %v2101, float %v2108, !dbg !7
  %v2110 = select i1 %.not, float %v2108, float %v2102, !dbg !7
  %v2111 = select i1 %.not, float %v2104, float %v2103, !dbg !7
  %v2112 = bitcast float %v2111 to i32, !dbg !7
  %v2113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2112), !dbg !7
  %v2114 = bitcast i32 %v2113 to float, !dbg !7
  %v2115 = select i1 %.not, float %v2103, float %v2114, !dbg !7
  %v2116 = select i1 %.not, float %v2114, float %v2104, !dbg !7
  %v2117 = select i1 %.not36, float %v2115, float %v2109, !dbg !7
  %v2118 = bitcast float %v2117 to i32, !dbg !7
  %v2119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2118), !dbg !7
  %v2120 = select i1 %.not36, float %v2116, float %v2110, !dbg !7
  %v2121 = bitcast float %v2120 to i32, !dbg !7
  %v2122 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2121), !dbg !7
  %v2123 = extractvalue [2 x <1 x float>] %v207, 1, !dbg !19
  %v2124 = shufflevector <1 x float> %v2123, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %v2125 = extractelement <8 x float> %v2013, i64 4, !dbg !7
  %v2126 = extractelement <8 x float> %v2013, i64 5, !dbg !7
  %v2127 = extractelement <8 x float> %v2013, i64 6, !dbg !7
  %v2128 = extractelement <8 x float> %v2013, i64 7, !dbg !7
  %v2129 = select i1 %.not, float %v2126, float %v2125, !dbg !7
  %v2130 = bitcast float %v2129 to i32, !dbg !7
  %v2131 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2130), !dbg !7
  %v2132 = bitcast i32 %v2131 to float, !dbg !7
  %v2133 = select i1 %.not, float %v2125, float %v2132, !dbg !7
  %v2134 = select i1 %.not, float %v2132, float %v2126, !dbg !7
  %v2135 = select i1 %.not, float %v2128, float %v2127, !dbg !7
  %v2136 = bitcast float %v2135 to i32, !dbg !7
  %v2137 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2136), !dbg !7
  %v2138 = bitcast i32 %v2137 to float, !dbg !7
  %v2139 = select i1 %.not, float %v2127, float %v2138, !dbg !7
  %v2140 = select i1 %.not, float %v2138, float %v2128, !dbg !7
  %v2141 = select i1 %.not36, float %v2139, float %v2133, !dbg !7
  %v2142 = bitcast float %v2141 to i32, !dbg !7
  %v2143 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2142), !dbg !7
  %v2144 = bitcast i32 %v2143 to float, !dbg !7
  %v2145 = select i1 %.not36, float %v2133, float %v2144, !dbg !7
  %v2146 = select i1 %.not36, float %v2144, float %v2139, !dbg !7
  %v2147 = select i1 %.not36, float %v2140, float %v2134, !dbg !7
  %v2148 = bitcast float %v2147 to i32, !dbg !7
  %v2149 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2148), !dbg !7
  %v2150 = bitcast i32 %v2149 to float, !dbg !7
  %v2151 = select i1 %.not36, float %v2134, float %v2150, !dbg !7
  %v2152 = select i1 %.not36, float %v2150, float %v2140, !dbg !7
  %v2153 = extractelement <8 x float> %v2042, i64 4, !dbg !7
  %v2154 = extractelement <8 x float> %v2042, i64 5, !dbg !7
  %v2155 = extractelement <8 x float> %v2042, i64 6, !dbg !7
  %v2156 = extractelement <8 x float> %v2042, i64 7, !dbg !7
  %v2157 = select i1 %.not, float %v2154, float %v2153, !dbg !7
  %v2158 = bitcast float %v2157 to i32, !dbg !7
  %v2159 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2158), !dbg !7
  %v2160 = bitcast i32 %v2159 to float, !dbg !7
  %v2161 = select i1 %.not, float %v2153, float %v2160, !dbg !7
  %v2162 = select i1 %.not, float %v2160, float %v2154, !dbg !7
  %v2163 = select i1 %.not, float %v2156, float %v2155, !dbg !7
  %v2164 = bitcast float %v2163 to i32, !dbg !7
  %v2165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2164), !dbg !7
  %v2166 = bitcast i32 %v2165 to float, !dbg !7
  %v2167 = select i1 %.not, float %v2155, float %v2166, !dbg !7
  %v2168 = select i1 %.not, float %v2166, float %v2156, !dbg !7
  %v2169 = select i1 %.not36, float %v2167, float %v2161, !dbg !7
  %v2170 = bitcast float %v2169 to i32, !dbg !7
  %v2171 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2170), !dbg !7
  %v2172 = bitcast i32 %v2171 to float, !dbg !7
  %v2173 = select i1 %.not36, float %v2161, float %v2172, !dbg !7
  %v2174 = select i1 %.not36, float %v2172, float %v2167, !dbg !7
  %v2175 = select i1 %.not36, float %v2168, float %v2162, !dbg !7
  %v2176 = bitcast float %v2175 to i32, !dbg !7
  %v2177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2176), !dbg !7
  %v2178 = bitcast i32 %v2177 to float, !dbg !7
  %v2179 = select i1 %.not36, float %v2162, float %v2178, !dbg !7
  %v2180 = select i1 %.not36, float %v2178, float %v2168, !dbg !7
  %v2181 = extractelement <8 x float> %v2071, i64 4, !dbg !7
  %v2182 = extractelement <8 x float> %v2071, i64 5, !dbg !7
  %v2183 = extractelement <8 x float> %v2071, i64 6, !dbg !7
  %v2184 = extractelement <8 x float> %v2071, i64 7, !dbg !7
  %v2185 = select i1 %.not, float %v2182, float %v2181, !dbg !7
  %v2186 = bitcast float %v2185 to i32, !dbg !7
  %v2187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2186), !dbg !7
  %v2188 = bitcast i32 %v2187 to float, !dbg !7
  %v2189 = select i1 %.not, float %v2181, float %v2188, !dbg !7
  %v2190 = select i1 %.not, float %v2188, float %v2182, !dbg !7
  %v2191 = select i1 %.not, float %v2184, float %v2183, !dbg !7
  %v2192 = bitcast float %v2191 to i32, !dbg !7
  %v2193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2192), !dbg !7
  %v2194 = bitcast i32 %v2193 to float, !dbg !7
  %v2195 = select i1 %.not, float %v2183, float %v2194, !dbg !7
  %v2196 = select i1 %.not, float %v2194, float %v2184, !dbg !7
  %v2197 = select i1 %.not36, float %v2195, float %v2189, !dbg !7
  %v2198 = bitcast float %v2197 to i32, !dbg !7
  %v2199 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2198), !dbg !7
  %v2200 = bitcast i32 %v2199 to float, !dbg !7
  %v2201 = select i1 %.not36, float %v2189, float %v2200, !dbg !7
  %v2202 = select i1 %.not36, float %v2200, float %v2195, !dbg !7
  %v2203 = select i1 %.not36, float %v2196, float %v2190, !dbg !7
  %v2204 = bitcast float %v2203 to i32, !dbg !7
  %v2205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2204), !dbg !7
  %v2206 = bitcast i32 %v2205 to float, !dbg !7
  %v2207 = select i1 %.not36, float %v2190, float %v2206, !dbg !7
  %v2208 = select i1 %.not36, float %v2206, float %v2196, !dbg !7
  %v2209 = extractelement <8 x float> %v2100, i64 4, !dbg !7
  %v2210 = extractelement <8 x float> %v2100, i64 5, !dbg !7
  %v2211 = extractelement <8 x float> %v2100, i64 6, !dbg !7
  %v2212 = extractelement <8 x float> %v2100, i64 7, !dbg !7
  %v2213 = select i1 %.not, float %v2210, float %v2209, !dbg !7
  %v2214 = bitcast float %v2213 to i32, !dbg !7
  %v2215 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2214), !dbg !7
  %v2216 = bitcast i32 %v2215 to float, !dbg !7
  %v2217 = select i1 %.not, float %v2209, float %v2216, !dbg !7
  %v2218 = select i1 %.not, float %v2216, float %v2210, !dbg !7
  %v2219 = select i1 %.not, float %v2212, float %v2211, !dbg !7
  %v2220 = bitcast float %v2219 to i32, !dbg !7
  %v2221 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2220), !dbg !7
  %v2222 = bitcast i32 %v2221 to float, !dbg !7
  %v2223 = select i1 %.not, float %v2211, float %v2222, !dbg !7
  %v2224 = select i1 %.not, float %v2222, float %v2212, !dbg !7
  %v2225 = select i1 %.not36, float %v2223, float %v2217, !dbg !7
  %v2226 = bitcast float %v2225 to i32, !dbg !7
  %v2227 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2226), !dbg !7
  %v2228 = select i1 %.not36, float %v2224, float %v2218, !dbg !7
  %v2229 = bitcast float %v2228 to i32, !dbg !7
  %v2230 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2229), !dbg !7
  br i1 true, label %v2231, label %v2295, !dbg !21

v2231:                                             ; preds = %v1780
  %v2232 = bitcast i32 %v2227 to float, !dbg !7
  %v2233 = select i1 %.not36, float %v2217, float %v2232, !dbg !7
  %v2234 = insertelement <4 x float> poison, float %v2233, i64 0, !dbg !19
  %v2235 = bitcast i32 %v2230 to float, !dbg !7
  %v2236 = select i1 %.not36, float %v2218, float %v2235, !dbg !7
  %v2237 = insertelement <4 x float> %v2234, float %v2236, i64 1, !dbg !19
  %v2238 = select i1 %.not36, float %v2232, float %v2223, !dbg !7
  %v2239 = insertelement <4 x float> %v2237, float %v2238, i64 2, !dbg !19
  %v2240 = select i1 %.not36, float %v2235, float %v2224, !dbg !7
  %v2241 = insertelement <4 x float> %v2239, float %v2240, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v2124, !dbg !19
  %normalized = fmul <4 x float> %v2241, %row.reciprocal, !dbg !19
  %v2242 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %v2243 = extractelement <4 x half> %v2242, i64 3, !dbg !20
  %v2244 = extractelement <4 x half> %v2242, i64 2, !dbg !20
  %v2245 = extractelement <4 x half> %v2242, i64 1, !dbg !20
  %v2246 = extractelement <4 x half> %v2242, i64 0, !dbg !20
  %v2247 = bitcast i32 %v2119 to float, !dbg !7
  %v2248 = select i1 %.not36, float %v2109, float %v2247, !dbg !7
  %v2249 = insertelement <4 x float> poison, float %v2248, i64 0, !dbg !19
  %v2250 = bitcast i32 %v2122 to float, !dbg !7
  %v2251 = select i1 %.not36, float %v2110, float %v2250, !dbg !7
  %v2252 = insertelement <4 x float> %v2249, float %v2251, i64 1, !dbg !19
  %v2253 = select i1 %.not36, float %v2247, float %v2115, !dbg !7
  %v2254 = insertelement <4 x float> %v2252, float %v2253, i64 2, !dbg !19
  %v2255 = select i1 %.not36, float %v2250, float %v2116, !dbg !7
  %v2256 = insertelement <4 x float> %v2254, float %v2255, i64 3, !dbg !19
  %normalized104 = fmul <4 x float> %v2256, %row.reciprocal, !dbg !19
  %v2257 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %v2258 = extractelement <4 x half> %v2257, i64 3, !dbg !20
  %v2259 = extractelement <4 x half> %v2257, i64 2, !dbg !20
  %v2260 = extractelement <4 x half> %v2257, i64 1, !dbg !20
  %v2261 = extractelement <4 x half> %v2257, i64 0, !dbg !20
  %v2262 = bitcast i32 %v2009 to float, !dbg !7
  %v2263 = select i1 %.not36, float %v1999, float %v2262, !dbg !7
  %v2264 = insertelement <4 x float> poison, float %v2263, i64 0, !dbg !19
  %v2265 = bitcast i32 %v2012 to float, !dbg !7
  %v2266 = select i1 %.not36, float %v2000, float %v2265, !dbg !7
  %v2267 = insertelement <4 x float> %v2264, float %v2266, i64 1, !dbg !19
  %v2268 = select i1 %.not36, float %v2262, float %v2005, !dbg !7
  %v2269 = insertelement <4 x float> %v2267, float %v2268, i64 2, !dbg !19
  %v2270 = select i1 %.not36, float %v2265, float %v2006, !dbg !7
  %v2271 = insertelement <4 x float> %v2269, float %v2270, i64 3, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1906, !dbg !19
  %normalized106 = fmul <4 x float> %v2271, %row.reciprocal105, !dbg !19
  %v2272 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %v2273 = extractelement <4 x half> %v2272, i64 3, !dbg !20
  %v2274 = extractelement <4 x half> %v2272, i64 2, !dbg !20
  %v2275 = extractelement <4 x half> %v2272, i64 1, !dbg !20
  %v2276 = extractelement <4 x half> %v2272, i64 0, !dbg !20
  %v2277 = bitcast i32 %v1901 to float, !dbg !7
  %v2278 = select i1 %.not36, float %v1891, float %v2277, !dbg !7
  %v2279 = insertelement <4 x float> poison, float %v2278, i64 0, !dbg !19
  %v2280 = bitcast i32 %v1904 to float, !dbg !7
  %v2281 = select i1 %.not36, float %v1892, float %v2280, !dbg !7
  %v2282 = insertelement <4 x float> %v2279, float %v2281, i64 1, !dbg !19
  %v2283 = select i1 %.not36, float %v2277, float %v1897, !dbg !7
  %v2284 = insertelement <4 x float> %v2282, float %v2283, i64 2, !dbg !19
  %v2285 = select i1 %.not36, float %v2280, float %v1898, !dbg !7
  %v2286 = insertelement <4 x float> %v2284, float %v2285, i64 3, !dbg !19
  %normalized107 = fmul <4 x float> %v2286, %row.reciprocal105, !dbg !19
  %v2287 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %v2288 = extractelement <4 x half> %v2287, i64 3, !dbg !20
  %v2289 = extractelement <4 x half> %v2287, i64 2, !dbg !20
  %v2290 = extractelement <4 x half> %v2287, i64 1, !dbg !20
  %v2291 = extractelement <4 x half> %v2287, i64 0, !dbg !20
  %v2292 = add nsw i64 %v16, %v22, !dbg !21
  %v2293 = shl i64 %v2292, 19, !dbg !21
  %v2294 = and i64 %v2293, 16252928, !dbg !21
  %address.thread934 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !8
  %address.local936 = add i64 %v25, %address.local935, !dbg !21
  %address.local937 = lshr i64 %address.local935, 6, !dbg !11
  %address.local938 = shl nuw nsw i64 %address.local937, 4, !dbg !21
  %address.local939 = add i64 %address.local936, %address.local938, !dbg !21
  %address.local940 = and i64 %address.local935, 240, !dbg !21
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !21
  %address.local942 = add i64 %address.local941, %v26, !dbg !21
  %address.local943 = shl nuw nsw i64 %address.local942, 7, !dbg !21
  %address.local944 = add nuw nsw i64 %v2294, %address.local943, !dbg !21
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local946 = lshr i64 %address.local935, 2, !dbg !11
  %address.local947 = and i64 %address.local946, 12, !dbg !11
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !21
  store half %v2291, ptr addrspace(1) %address.local948, align 2, !dbg !21
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !21
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local949, i64 %address.local950, !dbg !21
  store half %v2290, ptr addrspace(1) %address.local951, align 2, !dbg !21
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !21
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local952, i64 %address.local953, !dbg !21
  store half %v2289, ptr addrspace(1) %address.local954, align 2, !dbg !21
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !21
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local955, i64 %address.local956, !dbg !21
  store half %v2288, ptr addrspace(1) %address.local957, align 2, !dbg !21
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !21
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local958, i64 %address.local959, !dbg !21
  store half %v2276, ptr addrspace(1) %address.local960, align 2, !dbg !21
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !21
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local961, i64 %address.local962, !dbg !21
  store half %v2275, ptr addrspace(1) %address.local963, align 2, !dbg !21
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !21
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local964, i64 %address.local965, !dbg !21
  store half %v2274, ptr addrspace(1) %address.local966, align 2, !dbg !21
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !21
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local967, i64 %address.local968, !dbg !21
  store half %v2273, ptr addrspace(1) %address.local969, align 2, !dbg !21
  %address.local970 = shl i64 %address.local942, 7, !dbg !21
  %address.local971 = add i64 %address.local970, 8192, !dbg !21
  %address.local972 = add nuw nsw i64 %v2294, %address.local971, !dbg !21
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !21
  store half %v2261, ptr addrspace(1) %address.local974, align 2, !dbg !21
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local975, i64 %address.local950, !dbg !21
  store half %v2260, ptr addrspace(1) %address.local976, align 2, !dbg !21
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local953, !dbg !21
  store half %v2259, ptr addrspace(1) %address.local978, align 2, !dbg !21
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local979, i64 %address.local956, !dbg !21
  store half %v2258, ptr addrspace(1) %address.local980, align 2, !dbg !21
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local981, i64 %address.local959, !dbg !21
  store half %v2246, ptr addrspace(1) %address.local982, align 2, !dbg !21
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local962, !dbg !21
  store half %v2245, ptr addrspace(1) %address.local984, align 2, !dbg !21
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local965, !dbg !21
  store half %v2244, ptr addrspace(1) %address.local986, align 2, !dbg !21
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local987, i64 %address.local968, !dbg !21
  store half %v2243, ptr addrspace(1) %address.local988, align 2, !dbg !21
  br label %v2295, !dbg !21

v2295:                                             ; preds = %v2231, %v1780
  br i1 true, label %v2296, label %v2334, !dbg !21

v2296:                                             ; preds = %v2295
  %v2297 = insertelement <4 x float> poison, float %v2201, i64 0, !dbg !19
  %v2298 = insertelement <4 x float> %v2297, float %v2207, i64 1, !dbg !19
  %v2299 = insertelement <4 x float> %v2298, float %v2202, i64 2, !dbg !19
  %v2300 = insertelement <4 x float> %v2299, float %v2208, i64 3, !dbg !19
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v2124, !dbg !19
  %normalized109 = fmul <4 x float> %v2300, %row.reciprocal108, !dbg !19
  %v2301 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %v2302 = extractelement <4 x half> %v2301, i64 3, !dbg !20
  %v2303 = extractelement <4 x half> %v2301, i64 2, !dbg !20
  %v2304 = extractelement <4 x half> %v2301, i64 1, !dbg !20
  %v2305 = extractelement <4 x half> %v2301, i64 0, !dbg !20
  %v2306 = insertelement <4 x float> poison, float %v2092, i64 0, !dbg !19
  %v2307 = insertelement <4 x float> %v2306, float %v2098, i64 1, !dbg !19
  %v2308 = insertelement <4 x float> %v2307, float %v2093, i64 2, !dbg !19
  %v2309 = insertelement <4 x float> %v2308, float %v2099, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %v2309, %row.reciprocal108, !dbg !19
  %v2310 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %v2311 = extractelement <4 x half> %v2310, i64 3, !dbg !20
  %v2312 = extractelement <4 x half> %v2310, i64 2, !dbg !20
  %v2313 = extractelement <4 x half> %v2310, i64 1, !dbg !20
  %v2314 = extractelement <4 x half> %v2310, i64 0, !dbg !20
  %v2315 = insertelement <4 x float> poison, float %v1983, i64 0, !dbg !19
  %v2316 = insertelement <4 x float> %v2315, float %v1989, i64 1, !dbg !19
  %v2317 = insertelement <4 x float> %v2316, float %v1984, i64 2, !dbg !19
  %v2318 = insertelement <4 x float> %v2317, float %v1990, i64 3, !dbg !19
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1906, !dbg !19
  %normalized112 = fmul <4 x float> %v2318, %row.reciprocal111, !dbg !19
  %v2319 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %v2320 = extractelement <4 x half> %v2319, i64 3, !dbg !20
  %v2321 = extractelement <4 x half> %v2319, i64 2, !dbg !20
  %v2322 = extractelement <4 x half> %v2319, i64 1, !dbg !20
  %v2323 = extractelement <4 x half> %v2319, i64 0, !dbg !20
  %v2324 = insertelement <4 x float> poison, float %v1874, i64 0, !dbg !19
  %v2325 = insertelement <4 x float> %v2324, float %v1880, i64 1, !dbg !19
  %v2326 = insertelement <4 x float> %v2325, float %v1875, i64 2, !dbg !19
  %v2327 = insertelement <4 x float> %v2326, float %v1881, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %v2327, %row.reciprocal111, !dbg !19
  %v2328 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %v2329 = extractelement <4 x half> %v2328, i64 3, !dbg !20
  %v2330 = extractelement <4 x half> %v2328, i64 2, !dbg !20
  %v2331 = extractelement <4 x half> %v2328, i64 1, !dbg !20
  %v2332 = extractelement <4 x half> %v2328, i64 0, !dbg !20
  %v2333 = shl nuw nsw i64 %v24, 19, !dbg !21
  %address.thread989 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local990 = zext nneg i32 %address.thread989 to i64, !dbg !8
  %address.local991 = add i64 %v25, %address.local990, !dbg !21
  %address.local992 = lshr i64 %address.local990, 6, !dbg !11
  %address.local993 = shl nuw nsw i64 %address.local992, 4, !dbg !21
  %address.local994 = add i64 %address.local991, %address.local993, !dbg !21
  %address.local995 = and i64 %address.local990, 240, !dbg !21
  %address.local996 = sub i64 %address.local994, %address.local995, !dbg !21
  %address.local997 = add i64 %address.local996, %v26, !dbg !21
  %address.local998 = shl nuw nsw i64 %address.local997, 7, !dbg !21
  %address.local999 = add nuw nsw i64 %v2333, %address.local998, !dbg !21
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1001 = lshr i64 %address.local990, 2, !dbg !11
  %address.local1002 = and i64 %address.local1001, 12, !dbg !11
  %address.local1003 = or disjoint i64 %address.local1002, 32, !dbg !21
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local1003, !dbg !21
  store half %v2332, ptr addrspace(1) %address.local1004, align 2, !dbg !21
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1006 = or disjoint i64 %address.local1002, 33, !dbg !21
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1005, i64 %address.local1006, !dbg !21
  store half %v2331, ptr addrspace(1) %address.local1007, align 2, !dbg !21
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1009 = or disjoint i64 %address.local1002, 34, !dbg !21
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1008, i64 %address.local1009, !dbg !21
  store half %v2330, ptr addrspace(1) %address.local1010, align 2, !dbg !21
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1012 = or disjoint i64 %address.local1002, 35, !dbg !21
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local1012, !dbg !21
  store half %v2329, ptr addrspace(1) %address.local1013, align 2, !dbg !21
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1015 = or disjoint i64 %address.local1002, 48, !dbg !21
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local1015, !dbg !21
  store half %v2323, ptr addrspace(1) %address.local1016, align 2, !dbg !21
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1018 = or disjoint i64 %address.local1002, 49, !dbg !21
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local1018, !dbg !21
  store half %v2322, ptr addrspace(1) %address.local1019, align 2, !dbg !21
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1021 = or disjoint i64 %address.local1002, 50, !dbg !21
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1020, i64 %address.local1021, !dbg !21
  store half %v2321, ptr addrspace(1) %address.local1022, align 2, !dbg !21
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1024 = or i64 %address.local1001, 51, !dbg !21
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1024, !dbg !21
  store half %v2320, ptr addrspace(1) %address.local1025, align 2, !dbg !21
  %address.local1026 = shl i64 %address.local997, 7, !dbg !21
  %address.local1027 = add i64 %address.local1026, 8192, !dbg !21
  %address.local1028 = add nuw nsw i64 %v2333, %address.local1027, !dbg !21
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1029, i64 %address.local1003, !dbg !21
  store half %v2314, ptr addrspace(1) %address.local1030, align 2, !dbg !21
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1006, !dbg !21
  store half %v2313, ptr addrspace(1) %address.local1032, align 2, !dbg !21
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1009, !dbg !21
  store half %v2312, ptr addrspace(1) %address.local1034, align 2, !dbg !21
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1012, !dbg !21
  store half %v2311, ptr addrspace(1) %address.local1036, align 2, !dbg !21
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1015, !dbg !21
  store half %v2305, ptr addrspace(1) %address.local1038, align 2, !dbg !21
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1018, !dbg !21
  store half %v2304, ptr addrspace(1) %address.local1040, align 2, !dbg !21
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1021, !dbg !21
  store half %v2303, ptr addrspace(1) %address.local1042, align 2, !dbg !21
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1043, i64 %address.local1024, !dbg !21
  store half %v2302, ptr addrspace(1) %address.local1044, align 2, !dbg !21
  br label %v2334, !dbg !21

v2334:                                             ; preds = %v2296, %v2295
  br i1 true, label %v2335, label %v2373, !dbg !21

v2335:                                             ; preds = %v2334
  %v2336 = insertelement <4 x float> poison, float %v2173, i64 0, !dbg !19
  %v2337 = insertelement <4 x float> %v2336, float %v2179, i64 1, !dbg !19
  %v2338 = insertelement <4 x float> %v2337, float %v2174, i64 2, !dbg !19
  %v2339 = insertelement <4 x float> %v2338, float %v2180, i64 3, !dbg !19
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v2124, !dbg !19
  %normalized115 = fmul <4 x float> %v2339, %row.reciprocal114, !dbg !19
  %v2340 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %v2341 = extractelement <4 x half> %v2340, i64 3, !dbg !20
  %v2342 = extractelement <4 x half> %v2340, i64 2, !dbg !20
  %v2343 = extractelement <4 x half> %v2340, i64 1, !dbg !20
  %v2344 = extractelement <4 x half> %v2340, i64 0, !dbg !20
  %v2345 = insertelement <4 x float> poison, float %v2063, i64 0, !dbg !19
  %v2346 = insertelement <4 x float> %v2345, float %v2069, i64 1, !dbg !19
  %v2347 = insertelement <4 x float> %v2346, float %v2064, i64 2, !dbg !19
  %v2348 = insertelement <4 x float> %v2347, float %v2070, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %v2348, %row.reciprocal114, !dbg !19
  %v2349 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %v2350 = extractelement <4 x half> %v2349, i64 3, !dbg !20
  %v2351 = extractelement <4 x half> %v2349, i64 2, !dbg !20
  %v2352 = extractelement <4 x half> %v2349, i64 1, !dbg !20
  %v2353 = extractelement <4 x half> %v2349, i64 0, !dbg !20
  %v2354 = insertelement <4 x float> poison, float %v1955, i64 0, !dbg !19
  %v2355 = insertelement <4 x float> %v2354, float %v1961, i64 1, !dbg !19
  %v2356 = insertelement <4 x float> %v2355, float %v1956, i64 2, !dbg !19
  %v2357 = insertelement <4 x float> %v2356, float %v1962, i64 3, !dbg !19
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1906, !dbg !19
  %normalized118 = fmul <4 x float> %v2357, %row.reciprocal117, !dbg !19
  %v2358 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %v2359 = extractelement <4 x half> %v2358, i64 3, !dbg !20
  %v2360 = extractelement <4 x half> %v2358, i64 2, !dbg !20
  %v2361 = extractelement <4 x half> %v2358, i64 1, !dbg !20
  %v2362 = extractelement <4 x half> %v2358, i64 0, !dbg !20
  %v2363 = insertelement <4 x float> poison, float %v1845, i64 0, !dbg !19
  %v2364 = insertelement <4 x float> %v2363, float %v1851, i64 1, !dbg !19
  %v2365 = insertelement <4 x float> %v2364, float %v1846, i64 2, !dbg !19
  %v2366 = insertelement <4 x float> %v2365, float %v1852, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %v2366, %row.reciprocal117, !dbg !19
  %v2367 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %v2368 = extractelement <4 x half> %v2367, i64 3, !dbg !20
  %v2369 = extractelement <4 x half> %v2367, i64 2, !dbg !20
  %v2370 = extractelement <4 x half> %v2367, i64 1, !dbg !20
  %v2371 = extractelement <4 x half> %v2367, i64 0, !dbg !20
  %v2372 = shl nuw nsw i64 %v24, 19, !dbg !21
  %address.thread1045 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1046 = zext nneg i32 %address.thread1045 to i64, !dbg !8
  %address.local1047 = add i64 %v25, %address.local1046, !dbg !21
  %address.local1048 = lshr i64 %address.local1046, 6, !dbg !11
  %address.local1049 = shl nuw nsw i64 %address.local1048, 4, !dbg !21
  %address.local1050 = add i64 %address.local1047, %address.local1049, !dbg !21
  %address.local1051 = and i64 %address.local1046, 240, !dbg !21
  %address.local1052 = sub i64 %address.local1050, %address.local1051, !dbg !21
  %address.local1053 = add i64 %address.local1052, %v26, !dbg !21
  %address.local1054 = shl nuw nsw i64 %address.local1053, 7, !dbg !21
  %address.local1055 = add nuw nsw i64 %v2372, %address.local1054, !dbg !21
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1057 = lshr i64 %address.local1046, 2, !dbg !11
  %address.local1058 = and i64 %address.local1057, 12, !dbg !11
  %address.local1059 = or disjoint i64 %address.local1058, 64, !dbg !21
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1059, !dbg !21
  store half %v2371, ptr addrspace(1) %address.local1060, align 2, !dbg !21
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1062 = or disjoint i64 %address.local1058, 65, !dbg !21
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1061, i64 %address.local1062, !dbg !21
  store half %v2370, ptr addrspace(1) %address.local1063, align 2, !dbg !21
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1065 = or disjoint i64 %address.local1058, 66, !dbg !21
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1064, i64 %address.local1065, !dbg !21
  store half %v2369, ptr addrspace(1) %address.local1066, align 2, !dbg !21
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1068 = or disjoint i64 %address.local1058, 67, !dbg !21
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1068, !dbg !21
  store half %v2368, ptr addrspace(1) %address.local1069, align 2, !dbg !21
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1071 = or disjoint i64 %address.local1058, 80, !dbg !21
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1071, !dbg !21
  store half %v2362, ptr addrspace(1) %address.local1072, align 2, !dbg !21
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1074 = or disjoint i64 %address.local1058, 81, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1074, !dbg !21
  store half %v2361, ptr addrspace(1) %address.local1075, align 2, !dbg !21
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1077 = or disjoint i64 %address.local1058, 82, !dbg !21
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1076, i64 %address.local1077, !dbg !21
  store half %v2360, ptr addrspace(1) %address.local1078, align 2, !dbg !21
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1080 = or disjoint i64 %address.local1058, 83, !dbg !21
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1080, !dbg !21
  store half %v2359, ptr addrspace(1) %address.local1081, align 2, !dbg !21
  %address.local1082 = shl i64 %address.local1053, 7, !dbg !21
  %address.local1083 = add i64 %address.local1082, 8192, !dbg !21
  %address.local1084 = add nuw nsw i64 %v2372, %address.local1083, !dbg !21
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1085, i64 %address.local1059, !dbg !21
  store half %v2353, ptr addrspace(1) %address.local1086, align 2, !dbg !21
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1087, i64 %address.local1062, !dbg !21
  store half %v2352, ptr addrspace(1) %address.local1088, align 2, !dbg !21
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1065, !dbg !21
  store half %v2351, ptr addrspace(1) %address.local1090, align 2, !dbg !21
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1091, i64 %address.local1068, !dbg !21
  store half %v2350, ptr addrspace(1) %address.local1092, align 2, !dbg !21
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1071, !dbg !21
  store half %v2344, ptr addrspace(1) %address.local1094, align 2, !dbg !21
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1074, !dbg !21
  store half %v2343, ptr addrspace(1) %address.local1096, align 2, !dbg !21
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1097, i64 %address.local1077, !dbg !21
  store half %v2342, ptr addrspace(1) %address.local1098, align 2, !dbg !21
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1080, !dbg !21
  store half %v2341, ptr addrspace(1) %address.local1100, align 2, !dbg !21
  br label %v2373, !dbg !21

v2373:                                             ; preds = %v2335, %v2334
  br i1 true, label %v2374, label %v2412, !dbg !21

v2374:                                             ; preds = %v2373
  %v2375 = insertelement <4 x float> poison, float %v2145, i64 0, !dbg !19
  %v2376 = insertelement <4 x float> %v2375, float %v2151, i64 1, !dbg !19
  %v2377 = insertelement <4 x float> %v2376, float %v2146, i64 2, !dbg !19
  %v2378 = insertelement <4 x float> %v2377, float %v2152, i64 3, !dbg !19
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v2124, !dbg !19
  %normalized121 = fmul <4 x float> %v2378, %row.reciprocal120, !dbg !19
  %v2379 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %v2380 = extractelement <4 x half> %v2379, i64 3, !dbg !20
  %v2381 = extractelement <4 x half> %v2379, i64 2, !dbg !20
  %v2382 = extractelement <4 x half> %v2379, i64 1, !dbg !20
  %v2383 = extractelement <4 x half> %v2379, i64 0, !dbg !20
  %v2384 = insertelement <4 x float> poison, float %v2034, i64 0, !dbg !19
  %v2385 = insertelement <4 x float> %v2384, float %v2040, i64 1, !dbg !19
  %v2386 = insertelement <4 x float> %v2385, float %v2035, i64 2, !dbg !19
  %v2387 = insertelement <4 x float> %v2386, float %v2041, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %v2387, %row.reciprocal120, !dbg !19
  %v2388 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %v2389 = extractelement <4 x half> %v2388, i64 3, !dbg !20
  %v2390 = extractelement <4 x half> %v2388, i64 2, !dbg !20
  %v2391 = extractelement <4 x half> %v2388, i64 1, !dbg !20
  %v2392 = extractelement <4 x half> %v2388, i64 0, !dbg !20
  %v2393 = insertelement <4 x float> poison, float %v1927, i64 0, !dbg !19
  %v2394 = insertelement <4 x float> %v2393, float %v1933, i64 1, !dbg !19
  %v2395 = insertelement <4 x float> %v2394, float %v1928, i64 2, !dbg !19
  %v2396 = insertelement <4 x float> %v2395, float %v1934, i64 3, !dbg !19
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1906, !dbg !19
  %normalized124 = fmul <4 x float> %v2396, %row.reciprocal123, !dbg !19
  %v2397 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %v2398 = extractelement <4 x half> %v2397, i64 3, !dbg !20
  %v2399 = extractelement <4 x half> %v2397, i64 2, !dbg !20
  %v2400 = extractelement <4 x half> %v2397, i64 1, !dbg !20
  %v2401 = extractelement <4 x half> %v2397, i64 0, !dbg !20
  %v2402 = insertelement <4 x float> poison, float %v1816, i64 0, !dbg !19
  %v2403 = insertelement <4 x float> %v2402, float %v1822, i64 1, !dbg !19
  %v2404 = insertelement <4 x float> %v2403, float %v1817, i64 2, !dbg !19
  %v2405 = insertelement <4 x float> %v2404, float %v1823, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %v2405, %row.reciprocal123, !dbg !19
  %v2406 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %v2407 = extractelement <4 x half> %v2406, i64 3, !dbg !20
  %v2408 = extractelement <4 x half> %v2406, i64 2, !dbg !20
  %v2409 = extractelement <4 x half> %v2406, i64 1, !dbg !20
  %v2410 = extractelement <4 x half> %v2406, i64 0, !dbg !20
  %v2411 = shl nuw nsw i64 %v24, 19, !dbg !21
  %address.thread1101 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1102 = zext nneg i32 %address.thread1101 to i64, !dbg !8
  %address.local1103 = add i64 %v25, %address.local1102, !dbg !21
  %address.local1104 = lshr i64 %address.local1102, 6, !dbg !11
  %address.local1105 = shl nuw nsw i64 %address.local1104, 4, !dbg !21
  %address.local1106 = add i64 %address.local1103, %address.local1105, !dbg !21
  %address.local1107 = and i64 %address.local1102, 240, !dbg !21
  %address.local1108 = sub i64 %address.local1106, %address.local1107, !dbg !21
  %address.local1109 = add i64 %address.local1108, %v26, !dbg !21
  %address.local1110 = shl nuw nsw i64 %address.local1109, 7, !dbg !21
  %address.local1111 = add nuw nsw i64 %v2411, %address.local1110, !dbg !21
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1113 = lshr i64 %address.local1102, 2, !dbg !11
  %address.local1114 = and i64 %address.local1113, 12, !dbg !11
  %address.local1115 = or disjoint i64 %address.local1114, 96, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1115, !dbg !21
  store half %v2410, ptr addrspace(1) %address.local1116, align 2, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1118 = or disjoint i64 %address.local1114, 97, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1118, !dbg !21
  store half %v2409, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1121 = or disjoint i64 %address.local1114, 98, !dbg !21
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1121, !dbg !21
  store half %v2408, ptr addrspace(1) %address.local1122, align 2, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1124 = or disjoint i64 %address.local1114, 99, !dbg !21
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1124, !dbg !21
  store half %v2407, ptr addrspace(1) %address.local1125, align 2, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1127 = or disjoint i64 %address.local1114, 112, !dbg !21
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1126, i64 %address.local1127, !dbg !21
  store half %v2401, ptr addrspace(1) %address.local1128, align 2, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1130 = or disjoint i64 %address.local1114, 113, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1130, !dbg !21
  store half %v2400, ptr addrspace(1) %address.local1131, align 2, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1133 = or disjoint i64 %address.local1114, 114, !dbg !21
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1132, i64 %address.local1133, !dbg !21
  store half %v2399, ptr addrspace(1) %address.local1134, align 2, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1136 = or i64 %address.local1113, 115, !dbg !21
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1136, !dbg !21
  store half %v2398, ptr addrspace(1) %address.local1137, align 2, !dbg !21
  %address.local1138 = shl i64 %address.local1109, 7, !dbg !21
  %address.local1139 = add i64 %address.local1138, 8192, !dbg !21
  %address.local1140 = add nuw nsw i64 %v2411, %address.local1139, !dbg !21
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1115, !dbg !21
  store half %v2392, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1118, !dbg !21
  store half %v2391, ptr addrspace(1) %address.local1144, align 2, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1121, !dbg !21
  store half %v2390, ptr addrspace(1) %address.local1146, align 2, !dbg !21
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1124, !dbg !21
  store half %v2389, ptr addrspace(1) %address.local1148, align 2, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1127, !dbg !21
  store half %v2383, ptr addrspace(1) %address.local1150, align 2, !dbg !21
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1130, !dbg !21
  store half %v2382, ptr addrspace(1) %address.local1152, align 2, !dbg !21
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1133, !dbg !21
  store half %v2381, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1136, !dbg !21
  store half %v2380, ptr addrspace(1) %address.local1156, align 2, !dbg !21
  br label %v2412, !dbg !21

v2412:                                             ; preds = %v2374, %v2373
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 70, column: 22, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 77, column: 19, scope: !4)
!14 = !DILocation(line: 102, column: 23, scope: !4)
!15 = !DILocation(line: 0, scope: !4)
!16 = !DILocation(line: 71, column: 22, scope: !4)
!17 = !DILocation(line: 73, column: 26, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!23 = !DILocation(line: 79, column: 24, scope: !4)
!24 = !DILocation(line: 80, column: 23, scope: !4)
!25 = !DILocation(line: 81, column: 26, scope: !4)
!26 = !DILocation(line: 88, column: 21, scope: !4)
!27 = !DILocation(line: 89, column: 20, scope: !4)
!28 = !DILocation(line: 90, column: 22, scope: !4)
!29 = !DILocation(line: 91, column: 23, scope: !4)
!30 = !DILocation(line: 92, column: 20, scope: !4)
!31 = !DILocation(line: 93, column: 22, scope: !4)
!32 = !DILocation(line: 94, column: 22, scope: !4)
!33 = !DILocation(line: 95, column: 20, scope: !4)
!34 = !DILocation(line: 96, column: 16, scope: !4)
!35 = !DILocation(line: 103, column: 22, scope: !4)
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
