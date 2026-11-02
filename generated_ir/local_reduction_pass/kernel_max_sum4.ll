; ModuleID = 'generated_ir/online_softmax_mask_bypass/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !37 !reqd_work_group_size !65 {
  %v5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !38
  %v6 = sext i32 %v5 to i64, !dbg !38
  %v7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !38
  %v8 = sext i32 %v7 to i64, !dbg !38
  %v9 = shl nsw i64 %v8, 5, !dbg !38
  %v10 = add nsw i64 %v9, %v6, !dbg !38
  %v11 = lshr i64 %v10, 5, !dbg !38
  %v12 = and i64 %v11, 576460752303423484, !dbg !38
  %v13 = and i64 %v6, 3, !dbg !38
  %v14 = or disjoint i64 %v12, %v13, !dbg !38
  %v15 = sub nsw i64 31, %v14, !dbg !38
  %v16 = lshr i64 %v10, 2, !dbg !38
  %v17 = icmp ugt i64 %v10, 1023, !dbg !40
  %v18 = add nsw i64 %v14, -32, !dbg !40
  %v19 = select i1 %v17, i64 %v18, i64 %v15, !dbg !40
  %v20 = sdiv i64 %v19, 32, !dbg !40
  %v21 = sext i1 %v17 to i64, !dbg !40
  %v22 = xor i64 %v20, %v21, !dbg !40
  %v23 = add nsw i64 %v16, %v22, !dbg !40
  %v24 = and i64 %v23, 31, !dbg !40
  %v25 = shl nsw i64 %v15, 7, !dbg !40
  %v26 = mul nsw i64 %v22, -4096, !dbg !40
  %v27 = add i64 %v25, %v26, !dbg !40
  %v28 = shl nuw nsw i64 %v24, 19, !dbg !40
  %v29 = shl nsw i64 %v27, 7, !dbg !40
  %v30 = add i64 %v28, %v29, !dbg !40
  %v31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !40, !range !67
  %v32 = zext nneg i32 %v31 to i64, !dbg !40
  %v33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %v30, !dbg !40
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !40, !range !67, !frisk.register_lifetimes !34
  %address.local = zext nneg i32 %address.thread to i64, !dbg !40
  %address.local339 = shl nuw nsw i64 %address.local, 3, !dbg !40
  %address.local340 = and i64 %address.local339, 1536, !dbg !40
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !40
  %address.local342 = ptrtoint ptr addrspace(3) %address.local341 to i64, !dbg !40
  %address.local343 = trunc i64 %address.local342 to i32, !dbg !40
  %v34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !40
  %v35 = zext i32 %v34 to i64, !dbg !40
  %v36 = inttoptr i64 %v35 to ptr addrspace(3), !dbg !40
  %address.local344 = shl nuw nsw i64 %address.local, 7, !dbg !40
  %address.local345 = and i64 %address.local344, 16256, !dbg !40
  %address.local346 = lshr i64 %address.local, 4, !dbg !40
  %address.local347 = and i64 %address.local346, 8, !dbg !40
  %address.local348 = or disjoint i64 %address.local345, %address.local347, !dbg !40
  %address.local349 = getelementptr [2 x i8], ptr addrspace(1) %v33, i64 %address.local348, !dbg !40
  %v37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %v38 = ptrtoint ptr addrspace(1) %address.local349 to i64, !dbg !40
  %v39 = sub i64 %v38, %v37, !dbg !40
  %dma.buffer.offset1612 = trunc i64 %v39 to i32, !dbg !40
  %v40 = trunc i64 %v37 to i32, !dbg !40
  %v41 = insertelement <4 x i32> poison, i32 %v40, i64 0, !dbg !40
  %v42 = lshr i64 %v37, 32, !dbg !40
  %v43 = trunc i64 %v42 to i32, !dbg !40
  %v44 = insertelement <4 x i32> %v41, i32 %v43, i64 1, !dbg !40
  %v45 = insertelement <4 x i32> %v44, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1613 = insertelement <4 x i32> %v45, i32 131072, i64 3, !dbg !40
  %v46 = ptrtoint ptr addrspace(3) %v36 to i32, !dbg !40
  %dma.buffer.lds1614 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v46), !dbg !40
  %v47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1612, <4 x i32> %dma.buffer.rsrc1613) #10, !dbg !40
  %address.local350 = or disjoint i64 %address.local, 256, !dbg !40
  %address.local351 = shl nuw nsw i64 %address.local350, 3, !dbg !40
  %address.local352 = and i64 %address.local351, 3584, !dbg !40
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !40
  %address.local354 = ptrtoint ptr addrspace(3) %address.local353 to i64, !dbg !40
  %address.local355 = trunc i64 %address.local354 to i32, !dbg !40
  %v48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !40
  %v49 = zext i32 %v48 to i64, !dbg !40
  %v50 = inttoptr i64 %v49 to ptr addrspace(3), !dbg !40
  %address.local356 = shl nuw nsw i64 %address.local, 7, !dbg !40
  %address.local357 = and i64 %address.local356, 16256, !dbg !40
  %address.local358 = lshr i64 %address.local350, 4, !dbg !40
  %address.local359 = and i64 %address.local358, 24, !dbg !40
  %address.local360 = or disjoint i64 %address.local357, %address.local359, !dbg !40
  %address.local361 = getelementptr [2 x i8], ptr addrspace(1) %v33, i64 %address.local360, !dbg !40
  %v51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %v52 = ptrtoint ptr addrspace(1) %address.local361 to i64, !dbg !40
  %v53 = sub i64 %v52, %v51, !dbg !40
  %dma.buffer.offset1609 = trunc i64 %v53 to i32, !dbg !40
  %v54 = trunc i64 %v51 to i32, !dbg !40
  %v55 = insertelement <4 x i32> poison, i32 %v54, i64 0, !dbg !40
  %v56 = lshr i64 %v51, 32, !dbg !40
  %v57 = trunc i64 %v56 to i32, !dbg !40
  %v58 = insertelement <4 x i32> %v55, i32 %v57, i64 1, !dbg !40
  %v59 = insertelement <4 x i32> %v58, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1610 = insertelement <4 x i32> %v59, i32 131072, i64 3, !dbg !40
  %v60 = ptrtoint ptr addrspace(3) %v50 to i32, !dbg !40
  %dma.buffer.lds1611 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v60), !dbg !40
  %v61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1609, <4 x i32> %dma.buffer.rsrc1610) #10, !dbg !40
  %v62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !38
  %v63 = sext i32 %v62 to i64, !dbg !38
  %v64 = shl i64 %v15, 7, !dbg !41
  %v65 = lshr i64 %v32, 6, !dbg !42
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %v66 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %v30, !dbg !40
  %v67 = getelementptr i8, ptr addrspace(1) %v66, i64 64, !dbg !40
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !40
  %address.local363 = ptrtoint ptr addrspace(3) %address.local362 to i64, !dbg !40
  %address.local364 = trunc i64 %address.local363 to i32, !dbg !40
  %v68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !40
  %v69 = zext i32 %v68 to i64, !dbg !40
  %v70 = inttoptr i64 %v69 to ptr addrspace(3), !dbg !40
  %address.local365 = getelementptr [2 x i8], ptr addrspace(1) %v67, i64 %address.local348, !dbg !40
  %v71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %v72 = ptrtoint ptr addrspace(1) %address.local365 to i64, !dbg !40
  %v73 = sub i64 %v72, %v71, !dbg !40
  %dma.buffer.offset1606 = trunc i64 %v73 to i32, !dbg !40
  %v74 = trunc i64 %v71 to i32, !dbg !40
  %v75 = insertelement <4 x i32> poison, i32 %v74, i64 0, !dbg !40
  %v76 = lshr i64 %v71, 32, !dbg !40
  %v77 = trunc i64 %v76 to i32, !dbg !40
  %v78 = insertelement <4 x i32> %v75, i32 %v77, i64 1, !dbg !40
  %v79 = insertelement <4 x i32> %v78, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1607 = insertelement <4 x i32> %v79, i32 131072, i64 3, !dbg !40
  %v80 = ptrtoint ptr addrspace(3) %v70 to i32, !dbg !40
  %dma.buffer.lds1608 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v80), !dbg !40
  %v81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1606, <4 x i32> %dma.buffer.rsrc1607) #10, !dbg !40
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !40
  %address.local367 = ptrtoint ptr addrspace(3) %address.local366 to i64, !dbg !40
  %address.local368 = trunc i64 %address.local367 to i32, !dbg !40
  %v82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !40
  %v83 = zext i32 %v82 to i64, !dbg !40
  %v84 = inttoptr i64 %v83 to ptr addrspace(3), !dbg !40
  %address.local369 = getelementptr [2 x i8], ptr addrspace(1) %v67, i64 %address.local360, !dbg !40
  %v85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %v86 = ptrtoint ptr addrspace(1) %address.local369 to i64, !dbg !40
  %v87 = sub i64 %v86, %v85, !dbg !40
  %dma.buffer.offset1603 = trunc i64 %v87 to i32, !dbg !40
  %v88 = trunc i64 %v85 to i32, !dbg !40
  %v89 = insertelement <4 x i32> poison, i32 %v88, i64 0, !dbg !40
  %v90 = lshr i64 %v85, 32, !dbg !40
  %v91 = trunc i64 %v90 to i32, !dbg !40
  %v92 = insertelement <4 x i32> %v89, i32 %v91, i64 1, !dbg !40
  %v93 = insertelement <4 x i32> %v92, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1604 = insertelement <4 x i32> %v93, i32 131072, i64 3, !dbg !40
  %v94 = ptrtoint ptr addrspace(3) %v84 to i32, !dbg !40
  %dma.buffer.lds1605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v94), !dbg !40
  %v95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1603, <4 x i32> %dma.buffer.rsrc1604) #10, !dbg !40
  %address.local370 = lshr i32 %address.thread, 2, !dbg !42
  %address.local371 = and i32 %address.local370, 4, !dbg !42
  %address.local372 = shl nuw nsw i32 %address.thread, 1, !dbg !42
  %address.local373 = and i32 %address.local372, 384, !dbg !42
  %address.local374 = or disjoint i32 %address.local371, %address.local373, !dbg !42
  %address.local375 = shl nuw nsw i32 %address.thread, 3, !dbg !42
  %address.local376 = and i32 %address.local375, 120, !dbg !42
  %address.local377 = or disjoint i32 %address.local374, %address.local376, !dbg !42
  %address.local378 = shl nuw nsw i32 %address.thread, 5, !dbg !42
  %address.local379 = and i32 %address.local378, 1024, !dbg !42
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !42
  %address.local381 = zext nneg i32 %address.local380 to i64, !dbg !42
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !42
  %v96 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !42
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !42
  %v97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !42
  %v98 = fmul <4 x half> %v96, splat (half 1.275630e-01), !dbg !42
  %lds.const.gep1413 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !42
  %v99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !42
  %v100 = fmul <4 x half> %v97, splat (half 1.275630e-01), !dbg !42
  %lds.const.gep1414 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !42
  %v101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !42
  %v102 = fmul <4 x half> %v99, splat (half 1.275630e-01), !dbg !42
  %v103 = fmul <4 x half> %v101, splat (half 1.275630e-01), !dbg !42
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %v104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %v30, !dbg !40
  %v105 = getelementptr i8, ptr addrspace(1) %v104, i64 128, !dbg !40
  %v106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !40
  %v107 = zext i32 %v106 to i64, !dbg !40
  %v108 = inttoptr i64 %v107 to ptr addrspace(3), !dbg !40
  %address.local425 = getelementptr [2 x i8], ptr addrspace(1) %v105, i64 %address.local348, !dbg !40
  %v109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %v110 = ptrtoint ptr addrspace(1) %address.local425 to i64, !dbg !40
  %v111 = sub i64 %v110, %v109, !dbg !40
  %dma.buffer.offset1600 = trunc i64 %v111 to i32, !dbg !40
  %v112 = trunc i64 %v109 to i32, !dbg !40
  %v113 = insertelement <4 x i32> poison, i32 %v112, i64 0, !dbg !40
  %v114 = lshr i64 %v109, 32, !dbg !40
  %v115 = trunc i64 %v114 to i32, !dbg !40
  %v116 = insertelement <4 x i32> %v113, i32 %v115, i64 1, !dbg !40
  %v117 = insertelement <4 x i32> %v116, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1601 = insertelement <4 x i32> %v117, i32 131072, i64 3, !dbg !40
  %v118 = ptrtoint ptr addrspace(3) %v108 to i32, !dbg !40
  %dma.buffer.lds1602 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v118), !dbg !40
  %v119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1602, i32 %dma.buffer.offset1600, <4 x i32> %dma.buffer.rsrc1601) #10, !dbg !40
  %v120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !40
  %v121 = zext i32 %v120 to i64, !dbg !40
  %v122 = inttoptr i64 %v121 to ptr addrspace(3), !dbg !40
  %address.local426 = getelementptr [2 x i8], ptr addrspace(1) %v105, i64 %address.local360, !dbg !40
  %v123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %v124 = ptrtoint ptr addrspace(1) %address.local426 to i64, !dbg !40
  %v125 = sub i64 %v124, %v123, !dbg !40
  %dma.buffer.offset1597 = trunc i64 %v125 to i32, !dbg !40
  %v126 = trunc i64 %v123 to i32, !dbg !40
  %v127 = insertelement <4 x i32> poison, i32 %v126, i64 0, !dbg !40
  %v128 = lshr i64 %v123, 32, !dbg !40
  %v129 = trunc i64 %v128 to i32, !dbg !40
  %v130 = insertelement <4 x i32> %v127, i32 %v129, i64 1, !dbg !40
  %v131 = insertelement <4 x i32> %v130, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1598 = insertelement <4 x i32> %v131, i32 131072, i64 3, !dbg !40
  %v132 = ptrtoint ptr addrspace(3) %v122 to i32, !dbg !40
  %dma.buffer.lds1599 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v132), !dbg !40
  %v133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1599, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1598) #10, !dbg !40
  %lds.const.gep1415 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !42
  %v134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !42
  %lds.const.gep1416 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !42
  %v135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !42
  %lds.const.gep1417 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !42
  %v136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !42
  %v137 = fmul <4 x half> %v134, splat (half 1.275630e-01), !dbg !42
  %lds.const.gep1418 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !42
  %v138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !42
  %v139 = fmul <4 x half> %v135, splat (half 1.275630e-01), !dbg !42
  %v140 = fmul <4 x half> %v136, splat (half 1.275630e-01), !dbg !42
  %v141 = fmul <4 x half> %v138, splat (half 1.275630e-01), !dbg !42
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %v142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %v30, !dbg !40
  %v143 = getelementptr i8, ptr addrspace(1) %v142, i64 192, !dbg !40
  %v144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !40
  %v145 = zext i32 %v144 to i64, !dbg !40
  %v146 = inttoptr i64 %v145 to ptr addrspace(3), !dbg !40
  %address.local431 = getelementptr [2 x i8], ptr addrspace(1) %v143, i64 %address.local348, !dbg !40
  %v147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %v148 = ptrtoint ptr addrspace(1) %address.local431 to i64, !dbg !40
  %v149 = sub i64 %v148, %v147, !dbg !40
  %dma.buffer.offset1594 = trunc i64 %v149 to i32, !dbg !40
  %v150 = trunc i64 %v147 to i32, !dbg !40
  %v151 = insertelement <4 x i32> poison, i32 %v150, i64 0, !dbg !40
  %v152 = lshr i64 %v147, 32, !dbg !40
  %v153 = trunc i64 %v152 to i32, !dbg !40
  %v154 = insertelement <4 x i32> %v151, i32 %v153, i64 1, !dbg !40
  %v155 = insertelement <4 x i32> %v154, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1595 = insertelement <4 x i32> %v155, i32 131072, i64 3, !dbg !40
  %v156 = ptrtoint ptr addrspace(3) %v146 to i32, !dbg !40
  %dma.buffer.lds1596 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v156), !dbg !40
  %v157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1596, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1595) #10, !dbg !40
  %v158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !40
  %v159 = zext i32 %v158 to i64, !dbg !40
  %v160 = inttoptr i64 %v159 to ptr addrspace(3), !dbg !40
  %address.local432 = getelementptr [2 x i8], ptr addrspace(1) %v143, i64 %address.local360, !dbg !40
  %v161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %v162 = ptrtoint ptr addrspace(1) %address.local432 to i64, !dbg !40
  %v163 = sub i64 %v162, %v161, !dbg !40
  %dma.buffer.offset1591 = trunc i64 %v163 to i32, !dbg !40
  %v164 = trunc i64 %v161 to i32, !dbg !40
  %v165 = insertelement <4 x i32> poison, i32 %v164, i64 0, !dbg !40
  %v166 = lshr i64 %v161, 32, !dbg !40
  %v167 = trunc i64 %v166 to i32, !dbg !40
  %v168 = insertelement <4 x i32> %v165, i32 %v167, i64 1, !dbg !40
  %v169 = insertelement <4 x i32> %v168, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1592 = insertelement <4 x i32> %v169, i32 131072, i64 3, !dbg !40
  %v170 = ptrtoint ptr addrspace(3) %v160 to i32, !dbg !40
  %dma.buffer.lds1593 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v170), !dbg !40
  %v171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1592) #10, !dbg !40
  %v172 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !42
  %lds.const.gep1419 = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !42
  %v173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1419, align 8, !dbg !42
  %lds.const.gep1420 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !42
  %v174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1420, align 8, !dbg !42
  %v175 = fmul <4 x half> %v172, splat (half 1.275630e-01), !dbg !42
  %lds.const.gep1421 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !42
  %v176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1421, align 8, !dbg !42
  %v177 = fmul <4 x half> %v173, splat (half 1.275630e-01), !dbg !42
  %v178 = fmul <4 x half> %v174, splat (half 1.275630e-01), !dbg !42
  %v179 = fmul <4 x half> %v176, splat (half 1.275630e-01), !dbg !42
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %lds.const.gep1422 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !42
  %v180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1422, align 8, !dbg !42
  %lds.const.gep1423 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !42
  %v181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1423, align 8, !dbg !42
  %lds.const.gep1424 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !42
  %v182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1424, align 8, !dbg !42
  %v183 = fmul <4 x half> %v180, splat (half 1.275630e-01), !dbg !42
  %lds.const.gep1425 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !42
  %v184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1425, align 8, !dbg !42
  %v185 = fmul <4 x half> %v181, splat (half 1.275630e-01), !dbg !42
  %v186 = fmul <4 x half> %v182, splat (half 1.275630e-01), !dbg !42
  %v187 = fmul <4 x half> %v184, splat (half 1.275630e-01), !dbg !42
  fence syncscope("workgroup") release, !dbg !39
  call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %v188 = or disjoint i64 %v64, 127, !dbg !43
  %v189 = add i64 %v64, 128, !dbg !43
  %v190 = icmp uge i64 %v189, %v188, !dbg !43
  %v191 = sub nsw i64 32, %v14, !dbg !45
  %v192 = shl i64 %v16, 7, !dbg !46
  %v193 = and i64 %v192, 3968, !dbg !46
  %v194 = shl nsw i64 %v63, 12, !dbg !46
  %v195 = or disjoint i64 %v193, %v194, !dbg !46
  %v196 = lshr i64 %v32, 2, !dbg !47
  %v197 = shl nuw nsw i64 %v32, 2, !dbg !47
  %v198 = mul nsw i64 %v196, -16, !dbg !47
  %v199 = add nsw i64 %v197, %v198, !dbg !47
  %v200 = and i32 %v31, 1, !dbg !47
  %index.compact279 = zext nneg i32 %v200 to i64, !dbg !47
  %v201 = or disjoint i64 %v199, %index.compact279, !dbg !47
  %v202 = add nuw nsw i64 %v201, 1, !dbg !47
  %v203 = sdiv i64 %v202, 2, !dbg !47
  br label %v204, !dbg !39

v204:                                             ; preds = %v848, %4
  %v205 = phi i64 [ %v1779, %v848 ], [ 0, %4 ], !dbg !39
  %v206 = phi [2 x <8 x float>] [ %v1764, %v848 ], [ zeroinitializer, %4 ], !dbg !39
  %v207 = phi [2 x <1 x float>] [ %v1370, %v848 ], [ zeroinitializer, %4 ], !dbg !39
  %v208 = phi [2 x <1 x float>] [ %v1766, %v848 ], [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], !dbg !39
  %v209 = phi [2 x <8 x float>] [ %v1770, %v848 ], [ zeroinitializer, %4 ], !dbg !39
  %v210 = phi [2 x <8 x float>] [ %v1774, %v848 ], [ zeroinitializer, %4 ], !dbg !39
  %v211 = phi [2 x <8 x float>] [ %v1778, %v848 ], [ zeroinitializer, %4 ], !dbg !39
  %v212 = icmp slt i64 %v205, %v191, !dbg !39
  br i1 %v212, label %v213, label %v1780, !dbg !39

v213:                                             ; preds = %v204
  %v214 = shl nsw i64 %v205, 7, !dbg !39
  %v215 = lshr i64 %v205, 12, !dbg !46
  %v216 = add nuw nsw i64 %v215, %v16, !dbg !46
  %v217 = lshr i64 %v205, 5, !dbg !46
  %v218 = and i64 %v217, 127, !dbg !46
  %v219 = shl i64 %v216, 19, !dbg !46
  %v220 = and i64 %v219, 16252928, !dbg !46
  %v221 = shl i64 %v217, 19, !dbg !46
  %v222 = shl i64 %v205, 14, !dbg !46
  %v223 = sub i64 %v222, %v221, !dbg !46
  %v224 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %v220, !dbg !46
  %v225 = getelementptr [2 x i8], ptr addrspace(1) %v224, i64 %v218, !dbg !46
  %v226 = getelementptr [2 x i8], ptr addrspace(1) %v225, i64 %v223, !dbg !46
  %address.thread433 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !46, !range !67, !frisk.register_lifetimes !34
  %address.local434 = zext nneg i32 %address.thread433 to i64, !dbg !40
  %address.local435 = shl nuw nsw i64 %address.local434, 3, !dbg !40
  %address.local436 = and i64 %address.local435, 1536, !dbg !40
  %address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local436, !dbg !40
  %address.local438 = ptrtoint ptr addrspace(3) %address.local437 to i64, !dbg !40
  %address.local439 = trunc i64 %address.local438 to i32, !dbg !40
  %v227 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !46
  %v228 = zext i32 %v227 to i64, !dbg !46
  %v229 = inttoptr i64 %v228 to ptr addrspace(3), !dbg !46
  %address.local440 = lshr i64 %address.local434, 4, !dbg !46
  %address.local441 = and i64 %address.local440, 8, !dbg !46
  %address.local442 = and i64 %address.local434, 123, !dbg !46
  %address.local443 = lshr i64 %address.local434, 2, !dbg !46
  %address.local444 = and i64 %address.local443, 4, !dbg !46
  %address.local445 = add nuw nsw i64 %address.local444, %address.local434, !dbg !46
  %address.local446 = and i64 %address.local445, 4, !dbg !46
  %address.local447 = or disjoint i64 %address.local442, %address.local446, !dbg !46
  %address.local448 = shl nuw nsw i64 %address.local447, 7, !dbg !46
  %address.local449 = or disjoint i64 %address.local441, %address.local448, !dbg !46
  %address.local450 = getelementptr [2 x i8], ptr addrspace(1) %v226, i64 %address.local449, !dbg !46
  %v230 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %v231 = ptrtoint ptr addrspace(1) %address.local450 to i64, !dbg !46
  %v232 = sub i64 %v231, %v230, !dbg !46
  %dma.buffer.offset1588 = trunc i64 %v232 to i32, !dbg !46
  %v233 = trunc i64 %v230 to i32, !dbg !46
  %v234 = insertelement <4 x i32> poison, i32 %v233, i64 0, !dbg !46
  %v235 = lshr i64 %v230, 32, !dbg !46
  %v236 = trunc i64 %v235 to i32, !dbg !46
  %v237 = insertelement <4 x i32> %v234, i32 %v236, i64 1, !dbg !46
  %v238 = insertelement <4 x i32> %v237, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1589 = insertelement <4 x i32> %v238, i32 131072, i64 3, !dbg !46
  %v239 = ptrtoint ptr addrspace(3) %v229 to i32, !dbg !46
  %dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v239), !dbg !46
  %v240 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #10, !dbg !46
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !40
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !40
  %address.local453 = and i64 %address.local452, 3584, !dbg !40
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !40
  %address.local455 = ptrtoint ptr addrspace(3) %address.local454 to i64, !dbg !40
  %address.local456 = trunc i64 %address.local455 to i32, !dbg !40
  %v241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !46
  %v242 = zext i32 %v241 to i64, !dbg !46
  %v243 = inttoptr i64 %v242 to ptr addrspace(3), !dbg !46
  %address.local457 = lshr i64 %address.local451, 4, !dbg !46
  %address.local458 = and i64 %address.local457, 24, !dbg !46
  %address.local459 = and i64 %address.local434, 123, !dbg !46
  %address.local460 = lshr i64 %address.local434, 2, !dbg !46
  %address.local461 = and i64 %address.local460, 4, !dbg !46
  %address.local462 = add nuw nsw i64 %address.local461, %address.local434, !dbg !46
  %address.local463 = and i64 %address.local462, 4, !dbg !46
  %address.local464 = or disjoint i64 %address.local459, %address.local463, !dbg !46
  %address.local465 = shl nuw nsw i64 %address.local464, 7, !dbg !46
  %address.local466 = or disjoint i64 %address.local458, %address.local465, !dbg !46
  %address.local467 = getelementptr [2 x i8], ptr addrspace(1) %v226, i64 %address.local466, !dbg !46
  %v244 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %v245 = ptrtoint ptr addrspace(1) %address.local467 to i64, !dbg !46
  %v246 = sub i64 %v245, %v244, !dbg !46
  %dma.buffer.offset1585 = trunc i64 %v246 to i32, !dbg !46
  %v247 = trunc i64 %v244 to i32, !dbg !46
  %v248 = insertelement <4 x i32> poison, i32 %v247, i64 0, !dbg !46
  %v249 = lshr i64 %v244, 32, !dbg !46
  %v250 = trunc i64 %v249 to i32, !dbg !46
  %v251 = insertelement <4 x i32> %v248, i32 %v250, i64 1, !dbg !46
  %v252 = insertelement <4 x i32> %v251, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1586 = insertelement <4 x i32> %v252, i32 131072, i64 3, !dbg !46
  %v253 = ptrtoint ptr addrspace(3) %v243 to i32, !dbg !46
  %dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v253), !dbg !46
  %v254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1586) #10, !dbg !46
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %v255 = add nuw i64 %v205, 1024, !dbg !46
  %v256 = icmp slt i64 %v255, 0, !dbg !46
  %v257 = sub nuw i64 -1025, %v205, !dbg !46
  %v258 = select i1 %v256, i64 %v257, i64 %v255, !dbg !46
  %v259 = sdiv i64 %v258, 4096, !dbg !46
  %.lobit37 = ashr i64 %v255, 63, !dbg !46
  %v260 = xor i64 %v259, %.lobit37, !dbg !46
  %v261 = add nsw i64 %v260, %v16, !dbg !46
  %v262 = add nsw i64 %v217, %v195, !dbg !46
  %v263 = add nsw i64 %v262, 32, !dbg !46
  %v264 = icmp slt i64 %v262, -32, !dbg !46
  %v265 = sub nsw i64 -33, %v262, !dbg !46
  %v266 = select i1 %v264, i64 %v265, i64 %v263, !dbg !46
  %v267 = sdiv i64 %v266, 128, !dbg !46
  %v268 = sext i1 %v264 to i64, !dbg !46
  %v269 = xor i64 %v267, %v268, !dbg !46
  %v270 = shl nsw i64 %v269, 7, !dbg !46
  %v271 = sub i64 %v195, %v270, !dbg !46
  %v272 = shl i64 %v261, 19, !dbg !46
  %v273 = and i64 %v272, 16252928, !dbg !46
  %v274 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %v273, !dbg !46
  %v275 = getelementptr [2 x i8], ptr addrspace(1) %v274, i64 %v217, !dbg !46
  %v276 = getelementptr [2 x i8], ptr addrspace(1) %v275, i64 %v271, !dbg !46
  %v277 = getelementptr i8, ptr addrspace(1) %v276, i64 64, !dbg !46
  %v278 = getelementptr [2 x i8], ptr addrspace(1) %v277, i64 %v223, !dbg !46
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !40
  %address.local469 = ptrtoint ptr addrspace(3) %address.local468 to i64, !dbg !40
  %address.local470 = trunc i64 %address.local469 to i32, !dbg !40
  %v279 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !46
  %v280 = zext i32 %v279 to i64, !dbg !46
  %v281 = inttoptr i64 %v280 to ptr addrspace(3), !dbg !46
  %address.local471 = getelementptr [2 x i8], ptr addrspace(1) %v278, i64 %address.local449, !dbg !46
  %v282 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %v283 = ptrtoint ptr addrspace(1) %address.local471 to i64, !dbg !46
  %v284 = sub i64 %v283, %v282, !dbg !46
  %dma.buffer.offset1582 = trunc i64 %v284 to i32, !dbg !46
  %v285 = trunc i64 %v282 to i32, !dbg !46
  %v286 = insertelement <4 x i32> poison, i32 %v285, i64 0, !dbg !46
  %v287 = lshr i64 %v282, 32, !dbg !46
  %v288 = trunc i64 %v287 to i32, !dbg !46
  %v289 = insertelement <4 x i32> %v286, i32 %v288, i64 1, !dbg !46
  %v290 = insertelement <4 x i32> %v289, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1583 = insertelement <4 x i32> %v290, i32 131072, i64 3, !dbg !46
  %v291 = ptrtoint ptr addrspace(3) %v281 to i32, !dbg !46
  %dma.buffer.lds1584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v291), !dbg !46
  %v292 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1583) #10, !dbg !46
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !40
  %address.local473 = ptrtoint ptr addrspace(3) %address.local472 to i64, !dbg !40
  %address.local474 = trunc i64 %address.local473 to i32, !dbg !40
  %v293 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !46
  %v294 = zext i32 %v293 to i64, !dbg !46
  %v295 = inttoptr i64 %v294 to ptr addrspace(3), !dbg !46
  %address.local475 = getelementptr [2 x i8], ptr addrspace(1) %v278, i64 %address.local466, !dbg !46
  %v296 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %v297 = ptrtoint ptr addrspace(1) %address.local475 to i64, !dbg !46
  %v298 = sub i64 %v297, %v296, !dbg !46
  %dma.buffer.offset1579 = trunc i64 %v298 to i32, !dbg !46
  %v299 = trunc i64 %v296 to i32, !dbg !46
  %v300 = insertelement <4 x i32> poison, i32 %v299, i64 0, !dbg !46
  %v301 = lshr i64 %v296, 32, !dbg !46
  %v302 = trunc i64 %v301 to i32, !dbg !46
  %v303 = insertelement <4 x i32> %v300, i32 %v302, i64 1, !dbg !46
  %v304 = insertelement <4 x i32> %v303, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1580 = insertelement <4 x i32> %v304, i32 131072, i64 3, !dbg !46
  %v305 = ptrtoint ptr addrspace(3) %v295 to i32, !dbg !46
  %dma.buffer.lds1581 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v305), !dbg !46
  %v306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1580) #10, !dbg !46
  %address.local476 = mul nuw nsw i64 %address.local434, 160, !dbg !47
  %address.local477 = lshr i64 %address.local434, 2, !dbg !47
  %address.local478 = mul nsw i64 %address.local477, -640, !dbg !47
  %address.local479 = add nsw i64 %address.local476, %address.local478, !dbg !47
  %address.local480 = shl nuw nsw i64 %address.local434, 1, !dbg !47
  %address.local481 = and i64 %address.local480, 24, !dbg !47
  %address.local482 = or disjoint i64 %address.local479, %address.local481, !dbg !47
  %address.local483 = shl i64 %address.local434, 62, !dbg !47
  %address.local484 = ashr exact i64 %address.local483, 57, !dbg !47
  %address.local485 = and i64 %address.local484, -64, !dbg !47
  %address.local486 = add nsw i64 %address.local482, %address.local485, !dbg !47
  %address.local487 = lshr i64 %address.local434, 2, !dbg !42
  %address.local488 = and i64 %address.local487, 12, !dbg !42
  %address.local489 = add nsw i64 %address.local486, %address.local488, !dbg !47
  %address.local490 = and i64 %address.local434, 63, !dbg !42
  %address.local491 = lshr i64 %address.local490, 5, !dbg !42
  %address.local492 = mul nuw nsw i64 %address.local491, 1016, !dbg !42
  %address.local493 = add i64 %address.local489, %address.local492, !dbg !47
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local493, !dbg !47
  %v307 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !47
  %v308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v307, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !47
  %v309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !47
  %address.local505 = shl nuw nsw i32 %address.thread433, 1, !dbg !47
  %address.local506 = and i32 %address.local505, 24, !dbg !47
  %address.local507 = shl nuw nsw i32 %address.thread433, 8, !dbg !47
  %address.local508 = and i32 %address.local507, 768, !dbg !47
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !47
  %address.local510 = shl nuw nsw i32 %address.thread433, 5, !dbg !47
  %address.local511 = and i32 %address.local510, 96, !dbg !47
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !47
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !47
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %v203) #9, !dbg !47, !frisk.register_lifetimes !34
  %address.local514 = lshr i32 %address.thread433, 1, !dbg !47
  %address.local515 = and i32 %address.local514, 1, !dbg !47
  %address.local516 = zext nneg i32 %address.local515 to i64, !dbg !47
  %address.local517 = add nsw i64 %address.base, %address.local516, !dbg !47
  %address.local518 = shl i64 %address.local517, 6, !dbg !47
  %address.local519 = sub i64 %address.local513, %address.local518, !dbg !47
  %address.local520 = add i64 %address.local519, %address.local488, !dbg !47
  %address.local521 = add i64 %address.local520, %address.local492, !dbg !47
  %address.local522 = add i64 %address.local521, 32, !dbg !47
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local522, !dbg !47
  %v310 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !47
  %v311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v309, <4 x float> %v308) #9, !dbg !47
  %v312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v310, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !47
  %v313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !47
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !47
  %v314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !47
  %v315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v313, <4 x float> %v312) #9, !dbg !47
  %v316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v314, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !47
  %v317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !47
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !47
  %v318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !47
  %v319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v317, <4 x float> %v316) #9, !dbg !47
  %v320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v318, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !47
  %v321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !47
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !47
  %v322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !47
  %v323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v321, <4 x float> %v320) #9, !dbg !47
  %v324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v322, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !47
  %v325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !47
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !47
  %v326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !47
  %v327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v325, <4 x float> %v324) #9, !dbg !47
  %v328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v326, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !47
  %v329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !47
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !47
  %v330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !47
  %v331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v329, <4 x float> %v328) #9, !dbg !47
  %v332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v330, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !47
  %v333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !47
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !47
  %v334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !47
  %v335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v333, <4 x float> %v332) #9, !dbg !47
  %v336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v334, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !47
  %v337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !47
  %v338 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !47
  %v339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v337, <4 x float> %v336) #9, !dbg !47
  %v340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v338, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !47
  %v341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !47
  %v342 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !47
  %v343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v341, <4 x float> %v340) #9, !dbg !47
  %v344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v342, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !47
  %v345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !47
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !47
  %v346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !47
  %v347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v345, <4 x float> %v344) #9, !dbg !47
  %v348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v346, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1443 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !47
  %v349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1443, align 8, !dbg !47
  %lds.const.gep1444 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !47
  %v350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1444, align 8, !dbg !47
  %v351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v349, <4 x float> %v348) #9, !dbg !47
  %v352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v350, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1445 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !47
  %v353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1445, align 8, !dbg !47
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !47
  %v354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !47
  %v355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v353, <4 x float> %v352) #9, !dbg !47
  %v356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v354, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !47
  %v357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !47
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !47
  %v358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !47
  %v359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v357, <4 x float> %v356) #9, !dbg !47
  %v360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v358, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !47
  %v361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !47
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !47
  %v362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !47
  %v363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v361, <4 x float> %v360) #9, !dbg !47
  %v364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v362, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !47
  %v365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !47
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !47
  %v366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !47
  %v367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v365, <4 x float> %v364) #9, !dbg !47
  %v368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v366, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !47
  %v369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !47
  %v370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v369, <4 x float> %v368) #9, !dbg !47
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %v371 = add nuw i64 %v205, 2048, !dbg !46
  %v372 = icmp slt i64 %v371, 0, !dbg !46
  %v373 = sub nuw i64 -2049, %v205, !dbg !46
  %v374 = select i1 %v372, i64 %v373, i64 %v371, !dbg !46
  %v375 = sdiv i64 %v374, 4096, !dbg !46
  %.lobit38 = ashr i64 %v371, 63, !dbg !46
  %v376 = xor i64 %v375, %.lobit38, !dbg !46
  %v377 = add nsw i64 %v376, %v16, !dbg !46
  %v378 = add nsw i64 %v262, 64, !dbg !46
  %v379 = icmp slt i64 %v262, -64, !dbg !46
  %v380 = sub nsw i64 -65, %v262, !dbg !46
  %v381 = select i1 %v379, i64 %v380, i64 %v378, !dbg !46
  %v382 = sdiv i64 %v381, 128, !dbg !46
  %v383 = sext i1 %v379 to i64, !dbg !46
  %v384 = xor i64 %v382, %v383, !dbg !46
  %v385 = shl nsw i64 %v384, 7, !dbg !46
  %v386 = sub i64 %v195, %v385, !dbg !46
  %v387 = shl i64 %v377, 19, !dbg !46
  %v388 = and i64 %v387, 16252928, !dbg !46
  %v389 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %v388, !dbg !46
  %v390 = getelementptr [2 x i8], ptr addrspace(1) %v389, i64 %v217, !dbg !46
  %v391 = getelementptr [2 x i8], ptr addrspace(1) %v390, i64 %v386, !dbg !46
  %v392 = getelementptr i8, ptr addrspace(1) %v391, i64 128, !dbg !46
  %v393 = getelementptr [2 x i8], ptr addrspace(1) %v392, i64 %v223, !dbg !46
  %v394 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !46
  %v395 = zext i32 %v394 to i64, !dbg !46
  %v396 = inttoptr i64 %v395 to ptr addrspace(3), !dbg !46
  %address.local832 = getelementptr [2 x i8], ptr addrspace(1) %v393, i64 %address.local449, !dbg !46
  %v397 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %v398 = ptrtoint ptr addrspace(1) %address.local832 to i64, !dbg !46
  %v399 = sub i64 %v398, %v397, !dbg !46
  %dma.buffer.offset1576 = trunc i64 %v399 to i32, !dbg !46
  %v400 = trunc i64 %v397 to i32, !dbg !46
  %v401 = insertelement <4 x i32> poison, i32 %v400, i64 0, !dbg !46
  %v402 = lshr i64 %v397, 32, !dbg !46
  %v403 = trunc i64 %v402 to i32, !dbg !46
  %v404 = insertelement <4 x i32> %v401, i32 %v403, i64 1, !dbg !46
  %v405 = insertelement <4 x i32> %v404, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1577 = insertelement <4 x i32> %v405, i32 131072, i64 3, !dbg !46
  %v406 = ptrtoint ptr addrspace(3) %v396 to i32, !dbg !46
  %dma.buffer.lds1578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v406), !dbg !46
  %v407 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1577) #10, !dbg !46
  %v408 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !46
  %v409 = zext i32 %v408 to i64, !dbg !46
  %v410 = inttoptr i64 %v409 to ptr addrspace(3), !dbg !46
  %address.local833 = getelementptr [2 x i8], ptr addrspace(1) %v393, i64 %address.local466, !dbg !46
  %v411 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %v412 = ptrtoint ptr addrspace(1) %address.local833 to i64, !dbg !46
  %v413 = sub i64 %v412, %v411, !dbg !46
  %dma.buffer.offset1573 = trunc i64 %v413 to i32, !dbg !46
  %v414 = trunc i64 %v411 to i32, !dbg !46
  %v415 = insertelement <4 x i32> poison, i32 %v414, i64 0, !dbg !46
  %v416 = lshr i64 %v411, 32, !dbg !46
  %v417 = trunc i64 %v416 to i32, !dbg !46
  %v418 = insertelement <4 x i32> %v415, i32 %v417, i64 1, !dbg !46
  %v419 = insertelement <4 x i32> %v418, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1574 = insertelement <4 x i32> %v419, i32 131072, i64 3, !dbg !46
  %v420 = ptrtoint ptr addrspace(3) %v410 to i32, !dbg !46
  %dma.buffer.lds1575 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v420), !dbg !46
  %v421 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1575, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1574) #10, !dbg !46
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !47
  %v422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !47
  %v423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v422, <4 x float> %v311) #9, !dbg !47
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !47
  %v424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !47
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !47
  %v425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !47
  %v426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v424, <4 x float> %v423) #9, !dbg !47
  %v427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v425, <4 x float> %v315) #9, !dbg !47
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !47
  %v428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !47
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !47
  %v429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !47
  %v430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v428, <4 x float> %v427) #9, !dbg !47
  %v431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v429, <4 x float> %v319) #9, !dbg !47
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !47
  %v432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !47
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !47
  %v433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !47
  %v434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v432, <4 x float> %v431) #9, !dbg !47
  %v435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v433, <4 x float> %v323) #9, !dbg !47
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !47
  %v436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !47
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !47
  %v437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !47
  %v438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v436, <4 x float> %v435) #9, !dbg !47
  %v439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v437, <4 x float> %v327) #9, !dbg !47
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !47
  %v440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !47
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !47
  %v441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !47
  %v442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v440, <4 x float> %v439) #9, !dbg !47
  %v443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v441, <4 x float> %v331) #9, !dbg !47
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !47
  %v444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !47
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !47
  %v445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !47
  %v446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v444, <4 x float> %v443) #9, !dbg !47
  %v447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v445, <4 x float> %v335) #9, !dbg !47
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !47
  %v448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !47
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !47
  %v449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !47
  %v450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v448, <4 x float> %v447) #9, !dbg !47
  %v451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v449, <4 x float> %v339) #9, !dbg !47
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !47
  %v452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !47
  %lds.const.gep1470 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !47
  %v453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1470, align 8, !dbg !47
  %v454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v452, <4 x float> %v451) #9, !dbg !47
  %v455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v453, <4 x float> %v343) #9, !dbg !47
  %lds.const.gep1471 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !47
  %v456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1471, align 8, !dbg !47
  %lds.const.gep1472 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !47
  %v457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1472, align 8, !dbg !47
  %v458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v456, <4 x float> %v455) #9, !dbg !47
  %v459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v457, <4 x float> %v347) #9, !dbg !47
  %lds.const.gep1473 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !47
  %v460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1473, align 8, !dbg !47
  %lds.const.gep1474 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !47
  %v461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1474, align 8, !dbg !47
  %v462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v460, <4 x float> %v459) #9, !dbg !47
  %v463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v461, <4 x float> %v351) #9, !dbg !47
  %lds.const.gep1475 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !47
  %v464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1475, align 8, !dbg !47
  %lds.const.gep1476 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !47
  %v465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1476, align 8, !dbg !47
  %v466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v464, <4 x float> %v463) #9, !dbg !47
  %v467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v465, <4 x float> %v355) #9, !dbg !47
  %lds.const.gep1477 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !47
  %v468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1477, align 8, !dbg !47
  %lds.const.gep1478 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !47
  %v469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !47
  %v470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v468, <4 x float> %v467) #9, !dbg !47
  %v471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v469, <4 x float> %v359) #9, !dbg !47
  %lds.const.gep1479 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !47
  %v472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !47
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !47
  %v473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !47
  %v474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v472, <4 x float> %v471) #9, !dbg !47
  %v475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v473, <4 x float> %v363) #9, !dbg !47
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !47
  %v476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !47
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !47
  %v477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !47
  %v478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v476, <4 x float> %v475) #9, !dbg !47
  %v479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v477, <4 x float> %v367) #9, !dbg !47
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !47
  %v480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !47
  %lds.const.gep1484 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !47
  %v481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1484, align 8, !dbg !47
  %v482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v480, <4 x float> %v479) #9, !dbg !47
  %v483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v481, <4 x float> %v370) #9, !dbg !47
  %lds.const.gep1485 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !47
  %v484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1485, align 8, !dbg !47
  %v485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v484, <4 x float> %v483) #9, !dbg !47
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %v486 = add nuw i64 %v205, 3072, !dbg !46
  %v487 = icmp slt i64 %v486, 0, !dbg !46
  %v488 = sub nuw i64 -3073, %v205, !dbg !46
  %v489 = select i1 %v487, i64 %v488, i64 %v486, !dbg !46
  %v490 = sdiv i64 %v489, 4096, !dbg !46
  %.lobit39 = ashr i64 %v486, 63, !dbg !46
  %v491 = xor i64 %v490, %.lobit39, !dbg !46
  %v492 = add nsw i64 %v491, %v16, !dbg !46
  %v493 = add nsw i64 %v262, 96, !dbg !46
  %v494 = icmp slt i64 %v262, -96, !dbg !46
  %v495 = sub nsw i64 -97, %v262, !dbg !46
  %v496 = select i1 %v494, i64 %v495, i64 %v493, !dbg !46
  %v497 = sdiv i64 %v496, 128, !dbg !46
  %v498 = sext i1 %v494 to i64, !dbg !46
  %v499 = xor i64 %v497, %v498, !dbg !46
  %v500 = shl nsw i64 %v499, 7, !dbg !46
  %v501 = sub i64 %v195, %v500, !dbg !46
  %v502 = shl i64 %v492, 19, !dbg !46
  %v503 = and i64 %v502, 16252928, !dbg !46
  %v504 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %v503, !dbg !46
  %v505 = getelementptr [2 x i8], ptr addrspace(1) %v504, i64 %v217, !dbg !46
  %v506 = getelementptr [2 x i8], ptr addrspace(1) %v505, i64 %v501, !dbg !46
  %v507 = getelementptr i8, ptr addrspace(1) %v506, i64 192, !dbg !46
  %v508 = getelementptr [2 x i8], ptr addrspace(1) %v507, i64 %v223, !dbg !46
  %v509 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !46
  %v510 = zext i32 %v509 to i64, !dbg !46
  %v511 = inttoptr i64 %v510 to ptr addrspace(3), !dbg !46
  %address.local850 = getelementptr [2 x i8], ptr addrspace(1) %v508, i64 %address.local449, !dbg !46
  %v512 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %v513 = ptrtoint ptr addrspace(1) %address.local850 to i64, !dbg !46
  %v514 = sub i64 %v513, %v512, !dbg !46
  %dma.buffer.offset1570 = trunc i64 %v514 to i32, !dbg !46
  %v515 = trunc i64 %v512 to i32, !dbg !46
  %v516 = insertelement <4 x i32> poison, i32 %v515, i64 0, !dbg !46
  %v517 = lshr i64 %v512, 32, !dbg !46
  %v518 = trunc i64 %v517 to i32, !dbg !46
  %v519 = insertelement <4 x i32> %v516, i32 %v518, i64 1, !dbg !46
  %v520 = insertelement <4 x i32> %v519, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1571 = insertelement <4 x i32> %v520, i32 131072, i64 3, !dbg !46
  %v521 = ptrtoint ptr addrspace(3) %v511 to i32, !dbg !46
  %dma.buffer.lds1572 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v521), !dbg !46
  %v522 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1572, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1571) #10, !dbg !46
  %v523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !46
  %v524 = zext i32 %v523 to i64, !dbg !46
  %v525 = inttoptr i64 %v524 to ptr addrspace(3), !dbg !46
  %address.local851 = getelementptr [2 x i8], ptr addrspace(1) %v508, i64 %address.local466, !dbg !46
  %v526 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %v527 = ptrtoint ptr addrspace(1) %address.local851 to i64, !dbg !46
  %v528 = sub i64 %v527, %v526, !dbg !46
  %dma.buffer.offset1567 = trunc i64 %v528 to i32, !dbg !46
  %v529 = trunc i64 %v526 to i32, !dbg !46
  %v530 = insertelement <4 x i32> poison, i32 %v529, i64 0, !dbg !46
  %v531 = lshr i64 %v526, 32, !dbg !46
  %v532 = trunc i64 %v531 to i32, !dbg !46
  %v533 = insertelement <4 x i32> %v530, i32 %v532, i64 1, !dbg !46
  %v534 = insertelement <4 x i32> %v533, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1568 = insertelement <4 x i32> %v534, i32 131072, i64 3, !dbg !46
  %v535 = ptrtoint ptr addrspace(3) %v525 to i32, !dbg !46
  %dma.buffer.lds1569 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v535), !dbg !46
  %v536 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1569, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1568) #10, !dbg !46
  %v537 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !47
  %v538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v537, <4 x float> %v426) #9, !dbg !47
  %lds.const.gep1486 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !47
  %v539 = load <4 x half>, ptr addrspace(3) %lds.const.gep1486, align 8, !dbg !47
  %v540 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !47
  %v541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v539, <4 x float> %v538) #9, !dbg !47
  %v542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v540, <4 x float> %v430) #9, !dbg !47
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !47
  %v543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !47
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !47
  %v544 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !47
  %v545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v543, <4 x float> %v542) #9, !dbg !47
  %v546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v544, <4 x float> %v434) #9, !dbg !47
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !47
  %v547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !47
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !47
  %v548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !47
  %v549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v547, <4 x float> %v546) #9, !dbg !47
  %v550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v548, <4 x float> %v438) #9, !dbg !47
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !47
  %v551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !47
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !47
  %v552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !47
  %v553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v551, <4 x float> %v550) #9, !dbg !47
  %v554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v552, <4 x float> %v442) #9, !dbg !47
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !47
  %v555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !47
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !47
  %v556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !47
  %v557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v555, <4 x float> %v554) #9, !dbg !47
  %v558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v556, <4 x float> %v446) #9, !dbg !47
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !47
  %v559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !47
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !47
  %v560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !47
  %v561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v559, <4 x float> %v558) #9, !dbg !47
  %v562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v560, <4 x float> %v450) #9, !dbg !47
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !47
  %v563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !47
  %lds.const.gep1498 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !47
  %v564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !47
  %v565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v563, <4 x float> %v562) #9, !dbg !47
  %v566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v564, <4 x float> %v454) #9, !dbg !47
  %lds.const.gep1499 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !47
  %v567 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !47
  %v568 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !47
  %v569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v567, <4 x float> %v566) #9, !dbg !47
  %v570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v568, <4 x float> %v458) #9, !dbg !47
  %lds.const.gep1500 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !47
  %v571 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !47
  %v572 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !47
  %v573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v571, <4 x float> %v570) #9, !dbg !47
  %v574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v572, <4 x float> %v462) #9, !dbg !47
  %lds.const.gep1501 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !47
  %v575 = load <4 x half>, ptr addrspace(3) %lds.const.gep1501, align 8, !dbg !47
  %lds.const.gep1502 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !47
  %v576 = load <4 x half>, ptr addrspace(3) %lds.const.gep1502, align 8, !dbg !47
  %v577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v575, <4 x float> %v574) #9, !dbg !47
  %v578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v576, <4 x float> %v466) #9, !dbg !47
  %lds.const.gep1503 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !47
  %v579 = load <4 x half>, ptr addrspace(3) %lds.const.gep1503, align 8, !dbg !47
  %lds.const.gep1504 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !47
  %v580 = load <4 x half>, ptr addrspace(3) %lds.const.gep1504, align 8, !dbg !47
  %v581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v579, <4 x float> %v578) #9, !dbg !47
  %v582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v580, <4 x float> %v470) #9, !dbg !47
  %lds.const.gep1505 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !47
  %v583 = load <4 x half>, ptr addrspace(3) %lds.const.gep1505, align 8, !dbg !47
  %lds.const.gep1506 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !47
  %v584 = load <4 x half>, ptr addrspace(3) %lds.const.gep1506, align 8, !dbg !47
  %v585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v583, <4 x float> %v582) #9, !dbg !47
  %v586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v584, <4 x float> %v474) #9, !dbg !47
  %lds.const.gep1507 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !47
  %v587 = load <4 x half>, ptr addrspace(3) %lds.const.gep1507, align 8, !dbg !47
  %lds.const.gep1508 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !47
  %v588 = load <4 x half>, ptr addrspace(3) %lds.const.gep1508, align 8, !dbg !47
  %v589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v587, <4 x float> %v586) #9, !dbg !47
  %v590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v588, <4 x float> %v478) #9, !dbg !47
  %lds.const.gep1509 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !47
  %v591 = load <4 x half>, ptr addrspace(3) %lds.const.gep1509, align 8, !dbg !47
  %lds.const.gep1510 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !47
  %v592 = load <4 x half>, ptr addrspace(3) %lds.const.gep1510, align 8, !dbg !47
  %v593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v591, <4 x float> %v590) #9, !dbg !47
  %v594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v592, <4 x float> %v482) #9, !dbg !47
  %lds.const.gep1511 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !47
  %v595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1511, align 8, !dbg !47
  %lds.const.gep1512 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !47
  %v596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !47
  %v597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v595, <4 x float> %v594) #9, !dbg !47
  %v598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v596, <4 x float> %v485) #9, !dbg !47
  %lds.const.gep1513 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !47
  %v599 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !47
  %v600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v599, <4 x float> %v598) #9, !dbg !47
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %v601 = add nuw nsw i64 %v217, %v16, !dbg !48
  %v602 = shl i64 %v601, 19, !dbg !48
  %v603 = and i64 %v602, 16252928, !dbg !48
  %v604 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %v603, !dbg !48
  %v605 = getelementptr [2 x i8], ptr addrspace(1) %v604, i64 %v223, !dbg !48
  %v606 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !48
  %v607 = zext i32 %v606 to i64, !dbg !48
  %v608 = inttoptr i64 %v607 to ptr addrspace(3), !dbg !48
  %address.local852 = shl nuw nsw i64 %address.local434, 5, !dbg !48
  %address.local853 = and i64 %address.local852, 8064, !dbg !48
  %address.local854 = shl nuw nsw i64 %address.local434, 3, !dbg !48
  %address.local855 = and i64 %address.local854, 24, !dbg !48
  %address.local856 = or disjoint i64 %address.local853, %address.local855, !dbg !48
  %address.local857 = getelementptr [2 x i8], ptr addrspace(1) %v605, i64 %address.local856, !dbg !48
  %v609 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %v610 = ptrtoint ptr addrspace(1) %address.local857 to i64, !dbg !48
  %v611 = sub i64 %v610, %v609, !dbg !48
  %dma.buffer.offset1564 = trunc i64 %v611 to i32, !dbg !48
  %v612 = trunc i64 %v609 to i32, !dbg !48
  %v613 = insertelement <4 x i32> poison, i32 %v612, i64 0, !dbg !48
  %v614 = lshr i64 %v609, 32, !dbg !48
  %v615 = trunc i64 %v614 to i32, !dbg !48
  %v616 = insertelement <4 x i32> %v613, i32 %v615, i64 1, !dbg !48
  %v617 = insertelement <4 x i32> %v616, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1565 = insertelement <4 x i32> %v617, i32 131072, i64 3, !dbg !48
  %v618 = ptrtoint ptr addrspace(3) %v608 to i32, !dbg !48
  %dma.buffer.lds1566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v618), !dbg !48
  %v619 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1566, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #10, !dbg !48
  %v620 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !48
  %v621 = zext i32 %v620 to i64, !dbg !48
  %v622 = inttoptr i64 %v621 to ptr addrspace(3), !dbg !48
  %address.local858 = shl nuw nsw i64 %address.local451, 5, !dbg !48
  %address.local859 = and i64 %address.local858, 16256, !dbg !48
  %address.local860 = shl nuw nsw i64 %address.local434, 3, !dbg !48
  %address.local861 = and i64 %address.local860, 24, !dbg !48
  %address.local862 = or disjoint i64 %address.local859, %address.local861, !dbg !48
  %address.local863 = getelementptr [2 x i8], ptr addrspace(1) %v605, i64 %address.local862, !dbg !48
  %v623 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %v624 = ptrtoint ptr addrspace(1) %address.local863 to i64, !dbg !48
  %v625 = sub i64 %v624, %v623, !dbg !48
  %dma.buffer.offset1561 = trunc i64 %v625 to i32, !dbg !48
  %v626 = trunc i64 %v623 to i32, !dbg !48
  %v627 = insertelement <4 x i32> poison, i32 %v626, i64 0, !dbg !48
  %v628 = lshr i64 %v623, 32, !dbg !48
  %v629 = trunc i64 %v628 to i32, !dbg !48
  %v630 = insertelement <4 x i32> %v627, i32 %v629, i64 1, !dbg !48
  %v631 = insertelement <4 x i32> %v630, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1562 = insertelement <4 x i32> %v631, i32 131072, i64 3, !dbg !48
  %v632 = ptrtoint ptr addrspace(3) %v622 to i32, !dbg !48
  %dma.buffer.lds1563 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v632), !dbg !48
  %v633 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1563, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1562) #10, !dbg !48
  %lds.const.gep1514 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !47
  %v634 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !47
  %v635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v634, <4 x float> %v541) #9, !dbg !47
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !47
  %v636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !47
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !47
  %v637 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !47
  %v638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v636, <4 x float> %v635) #9, !dbg !47
  %v639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v637, <4 x float> %v545) #9, !dbg !47
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !47
  %v640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !47
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !47
  %v641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !47
  %v642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v640, <4 x float> %v639) #9, !dbg !47
  %v643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v641, <4 x float> %v549) #9, !dbg !47
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !47
  %v644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !47
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !47
  %v645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !47
  %v646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v644, <4 x float> %v643) #9, !dbg !47
  %v647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v645, <4 x float> %v553) #9, !dbg !47
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !47
  %v648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !47
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !47
  %v649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !47
  %v650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v648, <4 x float> %v647) #9, !dbg !47
  %v651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v649, <4 x float> %v557) #9, !dbg !47
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !47
  %v652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !47
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !47
  %v653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !47
  %v654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v652, <4 x float> %v651) #9, !dbg !47
  %v655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v653, <4 x float> %v561) #9, !dbg !47
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !47
  %v656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !47
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !47
  %v657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !47
  %v658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v656, <4 x float> %v655) #9, !dbg !47
  %v659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v657, <4 x float> %v565) #9, !dbg !47
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !47
  %v660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !47
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !47
  %v661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !47
  %v662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v660, <4 x float> %v659) #9, !dbg !47
  %v663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v661, <4 x float> %v569) #9, !dbg !47
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !47
  %v664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !47
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !47
  %v665 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !47
  %v666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v664, <4 x float> %v663) #9, !dbg !47
  %v667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v665, <4 x float> %v573) #9, !dbg !47
  %lds.const.gep1531 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !47
  %v668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1531, align 8, !dbg !47
  %lds.const.gep1532 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !47
  %v669 = load <4 x half>, ptr addrspace(3) %lds.const.gep1532, align 8, !dbg !47
  %v670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v668, <4 x float> %v667) #9, !dbg !47
  %v671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v669, <4 x float> %v577) #9, !dbg !47
  %lds.const.gep1533 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !47
  %v672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1533, align 8, !dbg !47
  %lds.const.gep1534 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !47
  %v673 = load <4 x half>, ptr addrspace(3) %lds.const.gep1534, align 8, !dbg !47
  %v674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v672, <4 x float> %v671) #9, !dbg !47
  %v675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v673, <4 x float> %v581) #9, !dbg !47
  %lds.const.gep1535 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !47
  %v676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1535, align 8, !dbg !47
  %lds.const.gep1536 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !47
  %v677 = load <4 x half>, ptr addrspace(3) %lds.const.gep1536, align 8, !dbg !47
  %v678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v676, <4 x float> %v675) #9, !dbg !47
  %v679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v677, <4 x float> %v585) #9, !dbg !47
  %lds.const.gep1537 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !47
  %v680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1537, align 8, !dbg !47
  %lds.const.gep1538 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !47
  %v681 = load <4 x half>, ptr addrspace(3) %lds.const.gep1538, align 8, !dbg !47
  %v682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v680, <4 x float> %v679) #9, !dbg !47
  %v683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v681, <4 x float> %v589) #9, !dbg !47
  %lds.const.gep1539 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !47
  %v684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1539, align 8, !dbg !47
  %lds.const.gep1540 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !47
  %v685 = load <4 x half>, ptr addrspace(3) %lds.const.gep1540, align 8, !dbg !47
  %v686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v684, <4 x float> %v683) #9, !dbg !47
  %v687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v685, <4 x float> %v593) #9, !dbg !47
  %lds.const.gep1541 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !47
  %v688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1541, align 8, !dbg !47
  %lds.const.gep1542 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !47
  %v689 = load <4 x half>, ptr addrspace(3) %lds.const.gep1542, align 8, !dbg !47
  %v690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v688, <4 x float> %v687) #9, !dbg !47
  %v691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v689, <4 x float> %v597) #9, !dbg !47
  %lds.const.gep1543 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !47
  %v692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1543, align 8, !dbg !47
  %lds.const.gep1544 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !47
  %v693 = load <4 x half>, ptr addrspace(3) %lds.const.gep1544, align 8, !dbg !47
  %v694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v692, <4 x float> %v691) #9, !dbg !47
  %v695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v693, <4 x float> %v600) #9, !dbg !47
  %lds.const.gep1545 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !47
  %v696 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !47
  %v697 = or disjoint i64 %v214, 127, !dbg !43
  %v698 = icmp uge i64 %v64, %v697, !dbg !43
  %v699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v696, <4 x float> %v695) #9, !dbg !47
  %v700 = and i1 %v190, %v698, !dbg !43
  br i1 %v700, label %v701, label %v702, !dbg !43

v701:                                             ; preds = %v213
  br label %v846, !dbg !43

v702:                                             ; preds = %v213
  %v703 = add i64 %v64, %v32, !dbg !43
  %v704 = shl nuw nsw i64 %v65, 4, !dbg !43
  %v705 = add i64 %v703, %v704, !dbg !43
  %v706 = and i32 %v31, 240, !dbg !43
  %index.compact = zext nneg i32 %v706 to i64, !dbg !43
  %v707 = sub i64 %v705, %index.compact, !dbg !43
  %v708 = and i64 %v32, 48, !dbg !43
  %v709 = or disjoint i64 %v214, %v708, !dbg !43
  %v710 = add i64 %v707, 1, !dbg !52
  %v711 = lshr i64 %v710, 32, !dbg !53
  %compare.high = trunc i64 %v711 to i32, !dbg !53
  %v712 = lshr i64 %v709, 32, !dbg !53
  %compare.high1157 = trunc i64 %v712 to i32, !dbg !53
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !53
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !53
  %compare.low.base = trunc i64 %v710 to i32, !dbg !53
  %compare.low.base1158 = trunc i64 %v709 to i32, !dbg !53
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1158, !dbg !53
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !53
  %v713 = select i1 %compare.result, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 1) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local = or i32 %compare.local.base, 1, !dbg !53
  %compare.low1159 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !53
  %compare.result1160 = select i1 %compare.high.equal, i1 %compare.low1159, i1 %compare.high.order, !dbg !53
  %v714 = select i1 %compare.result1160, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1161 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 2) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1162 = or i32 %compare.local.base1161, 2, !dbg !53
  %compare.low1163 = icmp ugt i32 %compare.low.base, %compare.local1162, !dbg !53
  %compare.result1164 = select i1 %compare.high.equal, i1 %compare.low1163, i1 %compare.high.order, !dbg !53
  %v715 = select i1 %compare.result1164, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1165 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 3) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1166 = or i32 %compare.local.base1165, 3, !dbg !53
  %compare.low1167 = icmp ugt i32 %compare.low.base, %compare.local1166, !dbg !53
  %compare.result1168 = select i1 %compare.high.equal, i1 %compare.low1167, i1 %compare.high.order, !dbg !53
  %v716 = select i1 %compare.result1168, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1169 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 4) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1170 = or i32 %compare.local.base1169, 4, !dbg !53
  %compare.low1171 = icmp ugt i32 %compare.low.base, %compare.local1170, !dbg !53
  %compare.result1172 = select i1 %compare.high.equal, i1 %compare.low1171, i1 %compare.high.order, !dbg !53
  %v717 = select i1 %compare.result1172, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1173 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 5) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1174 = or i32 %compare.local.base1173, 5, !dbg !53
  %compare.low1175 = icmp ugt i32 %compare.low.base, %compare.local1174, !dbg !53
  %compare.result1176 = select i1 %compare.high.equal, i1 %compare.low1175, i1 %compare.high.order, !dbg !53
  %v718 = select i1 %compare.result1176, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1177 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 6) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1178 = or i32 %compare.local.base1177, 6, !dbg !53
  %compare.low1179 = icmp ugt i32 %compare.low.base, %compare.local1178, !dbg !53
  %compare.result1180 = select i1 %compare.high.equal, i1 %compare.low1179, i1 %compare.high.order, !dbg !53
  %v719 = select i1 %compare.result1180, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1181 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 7) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1182 = or i32 %compare.local.base1181, 7, !dbg !53
  %compare.low1183 = icmp ugt i32 %compare.low.base, %compare.local1182, !dbg !53
  %compare.result1184 = select i1 %compare.high.equal, i1 %compare.low1183, i1 %compare.high.order, !dbg !53
  %v720 = select i1 %compare.result1184, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1185 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 8) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1186 = or i32 %compare.local.base1185, 8, !dbg !53
  %compare.low1187 = icmp ugt i32 %compare.low.base, %compare.local1186, !dbg !53
  %compare.result1188 = select i1 %compare.high.equal, i1 %compare.low1187, i1 %compare.high.order, !dbg !53
  %v721 = select i1 %compare.result1188, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1189 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 9) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1190 = or i32 %compare.local.base1189, 9, !dbg !53
  %compare.low1191 = icmp ugt i32 %compare.low.base, %compare.local1190, !dbg !53
  %compare.result1192 = select i1 %compare.high.equal, i1 %compare.low1191, i1 %compare.high.order, !dbg !53
  %v722 = select i1 %compare.result1192, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1193 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 10) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1194 = or i32 %compare.local.base1193, 10, !dbg !53
  %compare.low1195 = icmp ugt i32 %compare.low.base, %compare.local1194, !dbg !53
  %compare.result1196 = select i1 %compare.high.equal, i1 %compare.low1195, i1 %compare.high.order, !dbg !53
  %v723 = select i1 %compare.result1196, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1197 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 11) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1198 = or i32 %compare.local.base1197, 11, !dbg !53
  %compare.low1199 = icmp ugt i32 %compare.low.base, %compare.local1198, !dbg !53
  %compare.result1200 = select i1 %compare.high.equal, i1 %compare.low1199, i1 %compare.high.order, !dbg !53
  %v724 = select i1 %compare.result1200, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1201 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 12) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1202 = or i32 %compare.local.base1201, 12, !dbg !53
  %compare.low1203 = icmp ugt i32 %compare.low.base, %compare.local1202, !dbg !53
  %compare.result1204 = select i1 %compare.high.equal, i1 %compare.low1203, i1 %compare.high.order, !dbg !53
  %v725 = select i1 %compare.result1204, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1205 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 13) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1206 = or i32 %compare.local.base1205, 13, !dbg !53
  %compare.low1207 = icmp ugt i32 %compare.low.base, %compare.local1206, !dbg !53
  %compare.result1208 = select i1 %compare.high.equal, i1 %compare.low1207, i1 %compare.high.order, !dbg !53
  %v726 = select i1 %compare.result1208, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1209 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 14) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1210 = or i32 %compare.local.base1209, 14, !dbg !53
  %compare.low1211 = icmp ugt i32 %compare.low.base, %compare.local1210, !dbg !53
  %compare.result1212 = select i1 %compare.high.equal, i1 %compare.low1211, i1 %compare.high.order, !dbg !53
  %v727 = select i1 %compare.result1212, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1213 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 15) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1214 = or i32 %compare.local.base1213, 15, !dbg !53
  %compare.low1215 = icmp ugt i32 %compare.low.base, %compare.local1214, !dbg !53
  %compare.result1216 = select i1 %compare.high.equal, i1 %compare.low1215, i1 %compare.high.order, !dbg !53
  %v728 = select i1 %compare.result1216, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1217 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 16) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1218 = or i32 %compare.local.base1217, 64, !dbg !53
  %compare.low1219 = icmp ugt i32 %compare.low.base, %compare.local1218, !dbg !53
  %compare.result1220 = select i1 %compare.high.equal, i1 %compare.low1219, i1 %compare.high.order, !dbg !53
  %v729 = select i1 %compare.result1220, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1221 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 17) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1222 = or i32 %compare.local.base1221, 65, !dbg !53
  %compare.low1223 = icmp ugt i32 %compare.low.base, %compare.local1222, !dbg !53
  %compare.result1224 = select i1 %compare.high.equal, i1 %compare.low1223, i1 %compare.high.order, !dbg !53
  %v730 = select i1 %compare.result1224, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1225 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 18) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1226 = or i32 %compare.local.base1225, 66, !dbg !53
  %compare.low1227 = icmp ugt i32 %compare.low.base, %compare.local1226, !dbg !53
  %compare.result1228 = select i1 %compare.high.equal, i1 %compare.low1227, i1 %compare.high.order, !dbg !53
  %v731 = select i1 %compare.result1228, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1229 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 19) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1230 = or i32 %compare.local.base1229, 67, !dbg !53
  %compare.low1231 = icmp ugt i32 %compare.low.base, %compare.local1230, !dbg !53
  %compare.result1232 = select i1 %compare.high.equal, i1 %compare.low1231, i1 %compare.high.order, !dbg !53
  %v732 = select i1 %compare.result1232, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1233 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 20) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1234 = or i32 %compare.local.base1233, 68, !dbg !53
  %compare.low1235 = icmp ugt i32 %compare.low.base, %compare.local1234, !dbg !53
  %compare.result1236 = select i1 %compare.high.equal, i1 %compare.low1235, i1 %compare.high.order, !dbg !53
  %v733 = select i1 %compare.result1236, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1237 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 21) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1238 = or i32 %compare.local.base1237, 69, !dbg !53
  %compare.low1239 = icmp ugt i32 %compare.low.base, %compare.local1238, !dbg !53
  %compare.result1240 = select i1 %compare.high.equal, i1 %compare.low1239, i1 %compare.high.order, !dbg !53
  %v734 = select i1 %compare.result1240, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1241 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 22) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1242 = or i32 %compare.local.base1241, 70, !dbg !53
  %compare.low1243 = icmp ugt i32 %compare.low.base, %compare.local1242, !dbg !53
  %compare.result1244 = select i1 %compare.high.equal, i1 %compare.low1243, i1 %compare.high.order, !dbg !53
  %v735 = select i1 %compare.result1244, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1245 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 23) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1246 = or i32 %compare.local.base1245, 71, !dbg !53
  %compare.low1247 = icmp ugt i32 %compare.low.base, %compare.local1246, !dbg !53
  %compare.result1248 = select i1 %compare.high.equal, i1 %compare.low1247, i1 %compare.high.order, !dbg !53
  %v736 = select i1 %compare.result1248, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1249 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 24) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1250 = or i32 %compare.local.base1249, 72, !dbg !53
  %compare.low1251 = icmp ugt i32 %compare.low.base, %compare.local1250, !dbg !53
  %compare.result1252 = select i1 %compare.high.equal, i1 %compare.low1251, i1 %compare.high.order, !dbg !53
  %v737 = select i1 %compare.result1252, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1253 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 25) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1254 = or i32 %compare.local.base1253, 73, !dbg !53
  %compare.low1255 = icmp ugt i32 %compare.low.base, %compare.local1254, !dbg !53
  %compare.result1256 = select i1 %compare.high.equal, i1 %compare.low1255, i1 %compare.high.order, !dbg !53
  %v738 = select i1 %compare.result1256, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1257 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 26) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1258 = or i32 %compare.local.base1257, 74, !dbg !53
  %compare.low1259 = icmp ugt i32 %compare.low.base, %compare.local1258, !dbg !53
  %compare.result1260 = select i1 %compare.high.equal, i1 %compare.low1259, i1 %compare.high.order, !dbg !53
  %v739 = select i1 %compare.result1260, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1261 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 27) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1262 = or i32 %compare.local.base1261, 75, !dbg !53
  %compare.low1263 = icmp ugt i32 %compare.low.base, %compare.local1262, !dbg !53
  %compare.result1264 = select i1 %compare.high.equal, i1 %compare.low1263, i1 %compare.high.order, !dbg !53
  %v740 = select i1 %compare.result1264, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1265 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 28) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1266 = or i32 %compare.local.base1265, 76, !dbg !53
  %compare.low1267 = icmp ugt i32 %compare.low.base, %compare.local1266, !dbg !53
  %compare.result1268 = select i1 %compare.high.equal, i1 %compare.low1267, i1 %compare.high.order, !dbg !53
  %v741 = select i1 %compare.result1268, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1269 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 29) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1270 = or i32 %compare.local.base1269, 77, !dbg !53
  %compare.low1271 = icmp ugt i32 %compare.low.base, %compare.local1270, !dbg !53
  %compare.result1272 = select i1 %compare.high.equal, i1 %compare.low1271, i1 %compare.high.order, !dbg !53
  %v742 = select i1 %compare.result1272, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1273 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 30) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1274 = or i32 %compare.local.base1273, 78, !dbg !53
  %compare.low1275 = icmp ugt i32 %compare.low.base, %compare.local1274, !dbg !53
  %compare.result1276 = select i1 %compare.high.equal, i1 %compare.low1275, i1 %compare.high.order, !dbg !53
  %v743 = select i1 %compare.result1276, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1277 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 31) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1278 = or i32 %compare.local.base1277, 79, !dbg !53
  %compare.low1279 = icmp ugt i32 %compare.low.base, %compare.local1278, !dbg !53
  %compare.result1280 = select i1 %compare.high.equal, i1 %compare.low1279, i1 %compare.high.order, !dbg !53
  %v744 = select i1 %compare.result1280, float 0.000000e+00, float -inf, !dbg !54
  %v745 = add i64 %v707, 65, !dbg !52
  %v746 = lshr i64 %v745, 32, !dbg !53
  %compare.high1281 = trunc i64 %v746 to i32, !dbg !53
  %v747 = lshr i64 %v709, 32, !dbg !53
  %compare.high1282 = trunc i64 %v747 to i32, !dbg !53
  %compare.high.equal1283 = icmp eq i32 %compare.high1281, %compare.high1282, !dbg !53
  %compare.high.order1284 = icmp ugt i32 %compare.high1281, %compare.high1282, !dbg !53
  %compare.low.base1285 = trunc i64 %v745 to i32, !dbg !53
  %compare.low.base1286 = trunc i64 %v709 to i32, !dbg !53
  %compare.low1287 = icmp ugt i32 %compare.low.base1285, %compare.low.base1286, !dbg !53
  %compare.result1288 = select i1 %compare.high.equal1283, i1 %compare.low1287, i1 %compare.high.order1284, !dbg !53
  %v748 = select i1 %compare.result1288, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1289 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 32) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1290 = or i32 %compare.local.base1289, 1, !dbg !53
  %compare.low1291 = icmp ugt i32 %compare.low.base1285, %compare.local1290, !dbg !53
  %compare.result1292 = select i1 %compare.high.equal1283, i1 %compare.low1291, i1 %compare.high.order1284, !dbg !53
  %v749 = select i1 %compare.result1292, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 33) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1294 = or i32 %compare.local.base1293, 2, !dbg !53
  %compare.low1295 = icmp ugt i32 %compare.low.base1285, %compare.local1294, !dbg !53
  %compare.result1296 = select i1 %compare.high.equal1283, i1 %compare.low1295, i1 %compare.high.order1284, !dbg !53
  %v750 = select i1 %compare.result1296, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 34) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1298 = or i32 %compare.local.base1297, 3, !dbg !53
  %compare.low1299 = icmp ugt i32 %compare.low.base1285, %compare.local1298, !dbg !53
  %compare.result1300 = select i1 %compare.high.equal1283, i1 %compare.low1299, i1 %compare.high.order1284, !dbg !53
  %v751 = select i1 %compare.result1300, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 35) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1302 = or i32 %compare.local.base1301, 4, !dbg !53
  %compare.low1303 = icmp ugt i32 %compare.low.base1285, %compare.local1302, !dbg !53
  %compare.result1304 = select i1 %compare.high.equal1283, i1 %compare.low1303, i1 %compare.high.order1284, !dbg !53
  %v752 = select i1 %compare.result1304, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 36) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1306 = or i32 %compare.local.base1305, 5, !dbg !53
  %compare.low1307 = icmp ugt i32 %compare.low.base1285, %compare.local1306, !dbg !53
  %compare.result1308 = select i1 %compare.high.equal1283, i1 %compare.low1307, i1 %compare.high.order1284, !dbg !53
  %v753 = select i1 %compare.result1308, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 37) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1310 = or i32 %compare.local.base1309, 6, !dbg !53
  %compare.low1311 = icmp ugt i32 %compare.low.base1285, %compare.local1310, !dbg !53
  %compare.result1312 = select i1 %compare.high.equal1283, i1 %compare.low1311, i1 %compare.high.order1284, !dbg !53
  %v754 = select i1 %compare.result1312, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 38) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1314 = or i32 %compare.local.base1313, 7, !dbg !53
  %compare.low1315 = icmp ugt i32 %compare.low.base1285, %compare.local1314, !dbg !53
  %compare.result1316 = select i1 %compare.high.equal1283, i1 %compare.low1315, i1 %compare.high.order1284, !dbg !53
  %v755 = select i1 %compare.result1316, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 39) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1318 = or i32 %compare.local.base1317, 8, !dbg !53
  %compare.low1319 = icmp ugt i32 %compare.low.base1285, %compare.local1318, !dbg !53
  %compare.result1320 = select i1 %compare.high.equal1283, i1 %compare.low1319, i1 %compare.high.order1284, !dbg !53
  %v756 = select i1 %compare.result1320, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 40) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1322 = or i32 %compare.local.base1321, 9, !dbg !53
  %compare.low1323 = icmp ugt i32 %compare.low.base1285, %compare.local1322, !dbg !53
  %compare.result1324 = select i1 %compare.high.equal1283, i1 %compare.low1323, i1 %compare.high.order1284, !dbg !53
  %v757 = select i1 %compare.result1324, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 41) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1326 = or i32 %compare.local.base1325, 10, !dbg !53
  %compare.low1327 = icmp ugt i32 %compare.low.base1285, %compare.local1326, !dbg !53
  %compare.result1328 = select i1 %compare.high.equal1283, i1 %compare.low1327, i1 %compare.high.order1284, !dbg !53
  %v758 = select i1 %compare.result1328, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 42) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1330 = or i32 %compare.local.base1329, 11, !dbg !53
  %compare.low1331 = icmp ugt i32 %compare.low.base1285, %compare.local1330, !dbg !53
  %compare.result1332 = select i1 %compare.high.equal1283, i1 %compare.low1331, i1 %compare.high.order1284, !dbg !53
  %v759 = select i1 %compare.result1332, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 43) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1334 = or i32 %compare.local.base1333, 12, !dbg !53
  %compare.low1335 = icmp ugt i32 %compare.low.base1285, %compare.local1334, !dbg !53
  %compare.result1336 = select i1 %compare.high.equal1283, i1 %compare.low1335, i1 %compare.high.order1284, !dbg !53
  %v760 = select i1 %compare.result1336, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 44) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1338 = or i32 %compare.local.base1337, 13, !dbg !53
  %compare.low1339 = icmp ugt i32 %compare.low.base1285, %compare.local1338, !dbg !53
  %compare.result1340 = select i1 %compare.high.equal1283, i1 %compare.low1339, i1 %compare.high.order1284, !dbg !53
  %v761 = select i1 %compare.result1340, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 45) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1342 = or i32 %compare.local.base1341, 14, !dbg !53
  %compare.low1343 = icmp ugt i32 %compare.low.base1285, %compare.local1342, !dbg !53
  %compare.result1344 = select i1 %compare.high.equal1283, i1 %compare.low1343, i1 %compare.high.order1284, !dbg !53
  %v762 = select i1 %compare.result1344, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 46) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1346 = or i32 %compare.local.base1345, 15, !dbg !53
  %compare.low1347 = icmp ugt i32 %compare.low.base1285, %compare.local1346, !dbg !53
  %compare.result1348 = select i1 %compare.high.equal1283, i1 %compare.low1347, i1 %compare.high.order1284, !dbg !53
  %v763 = select i1 %compare.result1348, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 47) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1350 = or i32 %compare.local.base1349, 64, !dbg !53
  %compare.low1351 = icmp ugt i32 %compare.low.base1285, %compare.local1350, !dbg !53
  %compare.result1352 = select i1 %compare.high.equal1283, i1 %compare.low1351, i1 %compare.high.order1284, !dbg !53
  %v764 = select i1 %compare.result1352, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 48) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1354 = or i32 %compare.local.base1353, 65, !dbg !53
  %compare.low1355 = icmp ugt i32 %compare.low.base1285, %compare.local1354, !dbg !53
  %compare.result1356 = select i1 %compare.high.equal1283, i1 %compare.low1355, i1 %compare.high.order1284, !dbg !53
  %v765 = select i1 %compare.result1356, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 49) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1358 = or i32 %compare.local.base1357, 66, !dbg !53
  %compare.low1359 = icmp ugt i32 %compare.low.base1285, %compare.local1358, !dbg !53
  %compare.result1360 = select i1 %compare.high.equal1283, i1 %compare.low1359, i1 %compare.high.order1284, !dbg !53
  %v766 = select i1 %compare.result1360, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 50) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1362 = or i32 %compare.local.base1361, 67, !dbg !53
  %compare.low1363 = icmp ugt i32 %compare.low.base1285, %compare.local1362, !dbg !53
  %compare.result1364 = select i1 %compare.high.equal1283, i1 %compare.low1363, i1 %compare.high.order1284, !dbg !53
  %v767 = select i1 %compare.result1364, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 51) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1366 = or i32 %compare.local.base1365, 68, !dbg !53
  %compare.low1367 = icmp ugt i32 %compare.low.base1285, %compare.local1366, !dbg !53
  %compare.result1368 = select i1 %compare.high.equal1283, i1 %compare.low1367, i1 %compare.high.order1284, !dbg !53
  %v768 = select i1 %compare.result1368, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 52) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1370 = or i32 %compare.local.base1369, 69, !dbg !53
  %compare.low1371 = icmp ugt i32 %compare.low.base1285, %compare.local1370, !dbg !53
  %compare.result1372 = select i1 %compare.high.equal1283, i1 %compare.low1371, i1 %compare.high.order1284, !dbg !53
  %v769 = select i1 %compare.result1372, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 53) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1374 = or i32 %compare.local.base1373, 70, !dbg !53
  %compare.low1375 = icmp ugt i32 %compare.low.base1285, %compare.local1374, !dbg !53
  %compare.result1376 = select i1 %compare.high.equal1283, i1 %compare.low1375, i1 %compare.high.order1284, !dbg !53
  %v770 = select i1 %compare.result1376, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 54) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1378 = or i32 %compare.local.base1377, 71, !dbg !53
  %compare.low1379 = icmp ugt i32 %compare.low.base1285, %compare.local1378, !dbg !53
  %compare.result1380 = select i1 %compare.high.equal1283, i1 %compare.low1379, i1 %compare.high.order1284, !dbg !53
  %v771 = select i1 %compare.result1380, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 55) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1382 = or i32 %compare.local.base1381, 72, !dbg !53
  %compare.low1383 = icmp ugt i32 %compare.low.base1285, %compare.local1382, !dbg !53
  %compare.result1384 = select i1 %compare.high.equal1283, i1 %compare.low1383, i1 %compare.high.order1284, !dbg !53
  %v772 = select i1 %compare.result1384, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 56) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1386 = or i32 %compare.local.base1385, 73, !dbg !53
  %compare.low1387 = icmp ugt i32 %compare.low.base1285, %compare.local1386, !dbg !53
  %compare.result1388 = select i1 %compare.high.equal1283, i1 %compare.low1387, i1 %compare.high.order1284, !dbg !53
  %v773 = select i1 %compare.result1388, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 57) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1390 = or i32 %compare.local.base1389, 74, !dbg !53
  %compare.low1391 = icmp ugt i32 %compare.low.base1285, %compare.local1390, !dbg !53
  %compare.result1392 = select i1 %compare.high.equal1283, i1 %compare.low1391, i1 %compare.high.order1284, !dbg !53
  %v774 = select i1 %compare.result1392, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1393 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 58) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1394 = or i32 %compare.local.base1393, 75, !dbg !53
  %compare.low1395 = icmp ugt i32 %compare.low.base1285, %compare.local1394, !dbg !53
  %compare.result1396 = select i1 %compare.high.equal1283, i1 %compare.low1395, i1 %compare.high.order1284, !dbg !53
  %v775 = select i1 %compare.result1396, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 59) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1398 = or i32 %compare.local.base1397, 76, !dbg !53
  %compare.low1399 = icmp ugt i32 %compare.low.base1285, %compare.local1398, !dbg !53
  %compare.result1400 = select i1 %compare.high.equal1283, i1 %compare.low1399, i1 %compare.high.order1284, !dbg !53
  %v776 = select i1 %compare.result1400, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 60) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1402 = or i32 %compare.local.base1401, 77, !dbg !53
  %compare.low1403 = icmp ugt i32 %compare.low.base1285, %compare.local1402, !dbg !53
  %compare.result1404 = select i1 %compare.high.equal1283, i1 %compare.low1403, i1 %compare.high.order1284, !dbg !53
  %v777 = select i1 %compare.result1404, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 61) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1406 = or i32 %compare.local.base1405, 78, !dbg !53
  %compare.low1407 = icmp ugt i32 %compare.low.base1285, %compare.local1406, !dbg !53
  %compare.result1408 = select i1 %compare.high.equal1283, i1 %compare.low1407, i1 %compare.high.order1284, !dbg !53
  %v778 = select i1 %compare.result1408, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 62) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1410 = or i32 %compare.local.base1409, 79, !dbg !53
  %compare.low1411 = icmp ugt i32 %compare.low.base1285, %compare.local1410, !dbg !53
  %compare.result1412 = select i1 %compare.high.equal1283, i1 %compare.low1411, i1 %compare.high.order1284, !dbg !53
  %v779 = select i1 %compare.result1412, float 0.000000e+00, float -inf, !dbg !54
  %v780 = insertelement <32 x float> poison, float %v713, i64 0, !dbg !43
  %v781 = insertelement <32 x float> %v780, float %v714, i64 1, !dbg !43
  %v782 = insertelement <32 x float> %v781, float %v715, i64 2, !dbg !43
  %v783 = insertelement <32 x float> %v782, float %v716, i64 3, !dbg !43
  %v784 = insertelement <32 x float> %v783, float %v717, i64 4, !dbg !43
  %v785 = insertelement <32 x float> %v784, float %v718, i64 5, !dbg !43
  %v786 = insertelement <32 x float> %v785, float %v719, i64 6, !dbg !43
  %v787 = insertelement <32 x float> %v786, float %v720, i64 7, !dbg !43
  %v788 = insertelement <32 x float> %v787, float %v721, i64 8, !dbg !43
  %v789 = insertelement <32 x float> %v788, float %v722, i64 9, !dbg !43
  %v790 = insertelement <32 x float> %v789, float %v723, i64 10, !dbg !43
  %v791 = insertelement <32 x float> %v790, float %v724, i64 11, !dbg !43
  %v792 = insertelement <32 x float> %v791, float %v725, i64 12, !dbg !43
  %v793 = insertelement <32 x float> %v792, float %v726, i64 13, !dbg !43
  %v794 = insertelement <32 x float> %v793, float %v727, i64 14, !dbg !43
  %v795 = insertelement <32 x float> %v794, float %v728, i64 15, !dbg !43
  %v796 = insertelement <32 x float> %v795, float %v729, i64 16, !dbg !43
  %v797 = insertelement <32 x float> %v796, float %v730, i64 17, !dbg !43
  %v798 = insertelement <32 x float> %v797, float %v731, i64 18, !dbg !43
  %v799 = insertelement <32 x float> %v798, float %v732, i64 19, !dbg !43
  %v800 = insertelement <32 x float> %v799, float %v733, i64 20, !dbg !43
  %v801 = insertelement <32 x float> %v800, float %v734, i64 21, !dbg !43
  %v802 = insertelement <32 x float> %v801, float %v735, i64 22, !dbg !43
  %v803 = insertelement <32 x float> %v802, float %v736, i64 23, !dbg !43
  %v804 = insertelement <32 x float> %v803, float %v737, i64 24, !dbg !43
  %v805 = insertelement <32 x float> %v804, float %v738, i64 25, !dbg !43
  %v806 = insertelement <32 x float> %v805, float %v739, i64 26, !dbg !43
  %v807 = insertelement <32 x float> %v806, float %v740, i64 27, !dbg !43
  %v808 = insertelement <32 x float> %v807, float %v741, i64 28, !dbg !43
  %v809 = insertelement <32 x float> %v808, float %v742, i64 29, !dbg !43
  %v810 = insertelement <32 x float> %v809, float %v743, i64 30, !dbg !43
  %v811 = insertelement <32 x float> %v810, float %v744, i64 31, !dbg !43
  %v812 = insertvalue [2 x <32 x float>] poison, <32 x float> %v811, 0, !dbg !43
  %v813 = insertelement <32 x float> poison, float %v748, i64 0, !dbg !43
  %v814 = insertelement <32 x float> %v813, float %v749, i64 1, !dbg !43
  %v815 = insertelement <32 x float> %v814, float %v750, i64 2, !dbg !43
  %v816 = insertelement <32 x float> %v815, float %v751, i64 3, !dbg !43
  %v817 = insertelement <32 x float> %v816, float %v752, i64 4, !dbg !43
  %v818 = insertelement <32 x float> %v817, float %v753, i64 5, !dbg !43
  %v819 = insertelement <32 x float> %v818, float %v754, i64 6, !dbg !43
  %v820 = insertelement <32 x float> %v819, float %v755, i64 7, !dbg !43
  %v821 = insertelement <32 x float> %v820, float %v756, i64 8, !dbg !43
  %v822 = insertelement <32 x float> %v821, float %v757, i64 9, !dbg !43
  %v823 = insertelement <32 x float> %v822, float %v758, i64 10, !dbg !43
  %v824 = insertelement <32 x float> %v823, float %v759, i64 11, !dbg !43
  %v825 = insertelement <32 x float> %v824, float %v760, i64 12, !dbg !43
  %v826 = insertelement <32 x float> %v825, float %v761, i64 13, !dbg !43
  %v827 = insertelement <32 x float> %v826, float %v762, i64 14, !dbg !43
  %v828 = insertelement <32 x float> %v827, float %v763, i64 15, !dbg !43
  %v829 = insertelement <32 x float> %v828, float %v764, i64 16, !dbg !43
  %v830 = insertelement <32 x float> %v829, float %v765, i64 17, !dbg !43
  %v831 = insertelement <32 x float> %v830, float %v766, i64 18, !dbg !43
  %v832 = insertelement <32 x float> %v831, float %v767, i64 19, !dbg !43
  %v833 = insertelement <32 x float> %v832, float %v768, i64 20, !dbg !43
  %v834 = insertelement <32 x float> %v833, float %v769, i64 21, !dbg !43
  %v835 = insertelement <32 x float> %v834, float %v770, i64 22, !dbg !43
  %v836 = insertelement <32 x float> %v835, float %v771, i64 23, !dbg !43
  %v837 = insertelement <32 x float> %v836, float %v772, i64 24, !dbg !43
  %v838 = insertelement <32 x float> %v837, float %v773, i64 25, !dbg !43
  %v839 = insertelement <32 x float> %v838, float %v774, i64 26, !dbg !43
  %v840 = insertelement <32 x float> %v839, float %v775, i64 27, !dbg !43
  %v841 = insertelement <32 x float> %v840, float %v776, i64 28, !dbg !43
  %v842 = insertelement <32 x float> %v841, float %v777, i64 29, !dbg !43
  %v843 = insertelement <32 x float> %v842, float %v778, i64 30, !dbg !43
  %v844 = insertelement <32 x float> %v843, float %v779, i64 31, !dbg !43
  %v845 = insertvalue [2 x <32 x float>] %v812, <32 x float> %v844, 1, !dbg !43
  %v849 = extractvalue [2 x <32 x float>] %v845, 0, !dbg !55
  %v850 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !55
  %v851.masked = fadd <4 x float> %v638, %v850, !dbg !55
  %v856 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !55
  %v857.masked = fadd <4 x float> %v642, %v856, !dbg !55
  %v862 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !55
  %v863.masked = fadd <4 x float> %v646, %v862, !dbg !55
  %v868 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !55
  %v869.masked = fadd <4 x float> %v650, %v868, !dbg !55
  %v874 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !55
  %v875.masked = fadd <4 x float> %v654, %v874, !dbg !55
  %v880 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !55
  %v881.masked = fadd <4 x float> %v658, %v880, !dbg !55
  %v886 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !55
  %v887.masked = fadd <4 x float> %v662, %v886, !dbg !55
  %v892 = shufflevector <32 x float> %v849, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !55
  %v893.masked = fadd <4 x float> %v666, %v892, !dbg !55
  %v898 = extractvalue [2 x <32 x float>] %v845, 1, !dbg !55
  %v899 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !55
  %v900.masked = fadd <4 x float> %v670, %v899, !dbg !55
  %v905 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !55
  %v906.masked = fadd <4 x float> %v674, %v905, !dbg !55
  %v911 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !55
  %v912.masked = fadd <4 x float> %v678, %v911, !dbg !55
  %v917 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !55
  %v918.masked = fadd <4 x float> %v682, %v917, !dbg !55
  %v923 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !55
  %v924.masked = fadd <4 x float> %v686, %v923, !dbg !55
  %v929 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !55
  %v930.masked = fadd <4 x float> %v690, %v929, !dbg !55
  %v935 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !55
  %v936.masked = fadd <4 x float> %v694, %v935, !dbg !55
  %v941 = shufflevector <32 x float> %v898, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !55
  %v942.masked = fadd <4 x float> %v699, %v941, !dbg !55
  br label %v846, !dbg !43

v846:                                             ; preds = %v702, %v701
  %v851 = phi <4 x float> [ %v851.masked, %v702 ], [ %v638, %v701 ], !dbg !55
  %v857 = phi <4 x float> [ %v857.masked, %v702 ], [ %v642, %v701 ], !dbg !55
  %v863 = phi <4 x float> [ %v863.masked, %v702 ], [ %v646, %v701 ], !dbg !55
  %v869 = phi <4 x float> [ %v869.masked, %v702 ], [ %v650, %v701 ], !dbg !55
  %v875 = phi <4 x float> [ %v875.masked, %v702 ], [ %v654, %v701 ], !dbg !55
  %v881 = phi <4 x float> [ %v881.masked, %v702 ], [ %v658, %v701 ], !dbg !55
  %v887 = phi <4 x float> [ %v887.masked, %v702 ], [ %v662, %v701 ], !dbg !55
  %v893 = phi <4 x float> [ %v893.masked, %v702 ], [ %v666, %v701 ], !dbg !55
  %v900 = phi <4 x float> [ %v900.masked, %v702 ], [ %v670, %v701 ], !dbg !55
  %v906 = phi <4 x float> [ %v906.masked, %v702 ], [ %v674, %v701 ], !dbg !55
  %v912 = phi <4 x float> [ %v912.masked, %v702 ], [ %v678, %v701 ], !dbg !55
  %v918 = phi <4 x float> [ %v918.masked, %v702 ], [ %v682, %v701 ], !dbg !55
  %v924 = phi <4 x float> [ %v924.masked, %v702 ], [ %v686, %v701 ], !dbg !55
  %v930 = phi <4 x float> [ %v930.masked, %v702 ], [ %v690, %v701 ], !dbg !55
  %v936 = phi <4 x float> [ %v936.masked, %v702 ], [ %v694, %v701 ], !dbg !55
  %v942 = phi <4 x float> [ %v942.masked, %v702 ], [ %v699, %v701 ], !dbg !55
  br label %v848, !dbg !43

v848:                                             ; preds = %v846
  %v852 = extractelement <4 x float> %v851, i64 0, !dbg !55
  %v853 = extractelement <4 x float> %v851, i64 1, !dbg !55
  %v854 = extractelement <4 x float> %v851, i64 2, !dbg !55
  %v855 = extractelement <4 x float> %v851, i64 3, !dbg !55
  %v858 = extractelement <4 x float> %v857, i64 0, !dbg !55
  %v859 = extractelement <4 x float> %v857, i64 1, !dbg !55
  %v860 = extractelement <4 x float> %v857, i64 2, !dbg !55
  %v861 = extractelement <4 x float> %v857, i64 3, !dbg !55
  %v864 = extractelement <4 x float> %v863, i64 0, !dbg !55
  %v865 = extractelement <4 x float> %v863, i64 1, !dbg !55
  %v866 = extractelement <4 x float> %v863, i64 2, !dbg !55
  %v867 = extractelement <4 x float> %v863, i64 3, !dbg !55
  %v870 = extractelement <4 x float> %v869, i64 0, !dbg !55
  %v871 = extractelement <4 x float> %v869, i64 1, !dbg !55
  %v872 = extractelement <4 x float> %v869, i64 2, !dbg !55
  %v873 = extractelement <4 x float> %v869, i64 3, !dbg !55
  %v876 = extractelement <4 x float> %v875, i64 0, !dbg !55
  %v877 = extractelement <4 x float> %v875, i64 1, !dbg !55
  %v878 = extractelement <4 x float> %v875, i64 2, !dbg !55
  %v879 = extractelement <4 x float> %v875, i64 3, !dbg !55
  %v882 = extractelement <4 x float> %v881, i64 0, !dbg !55
  %v883 = extractelement <4 x float> %v881, i64 1, !dbg !55
  %v884 = extractelement <4 x float> %v881, i64 2, !dbg !55
  %v885 = extractelement <4 x float> %v881, i64 3, !dbg !55
  %v888 = extractelement <4 x float> %v887, i64 0, !dbg !55
  %v889 = extractelement <4 x float> %v887, i64 1, !dbg !55
  %v890 = extractelement <4 x float> %v887, i64 2, !dbg !55
  %v891 = extractelement <4 x float> %v887, i64 3, !dbg !55
  %v894 = extractelement <4 x float> %v893, i64 0, !dbg !55
  %v895 = extractelement <4 x float> %v893, i64 1, !dbg !55
  %v896 = extractelement <4 x float> %v893, i64 2, !dbg !55
  %v897 = extractelement <4 x float> %v893, i64 3, !dbg !55
  %v901 = extractelement <4 x float> %v900, i64 0, !dbg !55
  %v902 = extractelement <4 x float> %v900, i64 1, !dbg !55
  %v903 = extractelement <4 x float> %v900, i64 2, !dbg !55
  %v904 = extractelement <4 x float> %v900, i64 3, !dbg !55
  %v907 = extractelement <4 x float> %v906, i64 0, !dbg !55
  %v908 = extractelement <4 x float> %v906, i64 1, !dbg !55
  %v909 = extractelement <4 x float> %v906, i64 2, !dbg !55
  %v910 = extractelement <4 x float> %v906, i64 3, !dbg !55
  %v913 = extractelement <4 x float> %v912, i64 0, !dbg !55
  %v914 = extractelement <4 x float> %v912, i64 1, !dbg !55
  %v915 = extractelement <4 x float> %v912, i64 2, !dbg !55
  %v916 = extractelement <4 x float> %v912, i64 3, !dbg !55
  %v919 = extractelement <4 x float> %v918, i64 0, !dbg !55
  %v920 = extractelement <4 x float> %v918, i64 1, !dbg !55
  %v921 = extractelement <4 x float> %v918, i64 2, !dbg !55
  %v922 = extractelement <4 x float> %v918, i64 3, !dbg !55
  %v925 = extractelement <4 x float> %v924, i64 0, !dbg !55
  %v926 = extractelement <4 x float> %v924, i64 1, !dbg !55
  %v927 = extractelement <4 x float> %v924, i64 2, !dbg !55
  %v928 = extractelement <4 x float> %v924, i64 3, !dbg !55
  %v931 = extractelement <4 x float> %v930, i64 0, !dbg !55
  %v932 = extractelement <4 x float> %v930, i64 1, !dbg !55
  %v933 = extractelement <4 x float> %v930, i64 2, !dbg !55
  %v934 = extractelement <4 x float> %v930, i64 3, !dbg !55
  %v937 = extractelement <4 x float> %v936, i64 0, !dbg !55
  %v938 = extractelement <4 x float> %v936, i64 1, !dbg !55
  %v939 = extractelement <4 x float> %v936, i64 2, !dbg !55
  %v940 = extractelement <4 x float> %v936, i64 3, !dbg !55
  %v943 = extractelement <4 x float> %v942, i64 0, !dbg !55
  %v944 = extractelement <4 x float> %v942, i64 1, !dbg !55
  %v945 = extractelement <4 x float> %v942, i64 2, !dbg !55
  %v946 = extractelement <4 x float> %v942, i64 3, !dbg !55
  %v947 = call float @llvm.maxnum.f32(float %v852, float -inf), !dbg !56, !frisk.local_reduction !34
  %local.reduce93 = call float @llvm.maxnum.f32(float %v947, float %v858), !dbg !56, !frisk.local_reduction !34
  %local.reduce94 = call float @llvm.maxnum.f32(float %v853, float %v859), !dbg !56, !frisk.local_reduction !34
  %local.reduce95 = call float @llvm.maxnum.f32(float %v854, float %v860), !dbg !56, !frisk.local_reduction !34
  %local.reduce96 = call float @llvm.maxnum.f32(float %v855, float %v861), !dbg !56, !frisk.local_reduction !34
  %local.reduce97 = call float @llvm.maxnum.f32(float %local.reduce93, float %v864), !dbg !56, !frisk.local_reduction !34
  %local.reduce98 = call float @llvm.maxnum.f32(float %local.reduce94, float %v865), !dbg !56, !frisk.local_reduction !34
  %local.reduce99 = call float @llvm.maxnum.f32(float %local.reduce95, float %v866), !dbg !56, !frisk.local_reduction !34
  %local.reduce100 = call float @llvm.maxnum.f32(float %local.reduce96, float %v867), !dbg !56, !frisk.local_reduction !34
  %local.reduce101 = call float @llvm.maxnum.f32(float %local.reduce97, float %v870), !dbg !56, !frisk.local_reduction !34
  %local.reduce102 = call float @llvm.maxnum.f32(float %local.reduce98, float %v871), !dbg !56, !frisk.local_reduction !34
  %local.reduce103 = call float @llvm.maxnum.f32(float %local.reduce99, float %v872), !dbg !56, !frisk.local_reduction !34
  %local.reduce104 = call float @llvm.maxnum.f32(float %local.reduce100, float %v873), !dbg !56, !frisk.local_reduction !34
  %local.reduce105 = call float @llvm.maxnum.f32(float %local.reduce101, float %v876), !dbg !56, !frisk.local_reduction !34
  %local.reduce106 = call float @llvm.maxnum.f32(float %local.reduce102, float %v877), !dbg !56, !frisk.local_reduction !34
  %local.reduce107 = call float @llvm.maxnum.f32(float %local.reduce103, float %v878), !dbg !56, !frisk.local_reduction !34
  %local.reduce108 = call float @llvm.maxnum.f32(float %local.reduce104, float %v879), !dbg !56, !frisk.local_reduction !34
  %local.reduce109 = call float @llvm.maxnum.f32(float %local.reduce105, float %v882), !dbg !56, !frisk.local_reduction !34
  %local.reduce110 = call float @llvm.maxnum.f32(float %local.reduce106, float %v883), !dbg !56, !frisk.local_reduction !34
  %local.reduce111 = call float @llvm.maxnum.f32(float %local.reduce107, float %v884), !dbg !56, !frisk.local_reduction !34
  %local.reduce112 = call float @llvm.maxnum.f32(float %local.reduce108, float %v885), !dbg !56, !frisk.local_reduction !34
  %local.reduce113 = call float @llvm.maxnum.f32(float %local.reduce109, float %v888), !dbg !56, !frisk.local_reduction !34
  %local.reduce114 = call float @llvm.maxnum.f32(float %local.reduce110, float %v889), !dbg !56, !frisk.local_reduction !34
  %local.reduce115 = call float @llvm.maxnum.f32(float %local.reduce111, float %v890), !dbg !56, !frisk.local_reduction !34
  %local.reduce116 = call float @llvm.maxnum.f32(float %local.reduce112, float %v891), !dbg !56, !frisk.local_reduction !34
  %local.reduce117 = call float @llvm.maxnum.f32(float %local.reduce113, float %v894), !dbg !56, !frisk.local_reduction !34
  %local.reduce118 = call float @llvm.maxnum.f32(float %local.reduce114, float %v895), !dbg !56, !frisk.local_reduction !34
  %local.reduce119 = call float @llvm.maxnum.f32(float %local.reduce115, float %v896), !dbg !56, !frisk.local_reduction !34
  %local.reduce120 = call float @llvm.maxnum.f32(float %local.reduce116, float %v897), !dbg !56, !frisk.local_reduction !34
  %local.reduce121 = call float @llvm.maxnum.f32(float %local.reduce117, float %local.reduce118), !dbg !56, !frisk.local_reduction !34
  %local.reduce122 = call float @llvm.maxnum.f32(float %local.reduce119, float %local.reduce120), !dbg !56, !frisk.local_reduction !34
  %v978 = call float @llvm.maxnum.f32(float %local.reduce121, float %local.reduce122), !dbg !56, !frisk.local_reduction !34
  %v979 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !56
  %v980 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %v979), !dbg !56
  %v981 = add nuw nsw i32 %v980, 64, !dbg !56
  %v982 = xor i32 %v980, 16, !dbg !56
  %v983 = and i32 %v981, 192, !dbg !56
  %v984 = icmp samesign ult i32 %v982, %v983, !dbg !56
  %v985 = select i1 %v984, i32 %v982, i32 %v980, !dbg !56
  %v986 = shl nuw nsw i32 %v985, 2, !dbg !56
  %v987 = bitcast float %v978 to i32, !dbg !56
  %v988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v987), !dbg !56
  %v989 = bitcast i32 %v988 to float, !dbg !56
  %v990 = call float @llvm.maxnum.f32(float %v978, float %v989), !dbg !56
  %v991 = xor i32 %v980, 32, !dbg !56
  %v992 = icmp samesign ult i32 %v991, %v983, !dbg !56
  %v993 = select i1 %v992, i32 %v991, i32 %v980, !dbg !56
  %v994 = shl nuw nsw i32 %v993, 2, !dbg !56
  %v995 = bitcast float %v990 to i32, !dbg !56
  %v996 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v995), !dbg !56
  %v997 = bitcast i32 %v996 to float, !dbg !56
  %v998 = call float @llvm.maxnum.f32(float %v990, float %v997), !dbg !56
  %v999 = extractvalue [2 x <1 x float>] %v208, 0, !dbg !56
  %v1000 = insertelement <1 x float> poison, float %v998, i64 0, !dbg !56
  %v1001 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %v1000, <1 x float> %v999), !dbg !56
  %v1002 = call float @llvm.maxnum.f32(float %v901, float -inf), !dbg !56, !frisk.local_reduction !34
  %local.reduce62 = call float @llvm.maxnum.f32(float %v1002, float %v907), !dbg !56, !frisk.local_reduction !34
  %local.reduce63 = call float @llvm.maxnum.f32(float %v902, float %v908), !dbg !56, !frisk.local_reduction !34
  %local.reduce64 = call float @llvm.maxnum.f32(float %v903, float %v909), !dbg !56, !frisk.local_reduction !34
  %local.reduce65 = call float @llvm.maxnum.f32(float %v904, float %v910), !dbg !56, !frisk.local_reduction !34
  %local.reduce66 = call float @llvm.maxnum.f32(float %local.reduce62, float %v913), !dbg !56, !frisk.local_reduction !34
  %local.reduce67 = call float @llvm.maxnum.f32(float %local.reduce63, float %v914), !dbg !56, !frisk.local_reduction !34
  %local.reduce68 = call float @llvm.maxnum.f32(float %local.reduce64, float %v915), !dbg !56, !frisk.local_reduction !34
  %local.reduce69 = call float @llvm.maxnum.f32(float %local.reduce65, float %v916), !dbg !56, !frisk.local_reduction !34
  %local.reduce70 = call float @llvm.maxnum.f32(float %local.reduce66, float %v919), !dbg !56, !frisk.local_reduction !34
  %local.reduce71 = call float @llvm.maxnum.f32(float %local.reduce67, float %v920), !dbg !56, !frisk.local_reduction !34
  %local.reduce72 = call float @llvm.maxnum.f32(float %local.reduce68, float %v921), !dbg !56, !frisk.local_reduction !34
  %local.reduce73 = call float @llvm.maxnum.f32(float %local.reduce69, float %v922), !dbg !56, !frisk.local_reduction !34
  %local.reduce74 = call float @llvm.maxnum.f32(float %local.reduce70, float %v925), !dbg !56, !frisk.local_reduction !34
  %local.reduce75 = call float @llvm.maxnum.f32(float %local.reduce71, float %v926), !dbg !56, !frisk.local_reduction !34
  %local.reduce76 = call float @llvm.maxnum.f32(float %local.reduce72, float %v927), !dbg !56, !frisk.local_reduction !34
  %local.reduce77 = call float @llvm.maxnum.f32(float %local.reduce73, float %v928), !dbg !56, !frisk.local_reduction !34
  %local.reduce78 = call float @llvm.maxnum.f32(float %local.reduce74, float %v931), !dbg !56, !frisk.local_reduction !34
  %local.reduce79 = call float @llvm.maxnum.f32(float %local.reduce75, float %v932), !dbg !56, !frisk.local_reduction !34
  %local.reduce80 = call float @llvm.maxnum.f32(float %local.reduce76, float %v933), !dbg !56, !frisk.local_reduction !34
  %local.reduce81 = call float @llvm.maxnum.f32(float %local.reduce77, float %v934), !dbg !56, !frisk.local_reduction !34
  %local.reduce82 = call float @llvm.maxnum.f32(float %local.reduce78, float %v937), !dbg !56, !frisk.local_reduction !34
  %local.reduce83 = call float @llvm.maxnum.f32(float %local.reduce79, float %v938), !dbg !56, !frisk.local_reduction !34
  %local.reduce84 = call float @llvm.maxnum.f32(float %local.reduce80, float %v939), !dbg !56, !frisk.local_reduction !34
  %local.reduce85 = call float @llvm.maxnum.f32(float %local.reduce81, float %v940), !dbg !56, !frisk.local_reduction !34
  %local.reduce86 = call float @llvm.maxnum.f32(float %local.reduce82, float %v943), !dbg !56, !frisk.local_reduction !34
  %local.reduce87 = call float @llvm.maxnum.f32(float %local.reduce83, float %v944), !dbg !56, !frisk.local_reduction !34
  %local.reduce88 = call float @llvm.maxnum.f32(float %local.reduce84, float %v945), !dbg !56, !frisk.local_reduction !34
  %local.reduce89 = call float @llvm.maxnum.f32(float %local.reduce85, float %v946), !dbg !56, !frisk.local_reduction !34
  %local.reduce90 = call float @llvm.maxnum.f32(float %local.reduce86, float %local.reduce87), !dbg !56, !frisk.local_reduction !34
  %local.reduce91 = call float @llvm.maxnum.f32(float %local.reduce88, float %local.reduce89), !dbg !56, !frisk.local_reduction !34
  %v1033 = call float @llvm.maxnum.f32(float %local.reduce90, float %local.reduce91), !dbg !56, !frisk.local_reduction !34
  %v1034 = bitcast float %v1033 to i32, !dbg !56
  %v1035 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v1034), !dbg !56
  %v1036 = bitcast i32 %v1035 to float, !dbg !56
  %v1037 = call float @llvm.maxnum.f32(float %v1033, float %v1036), !dbg !56
  %v1038 = bitcast float %v1037 to i32, !dbg !56
  %v1039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v1038), !dbg !56
  %v1040 = bitcast i32 %v1039 to float, !dbg !56
  %v1041 = call float @llvm.maxnum.f32(float %v1037, float %v1040), !dbg !56
  %v1042 = extractvalue [2 x <1 x float>] %v208, 1, !dbg !56
  %v1043 = insertelement <1 x float> poison, float %v1041, i64 0, !dbg !56
  %v1044 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %v1043, <1 x float> %v1042), !dbg !56
  %v1045 = shufflevector <1 x float> %v1001, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !57
  %v1046 = fsub <4 x float> %v851, %v1045, !dbg !57
  %v1047 = fsub <4 x float> %v857, %v1045, !dbg !57
  %v1048 = fsub <4 x float> %v863, %v1045, !dbg !57
  %v1049 = fsub <4 x float> %v869, %v1045, !dbg !57
  %v1050 = fsub <4 x float> %v875, %v1045, !dbg !57
  %v1051 = fsub <4 x float> %v881, %v1045, !dbg !57
  %v1052 = fsub <4 x float> %v887, %v1045, !dbg !57
  %v1053 = fsub <4 x float> %v893, %v1045, !dbg !57
  %v1054 = shufflevector <1 x float> %v1044, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !57
  %v1055 = fsub <4 x float> %v900, %v1054, !dbg !57
  %v1056 = fsub <4 x float> %v906, %v1054, !dbg !57
  %v1057 = fsub <4 x float> %v912, %v1054, !dbg !57
  %v1058 = fsub <4 x float> %v918, %v1054, !dbg !57
  %v1059 = fsub <4 x float> %v924, %v1054, !dbg !57
  %v1060 = fsub <4 x float> %v930, %v1054, !dbg !57
  %v1061 = fsub <4 x float> %v936, %v1054, !dbg !57
  %v1062 = fsub <4 x float> %v942, %v1054, !dbg !57
  %v1063 = fsub <1 x float> %v999, %v1001, !dbg !58
  %v1064 = extractelement <1 x float> %v1063, i64 0, !dbg !59
  %v1065 = call float @llvm.amdgcn.exp2.f32(float %v1064), !dbg !59
  %v1066 = insertelement <4 x float> poison, float %v1065, i64 0, !dbg !64
  %v1067 = shufflevector <4 x float> %v1066, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !64
  %v1068 = extractvalue [2 x <1 x float>] %v207, 0, !dbg !60
  %v1069 = extractelement <1 x float> %v1068, i64 0, !dbg !60
  %v1070 = fmul float %v1069, %v1065, !dbg !60
  %v1071 = fsub <1 x float> %v1042, %v1044, !dbg !58
  %v1072 = extractelement <1 x float> %v1071, i64 0, !dbg !59
  %v1073 = call float @llvm.amdgcn.exp2.f32(float %v1072), !dbg !59
  %v1074 = insertelement <4 x float> poison, float %v1073, i64 0, !dbg !64
  %v1075 = shufflevector <4 x float> %v1074, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !64
  %v1076 = extractvalue [2 x <1 x float>] %v207, 1, !dbg !60
  %v1077 = extractelement <1 x float> %v1076, i64 0, !dbg !60
  %v1078 = fmul float %v1077, %v1073, !dbg !60
  %v1079 = extractelement <4 x float> %v1046, i64 0, !dbg !61
  %v1080 = call float @llvm.amdgcn.exp2.f32(float %v1079), !dbg !61
  %v1081 = insertelement <4 x float> poison, float %v1080, i64 0, !dbg !63
  %v1082 = extractelement <4 x float> %v1046, i64 1, !dbg !61
  %v1083 = call float @llvm.amdgcn.exp2.f32(float %v1082), !dbg !61
  %v1084 = insertelement <4 x float> %v1081, float %v1083, i64 1, !dbg !63
  %v1086 = extractelement <4 x float> %v1046, i64 2, !dbg !61
  %v1087 = call float @llvm.amdgcn.exp2.f32(float %v1086), !dbg !61
  %v1088 = insertelement <4 x float> %v1084, float %v1087, i64 2, !dbg !63
  %v1090 = extractelement <4 x float> %v1046, i64 3, !dbg !61
  %v1091 = call float @llvm.amdgcn.exp2.f32(float %v1090), !dbg !61
  %v1092 = insertelement <4 x float> %v1088, float %v1091, i64 3, !dbg !63
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1092) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1093 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !63
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1093) #9, !dbg !62, !frisk.register_lifetimes !34
  %v1095 = extractelement <4 x float> %v1047, i64 0, !dbg !61
  %v1096 = call float @llvm.amdgcn.exp2.f32(float %v1095), !dbg !61
  %v1097 = insertelement <4 x float> poison, float %v1096, i64 0, !dbg !63
  %local.reduce31 = fadd float %v1080, %v1096, !dbg !62, !frisk.local_reduction !34
  %v1099 = extractelement <4 x float> %v1047, i64 1, !dbg !61
  %v1100 = call float @llvm.amdgcn.exp2.f32(float %v1099), !dbg !61
  %v1101 = insertelement <4 x float> %v1097, float %v1100, i64 1, !dbg !63
  %local.reduce32 = fadd float %v1083, %v1100, !dbg !62, !frisk.local_reduction !34
  %v1103 = extractelement <4 x float> %v1047, i64 2, !dbg !61
  %v1104 = call float @llvm.amdgcn.exp2.f32(float %v1103), !dbg !61
  %v1105 = insertelement <4 x float> %v1101, float %v1104, i64 2, !dbg !63
  %local.reduce33 = fadd float %v1087, %v1104, !dbg !62, !frisk.local_reduction !34
  %v1107 = extractelement <4 x float> %v1047, i64 3, !dbg !61
  %v1108 = call float @llvm.amdgcn.exp2.f32(float %v1107), !dbg !61
  %v1109 = insertelement <4 x float> %v1105, float %v1108, i64 3, !dbg !63
  %math.result309 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1109) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1110 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !63
  %math.packed310 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1110) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce34 = fadd float %v1091, %v1108, !dbg !62, !frisk.local_reduction !34
  %v1112 = extractelement <4 x float> %v1048, i64 0, !dbg !61
  %v1113 = call float @llvm.amdgcn.exp2.f32(float %v1112), !dbg !61
  %v1114 = insertelement <4 x float> poison, float %v1113, i64 0, !dbg !63
  %local.reduce35 = fadd float %local.reduce31, %v1113, !dbg !62, !frisk.local_reduction !34
  %v1116 = extractelement <4 x float> %v1048, i64 1, !dbg !61
  %v1117 = call float @llvm.amdgcn.exp2.f32(float %v1116), !dbg !61
  %v1118 = insertelement <4 x float> %v1114, float %v1117, i64 1, !dbg !63
  %local.reduce36 = fadd float %local.reduce32, %v1117, !dbg !62, !frisk.local_reduction !34
  %v1120 = extractelement <4 x float> %v1048, i64 2, !dbg !61
  %v1121 = call float @llvm.amdgcn.exp2.f32(float %v1120), !dbg !61
  %v1122 = insertelement <4 x float> %v1118, float %v1121, i64 2, !dbg !63
  %local.reduce37 = fadd float %local.reduce33, %v1121, !dbg !62, !frisk.local_reduction !34
  %v1124 = extractelement <4 x float> %v1048, i64 3, !dbg !61
  %v1125 = call float @llvm.amdgcn.exp2.f32(float %v1124), !dbg !61
  %v1126 = insertelement <4 x float> %v1122, float %v1125, i64 3, !dbg !63
  %math.result311 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1126) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1127 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !63
  %math.packed312 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1127) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce38 = fadd float %local.reduce34, %v1125, !dbg !62, !frisk.local_reduction !34
  %v1129 = extractelement <4 x float> %v1049, i64 0, !dbg !61
  %v1130 = call float @llvm.amdgcn.exp2.f32(float %v1129), !dbg !61
  %v1131 = insertelement <4 x float> poison, float %v1130, i64 0, !dbg !63
  %local.reduce39 = fadd float %local.reduce35, %v1130, !dbg !62, !frisk.local_reduction !34
  %v1133 = extractelement <4 x float> %v1049, i64 1, !dbg !61
  %v1134 = call float @llvm.amdgcn.exp2.f32(float %v1133), !dbg !61
  %v1135 = insertelement <4 x float> %v1131, float %v1134, i64 1, !dbg !63
  %local.reduce40 = fadd float %local.reduce36, %v1134, !dbg !62, !frisk.local_reduction !34
  %v1137 = extractelement <4 x float> %v1049, i64 2, !dbg !61
  %v1138 = call float @llvm.amdgcn.exp2.f32(float %v1137), !dbg !61
  %v1139 = insertelement <4 x float> %v1135, float %v1138, i64 2, !dbg !63
  %local.reduce41 = fadd float %local.reduce37, %v1138, !dbg !62, !frisk.local_reduction !34
  %v1141 = extractelement <4 x float> %v1049, i64 3, !dbg !61
  %v1142 = call float @llvm.amdgcn.exp2.f32(float %v1141), !dbg !61
  %v1143 = insertelement <4 x float> %v1139, float %v1142, i64 3, !dbg !63
  %math.result313 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1143) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1144 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !63
  %math.packed314 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1144) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce42 = fadd float %local.reduce38, %v1142, !dbg !62, !frisk.local_reduction !34
  %v1146 = extractelement <4 x float> %v1050, i64 0, !dbg !61
  %v1147 = call float @llvm.amdgcn.exp2.f32(float %v1146), !dbg !61
  %v1148 = insertelement <4 x float> poison, float %v1147, i64 0, !dbg !63
  %local.reduce43 = fadd float %local.reduce39, %v1147, !dbg !62, !frisk.local_reduction !34
  %v1150 = extractelement <4 x float> %v1050, i64 1, !dbg !61
  %v1151 = call float @llvm.amdgcn.exp2.f32(float %v1150), !dbg !61
  %v1152 = insertelement <4 x float> %v1148, float %v1151, i64 1, !dbg !63
  %local.reduce44 = fadd float %local.reduce40, %v1151, !dbg !62, !frisk.local_reduction !34
  %v1154 = extractelement <4 x float> %v1050, i64 2, !dbg !61
  %v1155 = call float @llvm.amdgcn.exp2.f32(float %v1154), !dbg !61
  %v1156 = insertelement <4 x float> %v1152, float %v1155, i64 2, !dbg !63
  %local.reduce45 = fadd float %local.reduce41, %v1155, !dbg !62, !frisk.local_reduction !34
  %v1158 = extractelement <4 x float> %v1050, i64 3, !dbg !61
  %v1159 = call float @llvm.amdgcn.exp2.f32(float %v1158), !dbg !61
  %v1160 = insertelement <4 x float> %v1156, float %v1159, i64 3, !dbg !63
  %math.result315 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1160) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1161 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !63
  %math.packed316 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1161) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce46 = fadd float %local.reduce42, %v1159, !dbg !62, !frisk.local_reduction !34
  %v1163 = extractelement <4 x float> %v1051, i64 0, !dbg !61
  %v1164 = call float @llvm.amdgcn.exp2.f32(float %v1163), !dbg !61
  %v1165 = insertelement <4 x float> poison, float %v1164, i64 0, !dbg !63
  %local.reduce47 = fadd float %local.reduce43, %v1164, !dbg !62, !frisk.local_reduction !34
  %v1167 = extractelement <4 x float> %v1051, i64 1, !dbg !61
  %v1168 = call float @llvm.amdgcn.exp2.f32(float %v1167), !dbg !61
  %v1169 = insertelement <4 x float> %v1165, float %v1168, i64 1, !dbg !63
  %local.reduce48 = fadd float %local.reduce44, %v1168, !dbg !62, !frisk.local_reduction !34
  %v1171 = extractelement <4 x float> %v1051, i64 2, !dbg !61
  %v1172 = call float @llvm.amdgcn.exp2.f32(float %v1171), !dbg !61
  %v1173 = insertelement <4 x float> %v1169, float %v1172, i64 2, !dbg !63
  %local.reduce49 = fadd float %local.reduce45, %v1172, !dbg !62, !frisk.local_reduction !34
  %v1175 = extractelement <4 x float> %v1051, i64 3, !dbg !61
  %v1176 = call float @llvm.amdgcn.exp2.f32(float %v1175), !dbg !61
  %v1177 = insertelement <4 x float> %v1173, float %v1176, i64 3, !dbg !63
  %math.result317 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1177) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1178 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !63
  %math.packed318 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1178) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce50 = fadd float %local.reduce46, %v1176, !dbg !62, !frisk.local_reduction !34
  %v1180 = extractelement <4 x float> %v1052, i64 0, !dbg !61
  %v1181 = call float @llvm.amdgcn.exp2.f32(float %v1180), !dbg !61
  %v1182 = insertelement <4 x float> poison, float %v1181, i64 0, !dbg !63
  %local.reduce51 = fadd float %local.reduce47, %v1181, !dbg !62, !frisk.local_reduction !34
  %v1184 = extractelement <4 x float> %v1052, i64 1, !dbg !61
  %v1185 = call float @llvm.amdgcn.exp2.f32(float %v1184), !dbg !61
  %v1186 = insertelement <4 x float> %v1182, float %v1185, i64 1, !dbg !63
  %local.reduce52 = fadd float %local.reduce48, %v1185, !dbg !62, !frisk.local_reduction !34
  %v1188 = extractelement <4 x float> %v1052, i64 2, !dbg !61
  %v1189 = call float @llvm.amdgcn.exp2.f32(float %v1188), !dbg !61
  %v1190 = insertelement <4 x float> %v1186, float %v1189, i64 2, !dbg !63
  %local.reduce53 = fadd float %local.reduce49, %v1189, !dbg !62, !frisk.local_reduction !34
  %v1192 = extractelement <4 x float> %v1052, i64 3, !dbg !61
  %v1193 = call float @llvm.amdgcn.exp2.f32(float %v1192), !dbg !61
  %v1194 = insertelement <4 x float> %v1190, float %v1193, i64 3, !dbg !63
  %math.result319 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1194) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1195 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !63
  %math.packed320 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1195) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce54 = fadd float %local.reduce50, %v1193, !dbg !62, !frisk.local_reduction !34
  %v1197 = extractelement <4 x float> %v1053, i64 0, !dbg !61
  %v1198 = call float @llvm.amdgcn.exp2.f32(float %v1197), !dbg !61
  %v1199 = insertelement <4 x float> poison, float %v1198, i64 0, !dbg !63
  %local.reduce55 = fadd float %local.reduce51, %v1198, !dbg !62, !frisk.local_reduction !34
  %v1201 = extractelement <4 x float> %v1053, i64 1, !dbg !61
  %v1202 = call float @llvm.amdgcn.exp2.f32(float %v1201), !dbg !61
  %v1203 = insertelement <4 x float> %v1199, float %v1202, i64 1, !dbg !63
  %local.reduce56 = fadd float %local.reduce52, %v1202, !dbg !62, !frisk.local_reduction !34
  %v1205 = extractelement <4 x float> %v1053, i64 2, !dbg !61
  %v1206 = call float @llvm.amdgcn.exp2.f32(float %v1205), !dbg !61
  %v1207 = insertelement <4 x float> %v1203, float %v1206, i64 2, !dbg !63
  %local.reduce57 = fadd float %local.reduce53, %v1206, !dbg !62, !frisk.local_reduction !34
  %v1209 = extractelement <4 x float> %v1053, i64 3, !dbg !61
  %v1210 = call float @llvm.amdgcn.exp2.f32(float %v1209), !dbg !61
  %v1211 = insertelement <4 x float> %v1207, float %v1210, i64 3, !dbg !63
  %math.result321 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1211) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1212 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !63
  %math.packed322 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1212) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce58 = fadd float %local.reduce54, %v1210, !dbg !62, !frisk.local_reduction !34
  %local.reduce59 = fadd float %local.reduce55, %local.reduce56, !dbg !62, !frisk.local_reduction !34
  %local.reduce60 = fadd float %local.reduce57, %local.reduce58, !dbg !62, !frisk.local_reduction !34
  %v1213 = fadd float %local.reduce59, %local.reduce60, !dbg !62, !frisk.local_reduction !34
  %v1214 = extractelement <4 x float> %v1055, i64 0, !dbg !61
  %v1215 = call float @llvm.amdgcn.exp2.f32(float %v1214), !dbg !61
  %v1216 = insertelement <4 x float> poison, float %v1215, i64 0, !dbg !63
  %v1217 = extractelement <4 x float> %v1055, i64 1, !dbg !61
  %v1218 = call float @llvm.amdgcn.exp2.f32(float %v1217), !dbg !61
  %v1219 = insertelement <4 x float> %v1216, float %v1218, i64 1, !dbg !63
  %v1221 = extractelement <4 x float> %v1055, i64 2, !dbg !61
  %v1222 = call float @llvm.amdgcn.exp2.f32(float %v1221), !dbg !61
  %v1223 = insertelement <4 x float> %v1219, float %v1222, i64 2, !dbg !63
  %v1225 = extractelement <4 x float> %v1055, i64 3, !dbg !61
  %v1226 = call float @llvm.amdgcn.exp2.f32(float %v1225), !dbg !61
  %v1227 = insertelement <4 x float> %v1223, float %v1226, i64 3, !dbg !63
  %math.result323 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1227) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1228 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !63
  %math.packed324 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1228) #9, !dbg !62, !frisk.register_lifetimes !34
  %v1230 = extractelement <4 x float> %v1056, i64 0, !dbg !61
  %v1231 = call float @llvm.amdgcn.exp2.f32(float %v1230), !dbg !61
  %v1232 = insertelement <4 x float> poison, float %v1231, i64 0, !dbg !63
  %local.reduce = fadd float %v1215, %v1231, !dbg !62, !frisk.local_reduction !34
  %v1234 = extractelement <4 x float> %v1056, i64 1, !dbg !61
  %v1235 = call float @llvm.amdgcn.exp2.f32(float %v1234), !dbg !61
  %v1236 = insertelement <4 x float> %v1232, float %v1235, i64 1, !dbg !63
  %local.reduce1 = fadd float %v1218, %v1235, !dbg !62, !frisk.local_reduction !34
  %v1238 = extractelement <4 x float> %v1056, i64 2, !dbg !61
  %v1239 = call float @llvm.amdgcn.exp2.f32(float %v1238), !dbg !61
  %v1240 = insertelement <4 x float> %v1236, float %v1239, i64 2, !dbg !63
  %local.reduce2 = fadd float %v1222, %v1239, !dbg !62, !frisk.local_reduction !34
  %v1242 = extractelement <4 x float> %v1056, i64 3, !dbg !61
  %v1243 = call float @llvm.amdgcn.exp2.f32(float %v1242), !dbg !61
  %v1244 = insertelement <4 x float> %v1240, float %v1243, i64 3, !dbg !63
  %math.result325 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1244) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1245 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !63
  %math.packed326 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1245) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce3 = fadd float %v1226, %v1243, !dbg !62, !frisk.local_reduction !34
  %v1247 = extractelement <4 x float> %v1057, i64 0, !dbg !61
  %v1248 = call float @llvm.amdgcn.exp2.f32(float %v1247), !dbg !61
  %v1249 = insertelement <4 x float> poison, float %v1248, i64 0, !dbg !63
  %local.reduce4 = fadd float %local.reduce, %v1248, !dbg !62, !frisk.local_reduction !34
  %v1251 = extractelement <4 x float> %v1057, i64 1, !dbg !61
  %v1252 = call float @llvm.amdgcn.exp2.f32(float %v1251), !dbg !61
  %v1253 = insertelement <4 x float> %v1249, float %v1252, i64 1, !dbg !63
  %local.reduce5 = fadd float %local.reduce1, %v1252, !dbg !62, !frisk.local_reduction !34
  %v1255 = extractelement <4 x float> %v1057, i64 2, !dbg !61
  %v1256 = call float @llvm.amdgcn.exp2.f32(float %v1255), !dbg !61
  %v1257 = insertelement <4 x float> %v1253, float %v1256, i64 2, !dbg !63
  %local.reduce6 = fadd float %local.reduce2, %v1256, !dbg !62, !frisk.local_reduction !34
  %v1259 = extractelement <4 x float> %v1057, i64 3, !dbg !61
  %v1260 = call float @llvm.amdgcn.exp2.f32(float %v1259), !dbg !61
  %v1261 = insertelement <4 x float> %v1257, float %v1260, i64 3, !dbg !63
  %math.result327 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1261) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1262 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !63
  %math.packed328 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1262) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce7 = fadd float %local.reduce3, %v1260, !dbg !62, !frisk.local_reduction !34
  %v1264 = extractelement <4 x float> %v1058, i64 0, !dbg !61
  %v1265 = call float @llvm.amdgcn.exp2.f32(float %v1264), !dbg !61
  %v1266 = insertelement <4 x float> poison, float %v1265, i64 0, !dbg !63
  %local.reduce8 = fadd float %local.reduce4, %v1265, !dbg !62, !frisk.local_reduction !34
  %v1268 = extractelement <4 x float> %v1058, i64 1, !dbg !61
  %v1269 = call float @llvm.amdgcn.exp2.f32(float %v1268), !dbg !61
  %v1270 = insertelement <4 x float> %v1266, float %v1269, i64 1, !dbg !63
  %local.reduce9 = fadd float %local.reduce5, %v1269, !dbg !62, !frisk.local_reduction !34
  %v1272 = extractelement <4 x float> %v1058, i64 2, !dbg !61
  %v1273 = call float @llvm.amdgcn.exp2.f32(float %v1272), !dbg !61
  %v1274 = insertelement <4 x float> %v1270, float %v1273, i64 2, !dbg !63
  %local.reduce10 = fadd float %local.reduce6, %v1273, !dbg !62, !frisk.local_reduction !34
  %v1276 = extractelement <4 x float> %v1058, i64 3, !dbg !61
  %v1277 = call float @llvm.amdgcn.exp2.f32(float %v1276), !dbg !61
  %v1278 = insertelement <4 x float> %v1274, float %v1277, i64 3, !dbg !63
  %math.result329 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1278) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1279 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !63
  %math.packed330 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1279) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce11 = fadd float %local.reduce7, %v1277, !dbg !62, !frisk.local_reduction !34
  %v1281 = extractelement <4 x float> %v1059, i64 0, !dbg !61
  %v1282 = call float @llvm.amdgcn.exp2.f32(float %v1281), !dbg !61
  %v1283 = insertelement <4 x float> poison, float %v1282, i64 0, !dbg !63
  %local.reduce12 = fadd float %local.reduce8, %v1282, !dbg !62, !frisk.local_reduction !34
  %v1285 = extractelement <4 x float> %v1059, i64 1, !dbg !61
  %v1286 = call float @llvm.amdgcn.exp2.f32(float %v1285), !dbg !61
  %v1287 = insertelement <4 x float> %v1283, float %v1286, i64 1, !dbg !63
  %local.reduce13 = fadd float %local.reduce9, %v1286, !dbg !62, !frisk.local_reduction !34
  %v1289 = extractelement <4 x float> %v1059, i64 2, !dbg !61
  %v1290 = call float @llvm.amdgcn.exp2.f32(float %v1289), !dbg !61
  %v1291 = insertelement <4 x float> %v1287, float %v1290, i64 2, !dbg !63
  %local.reduce14 = fadd float %local.reduce10, %v1290, !dbg !62, !frisk.local_reduction !34
  %v1293 = extractelement <4 x float> %v1059, i64 3, !dbg !61
  %v1294 = call float @llvm.amdgcn.exp2.f32(float %v1293), !dbg !61
  %v1295 = insertelement <4 x float> %v1291, float %v1294, i64 3, !dbg !63
  %math.result331 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1295) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1296 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !63
  %math.packed332 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1296) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce15 = fadd float %local.reduce11, %v1294, !dbg !62, !frisk.local_reduction !34
  %v1298 = extractelement <4 x float> %v1060, i64 0, !dbg !61
  %v1299 = call float @llvm.amdgcn.exp2.f32(float %v1298), !dbg !61
  %v1300 = insertelement <4 x float> poison, float %v1299, i64 0, !dbg !63
  %local.reduce16 = fadd float %local.reduce12, %v1299, !dbg !62, !frisk.local_reduction !34
  %v1302 = extractelement <4 x float> %v1060, i64 1, !dbg !61
  %v1303 = call float @llvm.amdgcn.exp2.f32(float %v1302), !dbg !61
  %v1304 = insertelement <4 x float> %v1300, float %v1303, i64 1, !dbg !63
  %local.reduce17 = fadd float %local.reduce13, %v1303, !dbg !62, !frisk.local_reduction !34
  %v1306 = extractelement <4 x float> %v1060, i64 2, !dbg !61
  %v1307 = call float @llvm.amdgcn.exp2.f32(float %v1306), !dbg !61
  %v1308 = insertelement <4 x float> %v1304, float %v1307, i64 2, !dbg !63
  %local.reduce18 = fadd float %local.reduce14, %v1307, !dbg !62, !frisk.local_reduction !34
  %v1310 = extractelement <4 x float> %v1060, i64 3, !dbg !61
  %v1311 = call float @llvm.amdgcn.exp2.f32(float %v1310), !dbg !61
  %v1312 = insertelement <4 x float> %v1308, float %v1311, i64 3, !dbg !63
  %math.result333 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1312) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1313 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !63
  %math.packed334 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1313) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce19 = fadd float %local.reduce15, %v1311, !dbg !62, !frisk.local_reduction !34
  %v1315 = extractelement <4 x float> %v1061, i64 0, !dbg !61
  %v1316 = call float @llvm.amdgcn.exp2.f32(float %v1315), !dbg !61
  %v1317 = insertelement <4 x float> poison, float %v1316, i64 0, !dbg !63
  %local.reduce20 = fadd float %local.reduce16, %v1316, !dbg !62, !frisk.local_reduction !34
  %v1319 = extractelement <4 x float> %v1061, i64 1, !dbg !61
  %v1320 = call float @llvm.amdgcn.exp2.f32(float %v1319), !dbg !61
  %v1321 = insertelement <4 x float> %v1317, float %v1320, i64 1, !dbg !63
  %local.reduce21 = fadd float %local.reduce17, %v1320, !dbg !62, !frisk.local_reduction !34
  %v1323 = extractelement <4 x float> %v1061, i64 2, !dbg !61
  %v1324 = call float @llvm.amdgcn.exp2.f32(float %v1323), !dbg !61
  %v1325 = insertelement <4 x float> %v1321, float %v1324, i64 2, !dbg !63
  %local.reduce22 = fadd float %local.reduce18, %v1324, !dbg !62, !frisk.local_reduction !34
  %v1327 = extractelement <4 x float> %v1061, i64 3, !dbg !61
  %v1328 = call float @llvm.amdgcn.exp2.f32(float %v1327), !dbg !61
  %v1329 = insertelement <4 x float> %v1325, float %v1328, i64 3, !dbg !63
  %math.result335 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1329) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1330 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !63
  %math.packed336 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1330) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce23 = fadd float %local.reduce19, %v1328, !dbg !62, !frisk.local_reduction !34
  %v1332 = extractelement <4 x float> %v1062, i64 0, !dbg !61
  %v1333 = call float @llvm.amdgcn.exp2.f32(float %v1332), !dbg !61
  %v1334 = insertelement <4 x float> poison, float %v1333, i64 0, !dbg !63
  %local.reduce24 = fadd float %local.reduce20, %v1333, !dbg !62, !frisk.local_reduction !34
  %v1336 = extractelement <4 x float> %v1062, i64 1, !dbg !61
  %v1337 = call float @llvm.amdgcn.exp2.f32(float %v1336), !dbg !61
  %v1338 = insertelement <4 x float> %v1334, float %v1337, i64 1, !dbg !63
  %local.reduce25 = fadd float %local.reduce21, %v1337, !dbg !62, !frisk.local_reduction !34
  %v1340 = extractelement <4 x float> %v1062, i64 2, !dbg !61
  %v1341 = call float @llvm.amdgcn.exp2.f32(float %v1340), !dbg !61
  %v1342 = insertelement <4 x float> %v1338, float %v1341, i64 2, !dbg !63
  %local.reduce26 = fadd float %local.reduce22, %v1341, !dbg !62, !frisk.local_reduction !34
  %v1344 = extractelement <4 x float> %v1062, i64 3, !dbg !61
  %v1345 = call float @llvm.amdgcn.exp2.f32(float %v1344), !dbg !61
  %v1346 = insertelement <4 x float> %v1342, float %v1345, i64 3, !dbg !63
  %math.result337 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1346) #9, !dbg !63, !frisk.register_lifetimes !34
  %v1347 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !63
  %math.packed338 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1347) #9, !dbg !62, !frisk.register_lifetimes !34
  %local.reduce27 = fadd float %local.reduce23, %v1345, !dbg !62, !frisk.local_reduction !34
  %local.reduce28 = fadd float %local.reduce24, %local.reduce25, !dbg !62, !frisk.local_reduction !34
  %local.reduce29 = fadd float %local.reduce26, %local.reduce27, !dbg !62, !frisk.local_reduction !34
  %v1348 = fadd float %local.reduce28, %local.reduce29, !dbg !62, !frisk.local_reduction !34
  %v1349 = bitcast float %v1213 to i32, !dbg !62
  %v1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v1349), !dbg !62
  %v1351 = bitcast i32 %v1350 to float, !dbg !62
  %v1352 = fadd float %v1213, %v1351, !dbg !62
  %v1353 = bitcast float %v1352 to i32, !dbg !62
  %v1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v1353), !dbg !62
  %v1355 = bitcast i32 %v1354 to float, !dbg !62
  %v1356 = fadd float %v1352, %v1355, !dbg !62
  %v1357 = fadd float %v1356, %v1070, !dbg !62
  %v1358 = insertelement <1 x float> poison, float %v1357, i64 0, !dbg !39
  %v1359 = insertvalue [2 x <1 x float>] poison, <1 x float> %v1358, 0, !dbg !39
  %v1360 = bitcast float %v1348 to i32, !dbg !62
  %v1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v986, i32 %v1360), !dbg !62
  %v1362 = bitcast i32 %v1361 to float, !dbg !62
  %v1363 = fadd float %v1348, %v1362, !dbg !62
  %v1364 = bitcast float %v1363 to i32, !dbg !62
  %v1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v994, i32 %v1364), !dbg !62
  %v1366 = bitcast i32 %v1365 to float, !dbg !62
  %v1367 = fadd float %v1363, %v1366, !dbg !62
  %v1368 = fadd float %v1367, %v1078, !dbg !62
  %v1369 = insertelement <1 x float> poison, float %v1368, i64 0, !dbg !39
  %v1370 = insertvalue [2 x <1 x float>] %v1359, <1 x float> %v1369, 1, !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !48
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %address.thread864 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !44, !range !67, !frisk.register_lifetimes !34
  %address.local865 = shl nuw nsw i32 %address.thread864, 3, !dbg !44
  %address.local866 = and i32 %address.local865, 120, !dbg !44
  %address.local867 = shl nuw nsw i32 %address.thread864, 5, !dbg !44
  %address.local868 = and i32 %address.local867, 1536, !dbg !44
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !44
  %address.local870 = zext nneg i32 %address.local869 to i64, !dbg !44
  %address.local871 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !44
  %v1371 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !44
  %v1372 = bitcast <4 x i32> %v1371 to <8 x half>, !dbg !44
  %address.local872 = zext nneg i32 %address.thread864 to i64, !dbg !40
  %address.local873 = and i64 %address.local872, 60, !dbg !44
  %address.local874 = add nuw nsw i64 %address.local873, %address.local872, !dbg !44
  %address.local875 = shl nuw nsw i64 %address.local874, 3, !dbg !44
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local877 = and i64 %address.local872, 63, !dbg !42
  %address.local878 = lshr i64 %address.local877, 4, !dbg !44
  %address.local879 = mul nuw nsw i64 %address.local878, 384, !dbg !44
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) %address.local876, i64 %address.local879, !dbg !44
  %address.local881 = lshr i64 %address.local872, 2, !dbg !47
  %address.local882 = mul nsw i64 %address.local881, -32, !dbg !44
  %address.local883 = getelementptr [2 x i8], ptr addrspace(3) %address.local880, i64 %address.local882, !dbg !44
  %address.local884 = getelementptr i8, ptr addrspace(3) %address.local883, i64 256, !dbg !44
  %v1373 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !44
  %v1374 = bitcast <4 x i32> %v1373 to <8 x half>, !dbg !44
  %address.local885 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local886 = getelementptr [2 x i8], ptr addrspace(3) %address.local885, i64 %address.local879, !dbg !44
  %address.local887 = getelementptr [2 x i8], ptr addrspace(3) %address.local886, i64 %address.local882, !dbg !44
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local887, i64 512, !dbg !44
  %v1375 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !44
  %v1376 = bitcast <4 x i32> %v1375 to <8 x half>, !dbg !44
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local890 = getelementptr [2 x i8], ptr addrspace(3) %address.local889, i64 %address.local879, !dbg !44
  %address.local891 = getelementptr [2 x i8], ptr addrspace(3) %address.local890, i64 %address.local882, !dbg !44
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local891, i64 768, !dbg !44
  %v1377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !44
  %v1378 = bitcast <4 x i32> %v1377 to <8 x half>, !dbg !44
  %address.local893 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !44
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local893, i64 4096, !dbg !44
  %v1379 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !44
  %v1380 = bitcast <4 x i32> %v1379 to <8 x half>, !dbg !44
  %address.local895 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local896 = getelementptr [2 x i8], ptr addrspace(3) %address.local895, i64 %address.local879, !dbg !44
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) %address.local896, i64 %address.local882, !dbg !44
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local897, i64 4352, !dbg !44
  %v1381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !44
  %v1382 = bitcast <4 x i32> %v1381 to <8 x half>, !dbg !44
  %address.local899 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local900 = getelementptr [2 x i8], ptr addrspace(3) %address.local899, i64 %address.local879, !dbg !44
  %address.local901 = getelementptr [2 x i8], ptr addrspace(3) %address.local900, i64 %address.local882, !dbg !44
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local901, i64 4608, !dbg !44
  %v1383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !44
  %v1384 = bitcast <4 x i32> %v1383 to <8 x half>, !dbg !44
  %address.local903 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) %address.local903, i64 %address.local879, !dbg !44
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) %address.local904, i64 %address.local882, !dbg !44
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 4864, !dbg !44
  %v1385 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !44
  %v1386 = bitcast <4 x i32> %v1385 to <8 x half>, !dbg !44
  %v1387 = add nuw nsw i64 %v217, %v16, !dbg !48
  %v1388 = shl i64 %v1387, 19, !dbg !48
  %v1389 = and i64 %v1388, 16252928, !dbg !48
  %v1390 = add i64 %v1389, %v223, !dbg !48
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %v1391 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %v1390, !dbg !48
  %v1392 = getelementptr i8, ptr addrspace(1) %v1391, i64 64, !dbg !48
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !40
  %address.local908 = and i64 %address.local907, 1536, !dbg !40
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !40
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !40
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !40
  %v1393 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !48
  %v1394 = zext i32 %v1393 to i64, !dbg !48
  %v1395 = inttoptr i64 %v1394 to ptr addrspace(3), !dbg !48
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !48
  %address.local913 = and i64 %address.local912, 8064, !dbg !48
  %address.local914 = shl nuw nsw i64 %address.local872, 3, !dbg !48
  %address.local915 = and i64 %address.local914, 24, !dbg !48
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !48
  %address.local917 = getelementptr [2 x i8], ptr addrspace(1) %v1392, i64 %address.local916, !dbg !48
  %v1396 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %v1397 = ptrtoint ptr addrspace(1) %address.local917 to i64, !dbg !48
  %v1398 = sub i64 %v1397, %v1396, !dbg !48
  %dma.buffer.offset1558 = trunc i64 %v1398 to i32, !dbg !48
  %v1399 = trunc i64 %v1396 to i32, !dbg !48
  %v1400 = insertelement <4 x i32> poison, i32 %v1399, i64 0, !dbg !48
  %v1401 = lshr i64 %v1396, 32, !dbg !48
  %v1402 = trunc i64 %v1401 to i32, !dbg !48
  %v1403 = insertelement <4 x i32> %v1400, i32 %v1402, i64 1, !dbg !48
  %v1404 = insertelement <4 x i32> %v1403, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1559 = insertelement <4 x i32> %v1404, i32 131072, i64 3, !dbg !48
  %v1405 = ptrtoint ptr addrspace(3) %v1395 to i32, !dbg !48
  %dma.buffer.lds1560 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1405), !dbg !48
  %v1406 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1559) #10, !dbg !48
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !40
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !40
  %address.local920 = and i64 %address.local919, 3584, !dbg !40
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !40
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !40
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !40
  %v1407 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !48
  %v1408 = zext i32 %v1407 to i64, !dbg !48
  %v1409 = inttoptr i64 %v1408 to ptr addrspace(3), !dbg !48
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !48
  %address.local925 = and i64 %address.local924, 16256, !dbg !48
  %address.local926 = shl nuw nsw i64 %address.local872, 3, !dbg !48
  %address.local927 = and i64 %address.local926, 24, !dbg !48
  %address.local928 = or disjoint i64 %address.local925, %address.local927, !dbg !48
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %v1392, i64 %address.local928, !dbg !48
  %v1410 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %v1411 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !48
  %v1412 = sub i64 %v1411, %v1410, !dbg !48
  %dma.buffer.offset1555 = trunc i64 %v1412 to i32, !dbg !48
  %v1413 = trunc i64 %v1410 to i32, !dbg !48
  %v1414 = insertelement <4 x i32> poison, i32 %v1413, i64 0, !dbg !48
  %v1415 = lshr i64 %v1410, 32, !dbg !48
  %v1416 = trunc i64 %v1415 to i32, !dbg !48
  %v1417 = insertelement <4 x i32> %v1414, i32 %v1416, i64 1, !dbg !48
  %v1418 = insertelement <4 x i32> %v1417, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1556 = insertelement <4 x i32> %v1418, i32 131072, i64 3, !dbg !48
  %v1419 = ptrtoint ptr addrspace(3) %v1409 to i32, !dbg !48
  %dma.buffer.lds1557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1419), !dbg !48
  %v1420 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1556) #10, !dbg !48
  %v1421 = shufflevector <8 x half> %v1372, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1422 = extractvalue [2 x <8 x float>] %v206, 0, !dbg !64
  %v1423 = shufflevector <8 x float> %v1422, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %v1424 = fmul <4 x float> %v1423, %v1067, !dbg !64
  %v1425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1421, <4 x float> %v1424) #9, !dbg !44
  %v1426 = shufflevector <8 x half> %v1374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1426, <4 x float> %v1425) #9, !dbg !44
  %v1428 = shufflevector <8 x half> %v1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1428, <4 x float> %v1427) #9, !dbg !44
  %v1430 = shufflevector <8 x half> %v1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1430, <4 x float> %v1429) #9, !dbg !44
  %v1432 = shufflevector <8 x half> %v1380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1432, <4 x float> %v1431) #9, !dbg !44
  %v1434 = shufflevector <8 x half> %v1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1434, <4 x float> %v1433) #9, !dbg !44
  %v1436 = shufflevector <8 x half> %v1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1436, <4 x float> %v1435) #9, !dbg !44
  %v1438 = shufflevector <8 x half> %v1386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1438, <4 x float> %v1437) #9, !dbg !44
  %v1440 = shufflevector <8 x half> %v1372, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1441 = shufflevector <8 x float> %v1422, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %v1442 = fmul <4 x float> %v1441, %v1067, !dbg !64
  %v1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1440, <4 x float> %v1442) #9, !dbg !44
  %v1444 = shufflevector <8 x half> %v1374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1444, <4 x float> %v1443) #9, !dbg !44
  %v1446 = shufflevector <8 x half> %v1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1446, <4 x float> %v1445) #9, !dbg !44
  %v1448 = shufflevector <8 x half> %v1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1448, <4 x float> %v1447) #9, !dbg !44
  %v1450 = shufflevector <8 x half> %v1380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1450, <4 x float> %v1449) #9, !dbg !44
  %v1452 = shufflevector <8 x half> %v1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1452, <4 x float> %v1451) #9, !dbg !44
  %v1454 = shufflevector <8 x half> %v1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1454, <4 x float> %v1453) #9, !dbg !44
  %v1456 = shufflevector <8 x half> %v1386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1456, <4 x float> %v1455) #9, !dbg !44
  %v1458 = extractvalue [2 x <8 x float>] %v206, 1, !dbg !64
  %v1459 = shufflevector <8 x float> %v1458, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %v1460 = fmul <4 x float> %v1459, %v1075, !dbg !64
  %v1461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1421, <4 x float> %v1460) #9, !dbg !44
  %v1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1426, <4 x float> %v1461) #9, !dbg !44
  %v1463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1428, <4 x float> %v1462) #9, !dbg !44
  %v1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1430, <4 x float> %v1463) #9, !dbg !44
  %v1465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1432, <4 x float> %v1464) #9, !dbg !44
  %v1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1434, <4 x float> %v1465) #9, !dbg !44
  %v1467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1436, <4 x float> %v1466) #9, !dbg !44
  %v1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1438, <4 x float> %v1467) #9, !dbg !44
  %v1469 = shufflevector <8 x float> %v1458, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %v1470 = fmul <4 x float> %v1469, %v1075, !dbg !64
  %v1471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1440, <4 x float> %v1470) #9, !dbg !44
  %v1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1444, <4 x float> %v1471) #9, !dbg !44
  %v1473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1446, <4 x float> %v1472) #9, !dbg !44
  %v1474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1448, <4 x float> %v1473) #9, !dbg !44
  %v1475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1450, <4 x float> %v1474) #9, !dbg !44
  %v1476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1452, <4 x float> %v1475) #9, !dbg !44
  %v1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1454, <4 x float> %v1476) #9, !dbg !44
  %v1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1456, <4 x float> %v1477) #9, !dbg !44
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !48
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %v1479 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !44
  %v1480 = bitcast <4 x i32> %v1479 to <8 x half>, !dbg !44
  %v1481 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !44
  %v1482 = bitcast <4 x i32> %v1481 to <8 x half>, !dbg !44
  %v1483 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !44
  %v1484 = bitcast <4 x i32> %v1483 to <8 x half>, !dbg !44
  %v1485 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !44
  %v1486 = bitcast <4 x i32> %v1485 to <8 x half>, !dbg !44
  %v1487 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !44
  %v1488 = bitcast <4 x i32> %v1487 to <8 x half>, !dbg !44
  %v1489 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !44
  %v1490 = bitcast <4 x i32> %v1489 to <8 x half>, !dbg !44
  %v1491 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !44
  %v1492 = bitcast <4 x i32> %v1491 to <8 x half>, !dbg !44
  %v1493 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !44
  %v1494 = bitcast <4 x i32> %v1493 to <8 x half>, !dbg !44
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %v1495 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %v1390, !dbg !48
  %v1496 = getelementptr i8, ptr addrspace(1) %v1495, i64 128, !dbg !48
  %v1497 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !48
  %v1498 = zext i32 %v1497 to i64, !dbg !48
  %v1499 = inttoptr i64 %v1498 to ptr addrspace(3), !dbg !48
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %v1496, i64 %address.local916, !dbg !48
  %v1500 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %v1501 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !48
  %v1502 = sub i64 %v1501, %v1500, !dbg !48
  %dma.buffer.offset1552 = trunc i64 %v1502 to i32, !dbg !48
  %v1503 = trunc i64 %v1500 to i32, !dbg !48
  %v1504 = insertelement <4 x i32> poison, i32 %v1503, i64 0, !dbg !48
  %v1505 = lshr i64 %v1500, 32, !dbg !48
  %v1506 = trunc i64 %v1505 to i32, !dbg !48
  %v1507 = insertelement <4 x i32> %v1504, i32 %v1506, i64 1, !dbg !48
  %v1508 = insertelement <4 x i32> %v1507, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1553 = insertelement <4 x i32> %v1508, i32 131072, i64 3, !dbg !48
  %v1509 = ptrtoint ptr addrspace(3) %v1499 to i32, !dbg !48
  %dma.buffer.lds1554 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1509), !dbg !48
  %v1510 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1554, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1553) #10, !dbg !48
  %v1511 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !48
  %v1512 = zext i32 %v1511 to i64, !dbg !48
  %v1513 = inttoptr i64 %v1512 to ptr addrspace(3), !dbg !48
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %v1496, i64 %address.local928, !dbg !48
  %v1514 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %v1515 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !48
  %v1516 = sub i64 %v1515, %v1514, !dbg !48
  %dma.buffer.offset1549 = trunc i64 %v1516 to i32, !dbg !48
  %v1517 = trunc i64 %v1514 to i32, !dbg !48
  %v1518 = insertelement <4 x i32> poison, i32 %v1517, i64 0, !dbg !48
  %v1519 = lshr i64 %v1514, 32, !dbg !48
  %v1520 = trunc i64 %v1519 to i32, !dbg !48
  %v1521 = insertelement <4 x i32> %v1518, i32 %v1520, i64 1, !dbg !48
  %v1522 = insertelement <4 x i32> %v1521, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1550 = insertelement <4 x i32> %v1522, i32 131072, i64 3, !dbg !48
  %v1523 = ptrtoint ptr addrspace(3) %v1513 to i32, !dbg !48
  %dma.buffer.lds1551 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1523), !dbg !48
  %v1524 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1551, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1550) #10, !dbg !48
  %v1525 = shufflevector <8 x half> %v1480, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1526 = extractvalue [2 x <8 x float>] %v209, 0, !dbg !64
  %v1527 = shufflevector <8 x float> %v1526, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %v1528 = fmul <4 x float> %v1527, %v1067, !dbg !64
  %v1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1525, <4 x float> %v1528) #9, !dbg !44
  %v1530 = shufflevector <8 x half> %v1482, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1530, <4 x float> %v1529) #9, !dbg !44
  %v1532 = shufflevector <8 x half> %v1484, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1532, <4 x float> %v1531) #9, !dbg !44
  %v1534 = shufflevector <8 x half> %v1486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1534, <4 x float> %v1533) #9, !dbg !44
  %v1536 = shufflevector <8 x half> %v1488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1536, <4 x float> %v1535) #9, !dbg !44
  %v1538 = shufflevector <8 x half> %v1490, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1538, <4 x float> %v1537) #9, !dbg !44
  %v1540 = shufflevector <8 x half> %v1492, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1540, <4 x float> %v1539) #9, !dbg !44
  %v1542 = shufflevector <8 x half> %v1494, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1542, <4 x float> %v1541) #9, !dbg !44
  %v1544 = shufflevector <8 x half> %v1480, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1545 = shufflevector <8 x float> %v1526, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %v1546 = fmul <4 x float> %v1545, %v1067, !dbg !64
  %v1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1544, <4 x float> %v1546) #9, !dbg !44
  %v1548 = shufflevector <8 x half> %v1482, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1548, <4 x float> %v1547) #9, !dbg !44
  %v1550 = shufflevector <8 x half> %v1484, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1550, <4 x float> %v1549) #9, !dbg !44
  %v1552 = shufflevector <8 x half> %v1486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1552, <4 x float> %v1551) #9, !dbg !44
  %v1554 = shufflevector <8 x half> %v1488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1554, <4 x float> %v1553) #9, !dbg !44
  %v1556 = shufflevector <8 x half> %v1490, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1556, <4 x float> %v1555) #9, !dbg !44
  %v1558 = shufflevector <8 x half> %v1492, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1558, <4 x float> %v1557) #9, !dbg !44
  %v1560 = shufflevector <8 x half> %v1494, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1560, <4 x float> %v1559) #9, !dbg !44
  %v1562 = extractvalue [2 x <8 x float>] %v209, 1, !dbg !64
  %v1563 = shufflevector <8 x float> %v1562, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %v1564 = fmul <4 x float> %v1563, %v1075, !dbg !64
  %v1565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1525, <4 x float> %v1564) #9, !dbg !44
  %v1566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1530, <4 x float> %v1565) #9, !dbg !44
  %v1567 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1532, <4 x float> %v1566) #9, !dbg !44
  %v1568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1534, <4 x float> %v1567) #9, !dbg !44
  %v1569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1536, <4 x float> %v1568) #9, !dbg !44
  %v1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1538, <4 x float> %v1569) #9, !dbg !44
  %v1571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1540, <4 x float> %v1570) #9, !dbg !44
  %v1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1542, <4 x float> %v1571) #9, !dbg !44
  %v1573 = shufflevector <8 x float> %v1562, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %v1574 = fmul <4 x float> %v1573, %v1075, !dbg !64
  %v1575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1544, <4 x float> %v1574) #9, !dbg !44
  %v1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1548, <4 x float> %v1575) #9, !dbg !44
  %v1577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1550, <4 x float> %v1576) #9, !dbg !44
  %v1578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1552, <4 x float> %v1577) #9, !dbg !44
  %v1579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1554, <4 x float> %v1578) #9, !dbg !44
  %v1580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1556, <4 x float> %v1579) #9, !dbg !44
  %v1581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1558, <4 x float> %v1580) #9, !dbg !44
  %v1582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1560, <4 x float> %v1581) #9, !dbg !44
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !48
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %v1583 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !44
  %v1584 = bitcast <4 x i32> %v1583 to <8 x half>, !dbg !44
  %v1585 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !44
  %v1586 = bitcast <4 x i32> %v1585 to <8 x half>, !dbg !44
  %v1587 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !44
  %v1588 = bitcast <4 x i32> %v1587 to <8 x half>, !dbg !44
  %v1589 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !44
  %v1590 = bitcast <4 x i32> %v1589 to <8 x half>, !dbg !44
  %v1591 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !44
  %v1592 = bitcast <4 x i32> %v1591 to <8 x half>, !dbg !44
  %v1593 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !44
  %v1594 = bitcast <4 x i32> %v1593 to <8 x half>, !dbg !44
  %v1595 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !44
  %v1596 = bitcast <4 x i32> %v1595 to <8 x half>, !dbg !44
  %v1597 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !44
  %v1598 = bitcast <4 x i32> %v1597 to <8 x half>, !dbg !44
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %v1599 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %v1390, !dbg !48
  %v1600 = getelementptr i8, ptr addrspace(1) %v1599, i64 192, !dbg !48
  %v1601 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !48
  %v1602 = zext i32 %v1601 to i64, !dbg !48
  %v1603 = inttoptr i64 %v1602 to ptr addrspace(3), !dbg !48
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %v1600, i64 %address.local916, !dbg !48
  %v1604 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %v1605 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !48
  %v1606 = sub i64 %v1605, %v1604, !dbg !48
  %dma.buffer.offset1546 = trunc i64 %v1606 to i32, !dbg !48
  %v1607 = trunc i64 %v1604 to i32, !dbg !48
  %v1608 = insertelement <4 x i32> poison, i32 %v1607, i64 0, !dbg !48
  %v1609 = lshr i64 %v1604, 32, !dbg !48
  %v1610 = trunc i64 %v1609 to i32, !dbg !48
  %v1611 = insertelement <4 x i32> %v1608, i32 %v1610, i64 1, !dbg !48
  %v1612 = insertelement <4 x i32> %v1611, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1547 = insertelement <4 x i32> %v1612, i32 131072, i64 3, !dbg !48
  %v1613 = ptrtoint ptr addrspace(3) %v1603 to i32, !dbg !48
  %dma.buffer.lds1548 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1613), !dbg !48
  %v1614 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1548, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1547) #10, !dbg !48
  %v1615 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !48
  %v1616 = zext i32 %v1615 to i64, !dbg !48
  %v1617 = inttoptr i64 %v1616 to ptr addrspace(3), !dbg !48
  %address.local933 = getelementptr [2 x i8], ptr addrspace(1) %v1600, i64 %address.local928, !dbg !48
  %v1618 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %v1619 = ptrtoint ptr addrspace(1) %address.local933 to i64, !dbg !48
  %v1620 = sub i64 %v1619, %v1618, !dbg !48
  %dma.buffer.offset = trunc i64 %v1620 to i32, !dbg !48
  %v1621 = trunc i64 %v1618 to i32, !dbg !48
  %v1622 = insertelement <4 x i32> poison, i32 %v1621, i64 0, !dbg !48
  %v1623 = lshr i64 %v1618, 32, !dbg !48
  %v1624 = trunc i64 %v1623 to i32, !dbg !48
  %v1625 = insertelement <4 x i32> %v1622, i32 %v1624, i64 1, !dbg !48
  %v1626 = insertelement <4 x i32> %v1625, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc = insertelement <4 x i32> %v1626, i32 131072, i64 3, !dbg !48
  %v1627 = ptrtoint ptr addrspace(3) %v1617 to i32, !dbg !48
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1627), !dbg !48
  %v1628 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !48
  %v1629 = shufflevector <8 x half> %v1584, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1630 = extractvalue [2 x <8 x float>] %v210, 0, !dbg !64
  %v1631 = shufflevector <8 x float> %v1630, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %v1632 = fmul <4 x float> %v1631, %v1067, !dbg !64
  %v1633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1629, <4 x float> %v1632) #9, !dbg !44
  %v1634 = shufflevector <8 x half> %v1586, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1634, <4 x float> %v1633) #9, !dbg !44
  %v1636 = shufflevector <8 x half> %v1588, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1636, <4 x float> %v1635) #9, !dbg !44
  %v1638 = shufflevector <8 x half> %v1590, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1638, <4 x float> %v1637) #9, !dbg !44
  %v1640 = shufflevector <8 x half> %v1592, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1640, <4 x float> %v1639) #9, !dbg !44
  %v1642 = shufflevector <8 x half> %v1594, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1642, <4 x float> %v1641) #9, !dbg !44
  %v1644 = shufflevector <8 x half> %v1596, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1644, <4 x float> %v1643) #9, !dbg !44
  %v1646 = shufflevector <8 x half> %v1598, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1646, <4 x float> %v1645) #9, !dbg !44
  %v1648 = shufflevector <8 x half> %v1584, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1649 = shufflevector <8 x float> %v1630, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %v1650 = fmul <4 x float> %v1649, %v1067, !dbg !64
  %v1651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1648, <4 x float> %v1650) #9, !dbg !44
  %v1652 = shufflevector <8 x half> %v1586, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1652, <4 x float> %v1651) #9, !dbg !44
  %v1654 = shufflevector <8 x half> %v1588, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1654, <4 x float> %v1653) #9, !dbg !44
  %v1656 = shufflevector <8 x half> %v1590, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1656, <4 x float> %v1655) #9, !dbg !44
  %v1658 = shufflevector <8 x half> %v1592, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1658, <4 x float> %v1657) #9, !dbg !44
  %v1660 = shufflevector <8 x half> %v1594, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1660, <4 x float> %v1659) #9, !dbg !44
  %v1662 = shufflevector <8 x half> %v1596, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1662, <4 x float> %v1661) #9, !dbg !44
  %v1664 = shufflevector <8 x half> %v1598, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1664, <4 x float> %v1663) #9, !dbg !44
  %v1666 = extractvalue [2 x <8 x float>] %v210, 1, !dbg !64
  %v1667 = shufflevector <8 x float> %v1666, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %v1668 = fmul <4 x float> %v1667, %v1075, !dbg !64
  %v1669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1629, <4 x float> %v1668) #9, !dbg !44
  %v1670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1634, <4 x float> %v1669) #9, !dbg !44
  %v1671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1636, <4 x float> %v1670) #9, !dbg !44
  %v1672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1638, <4 x float> %v1671) #9, !dbg !44
  %v1673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1640, <4 x float> %v1672) #9, !dbg !44
  %v1674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1642, <4 x float> %v1673) #9, !dbg !44
  %v1675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1644, <4 x float> %v1674) #9, !dbg !44
  %v1676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1646, <4 x float> %v1675) #9, !dbg !44
  %v1677 = shufflevector <8 x float> %v1666, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %v1678 = fmul <4 x float> %v1677, %v1075, !dbg !64
  %v1679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1648, <4 x float> %v1678) #9, !dbg !44
  %v1680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1652, <4 x float> %v1679) #9, !dbg !44
  %v1681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1654, <4 x float> %v1680) #9, !dbg !44
  %v1682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1656, <4 x float> %v1681) #9, !dbg !44
  %v1683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1658, <4 x float> %v1682) #9, !dbg !44
  %v1684 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1660, <4 x float> %v1683) #9, !dbg !44
  %v1685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1662, <4 x float> %v1684) #9, !dbg !44
  %v1686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1664, <4 x float> %v1685) #9, !dbg !44
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !48
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %v1687 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !44
  %v1688 = bitcast <4 x i32> %v1687 to <8 x half>, !dbg !44
  %v1689 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !44
  %v1690 = bitcast <4 x i32> %v1689 to <8 x half>, !dbg !44
  %v1691 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !44
  %v1692 = bitcast <4 x i32> %v1691 to <8 x half>, !dbg !44
  %v1693 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !44
  %v1694 = bitcast <4 x i32> %v1693 to <8 x half>, !dbg !44
  %v1695 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !44
  %v1696 = bitcast <4 x i32> %v1695 to <8 x half>, !dbg !44
  %v1697 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !44
  %v1698 = bitcast <4 x i32> %v1697 to <8 x half>, !dbg !44
  %v1699 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !44
  %v1700 = bitcast <4 x i32> %v1699 to <8 x half>, !dbg !44
  %v1701 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !44
  %v1702 = bitcast <4 x i32> %v1701 to <8 x half>, !dbg !44
  %v1703 = shufflevector <8 x half> %v1688, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1704 = extractvalue [2 x <8 x float>] %v211, 0, !dbg !64
  %v1705 = shufflevector <8 x float> %v1704, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %v1706 = fmul <4 x float> %v1705, %v1067, !dbg !64
  %v1707 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1703, <4 x float> %v1706) #9, !dbg !44
  %v1708 = shufflevector <8 x half> %v1690, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1708, <4 x float> %v1707) #9, !dbg !44
  %v1710 = shufflevector <8 x half> %v1692, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1710, <4 x float> %v1709) #9, !dbg !44
  %v1712 = shufflevector <8 x half> %v1694, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1712, <4 x float> %v1711) #9, !dbg !44
  %v1714 = shufflevector <8 x half> %v1696, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1715 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1714, <4 x float> %v1713) #9, !dbg !44
  %v1716 = shufflevector <8 x half> %v1698, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1717 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1716, <4 x float> %v1715) #9, !dbg !44
  %v1718 = shufflevector <8 x half> %v1700, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1719 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1718, <4 x float> %v1717) #9, !dbg !44
  %v1720 = shufflevector <8 x half> %v1702, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %v1721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1720, <4 x float> %v1719) #9, !dbg !44
  %v1722 = shufflevector <8 x half> %v1688, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1723 = shufflevector <8 x float> %v1704, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %v1724 = fmul <4 x float> %v1723, %v1067, !dbg !64
  %v1725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1722, <4 x float> %v1724) #9, !dbg !44
  %v1726 = shufflevector <8 x half> %v1690, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %v1726, <4 x float> %v1725) #9, !dbg !44
  %v1728 = shufflevector <8 x half> %v1692, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %v1728, <4 x float> %v1727) #9, !dbg !44
  %v1730 = shufflevector <8 x half> %v1694, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %v1730, <4 x float> %v1729) #9, !dbg !44
  %v1732 = shufflevector <8 x half> %v1696, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1733 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %v1732, <4 x float> %v1731) #9, !dbg !44
  %v1734 = shufflevector <8 x half> %v1698, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1735 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %v1734, <4 x float> %v1733) #9, !dbg !44
  %v1736 = shufflevector <8 x half> %v1700, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %v1736, <4 x float> %v1735) #9, !dbg !44
  %v1738 = shufflevector <8 x half> %v1702, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %v1739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %v1738, <4 x float> %v1737) #9, !dbg !44
  %v1740 = extractvalue [2 x <8 x float>] %v211, 1, !dbg !64
  %v1741 = shufflevector <8 x float> %v1740, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %v1742 = fmul <4 x float> %v1741, %v1075, !dbg !64
  %v1743 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1703, <4 x float> %v1742) #9, !dbg !44
  %v1744 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1708, <4 x float> %v1743) #9, !dbg !44
  %v1745 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1710, <4 x float> %v1744) #9, !dbg !44
  %v1746 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1712, <4 x float> %v1745) #9, !dbg !44
  %v1747 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1714, <4 x float> %v1746) #9, !dbg !44
  %v1748 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1716, <4 x float> %v1747) #9, !dbg !44
  %v1749 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1718, <4 x float> %v1748) #9, !dbg !44
  %v1750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1720, <4 x float> %v1749) #9, !dbg !44
  %v1751 = shufflevector <8 x float> %v1740, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %v1752 = fmul <4 x float> %v1751, %v1075, !dbg !64
  %v1753 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %v1722, <4 x float> %v1752) #9, !dbg !44
  %v1754 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %v1726, <4 x float> %v1753) #9, !dbg !44
  %v1755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %v1728, <4 x float> %v1754) #9, !dbg !44
  %v1756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %v1730, <4 x float> %v1755) #9, !dbg !44
  %v1757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %v1732, <4 x float> %v1756) #9, !dbg !44
  %v1758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %v1734, <4 x float> %v1757) #9, !dbg !44
  %v1759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %v1736, <4 x float> %v1758) #9, !dbg !44
  %v1760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %v1738, <4 x float> %v1759) #9, !dbg !44
  %v1761 = shufflevector <4 x float> %v1439, <4 x float> %v1457, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %v1762 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1761, 0, !dbg !39
  %v1763 = shufflevector <4 x float> %v1468, <4 x float> %v1478, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %v1764 = insertvalue [2 x <8 x float>] %v1762, <8 x float> %v1763, 1, !dbg !39
  %v1765 = insertvalue [2 x <1 x float>] poison, <1 x float> %v1001, 0, !dbg !39
  %v1766 = insertvalue [2 x <1 x float>] %v1765, <1 x float> %v1044, 1, !dbg !39
  %v1767 = shufflevector <4 x float> %v1543, <4 x float> %v1561, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %v1768 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1767, 0, !dbg !39
  %v1769 = shufflevector <4 x float> %v1572, <4 x float> %v1582, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %v1770 = insertvalue [2 x <8 x float>] %v1768, <8 x float> %v1769, 1, !dbg !39
  %v1771 = shufflevector <4 x float> %v1647, <4 x float> %v1665, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %v1772 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1771, 0, !dbg !39
  %v1773 = shufflevector <4 x float> %v1676, <4 x float> %v1686, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %v1774 = insertvalue [2 x <8 x float>] %v1772, <8 x float> %v1773, 1, !dbg !39
  %v1775 = shufflevector <4 x float> %v1721, <4 x float> %v1739, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %v1776 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1775, 0, !dbg !39
  %v1777 = shufflevector <4 x float> %v1750, <4 x float> %v1760, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %v1778 = insertvalue [2 x <8 x float>] %v1776, <8 x float> %v1777, 1, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %v1779 = add nuw nsw i64 %v205, 1, !dbg !39
  br label %v204, !dbg !39

v1780:                                            ; preds = %v204
  %v1781 = and i32 %v31, 16, !dbg !39
  %.not = icmp eq i32 %v1781, 0, !dbg !39
  %v1782 = and i32 %v31, 32, !dbg !39
  %.not36 = icmp eq i32 %v1782, 0, !dbg !39
  %v1783 = extractvalue [2 x <8 x float>] %v211, 0, !dbg !39
  %v1784 = extractelement <8 x float> %v1783, i64 0, !dbg !39
  %v1785 = extractelement <8 x float> %v1783, i64 1, !dbg !39
  %v1786 = extractelement <8 x float> %v1783, i64 2, !dbg !39
  %v1787 = extractelement <8 x float> %v1783, i64 3, !dbg !39
  %v1788 = select i1 %.not, float %v1785, float %v1784, !dbg !39
  %v1789 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !39
  %v1790 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %v1789), !dbg !39
  %v1791 = add nuw nsw i32 %v1790, 64, !dbg !39
  %v1792 = xor i32 %v1790, 16, !dbg !39
  %v1793 = and i32 %v1791, 192, !dbg !39
  %v1794 = icmp samesign ult i32 %v1792, %v1793, !dbg !39
  %v1795 = select i1 %v1794, i32 %v1792, i32 %v1790, !dbg !39
  %v1796 = shl nuw nsw i32 %v1795, 2, !dbg !39
  %v1797 = bitcast float %v1788 to i32, !dbg !39
  %v1798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1797), !dbg !39
  %v1799 = bitcast i32 %v1798 to float, !dbg !39
  %v1800 = select i1 %.not, float %v1784, float %v1799, !dbg !39
  %v1801 = select i1 %.not, float %v1799, float %v1785, !dbg !39
  %v1802 = select i1 %.not, float %v1787, float %v1786, !dbg !39
  %v1803 = bitcast float %v1802 to i32, !dbg !39
  %v1804 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1803), !dbg !39
  %v1805 = bitcast i32 %v1804 to float, !dbg !39
  %v1806 = select i1 %.not, float %v1786, float %v1805, !dbg !39
  %v1807 = select i1 %.not, float %v1805, float %v1787, !dbg !39
  %v1808 = select i1 %.not36, float %v1806, float %v1800, !dbg !39
  %v1809 = xor i32 %v1790, 32, !dbg !39
  %v1810 = icmp samesign ult i32 %v1809, %v1793, !dbg !39
  %v1811 = select i1 %v1810, i32 %v1809, i32 %v1790, !dbg !39
  %v1812 = shl nuw nsw i32 %v1811, 2, !dbg !39
  %v1813 = bitcast float %v1808 to i32, !dbg !39
  %v1814 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1813), !dbg !39
  %v1815 = bitcast i32 %v1814 to float, !dbg !39
  %v1816 = select i1 %.not36, float %v1800, float %v1815, !dbg !39
  %v1817 = select i1 %.not36, float %v1815, float %v1806, !dbg !39
  %v1818 = select i1 %.not36, float %v1807, float %v1801, !dbg !39
  %v1819 = bitcast float %v1818 to i32, !dbg !39
  %v1820 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1819), !dbg !39
  %v1821 = bitcast i32 %v1820 to float, !dbg !39
  %v1822 = select i1 %.not36, float %v1801, float %v1821, !dbg !39
  %v1823 = select i1 %.not36, float %v1821, float %v1807, !dbg !39
  %v1824 = extractvalue [2 x <8 x float>] %v210, 0, !dbg !39
  %v1825 = extractelement <8 x float> %v1824, i64 0, !dbg !39
  %v1826 = extractelement <8 x float> %v1824, i64 1, !dbg !39
  %v1827 = extractelement <8 x float> %v1824, i64 2, !dbg !39
  %v1828 = extractelement <8 x float> %v1824, i64 3, !dbg !39
  %v1829 = select i1 %.not, float %v1826, float %v1825, !dbg !39
  %v1830 = bitcast float %v1829 to i32, !dbg !39
  %v1831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1830), !dbg !39
  %v1832 = bitcast i32 %v1831 to float, !dbg !39
  %v1833 = select i1 %.not, float %v1825, float %v1832, !dbg !39
  %v1834 = select i1 %.not, float %v1832, float %v1826, !dbg !39
  %v1835 = select i1 %.not, float %v1828, float %v1827, !dbg !39
  %v1836 = bitcast float %v1835 to i32, !dbg !39
  %v1837 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1836), !dbg !39
  %v1838 = bitcast i32 %v1837 to float, !dbg !39
  %v1839 = select i1 %.not, float %v1827, float %v1838, !dbg !39
  %v1840 = select i1 %.not, float %v1838, float %v1828, !dbg !39
  %v1841 = select i1 %.not36, float %v1839, float %v1833, !dbg !39
  %v1842 = bitcast float %v1841 to i32, !dbg !39
  %v1843 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1842), !dbg !39
  %v1844 = bitcast i32 %v1843 to float, !dbg !39
  %v1845 = select i1 %.not36, float %v1833, float %v1844, !dbg !39
  %v1846 = select i1 %.not36, float %v1844, float %v1839, !dbg !39
  %v1847 = select i1 %.not36, float %v1840, float %v1834, !dbg !39
  %v1848 = bitcast float %v1847 to i32, !dbg !39
  %v1849 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1848), !dbg !39
  %v1850 = bitcast i32 %v1849 to float, !dbg !39
  %v1851 = select i1 %.not36, float %v1834, float %v1850, !dbg !39
  %v1852 = select i1 %.not36, float %v1850, float %v1840, !dbg !39
  %v1853 = extractvalue [2 x <8 x float>] %v209, 0, !dbg !39
  %v1854 = extractelement <8 x float> %v1853, i64 0, !dbg !39
  %v1855 = extractelement <8 x float> %v1853, i64 1, !dbg !39
  %v1856 = extractelement <8 x float> %v1853, i64 2, !dbg !39
  %v1857 = extractelement <8 x float> %v1853, i64 3, !dbg !39
  %v1858 = select i1 %.not, float %v1855, float %v1854, !dbg !39
  %v1859 = bitcast float %v1858 to i32, !dbg !39
  %v1860 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1859), !dbg !39
  %v1861 = bitcast i32 %v1860 to float, !dbg !39
  %v1862 = select i1 %.not, float %v1854, float %v1861, !dbg !39
  %v1863 = select i1 %.not, float %v1861, float %v1855, !dbg !39
  %v1864 = select i1 %.not, float %v1857, float %v1856, !dbg !39
  %v1865 = bitcast float %v1864 to i32, !dbg !39
  %v1866 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1865), !dbg !39
  %v1867 = bitcast i32 %v1866 to float, !dbg !39
  %v1868 = select i1 %.not, float %v1856, float %v1867, !dbg !39
  %v1869 = select i1 %.not, float %v1867, float %v1857, !dbg !39
  %v1870 = select i1 %.not36, float %v1868, float %v1862, !dbg !39
  %v1871 = bitcast float %v1870 to i32, !dbg !39
  %v1872 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1871), !dbg !39
  %v1873 = bitcast i32 %v1872 to float, !dbg !39
  %v1874 = select i1 %.not36, float %v1862, float %v1873, !dbg !39
  %v1875 = select i1 %.not36, float %v1873, float %v1868, !dbg !39
  %v1876 = select i1 %.not36, float %v1869, float %v1863, !dbg !39
  %v1877 = bitcast float %v1876 to i32, !dbg !39
  %v1878 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1877), !dbg !39
  %v1879 = bitcast i32 %v1878 to float, !dbg !39
  %v1880 = select i1 %.not36, float %v1863, float %v1879, !dbg !39
  %v1881 = select i1 %.not36, float %v1879, float %v1869, !dbg !39
  %v1882 = extractvalue [2 x <8 x float>] %v206, 0, !dbg !39
  %v1883 = extractelement <8 x float> %v1882, i64 0, !dbg !39
  %v1884 = extractelement <8 x float> %v1882, i64 1, !dbg !39
  %v1885 = extractelement <8 x float> %v1882, i64 2, !dbg !39
  %v1886 = extractelement <8 x float> %v1882, i64 3, !dbg !39
  %v1887 = select i1 %.not, float %v1884, float %v1883, !dbg !39
  %v1888 = bitcast float %v1887 to i32, !dbg !39
  %v1889 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1888), !dbg !39
  %v1890 = bitcast i32 %v1889 to float, !dbg !39
  %v1891 = select i1 %.not, float %v1883, float %v1890, !dbg !39
  %v1892 = select i1 %.not, float %v1890, float %v1884, !dbg !39
  %v1893 = select i1 %.not, float %v1886, float %v1885, !dbg !39
  %v1894 = bitcast float %v1893 to i32, !dbg !39
  %v1895 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1894), !dbg !39
  %v1896 = bitcast i32 %v1895 to float, !dbg !39
  %v1897 = select i1 %.not, float %v1885, float %v1896, !dbg !39
  %v1898 = select i1 %.not, float %v1896, float %v1886, !dbg !39
  %v1899 = select i1 %.not36, float %v1897, float %v1891, !dbg !39
  %v1900 = bitcast float %v1899 to i32, !dbg !39
  %v1901 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1900), !dbg !39
  %v1902 = select i1 %.not36, float %v1898, float %v1892, !dbg !39
  %v1903 = bitcast float %v1902 to i32, !dbg !39
  %v1904 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1903), !dbg !39
  %v1905 = extractvalue [2 x <1 x float>] %v207, 0, !dbg !49
  %v1906 = shufflevector <1 x float> %v1905, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !49
  %v1907 = extractelement <8 x float> %v1783, i64 4, !dbg !39
  %v1908 = extractelement <8 x float> %v1783, i64 5, !dbg !39
  %v1909 = extractelement <8 x float> %v1783, i64 6, !dbg !39
  %v1910 = extractelement <8 x float> %v1783, i64 7, !dbg !39
  %v1911 = select i1 %.not, float %v1908, float %v1907, !dbg !39
  %v1912 = bitcast float %v1911 to i32, !dbg !39
  %v1913 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1912), !dbg !39
  %v1914 = bitcast i32 %v1913 to float, !dbg !39
  %v1915 = select i1 %.not, float %v1907, float %v1914, !dbg !39
  %v1916 = select i1 %.not, float %v1914, float %v1908, !dbg !39
  %v1917 = select i1 %.not, float %v1910, float %v1909, !dbg !39
  %v1918 = bitcast float %v1917 to i32, !dbg !39
  %v1919 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1918), !dbg !39
  %v1920 = bitcast i32 %v1919 to float, !dbg !39
  %v1921 = select i1 %.not, float %v1909, float %v1920, !dbg !39
  %v1922 = select i1 %.not, float %v1920, float %v1910, !dbg !39
  %v1923 = select i1 %.not36, float %v1921, float %v1915, !dbg !39
  %v1924 = bitcast float %v1923 to i32, !dbg !39
  %v1925 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1924), !dbg !39
  %v1926 = bitcast i32 %v1925 to float, !dbg !39
  %v1927 = select i1 %.not36, float %v1915, float %v1926, !dbg !39
  %v1928 = select i1 %.not36, float %v1926, float %v1921, !dbg !39
  %v1929 = select i1 %.not36, float %v1922, float %v1916, !dbg !39
  %v1930 = bitcast float %v1929 to i32, !dbg !39
  %v1931 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1930), !dbg !39
  %v1932 = bitcast i32 %v1931 to float, !dbg !39
  %v1933 = select i1 %.not36, float %v1916, float %v1932, !dbg !39
  %v1934 = select i1 %.not36, float %v1932, float %v1922, !dbg !39
  %v1935 = extractelement <8 x float> %v1824, i64 4, !dbg !39
  %v1936 = extractelement <8 x float> %v1824, i64 5, !dbg !39
  %v1937 = extractelement <8 x float> %v1824, i64 6, !dbg !39
  %v1938 = extractelement <8 x float> %v1824, i64 7, !dbg !39
  %v1939 = select i1 %.not, float %v1936, float %v1935, !dbg !39
  %v1940 = bitcast float %v1939 to i32, !dbg !39
  %v1941 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1940), !dbg !39
  %v1942 = bitcast i32 %v1941 to float, !dbg !39
  %v1943 = select i1 %.not, float %v1935, float %v1942, !dbg !39
  %v1944 = select i1 %.not, float %v1942, float %v1936, !dbg !39
  %v1945 = select i1 %.not, float %v1938, float %v1937, !dbg !39
  %v1946 = bitcast float %v1945 to i32, !dbg !39
  %v1947 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1946), !dbg !39
  %v1948 = bitcast i32 %v1947 to float, !dbg !39
  %v1949 = select i1 %.not, float %v1937, float %v1948, !dbg !39
  %v1950 = select i1 %.not, float %v1948, float %v1938, !dbg !39
  %v1951 = select i1 %.not36, float %v1949, float %v1943, !dbg !39
  %v1952 = bitcast float %v1951 to i32, !dbg !39
  %v1953 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1952), !dbg !39
  %v1954 = bitcast i32 %v1953 to float, !dbg !39
  %v1955 = select i1 %.not36, float %v1943, float %v1954, !dbg !39
  %v1956 = select i1 %.not36, float %v1954, float %v1949, !dbg !39
  %v1957 = select i1 %.not36, float %v1950, float %v1944, !dbg !39
  %v1958 = bitcast float %v1957 to i32, !dbg !39
  %v1959 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1958), !dbg !39
  %v1960 = bitcast i32 %v1959 to float, !dbg !39
  %v1961 = select i1 %.not36, float %v1944, float %v1960, !dbg !39
  %v1962 = select i1 %.not36, float %v1960, float %v1950, !dbg !39
  %v1963 = extractelement <8 x float> %v1853, i64 4, !dbg !39
  %v1964 = extractelement <8 x float> %v1853, i64 5, !dbg !39
  %v1965 = extractelement <8 x float> %v1853, i64 6, !dbg !39
  %v1966 = extractelement <8 x float> %v1853, i64 7, !dbg !39
  %v1967 = select i1 %.not, float %v1964, float %v1963, !dbg !39
  %v1968 = bitcast float %v1967 to i32, !dbg !39
  %v1969 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1968), !dbg !39
  %v1970 = bitcast i32 %v1969 to float, !dbg !39
  %v1971 = select i1 %.not, float %v1963, float %v1970, !dbg !39
  %v1972 = select i1 %.not, float %v1970, float %v1964, !dbg !39
  %v1973 = select i1 %.not, float %v1966, float %v1965, !dbg !39
  %v1974 = bitcast float %v1973 to i32, !dbg !39
  %v1975 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1974), !dbg !39
  %v1976 = bitcast i32 %v1975 to float, !dbg !39
  %v1977 = select i1 %.not, float %v1965, float %v1976, !dbg !39
  %v1978 = select i1 %.not, float %v1976, float %v1966, !dbg !39
  %v1979 = select i1 %.not36, float %v1977, float %v1971, !dbg !39
  %v1980 = bitcast float %v1979 to i32, !dbg !39
  %v1981 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1980), !dbg !39
  %v1982 = bitcast i32 %v1981 to float, !dbg !39
  %v1983 = select i1 %.not36, float %v1971, float %v1982, !dbg !39
  %v1984 = select i1 %.not36, float %v1982, float %v1977, !dbg !39
  %v1985 = select i1 %.not36, float %v1978, float %v1972, !dbg !39
  %v1986 = bitcast float %v1985 to i32, !dbg !39
  %v1987 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v1986), !dbg !39
  %v1988 = bitcast i32 %v1987 to float, !dbg !39
  %v1989 = select i1 %.not36, float %v1972, float %v1988, !dbg !39
  %v1990 = select i1 %.not36, float %v1988, float %v1978, !dbg !39
  %v1991 = extractelement <8 x float> %v1882, i64 4, !dbg !39
  %v1992 = extractelement <8 x float> %v1882, i64 5, !dbg !39
  %v1993 = extractelement <8 x float> %v1882, i64 6, !dbg !39
  %v1994 = extractelement <8 x float> %v1882, i64 7, !dbg !39
  %v1995 = select i1 %.not, float %v1992, float %v1991, !dbg !39
  %v1996 = bitcast float %v1995 to i32, !dbg !39
  %v1997 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v1996), !dbg !39
  %v1998 = bitcast i32 %v1997 to float, !dbg !39
  %v1999 = select i1 %.not, float %v1991, float %v1998, !dbg !39
  %v2000 = select i1 %.not, float %v1998, float %v1992, !dbg !39
  %v2001 = select i1 %.not, float %v1994, float %v1993, !dbg !39
  %v2002 = bitcast float %v2001 to i32, !dbg !39
  %v2003 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2002), !dbg !39
  %v2004 = bitcast i32 %v2003 to float, !dbg !39
  %v2005 = select i1 %.not, float %v1993, float %v2004, !dbg !39
  %v2006 = select i1 %.not, float %v2004, float %v1994, !dbg !39
  %v2007 = select i1 %.not36, float %v2005, float %v1999, !dbg !39
  %v2008 = bitcast float %v2007 to i32, !dbg !39
  %v2009 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2008), !dbg !39
  %v2010 = select i1 %.not36, float %v2006, float %v2000, !dbg !39
  %v2011 = bitcast float %v2010 to i32, !dbg !39
  %v2012 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2011), !dbg !39
  %v2013 = extractvalue [2 x <8 x float>] %v211, 1, !dbg !39
  %v2014 = extractelement <8 x float> %v2013, i64 0, !dbg !39
  %v2015 = extractelement <8 x float> %v2013, i64 1, !dbg !39
  %v2016 = extractelement <8 x float> %v2013, i64 2, !dbg !39
  %v2017 = extractelement <8 x float> %v2013, i64 3, !dbg !39
  %v2018 = select i1 %.not, float %v2015, float %v2014, !dbg !39
  %v2019 = bitcast float %v2018 to i32, !dbg !39
  %v2020 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2019), !dbg !39
  %v2021 = bitcast i32 %v2020 to float, !dbg !39
  %v2022 = select i1 %.not, float %v2014, float %v2021, !dbg !39
  %v2023 = select i1 %.not, float %v2021, float %v2015, !dbg !39
  %v2024 = select i1 %.not, float %v2017, float %v2016, !dbg !39
  %v2025 = bitcast float %v2024 to i32, !dbg !39
  %v2026 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2025), !dbg !39
  %v2027 = bitcast i32 %v2026 to float, !dbg !39
  %v2028 = select i1 %.not, float %v2016, float %v2027, !dbg !39
  %v2029 = select i1 %.not, float %v2027, float %v2017, !dbg !39
  %v2030 = select i1 %.not36, float %v2028, float %v2022, !dbg !39
  %v2031 = bitcast float %v2030 to i32, !dbg !39
  %v2032 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2031), !dbg !39
  %v2033 = bitcast i32 %v2032 to float, !dbg !39
  %v2034 = select i1 %.not36, float %v2022, float %v2033, !dbg !39
  %v2035 = select i1 %.not36, float %v2033, float %v2028, !dbg !39
  %v2036 = select i1 %.not36, float %v2029, float %v2023, !dbg !39
  %v2037 = bitcast float %v2036 to i32, !dbg !39
  %v2038 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2037), !dbg !39
  %v2039 = bitcast i32 %v2038 to float, !dbg !39
  %v2040 = select i1 %.not36, float %v2023, float %v2039, !dbg !39
  %v2041 = select i1 %.not36, float %v2039, float %v2029, !dbg !39
  %v2042 = extractvalue [2 x <8 x float>] %v210, 1, !dbg !39
  %v2043 = extractelement <8 x float> %v2042, i64 0, !dbg !39
  %v2044 = extractelement <8 x float> %v2042, i64 1, !dbg !39
  %v2045 = extractelement <8 x float> %v2042, i64 2, !dbg !39
  %v2046 = extractelement <8 x float> %v2042, i64 3, !dbg !39
  %v2047 = select i1 %.not, float %v2044, float %v2043, !dbg !39
  %v2048 = bitcast float %v2047 to i32, !dbg !39
  %v2049 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2048), !dbg !39
  %v2050 = bitcast i32 %v2049 to float, !dbg !39
  %v2051 = select i1 %.not, float %v2043, float %v2050, !dbg !39
  %v2052 = select i1 %.not, float %v2050, float %v2044, !dbg !39
  %v2053 = select i1 %.not, float %v2046, float %v2045, !dbg !39
  %v2054 = bitcast float %v2053 to i32, !dbg !39
  %v2055 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2054), !dbg !39
  %v2056 = bitcast i32 %v2055 to float, !dbg !39
  %v2057 = select i1 %.not, float %v2045, float %v2056, !dbg !39
  %v2058 = select i1 %.not, float %v2056, float %v2046, !dbg !39
  %v2059 = select i1 %.not36, float %v2057, float %v2051, !dbg !39
  %v2060 = bitcast float %v2059 to i32, !dbg !39
  %v2061 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2060), !dbg !39
  %v2062 = bitcast i32 %v2061 to float, !dbg !39
  %v2063 = select i1 %.not36, float %v2051, float %v2062, !dbg !39
  %v2064 = select i1 %.not36, float %v2062, float %v2057, !dbg !39
  %v2065 = select i1 %.not36, float %v2058, float %v2052, !dbg !39
  %v2066 = bitcast float %v2065 to i32, !dbg !39
  %v2067 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2066), !dbg !39
  %v2068 = bitcast i32 %v2067 to float, !dbg !39
  %v2069 = select i1 %.not36, float %v2052, float %v2068, !dbg !39
  %v2070 = select i1 %.not36, float %v2068, float %v2058, !dbg !39
  %v2071 = extractvalue [2 x <8 x float>] %v209, 1, !dbg !39
  %v2072 = extractelement <8 x float> %v2071, i64 0, !dbg !39
  %v2073 = extractelement <8 x float> %v2071, i64 1, !dbg !39
  %v2074 = extractelement <8 x float> %v2071, i64 2, !dbg !39
  %v2075 = extractelement <8 x float> %v2071, i64 3, !dbg !39
  %v2076 = select i1 %.not, float %v2073, float %v2072, !dbg !39
  %v2077 = bitcast float %v2076 to i32, !dbg !39
  %v2078 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2077), !dbg !39
  %v2079 = bitcast i32 %v2078 to float, !dbg !39
  %v2080 = select i1 %.not, float %v2072, float %v2079, !dbg !39
  %v2081 = select i1 %.not, float %v2079, float %v2073, !dbg !39
  %v2082 = select i1 %.not, float %v2075, float %v2074, !dbg !39
  %v2083 = bitcast float %v2082 to i32, !dbg !39
  %v2084 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2083), !dbg !39
  %v2085 = bitcast i32 %v2084 to float, !dbg !39
  %v2086 = select i1 %.not, float %v2074, float %v2085, !dbg !39
  %v2087 = select i1 %.not, float %v2085, float %v2075, !dbg !39
  %v2088 = select i1 %.not36, float %v2086, float %v2080, !dbg !39
  %v2089 = bitcast float %v2088 to i32, !dbg !39
  %v2090 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2089), !dbg !39
  %v2091 = bitcast i32 %v2090 to float, !dbg !39
  %v2092 = select i1 %.not36, float %v2080, float %v2091, !dbg !39
  %v2093 = select i1 %.not36, float %v2091, float %v2086, !dbg !39
  %v2094 = select i1 %.not36, float %v2087, float %v2081, !dbg !39
  %v2095 = bitcast float %v2094 to i32, !dbg !39
  %v2096 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2095), !dbg !39
  %v2097 = bitcast i32 %v2096 to float, !dbg !39
  %v2098 = select i1 %.not36, float %v2081, float %v2097, !dbg !39
  %v2099 = select i1 %.not36, float %v2097, float %v2087, !dbg !39
  %v2100 = extractvalue [2 x <8 x float>] %v206, 1, !dbg !39
  %v2101 = extractelement <8 x float> %v2100, i64 0, !dbg !39
  %v2102 = extractelement <8 x float> %v2100, i64 1, !dbg !39
  %v2103 = extractelement <8 x float> %v2100, i64 2, !dbg !39
  %v2104 = extractelement <8 x float> %v2100, i64 3, !dbg !39
  %v2105 = select i1 %.not, float %v2102, float %v2101, !dbg !39
  %v2106 = bitcast float %v2105 to i32, !dbg !39
  %v2107 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2106), !dbg !39
  %v2108 = bitcast i32 %v2107 to float, !dbg !39
  %v2109 = select i1 %.not, float %v2101, float %v2108, !dbg !39
  %v2110 = select i1 %.not, float %v2108, float %v2102, !dbg !39
  %v2111 = select i1 %.not, float %v2104, float %v2103, !dbg !39
  %v2112 = bitcast float %v2111 to i32, !dbg !39
  %v2113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2112), !dbg !39
  %v2114 = bitcast i32 %v2113 to float, !dbg !39
  %v2115 = select i1 %.not, float %v2103, float %v2114, !dbg !39
  %v2116 = select i1 %.not, float %v2114, float %v2104, !dbg !39
  %v2117 = select i1 %.not36, float %v2115, float %v2109, !dbg !39
  %v2118 = bitcast float %v2117 to i32, !dbg !39
  %v2119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2118), !dbg !39
  %v2120 = select i1 %.not36, float %v2116, float %v2110, !dbg !39
  %v2121 = bitcast float %v2120 to i32, !dbg !39
  %v2122 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2121), !dbg !39
  %v2123 = extractvalue [2 x <1 x float>] %v207, 1, !dbg !49
  %v2124 = shufflevector <1 x float> %v2123, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !49
  %v2125 = extractelement <8 x float> %v2013, i64 4, !dbg !39
  %v2126 = extractelement <8 x float> %v2013, i64 5, !dbg !39
  %v2127 = extractelement <8 x float> %v2013, i64 6, !dbg !39
  %v2128 = extractelement <8 x float> %v2013, i64 7, !dbg !39
  %v2129 = select i1 %.not, float %v2126, float %v2125, !dbg !39
  %v2130 = bitcast float %v2129 to i32, !dbg !39
  %v2131 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2130), !dbg !39
  %v2132 = bitcast i32 %v2131 to float, !dbg !39
  %v2133 = select i1 %.not, float %v2125, float %v2132, !dbg !39
  %v2134 = select i1 %.not, float %v2132, float %v2126, !dbg !39
  %v2135 = select i1 %.not, float %v2128, float %v2127, !dbg !39
  %v2136 = bitcast float %v2135 to i32, !dbg !39
  %v2137 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2136), !dbg !39
  %v2138 = bitcast i32 %v2137 to float, !dbg !39
  %v2139 = select i1 %.not, float %v2127, float %v2138, !dbg !39
  %v2140 = select i1 %.not, float %v2138, float %v2128, !dbg !39
  %v2141 = select i1 %.not36, float %v2139, float %v2133, !dbg !39
  %v2142 = bitcast float %v2141 to i32, !dbg !39
  %v2143 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2142), !dbg !39
  %v2144 = bitcast i32 %v2143 to float, !dbg !39
  %v2145 = select i1 %.not36, float %v2133, float %v2144, !dbg !39
  %v2146 = select i1 %.not36, float %v2144, float %v2139, !dbg !39
  %v2147 = select i1 %.not36, float %v2140, float %v2134, !dbg !39
  %v2148 = bitcast float %v2147 to i32, !dbg !39
  %v2149 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2148), !dbg !39
  %v2150 = bitcast i32 %v2149 to float, !dbg !39
  %v2151 = select i1 %.not36, float %v2134, float %v2150, !dbg !39
  %v2152 = select i1 %.not36, float %v2150, float %v2140, !dbg !39
  %v2153 = extractelement <8 x float> %v2042, i64 4, !dbg !39
  %v2154 = extractelement <8 x float> %v2042, i64 5, !dbg !39
  %v2155 = extractelement <8 x float> %v2042, i64 6, !dbg !39
  %v2156 = extractelement <8 x float> %v2042, i64 7, !dbg !39
  %v2157 = select i1 %.not, float %v2154, float %v2153, !dbg !39
  %v2158 = bitcast float %v2157 to i32, !dbg !39
  %v2159 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2158), !dbg !39
  %v2160 = bitcast i32 %v2159 to float, !dbg !39
  %v2161 = select i1 %.not, float %v2153, float %v2160, !dbg !39
  %v2162 = select i1 %.not, float %v2160, float %v2154, !dbg !39
  %v2163 = select i1 %.not, float %v2156, float %v2155, !dbg !39
  %v2164 = bitcast float %v2163 to i32, !dbg !39
  %v2165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2164), !dbg !39
  %v2166 = bitcast i32 %v2165 to float, !dbg !39
  %v2167 = select i1 %.not, float %v2155, float %v2166, !dbg !39
  %v2168 = select i1 %.not, float %v2166, float %v2156, !dbg !39
  %v2169 = select i1 %.not36, float %v2167, float %v2161, !dbg !39
  %v2170 = bitcast float %v2169 to i32, !dbg !39
  %v2171 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2170), !dbg !39
  %v2172 = bitcast i32 %v2171 to float, !dbg !39
  %v2173 = select i1 %.not36, float %v2161, float %v2172, !dbg !39
  %v2174 = select i1 %.not36, float %v2172, float %v2167, !dbg !39
  %v2175 = select i1 %.not36, float %v2168, float %v2162, !dbg !39
  %v2176 = bitcast float %v2175 to i32, !dbg !39
  %v2177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2176), !dbg !39
  %v2178 = bitcast i32 %v2177 to float, !dbg !39
  %v2179 = select i1 %.not36, float %v2162, float %v2178, !dbg !39
  %v2180 = select i1 %.not36, float %v2178, float %v2168, !dbg !39
  %v2181 = extractelement <8 x float> %v2071, i64 4, !dbg !39
  %v2182 = extractelement <8 x float> %v2071, i64 5, !dbg !39
  %v2183 = extractelement <8 x float> %v2071, i64 6, !dbg !39
  %v2184 = extractelement <8 x float> %v2071, i64 7, !dbg !39
  %v2185 = select i1 %.not, float %v2182, float %v2181, !dbg !39
  %v2186 = bitcast float %v2185 to i32, !dbg !39
  %v2187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2186), !dbg !39
  %v2188 = bitcast i32 %v2187 to float, !dbg !39
  %v2189 = select i1 %.not, float %v2181, float %v2188, !dbg !39
  %v2190 = select i1 %.not, float %v2188, float %v2182, !dbg !39
  %v2191 = select i1 %.not, float %v2184, float %v2183, !dbg !39
  %v2192 = bitcast float %v2191 to i32, !dbg !39
  %v2193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2192), !dbg !39
  %v2194 = bitcast i32 %v2193 to float, !dbg !39
  %v2195 = select i1 %.not, float %v2183, float %v2194, !dbg !39
  %v2196 = select i1 %.not, float %v2194, float %v2184, !dbg !39
  %v2197 = select i1 %.not36, float %v2195, float %v2189, !dbg !39
  %v2198 = bitcast float %v2197 to i32, !dbg !39
  %v2199 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2198), !dbg !39
  %v2200 = bitcast i32 %v2199 to float, !dbg !39
  %v2201 = select i1 %.not36, float %v2189, float %v2200, !dbg !39
  %v2202 = select i1 %.not36, float %v2200, float %v2195, !dbg !39
  %v2203 = select i1 %.not36, float %v2196, float %v2190, !dbg !39
  %v2204 = bitcast float %v2203 to i32, !dbg !39
  %v2205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2204), !dbg !39
  %v2206 = bitcast i32 %v2205 to float, !dbg !39
  %v2207 = select i1 %.not36, float %v2190, float %v2206, !dbg !39
  %v2208 = select i1 %.not36, float %v2206, float %v2196, !dbg !39
  %v2209 = extractelement <8 x float> %v2100, i64 4, !dbg !39
  %v2210 = extractelement <8 x float> %v2100, i64 5, !dbg !39
  %v2211 = extractelement <8 x float> %v2100, i64 6, !dbg !39
  %v2212 = extractelement <8 x float> %v2100, i64 7, !dbg !39
  %v2213 = select i1 %.not, float %v2210, float %v2209, !dbg !39
  %v2214 = bitcast float %v2213 to i32, !dbg !39
  %v2215 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2214), !dbg !39
  %v2216 = bitcast i32 %v2215 to float, !dbg !39
  %v2217 = select i1 %.not, float %v2209, float %v2216, !dbg !39
  %v2218 = select i1 %.not, float %v2216, float %v2210, !dbg !39
  %v2219 = select i1 %.not, float %v2212, float %v2211, !dbg !39
  %v2220 = bitcast float %v2219 to i32, !dbg !39
  %v2221 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1796, i32 %v2220), !dbg !39
  %v2222 = bitcast i32 %v2221 to float, !dbg !39
  %v2223 = select i1 %.not, float %v2211, float %v2222, !dbg !39
  %v2224 = select i1 %.not, float %v2222, float %v2212, !dbg !39
  %v2225 = select i1 %.not36, float %v2223, float %v2217, !dbg !39
  %v2226 = bitcast float %v2225 to i32, !dbg !39
  %v2227 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2226), !dbg !39
  %v2228 = select i1 %.not36, float %v2224, float %v2218, !dbg !39
  %v2229 = bitcast float %v2228 to i32, !dbg !39
  %v2230 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1812, i32 %v2229), !dbg !39
  br i1 true, label %v2231, label %v2295, !dbg !51

v2231:                                            ; preds = %v1780
  %v2232 = bitcast i32 %v2227 to float, !dbg !39
  %v2233 = select i1 %.not36, float %v2217, float %v2232, !dbg !39
  %v2234 = insertelement <4 x float> poison, float %v2233, i64 0, !dbg !49
  %v2235 = bitcast i32 %v2230 to float, !dbg !39
  %v2236 = select i1 %.not36, float %v2218, float %v2235, !dbg !39
  %v2237 = insertelement <4 x float> %v2234, float %v2236, i64 1, !dbg !49
  %v2238 = select i1 %.not36, float %v2232, float %v2223, !dbg !39
  %v2239 = insertelement <4 x float> %v2237, float %v2238, i64 2, !dbg !49
  %v2240 = select i1 %.not36, float %v2235, float %v2224, !dbg !39
  %v2241 = insertelement <4 x float> %v2239, float %v2240, i64 3, !dbg !49
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %v2124, !dbg !49
  %normalized = fmul <4 x float> %v2241, %row.reciprocal, !dbg !49
  %v2242 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !50
  %v2243 = extractelement <4 x half> %v2242, i64 3, !dbg !50
  %v2244 = extractelement <4 x half> %v2242, i64 2, !dbg !50
  %v2245 = extractelement <4 x half> %v2242, i64 1, !dbg !50
  %v2246 = extractelement <4 x half> %v2242, i64 0, !dbg !50
  %v2247 = bitcast i32 %v2119 to float, !dbg !39
  %v2248 = select i1 %.not36, float %v2109, float %v2247, !dbg !39
  %v2249 = insertelement <4 x float> poison, float %v2248, i64 0, !dbg !49
  %v2250 = bitcast i32 %v2122 to float, !dbg !39
  %v2251 = select i1 %.not36, float %v2110, float %v2250, !dbg !39
  %v2252 = insertelement <4 x float> %v2249, float %v2251, i64 1, !dbg !49
  %v2253 = select i1 %.not36, float %v2247, float %v2115, !dbg !39
  %v2254 = insertelement <4 x float> %v2252, float %v2253, i64 2, !dbg !49
  %v2255 = select i1 %.not36, float %v2250, float %v2116, !dbg !39
  %v2256 = insertelement <4 x float> %v2254, float %v2255, i64 3, !dbg !49
  %normalized104 = fmul <4 x float> %v2256, %row.reciprocal, !dbg !49
  %v2257 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !50
  %v2258 = extractelement <4 x half> %v2257, i64 3, !dbg !50
  %v2259 = extractelement <4 x half> %v2257, i64 2, !dbg !50
  %v2260 = extractelement <4 x half> %v2257, i64 1, !dbg !50
  %v2261 = extractelement <4 x half> %v2257, i64 0, !dbg !50
  %v2262 = bitcast i32 %v2009 to float, !dbg !39
  %v2263 = select i1 %.not36, float %v1999, float %v2262, !dbg !39
  %v2264 = insertelement <4 x float> poison, float %v2263, i64 0, !dbg !49
  %v2265 = bitcast i32 %v2012 to float, !dbg !39
  %v2266 = select i1 %.not36, float %v2000, float %v2265, !dbg !39
  %v2267 = insertelement <4 x float> %v2264, float %v2266, i64 1, !dbg !49
  %v2268 = select i1 %.not36, float %v2262, float %v2005, !dbg !39
  %v2269 = insertelement <4 x float> %v2267, float %v2268, i64 2, !dbg !49
  %v2270 = select i1 %.not36, float %v2265, float %v2006, !dbg !39
  %v2271 = insertelement <4 x float> %v2269, float %v2270, i64 3, !dbg !49
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %v1906, !dbg !49
  %normalized106 = fmul <4 x float> %v2271, %row.reciprocal105, !dbg !49
  %v2272 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !50
  %v2273 = extractelement <4 x half> %v2272, i64 3, !dbg !50
  %v2274 = extractelement <4 x half> %v2272, i64 2, !dbg !50
  %v2275 = extractelement <4 x half> %v2272, i64 1, !dbg !50
  %v2276 = extractelement <4 x half> %v2272, i64 0, !dbg !50
  %v2277 = bitcast i32 %v1901 to float, !dbg !39
  %v2278 = select i1 %.not36, float %v1891, float %v2277, !dbg !39
  %v2279 = insertelement <4 x float> poison, float %v2278, i64 0, !dbg !49
  %v2280 = bitcast i32 %v1904 to float, !dbg !39
  %v2281 = select i1 %.not36, float %v1892, float %v2280, !dbg !39
  %v2282 = insertelement <4 x float> %v2279, float %v2281, i64 1, !dbg !49
  %v2283 = select i1 %.not36, float %v2277, float %v1897, !dbg !39
  %v2284 = insertelement <4 x float> %v2282, float %v2283, i64 2, !dbg !49
  %v2285 = select i1 %.not36, float %v2280, float %v1898, !dbg !39
  %v2286 = insertelement <4 x float> %v2284, float %v2285, i64 3, !dbg !49
  %normalized107 = fmul <4 x float> %v2286, %row.reciprocal105, !dbg !49
  %v2287 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !50
  %v2288 = extractelement <4 x half> %v2287, i64 3, !dbg !50
  %v2289 = extractelement <4 x half> %v2287, i64 2, !dbg !50
  %v2290 = extractelement <4 x half> %v2287, i64 1, !dbg !50
  %v2291 = extractelement <4 x half> %v2287, i64 0, !dbg !50
  %v2292 = add nsw i64 %v16, %v22, !dbg !51
  %v2293 = shl i64 %v2292, 19, !dbg !51
  %v2294 = and i64 %v2293, 16252928, !dbg !51
  %address.thread934 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !51, !range !67, !frisk.register_lifetimes !34
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !40
  %address.local936 = add i64 %v25, %address.local935, !dbg !51
  %address.local937 = lshr i64 %address.local935, 6, !dbg !42
  %address.local938 = shl nuw nsw i64 %address.local937, 4, !dbg !51
  %address.local939 = add i64 %address.local936, %address.local938, !dbg !51
  %address.local940 = and i64 %address.local935, 240, !dbg !51
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !51
  %address.local942 = add i64 %address.local941, %v26, !dbg !51
  %address.local943 = shl nuw nsw i64 %address.local942, 7, !dbg !51
  %address.local944 = add nuw nsw i64 %v2294, %address.local943, !dbg !51
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local946 = lshr i64 %address.local935, 2, !dbg !42
  %address.local947 = and i64 %address.local946, 12, !dbg !42
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !51
  store half %v2291, ptr addrspace(1) %address.local948, align 2, !dbg !51
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !51
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local949, i64 %address.local950, !dbg !51
  store half %v2290, ptr addrspace(1) %address.local951, align 2, !dbg !51
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !51
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local952, i64 %address.local953, !dbg !51
  store half %v2289, ptr addrspace(1) %address.local954, align 2, !dbg !51
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !51
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local955, i64 %address.local956, !dbg !51
  store half %v2288, ptr addrspace(1) %address.local957, align 2, !dbg !51
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !51
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local958, i64 %address.local959, !dbg !51
  store half %v2276, ptr addrspace(1) %address.local960, align 2, !dbg !51
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !51
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local961, i64 %address.local962, !dbg !51
  store half %v2275, ptr addrspace(1) %address.local963, align 2, !dbg !51
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !51
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local964, i64 %address.local965, !dbg !51
  store half %v2274, ptr addrspace(1) %address.local966, align 2, !dbg !51
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !51
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local967, i64 %address.local968, !dbg !51
  store half %v2273, ptr addrspace(1) %address.local969, align 2, !dbg !51
  %address.local970 = shl i64 %address.local942, 7, !dbg !51
  %address.local971 = add i64 %address.local970, 8192, !dbg !51
  %address.local972 = add nuw nsw i64 %v2294, %address.local971, !dbg !51
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !51
  store half %v2261, ptr addrspace(1) %address.local974, align 2, !dbg !51
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local975, i64 %address.local950, !dbg !51
  store half %v2260, ptr addrspace(1) %address.local976, align 2, !dbg !51
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local953, !dbg !51
  store half %v2259, ptr addrspace(1) %address.local978, align 2, !dbg !51
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local979, i64 %address.local956, !dbg !51
  store half %v2258, ptr addrspace(1) %address.local980, align 2, !dbg !51
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local981, i64 %address.local959, !dbg !51
  store half %v2246, ptr addrspace(1) %address.local982, align 2, !dbg !51
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local962, !dbg !51
  store half %v2245, ptr addrspace(1) %address.local984, align 2, !dbg !51
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local965, !dbg !51
  store half %v2244, ptr addrspace(1) %address.local986, align 2, !dbg !51
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local987, i64 %address.local968, !dbg !51
  store half %v2243, ptr addrspace(1) %address.local988, align 2, !dbg !51
  br label %v2295, !dbg !51

v2295:                                            ; preds = %v2231, %v1780
  br i1 true, label %v2296, label %v2334, !dbg !51

v2296:                                            ; preds = %v2295
  %v2297 = insertelement <4 x float> poison, float %v2201, i64 0, !dbg !49
  %v2298 = insertelement <4 x float> %v2297, float %v2207, i64 1, !dbg !49
  %v2299 = insertelement <4 x float> %v2298, float %v2202, i64 2, !dbg !49
  %v2300 = insertelement <4 x float> %v2299, float %v2208, i64 3, !dbg !49
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 1.000000e+00), %v2124, !dbg !49
  %normalized109 = fmul <4 x float> %v2300, %row.reciprocal108, !dbg !49
  %v2301 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !50
  %v2302 = extractelement <4 x half> %v2301, i64 3, !dbg !50
  %v2303 = extractelement <4 x half> %v2301, i64 2, !dbg !50
  %v2304 = extractelement <4 x half> %v2301, i64 1, !dbg !50
  %v2305 = extractelement <4 x half> %v2301, i64 0, !dbg !50
  %v2306 = insertelement <4 x float> poison, float %v2092, i64 0, !dbg !49
  %v2307 = insertelement <4 x float> %v2306, float %v2098, i64 1, !dbg !49
  %v2308 = insertelement <4 x float> %v2307, float %v2093, i64 2, !dbg !49
  %v2309 = insertelement <4 x float> %v2308, float %v2099, i64 3, !dbg !49
  %normalized110 = fmul <4 x float> %v2309, %row.reciprocal108, !dbg !49
  %v2310 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !50
  %v2311 = extractelement <4 x half> %v2310, i64 3, !dbg !50
  %v2312 = extractelement <4 x half> %v2310, i64 2, !dbg !50
  %v2313 = extractelement <4 x half> %v2310, i64 1, !dbg !50
  %v2314 = extractelement <4 x half> %v2310, i64 0, !dbg !50
  %v2315 = insertelement <4 x float> poison, float %v1983, i64 0, !dbg !49
  %v2316 = insertelement <4 x float> %v2315, float %v1989, i64 1, !dbg !49
  %v2317 = insertelement <4 x float> %v2316, float %v1984, i64 2, !dbg !49
  %v2318 = insertelement <4 x float> %v2317, float %v1990, i64 3, !dbg !49
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 1.000000e+00), %v1906, !dbg !49
  %normalized112 = fmul <4 x float> %v2318, %row.reciprocal111, !dbg !49
  %v2319 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !50
  %v2320 = extractelement <4 x half> %v2319, i64 3, !dbg !50
  %v2321 = extractelement <4 x half> %v2319, i64 2, !dbg !50
  %v2322 = extractelement <4 x half> %v2319, i64 1, !dbg !50
  %v2323 = extractelement <4 x half> %v2319, i64 0, !dbg !50
  %v2324 = insertelement <4 x float> poison, float %v1874, i64 0, !dbg !49
  %v2325 = insertelement <4 x float> %v2324, float %v1880, i64 1, !dbg !49
  %v2326 = insertelement <4 x float> %v2325, float %v1875, i64 2, !dbg !49
  %v2327 = insertelement <4 x float> %v2326, float %v1881, i64 3, !dbg !49
  %normalized113 = fmul <4 x float> %v2327, %row.reciprocal111, !dbg !49
  %v2328 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !50
  %v2329 = extractelement <4 x half> %v2328, i64 3, !dbg !50
  %v2330 = extractelement <4 x half> %v2328, i64 2, !dbg !50
  %v2331 = extractelement <4 x half> %v2328, i64 1, !dbg !50
  %v2332 = extractelement <4 x half> %v2328, i64 0, !dbg !50
  %v2333 = shl nuw nsw i64 %v24, 19, !dbg !51
  %address.thread989 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !51, !range !67, !frisk.register_lifetimes !34
  %address.local990 = zext nneg i32 %address.thread989 to i64, !dbg !40
  %address.local991 = add i64 %v25, %address.local990, !dbg !51
  %address.local992 = lshr i64 %address.local990, 6, !dbg !42
  %address.local993 = shl nuw nsw i64 %address.local992, 4, !dbg !51
  %address.local994 = add i64 %address.local991, %address.local993, !dbg !51
  %address.local995 = and i64 %address.local990, 240, !dbg !51
  %address.local996 = sub i64 %address.local994, %address.local995, !dbg !51
  %address.local997 = add i64 %address.local996, %v26, !dbg !51
  %address.local998 = shl nuw nsw i64 %address.local997, 7, !dbg !51
  %address.local999 = add nuw nsw i64 %v2333, %address.local998, !dbg !51
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1001 = lshr i64 %address.local990, 2, !dbg !42
  %address.local1002 = and i64 %address.local1001, 12, !dbg !42
  %address.local1003 = or disjoint i64 %address.local1002, 32, !dbg !51
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local1003, !dbg !51
  store half %v2332, ptr addrspace(1) %address.local1004, align 2, !dbg !51
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1006 = or disjoint i64 %address.local1002, 33, !dbg !51
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1005, i64 %address.local1006, !dbg !51
  store half %v2331, ptr addrspace(1) %address.local1007, align 2, !dbg !51
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1009 = or disjoint i64 %address.local1002, 34, !dbg !51
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1008, i64 %address.local1009, !dbg !51
  store half %v2330, ptr addrspace(1) %address.local1010, align 2, !dbg !51
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1012 = or disjoint i64 %address.local1002, 35, !dbg !51
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local1012, !dbg !51
  store half %v2329, ptr addrspace(1) %address.local1013, align 2, !dbg !51
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1015 = or disjoint i64 %address.local1002, 48, !dbg !51
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local1015, !dbg !51
  store half %v2323, ptr addrspace(1) %address.local1016, align 2, !dbg !51
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1018 = or disjoint i64 %address.local1002, 49, !dbg !51
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local1018, !dbg !51
  store half %v2322, ptr addrspace(1) %address.local1019, align 2, !dbg !51
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1021 = or disjoint i64 %address.local1002, 50, !dbg !51
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1020, i64 %address.local1021, !dbg !51
  store half %v2321, ptr addrspace(1) %address.local1022, align 2, !dbg !51
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1024 = or i64 %address.local1001, 51, !dbg !51
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1024, !dbg !51
  store half %v2320, ptr addrspace(1) %address.local1025, align 2, !dbg !51
  %address.local1026 = shl i64 %address.local997, 7, !dbg !51
  %address.local1027 = add i64 %address.local1026, 8192, !dbg !51
  %address.local1028 = add nuw nsw i64 %v2333, %address.local1027, !dbg !51
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1029, i64 %address.local1003, !dbg !51
  store half %v2314, ptr addrspace(1) %address.local1030, align 2, !dbg !51
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1006, !dbg !51
  store half %v2313, ptr addrspace(1) %address.local1032, align 2, !dbg !51
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1009, !dbg !51
  store half %v2312, ptr addrspace(1) %address.local1034, align 2, !dbg !51
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1012, !dbg !51
  store half %v2311, ptr addrspace(1) %address.local1036, align 2, !dbg !51
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1015, !dbg !51
  store half %v2305, ptr addrspace(1) %address.local1038, align 2, !dbg !51
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1018, !dbg !51
  store half %v2304, ptr addrspace(1) %address.local1040, align 2, !dbg !51
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1021, !dbg !51
  store half %v2303, ptr addrspace(1) %address.local1042, align 2, !dbg !51
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1043, i64 %address.local1024, !dbg !51
  store half %v2302, ptr addrspace(1) %address.local1044, align 2, !dbg !51
  br label %v2334, !dbg !51

v2334:                                            ; preds = %v2296, %v2295
  br i1 true, label %v2335, label %v2373, !dbg !51

v2335:                                            ; preds = %v2334
  %v2336 = insertelement <4 x float> poison, float %v2173, i64 0, !dbg !49
  %v2337 = insertelement <4 x float> %v2336, float %v2179, i64 1, !dbg !49
  %v2338 = insertelement <4 x float> %v2337, float %v2174, i64 2, !dbg !49
  %v2339 = insertelement <4 x float> %v2338, float %v2180, i64 3, !dbg !49
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 1.000000e+00), %v2124, !dbg !49
  %normalized115 = fmul <4 x float> %v2339, %row.reciprocal114, !dbg !49
  %v2340 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !50
  %v2341 = extractelement <4 x half> %v2340, i64 3, !dbg !50
  %v2342 = extractelement <4 x half> %v2340, i64 2, !dbg !50
  %v2343 = extractelement <4 x half> %v2340, i64 1, !dbg !50
  %v2344 = extractelement <4 x half> %v2340, i64 0, !dbg !50
  %v2345 = insertelement <4 x float> poison, float %v2063, i64 0, !dbg !49
  %v2346 = insertelement <4 x float> %v2345, float %v2069, i64 1, !dbg !49
  %v2347 = insertelement <4 x float> %v2346, float %v2064, i64 2, !dbg !49
  %v2348 = insertelement <4 x float> %v2347, float %v2070, i64 3, !dbg !49
  %normalized116 = fmul <4 x float> %v2348, %row.reciprocal114, !dbg !49
  %v2349 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !50
  %v2350 = extractelement <4 x half> %v2349, i64 3, !dbg !50
  %v2351 = extractelement <4 x half> %v2349, i64 2, !dbg !50
  %v2352 = extractelement <4 x half> %v2349, i64 1, !dbg !50
  %v2353 = extractelement <4 x half> %v2349, i64 0, !dbg !50
  %v2354 = insertelement <4 x float> poison, float %v1955, i64 0, !dbg !49
  %v2355 = insertelement <4 x float> %v2354, float %v1961, i64 1, !dbg !49
  %v2356 = insertelement <4 x float> %v2355, float %v1956, i64 2, !dbg !49
  %v2357 = insertelement <4 x float> %v2356, float %v1962, i64 3, !dbg !49
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 1.000000e+00), %v1906, !dbg !49
  %normalized118 = fmul <4 x float> %v2357, %row.reciprocal117, !dbg !49
  %v2358 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !50
  %v2359 = extractelement <4 x half> %v2358, i64 3, !dbg !50
  %v2360 = extractelement <4 x half> %v2358, i64 2, !dbg !50
  %v2361 = extractelement <4 x half> %v2358, i64 1, !dbg !50
  %v2362 = extractelement <4 x half> %v2358, i64 0, !dbg !50
  %v2363 = insertelement <4 x float> poison, float %v1845, i64 0, !dbg !49
  %v2364 = insertelement <4 x float> %v2363, float %v1851, i64 1, !dbg !49
  %v2365 = insertelement <4 x float> %v2364, float %v1846, i64 2, !dbg !49
  %v2366 = insertelement <4 x float> %v2365, float %v1852, i64 3, !dbg !49
  %normalized119 = fmul <4 x float> %v2366, %row.reciprocal117, !dbg !49
  %v2367 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !50
  %v2368 = extractelement <4 x half> %v2367, i64 3, !dbg !50
  %v2369 = extractelement <4 x half> %v2367, i64 2, !dbg !50
  %v2370 = extractelement <4 x half> %v2367, i64 1, !dbg !50
  %v2371 = extractelement <4 x half> %v2367, i64 0, !dbg !50
  %v2372 = shl nuw nsw i64 %v24, 19, !dbg !51
  %address.thread1045 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !51, !range !67, !frisk.register_lifetimes !34
  %address.local1046 = zext nneg i32 %address.thread1045 to i64, !dbg !40
  %address.local1047 = add i64 %v25, %address.local1046, !dbg !51
  %address.local1048 = lshr i64 %address.local1046, 6, !dbg !42
  %address.local1049 = shl nuw nsw i64 %address.local1048, 4, !dbg !51
  %address.local1050 = add i64 %address.local1047, %address.local1049, !dbg !51
  %address.local1051 = and i64 %address.local1046, 240, !dbg !51
  %address.local1052 = sub i64 %address.local1050, %address.local1051, !dbg !51
  %address.local1053 = add i64 %address.local1052, %v26, !dbg !51
  %address.local1054 = shl nuw nsw i64 %address.local1053, 7, !dbg !51
  %address.local1055 = add nuw nsw i64 %v2372, %address.local1054, !dbg !51
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1057 = lshr i64 %address.local1046, 2, !dbg !42
  %address.local1058 = and i64 %address.local1057, 12, !dbg !42
  %address.local1059 = or disjoint i64 %address.local1058, 64, !dbg !51
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1059, !dbg !51
  store half %v2371, ptr addrspace(1) %address.local1060, align 2, !dbg !51
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1062 = or disjoint i64 %address.local1058, 65, !dbg !51
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1061, i64 %address.local1062, !dbg !51
  store half %v2370, ptr addrspace(1) %address.local1063, align 2, !dbg !51
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1065 = or disjoint i64 %address.local1058, 66, !dbg !51
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1064, i64 %address.local1065, !dbg !51
  store half %v2369, ptr addrspace(1) %address.local1066, align 2, !dbg !51
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1068 = or disjoint i64 %address.local1058, 67, !dbg !51
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1068, !dbg !51
  store half %v2368, ptr addrspace(1) %address.local1069, align 2, !dbg !51
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1071 = or disjoint i64 %address.local1058, 80, !dbg !51
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1071, !dbg !51
  store half %v2362, ptr addrspace(1) %address.local1072, align 2, !dbg !51
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1074 = or disjoint i64 %address.local1058, 81, !dbg !51
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1074, !dbg !51
  store half %v2361, ptr addrspace(1) %address.local1075, align 2, !dbg !51
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1077 = or disjoint i64 %address.local1058, 82, !dbg !51
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1076, i64 %address.local1077, !dbg !51
  store half %v2360, ptr addrspace(1) %address.local1078, align 2, !dbg !51
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1080 = or disjoint i64 %address.local1058, 83, !dbg !51
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1080, !dbg !51
  store half %v2359, ptr addrspace(1) %address.local1081, align 2, !dbg !51
  %address.local1082 = shl i64 %address.local1053, 7, !dbg !51
  %address.local1083 = add i64 %address.local1082, 8192, !dbg !51
  %address.local1084 = add nuw nsw i64 %v2372, %address.local1083, !dbg !51
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1085, i64 %address.local1059, !dbg !51
  store half %v2353, ptr addrspace(1) %address.local1086, align 2, !dbg !51
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1087, i64 %address.local1062, !dbg !51
  store half %v2352, ptr addrspace(1) %address.local1088, align 2, !dbg !51
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1065, !dbg !51
  store half %v2351, ptr addrspace(1) %address.local1090, align 2, !dbg !51
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1091, i64 %address.local1068, !dbg !51
  store half %v2350, ptr addrspace(1) %address.local1092, align 2, !dbg !51
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1071, !dbg !51
  store half %v2344, ptr addrspace(1) %address.local1094, align 2, !dbg !51
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1074, !dbg !51
  store half %v2343, ptr addrspace(1) %address.local1096, align 2, !dbg !51
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1097, i64 %address.local1077, !dbg !51
  store half %v2342, ptr addrspace(1) %address.local1098, align 2, !dbg !51
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1080, !dbg !51
  store half %v2341, ptr addrspace(1) %address.local1100, align 2, !dbg !51
  br label %v2373, !dbg !51

v2373:                                            ; preds = %v2335, %v2334
  br i1 true, label %v2374, label %v2412, !dbg !51

v2374:                                            ; preds = %v2373
  %v2375 = insertelement <4 x float> poison, float %v2145, i64 0, !dbg !49
  %v2376 = insertelement <4 x float> %v2375, float %v2151, i64 1, !dbg !49
  %v2377 = insertelement <4 x float> %v2376, float %v2146, i64 2, !dbg !49
  %v2378 = insertelement <4 x float> %v2377, float %v2152, i64 3, !dbg !49
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 1.000000e+00), %v2124, !dbg !49
  %normalized121 = fmul <4 x float> %v2378, %row.reciprocal120, !dbg !49
  %v2379 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !50
  %v2380 = extractelement <4 x half> %v2379, i64 3, !dbg !50
  %v2381 = extractelement <4 x half> %v2379, i64 2, !dbg !50
  %v2382 = extractelement <4 x half> %v2379, i64 1, !dbg !50
  %v2383 = extractelement <4 x half> %v2379, i64 0, !dbg !50
  %v2384 = insertelement <4 x float> poison, float %v2034, i64 0, !dbg !49
  %v2385 = insertelement <4 x float> %v2384, float %v2040, i64 1, !dbg !49
  %v2386 = insertelement <4 x float> %v2385, float %v2035, i64 2, !dbg !49
  %v2387 = insertelement <4 x float> %v2386, float %v2041, i64 3, !dbg !49
  %normalized122 = fmul <4 x float> %v2387, %row.reciprocal120, !dbg !49
  %v2388 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !50
  %v2389 = extractelement <4 x half> %v2388, i64 3, !dbg !50
  %v2390 = extractelement <4 x half> %v2388, i64 2, !dbg !50
  %v2391 = extractelement <4 x half> %v2388, i64 1, !dbg !50
  %v2392 = extractelement <4 x half> %v2388, i64 0, !dbg !50
  %v2393 = insertelement <4 x float> poison, float %v1927, i64 0, !dbg !49
  %v2394 = insertelement <4 x float> %v2393, float %v1933, i64 1, !dbg !49
  %v2395 = insertelement <4 x float> %v2394, float %v1928, i64 2, !dbg !49
  %v2396 = insertelement <4 x float> %v2395, float %v1934, i64 3, !dbg !49
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 1.000000e+00), %v1906, !dbg !49
  %normalized124 = fmul <4 x float> %v2396, %row.reciprocal123, !dbg !49
  %v2397 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !50
  %v2398 = extractelement <4 x half> %v2397, i64 3, !dbg !50
  %v2399 = extractelement <4 x half> %v2397, i64 2, !dbg !50
  %v2400 = extractelement <4 x half> %v2397, i64 1, !dbg !50
  %v2401 = extractelement <4 x half> %v2397, i64 0, !dbg !50
  %v2402 = insertelement <4 x float> poison, float %v1816, i64 0, !dbg !49
  %v2403 = insertelement <4 x float> %v2402, float %v1822, i64 1, !dbg !49
  %v2404 = insertelement <4 x float> %v2403, float %v1817, i64 2, !dbg !49
  %v2405 = insertelement <4 x float> %v2404, float %v1823, i64 3, !dbg !49
  %normalized125 = fmul <4 x float> %v2405, %row.reciprocal123, !dbg !49
  %v2406 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !50
  %v2407 = extractelement <4 x half> %v2406, i64 3, !dbg !50
  %v2408 = extractelement <4 x half> %v2406, i64 2, !dbg !50
  %v2409 = extractelement <4 x half> %v2406, i64 1, !dbg !50
  %v2410 = extractelement <4 x half> %v2406, i64 0, !dbg !50
  %v2411 = shl nuw nsw i64 %v24, 19, !dbg !51
  %address.thread1101 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !51, !range !67, !frisk.register_lifetimes !34
  %address.local1102 = zext nneg i32 %address.thread1101 to i64, !dbg !40
  %address.local1103 = add i64 %v25, %address.local1102, !dbg !51
  %address.local1104 = lshr i64 %address.local1102, 6, !dbg !42
  %address.local1105 = shl nuw nsw i64 %address.local1104, 4, !dbg !51
  %address.local1106 = add i64 %address.local1103, %address.local1105, !dbg !51
  %address.local1107 = and i64 %address.local1102, 240, !dbg !51
  %address.local1108 = sub i64 %address.local1106, %address.local1107, !dbg !51
  %address.local1109 = add i64 %address.local1108, %v26, !dbg !51
  %address.local1110 = shl nuw nsw i64 %address.local1109, 7, !dbg !51
  %address.local1111 = add nuw nsw i64 %v2411, %address.local1110, !dbg !51
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1113 = lshr i64 %address.local1102, 2, !dbg !42
  %address.local1114 = and i64 %address.local1113, 12, !dbg !42
  %address.local1115 = or disjoint i64 %address.local1114, 96, !dbg !51
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1115, !dbg !51
  store half %v2410, ptr addrspace(1) %address.local1116, align 2, !dbg !51
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1118 = or disjoint i64 %address.local1114, 97, !dbg !51
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1118, !dbg !51
  store half %v2409, ptr addrspace(1) %address.local1119, align 2, !dbg !51
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1121 = or disjoint i64 %address.local1114, 98, !dbg !51
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1121, !dbg !51
  store half %v2408, ptr addrspace(1) %address.local1122, align 2, !dbg !51
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1124 = or disjoint i64 %address.local1114, 99, !dbg !51
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1124, !dbg !51
  store half %v2407, ptr addrspace(1) %address.local1125, align 2, !dbg !51
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1127 = or disjoint i64 %address.local1114, 112, !dbg !51
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1126, i64 %address.local1127, !dbg !51
  store half %v2401, ptr addrspace(1) %address.local1128, align 2, !dbg !51
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1130 = or disjoint i64 %address.local1114, 113, !dbg !51
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1130, !dbg !51
  store half %v2400, ptr addrspace(1) %address.local1131, align 2, !dbg !51
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1133 = or disjoint i64 %address.local1114, 114, !dbg !51
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1132, i64 %address.local1133, !dbg !51
  store half %v2399, ptr addrspace(1) %address.local1134, align 2, !dbg !51
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1136 = or i64 %address.local1113, 115, !dbg !51
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1136, !dbg !51
  store half %v2398, ptr addrspace(1) %address.local1137, align 2, !dbg !51
  %address.local1138 = shl i64 %address.local1109, 7, !dbg !51
  %address.local1139 = add i64 %address.local1138, 8192, !dbg !51
  %address.local1140 = add nuw nsw i64 %v2411, %address.local1139, !dbg !51
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1115, !dbg !51
  store half %v2392, ptr addrspace(1) %address.local1142, align 2, !dbg !51
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1118, !dbg !51
  store half %v2391, ptr addrspace(1) %address.local1144, align 2, !dbg !51
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1121, !dbg !51
  store half %v2390, ptr addrspace(1) %address.local1146, align 2, !dbg !51
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1124, !dbg !51
  store half %v2389, ptr addrspace(1) %address.local1148, align 2, !dbg !51
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1127, !dbg !51
  store half %v2383, ptr addrspace(1) %address.local1150, align 2, !dbg !51
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1130, !dbg !51
  store half %v2382, ptr addrspace(1) %address.local1152, align 2, !dbg !51
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1133, !dbg !51
  store half %v2381, ptr addrspace(1) %address.local1154, align 2, !dbg !51
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1136, !dbg !51
  store half %v2380, ptr addrspace(1) %address.local1156, align 2, !dbg !51
  br label %v2412, !dbg !51

v2412:                                            ; preds = %v2374, %v2373
  ret void, !dbg !38
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

!llvm.dbg.cu = !{!36}
!llvm.module.flags = !{!66}

!33 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline")
!34 = !{}
!35 = !DISubroutineType(types: !34)
!36 = distinct !DICompileUnit(language: DW_LANG_C, file: !33, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!37 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !33, file: !33, line: 23, type: !35, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !36)
!38 = !DILocation(line: 23, column: 3, scope: !37)
!39 = !DILocation(line: 60, column: 24, scope: !37)
!40 = !DILocation(line: 70, column: 22, scope: !37)
!41 = !DILocation(line: 48, column: 17, scope: !37)
!42 = !DILocation(line: 72, column: 24, scope: !37)
!43 = !DILocation(line: 77, column: 19, scope: !37)
!44 = !DILocation(line: 102, column: 23, scope: !37)
!45 = !DILocation(line: 0, scope: !37)
!46 = !DILocation(line: 71, column: 22, scope: !37)
!47 = !DILocation(line: 73, column: 26, scope: !37)
!48 = !DILocation(line: 101, column: 20, scope: !37)
!49 = !DILocation(line: 107, column: 23, scope: !37)
!50 = !DILocation(line: 108, column: 19, scope: !37)
!51 = !DILocation(line: 109, column: 9, scope: !37)
!52 = !DILocation(line: 79, column: 24, scope: !37)
!53 = !DILocation(line: 80, column: 23, scope: !37)
!54 = !DILocation(line: 81, column: 26, scope: !37)
!55 = !DILocation(line: 88, column: 21, scope: !37)
!56 = !DILocation(line: 89, column: 20, scope: !37)
!57 = !DILocation(line: 90, column: 22, scope: !37)
!58 = !DILocation(line: 91, column: 23, scope: !37)
!59 = !DILocation(line: 92, column: 20, scope: !37)
!60 = !DILocation(line: 93, column: 22, scope: !37)
!61 = !DILocation(line: 94, column: 22, scope: !37)
!62 = !DILocation(line: 95, column: 20, scope: !37)
!63 = !DILocation(line: 96, column: 16, scope: !37)
!64 = !DILocation(line: 103, column: 22, scope: !37)
!65 = !{i32 256, i32 1, i32 1}
!66 = !{i32 2, !"Debug Info Version", i32 3}
!67 = !{i32 0, i32 256}
