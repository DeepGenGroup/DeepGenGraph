; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/local_reduction_pass/compiler-bn128/run-20261026-112711-jne7_i1s/grid-000003/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/local_reduction_pass/compiler-bn128/run-20261026-112711-jne7_i1s/grid-000003/input.mlir"

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
  %address.local495 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local496 = and i64 %address.local495, 1536, !dbg !8
  %address.local497 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local496, !dbg !8
  %address.local498 = ptrtoint ptr addrspace(3) %address.local497 to i64, !dbg !8
  %address.local499 = trunc i64 %address.local498 to i32, !dbg !8
  %v34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local499), !dbg !8
  %v35 = zext i32 %v34 to i64, !dbg !8
  %v36 = inttoptr i64 %v35 to ptr addrspace(3), !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local501 = and i64 %address.local500, 16256, !dbg !8
  %address.local502 = lshr i64 %address.local, 4, !dbg !8
  %address.local503 = and i64 %address.local502, 8, !dbg !8
  %address.local504 = or disjoint i64 %address.local501, %address.local503, !dbg !8
  %address.local505 = getelementptr [2 x i8], ptr addrspace(1) %v33, i64 %address.local504, !dbg !8
  %v37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v38 = ptrtoint ptr addrspace(1) %address.local505 to i64, !dbg !8
  %v39 = sub i64 %v38, %v37, !dbg !8
  %dma.buffer.offset1768 = trunc i64 %v39 to i32, !dbg !8
  %v40 = trunc i64 %v37 to i32, !dbg !8
  %v41 = insertelement <4 x i32> poison, i32 %v40, i64 0, !dbg !8
  %v42 = lshr i64 %v37, 32, !dbg !8
  %v43 = trunc i64 %v42 to i32, !dbg !8
  %v44 = insertelement <4 x i32> %v41, i32 %v43, i64 1, !dbg !8
  %v45 = insertelement <4 x i32> %v44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1769 = insertelement <4 x i32> %v45, i32 131072, i64 3, !dbg !8
  %v46 = ptrtoint ptr addrspace(3) %v36 to i32, !dbg !8
  %dma.buffer.lds1770 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v46), !dbg !8
  %v47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1770, i32 %dma.buffer.offset1768, <4 x i32> %dma.buffer.rsrc1769) #10, !dbg !8
  %address.local506 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local507 = shl nuw nsw i64 %address.local506, 3, !dbg !8
  %address.local508 = and i64 %address.local507, 3584, !dbg !8
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local508, !dbg !8
  %address.local510 = ptrtoint ptr addrspace(3) %address.local509 to i64, !dbg !8
  %address.local511 = trunc i64 %address.local510 to i32, !dbg !8
  %v48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local511), !dbg !8
  %v49 = zext i32 %v48 to i64, !dbg !8
  %v50 = inttoptr i64 %v49 to ptr addrspace(3), !dbg !8
  %address.local512 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local513 = and i64 %address.local512, 16256, !dbg !8
  %address.local514 = lshr i64 %address.local506, 4, !dbg !8
  %address.local515 = and i64 %address.local514, 24, !dbg !8
  %address.local516 = or disjoint i64 %address.local513, %address.local515, !dbg !8
  %address.local517 = getelementptr [2 x i8], ptr addrspace(1) %v33, i64 %address.local516, !dbg !8
  %v51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v52 = ptrtoint ptr addrspace(1) %address.local517 to i64, !dbg !8
  %v53 = sub i64 %v52, %v51, !dbg !8
  %dma.buffer.offset1765 = trunc i64 %v53 to i32, !dbg !8
  %v54 = trunc i64 %v51 to i32, !dbg !8
  %v55 = insertelement <4 x i32> poison, i32 %v54, i64 0, !dbg !8
  %v56 = lshr i64 %v51, 32, !dbg !8
  %v57 = trunc i64 %v56 to i32, !dbg !8
  %v58 = insertelement <4 x i32> %v55, i32 %v57, i64 1, !dbg !8
  %v59 = insertelement <4 x i32> %v58, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1766 = insertelement <4 x i32> %v59, i32 131072, i64 3, !dbg !8
  %v60 = ptrtoint ptr addrspace(3) %v50 to i32, !dbg !8
  %dma.buffer.lds1767 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v60), !dbg !8
  %v61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1767, i32 %dma.buffer.offset1765, <4 x i32> %dma.buffer.rsrc1766) #10, !dbg !8
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
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local496, !dbg !8
  %address.local519 = ptrtoint ptr addrspace(3) %address.local518 to i64, !dbg !8
  %address.local520 = trunc i64 %address.local519 to i32, !dbg !8
  %v68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local520), !dbg !8
  %v69 = zext i32 %v68 to i64, !dbg !8
  %v70 = inttoptr i64 %v69 to ptr addrspace(3), !dbg !8
  %address.local521 = getelementptr [2 x i8], ptr addrspace(1) %v67, i64 %address.local504, !dbg !8
  %v71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v72 = ptrtoint ptr addrspace(1) %address.local521 to i64, !dbg !8
  %v73 = sub i64 %v72, %v71, !dbg !8
  %dma.buffer.offset1762 = trunc i64 %v73 to i32, !dbg !8
  %v74 = trunc i64 %v71 to i32, !dbg !8
  %v75 = insertelement <4 x i32> poison, i32 %v74, i64 0, !dbg !8
  %v76 = lshr i64 %v71, 32, !dbg !8
  %v77 = trunc i64 %v76 to i32, !dbg !8
  %v78 = insertelement <4 x i32> %v75, i32 %v77, i64 1, !dbg !8
  %v79 = insertelement <4 x i32> %v78, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1763 = insertelement <4 x i32> %v79, i32 131072, i64 3, !dbg !8
  %v80 = ptrtoint ptr addrspace(3) %v70 to i32, !dbg !8
  %dma.buffer.lds1764 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v80), !dbg !8
  %v81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1764, i32 %dma.buffer.offset1762, <4 x i32> %dma.buffer.rsrc1763) #10, !dbg !8
  %address.local522 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local508, !dbg !8
  %address.local523 = ptrtoint ptr addrspace(3) %address.local522 to i64, !dbg !8
  %address.local524 = trunc i64 %address.local523 to i32, !dbg !8
  %v82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local524), !dbg !8
  %v83 = zext i32 %v82 to i64, !dbg !8
  %v84 = inttoptr i64 %v83 to ptr addrspace(3), !dbg !8
  %address.local525 = getelementptr [2 x i8], ptr addrspace(1) %v67, i64 %address.local516, !dbg !8
  %v85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v86 = ptrtoint ptr addrspace(1) %address.local525 to i64, !dbg !8
  %v87 = sub i64 %v86, %v85, !dbg !8
  %dma.buffer.offset1759 = trunc i64 %v87 to i32, !dbg !8
  %v88 = trunc i64 %v85 to i32, !dbg !8
  %v89 = insertelement <4 x i32> poison, i32 %v88, i64 0, !dbg !8
  %v90 = lshr i64 %v85, 32, !dbg !8
  %v91 = trunc i64 %v90 to i32, !dbg !8
  %v92 = insertelement <4 x i32> %v89, i32 %v91, i64 1, !dbg !8
  %v93 = insertelement <4 x i32> %v92, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1760 = insertelement <4 x i32> %v93, i32 131072, i64 3, !dbg !8
  %v94 = ptrtoint ptr addrspace(3) %v84 to i32, !dbg !8
  %dma.buffer.lds1761 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v94), !dbg !8
  %v95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1761, i32 %dma.buffer.offset1759, <4 x i32> %dma.buffer.rsrc1760) #10, !dbg !8
  %address.local526 = lshr i32 %address.thread, 2, !dbg !11
  %address.local527 = and i32 %address.local526, 4, !dbg !11
  %address.local528 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local529 = and i32 %address.local528, 384, !dbg !11
  %address.local530 = or disjoint i32 %address.local527, %address.local529, !dbg !11
  %address.local531 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local532 = and i32 %address.local531, 120, !dbg !11
  %address.local533 = or disjoint i32 %address.local530, %address.local532, !dbg !11
  %address.local534 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local535 = and i32 %address.local534, 1024, !dbg !11
  %address.local536 = or disjoint i32 %address.local533, %address.local535, !dbg !11
  %address.local537 = zext nneg i32 %address.local536 to i64, !dbg !11
  %address.local538 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local537, !dbg !11
  %v96 = load <4 x half>, ptr addrspace(3) %address.local538, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local538, i64 4096, !dbg !11
  %v97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %v98 = fmul <4 x half> %v96, splat (half 0xH3015), !dbg !11
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local538, i64 1024, !dbg !11
  %v99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !11
  %v100 = fmul <4 x half> %v97, splat (half 0xH3015), !dbg !11
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local538, i64 5120, !dbg !11
  %v101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !11
  %v102 = fmul <4 x half> %v99, splat (half 0xH3015), !dbg !11
  %v103 = fmul <4 x half> %v101, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %v104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %v30, !dbg !8
  %v105 = getelementptr i8, ptr addrspace(1) %v104, i64 128, !dbg !8
  %v106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local499), !dbg !8
  %v107 = zext i32 %v106 to i64, !dbg !8
  %v108 = inttoptr i64 %v107 to ptr addrspace(3), !dbg !8
  %address.local581 = getelementptr [2 x i8], ptr addrspace(1) %v105, i64 %address.local504, !dbg !8
  %v109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v110 = ptrtoint ptr addrspace(1) %address.local581 to i64, !dbg !8
  %v111 = sub i64 %v110, %v109, !dbg !8
  %dma.buffer.offset1756 = trunc i64 %v111 to i32, !dbg !8
  %v112 = trunc i64 %v109 to i32, !dbg !8
  %v113 = insertelement <4 x i32> poison, i32 %v112, i64 0, !dbg !8
  %v114 = lshr i64 %v109, 32, !dbg !8
  %v115 = trunc i64 %v114 to i32, !dbg !8
  %v116 = insertelement <4 x i32> %v113, i32 %v115, i64 1, !dbg !8
  %v117 = insertelement <4 x i32> %v116, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1757 = insertelement <4 x i32> %v117, i32 131072, i64 3, !dbg !8
  %v118 = ptrtoint ptr addrspace(3) %v108 to i32, !dbg !8
  %dma.buffer.lds1758 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v118), !dbg !8
  %v119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1758, i32 %dma.buffer.offset1756, <4 x i32> %dma.buffer.rsrc1757) #10, !dbg !8
  %v120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local511), !dbg !8
  %v121 = zext i32 %v120 to i64, !dbg !8
  %v122 = inttoptr i64 %v121 to ptr addrspace(3), !dbg !8
  %address.local582 = getelementptr [2 x i8], ptr addrspace(1) %v105, i64 %address.local516, !dbg !8
  %v123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v124 = ptrtoint ptr addrspace(1) %address.local582 to i64, !dbg !8
  %v125 = sub i64 %v124, %v123, !dbg !8
  %dma.buffer.offset1753 = trunc i64 %v125 to i32, !dbg !8
  %v126 = trunc i64 %v123 to i32, !dbg !8
  %v127 = insertelement <4 x i32> poison, i32 %v126, i64 0, !dbg !8
  %v128 = lshr i64 %v123, 32, !dbg !8
  %v129 = trunc i64 %v128 to i32, !dbg !8
  %v130 = insertelement <4 x i32> %v127, i32 %v129, i64 1, !dbg !8
  %v131 = insertelement <4 x i32> %v130, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1754 = insertelement <4 x i32> %v131, i32 131072, i64 3, !dbg !8
  %v132 = ptrtoint ptr addrspace(3) %v122 to i32, !dbg !8
  %dma.buffer.lds1755 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v132), !dbg !8
  %v133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1755, i32 %dma.buffer.offset1753, <4 x i32> %dma.buffer.rsrc1754) #10, !dbg !8
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local538, i64 8192, !dbg !11
  %v134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !11
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local538, i64 12288, !dbg !11
  %v135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !11
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local538, i64 9216, !dbg !11
  %v136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !11
  %v137 = fmul <4 x half> %v134, splat (half 0xH3015), !dbg !11
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local538, i64 13312, !dbg !11
  %v138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !11
  %v139 = fmul <4 x half> %v135, splat (half 0xH3015), !dbg !11
  %v140 = fmul <4 x half> %v136, splat (half 0xH3015), !dbg !11
  %v141 = fmul <4 x half> %v138, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %v142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %v30, !dbg !8
  %v143 = getelementptr i8, ptr addrspace(1) %v142, i64 192, !dbg !8
  %v144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local520), !dbg !8
  %v145 = zext i32 %v144 to i64, !dbg !8
  %v146 = inttoptr i64 %v145 to ptr addrspace(3), !dbg !8
  %address.local587 = getelementptr [2 x i8], ptr addrspace(1) %v143, i64 %address.local504, !dbg !8
  %v147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v148 = ptrtoint ptr addrspace(1) %address.local587 to i64, !dbg !8
  %v149 = sub i64 %v148, %v147, !dbg !8
  %dma.buffer.offset1750 = trunc i64 %v149 to i32, !dbg !8
  %v150 = trunc i64 %v147 to i32, !dbg !8
  %v151 = insertelement <4 x i32> poison, i32 %v150, i64 0, !dbg !8
  %v152 = lshr i64 %v147, 32, !dbg !8
  %v153 = trunc i64 %v152 to i32, !dbg !8
  %v154 = insertelement <4 x i32> %v151, i32 %v153, i64 1, !dbg !8
  %v155 = insertelement <4 x i32> %v154, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1751 = insertelement <4 x i32> %v155, i32 131072, i64 3, !dbg !8
  %v156 = ptrtoint ptr addrspace(3) %v146 to i32, !dbg !8
  %dma.buffer.lds1752 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v156), !dbg !8
  %v157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1752, i32 %dma.buffer.offset1750, <4 x i32> %dma.buffer.rsrc1751) #10, !dbg !8
  %v158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local524), !dbg !8
  %v159 = zext i32 %v158 to i64, !dbg !8
  %v160 = inttoptr i64 %v159 to ptr addrspace(3), !dbg !8
  %address.local588 = getelementptr [2 x i8], ptr addrspace(1) %v143, i64 %address.local516, !dbg !8
  %v161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %v162 = ptrtoint ptr addrspace(1) %address.local588 to i64, !dbg !8
  %v163 = sub i64 %v162, %v161, !dbg !8
  %dma.buffer.offset1747 = trunc i64 %v163 to i32, !dbg !8
  %v164 = trunc i64 %v161 to i32, !dbg !8
  %v165 = insertelement <4 x i32> poison, i32 %v164, i64 0, !dbg !8
  %v166 = lshr i64 %v161, 32, !dbg !8
  %v167 = trunc i64 %v166 to i32, !dbg !8
  %v168 = insertelement <4 x i32> %v165, i32 %v167, i64 1, !dbg !8
  %v169 = insertelement <4 x i32> %v168, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1748 = insertelement <4 x i32> %v169, i32 131072, i64 3, !dbg !8
  %v170 = ptrtoint ptr addrspace(3) %v160 to i32, !dbg !8
  %dma.buffer.lds1749 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v170), !dbg !8
  %v171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1749, i32 %dma.buffer.offset1747, <4 x i32> %dma.buffer.rsrc1748) #10, !dbg !8
  %v172 = load <4 x half>, ptr addrspace(3) %address.local538, align 8, !dbg !11
  %lds.const.gep1575 = getelementptr i8, ptr addrspace(3) %address.local538, i64 4096, !dbg !11
  %v173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1575, align 8, !dbg !11
  %lds.const.gep1576 = getelementptr i8, ptr addrspace(3) %address.local538, i64 1024, !dbg !11
  %v174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1576, align 8, !dbg !11
  %v175 = fmul <4 x half> %v172, splat (half 0xH3015), !dbg !11
  %lds.const.gep1577 = getelementptr i8, ptr addrspace(3) %address.local538, i64 5120, !dbg !11
  %v176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1577, align 8, !dbg !11
  %v177 = fmul <4 x half> %v173, splat (half 0xH3015), !dbg !11
  %v178 = fmul <4 x half> %v174, splat (half 0xH3015), !dbg !11
  %v179 = fmul <4 x half> %v176, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1578 = getelementptr i8, ptr addrspace(3) %address.local538, i64 8192, !dbg !11
  %v180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1578, align 8, !dbg !11
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local538, i64 12288, !dbg !11
  %v181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !11
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local538, i64 9216, !dbg !11
  %v182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !11
  %v183 = fmul <4 x half> %v180, splat (half 0xH3015), !dbg !11
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local538, i64 13312, !dbg !11
  %v184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !11
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
  %index.compact435 = zext nneg i32 %v200 to i64, !dbg !17
  %v201 = or disjoint i64 %v199, %index.compact435, !dbg !17
  %v202 = add nuw nsw i64 %v201, 1, !dbg !17
  %v203 = sdiv i64 %v202, 2, !dbg !17
  br label %v204, !dbg !7

v204:                                              ; preds = %v845, %4
  %v205 = phi i64 [ %v1622, %v845 ], [ 0, %4 ], !dbg !7
  %v206 = phi [2 x <8 x float>] [ %v1607, %v845 ], [ zeroinitializer, %4 ], !dbg !7
  %v207 = phi [2 x <1 x float>] [ %v1213, %v845 ], [ zeroinitializer, %4 ], !dbg !7
  %v208 = phi [2 x <1 x float>] [ %v1609, %v845 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %v209 = phi [2 x <8 x float>] [ %v1613, %v845 ], [ zeroinitializer, %4 ], !dbg !7
  %v210 = phi [2 x <8 x float>] [ %v1617, %v845 ], [ zeroinitializer, %4 ], !dbg !7
  %v211 = phi [2 x <8 x float>] [ %v1621, %v845 ], [ zeroinitializer, %4 ], !dbg !7
  %v212 = icmp slt i64 %v205, %v191, !dbg !7
  br i1 %v212, label %v213, label %v1623, !dbg !7

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
  %address.thread589 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local590 = zext nneg i32 %address.thread589 to i64, !dbg !8
  %address.local591 = shl nuw nsw i64 %address.local590, 3, !dbg !8
  %address.local592 = and i64 %address.local591, 1536, !dbg !8
  %address.local593 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local592, !dbg !8
  %address.local594 = ptrtoint ptr addrspace(3) %address.local593 to i64, !dbg !8
  %address.local595 = trunc i64 %address.local594 to i32, !dbg !8
  %v227 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local595), !dbg !16
  %v228 = zext i32 %v227 to i64, !dbg !16
  %v229 = inttoptr i64 %v228 to ptr addrspace(3), !dbg !16
  %address.local596 = lshr i64 %address.local590, 4, !dbg !16
  %address.local597 = and i64 %address.local596, 8, !dbg !16
  %address.local598 = and i64 %address.local590, 123, !dbg !16
  %address.local599 = lshr i64 %address.local590, 2, !dbg !16
  %address.local600 = and i64 %address.local599, 4, !dbg !16
  %address.local601 = add nuw nsw i64 %address.local600, %address.local590, !dbg !16
  %address.local602 = and i64 %address.local601, 4, !dbg !16
  %address.local603 = or disjoint i64 %address.local598, %address.local602, !dbg !16
  %address.local604 = shl nuw nsw i64 %address.local603, 7, !dbg !16
  %address.local605 = or disjoint i64 %address.local597, %address.local604, !dbg !16
  %address.local606 = getelementptr [2 x i8], ptr addrspace(1) %v226, i64 %address.local605, !dbg !16
  %v230 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v231 = ptrtoint ptr addrspace(1) %address.local606 to i64, !dbg !16
  %v232 = sub i64 %v231, %v230, !dbg !16
  %dma.buffer.offset1744 = trunc i64 %v232 to i32, !dbg !16
  %v233 = trunc i64 %v230 to i32, !dbg !16
  %v234 = insertelement <4 x i32> poison, i32 %v233, i64 0, !dbg !16
  %v235 = lshr i64 %v230, 32, !dbg !16
  %v236 = trunc i64 %v235 to i32, !dbg !16
  %v237 = insertelement <4 x i32> %v234, i32 %v236, i64 1, !dbg !16
  %v238 = insertelement <4 x i32> %v237, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1745 = insertelement <4 x i32> %v238, i32 131072, i64 3, !dbg !16
  %v239 = ptrtoint ptr addrspace(3) %v229 to i32, !dbg !16
  %dma.buffer.lds1746 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v239), !dbg !16
  %v240 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1746, i32 %dma.buffer.offset1744, <4 x i32> %dma.buffer.rsrc1745) #10, !dbg !16
  %address.local607 = or disjoint i64 %address.local590, 256, !dbg !8
  %address.local608 = shl nuw nsw i64 %address.local607, 3, !dbg !8
  %address.local609 = and i64 %address.local608, 3584, !dbg !8
  %address.local610 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local609, !dbg !8
  %address.local611 = ptrtoint ptr addrspace(3) %address.local610 to i64, !dbg !8
  %address.local612 = trunc i64 %address.local611 to i32, !dbg !8
  %v241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local612), !dbg !16
  %v242 = zext i32 %v241 to i64, !dbg !16
  %v243 = inttoptr i64 %v242 to ptr addrspace(3), !dbg !16
  %address.local613 = lshr i64 %address.local607, 4, !dbg !16
  %address.local614 = and i64 %address.local613, 24, !dbg !16
  %address.local615 = and i64 %address.local590, 123, !dbg !16
  %address.local616 = lshr i64 %address.local590, 2, !dbg !16
  %address.local617 = and i64 %address.local616, 4, !dbg !16
  %address.local618 = add nuw nsw i64 %address.local617, %address.local590, !dbg !16
  %address.local619 = and i64 %address.local618, 4, !dbg !16
  %address.local620 = or disjoint i64 %address.local615, %address.local619, !dbg !16
  %address.local621 = shl nuw nsw i64 %address.local620, 7, !dbg !16
  %address.local622 = or disjoint i64 %address.local614, %address.local621, !dbg !16
  %address.local623 = getelementptr [2 x i8], ptr addrspace(1) %v226, i64 %address.local622, !dbg !16
  %v244 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v245 = ptrtoint ptr addrspace(1) %address.local623 to i64, !dbg !16
  %v246 = sub i64 %v245, %v244, !dbg !16
  %dma.buffer.offset1741 = trunc i64 %v246 to i32, !dbg !16
  %v247 = trunc i64 %v244 to i32, !dbg !16
  %v248 = insertelement <4 x i32> poison, i32 %v247, i64 0, !dbg !16
  %v249 = lshr i64 %v244, 32, !dbg !16
  %v250 = trunc i64 %v249 to i32, !dbg !16
  %v251 = insertelement <4 x i32> %v248, i32 %v250, i64 1, !dbg !16
  %v252 = insertelement <4 x i32> %v251, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1742 = insertelement <4 x i32> %v252, i32 131072, i64 3, !dbg !16
  %v253 = ptrtoint ptr addrspace(3) %v243 to i32, !dbg !16
  %dma.buffer.lds1743 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v253), !dbg !16
  %v254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1743, i32 %dma.buffer.offset1741, <4 x i32> %dma.buffer.rsrc1742) #10, !dbg !16
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
  %address.local624 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local592, !dbg !8
  %address.local625 = ptrtoint ptr addrspace(3) %address.local624 to i64, !dbg !8
  %address.local626 = trunc i64 %address.local625 to i32, !dbg !8
  %v279 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local626), !dbg !16
  %v280 = zext i32 %v279 to i64, !dbg !16
  %v281 = inttoptr i64 %v280 to ptr addrspace(3), !dbg !16
  %address.local627 = getelementptr [2 x i8], ptr addrspace(1) %v278, i64 %address.local605, !dbg !16
  %v282 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v283 = ptrtoint ptr addrspace(1) %address.local627 to i64, !dbg !16
  %v284 = sub i64 %v283, %v282, !dbg !16
  %dma.buffer.offset1738 = trunc i64 %v284 to i32, !dbg !16
  %v285 = trunc i64 %v282 to i32, !dbg !16
  %v286 = insertelement <4 x i32> poison, i32 %v285, i64 0, !dbg !16
  %v287 = lshr i64 %v282, 32, !dbg !16
  %v288 = trunc i64 %v287 to i32, !dbg !16
  %v289 = insertelement <4 x i32> %v286, i32 %v288, i64 1, !dbg !16
  %v290 = insertelement <4 x i32> %v289, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1739 = insertelement <4 x i32> %v290, i32 131072, i64 3, !dbg !16
  %v291 = ptrtoint ptr addrspace(3) %v281 to i32, !dbg !16
  %dma.buffer.lds1740 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v291), !dbg !16
  %v292 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1740, i32 %dma.buffer.offset1738, <4 x i32> %dma.buffer.rsrc1739) #10, !dbg !16
  %address.local628 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local609, !dbg !8
  %address.local629 = ptrtoint ptr addrspace(3) %address.local628 to i64, !dbg !8
  %address.local630 = trunc i64 %address.local629 to i32, !dbg !8
  %v293 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local630), !dbg !16
  %v294 = zext i32 %v293 to i64, !dbg !16
  %v295 = inttoptr i64 %v294 to ptr addrspace(3), !dbg !16
  %address.local631 = getelementptr [2 x i8], ptr addrspace(1) %v278, i64 %address.local622, !dbg !16
  %v296 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v297 = ptrtoint ptr addrspace(1) %address.local631 to i64, !dbg !16
  %v298 = sub i64 %v297, %v296, !dbg !16
  %dma.buffer.offset1735 = trunc i64 %v298 to i32, !dbg !16
  %v299 = trunc i64 %v296 to i32, !dbg !16
  %v300 = insertelement <4 x i32> poison, i32 %v299, i64 0, !dbg !16
  %v301 = lshr i64 %v296, 32, !dbg !16
  %v302 = trunc i64 %v301 to i32, !dbg !16
  %v303 = insertelement <4 x i32> %v300, i32 %v302, i64 1, !dbg !16
  %v304 = insertelement <4 x i32> %v303, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1736 = insertelement <4 x i32> %v304, i32 131072, i64 3, !dbg !16
  %v305 = ptrtoint ptr addrspace(3) %v295 to i32, !dbg !16
  %dma.buffer.lds1737 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v305), !dbg !16
  %v306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1737, i32 %dma.buffer.offset1735, <4 x i32> %dma.buffer.rsrc1736) #10, !dbg !16
  %address.local632 = mul nuw nsw i64 %address.local590, 160, !dbg !17
  %address.local633 = lshr i64 %address.local590, 2, !dbg !17
  %address.local634 = mul nsw i64 %address.local633, -640, !dbg !17
  %address.local635 = add nsw i64 %address.local632, %address.local634, !dbg !17
  %address.local636 = shl nuw nsw i64 %address.local590, 1, !dbg !17
  %address.local637 = and i64 %address.local636, 24, !dbg !17
  %address.local638 = or disjoint i64 %address.local635, %address.local637, !dbg !17
  %address.local639 = shl i64 %address.local590, 62, !dbg !17
  %address.local640 = ashr exact i64 %address.local639, 57, !dbg !17
  %address.local641 = and i64 %address.local640, -64, !dbg !17
  %address.local642 = add nsw i64 %address.local638, %address.local641, !dbg !17
  %address.local643 = lshr i64 %address.local590, 2, !dbg !11
  %address.local644 = and i64 %address.local643, 12, !dbg !11
  %address.local645 = add nsw i64 %address.local642, %address.local644, !dbg !17
  %address.local646 = and i64 %address.local590, 63, !dbg !11
  %address.local647 = lshr i64 %address.local646, 5, !dbg !11
  %address.local648 = mul nuw nsw i64 %address.local647, 1016, !dbg !11
  %address.local649 = add i64 %address.local645, %address.local648, !dbg !17
  %address.local650 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local649, !dbg !17
  %v307 = load <4 x half>, ptr addrspace(3) %address.local650, align 8, !dbg !17
  %v308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v307, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4096, !dbg !17
  %v309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !17
  %address.local661 = shl nuw nsw i32 %address.thread589, 1, !dbg !17
  %address.local662 = and i32 %address.local661, 24, !dbg !17
  %address.local663 = shl nuw nsw i32 %address.thread589, 8, !dbg !17
  %address.local664 = and i32 %address.local663, 768, !dbg !17
  %address.local665 = or disjoint i32 %address.local662, %address.local664, !dbg !17
  %address.local666 = shl nuw nsw i32 %address.thread589, 5, !dbg !17
  %address.local667 = and i32 %address.local666, 96, !dbg !17
  %address.local668 = or disjoint i32 %address.local665, %address.local667, !dbg !17
  %address.local669 = zext nneg i32 %address.local668 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %v203) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local670 = lshr i32 %address.thread589, 1, !dbg !17
  %address.local671 = and i32 %address.local670, 1, !dbg !17
  %address.local672 = zext nneg i32 %address.local671 to i64, !dbg !17
  %address.local673 = add nsw i64 %address.base, %address.local672, !dbg !17
  %address.local674 = shl i64 %address.local673, 6, !dbg !17
  %address.local675 = sub i64 %address.local669, %address.local674, !dbg !17
  %address.local676 = add i64 %address.local675, %address.local644, !dbg !17
  %address.local677 = add i64 %address.local676, %address.local648, !dbg !17
  %address.local678 = add i64 %address.local677, 32, !dbg !17
  %address.local679 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local678, !dbg !17
  %v310 = load <4 x half>, ptr addrspace(3) %address.local679, align 8, !dbg !17
  %v311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v309, <4 x float> %v308) #9, !dbg !17
  %v312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v310, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4096, !dbg !17
  %v313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !17
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local650, i64 128, !dbg !17
  %v314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !17
  %v315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v313, <4 x float> %v312) #9, !dbg !17
  %v316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v314, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4224, !dbg !17
  %v317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !17
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local679, i64 128, !dbg !17
  %v318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !17
  %v319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v317, <4 x float> %v316) #9, !dbg !17
  %v320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v318, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4224, !dbg !17
  %v321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !17
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1024, !dbg !17
  %v322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !17
  %v323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v321, <4 x float> %v320) #9, !dbg !17
  %v324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v322, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5120, !dbg !17
  %v325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !17
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1024, !dbg !17
  %v326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !17
  %v327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v325, <4 x float> %v324) #9, !dbg !17
  %v328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v326, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5120, !dbg !17
  %v329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !17
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1152, !dbg !17
  %v330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !17
  %v331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v329, <4 x float> %v328) #9, !dbg !17
  %v332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v330, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5248, !dbg !17
  %v333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !17
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1152, !dbg !17
  %v334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !17
  %v335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v333, <4 x float> %v332) #9, !dbg !17
  %v336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v98, <4 x half> %v334, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5248, !dbg !17
  %v337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !17
  %v338 = load <4 x half>, ptr addrspace(3) %address.local650, align 8, !dbg !17
  %v339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v100, <4 x half> %v337, <4 x float> %v336) #9, !dbg !17
  %v340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v338, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4096, !dbg !17
  %v341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !17
  %v342 = load <4 x half>, ptr addrspace(3) %address.local679, align 8, !dbg !17
  %v343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v341, <4 x float> %v340) #9, !dbg !17
  %v344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v342, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4096, !dbg !17
  %v345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !17
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local650, i64 128, !dbg !17
  %v346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !17
  %v347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v345, <4 x float> %v344) #9, !dbg !17
  %v348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v346, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4224, !dbg !17
  %v349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !17
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %address.local679, i64 128, !dbg !17
  %v350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !17
  %v351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v349, <4 x float> %v348) #9, !dbg !17
  %v352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v350, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4224, !dbg !17
  %v353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !17
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1024, !dbg !17
  %v354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !17
  %v355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v353, <4 x float> %v352) #9, !dbg !17
  %v356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v354, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5120, !dbg !17
  %v357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !17
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1024, !dbg !17
  %v358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !17
  %v359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v357, <4 x float> %v356) #9, !dbg !17
  %v360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v358, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5120, !dbg !17
  %v361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !17
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1152, !dbg !17
  %v362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !17
  %v363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v361, <4 x float> %v360) #9, !dbg !17
  %v364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v362, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5248, !dbg !17
  %v365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !17
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1152, !dbg !17
  %v366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !17
  %v367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v103, <4 x half> %v365, <4 x float> %v364) #9, !dbg !17
  %v368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v102, <4 x half> %v366, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5248, !dbg !17
  %v369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !17
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
  %v394 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local595), !dbg !16
  %v395 = zext i32 %v394 to i64, !dbg !16
  %v396 = inttoptr i64 %v395 to ptr addrspace(3), !dbg !16
  %address.local988 = getelementptr [2 x i8], ptr addrspace(1) %v393, i64 %address.local605, !dbg !16
  %v397 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v398 = ptrtoint ptr addrspace(1) %address.local988 to i64, !dbg !16
  %v399 = sub i64 %v398, %v397, !dbg !16
  %dma.buffer.offset1732 = trunc i64 %v399 to i32, !dbg !16
  %v400 = trunc i64 %v397 to i32, !dbg !16
  %v401 = insertelement <4 x i32> poison, i32 %v400, i64 0, !dbg !16
  %v402 = lshr i64 %v397, 32, !dbg !16
  %v403 = trunc i64 %v402 to i32, !dbg !16
  %v404 = insertelement <4 x i32> %v401, i32 %v403, i64 1, !dbg !16
  %v405 = insertelement <4 x i32> %v404, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1733 = insertelement <4 x i32> %v405, i32 131072, i64 3, !dbg !16
  %v406 = ptrtoint ptr addrspace(3) %v396 to i32, !dbg !16
  %dma.buffer.lds1734 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v406), !dbg !16
  %v407 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1734, i32 %dma.buffer.offset1732, <4 x i32> %dma.buffer.rsrc1733) #10, !dbg !16
  %v408 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local612), !dbg !16
  %v409 = zext i32 %v408 to i64, !dbg !16
  %v410 = inttoptr i64 %v409 to ptr addrspace(3), !dbg !16
  %address.local989 = getelementptr [2 x i8], ptr addrspace(1) %v393, i64 %address.local622, !dbg !16
  %v411 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v412 = ptrtoint ptr addrspace(1) %address.local989 to i64, !dbg !16
  %v413 = sub i64 %v412, %v411, !dbg !16
  %dma.buffer.offset1729 = trunc i64 %v413 to i32, !dbg !16
  %v414 = trunc i64 %v411 to i32, !dbg !16
  %v415 = insertelement <4 x i32> poison, i32 %v414, i64 0, !dbg !16
  %v416 = lshr i64 %v411, 32, !dbg !16
  %v417 = trunc i64 %v416 to i32, !dbg !16
  %v418 = insertelement <4 x i32> %v415, i32 %v417, i64 1, !dbg !16
  %v419 = insertelement <4 x i32> %v418, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1730 = insertelement <4 x i32> %v419, i32 131072, i64 3, !dbg !16
  %v420 = ptrtoint ptr addrspace(3) %v410 to i32, !dbg !16
  %dma.buffer.lds1731 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v420), !dbg !16
  %v421 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1731, i32 %dma.buffer.offset1729, <4 x i32> %dma.buffer.rsrc1730) #10, !dbg !16
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8192, !dbg !17
  %v422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !17
  %v423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v422, <4 x float> %v311) #9, !dbg !17
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12288, !dbg !17
  %v424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !17
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8192, !dbg !17
  %v425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !17
  %v426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v424, <4 x float> %v423) #9, !dbg !17
  %v427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v425, <4 x float> %v315) #9, !dbg !17
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12288, !dbg !17
  %v428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !17
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8320, !dbg !17
  %v429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !17
  %v430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v428, <4 x float> %v427) #9, !dbg !17
  %v431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v429, <4 x float> %v319) #9, !dbg !17
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12416, !dbg !17
  %v432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !17
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8320, !dbg !17
  %v433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !17
  %v434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v432, <4 x float> %v431) #9, !dbg !17
  %v435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v433, <4 x float> %v323) #9, !dbg !17
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12416, !dbg !17
  %v436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !17
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9216, !dbg !17
  %v437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !17
  %v438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v436, <4 x float> %v435) #9, !dbg !17
  %v439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v437, <4 x float> %v327) #9, !dbg !17
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13312, !dbg !17
  %v440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !17
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9216, !dbg !17
  %v441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !17
  %v442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v440, <4 x float> %v439) #9, !dbg !17
  %v443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v441, <4 x float> %v331) #9, !dbg !17
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13312, !dbg !17
  %v444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !17
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9344, !dbg !17
  %v445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !17
  %v446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v444, <4 x float> %v443) #9, !dbg !17
  %v447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v445, <4 x float> %v335) #9, !dbg !17
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13440, !dbg !17
  %v448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !17
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9344, !dbg !17
  %v449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !17
  %v450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v448, <4 x float> %v447) #9, !dbg !17
  %v451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v137, <4 x half> %v449, <4 x float> %v339) #9, !dbg !17
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13440, !dbg !17
  %v452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !17
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8192, !dbg !17
  %v453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !17
  %v454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v139, <4 x half> %v452, <4 x float> %v451) #9, !dbg !17
  %v455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v453, <4 x float> %v343) #9, !dbg !17
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12288, !dbg !17
  %v456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !17
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8192, !dbg !17
  %v457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !17
  %v458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v456, <4 x float> %v455) #9, !dbg !17
  %v459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v457, <4 x float> %v347) #9, !dbg !17
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12288, !dbg !17
  %v460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !17
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8320, !dbg !17
  %v461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !17
  %v462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v460, <4 x float> %v459) #9, !dbg !17
  %v463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v461, <4 x float> %v351) #9, !dbg !17
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12416, !dbg !17
  %v464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !17
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8320, !dbg !17
  %v465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !17
  %v466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v464, <4 x float> %v463) #9, !dbg !17
  %v467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v465, <4 x float> %v355) #9, !dbg !17
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12416, !dbg !17
  %v468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !17
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9216, !dbg !17
  %v469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !17
  %v470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v468, <4 x float> %v467) #9, !dbg !17
  %v471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v469, <4 x float> %v359) #9, !dbg !17
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13312, !dbg !17
  %v472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !17
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9216, !dbg !17
  %v473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !17
  %v474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v472, <4 x float> %v471) #9, !dbg !17
  %v475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v473, <4 x float> %v363) #9, !dbg !17
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13312, !dbg !17
  %v476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !17
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9344, !dbg !17
  %v477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !17
  %v478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v476, <4 x float> %v475) #9, !dbg !17
  %v479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v477, <4 x float> %v367) #9, !dbg !17
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13440, !dbg !17
  %v480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !17
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9344, !dbg !17
  %v481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !17
  %v482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v141, <4 x half> %v480, <4 x float> %v479) #9, !dbg !17
  %v483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v140, <4 x half> %v481, <4 x float> %v370) #9, !dbg !17
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13440, !dbg !17
  %v484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !17
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
  %v509 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local626), !dbg !16
  %v510 = zext i32 %v509 to i64, !dbg !16
  %v511 = inttoptr i64 %v510 to ptr addrspace(3), !dbg !16
  %address.local1006 = getelementptr [2 x i8], ptr addrspace(1) %v508, i64 %address.local605, !dbg !16
  %v512 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v513 = ptrtoint ptr addrspace(1) %address.local1006 to i64, !dbg !16
  %v514 = sub i64 %v513, %v512, !dbg !16
  %dma.buffer.offset1726 = trunc i64 %v514 to i32, !dbg !16
  %v515 = trunc i64 %v512 to i32, !dbg !16
  %v516 = insertelement <4 x i32> poison, i32 %v515, i64 0, !dbg !16
  %v517 = lshr i64 %v512, 32, !dbg !16
  %v518 = trunc i64 %v517 to i32, !dbg !16
  %v519 = insertelement <4 x i32> %v516, i32 %v518, i64 1, !dbg !16
  %v520 = insertelement <4 x i32> %v519, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1727 = insertelement <4 x i32> %v520, i32 131072, i64 3, !dbg !16
  %v521 = ptrtoint ptr addrspace(3) %v511 to i32, !dbg !16
  %dma.buffer.lds1728 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v521), !dbg !16
  %v522 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1728, i32 %dma.buffer.offset1726, <4 x i32> %dma.buffer.rsrc1727) #10, !dbg !16
  %v523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local630), !dbg !16
  %v524 = zext i32 %v523 to i64, !dbg !16
  %v525 = inttoptr i64 %v524 to ptr addrspace(3), !dbg !16
  %address.local1007 = getelementptr [2 x i8], ptr addrspace(1) %v508, i64 %address.local622, !dbg !16
  %v526 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %v527 = ptrtoint ptr addrspace(1) %address.local1007 to i64, !dbg !16
  %v528 = sub i64 %v527, %v526, !dbg !16
  %dma.buffer.offset1723 = trunc i64 %v528 to i32, !dbg !16
  %v529 = trunc i64 %v526 to i32, !dbg !16
  %v530 = insertelement <4 x i32> poison, i32 %v529, i64 0, !dbg !16
  %v531 = lshr i64 %v526, 32, !dbg !16
  %v532 = trunc i64 %v531 to i32, !dbg !16
  %v533 = insertelement <4 x i32> %v530, i32 %v532, i64 1, !dbg !16
  %v534 = insertelement <4 x i32> %v533, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1724 = insertelement <4 x i32> %v534, i32 131072, i64 3, !dbg !16
  %v535 = ptrtoint ptr addrspace(3) %v525 to i32, !dbg !16
  %dma.buffer.lds1725 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v535), !dbg !16
  %v536 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1725, i32 %dma.buffer.offset1723, <4 x i32> %dma.buffer.rsrc1724) #10, !dbg !16
  %v537 = load <4 x half>, ptr addrspace(3) %address.local650, align 8, !dbg !17
  %v538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v537, <4 x float> %v426) #9, !dbg !17
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4096, !dbg !17
  %v539 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !17
  %v540 = load <4 x half>, ptr addrspace(3) %address.local679, align 8, !dbg !17
  %v541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v539, <4 x float> %v538) #9, !dbg !17
  %v542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v540, <4 x float> %v430) #9, !dbg !17
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4096, !dbg !17
  %v543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !17
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local650, i64 128, !dbg !17
  %v544 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !17
  %v545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v543, <4 x float> %v542) #9, !dbg !17
  %v546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v544, <4 x float> %v434) #9, !dbg !17
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4224, !dbg !17
  %v547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !17
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %address.local679, i64 128, !dbg !17
  %v548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !17
  %v549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v547, <4 x float> %v546) #9, !dbg !17
  %v550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v548, <4 x float> %v438) #9, !dbg !17
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4224, !dbg !17
  %v551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !17
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1024, !dbg !17
  %v552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !17
  %v553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v551, <4 x float> %v550) #9, !dbg !17
  %v554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v552, <4 x float> %v442) #9, !dbg !17
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5120, !dbg !17
  %v555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !17
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1024, !dbg !17
  %v556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !17
  %v557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v555, <4 x float> %v554) #9, !dbg !17
  %v558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v556, <4 x float> %v446) #9, !dbg !17
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5120, !dbg !17
  %v559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !17
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1152, !dbg !17
  %v560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !17
  %v561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v559, <4 x float> %v558) #9, !dbg !17
  %v562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v560, <4 x float> %v450) #9, !dbg !17
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5248, !dbg !17
  %v563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !17
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1152, !dbg !17
  %v564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !17
  %v565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v563, <4 x float> %v562) #9, !dbg !17
  %v566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v175, <4 x half> %v564, <4 x float> %v454) #9, !dbg !17
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5248, !dbg !17
  %v567 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !17
  %v568 = load <4 x half>, ptr addrspace(3) %address.local650, align 8, !dbg !17
  %v569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v177, <4 x half> %v567, <4 x float> %v566) #9, !dbg !17
  %v570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v568, <4 x float> %v458) #9, !dbg !17
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4096, !dbg !17
  %v571 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !17
  %v572 = load <4 x half>, ptr addrspace(3) %address.local679, align 8, !dbg !17
  %v573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v571, <4 x float> %v570) #9, !dbg !17
  %v574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v572, <4 x float> %v462) #9, !dbg !17
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4096, !dbg !17
  %v575 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !17
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local650, i64 128, !dbg !17
  %v576 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !17
  %v577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v575, <4 x float> %v574) #9, !dbg !17
  %v578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v576, <4 x float> %v466) #9, !dbg !17
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4224, !dbg !17
  %v579 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !17
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %address.local679, i64 128, !dbg !17
  %v580 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !17
  %v581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v579, <4 x float> %v578) #9, !dbg !17
  %v582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v580, <4 x float> %v470) #9, !dbg !17
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4224, !dbg !17
  %v583 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !17
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1024, !dbg !17
  %v584 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !17
  %v585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v583, <4 x float> %v582) #9, !dbg !17
  %v586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v584, <4 x float> %v474) #9, !dbg !17
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5120, !dbg !17
  %v587 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !17
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1024, !dbg !17
  %v588 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !17
  %v589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v587, <4 x float> %v586) #9, !dbg !17
  %v590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v588, <4 x float> %v478) #9, !dbg !17
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5120, !dbg !17
  %v591 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !17
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1152, !dbg !17
  %v592 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !17
  %v593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v591, <4 x float> %v590) #9, !dbg !17
  %v594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v592, <4 x float> %v482) #9, !dbg !17
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5248, !dbg !17
  %v595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !17
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1152, !dbg !17
  %v596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1668, align 8, !dbg !17
  %v597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v179, <4 x half> %v595, <4 x float> %v594) #9, !dbg !17
  %v598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v178, <4 x half> %v596, <4 x float> %v485) #9, !dbg !17
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5248, !dbg !17
  %v599 = load <4 x half>, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !17
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
  %v606 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local595), !dbg !18
  %v607 = zext i32 %v606 to i64, !dbg !18
  %v608 = inttoptr i64 %v607 to ptr addrspace(3), !dbg !18
  %address.local1008 = shl nuw nsw i64 %address.local590, 5, !dbg !18
  %address.local1009 = and i64 %address.local1008, 8064, !dbg !18
  %address.local1010 = shl nuw nsw i64 %address.local590, 3, !dbg !18
  %address.local1011 = and i64 %address.local1010, 24, !dbg !18
  %address.local1012 = or disjoint i64 %address.local1009, %address.local1011, !dbg !18
  %address.local1013 = getelementptr [2 x i8], ptr addrspace(1) %v605, i64 %address.local1012, !dbg !18
  %v609 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v610 = ptrtoint ptr addrspace(1) %address.local1013 to i64, !dbg !18
  %v611 = sub i64 %v610, %v609, !dbg !18
  %dma.buffer.offset1720 = trunc i64 %v611 to i32, !dbg !18
  %v612 = trunc i64 %v609 to i32, !dbg !18
  %v613 = insertelement <4 x i32> poison, i32 %v612, i64 0, !dbg !18
  %v614 = lshr i64 %v609, 32, !dbg !18
  %v615 = trunc i64 %v614 to i32, !dbg !18
  %v616 = insertelement <4 x i32> %v613, i32 %v615, i64 1, !dbg !18
  %v617 = insertelement <4 x i32> %v616, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1721 = insertelement <4 x i32> %v617, i32 131072, i64 3, !dbg !18
  %v618 = ptrtoint ptr addrspace(3) %v608 to i32, !dbg !18
  %dma.buffer.lds1722 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v618), !dbg !18
  %v619 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1722, i32 %dma.buffer.offset1720, <4 x i32> %dma.buffer.rsrc1721) #10, !dbg !18
  %v620 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local612), !dbg !18
  %v621 = zext i32 %v620 to i64, !dbg !18
  %v622 = inttoptr i64 %v621 to ptr addrspace(3), !dbg !18
  %address.local1014 = shl nuw nsw i64 %address.local607, 5, !dbg !18
  %address.local1015 = and i64 %address.local1014, 16256, !dbg !18
  %address.local1016 = shl nuw nsw i64 %address.local590, 3, !dbg !18
  %address.local1017 = and i64 %address.local1016, 24, !dbg !18
  %address.local1018 = or disjoint i64 %address.local1015, %address.local1017, !dbg !18
  %address.local1019 = getelementptr [2 x i8], ptr addrspace(1) %v605, i64 %address.local1018, !dbg !18
  %v623 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v624 = ptrtoint ptr addrspace(1) %address.local1019 to i64, !dbg !18
  %v625 = sub i64 %v624, %v623, !dbg !18
  %dma.buffer.offset1717 = trunc i64 %v625 to i32, !dbg !18
  %v626 = trunc i64 %v623 to i32, !dbg !18
  %v627 = insertelement <4 x i32> poison, i32 %v626, i64 0, !dbg !18
  %v628 = lshr i64 %v623, 32, !dbg !18
  %v629 = trunc i64 %v628 to i32, !dbg !18
  %v630 = insertelement <4 x i32> %v627, i32 %v629, i64 1, !dbg !18
  %v631 = insertelement <4 x i32> %v630, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1718 = insertelement <4 x i32> %v631, i32 131072, i64 3, !dbg !18
  %v632 = ptrtoint ptr addrspace(3) %v622 to i32, !dbg !18
  %dma.buffer.lds1719 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v632), !dbg !18
  %v633 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1719, i32 %dma.buffer.offset1717, <4 x i32> %dma.buffer.rsrc1718) #10, !dbg !18
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8192, !dbg !17
  %v634 = load <4 x half>, ptr addrspace(3) %lds.const.gep1670, align 8, !dbg !17
  %v635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v634, <4 x float> %v541) #9, !dbg !17
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12288, !dbg !17
  %v636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1671, align 8, !dbg !17
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8192, !dbg !17
  %v637 = load <4 x half>, ptr addrspace(3) %lds.const.gep1672, align 8, !dbg !17
  %v638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v636, <4 x float> %v635) #9, !dbg !17
  %v639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v637, <4 x float> %v545) #9, !dbg !17
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12288, !dbg !17
  %v640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !17
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8320, !dbg !17
  %v641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1674, align 8, !dbg !17
  %v642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v640, <4 x float> %v639) #9, !dbg !17
  %v643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v641, <4 x float> %v549) #9, !dbg !17
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12416, !dbg !17
  %v644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1675, align 8, !dbg !17
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8320, !dbg !17
  %v645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1676, align 8, !dbg !17
  %v646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v644, <4 x float> %v643) #9, !dbg !17
  %v647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v645, <4 x float> %v553) #9, !dbg !17
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12416, !dbg !17
  %v648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !17
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9216, !dbg !17
  %v649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1678, align 8, !dbg !17
  %v650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v648, <4 x float> %v647) #9, !dbg !17
  %v651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v649, <4 x float> %v557) #9, !dbg !17
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13312, !dbg !17
  %v652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1679, align 8, !dbg !17
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9216, !dbg !17
  %v653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1680, align 8, !dbg !17
  %v654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v652, <4 x float> %v651) #9, !dbg !17
  %v655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v653, <4 x float> %v561) #9, !dbg !17
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13312, !dbg !17
  %v656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !17
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9344, !dbg !17
  %v657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1682, align 8, !dbg !17
  %v658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v656, <4 x float> %v655) #9, !dbg !17
  %v659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v657, <4 x float> %v565) #9, !dbg !17
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13440, !dbg !17
  %v660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !17
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9344, !dbg !17
  %v661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1684, align 8, !dbg !17
  %v662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v660, <4 x float> %v659) #9, !dbg !17
  %v663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v183, <4 x half> %v661, <4 x float> %v569) #9, !dbg !17
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13440, !dbg !17
  %v664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1685, align 8, !dbg !17
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8192, !dbg !17
  %v665 = load <4 x half>, ptr addrspace(3) %lds.const.gep1686, align 8, !dbg !17
  %v666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v185, <4 x half> %v664, <4 x float> %v663) #9, !dbg !17
  %v667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v665, <4 x float> %v573) #9, !dbg !17
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12288, !dbg !17
  %v668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !17
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8192, !dbg !17
  %v669 = load <4 x half>, ptr addrspace(3) %lds.const.gep1688, align 8, !dbg !17
  %v670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v668, <4 x float> %v667) #9, !dbg !17
  %v671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v669, <4 x float> %v577) #9, !dbg !17
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12288, !dbg !17
  %v672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1689, align 8, !dbg !17
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8320, !dbg !17
  %v673 = load <4 x half>, ptr addrspace(3) %lds.const.gep1690, align 8, !dbg !17
  %v674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v672, <4 x float> %v671) #9, !dbg !17
  %v675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v673, <4 x float> %v581) #9, !dbg !17
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12416, !dbg !17
  %v676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !17
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8320, !dbg !17
  %v677 = load <4 x half>, ptr addrspace(3) %lds.const.gep1692, align 8, !dbg !17
  %v678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v676, <4 x float> %v675) #9, !dbg !17
  %v679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v677, <4 x float> %v585) #9, !dbg !17
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12416, !dbg !17
  %v680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1693, align 8, !dbg !17
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9216, !dbg !17
  %v681 = load <4 x half>, ptr addrspace(3) %lds.const.gep1694, align 8, !dbg !17
  %v682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v680, <4 x float> %v679) #9, !dbg !17
  %v683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v681, <4 x float> %v589) #9, !dbg !17
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13312, !dbg !17
  %v684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !17
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9216, !dbg !17
  %v685 = load <4 x half>, ptr addrspace(3) %lds.const.gep1696, align 8, !dbg !17
  %v686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v684, <4 x float> %v683) #9, !dbg !17
  %v687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v685, <4 x float> %v593) #9, !dbg !17
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13312, !dbg !17
  %v688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1697, align 8, !dbg !17
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9344, !dbg !17
  %v689 = load <4 x half>, ptr addrspace(3) %lds.const.gep1698, align 8, !dbg !17
  %v690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v688, <4 x float> %v687) #9, !dbg !17
  %v691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v689, <4 x float> %v597) #9, !dbg !17
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13440, !dbg !17
  %v692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !17
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9344, !dbg !17
  %v693 = load <4 x half>, ptr addrspace(3) %lds.const.gep1700, align 8, !dbg !17
  %v694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v692, <4 x float> %v691) #9, !dbg !17
  %v695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v186, <4 x half> %v693, <4 x float> %v600) #9, !dbg !17
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13440, !dbg !17
  %v696 = load <4 x half>, ptr addrspace(3) %lds.const.gep1701, align 8, !dbg !17
  %v697 = or disjoint i64 %v214, 127, !dbg !13
  %v698 = icmp uge i64 %v64, %v697, !dbg !13
  %v699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v187, <4 x half> %v696, <4 x float> %v695) #9, !dbg !17
  %v700 = and i1 %v190, %v698, !dbg !13
  br i1 %v700, label %v701, label %v702, !dbg !13

v701:                                              ; preds = %v213
  br label %v828, !dbg !13

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
  %compare.high1313 = trunc i64 %v712 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1313, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1313, !dbg !24
  %compare.low.base = trunc i64 %v710 to i32, !dbg !24
  %compare.low.base1314 = trunc i64 %v709 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1314, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %v713 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1315 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result1316 = select i1 %compare.high.equal, i1 %compare.low1315, i1 %compare.high.order, !dbg !24
  %v714 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 2, !dbg !24
  %compare.low1319 = icmp ugt i32 %compare.low.base, %compare.local1318, !dbg !24
  %compare.result1320 = select i1 %compare.high.equal, i1 %compare.low1319, i1 %compare.high.order, !dbg !24
  %v715 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 3, !dbg !24
  %compare.low1323 = icmp ugt i32 %compare.low.base, %compare.local1322, !dbg !24
  %compare.result1324 = select i1 %compare.high.equal, i1 %compare.low1323, i1 %compare.high.order, !dbg !24
  %v716 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 4, !dbg !24
  %compare.low1327 = icmp ugt i32 %compare.low.base, %compare.local1326, !dbg !24
  %compare.result1328 = select i1 %compare.high.equal, i1 %compare.low1327, i1 %compare.high.order, !dbg !24
  %v717 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 5, !dbg !24
  %compare.low1331 = icmp ugt i32 %compare.low.base, %compare.local1330, !dbg !24
  %compare.result1332 = select i1 %compare.high.equal, i1 %compare.low1331, i1 %compare.high.order, !dbg !24
  %v718 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 6, !dbg !24
  %compare.low1335 = icmp ugt i32 %compare.low.base, %compare.local1334, !dbg !24
  %compare.result1336 = select i1 %compare.high.equal, i1 %compare.low1335, i1 %compare.high.order, !dbg !24
  %v719 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 7, !dbg !24
  %compare.low1339 = icmp ugt i32 %compare.low.base, %compare.local1338, !dbg !24
  %compare.result1340 = select i1 %compare.high.equal, i1 %compare.low1339, i1 %compare.high.order, !dbg !24
  %v720 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 8, !dbg !24
  %compare.low1343 = icmp ugt i32 %compare.low.base, %compare.local1342, !dbg !24
  %compare.result1344 = select i1 %compare.high.equal, i1 %compare.low1343, i1 %compare.high.order, !dbg !24
  %v721 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 9, !dbg !24
  %compare.low1347 = icmp ugt i32 %compare.low.base, %compare.local1346, !dbg !24
  %compare.result1348 = select i1 %compare.high.equal, i1 %compare.low1347, i1 %compare.high.order, !dbg !24
  %v722 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 10, !dbg !24
  %compare.low1351 = icmp ugt i32 %compare.low.base, %compare.local1350, !dbg !24
  %compare.result1352 = select i1 %compare.high.equal, i1 %compare.low1351, i1 %compare.high.order, !dbg !24
  %v723 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 11, !dbg !24
  %compare.low1355 = icmp ugt i32 %compare.low.base, %compare.local1354, !dbg !24
  %compare.result1356 = select i1 %compare.high.equal, i1 %compare.low1355, i1 %compare.high.order, !dbg !24
  %v724 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 12, !dbg !24
  %compare.low1359 = icmp ugt i32 %compare.low.base, %compare.local1358, !dbg !24
  %compare.result1360 = select i1 %compare.high.equal, i1 %compare.low1359, i1 %compare.high.order, !dbg !24
  %v725 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 13, !dbg !24
  %compare.low1363 = icmp ugt i32 %compare.low.base, %compare.local1362, !dbg !24
  %compare.result1364 = select i1 %compare.high.equal, i1 %compare.low1363, i1 %compare.high.order, !dbg !24
  %v726 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 14, !dbg !24
  %compare.low1367 = icmp ugt i32 %compare.low.base, %compare.local1366, !dbg !24
  %compare.result1368 = select i1 %compare.high.equal, i1 %compare.low1367, i1 %compare.high.order, !dbg !24
  %v727 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 15, !dbg !24
  %compare.low1371 = icmp ugt i32 %compare.low.base, %compare.local1370, !dbg !24
  %compare.result1372 = select i1 %compare.high.equal, i1 %compare.low1371, i1 %compare.high.order, !dbg !24
  %v728 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 64, !dbg !24
  %compare.low1375 = icmp ugt i32 %compare.low.base, %compare.local1374, !dbg !24
  %compare.result1376 = select i1 %compare.high.equal, i1 %compare.low1375, i1 %compare.high.order, !dbg !24
  %v729 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 65, !dbg !24
  %compare.low1379 = icmp ugt i32 %compare.low.base, %compare.local1378, !dbg !24
  %compare.result1380 = select i1 %compare.high.equal, i1 %compare.low1379, i1 %compare.high.order, !dbg !24
  %v730 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 66, !dbg !24
  %compare.low1383 = icmp ugt i32 %compare.low.base, %compare.local1382, !dbg !24
  %compare.result1384 = select i1 %compare.high.equal, i1 %compare.low1383, i1 %compare.high.order, !dbg !24
  %v731 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 67, !dbg !24
  %compare.low1387 = icmp ugt i32 %compare.low.base, %compare.local1386, !dbg !24
  %compare.result1388 = select i1 %compare.high.equal, i1 %compare.low1387, i1 %compare.high.order, !dbg !24
  %v732 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 68, !dbg !24
  %compare.low1391 = icmp ugt i32 %compare.low.base, %compare.local1390, !dbg !24
  %compare.result1392 = select i1 %compare.high.equal, i1 %compare.low1391, i1 %compare.high.order, !dbg !24
  %v733 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1393 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1394 = or i32 %compare.local.base1393, 69, !dbg !24
  %compare.low1395 = icmp ugt i32 %compare.low.base, %compare.local1394, !dbg !24
  %compare.result1396 = select i1 %compare.high.equal, i1 %compare.low1395, i1 %compare.high.order, !dbg !24
  %v734 = select i1 %compare.result1396, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1398 = or i32 %compare.local.base1397, 70, !dbg !24
  %compare.low1399 = icmp ugt i32 %compare.low.base, %compare.local1398, !dbg !24
  %compare.result1400 = select i1 %compare.high.equal, i1 %compare.low1399, i1 %compare.high.order, !dbg !24
  %v735 = select i1 %compare.result1400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1402 = or i32 %compare.local.base1401, 71, !dbg !24
  %compare.low1403 = icmp ugt i32 %compare.low.base, %compare.local1402, !dbg !24
  %compare.result1404 = select i1 %compare.high.equal, i1 %compare.low1403, i1 %compare.high.order, !dbg !24
  %v736 = select i1 %compare.result1404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1406 = or i32 %compare.local.base1405, 72, !dbg !24
  %compare.low1407 = icmp ugt i32 %compare.low.base, %compare.local1406, !dbg !24
  %compare.result1408 = select i1 %compare.high.equal, i1 %compare.low1407, i1 %compare.high.order, !dbg !24
  %v737 = select i1 %compare.result1408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1410 = or i32 %compare.local.base1409, 73, !dbg !24
  %compare.low1411 = icmp ugt i32 %compare.low.base, %compare.local1410, !dbg !24
  %compare.result1412 = select i1 %compare.high.equal, i1 %compare.low1411, i1 %compare.high.order, !dbg !24
  %v738 = select i1 %compare.result1412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1413 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1414 = or i32 %compare.local.base1413, 74, !dbg !24
  %compare.low1415 = icmp ugt i32 %compare.low.base, %compare.local1414, !dbg !24
  %compare.result1416 = select i1 %compare.high.equal, i1 %compare.low1415, i1 %compare.high.order, !dbg !24
  %v739 = select i1 %compare.result1416, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1417 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1418 = or i32 %compare.local.base1417, 75, !dbg !24
  %compare.low1419 = icmp ugt i32 %compare.low.base, %compare.local1418, !dbg !24
  %compare.result1420 = select i1 %compare.high.equal, i1 %compare.low1419, i1 %compare.high.order, !dbg !24
  %v740 = select i1 %compare.result1420, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1421 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1422 = or i32 %compare.local.base1421, 76, !dbg !24
  %compare.low1423 = icmp ugt i32 %compare.low.base, %compare.local1422, !dbg !24
  %compare.result1424 = select i1 %compare.high.equal, i1 %compare.low1423, i1 %compare.high.order, !dbg !24
  %v741 = select i1 %compare.result1424, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1425 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1426 = or i32 %compare.local.base1425, 77, !dbg !24
  %compare.low1427 = icmp ugt i32 %compare.low.base, %compare.local1426, !dbg !24
  %compare.result1428 = select i1 %compare.high.equal, i1 %compare.low1427, i1 %compare.high.order, !dbg !24
  %v742 = select i1 %compare.result1428, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1429 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1430 = or i32 %compare.local.base1429, 78, !dbg !24
  %compare.low1431 = icmp ugt i32 %compare.low.base, %compare.local1430, !dbg !24
  %compare.result1432 = select i1 %compare.high.equal, i1 %compare.low1431, i1 %compare.high.order, !dbg !24
  %v743 = select i1 %compare.result1432, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1433 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1434 = or i32 %compare.local.base1433, 79, !dbg !24
  %compare.low1435 = icmp ugt i32 %compare.low.base, %compare.local1434, !dbg !24
  %compare.result1436 = select i1 %compare.high.equal, i1 %compare.low1435, i1 %compare.high.order, !dbg !24
  %v744 = select i1 %compare.result1436, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %v745 = add i64 %v707, 65, !dbg !23
  %v746 = lshr i64 %v745, 32, !dbg !24
  %compare.high1437 = trunc i64 %v746 to i32, !dbg !24
  %v747 = lshr i64 %v709, 32, !dbg !24
  %compare.high1438 = trunc i64 %v747 to i32, !dbg !24
  %compare.high.equal1439 = icmp eq i32 %compare.high1437, %compare.high1438, !dbg !24
  %compare.high.order1440 = icmp ugt i32 %compare.high1437, %compare.high1438, !dbg !24
  %compare.low.base1441 = trunc i64 %v745 to i32, !dbg !24
  %compare.low.base1442 = trunc i64 %v709 to i32, !dbg !24
  %compare.low1443 = icmp ugt i32 %compare.low.base1441, %compare.low.base1442, !dbg !24
  %compare.result1444 = select i1 %compare.high.equal1439, i1 %compare.low1443, i1 %compare.high.order1440, !dbg !24
  %v748 = select i1 %compare.result1444, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1445 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1446 = or i32 %compare.local.base1445, 1, !dbg !24
  %compare.low1447 = icmp ugt i32 %compare.low.base1441, %compare.local1446, !dbg !24
  %compare.result1448 = select i1 %compare.high.equal1439, i1 %compare.low1447, i1 %compare.high.order1440, !dbg !24
  %v749 = select i1 %compare.result1448, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1449 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1450 = or i32 %compare.local.base1449, 2, !dbg !24
  %compare.low1451 = icmp ugt i32 %compare.low.base1441, %compare.local1450, !dbg !24
  %compare.result1452 = select i1 %compare.high.equal1439, i1 %compare.low1451, i1 %compare.high.order1440, !dbg !24
  %v750 = select i1 %compare.result1452, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1453 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1454 = or i32 %compare.local.base1453, 3, !dbg !24
  %compare.low1455 = icmp ugt i32 %compare.low.base1441, %compare.local1454, !dbg !24
  %compare.result1456 = select i1 %compare.high.equal1439, i1 %compare.low1455, i1 %compare.high.order1440, !dbg !24
  %v751 = select i1 %compare.result1456, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1457 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1458 = or i32 %compare.local.base1457, 4, !dbg !24
  %compare.low1459 = icmp ugt i32 %compare.low.base1441, %compare.local1458, !dbg !24
  %compare.result1460 = select i1 %compare.high.equal1439, i1 %compare.low1459, i1 %compare.high.order1440, !dbg !24
  %v752 = select i1 %compare.result1460, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1461 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1462 = or i32 %compare.local.base1461, 5, !dbg !24
  %compare.low1463 = icmp ugt i32 %compare.low.base1441, %compare.local1462, !dbg !24
  %compare.result1464 = select i1 %compare.high.equal1439, i1 %compare.low1463, i1 %compare.high.order1440, !dbg !24
  %v753 = select i1 %compare.result1464, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1465 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1466 = or i32 %compare.local.base1465, 6, !dbg !24
  %compare.low1467 = icmp ugt i32 %compare.low.base1441, %compare.local1466, !dbg !24
  %compare.result1468 = select i1 %compare.high.equal1439, i1 %compare.low1467, i1 %compare.high.order1440, !dbg !24
  %v754 = select i1 %compare.result1468, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1469 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1470 = or i32 %compare.local.base1469, 7, !dbg !24
  %compare.low1471 = icmp ugt i32 %compare.low.base1441, %compare.local1470, !dbg !24
  %compare.result1472 = select i1 %compare.high.equal1439, i1 %compare.low1471, i1 %compare.high.order1440, !dbg !24
  %v755 = select i1 %compare.result1472, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1473 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1474 = or i32 %compare.local.base1473, 8, !dbg !24
  %compare.low1475 = icmp ugt i32 %compare.low.base1441, %compare.local1474, !dbg !24
  %compare.result1476 = select i1 %compare.high.equal1439, i1 %compare.low1475, i1 %compare.high.order1440, !dbg !24
  %v756 = select i1 %compare.result1476, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1477 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1478 = or i32 %compare.local.base1477, 9, !dbg !24
  %compare.low1479 = icmp ugt i32 %compare.low.base1441, %compare.local1478, !dbg !24
  %compare.result1480 = select i1 %compare.high.equal1439, i1 %compare.low1479, i1 %compare.high.order1440, !dbg !24
  %v757 = select i1 %compare.result1480, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1481 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1482 = or i32 %compare.local.base1481, 10, !dbg !24
  %compare.low1483 = icmp ugt i32 %compare.low.base1441, %compare.local1482, !dbg !24
  %compare.result1484 = select i1 %compare.high.equal1439, i1 %compare.low1483, i1 %compare.high.order1440, !dbg !24
  %v758 = select i1 %compare.result1484, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1485 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1486 = or i32 %compare.local.base1485, 11, !dbg !24
  %compare.low1487 = icmp ugt i32 %compare.low.base1441, %compare.local1486, !dbg !24
  %compare.result1488 = select i1 %compare.high.equal1439, i1 %compare.low1487, i1 %compare.high.order1440, !dbg !24
  %v759 = select i1 %compare.result1488, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1489 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1490 = or i32 %compare.local.base1489, 12, !dbg !24
  %compare.low1491 = icmp ugt i32 %compare.low.base1441, %compare.local1490, !dbg !24
  %compare.result1492 = select i1 %compare.high.equal1439, i1 %compare.low1491, i1 %compare.high.order1440, !dbg !24
  %v760 = select i1 %compare.result1492, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1493 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1494 = or i32 %compare.local.base1493, 13, !dbg !24
  %compare.low1495 = icmp ugt i32 %compare.low.base1441, %compare.local1494, !dbg !24
  %compare.result1496 = select i1 %compare.high.equal1439, i1 %compare.low1495, i1 %compare.high.order1440, !dbg !24
  %v761 = select i1 %compare.result1496, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1497 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1498 = or i32 %compare.local.base1497, 14, !dbg !24
  %compare.low1499 = icmp ugt i32 %compare.low.base1441, %compare.local1498, !dbg !24
  %compare.result1500 = select i1 %compare.high.equal1439, i1 %compare.low1499, i1 %compare.high.order1440, !dbg !24
  %v762 = select i1 %compare.result1500, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1501 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1502 = or i32 %compare.local.base1501, 15, !dbg !24
  %compare.low1503 = icmp ugt i32 %compare.low.base1441, %compare.local1502, !dbg !24
  %compare.result1504 = select i1 %compare.high.equal1439, i1 %compare.low1503, i1 %compare.high.order1440, !dbg !24
  %v763 = select i1 %compare.result1504, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1505 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1506 = or i32 %compare.local.base1505, 64, !dbg !24
  %compare.low1507 = icmp ugt i32 %compare.low.base1441, %compare.local1506, !dbg !24
  %compare.result1508 = select i1 %compare.high.equal1439, i1 %compare.low1507, i1 %compare.high.order1440, !dbg !24
  %v764 = select i1 %compare.result1508, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1509 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1510 = or i32 %compare.local.base1509, 65, !dbg !24
  %compare.low1511 = icmp ugt i32 %compare.low.base1441, %compare.local1510, !dbg !24
  %compare.result1512 = select i1 %compare.high.equal1439, i1 %compare.low1511, i1 %compare.high.order1440, !dbg !24
  %v765 = select i1 %compare.result1512, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1513 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1514 = or i32 %compare.local.base1513, 66, !dbg !24
  %compare.low1515 = icmp ugt i32 %compare.low.base1441, %compare.local1514, !dbg !24
  %compare.result1516 = select i1 %compare.high.equal1439, i1 %compare.low1515, i1 %compare.high.order1440, !dbg !24
  %v766 = select i1 %compare.result1516, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1517 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1518 = or i32 %compare.local.base1517, 67, !dbg !24
  %compare.low1519 = icmp ugt i32 %compare.low.base1441, %compare.local1518, !dbg !24
  %compare.result1520 = select i1 %compare.high.equal1439, i1 %compare.low1519, i1 %compare.high.order1440, !dbg !24
  %v767 = select i1 %compare.result1520, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1521 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1522 = or i32 %compare.local.base1521, 68, !dbg !24
  %compare.low1523 = icmp ugt i32 %compare.low.base1441, %compare.local1522, !dbg !24
  %compare.result1524 = select i1 %compare.high.equal1439, i1 %compare.low1523, i1 %compare.high.order1440, !dbg !24
  %v768 = select i1 %compare.result1524, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1525 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1526 = or i32 %compare.local.base1525, 69, !dbg !24
  %compare.low1527 = icmp ugt i32 %compare.low.base1441, %compare.local1526, !dbg !24
  %compare.result1528 = select i1 %compare.high.equal1439, i1 %compare.low1527, i1 %compare.high.order1440, !dbg !24
  %v769 = select i1 %compare.result1528, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1529 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1530 = or i32 %compare.local.base1529, 70, !dbg !24
  %compare.low1531 = icmp ugt i32 %compare.low.base1441, %compare.local1530, !dbg !24
  %compare.result1532 = select i1 %compare.high.equal1439, i1 %compare.low1531, i1 %compare.high.order1440, !dbg !24
  %v770 = select i1 %compare.result1532, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1533 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1534 = or i32 %compare.local.base1533, 71, !dbg !24
  %compare.low1535 = icmp ugt i32 %compare.low.base1441, %compare.local1534, !dbg !24
  %compare.result1536 = select i1 %compare.high.equal1439, i1 %compare.low1535, i1 %compare.high.order1440, !dbg !24
  %v771 = select i1 %compare.result1536, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1537 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1538 = or i32 %compare.local.base1537, 72, !dbg !24
  %compare.low1539 = icmp ugt i32 %compare.low.base1441, %compare.local1538, !dbg !24
  %compare.result1540 = select i1 %compare.high.equal1439, i1 %compare.low1539, i1 %compare.high.order1440, !dbg !24
  %v772 = select i1 %compare.result1540, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1541 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1542 = or i32 %compare.local.base1541, 73, !dbg !24
  %compare.low1543 = icmp ugt i32 %compare.low.base1441, %compare.local1542, !dbg !24
  %compare.result1544 = select i1 %compare.high.equal1439, i1 %compare.low1543, i1 %compare.high.order1440, !dbg !24
  %v773 = select i1 %compare.result1544, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1545 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1546 = or i32 %compare.local.base1545, 74, !dbg !24
  %compare.low1547 = icmp ugt i32 %compare.low.base1441, %compare.local1546, !dbg !24
  %compare.result1548 = select i1 %compare.high.equal1439, i1 %compare.low1547, i1 %compare.high.order1440, !dbg !24
  %v774 = select i1 %compare.result1548, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1549 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1550 = or i32 %compare.local.base1549, 75, !dbg !24
  %compare.low1551 = icmp ugt i32 %compare.low.base1441, %compare.local1550, !dbg !24
  %compare.result1552 = select i1 %compare.high.equal1439, i1 %compare.low1551, i1 %compare.high.order1440, !dbg !24
  %v775 = select i1 %compare.result1552, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1553 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1554 = or i32 %compare.local.base1553, 76, !dbg !24
  %compare.low1555 = icmp ugt i32 %compare.low.base1441, %compare.local1554, !dbg !24
  %compare.result1556 = select i1 %compare.high.equal1439, i1 %compare.low1555, i1 %compare.high.order1440, !dbg !24
  %v776 = select i1 %compare.result1556, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1557 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1558 = or i32 %compare.local.base1557, 77, !dbg !24
  %compare.low1559 = icmp ugt i32 %compare.low.base1441, %compare.local1558, !dbg !24
  %compare.result1560 = select i1 %compare.high.equal1439, i1 %compare.low1559, i1 %compare.high.order1440, !dbg !24
  %v777 = select i1 %compare.result1560, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1561 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1562 = or i32 %compare.local.base1561, 78, !dbg !24
  %compare.low1563 = icmp ugt i32 %compare.low.base1441, %compare.local1562, !dbg !24
  %compare.result1564 = select i1 %compare.high.equal1439, i1 %compare.low1563, i1 %compare.high.order1440, !dbg !24
  %v778 = select i1 %compare.result1564, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1565 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1566 = or i32 %compare.local.base1565, 79, !dbg !24
  %compare.low1567 = icmp ugt i32 %compare.low.base1441, %compare.local1566, !dbg !24
  %compare.result1568 = select i1 %compare.high.equal1439, i1 %compare.low1567, i1 %compare.high.order1440, !dbg !24
  %v779 = select i1 %compare.result1568, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %v780 = insertelement <4 x float> poison, float %v713, i64 0, !dbg !13
  %v781 = insertelement <4 x float> %v780, float %v714, i64 1, !dbg !13
  %v782 = insertelement <4 x float> %v781, float %v715, i64 2, !dbg !13
  %.masked126 = insertelement <4 x float> %v782, float %v716, i64 3, !dbg !13
  %v783 = insertelement <4 x float> poison, float %v717, i64 0, !dbg !13
  %v784 = insertelement <4 x float> %v783, float %v718, i64 1, !dbg !13
  %v785 = insertelement <4 x float> %v784, float %v719, i64 2, !dbg !13
  %.masked128 = insertelement <4 x float> %v785, float %v720, i64 3, !dbg !13
  %v786 = insertelement <4 x float> poison, float %v721, i64 0, !dbg !13
  %v787 = insertelement <4 x float> %v786, float %v722, i64 1, !dbg !13
  %v788 = insertelement <4 x float> %v787, float %v723, i64 2, !dbg !13
  %.masked130 = insertelement <4 x float> %v788, float %v724, i64 3, !dbg !13
  %v789 = insertelement <4 x float> poison, float %v725, i64 0, !dbg !13
  %v790 = insertelement <4 x float> %v789, float %v726, i64 1, !dbg !13
  %v791 = insertelement <4 x float> %v790, float %v727, i64 2, !dbg !13
  %.masked132 = insertelement <4 x float> %v791, float %v728, i64 3, !dbg !13
  %v792 = insertelement <4 x float> poison, float %v729, i64 0, !dbg !13
  %v793 = insertelement <4 x float> %v792, float %v730, i64 1, !dbg !13
  %v794 = insertelement <4 x float> %v793, float %v731, i64 2, !dbg !13
  %.masked134 = insertelement <4 x float> %v794, float %v732, i64 3, !dbg !13
  %v795 = insertelement <4 x float> poison, float %v733, i64 0, !dbg !13
  %v796 = insertelement <4 x float> %v795, float %v734, i64 1, !dbg !13
  %v797 = insertelement <4 x float> %v796, float %v735, i64 2, !dbg !13
  %.masked136 = insertelement <4 x float> %v797, float %v736, i64 3, !dbg !13
  %v798 = insertelement <4 x float> poison, float %v737, i64 0, !dbg !13
  %v799 = insertelement <4 x float> %v798, float %v738, i64 1, !dbg !13
  %v800 = insertelement <4 x float> %v799, float %v739, i64 2, !dbg !13
  %.masked138 = insertelement <4 x float> %v800, float %v740, i64 3, !dbg !13
  %v801 = insertelement <4 x float> poison, float %v741, i64 0, !dbg !13
  %v802 = insertelement <4 x float> %v801, float %v742, i64 1, !dbg !13
  %v803 = insertelement <4 x float> %v802, float %v743, i64 2, !dbg !13
  %.masked140 = insertelement <4 x float> %v803, float %v744, i64 3, !dbg !13
  %v804 = insertelement <4 x float> poison, float %v748, i64 0, !dbg !13
  %v805 = insertelement <4 x float> %v804, float %v749, i64 1, !dbg !13
  %v806 = insertelement <4 x float> %v805, float %v750, i64 2, !dbg !13
  %.masked143 = insertelement <4 x float> %v806, float %v751, i64 3, !dbg !13
  %v807 = insertelement <4 x float> poison, float %v752, i64 0, !dbg !13
  %v808 = insertelement <4 x float> %v807, float %v753, i64 1, !dbg !13
  %v809 = insertelement <4 x float> %v808, float %v754, i64 2, !dbg !13
  %.masked145 = insertelement <4 x float> %v809, float %v755, i64 3, !dbg !13
  %v810 = insertelement <4 x float> poison, float %v756, i64 0, !dbg !13
  %v811 = insertelement <4 x float> %v810, float %v757, i64 1, !dbg !13
  %v812 = insertelement <4 x float> %v811, float %v758, i64 2, !dbg !13
  %.masked147 = insertelement <4 x float> %v812, float %v759, i64 3, !dbg !13
  %v813 = insertelement <4 x float> poison, float %v760, i64 0, !dbg !13
  %v814 = insertelement <4 x float> %v813, float %v761, i64 1, !dbg !13
  %v815 = insertelement <4 x float> %v814, float %v762, i64 2, !dbg !13
  %.masked149 = insertelement <4 x float> %v815, float %v763, i64 3, !dbg !13
  %v816 = insertelement <4 x float> poison, float %v764, i64 0, !dbg !13
  %v817 = insertelement <4 x float> %v816, float %v765, i64 1, !dbg !13
  %v818 = insertelement <4 x float> %v817, float %v766, i64 2, !dbg !13
  %.masked151 = insertelement <4 x float> %v818, float %v767, i64 3, !dbg !13
  %v819 = insertelement <4 x float> poison, float %v768, i64 0, !dbg !13
  %v820 = insertelement <4 x float> %v819, float %v769, i64 1, !dbg !13
  %v821 = insertelement <4 x float> %v820, float %v770, i64 2, !dbg !13
  %.masked153 = insertelement <4 x float> %v821, float %v771, i64 3, !dbg !13
  %v822 = insertelement <4 x float> poison, float %v772, i64 0, !dbg !13
  %v823 = insertelement <4 x float> %v822, float %v773, i64 1, !dbg !13
  %v824 = insertelement <4 x float> %v823, float %v774, i64 2, !dbg !13
  %.masked155 = insertelement <4 x float> %v824, float %v775, i64 3, !dbg !13
  %v825 = insertelement <4 x float> poison, float %v776, i64 0, !dbg !13
  %v826 = insertelement <4 x float> %v825, float %v777, i64 1, !dbg !13
  %v827 = insertelement <4 x float> %v826, float %v778, i64 2, !dbg !13
  %.masked157 = insertelement <4 x float> %v827, float %v779, i64 3, !dbg !13
  %.masked127 = fadd <4 x float> %v638, %.masked126, !dbg !26
  %.masked129 = fadd <4 x float> %v642, %.masked128, !dbg !26
  %.masked131 = fadd <4 x float> %v646, %.masked130, !dbg !26
  %.masked133 = fadd <4 x float> %v650, %.masked132, !dbg !26
  %.masked135 = fadd <4 x float> %v654, %.masked134, !dbg !26
  %.masked137 = fadd <4 x float> %v658, %.masked136, !dbg !26
  %.masked139 = fadd <4 x float> %v662, %.masked138, !dbg !26
  %.masked141 = fadd <4 x float> %v666, %.masked140, !dbg !26
  %.masked144 = fadd <4 x float> %v670, %.masked143, !dbg !26
  %.masked146 = fadd <4 x float> %v674, %.masked145, !dbg !26
  %.masked148 = fadd <4 x float> %v678, %.masked147, !dbg !26
  %.masked150 = fadd <4 x float> %v682, %.masked149, !dbg !26
  %.masked152 = fadd <4 x float> %v686, %.masked151, !dbg !26
  %.masked154 = fadd <4 x float> %v690, %.masked153, !dbg !26
  %.masked156 = fadd <4 x float> %v694, %.masked155, !dbg !26
  %.masked158 = fadd <4 x float> %v699, %.masked157, !dbg !26
  br label %v828, !dbg !13

v828:                                              ; preds = %v701, %v702
  %v829 = phi <4 x float> [ %.masked127, %v702 ], [ %v638, %v701 ], !dbg !26
  %v830 = phi <4 x float> [ %.masked129, %v702 ], [ %v642, %v701 ], !dbg !26
  %v831 = phi <4 x float> [ %.masked131, %v702 ], [ %v646, %v701 ], !dbg !26
  %v832 = phi <4 x float> [ %.masked133, %v702 ], [ %v650, %v701 ], !dbg !26
  %v833 = phi <4 x float> [ %.masked135, %v702 ], [ %v654, %v701 ], !dbg !26
  %v834 = phi <4 x float> [ %.masked137, %v702 ], [ %v658, %v701 ], !dbg !26
  %v835 = phi <4 x float> [ %.masked139, %v702 ], [ %v662, %v701 ], !dbg !26
  %v836 = phi <4 x float> [ %.masked141, %v702 ], [ %v666, %v701 ], !dbg !26
  %v837 = phi <4 x float> [ %.masked144, %v702 ], [ %v670, %v701 ], !dbg !26
  %v838 = phi <4 x float> [ %.masked146, %v702 ], [ %v674, %v701 ], !dbg !26
  %v839 = phi <4 x float> [ %.masked148, %v702 ], [ %v678, %v701 ], !dbg !26
  %v840 = phi <4 x float> [ %.masked150, %v702 ], [ %v682, %v701 ], !dbg !26
  %v841 = phi <4 x float> [ %.masked152, %v702 ], [ %v686, %v701 ], !dbg !26
  %v842 = phi <4 x float> [ %.masked154, %v702 ], [ %v690, %v701 ], !dbg !26
  %v843 = phi <4 x float> [ %.masked156, %v702 ], [ %v694, %v701 ], !dbg !26
  %v844 = phi <4 x float> [ %.masked158, %v702 ], [ %v699, %v701 ], !dbg !26
  br label %v845, !dbg !13

v845:                                              ; preds = %v828
  %v846 = extractelement <4 x float> %v829, i64 0, !dbg !26
  %v847 = extractelement <4 x float> %v829, i64 1, !dbg !26
  %v848 = extractelement <4 x float> %v829, i64 2, !dbg !26
  %v849 = extractelement <4 x float> %v829, i64 3, !dbg !26
  %v850 = extractelement <4 x float> %v830, i64 0, !dbg !26
  %v851 = extractelement <4 x float> %v830, i64 1, !dbg !26
  %v852 = extractelement <4 x float> %v830, i64 2, !dbg !26
  %v853 = extractelement <4 x float> %v830, i64 3, !dbg !26
  %v854 = extractelement <4 x float> %v831, i64 0, !dbg !26
  %v855 = extractelement <4 x float> %v831, i64 1, !dbg !26
  %v856 = extractelement <4 x float> %v831, i64 2, !dbg !26
  %v857 = extractelement <4 x float> %v831, i64 3, !dbg !26
  %v858 = extractelement <4 x float> %v832, i64 0, !dbg !26
  %v859 = extractelement <4 x float> %v832, i64 1, !dbg !26
  %v860 = extractelement <4 x float> %v832, i64 2, !dbg !26
  %v861 = extractelement <4 x float> %v832, i64 3, !dbg !26
  %v862 = extractelement <4 x float> %v833, i64 0, !dbg !26
  %v863 = extractelement <4 x float> %v833, i64 1, !dbg !26
  %v864 = extractelement <4 x float> %v833, i64 2, !dbg !26
  %v865 = extractelement <4 x float> %v833, i64 3, !dbg !26
  %v866 = extractelement <4 x float> %v834, i64 0, !dbg !26
  %v867 = extractelement <4 x float> %v834, i64 1, !dbg !26
  %v868 = extractelement <4 x float> %v834, i64 2, !dbg !26
  %v869 = extractelement <4 x float> %v834, i64 3, !dbg !26
  %v870 = extractelement <4 x float> %v835, i64 0, !dbg !26
  %v871 = extractelement <4 x float> %v835, i64 1, !dbg !26
  %v872 = extractelement <4 x float> %v835, i64 2, !dbg !26
  %v873 = extractelement <4 x float> %v835, i64 3, !dbg !26
  %v874 = extractelement <4 x float> %v836, i64 0, !dbg !26
  %v875 = extractelement <4 x float> %v836, i64 1, !dbg !26
  %v876 = extractelement <4 x float> %v836, i64 2, !dbg !26
  %v877 = extractelement <4 x float> %v836, i64 3, !dbg !26
  %v878 = extractelement <4 x float> %v837, i64 0, !dbg !26
  %v879 = extractelement <4 x float> %v837, i64 1, !dbg !26
  %v880 = extractelement <4 x float> %v837, i64 2, !dbg !26
  %v881 = extractelement <4 x float> %v837, i64 3, !dbg !26
  %v882 = extractelement <4 x float> %v838, i64 0, !dbg !26
  %v883 = extractelement <4 x float> %v838, i64 1, !dbg !26
  %v884 = extractelement <4 x float> %v838, i64 2, !dbg !26
  %v885 = extractelement <4 x float> %v838, i64 3, !dbg !26
  %v886 = extractelement <4 x float> %v839, i64 0, !dbg !26
  %v887 = extractelement <4 x float> %v839, i64 1, !dbg !26
  %v888 = extractelement <4 x float> %v839, i64 2, !dbg !26
  %v889 = extractelement <4 x float> %v839, i64 3, !dbg !26
  %v890 = extractelement <4 x float> %v840, i64 0, !dbg !26
  %v891 = extractelement <4 x float> %v840, i64 1, !dbg !26
  %v892 = extractelement <4 x float> %v840, i64 2, !dbg !26
  %v893 = extractelement <4 x float> %v840, i64 3, !dbg !26
  %v894 = extractelement <4 x float> %v841, i64 0, !dbg !26
  %v895 = extractelement <4 x float> %v841, i64 1, !dbg !26
  %v896 = extractelement <4 x float> %v841, i64 2, !dbg !26
  %v897 = extractelement <4 x float> %v841, i64 3, !dbg !26
  %v898 = extractelement <4 x float> %v842, i64 0, !dbg !26
  %v899 = extractelement <4 x float> %v842, i64 1, !dbg !26
  %v900 = extractelement <4 x float> %v842, i64 2, !dbg !26
  %v901 = extractelement <4 x float> %v842, i64 3, !dbg !26
  %v902 = extractelement <4 x float> %v843, i64 0, !dbg !26
  %v903 = extractelement <4 x float> %v843, i64 1, !dbg !26
  %v904 = extractelement <4 x float> %v843, i64 2, !dbg !26
  %v905 = extractelement <4 x float> %v843, i64 3, !dbg !26
  %v906 = extractelement <4 x float> %v844, i64 0, !dbg !26
  %v907 = extractelement <4 x float> %v844, i64 1, !dbg !26
  %v908 = extractelement <4 x float> %v844, i64 2, !dbg !26
  %v909 = extractelement <4 x float> %v844, i64 3, !dbg !26
  %v910 = call float @llvm.maxnum.f32(float %v846, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %local.reduce251 = call float @llvm.maxnum.f32(float %v910, float %v850), !dbg !27, !frisk.local_reduction !1
  %local.reduce252 = call float @llvm.maxnum.f32(float %v847, float %v851), !dbg !27, !frisk.local_reduction !1
  %local.reduce253 = call float @llvm.maxnum.f32(float %v848, float %v852), !dbg !27, !frisk.local_reduction !1
  %local.reduce254 = call float @llvm.maxnum.f32(float %v849, float %v853), !dbg !27, !frisk.local_reduction !1
  %local.reduce255 = call float @llvm.maxnum.f32(float %local.reduce251, float %v854), !dbg !27, !frisk.local_reduction !1
  %local.reduce256 = call float @llvm.maxnum.f32(float %local.reduce252, float %v855), !dbg !27, !frisk.local_reduction !1
  %local.reduce257 = call float @llvm.maxnum.f32(float %local.reduce253, float %v856), !dbg !27, !frisk.local_reduction !1
  %local.reduce258 = call float @llvm.maxnum.f32(float %local.reduce254, float %v857), !dbg !27, !frisk.local_reduction !1
  %local.reduce259 = call float @llvm.maxnum.f32(float %local.reduce255, float %v858), !dbg !27, !frisk.local_reduction !1
  %local.reduce260 = call float @llvm.maxnum.f32(float %local.reduce256, float %v859), !dbg !27, !frisk.local_reduction !1
  %local.reduce261 = call float @llvm.maxnum.f32(float %local.reduce257, float %v860), !dbg !27, !frisk.local_reduction !1
  %local.reduce262 = call float @llvm.maxnum.f32(float %local.reduce258, float %v861), !dbg !27, !frisk.local_reduction !1
  %local.reduce263 = call float @llvm.maxnum.f32(float %local.reduce259, float %v862), !dbg !27, !frisk.local_reduction !1
  %local.reduce264 = call float @llvm.maxnum.f32(float %local.reduce260, float %v863), !dbg !27, !frisk.local_reduction !1
  %local.reduce265 = call float @llvm.maxnum.f32(float %local.reduce261, float %v864), !dbg !27, !frisk.local_reduction !1
  %local.reduce266 = call float @llvm.maxnum.f32(float %local.reduce262, float %v865), !dbg !27, !frisk.local_reduction !1
  %local.reduce267 = call float @llvm.maxnum.f32(float %local.reduce263, float %v866), !dbg !27, !frisk.local_reduction !1
  %local.reduce268 = call float @llvm.maxnum.f32(float %local.reduce264, float %v867), !dbg !27, !frisk.local_reduction !1
  %local.reduce269 = call float @llvm.maxnum.f32(float %local.reduce265, float %v868), !dbg !27, !frisk.local_reduction !1
  %local.reduce270 = call float @llvm.maxnum.f32(float %local.reduce266, float %v869), !dbg !27, !frisk.local_reduction !1
  %local.reduce271 = call float @llvm.maxnum.f32(float %local.reduce267, float %v870), !dbg !27, !frisk.local_reduction !1
  %local.reduce272 = call float @llvm.maxnum.f32(float %local.reduce268, float %v871), !dbg !27, !frisk.local_reduction !1
  %local.reduce273 = call float @llvm.maxnum.f32(float %local.reduce269, float %v872), !dbg !27, !frisk.local_reduction !1
  %local.reduce274 = call float @llvm.maxnum.f32(float %local.reduce270, float %v873), !dbg !27, !frisk.local_reduction !1
  %local.reduce275 = call float @llvm.maxnum.f32(float %local.reduce271, float %v874), !dbg !27, !frisk.local_reduction !1
  %local.reduce276 = call float @llvm.maxnum.f32(float %local.reduce272, float %v875), !dbg !27, !frisk.local_reduction !1
  %local.reduce277 = call float @llvm.maxnum.f32(float %local.reduce273, float %v876), !dbg !27, !frisk.local_reduction !1
  %local.reduce278 = call float @llvm.maxnum.f32(float %local.reduce274, float %v877), !dbg !27, !frisk.local_reduction !1
  %local.reduce279 = call float @llvm.maxnum.f32(float %local.reduce275, float %local.reduce276), !dbg !27, !frisk.local_reduction !1
  %local.reduce280 = call float @llvm.maxnum.f32(float %local.reduce277, float %local.reduce278), !dbg !27, !frisk.local_reduction !1
  %v911 = call float @llvm.maxnum.f32(float %local.reduce279, float %local.reduce280), !dbg !27, !frisk.local_reduction !1
  %v912 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %v913 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %v912), !dbg !27
  %v914 = add nuw nsw i32 %v913, 64, !dbg !27
  %v915 = xor i32 %v913, 16, !dbg !27
  %v916 = and i32 %v914, 192, !dbg !27
  %v917 = icmp samesign ult i32 %v915, %v916, !dbg !27
  %v918 = select i1 %v917, i32 %v915, i32 %v913, !dbg !27
  %v919 = shl nuw nsw i32 %v918, 2, !dbg !27
  %v920 = bitcast float %v911 to i32, !dbg !27
  %v921 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v919, i32 %v920), !dbg !27
  %v922 = bitcast i32 %v921 to float, !dbg !27
  %v923 = call float @llvm.maxnum.f32(float %v911, float %v922), !dbg !27
  %v924 = xor i32 %v913, 32, !dbg !27
  %v925 = icmp samesign ult i32 %v924, %v916, !dbg !27
  %v926 = select i1 %v925, i32 %v924, i32 %v913, !dbg !27
  %v927 = shl nuw nsw i32 %v926, 2, !dbg !27
  %v928 = bitcast float %v923 to i32, !dbg !27
  %v929 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v927, i32 %v928), !dbg !27
  %v930 = bitcast i32 %v929 to float, !dbg !27
  %v931 = call float @llvm.maxnum.f32(float %v923, float %v930), !dbg !27
  %v932 = extractvalue [2 x <1 x float>] %v208, 0, !dbg !27
  %v933 = insertelement <1 x float> poison, float %v931, i64 0, !dbg !27
  %v934 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %v933, <1 x float> %v932), !dbg !27
  %v935 = call float @llvm.maxnum.f32(float %v878, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %local.reduce220 = call float @llvm.maxnum.f32(float %v935, float %v882), !dbg !27, !frisk.local_reduction !1
  %local.reduce221 = call float @llvm.maxnum.f32(float %v879, float %v883), !dbg !27, !frisk.local_reduction !1
  %local.reduce222 = call float @llvm.maxnum.f32(float %v880, float %v884), !dbg !27, !frisk.local_reduction !1
  %local.reduce223 = call float @llvm.maxnum.f32(float %v881, float %v885), !dbg !27, !frisk.local_reduction !1
  %local.reduce224 = call float @llvm.maxnum.f32(float %local.reduce220, float %v886), !dbg !27, !frisk.local_reduction !1
  %local.reduce225 = call float @llvm.maxnum.f32(float %local.reduce221, float %v887), !dbg !27, !frisk.local_reduction !1
  %local.reduce226 = call float @llvm.maxnum.f32(float %local.reduce222, float %v888), !dbg !27, !frisk.local_reduction !1
  %local.reduce227 = call float @llvm.maxnum.f32(float %local.reduce223, float %v889), !dbg !27, !frisk.local_reduction !1
  %local.reduce228 = call float @llvm.maxnum.f32(float %local.reduce224, float %v890), !dbg !27, !frisk.local_reduction !1
  %local.reduce229 = call float @llvm.maxnum.f32(float %local.reduce225, float %v891), !dbg !27, !frisk.local_reduction !1
  %local.reduce230 = call float @llvm.maxnum.f32(float %local.reduce226, float %v892), !dbg !27, !frisk.local_reduction !1
  %local.reduce231 = call float @llvm.maxnum.f32(float %local.reduce227, float %v893), !dbg !27, !frisk.local_reduction !1
  %local.reduce232 = call float @llvm.maxnum.f32(float %local.reduce228, float %v894), !dbg !27, !frisk.local_reduction !1
  %local.reduce233 = call float @llvm.maxnum.f32(float %local.reduce229, float %v895), !dbg !27, !frisk.local_reduction !1
  %local.reduce234 = call float @llvm.maxnum.f32(float %local.reduce230, float %v896), !dbg !27, !frisk.local_reduction !1
  %local.reduce235 = call float @llvm.maxnum.f32(float %local.reduce231, float %v897), !dbg !27, !frisk.local_reduction !1
  %local.reduce236 = call float @llvm.maxnum.f32(float %local.reduce232, float %v898), !dbg !27, !frisk.local_reduction !1
  %local.reduce237 = call float @llvm.maxnum.f32(float %local.reduce233, float %v899), !dbg !27, !frisk.local_reduction !1
  %local.reduce238 = call float @llvm.maxnum.f32(float %local.reduce234, float %v900), !dbg !27, !frisk.local_reduction !1
  %local.reduce239 = call float @llvm.maxnum.f32(float %local.reduce235, float %v901), !dbg !27, !frisk.local_reduction !1
  %local.reduce240 = call float @llvm.maxnum.f32(float %local.reduce236, float %v902), !dbg !27, !frisk.local_reduction !1
  %local.reduce241 = call float @llvm.maxnum.f32(float %local.reduce237, float %v903), !dbg !27, !frisk.local_reduction !1
  %local.reduce242 = call float @llvm.maxnum.f32(float %local.reduce238, float %v904), !dbg !27, !frisk.local_reduction !1
  %local.reduce243 = call float @llvm.maxnum.f32(float %local.reduce239, float %v905), !dbg !27, !frisk.local_reduction !1
  %local.reduce244 = call float @llvm.maxnum.f32(float %local.reduce240, float %v906), !dbg !27, !frisk.local_reduction !1
  %local.reduce245 = call float @llvm.maxnum.f32(float %local.reduce241, float %v907), !dbg !27, !frisk.local_reduction !1
  %local.reduce246 = call float @llvm.maxnum.f32(float %local.reduce242, float %v908), !dbg !27, !frisk.local_reduction !1
  %local.reduce247 = call float @llvm.maxnum.f32(float %local.reduce243, float %v909), !dbg !27, !frisk.local_reduction !1
  %local.reduce248 = call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !27, !frisk.local_reduction !1
  %local.reduce249 = call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !27, !frisk.local_reduction !1
  %v936 = call float @llvm.maxnum.f32(float %local.reduce248, float %local.reduce249), !dbg !27, !frisk.local_reduction !1
  %v937 = bitcast float %v936 to i32, !dbg !27
  %v938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v919, i32 %v937), !dbg !27
  %v939 = bitcast i32 %v938 to float, !dbg !27
  %v940 = call float @llvm.maxnum.f32(float %v936, float %v939), !dbg !27
  %v941 = bitcast float %v940 to i32, !dbg !27
  %v942 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v927, i32 %v941), !dbg !27
  %v943 = bitcast i32 %v942 to float, !dbg !27
  %v944 = call float @llvm.maxnum.f32(float %v940, float %v943), !dbg !27
  %v945 = extractvalue [2 x <1 x float>] %v208, 1, !dbg !27
  %v946 = insertelement <1 x float> poison, float %v944, i64 0, !dbg !27
  %v947 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %v946, <1 x float> %v945), !dbg !27
  %v948 = shufflevector <1 x float> %v934, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %v949 = fsub <4 x float> %v829, %v948, !dbg !28
  %v950 = fsub <4 x float> %v830, %v948, !dbg !28
  %v951 = fsub <4 x float> %v831, %v948, !dbg !28
  %v952 = fsub <4 x float> %v832, %v948, !dbg !28
  %v953 = fsub <4 x float> %v833, %v948, !dbg !28
  %v954 = fsub <4 x float> %v834, %v948, !dbg !28
  %v955 = fsub <4 x float> %v835, %v948, !dbg !28
  %v956 = fsub <4 x float> %v836, %v948, !dbg !28
  %v957 = shufflevector <1 x float> %v947, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %v958 = fsub <4 x float> %v837, %v957, !dbg !28
  %v959 = fsub <4 x float> %v838, %v957, !dbg !28
  %v960 = fsub <4 x float> %v839, %v957, !dbg !28
  %v961 = fsub <4 x float> %v840, %v957, !dbg !28
  %v962 = fsub <4 x float> %v841, %v957, !dbg !28
  %v963 = fsub <4 x float> %v842, %v957, !dbg !28
  %v964 = fsub <4 x float> %v843, %v957, !dbg !28
  %v965 = fsub <4 x float> %v844, %v957, !dbg !28
  %v966 = fsub <1 x float> %v932, %v934, !dbg !29
  %v967 = extractelement <1 x float> %v966, i64 0, !dbg !30
  %v968 = call float @llvm.amdgcn.exp2.f32(float %v967), !dbg !30
  %v969 = insertelement <4 x float> poison, float %v968, i64 0, !dbg !35
  %v970 = shufflevector <4 x float> %v969, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %v971 = extractvalue [2 x <1 x float>] %v207, 0, !dbg !31
  %v972 = extractelement <1 x float> %v971, i64 0, !dbg !31
  %v973 = fmul float %v972, %v968, !dbg !31
  %v974 = fsub <1 x float> %v945, %v947, !dbg !29
  %v975 = extractelement <1 x float> %v974, i64 0, !dbg !30
  %v976 = call float @llvm.amdgcn.exp2.f32(float %v975), !dbg !30
  %v977 = insertelement <4 x float> poison, float %v976, i64 0, !dbg !35
  %v978 = shufflevector <4 x float> %v977, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %v979 = extractvalue [2 x <1 x float>] %v207, 1, !dbg !31
  %v980 = extractelement <1 x float> %v979, i64 0, !dbg !31
  %v981 = fmul float %v980, %v976, !dbg !31
  %v982 = extractelement <4 x float> %v949, i64 0, !dbg !32
  %v983 = call float @llvm.amdgcn.exp2.f32(float %v982), !dbg !32
  %v984 = insertelement <4 x float> poison, float %v983, i64 0, !dbg !34
  %v985 = extractelement <4 x float> %v949, i64 1, !dbg !32
  %v986 = call float @llvm.amdgcn.exp2.f32(float %v985), !dbg !32
  %v987 = insertelement <4 x float> %v984, float %v986, i64 1, !dbg !34
  %v988 = extractelement <4 x float> %v949, i64 2, !dbg !32
  %v989 = call float @llvm.amdgcn.exp2.f32(float %v988), !dbg !32
  %v990 = insertelement <4 x float> %v987, float %v989, i64 2, !dbg !34
  %v991 = extractelement <4 x float> %v949, i64 3, !dbg !32
  %v992 = call float @llvm.amdgcn.exp2.f32(float %v991), !dbg !32
  %v993 = insertelement <4 x float> %v990, float %v992, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v993) #9, !dbg !34, !frisk.register_lifetimes !1
  %v994 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v994) #9, !dbg !32, !frisk.register_lifetimes !1
  %v995 = extractelement <4 x float> %v950, i64 0, !dbg !32
  %v996 = call float @llvm.amdgcn.exp2.f32(float %v995), !dbg !32
  %v997 = insertelement <4 x float> poison, float %v996, i64 0, !dbg !34
  %local.reduce189 = fadd float %v983, %v996, !dbg !33, !frisk.local_reduction !1
  %v998 = extractelement <4 x float> %v950, i64 1, !dbg !32
  %v999 = call float @llvm.amdgcn.exp2.f32(float %v998), !dbg !32
  %v1000 = insertelement <4 x float> %v997, float %v999, i64 1, !dbg !34
  %local.reduce190 = fadd float %v986, %v999, !dbg !33, !frisk.local_reduction !1
  %v1001 = extractelement <4 x float> %v950, i64 2, !dbg !32
  %v1002 = call float @llvm.amdgcn.exp2.f32(float %v1001), !dbg !32
  %v1003 = insertelement <4 x float> %v1000, float %v1002, i64 2, !dbg !34
  %local.reduce191 = fadd float %v989, %v1002, !dbg !33, !frisk.local_reduction !1
  %v1004 = extractelement <4 x float> %v950, i64 3, !dbg !32
  %v1005 = call float @llvm.amdgcn.exp2.f32(float %v1004), !dbg !32
  %v1006 = insertelement <4 x float> %v1003, float %v1005, i64 3, !dbg !34
  %math.result465 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1006) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1007 = fptrunc <4 x float> %math.result465 to <4 x half>, !dbg !34
  %math.packed466 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1007) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce192 = fadd float %v992, %v1005, !dbg !33, !frisk.local_reduction !1
  %v1008 = extractelement <4 x float> %v951, i64 0, !dbg !32
  %v1009 = call float @llvm.amdgcn.exp2.f32(float %v1008), !dbg !32
  %v1010 = insertelement <4 x float> poison, float %v1009, i64 0, !dbg !34
  %local.reduce193 = fadd float %local.reduce189, %v1009, !dbg !33, !frisk.local_reduction !1
  %v1011 = extractelement <4 x float> %v951, i64 1, !dbg !32
  %v1012 = call float @llvm.amdgcn.exp2.f32(float %v1011), !dbg !32
  %v1013 = insertelement <4 x float> %v1010, float %v1012, i64 1, !dbg !34
  %local.reduce194 = fadd float %local.reduce190, %v1012, !dbg !33, !frisk.local_reduction !1
  %v1014 = extractelement <4 x float> %v951, i64 2, !dbg !32
  %v1015 = call float @llvm.amdgcn.exp2.f32(float %v1014), !dbg !32
  %v1016 = insertelement <4 x float> %v1013, float %v1015, i64 2, !dbg !34
  %local.reduce195 = fadd float %local.reduce191, %v1015, !dbg !33, !frisk.local_reduction !1
  %v1017 = extractelement <4 x float> %v951, i64 3, !dbg !32
  %v1018 = call float @llvm.amdgcn.exp2.f32(float %v1017), !dbg !32
  %v1019 = insertelement <4 x float> %v1016, float %v1018, i64 3, !dbg !34
  %math.result467 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1019) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1020 = fptrunc <4 x float> %math.result467 to <4 x half>, !dbg !34
  %math.packed468 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1020) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce196 = fadd float %local.reduce192, %v1018, !dbg !33, !frisk.local_reduction !1
  %v1021 = extractelement <4 x float> %v952, i64 0, !dbg !32
  %v1022 = call float @llvm.amdgcn.exp2.f32(float %v1021), !dbg !32
  %v1023 = insertelement <4 x float> poison, float %v1022, i64 0, !dbg !34
  %local.reduce197 = fadd float %local.reduce193, %v1022, !dbg !33, !frisk.local_reduction !1
  %v1024 = extractelement <4 x float> %v952, i64 1, !dbg !32
  %v1025 = call float @llvm.amdgcn.exp2.f32(float %v1024), !dbg !32
  %v1026 = insertelement <4 x float> %v1023, float %v1025, i64 1, !dbg !34
  %local.reduce198 = fadd float %local.reduce194, %v1025, !dbg !33, !frisk.local_reduction !1
  %v1027 = extractelement <4 x float> %v952, i64 2, !dbg !32
  %v1028 = call float @llvm.amdgcn.exp2.f32(float %v1027), !dbg !32
  %v1029 = insertelement <4 x float> %v1026, float %v1028, i64 2, !dbg !34
  %local.reduce199 = fadd float %local.reduce195, %v1028, !dbg !33, !frisk.local_reduction !1
  %v1030 = extractelement <4 x float> %v952, i64 3, !dbg !32
  %v1031 = call float @llvm.amdgcn.exp2.f32(float %v1030), !dbg !32
  %v1032 = insertelement <4 x float> %v1029, float %v1031, i64 3, !dbg !34
  %math.result469 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1032) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1033 = fptrunc <4 x float> %math.result469 to <4 x half>, !dbg !34
  %math.packed470 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1033) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce200 = fadd float %local.reduce196, %v1031, !dbg !33, !frisk.local_reduction !1
  %v1034 = extractelement <4 x float> %v953, i64 0, !dbg !32
  %v1035 = call float @llvm.amdgcn.exp2.f32(float %v1034), !dbg !32
  %v1036 = insertelement <4 x float> poison, float %v1035, i64 0, !dbg !34
  %local.reduce201 = fadd float %local.reduce197, %v1035, !dbg !33, !frisk.local_reduction !1
  %v1037 = extractelement <4 x float> %v953, i64 1, !dbg !32
  %v1038 = call float @llvm.amdgcn.exp2.f32(float %v1037), !dbg !32
  %v1039 = insertelement <4 x float> %v1036, float %v1038, i64 1, !dbg !34
  %local.reduce202 = fadd float %local.reduce198, %v1038, !dbg !33, !frisk.local_reduction !1
  %v1040 = extractelement <4 x float> %v953, i64 2, !dbg !32
  %v1041 = call float @llvm.amdgcn.exp2.f32(float %v1040), !dbg !32
  %v1042 = insertelement <4 x float> %v1039, float %v1041, i64 2, !dbg !34
  %local.reduce203 = fadd float %local.reduce199, %v1041, !dbg !33, !frisk.local_reduction !1
  %v1043 = extractelement <4 x float> %v953, i64 3, !dbg !32
  %v1044 = call float @llvm.amdgcn.exp2.f32(float %v1043), !dbg !32
  %v1045 = insertelement <4 x float> %v1042, float %v1044, i64 3, !dbg !34
  %math.result471 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1045) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1046 = fptrunc <4 x float> %math.result471 to <4 x half>, !dbg !34
  %math.packed472 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1046) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce204 = fadd float %local.reduce200, %v1044, !dbg !33, !frisk.local_reduction !1
  %v1047 = extractelement <4 x float> %v954, i64 0, !dbg !32
  %v1048 = call float @llvm.amdgcn.exp2.f32(float %v1047), !dbg !32
  %v1049 = insertelement <4 x float> poison, float %v1048, i64 0, !dbg !34
  %local.reduce205 = fadd float %local.reduce201, %v1048, !dbg !33, !frisk.local_reduction !1
  %v1050 = extractelement <4 x float> %v954, i64 1, !dbg !32
  %v1051 = call float @llvm.amdgcn.exp2.f32(float %v1050), !dbg !32
  %v1052 = insertelement <4 x float> %v1049, float %v1051, i64 1, !dbg !34
  %local.reduce206 = fadd float %local.reduce202, %v1051, !dbg !33, !frisk.local_reduction !1
  %v1053 = extractelement <4 x float> %v954, i64 2, !dbg !32
  %v1054 = call float @llvm.amdgcn.exp2.f32(float %v1053), !dbg !32
  %v1055 = insertelement <4 x float> %v1052, float %v1054, i64 2, !dbg !34
  %local.reduce207 = fadd float %local.reduce203, %v1054, !dbg !33, !frisk.local_reduction !1
  %v1056 = extractelement <4 x float> %v954, i64 3, !dbg !32
  %v1057 = call float @llvm.amdgcn.exp2.f32(float %v1056), !dbg !32
  %v1058 = insertelement <4 x float> %v1055, float %v1057, i64 3, !dbg !34
  %math.result473 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1058) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1059 = fptrunc <4 x float> %math.result473 to <4 x half>, !dbg !34
  %math.packed474 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1059) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce208 = fadd float %local.reduce204, %v1057, !dbg !33, !frisk.local_reduction !1
  %v1060 = extractelement <4 x float> %v955, i64 0, !dbg !32
  %v1061 = call float @llvm.amdgcn.exp2.f32(float %v1060), !dbg !32
  %v1062 = insertelement <4 x float> poison, float %v1061, i64 0, !dbg !34
  %local.reduce209 = fadd float %local.reduce205, %v1061, !dbg !33, !frisk.local_reduction !1
  %v1063 = extractelement <4 x float> %v955, i64 1, !dbg !32
  %v1064 = call float @llvm.amdgcn.exp2.f32(float %v1063), !dbg !32
  %v1065 = insertelement <4 x float> %v1062, float %v1064, i64 1, !dbg !34
  %local.reduce210 = fadd float %local.reduce206, %v1064, !dbg !33, !frisk.local_reduction !1
  %v1066 = extractelement <4 x float> %v955, i64 2, !dbg !32
  %v1067 = call float @llvm.amdgcn.exp2.f32(float %v1066), !dbg !32
  %v1068 = insertelement <4 x float> %v1065, float %v1067, i64 2, !dbg !34
  %local.reduce211 = fadd float %local.reduce207, %v1067, !dbg !33, !frisk.local_reduction !1
  %v1069 = extractelement <4 x float> %v955, i64 3, !dbg !32
  %v1070 = call float @llvm.amdgcn.exp2.f32(float %v1069), !dbg !32
  %v1071 = insertelement <4 x float> %v1068, float %v1070, i64 3, !dbg !34
  %math.result475 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1071) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1072 = fptrunc <4 x float> %math.result475 to <4 x half>, !dbg !34
  %math.packed476 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1072) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce212 = fadd float %local.reduce208, %v1070, !dbg !33, !frisk.local_reduction !1
  %v1073 = extractelement <4 x float> %v956, i64 0, !dbg !32
  %v1074 = call float @llvm.amdgcn.exp2.f32(float %v1073), !dbg !32
  %v1075 = insertelement <4 x float> poison, float %v1074, i64 0, !dbg !34
  %local.reduce213 = fadd float %local.reduce209, %v1074, !dbg !33, !frisk.local_reduction !1
  %v1076 = extractelement <4 x float> %v956, i64 1, !dbg !32
  %v1077 = call float @llvm.amdgcn.exp2.f32(float %v1076), !dbg !32
  %v1078 = insertelement <4 x float> %v1075, float %v1077, i64 1, !dbg !34
  %local.reduce214 = fadd float %local.reduce210, %v1077, !dbg !33, !frisk.local_reduction !1
  %local.reduce217 = fadd float %local.reduce213, %local.reduce214, !dbg !33, !frisk.local_reduction !1
  %v1079 = extractelement <4 x float> %v956, i64 2, !dbg !32
  %v1080 = call float @llvm.amdgcn.exp2.f32(float %v1079), !dbg !32
  %v1081 = insertelement <4 x float> %v1078, float %v1080, i64 2, !dbg !34
  %local.reduce215 = fadd float %local.reduce211, %v1080, !dbg !33, !frisk.local_reduction !1
  %v1082 = extractelement <4 x float> %v956, i64 3, !dbg !32
  %v1083 = call float @llvm.amdgcn.exp2.f32(float %v1082), !dbg !32
  %v1084 = insertelement <4 x float> %v1081, float %v1083, i64 3, !dbg !34
  %math.result477 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1084) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1085 = fptrunc <4 x float> %math.result477 to <4 x half>, !dbg !34
  %math.packed478 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1085) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce216 = fadd float %local.reduce212, %v1083, !dbg !33, !frisk.local_reduction !1
  %local.reduce218 = fadd float %local.reduce215, %local.reduce216, !dbg !33, !frisk.local_reduction !1
  %v1086 = fadd float %local.reduce217, %local.reduce218, !dbg !33, !frisk.local_reduction !1
  %v1087 = extractelement <4 x float> %v958, i64 0, !dbg !32
  %v1088 = call float @llvm.amdgcn.exp2.f32(float %v1087), !dbg !32
  %v1089 = insertelement <4 x float> poison, float %v1088, i64 0, !dbg !34
  %v1090 = extractelement <4 x float> %v958, i64 1, !dbg !32
  %v1091 = call float @llvm.amdgcn.exp2.f32(float %v1090), !dbg !32
  %v1092 = insertelement <4 x float> %v1089, float %v1091, i64 1, !dbg !34
  %v1093 = extractelement <4 x float> %v958, i64 2, !dbg !32
  %v1094 = call float @llvm.amdgcn.exp2.f32(float %v1093), !dbg !32
  %v1095 = insertelement <4 x float> %v1092, float %v1094, i64 2, !dbg !34
  %v1096 = extractelement <4 x float> %v958, i64 3, !dbg !32
  %v1097 = call float @llvm.amdgcn.exp2.f32(float %v1096), !dbg !32
  %v1098 = insertelement <4 x float> %v1095, float %v1097, i64 3, !dbg !34
  %math.result479 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1098) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1099 = fptrunc <4 x float> %math.result479 to <4 x half>, !dbg !34
  %math.packed480 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1099) #9, !dbg !32, !frisk.register_lifetimes !1
  %v1100 = extractelement <4 x float> %v959, i64 0, !dbg !32
  %v1101 = call float @llvm.amdgcn.exp2.f32(float %v1100), !dbg !32
  %v1102 = insertelement <4 x float> poison, float %v1101, i64 0, !dbg !34
  %local.reduce = fadd float %v1088, %v1101, !dbg !33, !frisk.local_reduction !1
  %v1103 = extractelement <4 x float> %v959, i64 1, !dbg !32
  %v1104 = call float @llvm.amdgcn.exp2.f32(float %v1103), !dbg !32
  %v1105 = insertelement <4 x float> %v1102, float %v1104, i64 1, !dbg !34
  %local.reduce159 = fadd float %v1091, %v1104, !dbg !33, !frisk.local_reduction !1
  %v1106 = extractelement <4 x float> %v959, i64 2, !dbg !32
  %v1107 = call float @llvm.amdgcn.exp2.f32(float %v1106), !dbg !32
  %v1108 = insertelement <4 x float> %v1105, float %v1107, i64 2, !dbg !34
  %local.reduce160 = fadd float %v1094, %v1107, !dbg !33, !frisk.local_reduction !1
  %v1109 = extractelement <4 x float> %v959, i64 3, !dbg !32
  %v1110 = call float @llvm.amdgcn.exp2.f32(float %v1109), !dbg !32
  %v1111 = insertelement <4 x float> %v1108, float %v1110, i64 3, !dbg !34
  %math.result481 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1111) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1112 = fptrunc <4 x float> %math.result481 to <4 x half>, !dbg !34
  %math.packed482 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1112) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce161 = fadd float %v1097, %v1110, !dbg !33, !frisk.local_reduction !1
  %v1113 = extractelement <4 x float> %v960, i64 0, !dbg !32
  %v1114 = call float @llvm.amdgcn.exp2.f32(float %v1113), !dbg !32
  %v1115 = insertelement <4 x float> poison, float %v1114, i64 0, !dbg !34
  %local.reduce162 = fadd float %local.reduce, %v1114, !dbg !33, !frisk.local_reduction !1
  %v1116 = extractelement <4 x float> %v960, i64 1, !dbg !32
  %v1117 = call float @llvm.amdgcn.exp2.f32(float %v1116), !dbg !32
  %v1118 = insertelement <4 x float> %v1115, float %v1117, i64 1, !dbg !34
  %local.reduce163 = fadd float %local.reduce159, %v1117, !dbg !33, !frisk.local_reduction !1
  %v1119 = extractelement <4 x float> %v960, i64 2, !dbg !32
  %v1120 = call float @llvm.amdgcn.exp2.f32(float %v1119), !dbg !32
  %v1121 = insertelement <4 x float> %v1118, float %v1120, i64 2, !dbg !34
  %local.reduce164 = fadd float %local.reduce160, %v1120, !dbg !33, !frisk.local_reduction !1
  %v1122 = extractelement <4 x float> %v960, i64 3, !dbg !32
  %v1123 = call float @llvm.amdgcn.exp2.f32(float %v1122), !dbg !32
  %v1124 = insertelement <4 x float> %v1121, float %v1123, i64 3, !dbg !34
  %math.result483 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1124) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1125 = fptrunc <4 x float> %math.result483 to <4 x half>, !dbg !34
  %math.packed484 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1125) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce165 = fadd float %local.reduce161, %v1123, !dbg !33, !frisk.local_reduction !1
  %v1126 = extractelement <4 x float> %v961, i64 0, !dbg !32
  %v1127 = call float @llvm.amdgcn.exp2.f32(float %v1126), !dbg !32
  %v1128 = insertelement <4 x float> poison, float %v1127, i64 0, !dbg !34
  %local.reduce166 = fadd float %local.reduce162, %v1127, !dbg !33, !frisk.local_reduction !1
  %v1129 = extractelement <4 x float> %v961, i64 1, !dbg !32
  %v1130 = call float @llvm.amdgcn.exp2.f32(float %v1129), !dbg !32
  %v1131 = insertelement <4 x float> %v1128, float %v1130, i64 1, !dbg !34
  %local.reduce167 = fadd float %local.reduce163, %v1130, !dbg !33, !frisk.local_reduction !1
  %v1132 = extractelement <4 x float> %v961, i64 2, !dbg !32
  %v1133 = call float @llvm.amdgcn.exp2.f32(float %v1132), !dbg !32
  %v1134 = insertelement <4 x float> %v1131, float %v1133, i64 2, !dbg !34
  %local.reduce168 = fadd float %local.reduce164, %v1133, !dbg !33, !frisk.local_reduction !1
  %v1135 = extractelement <4 x float> %v961, i64 3, !dbg !32
  %v1136 = call float @llvm.amdgcn.exp2.f32(float %v1135), !dbg !32
  %v1137 = insertelement <4 x float> %v1134, float %v1136, i64 3, !dbg !34
  %math.result485 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1137) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1138 = fptrunc <4 x float> %math.result485 to <4 x half>, !dbg !34
  %math.packed486 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1138) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce169 = fadd float %local.reduce165, %v1136, !dbg !33, !frisk.local_reduction !1
  %v1139 = extractelement <4 x float> %v962, i64 0, !dbg !32
  %v1140 = call float @llvm.amdgcn.exp2.f32(float %v1139), !dbg !32
  %v1141 = insertelement <4 x float> poison, float %v1140, i64 0, !dbg !34
  %local.reduce170 = fadd float %local.reduce166, %v1140, !dbg !33, !frisk.local_reduction !1
  %v1142 = extractelement <4 x float> %v962, i64 1, !dbg !32
  %v1143 = call float @llvm.amdgcn.exp2.f32(float %v1142), !dbg !32
  %v1144 = insertelement <4 x float> %v1141, float %v1143, i64 1, !dbg !34
  %local.reduce171 = fadd float %local.reduce167, %v1143, !dbg !33, !frisk.local_reduction !1
  %v1145 = extractelement <4 x float> %v962, i64 2, !dbg !32
  %v1146 = call float @llvm.amdgcn.exp2.f32(float %v1145), !dbg !32
  %v1147 = insertelement <4 x float> %v1144, float %v1146, i64 2, !dbg !34
  %local.reduce172 = fadd float %local.reduce168, %v1146, !dbg !33, !frisk.local_reduction !1
  %v1148 = extractelement <4 x float> %v962, i64 3, !dbg !32
  %v1149 = call float @llvm.amdgcn.exp2.f32(float %v1148), !dbg !32
  %v1150 = insertelement <4 x float> %v1147, float %v1149, i64 3, !dbg !34
  %math.result487 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1150) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1151 = fptrunc <4 x float> %math.result487 to <4 x half>, !dbg !34
  %math.packed488 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1151) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce173 = fadd float %local.reduce169, %v1149, !dbg !33, !frisk.local_reduction !1
  %v1152 = extractelement <4 x float> %v963, i64 0, !dbg !32
  %v1153 = call float @llvm.amdgcn.exp2.f32(float %v1152), !dbg !32
  %v1154 = insertelement <4 x float> poison, float %v1153, i64 0, !dbg !34
  %local.reduce174 = fadd float %local.reduce170, %v1153, !dbg !33, !frisk.local_reduction !1
  %v1155 = extractelement <4 x float> %v963, i64 1, !dbg !32
  %v1156 = call float @llvm.amdgcn.exp2.f32(float %v1155), !dbg !32
  %v1157 = insertelement <4 x float> %v1154, float %v1156, i64 1, !dbg !34
  %local.reduce175 = fadd float %local.reduce171, %v1156, !dbg !33, !frisk.local_reduction !1
  %v1158 = extractelement <4 x float> %v963, i64 2, !dbg !32
  %v1159 = call float @llvm.amdgcn.exp2.f32(float %v1158), !dbg !32
  %v1160 = insertelement <4 x float> %v1157, float %v1159, i64 2, !dbg !34
  %local.reduce176 = fadd float %local.reduce172, %v1159, !dbg !33, !frisk.local_reduction !1
  %v1161 = extractelement <4 x float> %v963, i64 3, !dbg !32
  %v1162 = call float @llvm.amdgcn.exp2.f32(float %v1161), !dbg !32
  %v1163 = insertelement <4 x float> %v1160, float %v1162, i64 3, !dbg !34
  %math.result489 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1163) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1164 = fptrunc <4 x float> %math.result489 to <4 x half>, !dbg !34
  %math.packed490 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1164) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce177 = fadd float %local.reduce173, %v1162, !dbg !33, !frisk.local_reduction !1
  %v1165 = extractelement <4 x float> %v964, i64 0, !dbg !32
  %v1166 = call float @llvm.amdgcn.exp2.f32(float %v1165), !dbg !32
  %v1167 = insertelement <4 x float> poison, float %v1166, i64 0, !dbg !34
  %local.reduce178 = fadd float %local.reduce174, %v1166, !dbg !33, !frisk.local_reduction !1
  %v1168 = extractelement <4 x float> %v964, i64 1, !dbg !32
  %v1169 = call float @llvm.amdgcn.exp2.f32(float %v1168), !dbg !32
  %v1170 = insertelement <4 x float> %v1167, float %v1169, i64 1, !dbg !34
  %local.reduce179 = fadd float %local.reduce175, %v1169, !dbg !33, !frisk.local_reduction !1
  %v1171 = extractelement <4 x float> %v964, i64 2, !dbg !32
  %v1172 = call float @llvm.amdgcn.exp2.f32(float %v1171), !dbg !32
  %v1173 = insertelement <4 x float> %v1170, float %v1172, i64 2, !dbg !34
  %local.reduce180 = fadd float %local.reduce176, %v1172, !dbg !33, !frisk.local_reduction !1
  %v1174 = extractelement <4 x float> %v964, i64 3, !dbg !32
  %v1175 = call float @llvm.amdgcn.exp2.f32(float %v1174), !dbg !32
  %v1176 = insertelement <4 x float> %v1173, float %v1175, i64 3, !dbg !34
  %math.result491 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1176) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1177 = fptrunc <4 x float> %math.result491 to <4 x half>, !dbg !34
  %math.packed492 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1177) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce181 = fadd float %local.reduce177, %v1175, !dbg !33, !frisk.local_reduction !1
  %v1178 = extractelement <4 x float> %v965, i64 0, !dbg !32
  %v1179 = call float @llvm.amdgcn.exp2.f32(float %v1178), !dbg !32
  %v1180 = insertelement <4 x float> poison, float %v1179, i64 0, !dbg !34
  %local.reduce182 = fadd float %local.reduce178, %v1179, !dbg !33, !frisk.local_reduction !1
  %v1181 = extractelement <4 x float> %v965, i64 1, !dbg !32
  %v1182 = call float @llvm.amdgcn.exp2.f32(float %v1181), !dbg !32
  %v1183 = insertelement <4 x float> %v1180, float %v1182, i64 1, !dbg !34
  %local.reduce183 = fadd float %local.reduce179, %v1182, !dbg !33, !frisk.local_reduction !1
  %local.reduce186 = fadd float %local.reduce182, %local.reduce183, !dbg !33, !frisk.local_reduction !1
  %v1184 = extractelement <4 x float> %v965, i64 2, !dbg !32
  %v1185 = call float @llvm.amdgcn.exp2.f32(float %v1184), !dbg !32
  %v1186 = insertelement <4 x float> %v1183, float %v1185, i64 2, !dbg !34
  %local.reduce184 = fadd float %local.reduce180, %v1185, !dbg !33, !frisk.local_reduction !1
  %v1187 = extractelement <4 x float> %v965, i64 3, !dbg !32
  %v1188 = call float @llvm.amdgcn.exp2.f32(float %v1187), !dbg !32
  %v1189 = insertelement <4 x float> %v1186, float %v1188, i64 3, !dbg !34
  %math.result493 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %v1189) #9, !dbg !34, !frisk.register_lifetimes !1
  %v1190 = fptrunc <4 x float> %math.result493 to <4 x half>, !dbg !34
  %math.packed494 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %v1190) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce185 = fadd float %local.reduce181, %v1188, !dbg !33, !frisk.local_reduction !1
  %local.reduce187 = fadd float %local.reduce184, %local.reduce185, !dbg !33, !frisk.local_reduction !1
  %v1191 = fadd float %local.reduce186, %local.reduce187, !dbg !33, !frisk.local_reduction !1
  %v1200 = fadd float %v1086, %v973, !dbg !33
  %v1201 = insertelement <1 x float> poison, float %v1200, i64 0, !dbg !7
  %v1202 = insertvalue [2 x <1 x float>] poison, <1 x float> %v1201, 0, !dbg !7
  %v1211 = fadd float %v1191, %v981, !dbg !33
  %v1212 = insertelement <1 x float> poison, float %v1211, i64 0, !dbg !7
  %v1213 = insertvalue [2 x <1 x float>] %v1202, <1 x float> %v1212, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread1020 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !14, !range !39, !frisk.register_lifetimes !1
  %address.local1021 = shl nuw nsw i32 %address.thread1020, 3, !dbg !14
  %address.local1022 = and i32 %address.local1021, 120, !dbg !14
  %address.local1023 = shl nuw nsw i32 %address.thread1020, 5, !dbg !14
  %address.local1024 = and i32 %address.local1023, 1536, !dbg !14
  %address.local1025 = or disjoint i32 %address.local1022, %address.local1024, !dbg !14
  %address.local1026 = zext nneg i32 %address.local1025 to i64, !dbg !14
  %address.local1027 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1026, !dbg !14
  %v1214 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !14
  %v1215 = bitcast <4 x i32> %v1214 to <8 x half>, !dbg !14
  %address.local1028 = zext nneg i32 %address.thread1020 to i64, !dbg !8
  %address.local1029 = and i64 %address.local1028, 60, !dbg !14
  %address.local1030 = add nuw nsw i64 %address.local1029, %address.local1028, !dbg !14
  %address.local1031 = shl nuw nsw i64 %address.local1030, 3, !dbg !14
  %address.local1032 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1033 = and i64 %address.local1028, 63, !dbg !11
  %address.local1034 = lshr i64 %address.local1033, 4, !dbg !14
  %address.local1035 = mul nuw nsw i64 %address.local1034, 384, !dbg !14
  %address.local1036 = getelementptr [2 x i8], ptr addrspace(3) %address.local1032, i64 %address.local1035, !dbg !14
  %address.local1037 = lshr i64 %address.local1028, 2, !dbg !17
  %address.local1038 = mul nsw i64 %address.local1037, -32, !dbg !14
  %address.local1039 = getelementptr [2 x i8], ptr addrspace(3) %address.local1036, i64 %address.local1038, !dbg !14
  %address.local1040 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 256, !dbg !14
  %v1216 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !14
  %v1217 = bitcast <4 x i32> %v1216 to <8 x half>, !dbg !14
  %address.local1041 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1042 = getelementptr [2 x i8], ptr addrspace(3) %address.local1041, i64 %address.local1035, !dbg !14
  %address.local1043 = getelementptr [2 x i8], ptr addrspace(3) %address.local1042, i64 %address.local1038, !dbg !14
  %address.local1044 = getelementptr i8, ptr addrspace(3) %address.local1043, i64 512, !dbg !14
  %v1218 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !14
  %v1219 = bitcast <4 x i32> %v1218 to <8 x half>, !dbg !14
  %address.local1045 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1046 = getelementptr [2 x i8], ptr addrspace(3) %address.local1045, i64 %address.local1035, !dbg !14
  %address.local1047 = getelementptr [2 x i8], ptr addrspace(3) %address.local1046, i64 %address.local1038, !dbg !14
  %address.local1048 = getelementptr i8, ptr addrspace(3) %address.local1047, i64 768, !dbg !14
  %v1220 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !14
  %v1221 = bitcast <4 x i32> %v1220 to <8 x half>, !dbg !14
  %address.local1049 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1026, !dbg !14
  %address.local1050 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 4096, !dbg !14
  %v1222 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %v1223 = bitcast <4 x i32> %v1222 to <8 x half>, !dbg !14
  %address.local1051 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1052 = getelementptr [2 x i8], ptr addrspace(3) %address.local1051, i64 %address.local1035, !dbg !14
  %address.local1053 = getelementptr [2 x i8], ptr addrspace(3) %address.local1052, i64 %address.local1038, !dbg !14
  %address.local1054 = getelementptr i8, ptr addrspace(3) %address.local1053, i64 4352, !dbg !14
  %v1224 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %v1225 = bitcast <4 x i32> %v1224 to <8 x half>, !dbg !14
  %address.local1055 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1056 = getelementptr [2 x i8], ptr addrspace(3) %address.local1055, i64 %address.local1035, !dbg !14
  %address.local1057 = getelementptr [2 x i8], ptr addrspace(3) %address.local1056, i64 %address.local1038, !dbg !14
  %address.local1058 = getelementptr i8, ptr addrspace(3) %address.local1057, i64 4608, !dbg !14
  %v1226 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %v1227 = bitcast <4 x i32> %v1226 to <8 x half>, !dbg !14
  %address.local1059 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1060 = getelementptr [2 x i8], ptr addrspace(3) %address.local1059, i64 %address.local1035, !dbg !14
  %address.local1061 = getelementptr [2 x i8], ptr addrspace(3) %address.local1060, i64 %address.local1038, !dbg !14
  %address.local1062 = getelementptr i8, ptr addrspace(3) %address.local1061, i64 4864, !dbg !14
  %v1228 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #0, !dbg !14
  %v1229 = bitcast <4 x i32> %v1228 to <8 x half>, !dbg !14
  %v1230 = add nuw nsw i64 %v217, %v16, !dbg !18
  %v1231 = shl i64 %v1230, 19, !dbg !18
  %v1232 = and i64 %v1231, 16252928, !dbg !18
  %v1233 = add i64 %v1232, %v223, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1234 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %v1233, !dbg !18
  %v1235 = getelementptr i8, ptr addrspace(1) %v1234, i64 64, !dbg !18
  %address.local1063 = shl nuw nsw i64 %address.local1028, 3, !dbg !8
  %address.local1064 = and i64 %address.local1063, 1536, !dbg !8
  %address.local1065 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1064, !dbg !8
  %address.local1066 = ptrtoint ptr addrspace(3) %address.local1065 to i64, !dbg !8
  %address.local1067 = trunc i64 %address.local1066 to i32, !dbg !8
  %v1236 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1067), !dbg !18
  %v1237 = zext i32 %v1236 to i64, !dbg !18
  %v1238 = inttoptr i64 %v1237 to ptr addrspace(3), !dbg !18
  %address.local1068 = shl nuw nsw i64 %address.local1028, 5, !dbg !18
  %address.local1069 = and i64 %address.local1068, 8064, !dbg !18
  %address.local1070 = shl nuw nsw i64 %address.local1028, 3, !dbg !18
  %address.local1071 = and i64 %address.local1070, 24, !dbg !18
  %address.local1072 = or disjoint i64 %address.local1069, %address.local1071, !dbg !18
  %address.local1073 = getelementptr [2 x i8], ptr addrspace(1) %v1235, i64 %address.local1072, !dbg !18
  %v1239 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1240 = ptrtoint ptr addrspace(1) %address.local1073 to i64, !dbg !18
  %v1241 = sub i64 %v1240, %v1239, !dbg !18
  %dma.buffer.offset1714 = trunc i64 %v1241 to i32, !dbg !18
  %v1242 = trunc i64 %v1239 to i32, !dbg !18
  %v1243 = insertelement <4 x i32> poison, i32 %v1242, i64 0, !dbg !18
  %v1244 = lshr i64 %v1239, 32, !dbg !18
  %v1245 = trunc i64 %v1244 to i32, !dbg !18
  %v1246 = insertelement <4 x i32> %v1243, i32 %v1245, i64 1, !dbg !18
  %v1247 = insertelement <4 x i32> %v1246, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1715 = insertelement <4 x i32> %v1247, i32 131072, i64 3, !dbg !18
  %v1248 = ptrtoint ptr addrspace(3) %v1238 to i32, !dbg !18
  %dma.buffer.lds1716 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1248), !dbg !18
  %v1249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1716, i32 %dma.buffer.offset1714, <4 x i32> %dma.buffer.rsrc1715) #10, !dbg !18
  %address.local1074 = or disjoint i64 %address.local1028, 256, !dbg !8
  %address.local1075 = shl nuw nsw i64 %address.local1074, 3, !dbg !8
  %address.local1076 = and i64 %address.local1075, 3584, !dbg !8
  %address.local1077 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1076, !dbg !8
  %address.local1078 = ptrtoint ptr addrspace(3) %address.local1077 to i64, !dbg !8
  %address.local1079 = trunc i64 %address.local1078 to i32, !dbg !8
  %v1250 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1079), !dbg !18
  %v1251 = zext i32 %v1250 to i64, !dbg !18
  %v1252 = inttoptr i64 %v1251 to ptr addrspace(3), !dbg !18
  %address.local1080 = shl nuw nsw i64 %address.local1074, 5, !dbg !18
  %address.local1081 = and i64 %address.local1080, 16256, !dbg !18
  %address.local1082 = shl nuw nsw i64 %address.local1028, 3, !dbg !18
  %address.local1083 = and i64 %address.local1082, 24, !dbg !18
  %address.local1084 = or disjoint i64 %address.local1081, %address.local1083, !dbg !18
  %address.local1085 = getelementptr [2 x i8], ptr addrspace(1) %v1235, i64 %address.local1084, !dbg !18
  %v1253 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1254 = ptrtoint ptr addrspace(1) %address.local1085 to i64, !dbg !18
  %v1255 = sub i64 %v1254, %v1253, !dbg !18
  %dma.buffer.offset1711 = trunc i64 %v1255 to i32, !dbg !18
  %v1256 = trunc i64 %v1253 to i32, !dbg !18
  %v1257 = insertelement <4 x i32> poison, i32 %v1256, i64 0, !dbg !18
  %v1258 = lshr i64 %v1253, 32, !dbg !18
  %v1259 = trunc i64 %v1258 to i32, !dbg !18
  %v1260 = insertelement <4 x i32> %v1257, i32 %v1259, i64 1, !dbg !18
  %v1261 = insertelement <4 x i32> %v1260, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1712 = insertelement <4 x i32> %v1261, i32 131072, i64 3, !dbg !18
  %v1262 = ptrtoint ptr addrspace(3) %v1252 to i32, !dbg !18
  %dma.buffer.lds1713 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1262), !dbg !18
  %v1263 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1713, i32 %dma.buffer.offset1711, <4 x i32> %dma.buffer.rsrc1712) #10, !dbg !18
  %v1264 = shufflevector <8 x half> %v1215, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1265 = extractvalue [2 x <8 x float>] %v206, 0, !dbg !35
  %v1266 = shufflevector <8 x float> %v1265, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1267 = fmul <4 x float> %v1266, %v970, !dbg !35
  %v1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1264, <4 x float> %v1267) #9, !dbg !14
  %v1269 = shufflevector <8 x half> %v1217, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1269, <4 x float> %v1268) #9, !dbg !14
  %v1271 = shufflevector <8 x half> %v1219, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1271, <4 x float> %v1270) #9, !dbg !14
  %v1273 = shufflevector <8 x half> %v1221, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1273, <4 x float> %v1272) #9, !dbg !14
  %v1275 = shufflevector <8 x half> %v1223, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1275, <4 x float> %v1274) #9, !dbg !14
  %v1277 = shufflevector <8 x half> %v1225, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1277, <4 x float> %v1276) #9, !dbg !14
  %v1279 = shufflevector <8 x half> %v1227, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1279, <4 x float> %v1278) #9, !dbg !14
  %v1281 = shufflevector <8 x half> %v1229, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1281, <4 x float> %v1280) #9, !dbg !14
  %v1283 = shufflevector <8 x half> %v1215, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1284 = shufflevector <8 x float> %v1265, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1285 = fmul <4 x float> %v1284, %v970, !dbg !35
  %v1286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1283, <4 x float> %v1285) #9, !dbg !14
  %v1287 = shufflevector <8 x half> %v1217, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1287, <4 x float> %v1286) #9, !dbg !14
  %v1289 = shufflevector <8 x half> %v1219, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1289, <4 x float> %v1288) #9, !dbg !14
  %v1291 = shufflevector <8 x half> %v1221, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1291, <4 x float> %v1290) #9, !dbg !14
  %v1293 = shufflevector <8 x half> %v1223, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1293, <4 x float> %v1292) #9, !dbg !14
  %v1295 = shufflevector <8 x half> %v1225, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1295, <4 x float> %v1294) #9, !dbg !14
  %v1297 = shufflevector <8 x half> %v1227, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1297, <4 x float> %v1296) #9, !dbg !14
  %v1299 = shufflevector <8 x half> %v1229, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1299, <4 x float> %v1298) #9, !dbg !14
  %v1301 = extractvalue [2 x <8 x float>] %v206, 1, !dbg !35
  %v1302 = shufflevector <8 x float> %v1301, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1303 = fmul <4 x float> %v1302, %v978, !dbg !35
  %v1304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1264, <4 x float> %v1303) #9, !dbg !14
  %v1305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1269, <4 x float> %v1304) #9, !dbg !14
  %v1306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1271, <4 x float> %v1305) #9, !dbg !14
  %v1307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1273, <4 x float> %v1306) #9, !dbg !14
  %v1308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1275, <4 x float> %v1307) #9, !dbg !14
  %v1309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1277, <4 x float> %v1308) #9, !dbg !14
  %v1310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1279, <4 x float> %v1309) #9, !dbg !14
  %v1311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1281, <4 x float> %v1310) #9, !dbg !14
  %v1312 = shufflevector <8 x float> %v1301, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1313 = fmul <4 x float> %v1312, %v978, !dbg !35
  %v1314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1283, <4 x float> %v1313) #9, !dbg !14
  %v1315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1287, <4 x float> %v1314) #9, !dbg !14
  %v1316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1289, <4 x float> %v1315) #9, !dbg !14
  %v1317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1291, <4 x float> %v1316) #9, !dbg !14
  %v1318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1293, <4 x float> %v1317) #9, !dbg !14
  %v1319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1295, <4 x float> %v1318) #9, !dbg !14
  %v1320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1297, <4 x float> %v1319) #9, !dbg !14
  %v1321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1299, <4 x float> %v1320) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1322 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !14
  %v1323 = bitcast <4 x i32> %v1322 to <8 x half>, !dbg !14
  %v1324 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !14
  %v1325 = bitcast <4 x i32> %v1324 to <8 x half>, !dbg !14
  %v1326 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !14
  %v1327 = bitcast <4 x i32> %v1326 to <8 x half>, !dbg !14
  %v1328 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !14
  %v1329 = bitcast <4 x i32> %v1328 to <8 x half>, !dbg !14
  %v1330 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %v1331 = bitcast <4 x i32> %v1330 to <8 x half>, !dbg !14
  %v1332 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %v1333 = bitcast <4 x i32> %v1332 to <8 x half>, !dbg !14
  %v1334 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %v1335 = bitcast <4 x i32> %v1334 to <8 x half>, !dbg !14
  %v1336 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #0, !dbg !14
  %v1337 = bitcast <4 x i32> %v1336 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1338 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %v1233, !dbg !18
  %v1339 = getelementptr i8, ptr addrspace(1) %v1338, i64 128, !dbg !18
  %v1340 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1067), !dbg !18
  %v1341 = zext i32 %v1340 to i64, !dbg !18
  %v1342 = inttoptr i64 %v1341 to ptr addrspace(3), !dbg !18
  %address.local1086 = getelementptr [2 x i8], ptr addrspace(1) %v1339, i64 %address.local1072, !dbg !18
  %v1343 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1344 = ptrtoint ptr addrspace(1) %address.local1086 to i64, !dbg !18
  %v1345 = sub i64 %v1344, %v1343, !dbg !18
  %dma.buffer.offset1708 = trunc i64 %v1345 to i32, !dbg !18
  %v1346 = trunc i64 %v1343 to i32, !dbg !18
  %v1347 = insertelement <4 x i32> poison, i32 %v1346, i64 0, !dbg !18
  %v1348 = lshr i64 %v1343, 32, !dbg !18
  %v1349 = trunc i64 %v1348 to i32, !dbg !18
  %v1350 = insertelement <4 x i32> %v1347, i32 %v1349, i64 1, !dbg !18
  %v1351 = insertelement <4 x i32> %v1350, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1709 = insertelement <4 x i32> %v1351, i32 131072, i64 3, !dbg !18
  %v1352 = ptrtoint ptr addrspace(3) %v1342 to i32, !dbg !18
  %dma.buffer.lds1710 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1352), !dbg !18
  %v1353 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1710, i32 %dma.buffer.offset1708, <4 x i32> %dma.buffer.rsrc1709) #10, !dbg !18
  %v1354 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1079), !dbg !18
  %v1355 = zext i32 %v1354 to i64, !dbg !18
  %v1356 = inttoptr i64 %v1355 to ptr addrspace(3), !dbg !18
  %address.local1087 = getelementptr [2 x i8], ptr addrspace(1) %v1339, i64 %address.local1084, !dbg !18
  %v1357 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1358 = ptrtoint ptr addrspace(1) %address.local1087 to i64, !dbg !18
  %v1359 = sub i64 %v1358, %v1357, !dbg !18
  %dma.buffer.offset1705 = trunc i64 %v1359 to i32, !dbg !18
  %v1360 = trunc i64 %v1357 to i32, !dbg !18
  %v1361 = insertelement <4 x i32> poison, i32 %v1360, i64 0, !dbg !18
  %v1362 = lshr i64 %v1357, 32, !dbg !18
  %v1363 = trunc i64 %v1362 to i32, !dbg !18
  %v1364 = insertelement <4 x i32> %v1361, i32 %v1363, i64 1, !dbg !18
  %v1365 = insertelement <4 x i32> %v1364, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1706 = insertelement <4 x i32> %v1365, i32 131072, i64 3, !dbg !18
  %v1366 = ptrtoint ptr addrspace(3) %v1356 to i32, !dbg !18
  %dma.buffer.lds1707 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1366), !dbg !18
  %v1367 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1707, i32 %dma.buffer.offset1705, <4 x i32> %dma.buffer.rsrc1706) #10, !dbg !18
  %v1368 = shufflevector <8 x half> %v1323, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1369 = extractvalue [2 x <8 x float>] %v209, 0, !dbg !35
  %v1370 = shufflevector <8 x float> %v1369, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1371 = fmul <4 x float> %v1370, %v970, !dbg !35
  %v1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1368, <4 x float> %v1371) #9, !dbg !14
  %v1373 = shufflevector <8 x half> %v1325, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1373, <4 x float> %v1372) #9, !dbg !14
  %v1375 = shufflevector <8 x half> %v1327, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1375, <4 x float> %v1374) #9, !dbg !14
  %v1377 = shufflevector <8 x half> %v1329, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1377, <4 x float> %v1376) #9, !dbg !14
  %v1379 = shufflevector <8 x half> %v1331, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1379, <4 x float> %v1378) #9, !dbg !14
  %v1381 = shufflevector <8 x half> %v1333, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1381, <4 x float> %v1380) #9, !dbg !14
  %v1383 = shufflevector <8 x half> %v1335, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1383, <4 x float> %v1382) #9, !dbg !14
  %v1385 = shufflevector <8 x half> %v1337, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1385, <4 x float> %v1384) #9, !dbg !14
  %v1387 = shufflevector <8 x half> %v1323, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1388 = shufflevector <8 x float> %v1369, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1389 = fmul <4 x float> %v1388, %v970, !dbg !35
  %v1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1387, <4 x float> %v1389) #9, !dbg !14
  %v1391 = shufflevector <8 x half> %v1325, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1391, <4 x float> %v1390) #9, !dbg !14
  %v1393 = shufflevector <8 x half> %v1327, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1393, <4 x float> %v1392) #9, !dbg !14
  %v1395 = shufflevector <8 x half> %v1329, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1395, <4 x float> %v1394) #9, !dbg !14
  %v1397 = shufflevector <8 x half> %v1331, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1397, <4 x float> %v1396) #9, !dbg !14
  %v1399 = shufflevector <8 x half> %v1333, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1399, <4 x float> %v1398) #9, !dbg !14
  %v1401 = shufflevector <8 x half> %v1335, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1401, <4 x float> %v1400) #9, !dbg !14
  %v1403 = shufflevector <8 x half> %v1337, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1403, <4 x float> %v1402) #9, !dbg !14
  %v1405 = extractvalue [2 x <8 x float>] %v209, 1, !dbg !35
  %v1406 = shufflevector <8 x float> %v1405, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1407 = fmul <4 x float> %v1406, %v978, !dbg !35
  %v1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1368, <4 x float> %v1407) #9, !dbg !14
  %v1409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1373, <4 x float> %v1408) #9, !dbg !14
  %v1410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1375, <4 x float> %v1409) #9, !dbg !14
  %v1411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1377, <4 x float> %v1410) #9, !dbg !14
  %v1412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1379, <4 x float> %v1411) #9, !dbg !14
  %v1413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1381, <4 x float> %v1412) #9, !dbg !14
  %v1414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1383, <4 x float> %v1413) #9, !dbg !14
  %v1415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1385, <4 x float> %v1414) #9, !dbg !14
  %v1416 = shufflevector <8 x float> %v1405, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1417 = fmul <4 x float> %v1416, %v978, !dbg !35
  %v1418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1387, <4 x float> %v1417) #9, !dbg !14
  %v1419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1391, <4 x float> %v1418) #9, !dbg !14
  %v1420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1393, <4 x float> %v1419) #9, !dbg !14
  %v1421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1395, <4 x float> %v1420) #9, !dbg !14
  %v1422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1397, <4 x float> %v1421) #9, !dbg !14
  %v1423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1399, <4 x float> %v1422) #9, !dbg !14
  %v1424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1401, <4 x float> %v1423) #9, !dbg !14
  %v1425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1403, <4 x float> %v1424) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1426 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !14
  %v1427 = bitcast <4 x i32> %v1426 to <8 x half>, !dbg !14
  %v1428 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !14
  %v1429 = bitcast <4 x i32> %v1428 to <8 x half>, !dbg !14
  %v1430 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !14
  %v1431 = bitcast <4 x i32> %v1430 to <8 x half>, !dbg !14
  %v1432 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !14
  %v1433 = bitcast <4 x i32> %v1432 to <8 x half>, !dbg !14
  %v1434 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %v1435 = bitcast <4 x i32> %v1434 to <8 x half>, !dbg !14
  %v1436 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %v1437 = bitcast <4 x i32> %v1436 to <8 x half>, !dbg !14
  %v1438 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %v1439 = bitcast <4 x i32> %v1438 to <8 x half>, !dbg !14
  %v1440 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #0, !dbg !14
  %v1441 = bitcast <4 x i32> %v1440 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1442 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %v1233, !dbg !18
  %v1443 = getelementptr i8, ptr addrspace(1) %v1442, i64 192, !dbg !18
  %v1444 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1067), !dbg !18
  %v1445 = zext i32 %v1444 to i64, !dbg !18
  %v1446 = inttoptr i64 %v1445 to ptr addrspace(3), !dbg !18
  %address.local1088 = getelementptr [2 x i8], ptr addrspace(1) %v1443, i64 %address.local1072, !dbg !18
  %v1447 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1448 = ptrtoint ptr addrspace(1) %address.local1088 to i64, !dbg !18
  %v1449 = sub i64 %v1448, %v1447, !dbg !18
  %dma.buffer.offset1702 = trunc i64 %v1449 to i32, !dbg !18
  %v1450 = trunc i64 %v1447 to i32, !dbg !18
  %v1451 = insertelement <4 x i32> poison, i32 %v1450, i64 0, !dbg !18
  %v1452 = lshr i64 %v1447, 32, !dbg !18
  %v1453 = trunc i64 %v1452 to i32, !dbg !18
  %v1454 = insertelement <4 x i32> %v1451, i32 %v1453, i64 1, !dbg !18
  %v1455 = insertelement <4 x i32> %v1454, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1703 = insertelement <4 x i32> %v1455, i32 131072, i64 3, !dbg !18
  %v1456 = ptrtoint ptr addrspace(3) %v1446 to i32, !dbg !18
  %dma.buffer.lds1704 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1456), !dbg !18
  %v1457 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1704, i32 %dma.buffer.offset1702, <4 x i32> %dma.buffer.rsrc1703) #10, !dbg !18
  %v1458 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1079), !dbg !18
  %v1459 = zext i32 %v1458 to i64, !dbg !18
  %v1460 = inttoptr i64 %v1459 to ptr addrspace(3), !dbg !18
  %address.local1089 = getelementptr [2 x i8], ptr addrspace(1) %v1443, i64 %address.local1084, !dbg !18
  %v1461 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %v1462 = ptrtoint ptr addrspace(1) %address.local1089 to i64, !dbg !18
  %v1463 = sub i64 %v1462, %v1461, !dbg !18
  %dma.buffer.offset = trunc i64 %v1463 to i32, !dbg !18
  %v1464 = trunc i64 %v1461 to i32, !dbg !18
  %v1465 = insertelement <4 x i32> poison, i32 %v1464, i64 0, !dbg !18
  %v1466 = lshr i64 %v1461, 32, !dbg !18
  %v1467 = trunc i64 %v1466 to i32, !dbg !18
  %v1468 = insertelement <4 x i32> %v1465, i32 %v1467, i64 1, !dbg !18
  %v1469 = insertelement <4 x i32> %v1468, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %v1469, i32 131072, i64 3, !dbg !18
  %v1470 = ptrtoint ptr addrspace(3) %v1460 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v1470), !dbg !18
  %v1471 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %v1472 = shufflevector <8 x half> %v1427, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1473 = extractvalue [2 x <8 x float>] %v210, 0, !dbg !35
  %v1474 = shufflevector <8 x float> %v1473, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1475 = fmul <4 x float> %v1474, %v970, !dbg !35
  %v1476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1472, <4 x float> %v1475) #9, !dbg !14
  %v1477 = shufflevector <8 x half> %v1429, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1477, <4 x float> %v1476) #9, !dbg !14
  %v1479 = shufflevector <8 x half> %v1431, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1479, <4 x float> %v1478) #9, !dbg !14
  %v1481 = shufflevector <8 x half> %v1433, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1481, <4 x float> %v1480) #9, !dbg !14
  %v1483 = shufflevector <8 x half> %v1435, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1483, <4 x float> %v1482) #9, !dbg !14
  %v1485 = shufflevector <8 x half> %v1437, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1485, <4 x float> %v1484) #9, !dbg !14
  %v1487 = shufflevector <8 x half> %v1439, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1487, <4 x float> %v1486) #9, !dbg !14
  %v1489 = shufflevector <8 x half> %v1441, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1489, <4 x float> %v1488) #9, !dbg !14
  %v1491 = shufflevector <8 x half> %v1427, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1492 = shufflevector <8 x float> %v1473, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1493 = fmul <4 x float> %v1492, %v970, !dbg !35
  %v1494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1491, <4 x float> %v1493) #9, !dbg !14
  %v1495 = shufflevector <8 x half> %v1429, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1495, <4 x float> %v1494) #9, !dbg !14
  %v1497 = shufflevector <8 x half> %v1431, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1497, <4 x float> %v1496) #9, !dbg !14
  %v1499 = shufflevector <8 x half> %v1433, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1499, <4 x float> %v1498) #9, !dbg !14
  %v1501 = shufflevector <8 x half> %v1435, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1501, <4 x float> %v1500) #9, !dbg !14
  %v1503 = shufflevector <8 x half> %v1437, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1503, <4 x float> %v1502) #9, !dbg !14
  %v1505 = shufflevector <8 x half> %v1439, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1505, <4 x float> %v1504) #9, !dbg !14
  %v1507 = shufflevector <8 x half> %v1441, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1507, <4 x float> %v1506) #9, !dbg !14
  %v1509 = extractvalue [2 x <8 x float>] %v210, 1, !dbg !35
  %v1510 = shufflevector <8 x float> %v1509, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1511 = fmul <4 x float> %v1510, %v978, !dbg !35
  %v1512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1472, <4 x float> %v1511) #9, !dbg !14
  %v1513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1477, <4 x float> %v1512) #9, !dbg !14
  %v1514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1479, <4 x float> %v1513) #9, !dbg !14
  %v1515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1481, <4 x float> %v1514) #9, !dbg !14
  %v1516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1483, <4 x float> %v1515) #9, !dbg !14
  %v1517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1485, <4 x float> %v1516) #9, !dbg !14
  %v1518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1487, <4 x float> %v1517) #9, !dbg !14
  %v1519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1489, <4 x float> %v1518) #9, !dbg !14
  %v1520 = shufflevector <8 x float> %v1509, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1521 = fmul <4 x float> %v1520, %v978, !dbg !35
  %v1522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1491, <4 x float> %v1521) #9, !dbg !14
  %v1523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1495, <4 x float> %v1522) #9, !dbg !14
  %v1524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1497, <4 x float> %v1523) #9, !dbg !14
  %v1525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1499, <4 x float> %v1524) #9, !dbg !14
  %v1526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1501, <4 x float> %v1525) #9, !dbg !14
  %v1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1503, <4 x float> %v1526) #9, !dbg !14
  %v1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1505, <4 x float> %v1527) #9, !dbg !14
  %v1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1507, <4 x float> %v1528) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %v1530 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !14
  %v1531 = bitcast <4 x i32> %v1530 to <8 x half>, !dbg !14
  %v1532 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !14
  %v1533 = bitcast <4 x i32> %v1532 to <8 x half>, !dbg !14
  %v1534 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !14
  %v1535 = bitcast <4 x i32> %v1534 to <8 x half>, !dbg !14
  %v1536 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !14
  %v1537 = bitcast <4 x i32> %v1536 to <8 x half>, !dbg !14
  %v1538 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %v1539 = bitcast <4 x i32> %v1538 to <8 x half>, !dbg !14
  %v1540 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %v1541 = bitcast <4 x i32> %v1540 to <8 x half>, !dbg !14
  %v1542 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %v1543 = bitcast <4 x i32> %v1542 to <8 x half>, !dbg !14
  %v1544 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #0, !dbg !14
  %v1545 = bitcast <4 x i32> %v1544 to <8 x half>, !dbg !14
  %v1546 = shufflevector <8 x half> %v1531, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1547 = extractvalue [2 x <8 x float>] %v211, 0, !dbg !35
  %v1548 = shufflevector <8 x float> %v1547, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1549 = fmul <4 x float> %v1548, %v970, !dbg !35
  %v1550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1546, <4 x float> %v1549) #9, !dbg !14
  %v1551 = shufflevector <8 x half> %v1533, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1551, <4 x float> %v1550) #9, !dbg !14
  %v1553 = shufflevector <8 x half> %v1535, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1553, <4 x float> %v1552) #9, !dbg !14
  %v1555 = shufflevector <8 x half> %v1537, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1555, <4 x float> %v1554) #9, !dbg !14
  %v1557 = shufflevector <8 x half> %v1539, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1557, <4 x float> %v1556) #9, !dbg !14
  %v1559 = shufflevector <8 x half> %v1541, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1559, <4 x float> %v1558) #9, !dbg !14
  %v1561 = shufflevector <8 x half> %v1543, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1561, <4 x float> %v1560) #9, !dbg !14
  %v1563 = shufflevector <8 x half> %v1545, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %v1564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1563, <4 x float> %v1562) #9, !dbg !14
  %v1565 = shufflevector <8 x half> %v1531, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1566 = shufflevector <8 x float> %v1547, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1567 = fmul <4 x float> %v1566, %v970, !dbg !35
  %v1568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %v1565, <4 x float> %v1567) #9, !dbg !14
  %v1569 = shufflevector <8 x half> %v1533, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %v1569, <4 x float> %v1568) #9, !dbg !14
  %v1571 = shufflevector <8 x half> %v1535, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %v1571, <4 x float> %v1570) #9, !dbg !14
  %v1573 = shufflevector <8 x half> %v1537, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %v1573, <4 x float> %v1572) #9, !dbg !14
  %v1575 = shufflevector <8 x half> %v1539, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %v1575, <4 x float> %v1574) #9, !dbg !14
  %v1577 = shufflevector <8 x half> %v1541, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %v1577, <4 x float> %v1576) #9, !dbg !14
  %v1579 = shufflevector <8 x half> %v1543, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %v1579, <4 x float> %v1578) #9, !dbg !14
  %v1581 = shufflevector <8 x half> %v1545, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %v1582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %v1581, <4 x float> %v1580) #9, !dbg !14
  %v1583 = extractvalue [2 x <8 x float>] %v211, 1, !dbg !35
  %v1584 = shufflevector <8 x float> %v1583, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %v1585 = fmul <4 x float> %v1584, %v978, !dbg !35
  %v1586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1546, <4 x float> %v1585) #9, !dbg !14
  %v1587 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1551, <4 x float> %v1586) #9, !dbg !14
  %v1588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1553, <4 x float> %v1587) #9, !dbg !14
  %v1589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1555, <4 x float> %v1588) #9, !dbg !14
  %v1590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1557, <4 x float> %v1589) #9, !dbg !14
  %v1591 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1559, <4 x float> %v1590) #9, !dbg !14
  %v1592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1561, <4 x float> %v1591) #9, !dbg !14
  %v1593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1563, <4 x float> %v1592) #9, !dbg !14
  %v1594 = shufflevector <8 x float> %v1583, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %v1595 = fmul <4 x float> %v1594, %v978, !dbg !35
  %v1596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %v1565, <4 x float> %v1595) #9, !dbg !14
  %v1597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %v1569, <4 x float> %v1596) #9, !dbg !14
  %v1598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %v1571, <4 x float> %v1597) #9, !dbg !14
  %v1599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %v1573, <4 x float> %v1598) #9, !dbg !14
  %v1600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %v1575, <4 x float> %v1599) #9, !dbg !14
  %v1601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %v1577, <4 x float> %v1600) #9, !dbg !14
  %v1602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %v1579, <4 x float> %v1601) #9, !dbg !14
  %v1603 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %v1581, <4 x float> %v1602) #9, !dbg !14
  %v1604 = shufflevector <4 x float> %v1282, <4 x float> %v1300, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1605 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1604, 0, !dbg !7
  %v1606 = shufflevector <4 x float> %v1311, <4 x float> %v1321, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1607 = insertvalue [2 x <8 x float>] %v1605, <8 x float> %v1606, 1, !dbg !7
  %v1608 = insertvalue [2 x <1 x float>] poison, <1 x float> %v934, 0, !dbg !7
  %v1609 = insertvalue [2 x <1 x float>] %v1608, <1 x float> %v947, 1, !dbg !7
  %v1610 = shufflevector <4 x float> %v1386, <4 x float> %v1404, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1611 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1610, 0, !dbg !7
  %v1612 = shufflevector <4 x float> %v1415, <4 x float> %v1425, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1613 = insertvalue [2 x <8 x float>] %v1611, <8 x float> %v1612, 1, !dbg !7
  %v1614 = shufflevector <4 x float> %v1490, <4 x float> %v1508, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1615 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1614, 0, !dbg !7
  %v1616 = shufflevector <4 x float> %v1519, <4 x float> %v1529, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1617 = insertvalue [2 x <8 x float>] %v1615, <8 x float> %v1616, 1, !dbg !7
  %v1618 = shufflevector <4 x float> %v1564, <4 x float> %v1582, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1619 = insertvalue [2 x <8 x float>] poison, <8 x float> %v1618, 0, !dbg !7
  %v1620 = shufflevector <4 x float> %v1593, <4 x float> %v1603, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %v1621 = insertvalue [2 x <8 x float>] %v1619, <8 x float> %v1620, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %v1622 = add nuw nsw i64 %v205, 1, !dbg !7
  br label %v204, !dbg !7

v1623:                                             ; preds = %v204
  %v1624 = and i32 %v31, 16, !dbg !7
  %.not = icmp eq i32 %v1624, 0, !dbg !7
  %v1625 = and i32 %v31, 32, !dbg !7
  %.not36 = icmp eq i32 %v1625, 0, !dbg !7
  %v1626 = extractvalue [2 x <8 x float>] %v211, 0, !dbg !7
  %v1627 = extractelement <8 x float> %v1626, i64 0, !dbg !7
  %v1628 = extractelement <8 x float> %v1626, i64 1, !dbg !7
  %v1629 = extractelement <8 x float> %v1626, i64 2, !dbg !7
  %v1630 = extractelement <8 x float> %v1626, i64 3, !dbg !7
  %v1631 = select i1 %.not, float %v1628, float %v1627, !dbg !7
  %v1632 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %v1633 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %v1632), !dbg !7
  %v1634 = add nuw nsw i32 %v1633, 64, !dbg !7
  %v1635 = xor i32 %v1633, 16, !dbg !7
  %v1636 = and i32 %v1634, 192, !dbg !7
  %v1637 = icmp samesign ult i32 %v1635, %v1636, !dbg !7
  %v1638 = select i1 %v1637, i32 %v1635, i32 %v1633, !dbg !7
  %v1639 = shl nuw nsw i32 %v1638, 2, !dbg !7
  %v1640 = bitcast float %v1631 to i32, !dbg !7
  %v1641 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1640), !dbg !7
  %v1642 = bitcast i32 %v1641 to float, !dbg !7
  %v1643 = select i1 %.not, float %v1627, float %v1642, !dbg !7
  %v1644 = select i1 %.not, float %v1642, float %v1628, !dbg !7
  %v1645 = select i1 %.not, float %v1630, float %v1629, !dbg !7
  %v1646 = bitcast float %v1645 to i32, !dbg !7
  %v1647 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1646), !dbg !7
  %v1648 = bitcast i32 %v1647 to float, !dbg !7
  %v1649 = select i1 %.not, float %v1629, float %v1648, !dbg !7
  %v1650 = select i1 %.not, float %v1648, float %v1630, !dbg !7
  %v1651 = select i1 %.not36, float %v1649, float %v1643, !dbg !7
  %v1652 = xor i32 %v1633, 32, !dbg !7
  %v1653 = icmp samesign ult i32 %v1652, %v1636, !dbg !7
  %v1654 = select i1 %v1653, i32 %v1652, i32 %v1633, !dbg !7
  %v1655 = shl nuw nsw i32 %v1654, 2, !dbg !7
  %v1656 = bitcast float %v1651 to i32, !dbg !7
  %v1657 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1656), !dbg !7
  %v1658 = bitcast i32 %v1657 to float, !dbg !7
  %v1659 = select i1 %.not36, float %v1643, float %v1658, !dbg !7
  %v1660 = select i1 %.not36, float %v1658, float %v1649, !dbg !7
  %v1661 = select i1 %.not36, float %v1650, float %v1644, !dbg !7
  %v1662 = bitcast float %v1661 to i32, !dbg !7
  %v1663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1662), !dbg !7
  %v1664 = bitcast i32 %v1663 to float, !dbg !7
  %v1665 = select i1 %.not36, float %v1644, float %v1664, !dbg !7
  %v1666 = select i1 %.not36, float %v1664, float %v1650, !dbg !7
  %v1667 = extractvalue [2 x <8 x float>] %v210, 0, !dbg !7
  %v1668 = extractelement <8 x float> %v1667, i64 0, !dbg !7
  %v1669 = extractelement <8 x float> %v1667, i64 1, !dbg !7
  %v1670 = extractelement <8 x float> %v1667, i64 2, !dbg !7
  %v1671 = extractelement <8 x float> %v1667, i64 3, !dbg !7
  %v1672 = select i1 %.not, float %v1669, float %v1668, !dbg !7
  %v1673 = bitcast float %v1672 to i32, !dbg !7
  %v1674 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1673), !dbg !7
  %v1675 = bitcast i32 %v1674 to float, !dbg !7
  %v1676 = select i1 %.not, float %v1668, float %v1675, !dbg !7
  %v1677 = select i1 %.not, float %v1675, float %v1669, !dbg !7
  %v1678 = select i1 %.not, float %v1671, float %v1670, !dbg !7
  %v1679 = bitcast float %v1678 to i32, !dbg !7
  %v1680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1679), !dbg !7
  %v1681 = bitcast i32 %v1680 to float, !dbg !7
  %v1682 = select i1 %.not, float %v1670, float %v1681, !dbg !7
  %v1683 = select i1 %.not, float %v1681, float %v1671, !dbg !7
  %v1684 = select i1 %.not36, float %v1682, float %v1676, !dbg !7
  %v1685 = bitcast float %v1684 to i32, !dbg !7
  %v1686 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1685), !dbg !7
  %v1687 = bitcast i32 %v1686 to float, !dbg !7
  %v1688 = select i1 %.not36, float %v1676, float %v1687, !dbg !7
  %v1689 = select i1 %.not36, float %v1687, float %v1682, !dbg !7
  %v1690 = select i1 %.not36, float %v1683, float %v1677, !dbg !7
  %v1691 = bitcast float %v1690 to i32, !dbg !7
  %v1692 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1691), !dbg !7
  %v1693 = bitcast i32 %v1692 to float, !dbg !7
  %v1694 = select i1 %.not36, float %v1677, float %v1693, !dbg !7
  %v1695 = select i1 %.not36, float %v1693, float %v1683, !dbg !7
  %v1696 = extractvalue [2 x <8 x float>] %v209, 0, !dbg !7
  %v1697 = extractelement <8 x float> %v1696, i64 0, !dbg !7
  %v1698 = extractelement <8 x float> %v1696, i64 1, !dbg !7
  %v1699 = extractelement <8 x float> %v1696, i64 2, !dbg !7
  %v1700 = extractelement <8 x float> %v1696, i64 3, !dbg !7
  %v1701 = select i1 %.not, float %v1698, float %v1697, !dbg !7
  %v1702 = bitcast float %v1701 to i32, !dbg !7
  %v1703 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1702), !dbg !7
  %v1704 = bitcast i32 %v1703 to float, !dbg !7
  %v1705 = select i1 %.not, float %v1697, float %v1704, !dbg !7
  %v1706 = select i1 %.not, float %v1704, float %v1698, !dbg !7
  %v1707 = select i1 %.not, float %v1700, float %v1699, !dbg !7
  %v1708 = bitcast float %v1707 to i32, !dbg !7
  %v1709 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1708), !dbg !7
  %v1710 = bitcast i32 %v1709 to float, !dbg !7
  %v1711 = select i1 %.not, float %v1699, float %v1710, !dbg !7
  %v1712 = select i1 %.not, float %v1710, float %v1700, !dbg !7
  %v1713 = select i1 %.not36, float %v1711, float %v1705, !dbg !7
  %v1714 = bitcast float %v1713 to i32, !dbg !7
  %v1715 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1714), !dbg !7
  %v1716 = bitcast i32 %v1715 to float, !dbg !7
  %v1717 = select i1 %.not36, float %v1705, float %v1716, !dbg !7
  %v1718 = select i1 %.not36, float %v1716, float %v1711, !dbg !7
  %v1719 = select i1 %.not36, float %v1712, float %v1706, !dbg !7
  %v1720 = bitcast float %v1719 to i32, !dbg !7
  %v1721 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1720), !dbg !7
  %v1722 = bitcast i32 %v1721 to float, !dbg !7
  %v1723 = select i1 %.not36, float %v1706, float %v1722, !dbg !7
  %v1724 = select i1 %.not36, float %v1722, float %v1712, !dbg !7
  %v1725 = extractvalue [2 x <8 x float>] %v206, 0, !dbg !7
  %v1726 = extractelement <8 x float> %v1725, i64 0, !dbg !7
  %v1727 = extractelement <8 x float> %v1725, i64 1, !dbg !7
  %v1728 = extractelement <8 x float> %v1725, i64 2, !dbg !7
  %v1729 = extractelement <8 x float> %v1725, i64 3, !dbg !7
  %v1730 = select i1 %.not, float %v1727, float %v1726, !dbg !7
  %v1731 = bitcast float %v1730 to i32, !dbg !7
  %v1732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1731), !dbg !7
  %v1733 = bitcast i32 %v1732 to float, !dbg !7
  %v1734 = select i1 %.not, float %v1726, float %v1733, !dbg !7
  %v1735 = select i1 %.not, float %v1733, float %v1727, !dbg !7
  %v1736 = select i1 %.not, float %v1729, float %v1728, !dbg !7
  %v1737 = bitcast float %v1736 to i32, !dbg !7
  %v1738 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1737), !dbg !7
  %v1739 = bitcast i32 %v1738 to float, !dbg !7
  %v1740 = select i1 %.not, float %v1728, float %v1739, !dbg !7
  %v1741 = select i1 %.not, float %v1739, float %v1729, !dbg !7
  %v1742 = select i1 %.not36, float %v1740, float %v1734, !dbg !7
  %v1743 = bitcast float %v1742 to i32, !dbg !7
  %v1744 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1743), !dbg !7
  %v1745 = select i1 %.not36, float %v1741, float %v1735, !dbg !7
  %v1746 = bitcast float %v1745 to i32, !dbg !7
  %v1747 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1746), !dbg !7
  %denom.exit.r0.packed = extractvalue [2 x <1 x float>] %v207, 0, !dbg !19
  %denom.exit.r0.local = extractelement <1 x float> %denom.exit.r0.packed, i64 0, !dbg !19
  %denom.exit.r0.bits16 = bitcast float %denom.exit.r0.local to i32, !dbg !19
  %denom.exit.r0.peer16 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %denom.exit.r0.bits16), !dbg !19
  %denom.exit.r0.float16 = bitcast i32 %denom.exit.r0.peer16 to float, !dbg !19
  %denom.exit.r0.sum16 = fadd float %denom.exit.r0.local, %denom.exit.r0.float16, !dbg !19
  %denom.exit.r0.bits32 = bitcast float %denom.exit.r0.sum16 to i32, !dbg !19
  %denom.exit.r0.peer32 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %denom.exit.r0.bits32), !dbg !19
  %denom.exit.r0.float32 = bitcast i32 %denom.exit.r0.peer32 to float, !dbg !19
  %denom.exit.r0.sum32 = fadd float %denom.exit.r0.sum16, %denom.exit.r0.float32, !dbg !19
  %v1748 = insertelement <1 x float> poison, float %denom.exit.r0.sum32, i64 0, !dbg !19
  %v1749 = shufflevector <1 x float> %v1748, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %v1750 = extractelement <8 x float> %v1626, i64 4, !dbg !7
  %v1751 = extractelement <8 x float> %v1626, i64 5, !dbg !7
  %v1752 = extractelement <8 x float> %v1626, i64 6, !dbg !7
  %v1753 = extractelement <8 x float> %v1626, i64 7, !dbg !7
  %v1754 = select i1 %.not, float %v1751, float %v1750, !dbg !7
  %v1755 = bitcast float %v1754 to i32, !dbg !7
  %v1756 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1755), !dbg !7
  %v1757 = bitcast i32 %v1756 to float, !dbg !7
  %v1758 = select i1 %.not, float %v1750, float %v1757, !dbg !7
  %v1759 = select i1 %.not, float %v1757, float %v1751, !dbg !7
  %v1760 = select i1 %.not, float %v1753, float %v1752, !dbg !7
  %v1761 = bitcast float %v1760 to i32, !dbg !7
  %v1762 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1761), !dbg !7
  %v1763 = bitcast i32 %v1762 to float, !dbg !7
  %v1764 = select i1 %.not, float %v1752, float %v1763, !dbg !7
  %v1765 = select i1 %.not, float %v1763, float %v1753, !dbg !7
  %v1766 = select i1 %.not36, float %v1764, float %v1758, !dbg !7
  %v1767 = bitcast float %v1766 to i32, !dbg !7
  %v1768 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1767), !dbg !7
  %v1769 = bitcast i32 %v1768 to float, !dbg !7
  %v1770 = select i1 %.not36, float %v1758, float %v1769, !dbg !7
  %v1771 = select i1 %.not36, float %v1769, float %v1764, !dbg !7
  %v1772 = select i1 %.not36, float %v1765, float %v1759, !dbg !7
  %v1773 = bitcast float %v1772 to i32, !dbg !7
  %v1774 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1773), !dbg !7
  %v1775 = bitcast i32 %v1774 to float, !dbg !7
  %v1776 = select i1 %.not36, float %v1759, float %v1775, !dbg !7
  %v1777 = select i1 %.not36, float %v1775, float %v1765, !dbg !7
  %v1778 = extractelement <8 x float> %v1667, i64 4, !dbg !7
  %v1779 = extractelement <8 x float> %v1667, i64 5, !dbg !7
  %v1780 = extractelement <8 x float> %v1667, i64 6, !dbg !7
  %v1781 = extractelement <8 x float> %v1667, i64 7, !dbg !7
  %v1782 = select i1 %.not, float %v1779, float %v1778, !dbg !7
  %v1783 = bitcast float %v1782 to i32, !dbg !7
  %v1784 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1783), !dbg !7
  %v1785 = bitcast i32 %v1784 to float, !dbg !7
  %v1786 = select i1 %.not, float %v1778, float %v1785, !dbg !7
  %v1787 = select i1 %.not, float %v1785, float %v1779, !dbg !7
  %v1788 = select i1 %.not, float %v1781, float %v1780, !dbg !7
  %v1789 = bitcast float %v1788 to i32, !dbg !7
  %v1790 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1789), !dbg !7
  %v1791 = bitcast i32 %v1790 to float, !dbg !7
  %v1792 = select i1 %.not, float %v1780, float %v1791, !dbg !7
  %v1793 = select i1 %.not, float %v1791, float %v1781, !dbg !7
  %v1794 = select i1 %.not36, float %v1792, float %v1786, !dbg !7
  %v1795 = bitcast float %v1794 to i32, !dbg !7
  %v1796 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1795), !dbg !7
  %v1797 = bitcast i32 %v1796 to float, !dbg !7
  %v1798 = select i1 %.not36, float %v1786, float %v1797, !dbg !7
  %v1799 = select i1 %.not36, float %v1797, float %v1792, !dbg !7
  %v1800 = select i1 %.not36, float %v1793, float %v1787, !dbg !7
  %v1801 = bitcast float %v1800 to i32, !dbg !7
  %v1802 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1801), !dbg !7
  %v1803 = bitcast i32 %v1802 to float, !dbg !7
  %v1804 = select i1 %.not36, float %v1787, float %v1803, !dbg !7
  %v1805 = select i1 %.not36, float %v1803, float %v1793, !dbg !7
  %v1806 = extractelement <8 x float> %v1696, i64 4, !dbg !7
  %v1807 = extractelement <8 x float> %v1696, i64 5, !dbg !7
  %v1808 = extractelement <8 x float> %v1696, i64 6, !dbg !7
  %v1809 = extractelement <8 x float> %v1696, i64 7, !dbg !7
  %v1810 = select i1 %.not, float %v1807, float %v1806, !dbg !7
  %v1811 = bitcast float %v1810 to i32, !dbg !7
  %v1812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1811), !dbg !7
  %v1813 = bitcast i32 %v1812 to float, !dbg !7
  %v1814 = select i1 %.not, float %v1806, float %v1813, !dbg !7
  %v1815 = select i1 %.not, float %v1813, float %v1807, !dbg !7
  %v1816 = select i1 %.not, float %v1809, float %v1808, !dbg !7
  %v1817 = bitcast float %v1816 to i32, !dbg !7
  %v1818 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1817), !dbg !7
  %v1819 = bitcast i32 %v1818 to float, !dbg !7
  %v1820 = select i1 %.not, float %v1808, float %v1819, !dbg !7
  %v1821 = select i1 %.not, float %v1819, float %v1809, !dbg !7
  %v1822 = select i1 %.not36, float %v1820, float %v1814, !dbg !7
  %v1823 = bitcast float %v1822 to i32, !dbg !7
  %v1824 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1823), !dbg !7
  %v1825 = bitcast i32 %v1824 to float, !dbg !7
  %v1826 = select i1 %.not36, float %v1814, float %v1825, !dbg !7
  %v1827 = select i1 %.not36, float %v1825, float %v1820, !dbg !7
  %v1828 = select i1 %.not36, float %v1821, float %v1815, !dbg !7
  %v1829 = bitcast float %v1828 to i32, !dbg !7
  %v1830 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1829), !dbg !7
  %v1831 = bitcast i32 %v1830 to float, !dbg !7
  %v1832 = select i1 %.not36, float %v1815, float %v1831, !dbg !7
  %v1833 = select i1 %.not36, float %v1831, float %v1821, !dbg !7
  %v1834 = extractelement <8 x float> %v1725, i64 4, !dbg !7
  %v1835 = extractelement <8 x float> %v1725, i64 5, !dbg !7
  %v1836 = extractelement <8 x float> %v1725, i64 6, !dbg !7
  %v1837 = extractelement <8 x float> %v1725, i64 7, !dbg !7
  %v1838 = select i1 %.not, float %v1835, float %v1834, !dbg !7
  %v1839 = bitcast float %v1838 to i32, !dbg !7
  %v1840 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1839), !dbg !7
  %v1841 = bitcast i32 %v1840 to float, !dbg !7
  %v1842 = select i1 %.not, float %v1834, float %v1841, !dbg !7
  %v1843 = select i1 %.not, float %v1841, float %v1835, !dbg !7
  %v1844 = select i1 %.not, float %v1837, float %v1836, !dbg !7
  %v1845 = bitcast float %v1844 to i32, !dbg !7
  %v1846 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1845), !dbg !7
  %v1847 = bitcast i32 %v1846 to float, !dbg !7
  %v1848 = select i1 %.not, float %v1836, float %v1847, !dbg !7
  %v1849 = select i1 %.not, float %v1847, float %v1837, !dbg !7
  %v1850 = select i1 %.not36, float %v1848, float %v1842, !dbg !7
  %v1851 = bitcast float %v1850 to i32, !dbg !7
  %v1852 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1851), !dbg !7
  %v1853 = select i1 %.not36, float %v1849, float %v1843, !dbg !7
  %v1854 = bitcast float %v1853 to i32, !dbg !7
  %v1855 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1854), !dbg !7
  %v1856 = extractvalue [2 x <8 x float>] %v211, 1, !dbg !7
  %v1857 = extractelement <8 x float> %v1856, i64 0, !dbg !7
  %v1858 = extractelement <8 x float> %v1856, i64 1, !dbg !7
  %v1859 = extractelement <8 x float> %v1856, i64 2, !dbg !7
  %v1860 = extractelement <8 x float> %v1856, i64 3, !dbg !7
  %v1861 = select i1 %.not, float %v1858, float %v1857, !dbg !7
  %v1862 = bitcast float %v1861 to i32, !dbg !7
  %v1863 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1862), !dbg !7
  %v1864 = bitcast i32 %v1863 to float, !dbg !7
  %v1865 = select i1 %.not, float %v1857, float %v1864, !dbg !7
  %v1866 = select i1 %.not, float %v1864, float %v1858, !dbg !7
  %v1867 = select i1 %.not, float %v1860, float %v1859, !dbg !7
  %v1868 = bitcast float %v1867 to i32, !dbg !7
  %v1869 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1868), !dbg !7
  %v1870 = bitcast i32 %v1869 to float, !dbg !7
  %v1871 = select i1 %.not, float %v1859, float %v1870, !dbg !7
  %v1872 = select i1 %.not, float %v1870, float %v1860, !dbg !7
  %v1873 = select i1 %.not36, float %v1871, float %v1865, !dbg !7
  %v1874 = bitcast float %v1873 to i32, !dbg !7
  %v1875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1874), !dbg !7
  %v1876 = bitcast i32 %v1875 to float, !dbg !7
  %v1877 = select i1 %.not36, float %v1865, float %v1876, !dbg !7
  %v1878 = select i1 %.not36, float %v1876, float %v1871, !dbg !7
  %v1879 = select i1 %.not36, float %v1872, float %v1866, !dbg !7
  %v1880 = bitcast float %v1879 to i32, !dbg !7
  %v1881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1880), !dbg !7
  %v1882 = bitcast i32 %v1881 to float, !dbg !7
  %v1883 = select i1 %.not36, float %v1866, float %v1882, !dbg !7
  %v1884 = select i1 %.not36, float %v1882, float %v1872, !dbg !7
  %v1885 = extractvalue [2 x <8 x float>] %v210, 1, !dbg !7
  %v1886 = extractelement <8 x float> %v1885, i64 0, !dbg !7
  %v1887 = extractelement <8 x float> %v1885, i64 1, !dbg !7
  %v1888 = extractelement <8 x float> %v1885, i64 2, !dbg !7
  %v1889 = extractelement <8 x float> %v1885, i64 3, !dbg !7
  %v1890 = select i1 %.not, float %v1887, float %v1886, !dbg !7
  %v1891 = bitcast float %v1890 to i32, !dbg !7
  %v1892 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1891), !dbg !7
  %v1893 = bitcast i32 %v1892 to float, !dbg !7
  %v1894 = select i1 %.not, float %v1886, float %v1893, !dbg !7
  %v1895 = select i1 %.not, float %v1893, float %v1887, !dbg !7
  %v1896 = select i1 %.not, float %v1889, float %v1888, !dbg !7
  %v1897 = bitcast float %v1896 to i32, !dbg !7
  %v1898 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1897), !dbg !7
  %v1899 = bitcast i32 %v1898 to float, !dbg !7
  %v1900 = select i1 %.not, float %v1888, float %v1899, !dbg !7
  %v1901 = select i1 %.not, float %v1899, float %v1889, !dbg !7
  %v1902 = select i1 %.not36, float %v1900, float %v1894, !dbg !7
  %v1903 = bitcast float %v1902 to i32, !dbg !7
  %v1904 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1903), !dbg !7
  %v1905 = bitcast i32 %v1904 to float, !dbg !7
  %v1906 = select i1 %.not36, float %v1894, float %v1905, !dbg !7
  %v1907 = select i1 %.not36, float %v1905, float %v1900, !dbg !7
  %v1908 = select i1 %.not36, float %v1901, float %v1895, !dbg !7
  %v1909 = bitcast float %v1908 to i32, !dbg !7
  %v1910 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1909), !dbg !7
  %v1911 = bitcast i32 %v1910 to float, !dbg !7
  %v1912 = select i1 %.not36, float %v1895, float %v1911, !dbg !7
  %v1913 = select i1 %.not36, float %v1911, float %v1901, !dbg !7
  %v1914 = extractvalue [2 x <8 x float>] %v209, 1, !dbg !7
  %v1915 = extractelement <8 x float> %v1914, i64 0, !dbg !7
  %v1916 = extractelement <8 x float> %v1914, i64 1, !dbg !7
  %v1917 = extractelement <8 x float> %v1914, i64 2, !dbg !7
  %v1918 = extractelement <8 x float> %v1914, i64 3, !dbg !7
  %v1919 = select i1 %.not, float %v1916, float %v1915, !dbg !7
  %v1920 = bitcast float %v1919 to i32, !dbg !7
  %v1921 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1920), !dbg !7
  %v1922 = bitcast i32 %v1921 to float, !dbg !7
  %v1923 = select i1 %.not, float %v1915, float %v1922, !dbg !7
  %v1924 = select i1 %.not, float %v1922, float %v1916, !dbg !7
  %v1925 = select i1 %.not, float %v1918, float %v1917, !dbg !7
  %v1926 = bitcast float %v1925 to i32, !dbg !7
  %v1927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1926), !dbg !7
  %v1928 = bitcast i32 %v1927 to float, !dbg !7
  %v1929 = select i1 %.not, float %v1917, float %v1928, !dbg !7
  %v1930 = select i1 %.not, float %v1928, float %v1918, !dbg !7
  %v1931 = select i1 %.not36, float %v1929, float %v1923, !dbg !7
  %v1932 = bitcast float %v1931 to i32, !dbg !7
  %v1933 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1932), !dbg !7
  %v1934 = bitcast i32 %v1933 to float, !dbg !7
  %v1935 = select i1 %.not36, float %v1923, float %v1934, !dbg !7
  %v1936 = select i1 %.not36, float %v1934, float %v1929, !dbg !7
  %v1937 = select i1 %.not36, float %v1930, float %v1924, !dbg !7
  %v1938 = bitcast float %v1937 to i32, !dbg !7
  %v1939 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1938), !dbg !7
  %v1940 = bitcast i32 %v1939 to float, !dbg !7
  %v1941 = select i1 %.not36, float %v1924, float %v1940, !dbg !7
  %v1942 = select i1 %.not36, float %v1940, float %v1930, !dbg !7
  %v1943 = extractvalue [2 x <8 x float>] %v206, 1, !dbg !7
  %v1944 = extractelement <8 x float> %v1943, i64 0, !dbg !7
  %v1945 = extractelement <8 x float> %v1943, i64 1, !dbg !7
  %v1946 = extractelement <8 x float> %v1943, i64 2, !dbg !7
  %v1947 = extractelement <8 x float> %v1943, i64 3, !dbg !7
  %v1948 = select i1 %.not, float %v1945, float %v1944, !dbg !7
  %v1949 = bitcast float %v1948 to i32, !dbg !7
  %v1950 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1949), !dbg !7
  %v1951 = bitcast i32 %v1950 to float, !dbg !7
  %v1952 = select i1 %.not, float %v1944, float %v1951, !dbg !7
  %v1953 = select i1 %.not, float %v1951, float %v1945, !dbg !7
  %v1954 = select i1 %.not, float %v1947, float %v1946, !dbg !7
  %v1955 = bitcast float %v1954 to i32, !dbg !7
  %v1956 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1955), !dbg !7
  %v1957 = bitcast i32 %v1956 to float, !dbg !7
  %v1958 = select i1 %.not, float %v1946, float %v1957, !dbg !7
  %v1959 = select i1 %.not, float %v1957, float %v1947, !dbg !7
  %v1960 = select i1 %.not36, float %v1958, float %v1952, !dbg !7
  %v1961 = bitcast float %v1960 to i32, !dbg !7
  %v1962 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1961), !dbg !7
  %v1963 = select i1 %.not36, float %v1959, float %v1953, !dbg !7
  %v1964 = bitcast float %v1963 to i32, !dbg !7
  %v1965 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1964), !dbg !7
  %denom.exit.r1.packed = extractvalue [2 x <1 x float>] %v207, 1, !dbg !19
  %denom.exit.r1.local = extractelement <1 x float> %denom.exit.r1.packed, i64 0, !dbg !19
  %denom.exit.r1.bits16 = bitcast float %denom.exit.r1.local to i32, !dbg !19
  %denom.exit.r1.peer16 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %denom.exit.r1.bits16), !dbg !19
  %denom.exit.r1.float16 = bitcast i32 %denom.exit.r1.peer16 to float, !dbg !19
  %denom.exit.r1.sum16 = fadd float %denom.exit.r1.local, %denom.exit.r1.float16, !dbg !19
  %denom.exit.r1.bits32 = bitcast float %denom.exit.r1.sum16 to i32, !dbg !19
  %denom.exit.r1.peer32 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %denom.exit.r1.bits32), !dbg !19
  %denom.exit.r1.float32 = bitcast i32 %denom.exit.r1.peer32 to float, !dbg !19
  %denom.exit.r1.sum32 = fadd float %denom.exit.r1.sum16, %denom.exit.r1.float32, !dbg !19
  %v1966 = insertelement <1 x float> poison, float %denom.exit.r1.sum32, i64 0, !dbg !19
  %v1967 = shufflevector <1 x float> %v1966, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %v1968 = extractelement <8 x float> %v1856, i64 4, !dbg !7
  %v1969 = extractelement <8 x float> %v1856, i64 5, !dbg !7
  %v1970 = extractelement <8 x float> %v1856, i64 6, !dbg !7
  %v1971 = extractelement <8 x float> %v1856, i64 7, !dbg !7
  %v1972 = select i1 %.not, float %v1969, float %v1968, !dbg !7
  %v1973 = bitcast float %v1972 to i32, !dbg !7
  %v1974 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1973), !dbg !7
  %v1975 = bitcast i32 %v1974 to float, !dbg !7
  %v1976 = select i1 %.not, float %v1968, float %v1975, !dbg !7
  %v1977 = select i1 %.not, float %v1975, float %v1969, !dbg !7
  %v1978 = select i1 %.not, float %v1971, float %v1970, !dbg !7
  %v1979 = bitcast float %v1978 to i32, !dbg !7
  %v1980 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v1979), !dbg !7
  %v1981 = bitcast i32 %v1980 to float, !dbg !7
  %v1982 = select i1 %.not, float %v1970, float %v1981, !dbg !7
  %v1983 = select i1 %.not, float %v1981, float %v1971, !dbg !7
  %v1984 = select i1 %.not36, float %v1982, float %v1976, !dbg !7
  %v1985 = bitcast float %v1984 to i32, !dbg !7
  %v1986 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1985), !dbg !7
  %v1987 = bitcast i32 %v1986 to float, !dbg !7
  %v1988 = select i1 %.not36, float %v1976, float %v1987, !dbg !7
  %v1989 = select i1 %.not36, float %v1987, float %v1982, !dbg !7
  %v1990 = select i1 %.not36, float %v1983, float %v1977, !dbg !7
  %v1991 = bitcast float %v1990 to i32, !dbg !7
  %v1992 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v1991), !dbg !7
  %v1993 = bitcast i32 %v1992 to float, !dbg !7
  %v1994 = select i1 %.not36, float %v1977, float %v1993, !dbg !7
  %v1995 = select i1 %.not36, float %v1993, float %v1983, !dbg !7
  %v1996 = extractelement <8 x float> %v1885, i64 4, !dbg !7
  %v1997 = extractelement <8 x float> %v1885, i64 5, !dbg !7
  %v1998 = extractelement <8 x float> %v1885, i64 6, !dbg !7
  %v1999 = extractelement <8 x float> %v1885, i64 7, !dbg !7
  %v2000 = select i1 %.not, float %v1997, float %v1996, !dbg !7
  %v2001 = bitcast float %v2000 to i32, !dbg !7
  %v2002 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2001), !dbg !7
  %v2003 = bitcast i32 %v2002 to float, !dbg !7
  %v2004 = select i1 %.not, float %v1996, float %v2003, !dbg !7
  %v2005 = select i1 %.not, float %v2003, float %v1997, !dbg !7
  %v2006 = select i1 %.not, float %v1999, float %v1998, !dbg !7
  %v2007 = bitcast float %v2006 to i32, !dbg !7
  %v2008 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2007), !dbg !7
  %v2009 = bitcast i32 %v2008 to float, !dbg !7
  %v2010 = select i1 %.not, float %v1998, float %v2009, !dbg !7
  %v2011 = select i1 %.not, float %v2009, float %v1999, !dbg !7
  %v2012 = select i1 %.not36, float %v2010, float %v2004, !dbg !7
  %v2013 = bitcast float %v2012 to i32, !dbg !7
  %v2014 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2013), !dbg !7
  %v2015 = bitcast i32 %v2014 to float, !dbg !7
  %v2016 = select i1 %.not36, float %v2004, float %v2015, !dbg !7
  %v2017 = select i1 %.not36, float %v2015, float %v2010, !dbg !7
  %v2018 = select i1 %.not36, float %v2011, float %v2005, !dbg !7
  %v2019 = bitcast float %v2018 to i32, !dbg !7
  %v2020 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2019), !dbg !7
  %v2021 = bitcast i32 %v2020 to float, !dbg !7
  %v2022 = select i1 %.not36, float %v2005, float %v2021, !dbg !7
  %v2023 = select i1 %.not36, float %v2021, float %v2011, !dbg !7
  %v2024 = extractelement <8 x float> %v1914, i64 4, !dbg !7
  %v2025 = extractelement <8 x float> %v1914, i64 5, !dbg !7
  %v2026 = extractelement <8 x float> %v1914, i64 6, !dbg !7
  %v2027 = extractelement <8 x float> %v1914, i64 7, !dbg !7
  %v2028 = select i1 %.not, float %v2025, float %v2024, !dbg !7
  %v2029 = bitcast float %v2028 to i32, !dbg !7
  %v2030 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2029), !dbg !7
  %v2031 = bitcast i32 %v2030 to float, !dbg !7
  %v2032 = select i1 %.not, float %v2024, float %v2031, !dbg !7
  %v2033 = select i1 %.not, float %v2031, float %v2025, !dbg !7
  %v2034 = select i1 %.not, float %v2027, float %v2026, !dbg !7
  %v2035 = bitcast float %v2034 to i32, !dbg !7
  %v2036 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2035), !dbg !7
  %v2037 = bitcast i32 %v2036 to float, !dbg !7
  %v2038 = select i1 %.not, float %v2026, float %v2037, !dbg !7
  %v2039 = select i1 %.not, float %v2037, float %v2027, !dbg !7
  %v2040 = select i1 %.not36, float %v2038, float %v2032, !dbg !7
  %v2041 = bitcast float %v2040 to i32, !dbg !7
  %v2042 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2041), !dbg !7
  %v2043 = bitcast i32 %v2042 to float, !dbg !7
  %v2044 = select i1 %.not36, float %v2032, float %v2043, !dbg !7
  %v2045 = select i1 %.not36, float %v2043, float %v2038, !dbg !7
  %v2046 = select i1 %.not36, float %v2039, float %v2033, !dbg !7
  %v2047 = bitcast float %v2046 to i32, !dbg !7
  %v2048 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2047), !dbg !7
  %v2049 = bitcast i32 %v2048 to float, !dbg !7
  %v2050 = select i1 %.not36, float %v2033, float %v2049, !dbg !7
  %v2051 = select i1 %.not36, float %v2049, float %v2039, !dbg !7
  %v2052 = extractelement <8 x float> %v1943, i64 4, !dbg !7
  %v2053 = extractelement <8 x float> %v1943, i64 5, !dbg !7
  %v2054 = extractelement <8 x float> %v1943, i64 6, !dbg !7
  %v2055 = extractelement <8 x float> %v1943, i64 7, !dbg !7
  %v2056 = select i1 %.not, float %v2053, float %v2052, !dbg !7
  %v2057 = bitcast float %v2056 to i32, !dbg !7
  %v2058 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2057), !dbg !7
  %v2059 = bitcast i32 %v2058 to float, !dbg !7
  %v2060 = select i1 %.not, float %v2052, float %v2059, !dbg !7
  %v2061 = select i1 %.not, float %v2059, float %v2053, !dbg !7
  %v2062 = select i1 %.not, float %v2055, float %v2054, !dbg !7
  %v2063 = bitcast float %v2062 to i32, !dbg !7
  %v2064 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1639, i32 %v2063), !dbg !7
  %v2065 = bitcast i32 %v2064 to float, !dbg !7
  %v2066 = select i1 %.not, float %v2054, float %v2065, !dbg !7
  %v2067 = select i1 %.not, float %v2065, float %v2055, !dbg !7
  %v2068 = select i1 %.not36, float %v2066, float %v2060, !dbg !7
  %v2069 = bitcast float %v2068 to i32, !dbg !7
  %v2070 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2069), !dbg !7
  %v2071 = select i1 %.not36, float %v2067, float %v2061, !dbg !7
  %v2072 = bitcast float %v2071 to i32, !dbg !7
  %v2073 = call i32 @llvm.amdgcn.ds.bpermute(i32 %v1655, i32 %v2072), !dbg !7
  br i1 true, label %v2074, label %v2138, !dbg !21

v2074:                                             ; preds = %v1623
  %v2075 = bitcast i32 %v2070 to float, !dbg !7
  %v2076 = select i1 %.not36, float %v2060, float %v2075, !dbg !7
  %v2077 = insertelement <4 x float> poison, float %v2076, i64 0, !dbg !19
  %v2078 = bitcast i32 %v2073 to float, !dbg !7
  %v2079 = select i1 %.not36, float %v2061, float %v2078, !dbg !7
  %v2080 = insertelement <4 x float> %v2077, float %v2079, i64 1, !dbg !19
  %v2081 = select i1 %.not36, float %v2075, float %v2066, !dbg !7
  %v2082 = insertelement <4 x float> %v2080, float %v2081, i64 2, !dbg !19
  %v2083 = select i1 %.not36, float %v2078, float %v2067, !dbg !7
  %v2084 = insertelement <4 x float> %v2082, float %v2083, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1967, !dbg !19
  %normalized = fmul <4 x float> %v2084, %row.reciprocal, !dbg !19
  %v2085 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %v2086 = extractelement <4 x half> %v2085, i64 3, !dbg !20
  %v2087 = extractelement <4 x half> %v2085, i64 2, !dbg !20
  %v2088 = extractelement <4 x half> %v2085, i64 1, !dbg !20
  %v2089 = extractelement <4 x half> %v2085, i64 0, !dbg !20
  %v2090 = bitcast i32 %v1962 to float, !dbg !7
  %v2091 = select i1 %.not36, float %v1952, float %v2090, !dbg !7
  %v2092 = insertelement <4 x float> poison, float %v2091, i64 0, !dbg !19
  %v2093 = bitcast i32 %v1965 to float, !dbg !7
  %v2094 = select i1 %.not36, float %v1953, float %v2093, !dbg !7
  %v2095 = insertelement <4 x float> %v2092, float %v2094, i64 1, !dbg !19
  %v2096 = select i1 %.not36, float %v2090, float %v1958, !dbg !7
  %v2097 = insertelement <4 x float> %v2095, float %v2096, i64 2, !dbg !19
  %v2098 = select i1 %.not36, float %v2093, float %v1959, !dbg !7
  %v2099 = insertelement <4 x float> %v2097, float %v2098, i64 3, !dbg !19
  %normalized104 = fmul <4 x float> %v2099, %row.reciprocal, !dbg !19
  %v2100 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %v2101 = extractelement <4 x half> %v2100, i64 3, !dbg !20
  %v2102 = extractelement <4 x half> %v2100, i64 2, !dbg !20
  %v2103 = extractelement <4 x half> %v2100, i64 1, !dbg !20
  %v2104 = extractelement <4 x half> %v2100, i64 0, !dbg !20
  %v2105 = bitcast i32 %v1852 to float, !dbg !7
  %v2106 = select i1 %.not36, float %v1842, float %v2105, !dbg !7
  %v2107 = insertelement <4 x float> poison, float %v2106, i64 0, !dbg !19
  %v2108 = bitcast i32 %v1855 to float, !dbg !7
  %v2109 = select i1 %.not36, float %v1843, float %v2108, !dbg !7
  %v2110 = insertelement <4 x float> %v2107, float %v2109, i64 1, !dbg !19
  %v2111 = select i1 %.not36, float %v2105, float %v1848, !dbg !7
  %v2112 = insertelement <4 x float> %v2110, float %v2111, i64 2, !dbg !19
  %v2113 = select i1 %.not36, float %v2108, float %v1849, !dbg !7
  %v2114 = insertelement <4 x float> %v2112, float %v2113, i64 3, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1749, !dbg !19
  %normalized106 = fmul <4 x float> %v2114, %row.reciprocal105, !dbg !19
  %v2115 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %v2116 = extractelement <4 x half> %v2115, i64 3, !dbg !20
  %v2117 = extractelement <4 x half> %v2115, i64 2, !dbg !20
  %v2118 = extractelement <4 x half> %v2115, i64 1, !dbg !20
  %v2119 = extractelement <4 x half> %v2115, i64 0, !dbg !20
  %v2120 = bitcast i32 %v1744 to float, !dbg !7
  %v2121 = select i1 %.not36, float %v1734, float %v2120, !dbg !7
  %v2122 = insertelement <4 x float> poison, float %v2121, i64 0, !dbg !19
  %v2123 = bitcast i32 %v1747 to float, !dbg !7
  %v2124 = select i1 %.not36, float %v1735, float %v2123, !dbg !7
  %v2125 = insertelement <4 x float> %v2122, float %v2124, i64 1, !dbg !19
  %v2126 = select i1 %.not36, float %v2120, float %v1740, !dbg !7
  %v2127 = insertelement <4 x float> %v2125, float %v2126, i64 2, !dbg !19
  %v2128 = select i1 %.not36, float %v2123, float %v1741, !dbg !7
  %v2129 = insertelement <4 x float> %v2127, float %v2128, i64 3, !dbg !19
  %normalized107 = fmul <4 x float> %v2129, %row.reciprocal105, !dbg !19
  %v2130 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %v2131 = extractelement <4 x half> %v2130, i64 3, !dbg !20
  %v2132 = extractelement <4 x half> %v2130, i64 2, !dbg !20
  %v2133 = extractelement <4 x half> %v2130, i64 1, !dbg !20
  %v2134 = extractelement <4 x half> %v2130, i64 0, !dbg !20
  %v2135 = add nsw i64 %v16, %v22, !dbg !21
  %v2136 = shl i64 %v2135, 19, !dbg !21
  %v2137 = and i64 %v2136, 16252928, !dbg !21
  %address.thread1090 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1091 = zext nneg i32 %address.thread1090 to i64, !dbg !8
  %address.local1092 = add i64 %v25, %address.local1091, !dbg !21
  %address.local1093 = lshr i64 %address.local1091, 6, !dbg !11
  %address.local1094 = shl nuw nsw i64 %address.local1093, 4, !dbg !21
  %address.local1095 = add i64 %address.local1092, %address.local1094, !dbg !21
  %address.local1096 = and i64 %address.local1091, 240, !dbg !21
  %address.local1097 = sub i64 %address.local1095, %address.local1096, !dbg !21
  %address.local1098 = add i64 %address.local1097, %v26, !dbg !21
  %address.local1099 = shl nuw nsw i64 %address.local1098, 7, !dbg !21
  %address.local1100 = add nuw nsw i64 %v2137, %address.local1099, !dbg !21
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1102 = lshr i64 %address.local1091, 2, !dbg !11
  %address.local1103 = and i64 %address.local1102, 12, !dbg !11
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1103, !dbg !21
  store half %v2134, ptr addrspace(1) %address.local1104, align 2, !dbg !21
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1106 = or disjoint i64 %address.local1103, 1, !dbg !21
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1105, i64 %address.local1106, !dbg !21
  store half %v2133, ptr addrspace(1) %address.local1107, align 2, !dbg !21
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1109 = or disjoint i64 %address.local1103, 2, !dbg !21
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1108, i64 %address.local1109, !dbg !21
  store half %v2132, ptr addrspace(1) %address.local1110, align 2, !dbg !21
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1112 = or disjoint i64 %address.local1103, 3, !dbg !21
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1112, !dbg !21
  store half %v2131, ptr addrspace(1) %address.local1113, align 2, !dbg !21
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1115 = or disjoint i64 %address.local1103, 16, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1114, i64 %address.local1115, !dbg !21
  store half %v2119, ptr addrspace(1) %address.local1116, align 2, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1118 = or disjoint i64 %address.local1103, 17, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1118, !dbg !21
  store half %v2118, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1121 = or disjoint i64 %address.local1103, 18, !dbg !21
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1121, !dbg !21
  store half %v2117, ptr addrspace(1) %address.local1122, align 2, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1124 = or disjoint i64 %address.local1103, 19, !dbg !21
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1124, !dbg !21
  store half %v2116, ptr addrspace(1) %address.local1125, align 2, !dbg !21
  %address.local1126 = shl i64 %address.local1098, 7, !dbg !21
  %address.local1127 = add i64 %address.local1126, 8192, !dbg !21
  %address.local1128 = add nuw nsw i64 %v2137, %address.local1127, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1103, !dbg !21
  store half %v2104, ptr addrspace(1) %address.local1130, align 2, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1106, !dbg !21
  store half %v2103, ptr addrspace(1) %address.local1132, align 2, !dbg !21
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1109, !dbg !21
  store half %v2102, ptr addrspace(1) %address.local1134, align 2, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1112, !dbg !21
  store half %v2101, ptr addrspace(1) %address.local1136, align 2, !dbg !21
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1115, !dbg !21
  store half %v2089, ptr addrspace(1) %address.local1138, align 2, !dbg !21
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1118, !dbg !21
  store half %v2088, ptr addrspace(1) %address.local1140, align 2, !dbg !21
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1121, !dbg !21
  store half %v2087, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1124, !dbg !21
  store half %v2086, ptr addrspace(1) %address.local1144, align 2, !dbg !21
  br label %v2138, !dbg !21

v2138:                                             ; preds = %v2074, %v1623
  br i1 true, label %v2139, label %v2177, !dbg !21

v2139:                                             ; preds = %v2138
  %v2140 = insertelement <4 x float> poison, float %v2044, i64 0, !dbg !19
  %v2141 = insertelement <4 x float> %v2140, float %v2050, i64 1, !dbg !19
  %v2142 = insertelement <4 x float> %v2141, float %v2045, i64 2, !dbg !19
  %v2143 = insertelement <4 x float> %v2142, float %v2051, i64 3, !dbg !19
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1967, !dbg !19
  %normalized109 = fmul <4 x float> %v2143, %row.reciprocal108, !dbg !19
  %v2144 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %v2145 = extractelement <4 x half> %v2144, i64 3, !dbg !20
  %v2146 = extractelement <4 x half> %v2144, i64 2, !dbg !20
  %v2147 = extractelement <4 x half> %v2144, i64 1, !dbg !20
  %v2148 = extractelement <4 x half> %v2144, i64 0, !dbg !20
  %v2149 = insertelement <4 x float> poison, float %v1935, i64 0, !dbg !19
  %v2150 = insertelement <4 x float> %v2149, float %v1941, i64 1, !dbg !19
  %v2151 = insertelement <4 x float> %v2150, float %v1936, i64 2, !dbg !19
  %v2152 = insertelement <4 x float> %v2151, float %v1942, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %v2152, %row.reciprocal108, !dbg !19
  %v2153 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %v2154 = extractelement <4 x half> %v2153, i64 3, !dbg !20
  %v2155 = extractelement <4 x half> %v2153, i64 2, !dbg !20
  %v2156 = extractelement <4 x half> %v2153, i64 1, !dbg !20
  %v2157 = extractelement <4 x half> %v2153, i64 0, !dbg !20
  %v2158 = insertelement <4 x float> poison, float %v1826, i64 0, !dbg !19
  %v2159 = insertelement <4 x float> %v2158, float %v1832, i64 1, !dbg !19
  %v2160 = insertelement <4 x float> %v2159, float %v1827, i64 2, !dbg !19
  %v2161 = insertelement <4 x float> %v2160, float %v1833, i64 3, !dbg !19
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1749, !dbg !19
  %normalized112 = fmul <4 x float> %v2161, %row.reciprocal111, !dbg !19
  %v2162 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %v2163 = extractelement <4 x half> %v2162, i64 3, !dbg !20
  %v2164 = extractelement <4 x half> %v2162, i64 2, !dbg !20
  %v2165 = extractelement <4 x half> %v2162, i64 1, !dbg !20
  %v2166 = extractelement <4 x half> %v2162, i64 0, !dbg !20
  %v2167 = insertelement <4 x float> poison, float %v1717, i64 0, !dbg !19
  %v2168 = insertelement <4 x float> %v2167, float %v1723, i64 1, !dbg !19
  %v2169 = insertelement <4 x float> %v2168, float %v1718, i64 2, !dbg !19
  %v2170 = insertelement <4 x float> %v2169, float %v1724, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %v2170, %row.reciprocal111, !dbg !19
  %v2171 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %v2172 = extractelement <4 x half> %v2171, i64 3, !dbg !20
  %v2173 = extractelement <4 x half> %v2171, i64 2, !dbg !20
  %v2174 = extractelement <4 x half> %v2171, i64 1, !dbg !20
  %v2175 = extractelement <4 x half> %v2171, i64 0, !dbg !20
  %v2176 = shl nuw nsw i64 %v24, 19, !dbg !21
  %address.thread1145 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1146 = zext nneg i32 %address.thread1145 to i64, !dbg !8
  %address.local1147 = add i64 %v25, %address.local1146, !dbg !21
  %address.local1148 = lshr i64 %address.local1146, 6, !dbg !11
  %address.local1149 = shl nuw nsw i64 %address.local1148, 4, !dbg !21
  %address.local1150 = add i64 %address.local1147, %address.local1149, !dbg !21
  %address.local1151 = and i64 %address.local1146, 240, !dbg !21
  %address.local1152 = sub i64 %address.local1150, %address.local1151, !dbg !21
  %address.local1153 = add i64 %address.local1152, %v26, !dbg !21
  %address.local1154 = shl nuw nsw i64 %address.local1153, 7, !dbg !21
  %address.local1155 = add nuw nsw i64 %v2176, %address.local1154, !dbg !21
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1157 = lshr i64 %address.local1146, 2, !dbg !11
  %address.local1158 = and i64 %address.local1157, 12, !dbg !11
  %address.local1159 = or disjoint i64 %address.local1158, 32, !dbg !21
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1156, i64 %address.local1159, !dbg !21
  store half %v2175, ptr addrspace(1) %address.local1160, align 2, !dbg !21
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1162 = or disjoint i64 %address.local1158, 33, !dbg !21
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1161, i64 %address.local1162, !dbg !21
  store half %v2174, ptr addrspace(1) %address.local1163, align 2, !dbg !21
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1165 = or disjoint i64 %address.local1158, 34, !dbg !21
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1164, i64 %address.local1165, !dbg !21
  store half %v2173, ptr addrspace(1) %address.local1166, align 2, !dbg !21
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1168 = or disjoint i64 %address.local1158, 35, !dbg !21
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1167, i64 %address.local1168, !dbg !21
  store half %v2172, ptr addrspace(1) %address.local1169, align 2, !dbg !21
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1171 = or disjoint i64 %address.local1158, 48, !dbg !21
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1170, i64 %address.local1171, !dbg !21
  store half %v2166, ptr addrspace(1) %address.local1172, align 2, !dbg !21
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1174 = or disjoint i64 %address.local1158, 49, !dbg !21
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1174, !dbg !21
  store half %v2165, ptr addrspace(1) %address.local1175, align 2, !dbg !21
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1177 = or disjoint i64 %address.local1158, 50, !dbg !21
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1176, i64 %address.local1177, !dbg !21
  store half %v2164, ptr addrspace(1) %address.local1178, align 2, !dbg !21
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1180 = or i64 %address.local1157, 51, !dbg !21
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1179, i64 %address.local1180, !dbg !21
  store half %v2163, ptr addrspace(1) %address.local1181, align 2, !dbg !21
  %address.local1182 = shl i64 %address.local1153, 7, !dbg !21
  %address.local1183 = add i64 %address.local1182, 8192, !dbg !21
  %address.local1184 = add nuw nsw i64 %v2176, %address.local1183, !dbg !21
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1159, !dbg !21
  store half %v2157, ptr addrspace(1) %address.local1186, align 2, !dbg !21
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1187, i64 %address.local1162, !dbg !21
  store half %v2156, ptr addrspace(1) %address.local1188, align 2, !dbg !21
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1165, !dbg !21
  store half %v2155, ptr addrspace(1) %address.local1190, align 2, !dbg !21
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1191, i64 %address.local1168, !dbg !21
  store half %v2154, ptr addrspace(1) %address.local1192, align 2, !dbg !21
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1193, i64 %address.local1171, !dbg !21
  store half %v2148, ptr addrspace(1) %address.local1194, align 2, !dbg !21
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1174, !dbg !21
  store half %v2147, ptr addrspace(1) %address.local1196, align 2, !dbg !21
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1197, i64 %address.local1177, !dbg !21
  store half %v2146, ptr addrspace(1) %address.local1198, align 2, !dbg !21
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1180, !dbg !21
  store half %v2145, ptr addrspace(1) %address.local1200, align 2, !dbg !21
  br label %v2177, !dbg !21

v2177:                                             ; preds = %v2139, %v2138
  br i1 true, label %v2178, label %v2216, !dbg !21

v2178:                                             ; preds = %v2177
  %v2179 = insertelement <4 x float> poison, float %v2016, i64 0, !dbg !19
  %v2180 = insertelement <4 x float> %v2179, float %v2022, i64 1, !dbg !19
  %v2181 = insertelement <4 x float> %v2180, float %v2017, i64 2, !dbg !19
  %v2182 = insertelement <4 x float> %v2181, float %v2023, i64 3, !dbg !19
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1967, !dbg !19
  %normalized115 = fmul <4 x float> %v2182, %row.reciprocal114, !dbg !19
  %v2183 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %v2184 = extractelement <4 x half> %v2183, i64 3, !dbg !20
  %v2185 = extractelement <4 x half> %v2183, i64 2, !dbg !20
  %v2186 = extractelement <4 x half> %v2183, i64 1, !dbg !20
  %v2187 = extractelement <4 x half> %v2183, i64 0, !dbg !20
  %v2188 = insertelement <4 x float> poison, float %v1906, i64 0, !dbg !19
  %v2189 = insertelement <4 x float> %v2188, float %v1912, i64 1, !dbg !19
  %v2190 = insertelement <4 x float> %v2189, float %v1907, i64 2, !dbg !19
  %v2191 = insertelement <4 x float> %v2190, float %v1913, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %v2191, %row.reciprocal114, !dbg !19
  %v2192 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %v2193 = extractelement <4 x half> %v2192, i64 3, !dbg !20
  %v2194 = extractelement <4 x half> %v2192, i64 2, !dbg !20
  %v2195 = extractelement <4 x half> %v2192, i64 1, !dbg !20
  %v2196 = extractelement <4 x half> %v2192, i64 0, !dbg !20
  %v2197 = insertelement <4 x float> poison, float %v1798, i64 0, !dbg !19
  %v2198 = insertelement <4 x float> %v2197, float %v1804, i64 1, !dbg !19
  %v2199 = insertelement <4 x float> %v2198, float %v1799, i64 2, !dbg !19
  %v2200 = insertelement <4 x float> %v2199, float %v1805, i64 3, !dbg !19
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1749, !dbg !19
  %normalized118 = fmul <4 x float> %v2200, %row.reciprocal117, !dbg !19
  %v2201 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %v2202 = extractelement <4 x half> %v2201, i64 3, !dbg !20
  %v2203 = extractelement <4 x half> %v2201, i64 2, !dbg !20
  %v2204 = extractelement <4 x half> %v2201, i64 1, !dbg !20
  %v2205 = extractelement <4 x half> %v2201, i64 0, !dbg !20
  %v2206 = insertelement <4 x float> poison, float %v1688, i64 0, !dbg !19
  %v2207 = insertelement <4 x float> %v2206, float %v1694, i64 1, !dbg !19
  %v2208 = insertelement <4 x float> %v2207, float %v1689, i64 2, !dbg !19
  %v2209 = insertelement <4 x float> %v2208, float %v1695, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %v2209, %row.reciprocal117, !dbg !19
  %v2210 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %v2211 = extractelement <4 x half> %v2210, i64 3, !dbg !20
  %v2212 = extractelement <4 x half> %v2210, i64 2, !dbg !20
  %v2213 = extractelement <4 x half> %v2210, i64 1, !dbg !20
  %v2214 = extractelement <4 x half> %v2210, i64 0, !dbg !20
  %v2215 = shl nuw nsw i64 %v24, 19, !dbg !21
  %address.thread1201 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1202 = zext nneg i32 %address.thread1201 to i64, !dbg !8
  %address.local1203 = add i64 %v25, %address.local1202, !dbg !21
  %address.local1204 = lshr i64 %address.local1202, 6, !dbg !11
  %address.local1205 = shl nuw nsw i64 %address.local1204, 4, !dbg !21
  %address.local1206 = add i64 %address.local1203, %address.local1205, !dbg !21
  %address.local1207 = and i64 %address.local1202, 240, !dbg !21
  %address.local1208 = sub i64 %address.local1206, %address.local1207, !dbg !21
  %address.local1209 = add i64 %address.local1208, %v26, !dbg !21
  %address.local1210 = shl nuw nsw i64 %address.local1209, 7, !dbg !21
  %address.local1211 = add nuw nsw i64 %v2215, %address.local1210, !dbg !21
  %address.local1212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1213 = lshr i64 %address.local1202, 2, !dbg !11
  %address.local1214 = and i64 %address.local1213, 12, !dbg !11
  %address.local1215 = or disjoint i64 %address.local1214, 64, !dbg !21
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1212, i64 %address.local1215, !dbg !21
  store half %v2214, ptr addrspace(1) %address.local1216, align 2, !dbg !21
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1218 = or disjoint i64 %address.local1214, 65, !dbg !21
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1217, i64 %address.local1218, !dbg !21
  store half %v2213, ptr addrspace(1) %address.local1219, align 2, !dbg !21
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1221 = or disjoint i64 %address.local1214, 66, !dbg !21
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1220, i64 %address.local1221, !dbg !21
  store half %v2212, ptr addrspace(1) %address.local1222, align 2, !dbg !21
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1224 = or disjoint i64 %address.local1214, 67, !dbg !21
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1223, i64 %address.local1224, !dbg !21
  store half %v2211, ptr addrspace(1) %address.local1225, align 2, !dbg !21
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1227 = or disjoint i64 %address.local1214, 80, !dbg !21
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1226, i64 %address.local1227, !dbg !21
  store half %v2205, ptr addrspace(1) %address.local1228, align 2, !dbg !21
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1230 = or disjoint i64 %address.local1214, 81, !dbg !21
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1229, i64 %address.local1230, !dbg !21
  store half %v2204, ptr addrspace(1) %address.local1231, align 2, !dbg !21
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1233 = or disjoint i64 %address.local1214, 82, !dbg !21
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1232, i64 %address.local1233, !dbg !21
  store half %v2203, ptr addrspace(1) %address.local1234, align 2, !dbg !21
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1236 = or disjoint i64 %address.local1214, 83, !dbg !21
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1235, i64 %address.local1236, !dbg !21
  store half %v2202, ptr addrspace(1) %address.local1237, align 2, !dbg !21
  %address.local1238 = shl i64 %address.local1209, 7, !dbg !21
  %address.local1239 = add i64 %address.local1238, 8192, !dbg !21
  %address.local1240 = add nuw nsw i64 %v2215, %address.local1239, !dbg !21
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1241, i64 %address.local1215, !dbg !21
  store half %v2196, ptr addrspace(1) %address.local1242, align 2, !dbg !21
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1243, i64 %address.local1218, !dbg !21
  store half %v2195, ptr addrspace(1) %address.local1244, align 2, !dbg !21
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1245, i64 %address.local1221, !dbg !21
  store half %v2194, ptr addrspace(1) %address.local1246, align 2, !dbg !21
  %address.local1247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1247, i64 %address.local1224, !dbg !21
  store half %v2193, ptr addrspace(1) %address.local1248, align 2, !dbg !21
  %address.local1249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1250 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1249, i64 %address.local1227, !dbg !21
  store half %v2187, ptr addrspace(1) %address.local1250, align 2, !dbg !21
  %address.local1251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1251, i64 %address.local1230, !dbg !21
  store half %v2186, ptr addrspace(1) %address.local1252, align 2, !dbg !21
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1253, i64 %address.local1233, !dbg !21
  store half %v2185, ptr addrspace(1) %address.local1254, align 2, !dbg !21
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1255, i64 %address.local1236, !dbg !21
  store half %v2184, ptr addrspace(1) %address.local1256, align 2, !dbg !21
  br label %v2216, !dbg !21

v2216:                                             ; preds = %v2178, %v2177
  br i1 true, label %v2217, label %v2255, !dbg !21

v2217:                                             ; preds = %v2216
  %v2218 = insertelement <4 x float> poison, float %v1988, i64 0, !dbg !19
  %v2219 = insertelement <4 x float> %v2218, float %v1994, i64 1, !dbg !19
  %v2220 = insertelement <4 x float> %v2219, float %v1989, i64 2, !dbg !19
  %v2221 = insertelement <4 x float> %v2220, float %v1995, i64 3, !dbg !19
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1967, !dbg !19
  %normalized121 = fmul <4 x float> %v2221, %row.reciprocal120, !dbg !19
  %v2222 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %v2223 = extractelement <4 x half> %v2222, i64 3, !dbg !20
  %v2224 = extractelement <4 x half> %v2222, i64 2, !dbg !20
  %v2225 = extractelement <4 x half> %v2222, i64 1, !dbg !20
  %v2226 = extractelement <4 x half> %v2222, i64 0, !dbg !20
  %v2227 = insertelement <4 x float> poison, float %v1877, i64 0, !dbg !19
  %v2228 = insertelement <4 x float> %v2227, float %v1883, i64 1, !dbg !19
  %v2229 = insertelement <4 x float> %v2228, float %v1878, i64 2, !dbg !19
  %v2230 = insertelement <4 x float> %v2229, float %v1884, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %v2230, %row.reciprocal120, !dbg !19
  %v2231 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %v2232 = extractelement <4 x half> %v2231, i64 3, !dbg !20
  %v2233 = extractelement <4 x half> %v2231, i64 2, !dbg !20
  %v2234 = extractelement <4 x half> %v2231, i64 1, !dbg !20
  %v2235 = extractelement <4 x half> %v2231, i64 0, !dbg !20
  %v2236 = insertelement <4 x float> poison, float %v1770, i64 0, !dbg !19
  %v2237 = insertelement <4 x float> %v2236, float %v1776, i64 1, !dbg !19
  %v2238 = insertelement <4 x float> %v2237, float %v1771, i64 2, !dbg !19
  %v2239 = insertelement <4 x float> %v2238, float %v1777, i64 3, !dbg !19
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %v1749, !dbg !19
  %normalized124 = fmul <4 x float> %v2239, %row.reciprocal123, !dbg !19
  %v2240 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %v2241 = extractelement <4 x half> %v2240, i64 3, !dbg !20
  %v2242 = extractelement <4 x half> %v2240, i64 2, !dbg !20
  %v2243 = extractelement <4 x half> %v2240, i64 1, !dbg !20
  %v2244 = extractelement <4 x half> %v2240, i64 0, !dbg !20
  %v2245 = insertelement <4 x float> poison, float %v1659, i64 0, !dbg !19
  %v2246 = insertelement <4 x float> %v2245, float %v1665, i64 1, !dbg !19
  %v2247 = insertelement <4 x float> %v2246, float %v1660, i64 2, !dbg !19
  %v2248 = insertelement <4 x float> %v2247, float %v1666, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %v2248, %row.reciprocal123, !dbg !19
  %v2249 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %v2250 = extractelement <4 x half> %v2249, i64 3, !dbg !20
  %v2251 = extractelement <4 x half> %v2249, i64 2, !dbg !20
  %v2252 = extractelement <4 x half> %v2249, i64 1, !dbg !20
  %v2253 = extractelement <4 x half> %v2249, i64 0, !dbg !20
  %v2254 = shl nuw nsw i64 %v24, 19, !dbg !21
  %address.thread1257 = call i32 asm sideeffect "", "=v,0"(i32 %v31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1258 = zext nneg i32 %address.thread1257 to i64, !dbg !8
  %address.local1259 = add i64 %v25, %address.local1258, !dbg !21
  %address.local1260 = lshr i64 %address.local1258, 6, !dbg !11
  %address.local1261 = shl nuw nsw i64 %address.local1260, 4, !dbg !21
  %address.local1262 = add i64 %address.local1259, %address.local1261, !dbg !21
  %address.local1263 = and i64 %address.local1258, 240, !dbg !21
  %address.local1264 = sub i64 %address.local1262, %address.local1263, !dbg !21
  %address.local1265 = add i64 %address.local1264, %v26, !dbg !21
  %address.local1266 = shl nuw nsw i64 %address.local1265, 7, !dbg !21
  %address.local1267 = add nuw nsw i64 %v2254, %address.local1266, !dbg !21
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1269 = lshr i64 %address.local1258, 2, !dbg !11
  %address.local1270 = and i64 %address.local1269, 12, !dbg !11
  %address.local1271 = or disjoint i64 %address.local1270, 96, !dbg !21
  %address.local1272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1268, i64 %address.local1271, !dbg !21
  store half %v2253, ptr addrspace(1) %address.local1272, align 2, !dbg !21
  %address.local1273 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1274 = or disjoint i64 %address.local1270, 97, !dbg !21
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1273, i64 %address.local1274, !dbg !21
  store half %v2252, ptr addrspace(1) %address.local1275, align 2, !dbg !21
  %address.local1276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1277 = or disjoint i64 %address.local1270, 98, !dbg !21
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1276, i64 %address.local1277, !dbg !21
  store half %v2251, ptr addrspace(1) %address.local1278, align 2, !dbg !21
  %address.local1279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1280 = or disjoint i64 %address.local1270, 99, !dbg !21
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1279, i64 %address.local1280, !dbg !21
  store half %v2250, ptr addrspace(1) %address.local1281, align 2, !dbg !21
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1283 = or disjoint i64 %address.local1270, 112, !dbg !21
  %address.local1284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1282, i64 %address.local1283, !dbg !21
  store half %v2244, ptr addrspace(1) %address.local1284, align 2, !dbg !21
  %address.local1285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1286 = or disjoint i64 %address.local1270, 113, !dbg !21
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1285, i64 %address.local1286, !dbg !21
  store half %v2243, ptr addrspace(1) %address.local1287, align 2, !dbg !21
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1289 = or disjoint i64 %address.local1270, 114, !dbg !21
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1288, i64 %address.local1289, !dbg !21
  store half %v2242, ptr addrspace(1) %address.local1290, align 2, !dbg !21
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1292 = or i64 %address.local1269, 115, !dbg !21
  %address.local1293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1291, i64 %address.local1292, !dbg !21
  store half %v2241, ptr addrspace(1) %address.local1293, align 2, !dbg !21
  %address.local1294 = shl i64 %address.local1265, 7, !dbg !21
  %address.local1295 = add i64 %address.local1294, 8192, !dbg !21
  %address.local1296 = add nuw nsw i64 %v2254, %address.local1295, !dbg !21
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1297, i64 %address.local1271, !dbg !21
  store half %v2235, ptr addrspace(1) %address.local1298, align 2, !dbg !21
  %address.local1299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1299, i64 %address.local1274, !dbg !21
  store half %v2234, ptr addrspace(1) %address.local1300, align 2, !dbg !21
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1301, i64 %address.local1277, !dbg !21
  store half %v2233, ptr addrspace(1) %address.local1302, align 2, !dbg !21
  %address.local1303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1303, i64 %address.local1280, !dbg !21
  store half %v2232, ptr addrspace(1) %address.local1304, align 2, !dbg !21
  %address.local1305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1306 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1305, i64 %address.local1283, !dbg !21
  store half %v2226, ptr addrspace(1) %address.local1306, align 2, !dbg !21
  %address.local1307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1307, i64 %address.local1286, !dbg !21
  store half %v2225, ptr addrspace(1) %address.local1308, align 2, !dbg !21
  %address.local1309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1309, i64 %address.local1289, !dbg !21
  store half %v2224, ptr addrspace(1) %address.local1310, align 2, !dbg !21
  %address.local1311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1311, i64 %address.local1292, !dbg !21
  store half %v2223, ptr addrspace(1) %address.local1312, align 2, !dbg !21
  br label %v2255, !dbg !21

v2255:                                             ; preds = %v2217, %v2216
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/local_reduction_pass/compiler-bn128/run-20261026-112711-jne7_i1s/grid-000003")
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
