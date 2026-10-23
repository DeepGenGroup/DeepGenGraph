; ModuleID = '3rd/deepgengraph/test/test_input_sd16.mlir'
source_filename = "3rd/deepgengraph/test/test_input_sd16.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %v0, ptr addrspace(1) %v1, ptr addrspace(1) %v2, ptr addrspace(1) %v3) #1 !dbg !4 !reqd_work_group_size !32 {
  %v5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %v6 = sext i32 %v5 to i64, !dbg !5
  %v7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %v8 = sext i32 %v7 to i64, !dbg !5
  %v9 = shl nsw i64 %v6, 7, !dbg !10
  %.lobit = ashr i32 %v5, 31, !dbg !12
  %v10 = xor i32 %v5, %.lobit, !dbg !12
  %v11 = sdiv i32 %v10, 32, !dbg !12
  %.lobit1 = ashr i32 %v5, 31, !dbg !12
  %v12 = xor i32 %v11, %.lobit1, !dbg !12
  %v13 = sext i32 %v12 to i64, !dbg !12
  %v14 = add nsw i64 %v8, %v13, !dbg !12
  %v15 = and i64 %v14, 31, !dbg !12
  %v16 = shl nsw i64 %v6, 7, !dbg !12
  %v17 = mul nsw i64 %v13, -4096, !dbg !12
  %v18 = add nsw i64 %v16, %v17, !dbg !12
  %v19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !34
  %v20 = zext nneg i32 %v19 to i64, !dbg !13
  %v21 = lshr i64 %v20, 6, !dbg !13
  %v22 = or disjoint i64 %v9, 127, !dbg !14
  %v23 = add nsw i64 %v9, 128, !dbg !14
  %v24 = icmp uge i64 %v23, %v22, !dbg !14
  %v26 = add i64 %v6, 0
  %v38 = add i64 0, 0
  %v39 = insertvalue [2 x <32 x float>] zeroinitializer, <32 x float> zeroinitializer, 0
  %v33 = shl nsw i64 %v26, 7, !dbg !16
  %v34 = lshr i64 %v26, 5, !dbg !17
  %v35 = shl i64 %v34, 12, !dbg !17
  %v36 = sub i64 %v33, %v35, !dbg !17
  %v42 = shl nuw nsw i64 %v38, 5, !dbg !22
  %.idx = shl nuw nsw i64 %v15, 20, !dbg !12
  %v43 = getelementptr i8, ptr addrspace(1) %v0, i64 %.idx, !dbg !12
  %.idx68 = shl nsw i64 %v18, 8, !dbg !12
  %v44 = getelementptr i8, ptr addrspace(1) %v43, i64 %.idx68, !dbg !12
  %v45 = getelementptr [2 x i8], ptr addrspace(1) %v44, i64 %v42, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %v19) #9, !dbg !12, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local130 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local131 = and i64 %address.local130, 1536, !dbg !12
  %address.local132 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local131, !dbg !12
  %address.local133 = ptrtoint ptr addrspace(3) %address.local132 to i64, !dbg !12
  %address.local134 = trunc i64 %address.local133 to i32, !dbg !12
  %v46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local134), !dbg !12
  %v47 = zext i32 %v46 to i64, !dbg !12
  %v48 = inttoptr i64 %v47 to ptr addrspace(3), !dbg !12
  %address.local135 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local136 = and i64 %address.local135, 16256, !dbg !12
  %address.local137 = lshr i64 %address.local, 4, !dbg !12
  %address.local138 = and i64 %address.local137, 8, !dbg !12
  %address.local139 = or disjoint i64 %address.local136, %address.local138, !dbg !12
  %address.local140 = getelementptr [2 x i8], ptr addrspace(1) %v45, i64 %address.local139, !dbg !12
  %v49 = ptrtoint ptr addrspace(1) %v0 to i64, !dbg !12
  %v50 = ptrtoint ptr addrspace(1) %address.local140 to i64, !dbg !12
  %v51 = sub i64 %v50, %v49, !dbg !12
  %dma.buffer.offset876 = trunc i64 %v51 to i32, !dbg !12
  %v52 = trunc i64 %v49 to i32, !dbg !12
  %v53 = insertelement <4 x i32> poison, i32 %v52, i64 0, !dbg !12
  %v54 = lshr i64 %v49, 32, !dbg !12
  %v55 = trunc i64 %v54 to i32, !dbg !12
  %v56 = insertelement <4 x i32> %v53, i32 %v55, i64 1, !dbg !12
  %v57 = insertelement <4 x i32> %v56, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc877 = insertelement <4 x i32> %v57, i32 131072, i64 3, !dbg !12
  %v58 = ptrtoint ptr addrspace(3) %v48 to i32, !dbg !12
  %dma.buffer.lds878 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v58), !dbg !12
  %v59 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds878, i32 %dma.buffer.offset876, <4 x i32> %dma.buffer.rsrc877) #10, !dbg !12
  %address.local141 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local142 = shl nuw nsw i64 %address.local141, 3, !dbg !12
  %address.local143 = and i64 %address.local142, 3584, !dbg !12
  %address.local144 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local143, !dbg !12
  %address.local145 = ptrtoint ptr addrspace(3) %address.local144 to i64, !dbg !12
  %address.local146 = trunc i64 %address.local145 to i32, !dbg !12
  %v60 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local146), !dbg !12
  %v61 = zext i32 %v60 to i64, !dbg !12
  %v62 = inttoptr i64 %v61 to ptr addrspace(3), !dbg !12
  %address.local147 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local148 = and i64 %address.local147, 16256, !dbg !12
  %address.local149 = lshr i64 %address.local141, 4, !dbg !12
  %address.local150 = and i64 %address.local149, 24, !dbg !12
  %address.local151 = or disjoint i64 %address.local148, %address.local150, !dbg !12
  %address.local152 = getelementptr [2 x i8], ptr addrspace(1) %v45, i64 %address.local151, !dbg !12
  %v63 = ptrtoint ptr addrspace(1) %v0 to i64, !dbg !12
  %v64 = ptrtoint ptr addrspace(1) %address.local152 to i64, !dbg !12
  %v65 = sub i64 %v64, %v63, !dbg !12
  %dma.buffer.offset873 = trunc i64 %v65 to i32, !dbg !12
  %v66 = trunc i64 %v63 to i32, !dbg !12
  %v67 = insertelement <4 x i32> poison, i32 %v66, i64 0, !dbg !12
  %v68 = lshr i64 %v63, 32, !dbg !12
  %v69 = trunc i64 %v68 to i32, !dbg !12
  %v70 = insertelement <4 x i32> %v67, i32 %v69, i64 1, !dbg !12
  %v71 = insertelement <4 x i32> %v70, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc874 = insertelement <4 x i32> %v71, i32 131072, i64 3, !dbg !12
  %v72 = ptrtoint ptr addrspace(3) %v62 to i32, !dbg !12
  %dma.buffer.lds875 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v72), !dbg !12
  %v73 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds875, i32 %dma.buffer.offset873, <4 x i32> %dma.buffer.rsrc874) #10, !dbg !12
  %v74 = shl nuw nsw i64 %v38, 10, !dbg !17
  %v75 = add nuw i64 %v26, %v74, !dbg !17
  %.lobit69 = ashr i64 %v75, 63, !dbg !17
  %v76 = xor i64 %v75, %.lobit69, !dbg !17
  %v77 = sdiv i64 %v76, 4096, !dbg !17
  %.lobit70 = ashr i64 %v75, 63, !dbg !17
  %v78 = xor i64 %v77, %.lobit70, !dbg !17
  %v79 = add nsw i64 %v78, %v8, !dbg !17
  %v80 = add nuw nsw i64 %v42, %v34, !dbg !17
  %v81 = and i64 %v80, 127, !dbg !17
  %v82 = shl i64 %v79, 19, !dbg !17
  %v83 = and i64 %v82, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %v84 = getelementptr [2 x i8], ptr addrspace(1) %v1, i64 %v83, !dbg !17
  %v85 = getelementptr [2 x i8], ptr addrspace(1) %v84, i64 %v81, !dbg !17
  %.idx71 = shl i64 %v36, 8, !dbg !17
  %v86 = getelementptr i8, ptr addrspace(1) %v85, i64 %.idx71, !dbg !17
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local131, !dbg !17
  %address.local154 = ptrtoint ptr addrspace(3) %address.local153 to i64, !dbg !17
  %address.local155 = trunc i64 %address.local154 to i32, !dbg !17
  %v87 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local155), !dbg !17
  %v88 = zext i32 %v87 to i64, !dbg !17
  %v89 = inttoptr i64 %v88 to ptr addrspace(3), !dbg !17
  %address.local156 = getelementptr [2 x i8], ptr addrspace(1) %v86, i64 %address.local139, !dbg !17
  %v90 = ptrtoint ptr addrspace(1) %v1 to i64, !dbg !17
  %v91 = ptrtoint ptr addrspace(1) %address.local156 to i64, !dbg !17
  %v92 = sub i64 %v91, %v90, !dbg !17
  %dma.buffer.offset870 = trunc i64 %v92 to i32, !dbg !17
  %v93 = trunc i64 %v90 to i32, !dbg !17
  %v94 = insertelement <4 x i32> poison, i32 %v93, i64 0, !dbg !17
  %v95 = lshr i64 %v90, 32, !dbg !17
  %v96 = trunc i64 %v95 to i32, !dbg !17
  %v97 = insertelement <4 x i32> %v94, i32 %v96, i64 1, !dbg !17
  %v98 = insertelement <4 x i32> %v97, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc871 = insertelement <4 x i32> %v98, i32 131072, i64 3, !dbg !17
  %v99 = ptrtoint ptr addrspace(3) %v89 to i32, !dbg !17
  %dma.buffer.lds872 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v99), !dbg !17
  %v100 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds872, i32 %dma.buffer.offset870, <4 x i32> %dma.buffer.rsrc871) #10, !dbg !17
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local143, !dbg !17
  %address.local158 = ptrtoint ptr addrspace(3) %address.local157 to i64, !dbg !17
  %address.local159 = trunc i64 %address.local158 to i32, !dbg !17
  %v101 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local159), !dbg !17
  %v102 = zext i32 %v101 to i64, !dbg !17
  %v103 = inttoptr i64 %v102 to ptr addrspace(3), !dbg !17
  %address.local160 = getelementptr [2 x i8], ptr addrspace(1) %v86, i64 %address.local151, !dbg !17
  %v104 = ptrtoint ptr addrspace(1) %v1 to i64, !dbg !17
  %v105 = ptrtoint ptr addrspace(1) %address.local160 to i64, !dbg !17
  %v106 = sub i64 %v105, %v104, !dbg !17
  %dma.buffer.offset = trunc i64 %v106 to i32, !dbg !17
  %v107 = trunc i64 %v104 to i32, !dbg !17
  %v108 = insertelement <4 x i32> poison, i32 %v107, i64 0, !dbg !17
  %v109 = lshr i64 %v104, 32, !dbg !17
  %v110 = trunc i64 %v109 to i32, !dbg !17
  %v111 = insertelement <4 x i32> %v108, i32 %v110, i64 1, !dbg !17
  %v112 = insertelement <4 x i32> %v111, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %v112, i32 131072, i64 3, !dbg !17
  %v113 = ptrtoint ptr addrspace(3) %v103 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v113), !dbg !17
  %v114 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %address.local161 = lshr i32 %address.thread, 2, !dbg !13
  %address.local162 = and i32 %address.local161, 4, !dbg !13
  %address.local163 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local164 = and i32 %address.local163, 384, !dbg !13
  %address.local165 = or disjoint i32 %address.local162, %address.local164, !dbg !13
  %address.local166 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local167 = and i32 %address.local166, 120, !dbg !13
  %address.local168 = or disjoint i32 %address.local165, %address.local167, !dbg !13
  %address.local169 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local170 = and i32 %address.local169, 1024, !dbg !13
  %address.local171 = or disjoint i32 %address.local168, %address.local170, !dbg !13
  %address.local172 = zext nneg i32 %address.local171 to i64, !dbg !13
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !13
  %v115 = load <4 x half>, ptr addrspace(3) %address.local173, align 2, !dbg !13
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !13
  %address.local175 = getelementptr i8, ptr addrspace(3) %address.local174, i64 4096, !dbg !13
  %v116 = load <4 x half>, ptr addrspace(3) %address.local175, align 2, !dbg !13
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !13
  %address.local177 = getelementptr i8, ptr addrspace(3) %address.local176, i64 1024, !dbg !13
  %v117 = load <4 x half>, ptr addrspace(3) %address.local177, align 2, !dbg !13
  %v118 = fmul <4 x half> %v115, splat (half 0xH3015), !dbg !13
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !13
  %address.local179 = getelementptr i8, ptr addrspace(3) %address.local178, i64 5120, !dbg !13
  %v119 = load <4 x half>, ptr addrspace(3) %address.local179, align 2, !dbg !13
  %v120 = fmul <4 x half> %v116, splat (half 0xH3015), !dbg !13
  %v121 = fmul <4 x half> %v117, splat (half 0xH3015), !dbg !13
  %v122 = fmul <4 x half> %v119, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local180 = lshr i32 %address.thread, 2, !dbg !8
  %address.local181 = and i32 %address.local180, 4, !dbg !8
  %address.local182 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local183 = and i32 %address.local182, 24, !dbg !8
  %address.local184 = or disjoint i32 %address.local181, %address.local183, !dbg !8
  %address.local185 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local186 = and i32 %address.local185, 1024, !dbg !8
  %address.local187 = or disjoint i32 %address.local184, %address.local186, !dbg !8
  %address.local188 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local189 = and i32 %address.local188, 384, !dbg !8
  %address.local190 = or disjoint i32 %address.local187, %address.local189, !dbg !8
  %address.local191 = zext nneg i32 %address.local190 to i64, !dbg !8
  %address.local192 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local191, !dbg !8
  %v123 = load <4 x half>, ptr addrspace(3) %address.local192, align 2, !dbg !8
  %v124 = extractvalue [2 x <32 x float>] %v39, 0, !dbg !31
  %v125 = shufflevector <32 x float> %v124, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %address.local193 = lshr i32 %address.thread, 2, !dbg !8
  %address.local194 = and i32 %address.local193, 4, !dbg !8
  %address.local195 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local196 = and i32 %address.local195, 24, !dbg !8
  %address.local197 = or disjoint i32 %address.local194, %address.local196, !dbg !8
  %address.local198 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local199 = and i32 %address.local198, 1024, !dbg !8
  %address.local200 = or disjoint i32 %address.local197, %address.local199, !dbg !8
  %address.local201 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local202 = and i32 %address.local201, 384, !dbg !8
  %address.local203 = or disjoint i32 %address.local200, %address.local202, !dbg !8
  %address.local204 = zext nneg i32 %address.local203 to i64, !dbg !8
  %address.local205 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local204, !dbg !8
  %address.local206 = getelementptr i8, ptr addrspace(3) %address.local205, i64 4096, !dbg !8
  %v126 = load <4 x half>, ptr addrspace(3) %address.local206, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v127 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v123, <4 x float> %v125) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %v128 = shufflevector <32 x float> %v124, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %v129 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v126, <4 x float> %v127) #9, !dbg !8
  %v130 = shufflevector <32 x float> %v124, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %v131 = shufflevector <32 x float> %v124, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %v132 = shufflevector <32 x float> %v124, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %v133 = shufflevector <32 x float> %v124, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %v134 = shufflevector <32 x float> %v124, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %v135 = shufflevector <32 x float> %v124, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %v136 = extractvalue [2 x <32 x float>] %v39, 1, !dbg !31
  %v137 = shufflevector <32 x float> %v136, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %v138 = shufflevector <32 x float> %v136, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %v139 = shufflevector <32 x float> %v136, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %v140 = shufflevector <32 x float> %v136, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %v141 = shufflevector <32 x float> %v136, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %v142 = shufflevector <32 x float> %v136, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %v143 = shufflevector <32 x float> %v136, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %v144 = shufflevector <32 x float> %v136, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %address.local207 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local209 = lshr i64 %address.local, 2, !dbg !8
  %address.local210 = mul nsw i64 %address.local209, -512, !dbg !8
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) %address.local208, i64 %address.local210, !dbg !8
  %address.local212 = shl nuw nsw i64 %address.local, 1, !dbg !8
  %address.local213 = and i64 %address.local212, 24, !dbg !8
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local213, !dbg !8
  %address.local215 = lshr i64 %address.local, 2, !dbg !13
  %address.local216 = and i64 %address.local215, 12, !dbg !13
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local216, !dbg !8
  %address.local218 = and i64 %address.local, 63, !dbg !13
  %address.local219 = lshr i64 %address.local218, 5, !dbg !13
  %address.local220 = mul nuw nsw i64 %address.local219, 1016, !dbg !13
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local220, !dbg !8
  %address.local222 = getelementptr i8, ptr addrspace(3) %address.local221, i64 64, !dbg !8
  %v145 = load <4 x half>, ptr addrspace(3) %address.local222, align 2, !dbg !8
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local216, !dbg !8
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) %address.local224, i64 %address.local220, !dbg !8
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local210, !dbg !8
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local213, !dbg !8
  %address.local228 = getelementptr i8, ptr addrspace(3) %address.local227, i64 4160, !dbg !8
  %v146 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !8
  %v147 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v145, <4 x float> %v128) #9, !dbg !8
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local210, !dbg !8
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) %address.local230, i64 %address.local213, !dbg !8
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %address.local231, i64 %address.local216, !dbg !8
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) %address.local232, i64 %address.local220, !dbg !8
  %address.local234 = getelementptr i8, ptr addrspace(3) %address.local233, i64 128, !dbg !8
  %v148 = load <4 x half>, ptr addrspace(3) %address.local234, align 2, !dbg !8
  %v149 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v146, <4 x float> %v147) #9, !dbg !8
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local216, !dbg !8
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) %address.local236, i64 %address.local220, !dbg !8
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %address.local237, i64 %address.local210, !dbg !8
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) %address.local238, i64 %address.local213, !dbg !8
  %address.local240 = getelementptr i8, ptr addrspace(3) %address.local239, i64 4224, !dbg !8
  %v150 = load <4 x half>, ptr addrspace(3) %address.local240, align 2, !dbg !8
  %v151 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v148, <4 x float> %v130) #9, !dbg !8
  %v152 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v150, <4 x float> %v151) #9, !dbg !8
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local210, !dbg !8
  %address.local243 = getelementptr [2 x i8], ptr addrspace(3) %address.local242, i64 %address.local213, !dbg !8
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %address.local243, i64 %address.local216, !dbg !8
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local220, !dbg !8
  %address.local246 = getelementptr i8, ptr addrspace(3) %address.local245, i64 192, !dbg !8
  %v153 = load <4 x half>, ptr addrspace(3) %address.local246, align 2, !dbg !8
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local216, !dbg !8
  %address.local249 = getelementptr [2 x i8], ptr addrspace(3) %address.local248, i64 %address.local220, !dbg !8
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) %address.local249, i64 %address.local210, !dbg !8
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) %address.local250, i64 %address.local213, !dbg !8
  %address.local252 = getelementptr i8, ptr addrspace(3) %address.local251, i64 4288, !dbg !8
  %v154 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !8
  %v155 = shufflevector <4 x float> %v152, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %v156 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v153, <4 x float> %v131) #9, !dbg !8
  %address.local253 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local191, !dbg !8
  %address.local254 = getelementptr i8, ptr addrspace(3) %address.local253, i64 1024, !dbg !8
  %v157 = load <4 x half>, ptr addrspace(3) %address.local254, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v158 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v154, <4 x float> %v156) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %v159 = shufflevector <4 x float> %v158, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %address.local255 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local204, !dbg !8
  %address.local256 = getelementptr i8, ptr addrspace(3) %address.local255, i64 5120, !dbg !8
  %v160 = load <4 x half>, ptr addrspace(3) %address.local256, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v161 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v157, <4 x float> %v132) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v162 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v160, <4 x float> %v161) #9, !dbg !8
  %address.local257 = add nuw nsw i64 %address.local207, 512, !dbg !8
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local257, !dbg !8
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) %address.local258, i64 %address.local210, !dbg !8
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %address.local259, i64 %address.local213, !dbg !8
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local216, !dbg !8
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local220, !dbg !8
  %address.local263 = getelementptr i8, ptr addrspace(3) %address.local262, i64 64, !dbg !8
  %v163 = load <4 x half>, ptr addrspace(3) %address.local263, align 2, !dbg !8
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local257, !dbg !8
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local216, !dbg !8
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) %address.local265, i64 %address.local220, !dbg !8
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local266, i64 %address.local210, !dbg !8
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local213, !dbg !8
  %address.local269 = getelementptr i8, ptr addrspace(3) %address.local268, i64 4160, !dbg !8
  %v164 = load <4 x half>, ptr addrspace(3) %address.local269, align 2, !dbg !8
  %v165 = shufflevector <4 x float> %v162, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %v166 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v163, <4 x float> %v133) #9, !dbg !8
  %address.local270 = add nuw nsw i64 %address.local207, 512, !dbg !8
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local270, !dbg !8
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) %address.local271, i64 %address.local210, !dbg !8
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) %address.local272, i64 %address.local213, !dbg !8
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local216, !dbg !8
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local220, !dbg !8
  %address.local276 = getelementptr i8, ptr addrspace(3) %address.local275, i64 128, !dbg !8
  %v167 = load <4 x half>, ptr addrspace(3) %address.local276, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v168 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v164, <4 x float> %v166) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %v169 = shufflevector <4 x float> %v168, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local270, !dbg !8
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local216, !dbg !8
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) %address.local278, i64 %address.local220, !dbg !8
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %address.local279, i64 %address.local210, !dbg !8
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %address.local280, i64 %address.local213, !dbg !8
  %address.local282 = getelementptr i8, ptr addrspace(3) %address.local281, i64 4224, !dbg !8
  %v170 = load <4 x half>, ptr addrspace(3) %address.local282, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v171 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v167, <4 x float> %v134) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v172 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v170, <4 x float> %v171) #9, !dbg !8
  %address.local283 = add nuw nsw i64 %address.local207, 512, !dbg !8
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local283, !dbg !8
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) %address.local284, i64 %address.local210, !dbg !8
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) %address.local285, i64 %address.local213, !dbg !8
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %address.local286, i64 %address.local216, !dbg !8
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) %address.local287, i64 %address.local220, !dbg !8
  %address.local289 = getelementptr i8, ptr addrspace(3) %address.local288, i64 192, !dbg !8
  %v173 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !8
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local283, !dbg !8
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local216, !dbg !8
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) %address.local291, i64 %address.local220, !dbg !8
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %address.local292, i64 %address.local210, !dbg !8
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) %address.local293, i64 %address.local213, !dbg !8
  %address.local295 = getelementptr i8, ptr addrspace(3) %address.local294, i64 4288, !dbg !8
  %v174 = load <4 x half>, ptr addrspace(3) %address.local295, align 2, !dbg !8
  %v175 = shufflevector <4 x float> %v172, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %v176 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v173, <4 x float> %v135) #9, !dbg !8
  %v177 = load <4 x half>, ptr addrspace(3) %address.local192, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v178 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v174, <4 x float> %v176) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %v179 = shufflevector <4 x float> %v178, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %v180 = load <4 x half>, ptr addrspace(3) %address.local206, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v181 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v177, <4 x float> %v137) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v182 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v180, <4 x float> %v181) #9, !dbg !8
  %v183 = load <4 x half>, ptr addrspace(3) %address.local222, align 2, !dbg !8
  %v184 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !8
  %v185 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v183, <4 x float> %v138) #9, !dbg !8
  %v186 = load <4 x half>, ptr addrspace(3) %address.local234, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v187 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v184, <4 x float> %v185) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v188 = load <4 x half>, ptr addrspace(3) %address.local240, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v189 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v186, <4 x float> %v139) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v190 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v188, <4 x float> %v189) #9, !dbg !8
  %v191 = load <4 x half>, ptr addrspace(3) %address.local246, align 2, !dbg !8
  %v192 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !8
  %v193 = shufflevector <4 x float> %v190, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %v194 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v191, <4 x float> %v140) #9, !dbg !8
  %v195 = load <4 x half>, ptr addrspace(3) %address.local254, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v196 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v192, <4 x float> %v194) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %v197 = shufflevector <4 x float> %v196, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %v198 = load <4 x half>, ptr addrspace(3) %address.local256, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v199 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v195, <4 x float> %v141) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v200 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v198, <4 x float> %v199) #9, !dbg !8
  %v201 = load <4 x half>, ptr addrspace(3) %address.local263, align 2, !dbg !8
  %v202 = load <4 x half>, ptr addrspace(3) %address.local269, align 2, !dbg !8
  %v203 = shufflevector <4 x float> %v200, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %v204 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v201, <4 x float> %v142) #9, !dbg !8
  %v205 = load <4 x half>, ptr addrspace(3) %address.local276, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v206 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v202, <4 x float> %v204) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %v207 = shufflevector <4 x float> %v206, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %v208 = load <4 x half>, ptr addrspace(3) %address.local282, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v209 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v205, <4 x float> %v143) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %v210 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v208, <4 x float> %v209) #9, !dbg !8
  %v211 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !8
  %v212 = load <4 x half>, ptr addrspace(3) %address.local295, align 2, !dbg !8
  %v213 = shufflevector <4 x float> %v210, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %v214 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v211, <4 x float> %v144) #9, !dbg !8
  %v215 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v212, <4 x float> %v214) #9, !dbg !8
  %v216 = shufflevector <4 x float> %v215, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %v217 = shufflevector <4 x float> %v129, <4 x float> %v149, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v218 = shufflevector <32 x float> %v217, <32 x float> %v155, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v219 = shufflevector <32 x float> %v218, <32 x float> %v159, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v220 = shufflevector <32 x float> %v219, <32 x float> %v165, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v221 = shufflevector <32 x float> %v220, <32 x float> %v169, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v222 = shufflevector <32 x float> %v221, <32 x float> %v175, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v223 = shufflevector <32 x float> %v222, <32 x float> %v179, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %v224 = insertvalue [2 x <32 x float>] poison, <32 x float> %v223, 0, !dbg !22
  %v225 = shufflevector <4 x float> %v182, <4 x float> %v187, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v226 = shufflevector <32 x float> %v225, <32 x float> %v193, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v227 = shufflevector <32 x float> %v226, <32 x float> %v197, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v228 = shufflevector <32 x float> %v227, <32 x float> %v203, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v229 = shufflevector <32 x float> %v228, <32 x float> %v207, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v230 = shufflevector <32 x float> %v229, <32 x float> %v213, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %v231 = shufflevector <32 x float> %v230, <32 x float> %v216, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %v232 = insertvalue [2 x <32 x float>] %v224, <32 x float> %v231, 1, !dbg !22
  store [2 x <32 x float>] %v232, ptr addrspace(1) %v3, align 16
  ret void
}
; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #7

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!33}

!0 = !DIFile(filename: "test_input_sd16.mlir", directory: "3rd/deepgengraph/test")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!8 = !DILocation(line: 69, column: 26, scope: !4)
!10 = !DILocation(line: 46, column: 17, scope: !4)
!12 = !DILocation(line: 66, column: 22, scope: !4)
!13 = !DILocation(line: 68, column: 24, scope: !4)
!14 = !DILocation(line: 73, column: 19, scope: !4)
!15 = !DILocation(line: 93, column: 23, scope: !4)
!16 = !DILocation(line: 56, column: 24, scope: !4)
!17 = !DILocation(line: 67, column: 22, scope: !4)
!18 = !DILocation(line: 92, column: 20, scope: !4)
!19 = !DILocation(line: 97, column: 23, scope: !4)
!20 = !DILocation(line: 98, column: 19, scope: !4)
!21 = !DILocation(line: 99, column: 9, scope: !4)
!22 = !DILocation(line: 59, column: 17, scope: !4)
!23 = !DILocation(line: 75, column: 24, scope: !4)
!24 = !DILocation(line: 76, column: 23, scope: !4)
!25 = !DILocation(line: 77, column: 26, scope: !4)
!26 = !DILocation(line: 84, column: 21, scope: !4)
!27 = !DILocation(line: 85, column: 22, scope: !4)
!28 = !DILocation(line: 86, column: 20, scope: !4)
!29 = !DILocation(line: 87, column: 16, scope: !4)
!30 = !DILocation(line: 94, column: 20, scope: !4)
!31 = !DILocation(line: 70, column: 23, scope: !4)
!32 = !{i32 256, i32 1, i32 1}
!33 = !{i32 2, !"Debug Info Version", i32 3}
!34 = !{i32 0, i32 256}
