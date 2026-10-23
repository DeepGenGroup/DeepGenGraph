; ModuleID = 'temp.ll'
source_filename = "3rd/deepgengraph/test/test_input_sd16.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %v0, ptr addrspace(1) %v1, ptr addrspace(1) %v2, ptr addrspace(1) %v3) #1 !dbg !32 !reqd_work_group_size !55 {
  %v5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !33
  %v6 = sext i32 %v5 to i64, !dbg !33
  %v7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !33
  %v8 = sext i32 %v7 to i64, !dbg !33
  %v9 = shl nsw i64 %v6, 7, !dbg !35
  %.lobit = ashr i32 %v5, 31, !dbg !36
  %v10 = xor i32 %v5, %.lobit, !dbg !36
  %v11 = sdiv i32 %v10, 32, !dbg !36
  %.lobit1 = ashr i32 %v5, 31, !dbg !36
  %v12 = xor i32 %v11, %.lobit1, !dbg !36
  %v13 = sext i32 %v12 to i64, !dbg !36
  %v14 = add nsw i64 %v8, %v13, !dbg !36
  %v15 = and i64 %v14, 31, !dbg !36
  %v16 = shl nsw i64 %v6, 7, !dbg !36
  %v17 = mul nsw i64 %v13, -4096, !dbg !36
  %v18 = add nsw i64 %v16, %v17, !dbg !36
  %v19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !37, !range !57
  %v20 = zext nneg i32 %v19 to i64, !dbg !37
  %v21 = lshr i64 %v20, 6, !dbg !37
  %v22 = or disjoint i64 %v9, 127, !dbg !38
  %v23 = add nsw i64 %v9, 128, !dbg !38
  %v24 = icmp uge i64 %v23, %v22, !dbg !38
  %v26 = add i64 %v6, 0
  %v38 = add i64 0, 0
  %v39 = insertvalue [2 x <32 x float>] zeroinitializer, <32 x float> zeroinitializer, 0
  %v33 = shl nsw i64 %v26, 7, !dbg !40
  %v34 = lshr i64 %v26, 5, !dbg !41
  %v35 = shl i64 %v34, 12, !dbg !41
  %v36 = sub i64 %v33, %v35, !dbg !41
  %v42 = shl nuw nsw i64 %v38, 5, !dbg !46
  %.idx = shl nuw nsw i64 %v15, 20, !dbg !36
  %v43 = getelementptr i8, ptr addrspace(1) %v0, i64 %.idx, !dbg !36
  %.idx71 = shl nsw i64 %v18, 8, !dbg !36
  %v44 = getelementptr i8, ptr addrspace(1) %v43, i64 %.idx71, !dbg !36
  %v45 = getelementptr [2 x i8], ptr addrspace(1) %v44, i64 %v42, !dbg !36
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %v19) #9, !dbg !36, !range !57, !frisk.register_lifetimes !29
  %address.local = zext nneg i32 %address.thread to i64, !dbg !37
  %address.local133 = shl nuw nsw i64 %address.local, 3, !dbg !36
  %address.local134 = and i64 %address.local133, 1536, !dbg !36
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local134, !dbg !36
  %address.local136 = ptrtoint ptr addrspace(3) %address.local135 to i64, !dbg !36
  %address.local137 = trunc i64 %address.local136 to i32, !dbg !36
  %v46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local137), !dbg !36
  %v47 = zext i32 %v46 to i64, !dbg !36
  %v48 = inttoptr i64 %v47 to ptr addrspace(3), !dbg !36
  %address.local138 = shl nuw nsw i64 %address.local, 7, !dbg !36
  %address.local139 = and i64 %address.local138, 16256, !dbg !36
  %address.local140 = lshr i64 %address.local, 4, !dbg !36
  %address.local141 = and i64 %address.local140, 8, !dbg !36
  %address.local142 = or disjoint i64 %address.local139, %address.local141, !dbg !36
  %address.local143 = getelementptr [2 x i8], ptr addrspace(1) %v45, i64 %address.local142, !dbg !36
  %v49 = ptrtoint ptr addrspace(1) %v0 to i64, !dbg !36
  %v50 = ptrtoint ptr addrspace(1) %address.local143 to i64, !dbg !36
  %v51 = sub i64 %v50, %v49, !dbg !36
  %dma.buffer.offset623 = trunc i64 %v51 to i32, !dbg !36
  %v52 = trunc i64 %v49 to i32, !dbg !36
  %v53 = insertelement <4 x i32> poison, i32 %v52, i64 0, !dbg !36
  %v54 = lshr i64 %v49, 32, !dbg !36
  %v55 = trunc i64 %v54 to i32, !dbg !36
  %v56 = insertelement <4 x i32> %v53, i32 %v55, i64 1, !dbg !36
  %v57 = insertelement <4 x i32> %v56, i32 33554432, i64 2, !dbg !36
  %dma.buffer.rsrc624 = insertelement <4 x i32> %v57, i32 131072, i64 3, !dbg !36
  %v58 = ptrtoint ptr addrspace(3) %v48 to i32, !dbg !36
  %dma.buffer.lds625 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v58), !dbg !36
  %v59 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds625, i32 %dma.buffer.offset623, <4 x i32> %dma.buffer.rsrc624) #10, !dbg !36
  %address.local144 = or disjoint i64 %address.local, 256, !dbg !36
  %address.local145 = shl nuw nsw i64 %address.local144, 3, !dbg !36
  %address.local146 = and i64 %address.local145, 3584, !dbg !36
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !36
  %address.local148 = ptrtoint ptr addrspace(3) %address.local147 to i64, !dbg !36
  %address.local149 = trunc i64 %address.local148 to i32, !dbg !36
  %v60 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local149), !dbg !36
  %v61 = zext i32 %v60 to i64, !dbg !36
  %v62 = inttoptr i64 %v61 to ptr addrspace(3), !dbg !36
  %address.local150 = shl nuw nsw i64 %address.local, 7, !dbg !36
  %address.local151 = and i64 %address.local150, 16256, !dbg !36
  %address.local152 = lshr i64 %address.local144, 4, !dbg !36
  %address.local153 = and i64 %address.local152, 24, !dbg !36
  %address.local154 = or disjoint i64 %address.local151, %address.local153, !dbg !36
  %address.local155 = getelementptr [2 x i8], ptr addrspace(1) %v45, i64 %address.local154, !dbg !36
  %v63 = ptrtoint ptr addrspace(1) %v0 to i64, !dbg !36
  %v64 = ptrtoint ptr addrspace(1) %address.local155 to i64, !dbg !36
  %v65 = sub i64 %v64, %v63, !dbg !36
  %dma.buffer.offset620 = trunc i64 %v65 to i32, !dbg !36
  %v66 = trunc i64 %v63 to i32, !dbg !36
  %v67 = insertelement <4 x i32> poison, i32 %v66, i64 0, !dbg !36
  %v68 = lshr i64 %v63, 32, !dbg !36
  %v69 = trunc i64 %v68 to i32, !dbg !36
  %v70 = insertelement <4 x i32> %v67, i32 %v69, i64 1, !dbg !36
  %v71 = insertelement <4 x i32> %v70, i32 33554432, i64 2, !dbg !36
  %dma.buffer.rsrc621 = insertelement <4 x i32> %v71, i32 131072, i64 3, !dbg !36
  %v72 = ptrtoint ptr addrspace(3) %v62 to i32, !dbg !36
  %dma.buffer.lds622 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v72), !dbg !36
  %v73 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds622, i32 %dma.buffer.offset620, <4 x i32> %dma.buffer.rsrc621) #10, !dbg !36
  %v74 = shl nuw nsw i64 %v38, 10, !dbg !41
  %v75 = add nuw i64 %v26, %v74, !dbg !41
  %.lobit72 = ashr i64 %v75, 63, !dbg !41
  %v76 = xor i64 %v75, %.lobit72, !dbg !41
  %v77 = sdiv i64 %v76, 4096, !dbg !41
  %.lobit73 = ashr i64 %v75, 63, !dbg !41
  %v78 = xor i64 %v77, %.lobit73, !dbg !41
  %v79 = add nsw i64 %v78, %v8, !dbg !41
  %v80 = add nuw nsw i64 %v42, %v34, !dbg !41
  %v81 = and i64 %v80, 127, !dbg !41
  %v82 = shl i64 %v79, 19, !dbg !41
  %v83 = and i64 %v82, 16252928, !dbg !41
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %v84 = getelementptr [2 x i8], ptr addrspace(1) %v1, i64 %v83, !dbg !41
  %v85 = getelementptr [2 x i8], ptr addrspace(1) %v84, i64 %v81, !dbg !41
  %.idx74 = shl i64 %v36, 8, !dbg !41
  %v86 = getelementptr i8, ptr addrspace(1) %v85, i64 %.idx74, !dbg !41
  %v87 = getelementptr inbounds nuw i8, ptr addrspace(3) @shm_0, i64 8192, !dbg !34
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local134, !dbg !41
  %address.local157 = ptrtoint ptr addrspace(3) %address.local156 to i64, !dbg !41
  %address.local158 = trunc i64 %address.local157 to i32, !dbg !41
  %v88 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !41
  %v89 = zext i32 %v88 to i64, !dbg !41
  %v90 = inttoptr i64 %v89 to ptr addrspace(3), !dbg !41
  %address.local159 = getelementptr [2 x i8], ptr addrspace(1) %v86, i64 %address.local142, !dbg !41
  %v91 = ptrtoint ptr addrspace(1) %v1 to i64, !dbg !41
  %v92 = ptrtoint ptr addrspace(1) %address.local159 to i64, !dbg !41
  %v93 = sub i64 %v92, %v91, !dbg !41
  %dma.buffer.offset617 = trunc i64 %v93 to i32, !dbg !41
  %v94 = trunc i64 %v91 to i32, !dbg !41
  %v95 = insertelement <4 x i32> poison, i32 %v94, i64 0, !dbg !41
  %v96 = lshr i64 %v91, 32, !dbg !41
  %v97 = trunc i64 %v96 to i32, !dbg !41
  %v98 = insertelement <4 x i32> %v95, i32 %v97, i64 1, !dbg !41
  %v99 = insertelement <4 x i32> %v98, i32 33554432, i64 2, !dbg !41
  %dma.buffer.rsrc618 = insertelement <4 x i32> %v99, i32 131072, i64 3, !dbg !41
  %v100 = ptrtoint ptr addrspace(3) %v90 to i32, !dbg !41
  %dma.buffer.lds619 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v100), !dbg !41
  %v101 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds619, i32 %dma.buffer.offset617, <4 x i32> %dma.buffer.rsrc618) #10, !dbg !41
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local146, !dbg !41
  %address.local161 = ptrtoint ptr addrspace(3) %address.local160 to i64, !dbg !41
  %address.local162 = trunc i64 %address.local161 to i32, !dbg !41
  %v102 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local162), !dbg !41
  %v103 = zext i32 %v102 to i64, !dbg !41
  %v104 = inttoptr i64 %v103 to ptr addrspace(3), !dbg !41
  %address.local163 = getelementptr [2 x i8], ptr addrspace(1) %v86, i64 %address.local154, !dbg !41
  %v105 = ptrtoint ptr addrspace(1) %v1 to i64, !dbg !41
  %v106 = ptrtoint ptr addrspace(1) %address.local163 to i64, !dbg !41
  %v107 = sub i64 %v106, %v105, !dbg !41
  %dma.buffer.offset = trunc i64 %v107 to i32, !dbg !41
  %v108 = trunc i64 %v105 to i32, !dbg !41
  %v109 = insertelement <4 x i32> poison, i32 %v108, i64 0, !dbg !41
  %v110 = lshr i64 %v105, 32, !dbg !41
  %v111 = trunc i64 %v110 to i32, !dbg !41
  %v112 = insertelement <4 x i32> %v109, i32 %v111, i64 1, !dbg !41
  %v113 = insertelement <4 x i32> %v112, i32 33554432, i64 2, !dbg !41
  %dma.buffer.rsrc = insertelement <4 x i32> %v113, i32 131072, i64 3, !dbg !41
  %v114 = ptrtoint ptr addrspace(3) %v104 to i32, !dbg !41
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v114), !dbg !41
  %v115 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !41
  %address.local164 = lshr i32 %address.thread, 2, !dbg !37
  %address.local165 = and i32 %address.local164, 4, !dbg !37
  %address.local166 = shl nuw nsw i32 %address.thread, 1, !dbg !37
  %address.local167 = and i32 %address.local166, 384, !dbg !37
  %address.local168 = or disjoint i32 %address.local165, %address.local167, !dbg !37
  %address.local169 = shl nuw nsw i32 %address.thread, 3, !dbg !37
  %address.local170 = and i32 %address.local169, 120, !dbg !37
  %address.local171 = or disjoint i32 %address.local168, %address.local170, !dbg !37
  %address.local172 = shl nuw nsw i32 %address.thread, 5, !dbg !37
  %address.local173 = and i32 %address.local172, 1024, !dbg !37
  %address.local174 = or disjoint i32 %address.local171, %address.local173, !dbg !37
  %address.local175 = zext nneg i32 %address.local174 to i64, !dbg !37
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !37
  %v116 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !37
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !37
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local177, i64 4096, !dbg !37
  %v117 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !37
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !37
  %address.local180 = getelementptr i8, ptr addrspace(3) %address.local179, i64 1024, !dbg !37
  %v118 = load <4 x half>, ptr addrspace(3) %address.local180, align 2, !dbg !37
  %v119 = fmul <4 x half> %v116, splat (half 0xH3015), !dbg !37
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !37
  %address.local182 = getelementptr i8, ptr addrspace(3) %address.local181, i64 5120, !dbg !37
  %v120 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !37
  %v121 = fmul <4 x half> %v117, splat (half 0xH3015), !dbg !37
  %v122 = fmul <4 x half> %v118, splat (half 0xH3015), !dbg !37
  %v123 = fmul <4 x half> %v120, splat (half 0xH3015), !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %address.local183 = lshr i32 %address.thread, 2, !dbg !34
  %address.local184 = and i32 %address.local183, 4, !dbg !34
  %address.local185 = shl nuw nsw i32 %address.thread, 1, !dbg !34
  %address.local186 = and i32 %address.local185, 24, !dbg !34
  %address.local187 = or disjoint i32 %address.local184, %address.local186, !dbg !34
  %address.local188 = shl nuw nsw i32 %address.thread, 5, !dbg !34
  %address.local189 = and i32 %address.local188, 1024, !dbg !34
  %address.local190 = or disjoint i32 %address.local187, %address.local189, !dbg !34
  %address.local191 = shl nuw nsw i32 %address.thread, 7, !dbg !34
  %address.local192 = and i32 %address.local191, 384, !dbg !34
  %address.local193 = or disjoint i32 %address.local190, %address.local192, !dbg !34
  %address.local194 = zext nneg i32 %address.local193 to i64, !dbg !34
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local194, !dbg !34
  %v124 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !34
  %v125 = extractvalue [2 x <32 x float>] %v39, 0, !dbg !54
  %v126 = shufflevector <32 x float> %v125, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %address.local196 = lshr i32 %address.thread, 2, !dbg !34
  %address.local197 = and i32 %address.local196, 4, !dbg !34
  %address.local198 = shl nuw nsw i32 %address.thread, 1, !dbg !34
  %address.local199 = and i32 %address.local198, 24, !dbg !34
  %address.local200 = or disjoint i32 %address.local197, %address.local199, !dbg !34
  %address.local201 = shl nuw nsw i32 %address.thread, 5, !dbg !34
  %address.local202 = and i32 %address.local201, 1024, !dbg !34
  %address.local203 = or disjoint i32 %address.local200, %address.local202, !dbg !34
  %address.local204 = shl nuw nsw i32 %address.thread, 7, !dbg !34
  %address.local205 = and i32 %address.local204, 384, !dbg !34
  %address.local206 = or disjoint i32 %address.local203, %address.local205, !dbg !34
  %address.local207 = zext nneg i32 %address.local206 to i64, !dbg !34
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local207, !dbg !34
  %address.local209 = getelementptr i8, ptr addrspace(3) %address.local208, i64 4096, !dbg !34
  %v127 = load <4 x half>, ptr addrspace(3) %address.local209, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v128 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v124, <4 x float> %v126) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !54
  %v129 = shufflevector <32 x float> %v125, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %v130 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v127, <4 x float> %v128) #9, !dbg !34
  %v131 = shufflevector <32 x float> %v125, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !54
  %v132 = shufflevector <32 x float> %v125, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !54
  %v133 = shufflevector <32 x float> %v125, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !54
  %v134 = shufflevector <32 x float> %v125, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !54
  %v135 = shufflevector <32 x float> %v125, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !54
  %v136 = shufflevector <32 x float> %v125, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !54
  %v137 = extractvalue [2 x <32 x float>] %v39, 1, !dbg !54
  %v138 = shufflevector <32 x float> %v137, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %v139 = shufflevector <32 x float> %v137, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %v140 = shufflevector <32 x float> %v137, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !54
  %v141 = shufflevector <32 x float> %v137, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !54
  %v142 = shufflevector <32 x float> %v137, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !54
  %v143 = shufflevector <32 x float> %v137, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !54
  %v144 = shufflevector <32 x float> %v137, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !54
  %v145 = shufflevector <32 x float> %v137, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !54
  %address.local210 = shl nuw nsw i64 %address.local, 7, !dbg !34
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local210, !dbg !34
  %address.local212 = lshr i64 %address.local, 2, !dbg !34
  %address.local213 = mul nsw i64 %address.local212, -512, !dbg !34
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local213, !dbg !34
  %address.local215 = shl nuw nsw i64 %address.local, 1, !dbg !34
  %address.local216 = and i64 %address.local215, 24, !dbg !34
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local216, !dbg !34
  %address.local218 = lshr i64 %address.local, 2, !dbg !37
  %address.local219 = and i64 %address.local218, 12, !dbg !37
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local219, !dbg !34
  %address.local221 = and i64 %address.local, 63, !dbg !37
  %address.local222 = lshr i64 %address.local221, 5, !dbg !37
  %address.local223 = mul nuw nsw i64 %address.local222, 1016, !dbg !37
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local223, !dbg !34
  %address.local225 = getelementptr i8, ptr addrspace(3) %address.local224, i64 64, !dbg !34
  %v146 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !34
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local210, !dbg !34
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local219, !dbg !34
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local223, !dbg !34
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local213, !dbg !34
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local216, !dbg !34
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4160, !dbg !34
  %v147 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !34
  %v148 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v146, <4 x float> %v129) #9, !dbg !34
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local210, !dbg !34
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) %address.local232, i64 %address.local213, !dbg !34
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) %address.local233, i64 %address.local216, !dbg !34
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local219, !dbg !34
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local223, !dbg !34
  %address.local237 = getelementptr i8, ptr addrspace(3) %address.local236, i64 128, !dbg !34
  %v149 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !34
  %v150 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v147, <4 x float> %v148) #9, !dbg !34
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local210, !dbg !34
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) %address.local238, i64 %address.local219, !dbg !34
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) %address.local239, i64 %address.local223, !dbg !34
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local213, !dbg !34
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local216, !dbg !34
  %address.local243 = getelementptr i8, ptr addrspace(3) %address.local242, i64 4224, !dbg !34
  %v151 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !34
  %v152 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v149, <4 x float> %v131) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local210, !dbg !34
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local213, !dbg !34
  %address.local246 = getelementptr [2 x i8], ptr addrspace(3) %address.local245, i64 %address.local216, !dbg !34
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) %address.local246, i64 %address.local219, !dbg !34
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local223, !dbg !34
  %address.local249 = getelementptr i8, ptr addrspace(3) %address.local248, i64 192, !dbg !34
  %v153 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v154 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v151, <4 x float> %v152) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local210, !dbg !34
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) %address.local250, i64 %address.local219, !dbg !34
  %address.local252 = getelementptr [2 x i8], ptr addrspace(3) %address.local251, i64 %address.local223, !dbg !34
  %address.local253 = getelementptr [2 x i8], ptr addrspace(3) %address.local252, i64 %address.local213, !dbg !34
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) %address.local253, i64 %address.local216, !dbg !34
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local254, i64 4288, !dbg !34
  %v155 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !34
  %v156 = shufflevector <4 x float> %v154, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v157 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v153, <4 x float> %v132) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %address.local256 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local194, !dbg !34
  %address.local257 = getelementptr i8, ptr addrspace(3) %address.local256, i64 1024, !dbg !34
  %v158 = load <4 x half>, ptr addrspace(3) %address.local257, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v159 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v155, <4 x float> %v157) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local207, !dbg !34
  %address.local259 = getelementptr i8, ptr addrspace(3) %address.local258, i64 5120, !dbg !34
  %v160 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !34
  %v161 = shufflevector <4 x float> %v159, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v162 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v158, <4 x float> %v133) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %address.local260 = add nuw nsw i64 %address.local210, 512, !dbg !34
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local260, !dbg !34
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local213, !dbg !34
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) %address.local262, i64 %address.local216, !dbg !34
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local219, !dbg !34
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local223, !dbg !34
  %address.local266 = getelementptr i8, ptr addrspace(3) %address.local265, i64 64, !dbg !34
  %v163 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v164 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v160, <4 x float> %v162) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local260, !dbg !34
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local219, !dbg !34
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) %address.local268, i64 %address.local223, !dbg !34
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local213, !dbg !34
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local216, !dbg !34
  %address.local272 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4160, !dbg !34
  %v165 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !34
  %v166 = shufflevector <4 x float> %v164, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v167 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v163, <4 x float> %v134) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %address.local273 = add nuw nsw i64 %address.local210, 512, !dbg !34
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local273, !dbg !34
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local213, !dbg !34
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) %address.local275, i64 %address.local216, !dbg !34
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) %address.local276, i64 %address.local219, !dbg !34
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local223, !dbg !34
  %address.local279 = getelementptr i8, ptr addrspace(3) %address.local278, i64 128, !dbg !34
  %v168 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v169 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v165, <4 x float> %v167) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local273, !dbg !34
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %address.local280, i64 %address.local219, !dbg !34
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) %address.local281, i64 %address.local223, !dbg !34
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) %address.local282, i64 %address.local213, !dbg !34
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %address.local283, i64 %address.local216, !dbg !34
  %address.local285 = getelementptr i8, ptr addrspace(3) %address.local284, i64 4224, !dbg !34
  %v170 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !34
  %v171 = shufflevector <4 x float> %v169, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v172 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v168, <4 x float> %v135) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %address.local286 = add nuw nsw i64 %address.local210, 512, !dbg !34
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local286, !dbg !34
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) %address.local287, i64 %address.local213, !dbg !34
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) %address.local288, i64 %address.local216, !dbg !34
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %address.local289, i64 %address.local219, !dbg !34
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local223, !dbg !34
  %address.local292 = getelementptr i8, ptr addrspace(3) %address.local291, i64 192, !dbg !34
  %v173 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v174 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v170, <4 x float> %v172) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %v87, i64 %address.local286, !dbg !34
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) %address.local293, i64 %address.local219, !dbg !34
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) %address.local294, i64 %address.local223, !dbg !34
  %address.local296 = getelementptr [2 x i8], ptr addrspace(3) %address.local295, i64 %address.local213, !dbg !34
  %address.local297 = getelementptr [2 x i8], ptr addrspace(3) %address.local296, i64 %address.local216, !dbg !34
  %address.local298 = getelementptr i8, ptr addrspace(3) %address.local297, i64 4288, !dbg !34
  %v175 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !34
  %v176 = shufflevector <4 x float> %v174, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v177 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v173, <4 x float> %v136) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v178 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v179 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v175, <4 x float> %v177) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v180 = load <4 x half>, ptr addrspace(3) %address.local209, align 2, !dbg !34
  %v181 = shufflevector <4 x float> %v179, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v182 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v178, <4 x float> %v138) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v183 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v184 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v180, <4 x float> %v182) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v185 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !34
  %v186 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v183, <4 x float> %v139) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v187 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v188 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v185, <4 x float> %v186) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v189 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !34
  %v190 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v187, <4 x float> %v140) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v191 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v192 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v189, <4 x float> %v190) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v193 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !34
  %v194 = shufflevector <4 x float> %v192, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v195 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v191, <4 x float> %v141) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v196 = load <4 x half>, ptr addrspace(3) %address.local257, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v197 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v193, <4 x float> %v195) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v198 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !34
  %v199 = shufflevector <4 x float> %v197, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v200 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v196, <4 x float> %v142) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v201 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v202 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v198, <4 x float> %v200) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v203 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !34
  %v204 = shufflevector <4 x float> %v202, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v205 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v201, <4 x float> %v143) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v206 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v207 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v203, <4 x float> %v205) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v208 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !34
  %v209 = shufflevector <4 x float> %v207, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v210 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v206, <4 x float> %v144) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v211 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v212 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v208, <4 x float> %v210) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) memory(inaccessiblemem: readwrite), !dbg !34
  %v213 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !34
  %v214 = shufflevector <4 x float> %v212, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v215 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v211, <4 x float> %v145) #9, !dbg !34
  %v216 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v213, <4 x float> %v215) #9, !dbg !34
  %v217 = shufflevector <4 x float> %v216, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %v218 = shufflevector <4 x float> %v130, <4 x float> %v150, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v219 = shufflevector <32 x float> %v218, <32 x float> %v156, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v220 = shufflevector <32 x float> %v219, <32 x float> %v161, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v221 = shufflevector <32 x float> %v220, <32 x float> %v166, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v222 = shufflevector <32 x float> %v221, <32 x float> %v171, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v223 = shufflevector <32 x float> %v222, <32 x float> %v176, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v224 = shufflevector <32 x float> %v223, <32 x float> %v181, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !46
  %v225 = insertvalue [2 x <32 x float>] poison, <32 x float> %v224, 0, !dbg !46
  %v226 = shufflevector <4 x float> %v184, <4 x float> %v188, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v227 = shufflevector <32 x float> %v226, <32 x float> %v194, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v228 = shufflevector <32 x float> %v227, <32 x float> %v199, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v229 = shufflevector <32 x float> %v228, <32 x float> %v204, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v230 = shufflevector <32 x float> %v229, <32 x float> %v209, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v231 = shufflevector <32 x float> %v230, <32 x float> %v214, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %v232 = shufflevector <32 x float> %v231, <32 x float> %v217, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !46
  %v233 = insertvalue [2 x <32 x float>] %v225, <32 x float> %v232, 1, !dbg !46
  store [2 x <32 x float>] %v233, ptr addrspace(1) %v3, align 16
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

!llvm.dbg.cu = !{!31}
!llvm.module.flags = !{!56}

!28 = !DIFile(filename: "test_input_sd16.mlir", directory: "3rd/deepgengraph/test")
!29 = !{}
!30 = !DISubroutineType(types: !29)
!31 = distinct !DICompileUnit(language: DW_LANG_C, file: !28, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!32 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !28, file: !28, line: 22, type: !30, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !31)
!33 = !DILocation(line: 22, column: 3, scope: !32)
!34 = !DILocation(line: 69, column: 26, scope: !32)
!35 = !DILocation(line: 46, column: 17, scope: !32)
!36 = !DILocation(line: 66, column: 22, scope: !32)
!37 = !DILocation(line: 68, column: 24, scope: !32)
!38 = !DILocation(line: 73, column: 19, scope: !32)
!39 = !DILocation(line: 93, column: 23, scope: !32)
!40 = !DILocation(line: 56, column: 24, scope: !32)
!41 = !DILocation(line: 67, column: 22, scope: !32)
!42 = !DILocation(line: 92, column: 20, scope: !32)
!43 = !DILocation(line: 97, column: 23, scope: !32)
!44 = !DILocation(line: 98, column: 19, scope: !32)
!45 = !DILocation(line: 99, column: 9, scope: !32)
!46 = !DILocation(line: 59, column: 17, scope: !32)
!47 = !DILocation(line: 75, column: 24, scope: !32)
!48 = !DILocation(line: 76, column: 23, scope: !32)
!49 = !DILocation(line: 77, column: 26, scope: !32)
!50 = !DILocation(line: 85, column: 22, scope: !32)
!51 = !DILocation(line: 86, column: 20, scope: !32)
!52 = !DILocation(line: 87, column: 16, scope: !32)
!53 = !DILocation(line: 94, column: 20, scope: !32)
!54 = !DILocation(line: 70, column: 23, scope: !32)
!55 = !{i32 256, i32 1, i32 1}
!56 = !{i32 2, !"Debug Info Version", i32 3}
!57 = !{i32 0, i32 256}
