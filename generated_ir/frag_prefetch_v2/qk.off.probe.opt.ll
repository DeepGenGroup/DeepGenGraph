; ModuleID = '<stdin>'
source_filename = "3rd/deepgengraph/test/test_input_sd16.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: nounwind
define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %v0, ptr addrspace(1) %v1, ptr addrspace(1) nofree readnone captures(none) %v2, ptr addrspace(1) nofree writeonly captures(none) initializes((0, 256)) %v3) local_unnamed_addr #0 !dbg !3 !reqd_work_group_size !6 {
  %v5 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %v6 = sext i32 %v5 to i64, !dbg !7
  %v7 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %v8 = sext i32 %v7 to i64, !dbg !7
  %.lobit = ashr i32 %v5, 31, !dbg !8
  %v10 = xor i32 %.lobit, %v5, !dbg !8
  %v11 = sdiv i32 %v10, 32, !dbg !8
  %v12 = xor i32 %v11, %.lobit, !dbg !8
  %v13 = sext i32 %v12 to i64, !dbg !8
  %v14 = add nsw i64 %v13, %v8, !dbg !8
  %v19 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !9, !range !10
  %v34 = lshr i64 %v6, 5, !dbg !11
  %v35 = shl i64 %v34, 20, !dbg !11
  %v15 = shl nsw i64 %v14, 20, !dbg !8
  %.idx = and i64 %v15, 32505856, !dbg !8
  %1 = shl nsw i64 %v13, 20, !dbg !8
  %2 = shl nsw i64 %v6, 15, !dbg !8
  %.idx71 = sub nsw i64 %2, %1, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %v19) #5, !dbg !8, !range !10, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !9
  %address.local133 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local134 = and i64 %address.local133, 1536, !dbg !8
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local134, !dbg !8
  %address.local138 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local139 = and i64 %address.local138, 16256, !dbg !8
  %address.local140 = lshr i64 %address.local, 4, !dbg !8
  %address.local141 = and i64 %address.local140, 8, !dbg !8
  %address.local142 = or disjoint i64 %address.local139, %address.local141, !dbg !8
  %v49 = ptrtoint ptr addrspace(1) %v0 to i64, !dbg !8
  %3 = add nsw i64 %.idx, %.idx71, !dbg !8
  %address.local143.idx = shl nuw nsw i64 %address.local142, 1, !dbg !8
  %v51 = or disjoint i64 %address.local143.idx, %3, !dbg !8
  %dma.buffer.offset623 = trunc i64 %v51 to i32, !dbg !8
  %4 = insertelement <2 x i64> poison, i64 %v49, i64 0, !dbg !8
  %v56 = bitcast <2 x i64> %4 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc624 = shufflevector <4 x i32> %v56, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local144 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local145 = shl nuw nsw i64 %address.local144, 3, !dbg !8
  %address.local146 = and i64 %address.local145, 3584, !dbg !8
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !8
  %address.local152 = lshr i64 %address.local144, 4, !dbg !8
  %address.local153 = and i64 %address.local152, 24, !dbg !8
  %address.local154 = or disjoint i64 %address.local153, %address.local139, !dbg !8
  %address.local155.idx = shl nuw nsw i64 %address.local154, 1, !dbg !8
  %v65 = or disjoint i64 %address.local155.idx, %3, !dbg !8
  %dma.buffer.offset620 = trunc i64 %v65 to i32, !dbg !8
  %.lobit72 = ashr i64 %v6, 63, !dbg !11
  %v76 = xor i64 %.lobit72, %v6, !dbg !11
  %v77 = sdiv i64 %v76, 4096, !dbg !11
  %v78 = xor i64 %v77, %.lobit72, !dbg !11
  %v79 = add nsw i64 %v78, %v8, !dbg !11
  %v81 = and i64 %v34, 127, !dbg !11
  %v82 = shl nsw i64 %v79, 19, !dbg !11
  %v83 = and i64 %v82, 16252928, !dbg !11
  %.idx74 = sub i64 %2, %v35, !dbg !11
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local134, !dbg !11
  %v91 = ptrtoint ptr addrspace(1) %v1 to i64, !dbg !11
  %v84.idx1 = or disjoint i64 %v83, %v81, !dbg !11
  %5 = add nuw nsw i64 %address.local142, %v84.idx1, !dbg !11
  %6 = shl nuw nsw i64 %5, 1, !dbg !11
  %v93 = add i64 %6, %.idx74, !dbg !11
  %dma.buffer.offset617 = trunc i64 %v93 to i32, !dbg !11
  %7 = insertelement <2 x i64> poison, i64 %v91, i64 0, !dbg !11
  %v98 = bitcast <2 x i64> %7 to <4 x i32>, !dbg !11
  %dma.buffer.rsrc618 = shufflevector <4 x i32> %v98, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !11
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local146, !dbg !11
  %8 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local135, i64 0, !dbg !8
  %9 = insertelement <4 x ptr addrspace(3)> %8, ptr addrspace(3) %address.local147, i64 1, !dbg !8
  %10 = insertelement <4 x ptr addrspace(3)> %9, ptr addrspace(3) %address.local156, i64 2, !dbg !8
  %11 = insertelement <4 x ptr addrspace(3)> %10, ptr addrspace(3) %address.local160, i64 3, !dbg !8
  %12 = ptrtoint <4 x ptr addrspace(3)> %11 to <4 x i64>, !dbg !8
  %13 = trunc <4 x i64> %12 to <4 x i32>, !dbg !8
  %14 = extractelement <4 x i32> %13, i64 0, !dbg !8
  %v46 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %14), !dbg !8
  %dma.buffer.lds625 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v46), !dbg !8
  %v59 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds625, i32 %dma.buffer.offset623, <4 x i32> %dma.buffer.rsrc624) #6, !dbg !8
  %15 = extractelement <4 x i32> %13, i64 1, !dbg !8
  %v60 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %15), !dbg !8
  %dma.buffer.lds622 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v60), !dbg !8
  %v73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds622, i32 %dma.buffer.offset620, <4 x i32> %dma.buffer.rsrc624) #6, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #6, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %16 = extractelement <4 x i32> %13, i64 2, !dbg !11
  %v88 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %16), !dbg !11
  %dma.buffer.lds619 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v88), !dbg !11
  %v101 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds619, i32 %dma.buffer.offset617, <4 x i32> %dma.buffer.rsrc618) #6, !dbg !11
  %17 = extractelement <4 x i32> %13, i64 3, !dbg !11
  %v102 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %17), !dbg !11
  %18 = add nuw nsw i64 %address.local154, %v84.idx1, !dbg !11
  %19 = shl nuw nsw i64 %18, 1, !dbg !11
  %v107 = add i64 %19, %.idx74, !dbg !11
  %dma.buffer.offset = trunc i64 %v107 to i32, !dbg !11
  %dma.buffer.lds = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v102), !dbg !11
  %v115 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc618) #6, !dbg !11
  %20 = insertelement <4 x i32> poison, i32 %address.thread, i64 0, !dbg !9
  %21 = shufflevector <4 x i32> %20, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !9
  %22 = shl nuw nsw <4 x i32> %21, <i32 3, i32 1, i32 2, i32 5>, !dbg !9
  %23 = lshr <4 x i32> %21, <i32 3, i32 1, i32 2, i32 5>, !dbg !9
  %24 = shufflevector <4 x i32> %22, <4 x i32> %23, <4 x i32> <i32 0, i32 1, i32 6, i32 3>, !dbg !9
  %address.local166 = shl nuw nsw i32 %address.thread, 1, !dbg !9
  %25 = and <4 x i32> %24, <i32 120, i32 384, i32 4, i32 1024>, !dbg !9
  %26 = tail call i32 @llvm.vector.reduce.or.v4i32(<4 x i32> %25), !dbg !9
  %address.local175 = zext nneg i32 %26 to i64, !dbg !9
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !9
  %v116 = load <4 x half>, ptr addrspace(3) %address.local176, align 8, !dbg !9
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local176, i64 4096, !dbg !9
  %v117 = load <4 x half>, ptr addrspace(3) %address.local178, align 8, !dbg !9
  %address.local180 = getelementptr i8, ptr addrspace(3) %address.local176, i64 1024, !dbg !9
  %v118 = load <4 x half>, ptr addrspace(3) %address.local180, align 8, !dbg !9
  %v119 = fmul <4 x half> %v116, splat (half 1.275630e-01), !dbg !9
  %address.local182 = getelementptr i8, ptr addrspace(3) %address.local176, i64 5120, !dbg !9
  %v120 = load <4 x half>, ptr addrspace(3) %address.local182, align 8, !dbg !9
  %v121 = fmul <4 x half> %v117, splat (half 1.275630e-01), !dbg !9
  %v122 = fmul <4 x half> %v118, splat (half 1.275630e-01), !dbg !9
  %v123 = fmul <4 x half> %v120, splat (half 1.275630e-01), !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #6, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local186 = and i32 %address.local166, 24, !dbg !12
  %address.local191 = shl nuw nsw i32 %address.thread, 7, !dbg !12
  %address.local192 = and i32 %address.local191, 384, !dbg !12
  %27 = or disjoint i32 %address.local192, %address.local186, !dbg !12
  %28 = extractelement <4 x i32> %25, i64 2, !dbg !12
  %29 = or disjoint i32 %27, %28, !dbg !12
  %30 = extractelement <4 x i32> %25, i64 3, !dbg !12
  %address.local193 = or disjoint i32 %29, %30, !dbg !12
  %address.local194 = zext nneg i32 %address.local193 to i64, !dbg !12
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local194, !dbg !12
  %v124 = load <4 x half>, ptr addrspace(3) %address.local195, align 8, !dbg !12
  %v127 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v124, <4 x float> zeroinitializer) #5, !dbg !12
  %address.local209 = getelementptr i8, ptr addrspace(3) %address.local195, i64 4096, !dbg !12
  %v128 = load <4 x half>, ptr addrspace(3) %address.local209, align 8, !dbg !12
  %v130 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v128, <4 x float> %v127) #5, !dbg !12
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local138, !dbg !12
  %address.local212 = lshr i64 %address.local, 2, !dbg !12
  %address.local213 = mul nsw i64 %address.local212, -512, !dbg !12
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local213, !dbg !12
  %address.local215 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local216 = and i64 %address.local215, 24, !dbg !12
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local216, !dbg !12
  %address.local219 = and i64 %address.local212, 12, !dbg !9
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local219, !dbg !12
  %31 = and i64 %address.local, 32, !dbg !9
  %.not = icmp eq i64 %31, 0, !dbg !9
  %address.local223 = select i1 %.not, i64 0, i64 1016, !dbg !9
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local223, !dbg !12
  %address.local225 = getelementptr i8, ptr addrspace(3) %address.local224, i64 64, !dbg !12
  %v146 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !12
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local219, !dbg !12
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local223, !dbg !12
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local213, !dbg !12
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local216, !dbg !12
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4160, !dbg !12
  %v147 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !12
  %v148 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v146, <4 x float> zeroinitializer) #5, !dbg !12
  %address.local237 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !12
  %v149 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !12
  %v150 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v147, <4 x float> %v148) #5, !dbg !12
  %address.local243 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4224, !dbg !12
  %v151 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !12
  %v152 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v149, <4 x float> zeroinitializer) #5, !dbg !12
  %v153 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v151, <4 x float> %v152) #5, !dbg !12
  %address.local249 = getelementptr i8, ptr addrspace(3) %address.local224, i64 192, !dbg !12
  %v154 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !12
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4288, !dbg !12
  %v155 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !12
  %v156 = shufflevector <4 x float> %v153, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v157 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v154, <4 x float> zeroinitializer) #5, !dbg !12
  %v158 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v155, <4 x float> %v157) #5, !dbg !12
  %address.local257 = getelementptr i8, ptr addrspace(3) %address.local195, i64 1024, !dbg !12
  %v159 = load <4 x half>, ptr addrspace(3) %address.local257, align 8, !dbg !12
  %address.local259 = getelementptr i8, ptr addrspace(3) %address.local195, i64 5120, !dbg !12
  %v160 = load <4 x half>, ptr addrspace(3) %address.local259, align 8, !dbg !12
  %v161 = shufflevector <4 x float> %v158, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v162 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v159, <4 x float> zeroinitializer) #5, !dbg !12
  %v163 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v160, <4 x float> %v162) #5, !dbg !12
  %address.local261 = getelementptr i8, ptr addrspace(3) %address.local211, i64 1024, !dbg !12
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local213, !dbg !12
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) %address.local262, i64 %address.local216, !dbg !12
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local219, !dbg !12
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local223, !dbg !12
  %address.local266 = getelementptr i8, ptr addrspace(3) %address.local265, i64 64, !dbg !12
  %v164 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !12
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local219, !dbg !12
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) %address.local268, i64 %address.local223, !dbg !12
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local213, !dbg !12
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local216, !dbg !12
  %address.local272 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4160, !dbg !12
  %v165 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !12
  %v166 = shufflevector <4 x float> %v163, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v167 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v164, <4 x float> zeroinitializer) #5, !dbg !12
  %v168 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v165, <4 x float> %v167) #5, !dbg !12
  %address.local279 = getelementptr i8, ptr addrspace(3) %address.local265, i64 128, !dbg !12
  %v169 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !12
  %address.local285 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4224, !dbg !12
  %v170 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !12
  %v171 = shufflevector <4 x float> %v168, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v172 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v169, <4 x float> zeroinitializer) #5, !dbg !12
  %v173 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v170, <4 x float> %v172) #5, !dbg !12
  %address.local292 = getelementptr i8, ptr addrspace(3) %address.local265, i64 192, !dbg !12
  %v174 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !12
  %address.local298 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4288, !dbg !12
  %v175 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !12
  %v176 = shufflevector <4 x float> %v173, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v177 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v119, <4 x half> %v174, <4 x float> zeroinitializer) #5, !dbg !12
  %v178 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v175, <4 x float> %v177) #5, !dbg !12
  %v181 = shufflevector <4 x float> %v178, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v182 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v124, <4 x float> zeroinitializer) #5, !dbg !12
  %v183 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v128, <4 x float> %v182) #5, !dbg !12
  %v186 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v146, <4 x float> zeroinitializer) #5, !dbg !12
  %v187 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v147, <4 x float> %v186) #5, !dbg !12
  %v190 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v149, <4 x float> zeroinitializer) #5, !dbg !12
  %v191 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v151, <4 x float> %v190) #5, !dbg !12
  %v194 = shufflevector <4 x float> %v191, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v195 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v154, <4 x float> zeroinitializer) #5, !dbg !12
  %v196 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v155, <4 x float> %v195) #5, !dbg !12
  %v199 = shufflevector <4 x float> %v196, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v200 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v159, <4 x float> zeroinitializer) #5, !dbg !12
  %v201 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v160, <4 x float> %v200) #5, !dbg !12
  %v204 = shufflevector <4 x float> %v201, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v205 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v164, <4 x float> zeroinitializer) #5, !dbg !12
  %v206 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v165, <4 x float> %v205) #5, !dbg !12
  %v209 = shufflevector <4 x float> %v206, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v210 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v169, <4 x float> zeroinitializer) #5, !dbg !12
  %v211 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v170, <4 x float> %v210) #5, !dbg !12
  %v214 = shufflevector <4 x float> %v211, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v215 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v174, <4 x float> zeroinitializer) #5, !dbg !12
  %v216 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v123, <4 x half> %v175, <4 x float> %v215) #5, !dbg !12
  %v217 = shufflevector <4 x float> %v216, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v218 = shufflevector <4 x float> %v130, <4 x float> %v150, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v219 = shufflevector <32 x float> %v218, <32 x float> %v156, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v220 = shufflevector <32 x float> %v219, <32 x float> %v161, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v221 = shufflevector <32 x float> %v220, <32 x float> %v166, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v222 = shufflevector <32 x float> %v221, <32 x float> %v171, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v223 = shufflevector <32 x float> %v222, <32 x float> %v176, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v224 = shufflevector <32 x float> %v223, <32 x float> %v181, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !14
  %v226 = shufflevector <4 x float> %v183, <4 x float> %v187, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v227 = shufflevector <32 x float> %v226, <32 x float> %v194, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v228 = shufflevector <32 x float> %v227, <32 x float> %v199, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v229 = shufflevector <32 x float> %v228, <32 x float> %v204, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v230 = shufflevector <32 x float> %v229, <32 x float> %v209, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v231 = shufflevector <32 x float> %v230, <32 x float> %v214, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v232 = shufflevector <32 x float> %v231, <32 x float> %v217, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !14
  store <32 x float> %v224, ptr addrspace(1) %v3, align 16
  %v3.repack3 = getelementptr inbounds nuw i8, ptr addrspace(1) %v3, i64 128
  store <32 x float> %v232, ptr addrspace(1) %v3.repack3, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #2

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.or.v4i32(<4 x i32>) #4

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind willreturn memory(none) }
attributes #6 = { convergent nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_input_sd16.mlir", directory: "3rd/deepgengraph/test")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 22, type: !4, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 22, column: 3, scope: !3)
!8 = !DILocation(line: 66, column: 22, scope: !3)
!9 = !DILocation(line: 68, column: 24, scope: !3)
!10 = !{i32 0, i32 256}
!11 = !DILocation(line: 67, column: 22, scope: !3)
!12 = !DILocation(line: 69, column: 26, scope: !3)
!13 = !DILocation(line: 70, column: 23, scope: !3)
!14 = !DILocation(line: 59, column: 17, scope: !3)
