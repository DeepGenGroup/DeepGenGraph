; ModuleID = 'generated_ir/sd16_perf_repair/qk.on.probe.ll'
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
  %.idx68 = sub nsw i64 %2, %1, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %v19) #5, !dbg !8, !range !10, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !9
  %address.local130 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local131 = and i64 %address.local130, 1536, !dbg !8
  %address.local132 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local131, !dbg !8
  %address.local135 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local136 = and i64 %address.local135, 16256, !dbg !8
  %address.local137 = lshr i64 %address.local, 4, !dbg !8
  %address.local138 = and i64 %address.local137, 8, !dbg !8
  %address.local139 = or disjoint i64 %address.local136, %address.local138, !dbg !8
  %v49 = ptrtoint ptr addrspace(1) %v0 to i64, !dbg !8
  %3 = add nsw i64 %.idx, %.idx68, !dbg !8
  %address.local140.idx = shl nuw nsw i64 %address.local139, 1, !dbg !8
  %v51 = or disjoint i64 %address.local140.idx, %3, !dbg !8
  %dma.buffer.offset876 = trunc i64 %v51 to i32, !dbg !8
  %4 = insertelement <2 x i64> poison, i64 %v49, i64 0, !dbg !8
  %v56 = bitcast <2 x i64> %4 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc877 = shufflevector <4 x i32> %v56, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local141 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local142 = shl nuw nsw i64 %address.local141, 3, !dbg !8
  %address.local143 = and i64 %address.local142, 3584, !dbg !8
  %address.local144 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local143, !dbg !8
  %address.local149 = lshr i64 %address.local141, 4, !dbg !8
  %address.local150 = and i64 %address.local149, 24, !dbg !8
  %address.local151 = or disjoint i64 %address.local150, %address.local136, !dbg !8
  %address.local152.idx = shl nuw nsw i64 %address.local151, 1, !dbg !8
  %v65 = or disjoint i64 %address.local152.idx, %3, !dbg !8
  %dma.buffer.offset873 = trunc i64 %v65 to i32, !dbg !8
  %.lobit69 = ashr i64 %v6, 63, !dbg !11
  %v76 = xor i64 %.lobit69, %v6, !dbg !11
  %v77 = sdiv i64 %v76, 4096, !dbg !11
  %v78 = xor i64 %v77, %.lobit69, !dbg !11
  %v79 = add nsw i64 %v78, %v8, !dbg !11
  %v81 = and i64 %v34, 127, !dbg !11
  %v82 = shl nsw i64 %v79, 19, !dbg !11
  %v83 = and i64 %v82, 16252928, !dbg !11
  %.idx71 = sub i64 %2, %v35, !dbg !11
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local131, !dbg !11
  %v90 = ptrtoint ptr addrspace(1) %v1 to i64, !dbg !11
  %v84.idx1 = or disjoint i64 %v83, %v81, !dbg !11
  %5 = add nuw nsw i64 %address.local139, %v84.idx1, !dbg !11
  %6 = shl nuw nsw i64 %5, 1, !dbg !11
  %v92 = add i64 %6, %.idx71, !dbg !11
  %dma.buffer.offset870 = trunc i64 %v92 to i32, !dbg !11
  %7 = insertelement <2 x i64> poison, i64 %v90, i64 0, !dbg !11
  %v97 = bitcast <2 x i64> %7 to <4 x i32>, !dbg !11
  %dma.buffer.rsrc871 = shufflevector <4 x i32> %v97, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !11
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local143, !dbg !11
  %8 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local132, i64 0, !dbg !8
  %9 = insertelement <4 x ptr addrspace(3)> %8, ptr addrspace(3) %address.local144, i64 1, !dbg !8
  %10 = insertelement <4 x ptr addrspace(3)> %9, ptr addrspace(3) %address.local153, i64 2, !dbg !8
  %11 = insertelement <4 x ptr addrspace(3)> %10, ptr addrspace(3) %address.local157, i64 3, !dbg !8
  %12 = ptrtoint <4 x ptr addrspace(3)> %11 to <4 x i64>, !dbg !8
  %13 = trunc <4 x i64> %12 to <4 x i32>, !dbg !8
  %14 = extractelement <4 x i32> %13, i64 0, !dbg !8
  %v46 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %14), !dbg !8
  %dma.buffer.lds878 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v46), !dbg !8
  %v59 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds878, i32 %dma.buffer.offset876, <4 x i32> %dma.buffer.rsrc877) #6, !dbg !8
  %15 = extractelement <4 x i32> %13, i64 1, !dbg !8
  %v60 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %15), !dbg !8
  %dma.buffer.lds875 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v60), !dbg !8
  %v73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds875, i32 %dma.buffer.offset873, <4 x i32> %dma.buffer.rsrc877) #6, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #6, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %16 = extractelement <4 x i32> %13, i64 2, !dbg !11
  %v87 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %16), !dbg !11
  %dma.buffer.lds872 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v87), !dbg !11
  %v100 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds872, i32 %dma.buffer.offset870, <4 x i32> %dma.buffer.rsrc871) #6, !dbg !11
  %17 = extractelement <4 x i32> %13, i64 3, !dbg !11
  %v101 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %17), !dbg !11
  %18 = add nuw nsw i64 %address.local151, %v84.idx1, !dbg !11
  %19 = shl nuw nsw i64 %18, 1, !dbg !11
  %v106 = add i64 %19, %.idx71, !dbg !11
  %dma.buffer.offset = trunc i64 %v106 to i32, !dbg !11
  %dma.buffer.lds = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %v101), !dbg !11
  %v114 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc871) #6, !dbg !11
  %20 = insertelement <4 x i32> poison, i32 %address.thread, i64 0, !dbg !9
  %21 = shufflevector <4 x i32> %20, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !9
  %22 = shl nuw nsw <4 x i32> %21, <i32 3, i32 1, i32 2, i32 5>, !dbg !9
  %23 = lshr <4 x i32> %21, <i32 3, i32 1, i32 2, i32 5>, !dbg !9
  %24 = shufflevector <4 x i32> %22, <4 x i32> %23, <4 x i32> <i32 0, i32 1, i32 6, i32 3>, !dbg !9
  %address.local163 = shl nuw nsw i32 %address.thread, 1, !dbg !9
  %25 = and <4 x i32> %24, <i32 120, i32 384, i32 4, i32 1024>, !dbg !9
  %26 = tail call i32 @llvm.vector.reduce.or.v4i32(<4 x i32> %25), !dbg !9
  %address.local172 = zext nneg i32 %26 to i64, !dbg !9
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !9
  %v115 = load <4 x half>, ptr addrspace(3) %address.local173, align 8, !dbg !9
  %address.local175 = getelementptr i8, ptr addrspace(3) %address.local173, i64 4096, !dbg !9
  %v116 = load <4 x half>, ptr addrspace(3) %address.local175, align 8, !dbg !9
  %address.local177 = getelementptr i8, ptr addrspace(3) %address.local173, i64 1024, !dbg !9
  %v117 = load <4 x half>, ptr addrspace(3) %address.local177, align 8, !dbg !9
  %v118 = fmul <4 x half> %v115, splat (half 1.275630e-01), !dbg !9
  %address.local179 = getelementptr i8, ptr addrspace(3) %address.local173, i64 5120, !dbg !9
  %v119 = load <4 x half>, ptr addrspace(3) %address.local179, align 8, !dbg !9
  %v120 = fmul <4 x half> %v116, splat (half 1.275630e-01), !dbg !9
  %v121 = fmul <4 x half> %v117, splat (half 1.275630e-01), !dbg !9
  %v122 = fmul <4 x half> %v119, splat (half 1.275630e-01), !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #6, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local183 = and i32 %address.local163, 24, !dbg !12
  %address.local188 = shl nuw nsw i32 %address.thread, 7, !dbg !12
  %address.local189 = and i32 %address.local188, 384, !dbg !12
  %27 = or disjoint i32 %address.local189, %address.local183, !dbg !12
  %28 = extractelement <4 x i32> %25, i64 2, !dbg !12
  %29 = or disjoint i32 %27, %28, !dbg !12
  %30 = extractelement <4 x i32> %25, i64 3, !dbg !12
  %address.local190 = or disjoint i32 %29, %30, !dbg !12
  %address.local191 = zext nneg i32 %address.local190 to i64, !dbg !12
  %address.local192 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local191, !dbg !12
  %v123 = load <4 x half>, ptr addrspace(3) %address.local192, align 8, !dbg !12
  %address.local206 = getelementptr i8, ptr addrspace(3) %address.local192, i64 4096, !dbg !12
  %v126 = load <4 x half>, ptr addrspace(3) %address.local206, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v127 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v123, <4 x float> zeroinitializer) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !13
  %v129 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v126, <4 x float> %v127) #5, !dbg !12
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local135, !dbg !12
  %address.local209 = lshr i64 %address.local, 2, !dbg !12
  %address.local210 = mul nsw i64 %address.local209, -512, !dbg !12
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) %address.local208, i64 %address.local210, !dbg !12
  %address.local212 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local213 = and i64 %address.local212, 24, !dbg !12
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local213, !dbg !12
  %address.local216 = and i64 %address.local209, 12, !dbg !9
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local216, !dbg !12
  %31 = and i64 %address.local, 32, !dbg !9
  %.not = icmp eq i64 %31, 0, !dbg !9
  %address.local220 = select i1 %.not, i64 0, i64 1016, !dbg !9
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local220, !dbg !12
  %address.local222 = getelementptr i8, ptr addrspace(3) %address.local221, i64 64, !dbg !12
  %v145 = load <4 x half>, ptr addrspace(3) %address.local222, align 2, !dbg !12
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local208, i64 %address.local216, !dbg !12
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) %address.local224, i64 %address.local220, !dbg !12
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local210, !dbg !12
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local213, !dbg !12
  %address.local228 = getelementptr i8, ptr addrspace(3) %address.local227, i64 4160, !dbg !12
  %v146 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !12
  %v147 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v145, <4 x float> zeroinitializer) #5, !dbg !12
  %address.local234 = getelementptr i8, ptr addrspace(3) %address.local221, i64 128, !dbg !12
  %v148 = load <4 x half>, ptr addrspace(3) %address.local234, align 2, !dbg !12
  %v149 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v146, <4 x float> %v147) #5, !dbg !12
  %address.local240 = getelementptr i8, ptr addrspace(3) %address.local227, i64 4224, !dbg !12
  %v150 = load <4 x half>, ptr addrspace(3) %address.local240, align 2, !dbg !12
  %v151 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v148, <4 x float> zeroinitializer) #5, !dbg !12
  %v152 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v150, <4 x float> %v151) #5, !dbg !12
  %address.local246 = getelementptr i8, ptr addrspace(3) %address.local221, i64 192, !dbg !12
  %v153 = load <4 x half>, ptr addrspace(3) %address.local246, align 2, !dbg !12
  %address.local252 = getelementptr i8, ptr addrspace(3) %address.local227, i64 4288, !dbg !12
  %v154 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !12
  %v155 = shufflevector <4 x float> %v152, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v156 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v153, <4 x float> zeroinitializer) #5, !dbg !12
  %address.local254 = getelementptr i8, ptr addrspace(3) %address.local192, i64 1024, !dbg !12
  %v157 = load <4 x half>, ptr addrspace(3) %address.local254, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v158 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v154, <4 x float> %v156) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !13
  %v159 = shufflevector <4 x float> %v158, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %address.local256 = getelementptr i8, ptr addrspace(3) %address.local192, i64 5120, !dbg !12
  %v160 = load <4 x half>, ptr addrspace(3) %address.local256, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v161 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v157, <4 x float> zeroinitializer) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v162 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v160, <4 x float> %v161) #5, !dbg !12
  %address.local258 = getelementptr i8, ptr addrspace(3) %address.local208, i64 1024, !dbg !12
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) %address.local258, i64 %address.local210, !dbg !12
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %address.local259, i64 %address.local213, !dbg !12
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local216, !dbg !12
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local220, !dbg !12
  %address.local263 = getelementptr i8, ptr addrspace(3) %address.local262, i64 64, !dbg !12
  %v163 = load <4 x half>, ptr addrspace(3) %address.local263, align 2, !dbg !12
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local258, i64 %address.local216, !dbg !12
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) %address.local265, i64 %address.local220, !dbg !12
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local266, i64 %address.local210, !dbg !12
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local213, !dbg !12
  %address.local269 = getelementptr i8, ptr addrspace(3) %address.local268, i64 4160, !dbg !12
  %v164 = load <4 x half>, ptr addrspace(3) %address.local269, align 2, !dbg !12
  %v165 = shufflevector <4 x float> %v162, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v166 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v163, <4 x float> zeroinitializer) #5, !dbg !12
  %address.local276 = getelementptr i8, ptr addrspace(3) %address.local262, i64 128, !dbg !12
  %v167 = load <4 x half>, ptr addrspace(3) %address.local276, align 2, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v168 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v164, <4 x float> %v166) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !13
  %v169 = shufflevector <4 x float> %v168, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %address.local282 = getelementptr i8, ptr addrspace(3) %address.local268, i64 4224, !dbg !12
  %v170 = load <4 x half>, ptr addrspace(3) %address.local282, align 2, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v171 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v167, <4 x float> zeroinitializer) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v172 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v170, <4 x float> %v171) #5, !dbg !12
  %address.local289 = getelementptr i8, ptr addrspace(3) %address.local262, i64 192, !dbg !12
  %v173 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !12
  %address.local295 = getelementptr i8, ptr addrspace(3) %address.local268, i64 4288, !dbg !12
  %v174 = load <4 x half>, ptr addrspace(3) %address.local295, align 2, !dbg !12
  %v175 = shufflevector <4 x float> %v172, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v176 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v118, <4 x half> %v173, <4 x float> zeroinitializer) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v178 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v120, <4 x half> %v174, <4 x float> %v176) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !13
  %v179 = shufflevector <4 x float> %v178, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v181 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v123, <4 x float> zeroinitializer) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v182 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v126, <4 x float> %v181) #5, !dbg !12
  %v185 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v145, <4 x float> zeroinitializer) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v187 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v146, <4 x float> %v185) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v189 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v148, <4 x float> zeroinitializer) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v190 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v150, <4 x float> %v189) #5, !dbg !12
  %v193 = shufflevector <4 x float> %v190, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v194 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v153, <4 x float> zeroinitializer) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v196 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v154, <4 x float> %v194) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !13
  %v197 = shufflevector <4 x float> %v196, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v199 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v157, <4 x float> zeroinitializer) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v200 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v160, <4 x float> %v199) #5, !dbg !12
  %v203 = shufflevector <4 x float> %v200, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v204 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v163, <4 x float> zeroinitializer) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v206 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v164, <4 x float> %v204) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !13
  %v207 = shufflevector <4 x float> %v206, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v209 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v167, <4 x float> zeroinitializer) #5, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #7, !dbg !12
  %v210 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v170, <4 x float> %v209) #5, !dbg !12
  %v213 = shufflevector <4 x float> %v210, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v214 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v121, <4 x half> %v173, <4 x float> zeroinitializer) #5, !dbg !12
  %v215 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %v122, <4 x half> %v174, <4 x float> %v214) #5, !dbg !12
  %v216 = shufflevector <4 x float> %v215, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !13
  %v217 = shufflevector <4 x float> %v129, <4 x float> %v149, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v218 = shufflevector <32 x float> %v217, <32 x float> %v155, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v219 = shufflevector <32 x float> %v218, <32 x float> %v159, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v220 = shufflevector <32 x float> %v219, <32 x float> %v165, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v221 = shufflevector <32 x float> %v220, <32 x float> %v169, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v222 = shufflevector <32 x float> %v221, <32 x float> %v175, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v223 = shufflevector <32 x float> %v222, <32 x float> %v179, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !14
  %v225 = shufflevector <4 x float> %v182, <4 x float> %v187, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v226 = shufflevector <32 x float> %v225, <32 x float> %v193, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v227 = shufflevector <32 x float> %v226, <32 x float> %v197, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v228 = shufflevector <32 x float> %v227, <32 x float> %v203, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v229 = shufflevector <32 x float> %v228, <32 x float> %v207, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v230 = shufflevector <32 x float> %v229, <32 x float> %v213, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !14
  %v231 = shufflevector <32 x float> %v230, <32 x float> %v216, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !14
  store <32 x float> %v223, ptr addrspace(1) %v3, align 16
  %v3.repack3 = getelementptr inbounds nuw i8, ptr addrspace(1) %v3, i64 128
  store <32 x float> %v231, ptr addrspace(1) %v3.repack3, align 16
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

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.or.v4i32(<4 x i32>) #4

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind willreturn memory(none) }
attributes #6 = { convergent nounwind }
attributes #7 = { memory(inaccessiblemem: readwrite) }

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
