; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/nohint_unrolled_32k.bc'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/nohint_unrolled.mlir"

@shm_0 = addrspace(3) global [32768 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %10 = sext i32 %9 to i64, !dbg !7
  %11 = shl nsw i64 %8, 7, !dbg !8
  %.lobit = ashr i32 %7, 31, !dbg !9
  %12 = xor i32 %.lobit, %7, !dbg !9
  %13 = sdiv i32 %12, 32, !dbg !9
  %14 = xor i32 %13, %.lobit, !dbg !9
  %15 = sext i32 %14 to i64, !dbg !9
  %16 = add nsw i64 %15, %10, !dbg !9
  %17 = shl nsw i64 %15, 12, !dbg !9
  %18 = sub nsw i64 %11, %17, !dbg !9
  %19 = shl nsw i64 %16, 19, !dbg !9
  %20 = and i64 %19, 16252928, !dbg !9
  %21 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !10, !range !11
  %.idx = shl nsw i64 %18, 8, !dbg !9
  %22 = shl nsw i64 %10, 7, !dbg !12
  %23 = shl nsw i64 %6, 12, !dbg !12
  %24 = add nsw i64 %22, %23, !dbg !12
  %.not94 = icmp slt i32 %7, 0, !dbg !13
  br i1 %.not94, label %.._crit_edge_crit_edge, label %.lr.ph, !dbg !13

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %.pre108 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !13
  %.pre110 = add nuw nsw i32 %.pre108, 64, !dbg !13
  %.pre112 = xor i32 %.pre108, 16, !dbg !13
  %.pre114 = and i32 %.pre110, 192, !dbg !13
  %.pre116 = xor i32 %.pre108, 32, !dbg !13
  br label %._crit_edge, !dbg !13

.lr.ph:                                           ; preds = %4
  %25 = zext nneg i32 %21 to i64, !dbg !10
  %26 = ptrtoint ptr addrspace(1) %0 to i64
  %27 = insertelement <2 x i64> poison, i64 %26, i64 0
  %28 = bitcast <2 x i64> %27 to <4 x i32>
  %dma.buffer.rsrc935 = shufflevector <4 x i32> %28, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %29 = ptrtoint ptr addrspace(1) %1 to i64
  %30 = insertelement <2 x i64> poison, i64 %29, i64 0
  %31 = bitcast <2 x i64> %30 to <4 x i32>
  %dma.buffer.rsrc929 = shufflevector <4 x i32> %31, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %.idx13 = shl nuw nsw i64 %20, 1
  %32 = add nsw i64 %.idx13, %.idx
  %invariant.op = or disjoint i64 %32, 64, !dbg !13
  %invariant.op99 = or disjoint i64 %32, 128, !dbg !13
  %33 = ptrtoint ptr addrspace(1) %2 to i64
  %34 = insertelement <2 x i64> poison, i64 %33, i64 0
  %35 = bitcast <2 x i64> %34 to <4 x i32>
  %dma.buffer.rsrc887 = shufflevector <4 x i32> %35, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %36 = add nuw nsw i64 %11, %25
  %37 = lshr i64 %25, 2
  %38 = and i64 %37, 48
  %39 = and i32 %21, 240
  %index.compact = zext nneg i32 %39 to i64
  %40 = sub nsw i64 %36, %index.compact
  %41 = add nsw i64 %40, %38
  %42 = and i64 %25, 48
  %43 = add nsw i64 %41, 1
  %44 = lshr i64 %43, 32
  %compare.high = trunc nuw i64 %44 to i32
  %compare.low.base = trunc i64 %43 to i32
  %45 = add nsw i64 %41, 65
  %46 = lshr i64 %45, 32
  %compare.high736 = trunc nuw i64 %46 to i32
  %compare.low.base740 = trunc i64 %45 to i32
  %47 = insertelement <32 x i32> poison, i32 %compare.low.base740, i64 0, !dbg !14
  %48 = shufflevector <32 x i32> %47, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !14
  %49 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !14
  %50 = shufflevector <32 x i32> %49, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !14
  br label %51, !dbg !13

51:                                               ; preds = %.lr.ph, %494
  %52 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1148, %494 ]
  %53 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1144, %494 ]
  %54 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1140, %494 ]
  %55 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %837, %494 ]
  %56 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1136, %494 ]
  %57 = phi i64 [ 0, %.lr.ph ], [ %1149, %494 ]
  %58 = shl nuw nsw i64 %57, 7, !dbg !13
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %21) #8, !dbg !9, !range !11, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !10
  %address.local129 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local130 = and i64 %address.local129, 1536, !dbg !9
  %address.local131 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local130, !dbg !9
  %address.local134 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local135 = and i64 %address.local134, 16256, !dbg !9
  %address.local136 = lshr i64 %address.local, 4, !dbg !9
  %address.local137 = and i64 %address.local136, 8, !dbg !9
  %address.local138 = or disjoint i64 %address.local135, %address.local137, !dbg !9
  %59 = or disjoint i64 %address.local138, %20, !dbg !9
  %60 = shl nuw nsw i64 %59, 1, !dbg !9
  %61 = add nsw i64 %60, %.idx, !dbg !9
  %dma.buffer.offset934 = trunc i64 %61 to i32, !dbg !9
  %address.local140 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local141 = shl nuw nsw i64 %address.local140, 3, !dbg !9
  %address.local142 = and i64 %address.local141, 3584, !dbg !9
  %address.local143 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local142, !dbg !9
  %address.local148 = lshr i64 %address.local140, 4, !dbg !9
  %address.local149 = and i64 %address.local148, 24, !dbg !9
  %address.local150 = or disjoint i64 %address.local149, %address.local135, !dbg !9
  %62 = or disjoint i64 %address.local150, %20, !dbg !9
  %63 = shl nuw nsw i64 %62, 1, !dbg !9
  %64 = add nsw i64 %63, %.idx, !dbg !9
  %dma.buffer.offset931 = trunc i64 %64 to i32, !dbg !9
  %65 = lshr i64 %57, 12, !dbg !12
  %66 = add nsw i64 %65, %10, !dbg !12
  %67 = lshr i64 %57, 5, !dbg !12
  %68 = and i64 %67, 127, !dbg !12
  %69 = shl i64 %66, 19, !dbg !12
  %70 = and i64 %69, 16252928, !dbg !12
  %71 = shl i64 %67, 19, !dbg !12
  %72 = shl nuw nsw i64 %57, 14, !dbg !12
  %73 = sub i64 %72, %71, !dbg !12
  %address.local152 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local130, !dbg !12
  %.idx36 = or disjoint i64 %70, %68, !dbg !12
  %74 = add i64 %.idx36, %73, !dbg !12
  %75 = add i64 %address.local138, %74, !dbg !12
  %.tr = trunc i64 %75 to i32, !dbg !12
  %dma.buffer.offset928 = shl i32 %.tr, 1, !dbg !12
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local142, !dbg !12
  %76 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local131, i64 0, !dbg !9
  %77 = insertelement <4 x ptr addrspace(3)> %76, ptr addrspace(3) %address.local143, i64 1, !dbg !9
  %78 = insertelement <4 x ptr addrspace(3)> %77, ptr addrspace(3) %address.local152, i64 2, !dbg !9
  %79 = insertelement <4 x ptr addrspace(3)> %78, ptr addrspace(3) %address.local156, i64 3, !dbg !9
  %80 = ptrtoint <4 x ptr addrspace(3)> %79 to <4 x i64>, !dbg !9
  %81 = trunc <4 x i64> %80 to <4 x i32>, !dbg !9
  %82 = extractelement <4 x i32> %81, i64 0, !dbg !9
  %83 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %82), !dbg !9
  %dma.buffer.lds936 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83), !dbg !9
  %84 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds936, i32 %dma.buffer.offset934, <4 x i32> %dma.buffer.rsrc935) #9, !dbg !9
  %85 = extractelement <4 x i32> %81, i64 1, !dbg !9
  %86 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %85), !dbg !9
  %dma.buffer.lds933 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %86), !dbg !9
  %87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds933, i32 %dma.buffer.offset931, <4 x i32> %dma.buffer.rsrc935) #9, !dbg !9
  %88 = extractelement <4 x i32> %81, i64 2, !dbg !12
  %89 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %88), !dbg !12
  %dma.buffer.lds930 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %89), !dbg !12
  %90 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds930, i32 %dma.buffer.offset928, <4 x i32> %dma.buffer.rsrc929) #9, !dbg !12
  %91 = extractelement <4 x i32> %81, i64 3, !dbg !12
  %92 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %91), !dbg !12
  %93 = add i64 %address.local150, %74, !dbg !12
  %.tr11 = trunc i64 %93 to i32, !dbg !12
  %dma.buffer.offset925 = shl i32 %.tr11, 1, !dbg !12
  %dma.buffer.lds927 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %92), !dbg !12
  %94 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds927, i32 %dma.buffer.offset925, <4 x i32> %dma.buffer.rsrc929) #9, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local160 = lshr i32 %address.thread, 2, !dbg !10
  %address.local161 = and i32 %address.local160, 4, !dbg !10
  %address.local162 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local163 = and i32 %address.local162, 384, !dbg !10
  %address.local165 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local166 = and i32 %address.local165, 120, !dbg !10
  %address.local168 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local169 = and i32 %address.local168, 1024, !dbg !10
  %95 = or disjoint i32 %address.local166, %address.local163, !dbg !10
  %96 = or disjoint i32 %95, %address.local161, !dbg !10
  %address.local170 = or disjoint i32 %96, %address.local169, !dbg !10
  %address.local171 = zext nneg i32 %address.local170 to i64, !dbg !10
  %address.local172 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !10
  %97 = load <4 x half>, ptr addrspace(3) %address.local172, align 8, !dbg !10
  %address.local174 = getelementptr i8, ptr addrspace(3) %address.local172, i64 4096, !dbg !10
  %98 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !10
  %address.local176 = getelementptr i8, ptr addrspace(3) %address.local172, i64 1024, !dbg !10
  %99 = load <4 x half>, ptr addrspace(3) %address.local176, align 8, !dbg !10
  %100 = fmul <4 x half> %97, splat (half 1.275630e-01), !dbg !10
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local172, i64 5120, !dbg !10
  %101 = load <4 x half>, ptr addrspace(3) %address.local178, align 8, !dbg !10
  %102 = fmul <4 x half> %98, splat (half 1.275630e-01), !dbg !10
  %103 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !10
  %104 = fmul <4 x half> %101, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local182 = and i32 %address.local162, 24, !dbg !15
  %address.local187 = shl nuw nsw i32 %address.thread, 7, !dbg !15
  %address.local188 = and i32 %address.local187, 384, !dbg !15
  %105 = or disjoint i32 %address.local188, %address.local182, !dbg !15
  %106 = or disjoint i32 %105, %address.local161, !dbg !15
  %address.local189 = or disjoint i32 %106, %address.local169, !dbg !15
  %address.local190 = zext nneg i32 %address.local189 to i64, !dbg !15
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local190, !dbg !15
  %107 = load <4 x half>, ptr addrspace(3) %address.local191, align 8, !dbg !15
  %108 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %107, <4 x float> zeroinitializer) #8, !dbg !15
  %address.local205 = getelementptr i8, ptr addrspace(3) %address.local191, i64 4096, !dbg !15
  %109 = load <4 x half>, ptr addrspace(3) %address.local205, align 8, !dbg !15
  %address.local207 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local134, !dbg !15
  %address.local208 = lshr i64 %address.local, 2, !dbg !15
  %address.local209 = mul nsw i64 %address.local208, -512, !dbg !15
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i64 %address.local209, !dbg !15
  %address.local211 = shl nuw nsw i64 %address.local, 1, !dbg !15
  %address.local212 = and i64 %address.local211, 24, !dbg !15
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) %address.local210, i64 %address.local212, !dbg !15
  %address.local215 = and i64 %address.local208, 12, !dbg !10
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) %address.local213, i64 %address.local215, !dbg !15
  %110 = and i64 %address.local, 32, !dbg !10
  %.not12 = icmp eq i64 %110, 0, !dbg !10
  %address.local219 = select i1 %.not12, i64 0, i64 1016, !dbg !10
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local216, i64 %address.local219, !dbg !15
  %address.local221 = getelementptr i8, ptr addrspace(3) %address.local220, i64 64, !dbg !15
  %111 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !15
  %112 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %109, <4 x float> %108) #8, !dbg !15
  %113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %111, <4 x float> zeroinitializer) #8, !dbg !15
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i64 %address.local215, !dbg !15
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local219, !dbg !15
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) %address.local224, i64 %address.local209, !dbg !15
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local212, !dbg !15
  %address.local227 = getelementptr i8, ptr addrspace(3) %address.local226, i64 4160, !dbg !15
  %114 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !15
  %address.local233 = getelementptr i8, ptr addrspace(3) %address.local220, i64 128, !dbg !15
  %115 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !15
  %116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %114, <4 x float> %113) #8, !dbg !15
  %117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %115, <4 x float> zeroinitializer) #8, !dbg !15
  %address.local239 = getelementptr i8, ptr addrspace(3) %address.local226, i64 4224, !dbg !15
  %118 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !15
  %address.local245 = getelementptr i8, ptr addrspace(3) %address.local220, i64 192, !dbg !15
  %119 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !15
  %120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %118, <4 x float> %117) #8, !dbg !15
  %121 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %119, <4 x float> zeroinitializer) #8, !dbg !15
  %address.local251 = getelementptr i8, ptr addrspace(3) %address.local226, i64 4288, !dbg !15
  %122 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !15
  %address.local253 = getelementptr i8, ptr addrspace(3) %address.local191, i64 1024, !dbg !15
  %123 = load <4 x half>, ptr addrspace(3) %address.local253, align 8, !dbg !15
  %124 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %122, <4 x float> %121) #8, !dbg !15
  %125 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %123, <4 x float> zeroinitializer) #8, !dbg !15
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local191, i64 5120, !dbg !15
  %126 = load <4 x half>, ptr addrspace(3) %address.local255, align 8, !dbg !15
  %address.local257 = getelementptr i8, ptr addrspace(3) %address.local207, i64 1024, !dbg !15
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) %address.local257, i64 %address.local209, !dbg !15
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) %address.local258, i64 %address.local212, !dbg !15
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %address.local259, i64 %address.local215, !dbg !15
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local219, !dbg !15
  %address.local262 = getelementptr i8, ptr addrspace(3) %address.local261, i64 64, !dbg !15
  %127 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !15
  %128 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %126, <4 x float> %125) #8, !dbg !15
  %129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %127, <4 x float> zeroinitializer) #8, !dbg !15
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local257, i64 %address.local215, !dbg !15
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local219, !dbg !15
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) %address.local265, i64 %address.local209, !dbg !15
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local266, i64 %address.local212, !dbg !15
  %address.local268 = getelementptr i8, ptr addrspace(3) %address.local267, i64 4160, !dbg !15
  %130 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !15
  %address.local275 = getelementptr i8, ptr addrspace(3) %address.local261, i64 128, !dbg !15
  %131 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !15
  %132 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %130, <4 x float> %129) #8, !dbg !15
  %133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %131, <4 x float> zeroinitializer) #8, !dbg !15
  %address.local281 = getelementptr i8, ptr addrspace(3) %address.local267, i64 4224, !dbg !15
  %134 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !15
  %address.local288 = getelementptr i8, ptr addrspace(3) %address.local261, i64 192, !dbg !15
  %135 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !15
  %136 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %134, <4 x float> %133) #8, !dbg !15
  %137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %135, <4 x float> zeroinitializer) #8, !dbg !15
  %address.local294 = getelementptr i8, ptr addrspace(3) %address.local267, i64 4288, !dbg !15
  %138 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !15
  %139 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %138, <4 x float> %137) #8, !dbg !15
  %140 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %107, <4 x float> zeroinitializer) #8, !dbg !15
  %141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %109, <4 x float> %140) #8, !dbg !15
  %142 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %111, <4 x float> zeroinitializer) #8, !dbg !15
  %143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %114, <4 x float> %142) #8, !dbg !15
  %144 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %115, <4 x float> zeroinitializer) #8, !dbg !15
  %145 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %118, <4 x float> %144) #8, !dbg !15
  %146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %119, <4 x float> zeroinitializer) #8, !dbg !15
  %147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %122, <4 x float> %146) #8, !dbg !15
  %148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %123, <4 x float> zeroinitializer) #8, !dbg !15
  %149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %126, <4 x float> %148) #8, !dbg !15
  %150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %127, <4 x float> zeroinitializer) #8, !dbg !15
  %151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %130, <4 x float> %150) #8, !dbg !15
  %152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %131, <4 x float> zeroinitializer) #8, !dbg !15
  %153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %134, <4 x float> %152) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %135, <4 x float> zeroinitializer) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %138, <4 x float> %154) #8, !dbg !15
  %address.local295.idx = shl nuw nsw i64 %address.local138, 1, !dbg !9
  %156 = or disjoint i64 %address.local295.idx, %32, !dbg !9
  %157 = trunc i64 %156 to i32, !dbg !9
  %dma.buffer.offset922 = or disjoint i32 %157, 64, !dbg !9
  %158 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds936, i32 %dma.buffer.offset922, <4 x i32> %dma.buffer.rsrc935) #9, !dbg !9
  %address.local296.idx = shl nuw nsw i64 %address.local150, 1, !dbg !9
  %.reass = or disjoint i64 %address.local296.idx, %invariant.op
  %dma.buffer.offset919 = trunc i64 %.reass to i32, !dbg !9
  %159 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds933, i32 %dma.buffer.offset919, <4 x i32> %dma.buffer.rsrc935) #9, !dbg !9
  %160 = add nuw nsw i64 %57, 1024, !dbg !12
  %161 = lshr i64 %160, 12, !dbg !12
  %162 = add nsw i64 %161, %10, !dbg !12
  %163 = add nsw i64 %67, %24, !dbg !12
  %164 = shl i64 %162, 19, !dbg !12
  %165 = and i64 %164, 16252928, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %166 = add i64 %73, %67, !dbg !12
  %.idx1519 = add i64 %166, %24, !dbg !12
  %167 = add i64 %.idx1519, %165, !dbg !12
  %dma.buffer.offset910 = or disjoint i32 %157, 128, !dbg !9
  %.reass100 = or disjoint i64 %address.local296.idx, %invariant.op99
  %dma.buffer.offset907 = trunc i64 %.reass100 to i32, !dbg !9
  %168 = add nuw nsw i64 %57, 2048, !dbg !12
  %169 = lshr i64 %168, 12, !dbg !12
  %170 = add nsw i64 %169, %10, !dbg !12
  %171 = insertelement <2 x i64> poison, i64 %163, i64 0, !dbg !12
  %172 = shufflevector <2 x i64> %171, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !12
  %173 = icmp slt <2 x i64> %172, <i64 -32, i64 -64>, !dbg !12
  %174 = extractelement <2 x i1> %173, i64 0, !dbg !12
  %175 = sext i1 %174 to i64, !dbg !12
  %176 = add nsw <2 x i64> %172, <i64 32, i64 64>, !dbg !12
  %177 = sub nsw <2 x i64> <i64 -33, i64 -65>, %172, !dbg !12
  %178 = select <2 x i1> %173, <2 x i64> %177, <2 x i64> %176, !dbg !12
  %179 = sdiv <2 x i64> %178, splat (i64 128), !dbg !12
  %180 = extractelement <2 x i64> %179, i64 0, !dbg !12
  %181 = xor i64 %180, %175, !dbg !12
  %182 = shl nsw i64 %181, 7, !dbg !12
  %183 = sub i64 %167, %182, !dbg !12
  %184 = add i64 %address.local138, %183, !dbg !12
  %.tr20 = trunc i64 %184 to i32, !dbg !12
  %185 = shl i32 %.tr20, 1, !dbg !12
  %dma.buffer.offset916 = add i32 %185, 64, !dbg !12
  %186 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds930, i32 %dma.buffer.offset916, <4 x i32> %dma.buffer.rsrc929) #9, !dbg !12
  %187 = add i64 %address.local150, %183, !dbg !12
  %.tr26 = trunc i64 %187 to i32, !dbg !12
  %188 = shl i32 %.tr26, 1, !dbg !12
  %dma.buffer.offset913 = add i32 %188, 64, !dbg !12
  %189 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds927, i32 %dma.buffer.offset913, <4 x i32> %dma.buffer.rsrc929) #9, !dbg !12
  %190 = load <4 x half>, ptr addrspace(3) %address.local172, align 8, !dbg !10
  %191 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !10
  %192 = load <4 x half>, ptr addrspace(3) %address.local176, align 8, !dbg !10
  %193 = fmul <4 x half> %190, splat (half 1.275630e-01), !dbg !10
  %194 = load <4 x half>, ptr addrspace(3) %address.local178, align 8, !dbg !10
  %195 = fmul <4 x half> %191, splat (half 1.275630e-01), !dbg !10
  %196 = fmul <4 x half> %192, splat (half 1.275630e-01), !dbg !10
  %197 = fmul <4 x half> %194, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %198 = load <4 x half>, ptr addrspace(3) %address.local191, align 8, !dbg !15
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %198, <4 x float> %112) #8, !dbg !15
  %200 = load <4 x half>, ptr addrspace(3) %address.local205, align 8, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !15
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %195, <4 x half> %200, <4 x float> %199) #8, !dbg !15
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %201, <4 x float> %116) #8, !dbg !15
  %204 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !15
  %205 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !15
  %206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %195, <4 x half> %204, <4 x float> %203) #8, !dbg !15
  %207 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %205, <4 x float> %120) #8, !dbg !15
  %208 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !15
  %210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %195, <4 x half> %208, <4 x float> %207) #8, !dbg !15
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %209, <4 x float> %124) #8, !dbg !15
  %212 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !15
  %213 = load <4 x half>, ptr addrspace(3) %address.local253, align 8, !dbg !15
  %214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %195, <4 x half> %212, <4 x float> %211) #8, !dbg !15
  %215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %213, <4 x float> %128) #8, !dbg !15
  %216 = load <4 x half>, ptr addrspace(3) %address.local255, align 8, !dbg !15
  %217 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !15
  %218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %195, <4 x half> %216, <4 x float> %215) #8, !dbg !15
  %219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %217, <4 x float> %132) #8, !dbg !15
  %220 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !15
  %221 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !15
  %222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %195, <4 x half> %220, <4 x float> %219) #8, !dbg !15
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %221, <4 x float> %136) #8, !dbg !15
  %224 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !15
  %225 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !15
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %195, <4 x half> %224, <4 x float> %223) #8, !dbg !15
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %225, <4 x float> %139) #8, !dbg !15
  %228 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !15
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %195, <4 x half> %228, <4 x float> %227) #8, !dbg !15
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %198, <4 x float> %141) #8, !dbg !15
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %197, <4 x half> %200, <4 x float> %230) #8, !dbg !15
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %201, <4 x float> %143) #8, !dbg !15
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %197, <4 x half> %204, <4 x float> %232) #8, !dbg !15
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %205, <4 x float> %145) #8, !dbg !15
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %197, <4 x half> %208, <4 x float> %234) #8, !dbg !15
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %209, <4 x float> %147) #8, !dbg !15
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %197, <4 x half> %212, <4 x float> %236) #8, !dbg !15
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %213, <4 x float> %149) #8, !dbg !15
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %197, <4 x half> %216, <4 x float> %238) #8, !dbg !15
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %217, <4 x float> %151) #8, !dbg !15
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %197, <4 x half> %220, <4 x float> %240) #8, !dbg !15
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %221, <4 x float> %153) #8, !dbg !15
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %197, <4 x half> %224, <4 x float> %242) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %225, <4 x float> %155) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %197, <4 x half> %228, <4 x float> %244) #8, !dbg !15
  %246 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds936, i32 %dma.buffer.offset910, <4 x i32> %dma.buffer.rsrc935) #9, !dbg !9
  %247 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds933, i32 %dma.buffer.offset907, <4 x i32> %dma.buffer.rsrc935) #9, !dbg !9
  %248 = extractelement <2 x i1> %173, i64 1, !dbg !12
  %249 = sext i1 %248 to i64, !dbg !12
  %250 = extractelement <2 x i64> %179, i64 1, !dbg !12
  %251 = xor i64 %250, %249, !dbg !12
  %252 = shl i64 %170, 19, !dbg !12
  %253 = and i64 %252, 16252928, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %254 = add i64 %.idx1519, %253, !dbg !12
  %255 = shl nsw i64 %251, 7, !dbg !12
  %256 = sub i64 %254, %255, !dbg !12
  %257 = add i64 %address.local138, %256, !dbg !12
  %.tr34 = trunc i64 %257 to i32, !dbg !12
  %258 = shl i32 %.tr34, 1, !dbg !12
  %dma.buffer.offset904 = add i32 %258, 128, !dbg !12
  %259 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds930, i32 %dma.buffer.offset904, <4 x i32> %dma.buffer.rsrc929) #9, !dbg !12
  %260 = add i64 %address.local150, %256, !dbg !12
  %.tr41 = trunc i64 %260 to i32, !dbg !12
  %261 = shl i32 %.tr41, 1, !dbg !12
  %dma.buffer.offset901 = add i32 %261, 128, !dbg !12
  %262 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds927, i32 %dma.buffer.offset901, <4 x i32> %dma.buffer.rsrc929) #9, !dbg !12
  %263 = load <4 x half>, ptr addrspace(3) %address.local172, align 8, !dbg !10
  %264 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !10
  %265 = load <4 x half>, ptr addrspace(3) %address.local176, align 8, !dbg !10
  %266 = fmul <4 x half> %263, splat (half 1.275630e-01), !dbg !10
  %267 = load <4 x half>, ptr addrspace(3) %address.local178, align 8, !dbg !10
  %268 = fmul <4 x half> %264, splat (half 1.275630e-01), !dbg !10
  %269 = fmul <4 x half> %265, splat (half 1.275630e-01), !dbg !10
  %270 = fmul <4 x half> %267, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %271 = load <4 x half>, ptr addrspace(3) %address.local191, align 8, !dbg !15
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %266, <4 x half> %271, <4 x float> %202) #8, !dbg !15
  %273 = load <4 x half>, ptr addrspace(3) %address.local205, align 8, !dbg !15
  %274 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !15
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %268, <4 x half> %273, <4 x float> %272) #8, !dbg !15
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %266, <4 x half> %274, <4 x float> %206) #8, !dbg !15
  %277 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !15
  %278 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !15
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %268, <4 x half> %277, <4 x float> %276) #8, !dbg !15
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %266, <4 x half> %278, <4 x float> %210) #8, !dbg !15
  %281 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !15
  %282 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !15
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %268, <4 x half> %281, <4 x float> %280) #8, !dbg !15
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %266, <4 x half> %282, <4 x float> %214) #8, !dbg !15
  %285 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !15
  %286 = load <4 x half>, ptr addrspace(3) %address.local253, align 8, !dbg !15
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %268, <4 x half> %285, <4 x float> %284) #8, !dbg !15
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %266, <4 x half> %286, <4 x float> %218) #8, !dbg !15
  %289 = load <4 x half>, ptr addrspace(3) %address.local255, align 8, !dbg !15
  %290 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !15
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %268, <4 x half> %289, <4 x float> %288) #8, !dbg !15
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %266, <4 x half> %290, <4 x float> %222) #8, !dbg !15
  %293 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !15
  %294 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !15
  %295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %268, <4 x half> %293, <4 x float> %292) #8, !dbg !15
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %266, <4 x half> %294, <4 x float> %226) #8, !dbg !15
  %297 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !15
  %298 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !15
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %268, <4 x half> %297, <4 x float> %296) #8, !dbg !15
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %266, <4 x half> %298, <4 x float> %229) #8, !dbg !15
  %301 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !15
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %268, <4 x half> %301, <4 x float> %300) #8, !dbg !15
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %269, <4 x half> %271, <4 x float> %231) #8, !dbg !15
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %270, <4 x half> %273, <4 x float> %303) #8, !dbg !15
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %269, <4 x half> %274, <4 x float> %233) #8, !dbg !15
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %270, <4 x half> %277, <4 x float> %305) #8, !dbg !15
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %269, <4 x half> %278, <4 x float> %235) #8, !dbg !15
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %270, <4 x half> %281, <4 x float> %307) #8, !dbg !15
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %269, <4 x half> %282, <4 x float> %237) #8, !dbg !15
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %270, <4 x half> %285, <4 x float> %309) #8, !dbg !15
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %269, <4 x half> %286, <4 x float> %239) #8, !dbg !15
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %270, <4 x half> %289, <4 x float> %311) #8, !dbg !15
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %269, <4 x half> %290, <4 x float> %241) #8, !dbg !15
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %270, <4 x half> %293, <4 x float> %313) #8, !dbg !15
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %269, <4 x half> %294, <4 x float> %243) #8, !dbg !15
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %270, <4 x half> %297, <4 x float> %315) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %269, <4 x half> %298, <4 x float> %245) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %270, <4 x half> %301, <4 x float> %317) #8, !dbg !15
  %dma.buffer.offset898 = or disjoint i32 %157, 192, !dbg !9
  %319 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds936, i32 %dma.buffer.offset898, <4 x i32> %dma.buffer.rsrc935) #9, !dbg !9
  %320 = or disjoint i64 %address.local296.idx, %32, !dbg !9
  %321 = trunc i64 %320 to i32, !dbg !9
  %dma.buffer.offset895 = or disjoint i32 %321, 192, !dbg !9
  %322 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds933, i32 %dma.buffer.offset895, <4 x i32> %dma.buffer.rsrc935) #9, !dbg !9
  %323 = add nuw nsw i64 %57, 3072, !dbg !12
  %324 = lshr i64 %323, 12, !dbg !12
  %325 = add nsw i64 %324, %10, !dbg !12
  %326 = add nsw i64 %163, 96, !dbg !12
  %327 = icmp slt i64 %163, -96, !dbg !12
  %328 = sub nsw i64 -97, %163, !dbg !12
  %329 = select i1 %327, i64 %328, i64 %326, !dbg !12
  %330 = sdiv i64 %329, 128, !dbg !12
  %331 = sext i1 %327 to i64, !dbg !12
  %332 = xor i64 %330, %331, !dbg !12
  %333 = shl i64 %325, 19, !dbg !12
  %334 = and i64 %333, 16252928, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %335 = add i64 %.idx1519, %334, !dbg !12
  %336 = shl nsw i64 %332, 7, !dbg !12
  %337 = sub i64 %335, %336, !dbg !12
  %338 = add i64 %address.local138, %337, !dbg !12
  %.tr49 = trunc i64 %338 to i32, !dbg !12
  %339 = shl i32 %.tr49, 1, !dbg !12
  %dma.buffer.offset892 = add i32 %339, 192, !dbg !12
  %340 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds930, i32 %dma.buffer.offset892, <4 x i32> %dma.buffer.rsrc929) #9, !dbg !12
  %341 = add i64 %address.local150, %337, !dbg !12
  %.tr55 = trunc i64 %341 to i32, !dbg !12
  %342 = shl i32 %.tr55, 1, !dbg !12
  %dma.buffer.offset889 = add i32 %342, 192, !dbg !12
  %343 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds927, i32 %dma.buffer.offset889, <4 x i32> %dma.buffer.rsrc929) #9, !dbg !12
  %344 = load <4 x half>, ptr addrspace(3) %address.local172, align 8, !dbg !10
  %345 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !10
  %346 = load <4 x half>, ptr addrspace(3) %address.local176, align 8, !dbg !10
  %347 = fmul <4 x half> %344, splat (half 1.275630e-01), !dbg !10
  %348 = load <4 x half>, ptr addrspace(3) %address.local178, align 8, !dbg !10
  %349 = fmul <4 x half> %345, splat (half 1.275630e-01), !dbg !10
  %350 = fmul <4 x half> %346, splat (half 1.275630e-01), !dbg !10
  %351 = fmul <4 x half> %348, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %352 = add nsw i64 %67, %10, !dbg !16
  %353 = shl i64 %352, 19, !dbg !16
  %354 = and i64 %353, 16252928, !dbg !16
  %address.local307 = shl nuw nsw i64 %address.local, 6, !dbg !16
  %address.local308 = and i64 %address.local307, 16128, !dbg !16
  %address.local310 = and i64 %address.local129, 24, !dbg !16
  %.idx5659 = add i64 %354, %73, !dbg !16
  %355 = or disjoint i64 %address.local310, %.idx5659, !dbg !16
  %356 = shl i64 %355, 1, !dbg !16
  %357 = or disjoint i64 %356, %address.local308, !dbg !16
  %dma.buffer.offset886 = trunc i64 %357 to i32, !dbg !16
  %358 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds936, i32 %dma.buffer.offset886, <4 x i32> %dma.buffer.rsrc887) #9, !dbg !16
  %address.local313 = shl nuw nsw i64 %address.local140, 6, !dbg !16
  %address.local314 = and i64 %address.local313, 32512, !dbg !16
  %359 = or disjoint i64 %356, %address.local314, !dbg !16
  %dma.buffer.offset883 = trunc i64 %359 to i32, !dbg !16
  %360 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds933, i32 %dma.buffer.offset883, <4 x i32> %dma.buffer.rsrc887) #9, !dbg !16
  %361 = load <4 x half>, ptr addrspace(3) %address.local191, align 8, !dbg !15
  %362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %361, <4 x float> %275) #8, !dbg !15
  %363 = load <4 x half>, ptr addrspace(3) %address.local205, align 8, !dbg !15
  %364 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !15
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %363, <4 x float> %362) #8, !dbg !15
  %366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %364, <4 x float> %279) #8, !dbg !15
  %367 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !15
  %368 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !15
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %367, <4 x float> %366) #8, !dbg !15
  %370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %368, <4 x float> %283) #8, !dbg !15
  %371 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !15
  %372 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !15
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %371, <4 x float> %370) #8, !dbg !15
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %372, <4 x float> %287) #8, !dbg !15
  %375 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !15
  %376 = load <4 x half>, ptr addrspace(3) %address.local253, align 8, !dbg !15
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %375, <4 x float> %374) #8, !dbg !15
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %376, <4 x float> %291) #8, !dbg !15
  %379 = load <4 x half>, ptr addrspace(3) %address.local255, align 8, !dbg !15
  %380 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !15
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %379, <4 x float> %378) #8, !dbg !15
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %380, <4 x float> %295) #8, !dbg !15
  %383 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !15
  %384 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !15
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %383, <4 x float> %382) #8, !dbg !15
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %384, <4 x float> %299) #8, !dbg !15
  %387 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !15
  %388 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !15
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %387, <4 x float> %386) #8, !dbg !15
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %347, <4 x half> %388, <4 x float> %302) #8, !dbg !15
  %391 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !15
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %349, <4 x half> %391, <4 x float> %390) #8, !dbg !15
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %361, <4 x float> %304) #8, !dbg !15
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %351, <4 x half> %363, <4 x float> %393) #8, !dbg !15
  %395 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %364, <4 x float> %306) #8, !dbg !15
  %396 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %351, <4 x half> %367, <4 x float> %395) #8, !dbg !15
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %368, <4 x float> %308) #8, !dbg !15
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %351, <4 x half> %371, <4 x float> %397) #8, !dbg !15
  %399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %372, <4 x float> %310) #8, !dbg !15
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %351, <4 x half> %375, <4 x float> %399) #8, !dbg !15
  %401 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %376, <4 x float> %312) #8, !dbg !15
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %351, <4 x half> %379, <4 x float> %401) #8, !dbg !15
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %380, <4 x float> %314) #8, !dbg !15
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %351, <4 x half> %383, <4 x float> %403) #8, !dbg !15
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %384, <4 x float> %316) #8, !dbg !15
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %351, <4 x half> %387, <4 x float> %405) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15
  %407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %350, <4 x half> %388, <4 x float> %318) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %408 = or disjoint i64 %58, 127, !dbg !17
  %.not102 = icmp ult i64 %11, %408, !dbg !17
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %351, <4 x half> %391, <4 x float> %407) #8, !dbg !15
  br i1 %.not102, label %410, label %494, !dbg !17

410:                                              ; preds = %51
  %411 = or disjoint i64 %58, %42, !dbg !17
  %412 = lshr i64 %57, 25, !dbg !14
  %compare.high612 = trunc nuw i64 %412 to i32, !dbg !14
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high612, !dbg !14
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high612, !dbg !14
  %compare.low.base613 = trunc i64 %411 to i32, !dbg !14
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 1) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base616 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 2) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base620 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 3) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base624 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 4) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base628 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 5) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base632 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 6) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base636 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 7) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base640 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 8) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base644 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 9) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base648 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 10) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base652 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 11) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base656 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 12) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base660 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 13) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base664 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 14) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base668 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 15) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base672 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 16) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base676 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 17) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base680 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 18) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base684 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 19) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base688 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 20) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base692 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 21) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base696 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 22) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base700 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 23) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base704 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 24) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base708 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 25) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base712 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 26) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base716 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 27) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base720 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 28) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base724 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 29) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base728 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 30) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base732 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 31) #8, !dbg !14, !frisk.register_lifetimes !5
  %413 = insertelement <32 x i32> poison, i32 %compare.low.base613, i64 0, !dbg !14
  %414 = insertelement <32 x i32> %413, i32 %compare.local.base, i64 1, !dbg !14
  %415 = insertelement <32 x i32> %414, i32 %compare.local.base616, i64 2, !dbg !14
  %416 = insertelement <32 x i32> %415, i32 %compare.local.base620, i64 3, !dbg !14
  %417 = insertelement <32 x i32> %416, i32 %compare.local.base624, i64 4, !dbg !14
  %418 = insertelement <32 x i32> %417, i32 %compare.local.base628, i64 5, !dbg !14
  %419 = insertelement <32 x i32> %418, i32 %compare.local.base632, i64 6, !dbg !14
  %420 = insertelement <32 x i32> %419, i32 %compare.local.base636, i64 7, !dbg !14
  %421 = insertelement <32 x i32> %420, i32 %compare.local.base640, i64 8, !dbg !14
  %422 = insertelement <32 x i32> %421, i32 %compare.local.base644, i64 9, !dbg !14
  %423 = insertelement <32 x i32> %422, i32 %compare.local.base648, i64 10, !dbg !14
  %424 = insertelement <32 x i32> %423, i32 %compare.local.base652, i64 11, !dbg !14
  %425 = insertelement <32 x i32> %424, i32 %compare.local.base656, i64 12, !dbg !14
  %426 = insertelement <32 x i32> %425, i32 %compare.local.base660, i64 13, !dbg !14
  %427 = insertelement <32 x i32> %426, i32 %compare.local.base664, i64 14, !dbg !14
  %428 = insertelement <32 x i32> %427, i32 %compare.local.base668, i64 15, !dbg !14
  %429 = insertelement <32 x i32> %428, i32 %compare.local.base672, i64 16, !dbg !14
  %430 = insertelement <32 x i32> %429, i32 %compare.local.base676, i64 17, !dbg !14
  %431 = insertelement <32 x i32> %430, i32 %compare.local.base680, i64 18, !dbg !14
  %432 = insertelement <32 x i32> %431, i32 %compare.local.base684, i64 19, !dbg !14
  %433 = insertelement <32 x i32> %432, i32 %compare.local.base688, i64 20, !dbg !14
  %434 = insertelement <32 x i32> %433, i32 %compare.local.base692, i64 21, !dbg !14
  %435 = insertelement <32 x i32> %434, i32 %compare.local.base696, i64 22, !dbg !14
  %436 = insertelement <32 x i32> %435, i32 %compare.local.base700, i64 23, !dbg !14
  %437 = insertelement <32 x i32> %436, i32 %compare.local.base704, i64 24, !dbg !14
  %438 = insertelement <32 x i32> %437, i32 %compare.local.base708, i64 25, !dbg !14
  %439 = insertelement <32 x i32> %438, i32 %compare.local.base712, i64 26, !dbg !14
  %440 = insertelement <32 x i32> %439, i32 %compare.local.base716, i64 27, !dbg !14
  %441 = insertelement <32 x i32> %440, i32 %compare.local.base720, i64 28, !dbg !14
  %442 = insertelement <32 x i32> %441, i32 %compare.local.base724, i64 29, !dbg !14
  %443 = insertelement <32 x i32> %442, i32 %compare.local.base728, i64 30, !dbg !14
  %444 = insertelement <32 x i32> %443, i32 %compare.local.base732, i64 31, !dbg !14
  %445 = or <32 x i32> %444, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !14
  %446 = icmp ugt <32 x i32> %50, %445, !dbg !14
  %447 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !14
  %448 = shufflevector <32 x i1> %447, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !14
  %449 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !14
  %450 = shufflevector <32 x i1> %449, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !14
  %451 = select <32 x i1> %448, <32 x i1> %446, <32 x i1> %450, !dbg !14
  %452 = select <32 x i1> %451, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %compare.high.equal738 = icmp eq i32 %compare.high736, %compare.high612, !dbg !14
  %compare.high.order739 = icmp ugt i32 %compare.high736, %compare.high612, !dbg !14
  %compare.local.base744 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 32) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base748 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 33) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base752 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 34) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base756 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 35) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base760 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 36) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base764 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 37) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base768 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 38) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base772 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 39) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base776 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 40) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base780 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 41) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base784 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 42) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base788 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 43) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base792 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 44) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base796 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 45) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base800 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 46) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base804 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 47) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base808 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 48) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base812 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 49) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base816 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 50) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base820 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 51) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base824 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 52) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base828 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 53) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base832 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 54) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base836 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 55) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base840 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 56) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base844 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 57) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base848 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 58) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base852 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 59) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base856 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 60) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base860 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 61) #8, !dbg !14, !frisk.register_lifetimes !5
  %compare.local.base864 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 62) #8, !dbg !14, !frisk.register_lifetimes !5
  %453 = insertelement <32 x i32> %413, i32 %compare.local.base744, i64 1, !dbg !14
  %454 = insertelement <32 x i32> %453, i32 %compare.local.base748, i64 2, !dbg !14
  %455 = insertelement <32 x i32> %454, i32 %compare.local.base752, i64 3, !dbg !14
  %456 = insertelement <32 x i32> %455, i32 %compare.local.base756, i64 4, !dbg !14
  %457 = insertelement <32 x i32> %456, i32 %compare.local.base760, i64 5, !dbg !14
  %458 = insertelement <32 x i32> %457, i32 %compare.local.base764, i64 6, !dbg !14
  %459 = insertelement <32 x i32> %458, i32 %compare.local.base768, i64 7, !dbg !14
  %460 = insertelement <32 x i32> %459, i32 %compare.local.base772, i64 8, !dbg !14
  %461 = insertelement <32 x i32> %460, i32 %compare.local.base776, i64 9, !dbg !14
  %462 = insertelement <32 x i32> %461, i32 %compare.local.base780, i64 10, !dbg !14
  %463 = insertelement <32 x i32> %462, i32 %compare.local.base784, i64 11, !dbg !14
  %464 = insertelement <32 x i32> %463, i32 %compare.local.base788, i64 12, !dbg !14
  %465 = insertelement <32 x i32> %464, i32 %compare.local.base792, i64 13, !dbg !14
  %466 = insertelement <32 x i32> %465, i32 %compare.local.base796, i64 14, !dbg !14
  %467 = insertelement <32 x i32> %466, i32 %compare.local.base800, i64 15, !dbg !14
  %468 = insertelement <32 x i32> %467, i32 %compare.local.base804, i64 16, !dbg !14
  %469 = insertelement <32 x i32> %468, i32 %compare.local.base808, i64 17, !dbg !14
  %470 = insertelement <32 x i32> %469, i32 %compare.local.base812, i64 18, !dbg !14
  %471 = insertelement <32 x i32> %470, i32 %compare.local.base816, i64 19, !dbg !14
  %472 = insertelement <32 x i32> %471, i32 %compare.local.base820, i64 20, !dbg !14
  %473 = insertelement <32 x i32> %472, i32 %compare.local.base824, i64 21, !dbg !14
  %474 = insertelement <32 x i32> %473, i32 %compare.local.base828, i64 22, !dbg !14
  %475 = insertelement <32 x i32> %474, i32 %compare.local.base832, i64 23, !dbg !14
  %476 = insertelement <32 x i32> %475, i32 %compare.local.base836, i64 24, !dbg !14
  %477 = insertelement <32 x i32> %476, i32 %compare.local.base840, i64 25, !dbg !14
  %478 = insertelement <32 x i32> %477, i32 %compare.local.base844, i64 26, !dbg !14
  %479 = insertelement <32 x i32> %478, i32 %compare.local.base848, i64 27, !dbg !14
  %480 = insertelement <32 x i32> %479, i32 %compare.local.base852, i64 28, !dbg !14
  %481 = insertelement <32 x i32> %480, i32 %compare.local.base856, i64 29, !dbg !14
  %482 = insertelement <32 x i32> %481, i32 %compare.local.base860, i64 30, !dbg !14
  %483 = insertelement <32 x i32> %482, i32 %compare.local.base864, i64 31, !dbg !14
  %484 = or <32 x i32> %483, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !14
  %485 = icmp ugt <32 x i32> %48, %484, !dbg !14
  %486 = insertelement <32 x i1> poison, i1 %compare.high.equal738, i64 0, !dbg !14
  %487 = shufflevector <32 x i1> %486, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !14
  %488 = insertelement <32 x i1> poison, i1 %compare.high.order739, i64 0, !dbg !14
  %489 = shufflevector <32 x i1> %488, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !14
  %490 = select <32 x i1> %487, <32 x i1> %485, <32 x i1> %489, !dbg !14
  %491 = select <32 x i1> %490, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !18
  %492 = insertvalue [2 x <32 x float>] poison, <32 x float> %452, 0, !dbg !17
  %493 = insertvalue [2 x <32 x float>] %492, <32 x float> %491, 1, !dbg !17
  br label %494, !dbg !17

494:                                              ; preds = %410, %51
  %495 = phi [2 x <32 x float>] [ %493, %410 ], [ zeroinitializer, %51 ], !dbg !17
  %496 = extractvalue [2 x <32 x float>] %495, 0, !dbg !19
  %497 = shufflevector <32 x float> %496, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %498 = fadd <4 x float> %365, %497, !dbg !19
  %499 = extractelement <4 x float> %498, i64 0, !dbg !20
  %500 = tail call float @llvm.amdgcn.exp2.f32(float %499), !dbg !20
  %501 = insertelement <4 x float> poison, float %500, i64 0, !dbg !21
  %502 = extractelement <4 x float> %498, i64 1, !dbg !20
  %503 = tail call float @llvm.amdgcn.exp2.f32(float %502), !dbg !20
  %504 = insertelement <4 x float> %501, float %503, i64 1, !dbg !21
  %505 = fadd float %500, %503, !dbg !22
  %506 = extractelement <4 x float> %498, i64 2, !dbg !20
  %507 = tail call float @llvm.amdgcn.exp2.f32(float %506), !dbg !20
  %508 = insertelement <4 x float> %504, float %507, i64 2, !dbg !21
  %509 = fadd float %507, %505, !dbg !22
  %510 = extractelement <4 x float> %498, i64 3, !dbg !20
  %511 = tail call float @llvm.amdgcn.exp2.f32(float %510), !dbg !20
  %512 = insertelement <4 x float> %508, float %511, i64 3, !dbg !21
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %512) #8, !dbg !21, !frisk.register_lifetimes !5
  %513 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !21
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %513) #8, !dbg !22, !frisk.register_lifetimes !5
  %514 = fadd float %511, %509, !dbg !22
  %515 = shufflevector <32 x float> %496, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %516 = fadd <4 x float> %369, %515, !dbg !19
  %517 = extractelement <4 x float> %516, i64 0, !dbg !20
  %518 = tail call float @llvm.amdgcn.exp2.f32(float %517), !dbg !20
  %519 = insertelement <4 x float> poison, float %518, i64 0, !dbg !21
  %520 = fadd float %518, %514, !dbg !22
  %521 = extractelement <4 x float> %516, i64 1, !dbg !20
  %522 = tail call float @llvm.amdgcn.exp2.f32(float %521), !dbg !20
  %523 = insertelement <4 x float> %519, float %522, i64 1, !dbg !21
  %524 = fadd float %522, %520, !dbg !22
  %525 = extractelement <4 x float> %516, i64 2, !dbg !20
  %526 = tail call float @llvm.amdgcn.exp2.f32(float %525), !dbg !20
  %527 = insertelement <4 x float> %523, float %526, i64 2, !dbg !21
  %528 = fadd float %526, %524, !dbg !22
  %529 = extractelement <4 x float> %516, i64 3, !dbg !20
  %530 = tail call float @llvm.amdgcn.exp2.f32(float %529), !dbg !20
  %531 = insertelement <4 x float> %527, float %530, i64 3, !dbg !21
  %math.result99 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %531) #8, !dbg !21, !frisk.register_lifetimes !5
  %532 = fptrunc <4 x float> %math.result99 to <4 x half>, !dbg !21
  %math.packed100 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %532) #8, !dbg !22, !frisk.register_lifetimes !5
  %533 = fadd float %530, %528, !dbg !22
  %534 = shufflevector <32 x float> %496, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %535 = fadd <4 x float> %373, %534, !dbg !19
  %536 = extractelement <4 x float> %535, i64 0, !dbg !20
  %537 = tail call float @llvm.amdgcn.exp2.f32(float %536), !dbg !20
  %538 = insertelement <4 x float> poison, float %537, i64 0, !dbg !21
  %539 = fadd float %537, %533, !dbg !22
  %540 = extractelement <4 x float> %535, i64 1, !dbg !20
  %541 = tail call float @llvm.amdgcn.exp2.f32(float %540), !dbg !20
  %542 = insertelement <4 x float> %538, float %541, i64 1, !dbg !21
  %543 = fadd float %541, %539, !dbg !22
  %544 = extractelement <4 x float> %535, i64 2, !dbg !20
  %545 = tail call float @llvm.amdgcn.exp2.f32(float %544), !dbg !20
  %546 = insertelement <4 x float> %542, float %545, i64 2, !dbg !21
  %547 = fadd float %545, %543, !dbg !22
  %548 = extractelement <4 x float> %535, i64 3, !dbg !20
  %549 = tail call float @llvm.amdgcn.exp2.f32(float %548), !dbg !20
  %550 = insertelement <4 x float> %546, float %549, i64 3, !dbg !21
  %math.result101 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %550) #8, !dbg !21, !frisk.register_lifetimes !5
  %551 = fptrunc <4 x float> %math.result101 to <4 x half>, !dbg !21
  %math.packed102 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %551) #8, !dbg !22, !frisk.register_lifetimes !5
  %552 = fadd float %549, %547, !dbg !22
  %553 = shufflevector <32 x float> %496, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %554 = fadd <4 x float> %377, %553, !dbg !19
  %555 = extractelement <4 x float> %554, i64 0, !dbg !20
  %556 = tail call float @llvm.amdgcn.exp2.f32(float %555), !dbg !20
  %557 = insertelement <4 x float> poison, float %556, i64 0, !dbg !21
  %558 = fadd float %556, %552, !dbg !22
  %559 = extractelement <4 x float> %554, i64 1, !dbg !20
  %560 = tail call float @llvm.amdgcn.exp2.f32(float %559), !dbg !20
  %561 = insertelement <4 x float> %557, float %560, i64 1, !dbg !21
  %562 = fadd float %560, %558, !dbg !22
  %563 = extractelement <4 x float> %554, i64 2, !dbg !20
  %564 = tail call float @llvm.amdgcn.exp2.f32(float %563), !dbg !20
  %565 = insertelement <4 x float> %561, float %564, i64 2, !dbg !21
  %566 = fadd float %564, %562, !dbg !22
  %567 = extractelement <4 x float> %554, i64 3, !dbg !20
  %568 = tail call float @llvm.amdgcn.exp2.f32(float %567), !dbg !20
  %569 = insertelement <4 x float> %565, float %568, i64 3, !dbg !21
  %math.result103 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %569) #8, !dbg !21, !frisk.register_lifetimes !5
  %570 = fptrunc <4 x float> %math.result103 to <4 x half>, !dbg !21
  %math.packed104 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %570) #8, !dbg !22, !frisk.register_lifetimes !5
  %571 = fadd float %568, %566, !dbg !22
  %572 = shufflevector <32 x float> %496, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %573 = fadd <4 x float> %381, %572, !dbg !19
  %574 = extractelement <4 x float> %573, i64 0, !dbg !20
  %575 = tail call float @llvm.amdgcn.exp2.f32(float %574), !dbg !20
  %576 = insertelement <4 x float> poison, float %575, i64 0, !dbg !21
  %577 = fadd float %575, %571, !dbg !22
  %578 = extractelement <4 x float> %573, i64 1, !dbg !20
  %579 = tail call float @llvm.amdgcn.exp2.f32(float %578), !dbg !20
  %580 = insertelement <4 x float> %576, float %579, i64 1, !dbg !21
  %581 = fadd float %579, %577, !dbg !22
  %582 = extractelement <4 x float> %573, i64 2, !dbg !20
  %583 = tail call float @llvm.amdgcn.exp2.f32(float %582), !dbg !20
  %584 = insertelement <4 x float> %580, float %583, i64 2, !dbg !21
  %585 = fadd float %583, %581, !dbg !22
  %586 = extractelement <4 x float> %573, i64 3, !dbg !20
  %587 = tail call float @llvm.amdgcn.exp2.f32(float %586), !dbg !20
  %588 = insertelement <4 x float> %584, float %587, i64 3, !dbg !21
  %math.result105 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %588) #8, !dbg !21, !frisk.register_lifetimes !5
  %589 = fptrunc <4 x float> %math.result105 to <4 x half>, !dbg !21
  %math.packed106 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %589) #8, !dbg !22, !frisk.register_lifetimes !5
  %590 = fadd float %587, %585, !dbg !22
  %591 = shufflevector <32 x float> %496, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %592 = fadd <4 x float> %385, %591, !dbg !19
  %593 = extractelement <4 x float> %592, i64 0, !dbg !20
  %594 = tail call float @llvm.amdgcn.exp2.f32(float %593), !dbg !20
  %595 = insertelement <4 x float> poison, float %594, i64 0, !dbg !21
  %596 = fadd float %594, %590, !dbg !22
  %597 = extractelement <4 x float> %592, i64 1, !dbg !20
  %598 = tail call float @llvm.amdgcn.exp2.f32(float %597), !dbg !20
  %599 = insertelement <4 x float> %595, float %598, i64 1, !dbg !21
  %600 = fadd float %598, %596, !dbg !22
  %601 = extractelement <4 x float> %592, i64 2, !dbg !20
  %602 = tail call float @llvm.amdgcn.exp2.f32(float %601), !dbg !20
  %603 = insertelement <4 x float> %599, float %602, i64 2, !dbg !21
  %604 = fadd float %602, %600, !dbg !22
  %605 = extractelement <4 x float> %592, i64 3, !dbg !20
  %606 = tail call float @llvm.amdgcn.exp2.f32(float %605), !dbg !20
  %607 = insertelement <4 x float> %603, float %606, i64 3, !dbg !21
  %math.result107 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %607) #8, !dbg !21, !frisk.register_lifetimes !5
  %608 = fptrunc <4 x float> %math.result107 to <4 x half>, !dbg !21
  %math.packed108 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %608) #8, !dbg !22, !frisk.register_lifetimes !5
  %609 = fadd float %606, %604, !dbg !22
  %610 = shufflevector <32 x float> %496, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %611 = fadd <4 x float> %389, %610, !dbg !19
  %612 = extractelement <4 x float> %611, i64 0, !dbg !20
  %613 = tail call float @llvm.amdgcn.exp2.f32(float %612), !dbg !20
  %614 = insertelement <4 x float> poison, float %613, i64 0, !dbg !21
  %615 = fadd float %613, %609, !dbg !22
  %616 = extractelement <4 x float> %611, i64 1, !dbg !20
  %617 = tail call float @llvm.amdgcn.exp2.f32(float %616), !dbg !20
  %618 = insertelement <4 x float> %614, float %617, i64 1, !dbg !21
  %619 = fadd float %617, %615, !dbg !22
  %620 = extractelement <4 x float> %611, i64 2, !dbg !20
  %621 = tail call float @llvm.amdgcn.exp2.f32(float %620), !dbg !20
  %622 = insertelement <4 x float> %618, float %621, i64 2, !dbg !21
  %623 = fadd float %621, %619, !dbg !22
  %624 = extractelement <4 x float> %611, i64 3, !dbg !20
  %625 = tail call float @llvm.amdgcn.exp2.f32(float %624), !dbg !20
  %626 = insertelement <4 x float> %622, float %625, i64 3, !dbg !21
  %math.result109 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %626) #8, !dbg !21, !frisk.register_lifetimes !5
  %627 = fptrunc <4 x float> %math.result109 to <4 x half>, !dbg !21
  %math.packed110 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %627) #8, !dbg !22, !frisk.register_lifetimes !5
  %628 = fadd float %625, %623, !dbg !22
  %629 = shufflevector <32 x float> %496, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %630 = fadd <4 x float> %392, %629, !dbg !19
  %631 = extractelement <4 x float> %630, i64 0, !dbg !20
  %632 = tail call float @llvm.amdgcn.exp2.f32(float %631), !dbg !20
  %633 = insertelement <4 x float> poison, float %632, i64 0, !dbg !21
  %634 = fadd float %632, %628, !dbg !22
  %635 = extractelement <4 x float> %630, i64 1, !dbg !20
  %636 = tail call float @llvm.amdgcn.exp2.f32(float %635), !dbg !20
  %637 = insertelement <4 x float> %633, float %636, i64 1, !dbg !21
  %638 = fadd float %636, %634, !dbg !22
  %639 = extractelement <4 x float> %630, i64 2, !dbg !20
  %640 = tail call float @llvm.amdgcn.exp2.f32(float %639), !dbg !20
  %641 = insertelement <4 x float> %637, float %640, i64 2, !dbg !21
  %642 = fadd float %640, %638, !dbg !22
  %643 = extractelement <4 x float> %630, i64 3, !dbg !20
  %644 = tail call float @llvm.amdgcn.exp2.f32(float %643), !dbg !20
  %645 = insertelement <4 x float> %641, float %644, i64 3, !dbg !21
  %math.result111 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %645) #8, !dbg !21, !frisk.register_lifetimes !5
  %646 = fptrunc <4 x float> %math.result111 to <4 x half>, !dbg !21
  %math.packed112 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %646) #8, !dbg !22, !frisk.register_lifetimes !5
  %647 = fadd float %644, %642, !dbg !22
  %648 = extractvalue [2 x <32 x float>] %495, 1, !dbg !19
  %649 = shufflevector <32 x float> %648, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %650 = fadd <4 x float> %394, %649, !dbg !19
  %651 = extractelement <4 x float> %650, i64 0, !dbg !20
  %652 = tail call float @llvm.amdgcn.exp2.f32(float %651), !dbg !20
  %653 = insertelement <4 x float> poison, float %652, i64 0, !dbg !21
  %654 = extractelement <4 x float> %650, i64 1, !dbg !20
  %655 = tail call float @llvm.amdgcn.exp2.f32(float %654), !dbg !20
  %656 = insertelement <4 x float> %653, float %655, i64 1, !dbg !21
  %657 = fadd float %652, %655, !dbg !22
  %658 = extractelement <4 x float> %650, i64 2, !dbg !20
  %659 = tail call float @llvm.amdgcn.exp2.f32(float %658), !dbg !20
  %660 = insertelement <4 x float> %656, float %659, i64 2, !dbg !21
  %661 = fadd float %659, %657, !dbg !22
  %662 = extractelement <4 x float> %650, i64 3, !dbg !20
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %662), !dbg !20
  %664 = insertelement <4 x float> %660, float %663, i64 3, !dbg !21
  %math.result113 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %664) #8, !dbg !21, !frisk.register_lifetimes !5
  %665 = fptrunc <4 x float> %math.result113 to <4 x half>, !dbg !21
  %math.packed114 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %665) #8, !dbg !22, !frisk.register_lifetimes !5
  %666 = fadd float %663, %661, !dbg !22
  %667 = shufflevector <32 x float> %648, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %668 = fadd <4 x float> %396, %667, !dbg !19
  %669 = extractelement <4 x float> %668, i64 0, !dbg !20
  %670 = tail call float @llvm.amdgcn.exp2.f32(float %669), !dbg !20
  %671 = insertelement <4 x float> poison, float %670, i64 0, !dbg !21
  %672 = fadd float %670, %666, !dbg !22
  %673 = extractelement <4 x float> %668, i64 1, !dbg !20
  %674 = tail call float @llvm.amdgcn.exp2.f32(float %673), !dbg !20
  %675 = insertelement <4 x float> %671, float %674, i64 1, !dbg !21
  %676 = fadd float %674, %672, !dbg !22
  %677 = extractelement <4 x float> %668, i64 2, !dbg !20
  %678 = tail call float @llvm.amdgcn.exp2.f32(float %677), !dbg !20
  %679 = insertelement <4 x float> %675, float %678, i64 2, !dbg !21
  %680 = fadd float %678, %676, !dbg !22
  %681 = extractelement <4 x float> %668, i64 3, !dbg !20
  %682 = tail call float @llvm.amdgcn.exp2.f32(float %681), !dbg !20
  %683 = insertelement <4 x float> %679, float %682, i64 3, !dbg !21
  %math.result115 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %683) #8, !dbg !21, !frisk.register_lifetimes !5
  %684 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !21
  %math.packed116 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %684) #8, !dbg !22, !frisk.register_lifetimes !5
  %685 = fadd float %682, %680, !dbg !22
  %686 = shufflevector <32 x float> %648, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %687 = fadd <4 x float> %398, %686, !dbg !19
  %688 = extractelement <4 x float> %687, i64 0, !dbg !20
  %689 = tail call float @llvm.amdgcn.exp2.f32(float %688), !dbg !20
  %690 = insertelement <4 x float> poison, float %689, i64 0, !dbg !21
  %691 = fadd float %689, %685, !dbg !22
  %692 = extractelement <4 x float> %687, i64 1, !dbg !20
  %693 = tail call float @llvm.amdgcn.exp2.f32(float %692), !dbg !20
  %694 = insertelement <4 x float> %690, float %693, i64 1, !dbg !21
  %695 = fadd float %693, %691, !dbg !22
  %696 = extractelement <4 x float> %687, i64 2, !dbg !20
  %697 = tail call float @llvm.amdgcn.exp2.f32(float %696), !dbg !20
  %698 = insertelement <4 x float> %694, float %697, i64 2, !dbg !21
  %699 = fadd float %697, %695, !dbg !22
  %700 = extractelement <4 x float> %687, i64 3, !dbg !20
  %701 = tail call float @llvm.amdgcn.exp2.f32(float %700), !dbg !20
  %702 = insertelement <4 x float> %698, float %701, i64 3, !dbg !21
  %math.result117 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %702) #8, !dbg !21, !frisk.register_lifetimes !5
  %703 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !21
  %math.packed118 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %703) #8, !dbg !22, !frisk.register_lifetimes !5
  %704 = fadd float %701, %699, !dbg !22
  %705 = shufflevector <32 x float> %648, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %706 = fadd <4 x float> %400, %705, !dbg !19
  %707 = extractelement <4 x float> %706, i64 0, !dbg !20
  %708 = tail call float @llvm.amdgcn.exp2.f32(float %707), !dbg !20
  %709 = insertelement <4 x float> poison, float %708, i64 0, !dbg !21
  %710 = fadd float %708, %704, !dbg !22
  %711 = extractelement <4 x float> %706, i64 1, !dbg !20
  %712 = tail call float @llvm.amdgcn.exp2.f32(float %711), !dbg !20
  %713 = insertelement <4 x float> %709, float %712, i64 1, !dbg !21
  %714 = fadd float %712, %710, !dbg !22
  %715 = extractelement <4 x float> %706, i64 2, !dbg !20
  %716 = tail call float @llvm.amdgcn.exp2.f32(float %715), !dbg !20
  %717 = insertelement <4 x float> %713, float %716, i64 2, !dbg !21
  %718 = fadd float %716, %714, !dbg !22
  %719 = extractelement <4 x float> %706, i64 3, !dbg !20
  %720 = tail call float @llvm.amdgcn.exp2.f32(float %719), !dbg !20
  %721 = insertelement <4 x float> %717, float %720, i64 3, !dbg !21
  %math.result119 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %721) #8, !dbg !21, !frisk.register_lifetimes !5
  %722 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !21
  %math.packed120 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %722) #8, !dbg !22, !frisk.register_lifetimes !5
  %723 = fadd float %720, %718, !dbg !22
  %724 = shufflevector <32 x float> %648, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %725 = fadd <4 x float> %402, %724, !dbg !19
  %726 = extractelement <4 x float> %725, i64 0, !dbg !20
  %727 = tail call float @llvm.amdgcn.exp2.f32(float %726), !dbg !20
  %728 = insertelement <4 x float> poison, float %727, i64 0, !dbg !21
  %729 = fadd float %727, %723, !dbg !22
  %730 = extractelement <4 x float> %725, i64 1, !dbg !20
  %731 = tail call float @llvm.amdgcn.exp2.f32(float %730), !dbg !20
  %732 = insertelement <4 x float> %728, float %731, i64 1, !dbg !21
  %733 = fadd float %731, %729, !dbg !22
  %734 = extractelement <4 x float> %725, i64 2, !dbg !20
  %735 = tail call float @llvm.amdgcn.exp2.f32(float %734), !dbg !20
  %736 = insertelement <4 x float> %732, float %735, i64 2, !dbg !21
  %737 = fadd float %735, %733, !dbg !22
  %738 = extractelement <4 x float> %725, i64 3, !dbg !20
  %739 = tail call float @llvm.amdgcn.exp2.f32(float %738), !dbg !20
  %740 = insertelement <4 x float> %736, float %739, i64 3, !dbg !21
  %math.result121 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %740) #8, !dbg !21, !frisk.register_lifetimes !5
  %741 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !21
  %math.packed122 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %741) #8, !dbg !22, !frisk.register_lifetimes !5
  %742 = fadd float %739, %737, !dbg !22
  %743 = shufflevector <32 x float> %648, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %744 = fadd <4 x float> %404, %743, !dbg !19
  %745 = extractelement <4 x float> %744, i64 0, !dbg !20
  %746 = tail call float @llvm.amdgcn.exp2.f32(float %745), !dbg !20
  %747 = insertelement <4 x float> poison, float %746, i64 0, !dbg !21
  %748 = fadd float %746, %742, !dbg !22
  %749 = extractelement <4 x float> %744, i64 1, !dbg !20
  %750 = tail call float @llvm.amdgcn.exp2.f32(float %749), !dbg !20
  %751 = insertelement <4 x float> %747, float %750, i64 1, !dbg !21
  %752 = fadd float %750, %748, !dbg !22
  %753 = extractelement <4 x float> %744, i64 2, !dbg !20
  %754 = tail call float @llvm.amdgcn.exp2.f32(float %753), !dbg !20
  %755 = insertelement <4 x float> %751, float %754, i64 2, !dbg !21
  %756 = fadd float %754, %752, !dbg !22
  %757 = extractelement <4 x float> %744, i64 3, !dbg !20
  %758 = tail call float @llvm.amdgcn.exp2.f32(float %757), !dbg !20
  %759 = insertelement <4 x float> %755, float %758, i64 3, !dbg !21
  %math.result123 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %759) #8, !dbg !21, !frisk.register_lifetimes !5
  %760 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !21
  %math.packed124 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %760) #8, !dbg !22, !frisk.register_lifetimes !5
  %761 = fadd float %758, %756, !dbg !22
  %762 = shufflevector <32 x float> %648, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %763 = fadd <4 x float> %406, %762, !dbg !19
  %764 = extractelement <4 x float> %763, i64 0, !dbg !20
  %765 = tail call float @llvm.amdgcn.exp2.f32(float %764), !dbg !20
  %766 = insertelement <4 x float> poison, float %765, i64 0, !dbg !21
  %767 = fadd float %765, %761, !dbg !22
  %768 = extractelement <4 x float> %763, i64 1, !dbg !20
  %769 = tail call float @llvm.amdgcn.exp2.f32(float %768), !dbg !20
  %770 = insertelement <4 x float> %766, float %769, i64 1, !dbg !21
  %771 = fadd float %769, %767, !dbg !22
  %772 = extractelement <4 x float> %763, i64 2, !dbg !20
  %773 = tail call float @llvm.amdgcn.exp2.f32(float %772), !dbg !20
  %774 = insertelement <4 x float> %770, float %773, i64 2, !dbg !21
  %775 = fadd float %773, %771, !dbg !22
  %776 = extractelement <4 x float> %763, i64 3, !dbg !20
  %777 = tail call float @llvm.amdgcn.exp2.f32(float %776), !dbg !20
  %778 = insertelement <4 x float> %774, float %777, i64 3, !dbg !21
  %math.result125 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %778) #8, !dbg !21, !frisk.register_lifetimes !5
  %779 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !21
  %math.packed126 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %779) #8, !dbg !22, !frisk.register_lifetimes !5
  %780 = fadd float %777, %775, !dbg !22
  %781 = shufflevector <32 x float> %648, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %782 = fadd <4 x float> %409, %781, !dbg !19
  %783 = extractelement <4 x float> %782, i64 0, !dbg !20
  %784 = tail call float @llvm.amdgcn.exp2.f32(float %783), !dbg !20
  %785 = insertelement <4 x float> poison, float %784, i64 0, !dbg !21
  %786 = fadd float %784, %780, !dbg !22
  %787 = extractelement <4 x float> %782, i64 1, !dbg !20
  %788 = tail call float @llvm.amdgcn.exp2.f32(float %787), !dbg !20
  %789 = insertelement <4 x float> %785, float %788, i64 1, !dbg !21
  %790 = fadd float %788, %786, !dbg !22
  %791 = extractelement <4 x float> %782, i64 2, !dbg !20
  %792 = tail call float @llvm.amdgcn.exp2.f32(float %791), !dbg !20
  %793 = insertelement <4 x float> %789, float %792, i64 2, !dbg !21
  %794 = fadd float %792, %790, !dbg !22
  %795 = extractelement <4 x float> %782, i64 3, !dbg !20
  %796 = tail call float @llvm.amdgcn.exp2.f32(float %795), !dbg !20
  %797 = insertelement <4 x float> %793, float %796, i64 3, !dbg !21
  %math.result127 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %797) #8, !dbg !21, !frisk.register_lifetimes !5
  %798 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !21
  %math.packed128 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %798) #8, !dbg !22, !frisk.register_lifetimes !5
  %799 = fadd float %796, %794, !dbg !22
  %800 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !22
  %801 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %800), !dbg !22
  %802 = add nuw nsw i32 %801, 64, !dbg !22
  %803 = xor i32 %801, 16, !dbg !22
  %804 = and i32 %802, 192, !dbg !22
  %805 = icmp samesign ult i32 %803, %804, !dbg !22
  %806 = select i1 %805, i32 %803, i32 %801, !dbg !22
  %807 = shl nuw nsw i32 %806, 2, !dbg !22
  %808 = bitcast float %647 to i32, !dbg !22
  %809 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %808), !dbg !22
  %810 = bitcast i32 %809 to float, !dbg !22
  %811 = fadd float %647, %810, !dbg !22
  %812 = xor i32 %801, 32, !dbg !22
  %813 = icmp samesign ult i32 %812, %804, !dbg !22
  %814 = select i1 %813, i32 %812, i32 %801, !dbg !22
  %815 = shl nuw nsw i32 %814, 2, !dbg !22
  %816 = bitcast float %811 to i32, !dbg !22
  %817 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %815, i32 %816), !dbg !22
  %818 = bitcast i32 %817 to float, !dbg !22
  %819 = fadd float %811, %818, !dbg !22
  %820 = extractvalue [2 x <1 x float>] %55, 0, !dbg !22
  %821 = extractelement <1 x float> %820, i64 0, !dbg !22
  %822 = fadd float %821, %819, !dbg !22
  %823 = insertelement <1 x float> poison, float %822, i64 0, !dbg !13
  %824 = insertvalue [2 x <1 x float>] poison, <1 x float> %823, 0, !dbg !13
  %825 = bitcast float %799 to i32, !dbg !22
  %826 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %825), !dbg !22
  %827 = bitcast i32 %826 to float, !dbg !22
  %828 = fadd float %799, %827, !dbg !22
  %829 = bitcast float %828 to i32, !dbg !22
  %830 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %815, i32 %829), !dbg !22
  %831 = bitcast i32 %830 to float, !dbg !22
  %832 = fadd float %828, %831, !dbg !22
  %833 = extractvalue [2 x <1 x float>] %55, 1, !dbg !22
  %834 = extractelement <1 x float> %833, i64 0, !dbg !22
  %835 = fadd float %834, %832, !dbg !22
  %836 = insertelement <1 x float> poison, float %835, i64 0, !dbg !13
  %837 = insertvalue [2 x <1 x float>] %824, <1 x float> %836, 1, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.thread319 = tail call i32 asm sideeffect "", "=v,0"(i32 %21) #8, !dbg !23, !range !11, !frisk.register_lifetimes !5
  %address.local320 = shl nuw nsw i32 %address.thread319, 3, !dbg !23
  %address.local321 = and i32 %address.local320, 120, !dbg !23
  %address.local322 = shl nuw nsw i32 %address.thread319, 5, !dbg !23
  %address.local323 = and i32 %address.local322, 1536, !dbg !23
  %address.local324 = or disjoint i32 %address.local321, %address.local323, !dbg !23
  %address.local325 = zext nneg i32 %address.local324 to i64, !dbg !23
  %address.local326 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local325, !dbg !23
  %838 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #11, !dbg !23
  %839 = bitcast <4 x i32> %838 to <8 x half>, !dbg !23
  %address.local327 = zext nneg i32 %address.thread319 to i64, !dbg !10
  %address.local328 = and i64 %address.local327, 60, !dbg !23
  %address.local329 = add nuw nsw i64 %address.local328, %address.local327, !dbg !23
  %address.local331.idx = shl nuw nsw i64 %address.local329, 4, !dbg !23
  %address.local331 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local331.idx, !dbg !23
  %address.local332 = lshr i64 %address.local327, 4, !dbg !23
  %address.local333 = and i64 %address.local332, 3, !dbg !23
  %address.local335.idx = mul nuw nsw i64 %address.local333, 768, !dbg !23
  %address.local335 = getelementptr i8, ptr addrspace(3) %address.local331, i64 %address.local335.idx, !dbg !23
  %address.local336 = lshr i64 %address.local327, 2, !dbg !15
  %address.local338.idx = mul nsw i64 %address.local336, -64, !dbg !23
  %address.local338 = getelementptr i8, ptr addrspace(3) %address.local335, i64 %address.local338.idx, !dbg !23
  %address.local339 = getelementptr i8, ptr addrspace(3) %address.local338, i64 256, !dbg !23
  %840 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #11, !dbg !23
  %841 = bitcast <4 x i32> %840 to <8 x half>, !dbg !23
  %address.local343 = getelementptr i8, ptr addrspace(3) %address.local338, i64 512, !dbg !23
  %842 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #11, !dbg !23
  %843 = bitcast <4 x i32> %842 to <8 x half>, !dbg !23
  %address.local347 = getelementptr i8, ptr addrspace(3) %address.local338, i64 768, !dbg !23
  %844 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #11, !dbg !23
  %845 = bitcast <4 x i32> %844 to <8 x half>, !dbg !23
  %address.local349 = getelementptr i8, ptr addrspace(3) %address.local326, i64 4096, !dbg !23
  %846 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #11, !dbg !23
  %847 = bitcast <4 x i32> %846 to <8 x half>, !dbg !23
  %address.local353 = getelementptr i8, ptr addrspace(3) %address.local338, i64 4352, !dbg !23
  %848 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #11, !dbg !23
  %849 = bitcast <4 x i32> %848 to <8 x half>, !dbg !23
  %address.local357 = getelementptr i8, ptr addrspace(3) %address.local338, i64 4608, !dbg !23
  %850 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #11, !dbg !23
  %851 = bitcast <4 x i32> %850 to <8 x half>, !dbg !23
  %address.local361 = getelementptr i8, ptr addrspace(3) %address.local338, i64 4864, !dbg !23
  %852 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #11, !dbg !23
  %853 = bitcast <4 x i32> %852 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local362 = shl nuw nsw i64 %address.local327, 3, !dbg !9
  %address.local363 = and i64 %address.local362, 1536, !dbg !9
  %address.local364 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local363, !dbg !9
  %address.local365 = ptrtoint ptr addrspace(3) %address.local364 to i64, !dbg !9
  %address.local366 = trunc i64 %address.local365 to i32, !dbg !9
  %854 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local366), !dbg !16
  %address.local367 = shl nuw nsw i64 %address.local327, 5, !dbg !16
  %address.local368 = and i64 %address.local367, 8064, !dbg !16
  %address.local370 = and i64 %address.local362, 24, !dbg !16
  %address.local371 = or disjoint i64 %address.local368, %address.local370, !dbg !16
  %855 = or disjoint i64 %.idx5659, %address.local371, !dbg !16
  %856 = trunc i64 %855 to i32, !dbg !16
  %.tr67 = shl i32 %856, 1, !dbg !16
  %dma.buffer.offset880 = or i32 %.tr67, 64, !dbg !16
  %dma.buffer.lds882 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %854), !dbg !16
  %857 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds882, i32 %dma.buffer.offset880, <4 x i32> %dma.buffer.rsrc887) #9, !dbg !16
  %address.local373 = or disjoint i64 %address.local327, 256, !dbg !9
  %address.local374 = shl nuw nsw i64 %address.local373, 3, !dbg !9
  %address.local375 = and i64 %address.local374, 3584, !dbg !9
  %address.local376 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local375, !dbg !9
  %address.local377 = ptrtoint ptr addrspace(3) %address.local376 to i64, !dbg !9
  %address.local378 = trunc i64 %address.local377 to i32, !dbg !9
  %858 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local378), !dbg !16
  %address.local379 = shl nuw nsw i64 %address.local373, 5, !dbg !16
  %address.local380 = and i64 %address.local379, 16256, !dbg !16
  %address.local383 = or disjoint i64 %address.local380, %address.local370, !dbg !16
  %859 = or disjoint i64 %.idx5659, %address.local383, !dbg !16
  %860 = trunc i64 %859 to i32, !dbg !16
  %.tr71 = shl i32 %860, 1, !dbg !16
  %dma.buffer.offset877 = or i32 %.tr71, 64, !dbg !16
  %dma.buffer.lds879 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %858), !dbg !16
  %861 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds879, i32 %dma.buffer.offset877, <4 x i32> %dma.buffer.rsrc887) #9, !dbg !16
  %862 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %863 = extractvalue [2 x <8 x float>] %56, 0, !dbg !24
  %864 = shufflevector <8 x float> %863, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %862, <4 x float> %864) #8, !dbg !23
  %866 = shufflevector <8 x half> %841, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %866, <4 x float> %865) #8, !dbg !23
  %868 = shufflevector <8 x half> %843, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %868, <4 x float> %867) #8, !dbg !23
  %870 = shufflevector <8 x half> %845, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %870, <4 x float> %869) #8, !dbg !23
  %872 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %872, <4 x float> %871) #8, !dbg !23
  %874 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %874, <4 x float> %873) #8, !dbg !23
  %876 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %876, <4 x float> %875) #8, !dbg !23
  %878 = shufflevector <8 x half> %853, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %878, <4 x float> %877) #8, !dbg !23
  %880 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %881 = shufflevector <8 x float> %863, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %882 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %880, <4 x float> %881) #8, !dbg !23
  %883 = shufflevector <8 x half> %841, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %883, <4 x float> %882) #8, !dbg !23
  %885 = shufflevector <8 x half> %843, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %885, <4 x float> %884) #8, !dbg !23
  %887 = shufflevector <8 x half> %845, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %888 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %887, <4 x float> %886) #8, !dbg !23
  %889 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %890 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %889, <4 x float> %888) #8, !dbg !23
  %891 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %892 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %891, <4 x float> %890) #8, !dbg !23
  %893 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %893, <4 x float> %892) #8, !dbg !23
  %895 = shufflevector <8 x half> %853, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %895, <4 x float> %894) #8, !dbg !23
  %897 = extractvalue [2 x <8 x float>] %56, 1, !dbg !24
  %898 = shufflevector <8 x float> %897, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %862, <4 x float> %898) #8, !dbg !23
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %866, <4 x float> %899) #8, !dbg !23
  %901 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %868, <4 x float> %900) #8, !dbg !23
  %902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %870, <4 x float> %901) #8, !dbg !23
  %903 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %872, <4 x float> %902) #8, !dbg !23
  %904 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %874, <4 x float> %903) #8, !dbg !23
  %905 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %876, <4 x float> %904) #8, !dbg !23
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %878, <4 x float> %905) #8, !dbg !23
  %907 = shufflevector <8 x float> %897, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %908 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %880, <4 x float> %907) #8, !dbg !23
  %909 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %883, <4 x float> %908) #8, !dbg !23
  %910 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %885, <4 x float> %909) #8, !dbg !23
  %911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %887, <4 x float> %910) #8, !dbg !23
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %889, <4 x float> %911) #8, !dbg !23
  %913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %891, <4 x float> %912) #8, !dbg !23
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %893, <4 x float> %913) #8, !dbg !23
  %915 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %895, <4 x float> %914) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %916 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #11, !dbg !23
  %917 = bitcast <4 x i32> %916 to <8 x half>, !dbg !23
  %918 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #11, !dbg !23
  %919 = bitcast <4 x i32> %918 to <8 x half>, !dbg !23
  %920 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #11, !dbg !23
  %921 = bitcast <4 x i32> %920 to <8 x half>, !dbg !23
  %922 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #11, !dbg !23
  %923 = bitcast <4 x i32> %922 to <8 x half>, !dbg !23
  %924 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #11, !dbg !23
  %925 = bitcast <4 x i32> %924 to <8 x half>, !dbg !23
  %926 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #11, !dbg !23
  %927 = bitcast <4 x i32> %926 to <8 x half>, !dbg !23
  %928 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #11, !dbg !23
  %929 = bitcast <4 x i32> %928 to <8 x half>, !dbg !23
  %930 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #11, !dbg !23
  %931 = bitcast <4 x i32> %930 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %dma.buffer.offset874 = or i32 %.tr67, 128, !dbg !16
  %932 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds882, i32 %dma.buffer.offset874, <4 x i32> %dma.buffer.rsrc887) #9, !dbg !16
  %dma.buffer.offset871 = or i32 %.tr71, 128, !dbg !16
  %933 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds879, i32 %dma.buffer.offset871, <4 x i32> %dma.buffer.rsrc887) #9, !dbg !16
  %934 = shufflevector <8 x half> %917, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %935 = extractvalue [2 x <8 x float>] %54, 0, !dbg !24
  %936 = shufflevector <8 x float> %935, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %937 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %934, <4 x float> %936) #8, !dbg !23
  %938 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %939 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %938, <4 x float> %937) #8, !dbg !23
  %940 = shufflevector <8 x half> %921, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %941 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %940, <4 x float> %939) #8, !dbg !23
  %942 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %943 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %942, <4 x float> %941) #8, !dbg !23
  %944 = shufflevector <8 x half> %925, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %945 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %944, <4 x float> %943) #8, !dbg !23
  %946 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %947 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %946, <4 x float> %945) #8, !dbg !23
  %948 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %949 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %948, <4 x float> %947) #8, !dbg !23
  %950 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %951 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %950, <4 x float> %949) #8, !dbg !23
  %952 = shufflevector <8 x half> %917, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %953 = shufflevector <8 x float> %935, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %952, <4 x float> %953) #8, !dbg !23
  %955 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %955, <4 x float> %954) #8, !dbg !23
  %957 = shufflevector <8 x half> %921, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %958 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %957, <4 x float> %956) #8, !dbg !23
  %959 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %959, <4 x float> %958) #8, !dbg !23
  %961 = shufflevector <8 x half> %925, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %961, <4 x float> %960) #8, !dbg !23
  %963 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %963, <4 x float> %962) #8, !dbg !23
  %965 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %966 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %965, <4 x float> %964) #8, !dbg !23
  %967 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %967, <4 x float> %966) #8, !dbg !23
  %969 = extractvalue [2 x <8 x float>] %54, 1, !dbg !24
  %970 = shufflevector <8 x float> %969, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %971 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %934, <4 x float> %970) #8, !dbg !23
  %972 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %938, <4 x float> %971) #8, !dbg !23
  %973 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %940, <4 x float> %972) #8, !dbg !23
  %974 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %942, <4 x float> %973) #8, !dbg !23
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %944, <4 x float> %974) #8, !dbg !23
  %976 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %946, <4 x float> %975) #8, !dbg !23
  %977 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %948, <4 x float> %976) #8, !dbg !23
  %978 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %950, <4 x float> %977) #8, !dbg !23
  %979 = shufflevector <8 x float> %969, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %952, <4 x float> %979) #8, !dbg !23
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %955, <4 x float> %980) #8, !dbg !23
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %957, <4 x float> %981) #8, !dbg !23
  %983 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %959, <4 x float> %982) #8, !dbg !23
  %984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %961, <4 x float> %983) #8, !dbg !23
  %985 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %963, <4 x float> %984) #8, !dbg !23
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %965, <4 x float> %985) #8, !dbg !23
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %967, <4 x float> %986) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %988 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #11, !dbg !23
  %989 = bitcast <4 x i32> %988 to <8 x half>, !dbg !23
  %990 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #11, !dbg !23
  %991 = bitcast <4 x i32> %990 to <8 x half>, !dbg !23
  %992 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #11, !dbg !23
  %993 = bitcast <4 x i32> %992 to <8 x half>, !dbg !23
  %994 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #11, !dbg !23
  %995 = bitcast <4 x i32> %994 to <8 x half>, !dbg !23
  %996 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #11, !dbg !23
  %997 = bitcast <4 x i32> %996 to <8 x half>, !dbg !23
  %998 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #11, !dbg !23
  %999 = bitcast <4 x i32> %998 to <8 x half>, !dbg !23
  %1000 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #11, !dbg !23
  %1001 = bitcast <4 x i32> %1000 to <8 x half>, !dbg !23
  %1002 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #11, !dbg !23
  %1003 = bitcast <4 x i32> %1002 to <8 x half>, !dbg !23
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1004 = or disjoint i64 %.idx5659, 96, !dbg !16
  %1005 = or disjoint i64 %address.local371, %1004, !dbg !16
  %.tr83 = trunc i64 %1005 to i32, !dbg !16
  %dma.buffer.offset868 = shl i32 %.tr83, 1, !dbg !16
  %1006 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds882, i32 %dma.buffer.offset868, <4 x i32> %dma.buffer.rsrc887) #9, !dbg !16
  %1007 = or disjoint i64 %address.local383, %1004, !dbg !16
  %.tr87 = trunc i64 %1007 to i32, !dbg !16
  %dma.buffer.offset = shl i32 %.tr87, 1, !dbg !16
  %1008 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds879, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc887) #9, !dbg !16
  %1009 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1010 = extractvalue [2 x <8 x float>] %53, 0, !dbg !24
  %1011 = shufflevector <8 x float> %1010, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1009, <4 x float> %1011) #8, !dbg !23
  %1013 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1013, <4 x float> %1012) #8, !dbg !23
  %1015 = shufflevector <8 x half> %993, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1016 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1015, <4 x float> %1014) #8, !dbg !23
  %1017 = shufflevector <8 x half> %995, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1018 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1017, <4 x float> %1016) #8, !dbg !23
  %1019 = shufflevector <8 x half> %997, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1019, <4 x float> %1018) #8, !dbg !23
  %1021 = shufflevector <8 x half> %999, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1022 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1021, <4 x float> %1020) #8, !dbg !23
  %1023 = shufflevector <8 x half> %1001, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1023, <4 x float> %1022) #8, !dbg !23
  %1025 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1025, <4 x float> %1024) #8, !dbg !23
  %1027 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1028 = shufflevector <8 x float> %1010, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1029 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1027, <4 x float> %1028) #8, !dbg !23
  %1030 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1031 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1030, <4 x float> %1029) #8, !dbg !23
  %1032 = shufflevector <8 x half> %993, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1033 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1032, <4 x float> %1031) #8, !dbg !23
  %1034 = shufflevector <8 x half> %995, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1035 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1034, <4 x float> %1033) #8, !dbg !23
  %1036 = shufflevector <8 x half> %997, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1037 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1036, <4 x float> %1035) #8, !dbg !23
  %1038 = shufflevector <8 x half> %999, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1039 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1038, <4 x float> %1037) #8, !dbg !23
  %1040 = shufflevector <8 x half> %1001, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1041 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1040, <4 x float> %1039) #8, !dbg !23
  %1042 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1043 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1042, <4 x float> %1041) #8, !dbg !23
  %1044 = extractvalue [2 x <8 x float>] %53, 1, !dbg !24
  %1045 = shufflevector <8 x float> %1044, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1046 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1009, <4 x float> %1045) #8, !dbg !23
  %1047 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1013, <4 x float> %1046) #8, !dbg !23
  %1048 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1015, <4 x float> %1047) #8, !dbg !23
  %1049 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1017, <4 x float> %1048) #8, !dbg !23
  %1050 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1019, <4 x float> %1049) #8, !dbg !23
  %1051 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1021, <4 x float> %1050) #8, !dbg !23
  %1052 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1023, <4 x float> %1051) #8, !dbg !23
  %1053 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1025, <4 x float> %1052) #8, !dbg !23
  %1054 = shufflevector <8 x float> %1044, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1055 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1027, <4 x float> %1054) #8, !dbg !23
  %1056 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1030, <4 x float> %1055) #8, !dbg !23
  %1057 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1032, <4 x float> %1056) #8, !dbg !23
  %1058 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1034, <4 x float> %1057) #8, !dbg !23
  %1059 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1036, <4 x float> %1058) #8, !dbg !23
  %1060 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1038, <4 x float> %1059) #8, !dbg !23
  %1061 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1040, <4 x float> %1060) #8, !dbg !23
  %1062 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1042, <4 x float> %1061) #8, !dbg !23
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1063 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #11, !dbg !23
  %1064 = bitcast <4 x i32> %1063 to <8 x half>, !dbg !23
  %1065 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #11, !dbg !23
  %1066 = bitcast <4 x i32> %1065 to <8 x half>, !dbg !23
  %1067 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #11, !dbg !23
  %1068 = bitcast <4 x i32> %1067 to <8 x half>, !dbg !23
  %1069 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #11, !dbg !23
  %1070 = bitcast <4 x i32> %1069 to <8 x half>, !dbg !23
  %1071 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #11, !dbg !23
  %1072 = bitcast <4 x i32> %1071 to <8 x half>, !dbg !23
  %1073 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #11, !dbg !23
  %1074 = bitcast <4 x i32> %1073 to <8 x half>, !dbg !23
  %1075 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #11, !dbg !23
  %1076 = bitcast <4 x i32> %1075 to <8 x half>, !dbg !23
  %1077 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #11, !dbg !23
  %1078 = bitcast <4 x i32> %1077 to <8 x half>, !dbg !23
  %1079 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1080 = extractvalue [2 x <8 x float>] %52, 0, !dbg !24
  %1081 = shufflevector <8 x float> %1080, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1079, <4 x float> %1081) #8, !dbg !23
  %1083 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1083, <4 x float> %1082) #8, !dbg !23
  %1085 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1085, <4 x float> %1084) #8, !dbg !23
  %1087 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1087, <4 x float> %1086) #8, !dbg !23
  %1089 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1089, <4 x float> %1088) #8, !dbg !23
  %1091 = shufflevector <8 x half> %1074, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1091, <4 x float> %1090) #8, !dbg !23
  %1093 = shufflevector <8 x half> %1076, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1093, <4 x float> %1092) #8, !dbg !23
  %1095 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1095, <4 x float> %1094) #8, !dbg !23
  %1097 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1098 = shufflevector <8 x float> %1080, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1099 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1097, <4 x float> %1098) #8, !dbg !23
  %1100 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1101 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1100, <4 x float> %1099) #8, !dbg !23
  %1102 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1103 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1102, <4 x float> %1101) #8, !dbg !23
  %1104 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1105 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1104, <4 x float> %1103) #8, !dbg !23
  %1106 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1107 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1106, <4 x float> %1105) #8, !dbg !23
  %1108 = shufflevector <8 x half> %1074, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1109 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1108, <4 x float> %1107) #8, !dbg !23
  %1110 = shufflevector <8 x half> %1076, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1111 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1110, <4 x float> %1109) #8, !dbg !23
  %1112 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %1113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1112, <4 x float> %1111) #8, !dbg !23
  %1114 = extractvalue [2 x <8 x float>] %52, 1, !dbg !24
  %1115 = shufflevector <8 x float> %1114, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1079, <4 x float> %1115) #8, !dbg !23
  %1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1083, <4 x float> %1116) #8, !dbg !23
  %1118 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1085, <4 x float> %1117) #8, !dbg !23
  %1119 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1087, <4 x float> %1118) #8, !dbg !23
  %1120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1089, <4 x float> %1119) #8, !dbg !23
  %1121 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1091, <4 x float> %1120) #8, !dbg !23
  %1122 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1093, <4 x float> %1121) #8, !dbg !23
  %1123 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1095, <4 x float> %1122) #8, !dbg !23
  %1124 = shufflevector <8 x float> %1114, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1125 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1097, <4 x float> %1124) #8, !dbg !23
  %1126 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1100, <4 x float> %1125) #8, !dbg !23
  %1127 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1102, <4 x float> %1126) #8, !dbg !23
  %1128 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1104, <4 x float> %1127) #8, !dbg !23
  %1129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1106, <4 x float> %1128) #8, !dbg !23
  %1130 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1108, <4 x float> %1129) #8, !dbg !23
  %1131 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1110, <4 x float> %1130) #8, !dbg !23
  %1132 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1112, <4 x float> %1131) #8, !dbg !23
  %1133 = shufflevector <4 x float> %879, <4 x float> %896, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1134 = insertvalue [2 x <8 x float>] poison, <8 x float> %1133, 0, !dbg !13
  %1135 = shufflevector <4 x float> %906, <4 x float> %915, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1136 = insertvalue [2 x <8 x float>] %1134, <8 x float> %1135, 1, !dbg !13
  %1137 = shufflevector <4 x float> %951, <4 x float> %968, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1138 = insertvalue [2 x <8 x float>] poison, <8 x float> %1137, 0, !dbg !13
  %1139 = shufflevector <4 x float> %978, <4 x float> %987, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1140 = insertvalue [2 x <8 x float>] %1138, <8 x float> %1139, 1, !dbg !13
  %1141 = shufflevector <4 x float> %1026, <4 x float> %1043, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1142 = insertvalue [2 x <8 x float>] poison, <8 x float> %1141, 0, !dbg !13
  %1143 = shufflevector <4 x float> %1053, <4 x float> %1062, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1144 = insertvalue [2 x <8 x float>] %1142, <8 x float> %1143, 1, !dbg !13
  %1145 = shufflevector <4 x float> %1096, <4 x float> %1113, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1146 = insertvalue [2 x <8 x float>] poison, <8 x float> %1145, 0, !dbg !13
  %1147 = shufflevector <4 x float> %1123, <4 x float> %1132, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1148 = insertvalue [2 x <8 x float>] %1146, <8 x float> %1147, 1, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %1149 = add nuw nsw i64 %57, 1, !dbg !13
  %.not.not = icmp slt i64 %57, %8, !dbg !13
  br i1 %.not.not, label %51, label %._crit_edge, !dbg !13

._crit_edge:                                      ; preds = %494, %.._crit_edge_crit_edge
  %.pre-phi117 = phi i32 [ %.pre116, %.._crit_edge_crit_edge ], [ %812, %494 ], !dbg !13
  %.pre-phi115 = phi i32 [ %.pre114, %.._crit_edge_crit_edge ], [ %804, %494 ], !dbg !13
  %.pre-phi113 = phi i32 [ %.pre112, %.._crit_edge_crit_edge ], [ %803, %494 ], !dbg !13
  %.pre-phi109 = phi i32 [ %.pre108, %.._crit_edge_crit_edge ], [ %801, %494 ], !dbg !13
  %.lcssa93 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1136, %494 ], !dbg !13
  %.lcssa92 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %837, %494 ], !dbg !13
  %.lcssa91 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1140, %494 ], !dbg !13
  %.lcssa90 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1144, %494 ], !dbg !13
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1148, %494 ], !dbg !13
  %1150 = and i32 %21, 16, !dbg !13
  %.not2 = icmp eq i32 %1150, 0, !dbg !13
  %1151 = and i32 %21, 32, !dbg !13
  %.not3 = icmp eq i32 %1151, 0, !dbg !13
  %1152 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !13
  %1153 = extractelement <8 x float> %1152, i64 0, !dbg !13
  %1154 = extractelement <8 x float> %1152, i64 1, !dbg !13
  %1155 = extractelement <8 x float> %1152, i64 2, !dbg !13
  %1156 = extractelement <8 x float> %1152, i64 3, !dbg !13
  %1157 = select i1 %.not2, float %1154, float %1153, !dbg !13
  %1158 = icmp samesign ult i32 %.pre-phi113, %.pre-phi115, !dbg !13
  %1159 = select i1 %1158, i32 %.pre-phi113, i32 %.pre-phi109, !dbg !13
  %1160 = shl nuw nsw i32 %1159, 2, !dbg !13
  %1161 = bitcast float %1157 to i32, !dbg !13
  %1162 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1161), !dbg !13
  %1163 = bitcast i32 %1162 to float, !dbg !13
  %1164 = select i1 %.not2, float %1153, float %1163, !dbg !13
  %1165 = select i1 %.not2, float %1163, float %1154, !dbg !13
  %1166 = select i1 %.not2, float %1156, float %1155, !dbg !13
  %1167 = bitcast float %1166 to i32, !dbg !13
  %1168 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1167), !dbg !13
  %1169 = bitcast i32 %1168 to float, !dbg !13
  %1170 = select i1 %.not2, float %1155, float %1169, !dbg !13
  %1171 = select i1 %.not2, float %1169, float %1156, !dbg !13
  %1172 = select i1 %.not3, float %1170, float %1164, !dbg !13
  %1173 = icmp samesign ult i32 %.pre-phi117, %.pre-phi115, !dbg !13
  %1174 = select i1 %1173, i32 %.pre-phi117, i32 %.pre-phi109, !dbg !13
  %1175 = shl nuw nsw i32 %1174, 2, !dbg !13
  %1176 = bitcast float %1172 to i32, !dbg !13
  %1177 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1176), !dbg !13
  %1178 = bitcast i32 %1177 to float, !dbg !13
  %1179 = select i1 %.not3, float %1171, float %1165, !dbg !13
  %1180 = bitcast float %1179 to i32, !dbg !13
  %1181 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1180), !dbg !13
  %1182 = bitcast i32 %1181 to float, !dbg !13
  %1183 = insertelement <4 x i1> poison, i1 %.not3, i64 0, !dbg !13
  %1184 = shufflevector <4 x i1> %1183, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !13
  %1185 = insertelement <4 x float> poison, float %1164, i64 0, !dbg !13
  %1186 = insertelement <4 x float> %1185, float %1165, i64 1, !dbg !13
  %1187 = insertelement <4 x float> %1186, float %1178, i64 2, !dbg !13
  %1188 = insertelement <4 x float> %1187, float %1182, i64 3, !dbg !13
  %1189 = insertelement <4 x float> poison, float %1178, i64 0, !dbg !13
  %1190 = insertelement <4 x float> %1189, float %1182, i64 1, !dbg !13
  %1191 = insertelement <4 x float> %1190, float %1170, i64 2, !dbg !13
  %1192 = insertelement <4 x float> %1191, float %1171, i64 3, !dbg !13
  %1193 = select <4 x i1> %1184, <4 x float> %1188, <4 x float> %1192, !dbg !13
  %1194 = extractvalue [2 x <8 x float>] %.lcssa90, 0, !dbg !13
  %1195 = extractelement <8 x float> %1194, i64 0, !dbg !13
  %1196 = extractelement <8 x float> %1194, i64 1, !dbg !13
  %1197 = extractelement <8 x float> %1194, i64 2, !dbg !13
  %1198 = extractelement <8 x float> %1194, i64 3, !dbg !13
  %1199 = select i1 %.not2, float %1196, float %1195, !dbg !13
  %1200 = bitcast float %1199 to i32, !dbg !13
  %1201 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1200), !dbg !13
  %1202 = bitcast i32 %1201 to float, !dbg !13
  %1203 = select i1 %.not2, float %1195, float %1202, !dbg !13
  %1204 = select i1 %.not2, float %1202, float %1196, !dbg !13
  %1205 = select i1 %.not2, float %1198, float %1197, !dbg !13
  %1206 = bitcast float %1205 to i32, !dbg !13
  %1207 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1206), !dbg !13
  %1208 = bitcast i32 %1207 to float, !dbg !13
  %1209 = select i1 %.not2, float %1197, float %1208, !dbg !13
  %1210 = select i1 %.not2, float %1208, float %1198, !dbg !13
  %1211 = select i1 %.not3, float %1209, float %1203, !dbg !13
  %1212 = bitcast float %1211 to i32, !dbg !13
  %1213 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1212), !dbg !13
  %1214 = bitcast i32 %1213 to float, !dbg !13
  %1215 = select i1 %.not3, float %1210, float %1204, !dbg !13
  %1216 = bitcast float %1215 to i32, !dbg !13
  %1217 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1216), !dbg !13
  %1218 = bitcast i32 %1217 to float, !dbg !13
  %1219 = insertelement <4 x float> poison, float %1203, i64 0, !dbg !13
  %1220 = insertelement <4 x float> %1219, float %1204, i64 1, !dbg !13
  %1221 = insertelement <4 x float> %1220, float %1214, i64 2, !dbg !13
  %1222 = insertelement <4 x float> %1221, float %1218, i64 3, !dbg !13
  %1223 = insertelement <4 x float> poison, float %1214, i64 0, !dbg !13
  %1224 = insertelement <4 x float> %1223, float %1218, i64 1, !dbg !13
  %1225 = insertelement <4 x float> %1224, float %1209, i64 2, !dbg !13
  %1226 = insertelement <4 x float> %1225, float %1210, i64 3, !dbg !13
  %1227 = select <4 x i1> %1184, <4 x float> %1222, <4 x float> %1226, !dbg !13
  %1228 = extractvalue [2 x <8 x float>] %.lcssa91, 0, !dbg !13
  %1229 = extractelement <8 x float> %1228, i64 0, !dbg !13
  %1230 = extractelement <8 x float> %1228, i64 1, !dbg !13
  %1231 = extractelement <8 x float> %1228, i64 2, !dbg !13
  %1232 = extractelement <8 x float> %1228, i64 3, !dbg !13
  %1233 = select i1 %.not2, float %1230, float %1229, !dbg !13
  %1234 = bitcast float %1233 to i32, !dbg !13
  %1235 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1234), !dbg !13
  %1236 = bitcast i32 %1235 to float, !dbg !13
  %1237 = select i1 %.not2, float %1229, float %1236, !dbg !13
  %1238 = select i1 %.not2, float %1236, float %1230, !dbg !13
  %1239 = select i1 %.not2, float %1232, float %1231, !dbg !13
  %1240 = bitcast float %1239 to i32, !dbg !13
  %1241 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1240), !dbg !13
  %1242 = bitcast i32 %1241 to float, !dbg !13
  %1243 = select i1 %.not2, float %1231, float %1242, !dbg !13
  %1244 = select i1 %.not2, float %1242, float %1232, !dbg !13
  %1245 = select i1 %.not3, float %1243, float %1237, !dbg !13
  %1246 = bitcast float %1245 to i32, !dbg !13
  %1247 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1246), !dbg !13
  %1248 = bitcast i32 %1247 to float, !dbg !13
  %1249 = select i1 %.not3, float %1244, float %1238, !dbg !13
  %1250 = bitcast float %1249 to i32, !dbg !13
  %1251 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1250), !dbg !13
  %1252 = bitcast i32 %1251 to float, !dbg !13
  %1253 = insertelement <4 x float> poison, float %1237, i64 0, !dbg !13
  %1254 = insertelement <4 x float> %1253, float %1238, i64 1, !dbg !13
  %1255 = insertelement <4 x float> %1254, float %1248, i64 2, !dbg !13
  %1256 = insertelement <4 x float> %1255, float %1252, i64 3, !dbg !13
  %1257 = insertelement <4 x float> poison, float %1248, i64 0, !dbg !13
  %1258 = insertelement <4 x float> %1257, float %1252, i64 1, !dbg !13
  %1259 = insertelement <4 x float> %1258, float %1243, i64 2, !dbg !13
  %1260 = insertelement <4 x float> %1259, float %1244, i64 3, !dbg !13
  %1261 = select <4 x i1> %1184, <4 x float> %1256, <4 x float> %1260, !dbg !13
  %1262 = extractvalue [2 x <8 x float>] %.lcssa93, 0, !dbg !13
  %1263 = extractelement <8 x float> %1262, i64 0, !dbg !13
  %1264 = extractelement <8 x float> %1262, i64 1, !dbg !13
  %1265 = extractelement <8 x float> %1262, i64 2, !dbg !13
  %1266 = extractelement <8 x float> %1262, i64 3, !dbg !13
  %1267 = select i1 %.not2, float %1264, float %1263, !dbg !13
  %1268 = bitcast float %1267 to i32, !dbg !13
  %1269 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1268), !dbg !13
  %1270 = bitcast i32 %1269 to float, !dbg !13
  %1271 = select i1 %.not2, float %1263, float %1270, !dbg !13
  %1272 = select i1 %.not2, float %1270, float %1264, !dbg !13
  %1273 = select i1 %.not2, float %1266, float %1265, !dbg !13
  %1274 = bitcast float %1273 to i32, !dbg !13
  %1275 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1274), !dbg !13
  %1276 = bitcast i32 %1275 to float, !dbg !13
  %1277 = select i1 %.not2, float %1265, float %1276, !dbg !13
  %1278 = select i1 %.not2, float %1276, float %1266, !dbg !13
  %1279 = select i1 %.not3, float %1277, float %1271, !dbg !13
  %1280 = bitcast float %1279 to i32, !dbg !13
  %1281 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1280), !dbg !13
  %1282 = select i1 %.not3, float %1278, float %1272, !dbg !13
  %1283 = bitcast float %1282 to i32, !dbg !13
  %1284 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1283), !dbg !13
  %1285 = extractvalue [2 x <1 x float>] %.lcssa92, 0, !dbg !25
  %1286 = shufflevector <1 x float> %1285, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %1287 = extractelement <8 x float> %1152, i64 4, !dbg !13
  %1288 = extractelement <8 x float> %1152, i64 5, !dbg !13
  %1289 = extractelement <8 x float> %1152, i64 6, !dbg !13
  %1290 = extractelement <8 x float> %1152, i64 7, !dbg !13
  %1291 = select i1 %.not2, float %1288, float %1287, !dbg !13
  %1292 = bitcast float %1291 to i32, !dbg !13
  %1293 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1292), !dbg !13
  %1294 = bitcast i32 %1293 to float, !dbg !13
  %1295 = select i1 %.not2, float %1287, float %1294, !dbg !13
  %1296 = select i1 %.not2, float %1294, float %1288, !dbg !13
  %1297 = select i1 %.not2, float %1290, float %1289, !dbg !13
  %1298 = bitcast float %1297 to i32, !dbg !13
  %1299 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1298), !dbg !13
  %1300 = bitcast i32 %1299 to float, !dbg !13
  %1301 = select i1 %.not2, float %1289, float %1300, !dbg !13
  %1302 = select i1 %.not2, float %1300, float %1290, !dbg !13
  %1303 = select i1 %.not3, float %1301, float %1295, !dbg !13
  %1304 = bitcast float %1303 to i32, !dbg !13
  %1305 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1304), !dbg !13
  %1306 = bitcast i32 %1305 to float, !dbg !13
  %1307 = select i1 %.not3, float %1302, float %1296, !dbg !13
  %1308 = bitcast float %1307 to i32, !dbg !13
  %1309 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1308), !dbg !13
  %1310 = bitcast i32 %1309 to float, !dbg !13
  %1311 = insertelement <4 x float> poison, float %1295, i64 0, !dbg !13
  %1312 = insertelement <4 x float> %1311, float %1296, i64 1, !dbg !13
  %1313 = insertelement <4 x float> %1312, float %1306, i64 2, !dbg !13
  %1314 = insertelement <4 x float> %1313, float %1310, i64 3, !dbg !13
  %1315 = insertelement <4 x float> poison, float %1306, i64 0, !dbg !13
  %1316 = insertelement <4 x float> %1315, float %1310, i64 1, !dbg !13
  %1317 = insertelement <4 x float> %1316, float %1301, i64 2, !dbg !13
  %1318 = insertelement <4 x float> %1317, float %1302, i64 3, !dbg !13
  %1319 = select <4 x i1> %1184, <4 x float> %1314, <4 x float> %1318, !dbg !13
  %1320 = extractelement <8 x float> %1194, i64 4, !dbg !13
  %1321 = extractelement <8 x float> %1194, i64 5, !dbg !13
  %1322 = extractelement <8 x float> %1194, i64 6, !dbg !13
  %1323 = extractelement <8 x float> %1194, i64 7, !dbg !13
  %1324 = select i1 %.not2, float %1321, float %1320, !dbg !13
  %1325 = bitcast float %1324 to i32, !dbg !13
  %1326 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1325), !dbg !13
  %1327 = bitcast i32 %1326 to float, !dbg !13
  %1328 = select i1 %.not2, float %1320, float %1327, !dbg !13
  %1329 = select i1 %.not2, float %1327, float %1321, !dbg !13
  %1330 = select i1 %.not2, float %1323, float %1322, !dbg !13
  %1331 = bitcast float %1330 to i32, !dbg !13
  %1332 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1331), !dbg !13
  %1333 = bitcast i32 %1332 to float, !dbg !13
  %1334 = select i1 %.not2, float %1322, float %1333, !dbg !13
  %1335 = select i1 %.not2, float %1333, float %1323, !dbg !13
  %1336 = select i1 %.not3, float %1334, float %1328, !dbg !13
  %1337 = bitcast float %1336 to i32, !dbg !13
  %1338 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1337), !dbg !13
  %1339 = bitcast i32 %1338 to float, !dbg !13
  %1340 = select i1 %.not3, float %1335, float %1329, !dbg !13
  %1341 = bitcast float %1340 to i32, !dbg !13
  %1342 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1341), !dbg !13
  %1343 = bitcast i32 %1342 to float, !dbg !13
  %1344 = insertelement <4 x float> poison, float %1328, i64 0, !dbg !13
  %1345 = insertelement <4 x float> %1344, float %1329, i64 1, !dbg !13
  %1346 = insertelement <4 x float> %1345, float %1339, i64 2, !dbg !13
  %1347 = insertelement <4 x float> %1346, float %1343, i64 3, !dbg !13
  %1348 = insertelement <4 x float> poison, float %1339, i64 0, !dbg !13
  %1349 = insertelement <4 x float> %1348, float %1343, i64 1, !dbg !13
  %1350 = insertelement <4 x float> %1349, float %1334, i64 2, !dbg !13
  %1351 = insertelement <4 x float> %1350, float %1335, i64 3, !dbg !13
  %1352 = select <4 x i1> %1184, <4 x float> %1347, <4 x float> %1351, !dbg !13
  %1353 = extractelement <8 x float> %1228, i64 4, !dbg !13
  %1354 = extractelement <8 x float> %1228, i64 5, !dbg !13
  %1355 = extractelement <8 x float> %1228, i64 6, !dbg !13
  %1356 = extractelement <8 x float> %1228, i64 7, !dbg !13
  %1357 = select i1 %.not2, float %1354, float %1353, !dbg !13
  %1358 = bitcast float %1357 to i32, !dbg !13
  %1359 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1358), !dbg !13
  %1360 = bitcast i32 %1359 to float, !dbg !13
  %1361 = select i1 %.not2, float %1353, float %1360, !dbg !13
  %1362 = select i1 %.not2, float %1360, float %1354, !dbg !13
  %1363 = select i1 %.not2, float %1356, float %1355, !dbg !13
  %1364 = bitcast float %1363 to i32, !dbg !13
  %1365 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1364), !dbg !13
  %1366 = bitcast i32 %1365 to float, !dbg !13
  %1367 = select i1 %.not2, float %1355, float %1366, !dbg !13
  %1368 = select i1 %.not2, float %1366, float %1356, !dbg !13
  %1369 = select i1 %.not3, float %1367, float %1361, !dbg !13
  %1370 = bitcast float %1369 to i32, !dbg !13
  %1371 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1370), !dbg !13
  %1372 = bitcast i32 %1371 to float, !dbg !13
  %1373 = select i1 %.not3, float %1368, float %1362, !dbg !13
  %1374 = bitcast float %1373 to i32, !dbg !13
  %1375 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1374), !dbg !13
  %1376 = bitcast i32 %1375 to float, !dbg !13
  %1377 = insertelement <4 x float> poison, float %1361, i64 0, !dbg !13
  %1378 = insertelement <4 x float> %1377, float %1362, i64 1, !dbg !13
  %1379 = insertelement <4 x float> %1378, float %1372, i64 2, !dbg !13
  %1380 = insertelement <4 x float> %1379, float %1376, i64 3, !dbg !13
  %1381 = insertelement <4 x float> poison, float %1372, i64 0, !dbg !13
  %1382 = insertelement <4 x float> %1381, float %1376, i64 1, !dbg !13
  %1383 = insertelement <4 x float> %1382, float %1367, i64 2, !dbg !13
  %1384 = insertelement <4 x float> %1383, float %1368, i64 3, !dbg !13
  %1385 = select <4 x i1> %1184, <4 x float> %1380, <4 x float> %1384, !dbg !13
  %1386 = extractelement <8 x float> %1262, i64 4, !dbg !13
  %1387 = extractelement <8 x float> %1262, i64 5, !dbg !13
  %1388 = extractelement <8 x float> %1262, i64 6, !dbg !13
  %1389 = extractelement <8 x float> %1262, i64 7, !dbg !13
  %1390 = select i1 %.not2, float %1387, float %1386, !dbg !13
  %1391 = bitcast float %1390 to i32, !dbg !13
  %1392 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1391), !dbg !13
  %1393 = bitcast i32 %1392 to float, !dbg !13
  %1394 = select i1 %.not2, float %1386, float %1393, !dbg !13
  %1395 = select i1 %.not2, float %1393, float %1387, !dbg !13
  %1396 = select i1 %.not2, float %1389, float %1388, !dbg !13
  %1397 = bitcast float %1396 to i32, !dbg !13
  %1398 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1397), !dbg !13
  %1399 = bitcast i32 %1398 to float, !dbg !13
  %1400 = select i1 %.not2, float %1388, float %1399, !dbg !13
  %1401 = select i1 %.not2, float %1399, float %1389, !dbg !13
  %1402 = select i1 %.not3, float %1400, float %1394, !dbg !13
  %1403 = bitcast float %1402 to i32, !dbg !13
  %1404 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1403), !dbg !13
  %1405 = select i1 %.not3, float %1401, float %1395, !dbg !13
  %1406 = bitcast float %1405 to i32, !dbg !13
  %1407 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1406), !dbg !13
  %1408 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !13
  %1409 = extractelement <8 x float> %1408, i64 0, !dbg !13
  %1410 = extractelement <8 x float> %1408, i64 1, !dbg !13
  %1411 = extractelement <8 x float> %1408, i64 2, !dbg !13
  %1412 = extractelement <8 x float> %1408, i64 3, !dbg !13
  %1413 = select i1 %.not2, float %1410, float %1409, !dbg !13
  %1414 = bitcast float %1413 to i32, !dbg !13
  %1415 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1414), !dbg !13
  %1416 = bitcast i32 %1415 to float, !dbg !13
  %1417 = select i1 %.not2, float %1409, float %1416, !dbg !13
  %1418 = select i1 %.not2, float %1416, float %1410, !dbg !13
  %1419 = select i1 %.not2, float %1412, float %1411, !dbg !13
  %1420 = bitcast float %1419 to i32, !dbg !13
  %1421 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1420), !dbg !13
  %1422 = bitcast i32 %1421 to float, !dbg !13
  %1423 = select i1 %.not2, float %1411, float %1422, !dbg !13
  %1424 = select i1 %.not2, float %1422, float %1412, !dbg !13
  %1425 = select i1 %.not3, float %1423, float %1417, !dbg !13
  %1426 = bitcast float %1425 to i32, !dbg !13
  %1427 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1426), !dbg !13
  %1428 = bitcast i32 %1427 to float, !dbg !13
  %1429 = select i1 %.not3, float %1424, float %1418, !dbg !13
  %1430 = bitcast float %1429 to i32, !dbg !13
  %1431 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1430), !dbg !13
  %1432 = bitcast i32 %1431 to float, !dbg !13
  %1433 = insertelement <4 x float> poison, float %1417, i64 0, !dbg !13
  %1434 = insertelement <4 x float> %1433, float %1418, i64 1, !dbg !13
  %1435 = insertelement <4 x float> %1434, float %1428, i64 2, !dbg !13
  %1436 = insertelement <4 x float> %1435, float %1432, i64 3, !dbg !13
  %1437 = insertelement <4 x float> poison, float %1428, i64 0, !dbg !13
  %1438 = insertelement <4 x float> %1437, float %1432, i64 1, !dbg !13
  %1439 = insertelement <4 x float> %1438, float %1423, i64 2, !dbg !13
  %1440 = insertelement <4 x float> %1439, float %1424, i64 3, !dbg !13
  %1441 = select <4 x i1> %1184, <4 x float> %1436, <4 x float> %1440, !dbg !13
  %1442 = extractvalue [2 x <8 x float>] %.lcssa90, 1, !dbg !13
  %1443 = extractelement <8 x float> %1442, i64 0, !dbg !13
  %1444 = extractelement <8 x float> %1442, i64 1, !dbg !13
  %1445 = extractelement <8 x float> %1442, i64 2, !dbg !13
  %1446 = extractelement <8 x float> %1442, i64 3, !dbg !13
  %1447 = select i1 %.not2, float %1444, float %1443, !dbg !13
  %1448 = bitcast float %1447 to i32, !dbg !13
  %1449 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1448), !dbg !13
  %1450 = bitcast i32 %1449 to float, !dbg !13
  %1451 = select i1 %.not2, float %1443, float %1450, !dbg !13
  %1452 = select i1 %.not2, float %1450, float %1444, !dbg !13
  %1453 = select i1 %.not2, float %1446, float %1445, !dbg !13
  %1454 = bitcast float %1453 to i32, !dbg !13
  %1455 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1454), !dbg !13
  %1456 = bitcast i32 %1455 to float, !dbg !13
  %1457 = select i1 %.not2, float %1445, float %1456, !dbg !13
  %1458 = select i1 %.not2, float %1456, float %1446, !dbg !13
  %1459 = select i1 %.not3, float %1457, float %1451, !dbg !13
  %1460 = bitcast float %1459 to i32, !dbg !13
  %1461 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1460), !dbg !13
  %1462 = bitcast i32 %1461 to float, !dbg !13
  %1463 = select i1 %.not3, float %1458, float %1452, !dbg !13
  %1464 = bitcast float %1463 to i32, !dbg !13
  %1465 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1464), !dbg !13
  %1466 = bitcast i32 %1465 to float, !dbg !13
  %1467 = insertelement <4 x float> poison, float %1451, i64 0, !dbg !13
  %1468 = insertelement <4 x float> %1467, float %1452, i64 1, !dbg !13
  %1469 = insertelement <4 x float> %1468, float %1462, i64 2, !dbg !13
  %1470 = insertelement <4 x float> %1469, float %1466, i64 3, !dbg !13
  %1471 = insertelement <4 x float> poison, float %1462, i64 0, !dbg !13
  %1472 = insertelement <4 x float> %1471, float %1466, i64 1, !dbg !13
  %1473 = insertelement <4 x float> %1472, float %1457, i64 2, !dbg !13
  %1474 = insertelement <4 x float> %1473, float %1458, i64 3, !dbg !13
  %1475 = select <4 x i1> %1184, <4 x float> %1470, <4 x float> %1474, !dbg !13
  %1476 = extractvalue [2 x <8 x float>] %.lcssa91, 1, !dbg !13
  %1477 = extractelement <8 x float> %1476, i64 0, !dbg !13
  %1478 = extractelement <8 x float> %1476, i64 1, !dbg !13
  %1479 = extractelement <8 x float> %1476, i64 2, !dbg !13
  %1480 = extractelement <8 x float> %1476, i64 3, !dbg !13
  %1481 = select i1 %.not2, float %1478, float %1477, !dbg !13
  %1482 = bitcast float %1481 to i32, !dbg !13
  %1483 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1482), !dbg !13
  %1484 = bitcast i32 %1483 to float, !dbg !13
  %1485 = select i1 %.not2, float %1477, float %1484, !dbg !13
  %1486 = select i1 %.not2, float %1484, float %1478, !dbg !13
  %1487 = select i1 %.not2, float %1480, float %1479, !dbg !13
  %1488 = bitcast float %1487 to i32, !dbg !13
  %1489 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1488), !dbg !13
  %1490 = bitcast i32 %1489 to float, !dbg !13
  %1491 = select i1 %.not2, float %1479, float %1490, !dbg !13
  %1492 = select i1 %.not2, float %1490, float %1480, !dbg !13
  %1493 = select i1 %.not3, float %1491, float %1485, !dbg !13
  %1494 = bitcast float %1493 to i32, !dbg !13
  %1495 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1494), !dbg !13
  %1496 = bitcast i32 %1495 to float, !dbg !13
  %1497 = select i1 %.not3, float %1492, float %1486, !dbg !13
  %1498 = bitcast float %1497 to i32, !dbg !13
  %1499 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1498), !dbg !13
  %1500 = bitcast i32 %1499 to float, !dbg !13
  %1501 = insertelement <4 x float> poison, float %1485, i64 0, !dbg !13
  %1502 = insertelement <4 x float> %1501, float %1486, i64 1, !dbg !13
  %1503 = insertelement <4 x float> %1502, float %1496, i64 2, !dbg !13
  %1504 = insertelement <4 x float> %1503, float %1500, i64 3, !dbg !13
  %1505 = insertelement <4 x float> poison, float %1496, i64 0, !dbg !13
  %1506 = insertelement <4 x float> %1505, float %1500, i64 1, !dbg !13
  %1507 = insertelement <4 x float> %1506, float %1491, i64 2, !dbg !13
  %1508 = insertelement <4 x float> %1507, float %1492, i64 3, !dbg !13
  %1509 = select <4 x i1> %1184, <4 x float> %1504, <4 x float> %1508, !dbg !13
  %1510 = extractvalue [2 x <8 x float>] %.lcssa93, 1, !dbg !13
  %1511 = extractelement <8 x float> %1510, i64 0, !dbg !13
  %1512 = extractelement <8 x float> %1510, i64 1, !dbg !13
  %1513 = extractelement <8 x float> %1510, i64 2, !dbg !13
  %1514 = extractelement <8 x float> %1510, i64 3, !dbg !13
  %1515 = select i1 %.not2, float %1512, float %1511, !dbg !13
  %1516 = bitcast float %1515 to i32, !dbg !13
  %1517 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1516), !dbg !13
  %1518 = bitcast i32 %1517 to float, !dbg !13
  %1519 = select i1 %.not2, float %1511, float %1518, !dbg !13
  %1520 = select i1 %.not2, float %1518, float %1512, !dbg !13
  %1521 = select i1 %.not2, float %1514, float %1513, !dbg !13
  %1522 = bitcast float %1521 to i32, !dbg !13
  %1523 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1522), !dbg !13
  %1524 = bitcast i32 %1523 to float, !dbg !13
  %1525 = select i1 %.not2, float %1513, float %1524, !dbg !13
  %1526 = select i1 %.not2, float %1524, float %1514, !dbg !13
  %1527 = select i1 %.not3, float %1525, float %1519, !dbg !13
  %1528 = bitcast float %1527 to i32, !dbg !13
  %1529 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1528), !dbg !13
  %1530 = select i1 %.not3, float %1526, float %1520, !dbg !13
  %1531 = bitcast float %1530 to i32, !dbg !13
  %1532 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1531), !dbg !13
  %1533 = extractvalue [2 x <1 x float>] %.lcssa92, 1, !dbg !25
  %1534 = shufflevector <1 x float> %1533, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !25
  %1535 = extractelement <8 x float> %1408, i64 4, !dbg !13
  %1536 = extractelement <8 x float> %1408, i64 5, !dbg !13
  %1537 = extractelement <8 x float> %1408, i64 6, !dbg !13
  %1538 = extractelement <8 x float> %1408, i64 7, !dbg !13
  %1539 = select i1 %.not2, float %1536, float %1535, !dbg !13
  %1540 = bitcast float %1539 to i32, !dbg !13
  %1541 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1540), !dbg !13
  %1542 = bitcast i32 %1541 to float, !dbg !13
  %1543 = select i1 %.not2, float %1535, float %1542, !dbg !13
  %1544 = select i1 %.not2, float %1542, float %1536, !dbg !13
  %1545 = select i1 %.not2, float %1538, float %1537, !dbg !13
  %1546 = bitcast float %1545 to i32, !dbg !13
  %1547 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1546), !dbg !13
  %1548 = bitcast i32 %1547 to float, !dbg !13
  %1549 = select i1 %.not2, float %1537, float %1548, !dbg !13
  %1550 = select i1 %.not2, float %1548, float %1538, !dbg !13
  %1551 = select i1 %.not3, float %1549, float %1543, !dbg !13
  %1552 = bitcast float %1551 to i32, !dbg !13
  %1553 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1552), !dbg !13
  %1554 = bitcast i32 %1553 to float, !dbg !13
  %1555 = select i1 %.not3, float %1550, float %1544, !dbg !13
  %1556 = bitcast float %1555 to i32, !dbg !13
  %1557 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1556), !dbg !13
  %1558 = bitcast i32 %1557 to float, !dbg !13
  %1559 = insertelement <4 x float> poison, float %1543, i64 0, !dbg !13
  %1560 = insertelement <4 x float> %1559, float %1544, i64 1, !dbg !13
  %1561 = insertelement <4 x float> %1560, float %1554, i64 2, !dbg !13
  %1562 = insertelement <4 x float> %1561, float %1558, i64 3, !dbg !13
  %1563 = insertelement <4 x float> poison, float %1554, i64 0, !dbg !13
  %1564 = insertelement <4 x float> %1563, float %1558, i64 1, !dbg !13
  %1565 = insertelement <4 x float> %1564, float %1549, i64 2, !dbg !13
  %1566 = insertelement <4 x float> %1565, float %1550, i64 3, !dbg !13
  %1567 = select <4 x i1> %1184, <4 x float> %1562, <4 x float> %1566, !dbg !13
  %1568 = extractelement <8 x float> %1442, i64 4, !dbg !13
  %1569 = extractelement <8 x float> %1442, i64 5, !dbg !13
  %1570 = extractelement <8 x float> %1442, i64 6, !dbg !13
  %1571 = extractelement <8 x float> %1442, i64 7, !dbg !13
  %1572 = select i1 %.not2, float %1569, float %1568, !dbg !13
  %1573 = bitcast float %1572 to i32, !dbg !13
  %1574 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1573), !dbg !13
  %1575 = bitcast i32 %1574 to float, !dbg !13
  %1576 = select i1 %.not2, float %1568, float %1575, !dbg !13
  %1577 = select i1 %.not2, float %1575, float %1569, !dbg !13
  %1578 = select i1 %.not2, float %1571, float %1570, !dbg !13
  %1579 = bitcast float %1578 to i32, !dbg !13
  %1580 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1579), !dbg !13
  %1581 = bitcast i32 %1580 to float, !dbg !13
  %1582 = select i1 %.not2, float %1570, float %1581, !dbg !13
  %1583 = select i1 %.not2, float %1581, float %1571, !dbg !13
  %1584 = select i1 %.not3, float %1582, float %1576, !dbg !13
  %1585 = bitcast float %1584 to i32, !dbg !13
  %1586 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1585), !dbg !13
  %1587 = bitcast i32 %1586 to float, !dbg !13
  %1588 = select i1 %.not3, float %1583, float %1577, !dbg !13
  %1589 = bitcast float %1588 to i32, !dbg !13
  %1590 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1589), !dbg !13
  %1591 = bitcast i32 %1590 to float, !dbg !13
  %1592 = insertelement <4 x float> poison, float %1576, i64 0, !dbg !13
  %1593 = insertelement <4 x float> %1592, float %1577, i64 1, !dbg !13
  %1594 = insertelement <4 x float> %1593, float %1587, i64 2, !dbg !13
  %1595 = insertelement <4 x float> %1594, float %1591, i64 3, !dbg !13
  %1596 = insertelement <4 x float> poison, float %1587, i64 0, !dbg !13
  %1597 = insertelement <4 x float> %1596, float %1591, i64 1, !dbg !13
  %1598 = insertelement <4 x float> %1597, float %1582, i64 2, !dbg !13
  %1599 = insertelement <4 x float> %1598, float %1583, i64 3, !dbg !13
  %1600 = select <4 x i1> %1184, <4 x float> %1595, <4 x float> %1599, !dbg !13
  %1601 = extractelement <8 x float> %1476, i64 4, !dbg !13
  %1602 = extractelement <8 x float> %1476, i64 5, !dbg !13
  %1603 = extractelement <8 x float> %1476, i64 6, !dbg !13
  %1604 = extractelement <8 x float> %1476, i64 7, !dbg !13
  %1605 = select i1 %.not2, float %1602, float %1601, !dbg !13
  %1606 = bitcast float %1605 to i32, !dbg !13
  %1607 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1606), !dbg !13
  %1608 = bitcast i32 %1607 to float, !dbg !13
  %1609 = select i1 %.not2, float %1601, float %1608, !dbg !13
  %1610 = select i1 %.not2, float %1608, float %1602, !dbg !13
  %1611 = select i1 %.not2, float %1604, float %1603, !dbg !13
  %1612 = bitcast float %1611 to i32, !dbg !13
  %1613 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1612), !dbg !13
  %1614 = bitcast i32 %1613 to float, !dbg !13
  %1615 = select i1 %.not2, float %1603, float %1614, !dbg !13
  %1616 = select i1 %.not2, float %1614, float %1604, !dbg !13
  %1617 = select i1 %.not3, float %1615, float %1609, !dbg !13
  %1618 = bitcast float %1617 to i32, !dbg !13
  %1619 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1618), !dbg !13
  %1620 = bitcast i32 %1619 to float, !dbg !13
  %1621 = select i1 %.not3, float %1616, float %1610, !dbg !13
  %1622 = bitcast float %1621 to i32, !dbg !13
  %1623 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1622), !dbg !13
  %1624 = bitcast i32 %1623 to float, !dbg !13
  %1625 = insertelement <4 x float> poison, float %1609, i64 0, !dbg !13
  %1626 = insertelement <4 x float> %1625, float %1610, i64 1, !dbg !13
  %1627 = insertelement <4 x float> %1626, float %1620, i64 2, !dbg !13
  %1628 = insertelement <4 x float> %1627, float %1624, i64 3, !dbg !13
  %1629 = insertelement <4 x float> poison, float %1620, i64 0, !dbg !13
  %1630 = insertelement <4 x float> %1629, float %1624, i64 1, !dbg !13
  %1631 = insertelement <4 x float> %1630, float %1615, i64 2, !dbg !13
  %1632 = insertelement <4 x float> %1631, float %1616, i64 3, !dbg !13
  %1633 = select <4 x i1> %1184, <4 x float> %1628, <4 x float> %1632, !dbg !13
  %1634 = extractelement <8 x float> %1510, i64 4, !dbg !13
  %1635 = extractelement <8 x float> %1510, i64 5, !dbg !13
  %1636 = extractelement <8 x float> %1510, i64 6, !dbg !13
  %1637 = extractelement <8 x float> %1510, i64 7, !dbg !13
  %1638 = select i1 %.not2, float %1635, float %1634, !dbg !13
  %1639 = bitcast float %1638 to i32, !dbg !13
  %1640 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1639), !dbg !13
  %1641 = bitcast i32 %1640 to float, !dbg !13
  %1642 = select i1 %.not2, float %1634, float %1641, !dbg !13
  %1643 = select i1 %.not2, float %1641, float %1635, !dbg !13
  %1644 = select i1 %.not2, float %1637, float %1636, !dbg !13
  %1645 = bitcast float %1644 to i32, !dbg !13
  %1646 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1645), !dbg !13
  %1647 = bitcast i32 %1646 to float, !dbg !13
  %1648 = select i1 %.not2, float %1636, float %1647, !dbg !13
  %1649 = select i1 %.not2, float %1647, float %1637, !dbg !13
  %1650 = select i1 %.not3, float %1648, float %1642, !dbg !13
  %1651 = bitcast float %1650 to i32, !dbg !13
  %1652 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1651), !dbg !13
  %1653 = select i1 %.not3, float %1649, float %1643, !dbg !13
  %1654 = bitcast float %1653 to i32, !dbg !13
  %1655 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1175, i32 %1654), !dbg !13
  %1656 = bitcast i32 %1652 to float, !dbg !13
  %1657 = bitcast i32 %1655 to float, !dbg !13
  %1658 = insertelement <4 x float> poison, float %1642, i64 0, !dbg !13
  %1659 = insertelement <4 x float> %1658, float %1643, i64 1, !dbg !13
  %1660 = insertelement <4 x float> %1659, float %1656, i64 2, !dbg !13
  %1661 = insertelement <4 x float> %1660, float %1657, i64 3, !dbg !13
  %1662 = insertelement <4 x float> poison, float %1656, i64 0, !dbg !13
  %1663 = insertelement <4 x float> %1662, float %1657, i64 1, !dbg !13
  %1664 = insertelement <4 x float> %1663, float %1648, i64 2, !dbg !13
  %1665 = insertelement <4 x float> %1664, float %1649, i64 3, !dbg !13
  %1666 = select <4 x i1> %1184, <4 x float> %1661, <4 x float> %1665, !dbg !13
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1534, !dbg !25
  %normalized = fmul <4 x float> %row.reciprocal, %1666, !dbg !25
  %1667 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !26
  %1668 = extractelement <4 x half> %1667, i64 3, !dbg !26
  %1669 = extractelement <4 x half> %1667, i64 2, !dbg !26
  %1670 = extractelement <4 x half> %1667, i64 1, !dbg !26
  %1671 = extractelement <4 x half> %1667, i64 0, !dbg !26
  %1672 = bitcast i32 %1529 to float, !dbg !13
  %1673 = bitcast i32 %1532 to float, !dbg !13
  %1674 = insertelement <4 x float> poison, float %1519, i64 0, !dbg !13
  %1675 = insertelement <4 x float> %1674, float %1520, i64 1, !dbg !13
  %1676 = insertelement <4 x float> %1675, float %1672, i64 2, !dbg !13
  %1677 = insertelement <4 x float> %1676, float %1673, i64 3, !dbg !13
  %1678 = insertelement <4 x float> poison, float %1672, i64 0, !dbg !13
  %1679 = insertelement <4 x float> %1678, float %1673, i64 1, !dbg !13
  %1680 = insertelement <4 x float> %1679, float %1525, i64 2, !dbg !13
  %1681 = insertelement <4 x float> %1680, float %1526, i64 3, !dbg !13
  %1682 = select <4 x i1> %1184, <4 x float> %1677, <4 x float> %1681, !dbg !13
  %normalized71 = fmul <4 x float> %row.reciprocal, %1682, !dbg !25
  %1683 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !26
  %1684 = extractelement <4 x half> %1683, i64 3, !dbg !26
  %1685 = extractelement <4 x half> %1683, i64 2, !dbg !26
  %1686 = extractelement <4 x half> %1683, i64 1, !dbg !26
  %1687 = extractelement <4 x half> %1683, i64 0, !dbg !26
  %1688 = bitcast i32 %1404 to float, !dbg !13
  %1689 = bitcast i32 %1407 to float, !dbg !13
  %1690 = insertelement <4 x float> poison, float %1394, i64 0, !dbg !13
  %1691 = insertelement <4 x float> %1690, float %1395, i64 1, !dbg !13
  %1692 = insertelement <4 x float> %1691, float %1688, i64 2, !dbg !13
  %1693 = insertelement <4 x float> %1692, float %1689, i64 3, !dbg !13
  %1694 = insertelement <4 x float> poison, float %1688, i64 0, !dbg !13
  %1695 = insertelement <4 x float> %1694, float %1689, i64 1, !dbg !13
  %1696 = insertelement <4 x float> %1695, float %1400, i64 2, !dbg !13
  %1697 = insertelement <4 x float> %1696, float %1401, i64 3, !dbg !13
  %1698 = select <4 x i1> %1184, <4 x float> %1693, <4 x float> %1697, !dbg !13
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1286, !dbg !25
  %normalized73 = fmul <4 x float> %row.reciprocal72, %1698, !dbg !25
  %1699 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !26
  %1700 = extractelement <4 x half> %1699, i64 3, !dbg !26
  %1701 = extractelement <4 x half> %1699, i64 2, !dbg !26
  %1702 = extractelement <4 x half> %1699, i64 1, !dbg !26
  %1703 = extractelement <4 x half> %1699, i64 0, !dbg !26
  %1704 = bitcast i32 %1281 to float, !dbg !13
  %1705 = bitcast i32 %1284 to float, !dbg !13
  %1706 = insertelement <4 x float> poison, float %1271, i64 0, !dbg !13
  %1707 = insertelement <4 x float> %1706, float %1272, i64 1, !dbg !13
  %1708 = insertelement <4 x float> %1707, float %1704, i64 2, !dbg !13
  %1709 = insertelement <4 x float> %1708, float %1705, i64 3, !dbg !13
  %1710 = insertelement <4 x float> poison, float %1704, i64 0, !dbg !13
  %1711 = insertelement <4 x float> %1710, float %1705, i64 1, !dbg !13
  %1712 = insertelement <4 x float> %1711, float %1277, i64 2, !dbg !13
  %1713 = insertelement <4 x float> %1712, float %1278, i64 3, !dbg !13
  %1714 = select <4 x i1> %1184, <4 x float> %1709, <4 x float> %1713, !dbg !13
  %normalized74 = fmul <4 x float> %row.reciprocal72, %1714, !dbg !25
  %1715 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !26
  %1716 = extractelement <4 x half> %1715, i64 3, !dbg !26
  %1717 = extractelement <4 x half> %1715, i64 2, !dbg !26
  %1718 = extractelement <4 x half> %1715, i64 1, !dbg !26
  %1719 = extractelement <4 x half> %1715, i64 0, !dbg !26
  %address.thread389 = tail call i32 asm sideeffect "", "=v,0"(i32 %21) #8, !dbg !27, !range !11, !frisk.register_lifetimes !5
  %address.local390 = zext nneg i32 %address.thread389 to i64, !dbg !10
  %1720 = lshr i64 %address.local390, 2, !dbg !27
  %address.local393 = and i64 %1720, 48, !dbg !27
  %address.local395 = and i64 %address.local390, 240, !dbg !27
  %address.local394 = add nsw i64 %18, %address.local390, !dbg !27
  %address.local396 = sub nsw i64 %address.local394, %address.local395, !dbg !27
  %address.local397 = add nsw i64 %address.local396, %address.local393, !dbg !27
  %1721 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %20, !dbg !27
  %address.local400.idx = shl nuw nsw i64 %address.local397, 8, !dbg !27
  %address.local400 = getelementptr inbounds nuw i8, ptr addrspace(1) %1721, i64 %address.local400.idx, !dbg !27
  %address.local402 = and i64 %1720, 12, !dbg !10
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local402, !dbg !27
  store half %1719, ptr addrspace(1) %address.local403, align 2, !dbg !27
  %address.local405 = or disjoint i64 %address.local402, 1, !dbg !27
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local405, !dbg !27
  store half %1718, ptr addrspace(1) %address.local406, align 2, !dbg !27
  %address.local408 = or disjoint i64 %address.local402, 2, !dbg !27
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local408, !dbg !27
  store half %1717, ptr addrspace(1) %address.local409, align 2, !dbg !27
  %address.local411 = or disjoint i64 %address.local402, 3, !dbg !27
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local411, !dbg !27
  store half %1716, ptr addrspace(1) %address.local412, align 2, !dbg !27
  %address.local414 = or disjoint i64 %address.local402, 16, !dbg !27
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local414, !dbg !27
  store half %1703, ptr addrspace(1) %address.local415, align 2, !dbg !27
  %address.local417 = or disjoint i64 %address.local402, 17, !dbg !27
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local417, !dbg !27
  store half %1702, ptr addrspace(1) %address.local418, align 2, !dbg !27
  %address.local420 = or disjoint i64 %address.local402, 18, !dbg !27
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local420, !dbg !27
  store half %1701, ptr addrspace(1) %address.local421, align 2, !dbg !27
  %address.local423 = or disjoint i64 %address.local402, 19, !dbg !27
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local423, !dbg !27
  store half %1700, ptr addrspace(1) %address.local424, align 2, !dbg !27
  %address.local428 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local400, i64 16384, !dbg !27
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local402, !dbg !27
  store half %1687, ptr addrspace(1) %address.local429, align 2, !dbg !27
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local405, !dbg !27
  store half %1686, ptr addrspace(1) %address.local431, align 2, !dbg !27
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local408, !dbg !27
  store half %1685, ptr addrspace(1) %address.local433, align 2, !dbg !27
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local411, !dbg !27
  store half %1684, ptr addrspace(1) %address.local435, align 2, !dbg !27
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local414, !dbg !27
  store half %1671, ptr addrspace(1) %address.local437, align 2, !dbg !27
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local417, !dbg !27
  store half %1670, ptr addrspace(1) %address.local439, align 2, !dbg !27
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local420, !dbg !27
  store half %1669, ptr addrspace(1) %address.local441, align 2, !dbg !27
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local423, !dbg !27
  store half %1668, ptr addrspace(1) %address.local443, align 2, !dbg !27
  %normalized76 = fmul <4 x float> %row.reciprocal, %1633, !dbg !25
  %1722 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !26
  %1723 = extractelement <4 x half> %1722, i64 3, !dbg !26
  %1724 = extractelement <4 x half> %1722, i64 2, !dbg !26
  %1725 = extractelement <4 x half> %1722, i64 1, !dbg !26
  %1726 = extractelement <4 x half> %1722, i64 0, !dbg !26
  %normalized77 = fmul <4 x float> %row.reciprocal, %1509, !dbg !25
  %1727 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !26
  %1728 = extractelement <4 x half> %1727, i64 3, !dbg !26
  %1729 = extractelement <4 x half> %1727, i64 2, !dbg !26
  %1730 = extractelement <4 x half> %1727, i64 1, !dbg !26
  %1731 = extractelement <4 x half> %1727, i64 0, !dbg !26
  %normalized79 = fmul <4 x float> %row.reciprocal72, %1385, !dbg !25
  %1732 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !26
  %1733 = extractelement <4 x half> %1732, i64 3, !dbg !26
  %1734 = extractelement <4 x half> %1732, i64 2, !dbg !26
  %1735 = extractelement <4 x half> %1732, i64 1, !dbg !26
  %1736 = extractelement <4 x half> %1732, i64 0, !dbg !26
  %normalized80 = fmul <4 x float> %row.reciprocal72, %1261, !dbg !25
  %1737 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !26
  %1738 = extractelement <4 x half> %1737, i64 3, !dbg !26
  %1739 = extractelement <4 x half> %1737, i64 2, !dbg !26
  %1740 = extractelement <4 x half> %1737, i64 1, !dbg !26
  %1741 = extractelement <4 x half> %1737, i64 0, !dbg !26
  %address.local458 = or disjoint i64 %address.local402, 32, !dbg !27
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local458, !dbg !27
  store half %1741, ptr addrspace(1) %address.local459, align 2, !dbg !27
  %address.local461 = or disjoint i64 %address.local402, 33, !dbg !27
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local461, !dbg !27
  store half %1740, ptr addrspace(1) %address.local462, align 2, !dbg !27
  %address.local464 = or disjoint i64 %address.local402, 34, !dbg !27
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local464, !dbg !27
  store half %1739, ptr addrspace(1) %address.local465, align 2, !dbg !27
  %address.local467 = or disjoint i64 %address.local402, 35, !dbg !27
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local467, !dbg !27
  store half %1738, ptr addrspace(1) %address.local468, align 2, !dbg !27
  %address.local470 = or disjoint i64 %address.local402, 48, !dbg !27
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local470, !dbg !27
  store half %1736, ptr addrspace(1) %address.local471, align 2, !dbg !27
  %address.local473 = or disjoint i64 %address.local402, 49, !dbg !27
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local473, !dbg !27
  store half %1735, ptr addrspace(1) %address.local474, align 2, !dbg !27
  %address.local476 = or disjoint i64 %address.local402, 50, !dbg !27
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local476, !dbg !27
  store half %1734, ptr addrspace(1) %address.local477, align 2, !dbg !27
  %address.local479 = or i64 %1720, 51, !dbg !27
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local479, !dbg !27
  store half %1733, ptr addrspace(1) %address.local480, align 2, !dbg !27
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local458, !dbg !27
  store half %1731, ptr addrspace(1) %address.local485, align 2, !dbg !27
  %address.local487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local461, !dbg !27
  store half %1730, ptr addrspace(1) %address.local487, align 2, !dbg !27
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local464, !dbg !27
  store half %1729, ptr addrspace(1) %address.local489, align 2, !dbg !27
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local467, !dbg !27
  store half %1728, ptr addrspace(1) %address.local491, align 2, !dbg !27
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local470, !dbg !27
  store half %1726, ptr addrspace(1) %address.local493, align 2, !dbg !27
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local473, !dbg !27
  store half %1725, ptr addrspace(1) %address.local495, align 2, !dbg !27
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local476, !dbg !27
  store half %1724, ptr addrspace(1) %address.local497, align 2, !dbg !27
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local479, !dbg !27
  store half %1723, ptr addrspace(1) %address.local499, align 2, !dbg !27
  %normalized82 = fmul <4 x float> %row.reciprocal, %1600, !dbg !25
  %1742 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !26
  %1743 = extractelement <4 x half> %1742, i64 3, !dbg !26
  %1744 = extractelement <4 x half> %1742, i64 2, !dbg !26
  %1745 = extractelement <4 x half> %1742, i64 1, !dbg !26
  %1746 = extractelement <4 x half> %1742, i64 0, !dbg !26
  %normalized83 = fmul <4 x float> %row.reciprocal, %1475, !dbg !25
  %1747 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !26
  %1748 = extractelement <4 x half> %1747, i64 3, !dbg !26
  %1749 = extractelement <4 x half> %1747, i64 2, !dbg !26
  %1750 = extractelement <4 x half> %1747, i64 1, !dbg !26
  %1751 = extractelement <4 x half> %1747, i64 0, !dbg !26
  %normalized85 = fmul <4 x float> %row.reciprocal72, %1352, !dbg !25
  %1752 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !26
  %1753 = extractelement <4 x half> %1752, i64 3, !dbg !26
  %1754 = extractelement <4 x half> %1752, i64 2, !dbg !26
  %1755 = extractelement <4 x half> %1752, i64 1, !dbg !26
  %1756 = extractelement <4 x half> %1752, i64 0, !dbg !26
  %normalized86 = fmul <4 x float> %row.reciprocal72, %1227, !dbg !25
  %1757 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !26
  %1758 = extractelement <4 x half> %1757, i64 3, !dbg !26
  %1759 = extractelement <4 x half> %1757, i64 2, !dbg !26
  %1760 = extractelement <4 x half> %1757, i64 1, !dbg !26
  %1761 = extractelement <4 x half> %1757, i64 0, !dbg !26
  %address.local514 = or disjoint i64 %address.local402, 64, !dbg !27
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local514, !dbg !27
  store half %1761, ptr addrspace(1) %address.local515, align 2, !dbg !27
  %address.local517 = or disjoint i64 %address.local402, 65, !dbg !27
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local517, !dbg !27
  store half %1760, ptr addrspace(1) %address.local518, align 2, !dbg !27
  %address.local520 = or disjoint i64 %address.local402, 66, !dbg !27
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local520, !dbg !27
  store half %1759, ptr addrspace(1) %address.local521, align 2, !dbg !27
  %address.local523 = or disjoint i64 %address.local402, 67, !dbg !27
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local523, !dbg !27
  store half %1758, ptr addrspace(1) %address.local524, align 2, !dbg !27
  %address.local526 = or disjoint i64 %address.local402, 80, !dbg !27
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local526, !dbg !27
  store half %1756, ptr addrspace(1) %address.local527, align 2, !dbg !27
  %address.local529 = or disjoint i64 %address.local402, 81, !dbg !27
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local529, !dbg !27
  store half %1755, ptr addrspace(1) %address.local530, align 2, !dbg !27
  %address.local532 = or disjoint i64 %address.local402, 82, !dbg !27
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local532, !dbg !27
  store half %1754, ptr addrspace(1) %address.local533, align 2, !dbg !27
  %address.local535 = or disjoint i64 %address.local402, 83, !dbg !27
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local535, !dbg !27
  store half %1753, ptr addrspace(1) %address.local536, align 2, !dbg !27
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local514, !dbg !27
  store half %1751, ptr addrspace(1) %address.local541, align 2, !dbg !27
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local517, !dbg !27
  store half %1750, ptr addrspace(1) %address.local543, align 2, !dbg !27
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local520, !dbg !27
  store half %1749, ptr addrspace(1) %address.local545, align 2, !dbg !27
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local523, !dbg !27
  store half %1748, ptr addrspace(1) %address.local547, align 2, !dbg !27
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local526, !dbg !27
  store half %1746, ptr addrspace(1) %address.local549, align 2, !dbg !27
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local529, !dbg !27
  store half %1745, ptr addrspace(1) %address.local551, align 2, !dbg !27
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local532, !dbg !27
  store half %1744, ptr addrspace(1) %address.local553, align 2, !dbg !27
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local535, !dbg !27
  store half %1743, ptr addrspace(1) %address.local555, align 2, !dbg !27
  %normalized88 = fmul <4 x float> %row.reciprocal, %1567, !dbg !25
  %1762 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !26
  %1763 = extractelement <4 x half> %1762, i64 3, !dbg !26
  %1764 = extractelement <4 x half> %1762, i64 2, !dbg !26
  %1765 = extractelement <4 x half> %1762, i64 1, !dbg !26
  %1766 = extractelement <4 x half> %1762, i64 0, !dbg !26
  %normalized89 = fmul <4 x float> %row.reciprocal, %1441, !dbg !25
  %1767 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !26
  %1768 = extractelement <4 x half> %1767, i64 3, !dbg !26
  %1769 = extractelement <4 x half> %1767, i64 2, !dbg !26
  %1770 = extractelement <4 x half> %1767, i64 1, !dbg !26
  %1771 = extractelement <4 x half> %1767, i64 0, !dbg !26
  %normalized91 = fmul <4 x float> %row.reciprocal72, %1319, !dbg !25
  %1772 = fptrunc <4 x float> %normalized91 to <4 x half>, !dbg !26
  %1773 = extractelement <4 x half> %1772, i64 3, !dbg !26
  %1774 = extractelement <4 x half> %1772, i64 2, !dbg !26
  %1775 = extractelement <4 x half> %1772, i64 1, !dbg !26
  %1776 = extractelement <4 x half> %1772, i64 0, !dbg !26
  %normalized92 = fmul <4 x float> %row.reciprocal72, %1193, !dbg !25
  %1777 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !26
  %1778 = extractelement <4 x half> %1777, i64 3, !dbg !26
  %1779 = extractelement <4 x half> %1777, i64 2, !dbg !26
  %1780 = extractelement <4 x half> %1777, i64 1, !dbg !26
  %1781 = extractelement <4 x half> %1777, i64 0, !dbg !26
  %address.local570 = or disjoint i64 %address.local402, 96, !dbg !27
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local570, !dbg !27
  store half %1781, ptr addrspace(1) %address.local571, align 2, !dbg !27
  %address.local573 = or disjoint i64 %address.local402, 97, !dbg !27
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local573, !dbg !27
  store half %1780, ptr addrspace(1) %address.local574, align 2, !dbg !27
  %address.local576 = or disjoint i64 %address.local402, 98, !dbg !27
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local576, !dbg !27
  store half %1779, ptr addrspace(1) %address.local577, align 2, !dbg !27
  %address.local579 = or disjoint i64 %address.local402, 99, !dbg !27
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local579, !dbg !27
  store half %1778, ptr addrspace(1) %address.local580, align 2, !dbg !27
  %address.local582 = or disjoint i64 %address.local402, 112, !dbg !27
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local582, !dbg !27
  store half %1776, ptr addrspace(1) %address.local583, align 2, !dbg !27
  %address.local585 = or disjoint i64 %address.local402, 113, !dbg !27
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local585, !dbg !27
  store half %1775, ptr addrspace(1) %address.local586, align 2, !dbg !27
  %address.local588 = or disjoint i64 %address.local402, 114, !dbg !27
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local588, !dbg !27
  store half %1774, ptr addrspace(1) %address.local589, align 2, !dbg !27
  %address.local591 = or i64 %1720, 115, !dbg !27
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local591, !dbg !27
  store half %1773, ptr addrspace(1) %address.local592, align 2, !dbg !27
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local570, !dbg !27
  store half %1771, ptr addrspace(1) %address.local597, align 2, !dbg !27
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local573, !dbg !27
  store half %1770, ptr addrspace(1) %address.local599, align 2, !dbg !27
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local576, !dbg !27
  store half %1769, ptr addrspace(1) %address.local601, align 2, !dbg !27
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local579, !dbg !27
  store half %1768, ptr addrspace(1) %address.local603, align 2, !dbg !27
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local582, !dbg !27
  store half %1766, ptr addrspace(1) %address.local605, align 2, !dbg !27
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local585, !dbg !27
  store half %1765, ptr addrspace(1) %address.local607, align 2, !dbg !27
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local588, !dbg !27
  store half %1764, ptr addrspace(1) %address.local609, align 2, !dbg !27
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local591, !dbg !27
  store half %1763, ptr addrspace(1) %address.local611, align 2, !dbg !27
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #6

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "nohint_unrolled.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 22, type: !4, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 22, column: 3, scope: !3)
!8 = !DILocation(line: 46, column: 17, scope: !3)
!9 = !DILocation(line: 66, column: 22, scope: !3)
!10 = !DILocation(line: 68, column: 24, scope: !3)
!11 = !{i32 0, i32 256}
!12 = !DILocation(line: 67, column: 22, scope: !3)
!13 = !DILocation(line: 56, column: 24, scope: !3)
!14 = !DILocation(line: 76, column: 23, scope: !3)
!15 = !DILocation(line: 69, column: 26, scope: !3)
!16 = !DILocation(line: 92, column: 20, scope: !3)
!17 = !DILocation(line: 73, column: 19, scope: !3)
!18 = !DILocation(line: 77, column: 26, scope: !3)
!19 = !DILocation(line: 84, column: 21, scope: !3)
!20 = !DILocation(line: 85, column: 22, scope: !3)
!21 = !DILocation(line: 87, column: 16, scope: !3)
!22 = !DILocation(line: 86, column: 20, scope: !3)
!23 = !DILocation(line: 93, column: 23, scope: !3)
!24 = !DILocation(line: 94, column: 20, scope: !3)
!25 = !DILocation(line: 97, column: 23, scope: !3)
!26 = !DILocation(line: 98, column: 19, scope: !3)
!27 = !DILocation(line: 99, column: 9, scope: !3)
