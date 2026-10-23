; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles/unequal_input_steps.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles/unequal_input_steps.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !32 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %6, 7, !dbg !10
  %.lobit = ashr i32 %5, 31, !dbg !12
  %10 = xor i32 %5, %.lobit, !dbg !12
  %11 = sdiv i32 %10, 32, !dbg !12
  %.lobit1 = ashr i32 %5, 31, !dbg !12
  %12 = xor i32 %11, %.lobit1, !dbg !12
  %13 = sext i32 %12 to i64, !dbg !12
  %14 = add nsw i64 %8, %13, !dbg !12
  %15 = and i64 %14, 31, !dbg !12
  %16 = shl nsw i64 %6, 7, !dbg !12
  %17 = mul nsw i64 %13, -4096, !dbg !12
  %18 = add nsw i64 %16, %17, !dbg !12
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !34
  %20 = zext nneg i32 %19 to i64, !dbg !13
  %21 = lshr i64 %20, 6, !dbg !13
  %22 = or disjoint i64 %9, 127, !dbg !14
  %23 = add nsw i64 %9, 128, !dbg !14
  %24 = icmp uge i64 %23, %22, !dbg !14
  br label %25, !dbg !15

25:                                               ; preds = %472, %4
  %26 = phi i64 [ %1195, %472 ], [ 0, %4 ], !dbg !15
  %27 = phi [2 x <16 x float>] [ %1186, %472 ], [ zeroinitializer, %4 ], !dbg !15
  %28 = phi [2 x <1 x float>] [ %832, %472 ], [ zeroinitializer, %4 ], !dbg !15
  %29 = phi [2 x <16 x float>] [ %1194, %472 ], [ zeroinitializer, %4 ], !dbg !15
  %.not = icmp sgt i64 %26, %6, !dbg !15
  br i1 %.not, label %1196, label %30, !dbg !15

30:                                               ; preds = %25
  %31 = shl nsw i64 %26, 7, !dbg !15
  %32 = lshr i64 %26, 5, !dbg !16
  %33 = shl i64 %32, 12, !dbg !16
  %34 = sub i64 %31, %33, !dbg !16
  br label %35, !dbg !22

35:                                               ; preds = %39, %30
  %36 = phi i64 [ %231, %39 ], [ 0, %30 ], !dbg !22
  %37 = phi [2 x <32 x float>] [ %230, %39 ], [ zeroinitializer, %30 ], !dbg !22
  %38 = icmp samesign ult i64 %36, 4, !dbg !22
  br i1 %38, label %39, label %232, !dbg !22

39:                                               ; preds = %35
  %40 = shl nuw nsw i64 %36, 5, !dbg !22
  %.idx = shl nuw nsw i64 %15, 20, !dbg !12
  %41 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx68 = shl nsw i64 %18, 8, !dbg !12
  %42 = getelementptr i8, ptr addrspace(1) %41, i64 %.idx68, !dbg !12
  %43 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %40, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !12, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local133 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local134 = and i64 %address.local133, 1536, !dbg !12
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local134, !dbg !12
  %address.local136 = ptrtoint ptr addrspace(3) %address.local135 to i64, !dbg !12
  %address.local137 = trunc i64 %address.local136 to i32, !dbg !12
  %44 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local137), !dbg !12
  %45 = zext i32 %44 to i64, !dbg !12
  %46 = inttoptr i64 %45 to ptr addrspace(3), !dbg !12
  %address.local138 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local139 = and i64 %address.local138, 16256, !dbg !12
  %address.local140 = lshr i64 %address.local, 4, !dbg !12
  %address.local141 = and i64 %address.local140, 8, !dbg !12
  %address.local142 = or disjoint i64 %address.local139, %address.local141, !dbg !12
  %address.local143 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local142, !dbg !12
  %47 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %48 = ptrtoint ptr addrspace(1) %address.local143 to i64, !dbg !12
  %49 = sub i64 %48, %47, !dbg !12
  %dma.buffer.offset715 = trunc i64 %49 to i32, !dbg !12
  %50 = trunc i64 %47 to i32, !dbg !12
  %51 = insertelement <4 x i32> poison, i32 %50, i64 0, !dbg !12
  %52 = lshr i64 %47, 32, !dbg !12
  %53 = trunc i64 %52 to i32, !dbg !12
  %54 = insertelement <4 x i32> %51, i32 %53, i64 1, !dbg !12
  %55 = insertelement <4 x i32> %54, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc716 = insertelement <4 x i32> %55, i32 131072, i64 3, !dbg !12
  %56 = ptrtoint ptr addrspace(3) %46 to i32, !dbg !12
  %dma.buffer.lds717 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %56), !dbg !12
  %57 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds717, i32 %dma.buffer.offset715, <4 x i32> %dma.buffer.rsrc716) #10, !dbg !12
  %address.local144 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local145 = shl nuw nsw i64 %address.local144, 3, !dbg !12
  %address.local146 = and i64 %address.local145, 3584, !dbg !12
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !12
  %address.local148 = ptrtoint ptr addrspace(3) %address.local147 to i64, !dbg !12
  %address.local149 = trunc i64 %address.local148 to i32, !dbg !12
  %58 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local149), !dbg !12
  %59 = zext i32 %58 to i64, !dbg !12
  %60 = inttoptr i64 %59 to ptr addrspace(3), !dbg !12
  %address.local150 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local151 = and i64 %address.local150, 16256, !dbg !12
  %address.local152 = lshr i64 %address.local144, 4, !dbg !12
  %address.local153 = and i64 %address.local152, 24, !dbg !12
  %address.local154 = or disjoint i64 %address.local151, %address.local153, !dbg !12
  %address.local155 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local154, !dbg !12
  %61 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %62 = ptrtoint ptr addrspace(1) %address.local155 to i64, !dbg !12
  %63 = sub i64 %62, %61, !dbg !12
  %dma.buffer.offset712 = trunc i64 %63 to i32, !dbg !12
  %64 = trunc i64 %61 to i32, !dbg !12
  %65 = insertelement <4 x i32> poison, i32 %64, i64 0, !dbg !12
  %66 = lshr i64 %61, 32, !dbg !12
  %67 = trunc i64 %66 to i32, !dbg !12
  %68 = insertelement <4 x i32> %65, i32 %67, i64 1, !dbg !12
  %69 = insertelement <4 x i32> %68, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc713 = insertelement <4 x i32> %69, i32 131072, i64 3, !dbg !12
  %70 = ptrtoint ptr addrspace(3) %60 to i32, !dbg !12
  %dma.buffer.lds714 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %70), !dbg !12
  %71 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds714, i32 %dma.buffer.offset712, <4 x i32> %dma.buffer.rsrc713) #10, !dbg !12
  %72 = shl nuw nsw i64 %36, 10, !dbg !16
  %73 = add nuw i64 %26, %72, !dbg !16
  %.lobit69 = ashr i64 %73, 63, !dbg !16
  %74 = xor i64 %73, %.lobit69, !dbg !16
  %75 = sdiv i64 %74, 4096, !dbg !16
  %.lobit70 = ashr i64 %73, 63, !dbg !16
  %76 = xor i64 %75, %.lobit70, !dbg !16
  %77 = add nsw i64 %76, %8, !dbg !16
  %78 = add nuw nsw i64 %40, %32, !dbg !16
  %79 = and i64 %78, 127, !dbg !16
  %80 = shl i64 %77, 19, !dbg !16
  %81 = and i64 %80, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %82 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %81, !dbg !16
  %83 = getelementptr [2 x i8], ptr addrspace(1) %82, i64 %79, !dbg !16
  %.idx71 = shl i64 %34, 8, !dbg !16
  %84 = getelementptr i8, ptr addrspace(1) %83, i64 %.idx71, !dbg !16
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local134, !dbg !16
  %address.local157 = ptrtoint ptr addrspace(3) %address.local156 to i64, !dbg !16
  %address.local158 = trunc i64 %address.local157 to i32, !dbg !16
  %85 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !16
  %86 = zext i32 %85 to i64, !dbg !16
  %87 = inttoptr i64 %86 to ptr addrspace(3), !dbg !16
  %address.local159 = getelementptr [2 x i8], ptr addrspace(1) %84, i64 %address.local142, !dbg !16
  %88 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %89 = ptrtoint ptr addrspace(1) %address.local159 to i64, !dbg !16
  %90 = sub i64 %89, %88, !dbg !16
  %dma.buffer.offset709 = trunc i64 %90 to i32, !dbg !16
  %91 = trunc i64 %88 to i32, !dbg !16
  %92 = insertelement <4 x i32> poison, i32 %91, i64 0, !dbg !16
  %93 = lshr i64 %88, 32, !dbg !16
  %94 = trunc i64 %93 to i32, !dbg !16
  %95 = insertelement <4 x i32> %92, i32 %94, i64 1, !dbg !16
  %96 = insertelement <4 x i32> %95, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc710 = insertelement <4 x i32> %96, i32 131072, i64 3, !dbg !16
  %97 = ptrtoint ptr addrspace(3) %87 to i32, !dbg !16
  %dma.buffer.lds711 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %97), !dbg !16
  %98 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds711, i32 %dma.buffer.offset709, <4 x i32> %dma.buffer.rsrc710) #10, !dbg !16
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local146, !dbg !16
  %address.local161 = ptrtoint ptr addrspace(3) %address.local160 to i64, !dbg !16
  %address.local162 = trunc i64 %address.local161 to i32, !dbg !16
  %99 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local162), !dbg !16
  %100 = zext i32 %99 to i64, !dbg !16
  %101 = inttoptr i64 %100 to ptr addrspace(3), !dbg !16
  %address.local163 = getelementptr [2 x i8], ptr addrspace(1) %84, i64 %address.local154, !dbg !16
  %102 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %103 = ptrtoint ptr addrspace(1) %address.local163 to i64, !dbg !16
  %104 = sub i64 %103, %102, !dbg !16
  %dma.buffer.offset = trunc i64 %104 to i32, !dbg !16
  %105 = trunc i64 %102 to i32, !dbg !16
  %106 = insertelement <4 x i32> poison, i32 %105, i64 0, !dbg !16
  %107 = lshr i64 %102, 32, !dbg !16
  %108 = trunc i64 %107 to i32, !dbg !16
  %109 = insertelement <4 x i32> %106, i32 %108, i64 1, !dbg !16
  %110 = insertelement <4 x i32> %109, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc = insertelement <4 x i32> %110, i32 131072, i64 3, !dbg !16
  %111 = ptrtoint ptr addrspace(3) %101 to i32, !dbg !16
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %111), !dbg !16
  %112 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !16
  %address.local164 = lshr i32 %address.thread, 2, !dbg !13
  %address.local165 = and i32 %address.local164, 4, !dbg !13
  %address.local166 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local167 = and i32 %address.local166, 384, !dbg !13
  %address.local168 = or disjoint i32 %address.local165, %address.local167, !dbg !13
  %address.local169 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local170 = and i32 %address.local169, 120, !dbg !13
  %address.local171 = or disjoint i32 %address.local168, %address.local170, !dbg !13
  %address.local172 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local173 = and i32 %address.local172, 1024, !dbg !13
  %address.local174 = or disjoint i32 %address.local171, %address.local173, !dbg !13
  %address.local175 = zext nneg i32 %address.local174 to i64, !dbg !13
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !13
  %113 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !13
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !13
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local177, i64 4096, !dbg !13
  %114 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !13
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !13
  %address.local180 = getelementptr i8, ptr addrspace(3) %address.local179, i64 1024, !dbg !13
  %115 = load <4 x half>, ptr addrspace(3) %address.local180, align 2, !dbg !13
  %116 = fmul <4 x half> %113, splat (half 0xH3015), !dbg !13
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !13
  %address.local182 = getelementptr i8, ptr addrspace(3) %address.local181, i64 5120, !dbg !13
  %117 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !13
  %118 = fmul <4 x half> %114, splat (half 0xH3015), !dbg !13
  %119 = fmul <4 x half> %115, splat (half 0xH3015), !dbg !13
  %120 = fmul <4 x half> %117, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local183 = lshr i32 %address.thread, 2, !dbg !8
  %address.local184 = and i32 %address.local183, 4, !dbg !8
  %address.local185 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local186 = and i32 %address.local185, 24, !dbg !8
  %address.local187 = or disjoint i32 %address.local184, %address.local186, !dbg !8
  %address.local188 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local189 = and i32 %address.local188, 1024, !dbg !8
  %address.local190 = or disjoint i32 %address.local187, %address.local189, !dbg !8
  %address.local191 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local192 = and i32 %address.local191, 384, !dbg !8
  %address.local193 = or disjoint i32 %address.local190, %address.local192, !dbg !8
  %address.local194 = zext nneg i32 %address.local193 to i64, !dbg !8
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local194, !dbg !8
  %121 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !8
  %122 = extractvalue [2 x <32 x float>] %37, 0, !dbg !31
  %123 = shufflevector <32 x float> %122, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %124 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %121, <4 x float> %123) #11, !dbg !8
  %address.local196 = lshr i32 %address.thread, 2, !dbg !8
  %address.local197 = and i32 %address.local196, 4, !dbg !8
  %address.local198 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local199 = and i32 %address.local198, 24, !dbg !8
  %address.local200 = or disjoint i32 %address.local197, %address.local199, !dbg !8
  %address.local201 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local202 = and i32 %address.local201, 1024, !dbg !8
  %address.local203 = or disjoint i32 %address.local200, %address.local202, !dbg !8
  %address.local204 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local205 = and i32 %address.local204, 384, !dbg !8
  %address.local206 = or disjoint i32 %address.local203, %address.local205, !dbg !8
  %address.local207 = zext nneg i32 %address.local206 to i64, !dbg !8
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local209 = getelementptr i8, ptr addrspace(3) %address.local208, i64 4096, !dbg !8
  %125 = load <4 x half>, ptr addrspace(3) %address.local209, align 2, !dbg !8
  %126 = shufflevector <32 x float> %122, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %127 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %125, <4 x float> %124) #11, !dbg !8
  %128 = shufflevector <32 x float> %122, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %129 = shufflevector <32 x float> %122, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %130 = shufflevector <32 x float> %122, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %131 = shufflevector <32 x float> %122, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %132 = shufflevector <32 x float> %122, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %133 = shufflevector <32 x float> %122, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %134 = extractvalue [2 x <32 x float>] %37, 1, !dbg !31
  %135 = shufflevector <32 x float> %134, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %136 = shufflevector <32 x float> %134, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %137 = shufflevector <32 x float> %134, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %138 = shufflevector <32 x float> %134, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %139 = shufflevector <32 x float> %134, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %140 = shufflevector <32 x float> %134, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %141 = shufflevector <32 x float> %134, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %142 = shufflevector <32 x float> %134, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %address.local210 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local212 = lshr i64 %address.local, 2, !dbg !8
  %address.local213 = mul nsw i64 %address.local212, -512, !dbg !8
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local213, !dbg !8
  %address.local215 = shl nuw nsw i64 %address.local, 1, !dbg !8
  %address.local216 = and i64 %address.local215, 24, !dbg !8
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local216, !dbg !8
  %address.local218 = lshr i64 %address.local, 2, !dbg !13
  %address.local219 = and i64 %address.local218, 12, !dbg !13
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local219, !dbg !8
  %address.local221 = and i64 %address.local, 32, !dbg !13
  %address.local222 = icmp eq i64 %address.local221, 0, !dbg !13
  %address.local223 = select i1 %address.local222, i64 0, i64 1016, !dbg !13
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local223, !dbg !8
  %address.local225 = getelementptr i8, ptr addrspace(3) %address.local224, i64 64, !dbg !8
  %143 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !8
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local219, !dbg !8
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local223, !dbg !8
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local213, !dbg !8
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local216, !dbg !8
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4160, !dbg !8
  %144 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !8
  %145 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %143, <4 x float> %126) #11, !dbg !8
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) %address.local232, i64 %address.local213, !dbg !8
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) %address.local233, i64 %address.local216, !dbg !8
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local219, !dbg !8
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local223, !dbg !8
  %address.local237 = getelementptr i8, ptr addrspace(3) %address.local236, i64 128, !dbg !8
  %146 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !8
  %147 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %144, <4 x float> %145) #11, !dbg !8
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) %address.local238, i64 %address.local219, !dbg !8
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) %address.local239, i64 %address.local223, !dbg !8
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local213, !dbg !8
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local216, !dbg !8
  %address.local243 = getelementptr i8, ptr addrspace(3) %address.local242, i64 4224, !dbg !8
  %148 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !8
  %149 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %146, <4 x float> %128) #11, !dbg !8
  %150 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %148, <4 x float> %149) #11, !dbg !8
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local213, !dbg !8
  %address.local246 = getelementptr [2 x i8], ptr addrspace(3) %address.local245, i64 %address.local216, !dbg !8
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) %address.local246, i64 %address.local219, !dbg !8
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local223, !dbg !8
  %address.local249 = getelementptr i8, ptr addrspace(3) %address.local248, i64 192, !dbg !8
  %151 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !8
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) %address.local250, i64 %address.local219, !dbg !8
  %address.local252 = getelementptr [2 x i8], ptr addrspace(3) %address.local251, i64 %address.local223, !dbg !8
  %address.local253 = getelementptr [2 x i8], ptr addrspace(3) %address.local252, i64 %address.local213, !dbg !8
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) %address.local253, i64 %address.local216, !dbg !8
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local254, i64 4288, !dbg !8
  %152 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !8
  %153 = shufflevector <4 x float> %150, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %154 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %151, <4 x float> %129) #11, !dbg !8
  %155 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %152, <4 x float> %154) #11, !dbg !8
  %address.local256 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local194, !dbg !8
  %address.local257 = getelementptr i8, ptr addrspace(3) %address.local256, i64 1024, !dbg !8
  %156 = load <4 x half>, ptr addrspace(3) %address.local257, align 2, !dbg !8
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local259 = getelementptr i8, ptr addrspace(3) %address.local258, i64 5120, !dbg !8
  %157 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !8
  %158 = shufflevector <4 x float> %155, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %159 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %156, <4 x float> %130) #11, !dbg !8
  %160 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %157, <4 x float> %159) #11, !dbg !8
  %address.local260 = add nuw nsw i64 %address.local210, 512, !dbg !8
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local260, !dbg !8
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local213, !dbg !8
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) %address.local262, i64 %address.local216, !dbg !8
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local219, !dbg !8
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local223, !dbg !8
  %address.local266 = getelementptr i8, ptr addrspace(3) %address.local265, i64 64, !dbg !8
  %161 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !8
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local260, !dbg !8
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local219, !dbg !8
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) %address.local268, i64 %address.local223, !dbg !8
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local213, !dbg !8
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local216, !dbg !8
  %address.local272 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4160, !dbg !8
  %162 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !8
  %163 = shufflevector <4 x float> %160, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %164 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %161, <4 x float> %131) #11, !dbg !8
  %165 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %162, <4 x float> %164) #11, !dbg !8
  %address.local273 = add nuw nsw i64 %address.local210, 512, !dbg !8
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local273, !dbg !8
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local213, !dbg !8
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) %address.local275, i64 %address.local216, !dbg !8
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) %address.local276, i64 %address.local219, !dbg !8
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local223, !dbg !8
  %address.local279 = getelementptr i8, ptr addrspace(3) %address.local278, i64 128, !dbg !8
  %166 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !8
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local273, !dbg !8
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %address.local280, i64 %address.local219, !dbg !8
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) %address.local281, i64 %address.local223, !dbg !8
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) %address.local282, i64 %address.local213, !dbg !8
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %address.local283, i64 %address.local216, !dbg !8
  %address.local285 = getelementptr i8, ptr addrspace(3) %address.local284, i64 4224, !dbg !8
  %167 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !8
  %168 = shufflevector <4 x float> %165, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %169 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %166, <4 x float> %132) #11, !dbg !8
  %170 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %167, <4 x float> %169) #11, !dbg !8
  %address.local286 = add nuw nsw i64 %address.local210, 512, !dbg !8
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local286, !dbg !8
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) %address.local287, i64 %address.local213, !dbg !8
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) %address.local288, i64 %address.local216, !dbg !8
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %address.local289, i64 %address.local219, !dbg !8
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local223, !dbg !8
  %address.local292 = getelementptr i8, ptr addrspace(3) %address.local291, i64 192, !dbg !8
  %171 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !8
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local286, !dbg !8
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) %address.local293, i64 %address.local219, !dbg !8
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) %address.local294, i64 %address.local223, !dbg !8
  %address.local296 = getelementptr [2 x i8], ptr addrspace(3) %address.local295, i64 %address.local213, !dbg !8
  %address.local297 = getelementptr [2 x i8], ptr addrspace(3) %address.local296, i64 %address.local216, !dbg !8
  %address.local298 = getelementptr i8, ptr addrspace(3) %address.local297, i64 4288, !dbg !8
  %172 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !8
  %173 = shufflevector <4 x float> %170, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %174 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %171, <4 x float> %133) #11, !dbg !8
  %175 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %172, <4 x float> %174) #11, !dbg !8
  %176 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !8
  %177 = load <4 x half>, ptr addrspace(3) %address.local209, align 2, !dbg !8
  %178 = shufflevector <4 x float> %175, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %179 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %176, <4 x float> %135) #11, !dbg !8
  %180 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %177, <4 x float> %179) #11, !dbg !8
  %181 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !8
  %182 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !8
  %183 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %181, <4 x float> %136) #11, !dbg !8
  %184 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %182, <4 x float> %183) #11, !dbg !8
  %185 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !8
  %186 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !8
  %187 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %185, <4 x float> %137) #11, !dbg !8
  %188 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %186, <4 x float> %187) #11, !dbg !8
  %189 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !8
  %190 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !8
  %191 = shufflevector <4 x float> %188, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %192 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %189, <4 x float> %138) #11, !dbg !8
  %193 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %190, <4 x float> %192) #11, !dbg !8
  %194 = load <4 x half>, ptr addrspace(3) %address.local257, align 2, !dbg !8
  %195 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !8
  %196 = shufflevector <4 x float> %193, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %197 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %194, <4 x float> %139) #11, !dbg !8
  %198 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %195, <4 x float> %197) #11, !dbg !8
  %199 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !8
  %200 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !8
  %201 = shufflevector <4 x float> %198, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %202 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %199, <4 x float> %140) #11, !dbg !8
  %203 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %200, <4 x float> %202) #11, !dbg !8
  %204 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !8
  %205 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !8
  %206 = shufflevector <4 x float> %203, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %207 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %204, <4 x float> %141) #11, !dbg !8
  %208 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %205, <4 x float> %207) #11, !dbg !8
  %209 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !8
  %210 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !8
  %211 = shufflevector <4 x float> %208, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %212 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %209, <4 x float> %142) #11, !dbg !8
  %213 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %210, <4 x float> %212) #11, !dbg !8
  %214 = shufflevector <4 x float> %213, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %215 = shufflevector <4 x float> %127, <4 x float> %147, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %216 = shufflevector <32 x float> %215, <32 x float> %153, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %217 = shufflevector <32 x float> %216, <32 x float> %158, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %218 = shufflevector <32 x float> %217, <32 x float> %163, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %219 = shufflevector <32 x float> %218, <32 x float> %168, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %220 = shufflevector <32 x float> %219, <32 x float> %173, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %221 = shufflevector <32 x float> %220, <32 x float> %178, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %222 = insertvalue [2 x <32 x float>] poison, <32 x float> %221, 0, !dbg !22
  %223 = shufflevector <4 x float> %180, <4 x float> %184, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %224 = shufflevector <32 x float> %223, <32 x float> %191, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %225 = shufflevector <32 x float> %224, <32 x float> %196, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %226 = shufflevector <32 x float> %225, <32 x float> %201, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %227 = shufflevector <32 x float> %226, <32 x float> %206, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %228 = shufflevector <32 x float> %227, <32 x float> %211, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %229 = shufflevector <32 x float> %228, <32 x float> %214, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %230 = insertvalue [2 x <32 x float>] %222, <32 x float> %229, 1, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %231 = add nuw nsw i64 %36, 1, !dbg !22
  br label %35, !dbg !22

232:                                              ; preds = %35
  %233 = add nsw i64 %32, %8, !dbg !17
  %234 = shl i64 %233, 19, !dbg !17
  %235 = and i64 %234, 16252928, !dbg !17
  %236 = shl nsw i64 %34, 7, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %237 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %235, !dbg !17
  %238 = getelementptr [2 x i8], ptr addrspace(1) %237, i64 %236, !dbg !17
  %address.thread299 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !17, !range !34, !frisk.register_lifetimes !1
  %address.local300 = zext nneg i32 %address.thread299 to i64, !dbg !13
  %address.local301 = shl nuw nsw i64 %address.local300, 3, !dbg !12
  %address.local302 = and i64 %address.local301, 1536, !dbg !12
  %address.local303 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local302, !dbg !17
  %address.local304 = ptrtoint ptr addrspace(3) %address.local303 to i64, !dbg !17
  %address.local305 = trunc i64 %address.local304 to i32, !dbg !17
  %239 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local305), !dbg !17
  %240 = zext i32 %239 to i64, !dbg !17
  %241 = inttoptr i64 %240 to ptr addrspace(3), !dbg !17
  %address.local306 = lshr i64 %address.local300, 3, !dbg !17
  %address.local307 = and i64 %address.local306, 30, !dbg !17
  %address.local308 = and i64 %address.local300, 1, !dbg !17
  %address.local309 = or disjoint i64 %address.local307, %address.local308, !dbg !17
  %address.local310 = shl nuw nsw i64 %address.local309, 7, !dbg !17
  %address.local311 = shl nuw nsw i64 %address.local300, 2, !dbg !17
  %address.local312 = and i64 %address.local311, 56, !dbg !17
  %address.local313 = or disjoint i64 %address.local310, %address.local312, !dbg !17
  %address.local314 = getelementptr [2 x i8], ptr addrspace(1) %238, i64 %address.local313, !dbg !17
  %242 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %243 = ptrtoint ptr addrspace(1) %address.local314 to i64, !dbg !17
  %244 = sub i64 %243, %242, !dbg !17
  %dma.buffer.offset739 = trunc i64 %244 to i32, !dbg !17
  %245 = trunc i64 %242 to i32, !dbg !17
  %246 = insertelement <4 x i32> poison, i32 %245, i64 0, !dbg !17
  %247 = lshr i64 %242, 32, !dbg !17
  %248 = trunc i64 %247 to i32, !dbg !17
  %249 = insertelement <4 x i32> %246, i32 %248, i64 1, !dbg !17
  %250 = insertelement <4 x i32> %249, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc740 = insertelement <4 x i32> %250, i32 131072, i64 3, !dbg !17
  %251 = ptrtoint ptr addrspace(3) %241 to i32, !dbg !17
  %dma.buffer.lds741 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %251), !dbg !17
  %252 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds741, i32 %dma.buffer.offset739, <4 x i32> %dma.buffer.rsrc740) #10, !dbg !17
  %address.local315 = or disjoint i64 %address.local300, 256, !dbg !12
  %address.local316 = shl nuw nsw i64 %address.local315, 3, !dbg !12
  %address.local317 = and i64 %address.local316, 3584, !dbg !17
  %address.local318 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local317, !dbg !17
  %address.local319 = ptrtoint ptr addrspace(3) %address.local318 to i64, !dbg !17
  %address.local320 = trunc i64 %address.local319 to i32, !dbg !17
  %253 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local320), !dbg !17
  %254 = zext i32 %253 to i64, !dbg !17
  %255 = inttoptr i64 %254 to ptr addrspace(3), !dbg !17
  %address.local321 = lshr i64 %address.local315, 3, !dbg !17
  %address.local322 = and i64 %address.local321, 62, !dbg !17
  %address.local323 = and i64 %address.local300, 1, !dbg !17
  %address.local324 = or disjoint i64 %address.local322, %address.local323, !dbg !17
  %address.local325 = shl nuw nsw i64 %address.local324, 7, !dbg !17
  %address.local326 = shl nuw nsw i64 %address.local300, 2, !dbg !17
  %address.local327 = and i64 %address.local326, 56, !dbg !17
  %address.local328 = or disjoint i64 %address.local325, %address.local327, !dbg !17
  %address.local329 = getelementptr [2 x i8], ptr addrspace(1) %238, i64 %address.local328, !dbg !17
  %256 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %257 = ptrtoint ptr addrspace(1) %address.local329 to i64, !dbg !17
  %258 = sub i64 %257, %256, !dbg !17
  %dma.buffer.offset736 = trunc i64 %258 to i32, !dbg !17
  %259 = trunc i64 %256 to i32, !dbg !17
  %260 = insertelement <4 x i32> poison, i32 %259, i64 0, !dbg !17
  %261 = lshr i64 %256, 32, !dbg !17
  %262 = trunc i64 %261 to i32, !dbg !17
  %263 = insertelement <4 x i32> %260, i32 %262, i64 1, !dbg !17
  %264 = insertelement <4 x i32> %263, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc737 = insertelement <4 x i32> %264, i32 131072, i64 3, !dbg !17
  %265 = ptrtoint ptr addrspace(3) %255 to i32, !dbg !17
  %dma.buffer.lds738 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %265), !dbg !17
  %266 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds738, i32 %dma.buffer.offset736, <4 x i32> %dma.buffer.rsrc737) #10, !dbg !17
  %address.local330 = or disjoint i64 %address.local300, 512, !dbg !17
  %address.local331 = shl nuw nsw i64 %address.local330, 3, !dbg !17
  %address.local332 = and i64 %address.local331, 5632, !dbg !17
  %address.local333 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !17
  %address.local334 = ptrtoint ptr addrspace(3) %address.local333 to i64, !dbg !17
  %address.local335 = trunc i64 %address.local334 to i32, !dbg !17
  %267 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local335), !dbg !17
  %268 = zext i32 %267 to i64, !dbg !17
  %269 = inttoptr i64 %268 to ptr addrspace(3), !dbg !17
  %address.local336 = lshr i64 %address.local330, 3, !dbg !17
  %address.local337 = and i64 %address.local336, 94, !dbg !17
  %address.local338 = and i64 %address.local300, 1, !dbg !17
  %address.local339 = or disjoint i64 %address.local337, %address.local338, !dbg !17
  %address.local340 = shl nuw nsw i64 %address.local339, 7, !dbg !17
  %address.local341 = shl nuw nsw i64 %address.local300, 2, !dbg !17
  %address.local342 = and i64 %address.local341, 56, !dbg !17
  %address.local343 = or disjoint i64 %address.local340, %address.local342, !dbg !17
  %address.local344 = getelementptr [2 x i8], ptr addrspace(1) %238, i64 %address.local343, !dbg !17
  %270 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %271 = ptrtoint ptr addrspace(1) %address.local344 to i64, !dbg !17
  %272 = sub i64 %271, %270, !dbg !17
  %dma.buffer.offset733 = trunc i64 %272 to i32, !dbg !17
  %273 = trunc i64 %270 to i32, !dbg !17
  %274 = insertelement <4 x i32> poison, i32 %273, i64 0, !dbg !17
  %275 = lshr i64 %270, 32, !dbg !17
  %276 = trunc i64 %275 to i32, !dbg !17
  %277 = insertelement <4 x i32> %274, i32 %276, i64 1, !dbg !17
  %278 = insertelement <4 x i32> %277, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc734 = insertelement <4 x i32> %278, i32 131072, i64 3, !dbg !17
  %279 = ptrtoint ptr addrspace(3) %269 to i32, !dbg !17
  %dma.buffer.lds735 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %279), !dbg !17
  %280 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds735, i32 %dma.buffer.offset733, <4 x i32> %dma.buffer.rsrc734) #10, !dbg !17
  %address.local345 = or disjoint i64 %address.local300, 768, !dbg !17
  %address.local346 = shl nuw nsw i64 %address.local345, 3, !dbg !17
  %address.local347 = and i64 %address.local346, 7680, !dbg !17
  %address.local348 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local347, !dbg !17
  %address.local349 = ptrtoint ptr addrspace(3) %address.local348 to i64, !dbg !17
  %address.local350 = trunc i64 %address.local349 to i32, !dbg !17
  %281 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local350), !dbg !17
  %282 = zext i32 %281 to i64, !dbg !17
  %283 = inttoptr i64 %282 to ptr addrspace(3), !dbg !17
  %address.local351 = lshr i64 %address.local345, 3, !dbg !17
  %address.local352 = and i64 %address.local351, 126, !dbg !17
  %address.local353 = and i64 %address.local300, 1, !dbg !17
  %address.local354 = or disjoint i64 %address.local352, %address.local353, !dbg !17
  %address.local355 = shl nuw nsw i64 %address.local354, 7, !dbg !17
  %address.local356 = shl nuw nsw i64 %address.local300, 2, !dbg !17
  %address.local357 = and i64 %address.local356, 56, !dbg !17
  %address.local358 = or disjoint i64 %address.local355, %address.local357, !dbg !17
  %address.local359 = getelementptr [2 x i8], ptr addrspace(1) %238, i64 %address.local358, !dbg !17
  %284 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %285 = ptrtoint ptr addrspace(1) %address.local359 to i64, !dbg !17
  %286 = sub i64 %285, %284, !dbg !17
  %dma.buffer.offset730 = trunc i64 %286 to i32, !dbg !17
  %287 = trunc i64 %284 to i32, !dbg !17
  %288 = insertelement <4 x i32> poison, i32 %287, i64 0, !dbg !17
  %289 = lshr i64 %284, 32, !dbg !17
  %290 = trunc i64 %289 to i32, !dbg !17
  %291 = insertelement <4 x i32> %288, i32 %290, i64 1, !dbg !17
  %292 = insertelement <4 x i32> %291, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc731 = insertelement <4 x i32> %292, i32 131072, i64 3, !dbg !17
  %293 = ptrtoint ptr addrspace(3) %283 to i32, !dbg !17
  %dma.buffer.lds732 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %293), !dbg !17
  %294 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds732, i32 %dma.buffer.offset730, <4 x i32> %dma.buffer.rsrc731) #10, !dbg !17
  %295 = or disjoint i64 %31, 127, !dbg !14
  %296 = icmp uge i64 %9, %295, !dbg !14
  %297 = and i1 %24, %296, !dbg !14
  br i1 %297, label %298, label %299, !dbg !14

298:                                              ; preds = %232
  br label %470, !dbg !14

299:                                              ; preds = %232
  %300 = add nsw i64 %9, %20, !dbg !14
  %301 = shl nuw nsw i64 %21, 4, !dbg !14
  %302 = add nsw i64 %300, %301, !dbg !14
  %303 = and i32 %19, 240, !dbg !14
  %index.compact = zext nneg i32 %303 to i64, !dbg !14
  %304 = sub nsw i64 %302, %index.compact, !dbg !14
  %305 = and i64 %20, 48, !dbg !14
  %306 = or disjoint i64 %31, %305, !dbg !14
  %307 = add nsw i64 %304, 1, !dbg !23
  %.not4 = icmp ugt i64 %307, %306, !dbg !24
  %308 = select i1 %.not4, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %309 = or disjoint i64 %306, 1, !dbg !14
  %.not5 = icmp ugt i64 %307, %309, !dbg !24
  %310 = select i1 %.not5, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %311 = or disjoint i64 %306, 2, !dbg !14
  %.not6 = icmp ugt i64 %307, %311, !dbg !24
  %312 = select i1 %.not6, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %313 = or disjoint i64 %306, 3, !dbg !14
  %.not7 = icmp ugt i64 %307, %313, !dbg !24
  %314 = select i1 %.not7, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %315 = or disjoint i64 %306, 4, !dbg !14
  %.not8 = icmp ugt i64 %307, %315, !dbg !24
  %316 = select i1 %.not8, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %317 = or disjoint i64 %306, 5, !dbg !14
  %.not9 = icmp ugt i64 %307, %317, !dbg !24
  %318 = select i1 %.not9, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %319 = or disjoint i64 %306, 6, !dbg !14
  %.not10 = icmp ugt i64 %307, %319, !dbg !24
  %320 = select i1 %.not10, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %321 = or disjoint i64 %306, 7, !dbg !14
  %.not11 = icmp ugt i64 %307, %321, !dbg !24
  %322 = select i1 %.not11, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %323 = or disjoint i64 %306, 8, !dbg !14
  %.not12 = icmp ugt i64 %307, %323, !dbg !24
  %324 = select i1 %.not12, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %325 = or disjoint i64 %306, 9, !dbg !14
  %.not13 = icmp ugt i64 %307, %325, !dbg !24
  %326 = select i1 %.not13, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %327 = or disjoint i64 %306, 10, !dbg !14
  %.not14 = icmp ugt i64 %307, %327, !dbg !24
  %328 = select i1 %.not14, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %329 = or disjoint i64 %306, 11, !dbg !14
  %.not15 = icmp ugt i64 %307, %329, !dbg !24
  %330 = select i1 %.not15, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %331 = or disjoint i64 %306, 12, !dbg !14
  %.not16 = icmp ugt i64 %307, %331, !dbg !24
  %332 = select i1 %.not16, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %333 = or disjoint i64 %306, 13, !dbg !14
  %.not17 = icmp ugt i64 %307, %333, !dbg !24
  %334 = select i1 %.not17, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %335 = or disjoint i64 %306, 14, !dbg !14
  %.not18 = icmp ugt i64 %307, %335, !dbg !24
  %336 = select i1 %.not18, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %337 = or disjoint i64 %306, 15, !dbg !14
  %.not19 = icmp ugt i64 %307, %337, !dbg !24
  %338 = select i1 %.not19, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %339 = or disjoint i64 %306, 64, !dbg !14
  %.not20 = icmp ugt i64 %307, %339, !dbg !24
  %340 = select i1 %.not20, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %341 = or disjoint i64 %306, 65, !dbg !14
  %.not21 = icmp ugt i64 %307, %341, !dbg !24
  %342 = select i1 %.not21, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %343 = or disjoint i64 %306, 66, !dbg !14
  %.not22 = icmp ugt i64 %307, %343, !dbg !24
  %344 = select i1 %.not22, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %345 = or disjoint i64 %306, 67, !dbg !14
  %.not23 = icmp ugt i64 %307, %345, !dbg !24
  %346 = select i1 %.not23, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %347 = or disjoint i64 %306, 68, !dbg !14
  %.not24 = icmp ugt i64 %307, %347, !dbg !24
  %348 = select i1 %.not24, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %349 = or disjoint i64 %306, 69, !dbg !14
  %.not25 = icmp ugt i64 %307, %349, !dbg !24
  %350 = select i1 %.not25, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %351 = or disjoint i64 %306, 70, !dbg !14
  %.not26 = icmp ugt i64 %307, %351, !dbg !24
  %352 = select i1 %.not26, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %353 = or disjoint i64 %306, 71, !dbg !14
  %.not27 = icmp ugt i64 %307, %353, !dbg !24
  %354 = select i1 %.not27, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %355 = or disjoint i64 %306, 72, !dbg !14
  %.not28 = icmp ugt i64 %307, %355, !dbg !24
  %356 = select i1 %.not28, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %357 = or disjoint i64 %306, 73, !dbg !14
  %.not29 = icmp ugt i64 %307, %357, !dbg !24
  %358 = select i1 %.not29, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %359 = or disjoint i64 %306, 74, !dbg !14
  %.not30 = icmp ugt i64 %307, %359, !dbg !24
  %360 = select i1 %.not30, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %361 = or disjoint i64 %306, 75, !dbg !14
  %.not31 = icmp ugt i64 %307, %361, !dbg !24
  %362 = select i1 %.not31, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %363 = or disjoint i64 %306, 76, !dbg !14
  %.not32 = icmp ugt i64 %307, %363, !dbg !24
  %364 = select i1 %.not32, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %365 = or disjoint i64 %306, 77, !dbg !14
  %.not33 = icmp ugt i64 %307, %365, !dbg !24
  %366 = select i1 %.not33, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %367 = or disjoint i64 %306, 78, !dbg !14
  %.not34 = icmp ugt i64 %307, %367, !dbg !24
  %368 = select i1 %.not34, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %369 = or disjoint i64 %306, 79, !dbg !14
  %.not35 = icmp ugt i64 %307, %369, !dbg !24
  %370 = select i1 %.not35, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %371 = add nsw i64 %304, 65, !dbg !23
  %.not36 = icmp ugt i64 %371, %306, !dbg !24
  %372 = select i1 %.not36, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not37 = icmp ugt i64 %371, %309, !dbg !24
  %373 = select i1 %.not37, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not38 = icmp ugt i64 %371, %311, !dbg !24
  %374 = select i1 %.not38, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not39 = icmp ugt i64 %371, %313, !dbg !24
  %375 = select i1 %.not39, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not40 = icmp ugt i64 %371, %315, !dbg !24
  %376 = select i1 %.not40, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not41 = icmp ugt i64 %371, %317, !dbg !24
  %377 = select i1 %.not41, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not42 = icmp ugt i64 %371, %319, !dbg !24
  %378 = select i1 %.not42, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not43 = icmp ugt i64 %371, %321, !dbg !24
  %379 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not44 = icmp ugt i64 %371, %323, !dbg !24
  %380 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not45 = icmp ugt i64 %371, %325, !dbg !24
  %381 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not46 = icmp ugt i64 %371, %327, !dbg !24
  %382 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not47 = icmp ugt i64 %371, %329, !dbg !24
  %383 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not48 = icmp ugt i64 %371, %331, !dbg !24
  %384 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not49 = icmp ugt i64 %371, %333, !dbg !24
  %385 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not50 = icmp ugt i64 %371, %335, !dbg !24
  %386 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not51 = icmp ugt i64 %371, %337, !dbg !24
  %387 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not52 = icmp ugt i64 %371, %339, !dbg !24
  %388 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not53 = icmp ugt i64 %371, %341, !dbg !24
  %389 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not54 = icmp ugt i64 %371, %343, !dbg !24
  %390 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not55 = icmp ugt i64 %371, %345, !dbg !24
  %391 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not56 = icmp ugt i64 %371, %347, !dbg !24
  %392 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not57 = icmp ugt i64 %371, %349, !dbg !24
  %393 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not58 = icmp ugt i64 %371, %351, !dbg !24
  %394 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not59 = icmp ugt i64 %371, %353, !dbg !24
  %395 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not60 = icmp ugt i64 %371, %355, !dbg !24
  %396 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not61 = icmp ugt i64 %371, %357, !dbg !24
  %397 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not62 = icmp ugt i64 %371, %359, !dbg !24
  %398 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not63 = icmp ugt i64 %371, %361, !dbg !24
  %399 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not64 = icmp ugt i64 %371, %363, !dbg !24
  %400 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not65 = icmp ugt i64 %371, %365, !dbg !24
  %401 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not66 = icmp ugt i64 %371, %367, !dbg !24
  %402 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not67 = icmp ugt i64 %371, %369, !dbg !24
  %403 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %404 = insertelement <32 x float> poison, float %308, i64 0, !dbg !14
  %405 = insertelement <32 x float> %404, float %310, i64 1, !dbg !14
  %406 = insertelement <32 x float> %405, float %312, i64 2, !dbg !14
  %407 = insertelement <32 x float> %406, float %314, i64 3, !dbg !14
  %408 = insertelement <32 x float> %407, float %316, i64 4, !dbg !14
  %409 = insertelement <32 x float> %408, float %318, i64 5, !dbg !14
  %410 = insertelement <32 x float> %409, float %320, i64 6, !dbg !14
  %411 = insertelement <32 x float> %410, float %322, i64 7, !dbg !14
  %412 = insertelement <32 x float> %411, float %324, i64 8, !dbg !14
  %413 = insertelement <32 x float> %412, float %326, i64 9, !dbg !14
  %414 = insertelement <32 x float> %413, float %328, i64 10, !dbg !14
  %415 = insertelement <32 x float> %414, float %330, i64 11, !dbg !14
  %416 = insertelement <32 x float> %415, float %332, i64 12, !dbg !14
  %417 = insertelement <32 x float> %416, float %334, i64 13, !dbg !14
  %418 = insertelement <32 x float> %417, float %336, i64 14, !dbg !14
  %419 = insertelement <32 x float> %418, float %338, i64 15, !dbg !14
  %420 = insertelement <32 x float> %419, float %340, i64 16, !dbg !14
  %421 = insertelement <32 x float> %420, float %342, i64 17, !dbg !14
  %422 = insertelement <32 x float> %421, float %344, i64 18, !dbg !14
  %423 = insertelement <32 x float> %422, float %346, i64 19, !dbg !14
  %424 = insertelement <32 x float> %423, float %348, i64 20, !dbg !14
  %425 = insertelement <32 x float> %424, float %350, i64 21, !dbg !14
  %426 = insertelement <32 x float> %425, float %352, i64 22, !dbg !14
  %427 = insertelement <32 x float> %426, float %354, i64 23, !dbg !14
  %428 = insertelement <32 x float> %427, float %356, i64 24, !dbg !14
  %429 = insertelement <32 x float> %428, float %358, i64 25, !dbg !14
  %430 = insertelement <32 x float> %429, float %360, i64 26, !dbg !14
  %431 = insertelement <32 x float> %430, float %362, i64 27, !dbg !14
  %432 = insertelement <32 x float> %431, float %364, i64 28, !dbg !14
  %433 = insertelement <32 x float> %432, float %366, i64 29, !dbg !14
  %434 = insertelement <32 x float> %433, float %368, i64 30, !dbg !14
  %435 = insertelement <32 x float> %434, float %370, i64 31, !dbg !14
  %436 = insertvalue [2 x <32 x float>] poison, <32 x float> %435, 0, !dbg !14
  %437 = insertelement <32 x float> poison, float %372, i64 0, !dbg !14
  %438 = insertelement <32 x float> %437, float %373, i64 1, !dbg !14
  %439 = insertelement <32 x float> %438, float %374, i64 2, !dbg !14
  %440 = insertelement <32 x float> %439, float %375, i64 3, !dbg !14
  %441 = insertelement <32 x float> %440, float %376, i64 4, !dbg !14
  %442 = insertelement <32 x float> %441, float %377, i64 5, !dbg !14
  %443 = insertelement <32 x float> %442, float %378, i64 6, !dbg !14
  %444 = insertelement <32 x float> %443, float %379, i64 7, !dbg !14
  %445 = insertelement <32 x float> %444, float %380, i64 8, !dbg !14
  %446 = insertelement <32 x float> %445, float %381, i64 9, !dbg !14
  %447 = insertelement <32 x float> %446, float %382, i64 10, !dbg !14
  %448 = insertelement <32 x float> %447, float %383, i64 11, !dbg !14
  %449 = insertelement <32 x float> %448, float %384, i64 12, !dbg !14
  %450 = insertelement <32 x float> %449, float %385, i64 13, !dbg !14
  %451 = insertelement <32 x float> %450, float %386, i64 14, !dbg !14
  %452 = insertelement <32 x float> %451, float %387, i64 15, !dbg !14
  %453 = insertelement <32 x float> %452, float %388, i64 16, !dbg !14
  %454 = insertelement <32 x float> %453, float %389, i64 17, !dbg !14
  %455 = insertelement <32 x float> %454, float %390, i64 18, !dbg !14
  %456 = insertelement <32 x float> %455, float %391, i64 19, !dbg !14
  %457 = insertelement <32 x float> %456, float %392, i64 20, !dbg !14
  %458 = insertelement <32 x float> %457, float %393, i64 21, !dbg !14
  %459 = insertelement <32 x float> %458, float %394, i64 22, !dbg !14
  %460 = insertelement <32 x float> %459, float %395, i64 23, !dbg !14
  %461 = insertelement <32 x float> %460, float %396, i64 24, !dbg !14
  %462 = insertelement <32 x float> %461, float %397, i64 25, !dbg !14
  %463 = insertelement <32 x float> %462, float %398, i64 26, !dbg !14
  %464 = insertelement <32 x float> %463, float %399, i64 27, !dbg !14
  %465 = insertelement <32 x float> %464, float %400, i64 28, !dbg !14
  %466 = insertelement <32 x float> %465, float %401, i64 29, !dbg !14
  %467 = insertelement <32 x float> %466, float %402, i64 30, !dbg !14
  %468 = insertelement <32 x float> %467, float %403, i64 31, !dbg !14
  %469 = insertvalue [2 x <32 x float>] %436, <32 x float> %468, 1, !dbg !14
  br label %470, !dbg !14

470:                                              ; preds = %298, %299
  %471 = phi [2 x <32 x float>] [ %469, %299 ], [ zeroinitializer, %298 ], !dbg !14
  br label %472, !dbg !14

472:                                              ; preds = %470
  %473 = extractvalue [2 x <32 x float>] %37, 0, !dbg !26
  %474 = extractvalue [2 x <32 x float>] %471, 0, !dbg !26
  %475 = shufflevector <32 x float> %473, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %476 = shufflevector <32 x float> %474, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %477 = fadd <4 x float> %475, %476, !dbg !26
  %478 = extractelement <4 x float> %477, i64 0, !dbg !27
  %479 = call float @llvm.amdgcn.exp2.f32(float %478), !dbg !27
  %480 = insertelement <4 x float> poison, float %479, i64 0, !dbg !27
  %481 = extractelement <4 x float> %477, i64 1, !dbg !27
  %482 = call float @llvm.amdgcn.exp2.f32(float %481), !dbg !27
  %483 = fadd float %479, %482, !dbg !28
  %484 = insertelement <4 x float> %480, float %482, i64 1, !dbg !27
  %485 = extractelement <4 x float> %477, i64 2, !dbg !27
  %486 = call float @llvm.amdgcn.exp2.f32(float %485), !dbg !27
  %487 = fadd float %483, %486, !dbg !28
  %488 = insertelement <4 x float> %484, float %486, i64 2, !dbg !27
  %489 = extractelement <4 x float> %477, i64 3, !dbg !27
  %490 = call float @llvm.amdgcn.exp2.f32(float %489), !dbg !27
  %491 = fadd float %487, %490, !dbg !28
  %492 = insertelement <4 x float> %488, float %490, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %492) #9, !dbg !29, !frisk.register_lifetimes !1
  %493 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %493) #9, !dbg !26, !frisk.register_lifetimes !1
  %494 = shufflevector <32 x float> %473, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %495 = shufflevector <32 x float> %474, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %496 = fadd <4 x float> %494, %495, !dbg !26
  %497 = extractelement <4 x float> %496, i64 0, !dbg !27
  %498 = call float @llvm.amdgcn.exp2.f32(float %497), !dbg !27
  %499 = fadd float %491, %498, !dbg !28
  %500 = insertelement <4 x float> poison, float %498, i64 0, !dbg !27
  %501 = extractelement <4 x float> %496, i64 1, !dbg !27
  %502 = call float @llvm.amdgcn.exp2.f32(float %501), !dbg !27
  %503 = fadd float %499, %502, !dbg !28
  %504 = insertelement <4 x float> %500, float %502, i64 1, !dbg !27
  %505 = extractelement <4 x float> %496, i64 2, !dbg !27
  %506 = call float @llvm.amdgcn.exp2.f32(float %505), !dbg !27
  %507 = fadd float %503, %506, !dbg !28
  %508 = insertelement <4 x float> %504, float %506, i64 2, !dbg !27
  %509 = extractelement <4 x float> %496, i64 3, !dbg !27
  %510 = call float @llvm.amdgcn.exp2.f32(float %509), !dbg !27
  %511 = fadd float %507, %510, !dbg !28
  %512 = insertelement <4 x float> %508, float %510, i64 3, !dbg !27
  %math.result103 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %512) #9, !dbg !29, !frisk.register_lifetimes !1
  %513 = fptrunc <4 x float> %math.result103 to <4 x half>, !dbg !29
  %math.packed104 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %513) #9, !dbg !26, !frisk.register_lifetimes !1
  %514 = shufflevector <32 x float> %473, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %515 = shufflevector <32 x float> %474, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %516 = fadd <4 x float> %514, %515, !dbg !26
  %517 = extractelement <4 x float> %516, i64 0, !dbg !27
  %518 = call float @llvm.amdgcn.exp2.f32(float %517), !dbg !27
  %519 = fadd float %511, %518, !dbg !28
  %520 = insertelement <4 x float> poison, float %518, i64 0, !dbg !27
  %521 = extractelement <4 x float> %516, i64 1, !dbg !27
  %522 = call float @llvm.amdgcn.exp2.f32(float %521), !dbg !27
  %523 = fadd float %519, %522, !dbg !28
  %524 = insertelement <4 x float> %520, float %522, i64 1, !dbg !27
  %525 = extractelement <4 x float> %516, i64 2, !dbg !27
  %526 = call float @llvm.amdgcn.exp2.f32(float %525), !dbg !27
  %527 = fadd float %523, %526, !dbg !28
  %528 = insertelement <4 x float> %524, float %526, i64 2, !dbg !27
  %529 = extractelement <4 x float> %516, i64 3, !dbg !27
  %530 = call float @llvm.amdgcn.exp2.f32(float %529), !dbg !27
  %531 = fadd float %527, %530, !dbg !28
  %532 = insertelement <4 x float> %528, float %530, i64 3, !dbg !27
  %math.result105 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %532) #9, !dbg !29, !frisk.register_lifetimes !1
  %533 = fptrunc <4 x float> %math.result105 to <4 x half>, !dbg !29
  %math.packed106 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %533) #9, !dbg !26, !frisk.register_lifetimes !1
  %534 = shufflevector <32 x float> %473, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %535 = shufflevector <32 x float> %474, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %536 = fadd <4 x float> %534, %535, !dbg !26
  %537 = extractelement <4 x float> %536, i64 0, !dbg !27
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !27
  %539 = fadd float %531, %538, !dbg !28
  %540 = insertelement <4 x float> poison, float %538, i64 0, !dbg !27
  %541 = extractelement <4 x float> %536, i64 1, !dbg !27
  %542 = call float @llvm.amdgcn.exp2.f32(float %541), !dbg !27
  %543 = fadd float %539, %542, !dbg !28
  %544 = insertelement <4 x float> %540, float %542, i64 1, !dbg !27
  %545 = extractelement <4 x float> %536, i64 2, !dbg !27
  %546 = call float @llvm.amdgcn.exp2.f32(float %545), !dbg !27
  %547 = fadd float %543, %546, !dbg !28
  %548 = insertelement <4 x float> %544, float %546, i64 2, !dbg !27
  %549 = extractelement <4 x float> %536, i64 3, !dbg !27
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !27
  %551 = fadd float %547, %550, !dbg !28
  %552 = insertelement <4 x float> %548, float %550, i64 3, !dbg !27
  %math.result107 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %552) #9, !dbg !29, !frisk.register_lifetimes !1
  %553 = fptrunc <4 x float> %math.result107 to <4 x half>, !dbg !29
  %math.packed108 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %553) #9, !dbg !26, !frisk.register_lifetimes !1
  %554 = shufflevector <32 x float> %473, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %555 = shufflevector <32 x float> %474, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %556 = fadd <4 x float> %554, %555, !dbg !26
  %557 = extractelement <4 x float> %556, i64 0, !dbg !27
  %558 = call float @llvm.amdgcn.exp2.f32(float %557), !dbg !27
  %559 = fadd float %551, %558, !dbg !28
  %560 = insertelement <4 x float> poison, float %558, i64 0, !dbg !27
  %561 = extractelement <4 x float> %556, i64 1, !dbg !27
  %562 = call float @llvm.amdgcn.exp2.f32(float %561), !dbg !27
  %563 = fadd float %559, %562, !dbg !28
  %564 = insertelement <4 x float> %560, float %562, i64 1, !dbg !27
  %565 = extractelement <4 x float> %556, i64 2, !dbg !27
  %566 = call float @llvm.amdgcn.exp2.f32(float %565), !dbg !27
  %567 = fadd float %563, %566, !dbg !28
  %568 = insertelement <4 x float> %564, float %566, i64 2, !dbg !27
  %569 = extractelement <4 x float> %556, i64 3, !dbg !27
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !27
  %571 = fadd float %567, %570, !dbg !28
  %572 = insertelement <4 x float> %568, float %570, i64 3, !dbg !27
  %math.result109 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %572) #9, !dbg !29, !frisk.register_lifetimes !1
  %573 = fptrunc <4 x float> %math.result109 to <4 x half>, !dbg !29
  %math.packed110 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %573) #9, !dbg !26, !frisk.register_lifetimes !1
  %574 = shufflevector <32 x float> %473, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %575 = shufflevector <32 x float> %474, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %576 = fadd <4 x float> %574, %575, !dbg !26
  %577 = extractelement <4 x float> %576, i64 0, !dbg !27
  %578 = call float @llvm.amdgcn.exp2.f32(float %577), !dbg !27
  %579 = fadd float %571, %578, !dbg !28
  %580 = insertelement <4 x float> poison, float %578, i64 0, !dbg !27
  %581 = extractelement <4 x float> %576, i64 1, !dbg !27
  %582 = call float @llvm.amdgcn.exp2.f32(float %581), !dbg !27
  %583 = fadd float %579, %582, !dbg !28
  %584 = insertelement <4 x float> %580, float %582, i64 1, !dbg !27
  %585 = extractelement <4 x float> %576, i64 2, !dbg !27
  %586 = call float @llvm.amdgcn.exp2.f32(float %585), !dbg !27
  %587 = fadd float %583, %586, !dbg !28
  %588 = insertelement <4 x float> %584, float %586, i64 2, !dbg !27
  %589 = extractelement <4 x float> %576, i64 3, !dbg !27
  %590 = call float @llvm.amdgcn.exp2.f32(float %589), !dbg !27
  %591 = fadd float %587, %590, !dbg !28
  %592 = insertelement <4 x float> %588, float %590, i64 3, !dbg !27
  %math.result111 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %592) #9, !dbg !29, !frisk.register_lifetimes !1
  %593 = fptrunc <4 x float> %math.result111 to <4 x half>, !dbg !29
  %math.packed112 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %593) #9, !dbg !26, !frisk.register_lifetimes !1
  %594 = shufflevector <32 x float> %473, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %595 = shufflevector <32 x float> %474, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %596 = fadd <4 x float> %594, %595, !dbg !26
  %597 = extractelement <4 x float> %596, i64 0, !dbg !27
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !27
  %599 = fadd float %591, %598, !dbg !28
  %600 = insertelement <4 x float> poison, float %598, i64 0, !dbg !27
  %601 = extractelement <4 x float> %596, i64 1, !dbg !27
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !27
  %603 = fadd float %599, %602, !dbg !28
  %604 = insertelement <4 x float> %600, float %602, i64 1, !dbg !27
  %605 = extractelement <4 x float> %596, i64 2, !dbg !27
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !27
  %607 = fadd float %603, %606, !dbg !28
  %608 = insertelement <4 x float> %604, float %606, i64 2, !dbg !27
  %609 = extractelement <4 x float> %596, i64 3, !dbg !27
  %610 = call float @llvm.amdgcn.exp2.f32(float %609), !dbg !27
  %611 = fadd float %607, %610, !dbg !28
  %612 = insertelement <4 x float> %608, float %610, i64 3, !dbg !27
  %math.result113 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %612) #9, !dbg !29, !frisk.register_lifetimes !1
  %613 = fptrunc <4 x float> %math.result113 to <4 x half>, !dbg !29
  %math.packed114 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %613) #9, !dbg !26, !frisk.register_lifetimes !1
  %614 = shufflevector <32 x float> %473, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %615 = shufflevector <32 x float> %474, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %616 = fadd <4 x float> %614, %615, !dbg !26
  %617 = extractelement <4 x float> %616, i64 0, !dbg !27
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !27
  %619 = fadd float %611, %618, !dbg !28
  %620 = insertelement <4 x float> poison, float %618, i64 0, !dbg !27
  %621 = extractelement <4 x float> %616, i64 1, !dbg !27
  %622 = call float @llvm.amdgcn.exp2.f32(float %621), !dbg !27
  %623 = fadd float %619, %622, !dbg !28
  %624 = insertelement <4 x float> %620, float %622, i64 1, !dbg !27
  %625 = extractelement <4 x float> %616, i64 2, !dbg !27
  %626 = call float @llvm.amdgcn.exp2.f32(float %625), !dbg !27
  %627 = fadd float %623, %626, !dbg !28
  %628 = insertelement <4 x float> %624, float %626, i64 2, !dbg !27
  %629 = extractelement <4 x float> %616, i64 3, !dbg !27
  %630 = call float @llvm.amdgcn.exp2.f32(float %629), !dbg !27
  %631 = fadd float %627, %630, !dbg !28
  %632 = insertelement <4 x float> %628, float %630, i64 3, !dbg !27
  %math.result115 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %632) #9, !dbg !29, !frisk.register_lifetimes !1
  %633 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !29
  %math.packed116 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %633) #9, !dbg !26, !frisk.register_lifetimes !1
  %634 = extractvalue [2 x <32 x float>] %37, 1, !dbg !26
  %635 = extractvalue [2 x <32 x float>] %471, 1, !dbg !26
  %636 = shufflevector <32 x float> %634, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %637 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %638 = fadd <4 x float> %636, %637, !dbg !26
  %639 = extractelement <4 x float> %638, i64 0, !dbg !27
  %640 = call float @llvm.amdgcn.exp2.f32(float %639), !dbg !27
  %641 = insertelement <4 x float> poison, float %640, i64 0, !dbg !27
  %642 = extractelement <4 x float> %638, i64 1, !dbg !27
  %643 = call float @llvm.amdgcn.exp2.f32(float %642), !dbg !27
  %644 = fadd float %640, %643, !dbg !28
  %645 = insertelement <4 x float> %641, float %643, i64 1, !dbg !27
  %646 = extractelement <4 x float> %638, i64 2, !dbg !27
  %647 = call float @llvm.amdgcn.exp2.f32(float %646), !dbg !27
  %648 = fadd float %644, %647, !dbg !28
  %649 = insertelement <4 x float> %645, float %647, i64 2, !dbg !27
  %650 = extractelement <4 x float> %638, i64 3, !dbg !27
  %651 = call float @llvm.amdgcn.exp2.f32(float %650), !dbg !27
  %652 = fadd float %648, %651, !dbg !28
  %653 = insertelement <4 x float> %649, float %651, i64 3, !dbg !27
  %math.result117 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %653) #9, !dbg !29, !frisk.register_lifetimes !1
  %654 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !29
  %math.packed118 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %654) #9, !dbg !26, !frisk.register_lifetimes !1
  %655 = shufflevector <32 x float> %634, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %656 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %657 = fadd <4 x float> %655, %656, !dbg !26
  %658 = extractelement <4 x float> %657, i64 0, !dbg !27
  %659 = call float @llvm.amdgcn.exp2.f32(float %658), !dbg !27
  %660 = fadd float %652, %659, !dbg !28
  %661 = insertelement <4 x float> poison, float %659, i64 0, !dbg !27
  %662 = extractelement <4 x float> %657, i64 1, !dbg !27
  %663 = call float @llvm.amdgcn.exp2.f32(float %662), !dbg !27
  %664 = fadd float %660, %663, !dbg !28
  %665 = insertelement <4 x float> %661, float %663, i64 1, !dbg !27
  %666 = extractelement <4 x float> %657, i64 2, !dbg !27
  %667 = call float @llvm.amdgcn.exp2.f32(float %666), !dbg !27
  %668 = fadd float %664, %667, !dbg !28
  %669 = insertelement <4 x float> %665, float %667, i64 2, !dbg !27
  %670 = extractelement <4 x float> %657, i64 3, !dbg !27
  %671 = call float @llvm.amdgcn.exp2.f32(float %670), !dbg !27
  %672 = fadd float %668, %671, !dbg !28
  %673 = insertelement <4 x float> %669, float %671, i64 3, !dbg !27
  %math.result119 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %673) #9, !dbg !29, !frisk.register_lifetimes !1
  %674 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !29
  %math.packed120 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %674) #9, !dbg !26, !frisk.register_lifetimes !1
  %675 = shufflevector <32 x float> %634, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %676 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %677 = fadd <4 x float> %675, %676, !dbg !26
  %678 = extractelement <4 x float> %677, i64 0, !dbg !27
  %679 = call float @llvm.amdgcn.exp2.f32(float %678), !dbg !27
  %680 = fadd float %672, %679, !dbg !28
  %681 = insertelement <4 x float> poison, float %679, i64 0, !dbg !27
  %682 = extractelement <4 x float> %677, i64 1, !dbg !27
  %683 = call float @llvm.amdgcn.exp2.f32(float %682), !dbg !27
  %684 = fadd float %680, %683, !dbg !28
  %685 = insertelement <4 x float> %681, float %683, i64 1, !dbg !27
  %686 = extractelement <4 x float> %677, i64 2, !dbg !27
  %687 = call float @llvm.amdgcn.exp2.f32(float %686), !dbg !27
  %688 = fadd float %684, %687, !dbg !28
  %689 = insertelement <4 x float> %685, float %687, i64 2, !dbg !27
  %690 = extractelement <4 x float> %677, i64 3, !dbg !27
  %691 = call float @llvm.amdgcn.exp2.f32(float %690), !dbg !27
  %692 = fadd float %688, %691, !dbg !28
  %693 = insertelement <4 x float> %689, float %691, i64 3, !dbg !27
  %math.result121 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %693) #9, !dbg !29, !frisk.register_lifetimes !1
  %694 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !29
  %math.packed122 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %694) #9, !dbg !26, !frisk.register_lifetimes !1
  %695 = shufflevector <32 x float> %634, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %696 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %697 = fadd <4 x float> %695, %696, !dbg !26
  %698 = extractelement <4 x float> %697, i64 0, !dbg !27
  %699 = call float @llvm.amdgcn.exp2.f32(float %698), !dbg !27
  %700 = fadd float %692, %699, !dbg !28
  %701 = insertelement <4 x float> poison, float %699, i64 0, !dbg !27
  %702 = extractelement <4 x float> %697, i64 1, !dbg !27
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !27
  %704 = fadd float %700, %703, !dbg !28
  %705 = insertelement <4 x float> %701, float %703, i64 1, !dbg !27
  %706 = extractelement <4 x float> %697, i64 2, !dbg !27
  %707 = call float @llvm.amdgcn.exp2.f32(float %706), !dbg !27
  %708 = fadd float %704, %707, !dbg !28
  %709 = insertelement <4 x float> %705, float %707, i64 2, !dbg !27
  %710 = extractelement <4 x float> %697, i64 3, !dbg !27
  %711 = call float @llvm.amdgcn.exp2.f32(float %710), !dbg !27
  %712 = fadd float %708, %711, !dbg !28
  %713 = insertelement <4 x float> %709, float %711, i64 3, !dbg !27
  %math.result123 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %713) #9, !dbg !29, !frisk.register_lifetimes !1
  %714 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !29
  %math.packed124 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %714) #9, !dbg !26, !frisk.register_lifetimes !1
  %715 = shufflevector <32 x float> %634, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %716 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %717 = fadd <4 x float> %715, %716, !dbg !26
  %718 = extractelement <4 x float> %717, i64 0, !dbg !27
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !27
  %720 = fadd float %712, %719, !dbg !28
  %721 = insertelement <4 x float> poison, float %719, i64 0, !dbg !27
  %722 = extractelement <4 x float> %717, i64 1, !dbg !27
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !27
  %724 = fadd float %720, %723, !dbg !28
  %725 = insertelement <4 x float> %721, float %723, i64 1, !dbg !27
  %726 = extractelement <4 x float> %717, i64 2, !dbg !27
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !27
  %728 = fadd float %724, %727, !dbg !28
  %729 = insertelement <4 x float> %725, float %727, i64 2, !dbg !27
  %730 = extractelement <4 x float> %717, i64 3, !dbg !27
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !27
  %732 = fadd float %728, %731, !dbg !28
  %733 = insertelement <4 x float> %729, float %731, i64 3, !dbg !27
  %math.result125 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %733) #9, !dbg !29, !frisk.register_lifetimes !1
  %734 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !29
  %math.packed126 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %734) #9, !dbg !26, !frisk.register_lifetimes !1
  %735 = shufflevector <32 x float> %634, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %736 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %737 = fadd <4 x float> %735, %736, !dbg !26
  %738 = extractelement <4 x float> %737, i64 0, !dbg !27
  %739 = call float @llvm.amdgcn.exp2.f32(float %738), !dbg !27
  %740 = fadd float %732, %739, !dbg !28
  %741 = insertelement <4 x float> poison, float %739, i64 0, !dbg !27
  %742 = extractelement <4 x float> %737, i64 1, !dbg !27
  %743 = call float @llvm.amdgcn.exp2.f32(float %742), !dbg !27
  %744 = fadd float %740, %743, !dbg !28
  %745 = insertelement <4 x float> %741, float %743, i64 1, !dbg !27
  %746 = extractelement <4 x float> %737, i64 2, !dbg !27
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !27
  %748 = fadd float %744, %747, !dbg !28
  %749 = insertelement <4 x float> %745, float %747, i64 2, !dbg !27
  %750 = extractelement <4 x float> %737, i64 3, !dbg !27
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !27
  %752 = fadd float %748, %751, !dbg !28
  %753 = insertelement <4 x float> %749, float %751, i64 3, !dbg !27
  %math.result127 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %753) #9, !dbg !29, !frisk.register_lifetimes !1
  %754 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !29
  %math.packed128 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %754) #9, !dbg !26, !frisk.register_lifetimes !1
  %755 = shufflevector <32 x float> %634, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %756 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %757 = fadd <4 x float> %755, %756, !dbg !26
  %758 = extractelement <4 x float> %757, i64 0, !dbg !27
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !27
  %760 = fadd float %752, %759, !dbg !28
  %761 = insertelement <4 x float> poison, float %759, i64 0, !dbg !27
  %762 = extractelement <4 x float> %757, i64 1, !dbg !27
  %763 = call float @llvm.amdgcn.exp2.f32(float %762), !dbg !27
  %764 = fadd float %760, %763, !dbg !28
  %765 = insertelement <4 x float> %761, float %763, i64 1, !dbg !27
  %766 = extractelement <4 x float> %757, i64 2, !dbg !27
  %767 = call float @llvm.amdgcn.exp2.f32(float %766), !dbg !27
  %768 = fadd float %764, %767, !dbg !28
  %769 = insertelement <4 x float> %765, float %767, i64 2, !dbg !27
  %770 = extractelement <4 x float> %757, i64 3, !dbg !27
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !27
  %772 = fadd float %768, %771, !dbg !28
  %773 = insertelement <4 x float> %769, float %771, i64 3, !dbg !27
  %math.result129 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %773) #9, !dbg !29, !frisk.register_lifetimes !1
  %774 = fptrunc <4 x float> %math.result129 to <4 x half>, !dbg !29
  %math.packed130 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %774) #9, !dbg !26, !frisk.register_lifetimes !1
  %775 = shufflevector <32 x float> %634, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %776 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %777 = fadd <4 x float> %775, %776, !dbg !26
  %778 = extractelement <4 x float> %777, i64 0, !dbg !27
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !27
  %780 = fadd float %772, %779, !dbg !28
  %781 = insertelement <4 x float> poison, float %779, i64 0, !dbg !27
  %782 = extractelement <4 x float> %777, i64 1, !dbg !27
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !27
  %784 = fadd float %780, %783, !dbg !28
  %785 = insertelement <4 x float> %781, float %783, i64 1, !dbg !27
  %786 = extractelement <4 x float> %777, i64 2, !dbg !27
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !27
  %788 = fadd float %784, %787, !dbg !28
  %789 = insertelement <4 x float> %785, float %787, i64 2, !dbg !27
  %790 = extractelement <4 x float> %777, i64 3, !dbg !27
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !27
  %792 = fadd float %788, %791, !dbg !28
  %793 = insertelement <4 x float> %789, float %791, i64 3, !dbg !27
  %math.result131 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %793) #9, !dbg !29, !frisk.register_lifetimes !1
  %794 = fptrunc <4 x float> %math.result131 to <4 x half>, !dbg !29
  %math.packed132 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %794) #9, !dbg !28, !frisk.register_lifetimes !1
  %795 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %796 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %795), !dbg !28
  %797 = add nuw nsw i32 %796, 64, !dbg !28
  %798 = xor i32 %796, 16, !dbg !28
  %799 = and i32 %797, 192, !dbg !28
  %800 = icmp samesign ult i32 %798, %799, !dbg !28
  %801 = select i1 %800, i32 %798, i32 %796, !dbg !28
  %802 = shl nuw nsw i32 %801, 2, !dbg !28
  %803 = bitcast float %631 to i32, !dbg !28
  %804 = call i32 @llvm.amdgcn.ds.bpermute(i32 %802, i32 %803), !dbg !28
  %805 = bitcast i32 %804 to float, !dbg !28
  %806 = fadd float %631, %805, !dbg !28
  %807 = xor i32 %796, 32, !dbg !28
  %808 = icmp samesign ult i32 %807, %799, !dbg !28
  %809 = select i1 %808, i32 %807, i32 %796, !dbg !28
  %810 = shl nuw nsw i32 %809, 2, !dbg !28
  %811 = bitcast float %806 to i32, !dbg !28
  %812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %810, i32 %811), !dbg !28
  %813 = bitcast i32 %812 to float, !dbg !28
  %814 = fadd float %806, %813, !dbg !28
  %815 = extractvalue [2 x <1 x float>] %28, 0, !dbg !28
  %816 = extractelement <1 x float> %815, i64 0, !dbg !28
  %817 = fadd float %814, %816, !dbg !28
  %818 = insertelement <1 x float> poison, float %817, i64 0, !dbg !15
  %819 = insertvalue [2 x <1 x float>] poison, <1 x float> %818, 0, !dbg !15
  %820 = bitcast float %792 to i32, !dbg !28
  %821 = call i32 @llvm.amdgcn.ds.bpermute(i32 %802, i32 %820), !dbg !28
  %822 = bitcast i32 %821 to float, !dbg !28
  %823 = fadd float %792, %822, !dbg !28
  %824 = bitcast float %823 to i32, !dbg !28
  %825 = call i32 @llvm.amdgcn.ds.bpermute(i32 %810, i32 %824), !dbg !28
  %826 = bitcast i32 %825 to float, !dbg !28
  %827 = fadd float %823, %826, !dbg !28
  %828 = extractvalue [2 x <1 x float>] %28, 1, !dbg !28
  %829 = extractelement <1 x float> %828, i64 0, !dbg !28
  %830 = fadd float %827, %829, !dbg !28
  %831 = insertelement <1 x float> poison, float %830, i64 0, !dbg !15
  %832 = insertvalue [2 x <1 x float>] %819, <1 x float> %831, 1, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread360 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !18, !range !34, !frisk.register_lifetimes !1
  %address.local361 = shl nuw nsw i32 %address.thread360, 1, !dbg !18
  %address.local362 = and i32 %address.local361, 8, !dbg !18
  %address.local363 = shl nuw nsw i32 %address.thread360, 4, !dbg !18
  %address.local364 = and i32 %address.local363, 176, !dbg !18
  %address.local365 = or disjoint i32 %address.local362, %address.local364, !dbg !18
  %address.local366 = shl nuw nsw i32 %address.thread360, 6, !dbg !18
  %address.local367 = and i32 %address.local366, 3072, !dbg !18
  %address.local368 = or disjoint i32 %address.local365, %address.local367, !dbg !18
  %address.local369 = zext nneg i32 %address.local368 to i64, !dbg !18
  %address.local370 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local369, !dbg !18
  %833 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local370, i32 0) #0, !dbg !18
  %834 = bitcast <4 x i32> %833 to <8 x half>, !dbg !18
  %address.local371 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local369, !dbg !18
  %address.local372 = getelementptr i8, ptr addrspace(3) %address.local371, i64 128, !dbg !18
  %835 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local372, i32 0) #0, !dbg !18
  %836 = bitcast <4 x i32> %835 to <8 x half>, !dbg !18
  %address.local373 = shl nuw nsw i32 %address.thread360, 1, !dbg !18
  %address.local374 = and i32 %address.local373, 8, !dbg !18
  %address.local375 = shl nuw nsw i32 %address.thread360, 4, !dbg !18
  %address.local376 = and i32 %address.local375, 176, !dbg !18
  %address.local377 = or disjoint i32 %address.local374, %address.local376, !dbg !18
  %address.local378 = shl nuw nsw i32 %address.thread360, 6, !dbg !18
  %address.local379 = and i32 %address.local378, 3072, !dbg !18
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !18
  %address.local381 = zext nneg i32 %address.local380 to i64, !dbg !18
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !18
  %address.local383 = getelementptr i8, ptr addrspace(3) %address.local382, i64 512, !dbg !18
  %837 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local383, i32 0) #0, !dbg !18
  %838 = bitcast <4 x i32> %837 to <8 x half>, !dbg !18
  %address.local384 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !18
  %address.local385 = getelementptr i8, ptr addrspace(3) %address.local384, i64 640, !dbg !18
  %839 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local385, i32 0) #0, !dbg !18
  %840 = bitcast <4 x i32> %839 to <8 x half>, !dbg !18
  %address.local386 = shl nuw nsw i32 %address.thread360, 1, !dbg !18
  %address.local387 = and i32 %address.local386, 8, !dbg !18
  %address.local388 = shl nuw nsw i32 %address.thread360, 4, !dbg !18
  %address.local389 = and i32 %address.local388, 176, !dbg !18
  %address.local390 = or disjoint i32 %address.local387, %address.local389, !dbg !18
  %address.local391 = shl nuw nsw i32 %address.thread360, 6, !dbg !18
  %address.local392 = and i32 %address.local391, 3072, !dbg !18
  %address.local393 = or disjoint i32 %address.local390, %address.local392, !dbg !18
  %address.local394 = zext nneg i32 %address.local393 to i64, !dbg !18
  %address.local395 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local394, !dbg !18
  %address.local396 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1024, !dbg !18
  %841 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local396, i32 0) #0, !dbg !18
  %842 = bitcast <4 x i32> %841 to <8 x half>, !dbg !18
  %address.local397 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local394, !dbg !18
  %address.local398 = getelementptr i8, ptr addrspace(3) %address.local397, i64 1152, !dbg !18
  %843 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local398, i32 0) #0, !dbg !18
  %844 = bitcast <4 x i32> %843 to <8 x half>, !dbg !18
  %address.local399 = shl nuw nsw i32 %address.thread360, 1, !dbg !18
  %address.local400 = and i32 %address.local399, 8, !dbg !18
  %address.local401 = shl nuw nsw i32 %address.thread360, 4, !dbg !18
  %address.local402 = and i32 %address.local401, 176, !dbg !18
  %address.local403 = or disjoint i32 %address.local400, %address.local402, !dbg !18
  %address.local404 = shl nuw nsw i32 %address.thread360, 6, !dbg !18
  %address.local405 = and i32 %address.local404, 3072, !dbg !18
  %address.local406 = or disjoint i32 %address.local403, %address.local405, !dbg !18
  %address.local407 = zext nneg i32 %address.local406 to i64, !dbg !18
  %address.local408 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local407, !dbg !18
  %address.local409 = getelementptr i8, ptr addrspace(3) %address.local408, i64 1536, !dbg !18
  %845 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local409, i32 0) #0, !dbg !18
  %846 = bitcast <4 x i32> %845 to <8 x half>, !dbg !18
  %address.local410 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local407, !dbg !18
  %address.local411 = getelementptr i8, ptr addrspace(3) %address.local410, i64 1664, !dbg !18
  %847 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local411, i32 0) #0, !dbg !18
  %848 = bitcast <4 x i32> %847 to <8 x half>, !dbg !18
  %address.local412 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local369, !dbg !18
  %address.local413 = getelementptr i8, ptr addrspace(3) %address.local412, i64 8192, !dbg !18
  %849 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local413, i32 0) #0, !dbg !18
  %850 = bitcast <4 x i32> %849 to <8 x half>, !dbg !18
  %address.local414 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local369, !dbg !18
  %address.local415 = getelementptr i8, ptr addrspace(3) %address.local414, i64 8320, !dbg !18
  %851 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local415, i32 0) #0, !dbg !18
  %852 = bitcast <4 x i32> %851 to <8 x half>, !dbg !18
  %address.local416 = shl nuw nsw i32 %address.thread360, 1, !dbg !18
  %address.local417 = and i32 %address.local416, 8, !dbg !18
  %address.local418 = shl nuw nsw i32 %address.thread360, 4, !dbg !18
  %address.local419 = and i32 %address.local418, 176, !dbg !18
  %address.local420 = or disjoint i32 %address.local417, %address.local419, !dbg !18
  %address.local421 = shl nuw nsw i32 %address.thread360, 6, !dbg !18
  %address.local422 = and i32 %address.local421, 3072, !dbg !18
  %address.local423 = or disjoint i32 %address.local420, %address.local422, !dbg !18
  %address.local424 = or disjoint i32 %address.local423, 4096, !dbg !18
  %address.local425 = zext nneg i32 %address.local424 to i64, !dbg !18
  %address.local426 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local425, !dbg !18
  %address.local427 = getelementptr i8, ptr addrspace(3) %address.local426, i64 512, !dbg !18
  %853 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local427, i32 0) #0, !dbg !18
  %854 = bitcast <4 x i32> %853 to <8 x half>, !dbg !18
  %address.local428 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local425, !dbg !18
  %address.local429 = getelementptr i8, ptr addrspace(3) %address.local428, i64 640, !dbg !18
  %855 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local429, i32 0) #0, !dbg !18
  %856 = bitcast <4 x i32> %855 to <8 x half>, !dbg !18
  %address.local430 = shl nuw nsw i32 %address.thread360, 1, !dbg !18
  %address.local431 = and i32 %address.local430, 8, !dbg !18
  %address.local432 = shl nuw nsw i32 %address.thread360, 4, !dbg !18
  %address.local433 = and i32 %address.local432, 176, !dbg !18
  %address.local434 = or disjoint i32 %address.local431, %address.local433, !dbg !18
  %address.local435 = shl nuw nsw i32 %address.thread360, 6, !dbg !18
  %address.local436 = and i32 %address.local435, 3072, !dbg !18
  %address.local437 = or disjoint i32 %address.local434, %address.local436, !dbg !18
  %address.local438 = or disjoint i32 %address.local437, 4096, !dbg !18
  %address.local439 = zext nneg i32 %address.local438 to i64, !dbg !18
  %address.local440 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local439, !dbg !18
  %address.local441 = getelementptr i8, ptr addrspace(3) %address.local440, i64 1024, !dbg !18
  %857 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local441, i32 0) #0, !dbg !18
  %858 = bitcast <4 x i32> %857 to <8 x half>, !dbg !18
  %address.local442 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local439, !dbg !18
  %address.local443 = getelementptr i8, ptr addrspace(3) %address.local442, i64 1152, !dbg !18
  %859 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local443, i32 0) #0, !dbg !18
  %860 = bitcast <4 x i32> %859 to <8 x half>, !dbg !18
  %address.local444 = shl nuw nsw i32 %address.thread360, 1, !dbg !18
  %address.local445 = and i32 %address.local444, 8, !dbg !18
  %address.local446 = shl nuw nsw i32 %address.thread360, 4, !dbg !18
  %address.local447 = and i32 %address.local446, 176, !dbg !18
  %address.local448 = or disjoint i32 %address.local445, %address.local447, !dbg !18
  %address.local449 = shl nuw nsw i32 %address.thread360, 6, !dbg !18
  %address.local450 = and i32 %address.local449, 3072, !dbg !18
  %address.local451 = or disjoint i32 %address.local448, %address.local450, !dbg !18
  %address.local452 = or disjoint i32 %address.local451, 4096, !dbg !18
  %address.local453 = zext nneg i32 %address.local452 to i64, !dbg !18
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !18
  %address.local455 = getelementptr i8, ptr addrspace(3) %address.local454, i64 1536, !dbg !18
  %861 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local455, i32 0) #0, !dbg !18
  %862 = bitcast <4 x i32> %861 to <8 x half>, !dbg !18
  %address.local456 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !18
  %address.local457 = getelementptr i8, ptr addrspace(3) %address.local456, i64 1664, !dbg !18
  %863 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local457, i32 0) #0, !dbg !18
  %864 = bitcast <4 x i32> %863 to <8 x half>, !dbg !18
  %865 = add nsw i64 %32, %8, !dbg !17
  %866 = shl i64 %865, 19, !dbg !17
  %867 = and i64 %866, 16252928, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %868 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %867, !dbg !17
  %869 = getelementptr [2 x i8], ptr addrspace(1) %868, i64 %236, !dbg !17
  %870 = getelementptr i8, ptr addrspace(1) %869, i64 128, !dbg !17
  %address.local458 = zext nneg i32 %address.thread360 to i64, !dbg !13
  %address.local459 = shl nuw nsw i64 %address.local458, 3, !dbg !12
  %address.local460 = and i64 %address.local459, 1536, !dbg !12
  %address.local461 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local460, !dbg !17
  %address.local462 = ptrtoint ptr addrspace(3) %address.local461 to i64, !dbg !17
  %address.local463 = trunc i64 %address.local462 to i32, !dbg !17
  %871 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local463), !dbg !17
  %872 = zext i32 %871 to i64, !dbg !17
  %873 = inttoptr i64 %872 to ptr addrspace(3), !dbg !17
  %address.local464 = lshr i64 %address.local458, 3, !dbg !17
  %address.local465 = and i64 %address.local464, 30, !dbg !17
  %address.local466 = and i64 %address.local458, 1, !dbg !17
  %address.local467 = or disjoint i64 %address.local465, %address.local466, !dbg !17
  %address.local468 = shl nuw nsw i64 %address.local467, 7, !dbg !17
  %address.local469 = shl nuw nsw i64 %address.local458, 2, !dbg !17
  %address.local470 = and i64 %address.local469, 56, !dbg !17
  %address.local471 = or disjoint i64 %address.local468, %address.local470, !dbg !17
  %address.local472 = getelementptr [2 x i8], ptr addrspace(1) %870, i64 %address.local471, !dbg !17
  %874 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %875 = ptrtoint ptr addrspace(1) %address.local472 to i64, !dbg !17
  %876 = sub i64 %875, %874, !dbg !17
  %dma.buffer.offset727 = trunc i64 %876 to i32, !dbg !17
  %877 = trunc i64 %874 to i32, !dbg !17
  %878 = insertelement <4 x i32> poison, i32 %877, i64 0, !dbg !17
  %879 = lshr i64 %874, 32, !dbg !17
  %880 = trunc i64 %879 to i32, !dbg !17
  %881 = insertelement <4 x i32> %878, i32 %880, i64 1, !dbg !17
  %882 = insertelement <4 x i32> %881, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc728 = insertelement <4 x i32> %882, i32 131072, i64 3, !dbg !17
  %883 = ptrtoint ptr addrspace(3) %873 to i32, !dbg !17
  %dma.buffer.lds729 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %883), !dbg !17
  %884 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds729, i32 %dma.buffer.offset727, <4 x i32> %dma.buffer.rsrc728) #10, !dbg !17
  %address.local473 = or disjoint i64 %address.local458, 256, !dbg !12
  %address.local474 = shl nuw nsw i64 %address.local473, 3, !dbg !12
  %address.local475 = and i64 %address.local474, 3584, !dbg !17
  %address.local476 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local475, !dbg !17
  %address.local477 = ptrtoint ptr addrspace(3) %address.local476 to i64, !dbg !17
  %address.local478 = trunc i64 %address.local477 to i32, !dbg !17
  %885 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local478), !dbg !17
  %886 = zext i32 %885 to i64, !dbg !17
  %887 = inttoptr i64 %886 to ptr addrspace(3), !dbg !17
  %address.local479 = lshr i64 %address.local473, 3, !dbg !17
  %address.local480 = and i64 %address.local479, 62, !dbg !17
  %address.local481 = and i64 %address.local458, 1, !dbg !17
  %address.local482 = or disjoint i64 %address.local480, %address.local481, !dbg !17
  %address.local483 = shl nuw nsw i64 %address.local482, 7, !dbg !17
  %address.local484 = shl nuw nsw i64 %address.local458, 2, !dbg !17
  %address.local485 = and i64 %address.local484, 56, !dbg !17
  %address.local486 = or disjoint i64 %address.local483, %address.local485, !dbg !17
  %address.local487 = getelementptr [2 x i8], ptr addrspace(1) %870, i64 %address.local486, !dbg !17
  %888 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %889 = ptrtoint ptr addrspace(1) %address.local487 to i64, !dbg !17
  %890 = sub i64 %889, %888, !dbg !17
  %dma.buffer.offset724 = trunc i64 %890 to i32, !dbg !17
  %891 = trunc i64 %888 to i32, !dbg !17
  %892 = insertelement <4 x i32> poison, i32 %891, i64 0, !dbg !17
  %893 = lshr i64 %888, 32, !dbg !17
  %894 = trunc i64 %893 to i32, !dbg !17
  %895 = insertelement <4 x i32> %892, i32 %894, i64 1, !dbg !17
  %896 = insertelement <4 x i32> %895, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc725 = insertelement <4 x i32> %896, i32 131072, i64 3, !dbg !17
  %897 = ptrtoint ptr addrspace(3) %887 to i32, !dbg !17
  %dma.buffer.lds726 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %897), !dbg !17
  %898 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds726, i32 %dma.buffer.offset724, <4 x i32> %dma.buffer.rsrc725) #10, !dbg !17
  %address.local488 = or disjoint i64 %address.local458, 512, !dbg !17
  %address.local489 = shl nuw nsw i64 %address.local488, 3, !dbg !17
  %address.local490 = and i64 %address.local489, 5632, !dbg !17
  %address.local491 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local490, !dbg !17
  %address.local492 = ptrtoint ptr addrspace(3) %address.local491 to i64, !dbg !17
  %address.local493 = trunc i64 %address.local492 to i32, !dbg !17
  %899 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local493), !dbg !17
  %900 = zext i32 %899 to i64, !dbg !17
  %901 = inttoptr i64 %900 to ptr addrspace(3), !dbg !17
  %address.local494 = lshr i64 %address.local488, 3, !dbg !17
  %address.local495 = and i64 %address.local494, 94, !dbg !17
  %address.local496 = and i64 %address.local458, 1, !dbg !17
  %address.local497 = or disjoint i64 %address.local495, %address.local496, !dbg !17
  %address.local498 = shl nuw nsw i64 %address.local497, 7, !dbg !17
  %address.local499 = shl nuw nsw i64 %address.local458, 2, !dbg !17
  %address.local500 = and i64 %address.local499, 56, !dbg !17
  %address.local501 = or disjoint i64 %address.local498, %address.local500, !dbg !17
  %address.local502 = getelementptr [2 x i8], ptr addrspace(1) %870, i64 %address.local501, !dbg !17
  %902 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %903 = ptrtoint ptr addrspace(1) %address.local502 to i64, !dbg !17
  %904 = sub i64 %903, %902, !dbg !17
  %dma.buffer.offset721 = trunc i64 %904 to i32, !dbg !17
  %905 = trunc i64 %902 to i32, !dbg !17
  %906 = insertelement <4 x i32> poison, i32 %905, i64 0, !dbg !17
  %907 = lshr i64 %902, 32, !dbg !17
  %908 = trunc i64 %907 to i32, !dbg !17
  %909 = insertelement <4 x i32> %906, i32 %908, i64 1, !dbg !17
  %910 = insertelement <4 x i32> %909, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc722 = insertelement <4 x i32> %910, i32 131072, i64 3, !dbg !17
  %911 = ptrtoint ptr addrspace(3) %901 to i32, !dbg !17
  %dma.buffer.lds723 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %911), !dbg !17
  %912 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds723, i32 %dma.buffer.offset721, <4 x i32> %dma.buffer.rsrc722) #10, !dbg !17
  %address.local503 = or disjoint i64 %address.local458, 768, !dbg !17
  %address.local504 = shl nuw nsw i64 %address.local503, 3, !dbg !17
  %address.local505 = and i64 %address.local504, 7680, !dbg !17
  %address.local506 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local505, !dbg !17
  %address.local507 = ptrtoint ptr addrspace(3) %address.local506 to i64, !dbg !17
  %address.local508 = trunc i64 %address.local507 to i32, !dbg !17
  %913 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local508), !dbg !17
  %914 = zext i32 %913 to i64, !dbg !17
  %915 = inttoptr i64 %914 to ptr addrspace(3), !dbg !17
  %address.local509 = lshr i64 %address.local503, 3, !dbg !17
  %address.local510 = and i64 %address.local509, 126, !dbg !17
  %address.local511 = and i64 %address.local458, 1, !dbg !17
  %address.local512 = or disjoint i64 %address.local510, %address.local511, !dbg !17
  %address.local513 = shl nuw nsw i64 %address.local512, 7, !dbg !17
  %address.local514 = shl nuw nsw i64 %address.local458, 2, !dbg !17
  %address.local515 = and i64 %address.local514, 56, !dbg !17
  %address.local516 = or disjoint i64 %address.local513, %address.local515, !dbg !17
  %address.local517 = getelementptr [2 x i8], ptr addrspace(1) %870, i64 %address.local516, !dbg !17
  %916 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %917 = ptrtoint ptr addrspace(1) %address.local517 to i64, !dbg !17
  %918 = sub i64 %917, %916, !dbg !17
  %dma.buffer.offset718 = trunc i64 %918 to i32, !dbg !17
  %919 = trunc i64 %916 to i32, !dbg !17
  %920 = insertelement <4 x i32> poison, i32 %919, i64 0, !dbg !17
  %921 = lshr i64 %916, 32, !dbg !17
  %922 = trunc i64 %921 to i32, !dbg !17
  %923 = insertelement <4 x i32> %920, i32 %922, i64 1, !dbg !17
  %924 = insertelement <4 x i32> %923, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc719 = insertelement <4 x i32> %924, i32 131072, i64 3, !dbg !17
  %925 = ptrtoint ptr addrspace(3) %915 to i32, !dbg !17
  %dma.buffer.lds720 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %925), !dbg !17
  %926 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds720, i32 %dma.buffer.offset718, <4 x i32> %dma.buffer.rsrc719) #10, !dbg !17
  %927 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %928 = extractvalue [2 x <16 x float>] %27, 0, !dbg !30
  %929 = shufflevector <16 x float> %928, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %930 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %927, <4 x float> %929) #11, !dbg !18
  %931 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %932 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %931, <4 x float> %930) #11, !dbg !18
  %933 = shufflevector <8 x half> %842, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %934 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %933, <4 x float> %932) #11, !dbg !18
  %935 = shufflevector <8 x half> %846, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %936 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %935, <4 x float> %934) #11, !dbg !18
  %937 = shufflevector <8 x half> %850, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %938 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %937, <4 x float> %936) #11, !dbg !18
  %939 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %940 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %939, <4 x float> %938) #11, !dbg !18
  %941 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %942 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %941, <4 x float> %940) #11, !dbg !18
  %943 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %944 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %943, <4 x float> %942) #11, !dbg !18
  %945 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %946 = shufflevector <16 x float> %928, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %947 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %945, <4 x float> %946) #11, !dbg !18
  %948 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %949 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %948, <4 x float> %947) #11, !dbg !18
  %950 = shufflevector <8 x half> %842, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %951 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %950, <4 x float> %949) #11, !dbg !18
  %952 = shufflevector <8 x half> %846, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %953 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %952, <4 x float> %951) #11, !dbg !18
  %954 = shufflevector <8 x half> %850, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %955 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %954, <4 x float> %953) #11, !dbg !18
  %956 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %957 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %956, <4 x float> %955) #11, !dbg !18
  %958 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %959 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %958, <4 x float> %957) #11, !dbg !18
  %960 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %961 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %960, <4 x float> %959) #11, !dbg !18
  %962 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %963 = shufflevector <16 x float> %928, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %964 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %962, <4 x float> %963) #11, !dbg !18
  %965 = shufflevector <8 x half> %840, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %966 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %965, <4 x float> %964) #11, !dbg !18
  %967 = shufflevector <8 x half> %844, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %968 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %967, <4 x float> %966) #11, !dbg !18
  %969 = shufflevector <8 x half> %848, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %970 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %969, <4 x float> %968) #11, !dbg !18
  %971 = shufflevector <8 x half> %852, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %972 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %971, <4 x float> %970) #11, !dbg !18
  %973 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %974 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %973, <4 x float> %972) #11, !dbg !18
  %975 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %976 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %975, <4 x float> %974) #11, !dbg !18
  %977 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %978 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %977, <4 x float> %976) #11, !dbg !18
  %979 = shufflevector <4 x float> %978, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %980 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %981 = shufflevector <16 x float> %928, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %982 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %980, <4 x float> %981) #11, !dbg !18
  %983 = shufflevector <8 x half> %840, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %984 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %983, <4 x float> %982) #11, !dbg !18
  %985 = shufflevector <8 x half> %844, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %986 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %985, <4 x float> %984) #11, !dbg !18
  %987 = shufflevector <8 x half> %848, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %988 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %987, <4 x float> %986) #11, !dbg !18
  %989 = shufflevector <8 x half> %852, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %990 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %989, <4 x float> %988) #11, !dbg !18
  %991 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %992 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %991, <4 x float> %990) #11, !dbg !18
  %993 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %994 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %993, <4 x float> %992) #11, !dbg !18
  %995 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %996 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %995, <4 x float> %994) #11, !dbg !18
  %997 = shufflevector <4 x float> %996, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %998 = extractvalue [2 x <16 x float>] %27, 1, !dbg !30
  %999 = shufflevector <16 x float> %998, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1000 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %927, <4 x float> %999) #11, !dbg !18
  %1001 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %931, <4 x float> %1000) #11, !dbg !18
  %1002 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %933, <4 x float> %1001) #11, !dbg !18
  %1003 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %935, <4 x float> %1002) #11, !dbg !18
  %1004 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %937, <4 x float> %1003) #11, !dbg !18
  %1005 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %939, <4 x float> %1004) #11, !dbg !18
  %1006 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %941, <4 x float> %1005) #11, !dbg !18
  %1007 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %943, <4 x float> %1006) #11, !dbg !18
  %1008 = shufflevector <16 x float> %998, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1009 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %945, <4 x float> %1008) #11, !dbg !18
  %1010 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %948, <4 x float> %1009) #11, !dbg !18
  %1011 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %950, <4 x float> %1010) #11, !dbg !18
  %1012 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %952, <4 x float> %1011) #11, !dbg !18
  %1013 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %954, <4 x float> %1012) #11, !dbg !18
  %1014 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %956, <4 x float> %1013) #11, !dbg !18
  %1015 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %958, <4 x float> %1014) #11, !dbg !18
  %1016 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %960, <4 x float> %1015) #11, !dbg !18
  %1017 = shufflevector <16 x float> %998, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %1018 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %962, <4 x float> %1017) #11, !dbg !18
  %1019 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %965, <4 x float> %1018) #11, !dbg !18
  %1020 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %967, <4 x float> %1019) #11, !dbg !18
  %1021 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %969, <4 x float> %1020) #11, !dbg !18
  %1022 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %971, <4 x float> %1021) #11, !dbg !18
  %1023 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %973, <4 x float> %1022) #11, !dbg !18
  %1024 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %975, <4 x float> %1023) #11, !dbg !18
  %1025 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %977, <4 x float> %1024) #11, !dbg !18
  %1026 = shufflevector <4 x float> %1025, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1027 = shufflevector <16 x float> %998, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %1028 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %980, <4 x float> %1027) #11, !dbg !18
  %1029 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %983, <4 x float> %1028) #11, !dbg !18
  %1030 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %985, <4 x float> %1029) #11, !dbg !18
  %1031 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %987, <4 x float> %1030) #11, !dbg !18
  %1032 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %989, <4 x float> %1031) #11, !dbg !18
  %1033 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %991, <4 x float> %1032) #11, !dbg !18
  %1034 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %993, <4 x float> %1033) #11, !dbg !18
  %1035 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %995, <4 x float> %1034) #11, !dbg !18
  %1036 = shufflevector <4 x float> %1035, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1037 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local370, i32 0) #0, !dbg !18
  %1038 = bitcast <4 x i32> %1037 to <8 x half>, !dbg !18
  %1039 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local372, i32 0) #0, !dbg !18
  %1040 = bitcast <4 x i32> %1039 to <8 x half>, !dbg !18
  %1041 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local383, i32 0) #0, !dbg !18
  %1042 = bitcast <4 x i32> %1041 to <8 x half>, !dbg !18
  %1043 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local385, i32 0) #0, !dbg !18
  %1044 = bitcast <4 x i32> %1043 to <8 x half>, !dbg !18
  %1045 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local396, i32 0) #0, !dbg !18
  %1046 = bitcast <4 x i32> %1045 to <8 x half>, !dbg !18
  %1047 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local398, i32 0) #0, !dbg !18
  %1048 = bitcast <4 x i32> %1047 to <8 x half>, !dbg !18
  %1049 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local409, i32 0) #0, !dbg !18
  %1050 = bitcast <4 x i32> %1049 to <8 x half>, !dbg !18
  %1051 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local411, i32 0) #0, !dbg !18
  %1052 = bitcast <4 x i32> %1051 to <8 x half>, !dbg !18
  %1053 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local413, i32 0) #0, !dbg !18
  %1054 = bitcast <4 x i32> %1053 to <8 x half>, !dbg !18
  %1055 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local415, i32 0) #0, !dbg !18
  %1056 = bitcast <4 x i32> %1055 to <8 x half>, !dbg !18
  %1057 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local427, i32 0) #0, !dbg !18
  %1058 = bitcast <4 x i32> %1057 to <8 x half>, !dbg !18
  %1059 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local429, i32 0) #0, !dbg !18
  %1060 = bitcast <4 x i32> %1059 to <8 x half>, !dbg !18
  %1061 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local441, i32 0) #0, !dbg !18
  %1062 = bitcast <4 x i32> %1061 to <8 x half>, !dbg !18
  %1063 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local443, i32 0) #0, !dbg !18
  %1064 = bitcast <4 x i32> %1063 to <8 x half>, !dbg !18
  %1065 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local455, i32 0) #0, !dbg !18
  %1066 = bitcast <4 x i32> %1065 to <8 x half>, !dbg !18
  %1067 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local457, i32 0) #0, !dbg !18
  %1068 = bitcast <4 x i32> %1067 to <8 x half>, !dbg !18
  %1069 = shufflevector <8 x half> %1038, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1070 = extractvalue [2 x <16 x float>] %29, 0, !dbg !30
  %1071 = shufflevector <16 x float> %1070, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1072 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1069, <4 x float> %1071) #11, !dbg !18
  %1073 = shufflevector <8 x half> %1042, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1074 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1073, <4 x float> %1072) #11, !dbg !18
  %1075 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1076 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1075, <4 x float> %1074) #11, !dbg !18
  %1077 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1078 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1077, <4 x float> %1076) #11, !dbg !18
  %1079 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1080 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1079, <4 x float> %1078) #11, !dbg !18
  %1081 = shufflevector <8 x half> %1058, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1082 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1081, <4 x float> %1080) #11, !dbg !18
  %1083 = shufflevector <8 x half> %1062, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1084 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1083, <4 x float> %1082) #11, !dbg !18
  %1085 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1086 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1085, <4 x float> %1084) #11, !dbg !18
  %1087 = shufflevector <8 x half> %1038, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1088 = shufflevector <16 x float> %1070, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1089 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1087, <4 x float> %1088) #11, !dbg !18
  %1090 = shufflevector <8 x half> %1042, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1091 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1090, <4 x float> %1089) #11, !dbg !18
  %1092 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1093 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1092, <4 x float> %1091) #11, !dbg !18
  %1094 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1095 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1094, <4 x float> %1093) #11, !dbg !18
  %1096 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1097 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1096, <4 x float> %1095) #11, !dbg !18
  %1098 = shufflevector <8 x half> %1058, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1099 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1098, <4 x float> %1097) #11, !dbg !18
  %1100 = shufflevector <8 x half> %1062, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1101 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1100, <4 x float> %1099) #11, !dbg !18
  %1102 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1103 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1102, <4 x float> %1101) #11, !dbg !18
  %1104 = shufflevector <8 x half> %1040, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1105 = shufflevector <16 x float> %1070, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %1106 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1104, <4 x float> %1105) #11, !dbg !18
  %1107 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1108 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1107, <4 x float> %1106) #11, !dbg !18
  %1109 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1110 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1109, <4 x float> %1108) #11, !dbg !18
  %1111 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1112 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1111, <4 x float> %1110) #11, !dbg !18
  %1113 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1114 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1113, <4 x float> %1112) #11, !dbg !18
  %1115 = shufflevector <8 x half> %1060, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1116 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1115, <4 x float> %1114) #11, !dbg !18
  %1117 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1118 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1117, <4 x float> %1116) #11, !dbg !18
  %1119 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1120 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1119, <4 x float> %1118) #11, !dbg !18
  %1121 = shufflevector <4 x float> %1120, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1122 = shufflevector <8 x half> %1040, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1123 = shufflevector <16 x float> %1070, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %1124 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1122, <4 x float> %1123) #11, !dbg !18
  %1125 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1126 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1125, <4 x float> %1124) #11, !dbg !18
  %1127 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1128 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1127, <4 x float> %1126) #11, !dbg !18
  %1129 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1130 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1129, <4 x float> %1128) #11, !dbg !18
  %1131 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1132 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1131, <4 x float> %1130) #11, !dbg !18
  %1133 = shufflevector <8 x half> %1060, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1134 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1133, <4 x float> %1132) #11, !dbg !18
  %1135 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1136 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1135, <4 x float> %1134) #11, !dbg !18
  %1137 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1138 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1137, <4 x float> %1136) #11, !dbg !18
  %1139 = shufflevector <4 x float> %1138, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1140 = extractvalue [2 x <16 x float>] %29, 1, !dbg !30
  %1141 = shufflevector <16 x float> %1140, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1142 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1069, <4 x float> %1141) #11, !dbg !18
  %1143 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1073, <4 x float> %1142) #11, !dbg !18
  %1144 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1075, <4 x float> %1143) #11, !dbg !18
  %1145 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1077, <4 x float> %1144) #11, !dbg !18
  %1146 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1079, <4 x float> %1145) #11, !dbg !18
  %1147 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1081, <4 x float> %1146) #11, !dbg !18
  %1148 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1083, <4 x float> %1147) #11, !dbg !18
  %1149 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1085, <4 x float> %1148) #11, !dbg !18
  %1150 = shufflevector <16 x float> %1140, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1151 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1087, <4 x float> %1150) #11, !dbg !18
  %1152 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1090, <4 x float> %1151) #11, !dbg !18
  %1153 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1092, <4 x float> %1152) #11, !dbg !18
  %1154 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1094, <4 x float> %1153) #11, !dbg !18
  %1155 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1096, <4 x float> %1154) #11, !dbg !18
  %1156 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1098, <4 x float> %1155) #11, !dbg !18
  %1157 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1100, <4 x float> %1156) #11, !dbg !18
  %1158 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1102, <4 x float> %1157) #11, !dbg !18
  %1159 = shufflevector <16 x float> %1140, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %1160 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1104, <4 x float> %1159) #11, !dbg !18
  %1161 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1107, <4 x float> %1160) #11, !dbg !18
  %1162 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1109, <4 x float> %1161) #11, !dbg !18
  %1163 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1111, <4 x float> %1162) #11, !dbg !18
  %1164 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1113, <4 x float> %1163) #11, !dbg !18
  %1165 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1115, <4 x float> %1164) #11, !dbg !18
  %1166 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1117, <4 x float> %1165) #11, !dbg !18
  %1167 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1119, <4 x float> %1166) #11, !dbg !18
  %1168 = shufflevector <4 x float> %1167, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1169 = shufflevector <16 x float> %1140, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %1170 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1122, <4 x float> %1169) #11, !dbg !18
  %1171 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1125, <4 x float> %1170) #11, !dbg !18
  %1172 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1127, <4 x float> %1171) #11, !dbg !18
  %1173 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1129, <4 x float> %1172) #11, !dbg !18
  %1174 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1131, <4 x float> %1173) #11, !dbg !18
  %1175 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1133, <4 x float> %1174) #11, !dbg !18
  %1176 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1135, <4 x float> %1175) #11, !dbg !18
  %1177 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1137, <4 x float> %1176) #11, !dbg !18
  %1178 = shufflevector <4 x float> %1177, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1179 = shufflevector <4 x float> %944, <4 x float> %961, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1180 = shufflevector <16 x float> %1179, <16 x float> %979, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1181 = shufflevector <16 x float> %1180, <16 x float> %997, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !15
  %1182 = insertvalue [2 x <16 x float>] poison, <16 x float> %1181, 0, !dbg !15
  %1183 = shufflevector <4 x float> %1007, <4 x float> %1016, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1184 = shufflevector <16 x float> %1183, <16 x float> %1026, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1185 = shufflevector <16 x float> %1184, <16 x float> %1036, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !15
  %1186 = insertvalue [2 x <16 x float>] %1182, <16 x float> %1185, 1, !dbg !15
  %1187 = shufflevector <4 x float> %1086, <4 x float> %1103, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1188 = shufflevector <16 x float> %1187, <16 x float> %1121, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1189 = shufflevector <16 x float> %1188, <16 x float> %1139, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !15
  %1190 = insertvalue [2 x <16 x float>] poison, <16 x float> %1189, 0, !dbg !15
  %1191 = shufflevector <4 x float> %1149, <4 x float> %1158, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1192 = shufflevector <16 x float> %1191, <16 x float> %1168, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1193 = shufflevector <16 x float> %1192, <16 x float> %1178, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !15
  %1194 = insertvalue [2 x <16 x float>] %1190, <16 x float> %1193, 1, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1195 = add nuw nsw i64 %26, 1, !dbg !15
  br label %25, !dbg !15

1196:                                             ; preds = %25
  %1197 = and i32 %19, 16, !dbg !15
  %.not2 = icmp eq i32 %1197, 0, !dbg !15
  %1198 = and i32 %19, 32, !dbg !15
  %.not3 = icmp eq i32 %1198, 0, !dbg !15
  %1199 = extractvalue [2 x <16 x float>] %29, 0, !dbg !15
  %1200 = extractelement <16 x float> %1199, i64 0, !dbg !15
  %1201 = extractelement <16 x float> %1199, i64 1, !dbg !15
  %1202 = extractelement <16 x float> %1199, i64 2, !dbg !15
  %1203 = extractelement <16 x float> %1199, i64 3, !dbg !15
  %1204 = select i1 %.not2, float %1201, float %1200, !dbg !15
  %1205 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !15
  %1206 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1205), !dbg !15
  %1207 = add nuw nsw i32 %1206, 64, !dbg !15
  %1208 = xor i32 %1206, 16, !dbg !15
  %1209 = and i32 %1207, 192, !dbg !15
  %1210 = icmp samesign ult i32 %1208, %1209, !dbg !15
  %1211 = select i1 %1210, i32 %1208, i32 %1206, !dbg !15
  %1212 = shl nuw nsw i32 %1211, 2, !dbg !15
  %1213 = bitcast float %1204 to i32, !dbg !15
  %1214 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1213), !dbg !15
  %1215 = bitcast i32 %1214 to float, !dbg !15
  %1216 = select i1 %.not2, float %1200, float %1215, !dbg !15
  %1217 = select i1 %.not2, float %1215, float %1201, !dbg !15
  %1218 = select i1 %.not2, float %1203, float %1202, !dbg !15
  %1219 = bitcast float %1218 to i32, !dbg !15
  %1220 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1219), !dbg !15
  %1221 = bitcast i32 %1220 to float, !dbg !15
  %1222 = select i1 %.not2, float %1202, float %1221, !dbg !15
  %1223 = select i1 %.not2, float %1221, float %1203, !dbg !15
  %1224 = select i1 %.not3, float %1222, float %1216, !dbg !15
  %1225 = xor i32 %1206, 32, !dbg !15
  %1226 = icmp samesign ult i32 %1225, %1209, !dbg !15
  %1227 = select i1 %1226, i32 %1225, i32 %1206, !dbg !15
  %1228 = shl nuw nsw i32 %1227, 2, !dbg !15
  %1229 = bitcast float %1224 to i32, !dbg !15
  %1230 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1229), !dbg !15
  %1231 = bitcast i32 %1230 to float, !dbg !15
  %1232 = select i1 %.not3, float %1216, float %1231, !dbg !15
  %1233 = select i1 %.not3, float %1231, float %1222, !dbg !15
  %1234 = select i1 %.not3, float %1223, float %1217, !dbg !15
  %1235 = bitcast float %1234 to i32, !dbg !15
  %1236 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1235), !dbg !15
  %1237 = bitcast i32 %1236 to float, !dbg !15
  %1238 = select i1 %.not3, float %1217, float %1237, !dbg !15
  %1239 = select i1 %.not3, float %1237, float %1223, !dbg !15
  %1240 = extractvalue [2 x <16 x float>] %27, 0, !dbg !15
  %1241 = extractelement <16 x float> %1240, i64 0, !dbg !15
  %1242 = extractelement <16 x float> %1240, i64 1, !dbg !15
  %1243 = extractelement <16 x float> %1240, i64 2, !dbg !15
  %1244 = extractelement <16 x float> %1240, i64 3, !dbg !15
  %1245 = select i1 %.not2, float %1242, float %1241, !dbg !15
  %1246 = bitcast float %1245 to i32, !dbg !15
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1246), !dbg !15
  %1248 = bitcast i32 %1247 to float, !dbg !15
  %1249 = select i1 %.not2, float %1241, float %1248, !dbg !15
  %1250 = select i1 %.not2, float %1248, float %1242, !dbg !15
  %1251 = select i1 %.not2, float %1244, float %1243, !dbg !15
  %1252 = bitcast float %1251 to i32, !dbg !15
  %1253 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1252), !dbg !15
  %1254 = bitcast i32 %1253 to float, !dbg !15
  %1255 = select i1 %.not2, float %1243, float %1254, !dbg !15
  %1256 = select i1 %.not2, float %1254, float %1244, !dbg !15
  %1257 = select i1 %.not3, float %1255, float %1249, !dbg !15
  %1258 = bitcast float %1257 to i32, !dbg !15
  %1259 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1258), !dbg !15
  %1260 = select i1 %.not3, float %1256, float %1250, !dbg !15
  %1261 = bitcast float %1260 to i32, !dbg !15
  %1262 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1261), !dbg !15
  %1263 = extractvalue [2 x <1 x float>] %28, 0, !dbg !19
  %1264 = shufflevector <1 x float> %1263, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1265 = extractelement <16 x float> %1199, i64 4, !dbg !15
  %1266 = extractelement <16 x float> %1199, i64 5, !dbg !15
  %1267 = extractelement <16 x float> %1199, i64 6, !dbg !15
  %1268 = extractelement <16 x float> %1199, i64 7, !dbg !15
  %1269 = select i1 %.not2, float %1266, float %1265, !dbg !15
  %1270 = bitcast float %1269 to i32, !dbg !15
  %1271 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1270), !dbg !15
  %1272 = bitcast i32 %1271 to float, !dbg !15
  %1273 = select i1 %.not2, float %1265, float %1272, !dbg !15
  %1274 = select i1 %.not2, float %1272, float %1266, !dbg !15
  %1275 = select i1 %.not2, float %1268, float %1267, !dbg !15
  %1276 = bitcast float %1275 to i32, !dbg !15
  %1277 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1276), !dbg !15
  %1278 = bitcast i32 %1277 to float, !dbg !15
  %1279 = select i1 %.not2, float %1267, float %1278, !dbg !15
  %1280 = select i1 %.not2, float %1278, float %1268, !dbg !15
  %1281 = select i1 %.not3, float %1279, float %1273, !dbg !15
  %1282 = bitcast float %1281 to i32, !dbg !15
  %1283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1282), !dbg !15
  %1284 = bitcast i32 %1283 to float, !dbg !15
  %1285 = select i1 %.not3, float %1273, float %1284, !dbg !15
  %1286 = select i1 %.not3, float %1284, float %1279, !dbg !15
  %1287 = select i1 %.not3, float %1280, float %1274, !dbg !15
  %1288 = bitcast float %1287 to i32, !dbg !15
  %1289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1288), !dbg !15
  %1290 = bitcast i32 %1289 to float, !dbg !15
  %1291 = select i1 %.not3, float %1274, float %1290, !dbg !15
  %1292 = select i1 %.not3, float %1290, float %1280, !dbg !15
  %1293 = extractelement <16 x float> %1240, i64 4, !dbg !15
  %1294 = extractelement <16 x float> %1240, i64 5, !dbg !15
  %1295 = extractelement <16 x float> %1240, i64 6, !dbg !15
  %1296 = extractelement <16 x float> %1240, i64 7, !dbg !15
  %1297 = select i1 %.not2, float %1294, float %1293, !dbg !15
  %1298 = bitcast float %1297 to i32, !dbg !15
  %1299 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1298), !dbg !15
  %1300 = bitcast i32 %1299 to float, !dbg !15
  %1301 = select i1 %.not2, float %1293, float %1300, !dbg !15
  %1302 = select i1 %.not2, float %1300, float %1294, !dbg !15
  %1303 = select i1 %.not2, float %1296, float %1295, !dbg !15
  %1304 = bitcast float %1303 to i32, !dbg !15
  %1305 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1304), !dbg !15
  %1306 = bitcast i32 %1305 to float, !dbg !15
  %1307 = select i1 %.not2, float %1295, float %1306, !dbg !15
  %1308 = select i1 %.not2, float %1306, float %1296, !dbg !15
  %1309 = select i1 %.not3, float %1307, float %1301, !dbg !15
  %1310 = bitcast float %1309 to i32, !dbg !15
  %1311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1310), !dbg !15
  %1312 = select i1 %.not3, float %1308, float %1302, !dbg !15
  %1313 = bitcast float %1312 to i32, !dbg !15
  %1314 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1313), !dbg !15
  %1315 = extractelement <16 x float> %1199, i64 8, !dbg !15
  %1316 = extractelement <16 x float> %1199, i64 9, !dbg !15
  %1317 = extractelement <16 x float> %1199, i64 10, !dbg !15
  %1318 = extractelement <16 x float> %1199, i64 11, !dbg !15
  %1319 = select i1 %.not2, float %1316, float %1315, !dbg !15
  %1320 = bitcast float %1319 to i32, !dbg !15
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1320), !dbg !15
  %1322 = bitcast i32 %1321 to float, !dbg !15
  %1323 = select i1 %.not2, float %1315, float %1322, !dbg !15
  %1324 = select i1 %.not2, float %1322, float %1316, !dbg !15
  %1325 = select i1 %.not2, float %1318, float %1317, !dbg !15
  %1326 = bitcast float %1325 to i32, !dbg !15
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1326), !dbg !15
  %1328 = bitcast i32 %1327 to float, !dbg !15
  %1329 = select i1 %.not2, float %1317, float %1328, !dbg !15
  %1330 = select i1 %.not2, float %1328, float %1318, !dbg !15
  %1331 = select i1 %.not3, float %1329, float %1323, !dbg !15
  %1332 = bitcast float %1331 to i32, !dbg !15
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1332), !dbg !15
  %1334 = bitcast i32 %1333 to float, !dbg !15
  %1335 = select i1 %.not3, float %1323, float %1334, !dbg !15
  %1336 = select i1 %.not3, float %1334, float %1329, !dbg !15
  %1337 = select i1 %.not3, float %1330, float %1324, !dbg !15
  %1338 = bitcast float %1337 to i32, !dbg !15
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1338), !dbg !15
  %1340 = bitcast i32 %1339 to float, !dbg !15
  %1341 = select i1 %.not3, float %1324, float %1340, !dbg !15
  %1342 = select i1 %.not3, float %1340, float %1330, !dbg !15
  %1343 = extractelement <16 x float> %1240, i64 8, !dbg !15
  %1344 = extractelement <16 x float> %1240, i64 9, !dbg !15
  %1345 = extractelement <16 x float> %1240, i64 10, !dbg !15
  %1346 = extractelement <16 x float> %1240, i64 11, !dbg !15
  %1347 = select i1 %.not2, float %1344, float %1343, !dbg !15
  %1348 = bitcast float %1347 to i32, !dbg !15
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1348), !dbg !15
  %1350 = bitcast i32 %1349 to float, !dbg !15
  %1351 = select i1 %.not2, float %1343, float %1350, !dbg !15
  %1352 = select i1 %.not2, float %1350, float %1344, !dbg !15
  %1353 = select i1 %.not2, float %1346, float %1345, !dbg !15
  %1354 = bitcast float %1353 to i32, !dbg !15
  %1355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1354), !dbg !15
  %1356 = bitcast i32 %1355 to float, !dbg !15
  %1357 = select i1 %.not2, float %1345, float %1356, !dbg !15
  %1358 = select i1 %.not2, float %1356, float %1346, !dbg !15
  %1359 = select i1 %.not3, float %1357, float %1351, !dbg !15
  %1360 = bitcast float %1359 to i32, !dbg !15
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1360), !dbg !15
  %1362 = select i1 %.not3, float %1358, float %1352, !dbg !15
  %1363 = bitcast float %1362 to i32, !dbg !15
  %1364 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1363), !dbg !15
  %1365 = extractelement <16 x float> %1199, i64 12, !dbg !15
  %1366 = extractelement <16 x float> %1199, i64 13, !dbg !15
  %1367 = extractelement <16 x float> %1199, i64 14, !dbg !15
  %1368 = extractelement <16 x float> %1199, i64 15, !dbg !15
  %1369 = select i1 %.not2, float %1366, float %1365, !dbg !15
  %1370 = bitcast float %1369 to i32, !dbg !15
  %1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1370), !dbg !15
  %1372 = bitcast i32 %1371 to float, !dbg !15
  %1373 = select i1 %.not2, float %1365, float %1372, !dbg !15
  %1374 = select i1 %.not2, float %1372, float %1366, !dbg !15
  %1375 = select i1 %.not2, float %1368, float %1367, !dbg !15
  %1376 = bitcast float %1375 to i32, !dbg !15
  %1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1376), !dbg !15
  %1378 = bitcast i32 %1377 to float, !dbg !15
  %1379 = select i1 %.not2, float %1367, float %1378, !dbg !15
  %1380 = select i1 %.not2, float %1378, float %1368, !dbg !15
  %1381 = select i1 %.not3, float %1379, float %1373, !dbg !15
  %1382 = bitcast float %1381 to i32, !dbg !15
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1382), !dbg !15
  %1384 = bitcast i32 %1383 to float, !dbg !15
  %1385 = select i1 %.not3, float %1373, float %1384, !dbg !15
  %1386 = select i1 %.not3, float %1384, float %1379, !dbg !15
  %1387 = select i1 %.not3, float %1380, float %1374, !dbg !15
  %1388 = bitcast float %1387 to i32, !dbg !15
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1388), !dbg !15
  %1390 = bitcast i32 %1389 to float, !dbg !15
  %1391 = select i1 %.not3, float %1374, float %1390, !dbg !15
  %1392 = select i1 %.not3, float %1390, float %1380, !dbg !15
  %1393 = extractelement <16 x float> %1240, i64 12, !dbg !15
  %1394 = extractelement <16 x float> %1240, i64 13, !dbg !15
  %1395 = extractelement <16 x float> %1240, i64 14, !dbg !15
  %1396 = extractelement <16 x float> %1240, i64 15, !dbg !15
  %1397 = select i1 %.not2, float %1394, float %1393, !dbg !15
  %1398 = bitcast float %1397 to i32, !dbg !15
  %1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1398), !dbg !15
  %1400 = bitcast i32 %1399 to float, !dbg !15
  %1401 = select i1 %.not2, float %1393, float %1400, !dbg !15
  %1402 = select i1 %.not2, float %1400, float %1394, !dbg !15
  %1403 = select i1 %.not2, float %1396, float %1395, !dbg !15
  %1404 = bitcast float %1403 to i32, !dbg !15
  %1405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1404), !dbg !15
  %1406 = bitcast i32 %1405 to float, !dbg !15
  %1407 = select i1 %.not2, float %1395, float %1406, !dbg !15
  %1408 = select i1 %.not2, float %1406, float %1396, !dbg !15
  %1409 = select i1 %.not3, float %1407, float %1401, !dbg !15
  %1410 = bitcast float %1409 to i32, !dbg !15
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1410), !dbg !15
  %1412 = select i1 %.not3, float %1408, float %1402, !dbg !15
  %1413 = bitcast float %1412 to i32, !dbg !15
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1413), !dbg !15
  %1415 = extractvalue [2 x <16 x float>] %29, 1, !dbg !15
  %1416 = extractelement <16 x float> %1415, i64 0, !dbg !15
  %1417 = extractelement <16 x float> %1415, i64 1, !dbg !15
  %1418 = extractelement <16 x float> %1415, i64 2, !dbg !15
  %1419 = extractelement <16 x float> %1415, i64 3, !dbg !15
  %1420 = select i1 %.not2, float %1417, float %1416, !dbg !15
  %1421 = bitcast float %1420 to i32, !dbg !15
  %1422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1421), !dbg !15
  %1423 = bitcast i32 %1422 to float, !dbg !15
  %1424 = select i1 %.not2, float %1416, float %1423, !dbg !15
  %1425 = select i1 %.not2, float %1423, float %1417, !dbg !15
  %1426 = select i1 %.not2, float %1419, float %1418, !dbg !15
  %1427 = bitcast float %1426 to i32, !dbg !15
  %1428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1427), !dbg !15
  %1429 = bitcast i32 %1428 to float, !dbg !15
  %1430 = select i1 %.not2, float %1418, float %1429, !dbg !15
  %1431 = select i1 %.not2, float %1429, float %1419, !dbg !15
  %1432 = select i1 %.not3, float %1430, float %1424, !dbg !15
  %1433 = bitcast float %1432 to i32, !dbg !15
  %1434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1433), !dbg !15
  %1435 = bitcast i32 %1434 to float, !dbg !15
  %1436 = select i1 %.not3, float %1424, float %1435, !dbg !15
  %1437 = select i1 %.not3, float %1435, float %1430, !dbg !15
  %1438 = select i1 %.not3, float %1431, float %1425, !dbg !15
  %1439 = bitcast float %1438 to i32, !dbg !15
  %1440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1439), !dbg !15
  %1441 = bitcast i32 %1440 to float, !dbg !15
  %1442 = select i1 %.not3, float %1425, float %1441, !dbg !15
  %1443 = select i1 %.not3, float %1441, float %1431, !dbg !15
  %1444 = extractvalue [2 x <16 x float>] %27, 1, !dbg !15
  %1445 = extractelement <16 x float> %1444, i64 0, !dbg !15
  %1446 = extractelement <16 x float> %1444, i64 1, !dbg !15
  %1447 = extractelement <16 x float> %1444, i64 2, !dbg !15
  %1448 = extractelement <16 x float> %1444, i64 3, !dbg !15
  %1449 = select i1 %.not2, float %1446, float %1445, !dbg !15
  %1450 = bitcast float %1449 to i32, !dbg !15
  %1451 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1450), !dbg !15
  %1452 = bitcast i32 %1451 to float, !dbg !15
  %1453 = select i1 %.not2, float %1445, float %1452, !dbg !15
  %1454 = select i1 %.not2, float %1452, float %1446, !dbg !15
  %1455 = select i1 %.not2, float %1448, float %1447, !dbg !15
  %1456 = bitcast float %1455 to i32, !dbg !15
  %1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1456), !dbg !15
  %1458 = bitcast i32 %1457 to float, !dbg !15
  %1459 = select i1 %.not2, float %1447, float %1458, !dbg !15
  %1460 = select i1 %.not2, float %1458, float %1448, !dbg !15
  %1461 = select i1 %.not3, float %1459, float %1453, !dbg !15
  %1462 = bitcast float %1461 to i32, !dbg !15
  %1463 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1462), !dbg !15
  %1464 = select i1 %.not3, float %1460, float %1454, !dbg !15
  %1465 = bitcast float %1464 to i32, !dbg !15
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1465), !dbg !15
  %1467 = extractvalue [2 x <1 x float>] %28, 1, !dbg !19
  %1468 = shufflevector <1 x float> %1467, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1469 = extractelement <16 x float> %1415, i64 4, !dbg !15
  %1470 = extractelement <16 x float> %1415, i64 5, !dbg !15
  %1471 = extractelement <16 x float> %1415, i64 6, !dbg !15
  %1472 = extractelement <16 x float> %1415, i64 7, !dbg !15
  %1473 = select i1 %.not2, float %1470, float %1469, !dbg !15
  %1474 = bitcast float %1473 to i32, !dbg !15
  %1475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1474), !dbg !15
  %1476 = bitcast i32 %1475 to float, !dbg !15
  %1477 = select i1 %.not2, float %1469, float %1476, !dbg !15
  %1478 = select i1 %.not2, float %1476, float %1470, !dbg !15
  %1479 = select i1 %.not2, float %1472, float %1471, !dbg !15
  %1480 = bitcast float %1479 to i32, !dbg !15
  %1481 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1480), !dbg !15
  %1482 = bitcast i32 %1481 to float, !dbg !15
  %1483 = select i1 %.not2, float %1471, float %1482, !dbg !15
  %1484 = select i1 %.not2, float %1482, float %1472, !dbg !15
  %1485 = select i1 %.not3, float %1483, float %1477, !dbg !15
  %1486 = bitcast float %1485 to i32, !dbg !15
  %1487 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1486), !dbg !15
  %1488 = bitcast i32 %1487 to float, !dbg !15
  %1489 = select i1 %.not3, float %1477, float %1488, !dbg !15
  %1490 = select i1 %.not3, float %1488, float %1483, !dbg !15
  %1491 = select i1 %.not3, float %1484, float %1478, !dbg !15
  %1492 = bitcast float %1491 to i32, !dbg !15
  %1493 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1492), !dbg !15
  %1494 = bitcast i32 %1493 to float, !dbg !15
  %1495 = select i1 %.not3, float %1478, float %1494, !dbg !15
  %1496 = select i1 %.not3, float %1494, float %1484, !dbg !15
  %1497 = extractelement <16 x float> %1444, i64 4, !dbg !15
  %1498 = extractelement <16 x float> %1444, i64 5, !dbg !15
  %1499 = extractelement <16 x float> %1444, i64 6, !dbg !15
  %1500 = extractelement <16 x float> %1444, i64 7, !dbg !15
  %1501 = select i1 %.not2, float %1498, float %1497, !dbg !15
  %1502 = bitcast float %1501 to i32, !dbg !15
  %1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1502), !dbg !15
  %1504 = bitcast i32 %1503 to float, !dbg !15
  %1505 = select i1 %.not2, float %1497, float %1504, !dbg !15
  %1506 = select i1 %.not2, float %1504, float %1498, !dbg !15
  %1507 = select i1 %.not2, float %1500, float %1499, !dbg !15
  %1508 = bitcast float %1507 to i32, !dbg !15
  %1509 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1508), !dbg !15
  %1510 = bitcast i32 %1509 to float, !dbg !15
  %1511 = select i1 %.not2, float %1499, float %1510, !dbg !15
  %1512 = select i1 %.not2, float %1510, float %1500, !dbg !15
  %1513 = select i1 %.not3, float %1511, float %1505, !dbg !15
  %1514 = bitcast float %1513 to i32, !dbg !15
  %1515 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1514), !dbg !15
  %1516 = select i1 %.not3, float %1512, float %1506, !dbg !15
  %1517 = bitcast float %1516 to i32, !dbg !15
  %1518 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1517), !dbg !15
  %1519 = extractelement <16 x float> %1415, i64 8, !dbg !15
  %1520 = extractelement <16 x float> %1415, i64 9, !dbg !15
  %1521 = extractelement <16 x float> %1415, i64 10, !dbg !15
  %1522 = extractelement <16 x float> %1415, i64 11, !dbg !15
  %1523 = select i1 %.not2, float %1520, float %1519, !dbg !15
  %1524 = bitcast float %1523 to i32, !dbg !15
  %1525 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1524), !dbg !15
  %1526 = bitcast i32 %1525 to float, !dbg !15
  %1527 = select i1 %.not2, float %1519, float %1526, !dbg !15
  %1528 = select i1 %.not2, float %1526, float %1520, !dbg !15
  %1529 = select i1 %.not2, float %1522, float %1521, !dbg !15
  %1530 = bitcast float %1529 to i32, !dbg !15
  %1531 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1530), !dbg !15
  %1532 = bitcast i32 %1531 to float, !dbg !15
  %1533 = select i1 %.not2, float %1521, float %1532, !dbg !15
  %1534 = select i1 %.not2, float %1532, float %1522, !dbg !15
  %1535 = select i1 %.not3, float %1533, float %1527, !dbg !15
  %1536 = bitcast float %1535 to i32, !dbg !15
  %1537 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1536), !dbg !15
  %1538 = bitcast i32 %1537 to float, !dbg !15
  %1539 = select i1 %.not3, float %1527, float %1538, !dbg !15
  %1540 = select i1 %.not3, float %1538, float %1533, !dbg !15
  %1541 = select i1 %.not3, float %1534, float %1528, !dbg !15
  %1542 = bitcast float %1541 to i32, !dbg !15
  %1543 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1542), !dbg !15
  %1544 = bitcast i32 %1543 to float, !dbg !15
  %1545 = select i1 %.not3, float %1528, float %1544, !dbg !15
  %1546 = select i1 %.not3, float %1544, float %1534, !dbg !15
  %1547 = extractelement <16 x float> %1444, i64 8, !dbg !15
  %1548 = extractelement <16 x float> %1444, i64 9, !dbg !15
  %1549 = extractelement <16 x float> %1444, i64 10, !dbg !15
  %1550 = extractelement <16 x float> %1444, i64 11, !dbg !15
  %1551 = select i1 %.not2, float %1548, float %1547, !dbg !15
  %1552 = bitcast float %1551 to i32, !dbg !15
  %1553 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1552), !dbg !15
  %1554 = bitcast i32 %1553 to float, !dbg !15
  %1555 = select i1 %.not2, float %1547, float %1554, !dbg !15
  %1556 = select i1 %.not2, float %1554, float %1548, !dbg !15
  %1557 = select i1 %.not2, float %1550, float %1549, !dbg !15
  %1558 = bitcast float %1557 to i32, !dbg !15
  %1559 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1558), !dbg !15
  %1560 = bitcast i32 %1559 to float, !dbg !15
  %1561 = select i1 %.not2, float %1549, float %1560, !dbg !15
  %1562 = select i1 %.not2, float %1560, float %1550, !dbg !15
  %1563 = select i1 %.not3, float %1561, float %1555, !dbg !15
  %1564 = bitcast float %1563 to i32, !dbg !15
  %1565 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1564), !dbg !15
  %1566 = select i1 %.not3, float %1562, float %1556, !dbg !15
  %1567 = bitcast float %1566 to i32, !dbg !15
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1567), !dbg !15
  %1569 = extractelement <16 x float> %1415, i64 12, !dbg !15
  %1570 = extractelement <16 x float> %1415, i64 13, !dbg !15
  %1571 = extractelement <16 x float> %1415, i64 14, !dbg !15
  %1572 = extractelement <16 x float> %1415, i64 15, !dbg !15
  %1573 = select i1 %.not2, float %1570, float %1569, !dbg !15
  %1574 = bitcast float %1573 to i32, !dbg !15
  %1575 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1574), !dbg !15
  %1576 = bitcast i32 %1575 to float, !dbg !15
  %1577 = select i1 %.not2, float %1569, float %1576, !dbg !15
  %1578 = select i1 %.not2, float %1576, float %1570, !dbg !15
  %1579 = select i1 %.not2, float %1572, float %1571, !dbg !15
  %1580 = bitcast float %1579 to i32, !dbg !15
  %1581 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1580), !dbg !15
  %1582 = bitcast i32 %1581 to float, !dbg !15
  %1583 = select i1 %.not2, float %1571, float %1582, !dbg !15
  %1584 = select i1 %.not2, float %1582, float %1572, !dbg !15
  %1585 = select i1 %.not3, float %1583, float %1577, !dbg !15
  %1586 = bitcast float %1585 to i32, !dbg !15
  %1587 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1586), !dbg !15
  %1588 = bitcast i32 %1587 to float, !dbg !15
  %1589 = select i1 %.not3, float %1577, float %1588, !dbg !15
  %1590 = select i1 %.not3, float %1588, float %1583, !dbg !15
  %1591 = select i1 %.not3, float %1584, float %1578, !dbg !15
  %1592 = bitcast float %1591 to i32, !dbg !15
  %1593 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1592), !dbg !15
  %1594 = bitcast i32 %1593 to float, !dbg !15
  %1595 = select i1 %.not3, float %1578, float %1594, !dbg !15
  %1596 = select i1 %.not3, float %1594, float %1584, !dbg !15
  %1597 = extractelement <16 x float> %1444, i64 12, !dbg !15
  %1598 = extractelement <16 x float> %1444, i64 13, !dbg !15
  %1599 = extractelement <16 x float> %1444, i64 14, !dbg !15
  %1600 = extractelement <16 x float> %1444, i64 15, !dbg !15
  %1601 = select i1 %.not2, float %1598, float %1597, !dbg !15
  %1602 = bitcast float %1601 to i32, !dbg !15
  %1603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1602), !dbg !15
  %1604 = bitcast i32 %1603 to float, !dbg !15
  %1605 = select i1 %.not2, float %1597, float %1604, !dbg !15
  %1606 = select i1 %.not2, float %1604, float %1598, !dbg !15
  %1607 = select i1 %.not2, float %1600, float %1599, !dbg !15
  %1608 = bitcast float %1607 to i32, !dbg !15
  %1609 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1608), !dbg !15
  %1610 = bitcast i32 %1609 to float, !dbg !15
  %1611 = select i1 %.not2, float %1599, float %1610, !dbg !15
  %1612 = select i1 %.not2, float %1610, float %1600, !dbg !15
  %1613 = select i1 %.not3, float %1611, float %1605, !dbg !15
  %1614 = bitcast float %1613 to i32, !dbg !15
  %1615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1614), !dbg !15
  %1616 = select i1 %.not3, float %1612, float %1606, !dbg !15
  %1617 = bitcast float %1616 to i32, !dbg !15
  %1618 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1228, i32 %1617), !dbg !15
  br i1 true, label %1619, label %1743, !dbg !21

1619:                                             ; preds = %1196
  %1620 = bitcast i32 %1615 to float, !dbg !15
  %1621 = select i1 %.not3, float %1605, float %1620, !dbg !15
  %1622 = insertelement <4 x float> poison, float %1621, i64 0, !dbg !19
  %1623 = bitcast i32 %1618 to float, !dbg !15
  %1624 = select i1 %.not3, float %1606, float %1623, !dbg !15
  %1625 = insertelement <4 x float> %1622, float %1624, i64 1, !dbg !19
  %1626 = select i1 %.not3, float %1620, float %1611, !dbg !15
  %1627 = insertelement <4 x float> %1625, float %1626, i64 2, !dbg !19
  %1628 = select i1 %.not3, float %1623, float %1612, !dbg !15
  %1629 = insertelement <4 x float> %1627, float %1628, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1468, !dbg !19
  %normalized = fmul <4 x float> %1629, %row.reciprocal, !dbg !19
  %1630 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1631 = extractelement <4 x half> %1630, i64 3, !dbg !20
  %1632 = extractelement <4 x half> %1630, i64 2, !dbg !20
  %1633 = extractelement <4 x half> %1630, i64 1, !dbg !20
  %1634 = extractelement <4 x half> %1630, i64 0, !dbg !20
  %1635 = bitcast i32 %1565 to float, !dbg !15
  %1636 = select i1 %.not3, float %1555, float %1635, !dbg !15
  %1637 = insertelement <4 x float> poison, float %1636, i64 0, !dbg !19
  %1638 = bitcast i32 %1568 to float, !dbg !15
  %1639 = select i1 %.not3, float %1556, float %1638, !dbg !15
  %1640 = insertelement <4 x float> %1637, float %1639, i64 1, !dbg !19
  %1641 = select i1 %.not3, float %1635, float %1561, !dbg !15
  %1642 = insertelement <4 x float> %1640, float %1641, i64 2, !dbg !19
  %1643 = select i1 %.not3, float %1638, float %1562, !dbg !15
  %1644 = insertelement <4 x float> %1642, float %1643, i64 3, !dbg !19
  %normalized72 = fmul <4 x float> %1644, %row.reciprocal, !dbg !19
  %1645 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1646 = extractelement <4 x half> %1645, i64 3, !dbg !20
  %1647 = extractelement <4 x half> %1645, i64 2, !dbg !20
  %1648 = extractelement <4 x half> %1645, i64 1, !dbg !20
  %1649 = extractelement <4 x half> %1645, i64 0, !dbg !20
  %1650 = bitcast i32 %1515 to float, !dbg !15
  %1651 = select i1 %.not3, float %1505, float %1650, !dbg !15
  %1652 = insertelement <4 x float> poison, float %1651, i64 0, !dbg !19
  %1653 = bitcast i32 %1518 to float, !dbg !15
  %1654 = select i1 %.not3, float %1506, float %1653, !dbg !15
  %1655 = insertelement <4 x float> %1652, float %1654, i64 1, !dbg !19
  %1656 = select i1 %.not3, float %1650, float %1511, !dbg !15
  %1657 = insertelement <4 x float> %1655, float %1656, i64 2, !dbg !19
  %1658 = select i1 %.not3, float %1653, float %1512, !dbg !15
  %1659 = insertelement <4 x float> %1657, float %1658, i64 3, !dbg !19
  %normalized73 = fmul <4 x float> %1659, %row.reciprocal, !dbg !19
  %1660 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !20
  %1661 = extractelement <4 x half> %1660, i64 3, !dbg !20
  %1662 = extractelement <4 x half> %1660, i64 2, !dbg !20
  %1663 = extractelement <4 x half> %1660, i64 1, !dbg !20
  %1664 = extractelement <4 x half> %1660, i64 0, !dbg !20
  %1665 = bitcast i32 %1463 to float, !dbg !15
  %1666 = select i1 %.not3, float %1453, float %1665, !dbg !15
  %1667 = insertelement <4 x float> poison, float %1666, i64 0, !dbg !19
  %1668 = bitcast i32 %1466 to float, !dbg !15
  %1669 = select i1 %.not3, float %1454, float %1668, !dbg !15
  %1670 = insertelement <4 x float> %1667, float %1669, i64 1, !dbg !19
  %1671 = select i1 %.not3, float %1665, float %1459, !dbg !15
  %1672 = insertelement <4 x float> %1670, float %1671, i64 2, !dbg !19
  %1673 = select i1 %.not3, float %1668, float %1460, !dbg !15
  %1674 = insertelement <4 x float> %1672, float %1673, i64 3, !dbg !19
  %normalized74 = fmul <4 x float> %1674, %row.reciprocal, !dbg !19
  %1675 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1676 = extractelement <4 x half> %1675, i64 3, !dbg !20
  %1677 = extractelement <4 x half> %1675, i64 2, !dbg !20
  %1678 = extractelement <4 x half> %1675, i64 1, !dbg !20
  %1679 = extractelement <4 x half> %1675, i64 0, !dbg !20
  %1680 = bitcast i32 %1411 to float, !dbg !15
  %1681 = select i1 %.not3, float %1401, float %1680, !dbg !15
  %1682 = insertelement <4 x float> poison, float %1681, i64 0, !dbg !19
  %1683 = bitcast i32 %1414 to float, !dbg !15
  %1684 = select i1 %.not3, float %1402, float %1683, !dbg !15
  %1685 = insertelement <4 x float> %1682, float %1684, i64 1, !dbg !19
  %1686 = select i1 %.not3, float %1680, float %1407, !dbg !15
  %1687 = insertelement <4 x float> %1685, float %1686, i64 2, !dbg !19
  %1688 = select i1 %.not3, float %1683, float %1408, !dbg !15
  %1689 = insertelement <4 x float> %1687, float %1688, i64 3, !dbg !19
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1264, !dbg !19
  %normalized76 = fmul <4 x float> %1689, %row.reciprocal75, !dbg !19
  %1690 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !20
  %1691 = extractelement <4 x half> %1690, i64 3, !dbg !20
  %1692 = extractelement <4 x half> %1690, i64 2, !dbg !20
  %1693 = extractelement <4 x half> %1690, i64 1, !dbg !20
  %1694 = extractelement <4 x half> %1690, i64 0, !dbg !20
  %1695 = bitcast i32 %1361 to float, !dbg !15
  %1696 = select i1 %.not3, float %1351, float %1695, !dbg !15
  %1697 = insertelement <4 x float> poison, float %1696, i64 0, !dbg !19
  %1698 = bitcast i32 %1364 to float, !dbg !15
  %1699 = select i1 %.not3, float %1352, float %1698, !dbg !15
  %1700 = insertelement <4 x float> %1697, float %1699, i64 1, !dbg !19
  %1701 = select i1 %.not3, float %1695, float %1357, !dbg !15
  %1702 = insertelement <4 x float> %1700, float %1701, i64 2, !dbg !19
  %1703 = select i1 %.not3, float %1698, float %1358, !dbg !15
  %1704 = insertelement <4 x float> %1702, float %1703, i64 3, !dbg !19
  %normalized77 = fmul <4 x float> %1704, %row.reciprocal75, !dbg !19
  %1705 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %1706 = extractelement <4 x half> %1705, i64 3, !dbg !20
  %1707 = extractelement <4 x half> %1705, i64 2, !dbg !20
  %1708 = extractelement <4 x half> %1705, i64 1, !dbg !20
  %1709 = extractelement <4 x half> %1705, i64 0, !dbg !20
  %1710 = bitcast i32 %1311 to float, !dbg !15
  %1711 = select i1 %.not3, float %1301, float %1710, !dbg !15
  %1712 = insertelement <4 x float> poison, float %1711, i64 0, !dbg !19
  %1713 = bitcast i32 %1314 to float, !dbg !15
  %1714 = select i1 %.not3, float %1302, float %1713, !dbg !15
  %1715 = insertelement <4 x float> %1712, float %1714, i64 1, !dbg !19
  %1716 = select i1 %.not3, float %1710, float %1307, !dbg !15
  %1717 = insertelement <4 x float> %1715, float %1716, i64 2, !dbg !19
  %1718 = select i1 %.not3, float %1713, float %1308, !dbg !15
  %1719 = insertelement <4 x float> %1717, float %1718, i64 3, !dbg !19
  %normalized78 = fmul <4 x float> %1719, %row.reciprocal75, !dbg !19
  %1720 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !20
  %1721 = extractelement <4 x half> %1720, i64 3, !dbg !20
  %1722 = extractelement <4 x half> %1720, i64 2, !dbg !20
  %1723 = extractelement <4 x half> %1720, i64 1, !dbg !20
  %1724 = extractelement <4 x half> %1720, i64 0, !dbg !20
  %1725 = bitcast i32 %1259 to float, !dbg !15
  %1726 = select i1 %.not3, float %1249, float %1725, !dbg !15
  %1727 = insertelement <4 x float> poison, float %1726, i64 0, !dbg !19
  %1728 = bitcast i32 %1262 to float, !dbg !15
  %1729 = select i1 %.not3, float %1250, float %1728, !dbg !15
  %1730 = insertelement <4 x float> %1727, float %1729, i64 1, !dbg !19
  %1731 = select i1 %.not3, float %1725, float %1255, !dbg !15
  %1732 = insertelement <4 x float> %1730, float %1731, i64 2, !dbg !19
  %1733 = select i1 %.not3, float %1728, float %1256, !dbg !15
  %1734 = insertelement <4 x float> %1732, float %1733, i64 3, !dbg !19
  %normalized79 = fmul <4 x float> %1734, %row.reciprocal75, !dbg !19
  %1735 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !20
  %1736 = extractelement <4 x half> %1735, i64 3, !dbg !20
  %1737 = extractelement <4 x half> %1735, i64 2, !dbg !20
  %1738 = extractelement <4 x half> %1735, i64 1, !dbg !20
  %1739 = extractelement <4 x half> %1735, i64 0, !dbg !20
  %1740 = add nsw i64 %8, %13, !dbg !21
  %1741 = shl nsw i64 %1740, 19, !dbg !21
  %1742 = and i64 %1741, 16252928, !dbg !21
  %address.thread518 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local519 = zext nneg i32 %address.thread518 to i64, !dbg !13
  %address.local520 = add nsw i64 %16, %address.local519, !dbg !21
  %address.local521 = lshr i64 %address.local519, 6, !dbg !13
  %address.local522 = shl nuw nsw i64 %address.local521, 4, !dbg !21
  %address.local523 = add nsw i64 %address.local520, %address.local522, !dbg !21
  %address.local524 = and i64 %address.local519, 240, !dbg !21
  %address.local525 = sub nsw i64 %address.local523, %address.local524, !dbg !21
  %address.local526 = add nsw i64 %address.local525, %17, !dbg !21
  %address.local527 = shl nuw nsw i64 %address.local526, 7, !dbg !21
  %address.local528 = add nuw nsw i64 %1742, %address.local527, !dbg !21
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local530 = lshr i64 %address.local519, 2, !dbg !13
  %address.local531 = and i64 %address.local530, 12, !dbg !13
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local529, i64 %address.local531, !dbg !21
  store half %1739, ptr addrspace(1) %address.local532, align 2, !dbg !21
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local534 = or disjoint i64 %address.local531, 1, !dbg !21
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local533, i64 %address.local534, !dbg !21
  store half %1738, ptr addrspace(1) %address.local535, align 2, !dbg !21
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local537 = or disjoint i64 %address.local531, 2, !dbg !21
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local536, i64 %address.local537, !dbg !21
  store half %1737, ptr addrspace(1) %address.local538, align 2, !dbg !21
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local540 = or disjoint i64 %address.local531, 3, !dbg !21
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local539, i64 %address.local540, !dbg !21
  store half %1736, ptr addrspace(1) %address.local541, align 2, !dbg !21
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local543 = or disjoint i64 %address.local531, 16, !dbg !21
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local542, i64 %address.local543, !dbg !21
  store half %1724, ptr addrspace(1) %address.local544, align 2, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local546 = or disjoint i64 %address.local531, 17, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local546, !dbg !21
  store half %1723, ptr addrspace(1) %address.local547, align 2, !dbg !21
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local549 = or disjoint i64 %address.local531, 18, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local548, i64 %address.local549, !dbg !21
  store half %1722, ptr addrspace(1) %address.local550, align 2, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local552 = or disjoint i64 %address.local531, 19, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local551, i64 %address.local552, !dbg !21
  store half %1721, ptr addrspace(1) %address.local553, align 2, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local555 = or disjoint i64 %address.local531, 32, !dbg !21
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local554, i64 %address.local555, !dbg !21
  store half %1709, ptr addrspace(1) %address.local556, align 2, !dbg !21
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local558 = or disjoint i64 %address.local531, 33, !dbg !21
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local557, i64 %address.local558, !dbg !21
  store half %1708, ptr addrspace(1) %address.local559, align 2, !dbg !21
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local561 = or disjoint i64 %address.local531, 34, !dbg !21
  %address.local562 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local560, i64 %address.local561, !dbg !21
  store half %1707, ptr addrspace(1) %address.local562, align 2, !dbg !21
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local564 = or disjoint i64 %address.local531, 35, !dbg !21
  %address.local565 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local563, i64 %address.local564, !dbg !21
  store half %1706, ptr addrspace(1) %address.local565, align 2, !dbg !21
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local567 = or disjoint i64 %address.local531, 48, !dbg !21
  %address.local568 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local566, i64 %address.local567, !dbg !21
  store half %1694, ptr addrspace(1) %address.local568, align 2, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local570 = or disjoint i64 %address.local531, 49, !dbg !21
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local569, i64 %address.local570, !dbg !21
  store half %1693, ptr addrspace(1) %address.local571, align 2, !dbg !21
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local573 = or disjoint i64 %address.local531, 50, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local573, !dbg !21
  store half %1692, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local528, !dbg !21
  %address.local576 = or i64 %address.local530, 51, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local576, !dbg !21
  store half %1691, ptr addrspace(1) %address.local577, align 2, !dbg !21
  %address.local578 = shl nsw i64 %address.local526, 7, !dbg !21
  %address.local579 = add i64 %address.local578, 8192, !dbg !21
  %address.local580 = add nuw nsw i64 %1742, %address.local579, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local531, !dbg !21
  store half %1679, ptr addrspace(1) %address.local582, align 2, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local534, !dbg !21
  store half %1678, ptr addrspace(1) %address.local584, align 2, !dbg !21
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local585, i64 %address.local537, !dbg !21
  store half %1677, ptr addrspace(1) %address.local586, align 2, !dbg !21
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local540, !dbg !21
  store half %1676, ptr addrspace(1) %address.local588, align 2, !dbg !21
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local590 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local589, i64 %address.local543, !dbg !21
  store half %1664, ptr addrspace(1) %address.local590, align 2, !dbg !21
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local591, i64 %address.local546, !dbg !21
  store half %1663, ptr addrspace(1) %address.local592, align 2, !dbg !21
  %address.local593 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local593, i64 %address.local549, !dbg !21
  store half %1662, ptr addrspace(1) %address.local594, align 2, !dbg !21
  %address.local595 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local595, i64 %address.local552, !dbg !21
  store half %1661, ptr addrspace(1) %address.local596, align 2, !dbg !21
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local598 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local597, i64 %address.local555, !dbg !21
  store half %1649, ptr addrspace(1) %address.local598, align 2, !dbg !21
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local599, i64 %address.local558, !dbg !21
  store half %1648, ptr addrspace(1) %address.local600, align 2, !dbg !21
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local601, i64 %address.local561, !dbg !21
  store half %1647, ptr addrspace(1) %address.local602, align 2, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local603, i64 %address.local564, !dbg !21
  store half %1646, ptr addrspace(1) %address.local604, align 2, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local567, !dbg !21
  store half %1634, ptr addrspace(1) %address.local606, align 2, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local607, i64 %address.local570, !dbg !21
  store half %1633, ptr addrspace(1) %address.local608, align 2, !dbg !21
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local609, i64 %address.local573, !dbg !21
  store half %1632, ptr addrspace(1) %address.local610, align 2, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local576, !dbg !21
  store half %1631, ptr addrspace(1) %address.local612, align 2, !dbg !21
  br label %1743, !dbg !21

1743:                                             ; preds = %1619, %1196
  br i1 true, label %1744, label %1818, !dbg !21

1744:                                             ; preds = %1743
  %1745 = insertelement <4 x float> poison, float %1589, i64 0, !dbg !19
  %1746 = insertelement <4 x float> %1745, float %1595, i64 1, !dbg !19
  %1747 = insertelement <4 x float> %1746, float %1590, i64 2, !dbg !19
  %1748 = insertelement <4 x float> %1747, float %1596, i64 3, !dbg !19
  %row.reciprocal80 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1468, !dbg !19
  %normalized81 = fmul <4 x float> %1748, %row.reciprocal80, !dbg !19
  %1749 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !20
  %1750 = extractelement <4 x half> %1749, i64 3, !dbg !20
  %1751 = extractelement <4 x half> %1749, i64 2, !dbg !20
  %1752 = extractelement <4 x half> %1749, i64 1, !dbg !20
  %1753 = extractelement <4 x half> %1749, i64 0, !dbg !20
  %1754 = insertelement <4 x float> poison, float %1539, i64 0, !dbg !19
  %1755 = insertelement <4 x float> %1754, float %1545, i64 1, !dbg !19
  %1756 = insertelement <4 x float> %1755, float %1540, i64 2, !dbg !19
  %1757 = insertelement <4 x float> %1756, float %1546, i64 3, !dbg !19
  %normalized82 = fmul <4 x float> %1757, %row.reciprocal80, !dbg !19
  %1758 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !20
  %1759 = extractelement <4 x half> %1758, i64 3, !dbg !20
  %1760 = extractelement <4 x half> %1758, i64 2, !dbg !20
  %1761 = extractelement <4 x half> %1758, i64 1, !dbg !20
  %1762 = extractelement <4 x half> %1758, i64 0, !dbg !20
  %1763 = insertelement <4 x float> poison, float %1489, i64 0, !dbg !19
  %1764 = insertelement <4 x float> %1763, float %1495, i64 1, !dbg !19
  %1765 = insertelement <4 x float> %1764, float %1490, i64 2, !dbg !19
  %1766 = insertelement <4 x float> %1765, float %1496, i64 3, !dbg !19
  %normalized83 = fmul <4 x float> %1766, %row.reciprocal80, !dbg !19
  %1767 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !20
  %1768 = extractelement <4 x half> %1767, i64 3, !dbg !20
  %1769 = extractelement <4 x half> %1767, i64 2, !dbg !20
  %1770 = extractelement <4 x half> %1767, i64 1, !dbg !20
  %1771 = extractelement <4 x half> %1767, i64 0, !dbg !20
  %1772 = insertelement <4 x float> poison, float %1436, i64 0, !dbg !19
  %1773 = insertelement <4 x float> %1772, float %1442, i64 1, !dbg !19
  %1774 = insertelement <4 x float> %1773, float %1437, i64 2, !dbg !19
  %1775 = insertelement <4 x float> %1774, float %1443, i64 3, !dbg !19
  %normalized84 = fmul <4 x float> %1775, %row.reciprocal80, !dbg !19
  %1776 = fptrunc <4 x float> %normalized84 to <4 x half>, !dbg !20
  %1777 = extractelement <4 x half> %1776, i64 3, !dbg !20
  %1778 = extractelement <4 x half> %1776, i64 2, !dbg !20
  %1779 = extractelement <4 x half> %1776, i64 1, !dbg !20
  %1780 = extractelement <4 x half> %1776, i64 0, !dbg !20
  %1781 = insertelement <4 x float> poison, float %1385, i64 0, !dbg !19
  %1782 = insertelement <4 x float> %1781, float %1391, i64 1, !dbg !19
  %1783 = insertelement <4 x float> %1782, float %1386, i64 2, !dbg !19
  %1784 = insertelement <4 x float> %1783, float %1392, i64 3, !dbg !19
  %row.reciprocal85 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1264, !dbg !19
  %normalized86 = fmul <4 x float> %1784, %row.reciprocal85, !dbg !19
  %1785 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !20
  %1786 = extractelement <4 x half> %1785, i64 3, !dbg !20
  %1787 = extractelement <4 x half> %1785, i64 2, !dbg !20
  %1788 = extractelement <4 x half> %1785, i64 1, !dbg !20
  %1789 = extractelement <4 x half> %1785, i64 0, !dbg !20
  %1790 = insertelement <4 x float> poison, float %1335, i64 0, !dbg !19
  %1791 = insertelement <4 x float> %1790, float %1341, i64 1, !dbg !19
  %1792 = insertelement <4 x float> %1791, float %1336, i64 2, !dbg !19
  %1793 = insertelement <4 x float> %1792, float %1342, i64 3, !dbg !19
  %normalized87 = fmul <4 x float> %1793, %row.reciprocal85, !dbg !19
  %1794 = fptrunc <4 x float> %normalized87 to <4 x half>, !dbg !20
  %1795 = extractelement <4 x half> %1794, i64 3, !dbg !20
  %1796 = extractelement <4 x half> %1794, i64 2, !dbg !20
  %1797 = extractelement <4 x half> %1794, i64 1, !dbg !20
  %1798 = extractelement <4 x half> %1794, i64 0, !dbg !20
  %1799 = insertelement <4 x float> poison, float %1285, i64 0, !dbg !19
  %1800 = insertelement <4 x float> %1799, float %1291, i64 1, !dbg !19
  %1801 = insertelement <4 x float> %1800, float %1286, i64 2, !dbg !19
  %1802 = insertelement <4 x float> %1801, float %1292, i64 3, !dbg !19
  %normalized88 = fmul <4 x float> %1802, %row.reciprocal85, !dbg !19
  %1803 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !20
  %1804 = extractelement <4 x half> %1803, i64 3, !dbg !20
  %1805 = extractelement <4 x half> %1803, i64 2, !dbg !20
  %1806 = extractelement <4 x half> %1803, i64 1, !dbg !20
  %1807 = extractelement <4 x half> %1803, i64 0, !dbg !20
  %1808 = insertelement <4 x float> poison, float %1232, i64 0, !dbg !19
  %1809 = insertelement <4 x float> %1808, float %1238, i64 1, !dbg !19
  %1810 = insertelement <4 x float> %1809, float %1233, i64 2, !dbg !19
  %1811 = insertelement <4 x float> %1810, float %1239, i64 3, !dbg !19
  %normalized89 = fmul <4 x float> %1811, %row.reciprocal85, !dbg !19
  %1812 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !20
  %1813 = extractelement <4 x half> %1812, i64 3, !dbg !20
  %1814 = extractelement <4 x half> %1812, i64 2, !dbg !20
  %1815 = extractelement <4 x half> %1812, i64 1, !dbg !20
  %1816 = extractelement <4 x half> %1812, i64 0, !dbg !20
  %1817 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread613 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local614 = zext nneg i32 %address.thread613 to i64, !dbg !13
  %address.local615 = add nsw i64 %16, %address.local614, !dbg !21
  %address.local616 = lshr i64 %address.local614, 6, !dbg !13
  %address.local617 = shl nuw nsw i64 %address.local616, 4, !dbg !21
  %address.local618 = add nsw i64 %address.local615, %address.local617, !dbg !21
  %address.local619 = and i64 %address.local614, 240, !dbg !21
  %address.local620 = sub nsw i64 %address.local618, %address.local619, !dbg !21
  %address.local621 = add nsw i64 %address.local620, %17, !dbg !21
  %address.local622 = shl nuw nsw i64 %address.local621, 7, !dbg !21
  %address.local623 = add nuw nsw i64 %1817, %address.local622, !dbg !21
  %address.local624 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local625 = lshr i64 %address.local614, 2, !dbg !13
  %address.local626 = and i64 %address.local625, 12, !dbg !13
  %address.local627 = or disjoint i64 %address.local626, 64, !dbg !21
  %address.local628 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local624, i64 %address.local627, !dbg !21
  store half %1816, ptr addrspace(1) %address.local628, align 2, !dbg !21
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local630 = or disjoint i64 %address.local626, 65, !dbg !21
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local629, i64 %address.local630, !dbg !21
  store half %1815, ptr addrspace(1) %address.local631, align 2, !dbg !21
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local633 = or disjoint i64 %address.local626, 66, !dbg !21
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local632, i64 %address.local633, !dbg !21
  store half %1814, ptr addrspace(1) %address.local634, align 2, !dbg !21
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local636 = or disjoint i64 %address.local626, 67, !dbg !21
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local636, !dbg !21
  store half %1813, ptr addrspace(1) %address.local637, align 2, !dbg !21
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local639 = or disjoint i64 %address.local626, 80, !dbg !21
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local638, i64 %address.local639, !dbg !21
  store half %1807, ptr addrspace(1) %address.local640, align 2, !dbg !21
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local642 = or disjoint i64 %address.local626, 81, !dbg !21
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local641, i64 %address.local642, !dbg !21
  store half %1806, ptr addrspace(1) %address.local643, align 2, !dbg !21
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local645 = or disjoint i64 %address.local626, 82, !dbg !21
  %address.local646 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local644, i64 %address.local645, !dbg !21
  store half %1805, ptr addrspace(1) %address.local646, align 2, !dbg !21
  %address.local647 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local648 = or disjoint i64 %address.local626, 83, !dbg !21
  %address.local649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local647, i64 %address.local648, !dbg !21
  store half %1804, ptr addrspace(1) %address.local649, align 2, !dbg !21
  %address.local650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local651 = or disjoint i64 %address.local626, 96, !dbg !21
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local650, i64 %address.local651, !dbg !21
  store half %1798, ptr addrspace(1) %address.local652, align 2, !dbg !21
  %address.local653 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local654 = or disjoint i64 %address.local626, 97, !dbg !21
  %address.local655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local653, i64 %address.local654, !dbg !21
  store half %1797, ptr addrspace(1) %address.local655, align 2, !dbg !21
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local657 = or disjoint i64 %address.local626, 98, !dbg !21
  %address.local658 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local656, i64 %address.local657, !dbg !21
  store half %1796, ptr addrspace(1) %address.local658, align 2, !dbg !21
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local660 = or disjoint i64 %address.local626, 99, !dbg !21
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local660, !dbg !21
  store half %1795, ptr addrspace(1) %address.local661, align 2, !dbg !21
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local663 = or disjoint i64 %address.local626, 112, !dbg !21
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local663, !dbg !21
  store half %1789, ptr addrspace(1) %address.local664, align 2, !dbg !21
  %address.local665 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local666 = or disjoint i64 %address.local626, 113, !dbg !21
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local665, i64 %address.local666, !dbg !21
  store half %1788, ptr addrspace(1) %address.local667, align 2, !dbg !21
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local669 = or disjoint i64 %address.local626, 114, !dbg !21
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local668, i64 %address.local669, !dbg !21
  store half %1787, ptr addrspace(1) %address.local670, align 2, !dbg !21
  %address.local671 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local623, !dbg !21
  %address.local672 = or i64 %address.local625, 115, !dbg !21
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local671, i64 %address.local672, !dbg !21
  store half %1786, ptr addrspace(1) %address.local673, align 2, !dbg !21
  %address.local674 = shl nsw i64 %address.local621, 7, !dbg !21
  %address.local675 = add i64 %address.local674, 8192, !dbg !21
  %address.local676 = add nuw nsw i64 %1817, %address.local675, !dbg !21
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local678 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local677, i64 %address.local627, !dbg !21
  store half %1780, ptr addrspace(1) %address.local678, align 2, !dbg !21
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local679, i64 %address.local630, !dbg !21
  store half %1779, ptr addrspace(1) %address.local680, align 2, !dbg !21
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local681, i64 %address.local633, !dbg !21
  store half %1778, ptr addrspace(1) %address.local682, align 2, !dbg !21
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local684 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local683, i64 %address.local636, !dbg !21
  store half %1777, ptr addrspace(1) %address.local684, align 2, !dbg !21
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local685, i64 %address.local639, !dbg !21
  store half %1771, ptr addrspace(1) %address.local686, align 2, !dbg !21
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local687, i64 %address.local642, !dbg !21
  store half %1770, ptr addrspace(1) %address.local688, align 2, !dbg !21
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local645, !dbg !21
  store half %1769, ptr addrspace(1) %address.local690, align 2, !dbg !21
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local648, !dbg !21
  store half %1768, ptr addrspace(1) %address.local692, align 2, !dbg !21
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local651, !dbg !21
  store half %1762, ptr addrspace(1) %address.local694, align 2, !dbg !21
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local654, !dbg !21
  store half %1761, ptr addrspace(1) %address.local696, align 2, !dbg !21
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local697, i64 %address.local657, !dbg !21
  store half %1760, ptr addrspace(1) %address.local698, align 2, !dbg !21
  %address.local699 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local699, i64 %address.local660, !dbg !21
  store half %1759, ptr addrspace(1) %address.local700, align 2, !dbg !21
  %address.local701 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local702 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local701, i64 %address.local663, !dbg !21
  store half %1753, ptr addrspace(1) %address.local702, align 2, !dbg !21
  %address.local703 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local704 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local703, i64 %address.local666, !dbg !21
  store half %1752, ptr addrspace(1) %address.local704, align 2, !dbg !21
  %address.local705 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local705, i64 %address.local669, !dbg !21
  store half %1751, ptr addrspace(1) %address.local706, align 2, !dbg !21
  %address.local707 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local676, !dbg !21
  %address.local708 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local707, i64 %address.local672, !dbg !21
  store half %1750, ptr addrspace(1) %address.local708, align 2, !dbg !21
  br label %1818, !dbg !21

1818:                                             ; preds = %1744, %1743
  ret void, !dbg !5
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

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { convergent nounwind memory(none) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!33}

!0 = !DIFile(filename: "unequal_input_steps.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles")
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
!15 = !DILocation(line: 56, column: 24, scope: !4)
!16 = !DILocation(line: 67, column: 22, scope: !4)
!17 = !DILocation(line: 92, column: 20, scope: !4)
!18 = !DILocation(line: 93, column: 23, scope: !4)
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
