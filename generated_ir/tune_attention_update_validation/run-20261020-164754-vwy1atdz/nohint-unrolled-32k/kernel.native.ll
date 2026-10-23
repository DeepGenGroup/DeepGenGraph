; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/tune_attention_update_validation/run-20261020-164754-vwy1atdz/nohint-unrolled-32k/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/tune_attention_update_validation/run-20261020-164754-vwy1atdz/nohint-unrolled-32k/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !31 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %10 = sext i32 %9 to i64, !dbg !5
  %11 = shl nsw i64 %8, 7, !dbg !9
  %.lobit = ashr i32 %7, 31, !dbg !11
  %12 = xor i32 %7, %.lobit, !dbg !11
  %13 = sdiv i32 %12, 32, !dbg !11
  %.lobit1 = ashr i32 %7, 31, !dbg !11
  %14 = xor i32 %13, %.lobit1, !dbg !11
  %15 = sext i32 %14 to i64, !dbg !11
  %16 = add nsw i64 %10, %15, !dbg !11
  %17 = and i64 %16, 31, !dbg !11
  %18 = shl nsw i64 %8, 7, !dbg !11
  %19 = mul nsw i64 %15, -4096, !dbg !11
  %20 = add nsw i64 %18, %19, !dbg !11
  %21 = shl nuw nsw i64 %17, 19, !dbg !11
  %22 = shl nsw i64 %20, 7, !dbg !11
  %23 = add nsw i64 %21, %22, !dbg !11
  %24 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !33
  %25 = zext nneg i32 %24 to i64, !dbg !12
  %26 = lshr i64 %25, 6, !dbg !12
  %27 = or disjoint i64 %11, 127, !dbg !13
  %28 = add nsw i64 %11, 128, !dbg !13
  %29 = icmp uge i64 %28, %27, !dbg !13
  %30 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %31 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %32 = getelementptr i8, ptr addrspace(1) %31, i64 64, !dbg !11
  %33 = shl nsw i64 %10, 7, !dbg !16
  %34 = shl nsw i64 %6, 12, !dbg !16
  %35 = add nsw i64 %33, %34, !dbg !16
  %36 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %37 = getelementptr i8, ptr addrspace(1) %36, i64 128, !dbg !11
  %38 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %23, !dbg !11
  %39 = getelementptr i8, ptr addrspace(1) %38, i64 192, !dbg !11
  br label %40, !dbg !15

40:                                               ; preds = %826, %4
  %41 = phi i64 [ %1559, %826 ], [ 0, %4 ], !dbg !15
  %42 = phi [2 x <8 x float>] [ %1546, %826 ], [ zeroinitializer, %4 ], !dbg !15
  %43 = phi [2 x <1 x float>] [ %1168, %826 ], [ zeroinitializer, %4 ], !dbg !15
  %44 = phi [2 x <8 x float>] [ %1550, %826 ], [ zeroinitializer, %4 ], !dbg !15
  %45 = phi [2 x <8 x float>] [ %1554, %826 ], [ zeroinitializer, %4 ], !dbg !15
  %46 = phi [2 x <8 x float>] [ %1558, %826 ], [ zeroinitializer, %4 ], !dbg !15
  %.not = icmp sgt i64 %41, %8, !dbg !15
  br i1 %.not, label %1560, label %47, !dbg !15

47:                                               ; preds = %40
  %48 = shl nsw i64 %41, 7, !dbg !15
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !11, !range !33, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local129 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local130 = and i64 %address.local129, 1536, !dbg !11
  %address.local131 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local130, !dbg !11
  %address.local132 = ptrtoint ptr addrspace(3) %address.local131 to i64, !dbg !11
  %address.local133 = trunc i64 %address.local132 to i32, !dbg !11
  %49 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %50 = zext i32 %49 to i64, !dbg !11
  %51 = inttoptr i64 %50 to ptr addrspace(3), !dbg !11
  %address.local134 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local135 = and i64 %address.local134, 16256, !dbg !11
  %address.local136 = lshr i64 %address.local, 4, !dbg !11
  %address.local137 = and i64 %address.local136, 8, !dbg !11
  %address.local138 = or disjoint i64 %address.local135, %address.local137, !dbg !11
  %address.local139 = getelementptr [2 x i8], ptr addrspace(1) %30, i64 %address.local138, !dbg !11
  %52 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %53 = ptrtoint ptr addrspace(1) %address.local139 to i64, !dbg !11
  %54 = sub i64 %53, %52, !dbg !11
  %dma.buffer.offset934 = trunc i64 %54 to i32, !dbg !11
  %55 = trunc i64 %52 to i32, !dbg !11
  %56 = insertelement <4 x i32> poison, i32 %55, i64 0, !dbg !11
  %57 = lshr i64 %52, 32, !dbg !11
  %58 = trunc i64 %57 to i32, !dbg !11
  %59 = insertelement <4 x i32> %56, i32 %58, i64 1, !dbg !11
  %60 = insertelement <4 x i32> %59, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc935 = insertelement <4 x i32> %60, i32 131072, i64 3, !dbg !11
  %61 = ptrtoint ptr addrspace(3) %51 to i32, !dbg !11
  %dma.buffer.lds936 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !11
  %62 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds936, i32 %dma.buffer.offset934, <4 x i32> %dma.buffer.rsrc935) #10, !dbg !11
  %address.local140 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local141 = shl nuw nsw i64 %address.local140, 3, !dbg !11
  %address.local142 = and i64 %address.local141, 3584, !dbg !11
  %address.local143 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local142, !dbg !11
  %address.local144 = ptrtoint ptr addrspace(3) %address.local143 to i64, !dbg !11
  %address.local145 = trunc i64 %address.local144 to i32, !dbg !11
  %63 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %64 = zext i32 %63 to i64, !dbg !11
  %65 = inttoptr i64 %64 to ptr addrspace(3), !dbg !11
  %address.local146 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local147 = and i64 %address.local146, 16256, !dbg !11
  %address.local148 = lshr i64 %address.local140, 4, !dbg !11
  %address.local149 = and i64 %address.local148, 24, !dbg !11
  %address.local150 = or disjoint i64 %address.local147, %address.local149, !dbg !11
  %address.local151 = getelementptr [2 x i8], ptr addrspace(1) %30, i64 %address.local150, !dbg !11
  %66 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %67 = ptrtoint ptr addrspace(1) %address.local151 to i64, !dbg !11
  %68 = sub i64 %67, %66, !dbg !11
  %dma.buffer.offset931 = trunc i64 %68 to i32, !dbg !11
  %69 = trunc i64 %66 to i32, !dbg !11
  %70 = insertelement <4 x i32> poison, i32 %69, i64 0, !dbg !11
  %71 = lshr i64 %66, 32, !dbg !11
  %72 = trunc i64 %71 to i32, !dbg !11
  %73 = insertelement <4 x i32> %70, i32 %72, i64 1, !dbg !11
  %74 = insertelement <4 x i32> %73, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc932 = insertelement <4 x i32> %74, i32 131072, i64 3, !dbg !11
  %75 = ptrtoint ptr addrspace(3) %65 to i32, !dbg !11
  %dma.buffer.lds933 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %75), !dbg !11
  %76 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds933, i32 %dma.buffer.offset931, <4 x i32> %dma.buffer.rsrc932) #10, !dbg !11
  %77 = lshr i64 %41, 12, !dbg !16
  %78 = add nsw i64 %77, %10, !dbg !16
  %79 = lshr i64 %41, 5, !dbg !16
  %80 = and i64 %79, 127, !dbg !16
  %81 = shl i64 %78, 19, !dbg !16
  %82 = and i64 %81, 16252928, !dbg !16
  %83 = shl i64 %79, 19, !dbg !16
  %84 = shl i64 %41, 14, !dbg !16
  %85 = sub i64 %84, %83, !dbg !16
  %86 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %82, !dbg !16
  %87 = getelementptr [2 x i8], ptr addrspace(1) %86, i64 %80, !dbg !16
  %88 = getelementptr [2 x i8], ptr addrspace(1) %87, i64 %85, !dbg !16
  %address.local152 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local130, !dbg !16
  %address.local153 = ptrtoint ptr addrspace(3) %address.local152 to i64, !dbg !16
  %address.local154 = trunc i64 %address.local153 to i32, !dbg !16
  %89 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !16
  %90 = zext i32 %89 to i64, !dbg !16
  %91 = inttoptr i64 %90 to ptr addrspace(3), !dbg !16
  %address.local155 = getelementptr [2 x i8], ptr addrspace(1) %88, i64 %address.local138, !dbg !16
  %92 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %93 = ptrtoint ptr addrspace(1) %address.local155 to i64, !dbg !16
  %94 = sub i64 %93, %92, !dbg !16
  %dma.buffer.offset928 = trunc i64 %94 to i32, !dbg !16
  %95 = trunc i64 %92 to i32, !dbg !16
  %96 = insertelement <4 x i32> poison, i32 %95, i64 0, !dbg !16
  %97 = lshr i64 %92, 32, !dbg !16
  %98 = trunc i64 %97 to i32, !dbg !16
  %99 = insertelement <4 x i32> %96, i32 %98, i64 1, !dbg !16
  %100 = insertelement <4 x i32> %99, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc929 = insertelement <4 x i32> %100, i32 131072, i64 3, !dbg !16
  %101 = ptrtoint ptr addrspace(3) %91 to i32, !dbg !16
  %dma.buffer.lds930 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %101), !dbg !16
  %102 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds930, i32 %dma.buffer.offset928, <4 x i32> %dma.buffer.rsrc929) #10, !dbg !16
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local142, !dbg !16
  %address.local157 = ptrtoint ptr addrspace(3) %address.local156 to i64, !dbg !16
  %address.local158 = trunc i64 %address.local157 to i32, !dbg !16
  %103 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !16
  %104 = zext i32 %103 to i64, !dbg !16
  %105 = inttoptr i64 %104 to ptr addrspace(3), !dbg !16
  %address.local159 = getelementptr [2 x i8], ptr addrspace(1) %88, i64 %address.local150, !dbg !16
  %106 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %107 = ptrtoint ptr addrspace(1) %address.local159 to i64, !dbg !16
  %108 = sub i64 %107, %106, !dbg !16
  %dma.buffer.offset925 = trunc i64 %108 to i32, !dbg !16
  %109 = trunc i64 %106 to i32, !dbg !16
  %110 = insertelement <4 x i32> poison, i32 %109, i64 0, !dbg !16
  %111 = lshr i64 %106, 32, !dbg !16
  %112 = trunc i64 %111 to i32, !dbg !16
  %113 = insertelement <4 x i32> %110, i32 %112, i64 1, !dbg !16
  %114 = insertelement <4 x i32> %113, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc926 = insertelement <4 x i32> %114, i32 131072, i64 3, !dbg !16
  %115 = ptrtoint ptr addrspace(3) %105 to i32, !dbg !16
  %dma.buffer.lds927 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %115), !dbg !16
  %116 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds927, i32 %dma.buffer.offset925, <4 x i32> %dma.buffer.rsrc926) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local160 = lshr i32 %address.thread, 2, !dbg !12
  %address.local161 = and i32 %address.local160, 4, !dbg !12
  %address.local162 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local163 = and i32 %address.local162, 384, !dbg !12
  %address.local164 = or disjoint i32 %address.local161, %address.local163, !dbg !12
  %address.local165 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local166 = and i32 %address.local165, 120, !dbg !12
  %address.local167 = or disjoint i32 %address.local164, %address.local166, !dbg !12
  %address.local168 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local169 = and i32 %address.local168, 1024, !dbg !12
  %address.local170 = or disjoint i32 %address.local167, %address.local169, !dbg !12
  %address.local171 = zext nneg i32 %address.local170 to i64, !dbg !12
  %address.local172 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %117 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !12
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %address.local174 = getelementptr i8, ptr addrspace(3) %address.local173, i64 4096, !dbg !12
  %118 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !12
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %address.local176 = getelementptr i8, ptr addrspace(3) %address.local175, i64 1024, !dbg !12
  %119 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !12
  %120 = fmul <4 x half> %117, splat (half 0xH3015), !dbg !12
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local177, i64 5120, !dbg !12
  %121 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !12
  %122 = fmul <4 x half> %118, splat (half 0xH3015), !dbg !12
  %123 = fmul <4 x half> %119, splat (half 0xH3015), !dbg !12
  %124 = fmul <4 x half> %121, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local179 = lshr i32 %address.thread, 2, !dbg !17
  %address.local180 = and i32 %address.local179, 4, !dbg !17
  %address.local181 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local182 = and i32 %address.local181, 24, !dbg !17
  %address.local183 = or disjoint i32 %address.local180, %address.local182, !dbg !17
  %address.local184 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local185 = and i32 %address.local184, 1024, !dbg !17
  %address.local186 = or disjoint i32 %address.local183, %address.local185, !dbg !17
  %address.local187 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local188 = and i32 %address.local187, 384, !dbg !17
  %address.local189 = or disjoint i32 %address.local186, %address.local188, !dbg !17
  %address.local190 = zext nneg i32 %address.local189 to i64, !dbg !17
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local190, !dbg !17
  %125 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !17
  %126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %125, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local192 = lshr i32 %address.thread, 2, !dbg !17
  %address.local193 = and i32 %address.local192, 4, !dbg !17
  %address.local194 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local195 = and i32 %address.local194, 24, !dbg !17
  %address.local196 = or disjoint i32 %address.local193, %address.local195, !dbg !17
  %address.local197 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local198 = and i32 %address.local197, 1024, !dbg !17
  %address.local199 = or disjoint i32 %address.local196, %address.local198, !dbg !17
  %address.local200 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local201 = and i32 %address.local200, 384, !dbg !17
  %address.local202 = or disjoint i32 %address.local199, %address.local201, !dbg !17
  %address.local203 = zext nneg i32 %address.local202 to i64, !dbg !17
  %address.local204 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !17
  %address.local205 = getelementptr i8, ptr addrspace(3) %address.local204, i64 4096, !dbg !17
  %127 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !17
  %address.local206 = shl nuw nsw i64 %address.local, 7, !dbg !17
  %address.local207 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local208 = lshr i64 %address.local, 2, !dbg !17
  %address.local209 = mul nsw i64 %address.local208, -512, !dbg !17
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i64 %address.local209, !dbg !17
  %address.local211 = shl nuw nsw i64 %address.local, 1, !dbg !17
  %address.local212 = and i64 %address.local211, 24, !dbg !17
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) %address.local210, i64 %address.local212, !dbg !17
  %address.local214 = lshr i64 %address.local, 2, !dbg !12
  %address.local215 = and i64 %address.local214, 12, !dbg !12
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) %address.local213, i64 %address.local215, !dbg !17
  %address.local217 = and i64 %address.local, 63, !dbg !12
  %address.local218 = lshr i64 %address.local217, 5, !dbg !12
  %address.local219 = mul nuw nsw i64 %address.local218, 1016, !dbg !12
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local216, i64 %address.local219, !dbg !17
  %address.local221 = getelementptr i8, ptr addrspace(3) %address.local220, i64 64, !dbg !17
  %128 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !17
  %129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %127, <4 x float> %126) #9, !dbg !17
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %128, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local222, i64 %address.local215, !dbg !17
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local219, !dbg !17
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) %address.local224, i64 %address.local209, !dbg !17
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local212, !dbg !17
  %address.local227 = getelementptr i8, ptr addrspace(3) %address.local226, i64 4160, !dbg !17
  %131 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !17
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local209, !dbg !17
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local212, !dbg !17
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) %address.local230, i64 %address.local215, !dbg !17
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %address.local231, i64 %address.local219, !dbg !17
  %address.local233 = getelementptr i8, ptr addrspace(3) %address.local232, i64 128, !dbg !17
  %132 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !17
  %133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %131, <4 x float> %130) #9, !dbg !17
  %134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %132, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local215, !dbg !17
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local219, !dbg !17
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) %address.local236, i64 %address.local209, !dbg !17
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %address.local237, i64 %address.local212, !dbg !17
  %address.local239 = getelementptr i8, ptr addrspace(3) %address.local238, i64 4224, !dbg !17
  %135 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !17
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local209, !dbg !17
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local212, !dbg !17
  %address.local243 = getelementptr [2 x i8], ptr addrspace(3) %address.local242, i64 %address.local215, !dbg !17
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %address.local243, i64 %address.local219, !dbg !17
  %address.local245 = getelementptr i8, ptr addrspace(3) %address.local244, i64 192, !dbg !17
  %136 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !17
  %137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %135, <4 x float> %134) #9, !dbg !17
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %136, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local246 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) %address.local246, i64 %address.local215, !dbg !17
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local219, !dbg !17
  %address.local249 = getelementptr [2 x i8], ptr addrspace(3) %address.local248, i64 %address.local209, !dbg !17
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) %address.local249, i64 %address.local212, !dbg !17
  %address.local251 = getelementptr i8, ptr addrspace(3) %address.local250, i64 4288, !dbg !17
  %139 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !17
  %address.local252 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local190, !dbg !17
  %address.local253 = getelementptr i8, ptr addrspace(3) %address.local252, i64 1024, !dbg !17
  %140 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !17
  %141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %139, <4 x float> %138) #9, !dbg !17
  %142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %140, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !17
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local254, i64 5120, !dbg !17
  %143 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !17
  %address.local256 = add nuw nsw i64 %address.local206, 512, !dbg !17
  %address.local257 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local256, !dbg !17
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) %address.local257, i64 %address.local209, !dbg !17
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) %address.local258, i64 %address.local212, !dbg !17
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %address.local259, i64 %address.local215, !dbg !17
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local219, !dbg !17
  %address.local262 = getelementptr i8, ptr addrspace(3) %address.local261, i64 64, !dbg !17
  %144 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !17
  %145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %143, <4 x float> %142) #9, !dbg !17
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %144, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local256, !dbg !17
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local215, !dbg !17
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local219, !dbg !17
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) %address.local265, i64 %address.local209, !dbg !17
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local266, i64 %address.local212, !dbg !17
  %address.local268 = getelementptr i8, ptr addrspace(3) %address.local267, i64 4160, !dbg !17
  %147 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !17
  %address.local269 = add nuw nsw i64 %address.local206, 512, !dbg !17
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local269, !dbg !17
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local209, !dbg !17
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) %address.local271, i64 %address.local212, !dbg !17
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) %address.local272, i64 %address.local215, !dbg !17
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local219, !dbg !17
  %address.local275 = getelementptr i8, ptr addrspace(3) %address.local274, i64 128, !dbg !17
  %148 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !17
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %147, <4 x float> %146) #9, !dbg !17
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %148, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local269, !dbg !17
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) %address.local276, i64 %address.local215, !dbg !17
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local219, !dbg !17
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) %address.local278, i64 %address.local209, !dbg !17
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %address.local279, i64 %address.local212, !dbg !17
  %address.local281 = getelementptr i8, ptr addrspace(3) %address.local280, i64 4224, !dbg !17
  %151 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !17
  %address.local282 = add nuw nsw i64 %address.local206, 512, !dbg !17
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local282, !dbg !17
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %address.local283, i64 %address.local209, !dbg !17
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) %address.local284, i64 %address.local212, !dbg !17
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) %address.local285, i64 %address.local215, !dbg !17
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %address.local286, i64 %address.local219, !dbg !17
  %address.local288 = getelementptr i8, ptr addrspace(3) %address.local287, i64 192, !dbg !17
  %152 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %151, <4 x float> %150) #9, !dbg !17
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %152, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local282, !dbg !17
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %address.local289, i64 %address.local215, !dbg !17
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local219, !dbg !17
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) %address.local291, i64 %address.local209, !dbg !17
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %address.local292, i64 %address.local212, !dbg !17
  %address.local294 = getelementptr i8, ptr addrspace(3) %address.local293, i64 4288, !dbg !17
  %155 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  %156 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !17
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %155, <4 x float> %154) #9, !dbg !17
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %156, <4 x float> zeroinitializer) #9, !dbg !17
  %159 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !17
  %160 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !17
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %159, <4 x float> %158) #9, !dbg !17
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %160, <4 x float> zeroinitializer) #9, !dbg !17
  %163 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !17
  %164 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !17
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %163, <4 x float> %162) #9, !dbg !17
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %164, <4 x float> zeroinitializer) #9, !dbg !17
  %167 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !17
  %168 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !17
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %167, <4 x float> %166) #9, !dbg !17
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %168, <4 x float> zeroinitializer) #9, !dbg !17
  %171 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !17
  %172 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !17
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %171, <4 x float> %170) #9, !dbg !17
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %172, <4 x float> zeroinitializer) #9, !dbg !17
  %175 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !17
  %176 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !17
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %175, <4 x float> %174) #9, !dbg !17
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %176, <4 x float> zeroinitializer) #9, !dbg !17
  %179 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !17
  %180 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !17
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %179, <4 x float> %178) #9, !dbg !17
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %180, <4 x float> zeroinitializer) #9, !dbg !17
  %183 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !17
  %184 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %183, <4 x float> %182) #9, !dbg !17
  %186 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %184, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %186, <4 x float> %187) #9, !dbg !17
  %189 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %190 = zext i32 %189 to i64, !dbg !11
  %191 = inttoptr i64 %190 to ptr addrspace(3), !dbg !11
  %address.local295 = getelementptr [2 x i8], ptr addrspace(1) %32, i64 %address.local138, !dbg !11
  %192 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %193 = ptrtoint ptr addrspace(1) %address.local295 to i64, !dbg !11
  %194 = sub i64 %193, %192, !dbg !11
  %dma.buffer.offset922 = trunc i64 %194 to i32, !dbg !11
  %195 = trunc i64 %192 to i32, !dbg !11
  %196 = insertelement <4 x i32> poison, i32 %195, i64 0, !dbg !11
  %197 = lshr i64 %192, 32, !dbg !11
  %198 = trunc i64 %197 to i32, !dbg !11
  %199 = insertelement <4 x i32> %196, i32 %198, i64 1, !dbg !11
  %200 = insertelement <4 x i32> %199, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc923 = insertelement <4 x i32> %200, i32 131072, i64 3, !dbg !11
  %201 = ptrtoint ptr addrspace(3) %191 to i32, !dbg !11
  %dma.buffer.lds924 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %201), !dbg !11
  %202 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds924, i32 %dma.buffer.offset922, <4 x i32> %dma.buffer.rsrc923) #10, !dbg !11
  %203 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %204 = zext i32 %203 to i64, !dbg !11
  %205 = inttoptr i64 %204 to ptr addrspace(3), !dbg !11
  %address.local296 = getelementptr [2 x i8], ptr addrspace(1) %32, i64 %address.local150, !dbg !11
  %206 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %207 = ptrtoint ptr addrspace(1) %address.local296 to i64, !dbg !11
  %208 = sub i64 %207, %206, !dbg !11
  %dma.buffer.offset919 = trunc i64 %208 to i32, !dbg !11
  %209 = trunc i64 %206 to i32, !dbg !11
  %210 = insertelement <4 x i32> poison, i32 %209, i64 0, !dbg !11
  %211 = lshr i64 %206, 32, !dbg !11
  %212 = trunc i64 %211 to i32, !dbg !11
  %213 = insertelement <4 x i32> %210, i32 %212, i64 1, !dbg !11
  %214 = insertelement <4 x i32> %213, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc920 = insertelement <4 x i32> %214, i32 131072, i64 3, !dbg !11
  %215 = ptrtoint ptr addrspace(3) %205 to i32, !dbg !11
  %dma.buffer.lds921 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %215), !dbg !11
  %216 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds921, i32 %dma.buffer.offset919, <4 x i32> %dma.buffer.rsrc920) #10, !dbg !11
  %217 = add nuw i64 %41, 1024, !dbg !16
  %218 = icmp slt i64 %217, 0, !dbg !16
  %219 = sub nuw i64 -1025, %41, !dbg !16
  %220 = select i1 %218, i64 %219, i64 %217, !dbg !16
  %221 = sdiv i64 %220, 4096, !dbg !16
  %.lobit4 = ashr i64 %217, 63, !dbg !16
  %222 = xor i64 %221, %.lobit4, !dbg !16
  %223 = add nsw i64 %222, %10, !dbg !16
  %224 = add nsw i64 %79, %35, !dbg !16
  %225 = add nsw i64 %224, 32, !dbg !16
  %226 = icmp slt i64 %224, -32, !dbg !16
  %227 = sub nsw i64 -33, %224, !dbg !16
  %228 = select i1 %226, i64 %227, i64 %225, !dbg !16
  %229 = sdiv i64 %228, 128, !dbg !16
  %230 = sext i1 %226 to i64, !dbg !16
  %231 = xor i64 %229, %230, !dbg !16
  %232 = shl nsw i64 %231, 7, !dbg !16
  %233 = sub i64 %35, %232, !dbg !16
  %234 = shl i64 %223, 19, !dbg !16
  %235 = and i64 %234, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %236 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %235, !dbg !16
  %237 = getelementptr [2 x i8], ptr addrspace(1) %236, i64 %79, !dbg !16
  %238 = getelementptr [2 x i8], ptr addrspace(1) %237, i64 %233, !dbg !16
  %239 = getelementptr i8, ptr addrspace(1) %238, i64 64, !dbg !16
  %240 = getelementptr [2 x i8], ptr addrspace(1) %239, i64 %85, !dbg !16
  %241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !16
  %242 = zext i32 %241 to i64, !dbg !16
  %243 = inttoptr i64 %242 to ptr addrspace(3), !dbg !16
  %address.local297 = getelementptr [2 x i8], ptr addrspace(1) %240, i64 %address.local138, !dbg !16
  %244 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %245 = ptrtoint ptr addrspace(1) %address.local297 to i64, !dbg !16
  %246 = sub i64 %245, %244, !dbg !16
  %dma.buffer.offset916 = trunc i64 %246 to i32, !dbg !16
  %247 = trunc i64 %244 to i32, !dbg !16
  %248 = insertelement <4 x i32> poison, i32 %247, i64 0, !dbg !16
  %249 = lshr i64 %244, 32, !dbg !16
  %250 = trunc i64 %249 to i32, !dbg !16
  %251 = insertelement <4 x i32> %248, i32 %250, i64 1, !dbg !16
  %252 = insertelement <4 x i32> %251, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc917 = insertelement <4 x i32> %252, i32 131072, i64 3, !dbg !16
  %253 = ptrtoint ptr addrspace(3) %243 to i32, !dbg !16
  %dma.buffer.lds918 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %253), !dbg !16
  %254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds918, i32 %dma.buffer.offset916, <4 x i32> %dma.buffer.rsrc917) #10, !dbg !16
  %255 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !16
  %256 = zext i32 %255 to i64, !dbg !16
  %257 = inttoptr i64 %256 to ptr addrspace(3), !dbg !16
  %address.local298 = getelementptr [2 x i8], ptr addrspace(1) %240, i64 %address.local150, !dbg !16
  %258 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %259 = ptrtoint ptr addrspace(1) %address.local298 to i64, !dbg !16
  %260 = sub i64 %259, %258, !dbg !16
  %dma.buffer.offset913 = trunc i64 %260 to i32, !dbg !16
  %261 = trunc i64 %258 to i32, !dbg !16
  %262 = insertelement <4 x i32> poison, i32 %261, i64 0, !dbg !16
  %263 = lshr i64 %258, 32, !dbg !16
  %264 = trunc i64 %263 to i32, !dbg !16
  %265 = insertelement <4 x i32> %262, i32 %264, i64 1, !dbg !16
  %266 = insertelement <4 x i32> %265, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc914 = insertelement <4 x i32> %266, i32 131072, i64 3, !dbg !16
  %267 = ptrtoint ptr addrspace(3) %257 to i32, !dbg !16
  %dma.buffer.lds915 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %267), !dbg !16
  %268 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds915, i32 %dma.buffer.offset913, <4 x i32> %dma.buffer.rsrc914) #10, !dbg !16
  %269 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !12
  %270 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !12
  %271 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !12
  %272 = fmul <4 x half> %269, splat (half 0xH3015), !dbg !12
  %273 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !12
  %274 = fmul <4 x half> %270, splat (half 0xH3015), !dbg !12
  %275 = fmul <4 x half> %271, splat (half 0xH3015), !dbg !12
  %276 = fmul <4 x half> %273, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %277 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !17
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %272, <4 x half> %277, <4 x float> %129) #9, !dbg !17
  %279 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !17
  %280 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !17
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %274, <4 x half> %279, <4 x float> %278) #9, !dbg !17
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %272, <4 x half> %280, <4 x float> %133) #9, !dbg !17
  %283 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !17
  %284 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !17
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %274, <4 x half> %283, <4 x float> %282) #9, !dbg !17
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %272, <4 x half> %284, <4 x float> %137) #9, !dbg !17
  %287 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !17
  %288 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !17
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %274, <4 x half> %287, <4 x float> %286) #9, !dbg !17
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %272, <4 x half> %288, <4 x float> %141) #9, !dbg !17
  %291 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !17
  %292 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !17
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %274, <4 x half> %291, <4 x float> %290) #9, !dbg !17
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %272, <4 x half> %292, <4 x float> %145) #9, !dbg !17
  %295 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !17
  %296 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !17
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %274, <4 x half> %295, <4 x float> %294) #9, !dbg !17
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %272, <4 x half> %296, <4 x float> %149) #9, !dbg !17
  %299 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !17
  %300 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !17
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %274, <4 x half> %299, <4 x float> %298) #9, !dbg !17
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %272, <4 x half> %300, <4 x float> %153) #9, !dbg !17
  %303 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !17
  %304 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %274, <4 x half> %303, <4 x float> %302) #9, !dbg !17
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %272, <4 x half> %304, <4 x float> %157) #9, !dbg !17
  %307 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  %308 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !17
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %274, <4 x half> %307, <4 x float> %306) #9, !dbg !17
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %308, <4 x float> %161) #9, !dbg !17
  %311 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !17
  %312 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !17
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %311, <4 x float> %310) #9, !dbg !17
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %312, <4 x float> %165) #9, !dbg !17
  %315 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !17
  %316 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !17
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %315, <4 x float> %314) #9, !dbg !17
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %316, <4 x float> %169) #9, !dbg !17
  %319 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !17
  %320 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !17
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %319, <4 x float> %318) #9, !dbg !17
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %320, <4 x float> %173) #9, !dbg !17
  %323 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !17
  %324 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !17
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %323, <4 x float> %322) #9, !dbg !17
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %324, <4 x float> %177) #9, !dbg !17
  %327 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !17
  %328 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !17
  %329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %327, <4 x float> %326) #9, !dbg !17
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %328, <4 x float> %181) #9, !dbg !17
  %331 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !17
  %332 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !17
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %331, <4 x float> %330) #9, !dbg !17
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %332, <4 x float> %185) #9, !dbg !17
  %335 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !17
  %336 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %335, <4 x float> %334) #9, !dbg !17
  %338 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %275, <4 x half> %336, <4 x float> %188) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %276, <4 x half> %338, <4 x float> %339) #9, !dbg !17
  %341 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %342 = zext i32 %341 to i64, !dbg !11
  %343 = inttoptr i64 %342 to ptr addrspace(3), !dbg !11
  %address.local299 = getelementptr [2 x i8], ptr addrspace(1) %37, i64 %address.local138, !dbg !11
  %344 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %345 = ptrtoint ptr addrspace(1) %address.local299 to i64, !dbg !11
  %346 = sub i64 %345, %344, !dbg !11
  %dma.buffer.offset910 = trunc i64 %346 to i32, !dbg !11
  %347 = trunc i64 %344 to i32, !dbg !11
  %348 = insertelement <4 x i32> poison, i32 %347, i64 0, !dbg !11
  %349 = lshr i64 %344, 32, !dbg !11
  %350 = trunc i64 %349 to i32, !dbg !11
  %351 = insertelement <4 x i32> %348, i32 %350, i64 1, !dbg !11
  %352 = insertelement <4 x i32> %351, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc911 = insertelement <4 x i32> %352, i32 131072, i64 3, !dbg !11
  %353 = ptrtoint ptr addrspace(3) %343 to i32, !dbg !11
  %dma.buffer.lds912 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %353), !dbg !11
  %354 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds912, i32 %dma.buffer.offset910, <4 x i32> %dma.buffer.rsrc911) #10, !dbg !11
  %355 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %356 = zext i32 %355 to i64, !dbg !11
  %357 = inttoptr i64 %356 to ptr addrspace(3), !dbg !11
  %address.local300 = getelementptr [2 x i8], ptr addrspace(1) %37, i64 %address.local150, !dbg !11
  %358 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %359 = ptrtoint ptr addrspace(1) %address.local300 to i64, !dbg !11
  %360 = sub i64 %359, %358, !dbg !11
  %dma.buffer.offset907 = trunc i64 %360 to i32, !dbg !11
  %361 = trunc i64 %358 to i32, !dbg !11
  %362 = insertelement <4 x i32> poison, i32 %361, i64 0, !dbg !11
  %363 = lshr i64 %358, 32, !dbg !11
  %364 = trunc i64 %363 to i32, !dbg !11
  %365 = insertelement <4 x i32> %362, i32 %364, i64 1, !dbg !11
  %366 = insertelement <4 x i32> %365, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc908 = insertelement <4 x i32> %366, i32 131072, i64 3, !dbg !11
  %367 = ptrtoint ptr addrspace(3) %357 to i32, !dbg !11
  %dma.buffer.lds909 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %367), !dbg !11
  %368 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds909, i32 %dma.buffer.offset907, <4 x i32> %dma.buffer.rsrc908) #10, !dbg !11
  %369 = add nuw i64 %41, 2048, !dbg !16
  %370 = icmp slt i64 %369, 0, !dbg !16
  %371 = sub nuw i64 -2049, %41, !dbg !16
  %372 = select i1 %370, i64 %371, i64 %369, !dbg !16
  %373 = sdiv i64 %372, 4096, !dbg !16
  %.lobit5 = ashr i64 %369, 63, !dbg !16
  %374 = xor i64 %373, %.lobit5, !dbg !16
  %375 = add nsw i64 %374, %10, !dbg !16
  %376 = add nsw i64 %224, 64, !dbg !16
  %377 = icmp slt i64 %224, -64, !dbg !16
  %378 = sub nsw i64 -65, %224, !dbg !16
  %379 = select i1 %377, i64 %378, i64 %376, !dbg !16
  %380 = sdiv i64 %379, 128, !dbg !16
  %381 = sext i1 %377 to i64, !dbg !16
  %382 = xor i64 %380, %381, !dbg !16
  %383 = shl nsw i64 %382, 7, !dbg !16
  %384 = sub i64 %35, %383, !dbg !16
  %385 = shl i64 %375, 19, !dbg !16
  %386 = and i64 %385, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %387 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %386, !dbg !16
  %388 = getelementptr [2 x i8], ptr addrspace(1) %387, i64 %79, !dbg !16
  %389 = getelementptr [2 x i8], ptr addrspace(1) %388, i64 %384, !dbg !16
  %390 = getelementptr i8, ptr addrspace(1) %389, i64 128, !dbg !16
  %391 = getelementptr [2 x i8], ptr addrspace(1) %390, i64 %85, !dbg !16
  %392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !16
  %393 = zext i32 %392 to i64, !dbg !16
  %394 = inttoptr i64 %393 to ptr addrspace(3), !dbg !16
  %address.local301 = getelementptr [2 x i8], ptr addrspace(1) %391, i64 %address.local138, !dbg !16
  %395 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %396 = ptrtoint ptr addrspace(1) %address.local301 to i64, !dbg !16
  %397 = sub i64 %396, %395, !dbg !16
  %dma.buffer.offset904 = trunc i64 %397 to i32, !dbg !16
  %398 = trunc i64 %395 to i32, !dbg !16
  %399 = insertelement <4 x i32> poison, i32 %398, i64 0, !dbg !16
  %400 = lshr i64 %395, 32, !dbg !16
  %401 = trunc i64 %400 to i32, !dbg !16
  %402 = insertelement <4 x i32> %399, i32 %401, i64 1, !dbg !16
  %403 = insertelement <4 x i32> %402, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc905 = insertelement <4 x i32> %403, i32 131072, i64 3, !dbg !16
  %404 = ptrtoint ptr addrspace(3) %394 to i32, !dbg !16
  %dma.buffer.lds906 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %404), !dbg !16
  %405 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds906, i32 %dma.buffer.offset904, <4 x i32> %dma.buffer.rsrc905) #10, !dbg !16
  %406 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !16
  %407 = zext i32 %406 to i64, !dbg !16
  %408 = inttoptr i64 %407 to ptr addrspace(3), !dbg !16
  %address.local302 = getelementptr [2 x i8], ptr addrspace(1) %391, i64 %address.local150, !dbg !16
  %409 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %410 = ptrtoint ptr addrspace(1) %address.local302 to i64, !dbg !16
  %411 = sub i64 %410, %409, !dbg !16
  %dma.buffer.offset901 = trunc i64 %411 to i32, !dbg !16
  %412 = trunc i64 %409 to i32, !dbg !16
  %413 = insertelement <4 x i32> poison, i32 %412, i64 0, !dbg !16
  %414 = lshr i64 %409, 32, !dbg !16
  %415 = trunc i64 %414 to i32, !dbg !16
  %416 = insertelement <4 x i32> %413, i32 %415, i64 1, !dbg !16
  %417 = insertelement <4 x i32> %416, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc902 = insertelement <4 x i32> %417, i32 131072, i64 3, !dbg !16
  %418 = ptrtoint ptr addrspace(3) %408 to i32, !dbg !16
  %dma.buffer.lds903 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %418), !dbg !16
  %419 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds903, i32 %dma.buffer.offset901, <4 x i32> %dma.buffer.rsrc902) #10, !dbg !16
  %420 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !12
  %421 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !12
  %422 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !12
  %423 = fmul <4 x half> %420, splat (half 0xH3015), !dbg !12
  %424 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !12
  %425 = fmul <4 x half> %421, splat (half 0xH3015), !dbg !12
  %426 = fmul <4 x half> %422, splat (half 0xH3015), !dbg !12
  %427 = fmul <4 x half> %424, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %428 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !17
  %429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %423, <4 x half> %428, <4 x float> %281) #9, !dbg !17
  %430 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !17
  %431 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !17
  %432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %425, <4 x half> %430, <4 x float> %429) #9, !dbg !17
  %433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %423, <4 x half> %431, <4 x float> %285) #9, !dbg !17
  %434 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !17
  %435 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !17
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %425, <4 x half> %434, <4 x float> %433) #9, !dbg !17
  %437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %423, <4 x half> %435, <4 x float> %289) #9, !dbg !17
  %438 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !17
  %439 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !17
  %440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %425, <4 x half> %438, <4 x float> %437) #9, !dbg !17
  %441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %423, <4 x half> %439, <4 x float> %293) #9, !dbg !17
  %442 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !17
  %443 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !17
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %425, <4 x half> %442, <4 x float> %441) #9, !dbg !17
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %423, <4 x half> %443, <4 x float> %297) #9, !dbg !17
  %446 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !17
  %447 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !17
  %448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %425, <4 x half> %446, <4 x float> %445) #9, !dbg !17
  %449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %423, <4 x half> %447, <4 x float> %301) #9, !dbg !17
  %450 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !17
  %451 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !17
  %452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %425, <4 x half> %450, <4 x float> %449) #9, !dbg !17
  %453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %423, <4 x half> %451, <4 x float> %305) #9, !dbg !17
  %454 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !17
  %455 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %425, <4 x half> %454, <4 x float> %453) #9, !dbg !17
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %423, <4 x half> %455, <4 x float> %309) #9, !dbg !17
  %458 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  %459 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !17
  %460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %425, <4 x half> %458, <4 x float> %457) #9, !dbg !17
  %461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %459, <4 x float> %313) #9, !dbg !17
  %462 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !17
  %463 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !17
  %464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %462, <4 x float> %461) #9, !dbg !17
  %465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %463, <4 x float> %317) #9, !dbg !17
  %466 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !17
  %467 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !17
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %466, <4 x float> %465) #9, !dbg !17
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %467, <4 x float> %321) #9, !dbg !17
  %470 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !17
  %471 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !17
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %470, <4 x float> %469) #9, !dbg !17
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %471, <4 x float> %325) #9, !dbg !17
  %474 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !17
  %475 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !17
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %474, <4 x float> %473) #9, !dbg !17
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %475, <4 x float> %329) #9, !dbg !17
  %478 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !17
  %479 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !17
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %478, <4 x float> %477) #9, !dbg !17
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %479, <4 x float> %333) #9, !dbg !17
  %482 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !17
  %483 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !17
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %482, <4 x float> %481) #9, !dbg !17
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %483, <4 x float> %337) #9, !dbg !17
  %486 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !17
  %487 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %486, <4 x float> %485) #9, !dbg !17
  %489 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %426, <4 x half> %487, <4 x float> %340) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %427, <4 x half> %489, <4 x float> %490) #9, !dbg !17
  %492 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %493 = zext i32 %492 to i64, !dbg !11
  %494 = inttoptr i64 %493 to ptr addrspace(3), !dbg !11
  %address.local303 = getelementptr [2 x i8], ptr addrspace(1) %39, i64 %address.local138, !dbg !11
  %495 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %496 = ptrtoint ptr addrspace(1) %address.local303 to i64, !dbg !11
  %497 = sub i64 %496, %495, !dbg !11
  %dma.buffer.offset898 = trunc i64 %497 to i32, !dbg !11
  %498 = trunc i64 %495 to i32, !dbg !11
  %499 = insertelement <4 x i32> poison, i32 %498, i64 0, !dbg !11
  %500 = lshr i64 %495, 32, !dbg !11
  %501 = trunc i64 %500 to i32, !dbg !11
  %502 = insertelement <4 x i32> %499, i32 %501, i64 1, !dbg !11
  %503 = insertelement <4 x i32> %502, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc899 = insertelement <4 x i32> %503, i32 131072, i64 3, !dbg !11
  %504 = ptrtoint ptr addrspace(3) %494 to i32, !dbg !11
  %dma.buffer.lds900 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %504), !dbg !11
  %505 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds900, i32 %dma.buffer.offset898, <4 x i32> %dma.buffer.rsrc899) #10, !dbg !11
  %506 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %507 = zext i32 %506 to i64, !dbg !11
  %508 = inttoptr i64 %507 to ptr addrspace(3), !dbg !11
  %address.local304 = getelementptr [2 x i8], ptr addrspace(1) %39, i64 %address.local150, !dbg !11
  %509 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %510 = ptrtoint ptr addrspace(1) %address.local304 to i64, !dbg !11
  %511 = sub i64 %510, %509, !dbg !11
  %dma.buffer.offset895 = trunc i64 %511 to i32, !dbg !11
  %512 = trunc i64 %509 to i32, !dbg !11
  %513 = insertelement <4 x i32> poison, i32 %512, i64 0, !dbg !11
  %514 = lshr i64 %509, 32, !dbg !11
  %515 = trunc i64 %514 to i32, !dbg !11
  %516 = insertelement <4 x i32> %513, i32 %515, i64 1, !dbg !11
  %517 = insertelement <4 x i32> %516, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc896 = insertelement <4 x i32> %517, i32 131072, i64 3, !dbg !11
  %518 = ptrtoint ptr addrspace(3) %508 to i32, !dbg !11
  %dma.buffer.lds897 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %518), !dbg !11
  %519 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds897, i32 %dma.buffer.offset895, <4 x i32> %dma.buffer.rsrc896) #10, !dbg !11
  %520 = add nuw i64 %41, 3072, !dbg !16
  %521 = icmp slt i64 %520, 0, !dbg !16
  %522 = sub nuw i64 -3073, %41, !dbg !16
  %523 = select i1 %521, i64 %522, i64 %520, !dbg !16
  %524 = sdiv i64 %523, 4096, !dbg !16
  %.lobit6 = ashr i64 %520, 63, !dbg !16
  %525 = xor i64 %524, %.lobit6, !dbg !16
  %526 = add nsw i64 %525, %10, !dbg !16
  %527 = add nsw i64 %224, 96, !dbg !16
  %528 = icmp slt i64 %224, -96, !dbg !16
  %529 = sub nsw i64 -97, %224, !dbg !16
  %530 = select i1 %528, i64 %529, i64 %527, !dbg !16
  %531 = sdiv i64 %530, 128, !dbg !16
  %532 = sext i1 %528 to i64, !dbg !16
  %533 = xor i64 %531, %532, !dbg !16
  %534 = shl nsw i64 %533, 7, !dbg !16
  %535 = sub i64 %35, %534, !dbg !16
  %536 = shl i64 %526, 19, !dbg !16
  %537 = and i64 %536, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %538 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %537, !dbg !16
  %539 = getelementptr [2 x i8], ptr addrspace(1) %538, i64 %79, !dbg !16
  %540 = getelementptr [2 x i8], ptr addrspace(1) %539, i64 %535, !dbg !16
  %541 = getelementptr i8, ptr addrspace(1) %540, i64 192, !dbg !16
  %542 = getelementptr [2 x i8], ptr addrspace(1) %541, i64 %85, !dbg !16
  %543 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !16
  %544 = zext i32 %543 to i64, !dbg !16
  %545 = inttoptr i64 %544 to ptr addrspace(3), !dbg !16
  %address.local305 = getelementptr [2 x i8], ptr addrspace(1) %542, i64 %address.local138, !dbg !16
  %546 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %547 = ptrtoint ptr addrspace(1) %address.local305 to i64, !dbg !16
  %548 = sub i64 %547, %546, !dbg !16
  %dma.buffer.offset892 = trunc i64 %548 to i32, !dbg !16
  %549 = trunc i64 %546 to i32, !dbg !16
  %550 = insertelement <4 x i32> poison, i32 %549, i64 0, !dbg !16
  %551 = lshr i64 %546, 32, !dbg !16
  %552 = trunc i64 %551 to i32, !dbg !16
  %553 = insertelement <4 x i32> %550, i32 %552, i64 1, !dbg !16
  %554 = insertelement <4 x i32> %553, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc893 = insertelement <4 x i32> %554, i32 131072, i64 3, !dbg !16
  %555 = ptrtoint ptr addrspace(3) %545 to i32, !dbg !16
  %dma.buffer.lds894 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %555), !dbg !16
  %556 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds894, i32 %dma.buffer.offset892, <4 x i32> %dma.buffer.rsrc893) #10, !dbg !16
  %557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !16
  %558 = zext i32 %557 to i64, !dbg !16
  %559 = inttoptr i64 %558 to ptr addrspace(3), !dbg !16
  %address.local306 = getelementptr [2 x i8], ptr addrspace(1) %542, i64 %address.local150, !dbg !16
  %560 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %561 = ptrtoint ptr addrspace(1) %address.local306 to i64, !dbg !16
  %562 = sub i64 %561, %560, !dbg !16
  %dma.buffer.offset889 = trunc i64 %562 to i32, !dbg !16
  %563 = trunc i64 %560 to i32, !dbg !16
  %564 = insertelement <4 x i32> poison, i32 %563, i64 0, !dbg !16
  %565 = lshr i64 %560, 32, !dbg !16
  %566 = trunc i64 %565 to i32, !dbg !16
  %567 = insertelement <4 x i32> %564, i32 %566, i64 1, !dbg !16
  %568 = insertelement <4 x i32> %567, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc890 = insertelement <4 x i32> %568, i32 131072, i64 3, !dbg !16
  %569 = ptrtoint ptr addrspace(3) %559 to i32, !dbg !16
  %dma.buffer.lds891 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %569), !dbg !16
  %570 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds891, i32 %dma.buffer.offset889, <4 x i32> %dma.buffer.rsrc890) #10, !dbg !16
  %571 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !12
  %572 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !12
  %573 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !12
  %574 = fmul <4 x half> %571, splat (half 0xH3015), !dbg !12
  %575 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !12
  %576 = fmul <4 x half> %572, splat (half 0xH3015), !dbg !12
  %577 = fmul <4 x half> %573, splat (half 0xH3015), !dbg !12
  %578 = fmul <4 x half> %575, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %579 = add nsw i64 %79, %10, !dbg !18
  %580 = shl i64 %579, 19, !dbg !18
  %581 = and i64 %580, 16252928, !dbg !18
  %582 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %581, !dbg !18
  %583 = getelementptr [2 x i8], ptr addrspace(1) %582, i64 %85, !dbg !18
  %584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !18
  %585 = zext i32 %584 to i64, !dbg !18
  %586 = inttoptr i64 %585 to ptr addrspace(3), !dbg !18
  %address.local307 = shl nuw nsw i64 %address.local, 5, !dbg !18
  %address.local308 = and i64 %address.local307, 8064, !dbg !18
  %address.local309 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local310 = and i64 %address.local309, 24, !dbg !18
  %address.local311 = or disjoint i64 %address.local308, %address.local310, !dbg !18
  %address.local312 = getelementptr [2 x i8], ptr addrspace(1) %583, i64 %address.local311, !dbg !18
  %587 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %588 = ptrtoint ptr addrspace(1) %address.local312 to i64, !dbg !18
  %589 = sub i64 %588, %587, !dbg !18
  %dma.buffer.offset886 = trunc i64 %589 to i32, !dbg !18
  %590 = trunc i64 %587 to i32, !dbg !18
  %591 = insertelement <4 x i32> poison, i32 %590, i64 0, !dbg !18
  %592 = lshr i64 %587, 32, !dbg !18
  %593 = trunc i64 %592 to i32, !dbg !18
  %594 = insertelement <4 x i32> %591, i32 %593, i64 1, !dbg !18
  %595 = insertelement <4 x i32> %594, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc887 = insertelement <4 x i32> %595, i32 131072, i64 3, !dbg !18
  %596 = ptrtoint ptr addrspace(3) %586 to i32, !dbg !18
  %dma.buffer.lds888 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %596), !dbg !18
  %597 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds888, i32 %dma.buffer.offset886, <4 x i32> %dma.buffer.rsrc887) #10, !dbg !18
  %598 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !18
  %599 = zext i32 %598 to i64, !dbg !18
  %600 = inttoptr i64 %599 to ptr addrspace(3), !dbg !18
  %address.local313 = shl nuw nsw i64 %address.local140, 5, !dbg !18
  %address.local314 = and i64 %address.local313, 16256, !dbg !18
  %address.local315 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local316 = and i64 %address.local315, 24, !dbg !18
  %address.local317 = or disjoint i64 %address.local314, %address.local316, !dbg !18
  %address.local318 = getelementptr [2 x i8], ptr addrspace(1) %583, i64 %address.local317, !dbg !18
  %601 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %602 = ptrtoint ptr addrspace(1) %address.local318 to i64, !dbg !18
  %603 = sub i64 %602, %601, !dbg !18
  %dma.buffer.offset883 = trunc i64 %603 to i32, !dbg !18
  %604 = trunc i64 %601 to i32, !dbg !18
  %605 = insertelement <4 x i32> poison, i32 %604, i64 0, !dbg !18
  %606 = lshr i64 %601, 32, !dbg !18
  %607 = trunc i64 %606 to i32, !dbg !18
  %608 = insertelement <4 x i32> %605, i32 %607, i64 1, !dbg !18
  %609 = insertelement <4 x i32> %608, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc884 = insertelement <4 x i32> %609, i32 131072, i64 3, !dbg !18
  %610 = ptrtoint ptr addrspace(3) %600 to i32, !dbg !18
  %dma.buffer.lds885 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %610), !dbg !18
  %611 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds885, i32 %dma.buffer.offset883, <4 x i32> %dma.buffer.rsrc884) #10, !dbg !18
  %612 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !17
  %613 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %574, <4 x half> %612, <4 x float> %432) #9, !dbg !17
  %614 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !17
  %615 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !17
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %576, <4 x half> %614, <4 x float> %613) #9, !dbg !17
  %617 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %574, <4 x half> %615, <4 x float> %436) #9, !dbg !17
  %618 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !17
  %619 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !17
  %620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %576, <4 x half> %618, <4 x float> %617) #9, !dbg !17
  %621 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %574, <4 x half> %619, <4 x float> %440) #9, !dbg !17
  %622 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !17
  %623 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !17
  %624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %576, <4 x half> %622, <4 x float> %621) #9, !dbg !17
  %625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %574, <4 x half> %623, <4 x float> %444) #9, !dbg !17
  %626 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !17
  %627 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !17
  %628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %576, <4 x half> %626, <4 x float> %625) #9, !dbg !17
  %629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %574, <4 x half> %627, <4 x float> %448) #9, !dbg !17
  %630 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !17
  %631 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !17
  %632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %576, <4 x half> %630, <4 x float> %629) #9, !dbg !17
  %633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %574, <4 x half> %631, <4 x float> %452) #9, !dbg !17
  %634 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !17
  %635 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !17
  %636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %576, <4 x half> %634, <4 x float> %633) #9, !dbg !17
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %574, <4 x half> %635, <4 x float> %456) #9, !dbg !17
  %638 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !17
  %639 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %576, <4 x half> %638, <4 x float> %637) #9, !dbg !17
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %574, <4 x half> %639, <4 x float> %460) #9, !dbg !17
  %642 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  %643 = load <4 x half>, ptr addrspace(3) %address.local191, align 2, !dbg !17
  %644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %576, <4 x half> %642, <4 x float> %641) #9, !dbg !17
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %643, <4 x float> %464) #9, !dbg !17
  %646 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !17
  %647 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !17
  %648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %646, <4 x float> %645) #9, !dbg !17
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %647, <4 x float> %468) #9, !dbg !17
  %650 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !17
  %651 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !17
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %650, <4 x float> %649) #9, !dbg !17
  %653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %651, <4 x float> %472) #9, !dbg !17
  %654 = load <4 x half>, ptr addrspace(3) %address.local239, align 2, !dbg !17
  %655 = load <4 x half>, ptr addrspace(3) %address.local245, align 2, !dbg !17
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %654, <4 x float> %653) #9, !dbg !17
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %655, <4 x float> %476) #9, !dbg !17
  %658 = load <4 x half>, ptr addrspace(3) %address.local251, align 2, !dbg !17
  %659 = load <4 x half>, ptr addrspace(3) %address.local253, align 2, !dbg !17
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %658, <4 x float> %657) #9, !dbg !17
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %659, <4 x float> %480) #9, !dbg !17
  %662 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !17
  %663 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !17
  %664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %662, <4 x float> %661) #9, !dbg !17
  %665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %663, <4 x float> %484) #9, !dbg !17
  %666 = load <4 x half>, ptr addrspace(3) %address.local268, align 2, !dbg !17
  %667 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !17
  %668 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %666, <4 x float> %665) #9, !dbg !17
  %669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %667, <4 x float> %488) #9, !dbg !17
  %670 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !17
  %671 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %670, <4 x float> %669) #9, !dbg !17
  %673 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %577, <4 x half> %671, <4 x float> %491) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %675 = or disjoint i64 %48, 127, !dbg !13
  %676 = icmp uge i64 %11, %675, !dbg !13
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %578, <4 x half> %673, <4 x float> %674) #9, !dbg !17
  %678 = and i1 %29, %676, !dbg !13
  br i1 %678, label %679, label %680, !dbg !13

679:                                              ; preds = %47
  br label %824, !dbg !13

680:                                              ; preds = %47
  %681 = add nsw i64 %11, %25, !dbg !13
  %682 = shl nuw nsw i64 %26, 4, !dbg !13
  %683 = add nsw i64 %681, %682, !dbg !13
  %684 = and i32 %24, 240, !dbg !13
  %index.compact = zext nneg i32 %684 to i64, !dbg !13
  %685 = sub nsw i64 %683, %index.compact, !dbg !13
  %686 = and i64 %25, 48, !dbg !13
  %687 = or disjoint i64 %48, %686, !dbg !13
  %688 = add nsw i64 %685, 1, !dbg !23
  %689 = lshr i64 %688, 32, !dbg !24
  %compare.high = trunc i64 %689 to i32, !dbg !24
  %690 = lshr i64 %687, 32, !dbg !24
  %compare.high612 = trunc i64 %690 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high612, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high612, !dbg !24
  %compare.low.base = trunc i64 %688 to i32, !dbg !24
  %compare.low.base613 = trunc i64 %687 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base613, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %691 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low614 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result615 = select i1 %compare.high.equal, i1 %compare.low614, i1 %compare.high.order, !dbg !24
  %692 = select i1 %compare.result615, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base616 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local617 = or i32 %compare.local.base616, 2, !dbg !24
  %compare.low618 = icmp ugt i32 %compare.low.base, %compare.local617, !dbg !24
  %compare.result619 = select i1 %compare.high.equal, i1 %compare.low618, i1 %compare.high.order, !dbg !24
  %693 = select i1 %compare.result619, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base620 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local621 = or i32 %compare.local.base620, 3, !dbg !24
  %compare.low622 = icmp ugt i32 %compare.low.base, %compare.local621, !dbg !24
  %compare.result623 = select i1 %compare.high.equal, i1 %compare.low622, i1 %compare.high.order, !dbg !24
  %694 = select i1 %compare.result623, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base624 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local625 = or i32 %compare.local.base624, 4, !dbg !24
  %compare.low626 = icmp ugt i32 %compare.low.base, %compare.local625, !dbg !24
  %compare.result627 = select i1 %compare.high.equal, i1 %compare.low626, i1 %compare.high.order, !dbg !24
  %695 = select i1 %compare.result627, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base628 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local629 = or i32 %compare.local.base628, 5, !dbg !24
  %compare.low630 = icmp ugt i32 %compare.low.base, %compare.local629, !dbg !24
  %compare.result631 = select i1 %compare.high.equal, i1 %compare.low630, i1 %compare.high.order, !dbg !24
  %696 = select i1 %compare.result631, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base632 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local633 = or i32 %compare.local.base632, 6, !dbg !24
  %compare.low634 = icmp ugt i32 %compare.low.base, %compare.local633, !dbg !24
  %compare.result635 = select i1 %compare.high.equal, i1 %compare.low634, i1 %compare.high.order, !dbg !24
  %697 = select i1 %compare.result635, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base636 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local637 = or i32 %compare.local.base636, 7, !dbg !24
  %compare.low638 = icmp ugt i32 %compare.low.base, %compare.local637, !dbg !24
  %compare.result639 = select i1 %compare.high.equal, i1 %compare.low638, i1 %compare.high.order, !dbg !24
  %698 = select i1 %compare.result639, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base640 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local641 = or i32 %compare.local.base640, 8, !dbg !24
  %compare.low642 = icmp ugt i32 %compare.low.base, %compare.local641, !dbg !24
  %compare.result643 = select i1 %compare.high.equal, i1 %compare.low642, i1 %compare.high.order, !dbg !24
  %699 = select i1 %compare.result643, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base644 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local645 = or i32 %compare.local.base644, 9, !dbg !24
  %compare.low646 = icmp ugt i32 %compare.low.base, %compare.local645, !dbg !24
  %compare.result647 = select i1 %compare.high.equal, i1 %compare.low646, i1 %compare.high.order, !dbg !24
  %700 = select i1 %compare.result647, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base648 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local649 = or i32 %compare.local.base648, 10, !dbg !24
  %compare.low650 = icmp ugt i32 %compare.low.base, %compare.local649, !dbg !24
  %compare.result651 = select i1 %compare.high.equal, i1 %compare.low650, i1 %compare.high.order, !dbg !24
  %701 = select i1 %compare.result651, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base652 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local653 = or i32 %compare.local.base652, 11, !dbg !24
  %compare.low654 = icmp ugt i32 %compare.low.base, %compare.local653, !dbg !24
  %compare.result655 = select i1 %compare.high.equal, i1 %compare.low654, i1 %compare.high.order, !dbg !24
  %702 = select i1 %compare.result655, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base656 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local657 = or i32 %compare.local.base656, 12, !dbg !24
  %compare.low658 = icmp ugt i32 %compare.low.base, %compare.local657, !dbg !24
  %compare.result659 = select i1 %compare.high.equal, i1 %compare.low658, i1 %compare.high.order, !dbg !24
  %703 = select i1 %compare.result659, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base660 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local661 = or i32 %compare.local.base660, 13, !dbg !24
  %compare.low662 = icmp ugt i32 %compare.low.base, %compare.local661, !dbg !24
  %compare.result663 = select i1 %compare.high.equal, i1 %compare.low662, i1 %compare.high.order, !dbg !24
  %704 = select i1 %compare.result663, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base664 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local665 = or i32 %compare.local.base664, 14, !dbg !24
  %compare.low666 = icmp ugt i32 %compare.low.base, %compare.local665, !dbg !24
  %compare.result667 = select i1 %compare.high.equal, i1 %compare.low666, i1 %compare.high.order, !dbg !24
  %705 = select i1 %compare.result667, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base668 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local669 = or i32 %compare.local.base668, 15, !dbg !24
  %compare.low670 = icmp ugt i32 %compare.low.base, %compare.local669, !dbg !24
  %compare.result671 = select i1 %compare.high.equal, i1 %compare.low670, i1 %compare.high.order, !dbg !24
  %706 = select i1 %compare.result671, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base672 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local673 = or i32 %compare.local.base672, 64, !dbg !24
  %compare.low674 = icmp ugt i32 %compare.low.base, %compare.local673, !dbg !24
  %compare.result675 = select i1 %compare.high.equal, i1 %compare.low674, i1 %compare.high.order, !dbg !24
  %707 = select i1 %compare.result675, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base676 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local677 = or i32 %compare.local.base676, 65, !dbg !24
  %compare.low678 = icmp ugt i32 %compare.low.base, %compare.local677, !dbg !24
  %compare.result679 = select i1 %compare.high.equal, i1 %compare.low678, i1 %compare.high.order, !dbg !24
  %708 = select i1 %compare.result679, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base680 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local681 = or i32 %compare.local.base680, 66, !dbg !24
  %compare.low682 = icmp ugt i32 %compare.low.base, %compare.local681, !dbg !24
  %compare.result683 = select i1 %compare.high.equal, i1 %compare.low682, i1 %compare.high.order, !dbg !24
  %709 = select i1 %compare.result683, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base684 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local685 = or i32 %compare.local.base684, 67, !dbg !24
  %compare.low686 = icmp ugt i32 %compare.low.base, %compare.local685, !dbg !24
  %compare.result687 = select i1 %compare.high.equal, i1 %compare.low686, i1 %compare.high.order, !dbg !24
  %710 = select i1 %compare.result687, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base688 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local689 = or i32 %compare.local.base688, 68, !dbg !24
  %compare.low690 = icmp ugt i32 %compare.low.base, %compare.local689, !dbg !24
  %compare.result691 = select i1 %compare.high.equal, i1 %compare.low690, i1 %compare.high.order, !dbg !24
  %711 = select i1 %compare.result691, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base692 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local693 = or i32 %compare.local.base692, 69, !dbg !24
  %compare.low694 = icmp ugt i32 %compare.low.base, %compare.local693, !dbg !24
  %compare.result695 = select i1 %compare.high.equal, i1 %compare.low694, i1 %compare.high.order, !dbg !24
  %712 = select i1 %compare.result695, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base696 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local697 = or i32 %compare.local.base696, 70, !dbg !24
  %compare.low698 = icmp ugt i32 %compare.low.base, %compare.local697, !dbg !24
  %compare.result699 = select i1 %compare.high.equal, i1 %compare.low698, i1 %compare.high.order, !dbg !24
  %713 = select i1 %compare.result699, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base700 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local701 = or i32 %compare.local.base700, 71, !dbg !24
  %compare.low702 = icmp ugt i32 %compare.low.base, %compare.local701, !dbg !24
  %compare.result703 = select i1 %compare.high.equal, i1 %compare.low702, i1 %compare.high.order, !dbg !24
  %714 = select i1 %compare.result703, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base704 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local705 = or i32 %compare.local.base704, 72, !dbg !24
  %compare.low706 = icmp ugt i32 %compare.low.base, %compare.local705, !dbg !24
  %compare.result707 = select i1 %compare.high.equal, i1 %compare.low706, i1 %compare.high.order, !dbg !24
  %715 = select i1 %compare.result707, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base708 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local709 = or i32 %compare.local.base708, 73, !dbg !24
  %compare.low710 = icmp ugt i32 %compare.low.base, %compare.local709, !dbg !24
  %compare.result711 = select i1 %compare.high.equal, i1 %compare.low710, i1 %compare.high.order, !dbg !24
  %716 = select i1 %compare.result711, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base712 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local713 = or i32 %compare.local.base712, 74, !dbg !24
  %compare.low714 = icmp ugt i32 %compare.low.base, %compare.local713, !dbg !24
  %compare.result715 = select i1 %compare.high.equal, i1 %compare.low714, i1 %compare.high.order, !dbg !24
  %717 = select i1 %compare.result715, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base716 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local717 = or i32 %compare.local.base716, 75, !dbg !24
  %compare.low718 = icmp ugt i32 %compare.low.base, %compare.local717, !dbg !24
  %compare.result719 = select i1 %compare.high.equal, i1 %compare.low718, i1 %compare.high.order, !dbg !24
  %718 = select i1 %compare.result719, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base720 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local721 = or i32 %compare.local.base720, 76, !dbg !24
  %compare.low722 = icmp ugt i32 %compare.low.base, %compare.local721, !dbg !24
  %compare.result723 = select i1 %compare.high.equal, i1 %compare.low722, i1 %compare.high.order, !dbg !24
  %719 = select i1 %compare.result723, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base724 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local725 = or i32 %compare.local.base724, 77, !dbg !24
  %compare.low726 = icmp ugt i32 %compare.low.base, %compare.local725, !dbg !24
  %compare.result727 = select i1 %compare.high.equal, i1 %compare.low726, i1 %compare.high.order, !dbg !24
  %720 = select i1 %compare.result727, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base728 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local729 = or i32 %compare.local.base728, 78, !dbg !24
  %compare.low730 = icmp ugt i32 %compare.low.base, %compare.local729, !dbg !24
  %compare.result731 = select i1 %compare.high.equal, i1 %compare.low730, i1 %compare.high.order, !dbg !24
  %721 = select i1 %compare.result731, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base732 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local733 = or i32 %compare.local.base732, 79, !dbg !24
  %compare.low734 = icmp ugt i32 %compare.low.base, %compare.local733, !dbg !24
  %compare.result735 = select i1 %compare.high.equal, i1 %compare.low734, i1 %compare.high.order, !dbg !24
  %722 = select i1 %compare.result735, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %723 = add nsw i64 %685, 65, !dbg !23
  %724 = lshr i64 %723, 32, !dbg !24
  %compare.high736 = trunc i64 %724 to i32, !dbg !24
  %725 = lshr i64 %687, 32, !dbg !24
  %compare.high737 = trunc i64 %725 to i32, !dbg !24
  %compare.high.equal738 = icmp eq i32 %compare.high736, %compare.high737, !dbg !24
  %compare.high.order739 = icmp ugt i32 %compare.high736, %compare.high737, !dbg !24
  %compare.low.base740 = trunc i64 %723 to i32, !dbg !24
  %compare.low.base741 = trunc i64 %687 to i32, !dbg !24
  %compare.low742 = icmp ugt i32 %compare.low.base740, %compare.low.base741, !dbg !24
  %compare.result743 = select i1 %compare.high.equal738, i1 %compare.low742, i1 %compare.high.order739, !dbg !24
  %726 = select i1 %compare.result743, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base744 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local745 = or i32 %compare.local.base744, 1, !dbg !24
  %compare.low746 = icmp ugt i32 %compare.low.base740, %compare.local745, !dbg !24
  %compare.result747 = select i1 %compare.high.equal738, i1 %compare.low746, i1 %compare.high.order739, !dbg !24
  %727 = select i1 %compare.result747, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base748 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local749 = or i32 %compare.local.base748, 2, !dbg !24
  %compare.low750 = icmp ugt i32 %compare.low.base740, %compare.local749, !dbg !24
  %compare.result751 = select i1 %compare.high.equal738, i1 %compare.low750, i1 %compare.high.order739, !dbg !24
  %728 = select i1 %compare.result751, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base752 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local753 = or i32 %compare.local.base752, 3, !dbg !24
  %compare.low754 = icmp ugt i32 %compare.low.base740, %compare.local753, !dbg !24
  %compare.result755 = select i1 %compare.high.equal738, i1 %compare.low754, i1 %compare.high.order739, !dbg !24
  %729 = select i1 %compare.result755, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base756 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local757 = or i32 %compare.local.base756, 4, !dbg !24
  %compare.low758 = icmp ugt i32 %compare.low.base740, %compare.local757, !dbg !24
  %compare.result759 = select i1 %compare.high.equal738, i1 %compare.low758, i1 %compare.high.order739, !dbg !24
  %730 = select i1 %compare.result759, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base760 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local761 = or i32 %compare.local.base760, 5, !dbg !24
  %compare.low762 = icmp ugt i32 %compare.low.base740, %compare.local761, !dbg !24
  %compare.result763 = select i1 %compare.high.equal738, i1 %compare.low762, i1 %compare.high.order739, !dbg !24
  %731 = select i1 %compare.result763, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base764 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local765 = or i32 %compare.local.base764, 6, !dbg !24
  %compare.low766 = icmp ugt i32 %compare.low.base740, %compare.local765, !dbg !24
  %compare.result767 = select i1 %compare.high.equal738, i1 %compare.low766, i1 %compare.high.order739, !dbg !24
  %732 = select i1 %compare.result767, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base768 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local769 = or i32 %compare.local.base768, 7, !dbg !24
  %compare.low770 = icmp ugt i32 %compare.low.base740, %compare.local769, !dbg !24
  %compare.result771 = select i1 %compare.high.equal738, i1 %compare.low770, i1 %compare.high.order739, !dbg !24
  %733 = select i1 %compare.result771, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base772 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local773 = or i32 %compare.local.base772, 8, !dbg !24
  %compare.low774 = icmp ugt i32 %compare.low.base740, %compare.local773, !dbg !24
  %compare.result775 = select i1 %compare.high.equal738, i1 %compare.low774, i1 %compare.high.order739, !dbg !24
  %734 = select i1 %compare.result775, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base776 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local777 = or i32 %compare.local.base776, 9, !dbg !24
  %compare.low778 = icmp ugt i32 %compare.low.base740, %compare.local777, !dbg !24
  %compare.result779 = select i1 %compare.high.equal738, i1 %compare.low778, i1 %compare.high.order739, !dbg !24
  %735 = select i1 %compare.result779, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base780 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local781 = or i32 %compare.local.base780, 10, !dbg !24
  %compare.low782 = icmp ugt i32 %compare.low.base740, %compare.local781, !dbg !24
  %compare.result783 = select i1 %compare.high.equal738, i1 %compare.low782, i1 %compare.high.order739, !dbg !24
  %736 = select i1 %compare.result783, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base784 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local785 = or i32 %compare.local.base784, 11, !dbg !24
  %compare.low786 = icmp ugt i32 %compare.low.base740, %compare.local785, !dbg !24
  %compare.result787 = select i1 %compare.high.equal738, i1 %compare.low786, i1 %compare.high.order739, !dbg !24
  %737 = select i1 %compare.result787, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base788 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local789 = or i32 %compare.local.base788, 12, !dbg !24
  %compare.low790 = icmp ugt i32 %compare.low.base740, %compare.local789, !dbg !24
  %compare.result791 = select i1 %compare.high.equal738, i1 %compare.low790, i1 %compare.high.order739, !dbg !24
  %738 = select i1 %compare.result791, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base792 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local793 = or i32 %compare.local.base792, 13, !dbg !24
  %compare.low794 = icmp ugt i32 %compare.low.base740, %compare.local793, !dbg !24
  %compare.result795 = select i1 %compare.high.equal738, i1 %compare.low794, i1 %compare.high.order739, !dbg !24
  %739 = select i1 %compare.result795, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base796 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local797 = or i32 %compare.local.base796, 14, !dbg !24
  %compare.low798 = icmp ugt i32 %compare.low.base740, %compare.local797, !dbg !24
  %compare.result799 = select i1 %compare.high.equal738, i1 %compare.low798, i1 %compare.high.order739, !dbg !24
  %740 = select i1 %compare.result799, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base800 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local801 = or i32 %compare.local.base800, 15, !dbg !24
  %compare.low802 = icmp ugt i32 %compare.low.base740, %compare.local801, !dbg !24
  %compare.result803 = select i1 %compare.high.equal738, i1 %compare.low802, i1 %compare.high.order739, !dbg !24
  %741 = select i1 %compare.result803, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base804 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local805 = or i32 %compare.local.base804, 64, !dbg !24
  %compare.low806 = icmp ugt i32 %compare.low.base740, %compare.local805, !dbg !24
  %compare.result807 = select i1 %compare.high.equal738, i1 %compare.low806, i1 %compare.high.order739, !dbg !24
  %742 = select i1 %compare.result807, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base808 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local809 = or i32 %compare.local.base808, 65, !dbg !24
  %compare.low810 = icmp ugt i32 %compare.low.base740, %compare.local809, !dbg !24
  %compare.result811 = select i1 %compare.high.equal738, i1 %compare.low810, i1 %compare.high.order739, !dbg !24
  %743 = select i1 %compare.result811, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base812 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local813 = or i32 %compare.local.base812, 66, !dbg !24
  %compare.low814 = icmp ugt i32 %compare.low.base740, %compare.local813, !dbg !24
  %compare.result815 = select i1 %compare.high.equal738, i1 %compare.low814, i1 %compare.high.order739, !dbg !24
  %744 = select i1 %compare.result815, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base816 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local817 = or i32 %compare.local.base816, 67, !dbg !24
  %compare.low818 = icmp ugt i32 %compare.low.base740, %compare.local817, !dbg !24
  %compare.result819 = select i1 %compare.high.equal738, i1 %compare.low818, i1 %compare.high.order739, !dbg !24
  %745 = select i1 %compare.result819, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base820 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local821 = or i32 %compare.local.base820, 68, !dbg !24
  %compare.low822 = icmp ugt i32 %compare.low.base740, %compare.local821, !dbg !24
  %compare.result823 = select i1 %compare.high.equal738, i1 %compare.low822, i1 %compare.high.order739, !dbg !24
  %746 = select i1 %compare.result823, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base824 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local825 = or i32 %compare.local.base824, 69, !dbg !24
  %compare.low826 = icmp ugt i32 %compare.low.base740, %compare.local825, !dbg !24
  %compare.result827 = select i1 %compare.high.equal738, i1 %compare.low826, i1 %compare.high.order739, !dbg !24
  %747 = select i1 %compare.result827, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base828 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local829 = or i32 %compare.local.base828, 70, !dbg !24
  %compare.low830 = icmp ugt i32 %compare.low.base740, %compare.local829, !dbg !24
  %compare.result831 = select i1 %compare.high.equal738, i1 %compare.low830, i1 %compare.high.order739, !dbg !24
  %748 = select i1 %compare.result831, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base832 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local833 = or i32 %compare.local.base832, 71, !dbg !24
  %compare.low834 = icmp ugt i32 %compare.low.base740, %compare.local833, !dbg !24
  %compare.result835 = select i1 %compare.high.equal738, i1 %compare.low834, i1 %compare.high.order739, !dbg !24
  %749 = select i1 %compare.result835, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base836 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local837 = or i32 %compare.local.base836, 72, !dbg !24
  %compare.low838 = icmp ugt i32 %compare.low.base740, %compare.local837, !dbg !24
  %compare.result839 = select i1 %compare.high.equal738, i1 %compare.low838, i1 %compare.high.order739, !dbg !24
  %750 = select i1 %compare.result839, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base840 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local841 = or i32 %compare.local.base840, 73, !dbg !24
  %compare.low842 = icmp ugt i32 %compare.low.base740, %compare.local841, !dbg !24
  %compare.result843 = select i1 %compare.high.equal738, i1 %compare.low842, i1 %compare.high.order739, !dbg !24
  %751 = select i1 %compare.result843, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base844 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local845 = or i32 %compare.local.base844, 74, !dbg !24
  %compare.low846 = icmp ugt i32 %compare.low.base740, %compare.local845, !dbg !24
  %compare.result847 = select i1 %compare.high.equal738, i1 %compare.low846, i1 %compare.high.order739, !dbg !24
  %752 = select i1 %compare.result847, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base848 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local849 = or i32 %compare.local.base848, 75, !dbg !24
  %compare.low850 = icmp ugt i32 %compare.low.base740, %compare.local849, !dbg !24
  %compare.result851 = select i1 %compare.high.equal738, i1 %compare.low850, i1 %compare.high.order739, !dbg !24
  %753 = select i1 %compare.result851, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base852 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local853 = or i32 %compare.local.base852, 76, !dbg !24
  %compare.low854 = icmp ugt i32 %compare.low.base740, %compare.local853, !dbg !24
  %compare.result855 = select i1 %compare.high.equal738, i1 %compare.low854, i1 %compare.high.order739, !dbg !24
  %754 = select i1 %compare.result855, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base856 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local857 = or i32 %compare.local.base856, 77, !dbg !24
  %compare.low858 = icmp ugt i32 %compare.low.base740, %compare.local857, !dbg !24
  %compare.result859 = select i1 %compare.high.equal738, i1 %compare.low858, i1 %compare.high.order739, !dbg !24
  %755 = select i1 %compare.result859, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base860 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local861 = or i32 %compare.local.base860, 78, !dbg !24
  %compare.low862 = icmp ugt i32 %compare.low.base740, %compare.local861, !dbg !24
  %compare.result863 = select i1 %compare.high.equal738, i1 %compare.low862, i1 %compare.high.order739, !dbg !24
  %756 = select i1 %compare.result863, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base864 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local865 = or i32 %compare.local.base864, 79, !dbg !24
  %compare.low866 = icmp ugt i32 %compare.low.base740, %compare.local865, !dbg !24
  %compare.result867 = select i1 %compare.high.equal738, i1 %compare.low866, i1 %compare.high.order739, !dbg !24
  %757 = select i1 %compare.result867, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %758 = insertelement <32 x float> poison, float %691, i64 0, !dbg !13
  %759 = insertelement <32 x float> %758, float %692, i64 1, !dbg !13
  %760 = insertelement <32 x float> %759, float %693, i64 2, !dbg !13
  %761 = insertelement <32 x float> %760, float %694, i64 3, !dbg !13
  %762 = insertelement <32 x float> %761, float %695, i64 4, !dbg !13
  %763 = insertelement <32 x float> %762, float %696, i64 5, !dbg !13
  %764 = insertelement <32 x float> %763, float %697, i64 6, !dbg !13
  %765 = insertelement <32 x float> %764, float %698, i64 7, !dbg !13
  %766 = insertelement <32 x float> %765, float %699, i64 8, !dbg !13
  %767 = insertelement <32 x float> %766, float %700, i64 9, !dbg !13
  %768 = insertelement <32 x float> %767, float %701, i64 10, !dbg !13
  %769 = insertelement <32 x float> %768, float %702, i64 11, !dbg !13
  %770 = insertelement <32 x float> %769, float %703, i64 12, !dbg !13
  %771 = insertelement <32 x float> %770, float %704, i64 13, !dbg !13
  %772 = insertelement <32 x float> %771, float %705, i64 14, !dbg !13
  %773 = insertelement <32 x float> %772, float %706, i64 15, !dbg !13
  %774 = insertelement <32 x float> %773, float %707, i64 16, !dbg !13
  %775 = insertelement <32 x float> %774, float %708, i64 17, !dbg !13
  %776 = insertelement <32 x float> %775, float %709, i64 18, !dbg !13
  %777 = insertelement <32 x float> %776, float %710, i64 19, !dbg !13
  %778 = insertelement <32 x float> %777, float %711, i64 20, !dbg !13
  %779 = insertelement <32 x float> %778, float %712, i64 21, !dbg !13
  %780 = insertelement <32 x float> %779, float %713, i64 22, !dbg !13
  %781 = insertelement <32 x float> %780, float %714, i64 23, !dbg !13
  %782 = insertelement <32 x float> %781, float %715, i64 24, !dbg !13
  %783 = insertelement <32 x float> %782, float %716, i64 25, !dbg !13
  %784 = insertelement <32 x float> %783, float %717, i64 26, !dbg !13
  %785 = insertelement <32 x float> %784, float %718, i64 27, !dbg !13
  %786 = insertelement <32 x float> %785, float %719, i64 28, !dbg !13
  %787 = insertelement <32 x float> %786, float %720, i64 29, !dbg !13
  %788 = insertelement <32 x float> %787, float %721, i64 30, !dbg !13
  %789 = insertelement <32 x float> %788, float %722, i64 31, !dbg !13
  %790 = insertvalue [2 x <32 x float>] poison, <32 x float> %789, 0, !dbg !13
  %791 = insertelement <32 x float> poison, float %726, i64 0, !dbg !13
  %792 = insertelement <32 x float> %791, float %727, i64 1, !dbg !13
  %793 = insertelement <32 x float> %792, float %728, i64 2, !dbg !13
  %794 = insertelement <32 x float> %793, float %729, i64 3, !dbg !13
  %795 = insertelement <32 x float> %794, float %730, i64 4, !dbg !13
  %796 = insertelement <32 x float> %795, float %731, i64 5, !dbg !13
  %797 = insertelement <32 x float> %796, float %732, i64 6, !dbg !13
  %798 = insertelement <32 x float> %797, float %733, i64 7, !dbg !13
  %799 = insertelement <32 x float> %798, float %734, i64 8, !dbg !13
  %800 = insertelement <32 x float> %799, float %735, i64 9, !dbg !13
  %801 = insertelement <32 x float> %800, float %736, i64 10, !dbg !13
  %802 = insertelement <32 x float> %801, float %737, i64 11, !dbg !13
  %803 = insertelement <32 x float> %802, float %738, i64 12, !dbg !13
  %804 = insertelement <32 x float> %803, float %739, i64 13, !dbg !13
  %805 = insertelement <32 x float> %804, float %740, i64 14, !dbg !13
  %806 = insertelement <32 x float> %805, float %741, i64 15, !dbg !13
  %807 = insertelement <32 x float> %806, float %742, i64 16, !dbg !13
  %808 = insertelement <32 x float> %807, float %743, i64 17, !dbg !13
  %809 = insertelement <32 x float> %808, float %744, i64 18, !dbg !13
  %810 = insertelement <32 x float> %809, float %745, i64 19, !dbg !13
  %811 = insertelement <32 x float> %810, float %746, i64 20, !dbg !13
  %812 = insertelement <32 x float> %811, float %747, i64 21, !dbg !13
  %813 = insertelement <32 x float> %812, float %748, i64 22, !dbg !13
  %814 = insertelement <32 x float> %813, float %749, i64 23, !dbg !13
  %815 = insertelement <32 x float> %814, float %750, i64 24, !dbg !13
  %816 = insertelement <32 x float> %815, float %751, i64 25, !dbg !13
  %817 = insertelement <32 x float> %816, float %752, i64 26, !dbg !13
  %818 = insertelement <32 x float> %817, float %753, i64 27, !dbg !13
  %819 = insertelement <32 x float> %818, float %754, i64 28, !dbg !13
  %820 = insertelement <32 x float> %819, float %755, i64 29, !dbg !13
  %821 = insertelement <32 x float> %820, float %756, i64 30, !dbg !13
  %822 = insertelement <32 x float> %821, float %757, i64 31, !dbg !13
  %823 = insertvalue [2 x <32 x float>] %790, <32 x float> %822, 1, !dbg !13
  br label %824, !dbg !13

824:                                              ; preds = %679, %680
  %825 = phi [2 x <32 x float>] [ %823, %680 ], [ zeroinitializer, %679 ], !dbg !13
  br label %826, !dbg !13

826:                                              ; preds = %824
  %827 = extractvalue [2 x <32 x float>] %825, 0, !dbg !26
  %828 = shufflevector <32 x float> %827, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %829 = fadd <4 x float> %616, %828, !dbg !26
  %830 = extractelement <4 x float> %829, i64 0, !dbg !27
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !27
  %832 = insertelement <4 x float> poison, float %831, i64 0, !dbg !29
  %833 = extractelement <4 x float> %829, i64 1, !dbg !27
  %834 = call float @llvm.amdgcn.exp2.f32(float %833), !dbg !27
  %835 = insertelement <4 x float> %832, float %834, i64 1, !dbg !29
  %836 = fadd float %831, %834, !dbg !28
  %837 = extractelement <4 x float> %829, i64 2, !dbg !27
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !27
  %839 = insertelement <4 x float> %835, float %838, i64 2, !dbg !29
  %840 = fadd float %836, %838, !dbg !28
  %841 = extractelement <4 x float> %829, i64 3, !dbg !27
  %842 = call float @llvm.amdgcn.exp2.f32(float %841), !dbg !27
  %843 = insertelement <4 x float> %839, float %842, i64 3, !dbg !29
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %843) #9, !dbg !29, !frisk.register_lifetimes !1
  %844 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %844) #9, !dbg !28, !frisk.register_lifetimes !1
  %845 = fadd float %840, %842, !dbg !28
  %846 = shufflevector <32 x float> %827, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %847 = fadd <4 x float> %620, %846, !dbg !26
  %848 = extractelement <4 x float> %847, i64 0, !dbg !27
  %849 = call float @llvm.amdgcn.exp2.f32(float %848), !dbg !27
  %850 = insertelement <4 x float> poison, float %849, i64 0, !dbg !29
  %851 = fadd float %845, %849, !dbg !28
  %852 = extractelement <4 x float> %847, i64 1, !dbg !27
  %853 = call float @llvm.amdgcn.exp2.f32(float %852), !dbg !27
  %854 = insertelement <4 x float> %850, float %853, i64 1, !dbg !29
  %855 = fadd float %851, %853, !dbg !28
  %856 = extractelement <4 x float> %847, i64 2, !dbg !27
  %857 = call float @llvm.amdgcn.exp2.f32(float %856), !dbg !27
  %858 = insertelement <4 x float> %854, float %857, i64 2, !dbg !29
  %859 = fadd float %855, %857, !dbg !28
  %860 = extractelement <4 x float> %847, i64 3, !dbg !27
  %861 = call float @llvm.amdgcn.exp2.f32(float %860), !dbg !27
  %862 = insertelement <4 x float> %858, float %861, i64 3, !dbg !29
  %math.result99 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %862) #9, !dbg !29, !frisk.register_lifetimes !1
  %863 = fptrunc <4 x float> %math.result99 to <4 x half>, !dbg !29
  %math.packed100 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %863) #9, !dbg !28, !frisk.register_lifetimes !1
  %864 = fadd float %859, %861, !dbg !28
  %865 = shufflevector <32 x float> %827, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %866 = fadd <4 x float> %624, %865, !dbg !26
  %867 = extractelement <4 x float> %866, i64 0, !dbg !27
  %868 = call float @llvm.amdgcn.exp2.f32(float %867), !dbg !27
  %869 = insertelement <4 x float> poison, float %868, i64 0, !dbg !29
  %870 = fadd float %864, %868, !dbg !28
  %871 = extractelement <4 x float> %866, i64 1, !dbg !27
  %872 = call float @llvm.amdgcn.exp2.f32(float %871), !dbg !27
  %873 = insertelement <4 x float> %869, float %872, i64 1, !dbg !29
  %874 = fadd float %870, %872, !dbg !28
  %875 = extractelement <4 x float> %866, i64 2, !dbg !27
  %876 = call float @llvm.amdgcn.exp2.f32(float %875), !dbg !27
  %877 = insertelement <4 x float> %873, float %876, i64 2, !dbg !29
  %878 = fadd float %874, %876, !dbg !28
  %879 = extractelement <4 x float> %866, i64 3, !dbg !27
  %880 = call float @llvm.amdgcn.exp2.f32(float %879), !dbg !27
  %881 = insertelement <4 x float> %877, float %880, i64 3, !dbg !29
  %math.result101 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %881) #9, !dbg !29, !frisk.register_lifetimes !1
  %882 = fptrunc <4 x float> %math.result101 to <4 x half>, !dbg !29
  %math.packed102 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %882) #9, !dbg !28, !frisk.register_lifetimes !1
  %883 = fadd float %878, %880, !dbg !28
  %884 = shufflevector <32 x float> %827, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %885 = fadd <4 x float> %628, %884, !dbg !26
  %886 = extractelement <4 x float> %885, i64 0, !dbg !27
  %887 = call float @llvm.amdgcn.exp2.f32(float %886), !dbg !27
  %888 = insertelement <4 x float> poison, float %887, i64 0, !dbg !29
  %889 = fadd float %883, %887, !dbg !28
  %890 = extractelement <4 x float> %885, i64 1, !dbg !27
  %891 = call float @llvm.amdgcn.exp2.f32(float %890), !dbg !27
  %892 = insertelement <4 x float> %888, float %891, i64 1, !dbg !29
  %893 = fadd float %889, %891, !dbg !28
  %894 = extractelement <4 x float> %885, i64 2, !dbg !27
  %895 = call float @llvm.amdgcn.exp2.f32(float %894), !dbg !27
  %896 = insertelement <4 x float> %892, float %895, i64 2, !dbg !29
  %897 = fadd float %893, %895, !dbg !28
  %898 = extractelement <4 x float> %885, i64 3, !dbg !27
  %899 = call float @llvm.amdgcn.exp2.f32(float %898), !dbg !27
  %900 = insertelement <4 x float> %896, float %899, i64 3, !dbg !29
  %math.result103 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %900) #9, !dbg !29, !frisk.register_lifetimes !1
  %901 = fptrunc <4 x float> %math.result103 to <4 x half>, !dbg !29
  %math.packed104 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %901) #9, !dbg !28, !frisk.register_lifetimes !1
  %902 = fadd float %897, %899, !dbg !28
  %903 = shufflevector <32 x float> %827, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %904 = fadd <4 x float> %632, %903, !dbg !26
  %905 = extractelement <4 x float> %904, i64 0, !dbg !27
  %906 = call float @llvm.amdgcn.exp2.f32(float %905), !dbg !27
  %907 = insertelement <4 x float> poison, float %906, i64 0, !dbg !29
  %908 = fadd float %902, %906, !dbg !28
  %909 = extractelement <4 x float> %904, i64 1, !dbg !27
  %910 = call float @llvm.amdgcn.exp2.f32(float %909), !dbg !27
  %911 = insertelement <4 x float> %907, float %910, i64 1, !dbg !29
  %912 = fadd float %908, %910, !dbg !28
  %913 = extractelement <4 x float> %904, i64 2, !dbg !27
  %914 = call float @llvm.amdgcn.exp2.f32(float %913), !dbg !27
  %915 = insertelement <4 x float> %911, float %914, i64 2, !dbg !29
  %916 = fadd float %912, %914, !dbg !28
  %917 = extractelement <4 x float> %904, i64 3, !dbg !27
  %918 = call float @llvm.amdgcn.exp2.f32(float %917), !dbg !27
  %919 = insertelement <4 x float> %915, float %918, i64 3, !dbg !29
  %math.result105 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %919) #9, !dbg !29, !frisk.register_lifetimes !1
  %920 = fptrunc <4 x float> %math.result105 to <4 x half>, !dbg !29
  %math.packed106 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %920) #9, !dbg !28, !frisk.register_lifetimes !1
  %921 = fadd float %916, %918, !dbg !28
  %922 = shufflevector <32 x float> %827, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %923 = fadd <4 x float> %636, %922, !dbg !26
  %924 = extractelement <4 x float> %923, i64 0, !dbg !27
  %925 = call float @llvm.amdgcn.exp2.f32(float %924), !dbg !27
  %926 = insertelement <4 x float> poison, float %925, i64 0, !dbg !29
  %927 = fadd float %921, %925, !dbg !28
  %928 = extractelement <4 x float> %923, i64 1, !dbg !27
  %929 = call float @llvm.amdgcn.exp2.f32(float %928), !dbg !27
  %930 = insertelement <4 x float> %926, float %929, i64 1, !dbg !29
  %931 = fadd float %927, %929, !dbg !28
  %932 = extractelement <4 x float> %923, i64 2, !dbg !27
  %933 = call float @llvm.amdgcn.exp2.f32(float %932), !dbg !27
  %934 = insertelement <4 x float> %930, float %933, i64 2, !dbg !29
  %935 = fadd float %931, %933, !dbg !28
  %936 = extractelement <4 x float> %923, i64 3, !dbg !27
  %937 = call float @llvm.amdgcn.exp2.f32(float %936), !dbg !27
  %938 = insertelement <4 x float> %934, float %937, i64 3, !dbg !29
  %math.result107 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %938) #9, !dbg !29, !frisk.register_lifetimes !1
  %939 = fptrunc <4 x float> %math.result107 to <4 x half>, !dbg !29
  %math.packed108 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %939) #9, !dbg !28, !frisk.register_lifetimes !1
  %940 = fadd float %935, %937, !dbg !28
  %941 = shufflevector <32 x float> %827, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %942 = fadd <4 x float> %640, %941, !dbg !26
  %943 = extractelement <4 x float> %942, i64 0, !dbg !27
  %944 = call float @llvm.amdgcn.exp2.f32(float %943), !dbg !27
  %945 = insertelement <4 x float> poison, float %944, i64 0, !dbg !29
  %946 = fadd float %940, %944, !dbg !28
  %947 = extractelement <4 x float> %942, i64 1, !dbg !27
  %948 = call float @llvm.amdgcn.exp2.f32(float %947), !dbg !27
  %949 = insertelement <4 x float> %945, float %948, i64 1, !dbg !29
  %950 = fadd float %946, %948, !dbg !28
  %951 = extractelement <4 x float> %942, i64 2, !dbg !27
  %952 = call float @llvm.amdgcn.exp2.f32(float %951), !dbg !27
  %953 = insertelement <4 x float> %949, float %952, i64 2, !dbg !29
  %954 = fadd float %950, %952, !dbg !28
  %955 = extractelement <4 x float> %942, i64 3, !dbg !27
  %956 = call float @llvm.amdgcn.exp2.f32(float %955), !dbg !27
  %957 = insertelement <4 x float> %953, float %956, i64 3, !dbg !29
  %math.result109 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %957) #9, !dbg !29, !frisk.register_lifetimes !1
  %958 = fptrunc <4 x float> %math.result109 to <4 x half>, !dbg !29
  %math.packed110 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %958) #9, !dbg !28, !frisk.register_lifetimes !1
  %959 = fadd float %954, %956, !dbg !28
  %960 = shufflevector <32 x float> %827, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %961 = fadd <4 x float> %644, %960, !dbg !26
  %962 = extractelement <4 x float> %961, i64 0, !dbg !27
  %963 = call float @llvm.amdgcn.exp2.f32(float %962), !dbg !27
  %964 = insertelement <4 x float> poison, float %963, i64 0, !dbg !29
  %965 = fadd float %959, %963, !dbg !28
  %966 = extractelement <4 x float> %961, i64 1, !dbg !27
  %967 = call float @llvm.amdgcn.exp2.f32(float %966), !dbg !27
  %968 = insertelement <4 x float> %964, float %967, i64 1, !dbg !29
  %969 = fadd float %965, %967, !dbg !28
  %970 = extractelement <4 x float> %961, i64 2, !dbg !27
  %971 = call float @llvm.amdgcn.exp2.f32(float %970), !dbg !27
  %972 = insertelement <4 x float> %968, float %971, i64 2, !dbg !29
  %973 = fadd float %969, %971, !dbg !28
  %974 = extractelement <4 x float> %961, i64 3, !dbg !27
  %975 = call float @llvm.amdgcn.exp2.f32(float %974), !dbg !27
  %976 = insertelement <4 x float> %972, float %975, i64 3, !dbg !29
  %math.result111 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %976) #9, !dbg !29, !frisk.register_lifetimes !1
  %977 = fptrunc <4 x float> %math.result111 to <4 x half>, !dbg !29
  %math.packed112 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %977) #9, !dbg !28, !frisk.register_lifetimes !1
  %978 = fadd float %973, %975, !dbg !28
  %979 = extractvalue [2 x <32 x float>] %825, 1, !dbg !26
  %980 = shufflevector <32 x float> %979, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %981 = fadd <4 x float> %648, %980, !dbg !26
  %982 = extractelement <4 x float> %981, i64 0, !dbg !27
  %983 = call float @llvm.amdgcn.exp2.f32(float %982), !dbg !27
  %984 = insertelement <4 x float> poison, float %983, i64 0, !dbg !29
  %985 = extractelement <4 x float> %981, i64 1, !dbg !27
  %986 = call float @llvm.amdgcn.exp2.f32(float %985), !dbg !27
  %987 = insertelement <4 x float> %984, float %986, i64 1, !dbg !29
  %988 = fadd float %983, %986, !dbg !28
  %989 = extractelement <4 x float> %981, i64 2, !dbg !27
  %990 = call float @llvm.amdgcn.exp2.f32(float %989), !dbg !27
  %991 = insertelement <4 x float> %987, float %990, i64 2, !dbg !29
  %992 = fadd float %988, %990, !dbg !28
  %993 = extractelement <4 x float> %981, i64 3, !dbg !27
  %994 = call float @llvm.amdgcn.exp2.f32(float %993), !dbg !27
  %995 = insertelement <4 x float> %991, float %994, i64 3, !dbg !29
  %math.result113 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %995) #9, !dbg !29, !frisk.register_lifetimes !1
  %996 = fptrunc <4 x float> %math.result113 to <4 x half>, !dbg !29
  %math.packed114 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %996) #9, !dbg !28, !frisk.register_lifetimes !1
  %997 = fadd float %992, %994, !dbg !28
  %998 = shufflevector <32 x float> %979, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %999 = fadd <4 x float> %652, %998, !dbg !26
  %1000 = extractelement <4 x float> %999, i64 0, !dbg !27
  %1001 = call float @llvm.amdgcn.exp2.f32(float %1000), !dbg !27
  %1002 = insertelement <4 x float> poison, float %1001, i64 0, !dbg !29
  %1003 = fadd float %997, %1001, !dbg !28
  %1004 = extractelement <4 x float> %999, i64 1, !dbg !27
  %1005 = call float @llvm.amdgcn.exp2.f32(float %1004), !dbg !27
  %1006 = insertelement <4 x float> %1002, float %1005, i64 1, !dbg !29
  %1007 = fadd float %1003, %1005, !dbg !28
  %1008 = extractelement <4 x float> %999, i64 2, !dbg !27
  %1009 = call float @llvm.amdgcn.exp2.f32(float %1008), !dbg !27
  %1010 = insertelement <4 x float> %1006, float %1009, i64 2, !dbg !29
  %1011 = fadd float %1007, %1009, !dbg !28
  %1012 = extractelement <4 x float> %999, i64 3, !dbg !27
  %1013 = call float @llvm.amdgcn.exp2.f32(float %1012), !dbg !27
  %1014 = insertelement <4 x float> %1010, float %1013, i64 3, !dbg !29
  %math.result115 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1014) #9, !dbg !29, !frisk.register_lifetimes !1
  %1015 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !29
  %math.packed116 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1015) #9, !dbg !28, !frisk.register_lifetimes !1
  %1016 = fadd float %1011, %1013, !dbg !28
  %1017 = shufflevector <32 x float> %979, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %1018 = fadd <4 x float> %656, %1017, !dbg !26
  %1019 = extractelement <4 x float> %1018, i64 0, !dbg !27
  %1020 = call float @llvm.amdgcn.exp2.f32(float %1019), !dbg !27
  %1021 = insertelement <4 x float> poison, float %1020, i64 0, !dbg !29
  %1022 = fadd float %1016, %1020, !dbg !28
  %1023 = extractelement <4 x float> %1018, i64 1, !dbg !27
  %1024 = call float @llvm.amdgcn.exp2.f32(float %1023), !dbg !27
  %1025 = insertelement <4 x float> %1021, float %1024, i64 1, !dbg !29
  %1026 = fadd float %1022, %1024, !dbg !28
  %1027 = extractelement <4 x float> %1018, i64 2, !dbg !27
  %1028 = call float @llvm.amdgcn.exp2.f32(float %1027), !dbg !27
  %1029 = insertelement <4 x float> %1025, float %1028, i64 2, !dbg !29
  %1030 = fadd float %1026, %1028, !dbg !28
  %1031 = extractelement <4 x float> %1018, i64 3, !dbg !27
  %1032 = call float @llvm.amdgcn.exp2.f32(float %1031), !dbg !27
  %1033 = insertelement <4 x float> %1029, float %1032, i64 3, !dbg !29
  %math.result117 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1033) #9, !dbg !29, !frisk.register_lifetimes !1
  %1034 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !29
  %math.packed118 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1034) #9, !dbg !28, !frisk.register_lifetimes !1
  %1035 = fadd float %1030, %1032, !dbg !28
  %1036 = shufflevector <32 x float> %979, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %1037 = fadd <4 x float> %660, %1036, !dbg !26
  %1038 = extractelement <4 x float> %1037, i64 0, !dbg !27
  %1039 = call float @llvm.amdgcn.exp2.f32(float %1038), !dbg !27
  %1040 = insertelement <4 x float> poison, float %1039, i64 0, !dbg !29
  %1041 = fadd float %1035, %1039, !dbg !28
  %1042 = extractelement <4 x float> %1037, i64 1, !dbg !27
  %1043 = call float @llvm.amdgcn.exp2.f32(float %1042), !dbg !27
  %1044 = insertelement <4 x float> %1040, float %1043, i64 1, !dbg !29
  %1045 = fadd float %1041, %1043, !dbg !28
  %1046 = extractelement <4 x float> %1037, i64 2, !dbg !27
  %1047 = call float @llvm.amdgcn.exp2.f32(float %1046), !dbg !27
  %1048 = insertelement <4 x float> %1044, float %1047, i64 2, !dbg !29
  %1049 = fadd float %1045, %1047, !dbg !28
  %1050 = extractelement <4 x float> %1037, i64 3, !dbg !27
  %1051 = call float @llvm.amdgcn.exp2.f32(float %1050), !dbg !27
  %1052 = insertelement <4 x float> %1048, float %1051, i64 3, !dbg !29
  %math.result119 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1052) #9, !dbg !29, !frisk.register_lifetimes !1
  %1053 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !29
  %math.packed120 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1053) #9, !dbg !28, !frisk.register_lifetimes !1
  %1054 = fadd float %1049, %1051, !dbg !28
  %1055 = shufflevector <32 x float> %979, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %1056 = fadd <4 x float> %664, %1055, !dbg !26
  %1057 = extractelement <4 x float> %1056, i64 0, !dbg !27
  %1058 = call float @llvm.amdgcn.exp2.f32(float %1057), !dbg !27
  %1059 = insertelement <4 x float> poison, float %1058, i64 0, !dbg !29
  %1060 = fadd float %1054, %1058, !dbg !28
  %1061 = extractelement <4 x float> %1056, i64 1, !dbg !27
  %1062 = call float @llvm.amdgcn.exp2.f32(float %1061), !dbg !27
  %1063 = insertelement <4 x float> %1059, float %1062, i64 1, !dbg !29
  %1064 = fadd float %1060, %1062, !dbg !28
  %1065 = extractelement <4 x float> %1056, i64 2, !dbg !27
  %1066 = call float @llvm.amdgcn.exp2.f32(float %1065), !dbg !27
  %1067 = insertelement <4 x float> %1063, float %1066, i64 2, !dbg !29
  %1068 = fadd float %1064, %1066, !dbg !28
  %1069 = extractelement <4 x float> %1056, i64 3, !dbg !27
  %1070 = call float @llvm.amdgcn.exp2.f32(float %1069), !dbg !27
  %1071 = insertelement <4 x float> %1067, float %1070, i64 3, !dbg !29
  %math.result121 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1071) #9, !dbg !29, !frisk.register_lifetimes !1
  %1072 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !29
  %math.packed122 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1072) #9, !dbg !28, !frisk.register_lifetimes !1
  %1073 = fadd float %1068, %1070, !dbg !28
  %1074 = shufflevector <32 x float> %979, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %1075 = fadd <4 x float> %668, %1074, !dbg !26
  %1076 = extractelement <4 x float> %1075, i64 0, !dbg !27
  %1077 = call float @llvm.amdgcn.exp2.f32(float %1076), !dbg !27
  %1078 = insertelement <4 x float> poison, float %1077, i64 0, !dbg !29
  %1079 = fadd float %1073, %1077, !dbg !28
  %1080 = extractelement <4 x float> %1075, i64 1, !dbg !27
  %1081 = call float @llvm.amdgcn.exp2.f32(float %1080), !dbg !27
  %1082 = insertelement <4 x float> %1078, float %1081, i64 1, !dbg !29
  %1083 = fadd float %1079, %1081, !dbg !28
  %1084 = extractelement <4 x float> %1075, i64 2, !dbg !27
  %1085 = call float @llvm.amdgcn.exp2.f32(float %1084), !dbg !27
  %1086 = insertelement <4 x float> %1082, float %1085, i64 2, !dbg !29
  %1087 = fadd float %1083, %1085, !dbg !28
  %1088 = extractelement <4 x float> %1075, i64 3, !dbg !27
  %1089 = call float @llvm.amdgcn.exp2.f32(float %1088), !dbg !27
  %1090 = insertelement <4 x float> %1086, float %1089, i64 3, !dbg !29
  %math.result123 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1090) #9, !dbg !29, !frisk.register_lifetimes !1
  %1091 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !29
  %math.packed124 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1091) #9, !dbg !28, !frisk.register_lifetimes !1
  %1092 = fadd float %1087, %1089, !dbg !28
  %1093 = shufflevector <32 x float> %979, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %1094 = fadd <4 x float> %672, %1093, !dbg !26
  %1095 = extractelement <4 x float> %1094, i64 0, !dbg !27
  %1096 = call float @llvm.amdgcn.exp2.f32(float %1095), !dbg !27
  %1097 = insertelement <4 x float> poison, float %1096, i64 0, !dbg !29
  %1098 = fadd float %1092, %1096, !dbg !28
  %1099 = extractelement <4 x float> %1094, i64 1, !dbg !27
  %1100 = call float @llvm.amdgcn.exp2.f32(float %1099), !dbg !27
  %1101 = insertelement <4 x float> %1097, float %1100, i64 1, !dbg !29
  %1102 = fadd float %1098, %1100, !dbg !28
  %1103 = extractelement <4 x float> %1094, i64 2, !dbg !27
  %1104 = call float @llvm.amdgcn.exp2.f32(float %1103), !dbg !27
  %1105 = insertelement <4 x float> %1101, float %1104, i64 2, !dbg !29
  %1106 = fadd float %1102, %1104, !dbg !28
  %1107 = extractelement <4 x float> %1094, i64 3, !dbg !27
  %1108 = call float @llvm.amdgcn.exp2.f32(float %1107), !dbg !27
  %1109 = insertelement <4 x float> %1105, float %1108, i64 3, !dbg !29
  %math.result125 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1109) #9, !dbg !29, !frisk.register_lifetimes !1
  %1110 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !29
  %math.packed126 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1110) #9, !dbg !28, !frisk.register_lifetimes !1
  %1111 = fadd float %1106, %1108, !dbg !28
  %1112 = shufflevector <32 x float> %979, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %1113 = fadd <4 x float> %677, %1112, !dbg !26
  %1114 = extractelement <4 x float> %1113, i64 0, !dbg !27
  %1115 = call float @llvm.amdgcn.exp2.f32(float %1114), !dbg !27
  %1116 = insertelement <4 x float> poison, float %1115, i64 0, !dbg !29
  %1117 = fadd float %1111, %1115, !dbg !28
  %1118 = extractelement <4 x float> %1113, i64 1, !dbg !27
  %1119 = call float @llvm.amdgcn.exp2.f32(float %1118), !dbg !27
  %1120 = insertelement <4 x float> %1116, float %1119, i64 1, !dbg !29
  %1121 = fadd float %1117, %1119, !dbg !28
  %1122 = extractelement <4 x float> %1113, i64 2, !dbg !27
  %1123 = call float @llvm.amdgcn.exp2.f32(float %1122), !dbg !27
  %1124 = insertelement <4 x float> %1120, float %1123, i64 2, !dbg !29
  %1125 = fadd float %1121, %1123, !dbg !28
  %1126 = extractelement <4 x float> %1113, i64 3, !dbg !27
  %1127 = call float @llvm.amdgcn.exp2.f32(float %1126), !dbg !27
  %1128 = insertelement <4 x float> %1124, float %1127, i64 3, !dbg !29
  %math.result127 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1128) #9, !dbg !29, !frisk.register_lifetimes !1
  %1129 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !29
  %math.packed128 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1129) #9, !dbg !28, !frisk.register_lifetimes !1
  %1130 = fadd float %1125, %1127, !dbg !28
  %1131 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %1132 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1131), !dbg !28
  %1133 = add nuw nsw i32 %1132, 64, !dbg !28
  %1134 = xor i32 %1132, 16, !dbg !28
  %1135 = and i32 %1133, 192, !dbg !28
  %1136 = icmp samesign ult i32 %1134, %1135, !dbg !28
  %1137 = select i1 %1136, i32 %1134, i32 %1132, !dbg !28
  %1138 = shl nuw nsw i32 %1137, 2, !dbg !28
  %1139 = bitcast float %978 to i32, !dbg !28
  %1140 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1139), !dbg !28
  %1141 = bitcast i32 %1140 to float, !dbg !28
  %1142 = fadd float %978, %1141, !dbg !28
  %1143 = xor i32 %1132, 32, !dbg !28
  %1144 = icmp samesign ult i32 %1143, %1135, !dbg !28
  %1145 = select i1 %1144, i32 %1143, i32 %1132, !dbg !28
  %1146 = shl nuw nsw i32 %1145, 2, !dbg !28
  %1147 = bitcast float %1142 to i32, !dbg !28
  %1148 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1147), !dbg !28
  %1149 = bitcast i32 %1148 to float, !dbg !28
  %1150 = fadd float %1142, %1149, !dbg !28
  %1151 = extractvalue [2 x <1 x float>] %43, 0, !dbg !28
  %1152 = extractelement <1 x float> %1151, i64 0, !dbg !28
  %1153 = fadd float %1150, %1152, !dbg !28
  %1154 = insertelement <1 x float> poison, float %1153, i64 0, !dbg !15
  %1155 = insertvalue [2 x <1 x float>] poison, <1 x float> %1154, 0, !dbg !15
  %1156 = bitcast float %1130 to i32, !dbg !28
  %1157 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1138, i32 %1156), !dbg !28
  %1158 = bitcast i32 %1157 to float, !dbg !28
  %1159 = fadd float %1130, %1158, !dbg !28
  %1160 = bitcast float %1159 to i32, !dbg !28
  %1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1160), !dbg !28
  %1162 = bitcast i32 %1161 to float, !dbg !28
  %1163 = fadd float %1159, %1162, !dbg !28
  %1164 = extractvalue [2 x <1 x float>] %43, 1, !dbg !28
  %1165 = extractelement <1 x float> %1164, i64 0, !dbg !28
  %1166 = fadd float %1163, %1165, !dbg !28
  %1167 = insertelement <1 x float> poison, float %1166, i64 0, !dbg !15
  %1168 = insertvalue [2 x <1 x float>] %1155, <1 x float> %1167, 1, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread319 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !14, !range !33, !frisk.register_lifetimes !1
  %address.local320 = shl nuw nsw i32 %address.thread319, 3, !dbg !14
  %address.local321 = and i32 %address.local320, 120, !dbg !14
  %address.local322 = shl nuw nsw i32 %address.thread319, 5, !dbg !14
  %address.local323 = and i32 %address.local322, 1536, !dbg !14
  %address.local324 = or disjoint i32 %address.local321, %address.local323, !dbg !14
  %address.local325 = zext nneg i32 %address.local324 to i64, !dbg !14
  %address.local326 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local325, !dbg !14
  %1169 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %1170 = bitcast <4 x i32> %1169 to <8 x half>, !dbg !14
  %address.local327 = zext nneg i32 %address.thread319 to i64, !dbg !12
  %address.local328 = and i64 %address.local327, 60, !dbg !14
  %address.local329 = add nuw nsw i64 %address.local328, %address.local327, !dbg !14
  %address.local330 = shl nuw nsw i64 %address.local329, 3, !dbg !14
  %address.local331 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local332 = and i64 %address.local327, 63, !dbg !12
  %address.local333 = lshr i64 %address.local332, 4, !dbg !14
  %address.local334 = mul nuw nsw i64 %address.local333, 384, !dbg !14
  %address.local335 = getelementptr [2 x i8], ptr addrspace(3) %address.local331, i64 %address.local334, !dbg !14
  %address.local336 = lshr i64 %address.local327, 2, !dbg !17
  %address.local337 = mul nsw i64 %address.local336, -32, !dbg !14
  %address.local338 = getelementptr [2 x i8], ptr addrspace(3) %address.local335, i64 %address.local337, !dbg !14
  %address.local339 = getelementptr i8, ptr addrspace(3) %address.local338, i64 256, !dbg !14
  %1171 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %1172 = bitcast <4 x i32> %1171 to <8 x half>, !dbg !14
  %address.local340 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) %address.local340, i64 %address.local334, !dbg !14
  %address.local342 = getelementptr [2 x i8], ptr addrspace(3) %address.local341, i64 %address.local337, !dbg !14
  %address.local343 = getelementptr i8, ptr addrspace(3) %address.local342, i64 512, !dbg !14
  %1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %1174 = bitcast <4 x i32> %1173 to <8 x half>, !dbg !14
  %address.local344 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) %address.local344, i64 %address.local334, !dbg !14
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) %address.local345, i64 %address.local337, !dbg !14
  %address.local347 = getelementptr i8, ptr addrspace(3) %address.local346, i64 768, !dbg !14
  %1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %1176 = bitcast <4 x i32> %1175 to <8 x half>, !dbg !14
  %address.local348 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local325, !dbg !14
  %address.local349 = getelementptr i8, ptr addrspace(3) %address.local348, i64 4096, !dbg !14
  %1177 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %1178 = bitcast <4 x i32> %1177 to <8 x half>, !dbg !14
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local351 = getelementptr [2 x i8], ptr addrspace(3) %address.local350, i64 %address.local334, !dbg !14
  %address.local352 = getelementptr [2 x i8], ptr addrspace(3) %address.local351, i64 %address.local337, !dbg !14
  %address.local353 = getelementptr i8, ptr addrspace(3) %address.local352, i64 4352, !dbg !14
  %1179 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %1180 = bitcast <4 x i32> %1179 to <8 x half>, !dbg !14
  %address.local354 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local355 = getelementptr [2 x i8], ptr addrspace(3) %address.local354, i64 %address.local334, !dbg !14
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) %address.local355, i64 %address.local337, !dbg !14
  %address.local357 = getelementptr i8, ptr addrspace(3) %address.local356, i64 4608, !dbg !14
  %1181 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %1182 = bitcast <4 x i32> %1181 to <8 x half>, !dbg !14
  %address.local358 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) %address.local358, i64 %address.local334, !dbg !14
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) %address.local359, i64 %address.local337, !dbg !14
  %address.local361 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4864, !dbg !14
  %1183 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1184 = bitcast <4 x i32> %1183 to <8 x half>, !dbg !14
  %1185 = add nsw i64 %79, %10, !dbg !18
  %1186 = shl i64 %1185, 19, !dbg !18
  %1187 = and i64 %1186, 16252928, !dbg !18
  %1188 = add i64 %1187, %85, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1189 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1188, !dbg !18
  %1190 = getelementptr i8, ptr addrspace(1) %1189, i64 64, !dbg !18
  %address.local362 = shl nuw nsw i64 %address.local327, 3, !dbg !11
  %address.local363 = and i64 %address.local362, 1536, !dbg !11
  %address.local364 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local363, !dbg !11
  %address.local365 = ptrtoint ptr addrspace(3) %address.local364 to i64, !dbg !11
  %address.local366 = trunc i64 %address.local365 to i32, !dbg !11
  %1191 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local366), !dbg !18
  %1192 = zext i32 %1191 to i64, !dbg !18
  %1193 = inttoptr i64 %1192 to ptr addrspace(3), !dbg !18
  %address.local367 = shl nuw nsw i64 %address.local327, 5, !dbg !18
  %address.local368 = and i64 %address.local367, 8064, !dbg !18
  %address.local369 = shl nuw nsw i64 %address.local327, 3, !dbg !18
  %address.local370 = and i64 %address.local369, 24, !dbg !18
  %address.local371 = or disjoint i64 %address.local368, %address.local370, !dbg !18
  %address.local372 = getelementptr [2 x i8], ptr addrspace(1) %1190, i64 %address.local371, !dbg !18
  %1194 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1195 = ptrtoint ptr addrspace(1) %address.local372 to i64, !dbg !18
  %1196 = sub i64 %1195, %1194, !dbg !18
  %dma.buffer.offset880 = trunc i64 %1196 to i32, !dbg !18
  %1197 = trunc i64 %1194 to i32, !dbg !18
  %1198 = insertelement <4 x i32> poison, i32 %1197, i64 0, !dbg !18
  %1199 = lshr i64 %1194, 32, !dbg !18
  %1200 = trunc i64 %1199 to i32, !dbg !18
  %1201 = insertelement <4 x i32> %1198, i32 %1200, i64 1, !dbg !18
  %1202 = insertelement <4 x i32> %1201, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc881 = insertelement <4 x i32> %1202, i32 131072, i64 3, !dbg !18
  %1203 = ptrtoint ptr addrspace(3) %1193 to i32, !dbg !18
  %dma.buffer.lds882 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1203), !dbg !18
  %1204 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds882, i32 %dma.buffer.offset880, <4 x i32> %dma.buffer.rsrc881) #10, !dbg !18
  %address.local373 = or disjoint i64 %address.local327, 256, !dbg !11
  %address.local374 = shl nuw nsw i64 %address.local373, 3, !dbg !11
  %address.local375 = and i64 %address.local374, 3584, !dbg !11
  %address.local376 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local375, !dbg !11
  %address.local377 = ptrtoint ptr addrspace(3) %address.local376 to i64, !dbg !11
  %address.local378 = trunc i64 %address.local377 to i32, !dbg !11
  %1205 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local378), !dbg !18
  %1206 = zext i32 %1205 to i64, !dbg !18
  %1207 = inttoptr i64 %1206 to ptr addrspace(3), !dbg !18
  %address.local379 = shl nuw nsw i64 %address.local373, 5, !dbg !18
  %address.local380 = and i64 %address.local379, 16256, !dbg !18
  %address.local381 = shl nuw nsw i64 %address.local327, 3, !dbg !18
  %address.local382 = and i64 %address.local381, 24, !dbg !18
  %address.local383 = or disjoint i64 %address.local380, %address.local382, !dbg !18
  %address.local384 = getelementptr [2 x i8], ptr addrspace(1) %1190, i64 %address.local383, !dbg !18
  %1208 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1209 = ptrtoint ptr addrspace(1) %address.local384 to i64, !dbg !18
  %1210 = sub i64 %1209, %1208, !dbg !18
  %dma.buffer.offset877 = trunc i64 %1210 to i32, !dbg !18
  %1211 = trunc i64 %1208 to i32, !dbg !18
  %1212 = insertelement <4 x i32> poison, i32 %1211, i64 0, !dbg !18
  %1213 = lshr i64 %1208, 32, !dbg !18
  %1214 = trunc i64 %1213 to i32, !dbg !18
  %1215 = insertelement <4 x i32> %1212, i32 %1214, i64 1, !dbg !18
  %1216 = insertelement <4 x i32> %1215, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc878 = insertelement <4 x i32> %1216, i32 131072, i64 3, !dbg !18
  %1217 = ptrtoint ptr addrspace(3) %1207 to i32, !dbg !18
  %dma.buffer.lds879 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1217), !dbg !18
  %1218 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds879, i32 %dma.buffer.offset877, <4 x i32> %dma.buffer.rsrc878) #10, !dbg !18
  %1219 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1220 = extractvalue [2 x <8 x float>] %42, 0, !dbg !30
  %1221 = shufflevector <8 x float> %1220, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1219, <4 x float> %1221) #9, !dbg !14
  %1223 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1223, <4 x float> %1222) #9, !dbg !14
  %1225 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1225, <4 x float> %1224) #9, !dbg !14
  %1227 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1227, <4 x float> %1226) #9, !dbg !14
  %1229 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1229, <4 x float> %1228) #9, !dbg !14
  %1231 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1231, <4 x float> %1230) #9, !dbg !14
  %1233 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1233, <4 x float> %1232) #9, !dbg !14
  %1235 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1235, <4 x float> %1234) #9, !dbg !14
  %1237 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1238 = shufflevector <8 x float> %1220, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1237, <4 x float> %1238) #9, !dbg !14
  %1240 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1240, <4 x float> %1239) #9, !dbg !14
  %1242 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1242, <4 x float> %1241) #9, !dbg !14
  %1244 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1244, <4 x float> %1243) #9, !dbg !14
  %1246 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1246, <4 x float> %1245) #9, !dbg !14
  %1248 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1248, <4 x float> %1247) #9, !dbg !14
  %1250 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1250, <4 x float> %1249) #9, !dbg !14
  %1252 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1252, <4 x float> %1251) #9, !dbg !14
  %1254 = extractvalue [2 x <8 x float>] %42, 1, !dbg !30
  %1255 = shufflevector <8 x float> %1254, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1219, <4 x float> %1255) #9, !dbg !14
  %1257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1223, <4 x float> %1256) #9, !dbg !14
  %1258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1225, <4 x float> %1257) #9, !dbg !14
  %1259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1227, <4 x float> %1258) #9, !dbg !14
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1229, <4 x float> %1259) #9, !dbg !14
  %1261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1231, <4 x float> %1260) #9, !dbg !14
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1233, <4 x float> %1261) #9, !dbg !14
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1235, <4 x float> %1262) #9, !dbg !14
  %1264 = shufflevector <8 x float> %1254, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1237, <4 x float> %1264) #9, !dbg !14
  %1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1240, <4 x float> %1265) #9, !dbg !14
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1242, <4 x float> %1266) #9, !dbg !14
  %1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1244, <4 x float> %1267) #9, !dbg !14
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1246, <4 x float> %1268) #9, !dbg !14
  %1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1248, <4 x float> %1269) #9, !dbg !14
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1250, <4 x float> %1270) #9, !dbg !14
  %1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1252, <4 x float> %1271) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1273 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %1274 = bitcast <4 x i32> %1273 to <8 x half>, !dbg !14
  %1275 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %1276 = bitcast <4 x i32> %1275 to <8 x half>, !dbg !14
  %1277 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %1278 = bitcast <4 x i32> %1277 to <8 x half>, !dbg !14
  %1279 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %1280 = bitcast <4 x i32> %1279 to <8 x half>, !dbg !14
  %1281 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %1282 = bitcast <4 x i32> %1281 to <8 x half>, !dbg !14
  %1283 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %1284 = bitcast <4 x i32> %1283 to <8 x half>, !dbg !14
  %1285 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %1286 = bitcast <4 x i32> %1285 to <8 x half>, !dbg !14
  %1287 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1288 = bitcast <4 x i32> %1287 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1289 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1188, !dbg !18
  %1290 = getelementptr i8, ptr addrspace(1) %1289, i64 128, !dbg !18
  %1291 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local366), !dbg !18
  %1292 = zext i32 %1291 to i64, !dbg !18
  %1293 = inttoptr i64 %1292 to ptr addrspace(3), !dbg !18
  %address.local385 = getelementptr [2 x i8], ptr addrspace(1) %1290, i64 %address.local371, !dbg !18
  %1294 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1295 = ptrtoint ptr addrspace(1) %address.local385 to i64, !dbg !18
  %1296 = sub i64 %1295, %1294, !dbg !18
  %dma.buffer.offset874 = trunc i64 %1296 to i32, !dbg !18
  %1297 = trunc i64 %1294 to i32, !dbg !18
  %1298 = insertelement <4 x i32> poison, i32 %1297, i64 0, !dbg !18
  %1299 = lshr i64 %1294, 32, !dbg !18
  %1300 = trunc i64 %1299 to i32, !dbg !18
  %1301 = insertelement <4 x i32> %1298, i32 %1300, i64 1, !dbg !18
  %1302 = insertelement <4 x i32> %1301, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc875 = insertelement <4 x i32> %1302, i32 131072, i64 3, !dbg !18
  %1303 = ptrtoint ptr addrspace(3) %1293 to i32, !dbg !18
  %dma.buffer.lds876 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1303), !dbg !18
  %1304 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds876, i32 %dma.buffer.offset874, <4 x i32> %dma.buffer.rsrc875) #10, !dbg !18
  %1305 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local378), !dbg !18
  %1306 = zext i32 %1305 to i64, !dbg !18
  %1307 = inttoptr i64 %1306 to ptr addrspace(3), !dbg !18
  %address.local386 = getelementptr [2 x i8], ptr addrspace(1) %1290, i64 %address.local383, !dbg !18
  %1308 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1309 = ptrtoint ptr addrspace(1) %address.local386 to i64, !dbg !18
  %1310 = sub i64 %1309, %1308, !dbg !18
  %dma.buffer.offset871 = trunc i64 %1310 to i32, !dbg !18
  %1311 = trunc i64 %1308 to i32, !dbg !18
  %1312 = insertelement <4 x i32> poison, i32 %1311, i64 0, !dbg !18
  %1313 = lshr i64 %1308, 32, !dbg !18
  %1314 = trunc i64 %1313 to i32, !dbg !18
  %1315 = insertelement <4 x i32> %1312, i32 %1314, i64 1, !dbg !18
  %1316 = insertelement <4 x i32> %1315, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc872 = insertelement <4 x i32> %1316, i32 131072, i64 3, !dbg !18
  %1317 = ptrtoint ptr addrspace(3) %1307 to i32, !dbg !18
  %dma.buffer.lds873 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1317), !dbg !18
  %1318 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds873, i32 %dma.buffer.offset871, <4 x i32> %dma.buffer.rsrc872) #10, !dbg !18
  %1319 = shufflevector <8 x half> %1274, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1320 = extractvalue [2 x <8 x float>] %44, 0, !dbg !30
  %1321 = shufflevector <8 x float> %1320, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1319, <4 x float> %1321) #9, !dbg !14
  %1323 = shufflevector <8 x half> %1276, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1323, <4 x float> %1322) #9, !dbg !14
  %1325 = shufflevector <8 x half> %1278, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1325, <4 x float> %1324) #9, !dbg !14
  %1327 = shufflevector <8 x half> %1280, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1327, <4 x float> %1326) #9, !dbg !14
  %1329 = shufflevector <8 x half> %1282, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1329, <4 x float> %1328) #9, !dbg !14
  %1331 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1331, <4 x float> %1330) #9, !dbg !14
  %1333 = shufflevector <8 x half> %1286, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1333, <4 x float> %1332) #9, !dbg !14
  %1335 = shufflevector <8 x half> %1288, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1335, <4 x float> %1334) #9, !dbg !14
  %1337 = shufflevector <8 x half> %1274, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1338 = shufflevector <8 x float> %1320, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1337, <4 x float> %1338) #9, !dbg !14
  %1340 = shufflevector <8 x half> %1276, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1340, <4 x float> %1339) #9, !dbg !14
  %1342 = shufflevector <8 x half> %1278, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1342, <4 x float> %1341) #9, !dbg !14
  %1344 = shufflevector <8 x half> %1280, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1344, <4 x float> %1343) #9, !dbg !14
  %1346 = shufflevector <8 x half> %1282, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1346, <4 x float> %1345) #9, !dbg !14
  %1348 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1348, <4 x float> %1347) #9, !dbg !14
  %1350 = shufflevector <8 x half> %1286, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1350, <4 x float> %1349) #9, !dbg !14
  %1352 = shufflevector <8 x half> %1288, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1352, <4 x float> %1351) #9, !dbg !14
  %1354 = extractvalue [2 x <8 x float>] %44, 1, !dbg !30
  %1355 = shufflevector <8 x float> %1354, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1319, <4 x float> %1355) #9, !dbg !14
  %1357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1323, <4 x float> %1356) #9, !dbg !14
  %1358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1325, <4 x float> %1357) #9, !dbg !14
  %1359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1327, <4 x float> %1358) #9, !dbg !14
  %1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1329, <4 x float> %1359) #9, !dbg !14
  %1361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1331, <4 x float> %1360) #9, !dbg !14
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1333, <4 x float> %1361) #9, !dbg !14
  %1363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1335, <4 x float> %1362) #9, !dbg !14
  %1364 = shufflevector <8 x float> %1354, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1337, <4 x float> %1364) #9, !dbg !14
  %1366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1340, <4 x float> %1365) #9, !dbg !14
  %1367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1342, <4 x float> %1366) #9, !dbg !14
  %1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1344, <4 x float> %1367) #9, !dbg !14
  %1369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1346, <4 x float> %1368) #9, !dbg !14
  %1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1348, <4 x float> %1369) #9, !dbg !14
  %1371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1350, <4 x float> %1370) #9, !dbg !14
  %1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1352, <4 x float> %1371) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1373 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %1374 = bitcast <4 x i32> %1373 to <8 x half>, !dbg !14
  %1375 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %1376 = bitcast <4 x i32> %1375 to <8 x half>, !dbg !14
  %1377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %1378 = bitcast <4 x i32> %1377 to <8 x half>, !dbg !14
  %1379 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %1380 = bitcast <4 x i32> %1379 to <8 x half>, !dbg !14
  %1381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %1382 = bitcast <4 x i32> %1381 to <8 x half>, !dbg !14
  %1383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %1384 = bitcast <4 x i32> %1383 to <8 x half>, !dbg !14
  %1385 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %1386 = bitcast <4 x i32> %1385 to <8 x half>, !dbg !14
  %1387 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1388 = bitcast <4 x i32> %1387 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1389 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1188, !dbg !18
  %1390 = getelementptr i8, ptr addrspace(1) %1389, i64 192, !dbg !18
  %1391 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local366), !dbg !18
  %1392 = zext i32 %1391 to i64, !dbg !18
  %1393 = inttoptr i64 %1392 to ptr addrspace(3), !dbg !18
  %address.local387 = getelementptr [2 x i8], ptr addrspace(1) %1390, i64 %address.local371, !dbg !18
  %1394 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1395 = ptrtoint ptr addrspace(1) %address.local387 to i64, !dbg !18
  %1396 = sub i64 %1395, %1394, !dbg !18
  %dma.buffer.offset868 = trunc i64 %1396 to i32, !dbg !18
  %1397 = trunc i64 %1394 to i32, !dbg !18
  %1398 = insertelement <4 x i32> poison, i32 %1397, i64 0, !dbg !18
  %1399 = lshr i64 %1394, 32, !dbg !18
  %1400 = trunc i64 %1399 to i32, !dbg !18
  %1401 = insertelement <4 x i32> %1398, i32 %1400, i64 1, !dbg !18
  %1402 = insertelement <4 x i32> %1401, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc869 = insertelement <4 x i32> %1402, i32 131072, i64 3, !dbg !18
  %1403 = ptrtoint ptr addrspace(3) %1393 to i32, !dbg !18
  %dma.buffer.lds870 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1403), !dbg !18
  %1404 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds870, i32 %dma.buffer.offset868, <4 x i32> %dma.buffer.rsrc869) #10, !dbg !18
  %1405 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local378), !dbg !18
  %1406 = zext i32 %1405 to i64, !dbg !18
  %1407 = inttoptr i64 %1406 to ptr addrspace(3), !dbg !18
  %address.local388 = getelementptr [2 x i8], ptr addrspace(1) %1390, i64 %address.local383, !dbg !18
  %1408 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1409 = ptrtoint ptr addrspace(1) %address.local388 to i64, !dbg !18
  %1410 = sub i64 %1409, %1408, !dbg !18
  %dma.buffer.offset = trunc i64 %1410 to i32, !dbg !18
  %1411 = trunc i64 %1408 to i32, !dbg !18
  %1412 = insertelement <4 x i32> poison, i32 %1411, i64 0, !dbg !18
  %1413 = lshr i64 %1408, 32, !dbg !18
  %1414 = trunc i64 %1413 to i32, !dbg !18
  %1415 = insertelement <4 x i32> %1412, i32 %1414, i64 1, !dbg !18
  %1416 = insertelement <4 x i32> %1415, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1416, i32 131072, i64 3, !dbg !18
  %1417 = ptrtoint ptr addrspace(3) %1407 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1417), !dbg !18
  %1418 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1419 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1420 = extractvalue [2 x <8 x float>] %45, 0, !dbg !30
  %1421 = shufflevector <8 x float> %1420, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1419, <4 x float> %1421) #9, !dbg !14
  %1423 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1423, <4 x float> %1422) #9, !dbg !14
  %1425 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1425, <4 x float> %1424) #9, !dbg !14
  %1427 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1427, <4 x float> %1426) #9, !dbg !14
  %1429 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1429, <4 x float> %1428) #9, !dbg !14
  %1431 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1431, <4 x float> %1430) #9, !dbg !14
  %1433 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1433, <4 x float> %1432) #9, !dbg !14
  %1435 = shufflevector <8 x half> %1388, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1435, <4 x float> %1434) #9, !dbg !14
  %1437 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1438 = shufflevector <8 x float> %1420, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1437, <4 x float> %1438) #9, !dbg !14
  %1440 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1440, <4 x float> %1439) #9, !dbg !14
  %1442 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1442, <4 x float> %1441) #9, !dbg !14
  %1444 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1444, <4 x float> %1443) #9, !dbg !14
  %1446 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1446, <4 x float> %1445) #9, !dbg !14
  %1448 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1448, <4 x float> %1447) #9, !dbg !14
  %1450 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1450, <4 x float> %1449) #9, !dbg !14
  %1452 = shufflevector <8 x half> %1388, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1452, <4 x float> %1451) #9, !dbg !14
  %1454 = extractvalue [2 x <8 x float>] %45, 1, !dbg !30
  %1455 = shufflevector <8 x float> %1454, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1419, <4 x float> %1455) #9, !dbg !14
  %1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1423, <4 x float> %1456) #9, !dbg !14
  %1458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1425, <4 x float> %1457) #9, !dbg !14
  %1459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1427, <4 x float> %1458) #9, !dbg !14
  %1460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1429, <4 x float> %1459) #9, !dbg !14
  %1461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1431, <4 x float> %1460) #9, !dbg !14
  %1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1433, <4 x float> %1461) #9, !dbg !14
  %1463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1435, <4 x float> %1462) #9, !dbg !14
  %1464 = shufflevector <8 x float> %1454, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1437, <4 x float> %1464) #9, !dbg !14
  %1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1440, <4 x float> %1465) #9, !dbg !14
  %1467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1442, <4 x float> %1466) #9, !dbg !14
  %1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1444, <4 x float> %1467) #9, !dbg !14
  %1469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1446, <4 x float> %1468) #9, !dbg !14
  %1470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1448, <4 x float> %1469) #9, !dbg !14
  %1471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1450, <4 x float> %1470) #9, !dbg !14
  %1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1452, <4 x float> %1471) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1473 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %1474 = bitcast <4 x i32> %1473 to <8 x half>, !dbg !14
  %1475 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %1476 = bitcast <4 x i32> %1475 to <8 x half>, !dbg !14
  %1477 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %1478 = bitcast <4 x i32> %1477 to <8 x half>, !dbg !14
  %1479 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %1480 = bitcast <4 x i32> %1479 to <8 x half>, !dbg !14
  %1481 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %1482 = bitcast <4 x i32> %1481 to <8 x half>, !dbg !14
  %1483 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %1484 = bitcast <4 x i32> %1483 to <8 x half>, !dbg !14
  %1485 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %1486 = bitcast <4 x i32> %1485 to <8 x half>, !dbg !14
  %1487 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1488 = bitcast <4 x i32> %1487 to <8 x half>, !dbg !14
  %1489 = shufflevector <8 x half> %1474, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1490 = extractvalue [2 x <8 x float>] %46, 0, !dbg !30
  %1491 = shufflevector <8 x float> %1490, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1489, <4 x float> %1491) #9, !dbg !14
  %1493 = shufflevector <8 x half> %1476, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1493, <4 x float> %1492) #9, !dbg !14
  %1495 = shufflevector <8 x half> %1478, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1495, <4 x float> %1494) #9, !dbg !14
  %1497 = shufflevector <8 x half> %1480, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1497, <4 x float> %1496) #9, !dbg !14
  %1499 = shufflevector <8 x half> %1482, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1499, <4 x float> %1498) #9, !dbg !14
  %1501 = shufflevector <8 x half> %1484, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1501, <4 x float> %1500) #9, !dbg !14
  %1503 = shufflevector <8 x half> %1486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1503, <4 x float> %1502) #9, !dbg !14
  %1505 = shufflevector <8 x half> %1488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1505, <4 x float> %1504) #9, !dbg !14
  %1507 = shufflevector <8 x half> %1474, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1508 = shufflevector <8 x float> %1490, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1507, <4 x float> %1508) #9, !dbg !14
  %1510 = shufflevector <8 x half> %1476, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %1510, <4 x float> %1509) #9, !dbg !14
  %1512 = shufflevector <8 x half> %1478, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %1512, <4 x float> %1511) #9, !dbg !14
  %1514 = shufflevector <8 x half> %1480, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1514, <4 x float> %1513) #9, !dbg !14
  %1516 = shufflevector <8 x half> %1482, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1516, <4 x float> %1515) #9, !dbg !14
  %1518 = shufflevector <8 x half> %1484, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1518, <4 x float> %1517) #9, !dbg !14
  %1520 = shufflevector <8 x half> %1486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1520, <4 x float> %1519) #9, !dbg !14
  %1522 = shufflevector <8 x half> %1488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1522, <4 x float> %1521) #9, !dbg !14
  %1524 = extractvalue [2 x <8 x float>] %46, 1, !dbg !30
  %1525 = shufflevector <8 x float> %1524, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1489, <4 x float> %1525) #9, !dbg !14
  %1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1493, <4 x float> %1526) #9, !dbg !14
  %1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1495, <4 x float> %1527) #9, !dbg !14
  %1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1497, <4 x float> %1528) #9, !dbg !14
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1499, <4 x float> %1529) #9, !dbg !14
  %1531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1501, <4 x float> %1530) #9, !dbg !14
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1503, <4 x float> %1531) #9, !dbg !14
  %1533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1505, <4 x float> %1532) #9, !dbg !14
  %1534 = shufflevector <8 x float> %1524, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1507, <4 x float> %1534) #9, !dbg !14
  %1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1510, <4 x float> %1535) #9, !dbg !14
  %1537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1512, <4 x float> %1536) #9, !dbg !14
  %1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1514, <4 x float> %1537) #9, !dbg !14
  %1539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1516, <4 x float> %1538) #9, !dbg !14
  %1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1518, <4 x float> %1539) #9, !dbg !14
  %1541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1520, <4 x float> %1540) #9, !dbg !14
  %1542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1522, <4 x float> %1541) #9, !dbg !14
  %1543 = shufflevector <4 x float> %1236, <4 x float> %1253, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1544 = insertvalue [2 x <8 x float>] poison, <8 x float> %1543, 0, !dbg !15
  %1545 = shufflevector <4 x float> %1263, <4 x float> %1272, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1546 = insertvalue [2 x <8 x float>] %1544, <8 x float> %1545, 1, !dbg !15
  %1547 = shufflevector <4 x float> %1336, <4 x float> %1353, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1548 = insertvalue [2 x <8 x float>] poison, <8 x float> %1547, 0, !dbg !15
  %1549 = shufflevector <4 x float> %1363, <4 x float> %1372, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1550 = insertvalue [2 x <8 x float>] %1548, <8 x float> %1549, 1, !dbg !15
  %1551 = shufflevector <4 x float> %1436, <4 x float> %1453, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1552 = insertvalue [2 x <8 x float>] poison, <8 x float> %1551, 0, !dbg !15
  %1553 = shufflevector <4 x float> %1463, <4 x float> %1472, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1554 = insertvalue [2 x <8 x float>] %1552, <8 x float> %1553, 1, !dbg !15
  %1555 = shufflevector <4 x float> %1506, <4 x float> %1523, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1556 = insertvalue [2 x <8 x float>] poison, <8 x float> %1555, 0, !dbg !15
  %1557 = shufflevector <4 x float> %1533, <4 x float> %1542, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1558 = insertvalue [2 x <8 x float>] %1556, <8 x float> %1557, 1, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1559 = add nuw nsw i64 %41, 1, !dbg !15
  br label %40, !dbg !15

1560:                                             ; preds = %40
  %1561 = and i32 %24, 16, !dbg !15
  %.not2 = icmp eq i32 %1561, 0, !dbg !15
  %1562 = and i32 %24, 32, !dbg !15
  %.not3 = icmp eq i32 %1562, 0, !dbg !15
  %1563 = extractvalue [2 x <8 x float>] %46, 0, !dbg !15
  %1564 = extractelement <8 x float> %1563, i64 0, !dbg !15
  %1565 = extractelement <8 x float> %1563, i64 1, !dbg !15
  %1566 = extractelement <8 x float> %1563, i64 2, !dbg !15
  %1567 = extractelement <8 x float> %1563, i64 3, !dbg !15
  %1568 = select i1 %.not2, float %1565, float %1564, !dbg !15
  %1569 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !15
  %1570 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1569), !dbg !15
  %1571 = add nuw nsw i32 %1570, 64, !dbg !15
  %1572 = xor i32 %1570, 16, !dbg !15
  %1573 = and i32 %1571, 192, !dbg !15
  %1574 = icmp samesign ult i32 %1572, %1573, !dbg !15
  %1575 = select i1 %1574, i32 %1572, i32 %1570, !dbg !15
  %1576 = shl nuw nsw i32 %1575, 2, !dbg !15
  %1577 = bitcast float %1568 to i32, !dbg !15
  %1578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1577), !dbg !15
  %1579 = bitcast i32 %1578 to float, !dbg !15
  %1580 = select i1 %.not2, float %1564, float %1579, !dbg !15
  %1581 = select i1 %.not2, float %1579, float %1565, !dbg !15
  %1582 = select i1 %.not2, float %1567, float %1566, !dbg !15
  %1583 = bitcast float %1582 to i32, !dbg !15
  %1584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1583), !dbg !15
  %1585 = bitcast i32 %1584 to float, !dbg !15
  %1586 = select i1 %.not2, float %1566, float %1585, !dbg !15
  %1587 = select i1 %.not2, float %1585, float %1567, !dbg !15
  %1588 = select i1 %.not3, float %1586, float %1580, !dbg !15
  %1589 = xor i32 %1570, 32, !dbg !15
  %1590 = icmp samesign ult i32 %1589, %1573, !dbg !15
  %1591 = select i1 %1590, i32 %1589, i32 %1570, !dbg !15
  %1592 = shl nuw nsw i32 %1591, 2, !dbg !15
  %1593 = bitcast float %1588 to i32, !dbg !15
  %1594 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1593), !dbg !15
  %1595 = bitcast i32 %1594 to float, !dbg !15
  %1596 = select i1 %.not3, float %1580, float %1595, !dbg !15
  %1597 = select i1 %.not3, float %1595, float %1586, !dbg !15
  %1598 = select i1 %.not3, float %1587, float %1581, !dbg !15
  %1599 = bitcast float %1598 to i32, !dbg !15
  %1600 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1599), !dbg !15
  %1601 = bitcast i32 %1600 to float, !dbg !15
  %1602 = select i1 %.not3, float %1581, float %1601, !dbg !15
  %1603 = select i1 %.not3, float %1601, float %1587, !dbg !15
  %1604 = extractvalue [2 x <8 x float>] %45, 0, !dbg !15
  %1605 = extractelement <8 x float> %1604, i64 0, !dbg !15
  %1606 = extractelement <8 x float> %1604, i64 1, !dbg !15
  %1607 = extractelement <8 x float> %1604, i64 2, !dbg !15
  %1608 = extractelement <8 x float> %1604, i64 3, !dbg !15
  %1609 = select i1 %.not2, float %1606, float %1605, !dbg !15
  %1610 = bitcast float %1609 to i32, !dbg !15
  %1611 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1610), !dbg !15
  %1612 = bitcast i32 %1611 to float, !dbg !15
  %1613 = select i1 %.not2, float %1605, float %1612, !dbg !15
  %1614 = select i1 %.not2, float %1612, float %1606, !dbg !15
  %1615 = select i1 %.not2, float %1608, float %1607, !dbg !15
  %1616 = bitcast float %1615 to i32, !dbg !15
  %1617 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1616), !dbg !15
  %1618 = bitcast i32 %1617 to float, !dbg !15
  %1619 = select i1 %.not2, float %1607, float %1618, !dbg !15
  %1620 = select i1 %.not2, float %1618, float %1608, !dbg !15
  %1621 = select i1 %.not3, float %1619, float %1613, !dbg !15
  %1622 = bitcast float %1621 to i32, !dbg !15
  %1623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1622), !dbg !15
  %1624 = bitcast i32 %1623 to float, !dbg !15
  %1625 = select i1 %.not3, float %1613, float %1624, !dbg !15
  %1626 = select i1 %.not3, float %1624, float %1619, !dbg !15
  %1627 = select i1 %.not3, float %1620, float %1614, !dbg !15
  %1628 = bitcast float %1627 to i32, !dbg !15
  %1629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1628), !dbg !15
  %1630 = bitcast i32 %1629 to float, !dbg !15
  %1631 = select i1 %.not3, float %1614, float %1630, !dbg !15
  %1632 = select i1 %.not3, float %1630, float %1620, !dbg !15
  %1633 = extractvalue [2 x <8 x float>] %44, 0, !dbg !15
  %1634 = extractelement <8 x float> %1633, i64 0, !dbg !15
  %1635 = extractelement <8 x float> %1633, i64 1, !dbg !15
  %1636 = extractelement <8 x float> %1633, i64 2, !dbg !15
  %1637 = extractelement <8 x float> %1633, i64 3, !dbg !15
  %1638 = select i1 %.not2, float %1635, float %1634, !dbg !15
  %1639 = bitcast float %1638 to i32, !dbg !15
  %1640 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1639), !dbg !15
  %1641 = bitcast i32 %1640 to float, !dbg !15
  %1642 = select i1 %.not2, float %1634, float %1641, !dbg !15
  %1643 = select i1 %.not2, float %1641, float %1635, !dbg !15
  %1644 = select i1 %.not2, float %1637, float %1636, !dbg !15
  %1645 = bitcast float %1644 to i32, !dbg !15
  %1646 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1645), !dbg !15
  %1647 = bitcast i32 %1646 to float, !dbg !15
  %1648 = select i1 %.not2, float %1636, float %1647, !dbg !15
  %1649 = select i1 %.not2, float %1647, float %1637, !dbg !15
  %1650 = select i1 %.not3, float %1648, float %1642, !dbg !15
  %1651 = bitcast float %1650 to i32, !dbg !15
  %1652 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1651), !dbg !15
  %1653 = bitcast i32 %1652 to float, !dbg !15
  %1654 = select i1 %.not3, float %1642, float %1653, !dbg !15
  %1655 = select i1 %.not3, float %1653, float %1648, !dbg !15
  %1656 = select i1 %.not3, float %1649, float %1643, !dbg !15
  %1657 = bitcast float %1656 to i32, !dbg !15
  %1658 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1657), !dbg !15
  %1659 = bitcast i32 %1658 to float, !dbg !15
  %1660 = select i1 %.not3, float %1643, float %1659, !dbg !15
  %1661 = select i1 %.not3, float %1659, float %1649, !dbg !15
  %1662 = extractvalue [2 x <8 x float>] %42, 0, !dbg !15
  %1663 = extractelement <8 x float> %1662, i64 0, !dbg !15
  %1664 = extractelement <8 x float> %1662, i64 1, !dbg !15
  %1665 = extractelement <8 x float> %1662, i64 2, !dbg !15
  %1666 = extractelement <8 x float> %1662, i64 3, !dbg !15
  %1667 = select i1 %.not2, float %1664, float %1663, !dbg !15
  %1668 = bitcast float %1667 to i32, !dbg !15
  %1669 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1668), !dbg !15
  %1670 = bitcast i32 %1669 to float, !dbg !15
  %1671 = select i1 %.not2, float %1663, float %1670, !dbg !15
  %1672 = select i1 %.not2, float %1670, float %1664, !dbg !15
  %1673 = select i1 %.not2, float %1666, float %1665, !dbg !15
  %1674 = bitcast float %1673 to i32, !dbg !15
  %1675 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1674), !dbg !15
  %1676 = bitcast i32 %1675 to float, !dbg !15
  %1677 = select i1 %.not2, float %1665, float %1676, !dbg !15
  %1678 = select i1 %.not2, float %1676, float %1666, !dbg !15
  %1679 = select i1 %.not3, float %1677, float %1671, !dbg !15
  %1680 = bitcast float %1679 to i32, !dbg !15
  %1681 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1680), !dbg !15
  %1682 = select i1 %.not3, float %1678, float %1672, !dbg !15
  %1683 = bitcast float %1682 to i32, !dbg !15
  %1684 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1683), !dbg !15
  %1685 = extractvalue [2 x <1 x float>] %43, 0, !dbg !19
  %1686 = shufflevector <1 x float> %1685, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1687 = extractelement <8 x float> %1563, i64 4, !dbg !15
  %1688 = extractelement <8 x float> %1563, i64 5, !dbg !15
  %1689 = extractelement <8 x float> %1563, i64 6, !dbg !15
  %1690 = extractelement <8 x float> %1563, i64 7, !dbg !15
  %1691 = select i1 %.not2, float %1688, float %1687, !dbg !15
  %1692 = bitcast float %1691 to i32, !dbg !15
  %1693 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1692), !dbg !15
  %1694 = bitcast i32 %1693 to float, !dbg !15
  %1695 = select i1 %.not2, float %1687, float %1694, !dbg !15
  %1696 = select i1 %.not2, float %1694, float %1688, !dbg !15
  %1697 = select i1 %.not2, float %1690, float %1689, !dbg !15
  %1698 = bitcast float %1697 to i32, !dbg !15
  %1699 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1698), !dbg !15
  %1700 = bitcast i32 %1699 to float, !dbg !15
  %1701 = select i1 %.not2, float %1689, float %1700, !dbg !15
  %1702 = select i1 %.not2, float %1700, float %1690, !dbg !15
  %1703 = select i1 %.not3, float %1701, float %1695, !dbg !15
  %1704 = bitcast float %1703 to i32, !dbg !15
  %1705 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1704), !dbg !15
  %1706 = bitcast i32 %1705 to float, !dbg !15
  %1707 = select i1 %.not3, float %1695, float %1706, !dbg !15
  %1708 = select i1 %.not3, float %1706, float %1701, !dbg !15
  %1709 = select i1 %.not3, float %1702, float %1696, !dbg !15
  %1710 = bitcast float %1709 to i32, !dbg !15
  %1711 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1710), !dbg !15
  %1712 = bitcast i32 %1711 to float, !dbg !15
  %1713 = select i1 %.not3, float %1696, float %1712, !dbg !15
  %1714 = select i1 %.not3, float %1712, float %1702, !dbg !15
  %1715 = extractelement <8 x float> %1604, i64 4, !dbg !15
  %1716 = extractelement <8 x float> %1604, i64 5, !dbg !15
  %1717 = extractelement <8 x float> %1604, i64 6, !dbg !15
  %1718 = extractelement <8 x float> %1604, i64 7, !dbg !15
  %1719 = select i1 %.not2, float %1716, float %1715, !dbg !15
  %1720 = bitcast float %1719 to i32, !dbg !15
  %1721 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1720), !dbg !15
  %1722 = bitcast i32 %1721 to float, !dbg !15
  %1723 = select i1 %.not2, float %1715, float %1722, !dbg !15
  %1724 = select i1 %.not2, float %1722, float %1716, !dbg !15
  %1725 = select i1 %.not2, float %1718, float %1717, !dbg !15
  %1726 = bitcast float %1725 to i32, !dbg !15
  %1727 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1726), !dbg !15
  %1728 = bitcast i32 %1727 to float, !dbg !15
  %1729 = select i1 %.not2, float %1717, float %1728, !dbg !15
  %1730 = select i1 %.not2, float %1728, float %1718, !dbg !15
  %1731 = select i1 %.not3, float %1729, float %1723, !dbg !15
  %1732 = bitcast float %1731 to i32, !dbg !15
  %1733 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1732), !dbg !15
  %1734 = bitcast i32 %1733 to float, !dbg !15
  %1735 = select i1 %.not3, float %1723, float %1734, !dbg !15
  %1736 = select i1 %.not3, float %1734, float %1729, !dbg !15
  %1737 = select i1 %.not3, float %1730, float %1724, !dbg !15
  %1738 = bitcast float %1737 to i32, !dbg !15
  %1739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1738), !dbg !15
  %1740 = bitcast i32 %1739 to float, !dbg !15
  %1741 = select i1 %.not3, float %1724, float %1740, !dbg !15
  %1742 = select i1 %.not3, float %1740, float %1730, !dbg !15
  %1743 = extractelement <8 x float> %1633, i64 4, !dbg !15
  %1744 = extractelement <8 x float> %1633, i64 5, !dbg !15
  %1745 = extractelement <8 x float> %1633, i64 6, !dbg !15
  %1746 = extractelement <8 x float> %1633, i64 7, !dbg !15
  %1747 = select i1 %.not2, float %1744, float %1743, !dbg !15
  %1748 = bitcast float %1747 to i32, !dbg !15
  %1749 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1748), !dbg !15
  %1750 = bitcast i32 %1749 to float, !dbg !15
  %1751 = select i1 %.not2, float %1743, float %1750, !dbg !15
  %1752 = select i1 %.not2, float %1750, float %1744, !dbg !15
  %1753 = select i1 %.not2, float %1746, float %1745, !dbg !15
  %1754 = bitcast float %1753 to i32, !dbg !15
  %1755 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1754), !dbg !15
  %1756 = bitcast i32 %1755 to float, !dbg !15
  %1757 = select i1 %.not2, float %1745, float %1756, !dbg !15
  %1758 = select i1 %.not2, float %1756, float %1746, !dbg !15
  %1759 = select i1 %.not3, float %1757, float %1751, !dbg !15
  %1760 = bitcast float %1759 to i32, !dbg !15
  %1761 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1760), !dbg !15
  %1762 = bitcast i32 %1761 to float, !dbg !15
  %1763 = select i1 %.not3, float %1751, float %1762, !dbg !15
  %1764 = select i1 %.not3, float %1762, float %1757, !dbg !15
  %1765 = select i1 %.not3, float %1758, float %1752, !dbg !15
  %1766 = bitcast float %1765 to i32, !dbg !15
  %1767 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1766), !dbg !15
  %1768 = bitcast i32 %1767 to float, !dbg !15
  %1769 = select i1 %.not3, float %1752, float %1768, !dbg !15
  %1770 = select i1 %.not3, float %1768, float %1758, !dbg !15
  %1771 = extractelement <8 x float> %1662, i64 4, !dbg !15
  %1772 = extractelement <8 x float> %1662, i64 5, !dbg !15
  %1773 = extractelement <8 x float> %1662, i64 6, !dbg !15
  %1774 = extractelement <8 x float> %1662, i64 7, !dbg !15
  %1775 = select i1 %.not2, float %1772, float %1771, !dbg !15
  %1776 = bitcast float %1775 to i32, !dbg !15
  %1777 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1776), !dbg !15
  %1778 = bitcast i32 %1777 to float, !dbg !15
  %1779 = select i1 %.not2, float %1771, float %1778, !dbg !15
  %1780 = select i1 %.not2, float %1778, float %1772, !dbg !15
  %1781 = select i1 %.not2, float %1774, float %1773, !dbg !15
  %1782 = bitcast float %1781 to i32, !dbg !15
  %1783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1782), !dbg !15
  %1784 = bitcast i32 %1783 to float, !dbg !15
  %1785 = select i1 %.not2, float %1773, float %1784, !dbg !15
  %1786 = select i1 %.not2, float %1784, float %1774, !dbg !15
  %1787 = select i1 %.not3, float %1785, float %1779, !dbg !15
  %1788 = bitcast float %1787 to i32, !dbg !15
  %1789 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1788), !dbg !15
  %1790 = select i1 %.not3, float %1786, float %1780, !dbg !15
  %1791 = bitcast float %1790 to i32, !dbg !15
  %1792 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1791), !dbg !15
  %1793 = extractvalue [2 x <8 x float>] %46, 1, !dbg !15
  %1794 = extractelement <8 x float> %1793, i64 0, !dbg !15
  %1795 = extractelement <8 x float> %1793, i64 1, !dbg !15
  %1796 = extractelement <8 x float> %1793, i64 2, !dbg !15
  %1797 = extractelement <8 x float> %1793, i64 3, !dbg !15
  %1798 = select i1 %.not2, float %1795, float %1794, !dbg !15
  %1799 = bitcast float %1798 to i32, !dbg !15
  %1800 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1799), !dbg !15
  %1801 = bitcast i32 %1800 to float, !dbg !15
  %1802 = select i1 %.not2, float %1794, float %1801, !dbg !15
  %1803 = select i1 %.not2, float %1801, float %1795, !dbg !15
  %1804 = select i1 %.not2, float %1797, float %1796, !dbg !15
  %1805 = bitcast float %1804 to i32, !dbg !15
  %1806 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1805), !dbg !15
  %1807 = bitcast i32 %1806 to float, !dbg !15
  %1808 = select i1 %.not2, float %1796, float %1807, !dbg !15
  %1809 = select i1 %.not2, float %1807, float %1797, !dbg !15
  %1810 = select i1 %.not3, float %1808, float %1802, !dbg !15
  %1811 = bitcast float %1810 to i32, !dbg !15
  %1812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1811), !dbg !15
  %1813 = bitcast i32 %1812 to float, !dbg !15
  %1814 = select i1 %.not3, float %1802, float %1813, !dbg !15
  %1815 = select i1 %.not3, float %1813, float %1808, !dbg !15
  %1816 = select i1 %.not3, float %1809, float %1803, !dbg !15
  %1817 = bitcast float %1816 to i32, !dbg !15
  %1818 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1817), !dbg !15
  %1819 = bitcast i32 %1818 to float, !dbg !15
  %1820 = select i1 %.not3, float %1803, float %1819, !dbg !15
  %1821 = select i1 %.not3, float %1819, float %1809, !dbg !15
  %1822 = extractvalue [2 x <8 x float>] %45, 1, !dbg !15
  %1823 = extractelement <8 x float> %1822, i64 0, !dbg !15
  %1824 = extractelement <8 x float> %1822, i64 1, !dbg !15
  %1825 = extractelement <8 x float> %1822, i64 2, !dbg !15
  %1826 = extractelement <8 x float> %1822, i64 3, !dbg !15
  %1827 = select i1 %.not2, float %1824, float %1823, !dbg !15
  %1828 = bitcast float %1827 to i32, !dbg !15
  %1829 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1828), !dbg !15
  %1830 = bitcast i32 %1829 to float, !dbg !15
  %1831 = select i1 %.not2, float %1823, float %1830, !dbg !15
  %1832 = select i1 %.not2, float %1830, float %1824, !dbg !15
  %1833 = select i1 %.not2, float %1826, float %1825, !dbg !15
  %1834 = bitcast float %1833 to i32, !dbg !15
  %1835 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1834), !dbg !15
  %1836 = bitcast i32 %1835 to float, !dbg !15
  %1837 = select i1 %.not2, float %1825, float %1836, !dbg !15
  %1838 = select i1 %.not2, float %1836, float %1826, !dbg !15
  %1839 = select i1 %.not3, float %1837, float %1831, !dbg !15
  %1840 = bitcast float %1839 to i32, !dbg !15
  %1841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1840), !dbg !15
  %1842 = bitcast i32 %1841 to float, !dbg !15
  %1843 = select i1 %.not3, float %1831, float %1842, !dbg !15
  %1844 = select i1 %.not3, float %1842, float %1837, !dbg !15
  %1845 = select i1 %.not3, float %1838, float %1832, !dbg !15
  %1846 = bitcast float %1845 to i32, !dbg !15
  %1847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1846), !dbg !15
  %1848 = bitcast i32 %1847 to float, !dbg !15
  %1849 = select i1 %.not3, float %1832, float %1848, !dbg !15
  %1850 = select i1 %.not3, float %1848, float %1838, !dbg !15
  %1851 = extractvalue [2 x <8 x float>] %44, 1, !dbg !15
  %1852 = extractelement <8 x float> %1851, i64 0, !dbg !15
  %1853 = extractelement <8 x float> %1851, i64 1, !dbg !15
  %1854 = extractelement <8 x float> %1851, i64 2, !dbg !15
  %1855 = extractelement <8 x float> %1851, i64 3, !dbg !15
  %1856 = select i1 %.not2, float %1853, float %1852, !dbg !15
  %1857 = bitcast float %1856 to i32, !dbg !15
  %1858 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1857), !dbg !15
  %1859 = bitcast i32 %1858 to float, !dbg !15
  %1860 = select i1 %.not2, float %1852, float %1859, !dbg !15
  %1861 = select i1 %.not2, float %1859, float %1853, !dbg !15
  %1862 = select i1 %.not2, float %1855, float %1854, !dbg !15
  %1863 = bitcast float %1862 to i32, !dbg !15
  %1864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1863), !dbg !15
  %1865 = bitcast i32 %1864 to float, !dbg !15
  %1866 = select i1 %.not2, float %1854, float %1865, !dbg !15
  %1867 = select i1 %.not2, float %1865, float %1855, !dbg !15
  %1868 = select i1 %.not3, float %1866, float %1860, !dbg !15
  %1869 = bitcast float %1868 to i32, !dbg !15
  %1870 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1869), !dbg !15
  %1871 = bitcast i32 %1870 to float, !dbg !15
  %1872 = select i1 %.not3, float %1860, float %1871, !dbg !15
  %1873 = select i1 %.not3, float %1871, float %1866, !dbg !15
  %1874 = select i1 %.not3, float %1867, float %1861, !dbg !15
  %1875 = bitcast float %1874 to i32, !dbg !15
  %1876 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1875), !dbg !15
  %1877 = bitcast i32 %1876 to float, !dbg !15
  %1878 = select i1 %.not3, float %1861, float %1877, !dbg !15
  %1879 = select i1 %.not3, float %1877, float %1867, !dbg !15
  %1880 = extractvalue [2 x <8 x float>] %42, 1, !dbg !15
  %1881 = extractelement <8 x float> %1880, i64 0, !dbg !15
  %1882 = extractelement <8 x float> %1880, i64 1, !dbg !15
  %1883 = extractelement <8 x float> %1880, i64 2, !dbg !15
  %1884 = extractelement <8 x float> %1880, i64 3, !dbg !15
  %1885 = select i1 %.not2, float %1882, float %1881, !dbg !15
  %1886 = bitcast float %1885 to i32, !dbg !15
  %1887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1886), !dbg !15
  %1888 = bitcast i32 %1887 to float, !dbg !15
  %1889 = select i1 %.not2, float %1881, float %1888, !dbg !15
  %1890 = select i1 %.not2, float %1888, float %1882, !dbg !15
  %1891 = select i1 %.not2, float %1884, float %1883, !dbg !15
  %1892 = bitcast float %1891 to i32, !dbg !15
  %1893 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1892), !dbg !15
  %1894 = bitcast i32 %1893 to float, !dbg !15
  %1895 = select i1 %.not2, float %1883, float %1894, !dbg !15
  %1896 = select i1 %.not2, float %1894, float %1884, !dbg !15
  %1897 = select i1 %.not3, float %1895, float %1889, !dbg !15
  %1898 = bitcast float %1897 to i32, !dbg !15
  %1899 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1898), !dbg !15
  %1900 = select i1 %.not3, float %1896, float %1890, !dbg !15
  %1901 = bitcast float %1900 to i32, !dbg !15
  %1902 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1901), !dbg !15
  %1903 = extractvalue [2 x <1 x float>] %43, 1, !dbg !19
  %1904 = shufflevector <1 x float> %1903, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1905 = extractelement <8 x float> %1793, i64 4, !dbg !15
  %1906 = extractelement <8 x float> %1793, i64 5, !dbg !15
  %1907 = extractelement <8 x float> %1793, i64 6, !dbg !15
  %1908 = extractelement <8 x float> %1793, i64 7, !dbg !15
  %1909 = select i1 %.not2, float %1906, float %1905, !dbg !15
  %1910 = bitcast float %1909 to i32, !dbg !15
  %1911 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1910), !dbg !15
  %1912 = bitcast i32 %1911 to float, !dbg !15
  %1913 = select i1 %.not2, float %1905, float %1912, !dbg !15
  %1914 = select i1 %.not2, float %1912, float %1906, !dbg !15
  %1915 = select i1 %.not2, float %1908, float %1907, !dbg !15
  %1916 = bitcast float %1915 to i32, !dbg !15
  %1917 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1916), !dbg !15
  %1918 = bitcast i32 %1917 to float, !dbg !15
  %1919 = select i1 %.not2, float %1907, float %1918, !dbg !15
  %1920 = select i1 %.not2, float %1918, float %1908, !dbg !15
  %1921 = select i1 %.not3, float %1919, float %1913, !dbg !15
  %1922 = bitcast float %1921 to i32, !dbg !15
  %1923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1922), !dbg !15
  %1924 = bitcast i32 %1923 to float, !dbg !15
  %1925 = select i1 %.not3, float %1913, float %1924, !dbg !15
  %1926 = select i1 %.not3, float %1924, float %1919, !dbg !15
  %1927 = select i1 %.not3, float %1920, float %1914, !dbg !15
  %1928 = bitcast float %1927 to i32, !dbg !15
  %1929 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1928), !dbg !15
  %1930 = bitcast i32 %1929 to float, !dbg !15
  %1931 = select i1 %.not3, float %1914, float %1930, !dbg !15
  %1932 = select i1 %.not3, float %1930, float %1920, !dbg !15
  %1933 = extractelement <8 x float> %1822, i64 4, !dbg !15
  %1934 = extractelement <8 x float> %1822, i64 5, !dbg !15
  %1935 = extractelement <8 x float> %1822, i64 6, !dbg !15
  %1936 = extractelement <8 x float> %1822, i64 7, !dbg !15
  %1937 = select i1 %.not2, float %1934, float %1933, !dbg !15
  %1938 = bitcast float %1937 to i32, !dbg !15
  %1939 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1938), !dbg !15
  %1940 = bitcast i32 %1939 to float, !dbg !15
  %1941 = select i1 %.not2, float %1933, float %1940, !dbg !15
  %1942 = select i1 %.not2, float %1940, float %1934, !dbg !15
  %1943 = select i1 %.not2, float %1936, float %1935, !dbg !15
  %1944 = bitcast float %1943 to i32, !dbg !15
  %1945 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1944), !dbg !15
  %1946 = bitcast i32 %1945 to float, !dbg !15
  %1947 = select i1 %.not2, float %1935, float %1946, !dbg !15
  %1948 = select i1 %.not2, float %1946, float %1936, !dbg !15
  %1949 = select i1 %.not3, float %1947, float %1941, !dbg !15
  %1950 = bitcast float %1949 to i32, !dbg !15
  %1951 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1950), !dbg !15
  %1952 = bitcast i32 %1951 to float, !dbg !15
  %1953 = select i1 %.not3, float %1941, float %1952, !dbg !15
  %1954 = select i1 %.not3, float %1952, float %1947, !dbg !15
  %1955 = select i1 %.not3, float %1948, float %1942, !dbg !15
  %1956 = bitcast float %1955 to i32, !dbg !15
  %1957 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1956), !dbg !15
  %1958 = bitcast i32 %1957 to float, !dbg !15
  %1959 = select i1 %.not3, float %1942, float %1958, !dbg !15
  %1960 = select i1 %.not3, float %1958, float %1948, !dbg !15
  %1961 = extractelement <8 x float> %1851, i64 4, !dbg !15
  %1962 = extractelement <8 x float> %1851, i64 5, !dbg !15
  %1963 = extractelement <8 x float> %1851, i64 6, !dbg !15
  %1964 = extractelement <8 x float> %1851, i64 7, !dbg !15
  %1965 = select i1 %.not2, float %1962, float %1961, !dbg !15
  %1966 = bitcast float %1965 to i32, !dbg !15
  %1967 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1966), !dbg !15
  %1968 = bitcast i32 %1967 to float, !dbg !15
  %1969 = select i1 %.not2, float %1961, float %1968, !dbg !15
  %1970 = select i1 %.not2, float %1968, float %1962, !dbg !15
  %1971 = select i1 %.not2, float %1964, float %1963, !dbg !15
  %1972 = bitcast float %1971 to i32, !dbg !15
  %1973 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1972), !dbg !15
  %1974 = bitcast i32 %1973 to float, !dbg !15
  %1975 = select i1 %.not2, float %1963, float %1974, !dbg !15
  %1976 = select i1 %.not2, float %1974, float %1964, !dbg !15
  %1977 = select i1 %.not3, float %1975, float %1969, !dbg !15
  %1978 = bitcast float %1977 to i32, !dbg !15
  %1979 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1978), !dbg !15
  %1980 = bitcast i32 %1979 to float, !dbg !15
  %1981 = select i1 %.not3, float %1969, float %1980, !dbg !15
  %1982 = select i1 %.not3, float %1980, float %1975, !dbg !15
  %1983 = select i1 %.not3, float %1976, float %1970, !dbg !15
  %1984 = bitcast float %1983 to i32, !dbg !15
  %1985 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %1984), !dbg !15
  %1986 = bitcast i32 %1985 to float, !dbg !15
  %1987 = select i1 %.not3, float %1970, float %1986, !dbg !15
  %1988 = select i1 %.not3, float %1986, float %1976, !dbg !15
  %1989 = extractelement <8 x float> %1880, i64 4, !dbg !15
  %1990 = extractelement <8 x float> %1880, i64 5, !dbg !15
  %1991 = extractelement <8 x float> %1880, i64 6, !dbg !15
  %1992 = extractelement <8 x float> %1880, i64 7, !dbg !15
  %1993 = select i1 %.not2, float %1990, float %1989, !dbg !15
  %1994 = bitcast float %1993 to i32, !dbg !15
  %1995 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %1994), !dbg !15
  %1996 = bitcast i32 %1995 to float, !dbg !15
  %1997 = select i1 %.not2, float %1989, float %1996, !dbg !15
  %1998 = select i1 %.not2, float %1996, float %1990, !dbg !15
  %1999 = select i1 %.not2, float %1992, float %1991, !dbg !15
  %2000 = bitcast float %1999 to i32, !dbg !15
  %2001 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1576, i32 %2000), !dbg !15
  %2002 = bitcast i32 %2001 to float, !dbg !15
  %2003 = select i1 %.not2, float %1991, float %2002, !dbg !15
  %2004 = select i1 %.not2, float %2002, float %1992, !dbg !15
  %2005 = select i1 %.not3, float %2003, float %1997, !dbg !15
  %2006 = bitcast float %2005 to i32, !dbg !15
  %2007 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %2006), !dbg !15
  %2008 = select i1 %.not3, float %2004, float %1998, !dbg !15
  %2009 = bitcast float %2008 to i32, !dbg !15
  %2010 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1592, i32 %2009), !dbg !15
  br i1 true, label %2011, label %2075, !dbg !21

2011:                                             ; preds = %1560
  %2012 = bitcast i32 %2007 to float, !dbg !15
  %2013 = select i1 %.not3, float %1997, float %2012, !dbg !15
  %2014 = insertelement <4 x float> poison, float %2013, i64 0, !dbg !19
  %2015 = bitcast i32 %2010 to float, !dbg !15
  %2016 = select i1 %.not3, float %1998, float %2015, !dbg !15
  %2017 = insertelement <4 x float> %2014, float %2016, i64 1, !dbg !19
  %2018 = select i1 %.not3, float %2012, float %2003, !dbg !15
  %2019 = insertelement <4 x float> %2017, float %2018, i64 2, !dbg !19
  %2020 = select i1 %.not3, float %2015, float %2004, !dbg !15
  %2021 = insertelement <4 x float> %2019, float %2020, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1904, !dbg !19
  %normalized = fmul <4 x float> %2021, %row.reciprocal, !dbg !19
  %2022 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %2023 = extractelement <4 x half> %2022, i64 3, !dbg !20
  %2024 = extractelement <4 x half> %2022, i64 2, !dbg !20
  %2025 = extractelement <4 x half> %2022, i64 1, !dbg !20
  %2026 = extractelement <4 x half> %2022, i64 0, !dbg !20
  %2027 = bitcast i32 %1899 to float, !dbg !15
  %2028 = select i1 %.not3, float %1889, float %2027, !dbg !15
  %2029 = insertelement <4 x float> poison, float %2028, i64 0, !dbg !19
  %2030 = bitcast i32 %1902 to float, !dbg !15
  %2031 = select i1 %.not3, float %1890, float %2030, !dbg !15
  %2032 = insertelement <4 x float> %2029, float %2031, i64 1, !dbg !19
  %2033 = select i1 %.not3, float %2027, float %1895, !dbg !15
  %2034 = insertelement <4 x float> %2032, float %2033, i64 2, !dbg !19
  %2035 = select i1 %.not3, float %2030, float %1896, !dbg !15
  %2036 = insertelement <4 x float> %2034, float %2035, i64 3, !dbg !19
  %normalized71 = fmul <4 x float> %2036, %row.reciprocal, !dbg !19
  %2037 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !20
  %2038 = extractelement <4 x half> %2037, i64 3, !dbg !20
  %2039 = extractelement <4 x half> %2037, i64 2, !dbg !20
  %2040 = extractelement <4 x half> %2037, i64 1, !dbg !20
  %2041 = extractelement <4 x half> %2037, i64 0, !dbg !20
  %2042 = bitcast i32 %1789 to float, !dbg !15
  %2043 = select i1 %.not3, float %1779, float %2042, !dbg !15
  %2044 = insertelement <4 x float> poison, float %2043, i64 0, !dbg !19
  %2045 = bitcast i32 %1792 to float, !dbg !15
  %2046 = select i1 %.not3, float %1780, float %2045, !dbg !15
  %2047 = insertelement <4 x float> %2044, float %2046, i64 1, !dbg !19
  %2048 = select i1 %.not3, float %2042, float %1785, !dbg !15
  %2049 = insertelement <4 x float> %2047, float %2048, i64 2, !dbg !19
  %2050 = select i1 %.not3, float %2045, float %1786, !dbg !15
  %2051 = insertelement <4 x float> %2049, float %2050, i64 3, !dbg !19
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1686, !dbg !19
  %normalized73 = fmul <4 x float> %2051, %row.reciprocal72, !dbg !19
  %2052 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !20
  %2053 = extractelement <4 x half> %2052, i64 3, !dbg !20
  %2054 = extractelement <4 x half> %2052, i64 2, !dbg !20
  %2055 = extractelement <4 x half> %2052, i64 1, !dbg !20
  %2056 = extractelement <4 x half> %2052, i64 0, !dbg !20
  %2057 = bitcast i32 %1681 to float, !dbg !15
  %2058 = select i1 %.not3, float %1671, float %2057, !dbg !15
  %2059 = insertelement <4 x float> poison, float %2058, i64 0, !dbg !19
  %2060 = bitcast i32 %1684 to float, !dbg !15
  %2061 = select i1 %.not3, float %1672, float %2060, !dbg !15
  %2062 = insertelement <4 x float> %2059, float %2061, i64 1, !dbg !19
  %2063 = select i1 %.not3, float %2057, float %1677, !dbg !15
  %2064 = insertelement <4 x float> %2062, float %2063, i64 2, !dbg !19
  %2065 = select i1 %.not3, float %2060, float %1678, !dbg !15
  %2066 = insertelement <4 x float> %2064, float %2065, i64 3, !dbg !19
  %normalized74 = fmul <4 x float> %2066, %row.reciprocal72, !dbg !19
  %2067 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %2068 = extractelement <4 x half> %2067, i64 3, !dbg !20
  %2069 = extractelement <4 x half> %2067, i64 2, !dbg !20
  %2070 = extractelement <4 x half> %2067, i64 1, !dbg !20
  %2071 = extractelement <4 x half> %2067, i64 0, !dbg !20
  %2072 = add nsw i64 %10, %15, !dbg !21
  %2073 = shl nsw i64 %2072, 19, !dbg !21
  %2074 = and i64 %2073, 16252928, !dbg !21
  %address.thread389 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local390 = zext nneg i32 %address.thread389 to i64, !dbg !12
  %address.local391 = add nsw i64 %18, %address.local390, !dbg !21
  %address.local392 = lshr i64 %address.local390, 6, !dbg !12
  %address.local393 = shl nuw nsw i64 %address.local392, 4, !dbg !21
  %address.local394 = add nsw i64 %address.local391, %address.local393, !dbg !21
  %address.local395 = and i64 %address.local390, 240, !dbg !21
  %address.local396 = sub nsw i64 %address.local394, %address.local395, !dbg !21
  %address.local397 = add nsw i64 %address.local396, %19, !dbg !21
  %address.local398 = shl nuw nsw i64 %address.local397, 7, !dbg !21
  %address.local399 = add nuw nsw i64 %2074, %address.local398, !dbg !21
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local401 = lshr i64 %address.local390, 2, !dbg !12
  %address.local402 = and i64 %address.local401, 12, !dbg !12
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local402, !dbg !21
  store half %2071, ptr addrspace(1) %address.local403, align 2, !dbg !21
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local405 = or disjoint i64 %address.local402, 1, !dbg !21
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local404, i64 %address.local405, !dbg !21
  store half %2070, ptr addrspace(1) %address.local406, align 2, !dbg !21
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local408 = or disjoint i64 %address.local402, 2, !dbg !21
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local407, i64 %address.local408, !dbg !21
  store half %2069, ptr addrspace(1) %address.local409, align 2, !dbg !21
  %address.local410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local411 = or disjoint i64 %address.local402, 3, !dbg !21
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local410, i64 %address.local411, !dbg !21
  store half %2068, ptr addrspace(1) %address.local412, align 2, !dbg !21
  %address.local413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local414 = or disjoint i64 %address.local402, 16, !dbg !21
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local413, i64 %address.local414, !dbg !21
  store half %2056, ptr addrspace(1) %address.local415, align 2, !dbg !21
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local417 = or disjoint i64 %address.local402, 17, !dbg !21
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local416, i64 %address.local417, !dbg !21
  store half %2055, ptr addrspace(1) %address.local418, align 2, !dbg !21
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local420 = or disjoint i64 %address.local402, 18, !dbg !21
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local419, i64 %address.local420, !dbg !21
  store half %2054, ptr addrspace(1) %address.local421, align 2, !dbg !21
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local423 = or disjoint i64 %address.local402, 19, !dbg !21
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local422, i64 %address.local423, !dbg !21
  store half %2053, ptr addrspace(1) %address.local424, align 2, !dbg !21
  %address.local425 = shl nsw i64 %address.local397, 7, !dbg !21
  %address.local426 = add i64 %address.local425, 8192, !dbg !21
  %address.local427 = add nuw nsw i64 %2074, %address.local426, !dbg !21
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local402, !dbg !21
  store half %2041, ptr addrspace(1) %address.local429, align 2, !dbg !21
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local430, i64 %address.local405, !dbg !21
  store half %2040, ptr addrspace(1) %address.local431, align 2, !dbg !21
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local432, i64 %address.local408, !dbg !21
  store half %2039, ptr addrspace(1) %address.local433, align 2, !dbg !21
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local434, i64 %address.local411, !dbg !21
  store half %2038, ptr addrspace(1) %address.local435, align 2, !dbg !21
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local436, i64 %address.local414, !dbg !21
  store half %2026, ptr addrspace(1) %address.local437, align 2, !dbg !21
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local438, i64 %address.local417, !dbg !21
  store half %2025, ptr addrspace(1) %address.local439, align 2, !dbg !21
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local440, i64 %address.local420, !dbg !21
  store half %2024, ptr addrspace(1) %address.local441, align 2, !dbg !21
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local442, i64 %address.local423, !dbg !21
  store half %2023, ptr addrspace(1) %address.local443, align 2, !dbg !21
  br label %2075, !dbg !21

2075:                                             ; preds = %2011, %1560
  br i1 true, label %2076, label %2114, !dbg !21

2076:                                             ; preds = %2075
  %2077 = insertelement <4 x float> poison, float %1981, i64 0, !dbg !19
  %2078 = insertelement <4 x float> %2077, float %1987, i64 1, !dbg !19
  %2079 = insertelement <4 x float> %2078, float %1982, i64 2, !dbg !19
  %2080 = insertelement <4 x float> %2079, float %1988, i64 3, !dbg !19
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1904, !dbg !19
  %normalized76 = fmul <4 x float> %2080, %row.reciprocal75, !dbg !19
  %2081 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !20
  %2082 = extractelement <4 x half> %2081, i64 3, !dbg !20
  %2083 = extractelement <4 x half> %2081, i64 2, !dbg !20
  %2084 = extractelement <4 x half> %2081, i64 1, !dbg !20
  %2085 = extractelement <4 x half> %2081, i64 0, !dbg !20
  %2086 = insertelement <4 x float> poison, float %1872, i64 0, !dbg !19
  %2087 = insertelement <4 x float> %2086, float %1878, i64 1, !dbg !19
  %2088 = insertelement <4 x float> %2087, float %1873, i64 2, !dbg !19
  %2089 = insertelement <4 x float> %2088, float %1879, i64 3, !dbg !19
  %normalized77 = fmul <4 x float> %2089, %row.reciprocal75, !dbg !19
  %2090 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %2091 = extractelement <4 x half> %2090, i64 3, !dbg !20
  %2092 = extractelement <4 x half> %2090, i64 2, !dbg !20
  %2093 = extractelement <4 x half> %2090, i64 1, !dbg !20
  %2094 = extractelement <4 x half> %2090, i64 0, !dbg !20
  %2095 = insertelement <4 x float> poison, float %1763, i64 0, !dbg !19
  %2096 = insertelement <4 x float> %2095, float %1769, i64 1, !dbg !19
  %2097 = insertelement <4 x float> %2096, float %1764, i64 2, !dbg !19
  %2098 = insertelement <4 x float> %2097, float %1770, i64 3, !dbg !19
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1686, !dbg !19
  %normalized79 = fmul <4 x float> %2098, %row.reciprocal78, !dbg !19
  %2099 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !20
  %2100 = extractelement <4 x half> %2099, i64 3, !dbg !20
  %2101 = extractelement <4 x half> %2099, i64 2, !dbg !20
  %2102 = extractelement <4 x half> %2099, i64 1, !dbg !20
  %2103 = extractelement <4 x half> %2099, i64 0, !dbg !20
  %2104 = insertelement <4 x float> poison, float %1654, i64 0, !dbg !19
  %2105 = insertelement <4 x float> %2104, float %1660, i64 1, !dbg !19
  %2106 = insertelement <4 x float> %2105, float %1655, i64 2, !dbg !19
  %2107 = insertelement <4 x float> %2106, float %1661, i64 3, !dbg !19
  %normalized80 = fmul <4 x float> %2107, %row.reciprocal78, !dbg !19
  %2108 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !20
  %2109 = extractelement <4 x half> %2108, i64 3, !dbg !20
  %2110 = extractelement <4 x half> %2108, i64 2, !dbg !20
  %2111 = extractelement <4 x half> %2108, i64 1, !dbg !20
  %2112 = extractelement <4 x half> %2108, i64 0, !dbg !20
  %2113 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread444 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local445 = zext nneg i32 %address.thread444 to i64, !dbg !12
  %address.local446 = add nsw i64 %18, %address.local445, !dbg !21
  %address.local447 = lshr i64 %address.local445, 6, !dbg !12
  %address.local448 = shl nuw nsw i64 %address.local447, 4, !dbg !21
  %address.local449 = add nsw i64 %address.local446, %address.local448, !dbg !21
  %address.local450 = and i64 %address.local445, 240, !dbg !21
  %address.local451 = sub nsw i64 %address.local449, %address.local450, !dbg !21
  %address.local452 = add nsw i64 %address.local451, %19, !dbg !21
  %address.local453 = shl nuw nsw i64 %address.local452, 7, !dbg !21
  %address.local454 = add nuw nsw i64 %2113, %address.local453, !dbg !21
  %address.local455 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local456 = lshr i64 %address.local445, 2, !dbg !12
  %address.local457 = and i64 %address.local456, 12, !dbg !12
  %address.local458 = or disjoint i64 %address.local457, 32, !dbg !21
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local455, i64 %address.local458, !dbg !21
  store half %2112, ptr addrspace(1) %address.local459, align 2, !dbg !21
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local461 = or disjoint i64 %address.local457, 33, !dbg !21
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local461, !dbg !21
  store half %2111, ptr addrspace(1) %address.local462, align 2, !dbg !21
  %address.local463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local464 = or disjoint i64 %address.local457, 34, !dbg !21
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local463, i64 %address.local464, !dbg !21
  store half %2110, ptr addrspace(1) %address.local465, align 2, !dbg !21
  %address.local466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local467 = or disjoint i64 %address.local457, 35, !dbg !21
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local466, i64 %address.local467, !dbg !21
  store half %2109, ptr addrspace(1) %address.local468, align 2, !dbg !21
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local470 = or disjoint i64 %address.local457, 48, !dbg !21
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local469, i64 %address.local470, !dbg !21
  store half %2103, ptr addrspace(1) %address.local471, align 2, !dbg !21
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local473 = or disjoint i64 %address.local457, 49, !dbg !21
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local472, i64 %address.local473, !dbg !21
  store half %2102, ptr addrspace(1) %address.local474, align 2, !dbg !21
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local476 = or disjoint i64 %address.local457, 50, !dbg !21
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local475, i64 %address.local476, !dbg !21
  store half %2101, ptr addrspace(1) %address.local477, align 2, !dbg !21
  %address.local478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local479 = or i64 %address.local456, 51, !dbg !21
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local478, i64 %address.local479, !dbg !21
  store half %2100, ptr addrspace(1) %address.local480, align 2, !dbg !21
  %address.local481 = shl nsw i64 %address.local452, 7, !dbg !21
  %address.local482 = add i64 %address.local481, 8192, !dbg !21
  %address.local483 = add nuw nsw i64 %2113, %address.local482, !dbg !21
  %address.local484 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local484, i64 %address.local458, !dbg !21
  store half %2094, ptr addrspace(1) %address.local485, align 2, !dbg !21
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local486, i64 %address.local461, !dbg !21
  store half %2093, ptr addrspace(1) %address.local487, align 2, !dbg !21
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local488, i64 %address.local464, !dbg !21
  store half %2092, ptr addrspace(1) %address.local489, align 2, !dbg !21
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local490, i64 %address.local467, !dbg !21
  store half %2091, ptr addrspace(1) %address.local491, align 2, !dbg !21
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local492, i64 %address.local470, !dbg !21
  store half %2085, ptr addrspace(1) %address.local493, align 2, !dbg !21
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local494, i64 %address.local473, !dbg !21
  store half %2084, ptr addrspace(1) %address.local495, align 2, !dbg !21
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local496, i64 %address.local476, !dbg !21
  store half %2083, ptr addrspace(1) %address.local497, align 2, !dbg !21
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local498, i64 %address.local479, !dbg !21
  store half %2082, ptr addrspace(1) %address.local499, align 2, !dbg !21
  br label %2114, !dbg !21

2114:                                             ; preds = %2076, %2075
  br i1 true, label %2115, label %2153, !dbg !21

2115:                                             ; preds = %2114
  %2116 = insertelement <4 x float> poison, float %1953, i64 0, !dbg !19
  %2117 = insertelement <4 x float> %2116, float %1959, i64 1, !dbg !19
  %2118 = insertelement <4 x float> %2117, float %1954, i64 2, !dbg !19
  %2119 = insertelement <4 x float> %2118, float %1960, i64 3, !dbg !19
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1904, !dbg !19
  %normalized82 = fmul <4 x float> %2119, %row.reciprocal81, !dbg !19
  %2120 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !20
  %2121 = extractelement <4 x half> %2120, i64 3, !dbg !20
  %2122 = extractelement <4 x half> %2120, i64 2, !dbg !20
  %2123 = extractelement <4 x half> %2120, i64 1, !dbg !20
  %2124 = extractelement <4 x half> %2120, i64 0, !dbg !20
  %2125 = insertelement <4 x float> poison, float %1843, i64 0, !dbg !19
  %2126 = insertelement <4 x float> %2125, float %1849, i64 1, !dbg !19
  %2127 = insertelement <4 x float> %2126, float %1844, i64 2, !dbg !19
  %2128 = insertelement <4 x float> %2127, float %1850, i64 3, !dbg !19
  %normalized83 = fmul <4 x float> %2128, %row.reciprocal81, !dbg !19
  %2129 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !20
  %2130 = extractelement <4 x half> %2129, i64 3, !dbg !20
  %2131 = extractelement <4 x half> %2129, i64 2, !dbg !20
  %2132 = extractelement <4 x half> %2129, i64 1, !dbg !20
  %2133 = extractelement <4 x half> %2129, i64 0, !dbg !20
  %2134 = insertelement <4 x float> poison, float %1735, i64 0, !dbg !19
  %2135 = insertelement <4 x float> %2134, float %1741, i64 1, !dbg !19
  %2136 = insertelement <4 x float> %2135, float %1736, i64 2, !dbg !19
  %2137 = insertelement <4 x float> %2136, float %1742, i64 3, !dbg !19
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1686, !dbg !19
  %normalized85 = fmul <4 x float> %2137, %row.reciprocal84, !dbg !19
  %2138 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !20
  %2139 = extractelement <4 x half> %2138, i64 3, !dbg !20
  %2140 = extractelement <4 x half> %2138, i64 2, !dbg !20
  %2141 = extractelement <4 x half> %2138, i64 1, !dbg !20
  %2142 = extractelement <4 x half> %2138, i64 0, !dbg !20
  %2143 = insertelement <4 x float> poison, float %1625, i64 0, !dbg !19
  %2144 = insertelement <4 x float> %2143, float %1631, i64 1, !dbg !19
  %2145 = insertelement <4 x float> %2144, float %1626, i64 2, !dbg !19
  %2146 = insertelement <4 x float> %2145, float %1632, i64 3, !dbg !19
  %normalized86 = fmul <4 x float> %2146, %row.reciprocal84, !dbg !19
  %2147 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !20
  %2148 = extractelement <4 x half> %2147, i64 3, !dbg !20
  %2149 = extractelement <4 x half> %2147, i64 2, !dbg !20
  %2150 = extractelement <4 x half> %2147, i64 1, !dbg !20
  %2151 = extractelement <4 x half> %2147, i64 0, !dbg !20
  %2152 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread500 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local501 = zext nneg i32 %address.thread500 to i64, !dbg !12
  %address.local502 = add nsw i64 %18, %address.local501, !dbg !21
  %address.local503 = lshr i64 %address.local501, 6, !dbg !12
  %address.local504 = shl nuw nsw i64 %address.local503, 4, !dbg !21
  %address.local505 = add nsw i64 %address.local502, %address.local504, !dbg !21
  %address.local506 = and i64 %address.local501, 240, !dbg !21
  %address.local507 = sub nsw i64 %address.local505, %address.local506, !dbg !21
  %address.local508 = add nsw i64 %address.local507, %19, !dbg !21
  %address.local509 = shl nuw nsw i64 %address.local508, 7, !dbg !21
  %address.local510 = add nuw nsw i64 %2152, %address.local509, !dbg !21
  %address.local511 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local512 = lshr i64 %address.local501, 2, !dbg !12
  %address.local513 = and i64 %address.local512, 12, !dbg !12
  %address.local514 = or disjoint i64 %address.local513, 64, !dbg !21
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local511, i64 %address.local514, !dbg !21
  store half %2151, ptr addrspace(1) %address.local515, align 2, !dbg !21
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local517 = or disjoint i64 %address.local513, 65, !dbg !21
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local516, i64 %address.local517, !dbg !21
  store half %2150, ptr addrspace(1) %address.local518, align 2, !dbg !21
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local520 = or disjoint i64 %address.local513, 66, !dbg !21
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local519, i64 %address.local520, !dbg !21
  store half %2149, ptr addrspace(1) %address.local521, align 2, !dbg !21
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local523 = or disjoint i64 %address.local513, 67, !dbg !21
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local522, i64 %address.local523, !dbg !21
  store half %2148, ptr addrspace(1) %address.local524, align 2, !dbg !21
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local526 = or disjoint i64 %address.local513, 80, !dbg !21
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local525, i64 %address.local526, !dbg !21
  store half %2142, ptr addrspace(1) %address.local527, align 2, !dbg !21
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local529 = or disjoint i64 %address.local513, 81, !dbg !21
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local528, i64 %address.local529, !dbg !21
  store half %2141, ptr addrspace(1) %address.local530, align 2, !dbg !21
  %address.local531 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local532 = or disjoint i64 %address.local513, 82, !dbg !21
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local531, i64 %address.local532, !dbg !21
  store half %2140, ptr addrspace(1) %address.local533, align 2, !dbg !21
  %address.local534 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local535 = or disjoint i64 %address.local513, 83, !dbg !21
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local534, i64 %address.local535, !dbg !21
  store half %2139, ptr addrspace(1) %address.local536, align 2, !dbg !21
  %address.local537 = shl nsw i64 %address.local508, 7, !dbg !21
  %address.local538 = add i64 %address.local537, 8192, !dbg !21
  %address.local539 = add nuw nsw i64 %2152, %address.local538, !dbg !21
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local540, i64 %address.local514, !dbg !21
  store half %2133, ptr addrspace(1) %address.local541, align 2, !dbg !21
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local542, i64 %address.local517, !dbg !21
  store half %2132, ptr addrspace(1) %address.local543, align 2, !dbg !21
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local544, i64 %address.local520, !dbg !21
  store half %2131, ptr addrspace(1) %address.local545, align 2, !dbg !21
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local546, i64 %address.local523, !dbg !21
  store half %2130, ptr addrspace(1) %address.local547, align 2, !dbg !21
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local548, i64 %address.local526, !dbg !21
  store half %2124, ptr addrspace(1) %address.local549, align 2, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local550, i64 %address.local529, !dbg !21
  store half %2123, ptr addrspace(1) %address.local551, align 2, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local532, !dbg !21
  store half %2122, ptr addrspace(1) %address.local553, align 2, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local554, i64 %address.local535, !dbg !21
  store half %2121, ptr addrspace(1) %address.local555, align 2, !dbg !21
  br label %2153, !dbg !21

2153:                                             ; preds = %2115, %2114
  br i1 true, label %2154, label %2192, !dbg !21

2154:                                             ; preds = %2153
  %2155 = insertelement <4 x float> poison, float %1925, i64 0, !dbg !19
  %2156 = insertelement <4 x float> %2155, float %1931, i64 1, !dbg !19
  %2157 = insertelement <4 x float> %2156, float %1926, i64 2, !dbg !19
  %2158 = insertelement <4 x float> %2157, float %1932, i64 3, !dbg !19
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1904, !dbg !19
  %normalized88 = fmul <4 x float> %2158, %row.reciprocal87, !dbg !19
  %2159 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !20
  %2160 = extractelement <4 x half> %2159, i64 3, !dbg !20
  %2161 = extractelement <4 x half> %2159, i64 2, !dbg !20
  %2162 = extractelement <4 x half> %2159, i64 1, !dbg !20
  %2163 = extractelement <4 x half> %2159, i64 0, !dbg !20
  %2164 = insertelement <4 x float> poison, float %1814, i64 0, !dbg !19
  %2165 = insertelement <4 x float> %2164, float %1820, i64 1, !dbg !19
  %2166 = insertelement <4 x float> %2165, float %1815, i64 2, !dbg !19
  %2167 = insertelement <4 x float> %2166, float %1821, i64 3, !dbg !19
  %normalized89 = fmul <4 x float> %2167, %row.reciprocal87, !dbg !19
  %2168 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !20
  %2169 = extractelement <4 x half> %2168, i64 3, !dbg !20
  %2170 = extractelement <4 x half> %2168, i64 2, !dbg !20
  %2171 = extractelement <4 x half> %2168, i64 1, !dbg !20
  %2172 = extractelement <4 x half> %2168, i64 0, !dbg !20
  %2173 = insertelement <4 x float> poison, float %1707, i64 0, !dbg !19
  %2174 = insertelement <4 x float> %2173, float %1713, i64 1, !dbg !19
  %2175 = insertelement <4 x float> %2174, float %1708, i64 2, !dbg !19
  %2176 = insertelement <4 x float> %2175, float %1714, i64 3, !dbg !19
  %row.reciprocal90 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1686, !dbg !19
  %normalized91 = fmul <4 x float> %2176, %row.reciprocal90, !dbg !19
  %2177 = fptrunc <4 x float> %normalized91 to <4 x half>, !dbg !20
  %2178 = extractelement <4 x half> %2177, i64 3, !dbg !20
  %2179 = extractelement <4 x half> %2177, i64 2, !dbg !20
  %2180 = extractelement <4 x half> %2177, i64 1, !dbg !20
  %2181 = extractelement <4 x half> %2177, i64 0, !dbg !20
  %2182 = insertelement <4 x float> poison, float %1596, i64 0, !dbg !19
  %2183 = insertelement <4 x float> %2182, float %1602, i64 1, !dbg !19
  %2184 = insertelement <4 x float> %2183, float %1597, i64 2, !dbg !19
  %2185 = insertelement <4 x float> %2184, float %1603, i64 3, !dbg !19
  %normalized92 = fmul <4 x float> %2185, %row.reciprocal90, !dbg !19
  %2186 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !20
  %2187 = extractelement <4 x half> %2186, i64 3, !dbg !20
  %2188 = extractelement <4 x half> %2186, i64 2, !dbg !20
  %2189 = extractelement <4 x half> %2186, i64 1, !dbg !20
  %2190 = extractelement <4 x half> %2186, i64 0, !dbg !20
  %2191 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread556 = call i32 asm sideeffect "", "=v,0"(i32 %24) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local557 = zext nneg i32 %address.thread556 to i64, !dbg !12
  %address.local558 = add nsw i64 %18, %address.local557, !dbg !21
  %address.local559 = lshr i64 %address.local557, 6, !dbg !12
  %address.local560 = shl nuw nsw i64 %address.local559, 4, !dbg !21
  %address.local561 = add nsw i64 %address.local558, %address.local560, !dbg !21
  %address.local562 = and i64 %address.local557, 240, !dbg !21
  %address.local563 = sub nsw i64 %address.local561, %address.local562, !dbg !21
  %address.local564 = add nsw i64 %address.local563, %19, !dbg !21
  %address.local565 = shl nuw nsw i64 %address.local564, 7, !dbg !21
  %address.local566 = add nuw nsw i64 %2191, %address.local565, !dbg !21
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local568 = lshr i64 %address.local557, 2, !dbg !12
  %address.local569 = and i64 %address.local568, 12, !dbg !12
  %address.local570 = or disjoint i64 %address.local569, 96, !dbg !21
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local567, i64 %address.local570, !dbg !21
  store half %2190, ptr addrspace(1) %address.local571, align 2, !dbg !21
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local573 = or disjoint i64 %address.local569, 97, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local573, !dbg !21
  store half %2189, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local576 = or disjoint i64 %address.local569, 98, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local576, !dbg !21
  store half %2188, ptr addrspace(1) %address.local577, align 2, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local579 = or disjoint i64 %address.local569, 99, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local578, i64 %address.local579, !dbg !21
  store half %2187, ptr addrspace(1) %address.local580, align 2, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local582 = or disjoint i64 %address.local569, 112, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local582, !dbg !21
  store half %2181, ptr addrspace(1) %address.local583, align 2, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local585 = or disjoint i64 %address.local569, 113, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local584, i64 %address.local585, !dbg !21
  store half %2180, ptr addrspace(1) %address.local586, align 2, !dbg !21
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local588 = or disjoint i64 %address.local569, 114, !dbg !21
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local588, !dbg !21
  store half %2179, ptr addrspace(1) %address.local589, align 2, !dbg !21
  %address.local590 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local591 = or i64 %address.local568, 115, !dbg !21
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local590, i64 %address.local591, !dbg !21
  store half %2178, ptr addrspace(1) %address.local592, align 2, !dbg !21
  %address.local593 = shl nsw i64 %address.local564, 7, !dbg !21
  %address.local594 = add i64 %address.local593, 8192, !dbg !21
  %address.local595 = add nuw nsw i64 %2191, %address.local594, !dbg !21
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local596, i64 %address.local570, !dbg !21
  store half %2172, ptr addrspace(1) %address.local597, align 2, !dbg !21
  %address.local598 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local598, i64 %address.local573, !dbg !21
  store half %2171, ptr addrspace(1) %address.local599, align 2, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local576, !dbg !21
  store half %2170, ptr addrspace(1) %address.local601, align 2, !dbg !21
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local602, i64 %address.local579, !dbg !21
  store half %2169, ptr addrspace(1) %address.local603, align 2, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local604, i64 %address.local582, !dbg !21
  store half %2163, ptr addrspace(1) %address.local605, align 2, !dbg !21
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local606, i64 %address.local585, !dbg !21
  store half %2162, ptr addrspace(1) %address.local607, align 2, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local608, i64 %address.local588, !dbg !21
  store half %2161, ptr addrspace(1) %address.local609, align 2, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local610, i64 %address.local591, !dbg !21
  store half %2160, ptr addrspace(1) %address.local611, align 2, !dbg !21
  br label %2192, !dbg !21

2192:                                             ; preds = %2154, %2153
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

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #7

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!32}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/tune_attention_update_validation/run-20261020-164754-vwy1atdz/nohint-unrolled-32k")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!9 = !DILocation(line: 46, column: 17, scope: !4)
!11 = !DILocation(line: 66, column: 22, scope: !4)
!12 = !DILocation(line: 68, column: 24, scope: !4)
!13 = !DILocation(line: 73, column: 19, scope: !4)
!14 = !DILocation(line: 93, column: 23, scope: !4)
!15 = !DILocation(line: 56, column: 24, scope: !4)
!16 = !DILocation(line: 67, column: 22, scope: !4)
!17 = !DILocation(line: 69, column: 26, scope: !4)
!18 = !DILocation(line: 92, column: 20, scope: !4)
!19 = !DILocation(line: 97, column: 23, scope: !4)
!20 = !DILocation(line: 98, column: 19, scope: !4)
!21 = !DILocation(line: 99, column: 9, scope: !4)
!23 = !DILocation(line: 75, column: 24, scope: !4)
!24 = !DILocation(line: 76, column: 23, scope: !4)
!25 = !DILocation(line: 77, column: 26, scope: !4)
!26 = !DILocation(line: 84, column: 21, scope: !4)
!27 = !DILocation(line: 85, column: 22, scope: !4)
!28 = !DILocation(line: 86, column: 20, scope: !4)
!29 = !DILocation(line: 87, column: 16, scope: !4)
!30 = !DILocation(line: 94, column: 20, scope: !4)
!31 = !{i32 256, i32 1, i32 1}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 0, i32 256}
