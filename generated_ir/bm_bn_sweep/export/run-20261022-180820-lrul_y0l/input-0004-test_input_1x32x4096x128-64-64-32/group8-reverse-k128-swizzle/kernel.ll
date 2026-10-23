; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/bm_bn_sweep/export/run-20261022-180820-lrul_y0l/input-0004-test_input_1x32x4096x128-64-64-32/group8-reverse-k128-swizzle/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/bm_bn_sweep/export/run-20261022-180820-lrul_y0l/input-0004-test_input_1x32x4096x128-64-64-32/group8-reverse-k128-swizzle/input.mlir"

@shm_0 = addrspace(3) global [8192 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !31 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 6, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423480, !dbg !5
  %13 = and i64 %6, 7, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = lshr i64 %10, 3, !dbg !5
  %16 = sub nsw i64 63, %14, !dbg !5
  %17 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %18 = sext i32 %17 to i64, !dbg !5
  %19 = shl i64 %16, 6, !dbg !10
  %20 = icmp ugt i64 %10, 2047, !dbg !12
  %21 = add nsw i64 %14, -64, !dbg !12
  %22 = select i1 %20, i64 %21, i64 %16, !dbg !12
  %23 = sdiv i64 %22, 64, !dbg !12
  %24 = sext i1 %20 to i64, !dbg !12
  %25 = xor i64 %23, %24, !dbg !12
  %26 = add nsw i64 %15, %25, !dbg !12
  %27 = and i64 %26, 31, !dbg !12
  %28 = shl nsw i64 %16, 6, !dbg !12
  %29 = mul nsw i64 %25, -4096, !dbg !12
  %30 = add i64 %28, %29, !dbg !12
  %31 = shl nuw nsw i64 %27, 19, !dbg !12
  %32 = shl nsw i64 %30, 7, !dbg !12
  %33 = add i64 %31, %32, !dbg !12
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !33
  %35 = zext nneg i32 %34 to i64, !dbg !13
  %36 = lshr i64 %35, 6, !dbg !13
  %37 = or disjoint i64 %19, 63, !dbg !14
  %38 = add i64 %19, 64, !dbg !14
  %39 = icmp uge i64 %38, %37, !dbg !14
  %40 = sub nsw i64 64, %14, !dbg !8
  %41 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !12
  %42 = lshr i64 %35, 2, !dbg !17
  %43 = shl nuw nsw i64 %35, 2, !dbg !17
  %44 = mul nsw i64 %42, -16, !dbg !17
  %45 = add nsw i64 %43, %44, !dbg !17
  %46 = and i32 %34, 1, !dbg !17
  %index.compact114 = zext nneg i32 %46 to i64, !dbg !17
  %47 = or disjoint i64 %45, %index.compact114, !dbg !17
  %48 = add nuw nsw i64 %47, 1, !dbg !17
  %49 = sdiv i64 %48, 2, !dbg !17
  %50 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !12
  %51 = getelementptr i8, ptr addrspace(1) %50, i64 64, !dbg !12
  %52 = shl i64 %15, 7, !dbg !16
  %53 = and i64 %52, 3968, !dbg !16
  %54 = shl nsw i64 %18, 12, !dbg !16
  %55 = or disjoint i64 %53, %54, !dbg !16
  %56 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !12
  %57 = getelementptr i8, ptr addrspace(1) %56, i64 128, !dbg !12
  %58 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !12
  %59 = getelementptr i8, ptr addrspace(1) %58, i64 192, !dbg !12
  br label %60, !dbg !8

60:                                               ; preds = %420, %4
  %61 = phi i64 [ %671, %420 ], [ 0, %4 ], !dbg !8
  %62 = phi <4 x float> [ %554, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %63 = phi <4 x float> [ %562, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %64 = phi <8 x float> [ %667, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %65 = phi <1 x float> [ %518, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %66 = phi <4 x float> [ %594, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %67 = phi <4 x float> [ %602, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %68 = phi <8 x float> [ %668, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %69 = phi <4 x float> [ %634, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %70 = phi <4 x float> [ %642, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %71 = phi <8 x float> [ %669, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %72 = phi <4 x float> [ %658, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %73 = phi <4 x float> [ %666, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %74 = phi <8 x float> [ %670, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %75 = icmp slt i64 %61, %40, !dbg !8
  br i1 %75, label %76, label %672, !dbg !8

76:                                               ; preds = %60
  %77 = shl nsw i64 %61, 6, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !33, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local137 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local138 = and i64 %address.local137, 1536, !dbg !12
  %address.local139 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local138, !dbg !12
  %address.local140 = ptrtoint ptr addrspace(3) %address.local139 to i64, !dbg !12
  %address.local141 = trunc i64 %address.local140 to i32, !dbg !12
  %78 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !12
  %79 = zext i32 %78 to i64, !dbg !12
  %80 = inttoptr i64 %79 to ptr addrspace(3), !dbg !12
  %address.local142 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local143 = and i64 %address.local142, 8064, !dbg !12
  %address.local144 = lshr i64 %address.local, 3, !dbg !12
  %address.local145 = and i64 %address.local144, 24, !dbg !12
  %address.local146 = or disjoint i64 %address.local143, %address.local145, !dbg !12
  %address.local147 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local146, !dbg !12
  %81 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %82 = ptrtoint ptr addrspace(1) %address.local147 to i64, !dbg !12
  %83 = sub i64 %82, %81, !dbg !12
  %dma.buffer.offset679 = trunc i64 %83 to i32, !dbg !12
  %84 = trunc i64 %81 to i32, !dbg !12
  %85 = insertelement <4 x i32> poison, i32 %84, i64 0, !dbg !12
  %86 = lshr i64 %81, 32, !dbg !12
  %87 = trunc i64 %86 to i32, !dbg !12
  %88 = insertelement <4 x i32> %85, i32 %87, i64 1, !dbg !12
  %89 = insertelement <4 x i32> %88, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc680 = insertelement <4 x i32> %89, i32 131072, i64 3, !dbg !12
  %90 = ptrtoint ptr addrspace(3) %80 to i32, !dbg !12
  %dma.buffer.lds681 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %90), !dbg !12
  %91 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds681, i32 %dma.buffer.offset679, <4 x i32> %dma.buffer.rsrc680) #10, !dbg !12
  %92 = lshr i64 %61, 13, !dbg !16
  %93 = add nuw nsw i64 %92, %15, !dbg !16
  %94 = lshr i64 %61, 6, !dbg !16
  %95 = and i64 %94, 127, !dbg !16
  %96 = shl i64 %93, 19, !dbg !16
  %97 = and i64 %96, 16252928, !dbg !16
  %98 = shl i64 %94, 19, !dbg !16
  %99 = shl i64 %61, 13, !dbg !16
  %100 = sub i64 %99, %98, !dbg !16
  %101 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %97, !dbg !16
  %102 = getelementptr [2 x i8], ptr addrspace(1) %101, i64 %95, !dbg !16
  %103 = getelementptr [2 x i8], ptr addrspace(1) %102, i64 %100, !dbg !16
  %address.local148 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local138, !dbg !16
  %address.local149 = ptrtoint ptr addrspace(3) %address.local148 to i64, !dbg !16
  %address.local150 = trunc i64 %address.local149 to i32, !dbg !16
  %104 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local150), !dbg !16
  %105 = zext i32 %104 to i64, !dbg !16
  %106 = inttoptr i64 %105 to ptr addrspace(3), !dbg !16
  %address.local151 = lshr i64 %address.local, 3, !dbg !16
  %address.local152 = and i64 %address.local151, 24, !dbg !16
  %address.local153 = and i64 %address.local, 59, !dbg !16
  %address.local154 = lshr i64 %address.local, 2, !dbg !16
  %address.local155 = and i64 %address.local154, 4, !dbg !16
  %address.local156 = add nuw nsw i64 %address.local155, %address.local, !dbg !16
  %address.local157 = and i64 %address.local156, 4, !dbg !16
  %address.local158 = or disjoint i64 %address.local153, %address.local157, !dbg !16
  %address.local159 = shl nuw nsw i64 %address.local158, 7, !dbg !16
  %address.local160 = or disjoint i64 %address.local152, %address.local159, !dbg !16
  %address.local161 = getelementptr [2 x i8], ptr addrspace(1) %103, i64 %address.local160, !dbg !16
  %107 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %108 = ptrtoint ptr addrspace(1) %address.local161 to i64, !dbg !16
  %109 = sub i64 %108, %107, !dbg !16
  %dma.buffer.offset676 = trunc i64 %109 to i32, !dbg !16
  %110 = trunc i64 %107 to i32, !dbg !16
  %111 = insertelement <4 x i32> poison, i32 %110, i64 0, !dbg !16
  %112 = lshr i64 %107, 32, !dbg !16
  %113 = trunc i64 %112 to i32, !dbg !16
  %114 = insertelement <4 x i32> %111, i32 %113, i64 1, !dbg !16
  %115 = insertelement <4 x i32> %114, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc677 = insertelement <4 x i32> %115, i32 131072, i64 3, !dbg !16
  %116 = ptrtoint ptr addrspace(3) %106 to i32, !dbg !16
  %dma.buffer.lds678 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %116), !dbg !16
  %117 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds678, i32 %dma.buffer.offset676, <4 x i32> %dma.buffer.rsrc677) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local162 = lshr i32 %address.thread, 2, !dbg !13
  %address.local163 = and i32 %address.local162, 4, !dbg !13
  %address.local164 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local165 = and i32 %address.local164, 384, !dbg !13
  %address.local166 = or disjoint i32 %address.local163, %address.local165, !dbg !13
  %address.local167 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local168 = and i32 %address.local167, 120, !dbg !13
  %address.local169 = or disjoint i32 %address.local166, %address.local168, !dbg !13
  %address.local170 = shl nuw nsw i32 %address.thread, 4, !dbg !13
  %address.local171 = and i32 %address.local170, 512, !dbg !13
  %address.local172 = or disjoint i32 %address.local169, %address.local171, !dbg !13
  %address.local173 = zext nneg i32 %address.local172 to i64, !dbg !13
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local173, !dbg !13
  %118 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !13
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local174, i64 2048, !dbg !13
  %119 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !13
  %120 = fmul <4 x half> %118, splat (half 0xH3015), !dbg !13
  %121 = fmul <4 x half> %119, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local177 = mul nuw nsw i64 %address.local, 160, !dbg !17
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local177, !dbg !17
  %address.local179 = lshr i64 %address.local, 2, !dbg !17
  %address.local180 = mul nsw i64 %address.local179, -1280, !dbg !17
  %address.local181 = getelementptr i8, ptr addrspace(3) %address.local178, i64 %address.local180, !dbg !17
  %address.local182 = shl nuw nsw i64 %address.local, 1, !dbg !17
  %address.local183 = and i64 %address.local182, 24, !dbg !17
  %address.local184 = getelementptr [2 x i8], ptr addrspace(3) %address.local181, i64 %address.local183, !dbg !17
  %address.local185 = shl i64 %address.local, 62, !dbg !17
  %address.local186 = ashr exact i64 %address.local185, 57, !dbg !17
  %address.local187 = and i64 %address.local186, -64, !dbg !17
  %address.local188 = getelementptr [2 x i8], ptr addrspace(3) %address.local184, i64 %address.local187, !dbg !17
  %address.local189 = lshr i64 %address.local, 2, !dbg !13
  %address.local190 = and i64 %address.local189, 12, !dbg !13
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) %address.local188, i64 %address.local190, !dbg !17
  %address.local192 = and i64 %address.local, 63, !dbg !13
  %address.local193 = lshr i64 %address.local192, 5, !dbg !13
  %address.local194 = mul nuw nsw i64 %address.local193, 504, !dbg !13
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) %address.local191, i64 %address.local194, !dbg !17
  %122 = load <4 x half>, ptr addrspace(3) %address.local195, align 8, !dbg !17
  %lds.const.gep622 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2048, !dbg !17
  %123 = load <4 x half>, ptr addrspace(3) %lds.const.gep622, align 8, !dbg !17
  %124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %122, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local206 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local207 = and i32 %address.local206, 24, !dbg !17
  %address.local208 = shl nuw nsw i32 %address.thread, 8, !dbg !17
  %address.local209 = and i32 %address.local208, 768, !dbg !17
  %address.local210 = or disjoint i32 %address.local207, %address.local209, !dbg !17
  %address.local211 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local212 = and i32 %address.local211, 96, !dbg !17
  %address.local213 = or disjoint i32 %address.local210, %address.local212, !dbg !17
  %address.local214 = zext nneg i32 %address.local213 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %49) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local215 = lshr i32 %address.thread, 1, !dbg !17
  %address.local216 = and i32 %address.local215, 1, !dbg !17
  %address.local217 = zext nneg i32 %address.local216 to i64, !dbg !17
  %address.local218 = add nsw i64 %address.base, %address.local217, !dbg !17
  %address.local219 = shl i64 %address.local218, 6, !dbg !17
  %address.local220 = sub i64 %address.local214, %address.local219, !dbg !17
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local220, !dbg !17
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) %address.local221, i64 %address.local190, !dbg !17
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local222, i64 %address.local194, !dbg !17
  %address.local224 = getelementptr i8, ptr addrspace(3) %address.local223, i64 64, !dbg !17
  %125 = load <4 x half>, ptr addrspace(3) %address.local224, align 8, !dbg !17
  %126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %123, <4 x float> %124) #9, !dbg !17
  %lds.const.gep623 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2048, !dbg !17
  %127 = load <4 x half>, ptr addrspace(3) %lds.const.gep623, align 8, !dbg !17
  %lds.const.gep624 = getelementptr i8, ptr addrspace(3) %address.local195, i64 128, !dbg !17
  %128 = load <4 x half>, ptr addrspace(3) %lds.const.gep624, align 8, !dbg !17
  %129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %125, <4 x float> zeroinitializer) #9, !dbg !17
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %127, <4 x float> %129) #9, !dbg !17
  %lds.const.gep625 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2176, !dbg !17
  %131 = load <4 x half>, ptr addrspace(3) %lds.const.gep625, align 8, !dbg !17
  %132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %128, <4 x float> zeroinitializer) #9, !dbg !17
  %133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %131, <4 x float> %132) #9, !dbg !17
  %lds.const.gep626 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !17
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep626, align 8, !dbg !17
  %lds.const.gep627 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2176, !dbg !17
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep627, align 8, !dbg !17
  %136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %134, <4 x float> zeroinitializer) #9, !dbg !17
  %137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %135, <4 x float> %136) #9, !dbg !17
  %138 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !12
  %139 = zext i32 %138 to i64, !dbg !12
  %140 = inttoptr i64 %139 to ptr addrspace(3), !dbg !12
  %address.local365 = getelementptr [2 x i8], ptr addrspace(1) %51, i64 %address.local146, !dbg !12
  %141 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %142 = ptrtoint ptr addrspace(1) %address.local365 to i64, !dbg !12
  %143 = sub i64 %142, %141, !dbg !12
  %dma.buffer.offset673 = trunc i64 %143 to i32, !dbg !12
  %144 = trunc i64 %141 to i32, !dbg !12
  %145 = insertelement <4 x i32> poison, i32 %144, i64 0, !dbg !12
  %146 = lshr i64 %141, 32, !dbg !12
  %147 = trunc i64 %146 to i32, !dbg !12
  %148 = insertelement <4 x i32> %145, i32 %147, i64 1, !dbg !12
  %149 = insertelement <4 x i32> %148, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc674 = insertelement <4 x i32> %149, i32 131072, i64 3, !dbg !12
  %150 = ptrtoint ptr addrspace(3) %140 to i32, !dbg !12
  %dma.buffer.lds675 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %150), !dbg !12
  %151 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds675, i32 %dma.buffer.offset673, <4 x i32> %dma.buffer.rsrc674) #10, !dbg !12
  %152 = add nuw i64 %61, 2048, !dbg !16
  %153 = icmp slt i64 %152, 0, !dbg !16
  %154 = sub nuw i64 -2049, %61, !dbg !16
  %155 = select i1 %153, i64 %154, i64 %152, !dbg !16
  %156 = sdiv i64 %155, 8192, !dbg !16
  %.lobit19 = ashr i64 %152, 63, !dbg !16
  %157 = xor i64 %156, %.lobit19, !dbg !16
  %158 = add nsw i64 %157, %15, !dbg !16
  %159 = add nsw i64 %94, %55, !dbg !16
  %160 = add nsw i64 %159, 32, !dbg !16
  %161 = icmp slt i64 %159, -32, !dbg !16
  %162 = sub nsw i64 -33, %159, !dbg !16
  %163 = select i1 %161, i64 %162, i64 %160, !dbg !16
  %164 = sdiv i64 %163, 128, !dbg !16
  %165 = sext i1 %161 to i64, !dbg !16
  %166 = xor i64 %164, %165, !dbg !16
  %167 = shl nsw i64 %166, 7, !dbg !16
  %168 = sub i64 %55, %167, !dbg !16
  %169 = shl i64 %158, 19, !dbg !16
  %170 = and i64 %169, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %171 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %170, !dbg !16
  %172 = getelementptr [2 x i8], ptr addrspace(1) %171, i64 %94, !dbg !16
  %173 = getelementptr [2 x i8], ptr addrspace(1) %172, i64 %168, !dbg !16
  %174 = getelementptr i8, ptr addrspace(1) %173, i64 64, !dbg !16
  %175 = getelementptr [2 x i8], ptr addrspace(1) %174, i64 %100, !dbg !16
  %176 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local150), !dbg !16
  %177 = zext i32 %176 to i64, !dbg !16
  %178 = inttoptr i64 %177 to ptr addrspace(3), !dbg !16
  %address.local366 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %address.local160, !dbg !16
  %179 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %180 = ptrtoint ptr addrspace(1) %address.local366 to i64, !dbg !16
  %181 = sub i64 %180, %179, !dbg !16
  %dma.buffer.offset670 = trunc i64 %181 to i32, !dbg !16
  %182 = trunc i64 %179 to i32, !dbg !16
  %183 = insertelement <4 x i32> poison, i32 %182, i64 0, !dbg !16
  %184 = lshr i64 %179, 32, !dbg !16
  %185 = trunc i64 %184 to i32, !dbg !16
  %186 = insertelement <4 x i32> %183, i32 %185, i64 1, !dbg !16
  %187 = insertelement <4 x i32> %186, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc671 = insertelement <4 x i32> %187, i32 131072, i64 3, !dbg !16
  %188 = ptrtoint ptr addrspace(3) %178 to i32, !dbg !16
  %dma.buffer.lds672 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %188), !dbg !16
  %189 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds672, i32 %dma.buffer.offset670, <4 x i32> %dma.buffer.rsrc671) #10, !dbg !16
  %190 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !13
  %lds.const.gep628 = getelementptr i8, ptr addrspace(3) %address.local174, i64 2048, !dbg !13
  %191 = load <4 x half>, ptr addrspace(3) %lds.const.gep628, align 8, !dbg !13
  %192 = fmul <4 x half> %190, splat (half 0xH3015), !dbg !13
  %193 = fmul <4 x half> %191, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %194 = load <4 x half>, ptr addrspace(3) %address.local195, align 8, !dbg !17
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %194, <4 x float> %126) #9, !dbg !17
  %lds.const.gep629 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2048, !dbg !17
  %196 = load <4 x half>, ptr addrspace(3) %lds.const.gep629, align 8, !dbg !17
  %197 = load <4 x half>, ptr addrspace(3) %address.local224, align 8, !dbg !17
  %198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %196, <4 x float> %195) #9, !dbg !17
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %197, <4 x float> %130) #9, !dbg !17
  %lds.const.gep630 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2048, !dbg !17
  %200 = load <4 x half>, ptr addrspace(3) %lds.const.gep630, align 8, !dbg !17
  %lds.const.gep631 = getelementptr i8, ptr addrspace(3) %address.local195, i64 128, !dbg !17
  %201 = load <4 x half>, ptr addrspace(3) %lds.const.gep631, align 8, !dbg !17
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %200, <4 x float> %199) #9, !dbg !17
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %201, <4 x float> %133) #9, !dbg !17
  %lds.const.gep632 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2176, !dbg !17
  %204 = load <4 x half>, ptr addrspace(3) %lds.const.gep632, align 8, !dbg !17
  %lds.const.gep633 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !17
  %205 = load <4 x half>, ptr addrspace(3) %lds.const.gep633, align 8, !dbg !17
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %204, <4 x float> %203) #9, !dbg !17
  %lds.const.gep634 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2176, !dbg !17
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep634, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %205, <4 x float> %137) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %207, <4 x float> %208) #9, !dbg !17
  %210 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !12
  %211 = zext i32 %210 to i64, !dbg !12
  %212 = inttoptr i64 %211 to ptr addrspace(3), !dbg !12
  %address.local367 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local146, !dbg !12
  %213 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %214 = ptrtoint ptr addrspace(1) %address.local367 to i64, !dbg !12
  %215 = sub i64 %214, %213, !dbg !12
  %dma.buffer.offset667 = trunc i64 %215 to i32, !dbg !12
  %216 = trunc i64 %213 to i32, !dbg !12
  %217 = insertelement <4 x i32> poison, i32 %216, i64 0, !dbg !12
  %218 = lshr i64 %213, 32, !dbg !12
  %219 = trunc i64 %218 to i32, !dbg !12
  %220 = insertelement <4 x i32> %217, i32 %219, i64 1, !dbg !12
  %221 = insertelement <4 x i32> %220, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc668 = insertelement <4 x i32> %221, i32 131072, i64 3, !dbg !12
  %222 = ptrtoint ptr addrspace(3) %212 to i32, !dbg !12
  %dma.buffer.lds669 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %222), !dbg !12
  %223 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds669, i32 %dma.buffer.offset667, <4 x i32> %dma.buffer.rsrc668) #10, !dbg !12
  %224 = add nuw i64 %61, 4096, !dbg !16
  %225 = icmp slt i64 %224, 0, !dbg !16
  %226 = sub nuw i64 -4097, %61, !dbg !16
  %227 = select i1 %225, i64 %226, i64 %224, !dbg !16
  %228 = sdiv i64 %227, 8192, !dbg !16
  %.lobit20 = ashr i64 %224, 63, !dbg !16
  %229 = xor i64 %228, %.lobit20, !dbg !16
  %230 = add nsw i64 %229, %15, !dbg !16
  %231 = add nsw i64 %159, 64, !dbg !16
  %232 = icmp slt i64 %159, -64, !dbg !16
  %233 = sub nsw i64 -65, %159, !dbg !16
  %234 = select i1 %232, i64 %233, i64 %231, !dbg !16
  %235 = sdiv i64 %234, 128, !dbg !16
  %236 = sext i1 %232 to i64, !dbg !16
  %237 = xor i64 %235, %236, !dbg !16
  %238 = shl nsw i64 %237, 7, !dbg !16
  %239 = sub i64 %55, %238, !dbg !16
  %240 = shl i64 %230, 19, !dbg !16
  %241 = and i64 %240, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %242 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %241, !dbg !16
  %243 = getelementptr [2 x i8], ptr addrspace(1) %242, i64 %94, !dbg !16
  %244 = getelementptr [2 x i8], ptr addrspace(1) %243, i64 %239, !dbg !16
  %245 = getelementptr i8, ptr addrspace(1) %244, i64 128, !dbg !16
  %246 = getelementptr [2 x i8], ptr addrspace(1) %245, i64 %100, !dbg !16
  %247 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local150), !dbg !16
  %248 = zext i32 %247 to i64, !dbg !16
  %249 = inttoptr i64 %248 to ptr addrspace(3), !dbg !16
  %address.local368 = getelementptr [2 x i8], ptr addrspace(1) %246, i64 %address.local160, !dbg !16
  %250 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %251 = ptrtoint ptr addrspace(1) %address.local368 to i64, !dbg !16
  %252 = sub i64 %251, %250, !dbg !16
  %dma.buffer.offset664 = trunc i64 %252 to i32, !dbg !16
  %253 = trunc i64 %250 to i32, !dbg !16
  %254 = insertelement <4 x i32> poison, i32 %253, i64 0, !dbg !16
  %255 = lshr i64 %250, 32, !dbg !16
  %256 = trunc i64 %255 to i32, !dbg !16
  %257 = insertelement <4 x i32> %254, i32 %256, i64 1, !dbg !16
  %258 = insertelement <4 x i32> %257, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc665 = insertelement <4 x i32> %258, i32 131072, i64 3, !dbg !16
  %259 = ptrtoint ptr addrspace(3) %249 to i32, !dbg !16
  %dma.buffer.lds666 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %259), !dbg !16
  %260 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds666, i32 %dma.buffer.offset664, <4 x i32> %dma.buffer.rsrc665) #10, !dbg !16
  %261 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !13
  %lds.const.gep635 = getelementptr i8, ptr addrspace(3) %address.local174, i64 2048, !dbg !13
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep635, align 8, !dbg !13
  %263 = fmul <4 x half> %261, splat (half 0xH3015), !dbg !13
  %264 = fmul <4 x half> %262, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %265 = load <4 x half>, ptr addrspace(3) %address.local195, align 8, !dbg !17
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %263, <4 x half> %265, <4 x float> %198) #9, !dbg !17
  %lds.const.gep636 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2048, !dbg !17
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep636, align 8, !dbg !17
  %268 = load <4 x half>, ptr addrspace(3) %address.local224, align 8, !dbg !17
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %264, <4 x half> %267, <4 x float> %266) #9, !dbg !17
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %263, <4 x half> %268, <4 x float> %202) #9, !dbg !17
  %lds.const.gep637 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2048, !dbg !17
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep637, align 8, !dbg !17
  %lds.const.gep638 = getelementptr i8, ptr addrspace(3) %address.local195, i64 128, !dbg !17
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep638, align 8, !dbg !17
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %264, <4 x half> %271, <4 x float> %270) #9, !dbg !17
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %263, <4 x half> %272, <4 x float> %206) #9, !dbg !17
  %lds.const.gep639 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2176, !dbg !17
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep639, align 8, !dbg !17
  %lds.const.gep640 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !17
  %276 = load <4 x half>, ptr addrspace(3) %lds.const.gep640, align 8, !dbg !17
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %264, <4 x half> %275, <4 x float> %274) #9, !dbg !17
  %lds.const.gep641 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2176, !dbg !17
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep641, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %263, <4 x half> %276, <4 x float> %209) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %264, <4 x half> %278, <4 x float> %279) #9, !dbg !17
  %281 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !12
  %282 = zext i32 %281 to i64, !dbg !12
  %283 = inttoptr i64 %282 to ptr addrspace(3), !dbg !12
  %address.local369 = getelementptr [2 x i8], ptr addrspace(1) %59, i64 %address.local146, !dbg !12
  %284 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %285 = ptrtoint ptr addrspace(1) %address.local369 to i64, !dbg !12
  %286 = sub i64 %285, %284, !dbg !12
  %dma.buffer.offset661 = trunc i64 %286 to i32, !dbg !12
  %287 = trunc i64 %284 to i32, !dbg !12
  %288 = insertelement <4 x i32> poison, i32 %287, i64 0, !dbg !12
  %289 = lshr i64 %284, 32, !dbg !12
  %290 = trunc i64 %289 to i32, !dbg !12
  %291 = insertelement <4 x i32> %288, i32 %290, i64 1, !dbg !12
  %292 = insertelement <4 x i32> %291, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc662 = insertelement <4 x i32> %292, i32 131072, i64 3, !dbg !12
  %293 = ptrtoint ptr addrspace(3) %283 to i32, !dbg !12
  %dma.buffer.lds663 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %293), !dbg !12
  %294 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds663, i32 %dma.buffer.offset661, <4 x i32> %dma.buffer.rsrc662) #10, !dbg !12
  %295 = add nuw i64 %61, 6144, !dbg !16
  %296 = icmp slt i64 %295, 0, !dbg !16
  %297 = sub nuw i64 -6145, %61, !dbg !16
  %298 = select i1 %296, i64 %297, i64 %295, !dbg !16
  %299 = sdiv i64 %298, 8192, !dbg !16
  %.lobit21 = ashr i64 %295, 63, !dbg !16
  %300 = xor i64 %299, %.lobit21, !dbg !16
  %301 = add nsw i64 %300, %15, !dbg !16
  %302 = add nsw i64 %159, 96, !dbg !16
  %303 = icmp slt i64 %159, -96, !dbg !16
  %304 = sub nsw i64 -97, %159, !dbg !16
  %305 = select i1 %303, i64 %304, i64 %302, !dbg !16
  %306 = sdiv i64 %305, 128, !dbg !16
  %307 = sext i1 %303 to i64, !dbg !16
  %308 = xor i64 %306, %307, !dbg !16
  %309 = shl nsw i64 %308, 7, !dbg !16
  %310 = sub i64 %55, %309, !dbg !16
  %311 = shl i64 %301, 19, !dbg !16
  %312 = and i64 %311, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %313 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %312, !dbg !16
  %314 = getelementptr [2 x i8], ptr addrspace(1) %313, i64 %94, !dbg !16
  %315 = getelementptr [2 x i8], ptr addrspace(1) %314, i64 %310, !dbg !16
  %316 = getelementptr i8, ptr addrspace(1) %315, i64 192, !dbg !16
  %317 = getelementptr [2 x i8], ptr addrspace(1) %316, i64 %100, !dbg !16
  %318 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local150), !dbg !16
  %319 = zext i32 %318 to i64, !dbg !16
  %320 = inttoptr i64 %319 to ptr addrspace(3), !dbg !16
  %address.local370 = getelementptr [2 x i8], ptr addrspace(1) %317, i64 %address.local160, !dbg !16
  %321 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %322 = ptrtoint ptr addrspace(1) %address.local370 to i64, !dbg !16
  %323 = sub i64 %322, %321, !dbg !16
  %dma.buffer.offset658 = trunc i64 %323 to i32, !dbg !16
  %324 = trunc i64 %321 to i32, !dbg !16
  %325 = insertelement <4 x i32> poison, i32 %324, i64 0, !dbg !16
  %326 = lshr i64 %321, 32, !dbg !16
  %327 = trunc i64 %326 to i32, !dbg !16
  %328 = insertelement <4 x i32> %325, i32 %327, i64 1, !dbg !16
  %329 = insertelement <4 x i32> %328, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc659 = insertelement <4 x i32> %329, i32 131072, i64 3, !dbg !16
  %330 = ptrtoint ptr addrspace(3) %320 to i32, !dbg !16
  %dma.buffer.lds660 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %330), !dbg !16
  %331 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds660, i32 %dma.buffer.offset658, <4 x i32> %dma.buffer.rsrc659) #10, !dbg !16
  %332 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !13
  %lds.const.gep642 = getelementptr i8, ptr addrspace(3) %address.local174, i64 2048, !dbg !13
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep642, align 8, !dbg !13
  %334 = fmul <4 x half> %332, splat (half 0xH3015), !dbg !13
  %335 = fmul <4 x half> %333, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %336 = add nuw nsw i64 %94, %15, !dbg !18
  %337 = shl i64 %336, 19, !dbg !18
  %338 = and i64 %337, 16252928, !dbg !18
  %339 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %338, !dbg !18
  %340 = getelementptr [2 x i8], ptr addrspace(1) %339, i64 %100, !dbg !18
  %341 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !18
  %342 = zext i32 %341 to i64, !dbg !18
  %343 = inttoptr i64 %342 to ptr addrspace(3), !dbg !18
  %address.local371 = shl nuw nsw i64 %address.local, 5, !dbg !18
  %address.local372 = and i64 %address.local371, 8064, !dbg !18
  %address.local373 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local374 = and i64 %address.local373, 24, !dbg !18
  %address.local375 = or disjoint i64 %address.local372, %address.local374, !dbg !18
  %address.local376 = getelementptr [2 x i8], ptr addrspace(1) %340, i64 %address.local375, !dbg !18
  %344 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %345 = ptrtoint ptr addrspace(1) %address.local376 to i64, !dbg !18
  %346 = sub i64 %345, %344, !dbg !18
  %dma.buffer.offset655 = trunc i64 %346 to i32, !dbg !18
  %347 = trunc i64 %344 to i32, !dbg !18
  %348 = insertelement <4 x i32> poison, i32 %347, i64 0, !dbg !18
  %349 = lshr i64 %344, 32, !dbg !18
  %350 = trunc i64 %349 to i32, !dbg !18
  %351 = insertelement <4 x i32> %348, i32 %350, i64 1, !dbg !18
  %352 = insertelement <4 x i32> %351, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc656 = insertelement <4 x i32> %352, i32 131072, i64 3, !dbg !18
  %353 = ptrtoint ptr addrspace(3) %343 to i32, !dbg !18
  %dma.buffer.lds657 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %353), !dbg !18
  %354 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds657, i32 %dma.buffer.offset655, <4 x i32> %dma.buffer.rsrc656) #10, !dbg !18
  %355 = load <4 x half>, ptr addrspace(3) %address.local195, align 8, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %355, <4 x float> %269) #9, !dbg !17
  %lds.const.gep643 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2048, !dbg !17
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep643, align 8, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %address.local224, align 8, !dbg !17
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %357, <4 x float> %356) #9, !dbg !17
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %358, <4 x float> %273) #9, !dbg !17
  %lds.const.gep644 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2048, !dbg !17
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep644, align 8, !dbg !17
  %lds.const.gep645 = getelementptr i8, ptr addrspace(3) %address.local195, i64 128, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep645, align 8, !dbg !17
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %361, <4 x float> %360) #9, !dbg !17
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %362, <4 x float> %277) #9, !dbg !17
  %lds.const.gep646 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2176, !dbg !17
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep646, align 8, !dbg !17
  %lds.const.gep647 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep647, align 8, !dbg !17
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %365, <4 x float> %364) #9, !dbg !17
  %lds.const.gep648 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2176, !dbg !17
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep648, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %366, <4 x float> %280) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %370 = or disjoint i64 %77, 63, !dbg !14
  %371 = icmp uge i64 %19, %370, !dbg !14
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %368, <4 x float> %369) #9, !dbg !17
  %373 = and i1 %39, %371, !dbg !14
  br i1 %373, label %374, label %375, !dbg !14

374:                                              ; preds = %76
  br label %418, !dbg !14

375:                                              ; preds = %76
  %376 = add i64 %19, %35, !dbg !14
  %377 = shl nuw nsw i64 %36, 4, !dbg !14
  %378 = add i64 %376, %377, !dbg !14
  %379 = and i32 %34, 240, !dbg !14
  %index.compact = zext nneg i32 %379 to i64, !dbg !14
  %380 = sub i64 %378, %index.compact, !dbg !14
  %381 = add i64 %380, 1, !dbg !23
  %382 = and i64 %35, 48, !dbg !14
  %383 = or disjoint i64 %77, %382, !dbg !14
  %384 = lshr i64 %381, 32, !dbg !24
  %compare.high = trunc i64 %384 to i32, !dbg !24
  %385 = lshr i64 %383, 32, !dbg !24
  %compare.high562 = trunc i64 %385 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high562, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high562, !dbg !24
  %compare.low.base = trunc i64 %381 to i32, !dbg !24
  %compare.low.base563 = trunc i64 %383 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base563, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %386 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low564 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result565 = select i1 %compare.high.equal, i1 %compare.low564, i1 %compare.high.order, !dbg !24
  %387 = select i1 %compare.result565, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base566 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local567 = or i32 %compare.local.base566, 2, !dbg !24
  %compare.low568 = icmp ugt i32 %compare.low.base, %compare.local567, !dbg !24
  %compare.result569 = select i1 %compare.high.equal, i1 %compare.low568, i1 %compare.high.order, !dbg !24
  %388 = select i1 %compare.result569, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base570 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local571 = or i32 %compare.local.base570, 3, !dbg !24
  %compare.low572 = icmp ugt i32 %compare.low.base, %compare.local571, !dbg !24
  %compare.result573 = select i1 %compare.high.equal, i1 %compare.low572, i1 %compare.high.order, !dbg !24
  %389 = select i1 %compare.result573, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base574 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local575 = or i32 %compare.local.base574, 4, !dbg !24
  %compare.low576 = icmp ugt i32 %compare.low.base, %compare.local575, !dbg !24
  %compare.result577 = select i1 %compare.high.equal, i1 %compare.low576, i1 %compare.high.order, !dbg !24
  %390 = select i1 %compare.result577, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base578 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local579 = or i32 %compare.local.base578, 5, !dbg !24
  %compare.low580 = icmp ugt i32 %compare.low.base, %compare.local579, !dbg !24
  %compare.result581 = select i1 %compare.high.equal, i1 %compare.low580, i1 %compare.high.order, !dbg !24
  %391 = select i1 %compare.result581, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base582 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local583 = or i32 %compare.local.base582, 6, !dbg !24
  %compare.low584 = icmp ugt i32 %compare.low.base, %compare.local583, !dbg !24
  %compare.result585 = select i1 %compare.high.equal, i1 %compare.low584, i1 %compare.high.order, !dbg !24
  %392 = select i1 %compare.result585, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base586 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local587 = or i32 %compare.local.base586, 7, !dbg !24
  %compare.low588 = icmp ugt i32 %compare.low.base, %compare.local587, !dbg !24
  %compare.result589 = select i1 %compare.high.equal, i1 %compare.low588, i1 %compare.high.order, !dbg !24
  %393 = select i1 %compare.result589, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base590 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local591 = or i32 %compare.local.base590, 8, !dbg !24
  %compare.low592 = icmp ugt i32 %compare.low.base, %compare.local591, !dbg !24
  %compare.result593 = select i1 %compare.high.equal, i1 %compare.low592, i1 %compare.high.order, !dbg !24
  %394 = select i1 %compare.result593, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base594 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local595 = or i32 %compare.local.base594, 9, !dbg !24
  %compare.low596 = icmp ugt i32 %compare.low.base, %compare.local595, !dbg !24
  %compare.result597 = select i1 %compare.high.equal, i1 %compare.low596, i1 %compare.high.order, !dbg !24
  %395 = select i1 %compare.result597, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base598 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local599 = or i32 %compare.local.base598, 10, !dbg !24
  %compare.low600 = icmp ugt i32 %compare.low.base, %compare.local599, !dbg !24
  %compare.result601 = select i1 %compare.high.equal, i1 %compare.low600, i1 %compare.high.order, !dbg !24
  %396 = select i1 %compare.result601, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base602 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local603 = or i32 %compare.local.base602, 11, !dbg !24
  %compare.low604 = icmp ugt i32 %compare.low.base, %compare.local603, !dbg !24
  %compare.result605 = select i1 %compare.high.equal, i1 %compare.low604, i1 %compare.high.order, !dbg !24
  %397 = select i1 %compare.result605, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base606 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local607 = or i32 %compare.local.base606, 12, !dbg !24
  %compare.low608 = icmp ugt i32 %compare.low.base, %compare.local607, !dbg !24
  %compare.result609 = select i1 %compare.high.equal, i1 %compare.low608, i1 %compare.high.order, !dbg !24
  %398 = select i1 %compare.result609, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base610 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local611 = or i32 %compare.local.base610, 13, !dbg !24
  %compare.low612 = icmp ugt i32 %compare.low.base, %compare.local611, !dbg !24
  %compare.result613 = select i1 %compare.high.equal, i1 %compare.low612, i1 %compare.high.order, !dbg !24
  %399 = select i1 %compare.result613, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base614 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local615 = or i32 %compare.local.base614, 14, !dbg !24
  %compare.low616 = icmp ugt i32 %compare.low.base, %compare.local615, !dbg !24
  %compare.result617 = select i1 %compare.high.equal, i1 %compare.low616, i1 %compare.high.order, !dbg !24
  %400 = select i1 %compare.result617, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base618 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local619 = or i32 %compare.local.base618, 15, !dbg !24
  %compare.low620 = icmp ugt i32 %compare.low.base, %compare.local619, !dbg !24
  %compare.result621 = select i1 %compare.high.equal, i1 %compare.low620, i1 %compare.high.order, !dbg !24
  %401 = select i1 %compare.result621, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %402 = insertelement <16 x float> poison, float %386, i64 0, !dbg !14
  %403 = insertelement <16 x float> %402, float %387, i64 1, !dbg !14
  %404 = insertelement <16 x float> %403, float %388, i64 2, !dbg !14
  %405 = insertelement <16 x float> %404, float %389, i64 3, !dbg !14
  %406 = insertelement <16 x float> %405, float %390, i64 4, !dbg !14
  %407 = insertelement <16 x float> %406, float %391, i64 5, !dbg !14
  %408 = insertelement <16 x float> %407, float %392, i64 6, !dbg !14
  %409 = insertelement <16 x float> %408, float %393, i64 7, !dbg !14
  %410 = insertelement <16 x float> %409, float %394, i64 8, !dbg !14
  %411 = insertelement <16 x float> %410, float %395, i64 9, !dbg !14
  %412 = insertelement <16 x float> %411, float %396, i64 10, !dbg !14
  %413 = insertelement <16 x float> %412, float %397, i64 11, !dbg !14
  %414 = insertelement <16 x float> %413, float %398, i64 12, !dbg !14
  %415 = insertelement <16 x float> %414, float %399, i64 13, !dbg !14
  %416 = insertelement <16 x float> %415, float %400, i64 14, !dbg !14
  %417 = insertelement <16 x float> %416, float %401, i64 15, !dbg !14
  br label %418, !dbg !14

418:                                              ; preds = %374, %375
  %419 = phi <16 x float> [ %417, %375 ], [ zeroinitializer, %374 ], !dbg !14
  br label %420, !dbg !14

420:                                              ; preds = %418
  %421 = shufflevector <16 x float> %419, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %422 = fadd <4 x float> %359, %421, !dbg !26
  %423 = extractelement <4 x float> %422, i64 0, !dbg !27
  %424 = call float @llvm.amdgcn.exp2.f32(float %423), !dbg !27
  %425 = insertelement <4 x float> poison, float %424, i64 0, !dbg !29
  %426 = extractelement <4 x float> %422, i64 1, !dbg !27
  %427 = call float @llvm.amdgcn.exp2.f32(float %426), !dbg !27
  %428 = insertelement <4 x float> %425, float %427, i64 1, !dbg !29
  %429 = fadd float %424, %427, !dbg !28
  %430 = extractelement <4 x float> %422, i64 2, !dbg !27
  %431 = call float @llvm.amdgcn.exp2.f32(float %430), !dbg !27
  %432 = insertelement <4 x float> %428, float %431, i64 2, !dbg !29
  %433 = fadd float %429, %431, !dbg !28
  %434 = extractelement <4 x float> %422, i64 3, !dbg !27
  %435 = call float @llvm.amdgcn.exp2.f32(float %434), !dbg !27
  %436 = insertelement <4 x float> %432, float %435, i64 3, !dbg !29
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %436) #9, !dbg !29, !frisk.register_lifetimes !1
  %437 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %437) #9, !dbg !28, !frisk.register_lifetimes !1
  %438 = fadd float %433, %435, !dbg !28
  %439 = shufflevector <16 x float> %419, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %440 = fadd <4 x float> %363, %439, !dbg !26
  %441 = extractelement <4 x float> %440, i64 0, !dbg !27
  %442 = call float @llvm.amdgcn.exp2.f32(float %441), !dbg !27
  %443 = insertelement <4 x float> poison, float %442, i64 0, !dbg !29
  %444 = fadd float %438, %442, !dbg !28
  %445 = extractelement <4 x float> %440, i64 1, !dbg !27
  %446 = call float @llvm.amdgcn.exp2.f32(float %445), !dbg !27
  %447 = insertelement <4 x float> %443, float %446, i64 1, !dbg !29
  %448 = fadd float %444, %446, !dbg !28
  %449 = extractelement <4 x float> %440, i64 2, !dbg !27
  %450 = call float @llvm.amdgcn.exp2.f32(float %449), !dbg !27
  %451 = insertelement <4 x float> %447, float %450, i64 2, !dbg !29
  %452 = fadd float %448, %450, !dbg !28
  %453 = extractelement <4 x float> %440, i64 3, !dbg !27
  %454 = call float @llvm.amdgcn.exp2.f32(float %453), !dbg !27
  %455 = insertelement <4 x float> %451, float %454, i64 3, !dbg !29
  %math.result131 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %455) #9, !dbg !29, !frisk.register_lifetimes !1
  %456 = fptrunc <4 x float> %math.result131 to <4 x half>, !dbg !29
  %math.packed132 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %456) #9, !dbg !28, !frisk.register_lifetimes !1
  %457 = fadd float %452, %454, !dbg !28
  %458 = shufflevector <16 x float> %419, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %459 = fadd <4 x float> %367, %458, !dbg !26
  %460 = extractelement <4 x float> %459, i64 0, !dbg !27
  %461 = call float @llvm.amdgcn.exp2.f32(float %460), !dbg !27
  %462 = insertelement <4 x float> poison, float %461, i64 0, !dbg !29
  %463 = fadd float %457, %461, !dbg !28
  %464 = extractelement <4 x float> %459, i64 1, !dbg !27
  %465 = call float @llvm.amdgcn.exp2.f32(float %464), !dbg !27
  %466 = insertelement <4 x float> %462, float %465, i64 1, !dbg !29
  %467 = fadd float %463, %465, !dbg !28
  %468 = extractelement <4 x float> %459, i64 2, !dbg !27
  %469 = call float @llvm.amdgcn.exp2.f32(float %468), !dbg !27
  %470 = insertelement <4 x float> %466, float %469, i64 2, !dbg !29
  %471 = fadd float %467, %469, !dbg !28
  %472 = extractelement <4 x float> %459, i64 3, !dbg !27
  %473 = call float @llvm.amdgcn.exp2.f32(float %472), !dbg !27
  %474 = insertelement <4 x float> %470, float %473, i64 3, !dbg !29
  %math.result133 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %474) #9, !dbg !29, !frisk.register_lifetimes !1
  %475 = fptrunc <4 x float> %math.result133 to <4 x half>, !dbg !29
  %math.packed134 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %475) #9, !dbg !28, !frisk.register_lifetimes !1
  %476 = fadd float %471, %473, !dbg !28
  %477 = shufflevector <16 x float> %419, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %478 = fadd <4 x float> %372, %477, !dbg !26
  %479 = extractelement <4 x float> %478, i64 0, !dbg !27
  %480 = call float @llvm.amdgcn.exp2.f32(float %479), !dbg !27
  %481 = insertelement <4 x float> poison, float %480, i64 0, !dbg !29
  %482 = fadd float %476, %480, !dbg !28
  %483 = extractelement <4 x float> %478, i64 1, !dbg !27
  %484 = call float @llvm.amdgcn.exp2.f32(float %483), !dbg !27
  %485 = insertelement <4 x float> %481, float %484, i64 1, !dbg !29
  %486 = fadd float %482, %484, !dbg !28
  %487 = extractelement <4 x float> %478, i64 2, !dbg !27
  %488 = call float @llvm.amdgcn.exp2.f32(float %487), !dbg !27
  %489 = insertelement <4 x float> %485, float %488, i64 2, !dbg !29
  %490 = fadd float %486, %488, !dbg !28
  %491 = extractelement <4 x float> %478, i64 3, !dbg !27
  %492 = call float @llvm.amdgcn.exp2.f32(float %491), !dbg !27
  %493 = insertelement <4 x float> %489, float %492, i64 3, !dbg !29
  %math.result135 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %493) #9, !dbg !29, !frisk.register_lifetimes !1
  %494 = fptrunc <4 x float> %math.result135 to <4 x half>, !dbg !29
  %math.packed136 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %494) #9, !dbg !28, !frisk.register_lifetimes !1
  %495 = fadd float %490, %492, !dbg !28
  %496 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %497 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %496), !dbg !28
  %498 = add nuw nsw i32 %497, 64, !dbg !28
  %499 = xor i32 %497, 16, !dbg !28
  %500 = and i32 %498, 192, !dbg !28
  %501 = icmp samesign ult i32 %499, %500, !dbg !28
  %502 = select i1 %501, i32 %499, i32 %497, !dbg !28
  %503 = shl nuw nsw i32 %502, 2, !dbg !28
  %504 = bitcast float %495 to i32, !dbg !28
  %505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %503, i32 %504), !dbg !28
  %506 = bitcast i32 %505 to float, !dbg !28
  %507 = fadd float %495, %506, !dbg !28
  %508 = xor i32 %497, 32, !dbg !28
  %509 = icmp samesign ult i32 %508, %500, !dbg !28
  %510 = select i1 %509, i32 %508, i32 %497, !dbg !28
  %511 = shl nuw nsw i32 %510, 2, !dbg !28
  %512 = bitcast float %507 to i32, !dbg !28
  %513 = call i32 @llvm.amdgcn.ds.bpermute(i32 %511, i32 %512), !dbg !28
  %514 = bitcast i32 %513 to float, !dbg !28
  %515 = fadd float %507, %514, !dbg !28
  %516 = extractelement <1 x float> %65, i64 0, !dbg !28
  %517 = fadd float %515, %516, !dbg !28
  %518 = insertelement <1 x float> poison, float %517, i64 0, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread377 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !33, !frisk.register_lifetimes !1
  %address.local378 = zext nneg i32 %address.thread377 to i64, !dbg !13
  %address.local379 = and i64 %address.local378, 60, !dbg !15
  %address.local380 = add nuw nsw i64 %address.local379, %address.local378, !dbg !15
  %address.local381 = shl nuw nsw i64 %address.local380, 3, !dbg !15
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !15
  %address.local383 = and i64 %address.local378, 63, !dbg !13
  %address.local384 = lshr i64 %address.local383, 4, !dbg !15
  %address.local385 = mul nuw nsw i64 %address.local384, 384, !dbg !15
  %address.local386 = getelementptr [2 x i8], ptr addrspace(3) %address.local382, i64 %address.local385, !dbg !15
  %address.local387 = lshr i64 %address.local378, 2, !dbg !17
  %address.local388 = mul nsw i64 %address.local387, -32, !dbg !15
  %address.local389 = getelementptr [2 x i8], ptr addrspace(3) %address.local386, i64 %address.local388, !dbg !15
  %519 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local389, i32 0) #0, !dbg !15
  %520 = bitcast <4 x i32> %519 to <8 x half>, !dbg !15
  %address.local390 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !15
  %address.local391 = getelementptr [2 x i8], ptr addrspace(3) %address.local390, i64 %address.local385, !dbg !15
  %address.local392 = getelementptr [2 x i8], ptr addrspace(3) %address.local391, i64 %address.local388, !dbg !15
  %address.local393 = getelementptr i8, ptr addrspace(3) %address.local392, i64 256, !dbg !15
  %521 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local393, i32 0) #0, !dbg !15
  %522 = bitcast <4 x i32> %521 to <8 x half>, !dbg !15
  %address.local394 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !15
  %address.local395 = getelementptr [2 x i8], ptr addrspace(3) %address.local394, i64 %address.local385, !dbg !15
  %address.local396 = getelementptr [2 x i8], ptr addrspace(3) %address.local395, i64 %address.local388, !dbg !15
  %address.local397 = getelementptr i8, ptr addrspace(3) %address.local396, i64 512, !dbg !15
  %523 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local397, i32 0) #0, !dbg !15
  %524 = bitcast <4 x i32> %523 to <8 x half>, !dbg !15
  %address.local398 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !15
  %address.local399 = getelementptr [2 x i8], ptr addrspace(3) %address.local398, i64 %address.local385, !dbg !15
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) %address.local399, i64 %address.local388, !dbg !15
  %address.local401 = getelementptr i8, ptr addrspace(3) %address.local400, i64 768, !dbg !15
  %525 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local401, i32 0) #0, !dbg !15
  %526 = bitcast <4 x i32> %525 to <8 x half>, !dbg !15
  %527 = add nuw nsw i64 %94, %15, !dbg !18
  %528 = shl i64 %527, 19, !dbg !18
  %529 = and i64 %528, 16252928, !dbg !18
  %530 = add i64 %529, %100, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %531 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %530, !dbg !18
  %532 = getelementptr i8, ptr addrspace(1) %531, i64 64, !dbg !18
  %address.local402 = shl nuw nsw i64 %address.local378, 3, !dbg !12
  %address.local403 = and i64 %address.local402, 1536, !dbg !12
  %address.local404 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local403, !dbg !12
  %address.local405 = ptrtoint ptr addrspace(3) %address.local404 to i64, !dbg !12
  %address.local406 = trunc i64 %address.local405 to i32, !dbg !12
  %533 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local406), !dbg !18
  %534 = zext i32 %533 to i64, !dbg !18
  %535 = inttoptr i64 %534 to ptr addrspace(3), !dbg !18
  %address.local407 = shl nuw nsw i64 %address.local378, 5, !dbg !18
  %address.local408 = and i64 %address.local407, 8064, !dbg !18
  %address.local409 = shl nuw nsw i64 %address.local378, 3, !dbg !18
  %address.local410 = and i64 %address.local409, 24, !dbg !18
  %address.local411 = or disjoint i64 %address.local408, %address.local410, !dbg !18
  %address.local412 = getelementptr [2 x i8], ptr addrspace(1) %532, i64 %address.local411, !dbg !18
  %536 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %537 = ptrtoint ptr addrspace(1) %address.local412 to i64, !dbg !18
  %538 = sub i64 %537, %536, !dbg !18
  %dma.buffer.offset652 = trunc i64 %538 to i32, !dbg !18
  %539 = trunc i64 %536 to i32, !dbg !18
  %540 = insertelement <4 x i32> poison, i32 %539, i64 0, !dbg !18
  %541 = lshr i64 %536, 32, !dbg !18
  %542 = trunc i64 %541 to i32, !dbg !18
  %543 = insertelement <4 x i32> %540, i32 %542, i64 1, !dbg !18
  %544 = insertelement <4 x i32> %543, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc653 = insertelement <4 x i32> %544, i32 131072, i64 3, !dbg !18
  %545 = ptrtoint ptr addrspace(3) %535 to i32, !dbg !18
  %dma.buffer.lds654 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %545), !dbg !18
  %546 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds654, i32 %dma.buffer.offset652, <4 x i32> %dma.buffer.rsrc653) #10, !dbg !18
  %547 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %547, <4 x float> %62) #9, !dbg !15
  %549 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %549, <4 x float> %548) #9, !dbg !15
  %551 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %551, <4 x float> %550) #9, !dbg !15
  %553 = shufflevector <8 x half> %526, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %553, <4 x float> %552) #9, !dbg !15
  %555 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %555, <4 x float> %63) #9, !dbg !15
  %557 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %557, <4 x float> %556) #9, !dbg !15
  %559 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %559, <4 x float> %558) #9, !dbg !15
  %561 = shufflevector <8 x half> %526, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %561, <4 x float> %560) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %563 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local389, i32 0) #0, !dbg !15
  %564 = bitcast <4 x i32> %563 to <8 x half>, !dbg !15
  %565 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local393, i32 0) #0, !dbg !15
  %566 = bitcast <4 x i32> %565 to <8 x half>, !dbg !15
  %567 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local397, i32 0) #0, !dbg !15
  %568 = bitcast <4 x i32> %567 to <8 x half>, !dbg !15
  %569 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local401, i32 0) #0, !dbg !15
  %570 = bitcast <4 x i32> %569 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %571 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %530, !dbg !18
  %572 = getelementptr i8, ptr addrspace(1) %571, i64 128, !dbg !18
  %573 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local406), !dbg !18
  %574 = zext i32 %573 to i64, !dbg !18
  %575 = inttoptr i64 %574 to ptr addrspace(3), !dbg !18
  %address.local413 = getelementptr [2 x i8], ptr addrspace(1) %572, i64 %address.local411, !dbg !18
  %576 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %577 = ptrtoint ptr addrspace(1) %address.local413 to i64, !dbg !18
  %578 = sub i64 %577, %576, !dbg !18
  %dma.buffer.offset649 = trunc i64 %578 to i32, !dbg !18
  %579 = trunc i64 %576 to i32, !dbg !18
  %580 = insertelement <4 x i32> poison, i32 %579, i64 0, !dbg !18
  %581 = lshr i64 %576, 32, !dbg !18
  %582 = trunc i64 %581 to i32, !dbg !18
  %583 = insertelement <4 x i32> %580, i32 %582, i64 1, !dbg !18
  %584 = insertelement <4 x i32> %583, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc650 = insertelement <4 x i32> %584, i32 131072, i64 3, !dbg !18
  %585 = ptrtoint ptr addrspace(3) %575 to i32, !dbg !18
  %dma.buffer.lds651 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %585), !dbg !18
  %586 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds651, i32 %dma.buffer.offset649, <4 x i32> %dma.buffer.rsrc650) #10, !dbg !18
  %587 = shufflevector <8 x half> %564, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %587, <4 x float> %66) #9, !dbg !15
  %589 = shufflevector <8 x half> %566, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %589, <4 x float> %588) #9, !dbg !15
  %591 = shufflevector <8 x half> %568, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %591, <4 x float> %590) #9, !dbg !15
  %593 = shufflevector <8 x half> %570, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %593, <4 x float> %592) #9, !dbg !15
  %595 = shufflevector <8 x half> %564, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %595, <4 x float> %67) #9, !dbg !15
  %597 = shufflevector <8 x half> %566, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %597, <4 x float> %596) #9, !dbg !15
  %599 = shufflevector <8 x half> %568, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %599, <4 x float> %598) #9, !dbg !15
  %601 = shufflevector <8 x half> %570, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %601, <4 x float> %600) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %603 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local389, i32 0) #0, !dbg !15
  %604 = bitcast <4 x i32> %603 to <8 x half>, !dbg !15
  %605 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local393, i32 0) #0, !dbg !15
  %606 = bitcast <4 x i32> %605 to <8 x half>, !dbg !15
  %607 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local397, i32 0) #0, !dbg !15
  %608 = bitcast <4 x i32> %607 to <8 x half>, !dbg !15
  %609 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local401, i32 0) #0, !dbg !15
  %610 = bitcast <4 x i32> %609 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %611 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %530, !dbg !18
  %612 = getelementptr i8, ptr addrspace(1) %611, i64 192, !dbg !18
  %613 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local406), !dbg !18
  %614 = zext i32 %613 to i64, !dbg !18
  %615 = inttoptr i64 %614 to ptr addrspace(3), !dbg !18
  %address.local414 = getelementptr [2 x i8], ptr addrspace(1) %612, i64 %address.local411, !dbg !18
  %616 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %617 = ptrtoint ptr addrspace(1) %address.local414 to i64, !dbg !18
  %618 = sub i64 %617, %616, !dbg !18
  %dma.buffer.offset = trunc i64 %618 to i32, !dbg !18
  %619 = trunc i64 %616 to i32, !dbg !18
  %620 = insertelement <4 x i32> poison, i32 %619, i64 0, !dbg !18
  %621 = lshr i64 %616, 32, !dbg !18
  %622 = trunc i64 %621 to i32, !dbg !18
  %623 = insertelement <4 x i32> %620, i32 %622, i64 1, !dbg !18
  %624 = insertelement <4 x i32> %623, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %624, i32 131072, i64 3, !dbg !18
  %625 = ptrtoint ptr addrspace(3) %615 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %625), !dbg !18
  %626 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %627 = shufflevector <8 x half> %604, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %627, <4 x float> %69) #9, !dbg !15
  %629 = shufflevector <8 x half> %606, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %629, <4 x float> %628) #9, !dbg !15
  %631 = shufflevector <8 x half> %608, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %631, <4 x float> %630) #9, !dbg !15
  %633 = shufflevector <8 x half> %610, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %633, <4 x float> %632) #9, !dbg !15
  %635 = shufflevector <8 x half> %604, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %635, <4 x float> %70) #9, !dbg !15
  %637 = shufflevector <8 x half> %606, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %637, <4 x float> %636) #9, !dbg !15
  %639 = shufflevector <8 x half> %608, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %639, <4 x float> %638) #9, !dbg !15
  %641 = shufflevector <8 x half> %610, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %641, <4 x float> %640) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %643 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local389, i32 0) #0, !dbg !15
  %644 = bitcast <4 x i32> %643 to <8 x half>, !dbg !15
  %645 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local393, i32 0) #0, !dbg !15
  %646 = bitcast <4 x i32> %645 to <8 x half>, !dbg !15
  %647 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local397, i32 0) #0, !dbg !15
  %648 = bitcast <4 x i32> %647 to <8 x half>, !dbg !15
  %649 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local401, i32 0) #0, !dbg !15
  %650 = bitcast <4 x i32> %649 to <8 x half>, !dbg !15
  %651 = shufflevector <8 x half> %644, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %651, <4 x float> %72) #9, !dbg !15
  %653 = shufflevector <8 x half> %646, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %653, <4 x float> %652) #9, !dbg !15
  %655 = shufflevector <8 x half> %648, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %655, <4 x float> %654) #9, !dbg !15
  %657 = shufflevector <8 x half> %650, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %657, <4 x float> %656) #9, !dbg !15
  %659 = shufflevector <8 x half> %644, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %659, <4 x float> %73) #9, !dbg !15
  %661 = shufflevector <8 x half> %646, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %661, <4 x float> %660) #9, !dbg !15
  %663 = shufflevector <8 x half> %648, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %663, <4 x float> %662) #9, !dbg !15
  %665 = shufflevector <8 x half> %650, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %665, <4 x float> %664) #9, !dbg !15
  %667 = shufflevector <4 x float> %554, <4 x float> %562, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %668 = shufflevector <4 x float> %594, <4 x float> %602, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %669 = shufflevector <4 x float> %634, <4 x float> %642, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %670 = shufflevector <4 x float> %658, <4 x float> %666, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %671 = add nuw nsw i64 %61, 1, !dbg !8
  br label %60, !dbg !8

672:                                              ; preds = %60
  %673 = and i32 %34, 16, !dbg !8
  %.not = icmp eq i32 %673, 0, !dbg !8
  %674 = and i32 %34, 32, !dbg !8
  %.not18 = icmp eq i32 %674, 0, !dbg !8
  %675 = extractelement <8 x float> %74, i64 0, !dbg !8
  %676 = extractelement <8 x float> %74, i64 1, !dbg !8
  %677 = extractelement <8 x float> %74, i64 2, !dbg !8
  %678 = extractelement <8 x float> %74, i64 3, !dbg !8
  %679 = select i1 %.not, float %676, float %675, !dbg !8
  %680 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %681 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %680), !dbg !8
  %682 = add nuw nsw i32 %681, 64, !dbg !8
  %683 = xor i32 %681, 16, !dbg !8
  %684 = and i32 %682, 192, !dbg !8
  %685 = icmp samesign ult i32 %683, %684, !dbg !8
  %686 = select i1 %685, i32 %683, i32 %681, !dbg !8
  %687 = shl nuw nsw i32 %686, 2, !dbg !8
  %688 = bitcast float %679 to i32, !dbg !8
  %689 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %688), !dbg !8
  %690 = bitcast i32 %689 to float, !dbg !8
  %691 = select i1 %.not, float %675, float %690, !dbg !8
  %692 = select i1 %.not, float %690, float %676, !dbg !8
  %693 = select i1 %.not, float %678, float %677, !dbg !8
  %694 = bitcast float %693 to i32, !dbg !8
  %695 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %694), !dbg !8
  %696 = bitcast i32 %695 to float, !dbg !8
  %697 = select i1 %.not, float %677, float %696, !dbg !8
  %698 = select i1 %.not, float %696, float %678, !dbg !8
  %699 = select i1 %.not18, float %697, float %691, !dbg !8
  %700 = xor i32 %681, 32, !dbg !8
  %701 = icmp samesign ult i32 %700, %684, !dbg !8
  %702 = select i1 %701, i32 %700, i32 %681, !dbg !8
  %703 = shl nuw nsw i32 %702, 2, !dbg !8
  %704 = bitcast float %699 to i32, !dbg !8
  %705 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %704), !dbg !8
  %706 = bitcast i32 %705 to float, !dbg !8
  %707 = select i1 %.not18, float %691, float %706, !dbg !8
  %708 = select i1 %.not18, float %706, float %697, !dbg !8
  %709 = select i1 %.not18, float %698, float %692, !dbg !8
  %710 = bitcast float %709 to i32, !dbg !8
  %711 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %710), !dbg !8
  %712 = bitcast i32 %711 to float, !dbg !8
  %713 = select i1 %.not18, float %692, float %712, !dbg !8
  %714 = select i1 %.not18, float %712, float %698, !dbg !8
  %715 = extractelement <8 x float> %71, i64 0, !dbg !8
  %716 = extractelement <8 x float> %71, i64 1, !dbg !8
  %717 = extractelement <8 x float> %71, i64 2, !dbg !8
  %718 = extractelement <8 x float> %71, i64 3, !dbg !8
  %719 = select i1 %.not, float %716, float %715, !dbg !8
  %720 = bitcast float %719 to i32, !dbg !8
  %721 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %720), !dbg !8
  %722 = bitcast i32 %721 to float, !dbg !8
  %723 = select i1 %.not, float %715, float %722, !dbg !8
  %724 = select i1 %.not, float %722, float %716, !dbg !8
  %725 = select i1 %.not, float %718, float %717, !dbg !8
  %726 = bitcast float %725 to i32, !dbg !8
  %727 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %726), !dbg !8
  %728 = bitcast i32 %727 to float, !dbg !8
  %729 = select i1 %.not, float %717, float %728, !dbg !8
  %730 = select i1 %.not, float %728, float %718, !dbg !8
  %731 = select i1 %.not18, float %729, float %723, !dbg !8
  %732 = bitcast float %731 to i32, !dbg !8
  %733 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %732), !dbg !8
  %734 = bitcast i32 %733 to float, !dbg !8
  %735 = select i1 %.not18, float %723, float %734, !dbg !8
  %736 = select i1 %.not18, float %734, float %729, !dbg !8
  %737 = select i1 %.not18, float %730, float %724, !dbg !8
  %738 = bitcast float %737 to i32, !dbg !8
  %739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %738), !dbg !8
  %740 = bitcast i32 %739 to float, !dbg !8
  %741 = select i1 %.not18, float %724, float %740, !dbg !8
  %742 = select i1 %.not18, float %740, float %730, !dbg !8
  %743 = extractelement <8 x float> %68, i64 0, !dbg !8
  %744 = extractelement <8 x float> %68, i64 1, !dbg !8
  %745 = extractelement <8 x float> %68, i64 2, !dbg !8
  %746 = extractelement <8 x float> %68, i64 3, !dbg !8
  %747 = select i1 %.not, float %744, float %743, !dbg !8
  %748 = bitcast float %747 to i32, !dbg !8
  %749 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %748), !dbg !8
  %750 = bitcast i32 %749 to float, !dbg !8
  %751 = select i1 %.not, float %743, float %750, !dbg !8
  %752 = select i1 %.not, float %750, float %744, !dbg !8
  %753 = select i1 %.not, float %746, float %745, !dbg !8
  %754 = bitcast float %753 to i32, !dbg !8
  %755 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %754), !dbg !8
  %756 = bitcast i32 %755 to float, !dbg !8
  %757 = select i1 %.not, float %745, float %756, !dbg !8
  %758 = select i1 %.not, float %756, float %746, !dbg !8
  %759 = select i1 %.not18, float %757, float %751, !dbg !8
  %760 = bitcast float %759 to i32, !dbg !8
  %761 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %760), !dbg !8
  %762 = bitcast i32 %761 to float, !dbg !8
  %763 = select i1 %.not18, float %751, float %762, !dbg !8
  %764 = select i1 %.not18, float %762, float %757, !dbg !8
  %765 = select i1 %.not18, float %758, float %752, !dbg !8
  %766 = bitcast float %765 to i32, !dbg !8
  %767 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %766), !dbg !8
  %768 = bitcast i32 %767 to float, !dbg !8
  %769 = select i1 %.not18, float %752, float %768, !dbg !8
  %770 = select i1 %.not18, float %768, float %758, !dbg !8
  %771 = extractelement <8 x float> %64, i64 0, !dbg !8
  %772 = extractelement <8 x float> %64, i64 1, !dbg !8
  %773 = extractelement <8 x float> %64, i64 2, !dbg !8
  %774 = extractelement <8 x float> %64, i64 3, !dbg !8
  %775 = select i1 %.not, float %772, float %771, !dbg !8
  %776 = bitcast float %775 to i32, !dbg !8
  %777 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %776), !dbg !8
  %778 = bitcast i32 %777 to float, !dbg !8
  %779 = select i1 %.not, float %771, float %778, !dbg !8
  %780 = select i1 %.not, float %778, float %772, !dbg !8
  %781 = select i1 %.not, float %774, float %773, !dbg !8
  %782 = bitcast float %781 to i32, !dbg !8
  %783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %782), !dbg !8
  %784 = bitcast i32 %783 to float, !dbg !8
  %785 = select i1 %.not, float %773, float %784, !dbg !8
  %786 = select i1 %.not, float %784, float %774, !dbg !8
  %787 = select i1 %.not18, float %785, float %779, !dbg !8
  %788 = bitcast float %787 to i32, !dbg !8
  %789 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %788), !dbg !8
  %790 = select i1 %.not18, float %786, float %780, !dbg !8
  %791 = bitcast float %790 to i32, !dbg !8
  %792 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %791), !dbg !8
  %793 = shufflevector <1 x float> %65, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %794 = extractelement <8 x float> %74, i64 4, !dbg !8
  %795 = extractelement <8 x float> %74, i64 5, !dbg !8
  %796 = extractelement <8 x float> %74, i64 6, !dbg !8
  %797 = extractelement <8 x float> %74, i64 7, !dbg !8
  %798 = select i1 %.not, float %795, float %794, !dbg !8
  %799 = bitcast float %798 to i32, !dbg !8
  %800 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %799), !dbg !8
  %801 = bitcast i32 %800 to float, !dbg !8
  %802 = select i1 %.not, float %794, float %801, !dbg !8
  %803 = select i1 %.not, float %801, float %795, !dbg !8
  %804 = select i1 %.not, float %797, float %796, !dbg !8
  %805 = bitcast float %804 to i32, !dbg !8
  %806 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %805), !dbg !8
  %807 = bitcast i32 %806 to float, !dbg !8
  %808 = select i1 %.not, float %796, float %807, !dbg !8
  %809 = select i1 %.not, float %807, float %797, !dbg !8
  %810 = select i1 %.not18, float %808, float %802, !dbg !8
  %811 = bitcast float %810 to i32, !dbg !8
  %812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %811), !dbg !8
  %813 = bitcast i32 %812 to float, !dbg !8
  %814 = select i1 %.not18, float %802, float %813, !dbg !8
  %815 = select i1 %.not18, float %813, float %808, !dbg !8
  %816 = select i1 %.not18, float %809, float %803, !dbg !8
  %817 = bitcast float %816 to i32, !dbg !8
  %818 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %817), !dbg !8
  %819 = bitcast i32 %818 to float, !dbg !8
  %820 = select i1 %.not18, float %803, float %819, !dbg !8
  %821 = select i1 %.not18, float %819, float %809, !dbg !8
  %822 = extractelement <8 x float> %71, i64 4, !dbg !8
  %823 = extractelement <8 x float> %71, i64 5, !dbg !8
  %824 = extractelement <8 x float> %71, i64 6, !dbg !8
  %825 = extractelement <8 x float> %71, i64 7, !dbg !8
  %826 = select i1 %.not, float %823, float %822, !dbg !8
  %827 = bitcast float %826 to i32, !dbg !8
  %828 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %827), !dbg !8
  %829 = bitcast i32 %828 to float, !dbg !8
  %830 = select i1 %.not, float %822, float %829, !dbg !8
  %831 = select i1 %.not, float %829, float %823, !dbg !8
  %832 = select i1 %.not, float %825, float %824, !dbg !8
  %833 = bitcast float %832 to i32, !dbg !8
  %834 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %833), !dbg !8
  %835 = bitcast i32 %834 to float, !dbg !8
  %836 = select i1 %.not, float %824, float %835, !dbg !8
  %837 = select i1 %.not, float %835, float %825, !dbg !8
  %838 = select i1 %.not18, float %836, float %830, !dbg !8
  %839 = bitcast float %838 to i32, !dbg !8
  %840 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %839), !dbg !8
  %841 = bitcast i32 %840 to float, !dbg !8
  %842 = select i1 %.not18, float %830, float %841, !dbg !8
  %843 = select i1 %.not18, float %841, float %836, !dbg !8
  %844 = select i1 %.not18, float %837, float %831, !dbg !8
  %845 = bitcast float %844 to i32, !dbg !8
  %846 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %845), !dbg !8
  %847 = bitcast i32 %846 to float, !dbg !8
  %848 = select i1 %.not18, float %831, float %847, !dbg !8
  %849 = select i1 %.not18, float %847, float %837, !dbg !8
  %850 = extractelement <8 x float> %68, i64 4, !dbg !8
  %851 = extractelement <8 x float> %68, i64 5, !dbg !8
  %852 = extractelement <8 x float> %68, i64 6, !dbg !8
  %853 = extractelement <8 x float> %68, i64 7, !dbg !8
  %854 = select i1 %.not, float %851, float %850, !dbg !8
  %855 = bitcast float %854 to i32, !dbg !8
  %856 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %855), !dbg !8
  %857 = bitcast i32 %856 to float, !dbg !8
  %858 = select i1 %.not, float %850, float %857, !dbg !8
  %859 = select i1 %.not, float %857, float %851, !dbg !8
  %860 = select i1 %.not, float %853, float %852, !dbg !8
  %861 = bitcast float %860 to i32, !dbg !8
  %862 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %861), !dbg !8
  %863 = bitcast i32 %862 to float, !dbg !8
  %864 = select i1 %.not, float %852, float %863, !dbg !8
  %865 = select i1 %.not, float %863, float %853, !dbg !8
  %866 = select i1 %.not18, float %864, float %858, !dbg !8
  %867 = bitcast float %866 to i32, !dbg !8
  %868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %867), !dbg !8
  %869 = bitcast i32 %868 to float, !dbg !8
  %870 = select i1 %.not18, float %858, float %869, !dbg !8
  %871 = select i1 %.not18, float %869, float %864, !dbg !8
  %872 = select i1 %.not18, float %865, float %859, !dbg !8
  %873 = bitcast float %872 to i32, !dbg !8
  %874 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %873), !dbg !8
  %875 = bitcast i32 %874 to float, !dbg !8
  %876 = select i1 %.not18, float %859, float %875, !dbg !8
  %877 = select i1 %.not18, float %875, float %865, !dbg !8
  %878 = extractelement <8 x float> %64, i64 4, !dbg !8
  %879 = extractelement <8 x float> %64, i64 5, !dbg !8
  %880 = extractelement <8 x float> %64, i64 6, !dbg !8
  %881 = extractelement <8 x float> %64, i64 7, !dbg !8
  %882 = select i1 %.not, float %879, float %878, !dbg !8
  %883 = bitcast float %882 to i32, !dbg !8
  %884 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %883), !dbg !8
  %885 = bitcast i32 %884 to float, !dbg !8
  %886 = select i1 %.not, float %878, float %885, !dbg !8
  %887 = select i1 %.not, float %885, float %879, !dbg !8
  %888 = select i1 %.not, float %881, float %880, !dbg !8
  %889 = bitcast float %888 to i32, !dbg !8
  %890 = call i32 @llvm.amdgcn.ds.bpermute(i32 %687, i32 %889), !dbg !8
  %891 = bitcast i32 %890 to float, !dbg !8
  %892 = select i1 %.not, float %880, float %891, !dbg !8
  %893 = select i1 %.not, float %891, float %881, !dbg !8
  %894 = select i1 %.not18, float %892, float %886, !dbg !8
  %895 = bitcast float %894 to i32, !dbg !8
  %896 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %895), !dbg !8
  %897 = select i1 %.not18, float %893, float %887, !dbg !8
  %898 = bitcast float %897 to i32, !dbg !8
  %899 = call i32 @llvm.amdgcn.ds.bpermute(i32 %703, i32 %898), !dbg !8
  br i1 true, label %900, label %934, !dbg !21

900:                                              ; preds = %672
  %901 = bitcast i32 %896 to float, !dbg !8
  %902 = select i1 %.not18, float %886, float %901, !dbg !8
  %903 = insertelement <4 x float> poison, float %902, i64 0, !dbg !19
  %904 = bitcast i32 %899 to float, !dbg !8
  %905 = select i1 %.not18, float %887, float %904, !dbg !8
  %906 = insertelement <4 x float> %903, float %905, i64 1, !dbg !19
  %907 = select i1 %.not18, float %901, float %892, !dbg !8
  %908 = insertelement <4 x float> %906, float %907, i64 2, !dbg !19
  %909 = select i1 %.not18, float %904, float %893, !dbg !8
  %910 = insertelement <4 x float> %908, float %909, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %793, !dbg !19
  %normalized = fmul <4 x float> %910, %row.reciprocal, !dbg !19
  %911 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %912 = extractelement <4 x half> %911, i64 3, !dbg !20
  %913 = extractelement <4 x half> %911, i64 2, !dbg !20
  %914 = extractelement <4 x half> %911, i64 1, !dbg !20
  %915 = extractelement <4 x half> %911, i64 0, !dbg !20
  %916 = bitcast i32 %789 to float, !dbg !8
  %917 = select i1 %.not18, float %779, float %916, !dbg !8
  %918 = insertelement <4 x float> poison, float %917, i64 0, !dbg !19
  %919 = bitcast i32 %792 to float, !dbg !8
  %920 = select i1 %.not18, float %780, float %919, !dbg !8
  %921 = insertelement <4 x float> %918, float %920, i64 1, !dbg !19
  %922 = select i1 %.not18, float %916, float %785, !dbg !8
  %923 = insertelement <4 x float> %921, float %922, i64 2, !dbg !19
  %924 = select i1 %.not18, float %919, float %786, !dbg !8
  %925 = insertelement <4 x float> %923, float %924, i64 3, !dbg !19
  %normalized38 = fmul <4 x float> %925, %row.reciprocal, !dbg !19
  %926 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !20
  %927 = extractelement <4 x half> %926, i64 3, !dbg !20
  %928 = extractelement <4 x half> %926, i64 2, !dbg !20
  %929 = extractelement <4 x half> %926, i64 1, !dbg !20
  %930 = extractelement <4 x half> %926, i64 0, !dbg !20
  %931 = add nsw i64 %15, %25, !dbg !21
  %932 = shl i64 %931, 19, !dbg !21
  %933 = and i64 %932, 16252928, !dbg !21
  %address.thread415 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local416 = zext nneg i32 %address.thread415 to i64, !dbg !13
  %address.local417 = add i64 %28, %address.local416, !dbg !21
  %address.local418 = lshr i64 %address.local416, 6, !dbg !13
  %address.local419 = shl nuw nsw i64 %address.local418, 4, !dbg !21
  %address.local420 = add i64 %address.local417, %address.local419, !dbg !21
  %address.local421 = and i64 %address.local416, 240, !dbg !21
  %address.local422 = sub i64 %address.local420, %address.local421, !dbg !21
  %address.local423 = add i64 %address.local422, %29, !dbg !21
  %address.local424 = shl nuw nsw i64 %address.local423, 7, !dbg !21
  %address.local425 = add nuw nsw i64 %933, %address.local424, !dbg !21
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !21
  %address.local427 = lshr i64 %address.local416, 2, !dbg !13
  %address.local428 = and i64 %address.local427, 12, !dbg !13
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local426, i64 %address.local428, !dbg !21
  store half %930, ptr addrspace(1) %address.local429, align 2, !dbg !21
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !21
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local430, i64 %address.local428, !dbg !21
  %address.local432 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local431, i64 2, !dbg !21
  store half %929, ptr addrspace(1) %address.local432, align 2, !dbg !21
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !21
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local428, !dbg !21
  %address.local435 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local434, i64 4, !dbg !21
  store half %928, ptr addrspace(1) %address.local435, align 2, !dbg !21
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !21
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local436, i64 %address.local428, !dbg !21
  %address.local438 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local437, i64 6, !dbg !21
  store half %927, ptr addrspace(1) %address.local438, align 2, !dbg !21
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !21
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local428, !dbg !21
  %address.local441 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local440, i64 32, !dbg !21
  store half %915, ptr addrspace(1) %address.local441, align 2, !dbg !21
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !21
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local442, i64 %address.local428, !dbg !21
  %address.local444 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local443, i64 34, !dbg !21
  store half %914, ptr addrspace(1) %address.local444, align 2, !dbg !21
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !21
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local445, i64 %address.local428, !dbg !21
  %address.local447 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local446, i64 36, !dbg !21
  store half %913, ptr addrspace(1) %address.local447, align 2, !dbg !21
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !21
  %address.local449 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local448, i64 %address.local428, !dbg !21
  %address.local450 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local449, i64 38, !dbg !21
  store half %912, ptr addrspace(1) %address.local450, align 2, !dbg !21
  br label %934, !dbg !21

934:                                              ; preds = %900, %672
  br i1 true, label %935, label %955, !dbg !21

935:                                              ; preds = %934
  %936 = insertelement <4 x float> poison, float %870, i64 0, !dbg !19
  %937 = insertelement <4 x float> %936, float %876, i64 1, !dbg !19
  %938 = insertelement <4 x float> %937, float %871, i64 2, !dbg !19
  %939 = insertelement <4 x float> %938, float %877, i64 3, !dbg !19
  %row.reciprocal39 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %793, !dbg !19
  %normalized40 = fmul <4 x float> %939, %row.reciprocal39, !dbg !19
  %940 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !20
  %941 = extractelement <4 x half> %940, i64 3, !dbg !20
  %942 = extractelement <4 x half> %940, i64 2, !dbg !20
  %943 = extractelement <4 x half> %940, i64 1, !dbg !20
  %944 = extractelement <4 x half> %940, i64 0, !dbg !20
  %945 = insertelement <4 x float> poison, float %763, i64 0, !dbg !19
  %946 = insertelement <4 x float> %945, float %769, i64 1, !dbg !19
  %947 = insertelement <4 x float> %946, float %764, i64 2, !dbg !19
  %948 = insertelement <4 x float> %947, float %770, i64 3, !dbg !19
  %normalized41 = fmul <4 x float> %948, %row.reciprocal39, !dbg !19
  %949 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !20
  %950 = extractelement <4 x half> %949, i64 3, !dbg !20
  %951 = extractelement <4 x half> %949, i64 2, !dbg !20
  %952 = extractelement <4 x half> %949, i64 1, !dbg !20
  %953 = extractelement <4 x half> %949, i64 0, !dbg !20
  %954 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread451 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local452 = zext nneg i32 %address.thread451 to i64, !dbg !13
  %address.local453 = add i64 %28, %address.local452, !dbg !21
  %address.local454 = lshr i64 %address.local452, 6, !dbg !13
  %address.local455 = shl nuw nsw i64 %address.local454, 4, !dbg !21
  %address.local456 = add i64 %address.local453, %address.local455, !dbg !21
  %address.local457 = and i64 %address.local452, 240, !dbg !21
  %address.local458 = sub i64 %address.local456, %address.local457, !dbg !21
  %address.local459 = add i64 %address.local458, %29, !dbg !21
  %address.local460 = shl nuw nsw i64 %address.local459, 7, !dbg !21
  %address.local461 = add nuw nsw i64 %954, %address.local460, !dbg !21
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local463 = lshr i64 %address.local452, 2, !dbg !13
  %address.local464 = and i64 %address.local463, 12, !dbg !13
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local462, i64 %address.local464, !dbg !21
  %address.local466 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local465, i64 64, !dbg !21
  store half %953, ptr addrspace(1) %address.local466, align 2, !dbg !21
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local467, i64 %address.local464, !dbg !21
  %address.local469 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local468, i64 66, !dbg !21
  store half %952, ptr addrspace(1) %address.local469, align 2, !dbg !21
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local470, i64 %address.local464, !dbg !21
  %address.local472 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local471, i64 68, !dbg !21
  store half %951, ptr addrspace(1) %address.local472, align 2, !dbg !21
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local473, i64 %address.local464, !dbg !21
  %address.local475 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local474, i64 70, !dbg !21
  store half %950, ptr addrspace(1) %address.local475, align 2, !dbg !21
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local476, i64 %address.local464, !dbg !21
  %address.local478 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local477, i64 96, !dbg !21
  store half %944, ptr addrspace(1) %address.local478, align 2, !dbg !21
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local479, i64 %address.local464, !dbg !21
  %address.local481 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local480, i64 98, !dbg !21
  store half %943, ptr addrspace(1) %address.local481, align 2, !dbg !21
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local482, i64 %address.local464, !dbg !21
  %address.local484 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local483, i64 100, !dbg !21
  store half %942, ptr addrspace(1) %address.local484, align 2, !dbg !21
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local485, i64 %address.local464, !dbg !21
  %address.local487 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local486, i64 102, !dbg !21
  store half %941, ptr addrspace(1) %address.local487, align 2, !dbg !21
  br label %955, !dbg !21

955:                                              ; preds = %935, %934
  br i1 true, label %956, label %976, !dbg !21

956:                                              ; preds = %955
  %957 = insertelement <4 x float> poison, float %842, i64 0, !dbg !19
  %958 = insertelement <4 x float> %957, float %848, i64 1, !dbg !19
  %959 = insertelement <4 x float> %958, float %843, i64 2, !dbg !19
  %960 = insertelement <4 x float> %959, float %849, i64 3, !dbg !19
  %row.reciprocal42 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %793, !dbg !19
  %normalized43 = fmul <4 x float> %960, %row.reciprocal42, !dbg !19
  %961 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !20
  %962 = extractelement <4 x half> %961, i64 3, !dbg !20
  %963 = extractelement <4 x half> %961, i64 2, !dbg !20
  %964 = extractelement <4 x half> %961, i64 1, !dbg !20
  %965 = extractelement <4 x half> %961, i64 0, !dbg !20
  %966 = insertelement <4 x float> poison, float %735, i64 0, !dbg !19
  %967 = insertelement <4 x float> %966, float %741, i64 1, !dbg !19
  %968 = insertelement <4 x float> %967, float %736, i64 2, !dbg !19
  %969 = insertelement <4 x float> %968, float %742, i64 3, !dbg !19
  %normalized44 = fmul <4 x float> %969, %row.reciprocal42, !dbg !19
  %970 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !20
  %971 = extractelement <4 x half> %970, i64 3, !dbg !20
  %972 = extractelement <4 x half> %970, i64 2, !dbg !20
  %973 = extractelement <4 x half> %970, i64 1, !dbg !20
  %974 = extractelement <4 x half> %970, i64 0, !dbg !20
  %975 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread488 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local489 = zext nneg i32 %address.thread488 to i64, !dbg !13
  %address.local490 = add i64 %28, %address.local489, !dbg !21
  %address.local491 = lshr i64 %address.local489, 6, !dbg !13
  %address.local492 = shl nuw nsw i64 %address.local491, 4, !dbg !21
  %address.local493 = add i64 %address.local490, %address.local492, !dbg !21
  %address.local494 = and i64 %address.local489, 240, !dbg !21
  %address.local495 = sub i64 %address.local493, %address.local494, !dbg !21
  %address.local496 = add i64 %address.local495, %29, !dbg !21
  %address.local497 = shl nuw nsw i64 %address.local496, 7, !dbg !21
  %address.local498 = add nuw nsw i64 %975, %address.local497, !dbg !21
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !21
  %address.local500 = lshr i64 %address.local489, 2, !dbg !13
  %address.local501 = and i64 %address.local500, 12, !dbg !13
  %address.local502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local499, i64 %address.local501, !dbg !21
  %address.local503 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local502, i64 128, !dbg !21
  store half %974, ptr addrspace(1) %address.local503, align 2, !dbg !21
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !21
  %address.local505 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local504, i64 %address.local501, !dbg !21
  %address.local506 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local505, i64 130, !dbg !21
  store half %973, ptr addrspace(1) %address.local506, align 2, !dbg !21
  %address.local507 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !21
  %address.local508 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local507, i64 %address.local501, !dbg !21
  %address.local509 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local508, i64 132, !dbg !21
  store half %972, ptr addrspace(1) %address.local509, align 2, !dbg !21
  %address.local510 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !21
  %address.local511 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local510, i64 %address.local501, !dbg !21
  %address.local512 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local511, i64 134, !dbg !21
  store half %971, ptr addrspace(1) %address.local512, align 2, !dbg !21
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !21
  %address.local514 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local513, i64 %address.local501, !dbg !21
  %address.local515 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local514, i64 160, !dbg !21
  store half %965, ptr addrspace(1) %address.local515, align 2, !dbg !21
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !21
  %address.local517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local516, i64 %address.local501, !dbg !21
  %address.local518 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local517, i64 162, !dbg !21
  store half %964, ptr addrspace(1) %address.local518, align 2, !dbg !21
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !21
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local519, i64 %address.local501, !dbg !21
  %address.local521 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local520, i64 164, !dbg !21
  store half %963, ptr addrspace(1) %address.local521, align 2, !dbg !21
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !21
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local522, i64 %address.local501, !dbg !21
  %address.local524 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local523, i64 166, !dbg !21
  store half %962, ptr addrspace(1) %address.local524, align 2, !dbg !21
  br label %976, !dbg !21

976:                                              ; preds = %956, %955
  br i1 true, label %977, label %997, !dbg !21

977:                                              ; preds = %976
  %978 = insertelement <4 x float> poison, float %814, i64 0, !dbg !19
  %979 = insertelement <4 x float> %978, float %820, i64 1, !dbg !19
  %980 = insertelement <4 x float> %979, float %815, i64 2, !dbg !19
  %981 = insertelement <4 x float> %980, float %821, i64 3, !dbg !19
  %row.reciprocal45 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %793, !dbg !19
  %normalized46 = fmul <4 x float> %981, %row.reciprocal45, !dbg !19
  %982 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !20
  %983 = extractelement <4 x half> %982, i64 3, !dbg !20
  %984 = extractelement <4 x half> %982, i64 2, !dbg !20
  %985 = extractelement <4 x half> %982, i64 1, !dbg !20
  %986 = extractelement <4 x half> %982, i64 0, !dbg !20
  %987 = insertelement <4 x float> poison, float %707, i64 0, !dbg !19
  %988 = insertelement <4 x float> %987, float %713, i64 1, !dbg !19
  %989 = insertelement <4 x float> %988, float %708, i64 2, !dbg !19
  %990 = insertelement <4 x float> %989, float %714, i64 3, !dbg !19
  %normalized47 = fmul <4 x float> %990, %row.reciprocal45, !dbg !19
  %991 = fptrunc <4 x float> %normalized47 to <4 x half>, !dbg !20
  %992 = extractelement <4 x half> %991, i64 3, !dbg !20
  %993 = extractelement <4 x half> %991, i64 2, !dbg !20
  %994 = extractelement <4 x half> %991, i64 1, !dbg !20
  %995 = extractelement <4 x half> %991, i64 0, !dbg !20
  %996 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread525 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local526 = zext nneg i32 %address.thread525 to i64, !dbg !13
  %address.local527 = add i64 %28, %address.local526, !dbg !21
  %address.local528 = lshr i64 %address.local526, 6, !dbg !13
  %address.local529 = shl nuw nsw i64 %address.local528, 4, !dbg !21
  %address.local530 = add i64 %address.local527, %address.local529, !dbg !21
  %address.local531 = and i64 %address.local526, 240, !dbg !21
  %address.local532 = sub i64 %address.local530, %address.local531, !dbg !21
  %address.local533 = add i64 %address.local532, %29, !dbg !21
  %address.local534 = shl nuw nsw i64 %address.local533, 7, !dbg !21
  %address.local535 = add nuw nsw i64 %996, %address.local534, !dbg !21
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !21
  %address.local537 = lshr i64 %address.local526, 2, !dbg !13
  %address.local538 = and i64 %address.local537, 12, !dbg !13
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local536, i64 %address.local538, !dbg !21
  %address.local540 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local539, i64 192, !dbg !21
  store half %995, ptr addrspace(1) %address.local540, align 2, !dbg !21
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !21
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local541, i64 %address.local538, !dbg !21
  %address.local543 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local542, i64 194, !dbg !21
  store half %994, ptr addrspace(1) %address.local543, align 2, !dbg !21
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local544, i64 %address.local538, !dbg !21
  %address.local546 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local545, i64 196, !dbg !21
  store half %993, ptr addrspace(1) %address.local546, align 2, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !21
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local547, i64 %address.local538, !dbg !21
  %address.local549 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local548, i64 198, !dbg !21
  store half %992, ptr addrspace(1) %address.local549, align 2, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local550, i64 %address.local538, !dbg !21
  %address.local552 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local551, i64 224, !dbg !21
  store half %986, ptr addrspace(1) %address.local552, align 2, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local553, i64 %address.local538, !dbg !21
  %address.local555 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local554, i64 226, !dbg !21
  store half %985, ptr addrspace(1) %address.local555, align 2, !dbg !21
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !21
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local556, i64 %address.local538, !dbg !21
  %address.local558 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local557, i64 228, !dbg !21
  store half %984, ptr addrspace(1) %address.local558, align 2, !dbg !21
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !21
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local559, i64 %address.local538, !dbg !21
  %address.local561 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local560, i64 230, !dbg !21
  store half %983, ptr addrspace(1) %address.local561, align 2, !dbg !21
  br label %997, !dbg !21

997:                                              ; preds = %977, %976
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/bm_bn_sweep/export/run-20261022-180820-lrul_y0l/input-0004-test_input_1x32x4096x128-64-64-32/group8-reverse-k128-swizzle")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!8 = !DILocation(line: 57, column: 24, scope: !4)
!10 = !DILocation(line: 47, column: 17, scope: !4)
!12 = !DILocation(line: 67, column: 22, scope: !4)
!13 = !DILocation(line: 69, column: 24, scope: !4)
!14 = !DILocation(line: 74, column: 19, scope: !4)
!15 = !DILocation(line: 94, column: 23, scope: !4)
!16 = !DILocation(line: 68, column: 22, scope: !4)
!17 = !DILocation(line: 70, column: 26, scope: !4)
!18 = !DILocation(line: 93, column: 20, scope: !4)
!19 = !DILocation(line: 98, column: 23, scope: !4)
!20 = !DILocation(line: 99, column: 19, scope: !4)
!21 = !DILocation(line: 100, column: 9, scope: !4)
!23 = !DILocation(line: 76, column: 24, scope: !4)
!24 = !DILocation(line: 77, column: 23, scope: !4)
!25 = !DILocation(line: 78, column: 26, scope: !4)
!26 = !DILocation(line: 85, column: 21, scope: !4)
!27 = !DILocation(line: 86, column: 22, scope: !4)
!28 = !DILocation(line: 87, column: 20, scope: !4)
!29 = !DILocation(line: 88, column: 16, scope: !4)
!31 = !{i32 256, i32 1, i32 1}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 0, i32 256}
