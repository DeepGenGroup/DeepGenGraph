; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/bm_bn_sweep/export/run-20261022-180820-lrul_y0l/input-0003-test_input_1x32x4096x128-64-128-32/group8-reverse-k128-swizzle/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/bm_bn_sweep/export/run-20261022-180820-lrul_y0l/input-0003-test_input_1x32x4096x128-64-128-32/group8-reverse-k128-swizzle/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

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
  %40 = icmp ugt i64 %10, 2047, !dbg !8
  %41 = select i1 %40, i64 %21, i64 %16, !dbg !8
  %42 = sdiv i64 %41, 2, !dbg !8
  %43 = sub nsw i64 0, %42, !dbg !8
  %44 = add nsw i64 %42, 1, !dbg !8
  %45 = select i1 %40, i64 %43, i64 %44, !dbg !8
  %46 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !12
  %47 = lshr i64 %35, 2, !dbg !17
  %48 = shl nuw nsw i64 %35, 2, !dbg !17
  %49 = mul nsw i64 %47, -16, !dbg !17
  %50 = add nsw i64 %48, %49, !dbg !17
  %51 = and i32 %34, 1, !dbg !17
  %index.compact248 = zext nneg i32 %51 to i64, !dbg !17
  %52 = or disjoint i64 %50, %index.compact248, !dbg !17
  %53 = add nuw nsw i64 %52, 1, !dbg !17
  %54 = sdiv i64 %53, 2, !dbg !17
  %55 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !12
  %56 = getelementptr i8, ptr addrspace(1) %55, i64 64, !dbg !12
  %57 = shl i64 %15, 7, !dbg !16
  %58 = and i64 %57, 3968, !dbg !16
  %59 = shl nsw i64 %18, 12, !dbg !16
  %60 = or disjoint i64 %58, %59, !dbg !16
  %61 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !12
  %62 = getelementptr i8, ptr addrspace(1) %61, i64 128, !dbg !12
  %63 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !12
  %64 = getelementptr i8, ptr addrspace(1) %63, i64 192, !dbg !12
  br label %65, !dbg !8

65:                                               ; preds = %591, %4
  %66 = phi i64 [ %1056, %591 ], [ 0, %4 ], !dbg !8
  %67 = phi <4 x float> [ %831, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %68 = phi <4 x float> [ %847, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %69 = phi <8 x float> [ %1052, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %70 = phi <1 x float> [ %765, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %71 = phi <4 x float> [ %909, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %72 = phi <4 x float> [ %925, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %73 = phi <8 x float> [ %1053, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %74 = phi <4 x float> [ %987, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %75 = phi <4 x float> [ %1003, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %76 = phi <8 x float> [ %1054, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %77 = phi <4 x float> [ %1035, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %78 = phi <4 x float> [ %1051, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %79 = phi <8 x float> [ %1055, %591 ], [ zeroinitializer, %4 ], !dbg !8
  %80 = icmp slt i64 %66, %45, !dbg !8
  br i1 %80, label %81, label %1057, !dbg !8

81:                                               ; preds = %65
  %82 = shl nsw i64 %66, 7, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !33, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local290 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local291 = and i64 %address.local290, 1536, !dbg !12
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local291, !dbg !12
  %address.local293 = ptrtoint ptr addrspace(3) %address.local292 to i64, !dbg !12
  %address.local294 = trunc i64 %address.local293 to i32, !dbg !12
  %83 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local294), !dbg !12
  %84 = zext i32 %83 to i64, !dbg !12
  %85 = inttoptr i64 %84 to ptr addrspace(3), !dbg !12
  %address.local295 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local296 = and i64 %address.local295, 8064, !dbg !12
  %address.local297 = lshr i64 %address.local, 3, !dbg !12
  %address.local298 = and i64 %address.local297, 24, !dbg !12
  %address.local299 = or disjoint i64 %address.local296, %address.local298, !dbg !12
  %address.local300 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local299, !dbg !12
  %86 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %87 = ptrtoint ptr addrspace(1) %address.local300 to i64, !dbg !12
  %88 = sub i64 %87, %86, !dbg !12
  %dma.buffer.offset1215 = trunc i64 %88 to i32, !dbg !12
  %89 = trunc i64 %86 to i32, !dbg !12
  %90 = insertelement <4 x i32> poison, i32 %89, i64 0, !dbg !12
  %91 = lshr i64 %86, 32, !dbg !12
  %92 = trunc i64 %91 to i32, !dbg !12
  %93 = insertelement <4 x i32> %90, i32 %92, i64 1, !dbg !12
  %94 = insertelement <4 x i32> %93, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1216 = insertelement <4 x i32> %94, i32 131072, i64 3, !dbg !12
  %95 = ptrtoint ptr addrspace(3) %85 to i32, !dbg !12
  %dma.buffer.lds1217 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %95), !dbg !12
  %96 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1217, i32 %dma.buffer.offset1215, <4 x i32> %dma.buffer.rsrc1216) #10, !dbg !12
  %97 = lshr i64 %66, 12, !dbg !16
  %98 = add nuw nsw i64 %97, %15, !dbg !16
  %99 = lshr i64 %66, 5, !dbg !16
  %100 = and i64 %99, 127, !dbg !16
  %101 = shl i64 %98, 19, !dbg !16
  %102 = and i64 %101, 16252928, !dbg !16
  %103 = shl i64 %99, 19, !dbg !16
  %104 = shl i64 %66, 14, !dbg !16
  %105 = sub i64 %104, %103, !dbg !16
  %106 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %102, !dbg !16
  %107 = getelementptr [2 x i8], ptr addrspace(1) %106, i64 %100, !dbg !16
  %108 = getelementptr [2 x i8], ptr addrspace(1) %107, i64 %105, !dbg !16
  %address.local301 = and i64 %address.local290, 1536, !dbg !16
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local301, !dbg !16
  %address.local303 = ptrtoint ptr addrspace(3) %address.local302 to i64, !dbg !16
  %address.local304 = trunc i64 %address.local303 to i32, !dbg !16
  %109 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !16
  %110 = zext i32 %109 to i64, !dbg !16
  %111 = inttoptr i64 %110 to ptr addrspace(3), !dbg !16
  %address.local305 = lshr i64 %address.local, 4, !dbg !16
  %address.local306 = and i64 %address.local305, 8, !dbg !16
  %address.local307 = and i64 %address.local, 123, !dbg !16
  %address.local308 = lshr i64 %address.local, 2, !dbg !16
  %address.local309 = and i64 %address.local308, 4, !dbg !16
  %address.local310 = add nuw nsw i64 %address.local309, %address.local, !dbg !16
  %address.local311 = and i64 %address.local310, 4, !dbg !16
  %address.local312 = or disjoint i64 %address.local307, %address.local311, !dbg !16
  %address.local313 = shl nuw nsw i64 %address.local312, 7, !dbg !16
  %address.local314 = or disjoint i64 %address.local306, %address.local313, !dbg !16
  %address.local315 = getelementptr [2 x i8], ptr addrspace(1) %108, i64 %address.local314, !dbg !16
  %112 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %113 = ptrtoint ptr addrspace(1) %address.local315 to i64, !dbg !16
  %114 = sub i64 %113, %112, !dbg !16
  %dma.buffer.offset1212 = trunc i64 %114 to i32, !dbg !16
  %115 = trunc i64 %112 to i32, !dbg !16
  %116 = insertelement <4 x i32> poison, i32 %115, i64 0, !dbg !16
  %117 = lshr i64 %112, 32, !dbg !16
  %118 = trunc i64 %117 to i32, !dbg !16
  %119 = insertelement <4 x i32> %116, i32 %118, i64 1, !dbg !16
  %120 = insertelement <4 x i32> %119, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1213 = insertelement <4 x i32> %120, i32 131072, i64 3, !dbg !16
  %121 = ptrtoint ptr addrspace(3) %111 to i32, !dbg !16
  %dma.buffer.lds1214 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %121), !dbg !16
  %122 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1214, i32 %dma.buffer.offset1212, <4 x i32> %dma.buffer.rsrc1213) #10, !dbg !16
  %address.local316 = or disjoint i64 %address.local, 256, !dbg !16
  %address.local317 = shl nuw nsw i64 %address.local316, 3, !dbg !16
  %address.local318 = and i64 %address.local317, 3584, !dbg !16
  %address.local319 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local318, !dbg !16
  %address.local320 = ptrtoint ptr addrspace(3) %address.local319 to i64, !dbg !16
  %address.local321 = trunc i64 %address.local320 to i32, !dbg !16
  %123 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local321), !dbg !16
  %124 = zext i32 %123 to i64, !dbg !16
  %125 = inttoptr i64 %124 to ptr addrspace(3), !dbg !16
  %address.local322 = lshr i64 %address.local316, 4, !dbg !16
  %address.local323 = and i64 %address.local322, 24, !dbg !16
  %address.local324 = and i64 %address.local, 123, !dbg !16
  %address.local325 = lshr i64 %address.local, 2, !dbg !16
  %address.local326 = and i64 %address.local325, 4, !dbg !16
  %address.local327 = add nuw nsw i64 %address.local326, %address.local, !dbg !16
  %address.local328 = and i64 %address.local327, 4, !dbg !16
  %address.local329 = or disjoint i64 %address.local324, %address.local328, !dbg !16
  %address.local330 = shl nuw nsw i64 %address.local329, 7, !dbg !16
  %address.local331 = or disjoint i64 %address.local323, %address.local330, !dbg !16
  %address.local332 = getelementptr [2 x i8], ptr addrspace(1) %108, i64 %address.local331, !dbg !16
  %126 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %127 = ptrtoint ptr addrspace(1) %address.local332 to i64, !dbg !16
  %128 = sub i64 %127, %126, !dbg !16
  %dma.buffer.offset1209 = trunc i64 %128 to i32, !dbg !16
  %129 = trunc i64 %126 to i32, !dbg !16
  %130 = insertelement <4 x i32> poison, i32 %129, i64 0, !dbg !16
  %131 = lshr i64 %126, 32, !dbg !16
  %132 = trunc i64 %131 to i32, !dbg !16
  %133 = insertelement <4 x i32> %130, i32 %132, i64 1, !dbg !16
  %134 = insertelement <4 x i32> %133, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1210 = insertelement <4 x i32> %134, i32 131072, i64 3, !dbg !16
  %135 = ptrtoint ptr addrspace(3) %125 to i32, !dbg !16
  %dma.buffer.lds1211 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %135), !dbg !16
  %136 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1211, i32 %dma.buffer.offset1209, <4 x i32> %dma.buffer.rsrc1210) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local333 = lshr i32 %address.thread, 2, !dbg !13
  %address.local334 = and i32 %address.local333, 4, !dbg !13
  %address.local335 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local336 = and i32 %address.local335, 384, !dbg !13
  %address.local337 = or disjoint i32 %address.local334, %address.local336, !dbg !13
  %address.local338 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local339 = and i32 %address.local338, 120, !dbg !13
  %address.local340 = or disjoint i32 %address.local337, %address.local339, !dbg !13
  %address.local341 = shl nuw nsw i32 %address.thread, 4, !dbg !13
  %address.local342 = and i32 %address.local341, 512, !dbg !13
  %address.local343 = or disjoint i32 %address.local340, %address.local342, !dbg !13
  %address.local344 = zext nneg i32 %address.local343 to i64, !dbg !13
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local344, !dbg !13
  %137 = load <4 x half>, ptr addrspace(3) %address.local345, align 8, !dbg !13
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local345, i64 2048, !dbg !13
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !13
  %139 = fmul <4 x half> %137, splat (half 0xH3015), !dbg !13
  %140 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local348 = mul nuw nsw i64 %address.local, 160, !dbg !17
  %address.local349 = lshr i64 %address.local, 2, !dbg !17
  %address.local350 = mul nsw i64 %address.local349, -640, !dbg !17
  %address.local351 = add nsw i64 %address.local348, %address.local350, !dbg !17
  %address.local352 = shl nuw nsw i64 %address.local, 1, !dbg !17
  %address.local353 = and i64 %address.local352, 24, !dbg !17
  %address.local354 = or disjoint i64 %address.local351, %address.local353, !dbg !17
  %address.local355 = shl i64 %address.local, 62, !dbg !17
  %address.local356 = ashr exact i64 %address.local355, 57, !dbg !17
  %address.local357 = and i64 %address.local356, -64, !dbg !17
  %address.local358 = add nsw i64 %address.local354, %address.local357, !dbg !17
  %address.local359 = lshr i64 %address.local, 2, !dbg !13
  %address.local360 = and i64 %address.local359, 12, !dbg !13
  %address.local361 = add nsw i64 %address.local358, %address.local360, !dbg !17
  %address.local362 = and i64 %address.local, 32, !dbg !17
  %address.local363 = icmp eq i64 %address.local362, 0, !dbg !17
  %address.local364 = select i1 %address.local363, i64 0, i64 1016, !dbg !17
  %address.local365 = add i64 %address.local361, %address.local364, !dbg !17
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local365, !dbg !17
  %141 = load <4 x half>, ptr addrspace(3) %address.local366, align 8, !dbg !17
  %lds.const.gep1102 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4096, !dbg !17
  %142 = load <4 x half>, ptr addrspace(3) %lds.const.gep1102, align 8, !dbg !17
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %141, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local377 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local378 = and i32 %address.local377, 24, !dbg !17
  %address.local379 = shl nuw nsw i32 %address.thread, 8, !dbg !17
  %address.local380 = and i32 %address.local379, 768, !dbg !17
  %address.local381 = or disjoint i32 %address.local378, %address.local380, !dbg !17
  %address.local382 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local383 = and i32 %address.local382, 96, !dbg !17
  %address.local384 = or disjoint i32 %address.local381, %address.local383, !dbg !17
  %address.local385 = zext nneg i32 %address.local384 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %54) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local386 = lshr i32 %address.thread, 1, !dbg !17
  %address.local387 = and i32 %address.local386, 1, !dbg !17
  %address.local388 = zext nneg i32 %address.local387 to i64, !dbg !17
  %address.local389 = add nsw i64 %address.base, %address.local388, !dbg !17
  %address.local390 = shl i64 %address.local389, 6, !dbg !17
  %address.local391 = sub i64 %address.local385, %address.local390, !dbg !17
  %address.local392 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local391, !dbg !17
  %address.local393 = getelementptr [2 x i8], ptr addrspace(3) %address.local392, i64 %address.local360, !dbg !17
  %address.local394 = getelementptr [2 x i8], ptr addrspace(3) %address.local393, i64 %address.local364, !dbg !17
  %address.local395 = getelementptr i8, ptr addrspace(3) %address.local394, i64 64, !dbg !17
  %144 = load <4 x half>, ptr addrspace(3) %address.local395, align 8, !dbg !17
  %145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %142, <4 x float> %143) #9, !dbg !17
  %lds.const.gep1103 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4096, !dbg !17
  %146 = load <4 x half>, ptr addrspace(3) %lds.const.gep1103, align 8, !dbg !17
  %lds.const.gep1104 = getelementptr i8, ptr addrspace(3) %address.local366, i64 128, !dbg !17
  %147 = load <4 x half>, ptr addrspace(3) %lds.const.gep1104, align 8, !dbg !17
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %144, <4 x float> zeroinitializer) #9, !dbg !17
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %146, <4 x float> %148) #9, !dbg !17
  %lds.const.gep1105 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4224, !dbg !17
  %150 = load <4 x half>, ptr addrspace(3) %lds.const.gep1105, align 8, !dbg !17
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %147, <4 x float> zeroinitializer) #9, !dbg !17
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %150, <4 x float> %151) #9, !dbg !17
  %lds.const.gep1106 = getelementptr i8, ptr addrspace(3) %address.local395, i64 128, !dbg !17
  %153 = load <4 x half>, ptr addrspace(3) %lds.const.gep1106, align 8, !dbg !17
  %lds.const.gep1107 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4224, !dbg !17
  %154 = load <4 x half>, ptr addrspace(3) %lds.const.gep1107, align 8, !dbg !17
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1108 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1024, !dbg !17
  %156 = load <4 x half>, ptr addrspace(3) %lds.const.gep1108, align 8, !dbg !17
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %154, <4 x float> %155) #9, !dbg !17
  %lds.const.gep1109 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5120, !dbg !17
  %158 = load <4 x half>, ptr addrspace(3) %lds.const.gep1109, align 8, !dbg !17
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %156, <4 x float> zeroinitializer) #9, !dbg !17
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %158, <4 x float> %159) #9, !dbg !17
  %lds.const.gep1110 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1024, !dbg !17
  %161 = load <4 x half>, ptr addrspace(3) %lds.const.gep1110, align 8, !dbg !17
  %lds.const.gep1111 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5120, !dbg !17
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep1111, align 8, !dbg !17
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %161, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1112 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1152, !dbg !17
  %164 = load <4 x half>, ptr addrspace(3) %lds.const.gep1112, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %162, <4 x float> %163) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %lds.const.gep1113 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5248, !dbg !17
  %166 = load <4 x half>, ptr addrspace(3) %lds.const.gep1113, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %164, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %166, <4 x float> %167) #9, !dbg !17
  %lds.const.gep1114 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1152, !dbg !17
  %169 = load <4 x half>, ptr addrspace(3) %lds.const.gep1114, align 8, !dbg !17
  %lds.const.gep1115 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5248, !dbg !17
  %170 = load <4 x half>, ptr addrspace(3) %lds.const.gep1115, align 8, !dbg !17
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %169, <4 x float> zeroinitializer) #9, !dbg !17
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %170, <4 x float> %171) #9, !dbg !17
  %173 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local294), !dbg !12
  %174 = zext i32 %173 to i64, !dbg !12
  %175 = inttoptr i64 %174 to ptr addrspace(3), !dbg !12
  %address.local704 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local299, !dbg !12
  %176 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %177 = ptrtoint ptr addrspace(1) %address.local704 to i64, !dbg !12
  %178 = sub i64 %177, %176, !dbg !12
  %dma.buffer.offset1206 = trunc i64 %178 to i32, !dbg !12
  %179 = trunc i64 %176 to i32, !dbg !12
  %180 = insertelement <4 x i32> poison, i32 %179, i64 0, !dbg !12
  %181 = lshr i64 %176, 32, !dbg !12
  %182 = trunc i64 %181 to i32, !dbg !12
  %183 = insertelement <4 x i32> %180, i32 %182, i64 1, !dbg !12
  %184 = insertelement <4 x i32> %183, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1207 = insertelement <4 x i32> %184, i32 131072, i64 3, !dbg !12
  %185 = ptrtoint ptr addrspace(3) %175 to i32, !dbg !12
  %dma.buffer.lds1208 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %185), !dbg !12
  %186 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1208, i32 %dma.buffer.offset1206, <4 x i32> %dma.buffer.rsrc1207) #10, !dbg !12
  %187 = add nuw i64 %66, 1024, !dbg !16
  %188 = icmp slt i64 %187, 0, !dbg !16
  %189 = sub nuw i64 -1025, %66, !dbg !16
  %190 = select i1 %188, i64 %189, i64 %187, !dbg !16
  %191 = sdiv i64 %190, 4096, !dbg !16
  %.lobit37 = ashr i64 %187, 63, !dbg !16
  %192 = xor i64 %191, %.lobit37, !dbg !16
  %193 = add nsw i64 %192, %15, !dbg !16
  %194 = add nsw i64 %99, %60, !dbg !16
  %195 = add nsw i64 %194, 32, !dbg !16
  %196 = icmp slt i64 %194, -32, !dbg !16
  %197 = sub nsw i64 -33, %194, !dbg !16
  %198 = select i1 %196, i64 %197, i64 %195, !dbg !16
  %199 = sdiv i64 %198, 128, !dbg !16
  %200 = sext i1 %196 to i64, !dbg !16
  %201 = xor i64 %199, %200, !dbg !16
  %202 = shl nsw i64 %201, 7, !dbg !16
  %203 = sub i64 %60, %202, !dbg !16
  %204 = shl i64 %193, 19, !dbg !16
  %205 = and i64 %204, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %206 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %205, !dbg !16
  %207 = getelementptr [2 x i8], ptr addrspace(1) %206, i64 %99, !dbg !16
  %208 = getelementptr [2 x i8], ptr addrspace(1) %207, i64 %203, !dbg !16
  %209 = getelementptr i8, ptr addrspace(1) %208, i64 64, !dbg !16
  %210 = getelementptr [2 x i8], ptr addrspace(1) %209, i64 %105, !dbg !16
  %211 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !16
  %212 = zext i32 %211 to i64, !dbg !16
  %213 = inttoptr i64 %212 to ptr addrspace(3), !dbg !16
  %address.local705 = getelementptr [2 x i8], ptr addrspace(1) %210, i64 %address.local314, !dbg !16
  %214 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %215 = ptrtoint ptr addrspace(1) %address.local705 to i64, !dbg !16
  %216 = sub i64 %215, %214, !dbg !16
  %dma.buffer.offset1203 = trunc i64 %216 to i32, !dbg !16
  %217 = trunc i64 %214 to i32, !dbg !16
  %218 = insertelement <4 x i32> poison, i32 %217, i64 0, !dbg !16
  %219 = lshr i64 %214, 32, !dbg !16
  %220 = trunc i64 %219 to i32, !dbg !16
  %221 = insertelement <4 x i32> %218, i32 %220, i64 1, !dbg !16
  %222 = insertelement <4 x i32> %221, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1204 = insertelement <4 x i32> %222, i32 131072, i64 3, !dbg !16
  %223 = ptrtoint ptr addrspace(3) %213 to i32, !dbg !16
  %dma.buffer.lds1205 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %223), !dbg !16
  %224 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1205, i32 %dma.buffer.offset1203, <4 x i32> %dma.buffer.rsrc1204) #10, !dbg !16
  %225 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local321), !dbg !16
  %226 = zext i32 %225 to i64, !dbg !16
  %227 = inttoptr i64 %226 to ptr addrspace(3), !dbg !16
  %address.local706 = getelementptr [2 x i8], ptr addrspace(1) %210, i64 %address.local331, !dbg !16
  %228 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %229 = ptrtoint ptr addrspace(1) %address.local706 to i64, !dbg !16
  %230 = sub i64 %229, %228, !dbg !16
  %dma.buffer.offset1200 = trunc i64 %230 to i32, !dbg !16
  %231 = trunc i64 %228 to i32, !dbg !16
  %232 = insertelement <4 x i32> poison, i32 %231, i64 0, !dbg !16
  %233 = lshr i64 %228, 32, !dbg !16
  %234 = trunc i64 %233 to i32, !dbg !16
  %235 = insertelement <4 x i32> %232, i32 %234, i64 1, !dbg !16
  %236 = insertelement <4 x i32> %235, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1201 = insertelement <4 x i32> %236, i32 131072, i64 3, !dbg !16
  %237 = ptrtoint ptr addrspace(3) %227 to i32, !dbg !16
  %dma.buffer.lds1202 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %237), !dbg !16
  %238 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1202, i32 %dma.buffer.offset1200, <4 x i32> %dma.buffer.rsrc1201) #10, !dbg !16
  %239 = load <4 x half>, ptr addrspace(3) %address.local345, align 8, !dbg !13
  %lds.const.gep1116 = getelementptr i8, ptr addrspace(3) %address.local345, i64 2048, !dbg !13
  %240 = load <4 x half>, ptr addrspace(3) %lds.const.gep1116, align 8, !dbg !13
  %241 = fmul <4 x half> %239, splat (half 0xH3015), !dbg !13
  %242 = fmul <4 x half> %240, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %243 = load <4 x half>, ptr addrspace(3) %address.local366, align 8, !dbg !17
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %241, <4 x half> %243, <4 x float> %145) #9, !dbg !17
  %lds.const.gep1117 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4096, !dbg !17
  %245 = load <4 x half>, ptr addrspace(3) %lds.const.gep1117, align 8, !dbg !17
  %246 = load <4 x half>, ptr addrspace(3) %address.local395, align 8, !dbg !17
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %245, <4 x float> %244) #9, !dbg !17
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %241, <4 x half> %246, <4 x float> %149) #9, !dbg !17
  %lds.const.gep1118 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4096, !dbg !17
  %249 = load <4 x half>, ptr addrspace(3) %lds.const.gep1118, align 8, !dbg !17
  %lds.const.gep1119 = getelementptr i8, ptr addrspace(3) %address.local366, i64 128, !dbg !17
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep1119, align 8, !dbg !17
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %249, <4 x float> %248) #9, !dbg !17
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %241, <4 x half> %250, <4 x float> %152) #9, !dbg !17
  %lds.const.gep1120 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4224, !dbg !17
  %253 = load <4 x half>, ptr addrspace(3) %lds.const.gep1120, align 8, !dbg !17
  %lds.const.gep1121 = getelementptr i8, ptr addrspace(3) %address.local395, i64 128, !dbg !17
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1121, align 8, !dbg !17
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %253, <4 x float> %252) #9, !dbg !17
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %241, <4 x half> %254, <4 x float> %157) #9, !dbg !17
  %lds.const.gep1122 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4224, !dbg !17
  %257 = load <4 x half>, ptr addrspace(3) %lds.const.gep1122, align 8, !dbg !17
  %lds.const.gep1123 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1024, !dbg !17
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1123, align 8, !dbg !17
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %257, <4 x float> %256) #9, !dbg !17
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %241, <4 x half> %258, <4 x float> %160) #9, !dbg !17
  %lds.const.gep1124 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5120, !dbg !17
  %261 = load <4 x half>, ptr addrspace(3) %lds.const.gep1124, align 8, !dbg !17
  %lds.const.gep1125 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1024, !dbg !17
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1125, align 8, !dbg !17
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %261, <4 x float> %260) #9, !dbg !17
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %241, <4 x half> %262, <4 x float> %165) #9, !dbg !17
  %lds.const.gep1126 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5120, !dbg !17
  %265 = load <4 x half>, ptr addrspace(3) %lds.const.gep1126, align 8, !dbg !17
  %lds.const.gep1127 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1152, !dbg !17
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1127, align 8, !dbg !17
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %265, <4 x float> %264) #9, !dbg !17
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %241, <4 x half> %266, <4 x float> %168) #9, !dbg !17
  %lds.const.gep1128 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5248, !dbg !17
  %269 = load <4 x half>, ptr addrspace(3) %lds.const.gep1128, align 8, !dbg !17
  %lds.const.gep1129 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1152, !dbg !17
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1129, align 8, !dbg !17
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %269, <4 x float> %268) #9, !dbg !17
  %lds.const.gep1130 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5248, !dbg !17
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep1130, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %241, <4 x half> %270, <4 x float> %172) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %242, <4 x half> %272, <4 x float> %273) #9, !dbg !17
  %275 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local294), !dbg !12
  %276 = zext i32 %275 to i64, !dbg !12
  %277 = inttoptr i64 %276 to ptr addrspace(3), !dbg !12
  %address.local707 = getelementptr [2 x i8], ptr addrspace(1) %62, i64 %address.local299, !dbg !12
  %278 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %279 = ptrtoint ptr addrspace(1) %address.local707 to i64, !dbg !12
  %280 = sub i64 %279, %278, !dbg !12
  %dma.buffer.offset1197 = trunc i64 %280 to i32, !dbg !12
  %281 = trunc i64 %278 to i32, !dbg !12
  %282 = insertelement <4 x i32> poison, i32 %281, i64 0, !dbg !12
  %283 = lshr i64 %278, 32, !dbg !12
  %284 = trunc i64 %283 to i32, !dbg !12
  %285 = insertelement <4 x i32> %282, i32 %284, i64 1, !dbg !12
  %286 = insertelement <4 x i32> %285, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1198 = insertelement <4 x i32> %286, i32 131072, i64 3, !dbg !12
  %287 = ptrtoint ptr addrspace(3) %277 to i32, !dbg !12
  %dma.buffer.lds1199 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %287), !dbg !12
  %288 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1199, i32 %dma.buffer.offset1197, <4 x i32> %dma.buffer.rsrc1198) #10, !dbg !12
  %289 = add nuw i64 %66, 2048, !dbg !16
  %290 = icmp slt i64 %289, 0, !dbg !16
  %291 = sub nuw i64 -2049, %66, !dbg !16
  %292 = select i1 %290, i64 %291, i64 %289, !dbg !16
  %293 = sdiv i64 %292, 4096, !dbg !16
  %.lobit38 = ashr i64 %289, 63, !dbg !16
  %294 = xor i64 %293, %.lobit38, !dbg !16
  %295 = add nsw i64 %294, %15, !dbg !16
  %296 = add nsw i64 %194, 64, !dbg !16
  %297 = icmp slt i64 %194, -64, !dbg !16
  %298 = sub nsw i64 -65, %194, !dbg !16
  %299 = select i1 %297, i64 %298, i64 %296, !dbg !16
  %300 = sdiv i64 %299, 128, !dbg !16
  %301 = sext i1 %297 to i64, !dbg !16
  %302 = xor i64 %300, %301, !dbg !16
  %303 = shl nsw i64 %302, 7, !dbg !16
  %304 = sub i64 %60, %303, !dbg !16
  %305 = shl i64 %295, 19, !dbg !16
  %306 = and i64 %305, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %307 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %306, !dbg !16
  %308 = getelementptr [2 x i8], ptr addrspace(1) %307, i64 %99, !dbg !16
  %309 = getelementptr [2 x i8], ptr addrspace(1) %308, i64 %304, !dbg !16
  %310 = getelementptr i8, ptr addrspace(1) %309, i64 128, !dbg !16
  %311 = getelementptr [2 x i8], ptr addrspace(1) %310, i64 %105, !dbg !16
  %312 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !16
  %313 = zext i32 %312 to i64, !dbg !16
  %314 = inttoptr i64 %313 to ptr addrspace(3), !dbg !16
  %address.local708 = getelementptr [2 x i8], ptr addrspace(1) %311, i64 %address.local314, !dbg !16
  %315 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %316 = ptrtoint ptr addrspace(1) %address.local708 to i64, !dbg !16
  %317 = sub i64 %316, %315, !dbg !16
  %dma.buffer.offset1194 = trunc i64 %317 to i32, !dbg !16
  %318 = trunc i64 %315 to i32, !dbg !16
  %319 = insertelement <4 x i32> poison, i32 %318, i64 0, !dbg !16
  %320 = lshr i64 %315, 32, !dbg !16
  %321 = trunc i64 %320 to i32, !dbg !16
  %322 = insertelement <4 x i32> %319, i32 %321, i64 1, !dbg !16
  %323 = insertelement <4 x i32> %322, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1195 = insertelement <4 x i32> %323, i32 131072, i64 3, !dbg !16
  %324 = ptrtoint ptr addrspace(3) %314 to i32, !dbg !16
  %dma.buffer.lds1196 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %324), !dbg !16
  %325 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1196, i32 %dma.buffer.offset1194, <4 x i32> %dma.buffer.rsrc1195) #10, !dbg !16
  %326 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local321), !dbg !16
  %327 = zext i32 %326 to i64, !dbg !16
  %328 = inttoptr i64 %327 to ptr addrspace(3), !dbg !16
  %address.local709 = getelementptr [2 x i8], ptr addrspace(1) %311, i64 %address.local331, !dbg !16
  %329 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %330 = ptrtoint ptr addrspace(1) %address.local709 to i64, !dbg !16
  %331 = sub i64 %330, %329, !dbg !16
  %dma.buffer.offset1191 = trunc i64 %331 to i32, !dbg !16
  %332 = trunc i64 %329 to i32, !dbg !16
  %333 = insertelement <4 x i32> poison, i32 %332, i64 0, !dbg !16
  %334 = lshr i64 %329, 32, !dbg !16
  %335 = trunc i64 %334 to i32, !dbg !16
  %336 = insertelement <4 x i32> %333, i32 %335, i64 1, !dbg !16
  %337 = insertelement <4 x i32> %336, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1192 = insertelement <4 x i32> %337, i32 131072, i64 3, !dbg !16
  %338 = ptrtoint ptr addrspace(3) %328 to i32, !dbg !16
  %dma.buffer.lds1193 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %338), !dbg !16
  %339 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1193, i32 %dma.buffer.offset1191, <4 x i32> %dma.buffer.rsrc1192) #10, !dbg !16
  %340 = load <4 x half>, ptr addrspace(3) %address.local345, align 8, !dbg !13
  %lds.const.gep1131 = getelementptr i8, ptr addrspace(3) %address.local345, i64 2048, !dbg !13
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1131, align 8, !dbg !13
  %342 = fmul <4 x half> %340, splat (half 0xH3015), !dbg !13
  %343 = fmul <4 x half> %341, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %344 = load <4 x half>, ptr addrspace(3) %address.local366, align 8, !dbg !17
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %344, <4 x float> %247) #9, !dbg !17
  %lds.const.gep1132 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4096, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1132, align 8, !dbg !17
  %347 = load <4 x half>, ptr addrspace(3) %address.local395, align 8, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %346, <4 x float> %345) #9, !dbg !17
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %347, <4 x float> %251) #9, !dbg !17
  %lds.const.gep1133 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4096, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1133, align 8, !dbg !17
  %lds.const.gep1134 = getelementptr i8, ptr addrspace(3) %address.local366, i64 128, !dbg !17
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1134, align 8, !dbg !17
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %350, <4 x float> %349) #9, !dbg !17
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %351, <4 x float> %255) #9, !dbg !17
  %lds.const.gep1135 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4224, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1135, align 8, !dbg !17
  %lds.const.gep1136 = getelementptr i8, ptr addrspace(3) %address.local395, i64 128, !dbg !17
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1136, align 8, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %354, <4 x float> %353) #9, !dbg !17
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %355, <4 x float> %259) #9, !dbg !17
  %lds.const.gep1137 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4224, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1137, align 8, !dbg !17
  %lds.const.gep1138 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1024, !dbg !17
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1138, align 8, !dbg !17
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %358, <4 x float> %357) #9, !dbg !17
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %359, <4 x float> %263) #9, !dbg !17
  %lds.const.gep1139 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5120, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1139, align 8, !dbg !17
  %lds.const.gep1140 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1024, !dbg !17
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1140, align 8, !dbg !17
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %362, <4 x float> %361) #9, !dbg !17
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %363, <4 x float> %267) #9, !dbg !17
  %lds.const.gep1141 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5120, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1141, align 8, !dbg !17
  %lds.const.gep1142 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1152, !dbg !17
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1142, align 8, !dbg !17
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %366, <4 x float> %365) #9, !dbg !17
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %367, <4 x float> %271) #9, !dbg !17
  %lds.const.gep1143 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5248, !dbg !17
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep1143, align 8, !dbg !17
  %lds.const.gep1144 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1152, !dbg !17
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1144, align 8, !dbg !17
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %370, <4 x float> %369) #9, !dbg !17
  %lds.const.gep1145 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5248, !dbg !17
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep1145, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %371, <4 x float> %274) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %373, <4 x float> %374) #9, !dbg !17
  %376 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local294), !dbg !12
  %377 = zext i32 %376 to i64, !dbg !12
  %378 = inttoptr i64 %377 to ptr addrspace(3), !dbg !12
  %address.local710 = getelementptr [2 x i8], ptr addrspace(1) %64, i64 %address.local299, !dbg !12
  %379 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %380 = ptrtoint ptr addrspace(1) %address.local710 to i64, !dbg !12
  %381 = sub i64 %380, %379, !dbg !12
  %dma.buffer.offset1188 = trunc i64 %381 to i32, !dbg !12
  %382 = trunc i64 %379 to i32, !dbg !12
  %383 = insertelement <4 x i32> poison, i32 %382, i64 0, !dbg !12
  %384 = lshr i64 %379, 32, !dbg !12
  %385 = trunc i64 %384 to i32, !dbg !12
  %386 = insertelement <4 x i32> %383, i32 %385, i64 1, !dbg !12
  %387 = insertelement <4 x i32> %386, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1189 = insertelement <4 x i32> %387, i32 131072, i64 3, !dbg !12
  %388 = ptrtoint ptr addrspace(3) %378 to i32, !dbg !12
  %dma.buffer.lds1190 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %388), !dbg !12
  %389 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1190, i32 %dma.buffer.offset1188, <4 x i32> %dma.buffer.rsrc1189) #10, !dbg !12
  %390 = add nuw i64 %66, 3072, !dbg !16
  %391 = icmp slt i64 %390, 0, !dbg !16
  %392 = sub nuw i64 -3073, %66, !dbg !16
  %393 = select i1 %391, i64 %392, i64 %390, !dbg !16
  %394 = sdiv i64 %393, 4096, !dbg !16
  %.lobit39 = ashr i64 %390, 63, !dbg !16
  %395 = xor i64 %394, %.lobit39, !dbg !16
  %396 = add nsw i64 %395, %15, !dbg !16
  %397 = add nsw i64 %194, 96, !dbg !16
  %398 = icmp slt i64 %194, -96, !dbg !16
  %399 = sub nsw i64 -97, %194, !dbg !16
  %400 = select i1 %398, i64 %399, i64 %397, !dbg !16
  %401 = sdiv i64 %400, 128, !dbg !16
  %402 = sext i1 %398 to i64, !dbg !16
  %403 = xor i64 %401, %402, !dbg !16
  %404 = shl nsw i64 %403, 7, !dbg !16
  %405 = sub i64 %60, %404, !dbg !16
  %406 = shl i64 %396, 19, !dbg !16
  %407 = and i64 %406, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %408 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %407, !dbg !16
  %409 = getelementptr [2 x i8], ptr addrspace(1) %408, i64 %99, !dbg !16
  %410 = getelementptr [2 x i8], ptr addrspace(1) %409, i64 %405, !dbg !16
  %411 = getelementptr i8, ptr addrspace(1) %410, i64 192, !dbg !16
  %412 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %105, !dbg !16
  %413 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !16
  %414 = zext i32 %413 to i64, !dbg !16
  %415 = inttoptr i64 %414 to ptr addrspace(3), !dbg !16
  %address.local711 = getelementptr [2 x i8], ptr addrspace(1) %412, i64 %address.local314, !dbg !16
  %416 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %417 = ptrtoint ptr addrspace(1) %address.local711 to i64, !dbg !16
  %418 = sub i64 %417, %416, !dbg !16
  %dma.buffer.offset1185 = trunc i64 %418 to i32, !dbg !16
  %419 = trunc i64 %416 to i32, !dbg !16
  %420 = insertelement <4 x i32> poison, i32 %419, i64 0, !dbg !16
  %421 = lshr i64 %416, 32, !dbg !16
  %422 = trunc i64 %421 to i32, !dbg !16
  %423 = insertelement <4 x i32> %420, i32 %422, i64 1, !dbg !16
  %424 = insertelement <4 x i32> %423, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1186 = insertelement <4 x i32> %424, i32 131072, i64 3, !dbg !16
  %425 = ptrtoint ptr addrspace(3) %415 to i32, !dbg !16
  %dma.buffer.lds1187 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %425), !dbg !16
  %426 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1187, i32 %dma.buffer.offset1185, <4 x i32> %dma.buffer.rsrc1186) #10, !dbg !16
  %427 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local321), !dbg !16
  %428 = zext i32 %427 to i64, !dbg !16
  %429 = inttoptr i64 %428 to ptr addrspace(3), !dbg !16
  %address.local712 = getelementptr [2 x i8], ptr addrspace(1) %412, i64 %address.local331, !dbg !16
  %430 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %431 = ptrtoint ptr addrspace(1) %address.local712 to i64, !dbg !16
  %432 = sub i64 %431, %430, !dbg !16
  %dma.buffer.offset1182 = trunc i64 %432 to i32, !dbg !16
  %433 = trunc i64 %430 to i32, !dbg !16
  %434 = insertelement <4 x i32> poison, i32 %433, i64 0, !dbg !16
  %435 = lshr i64 %430, 32, !dbg !16
  %436 = trunc i64 %435 to i32, !dbg !16
  %437 = insertelement <4 x i32> %434, i32 %436, i64 1, !dbg !16
  %438 = insertelement <4 x i32> %437, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1183 = insertelement <4 x i32> %438, i32 131072, i64 3, !dbg !16
  %439 = ptrtoint ptr addrspace(3) %429 to i32, !dbg !16
  %dma.buffer.lds1184 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %439), !dbg !16
  %440 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1184, i32 %dma.buffer.offset1182, <4 x i32> %dma.buffer.rsrc1183) #10, !dbg !16
  %441 = load <4 x half>, ptr addrspace(3) %address.local345, align 8, !dbg !13
  %lds.const.gep1146 = getelementptr i8, ptr addrspace(3) %address.local345, i64 2048, !dbg !13
  %442 = load <4 x half>, ptr addrspace(3) %lds.const.gep1146, align 8, !dbg !13
  %443 = fmul <4 x half> %441, splat (half 0xH3015), !dbg !13
  %444 = fmul <4 x half> %442, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %445 = add nuw nsw i64 %99, %15, !dbg !18
  %446 = shl i64 %445, 19, !dbg !18
  %447 = and i64 %446, 16252928, !dbg !18
  %448 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %447, !dbg !18
  %449 = getelementptr [2 x i8], ptr addrspace(1) %448, i64 %105, !dbg !18
  %address.local713 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local301, !dbg !18
  %address.local714 = ptrtoint ptr addrspace(3) %address.local713 to i64, !dbg !18
  %address.local715 = trunc i64 %address.local714 to i32, !dbg !18
  %450 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local715), !dbg !18
  %451 = zext i32 %450 to i64, !dbg !18
  %452 = inttoptr i64 %451 to ptr addrspace(3), !dbg !18
  %address.local716 = shl nuw nsw i64 %address.local, 5, !dbg !18
  %address.local717 = and i64 %address.local716, 8064, !dbg !18
  %address.local718 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local719 = and i64 %address.local718, 24, !dbg !18
  %address.local720 = or disjoint i64 %address.local717, %address.local719, !dbg !18
  %address.local721 = getelementptr [2 x i8], ptr addrspace(1) %449, i64 %address.local720, !dbg !18
  %453 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %454 = ptrtoint ptr addrspace(1) %address.local721 to i64, !dbg !18
  %455 = sub i64 %454, %453, !dbg !18
  %dma.buffer.offset1179 = trunc i64 %455 to i32, !dbg !18
  %456 = trunc i64 %453 to i32, !dbg !18
  %457 = insertelement <4 x i32> poison, i32 %456, i64 0, !dbg !18
  %458 = lshr i64 %453, 32, !dbg !18
  %459 = trunc i64 %458 to i32, !dbg !18
  %460 = insertelement <4 x i32> %457, i32 %459, i64 1, !dbg !18
  %461 = insertelement <4 x i32> %460, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1180 = insertelement <4 x i32> %461, i32 131072, i64 3, !dbg !18
  %462 = ptrtoint ptr addrspace(3) %452 to i32, !dbg !18
  %dma.buffer.lds1181 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %462), !dbg !18
  %463 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1181, i32 %dma.buffer.offset1179, <4 x i32> %dma.buffer.rsrc1180) #10, !dbg !18
  %address.local722 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local318, !dbg !18
  %address.local723 = ptrtoint ptr addrspace(3) %address.local722 to i64, !dbg !18
  %address.local724 = trunc i64 %address.local723 to i32, !dbg !18
  %464 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local724), !dbg !18
  %465 = zext i32 %464 to i64, !dbg !18
  %466 = inttoptr i64 %465 to ptr addrspace(3), !dbg !18
  %address.local725 = shl nuw nsw i64 %address.local316, 5, !dbg !18
  %address.local726 = and i64 %address.local725, 16256, !dbg !18
  %address.local727 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local728 = and i64 %address.local727, 24, !dbg !18
  %address.local729 = or disjoint i64 %address.local726, %address.local728, !dbg !18
  %address.local730 = getelementptr [2 x i8], ptr addrspace(1) %449, i64 %address.local729, !dbg !18
  %467 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %468 = ptrtoint ptr addrspace(1) %address.local730 to i64, !dbg !18
  %469 = sub i64 %468, %467, !dbg !18
  %dma.buffer.offset1176 = trunc i64 %469 to i32, !dbg !18
  %470 = trunc i64 %467 to i32, !dbg !18
  %471 = insertelement <4 x i32> poison, i32 %470, i64 0, !dbg !18
  %472 = lshr i64 %467, 32, !dbg !18
  %473 = trunc i64 %472 to i32, !dbg !18
  %474 = insertelement <4 x i32> %471, i32 %473, i64 1, !dbg !18
  %475 = insertelement <4 x i32> %474, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1177 = insertelement <4 x i32> %475, i32 131072, i64 3, !dbg !18
  %476 = ptrtoint ptr addrspace(3) %466 to i32, !dbg !18
  %dma.buffer.lds1178 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %476), !dbg !18
  %477 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1178, i32 %dma.buffer.offset1176, <4 x i32> %dma.buffer.rsrc1177) #10, !dbg !18
  %478 = load <4 x half>, ptr addrspace(3) %address.local366, align 8, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %443, <4 x half> %478, <4 x float> %348) #9, !dbg !17
  %lds.const.gep1147 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4096, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1147, align 8, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %address.local395, align 8, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %480, <4 x float> %479) #9, !dbg !17
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %443, <4 x half> %481, <4 x float> %352) #9, !dbg !17
  %lds.const.gep1148 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4096, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1148, align 8, !dbg !17
  %lds.const.gep1149 = getelementptr i8, ptr addrspace(3) %address.local366, i64 128, !dbg !17
  %485 = load <4 x half>, ptr addrspace(3) %lds.const.gep1149, align 8, !dbg !17
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %484, <4 x float> %483) #9, !dbg !17
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %443, <4 x half> %485, <4 x float> %356) #9, !dbg !17
  %lds.const.gep1150 = getelementptr i8, ptr addrspace(3) %address.local366, i64 4224, !dbg !17
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep1150, align 8, !dbg !17
  %lds.const.gep1151 = getelementptr i8, ptr addrspace(3) %address.local395, i64 128, !dbg !17
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep1151, align 8, !dbg !17
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %488, <4 x float> %487) #9, !dbg !17
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %443, <4 x half> %489, <4 x float> %360) #9, !dbg !17
  %lds.const.gep1152 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4224, !dbg !17
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep1152, align 8, !dbg !17
  %lds.const.gep1153 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1024, !dbg !17
  %493 = load <4 x half>, ptr addrspace(3) %lds.const.gep1153, align 8, !dbg !17
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %492, <4 x float> %491) #9, !dbg !17
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %443, <4 x half> %493, <4 x float> %364) #9, !dbg !17
  %lds.const.gep1154 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5120, !dbg !17
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep1154, align 8, !dbg !17
  %lds.const.gep1155 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1024, !dbg !17
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep1155, align 8, !dbg !17
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %496, <4 x float> %495) #9, !dbg !17
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %443, <4 x half> %497, <4 x float> %368) #9, !dbg !17
  %lds.const.gep1156 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5120, !dbg !17
  %500 = load <4 x half>, ptr addrspace(3) %lds.const.gep1156, align 8, !dbg !17
  %lds.const.gep1157 = getelementptr i8, ptr addrspace(3) %address.local366, i64 1152, !dbg !17
  %501 = load <4 x half>, ptr addrspace(3) %lds.const.gep1157, align 8, !dbg !17
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %500, <4 x float> %499) #9, !dbg !17
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %443, <4 x half> %501, <4 x float> %372) #9, !dbg !17
  %lds.const.gep1158 = getelementptr i8, ptr addrspace(3) %address.local366, i64 5248, !dbg !17
  %504 = load <4 x half>, ptr addrspace(3) %lds.const.gep1158, align 8, !dbg !17
  %lds.const.gep1159 = getelementptr i8, ptr addrspace(3) %address.local395, i64 1152, !dbg !17
  %505 = load <4 x half>, ptr addrspace(3) %lds.const.gep1159, align 8, !dbg !17
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %504, <4 x float> %503) #9, !dbg !17
  %lds.const.gep1160 = getelementptr i8, ptr addrspace(3) %address.local395, i64 5248, !dbg !17
  %507 = load <4 x half>, ptr addrspace(3) %lds.const.gep1160, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %443, <4 x half> %505, <4 x float> %375) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %509 = or disjoint i64 %82, 127, !dbg !14
  %510 = icmp uge i64 %19, %509, !dbg !14
  %511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %507, <4 x float> %508) #9, !dbg !17
  %512 = and i1 %39, %510, !dbg !14
  br i1 %512, label %513, label %514, !dbg !14

513:                                              ; preds = %81
  br label %589, !dbg !14

514:                                              ; preds = %81
  %515 = add i64 %19, %35, !dbg !14
  %516 = shl nuw nsw i64 %36, 4, !dbg !14
  %517 = add i64 %515, %516, !dbg !14
  %518 = and i32 %34, 240, !dbg !14
  %index.compact = zext nneg i32 %518 to i64, !dbg !14
  %519 = sub i64 %517, %index.compact, !dbg !14
  %520 = add i64 %519, 1, !dbg !23
  %521 = and i64 %35, 48, !dbg !14
  %522 = or disjoint i64 %82, %521, !dbg !14
  %523 = lshr i64 %520, 32, !dbg !24
  %compare.high = trunc i64 %523 to i32, !dbg !24
  %524 = lshr i64 %522, 32, !dbg !24
  %compare.high978 = trunc i64 %524 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high978, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high978, !dbg !24
  %compare.low.base = trunc i64 %520 to i32, !dbg !24
  %compare.low.base979 = trunc i64 %522 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base979, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %525 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low980 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result981 = select i1 %compare.high.equal, i1 %compare.low980, i1 %compare.high.order, !dbg !24
  %526 = select i1 %compare.result981, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base982 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local983 = or i32 %compare.local.base982, 2, !dbg !24
  %compare.low984 = icmp ugt i32 %compare.low.base, %compare.local983, !dbg !24
  %compare.result985 = select i1 %compare.high.equal, i1 %compare.low984, i1 %compare.high.order, !dbg !24
  %527 = select i1 %compare.result985, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base986 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local987 = or i32 %compare.local.base986, 3, !dbg !24
  %compare.low988 = icmp ugt i32 %compare.low.base, %compare.local987, !dbg !24
  %compare.result989 = select i1 %compare.high.equal, i1 %compare.low988, i1 %compare.high.order, !dbg !24
  %528 = select i1 %compare.result989, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base990 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local991 = or i32 %compare.local.base990, 4, !dbg !24
  %compare.low992 = icmp ugt i32 %compare.low.base, %compare.local991, !dbg !24
  %compare.result993 = select i1 %compare.high.equal, i1 %compare.low992, i1 %compare.high.order, !dbg !24
  %529 = select i1 %compare.result993, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base994 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local995 = or i32 %compare.local.base994, 5, !dbg !24
  %compare.low996 = icmp ugt i32 %compare.low.base, %compare.local995, !dbg !24
  %compare.result997 = select i1 %compare.high.equal, i1 %compare.low996, i1 %compare.high.order, !dbg !24
  %530 = select i1 %compare.result997, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base998 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local999 = or i32 %compare.local.base998, 6, !dbg !24
  %compare.low1000 = icmp ugt i32 %compare.low.base, %compare.local999, !dbg !24
  %compare.result1001 = select i1 %compare.high.equal, i1 %compare.low1000, i1 %compare.high.order, !dbg !24
  %531 = select i1 %compare.result1001, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1002 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1003 = or i32 %compare.local.base1002, 7, !dbg !24
  %compare.low1004 = icmp ugt i32 %compare.low.base, %compare.local1003, !dbg !24
  %compare.result1005 = select i1 %compare.high.equal, i1 %compare.low1004, i1 %compare.high.order, !dbg !24
  %532 = select i1 %compare.result1005, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1006 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1007 = or i32 %compare.local.base1006, 8, !dbg !24
  %compare.low1008 = icmp ugt i32 %compare.low.base, %compare.local1007, !dbg !24
  %compare.result1009 = select i1 %compare.high.equal, i1 %compare.low1008, i1 %compare.high.order, !dbg !24
  %533 = select i1 %compare.result1009, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1010 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1011 = or i32 %compare.local.base1010, 9, !dbg !24
  %compare.low1012 = icmp ugt i32 %compare.low.base, %compare.local1011, !dbg !24
  %compare.result1013 = select i1 %compare.high.equal, i1 %compare.low1012, i1 %compare.high.order, !dbg !24
  %534 = select i1 %compare.result1013, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1014 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1015 = or i32 %compare.local.base1014, 10, !dbg !24
  %compare.low1016 = icmp ugt i32 %compare.low.base, %compare.local1015, !dbg !24
  %compare.result1017 = select i1 %compare.high.equal, i1 %compare.low1016, i1 %compare.high.order, !dbg !24
  %535 = select i1 %compare.result1017, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1018 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1019 = or i32 %compare.local.base1018, 11, !dbg !24
  %compare.low1020 = icmp ugt i32 %compare.low.base, %compare.local1019, !dbg !24
  %compare.result1021 = select i1 %compare.high.equal, i1 %compare.low1020, i1 %compare.high.order, !dbg !24
  %536 = select i1 %compare.result1021, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1022 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1023 = or i32 %compare.local.base1022, 12, !dbg !24
  %compare.low1024 = icmp ugt i32 %compare.low.base, %compare.local1023, !dbg !24
  %compare.result1025 = select i1 %compare.high.equal, i1 %compare.low1024, i1 %compare.high.order, !dbg !24
  %537 = select i1 %compare.result1025, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1026 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1027 = or i32 %compare.local.base1026, 13, !dbg !24
  %compare.low1028 = icmp ugt i32 %compare.low.base, %compare.local1027, !dbg !24
  %compare.result1029 = select i1 %compare.high.equal, i1 %compare.low1028, i1 %compare.high.order, !dbg !24
  %538 = select i1 %compare.result1029, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1030 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1031 = or i32 %compare.local.base1030, 14, !dbg !24
  %compare.low1032 = icmp ugt i32 %compare.low.base, %compare.local1031, !dbg !24
  %compare.result1033 = select i1 %compare.high.equal, i1 %compare.low1032, i1 %compare.high.order, !dbg !24
  %539 = select i1 %compare.result1033, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1034 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1035 = or i32 %compare.local.base1034, 15, !dbg !24
  %compare.low1036 = icmp ugt i32 %compare.low.base, %compare.local1035, !dbg !24
  %compare.result1037 = select i1 %compare.high.equal, i1 %compare.low1036, i1 %compare.high.order, !dbg !24
  %540 = select i1 %compare.result1037, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1038 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1039 = or i32 %compare.local.base1038, 64, !dbg !24
  %compare.low1040 = icmp ugt i32 %compare.low.base, %compare.local1039, !dbg !24
  %compare.result1041 = select i1 %compare.high.equal, i1 %compare.low1040, i1 %compare.high.order, !dbg !24
  %541 = select i1 %compare.result1041, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1042 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1043 = or i32 %compare.local.base1042, 65, !dbg !24
  %compare.low1044 = icmp ugt i32 %compare.low.base, %compare.local1043, !dbg !24
  %compare.result1045 = select i1 %compare.high.equal, i1 %compare.low1044, i1 %compare.high.order, !dbg !24
  %542 = select i1 %compare.result1045, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1046 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1047 = or i32 %compare.local.base1046, 66, !dbg !24
  %compare.low1048 = icmp ugt i32 %compare.low.base, %compare.local1047, !dbg !24
  %compare.result1049 = select i1 %compare.high.equal, i1 %compare.low1048, i1 %compare.high.order, !dbg !24
  %543 = select i1 %compare.result1049, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1050 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1051 = or i32 %compare.local.base1050, 67, !dbg !24
  %compare.low1052 = icmp ugt i32 %compare.low.base, %compare.local1051, !dbg !24
  %compare.result1053 = select i1 %compare.high.equal, i1 %compare.low1052, i1 %compare.high.order, !dbg !24
  %544 = select i1 %compare.result1053, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1054 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1055 = or i32 %compare.local.base1054, 68, !dbg !24
  %compare.low1056 = icmp ugt i32 %compare.low.base, %compare.local1055, !dbg !24
  %compare.result1057 = select i1 %compare.high.equal, i1 %compare.low1056, i1 %compare.high.order, !dbg !24
  %545 = select i1 %compare.result1057, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1058 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1059 = or i32 %compare.local.base1058, 69, !dbg !24
  %compare.low1060 = icmp ugt i32 %compare.low.base, %compare.local1059, !dbg !24
  %compare.result1061 = select i1 %compare.high.equal, i1 %compare.low1060, i1 %compare.high.order, !dbg !24
  %546 = select i1 %compare.result1061, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1062 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1063 = or i32 %compare.local.base1062, 70, !dbg !24
  %compare.low1064 = icmp ugt i32 %compare.low.base, %compare.local1063, !dbg !24
  %compare.result1065 = select i1 %compare.high.equal, i1 %compare.low1064, i1 %compare.high.order, !dbg !24
  %547 = select i1 %compare.result1065, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1066 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1067 = or i32 %compare.local.base1066, 71, !dbg !24
  %compare.low1068 = icmp ugt i32 %compare.low.base, %compare.local1067, !dbg !24
  %compare.result1069 = select i1 %compare.high.equal, i1 %compare.low1068, i1 %compare.high.order, !dbg !24
  %548 = select i1 %compare.result1069, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1070 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1071 = or i32 %compare.local.base1070, 72, !dbg !24
  %compare.low1072 = icmp ugt i32 %compare.low.base, %compare.local1071, !dbg !24
  %compare.result1073 = select i1 %compare.high.equal, i1 %compare.low1072, i1 %compare.high.order, !dbg !24
  %549 = select i1 %compare.result1073, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1074 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1075 = or i32 %compare.local.base1074, 73, !dbg !24
  %compare.low1076 = icmp ugt i32 %compare.low.base, %compare.local1075, !dbg !24
  %compare.result1077 = select i1 %compare.high.equal, i1 %compare.low1076, i1 %compare.high.order, !dbg !24
  %550 = select i1 %compare.result1077, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1078 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1079 = or i32 %compare.local.base1078, 74, !dbg !24
  %compare.low1080 = icmp ugt i32 %compare.low.base, %compare.local1079, !dbg !24
  %compare.result1081 = select i1 %compare.high.equal, i1 %compare.low1080, i1 %compare.high.order, !dbg !24
  %551 = select i1 %compare.result1081, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1082 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1083 = or i32 %compare.local.base1082, 75, !dbg !24
  %compare.low1084 = icmp ugt i32 %compare.low.base, %compare.local1083, !dbg !24
  %compare.result1085 = select i1 %compare.high.equal, i1 %compare.low1084, i1 %compare.high.order, !dbg !24
  %552 = select i1 %compare.result1085, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1086 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1087 = or i32 %compare.local.base1086, 76, !dbg !24
  %compare.low1088 = icmp ugt i32 %compare.low.base, %compare.local1087, !dbg !24
  %compare.result1089 = select i1 %compare.high.equal, i1 %compare.low1088, i1 %compare.high.order, !dbg !24
  %553 = select i1 %compare.result1089, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1090 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1091 = or i32 %compare.local.base1090, 77, !dbg !24
  %compare.low1092 = icmp ugt i32 %compare.low.base, %compare.local1091, !dbg !24
  %compare.result1093 = select i1 %compare.high.equal, i1 %compare.low1092, i1 %compare.high.order, !dbg !24
  %554 = select i1 %compare.result1093, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1094 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1095 = or i32 %compare.local.base1094, 78, !dbg !24
  %compare.low1096 = icmp ugt i32 %compare.low.base, %compare.local1095, !dbg !24
  %compare.result1097 = select i1 %compare.high.equal, i1 %compare.low1096, i1 %compare.high.order, !dbg !24
  %555 = select i1 %compare.result1097, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1098 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base979, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1099 = or i32 %compare.local.base1098, 79, !dbg !24
  %compare.low1100 = icmp ugt i32 %compare.low.base, %compare.local1099, !dbg !24
  %compare.result1101 = select i1 %compare.high.equal, i1 %compare.low1100, i1 %compare.high.order, !dbg !24
  %556 = select i1 %compare.result1101, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %557 = insertelement <32 x float> poison, float %525, i64 0, !dbg !14
  %558 = insertelement <32 x float> %557, float %526, i64 1, !dbg !14
  %559 = insertelement <32 x float> %558, float %527, i64 2, !dbg !14
  %560 = insertelement <32 x float> %559, float %528, i64 3, !dbg !14
  %561 = insertelement <32 x float> %560, float %529, i64 4, !dbg !14
  %562 = insertelement <32 x float> %561, float %530, i64 5, !dbg !14
  %563 = insertelement <32 x float> %562, float %531, i64 6, !dbg !14
  %564 = insertelement <32 x float> %563, float %532, i64 7, !dbg !14
  %565 = insertelement <32 x float> %564, float %533, i64 8, !dbg !14
  %566 = insertelement <32 x float> %565, float %534, i64 9, !dbg !14
  %567 = insertelement <32 x float> %566, float %535, i64 10, !dbg !14
  %568 = insertelement <32 x float> %567, float %536, i64 11, !dbg !14
  %569 = insertelement <32 x float> %568, float %537, i64 12, !dbg !14
  %570 = insertelement <32 x float> %569, float %538, i64 13, !dbg !14
  %571 = insertelement <32 x float> %570, float %539, i64 14, !dbg !14
  %572 = insertelement <32 x float> %571, float %540, i64 15, !dbg !14
  %573 = insertelement <32 x float> %572, float %541, i64 16, !dbg !14
  %574 = insertelement <32 x float> %573, float %542, i64 17, !dbg !14
  %575 = insertelement <32 x float> %574, float %543, i64 18, !dbg !14
  %576 = insertelement <32 x float> %575, float %544, i64 19, !dbg !14
  %577 = insertelement <32 x float> %576, float %545, i64 20, !dbg !14
  %578 = insertelement <32 x float> %577, float %546, i64 21, !dbg !14
  %579 = insertelement <32 x float> %578, float %547, i64 22, !dbg !14
  %580 = insertelement <32 x float> %579, float %548, i64 23, !dbg !14
  %581 = insertelement <32 x float> %580, float %549, i64 24, !dbg !14
  %582 = insertelement <32 x float> %581, float %550, i64 25, !dbg !14
  %583 = insertelement <32 x float> %582, float %551, i64 26, !dbg !14
  %584 = insertelement <32 x float> %583, float %552, i64 27, !dbg !14
  %585 = insertelement <32 x float> %584, float %553, i64 28, !dbg !14
  %586 = insertelement <32 x float> %585, float %554, i64 29, !dbg !14
  %587 = insertelement <32 x float> %586, float %555, i64 30, !dbg !14
  %588 = insertelement <32 x float> %587, float %556, i64 31, !dbg !14
  br label %589, !dbg !14

589:                                              ; preds = %513, %514
  %590 = phi <32 x float> [ %588, %514 ], [ zeroinitializer, %513 ], !dbg !14
  br label %591, !dbg !14

591:                                              ; preds = %589
  %592 = shufflevector <32 x float> %590, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %593 = fadd <4 x float> %482, %592, !dbg !26
  %594 = extractelement <4 x float> %593, i64 0, !dbg !27
  %595 = call float @llvm.amdgcn.exp2.f32(float %594), !dbg !27
  %596 = insertelement <4 x float> poison, float %595, i64 0, !dbg !29
  %597 = extractelement <4 x float> %593, i64 1, !dbg !27
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !27
  %599 = insertelement <4 x float> %596, float %598, i64 1, !dbg !29
  %600 = fadd float %595, %598, !dbg !28
  %601 = extractelement <4 x float> %593, i64 2, !dbg !27
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !27
  %603 = insertelement <4 x float> %599, float %602, i64 2, !dbg !29
  %604 = fadd float %600, %602, !dbg !28
  %605 = extractelement <4 x float> %593, i64 3, !dbg !27
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !27
  %607 = insertelement <4 x float> %603, float %606, i64 3, !dbg !29
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %607) #9, !dbg !29, !frisk.register_lifetimes !1
  %608 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %608) #9, !dbg !28, !frisk.register_lifetimes !1
  %609 = fadd float %604, %606, !dbg !28
  %610 = shufflevector <32 x float> %590, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %611 = fadd <4 x float> %486, %610, !dbg !26
  %612 = extractelement <4 x float> %611, i64 0, !dbg !27
  %613 = call float @llvm.amdgcn.exp2.f32(float %612), !dbg !27
  %614 = insertelement <4 x float> poison, float %613, i64 0, !dbg !29
  %615 = fadd float %609, %613, !dbg !28
  %616 = extractelement <4 x float> %611, i64 1, !dbg !27
  %617 = call float @llvm.amdgcn.exp2.f32(float %616), !dbg !27
  %618 = insertelement <4 x float> %614, float %617, i64 1, !dbg !29
  %619 = fadd float %615, %617, !dbg !28
  %620 = extractelement <4 x float> %611, i64 2, !dbg !27
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !27
  %622 = insertelement <4 x float> %618, float %621, i64 2, !dbg !29
  %623 = fadd float %619, %621, !dbg !28
  %624 = extractelement <4 x float> %611, i64 3, !dbg !27
  %625 = call float @llvm.amdgcn.exp2.f32(float %624), !dbg !27
  %626 = insertelement <4 x float> %622, float %625, i64 3, !dbg !29
  %math.result276 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %626) #9, !dbg !29, !frisk.register_lifetimes !1
  %627 = fptrunc <4 x float> %math.result276 to <4 x half>, !dbg !29
  %math.packed277 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %627) #9, !dbg !28, !frisk.register_lifetimes !1
  %628 = fadd float %623, %625, !dbg !28
  %629 = shufflevector <32 x float> %590, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %630 = fadd <4 x float> %490, %629, !dbg !26
  %631 = extractelement <4 x float> %630, i64 0, !dbg !27
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !27
  %633 = insertelement <4 x float> poison, float %632, i64 0, !dbg !29
  %634 = fadd float %628, %632, !dbg !28
  %635 = extractelement <4 x float> %630, i64 1, !dbg !27
  %636 = call float @llvm.amdgcn.exp2.f32(float %635), !dbg !27
  %637 = insertelement <4 x float> %633, float %636, i64 1, !dbg !29
  %638 = fadd float %634, %636, !dbg !28
  %639 = extractelement <4 x float> %630, i64 2, !dbg !27
  %640 = call float @llvm.amdgcn.exp2.f32(float %639), !dbg !27
  %641 = insertelement <4 x float> %637, float %640, i64 2, !dbg !29
  %642 = fadd float %638, %640, !dbg !28
  %643 = extractelement <4 x float> %630, i64 3, !dbg !27
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !27
  %645 = insertelement <4 x float> %641, float %644, i64 3, !dbg !29
  %math.result278 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %645) #9, !dbg !29, !frisk.register_lifetimes !1
  %646 = fptrunc <4 x float> %math.result278 to <4 x half>, !dbg !29
  %math.packed279 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %646) #9, !dbg !28, !frisk.register_lifetimes !1
  %647 = fadd float %642, %644, !dbg !28
  %648 = shufflevector <32 x float> %590, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %649 = fadd <4 x float> %494, %648, !dbg !26
  %650 = extractelement <4 x float> %649, i64 0, !dbg !27
  %651 = call float @llvm.amdgcn.exp2.f32(float %650), !dbg !27
  %652 = insertelement <4 x float> poison, float %651, i64 0, !dbg !29
  %653 = fadd float %647, %651, !dbg !28
  %654 = extractelement <4 x float> %649, i64 1, !dbg !27
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !27
  %656 = insertelement <4 x float> %652, float %655, i64 1, !dbg !29
  %657 = fadd float %653, %655, !dbg !28
  %658 = extractelement <4 x float> %649, i64 2, !dbg !27
  %659 = call float @llvm.amdgcn.exp2.f32(float %658), !dbg !27
  %660 = insertelement <4 x float> %656, float %659, i64 2, !dbg !29
  %661 = fadd float %657, %659, !dbg !28
  %662 = extractelement <4 x float> %649, i64 3, !dbg !27
  %663 = call float @llvm.amdgcn.exp2.f32(float %662), !dbg !27
  %664 = insertelement <4 x float> %660, float %663, i64 3, !dbg !29
  %math.result280 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %664) #9, !dbg !29, !frisk.register_lifetimes !1
  %665 = fptrunc <4 x float> %math.result280 to <4 x half>, !dbg !29
  %math.packed281 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %665) #9, !dbg !28, !frisk.register_lifetimes !1
  %666 = fadd float %661, %663, !dbg !28
  %667 = shufflevector <32 x float> %590, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %668 = fadd <4 x float> %498, %667, !dbg !26
  %669 = extractelement <4 x float> %668, i64 0, !dbg !27
  %670 = call float @llvm.amdgcn.exp2.f32(float %669), !dbg !27
  %671 = insertelement <4 x float> poison, float %670, i64 0, !dbg !29
  %672 = fadd float %666, %670, !dbg !28
  %673 = extractelement <4 x float> %668, i64 1, !dbg !27
  %674 = call float @llvm.amdgcn.exp2.f32(float %673), !dbg !27
  %675 = insertelement <4 x float> %671, float %674, i64 1, !dbg !29
  %676 = fadd float %672, %674, !dbg !28
  %677 = extractelement <4 x float> %668, i64 2, !dbg !27
  %678 = call float @llvm.amdgcn.exp2.f32(float %677), !dbg !27
  %679 = insertelement <4 x float> %675, float %678, i64 2, !dbg !29
  %680 = fadd float %676, %678, !dbg !28
  %681 = extractelement <4 x float> %668, i64 3, !dbg !27
  %682 = call float @llvm.amdgcn.exp2.f32(float %681), !dbg !27
  %683 = insertelement <4 x float> %679, float %682, i64 3, !dbg !29
  %math.result282 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %683) #9, !dbg !29, !frisk.register_lifetimes !1
  %684 = fptrunc <4 x float> %math.result282 to <4 x half>, !dbg !29
  %math.packed283 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %684) #9, !dbg !28, !frisk.register_lifetimes !1
  %685 = fadd float %680, %682, !dbg !28
  %686 = shufflevector <32 x float> %590, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %687 = fadd <4 x float> %502, %686, !dbg !26
  %688 = extractelement <4 x float> %687, i64 0, !dbg !27
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !27
  %690 = insertelement <4 x float> poison, float %689, i64 0, !dbg !29
  %691 = fadd float %685, %689, !dbg !28
  %692 = extractelement <4 x float> %687, i64 1, !dbg !27
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !27
  %694 = insertelement <4 x float> %690, float %693, i64 1, !dbg !29
  %695 = fadd float %691, %693, !dbg !28
  %696 = extractelement <4 x float> %687, i64 2, !dbg !27
  %697 = call float @llvm.amdgcn.exp2.f32(float %696), !dbg !27
  %698 = insertelement <4 x float> %694, float %697, i64 2, !dbg !29
  %699 = fadd float %695, %697, !dbg !28
  %700 = extractelement <4 x float> %687, i64 3, !dbg !27
  %701 = call float @llvm.amdgcn.exp2.f32(float %700), !dbg !27
  %702 = insertelement <4 x float> %698, float %701, i64 3, !dbg !29
  %math.result284 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %702) #9, !dbg !29, !frisk.register_lifetimes !1
  %703 = fptrunc <4 x float> %math.result284 to <4 x half>, !dbg !29
  %math.packed285 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %703) #9, !dbg !28, !frisk.register_lifetimes !1
  %704 = fadd float %699, %701, !dbg !28
  %705 = shufflevector <32 x float> %590, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %706 = fadd <4 x float> %506, %705, !dbg !26
  %707 = extractelement <4 x float> %706, i64 0, !dbg !27
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !27
  %709 = insertelement <4 x float> poison, float %708, i64 0, !dbg !29
  %710 = fadd float %704, %708, !dbg !28
  %711 = extractelement <4 x float> %706, i64 1, !dbg !27
  %712 = call float @llvm.amdgcn.exp2.f32(float %711), !dbg !27
  %713 = insertelement <4 x float> %709, float %712, i64 1, !dbg !29
  %714 = fadd float %710, %712, !dbg !28
  %715 = extractelement <4 x float> %706, i64 2, !dbg !27
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !27
  %717 = insertelement <4 x float> %713, float %716, i64 2, !dbg !29
  %718 = fadd float %714, %716, !dbg !28
  %719 = extractelement <4 x float> %706, i64 3, !dbg !27
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !27
  %721 = insertelement <4 x float> %717, float %720, i64 3, !dbg !29
  %math.result286 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %721) #9, !dbg !29, !frisk.register_lifetimes !1
  %722 = fptrunc <4 x float> %math.result286 to <4 x half>, !dbg !29
  %math.packed287 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %722) #9, !dbg !28, !frisk.register_lifetimes !1
  %723 = fadd float %718, %720, !dbg !28
  %724 = shufflevector <32 x float> %590, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %725 = fadd <4 x float> %511, %724, !dbg !26
  %726 = extractelement <4 x float> %725, i64 0, !dbg !27
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !27
  %728 = insertelement <4 x float> poison, float %727, i64 0, !dbg !29
  %729 = fadd float %723, %727, !dbg !28
  %730 = extractelement <4 x float> %725, i64 1, !dbg !27
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !27
  %732 = insertelement <4 x float> %728, float %731, i64 1, !dbg !29
  %733 = fadd float %729, %731, !dbg !28
  %734 = extractelement <4 x float> %725, i64 2, !dbg !27
  %735 = call float @llvm.amdgcn.exp2.f32(float %734), !dbg !27
  %736 = insertelement <4 x float> %732, float %735, i64 2, !dbg !29
  %737 = fadd float %733, %735, !dbg !28
  %738 = extractelement <4 x float> %725, i64 3, !dbg !27
  %739 = call float @llvm.amdgcn.exp2.f32(float %738), !dbg !27
  %740 = insertelement <4 x float> %736, float %739, i64 3, !dbg !29
  %math.result288 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %740) #9, !dbg !29, !frisk.register_lifetimes !1
  %741 = fptrunc <4 x float> %math.result288 to <4 x half>, !dbg !29
  %math.packed289 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %741) #9, !dbg !28, !frisk.register_lifetimes !1
  %742 = fadd float %737, %739, !dbg !28
  %743 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %744 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %743), !dbg !28
  %745 = add nuw nsw i32 %744, 64, !dbg !28
  %746 = xor i32 %744, 16, !dbg !28
  %747 = and i32 %745, 192, !dbg !28
  %748 = icmp samesign ult i32 %746, %747, !dbg !28
  %749 = select i1 %748, i32 %746, i32 %744, !dbg !28
  %750 = shl nuw nsw i32 %749, 2, !dbg !28
  %751 = bitcast float %742 to i32, !dbg !28
  %752 = call i32 @llvm.amdgcn.ds.bpermute(i32 %750, i32 %751), !dbg !28
  %753 = bitcast i32 %752 to float, !dbg !28
  %754 = fadd float %742, %753, !dbg !28
  %755 = xor i32 %744, 32, !dbg !28
  %756 = icmp samesign ult i32 %755, %747, !dbg !28
  %757 = select i1 %756, i32 %755, i32 %744, !dbg !28
  %758 = shl nuw nsw i32 %757, 2, !dbg !28
  %759 = bitcast float %754 to i32, !dbg !28
  %760 = call i32 @llvm.amdgcn.ds.bpermute(i32 %758, i32 %759), !dbg !28
  %761 = bitcast i32 %760 to float, !dbg !28
  %762 = fadd float %754, %761, !dbg !28
  %763 = extractelement <1 x float> %70, i64 0, !dbg !28
  %764 = fadd float %762, %763, !dbg !28
  %765 = insertelement <1 x float> poison, float %764, i64 0, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread731 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !33, !frisk.register_lifetimes !1
  %address.local732 = shl nuw nsw i32 %address.thread731, 3, !dbg !15
  %address.local733 = and i32 %address.local732, 120, !dbg !15
  %address.local734 = shl nuw nsw i32 %address.thread731, 5, !dbg !15
  %address.local735 = and i32 %address.local734, 1536, !dbg !15
  %address.local736 = or disjoint i32 %address.local733, %address.local735, !dbg !15
  %address.local737 = zext nneg i32 %address.local736 to i64, !dbg !15
  %address.local738 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local737, !dbg !15
  %766 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local738, i32 0) #0, !dbg !15
  %767 = bitcast <4 x i32> %766 to <8 x half>, !dbg !15
  %address.local739 = shl nuw nsw i32 %address.thread731, 3, !dbg !15
  %address.local740 = and i32 %address.local739, 120, !dbg !15
  %address.local741 = shl nuw nsw i32 %address.thread731, 5, !dbg !15
  %address.local742 = and i32 %address.local741, 1536, !dbg !15
  %address.local743 = or disjoint i32 %address.local740, %address.local742, !dbg !15
  %address.local744 = or disjoint i32 %address.local743, 128, !dbg !15
  %address.local745 = zext nneg i32 %address.local744 to i64, !dbg !15
  %address.local746 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local745, !dbg !15
  %768 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local746, i32 0) #0, !dbg !15
  %769 = bitcast <4 x i32> %768 to <8 x half>, !dbg !15
  %address.local747 = shl nuw nsw i32 %address.thread731, 3, !dbg !15
  %address.local748 = and i32 %address.local747, 120, !dbg !15
  %address.local749 = shl nuw nsw i32 %address.thread731, 5, !dbg !15
  %address.local750 = and i32 %address.local749, 1536, !dbg !15
  %address.local751 = or disjoint i32 %address.local748, %address.local750, !dbg !15
  %address.local752 = or disjoint i32 %address.local751, 256, !dbg !15
  %address.local753 = zext nneg i32 %address.local752 to i64, !dbg !15
  %address.local754 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local753, !dbg !15
  %770 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local754, i32 0) #0, !dbg !15
  %771 = bitcast <4 x i32> %770 to <8 x half>, !dbg !15
  %address.local755 = shl nuw nsw i32 %address.thread731, 3, !dbg !15
  %address.local756 = and i32 %address.local755, 120, !dbg !15
  %address.local757 = shl nuw nsw i32 %address.thread731, 5, !dbg !15
  %address.local758 = and i32 %address.local757, 1536, !dbg !15
  %address.local759 = or disjoint i32 %address.local756, %address.local758, !dbg !15
  %address.local760 = or disjoint i32 %address.local759, 384, !dbg !15
  %address.local761 = zext nneg i32 %address.local760 to i64, !dbg !15
  %address.local762 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local761, !dbg !15
  %772 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local762, i32 0) #0, !dbg !15
  %773 = bitcast <4 x i32> %772 to <8 x half>, !dbg !15
  %address.local763 = shl nuw nsw i32 %address.thread731, 3, !dbg !15
  %address.local764 = and i32 %address.local763, 120, !dbg !15
  %address.local765 = shl nuw nsw i32 %address.thread731, 5, !dbg !15
  %address.local766 = and i32 %address.local765, 1536, !dbg !15
  %address.local767 = or disjoint i32 %address.local764, %address.local766, !dbg !15
  %address.local768 = or disjoint i32 %address.local767, 2048, !dbg !15
  %address.local769 = zext nneg i32 %address.local768 to i64, !dbg !15
  %address.local770 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local769, !dbg !15
  %774 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local770, i32 0) #0, !dbg !15
  %775 = bitcast <4 x i32> %774 to <8 x half>, !dbg !15
  %address.local771 = shl nuw nsw i32 %address.thread731, 3, !dbg !15
  %address.local772 = and i32 %address.local771, 120, !dbg !15
  %address.local773 = shl nuw nsw i32 %address.thread731, 5, !dbg !15
  %address.local774 = and i32 %address.local773, 1536, !dbg !15
  %address.local775 = or disjoint i32 %address.local772, %address.local774, !dbg !15
  %address.local776 = or disjoint i32 %address.local775, 2176, !dbg !15
  %address.local777 = zext nneg i32 %address.local776 to i64, !dbg !15
  %address.local778 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local777, !dbg !15
  %776 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #0, !dbg !15
  %777 = bitcast <4 x i32> %776 to <8 x half>, !dbg !15
  %address.local779 = shl nuw nsw i32 %address.thread731, 3, !dbg !15
  %address.local780 = and i32 %address.local779, 120, !dbg !15
  %address.local781 = shl nuw nsw i32 %address.thread731, 5, !dbg !15
  %address.local782 = and i32 %address.local781, 1536, !dbg !15
  %address.local783 = or disjoint i32 %address.local780, %address.local782, !dbg !15
  %address.local784 = or disjoint i32 %address.local783, 2304, !dbg !15
  %address.local785 = zext nneg i32 %address.local784 to i64, !dbg !15
  %address.local786 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local785, !dbg !15
  %778 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local786, i32 0) #0, !dbg !15
  %779 = bitcast <4 x i32> %778 to <8 x half>, !dbg !15
  %address.local787 = shl nuw nsw i32 %address.thread731, 3, !dbg !15
  %address.local788 = and i32 %address.local787, 120, !dbg !15
  %address.local789 = shl nuw nsw i32 %address.thread731, 5, !dbg !15
  %address.local790 = and i32 %address.local789, 1536, !dbg !15
  %address.local791 = or disjoint i32 %address.local788, %address.local790, !dbg !15
  %address.local792 = or disjoint i32 %address.local791, 2432, !dbg !15
  %address.local793 = zext nneg i32 %address.local792 to i64, !dbg !15
  %address.local794 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local793, !dbg !15
  %780 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local794, i32 0) #0, !dbg !15
  %781 = bitcast <4 x i32> %780 to <8 x half>, !dbg !15
  %782 = add nuw nsw i64 %99, %15, !dbg !18
  %783 = shl i64 %782, 19, !dbg !18
  %784 = and i64 %783, 16252928, !dbg !18
  %785 = add i64 %784, %105, !dbg !18
  %786 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %785, !dbg !18
  %787 = getelementptr i8, ptr addrspace(1) %786, i64 64, !dbg !18
  %address.local795 = zext nneg i32 %address.thread731 to i64, !dbg !13
  %address.local796 = shl nuw nsw i64 %address.local795, 3, !dbg !12
  %address.local797 = and i64 %address.local796, 1536, !dbg !16
  %address.local798 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local797, !dbg !16
  %address.local799 = ptrtoint ptr addrspace(3) %address.local798 to i64, !dbg !16
  %address.local800 = trunc i64 %address.local799 to i32, !dbg !16
  %788 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local800), !dbg !18
  %789 = zext i32 %788 to i64, !dbg !18
  %790 = inttoptr i64 %789 to ptr addrspace(3), !dbg !18
  %address.local801 = shl nuw nsw i64 %address.local795, 5, !dbg !18
  %address.local802 = and i64 %address.local801, 8064, !dbg !18
  %address.local803 = shl nuw nsw i64 %address.local795, 3, !dbg !18
  %address.local804 = and i64 %address.local803, 24, !dbg !18
  %address.local805 = or disjoint i64 %address.local802, %address.local804, !dbg !18
  %address.local806 = getelementptr [2 x i8], ptr addrspace(1) %787, i64 %address.local805, !dbg !18
  %791 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %792 = ptrtoint ptr addrspace(1) %address.local806 to i64, !dbg !18
  %793 = sub i64 %792, %791, !dbg !18
  %dma.buffer.offset1173 = trunc i64 %793 to i32, !dbg !18
  %794 = trunc i64 %791 to i32, !dbg !18
  %795 = insertelement <4 x i32> poison, i32 %794, i64 0, !dbg !18
  %796 = lshr i64 %791, 32, !dbg !18
  %797 = trunc i64 %796 to i32, !dbg !18
  %798 = insertelement <4 x i32> %795, i32 %797, i64 1, !dbg !18
  %799 = insertelement <4 x i32> %798, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1174 = insertelement <4 x i32> %799, i32 131072, i64 3, !dbg !18
  %800 = ptrtoint ptr addrspace(3) %790 to i32, !dbg !18
  %dma.buffer.lds1175 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %800), !dbg !18
  %801 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1175, i32 %dma.buffer.offset1173, <4 x i32> %dma.buffer.rsrc1174) #10, !dbg !18
  %address.local807 = or disjoint i64 %address.local795, 256, !dbg !16
  %address.local808 = shl nuw nsw i64 %address.local807, 3, !dbg !16
  %address.local809 = and i64 %address.local808, 3584, !dbg !16
  %address.local810 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local809, !dbg !16
  %address.local811 = ptrtoint ptr addrspace(3) %address.local810 to i64, !dbg !16
  %address.local812 = trunc i64 %address.local811 to i32, !dbg !16
  %802 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local812), !dbg !18
  %803 = zext i32 %802 to i64, !dbg !18
  %804 = inttoptr i64 %803 to ptr addrspace(3), !dbg !18
  %address.local813 = shl nuw nsw i64 %address.local807, 5, !dbg !18
  %address.local814 = and i64 %address.local813, 16256, !dbg !18
  %address.local815 = shl nuw nsw i64 %address.local795, 3, !dbg !18
  %address.local816 = and i64 %address.local815, 24, !dbg !18
  %address.local817 = or disjoint i64 %address.local814, %address.local816, !dbg !18
  %address.local818 = getelementptr [2 x i8], ptr addrspace(1) %787, i64 %address.local817, !dbg !18
  %805 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %806 = ptrtoint ptr addrspace(1) %address.local818 to i64, !dbg !18
  %807 = sub i64 %806, %805, !dbg !18
  %dma.buffer.offset1170 = trunc i64 %807 to i32, !dbg !18
  %808 = trunc i64 %805 to i32, !dbg !18
  %809 = insertelement <4 x i32> poison, i32 %808, i64 0, !dbg !18
  %810 = lshr i64 %805, 32, !dbg !18
  %811 = trunc i64 %810 to i32, !dbg !18
  %812 = insertelement <4 x i32> %809, i32 %811, i64 1, !dbg !18
  %813 = insertelement <4 x i32> %812, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1171 = insertelement <4 x i32> %813, i32 131072, i64 3, !dbg !18
  %814 = ptrtoint ptr addrspace(3) %804 to i32, !dbg !18
  %dma.buffer.lds1172 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %814), !dbg !18
  %815 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1172, i32 %dma.buffer.offset1170, <4 x i32> %dma.buffer.rsrc1171) #10, !dbg !18
  %816 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %817 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %816, <4 x float> %67) #9, !dbg !15
  %818 = shufflevector <8 x half> %769, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %819 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %818, <4 x float> %817) #9, !dbg !15
  %820 = shufflevector <8 x half> %771, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %821 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %820, <4 x float> %819) #9, !dbg !15
  %822 = shufflevector <8 x half> %773, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %823 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %822, <4 x float> %821) #9, !dbg !15
  %824 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %824, <4 x float> %823) #9, !dbg !15
  %826 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %826, <4 x float> %825) #9, !dbg !15
  %828 = shufflevector <8 x half> %779, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %828, <4 x float> %827) #9, !dbg !15
  %830 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %830, <4 x float> %829) #9, !dbg !15
  %832 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %832, <4 x float> %68) #9, !dbg !15
  %834 = shufflevector <8 x half> %769, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %834, <4 x float> %833) #9, !dbg !15
  %836 = shufflevector <8 x half> %771, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %836, <4 x float> %835) #9, !dbg !15
  %838 = shufflevector <8 x half> %773, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %838, <4 x float> %837) #9, !dbg !15
  %840 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %840, <4 x float> %839) #9, !dbg !15
  %842 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %842, <4 x float> %841) #9, !dbg !15
  %844 = shufflevector <8 x half> %779, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %844, <4 x float> %843) #9, !dbg !15
  %846 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %846, <4 x float> %845) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.local819 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local737, !dbg !15
  %848 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #0, !dbg !15
  %849 = bitcast <4 x i32> %848 to <8 x half>, !dbg !15
  %address.local820 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local745, !dbg !15
  %850 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !15
  %851 = bitcast <4 x i32> %850 to <8 x half>, !dbg !15
  %address.local821 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local753, !dbg !15
  %852 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local821, i32 0) #0, !dbg !15
  %853 = bitcast <4 x i32> %852 to <8 x half>, !dbg !15
  %address.local822 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local761, !dbg !15
  %854 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !15
  %855 = bitcast <4 x i32> %854 to <8 x half>, !dbg !15
  %address.local823 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local769, !dbg !15
  %856 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #0, !dbg !15
  %857 = bitcast <4 x i32> %856 to <8 x half>, !dbg !15
  %address.local824 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local777, !dbg !15
  %858 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local824, i32 0) #0, !dbg !15
  %859 = bitcast <4 x i32> %858 to <8 x half>, !dbg !15
  %address.local825 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local785, !dbg !15
  %860 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local825, i32 0) #0, !dbg !15
  %861 = bitcast <4 x i32> %860 to <8 x half>, !dbg !15
  %address.local826 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local793, !dbg !15
  %862 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !15
  %863 = bitcast <4 x i32> %862 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %864 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %785, !dbg !18
  %865 = getelementptr i8, ptr addrspace(1) %864, i64 128, !dbg !18
  %866 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local800), !dbg !18
  %867 = zext i32 %866 to i64, !dbg !18
  %868 = inttoptr i64 %867 to ptr addrspace(3), !dbg !18
  %address.local827 = getelementptr [2 x i8], ptr addrspace(1) %865, i64 %address.local805, !dbg !18
  %869 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %870 = ptrtoint ptr addrspace(1) %address.local827 to i64, !dbg !18
  %871 = sub i64 %870, %869, !dbg !18
  %dma.buffer.offset1167 = trunc i64 %871 to i32, !dbg !18
  %872 = trunc i64 %869 to i32, !dbg !18
  %873 = insertelement <4 x i32> poison, i32 %872, i64 0, !dbg !18
  %874 = lshr i64 %869, 32, !dbg !18
  %875 = trunc i64 %874 to i32, !dbg !18
  %876 = insertelement <4 x i32> %873, i32 %875, i64 1, !dbg !18
  %877 = insertelement <4 x i32> %876, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1168 = insertelement <4 x i32> %877, i32 131072, i64 3, !dbg !18
  %878 = ptrtoint ptr addrspace(3) %868 to i32, !dbg !18
  %dma.buffer.lds1169 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %878), !dbg !18
  %879 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1169, i32 %dma.buffer.offset1167, <4 x i32> %dma.buffer.rsrc1168) #10, !dbg !18
  %880 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local812), !dbg !18
  %881 = zext i32 %880 to i64, !dbg !18
  %882 = inttoptr i64 %881 to ptr addrspace(3), !dbg !18
  %address.local828 = getelementptr [2 x i8], ptr addrspace(1) %865, i64 %address.local817, !dbg !18
  %883 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %884 = ptrtoint ptr addrspace(1) %address.local828 to i64, !dbg !18
  %885 = sub i64 %884, %883, !dbg !18
  %dma.buffer.offset1164 = trunc i64 %885 to i32, !dbg !18
  %886 = trunc i64 %883 to i32, !dbg !18
  %887 = insertelement <4 x i32> poison, i32 %886, i64 0, !dbg !18
  %888 = lshr i64 %883, 32, !dbg !18
  %889 = trunc i64 %888 to i32, !dbg !18
  %890 = insertelement <4 x i32> %887, i32 %889, i64 1, !dbg !18
  %891 = insertelement <4 x i32> %890, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1165 = insertelement <4 x i32> %891, i32 131072, i64 3, !dbg !18
  %892 = ptrtoint ptr addrspace(3) %882 to i32, !dbg !18
  %dma.buffer.lds1166 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %892), !dbg !18
  %893 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1166, i32 %dma.buffer.offset1164, <4 x i32> %dma.buffer.rsrc1165) #10, !dbg !18
  %894 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %894, <4 x float> %71) #9, !dbg !15
  %896 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %896, <4 x float> %895) #9, !dbg !15
  %898 = shufflevector <8 x half> %853, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %898, <4 x float> %897) #9, !dbg !15
  %900 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %900, <4 x float> %899) #9, !dbg !15
  %902 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %902, <4 x float> %901) #9, !dbg !15
  %904 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %904, <4 x float> %903) #9, !dbg !15
  %906 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %906, <4 x float> %905) #9, !dbg !15
  %908 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %908, <4 x float> %907) #9, !dbg !15
  %910 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %910, <4 x float> %72) #9, !dbg !15
  %912 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %912, <4 x float> %911) #9, !dbg !15
  %914 = shufflevector <8 x half> %853, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %914, <4 x float> %913) #9, !dbg !15
  %916 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %916, <4 x float> %915) #9, !dbg !15
  %918 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %918, <4 x float> %917) #9, !dbg !15
  %920 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %920, <4 x float> %919) #9, !dbg !15
  %922 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %922, <4 x float> %921) #9, !dbg !15
  %924 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %924, <4 x float> %923) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %926 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #0, !dbg !15
  %927 = bitcast <4 x i32> %926 to <8 x half>, !dbg !15
  %928 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !15
  %929 = bitcast <4 x i32> %928 to <8 x half>, !dbg !15
  %930 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local821, i32 0) #0, !dbg !15
  %931 = bitcast <4 x i32> %930 to <8 x half>, !dbg !15
  %932 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !15
  %933 = bitcast <4 x i32> %932 to <8 x half>, !dbg !15
  %934 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #0, !dbg !15
  %935 = bitcast <4 x i32> %934 to <8 x half>, !dbg !15
  %936 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local824, i32 0) #0, !dbg !15
  %937 = bitcast <4 x i32> %936 to <8 x half>, !dbg !15
  %938 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local825, i32 0) #0, !dbg !15
  %939 = bitcast <4 x i32> %938 to <8 x half>, !dbg !15
  %940 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !15
  %941 = bitcast <4 x i32> %940 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %942 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %785, !dbg !18
  %943 = getelementptr i8, ptr addrspace(1) %942, i64 192, !dbg !18
  %944 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local800), !dbg !18
  %945 = zext i32 %944 to i64, !dbg !18
  %946 = inttoptr i64 %945 to ptr addrspace(3), !dbg !18
  %address.local829 = getelementptr [2 x i8], ptr addrspace(1) %943, i64 %address.local805, !dbg !18
  %947 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %948 = ptrtoint ptr addrspace(1) %address.local829 to i64, !dbg !18
  %949 = sub i64 %948, %947, !dbg !18
  %dma.buffer.offset1161 = trunc i64 %949 to i32, !dbg !18
  %950 = trunc i64 %947 to i32, !dbg !18
  %951 = insertelement <4 x i32> poison, i32 %950, i64 0, !dbg !18
  %952 = lshr i64 %947, 32, !dbg !18
  %953 = trunc i64 %952 to i32, !dbg !18
  %954 = insertelement <4 x i32> %951, i32 %953, i64 1, !dbg !18
  %955 = insertelement <4 x i32> %954, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1162 = insertelement <4 x i32> %955, i32 131072, i64 3, !dbg !18
  %956 = ptrtoint ptr addrspace(3) %946 to i32, !dbg !18
  %dma.buffer.lds1163 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %956), !dbg !18
  %957 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1163, i32 %dma.buffer.offset1161, <4 x i32> %dma.buffer.rsrc1162) #10, !dbg !18
  %958 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local812), !dbg !18
  %959 = zext i32 %958 to i64, !dbg !18
  %960 = inttoptr i64 %959 to ptr addrspace(3), !dbg !18
  %address.local830 = getelementptr [2 x i8], ptr addrspace(1) %943, i64 %address.local817, !dbg !18
  %961 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %962 = ptrtoint ptr addrspace(1) %address.local830 to i64, !dbg !18
  %963 = sub i64 %962, %961, !dbg !18
  %dma.buffer.offset = trunc i64 %963 to i32, !dbg !18
  %964 = trunc i64 %961 to i32, !dbg !18
  %965 = insertelement <4 x i32> poison, i32 %964, i64 0, !dbg !18
  %966 = lshr i64 %961, 32, !dbg !18
  %967 = trunc i64 %966 to i32, !dbg !18
  %968 = insertelement <4 x i32> %965, i32 %967, i64 1, !dbg !18
  %969 = insertelement <4 x i32> %968, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %969, i32 131072, i64 3, !dbg !18
  %970 = ptrtoint ptr addrspace(3) %960 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %970), !dbg !18
  %971 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %972 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %972, <4 x float> %74) #9, !dbg !15
  %974 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %974, <4 x float> %973) #9, !dbg !15
  %976 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %976, <4 x float> %975) #9, !dbg !15
  %978 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %978, <4 x float> %977) #9, !dbg !15
  %980 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %980, <4 x float> %979) #9, !dbg !15
  %982 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %982, <4 x float> %981) #9, !dbg !15
  %984 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %984, <4 x float> %983) #9, !dbg !15
  %986 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %986, <4 x float> %985) #9, !dbg !15
  %988 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %988, <4 x float> %75) #9, !dbg !15
  %990 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %990, <4 x float> %989) #9, !dbg !15
  %992 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %992, <4 x float> %991) #9, !dbg !15
  %994 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %994, <4 x float> %993) #9, !dbg !15
  %996 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %996, <4 x float> %995) #9, !dbg !15
  %998 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %998, <4 x float> %997) #9, !dbg !15
  %1000 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1000, <4 x float> %999) #9, !dbg !15
  %1002 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1002, <4 x float> %1001) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1004 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #0, !dbg !15
  %1005 = bitcast <4 x i32> %1004 to <8 x half>, !dbg !15
  %1006 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !15
  %1007 = bitcast <4 x i32> %1006 to <8 x half>, !dbg !15
  %1008 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local821, i32 0) #0, !dbg !15
  %1009 = bitcast <4 x i32> %1008 to <8 x half>, !dbg !15
  %1010 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !15
  %1011 = bitcast <4 x i32> %1010 to <8 x half>, !dbg !15
  %1012 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #0, !dbg !15
  %1013 = bitcast <4 x i32> %1012 to <8 x half>, !dbg !15
  %1014 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local824, i32 0) #0, !dbg !15
  %1015 = bitcast <4 x i32> %1014 to <8 x half>, !dbg !15
  %1016 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local825, i32 0) #0, !dbg !15
  %1017 = bitcast <4 x i32> %1016 to <8 x half>, !dbg !15
  %1018 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !15
  %1019 = bitcast <4 x i32> %1018 to <8 x half>, !dbg !15
  %1020 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1020, <4 x float> %77) #9, !dbg !15
  %1022 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %1022, <4 x float> %1021) #9, !dbg !15
  %1024 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %1024, <4 x float> %1023) #9, !dbg !15
  %1026 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %1026, <4 x float> %1025) #9, !dbg !15
  %1028 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1028, <4 x float> %1027) #9, !dbg !15
  %1030 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1030, <4 x float> %1029) #9, !dbg !15
  %1032 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1032, <4 x float> %1031) #9, !dbg !15
  %1034 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1034, <4 x float> %1033) #9, !dbg !15
  %1036 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1036, <4 x float> %78) #9, !dbg !15
  %1038 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %1038, <4 x float> %1037) #9, !dbg !15
  %1040 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed279, <4 x half> %1040, <4 x float> %1039) #9, !dbg !15
  %1042 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed281, <4 x half> %1042, <4 x float> %1041) #9, !dbg !15
  %1044 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1044, <4 x float> %1043) #9, !dbg !15
  %1046 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1046, <4 x float> %1045) #9, !dbg !15
  %1048 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1048, <4 x float> %1047) #9, !dbg !15
  %1050 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1050, <4 x float> %1049) #9, !dbg !15
  %1052 = shufflevector <4 x float> %831, <4 x float> %847, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1053 = shufflevector <4 x float> %909, <4 x float> %925, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1054 = shufflevector <4 x float> %987, <4 x float> %1003, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1055 = shufflevector <4 x float> %1035, <4 x float> %1051, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %1056 = add nuw nsw i64 %66, 1, !dbg !8
  br label %65, !dbg !8

1057:                                             ; preds = %65
  %1058 = and i32 %34, 16, !dbg !8
  %.not = icmp eq i32 %1058, 0, !dbg !8
  %1059 = and i32 %34, 32, !dbg !8
  %.not36 = icmp eq i32 %1059, 0, !dbg !8
  %1060 = extractelement <8 x float> %79, i64 0, !dbg !8
  %1061 = extractelement <8 x float> %79, i64 1, !dbg !8
  %1062 = extractelement <8 x float> %79, i64 2, !dbg !8
  %1063 = extractelement <8 x float> %79, i64 3, !dbg !8
  %1064 = select i1 %.not, float %1061, float %1060, !dbg !8
  %1065 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %1066 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1065), !dbg !8
  %1067 = add nuw nsw i32 %1066, 64, !dbg !8
  %1068 = xor i32 %1066, 16, !dbg !8
  %1069 = and i32 %1067, 192, !dbg !8
  %1070 = icmp samesign ult i32 %1068, %1069, !dbg !8
  %1071 = select i1 %1070, i32 %1068, i32 %1066, !dbg !8
  %1072 = shl nuw nsw i32 %1071, 2, !dbg !8
  %1073 = bitcast float %1064 to i32, !dbg !8
  %1074 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1073), !dbg !8
  %1075 = bitcast i32 %1074 to float, !dbg !8
  %1076 = select i1 %.not, float %1060, float %1075, !dbg !8
  %1077 = select i1 %.not, float %1075, float %1061, !dbg !8
  %1078 = select i1 %.not, float %1063, float %1062, !dbg !8
  %1079 = bitcast float %1078 to i32, !dbg !8
  %1080 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1079), !dbg !8
  %1081 = bitcast i32 %1080 to float, !dbg !8
  %1082 = select i1 %.not, float %1062, float %1081, !dbg !8
  %1083 = select i1 %.not, float %1081, float %1063, !dbg !8
  %1084 = select i1 %.not36, float %1082, float %1076, !dbg !8
  %1085 = xor i32 %1066, 32, !dbg !8
  %1086 = icmp samesign ult i32 %1085, %1069, !dbg !8
  %1087 = select i1 %1086, i32 %1085, i32 %1066, !dbg !8
  %1088 = shl nuw nsw i32 %1087, 2, !dbg !8
  %1089 = bitcast float %1084 to i32, !dbg !8
  %1090 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1089), !dbg !8
  %1091 = bitcast i32 %1090 to float, !dbg !8
  %1092 = select i1 %.not36, float %1076, float %1091, !dbg !8
  %1093 = select i1 %.not36, float %1091, float %1082, !dbg !8
  %1094 = select i1 %.not36, float %1083, float %1077, !dbg !8
  %1095 = bitcast float %1094 to i32, !dbg !8
  %1096 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1095), !dbg !8
  %1097 = bitcast i32 %1096 to float, !dbg !8
  %1098 = select i1 %.not36, float %1077, float %1097, !dbg !8
  %1099 = select i1 %.not36, float %1097, float %1083, !dbg !8
  %1100 = extractelement <8 x float> %76, i64 0, !dbg !8
  %1101 = extractelement <8 x float> %76, i64 1, !dbg !8
  %1102 = extractelement <8 x float> %76, i64 2, !dbg !8
  %1103 = extractelement <8 x float> %76, i64 3, !dbg !8
  %1104 = select i1 %.not, float %1101, float %1100, !dbg !8
  %1105 = bitcast float %1104 to i32, !dbg !8
  %1106 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1105), !dbg !8
  %1107 = bitcast i32 %1106 to float, !dbg !8
  %1108 = select i1 %.not, float %1100, float %1107, !dbg !8
  %1109 = select i1 %.not, float %1107, float %1101, !dbg !8
  %1110 = select i1 %.not, float %1103, float %1102, !dbg !8
  %1111 = bitcast float %1110 to i32, !dbg !8
  %1112 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1111), !dbg !8
  %1113 = bitcast i32 %1112 to float, !dbg !8
  %1114 = select i1 %.not, float %1102, float %1113, !dbg !8
  %1115 = select i1 %.not, float %1113, float %1103, !dbg !8
  %1116 = select i1 %.not36, float %1114, float %1108, !dbg !8
  %1117 = bitcast float %1116 to i32, !dbg !8
  %1118 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1117), !dbg !8
  %1119 = bitcast i32 %1118 to float, !dbg !8
  %1120 = select i1 %.not36, float %1108, float %1119, !dbg !8
  %1121 = select i1 %.not36, float %1119, float %1114, !dbg !8
  %1122 = select i1 %.not36, float %1115, float %1109, !dbg !8
  %1123 = bitcast float %1122 to i32, !dbg !8
  %1124 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1123), !dbg !8
  %1125 = bitcast i32 %1124 to float, !dbg !8
  %1126 = select i1 %.not36, float %1109, float %1125, !dbg !8
  %1127 = select i1 %.not36, float %1125, float %1115, !dbg !8
  %1128 = extractelement <8 x float> %73, i64 0, !dbg !8
  %1129 = extractelement <8 x float> %73, i64 1, !dbg !8
  %1130 = extractelement <8 x float> %73, i64 2, !dbg !8
  %1131 = extractelement <8 x float> %73, i64 3, !dbg !8
  %1132 = select i1 %.not, float %1129, float %1128, !dbg !8
  %1133 = bitcast float %1132 to i32, !dbg !8
  %1134 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1133), !dbg !8
  %1135 = bitcast i32 %1134 to float, !dbg !8
  %1136 = select i1 %.not, float %1128, float %1135, !dbg !8
  %1137 = select i1 %.not, float %1135, float %1129, !dbg !8
  %1138 = select i1 %.not, float %1131, float %1130, !dbg !8
  %1139 = bitcast float %1138 to i32, !dbg !8
  %1140 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1139), !dbg !8
  %1141 = bitcast i32 %1140 to float, !dbg !8
  %1142 = select i1 %.not, float %1130, float %1141, !dbg !8
  %1143 = select i1 %.not, float %1141, float %1131, !dbg !8
  %1144 = select i1 %.not36, float %1142, float %1136, !dbg !8
  %1145 = bitcast float %1144 to i32, !dbg !8
  %1146 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1145), !dbg !8
  %1147 = bitcast i32 %1146 to float, !dbg !8
  %1148 = select i1 %.not36, float %1136, float %1147, !dbg !8
  %1149 = select i1 %.not36, float %1147, float %1142, !dbg !8
  %1150 = select i1 %.not36, float %1143, float %1137, !dbg !8
  %1151 = bitcast float %1150 to i32, !dbg !8
  %1152 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1151), !dbg !8
  %1153 = bitcast i32 %1152 to float, !dbg !8
  %1154 = select i1 %.not36, float %1137, float %1153, !dbg !8
  %1155 = select i1 %.not36, float %1153, float %1143, !dbg !8
  %1156 = extractelement <8 x float> %69, i64 0, !dbg !8
  %1157 = extractelement <8 x float> %69, i64 1, !dbg !8
  %1158 = extractelement <8 x float> %69, i64 2, !dbg !8
  %1159 = extractelement <8 x float> %69, i64 3, !dbg !8
  %1160 = select i1 %.not, float %1157, float %1156, !dbg !8
  %1161 = bitcast float %1160 to i32, !dbg !8
  %1162 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1161), !dbg !8
  %1163 = bitcast i32 %1162 to float, !dbg !8
  %1164 = select i1 %.not, float %1156, float %1163, !dbg !8
  %1165 = select i1 %.not, float %1163, float %1157, !dbg !8
  %1166 = select i1 %.not, float %1159, float %1158, !dbg !8
  %1167 = bitcast float %1166 to i32, !dbg !8
  %1168 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1167), !dbg !8
  %1169 = bitcast i32 %1168 to float, !dbg !8
  %1170 = select i1 %.not, float %1158, float %1169, !dbg !8
  %1171 = select i1 %.not, float %1169, float %1159, !dbg !8
  %1172 = select i1 %.not36, float %1170, float %1164, !dbg !8
  %1173 = bitcast float %1172 to i32, !dbg !8
  %1174 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1173), !dbg !8
  %1175 = select i1 %.not36, float %1171, float %1165, !dbg !8
  %1176 = bitcast float %1175 to i32, !dbg !8
  %1177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1176), !dbg !8
  %1178 = shufflevector <1 x float> %70, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1179 = extractelement <8 x float> %79, i64 4, !dbg !8
  %1180 = extractelement <8 x float> %79, i64 5, !dbg !8
  %1181 = extractelement <8 x float> %79, i64 6, !dbg !8
  %1182 = extractelement <8 x float> %79, i64 7, !dbg !8
  %1183 = select i1 %.not, float %1180, float %1179, !dbg !8
  %1184 = bitcast float %1183 to i32, !dbg !8
  %1185 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1184), !dbg !8
  %1186 = bitcast i32 %1185 to float, !dbg !8
  %1187 = select i1 %.not, float %1179, float %1186, !dbg !8
  %1188 = select i1 %.not, float %1186, float %1180, !dbg !8
  %1189 = select i1 %.not, float %1182, float %1181, !dbg !8
  %1190 = bitcast float %1189 to i32, !dbg !8
  %1191 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1190), !dbg !8
  %1192 = bitcast i32 %1191 to float, !dbg !8
  %1193 = select i1 %.not, float %1181, float %1192, !dbg !8
  %1194 = select i1 %.not, float %1192, float %1182, !dbg !8
  %1195 = select i1 %.not36, float %1193, float %1187, !dbg !8
  %1196 = bitcast float %1195 to i32, !dbg !8
  %1197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1196), !dbg !8
  %1198 = bitcast i32 %1197 to float, !dbg !8
  %1199 = select i1 %.not36, float %1187, float %1198, !dbg !8
  %1200 = select i1 %.not36, float %1198, float %1193, !dbg !8
  %1201 = select i1 %.not36, float %1194, float %1188, !dbg !8
  %1202 = bitcast float %1201 to i32, !dbg !8
  %1203 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1202), !dbg !8
  %1204 = bitcast i32 %1203 to float, !dbg !8
  %1205 = select i1 %.not36, float %1188, float %1204, !dbg !8
  %1206 = select i1 %.not36, float %1204, float %1194, !dbg !8
  %1207 = extractelement <8 x float> %76, i64 4, !dbg !8
  %1208 = extractelement <8 x float> %76, i64 5, !dbg !8
  %1209 = extractelement <8 x float> %76, i64 6, !dbg !8
  %1210 = extractelement <8 x float> %76, i64 7, !dbg !8
  %1211 = select i1 %.not, float %1208, float %1207, !dbg !8
  %1212 = bitcast float %1211 to i32, !dbg !8
  %1213 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1212), !dbg !8
  %1214 = bitcast i32 %1213 to float, !dbg !8
  %1215 = select i1 %.not, float %1207, float %1214, !dbg !8
  %1216 = select i1 %.not, float %1214, float %1208, !dbg !8
  %1217 = select i1 %.not, float %1210, float %1209, !dbg !8
  %1218 = bitcast float %1217 to i32, !dbg !8
  %1219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1218), !dbg !8
  %1220 = bitcast i32 %1219 to float, !dbg !8
  %1221 = select i1 %.not, float %1209, float %1220, !dbg !8
  %1222 = select i1 %.not, float %1220, float %1210, !dbg !8
  %1223 = select i1 %.not36, float %1221, float %1215, !dbg !8
  %1224 = bitcast float %1223 to i32, !dbg !8
  %1225 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1224), !dbg !8
  %1226 = bitcast i32 %1225 to float, !dbg !8
  %1227 = select i1 %.not36, float %1215, float %1226, !dbg !8
  %1228 = select i1 %.not36, float %1226, float %1221, !dbg !8
  %1229 = select i1 %.not36, float %1222, float %1216, !dbg !8
  %1230 = bitcast float %1229 to i32, !dbg !8
  %1231 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1230), !dbg !8
  %1232 = bitcast i32 %1231 to float, !dbg !8
  %1233 = select i1 %.not36, float %1216, float %1232, !dbg !8
  %1234 = select i1 %.not36, float %1232, float %1222, !dbg !8
  %1235 = extractelement <8 x float> %73, i64 4, !dbg !8
  %1236 = extractelement <8 x float> %73, i64 5, !dbg !8
  %1237 = extractelement <8 x float> %73, i64 6, !dbg !8
  %1238 = extractelement <8 x float> %73, i64 7, !dbg !8
  %1239 = select i1 %.not, float %1236, float %1235, !dbg !8
  %1240 = bitcast float %1239 to i32, !dbg !8
  %1241 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1240), !dbg !8
  %1242 = bitcast i32 %1241 to float, !dbg !8
  %1243 = select i1 %.not, float %1235, float %1242, !dbg !8
  %1244 = select i1 %.not, float %1242, float %1236, !dbg !8
  %1245 = select i1 %.not, float %1238, float %1237, !dbg !8
  %1246 = bitcast float %1245 to i32, !dbg !8
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1246), !dbg !8
  %1248 = bitcast i32 %1247 to float, !dbg !8
  %1249 = select i1 %.not, float %1237, float %1248, !dbg !8
  %1250 = select i1 %.not, float %1248, float %1238, !dbg !8
  %1251 = select i1 %.not36, float %1249, float %1243, !dbg !8
  %1252 = bitcast float %1251 to i32, !dbg !8
  %1253 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1252), !dbg !8
  %1254 = bitcast i32 %1253 to float, !dbg !8
  %1255 = select i1 %.not36, float %1243, float %1254, !dbg !8
  %1256 = select i1 %.not36, float %1254, float %1249, !dbg !8
  %1257 = select i1 %.not36, float %1250, float %1244, !dbg !8
  %1258 = bitcast float %1257 to i32, !dbg !8
  %1259 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1258), !dbg !8
  %1260 = bitcast i32 %1259 to float, !dbg !8
  %1261 = select i1 %.not36, float %1244, float %1260, !dbg !8
  %1262 = select i1 %.not36, float %1260, float %1250, !dbg !8
  %1263 = extractelement <8 x float> %69, i64 4, !dbg !8
  %1264 = extractelement <8 x float> %69, i64 5, !dbg !8
  %1265 = extractelement <8 x float> %69, i64 6, !dbg !8
  %1266 = extractelement <8 x float> %69, i64 7, !dbg !8
  %1267 = select i1 %.not, float %1264, float %1263, !dbg !8
  %1268 = bitcast float %1267 to i32, !dbg !8
  %1269 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1268), !dbg !8
  %1270 = bitcast i32 %1269 to float, !dbg !8
  %1271 = select i1 %.not, float %1263, float %1270, !dbg !8
  %1272 = select i1 %.not, float %1270, float %1264, !dbg !8
  %1273 = select i1 %.not, float %1266, float %1265, !dbg !8
  %1274 = bitcast float %1273 to i32, !dbg !8
  %1275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1072, i32 %1274), !dbg !8
  %1276 = bitcast i32 %1275 to float, !dbg !8
  %1277 = select i1 %.not, float %1265, float %1276, !dbg !8
  %1278 = select i1 %.not, float %1276, float %1266, !dbg !8
  %1279 = select i1 %.not36, float %1277, float %1271, !dbg !8
  %1280 = bitcast float %1279 to i32, !dbg !8
  %1281 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1280), !dbg !8
  %1282 = select i1 %.not36, float %1278, float %1272, !dbg !8
  %1283 = bitcast float %1282 to i32, !dbg !8
  %1284 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1088, i32 %1283), !dbg !8
  br i1 true, label %1285, label %1319, !dbg !21

1285:                                             ; preds = %1057
  %1286 = bitcast i32 %1281 to float, !dbg !8
  %1287 = select i1 %.not36, float %1271, float %1286, !dbg !8
  %1288 = insertelement <4 x float> poison, float %1287, i64 0, !dbg !19
  %1289 = bitcast i32 %1284 to float, !dbg !8
  %1290 = select i1 %.not36, float %1272, float %1289, !dbg !8
  %1291 = insertelement <4 x float> %1288, float %1290, i64 1, !dbg !19
  %1292 = select i1 %.not36, float %1286, float %1277, !dbg !8
  %1293 = insertelement <4 x float> %1291, float %1292, i64 2, !dbg !19
  %1294 = select i1 %.not36, float %1289, float %1278, !dbg !8
  %1295 = insertelement <4 x float> %1293, float %1294, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1178, !dbg !19
  %normalized = fmul <4 x float> %1295, %row.reciprocal, !dbg !19
  %1296 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1297 = extractelement <4 x half> %1296, i64 3, !dbg !20
  %1298 = extractelement <4 x half> %1296, i64 2, !dbg !20
  %1299 = extractelement <4 x half> %1296, i64 1, !dbg !20
  %1300 = extractelement <4 x half> %1296, i64 0, !dbg !20
  %1301 = bitcast i32 %1174 to float, !dbg !8
  %1302 = select i1 %.not36, float %1164, float %1301, !dbg !8
  %1303 = insertelement <4 x float> poison, float %1302, i64 0, !dbg !19
  %1304 = bitcast i32 %1177 to float, !dbg !8
  %1305 = select i1 %.not36, float %1165, float %1304, !dbg !8
  %1306 = insertelement <4 x float> %1303, float %1305, i64 1, !dbg !19
  %1307 = select i1 %.not36, float %1301, float %1170, !dbg !8
  %1308 = insertelement <4 x float> %1306, float %1307, i64 2, !dbg !19
  %1309 = select i1 %.not36, float %1304, float %1171, !dbg !8
  %1310 = insertelement <4 x float> %1308, float %1309, i64 3, !dbg !19
  %normalized72 = fmul <4 x float> %1310, %row.reciprocal, !dbg !19
  %1311 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1312 = extractelement <4 x half> %1311, i64 3, !dbg !20
  %1313 = extractelement <4 x half> %1311, i64 2, !dbg !20
  %1314 = extractelement <4 x half> %1311, i64 1, !dbg !20
  %1315 = extractelement <4 x half> %1311, i64 0, !dbg !20
  %1316 = add nsw i64 %15, %25, !dbg !21
  %1317 = shl i64 %1316, 19, !dbg !21
  %1318 = and i64 %1317, 16252928, !dbg !21
  %address.thread831 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local832 = zext nneg i32 %address.thread831 to i64, !dbg !13
  %address.local833 = add i64 %28, %address.local832, !dbg !21
  %address.local834 = lshr i64 %address.local832, 6, !dbg !13
  %address.local835 = shl nuw nsw i64 %address.local834, 4, !dbg !21
  %address.local836 = add i64 %address.local833, %address.local835, !dbg !21
  %address.local837 = and i64 %address.local832, 240, !dbg !21
  %address.local838 = sub i64 %address.local836, %address.local837, !dbg !21
  %address.local839 = add i64 %address.local838, %29, !dbg !21
  %address.local840 = shl nuw nsw i64 %address.local839, 7, !dbg !21
  %address.local841 = add nuw nsw i64 %1318, %address.local840, !dbg !21
  %address.local842 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !21
  %address.local843 = lshr i64 %address.local832, 2, !dbg !13
  %address.local844 = and i64 %address.local843, 12, !dbg !13
  %address.local845 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local842, i64 %address.local844, !dbg !21
  store half %1315, ptr addrspace(1) %address.local845, align 2, !dbg !21
  %address.local846 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !21
  %address.local847 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local846, i64 %address.local844, !dbg !21
  %address.local848 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local847, i64 2, !dbg !21
  store half %1314, ptr addrspace(1) %address.local848, align 2, !dbg !21
  %address.local849 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !21
  %address.local850 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local849, i64 %address.local844, !dbg !21
  %address.local851 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local850, i64 4, !dbg !21
  store half %1313, ptr addrspace(1) %address.local851, align 2, !dbg !21
  %address.local852 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !21
  %address.local853 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local844, !dbg !21
  %address.local854 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local853, i64 6, !dbg !21
  store half %1312, ptr addrspace(1) %address.local854, align 2, !dbg !21
  %address.local855 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !21
  %address.local856 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local855, i64 %address.local844, !dbg !21
  %address.local857 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local856, i64 32, !dbg !21
  store half %1300, ptr addrspace(1) %address.local857, align 2, !dbg !21
  %address.local858 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !21
  %address.local859 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local858, i64 %address.local844, !dbg !21
  %address.local860 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local859, i64 34, !dbg !21
  store half %1299, ptr addrspace(1) %address.local860, align 2, !dbg !21
  %address.local861 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !21
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local861, i64 %address.local844, !dbg !21
  %address.local863 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local862, i64 36, !dbg !21
  store half %1298, ptr addrspace(1) %address.local863, align 2, !dbg !21
  %address.local864 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local841, !dbg !21
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local864, i64 %address.local844, !dbg !21
  %address.local866 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local865, i64 38, !dbg !21
  store half %1297, ptr addrspace(1) %address.local866, align 2, !dbg !21
  br label %1319, !dbg !21

1319:                                             ; preds = %1285, %1057
  br i1 true, label %1320, label %1340, !dbg !21

1320:                                             ; preds = %1319
  %1321 = insertelement <4 x float> poison, float %1255, i64 0, !dbg !19
  %1322 = insertelement <4 x float> %1321, float %1261, i64 1, !dbg !19
  %1323 = insertelement <4 x float> %1322, float %1256, i64 2, !dbg !19
  %1324 = insertelement <4 x float> %1323, float %1262, i64 3, !dbg !19
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1178, !dbg !19
  %normalized74 = fmul <4 x float> %1324, %row.reciprocal73, !dbg !19
  %1325 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1326 = extractelement <4 x half> %1325, i64 3, !dbg !20
  %1327 = extractelement <4 x half> %1325, i64 2, !dbg !20
  %1328 = extractelement <4 x half> %1325, i64 1, !dbg !20
  %1329 = extractelement <4 x half> %1325, i64 0, !dbg !20
  %1330 = insertelement <4 x float> poison, float %1148, i64 0, !dbg !19
  %1331 = insertelement <4 x float> %1330, float %1154, i64 1, !dbg !19
  %1332 = insertelement <4 x float> %1331, float %1149, i64 2, !dbg !19
  %1333 = insertelement <4 x float> %1332, float %1155, i64 3, !dbg !19
  %normalized75 = fmul <4 x float> %1333, %row.reciprocal73, !dbg !19
  %1334 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1335 = extractelement <4 x half> %1334, i64 3, !dbg !20
  %1336 = extractelement <4 x half> %1334, i64 2, !dbg !20
  %1337 = extractelement <4 x half> %1334, i64 1, !dbg !20
  %1338 = extractelement <4 x half> %1334, i64 0, !dbg !20
  %1339 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread867 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local868 = zext nneg i32 %address.thread867 to i64, !dbg !13
  %address.local869 = add i64 %28, %address.local868, !dbg !21
  %address.local870 = lshr i64 %address.local868, 6, !dbg !13
  %address.local871 = shl nuw nsw i64 %address.local870, 4, !dbg !21
  %address.local872 = add i64 %address.local869, %address.local871, !dbg !21
  %address.local873 = and i64 %address.local868, 240, !dbg !21
  %address.local874 = sub i64 %address.local872, %address.local873, !dbg !21
  %address.local875 = add i64 %address.local874, %29, !dbg !21
  %address.local876 = shl nuw nsw i64 %address.local875, 7, !dbg !21
  %address.local877 = add nuw nsw i64 %1339, %address.local876, !dbg !21
  %address.local878 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !21
  %address.local879 = lshr i64 %address.local868, 2, !dbg !13
  %address.local880 = and i64 %address.local879, 12, !dbg !13
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local878, i64 %address.local880, !dbg !21
  %address.local882 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local881, i64 64, !dbg !21
  store half %1338, ptr addrspace(1) %address.local882, align 2, !dbg !21
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !21
  %address.local884 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local883, i64 %address.local880, !dbg !21
  %address.local885 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local884, i64 66, !dbg !21
  store half %1337, ptr addrspace(1) %address.local885, align 2, !dbg !21
  %address.local886 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !21
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local886, i64 %address.local880, !dbg !21
  %address.local888 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local887, i64 68, !dbg !21
  store half %1336, ptr addrspace(1) %address.local888, align 2, !dbg !21
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !21
  %address.local890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local889, i64 %address.local880, !dbg !21
  %address.local891 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local890, i64 70, !dbg !21
  store half %1335, ptr addrspace(1) %address.local891, align 2, !dbg !21
  %address.local892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !21
  %address.local893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local892, i64 %address.local880, !dbg !21
  %address.local894 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local893, i64 96, !dbg !21
  store half %1329, ptr addrspace(1) %address.local894, align 2, !dbg !21
  %address.local895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !21
  %address.local896 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local895, i64 %address.local880, !dbg !21
  %address.local897 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local896, i64 98, !dbg !21
  store half %1328, ptr addrspace(1) %address.local897, align 2, !dbg !21
  %address.local898 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !21
  %address.local899 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local898, i64 %address.local880, !dbg !21
  %address.local900 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local899, i64 100, !dbg !21
  store half %1327, ptr addrspace(1) %address.local900, align 2, !dbg !21
  %address.local901 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local877, !dbg !21
  %address.local902 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local901, i64 %address.local880, !dbg !21
  %address.local903 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local902, i64 102, !dbg !21
  store half %1326, ptr addrspace(1) %address.local903, align 2, !dbg !21
  br label %1340, !dbg !21

1340:                                             ; preds = %1320, %1319
  br i1 true, label %1341, label %1361, !dbg !21

1341:                                             ; preds = %1340
  %1342 = insertelement <4 x float> poison, float %1227, i64 0, !dbg !19
  %1343 = insertelement <4 x float> %1342, float %1233, i64 1, !dbg !19
  %1344 = insertelement <4 x float> %1343, float %1228, i64 2, !dbg !19
  %1345 = insertelement <4 x float> %1344, float %1234, i64 3, !dbg !19
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1178, !dbg !19
  %normalized77 = fmul <4 x float> %1345, %row.reciprocal76, !dbg !19
  %1346 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %1347 = extractelement <4 x half> %1346, i64 3, !dbg !20
  %1348 = extractelement <4 x half> %1346, i64 2, !dbg !20
  %1349 = extractelement <4 x half> %1346, i64 1, !dbg !20
  %1350 = extractelement <4 x half> %1346, i64 0, !dbg !20
  %1351 = insertelement <4 x float> poison, float %1120, i64 0, !dbg !19
  %1352 = insertelement <4 x float> %1351, float %1126, i64 1, !dbg !19
  %1353 = insertelement <4 x float> %1352, float %1121, i64 2, !dbg !19
  %1354 = insertelement <4 x float> %1353, float %1127, i64 3, !dbg !19
  %normalized78 = fmul <4 x float> %1354, %row.reciprocal76, !dbg !19
  %1355 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !20
  %1356 = extractelement <4 x half> %1355, i64 3, !dbg !20
  %1357 = extractelement <4 x half> %1355, i64 2, !dbg !20
  %1358 = extractelement <4 x half> %1355, i64 1, !dbg !20
  %1359 = extractelement <4 x half> %1355, i64 0, !dbg !20
  %1360 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread904 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local905 = zext nneg i32 %address.thread904 to i64, !dbg !13
  %address.local906 = add i64 %28, %address.local905, !dbg !21
  %address.local907 = lshr i64 %address.local905, 6, !dbg !13
  %address.local908 = shl nuw nsw i64 %address.local907, 4, !dbg !21
  %address.local909 = add i64 %address.local906, %address.local908, !dbg !21
  %address.local910 = and i64 %address.local905, 240, !dbg !21
  %address.local911 = sub i64 %address.local909, %address.local910, !dbg !21
  %address.local912 = add i64 %address.local911, %29, !dbg !21
  %address.local913 = shl nuw nsw i64 %address.local912, 7, !dbg !21
  %address.local914 = add nuw nsw i64 %1360, %address.local913, !dbg !21
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !21
  %address.local916 = lshr i64 %address.local905, 2, !dbg !13
  %address.local917 = and i64 %address.local916, 12, !dbg !13
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local915, i64 %address.local917, !dbg !21
  %address.local919 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local918, i64 128, !dbg !21
  store half %1359, ptr addrspace(1) %address.local919, align 2, !dbg !21
  %address.local920 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !21
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local920, i64 %address.local917, !dbg !21
  %address.local922 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local921, i64 130, !dbg !21
  store half %1358, ptr addrspace(1) %address.local922, align 2, !dbg !21
  %address.local923 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !21
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local923, i64 %address.local917, !dbg !21
  %address.local925 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local924, i64 132, !dbg !21
  store half %1357, ptr addrspace(1) %address.local925, align 2, !dbg !21
  %address.local926 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !21
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local926, i64 %address.local917, !dbg !21
  %address.local928 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local927, i64 134, !dbg !21
  store half %1356, ptr addrspace(1) %address.local928, align 2, !dbg !21
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !21
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local929, i64 %address.local917, !dbg !21
  %address.local931 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local930, i64 160, !dbg !21
  store half %1350, ptr addrspace(1) %address.local931, align 2, !dbg !21
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !21
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local932, i64 %address.local917, !dbg !21
  %address.local934 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local933, i64 162, !dbg !21
  store half %1349, ptr addrspace(1) %address.local934, align 2, !dbg !21
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !21
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local935, i64 %address.local917, !dbg !21
  %address.local937 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local936, i64 164, !dbg !21
  store half %1348, ptr addrspace(1) %address.local937, align 2, !dbg !21
  %address.local938 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local914, !dbg !21
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local938, i64 %address.local917, !dbg !21
  %address.local940 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local939, i64 166, !dbg !21
  store half %1347, ptr addrspace(1) %address.local940, align 2, !dbg !21
  br label %1361, !dbg !21

1361:                                             ; preds = %1341, %1340
  br i1 true, label %1362, label %1382, !dbg !21

1362:                                             ; preds = %1361
  %1363 = insertelement <4 x float> poison, float %1199, i64 0, !dbg !19
  %1364 = insertelement <4 x float> %1363, float %1205, i64 1, !dbg !19
  %1365 = insertelement <4 x float> %1364, float %1200, i64 2, !dbg !19
  %1366 = insertelement <4 x float> %1365, float %1206, i64 3, !dbg !19
  %row.reciprocal79 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1178, !dbg !19
  %normalized80 = fmul <4 x float> %1366, %row.reciprocal79, !dbg !19
  %1367 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !20
  %1368 = extractelement <4 x half> %1367, i64 3, !dbg !20
  %1369 = extractelement <4 x half> %1367, i64 2, !dbg !20
  %1370 = extractelement <4 x half> %1367, i64 1, !dbg !20
  %1371 = extractelement <4 x half> %1367, i64 0, !dbg !20
  %1372 = insertelement <4 x float> poison, float %1092, i64 0, !dbg !19
  %1373 = insertelement <4 x float> %1372, float %1098, i64 1, !dbg !19
  %1374 = insertelement <4 x float> %1373, float %1093, i64 2, !dbg !19
  %1375 = insertelement <4 x float> %1374, float %1099, i64 3, !dbg !19
  %normalized81 = fmul <4 x float> %1375, %row.reciprocal79, !dbg !19
  %1376 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !20
  %1377 = extractelement <4 x half> %1376, i64 3, !dbg !20
  %1378 = extractelement <4 x half> %1376, i64 2, !dbg !20
  %1379 = extractelement <4 x half> %1376, i64 1, !dbg !20
  %1380 = extractelement <4 x half> %1376, i64 0, !dbg !20
  %1381 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread941 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local942 = zext nneg i32 %address.thread941 to i64, !dbg !13
  %address.local943 = add i64 %28, %address.local942, !dbg !21
  %address.local944 = lshr i64 %address.local942, 6, !dbg !13
  %address.local945 = shl nuw nsw i64 %address.local944, 4, !dbg !21
  %address.local946 = add i64 %address.local943, %address.local945, !dbg !21
  %address.local947 = and i64 %address.local942, 240, !dbg !21
  %address.local948 = sub i64 %address.local946, %address.local947, !dbg !21
  %address.local949 = add i64 %address.local948, %29, !dbg !21
  %address.local950 = shl nuw nsw i64 %address.local949, 7, !dbg !21
  %address.local951 = add nuw nsw i64 %1381, %address.local950, !dbg !21
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !21
  %address.local953 = lshr i64 %address.local942, 2, !dbg !13
  %address.local954 = and i64 %address.local953, 12, !dbg !13
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local952, i64 %address.local954, !dbg !21
  %address.local956 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local955, i64 192, !dbg !21
  store half %1380, ptr addrspace(1) %address.local956, align 2, !dbg !21
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !21
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local957, i64 %address.local954, !dbg !21
  %address.local959 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local958, i64 194, !dbg !21
  store half %1379, ptr addrspace(1) %address.local959, align 2, !dbg !21
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !21
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local960, i64 %address.local954, !dbg !21
  %address.local962 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local961, i64 196, !dbg !21
  store half %1378, ptr addrspace(1) %address.local962, align 2, !dbg !21
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !21
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local963, i64 %address.local954, !dbg !21
  %address.local965 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local964, i64 198, !dbg !21
  store half %1377, ptr addrspace(1) %address.local965, align 2, !dbg !21
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !21
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local966, i64 %address.local954, !dbg !21
  %address.local968 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local967, i64 224, !dbg !21
  store half %1371, ptr addrspace(1) %address.local968, align 2, !dbg !21
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !21
  %address.local970 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local954, !dbg !21
  %address.local971 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local970, i64 226, !dbg !21
  store half %1370, ptr addrspace(1) %address.local971, align 2, !dbg !21
  %address.local972 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !21
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local972, i64 %address.local954, !dbg !21
  %address.local974 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local973, i64 228, !dbg !21
  store half %1369, ptr addrspace(1) %address.local974, align 2, !dbg !21
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local951, !dbg !21
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local975, i64 %address.local954, !dbg !21
  %address.local977 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local976, i64 230, !dbg !21
  store half %1368, ptr addrspace(1) %address.local977, align 2, !dbg !21
  br label %1382, !dbg !21

1382:                                             ; preds = %1362, %1361
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/bm_bn_sweep/export/run-20261022-180820-lrul_y0l/input-0003-test_input_1x32x4096x128-64-128-32/group8-reverse-k128-swizzle")
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
