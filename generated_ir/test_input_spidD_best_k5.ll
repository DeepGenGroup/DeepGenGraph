; ModuleID = '/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD.mlir'
source_filename = "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #0 !dbg !4 !reqd_work_group_size !32 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %10 = sext i32 %9 to i64, !dbg !5
  %11 = shl nsw i64 %8, 7, !dbg !11
  %12 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !34
  %13 = zext nneg i32 %12 to i64, !dbg !13
  %.lobit = ashr i32 %7, 31, !dbg !13
  %14 = xor i32 %7, %.lobit, !dbg !13
  %15 = sdiv i32 %14, 32, !dbg !13
  %.lobit1 = ashr i32 %7, 31, !dbg !13
  %16 = xor i32 %15, %.lobit1, !dbg !13
  %17 = sext i32 %16 to i64, !dbg !13
  %18 = add nsw i64 %10, %17, !dbg !13
  %19 = and i64 %18, 31, !dbg !13
  %20 = lshr i64 %13, 2, !dbg !14
  %21 = lshr i64 %13, 1, !dbg !14
  %22 = shl nuw nsw i64 %13, 2, !dbg !15
  %23 = lshr i64 %13, 6, !dbg !15
  %24 = lshr i64 %13, 4, !dbg !15
  %25 = and i64 %24, 3, !dbg !15
  %26 = lshr i64 %13, 4, !dbg !15
  %27 = or disjoint i64 %11, 127, !dbg !16
  %28 = add nsw i64 %11, 128, !dbg !16
  %29 = icmp uge i64 %28, %27, !dbg !16
  %30 = shl nsw i64 %8, 7, !dbg !13
  %31 = mul nsw i64 %26, -31, !dbg !13
  %32 = xor i64 %31, -1, !dbg !13
  %33 = sdiv i64 %32, 2, !dbg !13
  %34 = mul nsw i64 %17, -4096, !dbg !13
  %35 = lshr i64 %13, 3, !dbg !9
  %36 = mul nsw i64 %35, -36, !dbg !9
  %37 = add nsw i64 %22, %36, !dbg !9
  %38 = add nsw i64 %37, %21, !dbg !9
  %39 = add nsw i64 %38, 1, !dbg !9
  %.neg = sdiv i64 %39, -4, !dbg !9
  %40 = shl nsw i64 %10, 7, !dbg !14
  %41 = shl nsw i64 %6, 12, !dbg !14
  %42 = and i64 %13, 15, !dbg !8
  %43 = lshr i64 %42, 2, !dbg !8
  %44 = lshr i64 %25, 1, !dbg !8
  %45 = mul nsw i64 %20, -18, !dbg !8
  %46 = add nsw i64 %22, %45, !dbg !8
  %47 = add nsw i64 %46, %21, !dbg !8
  %48 = add nsw i64 %47, %44, !dbg !8
  %.neg208 = sdiv i64 %48, -2, !dbg !8
  %49 = mul nuw nsw i64 %13, 18, !dbg !8
  %50 = mul nsw i64 %20, -72, !dbg !8
  %51 = add nsw i64 %49, %50, !dbg !8
  %52 = add nsw i64 %51, %43, !dbg !8
  %.neg210 = sdiv i64 %52, -4, !dbg !8
  br label %53, !dbg !17

53:                                               ; preds = %414, %4
  %54 = phi i64 [ %1209, %414 ], [ 0, %4 ], !dbg !17
  %55 = phi [2 x <8 x float>] [ %1173, %414 ], [ zeroinitializer, %4 ], !dbg !17
  %56 = phi [2 x <1 x float>] [ %785, %414 ], [ zeroinitializer, %4 ], !dbg !17
  %57 = phi [2 x <8 x float>] [ %1180, %414 ], [ zeroinitializer, %4 ], !dbg !17
  %58 = phi [2 x <8 x float>] [ %1186, %414 ], [ zeroinitializer, %4 ], !dbg !17
  %59 = phi [2 x <8 x float>] [ %1208, %414 ], [ zeroinitializer, %4 ], !dbg !17
  %.not40 = icmp sgt i64 %54, %8, !dbg !17
  br i1 %.not40, label %1210, label %60, !dbg !17

60:                                               ; preds = %53
  %61 = shl nsw i64 %54, 7, !dbg !17
  %62 = lshr i64 %54, 5, !dbg !14
  %63 = shl i64 %62, 12, !dbg !14
  %64 = sub i64 %61, %63, !dbg !14
  br label %65, !dbg !22

65:                                               ; preds = %69, %60
  %66 = phi i64 [ %235, %69 ], [ 0, %60 ], !dbg !22
  %67 = phi [2 x <32 x float>] [ %234, %69 ], [ zeroinitializer, %60 ], !dbg !22
  %68 = icmp samesign ult i64 %66, 4, !dbg !22
  br i1 %68, label %69, label %236, !dbg !22

69:                                               ; preds = %65
  %70 = shl nuw nsw i64 %66, 5, !dbg !13
  %71 = shl nuw nsw i64 %19, 19, !dbg !13
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %12) #6, !dbg !13, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local283 = add nsw i64 %30, %address.local, !dbg !13
  %address.local284 = lshr i64 %address.local, 1, !dbg !13
  %address.local285 = and i64 %address.local284, 112, !dbg !13
  %address.local286 = add nsw i64 %address.local283, %address.local285, !dbg !13
  %address.local287 = lshr i64 %address.local, 4, !dbg !15
  %address.local288 = icmp eq i64 %address.local287, 0, !dbg !13
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %33) #6, !dbg !13, !frisk.register_lifetimes !1
  %address.local289 = xor i64 %address.base, -1, !dbg !13
  %address.local290 = select i1 %address.local288, i64 0, i64 %address.local289, !dbg !13
  %address.local291 = add nsw i64 %address.local286, %address.local290, !dbg !13
  %address.local292 = add i64 %address.local291, %34, !dbg !13
  %address.local293 = lshr i32 %address.thread, 5, !dbg !13
  %address.local294 = zext nneg i32 %address.local293 to i64, !dbg !13
  %address.local295 = sub i64 %address.local292, %address.local294, !dbg !13
  %address.local296 = shl i64 %address.local295, 7, !dbg !13
  %address.local297 = add i64 %71, %address.local296, !dbg !13
  %address.local298 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %address.local297, !dbg !13
  %address.local299 = mul nsw i64 %address.local287, -496, !dbg !13
  %address.local300 = add nsw i64 %70, %address.local299, !dbg !13
  %address.local301 = shl i64 %address.local290, 5, !dbg !13
  %address.local302 = sub i64 %address.local300, %address.local301, !dbg !13
  %address.local303 = getelementptr [2 x i8], ptr addrspace(1) %address.local298, i64 %address.local302, !dbg !13
  %72 = load <8 x half>, ptr addrspace(1) %address.local303, align 2, !dbg !13
  %address.local304 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %address.local297, !dbg !13
  %address.local305 = getelementptr [2 x i8], ptr addrspace(1) %address.local304, i64 %address.local302, !dbg !13
  %address.local306 = getelementptr i8, ptr addrspace(1) %address.local305, i64 16, !dbg !13
  %73 = load <8 x half>, ptr addrspace(1) %address.local306, align 2, !dbg !13
  %74 = shufflevector <8 x half> %72, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %address.local307 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local308 = and i32 %address.local307, 448, !dbg !13
  %address.local309 = shl nuw nsw i32 %address.thread, 2, !dbg !13
  %address.local310 = and i32 %address.local309, 60, !dbg !13
  %address.local311 = or disjoint i32 %address.local308, %address.local310, !dbg !13
  %address.local312 = shl nuw nsw i32 %address.thread, 7, !dbg !13
  %address.local313 = and i32 %address.local312, 2048, !dbg !13
  %address.local314 = or disjoint i32 %address.local311, %address.local313, !dbg !13
  %address.local315 = zext nneg i32 %address.local314 to i64, !dbg !13
  %address.local316 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local315, !dbg !13
  store <4 x half> %74, ptr addrspace(3) %address.local316, align 2, !dbg !13
  %75 = shufflevector <8 x half> %72, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %address.local317 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local315, !dbg !13
  %address.local318 = getelementptr i8, ptr addrspace(3) %address.local317, i64 1024, !dbg !13
  store <4 x half> %75, ptr addrspace(3) %address.local318, align 2, !dbg !13
  %76 = shufflevector <8 x half> %73, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %address.local319 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local315, !dbg !13
  %address.local320 = getelementptr i8, ptr addrspace(3) %address.local319, i64 2048, !dbg !13
  store <4 x half> %76, ptr addrspace(3) %address.local320, align 2, !dbg !13
  %77 = shufflevector <8 x half> %73, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local315, !dbg !13
  %address.local322 = getelementptr i8, ptr addrspace(3) %address.local321, i64 3072, !dbg !13
  store <4 x half> %77, ptr addrspace(3) %address.local322, align 2, !dbg !13
  %78 = shl nuw nsw i64 %66, 10, !dbg !14
  %79 = add nuw i64 %54, %78, !dbg !14
  %.lobit107 = ashr i64 %79, 63, !dbg !14
  %80 = xor i64 %79, %.lobit107, !dbg !14
  %81 = sdiv i64 %80, 4096, !dbg !14
  %.lobit108 = ashr i64 %79, 63, !dbg !14
  %82 = xor i64 %81, %.lobit108, !dbg !14
  %83 = add nsw i64 %82, %10, !dbg !14
  %84 = or disjoint i64 %70, %40, !dbg !14
  %85 = add nsw i64 %84, %41, !dbg !14
  %86 = add nsw i64 %85, %62, !dbg !14
  %.lobit109 = ashr i64 %86, 63, !dbg !14
  %87 = xor i64 %86, %.lobit109, !dbg !14
  %88 = sdiv i64 %87, 128, !dbg !14
  %.lobit110 = ashr i64 %86, 63, !dbg !14
  %89 = xor i64 %88, %.lobit110, !dbg !14
  %90 = shl nsw i64 %89, 7, !dbg !14
  %91 = shl i64 %83, 19, !dbg !14
  %92 = and i64 %91, 16252928, !dbg !14
  %93 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %92, !dbg !14
  %address.local323 = lshr i64 %address.local, 2, !dbg !14
  %address.local324 = or disjoint i64 %64, %address.local323, !dbg !14
  %address.local325 = shl i64 %address.local324, 8, !dbg !14
  %address.local326 = getelementptr i8, ptr addrspace(1) %93, i64 %address.local325, !dbg !14
  %address.local327 = shl nuw nsw i64 %address.local, 3, !dbg !14
  %address.local328 = add nsw i64 %85, %address.local327, !dbg !14
  %address.local329 = add nsw i64 %address.local328, %62, !dbg !14
  %address.local330 = shl nuw nsw i64 %address.local323, 5, !dbg !14
  %address.local331 = add i64 %90, %address.local330, !dbg !14
  %address.local332 = sub i64 %address.local329, %address.local331, !dbg !14
  %address.local333 = getelementptr [2 x i8], ptr addrspace(1) %address.local326, i64 %address.local332, !dbg !14
  %94 = load <8 x half>, ptr addrspace(1) %address.local333, align 2, !dbg !14
  %95 = shufflevector <8 x half> %94, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %address.local334 = mul nuw nsw i64 %address.local, 2080, !dbg !14
  %address.local335 = shl nuw nsw i64 %address.local323, 3, !dbg !14
  %address.local336 = add nuw nsw i64 %address.local334, %address.local335, !dbg !14
  %address.local337 = lshr i64 %address.local, 5, !dbg !14
  %address.local338 = mul nuw nsw i64 %address.local337, 72, !dbg !14
  %address.local339 = add nuw nsw i64 %address.local336, %address.local338, !dbg !14
  %address.local340 = lshr i64 %address.local, 4, !dbg !14
  %address.local341 = shl nuw nsw i64 %address.local340, 5, !dbg !14
  %address.local342 = add nuw nsw i64 %address.local339, %address.local341, !dbg !14
  %address.local343 = lshr i64 %address.local, 7, !dbg !14
  %address.local344 = shl nuw nsw i64 %address.local343, 5, !dbg !14
  %address.local345 = add nuw nsw i64 %address.local342, %address.local344, !dbg !14
  %address.local346 = lshr i64 %address.local, 1, !dbg !14
  %address.local347 = mul nsw i64 %address.local346, -4092, !dbg !14
  %address.local348 = add i64 %address.local345, %address.local347, !dbg !14
  %address.local349 = add nuw nsw i64 %address.local340, %address.local, !dbg !14
  %address.local350 = add nuw nsw i64 %address.local349, %address.local343, !dbg !14
  %address.local351 = shl nuw nsw i64 %address.local350, 5, !dbg !14
  %address.local352 = and i64 %address.local351, 17179869120, !dbg !14
  %address.local353 = sub i64 %address.local348, %address.local352, !dbg !14
  %address.local354 = add nuw nsw i64 %address.local323, %address.local337, !dbg !14
  %address.local355 = lshr i64 %address.local354, 2, !dbg !14
  %address.local356 = mul nsw i64 %address.local355, -32, !dbg !14
  %address.local357 = add i64 %address.local353, %address.local356, !dbg !14
  %address.local358 = lshr i64 %address.local, 2, !dbg !14
  %address.local359 = mul nsw i64 %address.local358, -8, !dbg !14
  %address.local360 = add i64 %address.local357, %address.local359, !dbg !14
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local360, !dbg !14
  store <4 x half> %95, ptr addrspace(3) %address.local361, align 2, !dbg !14
  %96 = shufflevector <8 x half> %94, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %address.local362 = mul nsw i64 %address.local323, -120, !dbg !14
  %address.local363 = add nsw i64 %address.local334, %address.local362, !dbg !14
  %address.local364 = add nsw i64 %address.local363, %address.local338, !dbg !14
  %address.local365 = add nsw i64 %address.local364, %address.local341, !dbg !14
  %address.local366 = add nsw i64 %address.local365, %address.local344, !dbg !14
  %address.local367 = add i64 %address.local366, %address.local347, !dbg !14
  %address.local368 = and i64 %address.local, 3, !dbg !14
  %address.local369 = add nuw nsw i64 %address.local368, %address.local340, !dbg !14
  %address.local370 = add nuw nsw i64 %address.local369, %address.local343, !dbg !14
  %address.local371 = shl nuw nsw i64 %address.local370, 5, !dbg !14
  %address.local372 = and i64 %address.local371, 17179869120, !dbg !14
  %address.local373 = sub i64 %address.local367, %address.local372, !dbg !14
  %address.local374 = add i64 %address.local373, %address.local356, !dbg !14
  %address.local375 = add i64 %address.local374, %address.local359, !dbg !14
  %address.local376 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local375, !dbg !14
  %address.local377 = getelementptr i8, ptr addrspace(3) %address.local376, i64 2048, !dbg !14
  store <4 x half> %96, ptr addrspace(3) %address.local377, align 2, !dbg !14
  %97 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %92, !dbg !14
  %address.local378 = shl i64 %address.local324, 8, !dbg !14
  %address.local379 = getelementptr i8, ptr addrspace(1) %97, i64 %address.local378, !dbg !14
  %address.local380 = getelementptr i8, ptr addrspace(1) %address.local379, i64 16384, !dbg !14
  %address.local381 = getelementptr [2 x i8], ptr addrspace(1) %address.local380, i64 %address.local332, !dbg !14
  %98 = load <8 x half>, ptr addrspace(1) %address.local381, align 2, !dbg !14
  %99 = shufflevector <8 x half> %98, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local360, !dbg !14
  %address.local383 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !14
  store <4 x half> %99, ptr addrspace(3) %address.local383, align 2, !dbg !14
  %100 = shufflevector <8 x half> %98, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %address.local384 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local375, !dbg !14
  %address.local385 = getelementptr i8, ptr addrspace(3) %address.local384, i64 3072, !dbg !14
  store <4 x half> %100, ptr addrspace(3) %address.local385, align 2, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local386 = and i32 %address.thread, 192, !dbg !15
  %address.local387 = shl nuw nsw i32 %address.thread, 2, !dbg !15
  %address.local388 = and i32 %address.local387, 60, !dbg !15
  %address.local389 = or disjoint i32 %address.local386, %address.local388, !dbg !15
  %address.local390 = shl nuw nsw i32 %address.thread, 5, !dbg !15
  %address.local391 = and i32 %address.local390, 1536, !dbg !15
  %address.local392 = or disjoint i32 %address.local389, %address.local391, !dbg !15
  %address.local393 = zext nneg i32 %address.local392 to i64, !dbg !15
  %address.local394 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local393, !dbg !15
  %101 = load <4 x half>, ptr addrspace(3) %address.local394, align 2, !dbg !15
  %102 = extractvalue [2 x <32 x float>] %67, 0, !dbg !31
  %103 = shufflevector <32 x float> %102, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %104 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !15
  %address.local395 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local393, !dbg !15
  %address.local396 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4096, !dbg !15
  %105 = load <4 x half>, ptr addrspace(3) %address.local396, align 2, !dbg !15
  %106 = shufflevector <32 x float> %102, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %107 = fmul <4 x half> %105, splat (half 0xH3015), !dbg !15
  %108 = shufflevector <32 x float> %102, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %109 = shufflevector <32 x float> %102, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %110 = shufflevector <32 x float> %102, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %111 = shufflevector <32 x float> %102, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %112 = shufflevector <32 x float> %102, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %113 = shufflevector <32 x float> %102, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %114 = extractvalue [2 x <32 x float>] %67, 1, !dbg !31
  %115 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %116 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %117 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %118 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %119 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %120 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %121 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %122 = shufflevector <32 x float> %114, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %address.local397 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local393, !dbg !15
  %address.local398 = getelementptr i8, ptr addrspace(3) %address.local397, i64 512, !dbg !15
  %123 = load <4 x half>, ptr addrspace(3) %address.local398, align 2, !dbg !15
  %address.local399 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local393, !dbg !15
  %address.local400 = getelementptr i8, ptr addrspace(3) %address.local399, i64 4608, !dbg !15
  %124 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !15
  %address.base401 = call i64 asm sideeffect "", "=v,0"(i64 %.neg208) #6, !dbg !8, !frisk.register_lifetimes !1
  %address.local402 = shl i64 %address.base401, 6, !dbg !8
  %address.base403 = call i64 asm sideeffect "", "=v,0"(i64 %.neg210) #6, !dbg !8, !frisk.register_lifetimes !1
  %address.local404 = shl i64 %address.base403, 5, !dbg !8
  %address.local405 = add i64 %address.local402, %address.local404, !dbg !8
  %address.local406 = mul nuw nsw i64 %address.local, 400, !dbg !8
  %address.local407 = mul nsw i64 %address.local323, -1664, !dbg !8
  %address.local408 = add nsw i64 %address.local406, %address.local407, !dbg !8
  %address.local409 = and i64 %address.local, 15, !dbg !8
  %address.local410 = lshr i64 %address.local409, 2, !dbg !8
  %address.local411 = shl nuw nsw i64 %address.local410, 3, !dbg !8
  %address.local412 = add nsw i64 %address.local408, %address.local411, !dbg !8
  %address.local413 = shl nuw nsw i64 %address.local346, 5, !dbg !8
  %address.local414 = add nsw i64 %address.local412, %address.local413, !dbg !8
  %address.local415 = lshr i64 %address.local, 4, !dbg !15
  %address.local416 = and i64 %address.local415, 3, !dbg !15
  %address.local417 = shl nuw nsw i64 %address.local416, 10, !dbg !8
  %address.local418 = add nsw i64 %address.local414, %address.local417, !dbg !8
  %address.local419 = and i32 %address.thread, 32, !dbg !8
  %address.local420 = zext nneg i32 %address.local419 to i64, !dbg !8
  %address.local421 = add i64 %address.local418, %address.local420, !dbg !8
  %address.local422 = add i64 %address.local405, %address.local421, !dbg !8
  %address.local423 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local422, !dbg !8
  %125 = load <8 x half>, ptr addrspace(3) %address.local423, align 2, !dbg !8
  %address.local424 = shl nuw nsw i64 %address.local, 2, !dbg !15
  %address.local425 = mul nsw i64 %address.local323, -16, !dbg !8
  %address.local426 = add nsw i64 %address.local424, %address.local425, !dbg !8
  %address.local427 = lshr i32 %address.thread, 1, !dbg !8
  %address.local428 = and i32 %address.local427, 1, !dbg !8
  %address.local429 = zext nneg i32 %address.local428 to i64, !dbg !8
  %address.local430 = or disjoint i64 %address.local426, %address.local429, !dbg !8
  %address.local431 = lshr i64 %address.local416, 1, !dbg !8
  %address.local432 = add nuw nsw i64 %address.local430, %address.local431, !dbg !8
  %address.local433 = add nsw i64 %address.local432, 1, !dbg !8
  %address.local434 = sdiv i64 %address.local433, -2, !dbg !8
  %address.local435 = shl i64 %address.local434, 7, !dbg !8
  %address.local436 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local435, !dbg !8
  %address.local437 = shl nuw nsw i64 %address.local, 4, !dbg !8
  %address.local438 = mul nsw i64 %address.local323, -64, !dbg !8
  %address.local439 = add nsw i64 %address.local437, %address.local438, !dbg !8
  %address.local440 = or disjoint i64 %address.local439, %address.local410, !dbg !8
  %address.local441 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local442 = and i32 %address.local441, 6, !dbg !8
  %address.local443 = zext nneg i32 %address.local442 to i64, !dbg !8
  %address.local444 = add nuw nsw i64 %address.local440, %address.local443, !dbg !8
  %address.local445 = sdiv i64 %address.local444, -4, !dbg !8
  %address.local446 = shl i64 %address.local445, 6, !dbg !8
  %address.local447 = getelementptr i8, ptr addrspace(3) %address.local436, i64 %address.local446, !dbg !8
  %address.local448 = mul nuw nsw i64 %address.local443, 144, !dbg !8
  %address.local449 = getelementptr i8, ptr addrspace(3) %address.local447, i64 %address.local448, !dbg !8
  %address.local450 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local451 = and i32 %address.local450, 24, !dbg !8
  %address.local452 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local453 = and i32 %address.local452, 768, !dbg !8
  %address.local454 = or disjoint i32 %address.local451, %address.local453, !dbg !8
  %address.local455 = zext nneg i32 %address.local454 to i64, !dbg !8
  %address.local456 = getelementptr [2 x i8], ptr addrspace(3) %address.local449, i64 %address.local455, !dbg !8
  %address.local457 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local458 = and i32 %address.local457, 32, !dbg !8
  %address.local459 = zext nneg i32 %address.local458 to i64, !dbg !8
  %address.local460 = getelementptr [2 x i8], ptr addrspace(3) %address.local456, i64 %address.local459, !dbg !8
  %address.local461 = getelementptr [2 x i8], ptr addrspace(3) %address.local460, i64 %address.local417, !dbg !8
  %address.local462 = getelementptr [2 x i8], ptr addrspace(3) %address.local461, i64 %address.local420, !dbg !8
  %address.local463 = getelementptr i8, ptr addrspace(3) %address.local462, i64 64, !dbg !8
  %126 = load <8 x half>, ptr addrspace(3) %address.local463, align 2, !dbg !8
  %127 = fmul <4 x half> %123, splat (half 0xH3015), !dbg !15
  %128 = fmul <4 x half> %124, splat (half 0xH3015), !dbg !15
  %129 = shufflevector <8 x half> %125, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %130 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %129, <4 x float> %103) #7, !dbg !8
  %131 = shufflevector <8 x half> %125, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %132 = shufflevector <8 x half> %126, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %133 = shufflevector <8 x half> %126, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %address.local464 = mul nsw i64 %address.local323, -18, !dbg !8
  %address.local465 = add nsw i64 %address.local424, %address.local464, !dbg !8
  %address.local466 = add nsw i64 %address.local465, %address.local346, !dbg !8
  %address.local467 = add nsw i64 %address.local466, %address.local431, !dbg !8
  %address.local468 = sdiv i64 %address.local467, -2, !dbg !8
  %address.local469 = shl i64 %address.local468, 7, !dbg !8
  %address.local470 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local469, !dbg !8
  %address.local471 = mul nuw nsw i64 %address.local, 18, !dbg !8
  %address.local472 = mul nsw i64 %address.local323, -72, !dbg !8
  %address.local473 = add nsw i64 %address.local471, %address.local472, !dbg !8
  %address.local474 = add nsw i64 %address.local473, %address.local410, !dbg !8
  %address.local475 = add nsw i64 %address.local474, 1, !dbg !8
  %address.local476 = sdiv i64 %address.local475, -4, !dbg !8
  %address.local477 = shl i64 %address.local476, 6, !dbg !8
  %address.local478 = getelementptr i8, ptr addrspace(3) %address.local470, i64 %address.local477, !dbg !8
  %address.local479 = getelementptr [2 x i8], ptr addrspace(3) %address.local478, i64 %address.local406, !dbg !8
  %address.local480 = getelementptr [2 x i8], ptr addrspace(3) %address.local479, i64 %address.local407, !dbg !8
  %address.local481 = getelementptr [2 x i8], ptr addrspace(3) %address.local480, i64 %address.local411, !dbg !8
  %address.local482 = getelementptr [2 x i8], ptr addrspace(3) %address.local481, i64 %address.local413, !dbg !8
  %address.local483 = getelementptr [2 x i8], ptr addrspace(3) %address.local482, i64 %address.local417, !dbg !8
  %address.local484 = getelementptr [2 x i8], ptr addrspace(3) %address.local483, i64 %address.local420, !dbg !8
  %address.local485 = getelementptr i8, ptr addrspace(3) %address.local484, i64 144, !dbg !8
  %134 = load <8 x half>, ptr addrspace(3) %address.local485, align 2, !dbg !8
  %address.local486 = add nsw i64 %address.local424, %address.local425, !dbg !8
  %address.local487 = lshr i32 %address.thread, 1, !dbg !8
  %address.local488 = and i32 %address.local487, 1, !dbg !8
  %address.local489 = zext nneg i32 %address.local488 to i64, !dbg !8
  %address.local490 = or disjoint i64 %address.local486, %address.local489, !dbg !8
  %address.local491 = add nuw nsw i64 %address.local490, %address.local431, !dbg !8
  %address.local492 = add nsw i64 %address.local491, 3, !dbg !8
  %address.local493 = sdiv i64 %address.local492, -2, !dbg !8
  %address.local494 = shl i64 %address.local493, 7, !dbg !8
  %address.local495 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local494, !dbg !8
  %address.local496 = add nsw i64 %address.local437, %address.local438, !dbg !8
  %address.local497 = or disjoint i64 %address.local496, %address.local410, !dbg !8
  %address.local498 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local499 = and i32 %address.local498, 6, !dbg !8
  %address.local500 = or disjoint i32 %address.local499, 1, !dbg !8
  %address.local501 = zext nneg i32 %address.local500 to i64, !dbg !8
  %address.local502 = add nuw nsw i64 %address.local497, %address.local501, !dbg !8
  %address.local503 = sdiv i64 %address.local502, -4, !dbg !8
  %address.local504 = shl i64 %address.local503, 6, !dbg !8
  %address.local505 = getelementptr i8, ptr addrspace(3) %address.local495, i64 %address.local504, !dbg !8
  %address.local506 = mul nuw nsw i64 %address.local501, 144, !dbg !8
  %address.local507 = getelementptr i8, ptr addrspace(3) %address.local505, i64 %address.local506, !dbg !8
  %address.local508 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local509 = and i32 %address.local508, 24, !dbg !8
  %address.local510 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local511 = and i32 %address.local510, 768, !dbg !8
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !8
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !8
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) %address.local507, i64 %address.local513, !dbg !8
  %address.local515 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local516 = and i32 %address.local515, 32, !dbg !8
  %address.local517 = zext nneg i32 %address.local516 to i64, !dbg !8
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) %address.local514, i64 %address.local517, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) %address.local518, i64 %address.local417, !dbg !8
  %address.local520 = getelementptr [2 x i8], ptr addrspace(3) %address.local519, i64 %address.local420, !dbg !8
  %address.local521 = getelementptr i8, ptr addrspace(3) %address.local520, i64 192, !dbg !8
  %135 = load <8 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !8
  %136 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %131, <4 x float> %130) #7, !dbg !8
  %address.local522 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local422, !dbg !8
  %address.local523 = getelementptr i8, ptr addrspace(3) %address.local522, i64 1024, !dbg !8
  %137 = load <8 x half>, ptr addrspace(3) %address.local523, align 2, !dbg !8
  %138 = shufflevector <8 x half> %134, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %139 = shufflevector <8 x half> %134, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %140 = shufflevector <8 x half> %135, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %141 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %132, <4 x float> %106) #7, !dbg !8
  %142 = shufflevector <8 x half> %135, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %143 = shufflevector <8 x half> %137, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %144 = shufflevector <8 x half> %137, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %145 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %133, <4 x float> %141) #7, !dbg !8
  %address.local524 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local525 = and i32 %address.local524, 6, !dbg !8
  %address.local526 = or disjoint i32 %address.local525, 8, !dbg !8
  %address.local527 = mul nuw nsw i32 %address.local526, 72, !dbg !8
  %address.local528 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local529 = and i32 %address.local528, 24, !dbg !8
  %address.local530 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local531 = and i32 %address.local530, 768, !dbg !8
  %address.local532 = or disjoint i32 %address.local529, %address.local531, !dbg !8
  %address.local533 = or disjoint i32 %address.local532, 1024, !dbg !8
  %address.local534 = add nuw nsw i32 %address.local527, %address.local533, !dbg !8
  %address.local535 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local536 = and i32 %address.local535, 32, !dbg !8
  %address.local537 = or disjoint i32 %address.local536, 64, !dbg !8
  %address.local538 = add nuw nsw i32 %address.local534, %address.local537, !dbg !8
  %address.local539 = zext nneg i32 %address.local538 to i64, !dbg !8
  %address.local540 = add nuw nsw i64 %address.local417, %address.local539, !dbg !8
  %address.local541 = add nuw nsw i64 %address.local540, %address.local420, !dbg !8
  %address.local542 = shl nuw nsw i32 %address.thread, 2, !dbg !8
  %address.local543 = and i32 %address.local542, 12, !dbg !8
  %address.local544 = lshr i32 %address.thread, 1, !dbg !8
  %address.local545 = and i32 %address.local544, 1, !dbg !8
  %address.local546 = or disjoint i32 %address.local543, %address.local545, !dbg !8
  %address.local547 = or disjoint i32 %address.local546, 18, !dbg !8
  %address.local548 = zext nneg i32 %address.local547 to i64, !dbg !8
  %address.local549 = add nuw nsw i64 %address.local431, %address.local548, !dbg !8
  %address.local550 = shl nuw nsw i64 %address.local549, 5, !dbg !8
  %address.local551 = add nuw nsw i64 %address.local550, 32, !dbg !8
  %address.local552 = and i64 %address.local551, 274877906880, !dbg !8
  %address.local553 = lshr i32 %address.thread, 2, !dbg !8
  %address.local554 = and i32 %address.local553, 2, !dbg !8
  %address.local555 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local556 = and i32 %address.local555, 48, !dbg !8
  %address.local557 = or disjoint i32 %address.local554, %address.local556, !dbg !8
  %address.local558 = or disjoint i32 %address.local557, 64, !dbg !8
  %address.local559 = add nuw nsw i32 %address.local558, %address.local526, !dbg !8
  %address.local560 = shl nuw nsw i32 %address.local559, 3, !dbg !8
  %address.local561 = and i32 %address.local560, 2016, !dbg !8
  %address.local562 = zext nneg i32 %address.local561 to i64, !dbg !8
  %address.local563 = add nuw nsw i64 %address.local552, %address.local562, !dbg !8
  %address.local564 = sub nsw i64 %address.local541, %address.local563, !dbg !8
  %address.local565 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local564, !dbg !8
  %address.local566 = getelementptr i8, ptr addrspace(3) %address.local565, i64 64, !dbg !8
  %146 = load <8 x half>, ptr addrspace(3) %address.local566, align 2, !dbg !8
  %address.local567 = add nuw nsw i64 %address.local424, 18, !dbg !8
  %address.local568 = add nsw i64 %address.local567, %address.local464, !dbg !8
  %address.local569 = add nsw i64 %address.local568, %address.local346, !dbg !8
  %address.local570 = add nsw i64 %address.local569, %address.local431, !dbg !8
  %address.local571 = sdiv i64 %address.local570, -2, !dbg !8
  %address.local572 = shl i64 %address.local571, 7, !dbg !8
  %address.local573 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local572, !dbg !8
  %address.local574 = add nuw nsw i64 %address.local471, 72, !dbg !8
  %address.local575 = add nsw i64 %address.local574, %address.local472, !dbg !8
  %address.local576 = add nsw i64 %address.local575, %address.local410, !dbg !8
  %address.local577 = add nsw i64 %address.local576, -15, !dbg !8
  %address.local578 = sdiv i64 %address.local577, -4, !dbg !8
  %address.local579 = shl i64 %address.local578, 6, !dbg !8
  %address.local580 = getelementptr i8, ptr addrspace(3) %address.local573, i64 %address.local579, !dbg !8
  %address.local581 = getelementptr [2 x i8], ptr addrspace(3) %address.local580, i64 %address.local406, !dbg !8
  %address.local582 = getelementptr [2 x i8], ptr addrspace(3) %address.local581, i64 %address.local407, !dbg !8
  %address.local583 = getelementptr [2 x i8], ptr addrspace(3) %address.local582, i64 %address.local411, !dbg !8
  %address.local584 = getelementptr [2 x i8], ptr addrspace(3) %address.local583, i64 %address.local413, !dbg !8
  %address.local585 = getelementptr [2 x i8], ptr addrspace(3) %address.local584, i64 %address.local417, !dbg !8
  %address.local586 = getelementptr [2 x i8], ptr addrspace(3) %address.local585, i64 %address.local420, !dbg !8
  %address.local587 = getelementptr i8, ptr addrspace(3) %address.local586, i64 3216, !dbg !8
  %147 = load <8 x half>, ptr addrspace(3) %address.local587, align 2, !dbg !8
  %148 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %138, <4 x float> %108) #7, !dbg !8
  %149 = shufflevector <8 x half> %146, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %150 = shufflevector <8 x half> %146, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %address.local588 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local589 = and i32 %address.local588, 6, !dbg !8
  %address.local590 = or disjoint i32 %address.local589, 9, !dbg !8
  %address.local591 = mul nuw nsw i32 %address.local590, 72, !dbg !8
  %address.local592 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local593 = and i32 %address.local592, 24, !dbg !8
  %address.local594 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local595 = and i32 %address.local594, 768, !dbg !8
  %address.local596 = or disjoint i32 %address.local593, %address.local595, !dbg !8
  %address.local597 = or disjoint i32 %address.local596, 1024, !dbg !8
  %address.local598 = add nuw nsw i32 %address.local591, %address.local597, !dbg !8
  %address.local599 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local600 = and i32 %address.local599, 32, !dbg !8
  %address.local601 = or disjoint i32 %address.local600, 64, !dbg !8
  %address.local602 = add nuw nsw i32 %address.local598, %address.local601, !dbg !8
  %address.local603 = zext nneg i32 %address.local602 to i64, !dbg !8
  %address.local604 = add nuw nsw i64 %address.local417, %address.local603, !dbg !8
  %address.local605 = add nuw nsw i64 %address.local604, %address.local420, !dbg !8
  %address.local606 = shl nuw nsw i32 %address.thread, 2, !dbg !8
  %address.local607 = and i32 %address.local606, 12, !dbg !8
  %address.local608 = lshr i32 %address.thread, 1, !dbg !8
  %address.local609 = and i32 %address.local608, 1, !dbg !8
  %address.local610 = or disjoint i32 %address.local607, %address.local609, !dbg !8
  %address.local611 = or disjoint i32 %address.local610, 18, !dbg !8
  %address.local612 = zext nneg i32 %address.local611 to i64, !dbg !8
  %address.local613 = add nuw nsw i64 %address.local431, %address.local612, !dbg !8
  %address.local614 = shl nuw nsw i64 %address.local613, 5, !dbg !8
  %address.local615 = add nuw nsw i64 %address.local614, 96, !dbg !8
  %address.local616 = and i64 %address.local615, 274877906880, !dbg !8
  %address.local617 = lshr i32 %address.thread, 2, !dbg !8
  %address.local618 = and i32 %address.local617, 3, !dbg !8
  %address.local619 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local620 = and i32 %address.local619, 48, !dbg !8
  %address.local621 = or disjoint i32 %address.local618, %address.local620, !dbg !8
  %address.local622 = or disjoint i32 %address.local621, 64, !dbg !8
  %address.local623 = add nuw nsw i32 %address.local622, %address.local590, !dbg !8
  %address.local624 = shl nuw nsw i32 %address.local623, 3, !dbg !8
  %address.local625 = and i32 %address.local624, 2016, !dbg !8
  %address.local626 = zext nneg i32 %address.local625 to i64, !dbg !8
  %address.local627 = add nuw nsw i64 %address.local616, %address.local626, !dbg !8
  %address.local628 = sub nsw i64 %address.local605, %address.local627, !dbg !8
  %address.local629 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local628, !dbg !8
  %address.local630 = getelementptr i8, ptr addrspace(3) %address.local629, i64 192, !dbg !8
  %151 = load <8 x half>, ptr addrspace(3) %address.local630, align 2, !dbg !8
  %152 = shufflevector <8 x half> %147, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %153 = shufflevector <8 x half> %147, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %154 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %139, <4 x float> %148) #7, !dbg !8
  %155 = shufflevector <8 x half> %151, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %156 = shufflevector <8 x half> %151, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %157 = load <8 x half>, ptr addrspace(3) %address.local423, align 2, !dbg !8
  %158 = shufflevector <4 x float> %154, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %159 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %140, <4 x float> %109) #7, !dbg !8
  %160 = shufflevector <8 x half> %157, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %161 = shufflevector <8 x half> %157, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %162 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %142, <4 x float> %159) #7, !dbg !8
  %163 = load <8 x half>, ptr addrspace(3) %address.local463, align 2, !dbg !8
  %164 = load <8 x half>, ptr addrspace(3) %address.local485, align 2, !dbg !8
  %165 = shufflevector <4 x float> %162, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %166 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %143, <4 x float> %110) #7, !dbg !8
  %167 = shufflevector <8 x half> %163, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %168 = shufflevector <8 x half> %163, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %169 = load <8 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !8
  %170 = shufflevector <8 x half> %164, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %171 = shufflevector <8 x half> %164, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %172 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %144, <4 x float> %166) #7, !dbg !8
  %173 = shufflevector <8 x half> %169, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %174 = shufflevector <8 x half> %169, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %175 = load <8 x half>, ptr addrspace(3) %address.local523, align 2, !dbg !8
  %176 = shufflevector <4 x float> %172, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %177 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %149, <4 x float> %111) #7, !dbg !8
  %178 = shufflevector <8 x half> %175, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %179 = shufflevector <8 x half> %175, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %180 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %150, <4 x float> %177) #7, !dbg !8
  %181 = load <8 x half>, ptr addrspace(3) %address.local566, align 2, !dbg !8
  %182 = load <8 x half>, ptr addrspace(3) %address.local587, align 2, !dbg !8
  %183 = shufflevector <4 x float> %180, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %184 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %152, <4 x float> %112) #7, !dbg !8
  %185 = shufflevector <8 x half> %181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %186 = shufflevector <8 x half> %181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %187 = load <8 x half>, ptr addrspace(3) %address.local630, align 2, !dbg !8
  %188 = shufflevector <8 x half> %182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %189 = shufflevector <8 x half> %182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %190 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %153, <4 x float> %184) #7, !dbg !8
  %191 = shufflevector <8 x half> %187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %192 = shufflevector <8 x half> %187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %193 = shufflevector <4 x float> %190, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %194 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %155, <4 x float> %113) #7, !dbg !8
  %195 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %156, <4 x float> %194) #7, !dbg !8
  %196 = shufflevector <4 x float> %195, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %197 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %160, <4 x float> %115) #7, !dbg !8
  %198 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %161, <4 x float> %197) #7, !dbg !8
  %199 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %167, <4 x float> %116) #7, !dbg !8
  %200 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %168, <4 x float> %199) #7, !dbg !8
  %201 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %170, <4 x float> %117) #7, !dbg !8
  %202 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %171, <4 x float> %201) #7, !dbg !8
  %203 = shufflevector <4 x float> %202, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %204 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %173, <4 x float> %118) #7, !dbg !8
  %205 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %174, <4 x float> %204) #7, !dbg !8
  %206 = shufflevector <4 x float> %205, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %207 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %178, <4 x float> %119) #7, !dbg !8
  %208 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %179, <4 x float> %207) #7, !dbg !8
  %209 = shufflevector <4 x float> %208, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %210 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %185, <4 x float> %120) #7, !dbg !8
  %211 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %186, <4 x float> %210) #7, !dbg !8
  %212 = shufflevector <4 x float> %211, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %213 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %188, <4 x float> %121) #7, !dbg !8
  %214 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %189, <4 x float> %213) #7, !dbg !8
  %215 = shufflevector <4 x float> %214, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %216 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %127, <4 x half> %191, <4 x float> %122) #7, !dbg !8
  %217 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %192, <4 x float> %216) #7, !dbg !8
  %218 = shufflevector <4 x float> %217, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %219 = shufflevector <4 x float> %136, <4 x float> %145, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %220 = shufflevector <32 x float> %219, <32 x float> %158, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %221 = shufflevector <32 x float> %220, <32 x float> %165, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %222 = shufflevector <32 x float> %221, <32 x float> %176, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %223 = shufflevector <32 x float> %222, <32 x float> %183, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %224 = shufflevector <32 x float> %223, <32 x float> %193, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %225 = shufflevector <32 x float> %224, <32 x float> %196, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %226 = insertvalue [2 x <32 x float>] poison, <32 x float> %225, 0, !dbg !22
  %227 = shufflevector <4 x float> %198, <4 x float> %200, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %228 = shufflevector <32 x float> %227, <32 x float> %203, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %229 = shufflevector <32 x float> %228, <32 x float> %206, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %230 = shufflevector <32 x float> %229, <32 x float> %209, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %231 = shufflevector <32 x float> %230, <32 x float> %212, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %232 = shufflevector <32 x float> %231, <32 x float> %215, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %233 = shufflevector <32 x float> %232, <32 x float> %218, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %234 = insertvalue [2 x <32 x float>] %226, <32 x float> %233, 1, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %235 = add nuw nsw i64 %66, 1, !dbg !22
  br label %65, !dbg !22

236:                                              ; preds = %65
  %237 = or disjoint i64 %61, 127, !dbg !16
  %238 = icmp uge i64 %11, %237, !dbg !16
  %239 = and i1 %29, %238, !dbg !16
  br i1 %239, label %240, label %241, !dbg !16

240:                                              ; preds = %236
  br label %412, !dbg !16

241:                                              ; preds = %236
  %242 = add nsw i64 %11, %13, !dbg !16
  %243 = shl nuw nsw i64 %23, 4, !dbg !16
  %244 = add nsw i64 %242, %243, !dbg !16
  %245 = and i32 %12, 240, !dbg !16
  %index.compact = zext nneg i32 %245 to i64, !dbg !16
  %246 = sub nsw i64 %244, %index.compact, !dbg !16
  %247 = shl nuw nsw i64 %25, 4, !dbg !16
  %248 = or disjoint i64 %61, %247, !dbg !16
  %249 = add nsw i64 %246, 1, !dbg !23
  %.not43 = icmp ugt i64 %249, %248, !dbg !24
  %250 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %251 = or disjoint i64 %248, 1, !dbg !16
  %.not44 = icmp ugt i64 %249, %251, !dbg !24
  %252 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %253 = or disjoint i64 %248, 2, !dbg !16
  %.not45 = icmp ugt i64 %249, %253, !dbg !24
  %254 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %255 = or disjoint i64 %248, 3, !dbg !16
  %.not46 = icmp ugt i64 %249, %255, !dbg !24
  %256 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %257 = or disjoint i64 %248, 4, !dbg !16
  %.not47 = icmp ugt i64 %249, %257, !dbg !24
  %258 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %259 = or disjoint i64 %248, 5, !dbg !16
  %.not48 = icmp ugt i64 %249, %259, !dbg !24
  %260 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %261 = or disjoint i64 %248, 6, !dbg !16
  %.not49 = icmp ugt i64 %249, %261, !dbg !24
  %262 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %263 = or disjoint i64 %248, 7, !dbg !16
  %.not50 = icmp ugt i64 %249, %263, !dbg !24
  %264 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %265 = or disjoint i64 %248, 8, !dbg !16
  %.not51 = icmp ugt i64 %249, %265, !dbg !24
  %266 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %267 = or disjoint i64 %248, 9, !dbg !16
  %.not52 = icmp ugt i64 %249, %267, !dbg !24
  %268 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %269 = or disjoint i64 %248, 10, !dbg !16
  %.not53 = icmp ugt i64 %249, %269, !dbg !24
  %270 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %271 = or disjoint i64 %248, 11, !dbg !16
  %.not54 = icmp ugt i64 %249, %271, !dbg !24
  %272 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %273 = or disjoint i64 %248, 12, !dbg !16
  %.not55 = icmp ugt i64 %249, %273, !dbg !24
  %274 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %275 = or disjoint i64 %248, 13, !dbg !16
  %.not56 = icmp ugt i64 %249, %275, !dbg !24
  %276 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %277 = or disjoint i64 %248, 14, !dbg !16
  %.not57 = icmp ugt i64 %249, %277, !dbg !24
  %278 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %279 = or disjoint i64 %248, 15, !dbg !16
  %.not58 = icmp ugt i64 %249, %279, !dbg !24
  %280 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %281 = or disjoint i64 %248, 64, !dbg !16
  %.not59 = icmp ugt i64 %249, %281, !dbg !24
  %282 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %283 = or disjoint i64 %248, 65, !dbg !16
  %.not60 = icmp ugt i64 %249, %283, !dbg !24
  %284 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %285 = or disjoint i64 %248, 66, !dbg !16
  %.not61 = icmp ugt i64 %249, %285, !dbg !24
  %286 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %287 = or disjoint i64 %248, 67, !dbg !16
  %.not62 = icmp ugt i64 %249, %287, !dbg !24
  %288 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %289 = or disjoint i64 %248, 68, !dbg !16
  %.not63 = icmp ugt i64 %249, %289, !dbg !24
  %290 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %291 = or disjoint i64 %248, 69, !dbg !16
  %.not64 = icmp ugt i64 %249, %291, !dbg !24
  %292 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %293 = or disjoint i64 %248, 70, !dbg !16
  %.not65 = icmp ugt i64 %249, %293, !dbg !24
  %294 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %295 = or disjoint i64 %248, 71, !dbg !16
  %.not66 = icmp ugt i64 %249, %295, !dbg !24
  %296 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %297 = or disjoint i64 %248, 72, !dbg !16
  %.not67 = icmp ugt i64 %249, %297, !dbg !24
  %298 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %299 = or disjoint i64 %248, 73, !dbg !16
  %.not68 = icmp ugt i64 %249, %299, !dbg !24
  %300 = select i1 %.not68, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %301 = or disjoint i64 %248, 74, !dbg !16
  %.not69 = icmp ugt i64 %249, %301, !dbg !24
  %302 = select i1 %.not69, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %303 = or disjoint i64 %248, 75, !dbg !16
  %.not70 = icmp ugt i64 %249, %303, !dbg !24
  %304 = select i1 %.not70, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %305 = or disjoint i64 %248, 76, !dbg !16
  %.not71 = icmp ugt i64 %249, %305, !dbg !24
  %306 = select i1 %.not71, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %307 = or disjoint i64 %248, 77, !dbg !16
  %.not72 = icmp ugt i64 %249, %307, !dbg !24
  %308 = select i1 %.not72, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %309 = or disjoint i64 %248, 78, !dbg !16
  %.not73 = icmp ugt i64 %249, %309, !dbg !24
  %310 = select i1 %.not73, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %311 = or disjoint i64 %248, 79, !dbg !16
  %.not74 = icmp ugt i64 %249, %311, !dbg !24
  %312 = select i1 %.not74, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %313 = add nsw i64 %246, 65, !dbg !23
  %.not75 = icmp ugt i64 %313, %248, !dbg !24
  %314 = select i1 %.not75, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not76 = icmp ugt i64 %313, %251, !dbg !24
  %315 = select i1 %.not76, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not77 = icmp ugt i64 %313, %253, !dbg !24
  %316 = select i1 %.not77, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not78 = icmp ugt i64 %313, %255, !dbg !24
  %317 = select i1 %.not78, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not79 = icmp ugt i64 %313, %257, !dbg !24
  %318 = select i1 %.not79, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not80 = icmp ugt i64 %313, %259, !dbg !24
  %319 = select i1 %.not80, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not81 = icmp ugt i64 %313, %261, !dbg !24
  %320 = select i1 %.not81, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not82 = icmp ugt i64 %313, %263, !dbg !24
  %321 = select i1 %.not82, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not83 = icmp ugt i64 %313, %265, !dbg !24
  %322 = select i1 %.not83, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not84 = icmp ugt i64 %313, %267, !dbg !24
  %323 = select i1 %.not84, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not85 = icmp ugt i64 %313, %269, !dbg !24
  %324 = select i1 %.not85, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not86 = icmp ugt i64 %313, %271, !dbg !24
  %325 = select i1 %.not86, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not87 = icmp ugt i64 %313, %273, !dbg !24
  %326 = select i1 %.not87, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not88 = icmp ugt i64 %313, %275, !dbg !24
  %327 = select i1 %.not88, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not89 = icmp ugt i64 %313, %277, !dbg !24
  %328 = select i1 %.not89, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not90 = icmp ugt i64 %313, %279, !dbg !24
  %329 = select i1 %.not90, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not91 = icmp ugt i64 %313, %281, !dbg !24
  %330 = select i1 %.not91, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not92 = icmp ugt i64 %313, %283, !dbg !24
  %331 = select i1 %.not92, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not93 = icmp ugt i64 %313, %285, !dbg !24
  %332 = select i1 %.not93, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not94 = icmp ugt i64 %313, %287, !dbg !24
  %333 = select i1 %.not94, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not95 = icmp ugt i64 %313, %289, !dbg !24
  %334 = select i1 %.not95, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not96 = icmp ugt i64 %313, %291, !dbg !24
  %335 = select i1 %.not96, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not97 = icmp ugt i64 %313, %293, !dbg !24
  %336 = select i1 %.not97, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not98 = icmp ugt i64 %313, %295, !dbg !24
  %337 = select i1 %.not98, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not99 = icmp ugt i64 %313, %297, !dbg !24
  %338 = select i1 %.not99, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not100 = icmp ugt i64 %313, %299, !dbg !24
  %339 = select i1 %.not100, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not101 = icmp ugt i64 %313, %301, !dbg !24
  %340 = select i1 %.not101, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not102 = icmp ugt i64 %313, %303, !dbg !24
  %341 = select i1 %.not102, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not103 = icmp ugt i64 %313, %305, !dbg !24
  %342 = select i1 %.not103, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not104 = icmp ugt i64 %313, %307, !dbg !24
  %343 = select i1 %.not104, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not105 = icmp ugt i64 %313, %309, !dbg !24
  %344 = select i1 %.not105, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not106 = icmp ugt i64 %313, %311, !dbg !24
  %345 = select i1 %.not106, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %346 = insertelement <32 x float> poison, float %250, i64 0, !dbg !16
  %347 = insertelement <32 x float> %346, float %252, i64 1, !dbg !16
  %348 = insertelement <32 x float> %347, float %254, i64 2, !dbg !16
  %349 = insertelement <32 x float> %348, float %256, i64 3, !dbg !16
  %350 = insertelement <32 x float> %349, float %258, i64 4, !dbg !16
  %351 = insertelement <32 x float> %350, float %260, i64 5, !dbg !16
  %352 = insertelement <32 x float> %351, float %262, i64 6, !dbg !16
  %353 = insertelement <32 x float> %352, float %264, i64 7, !dbg !16
  %354 = insertelement <32 x float> %353, float %266, i64 8, !dbg !16
  %355 = insertelement <32 x float> %354, float %268, i64 9, !dbg !16
  %356 = insertelement <32 x float> %355, float %270, i64 10, !dbg !16
  %357 = insertelement <32 x float> %356, float %272, i64 11, !dbg !16
  %358 = insertelement <32 x float> %357, float %274, i64 12, !dbg !16
  %359 = insertelement <32 x float> %358, float %276, i64 13, !dbg !16
  %360 = insertelement <32 x float> %359, float %278, i64 14, !dbg !16
  %361 = insertelement <32 x float> %360, float %280, i64 15, !dbg !16
  %362 = insertelement <32 x float> %361, float %282, i64 16, !dbg !16
  %363 = insertelement <32 x float> %362, float %284, i64 17, !dbg !16
  %364 = insertelement <32 x float> %363, float %286, i64 18, !dbg !16
  %365 = insertelement <32 x float> %364, float %288, i64 19, !dbg !16
  %366 = insertelement <32 x float> %365, float %290, i64 20, !dbg !16
  %367 = insertelement <32 x float> %366, float %292, i64 21, !dbg !16
  %368 = insertelement <32 x float> %367, float %294, i64 22, !dbg !16
  %369 = insertelement <32 x float> %368, float %296, i64 23, !dbg !16
  %370 = insertelement <32 x float> %369, float %298, i64 24, !dbg !16
  %371 = insertelement <32 x float> %370, float %300, i64 25, !dbg !16
  %372 = insertelement <32 x float> %371, float %302, i64 26, !dbg !16
  %373 = insertelement <32 x float> %372, float %304, i64 27, !dbg !16
  %374 = insertelement <32 x float> %373, float %306, i64 28, !dbg !16
  %375 = insertelement <32 x float> %374, float %308, i64 29, !dbg !16
  %376 = insertelement <32 x float> %375, float %310, i64 30, !dbg !16
  %377 = insertelement <32 x float> %376, float %312, i64 31, !dbg !16
  %378 = insertvalue [2 x <32 x float>] poison, <32 x float> %377, 0, !dbg !16
  %379 = insertelement <32 x float> poison, float %314, i64 0, !dbg !16
  %380 = insertelement <32 x float> %379, float %315, i64 1, !dbg !16
  %381 = insertelement <32 x float> %380, float %316, i64 2, !dbg !16
  %382 = insertelement <32 x float> %381, float %317, i64 3, !dbg !16
  %383 = insertelement <32 x float> %382, float %318, i64 4, !dbg !16
  %384 = insertelement <32 x float> %383, float %319, i64 5, !dbg !16
  %385 = insertelement <32 x float> %384, float %320, i64 6, !dbg !16
  %386 = insertelement <32 x float> %385, float %321, i64 7, !dbg !16
  %387 = insertelement <32 x float> %386, float %322, i64 8, !dbg !16
  %388 = insertelement <32 x float> %387, float %323, i64 9, !dbg !16
  %389 = insertelement <32 x float> %388, float %324, i64 10, !dbg !16
  %390 = insertelement <32 x float> %389, float %325, i64 11, !dbg !16
  %391 = insertelement <32 x float> %390, float %326, i64 12, !dbg !16
  %392 = insertelement <32 x float> %391, float %327, i64 13, !dbg !16
  %393 = insertelement <32 x float> %392, float %328, i64 14, !dbg !16
  %394 = insertelement <32 x float> %393, float %329, i64 15, !dbg !16
  %395 = insertelement <32 x float> %394, float %330, i64 16, !dbg !16
  %396 = insertelement <32 x float> %395, float %331, i64 17, !dbg !16
  %397 = insertelement <32 x float> %396, float %332, i64 18, !dbg !16
  %398 = insertelement <32 x float> %397, float %333, i64 19, !dbg !16
  %399 = insertelement <32 x float> %398, float %334, i64 20, !dbg !16
  %400 = insertelement <32 x float> %399, float %335, i64 21, !dbg !16
  %401 = insertelement <32 x float> %400, float %336, i64 22, !dbg !16
  %402 = insertelement <32 x float> %401, float %337, i64 23, !dbg !16
  %403 = insertelement <32 x float> %402, float %338, i64 24, !dbg !16
  %404 = insertelement <32 x float> %403, float %339, i64 25, !dbg !16
  %405 = insertelement <32 x float> %404, float %340, i64 26, !dbg !16
  %406 = insertelement <32 x float> %405, float %341, i64 27, !dbg !16
  %407 = insertelement <32 x float> %406, float %342, i64 28, !dbg !16
  %408 = insertelement <32 x float> %407, float %343, i64 29, !dbg !16
  %409 = insertelement <32 x float> %408, float %344, i64 30, !dbg !16
  %410 = insertelement <32 x float> %409, float %345, i64 31, !dbg !16
  %411 = insertvalue [2 x <32 x float>] %378, <32 x float> %410, 1, !dbg !16
  br label %412, !dbg !16

412:                                              ; preds = %240, %241
  %413 = phi [2 x <32 x float>] [ %411, %241 ], [ zeroinitializer, %240 ], !dbg !16
  br label %414, !dbg !16

414:                                              ; preds = %412
  %415 = extractvalue [2 x <32 x float>] %67, 0, !dbg !26
  %416 = extractvalue [2 x <32 x float>] %413, 0, !dbg !26
  %417 = shufflevector <32 x float> %415, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %418 = shufflevector <32 x float> %416, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %419 = fadd <4 x float> %417, %418, !dbg !26
  %420 = extractelement <4 x float> %419, i64 0, !dbg !27
  %421 = add nsw i64 %62, %10, !dbg !9
  %422 = shl i64 %421, 19, !dbg !9
  %423 = and i64 %422, 16252928, !dbg !9
  %424 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %423, !dbg !9
  %address.thread631 = call i32 asm sideeffect "", "=v,0"(i32 %12) #6, !dbg !9, !range !34, !frisk.register_lifetimes !1
  %address.local632 = zext nneg i32 %address.thread631 to i64, !dbg !13
  %address.local633 = lshr i64 %address.local632, 3, !dbg !9
  %address.local634 = shl nuw nsw i64 %address.local633, 2, !dbg !9
  %address.local635 = or disjoint i64 %64, %address.local634, !dbg !9
  %address.local636 = shl i64 %address.local635, 7, !dbg !9
  %address.local637 = getelementptr [2 x i8], ptr addrspace(1) %424, i64 %address.local636, !dbg !9
  %address.local638 = shl nuw nsw i64 %address.local632, 2, !dbg !15
  %address.local639 = shl nuw nsw i64 %address.local633, 5, !dbg !9
  %address.local640 = sub nsw i64 %address.local638, %address.local639, !dbg !9
  %address.local641 = getelementptr [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local640, !dbg !9
  %425 = load <4 x half>, ptr addrspace(1) %address.local641, align 2, !dbg !9
  %426 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %423, !dbg !9
  %address.local642 = shl i64 %address.local635, 7, !dbg !9
  %address.local643 = or disjoint i64 %address.local642, 128, !dbg !9
  %address.local644 = getelementptr [2 x i8], ptr addrspace(1) %426, i64 %address.local643, !dbg !9
  %address.local645 = getelementptr [2 x i8], ptr addrspace(1) %address.local644, i64 %address.local640, !dbg !9
  %427 = load <4 x half>, ptr addrspace(1) %address.local645, align 2, !dbg !9
  %428 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %423, !dbg !9
  %address.local646 = shl i64 %address.local635, 7, !dbg !9
  %address.local647 = or disjoint i64 %address.local646, 256, !dbg !9
  %address.local648 = getelementptr [2 x i8], ptr addrspace(1) %428, i64 %address.local647, !dbg !9
  %address.local649 = getelementptr [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local640, !dbg !9
  %429 = load <4 x half>, ptr addrspace(1) %address.local649, align 2, !dbg !9
  %430 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %423, !dbg !9
  %address.local650 = shl i64 %address.local635, 7, !dbg !9
  %address.local651 = or disjoint i64 %address.local650, 384, !dbg !9
  %address.local652 = getelementptr [2 x i8], ptr addrspace(1) %430, i64 %address.local651, !dbg !9
  %address.local653 = getelementptr [2 x i8], ptr addrspace(1) %address.local652, i64 %address.local640, !dbg !9
  %431 = load <4 x half>, ptr addrspace(1) %address.local653, align 2, !dbg !9
  %432 = call float @llvm.amdgcn.exp2.f32(float %420), !dbg !27
  %433 = insertelement <4 x float> poison, float %432, i64 0, !dbg !27
  %434 = extractelement <4 x float> %419, i64 1, !dbg !27
  %435 = call float @llvm.amdgcn.exp2.f32(float %434), !dbg !27
  %436 = fadd float %432, %435, !dbg !28
  %437 = insertelement <4 x float> %433, float %435, i64 1, !dbg !27
  %438 = extractelement <4 x float> %419, i64 2, !dbg !27
  %439 = call float @llvm.amdgcn.exp2.f32(float %438), !dbg !27
  %440 = fadd float %436, %439, !dbg !28
  %441 = insertelement <4 x float> %437, float %439, i64 2, !dbg !27
  %442 = extractelement <4 x float> %419, i64 3, !dbg !27
  %443 = call float @llvm.amdgcn.exp2.f32(float %442), !dbg !27
  %444 = fadd float %440, %443, !dbg !28
  %445 = insertelement <4 x float> %441, float %443, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %445) #6, !dbg !29, !frisk.register_lifetimes !1
  %446 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %446) #6, !dbg !26, !frisk.register_lifetimes !1
  %447 = shufflevector <32 x float> %415, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %448 = shufflevector <32 x float> %416, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %449 = fadd <4 x float> %447, %448, !dbg !26
  %450 = extractelement <4 x float> %449, i64 0, !dbg !27
  %451 = call float @llvm.amdgcn.exp2.f32(float %450), !dbg !27
  %452 = fadd float %444, %451, !dbg !28
  %453 = insertelement <4 x float> poison, float %451, i64 0, !dbg !27
  %454 = extractelement <4 x float> %449, i64 1, !dbg !27
  %455 = call float @llvm.amdgcn.exp2.f32(float %454), !dbg !27
  %456 = fadd float %452, %455, !dbg !28
  %457 = insertelement <4 x float> %453, float %455, i64 1, !dbg !27
  %458 = extractelement <4 x float> %449, i64 2, !dbg !27
  %459 = call float @llvm.amdgcn.exp2.f32(float %458), !dbg !27
  %460 = fadd float %456, %459, !dbg !28
  %461 = insertelement <4 x float> %457, float %459, i64 2, !dbg !27
  %462 = extractelement <4 x float> %449, i64 3, !dbg !27
  %463 = call float @llvm.amdgcn.exp2.f32(float %462), !dbg !27
  %464 = fadd float %460, %463, !dbg !28
  %465 = insertelement <4 x float> %461, float %463, i64 3, !dbg !27
  %math.result253 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %465) #6, !dbg !29, !frisk.register_lifetimes !1
  %466 = fptrunc <4 x float> %math.result253 to <4 x half>, !dbg !29
  %math.packed254 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %466) #6, !dbg !26, !frisk.register_lifetimes !1
  %467 = shufflevector <32 x float> %415, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %468 = shufflevector <32 x float> %416, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %469 = fadd <4 x float> %467, %468, !dbg !26
  %470 = extractelement <4 x float> %469, i64 0, !dbg !27
  %471 = call float @llvm.amdgcn.exp2.f32(float %470), !dbg !27
  %472 = fadd float %464, %471, !dbg !28
  %473 = insertelement <4 x float> poison, float %471, i64 0, !dbg !27
  %474 = extractelement <4 x float> %469, i64 1, !dbg !27
  %475 = call float @llvm.amdgcn.exp2.f32(float %474), !dbg !27
  %476 = fadd float %472, %475, !dbg !28
  %477 = insertelement <4 x float> %473, float %475, i64 1, !dbg !27
  %478 = extractelement <4 x float> %469, i64 2, !dbg !27
  %479 = call float @llvm.amdgcn.exp2.f32(float %478), !dbg !27
  %480 = fadd float %476, %479, !dbg !28
  %481 = insertelement <4 x float> %477, float %479, i64 2, !dbg !27
  %482 = extractelement <4 x float> %469, i64 3, !dbg !27
  %483 = call float @llvm.amdgcn.exp2.f32(float %482), !dbg !27
  %484 = fadd float %480, %483, !dbg !28
  %485 = insertelement <4 x float> %481, float %483, i64 3, !dbg !27
  %math.result255 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %485) #6, !dbg !29, !frisk.register_lifetimes !1
  %486 = fptrunc <4 x float> %math.result255 to <4 x half>, !dbg !29
  %math.packed256 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %486) #6, !dbg !26, !frisk.register_lifetimes !1
  %487 = shufflevector <32 x float> %415, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %488 = shufflevector <32 x float> %416, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %489 = fadd <4 x float> %487, %488, !dbg !26
  %490 = extractelement <4 x float> %489, i64 0, !dbg !27
  %491 = call float @llvm.amdgcn.exp2.f32(float %490), !dbg !27
  %492 = fadd float %484, %491, !dbg !28
  %493 = insertelement <4 x float> poison, float %491, i64 0, !dbg !27
  %494 = extractelement <4 x float> %489, i64 1, !dbg !27
  %495 = call float @llvm.amdgcn.exp2.f32(float %494), !dbg !27
  %496 = fadd float %492, %495, !dbg !28
  %497 = insertelement <4 x float> %493, float %495, i64 1, !dbg !27
  %498 = extractelement <4 x float> %489, i64 2, !dbg !27
  %499 = call float @llvm.amdgcn.exp2.f32(float %498), !dbg !27
  %500 = fadd float %496, %499, !dbg !28
  %501 = insertelement <4 x float> %497, float %499, i64 2, !dbg !27
  %502 = extractelement <4 x float> %489, i64 3, !dbg !27
  %503 = call float @llvm.amdgcn.exp2.f32(float %502), !dbg !27
  %504 = fadd float %500, %503, !dbg !28
  %505 = insertelement <4 x float> %501, float %503, i64 3, !dbg !27
  %math.result257 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %505) #6, !dbg !29, !frisk.register_lifetimes !1
  %506 = fptrunc <4 x float> %math.result257 to <4 x half>, !dbg !29
  %math.packed258 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %506) #6, !dbg !26, !frisk.register_lifetimes !1
  %507 = shufflevector <32 x float> %415, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %508 = shufflevector <32 x float> %416, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %509 = fadd <4 x float> %507, %508, !dbg !26
  %510 = extractelement <4 x float> %509, i64 0, !dbg !27
  %511 = call float @llvm.amdgcn.exp2.f32(float %510), !dbg !27
  %512 = fadd float %504, %511, !dbg !28
  %513 = insertelement <4 x float> poison, float %511, i64 0, !dbg !27
  %514 = extractelement <4 x float> %509, i64 1, !dbg !27
  %515 = call float @llvm.amdgcn.exp2.f32(float %514), !dbg !27
  %516 = fadd float %512, %515, !dbg !28
  %517 = insertelement <4 x float> %513, float %515, i64 1, !dbg !27
  %518 = extractelement <4 x float> %509, i64 2, !dbg !27
  %519 = call float @llvm.amdgcn.exp2.f32(float %518), !dbg !27
  %520 = fadd float %516, %519, !dbg !28
  %521 = insertelement <4 x float> %517, float %519, i64 2, !dbg !27
  %522 = extractelement <4 x float> %509, i64 3, !dbg !27
  %523 = call float @llvm.amdgcn.exp2.f32(float %522), !dbg !27
  %524 = fadd float %520, %523, !dbg !28
  %525 = insertelement <4 x float> %521, float %523, i64 3, !dbg !27
  %math.result259 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %525) #6, !dbg !29, !frisk.register_lifetimes !1
  %526 = fptrunc <4 x float> %math.result259 to <4 x half>, !dbg !29
  %math.packed260 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %526) #6, !dbg !26, !frisk.register_lifetimes !1
  %527 = shufflevector <32 x float> %415, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %528 = shufflevector <32 x float> %416, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %529 = fadd <4 x float> %527, %528, !dbg !26
  %530 = extractelement <4 x float> %529, i64 0, !dbg !27
  %531 = call float @llvm.amdgcn.exp2.f32(float %530), !dbg !27
  %532 = fadd float %524, %531, !dbg !28
  %533 = insertelement <4 x float> poison, float %531, i64 0, !dbg !27
  %534 = extractelement <4 x float> %529, i64 1, !dbg !27
  %535 = call float @llvm.amdgcn.exp2.f32(float %534), !dbg !27
  %536 = fadd float %532, %535, !dbg !28
  %537 = insertelement <4 x float> %533, float %535, i64 1, !dbg !27
  %538 = extractelement <4 x float> %529, i64 2, !dbg !27
  %539 = call float @llvm.amdgcn.exp2.f32(float %538), !dbg !27
  %540 = fadd float %536, %539, !dbg !28
  %541 = insertelement <4 x float> %537, float %539, i64 2, !dbg !27
  %542 = extractelement <4 x float> %529, i64 3, !dbg !27
  %543 = call float @llvm.amdgcn.exp2.f32(float %542), !dbg !27
  %544 = fadd float %540, %543, !dbg !28
  %545 = insertelement <4 x float> %541, float %543, i64 3, !dbg !27
  %math.result261 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %545) #6, !dbg !29, !frisk.register_lifetimes !1
  %546 = fptrunc <4 x float> %math.result261 to <4 x half>, !dbg !29
  %math.packed262 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %546) #6, !dbg !26, !frisk.register_lifetimes !1
  %547 = shufflevector <32 x float> %415, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %548 = shufflevector <32 x float> %416, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %549 = fadd <4 x float> %547, %548, !dbg !26
  %550 = extractelement <4 x float> %549, i64 0, !dbg !27
  %551 = call float @llvm.amdgcn.exp2.f32(float %550), !dbg !27
  %552 = fadd float %544, %551, !dbg !28
  %553 = insertelement <4 x float> poison, float %551, i64 0, !dbg !27
  %554 = extractelement <4 x float> %549, i64 1, !dbg !27
  %555 = call float @llvm.amdgcn.exp2.f32(float %554), !dbg !27
  %556 = fadd float %552, %555, !dbg !28
  %557 = insertelement <4 x float> %553, float %555, i64 1, !dbg !27
  %558 = extractelement <4 x float> %549, i64 2, !dbg !27
  %559 = call float @llvm.amdgcn.exp2.f32(float %558), !dbg !27
  %560 = fadd float %556, %559, !dbg !28
  %561 = insertelement <4 x float> %557, float %559, i64 2, !dbg !27
  %562 = extractelement <4 x float> %549, i64 3, !dbg !27
  %563 = call float @llvm.amdgcn.exp2.f32(float %562), !dbg !27
  %564 = fadd float %560, %563, !dbg !28
  %565 = insertelement <4 x float> %561, float %563, i64 3, !dbg !27
  %math.result263 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %565) #6, !dbg !29, !frisk.register_lifetimes !1
  %566 = fptrunc <4 x float> %math.result263 to <4 x half>, !dbg !29
  %math.packed264 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %566) #6, !dbg !26, !frisk.register_lifetimes !1
  %567 = shufflevector <32 x float> %415, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %568 = shufflevector <32 x float> %416, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %569 = fadd <4 x float> %567, %568, !dbg !26
  %570 = extractelement <4 x float> %569, i64 0, !dbg !27
  %571 = call float @llvm.amdgcn.exp2.f32(float %570), !dbg !27
  %572 = fadd float %564, %571, !dbg !28
  %573 = insertelement <4 x float> poison, float %571, i64 0, !dbg !27
  %574 = extractelement <4 x float> %569, i64 1, !dbg !27
  %575 = call float @llvm.amdgcn.exp2.f32(float %574), !dbg !27
  %576 = fadd float %572, %575, !dbg !28
  %577 = insertelement <4 x float> %573, float %575, i64 1, !dbg !27
  %578 = extractelement <4 x float> %569, i64 2, !dbg !27
  %579 = call float @llvm.amdgcn.exp2.f32(float %578), !dbg !27
  %580 = fadd float %576, %579, !dbg !28
  %581 = insertelement <4 x float> %577, float %579, i64 2, !dbg !27
  %582 = extractelement <4 x float> %569, i64 3, !dbg !27
  %583 = call float @llvm.amdgcn.exp2.f32(float %582), !dbg !27
  %584 = fadd float %580, %583, !dbg !28
  %585 = insertelement <4 x float> %581, float %583, i64 3, !dbg !27
  %math.result265 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %585) #6, !dbg !29, !frisk.register_lifetimes !1
  %586 = fptrunc <4 x float> %math.result265 to <4 x half>, !dbg !29
  %math.packed266 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %586) #6, !dbg !26, !frisk.register_lifetimes !1
  %587 = extractvalue [2 x <32 x float>] %67, 1, !dbg !26
  %588 = extractvalue [2 x <32 x float>] %413, 1, !dbg !26
  %589 = shufflevector <32 x float> %587, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %590 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %591 = fadd <4 x float> %589, %590, !dbg !26
  %592 = extractelement <4 x float> %591, i64 0, !dbg !27
  %593 = call float @llvm.amdgcn.exp2.f32(float %592), !dbg !27
  %594 = insertelement <4 x float> poison, float %593, i64 0, !dbg !27
  %595 = extractelement <4 x float> %591, i64 1, !dbg !27
  %596 = call float @llvm.amdgcn.exp2.f32(float %595), !dbg !27
  %597 = fadd float %593, %596, !dbg !28
  %598 = insertelement <4 x float> %594, float %596, i64 1, !dbg !27
  %599 = extractelement <4 x float> %591, i64 2, !dbg !27
  %600 = call float @llvm.amdgcn.exp2.f32(float %599), !dbg !27
  %601 = fadd float %597, %600, !dbg !28
  %602 = insertelement <4 x float> %598, float %600, i64 2, !dbg !27
  %603 = extractelement <4 x float> %591, i64 3, !dbg !27
  %604 = call float @llvm.amdgcn.exp2.f32(float %603), !dbg !27
  %605 = fadd float %601, %604, !dbg !28
  %606 = insertelement <4 x float> %602, float %604, i64 3, !dbg !27
  %math.result267 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %606) #6, !dbg !29, !frisk.register_lifetimes !1
  %607 = fptrunc <4 x float> %math.result267 to <4 x half>, !dbg !29
  %math.packed268 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %607) #6, !dbg !26, !frisk.register_lifetimes !1
  %608 = shufflevector <32 x float> %587, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %609 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %610 = fadd <4 x float> %608, %609, !dbg !26
  %611 = extractelement <4 x float> %610, i64 0, !dbg !27
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !27
  %613 = fadd float %605, %612, !dbg !28
  %614 = insertelement <4 x float> poison, float %612, i64 0, !dbg !27
  %615 = extractelement <4 x float> %610, i64 1, !dbg !27
  %616 = call float @llvm.amdgcn.exp2.f32(float %615), !dbg !27
  %617 = fadd float %613, %616, !dbg !28
  %618 = insertelement <4 x float> %614, float %616, i64 1, !dbg !27
  %619 = extractelement <4 x float> %610, i64 2, !dbg !27
  %620 = call float @llvm.amdgcn.exp2.f32(float %619), !dbg !27
  %621 = fadd float %617, %620, !dbg !28
  %622 = insertelement <4 x float> %618, float %620, i64 2, !dbg !27
  %623 = extractelement <4 x float> %610, i64 3, !dbg !27
  %624 = call float @llvm.amdgcn.exp2.f32(float %623), !dbg !27
  %625 = fadd float %621, %624, !dbg !28
  %626 = insertelement <4 x float> %622, float %624, i64 3, !dbg !27
  %math.result269 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %626) #6, !dbg !29, !frisk.register_lifetimes !1
  %627 = fptrunc <4 x float> %math.result269 to <4 x half>, !dbg !29
  %math.packed270 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %627) #6, !dbg !26, !frisk.register_lifetimes !1
  %628 = shufflevector <32 x float> %587, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %629 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %630 = fadd <4 x float> %628, %629, !dbg !26
  %631 = extractelement <4 x float> %630, i64 0, !dbg !27
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !27
  %633 = fadd float %625, %632, !dbg !28
  %634 = insertelement <4 x float> poison, float %632, i64 0, !dbg !27
  %635 = extractelement <4 x float> %630, i64 1, !dbg !27
  %636 = call float @llvm.amdgcn.exp2.f32(float %635), !dbg !27
  %637 = fadd float %633, %636, !dbg !28
  %638 = insertelement <4 x float> %634, float %636, i64 1, !dbg !27
  %639 = extractelement <4 x float> %630, i64 2, !dbg !27
  %640 = call float @llvm.amdgcn.exp2.f32(float %639), !dbg !27
  %641 = fadd float %637, %640, !dbg !28
  %642 = insertelement <4 x float> %638, float %640, i64 2, !dbg !27
  %643 = extractelement <4 x float> %630, i64 3, !dbg !27
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !27
  %645 = fadd float %641, %644, !dbg !28
  %646 = insertelement <4 x float> %642, float %644, i64 3, !dbg !27
  %math.result271 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %646) #6, !dbg !29, !frisk.register_lifetimes !1
  %647 = fptrunc <4 x float> %math.result271 to <4 x half>, !dbg !29
  %math.packed272 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %647) #6, !dbg !26, !frisk.register_lifetimes !1
  %648 = shufflevector <32 x float> %587, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %649 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %650 = fadd <4 x float> %648, %649, !dbg !26
  %651 = extractelement <4 x float> %650, i64 0, !dbg !27
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !27
  %653 = fadd float %645, %652, !dbg !28
  %654 = insertelement <4 x float> poison, float %652, i64 0, !dbg !27
  %655 = extractelement <4 x float> %650, i64 1, !dbg !27
  %656 = call float @llvm.amdgcn.exp2.f32(float %655), !dbg !27
  %657 = fadd float %653, %656, !dbg !28
  %658 = insertelement <4 x float> %654, float %656, i64 1, !dbg !27
  %659 = extractelement <4 x float> %650, i64 2, !dbg !27
  %660 = call float @llvm.amdgcn.exp2.f32(float %659), !dbg !27
  %661 = fadd float %657, %660, !dbg !28
  %662 = insertelement <4 x float> %658, float %660, i64 2, !dbg !27
  %663 = extractelement <4 x float> %650, i64 3, !dbg !27
  %664 = call float @llvm.amdgcn.exp2.f32(float %663), !dbg !27
  %665 = fadd float %661, %664, !dbg !28
  %666 = insertelement <4 x float> %662, float %664, i64 3, !dbg !27
  %math.result273 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %666) #6, !dbg !29, !frisk.register_lifetimes !1
  %667 = fptrunc <4 x float> %math.result273 to <4 x half>, !dbg !29
  %math.packed274 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %667) #6, !dbg !26, !frisk.register_lifetimes !1
  %668 = shufflevector <32 x float> %587, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %669 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %670 = fadd <4 x float> %668, %669, !dbg !26
  %671 = extractelement <4 x float> %670, i64 0, !dbg !27
  %672 = call float @llvm.amdgcn.exp2.f32(float %671), !dbg !27
  %673 = fadd float %665, %672, !dbg !28
  %674 = insertelement <4 x float> poison, float %672, i64 0, !dbg !27
  %675 = extractelement <4 x float> %670, i64 1, !dbg !27
  %676 = call float @llvm.amdgcn.exp2.f32(float %675), !dbg !27
  %677 = fadd float %673, %676, !dbg !28
  %678 = insertelement <4 x float> %674, float %676, i64 1, !dbg !27
  %679 = extractelement <4 x float> %670, i64 2, !dbg !27
  %680 = call float @llvm.amdgcn.exp2.f32(float %679), !dbg !27
  %681 = fadd float %677, %680, !dbg !28
  %682 = insertelement <4 x float> %678, float %680, i64 2, !dbg !27
  %683 = extractelement <4 x float> %670, i64 3, !dbg !27
  %684 = call float @llvm.amdgcn.exp2.f32(float %683), !dbg !27
  %685 = fadd float %681, %684, !dbg !28
  %686 = insertelement <4 x float> %682, float %684, i64 3, !dbg !27
  %math.result275 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %686) #6, !dbg !29, !frisk.register_lifetimes !1
  %687 = fptrunc <4 x float> %math.result275 to <4 x half>, !dbg !29
  %math.packed276 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %687) #6, !dbg !26, !frisk.register_lifetimes !1
  %688 = shufflevector <32 x float> %587, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %689 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %690 = fadd <4 x float> %688, %689, !dbg !26
  %691 = extractelement <4 x float> %690, i64 0, !dbg !27
  %692 = call float @llvm.amdgcn.exp2.f32(float %691), !dbg !27
  %693 = fadd float %685, %692, !dbg !28
  %694 = insertelement <4 x float> poison, float %692, i64 0, !dbg !27
  %695 = extractelement <4 x float> %690, i64 1, !dbg !27
  %696 = call float @llvm.amdgcn.exp2.f32(float %695), !dbg !27
  %697 = fadd float %693, %696, !dbg !28
  %698 = insertelement <4 x float> %694, float %696, i64 1, !dbg !27
  %699 = extractelement <4 x float> %690, i64 2, !dbg !27
  %700 = call float @llvm.amdgcn.exp2.f32(float %699), !dbg !27
  %701 = fadd float %697, %700, !dbg !28
  %702 = insertelement <4 x float> %698, float %700, i64 2, !dbg !27
  %703 = extractelement <4 x float> %690, i64 3, !dbg !27
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !27
  %705 = fadd float %701, %704, !dbg !28
  %706 = insertelement <4 x float> %702, float %704, i64 3, !dbg !27
  %math.result277 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %706) #6, !dbg !29, !frisk.register_lifetimes !1
  %707 = fptrunc <4 x float> %math.result277 to <4 x half>, !dbg !29
  %math.packed278 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %707) #6, !dbg !26, !frisk.register_lifetimes !1
  %708 = shufflevector <32 x float> %587, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %709 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %710 = fadd <4 x float> %708, %709, !dbg !26
  %711 = extractelement <4 x float> %710, i64 0, !dbg !27
  %712 = call float @llvm.amdgcn.exp2.f32(float %711), !dbg !27
  %713 = fadd float %705, %712, !dbg !28
  %714 = insertelement <4 x float> poison, float %712, i64 0, !dbg !27
  %715 = extractelement <4 x float> %710, i64 1, !dbg !27
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !27
  %717 = fadd float %713, %716, !dbg !28
  %718 = insertelement <4 x float> %714, float %716, i64 1, !dbg !27
  %719 = extractelement <4 x float> %710, i64 2, !dbg !27
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !27
  %721 = fadd float %717, %720, !dbg !28
  %722 = insertelement <4 x float> %718, float %720, i64 2, !dbg !27
  %723 = extractelement <4 x float> %710, i64 3, !dbg !27
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !27
  %725 = fadd float %721, %724, !dbg !28
  %726 = insertelement <4 x float> %722, float %724, i64 3, !dbg !27
  %math.result279 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %726) #6, !dbg !29, !frisk.register_lifetimes !1
  %727 = fptrunc <4 x float> %math.result279 to <4 x half>, !dbg !29
  %math.packed280 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %727) #6, !dbg !26, !frisk.register_lifetimes !1
  %728 = shufflevector <32 x float> %587, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %729 = shufflevector <32 x float> %588, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %730 = fadd <4 x float> %728, %729, !dbg !26
  %731 = extractelement <4 x float> %730, i64 0, !dbg !27
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !27
  %733 = fadd float %725, %732, !dbg !28
  %734 = insertelement <4 x float> poison, float %732, i64 0, !dbg !27
  %735 = extractelement <4 x float> %730, i64 1, !dbg !27
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !27
  %737 = fadd float %733, %736, !dbg !28
  %738 = insertelement <4 x float> %734, float %736, i64 1, !dbg !27
  %739 = extractelement <4 x float> %730, i64 2, !dbg !27
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !27
  %741 = fadd float %737, %740, !dbg !28
  %742 = insertelement <4 x float> %738, float %740, i64 2, !dbg !27
  %743 = extractelement <4 x float> %730, i64 3, !dbg !27
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !27
  %745 = fadd float %741, %744, !dbg !28
  %746 = insertelement <4 x float> %742, float %744, i64 3, !dbg !27
  %math.result281 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %746) #6, !dbg !29, !frisk.register_lifetimes !1
  %747 = fptrunc <4 x float> %math.result281 to <4 x half>, !dbg !29
  %math.packed282 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %747) #6, !dbg !28, !frisk.register_lifetimes !1
  %748 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %749 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %748), !dbg !28
  %750 = add nuw nsw i32 %749, 64, !dbg !28
  %751 = xor i32 %749, 16, !dbg !28
  %752 = and i32 %750, 192, !dbg !28
  %753 = icmp samesign ult i32 %751, %752, !dbg !28
  %754 = select i1 %753, i32 %751, i32 %749, !dbg !28
  %755 = shl nuw nsw i32 %754, 2, !dbg !28
  %756 = bitcast float %584 to i32, !dbg !28
  %757 = call i32 @llvm.amdgcn.ds.bpermute(i32 %755, i32 %756), !dbg !28
  %758 = bitcast i32 %757 to float, !dbg !28
  %759 = fadd float %584, %758, !dbg !28
  %760 = xor i32 %749, 32, !dbg !28
  %761 = icmp samesign ult i32 %760, %752, !dbg !28
  %762 = select i1 %761, i32 %760, i32 %749, !dbg !28
  %763 = shl nuw nsw i32 %762, 2, !dbg !28
  %764 = bitcast float %759 to i32, !dbg !28
  %765 = call i32 @llvm.amdgcn.ds.bpermute(i32 %763, i32 %764), !dbg !28
  %766 = bitcast i32 %765 to float, !dbg !28
  %767 = fadd float %759, %766, !dbg !28
  %768 = extractvalue [2 x <1 x float>] %56, 0, !dbg !28
  %769 = extractelement <1 x float> %768, i64 0, !dbg !28
  %770 = fadd float %767, %769, !dbg !28
  %771 = insertelement <1 x float> poison, float %770, i64 0, !dbg !17
  %772 = insertvalue [2 x <1 x float>] poison, <1 x float> %771, 0, !dbg !17
  %773 = bitcast float %745 to i32, !dbg !28
  %774 = call i32 @llvm.amdgcn.ds.bpermute(i32 %755, i32 %773), !dbg !28
  %775 = bitcast i32 %774 to float, !dbg !28
  %776 = fadd float %745, %775, !dbg !28
  %777 = bitcast float %776 to i32, !dbg !28
  %778 = call i32 @llvm.amdgcn.ds.bpermute(i32 %763, i32 %777), !dbg !28
  %779 = bitcast i32 %778 to float, !dbg !28
  %780 = fadd float %776, %779, !dbg !28
  %781 = extractvalue [2 x <1 x float>] %56, 1, !dbg !28
  %782 = extractelement <1 x float> %781, i64 0, !dbg !28
  %783 = fadd float %780, %782, !dbg !28
  %784 = insertelement <1 x float> poison, float %783, i64 0, !dbg !17
  %785 = insertvalue [2 x <1 x float>] %772, <1 x float> %784, 1, !dbg !17
  %786 = shufflevector <4 x half> %425, <4 x half> %427, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %787 = shufflevector <4 x half> %786, <4 x half> %429, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %788 = shufflevector <4 x half> %787, <4 x half> %431, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
  %address.local654 = shl nuw nsw i64 %address.local632, 5, !dbg !9
  %address.local655 = lshr i64 %address.local632, 1, !dbg !14
  %address.local656 = mul nuw nsw i64 %address.local655, 72, !dbg !9
  %address.local657 = add nuw nsw i64 %address.local654, %address.local656, !dbg !9
  %address.local658 = mul nsw i64 %address.local633, -540, !dbg !9
  %address.local659 = add nsw i64 %address.local657, %address.local658, !dbg !9
  %address.local660 = lshr i64 %address.local632, 4, !dbg !9
  %address.local661 = mul nuw nsw i64 %address.local660, 280, !dbg !9
  %address.local662 = add nsw i64 %address.local659, %address.local661, !dbg !9
  %address.local663 = lshr i32 %address.thread631, 4, !dbg !9
  %address.local664 = and i32 %address.thread631, 7, !dbg !9
  %address.local665 = add nuw nsw i32 %address.local663, %address.local664, !dbg !9
  %address.local666 = shl nuw nsw i32 %address.local665, 5, !dbg !9
  %address.local667 = and i32 %address.local666, 960, !dbg !9
  %address.local668 = zext nneg i32 %address.local667 to i64, !dbg !9
  %address.local669 = sub nsw i64 %address.local662, %address.local668, !dbg !9
  %address.local670 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local669, !dbg !9
  store <4 x half> %788, ptr addrspace(3) %address.local670, align 2, !dbg !9
  %789 = shufflevector <4 x half> %425, <4 x half> %427, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %790 = shufflevector <4 x half> %789, <4 x half> %429, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %791 = shufflevector <4 x half> %790, <4 x half> %431, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
  %address.base671 = call i64 asm sideeffect "", "=v,0"(i64 %.neg) #6, !dbg !9, !frisk.register_lifetimes !1
  %address.local672 = shl i64 %address.base671, 6, !dbg !9
  %address.local673 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local672, !dbg !9
  %address.local674 = shl nuw nsw i64 %address.local632, 6, !dbg !9
  %address.local675 = mul nsw i64 %address.local633, -828, !dbg !9
  %address.local676 = add nsw i64 %address.local674, %address.local675, !dbg !9
  %address.local677 = add nsw i64 %address.local676, %address.local656, !dbg !9
  %address.local678 = shl nuw nsw i64 %address.local632, 2, !dbg !9
  %address.local679 = and i64 %address.local678, 992, !dbg !9
  %address.local680 = add nsw i64 %address.local677, %address.local679, !dbg !9
  %address.local681 = add nsw i64 %address.local680, %address.local661, !dbg !9
  %address.local682 = lshr i32 %address.thread631, 4, !dbg !9
  %address.local683 = and i32 %address.thread631, 7, !dbg !9
  %address.local684 = add nuw nsw i32 %address.local682, %address.local683, !dbg !9
  %address.local685 = shl nuw nsw i32 %address.local684, 5, !dbg !9
  %address.local686 = and i32 %address.local685, 960, !dbg !9
  %address.local687 = zext nneg i32 %address.local686 to i64, !dbg !9
  %address.local688 = sub i64 %address.local681, %address.local687, !dbg !9
  %address.local689 = getelementptr [2 x i8], ptr addrspace(3) %address.local673, i64 %address.local688, !dbg !9
  %address.local690 = getelementptr i8, ptr addrspace(3) %address.local689, i64 16, !dbg !9
  store <4 x half> %791, ptr addrspace(3) %address.local690, align 2, !dbg !9
  %792 = shufflevector <4 x half> %425, <4 x half> %427, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %793 = shufflevector <4 x half> %792, <4 x half> %429, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %794 = shufflevector <4 x half> %793, <4 x half> %431, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  %address.local691 = mul nsw i64 %address.local633, -36, !dbg !9
  %address.local692 = add nsw i64 %address.local638, %address.local691, !dbg !9
  %address.local693 = add nsw i64 %address.local692, %address.local655, !dbg !9
  %address.local694 = add nsw i64 %address.local693, 2, !dbg !9
  %address.local695 = sdiv i64 %address.local694, -4, !dbg !9
  %address.local696 = shl i64 %address.local695, 6, !dbg !9
  %address.local697 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local696, !dbg !9
  %address.local698 = getelementptr [2 x i8], ptr addrspace(3) %address.local697, i64 %address.local688, !dbg !9
  %address.local699 = getelementptr i8, ptr addrspace(3) %address.local698, i64 32, !dbg !9
  store <4 x half> %794, ptr addrspace(3) %address.local699, align 2, !dbg !9
  %795 = shufflevector <4 x half> %425, <4 x half> %427, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %796 = shufflevector <4 x half> %795, <4 x half> %429, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %797 = shufflevector <4 x half> %796, <4 x half> %431, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  %address.local700 = add nsw i64 %address.local693, 3, !dbg !9
  %address.local701 = sdiv i64 %address.local700, -4, !dbg !9
  %address.local702 = shl i64 %address.local701, 6, !dbg !9
  %address.local703 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local702, !dbg !9
  %address.local704 = getelementptr [2 x i8], ptr addrspace(3) %address.local703, i64 %address.local688, !dbg !9
  %address.local705 = getelementptr i8, ptr addrspace(3) %address.local704, i64 48, !dbg !9
  store <4 x half> %797, ptr addrspace(3) %address.local705, align 2, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local706 = shl nuw nsw i64 %address.local632, 3, !dbg !14
  %address.local707 = lshr i64 %address.local632, 4, !dbg !15
  %address.local708 = mul nsw i64 %address.local707, -128, !dbg !18
  %address.local709 = add nsw i64 %address.local706, %address.local708, !dbg !18
  %address.local710 = and i64 %address.local632, 15, !dbg !8
  %address.local711 = lshr i64 %address.local710, 3, !dbg !18
  %address.local712 = mul nuw nsw i64 %address.local711, 72, !dbg !18
  %address.local713 = add nsw i64 %address.local709, %address.local712, !dbg !18
  %address.local714 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local713, !dbg !18
  %address.local715 = lshr i64 %address.local710, 2, !dbg !8
  %address.local716 = shl nuw nsw i64 %address.local715, 5, !dbg !18
  %address.local717 = getelementptr [2 x i8], ptr addrspace(3) %address.local714, i64 %address.local716, !dbg !18
  %address.local718 = lshr i64 %address.local710, 3, !dbg !18
  %address.local719 = mul nuw nsw i64 %address.local718, -64, !dbg !18
  %address.local720 = getelementptr [2 x i8], ptr addrspace(3) %address.local717, i64 %address.local719, !dbg !18
  %address.local721 = add nuw nsw i64 %address.local710, %address.local711, !dbg !18
  %address.local722 = lshr i64 %address.local721, 2, !dbg !18
  %address.local723 = mul nsw i64 %address.local722, -32, !dbg !18
  %address.local724 = getelementptr [2 x i8], ptr addrspace(3) %address.local720, i64 %address.local723, !dbg !18
  %address.local725 = lshr i64 %address.local632, 4, !dbg !15
  %address.local726 = and i64 %address.local725, 3, !dbg !15
  %address.local727 = shl nuw nsw i64 %address.local726, 9, !dbg !15
  %address.local728 = getelementptr [2 x i8], ptr addrspace(3) %address.local724, i64 %address.local727, !dbg !18
  %798 = load <8 x half>, ptr addrspace(3) %address.local728, align 2, !dbg !18
  %799 = extractvalue [2 x <8 x float>] %55, 0, !dbg !30
  %800 = shufflevector <8 x float> %799, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %801 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %802 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %801, <4 x float> %800) #7, !dbg !18
  %address.local729 = mul nuw nsw i64 %address.local726, 576, !dbg !18
  %address.local730 = add nuw nsw i64 %address.local706, %address.local729, !dbg !18
  %address.local731 = add nsw i64 %address.local730, %address.local708, !dbg !18
  %address.local732 = add nsw i64 %address.local731, %address.local712, !dbg !18
  %address.local733 = add nsw i64 %address.local732, %address.local716, !dbg !18
  %address.local734 = shl nuw nsw i32 %address.thread631, 2, !dbg !18
  %address.local735 = and i32 %address.local734, 192, !dbg !18
  %address.local736 = or disjoint i32 %address.local735, 32, !dbg !18
  %address.local737 = shl nuw nsw i32 %address.thread631, 3, !dbg !18
  %address.local738 = and i32 %address.local737, 96, !dbg !18
  %address.local739 = add nuw nsw i32 %address.local736, %address.local738, !dbg !18
  %address.local740 = and i32 %address.local739, 448, !dbg !18
  %address.local741 = zext nneg i32 %address.local740 to i64, !dbg !18
  %address.local742 = sub nsw i64 %address.local733, %address.local741, !dbg !18
  %address.local743 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local742, !dbg !18
  %address.local744 = getelementptr [2 x i8], ptr addrspace(3) %address.local743, i64 %address.local723, !dbg !18
  %address.local745 = getelementptr i8, ptr addrspace(3) %address.local744, i64 576, !dbg !18
  %803 = load <8 x half>, ptr addrspace(3) %address.local745, align 2, !dbg !18
  %804 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %805 = shufflevector <8 x float> %799, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %806 = extractvalue [2 x <8 x float>] %55, 1, !dbg !30
  %807 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %804, <4 x float> %802) #7, !dbg !18
  %808 = shufflevector <8 x half> %803, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %809 = shufflevector <8 x half> %803, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %810 = shufflevector <8 x float> %806, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %811 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %808, <4 x float> %807) #7, !dbg !18
  %812 = shufflevector <8 x float> %806, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %813 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %809, <4 x float> %811) #7, !dbg !18
  %address.local746 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local706, !dbg !18
  %address.local747 = getelementptr [2 x i8], ptr addrspace(3) %address.local746, i64 %address.local727, !dbg !18
  %address.local748 = getelementptr [2 x i8], ptr addrspace(3) %address.local747, i64 %address.local708, !dbg !18
  %address.local749 = getelementptr [2 x i8], ptr addrspace(3) %address.local748, i64 %address.local712, !dbg !18
  %address.local750 = getelementptr [2 x i8], ptr addrspace(3) %address.local749, i64 %address.local716, !dbg !18
  %address.local751 = getelementptr [2 x i8], ptr addrspace(3) %address.local750, i64 %address.local719, !dbg !18
  %address.local752 = getelementptr [2 x i8], ptr addrspace(3) %address.local751, i64 %address.local723, !dbg !18
  %address.local753 = getelementptr i8, ptr addrspace(3) %address.local752, i64 4096, !dbg !18
  %814 = load <8 x half>, ptr addrspace(3) %address.local753, align 2, !dbg !18
  %address.local754 = add nuw nsw i64 %address.local730, 2304, !dbg !18
  %address.local755 = add nsw i64 %address.local754, %address.local708, !dbg !18
  %address.local756 = add nsw i64 %address.local755, %address.local712, !dbg !18
  %address.local757 = add nsw i64 %address.local756, %address.local716, !dbg !18
  %address.local758 = shl nuw nsw i32 %address.thread631, 2, !dbg !18
  %address.local759 = and i32 %address.local758, 192, !dbg !18
  %address.local760 = or disjoint i32 %address.local759, 288, !dbg !18
  %address.local761 = shl nuw nsw i32 %address.thread631, 3, !dbg !18
  %address.local762 = and i32 %address.local761, 96, !dbg !18
  %address.local763 = add nuw nsw i32 %address.local760, %address.local762, !dbg !18
  %address.local764 = and i32 %address.local763, 960, !dbg !18
  %address.local765 = zext nneg i32 %address.local764 to i64, !dbg !18
  %address.local766 = sub i64 %address.local757, %address.local765, !dbg !18
  %address.local767 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local766, !dbg !18
  %address.local768 = getelementptr [2 x i8], ptr addrspace(3) %address.local767, i64 %address.local723, !dbg !18
  %address.local769 = getelementptr i8, ptr addrspace(3) %address.local768, i64 576, !dbg !18
  %815 = load <8 x half>, ptr addrspace(3) %address.local769, align 2, !dbg !18
  %address.local770 = add nsw i64 %address.local713, %address.local716, !dbg !18
  %address.local771 = shl nuw nsw i64 %address.local632, 3, !dbg !18
  %address.local772 = and i64 %address.local771, 64, !dbg !18
  %address.local773 = sub nsw i64 %address.local770, %address.local772, !dbg !18
  %address.local774 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local773, !dbg !18
  %address.local775 = add nuw nsw i64 %address.local721, 18, !dbg !18
  %address.local776 = lshr i64 %address.local775, 2, !dbg !18
  %address.local777 = mul nsw i64 %address.local776, -32, !dbg !18
  %address.local778 = getelementptr [2 x i8], ptr addrspace(3) %address.local774, i64 %address.local777, !dbg !18
  %address.local779 = getelementptr [2 x i8], ptr addrspace(3) %address.local778, i64 %address.local727, !dbg !18
  %address.local780 = getelementptr i8, ptr addrspace(3) %address.local779, i64 544, !dbg !18
  %816 = load <8 x half>, ptr addrspace(3) %address.local780, align 2, !dbg !18
  %817 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %818 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %817, <4 x float> %813) #7, !dbg !18
  %819 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %820 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %821 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %822 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %823 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %address.local781 = add nsw i64 %address.local732, %address.local716, !dbg !18
  %address.local782 = shl nuw nsw i32 %address.thread631, 2, !dbg !18
  %address.local783 = and i32 %address.local782, 192, !dbg !18
  %address.local784 = add nuw nsw i32 %address.local783, 160, !dbg !18
  %address.local785 = shl nuw nsw i32 %address.thread631, 3, !dbg !18
  %address.local786 = and i32 %address.local785, 96, !dbg !18
  %address.local787 = add nuw nsw i32 %address.local784, %address.local786, !dbg !18
  %address.local788 = and i32 %address.local787, 960, !dbg !18
  %address.local789 = zext nneg i32 %address.local788 to i64, !dbg !18
  %address.local790 = sub nsw i64 %address.local781, %address.local789, !dbg !18
  %address.local791 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local790, !dbg !18
  %address.local792 = getelementptr [2 x i8], ptr addrspace(3) %address.local791, i64 %address.local777, !dbg !18
  %address.local793 = getelementptr i8, ptr addrspace(3) %address.local792, i64 1376, !dbg !18
  %824 = load <8 x half>, ptr addrspace(3) %address.local793, align 2, !dbg !18
  %825 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %819, <4 x float> %818) #7, !dbg !18
  %address.local794 = add nsw i64 %address.local730, %address.local708, !dbg !18
  %address.local795 = add nsw i64 %address.local794, %address.local712, !dbg !18
  %address.local796 = add nsw i64 %address.local795, %address.local716, !dbg !18
  %address.local797 = lshr i64 %address.local710, 3, !dbg !18
  %address.local798 = add nuw nsw i64 %address.local797, %address.local726, !dbg !18
  %address.local799 = shl nuw nsw i64 %address.local798, 6, !dbg !18
  %address.local800 = sub nsw i64 %address.local796, %address.local799, !dbg !18
  %address.local801 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local800, !dbg !18
  %address.local802 = getelementptr [2 x i8], ptr addrspace(3) %address.local801, i64 %address.local777, !dbg !18
  %address.local803 = getelementptr i8, ptr addrspace(3) %address.local802, i64 4640, !dbg !18
  %826 = load <8 x half>, ptr addrspace(3) %address.local803, align 2, !dbg !18
  %address.local804 = add nsw i64 %address.local756, %address.local716, !dbg !18
  %address.local805 = shl nuw nsw i32 %address.thread631, 2, !dbg !18
  %address.local806 = and i32 %address.local805, 192, !dbg !18
  %address.local807 = add nuw nsw i32 %address.local806, 416, !dbg !18
  %address.local808 = shl nuw nsw i32 %address.thread631, 3, !dbg !18
  %address.local809 = and i32 %address.local808, 96, !dbg !18
  %address.local810 = add nuw nsw i32 %address.local807, %address.local809, !dbg !18
  %address.local811 = and i32 %address.local810, 1984, !dbg !18
  %address.local812 = zext nneg i32 %address.local811 to i64, !dbg !18
  %address.local813 = sub i64 %address.local804, %address.local812, !dbg !18
  %address.local814 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local813, !dbg !18
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) %address.local814, i64 %address.local777, !dbg !18
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local815, i64 1376, !dbg !18
  %827 = load <8 x half>, ptr addrspace(3) %address.local816, align 2, !dbg !18
  %828 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %829 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %820, <4 x float> %825) #7, !dbg !18
  %830 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %831 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %832 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %833 = shufflevector <8 x half> %827, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %834 = shufflevector <8 x half> %827, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %835 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %821, <4 x float> %829) #7, !dbg !18
  %836 = load <8 x half>, ptr addrspace(3) %address.local728, align 2, !dbg !18
  %837 = load <8 x half>, ptr addrspace(3) %address.local745, align 2, !dbg !18
  %838 = load <8 x half>, ptr addrspace(3) %address.local753, align 2, !dbg !18
  %839 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %822, <4 x float> %805) #7, !dbg !18
  %840 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %841 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %842 = shufflevector <8 x half> %837, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %843 = shufflevector <8 x half> %837, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %844 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %845 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %846 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %823, <4 x float> %839) #7, !dbg !18
  %847 = load <8 x half>, ptr addrspace(3) %address.local769, align 2, !dbg !18
  %848 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %828, <4 x float> %846) #7, !dbg !18
  %849 = load <8 x half>, ptr addrspace(3) %address.local780, align 2, !dbg !18
  %850 = load <8 x half>, ptr addrspace(3) %address.local793, align 2, !dbg !18
  %851 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %852 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %853 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %830, <4 x float> %848) #7, !dbg !18
  %854 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %855 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %856 = shufflevector <8 x half> %850, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %857 = shufflevector <8 x half> %850, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %858 = load <8 x half>, ptr addrspace(3) %address.local803, align 2, !dbg !18
  %859 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %831, <4 x float> %853) #7, !dbg !18
  %860 = load <8 x half>, ptr addrspace(3) %address.local816, align 2, !dbg !18
  %861 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %862 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %832, <4 x float> %859) #7, !dbg !18
  %863 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %864 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %865 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %866 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %833, <4 x float> %862) #7, !dbg !18
  %867 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %834, <4 x float> %866) #7, !dbg !18
  %868 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %840, <4 x float> %810) #7, !dbg !18
  %869 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %841, <4 x float> %868) #7, !dbg !18
  %870 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %842, <4 x float> %869) #7, !dbg !18
  %871 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %843, <4 x float> %870) #7, !dbg !18
  %872 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %844, <4 x float> %871) #7, !dbg !18
  %873 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %845, <4 x float> %872) #7, !dbg !18
  %874 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %851, <4 x float> %873) #7, !dbg !18
  %875 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %852, <4 x float> %874) #7, !dbg !18
  %876 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %854, <4 x float> %812) #7, !dbg !18
  %877 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %855, <4 x float> %876) #7, !dbg !18
  %878 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %856, <4 x float> %877) #7, !dbg !18
  %879 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %857, <4 x float> %878) #7, !dbg !18
  %880 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %861, <4 x float> %879) #7, !dbg !18
  %881 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %863, <4 x float> %880) #7, !dbg !18
  %882 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %864, <4 x float> %881) #7, !dbg !18
  %883 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %865, <4 x float> %882) #7, !dbg !18
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %884 = add nsw i64 %62, %10, !dbg !9
  %885 = shl i64 %884, 19, !dbg !9
  %886 = and i64 %885, 16252928, !dbg !9
  %address.local817 = add i64 %886, %address.local636, !dbg !9
  %address.local818 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local817, !dbg !9
  %address.local819 = shl nuw nsw i32 %address.thread631, 2, !dbg !9
  %address.local820 = and i32 %address.local819, 28, !dbg !9
  %address.local821 = or disjoint i32 %address.local820, 32, !dbg !9
  %address.local822 = zext nneg i32 %address.local821 to i64, !dbg !9
  %address.local823 = getelementptr [2 x i8], ptr addrspace(1) %address.local818, i64 %address.local822, !dbg !9
  %887 = load <4 x half>, ptr addrspace(1) %address.local823, align 2, !dbg !9
  %address.local824 = add i64 %886, %address.local643, !dbg !9
  %address.local825 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local824, !dbg !9
  %address.local826 = getelementptr [2 x i8], ptr addrspace(1) %address.local825, i64 %address.local822, !dbg !9
  %888 = load <4 x half>, ptr addrspace(1) %address.local826, align 2, !dbg !9
  %address.local827 = add i64 %886, %address.local647, !dbg !9
  %address.local828 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local827, !dbg !9
  %address.local829 = getelementptr [2 x i8], ptr addrspace(1) %address.local828, i64 %address.local822, !dbg !9
  %889 = load <4 x half>, ptr addrspace(1) %address.local829, align 2, !dbg !9
  %address.local830 = add i64 %886, %address.local651, !dbg !9
  %address.local831 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local830, !dbg !9
  %address.local832 = getelementptr [2 x i8], ptr addrspace(1) %address.local831, i64 %address.local822, !dbg !9
  %890 = load <4 x half>, ptr addrspace(1) %address.local832, align 2, !dbg !9
  %891 = shufflevector <4 x half> %887, <4 x half> %888, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %892 = shufflevector <4 x half> %891, <4 x half> %889, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %893 = shufflevector <4 x half> %892, <4 x half> %890, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
  store <4 x half> %893, ptr addrspace(3) %address.local670, align 2, !dbg !9
  %894 = shufflevector <4 x half> %887, <4 x half> %888, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %895 = shufflevector <4 x half> %894, <4 x half> %889, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %896 = shufflevector <4 x half> %895, <4 x half> %890, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
  store <4 x half> %896, ptr addrspace(3) %address.local690, align 2, !dbg !9
  %897 = shufflevector <4 x half> %887, <4 x half> %888, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %898 = shufflevector <4 x half> %897, <4 x half> %889, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %899 = shufflevector <4 x half> %898, <4 x half> %890, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  store <4 x half> %899, ptr addrspace(3) %address.local699, align 2, !dbg !9
  %900 = shufflevector <4 x half> %887, <4 x half> %888, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %901 = shufflevector <4 x half> %900, <4 x half> %889, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %902 = shufflevector <4 x half> %901, <4 x half> %890, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  store <4 x half> %902, ptr addrspace(3) %address.local705, align 2, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %903 = load <8 x half>, ptr addrspace(3) %address.local728, align 2, !dbg !18
  %904 = extractvalue [2 x <8 x float>] %57, 0, !dbg !30
  %905 = shufflevector <8 x float> %904, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %906 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %907 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %906, <4 x float> %905) #7, !dbg !18
  %908 = load <8 x half>, ptr addrspace(3) %address.local745, align 2, !dbg !18
  %909 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %910 = shufflevector <8 x float> %904, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %911 = extractvalue [2 x <8 x float>] %57, 1, !dbg !30
  %912 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %909, <4 x float> %907) #7, !dbg !18
  %913 = shufflevector <8 x half> %908, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %914 = shufflevector <8 x half> %908, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %915 = shufflevector <8 x float> %911, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %916 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %913, <4 x float> %912) #7, !dbg !18
  %917 = shufflevector <8 x float> %911, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %918 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %914, <4 x float> %916) #7, !dbg !18
  %919 = load <8 x half>, ptr addrspace(3) %address.local753, align 2, !dbg !18
  %920 = load <8 x half>, ptr addrspace(3) %address.local769, align 2, !dbg !18
  %921 = load <8 x half>, ptr addrspace(3) %address.local780, align 2, !dbg !18
  %922 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %923 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %922, <4 x float> %918) #7, !dbg !18
  %924 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %925 = shufflevector <8 x half> %920, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %926 = shufflevector <8 x half> %920, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %927 = shufflevector <8 x half> %921, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %928 = shufflevector <8 x half> %921, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %929 = load <8 x half>, ptr addrspace(3) %address.local793, align 2, !dbg !18
  %930 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %924, <4 x float> %923) #7, !dbg !18
  %931 = load <8 x half>, ptr addrspace(3) %address.local803, align 2, !dbg !18
  %932 = load <8 x half>, ptr addrspace(3) %address.local816, align 2, !dbg !18
  %933 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %934 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %925, <4 x float> %930) #7, !dbg !18
  %935 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %936 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %937 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %938 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %939 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %940 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %926, <4 x float> %934) #7, !dbg !18
  %941 = load <8 x half>, ptr addrspace(3) %address.local728, align 2, !dbg !18
  %942 = load <8 x half>, ptr addrspace(3) %address.local745, align 2, !dbg !18
  %943 = load <8 x half>, ptr addrspace(3) %address.local753, align 2, !dbg !18
  %944 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %927, <4 x float> %910) #7, !dbg !18
  %945 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %946 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %947 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %948 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %949 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %950 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %951 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %928, <4 x float> %944) #7, !dbg !18
  %952 = load <8 x half>, ptr addrspace(3) %address.local769, align 2, !dbg !18
  %953 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %933, <4 x float> %951) #7, !dbg !18
  %954 = load <8 x half>, ptr addrspace(3) %address.local780, align 2, !dbg !18
  %955 = load <8 x half>, ptr addrspace(3) %address.local793, align 2, !dbg !18
  %956 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %957 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %958 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %935, <4 x float> %953) #7, !dbg !18
  %959 = shufflevector <8 x half> %954, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %960 = shufflevector <8 x half> %954, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %961 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %962 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %963 = load <8 x half>, ptr addrspace(3) %address.local803, align 2, !dbg !18
  %964 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %936, <4 x float> %958) #7, !dbg !18
  %965 = load <8 x half>, ptr addrspace(3) %address.local816, align 2, !dbg !18
  %966 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %967 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %937, <4 x float> %964) #7, !dbg !18
  %968 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %969 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %970 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %971 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %938, <4 x float> %967) #7, !dbg !18
  %972 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %939, <4 x float> %971) #7, !dbg !18
  %973 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %945, <4 x float> %915) #7, !dbg !18
  %974 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %946, <4 x float> %973) #7, !dbg !18
  %975 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %947, <4 x float> %974) #7, !dbg !18
  %976 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %948, <4 x float> %975) #7, !dbg !18
  %977 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %949, <4 x float> %976) #7, !dbg !18
  %978 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %950, <4 x float> %977) #7, !dbg !18
  %979 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %956, <4 x float> %978) #7, !dbg !18
  %980 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %957, <4 x float> %979) #7, !dbg !18
  %981 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %959, <4 x float> %917) #7, !dbg !18
  %982 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %960, <4 x float> %981) #7, !dbg !18
  %983 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %961, <4 x float> %982) #7, !dbg !18
  %984 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %962, <4 x float> %983) #7, !dbg !18
  %985 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %966, <4 x float> %984) #7, !dbg !18
  %986 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %968, <4 x float> %985) #7, !dbg !18
  %987 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %969, <4 x float> %986) #7, !dbg !18
  %988 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %970, <4 x float> %987) #7, !dbg !18
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local833 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local817, !dbg !9
  %address.local834 = shl nuw nsw i32 %address.thread631, 2, !dbg !9
  %address.local835 = and i32 %address.local834, 28, !dbg !9
  %address.local836 = or disjoint i32 %address.local835, 64, !dbg !9
  %address.local837 = zext nneg i32 %address.local836 to i64, !dbg !9
  %address.local838 = getelementptr [2 x i8], ptr addrspace(1) %address.local833, i64 %address.local837, !dbg !9
  %989 = load <4 x half>, ptr addrspace(1) %address.local838, align 2, !dbg !9
  %address.local839 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local824, !dbg !9
  %address.local840 = getelementptr [2 x i8], ptr addrspace(1) %address.local839, i64 %address.local837, !dbg !9
  %990 = load <4 x half>, ptr addrspace(1) %address.local840, align 2, !dbg !9
  %address.local841 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local827, !dbg !9
  %address.local842 = getelementptr [2 x i8], ptr addrspace(1) %address.local841, i64 %address.local837, !dbg !9
  %991 = load <4 x half>, ptr addrspace(1) %address.local842, align 2, !dbg !9
  %address.local843 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local830, !dbg !9
  %address.local844 = getelementptr [2 x i8], ptr addrspace(1) %address.local843, i64 %address.local837, !dbg !9
  %992 = load <4 x half>, ptr addrspace(1) %address.local844, align 2, !dbg !9
  %993 = shufflevector <4 x half> %989, <4 x half> %990, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %994 = shufflevector <4 x half> %993, <4 x half> %991, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %995 = shufflevector <4 x half> %994, <4 x half> %992, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
  store <4 x half> %995, ptr addrspace(3) %address.local670, align 2, !dbg !9
  %996 = shufflevector <4 x half> %989, <4 x half> %990, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %997 = shufflevector <4 x half> %996, <4 x half> %991, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %998 = shufflevector <4 x half> %997, <4 x half> %992, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
  store <4 x half> %998, ptr addrspace(3) %address.local690, align 2, !dbg !9
  %999 = shufflevector <4 x half> %989, <4 x half> %990, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %1000 = shufflevector <4 x half> %999, <4 x half> %991, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %1001 = shufflevector <4 x half> %1000, <4 x half> %992, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  store <4 x half> %1001, ptr addrspace(3) %address.local699, align 2, !dbg !9
  %1002 = shufflevector <4 x half> %989, <4 x half> %990, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %1003 = shufflevector <4 x half> %1002, <4 x half> %991, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %1004 = shufflevector <4 x half> %1003, <4 x half> %992, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  store <4 x half> %1004, ptr addrspace(3) %address.local705, align 2, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %1005 = load <8 x half>, ptr addrspace(3) %address.local728, align 2, !dbg !18
  %1006 = extractvalue [2 x <8 x float>] %58, 0, !dbg !30
  %1007 = shufflevector <8 x float> %1006, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1008 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1009 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1008, <4 x float> %1007) #7, !dbg !18
  %1010 = load <8 x half>, ptr addrspace(3) %address.local745, align 2, !dbg !18
  %1011 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1012 = shufflevector <8 x float> %1006, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1013 = extractvalue [2 x <8 x float>] %58, 1, !dbg !30
  %1014 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1011, <4 x float> %1009) #7, !dbg !18
  %1015 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1016 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1017 = shufflevector <8 x float> %1013, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1018 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1015, <4 x float> %1014) #7, !dbg !18
  %1019 = shufflevector <8 x float> %1013, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1020 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1016, <4 x float> %1018) #7, !dbg !18
  %1021 = load <8 x half>, ptr addrspace(3) %address.local753, align 2, !dbg !18
  %1022 = load <8 x half>, ptr addrspace(3) %address.local769, align 2, !dbg !18
  %1023 = load <8 x half>, ptr addrspace(3) %address.local780, align 2, !dbg !18
  %1024 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1025 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1024, <4 x float> %1020) #7, !dbg !18
  %1026 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1027 = shufflevector <8 x half> %1022, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1028 = shufflevector <8 x half> %1022, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1029 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1030 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1031 = load <8 x half>, ptr addrspace(3) %address.local793, align 2, !dbg !18
  %1032 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1026, <4 x float> %1025) #7, !dbg !18
  %1033 = load <8 x half>, ptr addrspace(3) %address.local803, align 2, !dbg !18
  %1034 = load <8 x half>, ptr addrspace(3) %address.local816, align 2, !dbg !18
  %1035 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1036 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1027, <4 x float> %1032) #7, !dbg !18
  %1037 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1038 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1039 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1040 = shufflevector <8 x half> %1034, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1041 = shufflevector <8 x half> %1034, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1042 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1028, <4 x float> %1036) #7, !dbg !18
  %1043 = load <8 x half>, ptr addrspace(3) %address.local728, align 2, !dbg !18
  %1044 = load <8 x half>, ptr addrspace(3) %address.local745, align 2, !dbg !18
  %1045 = load <8 x half>, ptr addrspace(3) %address.local753, align 2, !dbg !18
  %1046 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1029, <4 x float> %1012) #7, !dbg !18
  %1047 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1048 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1049 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1050 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1051 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1052 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1053 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1030, <4 x float> %1046) #7, !dbg !18
  %1054 = load <8 x half>, ptr addrspace(3) %address.local769, align 2, !dbg !18
  %1055 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1035, <4 x float> %1053) #7, !dbg !18
  %1056 = load <8 x half>, ptr addrspace(3) %address.local780, align 2, !dbg !18
  %1057 = load <8 x half>, ptr addrspace(3) %address.local793, align 2, !dbg !18
  %1058 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1059 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1060 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1037, <4 x float> %1055) #7, !dbg !18
  %1061 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1062 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1063 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1064 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1065 = load <8 x half>, ptr addrspace(3) %address.local803, align 2, !dbg !18
  %1066 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1038, <4 x float> %1060) #7, !dbg !18
  %1067 = load <8 x half>, ptr addrspace(3) %address.local816, align 2, !dbg !18
  %1068 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1069 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1039, <4 x float> %1066) #7, !dbg !18
  %1070 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1071 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1072 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1073 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1040, <4 x float> %1069) #7, !dbg !18
  %1074 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1041, <4 x float> %1073) #7, !dbg !18
  %1075 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1047, <4 x float> %1017) #7, !dbg !18
  %1076 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1048, <4 x float> %1075) #7, !dbg !18
  %1077 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1049, <4 x float> %1076) #7, !dbg !18
  %1078 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1050, <4 x float> %1077) #7, !dbg !18
  %1079 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1051, <4 x float> %1078) #7, !dbg !18
  %1080 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1052, <4 x float> %1079) #7, !dbg !18
  %1081 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1058, <4 x float> %1080) #7, !dbg !18
  %1082 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1059, <4 x float> %1081) #7, !dbg !18
  %1083 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1061, <4 x float> %1019) #7, !dbg !18
  %1084 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1062, <4 x float> %1083) #7, !dbg !18
  %1085 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1063, <4 x float> %1084) #7, !dbg !18
  %1086 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1064, <4 x float> %1085) #7, !dbg !18
  %1087 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1068, <4 x float> %1086) #7, !dbg !18
  %1088 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1070, <4 x float> %1087) #7, !dbg !18
  %1089 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1071, <4 x float> %1088) #7, !dbg !18
  %1090 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1072, <4 x float> %1089) #7, !dbg !18
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local845 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local817, !dbg !9
  %address.local846 = shl nuw nsw i32 %address.thread631, 2, !dbg !9
  %address.local847 = and i32 %address.local846, 28, !dbg !9
  %address.local848 = or disjoint i32 %address.local847, 96, !dbg !9
  %address.local849 = zext nneg i32 %address.local848 to i64, !dbg !9
  %address.local850 = getelementptr [2 x i8], ptr addrspace(1) %address.local845, i64 %address.local849, !dbg !9
  %1091 = load <4 x half>, ptr addrspace(1) %address.local850, align 2, !dbg !9
  %address.local851 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local824, !dbg !9
  %address.local852 = getelementptr [2 x i8], ptr addrspace(1) %address.local851, i64 %address.local849, !dbg !9
  %1092 = load <4 x half>, ptr addrspace(1) %address.local852, align 2, !dbg !9
  %address.local853 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local827, !dbg !9
  %address.local854 = getelementptr [2 x i8], ptr addrspace(1) %address.local853, i64 %address.local849, !dbg !9
  %1093 = load <4 x half>, ptr addrspace(1) %address.local854, align 2, !dbg !9
  %address.local855 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local830, !dbg !9
  %address.local856 = getelementptr [2 x i8], ptr addrspace(1) %address.local855, i64 %address.local849, !dbg !9
  %1094 = load <4 x half>, ptr addrspace(1) %address.local856, align 2, !dbg !9
  %1095 = shufflevector <4 x half> %1091, <4 x half> %1092, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %1096 = shufflevector <4 x half> %1095, <4 x half> %1093, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %1097 = shufflevector <4 x half> %1096, <4 x half> %1094, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
  store <4 x half> %1097, ptr addrspace(3) %address.local670, align 2, !dbg !9
  %1098 = shufflevector <4 x half> %1091, <4 x half> %1092, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %1099 = shufflevector <4 x half> %1098, <4 x half> %1093, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %1100 = shufflevector <4 x half> %1099, <4 x half> %1094, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
  store <4 x half> %1100, ptr addrspace(3) %address.local690, align 2, !dbg !9
  %1101 = shufflevector <4 x half> %1091, <4 x half> %1092, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %1102 = shufflevector <4 x half> %1101, <4 x half> %1093, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %1103 = shufflevector <4 x half> %1102, <4 x half> %1094, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  store <4 x half> %1103, ptr addrspace(3) %address.local699, align 2, !dbg !9
  %1104 = shufflevector <4 x half> %1091, <4 x half> %1092, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %1105 = shufflevector <4 x half> %1104, <4 x half> %1093, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %1106 = shufflevector <4 x half> %1105, <4 x half> %1094, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  store <4 x half> %1106, ptr addrspace(3) %address.local705, align 2, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %1107 = load <8 x half>, ptr addrspace(3) %address.local728, align 2, !dbg !18
  %1108 = extractvalue [2 x <8 x float>] %59, 0, !dbg !30
  %1109 = shufflevector <8 x float> %1108, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1110 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1111 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1110, <4 x float> %1109) #7, !dbg !18
  %1112 = load <8 x half>, ptr addrspace(3) %address.local745, align 2, !dbg !18
  %1113 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1114 = shufflevector <8 x float> %1108, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1115 = extractvalue [2 x <8 x float>] %59, 1, !dbg !30
  %1116 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1113, <4 x float> %1111) #7, !dbg !18
  %1117 = shufflevector <8 x half> %1112, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1118 = shufflevector <8 x half> %1112, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1119 = shufflevector <8 x float> %1115, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1120 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1117, <4 x float> %1116) #7, !dbg !18
  %1121 = shufflevector <8 x float> %1115, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1122 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1118, <4 x float> %1120) #7, !dbg !18
  %1123 = load <8 x half>, ptr addrspace(3) %address.local753, align 2, !dbg !18
  %1124 = load <8 x half>, ptr addrspace(3) %address.local769, align 2, !dbg !18
  %1125 = load <8 x half>, ptr addrspace(3) %address.local780, align 2, !dbg !18
  %1126 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1127 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1126, <4 x float> %1122) #7, !dbg !18
  %1128 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1129 = shufflevector <8 x half> %1124, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1130 = shufflevector <8 x half> %1124, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1131 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1132 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1133 = load <8 x half>, ptr addrspace(3) %address.local793, align 2, !dbg !18
  %1134 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1128, <4 x float> %1127) #7, !dbg !18
  %1135 = load <8 x half>, ptr addrspace(3) %address.local803, align 2, !dbg !18
  %1136 = load <8 x half>, ptr addrspace(3) %address.local816, align 2, !dbg !18
  %1137 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1138 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1129, <4 x float> %1134) #7, !dbg !18
  %1139 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1140 = shufflevector <8 x half> %1135, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1141 = shufflevector <8 x half> %1135, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1142 = shufflevector <8 x half> %1136, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1143 = shufflevector <8 x half> %1136, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1144 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1130, <4 x float> %1138) #7, !dbg !18
  %1145 = load <8 x half>, ptr addrspace(3) %address.local728, align 2, !dbg !18
  %1146 = load <8 x half>, ptr addrspace(3) %address.local745, align 2, !dbg !18
  %1147 = load <8 x half>, ptr addrspace(3) %address.local753, align 2, !dbg !18
  %1148 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1131, <4 x float> %1114) #7, !dbg !18
  %1149 = shufflevector <8 x half> %1145, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1150 = shufflevector <8 x half> %1145, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1151 = shufflevector <8 x half> %1146, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1152 = shufflevector <8 x half> %1146, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1153 = shufflevector <8 x half> %1147, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1154 = shufflevector <8 x half> %1147, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1155 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1132, <4 x float> %1148) #7, !dbg !18
  %1156 = load <8 x half>, ptr addrspace(3) %address.local769, align 2, !dbg !18
  %1157 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1137, <4 x float> %1155) #7, !dbg !18
  %1158 = load <8 x half>, ptr addrspace(3) %address.local780, align 2, !dbg !18
  %1159 = load <8 x half>, ptr addrspace(3) %address.local793, align 2, !dbg !18
  %1160 = shufflevector <8 x half> %1156, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1161 = shufflevector <8 x half> %1156, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1162 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1139, <4 x float> %1157) #7, !dbg !18
  %1163 = shufflevector <8 x half> %1158, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1164 = shufflevector <8 x half> %1158, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1165 = shufflevector <8 x half> %1159, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1166 = shufflevector <8 x half> %1159, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1167 = load <8 x half>, ptr addrspace(3) %address.local803, align 2, !dbg !18
  %1168 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1140, <4 x float> %1162) #7, !dbg !18
  %1169 = load <8 x half>, ptr addrspace(3) %address.local816, align 2, !dbg !18
  %1170 = shufflevector <4 x float> %835, <4 x float> %867, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1171 = insertvalue [2 x <8 x float>] poison, <8 x float> %1170, 0, !dbg !17
  %1172 = shufflevector <4 x float> %875, <4 x float> %883, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1173 = insertvalue [2 x <8 x float>] %1171, <8 x float> %1172, 1, !dbg !17
  %1174 = shufflevector <8 x half> %1167, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1175 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1141, <4 x float> %1168) #7, !dbg !18
  %1176 = shufflevector <8 x half> %1167, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1177 = shufflevector <4 x float> %940, <4 x float> %972, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1178 = insertvalue [2 x <8 x float>] poison, <8 x float> %1177, 0, !dbg !17
  %1179 = shufflevector <4 x float> %980, <4 x float> %988, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1180 = insertvalue [2 x <8 x float>] %1178, <8 x float> %1179, 1, !dbg !17
  %1181 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1182 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1183 = shufflevector <4 x float> %1042, <4 x float> %1074, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1184 = insertvalue [2 x <8 x float>] poison, <8 x float> %1183, 0, !dbg !17
  %1185 = shufflevector <4 x float> %1082, <4 x float> %1090, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1186 = insertvalue [2 x <8 x float>] %1184, <8 x float> %1185, 1, !dbg !17
  %1187 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1142, <4 x float> %1175) #7, !dbg !18
  %1188 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1143, <4 x float> %1187) #7, !dbg !18
  %1189 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1149, <4 x float> %1119) #7, !dbg !18
  %1190 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1150, <4 x float> %1189) #7, !dbg !18
  %1191 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1151, <4 x float> %1190) #7, !dbg !18
  %1192 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1152, <4 x float> %1191) #7, !dbg !18
  %1193 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1153, <4 x float> %1192) #7, !dbg !18
  %1194 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1154, <4 x float> %1193) #7, !dbg !18
  %1195 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1160, <4 x float> %1194) #7, !dbg !18
  %1196 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1161, <4 x float> %1195) #7, !dbg !18
  %1197 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1163, <4 x float> %1121) #7, !dbg !18
  %1198 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1164, <4 x float> %1197) #7, !dbg !18
  %1199 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1165, <4 x float> %1198) #7, !dbg !18
  %1200 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1166, <4 x float> %1199) #7, !dbg !18
  %1201 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1174, <4 x float> %1200) #7, !dbg !18
  %1202 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1176, <4 x float> %1201) #7, !dbg !18
  %1203 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1181, <4 x float> %1202) #7, !dbg !18
  %1204 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1182, <4 x float> %1203) #7, !dbg !18
  %1205 = shufflevector <4 x float> %1144, <4 x float> %1188, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1206 = insertvalue [2 x <8 x float>] poison, <8 x float> %1205, 0, !dbg !17
  %1207 = shufflevector <4 x float> %1196, <4 x float> %1204, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1208 = insertvalue [2 x <8 x float>] %1206, <8 x float> %1207, 1, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1209 = add nuw nsw i64 %54, 1, !dbg !17
  br label %53, !dbg !17

1210:                                             ; preds = %53
  %1211 = and i32 %12, 16, !dbg !17
  %.not41 = icmp eq i32 %1211, 0, !dbg !17
  %1212 = and i32 %12, 32, !dbg !17
  %.not42 = icmp eq i32 %1212, 0, !dbg !17
  %1213 = extractvalue [2 x <8 x float>] %59, 0, !dbg !17
  %1214 = extractelement <8 x float> %1213, i64 0, !dbg !17
  %1215 = extractelement <8 x float> %1213, i64 1, !dbg !17
  %1216 = extractelement <8 x float> %1213, i64 2, !dbg !17
  %1217 = extractelement <8 x float> %1213, i64 3, !dbg !17
  %1218 = select i1 %.not41, float %1215, float %1214, !dbg !17
  %1219 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !17
  %1220 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1219), !dbg !17
  %1221 = add nuw nsw i32 %1220, 64, !dbg !17
  %1222 = xor i32 %1220, 16, !dbg !17
  %1223 = and i32 %1221, 192, !dbg !17
  %1224 = icmp samesign ult i32 %1222, %1223, !dbg !17
  %1225 = select i1 %1224, i32 %1222, i32 %1220, !dbg !17
  %1226 = shl nuw nsw i32 %1225, 2, !dbg !17
  %1227 = bitcast float %1218 to i32, !dbg !17
  %1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1227), !dbg !17
  %1229 = bitcast i32 %1228 to float, !dbg !17
  %1230 = select i1 %.not41, float %1214, float %1229, !dbg !17
  %1231 = select i1 %.not41, float %1229, float %1215, !dbg !17
  %1232 = select i1 %.not41, float %1217, float %1216, !dbg !17
  %1233 = bitcast float %1232 to i32, !dbg !17
  %1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1233), !dbg !17
  %1235 = bitcast i32 %1234 to float, !dbg !17
  %1236 = select i1 %.not41, float %1216, float %1235, !dbg !17
  %1237 = select i1 %.not41, float %1235, float %1217, !dbg !17
  %1238 = select i1 %.not42, float %1236, float %1230, !dbg !17
  %1239 = xor i32 %1220, 32, !dbg !17
  %1240 = icmp samesign ult i32 %1239, %1223, !dbg !17
  %1241 = select i1 %1240, i32 %1239, i32 %1220, !dbg !17
  %1242 = shl nuw nsw i32 %1241, 2, !dbg !17
  %1243 = bitcast float %1238 to i32, !dbg !17
  %1244 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1243), !dbg !17
  %1245 = bitcast i32 %1244 to float, !dbg !17
  %1246 = select i1 %.not42, float %1230, float %1245, !dbg !17
  %1247 = select i1 %.not42, float %1245, float %1236, !dbg !17
  %1248 = select i1 %.not42, float %1237, float %1231, !dbg !17
  %1249 = bitcast float %1248 to i32, !dbg !17
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1249), !dbg !17
  %1251 = bitcast i32 %1250 to float, !dbg !17
  %1252 = select i1 %.not42, float %1231, float %1251, !dbg !17
  %1253 = select i1 %.not42, float %1251, float %1237, !dbg !17
  %1254 = extractvalue [2 x <8 x float>] %58, 0, !dbg !17
  %1255 = extractelement <8 x float> %1254, i64 0, !dbg !17
  %1256 = extractelement <8 x float> %1254, i64 1, !dbg !17
  %1257 = extractelement <8 x float> %1254, i64 2, !dbg !17
  %1258 = extractelement <8 x float> %1254, i64 3, !dbg !17
  %1259 = select i1 %.not41, float %1256, float %1255, !dbg !17
  %1260 = bitcast float %1259 to i32, !dbg !17
  %1261 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1260), !dbg !17
  %1262 = bitcast i32 %1261 to float, !dbg !17
  %1263 = select i1 %.not41, float %1255, float %1262, !dbg !17
  %1264 = select i1 %.not41, float %1262, float %1256, !dbg !17
  %1265 = select i1 %.not41, float %1258, float %1257, !dbg !17
  %1266 = bitcast float %1265 to i32, !dbg !17
  %1267 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1266), !dbg !17
  %1268 = bitcast i32 %1267 to float, !dbg !17
  %1269 = select i1 %.not41, float %1257, float %1268, !dbg !17
  %1270 = select i1 %.not41, float %1268, float %1258, !dbg !17
  %1271 = select i1 %.not42, float %1269, float %1263, !dbg !17
  %1272 = bitcast float %1271 to i32, !dbg !17
  %1273 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1272), !dbg !17
  %1274 = bitcast i32 %1273 to float, !dbg !17
  %1275 = select i1 %.not42, float %1263, float %1274, !dbg !17
  %1276 = select i1 %.not42, float %1274, float %1269, !dbg !17
  %1277 = select i1 %.not42, float %1270, float %1264, !dbg !17
  %1278 = bitcast float %1277 to i32, !dbg !17
  %1279 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1278), !dbg !17
  %1280 = bitcast i32 %1279 to float, !dbg !17
  %1281 = select i1 %.not42, float %1264, float %1280, !dbg !17
  %1282 = select i1 %.not42, float %1280, float %1270, !dbg !17
  %1283 = extractvalue [2 x <8 x float>] %57, 0, !dbg !17
  %1284 = extractelement <8 x float> %1283, i64 0, !dbg !17
  %1285 = extractelement <8 x float> %1283, i64 1, !dbg !17
  %1286 = extractelement <8 x float> %1283, i64 2, !dbg !17
  %1287 = extractelement <8 x float> %1283, i64 3, !dbg !17
  %1288 = select i1 %.not41, float %1285, float %1284, !dbg !17
  %1289 = bitcast float %1288 to i32, !dbg !17
  %1290 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1289), !dbg !17
  %1291 = bitcast i32 %1290 to float, !dbg !17
  %1292 = select i1 %.not41, float %1284, float %1291, !dbg !17
  %1293 = select i1 %.not41, float %1291, float %1285, !dbg !17
  %1294 = select i1 %.not41, float %1287, float %1286, !dbg !17
  %1295 = bitcast float %1294 to i32, !dbg !17
  %1296 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1295), !dbg !17
  %1297 = bitcast i32 %1296 to float, !dbg !17
  %1298 = select i1 %.not41, float %1286, float %1297, !dbg !17
  %1299 = select i1 %.not41, float %1297, float %1287, !dbg !17
  %1300 = select i1 %.not42, float %1298, float %1292, !dbg !17
  %1301 = bitcast float %1300 to i32, !dbg !17
  %1302 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1301), !dbg !17
  %1303 = bitcast i32 %1302 to float, !dbg !17
  %1304 = select i1 %.not42, float %1292, float %1303, !dbg !17
  %1305 = select i1 %.not42, float %1303, float %1298, !dbg !17
  %1306 = select i1 %.not42, float %1299, float %1293, !dbg !17
  %1307 = bitcast float %1306 to i32, !dbg !17
  %1308 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1307), !dbg !17
  %1309 = bitcast i32 %1308 to float, !dbg !17
  %1310 = select i1 %.not42, float %1293, float %1309, !dbg !17
  %1311 = select i1 %.not42, float %1309, float %1299, !dbg !17
  %1312 = extractvalue [2 x <8 x float>] %55, 0, !dbg !17
  %1313 = extractelement <8 x float> %1312, i64 0, !dbg !17
  %1314 = extractelement <8 x float> %1312, i64 1, !dbg !17
  %1315 = extractelement <8 x float> %1312, i64 2, !dbg !17
  %1316 = extractelement <8 x float> %1312, i64 3, !dbg !17
  %1317 = select i1 %.not41, float %1314, float %1313, !dbg !17
  %1318 = bitcast float %1317 to i32, !dbg !17
  %1319 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1318), !dbg !17
  %1320 = bitcast i32 %1319 to float, !dbg !17
  %1321 = select i1 %.not41, float %1313, float %1320, !dbg !17
  %1322 = select i1 %.not41, float %1320, float %1314, !dbg !17
  %1323 = select i1 %.not41, float %1316, float %1315, !dbg !17
  %1324 = bitcast float %1323 to i32, !dbg !17
  %1325 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1324), !dbg !17
  %1326 = bitcast i32 %1325 to float, !dbg !17
  %1327 = select i1 %.not41, float %1315, float %1326, !dbg !17
  %1328 = select i1 %.not41, float %1326, float %1316, !dbg !17
  %1329 = select i1 %.not42, float %1327, float %1321, !dbg !17
  %1330 = bitcast float %1329 to i32, !dbg !17
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1330), !dbg !17
  %1332 = select i1 %.not42, float %1328, float %1322, !dbg !17
  %1333 = bitcast float %1332 to i32, !dbg !17
  %1334 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1333), !dbg !17
  %1335 = extractvalue [2 x <1 x float>] %56, 0, !dbg !19
  %1336 = shufflevector <1 x float> %1335, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1337 = extractelement <8 x float> %1213, i64 4, !dbg !17
  %1338 = extractelement <8 x float> %1213, i64 5, !dbg !17
  %1339 = extractelement <8 x float> %1213, i64 6, !dbg !17
  %1340 = extractelement <8 x float> %1213, i64 7, !dbg !17
  %1341 = select i1 %.not41, float %1338, float %1337, !dbg !17
  %1342 = bitcast float %1341 to i32, !dbg !17
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1342), !dbg !17
  %1344 = bitcast i32 %1343 to float, !dbg !17
  %1345 = select i1 %.not41, float %1337, float %1344, !dbg !17
  %1346 = select i1 %.not41, float %1344, float %1338, !dbg !17
  %1347 = select i1 %.not41, float %1340, float %1339, !dbg !17
  %1348 = bitcast float %1347 to i32, !dbg !17
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1348), !dbg !17
  %1350 = bitcast i32 %1349 to float, !dbg !17
  %1351 = select i1 %.not41, float %1339, float %1350, !dbg !17
  %1352 = select i1 %.not41, float %1350, float %1340, !dbg !17
  %1353 = select i1 %.not42, float %1351, float %1345, !dbg !17
  %1354 = bitcast float %1353 to i32, !dbg !17
  %1355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1354), !dbg !17
  %1356 = bitcast i32 %1355 to float, !dbg !17
  %1357 = select i1 %.not42, float %1345, float %1356, !dbg !17
  %1358 = select i1 %.not42, float %1356, float %1351, !dbg !17
  %1359 = select i1 %.not42, float %1352, float %1346, !dbg !17
  %1360 = bitcast float %1359 to i32, !dbg !17
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1360), !dbg !17
  %1362 = bitcast i32 %1361 to float, !dbg !17
  %1363 = select i1 %.not42, float %1346, float %1362, !dbg !17
  %1364 = select i1 %.not42, float %1362, float %1352, !dbg !17
  %1365 = extractelement <8 x float> %1254, i64 4, !dbg !17
  %1366 = extractelement <8 x float> %1254, i64 5, !dbg !17
  %1367 = extractelement <8 x float> %1254, i64 6, !dbg !17
  %1368 = extractelement <8 x float> %1254, i64 7, !dbg !17
  %1369 = select i1 %.not41, float %1366, float %1365, !dbg !17
  %1370 = bitcast float %1369 to i32, !dbg !17
  %1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1370), !dbg !17
  %1372 = bitcast i32 %1371 to float, !dbg !17
  %1373 = select i1 %.not41, float %1365, float %1372, !dbg !17
  %1374 = select i1 %.not41, float %1372, float %1366, !dbg !17
  %1375 = select i1 %.not41, float %1368, float %1367, !dbg !17
  %1376 = bitcast float %1375 to i32, !dbg !17
  %1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1376), !dbg !17
  %1378 = bitcast i32 %1377 to float, !dbg !17
  %1379 = select i1 %.not41, float %1367, float %1378, !dbg !17
  %1380 = select i1 %.not41, float %1378, float %1368, !dbg !17
  %1381 = select i1 %.not42, float %1379, float %1373, !dbg !17
  %1382 = bitcast float %1381 to i32, !dbg !17
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1382), !dbg !17
  %1384 = bitcast i32 %1383 to float, !dbg !17
  %1385 = select i1 %.not42, float %1373, float %1384, !dbg !17
  %1386 = select i1 %.not42, float %1384, float %1379, !dbg !17
  %1387 = select i1 %.not42, float %1380, float %1374, !dbg !17
  %1388 = bitcast float %1387 to i32, !dbg !17
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1388), !dbg !17
  %1390 = bitcast i32 %1389 to float, !dbg !17
  %1391 = select i1 %.not42, float %1374, float %1390, !dbg !17
  %1392 = select i1 %.not42, float %1390, float %1380, !dbg !17
  %1393 = extractelement <8 x float> %1283, i64 4, !dbg !17
  %1394 = extractelement <8 x float> %1283, i64 5, !dbg !17
  %1395 = extractelement <8 x float> %1283, i64 6, !dbg !17
  %1396 = extractelement <8 x float> %1283, i64 7, !dbg !17
  %1397 = select i1 %.not41, float %1394, float %1393, !dbg !17
  %1398 = bitcast float %1397 to i32, !dbg !17
  %1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1398), !dbg !17
  %1400 = bitcast i32 %1399 to float, !dbg !17
  %1401 = select i1 %.not41, float %1393, float %1400, !dbg !17
  %1402 = select i1 %.not41, float %1400, float %1394, !dbg !17
  %1403 = select i1 %.not41, float %1396, float %1395, !dbg !17
  %1404 = bitcast float %1403 to i32, !dbg !17
  %1405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1404), !dbg !17
  %1406 = bitcast i32 %1405 to float, !dbg !17
  %1407 = select i1 %.not41, float %1395, float %1406, !dbg !17
  %1408 = select i1 %.not41, float %1406, float %1396, !dbg !17
  %1409 = select i1 %.not42, float %1407, float %1401, !dbg !17
  %1410 = bitcast float %1409 to i32, !dbg !17
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1410), !dbg !17
  %1412 = bitcast i32 %1411 to float, !dbg !17
  %1413 = select i1 %.not42, float %1401, float %1412, !dbg !17
  %1414 = select i1 %.not42, float %1412, float %1407, !dbg !17
  %1415 = select i1 %.not42, float %1408, float %1402, !dbg !17
  %1416 = bitcast float %1415 to i32, !dbg !17
  %1417 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1416), !dbg !17
  %1418 = bitcast i32 %1417 to float, !dbg !17
  %1419 = select i1 %.not42, float %1402, float %1418, !dbg !17
  %1420 = select i1 %.not42, float %1418, float %1408, !dbg !17
  %1421 = extractelement <8 x float> %1312, i64 4, !dbg !17
  %1422 = extractelement <8 x float> %1312, i64 5, !dbg !17
  %1423 = extractelement <8 x float> %1312, i64 6, !dbg !17
  %1424 = extractelement <8 x float> %1312, i64 7, !dbg !17
  %1425 = select i1 %.not41, float %1422, float %1421, !dbg !17
  %1426 = bitcast float %1425 to i32, !dbg !17
  %1427 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1426), !dbg !17
  %1428 = bitcast i32 %1427 to float, !dbg !17
  %1429 = select i1 %.not41, float %1421, float %1428, !dbg !17
  %1430 = select i1 %.not41, float %1428, float %1422, !dbg !17
  %1431 = select i1 %.not41, float %1424, float %1423, !dbg !17
  %1432 = bitcast float %1431 to i32, !dbg !17
  %1433 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1432), !dbg !17
  %1434 = bitcast i32 %1433 to float, !dbg !17
  %1435 = select i1 %.not41, float %1423, float %1434, !dbg !17
  %1436 = select i1 %.not41, float %1434, float %1424, !dbg !17
  %1437 = select i1 %.not42, float %1435, float %1429, !dbg !17
  %1438 = bitcast float %1437 to i32, !dbg !17
  %1439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1438), !dbg !17
  %1440 = select i1 %.not42, float %1436, float %1430, !dbg !17
  %1441 = bitcast float %1440 to i32, !dbg !17
  %1442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1441), !dbg !17
  %1443 = extractvalue [2 x <8 x float>] %59, 1, !dbg !17
  %1444 = extractelement <8 x float> %1443, i64 0, !dbg !17
  %1445 = extractelement <8 x float> %1443, i64 1, !dbg !17
  %1446 = extractelement <8 x float> %1443, i64 2, !dbg !17
  %1447 = extractelement <8 x float> %1443, i64 3, !dbg !17
  %1448 = select i1 %.not41, float %1445, float %1444, !dbg !17
  %1449 = bitcast float %1448 to i32, !dbg !17
  %1450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1449), !dbg !17
  %1451 = bitcast i32 %1450 to float, !dbg !17
  %1452 = select i1 %.not41, float %1444, float %1451, !dbg !17
  %1453 = select i1 %.not41, float %1451, float %1445, !dbg !17
  %1454 = select i1 %.not41, float %1447, float %1446, !dbg !17
  %1455 = bitcast float %1454 to i32, !dbg !17
  %1456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1455), !dbg !17
  %1457 = bitcast i32 %1456 to float, !dbg !17
  %1458 = select i1 %.not41, float %1446, float %1457, !dbg !17
  %1459 = select i1 %.not41, float %1457, float %1447, !dbg !17
  %1460 = select i1 %.not42, float %1458, float %1452, !dbg !17
  %1461 = bitcast float %1460 to i32, !dbg !17
  %1462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1461), !dbg !17
  %1463 = bitcast i32 %1462 to float, !dbg !17
  %1464 = select i1 %.not42, float %1452, float %1463, !dbg !17
  %1465 = select i1 %.not42, float %1463, float %1458, !dbg !17
  %1466 = select i1 %.not42, float %1459, float %1453, !dbg !17
  %1467 = bitcast float %1466 to i32, !dbg !17
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1467), !dbg !17
  %1469 = bitcast i32 %1468 to float, !dbg !17
  %1470 = select i1 %.not42, float %1453, float %1469, !dbg !17
  %1471 = select i1 %.not42, float %1469, float %1459, !dbg !17
  %1472 = extractvalue [2 x <8 x float>] %58, 1, !dbg !17
  %1473 = extractelement <8 x float> %1472, i64 0, !dbg !17
  %1474 = extractelement <8 x float> %1472, i64 1, !dbg !17
  %1475 = extractelement <8 x float> %1472, i64 2, !dbg !17
  %1476 = extractelement <8 x float> %1472, i64 3, !dbg !17
  %1477 = select i1 %.not41, float %1474, float %1473, !dbg !17
  %1478 = bitcast float %1477 to i32, !dbg !17
  %1479 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1478), !dbg !17
  %1480 = bitcast i32 %1479 to float, !dbg !17
  %1481 = select i1 %.not41, float %1473, float %1480, !dbg !17
  %1482 = select i1 %.not41, float %1480, float %1474, !dbg !17
  %1483 = select i1 %.not41, float %1476, float %1475, !dbg !17
  %1484 = bitcast float %1483 to i32, !dbg !17
  %1485 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1484), !dbg !17
  %1486 = bitcast i32 %1485 to float, !dbg !17
  %1487 = select i1 %.not41, float %1475, float %1486, !dbg !17
  %1488 = select i1 %.not41, float %1486, float %1476, !dbg !17
  %1489 = select i1 %.not42, float %1487, float %1481, !dbg !17
  %1490 = bitcast float %1489 to i32, !dbg !17
  %1491 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1490), !dbg !17
  %1492 = bitcast i32 %1491 to float, !dbg !17
  %1493 = select i1 %.not42, float %1481, float %1492, !dbg !17
  %1494 = select i1 %.not42, float %1492, float %1487, !dbg !17
  %1495 = select i1 %.not42, float %1488, float %1482, !dbg !17
  %1496 = bitcast float %1495 to i32, !dbg !17
  %1497 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1496), !dbg !17
  %1498 = bitcast i32 %1497 to float, !dbg !17
  %1499 = select i1 %.not42, float %1482, float %1498, !dbg !17
  %1500 = select i1 %.not42, float %1498, float %1488, !dbg !17
  %1501 = extractvalue [2 x <8 x float>] %57, 1, !dbg !17
  %1502 = extractelement <8 x float> %1501, i64 0, !dbg !17
  %1503 = extractelement <8 x float> %1501, i64 1, !dbg !17
  %1504 = extractelement <8 x float> %1501, i64 2, !dbg !17
  %1505 = extractelement <8 x float> %1501, i64 3, !dbg !17
  %1506 = select i1 %.not41, float %1503, float %1502, !dbg !17
  %1507 = bitcast float %1506 to i32, !dbg !17
  %1508 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1507), !dbg !17
  %1509 = bitcast i32 %1508 to float, !dbg !17
  %1510 = select i1 %.not41, float %1502, float %1509, !dbg !17
  %1511 = select i1 %.not41, float %1509, float %1503, !dbg !17
  %1512 = select i1 %.not41, float %1505, float %1504, !dbg !17
  %1513 = bitcast float %1512 to i32, !dbg !17
  %1514 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1513), !dbg !17
  %1515 = bitcast i32 %1514 to float, !dbg !17
  %1516 = select i1 %.not41, float %1504, float %1515, !dbg !17
  %1517 = select i1 %.not41, float %1515, float %1505, !dbg !17
  %1518 = select i1 %.not42, float %1516, float %1510, !dbg !17
  %1519 = bitcast float %1518 to i32, !dbg !17
  %1520 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1519), !dbg !17
  %1521 = bitcast i32 %1520 to float, !dbg !17
  %1522 = select i1 %.not42, float %1510, float %1521, !dbg !17
  %1523 = select i1 %.not42, float %1521, float %1516, !dbg !17
  %1524 = select i1 %.not42, float %1517, float %1511, !dbg !17
  %1525 = bitcast float %1524 to i32, !dbg !17
  %1526 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1525), !dbg !17
  %1527 = bitcast i32 %1526 to float, !dbg !17
  %1528 = select i1 %.not42, float %1511, float %1527, !dbg !17
  %1529 = select i1 %.not42, float %1527, float %1517, !dbg !17
  %1530 = extractvalue [2 x <8 x float>] %55, 1, !dbg !17
  %1531 = extractelement <8 x float> %1530, i64 0, !dbg !17
  %1532 = extractelement <8 x float> %1530, i64 1, !dbg !17
  %1533 = extractelement <8 x float> %1530, i64 2, !dbg !17
  %1534 = extractelement <8 x float> %1530, i64 3, !dbg !17
  %1535 = select i1 %.not41, float %1532, float %1531, !dbg !17
  %1536 = bitcast float %1535 to i32, !dbg !17
  %1537 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1536), !dbg !17
  %1538 = bitcast i32 %1537 to float, !dbg !17
  %1539 = select i1 %.not41, float %1531, float %1538, !dbg !17
  %1540 = select i1 %.not41, float %1538, float %1532, !dbg !17
  %1541 = select i1 %.not41, float %1534, float %1533, !dbg !17
  %1542 = bitcast float %1541 to i32, !dbg !17
  %1543 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1542), !dbg !17
  %1544 = bitcast i32 %1543 to float, !dbg !17
  %1545 = select i1 %.not41, float %1533, float %1544, !dbg !17
  %1546 = select i1 %.not41, float %1544, float %1534, !dbg !17
  %1547 = select i1 %.not42, float %1545, float %1539, !dbg !17
  %1548 = bitcast float %1547 to i32, !dbg !17
  %1549 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1548), !dbg !17
  %1550 = select i1 %.not42, float %1546, float %1540, !dbg !17
  %1551 = bitcast float %1550 to i32, !dbg !17
  %1552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1551), !dbg !17
  %1553 = extractvalue [2 x <1 x float>] %56, 1, !dbg !19
  %1554 = shufflevector <1 x float> %1553, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1555 = extractelement <8 x float> %1443, i64 4, !dbg !17
  %1556 = extractelement <8 x float> %1443, i64 5, !dbg !17
  %1557 = extractelement <8 x float> %1443, i64 6, !dbg !17
  %1558 = extractelement <8 x float> %1443, i64 7, !dbg !17
  %1559 = select i1 %.not41, float %1556, float %1555, !dbg !17
  %1560 = bitcast float %1559 to i32, !dbg !17
  %1561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1560), !dbg !17
  %1562 = bitcast i32 %1561 to float, !dbg !17
  %1563 = select i1 %.not41, float %1555, float %1562, !dbg !17
  %1564 = select i1 %.not41, float %1562, float %1556, !dbg !17
  %1565 = select i1 %.not41, float %1558, float %1557, !dbg !17
  %1566 = bitcast float %1565 to i32, !dbg !17
  %1567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1566), !dbg !17
  %1568 = bitcast i32 %1567 to float, !dbg !17
  %1569 = select i1 %.not41, float %1557, float %1568, !dbg !17
  %1570 = select i1 %.not41, float %1568, float %1558, !dbg !17
  %1571 = select i1 %.not42, float %1569, float %1563, !dbg !17
  %1572 = bitcast float %1571 to i32, !dbg !17
  %1573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1572), !dbg !17
  %1574 = bitcast i32 %1573 to float, !dbg !17
  %1575 = select i1 %.not42, float %1563, float %1574, !dbg !17
  %1576 = select i1 %.not42, float %1574, float %1569, !dbg !17
  %1577 = select i1 %.not42, float %1570, float %1564, !dbg !17
  %1578 = bitcast float %1577 to i32, !dbg !17
  %1579 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1578), !dbg !17
  %1580 = bitcast i32 %1579 to float, !dbg !17
  %1581 = select i1 %.not42, float %1564, float %1580, !dbg !17
  %1582 = select i1 %.not42, float %1580, float %1570, !dbg !17
  %1583 = extractelement <8 x float> %1472, i64 4, !dbg !17
  %1584 = extractelement <8 x float> %1472, i64 5, !dbg !17
  %1585 = extractelement <8 x float> %1472, i64 6, !dbg !17
  %1586 = extractelement <8 x float> %1472, i64 7, !dbg !17
  %1587 = select i1 %.not41, float %1584, float %1583, !dbg !17
  %1588 = bitcast float %1587 to i32, !dbg !17
  %1589 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1588), !dbg !17
  %1590 = bitcast i32 %1589 to float, !dbg !17
  %1591 = select i1 %.not41, float %1583, float %1590, !dbg !17
  %1592 = select i1 %.not41, float %1590, float %1584, !dbg !17
  %1593 = select i1 %.not41, float %1586, float %1585, !dbg !17
  %1594 = bitcast float %1593 to i32, !dbg !17
  %1595 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1594), !dbg !17
  %1596 = bitcast i32 %1595 to float, !dbg !17
  %1597 = select i1 %.not41, float %1585, float %1596, !dbg !17
  %1598 = select i1 %.not41, float %1596, float %1586, !dbg !17
  %1599 = select i1 %.not42, float %1597, float %1591, !dbg !17
  %1600 = bitcast float %1599 to i32, !dbg !17
  %1601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1600), !dbg !17
  %1602 = bitcast i32 %1601 to float, !dbg !17
  %1603 = select i1 %.not42, float %1591, float %1602, !dbg !17
  %1604 = select i1 %.not42, float %1602, float %1597, !dbg !17
  %1605 = select i1 %.not42, float %1598, float %1592, !dbg !17
  %1606 = bitcast float %1605 to i32, !dbg !17
  %1607 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1606), !dbg !17
  %1608 = bitcast i32 %1607 to float, !dbg !17
  %1609 = select i1 %.not42, float %1592, float %1608, !dbg !17
  %1610 = select i1 %.not42, float %1608, float %1598, !dbg !17
  %1611 = extractelement <8 x float> %1501, i64 4, !dbg !17
  %1612 = extractelement <8 x float> %1501, i64 5, !dbg !17
  %1613 = extractelement <8 x float> %1501, i64 6, !dbg !17
  %1614 = extractelement <8 x float> %1501, i64 7, !dbg !17
  %1615 = select i1 %.not41, float %1612, float %1611, !dbg !17
  %1616 = bitcast float %1615 to i32, !dbg !17
  %1617 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1616), !dbg !17
  %1618 = bitcast i32 %1617 to float, !dbg !17
  %1619 = select i1 %.not41, float %1611, float %1618, !dbg !17
  %1620 = select i1 %.not41, float %1618, float %1612, !dbg !17
  %1621 = select i1 %.not41, float %1614, float %1613, !dbg !17
  %1622 = bitcast float %1621 to i32, !dbg !17
  %1623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1622), !dbg !17
  %1624 = bitcast i32 %1623 to float, !dbg !17
  %1625 = select i1 %.not41, float %1613, float %1624, !dbg !17
  %1626 = select i1 %.not41, float %1624, float %1614, !dbg !17
  %1627 = select i1 %.not42, float %1625, float %1619, !dbg !17
  %1628 = bitcast float %1627 to i32, !dbg !17
  %1629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1628), !dbg !17
  %1630 = bitcast i32 %1629 to float, !dbg !17
  %1631 = select i1 %.not42, float %1619, float %1630, !dbg !17
  %1632 = select i1 %.not42, float %1630, float %1625, !dbg !17
  %1633 = select i1 %.not42, float %1626, float %1620, !dbg !17
  %1634 = bitcast float %1633 to i32, !dbg !17
  %1635 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1634), !dbg !17
  %1636 = bitcast i32 %1635 to float, !dbg !17
  %1637 = select i1 %.not42, float %1620, float %1636, !dbg !17
  %1638 = select i1 %.not42, float %1636, float %1626, !dbg !17
  %1639 = extractelement <8 x float> %1530, i64 4, !dbg !17
  %1640 = extractelement <8 x float> %1530, i64 5, !dbg !17
  %1641 = extractelement <8 x float> %1530, i64 6, !dbg !17
  %1642 = extractelement <8 x float> %1530, i64 7, !dbg !17
  %1643 = select i1 %.not41, float %1640, float %1639, !dbg !17
  %1644 = bitcast float %1643 to i32, !dbg !17
  %1645 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1644), !dbg !17
  %1646 = bitcast i32 %1645 to float, !dbg !17
  %1647 = select i1 %.not41, float %1639, float %1646, !dbg !17
  %1648 = select i1 %.not41, float %1646, float %1640, !dbg !17
  %1649 = select i1 %.not41, float %1642, float %1641, !dbg !17
  %1650 = bitcast float %1649 to i32, !dbg !17
  %1651 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1226, i32 %1650), !dbg !17
  %1652 = bitcast i32 %1651 to float, !dbg !17
  %1653 = select i1 %.not41, float %1641, float %1652, !dbg !17
  %1654 = select i1 %.not41, float %1652, float %1642, !dbg !17
  %1655 = select i1 %.not42, float %1653, float %1647, !dbg !17
  %1656 = bitcast float %1655 to i32, !dbg !17
  %1657 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1656), !dbg !17
  %1658 = select i1 %.not42, float %1654, float %1648, !dbg !17
  %1659 = bitcast float %1658 to i32, !dbg !17
  %1660 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1242, i32 %1659), !dbg !17
  br i1 true, label %1661, label %1725, !dbg !21

1661:                                             ; preds = %1210
  %1662 = bitcast i32 %1657 to float, !dbg !17
  %1663 = select i1 %.not42, float %1647, float %1662, !dbg !17
  %1664 = insertelement <4 x float> poison, float %1663, i64 0, !dbg !19
  %1665 = bitcast i32 %1660 to float, !dbg !17
  %1666 = select i1 %.not42, float %1648, float %1665, !dbg !17
  %1667 = insertelement <4 x float> %1664, float %1666, i64 1, !dbg !19
  %1668 = select i1 %.not42, float %1662, float %1653, !dbg !17
  %1669 = insertelement <4 x float> %1667, float %1668, i64 2, !dbg !19
  %1670 = select i1 %.not42, float %1665, float %1654, !dbg !17
  %1671 = insertelement <4 x float> %1669, float %1670, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1554, !dbg !19
  %normalized = fmul <4 x float> %1671, %row.reciprocal, !dbg !19
  %1672 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1673 = extractelement <4 x half> %1672, i64 3, !dbg !20
  %1674 = extractelement <4 x half> %1672, i64 2, !dbg !20
  %1675 = extractelement <4 x half> %1672, i64 1, !dbg !20
  %1676 = extractelement <4 x half> %1672, i64 0, !dbg !20
  %1677 = bitcast i32 %1549 to float, !dbg !17
  %1678 = select i1 %.not42, float %1539, float %1677, !dbg !17
  %1679 = insertelement <4 x float> poison, float %1678, i64 0, !dbg !19
  %1680 = bitcast i32 %1552 to float, !dbg !17
  %1681 = select i1 %.not42, float %1540, float %1680, !dbg !17
  %1682 = insertelement <4 x float> %1679, float %1681, i64 1, !dbg !19
  %1683 = select i1 %.not42, float %1677, float %1545, !dbg !17
  %1684 = insertelement <4 x float> %1682, float %1683, i64 2, !dbg !19
  %1685 = select i1 %.not42, float %1680, float %1546, !dbg !17
  %1686 = insertelement <4 x float> %1684, float %1685, i64 3, !dbg !19
  %normalized112 = fmul <4 x float> %1686, %row.reciprocal, !dbg !19
  %1687 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %1688 = extractelement <4 x half> %1687, i64 3, !dbg !20
  %1689 = extractelement <4 x half> %1687, i64 2, !dbg !20
  %1690 = extractelement <4 x half> %1687, i64 1, !dbg !20
  %1691 = extractelement <4 x half> %1687, i64 0, !dbg !20
  %1692 = bitcast i32 %1439 to float, !dbg !17
  %1693 = select i1 %.not42, float %1429, float %1692, !dbg !17
  %1694 = insertelement <4 x float> poison, float %1693, i64 0, !dbg !19
  %1695 = bitcast i32 %1442 to float, !dbg !17
  %1696 = select i1 %.not42, float %1430, float %1695, !dbg !17
  %1697 = insertelement <4 x float> %1694, float %1696, i64 1, !dbg !19
  %1698 = select i1 %.not42, float %1692, float %1435, !dbg !17
  %1699 = insertelement <4 x float> %1697, float %1698, i64 2, !dbg !19
  %1700 = select i1 %.not42, float %1695, float %1436, !dbg !17
  %1701 = insertelement <4 x float> %1699, float %1700, i64 3, !dbg !19
  %row.reciprocal113 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1336, !dbg !19
  %normalized114 = fmul <4 x float> %1701, %row.reciprocal113, !dbg !19
  %1702 = fptrunc <4 x float> %normalized114 to <4 x half>, !dbg !20
  %1703 = extractelement <4 x half> %1702, i64 3, !dbg !20
  %1704 = extractelement <4 x half> %1702, i64 2, !dbg !20
  %1705 = extractelement <4 x half> %1702, i64 1, !dbg !20
  %1706 = extractelement <4 x half> %1702, i64 0, !dbg !20
  %1707 = bitcast i32 %1331 to float, !dbg !17
  %1708 = select i1 %.not42, float %1321, float %1707, !dbg !17
  %1709 = insertelement <4 x float> poison, float %1708, i64 0, !dbg !19
  %1710 = bitcast i32 %1334 to float, !dbg !17
  %1711 = select i1 %.not42, float %1322, float %1710, !dbg !17
  %1712 = insertelement <4 x float> %1709, float %1711, i64 1, !dbg !19
  %1713 = select i1 %.not42, float %1707, float %1327, !dbg !17
  %1714 = insertelement <4 x float> %1712, float %1713, i64 2, !dbg !19
  %1715 = select i1 %.not42, float %1710, float %1328, !dbg !17
  %1716 = insertelement <4 x float> %1714, float %1715, i64 3, !dbg !19
  %normalized115 = fmul <4 x float> %1716, %row.reciprocal113, !dbg !19
  %1717 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %1718 = extractelement <4 x half> %1717, i64 3, !dbg !20
  %1719 = extractelement <4 x half> %1717, i64 2, !dbg !20
  %1720 = extractelement <4 x half> %1717, i64 1, !dbg !20
  %1721 = extractelement <4 x half> %1717, i64 0, !dbg !20
  %1722 = add nsw i64 %10, %17, !dbg !21
  %1723 = shl nsw i64 %1722, 19, !dbg !21
  %1724 = and i64 %1723, 16252928, !dbg !21
  %address.thread857 = call i32 asm sideeffect "", "=v,0"(i32 %12) #6, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local858 = zext nneg i32 %address.thread857 to i64, !dbg !13
  %address.local859 = add nsw i64 %30, %address.local858, !dbg !13
  %address.local860 = lshr i64 %address.local858, 6, !dbg !15
  %address.local861 = shl nuw nsw i64 %address.local860, 4, !dbg !21
  %address.local862 = add nsw i64 %address.local859, %address.local861, !dbg !21
  %address.local863 = and i64 %address.local858, 240, !dbg !21
  %address.local864 = sub nsw i64 %address.local862, %address.local863, !dbg !21
  %address.local865 = add nsw i64 %address.local864, %34, !dbg !21
  %address.local866 = shl nuw nsw i64 %address.local865, 7, !dbg !21
  %address.local867 = add nuw nsw i64 %1724, %address.local866, !dbg !21
  %address.local868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local867, !dbg !21
  %address.local869 = lshr i64 %address.local858, 4, !dbg !15
  %address.local870 = and i64 %address.local869, 3, !dbg !15
  %address.local871 = shl nuw nsw i64 %address.local870, 2, !dbg !21
  %address.local872 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local868, i64 %address.local871, !dbg !21
  store half %1721, ptr addrspace(1) %address.local872, align 2, !dbg !21
  %address.local873 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local867, !dbg !21
  %address.local874 = or disjoint i64 %address.local871, 1, !dbg !21
  %address.local875 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local873, i64 %address.local874, !dbg !21
  store half %1720, ptr addrspace(1) %address.local875, align 2, !dbg !21
  %address.local876 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local867, !dbg !21
  %address.local877 = or disjoint i64 %address.local871, 2, !dbg !21
  %address.local878 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local877, !dbg !21
  store half %1719, ptr addrspace(1) %address.local878, align 2, !dbg !21
  %address.local879 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local867, !dbg !21
  %address.local880 = or disjoint i64 %address.local871, 3, !dbg !21
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local879, i64 %address.local880, !dbg !21
  store half %1718, ptr addrspace(1) %address.local881, align 2, !dbg !21
  %address.local882 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local867, !dbg !21
  %address.local883 = or disjoint i64 %address.local871, 16, !dbg !21
  %address.local884 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local882, i64 %address.local883, !dbg !21
  store half %1706, ptr addrspace(1) %address.local884, align 2, !dbg !21
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local867, !dbg !21
  %address.local886 = or disjoint i64 %address.local871, 17, !dbg !21
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local885, i64 %address.local886, !dbg !21
  store half %1705, ptr addrspace(1) %address.local887, align 2, !dbg !21
  %address.local888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local867, !dbg !21
  %address.local889 = or disjoint i64 %address.local871, 18, !dbg !21
  %address.local890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local888, i64 %address.local889, !dbg !21
  store half %1704, ptr addrspace(1) %address.local890, align 2, !dbg !21
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local867, !dbg !21
  %address.local892 = or disjoint i64 %address.local871, 19, !dbg !21
  %address.local893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local891, i64 %address.local892, !dbg !21
  store half %1703, ptr addrspace(1) %address.local893, align 2, !dbg !21
  %address.local894 = shl nsw i64 %address.local865, 7, !dbg !21
  %address.local895 = add i64 %address.local894, 8192, !dbg !21
  %address.local896 = add nuw nsw i64 %1724, %address.local895, !dbg !21
  %address.local897 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local896, !dbg !21
  %address.local898 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local897, i64 %address.local871, !dbg !21
  store half %1691, ptr addrspace(1) %address.local898, align 2, !dbg !21
  %address.local899 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local896, !dbg !21
  %address.local900 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local899, i64 %address.local874, !dbg !21
  store half %1690, ptr addrspace(1) %address.local900, align 2, !dbg !21
  %address.local901 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local896, !dbg !21
  %address.local902 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local901, i64 %address.local877, !dbg !21
  store half %1689, ptr addrspace(1) %address.local902, align 2, !dbg !21
  %address.local903 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local896, !dbg !21
  %address.local904 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local903, i64 %address.local880, !dbg !21
  store half %1688, ptr addrspace(1) %address.local904, align 2, !dbg !21
  %address.local905 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local896, !dbg !21
  %address.local906 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local905, i64 %address.local883, !dbg !21
  store half %1676, ptr addrspace(1) %address.local906, align 2, !dbg !21
  %address.local907 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local896, !dbg !21
  %address.local908 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local907, i64 %address.local886, !dbg !21
  store half %1675, ptr addrspace(1) %address.local908, align 2, !dbg !21
  %address.local909 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local896, !dbg !21
  %address.local910 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local909, i64 %address.local889, !dbg !21
  store half %1674, ptr addrspace(1) %address.local910, align 2, !dbg !21
  %address.local911 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local896, !dbg !21
  %address.local912 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local911, i64 %address.local892, !dbg !21
  store half %1673, ptr addrspace(1) %address.local912, align 2, !dbg !21
  br label %1725, !dbg !21

1725:                                             ; preds = %1661, %1210
  br i1 true, label %1726, label %1764, !dbg !21

1726:                                             ; preds = %1725
  %1727 = insertelement <4 x float> poison, float %1631, i64 0, !dbg !19
  %1728 = insertelement <4 x float> %1727, float %1637, i64 1, !dbg !19
  %1729 = insertelement <4 x float> %1728, float %1632, i64 2, !dbg !19
  %1730 = insertelement <4 x float> %1729, float %1638, i64 3, !dbg !19
  %row.reciprocal116 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1554, !dbg !19
  %normalized117 = fmul <4 x float> %1730, %row.reciprocal116, !dbg !19
  %1731 = fptrunc <4 x float> %normalized117 to <4 x half>, !dbg !20
  %1732 = extractelement <4 x half> %1731, i64 3, !dbg !20
  %1733 = extractelement <4 x half> %1731, i64 2, !dbg !20
  %1734 = extractelement <4 x half> %1731, i64 1, !dbg !20
  %1735 = extractelement <4 x half> %1731, i64 0, !dbg !20
  %1736 = insertelement <4 x float> poison, float %1522, i64 0, !dbg !19
  %1737 = insertelement <4 x float> %1736, float %1528, i64 1, !dbg !19
  %1738 = insertelement <4 x float> %1737, float %1523, i64 2, !dbg !19
  %1739 = insertelement <4 x float> %1738, float %1529, i64 3, !dbg !19
  %normalized118 = fmul <4 x float> %1739, %row.reciprocal116, !dbg !19
  %1740 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %1741 = extractelement <4 x half> %1740, i64 3, !dbg !20
  %1742 = extractelement <4 x half> %1740, i64 2, !dbg !20
  %1743 = extractelement <4 x half> %1740, i64 1, !dbg !20
  %1744 = extractelement <4 x half> %1740, i64 0, !dbg !20
  %1745 = insertelement <4 x float> poison, float %1413, i64 0, !dbg !19
  %1746 = insertelement <4 x float> %1745, float %1419, i64 1, !dbg !19
  %1747 = insertelement <4 x float> %1746, float %1414, i64 2, !dbg !19
  %1748 = insertelement <4 x float> %1747, float %1420, i64 3, !dbg !19
  %row.reciprocal119 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1336, !dbg !19
  %normalized120 = fmul <4 x float> %1748, %row.reciprocal119, !dbg !19
  %1749 = fptrunc <4 x float> %normalized120 to <4 x half>, !dbg !20
  %1750 = extractelement <4 x half> %1749, i64 3, !dbg !20
  %1751 = extractelement <4 x half> %1749, i64 2, !dbg !20
  %1752 = extractelement <4 x half> %1749, i64 1, !dbg !20
  %1753 = extractelement <4 x half> %1749, i64 0, !dbg !20
  %1754 = insertelement <4 x float> poison, float %1304, i64 0, !dbg !19
  %1755 = insertelement <4 x float> %1754, float %1310, i64 1, !dbg !19
  %1756 = insertelement <4 x float> %1755, float %1305, i64 2, !dbg !19
  %1757 = insertelement <4 x float> %1756, float %1311, i64 3, !dbg !19
  %normalized121 = fmul <4 x float> %1757, %row.reciprocal119, !dbg !19
  %1758 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %1759 = extractelement <4 x half> %1758, i64 3, !dbg !20
  %1760 = extractelement <4 x half> %1758, i64 2, !dbg !20
  %1761 = extractelement <4 x half> %1758, i64 1, !dbg !20
  %1762 = extractelement <4 x half> %1758, i64 0, !dbg !20
  %1763 = shl nuw nsw i64 %19, 19, !dbg !21
  %address.thread913 = call i32 asm sideeffect "", "=v,0"(i32 %12) #6, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local914 = zext nneg i32 %address.thread913 to i64, !dbg !13
  %address.local915 = add nsw i64 %30, %address.local914, !dbg !13
  %address.local916 = lshr i64 %address.local914, 6, !dbg !15
  %address.local917 = shl nuw nsw i64 %address.local916, 4, !dbg !21
  %address.local918 = add nsw i64 %address.local915, %address.local917, !dbg !21
  %address.local919 = and i64 %address.local914, 240, !dbg !21
  %address.local920 = sub nsw i64 %address.local918, %address.local919, !dbg !21
  %address.local921 = add nsw i64 %address.local920, %34, !dbg !21
  %address.local922 = shl nuw nsw i64 %address.local921, 7, !dbg !21
  %address.local923 = add nuw nsw i64 %1763, %address.local922, !dbg !21
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local923, !dbg !21
  %address.local925 = lshr i64 %address.local914, 4, !dbg !15
  %address.local926 = and i64 %address.local925, 3, !dbg !15
  %address.local927 = shl nuw nsw i64 %address.local926, 2, !dbg !21
  %address.local928 = or disjoint i64 %address.local927, 32, !dbg !21
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local924, i64 %address.local928, !dbg !21
  store half %1762, ptr addrspace(1) %address.local929, align 2, !dbg !21
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local923, !dbg !21
  %address.local931 = or disjoint i64 %address.local927, 33, !dbg !21
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local930, i64 %address.local931, !dbg !21
  store half %1761, ptr addrspace(1) %address.local932, align 2, !dbg !21
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local923, !dbg !21
  %address.local934 = or disjoint i64 %address.local927, 34, !dbg !21
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local933, i64 %address.local934, !dbg !21
  store half %1760, ptr addrspace(1) %address.local935, align 2, !dbg !21
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local923, !dbg !21
  %address.local937 = or disjoint i64 %address.local927, 35, !dbg !21
  %address.local938 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local936, i64 %address.local937, !dbg !21
  store half %1759, ptr addrspace(1) %address.local938, align 2, !dbg !21
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local923, !dbg !21
  %address.local940 = or disjoint i64 %address.local927, 48, !dbg !21
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local939, i64 %address.local940, !dbg !21
  store half %1753, ptr addrspace(1) %address.local941, align 2, !dbg !21
  %address.local942 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local923, !dbg !21
  %address.local943 = or disjoint i64 %address.local927, 49, !dbg !21
  %address.local944 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local942, i64 %address.local943, !dbg !21
  store half %1752, ptr addrspace(1) %address.local944, align 2, !dbg !21
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local923, !dbg !21
  %address.local946 = or disjoint i64 %address.local927, 50, !dbg !21
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local946, !dbg !21
  store half %1751, ptr addrspace(1) %address.local947, align 2, !dbg !21
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local923, !dbg !21
  %address.local949 = or disjoint i64 %address.local927, 51, !dbg !21
  %address.local950 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local948, i64 %address.local949, !dbg !21
  store half %1750, ptr addrspace(1) %address.local950, align 2, !dbg !21
  %address.local951 = shl nsw i64 %address.local921, 7, !dbg !21
  %address.local952 = add i64 %address.local951, 8192, !dbg !21
  %address.local953 = add nuw nsw i64 %1763, %address.local952, !dbg !21
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local954, i64 %address.local928, !dbg !21
  store half %1744, ptr addrspace(1) %address.local955, align 2, !dbg !21
  %address.local956 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local956, i64 %address.local931, !dbg !21
  store half %1743, ptr addrspace(1) %address.local957, align 2, !dbg !21
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local959 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local958, i64 %address.local934, !dbg !21
  store half %1742, ptr addrspace(1) %address.local959, align 2, !dbg !21
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local960, i64 %address.local937, !dbg !21
  store half %1741, ptr addrspace(1) %address.local961, align 2, !dbg !21
  %address.local962 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local940, !dbg !21
  store half %1735, ptr addrspace(1) %address.local963, align 2, !dbg !21
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local965 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local964, i64 %address.local943, !dbg !21
  store half %1734, ptr addrspace(1) %address.local965, align 2, !dbg !21
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local966, i64 %address.local946, !dbg !21
  store half %1733, ptr addrspace(1) %address.local967, align 2, !dbg !21
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local968, i64 %address.local949, !dbg !21
  store half %1732, ptr addrspace(1) %address.local969, align 2, !dbg !21
  br label %1764, !dbg !21

1764:                                             ; preds = %1726, %1725
  br i1 true, label %1765, label %1803, !dbg !21

1765:                                             ; preds = %1764
  %1766 = insertelement <4 x float> poison, float %1603, i64 0, !dbg !19
  %1767 = insertelement <4 x float> %1766, float %1609, i64 1, !dbg !19
  %1768 = insertelement <4 x float> %1767, float %1604, i64 2, !dbg !19
  %1769 = insertelement <4 x float> %1768, float %1610, i64 3, !dbg !19
  %row.reciprocal122 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1554, !dbg !19
  %normalized123 = fmul <4 x float> %1769, %row.reciprocal122, !dbg !19
  %1770 = fptrunc <4 x float> %normalized123 to <4 x half>, !dbg !20
  %1771 = extractelement <4 x half> %1770, i64 3, !dbg !20
  %1772 = extractelement <4 x half> %1770, i64 2, !dbg !20
  %1773 = extractelement <4 x half> %1770, i64 1, !dbg !20
  %1774 = extractelement <4 x half> %1770, i64 0, !dbg !20
  %1775 = insertelement <4 x float> poison, float %1493, i64 0, !dbg !19
  %1776 = insertelement <4 x float> %1775, float %1499, i64 1, !dbg !19
  %1777 = insertelement <4 x float> %1776, float %1494, i64 2, !dbg !19
  %1778 = insertelement <4 x float> %1777, float %1500, i64 3, !dbg !19
  %normalized124 = fmul <4 x float> %1778, %row.reciprocal122, !dbg !19
  %1779 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %1780 = extractelement <4 x half> %1779, i64 3, !dbg !20
  %1781 = extractelement <4 x half> %1779, i64 2, !dbg !20
  %1782 = extractelement <4 x half> %1779, i64 1, !dbg !20
  %1783 = extractelement <4 x half> %1779, i64 0, !dbg !20
  %1784 = insertelement <4 x float> poison, float %1385, i64 0, !dbg !19
  %1785 = insertelement <4 x float> %1784, float %1391, i64 1, !dbg !19
  %1786 = insertelement <4 x float> %1785, float %1386, i64 2, !dbg !19
  %1787 = insertelement <4 x float> %1786, float %1392, i64 3, !dbg !19
  %row.reciprocal125 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1336, !dbg !19
  %normalized126 = fmul <4 x float> %1787, %row.reciprocal125, !dbg !19
  %1788 = fptrunc <4 x float> %normalized126 to <4 x half>, !dbg !20
  %1789 = extractelement <4 x half> %1788, i64 3, !dbg !20
  %1790 = extractelement <4 x half> %1788, i64 2, !dbg !20
  %1791 = extractelement <4 x half> %1788, i64 1, !dbg !20
  %1792 = extractelement <4 x half> %1788, i64 0, !dbg !20
  %1793 = insertelement <4 x float> poison, float %1275, i64 0, !dbg !19
  %1794 = insertelement <4 x float> %1793, float %1281, i64 1, !dbg !19
  %1795 = insertelement <4 x float> %1794, float %1276, i64 2, !dbg !19
  %1796 = insertelement <4 x float> %1795, float %1282, i64 3, !dbg !19
  %normalized127 = fmul <4 x float> %1796, %row.reciprocal125, !dbg !19
  %1797 = fptrunc <4 x float> %normalized127 to <4 x half>, !dbg !20
  %1798 = extractelement <4 x half> %1797, i64 3, !dbg !20
  %1799 = extractelement <4 x half> %1797, i64 2, !dbg !20
  %1800 = extractelement <4 x half> %1797, i64 1, !dbg !20
  %1801 = extractelement <4 x half> %1797, i64 0, !dbg !20
  %1802 = shl nuw nsw i64 %19, 19, !dbg !21
  %address.thread970 = call i32 asm sideeffect "", "=v,0"(i32 %12) #6, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local971 = zext nneg i32 %address.thread970 to i64, !dbg !13
  %address.local972 = add nsw i64 %30, %address.local971, !dbg !13
  %address.local973 = lshr i64 %address.local971, 6, !dbg !15
  %address.local974 = shl nuw nsw i64 %address.local973, 4, !dbg !21
  %address.local975 = add nsw i64 %address.local972, %address.local974, !dbg !21
  %address.local976 = and i64 %address.local971, 240, !dbg !21
  %address.local977 = sub nsw i64 %address.local975, %address.local976, !dbg !21
  %address.local978 = add nsw i64 %address.local977, %34, !dbg !21
  %address.local979 = shl nuw nsw i64 %address.local978, 7, !dbg !21
  %address.local980 = add nuw nsw i64 %1802, %address.local979, !dbg !21
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local980, !dbg !21
  %address.local982 = lshr i64 %address.local971, 4, !dbg !15
  %address.local983 = and i64 %address.local982, 3, !dbg !15
  %address.local984 = shl nuw nsw i64 %address.local983, 2, !dbg !21
  %address.local985 = or disjoint i64 %address.local984, 64, !dbg !21
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local981, i64 %address.local985, !dbg !21
  store half %1801, ptr addrspace(1) %address.local986, align 2, !dbg !21
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local980, !dbg !21
  %address.local988 = or disjoint i64 %address.local984, 65, !dbg !21
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local987, i64 %address.local988, !dbg !21
  store half %1800, ptr addrspace(1) %address.local989, align 2, !dbg !21
  %address.local990 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local980, !dbg !21
  %address.local991 = or disjoint i64 %address.local984, 66, !dbg !21
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local991, !dbg !21
  store half %1799, ptr addrspace(1) %address.local992, align 2, !dbg !21
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local980, !dbg !21
  %address.local994 = or disjoint i64 %address.local984, 67, !dbg !21
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local993, i64 %address.local994, !dbg !21
  store half %1798, ptr addrspace(1) %address.local995, align 2, !dbg !21
  %address.local996 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local980, !dbg !21
  %address.local997 = or disjoint i64 %address.local984, 80, !dbg !21
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local996, i64 %address.local997, !dbg !21
  store half %1792, ptr addrspace(1) %address.local998, align 2, !dbg !21
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local980, !dbg !21
  %address.local1000 = or disjoint i64 %address.local984, 81, !dbg !21
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local999, i64 %address.local1000, !dbg !21
  store half %1791, ptr addrspace(1) %address.local1001, align 2, !dbg !21
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local980, !dbg !21
  %address.local1003 = or disjoint i64 %address.local984, 82, !dbg !21
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1002, i64 %address.local1003, !dbg !21
  store half %1790, ptr addrspace(1) %address.local1004, align 2, !dbg !21
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local980, !dbg !21
  %address.local1006 = or disjoint i64 %address.local984, 83, !dbg !21
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1005, i64 %address.local1006, !dbg !21
  store half %1789, ptr addrspace(1) %address.local1007, align 2, !dbg !21
  %address.local1008 = shl nsw i64 %address.local978, 7, !dbg !21
  %address.local1009 = add i64 %address.local1008, 8192, !dbg !21
  %address.local1010 = add nuw nsw i64 %1802, %address.local1009, !dbg !21
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1010, !dbg !21
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local985, !dbg !21
  store half %1783, ptr addrspace(1) %address.local1012, align 2, !dbg !21
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1010, !dbg !21
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local988, !dbg !21
  store half %1782, ptr addrspace(1) %address.local1014, align 2, !dbg !21
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1010, !dbg !21
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1015, i64 %address.local991, !dbg !21
  store half %1781, ptr addrspace(1) %address.local1016, align 2, !dbg !21
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1010, !dbg !21
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local994, !dbg !21
  store half %1780, ptr addrspace(1) %address.local1018, align 2, !dbg !21
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1010, !dbg !21
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local997, !dbg !21
  store half %1774, ptr addrspace(1) %address.local1020, align 2, !dbg !21
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1010, !dbg !21
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1021, i64 %address.local1000, !dbg !21
  store half %1773, ptr addrspace(1) %address.local1022, align 2, !dbg !21
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1010, !dbg !21
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1003, !dbg !21
  store half %1772, ptr addrspace(1) %address.local1024, align 2, !dbg !21
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1010, !dbg !21
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1025, i64 %address.local1006, !dbg !21
  store half %1771, ptr addrspace(1) %address.local1026, align 2, !dbg !21
  br label %1803, !dbg !21

1803:                                             ; preds = %1765, %1764
  br i1 true, label %1804, label %1842, !dbg !21

1804:                                             ; preds = %1803
  %1805 = insertelement <4 x float> poison, float %1575, i64 0, !dbg !19
  %1806 = insertelement <4 x float> %1805, float %1581, i64 1, !dbg !19
  %1807 = insertelement <4 x float> %1806, float %1576, i64 2, !dbg !19
  %1808 = insertelement <4 x float> %1807, float %1582, i64 3, !dbg !19
  %row.reciprocal128 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1554, !dbg !19
  %normalized129 = fmul <4 x float> %1808, %row.reciprocal128, !dbg !19
  %1809 = fptrunc <4 x float> %normalized129 to <4 x half>, !dbg !20
  %1810 = extractelement <4 x half> %1809, i64 3, !dbg !20
  %1811 = extractelement <4 x half> %1809, i64 2, !dbg !20
  %1812 = extractelement <4 x half> %1809, i64 1, !dbg !20
  %1813 = extractelement <4 x half> %1809, i64 0, !dbg !20
  %1814 = insertelement <4 x float> poison, float %1464, i64 0, !dbg !19
  %1815 = insertelement <4 x float> %1814, float %1470, i64 1, !dbg !19
  %1816 = insertelement <4 x float> %1815, float %1465, i64 2, !dbg !19
  %1817 = insertelement <4 x float> %1816, float %1471, i64 3, !dbg !19
  %normalized130 = fmul <4 x float> %1817, %row.reciprocal128, !dbg !19
  %1818 = fptrunc <4 x float> %normalized130 to <4 x half>, !dbg !20
  %1819 = extractelement <4 x half> %1818, i64 3, !dbg !20
  %1820 = extractelement <4 x half> %1818, i64 2, !dbg !20
  %1821 = extractelement <4 x half> %1818, i64 1, !dbg !20
  %1822 = extractelement <4 x half> %1818, i64 0, !dbg !20
  %1823 = insertelement <4 x float> poison, float %1357, i64 0, !dbg !19
  %1824 = insertelement <4 x float> %1823, float %1363, i64 1, !dbg !19
  %1825 = insertelement <4 x float> %1824, float %1358, i64 2, !dbg !19
  %1826 = insertelement <4 x float> %1825, float %1364, i64 3, !dbg !19
  %row.reciprocal131 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1336, !dbg !19
  %normalized132 = fmul <4 x float> %1826, %row.reciprocal131, !dbg !19
  %1827 = fptrunc <4 x float> %normalized132 to <4 x half>, !dbg !20
  %1828 = extractelement <4 x half> %1827, i64 3, !dbg !20
  %1829 = extractelement <4 x half> %1827, i64 2, !dbg !20
  %1830 = extractelement <4 x half> %1827, i64 1, !dbg !20
  %1831 = extractelement <4 x half> %1827, i64 0, !dbg !20
  %1832 = insertelement <4 x float> poison, float %1246, i64 0, !dbg !19
  %1833 = insertelement <4 x float> %1832, float %1252, i64 1, !dbg !19
  %1834 = insertelement <4 x float> %1833, float %1247, i64 2, !dbg !19
  %1835 = insertelement <4 x float> %1834, float %1253, i64 3, !dbg !19
  %normalized133 = fmul <4 x float> %1835, %row.reciprocal131, !dbg !19
  %1836 = fptrunc <4 x float> %normalized133 to <4 x half>, !dbg !20
  %1837 = extractelement <4 x half> %1836, i64 3, !dbg !20
  %1838 = extractelement <4 x half> %1836, i64 2, !dbg !20
  %1839 = extractelement <4 x half> %1836, i64 1, !dbg !20
  %1840 = extractelement <4 x half> %1836, i64 0, !dbg !20
  %1841 = shl nuw nsw i64 %19, 19, !dbg !21
  %address.thread1027 = call i32 asm sideeffect "", "=v,0"(i32 %12) #6, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local1028 = zext nneg i32 %address.thread1027 to i64, !dbg !13
  %address.local1029 = add nsw i64 %30, %address.local1028, !dbg !13
  %address.local1030 = lshr i64 %address.local1028, 6, !dbg !15
  %address.local1031 = shl nuw nsw i64 %address.local1030, 4, !dbg !21
  %address.local1032 = add nsw i64 %address.local1029, %address.local1031, !dbg !21
  %address.local1033 = and i64 %address.local1028, 240, !dbg !21
  %address.local1034 = sub nsw i64 %address.local1032, %address.local1033, !dbg !21
  %address.local1035 = add nsw i64 %address.local1034, %34, !dbg !21
  %address.local1036 = shl nuw nsw i64 %address.local1035, 7, !dbg !21
  %address.local1037 = add nuw nsw i64 %1841, %address.local1036, !dbg !21
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1037, !dbg !21
  %address.local1039 = lshr i64 %address.local1028, 4, !dbg !15
  %address.local1040 = and i64 %address.local1039, 3, !dbg !15
  %address.local1041 = shl nuw nsw i64 %address.local1040, 2, !dbg !21
  %address.local1042 = or disjoint i64 %address.local1041, 96, !dbg !21
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1042, !dbg !21
  store half %1840, ptr addrspace(1) %address.local1043, align 2, !dbg !21
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1037, !dbg !21
  %address.local1045 = or disjoint i64 %address.local1041, 97, !dbg !21
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1044, i64 %address.local1045, !dbg !21
  store half %1839, ptr addrspace(1) %address.local1046, align 2, !dbg !21
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1037, !dbg !21
  %address.local1048 = or disjoint i64 %address.local1041, 98, !dbg !21
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1047, i64 %address.local1048, !dbg !21
  store half %1838, ptr addrspace(1) %address.local1049, align 2, !dbg !21
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1037, !dbg !21
  %address.local1051 = or disjoint i64 %address.local1041, 99, !dbg !21
  %address.local1052 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1051, !dbg !21
  store half %1837, ptr addrspace(1) %address.local1052, align 2, !dbg !21
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1037, !dbg !21
  %address.local1054 = or disjoint i64 %address.local1041, 112, !dbg !21
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1054, !dbg !21
  store half %1831, ptr addrspace(1) %address.local1055, align 2, !dbg !21
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1037, !dbg !21
  %address.local1057 = or disjoint i64 %address.local1041, 113, !dbg !21
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1057, !dbg !21
  store half %1830, ptr addrspace(1) %address.local1058, align 2, !dbg !21
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1037, !dbg !21
  %address.local1060 = or disjoint i64 %address.local1041, 114, !dbg !21
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1059, i64 %address.local1060, !dbg !21
  store half %1829, ptr addrspace(1) %address.local1061, align 2, !dbg !21
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1037, !dbg !21
  %address.local1063 = or disjoint i64 %address.local1041, 115, !dbg !21
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1062, i64 %address.local1063, !dbg !21
  store half %1828, ptr addrspace(1) %address.local1064, align 2, !dbg !21
  %address.local1065 = shl nsw i64 %address.local1035, 7, !dbg !21
  %address.local1066 = add i64 %address.local1065, 8192, !dbg !21
  %address.local1067 = add nuw nsw i64 %1841, %address.local1066, !dbg !21
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1067, !dbg !21
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1068, i64 %address.local1042, !dbg !21
  store half %1822, ptr addrspace(1) %address.local1069, align 2, !dbg !21
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1067, !dbg !21
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1045, !dbg !21
  store half %1821, ptr addrspace(1) %address.local1071, align 2, !dbg !21
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1067, !dbg !21
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1072, i64 %address.local1048, !dbg !21
  store half %1820, ptr addrspace(1) %address.local1073, align 2, !dbg !21
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1067, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1074, i64 %address.local1051, !dbg !21
  store half %1819, ptr addrspace(1) %address.local1075, align 2, !dbg !21
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1067, !dbg !21
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1076, i64 %address.local1054, !dbg !21
  store half %1813, ptr addrspace(1) %address.local1077, align 2, !dbg !21
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1067, !dbg !21
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1057, !dbg !21
  store half %1812, ptr addrspace(1) %address.local1079, align 2, !dbg !21
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1067, !dbg !21
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1080, i64 %address.local1060, !dbg !21
  store half %1811, ptr addrspace(1) %address.local1081, align 2, !dbg !21
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1067, !dbg !21
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1063, !dbg !21
  store half %1810, ptr addrspace(1) %address.local1083, align 2, !dbg !21
  br label %1842, !dbg !21

1842:                                             ; preds = %1804, %1803
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #5

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn }
attributes #5 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { convergent nounwind willreturn memory(none) }
attributes #7 = { convergent nounwind memory(none) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!33}

!0 = !DIFile(filename: "test_input_spidD.mlir", directory: "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!8 = !DILocation(line: 69, column: 26, scope: !4)
!9 = !DILocation(line: 92, column: 20, scope: !4)
!11 = !DILocation(line: 46, column: 17, scope: !4)
!13 = !DILocation(line: 66, column: 22, scope: !4)
!14 = !DILocation(line: 67, column: 22, scope: !4)
!15 = !DILocation(line: 68, column: 24, scope: !4)
!16 = !DILocation(line: 73, column: 19, scope: !4)
!17 = !DILocation(line: 56, column: 24, scope: !4)
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
