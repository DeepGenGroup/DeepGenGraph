; ModuleID = '3rd/deepgengraph/test/test_input_spidD.mlir'
source_filename = "3rd/deepgengraph/test/test_input_spidD.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #0 !dbg !4 !reqd_work_group_size !32 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %6, 7, !dbg !11
  %.lobit = ashr i32 %5, 31, !dbg !13
  %10 = xor i32 %5, %.lobit, !dbg !13
  %11 = sdiv i32 %10, 32, !dbg !13
  %.lobit1 = ashr i32 %5, 31, !dbg !13
  %12 = xor i32 %11, %.lobit1, !dbg !13
  %13 = sext i32 %12 to i64, !dbg !13
  %14 = add nsw i64 %8, %13, !dbg !13
  %15 = and i64 %14, 31, !dbg !13
  %16 = shl nsw i64 %6, 7, !dbg !13
  %17 = mul nsw i64 %13, -4096, !dbg !13
  %18 = add nsw i64 %16, %17, !dbg !13
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !14, !range !34
  %20 = zext nneg i32 %19 to i64, !dbg !14
  %21 = shl nuw nsw i64 %20, 2, !dbg !14
  %22 = lshr i64 %20, 6, !dbg !14
  %23 = lshr i64 %20, 4, !dbg !14
  %24 = and i64 %23, 3, !dbg !14
  %25 = or disjoint i64 %9, 127, !dbg !15
  %26 = add nsw i64 %9, 128, !dbg !15
  %27 = icmp uge i64 %26, %25, !dbg !15
  %28 = lshr i64 %20, 3, !dbg !9
  %29 = lshr i64 %20, 1, !dbg !9
  %30 = mul nsw i64 %28, -36, !dbg !9
  %31 = add nsw i64 %21, %30, !dbg !9
  %32 = add nsw i64 %31, %29, !dbg !9
  %33 = add nsw i64 %32, 1, !dbg !9
  %.neg = sdiv i64 %33, -4, !dbg !9
  %34 = add nsw i64 %32, 2, !dbg !9
  %.neg230 = sdiv i64 %34, -4, !dbg !9
  %35 = lshr i64 %20, 2, !dbg !8
  %36 = and i64 %20, 15, !dbg !8
  %37 = lshr i64 %36, 2, !dbg !8
  %38 = lshr i64 %24, 1, !dbg !8
  %39 = mul nsw i64 %35, -18, !dbg !8
  %40 = add nsw i64 %21, %39, !dbg !8
  %41 = add nsw i64 %40, %29, !dbg !8
  %42 = add nsw i64 %41, %38, !dbg !8
  %.neg237 = sdiv i64 %42, -2, !dbg !8
  %43 = mul nuw nsw i64 %20, 18, !dbg !8
  %44 = mul nsw i64 %35, -72, !dbg !8
  %45 = add nsw i64 %43, %44, !dbg !8
  %46 = add nsw i64 %45, %37, !dbg !8
  %.neg239 = sdiv i64 %46, -4, !dbg !8
  br label %47, !dbg !16

47:                                               ; preds = %447, %4
  %48 = phi i64 [ %1234, %447 ], [ 0, %4 ], !dbg !16
  %49 = phi [2 x <8 x float>] [ %1198, %447 ], [ zeroinitializer, %4 ], !dbg !16
  %50 = phi [2 x <1 x float>] [ %818, %447 ], [ zeroinitializer, %4 ], !dbg !16
  %51 = phi [2 x <8 x float>] [ %1205, %447 ], [ zeroinitializer, %4 ], !dbg !16
  %52 = phi [2 x <8 x float>] [ %1211, %447 ], [ zeroinitializer, %4 ], !dbg !16
  %53 = phi [2 x <8 x float>] [ %1233, %447 ], [ zeroinitializer, %4 ], !dbg !16
  %.not = icmp sgt i64 %48, %6, !dbg !16
  br i1 %.not, label %1235, label %54, !dbg !16

54:                                               ; preds = %47
  %55 = shl nsw i64 %48, 7, !dbg !16
  %56 = lshr i64 %48, 5, !dbg !17
  %57 = shl i64 %56, 12, !dbg !17
  %58 = sub i64 %55, %57, !dbg !17
  br label %59, !dbg !22

59:                                               ; preds = %63, %54
  %60 = phi i64 [ %268, %63 ], [ 0, %54 ], !dbg !22
  %61 = phi [2 x <32 x float>] [ %267, %63 ], [ zeroinitializer, %54 ], !dbg !22
  %62 = icmp samesign ult i64 %60, 4, !dbg !22
  br i1 %62, label %63, label %269, !dbg !22

63:                                               ; preds = %59
  %64 = shl nuw nsw i64 %60, 5, !dbg !22
  %.idx = shl nuw nsw i64 %15, 20, !dbg !13
  %65 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !13
  %.idx104 = shl nsw i64 %18, 8, !dbg !13
  %66 = getelementptr i8, ptr addrspace(1) %65, i64 %.idx104, !dbg !13
  %67 = getelementptr [2 x i8], ptr addrspace(1) %66, i64 %64, !dbg !13
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !13, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !14
  %address.local312 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local313 = and i64 %address.local312, 384, !dbg !13
  %address.local314 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local313, !dbg !13
  %address.local315 = ptrtoint ptr addrspace(3) %address.local314 to i64, !dbg !13
  %address.local316 = trunc i64 %address.local315 to i32, !dbg !13
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local316), !dbg !13
  %69 = zext i32 %68 to i64, !dbg !13
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !13
  %address.local317 = shl nuw nsw i64 %address.local, 6, !dbg !13
  %address.local318 = and i64 %address.local317, 16256, !dbg !13
  %address.local319 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local318, !dbg !13
  %address.local320 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local321 = and i64 %address.local320, 2, !dbg !13
  %address.local322 = getelementptr [2 x i8], ptr addrspace(1) %address.local319, i64 %address.local321, !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local322, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !13
  %71 = getelementptr i8, ptr addrspace(1) %67, i64 8, !dbg !13
  %address.local323 = or disjoint i64 %address.local, 256, !dbg !13
  %address.local324 = shl nuw nsw i64 %address.local323, 1, !dbg !13
  %address.local325 = and i64 %address.local324, 896, !dbg !13
  %address.local326 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local325, !dbg !13
  %address.local327 = ptrtoint ptr addrspace(3) %address.local326 to i64, !dbg !13
  %address.local328 = trunc i64 %address.local327 to i32, !dbg !13
  %72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local328), !dbg !13
  %73 = zext i32 %72 to i64, !dbg !13
  %74 = inttoptr i64 %73 to ptr addrspace(3), !dbg !13
  %address.local329 = shl nuw nsw i64 %address.local, 6, !dbg !13
  %address.local330 = and i64 %address.local329, 16256, !dbg !13
  %address.local331 = getelementptr [2 x i8], ptr addrspace(1) %71, i64 %address.local330, !dbg !13
  %address.local332 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local333 = and i64 %address.local332, 2, !dbg !13
  %address.local334 = getelementptr [2 x i8], ptr addrspace(1) %address.local331, i64 %address.local333, !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local334, ptr addrspace(3) %74, i32 4, i32 0, i32 0), !dbg !13
  %75 = getelementptr i8, ptr addrspace(1) %67, i64 16, !dbg !13
  %address.local335 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local336 = and i64 %address.local335, 384, !dbg !13
  %address.local337 = or disjoint i64 %address.local336, 1024, !dbg !13
  %address.local338 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local337, !dbg !13
  %address.local339 = ptrtoint ptr addrspace(3) %address.local338 to i64, !dbg !13
  %address.local340 = trunc i64 %address.local339 to i32, !dbg !13
  %76 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !13
  %77 = zext i32 %76 to i64, !dbg !13
  %78 = inttoptr i64 %77 to ptr addrspace(3), !dbg !13
  %address.local341 = shl nuw nsw i64 %address.local, 6, !dbg !13
  %address.local342 = and i64 %address.local341, 16256, !dbg !13
  %address.local343 = getelementptr [2 x i8], ptr addrspace(1) %75, i64 %address.local342, !dbg !13
  %address.local344 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local345 = and i64 %address.local344, 2, !dbg !13
  %address.local346 = getelementptr [2 x i8], ptr addrspace(1) %address.local343, i64 %address.local345, !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local346, ptr addrspace(3) %78, i32 4, i32 0, i32 0), !dbg !13
  %79 = getelementptr i8, ptr addrspace(1) %67, i64 24, !dbg !13
  %address.local347 = or disjoint i64 %address.local, 768, !dbg !13
  %address.local348 = shl nuw nsw i64 %address.local347, 1, !dbg !13
  %address.local349 = and i64 %address.local348, 1920, !dbg !13
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local349, !dbg !13
  %address.local351 = ptrtoint ptr addrspace(3) %address.local350 to i64, !dbg !13
  %address.local352 = trunc i64 %address.local351 to i32, !dbg !13
  %80 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !13
  %81 = zext i32 %80 to i64, !dbg !13
  %82 = inttoptr i64 %81 to ptr addrspace(3), !dbg !13
  %address.local353 = shl nuw nsw i64 %address.local, 6, !dbg !13
  %address.local354 = and i64 %address.local353, 16256, !dbg !13
  %address.local355 = getelementptr [2 x i8], ptr addrspace(1) %79, i64 %address.local354, !dbg !13
  %address.local356 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local357 = and i64 %address.local356, 2, !dbg !13
  %address.local358 = getelementptr [2 x i8], ptr addrspace(1) %address.local355, i64 %address.local357, !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local358, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !13
  %83 = getelementptr i8, ptr addrspace(1) %67, i64 32, !dbg !13
  %address.local359 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local360 = and i64 %address.local359, 384, !dbg !13
  %address.local361 = or disjoint i64 %address.local360, 2048, !dbg !13
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local361, !dbg !13
  %address.local363 = ptrtoint ptr addrspace(3) %address.local362 to i64, !dbg !13
  %address.local364 = trunc i64 %address.local363 to i32, !dbg !13
  %84 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !13
  %85 = zext i32 %84 to i64, !dbg !13
  %86 = inttoptr i64 %85 to ptr addrspace(3), !dbg !13
  %address.local365 = shl nuw nsw i64 %address.local, 6, !dbg !13
  %address.local366 = and i64 %address.local365, 16256, !dbg !13
  %address.local367 = getelementptr [2 x i8], ptr addrspace(1) %83, i64 %address.local366, !dbg !13
  %address.local368 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local369 = and i64 %address.local368, 2, !dbg !13
  %address.local370 = getelementptr [2 x i8], ptr addrspace(1) %address.local367, i64 %address.local369, !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local370, ptr addrspace(3) %86, i32 4, i32 0, i32 0), !dbg !13
  %87 = getelementptr i8, ptr addrspace(1) %67, i64 40, !dbg !13
  %address.local371 = or disjoint i64 %address.local, 1280, !dbg !13
  %address.local372 = shl nuw nsw i64 %address.local371, 1, !dbg !13
  %address.local373 = and i64 %address.local372, 2944, !dbg !13
  %address.local374 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local373, !dbg !13
  %address.local375 = ptrtoint ptr addrspace(3) %address.local374 to i64, !dbg !13
  %address.local376 = trunc i64 %address.local375 to i32, !dbg !13
  %88 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local376), !dbg !13
  %89 = zext i32 %88 to i64, !dbg !13
  %90 = inttoptr i64 %89 to ptr addrspace(3), !dbg !13
  %address.local377 = shl nuw nsw i64 %address.local, 6, !dbg !13
  %address.local378 = and i64 %address.local377, 16256, !dbg !13
  %address.local379 = getelementptr [2 x i8], ptr addrspace(1) %87, i64 %address.local378, !dbg !13
  %address.local380 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local381 = and i64 %address.local380, 2, !dbg !13
  %address.local382 = getelementptr [2 x i8], ptr addrspace(1) %address.local379, i64 %address.local381, !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local382, ptr addrspace(3) %90, i32 4, i32 0, i32 0), !dbg !13
  %91 = getelementptr i8, ptr addrspace(1) %67, i64 48, !dbg !13
  %address.local383 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local384 = and i64 %address.local383, 384, !dbg !13
  %address.local385 = or disjoint i64 %address.local384, 3072, !dbg !13
  %address.local386 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local385, !dbg !13
  %address.local387 = ptrtoint ptr addrspace(3) %address.local386 to i64, !dbg !13
  %address.local388 = trunc i64 %address.local387 to i32, !dbg !13
  %92 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !13
  %93 = zext i32 %92 to i64, !dbg !13
  %94 = inttoptr i64 %93 to ptr addrspace(3), !dbg !13
  %address.local389 = shl nuw nsw i64 %address.local, 6, !dbg !13
  %address.local390 = and i64 %address.local389, 16256, !dbg !13
  %address.local391 = getelementptr [2 x i8], ptr addrspace(1) %91, i64 %address.local390, !dbg !13
  %address.local392 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local393 = and i64 %address.local392, 2, !dbg !13
  %address.local394 = getelementptr [2 x i8], ptr addrspace(1) %address.local391, i64 %address.local393, !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local394, ptr addrspace(3) %94, i32 4, i32 0, i32 0), !dbg !13
  %95 = getelementptr i8, ptr addrspace(1) %67, i64 56, !dbg !13
  %address.local395 = or disjoint i64 %address.local, 1792, !dbg !13
  %address.local396 = shl nuw nsw i64 %address.local395, 1, !dbg !13
  %address.local397 = and i64 %address.local396, 3968, !dbg !13
  %address.local398 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local397, !dbg !13
  %address.local399 = ptrtoint ptr addrspace(3) %address.local398 to i64, !dbg !13
  %address.local400 = trunc i64 %address.local399 to i32, !dbg !13
  %96 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local400), !dbg !13
  %97 = zext i32 %96 to i64, !dbg !13
  %98 = inttoptr i64 %97 to ptr addrspace(3), !dbg !13
  %address.local401 = shl nuw nsw i64 %address.local, 6, !dbg !13
  %address.local402 = and i64 %address.local401, 16256, !dbg !13
  %address.local403 = getelementptr [2 x i8], ptr addrspace(1) %95, i64 %address.local402, !dbg !13
  %address.local404 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local405 = and i64 %address.local404, 2, !dbg !13
  %address.local406 = getelementptr [2 x i8], ptr addrspace(1) %address.local403, i64 %address.local405, !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local406, ptr addrspace(3) %98, i32 4, i32 0, i32 0), !dbg !13
  call void @llvm.amdgcn.asyncmark(), !dbg !13
  %99 = shl nuw nsw i64 %60, 10, !dbg !17
  %100 = add nuw i64 %48, %99, !dbg !17
  %.lobit105 = ashr i64 %100, 63, !dbg !17
  %101 = xor i64 %100, %.lobit105, !dbg !17
  %102 = sdiv i64 %101, 4096, !dbg !17
  %.lobit106 = ashr i64 %100, 63, !dbg !17
  %103 = xor i64 %102, %.lobit106, !dbg !17
  %104 = add nsw i64 %103, %8, !dbg !17
  %105 = add nuw nsw i64 %64, %56, !dbg !17
  %106 = and i64 %105, 127, !dbg !17
  %107 = shl i64 %104, 19, !dbg !17
  %108 = and i64 %107, 16252928, !dbg !17
  %109 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %108, !dbg !17
  %110 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %106, !dbg !17
  %.idx107 = shl i64 %58, 8, !dbg !17
  %111 = getelementptr i8, ptr addrspace(1) %110, i64 %.idx107, !dbg !17
  %address.local407 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local313, !dbg !17
  %address.local408 = ptrtoint ptr addrspace(3) %address.local407 to i64, !dbg !17
  %address.local409 = trunc i64 %address.local408 to i32, !dbg !17
  %112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local409), !dbg !17
  %113 = zext i32 %112 to i64, !dbg !17
  %114 = inttoptr i64 %113 to ptr addrspace(3), !dbg !17
  %address.local410 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local411 = and i32 %address.local410, 32, !dbg !17
  %address.local412 = zext nneg i32 %address.local411 to i64, !dbg !17
  %address.local413 = getelementptr i8, ptr addrspace(1) %111, i64 %address.local412, !dbg !17
  %address.local414 = lshr i64 %address.local, 2, !dbg !17
  %address.local415 = and i64 %address.local414, 56, !dbg !17
  %address.local416 = lshr i64 %address.local, 4, !dbg !17
  %address.local417 = lshr i64 %address.local, 7, !dbg !17
  %address.local418 = sub nsw i64 %address.local416, %address.local417, !dbg !17
  %address.local419 = shl nsw i64 %address.local418, 2, !dbg !17
  %address.local420 = and i64 %address.local419, 4, !dbg !17
  %address.local421 = or disjoint i64 %address.local415, %address.local420, !dbg !17
  %address.local422 = lshr i64 %address.local, 5, !dbg !17
  %address.local423 = sub nsw i64 %address.local414, %address.local422, !dbg !17
  %address.local424 = and i64 %address.local423, 3, !dbg !17
  %address.local425 = or disjoint i64 %address.local421, %address.local424, !dbg !17
  %address.local426 = shl nuw nsw i64 %address.local425, 8, !dbg !17
  %address.local427 = getelementptr i8, ptr addrspace(1) %address.local413, i64 %address.local426, !dbg !17
  %address.local428 = getelementptr [2 x i8], ptr addrspace(1) %address.local427, i64 %address.local321, !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local428, ptr addrspace(3) %114, i32 4, i32 0, i32 0), !dbg !17
  %address.local429 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local325, !dbg !17
  %address.local430 = ptrtoint ptr addrspace(3) %address.local429 to i64, !dbg !17
  %address.local431 = trunc i64 %address.local430 to i32, !dbg !17
  %115 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local431), !dbg !17
  %116 = zext i32 %115 to i64, !dbg !17
  %117 = inttoptr i64 %116 to ptr addrspace(3), !dbg !17
  %address.local432 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local433 = and i32 %address.local432, 32, !dbg !17
  %address.local434 = zext nneg i32 %address.local433 to i64, !dbg !17
  %address.local435 = getelementptr i8, ptr addrspace(1) %111, i64 %address.local434, !dbg !17
  %address.local436 = getelementptr [2 x i8], ptr addrspace(1) %address.local435, i64 %address.local333, !dbg !17
  %address.local437 = lshr i64 %address.local323, 2, !dbg !17
  %address.local438 = and i64 %address.local437, 120, !dbg !17
  %address.local439 = lshr i64 %address.local, 4, !dbg !17
  %address.local440 = lshr i64 %address.local, 7, !dbg !17
  %address.local441 = sub nsw i64 %address.local439, %address.local440, !dbg !17
  %address.local442 = shl nsw i64 %address.local441, 2, !dbg !17
  %address.local443 = and i64 %address.local442, 4, !dbg !17
  %address.local444 = or disjoint i64 %address.local438, %address.local443, !dbg !17
  %address.local445 = lshr i64 %address.local, 5, !dbg !17
  %address.local446 = sub nuw nsw i64 %address.local437, %address.local445, !dbg !17
  %address.local447 = and i64 %address.local446, 3, !dbg !17
  %address.local448 = or disjoint i64 %address.local444, %address.local447, !dbg !17
  %address.local449 = shl nuw nsw i64 %address.local448, 8, !dbg !17
  %address.local450 = getelementptr i8, ptr addrspace(1) %address.local436, i64 %address.local449, !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local450, ptr addrspace(3) %117, i32 4, i32 0, i32 0), !dbg !17
  %address.local451 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local337, !dbg !17
  %address.local452 = ptrtoint ptr addrspace(3) %address.local451 to i64, !dbg !17
  %address.local453 = trunc i64 %address.local452 to i32, !dbg !17
  %118 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local453), !dbg !17
  %119 = zext i32 %118 to i64, !dbg !17
  %120 = inttoptr i64 %119 to ptr addrspace(3), !dbg !17
  %address.local454 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local455 = and i32 %address.local454, 32, !dbg !17
  %address.local456 = zext nneg i32 %address.local455 to i64, !dbg !17
  %address.local457 = getelementptr i8, ptr addrspace(1) %111, i64 %address.local456, !dbg !17
  %address.local458 = getelementptr i8, ptr addrspace(1) %address.local457, i64 8, !dbg !17
  %address.local459 = getelementptr [2 x i8], ptr addrspace(1) %address.local458, i64 %address.local345, !dbg !17
  %address.local460 = lshr i64 %address.local, 2, !dbg !17
  %address.local461 = and i64 %address.local460, 56, !dbg !17
  %address.local462 = lshr i64 %address.local, 4, !dbg !17
  %address.local463 = lshr i64 %address.local, 7, !dbg !17
  %address.local464 = sub nsw i64 %address.local462, %address.local463, !dbg !17
  %address.local465 = shl nsw i64 %address.local464, 2, !dbg !17
  %address.local466 = and i64 %address.local465, 4, !dbg !17
  %address.local467 = or disjoint i64 %address.local461, %address.local466, !dbg !17
  %address.local468 = lshr i64 %address.local, 5, !dbg !17
  %address.local469 = sub nsw i64 %address.local460, %address.local468, !dbg !17
  %address.local470 = and i64 %address.local469, 3, !dbg !17
  %address.local471 = or disjoint i64 %address.local467, %address.local470, !dbg !17
  %address.local472 = shl nuw nsw i64 %address.local471, 8, !dbg !17
  %address.local473 = getelementptr i8, ptr addrspace(1) %address.local459, i64 %address.local472, !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local473, ptr addrspace(3) %120, i32 4, i32 0, i32 0), !dbg !17
  %address.local474 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local349, !dbg !17
  %address.local475 = ptrtoint ptr addrspace(3) %address.local474 to i64, !dbg !17
  %address.local476 = trunc i64 %address.local475 to i32, !dbg !17
  %121 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local476), !dbg !17
  %122 = zext i32 %121 to i64, !dbg !17
  %123 = inttoptr i64 %122 to ptr addrspace(3), !dbg !17
  %address.local477 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local478 = and i32 %address.local477, 32, !dbg !17
  %address.local479 = zext nneg i32 %address.local478 to i64, !dbg !17
  %address.local480 = getelementptr i8, ptr addrspace(1) %111, i64 %address.local479, !dbg !17
  %address.local481 = getelementptr i8, ptr addrspace(1) %address.local480, i64 8, !dbg !17
  %address.local482 = getelementptr [2 x i8], ptr addrspace(1) %address.local481, i64 %address.local357, !dbg !17
  %address.local483 = lshr i64 %address.local347, 2, !dbg !17
  %address.local484 = and i64 %address.local483, 120, !dbg !17
  %address.local485 = lshr i64 %address.local, 4, !dbg !17
  %address.local486 = lshr i64 %address.local, 7, !dbg !17
  %address.local487 = sub nsw i64 %address.local485, %address.local486, !dbg !17
  %address.local488 = shl nsw i64 %address.local487, 2, !dbg !17
  %address.local489 = and i64 %address.local488, 4, !dbg !17
  %address.local490 = or disjoint i64 %address.local484, %address.local489, !dbg !17
  %address.local491 = lshr i64 %address.local, 5, !dbg !17
  %address.local492 = sub nuw nsw i64 %address.local483, %address.local491, !dbg !17
  %address.local493 = and i64 %address.local492, 3, !dbg !17
  %address.local494 = or disjoint i64 %address.local490, %address.local493, !dbg !17
  %address.local495 = shl nuw nsw i64 %address.local494, 8, !dbg !17
  %address.local496 = getelementptr i8, ptr addrspace(1) %address.local482, i64 %address.local495, !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local496, ptr addrspace(3) %123, i32 4, i32 0, i32 0), !dbg !17
  %address.local497 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local361, !dbg !17
  %address.local498 = ptrtoint ptr addrspace(3) %address.local497 to i64, !dbg !17
  %address.local499 = trunc i64 %address.local498 to i32, !dbg !17
  %124 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local499), !dbg !17
  %125 = zext i32 %124 to i64, !dbg !17
  %126 = inttoptr i64 %125 to ptr addrspace(3), !dbg !17
  %address.local500 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local501 = and i32 %address.local500, 32, !dbg !17
  %address.local502 = zext nneg i32 %address.local501 to i64, !dbg !17
  %address.local503 = getelementptr i8, ptr addrspace(1) %111, i64 %address.local502, !dbg !17
  %address.local504 = getelementptr i8, ptr addrspace(1) %address.local503, i64 16, !dbg !17
  %address.local505 = getelementptr [2 x i8], ptr addrspace(1) %address.local504, i64 %address.local369, !dbg !17
  %address.local506 = lshr i64 %address.local, 2, !dbg !17
  %address.local507 = and i64 %address.local506, 56, !dbg !17
  %address.local508 = lshr i64 %address.local, 4, !dbg !17
  %address.local509 = lshr i64 %address.local, 7, !dbg !17
  %address.local510 = xor i64 %address.local509, -1, !dbg !17
  %address.local511 = add nsw i64 %address.local508, %address.local510, !dbg !17
  %address.local512 = shl nsw i64 %address.local511, 2, !dbg !17
  %address.local513 = and i64 %address.local512, 4, !dbg !17
  %address.local514 = or disjoint i64 %address.local507, %address.local513, !dbg !17
  %address.local515 = lshr i64 %address.local, 5, !dbg !17
  %address.local516 = sub nsw i64 %address.local506, %address.local515, !dbg !17
  %address.local517 = and i64 %address.local516, 3, !dbg !17
  %address.local518 = or disjoint i64 %address.local514, %address.local517, !dbg !17
  %address.local519 = shl nuw nsw i64 %address.local518, 8, !dbg !17
  %address.local520 = getelementptr i8, ptr addrspace(1) %address.local505, i64 %address.local519, !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local520, ptr addrspace(3) %126, i32 4, i32 0, i32 0), !dbg !17
  %address.local521 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local373, !dbg !17
  %address.local522 = ptrtoint ptr addrspace(3) %address.local521 to i64, !dbg !17
  %address.local523 = trunc i64 %address.local522 to i32, !dbg !17
  %127 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local523), !dbg !17
  %128 = zext i32 %127 to i64, !dbg !17
  %129 = inttoptr i64 %128 to ptr addrspace(3), !dbg !17
  %address.local524 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local525 = and i32 %address.local524, 32, !dbg !17
  %address.local526 = zext nneg i32 %address.local525 to i64, !dbg !17
  %address.local527 = getelementptr i8, ptr addrspace(1) %111, i64 %address.local526, !dbg !17
  %address.local528 = getelementptr i8, ptr addrspace(1) %address.local527, i64 16, !dbg !17
  %address.local529 = lshr i64 %address.local371, 2, !dbg !17
  %address.local530 = and i64 %address.local529, 120, !dbg !17
  %address.local531 = lshr i64 %address.local, 4, !dbg !17
  %address.local532 = lshr i64 %address.local, 7, !dbg !17
  %address.local533 = xor i64 %address.local532, -1, !dbg !17
  %address.local534 = add nsw i64 %address.local531, %address.local533, !dbg !17
  %address.local535 = shl nsw i64 %address.local534, 2, !dbg !17
  %address.local536 = and i64 %address.local535, 4, !dbg !17
  %address.local537 = or disjoint i64 %address.local530, %address.local536, !dbg !17
  %address.local538 = lshr i64 %address.local, 5, !dbg !17
  %address.local539 = sub nuw nsw i64 %address.local529, %address.local538, !dbg !17
  %address.local540 = and i64 %address.local539, 3, !dbg !17
  %address.local541 = or disjoint i64 %address.local537, %address.local540, !dbg !17
  %address.local542 = shl nuw nsw i64 %address.local541, 8, !dbg !17
  %address.local543 = getelementptr i8, ptr addrspace(1) %address.local528, i64 %address.local542, !dbg !17
  %address.local544 = getelementptr [2 x i8], ptr addrspace(1) %address.local543, i64 %address.local381, !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local544, ptr addrspace(3) %129, i32 4, i32 0, i32 0), !dbg !17
  %address.local545 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local385, !dbg !17
  %address.local546 = ptrtoint ptr addrspace(3) %address.local545 to i64, !dbg !17
  %address.local547 = trunc i64 %address.local546 to i32, !dbg !17
  %130 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local547), !dbg !17
  %131 = zext i32 %130 to i64, !dbg !17
  %132 = inttoptr i64 %131 to ptr addrspace(3), !dbg !17
  %address.local548 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local549 = and i32 %address.local548, 32, !dbg !17
  %address.local550 = zext nneg i32 %address.local549 to i64, !dbg !17
  %address.local551 = getelementptr i8, ptr addrspace(1) %111, i64 %address.local550, !dbg !17
  %address.local552 = getelementptr i8, ptr addrspace(1) %address.local551, i64 24, !dbg !17
  %address.local553 = lshr i64 %address.local, 2, !dbg !17
  %address.local554 = and i64 %address.local553, 56, !dbg !17
  %address.local555 = lshr i64 %address.local, 4, !dbg !17
  %address.local556 = lshr i64 %address.local, 7, !dbg !17
  %address.local557 = xor i64 %address.local556, -1, !dbg !17
  %address.local558 = add nsw i64 %address.local555, %address.local557, !dbg !17
  %address.local559 = shl nsw i64 %address.local558, 2, !dbg !17
  %address.local560 = and i64 %address.local559, 4, !dbg !17
  %address.local561 = or disjoint i64 %address.local554, %address.local560, !dbg !17
  %address.local562 = lshr i64 %address.local, 5, !dbg !17
  %address.local563 = sub nsw i64 %address.local553, %address.local562, !dbg !17
  %address.local564 = and i64 %address.local563, 3, !dbg !17
  %address.local565 = or disjoint i64 %address.local561, %address.local564, !dbg !17
  %address.local566 = shl nuw nsw i64 %address.local565, 8, !dbg !17
  %address.local567 = getelementptr i8, ptr addrspace(1) %address.local552, i64 %address.local566, !dbg !17
  %address.local568 = getelementptr [2 x i8], ptr addrspace(1) %address.local567, i64 %address.local393, !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local568, ptr addrspace(3) %132, i32 4, i32 0, i32 0), !dbg !17
  %address.local569 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local397, !dbg !17
  %address.local570 = ptrtoint ptr addrspace(3) %address.local569 to i64, !dbg !17
  %address.local571 = trunc i64 %address.local570 to i32, !dbg !17
  %133 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local571), !dbg !17
  %134 = zext i32 %133 to i64, !dbg !17
  %135 = inttoptr i64 %134 to ptr addrspace(3), !dbg !17
  %address.local572 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local573 = and i32 %address.local572, 32, !dbg !17
  %address.local574 = zext nneg i32 %address.local573 to i64, !dbg !17
  %address.local575 = getelementptr i8, ptr addrspace(1) %111, i64 %address.local574, !dbg !17
  %address.local576 = getelementptr i8, ptr addrspace(1) %address.local575, i64 24, !dbg !17
  %address.local577 = lshr i64 %address.local395, 2, !dbg !17
  %address.local578 = and i64 %address.local577, 120, !dbg !17
  %address.local579 = lshr i64 %address.local, 4, !dbg !17
  %address.local580 = lshr i64 %address.local, 7, !dbg !17
  %address.local581 = xor i64 %address.local580, -1, !dbg !17
  %address.local582 = add nsw i64 %address.local579, %address.local581, !dbg !17
  %address.local583 = shl nsw i64 %address.local582, 2, !dbg !17
  %address.local584 = and i64 %address.local583, 4, !dbg !17
  %address.local585 = or disjoint i64 %address.local578, %address.local584, !dbg !17
  %address.local586 = lshr i64 %address.local, 5, !dbg !17
  %address.local587 = sub nuw nsw i64 %address.local577, %address.local586, !dbg !17
  %address.local588 = and i64 %address.local587, 3, !dbg !17
  %address.local589 = or disjoint i64 %address.local585, %address.local588, !dbg !17
  %address.local590 = shl nuw nsw i64 %address.local589, 8, !dbg !17
  %address.local591 = getelementptr i8, ptr addrspace(1) %address.local576, i64 %address.local590, !dbg !17
  %address.local592 = getelementptr [2 x i8], ptr addrspace(1) %address.local591, i64 %address.local405, !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %address.local592, ptr addrspace(3) %135, i32 4, i32 0, i32 0), !dbg !17
  call void @llvm.amdgcn.asyncmark(), !dbg !17
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local593 = and i32 %address.thread, 192, !dbg !14
  %address.local594 = shl nuw nsw i32 %address.thread, 2, !dbg !14
  %address.local595 = and i32 %address.local594, 60, !dbg !14
  %address.local596 = or disjoint i32 %address.local593, %address.local595, !dbg !14
  %address.local597 = shl nuw nsw i32 %address.thread, 5, !dbg !14
  %address.local598 = and i32 %address.local597, 1536, !dbg !14
  %address.local599 = or disjoint i32 %address.local596, %address.local598, !dbg !14
  %address.local600 = zext nneg i32 %address.local599 to i64, !dbg !14
  %address.local601 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local600, !dbg !14
  %136 = load <4 x half>, ptr addrspace(3) %address.local601, align 2, !dbg !14
  %address.local602 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local600, !dbg !14
  %address.local603 = getelementptr i8, ptr addrspace(3) %address.local602, i64 4096, !dbg !14
  %137 = load <4 x half>, ptr addrspace(3) %address.local603, align 2, !dbg !14
  %address.local604 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local600, !dbg !14
  %address.local605 = getelementptr i8, ptr addrspace(3) %address.local604, i64 512, !dbg !14
  %138 = load <4 x half>, ptr addrspace(3) %address.local605, align 2, !dbg !14
  %address.local606 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local600, !dbg !14
  %address.local607 = getelementptr i8, ptr addrspace(3) %address.local606, i64 4608, !dbg !14
  %139 = load <4 x half>, ptr addrspace(3) %address.local607, align 2, !dbg !14
  %140 = fmul <4 x half> %136, splat (half 0xH3015), !dbg !14
  %141 = fmul <4 x half> %137, splat (half 0xH3015), !dbg !14
  %142 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !14
  %143 = fmul <4 x half> %139, splat (half 0xH3015), !dbg !14
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %.neg237) #8, !dbg !8, !frisk.register_lifetimes !1
  %address.local608 = shl i64 %address.base, 6, !dbg !8
  %address.base609 = call i64 asm sideeffect "", "=v,0"(i64 %.neg239) #8, !dbg !8, !frisk.register_lifetimes !1
  %address.local610 = shl i64 %address.base609, 5, !dbg !8
  %address.local611 = add i64 %address.local608, %address.local610, !dbg !8
  %address.local612 = mul nuw nsw i64 %address.local, 400, !dbg !8
  %address.local613 = lshr i64 %address.local, 2, !dbg !8
  %address.local614 = mul nsw i64 %address.local613, -1664, !dbg !8
  %address.local615 = add nsw i64 %address.local612, %address.local614, !dbg !8
  %address.local616 = and i64 %address.local, 15, !dbg !8
  %address.local617 = lshr i64 %address.local616, 2, !dbg !8
  %address.local618 = shl nuw nsw i64 %address.local617, 3, !dbg !8
  %address.local619 = add nsw i64 %address.local615, %address.local618, !dbg !8
  %address.local620 = lshr i64 %address.local, 1, !dbg !9
  %address.local621 = shl nuw nsw i64 %address.local620, 5, !dbg !8
  %address.local622 = add nsw i64 %address.local619, %address.local621, !dbg !8
  %address.local623 = lshr i64 %address.local, 4, !dbg !14
  %address.local624 = and i64 %address.local623, 3, !dbg !14
  %address.local625 = shl nuw nsw i64 %address.local624, 10, !dbg !8
  %address.local626 = add nsw i64 %address.local622, %address.local625, !dbg !8
  %address.local627 = and i32 %address.thread, 32, !dbg !8
  %address.local628 = zext nneg i32 %address.local627 to i64, !dbg !8
  %address.local629 = add i64 %address.local626, %address.local628, !dbg !8
  %address.local630 = add i64 %address.local611, %address.local629, !dbg !8
  %address.local631 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local630, !dbg !8
  %144 = load <8 x half>, ptr addrspace(3) %address.local631, align 2, !dbg !8
  %145 = extractvalue [2 x <32 x float>] %61, 0, !dbg !31
  %146 = shufflevector <32 x float> %145, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %147 = shufflevector <8 x half> %144, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %148 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %147, <4 x float> %146) #9, !dbg !8
  %address.local632 = shl nuw nsw i64 %address.local, 2, !dbg !14
  %address.local633 = mul nsw i64 %address.local613, -16, !dbg !8
  %address.local634 = add nsw i64 %address.local632, %address.local633, !dbg !8
  %address.local635 = lshr i32 %address.thread, 1, !dbg !8
  %address.local636 = and i32 %address.local635, 1, !dbg !8
  %address.local637 = zext nneg i32 %address.local636 to i64, !dbg !8
  %address.local638 = or disjoint i64 %address.local634, %address.local637, !dbg !8
  %address.local639 = lshr i64 %address.local624, 1, !dbg !8
  %address.local640 = add nuw nsw i64 %address.local638, %address.local639, !dbg !8
  %address.local641 = add nsw i64 %address.local640, 1, !dbg !8
  %address.local642 = sdiv i64 %address.local641, -2, !dbg !8
  %address.local643 = shl i64 %address.local642, 7, !dbg !8
  %address.local644 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local643, !dbg !8
  %address.local645 = shl nuw nsw i64 %address.local, 4, !dbg !8
  %address.local646 = mul nsw i64 %address.local613, -64, !dbg !8
  %address.local647 = add nsw i64 %address.local645, %address.local646, !dbg !8
  %address.local648 = or disjoint i64 %address.local647, %address.local617, !dbg !8
  %address.local649 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local650 = and i32 %address.local649, 6, !dbg !8
  %address.local651 = zext nneg i32 %address.local650 to i64, !dbg !8
  %address.local652 = add nuw nsw i64 %address.local648, %address.local651, !dbg !8
  %address.local653 = sdiv i64 %address.local652, -4, !dbg !8
  %address.local654 = shl i64 %address.local653, 6, !dbg !8
  %address.local655 = getelementptr i8, ptr addrspace(3) %address.local644, i64 %address.local654, !dbg !8
  %address.local656 = mul nuw nsw i64 %address.local651, 144, !dbg !8
  %address.local657 = getelementptr i8, ptr addrspace(3) %address.local655, i64 %address.local656, !dbg !8
  %address.local658 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local659 = and i32 %address.local658, 24, !dbg !8
  %address.local660 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local661 = and i32 %address.local660, 768, !dbg !8
  %address.local662 = or disjoint i32 %address.local659, %address.local661, !dbg !8
  %address.local663 = zext nneg i32 %address.local662 to i64, !dbg !8
  %address.local664 = getelementptr [2 x i8], ptr addrspace(3) %address.local657, i64 %address.local663, !dbg !8
  %address.local665 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local666 = and i32 %address.local665, 32, !dbg !8
  %address.local667 = zext nneg i32 %address.local666 to i64, !dbg !8
  %address.local668 = getelementptr [2 x i8], ptr addrspace(3) %address.local664, i64 %address.local667, !dbg !8
  %address.local669 = getelementptr [2 x i8], ptr addrspace(3) %address.local668, i64 %address.local625, !dbg !8
  %address.local670 = getelementptr [2 x i8], ptr addrspace(3) %address.local669, i64 %address.local628, !dbg !8
  %address.local671 = getelementptr i8, ptr addrspace(3) %address.local670, i64 64, !dbg !8
  %149 = load <8 x half>, ptr addrspace(3) %address.local671, align 2, !dbg !8
  %150 = shufflevector <8 x half> %144, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %151 = shufflevector <32 x float> %145, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %152 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %150, <4 x float> %148) #9, !dbg !8
  %153 = shufflevector <8 x half> %149, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %154 = shufflevector <8 x half> %149, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %155 = shufflevector <32 x float> %145, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %156 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %153, <4 x float> %151) #9, !dbg !8
  %157 = shufflevector <32 x float> %145, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %158 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %154, <4 x float> %156) #9, !dbg !8
  %159 = shufflevector <32 x float> %145, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %160 = shufflevector <32 x float> %145, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %161 = shufflevector <32 x float> %145, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %162 = shufflevector <32 x float> %145, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %163 = extractvalue [2 x <32 x float>] %61, 1, !dbg !31
  %164 = shufflevector <32 x float> %163, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %165 = shufflevector <32 x float> %163, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %166 = shufflevector <32 x float> %163, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %167 = shufflevector <32 x float> %163, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %168 = shufflevector <32 x float> %163, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %169 = shufflevector <32 x float> %163, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %170 = shufflevector <32 x float> %163, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %171 = shufflevector <32 x float> %163, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %address.local672 = mul nsw i64 %address.local613, -18, !dbg !8
  %address.local673 = add nsw i64 %address.local632, %address.local672, !dbg !8
  %address.local674 = add nsw i64 %address.local673, %address.local620, !dbg !8
  %address.local675 = add nsw i64 %address.local674, %address.local639, !dbg !8
  %address.local676 = sdiv i64 %address.local675, -2, !dbg !8
  %address.local677 = shl i64 %address.local676, 7, !dbg !8
  %address.local678 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local677, !dbg !8
  %address.local679 = mul nuw nsw i64 %address.local, 18, !dbg !8
  %address.local680 = mul nsw i64 %address.local613, -72, !dbg !8
  %address.local681 = add nsw i64 %address.local679, %address.local680, !dbg !8
  %address.local682 = add nsw i64 %address.local681, %address.local617, !dbg !8
  %address.local683 = add nsw i64 %address.local682, 1, !dbg !8
  %address.local684 = sdiv i64 %address.local683, -4, !dbg !8
  %address.local685 = shl i64 %address.local684, 6, !dbg !8
  %address.local686 = getelementptr i8, ptr addrspace(3) %address.local678, i64 %address.local685, !dbg !8
  %address.local687 = getelementptr [2 x i8], ptr addrspace(3) %address.local686, i64 %address.local612, !dbg !8
  %address.local688 = getelementptr [2 x i8], ptr addrspace(3) %address.local687, i64 %address.local614, !dbg !8
  %address.local689 = getelementptr [2 x i8], ptr addrspace(3) %address.local688, i64 %address.local618, !dbg !8
  %address.local690 = getelementptr [2 x i8], ptr addrspace(3) %address.local689, i64 %address.local621, !dbg !8
  %address.local691 = getelementptr [2 x i8], ptr addrspace(3) %address.local690, i64 %address.local625, !dbg !8
  %address.local692 = getelementptr [2 x i8], ptr addrspace(3) %address.local691, i64 %address.local628, !dbg !8
  %address.local693 = getelementptr i8, ptr addrspace(3) %address.local692, i64 144, !dbg !8
  %172 = load <8 x half>, ptr addrspace(3) %address.local693, align 2, !dbg !8
  %address.local694 = add nsw i64 %address.local632, %address.local633, !dbg !8
  %address.local695 = lshr i32 %address.thread, 1, !dbg !8
  %address.local696 = and i32 %address.local695, 1, !dbg !8
  %address.local697 = zext nneg i32 %address.local696 to i64, !dbg !8
  %address.local698 = or disjoint i64 %address.local694, %address.local697, !dbg !8
  %address.local699 = add nuw nsw i64 %address.local698, %address.local639, !dbg !8
  %address.local700 = add nsw i64 %address.local699, 3, !dbg !8
  %address.local701 = sdiv i64 %address.local700, -2, !dbg !8
  %address.local702 = shl i64 %address.local701, 7, !dbg !8
  %address.local703 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local702, !dbg !8
  %address.local704 = add nsw i64 %address.local645, %address.local646, !dbg !8
  %address.local705 = or disjoint i64 %address.local704, %address.local617, !dbg !8
  %address.local706 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local707 = and i32 %address.local706, 6, !dbg !8
  %address.local708 = or disjoint i32 %address.local707, 1, !dbg !8
  %address.local709 = zext nneg i32 %address.local708 to i64, !dbg !8
  %address.local710 = add nuw nsw i64 %address.local705, %address.local709, !dbg !8
  %address.local711 = sdiv i64 %address.local710, -4, !dbg !8
  %address.local712 = shl i64 %address.local711, 6, !dbg !8
  %address.local713 = getelementptr i8, ptr addrspace(3) %address.local703, i64 %address.local712, !dbg !8
  %address.local714 = mul nuw nsw i64 %address.local709, 144, !dbg !8
  %address.local715 = getelementptr i8, ptr addrspace(3) %address.local713, i64 %address.local714, !dbg !8
  %address.local716 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local717 = and i32 %address.local716, 24, !dbg !8
  %address.local718 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local719 = and i32 %address.local718, 768, !dbg !8
  %address.local720 = or disjoint i32 %address.local717, %address.local719, !dbg !8
  %address.local721 = zext nneg i32 %address.local720 to i64, !dbg !8
  %address.local722 = getelementptr [2 x i8], ptr addrspace(3) %address.local715, i64 %address.local721, !dbg !8
  %address.local723 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local724 = and i32 %address.local723, 32, !dbg !8
  %address.local725 = zext nneg i32 %address.local724 to i64, !dbg !8
  %address.local726 = getelementptr [2 x i8], ptr addrspace(3) %address.local722, i64 %address.local725, !dbg !8
  %address.local727 = getelementptr [2 x i8], ptr addrspace(3) %address.local726, i64 %address.local625, !dbg !8
  %address.local728 = getelementptr [2 x i8], ptr addrspace(3) %address.local727, i64 %address.local628, !dbg !8
  %address.local729 = getelementptr i8, ptr addrspace(3) %address.local728, i64 192, !dbg !8
  %173 = load <8 x half>, ptr addrspace(3) %address.local729, align 2, !dbg !8
  %address.local730 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local630, !dbg !8
  %address.local731 = getelementptr i8, ptr addrspace(3) %address.local730, i64 1024, !dbg !8
  %174 = load <8 x half>, ptr addrspace(3) %address.local731, align 2, !dbg !8
  %175 = shufflevector <8 x half> %172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %176 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %175, <4 x float> %155) #9, !dbg !8
  %177 = shufflevector <8 x half> %172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %178 = shufflevector <8 x half> %173, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %179 = shufflevector <8 x half> %173, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %180 = shufflevector <8 x half> %174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %181 = shufflevector <8 x half> %174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %address.local732 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local733 = and i32 %address.local732, 6, !dbg !8
  %address.local734 = or disjoint i32 %address.local733, 8, !dbg !8
  %address.local735 = mul nuw nsw i32 %address.local734, 72, !dbg !8
  %address.local736 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local737 = and i32 %address.local736, 24, !dbg !8
  %address.local738 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local739 = and i32 %address.local738, 768, !dbg !8
  %address.local740 = or disjoint i32 %address.local737, %address.local739, !dbg !8
  %address.local741 = or disjoint i32 %address.local740, 1024, !dbg !8
  %address.local742 = add nuw nsw i32 %address.local735, %address.local741, !dbg !8
  %address.local743 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local744 = and i32 %address.local743, 32, !dbg !8
  %address.local745 = or disjoint i32 %address.local744, 64, !dbg !8
  %address.local746 = add nuw nsw i32 %address.local742, %address.local745, !dbg !8
  %address.local747 = zext nneg i32 %address.local746 to i64, !dbg !8
  %address.local748 = add nuw nsw i64 %address.local625, %address.local747, !dbg !8
  %address.local749 = add nuw nsw i64 %address.local748, %address.local628, !dbg !8
  %address.local750 = shl nuw nsw i32 %address.thread, 2, !dbg !8
  %address.local751 = and i32 %address.local750, 12, !dbg !8
  %address.local752 = lshr i32 %address.thread, 1, !dbg !8
  %address.local753 = and i32 %address.local752, 1, !dbg !8
  %address.local754 = or disjoint i32 %address.local751, %address.local753, !dbg !8
  %address.local755 = or disjoint i32 %address.local754, 18, !dbg !8
  %address.local756 = zext nneg i32 %address.local755 to i64, !dbg !8
  %address.local757 = add nuw nsw i64 %address.local639, %address.local756, !dbg !8
  %address.local758 = shl nuw nsw i64 %address.local757, 5, !dbg !8
  %address.local759 = add nuw nsw i64 %address.local758, 32, !dbg !8
  %address.local760 = and i64 %address.local759, 274877906880, !dbg !8
  %address.local761 = lshr i32 %address.thread, 2, !dbg !8
  %address.local762 = and i32 %address.local761, 2, !dbg !8
  %address.local763 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local764 = and i32 %address.local763, 48, !dbg !8
  %address.local765 = or disjoint i32 %address.local762, %address.local764, !dbg !8
  %address.local766 = or disjoint i32 %address.local765, 64, !dbg !8
  %address.local767 = add nuw nsw i32 %address.local766, %address.local734, !dbg !8
  %address.local768 = shl nuw nsw i32 %address.local767, 3, !dbg !8
  %address.local769 = and i32 %address.local768, 2016, !dbg !8
  %address.local770 = zext nneg i32 %address.local769 to i64, !dbg !8
  %address.local771 = add nuw nsw i64 %address.local760, %address.local770, !dbg !8
  %address.local772 = sub nsw i64 %address.local749, %address.local771, !dbg !8
  %address.local773 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local772, !dbg !8
  %address.local774 = getelementptr i8, ptr addrspace(3) %address.local773, i64 64, !dbg !8
  %182 = load <8 x half>, ptr addrspace(3) %address.local774, align 2, !dbg !8
  %183 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %177, <4 x float> %176) #9, !dbg !8
  %address.local775 = add nuw nsw i64 %address.local632, 18, !dbg !8
  %address.local776 = add nsw i64 %address.local775, %address.local672, !dbg !8
  %address.local777 = add nsw i64 %address.local776, %address.local620, !dbg !8
  %address.local778 = add nsw i64 %address.local777, %address.local639, !dbg !8
  %address.local779 = sdiv i64 %address.local778, -2, !dbg !8
  %address.local780 = shl i64 %address.local779, 7, !dbg !8
  %address.local781 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local780, !dbg !8
  %address.local782 = add nuw nsw i64 %address.local679, 72, !dbg !8
  %address.local783 = add nsw i64 %address.local782, %address.local680, !dbg !8
  %address.local784 = add nsw i64 %address.local783, %address.local617, !dbg !8
  %address.local785 = add nsw i64 %address.local784, -15, !dbg !8
  %address.local786 = sdiv i64 %address.local785, -4, !dbg !8
  %address.local787 = shl i64 %address.local786, 6, !dbg !8
  %address.local788 = getelementptr i8, ptr addrspace(3) %address.local781, i64 %address.local787, !dbg !8
  %address.local789 = getelementptr [2 x i8], ptr addrspace(3) %address.local788, i64 %address.local612, !dbg !8
  %address.local790 = getelementptr [2 x i8], ptr addrspace(3) %address.local789, i64 %address.local614, !dbg !8
  %address.local791 = getelementptr [2 x i8], ptr addrspace(3) %address.local790, i64 %address.local618, !dbg !8
  %address.local792 = getelementptr [2 x i8], ptr addrspace(3) %address.local791, i64 %address.local621, !dbg !8
  %address.local793 = getelementptr [2 x i8], ptr addrspace(3) %address.local792, i64 %address.local625, !dbg !8
  %address.local794 = getelementptr [2 x i8], ptr addrspace(3) %address.local793, i64 %address.local628, !dbg !8
  %address.local795 = getelementptr i8, ptr addrspace(3) %address.local794, i64 3216, !dbg !8
  %184 = load <8 x half>, ptr addrspace(3) %address.local795, align 2, !dbg !8
  %address.local796 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local797 = and i32 %address.local796, 6, !dbg !8
  %address.local798 = or disjoint i32 %address.local797, 9, !dbg !8
  %address.local799 = mul nuw nsw i32 %address.local798, 72, !dbg !8
  %address.local800 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local801 = and i32 %address.local800, 24, !dbg !8
  %address.local802 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local803 = and i32 %address.local802, 768, !dbg !8
  %address.local804 = or disjoint i32 %address.local801, %address.local803, !dbg !8
  %address.local805 = or disjoint i32 %address.local804, 1024, !dbg !8
  %address.local806 = add nuw nsw i32 %address.local799, %address.local805, !dbg !8
  %address.local807 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local808 = and i32 %address.local807, 32, !dbg !8
  %address.local809 = or disjoint i32 %address.local808, 64, !dbg !8
  %address.local810 = add nuw nsw i32 %address.local806, %address.local809, !dbg !8
  %address.local811 = zext nneg i32 %address.local810 to i64, !dbg !8
  %address.local812 = add nuw nsw i64 %address.local625, %address.local811, !dbg !8
  %address.local813 = add nuw nsw i64 %address.local812, %address.local628, !dbg !8
  %address.local814 = shl nuw nsw i32 %address.thread, 2, !dbg !8
  %address.local815 = and i32 %address.local814, 12, !dbg !8
  %address.local816 = lshr i32 %address.thread, 1, !dbg !8
  %address.local817 = and i32 %address.local816, 1, !dbg !8
  %address.local818 = or disjoint i32 %address.local815, %address.local817, !dbg !8
  %address.local819 = or disjoint i32 %address.local818, 18, !dbg !8
  %address.local820 = zext nneg i32 %address.local819 to i64, !dbg !8
  %address.local821 = add nuw nsw i64 %address.local639, %address.local820, !dbg !8
  %address.local822 = shl nuw nsw i64 %address.local821, 5, !dbg !8
  %address.local823 = add nuw nsw i64 %address.local822, 96, !dbg !8
  %address.local824 = and i64 %address.local823, 274877906880, !dbg !8
  %address.local825 = lshr i32 %address.thread, 2, !dbg !8
  %address.local826 = and i32 %address.local825, 3, !dbg !8
  %address.local827 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local828 = and i32 %address.local827, 48, !dbg !8
  %address.local829 = or disjoint i32 %address.local826, %address.local828, !dbg !8
  %address.local830 = or disjoint i32 %address.local829, 64, !dbg !8
  %address.local831 = add nuw nsw i32 %address.local830, %address.local798, !dbg !8
  %address.local832 = shl nuw nsw i32 %address.local831, 3, !dbg !8
  %address.local833 = and i32 %address.local832, 2016, !dbg !8
  %address.local834 = zext nneg i32 %address.local833 to i64, !dbg !8
  %address.local835 = add nuw nsw i64 %address.local824, %address.local834, !dbg !8
  %address.local836 = sub nsw i64 %address.local813, %address.local835, !dbg !8
  %address.local837 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local836, !dbg !8
  %address.local838 = getelementptr i8, ptr addrspace(3) %address.local837, i64 192, !dbg !8
  %185 = load <8 x half>, ptr addrspace(3) %address.local838, align 2, !dbg !8
  %186 = shufflevector <8 x half> %182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %187 = shufflevector <4 x float> %183, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %188 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %178, <4 x float> %157) #9, !dbg !8
  %189 = shufflevector <8 x half> %182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %190 = shufflevector <8 x half> %184, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %191 = shufflevector <8 x half> %184, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %192 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %193 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %194 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %179, <4 x float> %188) #9, !dbg !8
  %195 = shufflevector <4 x float> %194, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %196 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %180, <4 x float> %159) #9, !dbg !8
  %197 = shufflevector <8 x half> %144, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %198 = shufflevector <8 x half> %144, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %199 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %181, <4 x float> %196) #9, !dbg !8
  %200 = shufflevector <8 x half> %149, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %201 = shufflevector <8 x half> %149, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %202 = load <8 x half>, ptr addrspace(3) %address.local693, align 2, !dbg !8
  %203 = shufflevector <4 x float> %199, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %204 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %186, <4 x float> %160) #9, !dbg !8
  %205 = load <8 x half>, ptr addrspace(3) %address.local729, align 2, !dbg !8
  %206 = shufflevector <8 x half> %202, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %207 = shufflevector <8 x half> %202, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %208 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %189, <4 x float> %204) #9, !dbg !8
  %209 = shufflevector <8 x half> %205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %210 = shufflevector <8 x half> %205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %211 = load <8 x half>, ptr addrspace(3) %address.local731, align 2, !dbg !8
  %212 = shufflevector <4 x float> %208, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %213 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %190, <4 x float> %161) #9, !dbg !8
  %214 = load <8 x half>, ptr addrspace(3) %address.local774, align 2, !dbg !8
  %215 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %216 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %217 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %191, <4 x float> %213) #9, !dbg !8
  %218 = shufflevector <8 x half> %214, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %219 = shufflevector <8 x half> %214, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %220 = load <8 x half>, ptr addrspace(3) %address.local795, align 2, !dbg !8
  %221 = shufflevector <4 x float> %217, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %222 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %192, <4 x float> %162) #9, !dbg !8
  %223 = load <8 x half>, ptr addrspace(3) %address.local838, align 2, !dbg !8
  %224 = shufflevector <8 x half> %220, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %225 = shufflevector <8 x half> %220, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %226 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %193, <4 x float> %222) #9, !dbg !8
  %227 = shufflevector <8 x half> %223, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %228 = shufflevector <8 x half> %223, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %229 = shufflevector <4 x float> %226, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %230 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %197, <4 x float> %164) #9, !dbg !8
  %231 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %198, <4 x float> %230) #9, !dbg !8
  %232 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %200, <4 x float> %165) #9, !dbg !8
  %233 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %201, <4 x float> %232) #9, !dbg !8
  %234 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %206, <4 x float> %166) #9, !dbg !8
  %235 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %207, <4 x float> %234) #9, !dbg !8
  %236 = shufflevector <4 x float> %235, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %237 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %209, <4 x float> %167) #9, !dbg !8
  %238 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %210, <4 x float> %237) #9, !dbg !8
  %239 = shufflevector <4 x float> %238, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %240 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %215, <4 x float> %168) #9, !dbg !8
  %241 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %216, <4 x float> %240) #9, !dbg !8
  %242 = shufflevector <4 x float> %241, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %243 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %218, <4 x float> %169) #9, !dbg !8
  %244 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %219, <4 x float> %243) #9, !dbg !8
  %245 = shufflevector <4 x float> %244, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %246 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %224, <4 x float> %170) #9, !dbg !8
  %247 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %225, <4 x float> %246) #9, !dbg !8
  %248 = shufflevector <4 x float> %247, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %249 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %227, <4 x float> %171) #9, !dbg !8
  %250 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %228, <4 x float> %249) #9, !dbg !8
  %251 = shufflevector <4 x float> %250, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %252 = shufflevector <4 x float> %152, <4 x float> %158, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %253 = shufflevector <32 x float> %252, <32 x float> %187, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %254 = shufflevector <32 x float> %253, <32 x float> %195, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %255 = shufflevector <32 x float> %254, <32 x float> %203, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %256 = shufflevector <32 x float> %255, <32 x float> %212, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %257 = shufflevector <32 x float> %256, <32 x float> %221, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %258 = shufflevector <32 x float> %257, <32 x float> %229, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %259 = insertvalue [2 x <32 x float>] poison, <32 x float> %258, 0, !dbg !22
  %260 = shufflevector <4 x float> %231, <4 x float> %233, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %261 = shufflevector <32 x float> %260, <32 x float> %236, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %262 = shufflevector <32 x float> %261, <32 x float> %239, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %263 = shufflevector <32 x float> %262, <32 x float> %242, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %264 = shufflevector <32 x float> %263, <32 x float> %245, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %265 = shufflevector <32 x float> %264, <32 x float> %248, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %266 = shufflevector <32 x float> %265, <32 x float> %251, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %267 = insertvalue [2 x <32 x float>] %259, <32 x float> %266, 1, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %268 = add nuw nsw i64 %60, 1, !dbg !22
  br label %59, !dbg !22

269:                                              ; preds = %59
  %270 = or disjoint i64 %55, 127, !dbg !15
  %271 = icmp uge i64 %9, %270, !dbg !15
  %272 = and i1 %27, %271, !dbg !15
  br i1 %272, label %273, label %274, !dbg !15

273:                                              ; preds = %269
  br label %445, !dbg !15

274:                                              ; preds = %269
  %275 = add nsw i64 %9, %20, !dbg !15
  %276 = shl nuw nsw i64 %22, 4, !dbg !15
  %277 = add nsw i64 %275, %276, !dbg !15
  %278 = and i32 %19, 240, !dbg !15
  %index.compact = zext nneg i32 %278 to i64, !dbg !15
  %279 = sub nsw i64 %277, %index.compact, !dbg !15
  %280 = shl nuw nsw i64 %24, 4, !dbg !15
  %281 = or disjoint i64 %55, %280, !dbg !15
  %282 = add nsw i64 %279, 1, !dbg !23
  %.not40 = icmp ugt i64 %282, %281, !dbg !24
  %283 = select i1 %.not40, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %284 = or disjoint i64 %281, 1, !dbg !15
  %.not41 = icmp ugt i64 %282, %284, !dbg !24
  %285 = select i1 %.not41, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %286 = or disjoint i64 %281, 2, !dbg !15
  %.not42 = icmp ugt i64 %282, %286, !dbg !24
  %287 = select i1 %.not42, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %288 = or disjoint i64 %281, 3, !dbg !15
  %.not43 = icmp ugt i64 %282, %288, !dbg !24
  %289 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %290 = or disjoint i64 %281, 4, !dbg !15
  %.not44 = icmp ugt i64 %282, %290, !dbg !24
  %291 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %292 = or disjoint i64 %281, 5, !dbg !15
  %.not45 = icmp ugt i64 %282, %292, !dbg !24
  %293 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %294 = or disjoint i64 %281, 6, !dbg !15
  %.not46 = icmp ugt i64 %282, %294, !dbg !24
  %295 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %296 = or disjoint i64 %281, 7, !dbg !15
  %.not47 = icmp ugt i64 %282, %296, !dbg !24
  %297 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %298 = or disjoint i64 %281, 8, !dbg !15
  %.not48 = icmp ugt i64 %282, %298, !dbg !24
  %299 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %300 = or disjoint i64 %281, 9, !dbg !15
  %.not49 = icmp ugt i64 %282, %300, !dbg !24
  %301 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %302 = or disjoint i64 %281, 10, !dbg !15
  %.not50 = icmp ugt i64 %282, %302, !dbg !24
  %303 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %304 = or disjoint i64 %281, 11, !dbg !15
  %.not51 = icmp ugt i64 %282, %304, !dbg !24
  %305 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %306 = or disjoint i64 %281, 12, !dbg !15
  %.not52 = icmp ugt i64 %282, %306, !dbg !24
  %307 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %308 = or disjoint i64 %281, 13, !dbg !15
  %.not53 = icmp ugt i64 %282, %308, !dbg !24
  %309 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %310 = or disjoint i64 %281, 14, !dbg !15
  %.not54 = icmp ugt i64 %282, %310, !dbg !24
  %311 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %312 = or disjoint i64 %281, 15, !dbg !15
  %.not55 = icmp ugt i64 %282, %312, !dbg !24
  %313 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %314 = or disjoint i64 %281, 64, !dbg !15
  %.not56 = icmp ugt i64 %282, %314, !dbg !24
  %315 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %316 = or disjoint i64 %281, 65, !dbg !15
  %.not57 = icmp ugt i64 %282, %316, !dbg !24
  %317 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %318 = or disjoint i64 %281, 66, !dbg !15
  %.not58 = icmp ugt i64 %282, %318, !dbg !24
  %319 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %320 = or disjoint i64 %281, 67, !dbg !15
  %.not59 = icmp ugt i64 %282, %320, !dbg !24
  %321 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %322 = or disjoint i64 %281, 68, !dbg !15
  %.not60 = icmp ugt i64 %282, %322, !dbg !24
  %323 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %324 = or disjoint i64 %281, 69, !dbg !15
  %.not61 = icmp ugt i64 %282, %324, !dbg !24
  %325 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %326 = or disjoint i64 %281, 70, !dbg !15
  %.not62 = icmp ugt i64 %282, %326, !dbg !24
  %327 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %328 = or disjoint i64 %281, 71, !dbg !15
  %.not63 = icmp ugt i64 %282, %328, !dbg !24
  %329 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %330 = or disjoint i64 %281, 72, !dbg !15
  %.not64 = icmp ugt i64 %282, %330, !dbg !24
  %331 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %332 = or disjoint i64 %281, 73, !dbg !15
  %.not65 = icmp ugt i64 %282, %332, !dbg !24
  %333 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %334 = or disjoint i64 %281, 74, !dbg !15
  %.not66 = icmp ugt i64 %282, %334, !dbg !24
  %335 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %336 = or disjoint i64 %281, 75, !dbg !15
  %.not67 = icmp ugt i64 %282, %336, !dbg !24
  %337 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %338 = or disjoint i64 %281, 76, !dbg !15
  %.not68 = icmp ugt i64 %282, %338, !dbg !24
  %339 = select i1 %.not68, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %340 = or disjoint i64 %281, 77, !dbg !15
  %.not69 = icmp ugt i64 %282, %340, !dbg !24
  %341 = select i1 %.not69, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %342 = or disjoint i64 %281, 78, !dbg !15
  %.not70 = icmp ugt i64 %282, %342, !dbg !24
  %343 = select i1 %.not70, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %344 = or disjoint i64 %281, 79, !dbg !15
  %.not71 = icmp ugt i64 %282, %344, !dbg !24
  %345 = select i1 %.not71, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %346 = add nsw i64 %279, 65, !dbg !23
  %.not72 = icmp ugt i64 %346, %281, !dbg !24
  %347 = select i1 %.not72, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not73 = icmp ugt i64 %346, %284, !dbg !24
  %348 = select i1 %.not73, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not74 = icmp ugt i64 %346, %286, !dbg !24
  %349 = select i1 %.not74, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not75 = icmp ugt i64 %346, %288, !dbg !24
  %350 = select i1 %.not75, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not76 = icmp ugt i64 %346, %290, !dbg !24
  %351 = select i1 %.not76, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not77 = icmp ugt i64 %346, %292, !dbg !24
  %352 = select i1 %.not77, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not78 = icmp ugt i64 %346, %294, !dbg !24
  %353 = select i1 %.not78, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not79 = icmp ugt i64 %346, %296, !dbg !24
  %354 = select i1 %.not79, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not80 = icmp ugt i64 %346, %298, !dbg !24
  %355 = select i1 %.not80, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not81 = icmp ugt i64 %346, %300, !dbg !24
  %356 = select i1 %.not81, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not82 = icmp ugt i64 %346, %302, !dbg !24
  %357 = select i1 %.not82, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not83 = icmp ugt i64 %346, %304, !dbg !24
  %358 = select i1 %.not83, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not84 = icmp ugt i64 %346, %306, !dbg !24
  %359 = select i1 %.not84, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not85 = icmp ugt i64 %346, %308, !dbg !24
  %360 = select i1 %.not85, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not86 = icmp ugt i64 %346, %310, !dbg !24
  %361 = select i1 %.not86, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not87 = icmp ugt i64 %346, %312, !dbg !24
  %362 = select i1 %.not87, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not88 = icmp ugt i64 %346, %314, !dbg !24
  %363 = select i1 %.not88, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not89 = icmp ugt i64 %346, %316, !dbg !24
  %364 = select i1 %.not89, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not90 = icmp ugt i64 %346, %318, !dbg !24
  %365 = select i1 %.not90, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not91 = icmp ugt i64 %346, %320, !dbg !24
  %366 = select i1 %.not91, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not92 = icmp ugt i64 %346, %322, !dbg !24
  %367 = select i1 %.not92, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not93 = icmp ugt i64 %346, %324, !dbg !24
  %368 = select i1 %.not93, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not94 = icmp ugt i64 %346, %326, !dbg !24
  %369 = select i1 %.not94, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not95 = icmp ugt i64 %346, %328, !dbg !24
  %370 = select i1 %.not95, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not96 = icmp ugt i64 %346, %330, !dbg !24
  %371 = select i1 %.not96, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not97 = icmp ugt i64 %346, %332, !dbg !24
  %372 = select i1 %.not97, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not98 = icmp ugt i64 %346, %334, !dbg !24
  %373 = select i1 %.not98, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not99 = icmp ugt i64 %346, %336, !dbg !24
  %374 = select i1 %.not99, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not100 = icmp ugt i64 %346, %338, !dbg !24
  %375 = select i1 %.not100, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not101 = icmp ugt i64 %346, %340, !dbg !24
  %376 = select i1 %.not101, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not102 = icmp ugt i64 %346, %342, !dbg !24
  %377 = select i1 %.not102, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not103 = icmp ugt i64 %346, %344, !dbg !24
  %378 = select i1 %.not103, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %379 = insertelement <32 x float> poison, float %283, i64 0, !dbg !15
  %380 = insertelement <32 x float> %379, float %285, i64 1, !dbg !15
  %381 = insertelement <32 x float> %380, float %287, i64 2, !dbg !15
  %382 = insertelement <32 x float> %381, float %289, i64 3, !dbg !15
  %383 = insertelement <32 x float> %382, float %291, i64 4, !dbg !15
  %384 = insertelement <32 x float> %383, float %293, i64 5, !dbg !15
  %385 = insertelement <32 x float> %384, float %295, i64 6, !dbg !15
  %386 = insertelement <32 x float> %385, float %297, i64 7, !dbg !15
  %387 = insertelement <32 x float> %386, float %299, i64 8, !dbg !15
  %388 = insertelement <32 x float> %387, float %301, i64 9, !dbg !15
  %389 = insertelement <32 x float> %388, float %303, i64 10, !dbg !15
  %390 = insertelement <32 x float> %389, float %305, i64 11, !dbg !15
  %391 = insertelement <32 x float> %390, float %307, i64 12, !dbg !15
  %392 = insertelement <32 x float> %391, float %309, i64 13, !dbg !15
  %393 = insertelement <32 x float> %392, float %311, i64 14, !dbg !15
  %394 = insertelement <32 x float> %393, float %313, i64 15, !dbg !15
  %395 = insertelement <32 x float> %394, float %315, i64 16, !dbg !15
  %396 = insertelement <32 x float> %395, float %317, i64 17, !dbg !15
  %397 = insertelement <32 x float> %396, float %319, i64 18, !dbg !15
  %398 = insertelement <32 x float> %397, float %321, i64 19, !dbg !15
  %399 = insertelement <32 x float> %398, float %323, i64 20, !dbg !15
  %400 = insertelement <32 x float> %399, float %325, i64 21, !dbg !15
  %401 = insertelement <32 x float> %400, float %327, i64 22, !dbg !15
  %402 = insertelement <32 x float> %401, float %329, i64 23, !dbg !15
  %403 = insertelement <32 x float> %402, float %331, i64 24, !dbg !15
  %404 = insertelement <32 x float> %403, float %333, i64 25, !dbg !15
  %405 = insertelement <32 x float> %404, float %335, i64 26, !dbg !15
  %406 = insertelement <32 x float> %405, float %337, i64 27, !dbg !15
  %407 = insertelement <32 x float> %406, float %339, i64 28, !dbg !15
  %408 = insertelement <32 x float> %407, float %341, i64 29, !dbg !15
  %409 = insertelement <32 x float> %408, float %343, i64 30, !dbg !15
  %410 = insertelement <32 x float> %409, float %345, i64 31, !dbg !15
  %411 = insertvalue [2 x <32 x float>] poison, <32 x float> %410, 0, !dbg !15
  %412 = insertelement <32 x float> poison, float %347, i64 0, !dbg !15
  %413 = insertelement <32 x float> %412, float %348, i64 1, !dbg !15
  %414 = insertelement <32 x float> %413, float %349, i64 2, !dbg !15
  %415 = insertelement <32 x float> %414, float %350, i64 3, !dbg !15
  %416 = insertelement <32 x float> %415, float %351, i64 4, !dbg !15
  %417 = insertelement <32 x float> %416, float %352, i64 5, !dbg !15
  %418 = insertelement <32 x float> %417, float %353, i64 6, !dbg !15
  %419 = insertelement <32 x float> %418, float %354, i64 7, !dbg !15
  %420 = insertelement <32 x float> %419, float %355, i64 8, !dbg !15
  %421 = insertelement <32 x float> %420, float %356, i64 9, !dbg !15
  %422 = insertelement <32 x float> %421, float %357, i64 10, !dbg !15
  %423 = insertelement <32 x float> %422, float %358, i64 11, !dbg !15
  %424 = insertelement <32 x float> %423, float %359, i64 12, !dbg !15
  %425 = insertelement <32 x float> %424, float %360, i64 13, !dbg !15
  %426 = insertelement <32 x float> %425, float %361, i64 14, !dbg !15
  %427 = insertelement <32 x float> %426, float %362, i64 15, !dbg !15
  %428 = insertelement <32 x float> %427, float %363, i64 16, !dbg !15
  %429 = insertelement <32 x float> %428, float %364, i64 17, !dbg !15
  %430 = insertelement <32 x float> %429, float %365, i64 18, !dbg !15
  %431 = insertelement <32 x float> %430, float %366, i64 19, !dbg !15
  %432 = insertelement <32 x float> %431, float %367, i64 20, !dbg !15
  %433 = insertelement <32 x float> %432, float %368, i64 21, !dbg !15
  %434 = insertelement <32 x float> %433, float %369, i64 22, !dbg !15
  %435 = insertelement <32 x float> %434, float %370, i64 23, !dbg !15
  %436 = insertelement <32 x float> %435, float %371, i64 24, !dbg !15
  %437 = insertelement <32 x float> %436, float %372, i64 25, !dbg !15
  %438 = insertelement <32 x float> %437, float %373, i64 26, !dbg !15
  %439 = insertelement <32 x float> %438, float %374, i64 27, !dbg !15
  %440 = insertelement <32 x float> %439, float %375, i64 28, !dbg !15
  %441 = insertelement <32 x float> %440, float %376, i64 29, !dbg !15
  %442 = insertelement <32 x float> %441, float %377, i64 30, !dbg !15
  %443 = insertelement <32 x float> %442, float %378, i64 31, !dbg !15
  %444 = insertvalue [2 x <32 x float>] %411, <32 x float> %443, 1, !dbg !15
  br label %445, !dbg !15

445:                                              ; preds = %273, %274
  %446 = phi [2 x <32 x float>] [ %444, %274 ], [ zeroinitializer, %273 ], !dbg !15
  br label %447, !dbg !15

447:                                              ; preds = %445
  %448 = extractvalue [2 x <32 x float>] %61, 0, !dbg !26
  %449 = extractvalue [2 x <32 x float>] %446, 0, !dbg !26
  %450 = shufflevector <32 x float> %448, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %451 = shufflevector <32 x float> %449, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %452 = fadd <4 x float> %450, %451, !dbg !26
  %453 = extractelement <4 x float> %452, i64 0, !dbg !27
  %454 = add nsw i64 %56, %8, !dbg !9
  %455 = shl i64 %454, 19, !dbg !9
  %456 = and i64 %455, 16252928, !dbg !9
  %457 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %456, !dbg !9
  %address.thread839 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !9, !range !34, !frisk.register_lifetimes !1
  %address.local840 = zext nneg i32 %address.thread839 to i64, !dbg !14
  %address.local841 = lshr i64 %address.local840, 3, !dbg !9
  %address.local842 = shl nuw nsw i64 %address.local841, 2, !dbg !9
  %address.local843 = or disjoint i64 %58, %address.local842, !dbg !9
  %address.local844 = shl i64 %address.local843, 7, !dbg !9
  %address.local845 = getelementptr [2 x i8], ptr addrspace(1) %457, i64 %address.local844, !dbg !9
  %address.local846 = shl nuw nsw i64 %address.local840, 2, !dbg !14
  %address.local847 = shl nuw nsw i64 %address.local841, 5, !dbg !9
  %address.local848 = sub nsw i64 %address.local846, %address.local847, !dbg !9
  %address.local849 = getelementptr [2 x i8], ptr addrspace(1) %address.local845, i64 %address.local848, !dbg !9
  %458 = load <4 x half>, ptr addrspace(1) %address.local849, align 2, !dbg !9
  %459 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %456, !dbg !9
  %address.local850 = shl i64 %address.local843, 7, !dbg !9
  %address.local851 = or disjoint i64 %address.local850, 128, !dbg !9
  %address.local852 = getelementptr [2 x i8], ptr addrspace(1) %459, i64 %address.local851, !dbg !9
  %address.local853 = getelementptr [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local848, !dbg !9
  %460 = load <4 x half>, ptr addrspace(1) %address.local853, align 2, !dbg !9
  %461 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %456, !dbg !9
  %address.local854 = shl i64 %address.local843, 7, !dbg !9
  %address.local855 = or disjoint i64 %address.local854, 256, !dbg !9
  %address.local856 = getelementptr [2 x i8], ptr addrspace(1) %461, i64 %address.local855, !dbg !9
  %address.local857 = getelementptr [2 x i8], ptr addrspace(1) %address.local856, i64 %address.local848, !dbg !9
  %462 = load <4 x half>, ptr addrspace(1) %address.local857, align 2, !dbg !9
  %463 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %456, !dbg !9
  %address.local858 = shl i64 %address.local843, 7, !dbg !9
  %address.local859 = or disjoint i64 %address.local858, 384, !dbg !9
  %address.local860 = getelementptr [2 x i8], ptr addrspace(1) %463, i64 %address.local859, !dbg !9
  %address.local861 = getelementptr [2 x i8], ptr addrspace(1) %address.local860, i64 %address.local848, !dbg !9
  %464 = load <4 x half>, ptr addrspace(1) %address.local861, align 2, !dbg !9
  %465 = call float @llvm.amdgcn.exp2.f32(float %453), !dbg !27
  %466 = insertelement <4 x float> poison, float %465, i64 0, !dbg !27
  %467 = extractelement <4 x float> %452, i64 1, !dbg !27
  %468 = call float @llvm.amdgcn.exp2.f32(float %467), !dbg !27
  %469 = fadd float %465, %468, !dbg !28
  %470 = insertelement <4 x float> %466, float %468, i64 1, !dbg !27
  %471 = extractelement <4 x float> %452, i64 2, !dbg !27
  %472 = call float @llvm.amdgcn.exp2.f32(float %471), !dbg !27
  %473 = fadd float %469, %472, !dbg !28
  %474 = insertelement <4 x float> %470, float %472, i64 2, !dbg !27
  %475 = extractelement <4 x float> %452, i64 3, !dbg !27
  %476 = call float @llvm.amdgcn.exp2.f32(float %475), !dbg !27
  %477 = fadd float %473, %476, !dbg !28
  %478 = insertelement <4 x float> %474, float %476, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %478) #8, !dbg !29, !frisk.register_lifetimes !1
  %479 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %479) #8, !dbg !26, !frisk.register_lifetimes !1
  %480 = shufflevector <32 x float> %448, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %481 = shufflevector <32 x float> %449, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %482 = fadd <4 x float> %480, %481, !dbg !26
  %483 = extractelement <4 x float> %482, i64 0, !dbg !27
  %484 = call float @llvm.amdgcn.exp2.f32(float %483), !dbg !27
  %485 = fadd float %477, %484, !dbg !28
  %486 = insertelement <4 x float> poison, float %484, i64 0, !dbg !27
  %487 = extractelement <4 x float> %482, i64 1, !dbg !27
  %488 = call float @llvm.amdgcn.exp2.f32(float %487), !dbg !27
  %489 = fadd float %485, %488, !dbg !28
  %490 = insertelement <4 x float> %486, float %488, i64 1, !dbg !27
  %491 = extractelement <4 x float> %482, i64 2, !dbg !27
  %492 = call float @llvm.amdgcn.exp2.f32(float %491), !dbg !27
  %493 = fadd float %489, %492, !dbg !28
  %494 = insertelement <4 x float> %490, float %492, i64 2, !dbg !27
  %495 = extractelement <4 x float> %482, i64 3, !dbg !27
  %496 = call float @llvm.amdgcn.exp2.f32(float %495), !dbg !27
  %497 = fadd float %493, %496, !dbg !28
  %498 = insertelement <4 x float> %494, float %496, i64 3, !dbg !27
  %math.result282 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %498) #8, !dbg !29, !frisk.register_lifetimes !1
  %499 = fptrunc <4 x float> %math.result282 to <4 x half>, !dbg !29
  %math.packed283 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %499) #8, !dbg !26, !frisk.register_lifetimes !1
  %500 = shufflevector <32 x float> %448, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %501 = shufflevector <32 x float> %449, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %502 = fadd <4 x float> %500, %501, !dbg !26
  %503 = extractelement <4 x float> %502, i64 0, !dbg !27
  %504 = call float @llvm.amdgcn.exp2.f32(float %503), !dbg !27
  %505 = fadd float %497, %504, !dbg !28
  %506 = insertelement <4 x float> poison, float %504, i64 0, !dbg !27
  %507 = extractelement <4 x float> %502, i64 1, !dbg !27
  %508 = call float @llvm.amdgcn.exp2.f32(float %507), !dbg !27
  %509 = fadd float %505, %508, !dbg !28
  %510 = insertelement <4 x float> %506, float %508, i64 1, !dbg !27
  %511 = extractelement <4 x float> %502, i64 2, !dbg !27
  %512 = call float @llvm.amdgcn.exp2.f32(float %511), !dbg !27
  %513 = fadd float %509, %512, !dbg !28
  %514 = insertelement <4 x float> %510, float %512, i64 2, !dbg !27
  %515 = extractelement <4 x float> %502, i64 3, !dbg !27
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !27
  %517 = fadd float %513, %516, !dbg !28
  %518 = insertelement <4 x float> %514, float %516, i64 3, !dbg !27
  %math.result284 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %518) #8, !dbg !29, !frisk.register_lifetimes !1
  %519 = fptrunc <4 x float> %math.result284 to <4 x half>, !dbg !29
  %math.packed285 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %519) #8, !dbg !26, !frisk.register_lifetimes !1
  %520 = shufflevector <32 x float> %448, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %521 = shufflevector <32 x float> %449, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %522 = fadd <4 x float> %520, %521, !dbg !26
  %523 = extractelement <4 x float> %522, i64 0, !dbg !27
  %524 = call float @llvm.amdgcn.exp2.f32(float %523), !dbg !27
  %525 = fadd float %517, %524, !dbg !28
  %526 = insertelement <4 x float> poison, float %524, i64 0, !dbg !27
  %527 = extractelement <4 x float> %522, i64 1, !dbg !27
  %528 = call float @llvm.amdgcn.exp2.f32(float %527), !dbg !27
  %529 = fadd float %525, %528, !dbg !28
  %530 = insertelement <4 x float> %526, float %528, i64 1, !dbg !27
  %531 = extractelement <4 x float> %522, i64 2, !dbg !27
  %532 = call float @llvm.amdgcn.exp2.f32(float %531), !dbg !27
  %533 = fadd float %529, %532, !dbg !28
  %534 = insertelement <4 x float> %530, float %532, i64 2, !dbg !27
  %535 = extractelement <4 x float> %522, i64 3, !dbg !27
  %536 = call float @llvm.amdgcn.exp2.f32(float %535), !dbg !27
  %537 = fadd float %533, %536, !dbg !28
  %538 = insertelement <4 x float> %534, float %536, i64 3, !dbg !27
  %math.result286 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %538) #8, !dbg !29, !frisk.register_lifetimes !1
  %539 = fptrunc <4 x float> %math.result286 to <4 x half>, !dbg !29
  %math.packed287 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %539) #8, !dbg !26, !frisk.register_lifetimes !1
  %540 = shufflevector <32 x float> %448, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %541 = shufflevector <32 x float> %449, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %542 = fadd <4 x float> %540, %541, !dbg !26
  %543 = extractelement <4 x float> %542, i64 0, !dbg !27
  %544 = call float @llvm.amdgcn.exp2.f32(float %543), !dbg !27
  %545 = fadd float %537, %544, !dbg !28
  %546 = insertelement <4 x float> poison, float %544, i64 0, !dbg !27
  %547 = extractelement <4 x float> %542, i64 1, !dbg !27
  %548 = call float @llvm.amdgcn.exp2.f32(float %547), !dbg !27
  %549 = fadd float %545, %548, !dbg !28
  %550 = insertelement <4 x float> %546, float %548, i64 1, !dbg !27
  %551 = extractelement <4 x float> %542, i64 2, !dbg !27
  %552 = call float @llvm.amdgcn.exp2.f32(float %551), !dbg !27
  %553 = fadd float %549, %552, !dbg !28
  %554 = insertelement <4 x float> %550, float %552, i64 2, !dbg !27
  %555 = extractelement <4 x float> %542, i64 3, !dbg !27
  %556 = call float @llvm.amdgcn.exp2.f32(float %555), !dbg !27
  %557 = fadd float %553, %556, !dbg !28
  %558 = insertelement <4 x float> %554, float %556, i64 3, !dbg !27
  %math.result288 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %558) #8, !dbg !29, !frisk.register_lifetimes !1
  %559 = fptrunc <4 x float> %math.result288 to <4 x half>, !dbg !29
  %math.packed289 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %559) #8, !dbg !26, !frisk.register_lifetimes !1
  %560 = shufflevector <32 x float> %448, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %561 = shufflevector <32 x float> %449, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %562 = fadd <4 x float> %560, %561, !dbg !26
  %563 = extractelement <4 x float> %562, i64 0, !dbg !27
  %564 = call float @llvm.amdgcn.exp2.f32(float %563), !dbg !27
  %565 = fadd float %557, %564, !dbg !28
  %566 = insertelement <4 x float> poison, float %564, i64 0, !dbg !27
  %567 = extractelement <4 x float> %562, i64 1, !dbg !27
  %568 = call float @llvm.amdgcn.exp2.f32(float %567), !dbg !27
  %569 = fadd float %565, %568, !dbg !28
  %570 = insertelement <4 x float> %566, float %568, i64 1, !dbg !27
  %571 = extractelement <4 x float> %562, i64 2, !dbg !27
  %572 = call float @llvm.amdgcn.exp2.f32(float %571), !dbg !27
  %573 = fadd float %569, %572, !dbg !28
  %574 = insertelement <4 x float> %570, float %572, i64 2, !dbg !27
  %575 = extractelement <4 x float> %562, i64 3, !dbg !27
  %576 = call float @llvm.amdgcn.exp2.f32(float %575), !dbg !27
  %577 = fadd float %573, %576, !dbg !28
  %578 = insertelement <4 x float> %574, float %576, i64 3, !dbg !27
  %math.result290 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %578) #8, !dbg !29, !frisk.register_lifetimes !1
  %579 = fptrunc <4 x float> %math.result290 to <4 x half>, !dbg !29
  %math.packed291 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %579) #8, !dbg !26, !frisk.register_lifetimes !1
  %580 = shufflevector <32 x float> %448, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %581 = shufflevector <32 x float> %449, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %582 = fadd <4 x float> %580, %581, !dbg !26
  %583 = extractelement <4 x float> %582, i64 0, !dbg !27
  %584 = call float @llvm.amdgcn.exp2.f32(float %583), !dbg !27
  %585 = fadd float %577, %584, !dbg !28
  %586 = insertelement <4 x float> poison, float %584, i64 0, !dbg !27
  %587 = extractelement <4 x float> %582, i64 1, !dbg !27
  %588 = call float @llvm.amdgcn.exp2.f32(float %587), !dbg !27
  %589 = fadd float %585, %588, !dbg !28
  %590 = insertelement <4 x float> %586, float %588, i64 1, !dbg !27
  %591 = extractelement <4 x float> %582, i64 2, !dbg !27
  %592 = call float @llvm.amdgcn.exp2.f32(float %591), !dbg !27
  %593 = fadd float %589, %592, !dbg !28
  %594 = insertelement <4 x float> %590, float %592, i64 2, !dbg !27
  %595 = extractelement <4 x float> %582, i64 3, !dbg !27
  %596 = call float @llvm.amdgcn.exp2.f32(float %595), !dbg !27
  %597 = fadd float %593, %596, !dbg !28
  %598 = insertelement <4 x float> %594, float %596, i64 3, !dbg !27
  %math.result292 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %598) #8, !dbg !29, !frisk.register_lifetimes !1
  %599 = fptrunc <4 x float> %math.result292 to <4 x half>, !dbg !29
  %math.packed293 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %599) #8, !dbg !26, !frisk.register_lifetimes !1
  %600 = shufflevector <32 x float> %448, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %601 = shufflevector <32 x float> %449, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %602 = fadd <4 x float> %600, %601, !dbg !26
  %603 = extractelement <4 x float> %602, i64 0, !dbg !27
  %604 = call float @llvm.amdgcn.exp2.f32(float %603), !dbg !27
  %605 = fadd float %597, %604, !dbg !28
  %606 = insertelement <4 x float> poison, float %604, i64 0, !dbg !27
  %607 = extractelement <4 x float> %602, i64 1, !dbg !27
  %608 = call float @llvm.amdgcn.exp2.f32(float %607), !dbg !27
  %609 = fadd float %605, %608, !dbg !28
  %610 = insertelement <4 x float> %606, float %608, i64 1, !dbg !27
  %611 = extractelement <4 x float> %602, i64 2, !dbg !27
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !27
  %613 = fadd float %609, %612, !dbg !28
  %614 = insertelement <4 x float> %610, float %612, i64 2, !dbg !27
  %615 = extractelement <4 x float> %602, i64 3, !dbg !27
  %616 = call float @llvm.amdgcn.exp2.f32(float %615), !dbg !27
  %617 = fadd float %613, %616, !dbg !28
  %618 = insertelement <4 x float> %614, float %616, i64 3, !dbg !27
  %math.result294 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %618) #8, !dbg !29, !frisk.register_lifetimes !1
  %619 = fptrunc <4 x float> %math.result294 to <4 x half>, !dbg !29
  %math.packed295 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %619) #8, !dbg !26, !frisk.register_lifetimes !1
  %620 = extractvalue [2 x <32 x float>] %61, 1, !dbg !26
  %621 = extractvalue [2 x <32 x float>] %446, 1, !dbg !26
  %622 = shufflevector <32 x float> %620, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %623 = shufflevector <32 x float> %621, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %624 = fadd <4 x float> %622, %623, !dbg !26
  %625 = extractelement <4 x float> %624, i64 0, !dbg !27
  %626 = call float @llvm.amdgcn.exp2.f32(float %625), !dbg !27
  %627 = insertelement <4 x float> poison, float %626, i64 0, !dbg !27
  %628 = extractelement <4 x float> %624, i64 1, !dbg !27
  %629 = call float @llvm.amdgcn.exp2.f32(float %628), !dbg !27
  %630 = fadd float %626, %629, !dbg !28
  %631 = insertelement <4 x float> %627, float %629, i64 1, !dbg !27
  %632 = extractelement <4 x float> %624, i64 2, !dbg !27
  %633 = call float @llvm.amdgcn.exp2.f32(float %632), !dbg !27
  %634 = fadd float %630, %633, !dbg !28
  %635 = insertelement <4 x float> %631, float %633, i64 2, !dbg !27
  %636 = extractelement <4 x float> %624, i64 3, !dbg !27
  %637 = call float @llvm.amdgcn.exp2.f32(float %636), !dbg !27
  %638 = fadd float %634, %637, !dbg !28
  %639 = insertelement <4 x float> %635, float %637, i64 3, !dbg !27
  %math.result296 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %639) #8, !dbg !29, !frisk.register_lifetimes !1
  %640 = fptrunc <4 x float> %math.result296 to <4 x half>, !dbg !29
  %math.packed297 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %640) #8, !dbg !26, !frisk.register_lifetimes !1
  %641 = shufflevector <32 x float> %620, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %642 = shufflevector <32 x float> %621, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %643 = fadd <4 x float> %641, %642, !dbg !26
  %644 = extractelement <4 x float> %643, i64 0, !dbg !27
  %645 = call float @llvm.amdgcn.exp2.f32(float %644), !dbg !27
  %646 = fadd float %638, %645, !dbg !28
  %647 = insertelement <4 x float> poison, float %645, i64 0, !dbg !27
  %648 = extractelement <4 x float> %643, i64 1, !dbg !27
  %649 = call float @llvm.amdgcn.exp2.f32(float %648), !dbg !27
  %650 = fadd float %646, %649, !dbg !28
  %651 = insertelement <4 x float> %647, float %649, i64 1, !dbg !27
  %652 = extractelement <4 x float> %643, i64 2, !dbg !27
  %653 = call float @llvm.amdgcn.exp2.f32(float %652), !dbg !27
  %654 = fadd float %650, %653, !dbg !28
  %655 = insertelement <4 x float> %651, float %653, i64 2, !dbg !27
  %656 = extractelement <4 x float> %643, i64 3, !dbg !27
  %657 = call float @llvm.amdgcn.exp2.f32(float %656), !dbg !27
  %658 = fadd float %654, %657, !dbg !28
  %659 = insertelement <4 x float> %655, float %657, i64 3, !dbg !27
  %math.result298 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %659) #8, !dbg !29, !frisk.register_lifetimes !1
  %660 = fptrunc <4 x float> %math.result298 to <4 x half>, !dbg !29
  %math.packed299 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %660) #8, !dbg !26, !frisk.register_lifetimes !1
  %661 = shufflevector <32 x float> %620, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %662 = shufflevector <32 x float> %621, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %663 = fadd <4 x float> %661, %662, !dbg !26
  %664 = extractelement <4 x float> %663, i64 0, !dbg !27
  %665 = call float @llvm.amdgcn.exp2.f32(float %664), !dbg !27
  %666 = fadd float %658, %665, !dbg !28
  %667 = insertelement <4 x float> poison, float %665, i64 0, !dbg !27
  %668 = extractelement <4 x float> %663, i64 1, !dbg !27
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !27
  %670 = fadd float %666, %669, !dbg !28
  %671 = insertelement <4 x float> %667, float %669, i64 1, !dbg !27
  %672 = extractelement <4 x float> %663, i64 2, !dbg !27
  %673 = call float @llvm.amdgcn.exp2.f32(float %672), !dbg !27
  %674 = fadd float %670, %673, !dbg !28
  %675 = insertelement <4 x float> %671, float %673, i64 2, !dbg !27
  %676 = extractelement <4 x float> %663, i64 3, !dbg !27
  %677 = call float @llvm.amdgcn.exp2.f32(float %676), !dbg !27
  %678 = fadd float %674, %677, !dbg !28
  %679 = insertelement <4 x float> %675, float %677, i64 3, !dbg !27
  %math.result300 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %679) #8, !dbg !29, !frisk.register_lifetimes !1
  %680 = fptrunc <4 x float> %math.result300 to <4 x half>, !dbg !29
  %math.packed301 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %680) #8, !dbg !26, !frisk.register_lifetimes !1
  %681 = shufflevector <32 x float> %620, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %682 = shufflevector <32 x float> %621, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %683 = fadd <4 x float> %681, %682, !dbg !26
  %684 = extractelement <4 x float> %683, i64 0, !dbg !27
  %685 = call float @llvm.amdgcn.exp2.f32(float %684), !dbg !27
  %686 = fadd float %678, %685, !dbg !28
  %687 = insertelement <4 x float> poison, float %685, i64 0, !dbg !27
  %688 = extractelement <4 x float> %683, i64 1, !dbg !27
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !27
  %690 = fadd float %686, %689, !dbg !28
  %691 = insertelement <4 x float> %687, float %689, i64 1, !dbg !27
  %692 = extractelement <4 x float> %683, i64 2, !dbg !27
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !27
  %694 = fadd float %690, %693, !dbg !28
  %695 = insertelement <4 x float> %691, float %693, i64 2, !dbg !27
  %696 = extractelement <4 x float> %683, i64 3, !dbg !27
  %697 = call float @llvm.amdgcn.exp2.f32(float %696), !dbg !27
  %698 = fadd float %694, %697, !dbg !28
  %699 = insertelement <4 x float> %695, float %697, i64 3, !dbg !27
  %math.result302 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %699) #8, !dbg !29, !frisk.register_lifetimes !1
  %700 = fptrunc <4 x float> %math.result302 to <4 x half>, !dbg !29
  %math.packed303 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %700) #8, !dbg !26, !frisk.register_lifetimes !1
  %701 = shufflevector <32 x float> %620, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %702 = shufflevector <32 x float> %621, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %703 = fadd <4 x float> %701, %702, !dbg !26
  %704 = extractelement <4 x float> %703, i64 0, !dbg !27
  %705 = call float @llvm.amdgcn.exp2.f32(float %704), !dbg !27
  %706 = fadd float %698, %705, !dbg !28
  %707 = insertelement <4 x float> poison, float %705, i64 0, !dbg !27
  %708 = extractelement <4 x float> %703, i64 1, !dbg !27
  %709 = call float @llvm.amdgcn.exp2.f32(float %708), !dbg !27
  %710 = fadd float %706, %709, !dbg !28
  %711 = insertelement <4 x float> %707, float %709, i64 1, !dbg !27
  %712 = extractelement <4 x float> %703, i64 2, !dbg !27
  %713 = call float @llvm.amdgcn.exp2.f32(float %712), !dbg !27
  %714 = fadd float %710, %713, !dbg !28
  %715 = insertelement <4 x float> %711, float %713, i64 2, !dbg !27
  %716 = extractelement <4 x float> %703, i64 3, !dbg !27
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !27
  %718 = fadd float %714, %717, !dbg !28
  %719 = insertelement <4 x float> %715, float %717, i64 3, !dbg !27
  %math.result304 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %719) #8, !dbg !29, !frisk.register_lifetimes !1
  %720 = fptrunc <4 x float> %math.result304 to <4 x half>, !dbg !29
  %math.packed305 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %720) #8, !dbg !26, !frisk.register_lifetimes !1
  %721 = shufflevector <32 x float> %620, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %722 = shufflevector <32 x float> %621, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %723 = fadd <4 x float> %721, %722, !dbg !26
  %724 = extractelement <4 x float> %723, i64 0, !dbg !27
  %725 = call float @llvm.amdgcn.exp2.f32(float %724), !dbg !27
  %726 = fadd float %718, %725, !dbg !28
  %727 = insertelement <4 x float> poison, float %725, i64 0, !dbg !27
  %728 = extractelement <4 x float> %723, i64 1, !dbg !27
  %729 = call float @llvm.amdgcn.exp2.f32(float %728), !dbg !27
  %730 = fadd float %726, %729, !dbg !28
  %731 = insertelement <4 x float> %727, float %729, i64 1, !dbg !27
  %732 = extractelement <4 x float> %723, i64 2, !dbg !27
  %733 = call float @llvm.amdgcn.exp2.f32(float %732), !dbg !27
  %734 = fadd float %730, %733, !dbg !28
  %735 = insertelement <4 x float> %731, float %733, i64 2, !dbg !27
  %736 = extractelement <4 x float> %723, i64 3, !dbg !27
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !27
  %738 = fadd float %734, %737, !dbg !28
  %739 = insertelement <4 x float> %735, float %737, i64 3, !dbg !27
  %math.result306 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %739) #8, !dbg !29, !frisk.register_lifetimes !1
  %740 = fptrunc <4 x float> %math.result306 to <4 x half>, !dbg !29
  %math.packed307 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %740) #8, !dbg !26, !frisk.register_lifetimes !1
  %741 = shufflevector <32 x float> %620, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %742 = shufflevector <32 x float> %621, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %743 = fadd <4 x float> %741, %742, !dbg !26
  %744 = extractelement <4 x float> %743, i64 0, !dbg !27
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !27
  %746 = fadd float %738, %745, !dbg !28
  %747 = insertelement <4 x float> poison, float %745, i64 0, !dbg !27
  %748 = extractelement <4 x float> %743, i64 1, !dbg !27
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !27
  %750 = fadd float %746, %749, !dbg !28
  %751 = insertelement <4 x float> %747, float %749, i64 1, !dbg !27
  %752 = extractelement <4 x float> %743, i64 2, !dbg !27
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !27
  %754 = fadd float %750, %753, !dbg !28
  %755 = insertelement <4 x float> %751, float %753, i64 2, !dbg !27
  %756 = extractelement <4 x float> %743, i64 3, !dbg !27
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !27
  %758 = fadd float %754, %757, !dbg !28
  %759 = insertelement <4 x float> %755, float %757, i64 3, !dbg !27
  %math.result308 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %759) #8, !dbg !29, !frisk.register_lifetimes !1
  %760 = fptrunc <4 x float> %math.result308 to <4 x half>, !dbg !29
  %math.packed309 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %760) #8, !dbg !26, !frisk.register_lifetimes !1
  %761 = shufflevector <32 x float> %620, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %762 = shufflevector <32 x float> %621, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %763 = fadd <4 x float> %761, %762, !dbg !26
  %764 = extractelement <4 x float> %763, i64 0, !dbg !27
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !27
  %766 = fadd float %758, %765, !dbg !28
  %767 = insertelement <4 x float> poison, float %765, i64 0, !dbg !27
  %768 = extractelement <4 x float> %763, i64 1, !dbg !27
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !27
  %770 = fadd float %766, %769, !dbg !28
  %771 = insertelement <4 x float> %767, float %769, i64 1, !dbg !27
  %772 = extractelement <4 x float> %763, i64 2, !dbg !27
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !27
  %774 = fadd float %770, %773, !dbg !28
  %775 = insertelement <4 x float> %771, float %773, i64 2, !dbg !27
  %776 = extractelement <4 x float> %763, i64 3, !dbg !27
  %777 = call float @llvm.amdgcn.exp2.f32(float %776), !dbg !27
  %778 = fadd float %774, %777, !dbg !28
  %779 = insertelement <4 x float> %775, float %777, i64 3, !dbg !27
  %math.result310 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %779) #8, !dbg !29, !frisk.register_lifetimes !1
  %780 = fptrunc <4 x float> %math.result310 to <4 x half>, !dbg !29
  %math.packed311 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %780) #8, !dbg !28, !frisk.register_lifetimes !1
  %781 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %782 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %781), !dbg !28
  %783 = add nuw nsw i32 %782, 64, !dbg !28
  %784 = xor i32 %782, 16, !dbg !28
  %785 = and i32 %783, 192, !dbg !28
  %786 = icmp samesign ult i32 %784, %785, !dbg !28
  %787 = select i1 %786, i32 %784, i32 %782, !dbg !28
  %788 = shl nuw nsw i32 %787, 2, !dbg !28
  %789 = bitcast float %617 to i32, !dbg !28
  %790 = call i32 @llvm.amdgcn.ds.bpermute(i32 %788, i32 %789), !dbg !28
  %791 = bitcast i32 %790 to float, !dbg !28
  %792 = fadd float %617, %791, !dbg !28
  %793 = xor i32 %782, 32, !dbg !28
  %794 = icmp samesign ult i32 %793, %785, !dbg !28
  %795 = select i1 %794, i32 %793, i32 %782, !dbg !28
  %796 = shl nuw nsw i32 %795, 2, !dbg !28
  %797 = bitcast float %792 to i32, !dbg !28
  %798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %797), !dbg !28
  %799 = bitcast i32 %798 to float, !dbg !28
  %800 = fadd float %792, %799, !dbg !28
  %801 = extractvalue [2 x <1 x float>] %50, 0, !dbg !28
  %802 = extractelement <1 x float> %801, i64 0, !dbg !28
  %803 = fadd float %800, %802, !dbg !28
  %804 = insertelement <1 x float> poison, float %803, i64 0, !dbg !16
  %805 = insertvalue [2 x <1 x float>] poison, <1 x float> %804, 0, !dbg !16
  %806 = bitcast float %778 to i32, !dbg !28
  %807 = call i32 @llvm.amdgcn.ds.bpermute(i32 %788, i32 %806), !dbg !28
  %808 = bitcast i32 %807 to float, !dbg !28
  %809 = fadd float %778, %808, !dbg !28
  %810 = bitcast float %809 to i32, !dbg !28
  %811 = call i32 @llvm.amdgcn.ds.bpermute(i32 %796, i32 %810), !dbg !28
  %812 = bitcast i32 %811 to float, !dbg !28
  %813 = fadd float %809, %812, !dbg !28
  %814 = extractvalue [2 x <1 x float>] %50, 1, !dbg !28
  %815 = extractelement <1 x float> %814, i64 0, !dbg !28
  %816 = fadd float %813, %815, !dbg !28
  %817 = insertelement <1 x float> poison, float %816, i64 0, !dbg !16
  %818 = insertvalue [2 x <1 x float>] %805, <1 x float> %817, 1, !dbg !16
  %819 = shufflevector <4 x half> %458, <4 x half> %460, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %820 = shufflevector <4 x half> %819, <4 x half> %462, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %821 = shufflevector <4 x half> %820, <4 x half> %464, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
  %address.local862 = shl nuw nsw i64 %address.local840, 5, !dbg !9
  %address.local863 = lshr i64 %address.local840, 1, !dbg !9
  %address.local864 = mul nuw nsw i64 %address.local863, 72, !dbg !9
  %address.local865 = add nuw nsw i64 %address.local862, %address.local864, !dbg !9
  %address.local866 = mul nsw i64 %address.local841, -540, !dbg !9
  %address.local867 = add nsw i64 %address.local865, %address.local866, !dbg !9
  %address.local868 = lshr i64 %address.local840, 4, !dbg !9
  %address.local869 = mul nuw nsw i64 %address.local868, 280, !dbg !9
  %address.local870 = add nsw i64 %address.local867, %address.local869, !dbg !9
  %address.local871 = lshr i32 %address.thread839, 4, !dbg !9
  %address.local872 = and i32 %address.thread839, 7, !dbg !9
  %address.local873 = add nuw nsw i32 %address.local871, %address.local872, !dbg !9
  %address.local874 = shl nuw nsw i32 %address.local873, 5, !dbg !9
  %address.local875 = and i32 %address.local874, 960, !dbg !9
  %address.local876 = zext nneg i32 %address.local875 to i64, !dbg !9
  %address.local877 = sub nsw i64 %address.local870, %address.local876, !dbg !9
  %address.local878 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local877, !dbg !9
  store <4 x half> %821, ptr addrspace(3) %address.local878, align 2, !dbg !9
  %822 = shufflevector <4 x half> %458, <4 x half> %460, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %823 = shufflevector <4 x half> %822, <4 x half> %462, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %824 = shufflevector <4 x half> %823, <4 x half> %464, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
  %address.base879 = call i64 asm sideeffect "", "=v,0"(i64 %.neg) #8, !dbg !9, !frisk.register_lifetimes !1
  %address.local880 = shl i64 %address.base879, 6, !dbg !9
  %address.local881 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local880, !dbg !9
  %address.local882 = shl nuw nsw i64 %address.local840, 6, !dbg !9
  %address.local883 = mul nsw i64 %address.local841, -828, !dbg !9
  %address.local884 = add nsw i64 %address.local882, %address.local883, !dbg !9
  %address.local885 = add nsw i64 %address.local884, %address.local864, !dbg !9
  %address.local886 = shl nuw nsw i64 %address.local840, 2, !dbg !9
  %address.local887 = and i64 %address.local886, 992, !dbg !9
  %address.local888 = add nsw i64 %address.local885, %address.local887, !dbg !9
  %address.local889 = add nsw i64 %address.local888, %address.local869, !dbg !9
  %address.local890 = lshr i32 %address.thread839, 4, !dbg !9
  %address.local891 = and i32 %address.thread839, 7, !dbg !9
  %address.local892 = add nuw nsw i32 %address.local890, %address.local891, !dbg !9
  %address.local893 = shl nuw nsw i32 %address.local892, 5, !dbg !9
  %address.local894 = and i32 %address.local893, 960, !dbg !9
  %address.local895 = zext nneg i32 %address.local894 to i64, !dbg !9
  %address.local896 = sub i64 %address.local889, %address.local895, !dbg !9
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) %address.local881, i64 %address.local896, !dbg !9
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local897, i64 16, !dbg !9
  store <4 x half> %824, ptr addrspace(3) %address.local898, align 2, !dbg !9
  %825 = shufflevector <4 x half> %458, <4 x half> %460, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %826 = shufflevector <4 x half> %825, <4 x half> %462, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %827 = shufflevector <4 x half> %826, <4 x half> %464, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  %address.base899 = call i64 asm sideeffect "", "=v,0"(i64 %.neg230) #8, !dbg !9, !frisk.register_lifetimes !1
  %address.local900 = shl i64 %address.base899, 6, !dbg !9
  %address.local901 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local900, !dbg !9
  %address.local902 = getelementptr [2 x i8], ptr addrspace(3) %address.local901, i64 %address.local896, !dbg !9
  %address.local903 = getelementptr i8, ptr addrspace(3) %address.local902, i64 32, !dbg !9
  store <4 x half> %827, ptr addrspace(3) %address.local903, align 2, !dbg !9
  %828 = shufflevector <4 x half> %458, <4 x half> %460, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %829 = shufflevector <4 x half> %828, <4 x half> %462, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %830 = shufflevector <4 x half> %829, <4 x half> %464, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  %address.local904 = mul nsw i64 %address.local841, -36, !dbg !9
  %address.local905 = add nsw i64 %address.local846, %address.local904, !dbg !9
  %address.local906 = add nsw i64 %address.local905, %address.local863, !dbg !9
  %address.local907 = add nsw i64 %address.local906, 3, !dbg !9
  %address.local908 = sdiv i64 %address.local907, -4, !dbg !9
  %address.local909 = shl i64 %address.local908, 6, !dbg !9
  %address.local910 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local909, !dbg !9
  %address.local911 = getelementptr [2 x i8], ptr addrspace(3) %address.local910, i64 %address.local896, !dbg !9
  %address.local912 = getelementptr i8, ptr addrspace(3) %address.local911, i64 48, !dbg !9
  store <4 x half> %830, ptr addrspace(3) %address.local912, align 2, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local913 = shl nuw nsw i64 %address.local840, 3, !dbg !18
  %address.local914 = lshr i64 %address.local840, 4, !dbg !14
  %address.local915 = mul nsw i64 %address.local914, -128, !dbg !18
  %address.local916 = add nsw i64 %address.local913, %address.local915, !dbg !18
  %address.local917 = and i64 %address.local840, 15, !dbg !8
  %address.local918 = lshr i64 %address.local917, 3, !dbg !18
  %address.local919 = mul nuw nsw i64 %address.local918, 72, !dbg !18
  %address.local920 = add nsw i64 %address.local916, %address.local919, !dbg !18
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !18
  %address.local922 = lshr i64 %address.local917, 2, !dbg !8
  %address.local923 = shl nuw nsw i64 %address.local922, 5, !dbg !18
  %address.local924 = getelementptr [2 x i8], ptr addrspace(3) %address.local921, i64 %address.local923, !dbg !18
  %address.local925 = lshr i64 %address.local917, 3, !dbg !18
  %address.local926 = mul nuw nsw i64 %address.local925, -64, !dbg !18
  %address.local927 = getelementptr [2 x i8], ptr addrspace(3) %address.local924, i64 %address.local926, !dbg !18
  %address.local928 = add nuw nsw i64 %address.local917, %address.local918, !dbg !18
  %address.local929 = lshr i64 %address.local928, 2, !dbg !18
  %address.local930 = mul nsw i64 %address.local929, -32, !dbg !18
  %address.local931 = getelementptr [2 x i8], ptr addrspace(3) %address.local927, i64 %address.local930, !dbg !18
  %address.local932 = lshr i64 %address.local840, 4, !dbg !14
  %address.local933 = and i64 %address.local932, 3, !dbg !14
  %address.local934 = shl nuw nsw i64 %address.local933, 9, !dbg !14
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) %address.local931, i64 %address.local934, !dbg !18
  %831 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %832 = extractvalue [2 x <8 x float>] %49, 0, !dbg !30
  %833 = shufflevector <8 x float> %832, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %834 = shufflevector <8 x half> %831, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %835 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %834, <4 x float> %833) #9, !dbg !18
  %address.local936 = mul nuw nsw i64 %address.local933, 576, !dbg !18
  %address.local937 = add nuw nsw i64 %address.local913, %address.local936, !dbg !18
  %address.local938 = add nsw i64 %address.local937, %address.local915, !dbg !18
  %address.local939 = add nsw i64 %address.local938, %address.local919, !dbg !18
  %address.local940 = add nsw i64 %address.local939, %address.local923, !dbg !18
  %address.local941 = shl nuw nsw i32 %address.thread839, 2, !dbg !18
  %address.local942 = and i32 %address.local941, 192, !dbg !18
  %address.local943 = or disjoint i32 %address.local942, 32, !dbg !18
  %address.local944 = shl nuw nsw i32 %address.thread839, 3, !dbg !18
  %address.local945 = and i32 %address.local944, 96, !dbg !18
  %address.local946 = add nuw nsw i32 %address.local943, %address.local945, !dbg !18
  %address.local947 = and i32 %address.local946, 448, !dbg !18
  %address.local948 = zext nneg i32 %address.local947 to i64, !dbg !18
  %address.local949 = sub nsw i64 %address.local940, %address.local948, !dbg !18
  %address.local950 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local949, !dbg !18
  %address.local951 = getelementptr [2 x i8], ptr addrspace(3) %address.local950, i64 %address.local930, !dbg !18
  %address.local952 = getelementptr i8, ptr addrspace(3) %address.local951, i64 576, !dbg !18
  %836 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %837 = shufflevector <8 x half> %831, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %838 = shufflevector <8 x float> %832, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %839 = extractvalue [2 x <8 x float>] %49, 1, !dbg !30
  %840 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %837, <4 x float> %835) #9, !dbg !18
  %841 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %842 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %843 = shufflevector <8 x float> %839, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %844 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %841, <4 x float> %840) #9, !dbg !18
  %845 = shufflevector <8 x float> %839, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %846 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %842, <4 x float> %844) #9, !dbg !18
  %address.local953 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local913, !dbg !18
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) %address.local953, i64 %address.local934, !dbg !18
  %address.local955 = getelementptr [2 x i8], ptr addrspace(3) %address.local954, i64 %address.local915, !dbg !18
  %address.local956 = getelementptr [2 x i8], ptr addrspace(3) %address.local955, i64 %address.local919, !dbg !18
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) %address.local956, i64 %address.local923, !dbg !18
  %address.local958 = getelementptr [2 x i8], ptr addrspace(3) %address.local957, i64 %address.local926, !dbg !18
  %address.local959 = getelementptr [2 x i8], ptr addrspace(3) %address.local958, i64 %address.local930, !dbg !18
  %address.local960 = getelementptr i8, ptr addrspace(3) %address.local959, i64 4096, !dbg !18
  %847 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %address.local961 = add nuw nsw i64 %address.local937, 2304, !dbg !18
  %address.local962 = add nsw i64 %address.local961, %address.local915, !dbg !18
  %address.local963 = add nsw i64 %address.local962, %address.local919, !dbg !18
  %address.local964 = add nsw i64 %address.local963, %address.local923, !dbg !18
  %address.local965 = shl nuw nsw i32 %address.thread839, 2, !dbg !18
  %address.local966 = and i32 %address.local965, 192, !dbg !18
  %address.local967 = or disjoint i32 %address.local966, 288, !dbg !18
  %address.local968 = shl nuw nsw i32 %address.thread839, 3, !dbg !18
  %address.local969 = and i32 %address.local968, 96, !dbg !18
  %address.local970 = add nuw nsw i32 %address.local967, %address.local969, !dbg !18
  %address.local971 = and i32 %address.local970, 960, !dbg !18
  %address.local972 = zext nneg i32 %address.local971 to i64, !dbg !18
  %address.local973 = sub i64 %address.local964, %address.local972, !dbg !18
  %address.local974 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local973, !dbg !18
  %address.local975 = getelementptr [2 x i8], ptr addrspace(3) %address.local974, i64 %address.local930, !dbg !18
  %address.local976 = getelementptr i8, ptr addrspace(3) %address.local975, i64 576, !dbg !18
  %848 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %address.local977 = add nsw i64 %address.local920, %address.local923, !dbg !18
  %address.local978 = shl nuw nsw i64 %address.local840, 3, !dbg !18
  %address.local979 = and i64 %address.local978, 64, !dbg !18
  %address.local980 = sub nsw i64 %address.local977, %address.local979, !dbg !18
  %address.local981 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local980, !dbg !18
  %address.local982 = add nuw nsw i64 %address.local928, 18, !dbg !18
  %address.local983 = lshr i64 %address.local982, 2, !dbg !18
  %address.local984 = mul nsw i64 %address.local983, -32, !dbg !18
  %address.local985 = getelementptr [2 x i8], ptr addrspace(3) %address.local981, i64 %address.local984, !dbg !18
  %address.local986 = getelementptr [2 x i8], ptr addrspace(3) %address.local985, i64 %address.local934, !dbg !18
  %address.local987 = getelementptr i8, ptr addrspace(3) %address.local986, i64 544, !dbg !18
  %849 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %850 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %851 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %850, <4 x float> %846) #9, !dbg !18
  %852 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %853 = shufflevector <8 x half> %848, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %854 = shufflevector <8 x half> %848, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %855 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %856 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %address.local988 = add nsw i64 %address.local939, %address.local923, !dbg !18
  %address.local989 = shl nuw nsw i32 %address.thread839, 2, !dbg !18
  %address.local990 = and i32 %address.local989, 192, !dbg !18
  %address.local991 = add nuw nsw i32 %address.local990, 160, !dbg !18
  %address.local992 = shl nuw nsw i32 %address.thread839, 3, !dbg !18
  %address.local993 = and i32 %address.local992, 96, !dbg !18
  %address.local994 = add nuw nsw i32 %address.local991, %address.local993, !dbg !18
  %address.local995 = and i32 %address.local994, 960, !dbg !18
  %address.local996 = zext nneg i32 %address.local995 to i64, !dbg !18
  %address.local997 = sub nsw i64 %address.local988, %address.local996, !dbg !18
  %address.local998 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local997, !dbg !18
  %address.local999 = getelementptr [2 x i8], ptr addrspace(3) %address.local998, i64 %address.local984, !dbg !18
  %address.local1000 = getelementptr i8, ptr addrspace(3) %address.local999, i64 1376, !dbg !18
  %857 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %858 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %852, <4 x float> %851) #9, !dbg !18
  %address.local1001 = add nsw i64 %address.local937, %address.local915, !dbg !18
  %address.local1002 = add nsw i64 %address.local1001, %address.local919, !dbg !18
  %address.local1003 = add nsw i64 %address.local1002, %address.local923, !dbg !18
  %address.local1004 = lshr i64 %address.local917, 3, !dbg !18
  %address.local1005 = add nuw nsw i64 %address.local1004, %address.local933, !dbg !18
  %address.local1006 = shl nuw nsw i64 %address.local1005, 6, !dbg !18
  %address.local1007 = sub nsw i64 %address.local1003, %address.local1006, !dbg !18
  %address.local1008 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1007, !dbg !18
  %address.local1009 = getelementptr [2 x i8], ptr addrspace(3) %address.local1008, i64 %address.local984, !dbg !18
  %address.local1010 = getelementptr i8, ptr addrspace(3) %address.local1009, i64 4640, !dbg !18
  %859 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %address.local1011 = add nsw i64 %address.local963, %address.local923, !dbg !18
  %address.local1012 = shl nuw nsw i32 %address.thread839, 2, !dbg !18
  %address.local1013 = and i32 %address.local1012, 192, !dbg !18
  %address.local1014 = add nuw nsw i32 %address.local1013, 416, !dbg !18
  %address.local1015 = shl nuw nsw i32 %address.thread839, 3, !dbg !18
  %address.local1016 = and i32 %address.local1015, 96, !dbg !18
  %address.local1017 = add nuw nsw i32 %address.local1014, %address.local1016, !dbg !18
  %address.local1018 = and i32 %address.local1017, 1984, !dbg !18
  %address.local1019 = zext nneg i32 %address.local1018 to i64, !dbg !18
  %address.local1020 = sub i64 %address.local1011, %address.local1019, !dbg !18
  %address.local1021 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1020, !dbg !18
  %address.local1022 = getelementptr [2 x i8], ptr addrspace(3) %address.local1021, i64 %address.local984, !dbg !18
  %address.local1023 = getelementptr i8, ptr addrspace(3) %address.local1022, i64 1376, !dbg !18
  %860 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %861 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %862 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %853, <4 x float> %858) #9, !dbg !18
  %863 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %864 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %865 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %866 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %867 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %868 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %854, <4 x float> %862) #9, !dbg !18
  %869 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %870 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %855, <4 x float> %838) #9, !dbg !18
  %871 = shufflevector <8 x half> %831, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %872 = shufflevector <8 x half> %831, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %873 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %874 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %875 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %876 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %877 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %856, <4 x float> %870) #9, !dbg !18
  %878 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %879 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %861, <4 x float> %877) #9, !dbg !18
  %880 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %881 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %882 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %883 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %884 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %863, <4 x float> %879) #9, !dbg !18
  %885 = shufflevector <8 x half> %880, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %886 = shufflevector <8 x half> %880, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %887 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %888 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %889 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %890 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %864, <4 x float> %884) #9, !dbg !18
  %891 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %892 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %893 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %865, <4 x float> %890) #9, !dbg !18
  %894 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %895 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %896 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %897 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %866, <4 x float> %893) #9, !dbg !18
  %898 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %867, <4 x float> %897) #9, !dbg !18
  %899 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %871, <4 x float> %843) #9, !dbg !18
  %900 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %872, <4 x float> %899) #9, !dbg !18
  %901 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %873, <4 x float> %900) #9, !dbg !18
  %902 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %874, <4 x float> %901) #9, !dbg !18
  %903 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %875, <4 x float> %902) #9, !dbg !18
  %904 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %876, <4 x float> %903) #9, !dbg !18
  %905 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %882, <4 x float> %904) #9, !dbg !18
  %906 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %883, <4 x float> %905) #9, !dbg !18
  %907 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %885, <4 x float> %845) #9, !dbg !18
  %908 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %886, <4 x float> %907) #9, !dbg !18
  %909 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %887, <4 x float> %908) #9, !dbg !18
  %910 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %888, <4 x float> %909) #9, !dbg !18
  %911 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %892, <4 x float> %910) #9, !dbg !18
  %912 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %894, <4 x float> %911) #9, !dbg !18
  %913 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %895, <4 x float> %912) #9, !dbg !18
  %914 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %896, <4 x float> %913) #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %915 = add nsw i64 %56, %8, !dbg !9
  %916 = shl i64 %915, 19, !dbg !9
  %917 = and i64 %916, 16252928, !dbg !9
  %address.local1024 = add i64 %917, %address.local844, !dbg !9
  %address.local1025 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1024, !dbg !9
  %address.local1026 = shl nuw nsw i32 %address.thread839, 2, !dbg !9
  %address.local1027 = and i32 %address.local1026, 28, !dbg !9
  %address.local1028 = or disjoint i32 %address.local1027, 32, !dbg !9
  %address.local1029 = zext nneg i32 %address.local1028 to i64, !dbg !9
  %address.local1030 = getelementptr [2 x i8], ptr addrspace(1) %address.local1025, i64 %address.local1029, !dbg !9
  %918 = load <4 x half>, ptr addrspace(1) %address.local1030, align 2, !dbg !9
  %address.local1031 = add i64 %917, %address.local851, !dbg !9
  %address.local1032 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1031, !dbg !9
  %address.local1033 = getelementptr [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1029, !dbg !9
  %919 = load <4 x half>, ptr addrspace(1) %address.local1033, align 2, !dbg !9
  %address.local1034 = add i64 %917, %address.local855, !dbg !9
  %address.local1035 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1034, !dbg !9
  %address.local1036 = getelementptr [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1029, !dbg !9
  %920 = load <4 x half>, ptr addrspace(1) %address.local1036, align 2, !dbg !9
  %address.local1037 = add i64 %917, %address.local859, !dbg !9
  %address.local1038 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1037, !dbg !9
  %address.local1039 = getelementptr [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1029, !dbg !9
  %921 = load <4 x half>, ptr addrspace(1) %address.local1039, align 2, !dbg !9
  %922 = shufflevector <4 x half> %918, <4 x half> %919, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %923 = shufflevector <4 x half> %922, <4 x half> %920, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %924 = shufflevector <4 x half> %923, <4 x half> %921, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
  store <4 x half> %924, ptr addrspace(3) %address.local878, align 2, !dbg !9
  %925 = shufflevector <4 x half> %918, <4 x half> %919, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %926 = shufflevector <4 x half> %925, <4 x half> %920, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %927 = shufflevector <4 x half> %926, <4 x half> %921, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
  store <4 x half> %927, ptr addrspace(3) %address.local898, align 2, !dbg !9
  %928 = shufflevector <4 x half> %918, <4 x half> %919, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %929 = shufflevector <4 x half> %928, <4 x half> %920, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %930 = shufflevector <4 x half> %929, <4 x half> %921, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  store <4 x half> %930, ptr addrspace(3) %address.local903, align 2, !dbg !9
  %931 = shufflevector <4 x half> %918, <4 x half> %919, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %932 = shufflevector <4 x half> %931, <4 x half> %920, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %933 = shufflevector <4 x half> %932, <4 x half> %921, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  store <4 x half> %933, ptr addrspace(3) %address.local912, align 2, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %934 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %935 = extractvalue [2 x <8 x float>] %51, 0, !dbg !30
  %936 = shufflevector <8 x float> %935, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %937 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %938 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %937, <4 x float> %936) #9, !dbg !18
  %939 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %940 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %941 = shufflevector <8 x float> %935, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %942 = extractvalue [2 x <8 x float>] %51, 1, !dbg !30
  %943 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %940, <4 x float> %938) #9, !dbg !18
  %944 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %945 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %946 = shufflevector <8 x float> %942, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %947 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %944, <4 x float> %943) #9, !dbg !18
  %948 = shufflevector <8 x float> %942, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %949 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %945, <4 x float> %947) #9, !dbg !18
  %950 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %951 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %952 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %953 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %954 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %953, <4 x float> %949) #9, !dbg !18
  %955 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %956 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %957 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %958 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %959 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %960 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %961 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %955, <4 x float> %954) #9, !dbg !18
  %962 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %963 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %964 = shufflevector <8 x half> %960, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %965 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %956, <4 x float> %961) #9, !dbg !18
  %966 = shufflevector <8 x half> %960, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %967 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %968 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %969 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %970 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %971 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %957, <4 x float> %965) #9, !dbg !18
  %972 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %973 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %958, <4 x float> %941) #9, !dbg !18
  %974 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %975 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %976 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %977 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %978 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %979 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %980 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %959, <4 x float> %973) #9, !dbg !18
  %981 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %982 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %964, <4 x float> %980) #9, !dbg !18
  %983 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %984 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %985 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %986 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %987 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %966, <4 x float> %982) #9, !dbg !18
  %988 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %989 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %990 = shufflevector <8 x half> %984, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %991 = shufflevector <8 x half> %984, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %992 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %993 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %967, <4 x float> %987) #9, !dbg !18
  %994 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %995 = shufflevector <8 x half> %992, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %996 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %968, <4 x float> %993) #9, !dbg !18
  %997 = shufflevector <8 x half> %992, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %998 = shufflevector <8 x half> %994, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %999 = shufflevector <8 x half> %994, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1000 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %969, <4 x float> %996) #9, !dbg !18
  %1001 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %970, <4 x float> %1000) #9, !dbg !18
  %1002 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %974, <4 x float> %946) #9, !dbg !18
  %1003 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %975, <4 x float> %1002) #9, !dbg !18
  %1004 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %976, <4 x float> %1003) #9, !dbg !18
  %1005 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %977, <4 x float> %1004) #9, !dbg !18
  %1006 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %978, <4 x float> %1005) #9, !dbg !18
  %1007 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %979, <4 x float> %1006) #9, !dbg !18
  %1008 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %985, <4 x float> %1007) #9, !dbg !18
  %1009 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %986, <4 x float> %1008) #9, !dbg !18
  %1010 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %988, <4 x float> %948) #9, !dbg !18
  %1011 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %989, <4 x float> %1010) #9, !dbg !18
  %1012 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %990, <4 x float> %1011) #9, !dbg !18
  %1013 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %991, <4 x float> %1012) #9, !dbg !18
  %1014 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %995, <4 x float> %1013) #9, !dbg !18
  %1015 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %997, <4 x float> %1014) #9, !dbg !18
  %1016 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %998, <4 x float> %1015) #9, !dbg !18
  %1017 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %999, <4 x float> %1016) #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local1040 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1024, !dbg !9
  %address.local1041 = shl nuw nsw i32 %address.thread839, 2, !dbg !9
  %address.local1042 = and i32 %address.local1041, 28, !dbg !9
  %address.local1043 = or disjoint i32 %address.local1042, 64, !dbg !9
  %address.local1044 = zext nneg i32 %address.local1043 to i64, !dbg !9
  %address.local1045 = getelementptr [2 x i8], ptr addrspace(1) %address.local1040, i64 %address.local1044, !dbg !9
  %1018 = load <4 x half>, ptr addrspace(1) %address.local1045, align 2, !dbg !9
  %address.local1046 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1031, !dbg !9
  %address.local1047 = getelementptr [2 x i8], ptr addrspace(1) %address.local1046, i64 %address.local1044, !dbg !9
  %1019 = load <4 x half>, ptr addrspace(1) %address.local1047, align 2, !dbg !9
  %address.local1048 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1034, !dbg !9
  %address.local1049 = getelementptr [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1044, !dbg !9
  %1020 = load <4 x half>, ptr addrspace(1) %address.local1049, align 2, !dbg !9
  %address.local1050 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1037, !dbg !9
  %address.local1051 = getelementptr [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1044, !dbg !9
  %1021 = load <4 x half>, ptr addrspace(1) %address.local1051, align 2, !dbg !9
  %1022 = shufflevector <4 x half> %1018, <4 x half> %1019, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %1023 = shufflevector <4 x half> %1022, <4 x half> %1020, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %1024 = shufflevector <4 x half> %1023, <4 x half> %1021, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
  store <4 x half> %1024, ptr addrspace(3) %address.local878, align 2, !dbg !9
  %1025 = shufflevector <4 x half> %1018, <4 x half> %1019, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %1026 = shufflevector <4 x half> %1025, <4 x half> %1020, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %1027 = shufflevector <4 x half> %1026, <4 x half> %1021, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
  store <4 x half> %1027, ptr addrspace(3) %address.local898, align 2, !dbg !9
  %1028 = shufflevector <4 x half> %1018, <4 x half> %1019, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %1029 = shufflevector <4 x half> %1028, <4 x half> %1020, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %1030 = shufflevector <4 x half> %1029, <4 x half> %1021, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  store <4 x half> %1030, ptr addrspace(3) %address.local903, align 2, !dbg !9
  %1031 = shufflevector <4 x half> %1018, <4 x half> %1019, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %1032 = shufflevector <4 x half> %1031, <4 x half> %1020, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %1033 = shufflevector <4 x half> %1032, <4 x half> %1021, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  store <4 x half> %1033, ptr addrspace(3) %address.local912, align 2, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %1034 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %1035 = extractvalue [2 x <8 x float>] %52, 0, !dbg !30
  %1036 = shufflevector <8 x float> %1035, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1037 = shufflevector <8 x half> %1034, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1038 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1037, <4 x float> %1036) #9, !dbg !18
  %1039 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %1040 = shufflevector <8 x half> %1034, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1041 = shufflevector <8 x float> %1035, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1042 = extractvalue [2 x <8 x float>] %52, 1, !dbg !30
  %1043 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1040, <4 x float> %1038) #9, !dbg !18
  %1044 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1045 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1046 = shufflevector <8 x float> %1042, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1047 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1044, <4 x float> %1043) #9, !dbg !18
  %1048 = shufflevector <8 x float> %1042, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1049 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1045, <4 x float> %1047) #9, !dbg !18
  %1050 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %1051 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %1052 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %1053 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1054 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1053, <4 x float> %1049) #9, !dbg !18
  %1055 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1056 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1057 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1058 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1059 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1060 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %1061 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %1055, <4 x float> %1054) #9, !dbg !18
  %1062 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %1063 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %1064 = shufflevector <8 x half> %1060, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1065 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %1056, <4 x float> %1061) #9, !dbg !18
  %1066 = shufflevector <8 x half> %1060, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1067 = shufflevector <8 x half> %1062, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1068 = shufflevector <8 x half> %1062, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1069 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1070 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1071 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %1057, <4 x float> %1065) #9, !dbg !18
  %1072 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %1073 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1058, <4 x float> %1041) #9, !dbg !18
  %1074 = shufflevector <8 x half> %1034, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1075 = shufflevector <8 x half> %1034, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1076 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1077 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1078 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1079 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1080 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1059, <4 x float> %1073) #9, !dbg !18
  %1081 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %1082 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1064, <4 x float> %1080) #9, !dbg !18
  %1083 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %1084 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %1085 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1086 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1087 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1066, <4 x float> %1082) #9, !dbg !18
  %1088 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1089 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1090 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1091 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1092 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %1093 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1067, <4 x float> %1087) #9, !dbg !18
  %1094 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %1095 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1096 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %1068, <4 x float> %1093) #9, !dbg !18
  %1097 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1098 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1099 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1100 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %1069, <4 x float> %1096) #9, !dbg !18
  %1101 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %1070, <4 x float> %1100) #9, !dbg !18
  %1102 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %1074, <4 x float> %1046) #9, !dbg !18
  %1103 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %1075, <4 x float> %1102) #9, !dbg !18
  %1104 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %1076, <4 x float> %1103) #9, !dbg !18
  %1105 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %1077, <4 x float> %1104) #9, !dbg !18
  %1106 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %1078, <4 x float> %1105) #9, !dbg !18
  %1107 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1079, <4 x float> %1106) #9, !dbg !18
  %1108 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1085, <4 x float> %1107) #9, !dbg !18
  %1109 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1086, <4 x float> %1108) #9, !dbg !18
  %1110 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %1088, <4 x float> %1048) #9, !dbg !18
  %1111 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %1089, <4 x float> %1110) #9, !dbg !18
  %1112 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %1090, <4 x float> %1111) #9, !dbg !18
  %1113 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %1091, <4 x float> %1112) #9, !dbg !18
  %1114 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %1095, <4 x float> %1113) #9, !dbg !18
  %1115 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1097, <4 x float> %1114) #9, !dbg !18
  %1116 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1098, <4 x float> %1115) #9, !dbg !18
  %1117 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1099, <4 x float> %1116) #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local1052 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1024, !dbg !9
  %address.local1053 = shl nuw nsw i32 %address.thread839, 2, !dbg !9
  %address.local1054 = and i32 %address.local1053, 28, !dbg !9
  %address.local1055 = or disjoint i32 %address.local1054, 96, !dbg !9
  %address.local1056 = zext nneg i32 %address.local1055 to i64, !dbg !9
  %address.local1057 = getelementptr [2 x i8], ptr addrspace(1) %address.local1052, i64 %address.local1056, !dbg !9
  %1118 = load <4 x half>, ptr addrspace(1) %address.local1057, align 2, !dbg !9
  %address.local1058 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1031, !dbg !9
  %address.local1059 = getelementptr [2 x i8], ptr addrspace(1) %address.local1058, i64 %address.local1056, !dbg !9
  %1119 = load <4 x half>, ptr addrspace(1) %address.local1059, align 2, !dbg !9
  %address.local1060 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1034, !dbg !9
  %address.local1061 = getelementptr [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1056, !dbg !9
  %1120 = load <4 x half>, ptr addrspace(1) %address.local1061, align 2, !dbg !9
  %address.local1062 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1037, !dbg !9
  %address.local1063 = getelementptr [2 x i8], ptr addrspace(1) %address.local1062, i64 %address.local1056, !dbg !9
  %1121 = load <4 x half>, ptr addrspace(1) %address.local1063, align 2, !dbg !9
  %1122 = shufflevector <4 x half> %1118, <4 x half> %1119, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %1123 = shufflevector <4 x half> %1122, <4 x half> %1120, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %1124 = shufflevector <4 x half> %1123, <4 x half> %1121, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
  store <4 x half> %1124, ptr addrspace(3) %address.local878, align 2, !dbg !9
  %1125 = shufflevector <4 x half> %1118, <4 x half> %1119, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %1126 = shufflevector <4 x half> %1125, <4 x half> %1120, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %1127 = shufflevector <4 x half> %1126, <4 x half> %1121, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
  store <4 x half> %1127, ptr addrspace(3) %address.local898, align 2, !dbg !9
  %1128 = shufflevector <4 x half> %1118, <4 x half> %1119, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %1129 = shufflevector <4 x half> %1128, <4 x half> %1120, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %1130 = shufflevector <4 x half> %1129, <4 x half> %1121, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  store <4 x half> %1130, ptr addrspace(3) %address.local903, align 2, !dbg !9
  %1131 = shufflevector <4 x half> %1118, <4 x half> %1119, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %1132 = shufflevector <4 x half> %1131, <4 x half> %1120, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %1133 = shufflevector <4 x half> %1132, <4 x half> %1121, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  store <4 x half> %1133, ptr addrspace(3) %address.local912, align 2, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %1134 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %1135 = extractvalue [2 x <8 x float>] %53, 0, !dbg !30
  %1136 = shufflevector <8 x float> %1135, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1137 = shufflevector <8 x half> %1134, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1138 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1137, <4 x float> %1136) #9, !dbg !18
  %1139 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %1140 = shufflevector <8 x half> %1134, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1141 = shufflevector <8 x float> %1135, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1142 = extractvalue [2 x <8 x float>] %53, 1, !dbg !30
  %1143 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1140, <4 x float> %1138) #9, !dbg !18
  %1144 = shufflevector <8 x half> %1139, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1145 = shufflevector <8 x half> %1139, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1146 = shufflevector <8 x float> %1142, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1147 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1144, <4 x float> %1143) #9, !dbg !18
  %1148 = shufflevector <8 x float> %1142, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1149 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1145, <4 x float> %1147) #9, !dbg !18
  %1150 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %1151 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %1152 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %1153 = shufflevector <8 x half> %1150, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1154 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1153, <4 x float> %1149) #9, !dbg !18
  %1155 = shufflevector <8 x half> %1150, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1156 = shufflevector <8 x half> %1151, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1157 = shufflevector <8 x half> %1151, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1158 = shufflevector <8 x half> %1152, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1159 = shufflevector <8 x half> %1152, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1160 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %1161 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %1155, <4 x float> %1154) #9, !dbg !18
  %1162 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %1163 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %1164 = shufflevector <8 x half> %1160, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1165 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %1156, <4 x float> %1161) #9, !dbg !18
  %1166 = shufflevector <8 x half> %1160, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1167 = shufflevector <8 x half> %1162, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1168 = shufflevector <8 x half> %1162, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1169 = shufflevector <8 x half> %1163, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1170 = shufflevector <8 x half> %1163, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1171 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %1157, <4 x float> %1165) #9, !dbg !18
  %1172 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %1173 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1158, <4 x float> %1141) #9, !dbg !18
  %1174 = shufflevector <8 x half> %1134, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1175 = shufflevector <8 x half> %1134, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1176 = shufflevector <8 x half> %1139, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1177 = shufflevector <8 x half> %1139, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1178 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1179 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1180 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1159, <4 x float> %1173) #9, !dbg !18
  %1181 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %1182 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1164, <4 x float> %1180) #9, !dbg !18
  %1183 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %1184 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %1185 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1186 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1187 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1166, <4 x float> %1182) #9, !dbg !18
  %1188 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1189 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1190 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1191 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1192 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %1193 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1167, <4 x float> %1187) #9, !dbg !18
  %1194 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %1195 = shufflevector <4 x float> %868, <4 x float> %898, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1196 = insertvalue [2 x <8 x float>] poison, <8 x float> %1195, 0, !dbg !16
  %1197 = shufflevector <4 x float> %906, <4 x float> %914, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1198 = insertvalue [2 x <8 x float>] %1196, <8 x float> %1197, 1, !dbg !16
  %1199 = shufflevector <8 x half> %1192, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1200 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %1168, <4 x float> %1193) #9, !dbg !18
  %1201 = shufflevector <8 x half> %1192, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1202 = shufflevector <4 x float> %971, <4 x float> %1001, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1203 = insertvalue [2 x <8 x float>] poison, <8 x float> %1202, 0, !dbg !16
  %1204 = shufflevector <4 x float> %1009, <4 x float> %1017, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1205 = insertvalue [2 x <8 x float>] %1203, <8 x float> %1204, 1, !dbg !16
  %1206 = shufflevector <8 x half> %1194, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1207 = shufflevector <8 x half> %1194, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1208 = shufflevector <4 x float> %1071, <4 x float> %1101, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1209 = insertvalue [2 x <8 x float>] poison, <8 x float> %1208, 0, !dbg !16
  %1210 = shufflevector <4 x float> %1109, <4 x float> %1117, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1211 = insertvalue [2 x <8 x float>] %1209, <8 x float> %1210, 1, !dbg !16
  %1212 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %1169, <4 x float> %1200) #9, !dbg !18
  %1213 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %1170, <4 x float> %1212) #9, !dbg !18
  %1214 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %1174, <4 x float> %1146) #9, !dbg !18
  %1215 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %1175, <4 x float> %1214) #9, !dbg !18
  %1216 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %1176, <4 x float> %1215) #9, !dbg !18
  %1217 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %1177, <4 x float> %1216) #9, !dbg !18
  %1218 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %1178, <4 x float> %1217) #9, !dbg !18
  %1219 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1179, <4 x float> %1218) #9, !dbg !18
  %1220 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1185, <4 x float> %1219) #9, !dbg !18
  %1221 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1186, <4 x float> %1220) #9, !dbg !18
  %1222 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %1188, <4 x float> %1148) #9, !dbg !18
  %1223 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %1189, <4 x float> %1222) #9, !dbg !18
  %1224 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %1190, <4 x float> %1223) #9, !dbg !18
  %1225 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %1191, <4 x float> %1224) #9, !dbg !18
  %1226 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %1199, <4 x float> %1225) #9, !dbg !18
  %1227 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1201, <4 x float> %1226) #9, !dbg !18
  %1228 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1206, <4 x float> %1227) #9, !dbg !18
  %1229 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1207, <4 x float> %1228) #9, !dbg !18
  %1230 = shufflevector <4 x float> %1171, <4 x float> %1213, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1231 = insertvalue [2 x <8 x float>] poison, <8 x float> %1230, 0, !dbg !16
  %1232 = shufflevector <4 x float> %1221, <4 x float> %1229, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1233 = insertvalue [2 x <8 x float>] %1231, <8 x float> %1232, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1234 = add nuw nsw i64 %48, 1, !dbg !16
  br label %47, !dbg !16

1235:                                             ; preds = %47
  %1236 = and i32 %19, 16, !dbg !16
  %.not38 = icmp eq i32 %1236, 0, !dbg !16
  %1237 = and i32 %19, 32, !dbg !16
  %.not39 = icmp eq i32 %1237, 0, !dbg !16
  %1238 = extractvalue [2 x <8 x float>] %53, 0, !dbg !16
  %1239 = extractelement <8 x float> %1238, i64 0, !dbg !16
  %1240 = extractelement <8 x float> %1238, i64 1, !dbg !16
  %1241 = extractelement <8 x float> %1238, i64 2, !dbg !16
  %1242 = extractelement <8 x float> %1238, i64 3, !dbg !16
  %1243 = select i1 %.not38, float %1240, float %1239, !dbg !16
  %1244 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %1245 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1244), !dbg !16
  %1246 = add nuw nsw i32 %1245, 64, !dbg !16
  %1247 = xor i32 %1245, 16, !dbg !16
  %1248 = and i32 %1246, 192, !dbg !16
  %1249 = icmp samesign ult i32 %1247, %1248, !dbg !16
  %1250 = select i1 %1249, i32 %1247, i32 %1245, !dbg !16
  %1251 = shl nuw nsw i32 %1250, 2, !dbg !16
  %1252 = bitcast float %1243 to i32, !dbg !16
  %1253 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1252), !dbg !16
  %1254 = bitcast i32 %1253 to float, !dbg !16
  %1255 = select i1 %.not38, float %1239, float %1254, !dbg !16
  %1256 = select i1 %.not38, float %1254, float %1240, !dbg !16
  %1257 = select i1 %.not38, float %1242, float %1241, !dbg !16
  %1258 = bitcast float %1257 to i32, !dbg !16
  %1259 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1258), !dbg !16
  %1260 = bitcast i32 %1259 to float, !dbg !16
  %1261 = select i1 %.not38, float %1241, float %1260, !dbg !16
  %1262 = select i1 %.not38, float %1260, float %1242, !dbg !16
  %1263 = select i1 %.not39, float %1261, float %1255, !dbg !16
  %1264 = xor i32 %1245, 32, !dbg !16
  %1265 = icmp samesign ult i32 %1264, %1248, !dbg !16
  %1266 = select i1 %1265, i32 %1264, i32 %1245, !dbg !16
  %1267 = shl nuw nsw i32 %1266, 2, !dbg !16
  %1268 = bitcast float %1263 to i32, !dbg !16
  %1269 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1268), !dbg !16
  %1270 = bitcast i32 %1269 to float, !dbg !16
  %1271 = select i1 %.not39, float %1255, float %1270, !dbg !16
  %1272 = select i1 %.not39, float %1270, float %1261, !dbg !16
  %1273 = select i1 %.not39, float %1262, float %1256, !dbg !16
  %1274 = bitcast float %1273 to i32, !dbg !16
  %1275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1274), !dbg !16
  %1276 = bitcast i32 %1275 to float, !dbg !16
  %1277 = select i1 %.not39, float %1256, float %1276, !dbg !16
  %1278 = select i1 %.not39, float %1276, float %1262, !dbg !16
  %1279 = extractvalue [2 x <8 x float>] %52, 0, !dbg !16
  %1280 = extractelement <8 x float> %1279, i64 0, !dbg !16
  %1281 = extractelement <8 x float> %1279, i64 1, !dbg !16
  %1282 = extractelement <8 x float> %1279, i64 2, !dbg !16
  %1283 = extractelement <8 x float> %1279, i64 3, !dbg !16
  %1284 = select i1 %.not38, float %1281, float %1280, !dbg !16
  %1285 = bitcast float %1284 to i32, !dbg !16
  %1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1285), !dbg !16
  %1287 = bitcast i32 %1286 to float, !dbg !16
  %1288 = select i1 %.not38, float %1280, float %1287, !dbg !16
  %1289 = select i1 %.not38, float %1287, float %1281, !dbg !16
  %1290 = select i1 %.not38, float %1283, float %1282, !dbg !16
  %1291 = bitcast float %1290 to i32, !dbg !16
  %1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1291), !dbg !16
  %1293 = bitcast i32 %1292 to float, !dbg !16
  %1294 = select i1 %.not38, float %1282, float %1293, !dbg !16
  %1295 = select i1 %.not38, float %1293, float %1283, !dbg !16
  %1296 = select i1 %.not39, float %1294, float %1288, !dbg !16
  %1297 = bitcast float %1296 to i32, !dbg !16
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1297), !dbg !16
  %1299 = bitcast i32 %1298 to float, !dbg !16
  %1300 = select i1 %.not39, float %1288, float %1299, !dbg !16
  %1301 = select i1 %.not39, float %1299, float %1294, !dbg !16
  %1302 = select i1 %.not39, float %1295, float %1289, !dbg !16
  %1303 = bitcast float %1302 to i32, !dbg !16
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1303), !dbg !16
  %1305 = bitcast i32 %1304 to float, !dbg !16
  %1306 = select i1 %.not39, float %1289, float %1305, !dbg !16
  %1307 = select i1 %.not39, float %1305, float %1295, !dbg !16
  %1308 = extractvalue [2 x <8 x float>] %51, 0, !dbg !16
  %1309 = extractelement <8 x float> %1308, i64 0, !dbg !16
  %1310 = extractelement <8 x float> %1308, i64 1, !dbg !16
  %1311 = extractelement <8 x float> %1308, i64 2, !dbg !16
  %1312 = extractelement <8 x float> %1308, i64 3, !dbg !16
  %1313 = select i1 %.not38, float %1310, float %1309, !dbg !16
  %1314 = bitcast float %1313 to i32, !dbg !16
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1314), !dbg !16
  %1316 = bitcast i32 %1315 to float, !dbg !16
  %1317 = select i1 %.not38, float %1309, float %1316, !dbg !16
  %1318 = select i1 %.not38, float %1316, float %1310, !dbg !16
  %1319 = select i1 %.not38, float %1312, float %1311, !dbg !16
  %1320 = bitcast float %1319 to i32, !dbg !16
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1320), !dbg !16
  %1322 = bitcast i32 %1321 to float, !dbg !16
  %1323 = select i1 %.not38, float %1311, float %1322, !dbg !16
  %1324 = select i1 %.not38, float %1322, float %1312, !dbg !16
  %1325 = select i1 %.not39, float %1323, float %1317, !dbg !16
  %1326 = bitcast float %1325 to i32, !dbg !16
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1326), !dbg !16
  %1328 = bitcast i32 %1327 to float, !dbg !16
  %1329 = select i1 %.not39, float %1317, float %1328, !dbg !16
  %1330 = select i1 %.not39, float %1328, float %1323, !dbg !16
  %1331 = select i1 %.not39, float %1324, float %1318, !dbg !16
  %1332 = bitcast float %1331 to i32, !dbg !16
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1332), !dbg !16
  %1334 = bitcast i32 %1333 to float, !dbg !16
  %1335 = select i1 %.not39, float %1318, float %1334, !dbg !16
  %1336 = select i1 %.not39, float %1334, float %1324, !dbg !16
  %1337 = extractvalue [2 x <8 x float>] %49, 0, !dbg !16
  %1338 = extractelement <8 x float> %1337, i64 0, !dbg !16
  %1339 = extractelement <8 x float> %1337, i64 1, !dbg !16
  %1340 = extractelement <8 x float> %1337, i64 2, !dbg !16
  %1341 = extractelement <8 x float> %1337, i64 3, !dbg !16
  %1342 = select i1 %.not38, float %1339, float %1338, !dbg !16
  %1343 = bitcast float %1342 to i32, !dbg !16
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1343), !dbg !16
  %1345 = bitcast i32 %1344 to float, !dbg !16
  %1346 = select i1 %.not38, float %1338, float %1345, !dbg !16
  %1347 = select i1 %.not38, float %1345, float %1339, !dbg !16
  %1348 = select i1 %.not38, float %1341, float %1340, !dbg !16
  %1349 = bitcast float %1348 to i32, !dbg !16
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1349), !dbg !16
  %1351 = bitcast i32 %1350 to float, !dbg !16
  %1352 = select i1 %.not38, float %1340, float %1351, !dbg !16
  %1353 = select i1 %.not38, float %1351, float %1341, !dbg !16
  %1354 = select i1 %.not39, float %1352, float %1346, !dbg !16
  %1355 = bitcast float %1354 to i32, !dbg !16
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1355), !dbg !16
  %1357 = select i1 %.not39, float %1353, float %1347, !dbg !16
  %1358 = bitcast float %1357 to i32, !dbg !16
  %1359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1358), !dbg !16
  %1360 = extractvalue [2 x <1 x float>] %50, 0, !dbg !19
  %1361 = shufflevector <1 x float> %1360, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1362 = extractelement <8 x float> %1238, i64 4, !dbg !16
  %1363 = extractelement <8 x float> %1238, i64 5, !dbg !16
  %1364 = extractelement <8 x float> %1238, i64 6, !dbg !16
  %1365 = extractelement <8 x float> %1238, i64 7, !dbg !16
  %1366 = select i1 %.not38, float %1363, float %1362, !dbg !16
  %1367 = bitcast float %1366 to i32, !dbg !16
  %1368 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1367), !dbg !16
  %1369 = bitcast i32 %1368 to float, !dbg !16
  %1370 = select i1 %.not38, float %1362, float %1369, !dbg !16
  %1371 = select i1 %.not38, float %1369, float %1363, !dbg !16
  %1372 = select i1 %.not38, float %1365, float %1364, !dbg !16
  %1373 = bitcast float %1372 to i32, !dbg !16
  %1374 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1373), !dbg !16
  %1375 = bitcast i32 %1374 to float, !dbg !16
  %1376 = select i1 %.not38, float %1364, float %1375, !dbg !16
  %1377 = select i1 %.not38, float %1375, float %1365, !dbg !16
  %1378 = select i1 %.not39, float %1376, float %1370, !dbg !16
  %1379 = bitcast float %1378 to i32, !dbg !16
  %1380 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1379), !dbg !16
  %1381 = bitcast i32 %1380 to float, !dbg !16
  %1382 = select i1 %.not39, float %1370, float %1381, !dbg !16
  %1383 = select i1 %.not39, float %1381, float %1376, !dbg !16
  %1384 = select i1 %.not39, float %1377, float %1371, !dbg !16
  %1385 = bitcast float %1384 to i32, !dbg !16
  %1386 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1385), !dbg !16
  %1387 = bitcast i32 %1386 to float, !dbg !16
  %1388 = select i1 %.not39, float %1371, float %1387, !dbg !16
  %1389 = select i1 %.not39, float %1387, float %1377, !dbg !16
  %1390 = extractelement <8 x float> %1279, i64 4, !dbg !16
  %1391 = extractelement <8 x float> %1279, i64 5, !dbg !16
  %1392 = extractelement <8 x float> %1279, i64 6, !dbg !16
  %1393 = extractelement <8 x float> %1279, i64 7, !dbg !16
  %1394 = select i1 %.not38, float %1391, float %1390, !dbg !16
  %1395 = bitcast float %1394 to i32, !dbg !16
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1395), !dbg !16
  %1397 = bitcast i32 %1396 to float, !dbg !16
  %1398 = select i1 %.not38, float %1390, float %1397, !dbg !16
  %1399 = select i1 %.not38, float %1397, float %1391, !dbg !16
  %1400 = select i1 %.not38, float %1393, float %1392, !dbg !16
  %1401 = bitcast float %1400 to i32, !dbg !16
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1401), !dbg !16
  %1403 = bitcast i32 %1402 to float, !dbg !16
  %1404 = select i1 %.not38, float %1392, float %1403, !dbg !16
  %1405 = select i1 %.not38, float %1403, float %1393, !dbg !16
  %1406 = select i1 %.not39, float %1404, float %1398, !dbg !16
  %1407 = bitcast float %1406 to i32, !dbg !16
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1407), !dbg !16
  %1409 = bitcast i32 %1408 to float, !dbg !16
  %1410 = select i1 %.not39, float %1398, float %1409, !dbg !16
  %1411 = select i1 %.not39, float %1409, float %1404, !dbg !16
  %1412 = select i1 %.not39, float %1405, float %1399, !dbg !16
  %1413 = bitcast float %1412 to i32, !dbg !16
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1413), !dbg !16
  %1415 = bitcast i32 %1414 to float, !dbg !16
  %1416 = select i1 %.not39, float %1399, float %1415, !dbg !16
  %1417 = select i1 %.not39, float %1415, float %1405, !dbg !16
  %1418 = extractelement <8 x float> %1308, i64 4, !dbg !16
  %1419 = extractelement <8 x float> %1308, i64 5, !dbg !16
  %1420 = extractelement <8 x float> %1308, i64 6, !dbg !16
  %1421 = extractelement <8 x float> %1308, i64 7, !dbg !16
  %1422 = select i1 %.not38, float %1419, float %1418, !dbg !16
  %1423 = bitcast float %1422 to i32, !dbg !16
  %1424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1423), !dbg !16
  %1425 = bitcast i32 %1424 to float, !dbg !16
  %1426 = select i1 %.not38, float %1418, float %1425, !dbg !16
  %1427 = select i1 %.not38, float %1425, float %1419, !dbg !16
  %1428 = select i1 %.not38, float %1421, float %1420, !dbg !16
  %1429 = bitcast float %1428 to i32, !dbg !16
  %1430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1429), !dbg !16
  %1431 = bitcast i32 %1430 to float, !dbg !16
  %1432 = select i1 %.not38, float %1420, float %1431, !dbg !16
  %1433 = select i1 %.not38, float %1431, float %1421, !dbg !16
  %1434 = select i1 %.not39, float %1432, float %1426, !dbg !16
  %1435 = bitcast float %1434 to i32, !dbg !16
  %1436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1435), !dbg !16
  %1437 = bitcast i32 %1436 to float, !dbg !16
  %1438 = select i1 %.not39, float %1426, float %1437, !dbg !16
  %1439 = select i1 %.not39, float %1437, float %1432, !dbg !16
  %1440 = select i1 %.not39, float %1433, float %1427, !dbg !16
  %1441 = bitcast float %1440 to i32, !dbg !16
  %1442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1441), !dbg !16
  %1443 = bitcast i32 %1442 to float, !dbg !16
  %1444 = select i1 %.not39, float %1427, float %1443, !dbg !16
  %1445 = select i1 %.not39, float %1443, float %1433, !dbg !16
  %1446 = extractelement <8 x float> %1337, i64 4, !dbg !16
  %1447 = extractelement <8 x float> %1337, i64 5, !dbg !16
  %1448 = extractelement <8 x float> %1337, i64 6, !dbg !16
  %1449 = extractelement <8 x float> %1337, i64 7, !dbg !16
  %1450 = select i1 %.not38, float %1447, float %1446, !dbg !16
  %1451 = bitcast float %1450 to i32, !dbg !16
  %1452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1451), !dbg !16
  %1453 = bitcast i32 %1452 to float, !dbg !16
  %1454 = select i1 %.not38, float %1446, float %1453, !dbg !16
  %1455 = select i1 %.not38, float %1453, float %1447, !dbg !16
  %1456 = select i1 %.not38, float %1449, float %1448, !dbg !16
  %1457 = bitcast float %1456 to i32, !dbg !16
  %1458 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1457), !dbg !16
  %1459 = bitcast i32 %1458 to float, !dbg !16
  %1460 = select i1 %.not38, float %1448, float %1459, !dbg !16
  %1461 = select i1 %.not38, float %1459, float %1449, !dbg !16
  %1462 = select i1 %.not39, float %1460, float %1454, !dbg !16
  %1463 = bitcast float %1462 to i32, !dbg !16
  %1464 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1463), !dbg !16
  %1465 = select i1 %.not39, float %1461, float %1455, !dbg !16
  %1466 = bitcast float %1465 to i32, !dbg !16
  %1467 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1466), !dbg !16
  %1468 = extractvalue [2 x <8 x float>] %53, 1, !dbg !16
  %1469 = extractelement <8 x float> %1468, i64 0, !dbg !16
  %1470 = extractelement <8 x float> %1468, i64 1, !dbg !16
  %1471 = extractelement <8 x float> %1468, i64 2, !dbg !16
  %1472 = extractelement <8 x float> %1468, i64 3, !dbg !16
  %1473 = select i1 %.not38, float %1470, float %1469, !dbg !16
  %1474 = bitcast float %1473 to i32, !dbg !16
  %1475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1474), !dbg !16
  %1476 = bitcast i32 %1475 to float, !dbg !16
  %1477 = select i1 %.not38, float %1469, float %1476, !dbg !16
  %1478 = select i1 %.not38, float %1476, float %1470, !dbg !16
  %1479 = select i1 %.not38, float %1472, float %1471, !dbg !16
  %1480 = bitcast float %1479 to i32, !dbg !16
  %1481 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1480), !dbg !16
  %1482 = bitcast i32 %1481 to float, !dbg !16
  %1483 = select i1 %.not38, float %1471, float %1482, !dbg !16
  %1484 = select i1 %.not38, float %1482, float %1472, !dbg !16
  %1485 = select i1 %.not39, float %1483, float %1477, !dbg !16
  %1486 = bitcast float %1485 to i32, !dbg !16
  %1487 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1486), !dbg !16
  %1488 = bitcast i32 %1487 to float, !dbg !16
  %1489 = select i1 %.not39, float %1477, float %1488, !dbg !16
  %1490 = select i1 %.not39, float %1488, float %1483, !dbg !16
  %1491 = select i1 %.not39, float %1484, float %1478, !dbg !16
  %1492 = bitcast float %1491 to i32, !dbg !16
  %1493 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1492), !dbg !16
  %1494 = bitcast i32 %1493 to float, !dbg !16
  %1495 = select i1 %.not39, float %1478, float %1494, !dbg !16
  %1496 = select i1 %.not39, float %1494, float %1484, !dbg !16
  %1497 = extractvalue [2 x <8 x float>] %52, 1, !dbg !16
  %1498 = extractelement <8 x float> %1497, i64 0, !dbg !16
  %1499 = extractelement <8 x float> %1497, i64 1, !dbg !16
  %1500 = extractelement <8 x float> %1497, i64 2, !dbg !16
  %1501 = extractelement <8 x float> %1497, i64 3, !dbg !16
  %1502 = select i1 %.not38, float %1499, float %1498, !dbg !16
  %1503 = bitcast float %1502 to i32, !dbg !16
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1503), !dbg !16
  %1505 = bitcast i32 %1504 to float, !dbg !16
  %1506 = select i1 %.not38, float %1498, float %1505, !dbg !16
  %1507 = select i1 %.not38, float %1505, float %1499, !dbg !16
  %1508 = select i1 %.not38, float %1501, float %1500, !dbg !16
  %1509 = bitcast float %1508 to i32, !dbg !16
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1509), !dbg !16
  %1511 = bitcast i32 %1510 to float, !dbg !16
  %1512 = select i1 %.not38, float %1500, float %1511, !dbg !16
  %1513 = select i1 %.not38, float %1511, float %1501, !dbg !16
  %1514 = select i1 %.not39, float %1512, float %1506, !dbg !16
  %1515 = bitcast float %1514 to i32, !dbg !16
  %1516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1515), !dbg !16
  %1517 = bitcast i32 %1516 to float, !dbg !16
  %1518 = select i1 %.not39, float %1506, float %1517, !dbg !16
  %1519 = select i1 %.not39, float %1517, float %1512, !dbg !16
  %1520 = select i1 %.not39, float %1513, float %1507, !dbg !16
  %1521 = bitcast float %1520 to i32, !dbg !16
  %1522 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1521), !dbg !16
  %1523 = bitcast i32 %1522 to float, !dbg !16
  %1524 = select i1 %.not39, float %1507, float %1523, !dbg !16
  %1525 = select i1 %.not39, float %1523, float %1513, !dbg !16
  %1526 = extractvalue [2 x <8 x float>] %51, 1, !dbg !16
  %1527 = extractelement <8 x float> %1526, i64 0, !dbg !16
  %1528 = extractelement <8 x float> %1526, i64 1, !dbg !16
  %1529 = extractelement <8 x float> %1526, i64 2, !dbg !16
  %1530 = extractelement <8 x float> %1526, i64 3, !dbg !16
  %1531 = select i1 %.not38, float %1528, float %1527, !dbg !16
  %1532 = bitcast float %1531 to i32, !dbg !16
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1532), !dbg !16
  %1534 = bitcast i32 %1533 to float, !dbg !16
  %1535 = select i1 %.not38, float %1527, float %1534, !dbg !16
  %1536 = select i1 %.not38, float %1534, float %1528, !dbg !16
  %1537 = select i1 %.not38, float %1530, float %1529, !dbg !16
  %1538 = bitcast float %1537 to i32, !dbg !16
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1538), !dbg !16
  %1540 = bitcast i32 %1539 to float, !dbg !16
  %1541 = select i1 %.not38, float %1529, float %1540, !dbg !16
  %1542 = select i1 %.not38, float %1540, float %1530, !dbg !16
  %1543 = select i1 %.not39, float %1541, float %1535, !dbg !16
  %1544 = bitcast float %1543 to i32, !dbg !16
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1544), !dbg !16
  %1546 = bitcast i32 %1545 to float, !dbg !16
  %1547 = select i1 %.not39, float %1535, float %1546, !dbg !16
  %1548 = select i1 %.not39, float %1546, float %1541, !dbg !16
  %1549 = select i1 %.not39, float %1542, float %1536, !dbg !16
  %1550 = bitcast float %1549 to i32, !dbg !16
  %1551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1550), !dbg !16
  %1552 = bitcast i32 %1551 to float, !dbg !16
  %1553 = select i1 %.not39, float %1536, float %1552, !dbg !16
  %1554 = select i1 %.not39, float %1552, float %1542, !dbg !16
  %1555 = extractvalue [2 x <8 x float>] %49, 1, !dbg !16
  %1556 = extractelement <8 x float> %1555, i64 0, !dbg !16
  %1557 = extractelement <8 x float> %1555, i64 1, !dbg !16
  %1558 = extractelement <8 x float> %1555, i64 2, !dbg !16
  %1559 = extractelement <8 x float> %1555, i64 3, !dbg !16
  %1560 = select i1 %.not38, float %1557, float %1556, !dbg !16
  %1561 = bitcast float %1560 to i32, !dbg !16
  %1562 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1561), !dbg !16
  %1563 = bitcast i32 %1562 to float, !dbg !16
  %1564 = select i1 %.not38, float %1556, float %1563, !dbg !16
  %1565 = select i1 %.not38, float %1563, float %1557, !dbg !16
  %1566 = select i1 %.not38, float %1559, float %1558, !dbg !16
  %1567 = bitcast float %1566 to i32, !dbg !16
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1567), !dbg !16
  %1569 = bitcast i32 %1568 to float, !dbg !16
  %1570 = select i1 %.not38, float %1558, float %1569, !dbg !16
  %1571 = select i1 %.not38, float %1569, float %1559, !dbg !16
  %1572 = select i1 %.not39, float %1570, float %1564, !dbg !16
  %1573 = bitcast float %1572 to i32, !dbg !16
  %1574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1573), !dbg !16
  %1575 = select i1 %.not39, float %1571, float %1565, !dbg !16
  %1576 = bitcast float %1575 to i32, !dbg !16
  %1577 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1576), !dbg !16
  %1578 = extractvalue [2 x <1 x float>] %50, 1, !dbg !19
  %1579 = shufflevector <1 x float> %1578, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1580 = extractelement <8 x float> %1468, i64 4, !dbg !16
  %1581 = extractelement <8 x float> %1468, i64 5, !dbg !16
  %1582 = extractelement <8 x float> %1468, i64 6, !dbg !16
  %1583 = extractelement <8 x float> %1468, i64 7, !dbg !16
  %1584 = select i1 %.not38, float %1581, float %1580, !dbg !16
  %1585 = bitcast float %1584 to i32, !dbg !16
  %1586 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1585), !dbg !16
  %1587 = bitcast i32 %1586 to float, !dbg !16
  %1588 = select i1 %.not38, float %1580, float %1587, !dbg !16
  %1589 = select i1 %.not38, float %1587, float %1581, !dbg !16
  %1590 = select i1 %.not38, float %1583, float %1582, !dbg !16
  %1591 = bitcast float %1590 to i32, !dbg !16
  %1592 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1591), !dbg !16
  %1593 = bitcast i32 %1592 to float, !dbg !16
  %1594 = select i1 %.not38, float %1582, float %1593, !dbg !16
  %1595 = select i1 %.not38, float %1593, float %1583, !dbg !16
  %1596 = select i1 %.not39, float %1594, float %1588, !dbg !16
  %1597 = bitcast float %1596 to i32, !dbg !16
  %1598 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1597), !dbg !16
  %1599 = bitcast i32 %1598 to float, !dbg !16
  %1600 = select i1 %.not39, float %1588, float %1599, !dbg !16
  %1601 = select i1 %.not39, float %1599, float %1594, !dbg !16
  %1602 = select i1 %.not39, float %1595, float %1589, !dbg !16
  %1603 = bitcast float %1602 to i32, !dbg !16
  %1604 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1603), !dbg !16
  %1605 = bitcast i32 %1604 to float, !dbg !16
  %1606 = select i1 %.not39, float %1589, float %1605, !dbg !16
  %1607 = select i1 %.not39, float %1605, float %1595, !dbg !16
  %1608 = extractelement <8 x float> %1497, i64 4, !dbg !16
  %1609 = extractelement <8 x float> %1497, i64 5, !dbg !16
  %1610 = extractelement <8 x float> %1497, i64 6, !dbg !16
  %1611 = extractelement <8 x float> %1497, i64 7, !dbg !16
  %1612 = select i1 %.not38, float %1609, float %1608, !dbg !16
  %1613 = bitcast float %1612 to i32, !dbg !16
  %1614 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1613), !dbg !16
  %1615 = bitcast i32 %1614 to float, !dbg !16
  %1616 = select i1 %.not38, float %1608, float %1615, !dbg !16
  %1617 = select i1 %.not38, float %1615, float %1609, !dbg !16
  %1618 = select i1 %.not38, float %1611, float %1610, !dbg !16
  %1619 = bitcast float %1618 to i32, !dbg !16
  %1620 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1619), !dbg !16
  %1621 = bitcast i32 %1620 to float, !dbg !16
  %1622 = select i1 %.not38, float %1610, float %1621, !dbg !16
  %1623 = select i1 %.not38, float %1621, float %1611, !dbg !16
  %1624 = select i1 %.not39, float %1622, float %1616, !dbg !16
  %1625 = bitcast float %1624 to i32, !dbg !16
  %1626 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1625), !dbg !16
  %1627 = bitcast i32 %1626 to float, !dbg !16
  %1628 = select i1 %.not39, float %1616, float %1627, !dbg !16
  %1629 = select i1 %.not39, float %1627, float %1622, !dbg !16
  %1630 = select i1 %.not39, float %1623, float %1617, !dbg !16
  %1631 = bitcast float %1630 to i32, !dbg !16
  %1632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1631), !dbg !16
  %1633 = bitcast i32 %1632 to float, !dbg !16
  %1634 = select i1 %.not39, float %1617, float %1633, !dbg !16
  %1635 = select i1 %.not39, float %1633, float %1623, !dbg !16
  %1636 = extractelement <8 x float> %1526, i64 4, !dbg !16
  %1637 = extractelement <8 x float> %1526, i64 5, !dbg !16
  %1638 = extractelement <8 x float> %1526, i64 6, !dbg !16
  %1639 = extractelement <8 x float> %1526, i64 7, !dbg !16
  %1640 = select i1 %.not38, float %1637, float %1636, !dbg !16
  %1641 = bitcast float %1640 to i32, !dbg !16
  %1642 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1641), !dbg !16
  %1643 = bitcast i32 %1642 to float, !dbg !16
  %1644 = select i1 %.not38, float %1636, float %1643, !dbg !16
  %1645 = select i1 %.not38, float %1643, float %1637, !dbg !16
  %1646 = select i1 %.not38, float %1639, float %1638, !dbg !16
  %1647 = bitcast float %1646 to i32, !dbg !16
  %1648 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1647), !dbg !16
  %1649 = bitcast i32 %1648 to float, !dbg !16
  %1650 = select i1 %.not38, float %1638, float %1649, !dbg !16
  %1651 = select i1 %.not38, float %1649, float %1639, !dbg !16
  %1652 = select i1 %.not39, float %1650, float %1644, !dbg !16
  %1653 = bitcast float %1652 to i32, !dbg !16
  %1654 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1653), !dbg !16
  %1655 = bitcast i32 %1654 to float, !dbg !16
  %1656 = select i1 %.not39, float %1644, float %1655, !dbg !16
  %1657 = select i1 %.not39, float %1655, float %1650, !dbg !16
  %1658 = select i1 %.not39, float %1651, float %1645, !dbg !16
  %1659 = bitcast float %1658 to i32, !dbg !16
  %1660 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1659), !dbg !16
  %1661 = bitcast i32 %1660 to float, !dbg !16
  %1662 = select i1 %.not39, float %1645, float %1661, !dbg !16
  %1663 = select i1 %.not39, float %1661, float %1651, !dbg !16
  %1664 = extractelement <8 x float> %1555, i64 4, !dbg !16
  %1665 = extractelement <8 x float> %1555, i64 5, !dbg !16
  %1666 = extractelement <8 x float> %1555, i64 6, !dbg !16
  %1667 = extractelement <8 x float> %1555, i64 7, !dbg !16
  %1668 = select i1 %.not38, float %1665, float %1664, !dbg !16
  %1669 = bitcast float %1668 to i32, !dbg !16
  %1670 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1669), !dbg !16
  %1671 = bitcast i32 %1670 to float, !dbg !16
  %1672 = select i1 %.not38, float %1664, float %1671, !dbg !16
  %1673 = select i1 %.not38, float %1671, float %1665, !dbg !16
  %1674 = select i1 %.not38, float %1667, float %1666, !dbg !16
  %1675 = bitcast float %1674 to i32, !dbg !16
  %1676 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1251, i32 %1675), !dbg !16
  %1677 = bitcast i32 %1676 to float, !dbg !16
  %1678 = select i1 %.not38, float %1666, float %1677, !dbg !16
  %1679 = select i1 %.not38, float %1677, float %1667, !dbg !16
  %1680 = select i1 %.not39, float %1678, float %1672, !dbg !16
  %1681 = bitcast float %1680 to i32, !dbg !16
  %1682 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1681), !dbg !16
  %1683 = select i1 %.not39, float %1679, float %1673, !dbg !16
  %1684 = bitcast float %1683 to i32, !dbg !16
  %1685 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1267, i32 %1684), !dbg !16
  br i1 true, label %1686, label %1750, !dbg !21

1686:                                             ; preds = %1235
  %1687 = bitcast i32 %1682 to float, !dbg !16
  %1688 = select i1 %.not39, float %1672, float %1687, !dbg !16
  %1689 = insertelement <4 x float> poison, float %1688, i64 0, !dbg !19
  %1690 = bitcast i32 %1685 to float, !dbg !16
  %1691 = select i1 %.not39, float %1673, float %1690, !dbg !16
  %1692 = insertelement <4 x float> %1689, float %1691, i64 1, !dbg !19
  %1693 = select i1 %.not39, float %1687, float %1678, !dbg !16
  %1694 = insertelement <4 x float> %1692, float %1693, i64 2, !dbg !19
  %1695 = select i1 %.not39, float %1690, float %1679, !dbg !16
  %1696 = insertelement <4 x float> %1694, float %1695, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1579, !dbg !19
  %normalized = fmul <4 x float> %1696, %row.reciprocal, !dbg !19
  %1697 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1698 = extractelement <4 x half> %1697, i64 3, !dbg !20
  %1699 = extractelement <4 x half> %1697, i64 2, !dbg !20
  %1700 = extractelement <4 x half> %1697, i64 1, !dbg !20
  %1701 = extractelement <4 x half> %1697, i64 0, !dbg !20
  %1702 = bitcast i32 %1574 to float, !dbg !16
  %1703 = select i1 %.not39, float %1564, float %1702, !dbg !16
  %1704 = insertelement <4 x float> poison, float %1703, i64 0, !dbg !19
  %1705 = bitcast i32 %1577 to float, !dbg !16
  %1706 = select i1 %.not39, float %1565, float %1705, !dbg !16
  %1707 = insertelement <4 x float> %1704, float %1706, i64 1, !dbg !19
  %1708 = select i1 %.not39, float %1702, float %1570, !dbg !16
  %1709 = insertelement <4 x float> %1707, float %1708, i64 2, !dbg !19
  %1710 = select i1 %.not39, float %1705, float %1571, !dbg !16
  %1711 = insertelement <4 x float> %1709, float %1710, i64 3, !dbg !19
  %normalized124 = fmul <4 x float> %1711, %row.reciprocal, !dbg !19
  %1712 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %1713 = extractelement <4 x half> %1712, i64 3, !dbg !20
  %1714 = extractelement <4 x half> %1712, i64 2, !dbg !20
  %1715 = extractelement <4 x half> %1712, i64 1, !dbg !20
  %1716 = extractelement <4 x half> %1712, i64 0, !dbg !20
  %1717 = bitcast i32 %1464 to float, !dbg !16
  %1718 = select i1 %.not39, float %1454, float %1717, !dbg !16
  %1719 = insertelement <4 x float> poison, float %1718, i64 0, !dbg !19
  %1720 = bitcast i32 %1467 to float, !dbg !16
  %1721 = select i1 %.not39, float %1455, float %1720, !dbg !16
  %1722 = insertelement <4 x float> %1719, float %1721, i64 1, !dbg !19
  %1723 = select i1 %.not39, float %1717, float %1460, !dbg !16
  %1724 = insertelement <4 x float> %1722, float %1723, i64 2, !dbg !19
  %1725 = select i1 %.not39, float %1720, float %1461, !dbg !16
  %1726 = insertelement <4 x float> %1724, float %1725, i64 3, !dbg !19
  %row.reciprocal125 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1361, !dbg !19
  %normalized126 = fmul <4 x float> %1726, %row.reciprocal125, !dbg !19
  %1727 = fptrunc <4 x float> %normalized126 to <4 x half>, !dbg !20
  %1728 = extractelement <4 x half> %1727, i64 3, !dbg !20
  %1729 = extractelement <4 x half> %1727, i64 2, !dbg !20
  %1730 = extractelement <4 x half> %1727, i64 1, !dbg !20
  %1731 = extractelement <4 x half> %1727, i64 0, !dbg !20
  %1732 = bitcast i32 %1356 to float, !dbg !16
  %1733 = select i1 %.not39, float %1346, float %1732, !dbg !16
  %1734 = insertelement <4 x float> poison, float %1733, i64 0, !dbg !19
  %1735 = bitcast i32 %1359 to float, !dbg !16
  %1736 = select i1 %.not39, float %1347, float %1735, !dbg !16
  %1737 = insertelement <4 x float> %1734, float %1736, i64 1, !dbg !19
  %1738 = select i1 %.not39, float %1732, float %1352, !dbg !16
  %1739 = insertelement <4 x float> %1737, float %1738, i64 2, !dbg !19
  %1740 = select i1 %.not39, float %1735, float %1353, !dbg !16
  %1741 = insertelement <4 x float> %1739, float %1740, i64 3, !dbg !19
  %normalized127 = fmul <4 x float> %1741, %row.reciprocal125, !dbg !19
  %1742 = fptrunc <4 x float> %normalized127 to <4 x half>, !dbg !20
  %1743 = extractelement <4 x half> %1742, i64 3, !dbg !20
  %1744 = extractelement <4 x half> %1742, i64 2, !dbg !20
  %1745 = extractelement <4 x half> %1742, i64 1, !dbg !20
  %1746 = extractelement <4 x half> %1742, i64 0, !dbg !20
  %1747 = add nsw i64 %8, %13, !dbg !21
  %1748 = shl nsw i64 %1747, 19, !dbg !21
  %1749 = and i64 %1748, 16252928, !dbg !21
  %address.thread1064 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local1065 = zext nneg i32 %address.thread1064 to i64, !dbg !14
  %address.local1066 = add nsw i64 %16, %address.local1065, !dbg !21
  %address.local1067 = lshr i64 %address.local1065, 6, !dbg !14
  %address.local1068 = shl nuw nsw i64 %address.local1067, 4, !dbg !21
  %address.local1069 = add nsw i64 %address.local1066, %address.local1068, !dbg !21
  %address.local1070 = and i64 %address.local1065, 240, !dbg !21
  %address.local1071 = sub nsw i64 %address.local1069, %address.local1070, !dbg !21
  %address.local1072 = add nsw i64 %address.local1071, %17, !dbg !21
  %address.local1073 = shl nuw nsw i64 %address.local1072, 7, !dbg !21
  %address.local1074 = add nuw nsw i64 %1749, %address.local1073, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1076 = lshr i64 %address.local1065, 4, !dbg !14
  %address.local1077 = and i64 %address.local1076, 3, !dbg !14
  %address.local1078 = shl nuw nsw i64 %address.local1077, 2, !dbg !21
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1078, !dbg !21
  store half %1746, ptr addrspace(1) %address.local1079, align 2, !dbg !21
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1081 = or disjoint i64 %address.local1078, 1, !dbg !21
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1080, i64 %address.local1081, !dbg !21
  store half %1745, ptr addrspace(1) %address.local1082, align 2, !dbg !21
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1084 = or disjoint i64 %address.local1078, 2, !dbg !21
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1083, i64 %address.local1084, !dbg !21
  store half %1744, ptr addrspace(1) %address.local1085, align 2, !dbg !21
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1087 = or disjoint i64 %address.local1078, 3, !dbg !21
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1086, i64 %address.local1087, !dbg !21
  store half %1743, ptr addrspace(1) %address.local1088, align 2, !dbg !21
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1090 = or disjoint i64 %address.local1078, 16, !dbg !21
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1090, !dbg !21
  store half %1731, ptr addrspace(1) %address.local1091, align 2, !dbg !21
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1093 = or disjoint i64 %address.local1078, 17, !dbg !21
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1092, i64 %address.local1093, !dbg !21
  store half %1730, ptr addrspace(1) %address.local1094, align 2, !dbg !21
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1096 = or disjoint i64 %address.local1078, 18, !dbg !21
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1096, !dbg !21
  store half %1729, ptr addrspace(1) %address.local1097, align 2, !dbg !21
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1099 = or disjoint i64 %address.local1078, 19, !dbg !21
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1098, i64 %address.local1099, !dbg !21
  store half %1728, ptr addrspace(1) %address.local1100, align 2, !dbg !21
  %address.local1101 = shl nsw i64 %address.local1072, 7, !dbg !21
  %address.local1102 = add i64 %address.local1101, 8192, !dbg !21
  %address.local1103 = add nuw nsw i64 %1749, %address.local1102, !dbg !21
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1078, !dbg !21
  store half %1716, ptr addrspace(1) %address.local1105, align 2, !dbg !21
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1106, i64 %address.local1081, !dbg !21
  store half %1715, ptr addrspace(1) %address.local1107, align 2, !dbg !21
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1108, i64 %address.local1084, !dbg !21
  store half %1714, ptr addrspace(1) %address.local1109, align 2, !dbg !21
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1110, i64 %address.local1087, !dbg !21
  store half %1713, ptr addrspace(1) %address.local1111, align 2, !dbg !21
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1090, !dbg !21
  store half %1701, ptr addrspace(1) %address.local1113, align 2, !dbg !21
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1114, i64 %address.local1093, !dbg !21
  store half %1700, ptr addrspace(1) %address.local1115, align 2, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1116, i64 %address.local1096, !dbg !21
  store half %1699, ptr addrspace(1) %address.local1117, align 2, !dbg !21
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1118, i64 %address.local1099, !dbg !21
  store half %1698, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  br label %1750, !dbg !21

1750:                                             ; preds = %1686, %1235
  br i1 true, label %1751, label %1789, !dbg !21

1751:                                             ; preds = %1750
  %1752 = insertelement <4 x float> poison, float %1656, i64 0, !dbg !19
  %1753 = insertelement <4 x float> %1752, float %1662, i64 1, !dbg !19
  %1754 = insertelement <4 x float> %1753, float %1657, i64 2, !dbg !19
  %1755 = insertelement <4 x float> %1754, float %1663, i64 3, !dbg !19
  %row.reciprocal128 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1579, !dbg !19
  %normalized129 = fmul <4 x float> %1755, %row.reciprocal128, !dbg !19
  %1756 = fptrunc <4 x float> %normalized129 to <4 x half>, !dbg !20
  %1757 = extractelement <4 x half> %1756, i64 3, !dbg !20
  %1758 = extractelement <4 x half> %1756, i64 2, !dbg !20
  %1759 = extractelement <4 x half> %1756, i64 1, !dbg !20
  %1760 = extractelement <4 x half> %1756, i64 0, !dbg !20
  %1761 = insertelement <4 x float> poison, float %1547, i64 0, !dbg !19
  %1762 = insertelement <4 x float> %1761, float %1553, i64 1, !dbg !19
  %1763 = insertelement <4 x float> %1762, float %1548, i64 2, !dbg !19
  %1764 = insertelement <4 x float> %1763, float %1554, i64 3, !dbg !19
  %normalized130 = fmul <4 x float> %1764, %row.reciprocal128, !dbg !19
  %1765 = fptrunc <4 x float> %normalized130 to <4 x half>, !dbg !20
  %1766 = extractelement <4 x half> %1765, i64 3, !dbg !20
  %1767 = extractelement <4 x half> %1765, i64 2, !dbg !20
  %1768 = extractelement <4 x half> %1765, i64 1, !dbg !20
  %1769 = extractelement <4 x half> %1765, i64 0, !dbg !20
  %1770 = insertelement <4 x float> poison, float %1438, i64 0, !dbg !19
  %1771 = insertelement <4 x float> %1770, float %1444, i64 1, !dbg !19
  %1772 = insertelement <4 x float> %1771, float %1439, i64 2, !dbg !19
  %1773 = insertelement <4 x float> %1772, float %1445, i64 3, !dbg !19
  %row.reciprocal131 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1361, !dbg !19
  %normalized132 = fmul <4 x float> %1773, %row.reciprocal131, !dbg !19
  %1774 = fptrunc <4 x float> %normalized132 to <4 x half>, !dbg !20
  %1775 = extractelement <4 x half> %1774, i64 3, !dbg !20
  %1776 = extractelement <4 x half> %1774, i64 2, !dbg !20
  %1777 = extractelement <4 x half> %1774, i64 1, !dbg !20
  %1778 = extractelement <4 x half> %1774, i64 0, !dbg !20
  %1779 = insertelement <4 x float> poison, float %1329, i64 0, !dbg !19
  %1780 = insertelement <4 x float> %1779, float %1335, i64 1, !dbg !19
  %1781 = insertelement <4 x float> %1780, float %1330, i64 2, !dbg !19
  %1782 = insertelement <4 x float> %1781, float %1336, i64 3, !dbg !19
  %normalized133 = fmul <4 x float> %1782, %row.reciprocal131, !dbg !19
  %1783 = fptrunc <4 x float> %normalized133 to <4 x half>, !dbg !20
  %1784 = extractelement <4 x half> %1783, i64 3, !dbg !20
  %1785 = extractelement <4 x half> %1783, i64 2, !dbg !20
  %1786 = extractelement <4 x half> %1783, i64 1, !dbg !20
  %1787 = extractelement <4 x half> %1783, i64 0, !dbg !20
  %1788 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread1120 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local1121 = zext nneg i32 %address.thread1120 to i64, !dbg !14
  %address.local1122 = add nsw i64 %16, %address.local1121, !dbg !21
  %address.local1123 = lshr i64 %address.local1121, 6, !dbg !14
  %address.local1124 = shl nuw nsw i64 %address.local1123, 4, !dbg !21
  %address.local1125 = add nsw i64 %address.local1122, %address.local1124, !dbg !21
  %address.local1126 = and i64 %address.local1121, 240, !dbg !21
  %address.local1127 = sub nsw i64 %address.local1125, %address.local1126, !dbg !21
  %address.local1128 = add nsw i64 %address.local1127, %17, !dbg !21
  %address.local1129 = shl nuw nsw i64 %address.local1128, 7, !dbg !21
  %address.local1130 = add nuw nsw i64 %1788, %address.local1129, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1132 = lshr i64 %address.local1121, 4, !dbg !14
  %address.local1133 = and i64 %address.local1132, 3, !dbg !14
  %address.local1134 = shl nuw nsw i64 %address.local1133, 2, !dbg !21
  %address.local1135 = or disjoint i64 %address.local1134, 32, !dbg !21
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1135, !dbg !21
  store half %1787, ptr addrspace(1) %address.local1136, align 2, !dbg !21
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1138 = or disjoint i64 %address.local1134, 33, !dbg !21
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1138, !dbg !21
  store half %1786, ptr addrspace(1) %address.local1139, align 2, !dbg !21
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1141 = or disjoint i64 %address.local1134, 34, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1140, i64 %address.local1141, !dbg !21
  store half %1785, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1144 = or disjoint i64 %address.local1134, 35, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1144, !dbg !21
  store half %1784, ptr addrspace(1) %address.local1145, align 2, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1147 = or disjoint i64 %address.local1134, 48, !dbg !21
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1146, i64 %address.local1147, !dbg !21
  store half %1778, ptr addrspace(1) %address.local1148, align 2, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1150 = or disjoint i64 %address.local1134, 49, !dbg !21
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1150, !dbg !21
  store half %1777, ptr addrspace(1) %address.local1151, align 2, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1153 = or disjoint i64 %address.local1134, 50, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1152, i64 %address.local1153, !dbg !21
  store half %1776, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1156 = or disjoint i64 %address.local1134, 51, !dbg !21
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1156, !dbg !21
  store half %1775, ptr addrspace(1) %address.local1157, align 2, !dbg !21
  %address.local1158 = shl nsw i64 %address.local1128, 7, !dbg !21
  %address.local1159 = add i64 %address.local1158, 8192, !dbg !21
  %address.local1160 = add nuw nsw i64 %1788, %address.local1159, !dbg !21
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1161, i64 %address.local1135, !dbg !21
  store half %1769, ptr addrspace(1) %address.local1162, align 2, !dbg !21
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1163, i64 %address.local1138, !dbg !21
  store half %1768, ptr addrspace(1) %address.local1164, align 2, !dbg !21
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1165, i64 %address.local1141, !dbg !21
  store half %1767, ptr addrspace(1) %address.local1166, align 2, !dbg !21
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1167, i64 %address.local1144, !dbg !21
  store half %1766, ptr addrspace(1) %address.local1168, align 2, !dbg !21
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1169, i64 %address.local1147, !dbg !21
  store half %1760, ptr addrspace(1) %address.local1170, align 2, !dbg !21
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1150, !dbg !21
  store half %1759, ptr addrspace(1) %address.local1172, align 2, !dbg !21
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1153, !dbg !21
  store half %1758, ptr addrspace(1) %address.local1174, align 2, !dbg !21
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1156, !dbg !21
  store half %1757, ptr addrspace(1) %address.local1176, align 2, !dbg !21
  br label %1789, !dbg !21

1789:                                             ; preds = %1751, %1750
  br i1 true, label %1790, label %1828, !dbg !21

1790:                                             ; preds = %1789
  %1791 = insertelement <4 x float> poison, float %1628, i64 0, !dbg !19
  %1792 = insertelement <4 x float> %1791, float %1634, i64 1, !dbg !19
  %1793 = insertelement <4 x float> %1792, float %1629, i64 2, !dbg !19
  %1794 = insertelement <4 x float> %1793, float %1635, i64 3, !dbg !19
  %row.reciprocal134 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1579, !dbg !19
  %normalized135 = fmul <4 x float> %1794, %row.reciprocal134, !dbg !19
  %1795 = fptrunc <4 x float> %normalized135 to <4 x half>, !dbg !20
  %1796 = extractelement <4 x half> %1795, i64 3, !dbg !20
  %1797 = extractelement <4 x half> %1795, i64 2, !dbg !20
  %1798 = extractelement <4 x half> %1795, i64 1, !dbg !20
  %1799 = extractelement <4 x half> %1795, i64 0, !dbg !20
  %1800 = insertelement <4 x float> poison, float %1518, i64 0, !dbg !19
  %1801 = insertelement <4 x float> %1800, float %1524, i64 1, !dbg !19
  %1802 = insertelement <4 x float> %1801, float %1519, i64 2, !dbg !19
  %1803 = insertelement <4 x float> %1802, float %1525, i64 3, !dbg !19
  %normalized136 = fmul <4 x float> %1803, %row.reciprocal134, !dbg !19
  %1804 = fptrunc <4 x float> %normalized136 to <4 x half>, !dbg !20
  %1805 = extractelement <4 x half> %1804, i64 3, !dbg !20
  %1806 = extractelement <4 x half> %1804, i64 2, !dbg !20
  %1807 = extractelement <4 x half> %1804, i64 1, !dbg !20
  %1808 = extractelement <4 x half> %1804, i64 0, !dbg !20
  %1809 = insertelement <4 x float> poison, float %1410, i64 0, !dbg !19
  %1810 = insertelement <4 x float> %1809, float %1416, i64 1, !dbg !19
  %1811 = insertelement <4 x float> %1810, float %1411, i64 2, !dbg !19
  %1812 = insertelement <4 x float> %1811, float %1417, i64 3, !dbg !19
  %row.reciprocal137 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1361, !dbg !19
  %normalized138 = fmul <4 x float> %1812, %row.reciprocal137, !dbg !19
  %1813 = fptrunc <4 x float> %normalized138 to <4 x half>, !dbg !20
  %1814 = extractelement <4 x half> %1813, i64 3, !dbg !20
  %1815 = extractelement <4 x half> %1813, i64 2, !dbg !20
  %1816 = extractelement <4 x half> %1813, i64 1, !dbg !20
  %1817 = extractelement <4 x half> %1813, i64 0, !dbg !20
  %1818 = insertelement <4 x float> poison, float %1300, i64 0, !dbg !19
  %1819 = insertelement <4 x float> %1818, float %1306, i64 1, !dbg !19
  %1820 = insertelement <4 x float> %1819, float %1301, i64 2, !dbg !19
  %1821 = insertelement <4 x float> %1820, float %1307, i64 3, !dbg !19
  %normalized139 = fmul <4 x float> %1821, %row.reciprocal137, !dbg !19
  %1822 = fptrunc <4 x float> %normalized139 to <4 x half>, !dbg !20
  %1823 = extractelement <4 x half> %1822, i64 3, !dbg !20
  %1824 = extractelement <4 x half> %1822, i64 2, !dbg !20
  %1825 = extractelement <4 x half> %1822, i64 1, !dbg !20
  %1826 = extractelement <4 x half> %1822, i64 0, !dbg !20
  %1827 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread1177 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local1178 = zext nneg i32 %address.thread1177 to i64, !dbg !14
  %address.local1179 = add nsw i64 %16, %address.local1178, !dbg !21
  %address.local1180 = lshr i64 %address.local1178, 6, !dbg !14
  %address.local1181 = shl nuw nsw i64 %address.local1180, 4, !dbg !21
  %address.local1182 = add nsw i64 %address.local1179, %address.local1181, !dbg !21
  %address.local1183 = and i64 %address.local1178, 240, !dbg !21
  %address.local1184 = sub nsw i64 %address.local1182, %address.local1183, !dbg !21
  %address.local1185 = add nsw i64 %address.local1184, %17, !dbg !21
  %address.local1186 = shl nuw nsw i64 %address.local1185, 7, !dbg !21
  %address.local1187 = add nuw nsw i64 %1827, %address.local1186, !dbg !21
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1189 = lshr i64 %address.local1178, 4, !dbg !14
  %address.local1190 = and i64 %address.local1189, 3, !dbg !14
  %address.local1191 = shl nuw nsw i64 %address.local1190, 2, !dbg !21
  %address.local1192 = or disjoint i64 %address.local1191, 64, !dbg !21
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1188, i64 %address.local1192, !dbg !21
  store half %1826, ptr addrspace(1) %address.local1193, align 2, !dbg !21
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1195 = or disjoint i64 %address.local1191, 65, !dbg !21
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1194, i64 %address.local1195, !dbg !21
  store half %1825, ptr addrspace(1) %address.local1196, align 2, !dbg !21
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1198 = or disjoint i64 %address.local1191, 66, !dbg !21
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1197, i64 %address.local1198, !dbg !21
  store half %1824, ptr addrspace(1) %address.local1199, align 2, !dbg !21
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1201 = or disjoint i64 %address.local1191, 67, !dbg !21
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1200, i64 %address.local1201, !dbg !21
  store half %1823, ptr addrspace(1) %address.local1202, align 2, !dbg !21
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1204 = or disjoint i64 %address.local1191, 80, !dbg !21
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1203, i64 %address.local1204, !dbg !21
  store half %1817, ptr addrspace(1) %address.local1205, align 2, !dbg !21
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1207 = or disjoint i64 %address.local1191, 81, !dbg !21
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1206, i64 %address.local1207, !dbg !21
  store half %1816, ptr addrspace(1) %address.local1208, align 2, !dbg !21
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1210 = or disjoint i64 %address.local1191, 82, !dbg !21
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1209, i64 %address.local1210, !dbg !21
  store half %1815, ptr addrspace(1) %address.local1211, align 2, !dbg !21
  %address.local1212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1213 = or disjoint i64 %address.local1191, 83, !dbg !21
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1212, i64 %address.local1213, !dbg !21
  store half %1814, ptr addrspace(1) %address.local1214, align 2, !dbg !21
  %address.local1215 = shl nsw i64 %address.local1185, 7, !dbg !21
  %address.local1216 = add i64 %address.local1215, 8192, !dbg !21
  %address.local1217 = add nuw nsw i64 %1827, %address.local1216, !dbg !21
  %address.local1218 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1218, i64 %address.local1192, !dbg !21
  store half %1808, ptr addrspace(1) %address.local1219, align 2, !dbg !21
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1221 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1220, i64 %address.local1195, !dbg !21
  store half %1807, ptr addrspace(1) %address.local1221, align 2, !dbg !21
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1222, i64 %address.local1198, !dbg !21
  store half %1806, ptr addrspace(1) %address.local1223, align 2, !dbg !21
  %address.local1224 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1224, i64 %address.local1201, !dbg !21
  store half %1805, ptr addrspace(1) %address.local1225, align 2, !dbg !21
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1226, i64 %address.local1204, !dbg !21
  store half %1799, ptr addrspace(1) %address.local1227, align 2, !dbg !21
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1228, i64 %address.local1207, !dbg !21
  store half %1798, ptr addrspace(1) %address.local1229, align 2, !dbg !21
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1230, i64 %address.local1210, !dbg !21
  store half %1797, ptr addrspace(1) %address.local1231, align 2, !dbg !21
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1232, i64 %address.local1213, !dbg !21
  store half %1796, ptr addrspace(1) %address.local1233, align 2, !dbg !21
  br label %1828, !dbg !21

1828:                                             ; preds = %1790, %1789
  br i1 true, label %1829, label %1867, !dbg !21

1829:                                             ; preds = %1828
  %1830 = insertelement <4 x float> poison, float %1600, i64 0, !dbg !19
  %1831 = insertelement <4 x float> %1830, float %1606, i64 1, !dbg !19
  %1832 = insertelement <4 x float> %1831, float %1601, i64 2, !dbg !19
  %1833 = insertelement <4 x float> %1832, float %1607, i64 3, !dbg !19
  %row.reciprocal140 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1579, !dbg !19
  %normalized141 = fmul <4 x float> %1833, %row.reciprocal140, !dbg !19
  %1834 = fptrunc <4 x float> %normalized141 to <4 x half>, !dbg !20
  %1835 = extractelement <4 x half> %1834, i64 3, !dbg !20
  %1836 = extractelement <4 x half> %1834, i64 2, !dbg !20
  %1837 = extractelement <4 x half> %1834, i64 1, !dbg !20
  %1838 = extractelement <4 x half> %1834, i64 0, !dbg !20
  %1839 = insertelement <4 x float> poison, float %1489, i64 0, !dbg !19
  %1840 = insertelement <4 x float> %1839, float %1495, i64 1, !dbg !19
  %1841 = insertelement <4 x float> %1840, float %1490, i64 2, !dbg !19
  %1842 = insertelement <4 x float> %1841, float %1496, i64 3, !dbg !19
  %normalized142 = fmul <4 x float> %1842, %row.reciprocal140, !dbg !19
  %1843 = fptrunc <4 x float> %normalized142 to <4 x half>, !dbg !20
  %1844 = extractelement <4 x half> %1843, i64 3, !dbg !20
  %1845 = extractelement <4 x half> %1843, i64 2, !dbg !20
  %1846 = extractelement <4 x half> %1843, i64 1, !dbg !20
  %1847 = extractelement <4 x half> %1843, i64 0, !dbg !20
  %1848 = insertelement <4 x float> poison, float %1382, i64 0, !dbg !19
  %1849 = insertelement <4 x float> %1848, float %1388, i64 1, !dbg !19
  %1850 = insertelement <4 x float> %1849, float %1383, i64 2, !dbg !19
  %1851 = insertelement <4 x float> %1850, float %1389, i64 3, !dbg !19
  %row.reciprocal143 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1361, !dbg !19
  %normalized144 = fmul <4 x float> %1851, %row.reciprocal143, !dbg !19
  %1852 = fptrunc <4 x float> %normalized144 to <4 x half>, !dbg !20
  %1853 = extractelement <4 x half> %1852, i64 3, !dbg !20
  %1854 = extractelement <4 x half> %1852, i64 2, !dbg !20
  %1855 = extractelement <4 x half> %1852, i64 1, !dbg !20
  %1856 = extractelement <4 x half> %1852, i64 0, !dbg !20
  %1857 = insertelement <4 x float> poison, float %1271, i64 0, !dbg !19
  %1858 = insertelement <4 x float> %1857, float %1277, i64 1, !dbg !19
  %1859 = insertelement <4 x float> %1858, float %1272, i64 2, !dbg !19
  %1860 = insertelement <4 x float> %1859, float %1278, i64 3, !dbg !19
  %normalized145 = fmul <4 x float> %1860, %row.reciprocal143, !dbg !19
  %1861 = fptrunc <4 x float> %normalized145 to <4 x half>, !dbg !20
  %1862 = extractelement <4 x half> %1861, i64 3, !dbg !20
  %1863 = extractelement <4 x half> %1861, i64 2, !dbg !20
  %1864 = extractelement <4 x half> %1861, i64 1, !dbg !20
  %1865 = extractelement <4 x half> %1861, i64 0, !dbg !20
  %1866 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread1234 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local1235 = zext nneg i32 %address.thread1234 to i64, !dbg !14
  %address.local1236 = add nsw i64 %16, %address.local1235, !dbg !21
  %address.local1237 = lshr i64 %address.local1235, 6, !dbg !14
  %address.local1238 = shl nuw nsw i64 %address.local1237, 4, !dbg !21
  %address.local1239 = add nsw i64 %address.local1236, %address.local1238, !dbg !21
  %address.local1240 = and i64 %address.local1235, 240, !dbg !21
  %address.local1241 = sub nsw i64 %address.local1239, %address.local1240, !dbg !21
  %address.local1242 = add nsw i64 %address.local1241, %17, !dbg !21
  %address.local1243 = shl nuw nsw i64 %address.local1242, 7, !dbg !21
  %address.local1244 = add nuw nsw i64 %1866, %address.local1243, !dbg !21
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1246 = lshr i64 %address.local1235, 4, !dbg !14
  %address.local1247 = and i64 %address.local1246, 3, !dbg !14
  %address.local1248 = shl nuw nsw i64 %address.local1247, 2, !dbg !21
  %address.local1249 = or disjoint i64 %address.local1248, 96, !dbg !21
  %address.local1250 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1245, i64 %address.local1249, !dbg !21
  store half %1865, ptr addrspace(1) %address.local1250, align 2, !dbg !21
  %address.local1251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1252 = or disjoint i64 %address.local1248, 97, !dbg !21
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1251, i64 %address.local1252, !dbg !21
  store half %1864, ptr addrspace(1) %address.local1253, align 2, !dbg !21
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1255 = or disjoint i64 %address.local1248, 98, !dbg !21
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1254, i64 %address.local1255, !dbg !21
  store half %1863, ptr addrspace(1) %address.local1256, align 2, !dbg !21
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1258 = or disjoint i64 %address.local1248, 99, !dbg !21
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1258, !dbg !21
  store half %1862, ptr addrspace(1) %address.local1259, align 2, !dbg !21
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1261 = or disjoint i64 %address.local1248, 112, !dbg !21
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1260, i64 %address.local1261, !dbg !21
  store half %1856, ptr addrspace(1) %address.local1262, align 2, !dbg !21
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1264 = or disjoint i64 %address.local1248, 113, !dbg !21
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1264, !dbg !21
  store half %1855, ptr addrspace(1) %address.local1265, align 2, !dbg !21
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1267 = or disjoint i64 %address.local1248, 114, !dbg !21
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1266, i64 %address.local1267, !dbg !21
  store half %1854, ptr addrspace(1) %address.local1268, align 2, !dbg !21
  %address.local1269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1270 = or disjoint i64 %address.local1248, 115, !dbg !21
  %address.local1271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1269, i64 %address.local1270, !dbg !21
  store half %1853, ptr addrspace(1) %address.local1271, align 2, !dbg !21
  %address.local1272 = shl nsw i64 %address.local1242, 7, !dbg !21
  %address.local1273 = add i64 %address.local1272, 8192, !dbg !21
  %address.local1274 = add nuw nsw i64 %1866, %address.local1273, !dbg !21
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1275, i64 %address.local1249, !dbg !21
  store half %1847, ptr addrspace(1) %address.local1276, align 2, !dbg !21
  %address.local1277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1277, i64 %address.local1252, !dbg !21
  store half %1846, ptr addrspace(1) %address.local1278, align 2, !dbg !21
  %address.local1279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1279, i64 %address.local1255, !dbg !21
  store half %1845, ptr addrspace(1) %address.local1280, align 2, !dbg !21
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1281, i64 %address.local1258, !dbg !21
  store half %1844, ptr addrspace(1) %address.local1282, align 2, !dbg !21
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1283, i64 %address.local1261, !dbg !21
  store half %1838, ptr addrspace(1) %address.local1284, align 2, !dbg !21
  %address.local1285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1285, i64 %address.local1264, !dbg !21
  store half %1837, ptr addrspace(1) %address.local1286, align 2, !dbg !21
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1287, i64 %address.local1267, !dbg !21
  store half %1836, ptr addrspace(1) %address.local1288, align 2, !dbg !21
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1289, i64 %address.local1270, !dbg !21
  store half %1835, ptr addrspace(1) %address.local1290, align 2, !dbg !21
  br label %1867, !dbg !21

1867:                                             ; preds = %1829, %1828
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

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) captures(none), ptr addrspace(3) captures(none), i32 immarg, i32 immarg, i32 immarg) #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.lds_reuse"="8" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn }
attributes #7 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind memory(none) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!33}

!0 = !DIFile(filename: "test_input_spidD.mlir", directory: "3rd/deepgengraph/test")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!8 = !DILocation(line: 69, column: 26, scope: !4)
!9 = !DILocation(line: 92, column: 20, scope: !4)
!11 = !DILocation(line: 46, column: 17, scope: !4)
!13 = !DILocation(line: 66, column: 22, scope: !4)
!14 = !DILocation(line: 68, column: 24, scope: !4)
!15 = !DILocation(line: 73, column: 19, scope: !4)
!16 = !DILocation(line: 56, column: 24, scope: !4)
!17 = !DILocation(line: 67, column: 22, scope: !4)
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
