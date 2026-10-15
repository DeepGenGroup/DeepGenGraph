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

47:                                               ; preds = %449, %4
  %48 = phi i64 [ %1244, %449 ], [ 0, %4 ], !dbg !16
  %49 = phi [2 x <8 x float>] [ %1208, %449 ], [ zeroinitializer, %4 ], !dbg !16
  %50 = phi [2 x <1 x float>] [ %820, %449 ], [ zeroinitializer, %4 ], !dbg !16
  %51 = phi [2 x <8 x float>] [ %1215, %449 ], [ zeroinitializer, %4 ], !dbg !16
  %52 = phi [2 x <8 x float>] [ %1221, %449 ], [ zeroinitializer, %4 ], !dbg !16
  %53 = phi [2 x <8 x float>] [ %1243, %449 ], [ zeroinitializer, %4 ], !dbg !16
  %.not = icmp sgt i64 %48, %6, !dbg !16
  br i1 %.not, label %1245, label %54, !dbg !16

54:                                               ; preds = %47
  %55 = shl nsw i64 %48, 7, !dbg !16
  %56 = lshr i64 %48, 5, !dbg !17
  %57 = shl i64 %56, 12, !dbg !17
  %58 = sub i64 %55, %57, !dbg !17
  br label %59, !dbg !22

59:                                               ; preds = %63, %54
  %60 = phi i64 [ %270, %63 ], [ 0, %54 ], !dbg !22
  %61 = phi [2 x <32 x float>] [ %269, %63 ], [ zeroinitializer, %54 ], !dbg !22
  %62 = icmp samesign ult i64 %60, 4, !dbg !22
  br i1 %62, label %63, label %271, !dbg !22

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
  %195 = load <8 x half>, ptr addrspace(3) %address.local631, align 2, !dbg !8
  %196 = shufflevector <4 x float> %194, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %197 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %180, <4 x float> %159) #9, !dbg !8
  %198 = load <8 x half>, ptr addrspace(3) %address.local671, align 2, !dbg !8
  %199 = shufflevector <8 x half> %195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %200 = shufflevector <8 x half> %195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %201 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %181, <4 x float> %197) #9, !dbg !8
  %202 = shufflevector <8 x half> %198, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %203 = shufflevector <8 x half> %198, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %204 = load <8 x half>, ptr addrspace(3) %address.local693, align 2, !dbg !8
  %205 = shufflevector <4 x float> %201, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %206 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %186, <4 x float> %160) #9, !dbg !8
  %207 = load <8 x half>, ptr addrspace(3) %address.local729, align 2, !dbg !8
  %208 = shufflevector <8 x half> %204, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %209 = shufflevector <8 x half> %204, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %210 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %189, <4 x float> %206) #9, !dbg !8
  %211 = shufflevector <8 x half> %207, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %212 = shufflevector <8 x half> %207, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %213 = load <8 x half>, ptr addrspace(3) %address.local731, align 2, !dbg !8
  %214 = shufflevector <4 x float> %210, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %215 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %190, <4 x float> %161) #9, !dbg !8
  %216 = load <8 x half>, ptr addrspace(3) %address.local774, align 2, !dbg !8
  %217 = shufflevector <8 x half> %213, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %218 = shufflevector <8 x half> %213, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %219 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %191, <4 x float> %215) #9, !dbg !8
  %220 = shufflevector <8 x half> %216, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %221 = shufflevector <8 x half> %216, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %222 = load <8 x half>, ptr addrspace(3) %address.local795, align 2, !dbg !8
  %223 = shufflevector <4 x float> %219, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %224 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %192, <4 x float> %162) #9, !dbg !8
  %225 = load <8 x half>, ptr addrspace(3) %address.local838, align 2, !dbg !8
  %226 = shufflevector <8 x half> %222, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %227 = shufflevector <8 x half> %222, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %228 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %193, <4 x float> %224) #9, !dbg !8
  %229 = shufflevector <8 x half> %225, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %230 = shufflevector <8 x half> %225, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %231 = shufflevector <4 x float> %228, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %232 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %199, <4 x float> %164) #9, !dbg !8
  %233 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %200, <4 x float> %232) #9, !dbg !8
  %234 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %202, <4 x float> %165) #9, !dbg !8
  %235 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %203, <4 x float> %234) #9, !dbg !8
  %236 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %208, <4 x float> %166) #9, !dbg !8
  %237 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %209, <4 x float> %236) #9, !dbg !8
  %238 = shufflevector <4 x float> %237, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %239 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %211, <4 x float> %167) #9, !dbg !8
  %240 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %212, <4 x float> %239) #9, !dbg !8
  %241 = shufflevector <4 x float> %240, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %242 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %217, <4 x float> %168) #9, !dbg !8
  %243 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %218, <4 x float> %242) #9, !dbg !8
  %244 = shufflevector <4 x float> %243, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %245 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %220, <4 x float> %169) #9, !dbg !8
  %246 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %221, <4 x float> %245) #9, !dbg !8
  %247 = shufflevector <4 x float> %246, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %248 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %226, <4 x float> %170) #9, !dbg !8
  %249 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %227, <4 x float> %248) #9, !dbg !8
  %250 = shufflevector <4 x float> %249, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %251 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %229, <4 x float> %171) #9, !dbg !8
  %252 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %230, <4 x float> %251) #9, !dbg !8
  %253 = shufflevector <4 x float> %252, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %254 = shufflevector <4 x float> %152, <4 x float> %158, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %255 = shufflevector <32 x float> %254, <32 x float> %187, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %256 = shufflevector <32 x float> %255, <32 x float> %196, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %257 = shufflevector <32 x float> %256, <32 x float> %205, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %258 = shufflevector <32 x float> %257, <32 x float> %214, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %259 = shufflevector <32 x float> %258, <32 x float> %223, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %260 = shufflevector <32 x float> %259, <32 x float> %231, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %261 = insertvalue [2 x <32 x float>] poison, <32 x float> %260, 0, !dbg !22
  %262 = shufflevector <4 x float> %233, <4 x float> %235, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %263 = shufflevector <32 x float> %262, <32 x float> %238, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %264 = shufflevector <32 x float> %263, <32 x float> %241, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %265 = shufflevector <32 x float> %264, <32 x float> %244, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %266 = shufflevector <32 x float> %265, <32 x float> %247, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %267 = shufflevector <32 x float> %266, <32 x float> %250, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %268 = shufflevector <32 x float> %267, <32 x float> %253, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %269 = insertvalue [2 x <32 x float>] %261, <32 x float> %268, 1, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %270 = add nuw nsw i64 %60, 1, !dbg !22
  br label %59, !dbg !22

271:                                              ; preds = %59
  %272 = or disjoint i64 %55, 127, !dbg !15
  %273 = icmp uge i64 %9, %272, !dbg !15
  %274 = and i1 %27, %273, !dbg !15
  br i1 %274, label %275, label %276, !dbg !15

275:                                              ; preds = %271
  br label %447, !dbg !15

276:                                              ; preds = %271
  %277 = add nsw i64 %9, %20, !dbg !15
  %278 = shl nuw nsw i64 %22, 4, !dbg !15
  %279 = add nsw i64 %277, %278, !dbg !15
  %280 = and i32 %19, 240, !dbg !15
  %index.compact = zext nneg i32 %280 to i64, !dbg !15
  %281 = sub nsw i64 %279, %index.compact, !dbg !15
  %282 = shl nuw nsw i64 %24, 4, !dbg !15
  %283 = or disjoint i64 %55, %282, !dbg !15
  %284 = add nsw i64 %281, 1, !dbg !23
  %.not40 = icmp ugt i64 %284, %283, !dbg !24
  %285 = select i1 %.not40, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %286 = or disjoint i64 %283, 1, !dbg !15
  %.not41 = icmp ugt i64 %284, %286, !dbg !24
  %287 = select i1 %.not41, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %288 = or disjoint i64 %283, 2, !dbg !15
  %.not42 = icmp ugt i64 %284, %288, !dbg !24
  %289 = select i1 %.not42, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %290 = or disjoint i64 %283, 3, !dbg !15
  %.not43 = icmp ugt i64 %284, %290, !dbg !24
  %291 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %292 = or disjoint i64 %283, 4, !dbg !15
  %.not44 = icmp ugt i64 %284, %292, !dbg !24
  %293 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %294 = or disjoint i64 %283, 5, !dbg !15
  %.not45 = icmp ugt i64 %284, %294, !dbg !24
  %295 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %296 = or disjoint i64 %283, 6, !dbg !15
  %.not46 = icmp ugt i64 %284, %296, !dbg !24
  %297 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %298 = or disjoint i64 %283, 7, !dbg !15
  %.not47 = icmp ugt i64 %284, %298, !dbg !24
  %299 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %300 = or disjoint i64 %283, 8, !dbg !15
  %.not48 = icmp ugt i64 %284, %300, !dbg !24
  %301 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %302 = or disjoint i64 %283, 9, !dbg !15
  %.not49 = icmp ugt i64 %284, %302, !dbg !24
  %303 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %304 = or disjoint i64 %283, 10, !dbg !15
  %.not50 = icmp ugt i64 %284, %304, !dbg !24
  %305 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %306 = or disjoint i64 %283, 11, !dbg !15
  %.not51 = icmp ugt i64 %284, %306, !dbg !24
  %307 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %308 = or disjoint i64 %283, 12, !dbg !15
  %.not52 = icmp ugt i64 %284, %308, !dbg !24
  %309 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %310 = or disjoint i64 %283, 13, !dbg !15
  %.not53 = icmp ugt i64 %284, %310, !dbg !24
  %311 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %312 = or disjoint i64 %283, 14, !dbg !15
  %.not54 = icmp ugt i64 %284, %312, !dbg !24
  %313 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %314 = or disjoint i64 %283, 15, !dbg !15
  %.not55 = icmp ugt i64 %284, %314, !dbg !24
  %315 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %316 = or disjoint i64 %283, 64, !dbg !15
  %.not56 = icmp ugt i64 %284, %316, !dbg !24
  %317 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %318 = or disjoint i64 %283, 65, !dbg !15
  %.not57 = icmp ugt i64 %284, %318, !dbg !24
  %319 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %320 = or disjoint i64 %283, 66, !dbg !15
  %.not58 = icmp ugt i64 %284, %320, !dbg !24
  %321 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %322 = or disjoint i64 %283, 67, !dbg !15
  %.not59 = icmp ugt i64 %284, %322, !dbg !24
  %323 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %324 = or disjoint i64 %283, 68, !dbg !15
  %.not60 = icmp ugt i64 %284, %324, !dbg !24
  %325 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %326 = or disjoint i64 %283, 69, !dbg !15
  %.not61 = icmp ugt i64 %284, %326, !dbg !24
  %327 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %328 = or disjoint i64 %283, 70, !dbg !15
  %.not62 = icmp ugt i64 %284, %328, !dbg !24
  %329 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %330 = or disjoint i64 %283, 71, !dbg !15
  %.not63 = icmp ugt i64 %284, %330, !dbg !24
  %331 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %332 = or disjoint i64 %283, 72, !dbg !15
  %.not64 = icmp ugt i64 %284, %332, !dbg !24
  %333 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %334 = or disjoint i64 %283, 73, !dbg !15
  %.not65 = icmp ugt i64 %284, %334, !dbg !24
  %335 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %336 = or disjoint i64 %283, 74, !dbg !15
  %.not66 = icmp ugt i64 %284, %336, !dbg !24
  %337 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %338 = or disjoint i64 %283, 75, !dbg !15
  %.not67 = icmp ugt i64 %284, %338, !dbg !24
  %339 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %340 = or disjoint i64 %283, 76, !dbg !15
  %.not68 = icmp ugt i64 %284, %340, !dbg !24
  %341 = select i1 %.not68, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %342 = or disjoint i64 %283, 77, !dbg !15
  %.not69 = icmp ugt i64 %284, %342, !dbg !24
  %343 = select i1 %.not69, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %344 = or disjoint i64 %283, 78, !dbg !15
  %.not70 = icmp ugt i64 %284, %344, !dbg !24
  %345 = select i1 %.not70, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %346 = or disjoint i64 %283, 79, !dbg !15
  %.not71 = icmp ugt i64 %284, %346, !dbg !24
  %347 = select i1 %.not71, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %348 = add nsw i64 %281, 65, !dbg !23
  %.not72 = icmp ugt i64 %348, %283, !dbg !24
  %349 = select i1 %.not72, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not73 = icmp ugt i64 %348, %286, !dbg !24
  %350 = select i1 %.not73, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not74 = icmp ugt i64 %348, %288, !dbg !24
  %351 = select i1 %.not74, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not75 = icmp ugt i64 %348, %290, !dbg !24
  %352 = select i1 %.not75, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not76 = icmp ugt i64 %348, %292, !dbg !24
  %353 = select i1 %.not76, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not77 = icmp ugt i64 %348, %294, !dbg !24
  %354 = select i1 %.not77, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not78 = icmp ugt i64 %348, %296, !dbg !24
  %355 = select i1 %.not78, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not79 = icmp ugt i64 %348, %298, !dbg !24
  %356 = select i1 %.not79, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not80 = icmp ugt i64 %348, %300, !dbg !24
  %357 = select i1 %.not80, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not81 = icmp ugt i64 %348, %302, !dbg !24
  %358 = select i1 %.not81, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not82 = icmp ugt i64 %348, %304, !dbg !24
  %359 = select i1 %.not82, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not83 = icmp ugt i64 %348, %306, !dbg !24
  %360 = select i1 %.not83, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not84 = icmp ugt i64 %348, %308, !dbg !24
  %361 = select i1 %.not84, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not85 = icmp ugt i64 %348, %310, !dbg !24
  %362 = select i1 %.not85, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not86 = icmp ugt i64 %348, %312, !dbg !24
  %363 = select i1 %.not86, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not87 = icmp ugt i64 %348, %314, !dbg !24
  %364 = select i1 %.not87, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not88 = icmp ugt i64 %348, %316, !dbg !24
  %365 = select i1 %.not88, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not89 = icmp ugt i64 %348, %318, !dbg !24
  %366 = select i1 %.not89, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not90 = icmp ugt i64 %348, %320, !dbg !24
  %367 = select i1 %.not90, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not91 = icmp ugt i64 %348, %322, !dbg !24
  %368 = select i1 %.not91, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not92 = icmp ugt i64 %348, %324, !dbg !24
  %369 = select i1 %.not92, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not93 = icmp ugt i64 %348, %326, !dbg !24
  %370 = select i1 %.not93, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not94 = icmp ugt i64 %348, %328, !dbg !24
  %371 = select i1 %.not94, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not95 = icmp ugt i64 %348, %330, !dbg !24
  %372 = select i1 %.not95, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not96 = icmp ugt i64 %348, %332, !dbg !24
  %373 = select i1 %.not96, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not97 = icmp ugt i64 %348, %334, !dbg !24
  %374 = select i1 %.not97, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not98 = icmp ugt i64 %348, %336, !dbg !24
  %375 = select i1 %.not98, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not99 = icmp ugt i64 %348, %338, !dbg !24
  %376 = select i1 %.not99, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not100 = icmp ugt i64 %348, %340, !dbg !24
  %377 = select i1 %.not100, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not101 = icmp ugt i64 %348, %342, !dbg !24
  %378 = select i1 %.not101, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not102 = icmp ugt i64 %348, %344, !dbg !24
  %379 = select i1 %.not102, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not103 = icmp ugt i64 %348, %346, !dbg !24
  %380 = select i1 %.not103, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %381 = insertelement <32 x float> poison, float %285, i64 0, !dbg !15
  %382 = insertelement <32 x float> %381, float %287, i64 1, !dbg !15
  %383 = insertelement <32 x float> %382, float %289, i64 2, !dbg !15
  %384 = insertelement <32 x float> %383, float %291, i64 3, !dbg !15
  %385 = insertelement <32 x float> %384, float %293, i64 4, !dbg !15
  %386 = insertelement <32 x float> %385, float %295, i64 5, !dbg !15
  %387 = insertelement <32 x float> %386, float %297, i64 6, !dbg !15
  %388 = insertelement <32 x float> %387, float %299, i64 7, !dbg !15
  %389 = insertelement <32 x float> %388, float %301, i64 8, !dbg !15
  %390 = insertelement <32 x float> %389, float %303, i64 9, !dbg !15
  %391 = insertelement <32 x float> %390, float %305, i64 10, !dbg !15
  %392 = insertelement <32 x float> %391, float %307, i64 11, !dbg !15
  %393 = insertelement <32 x float> %392, float %309, i64 12, !dbg !15
  %394 = insertelement <32 x float> %393, float %311, i64 13, !dbg !15
  %395 = insertelement <32 x float> %394, float %313, i64 14, !dbg !15
  %396 = insertelement <32 x float> %395, float %315, i64 15, !dbg !15
  %397 = insertelement <32 x float> %396, float %317, i64 16, !dbg !15
  %398 = insertelement <32 x float> %397, float %319, i64 17, !dbg !15
  %399 = insertelement <32 x float> %398, float %321, i64 18, !dbg !15
  %400 = insertelement <32 x float> %399, float %323, i64 19, !dbg !15
  %401 = insertelement <32 x float> %400, float %325, i64 20, !dbg !15
  %402 = insertelement <32 x float> %401, float %327, i64 21, !dbg !15
  %403 = insertelement <32 x float> %402, float %329, i64 22, !dbg !15
  %404 = insertelement <32 x float> %403, float %331, i64 23, !dbg !15
  %405 = insertelement <32 x float> %404, float %333, i64 24, !dbg !15
  %406 = insertelement <32 x float> %405, float %335, i64 25, !dbg !15
  %407 = insertelement <32 x float> %406, float %337, i64 26, !dbg !15
  %408 = insertelement <32 x float> %407, float %339, i64 27, !dbg !15
  %409 = insertelement <32 x float> %408, float %341, i64 28, !dbg !15
  %410 = insertelement <32 x float> %409, float %343, i64 29, !dbg !15
  %411 = insertelement <32 x float> %410, float %345, i64 30, !dbg !15
  %412 = insertelement <32 x float> %411, float %347, i64 31, !dbg !15
  %413 = insertvalue [2 x <32 x float>] poison, <32 x float> %412, 0, !dbg !15
  %414 = insertelement <32 x float> poison, float %349, i64 0, !dbg !15
  %415 = insertelement <32 x float> %414, float %350, i64 1, !dbg !15
  %416 = insertelement <32 x float> %415, float %351, i64 2, !dbg !15
  %417 = insertelement <32 x float> %416, float %352, i64 3, !dbg !15
  %418 = insertelement <32 x float> %417, float %353, i64 4, !dbg !15
  %419 = insertelement <32 x float> %418, float %354, i64 5, !dbg !15
  %420 = insertelement <32 x float> %419, float %355, i64 6, !dbg !15
  %421 = insertelement <32 x float> %420, float %356, i64 7, !dbg !15
  %422 = insertelement <32 x float> %421, float %357, i64 8, !dbg !15
  %423 = insertelement <32 x float> %422, float %358, i64 9, !dbg !15
  %424 = insertelement <32 x float> %423, float %359, i64 10, !dbg !15
  %425 = insertelement <32 x float> %424, float %360, i64 11, !dbg !15
  %426 = insertelement <32 x float> %425, float %361, i64 12, !dbg !15
  %427 = insertelement <32 x float> %426, float %362, i64 13, !dbg !15
  %428 = insertelement <32 x float> %427, float %363, i64 14, !dbg !15
  %429 = insertelement <32 x float> %428, float %364, i64 15, !dbg !15
  %430 = insertelement <32 x float> %429, float %365, i64 16, !dbg !15
  %431 = insertelement <32 x float> %430, float %366, i64 17, !dbg !15
  %432 = insertelement <32 x float> %431, float %367, i64 18, !dbg !15
  %433 = insertelement <32 x float> %432, float %368, i64 19, !dbg !15
  %434 = insertelement <32 x float> %433, float %369, i64 20, !dbg !15
  %435 = insertelement <32 x float> %434, float %370, i64 21, !dbg !15
  %436 = insertelement <32 x float> %435, float %371, i64 22, !dbg !15
  %437 = insertelement <32 x float> %436, float %372, i64 23, !dbg !15
  %438 = insertelement <32 x float> %437, float %373, i64 24, !dbg !15
  %439 = insertelement <32 x float> %438, float %374, i64 25, !dbg !15
  %440 = insertelement <32 x float> %439, float %375, i64 26, !dbg !15
  %441 = insertelement <32 x float> %440, float %376, i64 27, !dbg !15
  %442 = insertelement <32 x float> %441, float %377, i64 28, !dbg !15
  %443 = insertelement <32 x float> %442, float %378, i64 29, !dbg !15
  %444 = insertelement <32 x float> %443, float %379, i64 30, !dbg !15
  %445 = insertelement <32 x float> %444, float %380, i64 31, !dbg !15
  %446 = insertvalue [2 x <32 x float>] %413, <32 x float> %445, 1, !dbg !15
  br label %447, !dbg !15

447:                                              ; preds = %275, %276
  %448 = phi [2 x <32 x float>] [ %446, %276 ], [ zeroinitializer, %275 ], !dbg !15
  br label %449, !dbg !15

449:                                              ; preds = %447
  %450 = extractvalue [2 x <32 x float>] %61, 0, !dbg !26
  %451 = extractvalue [2 x <32 x float>] %448, 0, !dbg !26
  %452 = shufflevector <32 x float> %450, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %453 = shufflevector <32 x float> %451, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %454 = fadd <4 x float> %452, %453, !dbg !26
  %455 = extractelement <4 x float> %454, i64 0, !dbg !27
  %456 = add nsw i64 %56, %8, !dbg !9
  %457 = shl i64 %456, 19, !dbg !9
  %458 = and i64 %457, 16252928, !dbg !9
  %459 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %458, !dbg !9
  %address.thread839 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !9, !range !34, !frisk.register_lifetimes !1
  %address.local840 = zext nneg i32 %address.thread839 to i64, !dbg !14
  %address.local841 = lshr i64 %address.local840, 3, !dbg !9
  %address.local842 = shl nuw nsw i64 %address.local841, 2, !dbg !9
  %address.local843 = or disjoint i64 %58, %address.local842, !dbg !9
  %address.local844 = shl i64 %address.local843, 7, !dbg !9
  %address.local845 = getelementptr [2 x i8], ptr addrspace(1) %459, i64 %address.local844, !dbg !9
  %address.local846 = shl nuw nsw i64 %address.local840, 2, !dbg !14
  %address.local847 = shl nuw nsw i64 %address.local841, 5, !dbg !9
  %address.local848 = sub nsw i64 %address.local846, %address.local847, !dbg !9
  %address.local849 = getelementptr [2 x i8], ptr addrspace(1) %address.local845, i64 %address.local848, !dbg !9
  %460 = load <4 x half>, ptr addrspace(1) %address.local849, align 2, !dbg !9
  %461 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %458, !dbg !9
  %address.local850 = shl i64 %address.local843, 7, !dbg !9
  %address.local851 = or disjoint i64 %address.local850, 128, !dbg !9
  %address.local852 = getelementptr [2 x i8], ptr addrspace(1) %461, i64 %address.local851, !dbg !9
  %address.local853 = getelementptr [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local848, !dbg !9
  %462 = load <4 x half>, ptr addrspace(1) %address.local853, align 2, !dbg !9
  %463 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %458, !dbg !9
  %address.local854 = shl i64 %address.local843, 7, !dbg !9
  %address.local855 = or disjoint i64 %address.local854, 256, !dbg !9
  %address.local856 = getelementptr [2 x i8], ptr addrspace(1) %463, i64 %address.local855, !dbg !9
  %address.local857 = getelementptr [2 x i8], ptr addrspace(1) %address.local856, i64 %address.local848, !dbg !9
  %464 = load <4 x half>, ptr addrspace(1) %address.local857, align 2, !dbg !9
  %465 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %458, !dbg !9
  %address.local858 = shl i64 %address.local843, 7, !dbg !9
  %address.local859 = or disjoint i64 %address.local858, 384, !dbg !9
  %address.local860 = getelementptr [2 x i8], ptr addrspace(1) %465, i64 %address.local859, !dbg !9
  %address.local861 = getelementptr [2 x i8], ptr addrspace(1) %address.local860, i64 %address.local848, !dbg !9
  %466 = load <4 x half>, ptr addrspace(1) %address.local861, align 2, !dbg !9
  %467 = call float @llvm.amdgcn.exp2.f32(float %455), !dbg !27
  %468 = insertelement <4 x float> poison, float %467, i64 0, !dbg !27
  %469 = extractelement <4 x float> %454, i64 1, !dbg !27
  %470 = call float @llvm.amdgcn.exp2.f32(float %469), !dbg !27
  %471 = fadd float %467, %470, !dbg !28
  %472 = insertelement <4 x float> %468, float %470, i64 1, !dbg !27
  %473 = extractelement <4 x float> %454, i64 2, !dbg !27
  %474 = call float @llvm.amdgcn.exp2.f32(float %473), !dbg !27
  %475 = fadd float %471, %474, !dbg !28
  %476 = insertelement <4 x float> %472, float %474, i64 2, !dbg !27
  %477 = extractelement <4 x float> %454, i64 3, !dbg !27
  %478 = call float @llvm.amdgcn.exp2.f32(float %477), !dbg !27
  %479 = fadd float %475, %478, !dbg !28
  %480 = insertelement <4 x float> %476, float %478, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %480) #8, !dbg !29, !frisk.register_lifetimes !1
  %481 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %481) #8, !dbg !26, !frisk.register_lifetimes !1
  %482 = shufflevector <32 x float> %450, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %483 = shufflevector <32 x float> %451, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %484 = fadd <4 x float> %482, %483, !dbg !26
  %485 = extractelement <4 x float> %484, i64 0, !dbg !27
  %486 = call float @llvm.amdgcn.exp2.f32(float %485), !dbg !27
  %487 = fadd float %479, %486, !dbg !28
  %488 = insertelement <4 x float> poison, float %486, i64 0, !dbg !27
  %489 = extractelement <4 x float> %484, i64 1, !dbg !27
  %490 = call float @llvm.amdgcn.exp2.f32(float %489), !dbg !27
  %491 = fadd float %487, %490, !dbg !28
  %492 = insertelement <4 x float> %488, float %490, i64 1, !dbg !27
  %493 = extractelement <4 x float> %484, i64 2, !dbg !27
  %494 = call float @llvm.amdgcn.exp2.f32(float %493), !dbg !27
  %495 = fadd float %491, %494, !dbg !28
  %496 = insertelement <4 x float> %492, float %494, i64 2, !dbg !27
  %497 = extractelement <4 x float> %484, i64 3, !dbg !27
  %498 = call float @llvm.amdgcn.exp2.f32(float %497), !dbg !27
  %499 = fadd float %495, %498, !dbg !28
  %500 = insertelement <4 x float> %496, float %498, i64 3, !dbg !27
  %math.result282 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %500) #8, !dbg !29, !frisk.register_lifetimes !1
  %501 = fptrunc <4 x float> %math.result282 to <4 x half>, !dbg !29
  %math.packed283 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %501) #8, !dbg !26, !frisk.register_lifetimes !1
  %502 = shufflevector <32 x float> %450, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %503 = shufflevector <32 x float> %451, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %504 = fadd <4 x float> %502, %503, !dbg !26
  %505 = extractelement <4 x float> %504, i64 0, !dbg !27
  %506 = call float @llvm.amdgcn.exp2.f32(float %505), !dbg !27
  %507 = fadd float %499, %506, !dbg !28
  %508 = insertelement <4 x float> poison, float %506, i64 0, !dbg !27
  %509 = extractelement <4 x float> %504, i64 1, !dbg !27
  %510 = call float @llvm.amdgcn.exp2.f32(float %509), !dbg !27
  %511 = fadd float %507, %510, !dbg !28
  %512 = insertelement <4 x float> %508, float %510, i64 1, !dbg !27
  %513 = extractelement <4 x float> %504, i64 2, !dbg !27
  %514 = call float @llvm.amdgcn.exp2.f32(float %513), !dbg !27
  %515 = fadd float %511, %514, !dbg !28
  %516 = insertelement <4 x float> %512, float %514, i64 2, !dbg !27
  %517 = extractelement <4 x float> %504, i64 3, !dbg !27
  %518 = call float @llvm.amdgcn.exp2.f32(float %517), !dbg !27
  %519 = fadd float %515, %518, !dbg !28
  %520 = insertelement <4 x float> %516, float %518, i64 3, !dbg !27
  %math.result284 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %520) #8, !dbg !29, !frisk.register_lifetimes !1
  %521 = fptrunc <4 x float> %math.result284 to <4 x half>, !dbg !29
  %math.packed285 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %521) #8, !dbg !26, !frisk.register_lifetimes !1
  %522 = shufflevector <32 x float> %450, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %523 = shufflevector <32 x float> %451, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %524 = fadd <4 x float> %522, %523, !dbg !26
  %525 = extractelement <4 x float> %524, i64 0, !dbg !27
  %526 = call float @llvm.amdgcn.exp2.f32(float %525), !dbg !27
  %527 = fadd float %519, %526, !dbg !28
  %528 = insertelement <4 x float> poison, float %526, i64 0, !dbg !27
  %529 = extractelement <4 x float> %524, i64 1, !dbg !27
  %530 = call float @llvm.amdgcn.exp2.f32(float %529), !dbg !27
  %531 = fadd float %527, %530, !dbg !28
  %532 = insertelement <4 x float> %528, float %530, i64 1, !dbg !27
  %533 = extractelement <4 x float> %524, i64 2, !dbg !27
  %534 = call float @llvm.amdgcn.exp2.f32(float %533), !dbg !27
  %535 = fadd float %531, %534, !dbg !28
  %536 = insertelement <4 x float> %532, float %534, i64 2, !dbg !27
  %537 = extractelement <4 x float> %524, i64 3, !dbg !27
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !27
  %539 = fadd float %535, %538, !dbg !28
  %540 = insertelement <4 x float> %536, float %538, i64 3, !dbg !27
  %math.result286 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %540) #8, !dbg !29, !frisk.register_lifetimes !1
  %541 = fptrunc <4 x float> %math.result286 to <4 x half>, !dbg !29
  %math.packed287 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %541) #8, !dbg !26, !frisk.register_lifetimes !1
  %542 = shufflevector <32 x float> %450, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %543 = shufflevector <32 x float> %451, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %544 = fadd <4 x float> %542, %543, !dbg !26
  %545 = extractelement <4 x float> %544, i64 0, !dbg !27
  %546 = call float @llvm.amdgcn.exp2.f32(float %545), !dbg !27
  %547 = fadd float %539, %546, !dbg !28
  %548 = insertelement <4 x float> poison, float %546, i64 0, !dbg !27
  %549 = extractelement <4 x float> %544, i64 1, !dbg !27
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !27
  %551 = fadd float %547, %550, !dbg !28
  %552 = insertelement <4 x float> %548, float %550, i64 1, !dbg !27
  %553 = extractelement <4 x float> %544, i64 2, !dbg !27
  %554 = call float @llvm.amdgcn.exp2.f32(float %553), !dbg !27
  %555 = fadd float %551, %554, !dbg !28
  %556 = insertelement <4 x float> %552, float %554, i64 2, !dbg !27
  %557 = extractelement <4 x float> %544, i64 3, !dbg !27
  %558 = call float @llvm.amdgcn.exp2.f32(float %557), !dbg !27
  %559 = fadd float %555, %558, !dbg !28
  %560 = insertelement <4 x float> %556, float %558, i64 3, !dbg !27
  %math.result288 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %560) #8, !dbg !29, !frisk.register_lifetimes !1
  %561 = fptrunc <4 x float> %math.result288 to <4 x half>, !dbg !29
  %math.packed289 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %561) #8, !dbg !26, !frisk.register_lifetimes !1
  %562 = shufflevector <32 x float> %450, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %563 = shufflevector <32 x float> %451, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %564 = fadd <4 x float> %562, %563, !dbg !26
  %565 = extractelement <4 x float> %564, i64 0, !dbg !27
  %566 = call float @llvm.amdgcn.exp2.f32(float %565), !dbg !27
  %567 = fadd float %559, %566, !dbg !28
  %568 = insertelement <4 x float> poison, float %566, i64 0, !dbg !27
  %569 = extractelement <4 x float> %564, i64 1, !dbg !27
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !27
  %571 = fadd float %567, %570, !dbg !28
  %572 = insertelement <4 x float> %568, float %570, i64 1, !dbg !27
  %573 = extractelement <4 x float> %564, i64 2, !dbg !27
  %574 = call float @llvm.amdgcn.exp2.f32(float %573), !dbg !27
  %575 = fadd float %571, %574, !dbg !28
  %576 = insertelement <4 x float> %572, float %574, i64 2, !dbg !27
  %577 = extractelement <4 x float> %564, i64 3, !dbg !27
  %578 = call float @llvm.amdgcn.exp2.f32(float %577), !dbg !27
  %579 = fadd float %575, %578, !dbg !28
  %580 = insertelement <4 x float> %576, float %578, i64 3, !dbg !27
  %math.result290 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %580) #8, !dbg !29, !frisk.register_lifetimes !1
  %581 = fptrunc <4 x float> %math.result290 to <4 x half>, !dbg !29
  %math.packed291 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %581) #8, !dbg !26, !frisk.register_lifetimes !1
  %582 = shufflevector <32 x float> %450, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %583 = shufflevector <32 x float> %451, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %584 = fadd <4 x float> %582, %583, !dbg !26
  %585 = extractelement <4 x float> %584, i64 0, !dbg !27
  %586 = call float @llvm.amdgcn.exp2.f32(float %585), !dbg !27
  %587 = fadd float %579, %586, !dbg !28
  %588 = insertelement <4 x float> poison, float %586, i64 0, !dbg !27
  %589 = extractelement <4 x float> %584, i64 1, !dbg !27
  %590 = call float @llvm.amdgcn.exp2.f32(float %589), !dbg !27
  %591 = fadd float %587, %590, !dbg !28
  %592 = insertelement <4 x float> %588, float %590, i64 1, !dbg !27
  %593 = extractelement <4 x float> %584, i64 2, !dbg !27
  %594 = call float @llvm.amdgcn.exp2.f32(float %593), !dbg !27
  %595 = fadd float %591, %594, !dbg !28
  %596 = insertelement <4 x float> %592, float %594, i64 2, !dbg !27
  %597 = extractelement <4 x float> %584, i64 3, !dbg !27
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !27
  %599 = fadd float %595, %598, !dbg !28
  %600 = insertelement <4 x float> %596, float %598, i64 3, !dbg !27
  %math.result292 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %600) #8, !dbg !29, !frisk.register_lifetimes !1
  %601 = fptrunc <4 x float> %math.result292 to <4 x half>, !dbg !29
  %math.packed293 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %601) #8, !dbg !26, !frisk.register_lifetimes !1
  %602 = shufflevector <32 x float> %450, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %603 = shufflevector <32 x float> %451, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %604 = fadd <4 x float> %602, %603, !dbg !26
  %605 = extractelement <4 x float> %604, i64 0, !dbg !27
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !27
  %607 = fadd float %599, %606, !dbg !28
  %608 = insertelement <4 x float> poison, float %606, i64 0, !dbg !27
  %609 = extractelement <4 x float> %604, i64 1, !dbg !27
  %610 = call float @llvm.amdgcn.exp2.f32(float %609), !dbg !27
  %611 = fadd float %607, %610, !dbg !28
  %612 = insertelement <4 x float> %608, float %610, i64 1, !dbg !27
  %613 = extractelement <4 x float> %604, i64 2, !dbg !27
  %614 = call float @llvm.amdgcn.exp2.f32(float %613), !dbg !27
  %615 = fadd float %611, %614, !dbg !28
  %616 = insertelement <4 x float> %612, float %614, i64 2, !dbg !27
  %617 = extractelement <4 x float> %604, i64 3, !dbg !27
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !27
  %619 = fadd float %615, %618, !dbg !28
  %620 = insertelement <4 x float> %616, float %618, i64 3, !dbg !27
  %math.result294 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %620) #8, !dbg !29, !frisk.register_lifetimes !1
  %621 = fptrunc <4 x float> %math.result294 to <4 x half>, !dbg !29
  %math.packed295 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %621) #8, !dbg !26, !frisk.register_lifetimes !1
  %622 = extractvalue [2 x <32 x float>] %61, 1, !dbg !26
  %623 = extractvalue [2 x <32 x float>] %448, 1, !dbg !26
  %624 = shufflevector <32 x float> %622, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %625 = shufflevector <32 x float> %623, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %626 = fadd <4 x float> %624, %625, !dbg !26
  %627 = extractelement <4 x float> %626, i64 0, !dbg !27
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !27
  %629 = insertelement <4 x float> poison, float %628, i64 0, !dbg !27
  %630 = extractelement <4 x float> %626, i64 1, !dbg !27
  %631 = call float @llvm.amdgcn.exp2.f32(float %630), !dbg !27
  %632 = fadd float %628, %631, !dbg !28
  %633 = insertelement <4 x float> %629, float %631, i64 1, !dbg !27
  %634 = extractelement <4 x float> %626, i64 2, !dbg !27
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !27
  %636 = fadd float %632, %635, !dbg !28
  %637 = insertelement <4 x float> %633, float %635, i64 2, !dbg !27
  %638 = extractelement <4 x float> %626, i64 3, !dbg !27
  %639 = call float @llvm.amdgcn.exp2.f32(float %638), !dbg !27
  %640 = fadd float %636, %639, !dbg !28
  %641 = insertelement <4 x float> %637, float %639, i64 3, !dbg !27
  %math.result296 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %641) #8, !dbg !29, !frisk.register_lifetimes !1
  %642 = fptrunc <4 x float> %math.result296 to <4 x half>, !dbg !29
  %math.packed297 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %642) #8, !dbg !26, !frisk.register_lifetimes !1
  %643 = shufflevector <32 x float> %622, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %644 = shufflevector <32 x float> %623, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %645 = fadd <4 x float> %643, %644, !dbg !26
  %646 = extractelement <4 x float> %645, i64 0, !dbg !27
  %647 = call float @llvm.amdgcn.exp2.f32(float %646), !dbg !27
  %648 = fadd float %640, %647, !dbg !28
  %649 = insertelement <4 x float> poison, float %647, i64 0, !dbg !27
  %650 = extractelement <4 x float> %645, i64 1, !dbg !27
  %651 = call float @llvm.amdgcn.exp2.f32(float %650), !dbg !27
  %652 = fadd float %648, %651, !dbg !28
  %653 = insertelement <4 x float> %649, float %651, i64 1, !dbg !27
  %654 = extractelement <4 x float> %645, i64 2, !dbg !27
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !27
  %656 = fadd float %652, %655, !dbg !28
  %657 = insertelement <4 x float> %653, float %655, i64 2, !dbg !27
  %658 = extractelement <4 x float> %645, i64 3, !dbg !27
  %659 = call float @llvm.amdgcn.exp2.f32(float %658), !dbg !27
  %660 = fadd float %656, %659, !dbg !28
  %661 = insertelement <4 x float> %657, float %659, i64 3, !dbg !27
  %math.result298 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %661) #8, !dbg !29, !frisk.register_lifetimes !1
  %662 = fptrunc <4 x float> %math.result298 to <4 x half>, !dbg !29
  %math.packed299 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %662) #8, !dbg !26, !frisk.register_lifetimes !1
  %663 = shufflevector <32 x float> %622, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %664 = shufflevector <32 x float> %623, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %665 = fadd <4 x float> %663, %664, !dbg !26
  %666 = extractelement <4 x float> %665, i64 0, !dbg !27
  %667 = call float @llvm.amdgcn.exp2.f32(float %666), !dbg !27
  %668 = fadd float %660, %667, !dbg !28
  %669 = insertelement <4 x float> poison, float %667, i64 0, !dbg !27
  %670 = extractelement <4 x float> %665, i64 1, !dbg !27
  %671 = call float @llvm.amdgcn.exp2.f32(float %670), !dbg !27
  %672 = fadd float %668, %671, !dbg !28
  %673 = insertelement <4 x float> %669, float %671, i64 1, !dbg !27
  %674 = extractelement <4 x float> %665, i64 2, !dbg !27
  %675 = call float @llvm.amdgcn.exp2.f32(float %674), !dbg !27
  %676 = fadd float %672, %675, !dbg !28
  %677 = insertelement <4 x float> %673, float %675, i64 2, !dbg !27
  %678 = extractelement <4 x float> %665, i64 3, !dbg !27
  %679 = call float @llvm.amdgcn.exp2.f32(float %678), !dbg !27
  %680 = fadd float %676, %679, !dbg !28
  %681 = insertelement <4 x float> %677, float %679, i64 3, !dbg !27
  %math.result300 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %681) #8, !dbg !29, !frisk.register_lifetimes !1
  %682 = fptrunc <4 x float> %math.result300 to <4 x half>, !dbg !29
  %math.packed301 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %682) #8, !dbg !26, !frisk.register_lifetimes !1
  %683 = shufflevector <32 x float> %622, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %684 = shufflevector <32 x float> %623, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %685 = fadd <4 x float> %683, %684, !dbg !26
  %686 = extractelement <4 x float> %685, i64 0, !dbg !27
  %687 = call float @llvm.amdgcn.exp2.f32(float %686), !dbg !27
  %688 = fadd float %680, %687, !dbg !28
  %689 = insertelement <4 x float> poison, float %687, i64 0, !dbg !27
  %690 = extractelement <4 x float> %685, i64 1, !dbg !27
  %691 = call float @llvm.amdgcn.exp2.f32(float %690), !dbg !27
  %692 = fadd float %688, %691, !dbg !28
  %693 = insertelement <4 x float> %689, float %691, i64 1, !dbg !27
  %694 = extractelement <4 x float> %685, i64 2, !dbg !27
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !27
  %696 = fadd float %692, %695, !dbg !28
  %697 = insertelement <4 x float> %693, float %695, i64 2, !dbg !27
  %698 = extractelement <4 x float> %685, i64 3, !dbg !27
  %699 = call float @llvm.amdgcn.exp2.f32(float %698), !dbg !27
  %700 = fadd float %696, %699, !dbg !28
  %701 = insertelement <4 x float> %697, float %699, i64 3, !dbg !27
  %math.result302 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %701) #8, !dbg !29, !frisk.register_lifetimes !1
  %702 = fptrunc <4 x float> %math.result302 to <4 x half>, !dbg !29
  %math.packed303 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %702) #8, !dbg !26, !frisk.register_lifetimes !1
  %703 = shufflevector <32 x float> %622, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %704 = shufflevector <32 x float> %623, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %705 = fadd <4 x float> %703, %704, !dbg !26
  %706 = extractelement <4 x float> %705, i64 0, !dbg !27
  %707 = call float @llvm.amdgcn.exp2.f32(float %706), !dbg !27
  %708 = fadd float %700, %707, !dbg !28
  %709 = insertelement <4 x float> poison, float %707, i64 0, !dbg !27
  %710 = extractelement <4 x float> %705, i64 1, !dbg !27
  %711 = call float @llvm.amdgcn.exp2.f32(float %710), !dbg !27
  %712 = fadd float %708, %711, !dbg !28
  %713 = insertelement <4 x float> %709, float %711, i64 1, !dbg !27
  %714 = extractelement <4 x float> %705, i64 2, !dbg !27
  %715 = call float @llvm.amdgcn.exp2.f32(float %714), !dbg !27
  %716 = fadd float %712, %715, !dbg !28
  %717 = insertelement <4 x float> %713, float %715, i64 2, !dbg !27
  %718 = extractelement <4 x float> %705, i64 3, !dbg !27
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !27
  %720 = fadd float %716, %719, !dbg !28
  %721 = insertelement <4 x float> %717, float %719, i64 3, !dbg !27
  %math.result304 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %721) #8, !dbg !29, !frisk.register_lifetimes !1
  %722 = fptrunc <4 x float> %math.result304 to <4 x half>, !dbg !29
  %math.packed305 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %722) #8, !dbg !26, !frisk.register_lifetimes !1
  %723 = shufflevector <32 x float> %622, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %724 = shufflevector <32 x float> %623, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %725 = fadd <4 x float> %723, %724, !dbg !26
  %726 = extractelement <4 x float> %725, i64 0, !dbg !27
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !27
  %728 = fadd float %720, %727, !dbg !28
  %729 = insertelement <4 x float> poison, float %727, i64 0, !dbg !27
  %730 = extractelement <4 x float> %725, i64 1, !dbg !27
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !27
  %732 = fadd float %728, %731, !dbg !28
  %733 = insertelement <4 x float> %729, float %731, i64 1, !dbg !27
  %734 = extractelement <4 x float> %725, i64 2, !dbg !27
  %735 = call float @llvm.amdgcn.exp2.f32(float %734), !dbg !27
  %736 = fadd float %732, %735, !dbg !28
  %737 = insertelement <4 x float> %733, float %735, i64 2, !dbg !27
  %738 = extractelement <4 x float> %725, i64 3, !dbg !27
  %739 = call float @llvm.amdgcn.exp2.f32(float %738), !dbg !27
  %740 = fadd float %736, %739, !dbg !28
  %741 = insertelement <4 x float> %737, float %739, i64 3, !dbg !27
  %math.result306 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %741) #8, !dbg !29, !frisk.register_lifetimes !1
  %742 = fptrunc <4 x float> %math.result306 to <4 x half>, !dbg !29
  %math.packed307 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %742) #8, !dbg !26, !frisk.register_lifetimes !1
  %743 = shufflevector <32 x float> %622, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %744 = shufflevector <32 x float> %623, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %745 = fadd <4 x float> %743, %744, !dbg !26
  %746 = extractelement <4 x float> %745, i64 0, !dbg !27
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !27
  %748 = fadd float %740, %747, !dbg !28
  %749 = insertelement <4 x float> poison, float %747, i64 0, !dbg !27
  %750 = extractelement <4 x float> %745, i64 1, !dbg !27
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !27
  %752 = fadd float %748, %751, !dbg !28
  %753 = insertelement <4 x float> %749, float %751, i64 1, !dbg !27
  %754 = extractelement <4 x float> %745, i64 2, !dbg !27
  %755 = call float @llvm.amdgcn.exp2.f32(float %754), !dbg !27
  %756 = fadd float %752, %755, !dbg !28
  %757 = insertelement <4 x float> %753, float %755, i64 2, !dbg !27
  %758 = extractelement <4 x float> %745, i64 3, !dbg !27
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !27
  %760 = fadd float %756, %759, !dbg !28
  %761 = insertelement <4 x float> %757, float %759, i64 3, !dbg !27
  %math.result308 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %761) #8, !dbg !29, !frisk.register_lifetimes !1
  %762 = fptrunc <4 x float> %math.result308 to <4 x half>, !dbg !29
  %math.packed309 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %762) #8, !dbg !26, !frisk.register_lifetimes !1
  %763 = shufflevector <32 x float> %622, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %764 = shufflevector <32 x float> %623, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %765 = fadd <4 x float> %763, %764, !dbg !26
  %766 = extractelement <4 x float> %765, i64 0, !dbg !27
  %767 = call float @llvm.amdgcn.exp2.f32(float %766), !dbg !27
  %768 = fadd float %760, %767, !dbg !28
  %769 = insertelement <4 x float> poison, float %767, i64 0, !dbg !27
  %770 = extractelement <4 x float> %765, i64 1, !dbg !27
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !27
  %772 = fadd float %768, %771, !dbg !28
  %773 = insertelement <4 x float> %769, float %771, i64 1, !dbg !27
  %774 = extractelement <4 x float> %765, i64 2, !dbg !27
  %775 = call float @llvm.amdgcn.exp2.f32(float %774), !dbg !27
  %776 = fadd float %772, %775, !dbg !28
  %777 = insertelement <4 x float> %773, float %775, i64 2, !dbg !27
  %778 = extractelement <4 x float> %765, i64 3, !dbg !27
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !27
  %780 = fadd float %776, %779, !dbg !28
  %781 = insertelement <4 x float> %777, float %779, i64 3, !dbg !27
  %math.result310 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %781) #8, !dbg !29, !frisk.register_lifetimes !1
  %782 = fptrunc <4 x float> %math.result310 to <4 x half>, !dbg !29
  %math.packed311 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %782) #8, !dbg !28, !frisk.register_lifetimes !1
  %783 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %784 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %783), !dbg !28
  %785 = add nuw nsw i32 %784, 64, !dbg !28
  %786 = xor i32 %784, 16, !dbg !28
  %787 = and i32 %785, 192, !dbg !28
  %788 = icmp samesign ult i32 %786, %787, !dbg !28
  %789 = select i1 %788, i32 %786, i32 %784, !dbg !28
  %790 = shl nuw nsw i32 %789, 2, !dbg !28
  %791 = bitcast float %619 to i32, !dbg !28
  %792 = call i32 @llvm.amdgcn.ds.bpermute(i32 %790, i32 %791), !dbg !28
  %793 = bitcast i32 %792 to float, !dbg !28
  %794 = fadd float %619, %793, !dbg !28
  %795 = xor i32 %784, 32, !dbg !28
  %796 = icmp samesign ult i32 %795, %787, !dbg !28
  %797 = select i1 %796, i32 %795, i32 %784, !dbg !28
  %798 = shl nuw nsw i32 %797, 2, !dbg !28
  %799 = bitcast float %794 to i32, !dbg !28
  %800 = call i32 @llvm.amdgcn.ds.bpermute(i32 %798, i32 %799), !dbg !28
  %801 = bitcast i32 %800 to float, !dbg !28
  %802 = fadd float %794, %801, !dbg !28
  %803 = extractvalue [2 x <1 x float>] %50, 0, !dbg !28
  %804 = extractelement <1 x float> %803, i64 0, !dbg !28
  %805 = fadd float %802, %804, !dbg !28
  %806 = insertelement <1 x float> poison, float %805, i64 0, !dbg !16
  %807 = insertvalue [2 x <1 x float>] poison, <1 x float> %806, 0, !dbg !16
  %808 = bitcast float %780 to i32, !dbg !28
  %809 = call i32 @llvm.amdgcn.ds.bpermute(i32 %790, i32 %808), !dbg !28
  %810 = bitcast i32 %809 to float, !dbg !28
  %811 = fadd float %780, %810, !dbg !28
  %812 = bitcast float %811 to i32, !dbg !28
  %813 = call i32 @llvm.amdgcn.ds.bpermute(i32 %798, i32 %812), !dbg !28
  %814 = bitcast i32 %813 to float, !dbg !28
  %815 = fadd float %811, %814, !dbg !28
  %816 = extractvalue [2 x <1 x float>] %50, 1, !dbg !28
  %817 = extractelement <1 x float> %816, i64 0, !dbg !28
  %818 = fadd float %815, %817, !dbg !28
  %819 = insertelement <1 x float> poison, float %818, i64 0, !dbg !16
  %820 = insertvalue [2 x <1 x float>] %807, <1 x float> %819, 1, !dbg !16
  %821 = shufflevector <4 x half> %460, <4 x half> %462, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %822 = shufflevector <4 x half> %821, <4 x half> %464, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %823 = shufflevector <4 x half> %822, <4 x half> %466, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
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
  store <4 x half> %823, ptr addrspace(3) %address.local878, align 2, !dbg !9
  %824 = shufflevector <4 x half> %460, <4 x half> %462, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %825 = shufflevector <4 x half> %824, <4 x half> %464, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %826 = shufflevector <4 x half> %825, <4 x half> %466, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
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
  store <4 x half> %826, ptr addrspace(3) %address.local898, align 2, !dbg !9
  %827 = shufflevector <4 x half> %460, <4 x half> %462, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %828 = shufflevector <4 x half> %827, <4 x half> %464, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %829 = shufflevector <4 x half> %828, <4 x half> %466, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  %address.base899 = call i64 asm sideeffect "", "=v,0"(i64 %.neg230) #8, !dbg !9, !frisk.register_lifetimes !1
  %address.local900 = shl i64 %address.base899, 6, !dbg !9
  %address.local901 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local900, !dbg !9
  %address.local902 = getelementptr [2 x i8], ptr addrspace(3) %address.local901, i64 %address.local896, !dbg !9
  %address.local903 = getelementptr i8, ptr addrspace(3) %address.local902, i64 32, !dbg !9
  store <4 x half> %829, ptr addrspace(3) %address.local903, align 2, !dbg !9
  %830 = shufflevector <4 x half> %460, <4 x half> %462, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %831 = shufflevector <4 x half> %830, <4 x half> %464, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %832 = shufflevector <4 x half> %831, <4 x half> %466, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  %address.local904 = mul nsw i64 %address.local841, -36, !dbg !9
  %address.local905 = add nsw i64 %address.local846, %address.local904, !dbg !9
  %address.local906 = add nsw i64 %address.local905, %address.local863, !dbg !9
  %address.local907 = add nsw i64 %address.local906, 3, !dbg !9
  %address.local908 = sdiv i64 %address.local907, -4, !dbg !9
  %address.local909 = shl i64 %address.local908, 6, !dbg !9
  %address.local910 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local909, !dbg !9
  %address.local911 = getelementptr [2 x i8], ptr addrspace(3) %address.local910, i64 %address.local896, !dbg !9
  %address.local912 = getelementptr i8, ptr addrspace(3) %address.local911, i64 48, !dbg !9
  store <4 x half> %832, ptr addrspace(3) %address.local912, align 2, !dbg !9
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
  %833 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %834 = extractvalue [2 x <8 x float>] %49, 0, !dbg !30
  %835 = shufflevector <8 x float> %834, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %836 = shufflevector <8 x half> %833, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %837 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %836, <4 x float> %835) #9, !dbg !18
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
  %838 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %839 = shufflevector <8 x half> %833, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %840 = shufflevector <8 x float> %834, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %841 = extractvalue [2 x <8 x float>] %49, 1, !dbg !30
  %842 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %839, <4 x float> %837) #9, !dbg !18
  %843 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %844 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %845 = shufflevector <8 x float> %841, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %846 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %843, <4 x float> %842) #9, !dbg !18
  %847 = shufflevector <8 x float> %841, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %848 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %844, <4 x float> %846) #9, !dbg !18
  %address.local953 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local913, !dbg !18
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) %address.local953, i64 %address.local934, !dbg !18
  %address.local955 = getelementptr [2 x i8], ptr addrspace(3) %address.local954, i64 %address.local915, !dbg !18
  %address.local956 = getelementptr [2 x i8], ptr addrspace(3) %address.local955, i64 %address.local919, !dbg !18
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) %address.local956, i64 %address.local923, !dbg !18
  %address.local958 = getelementptr [2 x i8], ptr addrspace(3) %address.local957, i64 %address.local926, !dbg !18
  %address.local959 = getelementptr [2 x i8], ptr addrspace(3) %address.local958, i64 %address.local930, !dbg !18
  %address.local960 = getelementptr i8, ptr addrspace(3) %address.local959, i64 4096, !dbg !18
  %849 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
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
  %850 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
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
  %851 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %852 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %853 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %852, <4 x float> %848) #9, !dbg !18
  %854 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %855 = shufflevector <8 x half> %850, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %856 = shufflevector <8 x half> %850, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %857 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %858 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
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
  %859 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %860 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %854, <4 x float> %853) #9, !dbg !18
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
  %861 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
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
  %862 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %863 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %864 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %855, <4 x float> %860) #9, !dbg !18
  %865 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %866 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %867 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %868 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %869 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %870 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %856, <4 x float> %864) #9, !dbg !18
  %871 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %872 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %873 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %874 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %857, <4 x float> %840) #9, !dbg !18
  %875 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %876 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %877 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %878 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %879 = shufflevector <8 x half> %873, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %880 = shufflevector <8 x half> %873, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %881 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %858, <4 x float> %874) #9, !dbg !18
  %882 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %883 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %863, <4 x float> %881) #9, !dbg !18
  %884 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %885 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %886 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %887 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %888 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %865, <4 x float> %883) #9, !dbg !18
  %889 = shufflevector <8 x half> %884, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %890 = shufflevector <8 x half> %884, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %891 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %892 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %893 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %894 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %866, <4 x float> %888) #9, !dbg !18
  %895 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %896 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %897 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %867, <4 x float> %894) #9, !dbg !18
  %898 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %899 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %900 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %901 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %868, <4 x float> %897) #9, !dbg !18
  %902 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %869, <4 x float> %901) #9, !dbg !18
  %903 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %875, <4 x float> %845) #9, !dbg !18
  %904 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %876, <4 x float> %903) #9, !dbg !18
  %905 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %877, <4 x float> %904) #9, !dbg !18
  %906 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %878, <4 x float> %905) #9, !dbg !18
  %907 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %879, <4 x float> %906) #9, !dbg !18
  %908 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %880, <4 x float> %907) #9, !dbg !18
  %909 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %886, <4 x float> %908) #9, !dbg !18
  %910 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %887, <4 x float> %909) #9, !dbg !18
  %911 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %889, <4 x float> %847) #9, !dbg !18
  %912 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %890, <4 x float> %911) #9, !dbg !18
  %913 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %891, <4 x float> %912) #9, !dbg !18
  %914 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %892, <4 x float> %913) #9, !dbg !18
  %915 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %896, <4 x float> %914) #9, !dbg !18
  %916 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %898, <4 x float> %915) #9, !dbg !18
  %917 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %899, <4 x float> %916) #9, !dbg !18
  %918 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %900, <4 x float> %917) #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %919 = add nsw i64 %56, %8, !dbg !9
  %920 = shl i64 %919, 19, !dbg !9
  %921 = and i64 %920, 16252928, !dbg !9
  %address.local1024 = add i64 %921, %address.local844, !dbg !9
  %address.local1025 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1024, !dbg !9
  %address.local1026 = shl nuw nsw i32 %address.thread839, 2, !dbg !9
  %address.local1027 = and i32 %address.local1026, 28, !dbg !9
  %address.local1028 = or disjoint i32 %address.local1027, 32, !dbg !9
  %address.local1029 = zext nneg i32 %address.local1028 to i64, !dbg !9
  %address.local1030 = getelementptr [2 x i8], ptr addrspace(1) %address.local1025, i64 %address.local1029, !dbg !9
  %922 = load <4 x half>, ptr addrspace(1) %address.local1030, align 2, !dbg !9
  %address.local1031 = add i64 %921, %address.local851, !dbg !9
  %address.local1032 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1031, !dbg !9
  %address.local1033 = getelementptr [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1029, !dbg !9
  %923 = load <4 x half>, ptr addrspace(1) %address.local1033, align 2, !dbg !9
  %address.local1034 = add i64 %921, %address.local855, !dbg !9
  %address.local1035 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1034, !dbg !9
  %address.local1036 = getelementptr [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1029, !dbg !9
  %924 = load <4 x half>, ptr addrspace(1) %address.local1036, align 2, !dbg !9
  %address.local1037 = add i64 %921, %address.local859, !dbg !9
  %address.local1038 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1037, !dbg !9
  %address.local1039 = getelementptr [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1029, !dbg !9
  %925 = load <4 x half>, ptr addrspace(1) %address.local1039, align 2, !dbg !9
  %926 = shufflevector <4 x half> %922, <4 x half> %923, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %927 = shufflevector <4 x half> %926, <4 x half> %924, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %928 = shufflevector <4 x half> %927, <4 x half> %925, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
  store <4 x half> %928, ptr addrspace(3) %address.local878, align 2, !dbg !9
  %929 = shufflevector <4 x half> %922, <4 x half> %923, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %930 = shufflevector <4 x half> %929, <4 x half> %924, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %931 = shufflevector <4 x half> %930, <4 x half> %925, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
  store <4 x half> %931, ptr addrspace(3) %address.local898, align 2, !dbg !9
  %932 = shufflevector <4 x half> %922, <4 x half> %923, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %933 = shufflevector <4 x half> %932, <4 x half> %924, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %934 = shufflevector <4 x half> %933, <4 x half> %925, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  store <4 x half> %934, ptr addrspace(3) %address.local903, align 2, !dbg !9
  %935 = shufflevector <4 x half> %922, <4 x half> %923, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %936 = shufflevector <4 x half> %935, <4 x half> %924, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %937 = shufflevector <4 x half> %936, <4 x half> %925, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  store <4 x half> %937, ptr addrspace(3) %address.local912, align 2, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %938 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %939 = extractvalue [2 x <8 x float>] %51, 0, !dbg !30
  %940 = shufflevector <8 x float> %939, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %941 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %942 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %941, <4 x float> %940) #9, !dbg !18
  %943 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %944 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %945 = shufflevector <8 x float> %939, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %946 = extractvalue [2 x <8 x float>] %51, 1, !dbg !30
  %947 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %944, <4 x float> %942) #9, !dbg !18
  %948 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %949 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %950 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %951 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %948, <4 x float> %947) #9, !dbg !18
  %952 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %953 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %949, <4 x float> %951) #9, !dbg !18
  %954 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %955 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %956 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %957 = shufflevector <8 x half> %954, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %958 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %957, <4 x float> %953) #9, !dbg !18
  %959 = shufflevector <8 x half> %954, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %960 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %961 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %962 = shufflevector <8 x half> %956, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %963 = shufflevector <8 x half> %956, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %964 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %965 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %959, <4 x float> %958) #9, !dbg !18
  %966 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %967 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %968 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %969 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %960, <4 x float> %965) #9, !dbg !18
  %970 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %971 = shufflevector <8 x half> %966, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %972 = shufflevector <8 x half> %966, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %973 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %974 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %975 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %961, <4 x float> %969) #9, !dbg !18
  %976 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %977 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %978 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %979 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %962, <4 x float> %945) #9, !dbg !18
  %980 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %981 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %982 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %983 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %984 = shufflevector <8 x half> %978, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %985 = shufflevector <8 x half> %978, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %986 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %963, <4 x float> %979) #9, !dbg !18
  %987 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %988 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %968, <4 x float> %986) #9, !dbg !18
  %989 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %990 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %991 = shufflevector <8 x half> %987, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %992 = shufflevector <8 x half> %987, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %993 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %970, <4 x float> %988) #9, !dbg !18
  %994 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %995 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %996 = shufflevector <8 x half> %990, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %997 = shufflevector <8 x half> %990, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %998 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %999 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %971, <4 x float> %993) #9, !dbg !18
  %1000 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %1001 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1002 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %972, <4 x float> %999) #9, !dbg !18
  %1003 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1004 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1005 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1006 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %973, <4 x float> %1002) #9, !dbg !18
  %1007 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %974, <4 x float> %1006) #9, !dbg !18
  %1008 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %980, <4 x float> %950) #9, !dbg !18
  %1009 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %981, <4 x float> %1008) #9, !dbg !18
  %1010 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %982, <4 x float> %1009) #9, !dbg !18
  %1011 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %983, <4 x float> %1010) #9, !dbg !18
  %1012 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %984, <4 x float> %1011) #9, !dbg !18
  %1013 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %985, <4 x float> %1012) #9, !dbg !18
  %1014 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %991, <4 x float> %1013) #9, !dbg !18
  %1015 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %992, <4 x float> %1014) #9, !dbg !18
  %1016 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %994, <4 x float> %952) #9, !dbg !18
  %1017 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %995, <4 x float> %1016) #9, !dbg !18
  %1018 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %996, <4 x float> %1017) #9, !dbg !18
  %1019 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %997, <4 x float> %1018) #9, !dbg !18
  %1020 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %1001, <4 x float> %1019) #9, !dbg !18
  %1021 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1003, <4 x float> %1020) #9, !dbg !18
  %1022 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1004, <4 x float> %1021) #9, !dbg !18
  %1023 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1005, <4 x float> %1022) #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local1040 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1024, !dbg !9
  %address.local1041 = shl nuw nsw i32 %address.thread839, 2, !dbg !9
  %address.local1042 = and i32 %address.local1041, 28, !dbg !9
  %address.local1043 = or disjoint i32 %address.local1042, 64, !dbg !9
  %address.local1044 = zext nneg i32 %address.local1043 to i64, !dbg !9
  %address.local1045 = getelementptr [2 x i8], ptr addrspace(1) %address.local1040, i64 %address.local1044, !dbg !9
  %1024 = load <4 x half>, ptr addrspace(1) %address.local1045, align 2, !dbg !9
  %address.local1046 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1031, !dbg !9
  %address.local1047 = getelementptr [2 x i8], ptr addrspace(1) %address.local1046, i64 %address.local1044, !dbg !9
  %1025 = load <4 x half>, ptr addrspace(1) %address.local1047, align 2, !dbg !9
  %address.local1048 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1034, !dbg !9
  %address.local1049 = getelementptr [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1044, !dbg !9
  %1026 = load <4 x half>, ptr addrspace(1) %address.local1049, align 2, !dbg !9
  %address.local1050 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1037, !dbg !9
  %address.local1051 = getelementptr [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1044, !dbg !9
  %1027 = load <4 x half>, ptr addrspace(1) %address.local1051, align 2, !dbg !9
  %1028 = shufflevector <4 x half> %1024, <4 x half> %1025, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %1029 = shufflevector <4 x half> %1028, <4 x half> %1026, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %1030 = shufflevector <4 x half> %1029, <4 x half> %1027, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
  store <4 x half> %1030, ptr addrspace(3) %address.local878, align 2, !dbg !9
  %1031 = shufflevector <4 x half> %1024, <4 x half> %1025, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %1032 = shufflevector <4 x half> %1031, <4 x half> %1026, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %1033 = shufflevector <4 x half> %1032, <4 x half> %1027, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
  store <4 x half> %1033, ptr addrspace(3) %address.local898, align 2, !dbg !9
  %1034 = shufflevector <4 x half> %1024, <4 x half> %1025, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %1035 = shufflevector <4 x half> %1034, <4 x half> %1026, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %1036 = shufflevector <4 x half> %1035, <4 x half> %1027, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  store <4 x half> %1036, ptr addrspace(3) %address.local903, align 2, !dbg !9
  %1037 = shufflevector <4 x half> %1024, <4 x half> %1025, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %1038 = shufflevector <4 x half> %1037, <4 x half> %1026, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %1039 = shufflevector <4 x half> %1038, <4 x half> %1027, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  store <4 x half> %1039, ptr addrspace(3) %address.local912, align 2, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %1040 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %1041 = extractvalue [2 x <8 x float>] %52, 0, !dbg !30
  %1042 = shufflevector <8 x float> %1041, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1043 = shufflevector <8 x half> %1040, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1044 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1043, <4 x float> %1042) #9, !dbg !18
  %1045 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %1046 = shufflevector <8 x half> %1040, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1047 = shufflevector <8 x float> %1041, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1048 = extractvalue [2 x <8 x float>] %52, 1, !dbg !30
  %1049 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1046, <4 x float> %1044) #9, !dbg !18
  %1050 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1051 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1052 = shufflevector <8 x float> %1048, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1053 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1050, <4 x float> %1049) #9, !dbg !18
  %1054 = shufflevector <8 x float> %1048, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1055 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1051, <4 x float> %1053) #9, !dbg !18
  %1056 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %1057 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %1058 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %1059 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1060 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1059, <4 x float> %1055) #9, !dbg !18
  %1061 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1062 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1063 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1064 = shufflevector <8 x half> %1058, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1065 = shufflevector <8 x half> %1058, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1066 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %1067 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %1061, <4 x float> %1060) #9, !dbg !18
  %1068 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %1069 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %1070 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1071 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %1062, <4 x float> %1067) #9, !dbg !18
  %1072 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1073 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1074 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1075 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1076 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1077 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %1063, <4 x float> %1071) #9, !dbg !18
  %1078 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %1079 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %1080 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %1081 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1064, <4 x float> %1047) #9, !dbg !18
  %1082 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1083 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1084 = shufflevector <8 x half> %1079, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1085 = shufflevector <8 x half> %1079, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1086 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1087 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1088 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1065, <4 x float> %1081) #9, !dbg !18
  %1089 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %1090 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1070, <4 x float> %1088) #9, !dbg !18
  %1091 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %1092 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %1093 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1094 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1095 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1072, <4 x float> %1090) #9, !dbg !18
  %1096 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1097 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1098 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1099 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1100 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %1101 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1073, <4 x float> %1095) #9, !dbg !18
  %1102 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %1103 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1104 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %1074, <4 x float> %1101) #9, !dbg !18
  %1105 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1106 = shufflevector <8 x half> %1102, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1107 = shufflevector <8 x half> %1102, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1108 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %1075, <4 x float> %1104) #9, !dbg !18
  %1109 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %1076, <4 x float> %1108) #9, !dbg !18
  %1110 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %1082, <4 x float> %1052) #9, !dbg !18
  %1111 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %1083, <4 x float> %1110) #9, !dbg !18
  %1112 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %1084, <4 x float> %1111) #9, !dbg !18
  %1113 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %1085, <4 x float> %1112) #9, !dbg !18
  %1114 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %1086, <4 x float> %1113) #9, !dbg !18
  %1115 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1087, <4 x float> %1114) #9, !dbg !18
  %1116 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1093, <4 x float> %1115) #9, !dbg !18
  %1117 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1094, <4 x float> %1116) #9, !dbg !18
  %1118 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %1096, <4 x float> %1054) #9, !dbg !18
  %1119 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %1097, <4 x float> %1118) #9, !dbg !18
  %1120 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %1098, <4 x float> %1119) #9, !dbg !18
  %1121 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %1099, <4 x float> %1120) #9, !dbg !18
  %1122 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %1103, <4 x float> %1121) #9, !dbg !18
  %1123 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1105, <4 x float> %1122) #9, !dbg !18
  %1124 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1106, <4 x float> %1123) #9, !dbg !18
  %1125 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1107, <4 x float> %1124) #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %address.local1052 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1024, !dbg !9
  %address.local1053 = shl nuw nsw i32 %address.thread839, 2, !dbg !9
  %address.local1054 = and i32 %address.local1053, 28, !dbg !9
  %address.local1055 = or disjoint i32 %address.local1054, 96, !dbg !9
  %address.local1056 = zext nneg i32 %address.local1055 to i64, !dbg !9
  %address.local1057 = getelementptr [2 x i8], ptr addrspace(1) %address.local1052, i64 %address.local1056, !dbg !9
  %1126 = load <4 x half>, ptr addrspace(1) %address.local1057, align 2, !dbg !9
  %address.local1058 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1031, !dbg !9
  %address.local1059 = getelementptr [2 x i8], ptr addrspace(1) %address.local1058, i64 %address.local1056, !dbg !9
  %1127 = load <4 x half>, ptr addrspace(1) %address.local1059, align 2, !dbg !9
  %address.local1060 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1034, !dbg !9
  %address.local1061 = getelementptr [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1056, !dbg !9
  %1128 = load <4 x half>, ptr addrspace(1) %address.local1061, align 2, !dbg !9
  %address.local1062 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %address.local1037, !dbg !9
  %address.local1063 = getelementptr [2 x i8], ptr addrspace(1) %address.local1062, i64 %address.local1056, !dbg !9
  %1129 = load <4 x half>, ptr addrspace(1) %address.local1063, align 2, !dbg !9
  %1130 = shufflevector <4 x half> %1126, <4 x half> %1127, <4 x i32> <i32 0, i32 4, i32 poison, i32 poison>, !dbg !9
  %1131 = shufflevector <4 x half> %1130, <4 x half> %1128, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !9
  %1132 = shufflevector <4 x half> %1131, <4 x half> %1129, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, !dbg !9
  store <4 x half> %1132, ptr addrspace(3) %address.local878, align 2, !dbg !9
  %1133 = shufflevector <4 x half> %1126, <4 x half> %1127, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !9
  %1134 = shufflevector <4 x half> %1133, <4 x half> %1128, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !9
  %1135 = shufflevector <4 x half> %1134, <4 x half> %1129, <4 x i32> <i32 0, i32 1, i32 2, i32 5>, !dbg !9
  store <4 x half> %1135, ptr addrspace(3) %address.local898, align 2, !dbg !9
  %1136 = shufflevector <4 x half> %1126, <4 x half> %1127, <4 x i32> <i32 2, i32 6, i32 poison, i32 poison>, !dbg !9
  %1137 = shufflevector <4 x half> %1136, <4 x half> %1128, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>, !dbg !9
  %1138 = shufflevector <4 x half> %1137, <4 x half> %1129, <4 x i32> <i32 0, i32 1, i32 2, i32 6>, !dbg !9
  store <4 x half> %1138, ptr addrspace(3) %address.local903, align 2, !dbg !9
  %1139 = shufflevector <4 x half> %1126, <4 x half> %1127, <4 x i32> <i32 3, i32 7, i32 poison, i32 poison>, !dbg !9
  %1140 = shufflevector <4 x half> %1139, <4 x half> %1128, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>, !dbg !9
  %1141 = shufflevector <4 x half> %1140, <4 x half> %1129, <4 x i32> <i32 0, i32 1, i32 2, i32 7>, !dbg !9
  store <4 x half> %1141, ptr addrspace(3) %address.local912, align 2, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %1142 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %1143 = extractvalue [2 x <8 x float>] %53, 0, !dbg !30
  %1144 = shufflevector <8 x float> %1143, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1145 = shufflevector <8 x half> %1142, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1146 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1145, <4 x float> %1144) #9, !dbg !18
  %1147 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %1148 = shufflevector <8 x half> %1142, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1149 = shufflevector <8 x float> %1143, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1150 = extractvalue [2 x <8 x float>] %53, 1, !dbg !30
  %1151 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1148, <4 x float> %1146) #9, !dbg !18
  %1152 = shufflevector <8 x half> %1147, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1153 = shufflevector <8 x half> %1147, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1154 = shufflevector <8 x float> %1150, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1155 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1152, <4 x float> %1151) #9, !dbg !18
  %1156 = shufflevector <8 x float> %1150, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1157 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1153, <4 x float> %1155) #9, !dbg !18
  %1158 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %1159 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %1160 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %1161 = shufflevector <8 x half> %1158, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1162 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1161, <4 x float> %1157) #9, !dbg !18
  %1163 = shufflevector <8 x half> %1158, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1164 = shufflevector <8 x half> %1159, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1165 = shufflevector <8 x half> %1159, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1166 = shufflevector <8 x half> %1160, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1167 = shufflevector <8 x half> %1160, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1168 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %1169 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %1163, <4 x float> %1162) #9, !dbg !18
  %1170 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %1171 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %1172 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1173 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %1164, <4 x float> %1169) #9, !dbg !18
  %1174 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1175 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1176 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1177 = shufflevector <8 x half> %1171, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1178 = shufflevector <8 x half> %1171, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1179 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %1165, <4 x float> %1173) #9, !dbg !18
  %1180 = load <8 x half>, ptr addrspace(3) %address.local935, align 2, !dbg !18
  %1181 = load <8 x half>, ptr addrspace(3) %address.local952, align 2, !dbg !18
  %1182 = load <8 x half>, ptr addrspace(3) %address.local960, align 2, !dbg !18
  %1183 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1166, <4 x float> %1149) #9, !dbg !18
  %1184 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1185 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1186 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1187 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1188 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1189 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1190 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed283, <4 x half> %1167, <4 x float> %1183) #9, !dbg !18
  %1191 = load <8 x half>, ptr addrspace(3) %address.local976, align 2, !dbg !18
  %1192 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed285, <4 x half> %1172, <4 x float> %1190) #9, !dbg !18
  %1193 = load <8 x half>, ptr addrspace(3) %address.local987, align 2, !dbg !18
  %1194 = load <8 x half>, ptr addrspace(3) %address.local1000, align 2, !dbg !18
  %1195 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1196 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1197 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %1174, <4 x float> %1192) #9, !dbg !18
  %1198 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1199 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1200 = shufflevector <8 x half> %1194, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1201 = shufflevector <8 x half> %1194, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1202 = load <8 x half>, ptr addrspace(3) %address.local1010, align 2, !dbg !18
  %1203 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %1175, <4 x float> %1197) #9, !dbg !18
  %1204 = load <8 x half>, ptr addrspace(3) %address.local1023, align 2, !dbg !18
  %1205 = shufflevector <4 x float> %870, <4 x float> %902, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1206 = insertvalue [2 x <8 x float>] poison, <8 x float> %1205, 0, !dbg !16
  %1207 = shufflevector <4 x float> %910, <4 x float> %918, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1208 = insertvalue [2 x <8 x float>] %1206, <8 x float> %1207, 1, !dbg !16
  %1209 = shufflevector <8 x half> %1202, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1210 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %1176, <4 x float> %1203) #9, !dbg !18
  %1211 = shufflevector <8 x half> %1202, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1212 = shufflevector <4 x float> %975, <4 x float> %1007, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1213 = insertvalue [2 x <8 x float>] poison, <8 x float> %1212, 0, !dbg !16
  %1214 = shufflevector <4 x float> %1015, <4 x float> %1023, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1215 = insertvalue [2 x <8 x float>] %1213, <8 x float> %1214, 1, !dbg !16
  %1216 = shufflevector <8 x half> %1204, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1217 = shufflevector <8 x half> %1204, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1218 = shufflevector <4 x float> %1077, <4 x float> %1109, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1219 = insertvalue [2 x <8 x float>] poison, <8 x float> %1218, 0, !dbg !16
  %1220 = shufflevector <4 x float> %1117, <4 x float> %1125, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1221 = insertvalue [2 x <8 x float>] %1219, <8 x float> %1220, 1, !dbg !16
  %1222 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %1177, <4 x float> %1210) #9, !dbg !18
  %1223 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %1178, <4 x float> %1222) #9, !dbg !18
  %1224 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %1184, <4 x float> %1154) #9, !dbg !18
  %1225 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %1185, <4 x float> %1224) #9, !dbg !18
  %1226 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %1186, <4 x float> %1225) #9, !dbg !18
  %1227 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %1187, <4 x float> %1226) #9, !dbg !18
  %1228 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %1188, <4 x float> %1227) #9, !dbg !18
  %1229 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1189, <4 x float> %1228) #9, !dbg !18
  %1230 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1195, <4 x float> %1229) #9, !dbg !18
  %1231 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1196, <4 x float> %1230) #9, !dbg !18
  %1232 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %1198, <4 x float> %1156) #9, !dbg !18
  %1233 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %1199, <4 x float> %1232) #9, !dbg !18
  %1234 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %1200, <4 x float> %1233) #9, !dbg !18
  %1235 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %1201, <4 x float> %1234) #9, !dbg !18
  %1236 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %1209, <4 x float> %1235) #9, !dbg !18
  %1237 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %1211, <4 x float> %1236) #9, !dbg !18
  %1238 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %1216, <4 x float> %1237) #9, !dbg !18
  %1239 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %1217, <4 x float> %1238) #9, !dbg !18
  %1240 = shufflevector <4 x float> %1179, <4 x float> %1223, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1241 = insertvalue [2 x <8 x float>] poison, <8 x float> %1240, 0, !dbg !16
  %1242 = shufflevector <4 x float> %1231, <4 x float> %1239, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1243 = insertvalue [2 x <8 x float>] %1241, <8 x float> %1242, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1244 = add nuw nsw i64 %48, 1, !dbg !16
  br label %47, !dbg !16

1245:                                             ; preds = %47
  %1246 = and i32 %19, 16, !dbg !16
  %.not38 = icmp eq i32 %1246, 0, !dbg !16
  %1247 = and i32 %19, 32, !dbg !16
  %.not39 = icmp eq i32 %1247, 0, !dbg !16
  %1248 = extractvalue [2 x <8 x float>] %53, 0, !dbg !16
  %1249 = extractelement <8 x float> %1248, i64 0, !dbg !16
  %1250 = extractelement <8 x float> %1248, i64 1, !dbg !16
  %1251 = extractelement <8 x float> %1248, i64 2, !dbg !16
  %1252 = extractelement <8 x float> %1248, i64 3, !dbg !16
  %1253 = select i1 %.not38, float %1250, float %1249, !dbg !16
  %1254 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %1255 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1254), !dbg !16
  %1256 = add nuw nsw i32 %1255, 64, !dbg !16
  %1257 = xor i32 %1255, 16, !dbg !16
  %1258 = and i32 %1256, 192, !dbg !16
  %1259 = icmp samesign ult i32 %1257, %1258, !dbg !16
  %1260 = select i1 %1259, i32 %1257, i32 %1255, !dbg !16
  %1261 = shl nuw nsw i32 %1260, 2, !dbg !16
  %1262 = bitcast float %1253 to i32, !dbg !16
  %1263 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1262), !dbg !16
  %1264 = bitcast i32 %1263 to float, !dbg !16
  %1265 = select i1 %.not38, float %1249, float %1264, !dbg !16
  %1266 = select i1 %.not38, float %1264, float %1250, !dbg !16
  %1267 = select i1 %.not38, float %1252, float %1251, !dbg !16
  %1268 = bitcast float %1267 to i32, !dbg !16
  %1269 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1268), !dbg !16
  %1270 = bitcast i32 %1269 to float, !dbg !16
  %1271 = select i1 %.not38, float %1251, float %1270, !dbg !16
  %1272 = select i1 %.not38, float %1270, float %1252, !dbg !16
  %1273 = select i1 %.not39, float %1271, float %1265, !dbg !16
  %1274 = xor i32 %1255, 32, !dbg !16
  %1275 = icmp samesign ult i32 %1274, %1258, !dbg !16
  %1276 = select i1 %1275, i32 %1274, i32 %1255, !dbg !16
  %1277 = shl nuw nsw i32 %1276, 2, !dbg !16
  %1278 = bitcast float %1273 to i32, !dbg !16
  %1279 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1278), !dbg !16
  %1280 = bitcast i32 %1279 to float, !dbg !16
  %1281 = select i1 %.not39, float %1265, float %1280, !dbg !16
  %1282 = select i1 %.not39, float %1280, float %1271, !dbg !16
  %1283 = select i1 %.not39, float %1272, float %1266, !dbg !16
  %1284 = bitcast float %1283 to i32, !dbg !16
  %1285 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1284), !dbg !16
  %1286 = bitcast i32 %1285 to float, !dbg !16
  %1287 = select i1 %.not39, float %1266, float %1286, !dbg !16
  %1288 = select i1 %.not39, float %1286, float %1272, !dbg !16
  %1289 = extractvalue [2 x <8 x float>] %52, 0, !dbg !16
  %1290 = extractelement <8 x float> %1289, i64 0, !dbg !16
  %1291 = extractelement <8 x float> %1289, i64 1, !dbg !16
  %1292 = extractelement <8 x float> %1289, i64 2, !dbg !16
  %1293 = extractelement <8 x float> %1289, i64 3, !dbg !16
  %1294 = select i1 %.not38, float %1291, float %1290, !dbg !16
  %1295 = bitcast float %1294 to i32, !dbg !16
  %1296 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1295), !dbg !16
  %1297 = bitcast i32 %1296 to float, !dbg !16
  %1298 = select i1 %.not38, float %1290, float %1297, !dbg !16
  %1299 = select i1 %.not38, float %1297, float %1291, !dbg !16
  %1300 = select i1 %.not38, float %1293, float %1292, !dbg !16
  %1301 = bitcast float %1300 to i32, !dbg !16
  %1302 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1301), !dbg !16
  %1303 = bitcast i32 %1302 to float, !dbg !16
  %1304 = select i1 %.not38, float %1292, float %1303, !dbg !16
  %1305 = select i1 %.not38, float %1303, float %1293, !dbg !16
  %1306 = select i1 %.not39, float %1304, float %1298, !dbg !16
  %1307 = bitcast float %1306 to i32, !dbg !16
  %1308 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1307), !dbg !16
  %1309 = bitcast i32 %1308 to float, !dbg !16
  %1310 = select i1 %.not39, float %1298, float %1309, !dbg !16
  %1311 = select i1 %.not39, float %1309, float %1304, !dbg !16
  %1312 = select i1 %.not39, float %1305, float %1299, !dbg !16
  %1313 = bitcast float %1312 to i32, !dbg !16
  %1314 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1313), !dbg !16
  %1315 = bitcast i32 %1314 to float, !dbg !16
  %1316 = select i1 %.not39, float %1299, float %1315, !dbg !16
  %1317 = select i1 %.not39, float %1315, float %1305, !dbg !16
  %1318 = extractvalue [2 x <8 x float>] %51, 0, !dbg !16
  %1319 = extractelement <8 x float> %1318, i64 0, !dbg !16
  %1320 = extractelement <8 x float> %1318, i64 1, !dbg !16
  %1321 = extractelement <8 x float> %1318, i64 2, !dbg !16
  %1322 = extractelement <8 x float> %1318, i64 3, !dbg !16
  %1323 = select i1 %.not38, float %1320, float %1319, !dbg !16
  %1324 = bitcast float %1323 to i32, !dbg !16
  %1325 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1324), !dbg !16
  %1326 = bitcast i32 %1325 to float, !dbg !16
  %1327 = select i1 %.not38, float %1319, float %1326, !dbg !16
  %1328 = select i1 %.not38, float %1326, float %1320, !dbg !16
  %1329 = select i1 %.not38, float %1322, float %1321, !dbg !16
  %1330 = bitcast float %1329 to i32, !dbg !16
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1330), !dbg !16
  %1332 = bitcast i32 %1331 to float, !dbg !16
  %1333 = select i1 %.not38, float %1321, float %1332, !dbg !16
  %1334 = select i1 %.not38, float %1332, float %1322, !dbg !16
  %1335 = select i1 %.not39, float %1333, float %1327, !dbg !16
  %1336 = bitcast float %1335 to i32, !dbg !16
  %1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1336), !dbg !16
  %1338 = bitcast i32 %1337 to float, !dbg !16
  %1339 = select i1 %.not39, float %1327, float %1338, !dbg !16
  %1340 = select i1 %.not39, float %1338, float %1333, !dbg !16
  %1341 = select i1 %.not39, float %1334, float %1328, !dbg !16
  %1342 = bitcast float %1341 to i32, !dbg !16
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1342), !dbg !16
  %1344 = bitcast i32 %1343 to float, !dbg !16
  %1345 = select i1 %.not39, float %1328, float %1344, !dbg !16
  %1346 = select i1 %.not39, float %1344, float %1334, !dbg !16
  %1347 = extractvalue [2 x <8 x float>] %49, 0, !dbg !16
  %1348 = extractelement <8 x float> %1347, i64 0, !dbg !16
  %1349 = extractelement <8 x float> %1347, i64 1, !dbg !16
  %1350 = extractelement <8 x float> %1347, i64 2, !dbg !16
  %1351 = extractelement <8 x float> %1347, i64 3, !dbg !16
  %1352 = select i1 %.not38, float %1349, float %1348, !dbg !16
  %1353 = bitcast float %1352 to i32, !dbg !16
  %1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1353), !dbg !16
  %1355 = bitcast i32 %1354 to float, !dbg !16
  %1356 = select i1 %.not38, float %1348, float %1355, !dbg !16
  %1357 = select i1 %.not38, float %1355, float %1349, !dbg !16
  %1358 = select i1 %.not38, float %1351, float %1350, !dbg !16
  %1359 = bitcast float %1358 to i32, !dbg !16
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1359), !dbg !16
  %1361 = bitcast i32 %1360 to float, !dbg !16
  %1362 = select i1 %.not38, float %1350, float %1361, !dbg !16
  %1363 = select i1 %.not38, float %1361, float %1351, !dbg !16
  %1364 = select i1 %.not39, float %1362, float %1356, !dbg !16
  %1365 = bitcast float %1364 to i32, !dbg !16
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1365), !dbg !16
  %1367 = select i1 %.not39, float %1363, float %1357, !dbg !16
  %1368 = bitcast float %1367 to i32, !dbg !16
  %1369 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1368), !dbg !16
  %1370 = extractvalue [2 x <1 x float>] %50, 0, !dbg !19
  %1371 = shufflevector <1 x float> %1370, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1372 = extractelement <8 x float> %1248, i64 4, !dbg !16
  %1373 = extractelement <8 x float> %1248, i64 5, !dbg !16
  %1374 = extractelement <8 x float> %1248, i64 6, !dbg !16
  %1375 = extractelement <8 x float> %1248, i64 7, !dbg !16
  %1376 = select i1 %.not38, float %1373, float %1372, !dbg !16
  %1377 = bitcast float %1376 to i32, !dbg !16
  %1378 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1377), !dbg !16
  %1379 = bitcast i32 %1378 to float, !dbg !16
  %1380 = select i1 %.not38, float %1372, float %1379, !dbg !16
  %1381 = select i1 %.not38, float %1379, float %1373, !dbg !16
  %1382 = select i1 %.not38, float %1375, float %1374, !dbg !16
  %1383 = bitcast float %1382 to i32, !dbg !16
  %1384 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1383), !dbg !16
  %1385 = bitcast i32 %1384 to float, !dbg !16
  %1386 = select i1 %.not38, float %1374, float %1385, !dbg !16
  %1387 = select i1 %.not38, float %1385, float %1375, !dbg !16
  %1388 = select i1 %.not39, float %1386, float %1380, !dbg !16
  %1389 = bitcast float %1388 to i32, !dbg !16
  %1390 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1389), !dbg !16
  %1391 = bitcast i32 %1390 to float, !dbg !16
  %1392 = select i1 %.not39, float %1380, float %1391, !dbg !16
  %1393 = select i1 %.not39, float %1391, float %1386, !dbg !16
  %1394 = select i1 %.not39, float %1387, float %1381, !dbg !16
  %1395 = bitcast float %1394 to i32, !dbg !16
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1395), !dbg !16
  %1397 = bitcast i32 %1396 to float, !dbg !16
  %1398 = select i1 %.not39, float %1381, float %1397, !dbg !16
  %1399 = select i1 %.not39, float %1397, float %1387, !dbg !16
  %1400 = extractelement <8 x float> %1289, i64 4, !dbg !16
  %1401 = extractelement <8 x float> %1289, i64 5, !dbg !16
  %1402 = extractelement <8 x float> %1289, i64 6, !dbg !16
  %1403 = extractelement <8 x float> %1289, i64 7, !dbg !16
  %1404 = select i1 %.not38, float %1401, float %1400, !dbg !16
  %1405 = bitcast float %1404 to i32, !dbg !16
  %1406 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1405), !dbg !16
  %1407 = bitcast i32 %1406 to float, !dbg !16
  %1408 = select i1 %.not38, float %1400, float %1407, !dbg !16
  %1409 = select i1 %.not38, float %1407, float %1401, !dbg !16
  %1410 = select i1 %.not38, float %1403, float %1402, !dbg !16
  %1411 = bitcast float %1410 to i32, !dbg !16
  %1412 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1411), !dbg !16
  %1413 = bitcast i32 %1412 to float, !dbg !16
  %1414 = select i1 %.not38, float %1402, float %1413, !dbg !16
  %1415 = select i1 %.not38, float %1413, float %1403, !dbg !16
  %1416 = select i1 %.not39, float %1414, float %1408, !dbg !16
  %1417 = bitcast float %1416 to i32, !dbg !16
  %1418 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1417), !dbg !16
  %1419 = bitcast i32 %1418 to float, !dbg !16
  %1420 = select i1 %.not39, float %1408, float %1419, !dbg !16
  %1421 = select i1 %.not39, float %1419, float %1414, !dbg !16
  %1422 = select i1 %.not39, float %1415, float %1409, !dbg !16
  %1423 = bitcast float %1422 to i32, !dbg !16
  %1424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1423), !dbg !16
  %1425 = bitcast i32 %1424 to float, !dbg !16
  %1426 = select i1 %.not39, float %1409, float %1425, !dbg !16
  %1427 = select i1 %.not39, float %1425, float %1415, !dbg !16
  %1428 = extractelement <8 x float> %1318, i64 4, !dbg !16
  %1429 = extractelement <8 x float> %1318, i64 5, !dbg !16
  %1430 = extractelement <8 x float> %1318, i64 6, !dbg !16
  %1431 = extractelement <8 x float> %1318, i64 7, !dbg !16
  %1432 = select i1 %.not38, float %1429, float %1428, !dbg !16
  %1433 = bitcast float %1432 to i32, !dbg !16
  %1434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1433), !dbg !16
  %1435 = bitcast i32 %1434 to float, !dbg !16
  %1436 = select i1 %.not38, float %1428, float %1435, !dbg !16
  %1437 = select i1 %.not38, float %1435, float %1429, !dbg !16
  %1438 = select i1 %.not38, float %1431, float %1430, !dbg !16
  %1439 = bitcast float %1438 to i32, !dbg !16
  %1440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1439), !dbg !16
  %1441 = bitcast i32 %1440 to float, !dbg !16
  %1442 = select i1 %.not38, float %1430, float %1441, !dbg !16
  %1443 = select i1 %.not38, float %1441, float %1431, !dbg !16
  %1444 = select i1 %.not39, float %1442, float %1436, !dbg !16
  %1445 = bitcast float %1444 to i32, !dbg !16
  %1446 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1445), !dbg !16
  %1447 = bitcast i32 %1446 to float, !dbg !16
  %1448 = select i1 %.not39, float %1436, float %1447, !dbg !16
  %1449 = select i1 %.not39, float %1447, float %1442, !dbg !16
  %1450 = select i1 %.not39, float %1443, float %1437, !dbg !16
  %1451 = bitcast float %1450 to i32, !dbg !16
  %1452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1451), !dbg !16
  %1453 = bitcast i32 %1452 to float, !dbg !16
  %1454 = select i1 %.not39, float %1437, float %1453, !dbg !16
  %1455 = select i1 %.not39, float %1453, float %1443, !dbg !16
  %1456 = extractelement <8 x float> %1347, i64 4, !dbg !16
  %1457 = extractelement <8 x float> %1347, i64 5, !dbg !16
  %1458 = extractelement <8 x float> %1347, i64 6, !dbg !16
  %1459 = extractelement <8 x float> %1347, i64 7, !dbg !16
  %1460 = select i1 %.not38, float %1457, float %1456, !dbg !16
  %1461 = bitcast float %1460 to i32, !dbg !16
  %1462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1461), !dbg !16
  %1463 = bitcast i32 %1462 to float, !dbg !16
  %1464 = select i1 %.not38, float %1456, float %1463, !dbg !16
  %1465 = select i1 %.not38, float %1463, float %1457, !dbg !16
  %1466 = select i1 %.not38, float %1459, float %1458, !dbg !16
  %1467 = bitcast float %1466 to i32, !dbg !16
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1467), !dbg !16
  %1469 = bitcast i32 %1468 to float, !dbg !16
  %1470 = select i1 %.not38, float %1458, float %1469, !dbg !16
  %1471 = select i1 %.not38, float %1469, float %1459, !dbg !16
  %1472 = select i1 %.not39, float %1470, float %1464, !dbg !16
  %1473 = bitcast float %1472 to i32, !dbg !16
  %1474 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1473), !dbg !16
  %1475 = select i1 %.not39, float %1471, float %1465, !dbg !16
  %1476 = bitcast float %1475 to i32, !dbg !16
  %1477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1476), !dbg !16
  %1478 = extractvalue [2 x <8 x float>] %53, 1, !dbg !16
  %1479 = extractelement <8 x float> %1478, i64 0, !dbg !16
  %1480 = extractelement <8 x float> %1478, i64 1, !dbg !16
  %1481 = extractelement <8 x float> %1478, i64 2, !dbg !16
  %1482 = extractelement <8 x float> %1478, i64 3, !dbg !16
  %1483 = select i1 %.not38, float %1480, float %1479, !dbg !16
  %1484 = bitcast float %1483 to i32, !dbg !16
  %1485 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1484), !dbg !16
  %1486 = bitcast i32 %1485 to float, !dbg !16
  %1487 = select i1 %.not38, float %1479, float %1486, !dbg !16
  %1488 = select i1 %.not38, float %1486, float %1480, !dbg !16
  %1489 = select i1 %.not38, float %1482, float %1481, !dbg !16
  %1490 = bitcast float %1489 to i32, !dbg !16
  %1491 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1490), !dbg !16
  %1492 = bitcast i32 %1491 to float, !dbg !16
  %1493 = select i1 %.not38, float %1481, float %1492, !dbg !16
  %1494 = select i1 %.not38, float %1492, float %1482, !dbg !16
  %1495 = select i1 %.not39, float %1493, float %1487, !dbg !16
  %1496 = bitcast float %1495 to i32, !dbg !16
  %1497 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1496), !dbg !16
  %1498 = bitcast i32 %1497 to float, !dbg !16
  %1499 = select i1 %.not39, float %1487, float %1498, !dbg !16
  %1500 = select i1 %.not39, float %1498, float %1493, !dbg !16
  %1501 = select i1 %.not39, float %1494, float %1488, !dbg !16
  %1502 = bitcast float %1501 to i32, !dbg !16
  %1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1502), !dbg !16
  %1504 = bitcast i32 %1503 to float, !dbg !16
  %1505 = select i1 %.not39, float %1488, float %1504, !dbg !16
  %1506 = select i1 %.not39, float %1504, float %1494, !dbg !16
  %1507 = extractvalue [2 x <8 x float>] %52, 1, !dbg !16
  %1508 = extractelement <8 x float> %1507, i64 0, !dbg !16
  %1509 = extractelement <8 x float> %1507, i64 1, !dbg !16
  %1510 = extractelement <8 x float> %1507, i64 2, !dbg !16
  %1511 = extractelement <8 x float> %1507, i64 3, !dbg !16
  %1512 = select i1 %.not38, float %1509, float %1508, !dbg !16
  %1513 = bitcast float %1512 to i32, !dbg !16
  %1514 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1513), !dbg !16
  %1515 = bitcast i32 %1514 to float, !dbg !16
  %1516 = select i1 %.not38, float %1508, float %1515, !dbg !16
  %1517 = select i1 %.not38, float %1515, float %1509, !dbg !16
  %1518 = select i1 %.not38, float %1511, float %1510, !dbg !16
  %1519 = bitcast float %1518 to i32, !dbg !16
  %1520 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1519), !dbg !16
  %1521 = bitcast i32 %1520 to float, !dbg !16
  %1522 = select i1 %.not38, float %1510, float %1521, !dbg !16
  %1523 = select i1 %.not38, float %1521, float %1511, !dbg !16
  %1524 = select i1 %.not39, float %1522, float %1516, !dbg !16
  %1525 = bitcast float %1524 to i32, !dbg !16
  %1526 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1525), !dbg !16
  %1527 = bitcast i32 %1526 to float, !dbg !16
  %1528 = select i1 %.not39, float %1516, float %1527, !dbg !16
  %1529 = select i1 %.not39, float %1527, float %1522, !dbg !16
  %1530 = select i1 %.not39, float %1523, float %1517, !dbg !16
  %1531 = bitcast float %1530 to i32, !dbg !16
  %1532 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1531), !dbg !16
  %1533 = bitcast i32 %1532 to float, !dbg !16
  %1534 = select i1 %.not39, float %1517, float %1533, !dbg !16
  %1535 = select i1 %.not39, float %1533, float %1523, !dbg !16
  %1536 = extractvalue [2 x <8 x float>] %51, 1, !dbg !16
  %1537 = extractelement <8 x float> %1536, i64 0, !dbg !16
  %1538 = extractelement <8 x float> %1536, i64 1, !dbg !16
  %1539 = extractelement <8 x float> %1536, i64 2, !dbg !16
  %1540 = extractelement <8 x float> %1536, i64 3, !dbg !16
  %1541 = select i1 %.not38, float %1538, float %1537, !dbg !16
  %1542 = bitcast float %1541 to i32, !dbg !16
  %1543 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1542), !dbg !16
  %1544 = bitcast i32 %1543 to float, !dbg !16
  %1545 = select i1 %.not38, float %1537, float %1544, !dbg !16
  %1546 = select i1 %.not38, float %1544, float %1538, !dbg !16
  %1547 = select i1 %.not38, float %1540, float %1539, !dbg !16
  %1548 = bitcast float %1547 to i32, !dbg !16
  %1549 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1548), !dbg !16
  %1550 = bitcast i32 %1549 to float, !dbg !16
  %1551 = select i1 %.not38, float %1539, float %1550, !dbg !16
  %1552 = select i1 %.not38, float %1550, float %1540, !dbg !16
  %1553 = select i1 %.not39, float %1551, float %1545, !dbg !16
  %1554 = bitcast float %1553 to i32, !dbg !16
  %1555 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1554), !dbg !16
  %1556 = bitcast i32 %1555 to float, !dbg !16
  %1557 = select i1 %.not39, float %1545, float %1556, !dbg !16
  %1558 = select i1 %.not39, float %1556, float %1551, !dbg !16
  %1559 = select i1 %.not39, float %1552, float %1546, !dbg !16
  %1560 = bitcast float %1559 to i32, !dbg !16
  %1561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1560), !dbg !16
  %1562 = bitcast i32 %1561 to float, !dbg !16
  %1563 = select i1 %.not39, float %1546, float %1562, !dbg !16
  %1564 = select i1 %.not39, float %1562, float %1552, !dbg !16
  %1565 = extractvalue [2 x <8 x float>] %49, 1, !dbg !16
  %1566 = extractelement <8 x float> %1565, i64 0, !dbg !16
  %1567 = extractelement <8 x float> %1565, i64 1, !dbg !16
  %1568 = extractelement <8 x float> %1565, i64 2, !dbg !16
  %1569 = extractelement <8 x float> %1565, i64 3, !dbg !16
  %1570 = select i1 %.not38, float %1567, float %1566, !dbg !16
  %1571 = bitcast float %1570 to i32, !dbg !16
  %1572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1571), !dbg !16
  %1573 = bitcast i32 %1572 to float, !dbg !16
  %1574 = select i1 %.not38, float %1566, float %1573, !dbg !16
  %1575 = select i1 %.not38, float %1573, float %1567, !dbg !16
  %1576 = select i1 %.not38, float %1569, float %1568, !dbg !16
  %1577 = bitcast float %1576 to i32, !dbg !16
  %1578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1577), !dbg !16
  %1579 = bitcast i32 %1578 to float, !dbg !16
  %1580 = select i1 %.not38, float %1568, float %1579, !dbg !16
  %1581 = select i1 %.not38, float %1579, float %1569, !dbg !16
  %1582 = select i1 %.not39, float %1580, float %1574, !dbg !16
  %1583 = bitcast float %1582 to i32, !dbg !16
  %1584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1583), !dbg !16
  %1585 = select i1 %.not39, float %1581, float %1575, !dbg !16
  %1586 = bitcast float %1585 to i32, !dbg !16
  %1587 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1586), !dbg !16
  %1588 = extractvalue [2 x <1 x float>] %50, 1, !dbg !19
  %1589 = shufflevector <1 x float> %1588, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1590 = extractelement <8 x float> %1478, i64 4, !dbg !16
  %1591 = extractelement <8 x float> %1478, i64 5, !dbg !16
  %1592 = extractelement <8 x float> %1478, i64 6, !dbg !16
  %1593 = extractelement <8 x float> %1478, i64 7, !dbg !16
  %1594 = select i1 %.not38, float %1591, float %1590, !dbg !16
  %1595 = bitcast float %1594 to i32, !dbg !16
  %1596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1595), !dbg !16
  %1597 = bitcast i32 %1596 to float, !dbg !16
  %1598 = select i1 %.not38, float %1590, float %1597, !dbg !16
  %1599 = select i1 %.not38, float %1597, float %1591, !dbg !16
  %1600 = select i1 %.not38, float %1593, float %1592, !dbg !16
  %1601 = bitcast float %1600 to i32, !dbg !16
  %1602 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1601), !dbg !16
  %1603 = bitcast i32 %1602 to float, !dbg !16
  %1604 = select i1 %.not38, float %1592, float %1603, !dbg !16
  %1605 = select i1 %.not38, float %1603, float %1593, !dbg !16
  %1606 = select i1 %.not39, float %1604, float %1598, !dbg !16
  %1607 = bitcast float %1606 to i32, !dbg !16
  %1608 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1607), !dbg !16
  %1609 = bitcast i32 %1608 to float, !dbg !16
  %1610 = select i1 %.not39, float %1598, float %1609, !dbg !16
  %1611 = select i1 %.not39, float %1609, float %1604, !dbg !16
  %1612 = select i1 %.not39, float %1605, float %1599, !dbg !16
  %1613 = bitcast float %1612 to i32, !dbg !16
  %1614 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1613), !dbg !16
  %1615 = bitcast i32 %1614 to float, !dbg !16
  %1616 = select i1 %.not39, float %1599, float %1615, !dbg !16
  %1617 = select i1 %.not39, float %1615, float %1605, !dbg !16
  %1618 = extractelement <8 x float> %1507, i64 4, !dbg !16
  %1619 = extractelement <8 x float> %1507, i64 5, !dbg !16
  %1620 = extractelement <8 x float> %1507, i64 6, !dbg !16
  %1621 = extractelement <8 x float> %1507, i64 7, !dbg !16
  %1622 = select i1 %.not38, float %1619, float %1618, !dbg !16
  %1623 = bitcast float %1622 to i32, !dbg !16
  %1624 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1623), !dbg !16
  %1625 = bitcast i32 %1624 to float, !dbg !16
  %1626 = select i1 %.not38, float %1618, float %1625, !dbg !16
  %1627 = select i1 %.not38, float %1625, float %1619, !dbg !16
  %1628 = select i1 %.not38, float %1621, float %1620, !dbg !16
  %1629 = bitcast float %1628 to i32, !dbg !16
  %1630 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1629), !dbg !16
  %1631 = bitcast i32 %1630 to float, !dbg !16
  %1632 = select i1 %.not38, float %1620, float %1631, !dbg !16
  %1633 = select i1 %.not38, float %1631, float %1621, !dbg !16
  %1634 = select i1 %.not39, float %1632, float %1626, !dbg !16
  %1635 = bitcast float %1634 to i32, !dbg !16
  %1636 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1635), !dbg !16
  %1637 = bitcast i32 %1636 to float, !dbg !16
  %1638 = select i1 %.not39, float %1626, float %1637, !dbg !16
  %1639 = select i1 %.not39, float %1637, float %1632, !dbg !16
  %1640 = select i1 %.not39, float %1633, float %1627, !dbg !16
  %1641 = bitcast float %1640 to i32, !dbg !16
  %1642 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1641), !dbg !16
  %1643 = bitcast i32 %1642 to float, !dbg !16
  %1644 = select i1 %.not39, float %1627, float %1643, !dbg !16
  %1645 = select i1 %.not39, float %1643, float %1633, !dbg !16
  %1646 = extractelement <8 x float> %1536, i64 4, !dbg !16
  %1647 = extractelement <8 x float> %1536, i64 5, !dbg !16
  %1648 = extractelement <8 x float> %1536, i64 6, !dbg !16
  %1649 = extractelement <8 x float> %1536, i64 7, !dbg !16
  %1650 = select i1 %.not38, float %1647, float %1646, !dbg !16
  %1651 = bitcast float %1650 to i32, !dbg !16
  %1652 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1651), !dbg !16
  %1653 = bitcast i32 %1652 to float, !dbg !16
  %1654 = select i1 %.not38, float %1646, float %1653, !dbg !16
  %1655 = select i1 %.not38, float %1653, float %1647, !dbg !16
  %1656 = select i1 %.not38, float %1649, float %1648, !dbg !16
  %1657 = bitcast float %1656 to i32, !dbg !16
  %1658 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1657), !dbg !16
  %1659 = bitcast i32 %1658 to float, !dbg !16
  %1660 = select i1 %.not38, float %1648, float %1659, !dbg !16
  %1661 = select i1 %.not38, float %1659, float %1649, !dbg !16
  %1662 = select i1 %.not39, float %1660, float %1654, !dbg !16
  %1663 = bitcast float %1662 to i32, !dbg !16
  %1664 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1663), !dbg !16
  %1665 = bitcast i32 %1664 to float, !dbg !16
  %1666 = select i1 %.not39, float %1654, float %1665, !dbg !16
  %1667 = select i1 %.not39, float %1665, float %1660, !dbg !16
  %1668 = select i1 %.not39, float %1661, float %1655, !dbg !16
  %1669 = bitcast float %1668 to i32, !dbg !16
  %1670 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1669), !dbg !16
  %1671 = bitcast i32 %1670 to float, !dbg !16
  %1672 = select i1 %.not39, float %1655, float %1671, !dbg !16
  %1673 = select i1 %.not39, float %1671, float %1661, !dbg !16
  %1674 = extractelement <8 x float> %1565, i64 4, !dbg !16
  %1675 = extractelement <8 x float> %1565, i64 5, !dbg !16
  %1676 = extractelement <8 x float> %1565, i64 6, !dbg !16
  %1677 = extractelement <8 x float> %1565, i64 7, !dbg !16
  %1678 = select i1 %.not38, float %1675, float %1674, !dbg !16
  %1679 = bitcast float %1678 to i32, !dbg !16
  %1680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1679), !dbg !16
  %1681 = bitcast i32 %1680 to float, !dbg !16
  %1682 = select i1 %.not38, float %1674, float %1681, !dbg !16
  %1683 = select i1 %.not38, float %1681, float %1675, !dbg !16
  %1684 = select i1 %.not38, float %1677, float %1676, !dbg !16
  %1685 = bitcast float %1684 to i32, !dbg !16
  %1686 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1685), !dbg !16
  %1687 = bitcast i32 %1686 to float, !dbg !16
  %1688 = select i1 %.not38, float %1676, float %1687, !dbg !16
  %1689 = select i1 %.not38, float %1687, float %1677, !dbg !16
  %1690 = select i1 %.not39, float %1688, float %1682, !dbg !16
  %1691 = bitcast float %1690 to i32, !dbg !16
  %1692 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1691), !dbg !16
  %1693 = select i1 %.not39, float %1689, float %1683, !dbg !16
  %1694 = bitcast float %1693 to i32, !dbg !16
  %1695 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1277, i32 %1694), !dbg !16
  br i1 true, label %1696, label %1760, !dbg !21

1696:                                             ; preds = %1245
  %1697 = bitcast i32 %1692 to float, !dbg !16
  %1698 = select i1 %.not39, float %1682, float %1697, !dbg !16
  %1699 = insertelement <4 x float> poison, float %1698, i64 0, !dbg !19
  %1700 = bitcast i32 %1695 to float, !dbg !16
  %1701 = select i1 %.not39, float %1683, float %1700, !dbg !16
  %1702 = insertelement <4 x float> %1699, float %1701, i64 1, !dbg !19
  %1703 = select i1 %.not39, float %1697, float %1688, !dbg !16
  %1704 = insertelement <4 x float> %1702, float %1703, i64 2, !dbg !19
  %1705 = select i1 %.not39, float %1700, float %1689, !dbg !16
  %1706 = insertelement <4 x float> %1704, float %1705, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1589, !dbg !19
  %normalized = fmul <4 x float> %1706, %row.reciprocal, !dbg !19
  %1707 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1708 = extractelement <4 x half> %1707, i64 3, !dbg !20
  %1709 = extractelement <4 x half> %1707, i64 2, !dbg !20
  %1710 = extractelement <4 x half> %1707, i64 1, !dbg !20
  %1711 = extractelement <4 x half> %1707, i64 0, !dbg !20
  %1712 = bitcast i32 %1584 to float, !dbg !16
  %1713 = select i1 %.not39, float %1574, float %1712, !dbg !16
  %1714 = insertelement <4 x float> poison, float %1713, i64 0, !dbg !19
  %1715 = bitcast i32 %1587 to float, !dbg !16
  %1716 = select i1 %.not39, float %1575, float %1715, !dbg !16
  %1717 = insertelement <4 x float> %1714, float %1716, i64 1, !dbg !19
  %1718 = select i1 %.not39, float %1712, float %1580, !dbg !16
  %1719 = insertelement <4 x float> %1717, float %1718, i64 2, !dbg !19
  %1720 = select i1 %.not39, float %1715, float %1581, !dbg !16
  %1721 = insertelement <4 x float> %1719, float %1720, i64 3, !dbg !19
  %normalized124 = fmul <4 x float> %1721, %row.reciprocal, !dbg !19
  %1722 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %1723 = extractelement <4 x half> %1722, i64 3, !dbg !20
  %1724 = extractelement <4 x half> %1722, i64 2, !dbg !20
  %1725 = extractelement <4 x half> %1722, i64 1, !dbg !20
  %1726 = extractelement <4 x half> %1722, i64 0, !dbg !20
  %1727 = bitcast i32 %1474 to float, !dbg !16
  %1728 = select i1 %.not39, float %1464, float %1727, !dbg !16
  %1729 = insertelement <4 x float> poison, float %1728, i64 0, !dbg !19
  %1730 = bitcast i32 %1477 to float, !dbg !16
  %1731 = select i1 %.not39, float %1465, float %1730, !dbg !16
  %1732 = insertelement <4 x float> %1729, float %1731, i64 1, !dbg !19
  %1733 = select i1 %.not39, float %1727, float %1470, !dbg !16
  %1734 = insertelement <4 x float> %1732, float %1733, i64 2, !dbg !19
  %1735 = select i1 %.not39, float %1730, float %1471, !dbg !16
  %1736 = insertelement <4 x float> %1734, float %1735, i64 3, !dbg !19
  %row.reciprocal125 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1371, !dbg !19
  %normalized126 = fmul <4 x float> %1736, %row.reciprocal125, !dbg !19
  %1737 = fptrunc <4 x float> %normalized126 to <4 x half>, !dbg !20
  %1738 = extractelement <4 x half> %1737, i64 3, !dbg !20
  %1739 = extractelement <4 x half> %1737, i64 2, !dbg !20
  %1740 = extractelement <4 x half> %1737, i64 1, !dbg !20
  %1741 = extractelement <4 x half> %1737, i64 0, !dbg !20
  %1742 = bitcast i32 %1366 to float, !dbg !16
  %1743 = select i1 %.not39, float %1356, float %1742, !dbg !16
  %1744 = insertelement <4 x float> poison, float %1743, i64 0, !dbg !19
  %1745 = bitcast i32 %1369 to float, !dbg !16
  %1746 = select i1 %.not39, float %1357, float %1745, !dbg !16
  %1747 = insertelement <4 x float> %1744, float %1746, i64 1, !dbg !19
  %1748 = select i1 %.not39, float %1742, float %1362, !dbg !16
  %1749 = insertelement <4 x float> %1747, float %1748, i64 2, !dbg !19
  %1750 = select i1 %.not39, float %1745, float %1363, !dbg !16
  %1751 = insertelement <4 x float> %1749, float %1750, i64 3, !dbg !19
  %normalized127 = fmul <4 x float> %1751, %row.reciprocal125, !dbg !19
  %1752 = fptrunc <4 x float> %normalized127 to <4 x half>, !dbg !20
  %1753 = extractelement <4 x half> %1752, i64 3, !dbg !20
  %1754 = extractelement <4 x half> %1752, i64 2, !dbg !20
  %1755 = extractelement <4 x half> %1752, i64 1, !dbg !20
  %1756 = extractelement <4 x half> %1752, i64 0, !dbg !20
  %1757 = add nsw i64 %8, %13, !dbg !21
  %1758 = shl nsw i64 %1757, 19, !dbg !21
  %1759 = and i64 %1758, 16252928, !dbg !21
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
  %address.local1074 = add nuw nsw i64 %1759, %address.local1073, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1076 = lshr i64 %address.local1065, 4, !dbg !14
  %address.local1077 = and i64 %address.local1076, 3, !dbg !14
  %address.local1078 = shl nuw nsw i64 %address.local1077, 2, !dbg !21
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1078, !dbg !21
  store half %1756, ptr addrspace(1) %address.local1079, align 2, !dbg !21
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1081 = or disjoint i64 %address.local1078, 1, !dbg !21
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1080, i64 %address.local1081, !dbg !21
  store half %1755, ptr addrspace(1) %address.local1082, align 2, !dbg !21
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1084 = or disjoint i64 %address.local1078, 2, !dbg !21
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1083, i64 %address.local1084, !dbg !21
  store half %1754, ptr addrspace(1) %address.local1085, align 2, !dbg !21
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1087 = or disjoint i64 %address.local1078, 3, !dbg !21
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1086, i64 %address.local1087, !dbg !21
  store half %1753, ptr addrspace(1) %address.local1088, align 2, !dbg !21
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1090 = or disjoint i64 %address.local1078, 16, !dbg !21
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1090, !dbg !21
  store half %1741, ptr addrspace(1) %address.local1091, align 2, !dbg !21
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1093 = or disjoint i64 %address.local1078, 17, !dbg !21
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1092, i64 %address.local1093, !dbg !21
  store half %1740, ptr addrspace(1) %address.local1094, align 2, !dbg !21
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1096 = or disjoint i64 %address.local1078, 18, !dbg !21
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1096, !dbg !21
  store half %1739, ptr addrspace(1) %address.local1097, align 2, !dbg !21
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1074, !dbg !21
  %address.local1099 = or disjoint i64 %address.local1078, 19, !dbg !21
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1098, i64 %address.local1099, !dbg !21
  store half %1738, ptr addrspace(1) %address.local1100, align 2, !dbg !21
  %address.local1101 = shl nsw i64 %address.local1072, 7, !dbg !21
  %address.local1102 = add i64 %address.local1101, 8192, !dbg !21
  %address.local1103 = add nuw nsw i64 %1759, %address.local1102, !dbg !21
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1078, !dbg !21
  store half %1726, ptr addrspace(1) %address.local1105, align 2, !dbg !21
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1106, i64 %address.local1081, !dbg !21
  store half %1725, ptr addrspace(1) %address.local1107, align 2, !dbg !21
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1108, i64 %address.local1084, !dbg !21
  store half %1724, ptr addrspace(1) %address.local1109, align 2, !dbg !21
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1110, i64 %address.local1087, !dbg !21
  store half %1723, ptr addrspace(1) %address.local1111, align 2, !dbg !21
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1090, !dbg !21
  store half %1711, ptr addrspace(1) %address.local1113, align 2, !dbg !21
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1114, i64 %address.local1093, !dbg !21
  store half %1710, ptr addrspace(1) %address.local1115, align 2, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1116, i64 %address.local1096, !dbg !21
  store half %1709, ptr addrspace(1) %address.local1117, align 2, !dbg !21
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1103, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1118, i64 %address.local1099, !dbg !21
  store half %1708, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  br label %1760, !dbg !21

1760:                                             ; preds = %1696, %1245
  br i1 true, label %1761, label %1799, !dbg !21

1761:                                             ; preds = %1760
  %1762 = insertelement <4 x float> poison, float %1666, i64 0, !dbg !19
  %1763 = insertelement <4 x float> %1762, float %1672, i64 1, !dbg !19
  %1764 = insertelement <4 x float> %1763, float %1667, i64 2, !dbg !19
  %1765 = insertelement <4 x float> %1764, float %1673, i64 3, !dbg !19
  %row.reciprocal128 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1589, !dbg !19
  %normalized129 = fmul <4 x float> %1765, %row.reciprocal128, !dbg !19
  %1766 = fptrunc <4 x float> %normalized129 to <4 x half>, !dbg !20
  %1767 = extractelement <4 x half> %1766, i64 3, !dbg !20
  %1768 = extractelement <4 x half> %1766, i64 2, !dbg !20
  %1769 = extractelement <4 x half> %1766, i64 1, !dbg !20
  %1770 = extractelement <4 x half> %1766, i64 0, !dbg !20
  %1771 = insertelement <4 x float> poison, float %1557, i64 0, !dbg !19
  %1772 = insertelement <4 x float> %1771, float %1563, i64 1, !dbg !19
  %1773 = insertelement <4 x float> %1772, float %1558, i64 2, !dbg !19
  %1774 = insertelement <4 x float> %1773, float %1564, i64 3, !dbg !19
  %normalized130 = fmul <4 x float> %1774, %row.reciprocal128, !dbg !19
  %1775 = fptrunc <4 x float> %normalized130 to <4 x half>, !dbg !20
  %1776 = extractelement <4 x half> %1775, i64 3, !dbg !20
  %1777 = extractelement <4 x half> %1775, i64 2, !dbg !20
  %1778 = extractelement <4 x half> %1775, i64 1, !dbg !20
  %1779 = extractelement <4 x half> %1775, i64 0, !dbg !20
  %1780 = insertelement <4 x float> poison, float %1448, i64 0, !dbg !19
  %1781 = insertelement <4 x float> %1780, float %1454, i64 1, !dbg !19
  %1782 = insertelement <4 x float> %1781, float %1449, i64 2, !dbg !19
  %1783 = insertelement <4 x float> %1782, float %1455, i64 3, !dbg !19
  %row.reciprocal131 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1371, !dbg !19
  %normalized132 = fmul <4 x float> %1783, %row.reciprocal131, !dbg !19
  %1784 = fptrunc <4 x float> %normalized132 to <4 x half>, !dbg !20
  %1785 = extractelement <4 x half> %1784, i64 3, !dbg !20
  %1786 = extractelement <4 x half> %1784, i64 2, !dbg !20
  %1787 = extractelement <4 x half> %1784, i64 1, !dbg !20
  %1788 = extractelement <4 x half> %1784, i64 0, !dbg !20
  %1789 = insertelement <4 x float> poison, float %1339, i64 0, !dbg !19
  %1790 = insertelement <4 x float> %1789, float %1345, i64 1, !dbg !19
  %1791 = insertelement <4 x float> %1790, float %1340, i64 2, !dbg !19
  %1792 = insertelement <4 x float> %1791, float %1346, i64 3, !dbg !19
  %normalized133 = fmul <4 x float> %1792, %row.reciprocal131, !dbg !19
  %1793 = fptrunc <4 x float> %normalized133 to <4 x half>, !dbg !20
  %1794 = extractelement <4 x half> %1793, i64 3, !dbg !20
  %1795 = extractelement <4 x half> %1793, i64 2, !dbg !20
  %1796 = extractelement <4 x half> %1793, i64 1, !dbg !20
  %1797 = extractelement <4 x half> %1793, i64 0, !dbg !20
  %1798 = shl nuw nsw i64 %15, 19, !dbg !21
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
  %address.local1130 = add nuw nsw i64 %1798, %address.local1129, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1132 = lshr i64 %address.local1121, 4, !dbg !14
  %address.local1133 = and i64 %address.local1132, 3, !dbg !14
  %address.local1134 = shl nuw nsw i64 %address.local1133, 2, !dbg !21
  %address.local1135 = or disjoint i64 %address.local1134, 32, !dbg !21
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1135, !dbg !21
  store half %1797, ptr addrspace(1) %address.local1136, align 2, !dbg !21
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1138 = or disjoint i64 %address.local1134, 33, !dbg !21
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1138, !dbg !21
  store half %1796, ptr addrspace(1) %address.local1139, align 2, !dbg !21
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1141 = or disjoint i64 %address.local1134, 34, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1140, i64 %address.local1141, !dbg !21
  store half %1795, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1144 = or disjoint i64 %address.local1134, 35, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1144, !dbg !21
  store half %1794, ptr addrspace(1) %address.local1145, align 2, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1147 = or disjoint i64 %address.local1134, 48, !dbg !21
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1146, i64 %address.local1147, !dbg !21
  store half %1788, ptr addrspace(1) %address.local1148, align 2, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1150 = or disjoint i64 %address.local1134, 49, !dbg !21
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1150, !dbg !21
  store half %1787, ptr addrspace(1) %address.local1151, align 2, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1153 = or disjoint i64 %address.local1134, 50, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1152, i64 %address.local1153, !dbg !21
  store half %1786, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1130, !dbg !21
  %address.local1156 = or disjoint i64 %address.local1134, 51, !dbg !21
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1156, !dbg !21
  store half %1785, ptr addrspace(1) %address.local1157, align 2, !dbg !21
  %address.local1158 = shl nsw i64 %address.local1128, 7, !dbg !21
  %address.local1159 = add i64 %address.local1158, 8192, !dbg !21
  %address.local1160 = add nuw nsw i64 %1798, %address.local1159, !dbg !21
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1161, i64 %address.local1135, !dbg !21
  store half %1779, ptr addrspace(1) %address.local1162, align 2, !dbg !21
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1163, i64 %address.local1138, !dbg !21
  store half %1778, ptr addrspace(1) %address.local1164, align 2, !dbg !21
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1165, i64 %address.local1141, !dbg !21
  store half %1777, ptr addrspace(1) %address.local1166, align 2, !dbg !21
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1167, i64 %address.local1144, !dbg !21
  store half %1776, ptr addrspace(1) %address.local1168, align 2, !dbg !21
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1169, i64 %address.local1147, !dbg !21
  store half %1770, ptr addrspace(1) %address.local1170, align 2, !dbg !21
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1150, !dbg !21
  store half %1769, ptr addrspace(1) %address.local1172, align 2, !dbg !21
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1153, !dbg !21
  store half %1768, ptr addrspace(1) %address.local1174, align 2, !dbg !21
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1160, !dbg !21
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1156, !dbg !21
  store half %1767, ptr addrspace(1) %address.local1176, align 2, !dbg !21
  br label %1799, !dbg !21

1799:                                             ; preds = %1761, %1760
  br i1 true, label %1800, label %1838, !dbg !21

1800:                                             ; preds = %1799
  %1801 = insertelement <4 x float> poison, float %1638, i64 0, !dbg !19
  %1802 = insertelement <4 x float> %1801, float %1644, i64 1, !dbg !19
  %1803 = insertelement <4 x float> %1802, float %1639, i64 2, !dbg !19
  %1804 = insertelement <4 x float> %1803, float %1645, i64 3, !dbg !19
  %row.reciprocal134 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1589, !dbg !19
  %normalized135 = fmul <4 x float> %1804, %row.reciprocal134, !dbg !19
  %1805 = fptrunc <4 x float> %normalized135 to <4 x half>, !dbg !20
  %1806 = extractelement <4 x half> %1805, i64 3, !dbg !20
  %1807 = extractelement <4 x half> %1805, i64 2, !dbg !20
  %1808 = extractelement <4 x half> %1805, i64 1, !dbg !20
  %1809 = extractelement <4 x half> %1805, i64 0, !dbg !20
  %1810 = insertelement <4 x float> poison, float %1528, i64 0, !dbg !19
  %1811 = insertelement <4 x float> %1810, float %1534, i64 1, !dbg !19
  %1812 = insertelement <4 x float> %1811, float %1529, i64 2, !dbg !19
  %1813 = insertelement <4 x float> %1812, float %1535, i64 3, !dbg !19
  %normalized136 = fmul <4 x float> %1813, %row.reciprocal134, !dbg !19
  %1814 = fptrunc <4 x float> %normalized136 to <4 x half>, !dbg !20
  %1815 = extractelement <4 x half> %1814, i64 3, !dbg !20
  %1816 = extractelement <4 x half> %1814, i64 2, !dbg !20
  %1817 = extractelement <4 x half> %1814, i64 1, !dbg !20
  %1818 = extractelement <4 x half> %1814, i64 0, !dbg !20
  %1819 = insertelement <4 x float> poison, float %1420, i64 0, !dbg !19
  %1820 = insertelement <4 x float> %1819, float %1426, i64 1, !dbg !19
  %1821 = insertelement <4 x float> %1820, float %1421, i64 2, !dbg !19
  %1822 = insertelement <4 x float> %1821, float %1427, i64 3, !dbg !19
  %row.reciprocal137 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1371, !dbg !19
  %normalized138 = fmul <4 x float> %1822, %row.reciprocal137, !dbg !19
  %1823 = fptrunc <4 x float> %normalized138 to <4 x half>, !dbg !20
  %1824 = extractelement <4 x half> %1823, i64 3, !dbg !20
  %1825 = extractelement <4 x half> %1823, i64 2, !dbg !20
  %1826 = extractelement <4 x half> %1823, i64 1, !dbg !20
  %1827 = extractelement <4 x half> %1823, i64 0, !dbg !20
  %1828 = insertelement <4 x float> poison, float %1310, i64 0, !dbg !19
  %1829 = insertelement <4 x float> %1828, float %1316, i64 1, !dbg !19
  %1830 = insertelement <4 x float> %1829, float %1311, i64 2, !dbg !19
  %1831 = insertelement <4 x float> %1830, float %1317, i64 3, !dbg !19
  %normalized139 = fmul <4 x float> %1831, %row.reciprocal137, !dbg !19
  %1832 = fptrunc <4 x float> %normalized139 to <4 x half>, !dbg !20
  %1833 = extractelement <4 x half> %1832, i64 3, !dbg !20
  %1834 = extractelement <4 x half> %1832, i64 2, !dbg !20
  %1835 = extractelement <4 x half> %1832, i64 1, !dbg !20
  %1836 = extractelement <4 x half> %1832, i64 0, !dbg !20
  %1837 = shl nuw nsw i64 %15, 19, !dbg !21
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
  %address.local1187 = add nuw nsw i64 %1837, %address.local1186, !dbg !21
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1189 = lshr i64 %address.local1178, 4, !dbg !14
  %address.local1190 = and i64 %address.local1189, 3, !dbg !14
  %address.local1191 = shl nuw nsw i64 %address.local1190, 2, !dbg !21
  %address.local1192 = or disjoint i64 %address.local1191, 64, !dbg !21
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1188, i64 %address.local1192, !dbg !21
  store half %1836, ptr addrspace(1) %address.local1193, align 2, !dbg !21
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1195 = or disjoint i64 %address.local1191, 65, !dbg !21
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1194, i64 %address.local1195, !dbg !21
  store half %1835, ptr addrspace(1) %address.local1196, align 2, !dbg !21
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1198 = or disjoint i64 %address.local1191, 66, !dbg !21
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1197, i64 %address.local1198, !dbg !21
  store half %1834, ptr addrspace(1) %address.local1199, align 2, !dbg !21
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1201 = or disjoint i64 %address.local1191, 67, !dbg !21
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1200, i64 %address.local1201, !dbg !21
  store half %1833, ptr addrspace(1) %address.local1202, align 2, !dbg !21
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1204 = or disjoint i64 %address.local1191, 80, !dbg !21
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1203, i64 %address.local1204, !dbg !21
  store half %1827, ptr addrspace(1) %address.local1205, align 2, !dbg !21
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1207 = or disjoint i64 %address.local1191, 81, !dbg !21
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1206, i64 %address.local1207, !dbg !21
  store half %1826, ptr addrspace(1) %address.local1208, align 2, !dbg !21
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1210 = or disjoint i64 %address.local1191, 82, !dbg !21
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1209, i64 %address.local1210, !dbg !21
  store half %1825, ptr addrspace(1) %address.local1211, align 2, !dbg !21
  %address.local1212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1187, !dbg !21
  %address.local1213 = or disjoint i64 %address.local1191, 83, !dbg !21
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1212, i64 %address.local1213, !dbg !21
  store half %1824, ptr addrspace(1) %address.local1214, align 2, !dbg !21
  %address.local1215 = shl nsw i64 %address.local1185, 7, !dbg !21
  %address.local1216 = add i64 %address.local1215, 8192, !dbg !21
  %address.local1217 = add nuw nsw i64 %1837, %address.local1216, !dbg !21
  %address.local1218 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1218, i64 %address.local1192, !dbg !21
  store half %1818, ptr addrspace(1) %address.local1219, align 2, !dbg !21
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1221 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1220, i64 %address.local1195, !dbg !21
  store half %1817, ptr addrspace(1) %address.local1221, align 2, !dbg !21
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1222, i64 %address.local1198, !dbg !21
  store half %1816, ptr addrspace(1) %address.local1223, align 2, !dbg !21
  %address.local1224 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1224, i64 %address.local1201, !dbg !21
  store half %1815, ptr addrspace(1) %address.local1225, align 2, !dbg !21
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1226, i64 %address.local1204, !dbg !21
  store half %1809, ptr addrspace(1) %address.local1227, align 2, !dbg !21
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1228, i64 %address.local1207, !dbg !21
  store half %1808, ptr addrspace(1) %address.local1229, align 2, !dbg !21
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1230, i64 %address.local1210, !dbg !21
  store half %1807, ptr addrspace(1) %address.local1231, align 2, !dbg !21
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1217, !dbg !21
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1232, i64 %address.local1213, !dbg !21
  store half %1806, ptr addrspace(1) %address.local1233, align 2, !dbg !21
  br label %1838, !dbg !21

1838:                                             ; preds = %1800, %1799
  br i1 true, label %1839, label %1877, !dbg !21

1839:                                             ; preds = %1838
  %1840 = insertelement <4 x float> poison, float %1610, i64 0, !dbg !19
  %1841 = insertelement <4 x float> %1840, float %1616, i64 1, !dbg !19
  %1842 = insertelement <4 x float> %1841, float %1611, i64 2, !dbg !19
  %1843 = insertelement <4 x float> %1842, float %1617, i64 3, !dbg !19
  %row.reciprocal140 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1589, !dbg !19
  %normalized141 = fmul <4 x float> %1843, %row.reciprocal140, !dbg !19
  %1844 = fptrunc <4 x float> %normalized141 to <4 x half>, !dbg !20
  %1845 = extractelement <4 x half> %1844, i64 3, !dbg !20
  %1846 = extractelement <4 x half> %1844, i64 2, !dbg !20
  %1847 = extractelement <4 x half> %1844, i64 1, !dbg !20
  %1848 = extractelement <4 x half> %1844, i64 0, !dbg !20
  %1849 = insertelement <4 x float> poison, float %1499, i64 0, !dbg !19
  %1850 = insertelement <4 x float> %1849, float %1505, i64 1, !dbg !19
  %1851 = insertelement <4 x float> %1850, float %1500, i64 2, !dbg !19
  %1852 = insertelement <4 x float> %1851, float %1506, i64 3, !dbg !19
  %normalized142 = fmul <4 x float> %1852, %row.reciprocal140, !dbg !19
  %1853 = fptrunc <4 x float> %normalized142 to <4 x half>, !dbg !20
  %1854 = extractelement <4 x half> %1853, i64 3, !dbg !20
  %1855 = extractelement <4 x half> %1853, i64 2, !dbg !20
  %1856 = extractelement <4 x half> %1853, i64 1, !dbg !20
  %1857 = extractelement <4 x half> %1853, i64 0, !dbg !20
  %1858 = insertelement <4 x float> poison, float %1392, i64 0, !dbg !19
  %1859 = insertelement <4 x float> %1858, float %1398, i64 1, !dbg !19
  %1860 = insertelement <4 x float> %1859, float %1393, i64 2, !dbg !19
  %1861 = insertelement <4 x float> %1860, float %1399, i64 3, !dbg !19
  %row.reciprocal143 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1371, !dbg !19
  %normalized144 = fmul <4 x float> %1861, %row.reciprocal143, !dbg !19
  %1862 = fptrunc <4 x float> %normalized144 to <4 x half>, !dbg !20
  %1863 = extractelement <4 x half> %1862, i64 3, !dbg !20
  %1864 = extractelement <4 x half> %1862, i64 2, !dbg !20
  %1865 = extractelement <4 x half> %1862, i64 1, !dbg !20
  %1866 = extractelement <4 x half> %1862, i64 0, !dbg !20
  %1867 = insertelement <4 x float> poison, float %1281, i64 0, !dbg !19
  %1868 = insertelement <4 x float> %1867, float %1287, i64 1, !dbg !19
  %1869 = insertelement <4 x float> %1868, float %1282, i64 2, !dbg !19
  %1870 = insertelement <4 x float> %1869, float %1288, i64 3, !dbg !19
  %normalized145 = fmul <4 x float> %1870, %row.reciprocal143, !dbg !19
  %1871 = fptrunc <4 x float> %normalized145 to <4 x half>, !dbg !20
  %1872 = extractelement <4 x half> %1871, i64 3, !dbg !20
  %1873 = extractelement <4 x half> %1871, i64 2, !dbg !20
  %1874 = extractelement <4 x half> %1871, i64 1, !dbg !20
  %1875 = extractelement <4 x half> %1871, i64 0, !dbg !20
  %1876 = shl nuw nsw i64 %15, 19, !dbg !21
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
  %address.local1244 = add nuw nsw i64 %1876, %address.local1243, !dbg !21
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1246 = lshr i64 %address.local1235, 4, !dbg !14
  %address.local1247 = and i64 %address.local1246, 3, !dbg !14
  %address.local1248 = shl nuw nsw i64 %address.local1247, 2, !dbg !21
  %address.local1249 = or disjoint i64 %address.local1248, 96, !dbg !21
  %address.local1250 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1245, i64 %address.local1249, !dbg !21
  store half %1875, ptr addrspace(1) %address.local1250, align 2, !dbg !21
  %address.local1251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1252 = or disjoint i64 %address.local1248, 97, !dbg !21
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1251, i64 %address.local1252, !dbg !21
  store half %1874, ptr addrspace(1) %address.local1253, align 2, !dbg !21
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1255 = or disjoint i64 %address.local1248, 98, !dbg !21
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1254, i64 %address.local1255, !dbg !21
  store half %1873, ptr addrspace(1) %address.local1256, align 2, !dbg !21
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1258 = or disjoint i64 %address.local1248, 99, !dbg !21
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1258, !dbg !21
  store half %1872, ptr addrspace(1) %address.local1259, align 2, !dbg !21
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1261 = or disjoint i64 %address.local1248, 112, !dbg !21
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1260, i64 %address.local1261, !dbg !21
  store half %1866, ptr addrspace(1) %address.local1262, align 2, !dbg !21
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1264 = or disjoint i64 %address.local1248, 113, !dbg !21
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1264, !dbg !21
  store half %1865, ptr addrspace(1) %address.local1265, align 2, !dbg !21
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1267 = or disjoint i64 %address.local1248, 114, !dbg !21
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1266, i64 %address.local1267, !dbg !21
  store half %1864, ptr addrspace(1) %address.local1268, align 2, !dbg !21
  %address.local1269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1244, !dbg !21
  %address.local1270 = or disjoint i64 %address.local1248, 115, !dbg !21
  %address.local1271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1269, i64 %address.local1270, !dbg !21
  store half %1863, ptr addrspace(1) %address.local1271, align 2, !dbg !21
  %address.local1272 = shl nsw i64 %address.local1242, 7, !dbg !21
  %address.local1273 = add i64 %address.local1272, 8192, !dbg !21
  %address.local1274 = add nuw nsw i64 %1876, %address.local1273, !dbg !21
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1275, i64 %address.local1249, !dbg !21
  store half %1857, ptr addrspace(1) %address.local1276, align 2, !dbg !21
  %address.local1277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1277, i64 %address.local1252, !dbg !21
  store half %1856, ptr addrspace(1) %address.local1278, align 2, !dbg !21
  %address.local1279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1279, i64 %address.local1255, !dbg !21
  store half %1855, ptr addrspace(1) %address.local1280, align 2, !dbg !21
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1281, i64 %address.local1258, !dbg !21
  store half %1854, ptr addrspace(1) %address.local1282, align 2, !dbg !21
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1283, i64 %address.local1261, !dbg !21
  store half %1848, ptr addrspace(1) %address.local1284, align 2, !dbg !21
  %address.local1285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1285, i64 %address.local1264, !dbg !21
  store half %1847, ptr addrspace(1) %address.local1286, align 2, !dbg !21
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1287, i64 %address.local1267, !dbg !21
  store half %1846, ptr addrspace(1) %address.local1288, align 2, !dbg !21
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1274, !dbg !21
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1289, i64 %address.local1270, !dbg !21
  store half %1845, ptr addrspace(1) %address.local1290, align 2, !dbg !21
  br label %1877, !dbg !21

1877:                                             ; preds = %1839, %1838
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

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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
