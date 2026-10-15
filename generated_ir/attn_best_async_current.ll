; ModuleID = '/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD.mlir'
source_filename = "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD.mlir"

@async_lds_staging_0 = private addrspace(3) global [4096 x half] undef, align 16
@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #0 !dbg !4 !reqd_work_group_size !31 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %6, 7, !dbg !12
  %.lobit = ashr i32 %5, 31, !dbg !14
  %10 = xor i32 %5, %.lobit, !dbg !14
  %11 = sdiv i32 %10, 32, !dbg !14
  %.lobit1 = ashr i32 %5, 31, !dbg !14
  %12 = xor i32 %11, %.lobit1, !dbg !14
  %13 = sext i32 %12 to i64, !dbg !14
  %14 = add nsw i64 %8, %13, !dbg !14
  %15 = and i64 %14, 31, !dbg !14
  %16 = shl nsw i64 %6, 7, !dbg !14
  %17 = mul nsw i64 %13, -4096, !dbg !14
  %18 = add nsw i64 %16, %17, !dbg !14
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !6, !range !33
  %20 = zext nneg i32 %19 to i64, !dbg !6
  %21 = shl nuw nsw i64 %20, 2, !dbg !6
  %22 = lshr i64 %20, 6, !dbg !6
  %23 = lshr i64 %20, 4, !dbg !6
  %24 = and i64 %23, 3, !dbg !6
  %25 = or disjoint i64 %9, 127, !dbg !15
  %26 = add nsw i64 %9, 128, !dbg !15
  %27 = icmp uge i64 %26, %25, !dbg !15
  %28 = lshr i64 %20, 2, !dbg !8
  %29 = and i64 %20, 15, !dbg !8
  %30 = lshr i64 %29, 2, !dbg !8
  %31 = lshr i64 %20, 1, !dbg !8
  %32 = lshr i64 %24, 1, !dbg !8
  %33 = mul nsw i64 %28, -18, !dbg !8
  %34 = add nsw i64 %21, %33, !dbg !8
  %35 = add nsw i64 %34, %31, !dbg !8
  %36 = add nsw i64 %35, %32, !dbg !8
  %.neg231 = sdiv i64 %36, -2, !dbg !8
  %37 = mul nuw nsw i64 %20, 18, !dbg !8
  %38 = mul nsw i64 %28, -72, !dbg !8
  %39 = add nsw i64 %37, %38, !dbg !8
  %40 = add nsw i64 %39, %30, !dbg !8
  %.neg233 = sdiv i64 %40, -4, !dbg !8
  %41 = shl nuw nsw i64 %20, 4, !dbg !8
  %42 = mul nsw i64 %28, -64, !dbg !8
  %43 = add nsw i64 %41, %42, !dbg !8
  %44 = or disjoint i64 %43, %30, !dbg !8
  %45 = shl nuw nsw i32 %19, 1, !dbg !8
  %46 = and i32 %45, 6, !dbg !8
  %index.compact210 = zext nneg i32 %46 to i64, !dbg !8
  %47 = lshr i32 %19, 1, !dbg !8
  %48 = and i32 %47, 1, !dbg !8
  %index.compact207 = zext nneg i32 %48 to i64, !dbg !8
  %49 = mul nsw i64 %28, -16, !dbg !8
  %50 = add nsw i64 %21, %49, !dbg !8
  %51 = or disjoint i64 %50, %index.compact207, !dbg !8
  %52 = add nuw nsw i64 %51, %32, !dbg !8
  %53 = add nsw i64 %52, 1, !dbg !8
  %.neg236 = sdiv i64 %53, -2, !dbg !8
  %54 = add nuw nsw i64 %44, %index.compact210, !dbg !8
  %.neg238 = sdiv i64 %54, -4, !dbg !8
  br label %55, !dbg !16

55:                                               ; preds = %1664, %4
  %56 = phi i64 [ %1771, %1664 ], [ 0, %4 ], !dbg !16
  %57 = phi [2 x <8 x float>] [ %1731, %1664 ], [ zeroinitializer, %4 ], !dbg !16
  %58 = phi [2 x <1 x float>] [ %1735, %1664 ], [ zeroinitializer, %4 ], !dbg !16
  %59 = phi [2 x <8 x float>] [ %1742, %1664 ], [ zeroinitializer, %4 ], !dbg !16
  %60 = phi [2 x <8 x float>] [ %1748, %1664 ], [ zeroinitializer, %4 ], !dbg !16
  %61 = phi [2 x <8 x float>] [ %1770, %1664 ], [ zeroinitializer, %4 ], !dbg !16
  %.not = icmp sgt i64 %56, %6, !dbg !16
  br i1 %.not, label %1772, label %62, !dbg !16

62:                                               ; preds = %55
  %63 = shl nsw i64 %56, 7, !dbg !16
  %64 = lshr i64 %56, 5, !dbg !17
  %65 = shl i64 %64, 12, !dbg !17
  %66 = sub i64 %63, %65, !dbg !17
  br label %67, !dbg !22

67:                                               ; preds = %71, %62
  %68 = phi i64 [ %278, %71 ], [ 0, %62 ], !dbg !22
  %69 = phi [2 x <32 x float>] [ %277, %71 ], [ zeroinitializer, %62 ], !dbg !22
  %70 = icmp samesign ult i64 %68, 4, !dbg !22
  br i1 %70, label %71, label %279, !dbg !22

71:                                               ; preds = %67
  %72 = shl nuw nsw i64 %68, 5, !dbg !22
  %.idx110 = shl nuw nsw i64 %15, 20, !dbg !14
  %73 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx110, !dbg !14
  %.idx111 = shl nsw i64 %18, 8, !dbg !14
  %74 = getelementptr i8, ptr addrspace(1) %73, i64 %.idx111, !dbg !14
  %75 = getelementptr [2 x i8], ptr addrspace(1) %74, i64 %72, !dbg !14
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !14, !range !33, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !6
  %address.local303 = shl nuw nsw i64 %address.local, 1, !dbg !14
  %address.local304 = and i64 %address.local303, 384, !dbg !14
  %address.local305 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local304, !dbg !14
  %address.local306 = ptrtoint ptr addrspace(3) %address.local305 to i64, !dbg !14
  %address.local307 = trunc i64 %address.local306 to i32, !dbg !14
  %76 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local307), !dbg !14
  %77 = zext i32 %76 to i64, !dbg !14
  %78 = inttoptr i64 %77 to ptr addrspace(3), !dbg !14
  %address.local308 = shl nuw nsw i64 %address.local, 6, !dbg !14
  %address.local309 = and i64 %address.local308, 16256, !dbg !14
  %address.local310 = getelementptr [2 x i8], ptr addrspace(1) %75, i64 %address.local309, !dbg !14
  %address.local311 = shl nuw nsw i64 %address.local, 1, !dbg !14
  %address.local312 = and i64 %address.local311, 2, !dbg !14
  %address.local313 = getelementptr [2 x i8], ptr addrspace(1) %address.local310, i64 %address.local312, !dbg !14
  %dma.source.bits = ptrtoint ptr addrspace(1) %address.local313 to i64, !dbg !14
  %dma.source.opaque = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #9, !dbg !14
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1), !dbg !14
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %78, i32 4, i32 0, i32 0), !dbg !14
  %79 = getelementptr i8, ptr addrspace(1) %75, i64 8, !dbg !14
  %address.local314 = or disjoint i64 %address.local, 256, !dbg !14
  %address.local315 = shl nuw nsw i64 %address.local314, 1, !dbg !14
  %address.local316 = and i64 %address.local315, 896, !dbg !14
  %address.local317 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local316, !dbg !14
  %address.local318 = ptrtoint ptr addrspace(3) %address.local317 to i64, !dbg !14
  %address.local319 = trunc i64 %address.local318 to i32, !dbg !14
  %80 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local319), !dbg !14
  %81 = zext i32 %80 to i64, !dbg !14
  %82 = inttoptr i64 %81 to ptr addrspace(3), !dbg !14
  %address.local320 = shl nuw nsw i64 %address.local, 6, !dbg !14
  %address.local321 = and i64 %address.local320, 16256, !dbg !14
  %address.local322 = getelementptr [2 x i8], ptr addrspace(1) %79, i64 %address.local321, !dbg !14
  %address.local323 = shl nuw nsw i64 %address.local314, 1, !dbg !14
  %address.local324 = and i64 %address.local323, 2, !dbg !14
  %address.local325 = getelementptr [2 x i8], ptr addrspace(1) %address.local322, i64 %address.local324, !dbg !14
  %dma.source.bits1897 = ptrtoint ptr addrspace(1) %address.local325 to i64, !dbg !14
  %dma.source.opaque1898 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1897) #9, !dbg !14
  %dma.source.ptr1899 = inttoptr i64 %dma.source.opaque1898 to ptr addrspace(1), !dbg !14
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1899, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !14
  %83 = getelementptr i8, ptr addrspace(1) %75, i64 16, !dbg !14
  %address.local326 = or disjoint i64 %address.local, 512, !dbg !14
  %address.local327 = shl nuw nsw i64 %address.local326, 1, !dbg !14
  %address.local328 = and i64 %address.local327, 1408, !dbg !14
  %address.local329 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local328, !dbg !14
  %address.local330 = ptrtoint ptr addrspace(3) %address.local329 to i64, !dbg !14
  %address.local331 = trunc i64 %address.local330 to i32, !dbg !14
  %84 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local331), !dbg !14
  %85 = zext i32 %84 to i64, !dbg !14
  %86 = inttoptr i64 %85 to ptr addrspace(3), !dbg !14
  %address.local332 = shl nuw nsw i64 %address.local, 6, !dbg !14
  %address.local333 = and i64 %address.local332, 16256, !dbg !14
  %address.local334 = getelementptr [2 x i8], ptr addrspace(1) %83, i64 %address.local333, !dbg !14
  %address.local335 = shl nuw nsw i64 %address.local326, 1, !dbg !14
  %address.local336 = and i64 %address.local335, 2, !dbg !14
  %address.local337 = getelementptr [2 x i8], ptr addrspace(1) %address.local334, i64 %address.local336, !dbg !14
  %dma.source.bits1900 = ptrtoint ptr addrspace(1) %address.local337 to i64, !dbg !14
  %dma.source.opaque1901 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1900) #9, !dbg !14
  %dma.source.ptr1902 = inttoptr i64 %dma.source.opaque1901 to ptr addrspace(1), !dbg !14
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1902, ptr addrspace(3) %86, i32 4, i32 0, i32 0), !dbg !14
  %87 = getelementptr i8, ptr addrspace(1) %75, i64 24, !dbg !14
  %address.local338 = or disjoint i64 %address.local, 768, !dbg !14
  %address.local339 = shl nuw nsw i64 %address.local338, 1, !dbg !14
  %address.local340 = and i64 %address.local339, 1920, !dbg !14
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !14
  %address.local342 = ptrtoint ptr addrspace(3) %address.local341 to i64, !dbg !14
  %address.local343 = trunc i64 %address.local342 to i32, !dbg !14
  %88 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !14
  %89 = zext i32 %88 to i64, !dbg !14
  %90 = inttoptr i64 %89 to ptr addrspace(3), !dbg !14
  %address.local344 = shl nuw nsw i64 %address.local, 6, !dbg !14
  %address.local345 = and i64 %address.local344, 16256, !dbg !14
  %address.local346 = getelementptr [2 x i8], ptr addrspace(1) %87, i64 %address.local345, !dbg !14
  %address.local347 = shl nuw nsw i64 %address.local338, 1, !dbg !14
  %address.local348 = and i64 %address.local347, 2, !dbg !14
  %address.local349 = getelementptr [2 x i8], ptr addrspace(1) %address.local346, i64 %address.local348, !dbg !14
  %dma.source.bits1903 = ptrtoint ptr addrspace(1) %address.local349 to i64, !dbg !14
  %dma.source.opaque1904 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1903) #9, !dbg !14
  %dma.source.ptr1905 = inttoptr i64 %dma.source.opaque1904 to ptr addrspace(1), !dbg !14
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1905, ptr addrspace(3) %90, i32 4, i32 0, i32 0), !dbg !14
  %91 = getelementptr i8, ptr addrspace(1) %75, i64 32, !dbg !14
  %address.local350 = or disjoint i64 %address.local, 1024, !dbg !14
  %address.local351 = shl nuw nsw i64 %address.local350, 1, !dbg !14
  %address.local352 = and i64 %address.local351, 2432, !dbg !14
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !14
  %address.local354 = ptrtoint ptr addrspace(3) %address.local353 to i64, !dbg !14
  %address.local355 = trunc i64 %address.local354 to i32, !dbg !14
  %92 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !14
  %93 = zext i32 %92 to i64, !dbg !14
  %94 = inttoptr i64 %93 to ptr addrspace(3), !dbg !14
  %address.local356 = shl nuw nsw i64 %address.local, 6, !dbg !14
  %address.local357 = and i64 %address.local356, 16256, !dbg !14
  %address.local358 = getelementptr [2 x i8], ptr addrspace(1) %91, i64 %address.local357, !dbg !14
  %address.local359 = shl nuw nsw i64 %address.local350, 1, !dbg !14
  %address.local360 = and i64 %address.local359, 2, !dbg !14
  %address.local361 = getelementptr [2 x i8], ptr addrspace(1) %address.local358, i64 %address.local360, !dbg !14
  %dma.source.bits1906 = ptrtoint ptr addrspace(1) %address.local361 to i64, !dbg !14
  %dma.source.opaque1907 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1906) #9, !dbg !14
  %dma.source.ptr1908 = inttoptr i64 %dma.source.opaque1907 to ptr addrspace(1), !dbg !14
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1908, ptr addrspace(3) %94, i32 4, i32 0, i32 0), !dbg !14
  %95 = getelementptr i8, ptr addrspace(1) %75, i64 40, !dbg !14
  %address.local362 = or disjoint i64 %address.local, 1280, !dbg !14
  %address.local363 = shl nuw nsw i64 %address.local362, 1, !dbg !14
  %address.local364 = and i64 %address.local363, 2944, !dbg !14
  %address.local365 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local364, !dbg !14
  %address.local366 = ptrtoint ptr addrspace(3) %address.local365 to i64, !dbg !14
  %address.local367 = trunc i64 %address.local366 to i32, !dbg !14
  %96 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local367), !dbg !14
  %97 = zext i32 %96 to i64, !dbg !14
  %98 = inttoptr i64 %97 to ptr addrspace(3), !dbg !14
  %address.local368 = shl nuw nsw i64 %address.local, 6, !dbg !14
  %address.local369 = and i64 %address.local368, 16256, !dbg !14
  %address.local370 = getelementptr [2 x i8], ptr addrspace(1) %95, i64 %address.local369, !dbg !14
  %address.local371 = shl nuw nsw i64 %address.local362, 1, !dbg !14
  %address.local372 = and i64 %address.local371, 2, !dbg !14
  %address.local373 = getelementptr [2 x i8], ptr addrspace(1) %address.local370, i64 %address.local372, !dbg !14
  %dma.source.bits1909 = ptrtoint ptr addrspace(1) %address.local373 to i64, !dbg !14
  %dma.source.opaque1910 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1909) #9, !dbg !14
  %dma.source.ptr1911 = inttoptr i64 %dma.source.opaque1910 to ptr addrspace(1), !dbg !14
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1911, ptr addrspace(3) %98, i32 4, i32 0, i32 0), !dbg !14
  %99 = getelementptr i8, ptr addrspace(1) %75, i64 48, !dbg !14
  %address.local374 = or disjoint i64 %address.local, 1536, !dbg !14
  %address.local375 = shl nuw nsw i64 %address.local374, 1, !dbg !14
  %address.local376 = and i64 %address.local375, 3456, !dbg !14
  %address.local377 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local376, !dbg !14
  %address.local378 = ptrtoint ptr addrspace(3) %address.local377 to i64, !dbg !14
  %address.local379 = trunc i64 %address.local378 to i32, !dbg !14
  %100 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local379), !dbg !14
  %101 = zext i32 %100 to i64, !dbg !14
  %102 = inttoptr i64 %101 to ptr addrspace(3), !dbg !14
  %address.local380 = shl nuw nsw i64 %address.local, 6, !dbg !14
  %address.local381 = and i64 %address.local380, 16256, !dbg !14
  %address.local382 = getelementptr [2 x i8], ptr addrspace(1) %99, i64 %address.local381, !dbg !14
  %address.local383 = shl nuw nsw i64 %address.local374, 1, !dbg !14
  %address.local384 = and i64 %address.local383, 2, !dbg !14
  %address.local385 = getelementptr [2 x i8], ptr addrspace(1) %address.local382, i64 %address.local384, !dbg !14
  %dma.source.bits1912 = ptrtoint ptr addrspace(1) %address.local385 to i64, !dbg !14
  %dma.source.opaque1913 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1912) #9, !dbg !14
  %dma.source.ptr1914 = inttoptr i64 %dma.source.opaque1913 to ptr addrspace(1), !dbg !14
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1914, ptr addrspace(3) %102, i32 4, i32 0, i32 0), !dbg !14
  %103 = getelementptr i8, ptr addrspace(1) %75, i64 56, !dbg !14
  %address.local386 = or disjoint i64 %address.local, 1792, !dbg !14
  %address.local387 = shl nuw nsw i64 %address.local386, 1, !dbg !14
  %address.local388 = and i64 %address.local387, 3968, !dbg !14
  %address.local389 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local388, !dbg !14
  %address.local390 = ptrtoint ptr addrspace(3) %address.local389 to i64, !dbg !14
  %address.local391 = trunc i64 %address.local390 to i32, !dbg !14
  %104 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local391), !dbg !14
  %105 = zext i32 %104 to i64, !dbg !14
  %106 = inttoptr i64 %105 to ptr addrspace(3), !dbg !14
  %address.local392 = shl nuw nsw i64 %address.local, 6, !dbg !14
  %address.local393 = and i64 %address.local392, 16256, !dbg !14
  %address.local394 = getelementptr [2 x i8], ptr addrspace(1) %103, i64 %address.local393, !dbg !14
  %address.local395 = shl nuw nsw i64 %address.local386, 1, !dbg !14
  %address.local396 = and i64 %address.local395, 2, !dbg !14
  %address.local397 = getelementptr [2 x i8], ptr addrspace(1) %address.local394, i64 %address.local396, !dbg !14
  %dma.source.bits1915 = ptrtoint ptr addrspace(1) %address.local397 to i64, !dbg !14
  %dma.source.opaque1916 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1915) #9, !dbg !14
  %dma.source.ptr1917 = inttoptr i64 %dma.source.opaque1916 to ptr addrspace(1), !dbg !14
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1917, ptr addrspace(3) %106, i32 4, i32 0, i32 0), !dbg !14
  call void @llvm.amdgcn.asyncmark(), !dbg !14
  %107 = shl nuw nsw i64 %68, 10, !dbg !17
  %108 = add nuw i64 %56, %107, !dbg !17
  %.lobit112 = ashr i64 %108, 63, !dbg !17
  %109 = xor i64 %108, %.lobit112, !dbg !17
  %110 = sdiv i64 %109, 4096, !dbg !17
  %.lobit113 = ashr i64 %108, 63, !dbg !17
  %111 = xor i64 %110, %.lobit113, !dbg !17
  %112 = add nsw i64 %111, %8, !dbg !17
  %113 = add nuw nsw i64 %72, %64, !dbg !17
  %114 = and i64 %113, 127, !dbg !17
  %115 = shl i64 %112, 19, !dbg !17
  %116 = and i64 %115, 16252928, !dbg !17
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %117 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %116, !dbg !17
  %118 = getelementptr [2 x i8], ptr addrspace(1) %117, i64 %114, !dbg !17
  %.idx114 = shl i64 %66, 8, !dbg !17
  %119 = getelementptr i8, ptr addrspace(1) %118, i64 %.idx114, !dbg !17
  %address.local398 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local304, !dbg !17
  %address.local399 = ptrtoint ptr addrspace(3) %address.local398 to i64, !dbg !17
  %address.local400 = trunc i64 %address.local399 to i32, !dbg !17
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local400), !dbg !17
  %121 = zext i32 %120 to i64, !dbg !17
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !17
  %address.local401 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local402 = and i32 %address.local401, 32, !dbg !17
  %address.local403 = zext nneg i32 %address.local402 to i64, !dbg !17
  %address.local404 = getelementptr i8, ptr addrspace(1) %119, i64 %address.local403, !dbg !17
  %address.local405 = lshr i64 %address.local, 2, !dbg !17
  %address.local406 = and i64 %address.local405, 56, !dbg !17
  %address.local407 = lshr i64 %address.local, 4, !dbg !17
  %address.local408 = lshr i64 %address.local, 7, !dbg !17
  %address.local409 = sub nsw i64 %address.local407, %address.local408, !dbg !17
  %address.local410 = shl nsw i64 %address.local409, 2, !dbg !17
  %address.local411 = and i64 %address.local410, 4, !dbg !17
  %address.local412 = or disjoint i64 %address.local406, %address.local411, !dbg !17
  %address.local413 = lshr i64 %address.local, 5, !dbg !17
  %address.local414 = sub nsw i64 %address.local405, %address.local413, !dbg !17
  %address.local415 = and i64 %address.local414, 3, !dbg !17
  %address.local416 = or disjoint i64 %address.local412, %address.local415, !dbg !17
  %address.local417 = shl nuw nsw i64 %address.local416, 8, !dbg !17
  %address.local418 = getelementptr i8, ptr addrspace(1) %address.local404, i64 %address.local417, !dbg !17
  %address.local419 = getelementptr [2 x i8], ptr addrspace(1) %address.local418, i64 %address.local312, !dbg !17
  %dma.source.bits1918 = ptrtoint ptr addrspace(1) %address.local419 to i64, !dbg !17
  %dma.source.opaque1919 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1918) #9, !dbg !17
  %dma.source.ptr1920 = inttoptr i64 %dma.source.opaque1919 to ptr addrspace(1), !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1920, ptr addrspace(3) %122, i32 4, i32 0, i32 0), !dbg !17
  %address.local420 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local316, !dbg !17
  %address.local421 = ptrtoint ptr addrspace(3) %address.local420 to i64, !dbg !17
  %address.local422 = trunc i64 %address.local421 to i32, !dbg !17
  %123 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local422), !dbg !17
  %124 = zext i32 %123 to i64, !dbg !17
  %125 = inttoptr i64 %124 to ptr addrspace(3), !dbg !17
  %address.local423 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local424 = and i32 %address.local423, 32, !dbg !17
  %address.local425 = zext nneg i32 %address.local424 to i64, !dbg !17
  %address.local426 = getelementptr i8, ptr addrspace(1) %119, i64 %address.local425, !dbg !17
  %address.local427 = getelementptr [2 x i8], ptr addrspace(1) %address.local426, i64 %address.local324, !dbg !17
  %address.local428 = lshr i64 %address.local314, 2, !dbg !17
  %address.local429 = and i64 %address.local428, 120, !dbg !17
  %address.local430 = lshr i64 %address.local, 4, !dbg !17
  %address.local431 = lshr i64 %address.local, 7, !dbg !17
  %address.local432 = sub nsw i64 %address.local430, %address.local431, !dbg !17
  %address.local433 = shl nsw i64 %address.local432, 2, !dbg !17
  %address.local434 = and i64 %address.local433, 4, !dbg !17
  %address.local435 = or disjoint i64 %address.local429, %address.local434, !dbg !17
  %address.local436 = lshr i64 %address.local, 5, !dbg !17
  %address.local437 = sub nuw nsw i64 %address.local428, %address.local436, !dbg !17
  %address.local438 = and i64 %address.local437, 3, !dbg !17
  %address.local439 = or disjoint i64 %address.local435, %address.local438, !dbg !17
  %address.local440 = shl nuw nsw i64 %address.local439, 8, !dbg !17
  %address.local441 = getelementptr i8, ptr addrspace(1) %address.local427, i64 %address.local440, !dbg !17
  %dma.source.bits1921 = ptrtoint ptr addrspace(1) %address.local441 to i64, !dbg !17
  %dma.source.opaque1922 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1921) #9, !dbg !17
  %dma.source.ptr1923 = inttoptr i64 %dma.source.opaque1922 to ptr addrspace(1), !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1923, ptr addrspace(3) %125, i32 4, i32 0, i32 0), !dbg !17
  %address.local442 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local328, !dbg !17
  %address.local443 = ptrtoint ptr addrspace(3) %address.local442 to i64, !dbg !17
  %address.local444 = trunc i64 %address.local443 to i32, !dbg !17
  %126 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local444), !dbg !17
  %127 = zext i32 %126 to i64, !dbg !17
  %128 = inttoptr i64 %127 to ptr addrspace(3), !dbg !17
  %address.local445 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local446 = and i32 %address.local445, 32, !dbg !17
  %address.local447 = zext nneg i32 %address.local446 to i64, !dbg !17
  %address.local448 = getelementptr i8, ptr addrspace(1) %119, i64 %address.local447, !dbg !17
  %address.local449 = getelementptr i8, ptr addrspace(1) %address.local448, i64 8, !dbg !17
  %address.local450 = getelementptr [2 x i8], ptr addrspace(1) %address.local449, i64 %address.local336, !dbg !17
  %address.local451 = lshr i64 %address.local, 2, !dbg !17
  %address.local452 = and i64 %address.local451, 56, !dbg !17
  %address.local453 = lshr i64 %address.local, 4, !dbg !17
  %address.local454 = lshr i64 %address.local, 7, !dbg !17
  %address.local455 = sub nsw i64 %address.local453, %address.local454, !dbg !17
  %address.local456 = shl nsw i64 %address.local455, 2, !dbg !17
  %address.local457 = and i64 %address.local456, 4, !dbg !17
  %address.local458 = or disjoint i64 %address.local452, %address.local457, !dbg !17
  %address.local459 = lshr i64 %address.local, 5, !dbg !17
  %address.local460 = sub nsw i64 %address.local451, %address.local459, !dbg !17
  %address.local461 = and i64 %address.local460, 3, !dbg !17
  %address.local462 = or disjoint i64 %address.local458, %address.local461, !dbg !17
  %address.local463 = shl nuw nsw i64 %address.local462, 8, !dbg !17
  %address.local464 = getelementptr i8, ptr addrspace(1) %address.local450, i64 %address.local463, !dbg !17
  %dma.source.bits1924 = ptrtoint ptr addrspace(1) %address.local464 to i64, !dbg !17
  %dma.source.opaque1925 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1924) #9, !dbg !17
  %dma.source.ptr1926 = inttoptr i64 %dma.source.opaque1925 to ptr addrspace(1), !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1926, ptr addrspace(3) %128, i32 4, i32 0, i32 0), !dbg !17
  %address.local465 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !17
  %address.local466 = ptrtoint ptr addrspace(3) %address.local465 to i64, !dbg !17
  %address.local467 = trunc i64 %address.local466 to i32, !dbg !17
  %129 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local467), !dbg !17
  %130 = zext i32 %129 to i64, !dbg !17
  %131 = inttoptr i64 %130 to ptr addrspace(3), !dbg !17
  %address.local468 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local469 = and i32 %address.local468, 32, !dbg !17
  %address.local470 = zext nneg i32 %address.local469 to i64, !dbg !17
  %address.local471 = getelementptr i8, ptr addrspace(1) %119, i64 %address.local470, !dbg !17
  %address.local472 = getelementptr i8, ptr addrspace(1) %address.local471, i64 8, !dbg !17
  %address.local473 = getelementptr [2 x i8], ptr addrspace(1) %address.local472, i64 %address.local348, !dbg !17
  %address.local474 = lshr i64 %address.local338, 2, !dbg !17
  %address.local475 = and i64 %address.local474, 120, !dbg !17
  %address.local476 = lshr i64 %address.local, 4, !dbg !17
  %address.local477 = lshr i64 %address.local, 7, !dbg !17
  %address.local478 = sub nsw i64 %address.local476, %address.local477, !dbg !17
  %address.local479 = shl nsw i64 %address.local478, 2, !dbg !17
  %address.local480 = and i64 %address.local479, 4, !dbg !17
  %address.local481 = or disjoint i64 %address.local475, %address.local480, !dbg !17
  %address.local482 = lshr i64 %address.local, 5, !dbg !17
  %address.local483 = sub nuw nsw i64 %address.local474, %address.local482, !dbg !17
  %address.local484 = and i64 %address.local483, 3, !dbg !17
  %address.local485 = or disjoint i64 %address.local481, %address.local484, !dbg !17
  %address.local486 = shl nuw nsw i64 %address.local485, 8, !dbg !17
  %address.local487 = getelementptr i8, ptr addrspace(1) %address.local473, i64 %address.local486, !dbg !17
  %dma.source.bits1927 = ptrtoint ptr addrspace(1) %address.local487 to i64, !dbg !17
  %dma.source.opaque1928 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1927) #9, !dbg !17
  %dma.source.ptr1929 = inttoptr i64 %dma.source.opaque1928 to ptr addrspace(1), !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1929, ptr addrspace(3) %131, i32 4, i32 0, i32 0), !dbg !17
  %address.local488 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !17
  %address.local489 = ptrtoint ptr addrspace(3) %address.local488 to i64, !dbg !17
  %address.local490 = trunc i64 %address.local489 to i32, !dbg !17
  %132 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local490), !dbg !17
  %133 = zext i32 %132 to i64, !dbg !17
  %134 = inttoptr i64 %133 to ptr addrspace(3), !dbg !17
  %address.local491 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local492 = and i32 %address.local491, 32, !dbg !17
  %address.local493 = zext nneg i32 %address.local492 to i64, !dbg !17
  %address.local494 = getelementptr i8, ptr addrspace(1) %119, i64 %address.local493, !dbg !17
  %address.local495 = getelementptr i8, ptr addrspace(1) %address.local494, i64 16, !dbg !17
  %address.local496 = getelementptr [2 x i8], ptr addrspace(1) %address.local495, i64 %address.local360, !dbg !17
  %address.local497 = lshr i64 %address.local, 2, !dbg !17
  %address.local498 = and i64 %address.local497, 56, !dbg !17
  %address.local499 = lshr i64 %address.local, 4, !dbg !17
  %address.local500 = lshr i64 %address.local, 7, !dbg !17
  %address.local501 = xor i64 %address.local500, -1, !dbg !17
  %address.local502 = add nsw i64 %address.local499, %address.local501, !dbg !17
  %address.local503 = shl nsw i64 %address.local502, 2, !dbg !17
  %address.local504 = and i64 %address.local503, 4, !dbg !17
  %address.local505 = or disjoint i64 %address.local498, %address.local504, !dbg !17
  %address.local506 = lshr i64 %address.local, 5, !dbg !17
  %address.local507 = sub nsw i64 %address.local497, %address.local506, !dbg !17
  %address.local508 = and i64 %address.local507, 3, !dbg !17
  %address.local509 = or disjoint i64 %address.local505, %address.local508, !dbg !17
  %address.local510 = shl nuw nsw i64 %address.local509, 8, !dbg !17
  %address.local511 = getelementptr i8, ptr addrspace(1) %address.local496, i64 %address.local510, !dbg !17
  %dma.source.bits1930 = ptrtoint ptr addrspace(1) %address.local511 to i64, !dbg !17
  %dma.source.opaque1931 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1930) #9, !dbg !17
  %dma.source.ptr1932 = inttoptr i64 %dma.source.opaque1931 to ptr addrspace(1), !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1932, ptr addrspace(3) %134, i32 4, i32 0, i32 0), !dbg !17
  %address.local512 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local364, !dbg !17
  %address.local513 = ptrtoint ptr addrspace(3) %address.local512 to i64, !dbg !17
  %address.local514 = trunc i64 %address.local513 to i32, !dbg !17
  %135 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local514), !dbg !17
  %136 = zext i32 %135 to i64, !dbg !17
  %137 = inttoptr i64 %136 to ptr addrspace(3), !dbg !17
  %address.local515 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local516 = and i32 %address.local515, 32, !dbg !17
  %address.local517 = zext nneg i32 %address.local516 to i64, !dbg !17
  %address.local518 = getelementptr i8, ptr addrspace(1) %119, i64 %address.local517, !dbg !17
  %address.local519 = getelementptr i8, ptr addrspace(1) %address.local518, i64 16, !dbg !17
  %address.local520 = lshr i64 %address.local362, 2, !dbg !17
  %address.local521 = and i64 %address.local520, 120, !dbg !17
  %address.local522 = lshr i64 %address.local, 4, !dbg !17
  %address.local523 = lshr i64 %address.local, 7, !dbg !17
  %address.local524 = xor i64 %address.local523, -1, !dbg !17
  %address.local525 = add nsw i64 %address.local522, %address.local524, !dbg !17
  %address.local526 = shl nsw i64 %address.local525, 2, !dbg !17
  %address.local527 = and i64 %address.local526, 4, !dbg !17
  %address.local528 = or disjoint i64 %address.local521, %address.local527, !dbg !17
  %address.local529 = lshr i64 %address.local, 5, !dbg !17
  %address.local530 = sub nuw nsw i64 %address.local520, %address.local529, !dbg !17
  %address.local531 = and i64 %address.local530, 3, !dbg !17
  %address.local532 = or disjoint i64 %address.local528, %address.local531, !dbg !17
  %address.local533 = shl nuw nsw i64 %address.local532, 8, !dbg !17
  %address.local534 = getelementptr i8, ptr addrspace(1) %address.local519, i64 %address.local533, !dbg !17
  %address.local535 = getelementptr [2 x i8], ptr addrspace(1) %address.local534, i64 %address.local372, !dbg !17
  %dma.source.bits1933 = ptrtoint ptr addrspace(1) %address.local535 to i64, !dbg !17
  %dma.source.opaque1934 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1933) #9, !dbg !17
  %dma.source.ptr1935 = inttoptr i64 %dma.source.opaque1934 to ptr addrspace(1), !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1935, ptr addrspace(3) %137, i32 4, i32 0, i32 0), !dbg !17
  %address.local536 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local376, !dbg !17
  %address.local537 = ptrtoint ptr addrspace(3) %address.local536 to i64, !dbg !17
  %address.local538 = trunc i64 %address.local537 to i32, !dbg !17
  %138 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local538), !dbg !17
  %139 = zext i32 %138 to i64, !dbg !17
  %140 = inttoptr i64 %139 to ptr addrspace(3), !dbg !17
  %address.local539 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local540 = and i32 %address.local539, 32, !dbg !17
  %address.local541 = zext nneg i32 %address.local540 to i64, !dbg !17
  %address.local542 = getelementptr i8, ptr addrspace(1) %119, i64 %address.local541, !dbg !17
  %address.local543 = getelementptr i8, ptr addrspace(1) %address.local542, i64 24, !dbg !17
  %address.local544 = lshr i64 %address.local, 2, !dbg !17
  %address.local545 = and i64 %address.local544, 56, !dbg !17
  %address.local546 = lshr i64 %address.local, 4, !dbg !17
  %address.local547 = lshr i64 %address.local, 7, !dbg !17
  %address.local548 = xor i64 %address.local547, -1, !dbg !17
  %address.local549 = add nsw i64 %address.local546, %address.local548, !dbg !17
  %address.local550 = shl nsw i64 %address.local549, 2, !dbg !17
  %address.local551 = and i64 %address.local550, 4, !dbg !17
  %address.local552 = or disjoint i64 %address.local545, %address.local551, !dbg !17
  %address.local553 = lshr i64 %address.local, 5, !dbg !17
  %address.local554 = sub nsw i64 %address.local544, %address.local553, !dbg !17
  %address.local555 = and i64 %address.local554, 3, !dbg !17
  %address.local556 = or disjoint i64 %address.local552, %address.local555, !dbg !17
  %address.local557 = shl nuw nsw i64 %address.local556, 8, !dbg !17
  %address.local558 = getelementptr i8, ptr addrspace(1) %address.local543, i64 %address.local557, !dbg !17
  %address.local559 = getelementptr [2 x i8], ptr addrspace(1) %address.local558, i64 %address.local384, !dbg !17
  %dma.source.bits1936 = ptrtoint ptr addrspace(1) %address.local559 to i64, !dbg !17
  %dma.source.opaque1937 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1936) #9, !dbg !17
  %dma.source.ptr1938 = inttoptr i64 %dma.source.opaque1937 to ptr addrspace(1), !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1938, ptr addrspace(3) %140, i32 4, i32 0, i32 0), !dbg !17
  %address.local560 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local388, !dbg !17
  %address.local561 = ptrtoint ptr addrspace(3) %address.local560 to i64, !dbg !17
  %address.local562 = trunc i64 %address.local561 to i32, !dbg !17
  %141 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local562), !dbg !17
  %142 = zext i32 %141 to i64, !dbg !17
  %143 = inttoptr i64 %142 to ptr addrspace(3), !dbg !17
  %address.local563 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local564 = and i32 %address.local563, 32, !dbg !17
  %address.local565 = zext nneg i32 %address.local564 to i64, !dbg !17
  %address.local566 = getelementptr i8, ptr addrspace(1) %119, i64 %address.local565, !dbg !17
  %address.local567 = getelementptr i8, ptr addrspace(1) %address.local566, i64 24, !dbg !17
  %address.local568 = lshr i64 %address.local386, 2, !dbg !17
  %address.local569 = and i64 %address.local568, 120, !dbg !17
  %address.local570 = lshr i64 %address.local, 4, !dbg !17
  %address.local571 = lshr i64 %address.local, 7, !dbg !17
  %address.local572 = xor i64 %address.local571, -1, !dbg !17
  %address.local573 = add nsw i64 %address.local570, %address.local572, !dbg !17
  %address.local574 = shl nsw i64 %address.local573, 2, !dbg !17
  %address.local575 = and i64 %address.local574, 4, !dbg !17
  %address.local576 = or disjoint i64 %address.local569, %address.local575, !dbg !17
  %address.local577 = lshr i64 %address.local, 5, !dbg !17
  %address.local578 = sub nuw nsw i64 %address.local568, %address.local577, !dbg !17
  %address.local579 = and i64 %address.local578, 3, !dbg !17
  %address.local580 = or disjoint i64 %address.local576, %address.local579, !dbg !17
  %address.local581 = shl nuw nsw i64 %address.local580, 8, !dbg !17
  %address.local582 = getelementptr i8, ptr addrspace(1) %address.local567, i64 %address.local581, !dbg !17
  %address.local583 = getelementptr [2 x i8], ptr addrspace(1) %address.local582, i64 %address.local396, !dbg !17
  %dma.source.bits1939 = ptrtoint ptr addrspace(1) %address.local583 to i64, !dbg !17
  %dma.source.opaque1940 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1939) #9, !dbg !17
  %dma.source.ptr1941 = inttoptr i64 %dma.source.opaque1940 to ptr addrspace(1), !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1941, ptr addrspace(3) %143, i32 4, i32 0, i32 0), !dbg !17
  call void @llvm.amdgcn.asyncmark(), !dbg !17
  %address.local584 = and i32 %address.thread, 192, !dbg !6
  %address.local585 = shl nuw nsw i32 %address.thread, 2, !dbg !6
  %address.local586 = and i32 %address.local585, 60, !dbg !6
  %address.local587 = or disjoint i32 %address.local584, %address.local586, !dbg !6
  %address.local588 = shl nuw nsw i32 %address.thread, 5, !dbg !6
  %address.local589 = and i32 %address.local588, 1536, !dbg !6
  %address.local590 = or disjoint i32 %address.local587, %address.local589, !dbg !6
  %address.local591 = zext nneg i32 %address.local590 to i64, !dbg !6
  %address.local592 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local591, !dbg !6
  %144 = load <4 x half>, ptr addrspace(3) %address.local592, align 2, !dbg !6
  %address.local593 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local591, !dbg !6
  %address.local594 = getelementptr i8, ptr addrspace(3) %address.local593, i64 4096, !dbg !6
  %145 = load <4 x half>, ptr addrspace(3) %address.local594, align 2, !dbg !6
  %address.local595 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local591, !dbg !6
  %address.local596 = getelementptr i8, ptr addrspace(3) %address.local595, i64 512, !dbg !6
  %146 = load <4 x half>, ptr addrspace(3) %address.local596, align 2, !dbg !6
  %address.local597 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local591, !dbg !6
  %address.local598 = getelementptr i8, ptr addrspace(3) %address.local597, i64 4608, !dbg !6
  %147 = load <4 x half>, ptr addrspace(3) %address.local598, align 2, !dbg !6
  %148 = fmul <4 x half> %144, splat (half 0xH3015), !dbg !6
  %149 = fmul <4 x half> %145, splat (half 0xH3015), !dbg !6
  %150 = fmul <4 x half> %146, splat (half 0xH3015), !dbg !6
  %151 = fmul <4 x half> %147, splat (half 0xH3015), !dbg !6
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %.neg231) #8, !dbg !8, !frisk.register_lifetimes !1
  %address.local599 = shl i64 %address.base, 6, !dbg !8
  %address.base600 = call i64 asm sideeffect "", "=v,0"(i64 %.neg233) #8, !dbg !8, !frisk.register_lifetimes !1
  %address.local601 = shl i64 %address.base600, 5, !dbg !8
  %address.local602 = add i64 %address.local599, %address.local601, !dbg !8
  %address.local603 = mul nuw nsw i64 %address.local, 400, !dbg !8
  %address.local604 = lshr i64 %address.local, 2, !dbg !8
  %address.local605 = mul nsw i64 %address.local604, -1664, !dbg !8
  %address.local606 = add nsw i64 %address.local603, %address.local605, !dbg !8
  %address.local607 = and i64 %address.local, 15, !dbg !8
  %address.local608 = lshr i64 %address.local607, 2, !dbg !8
  %address.local609 = shl nuw nsw i64 %address.local608, 3, !dbg !8
  %address.local610 = add nsw i64 %address.local606, %address.local609, !dbg !8
  %address.local611 = lshr i64 %address.local, 1, !dbg !8
  %address.local612 = shl nuw nsw i64 %address.local611, 5, !dbg !8
  %address.local613 = add nsw i64 %address.local610, %address.local612, !dbg !8
  %address.local614 = lshr i64 %address.local, 4, !dbg !6
  %address.local615 = and i64 %address.local614, 3, !dbg !6
  %address.local616 = shl nuw nsw i64 %address.local615, 10, !dbg !8
  %address.local617 = add nsw i64 %address.local613, %address.local616, !dbg !8
  %address.local618 = and i32 %address.thread, 32, !dbg !8
  %address.local619 = zext nneg i32 %address.local618 to i64, !dbg !8
  %address.local620 = add i64 %address.local617, %address.local619, !dbg !8
  %address.local621 = add i64 %address.local602, %address.local620, !dbg !8
  %address.local622 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local621, !dbg !8
  %152 = load <8 x half>, ptr addrspace(3) %address.local622, align 2, !dbg !8
  %153 = extractvalue [2 x <32 x float>] %69, 0, !dbg !30
  %154 = shufflevector <32 x float> %153, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %155 = shufflevector <8 x half> %152, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %156 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %155, <4 x float> %154) #9, !dbg !8
  %address.base623 = call i64 asm sideeffect "", "=v,0"(i64 %.neg236) #8, !dbg !8, !frisk.register_lifetimes !1
  %address.local624 = shl i64 %address.base623, 7, !dbg !8
  %address.local625 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local624, !dbg !8
  %address.base626 = call i64 asm sideeffect "", "=v,0"(i64 %.neg238) #8, !dbg !8, !frisk.register_lifetimes !1
  %address.local627 = shl i64 %address.base626, 6, !dbg !8
  %address.local628 = getelementptr i8, ptr addrspace(3) %address.local625, i64 %address.local627, !dbg !8
  %address.local629 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local630 = and i32 %address.local629, 6, !dbg !8
  %address.local631 = zext nneg i32 %address.local630 to i64, !dbg !8
  %address.local632 = mul nuw nsw i64 %address.local631, 144, !dbg !8
  %address.local633 = getelementptr i8, ptr addrspace(3) %address.local628, i64 %address.local632, !dbg !8
  %address.local634 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local635 = and i32 %address.local634, 24, !dbg !8
  %address.local636 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local637 = and i32 %address.local636, 768, !dbg !8
  %address.local638 = or disjoint i32 %address.local635, %address.local637, !dbg !8
  %address.local639 = zext nneg i32 %address.local638 to i64, !dbg !8
  %address.local640 = getelementptr [2 x i8], ptr addrspace(3) %address.local633, i64 %address.local639, !dbg !8
  %address.local641 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local642 = and i32 %address.local641, 32, !dbg !8
  %address.local643 = zext nneg i32 %address.local642 to i64, !dbg !8
  %address.local644 = getelementptr [2 x i8], ptr addrspace(3) %address.local640, i64 %address.local643, !dbg !8
  %address.local645 = getelementptr [2 x i8], ptr addrspace(3) %address.local644, i64 %address.local616, !dbg !8
  %address.local646 = getelementptr [2 x i8], ptr addrspace(3) %address.local645, i64 %address.local619, !dbg !8
  %address.local647 = getelementptr i8, ptr addrspace(3) %address.local646, i64 64, !dbg !8
  %157 = load <8 x half>, ptr addrspace(3) %address.local647, align 2, !dbg !8
  %158 = shufflevector <8 x half> %152, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %159 = shufflevector <32 x float> %153, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %160 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %158, <4 x float> %156) #9, !dbg !8
  %161 = shufflevector <8 x half> %157, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %162 = shufflevector <8 x half> %157, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %163 = shufflevector <32 x float> %153, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %164 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %161, <4 x float> %159) #9, !dbg !8
  %165 = shufflevector <32 x float> %153, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %166 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %162, <4 x float> %164) #9, !dbg !8
  %167 = shufflevector <32 x float> %153, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !30
  %168 = shufflevector <32 x float> %153, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !30
  %169 = shufflevector <32 x float> %153, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !30
  %170 = shufflevector <32 x float> %153, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !30
  %171 = extractvalue [2 x <32 x float>] %69, 1, !dbg !30
  %172 = shufflevector <32 x float> %171, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %173 = shufflevector <32 x float> %171, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %174 = shufflevector <32 x float> %171, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %175 = shufflevector <32 x float> %171, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %176 = shufflevector <32 x float> %171, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !30
  %177 = shufflevector <32 x float> %171, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !30
  %178 = shufflevector <32 x float> %171, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !30
  %179 = shufflevector <32 x float> %171, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !30
  %address.local648 = shl nuw nsw i64 %address.local, 2, !dbg !6
  %address.local649 = mul nsw i64 %address.local604, -18, !dbg !8
  %address.local650 = add nsw i64 %address.local648, %address.local649, !dbg !8
  %address.local651 = add nsw i64 %address.local650, %address.local611, !dbg !8
  %address.local652 = lshr i64 %address.local615, 1, !dbg !8
  %address.local653 = add nsw i64 %address.local651, %address.local652, !dbg !8
  %address.local654 = sdiv i64 %address.local653, -2, !dbg !8
  %address.local655 = shl i64 %address.local654, 7, !dbg !8
  %address.local656 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local655, !dbg !8
  %address.local657 = mul nuw nsw i64 %address.local, 18, !dbg !8
  %address.local658 = mul nsw i64 %address.local604, -72, !dbg !8
  %address.local659 = add nsw i64 %address.local657, %address.local658, !dbg !8
  %address.local660 = add nsw i64 %address.local659, %address.local608, !dbg !8
  %address.local661 = add nsw i64 %address.local660, 1, !dbg !8
  %address.local662 = sdiv i64 %address.local661, -4, !dbg !8
  %address.local663 = shl i64 %address.local662, 6, !dbg !8
  %address.local664 = getelementptr i8, ptr addrspace(3) %address.local656, i64 %address.local663, !dbg !8
  %address.local665 = getelementptr [2 x i8], ptr addrspace(3) %address.local664, i64 %address.local603, !dbg !8
  %address.local666 = getelementptr [2 x i8], ptr addrspace(3) %address.local665, i64 %address.local605, !dbg !8
  %address.local667 = getelementptr [2 x i8], ptr addrspace(3) %address.local666, i64 %address.local609, !dbg !8
  %address.local668 = getelementptr [2 x i8], ptr addrspace(3) %address.local667, i64 %address.local612, !dbg !8
  %address.local669 = getelementptr [2 x i8], ptr addrspace(3) %address.local668, i64 %address.local616, !dbg !8
  %address.local670 = getelementptr [2 x i8], ptr addrspace(3) %address.local669, i64 %address.local619, !dbg !8
  %address.local671 = getelementptr i8, ptr addrspace(3) %address.local670, i64 144, !dbg !8
  %180 = load <8 x half>, ptr addrspace(3) %address.local671, align 2, !dbg !8
  %address.local672 = mul nsw i64 %address.local604, -16, !dbg !8
  %address.local673 = add nsw i64 %address.local648, %address.local672, !dbg !8
  %address.local674 = lshr i32 %address.thread, 1, !dbg !8
  %address.local675 = and i32 %address.local674, 1, !dbg !8
  %address.local676 = zext nneg i32 %address.local675 to i64, !dbg !8
  %address.local677 = or disjoint i64 %address.local673, %address.local676, !dbg !8
  %address.local678 = add nuw nsw i64 %address.local677, %address.local652, !dbg !8
  %address.local679 = add nsw i64 %address.local678, 3, !dbg !8
  %address.local680 = sdiv i64 %address.local679, -2, !dbg !8
  %address.local681 = shl i64 %address.local680, 7, !dbg !8
  %address.local682 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local681, !dbg !8
  %address.local683 = shl nuw nsw i64 %address.local, 4, !dbg !8
  %address.local684 = mul nsw i64 %address.local604, -64, !dbg !8
  %address.local685 = add nsw i64 %address.local683, %address.local684, !dbg !8
  %address.local686 = or disjoint i64 %address.local685, %address.local608, !dbg !8
  %address.local687 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local688 = and i32 %address.local687, 6, !dbg !8
  %address.local689 = or disjoint i32 %address.local688, 1, !dbg !8
  %address.local690 = zext nneg i32 %address.local689 to i64, !dbg !8
  %address.local691 = add nuw nsw i64 %address.local686, %address.local690, !dbg !8
  %address.local692 = sdiv i64 %address.local691, -4, !dbg !8
  %address.local693 = shl i64 %address.local692, 6, !dbg !8
  %address.local694 = getelementptr i8, ptr addrspace(3) %address.local682, i64 %address.local693, !dbg !8
  %address.local695 = mul nuw nsw i64 %address.local690, 144, !dbg !8
  %address.local696 = getelementptr i8, ptr addrspace(3) %address.local694, i64 %address.local695, !dbg !8
  %address.local697 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local698 = and i32 %address.local697, 24, !dbg !8
  %address.local699 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local700 = and i32 %address.local699, 768, !dbg !8
  %address.local701 = or disjoint i32 %address.local698, %address.local700, !dbg !8
  %address.local702 = zext nneg i32 %address.local701 to i64, !dbg !8
  %address.local703 = getelementptr [2 x i8], ptr addrspace(3) %address.local696, i64 %address.local702, !dbg !8
  %address.local704 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local705 = and i32 %address.local704, 32, !dbg !8
  %address.local706 = zext nneg i32 %address.local705 to i64, !dbg !8
  %address.local707 = getelementptr [2 x i8], ptr addrspace(3) %address.local703, i64 %address.local706, !dbg !8
  %address.local708 = getelementptr [2 x i8], ptr addrspace(3) %address.local707, i64 %address.local616, !dbg !8
  %address.local709 = getelementptr [2 x i8], ptr addrspace(3) %address.local708, i64 %address.local619, !dbg !8
  %address.local710 = getelementptr i8, ptr addrspace(3) %address.local709, i64 192, !dbg !8
  %181 = load <8 x half>, ptr addrspace(3) %address.local710, align 2, !dbg !8
  %address.local711 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local621, !dbg !8
  %address.local712 = getelementptr i8, ptr addrspace(3) %address.local711, i64 1024, !dbg !8
  %182 = load <8 x half>, ptr addrspace(3) %address.local712, align 2, !dbg !8
  %183 = shufflevector <8 x half> %180, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %184 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %183, <4 x float> %163) #9, !dbg !8
  %185 = shufflevector <8 x half> %180, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %186 = shufflevector <8 x half> %181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %187 = shufflevector <8 x half> %181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %188 = shufflevector <8 x half> %182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %189 = shufflevector <8 x half> %182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %address.local713 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local714 = and i32 %address.local713, 6, !dbg !8
  %address.local715 = or disjoint i32 %address.local714, 8, !dbg !8
  %address.local716 = mul nuw nsw i32 %address.local715, 72, !dbg !8
  %address.local717 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local718 = and i32 %address.local717, 24, !dbg !8
  %address.local719 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local720 = and i32 %address.local719, 768, !dbg !8
  %address.local721 = or disjoint i32 %address.local718, %address.local720, !dbg !8
  %address.local722 = or disjoint i32 %address.local721, 1024, !dbg !8
  %address.local723 = add nuw nsw i32 %address.local716, %address.local722, !dbg !8
  %address.local724 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local725 = and i32 %address.local724, 32, !dbg !8
  %address.local726 = or disjoint i32 %address.local725, 64, !dbg !8
  %address.local727 = add nuw nsw i32 %address.local723, %address.local726, !dbg !8
  %address.local728 = zext nneg i32 %address.local727 to i64, !dbg !8
  %address.local729 = add nuw nsw i64 %address.local616, %address.local728, !dbg !8
  %address.local730 = add nuw nsw i64 %address.local729, %address.local619, !dbg !8
  %address.local731 = shl nuw nsw i32 %address.thread, 2, !dbg !8
  %address.local732 = and i32 %address.local731, 12, !dbg !8
  %address.local733 = lshr i32 %address.thread, 1, !dbg !8
  %address.local734 = and i32 %address.local733, 1, !dbg !8
  %address.local735 = or disjoint i32 %address.local732, %address.local734, !dbg !8
  %address.local736 = or disjoint i32 %address.local735, 18, !dbg !8
  %address.local737 = zext nneg i32 %address.local736 to i64, !dbg !8
  %address.local738 = add nuw nsw i64 %address.local652, %address.local737, !dbg !8
  %address.local739 = shl nuw nsw i64 %address.local738, 5, !dbg !8
  %address.local740 = add nuw nsw i64 %address.local739, 32, !dbg !8
  %address.local741 = and i64 %address.local740, 274877906880, !dbg !8
  %address.local742 = lshr i32 %address.thread, 2, !dbg !8
  %address.local743 = and i32 %address.local742, 2, !dbg !8
  %address.local744 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local745 = and i32 %address.local744, 48, !dbg !8
  %address.local746 = or disjoint i32 %address.local743, %address.local745, !dbg !8
  %address.local747 = or disjoint i32 %address.local746, 64, !dbg !8
  %address.local748 = add nuw nsw i32 %address.local747, %address.local715, !dbg !8
  %address.local749 = shl nuw nsw i32 %address.local748, 3, !dbg !8
  %address.local750 = and i32 %address.local749, 2016, !dbg !8
  %address.local751 = zext nneg i32 %address.local750 to i64, !dbg !8
  %address.local752 = add nuw nsw i64 %address.local741, %address.local751, !dbg !8
  %address.local753 = sub nsw i64 %address.local730, %address.local752, !dbg !8
  %address.local754 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local753, !dbg !8
  %address.local755 = getelementptr i8, ptr addrspace(3) %address.local754, i64 64, !dbg !8
  %190 = load <8 x half>, ptr addrspace(3) %address.local755, align 2, !dbg !8
  %191 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %185, <4 x float> %184) #9, !dbg !8
  %address.local756 = add nuw nsw i64 %address.local648, 18, !dbg !8
  %address.local757 = add nsw i64 %address.local756, %address.local649, !dbg !8
  %address.local758 = add nsw i64 %address.local757, %address.local611, !dbg !8
  %address.local759 = add nsw i64 %address.local758, %address.local652, !dbg !8
  %address.local760 = sdiv i64 %address.local759, -2, !dbg !8
  %address.local761 = shl i64 %address.local760, 7, !dbg !8
  %address.local762 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local761, !dbg !8
  %address.local763 = add nuw nsw i64 %address.local657, 72, !dbg !8
  %address.local764 = add nsw i64 %address.local763, %address.local658, !dbg !8
  %address.local765 = add nsw i64 %address.local764, %address.local608, !dbg !8
  %address.local766 = add nsw i64 %address.local765, -15, !dbg !8
  %address.local767 = sdiv i64 %address.local766, -4, !dbg !8
  %address.local768 = shl i64 %address.local767, 6, !dbg !8
  %address.local769 = getelementptr i8, ptr addrspace(3) %address.local762, i64 %address.local768, !dbg !8
  %address.local770 = getelementptr [2 x i8], ptr addrspace(3) %address.local769, i64 %address.local603, !dbg !8
  %address.local771 = getelementptr [2 x i8], ptr addrspace(3) %address.local770, i64 %address.local605, !dbg !8
  %address.local772 = getelementptr [2 x i8], ptr addrspace(3) %address.local771, i64 %address.local609, !dbg !8
  %address.local773 = getelementptr [2 x i8], ptr addrspace(3) %address.local772, i64 %address.local612, !dbg !8
  %address.local774 = getelementptr [2 x i8], ptr addrspace(3) %address.local773, i64 %address.local616, !dbg !8
  %address.local775 = getelementptr [2 x i8], ptr addrspace(3) %address.local774, i64 %address.local619, !dbg !8
  %address.local776 = getelementptr i8, ptr addrspace(3) %address.local775, i64 3216, !dbg !8
  %192 = load <8 x half>, ptr addrspace(3) %address.local776, align 2, !dbg !8
  %address.local777 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local778 = and i32 %address.local777, 6, !dbg !8
  %address.local779 = or disjoint i32 %address.local778, 9, !dbg !8
  %address.local780 = mul nuw nsw i32 %address.local779, 72, !dbg !8
  %address.local781 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local782 = and i32 %address.local781, 24, !dbg !8
  %address.local783 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local784 = and i32 %address.local783, 768, !dbg !8
  %address.local785 = or disjoint i32 %address.local782, %address.local784, !dbg !8
  %address.local786 = or disjoint i32 %address.local785, 1024, !dbg !8
  %address.local787 = add nuw nsw i32 %address.local780, %address.local786, !dbg !8
  %address.local788 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local789 = and i32 %address.local788, 32, !dbg !8
  %address.local790 = or disjoint i32 %address.local789, 64, !dbg !8
  %address.local791 = add nuw nsw i32 %address.local787, %address.local790, !dbg !8
  %address.local792 = zext nneg i32 %address.local791 to i64, !dbg !8
  %address.local793 = add nuw nsw i64 %address.local616, %address.local792, !dbg !8
  %address.local794 = add nuw nsw i64 %address.local793, %address.local619, !dbg !8
  %address.local795 = shl nuw nsw i32 %address.thread, 2, !dbg !8
  %address.local796 = and i32 %address.local795, 12, !dbg !8
  %address.local797 = lshr i32 %address.thread, 1, !dbg !8
  %address.local798 = and i32 %address.local797, 1, !dbg !8
  %address.local799 = or disjoint i32 %address.local796, %address.local798, !dbg !8
  %address.local800 = or disjoint i32 %address.local799, 18, !dbg !8
  %address.local801 = zext nneg i32 %address.local800 to i64, !dbg !8
  %address.local802 = add nuw nsw i64 %address.local652, %address.local801, !dbg !8
  %address.local803 = shl nuw nsw i64 %address.local802, 5, !dbg !8
  %address.local804 = add nuw nsw i64 %address.local803, 96, !dbg !8
  %address.local805 = and i64 %address.local804, 274877906880, !dbg !8
  %address.local806 = lshr i32 %address.thread, 2, !dbg !8
  %address.local807 = and i32 %address.local806, 3, !dbg !8
  %address.local808 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local809 = and i32 %address.local808, 48, !dbg !8
  %address.local810 = or disjoint i32 %address.local807, %address.local809, !dbg !8
  %address.local811 = or disjoint i32 %address.local810, 64, !dbg !8
  %address.local812 = add nuw nsw i32 %address.local811, %address.local779, !dbg !8
  %address.local813 = shl nuw nsw i32 %address.local812, 3, !dbg !8
  %address.local814 = and i32 %address.local813, 2016, !dbg !8
  %address.local815 = zext nneg i32 %address.local814 to i64, !dbg !8
  %address.local816 = add nuw nsw i64 %address.local805, %address.local815, !dbg !8
  %address.local817 = sub nsw i64 %address.local794, %address.local816, !dbg !8
  %address.local818 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local817, !dbg !8
  %address.local819 = getelementptr i8, ptr addrspace(3) %address.local818, i64 192, !dbg !8
  %193 = load <8 x half>, ptr addrspace(3) %address.local819, align 2, !dbg !8
  %194 = shufflevector <8 x half> %190, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %195 = shufflevector <4 x float> %191, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %196 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %186, <4 x float> %165) #9, !dbg !8
  %197 = shufflevector <8 x half> %190, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %198 = shufflevector <8 x half> %192, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %199 = shufflevector <8 x half> %192, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %200 = shufflevector <8 x half> %193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %201 = shufflevector <8 x half> %193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %202 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %187, <4 x float> %196) #9, !dbg !8
  %203 = load <8 x half>, ptr addrspace(3) %address.local622, align 2, !dbg !8
  %204 = shufflevector <4 x float> %202, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %205 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %188, <4 x float> %167) #9, !dbg !8
  %206 = load <8 x half>, ptr addrspace(3) %address.local647, align 2, !dbg !8
  %207 = shufflevector <8 x half> %203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %208 = shufflevector <8 x half> %203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %209 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %189, <4 x float> %205) #9, !dbg !8
  %210 = shufflevector <8 x half> %206, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %211 = shufflevector <8 x half> %206, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %212 = load <8 x half>, ptr addrspace(3) %address.local671, align 2, !dbg !8
  %213 = shufflevector <4 x float> %209, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %214 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %194, <4 x float> %168) #9, !dbg !8
  %215 = load <8 x half>, ptr addrspace(3) %address.local710, align 2, !dbg !8
  %216 = shufflevector <8 x half> %212, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %217 = shufflevector <8 x half> %212, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %218 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %197, <4 x float> %214) #9, !dbg !8
  %219 = shufflevector <8 x half> %215, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %220 = shufflevector <8 x half> %215, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %221 = load <8 x half>, ptr addrspace(3) %address.local712, align 2, !dbg !8
  %222 = shufflevector <4 x float> %218, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %223 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %198, <4 x float> %169) #9, !dbg !8
  %224 = load <8 x half>, ptr addrspace(3) %address.local755, align 2, !dbg !8
  %225 = shufflevector <8 x half> %221, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %226 = shufflevector <8 x half> %221, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %227 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %199, <4 x float> %223) #9, !dbg !8
  %228 = shufflevector <8 x half> %224, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %229 = shufflevector <8 x half> %224, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %230 = load <8 x half>, ptr addrspace(3) %address.local776, align 2, !dbg !8
  %231 = shufflevector <4 x float> %227, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %232 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %200, <4 x float> %170) #9, !dbg !8
  %233 = load <8 x half>, ptr addrspace(3) %address.local819, align 2, !dbg !8
  %234 = shufflevector <8 x half> %230, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %235 = shufflevector <8 x half> %230, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %236 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %201, <4 x float> %232) #9, !dbg !8
  %237 = shufflevector <8 x half> %233, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %238 = shufflevector <8 x half> %233, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %239 = shufflevector <4 x float> %236, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %240 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %150, <4 x half> %207, <4 x float> %172) #9, !dbg !8
  %241 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %208, <4 x float> %240) #9, !dbg !8
  %242 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %150, <4 x half> %210, <4 x float> %173) #9, !dbg !8
  %243 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %211, <4 x float> %242) #9, !dbg !8
  %244 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %150, <4 x half> %216, <4 x float> %174) #9, !dbg !8
  %245 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %217, <4 x float> %244) #9, !dbg !8
  %246 = shufflevector <4 x float> %245, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %247 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %150, <4 x half> %219, <4 x float> %175) #9, !dbg !8
  %248 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %220, <4 x float> %247) #9, !dbg !8
  %249 = shufflevector <4 x float> %248, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %250 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %150, <4 x half> %225, <4 x float> %176) #9, !dbg !8
  %251 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %226, <4 x float> %250) #9, !dbg !8
  %252 = shufflevector <4 x float> %251, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %253 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %150, <4 x half> %228, <4 x float> %177) #9, !dbg !8
  %254 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %229, <4 x float> %253) #9, !dbg !8
  %255 = shufflevector <4 x float> %254, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %256 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %150, <4 x half> %234, <4 x float> %178) #9, !dbg !8
  %257 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %235, <4 x float> %256) #9, !dbg !8
  %258 = shufflevector <4 x float> %257, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %259 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %150, <4 x half> %237, <4 x float> %179) #9, !dbg !8
  %260 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %238, <4 x float> %259) #9, !dbg !8
  %261 = shufflevector <4 x float> %260, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %262 = shufflevector <4 x float> %160, <4 x float> %166, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %263 = shufflevector <32 x float> %262, <32 x float> %195, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %264 = shufflevector <32 x float> %263, <32 x float> %204, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %265 = shufflevector <32 x float> %264, <32 x float> %213, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %266 = shufflevector <32 x float> %265, <32 x float> %222, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %267 = shufflevector <32 x float> %266, <32 x float> %231, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %268 = shufflevector <32 x float> %267, <32 x float> %239, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %269 = insertvalue [2 x <32 x float>] poison, <32 x float> %268, 0, !dbg !22
  %270 = shufflevector <4 x float> %241, <4 x float> %243, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %271 = shufflevector <32 x float> %270, <32 x float> %246, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %272 = shufflevector <32 x float> %271, <32 x float> %249, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %273 = shufflevector <32 x float> %272, <32 x float> %252, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %274 = shufflevector <32 x float> %273, <32 x float> %255, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %275 = shufflevector <32 x float> %274, <32 x float> %258, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %276 = shufflevector <32 x float> %275, <32 x float> %261, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %277 = insertvalue [2 x <32 x float>] %269, <32 x float> %276, 1, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %278 = add nuw nsw i64 %68, 1, !dbg !22
  br label %67, !dbg !22

279:                                              ; preds = %67
  %280 = add nsw i64 %64, %8, !dbg !7
  %281 = shl i64 %280, 19, !dbg !7
  %282 = and i64 %281, 16252928, !dbg !7
  %283 = shl nsw i64 %66, 7, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %284 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %282, !dbg !7
  %285 = getelementptr [2 x i8], ptr addrspace(1) %284, i64 %283, !dbg !7
  %address.thread820 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !7, !range !33, !frisk.register_lifetimes !1
  %address.local821 = zext nneg i32 %address.thread820 to i64, !dbg !6
  %address.local822 = shl nuw nsw i64 %address.local821, 1, !dbg !14
  %address.local823 = and i64 %address.local822, 384, !dbg !14
  %address.local824 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local823, !dbg !7
  %address.local825 = ptrtoint ptr addrspace(3) %address.local824 to i64, !dbg !7
  %address.local826 = trunc i64 %address.local825 to i32, !dbg !7
  %286 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local826), !dbg !7
  %287 = zext i32 %286 to i64, !dbg !7
  %288 = inttoptr i64 %287 to ptr addrspace(3), !dbg !7
  %address.local827 = shl nuw nsw i64 %address.local821, 3, !dbg !7
  %address.local828 = and i64 %address.local827, 1920, !dbg !7
  %address.local829 = shl nuw nsw i64 %address.local821, 1, !dbg !14
  %address.local830 = and i64 %address.local829, 30, !dbg !7
  %address.local831 = or disjoint i64 %address.local828, %address.local830, !dbg !7
  %address.local832 = getelementptr [2 x i8], ptr addrspace(1) %285, i64 %address.local831, !dbg !7
  %dma.source.bits1942 = ptrtoint ptr addrspace(1) %address.local832 to i64, !dbg !7
  %dma.source.opaque1943 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1942) #9, !dbg !7
  %dma.source.ptr1944 = inttoptr i64 %dma.source.opaque1943 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1944, ptr addrspace(3) %288, i32 4, i32 0, i32 0), !dbg !7
  %address.local833 = or disjoint i64 %address.local821, 256, !dbg !14
  %address.local834 = shl nuw nsw i64 %address.local833, 1, !dbg !14
  %address.local835 = and i64 %address.local834, 896, !dbg !14
  %address.local836 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local835, !dbg !7
  %address.local837 = ptrtoint ptr addrspace(3) %address.local836 to i64, !dbg !7
  %address.local838 = trunc i64 %address.local837 to i32, !dbg !7
  %289 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local838), !dbg !7
  %290 = zext i32 %289 to i64, !dbg !7
  %291 = inttoptr i64 %290 to ptr addrspace(3), !dbg !7
  %address.local839 = shl nuw nsw i64 %address.local833, 3, !dbg !7
  %address.local840 = and i64 %address.local839, 3968, !dbg !7
  %address.local841 = shl nuw nsw i64 %address.local833, 1, !dbg !14
  %address.local842 = and i64 %address.local841, 30, !dbg !7
  %address.local843 = or disjoint i64 %address.local840, %address.local842, !dbg !7
  %address.local844 = getelementptr [2 x i8], ptr addrspace(1) %285, i64 %address.local843, !dbg !7
  %dma.source.bits1945 = ptrtoint ptr addrspace(1) %address.local844 to i64, !dbg !7
  %dma.source.opaque1946 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1945) #9, !dbg !7
  %dma.source.ptr1947 = inttoptr i64 %dma.source.opaque1946 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1947, ptr addrspace(3) %291, i32 4, i32 0, i32 0), !dbg !7
  %address.local845 = or disjoint i64 %address.local821, 512, !dbg !14
  %address.local846 = shl nuw nsw i64 %address.local845, 1, !dbg !14
  %address.local847 = and i64 %address.local846, 1408, !dbg !14
  %address.local848 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local847, !dbg !7
  %address.local849 = ptrtoint ptr addrspace(3) %address.local848 to i64, !dbg !7
  %address.local850 = trunc i64 %address.local849 to i32, !dbg !7
  %292 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local850), !dbg !7
  %293 = zext i32 %292 to i64, !dbg !7
  %294 = inttoptr i64 %293 to ptr addrspace(3), !dbg !7
  %address.local851 = shl nuw nsw i64 %address.local845, 3, !dbg !7
  %address.local852 = and i64 %address.local851, 6016, !dbg !7
  %address.local853 = shl nuw nsw i64 %address.local845, 1, !dbg !14
  %address.local854 = and i64 %address.local853, 30, !dbg !7
  %address.local855 = or disjoint i64 %address.local852, %address.local854, !dbg !7
  %address.local856 = getelementptr [2 x i8], ptr addrspace(1) %285, i64 %address.local855, !dbg !7
  %dma.source.bits1948 = ptrtoint ptr addrspace(1) %address.local856 to i64, !dbg !7
  %dma.source.opaque1949 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1948) #9, !dbg !7
  %dma.source.ptr1950 = inttoptr i64 %dma.source.opaque1949 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1950, ptr addrspace(3) %294, i32 4, i32 0, i32 0), !dbg !7
  %address.local857 = or disjoint i64 %address.local821, 768, !dbg !14
  %address.local858 = shl nuw nsw i64 %address.local857, 1, !dbg !14
  %address.local859 = and i64 %address.local858, 1920, !dbg !14
  %address.local860 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local859, !dbg !7
  %address.local861 = ptrtoint ptr addrspace(3) %address.local860 to i64, !dbg !7
  %address.local862 = trunc i64 %address.local861 to i32, !dbg !7
  %295 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local862), !dbg !7
  %296 = zext i32 %295 to i64, !dbg !7
  %297 = inttoptr i64 %296 to ptr addrspace(3), !dbg !7
  %address.local863 = shl nuw nsw i64 %address.local857, 3, !dbg !7
  %address.local864 = and i64 %address.local863, 8064, !dbg !7
  %address.local865 = shl nuw nsw i64 %address.local857, 1, !dbg !14
  %address.local866 = and i64 %address.local865, 30, !dbg !7
  %address.local867 = or disjoint i64 %address.local864, %address.local866, !dbg !7
  %address.local868 = getelementptr [2 x i8], ptr addrspace(1) %285, i64 %address.local867, !dbg !7
  %dma.source.bits1951 = ptrtoint ptr addrspace(1) %address.local868 to i64, !dbg !7
  %dma.source.opaque1952 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1951) #9, !dbg !7
  %dma.source.ptr1953 = inttoptr i64 %dma.source.opaque1952 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1953, ptr addrspace(3) %297, i32 4, i32 0, i32 0), !dbg !7
  %address.local869 = or disjoint i64 %address.local821, 1024, !dbg !14
  %address.local870 = shl nuw nsw i64 %address.local869, 1, !dbg !14
  %address.local871 = and i64 %address.local870, 2432, !dbg !14
  %address.local872 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local871, !dbg !7
  %address.local873 = ptrtoint ptr addrspace(3) %address.local872 to i64, !dbg !7
  %address.local874 = trunc i64 %address.local873 to i32, !dbg !7
  %298 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local874), !dbg !7
  %299 = zext i32 %298 to i64, !dbg !7
  %300 = inttoptr i64 %299 to ptr addrspace(3), !dbg !7
  %address.local875 = shl nuw nsw i64 %address.local869, 3, !dbg !7
  %address.local876 = and i64 %address.local875, 10112, !dbg !7
  %address.local877 = shl nuw nsw i64 %address.local869, 1, !dbg !14
  %address.local878 = and i64 %address.local877, 30, !dbg !7
  %address.local879 = or disjoint i64 %address.local876, %address.local878, !dbg !7
  %address.local880 = getelementptr [2 x i8], ptr addrspace(1) %285, i64 %address.local879, !dbg !7
  %dma.source.bits1954 = ptrtoint ptr addrspace(1) %address.local880 to i64, !dbg !7
  %dma.source.opaque1955 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1954) #9, !dbg !7
  %dma.source.ptr1956 = inttoptr i64 %dma.source.opaque1955 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1956, ptr addrspace(3) %300, i32 4, i32 0, i32 0), !dbg !7
  %address.local881 = or disjoint i64 %address.local821, 1280, !dbg !14
  %address.local882 = shl nuw nsw i64 %address.local881, 1, !dbg !14
  %address.local883 = and i64 %address.local882, 2944, !dbg !14
  %address.local884 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local883, !dbg !7
  %address.local885 = ptrtoint ptr addrspace(3) %address.local884 to i64, !dbg !7
  %address.local886 = trunc i64 %address.local885 to i32, !dbg !7
  %301 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local886), !dbg !7
  %302 = zext i32 %301 to i64, !dbg !7
  %303 = inttoptr i64 %302 to ptr addrspace(3), !dbg !7
  %address.local887 = shl nuw nsw i64 %address.local881, 3, !dbg !7
  %address.local888 = and i64 %address.local887, 12160, !dbg !7
  %address.local889 = shl nuw nsw i64 %address.local881, 1, !dbg !14
  %address.local890 = and i64 %address.local889, 30, !dbg !7
  %address.local891 = or disjoint i64 %address.local888, %address.local890, !dbg !7
  %address.local892 = getelementptr [2 x i8], ptr addrspace(1) %285, i64 %address.local891, !dbg !7
  %dma.source.bits1957 = ptrtoint ptr addrspace(1) %address.local892 to i64, !dbg !7
  %dma.source.opaque1958 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1957) #9, !dbg !7
  %dma.source.ptr1959 = inttoptr i64 %dma.source.opaque1958 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1959, ptr addrspace(3) %303, i32 4, i32 0, i32 0), !dbg !7
  %address.local893 = or disjoint i64 %address.local821, 1536, !dbg !14
  %address.local894 = shl nuw nsw i64 %address.local893, 1, !dbg !14
  %address.local895 = and i64 %address.local894, 3456, !dbg !14
  %address.local896 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local895, !dbg !7
  %address.local897 = ptrtoint ptr addrspace(3) %address.local896 to i64, !dbg !7
  %address.local898 = trunc i64 %address.local897 to i32, !dbg !7
  %304 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local898), !dbg !7
  %305 = zext i32 %304 to i64, !dbg !7
  %306 = inttoptr i64 %305 to ptr addrspace(3), !dbg !7
  %address.local899 = shl nuw nsw i64 %address.local893, 3, !dbg !7
  %address.local900 = and i64 %address.local899, 14208, !dbg !7
  %address.local901 = shl nuw nsw i64 %address.local893, 1, !dbg !14
  %address.local902 = and i64 %address.local901, 30, !dbg !7
  %address.local903 = or disjoint i64 %address.local900, %address.local902, !dbg !7
  %address.local904 = getelementptr [2 x i8], ptr addrspace(1) %285, i64 %address.local903, !dbg !7
  %dma.source.bits1960 = ptrtoint ptr addrspace(1) %address.local904 to i64, !dbg !7
  %dma.source.opaque1961 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1960) #9, !dbg !7
  %dma.source.ptr1962 = inttoptr i64 %dma.source.opaque1961 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1962, ptr addrspace(3) %306, i32 4, i32 0, i32 0), !dbg !7
  %address.local905 = or disjoint i64 %address.local821, 1792, !dbg !14
  %address.local906 = shl nuw nsw i64 %address.local905, 1, !dbg !14
  %address.local907 = and i64 %address.local906, 3968, !dbg !14
  %address.local908 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local907, !dbg !7
  %address.local909 = ptrtoint ptr addrspace(3) %address.local908 to i64, !dbg !7
  %address.local910 = trunc i64 %address.local909 to i32, !dbg !7
  %307 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local910), !dbg !7
  %308 = zext i32 %307 to i64, !dbg !7
  %309 = inttoptr i64 %308 to ptr addrspace(3), !dbg !7
  %address.local911 = shl nuw nsw i64 %address.local905, 3, !dbg !7
  %address.local912 = and i64 %address.local911, 16256, !dbg !7
  %address.local913 = shl nuw nsw i64 %address.local905, 1, !dbg !14
  %address.local914 = and i64 %address.local913, 30, !dbg !7
  %address.local915 = or disjoint i64 %address.local912, %address.local914, !dbg !7
  %address.local916 = getelementptr [2 x i8], ptr addrspace(1) %285, i64 %address.local915, !dbg !7
  %dma.source.bits1963 = ptrtoint ptr addrspace(1) %address.local916 to i64, !dbg !7
  %dma.source.opaque1964 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1963) #9, !dbg !7
  %dma.source.ptr1965 = inttoptr i64 %dma.source.opaque1964 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1965, ptr addrspace(3) %309, i32 4, i32 0, i32 0), !dbg !7
  call void @llvm.amdgcn.asyncmark(), !dbg !7
  %310 = or disjoint i64 %63, 127, !dbg !15
  %311 = icmp uge i64 %9, %310, !dbg !15
  %312 = and i1 %27, %311, !dbg !15
  br i1 %312, label %313, label %508, !dbg !15

313:                                              ; preds = %279
  %314 = extractvalue [2 x <32 x float>] %69, 0, !dbg !15
  %315 = extractelement <32 x float> %314, i64 0, !dbg !15
  %316 = fadd float %315, 0x0000000000000000, !dbg !15
  %317 = extractelement <32 x float> %314, i64 1, !dbg !15
  %318 = fadd float %317, 0x0000000000000000, !dbg !15
  %319 = extractelement <32 x float> %314, i64 2, !dbg !15
  %320 = fadd float %319, 0x0000000000000000, !dbg !15
  %321 = extractelement <32 x float> %314, i64 3, !dbg !15
  %322 = fadd float %321, 0x0000000000000000, !dbg !15
  %323 = extractelement <32 x float> %314, i64 4, !dbg !15
  %324 = fadd float %323, 0x0000000000000000, !dbg !15
  %325 = extractelement <32 x float> %314, i64 5, !dbg !15
  %326 = fadd float %325, 0x0000000000000000, !dbg !15
  %327 = extractelement <32 x float> %314, i64 6, !dbg !15
  %328 = fadd float %327, 0x0000000000000000, !dbg !15
  %329 = extractelement <32 x float> %314, i64 7, !dbg !15
  %330 = fadd float %329, 0x0000000000000000, !dbg !15
  %331 = extractelement <32 x float> %314, i64 8, !dbg !15
  %332 = fadd float %331, 0x0000000000000000, !dbg !15
  %333 = extractelement <32 x float> %314, i64 9, !dbg !15
  %334 = fadd float %333, 0x0000000000000000, !dbg !15
  %335 = extractelement <32 x float> %314, i64 10, !dbg !15
  %336 = fadd float %335, 0x0000000000000000, !dbg !15
  %337 = extractelement <32 x float> %314, i64 11, !dbg !15
  %338 = fadd float %337, 0x0000000000000000, !dbg !15
  %339 = extractelement <32 x float> %314, i64 12, !dbg !15
  %340 = fadd float %339, 0x0000000000000000, !dbg !15
  %341 = extractelement <32 x float> %314, i64 13, !dbg !15
  %342 = fadd float %341, 0x0000000000000000, !dbg !15
  %343 = extractelement <32 x float> %314, i64 14, !dbg !15
  %344 = fadd float %343, 0x0000000000000000, !dbg !15
  %345 = extractelement <32 x float> %314, i64 15, !dbg !15
  %346 = fadd float %345, 0x0000000000000000, !dbg !15
  %347 = extractelement <32 x float> %314, i64 16, !dbg !15
  %348 = fadd float %347, 0x0000000000000000, !dbg !15
  %349 = extractelement <32 x float> %314, i64 17, !dbg !15
  %350 = fadd float %349, 0x0000000000000000, !dbg !15
  %351 = extractelement <32 x float> %314, i64 18, !dbg !15
  %352 = fadd float %351, 0x0000000000000000, !dbg !15
  %353 = extractelement <32 x float> %314, i64 19, !dbg !15
  %354 = fadd float %353, 0x0000000000000000, !dbg !15
  %355 = extractelement <32 x float> %314, i64 20, !dbg !15
  %356 = fadd float %355, 0x0000000000000000, !dbg !15
  %357 = extractelement <32 x float> %314, i64 21, !dbg !15
  %358 = fadd float %357, 0x0000000000000000, !dbg !15
  %359 = extractelement <32 x float> %314, i64 22, !dbg !15
  %360 = fadd float %359, 0x0000000000000000, !dbg !15
  %361 = extractelement <32 x float> %314, i64 23, !dbg !15
  %362 = fadd float %361, 0x0000000000000000, !dbg !15
  %363 = extractelement <32 x float> %314, i64 24, !dbg !15
  %364 = fadd float %363, 0x0000000000000000, !dbg !15
  %365 = extractelement <32 x float> %314, i64 25, !dbg !15
  %366 = fadd float %365, 0x0000000000000000, !dbg !15
  %367 = extractelement <32 x float> %314, i64 26, !dbg !15
  %368 = fadd float %367, 0x0000000000000000, !dbg !15
  %369 = extractelement <32 x float> %314, i64 27, !dbg !15
  %370 = fadd float %369, 0x0000000000000000, !dbg !15
  %371 = extractelement <32 x float> %314, i64 28, !dbg !15
  %372 = fadd float %371, 0x0000000000000000, !dbg !15
  %373 = extractelement <32 x float> %314, i64 29, !dbg !15
  %374 = fadd float %373, 0x0000000000000000, !dbg !15
  %375 = extractelement <32 x float> %314, i64 30, !dbg !15
  %376 = fadd float %375, 0x0000000000000000, !dbg !15
  %377 = extractelement <32 x float> %314, i64 31, !dbg !15
  %378 = fadd float %377, 0x0000000000000000, !dbg !15
  %379 = extractvalue [2 x <32 x float>] %69, 1, !dbg !15
  %380 = extractelement <32 x float> %379, i64 0, !dbg !15
  %381 = fadd float %380, 0x0000000000000000, !dbg !15
  %382 = extractelement <32 x float> %379, i64 1, !dbg !15
  %383 = fadd float %382, 0x0000000000000000, !dbg !15
  %384 = extractelement <32 x float> %379, i64 2, !dbg !15
  %385 = fadd float %384, 0x0000000000000000, !dbg !15
  %386 = extractelement <32 x float> %379, i64 3, !dbg !15
  %387 = fadd float %386, 0x0000000000000000, !dbg !15
  %388 = extractelement <32 x float> %379, i64 4, !dbg !15
  %389 = fadd float %388, 0x0000000000000000, !dbg !15
  %390 = extractelement <32 x float> %379, i64 5, !dbg !15
  %391 = fadd float %390, 0x0000000000000000, !dbg !15
  %392 = extractelement <32 x float> %379, i64 6, !dbg !15
  %393 = fadd float %392, 0x0000000000000000, !dbg !15
  %394 = extractelement <32 x float> %379, i64 7, !dbg !15
  %395 = fadd float %394, 0x0000000000000000, !dbg !15
  %396 = extractelement <32 x float> %379, i64 8, !dbg !15
  %397 = fadd float %396, 0x0000000000000000, !dbg !15
  %398 = extractelement <32 x float> %379, i64 9, !dbg !15
  %399 = fadd float %398, 0x0000000000000000, !dbg !15
  %400 = extractelement <32 x float> %379, i64 10, !dbg !15
  %401 = fadd float %400, 0x0000000000000000, !dbg !15
  %402 = extractelement <32 x float> %379, i64 11, !dbg !15
  %403 = fadd float %402, 0x0000000000000000, !dbg !15
  %404 = extractelement <32 x float> %379, i64 12, !dbg !15
  %405 = fadd float %404, 0x0000000000000000, !dbg !15
  %406 = extractelement <32 x float> %379, i64 13, !dbg !15
  %407 = fadd float %406, 0x0000000000000000, !dbg !15
  %408 = extractelement <32 x float> %379, i64 14, !dbg !15
  %409 = fadd float %408, 0x0000000000000000, !dbg !15
  %410 = extractelement <32 x float> %379, i64 15, !dbg !15
  %411 = fadd float %410, 0x0000000000000000, !dbg !15
  %412 = extractelement <32 x float> %379, i64 16, !dbg !15
  %413 = fadd float %412, 0x0000000000000000, !dbg !15
  %414 = extractelement <32 x float> %379, i64 17, !dbg !15
  %415 = fadd float %414, 0x0000000000000000, !dbg !15
  %416 = extractelement <32 x float> %379, i64 18, !dbg !15
  %417 = fadd float %416, 0x0000000000000000, !dbg !15
  %418 = extractelement <32 x float> %379, i64 19, !dbg !15
  %419 = fadd float %418, 0x0000000000000000, !dbg !15
  %420 = extractelement <32 x float> %379, i64 20, !dbg !15
  %421 = fadd float %420, 0x0000000000000000, !dbg !15
  %422 = extractelement <32 x float> %379, i64 21, !dbg !15
  %423 = fadd float %422, 0x0000000000000000, !dbg !15
  %424 = extractelement <32 x float> %379, i64 22, !dbg !15
  %425 = fadd float %424, 0x0000000000000000, !dbg !15
  %426 = extractelement <32 x float> %379, i64 23, !dbg !15
  %427 = fadd float %426, 0x0000000000000000, !dbg !15
  %428 = extractelement <32 x float> %379, i64 24, !dbg !15
  %429 = fadd float %428, 0x0000000000000000, !dbg !15
  %430 = extractelement <32 x float> %379, i64 25, !dbg !15
  %431 = fadd float %430, 0x0000000000000000, !dbg !15
  %432 = extractelement <32 x float> %379, i64 26, !dbg !15
  %433 = fadd float %432, 0x0000000000000000, !dbg !15
  %434 = extractelement <32 x float> %379, i64 27, !dbg !15
  %435 = fadd float %434, 0x0000000000000000, !dbg !15
  %436 = extractelement <32 x float> %379, i64 28, !dbg !15
  %437 = fadd float %436, 0x0000000000000000, !dbg !15
  %438 = extractelement <32 x float> %379, i64 29, !dbg !15
  %439 = fadd float %438, 0x0000000000000000, !dbg !15
  %440 = extractelement <32 x float> %379, i64 30, !dbg !15
  %441 = fadd float %440, 0x0000000000000000, !dbg !15
  %442 = extractelement <32 x float> %379, i64 31, !dbg !15
  %443 = fadd float %442, 0x0000000000000000, !dbg !15
  %444 = insertelement <32 x float> poison, float %316, i64 0, !dbg !15
  %445 = insertelement <32 x float> %444, float %318, i64 1, !dbg !15
  %446 = insertelement <32 x float> %445, float %320, i64 2, !dbg !15
  %447 = insertelement <32 x float> %446, float %322, i64 3, !dbg !15
  %448 = insertelement <32 x float> %447, float %324, i64 4, !dbg !15
  %449 = insertelement <32 x float> %448, float %326, i64 5, !dbg !15
  %450 = insertelement <32 x float> %449, float %328, i64 6, !dbg !15
  %451 = insertelement <32 x float> %450, float %330, i64 7, !dbg !15
  %452 = insertelement <32 x float> %451, float %332, i64 8, !dbg !15
  %453 = insertelement <32 x float> %452, float %334, i64 9, !dbg !15
  %454 = insertelement <32 x float> %453, float %336, i64 10, !dbg !15
  %455 = insertelement <32 x float> %454, float %338, i64 11, !dbg !15
  %456 = insertelement <32 x float> %455, float %340, i64 12, !dbg !15
  %457 = insertelement <32 x float> %456, float %342, i64 13, !dbg !15
  %458 = insertelement <32 x float> %457, float %344, i64 14, !dbg !15
  %459 = insertelement <32 x float> %458, float %346, i64 15, !dbg !15
  %460 = insertelement <32 x float> %459, float %348, i64 16, !dbg !15
  %461 = insertelement <32 x float> %460, float %350, i64 17, !dbg !15
  %462 = insertelement <32 x float> %461, float %352, i64 18, !dbg !15
  %463 = insertelement <32 x float> %462, float %354, i64 19, !dbg !15
  %464 = insertelement <32 x float> %463, float %356, i64 20, !dbg !15
  %465 = insertelement <32 x float> %464, float %358, i64 21, !dbg !15
  %466 = insertelement <32 x float> %465, float %360, i64 22, !dbg !15
  %467 = insertelement <32 x float> %466, float %362, i64 23, !dbg !15
  %468 = insertelement <32 x float> %467, float %364, i64 24, !dbg !15
  %469 = insertelement <32 x float> %468, float %366, i64 25, !dbg !15
  %470 = insertelement <32 x float> %469, float %368, i64 26, !dbg !15
  %471 = insertelement <32 x float> %470, float %370, i64 27, !dbg !15
  %472 = insertelement <32 x float> %471, float %372, i64 28, !dbg !15
  %473 = insertelement <32 x float> %472, float %374, i64 29, !dbg !15
  %474 = insertelement <32 x float> %473, float %376, i64 30, !dbg !15
  %475 = insertelement <32 x float> %474, float %378, i64 31, !dbg !15
  %476 = insertelement <32 x float> poison, float %381, i64 0, !dbg !15
  %477 = insertelement <32 x float> %476, float %383, i64 1, !dbg !15
  %478 = insertelement <32 x float> %477, float %385, i64 2, !dbg !15
  %479 = insertelement <32 x float> %478, float %387, i64 3, !dbg !15
  %480 = insertelement <32 x float> %479, float %389, i64 4, !dbg !15
  %481 = insertelement <32 x float> %480, float %391, i64 5, !dbg !15
  %482 = insertelement <32 x float> %481, float %393, i64 6, !dbg !15
  %483 = insertelement <32 x float> %482, float %395, i64 7, !dbg !15
  %484 = insertelement <32 x float> %483, float %397, i64 8, !dbg !15
  %485 = insertelement <32 x float> %484, float %399, i64 9, !dbg !15
  %486 = insertelement <32 x float> %485, float %401, i64 10, !dbg !15
  %487 = insertelement <32 x float> %486, float %403, i64 11, !dbg !15
  %488 = insertelement <32 x float> %487, float %405, i64 12, !dbg !15
  %489 = insertelement <32 x float> %488, float %407, i64 13, !dbg !15
  %490 = insertelement <32 x float> %489, float %409, i64 14, !dbg !15
  %491 = insertelement <32 x float> %490, float %411, i64 15, !dbg !15
  %492 = insertelement <32 x float> %491, float %413, i64 16, !dbg !15
  %493 = insertelement <32 x float> %492, float %415, i64 17, !dbg !15
  %494 = insertelement <32 x float> %493, float %417, i64 18, !dbg !15
  %495 = insertelement <32 x float> %494, float %419, i64 19, !dbg !15
  %496 = insertelement <32 x float> %495, float %421, i64 20, !dbg !15
  %497 = insertelement <32 x float> %496, float %423, i64 21, !dbg !15
  %498 = insertelement <32 x float> %497, float %425, i64 22, !dbg !15
  %499 = insertelement <32 x float> %498, float %427, i64 23, !dbg !15
  %500 = insertelement <32 x float> %499, float %429, i64 24, !dbg !15
  %501 = insertelement <32 x float> %500, float %431, i64 25, !dbg !15
  %502 = insertelement <32 x float> %501, float %433, i64 26, !dbg !15
  %503 = insertelement <32 x float> %502, float %435, i64 27, !dbg !15
  %504 = insertelement <32 x float> %503, float %437, i64 28, !dbg !15
  %505 = insertelement <32 x float> %504, float %439, i64 29, !dbg !15
  %506 = insertelement <32 x float> %505, float %441, i64 30, !dbg !15
  %507 = insertelement <32 x float> %506, float %443, i64 31, !dbg !15
  br label %807, !dbg !15

508:                                              ; preds = %279
  %509 = add nsw i64 %9, %20, !dbg !15
  %510 = shl nuw nsw i64 %22, 4, !dbg !15
  %511 = add nsw i64 %509, %510, !dbg !15
  %512 = and i32 %19, 240, !dbg !15
  %index.compact = zext nneg i32 %512 to i64, !dbg !15
  %513 = sub nsw i64 %511, %index.compact, !dbg !15
  %514 = shl nuw nsw i64 %24, 4, !dbg !15
  %515 = or disjoint i64 %63, %514, !dbg !15
  %516 = add nsw i64 %513, 1, !dbg !23
  %.not36 = icmp ugt i64 %516, %515, !dbg !24
  %517 = select i1 %.not36, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %518 = extractvalue [2 x <32 x float>] %69, 0, !dbg !15
  %519 = extractelement <32 x float> %518, i64 0, !dbg !15
  %520 = fadd float %519, %517, !dbg !15
  %521 = or disjoint i64 %515, 1, !dbg !15
  %.not37 = icmp ugt i64 %516, %521, !dbg !24
  %522 = select i1 %.not37, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %523 = extractelement <32 x float> %518, i64 1, !dbg !15
  %524 = fadd float %523, %522, !dbg !15
  %525 = or disjoint i64 %515, 2, !dbg !15
  %.not38 = icmp ugt i64 %516, %525, !dbg !24
  %526 = select i1 %.not38, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %527 = extractelement <32 x float> %518, i64 2, !dbg !15
  %528 = fadd float %527, %526, !dbg !15
  %529 = or disjoint i64 %515, 3, !dbg !15
  %.not39 = icmp ugt i64 %516, %529, !dbg !24
  %530 = select i1 %.not39, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %531 = extractelement <32 x float> %518, i64 3, !dbg !15
  %532 = fadd float %531, %530, !dbg !15
  %533 = or disjoint i64 %515, 4, !dbg !15
  %.not40 = icmp ugt i64 %516, %533, !dbg !24
  %534 = select i1 %.not40, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %535 = extractelement <32 x float> %518, i64 4, !dbg !15
  %536 = fadd float %535, %534, !dbg !15
  %537 = or disjoint i64 %515, 5, !dbg !15
  %.not41 = icmp ugt i64 %516, %537, !dbg !24
  %538 = select i1 %.not41, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %539 = extractelement <32 x float> %518, i64 5, !dbg !15
  %540 = fadd float %539, %538, !dbg !15
  %541 = or disjoint i64 %515, 6, !dbg !15
  %.not42 = icmp ugt i64 %516, %541, !dbg !24
  %542 = select i1 %.not42, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %543 = extractelement <32 x float> %518, i64 6, !dbg !15
  %544 = fadd float %543, %542, !dbg !15
  %545 = or disjoint i64 %515, 7, !dbg !15
  %.not43 = icmp ugt i64 %516, %545, !dbg !24
  %546 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %547 = extractelement <32 x float> %518, i64 7, !dbg !15
  %548 = fadd float %547, %546, !dbg !15
  %549 = or disjoint i64 %515, 8, !dbg !15
  %.not44 = icmp ugt i64 %516, %549, !dbg !24
  %550 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %551 = extractelement <32 x float> %518, i64 8, !dbg !15
  %552 = fadd float %551, %550, !dbg !15
  %553 = or disjoint i64 %515, 9, !dbg !15
  %.not45 = icmp ugt i64 %516, %553, !dbg !24
  %554 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %555 = extractelement <32 x float> %518, i64 9, !dbg !15
  %556 = fadd float %555, %554, !dbg !15
  %557 = or disjoint i64 %515, 10, !dbg !15
  %.not46 = icmp ugt i64 %516, %557, !dbg !24
  %558 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %559 = extractelement <32 x float> %518, i64 10, !dbg !15
  %560 = fadd float %559, %558, !dbg !15
  %561 = or disjoint i64 %515, 11, !dbg !15
  %.not47 = icmp ugt i64 %516, %561, !dbg !24
  %562 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %563 = extractelement <32 x float> %518, i64 11, !dbg !15
  %564 = fadd float %563, %562, !dbg !15
  %565 = or disjoint i64 %515, 12, !dbg !15
  %.not48 = icmp ugt i64 %516, %565, !dbg !24
  %566 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %567 = extractelement <32 x float> %518, i64 12, !dbg !15
  %568 = fadd float %567, %566, !dbg !15
  %569 = or disjoint i64 %515, 13, !dbg !15
  %.not49 = icmp ugt i64 %516, %569, !dbg !24
  %570 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %571 = extractelement <32 x float> %518, i64 13, !dbg !15
  %572 = fadd float %571, %570, !dbg !15
  %573 = or disjoint i64 %515, 14, !dbg !15
  %.not50 = icmp ugt i64 %516, %573, !dbg !24
  %574 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %575 = extractelement <32 x float> %518, i64 14, !dbg !15
  %576 = fadd float %575, %574, !dbg !15
  %577 = or disjoint i64 %515, 15, !dbg !15
  %.not51 = icmp ugt i64 %516, %577, !dbg !24
  %578 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %579 = extractelement <32 x float> %518, i64 15, !dbg !15
  %580 = fadd float %579, %578, !dbg !15
  %581 = or disjoint i64 %515, 64, !dbg !15
  %.not52 = icmp ugt i64 %516, %581, !dbg !24
  %582 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %583 = extractelement <32 x float> %518, i64 16, !dbg !15
  %584 = fadd float %583, %582, !dbg !15
  %585 = or disjoint i64 %515, 65, !dbg !15
  %.not53 = icmp ugt i64 %516, %585, !dbg !24
  %586 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %587 = extractelement <32 x float> %518, i64 17, !dbg !15
  %588 = fadd float %587, %586, !dbg !15
  %589 = or disjoint i64 %515, 66, !dbg !15
  %.not54 = icmp ugt i64 %516, %589, !dbg !24
  %590 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %591 = extractelement <32 x float> %518, i64 18, !dbg !15
  %592 = fadd float %591, %590, !dbg !15
  %593 = or disjoint i64 %515, 67, !dbg !15
  %.not55 = icmp ugt i64 %516, %593, !dbg !24
  %594 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %595 = extractelement <32 x float> %518, i64 19, !dbg !15
  %596 = fadd float %595, %594, !dbg !15
  %597 = or disjoint i64 %515, 68, !dbg !15
  %.not56 = icmp ugt i64 %516, %597, !dbg !24
  %598 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %599 = extractelement <32 x float> %518, i64 20, !dbg !15
  %600 = fadd float %599, %598, !dbg !15
  %601 = or disjoint i64 %515, 69, !dbg !15
  %.not57 = icmp ugt i64 %516, %601, !dbg !24
  %602 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %603 = extractelement <32 x float> %518, i64 21, !dbg !15
  %604 = fadd float %603, %602, !dbg !15
  %605 = or disjoint i64 %515, 70, !dbg !15
  %.not58 = icmp ugt i64 %516, %605, !dbg !24
  %606 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %607 = extractelement <32 x float> %518, i64 22, !dbg !15
  %608 = fadd float %607, %606, !dbg !15
  %609 = or disjoint i64 %515, 71, !dbg !15
  %.not59 = icmp ugt i64 %516, %609, !dbg !24
  %610 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %611 = extractelement <32 x float> %518, i64 23, !dbg !15
  %612 = fadd float %611, %610, !dbg !15
  %613 = or disjoint i64 %515, 72, !dbg !15
  %.not60 = icmp ugt i64 %516, %613, !dbg !24
  %614 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %615 = extractelement <32 x float> %518, i64 24, !dbg !15
  %616 = fadd float %615, %614, !dbg !15
  %617 = or disjoint i64 %515, 73, !dbg !15
  %.not61 = icmp ugt i64 %516, %617, !dbg !24
  %618 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %619 = extractelement <32 x float> %518, i64 25, !dbg !15
  %620 = fadd float %619, %618, !dbg !15
  %621 = or disjoint i64 %515, 74, !dbg !15
  %.not62 = icmp ugt i64 %516, %621, !dbg !24
  %622 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %623 = extractelement <32 x float> %518, i64 26, !dbg !15
  %624 = fadd float %623, %622, !dbg !15
  %625 = or disjoint i64 %515, 75, !dbg !15
  %.not63 = icmp ugt i64 %516, %625, !dbg !24
  %626 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %627 = extractelement <32 x float> %518, i64 27, !dbg !15
  %628 = fadd float %627, %626, !dbg !15
  %629 = or disjoint i64 %515, 76, !dbg !15
  %.not64 = icmp ugt i64 %516, %629, !dbg !24
  %630 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %631 = extractelement <32 x float> %518, i64 28, !dbg !15
  %632 = fadd float %631, %630, !dbg !15
  %633 = or disjoint i64 %515, 77, !dbg !15
  %.not65 = icmp ugt i64 %516, %633, !dbg !24
  %634 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %635 = extractelement <32 x float> %518, i64 29, !dbg !15
  %636 = fadd float %635, %634, !dbg !15
  %637 = or disjoint i64 %515, 78, !dbg !15
  %.not66 = icmp ugt i64 %516, %637, !dbg !24
  %638 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %639 = extractelement <32 x float> %518, i64 30, !dbg !15
  %640 = fadd float %639, %638, !dbg !15
  %641 = or disjoint i64 %515, 79, !dbg !15
  %.not67 = icmp ugt i64 %516, %641, !dbg !24
  %642 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %643 = extractelement <32 x float> %518, i64 31, !dbg !15
  %644 = fadd float %643, %642, !dbg !15
  %645 = add nsw i64 %513, 65, !dbg !23
  %.not68 = icmp ugt i64 %645, %515, !dbg !24
  %646 = select i1 %.not68, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %647 = extractvalue [2 x <32 x float>] %69, 1, !dbg !15
  %648 = extractelement <32 x float> %647, i64 0, !dbg !15
  %649 = fadd float %648, %646, !dbg !15
  %.not69 = icmp ugt i64 %645, %521, !dbg !24
  %650 = select i1 %.not69, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %651 = extractelement <32 x float> %647, i64 1, !dbg !15
  %652 = fadd float %651, %650, !dbg !15
  %.not70 = icmp ugt i64 %645, %525, !dbg !24
  %653 = select i1 %.not70, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %654 = extractelement <32 x float> %647, i64 2, !dbg !15
  %655 = fadd float %654, %653, !dbg !15
  %.not71 = icmp ugt i64 %645, %529, !dbg !24
  %656 = select i1 %.not71, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %657 = extractelement <32 x float> %647, i64 3, !dbg !15
  %658 = fadd float %657, %656, !dbg !15
  %.not72 = icmp ugt i64 %645, %533, !dbg !24
  %659 = select i1 %.not72, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %660 = extractelement <32 x float> %647, i64 4, !dbg !15
  %661 = fadd float %660, %659, !dbg !15
  %.not73 = icmp ugt i64 %645, %537, !dbg !24
  %662 = select i1 %.not73, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %663 = extractelement <32 x float> %647, i64 5, !dbg !15
  %664 = fadd float %663, %662, !dbg !15
  %.not74 = icmp ugt i64 %645, %541, !dbg !24
  %665 = select i1 %.not74, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %666 = extractelement <32 x float> %647, i64 6, !dbg !15
  %667 = fadd float %666, %665, !dbg !15
  %.not75 = icmp ugt i64 %645, %545, !dbg !24
  %668 = select i1 %.not75, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %669 = extractelement <32 x float> %647, i64 7, !dbg !15
  %670 = fadd float %669, %668, !dbg !15
  %.not76 = icmp ugt i64 %645, %549, !dbg !24
  %671 = select i1 %.not76, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %672 = extractelement <32 x float> %647, i64 8, !dbg !15
  %673 = fadd float %672, %671, !dbg !15
  %.not77 = icmp ugt i64 %645, %553, !dbg !24
  %674 = select i1 %.not77, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %675 = extractelement <32 x float> %647, i64 9, !dbg !15
  %676 = fadd float %675, %674, !dbg !15
  %.not78 = icmp ugt i64 %645, %557, !dbg !24
  %677 = select i1 %.not78, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %678 = extractelement <32 x float> %647, i64 10, !dbg !15
  %679 = fadd float %678, %677, !dbg !15
  %.not79 = icmp ugt i64 %645, %561, !dbg !24
  %680 = select i1 %.not79, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %681 = extractelement <32 x float> %647, i64 11, !dbg !15
  %682 = fadd float %681, %680, !dbg !15
  %.not80 = icmp ugt i64 %645, %565, !dbg !24
  %683 = select i1 %.not80, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %684 = extractelement <32 x float> %647, i64 12, !dbg !15
  %685 = fadd float %684, %683, !dbg !15
  %.not81 = icmp ugt i64 %645, %569, !dbg !24
  %686 = select i1 %.not81, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %687 = extractelement <32 x float> %647, i64 13, !dbg !15
  %688 = fadd float %687, %686, !dbg !15
  %.not82 = icmp ugt i64 %645, %573, !dbg !24
  %689 = select i1 %.not82, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %690 = extractelement <32 x float> %647, i64 14, !dbg !15
  %691 = fadd float %690, %689, !dbg !15
  %.not83 = icmp ugt i64 %645, %577, !dbg !24
  %692 = select i1 %.not83, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %693 = extractelement <32 x float> %647, i64 15, !dbg !15
  %694 = fadd float %693, %692, !dbg !15
  %.not84 = icmp ugt i64 %645, %581, !dbg !24
  %695 = select i1 %.not84, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %696 = extractelement <32 x float> %647, i64 16, !dbg !15
  %697 = fadd float %696, %695, !dbg !15
  %.not85 = icmp ugt i64 %645, %585, !dbg !24
  %698 = select i1 %.not85, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %699 = extractelement <32 x float> %647, i64 17, !dbg !15
  %700 = fadd float %699, %698, !dbg !15
  %.not86 = icmp ugt i64 %645, %589, !dbg !24
  %701 = select i1 %.not86, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %702 = extractelement <32 x float> %647, i64 18, !dbg !15
  %703 = fadd float %702, %701, !dbg !15
  %.not87 = icmp ugt i64 %645, %593, !dbg !24
  %704 = select i1 %.not87, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %705 = extractelement <32 x float> %647, i64 19, !dbg !15
  %706 = fadd float %705, %704, !dbg !15
  %.not88 = icmp ugt i64 %645, %597, !dbg !24
  %707 = select i1 %.not88, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %708 = extractelement <32 x float> %647, i64 20, !dbg !15
  %709 = fadd float %708, %707, !dbg !15
  %.not89 = icmp ugt i64 %645, %601, !dbg !24
  %710 = select i1 %.not89, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %711 = extractelement <32 x float> %647, i64 21, !dbg !15
  %712 = fadd float %711, %710, !dbg !15
  %.not90 = icmp ugt i64 %645, %605, !dbg !24
  %713 = select i1 %.not90, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %714 = extractelement <32 x float> %647, i64 22, !dbg !15
  %715 = fadd float %714, %713, !dbg !15
  %.not91 = icmp ugt i64 %645, %609, !dbg !24
  %716 = select i1 %.not91, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %717 = extractelement <32 x float> %647, i64 23, !dbg !15
  %718 = fadd float %717, %716, !dbg !15
  %.not92 = icmp ugt i64 %645, %613, !dbg !24
  %719 = select i1 %.not92, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %720 = extractelement <32 x float> %647, i64 24, !dbg !15
  %721 = fadd float %720, %719, !dbg !15
  %.not93 = icmp ugt i64 %645, %617, !dbg !24
  %722 = select i1 %.not93, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %723 = extractelement <32 x float> %647, i64 25, !dbg !15
  %724 = fadd float %723, %722, !dbg !15
  %.not94 = icmp ugt i64 %645, %621, !dbg !24
  %725 = select i1 %.not94, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %726 = extractelement <32 x float> %647, i64 26, !dbg !15
  %727 = fadd float %726, %725, !dbg !15
  %.not95 = icmp ugt i64 %645, %625, !dbg !24
  %728 = select i1 %.not95, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %729 = extractelement <32 x float> %647, i64 27, !dbg !15
  %730 = fadd float %729, %728, !dbg !15
  %.not96 = icmp ugt i64 %645, %629, !dbg !24
  %731 = select i1 %.not96, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %732 = extractelement <32 x float> %647, i64 28, !dbg !15
  %733 = fadd float %732, %731, !dbg !15
  %.not97 = icmp ugt i64 %645, %633, !dbg !24
  %734 = select i1 %.not97, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %735 = extractelement <32 x float> %647, i64 29, !dbg !15
  %736 = fadd float %735, %734, !dbg !15
  %.not98 = icmp ugt i64 %645, %637, !dbg !24
  %737 = select i1 %.not98, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %738 = extractelement <32 x float> %647, i64 30, !dbg !15
  %739 = fadd float %738, %737, !dbg !15
  %.not99 = icmp ugt i64 %645, %641, !dbg !24
  %740 = select i1 %.not99, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %741 = extractelement <32 x float> %647, i64 31, !dbg !15
  %742 = fadd float %741, %740, !dbg !15
  %743 = insertelement <32 x float> poison, float %520, i64 0, !dbg !15
  %744 = insertelement <32 x float> %743, float %524, i64 1, !dbg !15
  %745 = insertelement <32 x float> %744, float %528, i64 2, !dbg !15
  %746 = insertelement <32 x float> %745, float %532, i64 3, !dbg !15
  %747 = insertelement <32 x float> %746, float %536, i64 4, !dbg !15
  %748 = insertelement <32 x float> %747, float %540, i64 5, !dbg !15
  %749 = insertelement <32 x float> %748, float %544, i64 6, !dbg !15
  %750 = insertelement <32 x float> %749, float %548, i64 7, !dbg !15
  %751 = insertelement <32 x float> %750, float %552, i64 8, !dbg !15
  %752 = insertelement <32 x float> %751, float %556, i64 9, !dbg !15
  %753 = insertelement <32 x float> %752, float %560, i64 10, !dbg !15
  %754 = insertelement <32 x float> %753, float %564, i64 11, !dbg !15
  %755 = insertelement <32 x float> %754, float %568, i64 12, !dbg !15
  %756 = insertelement <32 x float> %755, float %572, i64 13, !dbg !15
  %757 = insertelement <32 x float> %756, float %576, i64 14, !dbg !15
  %758 = insertelement <32 x float> %757, float %580, i64 15, !dbg !15
  %759 = insertelement <32 x float> %758, float %584, i64 16, !dbg !15
  %760 = insertelement <32 x float> %759, float %588, i64 17, !dbg !15
  %761 = insertelement <32 x float> %760, float %592, i64 18, !dbg !15
  %762 = insertelement <32 x float> %761, float %596, i64 19, !dbg !15
  %763 = insertelement <32 x float> %762, float %600, i64 20, !dbg !15
  %764 = insertelement <32 x float> %763, float %604, i64 21, !dbg !15
  %765 = insertelement <32 x float> %764, float %608, i64 22, !dbg !15
  %766 = insertelement <32 x float> %765, float %612, i64 23, !dbg !15
  %767 = insertelement <32 x float> %766, float %616, i64 24, !dbg !15
  %768 = insertelement <32 x float> %767, float %620, i64 25, !dbg !15
  %769 = insertelement <32 x float> %768, float %624, i64 26, !dbg !15
  %770 = insertelement <32 x float> %769, float %628, i64 27, !dbg !15
  %771 = insertelement <32 x float> %770, float %632, i64 28, !dbg !15
  %772 = insertelement <32 x float> %771, float %636, i64 29, !dbg !15
  %773 = insertelement <32 x float> %772, float %640, i64 30, !dbg !15
  %774 = insertelement <32 x float> %773, float %644, i64 31, !dbg !15
  %775 = insertelement <32 x float> poison, float %649, i64 0, !dbg !15
  %776 = insertelement <32 x float> %775, float %652, i64 1, !dbg !15
  %777 = insertelement <32 x float> %776, float %655, i64 2, !dbg !15
  %778 = insertelement <32 x float> %777, float %658, i64 3, !dbg !15
  %779 = insertelement <32 x float> %778, float %661, i64 4, !dbg !15
  %780 = insertelement <32 x float> %779, float %664, i64 5, !dbg !15
  %781 = insertelement <32 x float> %780, float %667, i64 6, !dbg !15
  %782 = insertelement <32 x float> %781, float %670, i64 7, !dbg !15
  %783 = insertelement <32 x float> %782, float %673, i64 8, !dbg !15
  %784 = insertelement <32 x float> %783, float %676, i64 9, !dbg !15
  %785 = insertelement <32 x float> %784, float %679, i64 10, !dbg !15
  %786 = insertelement <32 x float> %785, float %682, i64 11, !dbg !15
  %787 = insertelement <32 x float> %786, float %685, i64 12, !dbg !15
  %788 = insertelement <32 x float> %787, float %688, i64 13, !dbg !15
  %789 = insertelement <32 x float> %788, float %691, i64 14, !dbg !15
  %790 = insertelement <32 x float> %789, float %694, i64 15, !dbg !15
  %791 = insertelement <32 x float> %790, float %697, i64 16, !dbg !15
  %792 = insertelement <32 x float> %791, float %700, i64 17, !dbg !15
  %793 = insertelement <32 x float> %792, float %703, i64 18, !dbg !15
  %794 = insertelement <32 x float> %793, float %706, i64 19, !dbg !15
  %795 = insertelement <32 x float> %794, float %709, i64 20, !dbg !15
  %796 = insertelement <32 x float> %795, float %712, i64 21, !dbg !15
  %797 = insertelement <32 x float> %796, float %715, i64 22, !dbg !15
  %798 = insertelement <32 x float> %797, float %718, i64 23, !dbg !15
  %799 = insertelement <32 x float> %798, float %721, i64 24, !dbg !15
  %800 = insertelement <32 x float> %799, float %724, i64 25, !dbg !15
  %801 = insertelement <32 x float> %800, float %727, i64 26, !dbg !15
  %802 = insertelement <32 x float> %801, float %730, i64 27, !dbg !15
  %803 = insertelement <32 x float> %802, float %733, i64 28, !dbg !15
  %804 = insertelement <32 x float> %803, float %736, i64 29, !dbg !15
  %805 = insertelement <32 x float> %804, float %739, i64 30, !dbg !15
  %806 = insertelement <32 x float> %805, float %742, i64 31, !dbg !15
  br label %807, !dbg !15

807:                                              ; preds = %313, %508
  %.pn102 = phi <32 x float> [ %774, %508 ], [ %475, %313 ]
  %.pn100 = phi <32 x float> [ %806, %508 ], [ %507, %313 ]
  br label %808, !dbg !15

808:                                              ; preds = %807
  %809 = extractelement <32 x float> %.pn102, i64 0, !dbg !26
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !26
  %811 = insertelement <4 x float> poison, float %810, i64 0, !dbg !26
  %812 = extractelement <32 x float> %.pn102, i64 1, !dbg !26
  %813 = call float @llvm.amdgcn.exp2.f32(float %812), !dbg !26
  %814 = fadd float %810, %813, !dbg !27
  %815 = insertelement <4 x float> %811, float %813, i64 1, !dbg !26
  %816 = extractelement <32 x float> %.pn102, i64 2, !dbg !26
  %817 = call float @llvm.amdgcn.exp2.f32(float %816), !dbg !26
  %818 = fadd float %814, %817, !dbg !27
  %819 = insertelement <4 x float> %815, float %817, i64 2, !dbg !26
  %820 = extractelement <32 x float> %.pn102, i64 3, !dbg !26
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !26
  %822 = fadd float %818, %821, !dbg !27
  %823 = insertelement <4 x float> %819, float %821, i64 3, !dbg !26
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %823) #8, !dbg !28, !frisk.register_lifetimes !1
  %824 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !28
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %824) #8, !dbg !26, !frisk.register_lifetimes !1
  %825 = extractelement <32 x float> %.pn102, i64 4, !dbg !26
  %826 = call float @llvm.amdgcn.exp2.f32(float %825), !dbg !26
  %827 = fadd float %822, %826, !dbg !27
  %828 = insertelement <4 x float> poison, float %826, i64 0, !dbg !26
  %829 = extractelement <32 x float> %.pn102, i64 5, !dbg !26
  %830 = call float @llvm.amdgcn.exp2.f32(float %829), !dbg !26
  %831 = fadd float %827, %830, !dbg !27
  %832 = insertelement <4 x float> %828, float %830, i64 1, !dbg !26
  %833 = extractelement <32 x float> %.pn102, i64 6, !dbg !26
  %834 = call float @llvm.amdgcn.exp2.f32(float %833), !dbg !26
  %835 = fadd float %831, %834, !dbg !27
  %836 = insertelement <4 x float> %832, float %834, i64 2, !dbg !26
  %837 = extractelement <32 x float> %.pn102, i64 7, !dbg !26
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !26
  %839 = fadd float %835, %838, !dbg !27
  %840 = insertelement <4 x float> %836, float %838, i64 3, !dbg !26
  %math.result273 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %840) #8, !dbg !28, !frisk.register_lifetimes !1
  %841 = fptrunc <4 x float> %math.result273 to <4 x half>, !dbg !28
  %math.packed274 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %841) #8, !dbg !26, !frisk.register_lifetimes !1
  %842 = extractelement <32 x float> %.pn102, i64 8, !dbg !26
  %843 = call float @llvm.amdgcn.exp2.f32(float %842), !dbg !26
  %844 = fadd float %839, %843, !dbg !27
  %845 = insertelement <4 x float> poison, float %843, i64 0, !dbg !26
  %846 = extractelement <32 x float> %.pn102, i64 9, !dbg !26
  %847 = call float @llvm.amdgcn.exp2.f32(float %846), !dbg !26
  %848 = fadd float %844, %847, !dbg !27
  %849 = insertelement <4 x float> %845, float %847, i64 1, !dbg !26
  %850 = extractelement <32 x float> %.pn102, i64 10, !dbg !26
  %851 = call float @llvm.amdgcn.exp2.f32(float %850), !dbg !26
  %852 = fadd float %848, %851, !dbg !27
  %853 = insertelement <4 x float> %849, float %851, i64 2, !dbg !26
  %854 = extractelement <32 x float> %.pn102, i64 11, !dbg !26
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !26
  %856 = fadd float %852, %855, !dbg !27
  %857 = insertelement <4 x float> %853, float %855, i64 3, !dbg !26
  %math.result275 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %857) #8, !dbg !28, !frisk.register_lifetimes !1
  %858 = fptrunc <4 x float> %math.result275 to <4 x half>, !dbg !28
  %math.packed276 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %858) #8, !dbg !26, !frisk.register_lifetimes !1
  %859 = extractelement <32 x float> %.pn102, i64 12, !dbg !26
  %860 = call float @llvm.amdgcn.exp2.f32(float %859), !dbg !26
  %861 = fadd float %856, %860, !dbg !27
  %862 = insertelement <4 x float> poison, float %860, i64 0, !dbg !26
  %863 = extractelement <32 x float> %.pn102, i64 13, !dbg !26
  %864 = call float @llvm.amdgcn.exp2.f32(float %863), !dbg !26
  %865 = fadd float %861, %864, !dbg !27
  %866 = insertelement <4 x float> %862, float %864, i64 1, !dbg !26
  %867 = extractelement <32 x float> %.pn102, i64 14, !dbg !26
  %868 = call float @llvm.amdgcn.exp2.f32(float %867), !dbg !26
  %869 = fadd float %865, %868, !dbg !27
  %870 = insertelement <4 x float> %866, float %868, i64 2, !dbg !26
  %871 = extractelement <32 x float> %.pn102, i64 15, !dbg !26
  %872 = call float @llvm.amdgcn.exp2.f32(float %871), !dbg !26
  %873 = fadd float %869, %872, !dbg !27
  %874 = insertelement <4 x float> %870, float %872, i64 3, !dbg !26
  %math.result277 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %874) #8, !dbg !28, !frisk.register_lifetimes !1
  %875 = fptrunc <4 x float> %math.result277 to <4 x half>, !dbg !28
  %math.packed278 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %875) #8, !dbg !26, !frisk.register_lifetimes !1
  %876 = extractelement <32 x float> %.pn102, i64 16, !dbg !26
  %877 = call float @llvm.amdgcn.exp2.f32(float %876), !dbg !26
  %878 = fadd float %873, %877, !dbg !27
  %879 = insertelement <4 x float> poison, float %877, i64 0, !dbg !26
  %880 = extractelement <32 x float> %.pn102, i64 17, !dbg !26
  %881 = call float @llvm.amdgcn.exp2.f32(float %880), !dbg !26
  %882 = fadd float %878, %881, !dbg !27
  %883 = insertelement <4 x float> %879, float %881, i64 1, !dbg !26
  %884 = extractelement <32 x float> %.pn102, i64 18, !dbg !26
  %885 = call float @llvm.amdgcn.exp2.f32(float %884), !dbg !26
  %886 = fadd float %882, %885, !dbg !27
  %887 = insertelement <4 x float> %883, float %885, i64 2, !dbg !26
  %888 = extractelement <32 x float> %.pn102, i64 19, !dbg !26
  %889 = call float @llvm.amdgcn.exp2.f32(float %888), !dbg !26
  %890 = fadd float %886, %889, !dbg !27
  %891 = insertelement <4 x float> %887, float %889, i64 3, !dbg !26
  %math.result279 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %891) #8, !dbg !28, !frisk.register_lifetimes !1
  %892 = fptrunc <4 x float> %math.result279 to <4 x half>, !dbg !28
  %math.packed280 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %892) #8, !dbg !26, !frisk.register_lifetimes !1
  %893 = extractelement <32 x float> %.pn102, i64 20, !dbg !26
  %894 = call float @llvm.amdgcn.exp2.f32(float %893), !dbg !26
  %895 = fadd float %890, %894, !dbg !27
  %896 = insertelement <4 x float> poison, float %894, i64 0, !dbg !26
  %897 = extractelement <32 x float> %.pn102, i64 21, !dbg !26
  %898 = call float @llvm.amdgcn.exp2.f32(float %897), !dbg !26
  %899 = fadd float %895, %898, !dbg !27
  %900 = insertelement <4 x float> %896, float %898, i64 1, !dbg !26
  %901 = extractelement <32 x float> %.pn102, i64 22, !dbg !26
  %902 = call float @llvm.amdgcn.exp2.f32(float %901), !dbg !26
  %903 = fadd float %899, %902, !dbg !27
  %904 = insertelement <4 x float> %900, float %902, i64 2, !dbg !26
  %905 = extractelement <32 x float> %.pn102, i64 23, !dbg !26
  %906 = call float @llvm.amdgcn.exp2.f32(float %905), !dbg !26
  %907 = fadd float %903, %906, !dbg !27
  %908 = insertelement <4 x float> %904, float %906, i64 3, !dbg !26
  %math.result281 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %908) #8, !dbg !28, !frisk.register_lifetimes !1
  %909 = fptrunc <4 x float> %math.result281 to <4 x half>, !dbg !28
  %math.packed282 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %909) #8, !dbg !26, !frisk.register_lifetimes !1
  %910 = extractelement <32 x float> %.pn102, i64 24, !dbg !26
  %911 = call float @llvm.amdgcn.exp2.f32(float %910), !dbg !26
  %912 = fadd float %907, %911, !dbg !27
  %913 = insertelement <4 x float> poison, float %911, i64 0, !dbg !26
  %914 = extractelement <32 x float> %.pn102, i64 25, !dbg !26
  %915 = call float @llvm.amdgcn.exp2.f32(float %914), !dbg !26
  %916 = fadd float %912, %915, !dbg !27
  %917 = insertelement <4 x float> %913, float %915, i64 1, !dbg !26
  %918 = extractelement <32 x float> %.pn102, i64 26, !dbg !26
  %919 = call float @llvm.amdgcn.exp2.f32(float %918), !dbg !26
  %920 = fadd float %916, %919, !dbg !27
  %921 = insertelement <4 x float> %917, float %919, i64 2, !dbg !26
  %922 = extractelement <32 x float> %.pn102, i64 27, !dbg !26
  %923 = call float @llvm.amdgcn.exp2.f32(float %922), !dbg !26
  %924 = fadd float %920, %923, !dbg !27
  %925 = insertelement <4 x float> %921, float %923, i64 3, !dbg !26
  %math.result283 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %925) #8, !dbg !28, !frisk.register_lifetimes !1
  %926 = fptrunc <4 x float> %math.result283 to <4 x half>, !dbg !28
  %math.packed284 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %926) #8, !dbg !26, !frisk.register_lifetimes !1
  %927 = extractelement <32 x float> %.pn102, i64 28, !dbg !26
  %928 = call float @llvm.amdgcn.exp2.f32(float %927), !dbg !26
  %929 = fadd float %924, %928, !dbg !27
  %930 = insertelement <4 x float> poison, float %928, i64 0, !dbg !26
  %931 = extractelement <32 x float> %.pn102, i64 29, !dbg !26
  %932 = call float @llvm.amdgcn.exp2.f32(float %931), !dbg !26
  %933 = fadd float %929, %932, !dbg !27
  %934 = insertelement <4 x float> %930, float %932, i64 1, !dbg !26
  %935 = extractelement <32 x float> %.pn102, i64 30, !dbg !26
  %936 = call float @llvm.amdgcn.exp2.f32(float %935), !dbg !26
  %937 = fadd float %933, %936, !dbg !27
  %938 = insertelement <4 x float> %934, float %936, i64 2, !dbg !26
  %939 = extractelement <32 x float> %.pn102, i64 31, !dbg !26
  %940 = call float @llvm.amdgcn.exp2.f32(float %939), !dbg !26
  %941 = fadd float %937, %940, !dbg !27
  %942 = insertelement <4 x float> %938, float %940, i64 3, !dbg !26
  %math.result285 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %942) #8, !dbg !28, !frisk.register_lifetimes !1
  %943 = fptrunc <4 x float> %math.result285 to <4 x half>, !dbg !28
  %math.packed286 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %943) #8, !dbg !26, !frisk.register_lifetimes !1
  %944 = extractelement <32 x float> %.pn100, i64 0, !dbg !26
  %945 = call float @llvm.amdgcn.exp2.f32(float %944), !dbg !26
  %946 = insertelement <4 x float> poison, float %945, i64 0, !dbg !26
  %947 = extractelement <32 x float> %.pn100, i64 1, !dbg !26
  %948 = call float @llvm.amdgcn.exp2.f32(float %947), !dbg !26
  %949 = fadd float %945, %948, !dbg !27
  %950 = insertelement <4 x float> %946, float %948, i64 1, !dbg !26
  %951 = extractelement <32 x float> %.pn100, i64 2, !dbg !26
  %952 = call float @llvm.amdgcn.exp2.f32(float %951), !dbg !26
  %953 = fadd float %949, %952, !dbg !27
  %954 = insertelement <4 x float> %950, float %952, i64 2, !dbg !26
  %955 = extractelement <32 x float> %.pn100, i64 3, !dbg !26
  %956 = call float @llvm.amdgcn.exp2.f32(float %955), !dbg !26
  %957 = fadd float %953, %956, !dbg !27
  %958 = insertelement <4 x float> %954, float %956, i64 3, !dbg !26
  %math.result287 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %958) #8, !dbg !28, !frisk.register_lifetimes !1
  %959 = fptrunc <4 x float> %math.result287 to <4 x half>, !dbg !28
  %math.packed288 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %959) #8, !dbg !26, !frisk.register_lifetimes !1
  %960 = extractelement <32 x float> %.pn100, i64 4, !dbg !26
  %961 = call float @llvm.amdgcn.exp2.f32(float %960), !dbg !26
  %962 = fadd float %957, %961, !dbg !27
  %963 = insertelement <4 x float> poison, float %961, i64 0, !dbg !26
  %964 = extractelement <32 x float> %.pn100, i64 5, !dbg !26
  %965 = call float @llvm.amdgcn.exp2.f32(float %964), !dbg !26
  %966 = fadd float %962, %965, !dbg !27
  %967 = insertelement <4 x float> %963, float %965, i64 1, !dbg !26
  %968 = extractelement <32 x float> %.pn100, i64 6, !dbg !26
  %969 = call float @llvm.amdgcn.exp2.f32(float %968), !dbg !26
  %970 = fadd float %966, %969, !dbg !27
  %971 = insertelement <4 x float> %967, float %969, i64 2, !dbg !26
  %972 = extractelement <32 x float> %.pn100, i64 7, !dbg !26
  %973 = call float @llvm.amdgcn.exp2.f32(float %972), !dbg !26
  %974 = fadd float %970, %973, !dbg !27
  %975 = insertelement <4 x float> %971, float %973, i64 3, !dbg !26
  %math.result289 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %975) #8, !dbg !28, !frisk.register_lifetimes !1
  %976 = fptrunc <4 x float> %math.result289 to <4 x half>, !dbg !28
  %math.packed290 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %976) #8, !dbg !26, !frisk.register_lifetimes !1
  %977 = extractelement <32 x float> %.pn100, i64 8, !dbg !26
  %978 = call float @llvm.amdgcn.exp2.f32(float %977), !dbg !26
  %979 = fadd float %974, %978, !dbg !27
  %980 = insertelement <4 x float> poison, float %978, i64 0, !dbg !26
  %981 = extractelement <32 x float> %.pn100, i64 9, !dbg !26
  %982 = call float @llvm.amdgcn.exp2.f32(float %981), !dbg !26
  %983 = fadd float %979, %982, !dbg !27
  %984 = insertelement <4 x float> %980, float %982, i64 1, !dbg !26
  %985 = extractelement <32 x float> %.pn100, i64 10, !dbg !26
  %986 = call float @llvm.amdgcn.exp2.f32(float %985), !dbg !26
  %987 = fadd float %983, %986, !dbg !27
  %988 = insertelement <4 x float> %984, float %986, i64 2, !dbg !26
  %989 = extractelement <32 x float> %.pn100, i64 11, !dbg !26
  %990 = call float @llvm.amdgcn.exp2.f32(float %989), !dbg !26
  %991 = fadd float %987, %990, !dbg !27
  %992 = insertelement <4 x float> %988, float %990, i64 3, !dbg !26
  %math.result291 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %992) #8, !dbg !28, !frisk.register_lifetimes !1
  %993 = fptrunc <4 x float> %math.result291 to <4 x half>, !dbg !28
  %math.packed292 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %993) #8, !dbg !26, !frisk.register_lifetimes !1
  %994 = extractelement <32 x float> %.pn100, i64 12, !dbg !26
  %995 = call float @llvm.amdgcn.exp2.f32(float %994), !dbg !26
  %996 = fadd float %991, %995, !dbg !27
  %997 = insertelement <4 x float> poison, float %995, i64 0, !dbg !26
  %998 = extractelement <32 x float> %.pn100, i64 13, !dbg !26
  %999 = call float @llvm.amdgcn.exp2.f32(float %998), !dbg !26
  %1000 = fadd float %996, %999, !dbg !27
  %1001 = insertelement <4 x float> %997, float %999, i64 1, !dbg !26
  %1002 = extractelement <32 x float> %.pn100, i64 14, !dbg !26
  %1003 = call float @llvm.amdgcn.exp2.f32(float %1002), !dbg !26
  %1004 = fadd float %1000, %1003, !dbg !27
  %1005 = insertelement <4 x float> %1001, float %1003, i64 2, !dbg !26
  %1006 = extractelement <32 x float> %.pn100, i64 15, !dbg !26
  %1007 = call float @llvm.amdgcn.exp2.f32(float %1006), !dbg !26
  %1008 = fadd float %1004, %1007, !dbg !27
  %1009 = insertelement <4 x float> %1005, float %1007, i64 3, !dbg !26
  %math.result293 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1009) #8, !dbg !28, !frisk.register_lifetimes !1
  %1010 = fptrunc <4 x float> %math.result293 to <4 x half>, !dbg !28
  %math.packed294 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1010) #8, !dbg !26, !frisk.register_lifetimes !1
  %1011 = extractelement <32 x float> %.pn100, i64 16, !dbg !26
  %1012 = call float @llvm.amdgcn.exp2.f32(float %1011), !dbg !26
  %1013 = fadd float %1008, %1012, !dbg !27
  %1014 = insertelement <4 x float> poison, float %1012, i64 0, !dbg !26
  %1015 = extractelement <32 x float> %.pn100, i64 17, !dbg !26
  %1016 = call float @llvm.amdgcn.exp2.f32(float %1015), !dbg !26
  %1017 = fadd float %1013, %1016, !dbg !27
  %1018 = insertelement <4 x float> %1014, float %1016, i64 1, !dbg !26
  %1019 = extractelement <32 x float> %.pn100, i64 18, !dbg !26
  %1020 = call float @llvm.amdgcn.exp2.f32(float %1019), !dbg !26
  %1021 = fadd float %1017, %1020, !dbg !27
  %1022 = insertelement <4 x float> %1018, float %1020, i64 2, !dbg !26
  %1023 = extractelement <32 x float> %.pn100, i64 19, !dbg !26
  %1024 = call float @llvm.amdgcn.exp2.f32(float %1023), !dbg !26
  %1025 = fadd float %1021, %1024, !dbg !27
  %1026 = insertelement <4 x float> %1022, float %1024, i64 3, !dbg !26
  %math.result295 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1026) #8, !dbg !28, !frisk.register_lifetimes !1
  %1027 = fptrunc <4 x float> %math.result295 to <4 x half>, !dbg !28
  %math.packed296 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1027) #8, !dbg !26, !frisk.register_lifetimes !1
  %1028 = extractelement <32 x float> %.pn100, i64 20, !dbg !26
  %1029 = call float @llvm.amdgcn.exp2.f32(float %1028), !dbg !26
  %1030 = fadd float %1025, %1029, !dbg !27
  %1031 = insertelement <4 x float> poison, float %1029, i64 0, !dbg !26
  %1032 = extractelement <32 x float> %.pn100, i64 21, !dbg !26
  %1033 = call float @llvm.amdgcn.exp2.f32(float %1032), !dbg !26
  %1034 = fadd float %1030, %1033, !dbg !27
  %1035 = insertelement <4 x float> %1031, float %1033, i64 1, !dbg !26
  %1036 = extractelement <32 x float> %.pn100, i64 22, !dbg !26
  %1037 = call float @llvm.amdgcn.exp2.f32(float %1036), !dbg !26
  %1038 = fadd float %1034, %1037, !dbg !27
  %1039 = insertelement <4 x float> %1035, float %1037, i64 2, !dbg !26
  %1040 = extractelement <32 x float> %.pn100, i64 23, !dbg !26
  %1041 = call float @llvm.amdgcn.exp2.f32(float %1040), !dbg !26
  %1042 = fadd float %1038, %1041, !dbg !27
  %1043 = insertelement <4 x float> %1039, float %1041, i64 3, !dbg !26
  %math.result297 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1043) #8, !dbg !28, !frisk.register_lifetimes !1
  %1044 = fptrunc <4 x float> %math.result297 to <4 x half>, !dbg !28
  %math.packed298 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1044) #8, !dbg !26, !frisk.register_lifetimes !1
  %1045 = extractelement <32 x float> %.pn100, i64 24, !dbg !26
  %1046 = call float @llvm.amdgcn.exp2.f32(float %1045), !dbg !26
  %1047 = fadd float %1042, %1046, !dbg !27
  %1048 = insertelement <4 x float> poison, float %1046, i64 0, !dbg !26
  %1049 = extractelement <32 x float> %.pn100, i64 25, !dbg !26
  %1050 = call float @llvm.amdgcn.exp2.f32(float %1049), !dbg !26
  %1051 = fadd float %1047, %1050, !dbg !27
  %1052 = insertelement <4 x float> %1048, float %1050, i64 1, !dbg !26
  %1053 = extractelement <32 x float> %.pn100, i64 26, !dbg !26
  %1054 = call float @llvm.amdgcn.exp2.f32(float %1053), !dbg !26
  %1055 = fadd float %1051, %1054, !dbg !27
  %1056 = insertelement <4 x float> %1052, float %1054, i64 2, !dbg !26
  %1057 = extractelement <32 x float> %.pn100, i64 27, !dbg !26
  %1058 = call float @llvm.amdgcn.exp2.f32(float %1057), !dbg !26
  %1059 = fadd float %1055, %1058, !dbg !27
  %1060 = insertelement <4 x float> %1056, float %1058, i64 3, !dbg !26
  %math.result299 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1060) #8, !dbg !28, !frisk.register_lifetimes !1
  %1061 = fptrunc <4 x float> %math.result299 to <4 x half>, !dbg !28
  %math.packed300 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1061) #8, !dbg !26, !frisk.register_lifetimes !1
  %1062 = extractelement <32 x float> %.pn100, i64 28, !dbg !26
  %1063 = call float @llvm.amdgcn.exp2.f32(float %1062), !dbg !26
  %1064 = fadd float %1059, %1063, !dbg !27
  %1065 = insertelement <4 x float> poison, float %1063, i64 0, !dbg !26
  %1066 = extractelement <32 x float> %.pn100, i64 29, !dbg !26
  %1067 = call float @llvm.amdgcn.exp2.f32(float %1066), !dbg !26
  %1068 = fadd float %1064, %1067, !dbg !27
  %1069 = insertelement <4 x float> %1065, float %1067, i64 1, !dbg !26
  %1070 = extractelement <32 x float> %.pn100, i64 30, !dbg !26
  %1071 = call float @llvm.amdgcn.exp2.f32(float %1070), !dbg !26
  %1072 = fadd float %1068, %1071, !dbg !27
  %1073 = insertelement <4 x float> %1069, float %1071, i64 2, !dbg !26
  %1074 = extractelement <32 x float> %.pn100, i64 31, !dbg !26
  %1075 = call float @llvm.amdgcn.exp2.f32(float %1074), !dbg !26
  %1076 = fadd float %1072, %1075, !dbg !27
  %1077 = insertelement <4 x float> %1073, float %1075, i64 3, !dbg !26
  %math.result301 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1077) #8, !dbg !28, !frisk.register_lifetimes !1
  %1078 = fptrunc <4 x float> %math.result301 to <4 x half>, !dbg !28
  %math.packed302 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1078) #8, !dbg !27, !frisk.register_lifetimes !1
  %1079 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %1080 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1079), !dbg !27
  %1081 = add nuw nsw i32 %1080, 64, !dbg !27
  %1082 = xor i32 %1080, 16, !dbg !27
  %1083 = and i32 %1081, 192, !dbg !27
  %1084 = icmp samesign ult i32 %1082, %1083, !dbg !27
  %1085 = select i1 %1084, i32 %1082, i32 %1080, !dbg !27
  %1086 = shl nuw nsw i32 %1085, 2, !dbg !27
  %1087 = bitcast float %941 to i32, !dbg !27
  %1088 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1086, i32 %1087), !dbg !27
  %1089 = bitcast i32 %1088 to float, !dbg !27
  %1090 = fadd float %941, %1089, !dbg !27
  %1091 = xor i32 %1080, 32, !dbg !27
  %1092 = icmp samesign ult i32 %1091, %1083, !dbg !27
  %1093 = select i1 %1092, i32 %1091, i32 %1080, !dbg !27
  %1094 = shl nuw nsw i32 %1093, 2, !dbg !27
  %1095 = bitcast float %1090 to i32, !dbg !27
  %1096 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1094, i32 %1095), !dbg !27
  %1097 = bitcast i32 %1096 to float, !dbg !27
  %1098 = fadd float %1090, %1097, !dbg !27
  %1099 = extractvalue [2 x <1 x float>] %58, 0, !dbg !27
  %1100 = extractelement <1 x float> %1099, i64 0, !dbg !27
  %1101 = fadd float %1098, %1100, !dbg !27
  %1102 = bitcast float %1076 to i32, !dbg !27
  %1103 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1086, i32 %1102), !dbg !27
  %1104 = bitcast i32 %1103 to float, !dbg !27
  %1105 = fadd float %1076, %1104, !dbg !27
  %1106 = bitcast float %1105 to i32, !dbg !27
  %1107 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1094, i32 %1106), !dbg !27
  %1108 = bitcast i32 %1107 to float, !dbg !27
  %1109 = fadd float %1105, %1108, !dbg !27
  %1110 = extractvalue [2 x <1 x float>] %58, 1, !dbg !27
  %1111 = extractelement <1 x float> %1110, i64 0, !dbg !27
  %1112 = fadd float %1109, %1111, !dbg !27
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %1113, !dbg !7

1113:                                             ; preds = %1116, %808
  %1114 = phi i64 [ %1164, %1116 ], [ %20, %808 ], !dbg !7
  %1115 = icmp samesign ult i64 %1114, 2048, !dbg !7
  br i1 %1115, label %1116, label %1165, !dbg !7

1116:                                             ; preds = %1113
  %1117 = shl nuw nsw i64 %1114, 1, !dbg !7
  %1118 = lshr i64 %1114, 2, !dbg !7
  %1119 = lshr i64 %1114, 5, !dbg !7
  %1120 = and i64 %1118, 24, !dbg !7
  %1121 = lshr i64 %1114, 4, !dbg !7
  %1122 = lshr i64 %1114, 7, !dbg !7
  %1123 = sub nsw i64 %1121, %1122, !dbg !7
  %1124 = shl nsw i64 %1123, 2, !dbg !7
  %1125 = and i64 %1124, 4, !dbg !7
  %1126 = sub nsw i64 %1118, %1119, !dbg !7
  %1127 = and i64 %1126, 3, !dbg !7
  %1128 = shl nuw nsw i64 %1122, 3, !dbg !7
  %1129 = shl nuw nsw i64 %1114, 1, !dbg !7
  %1130 = and i64 %1129, 4, !dbg !7
  %1131 = or disjoint i64 %1128, %1130, !dbg !7
  %1132 = and i64 %1117, 2, !dbg !7
  %1133 = or disjoint i64 %1131, %1132, !dbg !7
  %1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1120, !dbg !7
  %1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1134, i64 %1125, !dbg !7
  %1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1135, i64 %1127, !dbg !7
  %.idx108 = shl nuw nsw i64 %1133, 6, !dbg !7
  %1137 = getelementptr inbounds nuw i8, ptr addrspace(3) %1136, i64 %.idx108, !dbg !7
  %1138 = load half, ptr addrspace(3) %1137, align 2, !dbg !7
  %1139 = lshr i64 %1114, 2, !dbg !7
  %1140 = lshr i64 %1114, 5, !dbg !7
  %1141 = and i64 %1139, 24, !dbg !7
  %1142 = lshr i64 %1114, 4, !dbg !7
  %1143 = lshr i64 %1114, 7, !dbg !7
  %1144 = sub nsw i64 %1142, %1143, !dbg !7
  %1145 = shl nsw i64 %1144, 2, !dbg !7
  %1146 = and i64 %1145, 4, !dbg !7
  %1147 = sub nsw i64 %1139, %1140, !dbg !7
  %1148 = and i64 %1147, 3, !dbg !7
  %1149 = shl nuw nsw i64 %1143, 3, !dbg !7
  %1150 = shl nuw nsw i64 %1114, 1, !dbg !7
  %1151 = and i64 %1150, 4, !dbg !7
  %1152 = or disjoint i64 %1149, %1151, !dbg !7
  %1153 = and i64 %1117, 2, !dbg !7
  %1154 = or disjoint i64 %1153, 1, !dbg !7
  %1155 = or disjoint i64 %1152, %1154, !dbg !7
  %1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1141, !dbg !7
  %1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1156, i64 %1146, !dbg !7
  %1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1157, i64 %1148, !dbg !7
  %.idx109 = shl nuw nsw i64 %1155, 6, !dbg !7
  %1159 = getelementptr inbounds nuw i8, ptr addrspace(3) %1158, i64 %.idx109, !dbg !7
  %1160 = load half, ptr addrspace(3) %1159, align 2, !dbg !7
  %1161 = insertelement <2 x half> poison, half %1138, i64 0, !dbg !7
  %1162 = insertelement <2 x half> %1161, half %1160, i64 1, !dbg !7
  %1163 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %1117, !dbg !7
  store <2 x half> %1162, ptr addrspace(3) %1163, align 2, !dbg !7
  %1164 = add nuw nsw i64 %1114, 256, !dbg !7
  br label %1113, !dbg !7

1165:                                             ; preds = %1113
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1166 = add nsw i64 %64, %8, !dbg !7
  %1167 = shl i64 %1166, 19, !dbg !7
  %1168 = and i64 %1167, 16252928, !dbg !7
  %1169 = add i64 %1168, %283, !dbg !7
  %1170 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1169, !dbg !7
  %1171 = getelementptr i8, ptr addrspace(1) %1170, i64 64, !dbg !7
  %address.thread917 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !7, !range !33, !frisk.register_lifetimes !1
  %address.local918 = zext nneg i32 %address.thread917 to i64, !dbg !6
  %address.local919 = shl nuw nsw i64 %address.local918, 1, !dbg !14
  %address.local920 = and i64 %address.local919, 384, !dbg !14
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local920, !dbg !7
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !7
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !7
  %1172 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !7
  %1173 = zext i32 %1172 to i64, !dbg !7
  %1174 = inttoptr i64 %1173 to ptr addrspace(3), !dbg !7
  %address.local924 = shl nuw nsw i64 %address.local918, 3, !dbg !7
  %address.local925 = and i64 %address.local924, 1920, !dbg !7
  %address.local926 = shl nuw nsw i64 %address.local918, 1, !dbg !14
  %address.local927 = and i64 %address.local926, 30, !dbg !7
  %address.local928 = or disjoint i64 %address.local925, %address.local927, !dbg !7
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %1171, i64 %address.local928, !dbg !7
  %dma.source.bits1966 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !7
  %dma.source.opaque1967 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1966) #9, !dbg !7
  %dma.source.ptr1968 = inttoptr i64 %dma.source.opaque1967 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1968, ptr addrspace(3) %1174, i32 4, i32 0, i32 0), !dbg !7
  %address.local930 = or disjoint i64 %address.local918, 256, !dbg !14
  %address.local931 = shl nuw nsw i64 %address.local930, 1, !dbg !14
  %address.local932 = and i64 %address.local931, 896, !dbg !14
  %address.local933 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local932, !dbg !7
  %address.local934 = ptrtoint ptr addrspace(3) %address.local933 to i64, !dbg !7
  %address.local935 = trunc i64 %address.local934 to i32, !dbg !7
  %1175 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local935), !dbg !7
  %1176 = zext i32 %1175 to i64, !dbg !7
  %1177 = inttoptr i64 %1176 to ptr addrspace(3), !dbg !7
  %address.local936 = shl nuw nsw i64 %address.local930, 3, !dbg !7
  %address.local937 = and i64 %address.local936, 3968, !dbg !7
  %address.local938 = shl nuw nsw i64 %address.local930, 1, !dbg !14
  %address.local939 = and i64 %address.local938, 30, !dbg !7
  %address.local940 = or disjoint i64 %address.local937, %address.local939, !dbg !7
  %address.local941 = getelementptr [2 x i8], ptr addrspace(1) %1171, i64 %address.local940, !dbg !7
  %dma.source.bits1969 = ptrtoint ptr addrspace(1) %address.local941 to i64, !dbg !7
  %dma.source.opaque1970 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1969) #9, !dbg !7
  %dma.source.ptr1971 = inttoptr i64 %dma.source.opaque1970 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1971, ptr addrspace(3) %1177, i32 4, i32 0, i32 0), !dbg !7
  %address.local942 = or disjoint i64 %address.local918, 512, !dbg !14
  %address.local943 = shl nuw nsw i64 %address.local942, 1, !dbg !14
  %address.local944 = and i64 %address.local943, 1408, !dbg !14
  %address.local945 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local944, !dbg !7
  %address.local946 = ptrtoint ptr addrspace(3) %address.local945 to i64, !dbg !7
  %address.local947 = trunc i64 %address.local946 to i32, !dbg !7
  %1178 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local947), !dbg !7
  %1179 = zext i32 %1178 to i64, !dbg !7
  %1180 = inttoptr i64 %1179 to ptr addrspace(3), !dbg !7
  %address.local948 = shl nuw nsw i64 %address.local942, 3, !dbg !7
  %address.local949 = and i64 %address.local948, 6016, !dbg !7
  %address.local950 = shl nuw nsw i64 %address.local942, 1, !dbg !14
  %address.local951 = and i64 %address.local950, 30, !dbg !7
  %address.local952 = or disjoint i64 %address.local949, %address.local951, !dbg !7
  %address.local953 = getelementptr [2 x i8], ptr addrspace(1) %1171, i64 %address.local952, !dbg !7
  %dma.source.bits1972 = ptrtoint ptr addrspace(1) %address.local953 to i64, !dbg !7
  %dma.source.opaque1973 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1972) #9, !dbg !7
  %dma.source.ptr1974 = inttoptr i64 %dma.source.opaque1973 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1974, ptr addrspace(3) %1180, i32 4, i32 0, i32 0), !dbg !7
  %address.local954 = or disjoint i64 %address.local918, 768, !dbg !14
  %address.local955 = shl nuw nsw i64 %address.local954, 1, !dbg !14
  %address.local956 = and i64 %address.local955, 1920, !dbg !14
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local956, !dbg !7
  %address.local958 = ptrtoint ptr addrspace(3) %address.local957 to i64, !dbg !7
  %address.local959 = trunc i64 %address.local958 to i32, !dbg !7
  %1181 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local959), !dbg !7
  %1182 = zext i32 %1181 to i64, !dbg !7
  %1183 = inttoptr i64 %1182 to ptr addrspace(3), !dbg !7
  %address.local960 = shl nuw nsw i64 %address.local954, 3, !dbg !7
  %address.local961 = and i64 %address.local960, 8064, !dbg !7
  %address.local962 = shl nuw nsw i64 %address.local954, 1, !dbg !14
  %address.local963 = and i64 %address.local962, 30, !dbg !7
  %address.local964 = or disjoint i64 %address.local961, %address.local963, !dbg !7
  %address.local965 = getelementptr [2 x i8], ptr addrspace(1) %1171, i64 %address.local964, !dbg !7
  %dma.source.bits1975 = ptrtoint ptr addrspace(1) %address.local965 to i64, !dbg !7
  %dma.source.opaque1976 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1975) #9, !dbg !7
  %dma.source.ptr1977 = inttoptr i64 %dma.source.opaque1976 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1977, ptr addrspace(3) %1183, i32 4, i32 0, i32 0), !dbg !7
  %address.local966 = or disjoint i64 %address.local918, 1024, !dbg !14
  %address.local967 = shl nuw nsw i64 %address.local966, 1, !dbg !14
  %address.local968 = and i64 %address.local967, 2432, !dbg !14
  %address.local969 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local968, !dbg !7
  %address.local970 = ptrtoint ptr addrspace(3) %address.local969 to i64, !dbg !7
  %address.local971 = trunc i64 %address.local970 to i32, !dbg !7
  %1184 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local971), !dbg !7
  %1185 = zext i32 %1184 to i64, !dbg !7
  %1186 = inttoptr i64 %1185 to ptr addrspace(3), !dbg !7
  %address.local972 = shl nuw nsw i64 %address.local966, 3, !dbg !7
  %address.local973 = and i64 %address.local972, 10112, !dbg !7
  %address.local974 = shl nuw nsw i64 %address.local966, 1, !dbg !14
  %address.local975 = and i64 %address.local974, 30, !dbg !7
  %address.local976 = or disjoint i64 %address.local973, %address.local975, !dbg !7
  %address.local977 = getelementptr [2 x i8], ptr addrspace(1) %1171, i64 %address.local976, !dbg !7
  %dma.source.bits1978 = ptrtoint ptr addrspace(1) %address.local977 to i64, !dbg !7
  %dma.source.opaque1979 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1978) #9, !dbg !7
  %dma.source.ptr1980 = inttoptr i64 %dma.source.opaque1979 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1980, ptr addrspace(3) %1186, i32 4, i32 0, i32 0), !dbg !7
  %address.local978 = or disjoint i64 %address.local918, 1280, !dbg !14
  %address.local979 = shl nuw nsw i64 %address.local978, 1, !dbg !14
  %address.local980 = and i64 %address.local979, 2944, !dbg !14
  %address.local981 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local980, !dbg !7
  %address.local982 = ptrtoint ptr addrspace(3) %address.local981 to i64, !dbg !7
  %address.local983 = trunc i64 %address.local982 to i32, !dbg !7
  %1187 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local983), !dbg !7
  %1188 = zext i32 %1187 to i64, !dbg !7
  %1189 = inttoptr i64 %1188 to ptr addrspace(3), !dbg !7
  %address.local984 = shl nuw nsw i64 %address.local978, 3, !dbg !7
  %address.local985 = and i64 %address.local984, 12160, !dbg !7
  %address.local986 = shl nuw nsw i64 %address.local978, 1, !dbg !14
  %address.local987 = and i64 %address.local986, 30, !dbg !7
  %address.local988 = or disjoint i64 %address.local985, %address.local987, !dbg !7
  %address.local989 = getelementptr [2 x i8], ptr addrspace(1) %1171, i64 %address.local988, !dbg !7
  %dma.source.bits1981 = ptrtoint ptr addrspace(1) %address.local989 to i64, !dbg !7
  %dma.source.opaque1982 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1981) #9, !dbg !7
  %dma.source.ptr1983 = inttoptr i64 %dma.source.opaque1982 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1983, ptr addrspace(3) %1189, i32 4, i32 0, i32 0), !dbg !7
  %address.local990 = or disjoint i64 %address.local918, 1536, !dbg !14
  %address.local991 = shl nuw nsw i64 %address.local990, 1, !dbg !14
  %address.local992 = and i64 %address.local991, 3456, !dbg !14
  %address.local993 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local992, !dbg !7
  %address.local994 = ptrtoint ptr addrspace(3) %address.local993 to i64, !dbg !7
  %address.local995 = trunc i64 %address.local994 to i32, !dbg !7
  %1190 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local995), !dbg !7
  %1191 = zext i32 %1190 to i64, !dbg !7
  %1192 = inttoptr i64 %1191 to ptr addrspace(3), !dbg !7
  %address.local996 = shl nuw nsw i64 %address.local990, 3, !dbg !7
  %address.local997 = and i64 %address.local996, 14208, !dbg !7
  %address.local998 = shl nuw nsw i64 %address.local990, 1, !dbg !14
  %address.local999 = and i64 %address.local998, 30, !dbg !7
  %address.local1000 = or disjoint i64 %address.local997, %address.local999, !dbg !7
  %address.local1001 = getelementptr [2 x i8], ptr addrspace(1) %1171, i64 %address.local1000, !dbg !7
  %dma.source.bits1984 = ptrtoint ptr addrspace(1) %address.local1001 to i64, !dbg !7
  %dma.source.opaque1985 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1984) #9, !dbg !7
  %dma.source.ptr1986 = inttoptr i64 %dma.source.opaque1985 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1986, ptr addrspace(3) %1192, i32 4, i32 0, i32 0), !dbg !7
  %address.local1002 = or disjoint i64 %address.local918, 1792, !dbg !14
  %address.local1003 = shl nuw nsw i64 %address.local1002, 1, !dbg !14
  %address.local1004 = and i64 %address.local1003, 3968, !dbg !14
  %address.local1005 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1004, !dbg !7
  %address.local1006 = ptrtoint ptr addrspace(3) %address.local1005 to i64, !dbg !7
  %address.local1007 = trunc i64 %address.local1006 to i32, !dbg !7
  %1193 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1007), !dbg !7
  %1194 = zext i32 %1193 to i64, !dbg !7
  %1195 = inttoptr i64 %1194 to ptr addrspace(3), !dbg !7
  %address.local1008 = shl nuw nsw i64 %address.local1002, 3, !dbg !7
  %address.local1009 = and i64 %address.local1008, 16256, !dbg !7
  %address.local1010 = shl nuw nsw i64 %address.local1002, 1, !dbg !14
  %address.local1011 = and i64 %address.local1010, 30, !dbg !7
  %address.local1012 = or disjoint i64 %address.local1009, %address.local1011, !dbg !7
  %address.local1013 = getelementptr [2 x i8], ptr addrspace(1) %1171, i64 %address.local1012, !dbg !7
  %dma.source.bits1987 = ptrtoint ptr addrspace(1) %address.local1013 to i64, !dbg !7
  %dma.source.opaque1988 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1987) #9, !dbg !7
  %dma.source.ptr1989 = inttoptr i64 %dma.source.opaque1988 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1989, ptr addrspace(3) %1195, i32 4, i32 0, i32 0), !dbg !7
  call void @llvm.amdgcn.asyncmark(), !dbg !7
  %address.local1014 = shl nuw nsw i64 %address.local918, 3, !dbg !18
  %address.local1015 = lshr i64 %address.local918, 4, !dbg !6
  %address.local1016 = mul nsw i64 %address.local1015, -128, !dbg !18
  %address.local1017 = add nsw i64 %address.local1014, %address.local1016, !dbg !18
  %address.local1018 = and i64 %address.local918, 15, !dbg !8
  %address.local1019 = lshr i64 %address.local1018, 3, !dbg !18
  %address.local1020 = mul nuw nsw i64 %address.local1019, 72, !dbg !18
  %address.local1021 = add nsw i64 %address.local1017, %address.local1020, !dbg !18
  %address.local1022 = lshr i64 %address.local1018, 2, !dbg !8
  %address.local1023 = shl nuw nsw i64 %address.local1022, 5, !dbg !18
  %address.local1024 = add nsw i64 %address.local1021, %address.local1023, !dbg !18
  %address.local1025 = lshr i64 %address.local1018, 3, !dbg !18
  %address.local1026 = mul nuw nsw i64 %address.local1025, -64, !dbg !18
  %address.local1027 = add nsw i64 %address.local1024, %address.local1026, !dbg !18
  %address.local1028 = add nuw nsw i64 %address.local1018, %address.local1019, !dbg !18
  %address.local1029 = lshr i64 %address.local1028, 2, !dbg !18
  %address.local1030 = mul nsw i64 %address.local1029, -32, !dbg !18
  %address.local1031 = add i64 %address.local1027, %address.local1030, !dbg !18
  %address.local1032 = lshr i64 %address.local918, 4, !dbg !6
  %address.local1033 = and i64 %address.local1032, 3, !dbg !6
  %address.local1034 = shl nuw nsw i64 %address.local1033, 9, !dbg !6
  %address.local1035 = add i64 %address.local1031, %address.local1034, !dbg !18
  %address.local1036 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1035, !dbg !18
  %1196 = load <8 x half>, ptr addrspace(3) %address.local1036, align 2, !dbg !18
  %1197 = extractvalue [2 x <8 x float>] %57, 0, !dbg !29
  %1198 = shufflevector <8 x float> %1197, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1199 = shufflevector <8 x half> %1196, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1200 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1199, <4 x float> %1198) #9, !dbg !18
  %address.local1037 = mul nuw nsw i64 %address.local1033, 576, !dbg !18
  %address.local1038 = add nuw nsw i64 %address.local1014, %address.local1037, !dbg !18
  %address.local1039 = add nsw i64 %address.local1038, %address.local1016, !dbg !18
  %address.local1040 = add nsw i64 %address.local1039, %address.local1020, !dbg !18
  %address.local1041 = add nsw i64 %address.local1040, %address.local1023, !dbg !18
  %address.local1042 = shl nuw nsw i32 %address.thread917, 2, !dbg !18
  %address.local1043 = and i32 %address.local1042, 192, !dbg !18
  %address.local1044 = or disjoint i32 %address.local1043, 32, !dbg !18
  %address.local1045 = shl nuw nsw i32 %address.thread917, 3, !dbg !18
  %address.local1046 = and i32 %address.local1045, 96, !dbg !18
  %address.local1047 = add nuw nsw i32 %address.local1044, %address.local1046, !dbg !18
  %address.local1048 = and i32 %address.local1047, 448, !dbg !18
  %address.local1049 = zext nneg i32 %address.local1048 to i64, !dbg !18
  %address.local1050 = sub nsw i64 %address.local1041, %address.local1049, !dbg !18
  %address.local1051 = add i64 %address.local1050, %address.local1030, !dbg !18
  %address.local1052 = add i64 %address.local1051, 288, !dbg !18
  %address.local1053 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1052, !dbg !18
  %1201 = load <8 x half>, ptr addrspace(3) %address.local1053, align 2, !dbg !18
  %1202 = shufflevector <8 x half> %1196, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1203 = shufflevector <8 x float> %1197, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1204 = extractvalue [2 x <8 x float>] %57, 1, !dbg !29
  %1205 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1202, <4 x float> %1200) #9, !dbg !18
  %1206 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1207 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1208 = shufflevector <8 x float> %1204, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1209 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1206, <4 x float> %1205) #9, !dbg !18
  %1210 = shufflevector <8 x float> %1204, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1211 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1207, <4 x float> %1209) #9, !dbg !18
  %address.local1054 = shl nuw nsw i32 %address.thread917, 3, !dbg !18
  %address.local1055 = and i32 %address.local1054, 120, !dbg !18
  %address.local1056 = shl nuw nsw i32 %address.thread917, 5, !dbg !18
  %address.local1057 = and i32 %address.local1056, 1536, !dbg !18
  %address.local1058 = or disjoint i32 %address.local1055, %address.local1057, !dbg !18
  %address.local1059 = zext nneg i32 %address.local1058 to i64, !dbg !18
  %address.local1060 = add nuw nsw i64 %address.local1020, %address.local1059, !dbg !18
  %address.local1061 = add nuw nsw i64 %address.local1060, %address.local1023, !dbg !18
  %address.local1062 = add nsw i64 %address.local1061, %address.local1026, !dbg !18
  %address.local1063 = add nsw i64 %address.local1062, %address.local1030, !dbg !18
  %address.local1064 = add i64 %address.local1063, 2048, !dbg !18
  %address.local1065 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1064, !dbg !18
  %1212 = load <8 x half>, ptr addrspace(3) %address.local1065, align 2, !dbg !18
  %address.local1066 = add nuw nsw i64 %address.local1038, 2304, !dbg !18
  %address.local1067 = add nsw i64 %address.local1066, %address.local1016, !dbg !18
  %address.local1068 = add nsw i64 %address.local1067, %address.local1020, !dbg !18
  %address.local1069 = add nsw i64 %address.local1068, %address.local1023, !dbg !18
  %address.local1070 = shl nuw nsw i32 %address.thread917, 2, !dbg !18
  %address.local1071 = and i32 %address.local1070, 192, !dbg !18
  %address.local1072 = or disjoint i32 %address.local1071, 288, !dbg !18
  %address.local1073 = shl nuw nsw i32 %address.thread917, 3, !dbg !18
  %address.local1074 = and i32 %address.local1073, 96, !dbg !18
  %address.local1075 = add nuw nsw i32 %address.local1072, %address.local1074, !dbg !18
  %address.local1076 = and i32 %address.local1075, 960, !dbg !18
  %address.local1077 = zext nneg i32 %address.local1076 to i64, !dbg !18
  %address.local1078 = sub i64 %address.local1069, %address.local1077, !dbg !18
  %address.local1079 = add i64 %address.local1078, %address.local1030, !dbg !18
  %address.local1080 = add i64 %address.local1079, 288, !dbg !18
  %address.local1081 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1080, !dbg !18
  %1213 = load <8 x half>, ptr addrspace(3) %address.local1081, align 2, !dbg !18
  %address.local1082 = add nsw i64 %address.local1021, %address.local1023, !dbg !18
  %address.local1083 = shl nuw nsw i64 %address.local918, 3, !dbg !18
  %address.local1084 = and i64 %address.local1083, 64, !dbg !18
  %address.local1085 = sub nsw i64 %address.local1082, %address.local1084, !dbg !18
  %address.local1086 = add nuw nsw i64 %address.local1028, 18, !dbg !18
  %address.local1087 = lshr i64 %address.local1086, 2, !dbg !18
  %address.local1088 = mul nsw i64 %address.local1087, -32, !dbg !18
  %address.local1089 = add i64 %address.local1085, %address.local1088, !dbg !18
  %address.local1090 = add i64 %address.local1089, %address.local1034, !dbg !18
  %address.local1091 = add i64 %address.local1090, 272, !dbg !18
  %address.local1092 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1091, !dbg !18
  %1214 = load <8 x half>, ptr addrspace(3) %address.local1092, align 2, !dbg !18
  %1215 = shufflevector <8 x half> %1212, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1216 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1215, <4 x float> %1211) #9, !dbg !18
  %1217 = shufflevector <8 x half> %1212, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1218 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1219 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1220 = shufflevector <8 x half> %1214, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1221 = shufflevector <8 x half> %1214, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %address.local1093 = add nsw i64 %address.local1040, %address.local1023, !dbg !18
  %address.local1094 = shl nuw nsw i32 %address.thread917, 2, !dbg !18
  %address.local1095 = and i32 %address.local1094, 192, !dbg !18
  %address.local1096 = add nuw nsw i32 %address.local1095, 160, !dbg !18
  %address.local1097 = shl nuw nsw i32 %address.thread917, 3, !dbg !18
  %address.local1098 = and i32 %address.local1097, 96, !dbg !18
  %address.local1099 = add nuw nsw i32 %address.local1096, %address.local1098, !dbg !18
  %address.local1100 = and i32 %address.local1099, 960, !dbg !18
  %address.local1101 = zext nneg i32 %address.local1100 to i64, !dbg !18
  %address.local1102 = sub nsw i64 %address.local1093, %address.local1101, !dbg !18
  %address.local1103 = add i64 %address.local1102, %address.local1088, !dbg !18
  %address.local1104 = add i64 %address.local1103, 688, !dbg !18
  %address.local1105 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1104, !dbg !18
  %1222 = load <8 x half>, ptr addrspace(3) %address.local1105, align 2, !dbg !18
  %1223 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1217, <4 x float> %1216) #9, !dbg !18
  %address.local1106 = add nsw i64 %address.local1038, %address.local1016, !dbg !18
  %address.local1107 = add nsw i64 %address.local1106, %address.local1020, !dbg !18
  %address.local1108 = add nsw i64 %address.local1107, %address.local1023, !dbg !18
  %address.local1109 = lshr i64 %address.local1018, 3, !dbg !18
  %address.local1110 = add nuw nsw i64 %address.local1109, %address.local1033, !dbg !18
  %address.local1111 = shl nuw nsw i64 %address.local1110, 6, !dbg !18
  %address.local1112 = sub nsw i64 %address.local1108, %address.local1111, !dbg !18
  %address.local1113 = add i64 %address.local1112, %address.local1088, !dbg !18
  %address.local1114 = add i64 %address.local1113, 2320, !dbg !18
  %address.local1115 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1114, !dbg !18
  %1224 = load <8 x half>, ptr addrspace(3) %address.local1115, align 2, !dbg !18
  %address.local1116 = add nsw i64 %address.local1068, %address.local1023, !dbg !18
  %address.local1117 = shl nuw nsw i32 %address.thread917, 2, !dbg !18
  %address.local1118 = and i32 %address.local1117, 192, !dbg !18
  %address.local1119 = add nuw nsw i32 %address.local1118, 416, !dbg !18
  %address.local1120 = shl nuw nsw i32 %address.thread917, 3, !dbg !18
  %address.local1121 = and i32 %address.local1120, 96, !dbg !18
  %address.local1122 = add nuw nsw i32 %address.local1119, %address.local1121, !dbg !18
  %address.local1123 = and i32 %address.local1122, 1984, !dbg !18
  %address.local1124 = zext nneg i32 %address.local1123 to i64, !dbg !18
  %address.local1125 = sub i64 %address.local1116, %address.local1124, !dbg !18
  %address.local1126 = add i64 %address.local1125, %address.local1088, !dbg !18
  %address.local1127 = add i64 %address.local1126, 688, !dbg !18
  %address.local1128 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1127, !dbg !18
  %1225 = load <8 x half>, ptr addrspace(3) %address.local1128, align 2, !dbg !18
  %1226 = shufflevector <8 x half> %1222, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1227 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed284, <4 x half> %1218, <4 x float> %1223) #9, !dbg !18
  %1228 = shufflevector <8 x half> %1222, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1229 = shufflevector <8 x half> %1224, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1230 = shufflevector <8 x half> %1224, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1231 = shufflevector <8 x half> %1225, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1232 = shufflevector <8 x half> %1225, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1233 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed286, <4 x half> %1219, <4 x float> %1227) #9, !dbg !18
  %1234 = load <8 x half>, ptr addrspace(3) %address.local1036, align 2, !dbg !18
  %1235 = load <8 x half>, ptr addrspace(3) %address.local1053, align 2, !dbg !18
  %1236 = load <8 x half>, ptr addrspace(3) %address.local1065, align 2, !dbg !18
  %1237 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1220, <4 x float> %1203) #9, !dbg !18
  %1238 = shufflevector <8 x half> %1234, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1239 = shufflevector <8 x half> %1234, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1240 = shufflevector <8 x half> %1235, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1241 = shufflevector <8 x half> %1235, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1242 = shufflevector <8 x half> %1236, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1243 = shufflevector <8 x half> %1236, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1244 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1221, <4 x float> %1237) #9, !dbg !18
  %1245 = load <8 x half>, ptr addrspace(3) %address.local1081, align 2, !dbg !18
  %1246 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1226, <4 x float> %1244) #9, !dbg !18
  %1247 = load <8 x half>, ptr addrspace(3) %address.local1092, align 2, !dbg !18
  %1248 = load <8 x half>, ptr addrspace(3) %address.local1105, align 2, !dbg !18
  %1249 = shufflevector <8 x half> %1245, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1250 = shufflevector <8 x half> %1245, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1251 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1228, <4 x float> %1246) #9, !dbg !18
  %1252 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1253 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1254 = shufflevector <8 x half> %1248, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1255 = shufflevector <8 x half> %1248, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1256 = load <8 x half>, ptr addrspace(3) %address.local1115, align 2, !dbg !18
  %1257 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1229, <4 x float> %1251) #9, !dbg !18
  %1258 = load <8 x half>, ptr addrspace(3) %address.local1128, align 2, !dbg !18
  %1259 = shufflevector <8 x half> %1256, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1260 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1230, <4 x float> %1257) #9, !dbg !18
  %1261 = shufflevector <8 x half> %1256, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1262 = shufflevector <8 x half> %1258, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1263 = shufflevector <8 x half> %1258, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1264 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed284, <4 x half> %1231, <4 x float> %1260) #9, !dbg !18
  %1265 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed286, <4 x half> %1232, <4 x float> %1264) #9, !dbg !18
  %1266 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed288, <4 x half> %1238, <4 x float> %1208) #9, !dbg !18
  %1267 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed290, <4 x half> %1239, <4 x float> %1266) #9, !dbg !18
  %1268 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed292, <4 x half> %1240, <4 x float> %1267) #9, !dbg !18
  %1269 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed294, <4 x half> %1241, <4 x float> %1268) #9, !dbg !18
  %1270 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed296, <4 x half> %1242, <4 x float> %1269) #9, !dbg !18
  %1271 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed298, <4 x half> %1243, <4 x float> %1270) #9, !dbg !18
  %1272 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed300, <4 x half> %1249, <4 x float> %1271) #9, !dbg !18
  %1273 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1250, <4 x float> %1272) #9, !dbg !18
  %1274 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed288, <4 x half> %1252, <4 x float> %1210) #9, !dbg !18
  %1275 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed290, <4 x half> %1253, <4 x float> %1274) #9, !dbg !18
  %1276 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed292, <4 x half> %1254, <4 x float> %1275) #9, !dbg !18
  %1277 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed294, <4 x half> %1255, <4 x float> %1276) #9, !dbg !18
  %1278 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed296, <4 x half> %1259, <4 x float> %1277) #9, !dbg !18
  %1279 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed298, <4 x half> %1261, <4 x float> %1278) #9, !dbg !18
  %1280 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed300, <4 x half> %1262, <4 x float> %1279) #9, !dbg !18
  %1281 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1263, <4 x float> %1280) #9, !dbg !18
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %1282, !dbg !7

1282:                                             ; preds = %1285, %1165
  %1283 = phi i64 [ %1333, %1285 ], [ %20, %1165 ], !dbg !7
  %1284 = icmp samesign ult i64 %1283, 2048, !dbg !7
  br i1 %1284, label %1285, label %1334, !dbg !7

1285:                                             ; preds = %1282
  %1286 = shl nuw nsw i64 %1283, 1, !dbg !7
  %1287 = lshr i64 %1283, 2, !dbg !7
  %1288 = lshr i64 %1283, 5, !dbg !7
  %1289 = and i64 %1287, 24, !dbg !7
  %1290 = lshr i64 %1283, 4, !dbg !7
  %1291 = lshr i64 %1283, 7, !dbg !7
  %1292 = sub nsw i64 %1290, %1291, !dbg !7
  %1293 = shl nsw i64 %1292, 2, !dbg !7
  %1294 = and i64 %1293, 4, !dbg !7
  %1295 = sub nsw i64 %1287, %1288, !dbg !7
  %1296 = and i64 %1295, 3, !dbg !7
  %1297 = shl nuw nsw i64 %1291, 3, !dbg !7
  %1298 = shl nuw nsw i64 %1283, 1, !dbg !7
  %1299 = and i64 %1298, 4, !dbg !7
  %1300 = or disjoint i64 %1297, %1299, !dbg !7
  %1301 = and i64 %1286, 2, !dbg !7
  %1302 = or disjoint i64 %1300, %1301, !dbg !7
  %1303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1289, !dbg !7
  %1304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1303, i64 %1294, !dbg !7
  %1305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1304, i64 %1296, !dbg !7
  %.idx106 = shl nuw nsw i64 %1302, 6, !dbg !7
  %1306 = getelementptr inbounds nuw i8, ptr addrspace(3) %1305, i64 %.idx106, !dbg !7
  %1307 = load half, ptr addrspace(3) %1306, align 2, !dbg !7
  %1308 = lshr i64 %1283, 2, !dbg !7
  %1309 = lshr i64 %1283, 5, !dbg !7
  %1310 = and i64 %1308, 24, !dbg !7
  %1311 = lshr i64 %1283, 4, !dbg !7
  %1312 = lshr i64 %1283, 7, !dbg !7
  %1313 = sub nsw i64 %1311, %1312, !dbg !7
  %1314 = shl nsw i64 %1313, 2, !dbg !7
  %1315 = and i64 %1314, 4, !dbg !7
  %1316 = sub nsw i64 %1308, %1309, !dbg !7
  %1317 = and i64 %1316, 3, !dbg !7
  %1318 = shl nuw nsw i64 %1312, 3, !dbg !7
  %1319 = shl nuw nsw i64 %1283, 1, !dbg !7
  %1320 = and i64 %1319, 4, !dbg !7
  %1321 = or disjoint i64 %1318, %1320, !dbg !7
  %1322 = and i64 %1286, 2, !dbg !7
  %1323 = or disjoint i64 %1322, 1, !dbg !7
  %1324 = or disjoint i64 %1321, %1323, !dbg !7
  %1325 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1310, !dbg !7
  %1326 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1325, i64 %1315, !dbg !7
  %1327 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1326, i64 %1317, !dbg !7
  %.idx107 = shl nuw nsw i64 %1324, 6, !dbg !7
  %1328 = getelementptr inbounds nuw i8, ptr addrspace(3) %1327, i64 %.idx107, !dbg !7
  %1329 = load half, ptr addrspace(3) %1328, align 2, !dbg !7
  %1330 = insertelement <2 x half> poison, half %1307, i64 0, !dbg !7
  %1331 = insertelement <2 x half> %1330, half %1329, i64 1, !dbg !7
  %1332 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %1286, !dbg !7
  store <2 x half> %1331, ptr addrspace(3) %1332, align 2, !dbg !7
  %1333 = add nuw nsw i64 %1283, 256, !dbg !7
  br label %1282, !dbg !7

1334:                                             ; preds = %1282
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1335 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1169, !dbg !7
  %1336 = getelementptr i8, ptr addrspace(1) %1335, i64 128, !dbg !7
  %address.thread1129 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !7, !range !33, !frisk.register_lifetimes !1
  %address.local1130 = zext nneg i32 %address.thread1129 to i64, !dbg !6
  %address.local1131 = shl nuw nsw i64 %address.local1130, 1, !dbg !14
  %address.local1132 = and i64 %address.local1131, 384, !dbg !14
  %address.local1133 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1132, !dbg !7
  %address.local1134 = ptrtoint ptr addrspace(3) %address.local1133 to i64, !dbg !7
  %address.local1135 = trunc i64 %address.local1134 to i32, !dbg !7
  %1337 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1135), !dbg !7
  %1338 = zext i32 %1337 to i64, !dbg !7
  %1339 = inttoptr i64 %1338 to ptr addrspace(3), !dbg !7
  %address.local1136 = shl nuw nsw i64 %address.local1130, 3, !dbg !7
  %address.local1137 = and i64 %address.local1136, 1920, !dbg !7
  %address.local1138 = shl nuw nsw i64 %address.local1130, 1, !dbg !14
  %address.local1139 = and i64 %address.local1138, 30, !dbg !7
  %address.local1140 = or disjoint i64 %address.local1137, %address.local1139, !dbg !7
  %address.local1141 = getelementptr [2 x i8], ptr addrspace(1) %1336, i64 %address.local1140, !dbg !7
  %dma.source.bits1990 = ptrtoint ptr addrspace(1) %address.local1141 to i64, !dbg !7
  %dma.source.opaque1991 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1990) #9, !dbg !7
  %dma.source.ptr1992 = inttoptr i64 %dma.source.opaque1991 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1992, ptr addrspace(3) %1339, i32 4, i32 0, i32 0), !dbg !7
  %address.local1142 = or disjoint i64 %address.local1130, 256, !dbg !14
  %address.local1143 = shl nuw nsw i64 %address.local1142, 1, !dbg !14
  %address.local1144 = and i64 %address.local1143, 896, !dbg !14
  %address.local1145 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1144, !dbg !7
  %address.local1146 = ptrtoint ptr addrspace(3) %address.local1145 to i64, !dbg !7
  %address.local1147 = trunc i64 %address.local1146 to i32, !dbg !7
  %1340 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1147), !dbg !7
  %1341 = zext i32 %1340 to i64, !dbg !7
  %1342 = inttoptr i64 %1341 to ptr addrspace(3), !dbg !7
  %address.local1148 = shl nuw nsw i64 %address.local1142, 3, !dbg !7
  %address.local1149 = and i64 %address.local1148, 3968, !dbg !7
  %address.local1150 = shl nuw nsw i64 %address.local1142, 1, !dbg !14
  %address.local1151 = and i64 %address.local1150, 30, !dbg !7
  %address.local1152 = or disjoint i64 %address.local1149, %address.local1151, !dbg !7
  %address.local1153 = getelementptr [2 x i8], ptr addrspace(1) %1336, i64 %address.local1152, !dbg !7
  %dma.source.bits1993 = ptrtoint ptr addrspace(1) %address.local1153 to i64, !dbg !7
  %dma.source.opaque1994 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1993) #9, !dbg !7
  %dma.source.ptr1995 = inttoptr i64 %dma.source.opaque1994 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1995, ptr addrspace(3) %1342, i32 4, i32 0, i32 0), !dbg !7
  %address.local1154 = or disjoint i64 %address.local1130, 512, !dbg !14
  %address.local1155 = shl nuw nsw i64 %address.local1154, 1, !dbg !14
  %address.local1156 = and i64 %address.local1155, 1408, !dbg !14
  %address.local1157 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1156, !dbg !7
  %address.local1158 = ptrtoint ptr addrspace(3) %address.local1157 to i64, !dbg !7
  %address.local1159 = trunc i64 %address.local1158 to i32, !dbg !7
  %1343 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1159), !dbg !7
  %1344 = zext i32 %1343 to i64, !dbg !7
  %1345 = inttoptr i64 %1344 to ptr addrspace(3), !dbg !7
  %address.local1160 = shl nuw nsw i64 %address.local1154, 3, !dbg !7
  %address.local1161 = and i64 %address.local1160, 6016, !dbg !7
  %address.local1162 = shl nuw nsw i64 %address.local1154, 1, !dbg !14
  %address.local1163 = and i64 %address.local1162, 30, !dbg !7
  %address.local1164 = or disjoint i64 %address.local1161, %address.local1163, !dbg !7
  %address.local1165 = getelementptr [2 x i8], ptr addrspace(1) %1336, i64 %address.local1164, !dbg !7
  %dma.source.bits1996 = ptrtoint ptr addrspace(1) %address.local1165 to i64, !dbg !7
  %dma.source.opaque1997 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1996) #9, !dbg !7
  %dma.source.ptr1998 = inttoptr i64 %dma.source.opaque1997 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1998, ptr addrspace(3) %1345, i32 4, i32 0, i32 0), !dbg !7
  %address.local1166 = or disjoint i64 %address.local1130, 768, !dbg !14
  %address.local1167 = shl nuw nsw i64 %address.local1166, 1, !dbg !14
  %address.local1168 = and i64 %address.local1167, 1920, !dbg !14
  %address.local1169 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1168, !dbg !7
  %address.local1170 = ptrtoint ptr addrspace(3) %address.local1169 to i64, !dbg !7
  %address.local1171 = trunc i64 %address.local1170 to i32, !dbg !7
  %1346 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1171), !dbg !7
  %1347 = zext i32 %1346 to i64, !dbg !7
  %1348 = inttoptr i64 %1347 to ptr addrspace(3), !dbg !7
  %address.local1172 = shl nuw nsw i64 %address.local1166, 3, !dbg !7
  %address.local1173 = and i64 %address.local1172, 8064, !dbg !7
  %address.local1174 = shl nuw nsw i64 %address.local1166, 1, !dbg !14
  %address.local1175 = and i64 %address.local1174, 30, !dbg !7
  %address.local1176 = or disjoint i64 %address.local1173, %address.local1175, !dbg !7
  %address.local1177 = getelementptr [2 x i8], ptr addrspace(1) %1336, i64 %address.local1176, !dbg !7
  %dma.source.bits1999 = ptrtoint ptr addrspace(1) %address.local1177 to i64, !dbg !7
  %dma.source.opaque2000 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1999) #9, !dbg !7
  %dma.source.ptr2001 = inttoptr i64 %dma.source.opaque2000 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2001, ptr addrspace(3) %1348, i32 4, i32 0, i32 0), !dbg !7
  %address.local1178 = or disjoint i64 %address.local1130, 1024, !dbg !14
  %address.local1179 = shl nuw nsw i64 %address.local1178, 1, !dbg !14
  %address.local1180 = and i64 %address.local1179, 2432, !dbg !14
  %address.local1181 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1180, !dbg !7
  %address.local1182 = ptrtoint ptr addrspace(3) %address.local1181 to i64, !dbg !7
  %address.local1183 = trunc i64 %address.local1182 to i32, !dbg !7
  %1349 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1183), !dbg !7
  %1350 = zext i32 %1349 to i64, !dbg !7
  %1351 = inttoptr i64 %1350 to ptr addrspace(3), !dbg !7
  %address.local1184 = shl nuw nsw i64 %address.local1178, 3, !dbg !7
  %address.local1185 = and i64 %address.local1184, 10112, !dbg !7
  %address.local1186 = shl nuw nsw i64 %address.local1178, 1, !dbg !14
  %address.local1187 = and i64 %address.local1186, 30, !dbg !7
  %address.local1188 = or disjoint i64 %address.local1185, %address.local1187, !dbg !7
  %address.local1189 = getelementptr [2 x i8], ptr addrspace(1) %1336, i64 %address.local1188, !dbg !7
  %dma.source.bits2002 = ptrtoint ptr addrspace(1) %address.local1189 to i64, !dbg !7
  %dma.source.opaque2003 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2002) #9, !dbg !7
  %dma.source.ptr2004 = inttoptr i64 %dma.source.opaque2003 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2004, ptr addrspace(3) %1351, i32 4, i32 0, i32 0), !dbg !7
  %address.local1190 = or disjoint i64 %address.local1130, 1280, !dbg !14
  %address.local1191 = shl nuw nsw i64 %address.local1190, 1, !dbg !14
  %address.local1192 = and i64 %address.local1191, 2944, !dbg !14
  %address.local1193 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1192, !dbg !7
  %address.local1194 = ptrtoint ptr addrspace(3) %address.local1193 to i64, !dbg !7
  %address.local1195 = trunc i64 %address.local1194 to i32, !dbg !7
  %1352 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1195), !dbg !7
  %1353 = zext i32 %1352 to i64, !dbg !7
  %1354 = inttoptr i64 %1353 to ptr addrspace(3), !dbg !7
  %address.local1196 = shl nuw nsw i64 %address.local1190, 3, !dbg !7
  %address.local1197 = and i64 %address.local1196, 12160, !dbg !7
  %address.local1198 = shl nuw nsw i64 %address.local1190, 1, !dbg !14
  %address.local1199 = and i64 %address.local1198, 30, !dbg !7
  %address.local1200 = or disjoint i64 %address.local1197, %address.local1199, !dbg !7
  %address.local1201 = getelementptr [2 x i8], ptr addrspace(1) %1336, i64 %address.local1200, !dbg !7
  %dma.source.bits2005 = ptrtoint ptr addrspace(1) %address.local1201 to i64, !dbg !7
  %dma.source.opaque2006 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2005) #9, !dbg !7
  %dma.source.ptr2007 = inttoptr i64 %dma.source.opaque2006 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2007, ptr addrspace(3) %1354, i32 4, i32 0, i32 0), !dbg !7
  %address.local1202 = or disjoint i64 %address.local1130, 1536, !dbg !14
  %address.local1203 = shl nuw nsw i64 %address.local1202, 1, !dbg !14
  %address.local1204 = and i64 %address.local1203, 3456, !dbg !14
  %address.local1205 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1204, !dbg !7
  %address.local1206 = ptrtoint ptr addrspace(3) %address.local1205 to i64, !dbg !7
  %address.local1207 = trunc i64 %address.local1206 to i32, !dbg !7
  %1355 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1207), !dbg !7
  %1356 = zext i32 %1355 to i64, !dbg !7
  %1357 = inttoptr i64 %1356 to ptr addrspace(3), !dbg !7
  %address.local1208 = shl nuw nsw i64 %address.local1202, 3, !dbg !7
  %address.local1209 = and i64 %address.local1208, 14208, !dbg !7
  %address.local1210 = shl nuw nsw i64 %address.local1202, 1, !dbg !14
  %address.local1211 = and i64 %address.local1210, 30, !dbg !7
  %address.local1212 = or disjoint i64 %address.local1209, %address.local1211, !dbg !7
  %address.local1213 = getelementptr [2 x i8], ptr addrspace(1) %1336, i64 %address.local1212, !dbg !7
  %dma.source.bits2008 = ptrtoint ptr addrspace(1) %address.local1213 to i64, !dbg !7
  %dma.source.opaque2009 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2008) #9, !dbg !7
  %dma.source.ptr2010 = inttoptr i64 %dma.source.opaque2009 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2010, ptr addrspace(3) %1357, i32 4, i32 0, i32 0), !dbg !7
  %address.local1214 = or disjoint i64 %address.local1130, 1792, !dbg !14
  %address.local1215 = shl nuw nsw i64 %address.local1214, 1, !dbg !14
  %address.local1216 = and i64 %address.local1215, 3968, !dbg !14
  %address.local1217 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1216, !dbg !7
  %address.local1218 = ptrtoint ptr addrspace(3) %address.local1217 to i64, !dbg !7
  %address.local1219 = trunc i64 %address.local1218 to i32, !dbg !7
  %1358 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1219), !dbg !7
  %1359 = zext i32 %1358 to i64, !dbg !7
  %1360 = inttoptr i64 %1359 to ptr addrspace(3), !dbg !7
  %address.local1220 = shl nuw nsw i64 %address.local1214, 3, !dbg !7
  %address.local1221 = and i64 %address.local1220, 16256, !dbg !7
  %address.local1222 = shl nuw nsw i64 %address.local1214, 1, !dbg !14
  %address.local1223 = and i64 %address.local1222, 30, !dbg !7
  %address.local1224 = or disjoint i64 %address.local1221, %address.local1223, !dbg !7
  %address.local1225 = getelementptr [2 x i8], ptr addrspace(1) %1336, i64 %address.local1224, !dbg !7
  %dma.source.bits2011 = ptrtoint ptr addrspace(1) %address.local1225 to i64, !dbg !7
  %dma.source.opaque2012 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2011) #9, !dbg !7
  %dma.source.ptr2013 = inttoptr i64 %dma.source.opaque2012 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2013, ptr addrspace(3) %1360, i32 4, i32 0, i32 0), !dbg !7
  call void @llvm.amdgcn.asyncmark(), !dbg !7
  %address.local1226 = shl nuw nsw i64 %address.local1130, 3, !dbg !18
  %address.local1227 = lshr i64 %address.local1130, 4, !dbg !6
  %address.local1228 = mul nsw i64 %address.local1227, -128, !dbg !18
  %address.local1229 = add nsw i64 %address.local1226, %address.local1228, !dbg !18
  %address.local1230 = and i64 %address.local1130, 15, !dbg !8
  %address.local1231 = lshr i64 %address.local1230, 3, !dbg !18
  %address.local1232 = mul nuw nsw i64 %address.local1231, 72, !dbg !18
  %address.local1233 = add nsw i64 %address.local1229, %address.local1232, !dbg !18
  %address.local1234 = lshr i64 %address.local1230, 2, !dbg !8
  %address.local1235 = shl nuw nsw i64 %address.local1234, 5, !dbg !18
  %address.local1236 = add nsw i64 %address.local1233, %address.local1235, !dbg !18
  %address.local1237 = lshr i64 %address.local1230, 3, !dbg !18
  %address.local1238 = mul nuw nsw i64 %address.local1237, -64, !dbg !18
  %address.local1239 = add nsw i64 %address.local1236, %address.local1238, !dbg !18
  %address.local1240 = add nuw nsw i64 %address.local1230, %address.local1231, !dbg !18
  %address.local1241 = lshr i64 %address.local1240, 2, !dbg !18
  %address.local1242 = mul nsw i64 %address.local1241, -32, !dbg !18
  %address.local1243 = add i64 %address.local1239, %address.local1242, !dbg !18
  %address.local1244 = lshr i64 %address.local1130, 4, !dbg !6
  %address.local1245 = and i64 %address.local1244, 3, !dbg !6
  %address.local1246 = shl nuw nsw i64 %address.local1245, 9, !dbg !6
  %address.local1247 = add i64 %address.local1243, %address.local1246, !dbg !18
  %address.local1248 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1247, !dbg !18
  %1361 = load <8 x half>, ptr addrspace(3) %address.local1248, align 2, !dbg !18
  %1362 = extractvalue [2 x <8 x float>] %59, 0, !dbg !29
  %1363 = shufflevector <8 x float> %1362, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1364 = shufflevector <8 x half> %1361, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1365 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1364, <4 x float> %1363) #9, !dbg !18
  %address.local1249 = mul nuw nsw i64 %address.local1245, 576, !dbg !18
  %address.local1250 = add nuw nsw i64 %address.local1226, %address.local1249, !dbg !18
  %address.local1251 = add nsw i64 %address.local1250, %address.local1228, !dbg !18
  %address.local1252 = add nsw i64 %address.local1251, %address.local1232, !dbg !18
  %address.local1253 = add nsw i64 %address.local1252, %address.local1235, !dbg !18
  %address.local1254 = shl nuw nsw i32 %address.thread1129, 2, !dbg !18
  %address.local1255 = and i32 %address.local1254, 192, !dbg !18
  %address.local1256 = or disjoint i32 %address.local1255, 32, !dbg !18
  %address.local1257 = shl nuw nsw i32 %address.thread1129, 3, !dbg !18
  %address.local1258 = and i32 %address.local1257, 96, !dbg !18
  %address.local1259 = add nuw nsw i32 %address.local1256, %address.local1258, !dbg !18
  %address.local1260 = and i32 %address.local1259, 448, !dbg !18
  %address.local1261 = zext nneg i32 %address.local1260 to i64, !dbg !18
  %address.local1262 = sub nsw i64 %address.local1253, %address.local1261, !dbg !18
  %address.local1263 = add i64 %address.local1262, %address.local1242, !dbg !18
  %address.local1264 = add i64 %address.local1263, 288, !dbg !18
  %address.local1265 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1264, !dbg !18
  %1366 = load <8 x half>, ptr addrspace(3) %address.local1265, align 2, !dbg !18
  %1367 = shufflevector <8 x half> %1361, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1368 = shufflevector <8 x float> %1362, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1369 = extractvalue [2 x <8 x float>] %59, 1, !dbg !29
  %1370 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1367, <4 x float> %1365) #9, !dbg !18
  %1371 = shufflevector <8 x half> %1366, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1372 = shufflevector <8 x half> %1366, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1373 = shufflevector <8 x float> %1369, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1374 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1371, <4 x float> %1370) #9, !dbg !18
  %1375 = shufflevector <8 x float> %1369, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1376 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1372, <4 x float> %1374) #9, !dbg !18
  %address.local1266 = shl nuw nsw i32 %address.thread1129, 3, !dbg !18
  %address.local1267 = and i32 %address.local1266, 120, !dbg !18
  %address.local1268 = shl nuw nsw i32 %address.thread1129, 5, !dbg !18
  %address.local1269 = and i32 %address.local1268, 1536, !dbg !18
  %address.local1270 = or disjoint i32 %address.local1267, %address.local1269, !dbg !18
  %address.local1271 = zext nneg i32 %address.local1270 to i64, !dbg !18
  %address.local1272 = add nuw nsw i64 %address.local1232, %address.local1271, !dbg !18
  %address.local1273 = add nuw nsw i64 %address.local1272, %address.local1235, !dbg !18
  %address.local1274 = add nsw i64 %address.local1273, %address.local1238, !dbg !18
  %address.local1275 = add nsw i64 %address.local1274, %address.local1242, !dbg !18
  %address.local1276 = add i64 %address.local1275, 2048, !dbg !18
  %address.local1277 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1276, !dbg !18
  %1377 = load <8 x half>, ptr addrspace(3) %address.local1277, align 2, !dbg !18
  %address.local1278 = add nuw nsw i64 %address.local1250, 2304, !dbg !18
  %address.local1279 = add nsw i64 %address.local1278, %address.local1228, !dbg !18
  %address.local1280 = add nsw i64 %address.local1279, %address.local1232, !dbg !18
  %address.local1281 = add nsw i64 %address.local1280, %address.local1235, !dbg !18
  %address.local1282 = shl nuw nsw i32 %address.thread1129, 2, !dbg !18
  %address.local1283 = and i32 %address.local1282, 192, !dbg !18
  %address.local1284 = or disjoint i32 %address.local1283, 288, !dbg !18
  %address.local1285 = shl nuw nsw i32 %address.thread1129, 3, !dbg !18
  %address.local1286 = and i32 %address.local1285, 96, !dbg !18
  %address.local1287 = add nuw nsw i32 %address.local1284, %address.local1286, !dbg !18
  %address.local1288 = and i32 %address.local1287, 960, !dbg !18
  %address.local1289 = zext nneg i32 %address.local1288 to i64, !dbg !18
  %address.local1290 = sub i64 %address.local1281, %address.local1289, !dbg !18
  %address.local1291 = add i64 %address.local1290, %address.local1242, !dbg !18
  %address.local1292 = add i64 %address.local1291, 288, !dbg !18
  %address.local1293 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1292, !dbg !18
  %1378 = load <8 x half>, ptr addrspace(3) %address.local1293, align 2, !dbg !18
  %address.local1294 = add nsw i64 %address.local1233, %address.local1235, !dbg !18
  %address.local1295 = shl nuw nsw i64 %address.local1130, 3, !dbg !18
  %address.local1296 = and i64 %address.local1295, 64, !dbg !18
  %address.local1297 = sub nsw i64 %address.local1294, %address.local1296, !dbg !18
  %address.local1298 = add nuw nsw i64 %address.local1240, 18, !dbg !18
  %address.local1299 = lshr i64 %address.local1298, 2, !dbg !18
  %address.local1300 = mul nsw i64 %address.local1299, -32, !dbg !18
  %address.local1301 = add i64 %address.local1297, %address.local1300, !dbg !18
  %address.local1302 = add i64 %address.local1301, %address.local1246, !dbg !18
  %address.local1303 = add i64 %address.local1302, 272, !dbg !18
  %address.local1304 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1303, !dbg !18
  %1379 = load <8 x half>, ptr addrspace(3) %address.local1304, align 2, !dbg !18
  %1380 = shufflevector <8 x half> %1377, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1381 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1380, <4 x float> %1376) #9, !dbg !18
  %1382 = shufflevector <8 x half> %1377, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1383 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1384 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1385 = shufflevector <8 x half> %1379, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1386 = shufflevector <8 x half> %1379, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %address.local1305 = add nsw i64 %address.local1252, %address.local1235, !dbg !18
  %address.local1306 = shl nuw nsw i32 %address.thread1129, 2, !dbg !18
  %address.local1307 = and i32 %address.local1306, 192, !dbg !18
  %address.local1308 = add nuw nsw i32 %address.local1307, 160, !dbg !18
  %address.local1309 = shl nuw nsw i32 %address.thread1129, 3, !dbg !18
  %address.local1310 = and i32 %address.local1309, 96, !dbg !18
  %address.local1311 = add nuw nsw i32 %address.local1308, %address.local1310, !dbg !18
  %address.local1312 = and i32 %address.local1311, 960, !dbg !18
  %address.local1313 = zext nneg i32 %address.local1312 to i64, !dbg !18
  %address.local1314 = sub nsw i64 %address.local1305, %address.local1313, !dbg !18
  %address.local1315 = add i64 %address.local1314, %address.local1300, !dbg !18
  %address.local1316 = add i64 %address.local1315, 688, !dbg !18
  %address.local1317 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1316, !dbg !18
  %1387 = load <8 x half>, ptr addrspace(3) %address.local1317, align 2, !dbg !18
  %1388 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1382, <4 x float> %1381) #9, !dbg !18
  %address.local1318 = add nsw i64 %address.local1250, %address.local1228, !dbg !18
  %address.local1319 = add nsw i64 %address.local1318, %address.local1232, !dbg !18
  %address.local1320 = add nsw i64 %address.local1319, %address.local1235, !dbg !18
  %address.local1321 = lshr i64 %address.local1230, 3, !dbg !18
  %address.local1322 = add nuw nsw i64 %address.local1321, %address.local1245, !dbg !18
  %address.local1323 = shl nuw nsw i64 %address.local1322, 6, !dbg !18
  %address.local1324 = sub nsw i64 %address.local1320, %address.local1323, !dbg !18
  %address.local1325 = add i64 %address.local1324, %address.local1300, !dbg !18
  %address.local1326 = add i64 %address.local1325, 2320, !dbg !18
  %address.local1327 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1326, !dbg !18
  %1389 = load <8 x half>, ptr addrspace(3) %address.local1327, align 2, !dbg !18
  %address.local1328 = add nsw i64 %address.local1280, %address.local1235, !dbg !18
  %address.local1329 = shl nuw nsw i32 %address.thread1129, 2, !dbg !18
  %address.local1330 = and i32 %address.local1329, 192, !dbg !18
  %address.local1331 = add nuw nsw i32 %address.local1330, 416, !dbg !18
  %address.local1332 = shl nuw nsw i32 %address.thread1129, 3, !dbg !18
  %address.local1333 = and i32 %address.local1332, 96, !dbg !18
  %address.local1334 = add nuw nsw i32 %address.local1331, %address.local1333, !dbg !18
  %address.local1335 = and i32 %address.local1334, 1984, !dbg !18
  %address.local1336 = zext nneg i32 %address.local1335 to i64, !dbg !18
  %address.local1337 = sub i64 %address.local1328, %address.local1336, !dbg !18
  %address.local1338 = add i64 %address.local1337, %address.local1300, !dbg !18
  %address.local1339 = add i64 %address.local1338, 688, !dbg !18
  %address.local1340 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1339, !dbg !18
  %1390 = load <8 x half>, ptr addrspace(3) %address.local1340, align 2, !dbg !18
  %1391 = shufflevector <8 x half> %1387, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1392 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed284, <4 x half> %1383, <4 x float> %1388) #9, !dbg !18
  %1393 = shufflevector <8 x half> %1387, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1394 = shufflevector <8 x half> %1389, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1395 = shufflevector <8 x half> %1389, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1396 = shufflevector <8 x half> %1390, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1397 = shufflevector <8 x half> %1390, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1398 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed286, <4 x half> %1384, <4 x float> %1392) #9, !dbg !18
  %1399 = load <8 x half>, ptr addrspace(3) %address.local1248, align 2, !dbg !18
  %1400 = load <8 x half>, ptr addrspace(3) %address.local1265, align 2, !dbg !18
  %1401 = load <8 x half>, ptr addrspace(3) %address.local1277, align 2, !dbg !18
  %1402 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1385, <4 x float> %1368) #9, !dbg !18
  %1403 = shufflevector <8 x half> %1399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1404 = shufflevector <8 x half> %1399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1405 = shufflevector <8 x half> %1400, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1406 = shufflevector <8 x half> %1400, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1407 = shufflevector <8 x half> %1401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1408 = shufflevector <8 x half> %1401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1409 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1386, <4 x float> %1402) #9, !dbg !18
  %1410 = load <8 x half>, ptr addrspace(3) %address.local1293, align 2, !dbg !18
  %1411 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1391, <4 x float> %1409) #9, !dbg !18
  %1412 = load <8 x half>, ptr addrspace(3) %address.local1304, align 2, !dbg !18
  %1413 = load <8 x half>, ptr addrspace(3) %address.local1317, align 2, !dbg !18
  %1414 = shufflevector <8 x half> %1410, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1415 = shufflevector <8 x half> %1410, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1416 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1393, <4 x float> %1411) #9, !dbg !18
  %1417 = shufflevector <8 x half> %1412, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1418 = shufflevector <8 x half> %1412, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1419 = shufflevector <8 x half> %1413, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1420 = shufflevector <8 x half> %1413, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1421 = load <8 x half>, ptr addrspace(3) %address.local1327, align 2, !dbg !18
  %1422 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1394, <4 x float> %1416) #9, !dbg !18
  %1423 = load <8 x half>, ptr addrspace(3) %address.local1340, align 2, !dbg !18
  %1424 = shufflevector <8 x half> %1421, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1425 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1395, <4 x float> %1422) #9, !dbg !18
  %1426 = shufflevector <8 x half> %1421, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1427 = shufflevector <8 x half> %1423, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1428 = shufflevector <8 x half> %1423, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1429 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed284, <4 x half> %1396, <4 x float> %1425) #9, !dbg !18
  %1430 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed286, <4 x half> %1397, <4 x float> %1429) #9, !dbg !18
  %1431 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed288, <4 x half> %1403, <4 x float> %1373) #9, !dbg !18
  %1432 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed290, <4 x half> %1404, <4 x float> %1431) #9, !dbg !18
  %1433 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed292, <4 x half> %1405, <4 x float> %1432) #9, !dbg !18
  %1434 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed294, <4 x half> %1406, <4 x float> %1433) #9, !dbg !18
  %1435 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed296, <4 x half> %1407, <4 x float> %1434) #9, !dbg !18
  %1436 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed298, <4 x half> %1408, <4 x float> %1435) #9, !dbg !18
  %1437 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed300, <4 x half> %1414, <4 x float> %1436) #9, !dbg !18
  %1438 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1415, <4 x float> %1437) #9, !dbg !18
  %1439 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed288, <4 x half> %1417, <4 x float> %1375) #9, !dbg !18
  %1440 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed290, <4 x half> %1418, <4 x float> %1439) #9, !dbg !18
  %1441 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed292, <4 x half> %1419, <4 x float> %1440) #9, !dbg !18
  %1442 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed294, <4 x half> %1420, <4 x float> %1441) #9, !dbg !18
  %1443 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed296, <4 x half> %1424, <4 x float> %1442) #9, !dbg !18
  %1444 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed298, <4 x half> %1426, <4 x float> %1443) #9, !dbg !18
  %1445 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed300, <4 x half> %1427, <4 x float> %1444) #9, !dbg !18
  %1446 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1428, <4 x float> %1445) #9, !dbg !18
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %1447, !dbg !7

1447:                                             ; preds = %1450, %1334
  %1448 = phi i64 [ %1498, %1450 ], [ %20, %1334 ], !dbg !7
  %1449 = icmp samesign ult i64 %1448, 2048, !dbg !7
  br i1 %1449, label %1450, label %1499, !dbg !7

1450:                                             ; preds = %1447
  %1451 = shl nuw nsw i64 %1448, 1, !dbg !7
  %1452 = lshr i64 %1448, 2, !dbg !7
  %1453 = lshr i64 %1448, 5, !dbg !7
  %1454 = and i64 %1452, 24, !dbg !7
  %1455 = lshr i64 %1448, 4, !dbg !7
  %1456 = lshr i64 %1448, 7, !dbg !7
  %1457 = sub nsw i64 %1455, %1456, !dbg !7
  %1458 = shl nsw i64 %1457, 2, !dbg !7
  %1459 = and i64 %1458, 4, !dbg !7
  %1460 = sub nsw i64 %1452, %1453, !dbg !7
  %1461 = and i64 %1460, 3, !dbg !7
  %1462 = shl nuw nsw i64 %1456, 3, !dbg !7
  %1463 = shl nuw nsw i64 %1448, 1, !dbg !7
  %1464 = and i64 %1463, 4, !dbg !7
  %1465 = or disjoint i64 %1462, %1464, !dbg !7
  %1466 = and i64 %1451, 2, !dbg !7
  %1467 = or disjoint i64 %1465, %1466, !dbg !7
  %1468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1454, !dbg !7
  %1469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1468, i64 %1459, !dbg !7
  %1470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1469, i64 %1461, !dbg !7
  %.idx104 = shl nuw nsw i64 %1467, 6, !dbg !7
  %1471 = getelementptr inbounds nuw i8, ptr addrspace(3) %1470, i64 %.idx104, !dbg !7
  %1472 = load half, ptr addrspace(3) %1471, align 2, !dbg !7
  %1473 = lshr i64 %1448, 2, !dbg !7
  %1474 = lshr i64 %1448, 5, !dbg !7
  %1475 = and i64 %1473, 24, !dbg !7
  %1476 = lshr i64 %1448, 4, !dbg !7
  %1477 = lshr i64 %1448, 7, !dbg !7
  %1478 = sub nsw i64 %1476, %1477, !dbg !7
  %1479 = shl nsw i64 %1478, 2, !dbg !7
  %1480 = and i64 %1479, 4, !dbg !7
  %1481 = sub nsw i64 %1473, %1474, !dbg !7
  %1482 = and i64 %1481, 3, !dbg !7
  %1483 = shl nuw nsw i64 %1477, 3, !dbg !7
  %1484 = shl nuw nsw i64 %1448, 1, !dbg !7
  %1485 = and i64 %1484, 4, !dbg !7
  %1486 = or disjoint i64 %1483, %1485, !dbg !7
  %1487 = and i64 %1451, 2, !dbg !7
  %1488 = or disjoint i64 %1487, 1, !dbg !7
  %1489 = or disjoint i64 %1486, %1488, !dbg !7
  %1490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1475, !dbg !7
  %1491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1490, i64 %1480, !dbg !7
  %1492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1491, i64 %1482, !dbg !7
  %.idx105 = shl nuw nsw i64 %1489, 6, !dbg !7
  %1493 = getelementptr inbounds nuw i8, ptr addrspace(3) %1492, i64 %.idx105, !dbg !7
  %1494 = load half, ptr addrspace(3) %1493, align 2, !dbg !7
  %1495 = insertelement <2 x half> poison, half %1472, i64 0, !dbg !7
  %1496 = insertelement <2 x half> %1495, half %1494, i64 1, !dbg !7
  %1497 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %1451, !dbg !7
  store <2 x half> %1496, ptr addrspace(3) %1497, align 2, !dbg !7
  %1498 = add nuw nsw i64 %1448, 256, !dbg !7
  br label %1447, !dbg !7

1499:                                             ; preds = %1447
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1500 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1169, !dbg !7
  %1501 = getelementptr i8, ptr addrspace(1) %1500, i64 192, !dbg !7
  %address.thread1341 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !7, !range !33, !frisk.register_lifetimes !1
  %address.local1342 = zext nneg i32 %address.thread1341 to i64, !dbg !6
  %address.local1343 = shl nuw nsw i64 %address.local1342, 1, !dbg !14
  %address.local1344 = and i64 %address.local1343, 384, !dbg !14
  %address.local1345 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1344, !dbg !7
  %address.local1346 = ptrtoint ptr addrspace(3) %address.local1345 to i64, !dbg !7
  %address.local1347 = trunc i64 %address.local1346 to i32, !dbg !7
  %1502 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1347), !dbg !7
  %1503 = zext i32 %1502 to i64, !dbg !7
  %1504 = inttoptr i64 %1503 to ptr addrspace(3), !dbg !7
  %address.local1348 = shl nuw nsw i64 %address.local1342, 3, !dbg !7
  %address.local1349 = and i64 %address.local1348, 1920, !dbg !7
  %address.local1350 = shl nuw nsw i64 %address.local1342, 1, !dbg !14
  %address.local1351 = and i64 %address.local1350, 30, !dbg !7
  %address.local1352 = or disjoint i64 %address.local1349, %address.local1351, !dbg !7
  %address.local1353 = getelementptr [2 x i8], ptr addrspace(1) %1501, i64 %address.local1352, !dbg !7
  %dma.source.bits2014 = ptrtoint ptr addrspace(1) %address.local1353 to i64, !dbg !7
  %dma.source.opaque2015 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2014) #9, !dbg !7
  %dma.source.ptr2016 = inttoptr i64 %dma.source.opaque2015 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2016, ptr addrspace(3) %1504, i32 4, i32 0, i32 0), !dbg !7
  %address.local1354 = or disjoint i64 %address.local1342, 256, !dbg !14
  %address.local1355 = shl nuw nsw i64 %address.local1354, 1, !dbg !14
  %address.local1356 = and i64 %address.local1355, 896, !dbg !14
  %address.local1357 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1356, !dbg !7
  %address.local1358 = ptrtoint ptr addrspace(3) %address.local1357 to i64, !dbg !7
  %address.local1359 = trunc i64 %address.local1358 to i32, !dbg !7
  %1505 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1359), !dbg !7
  %1506 = zext i32 %1505 to i64, !dbg !7
  %1507 = inttoptr i64 %1506 to ptr addrspace(3), !dbg !7
  %address.local1360 = shl nuw nsw i64 %address.local1354, 3, !dbg !7
  %address.local1361 = and i64 %address.local1360, 3968, !dbg !7
  %address.local1362 = shl nuw nsw i64 %address.local1354, 1, !dbg !14
  %address.local1363 = and i64 %address.local1362, 30, !dbg !7
  %address.local1364 = or disjoint i64 %address.local1361, %address.local1363, !dbg !7
  %address.local1365 = getelementptr [2 x i8], ptr addrspace(1) %1501, i64 %address.local1364, !dbg !7
  %dma.source.bits2017 = ptrtoint ptr addrspace(1) %address.local1365 to i64, !dbg !7
  %dma.source.opaque2018 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2017) #9, !dbg !7
  %dma.source.ptr2019 = inttoptr i64 %dma.source.opaque2018 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2019, ptr addrspace(3) %1507, i32 4, i32 0, i32 0), !dbg !7
  %address.local1366 = or disjoint i64 %address.local1342, 512, !dbg !14
  %address.local1367 = shl nuw nsw i64 %address.local1366, 1, !dbg !14
  %address.local1368 = and i64 %address.local1367, 1408, !dbg !14
  %address.local1369 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1368, !dbg !7
  %address.local1370 = ptrtoint ptr addrspace(3) %address.local1369 to i64, !dbg !7
  %address.local1371 = trunc i64 %address.local1370 to i32, !dbg !7
  %1508 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1371), !dbg !7
  %1509 = zext i32 %1508 to i64, !dbg !7
  %1510 = inttoptr i64 %1509 to ptr addrspace(3), !dbg !7
  %address.local1372 = shl nuw nsw i64 %address.local1366, 3, !dbg !7
  %address.local1373 = and i64 %address.local1372, 6016, !dbg !7
  %address.local1374 = shl nuw nsw i64 %address.local1366, 1, !dbg !14
  %address.local1375 = and i64 %address.local1374, 30, !dbg !7
  %address.local1376 = or disjoint i64 %address.local1373, %address.local1375, !dbg !7
  %address.local1377 = getelementptr [2 x i8], ptr addrspace(1) %1501, i64 %address.local1376, !dbg !7
  %dma.source.bits2020 = ptrtoint ptr addrspace(1) %address.local1377 to i64, !dbg !7
  %dma.source.opaque2021 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2020) #9, !dbg !7
  %dma.source.ptr2022 = inttoptr i64 %dma.source.opaque2021 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2022, ptr addrspace(3) %1510, i32 4, i32 0, i32 0), !dbg !7
  %address.local1378 = or disjoint i64 %address.local1342, 768, !dbg !14
  %address.local1379 = shl nuw nsw i64 %address.local1378, 1, !dbg !14
  %address.local1380 = and i64 %address.local1379, 1920, !dbg !14
  %address.local1381 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1380, !dbg !7
  %address.local1382 = ptrtoint ptr addrspace(3) %address.local1381 to i64, !dbg !7
  %address.local1383 = trunc i64 %address.local1382 to i32, !dbg !7
  %1511 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1383), !dbg !7
  %1512 = zext i32 %1511 to i64, !dbg !7
  %1513 = inttoptr i64 %1512 to ptr addrspace(3), !dbg !7
  %address.local1384 = shl nuw nsw i64 %address.local1378, 3, !dbg !7
  %address.local1385 = and i64 %address.local1384, 8064, !dbg !7
  %address.local1386 = shl nuw nsw i64 %address.local1378, 1, !dbg !14
  %address.local1387 = and i64 %address.local1386, 30, !dbg !7
  %address.local1388 = or disjoint i64 %address.local1385, %address.local1387, !dbg !7
  %address.local1389 = getelementptr [2 x i8], ptr addrspace(1) %1501, i64 %address.local1388, !dbg !7
  %dma.source.bits2023 = ptrtoint ptr addrspace(1) %address.local1389 to i64, !dbg !7
  %dma.source.opaque2024 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2023) #9, !dbg !7
  %dma.source.ptr2025 = inttoptr i64 %dma.source.opaque2024 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2025, ptr addrspace(3) %1513, i32 4, i32 0, i32 0), !dbg !7
  %address.local1390 = or disjoint i64 %address.local1342, 1024, !dbg !14
  %address.local1391 = shl nuw nsw i64 %address.local1390, 1, !dbg !14
  %address.local1392 = and i64 %address.local1391, 2432, !dbg !14
  %address.local1393 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1392, !dbg !7
  %address.local1394 = ptrtoint ptr addrspace(3) %address.local1393 to i64, !dbg !7
  %address.local1395 = trunc i64 %address.local1394 to i32, !dbg !7
  %1514 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1395), !dbg !7
  %1515 = zext i32 %1514 to i64, !dbg !7
  %1516 = inttoptr i64 %1515 to ptr addrspace(3), !dbg !7
  %address.local1396 = shl nuw nsw i64 %address.local1390, 3, !dbg !7
  %address.local1397 = and i64 %address.local1396, 10112, !dbg !7
  %address.local1398 = shl nuw nsw i64 %address.local1390, 1, !dbg !14
  %address.local1399 = and i64 %address.local1398, 30, !dbg !7
  %address.local1400 = or disjoint i64 %address.local1397, %address.local1399, !dbg !7
  %address.local1401 = getelementptr [2 x i8], ptr addrspace(1) %1501, i64 %address.local1400, !dbg !7
  %dma.source.bits2026 = ptrtoint ptr addrspace(1) %address.local1401 to i64, !dbg !7
  %dma.source.opaque2027 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2026) #9, !dbg !7
  %dma.source.ptr2028 = inttoptr i64 %dma.source.opaque2027 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2028, ptr addrspace(3) %1516, i32 4, i32 0, i32 0), !dbg !7
  %address.local1402 = or disjoint i64 %address.local1342, 1280, !dbg !14
  %address.local1403 = shl nuw nsw i64 %address.local1402, 1, !dbg !14
  %address.local1404 = and i64 %address.local1403, 2944, !dbg !14
  %address.local1405 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1404, !dbg !7
  %address.local1406 = ptrtoint ptr addrspace(3) %address.local1405 to i64, !dbg !7
  %address.local1407 = trunc i64 %address.local1406 to i32, !dbg !7
  %1517 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1407), !dbg !7
  %1518 = zext i32 %1517 to i64, !dbg !7
  %1519 = inttoptr i64 %1518 to ptr addrspace(3), !dbg !7
  %address.local1408 = shl nuw nsw i64 %address.local1402, 3, !dbg !7
  %address.local1409 = and i64 %address.local1408, 12160, !dbg !7
  %address.local1410 = shl nuw nsw i64 %address.local1402, 1, !dbg !14
  %address.local1411 = and i64 %address.local1410, 30, !dbg !7
  %address.local1412 = or disjoint i64 %address.local1409, %address.local1411, !dbg !7
  %address.local1413 = getelementptr [2 x i8], ptr addrspace(1) %1501, i64 %address.local1412, !dbg !7
  %dma.source.bits2029 = ptrtoint ptr addrspace(1) %address.local1413 to i64, !dbg !7
  %dma.source.opaque2030 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2029) #9, !dbg !7
  %dma.source.ptr2031 = inttoptr i64 %dma.source.opaque2030 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2031, ptr addrspace(3) %1519, i32 4, i32 0, i32 0), !dbg !7
  %address.local1414 = or disjoint i64 %address.local1342, 1536, !dbg !14
  %address.local1415 = shl nuw nsw i64 %address.local1414, 1, !dbg !14
  %address.local1416 = and i64 %address.local1415, 3456, !dbg !14
  %address.local1417 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1416, !dbg !7
  %address.local1418 = ptrtoint ptr addrspace(3) %address.local1417 to i64, !dbg !7
  %address.local1419 = trunc i64 %address.local1418 to i32, !dbg !7
  %1520 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1419), !dbg !7
  %1521 = zext i32 %1520 to i64, !dbg !7
  %1522 = inttoptr i64 %1521 to ptr addrspace(3), !dbg !7
  %address.local1420 = shl nuw nsw i64 %address.local1414, 3, !dbg !7
  %address.local1421 = and i64 %address.local1420, 14208, !dbg !7
  %address.local1422 = shl nuw nsw i64 %address.local1414, 1, !dbg !14
  %address.local1423 = and i64 %address.local1422, 30, !dbg !7
  %address.local1424 = or disjoint i64 %address.local1421, %address.local1423, !dbg !7
  %address.local1425 = getelementptr [2 x i8], ptr addrspace(1) %1501, i64 %address.local1424, !dbg !7
  %dma.source.bits2032 = ptrtoint ptr addrspace(1) %address.local1425 to i64, !dbg !7
  %dma.source.opaque2033 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2032) #9, !dbg !7
  %dma.source.ptr2034 = inttoptr i64 %dma.source.opaque2033 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2034, ptr addrspace(3) %1522, i32 4, i32 0, i32 0), !dbg !7
  %address.local1426 = or disjoint i64 %address.local1342, 1792, !dbg !14
  %address.local1427 = shl nuw nsw i64 %address.local1426, 1, !dbg !14
  %address.local1428 = and i64 %address.local1427, 3968, !dbg !14
  %address.local1429 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local1428, !dbg !7
  %address.local1430 = ptrtoint ptr addrspace(3) %address.local1429 to i64, !dbg !7
  %address.local1431 = trunc i64 %address.local1430 to i32, !dbg !7
  %1523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1431), !dbg !7
  %1524 = zext i32 %1523 to i64, !dbg !7
  %1525 = inttoptr i64 %1524 to ptr addrspace(3), !dbg !7
  %address.local1432 = shl nuw nsw i64 %address.local1426, 3, !dbg !7
  %address.local1433 = and i64 %address.local1432, 16256, !dbg !7
  %address.local1434 = shl nuw nsw i64 %address.local1426, 1, !dbg !14
  %address.local1435 = and i64 %address.local1434, 30, !dbg !7
  %address.local1436 = or disjoint i64 %address.local1433, %address.local1435, !dbg !7
  %address.local1437 = getelementptr [2 x i8], ptr addrspace(1) %1501, i64 %address.local1436, !dbg !7
  %dma.source.bits2035 = ptrtoint ptr addrspace(1) %address.local1437 to i64, !dbg !7
  %dma.source.opaque2036 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2035) #9, !dbg !7
  %dma.source.ptr2037 = inttoptr i64 %dma.source.opaque2036 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2037, ptr addrspace(3) %1525, i32 4, i32 0, i32 0), !dbg !7
  call void @llvm.amdgcn.asyncmark(), !dbg !7
  %address.local1438 = shl nuw nsw i64 %address.local1342, 3, !dbg !18
  %address.local1439 = lshr i64 %address.local1342, 4, !dbg !6
  %address.local1440 = mul nsw i64 %address.local1439, -128, !dbg !18
  %address.local1441 = add nsw i64 %address.local1438, %address.local1440, !dbg !18
  %address.local1442 = and i64 %address.local1342, 15, !dbg !8
  %address.local1443 = lshr i64 %address.local1442, 3, !dbg !18
  %address.local1444 = mul nuw nsw i64 %address.local1443, 72, !dbg !18
  %address.local1445 = add nsw i64 %address.local1441, %address.local1444, !dbg !18
  %address.local1446 = lshr i64 %address.local1442, 2, !dbg !8
  %address.local1447 = shl nuw nsw i64 %address.local1446, 5, !dbg !18
  %address.local1448 = add nsw i64 %address.local1445, %address.local1447, !dbg !18
  %address.local1449 = lshr i64 %address.local1442, 3, !dbg !18
  %address.local1450 = mul nuw nsw i64 %address.local1449, -64, !dbg !18
  %address.local1451 = add nsw i64 %address.local1448, %address.local1450, !dbg !18
  %address.local1452 = add nuw nsw i64 %address.local1442, %address.local1443, !dbg !18
  %address.local1453 = lshr i64 %address.local1452, 2, !dbg !18
  %address.local1454 = mul nsw i64 %address.local1453, -32, !dbg !18
  %address.local1455 = add i64 %address.local1451, %address.local1454, !dbg !18
  %address.local1456 = lshr i64 %address.local1342, 4, !dbg !6
  %address.local1457 = and i64 %address.local1456, 3, !dbg !6
  %address.local1458 = shl nuw nsw i64 %address.local1457, 9, !dbg !6
  %address.local1459 = add i64 %address.local1455, %address.local1458, !dbg !18
  %address.local1460 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1459, !dbg !18
  %1526 = load <8 x half>, ptr addrspace(3) %address.local1460, align 2, !dbg !18
  %1527 = extractvalue [2 x <8 x float>] %60, 0, !dbg !29
  %1528 = shufflevector <8 x float> %1527, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1529 = shufflevector <8 x half> %1526, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1530 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1529, <4 x float> %1528) #9, !dbg !18
  %address.local1461 = mul nuw nsw i64 %address.local1457, 576, !dbg !18
  %address.local1462 = add nuw nsw i64 %address.local1438, %address.local1461, !dbg !18
  %address.local1463 = add nsw i64 %address.local1462, %address.local1440, !dbg !18
  %address.local1464 = add nsw i64 %address.local1463, %address.local1444, !dbg !18
  %address.local1465 = add nsw i64 %address.local1464, %address.local1447, !dbg !18
  %address.local1466 = shl nuw nsw i32 %address.thread1341, 2, !dbg !18
  %address.local1467 = and i32 %address.local1466, 192, !dbg !18
  %address.local1468 = or disjoint i32 %address.local1467, 32, !dbg !18
  %address.local1469 = shl nuw nsw i32 %address.thread1341, 3, !dbg !18
  %address.local1470 = and i32 %address.local1469, 96, !dbg !18
  %address.local1471 = add nuw nsw i32 %address.local1468, %address.local1470, !dbg !18
  %address.local1472 = and i32 %address.local1471, 448, !dbg !18
  %address.local1473 = zext nneg i32 %address.local1472 to i64, !dbg !18
  %address.local1474 = sub nsw i64 %address.local1465, %address.local1473, !dbg !18
  %address.local1475 = add i64 %address.local1474, %address.local1454, !dbg !18
  %address.local1476 = add i64 %address.local1475, 288, !dbg !18
  %address.local1477 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1476, !dbg !18
  %1531 = load <8 x half>, ptr addrspace(3) %address.local1477, align 2, !dbg !18
  %1532 = shufflevector <8 x half> %1526, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1533 = shufflevector <8 x float> %1527, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1534 = extractvalue [2 x <8 x float>] %60, 1, !dbg !29
  %1535 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1532, <4 x float> %1530) #9, !dbg !18
  %1536 = shufflevector <8 x half> %1531, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1537 = shufflevector <8 x half> %1531, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1538 = shufflevector <8 x float> %1534, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1539 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1536, <4 x float> %1535) #9, !dbg !18
  %1540 = shufflevector <8 x float> %1534, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1541 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1537, <4 x float> %1539) #9, !dbg !18
  %address.local1478 = shl nuw nsw i32 %address.thread1341, 3, !dbg !18
  %address.local1479 = and i32 %address.local1478, 120, !dbg !18
  %address.local1480 = shl nuw nsw i32 %address.thread1341, 5, !dbg !18
  %address.local1481 = and i32 %address.local1480, 1536, !dbg !18
  %address.local1482 = or disjoint i32 %address.local1479, %address.local1481, !dbg !18
  %address.local1483 = zext nneg i32 %address.local1482 to i64, !dbg !18
  %address.local1484 = add nuw nsw i64 %address.local1444, %address.local1483, !dbg !18
  %address.local1485 = add nuw nsw i64 %address.local1484, %address.local1447, !dbg !18
  %address.local1486 = add nsw i64 %address.local1485, %address.local1450, !dbg !18
  %address.local1487 = add nsw i64 %address.local1486, %address.local1454, !dbg !18
  %address.local1488 = add i64 %address.local1487, 2048, !dbg !18
  %address.local1489 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1488, !dbg !18
  %1542 = load <8 x half>, ptr addrspace(3) %address.local1489, align 2, !dbg !18
  %address.local1490 = add nuw nsw i64 %address.local1462, 2304, !dbg !18
  %address.local1491 = add nsw i64 %address.local1490, %address.local1440, !dbg !18
  %address.local1492 = add nsw i64 %address.local1491, %address.local1444, !dbg !18
  %address.local1493 = add nsw i64 %address.local1492, %address.local1447, !dbg !18
  %address.local1494 = shl nuw nsw i32 %address.thread1341, 2, !dbg !18
  %address.local1495 = and i32 %address.local1494, 192, !dbg !18
  %address.local1496 = or disjoint i32 %address.local1495, 288, !dbg !18
  %address.local1497 = shl nuw nsw i32 %address.thread1341, 3, !dbg !18
  %address.local1498 = and i32 %address.local1497, 96, !dbg !18
  %address.local1499 = add nuw nsw i32 %address.local1496, %address.local1498, !dbg !18
  %address.local1500 = and i32 %address.local1499, 960, !dbg !18
  %address.local1501 = zext nneg i32 %address.local1500 to i64, !dbg !18
  %address.local1502 = sub i64 %address.local1493, %address.local1501, !dbg !18
  %address.local1503 = add i64 %address.local1502, %address.local1454, !dbg !18
  %address.local1504 = add i64 %address.local1503, 288, !dbg !18
  %address.local1505 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1504, !dbg !18
  %1543 = load <8 x half>, ptr addrspace(3) %address.local1505, align 2, !dbg !18
  %address.local1506 = add nsw i64 %address.local1445, %address.local1447, !dbg !18
  %address.local1507 = shl nuw nsw i64 %address.local1342, 3, !dbg !18
  %address.local1508 = and i64 %address.local1507, 64, !dbg !18
  %address.local1509 = sub nsw i64 %address.local1506, %address.local1508, !dbg !18
  %address.local1510 = add nuw nsw i64 %address.local1452, 18, !dbg !18
  %address.local1511 = lshr i64 %address.local1510, 2, !dbg !18
  %address.local1512 = mul nsw i64 %address.local1511, -32, !dbg !18
  %address.local1513 = add i64 %address.local1509, %address.local1512, !dbg !18
  %address.local1514 = add i64 %address.local1513, %address.local1458, !dbg !18
  %address.local1515 = add i64 %address.local1514, 272, !dbg !18
  %address.local1516 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1515, !dbg !18
  %1544 = load <8 x half>, ptr addrspace(3) %address.local1516, align 2, !dbg !18
  %1545 = shufflevector <8 x half> %1542, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1546 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1545, <4 x float> %1541) #9, !dbg !18
  %1547 = shufflevector <8 x half> %1542, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1548 = shufflevector <8 x half> %1543, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1549 = shufflevector <8 x half> %1543, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1550 = shufflevector <8 x half> %1544, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1551 = shufflevector <8 x half> %1544, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %address.local1517 = add nsw i64 %address.local1464, %address.local1447, !dbg !18
  %address.local1518 = shl nuw nsw i32 %address.thread1341, 2, !dbg !18
  %address.local1519 = and i32 %address.local1518, 192, !dbg !18
  %address.local1520 = add nuw nsw i32 %address.local1519, 160, !dbg !18
  %address.local1521 = shl nuw nsw i32 %address.thread1341, 3, !dbg !18
  %address.local1522 = and i32 %address.local1521, 96, !dbg !18
  %address.local1523 = add nuw nsw i32 %address.local1520, %address.local1522, !dbg !18
  %address.local1524 = and i32 %address.local1523, 960, !dbg !18
  %address.local1525 = zext nneg i32 %address.local1524 to i64, !dbg !18
  %address.local1526 = sub nsw i64 %address.local1517, %address.local1525, !dbg !18
  %address.local1527 = add i64 %address.local1526, %address.local1512, !dbg !18
  %address.local1528 = add i64 %address.local1527, 688, !dbg !18
  %address.local1529 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1528, !dbg !18
  %1552 = load <8 x half>, ptr addrspace(3) %address.local1529, align 2, !dbg !18
  %1553 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1547, <4 x float> %1546) #9, !dbg !18
  %address.local1530 = add nsw i64 %address.local1462, %address.local1440, !dbg !18
  %address.local1531 = add nsw i64 %address.local1530, %address.local1444, !dbg !18
  %address.local1532 = add nsw i64 %address.local1531, %address.local1447, !dbg !18
  %address.local1533 = lshr i64 %address.local1442, 3, !dbg !18
  %address.local1534 = add nuw nsw i64 %address.local1533, %address.local1457, !dbg !18
  %address.local1535 = shl nuw nsw i64 %address.local1534, 6, !dbg !18
  %address.local1536 = sub nsw i64 %address.local1532, %address.local1535, !dbg !18
  %address.local1537 = add i64 %address.local1536, %address.local1512, !dbg !18
  %address.local1538 = add i64 %address.local1537, 2320, !dbg !18
  %address.local1539 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1538, !dbg !18
  %1554 = load <8 x half>, ptr addrspace(3) %address.local1539, align 2, !dbg !18
  %address.local1540 = add nsw i64 %address.local1492, %address.local1447, !dbg !18
  %address.local1541 = shl nuw nsw i32 %address.thread1341, 2, !dbg !18
  %address.local1542 = and i32 %address.local1541, 192, !dbg !18
  %address.local1543 = add nuw nsw i32 %address.local1542, 416, !dbg !18
  %address.local1544 = shl nuw nsw i32 %address.thread1341, 3, !dbg !18
  %address.local1545 = and i32 %address.local1544, 96, !dbg !18
  %address.local1546 = add nuw nsw i32 %address.local1543, %address.local1545, !dbg !18
  %address.local1547 = and i32 %address.local1546, 1984, !dbg !18
  %address.local1548 = zext nneg i32 %address.local1547 to i64, !dbg !18
  %address.local1549 = sub i64 %address.local1540, %address.local1548, !dbg !18
  %address.local1550 = add i64 %address.local1549, %address.local1512, !dbg !18
  %address.local1551 = add i64 %address.local1550, 688, !dbg !18
  %address.local1552 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1551, !dbg !18
  %1555 = load <8 x half>, ptr addrspace(3) %address.local1552, align 2, !dbg !18
  %1556 = shufflevector <8 x half> %1552, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1557 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed284, <4 x half> %1548, <4 x float> %1553) #9, !dbg !18
  %1558 = shufflevector <8 x half> %1552, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1559 = shufflevector <8 x half> %1554, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1560 = shufflevector <8 x half> %1554, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1561 = shufflevector <8 x half> %1555, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1562 = shufflevector <8 x half> %1555, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1563 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed286, <4 x half> %1549, <4 x float> %1557) #9, !dbg !18
  %1564 = load <8 x half>, ptr addrspace(3) %address.local1460, align 2, !dbg !18
  %1565 = load <8 x half>, ptr addrspace(3) %address.local1477, align 2, !dbg !18
  %1566 = load <8 x half>, ptr addrspace(3) %address.local1489, align 2, !dbg !18
  %1567 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1550, <4 x float> %1533) #9, !dbg !18
  %1568 = shufflevector <8 x half> %1564, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1569 = shufflevector <8 x half> %1564, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1570 = shufflevector <8 x half> %1565, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1571 = shufflevector <8 x half> %1565, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1572 = shufflevector <8 x half> %1566, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1573 = shufflevector <8 x half> %1566, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1574 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1551, <4 x float> %1567) #9, !dbg !18
  %1575 = load <8 x half>, ptr addrspace(3) %address.local1505, align 2, !dbg !18
  %1576 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1556, <4 x float> %1574) #9, !dbg !18
  %1577 = load <8 x half>, ptr addrspace(3) %address.local1516, align 2, !dbg !18
  %1578 = load <8 x half>, ptr addrspace(3) %address.local1529, align 2, !dbg !18
  %1579 = shufflevector <8 x half> %1575, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1580 = shufflevector <8 x half> %1575, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1581 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1558, <4 x float> %1576) #9, !dbg !18
  %1582 = shufflevector <8 x half> %1577, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1583 = shufflevector <8 x half> %1577, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1584 = shufflevector <8 x half> %1578, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1585 = shufflevector <8 x half> %1578, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1586 = load <8 x half>, ptr addrspace(3) %address.local1539, align 2, !dbg !18
  %1587 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1559, <4 x float> %1581) #9, !dbg !18
  %1588 = load <8 x half>, ptr addrspace(3) %address.local1552, align 2, !dbg !18
  %1589 = shufflevector <8 x half> %1586, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1590 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1560, <4 x float> %1587) #9, !dbg !18
  %1591 = shufflevector <8 x half> %1586, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1592 = shufflevector <8 x half> %1588, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1593 = shufflevector <8 x half> %1588, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1594 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed284, <4 x half> %1561, <4 x float> %1590) #9, !dbg !18
  %1595 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed286, <4 x half> %1562, <4 x float> %1594) #9, !dbg !18
  %1596 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed288, <4 x half> %1568, <4 x float> %1538) #9, !dbg !18
  %1597 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed290, <4 x half> %1569, <4 x float> %1596) #9, !dbg !18
  %1598 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed292, <4 x half> %1570, <4 x float> %1597) #9, !dbg !18
  %1599 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed294, <4 x half> %1571, <4 x float> %1598) #9, !dbg !18
  %1600 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed296, <4 x half> %1572, <4 x float> %1599) #9, !dbg !18
  %1601 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed298, <4 x half> %1573, <4 x float> %1600) #9, !dbg !18
  %1602 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed300, <4 x half> %1579, <4 x float> %1601) #9, !dbg !18
  %1603 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1580, <4 x float> %1602) #9, !dbg !18
  %1604 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed288, <4 x half> %1582, <4 x float> %1540) #9, !dbg !18
  %1605 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed290, <4 x half> %1583, <4 x float> %1604) #9, !dbg !18
  %1606 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed292, <4 x half> %1584, <4 x float> %1605) #9, !dbg !18
  %1607 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed294, <4 x half> %1585, <4 x float> %1606) #9, !dbg !18
  %1608 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed296, <4 x half> %1589, <4 x float> %1607) #9, !dbg !18
  %1609 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed298, <4 x half> %1591, <4 x float> %1608) #9, !dbg !18
  %1610 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed300, <4 x half> %1592, <4 x float> %1609) #9, !dbg !18
  %1611 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1593, <4 x float> %1610) #9, !dbg !18
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %1612, !dbg !7

1612:                                             ; preds = %1615, %1499
  %1613 = phi i64 [ %1663, %1615 ], [ %20, %1499 ], !dbg !7
  %1614 = icmp samesign ult i64 %1613, 2048, !dbg !7
  br i1 %1614, label %1615, label %1664, !dbg !7

1615:                                             ; preds = %1612
  %1616 = shl nuw nsw i64 %1613, 1, !dbg !7
  %1617 = lshr i64 %1613, 2, !dbg !7
  %1618 = lshr i64 %1613, 5, !dbg !7
  %1619 = and i64 %1617, 24, !dbg !7
  %1620 = lshr i64 %1613, 4, !dbg !7
  %1621 = lshr i64 %1613, 7, !dbg !7
  %1622 = sub nsw i64 %1620, %1621, !dbg !7
  %1623 = shl nsw i64 %1622, 2, !dbg !7
  %1624 = and i64 %1623, 4, !dbg !7
  %1625 = sub nsw i64 %1617, %1618, !dbg !7
  %1626 = and i64 %1625, 3, !dbg !7
  %1627 = shl nuw nsw i64 %1621, 3, !dbg !7
  %1628 = shl nuw nsw i64 %1613, 1, !dbg !7
  %1629 = and i64 %1628, 4, !dbg !7
  %1630 = or disjoint i64 %1627, %1629, !dbg !7
  %1631 = and i64 %1616, 2, !dbg !7
  %1632 = or disjoint i64 %1630, %1631, !dbg !7
  %1633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1619, !dbg !7
  %1634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1633, i64 %1624, !dbg !7
  %1635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1634, i64 %1626, !dbg !7
  %.idx = shl nuw nsw i64 %1632, 6, !dbg !7
  %1636 = getelementptr inbounds nuw i8, ptr addrspace(3) %1635, i64 %.idx, !dbg !7
  %1637 = load half, ptr addrspace(3) %1636, align 2, !dbg !7
  %1638 = lshr i64 %1613, 2, !dbg !7
  %1639 = lshr i64 %1613, 5, !dbg !7
  %1640 = and i64 %1638, 24, !dbg !7
  %1641 = lshr i64 %1613, 4, !dbg !7
  %1642 = lshr i64 %1613, 7, !dbg !7
  %1643 = sub nsw i64 %1641, %1642, !dbg !7
  %1644 = shl nsw i64 %1643, 2, !dbg !7
  %1645 = and i64 %1644, 4, !dbg !7
  %1646 = sub nsw i64 %1638, %1639, !dbg !7
  %1647 = and i64 %1646, 3, !dbg !7
  %1648 = shl nuw nsw i64 %1642, 3, !dbg !7
  %1649 = shl nuw nsw i64 %1613, 1, !dbg !7
  %1650 = and i64 %1649, 4, !dbg !7
  %1651 = or disjoint i64 %1648, %1650, !dbg !7
  %1652 = and i64 %1616, 2, !dbg !7
  %1653 = or disjoint i64 %1652, 1, !dbg !7
  %1654 = or disjoint i64 %1651, %1653, !dbg !7
  %1655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1640, !dbg !7
  %1656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1655, i64 %1645, !dbg !7
  %1657 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1656, i64 %1647, !dbg !7
  %.idx103 = shl nuw nsw i64 %1654, 6, !dbg !7
  %1658 = getelementptr inbounds nuw i8, ptr addrspace(3) %1657, i64 %.idx103, !dbg !7
  %1659 = load half, ptr addrspace(3) %1658, align 2, !dbg !7
  %1660 = insertelement <2 x half> poison, half %1637, i64 0, !dbg !7
  %1661 = insertelement <2 x half> %1660, half %1659, i64 1, !dbg !7
  %1662 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %1616, !dbg !7
  store <2 x half> %1661, ptr addrspace(3) %1662, align 2, !dbg !7
  %1663 = add nuw nsw i64 %1613, 256, !dbg !7
  br label %1612, !dbg !7

1664:                                             ; preds = %1612
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %address.thread1553 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !18, !range !33, !frisk.register_lifetimes !1
  %address.local1554 = zext nneg i32 %address.thread1553 to i64, !dbg !6
  %address.local1555 = shl nuw nsw i64 %address.local1554, 3, !dbg !18
  %address.local1556 = lshr i64 %address.local1554, 4, !dbg !6
  %address.local1557 = mul nsw i64 %address.local1556, -128, !dbg !18
  %address.local1558 = add nsw i64 %address.local1555, %address.local1557, !dbg !18
  %address.local1559 = and i64 %address.local1554, 15, !dbg !8
  %address.local1560 = lshr i64 %address.local1559, 3, !dbg !18
  %address.local1561 = mul nuw nsw i64 %address.local1560, 72, !dbg !18
  %address.local1562 = add nsw i64 %address.local1558, %address.local1561, !dbg !18
  %address.local1563 = lshr i64 %address.local1559, 2, !dbg !8
  %address.local1564 = shl nuw nsw i64 %address.local1563, 5, !dbg !18
  %address.local1565 = add nsw i64 %address.local1562, %address.local1564, !dbg !18
  %address.local1566 = lshr i64 %address.local1559, 3, !dbg !18
  %address.local1567 = mul nuw nsw i64 %address.local1566, -64, !dbg !18
  %address.local1568 = add nsw i64 %address.local1565, %address.local1567, !dbg !18
  %address.local1569 = add nuw nsw i64 %address.local1559, %address.local1560, !dbg !18
  %address.local1570 = lshr i64 %address.local1569, 2, !dbg !18
  %address.local1571 = mul nsw i64 %address.local1570, -32, !dbg !18
  %address.local1572 = add i64 %address.local1568, %address.local1571, !dbg !18
  %address.local1573 = lshr i64 %address.local1554, 4, !dbg !6
  %address.local1574 = and i64 %address.local1573, 3, !dbg !6
  %address.local1575 = shl nuw nsw i64 %address.local1574, 9, !dbg !6
  %address.local1576 = add i64 %address.local1572, %address.local1575, !dbg !18
  %address.local1577 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1576, !dbg !18
  %1665 = load <8 x half>, ptr addrspace(3) %address.local1577, align 2, !dbg !18
  %1666 = extractvalue [2 x <8 x float>] %61, 0, !dbg !29
  %1667 = shufflevector <8 x float> %1666, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1668 = shufflevector <8 x half> %1665, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1669 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1668, <4 x float> %1667) #9, !dbg !18
  %address.local1578 = mul nuw nsw i64 %address.local1574, 576, !dbg !18
  %address.local1579 = add nuw nsw i64 %address.local1555, %address.local1578, !dbg !18
  %address.local1580 = add nsw i64 %address.local1579, %address.local1557, !dbg !18
  %address.local1581 = add nsw i64 %address.local1580, %address.local1561, !dbg !18
  %address.local1582 = add nsw i64 %address.local1581, %address.local1564, !dbg !18
  %address.local1583 = shl nuw nsw i32 %address.thread1553, 2, !dbg !18
  %address.local1584 = and i32 %address.local1583, 192, !dbg !18
  %address.local1585 = or disjoint i32 %address.local1584, 32, !dbg !18
  %address.local1586 = shl nuw nsw i32 %address.thread1553, 3, !dbg !18
  %address.local1587 = and i32 %address.local1586, 96, !dbg !18
  %address.local1588 = add nuw nsw i32 %address.local1585, %address.local1587, !dbg !18
  %address.local1589 = and i32 %address.local1588, 448, !dbg !18
  %address.local1590 = zext nneg i32 %address.local1589 to i64, !dbg !18
  %address.local1591 = sub nsw i64 %address.local1582, %address.local1590, !dbg !18
  %address.local1592 = add i64 %address.local1591, %address.local1571, !dbg !18
  %address.local1593 = add i64 %address.local1592, 288, !dbg !18
  %address.local1594 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1593, !dbg !18
  %1670 = load <8 x half>, ptr addrspace(3) %address.local1594, align 2, !dbg !18
  %1671 = shufflevector <8 x half> %1665, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1672 = shufflevector <8 x float> %1666, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1673 = extractvalue [2 x <8 x float>] %61, 1, !dbg !29
  %1674 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1671, <4 x float> %1669) #9, !dbg !18
  %1675 = shufflevector <8 x half> %1670, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1676 = shufflevector <8 x half> %1670, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1677 = shufflevector <8 x float> %1673, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1678 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1675, <4 x float> %1674) #9, !dbg !18
  %1679 = shufflevector <8 x float> %1673, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1680 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1676, <4 x float> %1678) #9, !dbg !18
  %address.local1595 = shl nuw nsw i32 %address.thread1553, 3, !dbg !18
  %address.local1596 = and i32 %address.local1595, 120, !dbg !18
  %address.local1597 = shl nuw nsw i32 %address.thread1553, 5, !dbg !18
  %address.local1598 = and i32 %address.local1597, 1536, !dbg !18
  %address.local1599 = or disjoint i32 %address.local1596, %address.local1598, !dbg !18
  %address.local1600 = zext nneg i32 %address.local1599 to i64, !dbg !18
  %address.local1601 = add nuw nsw i64 %address.local1561, %address.local1600, !dbg !18
  %address.local1602 = add nuw nsw i64 %address.local1601, %address.local1564, !dbg !18
  %address.local1603 = add nsw i64 %address.local1602, %address.local1567, !dbg !18
  %address.local1604 = add nsw i64 %address.local1603, %address.local1571, !dbg !18
  %address.local1605 = add i64 %address.local1604, 2048, !dbg !18
  %address.local1606 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1605, !dbg !18
  %1681 = load <8 x half>, ptr addrspace(3) %address.local1606, align 2, !dbg !18
  %address.local1607 = add nuw nsw i64 %address.local1579, 2304, !dbg !18
  %address.local1608 = add nsw i64 %address.local1607, %address.local1557, !dbg !18
  %address.local1609 = add nsw i64 %address.local1608, %address.local1561, !dbg !18
  %address.local1610 = add nsw i64 %address.local1609, %address.local1564, !dbg !18
  %address.local1611 = shl nuw nsw i32 %address.thread1553, 2, !dbg !18
  %address.local1612 = and i32 %address.local1611, 192, !dbg !18
  %address.local1613 = or disjoint i32 %address.local1612, 288, !dbg !18
  %address.local1614 = shl nuw nsw i32 %address.thread1553, 3, !dbg !18
  %address.local1615 = and i32 %address.local1614, 96, !dbg !18
  %address.local1616 = add nuw nsw i32 %address.local1613, %address.local1615, !dbg !18
  %address.local1617 = and i32 %address.local1616, 960, !dbg !18
  %address.local1618 = zext nneg i32 %address.local1617 to i64, !dbg !18
  %address.local1619 = sub i64 %address.local1610, %address.local1618, !dbg !18
  %address.local1620 = add i64 %address.local1619, %address.local1571, !dbg !18
  %address.local1621 = add i64 %address.local1620, 288, !dbg !18
  %address.local1622 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1621, !dbg !18
  %1682 = load <8 x half>, ptr addrspace(3) %address.local1622, align 2, !dbg !18
  %address.local1623 = add nsw i64 %address.local1562, %address.local1564, !dbg !18
  %address.local1624 = shl nuw nsw i64 %address.local1554, 3, !dbg !18
  %address.local1625 = and i64 %address.local1624, 64, !dbg !18
  %address.local1626 = sub nsw i64 %address.local1623, %address.local1625, !dbg !18
  %address.local1627 = add nuw nsw i64 %address.local1569, 18, !dbg !18
  %address.local1628 = lshr i64 %address.local1627, 2, !dbg !18
  %address.local1629 = mul nsw i64 %address.local1628, -32, !dbg !18
  %address.local1630 = add i64 %address.local1626, %address.local1629, !dbg !18
  %address.local1631 = add i64 %address.local1630, %address.local1575, !dbg !18
  %address.local1632 = add i64 %address.local1631, 272, !dbg !18
  %address.local1633 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1632, !dbg !18
  %1683 = load <8 x half>, ptr addrspace(3) %address.local1633, align 2, !dbg !18
  %1684 = shufflevector <8 x half> %1681, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1685 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1684, <4 x float> %1680) #9, !dbg !18
  %1686 = shufflevector <8 x half> %1681, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1687 = shufflevector <8 x half> %1682, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1688 = shufflevector <8 x half> %1682, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1689 = shufflevector <8 x half> %1683, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1690 = shufflevector <8 x half> %1683, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %address.local1634 = add nsw i64 %address.local1581, %address.local1564, !dbg !18
  %address.local1635 = shl nuw nsw i32 %address.thread1553, 2, !dbg !18
  %address.local1636 = and i32 %address.local1635, 192, !dbg !18
  %address.local1637 = add nuw nsw i32 %address.local1636, 160, !dbg !18
  %address.local1638 = shl nuw nsw i32 %address.thread1553, 3, !dbg !18
  %address.local1639 = and i32 %address.local1638, 96, !dbg !18
  %address.local1640 = add nuw nsw i32 %address.local1637, %address.local1639, !dbg !18
  %address.local1641 = and i32 %address.local1640, 960, !dbg !18
  %address.local1642 = zext nneg i32 %address.local1641 to i64, !dbg !18
  %address.local1643 = sub nsw i64 %address.local1634, %address.local1642, !dbg !18
  %address.local1644 = add i64 %address.local1643, %address.local1629, !dbg !18
  %address.local1645 = add i64 %address.local1644, 688, !dbg !18
  %address.local1646 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1645, !dbg !18
  %1691 = load <8 x half>, ptr addrspace(3) %address.local1646, align 2, !dbg !18
  %1692 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1686, <4 x float> %1685) #9, !dbg !18
  %address.local1647 = add nsw i64 %address.local1579, %address.local1557, !dbg !18
  %address.local1648 = add nsw i64 %address.local1647, %address.local1561, !dbg !18
  %address.local1649 = add nsw i64 %address.local1648, %address.local1564, !dbg !18
  %address.local1650 = lshr i64 %address.local1559, 3, !dbg !18
  %address.local1651 = add nuw nsw i64 %address.local1650, %address.local1574, !dbg !18
  %address.local1652 = shl nuw nsw i64 %address.local1651, 6, !dbg !18
  %address.local1653 = sub nsw i64 %address.local1649, %address.local1652, !dbg !18
  %address.local1654 = add i64 %address.local1653, %address.local1629, !dbg !18
  %address.local1655 = add i64 %address.local1654, 2320, !dbg !18
  %address.local1656 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1655, !dbg !18
  %1693 = load <8 x half>, ptr addrspace(3) %address.local1656, align 2, !dbg !18
  %address.local1657 = add nsw i64 %address.local1609, %address.local1564, !dbg !18
  %address.local1658 = shl nuw nsw i32 %address.thread1553, 2, !dbg !18
  %address.local1659 = and i32 %address.local1658, 192, !dbg !18
  %address.local1660 = add nuw nsw i32 %address.local1659, 416, !dbg !18
  %address.local1661 = shl nuw nsw i32 %address.thread1553, 3, !dbg !18
  %address.local1662 = and i32 %address.local1661, 96, !dbg !18
  %address.local1663 = add nuw nsw i32 %address.local1660, %address.local1662, !dbg !18
  %address.local1664 = and i32 %address.local1663, 1984, !dbg !18
  %address.local1665 = zext nneg i32 %address.local1664 to i64, !dbg !18
  %address.local1666 = sub i64 %address.local1657, %address.local1665, !dbg !18
  %address.local1667 = add i64 %address.local1666, %address.local1629, !dbg !18
  %address.local1668 = add i64 %address.local1667, 688, !dbg !18
  %address.local1669 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1668, !dbg !18
  %1694 = load <8 x half>, ptr addrspace(3) %address.local1669, align 2, !dbg !18
  %1695 = shufflevector <8 x half> %1691, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1696 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed284, <4 x half> %1687, <4 x float> %1692) #9, !dbg !18
  %1697 = shufflevector <8 x half> %1691, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1698 = shufflevector <8 x half> %1693, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1699 = shufflevector <8 x half> %1693, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1700 = shufflevector <8 x half> %1694, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1701 = shufflevector <8 x half> %1694, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1702 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed286, <4 x half> %1688, <4 x float> %1696) #9, !dbg !18
  %1703 = load <8 x half>, ptr addrspace(3) %address.local1577, align 2, !dbg !18
  %1704 = load <8 x half>, ptr addrspace(3) %address.local1594, align 2, !dbg !18
  %1705 = load <8 x half>, ptr addrspace(3) %address.local1606, align 2, !dbg !18
  %1706 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1689, <4 x float> %1672) #9, !dbg !18
  %1707 = shufflevector <8 x half> %1703, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1708 = shufflevector <8 x half> %1703, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1709 = shufflevector <8 x half> %1704, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1710 = shufflevector <8 x half> %1704, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1711 = shufflevector <8 x half> %1705, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1712 = shufflevector <8 x half> %1705, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1713 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1690, <4 x float> %1706) #9, !dbg !18
  %1714 = load <8 x half>, ptr addrspace(3) %address.local1622, align 2, !dbg !18
  %1715 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1695, <4 x float> %1713) #9, !dbg !18
  %1716 = load <8 x half>, ptr addrspace(3) %address.local1633, align 2, !dbg !18
  %1717 = load <8 x half>, ptr addrspace(3) %address.local1646, align 2, !dbg !18
  %1718 = shufflevector <8 x half> %1714, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1719 = shufflevector <8 x half> %1714, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1720 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed278, <4 x half> %1697, <4 x float> %1715) #9, !dbg !18
  %1721 = shufflevector <8 x half> %1716, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1722 = shufflevector <8 x half> %1716, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1723 = shufflevector <8 x half> %1717, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1724 = shufflevector <8 x half> %1717, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1725 = load <8 x half>, ptr addrspace(3) %address.local1656, align 2, !dbg !18
  %1726 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed280, <4 x half> %1698, <4 x float> %1720) #9, !dbg !18
  %1727 = load <8 x half>, ptr addrspace(3) %address.local1669, align 2, !dbg !18
  %1728 = shufflevector <4 x float> %1233, <4 x float> %1265, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1729 = insertvalue [2 x <8 x float>] poison, <8 x float> %1728, 0, !dbg !16
  %1730 = shufflevector <4 x float> %1273, <4 x float> %1281, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1731 = insertvalue [2 x <8 x float>] %1729, <8 x float> %1730, 1, !dbg !16
  %1732 = insertelement <1 x float> poison, float %1101, i64 0, !dbg !16
  %1733 = insertvalue [2 x <1 x float>] poison, <1 x float> %1732, 0, !dbg !16
  %1734 = insertelement <1 x float> poison, float %1112, i64 0, !dbg !16
  %1735 = insertvalue [2 x <1 x float>] %1733, <1 x float> %1734, 1, !dbg !16
  %1736 = shufflevector <8 x half> %1725, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1737 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed282, <4 x half> %1699, <4 x float> %1726) #9, !dbg !18
  %1738 = shufflevector <8 x half> %1725, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1739 = shufflevector <4 x float> %1398, <4 x float> %1430, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1740 = insertvalue [2 x <8 x float>] poison, <8 x float> %1739, 0, !dbg !16
  %1741 = shufflevector <4 x float> %1438, <4 x float> %1446, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1742 = insertvalue [2 x <8 x float>] %1740, <8 x float> %1741, 1, !dbg !16
  %1743 = shufflevector <8 x half> %1727, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1744 = shufflevector <8 x half> %1727, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1745 = shufflevector <4 x float> %1563, <4 x float> %1595, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1746 = insertvalue [2 x <8 x float>] poison, <8 x float> %1745, 0, !dbg !16
  %1747 = shufflevector <4 x float> %1603, <4 x float> %1611, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1748 = insertvalue [2 x <8 x float>] %1746, <8 x float> %1747, 1, !dbg !16
  %1749 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed284, <4 x half> %1700, <4 x float> %1737) #9, !dbg !18
  %1750 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed286, <4 x half> %1701, <4 x float> %1749) #9, !dbg !18
  %1751 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed288, <4 x half> %1707, <4 x float> %1677) #9, !dbg !18
  %1752 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed290, <4 x half> %1708, <4 x float> %1751) #9, !dbg !18
  %1753 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed292, <4 x half> %1709, <4 x float> %1752) #9, !dbg !18
  %1754 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed294, <4 x half> %1710, <4 x float> %1753) #9, !dbg !18
  %1755 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed296, <4 x half> %1711, <4 x float> %1754) #9, !dbg !18
  %1756 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed298, <4 x half> %1712, <4 x float> %1755) #9, !dbg !18
  %1757 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed300, <4 x half> %1718, <4 x float> %1756) #9, !dbg !18
  %1758 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1719, <4 x float> %1757) #9, !dbg !18
  %1759 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed288, <4 x half> %1721, <4 x float> %1679) #9, !dbg !18
  %1760 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed290, <4 x half> %1722, <4 x float> %1759) #9, !dbg !18
  %1761 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed292, <4 x half> %1723, <4 x float> %1760) #9, !dbg !18
  %1762 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed294, <4 x half> %1724, <4 x float> %1761) #9, !dbg !18
  %1763 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed296, <4 x half> %1736, <4 x float> %1762) #9, !dbg !18
  %1764 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed298, <4 x half> %1738, <4 x float> %1763) #9, !dbg !18
  %1765 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed300, <4 x half> %1743, <4 x float> %1764) #9, !dbg !18
  %1766 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1744, <4 x float> %1765) #9, !dbg !18
  %1767 = shufflevector <4 x float> %1702, <4 x float> %1750, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1768 = insertvalue [2 x <8 x float>] poison, <8 x float> %1767, 0, !dbg !16
  %1769 = shufflevector <4 x float> %1758, <4 x float> %1766, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1770 = insertvalue [2 x <8 x float>] %1768, <8 x float> %1769, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1771 = add nuw nsw i64 %56, 1, !dbg !16
  br label %55, !dbg !16

1772:                                             ; preds = %55
  %1773 = and i32 %19, 16, !dbg !16
  %.not34 = icmp eq i32 %1773, 0, !dbg !16
  %1774 = and i32 %19, 32, !dbg !16
  %.not35 = icmp eq i32 %1774, 0, !dbg !16
  %1775 = extractvalue [2 x <8 x float>] %61, 0, !dbg !16
  %1776 = extractelement <8 x float> %1775, i64 0, !dbg !16
  %1777 = extractelement <8 x float> %1775, i64 1, !dbg !16
  %1778 = extractelement <8 x float> %1775, i64 2, !dbg !16
  %1779 = extractelement <8 x float> %1775, i64 3, !dbg !16
  %1780 = select i1 %.not34, float %1777, float %1776, !dbg !16
  %1781 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %1782 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1781), !dbg !16
  %1783 = add nuw nsw i32 %1782, 64, !dbg !16
  %1784 = xor i32 %1782, 16, !dbg !16
  %1785 = and i32 %1783, 192, !dbg !16
  %1786 = icmp samesign ult i32 %1784, %1785, !dbg !16
  %1787 = select i1 %1786, i32 %1784, i32 %1782, !dbg !16
  %1788 = shl nuw nsw i32 %1787, 2, !dbg !16
  %1789 = bitcast float %1780 to i32, !dbg !16
  %1790 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1789), !dbg !16
  %1791 = bitcast i32 %1790 to float, !dbg !16
  %1792 = select i1 %.not34, float %1776, float %1791, !dbg !16
  %1793 = select i1 %.not34, float %1791, float %1777, !dbg !16
  %1794 = select i1 %.not34, float %1779, float %1778, !dbg !16
  %1795 = bitcast float %1794 to i32, !dbg !16
  %1796 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1795), !dbg !16
  %1797 = bitcast i32 %1796 to float, !dbg !16
  %1798 = select i1 %.not34, float %1778, float %1797, !dbg !16
  %1799 = select i1 %.not34, float %1797, float %1779, !dbg !16
  %1800 = select i1 %.not35, float %1798, float %1792, !dbg !16
  %1801 = xor i32 %1782, 32, !dbg !16
  %1802 = icmp samesign ult i32 %1801, %1785, !dbg !16
  %1803 = select i1 %1802, i32 %1801, i32 %1782, !dbg !16
  %1804 = shl nuw nsw i32 %1803, 2, !dbg !16
  %1805 = bitcast float %1800 to i32, !dbg !16
  %1806 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1805), !dbg !16
  %1807 = bitcast i32 %1806 to float, !dbg !16
  %1808 = select i1 %.not35, float %1792, float %1807, !dbg !16
  %1809 = select i1 %.not35, float %1807, float %1798, !dbg !16
  %1810 = select i1 %.not35, float %1799, float %1793, !dbg !16
  %1811 = bitcast float %1810 to i32, !dbg !16
  %1812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1811), !dbg !16
  %1813 = bitcast i32 %1812 to float, !dbg !16
  %1814 = select i1 %.not35, float %1793, float %1813, !dbg !16
  %1815 = select i1 %.not35, float %1813, float %1799, !dbg !16
  %1816 = extractvalue [2 x <8 x float>] %60, 0, !dbg !16
  %1817 = extractelement <8 x float> %1816, i64 0, !dbg !16
  %1818 = extractelement <8 x float> %1816, i64 1, !dbg !16
  %1819 = extractelement <8 x float> %1816, i64 2, !dbg !16
  %1820 = extractelement <8 x float> %1816, i64 3, !dbg !16
  %1821 = select i1 %.not34, float %1818, float %1817, !dbg !16
  %1822 = bitcast float %1821 to i32, !dbg !16
  %1823 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1822), !dbg !16
  %1824 = bitcast i32 %1823 to float, !dbg !16
  %1825 = select i1 %.not34, float %1817, float %1824, !dbg !16
  %1826 = select i1 %.not34, float %1824, float %1818, !dbg !16
  %1827 = select i1 %.not34, float %1820, float %1819, !dbg !16
  %1828 = bitcast float %1827 to i32, !dbg !16
  %1829 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1828), !dbg !16
  %1830 = bitcast i32 %1829 to float, !dbg !16
  %1831 = select i1 %.not34, float %1819, float %1830, !dbg !16
  %1832 = select i1 %.not34, float %1830, float %1820, !dbg !16
  %1833 = select i1 %.not35, float %1831, float %1825, !dbg !16
  %1834 = bitcast float %1833 to i32, !dbg !16
  %1835 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1834), !dbg !16
  %1836 = bitcast i32 %1835 to float, !dbg !16
  %1837 = select i1 %.not35, float %1825, float %1836, !dbg !16
  %1838 = select i1 %.not35, float %1836, float %1831, !dbg !16
  %1839 = select i1 %.not35, float %1832, float %1826, !dbg !16
  %1840 = bitcast float %1839 to i32, !dbg !16
  %1841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1840), !dbg !16
  %1842 = bitcast i32 %1841 to float, !dbg !16
  %1843 = select i1 %.not35, float %1826, float %1842, !dbg !16
  %1844 = select i1 %.not35, float %1842, float %1832, !dbg !16
  %1845 = extractvalue [2 x <8 x float>] %59, 0, !dbg !16
  %1846 = extractelement <8 x float> %1845, i64 0, !dbg !16
  %1847 = extractelement <8 x float> %1845, i64 1, !dbg !16
  %1848 = extractelement <8 x float> %1845, i64 2, !dbg !16
  %1849 = extractelement <8 x float> %1845, i64 3, !dbg !16
  %1850 = select i1 %.not34, float %1847, float %1846, !dbg !16
  %1851 = bitcast float %1850 to i32, !dbg !16
  %1852 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1851), !dbg !16
  %1853 = bitcast i32 %1852 to float, !dbg !16
  %1854 = select i1 %.not34, float %1846, float %1853, !dbg !16
  %1855 = select i1 %.not34, float %1853, float %1847, !dbg !16
  %1856 = select i1 %.not34, float %1849, float %1848, !dbg !16
  %1857 = bitcast float %1856 to i32, !dbg !16
  %1858 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1857), !dbg !16
  %1859 = bitcast i32 %1858 to float, !dbg !16
  %1860 = select i1 %.not34, float %1848, float %1859, !dbg !16
  %1861 = select i1 %.not34, float %1859, float %1849, !dbg !16
  %1862 = select i1 %.not35, float %1860, float %1854, !dbg !16
  %1863 = bitcast float %1862 to i32, !dbg !16
  %1864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1863), !dbg !16
  %1865 = bitcast i32 %1864 to float, !dbg !16
  %1866 = select i1 %.not35, float %1854, float %1865, !dbg !16
  %1867 = select i1 %.not35, float %1865, float %1860, !dbg !16
  %1868 = select i1 %.not35, float %1861, float %1855, !dbg !16
  %1869 = bitcast float %1868 to i32, !dbg !16
  %1870 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1869), !dbg !16
  %1871 = bitcast i32 %1870 to float, !dbg !16
  %1872 = select i1 %.not35, float %1855, float %1871, !dbg !16
  %1873 = select i1 %.not35, float %1871, float %1861, !dbg !16
  %1874 = extractvalue [2 x <8 x float>] %57, 0, !dbg !16
  %1875 = extractelement <8 x float> %1874, i64 0, !dbg !16
  %1876 = extractelement <8 x float> %1874, i64 1, !dbg !16
  %1877 = extractelement <8 x float> %1874, i64 2, !dbg !16
  %1878 = extractelement <8 x float> %1874, i64 3, !dbg !16
  %1879 = select i1 %.not34, float %1876, float %1875, !dbg !16
  %1880 = bitcast float %1879 to i32, !dbg !16
  %1881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1880), !dbg !16
  %1882 = bitcast i32 %1881 to float, !dbg !16
  %1883 = select i1 %.not34, float %1875, float %1882, !dbg !16
  %1884 = select i1 %.not34, float %1882, float %1876, !dbg !16
  %1885 = select i1 %.not34, float %1878, float %1877, !dbg !16
  %1886 = bitcast float %1885 to i32, !dbg !16
  %1887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1886), !dbg !16
  %1888 = bitcast i32 %1887 to float, !dbg !16
  %1889 = select i1 %.not34, float %1877, float %1888, !dbg !16
  %1890 = select i1 %.not34, float %1888, float %1878, !dbg !16
  %1891 = select i1 %.not35, float %1889, float %1883, !dbg !16
  %1892 = bitcast float %1891 to i32, !dbg !16
  %1893 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1892), !dbg !16
  %1894 = select i1 %.not35, float %1890, float %1884, !dbg !16
  %1895 = bitcast float %1894 to i32, !dbg !16
  %1896 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1895), !dbg !16
  %1897 = extractvalue [2 x <1 x float>] %58, 0, !dbg !19
  %1898 = shufflevector <1 x float> %1897, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1899 = extractelement <8 x float> %1775, i64 4, !dbg !16
  %1900 = extractelement <8 x float> %1775, i64 5, !dbg !16
  %1901 = extractelement <8 x float> %1775, i64 6, !dbg !16
  %1902 = extractelement <8 x float> %1775, i64 7, !dbg !16
  %1903 = select i1 %.not34, float %1900, float %1899, !dbg !16
  %1904 = bitcast float %1903 to i32, !dbg !16
  %1905 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1904), !dbg !16
  %1906 = bitcast i32 %1905 to float, !dbg !16
  %1907 = select i1 %.not34, float %1899, float %1906, !dbg !16
  %1908 = select i1 %.not34, float %1906, float %1900, !dbg !16
  %1909 = select i1 %.not34, float %1902, float %1901, !dbg !16
  %1910 = bitcast float %1909 to i32, !dbg !16
  %1911 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1910), !dbg !16
  %1912 = bitcast i32 %1911 to float, !dbg !16
  %1913 = select i1 %.not34, float %1901, float %1912, !dbg !16
  %1914 = select i1 %.not34, float %1912, float %1902, !dbg !16
  %1915 = select i1 %.not35, float %1913, float %1907, !dbg !16
  %1916 = bitcast float %1915 to i32, !dbg !16
  %1917 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1916), !dbg !16
  %1918 = bitcast i32 %1917 to float, !dbg !16
  %1919 = select i1 %.not35, float %1907, float %1918, !dbg !16
  %1920 = select i1 %.not35, float %1918, float %1913, !dbg !16
  %1921 = select i1 %.not35, float %1914, float %1908, !dbg !16
  %1922 = bitcast float %1921 to i32, !dbg !16
  %1923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1922), !dbg !16
  %1924 = bitcast i32 %1923 to float, !dbg !16
  %1925 = select i1 %.not35, float %1908, float %1924, !dbg !16
  %1926 = select i1 %.not35, float %1924, float %1914, !dbg !16
  %1927 = extractelement <8 x float> %1816, i64 4, !dbg !16
  %1928 = extractelement <8 x float> %1816, i64 5, !dbg !16
  %1929 = extractelement <8 x float> %1816, i64 6, !dbg !16
  %1930 = extractelement <8 x float> %1816, i64 7, !dbg !16
  %1931 = select i1 %.not34, float %1928, float %1927, !dbg !16
  %1932 = bitcast float %1931 to i32, !dbg !16
  %1933 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1932), !dbg !16
  %1934 = bitcast i32 %1933 to float, !dbg !16
  %1935 = select i1 %.not34, float %1927, float %1934, !dbg !16
  %1936 = select i1 %.not34, float %1934, float %1928, !dbg !16
  %1937 = select i1 %.not34, float %1930, float %1929, !dbg !16
  %1938 = bitcast float %1937 to i32, !dbg !16
  %1939 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1938), !dbg !16
  %1940 = bitcast i32 %1939 to float, !dbg !16
  %1941 = select i1 %.not34, float %1929, float %1940, !dbg !16
  %1942 = select i1 %.not34, float %1940, float %1930, !dbg !16
  %1943 = select i1 %.not35, float %1941, float %1935, !dbg !16
  %1944 = bitcast float %1943 to i32, !dbg !16
  %1945 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1944), !dbg !16
  %1946 = bitcast i32 %1945 to float, !dbg !16
  %1947 = select i1 %.not35, float %1935, float %1946, !dbg !16
  %1948 = select i1 %.not35, float %1946, float %1941, !dbg !16
  %1949 = select i1 %.not35, float %1942, float %1936, !dbg !16
  %1950 = bitcast float %1949 to i32, !dbg !16
  %1951 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1950), !dbg !16
  %1952 = bitcast i32 %1951 to float, !dbg !16
  %1953 = select i1 %.not35, float %1936, float %1952, !dbg !16
  %1954 = select i1 %.not35, float %1952, float %1942, !dbg !16
  %1955 = extractelement <8 x float> %1845, i64 4, !dbg !16
  %1956 = extractelement <8 x float> %1845, i64 5, !dbg !16
  %1957 = extractelement <8 x float> %1845, i64 6, !dbg !16
  %1958 = extractelement <8 x float> %1845, i64 7, !dbg !16
  %1959 = select i1 %.not34, float %1956, float %1955, !dbg !16
  %1960 = bitcast float %1959 to i32, !dbg !16
  %1961 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1960), !dbg !16
  %1962 = bitcast i32 %1961 to float, !dbg !16
  %1963 = select i1 %.not34, float %1955, float %1962, !dbg !16
  %1964 = select i1 %.not34, float %1962, float %1956, !dbg !16
  %1965 = select i1 %.not34, float %1958, float %1957, !dbg !16
  %1966 = bitcast float %1965 to i32, !dbg !16
  %1967 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1966), !dbg !16
  %1968 = bitcast i32 %1967 to float, !dbg !16
  %1969 = select i1 %.not34, float %1957, float %1968, !dbg !16
  %1970 = select i1 %.not34, float %1968, float %1958, !dbg !16
  %1971 = select i1 %.not35, float %1969, float %1963, !dbg !16
  %1972 = bitcast float %1971 to i32, !dbg !16
  %1973 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1972), !dbg !16
  %1974 = bitcast i32 %1973 to float, !dbg !16
  %1975 = select i1 %.not35, float %1963, float %1974, !dbg !16
  %1976 = select i1 %.not35, float %1974, float %1969, !dbg !16
  %1977 = select i1 %.not35, float %1970, float %1964, !dbg !16
  %1978 = bitcast float %1977 to i32, !dbg !16
  %1979 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1978), !dbg !16
  %1980 = bitcast i32 %1979 to float, !dbg !16
  %1981 = select i1 %.not35, float %1964, float %1980, !dbg !16
  %1982 = select i1 %.not35, float %1980, float %1970, !dbg !16
  %1983 = extractelement <8 x float> %1874, i64 4, !dbg !16
  %1984 = extractelement <8 x float> %1874, i64 5, !dbg !16
  %1985 = extractelement <8 x float> %1874, i64 6, !dbg !16
  %1986 = extractelement <8 x float> %1874, i64 7, !dbg !16
  %1987 = select i1 %.not34, float %1984, float %1983, !dbg !16
  %1988 = bitcast float %1987 to i32, !dbg !16
  %1989 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1988), !dbg !16
  %1990 = bitcast i32 %1989 to float, !dbg !16
  %1991 = select i1 %.not34, float %1983, float %1990, !dbg !16
  %1992 = select i1 %.not34, float %1990, float %1984, !dbg !16
  %1993 = select i1 %.not34, float %1986, float %1985, !dbg !16
  %1994 = bitcast float %1993 to i32, !dbg !16
  %1995 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1994), !dbg !16
  %1996 = bitcast i32 %1995 to float, !dbg !16
  %1997 = select i1 %.not34, float %1985, float %1996, !dbg !16
  %1998 = select i1 %.not34, float %1996, float %1986, !dbg !16
  %1999 = select i1 %.not35, float %1997, float %1991, !dbg !16
  %2000 = bitcast float %1999 to i32, !dbg !16
  %2001 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2000), !dbg !16
  %2002 = select i1 %.not35, float %1998, float %1992, !dbg !16
  %2003 = bitcast float %2002 to i32, !dbg !16
  %2004 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2003), !dbg !16
  %2005 = extractvalue [2 x <8 x float>] %61, 1, !dbg !16
  %2006 = extractelement <8 x float> %2005, i64 0, !dbg !16
  %2007 = extractelement <8 x float> %2005, i64 1, !dbg !16
  %2008 = extractelement <8 x float> %2005, i64 2, !dbg !16
  %2009 = extractelement <8 x float> %2005, i64 3, !dbg !16
  %2010 = select i1 %.not34, float %2007, float %2006, !dbg !16
  %2011 = bitcast float %2010 to i32, !dbg !16
  %2012 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2011), !dbg !16
  %2013 = bitcast i32 %2012 to float, !dbg !16
  %2014 = select i1 %.not34, float %2006, float %2013, !dbg !16
  %2015 = select i1 %.not34, float %2013, float %2007, !dbg !16
  %2016 = select i1 %.not34, float %2009, float %2008, !dbg !16
  %2017 = bitcast float %2016 to i32, !dbg !16
  %2018 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2017), !dbg !16
  %2019 = bitcast i32 %2018 to float, !dbg !16
  %2020 = select i1 %.not34, float %2008, float %2019, !dbg !16
  %2021 = select i1 %.not34, float %2019, float %2009, !dbg !16
  %2022 = select i1 %.not35, float %2020, float %2014, !dbg !16
  %2023 = bitcast float %2022 to i32, !dbg !16
  %2024 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2023), !dbg !16
  %2025 = bitcast i32 %2024 to float, !dbg !16
  %2026 = select i1 %.not35, float %2014, float %2025, !dbg !16
  %2027 = select i1 %.not35, float %2025, float %2020, !dbg !16
  %2028 = select i1 %.not35, float %2021, float %2015, !dbg !16
  %2029 = bitcast float %2028 to i32, !dbg !16
  %2030 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2029), !dbg !16
  %2031 = bitcast i32 %2030 to float, !dbg !16
  %2032 = select i1 %.not35, float %2015, float %2031, !dbg !16
  %2033 = select i1 %.not35, float %2031, float %2021, !dbg !16
  %2034 = extractvalue [2 x <8 x float>] %60, 1, !dbg !16
  %2035 = extractelement <8 x float> %2034, i64 0, !dbg !16
  %2036 = extractelement <8 x float> %2034, i64 1, !dbg !16
  %2037 = extractelement <8 x float> %2034, i64 2, !dbg !16
  %2038 = extractelement <8 x float> %2034, i64 3, !dbg !16
  %2039 = select i1 %.not34, float %2036, float %2035, !dbg !16
  %2040 = bitcast float %2039 to i32, !dbg !16
  %2041 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2040), !dbg !16
  %2042 = bitcast i32 %2041 to float, !dbg !16
  %2043 = select i1 %.not34, float %2035, float %2042, !dbg !16
  %2044 = select i1 %.not34, float %2042, float %2036, !dbg !16
  %2045 = select i1 %.not34, float %2038, float %2037, !dbg !16
  %2046 = bitcast float %2045 to i32, !dbg !16
  %2047 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2046), !dbg !16
  %2048 = bitcast i32 %2047 to float, !dbg !16
  %2049 = select i1 %.not34, float %2037, float %2048, !dbg !16
  %2050 = select i1 %.not34, float %2048, float %2038, !dbg !16
  %2051 = select i1 %.not35, float %2049, float %2043, !dbg !16
  %2052 = bitcast float %2051 to i32, !dbg !16
  %2053 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2052), !dbg !16
  %2054 = bitcast i32 %2053 to float, !dbg !16
  %2055 = select i1 %.not35, float %2043, float %2054, !dbg !16
  %2056 = select i1 %.not35, float %2054, float %2049, !dbg !16
  %2057 = select i1 %.not35, float %2050, float %2044, !dbg !16
  %2058 = bitcast float %2057 to i32, !dbg !16
  %2059 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2058), !dbg !16
  %2060 = bitcast i32 %2059 to float, !dbg !16
  %2061 = select i1 %.not35, float %2044, float %2060, !dbg !16
  %2062 = select i1 %.not35, float %2060, float %2050, !dbg !16
  %2063 = extractvalue [2 x <8 x float>] %59, 1, !dbg !16
  %2064 = extractelement <8 x float> %2063, i64 0, !dbg !16
  %2065 = extractelement <8 x float> %2063, i64 1, !dbg !16
  %2066 = extractelement <8 x float> %2063, i64 2, !dbg !16
  %2067 = extractelement <8 x float> %2063, i64 3, !dbg !16
  %2068 = select i1 %.not34, float %2065, float %2064, !dbg !16
  %2069 = bitcast float %2068 to i32, !dbg !16
  %2070 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2069), !dbg !16
  %2071 = bitcast i32 %2070 to float, !dbg !16
  %2072 = select i1 %.not34, float %2064, float %2071, !dbg !16
  %2073 = select i1 %.not34, float %2071, float %2065, !dbg !16
  %2074 = select i1 %.not34, float %2067, float %2066, !dbg !16
  %2075 = bitcast float %2074 to i32, !dbg !16
  %2076 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2075), !dbg !16
  %2077 = bitcast i32 %2076 to float, !dbg !16
  %2078 = select i1 %.not34, float %2066, float %2077, !dbg !16
  %2079 = select i1 %.not34, float %2077, float %2067, !dbg !16
  %2080 = select i1 %.not35, float %2078, float %2072, !dbg !16
  %2081 = bitcast float %2080 to i32, !dbg !16
  %2082 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2081), !dbg !16
  %2083 = bitcast i32 %2082 to float, !dbg !16
  %2084 = select i1 %.not35, float %2072, float %2083, !dbg !16
  %2085 = select i1 %.not35, float %2083, float %2078, !dbg !16
  %2086 = select i1 %.not35, float %2079, float %2073, !dbg !16
  %2087 = bitcast float %2086 to i32, !dbg !16
  %2088 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2087), !dbg !16
  %2089 = bitcast i32 %2088 to float, !dbg !16
  %2090 = select i1 %.not35, float %2073, float %2089, !dbg !16
  %2091 = select i1 %.not35, float %2089, float %2079, !dbg !16
  %2092 = extractvalue [2 x <8 x float>] %57, 1, !dbg !16
  %2093 = extractelement <8 x float> %2092, i64 0, !dbg !16
  %2094 = extractelement <8 x float> %2092, i64 1, !dbg !16
  %2095 = extractelement <8 x float> %2092, i64 2, !dbg !16
  %2096 = extractelement <8 x float> %2092, i64 3, !dbg !16
  %2097 = select i1 %.not34, float %2094, float %2093, !dbg !16
  %2098 = bitcast float %2097 to i32, !dbg !16
  %2099 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2098), !dbg !16
  %2100 = bitcast i32 %2099 to float, !dbg !16
  %2101 = select i1 %.not34, float %2093, float %2100, !dbg !16
  %2102 = select i1 %.not34, float %2100, float %2094, !dbg !16
  %2103 = select i1 %.not34, float %2096, float %2095, !dbg !16
  %2104 = bitcast float %2103 to i32, !dbg !16
  %2105 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2104), !dbg !16
  %2106 = bitcast i32 %2105 to float, !dbg !16
  %2107 = select i1 %.not34, float %2095, float %2106, !dbg !16
  %2108 = select i1 %.not34, float %2106, float %2096, !dbg !16
  %2109 = select i1 %.not35, float %2107, float %2101, !dbg !16
  %2110 = bitcast float %2109 to i32, !dbg !16
  %2111 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2110), !dbg !16
  %2112 = select i1 %.not35, float %2108, float %2102, !dbg !16
  %2113 = bitcast float %2112 to i32, !dbg !16
  %2114 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2113), !dbg !16
  %2115 = extractvalue [2 x <1 x float>] %58, 1, !dbg !19
  %2116 = shufflevector <1 x float> %2115, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %2117 = extractelement <8 x float> %2005, i64 4, !dbg !16
  %2118 = extractelement <8 x float> %2005, i64 5, !dbg !16
  %2119 = extractelement <8 x float> %2005, i64 6, !dbg !16
  %2120 = extractelement <8 x float> %2005, i64 7, !dbg !16
  %2121 = select i1 %.not34, float %2118, float %2117, !dbg !16
  %2122 = bitcast float %2121 to i32, !dbg !16
  %2123 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2122), !dbg !16
  %2124 = bitcast i32 %2123 to float, !dbg !16
  %2125 = select i1 %.not34, float %2117, float %2124, !dbg !16
  %2126 = select i1 %.not34, float %2124, float %2118, !dbg !16
  %2127 = select i1 %.not34, float %2120, float %2119, !dbg !16
  %2128 = bitcast float %2127 to i32, !dbg !16
  %2129 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2128), !dbg !16
  %2130 = bitcast i32 %2129 to float, !dbg !16
  %2131 = select i1 %.not34, float %2119, float %2130, !dbg !16
  %2132 = select i1 %.not34, float %2130, float %2120, !dbg !16
  %2133 = select i1 %.not35, float %2131, float %2125, !dbg !16
  %2134 = bitcast float %2133 to i32, !dbg !16
  %2135 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2134), !dbg !16
  %2136 = bitcast i32 %2135 to float, !dbg !16
  %2137 = select i1 %.not35, float %2125, float %2136, !dbg !16
  %2138 = select i1 %.not35, float %2136, float %2131, !dbg !16
  %2139 = select i1 %.not35, float %2132, float %2126, !dbg !16
  %2140 = bitcast float %2139 to i32, !dbg !16
  %2141 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2140), !dbg !16
  %2142 = bitcast i32 %2141 to float, !dbg !16
  %2143 = select i1 %.not35, float %2126, float %2142, !dbg !16
  %2144 = select i1 %.not35, float %2142, float %2132, !dbg !16
  %2145 = extractelement <8 x float> %2034, i64 4, !dbg !16
  %2146 = extractelement <8 x float> %2034, i64 5, !dbg !16
  %2147 = extractelement <8 x float> %2034, i64 6, !dbg !16
  %2148 = extractelement <8 x float> %2034, i64 7, !dbg !16
  %2149 = select i1 %.not34, float %2146, float %2145, !dbg !16
  %2150 = bitcast float %2149 to i32, !dbg !16
  %2151 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2150), !dbg !16
  %2152 = bitcast i32 %2151 to float, !dbg !16
  %2153 = select i1 %.not34, float %2145, float %2152, !dbg !16
  %2154 = select i1 %.not34, float %2152, float %2146, !dbg !16
  %2155 = select i1 %.not34, float %2148, float %2147, !dbg !16
  %2156 = bitcast float %2155 to i32, !dbg !16
  %2157 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2156), !dbg !16
  %2158 = bitcast i32 %2157 to float, !dbg !16
  %2159 = select i1 %.not34, float %2147, float %2158, !dbg !16
  %2160 = select i1 %.not34, float %2158, float %2148, !dbg !16
  %2161 = select i1 %.not35, float %2159, float %2153, !dbg !16
  %2162 = bitcast float %2161 to i32, !dbg !16
  %2163 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2162), !dbg !16
  %2164 = bitcast i32 %2163 to float, !dbg !16
  %2165 = select i1 %.not35, float %2153, float %2164, !dbg !16
  %2166 = select i1 %.not35, float %2164, float %2159, !dbg !16
  %2167 = select i1 %.not35, float %2160, float %2154, !dbg !16
  %2168 = bitcast float %2167 to i32, !dbg !16
  %2169 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2168), !dbg !16
  %2170 = bitcast i32 %2169 to float, !dbg !16
  %2171 = select i1 %.not35, float %2154, float %2170, !dbg !16
  %2172 = select i1 %.not35, float %2170, float %2160, !dbg !16
  %2173 = extractelement <8 x float> %2063, i64 4, !dbg !16
  %2174 = extractelement <8 x float> %2063, i64 5, !dbg !16
  %2175 = extractelement <8 x float> %2063, i64 6, !dbg !16
  %2176 = extractelement <8 x float> %2063, i64 7, !dbg !16
  %2177 = select i1 %.not34, float %2174, float %2173, !dbg !16
  %2178 = bitcast float %2177 to i32, !dbg !16
  %2179 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2178), !dbg !16
  %2180 = bitcast i32 %2179 to float, !dbg !16
  %2181 = select i1 %.not34, float %2173, float %2180, !dbg !16
  %2182 = select i1 %.not34, float %2180, float %2174, !dbg !16
  %2183 = select i1 %.not34, float %2176, float %2175, !dbg !16
  %2184 = bitcast float %2183 to i32, !dbg !16
  %2185 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2184), !dbg !16
  %2186 = bitcast i32 %2185 to float, !dbg !16
  %2187 = select i1 %.not34, float %2175, float %2186, !dbg !16
  %2188 = select i1 %.not34, float %2186, float %2176, !dbg !16
  %2189 = select i1 %.not35, float %2187, float %2181, !dbg !16
  %2190 = bitcast float %2189 to i32, !dbg !16
  %2191 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2190), !dbg !16
  %2192 = bitcast i32 %2191 to float, !dbg !16
  %2193 = select i1 %.not35, float %2181, float %2192, !dbg !16
  %2194 = select i1 %.not35, float %2192, float %2187, !dbg !16
  %2195 = select i1 %.not35, float %2188, float %2182, !dbg !16
  %2196 = bitcast float %2195 to i32, !dbg !16
  %2197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2196), !dbg !16
  %2198 = bitcast i32 %2197 to float, !dbg !16
  %2199 = select i1 %.not35, float %2182, float %2198, !dbg !16
  %2200 = select i1 %.not35, float %2198, float %2188, !dbg !16
  %2201 = extractelement <8 x float> %2092, i64 4, !dbg !16
  %2202 = extractelement <8 x float> %2092, i64 5, !dbg !16
  %2203 = extractelement <8 x float> %2092, i64 6, !dbg !16
  %2204 = extractelement <8 x float> %2092, i64 7, !dbg !16
  %2205 = select i1 %.not34, float %2202, float %2201, !dbg !16
  %2206 = bitcast float %2205 to i32, !dbg !16
  %2207 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2206), !dbg !16
  %2208 = bitcast i32 %2207 to float, !dbg !16
  %2209 = select i1 %.not34, float %2201, float %2208, !dbg !16
  %2210 = select i1 %.not34, float %2208, float %2202, !dbg !16
  %2211 = select i1 %.not34, float %2204, float %2203, !dbg !16
  %2212 = bitcast float %2211 to i32, !dbg !16
  %2213 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2212), !dbg !16
  %2214 = bitcast i32 %2213 to float, !dbg !16
  %2215 = select i1 %.not34, float %2203, float %2214, !dbg !16
  %2216 = select i1 %.not34, float %2214, float %2204, !dbg !16
  %2217 = select i1 %.not35, float %2215, float %2209, !dbg !16
  %2218 = bitcast float %2217 to i32, !dbg !16
  %2219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2218), !dbg !16
  %2220 = select i1 %.not35, float %2216, float %2210, !dbg !16
  %2221 = bitcast float %2220 to i32, !dbg !16
  %2222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2221), !dbg !16
  br i1 true, label %2223, label %2287, !dbg !21

2223:                                             ; preds = %1772
  %2224 = bitcast i32 %2219 to float, !dbg !16
  %2225 = select i1 %.not35, float %2209, float %2224, !dbg !16
  %2226 = insertelement <4 x float> poison, float %2225, i64 0, !dbg !19
  %2227 = bitcast i32 %2222 to float, !dbg !16
  %2228 = select i1 %.not35, float %2210, float %2227, !dbg !16
  %2229 = insertelement <4 x float> %2226, float %2228, i64 1, !dbg !19
  %2230 = select i1 %.not35, float %2224, float %2215, !dbg !16
  %2231 = insertelement <4 x float> %2229, float %2230, i64 2, !dbg !19
  %2232 = select i1 %.not35, float %2227, float %2216, !dbg !16
  %2233 = insertelement <4 x float> %2231, float %2232, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2116, !dbg !19
  %normalized = fmul <4 x float> %2233, %row.reciprocal, !dbg !19
  %2234 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %2235 = extractelement <4 x half> %2234, i64 3, !dbg !20
  %2236 = extractelement <4 x half> %2234, i64 2, !dbg !20
  %2237 = extractelement <4 x half> %2234, i64 1, !dbg !20
  %2238 = extractelement <4 x half> %2234, i64 0, !dbg !20
  %2239 = bitcast i32 %2111 to float, !dbg !16
  %2240 = select i1 %.not35, float %2101, float %2239, !dbg !16
  %2241 = insertelement <4 x float> poison, float %2240, i64 0, !dbg !19
  %2242 = bitcast i32 %2114 to float, !dbg !16
  %2243 = select i1 %.not35, float %2102, float %2242, !dbg !16
  %2244 = insertelement <4 x float> %2241, float %2243, i64 1, !dbg !19
  %2245 = select i1 %.not35, float %2239, float %2107, !dbg !16
  %2246 = insertelement <4 x float> %2244, float %2245, i64 2, !dbg !19
  %2247 = select i1 %.not35, float %2242, float %2108, !dbg !16
  %2248 = insertelement <4 x float> %2246, float %2247, i64 3, !dbg !19
  %normalized131 = fmul <4 x float> %2248, %row.reciprocal, !dbg !19
  %2249 = fptrunc <4 x float> %normalized131 to <4 x half>, !dbg !20
  %2250 = extractelement <4 x half> %2249, i64 3, !dbg !20
  %2251 = extractelement <4 x half> %2249, i64 2, !dbg !20
  %2252 = extractelement <4 x half> %2249, i64 1, !dbg !20
  %2253 = extractelement <4 x half> %2249, i64 0, !dbg !20
  %2254 = bitcast i32 %2001 to float, !dbg !16
  %2255 = select i1 %.not35, float %1991, float %2254, !dbg !16
  %2256 = insertelement <4 x float> poison, float %2255, i64 0, !dbg !19
  %2257 = bitcast i32 %2004 to float, !dbg !16
  %2258 = select i1 %.not35, float %1992, float %2257, !dbg !16
  %2259 = insertelement <4 x float> %2256, float %2258, i64 1, !dbg !19
  %2260 = select i1 %.not35, float %2254, float %1997, !dbg !16
  %2261 = insertelement <4 x float> %2259, float %2260, i64 2, !dbg !19
  %2262 = select i1 %.not35, float %2257, float %1998, !dbg !16
  %2263 = insertelement <4 x float> %2261, float %2262, i64 3, !dbg !19
  %row.reciprocal132 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1898, !dbg !19
  %normalized133 = fmul <4 x float> %2263, %row.reciprocal132, !dbg !19
  %2264 = fptrunc <4 x float> %normalized133 to <4 x half>, !dbg !20
  %2265 = extractelement <4 x half> %2264, i64 3, !dbg !20
  %2266 = extractelement <4 x half> %2264, i64 2, !dbg !20
  %2267 = extractelement <4 x half> %2264, i64 1, !dbg !20
  %2268 = extractelement <4 x half> %2264, i64 0, !dbg !20
  %2269 = bitcast i32 %1893 to float, !dbg !16
  %2270 = select i1 %.not35, float %1883, float %2269, !dbg !16
  %2271 = insertelement <4 x float> poison, float %2270, i64 0, !dbg !19
  %2272 = bitcast i32 %1896 to float, !dbg !16
  %2273 = select i1 %.not35, float %1884, float %2272, !dbg !16
  %2274 = insertelement <4 x float> %2271, float %2273, i64 1, !dbg !19
  %2275 = select i1 %.not35, float %2269, float %1889, !dbg !16
  %2276 = insertelement <4 x float> %2274, float %2275, i64 2, !dbg !19
  %2277 = select i1 %.not35, float %2272, float %1890, !dbg !16
  %2278 = insertelement <4 x float> %2276, float %2277, i64 3, !dbg !19
  %normalized134 = fmul <4 x float> %2278, %row.reciprocal132, !dbg !19
  %2279 = fptrunc <4 x float> %normalized134 to <4 x half>, !dbg !20
  %2280 = extractelement <4 x half> %2279, i64 3, !dbg !20
  %2281 = extractelement <4 x half> %2279, i64 2, !dbg !20
  %2282 = extractelement <4 x half> %2279, i64 1, !dbg !20
  %2283 = extractelement <4 x half> %2279, i64 0, !dbg !20
  %2284 = add nsw i64 %8, %13, !dbg !21
  %2285 = shl nsw i64 %2284, 19, !dbg !21
  %2286 = and i64 %2285, 16252928, !dbg !21
  %address.thread1670 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local1671 = zext nneg i32 %address.thread1670 to i64, !dbg !6
  %address.local1672 = add nsw i64 %16, %address.local1671, !dbg !21
  %address.local1673 = lshr i64 %address.local1671, 6, !dbg !6
  %address.local1674 = shl nuw nsw i64 %address.local1673, 4, !dbg !21
  %address.local1675 = add nsw i64 %address.local1672, %address.local1674, !dbg !21
  %address.local1676 = and i64 %address.local1671, 240, !dbg !21
  %address.local1677 = sub nsw i64 %address.local1675, %address.local1676, !dbg !21
  %address.local1678 = add nsw i64 %address.local1677, %17, !dbg !21
  %address.local1679 = shl nuw nsw i64 %address.local1678, 7, !dbg !21
  %address.local1680 = add nuw nsw i64 %2286, %address.local1679, !dbg !21
  %address.local1681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1680, !dbg !21
  %address.local1682 = lshr i64 %address.local1671, 4, !dbg !6
  %address.local1683 = and i64 %address.local1682, 3, !dbg !6
  %address.local1684 = shl nuw nsw i64 %address.local1683, 2, !dbg !21
  %address.local1685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1681, i64 %address.local1684, !dbg !21
  store half %2283, ptr addrspace(1) %address.local1685, align 2, !dbg !21
  %address.local1686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1680, !dbg !21
  %address.local1687 = or disjoint i64 %address.local1684, 1, !dbg !21
  %address.local1688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1686, i64 %address.local1687, !dbg !21
  store half %2282, ptr addrspace(1) %address.local1688, align 2, !dbg !21
  %address.local1689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1680, !dbg !21
  %address.local1690 = or disjoint i64 %address.local1684, 2, !dbg !21
  %address.local1691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1689, i64 %address.local1690, !dbg !21
  store half %2281, ptr addrspace(1) %address.local1691, align 2, !dbg !21
  %address.local1692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1680, !dbg !21
  %address.local1693 = or disjoint i64 %address.local1684, 3, !dbg !21
  %address.local1694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1692, i64 %address.local1693, !dbg !21
  store half %2280, ptr addrspace(1) %address.local1694, align 2, !dbg !21
  %address.local1695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1680, !dbg !21
  %address.local1696 = or disjoint i64 %address.local1684, 16, !dbg !21
  %address.local1697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1695, i64 %address.local1696, !dbg !21
  store half %2268, ptr addrspace(1) %address.local1697, align 2, !dbg !21
  %address.local1698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1680, !dbg !21
  %address.local1699 = or disjoint i64 %address.local1684, 17, !dbg !21
  %address.local1700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1698, i64 %address.local1699, !dbg !21
  store half %2267, ptr addrspace(1) %address.local1700, align 2, !dbg !21
  %address.local1701 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1680, !dbg !21
  %address.local1702 = or disjoint i64 %address.local1684, 18, !dbg !21
  %address.local1703 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1701, i64 %address.local1702, !dbg !21
  store half %2266, ptr addrspace(1) %address.local1703, align 2, !dbg !21
  %address.local1704 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1680, !dbg !21
  %address.local1705 = or disjoint i64 %address.local1684, 19, !dbg !21
  %address.local1706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1704, i64 %address.local1705, !dbg !21
  store half %2265, ptr addrspace(1) %address.local1706, align 2, !dbg !21
  %address.local1707 = shl nsw i64 %address.local1678, 7, !dbg !21
  %address.local1708 = add i64 %address.local1707, 8192, !dbg !21
  %address.local1709 = add nuw nsw i64 %2286, %address.local1708, !dbg !21
  %address.local1710 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1709, !dbg !21
  %address.local1711 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1710, i64 %address.local1684, !dbg !21
  store half %2253, ptr addrspace(1) %address.local1711, align 2, !dbg !21
  %address.local1712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1709, !dbg !21
  %address.local1713 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1712, i64 %address.local1687, !dbg !21
  store half %2252, ptr addrspace(1) %address.local1713, align 2, !dbg !21
  %address.local1714 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1709, !dbg !21
  %address.local1715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1714, i64 %address.local1690, !dbg !21
  store half %2251, ptr addrspace(1) %address.local1715, align 2, !dbg !21
  %address.local1716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1709, !dbg !21
  %address.local1717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1716, i64 %address.local1693, !dbg !21
  store half %2250, ptr addrspace(1) %address.local1717, align 2, !dbg !21
  %address.local1718 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1709, !dbg !21
  %address.local1719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1718, i64 %address.local1696, !dbg !21
  store half %2238, ptr addrspace(1) %address.local1719, align 2, !dbg !21
  %address.local1720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1709, !dbg !21
  %address.local1721 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1720, i64 %address.local1699, !dbg !21
  store half %2237, ptr addrspace(1) %address.local1721, align 2, !dbg !21
  %address.local1722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1709, !dbg !21
  %address.local1723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1722, i64 %address.local1702, !dbg !21
  store half %2236, ptr addrspace(1) %address.local1723, align 2, !dbg !21
  %address.local1724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1709, !dbg !21
  %address.local1725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1724, i64 %address.local1705, !dbg !21
  store half %2235, ptr addrspace(1) %address.local1725, align 2, !dbg !21
  br label %2287, !dbg !21

2287:                                             ; preds = %2223, %1772
  br i1 true, label %2288, label %2326, !dbg !21

2288:                                             ; preds = %2287
  %2289 = insertelement <4 x float> poison, float %2193, i64 0, !dbg !19
  %2290 = insertelement <4 x float> %2289, float %2199, i64 1, !dbg !19
  %2291 = insertelement <4 x float> %2290, float %2194, i64 2, !dbg !19
  %2292 = insertelement <4 x float> %2291, float %2200, i64 3, !dbg !19
  %row.reciprocal135 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2116, !dbg !19
  %normalized136 = fmul <4 x float> %2292, %row.reciprocal135, !dbg !19
  %2293 = fptrunc <4 x float> %normalized136 to <4 x half>, !dbg !20
  %2294 = extractelement <4 x half> %2293, i64 3, !dbg !20
  %2295 = extractelement <4 x half> %2293, i64 2, !dbg !20
  %2296 = extractelement <4 x half> %2293, i64 1, !dbg !20
  %2297 = extractelement <4 x half> %2293, i64 0, !dbg !20
  %2298 = insertelement <4 x float> poison, float %2084, i64 0, !dbg !19
  %2299 = insertelement <4 x float> %2298, float %2090, i64 1, !dbg !19
  %2300 = insertelement <4 x float> %2299, float %2085, i64 2, !dbg !19
  %2301 = insertelement <4 x float> %2300, float %2091, i64 3, !dbg !19
  %normalized137 = fmul <4 x float> %2301, %row.reciprocal135, !dbg !19
  %2302 = fptrunc <4 x float> %normalized137 to <4 x half>, !dbg !20
  %2303 = extractelement <4 x half> %2302, i64 3, !dbg !20
  %2304 = extractelement <4 x half> %2302, i64 2, !dbg !20
  %2305 = extractelement <4 x half> %2302, i64 1, !dbg !20
  %2306 = extractelement <4 x half> %2302, i64 0, !dbg !20
  %2307 = insertelement <4 x float> poison, float %1975, i64 0, !dbg !19
  %2308 = insertelement <4 x float> %2307, float %1981, i64 1, !dbg !19
  %2309 = insertelement <4 x float> %2308, float %1976, i64 2, !dbg !19
  %2310 = insertelement <4 x float> %2309, float %1982, i64 3, !dbg !19
  %row.reciprocal138 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1898, !dbg !19
  %normalized139 = fmul <4 x float> %2310, %row.reciprocal138, !dbg !19
  %2311 = fptrunc <4 x float> %normalized139 to <4 x half>, !dbg !20
  %2312 = extractelement <4 x half> %2311, i64 3, !dbg !20
  %2313 = extractelement <4 x half> %2311, i64 2, !dbg !20
  %2314 = extractelement <4 x half> %2311, i64 1, !dbg !20
  %2315 = extractelement <4 x half> %2311, i64 0, !dbg !20
  %2316 = insertelement <4 x float> poison, float %1866, i64 0, !dbg !19
  %2317 = insertelement <4 x float> %2316, float %1872, i64 1, !dbg !19
  %2318 = insertelement <4 x float> %2317, float %1867, i64 2, !dbg !19
  %2319 = insertelement <4 x float> %2318, float %1873, i64 3, !dbg !19
  %normalized140 = fmul <4 x float> %2319, %row.reciprocal138, !dbg !19
  %2320 = fptrunc <4 x float> %normalized140 to <4 x half>, !dbg !20
  %2321 = extractelement <4 x half> %2320, i64 3, !dbg !20
  %2322 = extractelement <4 x half> %2320, i64 2, !dbg !20
  %2323 = extractelement <4 x half> %2320, i64 1, !dbg !20
  %2324 = extractelement <4 x half> %2320, i64 0, !dbg !20
  %2325 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread1726 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local1727 = zext nneg i32 %address.thread1726 to i64, !dbg !6
  %address.local1728 = add nsw i64 %16, %address.local1727, !dbg !21
  %address.local1729 = lshr i64 %address.local1727, 6, !dbg !6
  %address.local1730 = shl nuw nsw i64 %address.local1729, 4, !dbg !21
  %address.local1731 = add nsw i64 %address.local1728, %address.local1730, !dbg !21
  %address.local1732 = and i64 %address.local1727, 240, !dbg !21
  %address.local1733 = sub nsw i64 %address.local1731, %address.local1732, !dbg !21
  %address.local1734 = add nsw i64 %address.local1733, %17, !dbg !21
  %address.local1735 = shl nuw nsw i64 %address.local1734, 7, !dbg !21
  %address.local1736 = add nuw nsw i64 %2325, %address.local1735, !dbg !21
  %address.local1737 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1736, !dbg !21
  %address.local1738 = lshr i64 %address.local1727, 4, !dbg !6
  %address.local1739 = and i64 %address.local1738, 3, !dbg !6
  %address.local1740 = shl nuw nsw i64 %address.local1739, 2, !dbg !21
  %address.local1741 = or disjoint i64 %address.local1740, 32, !dbg !21
  %address.local1742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1737, i64 %address.local1741, !dbg !21
  store half %2324, ptr addrspace(1) %address.local1742, align 2, !dbg !21
  %address.local1743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1736, !dbg !21
  %address.local1744 = or disjoint i64 %address.local1740, 33, !dbg !21
  %address.local1745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1743, i64 %address.local1744, !dbg !21
  store half %2323, ptr addrspace(1) %address.local1745, align 2, !dbg !21
  %address.local1746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1736, !dbg !21
  %address.local1747 = or disjoint i64 %address.local1740, 34, !dbg !21
  %address.local1748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1746, i64 %address.local1747, !dbg !21
  store half %2322, ptr addrspace(1) %address.local1748, align 2, !dbg !21
  %address.local1749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1736, !dbg !21
  %address.local1750 = or disjoint i64 %address.local1740, 35, !dbg !21
  %address.local1751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1749, i64 %address.local1750, !dbg !21
  store half %2321, ptr addrspace(1) %address.local1751, align 2, !dbg !21
  %address.local1752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1736, !dbg !21
  %address.local1753 = or disjoint i64 %address.local1740, 48, !dbg !21
  %address.local1754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1752, i64 %address.local1753, !dbg !21
  store half %2315, ptr addrspace(1) %address.local1754, align 2, !dbg !21
  %address.local1755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1736, !dbg !21
  %address.local1756 = or disjoint i64 %address.local1740, 49, !dbg !21
  %address.local1757 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1755, i64 %address.local1756, !dbg !21
  store half %2314, ptr addrspace(1) %address.local1757, align 2, !dbg !21
  %address.local1758 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1736, !dbg !21
  %address.local1759 = or disjoint i64 %address.local1740, 50, !dbg !21
  %address.local1760 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1758, i64 %address.local1759, !dbg !21
  store half %2313, ptr addrspace(1) %address.local1760, align 2, !dbg !21
  %address.local1761 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1736, !dbg !21
  %address.local1762 = or disjoint i64 %address.local1740, 51, !dbg !21
  %address.local1763 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1761, i64 %address.local1762, !dbg !21
  store half %2312, ptr addrspace(1) %address.local1763, align 2, !dbg !21
  %address.local1764 = shl nsw i64 %address.local1734, 7, !dbg !21
  %address.local1765 = add i64 %address.local1764, 8192, !dbg !21
  %address.local1766 = add nuw nsw i64 %2325, %address.local1765, !dbg !21
  %address.local1767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1766, !dbg !21
  %address.local1768 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1767, i64 %address.local1741, !dbg !21
  store half %2306, ptr addrspace(1) %address.local1768, align 2, !dbg !21
  %address.local1769 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1766, !dbg !21
  %address.local1770 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1769, i64 %address.local1744, !dbg !21
  store half %2305, ptr addrspace(1) %address.local1770, align 2, !dbg !21
  %address.local1771 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1766, !dbg !21
  %address.local1772 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1771, i64 %address.local1747, !dbg !21
  store half %2304, ptr addrspace(1) %address.local1772, align 2, !dbg !21
  %address.local1773 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1766, !dbg !21
  %address.local1774 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1773, i64 %address.local1750, !dbg !21
  store half %2303, ptr addrspace(1) %address.local1774, align 2, !dbg !21
  %address.local1775 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1766, !dbg !21
  %address.local1776 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1775, i64 %address.local1753, !dbg !21
  store half %2297, ptr addrspace(1) %address.local1776, align 2, !dbg !21
  %address.local1777 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1766, !dbg !21
  %address.local1778 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1777, i64 %address.local1756, !dbg !21
  store half %2296, ptr addrspace(1) %address.local1778, align 2, !dbg !21
  %address.local1779 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1766, !dbg !21
  %address.local1780 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1779, i64 %address.local1759, !dbg !21
  store half %2295, ptr addrspace(1) %address.local1780, align 2, !dbg !21
  %address.local1781 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1766, !dbg !21
  %address.local1782 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1781, i64 %address.local1762, !dbg !21
  store half %2294, ptr addrspace(1) %address.local1782, align 2, !dbg !21
  br label %2326, !dbg !21

2326:                                             ; preds = %2288, %2287
  br i1 true, label %2327, label %2365, !dbg !21

2327:                                             ; preds = %2326
  %2328 = insertelement <4 x float> poison, float %2165, i64 0, !dbg !19
  %2329 = insertelement <4 x float> %2328, float %2171, i64 1, !dbg !19
  %2330 = insertelement <4 x float> %2329, float %2166, i64 2, !dbg !19
  %2331 = insertelement <4 x float> %2330, float %2172, i64 3, !dbg !19
  %row.reciprocal141 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2116, !dbg !19
  %normalized142 = fmul <4 x float> %2331, %row.reciprocal141, !dbg !19
  %2332 = fptrunc <4 x float> %normalized142 to <4 x half>, !dbg !20
  %2333 = extractelement <4 x half> %2332, i64 3, !dbg !20
  %2334 = extractelement <4 x half> %2332, i64 2, !dbg !20
  %2335 = extractelement <4 x half> %2332, i64 1, !dbg !20
  %2336 = extractelement <4 x half> %2332, i64 0, !dbg !20
  %2337 = insertelement <4 x float> poison, float %2055, i64 0, !dbg !19
  %2338 = insertelement <4 x float> %2337, float %2061, i64 1, !dbg !19
  %2339 = insertelement <4 x float> %2338, float %2056, i64 2, !dbg !19
  %2340 = insertelement <4 x float> %2339, float %2062, i64 3, !dbg !19
  %normalized143 = fmul <4 x float> %2340, %row.reciprocal141, !dbg !19
  %2341 = fptrunc <4 x float> %normalized143 to <4 x half>, !dbg !20
  %2342 = extractelement <4 x half> %2341, i64 3, !dbg !20
  %2343 = extractelement <4 x half> %2341, i64 2, !dbg !20
  %2344 = extractelement <4 x half> %2341, i64 1, !dbg !20
  %2345 = extractelement <4 x half> %2341, i64 0, !dbg !20
  %2346 = insertelement <4 x float> poison, float %1947, i64 0, !dbg !19
  %2347 = insertelement <4 x float> %2346, float %1953, i64 1, !dbg !19
  %2348 = insertelement <4 x float> %2347, float %1948, i64 2, !dbg !19
  %2349 = insertelement <4 x float> %2348, float %1954, i64 3, !dbg !19
  %row.reciprocal144 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1898, !dbg !19
  %normalized145 = fmul <4 x float> %2349, %row.reciprocal144, !dbg !19
  %2350 = fptrunc <4 x float> %normalized145 to <4 x half>, !dbg !20
  %2351 = extractelement <4 x half> %2350, i64 3, !dbg !20
  %2352 = extractelement <4 x half> %2350, i64 2, !dbg !20
  %2353 = extractelement <4 x half> %2350, i64 1, !dbg !20
  %2354 = extractelement <4 x half> %2350, i64 0, !dbg !20
  %2355 = insertelement <4 x float> poison, float %1837, i64 0, !dbg !19
  %2356 = insertelement <4 x float> %2355, float %1843, i64 1, !dbg !19
  %2357 = insertelement <4 x float> %2356, float %1838, i64 2, !dbg !19
  %2358 = insertelement <4 x float> %2357, float %1844, i64 3, !dbg !19
  %normalized146 = fmul <4 x float> %2358, %row.reciprocal144, !dbg !19
  %2359 = fptrunc <4 x float> %normalized146 to <4 x half>, !dbg !20
  %2360 = extractelement <4 x half> %2359, i64 3, !dbg !20
  %2361 = extractelement <4 x half> %2359, i64 2, !dbg !20
  %2362 = extractelement <4 x half> %2359, i64 1, !dbg !20
  %2363 = extractelement <4 x half> %2359, i64 0, !dbg !20
  %2364 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread1783 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local1784 = zext nneg i32 %address.thread1783 to i64, !dbg !6
  %address.local1785 = add nsw i64 %16, %address.local1784, !dbg !21
  %address.local1786 = lshr i64 %address.local1784, 6, !dbg !6
  %address.local1787 = shl nuw nsw i64 %address.local1786, 4, !dbg !21
  %address.local1788 = add nsw i64 %address.local1785, %address.local1787, !dbg !21
  %address.local1789 = and i64 %address.local1784, 240, !dbg !21
  %address.local1790 = sub nsw i64 %address.local1788, %address.local1789, !dbg !21
  %address.local1791 = add nsw i64 %address.local1790, %17, !dbg !21
  %address.local1792 = shl nuw nsw i64 %address.local1791, 7, !dbg !21
  %address.local1793 = add nuw nsw i64 %2364, %address.local1792, !dbg !21
  %address.local1794 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1793, !dbg !21
  %address.local1795 = lshr i64 %address.local1784, 4, !dbg !6
  %address.local1796 = and i64 %address.local1795, 3, !dbg !6
  %address.local1797 = shl nuw nsw i64 %address.local1796, 2, !dbg !21
  %address.local1798 = or disjoint i64 %address.local1797, 64, !dbg !21
  %address.local1799 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1794, i64 %address.local1798, !dbg !21
  store half %2363, ptr addrspace(1) %address.local1799, align 2, !dbg !21
  %address.local1800 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1793, !dbg !21
  %address.local1801 = or disjoint i64 %address.local1797, 65, !dbg !21
  %address.local1802 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1800, i64 %address.local1801, !dbg !21
  store half %2362, ptr addrspace(1) %address.local1802, align 2, !dbg !21
  %address.local1803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1793, !dbg !21
  %address.local1804 = or disjoint i64 %address.local1797, 66, !dbg !21
  %address.local1805 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1803, i64 %address.local1804, !dbg !21
  store half %2361, ptr addrspace(1) %address.local1805, align 2, !dbg !21
  %address.local1806 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1793, !dbg !21
  %address.local1807 = or disjoint i64 %address.local1797, 67, !dbg !21
  %address.local1808 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1806, i64 %address.local1807, !dbg !21
  store half %2360, ptr addrspace(1) %address.local1808, align 2, !dbg !21
  %address.local1809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1793, !dbg !21
  %address.local1810 = or disjoint i64 %address.local1797, 80, !dbg !21
  %address.local1811 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1809, i64 %address.local1810, !dbg !21
  store half %2354, ptr addrspace(1) %address.local1811, align 2, !dbg !21
  %address.local1812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1793, !dbg !21
  %address.local1813 = or disjoint i64 %address.local1797, 81, !dbg !21
  %address.local1814 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1812, i64 %address.local1813, !dbg !21
  store half %2353, ptr addrspace(1) %address.local1814, align 2, !dbg !21
  %address.local1815 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1793, !dbg !21
  %address.local1816 = or disjoint i64 %address.local1797, 82, !dbg !21
  %address.local1817 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1815, i64 %address.local1816, !dbg !21
  store half %2352, ptr addrspace(1) %address.local1817, align 2, !dbg !21
  %address.local1818 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1793, !dbg !21
  %address.local1819 = or disjoint i64 %address.local1797, 83, !dbg !21
  %address.local1820 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1818, i64 %address.local1819, !dbg !21
  store half %2351, ptr addrspace(1) %address.local1820, align 2, !dbg !21
  %address.local1821 = shl nsw i64 %address.local1791, 7, !dbg !21
  %address.local1822 = add i64 %address.local1821, 8192, !dbg !21
  %address.local1823 = add nuw nsw i64 %2364, %address.local1822, !dbg !21
  %address.local1824 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1823, !dbg !21
  %address.local1825 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1824, i64 %address.local1798, !dbg !21
  store half %2345, ptr addrspace(1) %address.local1825, align 2, !dbg !21
  %address.local1826 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1823, !dbg !21
  %address.local1827 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1826, i64 %address.local1801, !dbg !21
  store half %2344, ptr addrspace(1) %address.local1827, align 2, !dbg !21
  %address.local1828 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1823, !dbg !21
  %address.local1829 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1828, i64 %address.local1804, !dbg !21
  store half %2343, ptr addrspace(1) %address.local1829, align 2, !dbg !21
  %address.local1830 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1823, !dbg !21
  %address.local1831 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1830, i64 %address.local1807, !dbg !21
  store half %2342, ptr addrspace(1) %address.local1831, align 2, !dbg !21
  %address.local1832 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1823, !dbg !21
  %address.local1833 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1832, i64 %address.local1810, !dbg !21
  store half %2336, ptr addrspace(1) %address.local1833, align 2, !dbg !21
  %address.local1834 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1823, !dbg !21
  %address.local1835 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1834, i64 %address.local1813, !dbg !21
  store half %2335, ptr addrspace(1) %address.local1835, align 2, !dbg !21
  %address.local1836 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1823, !dbg !21
  %address.local1837 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1836, i64 %address.local1816, !dbg !21
  store half %2334, ptr addrspace(1) %address.local1837, align 2, !dbg !21
  %address.local1838 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1823, !dbg !21
  %address.local1839 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1838, i64 %address.local1819, !dbg !21
  store half %2333, ptr addrspace(1) %address.local1839, align 2, !dbg !21
  br label %2365, !dbg !21

2365:                                             ; preds = %2327, %2326
  br i1 true, label %2366, label %2404, !dbg !21

2366:                                             ; preds = %2365
  %2367 = insertelement <4 x float> poison, float %2137, i64 0, !dbg !19
  %2368 = insertelement <4 x float> %2367, float %2143, i64 1, !dbg !19
  %2369 = insertelement <4 x float> %2368, float %2138, i64 2, !dbg !19
  %2370 = insertelement <4 x float> %2369, float %2144, i64 3, !dbg !19
  %row.reciprocal147 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2116, !dbg !19
  %normalized148 = fmul <4 x float> %2370, %row.reciprocal147, !dbg !19
  %2371 = fptrunc <4 x float> %normalized148 to <4 x half>, !dbg !20
  %2372 = extractelement <4 x half> %2371, i64 3, !dbg !20
  %2373 = extractelement <4 x half> %2371, i64 2, !dbg !20
  %2374 = extractelement <4 x half> %2371, i64 1, !dbg !20
  %2375 = extractelement <4 x half> %2371, i64 0, !dbg !20
  %2376 = insertelement <4 x float> poison, float %2026, i64 0, !dbg !19
  %2377 = insertelement <4 x float> %2376, float %2032, i64 1, !dbg !19
  %2378 = insertelement <4 x float> %2377, float %2027, i64 2, !dbg !19
  %2379 = insertelement <4 x float> %2378, float %2033, i64 3, !dbg !19
  %normalized149 = fmul <4 x float> %2379, %row.reciprocal147, !dbg !19
  %2380 = fptrunc <4 x float> %normalized149 to <4 x half>, !dbg !20
  %2381 = extractelement <4 x half> %2380, i64 3, !dbg !20
  %2382 = extractelement <4 x half> %2380, i64 2, !dbg !20
  %2383 = extractelement <4 x half> %2380, i64 1, !dbg !20
  %2384 = extractelement <4 x half> %2380, i64 0, !dbg !20
  %2385 = insertelement <4 x float> poison, float %1919, i64 0, !dbg !19
  %2386 = insertelement <4 x float> %2385, float %1925, i64 1, !dbg !19
  %2387 = insertelement <4 x float> %2386, float %1920, i64 2, !dbg !19
  %2388 = insertelement <4 x float> %2387, float %1926, i64 3, !dbg !19
  %row.reciprocal150 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1898, !dbg !19
  %normalized151 = fmul <4 x float> %2388, %row.reciprocal150, !dbg !19
  %2389 = fptrunc <4 x float> %normalized151 to <4 x half>, !dbg !20
  %2390 = extractelement <4 x half> %2389, i64 3, !dbg !20
  %2391 = extractelement <4 x half> %2389, i64 2, !dbg !20
  %2392 = extractelement <4 x half> %2389, i64 1, !dbg !20
  %2393 = extractelement <4 x half> %2389, i64 0, !dbg !20
  %2394 = insertelement <4 x float> poison, float %1808, i64 0, !dbg !19
  %2395 = insertelement <4 x float> %2394, float %1814, i64 1, !dbg !19
  %2396 = insertelement <4 x float> %2395, float %1809, i64 2, !dbg !19
  %2397 = insertelement <4 x float> %2396, float %1815, i64 3, !dbg !19
  %normalized152 = fmul <4 x float> %2397, %row.reciprocal150, !dbg !19
  %2398 = fptrunc <4 x float> %normalized152 to <4 x half>, !dbg !20
  %2399 = extractelement <4 x half> %2398, i64 3, !dbg !20
  %2400 = extractelement <4 x half> %2398, i64 2, !dbg !20
  %2401 = extractelement <4 x half> %2398, i64 1, !dbg !20
  %2402 = extractelement <4 x half> %2398, i64 0, !dbg !20
  %2403 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread1840 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local1841 = zext nneg i32 %address.thread1840 to i64, !dbg !6
  %address.local1842 = add nsw i64 %16, %address.local1841, !dbg !21
  %address.local1843 = lshr i64 %address.local1841, 6, !dbg !6
  %address.local1844 = shl nuw nsw i64 %address.local1843, 4, !dbg !21
  %address.local1845 = add nsw i64 %address.local1842, %address.local1844, !dbg !21
  %address.local1846 = and i64 %address.local1841, 240, !dbg !21
  %address.local1847 = sub nsw i64 %address.local1845, %address.local1846, !dbg !21
  %address.local1848 = add nsw i64 %address.local1847, %17, !dbg !21
  %address.local1849 = shl nuw nsw i64 %address.local1848, 7, !dbg !21
  %address.local1850 = add nuw nsw i64 %2403, %address.local1849, !dbg !21
  %address.local1851 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1850, !dbg !21
  %address.local1852 = lshr i64 %address.local1841, 4, !dbg !6
  %address.local1853 = and i64 %address.local1852, 3, !dbg !6
  %address.local1854 = shl nuw nsw i64 %address.local1853, 2, !dbg !21
  %address.local1855 = or disjoint i64 %address.local1854, 96, !dbg !21
  %address.local1856 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1851, i64 %address.local1855, !dbg !21
  store half %2402, ptr addrspace(1) %address.local1856, align 2, !dbg !21
  %address.local1857 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1850, !dbg !21
  %address.local1858 = or disjoint i64 %address.local1854, 97, !dbg !21
  %address.local1859 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1857, i64 %address.local1858, !dbg !21
  store half %2401, ptr addrspace(1) %address.local1859, align 2, !dbg !21
  %address.local1860 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1850, !dbg !21
  %address.local1861 = or disjoint i64 %address.local1854, 98, !dbg !21
  %address.local1862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1860, i64 %address.local1861, !dbg !21
  store half %2400, ptr addrspace(1) %address.local1862, align 2, !dbg !21
  %address.local1863 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1850, !dbg !21
  %address.local1864 = or disjoint i64 %address.local1854, 99, !dbg !21
  %address.local1865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1863, i64 %address.local1864, !dbg !21
  store half %2399, ptr addrspace(1) %address.local1865, align 2, !dbg !21
  %address.local1866 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1850, !dbg !21
  %address.local1867 = or disjoint i64 %address.local1854, 112, !dbg !21
  %address.local1868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1866, i64 %address.local1867, !dbg !21
  store half %2393, ptr addrspace(1) %address.local1868, align 2, !dbg !21
  %address.local1869 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1850, !dbg !21
  %address.local1870 = or disjoint i64 %address.local1854, 113, !dbg !21
  %address.local1871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1869, i64 %address.local1870, !dbg !21
  store half %2392, ptr addrspace(1) %address.local1871, align 2, !dbg !21
  %address.local1872 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1850, !dbg !21
  %address.local1873 = or disjoint i64 %address.local1854, 114, !dbg !21
  %address.local1874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1872, i64 %address.local1873, !dbg !21
  store half %2391, ptr addrspace(1) %address.local1874, align 2, !dbg !21
  %address.local1875 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1850, !dbg !21
  %address.local1876 = or disjoint i64 %address.local1854, 115, !dbg !21
  %address.local1877 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1875, i64 %address.local1876, !dbg !21
  store half %2390, ptr addrspace(1) %address.local1877, align 2, !dbg !21
  %address.local1878 = shl nsw i64 %address.local1848, 7, !dbg !21
  %address.local1879 = add i64 %address.local1878, 8192, !dbg !21
  %address.local1880 = add nuw nsw i64 %2403, %address.local1879, !dbg !21
  %address.local1881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1880, !dbg !21
  %address.local1882 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1881, i64 %address.local1855, !dbg !21
  store half %2384, ptr addrspace(1) %address.local1882, align 2, !dbg !21
  %address.local1883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1880, !dbg !21
  %address.local1884 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1883, i64 %address.local1858, !dbg !21
  store half %2383, ptr addrspace(1) %address.local1884, align 2, !dbg !21
  %address.local1885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1880, !dbg !21
  %address.local1886 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1885, i64 %address.local1861, !dbg !21
  store half %2382, ptr addrspace(1) %address.local1886, align 2, !dbg !21
  %address.local1887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1880, !dbg !21
  %address.local1888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1887, i64 %address.local1864, !dbg !21
  store half %2381, ptr addrspace(1) %address.local1888, align 2, !dbg !21
  %address.local1889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1880, !dbg !21
  %address.local1890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1889, i64 %address.local1867, !dbg !21
  store half %2375, ptr addrspace(1) %address.local1890, align 2, !dbg !21
  %address.local1891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1880, !dbg !21
  %address.local1892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1891, i64 %address.local1870, !dbg !21
  store half %2374, ptr addrspace(1) %address.local1892, align 2, !dbg !21
  %address.local1893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1880, !dbg !21
  %address.local1894 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1893, i64 %address.local1873, !dbg !21
  store half %2373, ptr addrspace(1) %address.local1894, align 2, !dbg !21
  %address.local1895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1880, !dbg !21
  %address.local1896 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1895, i64 %address.local1876, !dbg !21
  store half %2372, ptr addrspace(1) %address.local1896, align 2, !dbg !21
  br label %2404, !dbg !21

2404:                                             ; preds = %2366, %2365
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
!llvm.module.flags = !{!32}

!0 = !DIFile(filename: "test_input_spidD.mlir", directory: "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!6 = !DILocation(line: 68, column: 24, scope: !4)
!7 = !DILocation(line: 92, column: 20, scope: !4)
!8 = !DILocation(line: 69, column: 26, scope: !4)
!12 = !DILocation(line: 46, column: 17, scope: !4)
!14 = !DILocation(line: 66, column: 22, scope: !4)
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
!26 = !DILocation(line: 85, column: 22, scope: !4)
!27 = !DILocation(line: 86, column: 20, scope: !4)
!28 = !DILocation(line: 87, column: 16, scope: !4)
!29 = !DILocation(line: 94, column: 20, scope: !4)
!30 = !DILocation(line: 70, column: 23, scope: !4)
!31 = !{i32 256, i32 1, i32 1}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 0, i32 256}
