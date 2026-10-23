; ModuleID = '3rd/deepgengraph/test/test_input_sd16.mlir'
source_filename = "3rd/deepgengraph/test/test_input_sd16.mlir"

@async_lds_staging_1 = private addrspace(3) global [2048 x half] undef, align 16
@async_lds_staging_0 = private addrspace(3) global [2048 x half] undef, align 16
@shm_0 = addrspace(3) global [8192 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !33 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %6, 7, !dbg !13
  %.lobit = ashr i32 %5, 31, !dbg !15
  %10 = xor i32 %5, %.lobit, !dbg !15
  %11 = sdiv i32 %10, 32, !dbg !15
  %.lobit154 = ashr i32 %5, 31, !dbg !15
  %12 = xor i32 %11, %.lobit154, !dbg !15
  %13 = sext i32 %12 to i64, !dbg !15
  %14 = add nsw i64 %8, %13, !dbg !15
  %15 = and i64 %14, 31, !dbg !15
  %16 = shl nsw i64 %6, 7, !dbg !15
  %17 = mul nsw i64 %13, -4096, !dbg !15
  %18 = add nsw i64 %16, %17, !dbg !15
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !16, !range !35
  %20 = zext nneg i32 %19 to i64, !dbg !16
  %21 = and i32 %19, 16, !dbg !17
  %.not = icmp eq i32 %21, 0, !dbg !17
  %22 = and i32 %19, 32, !dbg !17
  %.not155 = icmp eq i32 %22, 0, !dbg !17
  %23 = add nsw i64 %9, %20, !dbg !20
  %24 = lshr i32 %19, 2, !dbg !20
  %25 = and i32 %24, 48, !dbg !20
  %index.compact2 = zext nneg i32 %25 to i64, !dbg !20
  %26 = add nsw i64 %23, %index.compact2, !dbg !20
  %27 = and i32 %19, 240, !dbg !20
  %28 = sub nsw i32 0, %27, !dbg !20
  %index.compact1 = sext i32 %28 to i64, !dbg !20
  %29 = add nsw i64 %26, %index.compact1, !dbg !20
  %30 = lshr i32 %19, 2, !dbg !20
  %31 = and i32 %30, 12, !dbg !20
  %index.compact = zext nneg i32 %31 to i64, !dbg !20
  %32 = add nsw i64 %29, 1, !dbg !21
  %33 = add nsw i64 %29, 65, !dbg !21
  br label %34, !dbg !22

34:                                               ; preds = %1464, %4
  %35 = phi i64 [ %1465, %1464 ], [ 0, %4 ], !dbg !23
  %36 = icmp samesign ult i64 %35, 4, !dbg !22
  br i1 %36, label %37, label %1466, !dbg !22

37:                                               ; preds = %34
  %38 = shl nuw nsw i64 %35, 5, !dbg !22
  br label %39, !dbg !19

39:                                               ; preds = %1241, %37
  %40 = phi i64 [ %1285, %1241 ], [ 0, %37 ], !dbg !19
  %41 = phi [2 x <4 x float>] [ %1254, %1241 ], [ zeroinitializer, %37 ], !dbg !19
  %42 = phi [2 x <1 x float>] [ %1279, %1241 ], [ zeroinitializer, %37 ], !dbg !19
  %43 = phi [2 x <4 x float>] [ %1284, %1241 ], [ zeroinitializer, %37 ], !dbg !19
  %.not161 = icmp sgt i64 %40, %6, !dbg !19
  br i1 %.not161, label %1286, label %44, !dbg !19

44:                                               ; preds = %39
  %45 = shl nsw i64 %40, 7, !dbg !6
  %46 = lshr i64 %40, 5, !dbg !6
  %47 = shl i64 %46, 12, !dbg !6
  %48 = sub i64 %45, %47, !dbg !6
  br label %49, !dbg !17

49:                                               ; preds = %131, %44
  %50 = phi i64 [ %226, %131 ], [ 0, %44 ], !dbg !17
  %51 = phi [2 x <32 x float>] [ %225, %131 ], [ zeroinitializer, %44 ], !dbg !17
  %52 = icmp samesign ult i64 %50, 8, !dbg !17
  br i1 %52, label %53, label %227, !dbg !17

53:                                               ; preds = %49
  %54 = shl nuw nsw i64 %50, 4, !dbg !17
  %.idx234 = shl nuw nsw i64 %15, 20, !dbg !15
  %55 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx234, !dbg !15
  %.idx235 = shl nsw i64 %18, 8, !dbg !15
  %56 = getelementptr i8, ptr addrspace(1) %55, i64 %.idx235, !dbg !15
  %57 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %54, !dbg !15
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !15, !range !35, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !16
  %address.local271 = shl nuw nsw i64 %address.local, 1, !dbg !15
  %address.local272 = and i64 %address.local271, 384, !dbg !15
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local272, !dbg !15
  %address.local274 = ptrtoint ptr addrspace(3) %address.local273 to i64, !dbg !15
  %address.local275 = trunc i64 %address.local274 to i32, !dbg !15
  %58 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local275), !dbg !15
  %59 = zext i32 %58 to i64, !dbg !15
  %60 = inttoptr i64 %59 to ptr addrspace(3), !dbg !15
  %address.local276 = shl nuw nsw i32 %address.thread, 1, !dbg !15
  %address.local277 = and i32 %address.local276, 2, !dbg !15
  %address.local278 = shl nuw nsw i32 %address.thread, 6, !dbg !15
  %address.local279 = and i32 %address.local278, 16256, !dbg !15
  %address.local280 = or disjoint i32 %address.local277, %address.local279, !dbg !15
  %address.local281 = zext nneg i32 %address.local280 to i64, !dbg !15
  %address.local282 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local281, !dbg !15
  %dma.source.bits = ptrtoint ptr addrspace(1) %address.local282 to i64, !dbg !15
  %dma.source.opaque = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #9, !dbg !15
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1), !dbg !15
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %60, i32 4, i32 0, i32 0), !dbg !15
  %address.local283 = shl nuw nsw i64 %address.local, 1, !dbg !15
  %address.local284 = and i64 %address.local283, 384, !dbg !15
  %address.local285 = or disjoint i64 %address.local284, 512, !dbg !15
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local285, !dbg !15
  %address.local287 = ptrtoint ptr addrspace(3) %address.local286 to i64, !dbg !15
  %address.local288 = trunc i64 %address.local287 to i32, !dbg !15
  %61 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local288), !dbg !15
  %62 = zext i32 %61 to i64, !dbg !15
  %63 = inttoptr i64 %62 to ptr addrspace(3), !dbg !15
  %address.local289 = shl nuw nsw i32 %address.thread, 1, !dbg !15
  %address.local290 = and i32 %address.local289, 2, !dbg !15
  %address.local291 = shl nuw nsw i32 %address.thread, 6, !dbg !15
  %address.local292 = and i32 %address.local291, 16256, !dbg !15
  %address.local293 = or disjoint i32 %address.local290, %address.local292, !dbg !15
  %address.local294 = zext nneg i32 %address.local293 to i64, !dbg !15
  %address.local295 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local294, !dbg !15
  %address.local296 = getelementptr i8, ptr addrspace(1) %address.local295, i64 8, !dbg !15
  %dma.source.bits985 = ptrtoint ptr addrspace(1) %address.local296 to i64, !dbg !15
  %dma.source.opaque986 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits985) #9, !dbg !15
  %dma.source.ptr987 = inttoptr i64 %dma.source.opaque986 to ptr addrspace(1), !dbg !15
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr987, ptr addrspace(3) %63, i32 4, i32 0, i32 0), !dbg !15
  %address.local297 = shl nuw nsw i64 %address.local, 1, !dbg !15
  %address.local298 = and i64 %address.local297, 384, !dbg !15
  %address.local299 = or disjoint i64 %address.local298, 1024, !dbg !15
  %address.local300 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local299, !dbg !15
  %address.local301 = ptrtoint ptr addrspace(3) %address.local300 to i64, !dbg !15
  %address.local302 = trunc i64 %address.local301 to i32, !dbg !15
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local302), !dbg !15
  %65 = zext i32 %64 to i64, !dbg !15
  %66 = inttoptr i64 %65 to ptr addrspace(3), !dbg !15
  %address.local303 = shl nuw nsw i32 %address.thread, 1, !dbg !15
  %address.local304 = and i32 %address.local303, 2, !dbg !15
  %address.local305 = shl nuw nsw i32 %address.thread, 6, !dbg !15
  %address.local306 = and i32 %address.local305, 16256, !dbg !15
  %address.local307 = or disjoint i32 %address.local304, %address.local306, !dbg !15
  %address.local308 = zext nneg i32 %address.local307 to i64, !dbg !15
  %address.local309 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local308, !dbg !15
  %address.local310 = getelementptr i8, ptr addrspace(1) %address.local309, i64 16, !dbg !15
  %dma.source.bits988 = ptrtoint ptr addrspace(1) %address.local310 to i64, !dbg !15
  %dma.source.opaque989 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits988) #9, !dbg !15
  %dma.source.ptr990 = inttoptr i64 %dma.source.opaque989 to ptr addrspace(1), !dbg !15
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr990, ptr addrspace(3) %66, i32 4, i32 0, i32 0), !dbg !15
  %address.local311 = shl nuw nsw i64 %address.local, 1, !dbg !15
  %address.local312 = and i64 %address.local311, 384, !dbg !15
  %address.local313 = or disjoint i64 %address.local312, 1536, !dbg !15
  %address.local314 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local313, !dbg !15
  %address.local315 = ptrtoint ptr addrspace(3) %address.local314 to i64, !dbg !15
  %address.local316 = trunc i64 %address.local315 to i32, !dbg !15
  %67 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local316), !dbg !15
  %68 = zext i32 %67 to i64, !dbg !15
  %69 = inttoptr i64 %68 to ptr addrspace(3), !dbg !15
  %address.local317 = shl nuw nsw i32 %address.thread, 1, !dbg !15
  %address.local318 = and i32 %address.local317, 2, !dbg !15
  %address.local319 = shl nuw nsw i32 %address.thread, 6, !dbg !15
  %address.local320 = and i32 %address.local319, 16256, !dbg !15
  %address.local321 = or disjoint i32 %address.local318, %address.local320, !dbg !15
  %address.local322 = zext nneg i32 %address.local321 to i64, !dbg !15
  %address.local323 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local322, !dbg !15
  %address.local324 = getelementptr i8, ptr addrspace(1) %address.local323, i64 24, !dbg !15
  %dma.source.bits991 = ptrtoint ptr addrspace(1) %address.local324 to i64, !dbg !15
  %dma.source.opaque992 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits991) #9, !dbg !15
  %dma.source.ptr993 = inttoptr i64 %dma.source.opaque992 to ptr addrspace(1), !dbg !15
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr993, ptr addrspace(3) %69, i32 4, i32 0, i32 0), !dbg !15
  call void @llvm.amdgcn.asyncmark(), !dbg !15
  %70 = shl nuw nsw i64 %50, 9, !dbg !6
  %71 = add nuw i64 %40, %70, !dbg !6
  %.lobit236 = ashr i64 %71, 63, !dbg !6
  %72 = xor i64 %71, %.lobit236, !dbg !6
  %73 = sdiv i64 %72, 4096, !dbg !6
  %.lobit237 = ashr i64 %71, 63, !dbg !6
  %74 = xor i64 %73, %.lobit237, !dbg !6
  %75 = add nsw i64 %74, %8, !dbg !6
  %76 = add nuw nsw i64 %54, %46, !dbg !6
  %77 = and i64 %76, 127, !dbg !6
  %78 = shl i64 %75, 19, !dbg !6
  %79 = and i64 %78, 16252928, !dbg !6
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %80 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %79, !dbg !6
  %81 = getelementptr [2 x i8], ptr addrspace(1) %80, i64 %77, !dbg !6
  %.idx238 = shl i64 %48, 8, !dbg !6
  %82 = getelementptr i8, ptr addrspace(1) %81, i64 %.idx238, !dbg !6
  %address.local325 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local272, !dbg !6
  %address.local326 = ptrtoint ptr addrspace(3) %address.local325 to i64, !dbg !6
  %address.local327 = trunc i64 %address.local326 to i32, !dbg !6
  %83 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local327), !dbg !6
  %84 = zext i32 %83 to i64, !dbg !6
  %85 = inttoptr i64 %84 to ptr addrspace(3), !dbg !6
  %address.local328 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local329 = and i32 %address.local328, 14, !dbg !6
  %address.local330 = shl nuw nsw i32 %address.thread, 4, !dbg !6
  %address.local331 = and i32 %address.local330, 3968, !dbg !6
  %address.local332 = or disjoint i32 %address.local329, %address.local331, !dbg !6
  %address.local333 = zext nneg i32 %address.local332 to i64, !dbg !6
  %address.local334 = getelementptr [2 x i8], ptr addrspace(1) %82, i64 %address.local333, !dbg !6
  %dma.source.bits994 = ptrtoint ptr addrspace(1) %address.local334 to i64, !dbg !6
  %dma.source.opaque995 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits994) #9, !dbg !6
  %dma.source.ptr996 = inttoptr i64 %dma.source.opaque995 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr996, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !6
  %address.local335 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local285, !dbg !6
  %address.local336 = ptrtoint ptr addrspace(3) %address.local335 to i64, !dbg !6
  %address.local337 = trunc i64 %address.local336 to i32, !dbg !6
  %86 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local337), !dbg !6
  %87 = zext i32 %86 to i64, !dbg !6
  %88 = inttoptr i64 %87 to ptr addrspace(3), !dbg !6
  %address.local338 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local339 = and i32 %address.local338, 14, !dbg !6
  %address.local340 = shl nuw nsw i32 %address.thread, 4, !dbg !6
  %address.local341 = and i32 %address.local340, 3968, !dbg !6
  %address.local342 = or disjoint i32 %address.local339, %address.local341, !dbg !6
  %address.local343 = or disjoint i32 %address.local342, 4096, !dbg !6
  %address.local344 = zext nneg i32 %address.local343 to i64, !dbg !6
  %address.local345 = getelementptr [2 x i8], ptr addrspace(1) %82, i64 %address.local344, !dbg !6
  %dma.source.bits997 = ptrtoint ptr addrspace(1) %address.local345 to i64, !dbg !6
  %dma.source.opaque998 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits997) #9, !dbg !6
  %dma.source.ptr999 = inttoptr i64 %dma.source.opaque998 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr999, ptr addrspace(3) %88, i32 4, i32 0, i32 0), !dbg !6
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local299, !dbg !6
  %address.local347 = ptrtoint ptr addrspace(3) %address.local346 to i64, !dbg !6
  %address.local348 = trunc i64 %address.local347 to i32, !dbg !6
  %89 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local348), !dbg !6
  %90 = zext i32 %89 to i64, !dbg !6
  %91 = inttoptr i64 %90 to ptr addrspace(3), !dbg !6
  %address.local349 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local350 = and i32 %address.local349, 14, !dbg !6
  %address.local351 = shl nuw nsw i32 %address.thread, 4, !dbg !6
  %address.local352 = and i32 %address.local351, 3968, !dbg !6
  %address.local353 = or disjoint i32 %address.local350, %address.local352, !dbg !6
  %address.local354 = or disjoint i32 %address.local353, 8192, !dbg !6
  %address.local355 = zext nneg i32 %address.local354 to i64, !dbg !6
  %address.local356 = getelementptr [2 x i8], ptr addrspace(1) %82, i64 %address.local355, !dbg !6
  %dma.source.bits1000 = ptrtoint ptr addrspace(1) %address.local356 to i64, !dbg !6
  %dma.source.opaque1001 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1000) #9, !dbg !6
  %dma.source.ptr1002 = inttoptr i64 %dma.source.opaque1001 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1002, ptr addrspace(3) %91, i32 4, i32 0, i32 0), !dbg !6
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local313, !dbg !6
  %address.local358 = ptrtoint ptr addrspace(3) %address.local357 to i64, !dbg !6
  %address.local359 = trunc i64 %address.local358 to i32, !dbg !6
  %92 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local359), !dbg !6
  %93 = zext i32 %92 to i64, !dbg !6
  %94 = inttoptr i64 %93 to ptr addrspace(3), !dbg !6
  %address.local360 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local361 = and i32 %address.local360, 14, !dbg !6
  %address.local362 = shl nuw nsw i32 %address.thread, 4, !dbg !6
  %address.local363 = and i32 %address.local362, 3968, !dbg !6
  %address.local364 = or disjoint i32 %address.local361, %address.local363, !dbg !6
  %address.local365 = or disjoint i32 %address.local364, 12288, !dbg !6
  %address.local366 = zext nneg i32 %address.local365 to i64, !dbg !6
  %address.local367 = getelementptr [2 x i8], ptr addrspace(1) %82, i64 %address.local366, !dbg !6
  %dma.source.bits1003 = ptrtoint ptr addrspace(1) %address.local367 to i64, !dbg !6
  %dma.source.opaque1004 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1003) #9, !dbg !6
  %dma.source.ptr1005 = inttoptr i64 %dma.source.opaque1004 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1005, ptr addrspace(3) %94, i32 4, i32 0, i32 0), !dbg !6
  call void @llvm.amdgcn.asyncmark(), !dbg !6
  %address.local368 = and i32 %address.thread, 192, !dbg !16
  %address.local369 = shl nuw nsw i32 %address.thread, 2, !dbg !16
  %address.local370 = and i32 %address.local369, 60, !dbg !16
  %address.local371 = or disjoint i32 %address.local368, %address.local370, !dbg !16
  %address.local372 = shl nuw nsw i32 %address.thread, 5, !dbg !16
  %address.local373 = and i32 %address.local372, 1536, !dbg !16
  %address.local374 = or disjoint i32 %address.local371, %address.local373, !dbg !16
  %address.local375 = zext nneg i32 %address.local374 to i64, !dbg !16
  %address.local376 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local375, !dbg !16
  %95 = load <4 x half>, ptr addrspace(3) %address.local376, align 2, !dbg !16
  %address.local377 = and i32 %address.thread, 192, !dbg !16
  %address.local378 = shl nuw nsw i32 %address.thread, 2, !dbg !16
  %address.local379 = and i32 %address.local378, 60, !dbg !16
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !16
  %address.local381 = shl nuw nsw i32 %address.thread, 5, !dbg !16
  %address.local382 = and i32 %address.local381, 1536, !dbg !16
  %address.local383 = or disjoint i32 %address.local380, %address.local382, !dbg !16
  %address.local384 = zext nneg i32 %address.local383 to i64, !dbg !16
  %address.local385 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local384, !dbg !16
  %address.local386 = getelementptr i8, ptr addrspace(3) %address.local385, i64 512, !dbg !16
  %96 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !16
  %97 = fmul <4 x half> %95, splat (half 0xH3015), !dbg !16
  %98 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !16
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !6
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  br label %99, !dbg !6

99:                                               ; preds = %102, %53
  %100 = phi i64 [ %130, %102 ], [ %20, %53 ], !dbg !6
  %101 = icmp samesign ult i64 %100, 1024, !dbg !6
  br i1 %101, label %102, label %131, !dbg !6

102:                                              ; preds = %99
  %103 = shl nuw nsw i64 %100, 1, !dbg !6
  %104 = lshr i64 %100, 6, !dbg !6
  %105 = and i64 %104, 14, !dbg !6
  %106 = lshr i64 %100, 2, !dbg !6
  %107 = and i64 %106, 1, !dbg !6
  %108 = and i64 %100, 120, !dbg !6
  %109 = and i64 %103, 6, !dbg !6
  %110 = or disjoint i64 %108, %109, !dbg !6
  %111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %105, !dbg !6
  %112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %111, i64 %107, !dbg !6
  %.idx239 = shl nuw nsw i64 %110, 5, !dbg !6
  %113 = getelementptr inbounds nuw i8, ptr addrspace(3) %112, i64 %.idx239, !dbg !6
  %114 = load half, ptr addrspace(3) %113, align 2, !dbg !6
  %115 = lshr i64 %100, 6, !dbg !6
  %116 = and i64 %115, 14, !dbg !6
  %117 = lshr i64 %100, 2, !dbg !6
  %118 = and i64 %117, 1, !dbg !6
  %119 = and i64 %100, 120, !dbg !6
  %120 = and i64 %103, 6, !dbg !6
  %121 = or disjoint i64 %120, 1, !dbg !6
  %122 = or disjoint i64 %119, %121, !dbg !6
  %123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %116, !dbg !6
  %124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %123, i64 %118, !dbg !6
  %.idx240 = shl nuw nsw i64 %122, 5, !dbg !6
  %125 = getelementptr inbounds nuw i8, ptr addrspace(3) %124, i64 %.idx240, !dbg !6
  %126 = load half, ptr addrspace(3) %125, align 2, !dbg !6
  %127 = insertelement <2 x half> poison, half %114, i64 0, !dbg !6
  %128 = insertelement <2 x half> %127, half %126, i64 1, !dbg !6
  %129 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %103, !dbg !6
  store <2 x half> %128, ptr addrspace(3) %129, align 2, !dbg !6
  %130 = add nuw nsw i64 %100, 256, !dbg !6
  br label %99, !dbg !6

131:                                              ; preds = %99
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %address.thread387 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !9, !range !35, !frisk.register_lifetimes !1
  %address.local388 = shl nuw nsw i32 %address.thread387, 1, !dbg !9
  %address.local389 = and i32 %address.local388, 8, !dbg !9
  %address.local390 = shl nuw nsw i32 %address.thread387, 4, !dbg !9
  %address.local391 = and i32 %address.local390, 48, !dbg !9
  %address.local392 = or disjoint i32 %address.local389, %address.local391, !dbg !9
  %address.local393 = shl nuw nsw i32 %address.thread387, 5, !dbg !9
  %address.local394 = and i32 %address.local393, 1792, !dbg !9
  %address.local395 = or disjoint i32 %address.local392, %address.local394, !dbg !9
  %address.local396 = zext nneg i32 %address.local395 to i64, !dbg !9
  %address.local397 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local396, !dbg !9
  %132 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local397, i32 0) #0, !dbg !9
  %133 = bitcast <4 x i32> %132 to <8 x half>, !dbg !9
  %address.local398 = shl nuw nsw i32 %address.thread387, 1, !dbg !9
  %address.local399 = and i32 %address.local398, 8, !dbg !9
  %address.local400 = shl nuw nsw i32 %address.thread387, 4, !dbg !9
  %address.local401 = and i32 %address.local400, 48, !dbg !9
  %address.local402 = or disjoint i32 %address.local399, %address.local401, !dbg !9
  %address.local403 = shl nuw nsw i32 %address.thread387, 5, !dbg !9
  %address.local404 = and i32 %address.local403, 1792, !dbg !9
  %address.local405 = or disjoint i32 %address.local402, %address.local404, !dbg !9
  %address.local406 = zext nneg i32 %address.local405 to i64, !dbg !9
  %address.local407 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local406, !dbg !9
  %address.local408 = getelementptr i8, ptr addrspace(3) %address.local407, i64 128, !dbg !9
  %134 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local408, i32 0) #0, !dbg !9
  %135 = bitcast <4 x i32> %134 to <8 x half>, !dbg !9
  %address.local409 = shl nuw nsw i32 %address.thread387, 1, !dbg !9
  %address.local410 = and i32 %address.local409, 8, !dbg !9
  %address.local411 = shl nuw nsw i32 %address.thread387, 4, !dbg !9
  %address.local412 = and i32 %address.local411, 48, !dbg !9
  %address.local413 = or disjoint i32 %address.local410, %address.local412, !dbg !9
  %address.local414 = shl nuw nsw i32 %address.thread387, 5, !dbg !9
  %address.local415 = and i32 %address.local414, 1792, !dbg !9
  %address.local416 = or disjoint i32 %address.local413, %address.local415, !dbg !9
  %address.local417 = zext nneg i32 %address.local416 to i64, !dbg !9
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local417, !dbg !9
  %address.local419 = getelementptr i8, ptr addrspace(3) %address.local418, i64 256, !dbg !9
  %136 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local419, i32 0) #0, !dbg !9
  %137 = bitcast <4 x i32> %136 to <8 x half>, !dbg !9
  %address.local420 = shl nuw nsw i32 %address.thread387, 1, !dbg !9
  %address.local421 = and i32 %address.local420, 8, !dbg !9
  %address.local422 = shl nuw nsw i32 %address.thread387, 4, !dbg !9
  %address.local423 = and i32 %address.local422, 48, !dbg !9
  %address.local424 = or disjoint i32 %address.local421, %address.local423, !dbg !9
  %address.local425 = shl nuw nsw i32 %address.thread387, 5, !dbg !9
  %address.local426 = and i32 %address.local425, 1792, !dbg !9
  %address.local427 = or disjoint i32 %address.local424, %address.local426, !dbg !9
  %address.local428 = zext nneg i32 %address.local427 to i64, !dbg !9
  %address.local429 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local428, !dbg !9
  %address.local430 = getelementptr i8, ptr addrspace(3) %address.local429, i64 384, !dbg !9
  %138 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local430, i32 0) #0, !dbg !9
  %139 = bitcast <4 x i32> %138 to <8 x half>, !dbg !9
  %140 = shufflevector <8 x half> %133, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !9
  %141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %140, <4 x float> zeroinitializer) #8, !dbg !9
  %142 = extractvalue [2 x <32 x float>] %51, 0, !dbg !32
  %143 = shufflevector <32 x float> %142, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %144 = fadd <4 x float> %143, %141, !dbg !32
  %145 = shufflevector <8 x half> %133, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !9
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %145, <4 x float> zeroinitializer) #8, !dbg !9
  %147 = shufflevector <32 x float> %142, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %148 = fadd <4 x float> %147, %146, !dbg !32
  %149 = shufflevector <8 x half> %135, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !9
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %149, <4 x float> zeroinitializer) #8, !dbg !9
  %151 = shufflevector <32 x float> %142, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !32
  %152 = fadd <4 x float> %151, %150, !dbg !32
  %153 = shufflevector <4 x float> %152, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %154 = shufflevector <8 x half> %135, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !9
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %154, <4 x float> zeroinitializer) #8, !dbg !9
  %156 = shufflevector <32 x float> %142, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !32
  %157 = fadd <4 x float> %156, %155, !dbg !32
  %158 = shufflevector <4 x float> %157, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %159 = shufflevector <8 x half> %137, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !9
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %159, <4 x float> zeroinitializer) #8, !dbg !9
  %161 = shufflevector <32 x float> %142, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !32
  %162 = fadd <4 x float> %161, %160, !dbg !32
  %163 = shufflevector <4 x float> %162, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %164 = shufflevector <8 x half> %137, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !9
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %164, <4 x float> zeroinitializer) #8, !dbg !9
  %166 = shufflevector <32 x float> %142, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !32
  %167 = fadd <4 x float> %166, %165, !dbg !32
  %168 = shufflevector <4 x float> %167, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %169 = shufflevector <8 x half> %139, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !9
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %169, <4 x float> zeroinitializer) #8, !dbg !9
  %171 = shufflevector <32 x float> %142, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !32
  %172 = fadd <4 x float> %171, %170, !dbg !32
  %173 = shufflevector <4 x float> %172, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %174 = shufflevector <8 x half> %139, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !9
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %174, <4 x float> zeroinitializer) #8, !dbg !9
  %176 = shufflevector <32 x float> %142, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !32
  %177 = fadd <4 x float> %176, %175, !dbg !32
  %178 = shufflevector <4 x float> %177, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %140, <4 x float> zeroinitializer) #8, !dbg !9
  %180 = extractvalue [2 x <32 x float>] %51, 1, !dbg !32
  %181 = shufflevector <32 x float> %180, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %182 = fadd <4 x float> %181, %179, !dbg !32
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %145, <4 x float> zeroinitializer) #8, !dbg !9
  %184 = shufflevector <32 x float> %180, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %185 = fadd <4 x float> %184, %183, !dbg !32
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %149, <4 x float> zeroinitializer) #8, !dbg !9
  %187 = shufflevector <32 x float> %180, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !32
  %188 = fadd <4 x float> %187, %186, !dbg !32
  %189 = shufflevector <4 x float> %188, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %154, <4 x float> zeroinitializer) #8, !dbg !9
  %191 = shufflevector <32 x float> %180, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !32
  %192 = fadd <4 x float> %191, %190, !dbg !32
  %193 = shufflevector <4 x float> %192, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %159, <4 x float> zeroinitializer) #8, !dbg !9
  %195 = shufflevector <32 x float> %180, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !32
  %196 = fadd <4 x float> %195, %194, !dbg !32
  %197 = shufflevector <4 x float> %196, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %164, <4 x float> zeroinitializer) #8, !dbg !9
  %199 = shufflevector <32 x float> %180, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !32
  %200 = fadd <4 x float> %199, %198, !dbg !32
  %201 = shufflevector <4 x float> %200, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %169, <4 x float> zeroinitializer) #8, !dbg !9
  %203 = shufflevector <32 x float> %180, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !32
  %204 = fadd <4 x float> %203, %202, !dbg !32
  %205 = shufflevector <4 x float> %204, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %174, <4 x float> zeroinitializer) #8, !dbg !9
  %207 = shufflevector <32 x float> %180, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !32
  %208 = fadd <4 x float> %207, %206, !dbg !32
  %209 = shufflevector <4 x float> %208, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %210 = shufflevector <4 x float> %144, <4 x float> %148, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %211 = shufflevector <32 x float> %210, <32 x float> %153, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %212 = shufflevector <32 x float> %211, <32 x float> %158, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %213 = shufflevector <32 x float> %212, <32 x float> %163, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %214 = shufflevector <32 x float> %213, <32 x float> %168, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %215 = shufflevector <32 x float> %214, <32 x float> %173, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %216 = shufflevector <32 x float> %215, <32 x float> %178, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !17
  %217 = insertvalue [2 x <32 x float>] poison, <32 x float> %216, 0, !dbg !17
  %218 = shufflevector <4 x float> %182, <4 x float> %185, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %219 = shufflevector <32 x float> %218, <32 x float> %189, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %220 = shufflevector <32 x float> %219, <32 x float> %193, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %221 = shufflevector <32 x float> %220, <32 x float> %197, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %222 = shufflevector <32 x float> %221, <32 x float> %201, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %223 = shufflevector <32 x float> %222, <32 x float> %205, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !17
  %224 = shufflevector <32 x float> %223, <32 x float> %209, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !17
  %225 = insertvalue [2 x <32 x float>] %217, <32 x float> %224, 1, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %226 = add nuw nsw i64 %50, 1, !dbg !17
  br label %49, !dbg !17

227:                                              ; preds = %49
  %228 = add nsw i64 %46, %8, !dbg !7
  %229 = shl i64 %228, 19, !dbg !7
  %230 = and i64 %229, 16252928, !dbg !7
  %231 = shl nsw i64 %48, 7, !dbg !7
  %232 = add i64 %230, %231, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %233 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %232, !dbg !7
  %234 = getelementptr [2 x i8], ptr addrspace(1) %233, i64 %38, !dbg !7
  %address.thread431 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !7, !range !35, !frisk.register_lifetimes !1
  %address.local432 = zext nneg i32 %address.thread431 to i64, !dbg !16
  %address.local433 = shl nuw nsw i64 %address.local432, 1, !dbg !15
  %address.local434 = and i64 %address.local433, 384, !dbg !15
  %address.local435 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %address.local434, !dbg !7
  %address.local436 = ptrtoint ptr addrspace(3) %address.local435 to i64, !dbg !7
  %address.local437 = trunc i64 %address.local436 to i32, !dbg !7
  %235 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local437), !dbg !7
  %236 = zext i32 %235 to i64, !dbg !7
  %237 = inttoptr i64 %236 to ptr addrspace(3), !dbg !7
  %address.local438 = shl nuw nsw i32 %address.thread431, 1, !dbg !6
  %address.local439 = and i32 %address.local438, 14, !dbg !6
  %address.local440 = shl nuw nsw i32 %address.thread431, 4, !dbg !6
  %address.local441 = and i32 %address.local440, 3968, !dbg !6
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !6
  %address.local443 = zext nneg i32 %address.local442 to i64, !dbg !6
  %address.local444 = getelementptr [2 x i8], ptr addrspace(1) %234, i64 %address.local443, !dbg !7
  %dma.source.bits1006 = ptrtoint ptr addrspace(1) %address.local444 to i64, !dbg !7
  %dma.source.opaque1007 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1006) #9, !dbg !7
  %dma.source.ptr1008 = inttoptr i64 %dma.source.opaque1007 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1008, ptr addrspace(3) %237, i32 4, i32 0, i32 0), !dbg !7
  %address.local445 = shl nuw nsw i64 %address.local432, 1, !dbg !15
  %address.local446 = and i64 %address.local445, 384, !dbg !15
  %address.local447 = or disjoint i64 %address.local446, 512, !dbg !15
  %address.local448 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %address.local447, !dbg !7
  %address.local449 = ptrtoint ptr addrspace(3) %address.local448 to i64, !dbg !7
  %address.local450 = trunc i64 %address.local449 to i32, !dbg !7
  %238 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local450), !dbg !7
  %239 = zext i32 %238 to i64, !dbg !7
  %240 = inttoptr i64 %239 to ptr addrspace(3), !dbg !7
  %address.local451 = shl nuw nsw i32 %address.thread431, 1, !dbg !6
  %address.local452 = and i32 %address.local451, 14, !dbg !6
  %address.local453 = shl nuw nsw i32 %address.thread431, 4, !dbg !6
  %address.local454 = and i32 %address.local453, 3968, !dbg !6
  %address.local455 = or disjoint i32 %address.local452, %address.local454, !dbg !6
  %address.local456 = or disjoint i32 %address.local455, 4096, !dbg !6
  %address.local457 = zext nneg i32 %address.local456 to i64, !dbg !6
  %address.local458 = getelementptr [2 x i8], ptr addrspace(1) %234, i64 %address.local457, !dbg !7
  %dma.source.bits1009 = ptrtoint ptr addrspace(1) %address.local458 to i64, !dbg !7
  %dma.source.opaque1010 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1009) #9, !dbg !7
  %dma.source.ptr1011 = inttoptr i64 %dma.source.opaque1010 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1011, ptr addrspace(3) %240, i32 4, i32 0, i32 0), !dbg !7
  %address.local459 = shl nuw nsw i64 %address.local432, 1, !dbg !15
  %address.local460 = and i64 %address.local459, 384, !dbg !15
  %address.local461 = or disjoint i64 %address.local460, 1024, !dbg !15
  %address.local462 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %address.local461, !dbg !7
  %address.local463 = ptrtoint ptr addrspace(3) %address.local462 to i64, !dbg !7
  %address.local464 = trunc i64 %address.local463 to i32, !dbg !7
  %241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local464), !dbg !7
  %242 = zext i32 %241 to i64, !dbg !7
  %243 = inttoptr i64 %242 to ptr addrspace(3), !dbg !7
  %address.local465 = shl nuw nsw i32 %address.thread431, 1, !dbg !6
  %address.local466 = and i32 %address.local465, 14, !dbg !6
  %address.local467 = shl nuw nsw i32 %address.thread431, 4, !dbg !6
  %address.local468 = and i32 %address.local467, 3968, !dbg !6
  %address.local469 = or disjoint i32 %address.local466, %address.local468, !dbg !6
  %address.local470 = or disjoint i32 %address.local469, 8192, !dbg !6
  %address.local471 = zext nneg i32 %address.local470 to i64, !dbg !6
  %address.local472 = getelementptr [2 x i8], ptr addrspace(1) %234, i64 %address.local471, !dbg !7
  %dma.source.bits1012 = ptrtoint ptr addrspace(1) %address.local472 to i64, !dbg !7
  %dma.source.opaque1013 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1012) #9, !dbg !7
  %dma.source.ptr1014 = inttoptr i64 %dma.source.opaque1013 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1014, ptr addrspace(3) %243, i32 4, i32 0, i32 0), !dbg !7
  %address.local473 = shl nuw nsw i64 %address.local432, 1, !dbg !15
  %address.local474 = and i64 %address.local473, 384, !dbg !15
  %address.local475 = or disjoint i64 %address.local474, 1536, !dbg !15
  %address.local476 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %address.local475, !dbg !7
  %address.local477 = ptrtoint ptr addrspace(3) %address.local476 to i64, !dbg !7
  %address.local478 = trunc i64 %address.local477 to i32, !dbg !7
  %244 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local478), !dbg !7
  %245 = zext i32 %244 to i64, !dbg !7
  %246 = inttoptr i64 %245 to ptr addrspace(3), !dbg !7
  %address.local479 = shl nuw nsw i32 %address.thread431, 1, !dbg !6
  %address.local480 = and i32 %address.local479, 14, !dbg !6
  %address.local481 = shl nuw nsw i32 %address.thread431, 4, !dbg !6
  %address.local482 = and i32 %address.local481, 3968, !dbg !6
  %address.local483 = or disjoint i32 %address.local480, %address.local482, !dbg !6
  %address.local484 = or disjoint i32 %address.local483, 12288, !dbg !6
  %address.local485 = zext nneg i32 %address.local484 to i64, !dbg !6
  %address.local486 = getelementptr [2 x i8], ptr addrspace(1) %234, i64 %address.local485, !dbg !7
  %dma.source.bits1015 = ptrtoint ptr addrspace(1) %address.local486 to i64, !dbg !7
  %dma.source.opaque1016 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1015) #9, !dbg !7
  %dma.source.ptr1017 = inttoptr i64 %dma.source.opaque1016 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1017, ptr addrspace(3) %246, i32 4, i32 0, i32 0), !dbg !7
  call void @llvm.amdgcn.asyncmark(), !dbg !7
  %247 = extractvalue [2 x <32 x float>] %51, 0, !dbg !17
  %248 = extractelement <32 x float> %247, i64 0, !dbg !17
  %249 = extractelement <32 x float> %247, i64 1, !dbg !17
  %250 = extractelement <32 x float> %247, i64 2, !dbg !17
  %251 = extractelement <32 x float> %247, i64 3, !dbg !17
  %252 = select i1 %.not, float %249, float %248, !dbg !17
  %253 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !17
  %254 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %253), !dbg !17
  %255 = add nuw nsw i32 %254, 64, !dbg !17
  %256 = xor i32 %254, 16, !dbg !17
  %257 = and i32 %255, 192, !dbg !17
  %258 = icmp samesign ult i32 %256, %257, !dbg !17
  %259 = select i1 %258, i32 %256, i32 %254, !dbg !17
  %260 = shl nuw nsw i32 %259, 2, !dbg !17
  %261 = bitcast float %252 to i32, !dbg !17
  %262 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %261), !dbg !17
  %263 = bitcast i32 %262 to float, !dbg !17
  %264 = select i1 %.not, float %248, float %263, !dbg !17
  %265 = select i1 %.not, float %263, float %249, !dbg !17
  %266 = select i1 %.not, float %251, float %250, !dbg !17
  %267 = bitcast float %266 to i32, !dbg !17
  %268 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %267), !dbg !17
  %269 = bitcast i32 %268 to float, !dbg !17
  %270 = select i1 %.not, float %250, float %269, !dbg !17
  %271 = select i1 %.not, float %269, float %251, !dbg !17
  %272 = select i1 %.not155, float %270, float %264, !dbg !17
  %273 = xor i32 %254, 32, !dbg !17
  %274 = icmp samesign ult i32 %273, %257, !dbg !17
  %275 = select i1 %274, i32 %273, i32 %254, !dbg !17
  %276 = shl nuw nsw i32 %275, 2, !dbg !17
  %277 = bitcast float %272 to i32, !dbg !17
  %278 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %277), !dbg !17
  %279 = bitcast i32 %278 to float, !dbg !17
  %280 = select i1 %.not155, float %264, float %279, !dbg !17
  %281 = select i1 %.not155, float %279, float %270, !dbg !17
  %282 = select i1 %.not155, float %271, float %265, !dbg !17
  %283 = bitcast float %282 to i32, !dbg !17
  %284 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %283), !dbg !17
  %285 = bitcast i32 %284 to float, !dbg !17
  %286 = select i1 %.not155, float %265, float %285, !dbg !17
  %287 = select i1 %.not155, float %285, float %271, !dbg !17
  %288 = or disjoint i64 %45, %index.compact, !dbg !20
  %289 = lshr i64 %32, 32, !dbg !26
  %compare.high = trunc i64 %289 to i32, !dbg !26
  %290 = lshr i64 %288, 32, !dbg !26
  %compare.high729 = trunc i64 %290 to i32, !dbg !26
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high729, !dbg !26
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high729, !dbg !26
  %compare.low.base = trunc i64 %32 to i32, !dbg !26
  %compare.low.base730 = trunc i64 %288 to i32, !dbg !26
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base730, !dbg !26
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !26
  %291 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %292 = fadd float %280, %291, !dbg !20
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 1) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !26
  %compare.low731 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !26
  %compare.result732 = select i1 %compare.high.equal, i1 %compare.low731, i1 %compare.high.order, !dbg !26
  %293 = select i1 %compare.result732, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %294 = fadd float %286, %293, !dbg !20
  %compare.local.base733 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 2) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local734 = or i32 %compare.local.base733, 2, !dbg !26
  %compare.low735 = icmp ugt i32 %compare.low.base, %compare.local734, !dbg !26
  %compare.result736 = select i1 %compare.high.equal, i1 %compare.low735, i1 %compare.high.order, !dbg !26
  %295 = select i1 %compare.result736, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %296 = fadd float %281, %295, !dbg !20
  %compare.local.base737 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 3) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local738 = or i32 %compare.local.base737, 3, !dbg !26
  %compare.low739 = icmp ugt i32 %compare.low.base, %compare.local738, !dbg !26
  %compare.result740 = select i1 %compare.high.equal, i1 %compare.low739, i1 %compare.high.order, !dbg !26
  %297 = select i1 %compare.result740, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %298 = fadd float %287, %297, !dbg !20
  %299 = call float @__ocml_exp2_f32(float %292), !dbg !28
  %300 = insertelement <4 x float> poison, float %299, i64 0, !dbg !30
  %301 = fadd float %299, 0x0000000000000000, !dbg !29
  %302 = call float @__ocml_exp2_f32(float %294), !dbg !28
  %303 = insertelement <4 x float> %300, float %302, i64 1, !dbg !30
  %304 = fadd float %301, %302, !dbg !29
  %305 = call float @__ocml_exp2_f32(float %296), !dbg !28
  %306 = insertelement <4 x float> %303, float %305, i64 2, !dbg !30
  %307 = fadd float %304, %305, !dbg !29
  %308 = call float @__ocml_exp2_f32(float %298), !dbg !28
  %309 = insertelement <4 x float> %306, float %308, i64 3, !dbg !30
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %309) #8, !dbg !30, !frisk.register_lifetimes !1
  %310 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !30
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %310) #8, !dbg !29, !frisk.register_lifetimes !1
  %311 = fadd float %307, %308, !dbg !29
  %312 = extractelement <32 x float> %247, i64 4, !dbg !17
  %313 = extractelement <32 x float> %247, i64 5, !dbg !17
  %314 = extractelement <32 x float> %247, i64 6, !dbg !17
  %315 = extractelement <32 x float> %247, i64 7, !dbg !17
  %316 = select i1 %.not, float %313, float %312, !dbg !17
  %317 = bitcast float %316 to i32, !dbg !17
  %318 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %317), !dbg !17
  %319 = bitcast i32 %318 to float, !dbg !17
  %320 = select i1 %.not, float %312, float %319, !dbg !17
  %321 = select i1 %.not, float %319, float %313, !dbg !17
  %322 = select i1 %.not, float %315, float %314, !dbg !17
  %323 = bitcast float %322 to i32, !dbg !17
  %324 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %323), !dbg !17
  %325 = bitcast i32 %324 to float, !dbg !17
  %326 = select i1 %.not, float %314, float %325, !dbg !17
  %327 = select i1 %.not, float %325, float %315, !dbg !17
  %328 = select i1 %.not155, float %326, float %320, !dbg !17
  %329 = bitcast float %328 to i32, !dbg !17
  %330 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %329), !dbg !17
  %331 = bitcast i32 %330 to float, !dbg !17
  %332 = select i1 %.not155, float %320, float %331, !dbg !17
  %333 = select i1 %.not155, float %331, float %326, !dbg !17
  %334 = select i1 %.not155, float %327, float %321, !dbg !17
  %335 = bitcast float %334 to i32, !dbg !17
  %336 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %335), !dbg !17
  %337 = bitcast i32 %336 to float, !dbg !17
  %338 = select i1 %.not155, float %321, float %337, !dbg !17
  %339 = select i1 %.not155, float %337, float %327, !dbg !17
  %compare.local.base741 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 4) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local742 = or i32 %compare.local.base741, 16, !dbg !26
  %compare.low743 = icmp ugt i32 %compare.low.base, %compare.local742, !dbg !26
  %compare.result744 = select i1 %compare.high.equal, i1 %compare.low743, i1 %compare.high.order, !dbg !26
  %340 = select i1 %compare.result744, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %341 = fadd float %332, %340, !dbg !20
  %compare.local.base745 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 5) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local746 = or i32 %compare.local.base745, 17, !dbg !26
  %compare.low747 = icmp ugt i32 %compare.low.base, %compare.local746, !dbg !26
  %compare.result748 = select i1 %compare.high.equal, i1 %compare.low747, i1 %compare.high.order, !dbg !26
  %342 = select i1 %compare.result748, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %343 = fadd float %338, %342, !dbg !20
  %compare.local.base749 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 6) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local750 = or i32 %compare.local.base749, 18, !dbg !26
  %compare.low751 = icmp ugt i32 %compare.low.base, %compare.local750, !dbg !26
  %compare.result752 = select i1 %compare.high.equal, i1 %compare.low751, i1 %compare.high.order, !dbg !26
  %344 = select i1 %compare.result752, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %345 = fadd float %333, %344, !dbg !20
  %compare.local.base753 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 7) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local754 = or i32 %compare.local.base753, 19, !dbg !26
  %compare.low755 = icmp ugt i32 %compare.low.base, %compare.local754, !dbg !26
  %compare.result756 = select i1 %compare.high.equal, i1 %compare.low755, i1 %compare.high.order, !dbg !26
  %346 = select i1 %compare.result756, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %347 = fadd float %339, %346, !dbg !20
  %348 = call float @__ocml_exp2_f32(float %341), !dbg !28
  %349 = insertelement <4 x float> poison, float %348, i64 0, !dbg !30
  %350 = fadd float %311, %348, !dbg !29
  %351 = call float @__ocml_exp2_f32(float %343), !dbg !28
  %352 = insertelement <4 x float> %349, float %351, i64 1, !dbg !30
  %353 = fadd float %350, %351, !dbg !29
  %354 = call float @__ocml_exp2_f32(float %345), !dbg !28
  %355 = insertelement <4 x float> %352, float %354, i64 2, !dbg !30
  %356 = fadd float %353, %354, !dbg !29
  %357 = call float @__ocml_exp2_f32(float %347), !dbg !28
  %358 = insertelement <4 x float> %355, float %357, i64 3, !dbg !30
  %math.result241 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %358) #8, !dbg !30, !frisk.register_lifetimes !1
  %359 = fptrunc <4 x float> %math.result241 to <4 x half>, !dbg !30
  %math.packed242 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %359) #8, !dbg !29, !frisk.register_lifetimes !1
  %360 = fadd float %356, %357, !dbg !29
  %361 = extractelement <32 x float> %247, i64 8, !dbg !17
  %362 = extractelement <32 x float> %247, i64 9, !dbg !17
  %363 = extractelement <32 x float> %247, i64 10, !dbg !17
  %364 = extractelement <32 x float> %247, i64 11, !dbg !17
  %365 = select i1 %.not, float %362, float %361, !dbg !17
  %366 = bitcast float %365 to i32, !dbg !17
  %367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %366), !dbg !17
  %368 = bitcast i32 %367 to float, !dbg !17
  %369 = select i1 %.not, float %361, float %368, !dbg !17
  %370 = select i1 %.not, float %368, float %362, !dbg !17
  %371 = select i1 %.not, float %364, float %363, !dbg !17
  %372 = bitcast float %371 to i32, !dbg !17
  %373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %372), !dbg !17
  %374 = bitcast i32 %373 to float, !dbg !17
  %375 = select i1 %.not, float %363, float %374, !dbg !17
  %376 = select i1 %.not, float %374, float %364, !dbg !17
  %377 = select i1 %.not155, float %375, float %369, !dbg !17
  %378 = bitcast float %377 to i32, !dbg !17
  %379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %378), !dbg !17
  %380 = bitcast i32 %379 to float, !dbg !17
  %381 = select i1 %.not155, float %369, float %380, !dbg !17
  %382 = select i1 %.not155, float %380, float %375, !dbg !17
  %383 = select i1 %.not155, float %376, float %370, !dbg !17
  %384 = bitcast float %383 to i32, !dbg !17
  %385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %384), !dbg !17
  %386 = bitcast i32 %385 to float, !dbg !17
  %387 = select i1 %.not155, float %370, float %386, !dbg !17
  %388 = select i1 %.not155, float %386, float %376, !dbg !17
  %compare.local.base757 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 8) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local758 = or i32 %compare.local.base757, 32, !dbg !26
  %compare.low759 = icmp ugt i32 %compare.low.base, %compare.local758, !dbg !26
  %compare.result760 = select i1 %compare.high.equal, i1 %compare.low759, i1 %compare.high.order, !dbg !26
  %389 = select i1 %compare.result760, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %390 = fadd float %381, %389, !dbg !20
  %compare.local.base761 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 9) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local762 = or i32 %compare.local.base761, 33, !dbg !26
  %compare.low763 = icmp ugt i32 %compare.low.base, %compare.local762, !dbg !26
  %compare.result764 = select i1 %compare.high.equal, i1 %compare.low763, i1 %compare.high.order, !dbg !26
  %391 = select i1 %compare.result764, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %392 = fadd float %387, %391, !dbg !20
  %compare.local.base765 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 10) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local766 = or i32 %compare.local.base765, 34, !dbg !26
  %compare.low767 = icmp ugt i32 %compare.low.base, %compare.local766, !dbg !26
  %compare.result768 = select i1 %compare.high.equal, i1 %compare.low767, i1 %compare.high.order, !dbg !26
  %393 = select i1 %compare.result768, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %394 = fadd float %382, %393, !dbg !20
  %compare.local.base769 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 11) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local770 = or i32 %compare.local.base769, 35, !dbg !26
  %compare.low771 = icmp ugt i32 %compare.low.base, %compare.local770, !dbg !26
  %compare.result772 = select i1 %compare.high.equal, i1 %compare.low771, i1 %compare.high.order, !dbg !26
  %395 = select i1 %compare.result772, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %396 = fadd float %388, %395, !dbg !20
  %397 = call float @__ocml_exp2_f32(float %390), !dbg !28
  %398 = insertelement <4 x float> poison, float %397, i64 0, !dbg !30
  %399 = fadd float %360, %397, !dbg !29
  %400 = call float @__ocml_exp2_f32(float %392), !dbg !28
  %401 = insertelement <4 x float> %398, float %400, i64 1, !dbg !30
  %402 = fadd float %399, %400, !dbg !29
  %403 = call float @__ocml_exp2_f32(float %394), !dbg !28
  %404 = insertelement <4 x float> %401, float %403, i64 2, !dbg !30
  %405 = fadd float %402, %403, !dbg !29
  %406 = call float @__ocml_exp2_f32(float %396), !dbg !28
  %407 = insertelement <4 x float> %404, float %406, i64 3, !dbg !30
  %math.result243 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %407) #8, !dbg !30, !frisk.register_lifetimes !1
  %408 = fptrunc <4 x float> %math.result243 to <4 x half>, !dbg !30
  %math.packed244 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %408) #8, !dbg !29, !frisk.register_lifetimes !1
  %409 = fadd float %405, %406, !dbg !29
  %410 = extractelement <32 x float> %247, i64 12, !dbg !17
  %411 = extractelement <32 x float> %247, i64 13, !dbg !17
  %412 = extractelement <32 x float> %247, i64 14, !dbg !17
  %413 = extractelement <32 x float> %247, i64 15, !dbg !17
  %414 = select i1 %.not, float %411, float %410, !dbg !17
  %415 = bitcast float %414 to i32, !dbg !17
  %416 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %415), !dbg !17
  %417 = bitcast i32 %416 to float, !dbg !17
  %418 = select i1 %.not, float %410, float %417, !dbg !17
  %419 = select i1 %.not, float %417, float %411, !dbg !17
  %420 = select i1 %.not, float %413, float %412, !dbg !17
  %421 = bitcast float %420 to i32, !dbg !17
  %422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %421), !dbg !17
  %423 = bitcast i32 %422 to float, !dbg !17
  %424 = select i1 %.not, float %412, float %423, !dbg !17
  %425 = select i1 %.not, float %423, float %413, !dbg !17
  %426 = select i1 %.not155, float %424, float %418, !dbg !17
  %427 = bitcast float %426 to i32, !dbg !17
  %428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %427), !dbg !17
  %429 = bitcast i32 %428 to float, !dbg !17
  %430 = select i1 %.not155, float %418, float %429, !dbg !17
  %431 = select i1 %.not155, float %429, float %424, !dbg !17
  %432 = select i1 %.not155, float %425, float %419, !dbg !17
  %433 = bitcast float %432 to i32, !dbg !17
  %434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %433), !dbg !17
  %435 = bitcast i32 %434 to float, !dbg !17
  %436 = select i1 %.not155, float %419, float %435, !dbg !17
  %437 = select i1 %.not155, float %435, float %425, !dbg !17
  %compare.local.base773 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 12) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local774 = or i32 %compare.local.base773, 48, !dbg !26
  %compare.low775 = icmp ugt i32 %compare.low.base, %compare.local774, !dbg !26
  %compare.result776 = select i1 %compare.high.equal, i1 %compare.low775, i1 %compare.high.order, !dbg !26
  %438 = select i1 %compare.result776, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %439 = fadd float %430, %438, !dbg !20
  %compare.local.base777 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 13) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local778 = or i32 %compare.local.base777, 49, !dbg !26
  %compare.low779 = icmp ugt i32 %compare.low.base, %compare.local778, !dbg !26
  %compare.result780 = select i1 %compare.high.equal, i1 %compare.low779, i1 %compare.high.order, !dbg !26
  %440 = select i1 %compare.result780, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %441 = fadd float %436, %440, !dbg !20
  %compare.local.base781 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 14) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local782 = or i32 %compare.local.base781, 50, !dbg !26
  %compare.low783 = icmp ugt i32 %compare.low.base, %compare.local782, !dbg !26
  %compare.result784 = select i1 %compare.high.equal, i1 %compare.low783, i1 %compare.high.order, !dbg !26
  %442 = select i1 %compare.result784, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %443 = fadd float %431, %442, !dbg !20
  %compare.local.base785 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 15) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local786 = or i32 %compare.local.base785, 51, !dbg !26
  %compare.low787 = icmp ugt i32 %compare.low.base, %compare.local786, !dbg !26
  %compare.result788 = select i1 %compare.high.equal, i1 %compare.low787, i1 %compare.high.order, !dbg !26
  %444 = select i1 %compare.result788, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %445 = fadd float %437, %444, !dbg !20
  %446 = call float @__ocml_exp2_f32(float %439), !dbg !28
  %447 = insertelement <4 x float> poison, float %446, i64 0, !dbg !30
  %448 = fadd float %409, %446, !dbg !29
  %449 = call float @__ocml_exp2_f32(float %441), !dbg !28
  %450 = insertelement <4 x float> %447, float %449, i64 1, !dbg !30
  %451 = fadd float %448, %449, !dbg !29
  %452 = call float @__ocml_exp2_f32(float %443), !dbg !28
  %453 = insertelement <4 x float> %450, float %452, i64 2, !dbg !30
  %454 = fadd float %451, %452, !dbg !29
  %455 = call float @__ocml_exp2_f32(float %445), !dbg !28
  %456 = insertelement <4 x float> %453, float %455, i64 3, !dbg !30
  %math.result245 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %456) #8, !dbg !30, !frisk.register_lifetimes !1
  %457 = fptrunc <4 x float> %math.result245 to <4 x half>, !dbg !30
  %math.packed246 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %457) #8, !dbg !29, !frisk.register_lifetimes !1
  %458 = fadd float %454, %455, !dbg !29
  %459 = extractelement <32 x float> %247, i64 16, !dbg !17
  %460 = extractelement <32 x float> %247, i64 17, !dbg !17
  %461 = extractelement <32 x float> %247, i64 18, !dbg !17
  %462 = extractelement <32 x float> %247, i64 19, !dbg !17
  %463 = select i1 %.not, float %460, float %459, !dbg !17
  %464 = bitcast float %463 to i32, !dbg !17
  %465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %464), !dbg !17
  %466 = bitcast i32 %465 to float, !dbg !17
  %467 = select i1 %.not, float %459, float %466, !dbg !17
  %468 = select i1 %.not, float %466, float %460, !dbg !17
  %469 = select i1 %.not, float %462, float %461, !dbg !17
  %470 = bitcast float %469 to i32, !dbg !17
  %471 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %470), !dbg !17
  %472 = bitcast i32 %471 to float, !dbg !17
  %473 = select i1 %.not, float %461, float %472, !dbg !17
  %474 = select i1 %.not, float %472, float %462, !dbg !17
  %475 = select i1 %.not155, float %473, float %467, !dbg !17
  %476 = bitcast float %475 to i32, !dbg !17
  %477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %476), !dbg !17
  %478 = bitcast i32 %477 to float, !dbg !17
  %479 = select i1 %.not155, float %467, float %478, !dbg !17
  %480 = select i1 %.not155, float %478, float %473, !dbg !17
  %481 = select i1 %.not155, float %474, float %468, !dbg !17
  %482 = bitcast float %481 to i32, !dbg !17
  %483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %482), !dbg !17
  %484 = bitcast i32 %483 to float, !dbg !17
  %485 = select i1 %.not155, float %468, float %484, !dbg !17
  %486 = select i1 %.not155, float %484, float %474, !dbg !17
  %compare.local.base789 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 16) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local790 = or i32 %compare.local.base789, 64, !dbg !26
  %compare.low791 = icmp ugt i32 %compare.low.base, %compare.local790, !dbg !26
  %compare.result792 = select i1 %compare.high.equal, i1 %compare.low791, i1 %compare.high.order, !dbg !26
  %487 = select i1 %compare.result792, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %488 = fadd float %479, %487, !dbg !20
  %compare.local.base793 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 17) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local794 = or i32 %compare.local.base793, 65, !dbg !26
  %compare.low795 = icmp ugt i32 %compare.low.base, %compare.local794, !dbg !26
  %compare.result796 = select i1 %compare.high.equal, i1 %compare.low795, i1 %compare.high.order, !dbg !26
  %489 = select i1 %compare.result796, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %490 = fadd float %485, %489, !dbg !20
  %compare.local.base797 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 18) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local798 = or i32 %compare.local.base797, 66, !dbg !26
  %compare.low799 = icmp ugt i32 %compare.low.base, %compare.local798, !dbg !26
  %compare.result800 = select i1 %compare.high.equal, i1 %compare.low799, i1 %compare.high.order, !dbg !26
  %491 = select i1 %compare.result800, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %492 = fadd float %480, %491, !dbg !20
  %compare.local.base801 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 19) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local802 = or i32 %compare.local.base801, 67, !dbg !26
  %compare.low803 = icmp ugt i32 %compare.low.base, %compare.local802, !dbg !26
  %compare.result804 = select i1 %compare.high.equal, i1 %compare.low803, i1 %compare.high.order, !dbg !26
  %493 = select i1 %compare.result804, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %494 = fadd float %486, %493, !dbg !20
  %495 = call float @__ocml_exp2_f32(float %488), !dbg !28
  %496 = insertelement <4 x float> poison, float %495, i64 0, !dbg !30
  %497 = fadd float %458, %495, !dbg !29
  %498 = call float @__ocml_exp2_f32(float %490), !dbg !28
  %499 = insertelement <4 x float> %496, float %498, i64 1, !dbg !30
  %500 = fadd float %497, %498, !dbg !29
  %501 = call float @__ocml_exp2_f32(float %492), !dbg !28
  %502 = insertelement <4 x float> %499, float %501, i64 2, !dbg !30
  %503 = fadd float %500, %501, !dbg !29
  %504 = call float @__ocml_exp2_f32(float %494), !dbg !28
  %505 = insertelement <4 x float> %502, float %504, i64 3, !dbg !30
  %math.result247 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %505) #8, !dbg !30, !frisk.register_lifetimes !1
  %506 = fptrunc <4 x float> %math.result247 to <4 x half>, !dbg !30
  %math.packed248 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %506) #8, !dbg !29, !frisk.register_lifetimes !1
  %507 = fadd float %503, %504, !dbg !29
  %508 = extractelement <32 x float> %247, i64 20, !dbg !17
  %509 = extractelement <32 x float> %247, i64 21, !dbg !17
  %510 = extractelement <32 x float> %247, i64 22, !dbg !17
  %511 = extractelement <32 x float> %247, i64 23, !dbg !17
  %512 = select i1 %.not, float %509, float %508, !dbg !17
  %513 = bitcast float %512 to i32, !dbg !17
  %514 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %513), !dbg !17
  %515 = bitcast i32 %514 to float, !dbg !17
  %516 = select i1 %.not, float %508, float %515, !dbg !17
  %517 = select i1 %.not, float %515, float %509, !dbg !17
  %518 = select i1 %.not, float %511, float %510, !dbg !17
  %519 = bitcast float %518 to i32, !dbg !17
  %520 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %519), !dbg !17
  %521 = bitcast i32 %520 to float, !dbg !17
  %522 = select i1 %.not, float %510, float %521, !dbg !17
  %523 = select i1 %.not, float %521, float %511, !dbg !17
  %524 = select i1 %.not155, float %522, float %516, !dbg !17
  %525 = bitcast float %524 to i32, !dbg !17
  %526 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %525), !dbg !17
  %527 = bitcast i32 %526 to float, !dbg !17
  %528 = select i1 %.not155, float %516, float %527, !dbg !17
  %529 = select i1 %.not155, float %527, float %522, !dbg !17
  %530 = select i1 %.not155, float %523, float %517, !dbg !17
  %531 = bitcast float %530 to i32, !dbg !17
  %532 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %531), !dbg !17
  %533 = bitcast i32 %532 to float, !dbg !17
  %534 = select i1 %.not155, float %517, float %533, !dbg !17
  %535 = select i1 %.not155, float %533, float %523, !dbg !17
  %compare.local.base805 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 20) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local806 = or i32 %compare.local.base805, 80, !dbg !26
  %compare.low807 = icmp ugt i32 %compare.low.base, %compare.local806, !dbg !26
  %compare.result808 = select i1 %compare.high.equal, i1 %compare.low807, i1 %compare.high.order, !dbg !26
  %536 = select i1 %compare.result808, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %537 = fadd float %528, %536, !dbg !20
  %compare.local.base809 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 21) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local810 = or i32 %compare.local.base809, 81, !dbg !26
  %compare.low811 = icmp ugt i32 %compare.low.base, %compare.local810, !dbg !26
  %compare.result812 = select i1 %compare.high.equal, i1 %compare.low811, i1 %compare.high.order, !dbg !26
  %538 = select i1 %compare.result812, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %539 = fadd float %534, %538, !dbg !20
  %compare.local.base813 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 22) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local814 = or i32 %compare.local.base813, 82, !dbg !26
  %compare.low815 = icmp ugt i32 %compare.low.base, %compare.local814, !dbg !26
  %compare.result816 = select i1 %compare.high.equal, i1 %compare.low815, i1 %compare.high.order, !dbg !26
  %540 = select i1 %compare.result816, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %541 = fadd float %529, %540, !dbg !20
  %compare.local.base817 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 23) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local818 = or i32 %compare.local.base817, 83, !dbg !26
  %compare.low819 = icmp ugt i32 %compare.low.base, %compare.local818, !dbg !26
  %compare.result820 = select i1 %compare.high.equal, i1 %compare.low819, i1 %compare.high.order, !dbg !26
  %542 = select i1 %compare.result820, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %543 = fadd float %535, %542, !dbg !20
  %544 = call float @__ocml_exp2_f32(float %537), !dbg !28
  %545 = insertelement <4 x float> poison, float %544, i64 0, !dbg !30
  %546 = fadd float %507, %544, !dbg !29
  %547 = call float @__ocml_exp2_f32(float %539), !dbg !28
  %548 = insertelement <4 x float> %545, float %547, i64 1, !dbg !30
  %549 = fadd float %546, %547, !dbg !29
  %550 = call float @__ocml_exp2_f32(float %541), !dbg !28
  %551 = insertelement <4 x float> %548, float %550, i64 2, !dbg !30
  %552 = fadd float %549, %550, !dbg !29
  %553 = call float @__ocml_exp2_f32(float %543), !dbg !28
  %554 = insertelement <4 x float> %551, float %553, i64 3, !dbg !30
  %math.result249 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %554) #8, !dbg !30, !frisk.register_lifetimes !1
  %555 = fptrunc <4 x float> %math.result249 to <4 x half>, !dbg !30
  %math.packed250 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %555) #8, !dbg !29, !frisk.register_lifetimes !1
  %556 = fadd float %552, %553, !dbg !29
  %557 = extractelement <32 x float> %247, i64 24, !dbg !17
  %558 = extractelement <32 x float> %247, i64 25, !dbg !17
  %559 = extractelement <32 x float> %247, i64 26, !dbg !17
  %560 = extractelement <32 x float> %247, i64 27, !dbg !17
  %561 = select i1 %.not, float %558, float %557, !dbg !17
  %562 = bitcast float %561 to i32, !dbg !17
  %563 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %562), !dbg !17
  %564 = bitcast i32 %563 to float, !dbg !17
  %565 = select i1 %.not, float %557, float %564, !dbg !17
  %566 = select i1 %.not, float %564, float %558, !dbg !17
  %567 = select i1 %.not, float %560, float %559, !dbg !17
  %568 = bitcast float %567 to i32, !dbg !17
  %569 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %568), !dbg !17
  %570 = bitcast i32 %569 to float, !dbg !17
  %571 = select i1 %.not, float %559, float %570, !dbg !17
  %572 = select i1 %.not, float %570, float %560, !dbg !17
  %573 = select i1 %.not155, float %571, float %565, !dbg !17
  %574 = bitcast float %573 to i32, !dbg !17
  %575 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %574), !dbg !17
  %576 = bitcast i32 %575 to float, !dbg !17
  %577 = select i1 %.not155, float %565, float %576, !dbg !17
  %578 = select i1 %.not155, float %576, float %571, !dbg !17
  %579 = select i1 %.not155, float %572, float %566, !dbg !17
  %580 = bitcast float %579 to i32, !dbg !17
  %581 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %580), !dbg !17
  %582 = bitcast i32 %581 to float, !dbg !17
  %583 = select i1 %.not155, float %566, float %582, !dbg !17
  %584 = select i1 %.not155, float %582, float %572, !dbg !17
  %compare.local.base821 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 24) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local822 = or i32 %compare.local.base821, 96, !dbg !26
  %compare.low823 = icmp ugt i32 %compare.low.base, %compare.local822, !dbg !26
  %compare.result824 = select i1 %compare.high.equal, i1 %compare.low823, i1 %compare.high.order, !dbg !26
  %585 = select i1 %compare.result824, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %586 = fadd float %577, %585, !dbg !20
  %compare.local.base825 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 25) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local826 = or i32 %compare.local.base825, 97, !dbg !26
  %compare.low827 = icmp ugt i32 %compare.low.base, %compare.local826, !dbg !26
  %compare.result828 = select i1 %compare.high.equal, i1 %compare.low827, i1 %compare.high.order, !dbg !26
  %587 = select i1 %compare.result828, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %588 = fadd float %583, %587, !dbg !20
  %compare.local.base829 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 26) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local830 = or i32 %compare.local.base829, 98, !dbg !26
  %compare.low831 = icmp ugt i32 %compare.low.base, %compare.local830, !dbg !26
  %compare.result832 = select i1 %compare.high.equal, i1 %compare.low831, i1 %compare.high.order, !dbg !26
  %589 = select i1 %compare.result832, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %590 = fadd float %578, %589, !dbg !20
  %compare.local.base833 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 27) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local834 = or i32 %compare.local.base833, 99, !dbg !26
  %compare.low835 = icmp ugt i32 %compare.low.base, %compare.local834, !dbg !26
  %compare.result836 = select i1 %compare.high.equal, i1 %compare.low835, i1 %compare.high.order, !dbg !26
  %591 = select i1 %compare.result836, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %592 = fadd float %584, %591, !dbg !20
  %593 = call float @__ocml_exp2_f32(float %586), !dbg !28
  %594 = insertelement <4 x float> poison, float %593, i64 0, !dbg !30
  %595 = fadd float %556, %593, !dbg !29
  %596 = call float @__ocml_exp2_f32(float %588), !dbg !28
  %597 = insertelement <4 x float> %594, float %596, i64 1, !dbg !30
  %598 = fadd float %595, %596, !dbg !29
  %599 = call float @__ocml_exp2_f32(float %590), !dbg !28
  %600 = insertelement <4 x float> %597, float %599, i64 2, !dbg !30
  %601 = fadd float %598, %599, !dbg !29
  %602 = call float @__ocml_exp2_f32(float %592), !dbg !28
  %603 = insertelement <4 x float> %600, float %602, i64 3, !dbg !30
  %math.result251 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %603) #8, !dbg !30, !frisk.register_lifetimes !1
  %604 = fptrunc <4 x float> %math.result251 to <4 x half>, !dbg !30
  %math.packed252 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %604) #8, !dbg !29, !frisk.register_lifetimes !1
  %605 = fadd float %601, %602, !dbg !29
  %606 = extractelement <32 x float> %247, i64 28, !dbg !17
  %607 = extractelement <32 x float> %247, i64 29, !dbg !17
  %608 = extractelement <32 x float> %247, i64 30, !dbg !17
  %609 = extractelement <32 x float> %247, i64 31, !dbg !17
  %610 = select i1 %.not, float %607, float %606, !dbg !17
  %611 = bitcast float %610 to i32, !dbg !17
  %612 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %611), !dbg !17
  %613 = bitcast i32 %612 to float, !dbg !17
  %614 = select i1 %.not, float %606, float %613, !dbg !17
  %615 = select i1 %.not, float %613, float %607, !dbg !17
  %616 = select i1 %.not, float %609, float %608, !dbg !17
  %617 = bitcast float %616 to i32, !dbg !17
  %618 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %617), !dbg !17
  %619 = bitcast i32 %618 to float, !dbg !17
  %620 = select i1 %.not, float %608, float %619, !dbg !17
  %621 = select i1 %.not, float %619, float %609, !dbg !17
  %622 = select i1 %.not155, float %620, float %614, !dbg !17
  %623 = bitcast float %622 to i32, !dbg !17
  %624 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %623), !dbg !17
  %625 = bitcast i32 %624 to float, !dbg !17
  %626 = select i1 %.not155, float %614, float %625, !dbg !17
  %627 = select i1 %.not155, float %625, float %620, !dbg !17
  %628 = select i1 %.not155, float %621, float %615, !dbg !17
  %629 = bitcast float %628 to i32, !dbg !17
  %630 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %629), !dbg !17
  %631 = bitcast i32 %630 to float, !dbg !17
  %632 = select i1 %.not155, float %615, float %631, !dbg !17
  %633 = select i1 %.not155, float %631, float %621, !dbg !17
  %compare.local.base837 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 28) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local838 = or i32 %compare.local.base837, 112, !dbg !26
  %compare.low839 = icmp ugt i32 %compare.low.base, %compare.local838, !dbg !26
  %compare.result840 = select i1 %compare.high.equal, i1 %compare.low839, i1 %compare.high.order, !dbg !26
  %634 = select i1 %compare.result840, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %635 = fadd float %626, %634, !dbg !20
  %compare.local.base841 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 29) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local842 = or i32 %compare.local.base841, 113, !dbg !26
  %compare.low843 = icmp ugt i32 %compare.low.base, %compare.local842, !dbg !26
  %compare.result844 = select i1 %compare.high.equal, i1 %compare.low843, i1 %compare.high.order, !dbg !26
  %636 = select i1 %compare.result844, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %637 = fadd float %632, %636, !dbg !20
  %compare.local.base845 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 30) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local846 = or i32 %compare.local.base845, 114, !dbg !26
  %compare.low847 = icmp ugt i32 %compare.low.base, %compare.local846, !dbg !26
  %compare.result848 = select i1 %compare.high.equal, i1 %compare.low847, i1 %compare.high.order, !dbg !26
  %638 = select i1 %compare.result848, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %639 = fadd float %627, %638, !dbg !20
  %compare.local.base849 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 31) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local850 = or i32 %compare.local.base849, 115, !dbg !26
  %compare.low851 = icmp ugt i32 %compare.low.base, %compare.local850, !dbg !26
  %compare.result852 = select i1 %compare.high.equal, i1 %compare.low851, i1 %compare.high.order, !dbg !26
  %640 = select i1 %compare.result852, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %641 = fadd float %633, %640, !dbg !20
  %642 = call float @__ocml_exp2_f32(float %635), !dbg !28
  %643 = insertelement <4 x float> poison, float %642, i64 0, !dbg !30
  %644 = fadd float %605, %642, !dbg !29
  %645 = call float @__ocml_exp2_f32(float %637), !dbg !28
  %646 = insertelement <4 x float> %643, float %645, i64 1, !dbg !30
  %647 = fadd float %644, %645, !dbg !29
  %648 = call float @__ocml_exp2_f32(float %639), !dbg !28
  %649 = insertelement <4 x float> %646, float %648, i64 2, !dbg !30
  %650 = fadd float %647, %648, !dbg !29
  %651 = call float @__ocml_exp2_f32(float %641), !dbg !28
  %652 = insertelement <4 x float> %649, float %651, i64 3, !dbg !30
  %math.result253 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %652) #8, !dbg !30, !frisk.register_lifetimes !1
  %653 = fptrunc <4 x float> %math.result253 to <4 x half>, !dbg !30
  %math.packed254 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %653) #8, !dbg !29, !frisk.register_lifetimes !1
  %654 = fadd float %650, %651, !dbg !29
  %655 = extractvalue [2 x <32 x float>] %51, 1, !dbg !17
  %656 = extractelement <32 x float> %655, i64 0, !dbg !17
  %657 = extractelement <32 x float> %655, i64 1, !dbg !17
  %658 = extractelement <32 x float> %655, i64 2, !dbg !17
  %659 = extractelement <32 x float> %655, i64 3, !dbg !17
  %660 = select i1 %.not, float %657, float %656, !dbg !17
  %661 = bitcast float %660 to i32, !dbg !17
  %662 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %661), !dbg !17
  %663 = bitcast i32 %662 to float, !dbg !17
  %664 = select i1 %.not, float %656, float %663, !dbg !17
  %665 = select i1 %.not, float %663, float %657, !dbg !17
  %666 = select i1 %.not, float %659, float %658, !dbg !17
  %667 = bitcast float %666 to i32, !dbg !17
  %668 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %667), !dbg !17
  %669 = bitcast i32 %668 to float, !dbg !17
  %670 = select i1 %.not, float %658, float %669, !dbg !17
  %671 = select i1 %.not, float %669, float %659, !dbg !17
  %672 = select i1 %.not155, float %670, float %664, !dbg !17
  %673 = bitcast float %672 to i32, !dbg !17
  %674 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %673), !dbg !17
  %675 = bitcast i32 %674 to float, !dbg !17
  %676 = select i1 %.not155, float %664, float %675, !dbg !17
  %677 = select i1 %.not155, float %675, float %670, !dbg !17
  %678 = select i1 %.not155, float %671, float %665, !dbg !17
  %679 = bitcast float %678 to i32, !dbg !17
  %680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %679), !dbg !17
  %681 = bitcast i32 %680 to float, !dbg !17
  %682 = select i1 %.not155, float %665, float %681, !dbg !17
  %683 = select i1 %.not155, float %681, float %671, !dbg !17
  %684 = lshr i64 %33, 32, !dbg !26
  %compare.high853 = trunc i64 %684 to i32, !dbg !26
  %685 = lshr i64 %288, 32, !dbg !26
  %compare.high854 = trunc i64 %685 to i32, !dbg !26
  %compare.high.equal855 = icmp eq i32 %compare.high853, %compare.high854, !dbg !26
  %compare.high.order856 = icmp ugt i32 %compare.high853, %compare.high854, !dbg !26
  %compare.low.base857 = trunc i64 %33 to i32, !dbg !26
  %compare.low.base858 = trunc i64 %288 to i32, !dbg !26
  %compare.low859 = icmp ugt i32 %compare.low.base857, %compare.low.base858, !dbg !26
  %compare.result860 = select i1 %compare.high.equal855, i1 %compare.low859, i1 %compare.high.order856, !dbg !26
  %686 = select i1 %compare.result860, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %687 = fadd float %676, %686, !dbg !20
  %compare.local.base861 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 32) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local862 = or i32 %compare.local.base861, 1, !dbg !26
  %compare.low863 = icmp ugt i32 %compare.low.base857, %compare.local862, !dbg !26
  %compare.result864 = select i1 %compare.high.equal855, i1 %compare.low863, i1 %compare.high.order856, !dbg !26
  %688 = select i1 %compare.result864, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %689 = fadd float %682, %688, !dbg !20
  %compare.local.base865 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 33) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local866 = or i32 %compare.local.base865, 2, !dbg !26
  %compare.low867 = icmp ugt i32 %compare.low.base857, %compare.local866, !dbg !26
  %compare.result868 = select i1 %compare.high.equal855, i1 %compare.low867, i1 %compare.high.order856, !dbg !26
  %690 = select i1 %compare.result868, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %691 = fadd float %677, %690, !dbg !20
  %compare.local.base869 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 34) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local870 = or i32 %compare.local.base869, 3, !dbg !26
  %compare.low871 = icmp ugt i32 %compare.low.base857, %compare.local870, !dbg !26
  %compare.result872 = select i1 %compare.high.equal855, i1 %compare.low871, i1 %compare.high.order856, !dbg !26
  %692 = select i1 %compare.result872, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %693 = fadd float %683, %692, !dbg !20
  %694 = call float @__ocml_exp2_f32(float %687), !dbg !28
  %695 = insertelement <4 x float> poison, float %694, i64 0, !dbg !30
  %696 = fadd float %694, 0x0000000000000000, !dbg !29
  %697 = call float @__ocml_exp2_f32(float %689), !dbg !28
  %698 = insertelement <4 x float> %695, float %697, i64 1, !dbg !30
  %699 = fadd float %696, %697, !dbg !29
  %700 = call float @__ocml_exp2_f32(float %691), !dbg !28
  %701 = insertelement <4 x float> %698, float %700, i64 2, !dbg !30
  %702 = fadd float %699, %700, !dbg !29
  %703 = call float @__ocml_exp2_f32(float %693), !dbg !28
  %704 = insertelement <4 x float> %701, float %703, i64 3, !dbg !30
  %math.result255 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %704) #8, !dbg !30, !frisk.register_lifetimes !1
  %705 = fptrunc <4 x float> %math.result255 to <4 x half>, !dbg !30
  %math.packed256 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %705) #8, !dbg !29, !frisk.register_lifetimes !1
  %706 = fadd float %702, %703, !dbg !29
  %707 = extractelement <32 x float> %655, i64 4, !dbg !17
  %708 = extractelement <32 x float> %655, i64 5, !dbg !17
  %709 = extractelement <32 x float> %655, i64 6, !dbg !17
  %710 = extractelement <32 x float> %655, i64 7, !dbg !17
  %711 = select i1 %.not, float %708, float %707, !dbg !17
  %712 = bitcast float %711 to i32, !dbg !17
  %713 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %712), !dbg !17
  %714 = bitcast i32 %713 to float, !dbg !17
  %715 = select i1 %.not, float %707, float %714, !dbg !17
  %716 = select i1 %.not, float %714, float %708, !dbg !17
  %717 = select i1 %.not, float %710, float %709, !dbg !17
  %718 = bitcast float %717 to i32, !dbg !17
  %719 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %718), !dbg !17
  %720 = bitcast i32 %719 to float, !dbg !17
  %721 = select i1 %.not, float %709, float %720, !dbg !17
  %722 = select i1 %.not, float %720, float %710, !dbg !17
  %723 = select i1 %.not155, float %721, float %715, !dbg !17
  %724 = bitcast float %723 to i32, !dbg !17
  %725 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %724), !dbg !17
  %726 = bitcast i32 %725 to float, !dbg !17
  %727 = select i1 %.not155, float %715, float %726, !dbg !17
  %728 = select i1 %.not155, float %726, float %721, !dbg !17
  %729 = select i1 %.not155, float %722, float %716, !dbg !17
  %730 = bitcast float %729 to i32, !dbg !17
  %731 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %730), !dbg !17
  %732 = bitcast i32 %731 to float, !dbg !17
  %733 = select i1 %.not155, float %716, float %732, !dbg !17
  %734 = select i1 %.not155, float %732, float %722, !dbg !17
  %compare.local.base873 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 35) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local874 = or i32 %compare.local.base873, 16, !dbg !26
  %compare.low875 = icmp ugt i32 %compare.low.base857, %compare.local874, !dbg !26
  %compare.result876 = select i1 %compare.high.equal855, i1 %compare.low875, i1 %compare.high.order856, !dbg !26
  %735 = select i1 %compare.result876, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %736 = fadd float %727, %735, !dbg !20
  %compare.local.base877 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 36) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local878 = or i32 %compare.local.base877, 17, !dbg !26
  %compare.low879 = icmp ugt i32 %compare.low.base857, %compare.local878, !dbg !26
  %compare.result880 = select i1 %compare.high.equal855, i1 %compare.low879, i1 %compare.high.order856, !dbg !26
  %737 = select i1 %compare.result880, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %738 = fadd float %733, %737, !dbg !20
  %compare.local.base881 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 37) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local882 = or i32 %compare.local.base881, 18, !dbg !26
  %compare.low883 = icmp ugt i32 %compare.low.base857, %compare.local882, !dbg !26
  %compare.result884 = select i1 %compare.high.equal855, i1 %compare.low883, i1 %compare.high.order856, !dbg !26
  %739 = select i1 %compare.result884, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %740 = fadd float %728, %739, !dbg !20
  %compare.local.base885 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 38) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local886 = or i32 %compare.local.base885, 19, !dbg !26
  %compare.low887 = icmp ugt i32 %compare.low.base857, %compare.local886, !dbg !26
  %compare.result888 = select i1 %compare.high.equal855, i1 %compare.low887, i1 %compare.high.order856, !dbg !26
  %741 = select i1 %compare.result888, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %742 = fadd float %734, %741, !dbg !20
  %743 = call float @__ocml_exp2_f32(float %736), !dbg !28
  %744 = insertelement <4 x float> poison, float %743, i64 0, !dbg !30
  %745 = fadd float %706, %743, !dbg !29
  %746 = call float @__ocml_exp2_f32(float %738), !dbg !28
  %747 = insertelement <4 x float> %744, float %746, i64 1, !dbg !30
  %748 = fadd float %745, %746, !dbg !29
  %749 = call float @__ocml_exp2_f32(float %740), !dbg !28
  %750 = insertelement <4 x float> %747, float %749, i64 2, !dbg !30
  %751 = fadd float %748, %749, !dbg !29
  %752 = call float @__ocml_exp2_f32(float %742), !dbg !28
  %753 = insertelement <4 x float> %750, float %752, i64 3, !dbg !30
  %math.result257 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %753) #8, !dbg !30, !frisk.register_lifetimes !1
  %754 = fptrunc <4 x float> %math.result257 to <4 x half>, !dbg !30
  %math.packed258 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %754) #8, !dbg !29, !frisk.register_lifetimes !1
  %755 = fadd float %751, %752, !dbg !29
  %756 = extractelement <32 x float> %655, i64 8, !dbg !17
  %757 = extractelement <32 x float> %655, i64 9, !dbg !17
  %758 = extractelement <32 x float> %655, i64 10, !dbg !17
  %759 = extractelement <32 x float> %655, i64 11, !dbg !17
  %760 = select i1 %.not, float %757, float %756, !dbg !17
  %761 = bitcast float %760 to i32, !dbg !17
  %762 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %761), !dbg !17
  %763 = bitcast i32 %762 to float, !dbg !17
  %764 = select i1 %.not, float %756, float %763, !dbg !17
  %765 = select i1 %.not, float %763, float %757, !dbg !17
  %766 = select i1 %.not, float %759, float %758, !dbg !17
  %767 = bitcast float %766 to i32, !dbg !17
  %768 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %767), !dbg !17
  %769 = bitcast i32 %768 to float, !dbg !17
  %770 = select i1 %.not, float %758, float %769, !dbg !17
  %771 = select i1 %.not, float %769, float %759, !dbg !17
  %772 = select i1 %.not155, float %770, float %764, !dbg !17
  %773 = bitcast float %772 to i32, !dbg !17
  %774 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %773), !dbg !17
  %775 = bitcast i32 %774 to float, !dbg !17
  %776 = select i1 %.not155, float %764, float %775, !dbg !17
  %777 = select i1 %.not155, float %775, float %770, !dbg !17
  %778 = select i1 %.not155, float %771, float %765, !dbg !17
  %779 = bitcast float %778 to i32, !dbg !17
  %780 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %779), !dbg !17
  %781 = bitcast i32 %780 to float, !dbg !17
  %782 = select i1 %.not155, float %765, float %781, !dbg !17
  %783 = select i1 %.not155, float %781, float %771, !dbg !17
  %compare.local.base889 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 39) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local890 = or i32 %compare.local.base889, 32, !dbg !26
  %compare.low891 = icmp ugt i32 %compare.low.base857, %compare.local890, !dbg !26
  %compare.result892 = select i1 %compare.high.equal855, i1 %compare.low891, i1 %compare.high.order856, !dbg !26
  %784 = select i1 %compare.result892, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %785 = fadd float %776, %784, !dbg !20
  %compare.local.base893 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 40) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local894 = or i32 %compare.local.base893, 33, !dbg !26
  %compare.low895 = icmp ugt i32 %compare.low.base857, %compare.local894, !dbg !26
  %compare.result896 = select i1 %compare.high.equal855, i1 %compare.low895, i1 %compare.high.order856, !dbg !26
  %786 = select i1 %compare.result896, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %787 = fadd float %782, %786, !dbg !20
  %compare.local.base897 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 41) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local898 = or i32 %compare.local.base897, 34, !dbg !26
  %compare.low899 = icmp ugt i32 %compare.low.base857, %compare.local898, !dbg !26
  %compare.result900 = select i1 %compare.high.equal855, i1 %compare.low899, i1 %compare.high.order856, !dbg !26
  %788 = select i1 %compare.result900, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %789 = fadd float %777, %788, !dbg !20
  %compare.local.base901 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 42) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local902 = or i32 %compare.local.base901, 35, !dbg !26
  %compare.low903 = icmp ugt i32 %compare.low.base857, %compare.local902, !dbg !26
  %compare.result904 = select i1 %compare.high.equal855, i1 %compare.low903, i1 %compare.high.order856, !dbg !26
  %790 = select i1 %compare.result904, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %791 = fadd float %783, %790, !dbg !20
  %792 = call float @__ocml_exp2_f32(float %785), !dbg !28
  %793 = insertelement <4 x float> poison, float %792, i64 0, !dbg !30
  %794 = fadd float %755, %792, !dbg !29
  %795 = call float @__ocml_exp2_f32(float %787), !dbg !28
  %796 = insertelement <4 x float> %793, float %795, i64 1, !dbg !30
  %797 = fadd float %794, %795, !dbg !29
  %798 = call float @__ocml_exp2_f32(float %789), !dbg !28
  %799 = insertelement <4 x float> %796, float %798, i64 2, !dbg !30
  %800 = fadd float %797, %798, !dbg !29
  %801 = call float @__ocml_exp2_f32(float %791), !dbg !28
  %802 = insertelement <4 x float> %799, float %801, i64 3, !dbg !30
  %math.result259 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %802) #8, !dbg !30, !frisk.register_lifetimes !1
  %803 = fptrunc <4 x float> %math.result259 to <4 x half>, !dbg !30
  %math.packed260 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %803) #8, !dbg !29, !frisk.register_lifetimes !1
  %804 = fadd float %800, %801, !dbg !29
  %805 = extractelement <32 x float> %655, i64 12, !dbg !17
  %806 = extractelement <32 x float> %655, i64 13, !dbg !17
  %807 = extractelement <32 x float> %655, i64 14, !dbg !17
  %808 = extractelement <32 x float> %655, i64 15, !dbg !17
  %809 = select i1 %.not, float %806, float %805, !dbg !17
  %810 = bitcast float %809 to i32, !dbg !17
  %811 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %810), !dbg !17
  %812 = bitcast i32 %811 to float, !dbg !17
  %813 = select i1 %.not, float %805, float %812, !dbg !17
  %814 = select i1 %.not, float %812, float %806, !dbg !17
  %815 = select i1 %.not, float %808, float %807, !dbg !17
  %816 = bitcast float %815 to i32, !dbg !17
  %817 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %816), !dbg !17
  %818 = bitcast i32 %817 to float, !dbg !17
  %819 = select i1 %.not, float %807, float %818, !dbg !17
  %820 = select i1 %.not, float %818, float %808, !dbg !17
  %821 = select i1 %.not155, float %819, float %813, !dbg !17
  %822 = bitcast float %821 to i32, !dbg !17
  %823 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %822), !dbg !17
  %824 = bitcast i32 %823 to float, !dbg !17
  %825 = select i1 %.not155, float %813, float %824, !dbg !17
  %826 = select i1 %.not155, float %824, float %819, !dbg !17
  %827 = select i1 %.not155, float %820, float %814, !dbg !17
  %828 = bitcast float %827 to i32, !dbg !17
  %829 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %828), !dbg !17
  %830 = bitcast i32 %829 to float, !dbg !17
  %831 = select i1 %.not155, float %814, float %830, !dbg !17
  %832 = select i1 %.not155, float %830, float %820, !dbg !17
  %compare.local.base905 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 43) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local906 = or i32 %compare.local.base905, 48, !dbg !26
  %compare.low907 = icmp ugt i32 %compare.low.base857, %compare.local906, !dbg !26
  %compare.result908 = select i1 %compare.high.equal855, i1 %compare.low907, i1 %compare.high.order856, !dbg !26
  %833 = select i1 %compare.result908, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %834 = fadd float %825, %833, !dbg !20
  %compare.local.base909 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 44) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local910 = or i32 %compare.local.base909, 49, !dbg !26
  %compare.low911 = icmp ugt i32 %compare.low.base857, %compare.local910, !dbg !26
  %compare.result912 = select i1 %compare.high.equal855, i1 %compare.low911, i1 %compare.high.order856, !dbg !26
  %835 = select i1 %compare.result912, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %836 = fadd float %831, %835, !dbg !20
  %compare.local.base913 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 45) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local914 = or i32 %compare.local.base913, 50, !dbg !26
  %compare.low915 = icmp ugt i32 %compare.low.base857, %compare.local914, !dbg !26
  %compare.result916 = select i1 %compare.high.equal855, i1 %compare.low915, i1 %compare.high.order856, !dbg !26
  %837 = select i1 %compare.result916, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %838 = fadd float %826, %837, !dbg !20
  %compare.local.base917 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 46) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local918 = or i32 %compare.local.base917, 51, !dbg !26
  %compare.low919 = icmp ugt i32 %compare.low.base857, %compare.local918, !dbg !26
  %compare.result920 = select i1 %compare.high.equal855, i1 %compare.low919, i1 %compare.high.order856, !dbg !26
  %839 = select i1 %compare.result920, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %840 = fadd float %832, %839, !dbg !20
  %841 = call float @__ocml_exp2_f32(float %834), !dbg !28
  %842 = insertelement <4 x float> poison, float %841, i64 0, !dbg !30
  %843 = fadd float %804, %841, !dbg !29
  %844 = call float @__ocml_exp2_f32(float %836), !dbg !28
  %845 = insertelement <4 x float> %842, float %844, i64 1, !dbg !30
  %846 = fadd float %843, %844, !dbg !29
  %847 = call float @__ocml_exp2_f32(float %838), !dbg !28
  %848 = insertelement <4 x float> %845, float %847, i64 2, !dbg !30
  %849 = fadd float %846, %847, !dbg !29
  %850 = call float @__ocml_exp2_f32(float %840), !dbg !28
  %851 = insertelement <4 x float> %848, float %850, i64 3, !dbg !30
  %math.result261 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %851) #8, !dbg !30, !frisk.register_lifetimes !1
  %852 = fptrunc <4 x float> %math.result261 to <4 x half>, !dbg !30
  %math.packed262 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %852) #8, !dbg !29, !frisk.register_lifetimes !1
  %853 = fadd float %849, %850, !dbg !29
  %854 = extractelement <32 x float> %655, i64 16, !dbg !17
  %855 = extractelement <32 x float> %655, i64 17, !dbg !17
  %856 = extractelement <32 x float> %655, i64 18, !dbg !17
  %857 = extractelement <32 x float> %655, i64 19, !dbg !17
  %858 = select i1 %.not, float %855, float %854, !dbg !17
  %859 = bitcast float %858 to i32, !dbg !17
  %860 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %859), !dbg !17
  %861 = bitcast i32 %860 to float, !dbg !17
  %862 = select i1 %.not, float %854, float %861, !dbg !17
  %863 = select i1 %.not, float %861, float %855, !dbg !17
  %864 = select i1 %.not, float %857, float %856, !dbg !17
  %865 = bitcast float %864 to i32, !dbg !17
  %866 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %865), !dbg !17
  %867 = bitcast i32 %866 to float, !dbg !17
  %868 = select i1 %.not, float %856, float %867, !dbg !17
  %869 = select i1 %.not, float %867, float %857, !dbg !17
  %870 = select i1 %.not155, float %868, float %862, !dbg !17
  %871 = bitcast float %870 to i32, !dbg !17
  %872 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %871), !dbg !17
  %873 = bitcast i32 %872 to float, !dbg !17
  %874 = select i1 %.not155, float %862, float %873, !dbg !17
  %875 = select i1 %.not155, float %873, float %868, !dbg !17
  %876 = select i1 %.not155, float %869, float %863, !dbg !17
  %877 = bitcast float %876 to i32, !dbg !17
  %878 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %877), !dbg !17
  %879 = bitcast i32 %878 to float, !dbg !17
  %880 = select i1 %.not155, float %863, float %879, !dbg !17
  %881 = select i1 %.not155, float %879, float %869, !dbg !17
  %compare.local.base921 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 47) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local922 = or i32 %compare.local.base921, 64, !dbg !26
  %compare.low923 = icmp ugt i32 %compare.low.base857, %compare.local922, !dbg !26
  %compare.result924 = select i1 %compare.high.equal855, i1 %compare.low923, i1 %compare.high.order856, !dbg !26
  %882 = select i1 %compare.result924, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %883 = fadd float %874, %882, !dbg !20
  %compare.local.base925 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 48) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local926 = or i32 %compare.local.base925, 65, !dbg !26
  %compare.low927 = icmp ugt i32 %compare.low.base857, %compare.local926, !dbg !26
  %compare.result928 = select i1 %compare.high.equal855, i1 %compare.low927, i1 %compare.high.order856, !dbg !26
  %884 = select i1 %compare.result928, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %885 = fadd float %880, %884, !dbg !20
  %compare.local.base929 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 49) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local930 = or i32 %compare.local.base929, 66, !dbg !26
  %compare.low931 = icmp ugt i32 %compare.low.base857, %compare.local930, !dbg !26
  %compare.result932 = select i1 %compare.high.equal855, i1 %compare.low931, i1 %compare.high.order856, !dbg !26
  %886 = select i1 %compare.result932, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %887 = fadd float %875, %886, !dbg !20
  %compare.local.base933 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 50) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local934 = or i32 %compare.local.base933, 67, !dbg !26
  %compare.low935 = icmp ugt i32 %compare.low.base857, %compare.local934, !dbg !26
  %compare.result936 = select i1 %compare.high.equal855, i1 %compare.low935, i1 %compare.high.order856, !dbg !26
  %888 = select i1 %compare.result936, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %889 = fadd float %881, %888, !dbg !20
  %890 = call float @__ocml_exp2_f32(float %883), !dbg !28
  %891 = insertelement <4 x float> poison, float %890, i64 0, !dbg !30
  %892 = fadd float %853, %890, !dbg !29
  %893 = call float @__ocml_exp2_f32(float %885), !dbg !28
  %894 = insertelement <4 x float> %891, float %893, i64 1, !dbg !30
  %895 = fadd float %892, %893, !dbg !29
  %896 = call float @__ocml_exp2_f32(float %887), !dbg !28
  %897 = insertelement <4 x float> %894, float %896, i64 2, !dbg !30
  %898 = fadd float %895, %896, !dbg !29
  %899 = call float @__ocml_exp2_f32(float %889), !dbg !28
  %900 = insertelement <4 x float> %897, float %899, i64 3, !dbg !30
  %math.result263 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %900) #8, !dbg !30, !frisk.register_lifetimes !1
  %901 = fptrunc <4 x float> %math.result263 to <4 x half>, !dbg !30
  %math.packed264 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %901) #8, !dbg !29, !frisk.register_lifetimes !1
  %902 = fadd float %898, %899, !dbg !29
  %903 = extractelement <32 x float> %655, i64 20, !dbg !17
  %904 = extractelement <32 x float> %655, i64 21, !dbg !17
  %905 = extractelement <32 x float> %655, i64 22, !dbg !17
  %906 = extractelement <32 x float> %655, i64 23, !dbg !17
  %907 = select i1 %.not, float %904, float %903, !dbg !17
  %908 = bitcast float %907 to i32, !dbg !17
  %909 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %908), !dbg !17
  %910 = bitcast i32 %909 to float, !dbg !17
  %911 = select i1 %.not, float %903, float %910, !dbg !17
  %912 = select i1 %.not, float %910, float %904, !dbg !17
  %913 = select i1 %.not, float %906, float %905, !dbg !17
  %914 = bitcast float %913 to i32, !dbg !17
  %915 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %914), !dbg !17
  %916 = bitcast i32 %915 to float, !dbg !17
  %917 = select i1 %.not, float %905, float %916, !dbg !17
  %918 = select i1 %.not, float %916, float %906, !dbg !17
  %919 = select i1 %.not155, float %917, float %911, !dbg !17
  %920 = bitcast float %919 to i32, !dbg !17
  %921 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %920), !dbg !17
  %922 = bitcast i32 %921 to float, !dbg !17
  %923 = select i1 %.not155, float %911, float %922, !dbg !17
  %924 = select i1 %.not155, float %922, float %917, !dbg !17
  %925 = select i1 %.not155, float %918, float %912, !dbg !17
  %926 = bitcast float %925 to i32, !dbg !17
  %927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %926), !dbg !17
  %928 = bitcast i32 %927 to float, !dbg !17
  %929 = select i1 %.not155, float %912, float %928, !dbg !17
  %930 = select i1 %.not155, float %928, float %918, !dbg !17
  %compare.local.base937 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 51) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local938 = or i32 %compare.local.base937, 80, !dbg !26
  %compare.low939 = icmp ugt i32 %compare.low.base857, %compare.local938, !dbg !26
  %compare.result940 = select i1 %compare.high.equal855, i1 %compare.low939, i1 %compare.high.order856, !dbg !26
  %931 = select i1 %compare.result940, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %932 = fadd float %923, %931, !dbg !20
  %compare.local.base941 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 52) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local942 = or i32 %compare.local.base941, 81, !dbg !26
  %compare.low943 = icmp ugt i32 %compare.low.base857, %compare.local942, !dbg !26
  %compare.result944 = select i1 %compare.high.equal855, i1 %compare.low943, i1 %compare.high.order856, !dbg !26
  %933 = select i1 %compare.result944, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %934 = fadd float %929, %933, !dbg !20
  %compare.local.base945 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 53) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local946 = or i32 %compare.local.base945, 82, !dbg !26
  %compare.low947 = icmp ugt i32 %compare.low.base857, %compare.local946, !dbg !26
  %compare.result948 = select i1 %compare.high.equal855, i1 %compare.low947, i1 %compare.high.order856, !dbg !26
  %935 = select i1 %compare.result948, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %936 = fadd float %924, %935, !dbg !20
  %compare.local.base949 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 54) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local950 = or i32 %compare.local.base949, 83, !dbg !26
  %compare.low951 = icmp ugt i32 %compare.low.base857, %compare.local950, !dbg !26
  %compare.result952 = select i1 %compare.high.equal855, i1 %compare.low951, i1 %compare.high.order856, !dbg !26
  %937 = select i1 %compare.result952, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %938 = fadd float %930, %937, !dbg !20
  %939 = call float @__ocml_exp2_f32(float %932), !dbg !28
  %940 = insertelement <4 x float> poison, float %939, i64 0, !dbg !30
  %941 = fadd float %902, %939, !dbg !29
  %942 = call float @__ocml_exp2_f32(float %934), !dbg !28
  %943 = insertelement <4 x float> %940, float %942, i64 1, !dbg !30
  %944 = fadd float %941, %942, !dbg !29
  %945 = call float @__ocml_exp2_f32(float %936), !dbg !28
  %946 = insertelement <4 x float> %943, float %945, i64 2, !dbg !30
  %947 = fadd float %944, %945, !dbg !29
  %948 = call float @__ocml_exp2_f32(float %938), !dbg !28
  %949 = insertelement <4 x float> %946, float %948, i64 3, !dbg !30
  %math.result265 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %949) #8, !dbg !30, !frisk.register_lifetimes !1
  %950 = fptrunc <4 x float> %math.result265 to <4 x half>, !dbg !30
  %math.packed266 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %950) #8, !dbg !29, !frisk.register_lifetimes !1
  %951 = fadd float %947, %948, !dbg !29
  %952 = extractelement <32 x float> %655, i64 24, !dbg !17
  %953 = extractelement <32 x float> %655, i64 25, !dbg !17
  %954 = extractelement <32 x float> %655, i64 26, !dbg !17
  %955 = extractelement <32 x float> %655, i64 27, !dbg !17
  %956 = select i1 %.not, float %953, float %952, !dbg !17
  %957 = bitcast float %956 to i32, !dbg !17
  %958 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %957), !dbg !17
  %959 = bitcast i32 %958 to float, !dbg !17
  %960 = select i1 %.not, float %952, float %959, !dbg !17
  %961 = select i1 %.not, float %959, float %953, !dbg !17
  %962 = select i1 %.not, float %955, float %954, !dbg !17
  %963 = bitcast float %962 to i32, !dbg !17
  %964 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %963), !dbg !17
  %965 = bitcast i32 %964 to float, !dbg !17
  %966 = select i1 %.not, float %954, float %965, !dbg !17
  %967 = select i1 %.not, float %965, float %955, !dbg !17
  %968 = select i1 %.not155, float %966, float %960, !dbg !17
  %969 = bitcast float %968 to i32, !dbg !17
  %970 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %969), !dbg !17
  %971 = bitcast i32 %970 to float, !dbg !17
  %972 = select i1 %.not155, float %960, float %971, !dbg !17
  %973 = select i1 %.not155, float %971, float %966, !dbg !17
  %974 = select i1 %.not155, float %967, float %961, !dbg !17
  %975 = bitcast float %974 to i32, !dbg !17
  %976 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %975), !dbg !17
  %977 = bitcast i32 %976 to float, !dbg !17
  %978 = select i1 %.not155, float %961, float %977, !dbg !17
  %979 = select i1 %.not155, float %977, float %967, !dbg !17
  %compare.local.base953 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 55) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local954 = or i32 %compare.local.base953, 96, !dbg !26
  %compare.low955 = icmp ugt i32 %compare.low.base857, %compare.local954, !dbg !26
  %compare.result956 = select i1 %compare.high.equal855, i1 %compare.low955, i1 %compare.high.order856, !dbg !26
  %980 = select i1 %compare.result956, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %981 = fadd float %972, %980, !dbg !20
  %compare.local.base957 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 56) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local958 = or i32 %compare.local.base957, 97, !dbg !26
  %compare.low959 = icmp ugt i32 %compare.low.base857, %compare.local958, !dbg !26
  %compare.result960 = select i1 %compare.high.equal855, i1 %compare.low959, i1 %compare.high.order856, !dbg !26
  %982 = select i1 %compare.result960, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %983 = fadd float %978, %982, !dbg !20
  %compare.local.base961 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 57) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local962 = or i32 %compare.local.base961, 98, !dbg !26
  %compare.low963 = icmp ugt i32 %compare.low.base857, %compare.local962, !dbg !26
  %compare.result964 = select i1 %compare.high.equal855, i1 %compare.low963, i1 %compare.high.order856, !dbg !26
  %984 = select i1 %compare.result964, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %985 = fadd float %973, %984, !dbg !20
  %compare.local.base965 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 58) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local966 = or i32 %compare.local.base965, 99, !dbg !26
  %compare.low967 = icmp ugt i32 %compare.low.base857, %compare.local966, !dbg !26
  %compare.result968 = select i1 %compare.high.equal855, i1 %compare.low967, i1 %compare.high.order856, !dbg !26
  %986 = select i1 %compare.result968, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %987 = fadd float %979, %986, !dbg !20
  %988 = call float @__ocml_exp2_f32(float %981), !dbg !28
  %989 = insertelement <4 x float> poison, float %988, i64 0, !dbg !30
  %990 = fadd float %951, %988, !dbg !29
  %991 = call float @__ocml_exp2_f32(float %983), !dbg !28
  %992 = insertelement <4 x float> %989, float %991, i64 1, !dbg !30
  %993 = fadd float %990, %991, !dbg !29
  %994 = call float @__ocml_exp2_f32(float %985), !dbg !28
  %995 = insertelement <4 x float> %992, float %994, i64 2, !dbg !30
  %996 = fadd float %993, %994, !dbg !29
  %997 = call float @__ocml_exp2_f32(float %987), !dbg !28
  %998 = insertelement <4 x float> %995, float %997, i64 3, !dbg !30
  %math.result267 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %998) #8, !dbg !30, !frisk.register_lifetimes !1
  %999 = fptrunc <4 x float> %math.result267 to <4 x half>, !dbg !30
  %math.packed268 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %999) #8, !dbg !29, !frisk.register_lifetimes !1
  %1000 = fadd float %996, %997, !dbg !29
  %1001 = extractelement <32 x float> %655, i64 28, !dbg !17
  %1002 = extractelement <32 x float> %655, i64 29, !dbg !17
  %1003 = extractelement <32 x float> %655, i64 30, !dbg !17
  %1004 = extractelement <32 x float> %655, i64 31, !dbg !17
  %1005 = select i1 %.not, float %1002, float %1001, !dbg !17
  %1006 = bitcast float %1005 to i32, !dbg !17
  %1007 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %1006), !dbg !17
  %1008 = bitcast i32 %1007 to float, !dbg !17
  %1009 = select i1 %.not, float %1001, float %1008, !dbg !17
  %1010 = select i1 %.not, float %1008, float %1002, !dbg !17
  %1011 = select i1 %.not, float %1004, float %1003, !dbg !17
  %1012 = bitcast float %1011 to i32, !dbg !17
  %1013 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %1012), !dbg !17
  %1014 = bitcast i32 %1013 to float, !dbg !17
  %1015 = select i1 %.not, float %1003, float %1014, !dbg !17
  %1016 = select i1 %.not, float %1014, float %1004, !dbg !17
  %1017 = select i1 %.not155, float %1015, float %1009, !dbg !17
  %1018 = bitcast float %1017 to i32, !dbg !17
  %1019 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %1018), !dbg !17
  %1020 = bitcast i32 %1019 to float, !dbg !17
  %1021 = select i1 %.not155, float %1009, float %1020, !dbg !17
  %1022 = select i1 %.not155, float %1020, float %1015, !dbg !17
  %1023 = select i1 %.not155, float %1016, float %1010, !dbg !17
  %1024 = bitcast float %1023 to i32, !dbg !17
  %1025 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %1024), !dbg !17
  %1026 = bitcast i32 %1025 to float, !dbg !17
  %1027 = select i1 %.not155, float %1010, float %1026, !dbg !17
  %1028 = select i1 %.not155, float %1026, float %1016, !dbg !17
  %compare.local.base969 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 59) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local970 = or i32 %compare.local.base969, 112, !dbg !26
  %compare.low971 = icmp ugt i32 %compare.low.base857, %compare.local970, !dbg !26
  %compare.result972 = select i1 %compare.high.equal855, i1 %compare.low971, i1 %compare.high.order856, !dbg !26
  %1029 = select i1 %compare.result972, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1030 = fadd float %1021, %1029, !dbg !20
  %compare.local.base973 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 60) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local974 = or i32 %compare.local.base973, 113, !dbg !26
  %compare.low975 = icmp ugt i32 %compare.low.base857, %compare.local974, !dbg !26
  %compare.result976 = select i1 %compare.high.equal855, i1 %compare.low975, i1 %compare.high.order856, !dbg !26
  %1031 = select i1 %compare.result976, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1032 = fadd float %1027, %1031, !dbg !20
  %compare.local.base977 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 61) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local978 = or i32 %compare.local.base977, 114, !dbg !26
  %compare.low979 = icmp ugt i32 %compare.low.base857, %compare.local978, !dbg !26
  %compare.result980 = select i1 %compare.high.equal855, i1 %compare.low979, i1 %compare.high.order856, !dbg !26
  %1033 = select i1 %compare.result980, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1034 = fadd float %1022, %1033, !dbg !20
  %compare.local.base981 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base858, i32 62) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local982 = or i32 %compare.local.base981, 115, !dbg !26
  %compare.low983 = icmp ugt i32 %compare.low.base857, %compare.local982, !dbg !26
  %compare.result984 = select i1 %compare.high.equal855, i1 %compare.low983, i1 %compare.high.order856, !dbg !26
  %1035 = select i1 %compare.result984, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1036 = fadd float %1028, %1035, !dbg !20
  %1037 = call float @__ocml_exp2_f32(float %1030), !dbg !28
  %1038 = insertelement <4 x float> poison, float %1037, i64 0, !dbg !30
  %1039 = fadd float %1000, %1037, !dbg !29
  %1040 = call float @__ocml_exp2_f32(float %1032), !dbg !28
  %1041 = insertelement <4 x float> %1038, float %1040, i64 1, !dbg !30
  %1042 = fadd float %1039, %1040, !dbg !29
  %1043 = call float @__ocml_exp2_f32(float %1034), !dbg !28
  %1044 = insertelement <4 x float> %1041, float %1043, i64 2, !dbg !30
  %1045 = fadd float %1042, %1043, !dbg !29
  %1046 = call float @__ocml_exp2_f32(float %1036), !dbg !28
  %1047 = insertelement <4 x float> %1044, float %1046, i64 3, !dbg !30
  %math.result269 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1047) #8, !dbg !30, !frisk.register_lifetimes !1
  %1048 = fptrunc <4 x float> %math.result269 to <4 x half>, !dbg !30
  %math.packed270 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1048) #8, !dbg !29, !frisk.register_lifetimes !1
  %1049 = fadd float %1045, %1046, !dbg !29
  %1050 = bitcast float %654 to i32, !dbg !29
  %1051 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %1050), !dbg !29
  %1052 = bitcast i32 %1051 to float, !dbg !29
  %1053 = fadd float %654, %1052, !dbg !29
  %1054 = bitcast float %1053 to i32, !dbg !29
  %1055 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %1054), !dbg !29
  %1056 = bitcast i32 %1055 to float, !dbg !29
  %1057 = fadd float %1053, %1056, !dbg !29
  %1058 = extractvalue [2 x <1 x float>] %42, 0, !dbg !29
  %1059 = extractelement <1 x float> %1058, i64 0, !dbg !29
  %1060 = fadd float %1057, %1059, !dbg !29
  %1061 = bitcast float %1049 to i32, !dbg !29
  %1062 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %1061), !dbg !29
  %1063 = bitcast i32 %1062 to float, !dbg !29
  %1064 = fadd float %1049, %1063, !dbg !29
  %1065 = bitcast float %1064 to i32, !dbg !29
  %1066 = call i32 @llvm.amdgcn.ds.bpermute(i32 %276, i32 %1065), !dbg !29
  %1067 = bitcast i32 %1066 to float, !dbg !29
  %1068 = fadd float %1064, %1067, !dbg !29
  %1069 = extractvalue [2 x <1 x float>] %42, 1, !dbg !29
  %1070 = extractelement <1 x float> %1069, i64 0, !dbg !29
  %1071 = fadd float %1068, %1070, !dbg !29
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %1072, !dbg !7

1072:                                             ; preds = %1075, %227
  %1073 = phi i64 [ %1130, %1075 ], [ %20, %227 ], !dbg !7
  %1074 = icmp samesign ult i64 %1073, 1024, !dbg !7
  br i1 %1074, label %1075, label %1131, !dbg !7

1075:                                             ; preds = %1072
  %1076 = lshr i64 %1073, 2, !dbg !7
  %1077 = lshr i64 %1073, 5, !dbg !7
  %1078 = and i64 %1077, 1, !dbg !7
  %1079 = and i64 %1076, 8, !dbg !7
  %1080 = lshr i64 %1073, 4, !dbg !7
  %1081 = lshr i64 %1073, 7, !dbg !7
  %1082 = sub nsw i64 %1080, %1081, !dbg !7
  %1083 = shl nsw i64 %1082, 2, !dbg !7
  %1084 = and i64 %1083, 4, !dbg !7
  %1085 = sub nsw i64 %1076, %1078, !dbg !7
  %1086 = and i64 %1085, 3, !dbg !7
  %1087 = lshr i64 %1073, 5, !dbg !7
  %1088 = and i64 %1087, 24, !dbg !7
  %1089 = shl nuw nsw i64 %1073, 1, !dbg !7
  %1090 = and i64 %1089, 4, !dbg !7
  %1091 = or disjoint i64 %1088, %1090, !dbg !7
  %.mask229 = and i64 %1073, 192, !dbg !7
  %1092 = shl nuw nsw i64 %1073, 5, !dbg !7
  %1093 = and i64 %1092, 32, !dbg !7
  %1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %1079, !dbg !7
  %1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1094, i64 %1084, !dbg !7
  %1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1095, i64 %1086, !dbg !7
  %1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1096, i64 %.mask229, !dbg !7
  %.idx230 = shl nuw nsw i64 %1091, 7, !dbg !7
  %1098 = getelementptr inbounds nuw i8, ptr addrspace(3) %1097, i64 %.idx230, !dbg !7
  %1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1098, i64 %1093, !dbg !7
  %1100 = load half, ptr addrspace(3) %1099, align 2, !dbg !7
  %1101 = lshr i64 %1073, 2, !dbg !7
  %1102 = lshr i64 %1073, 5, !dbg !7
  %1103 = and i64 %1102, 1, !dbg !7
  %1104 = and i64 %1101, 8, !dbg !7
  %1105 = lshr i64 %1073, 4, !dbg !7
  %1106 = lshr i64 %1073, 7, !dbg !7
  %1107 = sub nsw i64 %1105, %1106, !dbg !7
  %1108 = shl nsw i64 %1107, 2, !dbg !7
  %1109 = and i64 %1108, 4, !dbg !7
  %1110 = sub nsw i64 %1101, %1103, !dbg !7
  %1111 = and i64 %1110, 3, !dbg !7
  %1112 = lshr i64 %1073, 5, !dbg !7
  %1113 = and i64 %1112, 24, !dbg !7
  %1114 = shl nuw nsw i64 %1073, 1, !dbg !7
  %1115 = and i64 %1114, 4, !dbg !7
  %1116 = or disjoint i64 %1113, %1115, !dbg !7
  %.mask231 = and i64 %1073, 192, !dbg !7
  %1117 = shl nuw nsw i64 %1073, 5, !dbg !7
  %1118 = and i64 %1117, 32, !dbg !7
  %1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %1104, !dbg !7
  %1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1119, i64 %1109, !dbg !7
  %1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1120, i64 %1111, !dbg !7
  %1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1121, i64 %.mask231, !dbg !7
  %.idx232 = shl nuw nsw i64 %1116, 7, !dbg !7
  %1123 = getelementptr inbounds nuw i8, ptr addrspace(3) %1122, i64 %.idx232, !dbg !7
  %1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1123, i64 %1118, !dbg !7
  %1125 = getelementptr inbounds nuw i8, ptr addrspace(3) %1124, i64 32, !dbg !7
  %1126 = load half, ptr addrspace(3) %1125, align 2, !dbg !7
  %1127 = insertelement <2 x half> poison, half %1100, i64 0, !dbg !7
  %1128 = insertelement <2 x half> %1127, half %1126, i64 1, !dbg !7
  %.idx233 = shl nuw nsw i64 %1073, 2, !dbg !7
  %1129 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx233, !dbg !7
  store <2 x half> %1128, ptr addrspace(3) %1129, align 2, !dbg !7
  %1130 = add nuw nsw i64 %1073, 256, !dbg !7
  br label %1072, !dbg !7

1131:                                             ; preds = %1072
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1132 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %232, !dbg !7
  %1133 = getelementptr [2 x i8], ptr addrspace(1) %1132, i64 %38, !dbg !7
  %1134 = getelementptr i8, ptr addrspace(1) %1133, i64 32, !dbg !7
  %address.thread487 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !7, !range !35, !frisk.register_lifetimes !1
  %address.local488 = zext nneg i32 %address.thread487 to i64, !dbg !16
  %address.local489 = shl nuw nsw i64 %address.local488, 1, !dbg !15
  %address.local490 = and i64 %address.local489, 384, !dbg !15
  %address.local491 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %address.local490, !dbg !7
  %address.local492 = ptrtoint ptr addrspace(3) %address.local491 to i64, !dbg !7
  %address.local493 = trunc i64 %address.local492 to i32, !dbg !7
  %1135 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local493), !dbg !7
  %1136 = zext i32 %1135 to i64, !dbg !7
  %1137 = inttoptr i64 %1136 to ptr addrspace(3), !dbg !7
  %address.local494 = shl nuw nsw i32 %address.thread487, 1, !dbg !6
  %address.local495 = and i32 %address.local494, 14, !dbg !6
  %address.local496 = shl nuw nsw i32 %address.thread487, 4, !dbg !6
  %address.local497 = and i32 %address.local496, 3968, !dbg !6
  %address.local498 = or disjoint i32 %address.local495, %address.local497, !dbg !6
  %address.local499 = zext nneg i32 %address.local498 to i64, !dbg !6
  %address.local500 = getelementptr [2 x i8], ptr addrspace(1) %1134, i64 %address.local499, !dbg !7
  %dma.source.bits1018 = ptrtoint ptr addrspace(1) %address.local500 to i64, !dbg !7
  %dma.source.opaque1019 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1018) #9, !dbg !7
  %dma.source.ptr1020 = inttoptr i64 %dma.source.opaque1019 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1020, ptr addrspace(3) %1137, i32 4, i32 0, i32 0), !dbg !7
  %address.local501 = shl nuw nsw i64 %address.local488, 1, !dbg !15
  %address.local502 = and i64 %address.local501, 384, !dbg !15
  %address.local503 = or disjoint i64 %address.local502, 512, !dbg !15
  %address.local504 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %address.local503, !dbg !7
  %address.local505 = ptrtoint ptr addrspace(3) %address.local504 to i64, !dbg !7
  %address.local506 = trunc i64 %address.local505 to i32, !dbg !7
  %1138 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local506), !dbg !7
  %1139 = zext i32 %1138 to i64, !dbg !7
  %1140 = inttoptr i64 %1139 to ptr addrspace(3), !dbg !7
  %address.local507 = shl nuw nsw i32 %address.thread487, 1, !dbg !6
  %address.local508 = and i32 %address.local507, 14, !dbg !6
  %address.local509 = shl nuw nsw i32 %address.thread487, 4, !dbg !6
  %address.local510 = and i32 %address.local509, 3968, !dbg !6
  %address.local511 = or disjoint i32 %address.local508, %address.local510, !dbg !6
  %address.local512 = or disjoint i32 %address.local511, 4096, !dbg !6
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !6
  %address.local514 = getelementptr [2 x i8], ptr addrspace(1) %1134, i64 %address.local513, !dbg !7
  %dma.source.bits1021 = ptrtoint ptr addrspace(1) %address.local514 to i64, !dbg !7
  %dma.source.opaque1022 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1021) #9, !dbg !7
  %dma.source.ptr1023 = inttoptr i64 %dma.source.opaque1022 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1023, ptr addrspace(3) %1140, i32 4, i32 0, i32 0), !dbg !7
  %address.local515 = shl nuw nsw i64 %address.local488, 1, !dbg !15
  %address.local516 = and i64 %address.local515, 384, !dbg !15
  %address.local517 = or disjoint i64 %address.local516, 1024, !dbg !15
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %address.local517, !dbg !7
  %address.local519 = ptrtoint ptr addrspace(3) %address.local518 to i64, !dbg !7
  %address.local520 = trunc i64 %address.local519 to i32, !dbg !7
  %1141 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local520), !dbg !7
  %1142 = zext i32 %1141 to i64, !dbg !7
  %1143 = inttoptr i64 %1142 to ptr addrspace(3), !dbg !7
  %address.local521 = shl nuw nsw i32 %address.thread487, 1, !dbg !6
  %address.local522 = and i32 %address.local521, 14, !dbg !6
  %address.local523 = shl nuw nsw i32 %address.thread487, 4, !dbg !6
  %address.local524 = and i32 %address.local523, 3968, !dbg !6
  %address.local525 = or disjoint i32 %address.local522, %address.local524, !dbg !6
  %address.local526 = or disjoint i32 %address.local525, 8192, !dbg !6
  %address.local527 = zext nneg i32 %address.local526 to i64, !dbg !6
  %address.local528 = getelementptr [2 x i8], ptr addrspace(1) %1134, i64 %address.local527, !dbg !7
  %dma.source.bits1024 = ptrtoint ptr addrspace(1) %address.local528 to i64, !dbg !7
  %dma.source.opaque1025 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1024) #9, !dbg !7
  %dma.source.ptr1026 = inttoptr i64 %dma.source.opaque1025 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1026, ptr addrspace(3) %1143, i32 4, i32 0, i32 0), !dbg !7
  %address.local529 = shl nuw nsw i64 %address.local488, 1, !dbg !15
  %address.local530 = and i64 %address.local529, 384, !dbg !15
  %address.local531 = or disjoint i64 %address.local530, 1536, !dbg !15
  %address.local532 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %address.local531, !dbg !7
  %address.local533 = ptrtoint ptr addrspace(3) %address.local532 to i64, !dbg !7
  %address.local534 = trunc i64 %address.local533 to i32, !dbg !7
  %1144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local534), !dbg !7
  %1145 = zext i32 %1144 to i64, !dbg !7
  %1146 = inttoptr i64 %1145 to ptr addrspace(3), !dbg !7
  %address.local535 = shl nuw nsw i32 %address.thread487, 1, !dbg !6
  %address.local536 = and i32 %address.local535, 14, !dbg !6
  %address.local537 = shl nuw nsw i32 %address.thread487, 4, !dbg !6
  %address.local538 = and i32 %address.local537, 3968, !dbg !6
  %address.local539 = or disjoint i32 %address.local536, %address.local538, !dbg !6
  %address.local540 = or disjoint i32 %address.local539, 12288, !dbg !6
  %address.local541 = zext nneg i32 %address.local540 to i64, !dbg !6
  %address.local542 = getelementptr [2 x i8], ptr addrspace(1) %1134, i64 %address.local541, !dbg !7
  %dma.source.bits1027 = ptrtoint ptr addrspace(1) %address.local542 to i64, !dbg !7
  %dma.source.opaque1028 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1027) #9, !dbg !7
  %dma.source.ptr1029 = inttoptr i64 %dma.source.opaque1028 to ptr addrspace(1), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1029, ptr addrspace(3) %1146, i32 4, i32 0, i32 0), !dbg !7
  call void @llvm.amdgcn.asyncmark(), !dbg !7
  %address.local543 = lshr i32 %address.thread487, 5, !dbg !8
  %address.local544 = and i32 %address.local543, 1, !dbg !8
  %address.local545 = lshr i32 %address.thread487, 2, !dbg !8
  %address.local546 = and i32 %address.local545, 3, !dbg !8
  %address.local547 = add nuw nsw i32 %address.local544, %address.local546, !dbg !8
  %address.local548 = lshr i32 %address.local547, 1, !dbg !8
  %address.local549 = zext nneg i32 %address.local548 to i64, !dbg !8
  %address.local550 = mul nsw i64 %address.local549, -64, !dbg !8
  %address.local551 = lshr i32 %address.thread487, 3, !dbg !8
  %address.local552 = and i32 %address.local551, 1, !dbg !8
  %address.local553 = mul nuw nsw i32 %address.local552, 72, !dbg !8
  %address.local554 = shl nuw nsw i32 %address.thread487, 3, !dbg !8
  %address.local555 = and i32 %address.local554, 504, !dbg !8
  %address.local556 = add nuw nsw i32 %address.local553, %address.local555, !dbg !8
  %address.local557 = shl nuw nsw i32 %address.thread487, 3, !dbg !8
  %address.local558 = and i32 %address.local557, 96, !dbg !8
  %address.local559 = add nuw nsw i32 %address.local556, %address.local558, !dbg !8
  %address.local560 = and i32 %address.thread487, 32, !dbg !8
  %address.local561 = add nuw nsw i32 %address.local559, %address.local560, !dbg !8
  %address.local562 = zext nneg i32 %address.local561 to i64, !dbg !8
  %address.local563 = add nsw i64 %address.local550, %address.local562, !dbg !8
  %address.local564 = and i32 %address.thread487, 15, !dbg !8
  %address.local565 = add nuw nsw i32 %address.local564, %address.local552, !dbg !8
  %address.local566 = lshr i32 %address.local565, 2, !dbg !8
  %address.local567 = zext nneg i32 %address.local566 to i64, !dbg !8
  %address.local568 = mul nsw i64 %address.local567, -32, !dbg !8
  %address.local569 = add nsw i64 %address.local563, %address.local568, !dbg !8
  %address.local570 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local569, !dbg !8
  %1147 = load <8 x half>, ptr addrspace(3) %address.local570, align 2, !dbg !8
  %1148 = shufflevector <8 x half> %1147, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1148, <4 x float> zeroinitializer) #8, !dbg !8
  %1150 = shufflevector <8 x half> %1147, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1151 = extractvalue [2 x <4 x float>] %41, 0, !dbg !31
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1150, <4 x float> %1149) #8, !dbg !8
  %1153 = extractvalue [2 x <4 x float>] %41, 1, !dbg !31
  %address.local571 = shl nuw nsw i32 %address.thread487, 3, !dbg !8
  %address.local572 = and i32 %address.local571, 1656, !dbg !8
  %address.local573 = shl nuw nsw i32 %address.thread487, 4, !dbg !8
  %address.local574 = and i32 %address.local573, 768, !dbg !8
  %address.local575 = add nuw nsw i32 %address.local572, %address.local574, !dbg !8
  %address.local576 = add nuw nsw i32 %address.local575, %address.local560, !dbg !8
  %address.local577 = zext nneg i32 %address.local576 to i64, !dbg !8
  %address.local578 = shl nuw nsw i32 %address.thread487, 3, !dbg !8
  %address.local579 = and i32 %address.local578, 1920, !dbg !8
  %address.local580 = sub nsw i32 0, %address.local579, !dbg !8
  %address.local581 = sext i32 %address.local580 to i64, !dbg !8
  %address.local582 = add nsw i64 %address.local577, %address.local581, !dbg !8
  %address.local583 = zext nneg i32 %address.local558 to i64, !dbg !8
  %address.local584 = add nsw i64 %address.local582, %address.local583, !dbg !8
  %address.local585 = zext nneg i32 %address.local553 to i64, !dbg !8
  %address.local586 = add nsw i64 %address.local584, %address.local585, !dbg !8
  %address.local587 = add nsw i64 %address.local586, %address.local550, !dbg !8
  %address.local588 = add nsw i64 %address.local587, %address.local568, !dbg !8
  %address.local589 = add nsw i64 %address.local588, 512, !dbg !8
  %address.local590 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local589, !dbg !8
  %1154 = load <8 x half>, ptr addrspace(3) %address.local590, align 2, !dbg !8
  %address.local591 = add nsw i64 %address.local588, 1024, !dbg !8
  %address.local592 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local591, !dbg !8
  %1155 = load <8 x half>, ptr addrspace(3) %address.local592, align 2, !dbg !8
  %address.local593 = add nsw i64 %address.local588, 1536, !dbg !8
  %address.local594 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local593, !dbg !8
  %1156 = load <8 x half>, ptr addrspace(3) %address.local594, align 2, !dbg !8
  %1157 = load <8 x half>, ptr addrspace(3) %address.local570, align 2, !dbg !8
  %1158 = shufflevector <8 x half> %1154, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %1159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1158, <4 x float> %1152) #8, !dbg !8
  %1160 = shufflevector <8 x half> %1154, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1160, <4 x float> %1159) #8, !dbg !8
  %1162 = shufflevector <8 x half> %1155, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %1163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1162, <4 x float> %1161) #8, !dbg !8
  %1164 = shufflevector <8 x half> %1155, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1164, <4 x float> %1163) #8, !dbg !8
  %1166 = shufflevector <8 x half> %1156, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %1167 = shufflevector <8 x half> %1156, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1166, <4 x float> %1165) #8, !dbg !8
  %1169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1167, <4 x float> %1168) #8, !dbg !8
  %1170 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %1171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1170, <4 x float> zeroinitializer) #8, !dbg !8
  %1172 = fadd <4 x float> %1151, %1169, !dbg !31
  %1173 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1173, <4 x float> %1171) #8, !dbg !8
  %1175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1158, <4 x float> %1174) #8, !dbg !8
  %1176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1160, <4 x float> %1175) #8, !dbg !8
  %1177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1162, <4 x float> %1176) #8, !dbg !8
  %1178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1164, <4 x float> %1177) #8, !dbg !8
  %1179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1166, <4 x float> %1178) #8, !dbg !8
  %1180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1167, <4 x float> %1179) #8, !dbg !8
  %1181 = fadd <4 x float> %1153, %1180, !dbg !31
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %1182, !dbg !7

1182:                                             ; preds = %1185, %1131
  %1183 = phi i64 [ %1240, %1185 ], [ %20, %1131 ], !dbg !7
  %1184 = icmp samesign ult i64 %1183, 1024, !dbg !7
  br i1 %1184, label %1185, label %1241, !dbg !7

1185:                                             ; preds = %1182
  %1186 = lshr i64 %1183, 2, !dbg !7
  %1187 = lshr i64 %1183, 5, !dbg !7
  %1188 = and i64 %1187, 1, !dbg !7
  %1189 = and i64 %1186, 8, !dbg !7
  %1190 = lshr i64 %1183, 4, !dbg !7
  %1191 = lshr i64 %1183, 7, !dbg !7
  %1192 = sub nsw i64 %1190, %1191, !dbg !7
  %1193 = shl nsw i64 %1192, 2, !dbg !7
  %1194 = and i64 %1193, 4, !dbg !7
  %1195 = sub nsw i64 %1186, %1188, !dbg !7
  %1196 = and i64 %1195, 3, !dbg !7
  %1197 = lshr i64 %1183, 5, !dbg !7
  %1198 = and i64 %1197, 24, !dbg !7
  %1199 = shl nuw nsw i64 %1183, 1, !dbg !7
  %1200 = and i64 %1199, 4, !dbg !7
  %1201 = or disjoint i64 %1198, %1200, !dbg !7
  %.mask = and i64 %1183, 192, !dbg !7
  %1202 = shl nuw nsw i64 %1183, 5, !dbg !7
  %1203 = and i64 %1202, 32, !dbg !7
  %1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %1189, !dbg !7
  %1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1204, i64 %1194, !dbg !7
  %1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1205, i64 %1196, !dbg !7
  %1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1206, i64 %.mask, !dbg !7
  %.idx = shl nuw nsw i64 %1201, 7, !dbg !7
  %1208 = getelementptr inbounds nuw i8, ptr addrspace(3) %1207, i64 %.idx, !dbg !7
  %1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1208, i64 %1203, !dbg !7
  %1210 = load half, ptr addrspace(3) %1209, align 2, !dbg !7
  %1211 = lshr i64 %1183, 2, !dbg !7
  %1212 = lshr i64 %1183, 5, !dbg !7
  %1213 = and i64 %1212, 1, !dbg !7
  %1214 = and i64 %1211, 8, !dbg !7
  %1215 = lshr i64 %1183, 4, !dbg !7
  %1216 = lshr i64 %1183, 7, !dbg !7
  %1217 = sub nsw i64 %1215, %1216, !dbg !7
  %1218 = shl nsw i64 %1217, 2, !dbg !7
  %1219 = and i64 %1218, 4, !dbg !7
  %1220 = sub nsw i64 %1211, %1213, !dbg !7
  %1221 = and i64 %1220, 3, !dbg !7
  %1222 = lshr i64 %1183, 5, !dbg !7
  %1223 = and i64 %1222, 24, !dbg !7
  %1224 = shl nuw nsw i64 %1183, 1, !dbg !7
  %1225 = and i64 %1224, 4, !dbg !7
  %1226 = or disjoint i64 %1223, %1225, !dbg !7
  %.mask226 = and i64 %1183, 192, !dbg !7
  %1227 = shl nuw nsw i64 %1183, 5, !dbg !7
  %1228 = and i64 %1227, 32, !dbg !7
  %1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %1214, !dbg !7
  %1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1229, i64 %1219, !dbg !7
  %1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1230, i64 %1221, !dbg !7
  %1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1231, i64 %.mask226, !dbg !7
  %.idx227 = shl nuw nsw i64 %1226, 7, !dbg !7
  %1233 = getelementptr inbounds nuw i8, ptr addrspace(3) %1232, i64 %.idx227, !dbg !7
  %1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1233, i64 %1228, !dbg !7
  %1235 = getelementptr inbounds nuw i8, ptr addrspace(3) %1234, i64 32, !dbg !7
  %1236 = load half, ptr addrspace(3) %1235, align 2, !dbg !7
  %1237 = insertelement <2 x half> poison, half %1210, i64 0, !dbg !7
  %1238 = insertelement <2 x half> %1237, half %1236, i64 1, !dbg !7
  %.idx228 = shl nuw nsw i64 %1183, 2, !dbg !7
  %1239 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %.idx228, !dbg !7
  store <2 x half> %1238, ptr addrspace(3) %1239, align 2, !dbg !7
  %1240 = add nuw nsw i64 %1183, 256, !dbg !7
  br label %1182, !dbg !7

1241:                                             ; preds = %1182
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %address.thread595 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !8, !range !35, !frisk.register_lifetimes !1
  %address.local596 = lshr i32 %address.thread595, 5, !dbg !8
  %address.local597 = and i32 %address.local596, 1, !dbg !8
  %address.local598 = lshr i32 %address.thread595, 2, !dbg !8
  %address.local599 = and i32 %address.local598, 3, !dbg !8
  %address.local600 = add nuw nsw i32 %address.local597, %address.local599, !dbg !8
  %address.local601 = lshr i32 %address.local600, 1, !dbg !8
  %address.local602 = zext nneg i32 %address.local601 to i64, !dbg !8
  %address.local603 = mul nsw i64 %address.local602, -64, !dbg !8
  %address.local604 = lshr i32 %address.thread595, 3, !dbg !8
  %address.local605 = and i32 %address.local604, 1, !dbg !8
  %address.local606 = mul nuw nsw i32 %address.local605, 72, !dbg !8
  %address.local607 = shl nuw nsw i32 %address.thread595, 3, !dbg !8
  %address.local608 = and i32 %address.local607, 504, !dbg !8
  %address.local609 = add nuw nsw i32 %address.local606, %address.local608, !dbg !8
  %address.local610 = shl nuw nsw i32 %address.thread595, 3, !dbg !8
  %address.local611 = and i32 %address.local610, 96, !dbg !8
  %address.local612 = add nuw nsw i32 %address.local609, %address.local611, !dbg !8
  %address.local613 = and i32 %address.thread595, 32, !dbg !8
  %address.local614 = add nuw nsw i32 %address.local612, %address.local613, !dbg !8
  %address.local615 = zext nneg i32 %address.local614 to i64, !dbg !8
  %address.local616 = add nsw i64 %address.local603, %address.local615, !dbg !8
  %address.local617 = and i32 %address.thread595, 15, !dbg !8
  %address.local618 = add nuw nsw i32 %address.local617, %address.local605, !dbg !8
  %address.local619 = lshr i32 %address.local618, 2, !dbg !8
  %address.local620 = zext nneg i32 %address.local619 to i64, !dbg !8
  %address.local621 = mul nsw i64 %address.local620, -32, !dbg !8
  %address.local622 = add nsw i64 %address.local616, %address.local621, !dbg !8
  %address.local623 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local622, !dbg !8
  %1242 = load <8 x half>, ptr addrspace(3) %address.local623, align 2, !dbg !8
  %1243 = shufflevector <8 x half> %1242, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %1244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1243, <4 x float> zeroinitializer) #8, !dbg !8
  %1245 = shufflevector <8 x half> %1242, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1246 = extractvalue [2 x <4 x float>] %43, 0, !dbg !31
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1245, <4 x float> %1244) #8, !dbg !8
  %1248 = extractvalue [2 x <4 x float>] %43, 1, !dbg !31
  %address.local624 = shl nuw nsw i32 %address.thread595, 3, !dbg !8
  %address.local625 = and i32 %address.local624, 1656, !dbg !8
  %address.local626 = shl nuw nsw i32 %address.thread595, 4, !dbg !8
  %address.local627 = and i32 %address.local626, 768, !dbg !8
  %address.local628 = add nuw nsw i32 %address.local625, %address.local627, !dbg !8
  %address.local629 = add nuw nsw i32 %address.local628, %address.local613, !dbg !8
  %address.local630 = zext nneg i32 %address.local629 to i64, !dbg !8
  %address.local631 = shl nuw nsw i32 %address.thread595, 3, !dbg !8
  %address.local632 = and i32 %address.local631, 1920, !dbg !8
  %address.local633 = sub nsw i32 0, %address.local632, !dbg !8
  %address.local634 = sext i32 %address.local633 to i64, !dbg !8
  %address.local635 = add nsw i64 %address.local630, %address.local634, !dbg !8
  %address.local636 = zext nneg i32 %address.local611 to i64, !dbg !8
  %address.local637 = add nsw i64 %address.local635, %address.local636, !dbg !8
  %address.local638 = zext nneg i32 %address.local606 to i64, !dbg !8
  %address.local639 = add nsw i64 %address.local637, %address.local638, !dbg !8
  %address.local640 = add nsw i64 %address.local639, %address.local603, !dbg !8
  %address.local641 = add nsw i64 %address.local640, %address.local621, !dbg !8
  %address.local642 = add nsw i64 %address.local641, 512, !dbg !8
  %address.local643 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local642, !dbg !8
  %1249 = load <8 x half>, ptr addrspace(3) %address.local643, align 2, !dbg !8
  %address.local644 = add nsw i64 %address.local641, 1024, !dbg !8
  %address.local645 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local644, !dbg !8
  %1250 = load <8 x half>, ptr addrspace(3) %address.local645, align 2, !dbg !8
  %address.local646 = add nsw i64 %address.local641, 1536, !dbg !8
  %address.local647 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local646, !dbg !8
  %1251 = load <8 x half>, ptr addrspace(3) %address.local647, align 2, !dbg !8
  %1252 = load <8 x half>, ptr addrspace(3) %address.local623, align 2, !dbg !8
  %1253 = insertvalue [2 x <4 x float>] poison, <4 x float> %1172, 0, !dbg !19
  %1254 = insertvalue [2 x <4 x float>] %1253, <4 x float> %1181, 1, !dbg !19
  %1255 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1255, <4 x float> %1247) #8, !dbg !8
  %1257 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1257, <4 x float> %1256) #8, !dbg !8
  %1259 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1259, <4 x float> %1258) #8, !dbg !8
  %1261 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1261, <4 x float> %1260) #8, !dbg !8
  %1263 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %1264 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1263, <4 x float> %1262) #8, !dbg !8
  %1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1264, <4 x float> %1265) #8, !dbg !8
  %1267 = shufflevector <8 x half> %1252, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1267, <4 x float> zeroinitializer) #8, !dbg !8
  %1269 = fadd <4 x float> %1246, %1266, !dbg !31
  %1270 = shufflevector <8 x half> %1252, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1270, <4 x float> %1268) #8, !dbg !8
  %1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1255, <4 x float> %1271) #8, !dbg !8
  %1273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1257, <4 x float> %1272) #8, !dbg !8
  %1274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1259, <4 x float> %1273) #8, !dbg !8
  %1275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1261, <4 x float> %1274) #8, !dbg !8
  %1276 = insertelement <1 x float> poison, float %1060, i64 0, !dbg !19
  %1277 = insertvalue [2 x <1 x float>] poison, <1 x float> %1276, 0, !dbg !19
  %1278 = insertelement <1 x float> poison, float %1071, i64 0, !dbg !19
  %1279 = insertvalue [2 x <1 x float>] %1277, <1 x float> %1278, 1, !dbg !19
  %1280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1263, <4 x float> %1275) #8, !dbg !8
  %1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1264, <4 x float> %1280) #8, !dbg !8
  %1282 = fadd <4 x float> %1248, %1281, !dbg !31
  %1283 = insertvalue [2 x <4 x float>] poison, <4 x float> %1269, 0, !dbg !19
  %1284 = insertvalue [2 x <4 x float>] %1283, <4 x float> %1282, 1, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1285 = add nuw nsw i64 %40, 1, !dbg !19
  br label %39, !dbg !19

1286:                                             ; preds = %39
  %1287 = extractvalue [2 x <4 x float>] %43, 0, !dbg !19
  %1288 = extractelement <4 x float> %1287, i64 0, !dbg !19
  %1289 = extractelement <4 x float> %1287, i64 1, !dbg !19
  %1290 = extractelement <4 x float> %1287, i64 2, !dbg !19
  %1291 = extractelement <4 x float> %1287, i64 3, !dbg !19
  %1292 = select i1 %.not, float %1289, float %1288, !dbg !19
  %1293 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !19
  %1294 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1293), !dbg !19
  %1295 = add nuw nsw i32 %1294, 64, !dbg !19
  %1296 = xor i32 %1294, 16, !dbg !19
  %1297 = and i32 %1295, 192, !dbg !19
  %1298 = icmp samesign ult i32 %1296, %1297, !dbg !19
  %1299 = select i1 %1298, i32 %1296, i32 %1294, !dbg !19
  %1300 = shl nuw nsw i32 %1299, 2, !dbg !19
  %1301 = bitcast float %1292 to i32, !dbg !19
  %1302 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1301), !dbg !19
  %1303 = bitcast i32 %1302 to float, !dbg !19
  %1304 = select i1 %.not, float %1288, float %1303, !dbg !19
  %1305 = select i1 %.not, float %1303, float %1289, !dbg !19
  %1306 = select i1 %.not, float %1291, float %1290, !dbg !19
  %1307 = bitcast float %1306 to i32, !dbg !19
  %1308 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1307), !dbg !19
  %1309 = bitcast i32 %1308 to float, !dbg !19
  %1310 = select i1 %.not, float %1290, float %1309, !dbg !19
  %1311 = select i1 %.not, float %1309, float %1291, !dbg !19
  %1312 = select i1 %.not155, float %1310, float %1304, !dbg !19
  %1313 = xor i32 %1294, 32, !dbg !19
  %1314 = icmp samesign ult i32 %1313, %1297, !dbg !19
  %1315 = select i1 %1314, i32 %1313, i32 %1294, !dbg !19
  %1316 = shl nuw nsw i32 %1315, 2, !dbg !19
  %1317 = bitcast float %1312 to i32, !dbg !19
  %1318 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1316, i32 %1317), !dbg !19
  %1319 = bitcast i32 %1318 to float, !dbg !19
  %1320 = select i1 %.not155, float %1304, float %1319, !dbg !19
  %1321 = select i1 %.not155, float %1319, float %1310, !dbg !19
  %1322 = select i1 %.not155, float %1311, float %1305, !dbg !19
  %1323 = bitcast float %1322 to i32, !dbg !19
  %1324 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1316, i32 %1323), !dbg !19
  %1325 = bitcast i32 %1324 to float, !dbg !19
  %1326 = select i1 %.not155, float %1305, float %1325, !dbg !19
  %1327 = select i1 %.not155, float %1325, float %1311, !dbg !19
  %1328 = extractvalue [2 x <4 x float>] %41, 0, !dbg !19
  %1329 = extractelement <4 x float> %1328, i64 0, !dbg !19
  %1330 = extractelement <4 x float> %1328, i64 1, !dbg !19
  %1331 = extractelement <4 x float> %1328, i64 2, !dbg !19
  %1332 = extractelement <4 x float> %1328, i64 3, !dbg !19
  %1333 = select i1 %.not, float %1330, float %1329, !dbg !19
  %1334 = bitcast float %1333 to i32, !dbg !19
  %1335 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1334), !dbg !19
  %1336 = bitcast i32 %1335 to float, !dbg !19
  %1337 = select i1 %.not, float %1329, float %1336, !dbg !19
  %1338 = select i1 %.not, float %1336, float %1330, !dbg !19
  %1339 = select i1 %.not, float %1332, float %1331, !dbg !19
  %1340 = bitcast float %1339 to i32, !dbg !19
  %1341 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1340), !dbg !19
  %1342 = bitcast i32 %1341 to float, !dbg !19
  %1343 = select i1 %.not, float %1331, float %1342, !dbg !19
  %1344 = select i1 %.not, float %1342, float %1332, !dbg !19
  %1345 = select i1 %.not155, float %1343, float %1337, !dbg !19
  %1346 = bitcast float %1345 to i32, !dbg !19
  %1347 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1316, i32 %1346), !dbg !19
  %1348 = select i1 %.not155, float %1344, float %1338, !dbg !19
  %1349 = bitcast float %1348 to i32, !dbg !19
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1316, i32 %1349), !dbg !19
  %1351 = extractvalue [2 x <1 x float>] %42, 0, !dbg !24
  %1352 = shufflevector <1 x float> %1351, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %1353 = extractvalue [2 x <4 x float>] %43, 1, !dbg !19
  %1354 = extractelement <4 x float> %1353, i64 0, !dbg !19
  %1355 = extractelement <4 x float> %1353, i64 1, !dbg !19
  %1356 = extractelement <4 x float> %1353, i64 2, !dbg !19
  %1357 = extractelement <4 x float> %1353, i64 3, !dbg !19
  %1358 = select i1 %.not, float %1355, float %1354, !dbg !19
  %1359 = bitcast float %1358 to i32, !dbg !19
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1359), !dbg !19
  %1361 = bitcast i32 %1360 to float, !dbg !19
  %1362 = select i1 %.not, float %1354, float %1361, !dbg !19
  %1363 = select i1 %.not, float %1361, float %1355, !dbg !19
  %1364 = select i1 %.not, float %1357, float %1356, !dbg !19
  %1365 = bitcast float %1364 to i32, !dbg !19
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1365), !dbg !19
  %1367 = bitcast i32 %1366 to float, !dbg !19
  %1368 = select i1 %.not, float %1356, float %1367, !dbg !19
  %1369 = select i1 %.not, float %1367, float %1357, !dbg !19
  %1370 = select i1 %.not155, float %1368, float %1362, !dbg !19
  %1371 = bitcast float %1370 to i32, !dbg !19
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1316, i32 %1371), !dbg !19
  %1373 = bitcast i32 %1372 to float, !dbg !19
  %1374 = select i1 %.not155, float %1362, float %1373, !dbg !19
  %1375 = select i1 %.not155, float %1373, float %1368, !dbg !19
  %1376 = select i1 %.not155, float %1369, float %1363, !dbg !19
  %1377 = bitcast float %1376 to i32, !dbg !19
  %1378 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1316, i32 %1377), !dbg !19
  %1379 = bitcast i32 %1378 to float, !dbg !19
  %1380 = select i1 %.not155, float %1363, float %1379, !dbg !19
  %1381 = select i1 %.not155, float %1379, float %1369, !dbg !19
  %1382 = extractvalue [2 x <4 x float>] %41, 1, !dbg !19
  %1383 = extractelement <4 x float> %1382, i64 0, !dbg !19
  %1384 = extractelement <4 x float> %1382, i64 1, !dbg !19
  %1385 = extractelement <4 x float> %1382, i64 2, !dbg !19
  %1386 = extractelement <4 x float> %1382, i64 3, !dbg !19
  %1387 = select i1 %.not, float %1384, float %1383, !dbg !19
  %1388 = bitcast float %1387 to i32, !dbg !19
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1388), !dbg !19
  %1390 = bitcast i32 %1389 to float, !dbg !19
  %1391 = select i1 %.not, float %1383, float %1390, !dbg !19
  %1392 = select i1 %.not, float %1390, float %1384, !dbg !19
  %1393 = select i1 %.not, float %1386, float %1385, !dbg !19
  %1394 = bitcast float %1393 to i32, !dbg !19
  %1395 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1394), !dbg !19
  %1396 = bitcast i32 %1395 to float, !dbg !19
  %1397 = select i1 %.not, float %1385, float %1396, !dbg !19
  %1398 = select i1 %.not, float %1396, float %1386, !dbg !19
  %1399 = select i1 %.not155, float %1397, float %1391, !dbg !19
  %1400 = bitcast float %1399 to i32, !dbg !19
  %1401 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1316, i32 %1400), !dbg !19
  %1402 = select i1 %.not155, float %1398, float %1392, !dbg !19
  %1403 = bitcast float %1402 to i32, !dbg !19
  %1404 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1316, i32 %1403), !dbg !19
  %1405 = extractvalue [2 x <1 x float>] %42, 1, !dbg !24
  %1406 = shufflevector <1 x float> %1405, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  br i1 true, label %1407, label %1441, !dbg !18

1407:                                             ; preds = %1286
  %1408 = bitcast i32 %1401 to float, !dbg !19
  %1409 = select i1 %.not155, float %1391, float %1408, !dbg !19
  %1410 = insertelement <4 x float> poison, float %1409, i64 0, !dbg !24
  %1411 = bitcast i32 %1404 to float, !dbg !19
  %1412 = select i1 %.not155, float %1392, float %1411, !dbg !19
  %1413 = insertelement <4 x float> %1410, float %1412, i64 1, !dbg !24
  %1414 = select i1 %.not155, float %1408, float %1397, !dbg !19
  %1415 = insertelement <4 x float> %1413, float %1414, i64 2, !dbg !24
  %1416 = select i1 %.not155, float %1411, float %1398, !dbg !19
  %1417 = insertelement <4 x float> %1415, float %1416, i64 3, !dbg !24
  %1418 = fdiv <4 x float> %1417, %1406, !dbg !24
  %1419 = fptrunc <4 x float> %1418 to <4 x half>, !dbg !25
  %1420 = extractelement <4 x half> %1419, i64 3, !dbg !25
  %1421 = extractelement <4 x half> %1419, i64 2, !dbg !25
  %1422 = extractelement <4 x half> %1419, i64 1, !dbg !25
  %1423 = extractelement <4 x half> %1419, i64 0, !dbg !25
  %1424 = bitcast i32 %1347 to float, !dbg !19
  %1425 = select i1 %.not155, float %1337, float %1424, !dbg !19
  %1426 = insertelement <4 x float> poison, float %1425, i64 0, !dbg !24
  %1427 = bitcast i32 %1350 to float, !dbg !19
  %1428 = select i1 %.not155, float %1338, float %1427, !dbg !19
  %1429 = insertelement <4 x float> %1426, float %1428, i64 1, !dbg !24
  %1430 = select i1 %.not155, float %1424, float %1343, !dbg !19
  %1431 = insertelement <4 x float> %1429, float %1430, i64 2, !dbg !24
  %1432 = select i1 %.not155, float %1427, float %1344, !dbg !19
  %1433 = insertelement <4 x float> %1431, float %1432, i64 3, !dbg !24
  %1434 = fdiv <4 x float> %1433, %1352, !dbg !24
  %1435 = fptrunc <4 x float> %1434 to <4 x half>, !dbg !25
  %1436 = extractelement <4 x half> %1435, i64 3, !dbg !25
  %1437 = extractelement <4 x half> %1435, i64 2, !dbg !25
  %1438 = extractelement <4 x half> %1435, i64 1, !dbg !25
  %1439 = extractelement <4 x half> %1435, i64 0, !dbg !25
  %1440 = shl nuw nsw i64 %15, 19, !dbg !18
  %address.thread648 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !18, !range !35, !frisk.register_lifetimes !1
  %address.local649 = zext nneg i32 %address.thread648 to i64, !dbg !16
  %address.local650 = add nsw i64 %16, %address.local649, !dbg !18
  %address.local651 = add nsw i64 %address.local650, %17, !dbg !18
  %address.local652 = lshr i32 %address.thread648, 2, !dbg !20
  %address.local653 = and i32 %address.local652, 48, !dbg !20
  %address.local654 = zext nneg i32 %address.local653 to i64, !dbg !20
  %address.local655 = add nsw i64 %address.local651, %address.local654, !dbg !18
  %address.local656 = and i32 %address.thread648, 240, !dbg !20
  %address.local657 = sub nsw i32 0, %address.local656, !dbg !20
  %address.local658 = sext i32 %address.local657 to i64, !dbg !20
  %address.local659 = add nsw i64 %address.local655, %address.local658, !dbg !18
  %address.local660 = shl nuw nsw i64 %address.local659, 7, !dbg !18
  %address.local661 = add nuw nsw i64 %1440, %address.local660, !dbg !18
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local661, !dbg !18
  %address.local663 = lshr i32 %address.thread648, 2, !dbg !20
  %address.local664 = and i32 %address.local663, 12, !dbg !20
  %address.local665 = zext nneg i32 %address.local664 to i64, !dbg !20
  %address.local666 = or disjoint i64 %38, %address.local665, !dbg !18
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local666, !dbg !18
  store half %1439, ptr addrspace(1) %address.local667, align 2, !dbg !18
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local661, !dbg !18
  %address.local669 = or disjoint i64 %address.local666, 1, !dbg !18
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local668, i64 %address.local669, !dbg !18
  store half %1438, ptr addrspace(1) %address.local670, align 2, !dbg !18
  %address.local671 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local661, !dbg !18
  %address.local672 = or disjoint i64 %address.local666, 2, !dbg !18
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local671, i64 %address.local672, !dbg !18
  store half %1437, ptr addrspace(1) %address.local673, align 2, !dbg !18
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local661, !dbg !18
  %address.local675 = or disjoint i64 %address.local666, 3, !dbg !18
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local674, i64 %address.local675, !dbg !18
  store half %1436, ptr addrspace(1) %address.local676, align 2, !dbg !18
  %address.local677 = shl nsw i64 %address.local659, 7, !dbg !18
  %address.local678 = add i64 %address.local677, 8192, !dbg !18
  %address.local679 = add nuw nsw i64 %1440, %address.local678, !dbg !18
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local679, !dbg !18
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local666, !dbg !18
  store half %1423, ptr addrspace(1) %address.local681, align 2, !dbg !18
  %address.local682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local679, !dbg !18
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local682, i64 %address.local669, !dbg !18
  store half %1422, ptr addrspace(1) %address.local683, align 2, !dbg !18
  %address.local684 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local679, !dbg !18
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local684, i64 %address.local672, !dbg !18
  store half %1421, ptr addrspace(1) %address.local685, align 2, !dbg !18
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local679, !dbg !18
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local686, i64 %address.local675, !dbg !18
  store half %1420, ptr addrspace(1) %address.local687, align 2, !dbg !18
  br label %1441, !dbg !18

1441:                                             ; preds = %1407, %1286
  br i1 true, label %1442, label %1464, !dbg !18

1442:                                             ; preds = %1441
  %1443 = insertelement <4 x float> poison, float %1374, i64 0, !dbg !24
  %1444 = insertelement <4 x float> %1443, float %1380, i64 1, !dbg !24
  %1445 = insertelement <4 x float> %1444, float %1375, i64 2, !dbg !24
  %1446 = insertelement <4 x float> %1445, float %1381, i64 3, !dbg !24
  %1447 = fdiv <4 x float> %1446, %1406, !dbg !24
  %1448 = fptrunc <4 x float> %1447 to <4 x half>, !dbg !25
  %1449 = extractelement <4 x half> %1448, i64 3, !dbg !25
  %1450 = extractelement <4 x half> %1448, i64 2, !dbg !25
  %1451 = extractelement <4 x half> %1448, i64 1, !dbg !25
  %1452 = extractelement <4 x half> %1448, i64 0, !dbg !25
  %1453 = insertelement <4 x float> poison, float %1320, i64 0, !dbg !24
  %1454 = insertelement <4 x float> %1453, float %1326, i64 1, !dbg !24
  %1455 = insertelement <4 x float> %1454, float %1321, i64 2, !dbg !24
  %1456 = insertelement <4 x float> %1455, float %1327, i64 3, !dbg !24
  %1457 = fdiv <4 x float> %1456, %1352, !dbg !24
  %1458 = fptrunc <4 x float> %1457 to <4 x half>, !dbg !25
  %1459 = extractelement <4 x half> %1458, i64 3, !dbg !25
  %1460 = extractelement <4 x half> %1458, i64 2, !dbg !25
  %1461 = extractelement <4 x half> %1458, i64 1, !dbg !25
  %1462 = extractelement <4 x half> %1458, i64 0, !dbg !25
  %1463 = shl nuw nsw i64 %15, 19, !dbg !18
  %address.thread688 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !18, !range !35, !frisk.register_lifetimes !1
  %address.local689 = zext nneg i32 %address.thread688 to i64, !dbg !16
  %address.local690 = add nsw i64 %16, %address.local689, !dbg !18
  %address.local691 = add nsw i64 %address.local690, %17, !dbg !18
  %address.local692 = lshr i32 %address.thread688, 2, !dbg !20
  %address.local693 = and i32 %address.local692, 48, !dbg !20
  %address.local694 = zext nneg i32 %address.local693 to i64, !dbg !20
  %address.local695 = add nsw i64 %address.local691, %address.local694, !dbg !18
  %address.local696 = and i32 %address.thread688, 240, !dbg !20
  %address.local697 = sub nsw i32 0, %address.local696, !dbg !20
  %address.local698 = sext i32 %address.local697 to i64, !dbg !20
  %address.local699 = add nsw i64 %address.local695, %address.local698, !dbg !18
  %address.local700 = shl nuw nsw i64 %address.local699, 7, !dbg !18
  %address.local701 = add nuw nsw i64 %1463, %address.local700, !dbg !18
  %address.local702 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !18
  %address.local703 = lshr i32 %address.thread688, 2, !dbg !20
  %address.local704 = and i32 %address.local703, 12, !dbg !20
  %address.local705 = zext nneg i32 %address.local704 to i64, !dbg !20
  %address.local706 = or disjoint i64 %38, %address.local705, !dbg !18
  %address.local707 = or disjoint i64 %address.local706, 16, !dbg !18
  %address.local708 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local702, i64 %address.local707, !dbg !18
  store half %1462, ptr addrspace(1) %address.local708, align 2, !dbg !18
  %address.local709 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !18
  %address.local710 = or disjoint i64 %address.local706, 17, !dbg !18
  %address.local711 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local709, i64 %address.local710, !dbg !18
  store half %1461, ptr addrspace(1) %address.local711, align 2, !dbg !18
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !18
  %address.local713 = or disjoint i64 %address.local706, 18, !dbg !18
  %address.local714 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local712, i64 %address.local713, !dbg !18
  store half %1460, ptr addrspace(1) %address.local714, align 2, !dbg !18
  %address.local715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !18
  %address.local716 = or disjoint i64 %address.local706, 19, !dbg !18
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local715, i64 %address.local716, !dbg !18
  store half %1459, ptr addrspace(1) %address.local717, align 2, !dbg !18
  %address.local718 = shl nsw i64 %address.local699, 7, !dbg !18
  %address.local719 = add i64 %address.local718, 8192, !dbg !18
  %address.local720 = add nuw nsw i64 %1463, %address.local719, !dbg !18
  %address.local721 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local720, !dbg !18
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local721, i64 %address.local707, !dbg !18
  store half %1452, ptr addrspace(1) %address.local722, align 2, !dbg !18
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local720, !dbg !18
  %address.local724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local723, i64 %address.local710, !dbg !18
  store half %1451, ptr addrspace(1) %address.local724, align 2, !dbg !18
  %address.local725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local720, !dbg !18
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local725, i64 %address.local713, !dbg !18
  store half %1450, ptr addrspace(1) %address.local726, align 2, !dbg !18
  %address.local727 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local720, !dbg !18
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local727, i64 %address.local716, !dbg !18
  store half %1449, ptr addrspace(1) %address.local728, align 2, !dbg !18
  br label %1464, !dbg !18

1464:                                             ; preds = %1442, %1441
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %1465 = add nuw nsw i64 %35, 1, !dbg !22
  br label %34, !dbg !22

1466:                                             ; preds = %34
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
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) captures(none), ptr addrspace(3) captures(none), i32 immarg, i32 immarg, i32 immarg) #7

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #7

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #7

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { nocallback nofree nounwind willreturn }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind memory(none) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!34}

!0 = !DIFile(filename: "test_input_sd16.mlir", directory: "3rd/deepgengraph/test")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!6 = !DILocation(line: 67, column: 22, scope: !4)
!7 = !DILocation(line: 92, column: 20, scope: !4)
!8 = !DILocation(line: 93, column: 23, scope: !4)
!9 = !DILocation(line: 69, column: 26, scope: !4)
!13 = !DILocation(line: 46, column: 17, scope: !4)
!15 = !DILocation(line: 66, column: 22, scope: !4)
!16 = !DILocation(line: 68, column: 24, scope: !4)
!17 = !DILocation(line: 59, column: 17, scope: !4)
!18 = !DILocation(line: 99, column: 9, scope: !4)
!19 = !DILocation(line: 56, column: 24, scope: !4)
!20 = !DILocation(line: 73, column: 19, scope: !4)
!21 = !DILocation(line: 75, column: 24, scope: !4)
!22 = !DILocation(line: 51, column: 7, scope: !4)
!23 = !DILocation(line: 51, column: 18, scope: !4)
!24 = !DILocation(line: 97, column: 23, scope: !4)
!25 = !DILocation(line: 98, column: 19, scope: !4)
!26 = !DILocation(line: 76, column: 23, scope: !4)
!27 = !DILocation(line: 77, column: 26, scope: !4)
!28 = !DILocation(line: 85, column: 22, scope: !4)
!29 = !DILocation(line: 86, column: 20, scope: !4)
!30 = !DILocation(line: 87, column: 16, scope: !4)
!31 = !DILocation(line: 94, column: 20, scope: !4)
!32 = !DILocation(line: 70, column: 23, scope: !4)
!33 = !{i32 256, i32 1, i32 1}
!34 = !{i32 2, !"Debug Info Version", i32 3}
!35 = !{i32 0, i32 256}
