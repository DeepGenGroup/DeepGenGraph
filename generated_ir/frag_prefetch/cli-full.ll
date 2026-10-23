; ModuleID = '3rd/deepgengraph/test/test_input_spidD.mlir'
source_filename = "3rd/deepgengraph/test/test_input_spidD.mlir"

@async_lds_staging_0 = private addrspace(3) global [4096 x half] undef, align 16
@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !33 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %6, 7, !dbg !11
  %.lobit = ashr i32 %5, 31, !dbg !13
  %10 = xor i32 %5, %.lobit, !dbg !13
  %11 = sdiv i32 %10, 32, !dbg !13
  %.lobit313 = ashr i32 %5, 31, !dbg !13
  %12 = xor i32 %11, %.lobit313, !dbg !13
  %13 = sext i32 %12 to i64, !dbg !13
  %14 = add nsw i64 %8, %13, !dbg !13
  %15 = and i64 %14, 31, !dbg !13
  %16 = shl nsw i64 %6, 7, !dbg !13
  %17 = mul nsw i64 %13, -4096, !dbg !13
  %18 = add nsw i64 %16, %17, !dbg !13
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !14, !range !35
  %20 = zext nneg i32 %19 to i64, !dbg !14
  %21 = and i32 %19, 16, !dbg !15
  %.not = icmp eq i32 %21, 0, !dbg !15
  %22 = and i32 %19, 32, !dbg !15
  %.not314 = icmp eq i32 %22, 0, !dbg !15
  %23 = add nsw i64 %9, %20, !dbg !20
  %24 = lshr i32 %19, 2, !dbg !20
  %25 = and i32 %24, 48, !dbg !20
  %index.compact13 = zext nneg i32 %25 to i64, !dbg !20
  %26 = add nsw i64 %23, %index.compact13, !dbg !20
  %27 = and i32 %19, 240, !dbg !20
  %28 = sub nsw i32 0, %27, !dbg !20
  %index.compact12 = sext i32 %28 to i64, !dbg !20
  %29 = add nsw i64 %26, %index.compact12, !dbg !20
  %30 = lshr i32 %19, 2, !dbg !20
  %31 = and i32 %30, 12, !dbg !20
  %index.compact11 = zext nneg i32 %31 to i64, !dbg !20
  %32 = add nsw i64 %29, 1, !dbg !21
  %33 = add nsw i64 %29, 65, !dbg !21
  br label %34, !dbg !22

34:                                               ; preds = %1398, %4
  %35 = phi i64 [ %1399, %1398 ], [ 0, %4 ], !dbg !23
  %36 = icmp samesign ult i64 %35, 4, !dbg !22
  br i1 %36, label %37, label %1400, !dbg !22

37:                                               ; preds = %34
  %38 = shl nuw nsw i64 %35, 5, !dbg !22
  br label %39, !dbg !18

39:                                               ; preds = %286, %37
  %40 = phi i64 [ %1224, %286 ], [ 0, %37 ], !dbg !18
  %41 = phi [2 x <8 x float>] [ %1223, %286 ], [ zeroinitializer, %37 ], !dbg !18
  %42 = phi [2 x <1 x float>] [ %1145, %286 ], [ zeroinitializer, %37 ], !dbg !18
  %.not315 = icmp sgt i64 %40, %6, !dbg !18
  br i1 %.not315, label %1225, label %43, !dbg !18

43:                                               ; preds = %39
  %44 = shl nsw i64 %40, 7, !dbg !6
  %45 = lshr i64 %40, 5, !dbg !6
  %46 = shl i64 %45, 12, !dbg !6
  %47 = sub i64 %44, %46, !dbg !6
  br label %48, !dbg !15

48:                                               ; preds = %158, %43
  %49 = phi i64 [ %285, %158 ], [ 0, %43 ], !dbg !15
  %50 = phi [2 x <32 x float>] [ %284, %158 ], [ zeroinitializer, %43 ], !dbg !15
  %51 = icmp samesign ult i64 %49, 4, !dbg !15
  br i1 %51, label %52, label %286, !dbg !15

52:                                               ; preds = %48
  %53 = shl nuw nsw i64 %49, 5, !dbg !15
  %.idx380 = shl nuw nsw i64 %15, 20, !dbg !13
  %54 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx380, !dbg !13
  %.idx381 = shl nsw i64 %18, 8, !dbg !13
  %55 = getelementptr i8, ptr addrspace(1) %54, i64 %.idx381, !dbg !13
  %56 = getelementptr [2 x i8], ptr addrspace(1) %55, i64 %53, !dbg !13
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !13, !range !35, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !14
  %address.local417 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local418 = and i64 %address.local417, 384, !dbg !13
  %address.local419 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !13
  %address.local420 = ptrtoint ptr addrspace(3) %address.local419 to i64, !dbg !13
  %address.local421 = trunc i64 %address.local420 to i32, !dbg !13
  %57 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local421), !dbg !13
  %58 = zext i32 %57 to i64, !dbg !13
  %59 = inttoptr i64 %58 to ptr addrspace(3), !dbg !13
  %address.local422 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local423 = and i32 %address.local422, 2, !dbg !13
  %address.local424 = shl nuw nsw i32 %address.thread, 6, !dbg !13
  %address.local425 = and i32 %address.local424, 16256, !dbg !13
  %address.local426 = or disjoint i32 %address.local423, %address.local425, !dbg !13
  %address.local427 = zext nneg i32 %address.local426 to i64, !dbg !13
  %address.local428 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local427, !dbg !13
  %dma.source.bits = ptrtoint ptr addrspace(1) %address.local428 to i64, !dbg !13
  %dma.source.opaque = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #9, !dbg !13
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1), !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %59, i32 4, i32 0, i32 0), !dbg !13
  %address.local429 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local430 = and i64 %address.local429, 384, !dbg !13
  %address.local431 = or disjoint i64 %address.local430, 512, !dbg !13
  %address.local432 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local431, !dbg !13
  %address.local433 = ptrtoint ptr addrspace(3) %address.local432 to i64, !dbg !13
  %address.local434 = trunc i64 %address.local433 to i32, !dbg !13
  %60 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local434), !dbg !13
  %61 = zext i32 %60 to i64, !dbg !13
  %62 = inttoptr i64 %61 to ptr addrspace(3), !dbg !13
  %address.local435 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local436 = and i32 %address.local435, 2, !dbg !13
  %address.local437 = shl nuw nsw i32 %address.thread, 6, !dbg !13
  %address.local438 = and i32 %address.local437, 16256, !dbg !13
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !13
  %address.local440 = zext nneg i32 %address.local439 to i64, !dbg !13
  %address.local441 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local440, !dbg !13
  %address.local442 = getelementptr i8, ptr addrspace(1) %address.local441, i64 8, !dbg !13
  %dma.source.bits1206 = ptrtoint ptr addrspace(1) %address.local442 to i64, !dbg !13
  %dma.source.opaque1207 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1206) #9, !dbg !13
  %dma.source.ptr1208 = inttoptr i64 %dma.source.opaque1207 to ptr addrspace(1), !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1208, ptr addrspace(3) %62, i32 4, i32 0, i32 0), !dbg !13
  %address.local443 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local444 = and i64 %address.local443, 384, !dbg !13
  %address.local445 = or disjoint i64 %address.local444, 1024, !dbg !13
  %address.local446 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local445, !dbg !13
  %address.local447 = ptrtoint ptr addrspace(3) %address.local446 to i64, !dbg !13
  %address.local448 = trunc i64 %address.local447 to i32, !dbg !13
  %63 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local448), !dbg !13
  %64 = zext i32 %63 to i64, !dbg !13
  %65 = inttoptr i64 %64 to ptr addrspace(3), !dbg !13
  %address.local449 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local450 = and i32 %address.local449, 2, !dbg !13
  %address.local451 = shl nuw nsw i32 %address.thread, 6, !dbg !13
  %address.local452 = and i32 %address.local451, 16256, !dbg !13
  %address.local453 = or disjoint i32 %address.local450, %address.local452, !dbg !13
  %address.local454 = zext nneg i32 %address.local453 to i64, !dbg !13
  %address.local455 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local454, !dbg !13
  %address.local456 = getelementptr i8, ptr addrspace(1) %address.local455, i64 16, !dbg !13
  %dma.source.bits1209 = ptrtoint ptr addrspace(1) %address.local456 to i64, !dbg !13
  %dma.source.opaque1210 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1209) #9, !dbg !13
  %dma.source.ptr1211 = inttoptr i64 %dma.source.opaque1210 to ptr addrspace(1), !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1211, ptr addrspace(3) %65, i32 4, i32 0, i32 0), !dbg !13
  %address.local457 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local458 = and i64 %address.local457, 384, !dbg !13
  %address.local459 = or disjoint i64 %address.local458, 1536, !dbg !13
  %address.local460 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local459, !dbg !13
  %address.local461 = ptrtoint ptr addrspace(3) %address.local460 to i64, !dbg !13
  %address.local462 = trunc i64 %address.local461 to i32, !dbg !13
  %66 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local462), !dbg !13
  %67 = zext i32 %66 to i64, !dbg !13
  %68 = inttoptr i64 %67 to ptr addrspace(3), !dbg !13
  %address.local463 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local464 = and i32 %address.local463, 2, !dbg !13
  %address.local465 = shl nuw nsw i32 %address.thread, 6, !dbg !13
  %address.local466 = and i32 %address.local465, 16256, !dbg !13
  %address.local467 = or disjoint i32 %address.local464, %address.local466, !dbg !13
  %address.local468 = zext nneg i32 %address.local467 to i64, !dbg !13
  %address.local469 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local468, !dbg !13
  %address.local470 = getelementptr i8, ptr addrspace(1) %address.local469, i64 24, !dbg !13
  %dma.source.bits1212 = ptrtoint ptr addrspace(1) %address.local470 to i64, !dbg !13
  %dma.source.opaque1213 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1212) #9, !dbg !13
  %dma.source.ptr1214 = inttoptr i64 %dma.source.opaque1213 to ptr addrspace(1), !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1214, ptr addrspace(3) %68, i32 4, i32 0, i32 0), !dbg !13
  %address.local471 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local472 = and i64 %address.local471, 384, !dbg !13
  %address.local473 = or disjoint i64 %address.local472, 2048, !dbg !13
  %address.local474 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local473, !dbg !13
  %address.local475 = ptrtoint ptr addrspace(3) %address.local474 to i64, !dbg !13
  %address.local476 = trunc i64 %address.local475 to i32, !dbg !13
  %69 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local476), !dbg !13
  %70 = zext i32 %69 to i64, !dbg !13
  %71 = inttoptr i64 %70 to ptr addrspace(3), !dbg !13
  %address.local477 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local478 = and i32 %address.local477, 2, !dbg !13
  %address.local479 = shl nuw nsw i32 %address.thread, 6, !dbg !13
  %address.local480 = and i32 %address.local479, 16256, !dbg !13
  %address.local481 = or disjoint i32 %address.local478, %address.local480, !dbg !13
  %address.local482 = zext nneg i32 %address.local481 to i64, !dbg !13
  %address.local483 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local482, !dbg !13
  %address.local484 = getelementptr i8, ptr addrspace(1) %address.local483, i64 32, !dbg !13
  %dma.source.bits1215 = ptrtoint ptr addrspace(1) %address.local484 to i64, !dbg !13
  %dma.source.opaque1216 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1215) #9, !dbg !13
  %dma.source.ptr1217 = inttoptr i64 %dma.source.opaque1216 to ptr addrspace(1), !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1217, ptr addrspace(3) %71, i32 4, i32 0, i32 0), !dbg !13
  %address.local485 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local486 = and i64 %address.local485, 384, !dbg !13
  %address.local487 = or disjoint i64 %address.local486, 2560, !dbg !13
  %address.local488 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local487, !dbg !13
  %address.local489 = ptrtoint ptr addrspace(3) %address.local488 to i64, !dbg !13
  %address.local490 = trunc i64 %address.local489 to i32, !dbg !13
  %72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local490), !dbg !13
  %73 = zext i32 %72 to i64, !dbg !13
  %74 = inttoptr i64 %73 to ptr addrspace(3), !dbg !13
  %address.local491 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local492 = and i32 %address.local491, 2, !dbg !13
  %address.local493 = shl nuw nsw i32 %address.thread, 6, !dbg !13
  %address.local494 = and i32 %address.local493, 16256, !dbg !13
  %address.local495 = or disjoint i32 %address.local492, %address.local494, !dbg !13
  %address.local496 = zext nneg i32 %address.local495 to i64, !dbg !13
  %address.local497 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local496, !dbg !13
  %address.local498 = getelementptr i8, ptr addrspace(1) %address.local497, i64 40, !dbg !13
  %dma.source.bits1218 = ptrtoint ptr addrspace(1) %address.local498 to i64, !dbg !13
  %dma.source.opaque1219 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1218) #9, !dbg !13
  %dma.source.ptr1220 = inttoptr i64 %dma.source.opaque1219 to ptr addrspace(1), !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1220, ptr addrspace(3) %74, i32 4, i32 0, i32 0), !dbg !13
  %address.local499 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local500 = and i64 %address.local499, 384, !dbg !13
  %address.local501 = or disjoint i64 %address.local500, 3072, !dbg !13
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !13
  %address.local503 = ptrtoint ptr addrspace(3) %address.local502 to i64, !dbg !13
  %address.local504 = trunc i64 %address.local503 to i32, !dbg !13
  %75 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !13
  %76 = zext i32 %75 to i64, !dbg !13
  %77 = inttoptr i64 %76 to ptr addrspace(3), !dbg !13
  %address.local505 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local506 = and i32 %address.local505, 2, !dbg !13
  %address.local507 = shl nuw nsw i32 %address.thread, 6, !dbg !13
  %address.local508 = and i32 %address.local507, 16256, !dbg !13
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !13
  %address.local510 = zext nneg i32 %address.local509 to i64, !dbg !13
  %address.local511 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local510, !dbg !13
  %address.local512 = getelementptr i8, ptr addrspace(1) %address.local511, i64 48, !dbg !13
  %dma.source.bits1221 = ptrtoint ptr addrspace(1) %address.local512 to i64, !dbg !13
  %dma.source.opaque1222 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1221) #9, !dbg !13
  %dma.source.ptr1223 = inttoptr i64 %dma.source.opaque1222 to ptr addrspace(1), !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1223, ptr addrspace(3) %77, i32 4, i32 0, i32 0), !dbg !13
  %address.local513 = shl nuw nsw i64 %address.local, 1, !dbg !13
  %address.local514 = and i64 %address.local513, 384, !dbg !13
  %address.local515 = or disjoint i64 %address.local514, 3584, !dbg !13
  %address.local516 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local515, !dbg !13
  %address.local517 = ptrtoint ptr addrspace(3) %address.local516 to i64, !dbg !13
  %address.local518 = trunc i64 %address.local517 to i32, !dbg !13
  %78 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local518), !dbg !13
  %79 = zext i32 %78 to i64, !dbg !13
  %80 = inttoptr i64 %79 to ptr addrspace(3), !dbg !13
  %address.local519 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local520 = and i32 %address.local519, 2, !dbg !13
  %address.local521 = shl nuw nsw i32 %address.thread, 6, !dbg !13
  %address.local522 = and i32 %address.local521, 16256, !dbg !13
  %address.local523 = or disjoint i32 %address.local520, %address.local522, !dbg !13
  %address.local524 = zext nneg i32 %address.local523 to i64, !dbg !13
  %address.local525 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local524, !dbg !13
  %address.local526 = getelementptr i8, ptr addrspace(1) %address.local525, i64 56, !dbg !13
  %dma.source.bits1224 = ptrtoint ptr addrspace(1) %address.local526 to i64, !dbg !13
  %dma.source.opaque1225 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1224) #9, !dbg !13
  %dma.source.ptr1226 = inttoptr i64 %dma.source.opaque1225 to ptr addrspace(1), !dbg !13
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1226, ptr addrspace(3) %80, i32 4, i32 0, i32 0), !dbg !13
  call void @llvm.amdgcn.asyncmark(), !dbg !13
  %81 = shl nuw nsw i64 %49, 10, !dbg !6
  %82 = add nuw i64 %40, %81, !dbg !6
  %.lobit382 = ashr i64 %82, 63, !dbg !6
  %83 = xor i64 %82, %.lobit382, !dbg !6
  %84 = sdiv i64 %83, 4096, !dbg !6
  %.lobit383 = ashr i64 %82, 63, !dbg !6
  %85 = xor i64 %84, %.lobit383, !dbg !6
  %86 = add nsw i64 %85, %8, !dbg !6
  %87 = add nuw nsw i64 %53, %45, !dbg !6
  %88 = and i64 %87, 127, !dbg !6
  %89 = shl i64 %86, 19, !dbg !6
  %90 = and i64 %89, 16252928, !dbg !6
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %91 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %90, !dbg !6
  %92 = getelementptr [2 x i8], ptr addrspace(1) %91, i64 %88, !dbg !6
  %.idx384 = shl i64 %47, 8, !dbg !6
  %93 = getelementptr i8, ptr addrspace(1) %92, i64 %.idx384, !dbg !6
  %address.local527 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local418, !dbg !6
  %address.local528 = ptrtoint ptr addrspace(3) %address.local527 to i64, !dbg !6
  %address.local529 = trunc i64 %address.local528 to i32, !dbg !6
  %94 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local529), !dbg !6
  %95 = zext i32 %94 to i64, !dbg !6
  %96 = inttoptr i64 %95 to ptr addrspace(3), !dbg !6
  %address.local530 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local531 = and i32 %address.local530, 30, !dbg !6
  %address.local532 = shl nuw nsw i32 %address.thread, 3, !dbg !6
  %address.local533 = and i32 %address.local532, 1920, !dbg !6
  %address.local534 = or disjoint i32 %address.local531, %address.local533, !dbg !6
  %address.local535 = zext nneg i32 %address.local534 to i64, !dbg !6
  %address.local536 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local535, !dbg !6
  %dma.source.bits1227 = ptrtoint ptr addrspace(1) %address.local536 to i64, !dbg !6
  %dma.source.opaque1228 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1227) #9, !dbg !6
  %dma.source.ptr1229 = inttoptr i64 %dma.source.opaque1228 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1229, ptr addrspace(3) %96, i32 4, i32 0, i32 0), !dbg !6
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local431, !dbg !6
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !6
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !6
  %97 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !6
  %98 = zext i32 %97 to i64, !dbg !6
  %99 = inttoptr i64 %98 to ptr addrspace(3), !dbg !6
  %address.local540 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local541 = and i32 %address.local540, 30, !dbg !6
  %address.local542 = shl nuw nsw i32 %address.thread, 3, !dbg !6
  %address.local543 = and i32 %address.local542, 1920, !dbg !6
  %address.local544 = or disjoint i32 %address.local541, %address.local543, !dbg !6
  %address.local545 = zext nneg i32 %address.local544 to i64, !dbg !6
  %address.local546 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local545, !dbg !6
  %address.local547 = getelementptr i8, ptr addrspace(1) %address.local546, i64 4096, !dbg !6
  %dma.source.bits1230 = ptrtoint ptr addrspace(1) %address.local547 to i64, !dbg !6
  %dma.source.opaque1231 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1230) #9, !dbg !6
  %dma.source.ptr1232 = inttoptr i64 %dma.source.opaque1231 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1232, ptr addrspace(3) %99, i32 4, i32 0, i32 0), !dbg !6
  %address.local548 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local445, !dbg !6
  %address.local549 = ptrtoint ptr addrspace(3) %address.local548 to i64, !dbg !6
  %address.local550 = trunc i64 %address.local549 to i32, !dbg !6
  %100 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local550), !dbg !6
  %101 = zext i32 %100 to i64, !dbg !6
  %102 = inttoptr i64 %101 to ptr addrspace(3), !dbg !6
  %address.local551 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local552 = and i32 %address.local551, 30, !dbg !6
  %address.local553 = shl nuw nsw i32 %address.thread, 3, !dbg !6
  %address.local554 = and i32 %address.local553, 1920, !dbg !6
  %address.local555 = or disjoint i32 %address.local552, %address.local554, !dbg !6
  %address.local556 = zext nneg i32 %address.local555 to i64, !dbg !6
  %address.local557 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local556, !dbg !6
  %address.local558 = getelementptr i8, ptr addrspace(1) %address.local557, i64 8192, !dbg !6
  %dma.source.bits1233 = ptrtoint ptr addrspace(1) %address.local558 to i64, !dbg !6
  %dma.source.opaque1234 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1233) #9, !dbg !6
  %dma.source.ptr1235 = inttoptr i64 %dma.source.opaque1234 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1235, ptr addrspace(3) %102, i32 4, i32 0, i32 0), !dbg !6
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local459, !dbg !6
  %address.local560 = ptrtoint ptr addrspace(3) %address.local559 to i64, !dbg !6
  %address.local561 = trunc i64 %address.local560 to i32, !dbg !6
  %103 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local561), !dbg !6
  %104 = zext i32 %103 to i64, !dbg !6
  %105 = inttoptr i64 %104 to ptr addrspace(3), !dbg !6
  %address.local562 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local563 = and i32 %address.local562, 30, !dbg !6
  %address.local564 = shl nuw nsw i32 %address.thread, 3, !dbg !6
  %address.local565 = and i32 %address.local564, 1920, !dbg !6
  %address.local566 = or disjoint i32 %address.local563, %address.local565, !dbg !6
  %address.local567 = zext nneg i32 %address.local566 to i64, !dbg !6
  %address.local568 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local567, !dbg !6
  %address.local569 = getelementptr i8, ptr addrspace(1) %address.local568, i64 12288, !dbg !6
  %dma.source.bits1236 = ptrtoint ptr addrspace(1) %address.local569 to i64, !dbg !6
  %dma.source.opaque1237 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1236) #9, !dbg !6
  %dma.source.ptr1238 = inttoptr i64 %dma.source.opaque1237 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1238, ptr addrspace(3) %105, i32 4, i32 0, i32 0), !dbg !6
  %address.local570 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local473, !dbg !6
  %address.local571 = ptrtoint ptr addrspace(3) %address.local570 to i64, !dbg !6
  %address.local572 = trunc i64 %address.local571 to i32, !dbg !6
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local572), !dbg !6
  %107 = zext i32 %106 to i64, !dbg !6
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !6
  %address.local573 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local574 = and i32 %address.local573, 30, !dbg !6
  %address.local575 = shl nuw nsw i32 %address.thread, 3, !dbg !6
  %address.local576 = and i32 %address.local575, 1920, !dbg !6
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !6
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !6
  %address.local579 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local578, !dbg !6
  %address.local580 = getelementptr i8, ptr addrspace(1) %address.local579, i64 16384, !dbg !6
  %dma.source.bits1239 = ptrtoint ptr addrspace(1) %address.local580 to i64, !dbg !6
  %dma.source.opaque1240 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1239) #9, !dbg !6
  %dma.source.ptr1241 = inttoptr i64 %dma.source.opaque1240 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1241, ptr addrspace(3) %108, i32 4, i32 0, i32 0), !dbg !6
  %address.local581 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local487, !dbg !6
  %address.local582 = ptrtoint ptr addrspace(3) %address.local581 to i64, !dbg !6
  %address.local583 = trunc i64 %address.local582 to i32, !dbg !6
  %109 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local583), !dbg !6
  %110 = zext i32 %109 to i64, !dbg !6
  %111 = inttoptr i64 %110 to ptr addrspace(3), !dbg !6
  %address.local584 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local585 = and i32 %address.local584, 30, !dbg !6
  %address.local586 = shl nuw nsw i32 %address.thread, 3, !dbg !6
  %address.local587 = and i32 %address.local586, 1920, !dbg !6
  %address.local588 = or disjoint i32 %address.local585, %address.local587, !dbg !6
  %address.local589 = zext nneg i32 %address.local588 to i64, !dbg !6
  %address.local590 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local589, !dbg !6
  %address.local591 = getelementptr i8, ptr addrspace(1) %address.local590, i64 20480, !dbg !6
  %dma.source.bits1242 = ptrtoint ptr addrspace(1) %address.local591 to i64, !dbg !6
  %dma.source.opaque1243 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1242) #9, !dbg !6
  %dma.source.ptr1244 = inttoptr i64 %dma.source.opaque1243 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1244, ptr addrspace(3) %111, i32 4, i32 0, i32 0), !dbg !6
  %address.local592 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local501, !dbg !6
  %address.local593 = ptrtoint ptr addrspace(3) %address.local592 to i64, !dbg !6
  %address.local594 = trunc i64 %address.local593 to i32, !dbg !6
  %112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local594), !dbg !6
  %113 = zext i32 %112 to i64, !dbg !6
  %114 = inttoptr i64 %113 to ptr addrspace(3), !dbg !6
  %address.local595 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local596 = and i32 %address.local595, 30, !dbg !6
  %address.local597 = shl nuw nsw i32 %address.thread, 3, !dbg !6
  %address.local598 = and i32 %address.local597, 1920, !dbg !6
  %address.local599 = or disjoint i32 %address.local596, %address.local598, !dbg !6
  %address.local600 = zext nneg i32 %address.local599 to i64, !dbg !6
  %address.local601 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local600, !dbg !6
  %address.local602 = getelementptr i8, ptr addrspace(1) %address.local601, i64 24576, !dbg !6
  %dma.source.bits1245 = ptrtoint ptr addrspace(1) %address.local602 to i64, !dbg !6
  %dma.source.opaque1246 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1245) #9, !dbg !6
  %dma.source.ptr1247 = inttoptr i64 %dma.source.opaque1246 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1247, ptr addrspace(3) %114, i32 4, i32 0, i32 0), !dbg !6
  %address.local603 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local515, !dbg !6
  %address.local604 = ptrtoint ptr addrspace(3) %address.local603 to i64, !dbg !6
  %address.local605 = trunc i64 %address.local604 to i32, !dbg !6
  %115 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local605), !dbg !6
  %116 = zext i32 %115 to i64, !dbg !6
  %117 = inttoptr i64 %116 to ptr addrspace(3), !dbg !6
  %address.local606 = shl nuw nsw i32 %address.thread, 1, !dbg !6
  %address.local607 = and i32 %address.local606, 30, !dbg !6
  %address.local608 = shl nuw nsw i32 %address.thread, 3, !dbg !6
  %address.local609 = and i32 %address.local608, 1920, !dbg !6
  %address.local610 = or disjoint i32 %address.local607, %address.local609, !dbg !6
  %address.local611 = zext nneg i32 %address.local610 to i64, !dbg !6
  %address.local612 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local611, !dbg !6
  %address.local613 = getelementptr i8, ptr addrspace(1) %address.local612, i64 28672, !dbg !6
  %dma.source.bits1248 = ptrtoint ptr addrspace(1) %address.local613 to i64, !dbg !6
  %dma.source.opaque1249 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1248) #9, !dbg !6
  %dma.source.ptr1250 = inttoptr i64 %dma.source.opaque1249 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1250, ptr addrspace(3) %117, i32 4, i32 0, i32 0), !dbg !6
  call void @llvm.amdgcn.asyncmark(), !dbg !6
  %address.local614 = and i32 %address.thread, 192, !dbg !14
  %address.local615 = shl nuw nsw i32 %address.thread, 2, !dbg !14
  %address.local616 = and i32 %address.local615, 60, !dbg !14
  %address.local617 = or disjoint i32 %address.local614, %address.local616, !dbg !14
  %address.local618 = shl nuw nsw i32 %address.thread, 5, !dbg !14
  %address.local619 = and i32 %address.local618, 1536, !dbg !14
  %address.local620 = or disjoint i32 %address.local617, %address.local619, !dbg !14
  %address.local621 = zext nneg i32 %address.local620 to i64, !dbg !14
  %address.local622 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local621, !dbg !14
  %118 = load <4 x half>, ptr addrspace(3) %address.local622, align 2, !dbg !14
  %address.local623 = and i32 %address.thread, 192, !dbg !14
  %address.local624 = shl nuw nsw i32 %address.thread, 2, !dbg !14
  %address.local625 = and i32 %address.local624, 60, !dbg !14
  %address.local626 = or disjoint i32 %address.local623, %address.local625, !dbg !14
  %address.local627 = shl nuw nsw i32 %address.thread, 5, !dbg !14
  %address.local628 = and i32 %address.local627, 1536, !dbg !14
  %address.local629 = or disjoint i32 %address.local626, %address.local628, !dbg !14
  %address.local630 = zext nneg i32 %address.local629 to i64, !dbg !14
  %address.local631 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local630, !dbg !14
  %address.local632 = getelementptr i8, ptr addrspace(3) %address.local631, i64 4096, !dbg !14
  %119 = load <4 x half>, ptr addrspace(3) %address.local632, align 2, !dbg !14
  %address.local633 = and i32 %address.thread, 192, !dbg !14
  %address.local634 = shl nuw nsw i32 %address.thread, 2, !dbg !14
  %address.local635 = and i32 %address.local634, 60, !dbg !14
  %address.local636 = or disjoint i32 %address.local633, %address.local635, !dbg !14
  %address.local637 = shl nuw nsw i32 %address.thread, 5, !dbg !14
  %address.local638 = and i32 %address.local637, 1536, !dbg !14
  %address.local639 = or disjoint i32 %address.local636, %address.local638, !dbg !14
  %address.local640 = zext nneg i32 %address.local639 to i64, !dbg !14
  %address.local641 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local640, !dbg !14
  %address.local642 = getelementptr i8, ptr addrspace(3) %address.local641, i64 512, !dbg !14
  %120 = load <4 x half>, ptr addrspace(3) %address.local642, align 2, !dbg !14
  %address.local643 = and i32 %address.thread, 192, !dbg !14
  %address.local644 = shl nuw nsw i32 %address.thread, 2, !dbg !14
  %address.local645 = and i32 %address.local644, 60, !dbg !14
  %address.local646 = or disjoint i32 %address.local643, %address.local645, !dbg !14
  %address.local647 = shl nuw nsw i32 %address.thread, 5, !dbg !14
  %address.local648 = and i32 %address.local647, 1536, !dbg !14
  %address.local649 = or disjoint i32 %address.local646, %address.local648, !dbg !14
  %address.local650 = zext nneg i32 %address.local649 to i64, !dbg !14
  %address.local651 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local650, !dbg !14
  %address.local652 = getelementptr i8, ptr addrspace(3) %address.local651, i64 4608, !dbg !14
  %121 = load <4 x half>, ptr addrspace(3) %address.local652, align 2, !dbg !14
  %122 = fmul <4 x half> %118, splat (half 0xH3015), !dbg !14
  %123 = fmul <4 x half> %119, splat (half 0xH3015), !dbg !14
  %124 = fmul <4 x half> %120, splat (half 0xH3015), !dbg !14
  %125 = fmul <4 x half> %121, splat (half 0xH3015), !dbg !14
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !6
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  br label %126, !dbg !6

126:                                              ; preds = %129, %52
  %127 = phi i64 [ %157, %129 ], [ %20, %52 ], !dbg !6
  %128 = icmp samesign ult i64 %127, 2048, !dbg !6
  br i1 %128, label %129, label %158, !dbg !6

129:                                              ; preds = %126
  %130 = shl nuw nsw i64 %127, 1, !dbg !6
  %131 = lshr i64 %127, 6, !dbg !6
  %132 = and i64 %131, 30, !dbg !6
  %133 = lshr i64 %127, 2, !dbg !6
  %134 = and i64 %133, 1, !dbg !6
  %135 = and i64 %127, 120, !dbg !6
  %136 = and i64 %130, 6, !dbg !6
  %137 = or disjoint i64 %135, %136, !dbg !6
  %138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %132, !dbg !6
  %139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %138, i64 %134, !dbg !6
  %.idx385 = shl nuw nsw i64 %137, 6, !dbg !6
  %140 = getelementptr inbounds nuw i8, ptr addrspace(3) %139, i64 %.idx385, !dbg !6
  %141 = load half, ptr addrspace(3) %140, align 2, !dbg !6
  %142 = lshr i64 %127, 6, !dbg !6
  %143 = and i64 %142, 30, !dbg !6
  %144 = lshr i64 %127, 2, !dbg !6
  %145 = and i64 %144, 1, !dbg !6
  %146 = and i64 %127, 120, !dbg !6
  %147 = and i64 %130, 6, !dbg !6
  %148 = or disjoint i64 %147, 1, !dbg !6
  %149 = or disjoint i64 %146, %148, !dbg !6
  %150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %143, !dbg !6
  %151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %150, i64 %145, !dbg !6
  %.idx386 = shl nuw nsw i64 %149, 6, !dbg !6
  %152 = getelementptr inbounds nuw i8, ptr addrspace(3) %151, i64 %.idx386, !dbg !6
  %153 = load half, ptr addrspace(3) %152, align 2, !dbg !6
  %154 = insertelement <2 x half> poison, half %141, i64 0, !dbg !6
  %155 = insertelement <2 x half> %154, half %153, i64 1, !dbg !6
  %156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %130, !dbg !6
  store <2 x half> %155, ptr addrspace(3) %156, align 2, !dbg !6
  %157 = add nuw nsw i64 %127, 256, !dbg !6
  br label %126, !dbg !6

158:                                              ; preds = %126
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %address.thread653 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !7, !range !35, !frisk.register_lifetimes !1
  %address.local654 = shl nuw nsw i32 %address.thread653, 1, !dbg !7
  %address.local655 = and i32 %address.local654, 8, !dbg !7
  %address.local656 = shl nuw nsw i32 %address.thread653, 4, !dbg !7
  %address.local657 = and i32 %address.local656, 48, !dbg !7
  %address.local658 = or disjoint i32 %address.local655, %address.local657, !dbg !7
  %address.local659 = shl nuw nsw i32 %address.thread653, 5, !dbg !7
  %address.local660 = and i32 %address.local659, 1792, !dbg !7
  %address.local661 = or disjoint i32 %address.local658, %address.local660, !dbg !7
  %address.local662 = zext nneg i32 %address.local661 to i64, !dbg !7
  %address.local663 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local662, !dbg !7
  %159 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local663, i32 0) #0, !dbg !7
  %160 = bitcast <4 x i32> %159 to <8 x half>, !dbg !7
  %address.local664 = shl nuw nsw i32 %address.thread653, 1, !dbg !7
  %address.local665 = and i32 %address.local664, 8, !dbg !7
  %address.local666 = shl nuw nsw i32 %address.thread653, 4, !dbg !7
  %address.local667 = and i32 %address.local666, 48, !dbg !7
  %address.local668 = or disjoint i32 %address.local665, %address.local667, !dbg !7
  %address.local669 = shl nuw nsw i32 %address.thread653, 5, !dbg !7
  %address.local670 = and i32 %address.local669, 1792, !dbg !7
  %address.local671 = or disjoint i32 %address.local668, %address.local670, !dbg !7
  %address.local672 = zext nneg i32 %address.local671 to i64, !dbg !7
  %address.local673 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local672, !dbg !7
  %address.local674 = getelementptr i8, ptr addrspace(3) %address.local673, i64 128, !dbg !7
  %161 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local674, i32 0) #0, !dbg !7
  %162 = bitcast <4 x i32> %161 to <8 x half>, !dbg !7
  %address.local675 = shl nuw nsw i32 %address.thread653, 1, !dbg !7
  %address.local676 = and i32 %address.local675, 8, !dbg !7
  %address.local677 = shl nuw nsw i32 %address.thread653, 4, !dbg !7
  %address.local678 = and i32 %address.local677, 48, !dbg !7
  %address.local679 = or disjoint i32 %address.local676, %address.local678, !dbg !7
  %address.local680 = shl nuw nsw i32 %address.thread653, 5, !dbg !7
  %address.local681 = and i32 %address.local680, 1792, !dbg !7
  %address.local682 = or disjoint i32 %address.local679, %address.local681, !dbg !7
  %address.local683 = zext nneg i32 %address.local682 to i64, !dbg !7
  %address.local684 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local683, !dbg !7
  %address.local685 = getelementptr i8, ptr addrspace(3) %address.local684, i64 256, !dbg !7
  %163 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local685, i32 0) #0, !dbg !7
  %164 = bitcast <4 x i32> %163 to <8 x half>, !dbg !7
  %address.local686 = shl nuw nsw i32 %address.thread653, 1, !dbg !7
  %address.local687 = and i32 %address.local686, 8, !dbg !7
  %address.local688 = shl nuw nsw i32 %address.thread653, 4, !dbg !7
  %address.local689 = and i32 %address.local688, 48, !dbg !7
  %address.local690 = or disjoint i32 %address.local687, %address.local689, !dbg !7
  %address.local691 = shl nuw nsw i32 %address.thread653, 5, !dbg !7
  %address.local692 = and i32 %address.local691, 1792, !dbg !7
  %address.local693 = or disjoint i32 %address.local690, %address.local692, !dbg !7
  %address.local694 = zext nneg i32 %address.local693 to i64, !dbg !7
  %address.local695 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local694, !dbg !7
  %address.local696 = getelementptr i8, ptr addrspace(3) %address.local695, i64 384, !dbg !7
  %165 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local696, i32 0) #0, !dbg !7
  %166 = bitcast <4 x i32> %165 to <8 x half>, !dbg !7
  %address.local697 = shl nuw nsw i32 %address.thread653, 1, !dbg !7
  %address.local698 = and i32 %address.local697, 8, !dbg !7
  %address.local699 = shl nuw nsw i32 %address.thread653, 4, !dbg !7
  %address.local700 = and i32 %address.local699, 48, !dbg !7
  %address.local701 = or disjoint i32 %address.local698, %address.local700, !dbg !7
  %address.local702 = shl nuw nsw i32 %address.thread653, 5, !dbg !7
  %address.local703 = and i32 %address.local702, 1792, !dbg !7
  %address.local704 = or disjoint i32 %address.local701, %address.local703, !dbg !7
  %address.local705 = zext nneg i32 %address.local704 to i64, !dbg !7
  %address.local706 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local705, !dbg !7
  %address.local707 = getelementptr i8, ptr addrspace(3) %address.local706, i64 4096, !dbg !7
  %167 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local707, i32 0) #0, !dbg !7
  %168 = bitcast <4 x i32> %167 to <8 x half>, !dbg !7
  %address.local708 = shl nuw nsw i32 %address.thread653, 1, !dbg !7
  %address.local709 = and i32 %address.local708, 8, !dbg !7
  %address.local710 = shl nuw nsw i32 %address.thread653, 4, !dbg !7
  %address.local711 = and i32 %address.local710, 48, !dbg !7
  %address.local712 = or disjoint i32 %address.local709, %address.local711, !dbg !7
  %address.local713 = shl nuw nsw i32 %address.thread653, 5, !dbg !7
  %address.local714 = and i32 %address.local713, 1792, !dbg !7
  %address.local715 = or disjoint i32 %address.local712, %address.local714, !dbg !7
  %address.local716 = zext nneg i32 %address.local715 to i64, !dbg !7
  %address.local717 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local716, !dbg !7
  %address.local718 = getelementptr i8, ptr addrspace(3) %address.local717, i64 4224, !dbg !7
  %169 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local718, i32 0) #0, !dbg !7
  %170 = bitcast <4 x i32> %169 to <8 x half>, !dbg !7
  %address.local719 = shl nuw nsw i32 %address.thread653, 1, !dbg !7
  %address.local720 = and i32 %address.local719, 8, !dbg !7
  %address.local721 = shl nuw nsw i32 %address.thread653, 4, !dbg !7
  %address.local722 = and i32 %address.local721, 48, !dbg !7
  %address.local723 = or disjoint i32 %address.local720, %address.local722, !dbg !7
  %address.local724 = shl nuw nsw i32 %address.thread653, 5, !dbg !7
  %address.local725 = and i32 %address.local724, 1792, !dbg !7
  %address.local726 = or disjoint i32 %address.local723, %address.local725, !dbg !7
  %address.local727 = zext nneg i32 %address.local726 to i64, !dbg !7
  %address.local728 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local727, !dbg !7
  %address.local729 = getelementptr i8, ptr addrspace(3) %address.local728, i64 4352, !dbg !7
  %171 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local729, i32 0) #0, !dbg !7
  %172 = bitcast <4 x i32> %171 to <8 x half>, !dbg !7
  %address.local730 = shl nuw nsw i32 %address.thread653, 1, !dbg !7
  %address.local731 = and i32 %address.local730, 8, !dbg !7
  %address.local732 = shl nuw nsw i32 %address.thread653, 4, !dbg !7
  %address.local733 = and i32 %address.local732, 48, !dbg !7
  %address.local734 = or disjoint i32 %address.local731, %address.local733, !dbg !7
  %address.local735 = shl nuw nsw i32 %address.thread653, 5, !dbg !7
  %address.local736 = and i32 %address.local735, 1792, !dbg !7
  %address.local737 = or disjoint i32 %address.local734, %address.local736, !dbg !7
  %address.local738 = zext nneg i32 %address.local737 to i64, !dbg !7
  %address.local739 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local738, !dbg !7
  %address.local740 = getelementptr i8, ptr addrspace(3) %address.local739, i64 4480, !dbg !7
  %173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local740, i32 0) #0, !dbg !7
  %174 = bitcast <4 x i32> %173 to <8 x half>, !dbg !7
  %175 = shufflevector <8 x half> %160, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !7
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %175, <4 x float> zeroinitializer) #8, !dbg !7
  %177 = shufflevector <8 x half> %168, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !7
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %177, <4 x float> %176) #8, !dbg !7
  %179 = extractvalue [2 x <32 x float>] %50, 0, !dbg !32
  %180 = shufflevector <32 x float> %179, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %181 = fadd <4 x float> %180, %178, !dbg !32
  %182 = shufflevector <8 x half> %160, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %182, <4 x float> zeroinitializer) #8, !dbg !7
  %184 = shufflevector <8 x half> %168, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %184, <4 x float> %183) #8, !dbg !7
  %186 = shufflevector <32 x float> %179, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %187 = fadd <4 x float> %186, %185, !dbg !32
  %188 = shufflevector <8 x half> %162, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !7
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %188, <4 x float> zeroinitializer) #8, !dbg !7
  %190 = shufflevector <8 x half> %170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !7
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %190, <4 x float> %189) #8, !dbg !7
  %192 = shufflevector <32 x float> %179, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !32
  %193 = fadd <4 x float> %192, %191, !dbg !32
  %194 = shufflevector <4 x float> %193, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %195 = shufflevector <8 x half> %162, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %195, <4 x float> zeroinitializer) #8, !dbg !7
  %197 = shufflevector <8 x half> %170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %197, <4 x float> %196) #8, !dbg !7
  %199 = shufflevector <32 x float> %179, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !32
  %200 = fadd <4 x float> %199, %198, !dbg !32
  %201 = shufflevector <4 x float> %200, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %202 = shufflevector <8 x half> %164, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !7
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %202, <4 x float> zeroinitializer) #8, !dbg !7
  %204 = shufflevector <8 x half> %172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !7
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %204, <4 x float> %203) #8, !dbg !7
  %206 = shufflevector <32 x float> %179, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !32
  %207 = fadd <4 x float> %206, %205, !dbg !32
  %208 = shufflevector <4 x float> %207, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %209 = shufflevector <8 x half> %164, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %209, <4 x float> zeroinitializer) #8, !dbg !7
  %211 = shufflevector <8 x half> %172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %211, <4 x float> %210) #8, !dbg !7
  %213 = shufflevector <32 x float> %179, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !32
  %214 = fadd <4 x float> %213, %212, !dbg !32
  %215 = shufflevector <4 x float> %214, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %216 = shufflevector <8 x half> %166, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !7
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %216, <4 x float> zeroinitializer) #8, !dbg !7
  %218 = shufflevector <8 x half> %174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !7
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %218, <4 x float> %217) #8, !dbg !7
  %220 = shufflevector <32 x float> %179, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !32
  %221 = fadd <4 x float> %220, %219, !dbg !32
  %222 = shufflevector <4 x float> %221, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %223 = shufflevector <8 x half> %166, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !7
  %225 = shufflevector <8 x half> %174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %225, <4 x float> %224) #8, !dbg !7
  %227 = shufflevector <32 x float> %179, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !32
  %228 = fadd <4 x float> %227, %226, !dbg !32
  %229 = shufflevector <4 x float> %228, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %175, <4 x float> zeroinitializer) #8, !dbg !7
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %177, <4 x float> %230) #8, !dbg !7
  %232 = extractvalue [2 x <32 x float>] %50, 1, !dbg !32
  %233 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %234 = fadd <4 x float> %233, %231, !dbg !32
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %182, <4 x float> zeroinitializer) #8, !dbg !7
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %184, <4 x float> %235) #8, !dbg !7
  %237 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %238 = fadd <4 x float> %237, %236, !dbg !32
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %188, <4 x float> zeroinitializer) #8, !dbg !7
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %190, <4 x float> %239) #8, !dbg !7
  %241 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !32
  %242 = fadd <4 x float> %241, %240, !dbg !32
  %243 = shufflevector <4 x float> %242, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %195, <4 x float> zeroinitializer) #8, !dbg !7
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %197, <4 x float> %244) #8, !dbg !7
  %246 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !32
  %247 = fadd <4 x float> %246, %245, !dbg !32
  %248 = shufflevector <4 x float> %247, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %202, <4 x float> zeroinitializer) #8, !dbg !7
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %204, <4 x float> %249) #8, !dbg !7
  %251 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !32
  %252 = fadd <4 x float> %251, %250, !dbg !32
  %253 = shufflevector <4 x float> %252, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %209, <4 x float> zeroinitializer) #8, !dbg !7
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %211, <4 x float> %254) #8, !dbg !7
  %256 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !32
  %257 = fadd <4 x float> %256, %255, !dbg !32
  %258 = shufflevector <4 x float> %257, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %216, <4 x float> zeroinitializer) #8, !dbg !7
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %218, <4 x float> %259) #8, !dbg !7
  %261 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !32
  %262 = fadd <4 x float> %261, %260, !dbg !32
  %263 = shufflevector <4 x float> %262, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !7
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %225, <4 x float> %264) #8, !dbg !7
  %266 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !32
  %267 = fadd <4 x float> %266, %265, !dbg !32
  %268 = shufflevector <4 x float> %267, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %269 = shufflevector <4 x float> %181, <4 x float> %187, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %270 = shufflevector <32 x float> %269, <32 x float> %194, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %271 = shufflevector <32 x float> %270, <32 x float> %201, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %272 = shufflevector <32 x float> %271, <32 x float> %208, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %273 = shufflevector <32 x float> %272, <32 x float> %215, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %274 = shufflevector <32 x float> %273, <32 x float> %222, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %275 = shufflevector <32 x float> %274, <32 x float> %229, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !15
  %276 = insertvalue [2 x <32 x float>] poison, <32 x float> %275, 0, !dbg !15
  %277 = shufflevector <4 x float> %234, <4 x float> %238, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %278 = shufflevector <32 x float> %277, <32 x float> %243, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %279 = shufflevector <32 x float> %278, <32 x float> %248, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %280 = shufflevector <32 x float> %279, <32 x float> %253, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %281 = shufflevector <32 x float> %280, <32 x float> %258, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %282 = shufflevector <32 x float> %281, <32 x float> %263, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %283 = shufflevector <32 x float> %282, <32 x float> %268, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !15
  %284 = insertvalue [2 x <32 x float>] %276, <32 x float> %283, 1, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %285 = add nuw nsw i64 %49, 1, !dbg !15
  br label %48, !dbg !15

286:                                              ; preds = %48
  %287 = add nsw i64 %45, %8, !dbg !19
  %288 = shl i64 %287, 19, !dbg !19
  %289 = and i64 %288, 16252928, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %290 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %289, !dbg !19
  %.idx = shl i64 %47, 8, !dbg !19
  %291 = getelementptr i8, ptr addrspace(1) %290, i64 %.idx, !dbg !19
  %292 = getelementptr [2 x i8], ptr addrspace(1) %291, i64 %38, !dbg !19
  %address.thread741 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !19, !range !35, !frisk.register_lifetimes !1
  %address.local742 = zext nneg i32 %address.thread741 to i64, !dbg !14
  %address.local743 = shl nuw nsw i64 %address.local742, 1, !dbg !13
  %address.local744 = and i64 %address.local743, 384, !dbg !13
  %address.local745 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local744, !dbg !13
  %address.local746 = ptrtoint ptr addrspace(3) %address.local745 to i64, !dbg !13
  %address.local747 = trunc i64 %address.local746 to i32, !dbg !13
  %293 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local747), !dbg !19
  %294 = zext i32 %293 to i64, !dbg !19
  %295 = inttoptr i64 %294 to ptr addrspace(3), !dbg !19
  %address.local748 = shl nuw nsw i32 %address.thread741, 1, !dbg !19
  %address.local749 = and i32 %address.local748, 30, !dbg !19
  %address.local750 = shl nuw nsw i32 %address.thread741, 3, !dbg !19
  %address.local751 = and i32 %address.local750, 1920, !dbg !19
  %address.local752 = or disjoint i32 %address.local749, %address.local751, !dbg !19
  %address.local753 = zext nneg i32 %address.local752 to i64, !dbg !19
  %address.local754 = getelementptr [2 x i8], ptr addrspace(1) %292, i64 %address.local753, !dbg !19
  %dma.source.bits1251 = ptrtoint ptr addrspace(1) %address.local754 to i64, !dbg !19
  %dma.source.opaque1252 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1251) #9, !dbg !19
  %dma.source.ptr1253 = inttoptr i64 %dma.source.opaque1252 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1253, ptr addrspace(3) %295, i32 4, i32 0, i32 0), !dbg !19
  %address.local755 = shl nuw nsw i64 %address.local742, 1, !dbg !13
  %address.local756 = and i64 %address.local755, 384, !dbg !13
  %address.local757 = or disjoint i64 %address.local756, 512, !dbg !13
  %address.local758 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local757, !dbg !13
  %address.local759 = ptrtoint ptr addrspace(3) %address.local758 to i64, !dbg !13
  %address.local760 = trunc i64 %address.local759 to i32, !dbg !13
  %296 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local760), !dbg !19
  %297 = zext i32 %296 to i64, !dbg !19
  %298 = inttoptr i64 %297 to ptr addrspace(3), !dbg !19
  %address.local761 = shl nuw nsw i32 %address.thread741, 1, !dbg !19
  %address.local762 = and i32 %address.local761, 30, !dbg !19
  %address.local763 = shl nuw nsw i32 %address.thread741, 3, !dbg !19
  %address.local764 = and i32 %address.local763, 1920, !dbg !19
  %address.local765 = or disjoint i32 %address.local762, %address.local764, !dbg !19
  %address.local766 = zext nneg i32 %address.local765 to i64, !dbg !19
  %address.local767 = getelementptr [2 x i8], ptr addrspace(1) %292, i64 %address.local766, !dbg !19
  %address.local768 = getelementptr i8, ptr addrspace(1) %address.local767, i64 4096, !dbg !19
  %dma.source.bits1254 = ptrtoint ptr addrspace(1) %address.local768 to i64, !dbg !19
  %dma.source.opaque1255 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1254) #9, !dbg !19
  %dma.source.ptr1256 = inttoptr i64 %dma.source.opaque1255 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1256, ptr addrspace(3) %298, i32 4, i32 0, i32 0), !dbg !19
  %address.local769 = shl nuw nsw i64 %address.local742, 1, !dbg !13
  %address.local770 = and i64 %address.local769, 384, !dbg !13
  %address.local771 = or disjoint i64 %address.local770, 1024, !dbg !13
  %address.local772 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local771, !dbg !13
  %address.local773 = ptrtoint ptr addrspace(3) %address.local772 to i64, !dbg !13
  %address.local774 = trunc i64 %address.local773 to i32, !dbg !13
  %299 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local774), !dbg !19
  %300 = zext i32 %299 to i64, !dbg !19
  %301 = inttoptr i64 %300 to ptr addrspace(3), !dbg !19
  %address.local775 = shl nuw nsw i32 %address.thread741, 1, !dbg !19
  %address.local776 = and i32 %address.local775, 30, !dbg !19
  %address.local777 = shl nuw nsw i32 %address.thread741, 3, !dbg !19
  %address.local778 = and i32 %address.local777, 1920, !dbg !19
  %address.local779 = or disjoint i32 %address.local776, %address.local778, !dbg !19
  %address.local780 = zext nneg i32 %address.local779 to i64, !dbg !19
  %address.local781 = getelementptr [2 x i8], ptr addrspace(1) %292, i64 %address.local780, !dbg !19
  %address.local782 = getelementptr i8, ptr addrspace(1) %address.local781, i64 8192, !dbg !19
  %dma.source.bits1257 = ptrtoint ptr addrspace(1) %address.local782 to i64, !dbg !19
  %dma.source.opaque1258 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1257) #9, !dbg !19
  %dma.source.ptr1259 = inttoptr i64 %dma.source.opaque1258 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1259, ptr addrspace(3) %301, i32 4, i32 0, i32 0), !dbg !19
  %address.local783 = shl nuw nsw i64 %address.local742, 1, !dbg !13
  %address.local784 = and i64 %address.local783, 384, !dbg !13
  %address.local785 = or disjoint i64 %address.local784, 1536, !dbg !13
  %address.local786 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local785, !dbg !13
  %address.local787 = ptrtoint ptr addrspace(3) %address.local786 to i64, !dbg !13
  %address.local788 = trunc i64 %address.local787 to i32, !dbg !13
  %302 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local788), !dbg !19
  %303 = zext i32 %302 to i64, !dbg !19
  %304 = inttoptr i64 %303 to ptr addrspace(3), !dbg !19
  %address.local789 = shl nuw nsw i32 %address.thread741, 1, !dbg !19
  %address.local790 = and i32 %address.local789, 30, !dbg !19
  %address.local791 = shl nuw nsw i32 %address.thread741, 3, !dbg !19
  %address.local792 = and i32 %address.local791, 1920, !dbg !19
  %address.local793 = or disjoint i32 %address.local790, %address.local792, !dbg !19
  %address.local794 = zext nneg i32 %address.local793 to i64, !dbg !19
  %address.local795 = getelementptr [2 x i8], ptr addrspace(1) %292, i64 %address.local794, !dbg !19
  %address.local796 = getelementptr i8, ptr addrspace(1) %address.local795, i64 12288, !dbg !19
  %dma.source.bits1260 = ptrtoint ptr addrspace(1) %address.local796 to i64, !dbg !19
  %dma.source.opaque1261 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1260) #9, !dbg !19
  %dma.source.ptr1262 = inttoptr i64 %dma.source.opaque1261 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1262, ptr addrspace(3) %304, i32 4, i32 0, i32 0), !dbg !19
  %address.local797 = shl nuw nsw i64 %address.local742, 1, !dbg !13
  %address.local798 = and i64 %address.local797, 384, !dbg !13
  %address.local799 = or disjoint i64 %address.local798, 2048, !dbg !13
  %address.local800 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local799, !dbg !13
  %address.local801 = ptrtoint ptr addrspace(3) %address.local800 to i64, !dbg !13
  %address.local802 = trunc i64 %address.local801 to i32, !dbg !13
  %305 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local802), !dbg !19
  %306 = zext i32 %305 to i64, !dbg !19
  %307 = inttoptr i64 %306 to ptr addrspace(3), !dbg !19
  %address.local803 = shl nuw nsw i32 %address.thread741, 1, !dbg !19
  %address.local804 = and i32 %address.local803, 30, !dbg !19
  %address.local805 = shl nuw nsw i32 %address.thread741, 3, !dbg !19
  %address.local806 = and i32 %address.local805, 1920, !dbg !19
  %address.local807 = or disjoint i32 %address.local804, %address.local806, !dbg !19
  %address.local808 = zext nneg i32 %address.local807 to i64, !dbg !19
  %address.local809 = getelementptr [2 x i8], ptr addrspace(1) %292, i64 %address.local808, !dbg !19
  %address.local810 = getelementptr i8, ptr addrspace(1) %address.local809, i64 16384, !dbg !19
  %dma.source.bits1263 = ptrtoint ptr addrspace(1) %address.local810 to i64, !dbg !19
  %dma.source.opaque1264 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1263) #9, !dbg !19
  %dma.source.ptr1265 = inttoptr i64 %dma.source.opaque1264 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1265, ptr addrspace(3) %307, i32 4, i32 0, i32 0), !dbg !19
  %address.local811 = shl nuw nsw i64 %address.local742, 1, !dbg !13
  %address.local812 = and i64 %address.local811, 384, !dbg !13
  %address.local813 = or disjoint i64 %address.local812, 2560, !dbg !13
  %address.local814 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local813, !dbg !13
  %address.local815 = ptrtoint ptr addrspace(3) %address.local814 to i64, !dbg !13
  %address.local816 = trunc i64 %address.local815 to i32, !dbg !13
  %308 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local816), !dbg !19
  %309 = zext i32 %308 to i64, !dbg !19
  %310 = inttoptr i64 %309 to ptr addrspace(3), !dbg !19
  %address.local817 = shl nuw nsw i32 %address.thread741, 1, !dbg !19
  %address.local818 = and i32 %address.local817, 30, !dbg !19
  %address.local819 = shl nuw nsw i32 %address.thread741, 3, !dbg !19
  %address.local820 = and i32 %address.local819, 1920, !dbg !19
  %address.local821 = or disjoint i32 %address.local818, %address.local820, !dbg !19
  %address.local822 = zext nneg i32 %address.local821 to i64, !dbg !19
  %address.local823 = getelementptr [2 x i8], ptr addrspace(1) %292, i64 %address.local822, !dbg !19
  %address.local824 = getelementptr i8, ptr addrspace(1) %address.local823, i64 20480, !dbg !19
  %dma.source.bits1266 = ptrtoint ptr addrspace(1) %address.local824 to i64, !dbg !19
  %dma.source.opaque1267 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1266) #9, !dbg !19
  %dma.source.ptr1268 = inttoptr i64 %dma.source.opaque1267 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1268, ptr addrspace(3) %310, i32 4, i32 0, i32 0), !dbg !19
  %address.local825 = shl nuw nsw i64 %address.local742, 1, !dbg !13
  %address.local826 = and i64 %address.local825, 384, !dbg !13
  %address.local827 = or disjoint i64 %address.local826, 3072, !dbg !13
  %address.local828 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local827, !dbg !13
  %address.local829 = ptrtoint ptr addrspace(3) %address.local828 to i64, !dbg !13
  %address.local830 = trunc i64 %address.local829 to i32, !dbg !13
  %311 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local830), !dbg !19
  %312 = zext i32 %311 to i64, !dbg !19
  %313 = inttoptr i64 %312 to ptr addrspace(3), !dbg !19
  %address.local831 = shl nuw nsw i32 %address.thread741, 1, !dbg !19
  %address.local832 = and i32 %address.local831, 30, !dbg !19
  %address.local833 = shl nuw nsw i32 %address.thread741, 3, !dbg !19
  %address.local834 = and i32 %address.local833, 1920, !dbg !19
  %address.local835 = or disjoint i32 %address.local832, %address.local834, !dbg !19
  %address.local836 = zext nneg i32 %address.local835 to i64, !dbg !19
  %address.local837 = getelementptr [2 x i8], ptr addrspace(1) %292, i64 %address.local836, !dbg !19
  %address.local838 = getelementptr i8, ptr addrspace(1) %address.local837, i64 24576, !dbg !19
  %dma.source.bits1269 = ptrtoint ptr addrspace(1) %address.local838 to i64, !dbg !19
  %dma.source.opaque1270 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1269) #9, !dbg !19
  %dma.source.ptr1271 = inttoptr i64 %dma.source.opaque1270 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1271, ptr addrspace(3) %313, i32 4, i32 0, i32 0), !dbg !19
  %address.local839 = shl nuw nsw i64 %address.local742, 1, !dbg !13
  %address.local840 = and i64 %address.local839, 384, !dbg !13
  %address.local841 = or disjoint i64 %address.local840, 3584, !dbg !13
  %address.local842 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local841, !dbg !13
  %address.local843 = ptrtoint ptr addrspace(3) %address.local842 to i64, !dbg !13
  %address.local844 = trunc i64 %address.local843 to i32, !dbg !13
  %314 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local844), !dbg !19
  %315 = zext i32 %314 to i64, !dbg !19
  %316 = inttoptr i64 %315 to ptr addrspace(3), !dbg !19
  %address.local845 = shl nuw nsw i32 %address.thread741, 1, !dbg !19
  %address.local846 = and i32 %address.local845, 30, !dbg !19
  %address.local847 = shl nuw nsw i32 %address.thread741, 3, !dbg !19
  %address.local848 = and i32 %address.local847, 1920, !dbg !19
  %address.local849 = or disjoint i32 %address.local846, %address.local848, !dbg !19
  %address.local850 = zext nneg i32 %address.local849 to i64, !dbg !19
  %address.local851 = getelementptr [2 x i8], ptr addrspace(1) %292, i64 %address.local850, !dbg !19
  %address.local852 = getelementptr i8, ptr addrspace(1) %address.local851, i64 28672, !dbg !19
  %dma.source.bits1272 = ptrtoint ptr addrspace(1) %address.local852 to i64, !dbg !19
  %dma.source.opaque1273 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1272) #9, !dbg !19
  %dma.source.ptr1274 = inttoptr i64 %dma.source.opaque1273 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1274, ptr addrspace(3) %316, i32 4, i32 0, i32 0), !dbg !19
  call void @llvm.amdgcn.asyncmark(), !dbg !19
  %317 = extractvalue [2 x <32 x float>] %50, 0, !dbg !15
  %318 = extractelement <32 x float> %317, i64 0, !dbg !15
  %319 = extractelement <32 x float> %317, i64 1, !dbg !15
  %320 = extractelement <32 x float> %317, i64 2, !dbg !15
  %321 = extractelement <32 x float> %317, i64 3, !dbg !15
  %322 = select i1 %.not, float %319, float %318, !dbg !15
  %323 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !15
  %324 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %323), !dbg !15
  %325 = add nuw nsw i32 %324, 64, !dbg !15
  %326 = xor i32 %324, 16, !dbg !15
  %327 = and i32 %325, 192, !dbg !15
  %328 = icmp samesign ult i32 %326, %327, !dbg !15
  %329 = select i1 %328, i32 %326, i32 %324, !dbg !15
  %330 = shl nuw nsw i32 %329, 2, !dbg !15
  %331 = bitcast float %322 to i32, !dbg !15
  %332 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %331), !dbg !15
  %333 = bitcast i32 %332 to float, !dbg !15
  %334 = select i1 %.not, float %318, float %333, !dbg !15
  %335 = select i1 %.not, float %333, float %319, !dbg !15
  %336 = select i1 %.not, float %321, float %320, !dbg !15
  %337 = bitcast float %336 to i32, !dbg !15
  %338 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %337), !dbg !15
  %339 = bitcast i32 %338 to float, !dbg !15
  %340 = select i1 %.not, float %320, float %339, !dbg !15
  %341 = select i1 %.not, float %339, float %321, !dbg !15
  %342 = select i1 %.not314, float %340, float %334, !dbg !15
  %343 = xor i32 %324, 32, !dbg !15
  %344 = icmp samesign ult i32 %343, %327, !dbg !15
  %345 = select i1 %344, i32 %343, i32 %324, !dbg !15
  %346 = shl nuw nsw i32 %345, 2, !dbg !15
  %347 = bitcast float %342 to i32, !dbg !15
  %348 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %347), !dbg !15
  %349 = bitcast i32 %348 to float, !dbg !15
  %350 = select i1 %.not314, float %334, float %349, !dbg !15
  %351 = select i1 %.not314, float %349, float %340, !dbg !15
  %352 = select i1 %.not314, float %341, float %335, !dbg !15
  %353 = bitcast float %352 to i32, !dbg !15
  %354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %353), !dbg !15
  %355 = bitcast i32 %354 to float, !dbg !15
  %356 = select i1 %.not314, float %335, float %355, !dbg !15
  %357 = select i1 %.not314, float %355, float %341, !dbg !15
  %358 = or disjoint i64 %44, %index.compact11, !dbg !20
  %359 = lshr i64 %32, 32, !dbg !26
  %compare.high = trunc i64 %359 to i32, !dbg !26
  %360 = lshr i64 %358, 32, !dbg !26
  %compare.high950 = trunc i64 %360 to i32, !dbg !26
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high950, !dbg !26
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high950, !dbg !26
  %compare.low.base = trunc i64 %32 to i32, !dbg !26
  %compare.low.base951 = trunc i64 %358 to i32, !dbg !26
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base951, !dbg !26
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !26
  %361 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %362 = fadd float %350, %361, !dbg !20
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 1) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !26
  %compare.low952 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !26
  %compare.result953 = select i1 %compare.high.equal, i1 %compare.low952, i1 %compare.high.order, !dbg !26
  %363 = select i1 %compare.result953, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %364 = fadd float %356, %363, !dbg !20
  %compare.local.base954 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 2) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local955 = or i32 %compare.local.base954, 2, !dbg !26
  %compare.low956 = icmp ugt i32 %compare.low.base, %compare.local955, !dbg !26
  %compare.result957 = select i1 %compare.high.equal, i1 %compare.low956, i1 %compare.high.order, !dbg !26
  %365 = select i1 %compare.result957, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %366 = fadd float %351, %365, !dbg !20
  %compare.local.base958 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 3) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local959 = or i32 %compare.local.base958, 3, !dbg !26
  %compare.low960 = icmp ugt i32 %compare.low.base, %compare.local959, !dbg !26
  %compare.result961 = select i1 %compare.high.equal, i1 %compare.low960, i1 %compare.high.order, !dbg !26
  %367 = select i1 %compare.result961, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %368 = fadd float %357, %367, !dbg !20
  %369 = call float @__ocml_exp2_f32(float %362), !dbg !28
  %370 = fadd float %369, 0x0000000000000000, !dbg !29
  %371 = insertelement <4 x float> poison, float %369, i64 0, !dbg !28
  %372 = call float @__ocml_exp2_f32(float %364), !dbg !28
  %373 = fadd float %370, %372, !dbg !29
  %374 = insertelement <4 x float> %371, float %372, i64 1, !dbg !28
  %375 = call float @__ocml_exp2_f32(float %366), !dbg !28
  %376 = fadd float %373, %375, !dbg !29
  %377 = insertelement <4 x float> %374, float %375, i64 2, !dbg !28
  %378 = call float @__ocml_exp2_f32(float %368), !dbg !28
  %379 = fadd float %376, %378, !dbg !29
  %380 = insertelement <4 x float> %377, float %378, i64 3, !dbg !28
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %380) #8, !dbg !30, !frisk.register_lifetimes !1
  %381 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !30
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %381) #8, !dbg !15, !frisk.register_lifetimes !1
  %382 = extractelement <32 x float> %317, i64 4, !dbg !15
  %383 = extractelement <32 x float> %317, i64 5, !dbg !15
  %384 = extractelement <32 x float> %317, i64 6, !dbg !15
  %385 = extractelement <32 x float> %317, i64 7, !dbg !15
  %386 = select i1 %.not, float %383, float %382, !dbg !15
  %387 = bitcast float %386 to i32, !dbg !15
  %388 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %387), !dbg !15
  %389 = bitcast i32 %388 to float, !dbg !15
  %390 = select i1 %.not, float %382, float %389, !dbg !15
  %391 = select i1 %.not, float %389, float %383, !dbg !15
  %392 = select i1 %.not, float %385, float %384, !dbg !15
  %393 = bitcast float %392 to i32, !dbg !15
  %394 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %393), !dbg !15
  %395 = bitcast i32 %394 to float, !dbg !15
  %396 = select i1 %.not, float %384, float %395, !dbg !15
  %397 = select i1 %.not, float %395, float %385, !dbg !15
  %398 = select i1 %.not314, float %396, float %390, !dbg !15
  %399 = bitcast float %398 to i32, !dbg !15
  %400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %399), !dbg !15
  %401 = bitcast i32 %400 to float, !dbg !15
  %402 = select i1 %.not314, float %390, float %401, !dbg !15
  %403 = select i1 %.not314, float %401, float %396, !dbg !15
  %404 = select i1 %.not314, float %397, float %391, !dbg !15
  %405 = bitcast float %404 to i32, !dbg !15
  %406 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %405), !dbg !15
  %407 = bitcast i32 %406 to float, !dbg !15
  %408 = select i1 %.not314, float %391, float %407, !dbg !15
  %409 = select i1 %.not314, float %407, float %397, !dbg !15
  %compare.local.base962 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 4) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local963 = or i32 %compare.local.base962, 16, !dbg !26
  %compare.low964 = icmp ugt i32 %compare.low.base, %compare.local963, !dbg !26
  %compare.result965 = select i1 %compare.high.equal, i1 %compare.low964, i1 %compare.high.order, !dbg !26
  %410 = select i1 %compare.result965, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %411 = fadd float %402, %410, !dbg !20
  %compare.local.base966 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 5) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local967 = or i32 %compare.local.base966, 17, !dbg !26
  %compare.low968 = icmp ugt i32 %compare.low.base, %compare.local967, !dbg !26
  %compare.result969 = select i1 %compare.high.equal, i1 %compare.low968, i1 %compare.high.order, !dbg !26
  %412 = select i1 %compare.result969, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %413 = fadd float %408, %412, !dbg !20
  %compare.local.base970 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 6) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local971 = or i32 %compare.local.base970, 18, !dbg !26
  %compare.low972 = icmp ugt i32 %compare.low.base, %compare.local971, !dbg !26
  %compare.result973 = select i1 %compare.high.equal, i1 %compare.low972, i1 %compare.high.order, !dbg !26
  %414 = select i1 %compare.result973, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %415 = fadd float %403, %414, !dbg !20
  %compare.local.base974 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 7) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local975 = or i32 %compare.local.base974, 19, !dbg !26
  %compare.low976 = icmp ugt i32 %compare.low.base, %compare.local975, !dbg !26
  %compare.result977 = select i1 %compare.high.equal, i1 %compare.low976, i1 %compare.high.order, !dbg !26
  %416 = select i1 %compare.result977, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %417 = fadd float %409, %416, !dbg !20
  %418 = call float @__ocml_exp2_f32(float %411), !dbg !28
  %419 = fadd float %379, %418, !dbg !29
  %420 = insertelement <4 x float> poison, float %418, i64 0, !dbg !28
  %421 = call float @__ocml_exp2_f32(float %413), !dbg !28
  %422 = fadd float %419, %421, !dbg !29
  %423 = insertelement <4 x float> %420, float %421, i64 1, !dbg !28
  %424 = call float @__ocml_exp2_f32(float %415), !dbg !28
  %425 = fadd float %422, %424, !dbg !29
  %426 = insertelement <4 x float> %423, float %424, i64 2, !dbg !28
  %427 = call float @__ocml_exp2_f32(float %417), !dbg !28
  %428 = fadd float %425, %427, !dbg !29
  %429 = insertelement <4 x float> %426, float %427, i64 3, !dbg !28
  %math.result387 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %429) #8, !dbg !30, !frisk.register_lifetimes !1
  %430 = fptrunc <4 x float> %math.result387 to <4 x half>, !dbg !30
  %math.packed388 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %430) #8, !dbg !15, !frisk.register_lifetimes !1
  %431 = extractelement <32 x float> %317, i64 8, !dbg !15
  %432 = extractelement <32 x float> %317, i64 9, !dbg !15
  %433 = extractelement <32 x float> %317, i64 10, !dbg !15
  %434 = extractelement <32 x float> %317, i64 11, !dbg !15
  %435 = select i1 %.not, float %432, float %431, !dbg !15
  %436 = bitcast float %435 to i32, !dbg !15
  %437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %436), !dbg !15
  %438 = bitcast i32 %437 to float, !dbg !15
  %439 = select i1 %.not, float %431, float %438, !dbg !15
  %440 = select i1 %.not, float %438, float %432, !dbg !15
  %441 = select i1 %.not, float %434, float %433, !dbg !15
  %442 = bitcast float %441 to i32, !dbg !15
  %443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %442), !dbg !15
  %444 = bitcast i32 %443 to float, !dbg !15
  %445 = select i1 %.not, float %433, float %444, !dbg !15
  %446 = select i1 %.not, float %444, float %434, !dbg !15
  %447 = select i1 %.not314, float %445, float %439, !dbg !15
  %448 = bitcast float %447 to i32, !dbg !15
  %449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %448), !dbg !15
  %450 = bitcast i32 %449 to float, !dbg !15
  %451 = select i1 %.not314, float %439, float %450, !dbg !15
  %452 = select i1 %.not314, float %450, float %445, !dbg !15
  %453 = select i1 %.not314, float %446, float %440, !dbg !15
  %454 = bitcast float %453 to i32, !dbg !15
  %455 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %454), !dbg !15
  %456 = bitcast i32 %455 to float, !dbg !15
  %457 = select i1 %.not314, float %440, float %456, !dbg !15
  %458 = select i1 %.not314, float %456, float %446, !dbg !15
  %compare.local.base978 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 8) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local979 = or i32 %compare.local.base978, 32, !dbg !26
  %compare.low980 = icmp ugt i32 %compare.low.base, %compare.local979, !dbg !26
  %compare.result981 = select i1 %compare.high.equal, i1 %compare.low980, i1 %compare.high.order, !dbg !26
  %459 = select i1 %compare.result981, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %460 = fadd float %451, %459, !dbg !20
  %compare.local.base982 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 9) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local983 = or i32 %compare.local.base982, 33, !dbg !26
  %compare.low984 = icmp ugt i32 %compare.low.base, %compare.local983, !dbg !26
  %compare.result985 = select i1 %compare.high.equal, i1 %compare.low984, i1 %compare.high.order, !dbg !26
  %461 = select i1 %compare.result985, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %462 = fadd float %457, %461, !dbg !20
  %compare.local.base986 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 10) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local987 = or i32 %compare.local.base986, 34, !dbg !26
  %compare.low988 = icmp ugt i32 %compare.low.base, %compare.local987, !dbg !26
  %compare.result989 = select i1 %compare.high.equal, i1 %compare.low988, i1 %compare.high.order, !dbg !26
  %463 = select i1 %compare.result989, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %464 = fadd float %452, %463, !dbg !20
  %compare.local.base990 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 11) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local991 = or i32 %compare.local.base990, 35, !dbg !26
  %compare.low992 = icmp ugt i32 %compare.low.base, %compare.local991, !dbg !26
  %compare.result993 = select i1 %compare.high.equal, i1 %compare.low992, i1 %compare.high.order, !dbg !26
  %465 = select i1 %compare.result993, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %466 = fadd float %458, %465, !dbg !20
  %467 = call float @__ocml_exp2_f32(float %460), !dbg !28
  %468 = fadd float %428, %467, !dbg !29
  %469 = insertelement <4 x float> poison, float %467, i64 0, !dbg !28
  %470 = call float @__ocml_exp2_f32(float %462), !dbg !28
  %471 = fadd float %468, %470, !dbg !29
  %472 = insertelement <4 x float> %469, float %470, i64 1, !dbg !28
  %473 = call float @__ocml_exp2_f32(float %464), !dbg !28
  %474 = fadd float %471, %473, !dbg !29
  %475 = insertelement <4 x float> %472, float %473, i64 2, !dbg !28
  %476 = call float @__ocml_exp2_f32(float %466), !dbg !28
  %477 = fadd float %474, %476, !dbg !29
  %478 = insertelement <4 x float> %475, float %476, i64 3, !dbg !28
  %math.result389 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %478) #8, !dbg !30, !frisk.register_lifetimes !1
  %479 = fptrunc <4 x float> %math.result389 to <4 x half>, !dbg !30
  %math.packed390 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %479) #8, !dbg !15, !frisk.register_lifetimes !1
  %480 = extractelement <32 x float> %317, i64 12, !dbg !15
  %481 = extractelement <32 x float> %317, i64 13, !dbg !15
  %482 = extractelement <32 x float> %317, i64 14, !dbg !15
  %483 = extractelement <32 x float> %317, i64 15, !dbg !15
  %484 = select i1 %.not, float %481, float %480, !dbg !15
  %485 = bitcast float %484 to i32, !dbg !15
  %486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %485), !dbg !15
  %487 = bitcast i32 %486 to float, !dbg !15
  %488 = select i1 %.not, float %480, float %487, !dbg !15
  %489 = select i1 %.not, float %487, float %481, !dbg !15
  %490 = select i1 %.not, float %483, float %482, !dbg !15
  %491 = bitcast float %490 to i32, !dbg !15
  %492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %491), !dbg !15
  %493 = bitcast i32 %492 to float, !dbg !15
  %494 = select i1 %.not, float %482, float %493, !dbg !15
  %495 = select i1 %.not, float %493, float %483, !dbg !15
  %496 = select i1 %.not314, float %494, float %488, !dbg !15
  %497 = bitcast float %496 to i32, !dbg !15
  %498 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %497), !dbg !15
  %499 = bitcast i32 %498 to float, !dbg !15
  %500 = select i1 %.not314, float %488, float %499, !dbg !15
  %501 = select i1 %.not314, float %499, float %494, !dbg !15
  %502 = select i1 %.not314, float %495, float %489, !dbg !15
  %503 = bitcast float %502 to i32, !dbg !15
  %504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %503), !dbg !15
  %505 = bitcast i32 %504 to float, !dbg !15
  %506 = select i1 %.not314, float %489, float %505, !dbg !15
  %507 = select i1 %.not314, float %505, float %495, !dbg !15
  %compare.local.base994 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 12) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local995 = or i32 %compare.local.base994, 48, !dbg !26
  %compare.low996 = icmp ugt i32 %compare.low.base, %compare.local995, !dbg !26
  %compare.result997 = select i1 %compare.high.equal, i1 %compare.low996, i1 %compare.high.order, !dbg !26
  %508 = select i1 %compare.result997, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %509 = fadd float %500, %508, !dbg !20
  %compare.local.base998 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 13) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local999 = or i32 %compare.local.base998, 49, !dbg !26
  %compare.low1000 = icmp ugt i32 %compare.low.base, %compare.local999, !dbg !26
  %compare.result1001 = select i1 %compare.high.equal, i1 %compare.low1000, i1 %compare.high.order, !dbg !26
  %510 = select i1 %compare.result1001, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %511 = fadd float %506, %510, !dbg !20
  %compare.local.base1002 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 14) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1003 = or i32 %compare.local.base1002, 50, !dbg !26
  %compare.low1004 = icmp ugt i32 %compare.low.base, %compare.local1003, !dbg !26
  %compare.result1005 = select i1 %compare.high.equal, i1 %compare.low1004, i1 %compare.high.order, !dbg !26
  %512 = select i1 %compare.result1005, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %513 = fadd float %501, %512, !dbg !20
  %compare.local.base1006 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 15) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1007 = or i32 %compare.local.base1006, 51, !dbg !26
  %compare.low1008 = icmp ugt i32 %compare.low.base, %compare.local1007, !dbg !26
  %compare.result1009 = select i1 %compare.high.equal, i1 %compare.low1008, i1 %compare.high.order, !dbg !26
  %514 = select i1 %compare.result1009, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %515 = fadd float %507, %514, !dbg !20
  %516 = call float @__ocml_exp2_f32(float %509), !dbg !28
  %517 = fadd float %477, %516, !dbg !29
  %518 = insertelement <4 x float> poison, float %516, i64 0, !dbg !28
  %519 = call float @__ocml_exp2_f32(float %511), !dbg !28
  %520 = fadd float %517, %519, !dbg !29
  %521 = insertelement <4 x float> %518, float %519, i64 1, !dbg !28
  %522 = call float @__ocml_exp2_f32(float %513), !dbg !28
  %523 = fadd float %520, %522, !dbg !29
  %524 = insertelement <4 x float> %521, float %522, i64 2, !dbg !28
  %525 = call float @__ocml_exp2_f32(float %515), !dbg !28
  %526 = fadd float %523, %525, !dbg !29
  %527 = insertelement <4 x float> %524, float %525, i64 3, !dbg !28
  %math.result391 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %527) #8, !dbg !30, !frisk.register_lifetimes !1
  %528 = fptrunc <4 x float> %math.result391 to <4 x half>, !dbg !30
  %math.packed392 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %528) #8, !dbg !15, !frisk.register_lifetimes !1
  %529 = extractelement <32 x float> %317, i64 16, !dbg !15
  %530 = extractelement <32 x float> %317, i64 17, !dbg !15
  %531 = extractelement <32 x float> %317, i64 18, !dbg !15
  %532 = extractelement <32 x float> %317, i64 19, !dbg !15
  %533 = select i1 %.not, float %530, float %529, !dbg !15
  %534 = bitcast float %533 to i32, !dbg !15
  %535 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %534), !dbg !15
  %536 = bitcast i32 %535 to float, !dbg !15
  %537 = select i1 %.not, float %529, float %536, !dbg !15
  %538 = select i1 %.not, float %536, float %530, !dbg !15
  %539 = select i1 %.not, float %532, float %531, !dbg !15
  %540 = bitcast float %539 to i32, !dbg !15
  %541 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %540), !dbg !15
  %542 = bitcast i32 %541 to float, !dbg !15
  %543 = select i1 %.not, float %531, float %542, !dbg !15
  %544 = select i1 %.not, float %542, float %532, !dbg !15
  %545 = select i1 %.not314, float %543, float %537, !dbg !15
  %546 = bitcast float %545 to i32, !dbg !15
  %547 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %546), !dbg !15
  %548 = bitcast i32 %547 to float, !dbg !15
  %549 = select i1 %.not314, float %537, float %548, !dbg !15
  %550 = select i1 %.not314, float %548, float %543, !dbg !15
  %551 = select i1 %.not314, float %544, float %538, !dbg !15
  %552 = bitcast float %551 to i32, !dbg !15
  %553 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %552), !dbg !15
  %554 = bitcast i32 %553 to float, !dbg !15
  %555 = select i1 %.not314, float %538, float %554, !dbg !15
  %556 = select i1 %.not314, float %554, float %544, !dbg !15
  %compare.local.base1010 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 16) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1011 = or i32 %compare.local.base1010, 64, !dbg !26
  %compare.low1012 = icmp ugt i32 %compare.low.base, %compare.local1011, !dbg !26
  %compare.result1013 = select i1 %compare.high.equal, i1 %compare.low1012, i1 %compare.high.order, !dbg !26
  %557 = select i1 %compare.result1013, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %558 = fadd float %549, %557, !dbg !20
  %compare.local.base1014 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 17) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1015 = or i32 %compare.local.base1014, 65, !dbg !26
  %compare.low1016 = icmp ugt i32 %compare.low.base, %compare.local1015, !dbg !26
  %compare.result1017 = select i1 %compare.high.equal, i1 %compare.low1016, i1 %compare.high.order, !dbg !26
  %559 = select i1 %compare.result1017, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %560 = fadd float %555, %559, !dbg !20
  %compare.local.base1018 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 18) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1019 = or i32 %compare.local.base1018, 66, !dbg !26
  %compare.low1020 = icmp ugt i32 %compare.low.base, %compare.local1019, !dbg !26
  %compare.result1021 = select i1 %compare.high.equal, i1 %compare.low1020, i1 %compare.high.order, !dbg !26
  %561 = select i1 %compare.result1021, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %562 = fadd float %550, %561, !dbg !20
  %compare.local.base1022 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 19) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1023 = or i32 %compare.local.base1022, 67, !dbg !26
  %compare.low1024 = icmp ugt i32 %compare.low.base, %compare.local1023, !dbg !26
  %compare.result1025 = select i1 %compare.high.equal, i1 %compare.low1024, i1 %compare.high.order, !dbg !26
  %563 = select i1 %compare.result1025, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %564 = fadd float %556, %563, !dbg !20
  %565 = call float @__ocml_exp2_f32(float %558), !dbg !28
  %566 = fadd float %526, %565, !dbg !29
  %567 = insertelement <4 x float> poison, float %565, i64 0, !dbg !28
  %568 = call float @__ocml_exp2_f32(float %560), !dbg !28
  %569 = fadd float %566, %568, !dbg !29
  %570 = insertelement <4 x float> %567, float %568, i64 1, !dbg !28
  %571 = call float @__ocml_exp2_f32(float %562), !dbg !28
  %572 = fadd float %569, %571, !dbg !29
  %573 = insertelement <4 x float> %570, float %571, i64 2, !dbg !28
  %574 = call float @__ocml_exp2_f32(float %564), !dbg !28
  %575 = fadd float %572, %574, !dbg !29
  %576 = insertelement <4 x float> %573, float %574, i64 3, !dbg !28
  %math.result393 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %576) #8, !dbg !30, !frisk.register_lifetimes !1
  %577 = fptrunc <4 x float> %math.result393 to <4 x half>, !dbg !30
  %math.packed394 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %577) #8, !dbg !15, !frisk.register_lifetimes !1
  %578 = extractelement <32 x float> %317, i64 20, !dbg !15
  %579 = extractelement <32 x float> %317, i64 21, !dbg !15
  %580 = extractelement <32 x float> %317, i64 22, !dbg !15
  %581 = extractelement <32 x float> %317, i64 23, !dbg !15
  %582 = select i1 %.not, float %579, float %578, !dbg !15
  %583 = bitcast float %582 to i32, !dbg !15
  %584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %583), !dbg !15
  %585 = bitcast i32 %584 to float, !dbg !15
  %586 = select i1 %.not, float %578, float %585, !dbg !15
  %587 = select i1 %.not, float %585, float %579, !dbg !15
  %588 = select i1 %.not, float %581, float %580, !dbg !15
  %589 = bitcast float %588 to i32, !dbg !15
  %590 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %589), !dbg !15
  %591 = bitcast i32 %590 to float, !dbg !15
  %592 = select i1 %.not, float %580, float %591, !dbg !15
  %593 = select i1 %.not, float %591, float %581, !dbg !15
  %594 = select i1 %.not314, float %592, float %586, !dbg !15
  %595 = bitcast float %594 to i32, !dbg !15
  %596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %595), !dbg !15
  %597 = bitcast i32 %596 to float, !dbg !15
  %598 = select i1 %.not314, float %586, float %597, !dbg !15
  %599 = select i1 %.not314, float %597, float %592, !dbg !15
  %600 = select i1 %.not314, float %593, float %587, !dbg !15
  %601 = bitcast float %600 to i32, !dbg !15
  %602 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %601), !dbg !15
  %603 = bitcast i32 %602 to float, !dbg !15
  %604 = select i1 %.not314, float %587, float %603, !dbg !15
  %605 = select i1 %.not314, float %603, float %593, !dbg !15
  %compare.local.base1026 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 20) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1027 = or i32 %compare.local.base1026, 80, !dbg !26
  %compare.low1028 = icmp ugt i32 %compare.low.base, %compare.local1027, !dbg !26
  %compare.result1029 = select i1 %compare.high.equal, i1 %compare.low1028, i1 %compare.high.order, !dbg !26
  %606 = select i1 %compare.result1029, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %607 = fadd float %598, %606, !dbg !20
  %compare.local.base1030 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 21) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1031 = or i32 %compare.local.base1030, 81, !dbg !26
  %compare.low1032 = icmp ugt i32 %compare.low.base, %compare.local1031, !dbg !26
  %compare.result1033 = select i1 %compare.high.equal, i1 %compare.low1032, i1 %compare.high.order, !dbg !26
  %608 = select i1 %compare.result1033, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %609 = fadd float %604, %608, !dbg !20
  %compare.local.base1034 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 22) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1035 = or i32 %compare.local.base1034, 82, !dbg !26
  %compare.low1036 = icmp ugt i32 %compare.low.base, %compare.local1035, !dbg !26
  %compare.result1037 = select i1 %compare.high.equal, i1 %compare.low1036, i1 %compare.high.order, !dbg !26
  %610 = select i1 %compare.result1037, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %611 = fadd float %599, %610, !dbg !20
  %compare.local.base1038 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 23) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1039 = or i32 %compare.local.base1038, 83, !dbg !26
  %compare.low1040 = icmp ugt i32 %compare.low.base, %compare.local1039, !dbg !26
  %compare.result1041 = select i1 %compare.high.equal, i1 %compare.low1040, i1 %compare.high.order, !dbg !26
  %612 = select i1 %compare.result1041, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %613 = fadd float %605, %612, !dbg !20
  %614 = call float @__ocml_exp2_f32(float %607), !dbg !28
  %615 = fadd float %575, %614, !dbg !29
  %616 = insertelement <4 x float> poison, float %614, i64 0, !dbg !28
  %617 = call float @__ocml_exp2_f32(float %609), !dbg !28
  %618 = fadd float %615, %617, !dbg !29
  %619 = insertelement <4 x float> %616, float %617, i64 1, !dbg !28
  %620 = call float @__ocml_exp2_f32(float %611), !dbg !28
  %621 = fadd float %618, %620, !dbg !29
  %622 = insertelement <4 x float> %619, float %620, i64 2, !dbg !28
  %623 = call float @__ocml_exp2_f32(float %613), !dbg !28
  %624 = fadd float %621, %623, !dbg !29
  %625 = insertelement <4 x float> %622, float %623, i64 3, !dbg !28
  %math.result395 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %625) #8, !dbg !30, !frisk.register_lifetimes !1
  %626 = fptrunc <4 x float> %math.result395 to <4 x half>, !dbg !30
  %math.packed396 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %626) #8, !dbg !15, !frisk.register_lifetimes !1
  %627 = extractelement <32 x float> %317, i64 24, !dbg !15
  %628 = extractelement <32 x float> %317, i64 25, !dbg !15
  %629 = extractelement <32 x float> %317, i64 26, !dbg !15
  %630 = extractelement <32 x float> %317, i64 27, !dbg !15
  %631 = select i1 %.not, float %628, float %627, !dbg !15
  %632 = bitcast float %631 to i32, !dbg !15
  %633 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %632), !dbg !15
  %634 = bitcast i32 %633 to float, !dbg !15
  %635 = select i1 %.not, float %627, float %634, !dbg !15
  %636 = select i1 %.not, float %634, float %628, !dbg !15
  %637 = select i1 %.not, float %630, float %629, !dbg !15
  %638 = bitcast float %637 to i32, !dbg !15
  %639 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %638), !dbg !15
  %640 = bitcast i32 %639 to float, !dbg !15
  %641 = select i1 %.not, float %629, float %640, !dbg !15
  %642 = select i1 %.not, float %640, float %630, !dbg !15
  %643 = select i1 %.not314, float %641, float %635, !dbg !15
  %644 = bitcast float %643 to i32, !dbg !15
  %645 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %644), !dbg !15
  %646 = bitcast i32 %645 to float, !dbg !15
  %647 = select i1 %.not314, float %635, float %646, !dbg !15
  %648 = select i1 %.not314, float %646, float %641, !dbg !15
  %649 = select i1 %.not314, float %642, float %636, !dbg !15
  %650 = bitcast float %649 to i32, !dbg !15
  %651 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %650), !dbg !15
  %652 = bitcast i32 %651 to float, !dbg !15
  %653 = select i1 %.not314, float %636, float %652, !dbg !15
  %654 = select i1 %.not314, float %652, float %642, !dbg !15
  %compare.local.base1042 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 24) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1043 = or i32 %compare.local.base1042, 96, !dbg !26
  %compare.low1044 = icmp ugt i32 %compare.low.base, %compare.local1043, !dbg !26
  %compare.result1045 = select i1 %compare.high.equal, i1 %compare.low1044, i1 %compare.high.order, !dbg !26
  %655 = select i1 %compare.result1045, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %656 = fadd float %647, %655, !dbg !20
  %compare.local.base1046 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 25) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1047 = or i32 %compare.local.base1046, 97, !dbg !26
  %compare.low1048 = icmp ugt i32 %compare.low.base, %compare.local1047, !dbg !26
  %compare.result1049 = select i1 %compare.high.equal, i1 %compare.low1048, i1 %compare.high.order, !dbg !26
  %657 = select i1 %compare.result1049, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %658 = fadd float %653, %657, !dbg !20
  %compare.local.base1050 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 26) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1051 = or i32 %compare.local.base1050, 98, !dbg !26
  %compare.low1052 = icmp ugt i32 %compare.low.base, %compare.local1051, !dbg !26
  %compare.result1053 = select i1 %compare.high.equal, i1 %compare.low1052, i1 %compare.high.order, !dbg !26
  %659 = select i1 %compare.result1053, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %660 = fadd float %648, %659, !dbg !20
  %compare.local.base1054 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 27) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1055 = or i32 %compare.local.base1054, 99, !dbg !26
  %compare.low1056 = icmp ugt i32 %compare.low.base, %compare.local1055, !dbg !26
  %compare.result1057 = select i1 %compare.high.equal, i1 %compare.low1056, i1 %compare.high.order, !dbg !26
  %661 = select i1 %compare.result1057, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %662 = fadd float %654, %661, !dbg !20
  %663 = call float @__ocml_exp2_f32(float %656), !dbg !28
  %664 = fadd float %624, %663, !dbg !29
  %665 = insertelement <4 x float> poison, float %663, i64 0, !dbg !28
  %666 = call float @__ocml_exp2_f32(float %658), !dbg !28
  %667 = fadd float %664, %666, !dbg !29
  %668 = insertelement <4 x float> %665, float %666, i64 1, !dbg !28
  %669 = call float @__ocml_exp2_f32(float %660), !dbg !28
  %670 = fadd float %667, %669, !dbg !29
  %671 = insertelement <4 x float> %668, float %669, i64 2, !dbg !28
  %672 = call float @__ocml_exp2_f32(float %662), !dbg !28
  %673 = fadd float %670, %672, !dbg !29
  %674 = insertelement <4 x float> %671, float %672, i64 3, !dbg !28
  %math.result397 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %674) #8, !dbg !30, !frisk.register_lifetimes !1
  %675 = fptrunc <4 x float> %math.result397 to <4 x half>, !dbg !30
  %math.packed398 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %675) #8, !dbg !15, !frisk.register_lifetimes !1
  %676 = extractelement <32 x float> %317, i64 28, !dbg !15
  %677 = extractelement <32 x float> %317, i64 29, !dbg !15
  %678 = extractelement <32 x float> %317, i64 30, !dbg !15
  %679 = extractelement <32 x float> %317, i64 31, !dbg !15
  %680 = select i1 %.not, float %677, float %676, !dbg !15
  %681 = bitcast float %680 to i32, !dbg !15
  %682 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %681), !dbg !15
  %683 = bitcast i32 %682 to float, !dbg !15
  %684 = select i1 %.not, float %676, float %683, !dbg !15
  %685 = select i1 %.not, float %683, float %677, !dbg !15
  %686 = select i1 %.not, float %679, float %678, !dbg !15
  %687 = bitcast float %686 to i32, !dbg !15
  %688 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %687), !dbg !15
  %689 = bitcast i32 %688 to float, !dbg !15
  %690 = select i1 %.not, float %678, float %689, !dbg !15
  %691 = select i1 %.not, float %689, float %679, !dbg !15
  %692 = select i1 %.not314, float %690, float %684, !dbg !15
  %693 = bitcast float %692 to i32, !dbg !15
  %694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %693), !dbg !15
  %695 = bitcast i32 %694 to float, !dbg !15
  %696 = select i1 %.not314, float %684, float %695, !dbg !15
  %697 = select i1 %.not314, float %695, float %690, !dbg !15
  %698 = select i1 %.not314, float %691, float %685, !dbg !15
  %699 = bitcast float %698 to i32, !dbg !15
  %700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %699), !dbg !15
  %701 = bitcast i32 %700 to float, !dbg !15
  %702 = select i1 %.not314, float %685, float %701, !dbg !15
  %703 = select i1 %.not314, float %701, float %691, !dbg !15
  %compare.local.base1058 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 28) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1059 = or i32 %compare.local.base1058, 112, !dbg !26
  %compare.low1060 = icmp ugt i32 %compare.low.base, %compare.local1059, !dbg !26
  %compare.result1061 = select i1 %compare.high.equal, i1 %compare.low1060, i1 %compare.high.order, !dbg !26
  %704 = select i1 %compare.result1061, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %705 = fadd float %696, %704, !dbg !20
  %compare.local.base1062 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 29) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1063 = or i32 %compare.local.base1062, 113, !dbg !26
  %compare.low1064 = icmp ugt i32 %compare.low.base, %compare.local1063, !dbg !26
  %compare.result1065 = select i1 %compare.high.equal, i1 %compare.low1064, i1 %compare.high.order, !dbg !26
  %706 = select i1 %compare.result1065, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %707 = fadd float %702, %706, !dbg !20
  %compare.local.base1066 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 30) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1067 = or i32 %compare.local.base1066, 114, !dbg !26
  %compare.low1068 = icmp ugt i32 %compare.low.base, %compare.local1067, !dbg !26
  %compare.result1069 = select i1 %compare.high.equal, i1 %compare.low1068, i1 %compare.high.order, !dbg !26
  %708 = select i1 %compare.result1069, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %709 = fadd float %697, %708, !dbg !20
  %compare.local.base1070 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 31) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1071 = or i32 %compare.local.base1070, 115, !dbg !26
  %compare.low1072 = icmp ugt i32 %compare.low.base, %compare.local1071, !dbg !26
  %compare.result1073 = select i1 %compare.high.equal, i1 %compare.low1072, i1 %compare.high.order, !dbg !26
  %710 = select i1 %compare.result1073, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %711 = fadd float %703, %710, !dbg !20
  %712 = call float @__ocml_exp2_f32(float %705), !dbg !28
  %713 = fadd float %673, %712, !dbg !29
  %714 = insertelement <4 x float> poison, float %712, i64 0, !dbg !28
  %715 = call float @__ocml_exp2_f32(float %707), !dbg !28
  %716 = fadd float %713, %715, !dbg !29
  %717 = insertelement <4 x float> %714, float %715, i64 1, !dbg !28
  %718 = call float @__ocml_exp2_f32(float %709), !dbg !28
  %719 = fadd float %716, %718, !dbg !29
  %720 = insertelement <4 x float> %717, float %718, i64 2, !dbg !28
  %721 = call float @__ocml_exp2_f32(float %711), !dbg !28
  %722 = fadd float %719, %721, !dbg !29
  %723 = insertelement <4 x float> %720, float %721, i64 3, !dbg !28
  %math.result399 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %723) #8, !dbg !30, !frisk.register_lifetimes !1
  %724 = fptrunc <4 x float> %math.result399 to <4 x half>, !dbg !30
  %math.packed400 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %724) #8, !dbg !15, !frisk.register_lifetimes !1
  %725 = extractvalue [2 x <32 x float>] %50, 1, !dbg !15
  %726 = extractelement <32 x float> %725, i64 0, !dbg !15
  %727 = extractelement <32 x float> %725, i64 1, !dbg !15
  %728 = extractelement <32 x float> %725, i64 2, !dbg !15
  %729 = extractelement <32 x float> %725, i64 3, !dbg !15
  %730 = select i1 %.not, float %727, float %726, !dbg !15
  %731 = bitcast float %730 to i32, !dbg !15
  %732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %731), !dbg !15
  %733 = bitcast i32 %732 to float, !dbg !15
  %734 = select i1 %.not, float %726, float %733, !dbg !15
  %735 = select i1 %.not, float %733, float %727, !dbg !15
  %736 = select i1 %.not, float %729, float %728, !dbg !15
  %737 = bitcast float %736 to i32, !dbg !15
  %738 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %737), !dbg !15
  %739 = bitcast i32 %738 to float, !dbg !15
  %740 = select i1 %.not, float %728, float %739, !dbg !15
  %741 = select i1 %.not, float %739, float %729, !dbg !15
  %742 = select i1 %.not314, float %740, float %734, !dbg !15
  %743 = bitcast float %742 to i32, !dbg !15
  %744 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %743), !dbg !15
  %745 = bitcast i32 %744 to float, !dbg !15
  %746 = select i1 %.not314, float %734, float %745, !dbg !15
  %747 = select i1 %.not314, float %745, float %740, !dbg !15
  %748 = select i1 %.not314, float %741, float %735, !dbg !15
  %749 = bitcast float %748 to i32, !dbg !15
  %750 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %749), !dbg !15
  %751 = bitcast i32 %750 to float, !dbg !15
  %752 = select i1 %.not314, float %735, float %751, !dbg !15
  %753 = select i1 %.not314, float %751, float %741, !dbg !15
  %754 = lshr i64 %33, 32, !dbg !26
  %compare.high1074 = trunc i64 %754 to i32, !dbg !26
  %755 = lshr i64 %358, 32, !dbg !26
  %compare.high1075 = trunc i64 %755 to i32, !dbg !26
  %compare.high.equal1076 = icmp eq i32 %compare.high1074, %compare.high1075, !dbg !26
  %compare.high.order1077 = icmp ugt i32 %compare.high1074, %compare.high1075, !dbg !26
  %compare.low.base1078 = trunc i64 %33 to i32, !dbg !26
  %compare.low.base1079 = trunc i64 %358 to i32, !dbg !26
  %compare.low1080 = icmp ugt i32 %compare.low.base1078, %compare.low.base1079, !dbg !26
  %compare.result1081 = select i1 %compare.high.equal1076, i1 %compare.low1080, i1 %compare.high.order1077, !dbg !26
  %756 = select i1 %compare.result1081, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %757 = fadd float %746, %756, !dbg !20
  %compare.local.base1082 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 32) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1083 = or i32 %compare.local.base1082, 1, !dbg !26
  %compare.low1084 = icmp ugt i32 %compare.low.base1078, %compare.local1083, !dbg !26
  %compare.result1085 = select i1 %compare.high.equal1076, i1 %compare.low1084, i1 %compare.high.order1077, !dbg !26
  %758 = select i1 %compare.result1085, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %759 = fadd float %752, %758, !dbg !20
  %compare.local.base1086 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 33) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1087 = or i32 %compare.local.base1086, 2, !dbg !26
  %compare.low1088 = icmp ugt i32 %compare.low.base1078, %compare.local1087, !dbg !26
  %compare.result1089 = select i1 %compare.high.equal1076, i1 %compare.low1088, i1 %compare.high.order1077, !dbg !26
  %760 = select i1 %compare.result1089, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %761 = fadd float %747, %760, !dbg !20
  %compare.local.base1090 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 34) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1091 = or i32 %compare.local.base1090, 3, !dbg !26
  %compare.low1092 = icmp ugt i32 %compare.low.base1078, %compare.local1091, !dbg !26
  %compare.result1093 = select i1 %compare.high.equal1076, i1 %compare.low1092, i1 %compare.high.order1077, !dbg !26
  %762 = select i1 %compare.result1093, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %763 = fadd float %753, %762, !dbg !20
  %764 = call float @__ocml_exp2_f32(float %757), !dbg !28
  %765 = fadd float %764, 0x0000000000000000, !dbg !29
  %766 = insertelement <4 x float> poison, float %764, i64 0, !dbg !28
  %767 = call float @__ocml_exp2_f32(float %759), !dbg !28
  %768 = fadd float %765, %767, !dbg !29
  %769 = insertelement <4 x float> %766, float %767, i64 1, !dbg !28
  %770 = call float @__ocml_exp2_f32(float %761), !dbg !28
  %771 = fadd float %768, %770, !dbg !29
  %772 = insertelement <4 x float> %769, float %770, i64 2, !dbg !28
  %773 = call float @__ocml_exp2_f32(float %763), !dbg !28
  %774 = fadd float %771, %773, !dbg !29
  %775 = insertelement <4 x float> %772, float %773, i64 3, !dbg !28
  %math.result401 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %775) #8, !dbg !30, !frisk.register_lifetimes !1
  %776 = fptrunc <4 x float> %math.result401 to <4 x half>, !dbg !30
  %math.packed402 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %776) #8, !dbg !15, !frisk.register_lifetimes !1
  %777 = extractelement <32 x float> %725, i64 4, !dbg !15
  %778 = extractelement <32 x float> %725, i64 5, !dbg !15
  %779 = extractelement <32 x float> %725, i64 6, !dbg !15
  %780 = extractelement <32 x float> %725, i64 7, !dbg !15
  %781 = select i1 %.not, float %778, float %777, !dbg !15
  %782 = bitcast float %781 to i32, !dbg !15
  %783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %782), !dbg !15
  %784 = bitcast i32 %783 to float, !dbg !15
  %785 = select i1 %.not, float %777, float %784, !dbg !15
  %786 = select i1 %.not, float %784, float %778, !dbg !15
  %787 = select i1 %.not, float %780, float %779, !dbg !15
  %788 = bitcast float %787 to i32, !dbg !15
  %789 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %788), !dbg !15
  %790 = bitcast i32 %789 to float, !dbg !15
  %791 = select i1 %.not, float %779, float %790, !dbg !15
  %792 = select i1 %.not, float %790, float %780, !dbg !15
  %793 = select i1 %.not314, float %791, float %785, !dbg !15
  %794 = bitcast float %793 to i32, !dbg !15
  %795 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %794), !dbg !15
  %796 = bitcast i32 %795 to float, !dbg !15
  %797 = select i1 %.not314, float %785, float %796, !dbg !15
  %798 = select i1 %.not314, float %796, float %791, !dbg !15
  %799 = select i1 %.not314, float %792, float %786, !dbg !15
  %800 = bitcast float %799 to i32, !dbg !15
  %801 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %800), !dbg !15
  %802 = bitcast i32 %801 to float, !dbg !15
  %803 = select i1 %.not314, float %786, float %802, !dbg !15
  %804 = select i1 %.not314, float %802, float %792, !dbg !15
  %compare.local.base1094 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 35) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1095 = or i32 %compare.local.base1094, 16, !dbg !26
  %compare.low1096 = icmp ugt i32 %compare.low.base1078, %compare.local1095, !dbg !26
  %compare.result1097 = select i1 %compare.high.equal1076, i1 %compare.low1096, i1 %compare.high.order1077, !dbg !26
  %805 = select i1 %compare.result1097, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %806 = fadd float %797, %805, !dbg !20
  %compare.local.base1098 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 36) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1099 = or i32 %compare.local.base1098, 17, !dbg !26
  %compare.low1100 = icmp ugt i32 %compare.low.base1078, %compare.local1099, !dbg !26
  %compare.result1101 = select i1 %compare.high.equal1076, i1 %compare.low1100, i1 %compare.high.order1077, !dbg !26
  %807 = select i1 %compare.result1101, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %808 = fadd float %803, %807, !dbg !20
  %compare.local.base1102 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 37) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1103 = or i32 %compare.local.base1102, 18, !dbg !26
  %compare.low1104 = icmp ugt i32 %compare.low.base1078, %compare.local1103, !dbg !26
  %compare.result1105 = select i1 %compare.high.equal1076, i1 %compare.low1104, i1 %compare.high.order1077, !dbg !26
  %809 = select i1 %compare.result1105, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %810 = fadd float %798, %809, !dbg !20
  %compare.local.base1106 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 38) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1107 = or i32 %compare.local.base1106, 19, !dbg !26
  %compare.low1108 = icmp ugt i32 %compare.low.base1078, %compare.local1107, !dbg !26
  %compare.result1109 = select i1 %compare.high.equal1076, i1 %compare.low1108, i1 %compare.high.order1077, !dbg !26
  %811 = select i1 %compare.result1109, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %812 = fadd float %804, %811, !dbg !20
  %813 = call float @__ocml_exp2_f32(float %806), !dbg !28
  %814 = fadd float %774, %813, !dbg !29
  %815 = insertelement <4 x float> poison, float %813, i64 0, !dbg !28
  %816 = call float @__ocml_exp2_f32(float %808), !dbg !28
  %817 = fadd float %814, %816, !dbg !29
  %818 = insertelement <4 x float> %815, float %816, i64 1, !dbg !28
  %819 = call float @__ocml_exp2_f32(float %810), !dbg !28
  %820 = fadd float %817, %819, !dbg !29
  %821 = insertelement <4 x float> %818, float %819, i64 2, !dbg !28
  %822 = call float @__ocml_exp2_f32(float %812), !dbg !28
  %823 = fadd float %820, %822, !dbg !29
  %824 = insertelement <4 x float> %821, float %822, i64 3, !dbg !28
  %math.result403 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %824) #8, !dbg !30, !frisk.register_lifetimes !1
  %825 = fptrunc <4 x float> %math.result403 to <4 x half>, !dbg !30
  %math.packed404 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %825) #8, !dbg !15, !frisk.register_lifetimes !1
  %826 = extractelement <32 x float> %725, i64 8, !dbg !15
  %827 = extractelement <32 x float> %725, i64 9, !dbg !15
  %828 = extractelement <32 x float> %725, i64 10, !dbg !15
  %829 = extractelement <32 x float> %725, i64 11, !dbg !15
  %830 = select i1 %.not, float %827, float %826, !dbg !15
  %831 = bitcast float %830 to i32, !dbg !15
  %832 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %831), !dbg !15
  %833 = bitcast i32 %832 to float, !dbg !15
  %834 = select i1 %.not, float %826, float %833, !dbg !15
  %835 = select i1 %.not, float %833, float %827, !dbg !15
  %836 = select i1 %.not, float %829, float %828, !dbg !15
  %837 = bitcast float %836 to i32, !dbg !15
  %838 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %837), !dbg !15
  %839 = bitcast i32 %838 to float, !dbg !15
  %840 = select i1 %.not, float %828, float %839, !dbg !15
  %841 = select i1 %.not, float %839, float %829, !dbg !15
  %842 = select i1 %.not314, float %840, float %834, !dbg !15
  %843 = bitcast float %842 to i32, !dbg !15
  %844 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %843), !dbg !15
  %845 = bitcast i32 %844 to float, !dbg !15
  %846 = select i1 %.not314, float %834, float %845, !dbg !15
  %847 = select i1 %.not314, float %845, float %840, !dbg !15
  %848 = select i1 %.not314, float %841, float %835, !dbg !15
  %849 = bitcast float %848 to i32, !dbg !15
  %850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %849), !dbg !15
  %851 = bitcast i32 %850 to float, !dbg !15
  %852 = select i1 %.not314, float %835, float %851, !dbg !15
  %853 = select i1 %.not314, float %851, float %841, !dbg !15
  %compare.local.base1110 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 39) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1111 = or i32 %compare.local.base1110, 32, !dbg !26
  %compare.low1112 = icmp ugt i32 %compare.low.base1078, %compare.local1111, !dbg !26
  %compare.result1113 = select i1 %compare.high.equal1076, i1 %compare.low1112, i1 %compare.high.order1077, !dbg !26
  %854 = select i1 %compare.result1113, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %855 = fadd float %846, %854, !dbg !20
  %compare.local.base1114 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 40) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1115 = or i32 %compare.local.base1114, 33, !dbg !26
  %compare.low1116 = icmp ugt i32 %compare.low.base1078, %compare.local1115, !dbg !26
  %compare.result1117 = select i1 %compare.high.equal1076, i1 %compare.low1116, i1 %compare.high.order1077, !dbg !26
  %856 = select i1 %compare.result1117, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %857 = fadd float %852, %856, !dbg !20
  %compare.local.base1118 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 41) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1119 = or i32 %compare.local.base1118, 34, !dbg !26
  %compare.low1120 = icmp ugt i32 %compare.low.base1078, %compare.local1119, !dbg !26
  %compare.result1121 = select i1 %compare.high.equal1076, i1 %compare.low1120, i1 %compare.high.order1077, !dbg !26
  %858 = select i1 %compare.result1121, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %859 = fadd float %847, %858, !dbg !20
  %compare.local.base1122 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 42) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1123 = or i32 %compare.local.base1122, 35, !dbg !26
  %compare.low1124 = icmp ugt i32 %compare.low.base1078, %compare.local1123, !dbg !26
  %compare.result1125 = select i1 %compare.high.equal1076, i1 %compare.low1124, i1 %compare.high.order1077, !dbg !26
  %860 = select i1 %compare.result1125, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %861 = fadd float %853, %860, !dbg !20
  %862 = call float @__ocml_exp2_f32(float %855), !dbg !28
  %863 = fadd float %823, %862, !dbg !29
  %864 = insertelement <4 x float> poison, float %862, i64 0, !dbg !28
  %865 = call float @__ocml_exp2_f32(float %857), !dbg !28
  %866 = fadd float %863, %865, !dbg !29
  %867 = insertelement <4 x float> %864, float %865, i64 1, !dbg !28
  %868 = call float @__ocml_exp2_f32(float %859), !dbg !28
  %869 = fadd float %866, %868, !dbg !29
  %870 = insertelement <4 x float> %867, float %868, i64 2, !dbg !28
  %871 = call float @__ocml_exp2_f32(float %861), !dbg !28
  %872 = fadd float %869, %871, !dbg !29
  %873 = insertelement <4 x float> %870, float %871, i64 3, !dbg !28
  %math.result405 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %873) #8, !dbg !30, !frisk.register_lifetimes !1
  %874 = fptrunc <4 x float> %math.result405 to <4 x half>, !dbg !30
  %math.packed406 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %874) #8, !dbg !15, !frisk.register_lifetimes !1
  %875 = extractelement <32 x float> %725, i64 12, !dbg !15
  %876 = extractelement <32 x float> %725, i64 13, !dbg !15
  %877 = extractelement <32 x float> %725, i64 14, !dbg !15
  %878 = extractelement <32 x float> %725, i64 15, !dbg !15
  %879 = select i1 %.not, float %876, float %875, !dbg !15
  %880 = bitcast float %879 to i32, !dbg !15
  %881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %880), !dbg !15
  %882 = bitcast i32 %881 to float, !dbg !15
  %883 = select i1 %.not, float %875, float %882, !dbg !15
  %884 = select i1 %.not, float %882, float %876, !dbg !15
  %885 = select i1 %.not, float %878, float %877, !dbg !15
  %886 = bitcast float %885 to i32, !dbg !15
  %887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %886), !dbg !15
  %888 = bitcast i32 %887 to float, !dbg !15
  %889 = select i1 %.not, float %877, float %888, !dbg !15
  %890 = select i1 %.not, float %888, float %878, !dbg !15
  %891 = select i1 %.not314, float %889, float %883, !dbg !15
  %892 = bitcast float %891 to i32, !dbg !15
  %893 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %892), !dbg !15
  %894 = bitcast i32 %893 to float, !dbg !15
  %895 = select i1 %.not314, float %883, float %894, !dbg !15
  %896 = select i1 %.not314, float %894, float %889, !dbg !15
  %897 = select i1 %.not314, float %890, float %884, !dbg !15
  %898 = bitcast float %897 to i32, !dbg !15
  %899 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %898), !dbg !15
  %900 = bitcast i32 %899 to float, !dbg !15
  %901 = select i1 %.not314, float %884, float %900, !dbg !15
  %902 = select i1 %.not314, float %900, float %890, !dbg !15
  %compare.local.base1126 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 43) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1127 = or i32 %compare.local.base1126, 48, !dbg !26
  %compare.low1128 = icmp ugt i32 %compare.low.base1078, %compare.local1127, !dbg !26
  %compare.result1129 = select i1 %compare.high.equal1076, i1 %compare.low1128, i1 %compare.high.order1077, !dbg !26
  %903 = select i1 %compare.result1129, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %904 = fadd float %895, %903, !dbg !20
  %compare.local.base1130 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 44) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1131 = or i32 %compare.local.base1130, 49, !dbg !26
  %compare.low1132 = icmp ugt i32 %compare.low.base1078, %compare.local1131, !dbg !26
  %compare.result1133 = select i1 %compare.high.equal1076, i1 %compare.low1132, i1 %compare.high.order1077, !dbg !26
  %905 = select i1 %compare.result1133, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %906 = fadd float %901, %905, !dbg !20
  %compare.local.base1134 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 45) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1135 = or i32 %compare.local.base1134, 50, !dbg !26
  %compare.low1136 = icmp ugt i32 %compare.low.base1078, %compare.local1135, !dbg !26
  %compare.result1137 = select i1 %compare.high.equal1076, i1 %compare.low1136, i1 %compare.high.order1077, !dbg !26
  %907 = select i1 %compare.result1137, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %908 = fadd float %896, %907, !dbg !20
  %compare.local.base1138 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 46) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1139 = or i32 %compare.local.base1138, 51, !dbg !26
  %compare.low1140 = icmp ugt i32 %compare.low.base1078, %compare.local1139, !dbg !26
  %compare.result1141 = select i1 %compare.high.equal1076, i1 %compare.low1140, i1 %compare.high.order1077, !dbg !26
  %909 = select i1 %compare.result1141, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %910 = fadd float %902, %909, !dbg !20
  %911 = call float @__ocml_exp2_f32(float %904), !dbg !28
  %912 = fadd float %872, %911, !dbg !29
  %913 = insertelement <4 x float> poison, float %911, i64 0, !dbg !28
  %914 = call float @__ocml_exp2_f32(float %906), !dbg !28
  %915 = fadd float %912, %914, !dbg !29
  %916 = insertelement <4 x float> %913, float %914, i64 1, !dbg !28
  %917 = call float @__ocml_exp2_f32(float %908), !dbg !28
  %918 = fadd float %915, %917, !dbg !29
  %919 = insertelement <4 x float> %916, float %917, i64 2, !dbg !28
  %920 = call float @__ocml_exp2_f32(float %910), !dbg !28
  %921 = fadd float %918, %920, !dbg !29
  %922 = insertelement <4 x float> %919, float %920, i64 3, !dbg !28
  %math.result407 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %922) #8, !dbg !30, !frisk.register_lifetimes !1
  %923 = fptrunc <4 x float> %math.result407 to <4 x half>, !dbg !30
  %math.packed408 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %923) #8, !dbg !15, !frisk.register_lifetimes !1
  %924 = extractelement <32 x float> %725, i64 16, !dbg !15
  %925 = extractelement <32 x float> %725, i64 17, !dbg !15
  %926 = extractelement <32 x float> %725, i64 18, !dbg !15
  %927 = extractelement <32 x float> %725, i64 19, !dbg !15
  %928 = select i1 %.not, float %925, float %924, !dbg !15
  %929 = bitcast float %928 to i32, !dbg !15
  %930 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %929), !dbg !15
  %931 = bitcast i32 %930 to float, !dbg !15
  %932 = select i1 %.not, float %924, float %931, !dbg !15
  %933 = select i1 %.not, float %931, float %925, !dbg !15
  %934 = select i1 %.not, float %927, float %926, !dbg !15
  %935 = bitcast float %934 to i32, !dbg !15
  %936 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %935), !dbg !15
  %937 = bitcast i32 %936 to float, !dbg !15
  %938 = select i1 %.not, float %926, float %937, !dbg !15
  %939 = select i1 %.not, float %937, float %927, !dbg !15
  %940 = select i1 %.not314, float %938, float %932, !dbg !15
  %941 = bitcast float %940 to i32, !dbg !15
  %942 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %941), !dbg !15
  %943 = bitcast i32 %942 to float, !dbg !15
  %944 = select i1 %.not314, float %932, float %943, !dbg !15
  %945 = select i1 %.not314, float %943, float %938, !dbg !15
  %946 = select i1 %.not314, float %939, float %933, !dbg !15
  %947 = bitcast float %946 to i32, !dbg !15
  %948 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %947), !dbg !15
  %949 = bitcast i32 %948 to float, !dbg !15
  %950 = select i1 %.not314, float %933, float %949, !dbg !15
  %951 = select i1 %.not314, float %949, float %939, !dbg !15
  %compare.local.base1142 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 47) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1143 = or i32 %compare.local.base1142, 64, !dbg !26
  %compare.low1144 = icmp ugt i32 %compare.low.base1078, %compare.local1143, !dbg !26
  %compare.result1145 = select i1 %compare.high.equal1076, i1 %compare.low1144, i1 %compare.high.order1077, !dbg !26
  %952 = select i1 %compare.result1145, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %953 = fadd float %944, %952, !dbg !20
  %compare.local.base1146 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 48) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1147 = or i32 %compare.local.base1146, 65, !dbg !26
  %compare.low1148 = icmp ugt i32 %compare.low.base1078, %compare.local1147, !dbg !26
  %compare.result1149 = select i1 %compare.high.equal1076, i1 %compare.low1148, i1 %compare.high.order1077, !dbg !26
  %954 = select i1 %compare.result1149, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %955 = fadd float %950, %954, !dbg !20
  %compare.local.base1150 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 49) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1151 = or i32 %compare.local.base1150, 66, !dbg !26
  %compare.low1152 = icmp ugt i32 %compare.low.base1078, %compare.local1151, !dbg !26
  %compare.result1153 = select i1 %compare.high.equal1076, i1 %compare.low1152, i1 %compare.high.order1077, !dbg !26
  %956 = select i1 %compare.result1153, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %957 = fadd float %945, %956, !dbg !20
  %compare.local.base1154 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 50) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1155 = or i32 %compare.local.base1154, 67, !dbg !26
  %compare.low1156 = icmp ugt i32 %compare.low.base1078, %compare.local1155, !dbg !26
  %compare.result1157 = select i1 %compare.high.equal1076, i1 %compare.low1156, i1 %compare.high.order1077, !dbg !26
  %958 = select i1 %compare.result1157, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %959 = fadd float %951, %958, !dbg !20
  %960 = call float @__ocml_exp2_f32(float %953), !dbg !28
  %961 = fadd float %921, %960, !dbg !29
  %962 = insertelement <4 x float> poison, float %960, i64 0, !dbg !28
  %963 = call float @__ocml_exp2_f32(float %955), !dbg !28
  %964 = fadd float %961, %963, !dbg !29
  %965 = insertelement <4 x float> %962, float %963, i64 1, !dbg !28
  %966 = call float @__ocml_exp2_f32(float %957), !dbg !28
  %967 = fadd float %964, %966, !dbg !29
  %968 = insertelement <4 x float> %965, float %966, i64 2, !dbg !28
  %969 = call float @__ocml_exp2_f32(float %959), !dbg !28
  %970 = fadd float %967, %969, !dbg !29
  %971 = insertelement <4 x float> %968, float %969, i64 3, !dbg !28
  %math.result409 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %971) #8, !dbg !30, !frisk.register_lifetimes !1
  %972 = fptrunc <4 x float> %math.result409 to <4 x half>, !dbg !30
  %math.packed410 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %972) #8, !dbg !15, !frisk.register_lifetimes !1
  %973 = extractelement <32 x float> %725, i64 20, !dbg !15
  %974 = extractelement <32 x float> %725, i64 21, !dbg !15
  %975 = extractelement <32 x float> %725, i64 22, !dbg !15
  %976 = extractelement <32 x float> %725, i64 23, !dbg !15
  %977 = select i1 %.not, float %974, float %973, !dbg !15
  %978 = bitcast float %977 to i32, !dbg !15
  %979 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %978), !dbg !15
  %980 = bitcast i32 %979 to float, !dbg !15
  %981 = select i1 %.not, float %973, float %980, !dbg !15
  %982 = select i1 %.not, float %980, float %974, !dbg !15
  %983 = select i1 %.not, float %976, float %975, !dbg !15
  %984 = bitcast float %983 to i32, !dbg !15
  %985 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %984), !dbg !15
  %986 = bitcast i32 %985 to float, !dbg !15
  %987 = select i1 %.not, float %975, float %986, !dbg !15
  %988 = select i1 %.not, float %986, float %976, !dbg !15
  %989 = select i1 %.not314, float %987, float %981, !dbg !15
  %990 = bitcast float %989 to i32, !dbg !15
  %991 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %990), !dbg !15
  %992 = bitcast i32 %991 to float, !dbg !15
  %993 = select i1 %.not314, float %981, float %992, !dbg !15
  %994 = select i1 %.not314, float %992, float %987, !dbg !15
  %995 = select i1 %.not314, float %988, float %982, !dbg !15
  %996 = bitcast float %995 to i32, !dbg !15
  %997 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %996), !dbg !15
  %998 = bitcast i32 %997 to float, !dbg !15
  %999 = select i1 %.not314, float %982, float %998, !dbg !15
  %1000 = select i1 %.not314, float %998, float %988, !dbg !15
  %compare.local.base1158 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 51) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1159 = or i32 %compare.local.base1158, 80, !dbg !26
  %compare.low1160 = icmp ugt i32 %compare.low.base1078, %compare.local1159, !dbg !26
  %compare.result1161 = select i1 %compare.high.equal1076, i1 %compare.low1160, i1 %compare.high.order1077, !dbg !26
  %1001 = select i1 %compare.result1161, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1002 = fadd float %993, %1001, !dbg !20
  %compare.local.base1162 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 52) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1163 = or i32 %compare.local.base1162, 81, !dbg !26
  %compare.low1164 = icmp ugt i32 %compare.low.base1078, %compare.local1163, !dbg !26
  %compare.result1165 = select i1 %compare.high.equal1076, i1 %compare.low1164, i1 %compare.high.order1077, !dbg !26
  %1003 = select i1 %compare.result1165, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1004 = fadd float %999, %1003, !dbg !20
  %compare.local.base1166 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 53) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1167 = or i32 %compare.local.base1166, 82, !dbg !26
  %compare.low1168 = icmp ugt i32 %compare.low.base1078, %compare.local1167, !dbg !26
  %compare.result1169 = select i1 %compare.high.equal1076, i1 %compare.low1168, i1 %compare.high.order1077, !dbg !26
  %1005 = select i1 %compare.result1169, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1006 = fadd float %994, %1005, !dbg !20
  %compare.local.base1170 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 54) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1171 = or i32 %compare.local.base1170, 83, !dbg !26
  %compare.low1172 = icmp ugt i32 %compare.low.base1078, %compare.local1171, !dbg !26
  %compare.result1173 = select i1 %compare.high.equal1076, i1 %compare.low1172, i1 %compare.high.order1077, !dbg !26
  %1007 = select i1 %compare.result1173, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1008 = fadd float %1000, %1007, !dbg !20
  %1009 = call float @__ocml_exp2_f32(float %1002), !dbg !28
  %1010 = fadd float %970, %1009, !dbg !29
  %1011 = insertelement <4 x float> poison, float %1009, i64 0, !dbg !28
  %1012 = call float @__ocml_exp2_f32(float %1004), !dbg !28
  %1013 = fadd float %1010, %1012, !dbg !29
  %1014 = insertelement <4 x float> %1011, float %1012, i64 1, !dbg !28
  %1015 = call float @__ocml_exp2_f32(float %1006), !dbg !28
  %1016 = fadd float %1013, %1015, !dbg !29
  %1017 = insertelement <4 x float> %1014, float %1015, i64 2, !dbg !28
  %1018 = call float @__ocml_exp2_f32(float %1008), !dbg !28
  %1019 = fadd float %1016, %1018, !dbg !29
  %1020 = insertelement <4 x float> %1017, float %1018, i64 3, !dbg !28
  %math.result411 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1020) #8, !dbg !30, !frisk.register_lifetimes !1
  %1021 = fptrunc <4 x float> %math.result411 to <4 x half>, !dbg !30
  %math.packed412 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1021) #8, !dbg !15, !frisk.register_lifetimes !1
  %1022 = extractelement <32 x float> %725, i64 24, !dbg !15
  %1023 = extractelement <32 x float> %725, i64 25, !dbg !15
  %1024 = extractelement <32 x float> %725, i64 26, !dbg !15
  %1025 = extractelement <32 x float> %725, i64 27, !dbg !15
  %1026 = select i1 %.not, float %1023, float %1022, !dbg !15
  %1027 = bitcast float %1026 to i32, !dbg !15
  %1028 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %1027), !dbg !15
  %1029 = bitcast i32 %1028 to float, !dbg !15
  %1030 = select i1 %.not, float %1022, float %1029, !dbg !15
  %1031 = select i1 %.not, float %1029, float %1023, !dbg !15
  %1032 = select i1 %.not, float %1025, float %1024, !dbg !15
  %1033 = bitcast float %1032 to i32, !dbg !15
  %1034 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %1033), !dbg !15
  %1035 = bitcast i32 %1034 to float, !dbg !15
  %1036 = select i1 %.not, float %1024, float %1035, !dbg !15
  %1037 = select i1 %.not, float %1035, float %1025, !dbg !15
  %1038 = select i1 %.not314, float %1036, float %1030, !dbg !15
  %1039 = bitcast float %1038 to i32, !dbg !15
  %1040 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %1039), !dbg !15
  %1041 = bitcast i32 %1040 to float, !dbg !15
  %1042 = select i1 %.not314, float %1030, float %1041, !dbg !15
  %1043 = select i1 %.not314, float %1041, float %1036, !dbg !15
  %1044 = select i1 %.not314, float %1037, float %1031, !dbg !15
  %1045 = bitcast float %1044 to i32, !dbg !15
  %1046 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %1045), !dbg !15
  %1047 = bitcast i32 %1046 to float, !dbg !15
  %1048 = select i1 %.not314, float %1031, float %1047, !dbg !15
  %1049 = select i1 %.not314, float %1047, float %1037, !dbg !15
  %compare.local.base1174 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 55) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1175 = or i32 %compare.local.base1174, 96, !dbg !26
  %compare.low1176 = icmp ugt i32 %compare.low.base1078, %compare.local1175, !dbg !26
  %compare.result1177 = select i1 %compare.high.equal1076, i1 %compare.low1176, i1 %compare.high.order1077, !dbg !26
  %1050 = select i1 %compare.result1177, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1051 = fadd float %1042, %1050, !dbg !20
  %compare.local.base1178 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 56) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1179 = or i32 %compare.local.base1178, 97, !dbg !26
  %compare.low1180 = icmp ugt i32 %compare.low.base1078, %compare.local1179, !dbg !26
  %compare.result1181 = select i1 %compare.high.equal1076, i1 %compare.low1180, i1 %compare.high.order1077, !dbg !26
  %1052 = select i1 %compare.result1181, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1053 = fadd float %1048, %1052, !dbg !20
  %compare.local.base1182 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 57) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1183 = or i32 %compare.local.base1182, 98, !dbg !26
  %compare.low1184 = icmp ugt i32 %compare.low.base1078, %compare.local1183, !dbg !26
  %compare.result1185 = select i1 %compare.high.equal1076, i1 %compare.low1184, i1 %compare.high.order1077, !dbg !26
  %1054 = select i1 %compare.result1185, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1055 = fadd float %1043, %1054, !dbg !20
  %compare.local.base1186 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 58) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1187 = or i32 %compare.local.base1186, 99, !dbg !26
  %compare.low1188 = icmp ugt i32 %compare.low.base1078, %compare.local1187, !dbg !26
  %compare.result1189 = select i1 %compare.high.equal1076, i1 %compare.low1188, i1 %compare.high.order1077, !dbg !26
  %1056 = select i1 %compare.result1189, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1057 = fadd float %1049, %1056, !dbg !20
  %1058 = call float @__ocml_exp2_f32(float %1051), !dbg !28
  %1059 = fadd float %1019, %1058, !dbg !29
  %1060 = insertelement <4 x float> poison, float %1058, i64 0, !dbg !28
  %1061 = call float @__ocml_exp2_f32(float %1053), !dbg !28
  %1062 = fadd float %1059, %1061, !dbg !29
  %1063 = insertelement <4 x float> %1060, float %1061, i64 1, !dbg !28
  %1064 = call float @__ocml_exp2_f32(float %1055), !dbg !28
  %1065 = fadd float %1062, %1064, !dbg !29
  %1066 = insertelement <4 x float> %1063, float %1064, i64 2, !dbg !28
  %1067 = call float @__ocml_exp2_f32(float %1057), !dbg !28
  %1068 = fadd float %1065, %1067, !dbg !29
  %1069 = insertelement <4 x float> %1066, float %1067, i64 3, !dbg !28
  %math.result413 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1069) #8, !dbg !30, !frisk.register_lifetimes !1
  %1070 = fptrunc <4 x float> %math.result413 to <4 x half>, !dbg !30
  %math.packed414 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1070) #8, !dbg !15, !frisk.register_lifetimes !1
  %1071 = extractelement <32 x float> %725, i64 28, !dbg !15
  %1072 = extractelement <32 x float> %725, i64 29, !dbg !15
  %1073 = extractelement <32 x float> %725, i64 30, !dbg !15
  %1074 = extractelement <32 x float> %725, i64 31, !dbg !15
  %1075 = select i1 %.not, float %1072, float %1071, !dbg !15
  %1076 = bitcast float %1075 to i32, !dbg !15
  %1077 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %1076), !dbg !15
  %1078 = bitcast i32 %1077 to float, !dbg !15
  %1079 = select i1 %.not, float %1071, float %1078, !dbg !15
  %1080 = select i1 %.not, float %1078, float %1072, !dbg !15
  %1081 = select i1 %.not, float %1074, float %1073, !dbg !15
  %1082 = bitcast float %1081 to i32, !dbg !15
  %1083 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %1082), !dbg !15
  %1084 = bitcast i32 %1083 to float, !dbg !15
  %1085 = select i1 %.not, float %1073, float %1084, !dbg !15
  %1086 = select i1 %.not, float %1084, float %1074, !dbg !15
  %1087 = select i1 %.not314, float %1085, float %1079, !dbg !15
  %1088 = bitcast float %1087 to i32, !dbg !15
  %1089 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %1088), !dbg !15
  %1090 = bitcast i32 %1089 to float, !dbg !15
  %1091 = select i1 %.not314, float %1079, float %1090, !dbg !15
  %1092 = select i1 %.not314, float %1090, float %1085, !dbg !15
  %1093 = select i1 %.not314, float %1086, float %1080, !dbg !15
  %1094 = bitcast float %1093 to i32, !dbg !15
  %1095 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %1094), !dbg !15
  %1096 = bitcast i32 %1095 to float, !dbg !15
  %1097 = select i1 %.not314, float %1080, float %1096, !dbg !15
  %1098 = select i1 %.not314, float %1096, float %1086, !dbg !15
  %compare.local.base1190 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 59) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1191 = or i32 %compare.local.base1190, 112, !dbg !26
  %compare.low1192 = icmp ugt i32 %compare.low.base1078, %compare.local1191, !dbg !26
  %compare.result1193 = select i1 %compare.high.equal1076, i1 %compare.low1192, i1 %compare.high.order1077, !dbg !26
  %1099 = select i1 %compare.result1193, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1100 = fadd float %1091, %1099, !dbg !20
  %compare.local.base1194 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 60) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1195 = or i32 %compare.local.base1194, 113, !dbg !26
  %compare.low1196 = icmp ugt i32 %compare.low.base1078, %compare.local1195, !dbg !26
  %compare.result1197 = select i1 %compare.high.equal1076, i1 %compare.low1196, i1 %compare.high.order1077, !dbg !26
  %1101 = select i1 %compare.result1197, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1102 = fadd float %1097, %1101, !dbg !20
  %compare.local.base1198 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 61) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1199 = or i32 %compare.local.base1198, 114, !dbg !26
  %compare.low1200 = icmp ugt i32 %compare.low.base1078, %compare.local1199, !dbg !26
  %compare.result1201 = select i1 %compare.high.equal1076, i1 %compare.low1200, i1 %compare.high.order1077, !dbg !26
  %1103 = select i1 %compare.result1201, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1104 = fadd float %1092, %1103, !dbg !20
  %compare.local.base1202 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1079, i32 62) #8, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1203 = or i32 %compare.local.base1202, 115, !dbg !26
  %compare.low1204 = icmp ugt i32 %compare.low.base1078, %compare.local1203, !dbg !26
  %compare.result1205 = select i1 %compare.high.equal1076, i1 %compare.low1204, i1 %compare.high.order1077, !dbg !26
  %1105 = select i1 %compare.result1205, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %1106 = fadd float %1098, %1105, !dbg !20
  %1107 = call float @__ocml_exp2_f32(float %1100), !dbg !28
  %1108 = fadd float %1068, %1107, !dbg !29
  %1109 = insertelement <4 x float> poison, float %1107, i64 0, !dbg !28
  %1110 = call float @__ocml_exp2_f32(float %1102), !dbg !28
  %1111 = fadd float %1108, %1110, !dbg !29
  %1112 = insertelement <4 x float> %1109, float %1110, i64 1, !dbg !28
  %1113 = call float @__ocml_exp2_f32(float %1104), !dbg !28
  %1114 = fadd float %1111, %1113, !dbg !29
  %1115 = insertelement <4 x float> %1112, float %1113, i64 2, !dbg !28
  %1116 = call float @__ocml_exp2_f32(float %1106), !dbg !28
  %1117 = fadd float %1114, %1116, !dbg !29
  %1118 = insertelement <4 x float> %1115, float %1116, i64 3, !dbg !28
  %math.result415 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1118) #8, !dbg !30, !frisk.register_lifetimes !1
  %1119 = fptrunc <4 x float> %math.result415 to <4 x half>, !dbg !30
  %math.packed416 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1119) #8, !dbg !29, !frisk.register_lifetimes !1
  %1120 = bitcast float %722 to i32, !dbg !29
  %1121 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %1120), !dbg !29
  %1122 = bitcast i32 %1121 to float, !dbg !29
  %1123 = fadd float %722, %1122, !dbg !29
  %1124 = bitcast float %1123 to i32, !dbg !29
  %1125 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %1124), !dbg !29
  %1126 = bitcast i32 %1125 to float, !dbg !29
  %1127 = fadd float %1123, %1126, !dbg !29
  %1128 = extractvalue [2 x <1 x float>] %42, 0, !dbg !29
  %1129 = extractelement <1 x float> %1128, i64 0, !dbg !29
  %1130 = fadd float %1127, %1129, !dbg !29
  %1131 = insertelement <1 x float> poison, float %1130, i64 0, !dbg !18
  %1132 = insertvalue [2 x <1 x float>] poison, <1 x float> %1131, 0, !dbg !18
  %1133 = bitcast float %1117 to i32, !dbg !29
  %1134 = call i32 @llvm.amdgcn.ds.bpermute(i32 %330, i32 %1133), !dbg !29
  %1135 = bitcast i32 %1134 to float, !dbg !29
  %1136 = fadd float %1117, %1135, !dbg !29
  %1137 = bitcast float %1136 to i32, !dbg !29
  %1138 = call i32 @llvm.amdgcn.ds.bpermute(i32 %346, i32 %1137), !dbg !29
  %1139 = bitcast i32 %1138 to float, !dbg !29
  %1140 = fadd float %1136, %1139, !dbg !29
  %1141 = extractvalue [2 x <1 x float>] %42, 1, !dbg !29
  %1142 = extractelement <1 x float> %1141, i64 0, !dbg !29
  %1143 = fadd float %1140, %1142, !dbg !29
  %1144 = insertelement <1 x float> poison, float %1143, i64 0, !dbg !18
  %1145 = insertvalue [2 x <1 x float>] %1132, <1 x float> %1144, 1, !dbg !18
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.local853 = shl nuw nsw i32 %address.thread741, 3, !dbg !16
  %address.local854 = and i32 %address.local853, 504, !dbg !16
  %address.local855 = zext nneg i32 %address.local854 to i64, !dbg !16
  %address.local856 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local855, !dbg !16
  %1146 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local856, i32 0) #0, !dbg !16
  %1147 = bitcast <4 x i32> %1146 to <8 x half>, !dbg !16
  %address.local857 = shl nuw nsw i32 %address.thread741, 3, !dbg !16
  %address.local858 = and i32 %address.local857, 504, !dbg !16
  %address.local859 = zext nneg i32 %address.local858 to i64, !dbg !16
  %address.local860 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local859, !dbg !16
  %address.local861 = getelementptr i8, ptr addrspace(3) %address.local860, i64 1024, !dbg !16
  %1148 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local861, i32 0) #0, !dbg !16
  %1149 = bitcast <4 x i32> %1148 to <8 x half>, !dbg !16
  %address.local862 = shl nuw nsw i32 %address.thread741, 3, !dbg !16
  %address.local863 = and i32 %address.local862, 504, !dbg !16
  %address.local864 = zext nneg i32 %address.local863 to i64, !dbg !16
  %address.local865 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local864, !dbg !16
  %address.local866 = getelementptr i8, ptr addrspace(3) %address.local865, i64 2048, !dbg !16
  %1150 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local866, i32 0) #0, !dbg !16
  %1151 = bitcast <4 x i32> %1150 to <8 x half>, !dbg !16
  %address.local867 = shl nuw nsw i32 %address.thread741, 3, !dbg !16
  %address.local868 = or i32 %address.local867, 1536, !dbg !16
  %address.local869 = zext nneg i32 %address.local868 to i64, !dbg !16
  %address.local870 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local869, !dbg !16
  %1152 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local870, i32 0) #0, !dbg !16
  %1153 = bitcast <4 x i32> %1152 to <8 x half>, !dbg !16
  %address.local871 = shl nuw nsw i32 %address.thread741, 3, !dbg !16
  %address.local872 = and i32 %address.local871, 504, !dbg !16
  %address.local873 = zext nneg i32 %address.local872 to i64, !dbg !16
  %address.local874 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local873, !dbg !16
  %address.local875 = getelementptr i8, ptr addrspace(3) %address.local874, i64 4096, !dbg !16
  %1154 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local875, i32 0) #0, !dbg !16
  %1155 = bitcast <4 x i32> %1154 to <8 x half>, !dbg !16
  %address.local876 = shl nuw nsw i32 %address.thread741, 3, !dbg !16
  %address.local877 = and i32 %address.local876, 504, !dbg !16
  %address.local878 = zext nneg i32 %address.local877 to i64, !dbg !16
  %address.local879 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local878, !dbg !16
  %address.local880 = getelementptr i8, ptr addrspace(3) %address.local879, i64 5120, !dbg !16
  %1156 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local880, i32 0) #0, !dbg !16
  %1157 = bitcast <4 x i32> %1156 to <8 x half>, !dbg !16
  %address.local881 = shl nuw nsw i32 %address.thread741, 3, !dbg !16
  %address.local882 = and i32 %address.local881, 504, !dbg !16
  %address.local883 = zext nneg i32 %address.local882 to i64, !dbg !16
  %address.local884 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local883, !dbg !16
  %address.local885 = getelementptr i8, ptr addrspace(3) %address.local884, i64 6144, !dbg !16
  %1158 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local885, i32 0) #0, !dbg !16
  %1159 = bitcast <4 x i32> %1158 to <8 x half>, !dbg !16
  %address.local886 = shl nuw nsw i32 %address.thread741, 3, !dbg !16
  %address.local887 = or i32 %address.local886, 3584, !dbg !16
  %address.local888 = zext nneg i32 %address.local887 to i64, !dbg !16
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local888, !dbg !16
  %1160 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local889, i32 0) #0, !dbg !16
  %1161 = bitcast <4 x i32> %1160 to <8 x half>, !dbg !16
  %1162 = shufflevector <8 x half> %1147, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1162, <4 x float> zeroinitializer) #8, !dbg !16
  %1164 = shufflevector <8 x half> %1149, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %1164, <4 x float> %1163) #8, !dbg !16
  %1166 = shufflevector <8 x half> %1151, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %1166, <4 x float> %1165) #8, !dbg !16
  %1168 = shufflevector <8 x half> %1153, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %1168, <4 x float> %1167) #8, !dbg !16
  %1170 = shufflevector <8 x half> %1155, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %1170, <4 x float> %1169) #8, !dbg !16
  %1172 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %1172, <4 x float> %1171) #8, !dbg !16
  %1174 = shufflevector <8 x half> %1159, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %1174, <4 x float> %1173) #8, !dbg !16
  %1176 = shufflevector <8 x half> %1161, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %1176, <4 x float> %1175) #8, !dbg !16
  %1178 = extractvalue [2 x <8 x float>] %41, 0, !dbg !31
  %1179 = shufflevector <8 x float> %1178, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1180 = fadd <4 x float> %1179, %1177, !dbg !31
  %1181 = shufflevector <8 x half> %1147, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1181, <4 x float> zeroinitializer) #8, !dbg !16
  %1183 = shufflevector <8 x half> %1149, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %1183, <4 x float> %1182) #8, !dbg !16
  %1185 = shufflevector <8 x half> %1151, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %1185, <4 x float> %1184) #8, !dbg !16
  %1187 = shufflevector <8 x half> %1153, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %1187, <4 x float> %1186) #8, !dbg !16
  %1189 = shufflevector <8 x half> %1155, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %1189, <4 x float> %1188) #8, !dbg !16
  %1191 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %1191, <4 x float> %1190) #8, !dbg !16
  %1193 = shufflevector <8 x half> %1159, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %1193, <4 x float> %1192) #8, !dbg !16
  %1195 = shufflevector <8 x half> %1161, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %1195, <4 x float> %1194) #8, !dbg !16
  %1197 = shufflevector <8 x float> %1178, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1198 = fadd <4 x float> %1197, %1196, !dbg !31
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %1162, <4 x float> zeroinitializer) #8, !dbg !16
  %1200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %1164, <4 x float> %1199) #8, !dbg !16
  %1201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %1166, <4 x float> %1200) #8, !dbg !16
  %1202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %1168, <4 x float> %1201) #8, !dbg !16
  %1203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %1170, <4 x float> %1202) #8, !dbg !16
  %1204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %1172, <4 x float> %1203) #8, !dbg !16
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %1174, <4 x float> %1204) #8, !dbg !16
  %1206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %1176, <4 x float> %1205) #8, !dbg !16
  %1207 = extractvalue [2 x <8 x float>] %41, 1, !dbg !31
  %1208 = shufflevector <8 x float> %1207, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1209 = fadd <4 x float> %1208, %1206, !dbg !31
  %1210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %1181, <4 x float> zeroinitializer) #8, !dbg !16
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %1183, <4 x float> %1210) #8, !dbg !16
  %1212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %1185, <4 x float> %1211) #8, !dbg !16
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %1187, <4 x float> %1212) #8, !dbg !16
  %1214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %1189, <4 x float> %1213) #8, !dbg !16
  %1215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %1191, <4 x float> %1214) #8, !dbg !16
  %1216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %1193, <4 x float> %1215) #8, !dbg !16
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %1195, <4 x float> %1216) #8, !dbg !16
  %1218 = shufflevector <8 x float> %1207, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1219 = fadd <4 x float> %1218, %1217, !dbg !31
  %1220 = shufflevector <4 x float> %1180, <4 x float> %1198, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1221 = insertvalue [2 x <8 x float>] poison, <8 x float> %1220, 0, !dbg !18
  %1222 = shufflevector <4 x float> %1209, <4 x float> %1219, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1223 = insertvalue [2 x <8 x float>] %1221, <8 x float> %1222, 1, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1224 = add nuw nsw i64 %40, 1, !dbg !18
  br label %39, !dbg !18

1225:                                             ; preds = %39
  %1226 = extractvalue [2 x <8 x float>] %41, 0, !dbg !18
  %1227 = extractelement <8 x float> %1226, i64 0, !dbg !18
  %1228 = extractelement <8 x float> %1226, i64 1, !dbg !18
  %1229 = extractelement <8 x float> %1226, i64 2, !dbg !18
  %1230 = extractelement <8 x float> %1226, i64 3, !dbg !18
  %1231 = select i1 %.not, float %1228, float %1227, !dbg !18
  %1232 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !18
  %1233 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1232), !dbg !18
  %1234 = add nuw nsw i32 %1233, 64, !dbg !18
  %1235 = xor i32 %1233, 16, !dbg !18
  %1236 = and i32 %1234, 192, !dbg !18
  %1237 = icmp samesign ult i32 %1235, %1236, !dbg !18
  %1238 = select i1 %1237, i32 %1235, i32 %1233, !dbg !18
  %1239 = shl nuw nsw i32 %1238, 2, !dbg !18
  %1240 = bitcast float %1231 to i32, !dbg !18
  %1241 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1239, i32 %1240), !dbg !18
  %1242 = bitcast i32 %1241 to float, !dbg !18
  %1243 = select i1 %.not, float %1227, float %1242, !dbg !18
  %1244 = select i1 %.not, float %1242, float %1228, !dbg !18
  %1245 = select i1 %.not, float %1230, float %1229, !dbg !18
  %1246 = bitcast float %1245 to i32, !dbg !18
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1239, i32 %1246), !dbg !18
  %1248 = bitcast i32 %1247 to float, !dbg !18
  %1249 = select i1 %.not, float %1229, float %1248, !dbg !18
  %1250 = select i1 %.not, float %1248, float %1230, !dbg !18
  %1251 = select i1 %.not314, float %1249, float %1243, !dbg !18
  %1252 = xor i32 %1233, 32, !dbg !18
  %1253 = icmp samesign ult i32 %1252, %1236, !dbg !18
  %1254 = select i1 %1253, i32 %1252, i32 %1233, !dbg !18
  %1255 = shl nuw nsw i32 %1254, 2, !dbg !18
  %1256 = bitcast float %1251 to i32, !dbg !18
  %1257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1255, i32 %1256), !dbg !18
  %1258 = select i1 %.not314, float %1250, float %1244, !dbg !18
  %1259 = bitcast float %1258 to i32, !dbg !18
  %1260 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1255, i32 %1259), !dbg !18
  %1261 = extractelement <8 x float> %1226, i64 4, !dbg !18
  %1262 = extractelement <8 x float> %1226, i64 5, !dbg !18
  %1263 = extractelement <8 x float> %1226, i64 6, !dbg !18
  %1264 = extractelement <8 x float> %1226, i64 7, !dbg !18
  %1265 = select i1 %.not, float %1262, float %1261, !dbg !18
  %1266 = bitcast float %1265 to i32, !dbg !18
  %1267 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1239, i32 %1266), !dbg !18
  %1268 = bitcast i32 %1267 to float, !dbg !18
  %1269 = select i1 %.not, float %1261, float %1268, !dbg !18
  %1270 = select i1 %.not, float %1268, float %1262, !dbg !18
  %1271 = select i1 %.not, float %1264, float %1263, !dbg !18
  %1272 = bitcast float %1271 to i32, !dbg !18
  %1273 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1239, i32 %1272), !dbg !18
  %1274 = bitcast i32 %1273 to float, !dbg !18
  %1275 = select i1 %.not, float %1263, float %1274, !dbg !18
  %1276 = select i1 %.not, float %1274, float %1264, !dbg !18
  %1277 = select i1 %.not314, float %1275, float %1269, !dbg !18
  %1278 = bitcast float %1277 to i32, !dbg !18
  %1279 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1255, i32 %1278), !dbg !18
  %1280 = select i1 %.not314, float %1276, float %1270, !dbg !18
  %1281 = bitcast float %1280 to i32, !dbg !18
  %1282 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1255, i32 %1281), !dbg !18
  %1283 = extractvalue [2 x <8 x float>] %41, 1, !dbg !18
  %1284 = extractelement <8 x float> %1283, i64 0, !dbg !18
  %1285 = extractelement <8 x float> %1283, i64 1, !dbg !18
  %1286 = extractelement <8 x float> %1283, i64 2, !dbg !18
  %1287 = extractelement <8 x float> %1283, i64 3, !dbg !18
  %1288 = select i1 %.not, float %1285, float %1284, !dbg !18
  %1289 = bitcast float %1288 to i32, !dbg !18
  %1290 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1239, i32 %1289), !dbg !18
  %1291 = bitcast i32 %1290 to float, !dbg !18
  %1292 = select i1 %.not, float %1284, float %1291, !dbg !18
  %1293 = select i1 %.not, float %1291, float %1285, !dbg !18
  %1294 = select i1 %.not, float %1287, float %1286, !dbg !18
  %1295 = bitcast float %1294 to i32, !dbg !18
  %1296 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1239, i32 %1295), !dbg !18
  %1297 = bitcast i32 %1296 to float, !dbg !18
  %1298 = select i1 %.not, float %1286, float %1297, !dbg !18
  %1299 = select i1 %.not, float %1297, float %1287, !dbg !18
  %1300 = select i1 %.not314, float %1298, float %1292, !dbg !18
  %1301 = bitcast float %1300 to i32, !dbg !18
  %1302 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1255, i32 %1301), !dbg !18
  %1303 = select i1 %.not314, float %1299, float %1293, !dbg !18
  %1304 = bitcast float %1303 to i32, !dbg !18
  %1305 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1255, i32 %1304), !dbg !18
  %1306 = extractelement <8 x float> %1283, i64 4, !dbg !18
  %1307 = extractelement <8 x float> %1283, i64 5, !dbg !18
  %1308 = extractelement <8 x float> %1283, i64 6, !dbg !18
  %1309 = extractelement <8 x float> %1283, i64 7, !dbg !18
  %1310 = select i1 %.not, float %1307, float %1306, !dbg !18
  %1311 = bitcast float %1310 to i32, !dbg !18
  %1312 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1239, i32 %1311), !dbg !18
  %1313 = bitcast i32 %1312 to float, !dbg !18
  %1314 = select i1 %.not, float %1306, float %1313, !dbg !18
  %1315 = select i1 %.not, float %1313, float %1307, !dbg !18
  %1316 = select i1 %.not, float %1309, float %1308, !dbg !18
  %1317 = bitcast float %1316 to i32, !dbg !18
  %1318 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1239, i32 %1317), !dbg !18
  %1319 = bitcast i32 %1318 to float, !dbg !18
  %1320 = select i1 %.not, float %1308, float %1319, !dbg !18
  %1321 = select i1 %.not, float %1319, float %1309, !dbg !18
  %1322 = select i1 %.not314, float %1320, float %1314, !dbg !18
  %1323 = bitcast float %1322 to i32, !dbg !18
  %1324 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1255, i32 %1323), !dbg !18
  %1325 = select i1 %.not314, float %1321, float %1315, !dbg !18
  %1326 = bitcast float %1325 to i32, !dbg !18
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1255, i32 %1326), !dbg !18
  br i1 true, label %1328, label %1398, !dbg !17

1328:                                             ; preds = %1225
  %1329 = bitcast i32 %1324 to float, !dbg !18
  %1330 = select i1 %.not314, float %1314, float %1329, !dbg !18
  %1331 = insertelement <4 x float> poison, float %1330, i64 0, !dbg !24
  %1332 = bitcast i32 %1327 to float, !dbg !18
  %1333 = select i1 %.not314, float %1315, float %1332, !dbg !18
  %1334 = insertelement <4 x float> %1331, float %1333, i64 1, !dbg !24
  %1335 = select i1 %.not314, float %1329, float %1320, !dbg !18
  %1336 = insertelement <4 x float> %1334, float %1335, i64 2, !dbg !24
  %1337 = select i1 %.not314, float %1332, float %1321, !dbg !18
  %1338 = insertelement <4 x float> %1336, float %1337, i64 3, !dbg !24
  %1339 = extractvalue [2 x <1 x float>] %42, 1, !dbg !24
  %1340 = shufflevector <1 x float> %1339, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %1341 = fdiv <4 x float> %1338, %1340, !dbg !24
  %1342 = fptrunc <4 x float> %1341 to <4 x half>, !dbg !25
  %1343 = extractelement <4 x half> %1342, i64 3, !dbg !25
  %1344 = extractelement <4 x half> %1342, i64 2, !dbg !25
  %1345 = extractelement <4 x half> %1342, i64 1, !dbg !25
  %1346 = extractelement <4 x half> %1342, i64 0, !dbg !25
  %1347 = bitcast i32 %1302 to float, !dbg !18
  %1348 = select i1 %.not314, float %1292, float %1347, !dbg !18
  %1349 = insertelement <4 x float> poison, float %1348, i64 0, !dbg !24
  %1350 = bitcast i32 %1305 to float, !dbg !18
  %1351 = select i1 %.not314, float %1293, float %1350, !dbg !18
  %1352 = insertelement <4 x float> %1349, float %1351, i64 1, !dbg !24
  %1353 = select i1 %.not314, float %1347, float %1298, !dbg !18
  %1354 = insertelement <4 x float> %1352, float %1353, i64 2, !dbg !24
  %1355 = select i1 %.not314, float %1350, float %1299, !dbg !18
  %1356 = insertelement <4 x float> %1354, float %1355, i64 3, !dbg !24
  %1357 = fdiv <4 x float> %1356, %1340, !dbg !24
  %1358 = fptrunc <4 x float> %1357 to <4 x half>, !dbg !25
  %1359 = extractelement <4 x half> %1358, i64 3, !dbg !25
  %1360 = extractelement <4 x half> %1358, i64 2, !dbg !25
  %1361 = extractelement <4 x half> %1358, i64 1, !dbg !25
  %1362 = extractelement <4 x half> %1358, i64 0, !dbg !25
  %1363 = bitcast i32 %1279 to float, !dbg !18
  %1364 = select i1 %.not314, float %1269, float %1363, !dbg !18
  %1365 = insertelement <4 x float> poison, float %1364, i64 0, !dbg !24
  %1366 = bitcast i32 %1282 to float, !dbg !18
  %1367 = select i1 %.not314, float %1270, float %1366, !dbg !18
  %1368 = insertelement <4 x float> %1365, float %1367, i64 1, !dbg !24
  %1369 = select i1 %.not314, float %1363, float %1275, !dbg !18
  %1370 = insertelement <4 x float> %1368, float %1369, i64 2, !dbg !24
  %1371 = select i1 %.not314, float %1366, float %1276, !dbg !18
  %1372 = insertelement <4 x float> %1370, float %1371, i64 3, !dbg !24
  %1373 = extractvalue [2 x <1 x float>] %42, 0, !dbg !24
  %1374 = shufflevector <1 x float> %1373, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %1375 = fdiv <4 x float> %1372, %1374, !dbg !24
  %1376 = fptrunc <4 x float> %1375 to <4 x half>, !dbg !25
  %1377 = extractelement <4 x half> %1376, i64 3, !dbg !25
  %1378 = extractelement <4 x half> %1376, i64 2, !dbg !25
  %1379 = extractelement <4 x half> %1376, i64 1, !dbg !25
  %1380 = extractelement <4 x half> %1376, i64 0, !dbg !25
  %1381 = bitcast i32 %1257 to float, !dbg !18
  %1382 = select i1 %.not314, float %1243, float %1381, !dbg !18
  %1383 = insertelement <4 x float> poison, float %1382, i64 0, !dbg !24
  %1384 = bitcast i32 %1260 to float, !dbg !18
  %1385 = select i1 %.not314, float %1244, float %1384, !dbg !18
  %1386 = insertelement <4 x float> %1383, float %1385, i64 1, !dbg !24
  %1387 = select i1 %.not314, float %1381, float %1249, !dbg !18
  %1388 = insertelement <4 x float> %1386, float %1387, i64 2, !dbg !24
  %1389 = select i1 %.not314, float %1384, float %1250, !dbg !18
  %1390 = insertelement <4 x float> %1388, float %1389, i64 3, !dbg !24
  %1391 = fdiv <4 x float> %1390, %1374, !dbg !24
  %1392 = fptrunc <4 x float> %1391 to <4 x half>, !dbg !25
  %1393 = extractelement <4 x half> %1392, i64 3, !dbg !25
  %1394 = extractelement <4 x half> %1392, i64 2, !dbg !25
  %1395 = extractelement <4 x half> %1392, i64 1, !dbg !25
  %1396 = extractelement <4 x half> %1392, i64 0, !dbg !25
  %1397 = shl nuw nsw i64 %15, 19, !dbg !17
  %address.thread890 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !17, !range !35, !frisk.register_lifetimes !1
  %address.local891 = zext nneg i32 %address.thread890 to i64, !dbg !14
  %address.local892 = add nsw i64 %16, %address.local891, !dbg !17
  %address.local893 = add nsw i64 %address.local892, %17, !dbg !17
  %address.local894 = lshr i32 %address.thread890, 2, !dbg !20
  %address.local895 = and i32 %address.local894, 48, !dbg !20
  %address.local896 = zext nneg i32 %address.local895 to i64, !dbg !20
  %address.local897 = add nsw i64 %address.local893, %address.local896, !dbg !17
  %address.local898 = and i32 %address.thread890, 240, !dbg !20
  %address.local899 = sub nsw i32 0, %address.local898, !dbg !20
  %address.local900 = sext i32 %address.local899 to i64, !dbg !20
  %address.local901 = add nsw i64 %address.local897, %address.local900, !dbg !17
  %address.local902 = shl nuw nsw i64 %address.local901, 7, !dbg !17
  %address.local903 = add nuw nsw i64 %1397, %address.local902, !dbg !17
  %address.local904 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local903, !dbg !17
  %address.local905 = lshr i32 %address.thread890, 2, !dbg !20
  %address.local906 = and i32 %address.local905, 12, !dbg !20
  %address.local907 = zext nneg i32 %address.local906 to i64, !dbg !20
  %address.local908 = or disjoint i64 %38, %address.local907, !dbg !17
  %address.local909 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local904, i64 %address.local908, !dbg !17
  store half %1396, ptr addrspace(1) %address.local909, align 2, !dbg !17
  %address.local910 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local903, !dbg !17
  %address.local911 = or disjoint i64 %address.local908, 1, !dbg !17
  %address.local912 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local910, i64 %address.local911, !dbg !17
  store half %1395, ptr addrspace(1) %address.local912, align 2, !dbg !17
  %address.local913 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local903, !dbg !17
  %address.local914 = or disjoint i64 %address.local908, 2, !dbg !17
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local913, i64 %address.local914, !dbg !17
  store half %1394, ptr addrspace(1) %address.local915, align 2, !dbg !17
  %address.local916 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local903, !dbg !17
  %address.local917 = or disjoint i64 %address.local908, 3, !dbg !17
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local917, !dbg !17
  store half %1393, ptr addrspace(1) %address.local918, align 2, !dbg !17
  %address.local919 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local903, !dbg !17
  %address.local920 = or disjoint i64 %address.local908, 16, !dbg !17
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local919, i64 %address.local920, !dbg !17
  store half %1380, ptr addrspace(1) %address.local921, align 2, !dbg !17
  %address.local922 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local903, !dbg !17
  %address.local923 = or disjoint i64 %address.local908, 17, !dbg !17
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local923, !dbg !17
  store half %1379, ptr addrspace(1) %address.local924, align 2, !dbg !17
  %address.local925 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local903, !dbg !17
  %address.local926 = or disjoint i64 %address.local908, 18, !dbg !17
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local926, !dbg !17
  store half %1378, ptr addrspace(1) %address.local927, align 2, !dbg !17
  %address.local928 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local903, !dbg !17
  %address.local929 = or disjoint i64 %address.local908, 19, !dbg !17
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local928, i64 %address.local929, !dbg !17
  store half %1377, ptr addrspace(1) %address.local930, align 2, !dbg !17
  %address.local931 = shl nsw i64 %address.local901, 7, !dbg !17
  %address.local932 = add i64 %address.local931, 8192, !dbg !17
  %address.local933 = add nuw nsw i64 %1397, %address.local932, !dbg !17
  %address.local934 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local933, !dbg !17
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local934, i64 %address.local908, !dbg !17
  store half %1362, ptr addrspace(1) %address.local935, align 2, !dbg !17
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local933, !dbg !17
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local936, i64 %address.local911, !dbg !17
  store half %1361, ptr addrspace(1) %address.local937, align 2, !dbg !17
  %address.local938 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local933, !dbg !17
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local938, i64 %address.local914, !dbg !17
  store half %1360, ptr addrspace(1) %address.local939, align 2, !dbg !17
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local933, !dbg !17
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local940, i64 %address.local917, !dbg !17
  store half %1359, ptr addrspace(1) %address.local941, align 2, !dbg !17
  %address.local942 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local933, !dbg !17
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local942, i64 %address.local920, !dbg !17
  store half %1346, ptr addrspace(1) %address.local943, align 2, !dbg !17
  %address.local944 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local933, !dbg !17
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local923, !dbg !17
  store half %1345, ptr addrspace(1) %address.local945, align 2, !dbg !17
  %address.local946 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local933, !dbg !17
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local946, i64 %address.local926, !dbg !17
  store half %1344, ptr addrspace(1) %address.local947, align 2, !dbg !17
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local933, !dbg !17
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local948, i64 %address.local929, !dbg !17
  store half %1343, ptr addrspace(1) %address.local949, align 2, !dbg !17
  br label %1398, !dbg !17

1398:                                             ; preds = %1328, %1225
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %1399 = add nuw nsw i64 %35, 1, !dbg !22
  br label %34, !dbg !22

1400:                                             ; preds = %34
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

!0 = !DIFile(filename: "test_input_spidD.mlir", directory: "3rd/deepgengraph/test")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!6 = !DILocation(line: 67, column: 22, scope: !4)
!7 = !DILocation(line: 69, column: 26, scope: !4)
!11 = !DILocation(line: 46, column: 17, scope: !4)
!13 = !DILocation(line: 66, column: 22, scope: !4)
!14 = !DILocation(line: 68, column: 24, scope: !4)
!15 = !DILocation(line: 59, column: 17, scope: !4)
!16 = !DILocation(line: 93, column: 23, scope: !4)
!17 = !DILocation(line: 99, column: 9, scope: !4)
!18 = !DILocation(line: 56, column: 24, scope: !4)
!19 = !DILocation(line: 92, column: 20, scope: !4)
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
