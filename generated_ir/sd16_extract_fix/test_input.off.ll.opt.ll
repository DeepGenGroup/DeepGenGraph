; ModuleID = 'generated_ir/sd16_extract_fix/test_input.off.ll'
source_filename = "3rd/deepgengraph/test/test_input.mlir"

@async_lds_staging_0 = private addrspace(3) global [4096 x half] undef, align 16
@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float) local_unnamed_addr

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %.lobit = ashr i32 %5, 31, !dbg !8
  %9 = xor i32 %.lobit, %5, !dbg !8
  %10 = sdiv i32 %9, 64, !dbg !8
  %11 = xor i32 %10, %.lobit, !dbg !8
  %12 = sext i32 %11 to i64, !dbg !8
  %13 = add nsw i64 %12, %8, !dbg !8
  %14 = shl nsw i64 %6, 6, !dbg !8
  %15 = shl nsw i64 %12, 12, !dbg !8
  %16 = sub nsw i64 %14, %15, !dbg !8
  %17 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %18 = zext nneg i32 %17 to i64, !dbg !8
  %19 = shl nsw i64 %13, 20, !dbg !8
  %.idx = and i64 %19, 32505856, !dbg !8
  %20 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !8
  %.idx565 = shl nsw i64 %16, 8, !dbg !8
  %21 = getelementptr i8, ptr addrspace(1) %20, i64 %.idx565, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %22 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %23 = and i32 %22, 384, !dbg !8
  %address.local578 = zext nneg i32 %23 to i64, !dbg !8
  %address.local579 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local578, !dbg !8
  %address.local582 = lshr i32 %address.thread, 5, !dbg !8
  %address.local583 = and i32 %address.local582, 4, !dbg !8
  %address.local585 = and i32 %22, 2, !dbg !8
  %address.local586 = or disjoint i32 %address.local583, %address.local585, !dbg !8
  %address.local587 = shl nuw nsw i32 %address.thread, 6, !dbg !8
  %address.local588 = and i32 %address.local587, 8064, !dbg !8
  %address.local589 = or disjoint i32 %address.local586, %address.local588, !dbg !8
  %address.local590 = zext nneg i32 %address.local589 to i64, !dbg !8
  %address.local591 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local590, !dbg !8
  %dma.source.bits = ptrtoint ptr addrspace(1) %address.local591 to i64, !dbg !8
  %dma.source.opaque = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #9, !dbg !8
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1), !dbg !8
  %address.local595 = getelementptr i8, ptr addrspace(3) %address.local579, i64 1024, !dbg !8
  %address.local608 = getelementptr i8, ptr addrspace(1) %address.local591, i64 16, !dbg !8
  %dma.source.bits1529 = ptrtoint ptr addrspace(1) %address.local608 to i64, !dbg !8
  %dma.source.opaque1530 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1529) #9, !dbg !8
  %dma.source.ptr1531 = inttoptr i64 %dma.source.opaque1530 to ptr addrspace(1), !dbg !8
  %address.local612 = getelementptr i8, ptr addrspace(3) %address.local579, i64 2048, !dbg !8
  %address.local625 = getelementptr i8, ptr addrspace(1) %address.local591, i64 32, !dbg !8
  %dma.source.bits1532 = ptrtoint ptr addrspace(1) %address.local625 to i64, !dbg !8
  %dma.source.opaque1533 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1532) #9, !dbg !8
  %dma.source.ptr1534 = inttoptr i64 %dma.source.opaque1533 to ptr addrspace(1), !dbg !8
  %address.local629 = getelementptr i8, ptr addrspace(3) %address.local579, i64 3072, !dbg !8
  %address.local642 = getelementptr i8, ptr addrspace(1) %address.local591, i64 48, !dbg !8
  %dma.source.bits1535 = ptrtoint ptr addrspace(1) %address.local642 to i64, !dbg !8
  %dma.source.opaque1536 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1535) #9, !dbg !8
  %dma.source.ptr1537 = inttoptr i64 %dma.source.opaque1536 to ptr addrspace(1), !dbg !8
  %address.local646 = getelementptr i8, ptr addrspace(3) %address.local579, i64 4096, !dbg !8
  %address.local659 = getelementptr i8, ptr addrspace(1) %address.local591, i64 64, !dbg !8
  %dma.source.bits1538 = ptrtoint ptr addrspace(1) %address.local659 to i64, !dbg !8
  %dma.source.opaque1539 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1538) #9, !dbg !8
  %dma.source.ptr1540 = inttoptr i64 %dma.source.opaque1539 to ptr addrspace(1), !dbg !8
  %address.local663 = getelementptr i8, ptr addrspace(3) %address.local579, i64 5120, !dbg !8
  %address.local676 = getelementptr i8, ptr addrspace(1) %address.local591, i64 80, !dbg !8
  %dma.source.bits1541 = ptrtoint ptr addrspace(1) %address.local676 to i64, !dbg !8
  %dma.source.opaque1542 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1541) #9, !dbg !8
  %dma.source.ptr1543 = inttoptr i64 %dma.source.opaque1542 to ptr addrspace(1), !dbg !8
  %address.local680 = getelementptr i8, ptr addrspace(3) %address.local579, i64 6144, !dbg !8
  %address.local693 = getelementptr i8, ptr addrspace(1) %address.local591, i64 96, !dbg !8
  %dma.source.bits1544 = ptrtoint ptr addrspace(1) %address.local693 to i64, !dbg !8
  %dma.source.opaque1545 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1544) #9, !dbg !8
  %dma.source.ptr1546 = inttoptr i64 %dma.source.opaque1545 to ptr addrspace(1), !dbg !8
  %address.local697 = getelementptr i8, ptr addrspace(3) %address.local579, i64 7168, !dbg !8
  %address.local710 = getelementptr i8, ptr addrspace(1) %address.local591, i64 112, !dbg !8
  %dma.source.bits1547 = ptrtoint ptr addrspace(1) %address.local710 to i64, !dbg !8
  %dma.source.opaque1548 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1547) #9, !dbg !8
  %dma.source.ptr1549 = inttoptr i64 %dma.source.opaque1548 to ptr addrspace(1), !dbg !8
  %address.local714 = getelementptr i8, ptr addrspace(3) %address.local579, i64 8192, !dbg !8
  %address.local727 = getelementptr i8, ptr addrspace(1) %address.local591, i64 128, !dbg !8
  %dma.source.bits1550 = ptrtoint ptr addrspace(1) %address.local727 to i64, !dbg !8
  %dma.source.opaque1551 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1550) #9, !dbg !8
  %dma.source.ptr1552 = inttoptr i64 %dma.source.opaque1551 to ptr addrspace(1), !dbg !8
  %address.local731 = getelementptr i8, ptr addrspace(3) %address.local579, i64 9216, !dbg !8
  %address.local744 = getelementptr i8, ptr addrspace(1) %address.local591, i64 144, !dbg !8
  %dma.source.bits1553 = ptrtoint ptr addrspace(1) %address.local744 to i64, !dbg !8
  %dma.source.opaque1554 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1553) #9, !dbg !8
  %dma.source.ptr1555 = inttoptr i64 %dma.source.opaque1554 to ptr addrspace(1), !dbg !8
  %address.local748 = getelementptr i8, ptr addrspace(3) %address.local579, i64 10240, !dbg !8
  %address.local761 = getelementptr i8, ptr addrspace(1) %address.local591, i64 160, !dbg !8
  %dma.source.bits1556 = ptrtoint ptr addrspace(1) %address.local761 to i64, !dbg !8
  %dma.source.opaque1557 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1556) #9, !dbg !8
  %dma.source.ptr1558 = inttoptr i64 %dma.source.opaque1557 to ptr addrspace(1), !dbg !8
  %address.local765 = getelementptr i8, ptr addrspace(3) %address.local579, i64 11264, !dbg !8
  %address.local778 = getelementptr i8, ptr addrspace(1) %address.local591, i64 176, !dbg !8
  %dma.source.bits1559 = ptrtoint ptr addrspace(1) %address.local778 to i64, !dbg !8
  %dma.source.opaque1560 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1559) #9, !dbg !8
  %dma.source.ptr1561 = inttoptr i64 %dma.source.opaque1560 to ptr addrspace(1), !dbg !8
  %address.local782 = getelementptr i8, ptr addrspace(3) %address.local579, i64 12288, !dbg !8
  %address.local795 = getelementptr i8, ptr addrspace(1) %address.local591, i64 192, !dbg !8
  %dma.source.bits1562 = ptrtoint ptr addrspace(1) %address.local795 to i64, !dbg !8
  %dma.source.opaque1563 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1562) #9, !dbg !8
  %dma.source.ptr1564 = inttoptr i64 %dma.source.opaque1563 to ptr addrspace(1), !dbg !8
  %address.local799 = getelementptr i8, ptr addrspace(3) %address.local579, i64 13312, !dbg !8
  %address.local812 = getelementptr i8, ptr addrspace(1) %address.local591, i64 208, !dbg !8
  %dma.source.bits1565 = ptrtoint ptr addrspace(1) %address.local812 to i64, !dbg !8
  %dma.source.opaque1566 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1565) #9, !dbg !8
  %dma.source.ptr1567 = inttoptr i64 %dma.source.opaque1566 to ptr addrspace(1), !dbg !8
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local579, i64 14336, !dbg !8
  %address.local829 = getelementptr i8, ptr addrspace(1) %address.local591, i64 224, !dbg !8
  %dma.source.bits1568 = ptrtoint ptr addrspace(1) %address.local829 to i64, !dbg !8
  %dma.source.opaque1569 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1568) #9, !dbg !8
  %dma.source.ptr1570 = inttoptr i64 %dma.source.opaque1569 to ptr addrspace(1), !dbg !8
  %address.local833 = getelementptr i8, ptr addrspace(3) %address.local579, i64 15360, !dbg !8
  %24 = insertelement <16 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local579, i64 0, !dbg !8
  %25 = insertelement <16 x ptr addrspace(3)> %24, ptr addrspace(3) %address.local595, i64 1, !dbg !8
  %26 = insertelement <16 x ptr addrspace(3)> %25, ptr addrspace(3) %address.local612, i64 2, !dbg !8
  %27 = insertelement <16 x ptr addrspace(3)> %26, ptr addrspace(3) %address.local629, i64 3, !dbg !8
  %28 = insertelement <16 x ptr addrspace(3)> %27, ptr addrspace(3) %address.local646, i64 4, !dbg !8
  %29 = insertelement <16 x ptr addrspace(3)> %28, ptr addrspace(3) %address.local663, i64 5, !dbg !8
  %30 = insertelement <16 x ptr addrspace(3)> %29, ptr addrspace(3) %address.local680, i64 6, !dbg !8
  %31 = insertelement <16 x ptr addrspace(3)> %30, ptr addrspace(3) %address.local697, i64 7, !dbg !8
  %32 = insertelement <16 x ptr addrspace(3)> %31, ptr addrspace(3) %address.local714, i64 8, !dbg !8
  %33 = insertelement <16 x ptr addrspace(3)> %32, ptr addrspace(3) %address.local731, i64 9, !dbg !8
  %34 = insertelement <16 x ptr addrspace(3)> %33, ptr addrspace(3) %address.local748, i64 10, !dbg !8
  %35 = insertelement <16 x ptr addrspace(3)> %34, ptr addrspace(3) %address.local765, i64 11, !dbg !8
  %36 = insertelement <16 x ptr addrspace(3)> %35, ptr addrspace(3) %address.local782, i64 12, !dbg !8
  %37 = insertelement <16 x ptr addrspace(3)> %36, ptr addrspace(3) %address.local799, i64 13, !dbg !8
  %38 = insertelement <16 x ptr addrspace(3)> %37, ptr addrspace(3) %address.local816, i64 14, !dbg !8
  %39 = insertelement <16 x ptr addrspace(3)> %38, ptr addrspace(3) %address.local833, i64 15, !dbg !8
  %40 = ptrtoint <16 x ptr addrspace(3)> %39 to <16 x i64>, !dbg !8
  %41 = trunc <16 x i64> %40 to <16 x i32>, !dbg !8
  %42 = extractelement <16 x i32> %41, i64 0, !dbg !8
  %43 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %42), !dbg !8
  %44 = zext i32 %43 to i64, !dbg !8
  %45 = inttoptr i64 %44 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %45, i32 4, i32 0, i32 0), !dbg !8
  %46 = extractelement <16 x i32> %41, i64 1, !dbg !8
  %47 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %48 = zext i32 %47 to i64, !dbg !8
  %49 = inttoptr i64 %48 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1531, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !8
  %50 = extractelement <16 x i32> %41, i64 2, !dbg !8
  %51 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %52 = zext i32 %51 to i64, !dbg !8
  %53 = inttoptr i64 %52 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1534, ptr addrspace(3) %53, i32 4, i32 0, i32 0), !dbg !8
  %54 = extractelement <16 x i32> %41, i64 3, !dbg !8
  %55 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %54), !dbg !8
  %56 = zext i32 %55 to i64, !dbg !8
  %57 = inttoptr i64 %56 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1537, ptr addrspace(3) %57, i32 4, i32 0, i32 0), !dbg !8
  %58 = extractelement <16 x i32> %41, i64 4, !dbg !8
  %59 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !8
  %60 = zext i32 %59 to i64, !dbg !8
  %61 = inttoptr i64 %60 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1540, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !8
  %62 = extractelement <16 x i32> %41, i64 5, !dbg !8
  %63 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %62), !dbg !8
  %64 = zext i32 %63 to i64, !dbg !8
  %65 = inttoptr i64 %64 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1543, ptr addrspace(3) %65, i32 4, i32 0, i32 0), !dbg !8
  %66 = extractelement <16 x i32> %41, i64 6, !dbg !8
  %67 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %66), !dbg !8
  %68 = zext i32 %67 to i64, !dbg !8
  %69 = inttoptr i64 %68 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1546, ptr addrspace(3) %69, i32 4, i32 0, i32 0), !dbg !8
  %70 = extractelement <16 x i32> %41, i64 7, !dbg !8
  %71 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %70), !dbg !8
  %72 = zext i32 %71 to i64, !dbg !8
  %73 = inttoptr i64 %72 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1549, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !8
  %74 = extractelement <16 x i32> %41, i64 8, !dbg !8
  %75 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %74), !dbg !8
  %76 = zext i32 %75 to i64, !dbg !8
  %77 = inttoptr i64 %76 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1552, ptr addrspace(3) %77, i32 4, i32 0, i32 0), !dbg !8
  %78 = extractelement <16 x i32> %41, i64 9, !dbg !8
  %79 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %78), !dbg !8
  %80 = zext i32 %79 to i64, !dbg !8
  %81 = inttoptr i64 %80 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1555, ptr addrspace(3) %81, i32 4, i32 0, i32 0), !dbg !8
  %82 = extractelement <16 x i32> %41, i64 10, !dbg !8
  %83 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %82), !dbg !8
  %84 = zext i32 %83 to i64, !dbg !8
  %85 = inttoptr i64 %84 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1558, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !8
  %86 = extractelement <16 x i32> %41, i64 11, !dbg !8
  %87 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %86), !dbg !8
  %88 = zext i32 %87 to i64, !dbg !8
  %89 = inttoptr i64 %88 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1561, ptr addrspace(3) %89, i32 4, i32 0, i32 0), !dbg !8
  %90 = extractelement <16 x i32> %41, i64 12, !dbg !8
  %91 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %90), !dbg !8
  %92 = zext i32 %91 to i64, !dbg !8
  %93 = inttoptr i64 %92 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1564, ptr addrspace(3) %93, i32 4, i32 0, i32 0), !dbg !8
  %94 = extractelement <16 x i32> %41, i64 13, !dbg !8
  %95 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !8
  %96 = zext i32 %95 to i64, !dbg !8
  %97 = inttoptr i64 %96 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1567, ptr addrspace(3) %97, i32 4, i32 0, i32 0), !dbg !8
  %98 = extractelement <16 x i32> %41, i64 14, !dbg !8
  %99 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %98), !dbg !8
  %100 = zext i32 %99 to i64, !dbg !8
  %101 = inttoptr i64 %100 to ptr addrspace(3), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1570, ptr addrspace(3) %101, i32 4, i32 0, i32 0), !dbg !8
  %102 = extractelement <16 x i32> %41, i64 15, !dbg !8
  %103 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %102), !dbg !8
  %104 = zext i32 %103 to i64, !dbg !8
  %105 = inttoptr i64 %104 to ptr addrspace(3), !dbg !8
  %address.local846 = getelementptr i8, ptr addrspace(1) %address.local591, i64 240, !dbg !8
  %dma.source.bits1571 = ptrtoint ptr addrspace(1) %address.local846 to i64, !dbg !8
  %dma.source.opaque1572 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1571) #9, !dbg !8
  %dma.source.ptr1573 = inttoptr i64 %dma.source.opaque1572 to ptr addrspace(1), !dbg !8
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1573, ptr addrspace(3) %105, i32 4, i32 0, i32 0), !dbg !8
  tail call void @llvm.amdgcn.asyncmark(), !dbg !8
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local847 = and i32 %address.thread, 192, !dbg !10
  %address.local848 = shl nuw nsw i32 %address.thread, 2, !dbg !10
  %address.local849 = and i32 %address.local848, 60, !dbg !10
  %address.local850 = or disjoint i32 %address.local849, %address.local847, !dbg !10
  %address.local851 = shl nuw nsw i32 %address.thread, 4, !dbg !10
  %address.local852 = and i32 %address.local851, 768, !dbg !10
  %address.local853 = or disjoint i32 %address.local850, %address.local852, !dbg !10
  %address.local854 = zext nneg i32 %address.local853 to i64, !dbg !10
  %address.local855 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local854, !dbg !10
  %106 = load <4 x half>, ptr addrspace(3) %address.local855, align 8, !dbg !10
  %address.local865 = getelementptr i8, ptr addrspace(3) %address.local855, i64 2048, !dbg !10
  %107 = load <4 x half>, ptr addrspace(3) %address.local865, align 8, !dbg !10
  %address.local875 = getelementptr i8, ptr addrspace(3) %address.local855, i64 4096, !dbg !10
  %108 = load <4 x half>, ptr addrspace(3) %address.local875, align 8, !dbg !10
  %address.local885 = getelementptr i8, ptr addrspace(3) %address.local855, i64 6144, !dbg !10
  %109 = load <4 x half>, ptr addrspace(3) %address.local885, align 8, !dbg !10
  %110 = fmul <4 x half> %106, splat (half 1.275630e-01), !dbg !10
  %address.local895 = getelementptr i8, ptr addrspace(3) %address.local855, i64 8192, !dbg !10
  %111 = load <4 x half>, ptr addrspace(3) %address.local895, align 8, !dbg !10
  %112 = fmul <4 x half> %107, splat (half 1.275630e-01), !dbg !10
  %address.local905 = getelementptr i8, ptr addrspace(3) %address.local855, i64 10240, !dbg !10
  %113 = load <4 x half>, ptr addrspace(3) %address.local905, align 8, !dbg !10
  %114 = fmul <4 x half> %108, splat (half 1.275630e-01), !dbg !10
  %address.local915 = getelementptr i8, ptr addrspace(3) %address.local855, i64 12288, !dbg !10
  %115 = load <4 x half>, ptr addrspace(3) %address.local915, align 8, !dbg !10
  %116 = fmul <4 x half> %109, splat (half 1.275630e-01), !dbg !10
  %117 = fmul <4 x half> %111, splat (half 1.275630e-01), !dbg !10
  %118 = fmul <4 x half> %113, splat (half 1.275630e-01), !dbg !10
  %119 = fmul <4 x half> %115, splat (half 1.275630e-01), !dbg !10
  %address.local925 = getelementptr i8, ptr addrspace(3) %address.local855, i64 14336, !dbg !10
  %120 = load <4 x half>, ptr addrspace(3) %address.local925, align 8, !dbg !10
  %121 = fmul <4 x half> %120, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %122 = and i32 %17, 16, !dbg !12
  %.not = icmp eq i32 %122, 0, !dbg !12
  %123 = and i32 %17, 32, !dbg !12
  %.not566 = icmp eq i32 %123, 0, !dbg !12
  %124 = lshr i32 %17, 2, !dbg !13
  %125 = shl nsw i64 %6, 1, !dbg !11
  %126 = and i32 %124, 12, !dbg !13
  %127 = icmp sgt i32 %5, -1, !dbg !11
  br i1 %127, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !11

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre9 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre11 = add nuw nsw i32 %.pre9, 64, !dbg !11
  %.pre13 = xor i32 %.pre9, 16, !dbg !11
  %.pre15 = and i32 %.pre11, 192, !dbg !11
  %.pre17 = xor i32 %.pre9, 32, !dbg !11
  br label %._crit_edge, !dbg !11

.lr.ph:                                           ; preds = %4
  %128 = add nuw nsw i64 %18, 1, !dbg !13
  %129 = add nuw nsw i64 %128, %14, !dbg !13
  %130 = and i32 %17, 240, !dbg !13
  %131 = sub nsw i32 0, %130, !dbg !13
  %index.compact296 = sext i32 %131 to i64, !dbg !13
  %132 = add nsw i64 %129, %index.compact296, !dbg !13
  %133 = and i32 %124, 48, !dbg !13
  %index.compact297 = zext nneg i32 %133 to i64, !dbg !13
  %134 = add nsw i64 %132, %index.compact297, !dbg !14
  %135 = lshr i64 %134, 32
  %compare.high = trunc nuw i64 %135 to i32
  %compare.low.base = trunc i64 %134 to i32
  %.mask = shl nuw nsw i32 %17, 9, !dbg !11
  %136 = and i32 %.mask, 7680, !dbg !11
  %137 = zext nneg i32 %136 to i64, !dbg !11
  %invariant.gep = getelementptr inbounds nuw i8, ptr addrspace(3) @async_lds_staging_0, i64 %137
  br label %138, !dbg !11

138:                                              ; preds = %.lr.ph, %212
  %139 = phi <1 x float> [ zeroinitializer, %.lr.ph ], [ %508, %212 ]
  %140 = phi <32 x float> [ zeroinitializer, %.lr.ph ], [ %507, %212 ]
  %141 = phi <4 x float> [ zeroinitializer, %.lr.ph ], [ %499, %212 ]
  %142 = phi i64 [ 0, %.lr.ph ], [ %509, %212 ]
  %143 = lshr i64 %142, 14, !dbg !15
  %144 = add nsw i64 %143, %8, !dbg !15
  %145 = lshr i64 %142, 7, !dbg !15
  %146 = and i64 %145, 127, !dbg !15
  %147 = shl i64 %144, 19, !dbg !15
  %148 = and i64 %147, 16252928, !dbg !15
  %149 = shl i64 %145, 19, !dbg !15
  %150 = shl nuw nsw i64 %142, 12, !dbg !15
  %151 = sub i64 %150, %149, !dbg !15
  %152 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %148, !dbg !15
  %153 = getelementptr [2 x i8], ptr addrspace(1) %152, i64 %146, !dbg !15
  %154 = getelementptr [2 x i8], ptr addrspace(1) %153, i64 %151, !dbg !15
  %address.thread926 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !15, !range !9, !frisk.register_lifetimes !5
  %155 = shl nuw nsw i32 %address.thread926, 1, !dbg !8
  %156 = and i32 %155, 384, !dbg !8
  %address.local929 = zext nneg i32 %156 to i64, !dbg !8
  %address.local930 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local929, !dbg !15
  %address.local934 = zext nneg i32 %155 to i64, !dbg !15
  %address.local935 = getelementptr [2 x i8], ptr addrspace(1) %154, i64 %address.local934, !dbg !15
  %dma.source.bits1574 = ptrtoint ptr addrspace(1) %address.local935 to i64, !dbg !15
  %dma.source.opaque1575 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1574) #9, !dbg !15
  %dma.source.ptr1576 = inttoptr i64 %dma.source.opaque1575 to ptr addrspace(1), !dbg !15
  %address.local939 = getelementptr i8, ptr addrspace(3) %address.local930, i64 1024, !dbg !15
  %address.local945 = getelementptr i8, ptr addrspace(1) %address.local935, i64 1024, !dbg !15
  %dma.source.bits1577 = ptrtoint ptr addrspace(1) %address.local945 to i64, !dbg !15
  %dma.source.opaque1578 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1577) #9, !dbg !15
  %dma.source.ptr1579 = inttoptr i64 %dma.source.opaque1578 to ptr addrspace(1), !dbg !15
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local930, i64 2048, !dbg !15
  %address.local955 = getelementptr i8, ptr addrspace(1) %address.local935, i64 2048, !dbg !15
  %dma.source.bits1580 = ptrtoint ptr addrspace(1) %address.local955 to i64, !dbg !15
  %dma.source.opaque1581 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1580) #9, !dbg !15
  %dma.source.ptr1582 = inttoptr i64 %dma.source.opaque1581 to ptr addrspace(1), !dbg !15
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local930, i64 3072, !dbg !15
  %address.local965 = getelementptr i8, ptr addrspace(1) %address.local935, i64 3072, !dbg !15
  %dma.source.bits1583 = ptrtoint ptr addrspace(1) %address.local965 to i64, !dbg !15
  %dma.source.opaque1584 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1583) #9, !dbg !15
  %dma.source.ptr1585 = inttoptr i64 %dma.source.opaque1584 to ptr addrspace(1), !dbg !15
  %address.local969 = getelementptr i8, ptr addrspace(3) %address.local930, i64 4096, !dbg !15
  %address.local975 = getelementptr i8, ptr addrspace(1) %address.local935, i64 4096, !dbg !15
  %dma.source.bits1586 = ptrtoint ptr addrspace(1) %address.local975 to i64, !dbg !15
  %dma.source.opaque1587 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1586) #9, !dbg !15
  %dma.source.ptr1588 = inttoptr i64 %dma.source.opaque1587 to ptr addrspace(1), !dbg !15
  %address.local979 = getelementptr i8, ptr addrspace(3) %address.local930, i64 5120, !dbg !15
  %address.local985 = getelementptr i8, ptr addrspace(1) %address.local935, i64 5120, !dbg !15
  %dma.source.bits1589 = ptrtoint ptr addrspace(1) %address.local985 to i64, !dbg !15
  %dma.source.opaque1590 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1589) #9, !dbg !15
  %dma.source.ptr1591 = inttoptr i64 %dma.source.opaque1590 to ptr addrspace(1), !dbg !15
  %address.local989 = getelementptr i8, ptr addrspace(3) %address.local930, i64 6144, !dbg !15
  %address.local995 = getelementptr i8, ptr addrspace(1) %address.local935, i64 6144, !dbg !15
  %dma.source.bits1592 = ptrtoint ptr addrspace(1) %address.local995 to i64, !dbg !15
  %dma.source.opaque1593 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1592) #9, !dbg !15
  %dma.source.ptr1594 = inttoptr i64 %dma.source.opaque1593 to ptr addrspace(1), !dbg !15
  %address.local999 = getelementptr i8, ptr addrspace(3) %address.local930, i64 7168, !dbg !15
  %157 = insertelement <8 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local930, i64 0, !dbg !15
  %158 = insertelement <8 x ptr addrspace(3)> %157, ptr addrspace(3) %address.local939, i64 1, !dbg !15
  %159 = insertelement <8 x ptr addrspace(3)> %158, ptr addrspace(3) %address.local949, i64 2, !dbg !15
  %160 = insertelement <8 x ptr addrspace(3)> %159, ptr addrspace(3) %address.local959, i64 3, !dbg !15
  %161 = insertelement <8 x ptr addrspace(3)> %160, ptr addrspace(3) %address.local969, i64 4, !dbg !15
  %162 = insertelement <8 x ptr addrspace(3)> %161, ptr addrspace(3) %address.local979, i64 5, !dbg !15
  %163 = insertelement <8 x ptr addrspace(3)> %162, ptr addrspace(3) %address.local989, i64 6, !dbg !15
  %164 = insertelement <8 x ptr addrspace(3)> %163, ptr addrspace(3) %address.local999, i64 7, !dbg !15
  %165 = ptrtoint <8 x ptr addrspace(3)> %164 to <8 x i64>, !dbg !15
  %166 = trunc <8 x i64> %165 to <8 x i32>, !dbg !15
  %167 = extractelement <8 x i32> %166, i64 0, !dbg !15
  %168 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %167), !dbg !15
  %169 = zext i32 %168 to i64, !dbg !15
  %170 = inttoptr i64 %169 to ptr addrspace(3), !dbg !15
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1576, ptr addrspace(3) %170, i32 4, i32 0, i32 0), !dbg !15
  %171 = extractelement <8 x i32> %166, i64 1, !dbg !15
  %172 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %171), !dbg !15
  %173 = zext i32 %172 to i64, !dbg !15
  %174 = inttoptr i64 %173 to ptr addrspace(3), !dbg !15
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1579, ptr addrspace(3) %174, i32 4, i32 0, i32 0), !dbg !15
  %175 = extractelement <8 x i32> %166, i64 2, !dbg !15
  %176 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %175), !dbg !15
  %177 = zext i32 %176 to i64, !dbg !15
  %178 = inttoptr i64 %177 to ptr addrspace(3), !dbg !15
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1582, ptr addrspace(3) %178, i32 4, i32 0, i32 0), !dbg !15
  %179 = extractelement <8 x i32> %166, i64 3, !dbg !15
  %180 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %179), !dbg !15
  %181 = zext i32 %180 to i64, !dbg !15
  %182 = inttoptr i64 %181 to ptr addrspace(3), !dbg !15
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1585, ptr addrspace(3) %182, i32 4, i32 0, i32 0), !dbg !15
  %183 = extractelement <8 x i32> %166, i64 4, !dbg !15
  %184 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %183), !dbg !15
  %185 = zext i32 %184 to i64, !dbg !15
  %186 = inttoptr i64 %185 to ptr addrspace(3), !dbg !15
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1588, ptr addrspace(3) %186, i32 4, i32 0, i32 0), !dbg !15
  %187 = extractelement <8 x i32> %166, i64 5, !dbg !15
  %188 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %187), !dbg !15
  %189 = zext i32 %188 to i64, !dbg !15
  %190 = inttoptr i64 %189 to ptr addrspace(3), !dbg !15
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1591, ptr addrspace(3) %190, i32 4, i32 0, i32 0), !dbg !15
  %191 = extractelement <8 x i32> %166, i64 6, !dbg !15
  %192 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %191), !dbg !15
  %193 = zext i32 %192 to i64, !dbg !15
  %194 = inttoptr i64 %193 to ptr addrspace(3), !dbg !15
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1594, ptr addrspace(3) %194, i32 4, i32 0, i32 0), !dbg !15
  %195 = extractelement <8 x i32> %166, i64 7, !dbg !15
  %196 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %195), !dbg !15
  %197 = zext i32 %196 to i64, !dbg !15
  %198 = inttoptr i64 %197 to ptr addrspace(3), !dbg !15
  %address.local1005 = getelementptr i8, ptr addrspace(1) %address.local935, i64 7168, !dbg !15
  %dma.source.bits1595 = ptrtoint ptr addrspace(1) %address.local1005 to i64, !dbg !15
  %dma.source.opaque1596 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1595) #9, !dbg !15
  %dma.source.ptr1597 = inttoptr i64 %dma.source.opaque1596 to ptr addrspace(1), !dbg !15
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1597, ptr addrspace(3) %198, i32 4, i32 0, i32 0), !dbg !15
  tail call void @llvm.amdgcn.asyncmark(), !dbg !15
  %199 = add nsw i64 %145, %8, !dbg !16
  %200 = shl i64 %199, 19, !dbg !16
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  br label %201, !dbg !15

201:                                              ; preds = %138, %201
  %202 = phi i64 [ %18, %138 ], [ %210, %201 ]
  %203 = lshr i64 %202, 4, !dbg !15
  %gep = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %invariant.gep, i64 %203, !dbg !15
  %204 = load half, ptr addrspace(3) %gep, align 2, !dbg !15
  %205 = getelementptr inbounds nuw i8, ptr addrspace(3) %gep, i64 256, !dbg !15
  %206 = load half, ptr addrspace(3) %205, align 2, !dbg !15
  %207 = insertelement <2 x half> poison, half %204, i64 0, !dbg !15
  %208 = insertelement <2 x half> %207, half %206, i64 1, !dbg !15
  %.idx3 = shl nuw nsw i64 %202, 2, !dbg !15
  %209 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx3, !dbg !15
  store <2 x half> %208, ptr addrspace(3) %209, align 4, !dbg !15
  %210 = add nuw nsw i64 %202, 256, !dbg !15
  %211 = icmp samesign ult i64 %202, 1792, !dbg !15
  br i1 %211, label %201, label %212, !dbg !15

212:                                              ; preds = %201
  %213 = and i64 %200, 16252928, !dbg !16
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %214 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %213, !dbg !16
  %215 = getelementptr [2 x i8], ptr addrspace(1) %214, i64 %151, !dbg !16
  %address.thread1006 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !16, !range !9, !frisk.register_lifetimes !5
  %216 = shl nuw nsw i32 %address.thread1006, 1, !dbg !8
  %217 = and i32 %216, 384, !dbg !8
  %address.local1009 = zext nneg i32 %217 to i64, !dbg !8
  %address.local1010 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1009, !dbg !16
  %address.local1013 = and i32 %address.thread1006, 120, !dbg !16
  %address.local1015 = and i32 %216, 262, !dbg !16
  %address.local1016 = or disjoint i32 %address.local1015, %address.local1013, !dbg !16
  %address.local1017 = shl nuw nsw i32 %address.thread1006, 5, !dbg !16
  %address.local1018 = and i32 %address.local1017, 128, !dbg !16
  %address.local1019 = or disjoint i32 %address.local1016, %address.local1018, !dbg !16
  %address.local1020 = zext nneg i32 %address.local1019 to i64, !dbg !16
  %address.local1021 = getelementptr [2 x i8], ptr addrspace(1) %215, i64 %address.local1020, !dbg !16
  %dma.source.bits1598 = ptrtoint ptr addrspace(1) %address.local1021 to i64, !dbg !16
  %dma.source.opaque1599 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1598) #9, !dbg !16
  %dma.source.ptr1600 = inttoptr i64 %dma.source.opaque1599 to ptr addrspace(1), !dbg !16
  %address.local1025 = getelementptr i8, ptr addrspace(3) %address.local1010, i64 1024, !dbg !16
  %address.local1037 = getelementptr i8, ptr addrspace(1) %address.local1021, i64 1024, !dbg !16
  %dma.source.bits1601 = ptrtoint ptr addrspace(1) %address.local1037 to i64, !dbg !16
  %dma.source.opaque1602 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1601) #9, !dbg !16
  %dma.source.ptr1603 = inttoptr i64 %dma.source.opaque1602 to ptr addrspace(1), !dbg !16
  %address.local1041 = getelementptr i8, ptr addrspace(3) %address.local1010, i64 2048, !dbg !16
  %address.local1053 = getelementptr i8, ptr addrspace(1) %address.local1021, i64 2048, !dbg !16
  %dma.source.bits1604 = ptrtoint ptr addrspace(1) %address.local1053 to i64, !dbg !16
  %dma.source.opaque1605 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1604) #9, !dbg !16
  %dma.source.ptr1606 = inttoptr i64 %dma.source.opaque1605 to ptr addrspace(1), !dbg !16
  %address.local1057 = getelementptr i8, ptr addrspace(3) %address.local1010, i64 3072, !dbg !16
  %address.local1069 = getelementptr i8, ptr addrspace(1) %address.local1021, i64 3072, !dbg !16
  %dma.source.bits1607 = ptrtoint ptr addrspace(1) %address.local1069 to i64, !dbg !16
  %dma.source.opaque1608 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1607) #9, !dbg !16
  %dma.source.ptr1609 = inttoptr i64 %dma.source.opaque1608 to ptr addrspace(1), !dbg !16
  %address.local1073 = getelementptr i8, ptr addrspace(3) %address.local1010, i64 4096, !dbg !16
  %address.local1085 = getelementptr i8, ptr addrspace(1) %address.local1021, i64 4096, !dbg !16
  %dma.source.bits1610 = ptrtoint ptr addrspace(1) %address.local1085 to i64, !dbg !16
  %dma.source.opaque1611 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1610) #9, !dbg !16
  %dma.source.ptr1612 = inttoptr i64 %dma.source.opaque1611 to ptr addrspace(1), !dbg !16
  %address.local1089 = getelementptr i8, ptr addrspace(3) %address.local1010, i64 5120, !dbg !16
  %address.local1101 = getelementptr i8, ptr addrspace(1) %address.local1021, i64 5120, !dbg !16
  %dma.source.bits1613 = ptrtoint ptr addrspace(1) %address.local1101 to i64, !dbg !16
  %dma.source.opaque1614 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1613) #9, !dbg !16
  %dma.source.ptr1615 = inttoptr i64 %dma.source.opaque1614 to ptr addrspace(1), !dbg !16
  %address.local1105 = getelementptr i8, ptr addrspace(3) %address.local1010, i64 6144, !dbg !16
  %address.local1117 = getelementptr i8, ptr addrspace(1) %address.local1021, i64 6144, !dbg !16
  %dma.source.bits1616 = ptrtoint ptr addrspace(1) %address.local1117 to i64, !dbg !16
  %dma.source.opaque1617 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1616) #9, !dbg !16
  %dma.source.ptr1618 = inttoptr i64 %dma.source.opaque1617 to ptr addrspace(1), !dbg !16
  %address.local1121 = getelementptr i8, ptr addrspace(3) %address.local1010, i64 7168, !dbg !16
  %218 = insertelement <8 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local1010, i64 0, !dbg !16
  %219 = insertelement <8 x ptr addrspace(3)> %218, ptr addrspace(3) %address.local1025, i64 1, !dbg !16
  %220 = insertelement <8 x ptr addrspace(3)> %219, ptr addrspace(3) %address.local1041, i64 2, !dbg !16
  %221 = insertelement <8 x ptr addrspace(3)> %220, ptr addrspace(3) %address.local1057, i64 3, !dbg !16
  %222 = insertelement <8 x ptr addrspace(3)> %221, ptr addrspace(3) %address.local1073, i64 4, !dbg !16
  %223 = insertelement <8 x ptr addrspace(3)> %222, ptr addrspace(3) %address.local1089, i64 5, !dbg !16
  %224 = insertelement <8 x ptr addrspace(3)> %223, ptr addrspace(3) %address.local1105, i64 6, !dbg !16
  %225 = insertelement <8 x ptr addrspace(3)> %224, ptr addrspace(3) %address.local1121, i64 7, !dbg !16
  %226 = ptrtoint <8 x ptr addrspace(3)> %225 to <8 x i64>, !dbg !16
  %227 = trunc <8 x i64> %226 to <8 x i32>, !dbg !16
  %228 = extractelement <8 x i32> %227, i64 0, !dbg !16
  %229 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %228), !dbg !16
  %230 = zext i32 %229 to i64, !dbg !16
  %231 = inttoptr i64 %230 to ptr addrspace(3), !dbg !16
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1600, ptr addrspace(3) %231, i32 4, i32 0, i32 0), !dbg !16
  %232 = extractelement <8 x i32> %227, i64 1, !dbg !16
  %233 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %232), !dbg !16
  %234 = zext i32 %233 to i64, !dbg !16
  %235 = inttoptr i64 %234 to ptr addrspace(3), !dbg !16
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1603, ptr addrspace(3) %235, i32 4, i32 0, i32 0), !dbg !16
  %236 = extractelement <8 x i32> %227, i64 2, !dbg !16
  %237 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %236), !dbg !16
  %238 = zext i32 %237 to i64, !dbg !16
  %239 = inttoptr i64 %238 to ptr addrspace(3), !dbg !16
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1606, ptr addrspace(3) %239, i32 4, i32 0, i32 0), !dbg !16
  %240 = extractelement <8 x i32> %227, i64 3, !dbg !16
  %241 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %240), !dbg !16
  %242 = zext i32 %241 to i64, !dbg !16
  %243 = inttoptr i64 %242 to ptr addrspace(3), !dbg !16
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1609, ptr addrspace(3) %243, i32 4, i32 0, i32 0), !dbg !16
  %244 = extractelement <8 x i32> %227, i64 4, !dbg !16
  %245 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %244), !dbg !16
  %246 = zext i32 %245 to i64, !dbg !16
  %247 = inttoptr i64 %246 to ptr addrspace(3), !dbg !16
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1612, ptr addrspace(3) %247, i32 4, i32 0, i32 0), !dbg !16
  %248 = extractelement <8 x i32> %227, i64 5, !dbg !16
  %249 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %248), !dbg !16
  %250 = zext i32 %249 to i64, !dbg !16
  %251 = inttoptr i64 %250 to ptr addrspace(3), !dbg !16
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1615, ptr addrspace(3) %251, i32 4, i32 0, i32 0), !dbg !16
  %252 = extractelement <8 x i32> %227, i64 6, !dbg !16
  %253 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %252), !dbg !16
  %254 = zext i32 %253 to i64, !dbg !16
  %255 = inttoptr i64 %254 to ptr addrspace(3), !dbg !16
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1618, ptr addrspace(3) %255, i32 4, i32 0, i32 0), !dbg !16
  %256 = extractelement <8 x i32> %227, i64 7, !dbg !16
  %257 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %256), !dbg !16
  %258 = zext i32 %257 to i64, !dbg !16
  %259 = inttoptr i64 %258 to ptr addrspace(3), !dbg !16
  %address.local1133 = getelementptr i8, ptr addrspace(1) %address.local1021, i64 7168, !dbg !16
  %dma.source.bits1619 = ptrtoint ptr addrspace(1) %address.local1133 to i64, !dbg !16
  %dma.source.opaque1620 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1619) #9, !dbg !16
  %dma.source.ptr1621 = inttoptr i64 %dma.source.opaque1620 to ptr addrspace(1), !dbg !16
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1621, ptr addrspace(3) %259, i32 4, i32 0, i32 0), !dbg !16
  tail call void @llvm.amdgcn.asyncmark(), !dbg !16
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local1134 = shl nuw nsw i32 %address.thread1006, 3, !dbg !12
  %address.local1135 = and i32 %address.local1134, 504, !dbg !12
  %address.local1136 = zext nneg i32 %address.local1135 to i64, !dbg !12
  %address.local1137 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1136, !dbg !12
  %260 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1137, i32 0) #10, !dbg !12
  %261 = bitcast <4 x i32> %260 to <8 x half>, !dbg !12
  %address.local1142 = getelementptr i8, ptr addrspace(3) %address.local1137, i64 1024, !dbg !12
  %262 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1142, i32 0) #10, !dbg !12
  %263 = bitcast <4 x i32> %262 to <8 x half>, !dbg !12
  %address.local1147 = getelementptr i8, ptr addrspace(3) %address.local1137, i64 2048, !dbg !12
  %264 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1147, i32 0) #10, !dbg !12
  %265 = bitcast <4 x i32> %264 to <8 x half>, !dbg !12
  %address.local1149 = or i32 %address.local1134, 1536, !dbg !12
  %address.local1150 = zext nneg i32 %address.local1149 to i64, !dbg !12
  %address.local1151 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1150, !dbg !12
  %266 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1151, i32 0) #10, !dbg !12
  %267 = bitcast <4 x i32> %266 to <8 x half>, !dbg !12
  %address.local1156 = getelementptr i8, ptr addrspace(3) %address.local1137, i64 4096, !dbg !12
  %268 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1156, i32 0) #10, !dbg !12
  %269 = bitcast <4 x i32> %268 to <8 x half>, !dbg !12
  %address.local1161 = getelementptr i8, ptr addrspace(3) %address.local1137, i64 5120, !dbg !12
  %270 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1161, i32 0) #10, !dbg !12
  %271 = bitcast <4 x i32> %270 to <8 x half>, !dbg !12
  %address.local1166 = getelementptr i8, ptr addrspace(3) %address.local1137, i64 6144, !dbg !12
  %272 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1166, i32 0) #10, !dbg !12
  %273 = bitcast <4 x i32> %272 to <8 x half>, !dbg !12
  %address.local1168 = or i32 %address.local1134, 3584, !dbg !12
  %address.local1169 = zext nneg i32 %address.local1168 to i64, !dbg !12
  %address.local1170 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1169, !dbg !12
  %274 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1170, i32 0) #10, !dbg !12
  %275 = bitcast <4 x i32> %274 to <8 x half>, !dbg !12
  %276 = shufflevector <8 x half> %261, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !12
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %276, <4 x float> zeroinitializer) #8, !dbg !12
  %278 = shufflevector <8 x half> %263, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !12
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %278, <4 x float> %277) #8, !dbg !12
  %280 = shufflevector <8 x half> %265, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !12
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %280, <4 x float> %279) #8, !dbg !12
  %282 = shufflevector <8 x half> %267, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !12
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %282, <4 x float> %281) #8, !dbg !12
  %284 = shufflevector <8 x half> %269, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !12
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %284, <4 x float> %283) #8, !dbg !12
  %286 = shufflevector <8 x half> %271, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !12
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %286, <4 x float> %285) #8, !dbg !12
  %288 = shufflevector <8 x half> %273, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !12
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %288, <4 x float> %287) #8, !dbg !12
  %290 = shufflevector <8 x half> %275, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !12
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %290, <4 x float> %289) #8, !dbg !12
  %292 = extractelement <4 x float> %291, i64 0, !dbg !12
  %293 = extractelement <4 x float> %291, i64 1, !dbg !12
  %294 = extractelement <4 x float> %291, i64 2, !dbg !12
  %295 = extractelement <4 x float> %291, i64 3, !dbg !12
  %296 = select i1 %.not, float %293, float %292, !dbg !12
  %297 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !12
  %298 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %297), !dbg !12
  %299 = add nuw nsw i32 %298, 64, !dbg !12
  %300 = xor i32 %298, 16, !dbg !12
  %301 = and i32 %299, 192, !dbg !12
  %302 = icmp samesign ult i32 %300, %301, !dbg !12
  %303 = select i1 %302, i32 %300, i32 %298, !dbg !12
  %304 = shl nuw nsw i32 %303, 2, !dbg !12
  %305 = bitcast float %296 to i32, !dbg !12
  %306 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %304, i32 %305), !dbg !12
  %307 = bitcast i32 %306 to float, !dbg !12
  %308 = select i1 %.not, float %292, float %307, !dbg !12
  %309 = select i1 %.not, float %307, float %293, !dbg !12
  %310 = select i1 %.not, float %295, float %294, !dbg !12
  %311 = bitcast float %310 to i32, !dbg !12
  %312 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %304, i32 %311), !dbg !12
  %313 = bitcast i32 %312 to float, !dbg !12
  %314 = select i1 %.not, float %294, float %313, !dbg !12
  %315 = select i1 %.not, float %313, float %295, !dbg !12
  %316 = select i1 %.not566, float %314, float %308, !dbg !12
  %317 = xor i32 %298, 32, !dbg !12
  %318 = icmp samesign ult i32 %317, %301, !dbg !12
  %319 = select i1 %318, i32 %317, i32 %298, !dbg !12
  %320 = shl nuw nsw i32 %319, 2, !dbg !12
  %321 = bitcast float %316 to i32, !dbg !12
  %322 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %320, i32 %321), !dbg !12
  %323 = bitcast i32 %322 to float, !dbg !12
  %324 = select i1 %.not566, float %308, float %323, !dbg !12
  %325 = select i1 %.not566, float %323, float %314, !dbg !12
  %326 = select i1 %.not566, float %315, float %309, !dbg !12
  %327 = bitcast float %326 to i32, !dbg !12
  %328 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %320, i32 %327), !dbg !12
  %329 = bitcast i32 %328 to float, !dbg !12
  %330 = select i1 %.not566, float %309, float %329, !dbg !12
  %331 = select i1 %.not566, float %329, float %315, !dbg !12
  %332 = lshr i64 %142, 27, !dbg !17
  %compare.high1501 = trunc nuw i64 %332 to i32, !dbg !17
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1501, !dbg !17
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1501, !dbg !17
  %.tr = trunc i64 %142 to i32, !dbg !17
  %333 = shl i32 %.tr, 5, !dbg !17
  %compare.low.base1502 = or disjoint i32 %333, %126, !dbg !17
  %compare.low = icmp ult i32 %compare.low.base1502, %compare.low.base, !dbg !17
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !17
  %334 = select i1 %compare.result, float 0.000000e+00, float -inf, !dbg !18
  %335 = fadd float %334, %324, !dbg !13
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 1) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local = or i32 %compare.local.base, 1, !dbg !17
  %compare.low1503 = icmp ult i32 %compare.local, %compare.low.base, !dbg !17
  %compare.result1504 = select i1 %compare.high.equal, i1 %compare.low1503, i1 %compare.high.order, !dbg !17
  %336 = select i1 %compare.result1504, float 0.000000e+00, float -inf, !dbg !18
  %337 = fadd float %330, %336, !dbg !13
  %compare.local.base1505 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 2) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local1506 = or i32 %compare.local.base1505, 2, !dbg !17
  %compare.low1507 = icmp ult i32 %compare.local1506, %compare.low.base, !dbg !17
  %compare.result1508 = select i1 %compare.high.equal, i1 %compare.low1507, i1 %compare.high.order, !dbg !17
  %338 = select i1 %compare.result1508, float 0.000000e+00, float -inf, !dbg !18
  %339 = fadd float %325, %338, !dbg !13
  %compare.local.base1509 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 3) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local1510 = or i32 %compare.local.base1509, 3, !dbg !17
  %compare.low1511 = icmp ult i32 %compare.local1510, %compare.low.base, !dbg !17
  %compare.result1512 = select i1 %compare.high.equal, i1 %compare.low1511, i1 %compare.high.order, !dbg !17
  %340 = select i1 %compare.result1512, float 0.000000e+00, float -inf, !dbg !18
  %341 = fadd float %331, %340, !dbg !13
  %342 = tail call float @__ocml_exp2_f32(float %335), !dbg !19
  %343 = fadd float %342, 0.000000e+00, !dbg !20
  %344 = tail call float @__ocml_exp2_f32(float %337), !dbg !19
  %345 = fadd float %343, %344, !dbg !20
  %346 = tail call float @__ocml_exp2_f32(float %339), !dbg !19
  %347 = fadd float %345, %346, !dbg !20
  %348 = tail call float @__ocml_exp2_f32(float %341), !dbg !19
  %349 = fadd float %347, %348, !dbg !20
  %350 = shufflevector <8 x half> %261, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %350, <4 x float> zeroinitializer) #8, !dbg !12
  %352 = shufflevector <8 x half> %263, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %352, <4 x float> %351) #8, !dbg !12
  %354 = shufflevector <8 x half> %265, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %354, <4 x float> %353) #8, !dbg !12
  %356 = shufflevector <8 x half> %267, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %356, <4 x float> %355) #8, !dbg !12
  %358 = shufflevector <8 x half> %269, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %358, <4 x float> %357) #8, !dbg !12
  %360 = shufflevector <8 x half> %271, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %360, <4 x float> %359) #8, !dbg !12
  %362 = shufflevector <8 x half> %273, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %362, <4 x float> %361) #8, !dbg !12
  %364 = shufflevector <8 x half> %275, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %364, <4 x float> %363) #8, !dbg !12
  %366 = extractelement <4 x float> %365, i64 0, !dbg !12
  %367 = extractelement <4 x float> %365, i64 1, !dbg !12
  %368 = extractelement <4 x float> %365, i64 2, !dbg !12
  %369 = extractelement <4 x float> %365, i64 3, !dbg !12
  %370 = select i1 %.not, float %367, float %366, !dbg !12
  %371 = bitcast float %370 to i32, !dbg !12
  %372 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %304, i32 %371), !dbg !12
  %373 = bitcast i32 %372 to float, !dbg !12
  %374 = select i1 %.not, float %366, float %373, !dbg !12
  %375 = select i1 %.not, float %373, float %367, !dbg !12
  %376 = select i1 %.not, float %369, float %368, !dbg !12
  %377 = bitcast float %376 to i32, !dbg !12
  %378 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %304, i32 %377), !dbg !12
  %379 = bitcast i32 %378 to float, !dbg !12
  %380 = select i1 %.not, float %368, float %379, !dbg !12
  %381 = select i1 %.not, float %379, float %369, !dbg !12
  %382 = select i1 %.not566, float %380, float %374, !dbg !12
  %383 = bitcast float %382 to i32, !dbg !12
  %384 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %320, i32 %383), !dbg !12
  %385 = bitcast i32 %384 to float, !dbg !12
  %386 = select i1 %.not566, float %374, float %385, !dbg !12
  %387 = select i1 %.not566, float %385, float %380, !dbg !12
  %388 = select i1 %.not566, float %381, float %375, !dbg !12
  %389 = bitcast float %388 to i32, !dbg !12
  %390 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %320, i32 %389), !dbg !12
  %391 = bitcast i32 %390 to float, !dbg !12
  %392 = select i1 %.not566, float %375, float %391, !dbg !12
  %393 = select i1 %.not566, float %391, float %381, !dbg !12
  %compare.local.base1513 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 4) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local1514 = or i32 %compare.local.base1513, 16, !dbg !17
  %compare.low1515 = icmp ult i32 %compare.local1514, %compare.low.base, !dbg !17
  %compare.result1516 = select i1 %compare.high.equal, i1 %compare.low1515, i1 %compare.high.order, !dbg !17
  %394 = select i1 %compare.result1516, float 0.000000e+00, float -inf, !dbg !18
  %395 = fadd float %386, %394, !dbg !13
  %compare.local.base1517 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 5) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local1518 = or i32 %compare.local.base1517, 17, !dbg !17
  %compare.low1519 = icmp ult i32 %compare.local1518, %compare.low.base, !dbg !17
  %compare.result1520 = select i1 %compare.high.equal, i1 %compare.low1519, i1 %compare.high.order, !dbg !17
  %396 = select i1 %compare.result1520, float 0.000000e+00, float -inf, !dbg !18
  %397 = fadd float %392, %396, !dbg !13
  %compare.local.base1521 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 6) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local1522 = or i32 %compare.local.base1521, 18, !dbg !17
  %compare.low1523 = icmp ult i32 %compare.local1522, %compare.low.base, !dbg !17
  %compare.result1524 = select i1 %compare.high.equal, i1 %compare.low1523, i1 %compare.high.order, !dbg !17
  %398 = select i1 %compare.result1524, float 0.000000e+00, float -inf, !dbg !18
  %399 = fadd float %387, %398, !dbg !13
  %compare.local.base1525 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 7) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local1526 = or i32 %compare.local.base1525, 19, !dbg !17
  %compare.low1527 = icmp ult i32 %compare.local1526, %compare.low.base, !dbg !17
  %compare.result1528 = select i1 %compare.high.equal, i1 %compare.low1527, i1 %compare.high.order, !dbg !17
  %400 = select i1 %compare.result1528, float 0.000000e+00, float -inf, !dbg !18
  %401 = fadd float %393, %400, !dbg !13
  %402 = tail call float @__ocml_exp2_f32(float %395), !dbg !19
  %403 = fadd float %349, %402, !dbg !20
  %404 = tail call float @__ocml_exp2_f32(float %397), !dbg !19
  %405 = fadd float %403, %404, !dbg !20
  %406 = tail call float @__ocml_exp2_f32(float %399), !dbg !19
  %407 = fadd float %405, %406, !dbg !20
  %408 = tail call float @__ocml_exp2_f32(float %401), !dbg !19
  %409 = fadd float %407, %408, !dbg !20
  %410 = bitcast float %409 to i32, !dbg !20
  %411 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %304, i32 %410), !dbg !20
  %412 = bitcast i32 %411 to float, !dbg !20
  %413 = fadd float %409, %412, !dbg !20
  %414 = bitcast float %413 to i32, !dbg !20
  %415 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %320, i32 %414), !dbg !20
  %416 = bitcast i32 %415 to float, !dbg !20
  %417 = fadd float %413, %416, !dbg !20
  %418 = extractelement <1 x float> %139, i64 0, !dbg !20
  %419 = fadd float %418, %417, !dbg !20
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  %420 = insertelement <4 x float> poison, float %402, i64 0, !dbg !21
  %421 = insertelement <4 x float> %420, float %404, i64 1, !dbg !21
  %422 = insertelement <4 x float> %421, float %406, i64 2, !dbg !21
  %423 = insertelement <4 x float> %422, float %408, i64 3, !dbg !21
  %424 = fptrunc <4 x float> %423 to <4 x half>, !dbg !21
  %425 = insertelement <4 x float> poison, float %342, i64 0, !dbg !21
  %426 = insertelement <4 x float> %425, float %344, i64 1, !dbg !21
  %427 = insertelement <4 x float> %426, float %346, i64 2, !dbg !21
  %428 = insertelement <4 x float> %427, float %348, i64 3, !dbg !21
  %429 = fptrunc <4 x float> %428 to <4 x half>, !dbg !21
  %address.local1172 = and i32 %address.thread1006, 192, !dbg !10
  %address.local1173 = shl nuw nsw i32 %address.thread1006, 2, !dbg !10
  %address.local1174 = and i32 %address.local1173, 60, !dbg !10
  %address.local1175 = or disjoint i32 %address.local1174, %address.local1172, !dbg !10
  %address.local1176 = shl nuw nsw i32 %address.thread1006, 4, !dbg !10
  %address.local1177 = and i32 %address.local1176, 768, !dbg !10
  %address.local1178 = or disjoint i32 %address.local1175, %address.local1177, !dbg !10
  %address.local1179 = zext nneg i32 %address.local1178 to i64, !dbg !10
  %address.local1180 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1179, !dbg !10
  store <4 x half> %429, ptr addrspace(3) %address.local1180, align 8, !dbg !21
  %address.local1190 = getelementptr i8, ptr addrspace(3) %address.local1180, i64 2048, !dbg !10
  store <4 x half> %424, ptr addrspace(3) %address.local1190, align 8, !dbg !21
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  %address.local1193 = and i32 %216, 8, !dbg !22
  %address.local1195 = and i32 %address.local1176, 48, !dbg !22
  %address.local1196 = or disjoint i32 %address.local1193, %address.local1195, !dbg !22
  %address.local1198 = and i32 %address.local1017, 1792, !dbg !22
  %address.local1199 = or disjoint i32 %address.local1196, %address.local1198, !dbg !22
  %address.local1200 = zext nneg i32 %address.local1199 to i64, !dbg !22
  %address.local1201 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1200, !dbg !22
  %430 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1201, i32 0) #10, !dbg !22
  %431 = bitcast <4 x i32> %430 to <8 x half>, !dbg !22
  %address.local1212 = getelementptr i8, ptr addrspace(3) %address.local1201, i64 128, !dbg !22
  %432 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1212, i32 0) #10, !dbg !22
  %433 = bitcast <4 x i32> %432 to <8 x half>, !dbg !22
  %address.local1223 = getelementptr i8, ptr addrspace(3) %address.local1201, i64 256, !dbg !22
  %434 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1223, i32 0) #10, !dbg !22
  %435 = bitcast <4 x i32> %434 to <8 x half>, !dbg !22
  %address.local1234 = getelementptr i8, ptr addrspace(3) %address.local1201, i64 384, !dbg !22
  %436 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1234, i32 0) #10, !dbg !22
  %437 = bitcast <4 x i32> %436 to <8 x half>, !dbg !22
  %address.local1245 = getelementptr i8, ptr addrspace(3) %address.local1201, i64 4096, !dbg !22
  %438 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1245, i32 0) #10, !dbg !22
  %439 = bitcast <4 x i32> %438 to <8 x half>, !dbg !22
  %address.local1256 = getelementptr i8, ptr addrspace(3) %address.local1201, i64 4224, !dbg !22
  %440 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1256, i32 0) #10, !dbg !22
  %441 = bitcast <4 x i32> %440 to <8 x half>, !dbg !22
  %address.local1267 = getelementptr i8, ptr addrspace(3) %address.local1201, i64 4352, !dbg !22
  %442 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1267, i32 0) #10, !dbg !22
  %443 = bitcast <4 x i32> %442 to <8 x half>, !dbg !22
  %address.local1278 = getelementptr i8, ptr addrspace(3) %address.local1201, i64 4480, !dbg !22
  %444 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1278, i32 0) #10, !dbg !22
  %445 = load <4 x half>, ptr addrspace(3) %address.local1180, align 8, !dbg !22
  %446 = bitcast <4 x i32> %444 to <8 x half>, !dbg !22
  %447 = shufflevector <8 x half> %431, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !22
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %447, <4 x float> zeroinitializer) #8, !dbg !22
  %449 = load <4 x half>, ptr addrspace(3) %address.local1190, align 8, !dbg !22
  %450 = shufflevector <8 x half> %439, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !22
  %451 = shufflevector <32 x float> %140, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %452 = shufflevector <8 x half> %431, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !22
  %453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %450, <4 x float> %448) #8, !dbg !22
  %454 = shufflevector <8 x half> %439, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !22
  %455 = shufflevector <32 x float> %140, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %456 = fadd <4 x float> %451, %453, !dbg !23
  %457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %452, <4 x float> zeroinitializer) #8, !dbg !22
  %458 = shufflevector <8 x half> %433, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !22
  %459 = shufflevector <8 x half> %441, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !22
  %460 = shufflevector <8 x half> %433, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !22
  %461 = shufflevector <8 x half> %441, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !22
  %462 = shufflevector <8 x half> %435, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !22
  %463 = shufflevector <8 x half> %443, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !22
  %464 = shufflevector <8 x half> %435, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !22
  %465 = shufflevector <8 x half> %443, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !22
  %466 = shufflevector <8 x half> %437, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !22
  %467 = shufflevector <8 x half> %446, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !22
  %468 = shufflevector <8 x half> %437, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !22
  %469 = shufflevector <8 x half> %446, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !22
  %470 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %454, <4 x float> %457) #8, !dbg !22
  %471 = fadd <4 x float> %455, %470, !dbg !23
  %472 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %458, <4 x float> zeroinitializer) #8, !dbg !22
  %473 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %459, <4 x float> %472) #8, !dbg !22
  %474 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %460, <4 x float> zeroinitializer) #8, !dbg !22
  %475 = shufflevector <32 x float> %140, <32 x float> poison, <32 x i32> <i32 8, i32 9, i32 10, i32 11, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %476 = shufflevector <4 x float> %473, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %477 = fadd <32 x float> %475, %476, !dbg !23
  %478 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %461, <4 x float> %474) #8, !dbg !22
  %479 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %462, <4 x float> zeroinitializer) #8, !dbg !22
  %480 = shufflevector <32 x float> %140, <32 x float> poison, <32 x i32> <i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %481 = shufflevector <4 x float> %478, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %482 = fadd <32 x float> %480, %481, !dbg !23
  %483 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %463, <4 x float> %479) #8, !dbg !22
  %484 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %464, <4 x float> zeroinitializer) #8, !dbg !22
  %485 = shufflevector <32 x float> %140, <32 x float> poison, <32 x i32> <i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %486 = shufflevector <4 x float> %483, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %487 = fadd <32 x float> %485, %486, !dbg !23
  %488 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %465, <4 x float> %484) #8, !dbg !22
  %489 = shufflevector <32 x float> %140, <32 x float> poison, <32 x i32> <i32 20, i32 21, i32 22, i32 23, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %490 = shufflevector <4 x float> %488, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %491 = fadd <32 x float> %489, %490, !dbg !23
  %492 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %466, <4 x float> zeroinitializer) #8, !dbg !22
  %493 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %467, <4 x float> %492) #8, !dbg !22
  %494 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %468, <4 x float> zeroinitializer) #8, !dbg !22
  %495 = shufflevector <32 x float> %140, <32 x float> poison, <32 x i32> <i32 24, i32 25, i32 26, i32 27, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %496 = shufflevector <4 x float> %493, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %497 = fadd <32 x float> %495, %496, !dbg !23
  %498 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %469, <4 x float> %494) #8, !dbg !22
  %499 = fadd <4 x float> %141, %498, !dbg !23
  %500 = shufflevector <4 x float> %499, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %501 = shufflevector <4 x float> %456, <4 x float> %471, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !11
  %502 = shufflevector <32 x float> %501, <32 x float> %477, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !11
  %503 = shufflevector <32 x float> %502, <32 x float> %482, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !11
  %504 = shufflevector <32 x float> %503, <32 x float> %487, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !11
  %505 = shufflevector <32 x float> %504, <32 x float> %491, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !11
  %506 = shufflevector <32 x float> %505, <32 x float> %497, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !11
  %507 = shufflevector <32 x float> %506, <32 x float> %500, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !11
  %508 = insertelement <1 x float> poison, float %419, i64 0, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %509 = add nuw nsw i64 %142, 1, !dbg !11
  %.not6 = icmp sgt i64 %142, %125, !dbg !11
  br i1 %.not6, label %._crit_edge.loopexit, label %138, !dbg !11

._crit_edge.loopexit:                             ; preds = %212
  %510 = shufflevector <1 x float> %508, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  br label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %.._crit_edge_crit_edge, %._crit_edge.loopexit
  %.pre-phi18 = phi i32 [ %.pre17, %.._crit_edge_crit_edge ], [ %317, %._crit_edge.loopexit ], !dbg !11
  %.pre-phi16 = phi i32 [ %.pre15, %.._crit_edge_crit_edge ], [ %301, %._crit_edge.loopexit ], !dbg !11
  %.pre-phi14 = phi i32 [ %.pre13, %.._crit_edge_crit_edge ], [ %300, %._crit_edge.loopexit ], !dbg !11
  %.pre-phi10 = phi i32 [ %.pre9, %.._crit_edge_crit_edge ], [ %298, %._crit_edge.loopexit ], !dbg !11
  %.lcssa4 = phi <32 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %507, %._crit_edge.loopexit ], !dbg !11
  %.lcssa = phi <4 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %510, %._crit_edge.loopexit ], !dbg !11
  %511 = extractelement <32 x float> %.lcssa4, i64 0, !dbg !11
  %512 = extractelement <32 x float> %.lcssa4, i64 1, !dbg !11
  %513 = extractelement <32 x float> %.lcssa4, i64 2, !dbg !11
  %514 = extractelement <32 x float> %.lcssa4, i64 3, !dbg !11
  %515 = select i1 %.not, float %512, float %511, !dbg !11
  %516 = icmp samesign ult i32 %.pre-phi14, %.pre-phi16, !dbg !11
  %517 = select i1 %516, i32 %.pre-phi14, i32 %.pre-phi10, !dbg !11
  %518 = shl nuw nsw i32 %517, 2, !dbg !11
  %519 = bitcast float %515 to i32, !dbg !11
  %520 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %519), !dbg !11
  %521 = bitcast i32 %520 to float, !dbg !11
  %522 = select i1 %.not, float %511, float %521, !dbg !11
  %523 = select i1 %.not, float %521, float %512, !dbg !11
  %524 = select i1 %.not, float %514, float %513, !dbg !11
  %525 = bitcast float %524 to i32, !dbg !11
  %526 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %525), !dbg !11
  %527 = bitcast i32 %526 to float, !dbg !11
  %528 = select i1 %.not, float %513, float %527, !dbg !11
  %529 = select i1 %.not, float %527, float %514, !dbg !11
  %530 = select i1 %.not566, float %528, float %522, !dbg !11
  %531 = icmp samesign ult i32 %.pre-phi18, %.pre-phi16, !dbg !11
  %532 = select i1 %531, i32 %.pre-phi18, i32 %.pre-phi10, !dbg !11
  %533 = shl nuw nsw i32 %532, 2, !dbg !11
  %534 = bitcast float %530 to i32, !dbg !11
  %535 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %534), !dbg !11
  %536 = select i1 %.not566, float %529, float %523, !dbg !11
  %537 = bitcast float %536 to i32, !dbg !11
  %538 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %537), !dbg !11
  %539 = extractelement <32 x float> %.lcssa4, i64 4, !dbg !11
  %540 = extractelement <32 x float> %.lcssa4, i64 5, !dbg !11
  %541 = extractelement <32 x float> %.lcssa4, i64 6, !dbg !11
  %542 = extractelement <32 x float> %.lcssa4, i64 7, !dbg !11
  %543 = select i1 %.not, float %540, float %539, !dbg !11
  %544 = bitcast float %543 to i32, !dbg !11
  %545 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %544), !dbg !11
  %546 = bitcast i32 %545 to float, !dbg !11
  %547 = select i1 %.not, float %539, float %546, !dbg !11
  %548 = select i1 %.not, float %546, float %540, !dbg !11
  %549 = select i1 %.not, float %542, float %541, !dbg !11
  %550 = bitcast float %549 to i32, !dbg !11
  %551 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %550), !dbg !11
  %552 = bitcast i32 %551 to float, !dbg !11
  %553 = select i1 %.not, float %541, float %552, !dbg !11
  %554 = select i1 %.not, float %552, float %542, !dbg !11
  %555 = select i1 %.not566, float %553, float %547, !dbg !11
  %556 = bitcast float %555 to i32, !dbg !11
  %557 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %556), !dbg !11
  %558 = select i1 %.not566, float %554, float %548, !dbg !11
  %559 = bitcast float %558 to i32, !dbg !11
  %560 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %559), !dbg !11
  %561 = extractelement <32 x float> %.lcssa4, i64 8, !dbg !11
  %562 = extractelement <32 x float> %.lcssa4, i64 9, !dbg !11
  %563 = extractelement <32 x float> %.lcssa4, i64 10, !dbg !11
  %564 = extractelement <32 x float> %.lcssa4, i64 11, !dbg !11
  %565 = select i1 %.not, float %562, float %561, !dbg !11
  %566 = bitcast float %565 to i32, !dbg !11
  %567 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %566), !dbg !11
  %568 = bitcast i32 %567 to float, !dbg !11
  %569 = select i1 %.not, float %561, float %568, !dbg !11
  %570 = select i1 %.not, float %568, float %562, !dbg !11
  %571 = select i1 %.not, float %564, float %563, !dbg !11
  %572 = bitcast float %571 to i32, !dbg !11
  %573 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %572), !dbg !11
  %574 = bitcast i32 %573 to float, !dbg !11
  %575 = select i1 %.not, float %563, float %574, !dbg !11
  %576 = select i1 %.not, float %574, float %564, !dbg !11
  %577 = select i1 %.not566, float %575, float %569, !dbg !11
  %578 = bitcast float %577 to i32, !dbg !11
  %579 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %578), !dbg !11
  %580 = select i1 %.not566, float %576, float %570, !dbg !11
  %581 = bitcast float %580 to i32, !dbg !11
  %582 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %581), !dbg !11
  %583 = extractelement <32 x float> %.lcssa4, i64 12, !dbg !11
  %584 = extractelement <32 x float> %.lcssa4, i64 13, !dbg !11
  %585 = extractelement <32 x float> %.lcssa4, i64 14, !dbg !11
  %586 = extractelement <32 x float> %.lcssa4, i64 15, !dbg !11
  %587 = select i1 %.not, float %584, float %583, !dbg !11
  %588 = bitcast float %587 to i32, !dbg !11
  %589 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %588), !dbg !11
  %590 = bitcast i32 %589 to float, !dbg !11
  %591 = select i1 %.not, float %583, float %590, !dbg !11
  %592 = select i1 %.not, float %590, float %584, !dbg !11
  %593 = select i1 %.not, float %586, float %585, !dbg !11
  %594 = bitcast float %593 to i32, !dbg !11
  %595 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %594), !dbg !11
  %596 = bitcast i32 %595 to float, !dbg !11
  %597 = select i1 %.not, float %585, float %596, !dbg !11
  %598 = select i1 %.not, float %596, float %586, !dbg !11
  %599 = select i1 %.not566, float %597, float %591, !dbg !11
  %600 = bitcast float %599 to i32, !dbg !11
  %601 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %600), !dbg !11
  %602 = select i1 %.not566, float %598, float %592, !dbg !11
  %603 = bitcast float %602 to i32, !dbg !11
  %604 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %603), !dbg !11
  %605 = extractelement <32 x float> %.lcssa4, i64 16, !dbg !11
  %606 = extractelement <32 x float> %.lcssa4, i64 17, !dbg !11
  %607 = extractelement <32 x float> %.lcssa4, i64 18, !dbg !11
  %608 = extractelement <32 x float> %.lcssa4, i64 19, !dbg !11
  %609 = select i1 %.not, float %606, float %605, !dbg !11
  %610 = bitcast float %609 to i32, !dbg !11
  %611 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %610), !dbg !11
  %612 = bitcast i32 %611 to float, !dbg !11
  %613 = select i1 %.not, float %605, float %612, !dbg !11
  %614 = select i1 %.not, float %612, float %606, !dbg !11
  %615 = select i1 %.not, float %608, float %607, !dbg !11
  %616 = bitcast float %615 to i32, !dbg !11
  %617 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %616), !dbg !11
  %618 = bitcast i32 %617 to float, !dbg !11
  %619 = select i1 %.not, float %607, float %618, !dbg !11
  %620 = select i1 %.not, float %618, float %608, !dbg !11
  %621 = select i1 %.not566, float %619, float %613, !dbg !11
  %622 = bitcast float %621 to i32, !dbg !11
  %623 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %622), !dbg !11
  %624 = select i1 %.not566, float %620, float %614, !dbg !11
  %625 = bitcast float %624 to i32, !dbg !11
  %626 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %625), !dbg !11
  %627 = extractelement <32 x float> %.lcssa4, i64 20, !dbg !11
  %628 = extractelement <32 x float> %.lcssa4, i64 21, !dbg !11
  %629 = extractelement <32 x float> %.lcssa4, i64 22, !dbg !11
  %630 = extractelement <32 x float> %.lcssa4, i64 23, !dbg !11
  %631 = select i1 %.not, float %628, float %627, !dbg !11
  %632 = bitcast float %631 to i32, !dbg !11
  %633 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %632), !dbg !11
  %634 = bitcast i32 %633 to float, !dbg !11
  %635 = select i1 %.not, float %627, float %634, !dbg !11
  %636 = select i1 %.not, float %634, float %628, !dbg !11
  %637 = select i1 %.not, float %630, float %629, !dbg !11
  %638 = bitcast float %637 to i32, !dbg !11
  %639 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %638), !dbg !11
  %640 = bitcast i32 %639 to float, !dbg !11
  %641 = select i1 %.not, float %629, float %640, !dbg !11
  %642 = select i1 %.not, float %640, float %630, !dbg !11
  %643 = select i1 %.not566, float %641, float %635, !dbg !11
  %644 = bitcast float %643 to i32, !dbg !11
  %645 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %644), !dbg !11
  %646 = select i1 %.not566, float %642, float %636, !dbg !11
  %647 = bitcast float %646 to i32, !dbg !11
  %648 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %647), !dbg !11
  %649 = extractelement <32 x float> %.lcssa4, i64 24, !dbg !11
  %650 = extractelement <32 x float> %.lcssa4, i64 25, !dbg !11
  %651 = extractelement <32 x float> %.lcssa4, i64 26, !dbg !11
  %652 = extractelement <32 x float> %.lcssa4, i64 27, !dbg !11
  %653 = select i1 %.not, float %650, float %649, !dbg !11
  %654 = bitcast float %653 to i32, !dbg !11
  %655 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %654), !dbg !11
  %656 = bitcast i32 %655 to float, !dbg !11
  %657 = select i1 %.not, float %649, float %656, !dbg !11
  %658 = select i1 %.not, float %656, float %650, !dbg !11
  %659 = select i1 %.not, float %652, float %651, !dbg !11
  %660 = bitcast float %659 to i32, !dbg !11
  %661 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %660), !dbg !11
  %662 = bitcast i32 %661 to float, !dbg !11
  %663 = select i1 %.not, float %651, float %662, !dbg !11
  %664 = select i1 %.not, float %662, float %652, !dbg !11
  %665 = select i1 %.not566, float %663, float %657, !dbg !11
  %666 = bitcast float %665 to i32, !dbg !11
  %667 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %666), !dbg !11
  %668 = select i1 %.not566, float %664, float %658, !dbg !11
  %669 = bitcast float %668 to i32, !dbg !11
  %670 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %669), !dbg !11
  %671 = extractelement <32 x float> %.lcssa4, i64 28, !dbg !11
  %672 = extractelement <32 x float> %.lcssa4, i64 29, !dbg !11
  %673 = extractelement <32 x float> %.lcssa4, i64 30, !dbg !11
  %674 = extractelement <32 x float> %.lcssa4, i64 31, !dbg !11
  %675 = select i1 %.not, float %672, float %671, !dbg !11
  %676 = bitcast float %675 to i32, !dbg !11
  %677 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %676), !dbg !11
  %678 = bitcast i32 %677 to float, !dbg !11
  %679 = select i1 %.not, float %671, float %678, !dbg !11
  %680 = select i1 %.not, float %678, float %672, !dbg !11
  %681 = select i1 %.not, float %674, float %673, !dbg !11
  %682 = bitcast float %681 to i32, !dbg !11
  %683 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %682), !dbg !11
  %684 = bitcast i32 %683 to float, !dbg !11
  %685 = select i1 %.not, float %673, float %684, !dbg !11
  %686 = select i1 %.not, float %684, float %674, !dbg !11
  %687 = select i1 %.not566, float %685, float %679, !dbg !11
  %688 = bitcast float %687 to i32, !dbg !11
  %689 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %688), !dbg !11
  %690 = select i1 %.not566, float %686, float %680, !dbg !11
  %691 = bitcast float %690 to i32, !dbg !11
  %692 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %691), !dbg !11
  %693 = bitcast i32 %689 to float, !dbg !11
  %694 = bitcast i32 %692 to float, !dbg !11
  %695 = insertelement <4 x i1> poison, i1 %.not566, i64 0, !dbg !11
  %696 = shufflevector <4 x i1> %695, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %697 = insertelement <4 x float> poison, float %679, i64 0, !dbg !11
  %698 = insertelement <4 x float> %697, float %680, i64 1, !dbg !11
  %699 = insertelement <4 x float> %698, float %693, i64 2, !dbg !11
  %700 = insertelement <4 x float> %699, float %694, i64 3, !dbg !11
  %701 = insertelement <4 x float> poison, float %693, i64 0, !dbg !11
  %702 = insertelement <4 x float> %701, float %694, i64 1, !dbg !11
  %703 = insertelement <4 x float> %702, float %685, i64 2, !dbg !11
  %704 = insertelement <4 x float> %703, float %686, i64 3, !dbg !11
  %705 = select <4 x i1> %696, <4 x float> %700, <4 x float> %704, !dbg !11
  %706 = fdiv <4 x float> %705, %.lcssa, !dbg !24
  %707 = fptrunc <4 x float> %706 to <4 x half>, !dbg !25
  %708 = extractelement <4 x half> %707, i64 3, !dbg !25
  %709 = extractelement <4 x half> %707, i64 2, !dbg !25
  %710 = extractelement <4 x half> %707, i64 1, !dbg !25
  %711 = extractelement <4 x half> %707, i64 0, !dbg !25
  %712 = bitcast i32 %667 to float, !dbg !11
  %713 = bitcast i32 %670 to float, !dbg !11
  %714 = insertelement <4 x float> poison, float %657, i64 0, !dbg !11
  %715 = insertelement <4 x float> %714, float %658, i64 1, !dbg !11
  %716 = insertelement <4 x float> %715, float %712, i64 2, !dbg !11
  %717 = insertelement <4 x float> %716, float %713, i64 3, !dbg !11
  %718 = insertelement <4 x float> poison, float %712, i64 0, !dbg !11
  %719 = insertelement <4 x float> %718, float %713, i64 1, !dbg !11
  %720 = insertelement <4 x float> %719, float %663, i64 2, !dbg !11
  %721 = insertelement <4 x float> %720, float %664, i64 3, !dbg !11
  %722 = select <4 x i1> %696, <4 x float> %717, <4 x float> %721, !dbg !11
  %723 = fdiv <4 x float> %722, %.lcssa, !dbg !24
  %724 = fptrunc <4 x float> %723 to <4 x half>, !dbg !25
  %725 = extractelement <4 x half> %724, i64 3, !dbg !25
  %726 = extractelement <4 x half> %724, i64 2, !dbg !25
  %727 = extractelement <4 x half> %724, i64 1, !dbg !25
  %728 = extractelement <4 x half> %724, i64 0, !dbg !25
  %729 = bitcast i32 %645 to float, !dbg !11
  %730 = bitcast i32 %648 to float, !dbg !11
  %731 = insertelement <4 x float> poison, float %635, i64 0, !dbg !11
  %732 = insertelement <4 x float> %731, float %636, i64 1, !dbg !11
  %733 = insertelement <4 x float> %732, float %729, i64 2, !dbg !11
  %734 = insertelement <4 x float> %733, float %730, i64 3, !dbg !11
  %735 = insertelement <4 x float> poison, float %729, i64 0, !dbg !11
  %736 = insertelement <4 x float> %735, float %730, i64 1, !dbg !11
  %737 = insertelement <4 x float> %736, float %641, i64 2, !dbg !11
  %738 = insertelement <4 x float> %737, float %642, i64 3, !dbg !11
  %739 = select <4 x i1> %696, <4 x float> %734, <4 x float> %738, !dbg !11
  %740 = fdiv <4 x float> %739, %.lcssa, !dbg !24
  %741 = fptrunc <4 x float> %740 to <4 x half>, !dbg !25
  %742 = extractelement <4 x half> %741, i64 3, !dbg !25
  %743 = extractelement <4 x half> %741, i64 2, !dbg !25
  %744 = extractelement <4 x half> %741, i64 1, !dbg !25
  %745 = extractelement <4 x half> %741, i64 0, !dbg !25
  %746 = bitcast i32 %623 to float, !dbg !11
  %747 = bitcast i32 %626 to float, !dbg !11
  %748 = insertelement <4 x float> poison, float %613, i64 0, !dbg !11
  %749 = insertelement <4 x float> %748, float %614, i64 1, !dbg !11
  %750 = insertelement <4 x float> %749, float %746, i64 2, !dbg !11
  %751 = insertelement <4 x float> %750, float %747, i64 3, !dbg !11
  %752 = insertelement <4 x float> poison, float %746, i64 0, !dbg !11
  %753 = insertelement <4 x float> %752, float %747, i64 1, !dbg !11
  %754 = insertelement <4 x float> %753, float %619, i64 2, !dbg !11
  %755 = insertelement <4 x float> %754, float %620, i64 3, !dbg !11
  %756 = select <4 x i1> %696, <4 x float> %751, <4 x float> %755, !dbg !11
  %757 = fdiv <4 x float> %756, %.lcssa, !dbg !24
  %758 = fptrunc <4 x float> %757 to <4 x half>, !dbg !25
  %759 = extractelement <4 x half> %758, i64 3, !dbg !25
  %760 = extractelement <4 x half> %758, i64 2, !dbg !25
  %761 = extractelement <4 x half> %758, i64 1, !dbg !25
  %762 = extractelement <4 x half> %758, i64 0, !dbg !25
  %763 = bitcast i32 %601 to float, !dbg !11
  %764 = bitcast i32 %604 to float, !dbg !11
  %765 = insertelement <4 x float> poison, float %591, i64 0, !dbg !11
  %766 = insertelement <4 x float> %765, float %592, i64 1, !dbg !11
  %767 = insertelement <4 x float> %766, float %763, i64 2, !dbg !11
  %768 = insertelement <4 x float> %767, float %764, i64 3, !dbg !11
  %769 = insertelement <4 x float> poison, float %763, i64 0, !dbg !11
  %770 = insertelement <4 x float> %769, float %764, i64 1, !dbg !11
  %771 = insertelement <4 x float> %770, float %597, i64 2, !dbg !11
  %772 = insertelement <4 x float> %771, float %598, i64 3, !dbg !11
  %773 = select <4 x i1> %696, <4 x float> %768, <4 x float> %772, !dbg !11
  %774 = fdiv <4 x float> %773, %.lcssa, !dbg !24
  %775 = fptrunc <4 x float> %774 to <4 x half>, !dbg !25
  %776 = extractelement <4 x half> %775, i64 3, !dbg !25
  %777 = extractelement <4 x half> %775, i64 2, !dbg !25
  %778 = extractelement <4 x half> %775, i64 1, !dbg !25
  %779 = extractelement <4 x half> %775, i64 0, !dbg !25
  %780 = bitcast i32 %579 to float, !dbg !11
  %781 = bitcast i32 %582 to float, !dbg !11
  %782 = insertelement <4 x float> poison, float %569, i64 0, !dbg !11
  %783 = insertelement <4 x float> %782, float %570, i64 1, !dbg !11
  %784 = insertelement <4 x float> %783, float %780, i64 2, !dbg !11
  %785 = insertelement <4 x float> %784, float %781, i64 3, !dbg !11
  %786 = insertelement <4 x float> poison, float %780, i64 0, !dbg !11
  %787 = insertelement <4 x float> %786, float %781, i64 1, !dbg !11
  %788 = insertelement <4 x float> %787, float %575, i64 2, !dbg !11
  %789 = insertelement <4 x float> %788, float %576, i64 3, !dbg !11
  %790 = select <4 x i1> %696, <4 x float> %785, <4 x float> %789, !dbg !11
  %791 = fdiv <4 x float> %790, %.lcssa, !dbg !24
  %792 = fptrunc <4 x float> %791 to <4 x half>, !dbg !25
  %793 = extractelement <4 x half> %792, i64 3, !dbg !25
  %794 = extractelement <4 x half> %792, i64 2, !dbg !25
  %795 = extractelement <4 x half> %792, i64 1, !dbg !25
  %796 = extractelement <4 x half> %792, i64 0, !dbg !25
  %797 = bitcast i32 %557 to float, !dbg !11
  %798 = bitcast i32 %560 to float, !dbg !11
  %799 = insertelement <4 x float> poison, float %547, i64 0, !dbg !11
  %800 = insertelement <4 x float> %799, float %548, i64 1, !dbg !11
  %801 = insertelement <4 x float> %800, float %797, i64 2, !dbg !11
  %802 = insertelement <4 x float> %801, float %798, i64 3, !dbg !11
  %803 = insertelement <4 x float> poison, float %797, i64 0, !dbg !11
  %804 = insertelement <4 x float> %803, float %798, i64 1, !dbg !11
  %805 = insertelement <4 x float> %804, float %553, i64 2, !dbg !11
  %806 = insertelement <4 x float> %805, float %554, i64 3, !dbg !11
  %807 = select <4 x i1> %696, <4 x float> %802, <4 x float> %806, !dbg !11
  %808 = fdiv <4 x float> %807, %.lcssa, !dbg !24
  %809 = fptrunc <4 x float> %808 to <4 x half>, !dbg !25
  %810 = extractelement <4 x half> %809, i64 3, !dbg !25
  %811 = extractelement <4 x half> %809, i64 2, !dbg !25
  %812 = extractelement <4 x half> %809, i64 1, !dbg !25
  %813 = extractelement <4 x half> %809, i64 0, !dbg !25
  %814 = bitcast i32 %535 to float, !dbg !11
  %815 = bitcast i32 %538 to float, !dbg !11
  %816 = insertelement <4 x float> poison, float %522, i64 0, !dbg !11
  %817 = insertelement <4 x float> %816, float %523, i64 1, !dbg !11
  %818 = insertelement <4 x float> %817, float %814, i64 2, !dbg !11
  %819 = insertelement <4 x float> %818, float %815, i64 3, !dbg !11
  %820 = insertelement <4 x float> poison, float %814, i64 0, !dbg !11
  %821 = insertelement <4 x float> %820, float %815, i64 1, !dbg !11
  %822 = insertelement <4 x float> %821, float %528, i64 2, !dbg !11
  %823 = insertelement <4 x float> %822, float %529, i64 3, !dbg !11
  %824 = select <4 x i1> %696, <4 x float> %819, <4 x float> %823, !dbg !11
  %825 = fdiv <4 x float> %824, %.lcssa, !dbg !24
  %826 = fptrunc <4 x float> %825 to <4 x half>, !dbg !25
  %827 = extractelement <4 x half> %826, i64 3, !dbg !25
  %828 = extractelement <4 x half> %826, i64 2, !dbg !25
  %829 = extractelement <4 x half> %826, i64 1, !dbg !25
  %830 = extractelement <4 x half> %826, i64 0, !dbg !25
  %address.thread1298 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !26, !range !9, !frisk.register_lifetimes !5
  %address.local1299 = zext nneg i32 %address.thread1298 to i64, !dbg !8
  %address.local1301 = lshr i32 %address.thread1298, 2, !dbg !13
  %address.local1302 = and i32 %address.local1301, 48, !dbg !13
  %address.local1303 = zext nneg i32 %address.local1302 to i64, !dbg !13
  %address.local1305 = and i32 %address.thread1298, 240, !dbg !13
  %address.local1306 = sub nsw i32 0, %address.local1305, !dbg !13
  %address.local1307 = sext i32 %address.local1306 to i64, !dbg !13
  %address.local1304 = add nsw i64 %16, %address.local1299, !dbg !26
  %address.local1308 = add nsw i64 %address.local1304, %address.local1307, !dbg !26
  %address.local1309 = add nsw i64 %address.local1308, %address.local1303, !dbg !26
  %831 = getelementptr inbounds nuw i8, ptr addrspace(1) %3, i64 %.idx, !dbg !26
  %address.local1312.idx = shl nuw nsw i64 %address.local1309, 8, !dbg !26
  %address.local1312 = getelementptr inbounds nuw i8, ptr addrspace(1) %831, i64 %address.local1312.idx, !dbg !26
  %address.local1314 = and i32 %address.local1301, 12, !dbg !13
  %address.local1315 = zext nneg i32 %address.local1314 to i64, !dbg !13
  %address.local1316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1312, i64 %address.local1315, !dbg !26
  store half %830, ptr addrspace(1) %address.local1316, align 2, !dbg !26
  %address.local1322 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 2, !dbg !26
  store half %829, ptr addrspace(1) %address.local1322, align 2, !dbg !26
  %address.local1328 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 4, !dbg !26
  store half %828, ptr addrspace(1) %address.local1328, align 2, !dbg !26
  %address.local1334 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 6, !dbg !26
  store half %827, ptr addrspace(1) %address.local1334, align 2, !dbg !26
  %address.local1340 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 32, !dbg !26
  store half %813, ptr addrspace(1) %address.local1340, align 2, !dbg !26
  %address.local1346 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 34, !dbg !26
  store half %812, ptr addrspace(1) %address.local1346, align 2, !dbg !26
  %address.local1352 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 36, !dbg !26
  store half %811, ptr addrspace(1) %address.local1352, align 2, !dbg !26
  %address.local1358 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 38, !dbg !26
  store half %810, ptr addrspace(1) %address.local1358, align 2, !dbg !26
  %address.local1364 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 64, !dbg !26
  store half %796, ptr addrspace(1) %address.local1364, align 2, !dbg !26
  %address.local1370 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 66, !dbg !26
  store half %795, ptr addrspace(1) %address.local1370, align 2, !dbg !26
  %address.local1376 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 68, !dbg !26
  store half %794, ptr addrspace(1) %address.local1376, align 2, !dbg !26
  %address.local1382 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 70, !dbg !26
  store half %793, ptr addrspace(1) %address.local1382, align 2, !dbg !26
  %address.local1388 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 96, !dbg !26
  store half %779, ptr addrspace(1) %address.local1388, align 2, !dbg !26
  %address.local1394 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 98, !dbg !26
  store half %778, ptr addrspace(1) %address.local1394, align 2, !dbg !26
  %address.local1400 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 100, !dbg !26
  store half %777, ptr addrspace(1) %address.local1400, align 2, !dbg !26
  %address.local1403 = or i32 %address.local1301, 51, !dbg !26
  %address.local1404 = zext nneg i32 %address.local1403 to i64, !dbg !26
  %address.local1405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1312, i64 %address.local1404, !dbg !26
  store half %776, ptr addrspace(1) %address.local1405, align 2, !dbg !26
  %address.local1411 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 128, !dbg !26
  store half %762, ptr addrspace(1) %address.local1411, align 2, !dbg !26
  %address.local1417 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 130, !dbg !26
  store half %761, ptr addrspace(1) %address.local1417, align 2, !dbg !26
  %address.local1423 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 132, !dbg !26
  store half %760, ptr addrspace(1) %address.local1423, align 2, !dbg !26
  %address.local1429 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 134, !dbg !26
  store half %759, ptr addrspace(1) %address.local1429, align 2, !dbg !26
  %address.local1435 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 160, !dbg !26
  store half %745, ptr addrspace(1) %address.local1435, align 2, !dbg !26
  %address.local1441 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 162, !dbg !26
  store half %744, ptr addrspace(1) %address.local1441, align 2, !dbg !26
  %address.local1447 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 164, !dbg !26
  store half %743, ptr addrspace(1) %address.local1447, align 2, !dbg !26
  %address.local1453 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 166, !dbg !26
  store half %742, ptr addrspace(1) %address.local1453, align 2, !dbg !26
  %address.local1459 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 192, !dbg !26
  store half %728, ptr addrspace(1) %address.local1459, align 2, !dbg !26
  %address.local1465 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 194, !dbg !26
  store half %727, ptr addrspace(1) %address.local1465, align 2, !dbg !26
  %address.local1471 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 196, !dbg !26
  store half %726, ptr addrspace(1) %address.local1471, align 2, !dbg !26
  %address.local1477 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 198, !dbg !26
  store half %725, ptr addrspace(1) %address.local1477, align 2, !dbg !26
  %address.local1483 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 224, !dbg !26
  store half %711, ptr addrspace(1) %address.local1483, align 2, !dbg !26
  %address.local1489 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 226, !dbg !26
  store half %710, ptr addrspace(1) %address.local1489, align 2, !dbg !26
  %address.local1495 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1316, i64 228, !dbg !26
  store half %709, ptr addrspace(1) %address.local1495, align 2, !dbg !26
  %address.local1498 = or i32 %address.local1301, 115, !dbg !26
  %address.local1499 = zext nneg i32 %address.local1498 to i64, !dbg !26
  %address.local1500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1312, i64 %address.local1499, !dbg !26
  store half %708, ptr addrspace(1) %address.local1500, align 2, !dbg !26
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) captures(none), ptr addrspace(3) captures(none), i32 immarg, i32 immarg, i32 immarg) #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #4

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #6

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #6 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind memory(none) }
attributes #10 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_input.mlir", directory: "3rd/deepgengraph/test")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 22, type: !4, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 22, column: 3, scope: !3)
!8 = !DILocation(line: 48, column: 13, scope: !3)
!9 = !{i32 0, i32 256}
!10 = !DILocation(line: 51, column: 13, scope: !3)
!11 = !DILocation(line: 59, column: 15, scope: !3)
!12 = !DILocation(line: 62, column: 15, scope: !3)
!13 = !DILocation(line: 63, column: 15, scope: !3)
!14 = !DILocation(line: 65, column: 17, scope: !3)
!15 = !DILocation(line: 60, column: 15, scope: !3)
!16 = !DILocation(line: 61, column: 15, scope: !3)
!17 = !DILocation(line: 66, column: 17, scope: !3)
!18 = !DILocation(line: 67, column: 17, scope: !3)
!19 = !DILocation(line: 75, column: 15, scope: !3)
!20 = !DILocation(line: 76, column: 15, scope: !3)
!21 = !DILocation(line: 77, column: 15, scope: !3)
!22 = !DILocation(line: 78, column: 15, scope: !3)
!23 = !DILocation(line: 79, column: 15, scope: !3)
!24 = !DILocation(line: 84, column: 13, scope: !3)
!25 = !DILocation(line: 85, column: 13, scope: !3)
!26 = !DILocation(line: 86, column: 7, scope: !3)
