; ModuleID = '3rd/deepgengraph/test/test_input.mlir'
source_filename = "3rd/deepgengraph/test/test_input.mlir"

@async_lds_staging_0 = private addrspace(3) global [4096 x half] undef, align 16
@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !29 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %.lobit = ashr i32 %5, 31, !dbg !12
  %9 = xor i32 %5, %.lobit, !dbg !12
  %10 = sdiv i32 %9, 64, !dbg !12
  %.lobit564 = ashr i32 %5, 31, !dbg !12
  %11 = xor i32 %10, %.lobit564, !dbg !12
  %12 = sext i32 %11 to i64, !dbg !12
  %13 = add nsw i64 %8, %12, !dbg !12
  %14 = and i64 %13, 31, !dbg !12
  %15 = shl nsw i64 %6, 6, !dbg !12
  %16 = mul nsw i64 %12, -4096, !dbg !12
  %17 = add nsw i64 %15, %16, !dbg !12
  %18 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !31
  %19 = zext nneg i32 %18 to i64, !dbg !12
  %.idx = shl nuw nsw i64 %14, 20, !dbg !12
  %20 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx565 = shl nsw i64 %17, 8, !dbg !12
  %21 = getelementptr i8, ptr addrspace(1) %20, i64 %.idx565, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %18) #8, !dbg !12, !range !31, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local577 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local578 = and i64 %address.local577, 384, !dbg !12
  %address.local579 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local578, !dbg !12
  %address.local580 = ptrtoint ptr addrspace(3) %address.local579 to i64, !dbg !12
  %address.local581 = trunc i64 %address.local580 to i32, !dbg !12
  %22 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local581), !dbg !12
  %23 = zext i32 %22 to i64, !dbg !12
  %24 = inttoptr i64 %23 to ptr addrspace(3), !dbg !12
  %address.local582 = lshr i32 %address.thread, 5, !dbg !12
  %address.local583 = and i32 %address.local582, 4, !dbg !12
  %address.local584 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local585 = and i32 %address.local584, 2, !dbg !12
  %address.local586 = or disjoint i32 %address.local583, %address.local585, !dbg !12
  %address.local587 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local588 = and i32 %address.local587, 8064, !dbg !12
  %address.local589 = or disjoint i32 %address.local586, %address.local588, !dbg !12
  %address.local590 = zext nneg i32 %address.local589 to i64, !dbg !12
  %address.local591 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local590, !dbg !12
  %dma.source.bits = ptrtoint ptr addrspace(1) %address.local591 to i64, !dbg !12
  %dma.source.opaque = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #9, !dbg !12
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %24, i32 4, i32 0, i32 0), !dbg !12
  %address.local592 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local593 = and i64 %address.local592, 384, !dbg !12
  %address.local594 = or disjoint i64 %address.local593, 512, !dbg !12
  %address.local595 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local594, !dbg !12
  %address.local596 = ptrtoint ptr addrspace(3) %address.local595 to i64, !dbg !12
  %address.local597 = trunc i64 %address.local596 to i32, !dbg !12
  %25 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local597), !dbg !12
  %26 = zext i32 %25 to i64, !dbg !12
  %27 = inttoptr i64 %26 to ptr addrspace(3), !dbg !12
  %address.local598 = lshr i32 %address.thread, 5, !dbg !12
  %address.local599 = and i32 %address.local598, 4, !dbg !12
  %address.local600 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local601 = and i32 %address.local600, 2, !dbg !12
  %address.local602 = or disjoint i32 %address.local599, %address.local601, !dbg !12
  %address.local603 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local604 = and i32 %address.local603, 8064, !dbg !12
  %address.local605 = or disjoint i32 %address.local602, %address.local604, !dbg !12
  %address.local606 = zext nneg i32 %address.local605 to i64, !dbg !12
  %address.local607 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local606, !dbg !12
  %address.local608 = getelementptr i8, ptr addrspace(1) %address.local607, i64 16, !dbg !12
  %dma.source.bits1529 = ptrtoint ptr addrspace(1) %address.local608 to i64, !dbg !12
  %dma.source.opaque1530 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1529) #9, !dbg !12
  %dma.source.ptr1531 = inttoptr i64 %dma.source.opaque1530 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1531, ptr addrspace(3) %27, i32 4, i32 0, i32 0), !dbg !12
  %address.local609 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local610 = and i64 %address.local609, 384, !dbg !12
  %address.local611 = or disjoint i64 %address.local610, 1024, !dbg !12
  %address.local612 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local611, !dbg !12
  %address.local613 = ptrtoint ptr addrspace(3) %address.local612 to i64, !dbg !12
  %address.local614 = trunc i64 %address.local613 to i32, !dbg !12
  %28 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local614), !dbg !12
  %29 = zext i32 %28 to i64, !dbg !12
  %30 = inttoptr i64 %29 to ptr addrspace(3), !dbg !12
  %address.local615 = lshr i32 %address.thread, 5, !dbg !12
  %address.local616 = and i32 %address.local615, 4, !dbg !12
  %address.local617 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local618 = and i32 %address.local617, 2, !dbg !12
  %address.local619 = or disjoint i32 %address.local616, %address.local618, !dbg !12
  %address.local620 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local621 = and i32 %address.local620, 8064, !dbg !12
  %address.local622 = or disjoint i32 %address.local619, %address.local621, !dbg !12
  %address.local623 = zext nneg i32 %address.local622 to i64, !dbg !12
  %address.local624 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local623, !dbg !12
  %address.local625 = getelementptr i8, ptr addrspace(1) %address.local624, i64 32, !dbg !12
  %dma.source.bits1532 = ptrtoint ptr addrspace(1) %address.local625 to i64, !dbg !12
  %dma.source.opaque1533 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1532) #9, !dbg !12
  %dma.source.ptr1534 = inttoptr i64 %dma.source.opaque1533 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1534, ptr addrspace(3) %30, i32 4, i32 0, i32 0), !dbg !12
  %address.local626 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local627 = and i64 %address.local626, 384, !dbg !12
  %address.local628 = or disjoint i64 %address.local627, 1536, !dbg !12
  %address.local629 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local628, !dbg !12
  %address.local630 = ptrtoint ptr addrspace(3) %address.local629 to i64, !dbg !12
  %address.local631 = trunc i64 %address.local630 to i32, !dbg !12
  %31 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local631), !dbg !12
  %32 = zext i32 %31 to i64, !dbg !12
  %33 = inttoptr i64 %32 to ptr addrspace(3), !dbg !12
  %address.local632 = lshr i32 %address.thread, 5, !dbg !12
  %address.local633 = and i32 %address.local632, 4, !dbg !12
  %address.local634 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local635 = and i32 %address.local634, 2, !dbg !12
  %address.local636 = or disjoint i32 %address.local633, %address.local635, !dbg !12
  %address.local637 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local638 = and i32 %address.local637, 8064, !dbg !12
  %address.local639 = or disjoint i32 %address.local636, %address.local638, !dbg !12
  %address.local640 = zext nneg i32 %address.local639 to i64, !dbg !12
  %address.local641 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local640, !dbg !12
  %address.local642 = getelementptr i8, ptr addrspace(1) %address.local641, i64 48, !dbg !12
  %dma.source.bits1535 = ptrtoint ptr addrspace(1) %address.local642 to i64, !dbg !12
  %dma.source.opaque1536 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1535) #9, !dbg !12
  %dma.source.ptr1537 = inttoptr i64 %dma.source.opaque1536 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1537, ptr addrspace(3) %33, i32 4, i32 0, i32 0), !dbg !12
  %address.local643 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local644 = and i64 %address.local643, 384, !dbg !12
  %address.local645 = or disjoint i64 %address.local644, 2048, !dbg !12
  %address.local646 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local645, !dbg !12
  %address.local647 = ptrtoint ptr addrspace(3) %address.local646 to i64, !dbg !12
  %address.local648 = trunc i64 %address.local647 to i32, !dbg !12
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local648), !dbg !12
  %35 = zext i32 %34 to i64, !dbg !12
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !12
  %address.local649 = lshr i32 %address.thread, 5, !dbg !12
  %address.local650 = and i32 %address.local649, 4, !dbg !12
  %address.local651 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local652 = and i32 %address.local651, 2, !dbg !12
  %address.local653 = or disjoint i32 %address.local650, %address.local652, !dbg !12
  %address.local654 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local655 = and i32 %address.local654, 8064, !dbg !12
  %address.local656 = or disjoint i32 %address.local653, %address.local655, !dbg !12
  %address.local657 = zext nneg i32 %address.local656 to i64, !dbg !12
  %address.local658 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local657, !dbg !12
  %address.local659 = getelementptr i8, ptr addrspace(1) %address.local658, i64 64, !dbg !12
  %dma.source.bits1538 = ptrtoint ptr addrspace(1) %address.local659 to i64, !dbg !12
  %dma.source.opaque1539 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1538) #9, !dbg !12
  %dma.source.ptr1540 = inttoptr i64 %dma.source.opaque1539 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1540, ptr addrspace(3) %36, i32 4, i32 0, i32 0), !dbg !12
  %address.local660 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local661 = and i64 %address.local660, 384, !dbg !12
  %address.local662 = or disjoint i64 %address.local661, 2560, !dbg !12
  %address.local663 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local662, !dbg !12
  %address.local664 = ptrtoint ptr addrspace(3) %address.local663 to i64, !dbg !12
  %address.local665 = trunc i64 %address.local664 to i32, !dbg !12
  %37 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local665), !dbg !12
  %38 = zext i32 %37 to i64, !dbg !12
  %39 = inttoptr i64 %38 to ptr addrspace(3), !dbg !12
  %address.local666 = lshr i32 %address.thread, 5, !dbg !12
  %address.local667 = and i32 %address.local666, 4, !dbg !12
  %address.local668 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local669 = and i32 %address.local668, 2, !dbg !12
  %address.local670 = or disjoint i32 %address.local667, %address.local669, !dbg !12
  %address.local671 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local672 = and i32 %address.local671, 8064, !dbg !12
  %address.local673 = or disjoint i32 %address.local670, %address.local672, !dbg !12
  %address.local674 = zext nneg i32 %address.local673 to i64, !dbg !12
  %address.local675 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local674, !dbg !12
  %address.local676 = getelementptr i8, ptr addrspace(1) %address.local675, i64 80, !dbg !12
  %dma.source.bits1541 = ptrtoint ptr addrspace(1) %address.local676 to i64, !dbg !12
  %dma.source.opaque1542 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1541) #9, !dbg !12
  %dma.source.ptr1543 = inttoptr i64 %dma.source.opaque1542 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1543, ptr addrspace(3) %39, i32 4, i32 0, i32 0), !dbg !12
  %address.local677 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local678 = and i64 %address.local677, 384, !dbg !12
  %address.local679 = or disjoint i64 %address.local678, 3072, !dbg !12
  %address.local680 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local679, !dbg !12
  %address.local681 = ptrtoint ptr addrspace(3) %address.local680 to i64, !dbg !12
  %address.local682 = trunc i64 %address.local681 to i32, !dbg !12
  %40 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local682), !dbg !12
  %41 = zext i32 %40 to i64, !dbg !12
  %42 = inttoptr i64 %41 to ptr addrspace(3), !dbg !12
  %address.local683 = lshr i32 %address.thread, 5, !dbg !12
  %address.local684 = and i32 %address.local683, 4, !dbg !12
  %address.local685 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local686 = and i32 %address.local685, 2, !dbg !12
  %address.local687 = or disjoint i32 %address.local684, %address.local686, !dbg !12
  %address.local688 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local689 = and i32 %address.local688, 8064, !dbg !12
  %address.local690 = or disjoint i32 %address.local687, %address.local689, !dbg !12
  %address.local691 = zext nneg i32 %address.local690 to i64, !dbg !12
  %address.local692 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local691, !dbg !12
  %address.local693 = getelementptr i8, ptr addrspace(1) %address.local692, i64 96, !dbg !12
  %dma.source.bits1544 = ptrtoint ptr addrspace(1) %address.local693 to i64, !dbg !12
  %dma.source.opaque1545 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1544) #9, !dbg !12
  %dma.source.ptr1546 = inttoptr i64 %dma.source.opaque1545 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1546, ptr addrspace(3) %42, i32 4, i32 0, i32 0), !dbg !12
  %address.local694 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local695 = and i64 %address.local694, 384, !dbg !12
  %address.local696 = or disjoint i64 %address.local695, 3584, !dbg !12
  %address.local697 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local696, !dbg !12
  %address.local698 = ptrtoint ptr addrspace(3) %address.local697 to i64, !dbg !12
  %address.local699 = trunc i64 %address.local698 to i32, !dbg !12
  %43 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local699), !dbg !12
  %44 = zext i32 %43 to i64, !dbg !12
  %45 = inttoptr i64 %44 to ptr addrspace(3), !dbg !12
  %address.local700 = lshr i32 %address.thread, 5, !dbg !12
  %address.local701 = and i32 %address.local700, 4, !dbg !12
  %address.local702 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local703 = and i32 %address.local702, 2, !dbg !12
  %address.local704 = or disjoint i32 %address.local701, %address.local703, !dbg !12
  %address.local705 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local706 = and i32 %address.local705, 8064, !dbg !12
  %address.local707 = or disjoint i32 %address.local704, %address.local706, !dbg !12
  %address.local708 = zext nneg i32 %address.local707 to i64, !dbg !12
  %address.local709 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local708, !dbg !12
  %address.local710 = getelementptr i8, ptr addrspace(1) %address.local709, i64 112, !dbg !12
  %dma.source.bits1547 = ptrtoint ptr addrspace(1) %address.local710 to i64, !dbg !12
  %dma.source.opaque1548 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1547) #9, !dbg !12
  %dma.source.ptr1549 = inttoptr i64 %dma.source.opaque1548 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1549, ptr addrspace(3) %45, i32 4, i32 0, i32 0), !dbg !12
  %address.local711 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local712 = and i64 %address.local711, 384, !dbg !12
  %address.local713 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local712, !dbg !12
  %address.local714 = getelementptr i8, ptr addrspace(3) %address.local713, i64 8192, !dbg !12
  %address.local715 = ptrtoint ptr addrspace(3) %address.local714 to i64, !dbg !12
  %address.local716 = trunc i64 %address.local715 to i32, !dbg !12
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local716), !dbg !12
  %47 = zext i32 %46 to i64, !dbg !12
  %48 = inttoptr i64 %47 to ptr addrspace(3), !dbg !12
  %address.local717 = lshr i32 %address.thread, 5, !dbg !12
  %address.local718 = and i32 %address.local717, 4, !dbg !12
  %address.local719 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local720 = and i32 %address.local719, 2, !dbg !12
  %address.local721 = or disjoint i32 %address.local718, %address.local720, !dbg !12
  %address.local722 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local723 = and i32 %address.local722, 8064, !dbg !12
  %address.local724 = or disjoint i32 %address.local721, %address.local723, !dbg !12
  %address.local725 = zext nneg i32 %address.local724 to i64, !dbg !12
  %address.local726 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local725, !dbg !12
  %address.local727 = getelementptr i8, ptr addrspace(1) %address.local726, i64 128, !dbg !12
  %dma.source.bits1550 = ptrtoint ptr addrspace(1) %address.local727 to i64, !dbg !12
  %dma.source.opaque1551 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1550) #9, !dbg !12
  %dma.source.ptr1552 = inttoptr i64 %dma.source.opaque1551 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1552, ptr addrspace(3) %48, i32 4, i32 0, i32 0), !dbg !12
  %address.local728 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local729 = and i64 %address.local728, 384, !dbg !12
  %address.local730 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local729, !dbg !12
  %address.local731 = getelementptr i8, ptr addrspace(3) %address.local730, i64 9216, !dbg !12
  %address.local732 = ptrtoint ptr addrspace(3) %address.local731 to i64, !dbg !12
  %address.local733 = trunc i64 %address.local732 to i32, !dbg !12
  %49 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local733), !dbg !12
  %50 = zext i32 %49 to i64, !dbg !12
  %51 = inttoptr i64 %50 to ptr addrspace(3), !dbg !12
  %address.local734 = lshr i32 %address.thread, 5, !dbg !12
  %address.local735 = and i32 %address.local734, 4, !dbg !12
  %address.local736 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local737 = and i32 %address.local736, 2, !dbg !12
  %address.local738 = or disjoint i32 %address.local735, %address.local737, !dbg !12
  %address.local739 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local740 = and i32 %address.local739, 8064, !dbg !12
  %address.local741 = or disjoint i32 %address.local738, %address.local740, !dbg !12
  %address.local742 = zext nneg i32 %address.local741 to i64, !dbg !12
  %address.local743 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local742, !dbg !12
  %address.local744 = getelementptr i8, ptr addrspace(1) %address.local743, i64 144, !dbg !12
  %dma.source.bits1553 = ptrtoint ptr addrspace(1) %address.local744 to i64, !dbg !12
  %dma.source.opaque1554 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1553) #9, !dbg !12
  %dma.source.ptr1555 = inttoptr i64 %dma.source.opaque1554 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1555, ptr addrspace(3) %51, i32 4, i32 0, i32 0), !dbg !12
  %address.local745 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local746 = and i64 %address.local745, 384, !dbg !12
  %address.local747 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local746, !dbg !12
  %address.local748 = getelementptr i8, ptr addrspace(3) %address.local747, i64 10240, !dbg !12
  %address.local749 = ptrtoint ptr addrspace(3) %address.local748 to i64, !dbg !12
  %address.local750 = trunc i64 %address.local749 to i32, !dbg !12
  %52 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local750), !dbg !12
  %53 = zext i32 %52 to i64, !dbg !12
  %54 = inttoptr i64 %53 to ptr addrspace(3), !dbg !12
  %address.local751 = lshr i32 %address.thread, 5, !dbg !12
  %address.local752 = and i32 %address.local751, 4, !dbg !12
  %address.local753 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local754 = and i32 %address.local753, 2, !dbg !12
  %address.local755 = or disjoint i32 %address.local752, %address.local754, !dbg !12
  %address.local756 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local757 = and i32 %address.local756, 8064, !dbg !12
  %address.local758 = or disjoint i32 %address.local755, %address.local757, !dbg !12
  %address.local759 = zext nneg i32 %address.local758 to i64, !dbg !12
  %address.local760 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local759, !dbg !12
  %address.local761 = getelementptr i8, ptr addrspace(1) %address.local760, i64 160, !dbg !12
  %dma.source.bits1556 = ptrtoint ptr addrspace(1) %address.local761 to i64, !dbg !12
  %dma.source.opaque1557 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1556) #9, !dbg !12
  %dma.source.ptr1558 = inttoptr i64 %dma.source.opaque1557 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1558, ptr addrspace(3) %54, i32 4, i32 0, i32 0), !dbg !12
  %address.local762 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local763 = and i64 %address.local762, 384, !dbg !12
  %address.local764 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local763, !dbg !12
  %address.local765 = getelementptr i8, ptr addrspace(3) %address.local764, i64 11264, !dbg !12
  %address.local766 = ptrtoint ptr addrspace(3) %address.local765 to i64, !dbg !12
  %address.local767 = trunc i64 %address.local766 to i32, !dbg !12
  %55 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local767), !dbg !12
  %56 = zext i32 %55 to i64, !dbg !12
  %57 = inttoptr i64 %56 to ptr addrspace(3), !dbg !12
  %address.local768 = lshr i32 %address.thread, 5, !dbg !12
  %address.local769 = and i32 %address.local768, 4, !dbg !12
  %address.local770 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local771 = and i32 %address.local770, 2, !dbg !12
  %address.local772 = or disjoint i32 %address.local769, %address.local771, !dbg !12
  %address.local773 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local774 = and i32 %address.local773, 8064, !dbg !12
  %address.local775 = or disjoint i32 %address.local772, %address.local774, !dbg !12
  %address.local776 = zext nneg i32 %address.local775 to i64, !dbg !12
  %address.local777 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local776, !dbg !12
  %address.local778 = getelementptr i8, ptr addrspace(1) %address.local777, i64 176, !dbg !12
  %dma.source.bits1559 = ptrtoint ptr addrspace(1) %address.local778 to i64, !dbg !12
  %dma.source.opaque1560 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1559) #9, !dbg !12
  %dma.source.ptr1561 = inttoptr i64 %dma.source.opaque1560 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1561, ptr addrspace(3) %57, i32 4, i32 0, i32 0), !dbg !12
  %address.local779 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local780 = and i64 %address.local779, 384, !dbg !12
  %address.local781 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local780, !dbg !12
  %address.local782 = getelementptr i8, ptr addrspace(3) %address.local781, i64 12288, !dbg !12
  %address.local783 = ptrtoint ptr addrspace(3) %address.local782 to i64, !dbg !12
  %address.local784 = trunc i64 %address.local783 to i32, !dbg !12
  %58 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local784), !dbg !12
  %59 = zext i32 %58 to i64, !dbg !12
  %60 = inttoptr i64 %59 to ptr addrspace(3), !dbg !12
  %address.local785 = lshr i32 %address.thread, 5, !dbg !12
  %address.local786 = and i32 %address.local785, 4, !dbg !12
  %address.local787 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local788 = and i32 %address.local787, 2, !dbg !12
  %address.local789 = or disjoint i32 %address.local786, %address.local788, !dbg !12
  %address.local790 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local791 = and i32 %address.local790, 8064, !dbg !12
  %address.local792 = or disjoint i32 %address.local789, %address.local791, !dbg !12
  %address.local793 = zext nneg i32 %address.local792 to i64, !dbg !12
  %address.local794 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local793, !dbg !12
  %address.local795 = getelementptr i8, ptr addrspace(1) %address.local794, i64 192, !dbg !12
  %dma.source.bits1562 = ptrtoint ptr addrspace(1) %address.local795 to i64, !dbg !12
  %dma.source.opaque1563 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1562) #9, !dbg !12
  %dma.source.ptr1564 = inttoptr i64 %dma.source.opaque1563 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1564, ptr addrspace(3) %60, i32 4, i32 0, i32 0), !dbg !12
  %address.local796 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local797 = and i64 %address.local796, 384, !dbg !12
  %address.local798 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local797, !dbg !12
  %address.local799 = getelementptr i8, ptr addrspace(3) %address.local798, i64 13312, !dbg !12
  %address.local800 = ptrtoint ptr addrspace(3) %address.local799 to i64, !dbg !12
  %address.local801 = trunc i64 %address.local800 to i32, !dbg !12
  %61 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local801), !dbg !12
  %62 = zext i32 %61 to i64, !dbg !12
  %63 = inttoptr i64 %62 to ptr addrspace(3), !dbg !12
  %address.local802 = lshr i32 %address.thread, 5, !dbg !12
  %address.local803 = and i32 %address.local802, 4, !dbg !12
  %address.local804 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local805 = and i32 %address.local804, 2, !dbg !12
  %address.local806 = or disjoint i32 %address.local803, %address.local805, !dbg !12
  %address.local807 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local808 = and i32 %address.local807, 8064, !dbg !12
  %address.local809 = or disjoint i32 %address.local806, %address.local808, !dbg !12
  %address.local810 = zext nneg i32 %address.local809 to i64, !dbg !12
  %address.local811 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local810, !dbg !12
  %address.local812 = getelementptr i8, ptr addrspace(1) %address.local811, i64 208, !dbg !12
  %dma.source.bits1565 = ptrtoint ptr addrspace(1) %address.local812 to i64, !dbg !12
  %dma.source.opaque1566 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1565) #9, !dbg !12
  %dma.source.ptr1567 = inttoptr i64 %dma.source.opaque1566 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1567, ptr addrspace(3) %63, i32 4, i32 0, i32 0), !dbg !12
  %address.local813 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local814 = and i64 %address.local813, 384, !dbg !12
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local814, !dbg !12
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local815, i64 14336, !dbg !12
  %address.local817 = ptrtoint ptr addrspace(3) %address.local816 to i64, !dbg !12
  %address.local818 = trunc i64 %address.local817 to i32, !dbg !12
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local818), !dbg !12
  %65 = zext i32 %64 to i64, !dbg !12
  %66 = inttoptr i64 %65 to ptr addrspace(3), !dbg !12
  %address.local819 = lshr i32 %address.thread, 5, !dbg !12
  %address.local820 = and i32 %address.local819, 4, !dbg !12
  %address.local821 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local822 = and i32 %address.local821, 2, !dbg !12
  %address.local823 = or disjoint i32 %address.local820, %address.local822, !dbg !12
  %address.local824 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local825 = and i32 %address.local824, 8064, !dbg !12
  %address.local826 = or disjoint i32 %address.local823, %address.local825, !dbg !12
  %address.local827 = zext nneg i32 %address.local826 to i64, !dbg !12
  %address.local828 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local827, !dbg !12
  %address.local829 = getelementptr i8, ptr addrspace(1) %address.local828, i64 224, !dbg !12
  %dma.source.bits1568 = ptrtoint ptr addrspace(1) %address.local829 to i64, !dbg !12
  %dma.source.opaque1569 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1568) #9, !dbg !12
  %dma.source.ptr1570 = inttoptr i64 %dma.source.opaque1569 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1570, ptr addrspace(3) %66, i32 4, i32 0, i32 0), !dbg !12
  %address.local830 = shl nuw nsw i64 %address.local, 1, !dbg !12
  %address.local831 = and i64 %address.local830, 384, !dbg !12
  %address.local832 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local831, !dbg !12
  %address.local833 = getelementptr i8, ptr addrspace(3) %address.local832, i64 15360, !dbg !12
  %address.local834 = ptrtoint ptr addrspace(3) %address.local833 to i64, !dbg !12
  %address.local835 = trunc i64 %address.local834 to i32, !dbg !12
  %67 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local835), !dbg !12
  %68 = zext i32 %67 to i64, !dbg !12
  %69 = inttoptr i64 %68 to ptr addrspace(3), !dbg !12
  %address.local836 = lshr i32 %address.thread, 5, !dbg !12
  %address.local837 = and i32 %address.local836, 4, !dbg !12
  %address.local838 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local839 = and i32 %address.local838, 2, !dbg !12
  %address.local840 = or disjoint i32 %address.local837, %address.local839, !dbg !12
  %address.local841 = shl nuw nsw i32 %address.thread, 6, !dbg !12
  %address.local842 = and i32 %address.local841, 8064, !dbg !12
  %address.local843 = or disjoint i32 %address.local840, %address.local842, !dbg !12
  %address.local844 = zext nneg i32 %address.local843 to i64, !dbg !12
  %address.local845 = getelementptr [2 x i8], ptr addrspace(1) %21, i64 %address.local844, !dbg !12
  %address.local846 = getelementptr i8, ptr addrspace(1) %address.local845, i64 240, !dbg !12
  %dma.source.bits1571 = ptrtoint ptr addrspace(1) %address.local846 to i64, !dbg !12
  %dma.source.opaque1572 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1571) #9, !dbg !12
  %dma.source.ptr1573 = inttoptr i64 %dma.source.opaque1572 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1573, ptr addrspace(3) %69, i32 4, i32 0, i32 0), !dbg !12
  call void @llvm.amdgcn.asyncmark(), !dbg !12
  %70 = shl nsw i64 %6, 6, !dbg !13
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local847 = and i32 %address.thread, 192, !dbg !7
  %address.local848 = shl nuw nsw i32 %address.thread, 2, !dbg !7
  %address.local849 = and i32 %address.local848, 60, !dbg !7
  %address.local850 = or disjoint i32 %address.local847, %address.local849, !dbg !7
  %address.local851 = shl nuw nsw i32 %address.thread, 4, !dbg !7
  %address.local852 = and i32 %address.local851, 768, !dbg !7
  %address.local853 = or disjoint i32 %address.local850, %address.local852, !dbg !7
  %address.local854 = zext nneg i32 %address.local853 to i64, !dbg !7
  %address.local855 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local854, !dbg !7
  %71 = load <4 x half>, ptr addrspace(3) %address.local855, align 2, !dbg !7
  %address.local856 = and i32 %address.thread, 192, !dbg !7
  %address.local857 = shl nuw nsw i32 %address.thread, 2, !dbg !7
  %address.local858 = and i32 %address.local857, 60, !dbg !7
  %address.local859 = or disjoint i32 %address.local856, %address.local858, !dbg !7
  %address.local860 = shl nuw nsw i32 %address.thread, 4, !dbg !7
  %address.local861 = and i32 %address.local860, 768, !dbg !7
  %address.local862 = or disjoint i32 %address.local859, %address.local861, !dbg !7
  %address.local863 = zext nneg i32 %address.local862 to i64, !dbg !7
  %address.local864 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local863, !dbg !7
  %address.local865 = getelementptr i8, ptr addrspace(3) %address.local864, i64 2048, !dbg !7
  %72 = load <4 x half>, ptr addrspace(3) %address.local865, align 2, !dbg !7
  %address.local866 = and i32 %address.thread, 192, !dbg !7
  %address.local867 = shl nuw nsw i32 %address.thread, 2, !dbg !7
  %address.local868 = and i32 %address.local867, 60, !dbg !7
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !7
  %address.local870 = shl nuw nsw i32 %address.thread, 4, !dbg !7
  %address.local871 = and i32 %address.local870, 768, !dbg !7
  %address.local872 = or disjoint i32 %address.local869, %address.local871, !dbg !7
  %address.local873 = zext nneg i32 %address.local872 to i64, !dbg !7
  %address.local874 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local873, !dbg !7
  %address.local875 = getelementptr i8, ptr addrspace(3) %address.local874, i64 4096, !dbg !7
  %73 = load <4 x half>, ptr addrspace(3) %address.local875, align 2, !dbg !7
  %address.local876 = and i32 %address.thread, 192, !dbg !7
  %address.local877 = shl nuw nsw i32 %address.thread, 2, !dbg !7
  %address.local878 = and i32 %address.local877, 60, !dbg !7
  %address.local879 = or disjoint i32 %address.local876, %address.local878, !dbg !7
  %address.local880 = shl nuw nsw i32 %address.thread, 4, !dbg !7
  %address.local881 = and i32 %address.local880, 768, !dbg !7
  %address.local882 = or disjoint i32 %address.local879, %address.local881, !dbg !7
  %address.local883 = zext nneg i32 %address.local882 to i64, !dbg !7
  %address.local884 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local883, !dbg !7
  %address.local885 = getelementptr i8, ptr addrspace(3) %address.local884, i64 6144, !dbg !7
  %74 = load <4 x half>, ptr addrspace(3) %address.local885, align 2, !dbg !7
  %75 = fmul <4 x half> %71, splat (half 0xH3015), !dbg !7
  %address.local886 = and i32 %address.thread, 192, !dbg !7
  %address.local887 = shl nuw nsw i32 %address.thread, 2, !dbg !7
  %address.local888 = and i32 %address.local887, 60, !dbg !7
  %address.local889 = or disjoint i32 %address.local886, %address.local888, !dbg !7
  %address.local890 = shl nuw nsw i32 %address.thread, 4, !dbg !7
  %address.local891 = and i32 %address.local890, 768, !dbg !7
  %address.local892 = or disjoint i32 %address.local889, %address.local891, !dbg !7
  %address.local893 = zext nneg i32 %address.local892 to i64, !dbg !7
  %address.local894 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local893, !dbg !7
  %address.local895 = getelementptr i8, ptr addrspace(3) %address.local894, i64 8192, !dbg !7
  %76 = load <4 x half>, ptr addrspace(3) %address.local895, align 2, !dbg !7
  %77 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !7
  %address.local896 = and i32 %address.thread, 192, !dbg !7
  %address.local897 = shl nuw nsw i32 %address.thread, 2, !dbg !7
  %address.local898 = and i32 %address.local897, 60, !dbg !7
  %address.local899 = or disjoint i32 %address.local896, %address.local898, !dbg !7
  %address.local900 = shl nuw nsw i32 %address.thread, 4, !dbg !7
  %address.local901 = and i32 %address.local900, 768, !dbg !7
  %address.local902 = or disjoint i32 %address.local899, %address.local901, !dbg !7
  %address.local903 = zext nneg i32 %address.local902 to i64, !dbg !7
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !7
  %address.local905 = getelementptr i8, ptr addrspace(3) %address.local904, i64 10240, !dbg !7
  %78 = load <4 x half>, ptr addrspace(3) %address.local905, align 2, !dbg !7
  %79 = fmul <4 x half> %73, splat (half 0xH3015), !dbg !7
  %address.local906 = and i32 %address.thread, 192, !dbg !7
  %address.local907 = shl nuw nsw i32 %address.thread, 2, !dbg !7
  %address.local908 = and i32 %address.local907, 60, !dbg !7
  %address.local909 = or disjoint i32 %address.local906, %address.local908, !dbg !7
  %address.local910 = shl nuw nsw i32 %address.thread, 4, !dbg !7
  %address.local911 = and i32 %address.local910, 768, !dbg !7
  %address.local912 = or disjoint i32 %address.local909, %address.local911, !dbg !7
  %address.local913 = zext nneg i32 %address.local912 to i64, !dbg !7
  %address.local914 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local913, !dbg !7
  %address.local915 = getelementptr i8, ptr addrspace(3) %address.local914, i64 12288, !dbg !7
  %80 = load <4 x half>, ptr addrspace(3) %address.local915, align 2, !dbg !7
  %81 = fmul <4 x half> %74, splat (half 0xH3015), !dbg !7
  %82 = fmul <4 x half> %76, splat (half 0xH3015), !dbg !7
  %83 = fmul <4 x half> %78, splat (half 0xH3015), !dbg !7
  %84 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !7
  %address.local916 = and i32 %address.thread, 192, !dbg !7
  %address.local917 = shl nuw nsw i32 %address.thread, 2, !dbg !7
  %address.local918 = and i32 %address.local917, 60, !dbg !7
  %address.local919 = or disjoint i32 %address.local916, %address.local918, !dbg !7
  %address.local920 = shl nuw nsw i32 %address.thread, 4, !dbg !7
  %address.local921 = and i32 %address.local920, 768, !dbg !7
  %address.local922 = or disjoint i32 %address.local919, %address.local921, !dbg !7
  %address.local923 = zext nneg i32 %address.local922 to i64, !dbg !7
  %address.local924 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local923, !dbg !7
  %address.local925 = getelementptr i8, ptr addrspace(3) %address.local924, i64 14336, !dbg !7
  %85 = load <4 x half>, ptr addrspace(3) %address.local925, align 2, !dbg !7
  %86 = fmul <4 x half> %85, splat (half 0xH3015), !dbg !7
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %87 = and i32 %18, 16, !dbg !8
  %.not = icmp eq i32 %87, 0, !dbg !8
  %88 = and i32 %18, 32, !dbg !8
  %.not566 = icmp eq i32 %88, 0, !dbg !8
  %89 = add nsw i64 %70, %19, !dbg !16
  %90 = lshr i32 %18, 2, !dbg !16
  %91 = and i32 %90, 48, !dbg !16
  %index.compact297 = zext nneg i32 %91 to i64, !dbg !16
  %92 = add nsw i64 %89, %index.compact297, !dbg !16
  %93 = and i32 %18, 240, !dbg !16
  %94 = sub nsw i32 0, %93, !dbg !16
  %index.compact296 = sext i32 %94 to i64, !dbg !16
  %95 = add nsw i64 %92, %index.compact296, !dbg !16
  %96 = add nsw i64 %95, 1, !dbg !17
  %97 = shl nsw i64 %6, 1, !dbg !9
  %98 = add nsw i64 %97, 2, !dbg !9
  %99 = lshr i32 %18, 2, !dbg !16
  %100 = and i32 %99, 12, !dbg !16
  %index.compact55 = zext nneg i32 %100 to i64, !dbg !16
  br label %101, !dbg !9

101:                                              ; preds = %372, %4
  %102 = phi i64 [ %459, %372 ], [ 0, %4 ], !dbg !9
  %103 = phi <32 x float> [ %457, %372 ], [ zeroinitializer, %4 ], !dbg !9
  %104 = phi <1 x float> [ %458, %372 ], [ zeroinitializer, %4 ], !dbg !9
  %105 = icmp slt i64 %102, %98, !dbg !9
  br i1 %105, label %106, label %460, !dbg !9

106:                                              ; preds = %101
  %107 = lshr i64 %102, 14, !dbg !6
  %108 = add nsw i64 %107, %8, !dbg !6
  %109 = lshr i64 %102, 7, !dbg !6
  %110 = and i64 %109, 127, !dbg !6
  %111 = shl nsw i64 %102, 5, !dbg !6
  %112 = shl i64 %108, 19, !dbg !6
  %113 = and i64 %112, 16252928, !dbg !6
  %114 = shl i64 %109, 19, !dbg !6
  %115 = shl i64 %102, 12, !dbg !6
  %116 = sub i64 %115, %114, !dbg !6
  %117 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %113, !dbg !6
  %118 = getelementptr [2 x i8], ptr addrspace(1) %117, i64 %110, !dbg !6
  %119 = getelementptr [2 x i8], ptr addrspace(1) %118, i64 %116, !dbg !6
  %address.thread926 = call i32 asm sideeffect "", "=v,0"(i32 %18) #8, !dbg !6, !range !31, !frisk.register_lifetimes !1
  %address.local927 = zext nneg i32 %address.thread926 to i64, !dbg !12
  %address.local928 = shl nuw nsw i64 %address.local927, 1, !dbg !12
  %address.local929 = and i64 %address.local928, 384, !dbg !12
  %address.local930 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local929, !dbg !6
  %address.local931 = ptrtoint ptr addrspace(3) %address.local930 to i64, !dbg !6
  %address.local932 = trunc i64 %address.local931 to i32, !dbg !6
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local932), !dbg !6
  %121 = zext i32 %120 to i64, !dbg !6
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !6
  %address.local933 = shl nuw nsw i32 %address.thread926, 1, !dbg !6
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !6
  %address.local935 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local934, !dbg !6
  %dma.source.bits1574 = ptrtoint ptr addrspace(1) %address.local935 to i64, !dbg !6
  %dma.source.opaque1575 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1574) #9, !dbg !6
  %dma.source.ptr1576 = inttoptr i64 %dma.source.opaque1575 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1576, ptr addrspace(3) %122, i32 4, i32 0, i32 0), !dbg !6
  %address.local936 = shl nuw nsw i64 %address.local927, 1, !dbg !12
  %address.local937 = and i64 %address.local936, 384, !dbg !12
  %address.local938 = or disjoint i64 %address.local937, 512, !dbg !12
  %address.local939 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local938, !dbg !6
  %address.local940 = ptrtoint ptr addrspace(3) %address.local939 to i64, !dbg !6
  %address.local941 = trunc i64 %address.local940 to i32, !dbg !6
  %123 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local941), !dbg !6
  %124 = zext i32 %123 to i64, !dbg !6
  %125 = inttoptr i64 %124 to ptr addrspace(3), !dbg !6
  %address.local942 = shl nuw nsw i32 %address.thread926, 1, !dbg !6
  %address.local943 = zext nneg i32 %address.local942 to i64, !dbg !6
  %address.local944 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local943, !dbg !6
  %address.local945 = getelementptr i8, ptr addrspace(1) %address.local944, i64 1024, !dbg !6
  %dma.source.bits1577 = ptrtoint ptr addrspace(1) %address.local945 to i64, !dbg !6
  %dma.source.opaque1578 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1577) #9, !dbg !6
  %dma.source.ptr1579 = inttoptr i64 %dma.source.opaque1578 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1579, ptr addrspace(3) %125, i32 4, i32 0, i32 0), !dbg !6
  %address.local946 = shl nuw nsw i64 %address.local927, 1, !dbg !12
  %address.local947 = and i64 %address.local946, 384, !dbg !12
  %address.local948 = or disjoint i64 %address.local947, 1024, !dbg !12
  %address.local949 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local948, !dbg !6
  %address.local950 = ptrtoint ptr addrspace(3) %address.local949 to i64, !dbg !6
  %address.local951 = trunc i64 %address.local950 to i32, !dbg !6
  %126 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local951), !dbg !6
  %127 = zext i32 %126 to i64, !dbg !6
  %128 = inttoptr i64 %127 to ptr addrspace(3), !dbg !6
  %address.local952 = shl nuw nsw i32 %address.thread926, 1, !dbg !6
  %address.local953 = zext nneg i32 %address.local952 to i64, !dbg !6
  %address.local954 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local953, !dbg !6
  %address.local955 = getelementptr i8, ptr addrspace(1) %address.local954, i64 2048, !dbg !6
  %dma.source.bits1580 = ptrtoint ptr addrspace(1) %address.local955 to i64, !dbg !6
  %dma.source.opaque1581 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1580) #9, !dbg !6
  %dma.source.ptr1582 = inttoptr i64 %dma.source.opaque1581 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1582, ptr addrspace(3) %128, i32 4, i32 0, i32 0), !dbg !6
  %address.local956 = shl nuw nsw i64 %address.local927, 1, !dbg !12
  %address.local957 = and i64 %address.local956, 384, !dbg !12
  %address.local958 = or disjoint i64 %address.local957, 1536, !dbg !12
  %address.local959 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local958, !dbg !6
  %address.local960 = ptrtoint ptr addrspace(3) %address.local959 to i64, !dbg !6
  %address.local961 = trunc i64 %address.local960 to i32, !dbg !6
  %129 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local961), !dbg !6
  %130 = zext i32 %129 to i64, !dbg !6
  %131 = inttoptr i64 %130 to ptr addrspace(3), !dbg !6
  %address.local962 = shl nuw nsw i32 %address.thread926, 1, !dbg !6
  %address.local963 = zext nneg i32 %address.local962 to i64, !dbg !6
  %address.local964 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local963, !dbg !6
  %address.local965 = getelementptr i8, ptr addrspace(1) %address.local964, i64 3072, !dbg !6
  %dma.source.bits1583 = ptrtoint ptr addrspace(1) %address.local965 to i64, !dbg !6
  %dma.source.opaque1584 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1583) #9, !dbg !6
  %dma.source.ptr1585 = inttoptr i64 %dma.source.opaque1584 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1585, ptr addrspace(3) %131, i32 4, i32 0, i32 0), !dbg !6
  %address.local966 = shl nuw nsw i64 %address.local927, 1, !dbg !12
  %address.local967 = and i64 %address.local966, 384, !dbg !12
  %address.local968 = or disjoint i64 %address.local967, 2048, !dbg !12
  %address.local969 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local968, !dbg !6
  %address.local970 = ptrtoint ptr addrspace(3) %address.local969 to i64, !dbg !6
  %address.local971 = trunc i64 %address.local970 to i32, !dbg !6
  %132 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local971), !dbg !6
  %133 = zext i32 %132 to i64, !dbg !6
  %134 = inttoptr i64 %133 to ptr addrspace(3), !dbg !6
  %address.local972 = shl nuw nsw i32 %address.thread926, 1, !dbg !6
  %address.local973 = zext nneg i32 %address.local972 to i64, !dbg !6
  %address.local974 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local973, !dbg !6
  %address.local975 = getelementptr i8, ptr addrspace(1) %address.local974, i64 4096, !dbg !6
  %dma.source.bits1586 = ptrtoint ptr addrspace(1) %address.local975 to i64, !dbg !6
  %dma.source.opaque1587 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1586) #9, !dbg !6
  %dma.source.ptr1588 = inttoptr i64 %dma.source.opaque1587 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1588, ptr addrspace(3) %134, i32 4, i32 0, i32 0), !dbg !6
  %address.local976 = shl nuw nsw i64 %address.local927, 1, !dbg !12
  %address.local977 = and i64 %address.local976, 384, !dbg !12
  %address.local978 = or disjoint i64 %address.local977, 2560, !dbg !12
  %address.local979 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local978, !dbg !6
  %address.local980 = ptrtoint ptr addrspace(3) %address.local979 to i64, !dbg !6
  %address.local981 = trunc i64 %address.local980 to i32, !dbg !6
  %135 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local981), !dbg !6
  %136 = zext i32 %135 to i64, !dbg !6
  %137 = inttoptr i64 %136 to ptr addrspace(3), !dbg !6
  %address.local982 = shl nuw nsw i32 %address.thread926, 1, !dbg !6
  %address.local983 = zext nneg i32 %address.local982 to i64, !dbg !6
  %address.local984 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local983, !dbg !6
  %address.local985 = getelementptr i8, ptr addrspace(1) %address.local984, i64 5120, !dbg !6
  %dma.source.bits1589 = ptrtoint ptr addrspace(1) %address.local985 to i64, !dbg !6
  %dma.source.opaque1590 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1589) #9, !dbg !6
  %dma.source.ptr1591 = inttoptr i64 %dma.source.opaque1590 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1591, ptr addrspace(3) %137, i32 4, i32 0, i32 0), !dbg !6
  %address.local986 = shl nuw nsw i64 %address.local927, 1, !dbg !12
  %address.local987 = and i64 %address.local986, 384, !dbg !12
  %address.local988 = or disjoint i64 %address.local987, 3072, !dbg !12
  %address.local989 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local988, !dbg !6
  %address.local990 = ptrtoint ptr addrspace(3) %address.local989 to i64, !dbg !6
  %address.local991 = trunc i64 %address.local990 to i32, !dbg !6
  %138 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local991), !dbg !6
  %139 = zext i32 %138 to i64, !dbg !6
  %140 = inttoptr i64 %139 to ptr addrspace(3), !dbg !6
  %address.local992 = shl nuw nsw i32 %address.thread926, 1, !dbg !6
  %address.local993 = zext nneg i32 %address.local992 to i64, !dbg !6
  %address.local994 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local993, !dbg !6
  %address.local995 = getelementptr i8, ptr addrspace(1) %address.local994, i64 6144, !dbg !6
  %dma.source.bits1592 = ptrtoint ptr addrspace(1) %address.local995 to i64, !dbg !6
  %dma.source.opaque1593 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1592) #9, !dbg !6
  %dma.source.ptr1594 = inttoptr i64 %dma.source.opaque1593 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1594, ptr addrspace(3) %140, i32 4, i32 0, i32 0), !dbg !6
  %address.local996 = shl nuw nsw i64 %address.local927, 1, !dbg !12
  %address.local997 = and i64 %address.local996, 384, !dbg !12
  %address.local998 = or disjoint i64 %address.local997, 3584, !dbg !12
  %address.local999 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local998, !dbg !6
  %address.local1000 = ptrtoint ptr addrspace(3) %address.local999 to i64, !dbg !6
  %address.local1001 = trunc i64 %address.local1000 to i32, !dbg !6
  %141 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1001), !dbg !6
  %142 = zext i32 %141 to i64, !dbg !6
  %143 = inttoptr i64 %142 to ptr addrspace(3), !dbg !6
  %address.local1002 = shl nuw nsw i32 %address.thread926, 1, !dbg !6
  %address.local1003 = zext nneg i32 %address.local1002 to i64, !dbg !6
  %address.local1004 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local1003, !dbg !6
  %address.local1005 = getelementptr i8, ptr addrspace(1) %address.local1004, i64 7168, !dbg !6
  %dma.source.bits1595 = ptrtoint ptr addrspace(1) %address.local1005 to i64, !dbg !6
  %dma.source.opaque1596 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1595) #9, !dbg !6
  %dma.source.ptr1597 = inttoptr i64 %dma.source.opaque1596 to ptr addrspace(1), !dbg !6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1597, ptr addrspace(3) %143, i32 4, i32 0, i32 0), !dbg !6
  call void @llvm.amdgcn.asyncmark(), !dbg !6
  %144 = add nsw i64 %109, %8, !dbg !19
  %145 = shl i64 %144, 19, !dbg !19
  %146 = and i64 %145, 16252928, !dbg !19
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !6
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  br label %147, !dbg !6

147:                                              ; preds = %150, %106
  %148 = phi i64 [ %172, %150 ], [ %19, %106 ], !dbg !6
  %149 = icmp samesign ult i64 %148, 2048, !dbg !6
  br i1 %149, label %150, label %173, !dbg !6

150:                                              ; preds = %147
  %151 = shl nuw nsw i64 %148, 1, !dbg !6
  %152 = lshr i64 %148, 4, !dbg !6
  %153 = shl nuw nsw i64 %148, 1, !dbg !6
  %154 = and i64 %153, 24, !dbg !6
  %155 = and i64 %151, 6, !dbg !6
  %156 = or disjoint i64 %154, %155, !dbg !6
  %157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %152, !dbg !6
  %.idx575 = shl nuw nsw i64 %156, 8, !dbg !6
  %158 = getelementptr inbounds nuw i8, ptr addrspace(3) %157, i64 %.idx575, !dbg !6
  %159 = load half, ptr addrspace(3) %158, align 2, !dbg !6
  %160 = lshr i64 %148, 4, !dbg !6
  %161 = shl nuw nsw i64 %148, 1, !dbg !6
  %162 = and i64 %161, 24, !dbg !6
  %163 = and i64 %151, 6, !dbg !6
  %164 = or disjoint i64 %163, 1, !dbg !6
  %165 = or disjoint i64 %162, %164, !dbg !6
  %166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %160, !dbg !6
  %.idx576 = shl nuw nsw i64 %165, 8, !dbg !6
  %167 = getelementptr inbounds nuw i8, ptr addrspace(3) %166, i64 %.idx576, !dbg !6
  %168 = load half, ptr addrspace(3) %167, align 2, !dbg !6
  %169 = insertelement <2 x half> poison, half %159, i64 0, !dbg !6
  %170 = insertelement <2 x half> %169, half %168, i64 1, !dbg !6
  %171 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %151, !dbg !6
  store <2 x half> %170, ptr addrspace(3) %171, align 2, !dbg !6
  %172 = add nuw nsw i64 %148, 256, !dbg !6
  br label %147, !dbg !6

173:                                              ; preds = %147
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %174 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %146, !dbg !19
  %175 = getelementptr [2 x i8], ptr addrspace(1) %174, i64 %116, !dbg !19
  %address.thread1006 = call i32 asm sideeffect "", "=v,0"(i32 %18) #8, !dbg !19, !range !31, !frisk.register_lifetimes !1
  %address.local1007 = zext nneg i32 %address.thread1006 to i64, !dbg !12
  %address.local1008 = shl nuw nsw i64 %address.local1007, 1, !dbg !12
  %address.local1009 = and i64 %address.local1008, 384, !dbg !12
  %address.local1010 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1009, !dbg !19
  %address.local1011 = ptrtoint ptr addrspace(3) %address.local1010 to i64, !dbg !19
  %address.local1012 = trunc i64 %address.local1011 to i32, !dbg !19
  %176 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1012), !dbg !19
  %177 = zext i32 %176 to i64, !dbg !19
  %178 = inttoptr i64 %177 to ptr addrspace(3), !dbg !19
  %address.local1013 = and i32 %address.thread1006, 120, !dbg !19
  %address.local1014 = shl nuw nsw i32 %address.thread1006, 1, !dbg !19
  %address.local1015 = and i32 %address.local1014, 262, !dbg !19
  %address.local1016 = or disjoint i32 %address.local1013, %address.local1015, !dbg !19
  %address.local1017 = shl nuw nsw i32 %address.thread1006, 5, !dbg !19
  %address.local1018 = and i32 %address.local1017, 128, !dbg !19
  %address.local1019 = or disjoint i32 %address.local1016, %address.local1018, !dbg !19
  %address.local1020 = zext nneg i32 %address.local1019 to i64, !dbg !19
  %address.local1021 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %address.local1020, !dbg !19
  %dma.source.bits1598 = ptrtoint ptr addrspace(1) %address.local1021 to i64, !dbg !19
  %dma.source.opaque1599 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1598) #9, !dbg !19
  %dma.source.ptr1600 = inttoptr i64 %dma.source.opaque1599 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1600, ptr addrspace(3) %178, i32 4, i32 0, i32 0), !dbg !19
  %address.local1022 = shl nuw nsw i64 %address.local1007, 1, !dbg !12
  %address.local1023 = and i64 %address.local1022, 384, !dbg !12
  %address.local1024 = or disjoint i64 %address.local1023, 512, !dbg !12
  %address.local1025 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1024, !dbg !19
  %address.local1026 = ptrtoint ptr addrspace(3) %address.local1025 to i64, !dbg !19
  %address.local1027 = trunc i64 %address.local1026 to i32, !dbg !19
  %179 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1027), !dbg !19
  %180 = zext i32 %179 to i64, !dbg !19
  %181 = inttoptr i64 %180 to ptr addrspace(3), !dbg !19
  %address.local1028 = and i32 %address.thread1006, 120, !dbg !19
  %address.local1029 = shl nuw nsw i32 %address.thread1006, 1, !dbg !19
  %address.local1030 = and i32 %address.local1029, 262, !dbg !19
  %address.local1031 = or disjoint i32 %address.local1028, %address.local1030, !dbg !19
  %address.local1032 = shl nuw nsw i32 %address.thread1006, 5, !dbg !19
  %address.local1033 = and i32 %address.local1032, 128, !dbg !19
  %address.local1034 = or disjoint i32 %address.local1031, %address.local1033, !dbg !19
  %address.local1035 = zext nneg i32 %address.local1034 to i64, !dbg !19
  %address.local1036 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %address.local1035, !dbg !19
  %address.local1037 = getelementptr i8, ptr addrspace(1) %address.local1036, i64 1024, !dbg !19
  %dma.source.bits1601 = ptrtoint ptr addrspace(1) %address.local1037 to i64, !dbg !19
  %dma.source.opaque1602 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1601) #9, !dbg !19
  %dma.source.ptr1603 = inttoptr i64 %dma.source.opaque1602 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1603, ptr addrspace(3) %181, i32 4, i32 0, i32 0), !dbg !19
  %address.local1038 = shl nuw nsw i64 %address.local1007, 1, !dbg !12
  %address.local1039 = and i64 %address.local1038, 384, !dbg !12
  %address.local1040 = or disjoint i64 %address.local1039, 1024, !dbg !12
  %address.local1041 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1040, !dbg !19
  %address.local1042 = ptrtoint ptr addrspace(3) %address.local1041 to i64, !dbg !19
  %address.local1043 = trunc i64 %address.local1042 to i32, !dbg !19
  %182 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1043), !dbg !19
  %183 = zext i32 %182 to i64, !dbg !19
  %184 = inttoptr i64 %183 to ptr addrspace(3), !dbg !19
  %address.local1044 = and i32 %address.thread1006, 120, !dbg !19
  %address.local1045 = shl nuw nsw i32 %address.thread1006, 1, !dbg !19
  %address.local1046 = and i32 %address.local1045, 262, !dbg !19
  %address.local1047 = or disjoint i32 %address.local1044, %address.local1046, !dbg !19
  %address.local1048 = shl nuw nsw i32 %address.thread1006, 5, !dbg !19
  %address.local1049 = and i32 %address.local1048, 128, !dbg !19
  %address.local1050 = or disjoint i32 %address.local1047, %address.local1049, !dbg !19
  %address.local1051 = zext nneg i32 %address.local1050 to i64, !dbg !19
  %address.local1052 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %address.local1051, !dbg !19
  %address.local1053 = getelementptr i8, ptr addrspace(1) %address.local1052, i64 2048, !dbg !19
  %dma.source.bits1604 = ptrtoint ptr addrspace(1) %address.local1053 to i64, !dbg !19
  %dma.source.opaque1605 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1604) #9, !dbg !19
  %dma.source.ptr1606 = inttoptr i64 %dma.source.opaque1605 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1606, ptr addrspace(3) %184, i32 4, i32 0, i32 0), !dbg !19
  %address.local1054 = shl nuw nsw i64 %address.local1007, 1, !dbg !12
  %address.local1055 = and i64 %address.local1054, 384, !dbg !12
  %address.local1056 = or disjoint i64 %address.local1055, 1536, !dbg !12
  %address.local1057 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1056, !dbg !19
  %address.local1058 = ptrtoint ptr addrspace(3) %address.local1057 to i64, !dbg !19
  %address.local1059 = trunc i64 %address.local1058 to i32, !dbg !19
  %185 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1059), !dbg !19
  %186 = zext i32 %185 to i64, !dbg !19
  %187 = inttoptr i64 %186 to ptr addrspace(3), !dbg !19
  %address.local1060 = and i32 %address.thread1006, 120, !dbg !19
  %address.local1061 = shl nuw nsw i32 %address.thread1006, 1, !dbg !19
  %address.local1062 = and i32 %address.local1061, 262, !dbg !19
  %address.local1063 = or disjoint i32 %address.local1060, %address.local1062, !dbg !19
  %address.local1064 = shl nuw nsw i32 %address.thread1006, 5, !dbg !19
  %address.local1065 = and i32 %address.local1064, 128, !dbg !19
  %address.local1066 = or disjoint i32 %address.local1063, %address.local1065, !dbg !19
  %address.local1067 = zext nneg i32 %address.local1066 to i64, !dbg !19
  %address.local1068 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %address.local1067, !dbg !19
  %address.local1069 = getelementptr i8, ptr addrspace(1) %address.local1068, i64 3072, !dbg !19
  %dma.source.bits1607 = ptrtoint ptr addrspace(1) %address.local1069 to i64, !dbg !19
  %dma.source.opaque1608 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1607) #9, !dbg !19
  %dma.source.ptr1609 = inttoptr i64 %dma.source.opaque1608 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1609, ptr addrspace(3) %187, i32 4, i32 0, i32 0), !dbg !19
  %address.local1070 = shl nuw nsw i64 %address.local1007, 1, !dbg !12
  %address.local1071 = and i64 %address.local1070, 384, !dbg !12
  %address.local1072 = or disjoint i64 %address.local1071, 2048, !dbg !12
  %address.local1073 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1072, !dbg !19
  %address.local1074 = ptrtoint ptr addrspace(3) %address.local1073 to i64, !dbg !19
  %address.local1075 = trunc i64 %address.local1074 to i32, !dbg !19
  %188 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1075), !dbg !19
  %189 = zext i32 %188 to i64, !dbg !19
  %190 = inttoptr i64 %189 to ptr addrspace(3), !dbg !19
  %address.local1076 = and i32 %address.thread1006, 120, !dbg !19
  %address.local1077 = shl nuw nsw i32 %address.thread1006, 1, !dbg !19
  %address.local1078 = and i32 %address.local1077, 262, !dbg !19
  %address.local1079 = or disjoint i32 %address.local1076, %address.local1078, !dbg !19
  %address.local1080 = shl nuw nsw i32 %address.thread1006, 5, !dbg !19
  %address.local1081 = and i32 %address.local1080, 128, !dbg !19
  %address.local1082 = or disjoint i32 %address.local1079, %address.local1081, !dbg !19
  %address.local1083 = zext nneg i32 %address.local1082 to i64, !dbg !19
  %address.local1084 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %address.local1083, !dbg !19
  %address.local1085 = getelementptr i8, ptr addrspace(1) %address.local1084, i64 4096, !dbg !19
  %dma.source.bits1610 = ptrtoint ptr addrspace(1) %address.local1085 to i64, !dbg !19
  %dma.source.opaque1611 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1610) #9, !dbg !19
  %dma.source.ptr1612 = inttoptr i64 %dma.source.opaque1611 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1612, ptr addrspace(3) %190, i32 4, i32 0, i32 0), !dbg !19
  %address.local1086 = shl nuw nsw i64 %address.local1007, 1, !dbg !12
  %address.local1087 = and i64 %address.local1086, 384, !dbg !12
  %address.local1088 = or disjoint i64 %address.local1087, 2560, !dbg !12
  %address.local1089 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1088, !dbg !19
  %address.local1090 = ptrtoint ptr addrspace(3) %address.local1089 to i64, !dbg !19
  %address.local1091 = trunc i64 %address.local1090 to i32, !dbg !19
  %191 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1091), !dbg !19
  %192 = zext i32 %191 to i64, !dbg !19
  %193 = inttoptr i64 %192 to ptr addrspace(3), !dbg !19
  %address.local1092 = and i32 %address.thread1006, 120, !dbg !19
  %address.local1093 = shl nuw nsw i32 %address.thread1006, 1, !dbg !19
  %address.local1094 = and i32 %address.local1093, 262, !dbg !19
  %address.local1095 = or disjoint i32 %address.local1092, %address.local1094, !dbg !19
  %address.local1096 = shl nuw nsw i32 %address.thread1006, 5, !dbg !19
  %address.local1097 = and i32 %address.local1096, 128, !dbg !19
  %address.local1098 = or disjoint i32 %address.local1095, %address.local1097, !dbg !19
  %address.local1099 = zext nneg i32 %address.local1098 to i64, !dbg !19
  %address.local1100 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %address.local1099, !dbg !19
  %address.local1101 = getelementptr i8, ptr addrspace(1) %address.local1100, i64 5120, !dbg !19
  %dma.source.bits1613 = ptrtoint ptr addrspace(1) %address.local1101 to i64, !dbg !19
  %dma.source.opaque1614 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1613) #9, !dbg !19
  %dma.source.ptr1615 = inttoptr i64 %dma.source.opaque1614 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1615, ptr addrspace(3) %193, i32 4, i32 0, i32 0), !dbg !19
  %address.local1102 = shl nuw nsw i64 %address.local1007, 1, !dbg !12
  %address.local1103 = and i64 %address.local1102, 384, !dbg !12
  %address.local1104 = or disjoint i64 %address.local1103, 3072, !dbg !12
  %address.local1105 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1104, !dbg !19
  %address.local1106 = ptrtoint ptr addrspace(3) %address.local1105 to i64, !dbg !19
  %address.local1107 = trunc i64 %address.local1106 to i32, !dbg !19
  %194 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1107), !dbg !19
  %195 = zext i32 %194 to i64, !dbg !19
  %196 = inttoptr i64 %195 to ptr addrspace(3), !dbg !19
  %address.local1108 = and i32 %address.thread1006, 120, !dbg !19
  %address.local1109 = shl nuw nsw i32 %address.thread1006, 1, !dbg !19
  %address.local1110 = and i32 %address.local1109, 262, !dbg !19
  %address.local1111 = or disjoint i32 %address.local1108, %address.local1110, !dbg !19
  %address.local1112 = shl nuw nsw i32 %address.thread1006, 5, !dbg !19
  %address.local1113 = and i32 %address.local1112, 128, !dbg !19
  %address.local1114 = or disjoint i32 %address.local1111, %address.local1113, !dbg !19
  %address.local1115 = zext nneg i32 %address.local1114 to i64, !dbg !19
  %address.local1116 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %address.local1115, !dbg !19
  %address.local1117 = getelementptr i8, ptr addrspace(1) %address.local1116, i64 6144, !dbg !19
  %dma.source.bits1616 = ptrtoint ptr addrspace(1) %address.local1117 to i64, !dbg !19
  %dma.source.opaque1617 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1616) #9, !dbg !19
  %dma.source.ptr1618 = inttoptr i64 %dma.source.opaque1617 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1618, ptr addrspace(3) %196, i32 4, i32 0, i32 0), !dbg !19
  %address.local1118 = shl nuw nsw i64 %address.local1007, 1, !dbg !12
  %address.local1119 = and i64 %address.local1118, 384, !dbg !12
  %address.local1120 = or disjoint i64 %address.local1119, 3584, !dbg !12
  %address.local1121 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1120, !dbg !19
  %address.local1122 = ptrtoint ptr addrspace(3) %address.local1121 to i64, !dbg !19
  %address.local1123 = trunc i64 %address.local1122 to i32, !dbg !19
  %197 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1123), !dbg !19
  %198 = zext i32 %197 to i64, !dbg !19
  %199 = inttoptr i64 %198 to ptr addrspace(3), !dbg !19
  %address.local1124 = and i32 %address.thread1006, 120, !dbg !19
  %address.local1125 = shl nuw nsw i32 %address.thread1006, 1, !dbg !19
  %address.local1126 = and i32 %address.local1125, 262, !dbg !19
  %address.local1127 = or disjoint i32 %address.local1124, %address.local1126, !dbg !19
  %address.local1128 = shl nuw nsw i32 %address.thread1006, 5, !dbg !19
  %address.local1129 = and i32 %address.local1128, 128, !dbg !19
  %address.local1130 = or disjoint i32 %address.local1127, %address.local1129, !dbg !19
  %address.local1131 = zext nneg i32 %address.local1130 to i64, !dbg !19
  %address.local1132 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %address.local1131, !dbg !19
  %address.local1133 = getelementptr i8, ptr addrspace(1) %address.local1132, i64 7168, !dbg !19
  %dma.source.bits1619 = ptrtoint ptr addrspace(1) %address.local1133 to i64, !dbg !19
  %dma.source.opaque1620 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1619) #9, !dbg !19
  %dma.source.ptr1621 = inttoptr i64 %dma.source.opaque1620 to ptr addrspace(1), !dbg !19
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1621, ptr addrspace(3) %199, i32 4, i32 0, i32 0), !dbg !19
  call void @llvm.amdgcn.asyncmark(), !dbg !19
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.local1134 = shl nuw nsw i32 %address.thread1006, 3, !dbg !8
  %address.local1135 = and i32 %address.local1134, 504, !dbg !8
  %address.local1136 = zext nneg i32 %address.local1135 to i64, !dbg !8
  %address.local1137 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1136, !dbg !8
  %200 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1137, i32 0) #0, !dbg !8
  %201 = bitcast <4 x i32> %200 to <8 x half>, !dbg !8
  %address.local1138 = shl nuw nsw i32 %address.thread1006, 3, !dbg !8
  %address.local1139 = and i32 %address.local1138, 504, !dbg !8
  %address.local1140 = zext nneg i32 %address.local1139 to i64, !dbg !8
  %address.local1141 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1140, !dbg !8
  %address.local1142 = getelementptr i8, ptr addrspace(3) %address.local1141, i64 1024, !dbg !8
  %202 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1142, i32 0) #0, !dbg !8
  %203 = bitcast <4 x i32> %202 to <8 x half>, !dbg !8
  %address.local1143 = shl nuw nsw i32 %address.thread1006, 3, !dbg !8
  %address.local1144 = and i32 %address.local1143, 504, !dbg !8
  %address.local1145 = zext nneg i32 %address.local1144 to i64, !dbg !8
  %address.local1146 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1145, !dbg !8
  %address.local1147 = getelementptr i8, ptr addrspace(3) %address.local1146, i64 2048, !dbg !8
  %204 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1147, i32 0) #0, !dbg !8
  %205 = bitcast <4 x i32> %204 to <8 x half>, !dbg !8
  %address.local1148 = shl nuw nsw i32 %address.thread1006, 3, !dbg !8
  %address.local1149 = or i32 %address.local1148, 1536, !dbg !8
  %address.local1150 = zext nneg i32 %address.local1149 to i64, !dbg !8
  %address.local1151 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1150, !dbg !8
  %206 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1151, i32 0) #0, !dbg !8
  %207 = bitcast <4 x i32> %206 to <8 x half>, !dbg !8
  %address.local1152 = shl nuw nsw i32 %address.thread1006, 3, !dbg !8
  %address.local1153 = and i32 %address.local1152, 504, !dbg !8
  %address.local1154 = zext nneg i32 %address.local1153 to i64, !dbg !8
  %address.local1155 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1154, !dbg !8
  %address.local1156 = getelementptr i8, ptr addrspace(3) %address.local1155, i64 4096, !dbg !8
  %208 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1156, i32 0) #0, !dbg !8
  %209 = bitcast <4 x i32> %208 to <8 x half>, !dbg !8
  %address.local1157 = shl nuw nsw i32 %address.thread1006, 3, !dbg !8
  %address.local1158 = and i32 %address.local1157, 504, !dbg !8
  %address.local1159 = zext nneg i32 %address.local1158 to i64, !dbg !8
  %address.local1160 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1159, !dbg !8
  %address.local1161 = getelementptr i8, ptr addrspace(3) %address.local1160, i64 5120, !dbg !8
  %210 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1161, i32 0) #0, !dbg !8
  %211 = bitcast <4 x i32> %210 to <8 x half>, !dbg !8
  %address.local1162 = shl nuw nsw i32 %address.thread1006, 3, !dbg !8
  %address.local1163 = and i32 %address.local1162, 504, !dbg !8
  %address.local1164 = zext nneg i32 %address.local1163 to i64, !dbg !8
  %address.local1165 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1164, !dbg !8
  %address.local1166 = getelementptr i8, ptr addrspace(3) %address.local1165, i64 6144, !dbg !8
  %212 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1166, i32 0) #0, !dbg !8
  %213 = bitcast <4 x i32> %212 to <8 x half>, !dbg !8
  %address.local1167 = shl nuw nsw i32 %address.thread1006, 3, !dbg !8
  %address.local1168 = or i32 %address.local1167, 3584, !dbg !8
  %address.local1169 = zext nneg i32 %address.local1168 to i64, !dbg !8
  %address.local1170 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1169, !dbg !8
  %214 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1170, i32 0) #0, !dbg !8
  %215 = bitcast <4 x i32> %214 to <8 x half>, !dbg !8
  %216 = shufflevector <8 x half> %201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %216, <4 x float> zeroinitializer) #8, !dbg !8
  %218 = shufflevector <8 x half> %203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %218, <4 x float> %217) #8, !dbg !8
  %220 = shufflevector <8 x half> %205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %220, <4 x float> %219) #8, !dbg !8
  %222 = shufflevector <8 x half> %207, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %222, <4 x float> %221) #8, !dbg !8
  %224 = shufflevector <8 x half> %209, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %224, <4 x float> %223) #8, !dbg !8
  %226 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %226, <4 x float> %225) #8, !dbg !8
  %228 = shufflevector <8 x half> %213, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %228, <4 x float> %227) #8, !dbg !8
  %230 = shufflevector <8 x half> %215, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %230, <4 x float> %229) #8, !dbg !8
  %232 = extractelement <4 x float> %231, i64 0, !dbg !8
  %233 = extractelement <4 x float> %231, i64 1, !dbg !8
  %234 = extractelement <4 x float> %231, i64 2, !dbg !8
  %235 = extractelement <4 x float> %231, i64 3, !dbg !8
  %236 = select i1 %.not, float %233, float %232, !dbg !8
  %237 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %238 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %237), !dbg !8
  %239 = add nuw nsw i32 %238, 64, !dbg !8
  %240 = xor i32 %238, 16, !dbg !8
  %241 = and i32 %239, 192, !dbg !8
  %242 = icmp samesign ult i32 %240, %241, !dbg !8
  %243 = select i1 %242, i32 %240, i32 %238, !dbg !8
  %244 = shl nuw nsw i32 %243, 2, !dbg !8
  %245 = bitcast float %236 to i32, !dbg !8
  %246 = call i32 @llvm.amdgcn.ds.bpermute(i32 %244, i32 %245), !dbg !8
  %247 = bitcast i32 %246 to float, !dbg !8
  %248 = select i1 %.not, float %232, float %247, !dbg !8
  %249 = select i1 %.not, float %247, float %233, !dbg !8
  %250 = select i1 %.not, float %235, float %234, !dbg !8
  %251 = bitcast float %250 to i32, !dbg !8
  %252 = call i32 @llvm.amdgcn.ds.bpermute(i32 %244, i32 %251), !dbg !8
  %253 = bitcast i32 %252 to float, !dbg !8
  %254 = select i1 %.not, float %234, float %253, !dbg !8
  %255 = select i1 %.not, float %253, float %235, !dbg !8
  %256 = select i1 %.not566, float %254, float %248, !dbg !8
  %257 = xor i32 %238, 32, !dbg !8
  %258 = icmp samesign ult i32 %257, %241, !dbg !8
  %259 = select i1 %258, i32 %257, i32 %238, !dbg !8
  %260 = shl nuw nsw i32 %259, 2, !dbg !8
  %261 = bitcast float %256 to i32, !dbg !8
  %262 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %261), !dbg !8
  %263 = bitcast i32 %262 to float, !dbg !8
  %264 = select i1 %.not566, float %248, float %263, !dbg !8
  %265 = select i1 %.not566, float %263, float %254, !dbg !8
  %266 = select i1 %.not566, float %255, float %249, !dbg !8
  %267 = bitcast float %266 to i32, !dbg !8
  %268 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %267), !dbg !8
  %269 = bitcast i32 %268 to float, !dbg !8
  %270 = select i1 %.not566, float %249, float %269, !dbg !8
  %271 = select i1 %.not566, float %269, float %255, !dbg !8
  %272 = or disjoint i64 %111, %index.compact55, !dbg !16
  %273 = lshr i64 %96, 32, !dbg !24
  %compare.high = trunc i64 %273 to i32, !dbg !24
  %274 = lshr i64 %272, 32, !dbg !24
  %compare.high1501 = trunc i64 %274 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1501, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1501, !dbg !24
  %compare.low.base = trunc i64 %96 to i32, !dbg !24
  %compare.low.base1502 = trunc i64 %272 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1502, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %275 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %276 = fadd float %264, %275, !dbg !16
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 1) #8, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1503 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result1504 = select i1 %compare.high.equal, i1 %compare.low1503, i1 %compare.high.order, !dbg !24
  %277 = select i1 %compare.result1504, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %278 = fadd float %270, %277, !dbg !16
  %compare.local.base1505 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 2) #8, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1506 = or i32 %compare.local.base1505, 2, !dbg !24
  %compare.low1507 = icmp ugt i32 %compare.low.base, %compare.local1506, !dbg !24
  %compare.result1508 = select i1 %compare.high.equal, i1 %compare.low1507, i1 %compare.high.order, !dbg !24
  %279 = select i1 %compare.result1508, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %280 = fadd float %265, %279, !dbg !16
  %compare.local.base1509 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 3) #8, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1510 = or i32 %compare.local.base1509, 3, !dbg !24
  %compare.low1511 = icmp ugt i32 %compare.low.base, %compare.local1510, !dbg !24
  %compare.result1512 = select i1 %compare.high.equal, i1 %compare.low1511, i1 %compare.high.order, !dbg !24
  %281 = select i1 %compare.result1512, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %282 = fadd float %271, %281, !dbg !16
  %283 = call float @__ocml_exp2_f32(float %276), !dbg !26
  %284 = fadd float %283, 0x0000000000000000, !dbg !27
  %285 = call float @__ocml_exp2_f32(float %278), !dbg !26
  %286 = fadd float %284, %285, !dbg !27
  %287 = call float @__ocml_exp2_f32(float %280), !dbg !26
  %288 = fadd float %286, %287, !dbg !27
  %289 = call float @__ocml_exp2_f32(float %282), !dbg !26
  %290 = fadd float %288, %289, !dbg !27
  %291 = shufflevector <8 x half> %201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %291, <4 x float> zeroinitializer) #8, !dbg !8
  %293 = shufflevector <8 x half> %203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %293, <4 x float> %292) #8, !dbg !8
  %295 = shufflevector <8 x half> %205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %295, <4 x float> %294) #8, !dbg !8
  %297 = shufflevector <8 x half> %207, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %297, <4 x float> %296) #8, !dbg !8
  %299 = shufflevector <8 x half> %209, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %299, <4 x float> %298) #8, !dbg !8
  %301 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %301, <4 x float> %300) #8, !dbg !8
  %303 = shufflevector <8 x half> %213, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %303, <4 x float> %302) #8, !dbg !8
  %305 = shufflevector <8 x half> %215, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %305, <4 x float> %304) #8, !dbg !8
  %307 = extractelement <4 x float> %306, i64 0, !dbg !8
  %308 = extractelement <4 x float> %306, i64 1, !dbg !8
  %309 = extractelement <4 x float> %306, i64 2, !dbg !8
  %310 = extractelement <4 x float> %306, i64 3, !dbg !8
  %311 = select i1 %.not, float %308, float %307, !dbg !8
  %312 = bitcast float %311 to i32, !dbg !8
  %313 = call i32 @llvm.amdgcn.ds.bpermute(i32 %244, i32 %312), !dbg !8
  %314 = bitcast i32 %313 to float, !dbg !8
  %315 = select i1 %.not, float %307, float %314, !dbg !8
  %316 = select i1 %.not, float %314, float %308, !dbg !8
  %317 = select i1 %.not, float %310, float %309, !dbg !8
  %318 = bitcast float %317 to i32, !dbg !8
  %319 = call i32 @llvm.amdgcn.ds.bpermute(i32 %244, i32 %318), !dbg !8
  %320 = bitcast i32 %319 to float, !dbg !8
  %321 = select i1 %.not, float %309, float %320, !dbg !8
  %322 = select i1 %.not, float %320, float %310, !dbg !8
  %323 = select i1 %.not566, float %321, float %315, !dbg !8
  %324 = bitcast float %323 to i32, !dbg !8
  %325 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %324), !dbg !8
  %326 = bitcast i32 %325 to float, !dbg !8
  %327 = select i1 %.not566, float %315, float %326, !dbg !8
  %328 = select i1 %.not566, float %326, float %321, !dbg !8
  %329 = select i1 %.not566, float %322, float %316, !dbg !8
  %330 = bitcast float %329 to i32, !dbg !8
  %331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %330), !dbg !8
  %332 = bitcast i32 %331 to float, !dbg !8
  %333 = select i1 %.not566, float %316, float %332, !dbg !8
  %334 = select i1 %.not566, float %332, float %322, !dbg !8
  %compare.local.base1513 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 4) #8, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1514 = or i32 %compare.local.base1513, 16, !dbg !24
  %compare.low1515 = icmp ugt i32 %compare.low.base, %compare.local1514, !dbg !24
  %compare.result1516 = select i1 %compare.high.equal, i1 %compare.low1515, i1 %compare.high.order, !dbg !24
  %335 = select i1 %compare.result1516, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %336 = fadd float %327, %335, !dbg !16
  %compare.local.base1517 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 5) #8, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1518 = or i32 %compare.local.base1517, 17, !dbg !24
  %compare.low1519 = icmp ugt i32 %compare.low.base, %compare.local1518, !dbg !24
  %compare.result1520 = select i1 %compare.high.equal, i1 %compare.low1519, i1 %compare.high.order, !dbg !24
  %337 = select i1 %compare.result1520, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %338 = fadd float %333, %337, !dbg !16
  %compare.local.base1521 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 6) #8, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1522 = or i32 %compare.local.base1521, 18, !dbg !24
  %compare.low1523 = icmp ugt i32 %compare.low.base, %compare.local1522, !dbg !24
  %compare.result1524 = select i1 %compare.high.equal, i1 %compare.low1523, i1 %compare.high.order, !dbg !24
  %339 = select i1 %compare.result1524, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %340 = fadd float %328, %339, !dbg !16
  %compare.local.base1525 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1502, i32 7) #8, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1526 = or i32 %compare.local.base1525, 19, !dbg !24
  %compare.low1527 = icmp ugt i32 %compare.low.base, %compare.local1526, !dbg !24
  %compare.result1528 = select i1 %compare.high.equal, i1 %compare.low1527, i1 %compare.high.order, !dbg !24
  %341 = select i1 %compare.result1528, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %342 = fadd float %334, %341, !dbg !16
  %343 = call float @__ocml_exp2_f32(float %336), !dbg !26
  %344 = fadd float %290, %343, !dbg !27
  %345 = call float @__ocml_exp2_f32(float %338), !dbg !26
  %346 = fadd float %344, %345, !dbg !27
  %347 = call float @__ocml_exp2_f32(float %340), !dbg !26
  %348 = fadd float %346, %347, !dbg !27
  %349 = call float @__ocml_exp2_f32(float %342), !dbg !26
  %350 = fadd float %348, %349, !dbg !27
  %351 = bitcast float %350 to i32, !dbg !27
  %352 = call i32 @llvm.amdgcn.ds.bpermute(i32 %244, i32 %351), !dbg !27
  %353 = bitcast i32 %352 to float, !dbg !27
  %354 = fadd float %350, %353, !dbg !27
  %355 = bitcast float %354 to i32, !dbg !27
  %356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %260, i32 %355), !dbg !27
  %357 = bitcast i32 %356 to float, !dbg !27
  %358 = fadd float %354, %357, !dbg !27
  %359 = extractelement <1 x float> %104, i64 0, !dbg !27
  %360 = fadd float %358, %359, !dbg !27
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  br i1 true, label %361, label %372, !dbg !18

361:                                              ; preds = %173
  %362 = insertelement <4 x float> poison, float %343, i64 0, !dbg !18
  %363 = insertelement <4 x float> %362, float %345, i64 1, !dbg !18
  %364 = insertelement <4 x float> %363, float %347, i64 2, !dbg !18
  %365 = insertelement <4 x float> %364, float %349, i64 3, !dbg !18
  %366 = fptrunc <4 x float> %365 to <4 x half>, !dbg !18
  %367 = insertelement <4 x float> poison, float %283, i64 0, !dbg !18
  %368 = insertelement <4 x float> %367, float %285, i64 1, !dbg !18
  %369 = insertelement <4 x float> %368, float %287, i64 2, !dbg !18
  %370 = insertelement <4 x float> %369, float %289, i64 3, !dbg !18
  %371 = fptrunc <4 x float> %370 to <4 x half>, !dbg !18
  %address.thread1171 = call i32 asm sideeffect "", "=v,0"(i32 %18) #8, !dbg !18, !range !31, !frisk.register_lifetimes !1
  %address.local1172 = and i32 %address.thread1171, 192, !dbg !7
  %address.local1173 = shl nuw nsw i32 %address.thread1171, 2, !dbg !7
  %address.local1174 = and i32 %address.local1173, 60, !dbg !7
  %address.local1175 = or disjoint i32 %address.local1172, %address.local1174, !dbg !7
  %address.local1176 = shl nuw nsw i32 %address.thread1171, 4, !dbg !7
  %address.local1177 = and i32 %address.local1176, 768, !dbg !7
  %address.local1178 = or disjoint i32 %address.local1175, %address.local1177, !dbg !7
  %address.local1179 = zext nneg i32 %address.local1178 to i64, !dbg !7
  %address.local1180 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1179, !dbg !7
  store <4 x half> %371, ptr addrspace(3) %address.local1180, align 2, !dbg !18
  %address.local1181 = and i32 %address.thread1171, 192, !dbg !7
  %address.local1182 = shl nuw nsw i32 %address.thread1171, 2, !dbg !7
  %address.local1183 = and i32 %address.local1182, 60, !dbg !7
  %address.local1184 = or disjoint i32 %address.local1181, %address.local1183, !dbg !7
  %address.local1185 = shl nuw nsw i32 %address.thread1171, 4, !dbg !7
  %address.local1186 = and i32 %address.local1185, 768, !dbg !7
  %address.local1187 = or disjoint i32 %address.local1184, %address.local1186, !dbg !7
  %address.local1188 = zext nneg i32 %address.local1187 to i64, !dbg !7
  %address.local1189 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1188, !dbg !7
  %address.local1190 = getelementptr i8, ptr addrspace(3) %address.local1189, i64 2048, !dbg !7
  store <4 x half> %366, ptr addrspace(3) %address.local1190, align 2, !dbg !18
  br label %372, !dbg !18

372:                                              ; preds = %361, %173
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread1191 = call i32 asm sideeffect "", "=v,0"(i32 %18) #8, !dbg !20, !range !31, !frisk.register_lifetimes !1
  %address.local1192 = shl nuw nsw i32 %address.thread1191, 1, !dbg !20
  %address.local1193 = and i32 %address.local1192, 8, !dbg !20
  %address.local1194 = shl nuw nsw i32 %address.thread1191, 4, !dbg !20
  %address.local1195 = and i32 %address.local1194, 48, !dbg !20
  %address.local1196 = or disjoint i32 %address.local1193, %address.local1195, !dbg !20
  %address.local1197 = shl nuw nsw i32 %address.thread1191, 5, !dbg !20
  %address.local1198 = and i32 %address.local1197, 1792, !dbg !20
  %address.local1199 = or disjoint i32 %address.local1196, %address.local1198, !dbg !20
  %address.local1200 = zext nneg i32 %address.local1199 to i64, !dbg !20
  %address.local1201 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1200, !dbg !20
  %373 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1201, i32 0) #0, !dbg !20
  %374 = bitcast <4 x i32> %373 to <8 x half>, !dbg !20
  %address.local1202 = shl nuw nsw i32 %address.thread1191, 1, !dbg !20
  %address.local1203 = and i32 %address.local1202, 8, !dbg !20
  %address.local1204 = shl nuw nsw i32 %address.thread1191, 4, !dbg !20
  %address.local1205 = and i32 %address.local1204, 48, !dbg !20
  %address.local1206 = or disjoint i32 %address.local1203, %address.local1205, !dbg !20
  %address.local1207 = shl nuw nsw i32 %address.thread1191, 5, !dbg !20
  %address.local1208 = and i32 %address.local1207, 1792, !dbg !20
  %address.local1209 = or disjoint i32 %address.local1206, %address.local1208, !dbg !20
  %address.local1210 = zext nneg i32 %address.local1209 to i64, !dbg !20
  %address.local1211 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1210, !dbg !20
  %address.local1212 = getelementptr i8, ptr addrspace(3) %address.local1211, i64 128, !dbg !20
  %375 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1212, i32 0) #0, !dbg !20
  %376 = bitcast <4 x i32> %375 to <8 x half>, !dbg !20
  %address.local1213 = shl nuw nsw i32 %address.thread1191, 1, !dbg !20
  %address.local1214 = and i32 %address.local1213, 8, !dbg !20
  %address.local1215 = shl nuw nsw i32 %address.thread1191, 4, !dbg !20
  %address.local1216 = and i32 %address.local1215, 48, !dbg !20
  %address.local1217 = or disjoint i32 %address.local1214, %address.local1216, !dbg !20
  %address.local1218 = shl nuw nsw i32 %address.thread1191, 5, !dbg !20
  %address.local1219 = and i32 %address.local1218, 1792, !dbg !20
  %address.local1220 = or disjoint i32 %address.local1217, %address.local1219, !dbg !20
  %address.local1221 = zext nneg i32 %address.local1220 to i64, !dbg !20
  %address.local1222 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1221, !dbg !20
  %address.local1223 = getelementptr i8, ptr addrspace(3) %address.local1222, i64 256, !dbg !20
  %377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1223, i32 0) #0, !dbg !20
  %378 = bitcast <4 x i32> %377 to <8 x half>, !dbg !20
  %address.local1224 = shl nuw nsw i32 %address.thread1191, 1, !dbg !20
  %address.local1225 = and i32 %address.local1224, 8, !dbg !20
  %address.local1226 = shl nuw nsw i32 %address.thread1191, 4, !dbg !20
  %address.local1227 = and i32 %address.local1226, 48, !dbg !20
  %address.local1228 = or disjoint i32 %address.local1225, %address.local1227, !dbg !20
  %address.local1229 = shl nuw nsw i32 %address.thread1191, 5, !dbg !20
  %address.local1230 = and i32 %address.local1229, 1792, !dbg !20
  %address.local1231 = or disjoint i32 %address.local1228, %address.local1230, !dbg !20
  %address.local1232 = zext nneg i32 %address.local1231 to i64, !dbg !20
  %address.local1233 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1232, !dbg !20
  %address.local1234 = getelementptr i8, ptr addrspace(3) %address.local1233, i64 384, !dbg !20
  %379 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1234, i32 0) #0, !dbg !20
  %380 = bitcast <4 x i32> %379 to <8 x half>, !dbg !20
  %address.local1235 = shl nuw nsw i32 %address.thread1191, 1, !dbg !20
  %address.local1236 = and i32 %address.local1235, 8, !dbg !20
  %address.local1237 = shl nuw nsw i32 %address.thread1191, 4, !dbg !20
  %address.local1238 = and i32 %address.local1237, 48, !dbg !20
  %address.local1239 = or disjoint i32 %address.local1236, %address.local1238, !dbg !20
  %address.local1240 = shl nuw nsw i32 %address.thread1191, 5, !dbg !20
  %address.local1241 = and i32 %address.local1240, 1792, !dbg !20
  %address.local1242 = or disjoint i32 %address.local1239, %address.local1241, !dbg !20
  %address.local1243 = zext nneg i32 %address.local1242 to i64, !dbg !20
  %address.local1244 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1243, !dbg !20
  %address.local1245 = getelementptr i8, ptr addrspace(3) %address.local1244, i64 4096, !dbg !20
  %381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1245, i32 0) #0, !dbg !20
  %382 = bitcast <4 x i32> %381 to <8 x half>, !dbg !20
  %address.local1246 = shl nuw nsw i32 %address.thread1191, 1, !dbg !20
  %address.local1247 = and i32 %address.local1246, 8, !dbg !20
  %address.local1248 = shl nuw nsw i32 %address.thread1191, 4, !dbg !20
  %address.local1249 = and i32 %address.local1248, 48, !dbg !20
  %address.local1250 = or disjoint i32 %address.local1247, %address.local1249, !dbg !20
  %address.local1251 = shl nuw nsw i32 %address.thread1191, 5, !dbg !20
  %address.local1252 = and i32 %address.local1251, 1792, !dbg !20
  %address.local1253 = or disjoint i32 %address.local1250, %address.local1252, !dbg !20
  %address.local1254 = zext nneg i32 %address.local1253 to i64, !dbg !20
  %address.local1255 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1254, !dbg !20
  %address.local1256 = getelementptr i8, ptr addrspace(3) %address.local1255, i64 4224, !dbg !20
  %383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1256, i32 0) #0, !dbg !20
  %384 = bitcast <4 x i32> %383 to <8 x half>, !dbg !20
  %address.local1257 = shl nuw nsw i32 %address.thread1191, 1, !dbg !20
  %address.local1258 = and i32 %address.local1257, 8, !dbg !20
  %address.local1259 = shl nuw nsw i32 %address.thread1191, 4, !dbg !20
  %address.local1260 = and i32 %address.local1259, 48, !dbg !20
  %address.local1261 = or disjoint i32 %address.local1258, %address.local1260, !dbg !20
  %address.local1262 = shl nuw nsw i32 %address.thread1191, 5, !dbg !20
  %address.local1263 = and i32 %address.local1262, 1792, !dbg !20
  %address.local1264 = or disjoint i32 %address.local1261, %address.local1263, !dbg !20
  %address.local1265 = zext nneg i32 %address.local1264 to i64, !dbg !20
  %address.local1266 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1265, !dbg !20
  %address.local1267 = getelementptr i8, ptr addrspace(3) %address.local1266, i64 4352, !dbg !20
  %385 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1267, i32 0) #0, !dbg !20
  %386 = bitcast <4 x i32> %385 to <8 x half>, !dbg !20
  %address.local1268 = shl nuw nsw i32 %address.thread1191, 1, !dbg !20
  %address.local1269 = and i32 %address.local1268, 8, !dbg !20
  %address.local1270 = shl nuw nsw i32 %address.thread1191, 4, !dbg !20
  %address.local1271 = and i32 %address.local1270, 48, !dbg !20
  %address.local1272 = or disjoint i32 %address.local1269, %address.local1271, !dbg !20
  %address.local1273 = shl nuw nsw i32 %address.thread1191, 5, !dbg !20
  %address.local1274 = and i32 %address.local1273, 1792, !dbg !20
  %address.local1275 = or disjoint i32 %address.local1272, %address.local1274, !dbg !20
  %address.local1276 = zext nneg i32 %address.local1275 to i64, !dbg !20
  %address.local1277 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1276, !dbg !20
  %address.local1278 = getelementptr i8, ptr addrspace(3) %address.local1277, i64 4480, !dbg !20
  %387 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1278, i32 0) #0, !dbg !20
  %address.local1279 = and i32 %address.thread1191, 192, !dbg !7
  %address.local1280 = shl nuw nsw i32 %address.thread1191, 2, !dbg !7
  %address.local1281 = and i32 %address.local1280, 60, !dbg !7
  %address.local1282 = or disjoint i32 %address.local1279, %address.local1281, !dbg !7
  %address.local1283 = shl nuw nsw i32 %address.thread1191, 4, !dbg !7
  %address.local1284 = and i32 %address.local1283, 768, !dbg !7
  %address.local1285 = or disjoint i32 %address.local1282, %address.local1284, !dbg !7
  %address.local1286 = zext nneg i32 %address.local1285 to i64, !dbg !7
  %address.local1287 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1286, !dbg !7
  %388 = load <4 x half>, ptr addrspace(3) %address.local1287, align 2, !dbg !20
  %389 = bitcast <4 x i32> %387 to <8 x half>, !dbg !20
  %390 = shufflevector <8 x half> %374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %388, <4 x half> %390, <4 x float> zeroinitializer) #8, !dbg !20
  %address.local1288 = and i32 %address.thread1191, 192, !dbg !7
  %address.local1289 = shl nuw nsw i32 %address.thread1191, 2, !dbg !7
  %address.local1290 = and i32 %address.local1289, 60, !dbg !7
  %address.local1291 = or disjoint i32 %address.local1288, %address.local1290, !dbg !7
  %address.local1292 = shl nuw nsw i32 %address.thread1191, 4, !dbg !7
  %address.local1293 = and i32 %address.local1292, 768, !dbg !7
  %address.local1294 = or disjoint i32 %address.local1291, %address.local1293, !dbg !7
  %address.local1295 = zext nneg i32 %address.local1294 to i64, !dbg !7
  %address.local1296 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1295, !dbg !7
  %address.local1297 = getelementptr i8, ptr addrspace(3) %address.local1296, i64 2048, !dbg !7
  %392 = load <4 x half>, ptr addrspace(3) %address.local1297, align 2, !dbg !20
  %393 = shufflevector <8 x half> %382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %394 = shufflevector <32 x float> %103, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %395 = shufflevector <8 x half> %374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %392, <4 x half> %393, <4 x float> %391) #8, !dbg !20
  %397 = load <4 x half>, ptr addrspace(3) %address.local1287, align 2, !dbg !20
  %398 = shufflevector <8 x half> %382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %399 = shufflevector <32 x float> %103, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %400 = fadd <4 x float> %394, %396, !dbg !28
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %397, <4 x half> %395, <4 x float> zeroinitializer) #8, !dbg !20
  %402 = shufflevector <8 x half> %376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %403 = shufflevector <8 x half> %384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %404 = shufflevector <32 x float> %103, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !28
  %405 = shufflevector <8 x half> %376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %406 = shufflevector <8 x half> %384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %407 = shufflevector <32 x float> %103, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !28
  %408 = shufflevector <8 x half> %378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %409 = shufflevector <8 x half> %386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %410 = shufflevector <32 x float> %103, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !28
  %411 = shufflevector <8 x half> %378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %412 = shufflevector <8 x half> %386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %413 = shufflevector <32 x float> %103, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !28
  %414 = shufflevector <8 x half> %380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %415 = shufflevector <8 x half> %389, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %416 = shufflevector <32 x float> %103, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !28
  %417 = shufflevector <8 x half> %380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %418 = shufflevector <8 x half> %389, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %419 = shufflevector <32 x float> %103, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !28
  %420 = load <4 x half>, ptr addrspace(3) %address.local1297, align 2, !dbg !20
  %421 = load <4 x half>, ptr addrspace(3) %address.local1287, align 2, !dbg !20
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %420, <4 x half> %398, <4 x float> %401) #8, !dbg !20
  %423 = fadd <4 x float> %399, %422, !dbg !28
  %424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %421, <4 x half> %402, <4 x float> zeroinitializer) #8, !dbg !20
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %420, <4 x half> %403, <4 x float> %424) #8, !dbg !20
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %421, <4 x half> %405, <4 x float> zeroinitializer) #8, !dbg !20
  %427 = fadd <4 x float> %404, %425, !dbg !28
  %428 = shufflevector <4 x float> %427, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !28
  %429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %420, <4 x half> %406, <4 x float> %426) #8, !dbg !20
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %421, <4 x half> %408, <4 x float> zeroinitializer) #8, !dbg !20
  %431 = fadd <4 x float> %407, %429, !dbg !28
  %432 = shufflevector <4 x float> %431, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !28
  %433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %420, <4 x half> %409, <4 x float> %430) #8, !dbg !20
  %434 = load <4 x half>, ptr addrspace(3) %address.local1297, align 2, !dbg !20
  %435 = load <4 x half>, ptr addrspace(3) %address.local1287, align 2, !dbg !20
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %421, <4 x half> %411, <4 x float> zeroinitializer) #8, !dbg !20
  %437 = fadd <4 x float> %410, %433, !dbg !28
  %438 = shufflevector <4 x float> %437, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !28
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %434, <4 x half> %412, <4 x float> %436) #8, !dbg !20
  %440 = fadd <4 x float> %413, %439, !dbg !28
  %441 = shufflevector <4 x float> %440, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !28
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %435, <4 x half> %414, <4 x float> zeroinitializer) #8, !dbg !20
  %443 = load <4 x half>, ptr addrspace(3) %address.local1297, align 2, !dbg !20
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %434, <4 x half> %415, <4 x float> %442) #8, !dbg !20
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %435, <4 x half> %417, <4 x float> zeroinitializer) #8, !dbg !20
  %446 = fadd <4 x float> %416, %444, !dbg !28
  %447 = shufflevector <4 x float> %446, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !28
  %448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %443, <4 x half> %418, <4 x float> %445) #8, !dbg !20
  %449 = fadd <4 x float> %419, %448, !dbg !28
  %450 = shufflevector <4 x float> %449, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !28
  %451 = shufflevector <4 x float> %400, <4 x float> %423, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !9
  %452 = shufflevector <32 x float> %451, <32 x float> %428, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !9
  %453 = shufflevector <32 x float> %452, <32 x float> %432, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !9
  %454 = shufflevector <32 x float> %453, <32 x float> %438, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !9
  %455 = shufflevector <32 x float> %454, <32 x float> %441, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !9
  %456 = shufflevector <32 x float> %455, <32 x float> %447, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !9
  %457 = shufflevector <32 x float> %456, <32 x float> %450, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !9
  %458 = insertelement <1 x float> poison, float %360, i64 0, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %459 = add nuw nsw i64 %102, 1, !dbg !9
  br label %101, !dbg !9

460:                                              ; preds = %101
  %461 = extractelement <32 x float> %103, i64 0, !dbg !9
  %462 = extractelement <32 x float> %103, i64 1, !dbg !9
  %463 = extractelement <32 x float> %103, i64 2, !dbg !9
  %464 = extractelement <32 x float> %103, i64 3, !dbg !9
  %465 = select i1 %.not, float %462, float %461, !dbg !9
  %466 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !9
  %467 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %466), !dbg !9
  %468 = add nuw nsw i32 %467, 64, !dbg !9
  %469 = xor i32 %467, 16, !dbg !9
  %470 = and i32 %468, 192, !dbg !9
  %471 = icmp samesign ult i32 %469, %470, !dbg !9
  %472 = select i1 %471, i32 %469, i32 %467, !dbg !9
  %473 = shl nuw nsw i32 %472, 2, !dbg !9
  %474 = bitcast float %465 to i32, !dbg !9
  %475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %474), !dbg !9
  %476 = bitcast i32 %475 to float, !dbg !9
  %477 = select i1 %.not, float %461, float %476, !dbg !9
  %478 = select i1 %.not, float %476, float %462, !dbg !9
  %479 = select i1 %.not, float %464, float %463, !dbg !9
  %480 = bitcast float %479 to i32, !dbg !9
  %481 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %480), !dbg !9
  %482 = bitcast i32 %481 to float, !dbg !9
  %483 = select i1 %.not, float %463, float %482, !dbg !9
  %484 = select i1 %.not, float %482, float %464, !dbg !9
  %485 = select i1 %.not566, float %483, float %477, !dbg !9
  %486 = xor i32 %467, 32, !dbg !9
  %487 = icmp samesign ult i32 %486, %470, !dbg !9
  %488 = select i1 %487, i32 %486, i32 %467, !dbg !9
  %489 = shl nuw nsw i32 %488, 2, !dbg !9
  %490 = bitcast float %485 to i32, !dbg !9
  %491 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %490), !dbg !9
  %492 = select i1 %.not566, float %484, float %478, !dbg !9
  %493 = bitcast float %492 to i32, !dbg !9
  %494 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %493), !dbg !9
  %495 = extractelement <32 x float> %103, i64 4, !dbg !9
  %496 = extractelement <32 x float> %103, i64 5, !dbg !9
  %497 = extractelement <32 x float> %103, i64 6, !dbg !9
  %498 = extractelement <32 x float> %103, i64 7, !dbg !9
  %499 = select i1 %.not, float %496, float %495, !dbg !9
  %500 = bitcast float %499 to i32, !dbg !9
  %501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %500), !dbg !9
  %502 = bitcast i32 %501 to float, !dbg !9
  %503 = select i1 %.not, float %495, float %502, !dbg !9
  %504 = select i1 %.not, float %502, float %496, !dbg !9
  %505 = select i1 %.not, float %498, float %497, !dbg !9
  %506 = bitcast float %505 to i32, !dbg !9
  %507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %506), !dbg !9
  %508 = bitcast i32 %507 to float, !dbg !9
  %509 = select i1 %.not, float %497, float %508, !dbg !9
  %510 = select i1 %.not, float %508, float %498, !dbg !9
  %511 = select i1 %.not566, float %509, float %503, !dbg !9
  %512 = bitcast float %511 to i32, !dbg !9
  %513 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %512), !dbg !9
  %514 = select i1 %.not566, float %510, float %504, !dbg !9
  %515 = bitcast float %514 to i32, !dbg !9
  %516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %515), !dbg !9
  %517 = extractelement <32 x float> %103, i64 8, !dbg !9
  %518 = extractelement <32 x float> %103, i64 9, !dbg !9
  %519 = extractelement <32 x float> %103, i64 10, !dbg !9
  %520 = extractelement <32 x float> %103, i64 11, !dbg !9
  %521 = select i1 %.not, float %518, float %517, !dbg !9
  %522 = bitcast float %521 to i32, !dbg !9
  %523 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %522), !dbg !9
  %524 = bitcast i32 %523 to float, !dbg !9
  %525 = select i1 %.not, float %517, float %524, !dbg !9
  %526 = select i1 %.not, float %524, float %518, !dbg !9
  %527 = select i1 %.not, float %520, float %519, !dbg !9
  %528 = bitcast float %527 to i32, !dbg !9
  %529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %528), !dbg !9
  %530 = bitcast i32 %529 to float, !dbg !9
  %531 = select i1 %.not, float %519, float %530, !dbg !9
  %532 = select i1 %.not, float %530, float %520, !dbg !9
  %533 = select i1 %.not566, float %531, float %525, !dbg !9
  %534 = bitcast float %533 to i32, !dbg !9
  %535 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %534), !dbg !9
  %536 = select i1 %.not566, float %532, float %526, !dbg !9
  %537 = bitcast float %536 to i32, !dbg !9
  %538 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %537), !dbg !9
  %539 = extractelement <32 x float> %103, i64 12, !dbg !9
  %540 = extractelement <32 x float> %103, i64 13, !dbg !9
  %541 = extractelement <32 x float> %103, i64 14, !dbg !9
  %542 = extractelement <32 x float> %103, i64 15, !dbg !9
  %543 = select i1 %.not, float %540, float %539, !dbg !9
  %544 = bitcast float %543 to i32, !dbg !9
  %545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %544), !dbg !9
  %546 = bitcast i32 %545 to float, !dbg !9
  %547 = select i1 %.not, float %539, float %546, !dbg !9
  %548 = select i1 %.not, float %546, float %540, !dbg !9
  %549 = select i1 %.not, float %542, float %541, !dbg !9
  %550 = bitcast float %549 to i32, !dbg !9
  %551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %550), !dbg !9
  %552 = bitcast i32 %551 to float, !dbg !9
  %553 = select i1 %.not, float %541, float %552, !dbg !9
  %554 = select i1 %.not, float %552, float %542, !dbg !9
  %555 = select i1 %.not566, float %553, float %547, !dbg !9
  %556 = bitcast float %555 to i32, !dbg !9
  %557 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %556), !dbg !9
  %558 = select i1 %.not566, float %554, float %548, !dbg !9
  %559 = bitcast float %558 to i32, !dbg !9
  %560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %559), !dbg !9
  %561 = extractelement <32 x float> %103, i64 16, !dbg !9
  %562 = extractelement <32 x float> %103, i64 17, !dbg !9
  %563 = extractelement <32 x float> %103, i64 18, !dbg !9
  %564 = extractelement <32 x float> %103, i64 19, !dbg !9
  %565 = select i1 %.not, float %562, float %561, !dbg !9
  %566 = bitcast float %565 to i32, !dbg !9
  %567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %566), !dbg !9
  %568 = bitcast i32 %567 to float, !dbg !9
  %569 = select i1 %.not, float %561, float %568, !dbg !9
  %570 = select i1 %.not, float %568, float %562, !dbg !9
  %571 = select i1 %.not, float %564, float %563, !dbg !9
  %572 = bitcast float %571 to i32, !dbg !9
  %573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %572), !dbg !9
  %574 = bitcast i32 %573 to float, !dbg !9
  %575 = select i1 %.not, float %563, float %574, !dbg !9
  %576 = select i1 %.not, float %574, float %564, !dbg !9
  %577 = select i1 %.not566, float %575, float %569, !dbg !9
  %578 = bitcast float %577 to i32, !dbg !9
  %579 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %578), !dbg !9
  %580 = select i1 %.not566, float %576, float %570, !dbg !9
  %581 = bitcast float %580 to i32, !dbg !9
  %582 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %581), !dbg !9
  %583 = extractelement <32 x float> %103, i64 20, !dbg !9
  %584 = extractelement <32 x float> %103, i64 21, !dbg !9
  %585 = extractelement <32 x float> %103, i64 22, !dbg !9
  %586 = extractelement <32 x float> %103, i64 23, !dbg !9
  %587 = select i1 %.not, float %584, float %583, !dbg !9
  %588 = bitcast float %587 to i32, !dbg !9
  %589 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %588), !dbg !9
  %590 = bitcast i32 %589 to float, !dbg !9
  %591 = select i1 %.not, float %583, float %590, !dbg !9
  %592 = select i1 %.not, float %590, float %584, !dbg !9
  %593 = select i1 %.not, float %586, float %585, !dbg !9
  %594 = bitcast float %593 to i32, !dbg !9
  %595 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %594), !dbg !9
  %596 = bitcast i32 %595 to float, !dbg !9
  %597 = select i1 %.not, float %585, float %596, !dbg !9
  %598 = select i1 %.not, float %596, float %586, !dbg !9
  %599 = select i1 %.not566, float %597, float %591, !dbg !9
  %600 = bitcast float %599 to i32, !dbg !9
  %601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %600), !dbg !9
  %602 = select i1 %.not566, float %598, float %592, !dbg !9
  %603 = bitcast float %602 to i32, !dbg !9
  %604 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %603), !dbg !9
  %605 = extractelement <32 x float> %103, i64 24, !dbg !9
  %606 = extractelement <32 x float> %103, i64 25, !dbg !9
  %607 = extractelement <32 x float> %103, i64 26, !dbg !9
  %608 = extractelement <32 x float> %103, i64 27, !dbg !9
  %609 = select i1 %.not, float %606, float %605, !dbg !9
  %610 = bitcast float %609 to i32, !dbg !9
  %611 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %610), !dbg !9
  %612 = bitcast i32 %611 to float, !dbg !9
  %613 = select i1 %.not, float %605, float %612, !dbg !9
  %614 = select i1 %.not, float %612, float %606, !dbg !9
  %615 = select i1 %.not, float %608, float %607, !dbg !9
  %616 = bitcast float %615 to i32, !dbg !9
  %617 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %616), !dbg !9
  %618 = bitcast i32 %617 to float, !dbg !9
  %619 = select i1 %.not, float %607, float %618, !dbg !9
  %620 = select i1 %.not, float %618, float %608, !dbg !9
  %621 = select i1 %.not566, float %619, float %613, !dbg !9
  %622 = bitcast float %621 to i32, !dbg !9
  %623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %622), !dbg !9
  %624 = select i1 %.not566, float %620, float %614, !dbg !9
  %625 = bitcast float %624 to i32, !dbg !9
  %626 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %625), !dbg !9
  %627 = extractelement <32 x float> %103, i64 28, !dbg !9
  %628 = extractelement <32 x float> %103, i64 29, !dbg !9
  %629 = extractelement <32 x float> %103, i64 30, !dbg !9
  %630 = extractelement <32 x float> %103, i64 31, !dbg !9
  %631 = select i1 %.not, float %628, float %627, !dbg !9
  %632 = bitcast float %631 to i32, !dbg !9
  %633 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %632), !dbg !9
  %634 = bitcast i32 %633 to float, !dbg !9
  %635 = select i1 %.not, float %627, float %634, !dbg !9
  %636 = select i1 %.not, float %634, float %628, !dbg !9
  %637 = select i1 %.not, float %630, float %629, !dbg !9
  %638 = bitcast float %637 to i32, !dbg !9
  %639 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %638), !dbg !9
  %640 = bitcast i32 %639 to float, !dbg !9
  %641 = select i1 %.not, float %629, float %640, !dbg !9
  %642 = select i1 %.not, float %640, float %630, !dbg !9
  %643 = select i1 %.not566, float %641, float %635, !dbg !9
  %644 = bitcast float %643 to i32, !dbg !9
  %645 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %644), !dbg !9
  %646 = select i1 %.not566, float %642, float %636, !dbg !9
  %647 = bitcast float %646 to i32, !dbg !9
  %648 = call i32 @llvm.amdgcn.ds.bpermute(i32 %489, i32 %647), !dbg !9
  br i1 true, label %649, label %780, !dbg !23

649:                                              ; preds = %460
  %650 = bitcast i32 %645 to float, !dbg !9
  %651 = select i1 %.not566, float %635, float %650, !dbg !9
  %652 = insertelement <4 x float> poison, float %651, i64 0, !dbg !21
  %653 = bitcast i32 %648 to float, !dbg !9
  %654 = select i1 %.not566, float %636, float %653, !dbg !9
  %655 = insertelement <4 x float> %652, float %654, i64 1, !dbg !21
  %656 = select i1 %.not566, float %650, float %641, !dbg !9
  %657 = insertelement <4 x float> %655, float %656, i64 2, !dbg !21
  %658 = select i1 %.not566, float %653, float %642, !dbg !9
  %659 = insertelement <4 x float> %657, float %658, i64 3, !dbg !21
  %660 = shufflevector <1 x float> %104, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %661 = fdiv <4 x float> %659, %660, !dbg !21
  %662 = fptrunc <4 x float> %661 to <4 x half>, !dbg !22
  %663 = extractelement <4 x half> %662, i64 3, !dbg !22
  %664 = extractelement <4 x half> %662, i64 2, !dbg !22
  %665 = extractelement <4 x half> %662, i64 1, !dbg !22
  %666 = extractelement <4 x half> %662, i64 0, !dbg !22
  %667 = bitcast i32 %623 to float, !dbg !9
  %668 = select i1 %.not566, float %613, float %667, !dbg !9
  %669 = insertelement <4 x float> poison, float %668, i64 0, !dbg !21
  %670 = bitcast i32 %626 to float, !dbg !9
  %671 = select i1 %.not566, float %614, float %670, !dbg !9
  %672 = insertelement <4 x float> %669, float %671, i64 1, !dbg !21
  %673 = select i1 %.not566, float %667, float %619, !dbg !9
  %674 = insertelement <4 x float> %672, float %673, i64 2, !dbg !21
  %675 = select i1 %.not566, float %670, float %620, !dbg !9
  %676 = insertelement <4 x float> %674, float %675, i64 3, !dbg !21
  %677 = fdiv <4 x float> %676, %660, !dbg !21
  %678 = fptrunc <4 x float> %677 to <4 x half>, !dbg !22
  %679 = extractelement <4 x half> %678, i64 3, !dbg !22
  %680 = extractelement <4 x half> %678, i64 2, !dbg !22
  %681 = extractelement <4 x half> %678, i64 1, !dbg !22
  %682 = extractelement <4 x half> %678, i64 0, !dbg !22
  %683 = bitcast i32 %601 to float, !dbg !9
  %684 = select i1 %.not566, float %591, float %683, !dbg !9
  %685 = insertelement <4 x float> poison, float %684, i64 0, !dbg !21
  %686 = bitcast i32 %604 to float, !dbg !9
  %687 = select i1 %.not566, float %592, float %686, !dbg !9
  %688 = insertelement <4 x float> %685, float %687, i64 1, !dbg !21
  %689 = select i1 %.not566, float %683, float %597, !dbg !9
  %690 = insertelement <4 x float> %688, float %689, i64 2, !dbg !21
  %691 = select i1 %.not566, float %686, float %598, !dbg !9
  %692 = insertelement <4 x float> %690, float %691, i64 3, !dbg !21
  %693 = fdiv <4 x float> %692, %660, !dbg !21
  %694 = fptrunc <4 x float> %693 to <4 x half>, !dbg !22
  %695 = extractelement <4 x half> %694, i64 3, !dbg !22
  %696 = extractelement <4 x half> %694, i64 2, !dbg !22
  %697 = extractelement <4 x half> %694, i64 1, !dbg !22
  %698 = extractelement <4 x half> %694, i64 0, !dbg !22
  %699 = bitcast i32 %579 to float, !dbg !9
  %700 = select i1 %.not566, float %569, float %699, !dbg !9
  %701 = insertelement <4 x float> poison, float %700, i64 0, !dbg !21
  %702 = bitcast i32 %582 to float, !dbg !9
  %703 = select i1 %.not566, float %570, float %702, !dbg !9
  %704 = insertelement <4 x float> %701, float %703, i64 1, !dbg !21
  %705 = select i1 %.not566, float %699, float %575, !dbg !9
  %706 = insertelement <4 x float> %704, float %705, i64 2, !dbg !21
  %707 = select i1 %.not566, float %702, float %576, !dbg !9
  %708 = insertelement <4 x float> %706, float %707, i64 3, !dbg !21
  %709 = fdiv <4 x float> %708, %660, !dbg !21
  %710 = fptrunc <4 x float> %709 to <4 x half>, !dbg !22
  %711 = extractelement <4 x half> %710, i64 3, !dbg !22
  %712 = extractelement <4 x half> %710, i64 2, !dbg !22
  %713 = extractelement <4 x half> %710, i64 1, !dbg !22
  %714 = extractelement <4 x half> %710, i64 0, !dbg !22
  %715 = bitcast i32 %557 to float, !dbg !9
  %716 = select i1 %.not566, float %547, float %715, !dbg !9
  %717 = insertelement <4 x float> poison, float %716, i64 0, !dbg !21
  %718 = bitcast i32 %560 to float, !dbg !9
  %719 = select i1 %.not566, float %548, float %718, !dbg !9
  %720 = insertelement <4 x float> %717, float %719, i64 1, !dbg !21
  %721 = select i1 %.not566, float %715, float %553, !dbg !9
  %722 = insertelement <4 x float> %720, float %721, i64 2, !dbg !21
  %723 = select i1 %.not566, float %718, float %554, !dbg !9
  %724 = insertelement <4 x float> %722, float %723, i64 3, !dbg !21
  %725 = fdiv <4 x float> %724, %660, !dbg !21
  %726 = fptrunc <4 x float> %725 to <4 x half>, !dbg !22
  %727 = extractelement <4 x half> %726, i64 3, !dbg !22
  %728 = extractelement <4 x half> %726, i64 2, !dbg !22
  %729 = extractelement <4 x half> %726, i64 1, !dbg !22
  %730 = extractelement <4 x half> %726, i64 0, !dbg !22
  %731 = bitcast i32 %535 to float, !dbg !9
  %732 = select i1 %.not566, float %525, float %731, !dbg !9
  %733 = insertelement <4 x float> poison, float %732, i64 0, !dbg !21
  %734 = bitcast i32 %538 to float, !dbg !9
  %735 = select i1 %.not566, float %526, float %734, !dbg !9
  %736 = insertelement <4 x float> %733, float %735, i64 1, !dbg !21
  %737 = select i1 %.not566, float %731, float %531, !dbg !9
  %738 = insertelement <4 x float> %736, float %737, i64 2, !dbg !21
  %739 = select i1 %.not566, float %734, float %532, !dbg !9
  %740 = insertelement <4 x float> %738, float %739, i64 3, !dbg !21
  %741 = fdiv <4 x float> %740, %660, !dbg !21
  %742 = fptrunc <4 x float> %741 to <4 x half>, !dbg !22
  %743 = extractelement <4 x half> %742, i64 3, !dbg !22
  %744 = extractelement <4 x half> %742, i64 2, !dbg !22
  %745 = extractelement <4 x half> %742, i64 1, !dbg !22
  %746 = extractelement <4 x half> %742, i64 0, !dbg !22
  %747 = bitcast i32 %513 to float, !dbg !9
  %748 = select i1 %.not566, float %503, float %747, !dbg !9
  %749 = insertelement <4 x float> poison, float %748, i64 0, !dbg !21
  %750 = bitcast i32 %516 to float, !dbg !9
  %751 = select i1 %.not566, float %504, float %750, !dbg !9
  %752 = insertelement <4 x float> %749, float %751, i64 1, !dbg !21
  %753 = select i1 %.not566, float %747, float %509, !dbg !9
  %754 = insertelement <4 x float> %752, float %753, i64 2, !dbg !21
  %755 = select i1 %.not566, float %750, float %510, !dbg !9
  %756 = insertelement <4 x float> %754, float %755, i64 3, !dbg !21
  %757 = fdiv <4 x float> %756, %660, !dbg !21
  %758 = fptrunc <4 x float> %757 to <4 x half>, !dbg !22
  %759 = extractelement <4 x half> %758, i64 3, !dbg !22
  %760 = extractelement <4 x half> %758, i64 2, !dbg !22
  %761 = extractelement <4 x half> %758, i64 1, !dbg !22
  %762 = extractelement <4 x half> %758, i64 0, !dbg !22
  %763 = bitcast i32 %491 to float, !dbg !9
  %764 = select i1 %.not566, float %477, float %763, !dbg !9
  %765 = insertelement <4 x float> poison, float %764, i64 0, !dbg !21
  %766 = bitcast i32 %494 to float, !dbg !9
  %767 = select i1 %.not566, float %478, float %766, !dbg !9
  %768 = insertelement <4 x float> %765, float %767, i64 1, !dbg !21
  %769 = select i1 %.not566, float %763, float %483, !dbg !9
  %770 = insertelement <4 x float> %768, float %769, i64 2, !dbg !21
  %771 = select i1 %.not566, float %766, float %484, !dbg !9
  %772 = insertelement <4 x float> %770, float %771, i64 3, !dbg !21
  %773 = fdiv <4 x float> %772, %660, !dbg !21
  %774 = fptrunc <4 x float> %773 to <4 x half>, !dbg !22
  %775 = extractelement <4 x half> %774, i64 3, !dbg !22
  %776 = extractelement <4 x half> %774, i64 2, !dbg !22
  %777 = extractelement <4 x half> %774, i64 1, !dbg !22
  %778 = extractelement <4 x half> %774, i64 0, !dbg !22
  %779 = shl nuw nsw i64 %14, 19, !dbg !23
  %address.thread1298 = call i32 asm sideeffect "", "=v,0"(i32 %18) #8, !dbg !23, !range !31, !frisk.register_lifetimes !1
  %address.local1299 = zext nneg i32 %address.thread1298 to i64, !dbg !12
  %address.local1300 = add nsw i64 %15, %address.local1299, !dbg !23
  %address.local1301 = lshr i32 %address.thread1298, 2, !dbg !16
  %address.local1302 = and i32 %address.local1301, 48, !dbg !16
  %address.local1303 = zext nneg i32 %address.local1302 to i64, !dbg !16
  %address.local1304 = add nsw i64 %address.local1300, %address.local1303, !dbg !23
  %address.local1305 = and i32 %address.thread1298, 240, !dbg !16
  %address.local1306 = sub nsw i32 0, %address.local1305, !dbg !16
  %address.local1307 = sext i32 %address.local1306 to i64, !dbg !16
  %address.local1308 = add nsw i64 %address.local1304, %address.local1307, !dbg !23
  %address.local1309 = add nsw i64 %address.local1308, %16, !dbg !23
  %address.local1310 = shl nuw nsw i64 %address.local1309, 7, !dbg !23
  %address.local1311 = add nuw nsw i64 %779, %address.local1310, !dbg !23
  %address.local1312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1313 = lshr i32 %address.thread1298, 2, !dbg !16
  %address.local1314 = and i32 %address.local1313, 12, !dbg !16
  %address.local1315 = zext nneg i32 %address.local1314 to i64, !dbg !16
  %address.local1316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1312, i64 %address.local1315, !dbg !23
  store half %778, ptr addrspace(1) %address.local1316, align 2, !dbg !23
  %address.local1317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1318 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1319 = and i32 %address.local1318, 12, !dbg !23
  %address.local1320 = zext nneg i32 %address.local1319 to i64, !dbg !23
  %address.local1321 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1317, i64 %address.local1320, !dbg !23
  %address.local1322 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1321, i64 2, !dbg !23
  store half %777, ptr addrspace(1) %address.local1322, align 2, !dbg !23
  %address.local1323 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1324 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1325 = and i32 %address.local1324, 12, !dbg !23
  %address.local1326 = zext nneg i32 %address.local1325 to i64, !dbg !23
  %address.local1327 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1323, i64 %address.local1326, !dbg !23
  %address.local1328 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1327, i64 4, !dbg !23
  store half %776, ptr addrspace(1) %address.local1328, align 2, !dbg !23
  %address.local1329 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1330 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1331 = and i32 %address.local1330, 12, !dbg !23
  %address.local1332 = zext nneg i32 %address.local1331 to i64, !dbg !23
  %address.local1333 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1329, i64 %address.local1332, !dbg !23
  %address.local1334 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1333, i64 6, !dbg !23
  store half %775, ptr addrspace(1) %address.local1334, align 2, !dbg !23
  %address.local1335 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1336 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1337 = and i32 %address.local1336, 12, !dbg !23
  %address.local1338 = zext nneg i32 %address.local1337 to i64, !dbg !23
  %address.local1339 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1335, i64 %address.local1338, !dbg !23
  %address.local1340 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1339, i64 32, !dbg !23
  store half %762, ptr addrspace(1) %address.local1340, align 2, !dbg !23
  %address.local1341 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1342 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1343 = and i32 %address.local1342, 12, !dbg !23
  %address.local1344 = zext nneg i32 %address.local1343 to i64, !dbg !23
  %address.local1345 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1341, i64 %address.local1344, !dbg !23
  %address.local1346 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1345, i64 34, !dbg !23
  store half %761, ptr addrspace(1) %address.local1346, align 2, !dbg !23
  %address.local1347 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1348 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1349 = and i32 %address.local1348, 12, !dbg !23
  %address.local1350 = zext nneg i32 %address.local1349 to i64, !dbg !23
  %address.local1351 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1347, i64 %address.local1350, !dbg !23
  %address.local1352 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1351, i64 36, !dbg !23
  store half %760, ptr addrspace(1) %address.local1352, align 2, !dbg !23
  %address.local1353 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1354 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1355 = and i32 %address.local1354, 12, !dbg !23
  %address.local1356 = zext nneg i32 %address.local1355 to i64, !dbg !23
  %address.local1357 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1353, i64 %address.local1356, !dbg !23
  %address.local1358 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1357, i64 38, !dbg !23
  store half %759, ptr addrspace(1) %address.local1358, align 2, !dbg !23
  %address.local1359 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1360 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1361 = and i32 %address.local1360, 12, !dbg !23
  %address.local1362 = zext nneg i32 %address.local1361 to i64, !dbg !23
  %address.local1363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1359, i64 %address.local1362, !dbg !23
  %address.local1364 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1363, i64 64, !dbg !23
  store half %746, ptr addrspace(1) %address.local1364, align 2, !dbg !23
  %address.local1365 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1366 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1367 = and i32 %address.local1366, 12, !dbg !23
  %address.local1368 = zext nneg i32 %address.local1367 to i64, !dbg !23
  %address.local1369 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1365, i64 %address.local1368, !dbg !23
  %address.local1370 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1369, i64 66, !dbg !23
  store half %745, ptr addrspace(1) %address.local1370, align 2, !dbg !23
  %address.local1371 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1372 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1373 = and i32 %address.local1372, 12, !dbg !23
  %address.local1374 = zext nneg i32 %address.local1373 to i64, !dbg !23
  %address.local1375 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1371, i64 %address.local1374, !dbg !23
  %address.local1376 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1375, i64 68, !dbg !23
  store half %744, ptr addrspace(1) %address.local1376, align 2, !dbg !23
  %address.local1377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1378 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1379 = and i32 %address.local1378, 12, !dbg !23
  %address.local1380 = zext nneg i32 %address.local1379 to i64, !dbg !23
  %address.local1381 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1377, i64 %address.local1380, !dbg !23
  %address.local1382 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1381, i64 70, !dbg !23
  store half %743, ptr addrspace(1) %address.local1382, align 2, !dbg !23
  %address.local1383 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1384 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1385 = and i32 %address.local1384, 12, !dbg !23
  %address.local1386 = zext nneg i32 %address.local1385 to i64, !dbg !23
  %address.local1387 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1383, i64 %address.local1386, !dbg !23
  %address.local1388 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1387, i64 96, !dbg !23
  store half %730, ptr addrspace(1) %address.local1388, align 2, !dbg !23
  %address.local1389 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1390 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1391 = and i32 %address.local1390, 12, !dbg !23
  %address.local1392 = zext nneg i32 %address.local1391 to i64, !dbg !23
  %address.local1393 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1389, i64 %address.local1392, !dbg !23
  %address.local1394 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1393, i64 98, !dbg !23
  store half %729, ptr addrspace(1) %address.local1394, align 2, !dbg !23
  %address.local1395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1396 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1397 = and i32 %address.local1396, 12, !dbg !23
  %address.local1398 = zext nneg i32 %address.local1397 to i64, !dbg !23
  %address.local1399 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1395, i64 %address.local1398, !dbg !23
  %address.local1400 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1399, i64 100, !dbg !23
  store half %728, ptr addrspace(1) %address.local1400, align 2, !dbg !23
  %address.local1401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1402 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1403 = or i32 %address.local1402, 51, !dbg !23
  %address.local1404 = zext nneg i32 %address.local1403 to i64, !dbg !23
  %address.local1405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1401, i64 %address.local1404, !dbg !23
  store half %727, ptr addrspace(1) %address.local1405, align 2, !dbg !23
  %address.local1406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1407 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1408 = and i32 %address.local1407, 12, !dbg !23
  %address.local1409 = zext nneg i32 %address.local1408 to i64, !dbg !23
  %address.local1410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1406, i64 %address.local1409, !dbg !23
  %address.local1411 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1410, i64 128, !dbg !23
  store half %714, ptr addrspace(1) %address.local1411, align 2, !dbg !23
  %address.local1412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1413 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1414 = and i32 %address.local1413, 12, !dbg !23
  %address.local1415 = zext nneg i32 %address.local1414 to i64, !dbg !23
  %address.local1416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1412, i64 %address.local1415, !dbg !23
  %address.local1417 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1416, i64 130, !dbg !23
  store half %713, ptr addrspace(1) %address.local1417, align 2, !dbg !23
  %address.local1418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1419 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1420 = and i32 %address.local1419, 12, !dbg !23
  %address.local1421 = zext nneg i32 %address.local1420 to i64, !dbg !23
  %address.local1422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1418, i64 %address.local1421, !dbg !23
  %address.local1423 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1422, i64 132, !dbg !23
  store half %712, ptr addrspace(1) %address.local1423, align 2, !dbg !23
  %address.local1424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1425 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1426 = and i32 %address.local1425, 12, !dbg !23
  %address.local1427 = zext nneg i32 %address.local1426 to i64, !dbg !23
  %address.local1428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1424, i64 %address.local1427, !dbg !23
  %address.local1429 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1428, i64 134, !dbg !23
  store half %711, ptr addrspace(1) %address.local1429, align 2, !dbg !23
  %address.local1430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1431 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1432 = and i32 %address.local1431, 12, !dbg !23
  %address.local1433 = zext nneg i32 %address.local1432 to i64, !dbg !23
  %address.local1434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1430, i64 %address.local1433, !dbg !23
  %address.local1435 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1434, i64 160, !dbg !23
  store half %698, ptr addrspace(1) %address.local1435, align 2, !dbg !23
  %address.local1436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1437 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1438 = and i32 %address.local1437, 12, !dbg !23
  %address.local1439 = zext nneg i32 %address.local1438 to i64, !dbg !23
  %address.local1440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1436, i64 %address.local1439, !dbg !23
  %address.local1441 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1440, i64 162, !dbg !23
  store half %697, ptr addrspace(1) %address.local1441, align 2, !dbg !23
  %address.local1442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1443 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1444 = and i32 %address.local1443, 12, !dbg !23
  %address.local1445 = zext nneg i32 %address.local1444 to i64, !dbg !23
  %address.local1446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1442, i64 %address.local1445, !dbg !23
  %address.local1447 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1446, i64 164, !dbg !23
  store half %696, ptr addrspace(1) %address.local1447, align 2, !dbg !23
  %address.local1448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1449 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1450 = and i32 %address.local1449, 12, !dbg !23
  %address.local1451 = zext nneg i32 %address.local1450 to i64, !dbg !23
  %address.local1452 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1448, i64 %address.local1451, !dbg !23
  %address.local1453 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1452, i64 166, !dbg !23
  store half %695, ptr addrspace(1) %address.local1453, align 2, !dbg !23
  %address.local1454 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1455 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1456 = and i32 %address.local1455, 12, !dbg !23
  %address.local1457 = zext nneg i32 %address.local1456 to i64, !dbg !23
  %address.local1458 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1454, i64 %address.local1457, !dbg !23
  %address.local1459 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1458, i64 192, !dbg !23
  store half %682, ptr addrspace(1) %address.local1459, align 2, !dbg !23
  %address.local1460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1461 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1462 = and i32 %address.local1461, 12, !dbg !23
  %address.local1463 = zext nneg i32 %address.local1462 to i64, !dbg !23
  %address.local1464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1460, i64 %address.local1463, !dbg !23
  %address.local1465 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1464, i64 194, !dbg !23
  store half %681, ptr addrspace(1) %address.local1465, align 2, !dbg !23
  %address.local1466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1467 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1468 = and i32 %address.local1467, 12, !dbg !23
  %address.local1469 = zext nneg i32 %address.local1468 to i64, !dbg !23
  %address.local1470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1466, i64 %address.local1469, !dbg !23
  %address.local1471 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1470, i64 196, !dbg !23
  store half %680, ptr addrspace(1) %address.local1471, align 2, !dbg !23
  %address.local1472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1473 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1474 = and i32 %address.local1473, 12, !dbg !23
  %address.local1475 = zext nneg i32 %address.local1474 to i64, !dbg !23
  %address.local1476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1472, i64 %address.local1475, !dbg !23
  %address.local1477 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1476, i64 198, !dbg !23
  store half %679, ptr addrspace(1) %address.local1477, align 2, !dbg !23
  %address.local1478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1479 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1480 = and i32 %address.local1479, 12, !dbg !23
  %address.local1481 = zext nneg i32 %address.local1480 to i64, !dbg !23
  %address.local1482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1478, i64 %address.local1481, !dbg !23
  %address.local1483 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1482, i64 224, !dbg !23
  store half %666, ptr addrspace(1) %address.local1483, align 2, !dbg !23
  %address.local1484 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1485 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1486 = and i32 %address.local1485, 12, !dbg !23
  %address.local1487 = zext nneg i32 %address.local1486 to i64, !dbg !23
  %address.local1488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1484, i64 %address.local1487, !dbg !23
  %address.local1489 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1488, i64 226, !dbg !23
  store half %665, ptr addrspace(1) %address.local1489, align 2, !dbg !23
  %address.local1490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1491 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1492 = and i32 %address.local1491, 12, !dbg !23
  %address.local1493 = zext nneg i32 %address.local1492 to i64, !dbg !23
  %address.local1494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1490, i64 %address.local1493, !dbg !23
  %address.local1495 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1494, i64 228, !dbg !23
  store half %664, ptr addrspace(1) %address.local1495, align 2, !dbg !23
  %address.local1496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1311, !dbg !23
  %address.local1497 = lshr i32 %address.thread1298, 2, !dbg !23
  %address.local1498 = or i32 %address.local1497, 115, !dbg !23
  %address.local1499 = zext nneg i32 %address.local1498 to i64, !dbg !23
  %address.local1500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1496, i64 %address.local1499, !dbg !23
  store half %663, ptr addrspace(1) %address.local1500, align 2, !dbg !23
  br label %780, !dbg !23

780:                                              ; preds = %649, %460
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) captures(none), ptr addrspace(3) captures(none), i32 immarg, i32 immarg, i32 immarg) #4

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind memory(none) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!30}

!0 = !DIFile(filename: "test_input.mlir", directory: "3rd/deepgengraph/test")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!6 = !DILocation(line: 60, column: 15, scope: !4)
!7 = !DILocation(line: 51, column: 13, scope: !4)
!8 = !DILocation(line: 62, column: 15, scope: !4)
!9 = !DILocation(line: 59, column: 15, scope: !4)
!12 = !DILocation(line: 48, column: 13, scope: !4)
!13 = !DILocation(line: 44, column: 12, scope: !4)
!16 = !DILocation(line: 63, column: 15, scope: !4)
!17 = !DILocation(line: 65, column: 17, scope: !4)
!18 = !DILocation(line: 77, column: 15, scope: !4)
!19 = !DILocation(line: 61, column: 15, scope: !4)
!20 = !DILocation(line: 78, column: 15, scope: !4)
!21 = !DILocation(line: 84, column: 13, scope: !4)
!22 = !DILocation(line: 85, column: 13, scope: !4)
!23 = !DILocation(line: 86, column: 7, scope: !4)
!24 = !DILocation(line: 66, column: 17, scope: !4)
!25 = !DILocation(line: 67, column: 17, scope: !4)
!26 = !DILocation(line: 75, column: 15, scope: !4)
!27 = !DILocation(line: 76, column: 15, scope: !4)
!28 = !DILocation(line: 79, column: 15, scope: !4)
!29 = !{i32 256, i32 1, i32 1}
!30 = !{i32 2, !"Debug Info Version", i32 3}
!31 = !{i32 0, i32 256}
