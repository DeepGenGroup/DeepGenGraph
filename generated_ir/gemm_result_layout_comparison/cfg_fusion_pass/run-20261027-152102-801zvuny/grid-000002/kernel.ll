; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-152102-801zvuny/grid-000002/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-152102-801zvuny/grid-000002/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 5, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423484, !dbg !5
  %13 = and i64 %6, 3, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = sub nsw i64 31, %14, !dbg !5
  %16 = lshr i64 %10, 2, !dbg !5
  %17 = icmp ugt i64 %10, 1023, !dbg !8
  %18 = add nsw i64 %14, -32, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 32, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %16, %22, !dbg !8
  %24 = and i64 %23, 31, !dbg !8
  %25 = shl nsw i64 %15, 7, !dbg !8
  %26 = mul nsw i64 %22, -4096, !dbg !8
  %27 = add i64 %25, %26, !dbg !8
  %28 = shl nuw nsw i64 %24, 19, !dbg !8
  %29 = shl nsw i64 %27, 7, !dbg !8
  %30 = add i64 %28, %29, !dbg !8
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !39
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !8, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local505 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local506 = and i64 %address.local505, 1536, !dbg !8
  %address.local507 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local506, !dbg !8
  %address.local508 = ptrtoint ptr addrspace(3) %address.local507 to i64, !dbg !8
  %address.local509 = trunc i64 %address.local508 to i32, !dbg !8
  %33 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local509), !dbg !8
  %address.local510 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local511 = and i64 %address.local510, 16256, !dbg !8
  %address.local512 = lshr i64 %address.local, 4, !dbg !8
  %address.local513 = and i64 %address.local512, 8, !dbg !8
  %address.local514 = or disjoint i64 %address.local511, %address.local513, !dbg !8
  %34 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1781 = or disjoint i64 %30, %address.local514, !dbg !8
  %.idx1781.tr = trunc i64 %.idx1781 to i32, !dbg !8
  %dma.buffer.offset1778 = shl i32 %.idx1781.tr, 1, !dbg !8
  %35 = insertelement <2 x i64> poison, i64 %34, i64 0, !dbg !8
  %36 = bitcast <2 x i64> %35 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1779 = shufflevector <4 x i32> %36, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1780 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %33), !dbg !8
  %37 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1780, i32 %dma.buffer.offset1778, <4 x i32> %dma.buffer.rsrc1779) #10, !dbg !8
  %address.local516 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !8
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !8
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !8
  %address.local522 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local523 = and i64 %address.local522, 16256, !dbg !8
  %address.local524 = lshr i64 %address.local516, 4, !dbg !8
  %address.local525 = and i64 %address.local524, 24, !dbg !8
  %address.local526 = or disjoint i64 %address.local523, %address.local525, !dbg !8
  %39 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1782 = or disjoint i64 %30, %address.local526, !dbg !8
  %.idx1782.tr = trunc i64 %.idx1782 to i32, !dbg !8
  %dma.buffer.offset1775 = shl i32 %.idx1782.tr, 1, !dbg !8
  %40 = insertelement <2 x i64> poison, i64 %39, i64 0, !dbg !8
  %41 = bitcast <2 x i64> %40 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1776 = shufflevector <4 x i32> %41, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1777 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38), !dbg !8
  %42 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1777, i32 %dma.buffer.offset1775, <4 x i32> %dma.buffer.rsrc1776) #10, !dbg !8
  %43 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %44 = sext i32 %43 to i64, !dbg !5
  %45 = shl i64 %15, 7, !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local528 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local506, !dbg !8
  %address.local529 = ptrtoint ptr addrspace(3) %address.local528 to i64, !dbg !8
  %address.local530 = trunc i64 %address.local529 to i32, !dbg !8
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local530), !dbg !8
  %47 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %48 = or disjoint i64 %30, 32, !dbg !8
  %49 = or disjoint i64 %address.local514, %48, !dbg !8
  %.tr = trunc i64 %49 to i32, !dbg !8
  %dma.buffer.offset1772 = shl i32 %.tr, 1, !dbg !8
  %50 = insertelement <2 x i64> poison, i64 %47, i64 0, !dbg !8
  %51 = bitcast <2 x i64> %50 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1773 = shufflevector <4 x i32> %51, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1774 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %52 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1774, i32 %dma.buffer.offset1772, <4 x i32> %dma.buffer.rsrc1773) #10, !dbg !8
  %address.local532 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %address.local533 = ptrtoint ptr addrspace(3) %address.local532 to i64, !dbg !8
  %address.local534 = trunc i64 %address.local533 to i32, !dbg !8
  %53 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local534), !dbg !8
  %54 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %55 = or disjoint i64 %30, 32, !dbg !8
  %56 = or disjoint i64 %address.local526, %55, !dbg !8
  %.tr1783 = trunc i64 %56 to i32, !dbg !8
  %dma.buffer.offset1769 = shl i32 %.tr1783, 1, !dbg !8
  %57 = insertelement <2 x i64> poison, i64 %54, i64 0, !dbg !8
  %58 = bitcast <2 x i64> %57 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1770 = shufflevector <4 x i32> %58, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1771 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %59 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1771, i32 %dma.buffer.offset1769, <4 x i32> %dma.buffer.rsrc1770) #10, !dbg !8
  %address.local536 = lshr i32 %address.thread, 2, !dbg !11
  %address.local537 = and i32 %address.local536, 4, !dbg !11
  %address.local538 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local539 = and i32 %address.local538, 384, !dbg !11
  %address.local540 = or disjoint i32 %address.local537, %address.local539, !dbg !11
  %address.local541 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local542 = and i32 %address.local541, 120, !dbg !11
  %address.local543 = or disjoint i32 %address.local540, %address.local542, !dbg !11
  %address.local544 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local545 = and i32 %address.local544, 1024, !dbg !11
  %address.local546 = or disjoint i32 %address.local543, %address.local545, !dbg !11
  %address.local547 = zext nneg i32 %address.local546 to i64, !dbg !11
  %address.local548 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local547, !dbg !11
  %60 = load <4 x half>, ptr addrspace(3) %address.local548, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local548, i64 4096, !dbg !11
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %62 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !11
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local548, i64 1024, !dbg !11
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !11
  %64 = fmul <4 x half> %61, splat (half 0xH3015), !dbg !11
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local548, i64 5120, !dbg !11
  %65 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !11
  %66 = fmul <4 x half> %63, splat (half 0xH3015), !dbg !11
  %67 = fmul <4 x half> %65, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local509), !dbg !8
  %69 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %70 = or disjoint i64 %30, 64, !dbg !8
  %71 = or disjoint i64 %address.local514, %70, !dbg !8
  %.tr1784 = trunc i64 %71 to i32, !dbg !8
  %dma.buffer.offset1766 = shl i32 %.tr1784, 1, !dbg !8
  %72 = insertelement <2 x i64> poison, i64 %69, i64 0, !dbg !8
  %73 = bitcast <2 x i64> %72 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1767 = shufflevector <4 x i32> %73, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1768 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %68), !dbg !8
  %74 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1768, i32 %dma.buffer.offset1766, <4 x i32> %dma.buffer.rsrc1767) #10, !dbg !8
  %75 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !8
  %76 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %77 = or disjoint i64 %30, 64, !dbg !8
  %78 = or disjoint i64 %address.local526, %77, !dbg !8
  %.tr1785 = trunc i64 %78 to i32, !dbg !8
  %dma.buffer.offset1763 = shl i32 %.tr1785, 1, !dbg !8
  %79 = insertelement <2 x i64> poison, i64 %76, i64 0, !dbg !8
  %80 = bitcast <2 x i64> %79 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1764 = shufflevector <4 x i32> %80, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1765 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %75), !dbg !8
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1765, i32 %dma.buffer.offset1763, <4 x i32> %dma.buffer.rsrc1764) #10, !dbg !8
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local548, i64 8192, !dbg !11
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !11
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local548, i64 12288, !dbg !11
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !11
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local548, i64 9216, !dbg !11
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !11
  %85 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !11
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local548, i64 13312, !dbg !11
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !11
  %87 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !11
  %88 = fmul <4 x half> %84, splat (half 0xH3015), !dbg !11
  %89 = fmul <4 x half> %86, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %90 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local530), !dbg !8
  %91 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %92 = or disjoint i64 %30, 96, !dbg !8
  %93 = or disjoint i64 %address.local514, %92, !dbg !8
  %.tr1786 = trunc i64 %93 to i32, !dbg !8
  %dma.buffer.offset1760 = shl i32 %.tr1786, 1, !dbg !8
  %94 = insertelement <2 x i64> poison, i64 %91, i64 0, !dbg !8
  %95 = bitcast <2 x i64> %94 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1761 = shufflevector <4 x i32> %95, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1762 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %90), !dbg !8
  %96 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1762, i32 %dma.buffer.offset1760, <4 x i32> %dma.buffer.rsrc1761) #10, !dbg !8
  %97 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local534), !dbg !8
  %98 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %99 = or disjoint i64 %30, 96, !dbg !8
  %100 = or disjoint i64 %address.local526, %99, !dbg !8
  %.tr1787 = trunc i64 %100 to i32, !dbg !8
  %dma.buffer.offset1757 = shl i32 %.tr1787, 1, !dbg !8
  %101 = insertelement <2 x i64> poison, i64 %98, i64 0, !dbg !8
  %102 = bitcast <2 x i64> %101 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1758 = shufflevector <4 x i32> %102, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1759 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %97), !dbg !8
  %103 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1759, i32 %dma.buffer.offset1757, <4 x i32> %dma.buffer.rsrc1758) #10, !dbg !8
  %104 = load <4 x half>, ptr addrspace(3) %address.local548, align 8, !dbg !11
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local548, i64 4096, !dbg !11
  %105 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !11
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local548, i64 1024, !dbg !11
  %106 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !11
  %107 = fmul <4 x half> %104, splat (half 0xH3015), !dbg !11
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local548, i64 5120, !dbg !11
  %108 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !11
  %109 = fmul <4 x half> %105, splat (half 0xH3015), !dbg !11
  %110 = fmul <4 x half> %106, splat (half 0xH3015), !dbg !11
  %111 = fmul <4 x half> %108, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local548, i64 8192, !dbg !11
  %112 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !11
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local548, i64 12288, !dbg !11
  %113 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !11
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local548, i64 9216, !dbg !11
  %114 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !11
  %115 = fmul <4 x half> %112, splat (half 0xH3015), !dbg !11
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local548, i64 13312, !dbg !11
  %116 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !11
  %117 = fmul <4 x half> %113, splat (half 0xH3015), !dbg !11
  %118 = fmul <4 x half> %114, splat (half 0xH3015), !dbg !11
  %119 = fmul <4 x half> %116, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %120 = or disjoint i64 %45, 127, !dbg !13
  %121 = add i64 %45, 128, !dbg !13
  %122 = icmp uge i64 %121, %120, !dbg !13
  %123 = sub nsw i64 32, %14, !dbg !15
  %124 = shl i64 %16, 7, !dbg !16
  %125 = and i64 %124, 3968, !dbg !16
  %126 = shl nsw i64 %44, 12, !dbg !16
  %127 = or disjoint i64 %125, %126, !dbg !16
  %128 = shl nuw nsw i64 %32, 2, !dbg !17
  %129 = shl nuw nsw i64 %32, 2, !dbg !17
  %130 = and i64 %129, 1008, !dbg !17
  %131 = sub nsw i64 %128, %130, !dbg !17
  %132 = and i32 %31, 1, !dbg !17
  %index.compact445 = zext nneg i32 %132 to i64, !dbg !17
  %133 = or disjoint i64 %131, %index.compact445, !dbg !17
  %134 = add nuw nsw i64 %133, 1, !dbg !17
  %135 = sdiv i64 %134, 2, !dbg !17
  br label %136, !dbg !7

136:                                              ; preds = %693, %4
  %137 = phi i64 [ %1422, %693 ], [ 0, %4 ], !dbg !7
  %138 = phi [2 x <8 x float>] [ %1407, %693 ], [ zeroinitializer, %4 ], !dbg !7
  %139 = phi [2 x <1 x float>] [ %1061, %693 ], [ zeroinitializer, %4 ], !dbg !7
  %140 = phi [2 x <1 x float>] [ %1409, %693 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %141 = phi [2 x <8 x float>] [ %1413, %693 ], [ zeroinitializer, %4 ], !dbg !7
  %142 = phi [2 x <8 x float>] [ %1417, %693 ], [ zeroinitializer, %4 ], !dbg !7
  %143 = phi [2 x <8 x float>] [ %1421, %693 ], [ zeroinitializer, %4 ], !dbg !7
  %144 = icmp slt i64 %137, %123, !dbg !7
  br i1 %144, label %145, label %1423, !dbg !7

145:                                              ; preds = %136
  %146 = shl nsw i64 %137, 7, !dbg !7
  %147 = lshr i64 %137, 12, !dbg !16
  %148 = add nuw nsw i64 %147, %16, !dbg !16
  %149 = lshr i64 %137, 5, !dbg !16
  %150 = and i64 %149, 127, !dbg !16
  %151 = shl i64 %148, 19, !dbg !16
  %152 = and i64 %151, 16252928, !dbg !16
  %153 = shl i64 %149, 19, !dbg !16
  %154 = shl i64 %137, 14, !dbg !16
  %155 = sub i64 %154, %153, !dbg !16
  %address.thread599 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local600 = zext nneg i32 %address.thread599 to i64, !dbg !8
  %address.local601 = shl nuw nsw i64 %address.local600, 3, !dbg !8
  %address.local602 = and i64 %address.local601, 1536, !dbg !8
  %address.local603 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local602, !dbg !8
  %address.local604 = ptrtoint ptr addrspace(3) %address.local603 to i64, !dbg !8
  %address.local605 = trunc i64 %address.local604 to i32, !dbg !8
  %156 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local605), !dbg !16
  %address.local606 = lshr i64 %address.local600, 4, !dbg !16
  %address.local607 = and i64 %address.local606, 8, !dbg !16
  %address.local608 = and i64 %address.local600, 123, !dbg !16
  %address.local609 = lshr i64 %address.local600, 2, !dbg !16
  %address.local610 = and i64 %address.local609, 4, !dbg !16
  %address.local611 = add nuw nsw i64 %address.local610, %address.local600, !dbg !16
  %address.local612 = and i64 %address.local611, 4, !dbg !16
  %address.local613 = or disjoint i64 %address.local608, %address.local612, !dbg !16
  %address.local614 = shl nuw nsw i64 %address.local613, 7, !dbg !16
  %address.local615 = or disjoint i64 %address.local607, %address.local614, !dbg !16
  %157 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx1790 = or disjoint i64 %152, %150, !dbg !16
  %158 = add i64 %.idx1790, %155, !dbg !16
  %159 = add i64 %158, %address.local615, !dbg !16
  %.tr1791 = trunc i64 %159 to i32, !dbg !16
  %dma.buffer.offset1754 = shl i32 %.tr1791, 1, !dbg !16
  %160 = insertelement <2 x i64> poison, i64 %157, i64 0, !dbg !16
  %161 = bitcast <2 x i64> %160 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1755 = shufflevector <4 x i32> %161, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %dma.buffer.lds1756 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !16
  %162 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1756, i32 %dma.buffer.offset1754, <4 x i32> %dma.buffer.rsrc1755) #10, !dbg !16
  %address.local617 = or disjoint i64 %address.local600, 256, !dbg !8
  %address.local618 = shl nuw nsw i64 %address.local617, 3, !dbg !8
  %address.local619 = and i64 %address.local618, 3584, !dbg !8
  %address.local620 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local619, !dbg !8
  %address.local621 = ptrtoint ptr addrspace(3) %address.local620 to i64, !dbg !8
  %address.local622 = trunc i64 %address.local621 to i32, !dbg !8
  %163 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local622), !dbg !16
  %address.local623 = lshr i64 %address.local617, 4, !dbg !16
  %address.local624 = and i64 %address.local623, 24, !dbg !16
  %address.local625 = and i64 %address.local600, 123, !dbg !16
  %address.local626 = lshr i64 %address.local600, 2, !dbg !16
  %address.local627 = and i64 %address.local626, 4, !dbg !16
  %address.local628 = add nuw nsw i64 %address.local627, %address.local600, !dbg !16
  %address.local629 = and i64 %address.local628, 4, !dbg !16
  %address.local630 = or disjoint i64 %address.local625, %address.local629, !dbg !16
  %address.local631 = shl nuw nsw i64 %address.local630, 7, !dbg !16
  %address.local632 = or disjoint i64 %address.local624, %address.local631, !dbg !16
  %164 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx1794 = or disjoint i64 %152, %150, !dbg !16
  %165 = add i64 %.idx1794, %155, !dbg !16
  %166 = add i64 %165, %address.local632, !dbg !16
  %.tr1795 = trunc i64 %166 to i32, !dbg !16
  %dma.buffer.offset1751 = shl i32 %.tr1795, 1, !dbg !16
  %167 = insertelement <2 x i64> poison, i64 %164, i64 0, !dbg !16
  %168 = bitcast <2 x i64> %167 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1752 = shufflevector <4 x i32> %168, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %dma.buffer.lds1753 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %163), !dbg !16
  %169 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1753, i32 %dma.buffer.offset1751, <4 x i32> %dma.buffer.rsrc1752) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %170 = add nuw i64 %137, 1024, !dbg !16
  %171 = icmp slt i64 %170, 0, !dbg !16
  %172 = sub nuw i64 -1025, %137, !dbg !16
  %173 = select i1 %171, i64 %172, i64 %170, !dbg !16
  %174 = sdiv i64 %173, 4096, !dbg !16
  %.lobit37 = ashr i64 %170, 63, !dbg !16
  %175 = xor i64 %174, %.lobit37, !dbg !16
  %176 = add nsw i64 %175, %16, !dbg !16
  %177 = add nsw i64 %149, %127, !dbg !16
  %178 = add nsw i64 %177, 32, !dbg !16
  %179 = icmp slt i64 %177, -32, !dbg !16
  %180 = sub nsw i64 -33, %177, !dbg !16
  %181 = select i1 %179, i64 %180, i64 %178, !dbg !16
  %182 = sdiv i64 %181, 128, !dbg !16
  %183 = sext i1 %179 to i64, !dbg !16
  %184 = xor i64 %182, %183, !dbg !16
  %185 = shl nsw i64 %184, 7, !dbg !16
  %186 = sub i64 %127, %185, !dbg !16
  %187 = shl i64 %176, 19, !dbg !16
  %188 = and i64 %187, 16252928, !dbg !16
  %address.local634 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local602, !dbg !8
  %address.local635 = ptrtoint ptr addrspace(3) %address.local634 to i64, !dbg !8
  %address.local636 = trunc i64 %address.local635 to i32, !dbg !8
  %189 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local636), !dbg !16
  %190 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx1799 = add nuw nsw i64 %188, %149, !dbg !16
  %191 = add i64 %.idx1799, %186, !dbg !16
  %192 = add i64 %191, %155, !dbg !16
  %193 = add i64 %192, %address.local615, !dbg !16
  %.tr1800 = trunc i64 %193 to i32, !dbg !16
  %194 = shl i32 %.tr1800, 1, !dbg !16
  %dma.buffer.offset1748 = add i32 %194, 64, !dbg !16
  %195 = insertelement <2 x i64> poison, i64 %190, i64 0, !dbg !16
  %196 = bitcast <2 x i64> %195 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1749 = shufflevector <4 x i32> %196, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %dma.buffer.lds1750 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %189), !dbg !16
  %197 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1750, i32 %dma.buffer.offset1748, <4 x i32> %dma.buffer.rsrc1749) #10, !dbg !16
  %address.local638 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local619, !dbg !8
  %address.local639 = ptrtoint ptr addrspace(3) %address.local638 to i64, !dbg !8
  %address.local640 = trunc i64 %address.local639 to i32, !dbg !8
  %198 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local640), !dbg !16
  %199 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx1804 = add nuw nsw i64 %188, %149, !dbg !16
  %200 = add i64 %.idx1804, %186, !dbg !16
  %201 = add i64 %200, %155, !dbg !16
  %202 = add i64 %201, %address.local632, !dbg !16
  %.tr1805 = trunc i64 %202 to i32, !dbg !16
  %203 = shl i32 %.tr1805, 1, !dbg !16
  %dma.buffer.offset1745 = add i32 %203, 64, !dbg !16
  %204 = insertelement <2 x i64> poison, i64 %199, i64 0, !dbg !16
  %205 = bitcast <2 x i64> %204 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1746 = shufflevector <4 x i32> %205, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %dma.buffer.lds1747 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %198), !dbg !16
  %206 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1747, i32 %dma.buffer.offset1745, <4 x i32> %dma.buffer.rsrc1746) #10, !dbg !16
  %address.local643 = lshr i64 %address.local600, 2, !dbg !17
  %address.local646 = shl nuw nsw i64 %address.local600, 1, !dbg !17
  %address.local647 = and i64 %address.local646, 24, !dbg !17
  %address.local649 = shl i64 %address.local600, 62, !dbg !17
  %address.local650 = ashr exact i64 %address.local649, 57, !dbg !17
  %address.local651 = and i64 %address.local650, -64, !dbg !17
  %address.local653 = lshr i64 %address.local600, 2, !dbg !11
  %address.local654 = and i64 %address.local653, 12, !dbg !11
  %207 = and i64 %address.local600, 32, !dbg !11
  %.not1806 = icmp eq i64 %207, 0, !dbg !11
  %address.local658 = select i1 %.not1806, i64 0, i64 1016, !dbg !11
  %.idx = mul nuw nsw i64 %address.local600, 320, !dbg !17
  %208 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !17
  %.idx1807 = mul nsw i64 %address.local643, -1280, !dbg !17
  %209 = getelementptr i8, ptr addrspace(3) %208, i64 %.idx1807, !dbg !17
  %210 = getelementptr [2 x i8], ptr addrspace(3) %209, i64 %address.local647, !dbg !17
  %211 = getelementptr [2 x i8], ptr addrspace(3) %210, i64 %address.local651, !dbg !17
  %212 = getelementptr [2 x i8], ptr addrspace(3) %211, i64 %address.local654, !dbg !17
  %address.local660 = getelementptr [2 x i8], ptr addrspace(3) %212, i64 %address.local658, !dbg !17
  %213 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %213, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %215 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !17
  %address.local671 = shl nuw nsw i32 %address.thread599, 1, !dbg !17
  %address.local672 = and i32 %address.local671, 24, !dbg !17
  %address.local673 = shl nuw nsw i32 %address.thread599, 8, !dbg !17
  %address.local674 = and i32 %address.local673, 768, !dbg !17
  %address.local675 = or disjoint i32 %address.local672, %address.local674, !dbg !17
  %address.local676 = shl nuw nsw i32 %address.thread599, 5, !dbg !17
  %address.local677 = and i32 %address.local676, 96, !dbg !17
  %address.local678 = or disjoint i32 %address.local675, %address.local677, !dbg !17
  %address.local679 = zext nneg i32 %address.local678 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %135) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local680 = lshr i32 %address.thread599, 1, !dbg !17
  %address.local681 = and i32 %address.local680, 1, !dbg !17
  %address.local682 = zext nneg i32 %address.local681 to i64, !dbg !17
  %address.local683 = add nsw i64 %address.base, %address.local682, !dbg !17
  %address.local684 = shl i64 %address.local683, 6, !dbg !17
  %address.local685 = sub i64 %address.local679, %address.local684, !dbg !17
  %216 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local685, !dbg !17
  %217 = getelementptr [2 x i8], ptr addrspace(3) %216, i64 %address.local654, !dbg !17
  %218 = getelementptr [2 x i8], ptr addrspace(3) %217, i64 %address.local658, !dbg !17
  %address.local689 = getelementptr i8, ptr addrspace(3) %218, i64 64, !dbg !17
  %219 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %215, <4 x float> %214) #9, !dbg !17
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %219, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %218, i64 4160, !dbg !17
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !17
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !17
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %222, <4 x float> %221) #9, !dbg !17
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %223, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !17
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %218, i64 192, !dbg !17
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !17
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %226, <4 x float> %225) #9, !dbg !17
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %227, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %218, i64 4288, !dbg !17
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !17
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !17
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %230, <4 x float> %229) #9, !dbg !17
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %231, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !17
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %218, i64 1088, !dbg !17
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !17
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %234, <4 x float> %233) #9, !dbg !17
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %235, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %218, i64 5184, !dbg !17
  %238 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !17
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !17
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %238, <4 x float> %237) #9, !dbg !17
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %239, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %242 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !17
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %218, i64 1216, !dbg !17
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !17
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %242, <4 x float> %241) #9, !dbg !17
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %243, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %218, i64 5312, !dbg !17
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !17
  %247 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %246, <4 x float> %245) #9, !dbg !17
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %247, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !17
  %251 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %250, <4 x float> %249) #9, !dbg !17
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %251, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %218, i64 4160, !dbg !17
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !17
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !17
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %254, <4 x float> %253) #9, !dbg !17
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %255, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !17
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %218, i64 192, !dbg !17
  %259 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !17
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %258, <4 x float> %257) #9, !dbg !17
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %259, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %218, i64 4288, !dbg !17
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !17
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %263 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !17
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %262, <4 x float> %261) #9, !dbg !17
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %263, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !17
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %218, i64 1088, !dbg !17
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !17
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %266, <4 x float> %265) #9, !dbg !17
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %267, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %218, i64 5184, !dbg !17
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !17
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !17
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %270, <4 x float> %269) #9, !dbg !17
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %271, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !17
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %218, i64 1216, !dbg !17
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !17
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %274, <4 x float> %273) #9, !dbg !17
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %275, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %218, i64 5312, !dbg !17
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !17
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %278, <4 x float> %277) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %280 = add nuw i64 %137, 2048, !dbg !16
  %281 = icmp slt i64 %280, 0, !dbg !16
  %282 = sub nuw i64 -2049, %137, !dbg !16
  %283 = select i1 %281, i64 %282, i64 %280, !dbg !16
  %284 = sdiv i64 %283, 4096, !dbg !16
  %.lobit38 = ashr i64 %280, 63, !dbg !16
  %285 = xor i64 %284, %.lobit38, !dbg !16
  %286 = add nsw i64 %285, %16, !dbg !16
  %287 = add nsw i64 %177, 64, !dbg !16
  %288 = icmp slt i64 %177, -64, !dbg !16
  %289 = sub nsw i64 -65, %177, !dbg !16
  %290 = select i1 %288, i64 %289, i64 %287, !dbg !16
  %291 = sdiv i64 %290, 128, !dbg !16
  %292 = sext i1 %288 to i64, !dbg !16
  %293 = xor i64 %291, %292, !dbg !16
  %294 = shl nsw i64 %293, 7, !dbg !16
  %295 = sub i64 %127, %294, !dbg !16
  %296 = shl i64 %286, 19, !dbg !16
  %297 = and i64 %296, 16252928, !dbg !16
  %298 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local605), !dbg !16
  %299 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx18081812 = add nuw nsw i64 %297, %149, !dbg !16
  %300 = add i64 %.idx18081812, %295, !dbg !16
  %301 = add i64 %300, %155, !dbg !16
  %302 = add i64 %301, %address.local615, !dbg !16
  %.tr1813 = trunc i64 %302 to i32, !dbg !16
  %303 = shl i32 %.tr1813, 1, !dbg !16
  %dma.buffer.offset1742 = add i32 %303, 128, !dbg !16
  %304 = insertelement <2 x i64> poison, i64 %299, i64 0, !dbg !16
  %305 = bitcast <2 x i64> %304 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1743 = shufflevector <4 x i32> %305, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %dma.buffer.lds1744 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %298), !dbg !16
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1744, i32 %dma.buffer.offset1742, <4 x i32> %dma.buffer.rsrc1743) #10, !dbg !16
  %307 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local622), !dbg !16
  %308 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx18141818 = add nuw nsw i64 %297, %149, !dbg !16
  %309 = add i64 %.idx18141818, %295, !dbg !16
  %310 = add i64 %309, %155, !dbg !16
  %311 = add i64 %310, %address.local632, !dbg !16
  %.tr1819 = trunc i64 %311 to i32, !dbg !16
  %312 = shl i32 %.tr1819, 1, !dbg !16
  %dma.buffer.offset1739 = add i32 %312, 128, !dbg !16
  %313 = insertelement <2 x i64> poison, i64 %308, i64 0, !dbg !16
  %314 = bitcast <2 x i64> %313 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1740 = shufflevector <4 x i32> %314, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %dma.buffer.lds1741 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %307), !dbg !16
  %315 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1741, i32 %dma.buffer.offset1739, <4 x i32> %dma.buffer.rsrc1740) #10, !dbg !16
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %316 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !17
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %316, <4 x float> %220) #9, !dbg !17
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !17
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %218, i64 8256, !dbg !17
  %319 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %318, <4 x float> %317) #9, !dbg !17
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %319, <4 x float> %224) #9, !dbg !17
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %218, i64 12352, !dbg !17
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !17
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !17
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %322, <4 x float> %321) #9, !dbg !17
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %323, <4 x float> %228) #9, !dbg !17
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !17
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %218, i64 8384, !dbg !17
  %327 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !17
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %326, <4 x float> %325) #9, !dbg !17
  %329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %327, <4 x float> %232) #9, !dbg !17
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %218, i64 12480, !dbg !17
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !17
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !17
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %330, <4 x float> %329) #9, !dbg !17
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %331, <4 x float> %236) #9, !dbg !17
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !17
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %218, i64 9280, !dbg !17
  %335 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !17
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %334, <4 x float> %333) #9, !dbg !17
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %335, <4 x float> %240) #9, !dbg !17
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %218, i64 13376, !dbg !17
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !17
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %339 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !17
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %338, <4 x float> %337) #9, !dbg !17
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %339, <4 x float> %244) #9, !dbg !17
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !17
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %218, i64 9408, !dbg !17
  %343 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !17
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %342, <4 x float> %341) #9, !dbg !17
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %343, <4 x float> %248) #9, !dbg !17
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %218, i64 13504, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !17
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %346, <4 x float> %345) #9, !dbg !17
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %347, <4 x float> %252) #9, !dbg !17
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !17
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %218, i64 8256, !dbg !17
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !17
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %350, <4 x float> %349) #9, !dbg !17
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %351, <4 x float> %256) #9, !dbg !17
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %218, i64 12352, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !17
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %354, <4 x float> %353) #9, !dbg !17
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %355, <4 x float> %260) #9, !dbg !17
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !17
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %218, i64 8384, !dbg !17
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !17
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %358, <4 x float> %357) #9, !dbg !17
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %359, <4 x float> %264) #9, !dbg !17
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %218, i64 12480, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !17
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !17
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %362, <4 x float> %361) #9, !dbg !17
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %363, <4 x float> %268) #9, !dbg !17
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !17
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %218, i64 9280, !dbg !17
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !17
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %366, <4 x float> %365) #9, !dbg !17
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %367, <4 x float> %272) #9, !dbg !17
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %218, i64 13376, !dbg !17
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !17
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !17
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %370, <4 x float> %369) #9, !dbg !17
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %371, <4 x float> %276) #9, !dbg !17
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !17
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %218, i64 9408, !dbg !17
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !17
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %374, <4 x float> %373) #9, !dbg !17
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %375, <4 x float> %279) #9, !dbg !17
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %218, i64 13504, !dbg !17
  %378 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !17
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %378, <4 x float> %377) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %380 = add nuw i64 %137, 3072, !dbg !16
  %381 = icmp slt i64 %380, 0, !dbg !16
  %382 = sub nuw i64 -3073, %137, !dbg !16
  %383 = select i1 %381, i64 %382, i64 %380, !dbg !16
  %384 = sdiv i64 %383, 4096, !dbg !16
  %.lobit39 = ashr i64 %380, 63, !dbg !16
  %385 = xor i64 %384, %.lobit39, !dbg !16
  %386 = add nsw i64 %385, %16, !dbg !16
  %387 = add nsw i64 %177, 96, !dbg !16
  %388 = icmp slt i64 %177, -96, !dbg !16
  %389 = sub nsw i64 -97, %177, !dbg !16
  %390 = select i1 %388, i64 %389, i64 %387, !dbg !16
  %391 = sdiv i64 %390, 128, !dbg !16
  %392 = sext i1 %388 to i64, !dbg !16
  %393 = xor i64 %391, %392, !dbg !16
  %394 = shl nsw i64 %393, 7, !dbg !16
  %395 = sub i64 %127, %394, !dbg !16
  %396 = shl i64 %386, 19, !dbg !16
  %397 = and i64 %396, 16252928, !dbg !16
  %398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local636), !dbg !16
  %399 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx18201824 = add nuw nsw i64 %397, %149, !dbg !16
  %400 = add i64 %.idx18201824, %395, !dbg !16
  %401 = add i64 %400, %155, !dbg !16
  %402 = add i64 %401, %address.local615, !dbg !16
  %.tr1825 = trunc i64 %402 to i32, !dbg !16
  %403 = shl i32 %.tr1825, 1, !dbg !16
  %dma.buffer.offset1736 = add i32 %403, 192, !dbg !16
  %404 = insertelement <2 x i64> poison, i64 %399, i64 0, !dbg !16
  %405 = bitcast <2 x i64> %404 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1737 = shufflevector <4 x i32> %405, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %dma.buffer.lds1738 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %398), !dbg !16
  %406 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1738, i32 %dma.buffer.offset1736, <4 x i32> %dma.buffer.rsrc1737) #10, !dbg !16
  %407 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local640), !dbg !16
  %408 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx18261830 = add nuw nsw i64 %397, %149, !dbg !16
  %409 = add i64 %.idx18261830, %395, !dbg !16
  %410 = add i64 %409, %155, !dbg !16
  %411 = add i64 %410, %address.local632, !dbg !16
  %.tr1831 = trunc i64 %411 to i32, !dbg !16
  %412 = shl i32 %.tr1831, 1, !dbg !16
  %dma.buffer.offset1733 = add i32 %412, 192, !dbg !16
  %413 = insertelement <2 x i64> poison, i64 %408, i64 0, !dbg !16
  %414 = bitcast <2 x i64> %413 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1734 = shufflevector <4 x i32> %414, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %dma.buffer.lds1735 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %407), !dbg !16
  %415 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1735, i32 %dma.buffer.offset1733, <4 x i32> %dma.buffer.rsrc1734) #10, !dbg !16
  %416 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %417 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %416, <4 x float> %320) #9, !dbg !17
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %418 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !17
  %419 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %418, <4 x float> %417) #9, !dbg !17
  %421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %419, <4 x float> %324) #9, !dbg !17
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %218, i64 4160, !dbg !17
  %422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !17
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %423 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !17
  %424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %422, <4 x float> %421) #9, !dbg !17
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %423, <4 x float> %328) #9, !dbg !17
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %426 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !17
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %218, i64 192, !dbg !17
  %427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !17
  %428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %426, <4 x float> %425) #9, !dbg !17
  %429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %427, <4 x float> %332) #9, !dbg !17
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %218, i64 4288, !dbg !17
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !17
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %431 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !17
  %432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %430, <4 x float> %429) #9, !dbg !17
  %433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %431, <4 x float> %336) #9, !dbg !17
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %434 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !17
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %218, i64 1088, !dbg !17
  %435 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !17
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %434, <4 x float> %433) #9, !dbg !17
  %437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %435, <4 x float> %340) #9, !dbg !17
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %218, i64 5184, !dbg !17
  %438 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !17
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %439 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !17
  %440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %438, <4 x float> %437) #9, !dbg !17
  %441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %439, <4 x float> %344) #9, !dbg !17
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %442 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !17
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %218, i64 1216, !dbg !17
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !17
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %442, <4 x float> %441) #9, !dbg !17
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %443, <4 x float> %348) #9, !dbg !17
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %218, i64 5312, !dbg !17
  %446 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !17
  %447 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %446, <4 x float> %445) #9, !dbg !17
  %449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %447, <4 x float> %352) #9, !dbg !17
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %450 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !17
  %451 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %450, <4 x float> %449) #9, !dbg !17
  %453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %451, <4 x float> %356) #9, !dbg !17
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %218, i64 4160, !dbg !17
  %454 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !17
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %455 = load <4 x half>, ptr addrspace(3) %lds.const.gep1668, align 8, !dbg !17
  %456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %454, <4 x float> %453) #9, !dbg !17
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %455, <4 x float> %360) #9, !dbg !17
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %458 = load <4 x half>, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !17
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %218, i64 192, !dbg !17
  %459 = load <4 x half>, ptr addrspace(3) %lds.const.gep1670, align 8, !dbg !17
  %460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %458, <4 x float> %457) #9, !dbg !17
  %461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %459, <4 x float> %364) #9, !dbg !17
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %218, i64 4288, !dbg !17
  %462 = load <4 x half>, ptr addrspace(3) %lds.const.gep1671, align 8, !dbg !17
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %463 = load <4 x half>, ptr addrspace(3) %lds.const.gep1672, align 8, !dbg !17
  %464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %462, <4 x float> %461) #9, !dbg !17
  %465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %463, <4 x float> %368) #9, !dbg !17
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !17
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %218, i64 1088, !dbg !17
  %467 = load <4 x half>, ptr addrspace(3) %lds.const.gep1674, align 8, !dbg !17
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %466, <4 x float> %465) #9, !dbg !17
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %467, <4 x float> %372) #9, !dbg !17
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %218, i64 5184, !dbg !17
  %470 = load <4 x half>, ptr addrspace(3) %lds.const.gep1675, align 8, !dbg !17
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %471 = load <4 x half>, ptr addrspace(3) %lds.const.gep1676, align 8, !dbg !17
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %470, <4 x float> %469) #9, !dbg !17
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %471, <4 x float> %376) #9, !dbg !17
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %474 = load <4 x half>, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !17
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %218, i64 1216, !dbg !17
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep1678, align 8, !dbg !17
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %474, <4 x float> %473) #9, !dbg !17
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %475, <4 x float> %379) #9, !dbg !17
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %218, i64 5312, !dbg !17
  %478 = load <4 x half>, ptr addrspace(3) %lds.const.gep1679, align 8, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %478, <4 x float> %477) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %480 = add nuw nsw i64 %149, %16, !dbg !18
  %481 = shl i64 %480, 19, !dbg !18
  %482 = and i64 %481, 16252928, !dbg !18
  %483 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local605), !dbg !18
  %address.local1018 = shl nuw nsw i64 %address.local600, 6, !dbg !18
  %address.local1019 = and i64 %address.local1018, 16128, !dbg !18
  %address.local1020 = shl nuw nsw i64 %address.local600, 4, !dbg !18
  %address.local1021 = and i64 %address.local1020, 48, !dbg !18
  %484 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %.idx18321835 = add i64 %482, %155, !dbg !18
  %485 = shl i64 %.idx18321835, 1, !dbg !18
  %486 = or disjoint i64 %address.local1019, %address.local1021, !dbg !18
  %487 = or disjoint i64 %485, %486, !dbg !18
  %dma.buffer.offset1730 = trunc i64 %487 to i32, !dbg !18
  %488 = insertelement <2 x i64> poison, i64 %484, i64 0, !dbg !18
  %489 = bitcast <2 x i64> %488 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1731 = shufflevector <4 x i32> %489, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1732 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %483), !dbg !18
  %490 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1732, i32 %dma.buffer.offset1730, <4 x i32> %dma.buffer.rsrc1731) #10, !dbg !18
  %491 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local622), !dbg !18
  %address.local1024 = shl nuw nsw i64 %address.local617, 6, !dbg !18
  %address.local1025 = and i64 %address.local1024, 32512, !dbg !18
  %address.local1026 = shl nuw nsw i64 %address.local600, 4, !dbg !18
  %address.local1027 = and i64 %address.local1026, 48, !dbg !18
  %492 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %.idx18361839 = add i64 %482, %155, !dbg !18
  %493 = shl i64 %.idx18361839, 1, !dbg !18
  %494 = or disjoint i64 %493, %address.local1025, !dbg !18
  %495 = or disjoint i64 %494, %address.local1027, !dbg !18
  %dma.buffer.offset1727 = trunc i64 %495 to i32, !dbg !18
  %496 = insertelement <2 x i64> poison, i64 %492, i64 0, !dbg !18
  %497 = bitcast <2 x i64> %496 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1728 = shufflevector <4 x i32> %497, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1729 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %491), !dbg !18
  %498 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1729, i32 %dma.buffer.offset1727, <4 x i32> %dma.buffer.rsrc1728) #10, !dbg !18
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep1680, align 8, !dbg !17
  %500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %499, <4 x float> %420) #9, !dbg !17
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %501 = load <4 x half>, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !17
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %218, i64 8256, !dbg !17
  %502 = load <4 x half>, ptr addrspace(3) %lds.const.gep1682, align 8, !dbg !17
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %501, <4 x float> %500) #9, !dbg !17
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %502, <4 x float> %424) #9, !dbg !17
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %218, i64 12352, !dbg !17
  %505 = load <4 x half>, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !17
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %506 = load <4 x half>, ptr addrspace(3) %lds.const.gep1684, align 8, !dbg !17
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %505, <4 x float> %504) #9, !dbg !17
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %506, <4 x float> %428) #9, !dbg !17
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %509 = load <4 x half>, ptr addrspace(3) %lds.const.gep1685, align 8, !dbg !17
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %218, i64 8384, !dbg !17
  %510 = load <4 x half>, ptr addrspace(3) %lds.const.gep1686, align 8, !dbg !17
  %511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %509, <4 x float> %508) #9, !dbg !17
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %510, <4 x float> %432) #9, !dbg !17
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %218, i64 12480, !dbg !17
  %513 = load <4 x half>, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !17
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %514 = load <4 x half>, ptr addrspace(3) %lds.const.gep1688, align 8, !dbg !17
  %515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %513, <4 x float> %512) #9, !dbg !17
  %516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %514, <4 x float> %436) #9, !dbg !17
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %517 = load <4 x half>, ptr addrspace(3) %lds.const.gep1689, align 8, !dbg !17
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %218, i64 9280, !dbg !17
  %518 = load <4 x half>, ptr addrspace(3) %lds.const.gep1690, align 8, !dbg !17
  %519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %517, <4 x float> %516) #9, !dbg !17
  %520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %518, <4 x float> %440) #9, !dbg !17
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %218, i64 13376, !dbg !17
  %521 = load <4 x half>, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !17
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %522 = load <4 x half>, ptr addrspace(3) %lds.const.gep1692, align 8, !dbg !17
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %521, <4 x float> %520) #9, !dbg !17
  %524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %522, <4 x float> %444) #9, !dbg !17
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %525 = load <4 x half>, ptr addrspace(3) %lds.const.gep1693, align 8, !dbg !17
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %218, i64 9408, !dbg !17
  %526 = load <4 x half>, ptr addrspace(3) %lds.const.gep1694, align 8, !dbg !17
  %527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %525, <4 x float> %524) #9, !dbg !17
  %528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %526, <4 x float> %448) #9, !dbg !17
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %218, i64 13504, !dbg !17
  %529 = load <4 x half>, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !17
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %530 = load <4 x half>, ptr addrspace(3) %lds.const.gep1696, align 8, !dbg !17
  %531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %529, <4 x float> %528) #9, !dbg !17
  %532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %530, <4 x float> %452) #9, !dbg !17
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %533 = load <4 x half>, ptr addrspace(3) %lds.const.gep1697, align 8, !dbg !17
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %218, i64 8256, !dbg !17
  %534 = load <4 x half>, ptr addrspace(3) %lds.const.gep1698, align 8, !dbg !17
  %535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %533, <4 x float> %532) #9, !dbg !17
  %536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %534, <4 x float> %456) #9, !dbg !17
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %218, i64 12352, !dbg !17
  %537 = load <4 x half>, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !17
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %538 = load <4 x half>, ptr addrspace(3) %lds.const.gep1700, align 8, !dbg !17
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %537, <4 x float> %536) #9, !dbg !17
  %540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %538, <4 x float> %460) #9, !dbg !17
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %541 = load <4 x half>, ptr addrspace(3) %lds.const.gep1701, align 8, !dbg !17
  %lds.const.gep1702 = getelementptr i8, ptr addrspace(3) %218, i64 8384, !dbg !17
  %542 = load <4 x half>, ptr addrspace(3) %lds.const.gep1702, align 8, !dbg !17
  %543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %541, <4 x float> %540) #9, !dbg !17
  %544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %542, <4 x float> %464) #9, !dbg !17
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %218, i64 12480, !dbg !17
  %545 = load <4 x half>, ptr addrspace(3) %lds.const.gep1703, align 8, !dbg !17
  %lds.const.gep1704 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %546 = load <4 x half>, ptr addrspace(3) %lds.const.gep1704, align 8, !dbg !17
  %547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %545, <4 x float> %544) #9, !dbg !17
  %548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %546, <4 x float> %468) #9, !dbg !17
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %549 = load <4 x half>, ptr addrspace(3) %lds.const.gep1705, align 8, !dbg !17
  %lds.const.gep1706 = getelementptr i8, ptr addrspace(3) %218, i64 9280, !dbg !17
  %550 = load <4 x half>, ptr addrspace(3) %lds.const.gep1706, align 8, !dbg !17
  %551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %549, <4 x float> %548) #9, !dbg !17
  %552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %550, <4 x float> %472) #9, !dbg !17
  %lds.const.gep1707 = getelementptr i8, ptr addrspace(3) %218, i64 13376, !dbg !17
  %553 = load <4 x half>, ptr addrspace(3) %lds.const.gep1707, align 8, !dbg !17
  %lds.const.gep1708 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %554 = load <4 x half>, ptr addrspace(3) %lds.const.gep1708, align 8, !dbg !17
  %555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %553, <4 x float> %552) #9, !dbg !17
  %556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %554, <4 x float> %476) #9, !dbg !17
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %557 = load <4 x half>, ptr addrspace(3) %lds.const.gep1709, align 8, !dbg !17
  %lds.const.gep1710 = getelementptr i8, ptr addrspace(3) %218, i64 9408, !dbg !17
  %558 = load <4 x half>, ptr addrspace(3) %lds.const.gep1710, align 8, !dbg !17
  %559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %557, <4 x float> %556) #9, !dbg !17
  %560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %558, <4 x float> %479) #9, !dbg !17
  %lds.const.gep1711 = getelementptr i8, ptr addrspace(3) %218, i64 13504, !dbg !17
  %561 = load <4 x half>, ptr addrspace(3) %lds.const.gep1711, align 8, !dbg !17
  %562 = or disjoint i64 %146, 127, !dbg !13
  %563 = icmp uge i64 %45, %562, !dbg !13
  %564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %561, <4 x float> %560) #9, !dbg !17
  %565 = and i1 %122, %563, !dbg !13
  br i1 %565, label %693, label %566, !dbg !13

566:                                              ; preds = %145
  %567 = add i64 %45, %32, !dbg !13
  %568 = lshr i64 %32, 2, !dbg !13
  %569 = and i64 %568, 48, !dbg !13
  %570 = add i64 %567, %569, !dbg !13
  %571 = and i32 %31, 240, !dbg !13
  %index.compact = zext nneg i32 %571 to i64, !dbg !13
  %572 = sub i64 %570, %index.compact, !dbg !13
  %573 = and i64 %32, 48, !dbg !13
  %574 = or disjoint i64 %146, %573, !dbg !13
  %575 = add i64 %572, 1, !dbg !23
  %576 = lshr i64 %575, 32, !dbg !24
  %compare.high = trunc nuw i64 %576 to i32, !dbg !24
  %577 = lshr i64 %146, 32, !dbg !24
  %compare.high1323 = trunc nuw i64 %577 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1323, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1323, !dbg !24
  %compare.low.base = trunc i64 %575 to i32, !dbg !24
  %compare.low.base1324 = trunc i64 %574 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1324, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %578 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1325 = icmp ult i32 %compare.local, %compare.low.base, !dbg !24
  %compare.result1326 = select i1 %compare.high.equal, i1 %compare.low1325, i1 %compare.high.order, !dbg !24
  %579 = select i1 %compare.result1326, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1327 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1328 = or i32 %compare.local.base1327, 2, !dbg !24
  %compare.low1329 = icmp ult i32 %compare.local1328, %compare.low.base, !dbg !24
  %compare.result1330 = select i1 %compare.high.equal, i1 %compare.low1329, i1 %compare.high.order, !dbg !24
  %580 = select i1 %compare.result1330, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1332 = or i32 %compare.local.base1331, 3, !dbg !24
  %compare.low1333 = icmp ult i32 %compare.local1332, %compare.low.base, !dbg !24
  %compare.result1334 = select i1 %compare.high.equal, i1 %compare.low1333, i1 %compare.high.order, !dbg !24
  %581 = select i1 %compare.result1334, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1336 = or i32 %compare.local.base1335, 4, !dbg !24
  %compare.low1337 = icmp ult i32 %compare.local1336, %compare.low.base, !dbg !24
  %compare.result1338 = select i1 %compare.high.equal, i1 %compare.low1337, i1 %compare.high.order, !dbg !24
  %582 = select i1 %compare.result1338, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1340 = or i32 %compare.local.base1339, 5, !dbg !24
  %compare.low1341 = icmp ult i32 %compare.local1340, %compare.low.base, !dbg !24
  %compare.result1342 = select i1 %compare.high.equal, i1 %compare.low1341, i1 %compare.high.order, !dbg !24
  %583 = select i1 %compare.result1342, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1344 = or i32 %compare.local.base1343, 6, !dbg !24
  %compare.low1345 = icmp ult i32 %compare.local1344, %compare.low.base, !dbg !24
  %compare.result1346 = select i1 %compare.high.equal, i1 %compare.low1345, i1 %compare.high.order, !dbg !24
  %584 = select i1 %compare.result1346, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1348 = or i32 %compare.local.base1347, 7, !dbg !24
  %compare.low1349 = icmp ult i32 %compare.local1348, %compare.low.base, !dbg !24
  %compare.result1350 = select i1 %compare.high.equal, i1 %compare.low1349, i1 %compare.high.order, !dbg !24
  %585 = select i1 %compare.result1350, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1352 = or i32 %compare.local.base1351, 8, !dbg !24
  %compare.low1353 = icmp ult i32 %compare.local1352, %compare.low.base, !dbg !24
  %compare.result1354 = select i1 %compare.high.equal, i1 %compare.low1353, i1 %compare.high.order, !dbg !24
  %586 = select i1 %compare.result1354, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1356 = or i32 %compare.local.base1355, 9, !dbg !24
  %compare.low1357 = icmp ult i32 %compare.local1356, %compare.low.base, !dbg !24
  %compare.result1358 = select i1 %compare.high.equal, i1 %compare.low1357, i1 %compare.high.order, !dbg !24
  %587 = select i1 %compare.result1358, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1360 = or i32 %compare.local.base1359, 10, !dbg !24
  %compare.low1361 = icmp ult i32 %compare.local1360, %compare.low.base, !dbg !24
  %compare.result1362 = select i1 %compare.high.equal, i1 %compare.low1361, i1 %compare.high.order, !dbg !24
  %588 = select i1 %compare.result1362, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1364 = or i32 %compare.local.base1363, 11, !dbg !24
  %compare.low1365 = icmp ult i32 %compare.local1364, %compare.low.base, !dbg !24
  %compare.result1366 = select i1 %compare.high.equal, i1 %compare.low1365, i1 %compare.high.order, !dbg !24
  %589 = select i1 %compare.result1366, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1368 = or i32 %compare.local.base1367, 12, !dbg !24
  %compare.low1369 = icmp ult i32 %compare.local1368, %compare.low.base, !dbg !24
  %compare.result1370 = select i1 %compare.high.equal, i1 %compare.low1369, i1 %compare.high.order, !dbg !24
  %590 = select i1 %compare.result1370, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1372 = or i32 %compare.local.base1371, 13, !dbg !24
  %compare.low1373 = icmp ult i32 %compare.local1372, %compare.low.base, !dbg !24
  %compare.result1374 = select i1 %compare.high.equal, i1 %compare.low1373, i1 %compare.high.order, !dbg !24
  %591 = select i1 %compare.result1374, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1376 = or i32 %compare.local.base1375, 14, !dbg !24
  %compare.low1377 = icmp ult i32 %compare.local1376, %compare.low.base, !dbg !24
  %compare.result1378 = select i1 %compare.high.equal, i1 %compare.low1377, i1 %compare.high.order, !dbg !24
  %592 = select i1 %compare.result1378, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1380 = or i32 %compare.local.base1379, 15, !dbg !24
  %compare.low1381 = icmp ult i32 %compare.local1380, %compare.low.base, !dbg !24
  %compare.result1382 = select i1 %compare.high.equal, i1 %compare.low1381, i1 %compare.high.order, !dbg !24
  %593 = select i1 %compare.result1382, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1384 = or i32 %compare.local.base1383, 64, !dbg !24
  %compare.low1385 = icmp ult i32 %compare.local1384, %compare.low.base, !dbg !24
  %compare.result1386 = select i1 %compare.high.equal, i1 %compare.low1385, i1 %compare.high.order, !dbg !24
  %594 = select i1 %compare.result1386, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1387 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1388 = or i32 %compare.local.base1387, 65, !dbg !24
  %compare.low1389 = icmp ult i32 %compare.local1388, %compare.low.base, !dbg !24
  %compare.result1390 = select i1 %compare.high.equal, i1 %compare.low1389, i1 %compare.high.order, !dbg !24
  %595 = select i1 %compare.result1390, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1391 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1392 = or i32 %compare.local.base1391, 66, !dbg !24
  %compare.low1393 = icmp ult i32 %compare.local1392, %compare.low.base, !dbg !24
  %compare.result1394 = select i1 %compare.high.equal, i1 %compare.low1393, i1 %compare.high.order, !dbg !24
  %596 = select i1 %compare.result1394, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1396 = or i32 %compare.local.base1395, 67, !dbg !24
  %compare.low1397 = icmp ult i32 %compare.local1396, %compare.low.base, !dbg !24
  %compare.result1398 = select i1 %compare.high.equal, i1 %compare.low1397, i1 %compare.high.order, !dbg !24
  %597 = select i1 %compare.result1398, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1400 = or i32 %compare.local.base1399, 68, !dbg !24
  %compare.low1401 = icmp ult i32 %compare.local1400, %compare.low.base, !dbg !24
  %compare.result1402 = select i1 %compare.high.equal, i1 %compare.low1401, i1 %compare.high.order, !dbg !24
  %598 = select i1 %compare.result1402, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1404 = or i32 %compare.local.base1403, 69, !dbg !24
  %compare.low1405 = icmp ult i32 %compare.local1404, %compare.low.base, !dbg !24
  %compare.result1406 = select i1 %compare.high.equal, i1 %compare.low1405, i1 %compare.high.order, !dbg !24
  %599 = select i1 %compare.result1406, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1408 = or i32 %compare.local.base1407, 70, !dbg !24
  %compare.low1409 = icmp ult i32 %compare.local1408, %compare.low.base, !dbg !24
  %compare.result1410 = select i1 %compare.high.equal, i1 %compare.low1409, i1 %compare.high.order, !dbg !24
  %600 = select i1 %compare.result1410, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1412 = or i32 %compare.local.base1411, 71, !dbg !24
  %compare.low1413 = icmp ult i32 %compare.local1412, %compare.low.base, !dbg !24
  %compare.result1414 = select i1 %compare.high.equal, i1 %compare.low1413, i1 %compare.high.order, !dbg !24
  %601 = select i1 %compare.result1414, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1416 = or i32 %compare.local.base1415, 72, !dbg !24
  %compare.low1417 = icmp ult i32 %compare.local1416, %compare.low.base, !dbg !24
  %compare.result1418 = select i1 %compare.high.equal, i1 %compare.low1417, i1 %compare.high.order, !dbg !24
  %602 = select i1 %compare.result1418, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1419 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1420 = or i32 %compare.local.base1419, 73, !dbg !24
  %compare.low1421 = icmp ult i32 %compare.local1420, %compare.low.base, !dbg !24
  %compare.result1422 = select i1 %compare.high.equal, i1 %compare.low1421, i1 %compare.high.order, !dbg !24
  %603 = select i1 %compare.result1422, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1423 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1424 = or i32 %compare.local.base1423, 74, !dbg !24
  %compare.low1425 = icmp ult i32 %compare.local1424, %compare.low.base, !dbg !24
  %compare.result1426 = select i1 %compare.high.equal, i1 %compare.low1425, i1 %compare.high.order, !dbg !24
  %604 = select i1 %compare.result1426, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1427 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1428 = or i32 %compare.local.base1427, 75, !dbg !24
  %compare.low1429 = icmp ult i32 %compare.local1428, %compare.low.base, !dbg !24
  %compare.result1430 = select i1 %compare.high.equal, i1 %compare.low1429, i1 %compare.high.order, !dbg !24
  %605 = select i1 %compare.result1430, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1431 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1432 = or i32 %compare.local.base1431, 76, !dbg !24
  %compare.low1433 = icmp ult i32 %compare.local1432, %compare.low.base, !dbg !24
  %compare.result1434 = select i1 %compare.high.equal, i1 %compare.low1433, i1 %compare.high.order, !dbg !24
  %606 = select i1 %compare.result1434, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1436 = or i32 %compare.local.base1435, 77, !dbg !24
  %compare.low1437 = icmp ult i32 %compare.local1436, %compare.low.base, !dbg !24
  %compare.result1438 = select i1 %compare.high.equal, i1 %compare.low1437, i1 %compare.high.order, !dbg !24
  %607 = select i1 %compare.result1438, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1440 = or i32 %compare.local.base1439, 78, !dbg !24
  %compare.low1441 = icmp ult i32 %compare.local1440, %compare.low.base, !dbg !24
  %compare.result1442 = select i1 %compare.high.equal, i1 %compare.low1441, i1 %compare.high.order, !dbg !24
  %608 = select i1 %compare.result1442, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1443 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1444 = or i32 %compare.local.base1443, 79, !dbg !24
  %compare.low1445 = icmp ult i32 %compare.local1444, %compare.low.base, !dbg !24
  %compare.result1446 = select i1 %compare.high.equal, i1 %compare.low1445, i1 %compare.high.order, !dbg !24
  %609 = select i1 %compare.result1446, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %610 = add i64 %572, 65, !dbg !23
  %611 = lshr i64 %610, 32, !dbg !24
  %compare.high1447 = trunc nuw i64 %611 to i32, !dbg !24
  %612 = lshr i64 %146, 32, !dbg !24
  %compare.high1448 = trunc nuw i64 %612 to i32, !dbg !24
  %compare.high.equal1449 = icmp eq i32 %compare.high1447, %compare.high1448, !dbg !24
  %compare.high.order1450 = icmp ugt i32 %compare.high1447, %compare.high1448, !dbg !24
  %compare.low.base1451 = trunc i64 %610 to i32, !dbg !24
  %compare.low.base1452 = trunc i64 %574 to i32, !dbg !24
  %compare.low1453 = icmp ugt i32 %compare.low.base1451, %compare.low.base1452, !dbg !24
  %compare.result1454 = select i1 %compare.high.equal1449, i1 %compare.low1453, i1 %compare.high.order1450, !dbg !24
  %613 = select i1 %compare.result1454, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1455 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1456 = or i32 %compare.local.base1455, 1, !dbg !24
  %compare.low1457 = icmp ult i32 %compare.local1456, %compare.low.base1451, !dbg !24
  %compare.result1458 = select i1 %compare.high.equal1449, i1 %compare.low1457, i1 %compare.high.order1450, !dbg !24
  %614 = select i1 %compare.result1458, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1460 = or i32 %compare.local.base1459, 2, !dbg !24
  %compare.low1461 = icmp ult i32 %compare.local1460, %compare.low.base1451, !dbg !24
  %compare.result1462 = select i1 %compare.high.equal1449, i1 %compare.low1461, i1 %compare.high.order1450, !dbg !24
  %615 = select i1 %compare.result1462, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1464 = or i32 %compare.local.base1463, 3, !dbg !24
  %compare.low1465 = icmp ult i32 %compare.local1464, %compare.low.base1451, !dbg !24
  %compare.result1466 = select i1 %compare.high.equal1449, i1 %compare.low1465, i1 %compare.high.order1450, !dbg !24
  %616 = select i1 %compare.result1466, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1468 = or i32 %compare.local.base1467, 4, !dbg !24
  %compare.low1469 = icmp ult i32 %compare.local1468, %compare.low.base1451, !dbg !24
  %compare.result1470 = select i1 %compare.high.equal1449, i1 %compare.low1469, i1 %compare.high.order1450, !dbg !24
  %617 = select i1 %compare.result1470, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1472 = or i32 %compare.local.base1471, 5, !dbg !24
  %compare.low1473 = icmp ult i32 %compare.local1472, %compare.low.base1451, !dbg !24
  %compare.result1474 = select i1 %compare.high.equal1449, i1 %compare.low1473, i1 %compare.high.order1450, !dbg !24
  %618 = select i1 %compare.result1474, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1476 = or i32 %compare.local.base1475, 6, !dbg !24
  %compare.low1477 = icmp ult i32 %compare.local1476, %compare.low.base1451, !dbg !24
  %compare.result1478 = select i1 %compare.high.equal1449, i1 %compare.low1477, i1 %compare.high.order1450, !dbg !24
  %619 = select i1 %compare.result1478, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1480 = or i32 %compare.local.base1479, 7, !dbg !24
  %compare.low1481 = icmp ult i32 %compare.local1480, %compare.low.base1451, !dbg !24
  %compare.result1482 = select i1 %compare.high.equal1449, i1 %compare.low1481, i1 %compare.high.order1450, !dbg !24
  %620 = select i1 %compare.result1482, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1483 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1484 = or i32 %compare.local.base1483, 8, !dbg !24
  %compare.low1485 = icmp ult i32 %compare.local1484, %compare.low.base1451, !dbg !24
  %compare.result1486 = select i1 %compare.high.equal1449, i1 %compare.low1485, i1 %compare.high.order1450, !dbg !24
  %621 = select i1 %compare.result1486, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1487 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1488 = or i32 %compare.local.base1487, 9, !dbg !24
  %compare.low1489 = icmp ult i32 %compare.local1488, %compare.low.base1451, !dbg !24
  %compare.result1490 = select i1 %compare.high.equal1449, i1 %compare.low1489, i1 %compare.high.order1450, !dbg !24
  %622 = select i1 %compare.result1490, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1491 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1492 = or i32 %compare.local.base1491, 10, !dbg !24
  %compare.low1493 = icmp ult i32 %compare.local1492, %compare.low.base1451, !dbg !24
  %compare.result1494 = select i1 %compare.high.equal1449, i1 %compare.low1493, i1 %compare.high.order1450, !dbg !24
  %623 = select i1 %compare.result1494, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1495 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1496 = or i32 %compare.local.base1495, 11, !dbg !24
  %compare.low1497 = icmp ult i32 %compare.local1496, %compare.low.base1451, !dbg !24
  %compare.result1498 = select i1 %compare.high.equal1449, i1 %compare.low1497, i1 %compare.high.order1450, !dbg !24
  %624 = select i1 %compare.result1498, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1499 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1500 = or i32 %compare.local.base1499, 12, !dbg !24
  %compare.low1501 = icmp ult i32 %compare.local1500, %compare.low.base1451, !dbg !24
  %compare.result1502 = select i1 %compare.high.equal1449, i1 %compare.low1501, i1 %compare.high.order1450, !dbg !24
  %625 = select i1 %compare.result1502, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1503 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1504 = or i32 %compare.local.base1503, 13, !dbg !24
  %compare.low1505 = icmp ult i32 %compare.local1504, %compare.low.base1451, !dbg !24
  %compare.result1506 = select i1 %compare.high.equal1449, i1 %compare.low1505, i1 %compare.high.order1450, !dbg !24
  %626 = select i1 %compare.result1506, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1507 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1508 = or i32 %compare.local.base1507, 14, !dbg !24
  %compare.low1509 = icmp ult i32 %compare.local1508, %compare.low.base1451, !dbg !24
  %compare.result1510 = select i1 %compare.high.equal1449, i1 %compare.low1509, i1 %compare.high.order1450, !dbg !24
  %627 = select i1 %compare.result1510, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1511 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1512 = or i32 %compare.local.base1511, 15, !dbg !24
  %compare.low1513 = icmp ult i32 %compare.local1512, %compare.low.base1451, !dbg !24
  %compare.result1514 = select i1 %compare.high.equal1449, i1 %compare.low1513, i1 %compare.high.order1450, !dbg !24
  %628 = select i1 %compare.result1514, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1515 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1516 = or i32 %compare.local.base1515, 64, !dbg !24
  %compare.low1517 = icmp ult i32 %compare.local1516, %compare.low.base1451, !dbg !24
  %compare.result1518 = select i1 %compare.high.equal1449, i1 %compare.low1517, i1 %compare.high.order1450, !dbg !24
  %629 = select i1 %compare.result1518, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1519 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1520 = or i32 %compare.local.base1519, 65, !dbg !24
  %compare.low1521 = icmp ult i32 %compare.local1520, %compare.low.base1451, !dbg !24
  %compare.result1522 = select i1 %compare.high.equal1449, i1 %compare.low1521, i1 %compare.high.order1450, !dbg !24
  %630 = select i1 %compare.result1522, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1523 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1524 = or i32 %compare.local.base1523, 66, !dbg !24
  %compare.low1525 = icmp ult i32 %compare.local1524, %compare.low.base1451, !dbg !24
  %compare.result1526 = select i1 %compare.high.equal1449, i1 %compare.low1525, i1 %compare.high.order1450, !dbg !24
  %631 = select i1 %compare.result1526, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1527 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1528 = or i32 %compare.local.base1527, 67, !dbg !24
  %compare.low1529 = icmp ult i32 %compare.local1528, %compare.low.base1451, !dbg !24
  %compare.result1530 = select i1 %compare.high.equal1449, i1 %compare.low1529, i1 %compare.high.order1450, !dbg !24
  %632 = select i1 %compare.result1530, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1531 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1532 = or i32 %compare.local.base1531, 68, !dbg !24
  %compare.low1533 = icmp ult i32 %compare.local1532, %compare.low.base1451, !dbg !24
  %compare.result1534 = select i1 %compare.high.equal1449, i1 %compare.low1533, i1 %compare.high.order1450, !dbg !24
  %633 = select i1 %compare.result1534, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1535 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1536 = or i32 %compare.local.base1535, 69, !dbg !24
  %compare.low1537 = icmp ult i32 %compare.local1536, %compare.low.base1451, !dbg !24
  %compare.result1538 = select i1 %compare.high.equal1449, i1 %compare.low1537, i1 %compare.high.order1450, !dbg !24
  %634 = select i1 %compare.result1538, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1539 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1540 = or i32 %compare.local.base1539, 70, !dbg !24
  %compare.low1541 = icmp ult i32 %compare.local1540, %compare.low.base1451, !dbg !24
  %compare.result1542 = select i1 %compare.high.equal1449, i1 %compare.low1541, i1 %compare.high.order1450, !dbg !24
  %635 = select i1 %compare.result1542, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1543 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1544 = or i32 %compare.local.base1543, 71, !dbg !24
  %compare.low1545 = icmp ult i32 %compare.local1544, %compare.low.base1451, !dbg !24
  %compare.result1546 = select i1 %compare.high.equal1449, i1 %compare.low1545, i1 %compare.high.order1450, !dbg !24
  %636 = select i1 %compare.result1546, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1547 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1548 = or i32 %compare.local.base1547, 72, !dbg !24
  %compare.low1549 = icmp ult i32 %compare.local1548, %compare.low.base1451, !dbg !24
  %compare.result1550 = select i1 %compare.high.equal1449, i1 %compare.low1549, i1 %compare.high.order1450, !dbg !24
  %637 = select i1 %compare.result1550, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1551 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1552 = or i32 %compare.local.base1551, 73, !dbg !24
  %compare.low1553 = icmp ult i32 %compare.local1552, %compare.low.base1451, !dbg !24
  %compare.result1554 = select i1 %compare.high.equal1449, i1 %compare.low1553, i1 %compare.high.order1450, !dbg !24
  %638 = select i1 %compare.result1554, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1555 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1556 = or i32 %compare.local.base1555, 74, !dbg !24
  %compare.low1557 = icmp ult i32 %compare.local1556, %compare.low.base1451, !dbg !24
  %compare.result1558 = select i1 %compare.high.equal1449, i1 %compare.low1557, i1 %compare.high.order1450, !dbg !24
  %639 = select i1 %compare.result1558, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1559 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1560 = or i32 %compare.local.base1559, 75, !dbg !24
  %compare.low1561 = icmp ult i32 %compare.local1560, %compare.low.base1451, !dbg !24
  %compare.result1562 = select i1 %compare.high.equal1449, i1 %compare.low1561, i1 %compare.high.order1450, !dbg !24
  %640 = select i1 %compare.result1562, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1563 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1564 = or i32 %compare.local.base1563, 76, !dbg !24
  %compare.low1565 = icmp ult i32 %compare.local1564, %compare.low.base1451, !dbg !24
  %compare.result1566 = select i1 %compare.high.equal1449, i1 %compare.low1565, i1 %compare.high.order1450, !dbg !24
  %641 = select i1 %compare.result1566, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1567 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1568 = or i32 %compare.local.base1567, 77, !dbg !24
  %compare.low1569 = icmp ult i32 %compare.local1568, %compare.low.base1451, !dbg !24
  %compare.result1570 = select i1 %compare.high.equal1449, i1 %compare.low1569, i1 %compare.high.order1450, !dbg !24
  %642 = select i1 %compare.result1570, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1571 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1572 = or i32 %compare.local.base1571, 78, !dbg !24
  %compare.low1573 = icmp ult i32 %compare.local1572, %compare.low.base1451, !dbg !24
  %compare.result1574 = select i1 %compare.high.equal1449, i1 %compare.low1573, i1 %compare.high.order1450, !dbg !24
  %643 = select i1 %compare.result1574, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1575 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1576 = or i32 %compare.local.base1575, 79, !dbg !24
  %compare.low1577 = icmp ult i32 %compare.local1576, %compare.low.base1451, !dbg !24
  %compare.result1578 = select i1 %compare.high.equal1449, i1 %compare.low1577, i1 %compare.high.order1450, !dbg !24
  %644 = select i1 %compare.result1578, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %645 = insertelement <4 x float> poison, float %578, i64 0, !dbg !13
  %646 = insertelement <4 x float> %645, float %579, i64 1, !dbg !13
  %647 = insertelement <4 x float> %646, float %580, i64 2, !dbg !13
  %.masked126 = insertelement <4 x float> %647, float %581, i64 3, !dbg !13
  %648 = insertelement <4 x float> poison, float %582, i64 0, !dbg !13
  %649 = insertelement <4 x float> %648, float %583, i64 1, !dbg !13
  %650 = insertelement <4 x float> %649, float %584, i64 2, !dbg !13
  %.masked128 = insertelement <4 x float> %650, float %585, i64 3, !dbg !13
  %651 = insertelement <4 x float> poison, float %586, i64 0, !dbg !13
  %652 = insertelement <4 x float> %651, float %587, i64 1, !dbg !13
  %653 = insertelement <4 x float> %652, float %588, i64 2, !dbg !13
  %.masked130 = insertelement <4 x float> %653, float %589, i64 3, !dbg !13
  %654 = insertelement <4 x float> poison, float %590, i64 0, !dbg !13
  %655 = insertelement <4 x float> %654, float %591, i64 1, !dbg !13
  %656 = insertelement <4 x float> %655, float %592, i64 2, !dbg !13
  %.masked132 = insertelement <4 x float> %656, float %593, i64 3, !dbg !13
  %657 = insertelement <4 x float> poison, float %594, i64 0, !dbg !13
  %658 = insertelement <4 x float> %657, float %595, i64 1, !dbg !13
  %659 = insertelement <4 x float> %658, float %596, i64 2, !dbg !13
  %.masked134 = insertelement <4 x float> %659, float %597, i64 3, !dbg !13
  %660 = insertelement <4 x float> poison, float %598, i64 0, !dbg !13
  %661 = insertelement <4 x float> %660, float %599, i64 1, !dbg !13
  %662 = insertelement <4 x float> %661, float %600, i64 2, !dbg !13
  %.masked136 = insertelement <4 x float> %662, float %601, i64 3, !dbg !13
  %663 = insertelement <4 x float> poison, float %602, i64 0, !dbg !13
  %664 = insertelement <4 x float> %663, float %603, i64 1, !dbg !13
  %665 = insertelement <4 x float> %664, float %604, i64 2, !dbg !13
  %.masked138 = insertelement <4 x float> %665, float %605, i64 3, !dbg !13
  %666 = insertelement <4 x float> poison, float %606, i64 0, !dbg !13
  %667 = insertelement <4 x float> %666, float %607, i64 1, !dbg !13
  %668 = insertelement <4 x float> %667, float %608, i64 2, !dbg !13
  %.masked140 = insertelement <4 x float> %668, float %609, i64 3, !dbg !13
  %669 = insertelement <4 x float> poison, float %613, i64 0, !dbg !13
  %670 = insertelement <4 x float> %669, float %614, i64 1, !dbg !13
  %671 = insertelement <4 x float> %670, float %615, i64 2, !dbg !13
  %.masked143 = insertelement <4 x float> %671, float %616, i64 3, !dbg !13
  %672 = insertelement <4 x float> poison, float %617, i64 0, !dbg !13
  %673 = insertelement <4 x float> %672, float %618, i64 1, !dbg !13
  %674 = insertelement <4 x float> %673, float %619, i64 2, !dbg !13
  %.masked145 = insertelement <4 x float> %674, float %620, i64 3, !dbg !13
  %675 = insertelement <4 x float> poison, float %621, i64 0, !dbg !13
  %676 = insertelement <4 x float> %675, float %622, i64 1, !dbg !13
  %677 = insertelement <4 x float> %676, float %623, i64 2, !dbg !13
  %.masked147 = insertelement <4 x float> %677, float %624, i64 3, !dbg !13
  %678 = insertelement <4 x float> poison, float %625, i64 0, !dbg !13
  %679 = insertelement <4 x float> %678, float %626, i64 1, !dbg !13
  %680 = insertelement <4 x float> %679, float %627, i64 2, !dbg !13
  %.masked149 = insertelement <4 x float> %680, float %628, i64 3, !dbg !13
  %681 = insertelement <4 x float> poison, float %629, i64 0, !dbg !13
  %682 = insertelement <4 x float> %681, float %630, i64 1, !dbg !13
  %683 = insertelement <4 x float> %682, float %631, i64 2, !dbg !13
  %.masked151 = insertelement <4 x float> %683, float %632, i64 3, !dbg !13
  %684 = insertelement <4 x float> poison, float %633, i64 0, !dbg !13
  %685 = insertelement <4 x float> %684, float %634, i64 1, !dbg !13
  %686 = insertelement <4 x float> %685, float %635, i64 2, !dbg !13
  %.masked153 = insertelement <4 x float> %686, float %636, i64 3, !dbg !13
  %687 = insertelement <4 x float> poison, float %637, i64 0, !dbg !13
  %688 = insertelement <4 x float> %687, float %638, i64 1, !dbg !13
  %689 = insertelement <4 x float> %688, float %639, i64 2, !dbg !13
  %.masked155 = insertelement <4 x float> %689, float %640, i64 3, !dbg !13
  %690 = insertelement <4 x float> poison, float %641, i64 0, !dbg !13
  %691 = insertelement <4 x float> %690, float %642, i64 1, !dbg !13
  %692 = insertelement <4 x float> %691, float %643, i64 2, !dbg !13
  %.masked157 = insertelement <4 x float> %692, float %644, i64 3, !dbg !13
  %.masked127 = fadd <4 x float> %503, %.masked126, !dbg !26
  %.masked129 = fadd <4 x float> %507, %.masked128, !dbg !26
  %.masked131 = fadd <4 x float> %511, %.masked130, !dbg !26
  %.masked133 = fadd <4 x float> %515, %.masked132, !dbg !26
  %.masked135 = fadd <4 x float> %519, %.masked134, !dbg !26
  %.masked137 = fadd <4 x float> %523, %.masked136, !dbg !26
  %.masked139 = fadd <4 x float> %527, %.masked138, !dbg !26
  %.masked141 = fadd <4 x float> %531, %.masked140, !dbg !26
  %.masked144 = fadd <4 x float> %535, %.masked143, !dbg !26
  %.masked146 = fadd <4 x float> %539, %.masked145, !dbg !26
  %.masked148 = fadd <4 x float> %543, %.masked147, !dbg !26
  %.masked150 = fadd <4 x float> %547, %.masked149, !dbg !26
  %.masked152 = fadd <4 x float> %551, %.masked151, !dbg !26
  %.masked154 = fadd <4 x float> %555, %.masked153, !dbg !26
  %.masked156 = fadd <4 x float> %559, %.masked155, !dbg !26
  %.masked158 = fadd <4 x float> %564, %.masked157, !dbg !26
  br label %693, !dbg !13

693:                                              ; preds = %566, %145
  %694 = phi <4 x float> [ %.masked127, %566 ], [ %503, %145 ], !dbg !26
  %695 = phi <4 x float> [ %.masked129, %566 ], [ %507, %145 ], !dbg !26
  %696 = phi <4 x float> [ %.masked131, %566 ], [ %511, %145 ], !dbg !26
  %697 = phi <4 x float> [ %.masked133, %566 ], [ %515, %145 ], !dbg !26
  %698 = phi <4 x float> [ %.masked135, %566 ], [ %519, %145 ], !dbg !26
  %699 = phi <4 x float> [ %.masked137, %566 ], [ %523, %145 ], !dbg !26
  %700 = phi <4 x float> [ %.masked139, %566 ], [ %527, %145 ], !dbg !26
  %701 = phi <4 x float> [ %.masked141, %566 ], [ %531, %145 ], !dbg !26
  %702 = phi <4 x float> [ %.masked144, %566 ], [ %535, %145 ], !dbg !26
  %703 = phi <4 x float> [ %.masked146, %566 ], [ %539, %145 ], !dbg !26
  %704 = phi <4 x float> [ %.masked148, %566 ], [ %543, %145 ], !dbg !26
  %705 = phi <4 x float> [ %.masked150, %566 ], [ %547, %145 ], !dbg !26
  %706 = phi <4 x float> [ %.masked152, %566 ], [ %551, %145 ], !dbg !26
  %707 = phi <4 x float> [ %.masked154, %566 ], [ %555, %145 ], !dbg !26
  %708 = phi <4 x float> [ %.masked156, %566 ], [ %559, %145 ], !dbg !26
  %709 = phi <4 x float> [ %.masked158, %566 ], [ %564, %145 ], !dbg !26
  %710 = extractelement <4 x float> %694, i64 0, !dbg !26
  %711 = extractelement <4 x float> %694, i64 1, !dbg !26
  %712 = extractelement <4 x float> %694, i64 2, !dbg !26
  %713 = extractelement <4 x float> %694, i64 3, !dbg !26
  %714 = extractelement <4 x float> %695, i64 0, !dbg !26
  %715 = extractelement <4 x float> %695, i64 1, !dbg !26
  %716 = extractelement <4 x float> %695, i64 2, !dbg !26
  %717 = extractelement <4 x float> %695, i64 3, !dbg !26
  %718 = extractelement <4 x float> %696, i64 0, !dbg !26
  %719 = extractelement <4 x float> %696, i64 1, !dbg !26
  %720 = extractelement <4 x float> %696, i64 2, !dbg !26
  %721 = extractelement <4 x float> %696, i64 3, !dbg !26
  %722 = extractelement <4 x float> %697, i64 0, !dbg !26
  %723 = extractelement <4 x float> %697, i64 1, !dbg !26
  %724 = extractelement <4 x float> %697, i64 2, !dbg !26
  %725 = extractelement <4 x float> %697, i64 3, !dbg !26
  %726 = extractelement <4 x float> %698, i64 0, !dbg !26
  %727 = extractelement <4 x float> %698, i64 1, !dbg !26
  %728 = extractelement <4 x float> %698, i64 2, !dbg !26
  %729 = extractelement <4 x float> %698, i64 3, !dbg !26
  %730 = extractelement <4 x float> %699, i64 0, !dbg !26
  %731 = extractelement <4 x float> %699, i64 1, !dbg !26
  %732 = extractelement <4 x float> %699, i64 2, !dbg !26
  %733 = extractelement <4 x float> %699, i64 3, !dbg !26
  %734 = extractelement <4 x float> %700, i64 0, !dbg !26
  %735 = extractelement <4 x float> %700, i64 1, !dbg !26
  %736 = extractelement <4 x float> %700, i64 2, !dbg !26
  %737 = extractelement <4 x float> %700, i64 3, !dbg !26
  %738 = extractelement <4 x float> %701, i64 0, !dbg !26
  %739 = extractelement <4 x float> %701, i64 1, !dbg !26
  %740 = extractelement <4 x float> %701, i64 2, !dbg !26
  %741 = extractelement <4 x float> %701, i64 3, !dbg !26
  %742 = extractelement <4 x float> %702, i64 0, !dbg !26
  %743 = extractelement <4 x float> %702, i64 1, !dbg !26
  %744 = extractelement <4 x float> %702, i64 2, !dbg !26
  %745 = extractelement <4 x float> %702, i64 3, !dbg !26
  %746 = extractelement <4 x float> %703, i64 0, !dbg !26
  %747 = extractelement <4 x float> %703, i64 1, !dbg !26
  %748 = extractelement <4 x float> %703, i64 2, !dbg !26
  %749 = extractelement <4 x float> %703, i64 3, !dbg !26
  %750 = extractelement <4 x float> %704, i64 0, !dbg !26
  %751 = extractelement <4 x float> %704, i64 1, !dbg !26
  %752 = extractelement <4 x float> %704, i64 2, !dbg !26
  %753 = extractelement <4 x float> %704, i64 3, !dbg !26
  %754 = extractelement <4 x float> %705, i64 0, !dbg !26
  %755 = extractelement <4 x float> %705, i64 1, !dbg !26
  %756 = extractelement <4 x float> %705, i64 2, !dbg !26
  %757 = extractelement <4 x float> %705, i64 3, !dbg !26
  %758 = extractelement <4 x float> %706, i64 0, !dbg !26
  %759 = extractelement <4 x float> %706, i64 1, !dbg !26
  %760 = extractelement <4 x float> %706, i64 2, !dbg !26
  %761 = extractelement <4 x float> %706, i64 3, !dbg !26
  %762 = extractelement <4 x float> %707, i64 0, !dbg !26
  %763 = extractelement <4 x float> %707, i64 1, !dbg !26
  %764 = extractelement <4 x float> %707, i64 2, !dbg !26
  %765 = extractelement <4 x float> %707, i64 3, !dbg !26
  %766 = extractelement <4 x float> %708, i64 0, !dbg !26
  %767 = extractelement <4 x float> %708, i64 1, !dbg !26
  %768 = extractelement <4 x float> %708, i64 2, !dbg !26
  %769 = extractelement <4 x float> %708, i64 3, !dbg !26
  %770 = extractelement <4 x float> %709, i64 0, !dbg !26
  %771 = extractelement <4 x float> %709, i64 1, !dbg !26
  %772 = extractelement <4 x float> %709, i64 2, !dbg !26
  %773 = extractelement <4 x float> %709, i64 3, !dbg !26
  %774 = call float @llvm.maxnum.f32(float %710, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %local.reduce251 = call float @llvm.maxnum.f32(float %774, float %714), !dbg !27, !frisk.local_reduction !1
  %local.reduce252 = call float @llvm.maxnum.f32(float %711, float %715), !dbg !27, !frisk.local_reduction !1
  %local.reduce253 = call float @llvm.maxnum.f32(float %712, float %716), !dbg !27, !frisk.local_reduction !1
  %local.reduce254 = call float @llvm.maxnum.f32(float %713, float %717), !dbg !27, !frisk.local_reduction !1
  %local.reduce255 = call float @llvm.maxnum.f32(float %local.reduce251, float %718), !dbg !27, !frisk.local_reduction !1
  %local.reduce256 = call float @llvm.maxnum.f32(float %local.reduce252, float %719), !dbg !27, !frisk.local_reduction !1
  %local.reduce257 = call float @llvm.maxnum.f32(float %local.reduce253, float %720), !dbg !27, !frisk.local_reduction !1
  %local.reduce258 = call float @llvm.maxnum.f32(float %local.reduce254, float %721), !dbg !27, !frisk.local_reduction !1
  %local.reduce259 = call float @llvm.maxnum.f32(float %local.reduce255, float %722), !dbg !27, !frisk.local_reduction !1
  %local.reduce260 = call float @llvm.maxnum.f32(float %local.reduce256, float %723), !dbg !27, !frisk.local_reduction !1
  %local.reduce261 = call float @llvm.maxnum.f32(float %local.reduce257, float %724), !dbg !27, !frisk.local_reduction !1
  %local.reduce262 = call float @llvm.maxnum.f32(float %local.reduce258, float %725), !dbg !27, !frisk.local_reduction !1
  %local.reduce263 = call float @llvm.maxnum.f32(float %local.reduce259, float %726), !dbg !27, !frisk.local_reduction !1
  %local.reduce264 = call float @llvm.maxnum.f32(float %local.reduce260, float %727), !dbg !27, !frisk.local_reduction !1
  %local.reduce265 = call float @llvm.maxnum.f32(float %local.reduce261, float %728), !dbg !27, !frisk.local_reduction !1
  %local.reduce266 = call float @llvm.maxnum.f32(float %local.reduce262, float %729), !dbg !27, !frisk.local_reduction !1
  %local.reduce267 = call float @llvm.maxnum.f32(float %local.reduce263, float %730), !dbg !27, !frisk.local_reduction !1
  %local.reduce268 = call float @llvm.maxnum.f32(float %local.reduce264, float %731), !dbg !27, !frisk.local_reduction !1
  %local.reduce269 = call float @llvm.maxnum.f32(float %local.reduce265, float %732), !dbg !27, !frisk.local_reduction !1
  %local.reduce270 = call float @llvm.maxnum.f32(float %local.reduce266, float %733), !dbg !27, !frisk.local_reduction !1
  %local.reduce271 = call float @llvm.maxnum.f32(float %local.reduce267, float %734), !dbg !27, !frisk.local_reduction !1
  %local.reduce272 = call float @llvm.maxnum.f32(float %local.reduce268, float %735), !dbg !27, !frisk.local_reduction !1
  %local.reduce273 = call float @llvm.maxnum.f32(float %local.reduce269, float %736), !dbg !27, !frisk.local_reduction !1
  %local.reduce274 = call float @llvm.maxnum.f32(float %local.reduce270, float %737), !dbg !27, !frisk.local_reduction !1
  %local.reduce275 = call float @llvm.maxnum.f32(float %local.reduce271, float %738), !dbg !27, !frisk.local_reduction !1
  %local.reduce276 = call float @llvm.maxnum.f32(float %local.reduce272, float %739), !dbg !27, !frisk.local_reduction !1
  %local.reduce277 = call float @llvm.maxnum.f32(float %local.reduce273, float %740), !dbg !27, !frisk.local_reduction !1
  %local.reduce278 = call float @llvm.maxnum.f32(float %local.reduce274, float %741), !dbg !27, !frisk.local_reduction !1
  %local.reduce279 = call float @llvm.maxnum.f32(float %local.reduce275, float %local.reduce276), !dbg !27, !frisk.local_reduction !1
  %local.reduce280 = call float @llvm.maxnum.f32(float %local.reduce277, float %local.reduce278), !dbg !27, !frisk.local_reduction !1
  %775 = call float @llvm.maxnum.f32(float %local.reduce279, float %local.reduce280), !dbg !27, !frisk.local_reduction !1
  %776 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %777 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %776), !dbg !27
  %778 = add nuw nsw i32 %777, 64, !dbg !27
  %779 = xor i32 %777, 16, !dbg !27
  %780 = and i32 %778, 192, !dbg !27
  %781 = icmp samesign ult i32 %779, %780, !dbg !27
  %782 = select i1 %781, i32 %779, i32 %777, !dbg !27
  %783 = shl nuw nsw i32 %782, 2, !dbg !27
  %784 = bitcast float %775 to i32, !dbg !27
  %785 = call i32 @llvm.amdgcn.ds.bpermute(i32 %783, i32 %784), !dbg !27
  %786 = bitcast i32 %785 to float, !dbg !27
  %787 = call float @llvm.maxnum.f32(float %775, float %786), !dbg !27
  %788 = xor i32 %777, 32, !dbg !27
  %789 = icmp samesign ult i32 %788, %780, !dbg !27
  %790 = select i1 %789, i32 %788, i32 %777, !dbg !27
  %791 = shl nuw nsw i32 %790, 2, !dbg !27
  %792 = bitcast float %787 to i32, !dbg !27
  %793 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %792), !dbg !27
  %794 = bitcast i32 %793 to float, !dbg !27
  %795 = call float @llvm.maxnum.f32(float %787, float %794), !dbg !27
  %796 = extractvalue [2 x <1 x float>] %140, 0, !dbg !27
  %797 = insertelement <1 x float> poison, float %795, i64 0, !dbg !27
  %798 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %797, <1 x float> %796), !dbg !27
  %799 = call float @llvm.maxnum.f32(float %742, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %local.reduce220 = call float @llvm.maxnum.f32(float %799, float %746), !dbg !27, !frisk.local_reduction !1
  %local.reduce221 = call float @llvm.maxnum.f32(float %743, float %747), !dbg !27, !frisk.local_reduction !1
  %local.reduce222 = call float @llvm.maxnum.f32(float %744, float %748), !dbg !27, !frisk.local_reduction !1
  %local.reduce223 = call float @llvm.maxnum.f32(float %745, float %749), !dbg !27, !frisk.local_reduction !1
  %local.reduce224 = call float @llvm.maxnum.f32(float %local.reduce220, float %750), !dbg !27, !frisk.local_reduction !1
  %local.reduce225 = call float @llvm.maxnum.f32(float %local.reduce221, float %751), !dbg !27, !frisk.local_reduction !1
  %local.reduce226 = call float @llvm.maxnum.f32(float %local.reduce222, float %752), !dbg !27, !frisk.local_reduction !1
  %local.reduce227 = call float @llvm.maxnum.f32(float %local.reduce223, float %753), !dbg !27, !frisk.local_reduction !1
  %local.reduce228 = call float @llvm.maxnum.f32(float %local.reduce224, float %754), !dbg !27, !frisk.local_reduction !1
  %local.reduce229 = call float @llvm.maxnum.f32(float %local.reduce225, float %755), !dbg !27, !frisk.local_reduction !1
  %local.reduce230 = call float @llvm.maxnum.f32(float %local.reduce226, float %756), !dbg !27, !frisk.local_reduction !1
  %local.reduce231 = call float @llvm.maxnum.f32(float %local.reduce227, float %757), !dbg !27, !frisk.local_reduction !1
  %local.reduce232 = call float @llvm.maxnum.f32(float %local.reduce228, float %758), !dbg !27, !frisk.local_reduction !1
  %local.reduce233 = call float @llvm.maxnum.f32(float %local.reduce229, float %759), !dbg !27, !frisk.local_reduction !1
  %local.reduce234 = call float @llvm.maxnum.f32(float %local.reduce230, float %760), !dbg !27, !frisk.local_reduction !1
  %local.reduce235 = call float @llvm.maxnum.f32(float %local.reduce231, float %761), !dbg !27, !frisk.local_reduction !1
  %local.reduce236 = call float @llvm.maxnum.f32(float %local.reduce232, float %762), !dbg !27, !frisk.local_reduction !1
  %local.reduce237 = call float @llvm.maxnum.f32(float %local.reduce233, float %763), !dbg !27, !frisk.local_reduction !1
  %local.reduce238 = call float @llvm.maxnum.f32(float %local.reduce234, float %764), !dbg !27, !frisk.local_reduction !1
  %local.reduce239 = call float @llvm.maxnum.f32(float %local.reduce235, float %765), !dbg !27, !frisk.local_reduction !1
  %local.reduce240 = call float @llvm.maxnum.f32(float %local.reduce236, float %766), !dbg !27, !frisk.local_reduction !1
  %local.reduce241 = call float @llvm.maxnum.f32(float %local.reduce237, float %767), !dbg !27, !frisk.local_reduction !1
  %local.reduce242 = call float @llvm.maxnum.f32(float %local.reduce238, float %768), !dbg !27, !frisk.local_reduction !1
  %local.reduce243 = call float @llvm.maxnum.f32(float %local.reduce239, float %769), !dbg !27, !frisk.local_reduction !1
  %local.reduce244 = call float @llvm.maxnum.f32(float %local.reduce240, float %770), !dbg !27, !frisk.local_reduction !1
  %local.reduce245 = call float @llvm.maxnum.f32(float %local.reduce241, float %771), !dbg !27, !frisk.local_reduction !1
  %local.reduce246 = call float @llvm.maxnum.f32(float %local.reduce242, float %772), !dbg !27, !frisk.local_reduction !1
  %local.reduce247 = call float @llvm.maxnum.f32(float %local.reduce243, float %773), !dbg !27, !frisk.local_reduction !1
  %local.reduce248 = call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !27, !frisk.local_reduction !1
  %local.reduce249 = call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !27, !frisk.local_reduction !1
  %800 = call float @llvm.maxnum.f32(float %local.reduce248, float %local.reduce249), !dbg !27, !frisk.local_reduction !1
  %801 = bitcast float %800 to i32, !dbg !27
  %802 = call i32 @llvm.amdgcn.ds.bpermute(i32 %783, i32 %801), !dbg !27
  %803 = bitcast i32 %802 to float, !dbg !27
  %804 = call float @llvm.maxnum.f32(float %800, float %803), !dbg !27
  %805 = bitcast float %804 to i32, !dbg !27
  %806 = call i32 @llvm.amdgcn.ds.bpermute(i32 %791, i32 %805), !dbg !27
  %807 = bitcast i32 %806 to float, !dbg !27
  %808 = call float @llvm.maxnum.f32(float %804, float %807), !dbg !27
  %809 = extractvalue [2 x <1 x float>] %140, 1, !dbg !27
  %810 = insertelement <1 x float> poison, float %808, i64 0, !dbg !27
  %811 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %810, <1 x float> %809), !dbg !27
  %812 = shufflevector <1 x float> %798, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %813 = fsub <4 x float> %694, %812, !dbg !28
  %814 = fsub <4 x float> %695, %812, !dbg !28
  %815 = fsub <4 x float> %696, %812, !dbg !28
  %816 = fsub <4 x float> %697, %812, !dbg !28
  %817 = fsub <4 x float> %698, %812, !dbg !28
  %818 = fsub <4 x float> %699, %812, !dbg !28
  %819 = fsub <4 x float> %700, %812, !dbg !28
  %820 = fsub <4 x float> %701, %812, !dbg !28
  %821 = shufflevector <1 x float> %811, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %822 = fsub <4 x float> %702, %821, !dbg !28
  %823 = fsub <4 x float> %703, %821, !dbg !28
  %824 = fsub <4 x float> %704, %821, !dbg !28
  %825 = fsub <4 x float> %705, %821, !dbg !28
  %826 = fsub <4 x float> %706, %821, !dbg !28
  %827 = fsub <4 x float> %707, %821, !dbg !28
  %828 = fsub <4 x float> %708, %821, !dbg !28
  %829 = fsub <4 x float> %709, %821, !dbg !28
  %830 = fsub <1 x float> %796, %798, !dbg !29
  %831 = extractelement <1 x float> %830, i64 0, !dbg !30
  %832 = call float @llvm.amdgcn.exp2.f32(float %831), !dbg !30
  %833 = insertelement <4 x float> poison, float %832, i64 0, !dbg !35
  %834 = shufflevector <4 x float> %833, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %835 = extractvalue [2 x <1 x float>] %139, 0, !dbg !31
  %836 = extractelement <1 x float> %835, i64 0, !dbg !31
  %837 = fmul float %836, %832, !dbg !31
  %838 = fsub <1 x float> %809, %811, !dbg !29
  %839 = extractelement <1 x float> %838, i64 0, !dbg !30
  %840 = call float @llvm.amdgcn.exp2.f32(float %839), !dbg !30
  %841 = insertelement <4 x float> poison, float %840, i64 0, !dbg !35
  %842 = shufflevector <4 x float> %841, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %843 = extractvalue [2 x <1 x float>] %139, 1, !dbg !31
  %844 = extractelement <1 x float> %843, i64 0, !dbg !31
  %845 = fmul float %844, %840, !dbg !31
  %846 = extractelement <4 x float> %813, i64 0, !dbg !32
  %847 = call float @llvm.amdgcn.exp2.f32(float %846), !dbg !32
  %848 = insertelement <4 x float> poison, float %847, i64 0, !dbg !34
  %849 = extractelement <4 x float> %813, i64 1, !dbg !32
  %850 = call float @llvm.amdgcn.exp2.f32(float %849), !dbg !32
  %851 = insertelement <4 x float> %848, float %850, i64 1, !dbg !34
  %852 = extractelement <4 x float> %813, i64 2, !dbg !32
  %853 = call float @llvm.amdgcn.exp2.f32(float %852), !dbg !32
  %854 = insertelement <4 x float> %851, float %853, i64 2, !dbg !34
  %855 = extractelement <4 x float> %813, i64 3, !dbg !32
  %856 = call float @llvm.amdgcn.exp2.f32(float %855), !dbg !32
  %857 = insertelement <4 x float> %854, float %856, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %857) #9, !dbg !34, !frisk.register_lifetimes !1
  %858 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %858) #9, !dbg !32, !frisk.register_lifetimes !1
  %859 = extractelement <4 x float> %814, i64 0, !dbg !32
  %860 = call float @llvm.amdgcn.exp2.f32(float %859), !dbg !32
  %861 = insertelement <4 x float> poison, float %860, i64 0, !dbg !34
  %local.reduce189 = fadd float %847, %860, !dbg !33, !frisk.local_reduction !1
  %862 = extractelement <4 x float> %814, i64 1, !dbg !32
  %863 = call float @llvm.amdgcn.exp2.f32(float %862), !dbg !32
  %864 = insertelement <4 x float> %861, float %863, i64 1, !dbg !34
  %local.reduce190 = fadd float %850, %863, !dbg !33, !frisk.local_reduction !1
  %865 = extractelement <4 x float> %814, i64 2, !dbg !32
  %866 = call float @llvm.amdgcn.exp2.f32(float %865), !dbg !32
  %867 = insertelement <4 x float> %864, float %866, i64 2, !dbg !34
  %local.reduce191 = fadd float %853, %866, !dbg !33, !frisk.local_reduction !1
  %868 = extractelement <4 x float> %814, i64 3, !dbg !32
  %869 = call float @llvm.amdgcn.exp2.f32(float %868), !dbg !32
  %870 = insertelement <4 x float> %867, float %869, i64 3, !dbg !34
  %math.result475 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %870) #9, !dbg !34, !frisk.register_lifetimes !1
  %871 = fptrunc <4 x float> %math.result475 to <4 x half>, !dbg !34
  %math.packed476 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %871) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce192 = fadd float %856, %869, !dbg !33, !frisk.local_reduction !1
  %872 = extractelement <4 x float> %815, i64 0, !dbg !32
  %873 = call float @llvm.amdgcn.exp2.f32(float %872), !dbg !32
  %874 = insertelement <4 x float> poison, float %873, i64 0, !dbg !34
  %local.reduce193 = fadd float %local.reduce189, %873, !dbg !33, !frisk.local_reduction !1
  %875 = extractelement <4 x float> %815, i64 1, !dbg !32
  %876 = call float @llvm.amdgcn.exp2.f32(float %875), !dbg !32
  %877 = insertelement <4 x float> %874, float %876, i64 1, !dbg !34
  %local.reduce194 = fadd float %local.reduce190, %876, !dbg !33, !frisk.local_reduction !1
  %878 = extractelement <4 x float> %815, i64 2, !dbg !32
  %879 = call float @llvm.amdgcn.exp2.f32(float %878), !dbg !32
  %880 = insertelement <4 x float> %877, float %879, i64 2, !dbg !34
  %local.reduce195 = fadd float %local.reduce191, %879, !dbg !33, !frisk.local_reduction !1
  %881 = extractelement <4 x float> %815, i64 3, !dbg !32
  %882 = call float @llvm.amdgcn.exp2.f32(float %881), !dbg !32
  %883 = insertelement <4 x float> %880, float %882, i64 3, !dbg !34
  %math.result477 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %883) #9, !dbg !34, !frisk.register_lifetimes !1
  %884 = fptrunc <4 x float> %math.result477 to <4 x half>, !dbg !34
  %math.packed478 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %884) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce196 = fadd float %local.reduce192, %882, !dbg !33, !frisk.local_reduction !1
  %885 = extractelement <4 x float> %816, i64 0, !dbg !32
  %886 = call float @llvm.amdgcn.exp2.f32(float %885), !dbg !32
  %887 = insertelement <4 x float> poison, float %886, i64 0, !dbg !34
  %local.reduce197 = fadd float %local.reduce193, %886, !dbg !33, !frisk.local_reduction !1
  %888 = extractelement <4 x float> %816, i64 1, !dbg !32
  %889 = call float @llvm.amdgcn.exp2.f32(float %888), !dbg !32
  %890 = insertelement <4 x float> %887, float %889, i64 1, !dbg !34
  %local.reduce198 = fadd float %local.reduce194, %889, !dbg !33, !frisk.local_reduction !1
  %891 = extractelement <4 x float> %816, i64 2, !dbg !32
  %892 = call float @llvm.amdgcn.exp2.f32(float %891), !dbg !32
  %893 = insertelement <4 x float> %890, float %892, i64 2, !dbg !34
  %local.reduce199 = fadd float %local.reduce195, %892, !dbg !33, !frisk.local_reduction !1
  %894 = extractelement <4 x float> %816, i64 3, !dbg !32
  %895 = call float @llvm.amdgcn.exp2.f32(float %894), !dbg !32
  %896 = insertelement <4 x float> %893, float %895, i64 3, !dbg !34
  %math.result479 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %896) #9, !dbg !34, !frisk.register_lifetimes !1
  %897 = fptrunc <4 x float> %math.result479 to <4 x half>, !dbg !34
  %math.packed480 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %897) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce200 = fadd float %local.reduce196, %895, !dbg !33, !frisk.local_reduction !1
  %898 = extractelement <4 x float> %817, i64 0, !dbg !32
  %899 = call float @llvm.amdgcn.exp2.f32(float %898), !dbg !32
  %900 = insertelement <4 x float> poison, float %899, i64 0, !dbg !34
  %local.reduce201 = fadd float %local.reduce197, %899, !dbg !33, !frisk.local_reduction !1
  %901 = extractelement <4 x float> %817, i64 1, !dbg !32
  %902 = call float @llvm.amdgcn.exp2.f32(float %901), !dbg !32
  %903 = insertelement <4 x float> %900, float %902, i64 1, !dbg !34
  %local.reduce202 = fadd float %local.reduce198, %902, !dbg !33, !frisk.local_reduction !1
  %904 = extractelement <4 x float> %817, i64 2, !dbg !32
  %905 = call float @llvm.amdgcn.exp2.f32(float %904), !dbg !32
  %906 = insertelement <4 x float> %903, float %905, i64 2, !dbg !34
  %local.reduce203 = fadd float %local.reduce199, %905, !dbg !33, !frisk.local_reduction !1
  %907 = extractelement <4 x float> %817, i64 3, !dbg !32
  %908 = call float @llvm.amdgcn.exp2.f32(float %907), !dbg !32
  %909 = insertelement <4 x float> %906, float %908, i64 3, !dbg !34
  %math.result481 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %909) #9, !dbg !34, !frisk.register_lifetimes !1
  %910 = fptrunc <4 x float> %math.result481 to <4 x half>, !dbg !34
  %math.packed482 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %910) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce204 = fadd float %local.reduce200, %908, !dbg !33, !frisk.local_reduction !1
  %911 = extractelement <4 x float> %818, i64 0, !dbg !32
  %912 = call float @llvm.amdgcn.exp2.f32(float %911), !dbg !32
  %913 = insertelement <4 x float> poison, float %912, i64 0, !dbg !34
  %local.reduce205 = fadd float %local.reduce201, %912, !dbg !33, !frisk.local_reduction !1
  %914 = extractelement <4 x float> %818, i64 1, !dbg !32
  %915 = call float @llvm.amdgcn.exp2.f32(float %914), !dbg !32
  %916 = insertelement <4 x float> %913, float %915, i64 1, !dbg !34
  %local.reduce206 = fadd float %local.reduce202, %915, !dbg !33, !frisk.local_reduction !1
  %917 = extractelement <4 x float> %818, i64 2, !dbg !32
  %918 = call float @llvm.amdgcn.exp2.f32(float %917), !dbg !32
  %919 = insertelement <4 x float> %916, float %918, i64 2, !dbg !34
  %local.reduce207 = fadd float %local.reduce203, %918, !dbg !33, !frisk.local_reduction !1
  %920 = extractelement <4 x float> %818, i64 3, !dbg !32
  %921 = call float @llvm.amdgcn.exp2.f32(float %920), !dbg !32
  %922 = insertelement <4 x float> %919, float %921, i64 3, !dbg !34
  %math.result483 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %922) #9, !dbg !34, !frisk.register_lifetimes !1
  %923 = fptrunc <4 x float> %math.result483 to <4 x half>, !dbg !34
  %math.packed484 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %923) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce208 = fadd float %local.reduce204, %921, !dbg !33, !frisk.local_reduction !1
  %924 = extractelement <4 x float> %819, i64 0, !dbg !32
  %925 = call float @llvm.amdgcn.exp2.f32(float %924), !dbg !32
  %926 = insertelement <4 x float> poison, float %925, i64 0, !dbg !34
  %local.reduce209 = fadd float %local.reduce205, %925, !dbg !33, !frisk.local_reduction !1
  %927 = extractelement <4 x float> %819, i64 1, !dbg !32
  %928 = call float @llvm.amdgcn.exp2.f32(float %927), !dbg !32
  %929 = insertelement <4 x float> %926, float %928, i64 1, !dbg !34
  %local.reduce210 = fadd float %local.reduce206, %928, !dbg !33, !frisk.local_reduction !1
  %930 = extractelement <4 x float> %819, i64 2, !dbg !32
  %931 = call float @llvm.amdgcn.exp2.f32(float %930), !dbg !32
  %932 = insertelement <4 x float> %929, float %931, i64 2, !dbg !34
  %local.reduce211 = fadd float %local.reduce207, %931, !dbg !33, !frisk.local_reduction !1
  %933 = extractelement <4 x float> %819, i64 3, !dbg !32
  %934 = call float @llvm.amdgcn.exp2.f32(float %933), !dbg !32
  %935 = insertelement <4 x float> %932, float %934, i64 3, !dbg !34
  %math.result485 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %935) #9, !dbg !34, !frisk.register_lifetimes !1
  %936 = fptrunc <4 x float> %math.result485 to <4 x half>, !dbg !34
  %math.packed486 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %936) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce212 = fadd float %local.reduce208, %934, !dbg !33, !frisk.local_reduction !1
  %937 = extractelement <4 x float> %820, i64 0, !dbg !32
  %938 = call float @llvm.amdgcn.exp2.f32(float %937), !dbg !32
  %939 = insertelement <4 x float> poison, float %938, i64 0, !dbg !34
  %local.reduce213 = fadd float %local.reduce209, %938, !dbg !33, !frisk.local_reduction !1
  %940 = extractelement <4 x float> %820, i64 1, !dbg !32
  %941 = call float @llvm.amdgcn.exp2.f32(float %940), !dbg !32
  %942 = insertelement <4 x float> %939, float %941, i64 1, !dbg !34
  %local.reduce214 = fadd float %local.reduce210, %941, !dbg !33, !frisk.local_reduction !1
  %local.reduce217 = fadd float %local.reduce213, %local.reduce214, !dbg !33, !frisk.local_reduction !1
  %943 = extractelement <4 x float> %820, i64 2, !dbg !32
  %944 = call float @llvm.amdgcn.exp2.f32(float %943), !dbg !32
  %945 = insertelement <4 x float> %942, float %944, i64 2, !dbg !34
  %local.reduce215 = fadd float %local.reduce211, %944, !dbg !33, !frisk.local_reduction !1
  %946 = extractelement <4 x float> %820, i64 3, !dbg !32
  %947 = call float @llvm.amdgcn.exp2.f32(float %946), !dbg !32
  %948 = insertelement <4 x float> %945, float %947, i64 3, !dbg !34
  %math.result487 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %948) #9, !dbg !34, !frisk.register_lifetimes !1
  %949 = fptrunc <4 x float> %math.result487 to <4 x half>, !dbg !34
  %math.packed488 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %949) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce216 = fadd float %local.reduce212, %947, !dbg !33, !frisk.local_reduction !1
  %local.reduce218 = fadd float %local.reduce215, %local.reduce216, !dbg !33, !frisk.local_reduction !1
  %950 = fadd float %local.reduce217, %local.reduce218, !dbg !33, !frisk.local_reduction !1
  %951 = fadd float %950, %837, !dbg !33
  %952 = insertelement <1 x float> poison, float %951, i64 0, !dbg !7
  %953 = insertvalue [2 x <1 x float>] poison, <1 x float> %952, 0, !dbg !7
  %954 = extractelement <4 x float> %822, i64 0, !dbg !32
  %955 = call float @llvm.amdgcn.exp2.f32(float %954), !dbg !32
  %956 = insertelement <4 x float> poison, float %955, i64 0, !dbg !34
  %957 = extractelement <4 x float> %822, i64 1, !dbg !32
  %958 = call float @llvm.amdgcn.exp2.f32(float %957), !dbg !32
  %959 = insertelement <4 x float> %956, float %958, i64 1, !dbg !34
  %960 = extractelement <4 x float> %822, i64 2, !dbg !32
  %961 = call float @llvm.amdgcn.exp2.f32(float %960), !dbg !32
  %962 = insertelement <4 x float> %959, float %961, i64 2, !dbg !34
  %963 = extractelement <4 x float> %822, i64 3, !dbg !32
  %964 = call float @llvm.amdgcn.exp2.f32(float %963), !dbg !32
  %965 = insertelement <4 x float> %962, float %964, i64 3, !dbg !34
  %math.result489 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %965) #9, !dbg !34, !frisk.register_lifetimes !1
  %966 = fptrunc <4 x float> %math.result489 to <4 x half>, !dbg !34
  %math.packed490 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %966) #9, !dbg !32, !frisk.register_lifetimes !1
  %967 = extractelement <4 x float> %823, i64 0, !dbg !32
  %968 = call float @llvm.amdgcn.exp2.f32(float %967), !dbg !32
  %969 = insertelement <4 x float> poison, float %968, i64 0, !dbg !34
  %local.reduce = fadd float %955, %968, !dbg !33, !frisk.local_reduction !1
  %970 = extractelement <4 x float> %823, i64 1, !dbg !32
  %971 = call float @llvm.amdgcn.exp2.f32(float %970), !dbg !32
  %972 = insertelement <4 x float> %969, float %971, i64 1, !dbg !34
  %local.reduce159 = fadd float %958, %971, !dbg !33, !frisk.local_reduction !1
  %973 = extractelement <4 x float> %823, i64 2, !dbg !32
  %974 = call float @llvm.amdgcn.exp2.f32(float %973), !dbg !32
  %975 = insertelement <4 x float> %972, float %974, i64 2, !dbg !34
  %local.reduce160 = fadd float %961, %974, !dbg !33, !frisk.local_reduction !1
  %976 = extractelement <4 x float> %823, i64 3, !dbg !32
  %977 = call float @llvm.amdgcn.exp2.f32(float %976), !dbg !32
  %978 = insertelement <4 x float> %975, float %977, i64 3, !dbg !34
  %math.result491 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %978) #9, !dbg !34, !frisk.register_lifetimes !1
  %979 = fptrunc <4 x float> %math.result491 to <4 x half>, !dbg !34
  %math.packed492 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %979) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce161 = fadd float %964, %977, !dbg !33, !frisk.local_reduction !1
  %980 = extractelement <4 x float> %824, i64 0, !dbg !32
  %981 = call float @llvm.amdgcn.exp2.f32(float %980), !dbg !32
  %982 = insertelement <4 x float> poison, float %981, i64 0, !dbg !34
  %local.reduce162 = fadd float %local.reduce, %981, !dbg !33, !frisk.local_reduction !1
  %983 = extractelement <4 x float> %824, i64 1, !dbg !32
  %984 = call float @llvm.amdgcn.exp2.f32(float %983), !dbg !32
  %985 = insertelement <4 x float> %982, float %984, i64 1, !dbg !34
  %local.reduce163 = fadd float %local.reduce159, %984, !dbg !33, !frisk.local_reduction !1
  %986 = extractelement <4 x float> %824, i64 2, !dbg !32
  %987 = call float @llvm.amdgcn.exp2.f32(float %986), !dbg !32
  %988 = insertelement <4 x float> %985, float %987, i64 2, !dbg !34
  %local.reduce164 = fadd float %local.reduce160, %987, !dbg !33, !frisk.local_reduction !1
  %989 = extractelement <4 x float> %824, i64 3, !dbg !32
  %990 = call float @llvm.amdgcn.exp2.f32(float %989), !dbg !32
  %991 = insertelement <4 x float> %988, float %990, i64 3, !dbg !34
  %math.result493 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %991) #9, !dbg !34, !frisk.register_lifetimes !1
  %992 = fptrunc <4 x float> %math.result493 to <4 x half>, !dbg !34
  %math.packed494 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %992) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce165 = fadd float %local.reduce161, %990, !dbg !33, !frisk.local_reduction !1
  %993 = extractelement <4 x float> %825, i64 0, !dbg !32
  %994 = call float @llvm.amdgcn.exp2.f32(float %993), !dbg !32
  %995 = insertelement <4 x float> poison, float %994, i64 0, !dbg !34
  %local.reduce166 = fadd float %local.reduce162, %994, !dbg !33, !frisk.local_reduction !1
  %996 = extractelement <4 x float> %825, i64 1, !dbg !32
  %997 = call float @llvm.amdgcn.exp2.f32(float %996), !dbg !32
  %998 = insertelement <4 x float> %995, float %997, i64 1, !dbg !34
  %local.reduce167 = fadd float %local.reduce163, %997, !dbg !33, !frisk.local_reduction !1
  %999 = extractelement <4 x float> %825, i64 2, !dbg !32
  %1000 = call float @llvm.amdgcn.exp2.f32(float %999), !dbg !32
  %1001 = insertelement <4 x float> %998, float %1000, i64 2, !dbg !34
  %local.reduce168 = fadd float %local.reduce164, %1000, !dbg !33, !frisk.local_reduction !1
  %1002 = extractelement <4 x float> %825, i64 3, !dbg !32
  %1003 = call float @llvm.amdgcn.exp2.f32(float %1002), !dbg !32
  %1004 = insertelement <4 x float> %1001, float %1003, i64 3, !dbg !34
  %math.result495 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1004) #9, !dbg !34, !frisk.register_lifetimes !1
  %1005 = fptrunc <4 x float> %math.result495 to <4 x half>, !dbg !34
  %math.packed496 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1005) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce169 = fadd float %local.reduce165, %1003, !dbg !33, !frisk.local_reduction !1
  %1006 = extractelement <4 x float> %826, i64 0, !dbg !32
  %1007 = call float @llvm.amdgcn.exp2.f32(float %1006), !dbg !32
  %1008 = insertelement <4 x float> poison, float %1007, i64 0, !dbg !34
  %local.reduce170 = fadd float %local.reduce166, %1007, !dbg !33, !frisk.local_reduction !1
  %1009 = extractelement <4 x float> %826, i64 1, !dbg !32
  %1010 = call float @llvm.amdgcn.exp2.f32(float %1009), !dbg !32
  %1011 = insertelement <4 x float> %1008, float %1010, i64 1, !dbg !34
  %local.reduce171 = fadd float %local.reduce167, %1010, !dbg !33, !frisk.local_reduction !1
  %1012 = extractelement <4 x float> %826, i64 2, !dbg !32
  %1013 = call float @llvm.amdgcn.exp2.f32(float %1012), !dbg !32
  %1014 = insertelement <4 x float> %1011, float %1013, i64 2, !dbg !34
  %local.reduce172 = fadd float %local.reduce168, %1013, !dbg !33, !frisk.local_reduction !1
  %1015 = extractelement <4 x float> %826, i64 3, !dbg !32
  %1016 = call float @llvm.amdgcn.exp2.f32(float %1015), !dbg !32
  %1017 = insertelement <4 x float> %1014, float %1016, i64 3, !dbg !34
  %math.result497 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1017) #9, !dbg !34, !frisk.register_lifetimes !1
  %1018 = fptrunc <4 x float> %math.result497 to <4 x half>, !dbg !34
  %math.packed498 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1018) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce173 = fadd float %local.reduce169, %1016, !dbg !33, !frisk.local_reduction !1
  %1019 = extractelement <4 x float> %827, i64 0, !dbg !32
  %1020 = call float @llvm.amdgcn.exp2.f32(float %1019), !dbg !32
  %1021 = insertelement <4 x float> poison, float %1020, i64 0, !dbg !34
  %local.reduce174 = fadd float %local.reduce170, %1020, !dbg !33, !frisk.local_reduction !1
  %1022 = extractelement <4 x float> %827, i64 1, !dbg !32
  %1023 = call float @llvm.amdgcn.exp2.f32(float %1022), !dbg !32
  %1024 = insertelement <4 x float> %1021, float %1023, i64 1, !dbg !34
  %local.reduce175 = fadd float %local.reduce171, %1023, !dbg !33, !frisk.local_reduction !1
  %1025 = extractelement <4 x float> %827, i64 2, !dbg !32
  %1026 = call float @llvm.amdgcn.exp2.f32(float %1025), !dbg !32
  %1027 = insertelement <4 x float> %1024, float %1026, i64 2, !dbg !34
  %local.reduce176 = fadd float %local.reduce172, %1026, !dbg !33, !frisk.local_reduction !1
  %1028 = extractelement <4 x float> %827, i64 3, !dbg !32
  %1029 = call float @llvm.amdgcn.exp2.f32(float %1028), !dbg !32
  %1030 = insertelement <4 x float> %1027, float %1029, i64 3, !dbg !34
  %math.result499 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1030) #9, !dbg !34, !frisk.register_lifetimes !1
  %1031 = fptrunc <4 x float> %math.result499 to <4 x half>, !dbg !34
  %math.packed500 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1031) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce177 = fadd float %local.reduce173, %1029, !dbg !33, !frisk.local_reduction !1
  %1032 = extractelement <4 x float> %828, i64 0, !dbg !32
  %1033 = call float @llvm.amdgcn.exp2.f32(float %1032), !dbg !32
  %1034 = insertelement <4 x float> poison, float %1033, i64 0, !dbg !34
  %local.reduce178 = fadd float %local.reduce174, %1033, !dbg !33, !frisk.local_reduction !1
  %1035 = extractelement <4 x float> %828, i64 1, !dbg !32
  %1036 = call float @llvm.amdgcn.exp2.f32(float %1035), !dbg !32
  %1037 = insertelement <4 x float> %1034, float %1036, i64 1, !dbg !34
  %local.reduce179 = fadd float %local.reduce175, %1036, !dbg !33, !frisk.local_reduction !1
  %1038 = extractelement <4 x float> %828, i64 2, !dbg !32
  %1039 = call float @llvm.amdgcn.exp2.f32(float %1038), !dbg !32
  %1040 = insertelement <4 x float> %1037, float %1039, i64 2, !dbg !34
  %local.reduce180 = fadd float %local.reduce176, %1039, !dbg !33, !frisk.local_reduction !1
  %1041 = extractelement <4 x float> %828, i64 3, !dbg !32
  %1042 = call float @llvm.amdgcn.exp2.f32(float %1041), !dbg !32
  %1043 = insertelement <4 x float> %1040, float %1042, i64 3, !dbg !34
  %math.result501 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1043) #9, !dbg !34, !frisk.register_lifetimes !1
  %1044 = fptrunc <4 x float> %math.result501 to <4 x half>, !dbg !34
  %math.packed502 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1044) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce181 = fadd float %local.reduce177, %1042, !dbg !33, !frisk.local_reduction !1
  %1045 = extractelement <4 x float> %829, i64 0, !dbg !32
  %1046 = call float @llvm.amdgcn.exp2.f32(float %1045), !dbg !32
  %1047 = insertelement <4 x float> poison, float %1046, i64 0, !dbg !34
  %local.reduce182 = fadd float %local.reduce178, %1046, !dbg !33, !frisk.local_reduction !1
  %1048 = extractelement <4 x float> %829, i64 1, !dbg !32
  %1049 = call float @llvm.amdgcn.exp2.f32(float %1048), !dbg !32
  %1050 = insertelement <4 x float> %1047, float %1049, i64 1, !dbg !34
  %local.reduce183 = fadd float %local.reduce179, %1049, !dbg !33, !frisk.local_reduction !1
  %local.reduce186 = fadd float %local.reduce182, %local.reduce183, !dbg !33, !frisk.local_reduction !1
  %1051 = extractelement <4 x float> %829, i64 2, !dbg !32
  %1052 = call float @llvm.amdgcn.exp2.f32(float %1051), !dbg !32
  %1053 = insertelement <4 x float> %1050, float %1052, i64 2, !dbg !34
  %local.reduce184 = fadd float %local.reduce180, %1052, !dbg !33, !frisk.local_reduction !1
  %1054 = extractelement <4 x float> %829, i64 3, !dbg !32
  %1055 = call float @llvm.amdgcn.exp2.f32(float %1054), !dbg !32
  %1056 = insertelement <4 x float> %1053, float %1055, i64 3, !dbg !34
  %math.result503 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1056) #9, !dbg !34, !frisk.register_lifetimes !1
  %1057 = fptrunc <4 x float> %math.result503 to <4 x half>, !dbg !34
  %math.packed504 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1057) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce185 = fadd float %local.reduce181, %1055, !dbg !33, !frisk.local_reduction !1
  %local.reduce187 = fadd float %local.reduce184, %local.reduce185, !dbg !33, !frisk.local_reduction !1
  %1058 = fadd float %local.reduce186, %local.reduce187, !dbg !33, !frisk.local_reduction !1
  %1059 = fadd float %1058, %845, !dbg !33
  %1060 = insertelement <1 x float> poison, float %1059, i64 0, !dbg !7
  %1061 = insertvalue [2 x <1 x float>] %953, <1 x float> %1060, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread1030 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !14, !range !39, !frisk.register_lifetimes !1
  %address.local1031 = shl nuw nsw i32 %address.thread1030, 3, !dbg !14
  %address.local1032 = and i32 %address.local1031, 120, !dbg !14
  %address.local1033 = shl nuw nsw i32 %address.thread1030, 5, !dbg !14
  %address.local1034 = and i32 %address.local1033, 1536, !dbg !14
  %address.local1035 = or disjoint i32 %address.local1032, %address.local1034, !dbg !14
  %address.local1036 = zext nneg i32 %address.local1035 to i64, !dbg !14
  %address.local1037 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1036, !dbg !14
  %1062 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !14
  %address.local1038 = zext nneg i32 %address.thread1030 to i64, !dbg !8
  %address.local1039 = and i64 %address.local1038, 60, !dbg !14
  %address.local1040 = add nuw nsw i64 %address.local1039, %address.local1038, !dbg !14
  %address.local1041 = shl nuw nsw i64 %address.local1040, 3, !dbg !14
  %address.local1042 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1043 = lshr i64 %address.local1038, 4, !dbg !14
  %address.local1044 = and i64 %address.local1043, 3, !dbg !14
  %address.local1045 = mul nuw nsw i64 %address.local1044, 384, !dbg !14
  %address.local1046 = getelementptr [2 x i8], ptr addrspace(3) %address.local1042, i64 %address.local1045, !dbg !14
  %address.local1047 = lshr i64 %address.local1038, 2, !dbg !17
  %address.local1048 = mul nsw i64 %address.local1047, -32, !dbg !14
  %address.local1049 = getelementptr [2 x i8], ptr addrspace(3) %address.local1046, i64 %address.local1048, !dbg !14
  %address.local1050 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 256, !dbg !14
  %1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !14
  %address.local1051 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1052 = getelementptr [2 x i8], ptr addrspace(3) %address.local1051, i64 %address.local1045, !dbg !14
  %address.local1053 = getelementptr [2 x i8], ptr addrspace(3) %address.local1052, i64 %address.local1048, !dbg !14
  %address.local1054 = getelementptr i8, ptr addrspace(3) %address.local1053, i64 512, !dbg !14
  %1066 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1067 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !14
  %address.local1055 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1056 = getelementptr [2 x i8], ptr addrspace(3) %address.local1055, i64 %address.local1045, !dbg !14
  %address.local1057 = getelementptr [2 x i8], ptr addrspace(3) %address.local1056, i64 %address.local1048, !dbg !14
  %address.local1058 = getelementptr i8, ptr addrspace(3) %address.local1057, i64 768, !dbg !14
  %1068 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1069 = bitcast <4 x i32> %1068 to <8 x half>, !dbg !14
  %address.local1059 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1036, !dbg !14
  %address.local1060 = getelementptr i8, ptr addrspace(3) %address.local1059, i64 4096, !dbg !14
  %1070 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1071 = bitcast <4 x i32> %1070 to <8 x half>, !dbg !14
  %address.local1061 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1062 = getelementptr [2 x i8], ptr addrspace(3) %address.local1061, i64 %address.local1045, !dbg !14
  %address.local1063 = getelementptr [2 x i8], ptr addrspace(3) %address.local1062, i64 %address.local1048, !dbg !14
  %address.local1064 = getelementptr i8, ptr addrspace(3) %address.local1063, i64 4352, !dbg !14
  %1072 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1073 = bitcast <4 x i32> %1072 to <8 x half>, !dbg !14
  %address.local1065 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1066 = getelementptr [2 x i8], ptr addrspace(3) %address.local1065, i64 %address.local1045, !dbg !14
  %address.local1067 = getelementptr [2 x i8], ptr addrspace(3) %address.local1066, i64 %address.local1048, !dbg !14
  %address.local1068 = getelementptr i8, ptr addrspace(3) %address.local1067, i64 4608, !dbg !14
  %1074 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1075 = bitcast <4 x i32> %1074 to <8 x half>, !dbg !14
  %address.local1069 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1070 = getelementptr [2 x i8], ptr addrspace(3) %address.local1069, i64 %address.local1045, !dbg !14
  %address.local1071 = getelementptr [2 x i8], ptr addrspace(3) %address.local1070, i64 %address.local1048, !dbg !14
  %address.local1072 = getelementptr i8, ptr addrspace(3) %address.local1071, i64 4864, !dbg !14
  %1076 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1077 = bitcast <4 x i32> %1076 to <8 x half>, !dbg !14
  %1078 = add nuw nsw i64 %149, %16, !dbg !18
  %1079 = shl i64 %1078, 19, !dbg !18
  %1080 = and i64 %1079, 16252928, !dbg !18
  %1081 = add i64 %1080, %155, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.local1073 = shl nuw nsw i64 %address.local1038, 3, !dbg !8
  %address.local1074 = and i64 %address.local1073, 1536, !dbg !8
  %address.local1075 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1074, !dbg !8
  %address.local1076 = ptrtoint ptr addrspace(3) %address.local1075 to i64, !dbg !8
  %address.local1077 = trunc i64 %address.local1076 to i32, !dbg !8
  %1082 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1077), !dbg !18
  %address.local1078 = shl nuw nsw i64 %address.local1038, 5, !dbg !18
  %address.local1079 = and i64 %address.local1078, 8064, !dbg !18
  %address.local1080 = shl nuw nsw i64 %address.local1038, 3, !dbg !18
  %address.local1081 = and i64 %address.local1080, 24, !dbg !18
  %address.local1082 = or disjoint i64 %address.local1079, %address.local1081, !dbg !18
  %1083 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1084 = or disjoint i64 %1081, 32, !dbg !18
  %1085 = or disjoint i64 %address.local1082, %1084, !dbg !18
  %.tr1841 = trunc i64 %1085 to i32, !dbg !18
  %dma.buffer.offset1724 = shl i32 %.tr1841, 1, !dbg !18
  %1086 = insertelement <2 x i64> poison, i64 %1083, i64 0, !dbg !18
  %1087 = bitcast <2 x i64> %1086 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1725 = shufflevector <4 x i32> %1087, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1726 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1082), !dbg !18
  %1088 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1724, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !18
  %address.local1084 = or disjoint i64 %address.local1038, 256, !dbg !8
  %address.local1085 = shl nuw nsw i64 %address.local1084, 3, !dbg !8
  %address.local1086 = and i64 %address.local1085, 3584, !dbg !8
  %address.local1087 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1086, !dbg !8
  %address.local1088 = ptrtoint ptr addrspace(3) %address.local1087 to i64, !dbg !8
  %address.local1089 = trunc i64 %address.local1088 to i32, !dbg !8
  %1089 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1089), !dbg !18
  %address.local1090 = shl nuw nsw i64 %address.local1084, 5, !dbg !18
  %address.local1091 = and i64 %address.local1090, 16256, !dbg !18
  %address.local1092 = shl nuw nsw i64 %address.local1038, 3, !dbg !18
  %address.local1093 = and i64 %address.local1092, 24, !dbg !18
  %address.local1094 = or disjoint i64 %address.local1091, %address.local1093, !dbg !18
  %1090 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1091 = or disjoint i64 %1081, 32, !dbg !18
  %1092 = or disjoint i64 %address.local1094, %1091, !dbg !18
  %.tr1843 = trunc i64 %1092 to i32, !dbg !18
  %dma.buffer.offset1721 = shl i32 %.tr1843, 1, !dbg !18
  %1093 = insertelement <2 x i64> poison, i64 %1090, i64 0, !dbg !18
  %1094 = bitcast <2 x i64> %1093 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1722 = shufflevector <4 x i32> %1094, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1723 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1089), !dbg !18
  %1095 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1721, <4 x i32> %dma.buffer.rsrc1722) #10, !dbg !18
  %1096 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1097 = extractvalue [2 x <8 x float>] %138, 0, !dbg !35
  %1098 = shufflevector <8 x float> %1097, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1099 = fmul <4 x float> %1098, %834, !dbg !35
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1096, <4 x float> %1099) #9, !dbg !14
  %1101 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1101, <4 x float> %1100) #9, !dbg !14
  %1103 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1103, <4 x float> %1102) #9, !dbg !14
  %1105 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1105, <4 x float> %1104) #9, !dbg !14
  %1107 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1107, <4 x float> %1106) #9, !dbg !14
  %1109 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1109, <4 x float> %1108) #9, !dbg !14
  %1111 = shufflevector <8 x half> %1075, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1111, <4 x float> %1110) #9, !dbg !14
  %1113 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1113, <4 x float> %1112) #9, !dbg !14
  %1115 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1116 = shufflevector <8 x float> %1097, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1117 = fmul <4 x float> %1116, %834, !dbg !35
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1115, <4 x float> %1117) #9, !dbg !14
  %1119 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1119, <4 x float> %1118) #9, !dbg !14
  %1121 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1121, <4 x float> %1120) #9, !dbg !14
  %1123 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1123, <4 x float> %1122) #9, !dbg !14
  %1125 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1125, <4 x float> %1124) #9, !dbg !14
  %1127 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1127, <4 x float> %1126) #9, !dbg !14
  %1129 = shufflevector <8 x half> %1075, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1129, <4 x float> %1128) #9, !dbg !14
  %1131 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1131, <4 x float> %1130) #9, !dbg !14
  %1133 = extractvalue [2 x <8 x float>] %138, 1, !dbg !35
  %1134 = shufflevector <8 x float> %1133, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1135 = fmul <4 x float> %1134, %842, !dbg !35
  %1136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1096, <4 x float> %1135) #9, !dbg !14
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1101, <4 x float> %1136) #9, !dbg !14
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1103, <4 x float> %1137) #9, !dbg !14
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1105, <4 x float> %1138) #9, !dbg !14
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1107, <4 x float> %1139) #9, !dbg !14
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1109, <4 x float> %1140) #9, !dbg !14
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1111, <4 x float> %1141) #9, !dbg !14
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1113, <4 x float> %1142) #9, !dbg !14
  %1144 = shufflevector <8 x float> %1133, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1145 = fmul <4 x float> %1144, %842, !dbg !35
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1115, <4 x float> %1145) #9, !dbg !14
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1119, <4 x float> %1146) #9, !dbg !14
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1121, <4 x float> %1147) #9, !dbg !14
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1123, <4 x float> %1148) #9, !dbg !14
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1125, <4 x float> %1149) #9, !dbg !14
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1127, <4 x float> %1150) #9, !dbg !14
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1129, <4 x float> %1151) #9, !dbg !14
  %1153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1131, <4 x float> %1152) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1154 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1155 = bitcast <4 x i32> %1154 to <8 x half>, !dbg !14
  %1156 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1157 = bitcast <4 x i32> %1156 to <8 x half>, !dbg !14
  %1158 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1159 = bitcast <4 x i32> %1158 to <8 x half>, !dbg !14
  %1160 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1161 = bitcast <4 x i32> %1160 to <8 x half>, !dbg !14
  %1162 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1163 = bitcast <4 x i32> %1162 to <8 x half>, !dbg !14
  %1164 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1165 = bitcast <4 x i32> %1164 to <8 x half>, !dbg !14
  %1166 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1167 = bitcast <4 x i32> %1166 to <8 x half>, !dbg !14
  %1168 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1169 = bitcast <4 x i32> %1168 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1170 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1077), !dbg !18
  %1171 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1172 = or disjoint i64 %1081, 64, !dbg !18
  %1173 = or disjoint i64 %address.local1082, %1172, !dbg !18
  %.tr1845 = trunc i64 %1173 to i32, !dbg !18
  %dma.buffer.offset1718 = shl i32 %.tr1845, 1, !dbg !18
  %1174 = insertelement <2 x i64> poison, i64 %1171, i64 0, !dbg !18
  %1175 = bitcast <2 x i64> %1174 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1719 = shufflevector <4 x i32> %1175, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1720 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1170), !dbg !18
  %1176 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1720, i32 %dma.buffer.offset1718, <4 x i32> %dma.buffer.rsrc1719) #10, !dbg !18
  %1177 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1089), !dbg !18
  %1178 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1179 = or disjoint i64 %1081, 64, !dbg !18
  %1180 = or disjoint i64 %address.local1094, %1179, !dbg !18
  %.tr1847 = trunc i64 %1180 to i32, !dbg !18
  %dma.buffer.offset1715 = shl i32 %.tr1847, 1, !dbg !18
  %1181 = insertelement <2 x i64> poison, i64 %1178, i64 0, !dbg !18
  %1182 = bitcast <2 x i64> %1181 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1716 = shufflevector <4 x i32> %1182, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1717 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1177), !dbg !18
  %1183 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1717, i32 %dma.buffer.offset1715, <4 x i32> %dma.buffer.rsrc1716) #10, !dbg !18
  %1184 = shufflevector <8 x half> %1155, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1185 = extractvalue [2 x <8 x float>] %141, 0, !dbg !35
  %1186 = shufflevector <8 x float> %1185, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1187 = fmul <4 x float> %1186, %834, !dbg !35
  %1188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1184, <4 x float> %1187) #9, !dbg !14
  %1189 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1189, <4 x float> %1188) #9, !dbg !14
  %1191 = shufflevector <8 x half> %1159, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1191, <4 x float> %1190) #9, !dbg !14
  %1193 = shufflevector <8 x half> %1161, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1193, <4 x float> %1192) #9, !dbg !14
  %1195 = shufflevector <8 x half> %1163, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1195, <4 x float> %1194) #9, !dbg !14
  %1197 = shufflevector <8 x half> %1165, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1197, <4 x float> %1196) #9, !dbg !14
  %1199 = shufflevector <8 x half> %1167, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1199, <4 x float> %1198) #9, !dbg !14
  %1201 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1201, <4 x float> %1200) #9, !dbg !14
  %1203 = shufflevector <8 x half> %1155, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1204 = shufflevector <8 x float> %1185, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1205 = fmul <4 x float> %1204, %834, !dbg !35
  %1206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1203, <4 x float> %1205) #9, !dbg !14
  %1207 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1207, <4 x float> %1206) #9, !dbg !14
  %1209 = shufflevector <8 x half> %1159, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1209, <4 x float> %1208) #9, !dbg !14
  %1211 = shufflevector <8 x half> %1161, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1211, <4 x float> %1210) #9, !dbg !14
  %1213 = shufflevector <8 x half> %1163, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1213, <4 x float> %1212) #9, !dbg !14
  %1215 = shufflevector <8 x half> %1165, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1215, <4 x float> %1214) #9, !dbg !14
  %1217 = shufflevector <8 x half> %1167, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1217, <4 x float> %1216) #9, !dbg !14
  %1219 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1219, <4 x float> %1218) #9, !dbg !14
  %1221 = extractvalue [2 x <8 x float>] %141, 1, !dbg !35
  %1222 = shufflevector <8 x float> %1221, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1223 = fmul <4 x float> %1222, %842, !dbg !35
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1184, <4 x float> %1223) #9, !dbg !14
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1189, <4 x float> %1224) #9, !dbg !14
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1191, <4 x float> %1225) #9, !dbg !14
  %1227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1193, <4 x float> %1226) #9, !dbg !14
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1195, <4 x float> %1227) #9, !dbg !14
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1197, <4 x float> %1228) #9, !dbg !14
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1199, <4 x float> %1229) #9, !dbg !14
  %1231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1201, <4 x float> %1230) #9, !dbg !14
  %1232 = shufflevector <8 x float> %1221, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1233 = fmul <4 x float> %1232, %842, !dbg !35
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1203, <4 x float> %1233) #9, !dbg !14
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1207, <4 x float> %1234) #9, !dbg !14
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1209, <4 x float> %1235) #9, !dbg !14
  %1237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1211, <4 x float> %1236) #9, !dbg !14
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1213, <4 x float> %1237) #9, !dbg !14
  %1239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1215, <4 x float> %1238) #9, !dbg !14
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1217, <4 x float> %1239) #9, !dbg !14
  %1241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1219, <4 x float> %1240) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1242 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1243 = bitcast <4 x i32> %1242 to <8 x half>, !dbg !14
  %1244 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1245 = bitcast <4 x i32> %1244 to <8 x half>, !dbg !14
  %1246 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1247 = bitcast <4 x i32> %1246 to <8 x half>, !dbg !14
  %1248 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1249 = bitcast <4 x i32> %1248 to <8 x half>, !dbg !14
  %1250 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1251 = bitcast <4 x i32> %1250 to <8 x half>, !dbg !14
  %1252 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1253 = bitcast <4 x i32> %1252 to <8 x half>, !dbg !14
  %1254 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1255 = bitcast <4 x i32> %1254 to <8 x half>, !dbg !14
  %1256 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1257 = bitcast <4 x i32> %1256 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1258 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1077), !dbg !18
  %1259 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1260 = or disjoint i64 %1081, 96, !dbg !18
  %1261 = or disjoint i64 %address.local1082, %1260, !dbg !18
  %.tr1849 = trunc i64 %1261 to i32, !dbg !18
  %dma.buffer.offset1712 = shl i32 %.tr1849, 1, !dbg !18
  %1262 = insertelement <2 x i64> poison, i64 %1259, i64 0, !dbg !18
  %1263 = bitcast <2 x i64> %1262 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1713 = shufflevector <4 x i32> %1263, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds1714 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1258), !dbg !18
  %1264 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1714, i32 %dma.buffer.offset1712, <4 x i32> %dma.buffer.rsrc1713) #10, !dbg !18
  %1265 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1089), !dbg !18
  %1266 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1267 = or disjoint i64 %1081, 96, !dbg !18
  %1268 = or disjoint i64 %address.local1094, %1267, !dbg !18
  %.tr1851 = trunc i64 %1268 to i32, !dbg !18
  %dma.buffer.offset = shl i32 %.tr1851, 1, !dbg !18
  %1269 = insertelement <2 x i64> poison, i64 %1266, i64 0, !dbg !18
  %1270 = bitcast <2 x i64> %1269 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc = shufflevector <4 x i32> %1270, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1265), !dbg !18
  %1271 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1272 = shufflevector <8 x half> %1243, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1273 = extractvalue [2 x <8 x float>] %142, 0, !dbg !35
  %1274 = shufflevector <8 x float> %1273, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1275 = fmul <4 x float> %1274, %834, !dbg !35
  %1276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1272, <4 x float> %1275) #9, !dbg !14
  %1277 = shufflevector <8 x half> %1245, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1277, <4 x float> %1276) #9, !dbg !14
  %1279 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1279, <4 x float> %1278) #9, !dbg !14
  %1281 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1281, <4 x float> %1280) #9, !dbg !14
  %1283 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1283, <4 x float> %1282) #9, !dbg !14
  %1285 = shufflevector <8 x half> %1253, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1285, <4 x float> %1284) #9, !dbg !14
  %1287 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1287, <4 x float> %1286) #9, !dbg !14
  %1289 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1289, <4 x float> %1288) #9, !dbg !14
  %1291 = shufflevector <8 x half> %1243, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1292 = shufflevector <8 x float> %1273, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1293 = fmul <4 x float> %1292, %834, !dbg !35
  %1294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1291, <4 x float> %1293) #9, !dbg !14
  %1295 = shufflevector <8 x half> %1245, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1295, <4 x float> %1294) #9, !dbg !14
  %1297 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1297, <4 x float> %1296) #9, !dbg !14
  %1299 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1299, <4 x float> %1298) #9, !dbg !14
  %1301 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1301, <4 x float> %1300) #9, !dbg !14
  %1303 = shufflevector <8 x half> %1253, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1303, <4 x float> %1302) #9, !dbg !14
  %1305 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1305, <4 x float> %1304) #9, !dbg !14
  %1307 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1307, <4 x float> %1306) #9, !dbg !14
  %1309 = extractvalue [2 x <8 x float>] %142, 1, !dbg !35
  %1310 = shufflevector <8 x float> %1309, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1311 = fmul <4 x float> %1310, %842, !dbg !35
  %1312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1272, <4 x float> %1311) #9, !dbg !14
  %1313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1277, <4 x float> %1312) #9, !dbg !14
  %1314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1279, <4 x float> %1313) #9, !dbg !14
  %1315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1281, <4 x float> %1314) #9, !dbg !14
  %1316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1283, <4 x float> %1315) #9, !dbg !14
  %1317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1285, <4 x float> %1316) #9, !dbg !14
  %1318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1287, <4 x float> %1317) #9, !dbg !14
  %1319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1289, <4 x float> %1318) #9, !dbg !14
  %1320 = shufflevector <8 x float> %1309, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1321 = fmul <4 x float> %1320, %842, !dbg !35
  %1322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1291, <4 x float> %1321) #9, !dbg !14
  %1323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1295, <4 x float> %1322) #9, !dbg !14
  %1324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1297, <4 x float> %1323) #9, !dbg !14
  %1325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1299, <4 x float> %1324) #9, !dbg !14
  %1326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1301, <4 x float> %1325) #9, !dbg !14
  %1327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1303, <4 x float> %1326) #9, !dbg !14
  %1328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1305, <4 x float> %1327) #9, !dbg !14
  %1329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1307, <4 x float> %1328) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1330 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1331 = bitcast <4 x i32> %1330 to <8 x half>, !dbg !14
  %1332 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1333 = bitcast <4 x i32> %1332 to <8 x half>, !dbg !14
  %1334 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1335 = bitcast <4 x i32> %1334 to <8 x half>, !dbg !14
  %1336 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1337 = bitcast <4 x i32> %1336 to <8 x half>, !dbg !14
  %1338 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1339 = bitcast <4 x i32> %1338 to <8 x half>, !dbg !14
  %1340 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1341 = bitcast <4 x i32> %1340 to <8 x half>, !dbg !14
  %1342 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1343 = bitcast <4 x i32> %1342 to <8 x half>, !dbg !14
  %1344 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1345 = bitcast <4 x i32> %1344 to <8 x half>, !dbg !14
  %1346 = shufflevector <8 x half> %1331, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1347 = extractvalue [2 x <8 x float>] %143, 0, !dbg !35
  %1348 = shufflevector <8 x float> %1347, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1349 = fmul <4 x float> %1348, %834, !dbg !35
  %1350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1346, <4 x float> %1349) #9, !dbg !14
  %1351 = shufflevector <8 x half> %1333, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1351, <4 x float> %1350) #9, !dbg !14
  %1353 = shufflevector <8 x half> %1335, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1353, <4 x float> %1352) #9, !dbg !14
  %1355 = shufflevector <8 x half> %1337, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1355, <4 x float> %1354) #9, !dbg !14
  %1357 = shufflevector <8 x half> %1339, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1357, <4 x float> %1356) #9, !dbg !14
  %1359 = shufflevector <8 x half> %1341, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1359, <4 x float> %1358) #9, !dbg !14
  %1361 = shufflevector <8 x half> %1343, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1361, <4 x float> %1360) #9, !dbg !14
  %1363 = shufflevector <8 x half> %1345, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1363, <4 x float> %1362) #9, !dbg !14
  %1365 = shufflevector <8 x half> %1331, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1366 = shufflevector <8 x float> %1347, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1367 = fmul <4 x float> %1366, %834, !dbg !35
  %1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1365, <4 x float> %1367) #9, !dbg !14
  %1369 = shufflevector <8 x half> %1333, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1369, <4 x float> %1368) #9, !dbg !14
  %1371 = shufflevector <8 x half> %1335, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1371, <4 x float> %1370) #9, !dbg !14
  %1373 = shufflevector <8 x half> %1337, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1373, <4 x float> %1372) #9, !dbg !14
  %1375 = shufflevector <8 x half> %1339, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1375, <4 x float> %1374) #9, !dbg !14
  %1377 = shufflevector <8 x half> %1341, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1377, <4 x float> %1376) #9, !dbg !14
  %1379 = shufflevector <8 x half> %1343, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1379, <4 x float> %1378) #9, !dbg !14
  %1381 = shufflevector <8 x half> %1345, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1381, <4 x float> %1380) #9, !dbg !14
  %1383 = extractvalue [2 x <8 x float>] %143, 1, !dbg !35
  %1384 = shufflevector <8 x float> %1383, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1385 = fmul <4 x float> %1384, %842, !dbg !35
  %1386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1346, <4 x float> %1385) #9, !dbg !14
  %1387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1351, <4 x float> %1386) #9, !dbg !14
  %1388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1353, <4 x float> %1387) #9, !dbg !14
  %1389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1355, <4 x float> %1388) #9, !dbg !14
  %1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1357, <4 x float> %1389) #9, !dbg !14
  %1391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1359, <4 x float> %1390) #9, !dbg !14
  %1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1361, <4 x float> %1391) #9, !dbg !14
  %1393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1363, <4 x float> %1392) #9, !dbg !14
  %1394 = shufflevector <8 x float> %1383, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1395 = fmul <4 x float> %1394, %842, !dbg !35
  %1396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1365, <4 x float> %1395) #9, !dbg !14
  %1397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1369, <4 x float> %1396) #9, !dbg !14
  %1398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1371, <4 x float> %1397) #9, !dbg !14
  %1399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1373, <4 x float> %1398) #9, !dbg !14
  %1400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1375, <4 x float> %1399) #9, !dbg !14
  %1401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1377, <4 x float> %1400) #9, !dbg !14
  %1402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1379, <4 x float> %1401) #9, !dbg !14
  %1403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1381, <4 x float> %1402) #9, !dbg !14
  %1404 = shufflevector <4 x float> %1114, <4 x float> %1132, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1405 = insertvalue [2 x <8 x float>] poison, <8 x float> %1404, 0, !dbg !7
  %1406 = shufflevector <4 x float> %1143, <4 x float> %1153, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1407 = insertvalue [2 x <8 x float>] %1405, <8 x float> %1406, 1, !dbg !7
  %1408 = insertvalue [2 x <1 x float>] poison, <1 x float> %798, 0, !dbg !7
  %1409 = insertvalue [2 x <1 x float>] %1408, <1 x float> %811, 1, !dbg !7
  %1410 = shufflevector <4 x float> %1202, <4 x float> %1220, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1411 = insertvalue [2 x <8 x float>] poison, <8 x float> %1410, 0, !dbg !7
  %1412 = shufflevector <4 x float> %1231, <4 x float> %1241, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1413 = insertvalue [2 x <8 x float>] %1411, <8 x float> %1412, 1, !dbg !7
  %1414 = shufflevector <4 x float> %1290, <4 x float> %1308, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1415 = insertvalue [2 x <8 x float>] poison, <8 x float> %1414, 0, !dbg !7
  %1416 = shufflevector <4 x float> %1319, <4 x float> %1329, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1417 = insertvalue [2 x <8 x float>] %1415, <8 x float> %1416, 1, !dbg !7
  %1418 = shufflevector <4 x float> %1364, <4 x float> %1382, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1419 = insertvalue [2 x <8 x float>] poison, <8 x float> %1418, 0, !dbg !7
  %1420 = shufflevector <4 x float> %1393, <4 x float> %1403, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1421 = insertvalue [2 x <8 x float>] %1419, <8 x float> %1420, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1422 = add nuw nsw i64 %137, 1, !dbg !7
  br label %136, !dbg !7

1423:                                             ; preds = %136
  %lane.sum.field284 = extractvalue [2 x <1 x float>] %139, 1, !dbg !7
  %lane.sum.local285 = extractelement <1 x float> %lane.sum.field284, i64 0, !dbg !7
  %lane.sum.lane.lo286 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane287 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo286), !dbg !7
  %1424 = shl nuw nsw i32 %lane.sum.lane287, 2, !dbg !7
  %1425 = xor i32 %1424, 64, !dbg !7
  %1426 = bitcast float %lane.sum.local285 to i32, !dbg !7
  %lane.sum.shuffle288 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1425, i32 %1426), !dbg !7
  %1427 = bitcast i32 %lane.sum.shuffle288 to float, !dbg !7
  %lane.sum.final289 = fadd float %lane.sum.local285, %1427, !dbg !7
  %1428 = shl nuw nsw i32 %lane.sum.lane287, 2, !dbg !7
  %1429 = xor i32 %1428, 128, !dbg !7
  %1430 = bitcast float %lane.sum.final289 to i32, !dbg !7
  %lane.sum.shuffle290 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1429, i32 %1430), !dbg !7
  %1431 = bitcast i32 %lane.sum.shuffle290 to float, !dbg !7
  %lane.sum.final291 = fadd float %lane.sum.final289, %1431, !dbg !7
  %1432 = insertelement <1 x float> poison, float %lane.sum.final291, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %139, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !7
  %1433 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1434 = xor i32 %1433, 64, !dbg !7
  %1435 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1434, i32 %1435), !dbg !7
  %1436 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1436, !dbg !7
  %1437 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1438 = xor i32 %1437, 128, !dbg !7
  %1439 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle282 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1438, i32 %1439), !dbg !7
  %1440 = bitcast i32 %lane.sum.shuffle282 to float, !dbg !7
  %lane.sum.final283 = fadd float %lane.sum.final, %1440, !dbg !7
  %1441 = insertelement <1 x float> poison, float %lane.sum.final283, i64 0, !dbg !7
  %1442 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1442, 0, !dbg !7
  %1443 = and i32 %31, 32, !dbg !7
  %.not36 = icmp eq i32 %1443, 0, !dbg !7
  %1444 = extractvalue [2 x <8 x float>] %143, 0, !dbg !7
  %1445 = extractelement <8 x float> %1444, i64 0, !dbg !7
  %1446 = extractelement <8 x float> %1444, i64 1, !dbg !7
  %1447 = extractelement <8 x float> %1444, i64 2, !dbg !7
  %1448 = extractelement <8 x float> %1444, i64 3, !dbg !7
  %1449 = select i1 %.not, float %1446, float %1445, !dbg !7
  %1450 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1451 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1450), !dbg !7
  %1452 = add nuw nsw i32 %1451, 64, !dbg !7
  %1453 = xor i32 %1451, 16, !dbg !7
  %1454 = and i32 %1452, 192, !dbg !7
  %1455 = icmp samesign ult i32 %1453, %1454, !dbg !7
  %1456 = select i1 %1455, i32 %1453, i32 %1451, !dbg !7
  %1457 = shl nuw nsw i32 %1456, 2, !dbg !7
  %1458 = bitcast float %1449 to i32, !dbg !7
  %1459 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1458), !dbg !7
  %1460 = bitcast i32 %1459 to float, !dbg !7
  %1461 = select i1 %.not, float %1445, float %1460, !dbg !7
  %1462 = select i1 %.not, float %1460, float %1446, !dbg !7
  %1463 = select i1 %.not, float %1448, float %1447, !dbg !7
  %1464 = bitcast float %1463 to i32, !dbg !7
  %1465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1464), !dbg !7
  %1466 = bitcast i32 %1465 to float, !dbg !7
  %1467 = select i1 %.not, float %1447, float %1466, !dbg !7
  %1468 = select i1 %.not, float %1466, float %1448, !dbg !7
  %1469 = select i1 %.not36, float %1467, float %1461, !dbg !7
  %1470 = xor i32 %1451, 32, !dbg !7
  %1471 = icmp samesign ult i32 %1470, %1454, !dbg !7
  %1472 = select i1 %1471, i32 %1470, i32 %1451, !dbg !7
  %1473 = shl nuw nsw i32 %1472, 2, !dbg !7
  %1474 = bitcast float %1469 to i32, !dbg !7
  %1475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1474), !dbg !7
  %1476 = bitcast i32 %1475 to float, !dbg !7
  %1477 = select i1 %.not36, float %1461, float %1476, !dbg !7
  %1478 = select i1 %.not36, float %1476, float %1467, !dbg !7
  %1479 = select i1 %.not36, float %1468, float %1462, !dbg !7
  %1480 = bitcast float %1479 to i32, !dbg !7
  %1481 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1480), !dbg !7
  %1482 = bitcast i32 %1481 to float, !dbg !7
  %1483 = select i1 %.not36, float %1462, float %1482, !dbg !7
  %1484 = select i1 %.not36, float %1482, float %1468, !dbg !7
  %1485 = extractvalue [2 x <8 x float>] %142, 0, !dbg !7
  %1486 = extractelement <8 x float> %1485, i64 0, !dbg !7
  %1487 = extractelement <8 x float> %1485, i64 1, !dbg !7
  %1488 = extractelement <8 x float> %1485, i64 2, !dbg !7
  %1489 = extractelement <8 x float> %1485, i64 3, !dbg !7
  %1490 = select i1 %.not, float %1487, float %1486, !dbg !7
  %1491 = bitcast float %1490 to i32, !dbg !7
  %1492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1491), !dbg !7
  %1493 = bitcast i32 %1492 to float, !dbg !7
  %1494 = select i1 %.not, float %1486, float %1493, !dbg !7
  %1495 = select i1 %.not, float %1493, float %1487, !dbg !7
  %1496 = select i1 %.not, float %1489, float %1488, !dbg !7
  %1497 = bitcast float %1496 to i32, !dbg !7
  %1498 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1497), !dbg !7
  %1499 = bitcast i32 %1498 to float, !dbg !7
  %1500 = select i1 %.not, float %1488, float %1499, !dbg !7
  %1501 = select i1 %.not, float %1499, float %1489, !dbg !7
  %1502 = select i1 %.not36, float %1500, float %1494, !dbg !7
  %1503 = bitcast float %1502 to i32, !dbg !7
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1503), !dbg !7
  %1505 = bitcast i32 %1504 to float, !dbg !7
  %1506 = select i1 %.not36, float %1494, float %1505, !dbg !7
  %1507 = select i1 %.not36, float %1505, float %1500, !dbg !7
  %1508 = select i1 %.not36, float %1501, float %1495, !dbg !7
  %1509 = bitcast float %1508 to i32, !dbg !7
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1509), !dbg !7
  %1511 = bitcast i32 %1510 to float, !dbg !7
  %1512 = select i1 %.not36, float %1495, float %1511, !dbg !7
  %1513 = select i1 %.not36, float %1511, float %1501, !dbg !7
  %1514 = extractvalue [2 x <8 x float>] %141, 0, !dbg !7
  %1515 = extractelement <8 x float> %1514, i64 0, !dbg !7
  %1516 = extractelement <8 x float> %1514, i64 1, !dbg !7
  %1517 = extractelement <8 x float> %1514, i64 2, !dbg !7
  %1518 = extractelement <8 x float> %1514, i64 3, !dbg !7
  %1519 = select i1 %.not, float %1516, float %1515, !dbg !7
  %1520 = bitcast float %1519 to i32, !dbg !7
  %1521 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1520), !dbg !7
  %1522 = bitcast i32 %1521 to float, !dbg !7
  %1523 = select i1 %.not, float %1515, float %1522, !dbg !7
  %1524 = select i1 %.not, float %1522, float %1516, !dbg !7
  %1525 = select i1 %.not, float %1518, float %1517, !dbg !7
  %1526 = bitcast float %1525 to i32, !dbg !7
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1526), !dbg !7
  %1528 = bitcast i32 %1527 to float, !dbg !7
  %1529 = select i1 %.not, float %1517, float %1528, !dbg !7
  %1530 = select i1 %.not, float %1528, float %1518, !dbg !7
  %1531 = select i1 %.not36, float %1529, float %1523, !dbg !7
  %1532 = bitcast float %1531 to i32, !dbg !7
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1532), !dbg !7
  %1534 = bitcast i32 %1533 to float, !dbg !7
  %1535 = select i1 %.not36, float %1523, float %1534, !dbg !7
  %1536 = select i1 %.not36, float %1534, float %1529, !dbg !7
  %1537 = select i1 %.not36, float %1530, float %1524, !dbg !7
  %1538 = bitcast float %1537 to i32, !dbg !7
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1538), !dbg !7
  %1540 = bitcast i32 %1539 to float, !dbg !7
  %1541 = select i1 %.not36, float %1524, float %1540, !dbg !7
  %1542 = select i1 %.not36, float %1540, float %1530, !dbg !7
  %1543 = extractvalue [2 x <8 x float>] %138, 0, !dbg !7
  %1544 = extractelement <8 x float> %1543, i64 0, !dbg !7
  %1545 = extractelement <8 x float> %1543, i64 1, !dbg !7
  %1546 = extractelement <8 x float> %1543, i64 2, !dbg !7
  %1547 = extractelement <8 x float> %1543, i64 3, !dbg !7
  %1548 = select i1 %.not, float %1545, float %1544, !dbg !7
  %1549 = bitcast float %1548 to i32, !dbg !7
  %1550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1549), !dbg !7
  %1551 = bitcast i32 %1550 to float, !dbg !7
  %1552 = select i1 %.not, float %1544, float %1551, !dbg !7
  %1553 = select i1 %.not, float %1551, float %1545, !dbg !7
  %1554 = select i1 %.not, float %1547, float %1546, !dbg !7
  %1555 = bitcast float %1554 to i32, !dbg !7
  %1556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1555), !dbg !7
  %1557 = bitcast i32 %1556 to float, !dbg !7
  %1558 = select i1 %.not, float %1546, float %1557, !dbg !7
  %1559 = select i1 %.not, float %1557, float %1547, !dbg !7
  %1560 = select i1 %.not36, float %1558, float %1552, !dbg !7
  %1561 = bitcast float %1560 to i32, !dbg !7
  %1562 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1561), !dbg !7
  %1563 = select i1 %.not36, float %1559, float %1553, !dbg !7
  %1564 = bitcast float %1563 to i32, !dbg !7
  %1565 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1564), !dbg !7
  %1566 = shufflevector <1 x float> %1441, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1567 = extractelement <8 x float> %1444, i64 4, !dbg !7
  %1568 = extractelement <8 x float> %1444, i64 5, !dbg !7
  %1569 = extractelement <8 x float> %1444, i64 6, !dbg !7
  %1570 = extractelement <8 x float> %1444, i64 7, !dbg !7
  %1571 = select i1 %.not, float %1568, float %1567, !dbg !7
  %1572 = bitcast float %1571 to i32, !dbg !7
  %1573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1572), !dbg !7
  %1574 = bitcast i32 %1573 to float, !dbg !7
  %1575 = select i1 %.not, float %1567, float %1574, !dbg !7
  %1576 = select i1 %.not, float %1574, float %1568, !dbg !7
  %1577 = select i1 %.not, float %1570, float %1569, !dbg !7
  %1578 = bitcast float %1577 to i32, !dbg !7
  %1579 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1578), !dbg !7
  %1580 = bitcast i32 %1579 to float, !dbg !7
  %1581 = select i1 %.not, float %1569, float %1580, !dbg !7
  %1582 = select i1 %.not, float %1580, float %1570, !dbg !7
  %1583 = select i1 %.not36, float %1581, float %1575, !dbg !7
  %1584 = bitcast float %1583 to i32, !dbg !7
  %1585 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1584), !dbg !7
  %1586 = bitcast i32 %1585 to float, !dbg !7
  %1587 = select i1 %.not36, float %1575, float %1586, !dbg !7
  %1588 = select i1 %.not36, float %1586, float %1581, !dbg !7
  %1589 = select i1 %.not36, float %1582, float %1576, !dbg !7
  %1590 = bitcast float %1589 to i32, !dbg !7
  %1591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1590), !dbg !7
  %1592 = bitcast i32 %1591 to float, !dbg !7
  %1593 = select i1 %.not36, float %1576, float %1592, !dbg !7
  %1594 = select i1 %.not36, float %1592, float %1582, !dbg !7
  %1595 = extractelement <8 x float> %1485, i64 4, !dbg !7
  %1596 = extractelement <8 x float> %1485, i64 5, !dbg !7
  %1597 = extractelement <8 x float> %1485, i64 6, !dbg !7
  %1598 = extractelement <8 x float> %1485, i64 7, !dbg !7
  %1599 = select i1 %.not, float %1596, float %1595, !dbg !7
  %1600 = bitcast float %1599 to i32, !dbg !7
  %1601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1600), !dbg !7
  %1602 = bitcast i32 %1601 to float, !dbg !7
  %1603 = select i1 %.not, float %1595, float %1602, !dbg !7
  %1604 = select i1 %.not, float %1602, float %1596, !dbg !7
  %1605 = select i1 %.not, float %1598, float %1597, !dbg !7
  %1606 = bitcast float %1605 to i32, !dbg !7
  %1607 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1606), !dbg !7
  %1608 = bitcast i32 %1607 to float, !dbg !7
  %1609 = select i1 %.not, float %1597, float %1608, !dbg !7
  %1610 = select i1 %.not, float %1608, float %1598, !dbg !7
  %1611 = select i1 %.not36, float %1609, float %1603, !dbg !7
  %1612 = bitcast float %1611 to i32, !dbg !7
  %1613 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1612), !dbg !7
  %1614 = bitcast i32 %1613 to float, !dbg !7
  %1615 = select i1 %.not36, float %1603, float %1614, !dbg !7
  %1616 = select i1 %.not36, float %1614, float %1609, !dbg !7
  %1617 = select i1 %.not36, float %1610, float %1604, !dbg !7
  %1618 = bitcast float %1617 to i32, !dbg !7
  %1619 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1618), !dbg !7
  %1620 = bitcast i32 %1619 to float, !dbg !7
  %1621 = select i1 %.not36, float %1604, float %1620, !dbg !7
  %1622 = select i1 %.not36, float %1620, float %1610, !dbg !7
  %1623 = extractelement <8 x float> %1514, i64 4, !dbg !7
  %1624 = extractelement <8 x float> %1514, i64 5, !dbg !7
  %1625 = extractelement <8 x float> %1514, i64 6, !dbg !7
  %1626 = extractelement <8 x float> %1514, i64 7, !dbg !7
  %1627 = select i1 %.not, float %1624, float %1623, !dbg !7
  %1628 = bitcast float %1627 to i32, !dbg !7
  %1629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1628), !dbg !7
  %1630 = bitcast i32 %1629 to float, !dbg !7
  %1631 = select i1 %.not, float %1623, float %1630, !dbg !7
  %1632 = select i1 %.not, float %1630, float %1624, !dbg !7
  %1633 = select i1 %.not, float %1626, float %1625, !dbg !7
  %1634 = bitcast float %1633 to i32, !dbg !7
  %1635 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1634), !dbg !7
  %1636 = bitcast i32 %1635 to float, !dbg !7
  %1637 = select i1 %.not, float %1625, float %1636, !dbg !7
  %1638 = select i1 %.not, float %1636, float %1626, !dbg !7
  %1639 = select i1 %.not36, float %1637, float %1631, !dbg !7
  %1640 = bitcast float %1639 to i32, !dbg !7
  %1641 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1640), !dbg !7
  %1642 = bitcast i32 %1641 to float, !dbg !7
  %1643 = select i1 %.not36, float %1631, float %1642, !dbg !7
  %1644 = select i1 %.not36, float %1642, float %1637, !dbg !7
  %1645 = select i1 %.not36, float %1638, float %1632, !dbg !7
  %1646 = bitcast float %1645 to i32, !dbg !7
  %1647 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1646), !dbg !7
  %1648 = bitcast i32 %1647 to float, !dbg !7
  %1649 = select i1 %.not36, float %1632, float %1648, !dbg !7
  %1650 = select i1 %.not36, float %1648, float %1638, !dbg !7
  %1651 = extractelement <8 x float> %1543, i64 4, !dbg !7
  %1652 = extractelement <8 x float> %1543, i64 5, !dbg !7
  %1653 = extractelement <8 x float> %1543, i64 6, !dbg !7
  %1654 = extractelement <8 x float> %1543, i64 7, !dbg !7
  %1655 = select i1 %.not, float %1652, float %1651, !dbg !7
  %1656 = bitcast float %1655 to i32, !dbg !7
  %1657 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1656), !dbg !7
  %1658 = bitcast i32 %1657 to float, !dbg !7
  %1659 = select i1 %.not, float %1651, float %1658, !dbg !7
  %1660 = select i1 %.not, float %1658, float %1652, !dbg !7
  %1661 = select i1 %.not, float %1654, float %1653, !dbg !7
  %1662 = bitcast float %1661 to i32, !dbg !7
  %1663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1662), !dbg !7
  %1664 = bitcast i32 %1663 to float, !dbg !7
  %1665 = select i1 %.not, float %1653, float %1664, !dbg !7
  %1666 = select i1 %.not, float %1664, float %1654, !dbg !7
  %1667 = select i1 %.not36, float %1665, float %1659, !dbg !7
  %1668 = bitcast float %1667 to i32, !dbg !7
  %1669 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1668), !dbg !7
  %1670 = select i1 %.not36, float %1666, float %1660, !dbg !7
  %1671 = bitcast float %1670 to i32, !dbg !7
  %1672 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1671), !dbg !7
  %1673 = extractvalue [2 x <8 x float>] %143, 1, !dbg !7
  %1674 = extractelement <8 x float> %1673, i64 0, !dbg !7
  %1675 = extractelement <8 x float> %1673, i64 1, !dbg !7
  %1676 = extractelement <8 x float> %1673, i64 2, !dbg !7
  %1677 = extractelement <8 x float> %1673, i64 3, !dbg !7
  %1678 = select i1 %.not, float %1675, float %1674, !dbg !7
  %1679 = bitcast float %1678 to i32, !dbg !7
  %1680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1679), !dbg !7
  %1681 = bitcast i32 %1680 to float, !dbg !7
  %1682 = select i1 %.not, float %1674, float %1681, !dbg !7
  %1683 = select i1 %.not, float %1681, float %1675, !dbg !7
  %1684 = select i1 %.not, float %1677, float %1676, !dbg !7
  %1685 = bitcast float %1684 to i32, !dbg !7
  %1686 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1685), !dbg !7
  %1687 = bitcast i32 %1686 to float, !dbg !7
  %1688 = select i1 %.not, float %1676, float %1687, !dbg !7
  %1689 = select i1 %.not, float %1687, float %1677, !dbg !7
  %1690 = select i1 %.not36, float %1688, float %1682, !dbg !7
  %1691 = bitcast float %1690 to i32, !dbg !7
  %1692 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1691), !dbg !7
  %1693 = bitcast i32 %1692 to float, !dbg !7
  %1694 = select i1 %.not36, float %1682, float %1693, !dbg !7
  %1695 = select i1 %.not36, float %1693, float %1688, !dbg !7
  %1696 = select i1 %.not36, float %1689, float %1683, !dbg !7
  %1697 = bitcast float %1696 to i32, !dbg !7
  %1698 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1697), !dbg !7
  %1699 = bitcast i32 %1698 to float, !dbg !7
  %1700 = select i1 %.not36, float %1683, float %1699, !dbg !7
  %1701 = select i1 %.not36, float %1699, float %1689, !dbg !7
  %1702 = extractvalue [2 x <8 x float>] %142, 1, !dbg !7
  %1703 = extractelement <8 x float> %1702, i64 0, !dbg !7
  %1704 = extractelement <8 x float> %1702, i64 1, !dbg !7
  %1705 = extractelement <8 x float> %1702, i64 2, !dbg !7
  %1706 = extractelement <8 x float> %1702, i64 3, !dbg !7
  %1707 = select i1 %.not, float %1704, float %1703, !dbg !7
  %1708 = bitcast float %1707 to i32, !dbg !7
  %1709 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1708), !dbg !7
  %1710 = bitcast i32 %1709 to float, !dbg !7
  %1711 = select i1 %.not, float %1703, float %1710, !dbg !7
  %1712 = select i1 %.not, float %1710, float %1704, !dbg !7
  %1713 = select i1 %.not, float %1706, float %1705, !dbg !7
  %1714 = bitcast float %1713 to i32, !dbg !7
  %1715 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1714), !dbg !7
  %1716 = bitcast i32 %1715 to float, !dbg !7
  %1717 = select i1 %.not, float %1705, float %1716, !dbg !7
  %1718 = select i1 %.not, float %1716, float %1706, !dbg !7
  %1719 = select i1 %.not36, float %1717, float %1711, !dbg !7
  %1720 = bitcast float %1719 to i32, !dbg !7
  %1721 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1720), !dbg !7
  %1722 = bitcast i32 %1721 to float, !dbg !7
  %1723 = select i1 %.not36, float %1711, float %1722, !dbg !7
  %1724 = select i1 %.not36, float %1722, float %1717, !dbg !7
  %1725 = select i1 %.not36, float %1718, float %1712, !dbg !7
  %1726 = bitcast float %1725 to i32, !dbg !7
  %1727 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1726), !dbg !7
  %1728 = bitcast i32 %1727 to float, !dbg !7
  %1729 = select i1 %.not36, float %1712, float %1728, !dbg !7
  %1730 = select i1 %.not36, float %1728, float %1718, !dbg !7
  %1731 = extractvalue [2 x <8 x float>] %141, 1, !dbg !7
  %1732 = extractelement <8 x float> %1731, i64 0, !dbg !7
  %1733 = extractelement <8 x float> %1731, i64 1, !dbg !7
  %1734 = extractelement <8 x float> %1731, i64 2, !dbg !7
  %1735 = extractelement <8 x float> %1731, i64 3, !dbg !7
  %1736 = select i1 %.not, float %1733, float %1732, !dbg !7
  %1737 = bitcast float %1736 to i32, !dbg !7
  %1738 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1737), !dbg !7
  %1739 = bitcast i32 %1738 to float, !dbg !7
  %1740 = select i1 %.not, float %1732, float %1739, !dbg !7
  %1741 = select i1 %.not, float %1739, float %1733, !dbg !7
  %1742 = select i1 %.not, float %1735, float %1734, !dbg !7
  %1743 = bitcast float %1742 to i32, !dbg !7
  %1744 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1743), !dbg !7
  %1745 = bitcast i32 %1744 to float, !dbg !7
  %1746 = select i1 %.not, float %1734, float %1745, !dbg !7
  %1747 = select i1 %.not, float %1745, float %1735, !dbg !7
  %1748 = select i1 %.not36, float %1746, float %1740, !dbg !7
  %1749 = bitcast float %1748 to i32, !dbg !7
  %1750 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1749), !dbg !7
  %1751 = bitcast i32 %1750 to float, !dbg !7
  %1752 = select i1 %.not36, float %1740, float %1751, !dbg !7
  %1753 = select i1 %.not36, float %1751, float %1746, !dbg !7
  %1754 = select i1 %.not36, float %1747, float %1741, !dbg !7
  %1755 = bitcast float %1754 to i32, !dbg !7
  %1756 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1755), !dbg !7
  %1757 = bitcast i32 %1756 to float, !dbg !7
  %1758 = select i1 %.not36, float %1741, float %1757, !dbg !7
  %1759 = select i1 %.not36, float %1757, float %1747, !dbg !7
  %1760 = extractvalue [2 x <8 x float>] %138, 1, !dbg !7
  %1761 = extractelement <8 x float> %1760, i64 0, !dbg !7
  %1762 = extractelement <8 x float> %1760, i64 1, !dbg !7
  %1763 = extractelement <8 x float> %1760, i64 2, !dbg !7
  %1764 = extractelement <8 x float> %1760, i64 3, !dbg !7
  %1765 = select i1 %.not, float %1762, float %1761, !dbg !7
  %1766 = bitcast float %1765 to i32, !dbg !7
  %1767 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1766), !dbg !7
  %1768 = bitcast i32 %1767 to float, !dbg !7
  %1769 = select i1 %.not, float %1761, float %1768, !dbg !7
  %1770 = select i1 %.not, float %1768, float %1762, !dbg !7
  %1771 = select i1 %.not, float %1764, float %1763, !dbg !7
  %1772 = bitcast float %1771 to i32, !dbg !7
  %1773 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1772), !dbg !7
  %1774 = bitcast i32 %1773 to float, !dbg !7
  %1775 = select i1 %.not, float %1763, float %1774, !dbg !7
  %1776 = select i1 %.not, float %1774, float %1764, !dbg !7
  %1777 = select i1 %.not36, float %1775, float %1769, !dbg !7
  %1778 = bitcast float %1777 to i32, !dbg !7
  %1779 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1778), !dbg !7
  %1780 = select i1 %.not36, float %1776, float %1770, !dbg !7
  %1781 = bitcast float %1780 to i32, !dbg !7
  %1782 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1781), !dbg !7
  %1783 = shufflevector <1 x float> %1432, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1784 = extractelement <8 x float> %1673, i64 4, !dbg !7
  %1785 = extractelement <8 x float> %1673, i64 5, !dbg !7
  %1786 = extractelement <8 x float> %1673, i64 6, !dbg !7
  %1787 = extractelement <8 x float> %1673, i64 7, !dbg !7
  %1788 = select i1 %.not, float %1785, float %1784, !dbg !7
  %1789 = bitcast float %1788 to i32, !dbg !7
  %1790 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1789), !dbg !7
  %1791 = bitcast i32 %1790 to float, !dbg !7
  %1792 = select i1 %.not, float %1784, float %1791, !dbg !7
  %1793 = select i1 %.not, float %1791, float %1785, !dbg !7
  %1794 = select i1 %.not, float %1787, float %1786, !dbg !7
  %1795 = bitcast float %1794 to i32, !dbg !7
  %1796 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1795), !dbg !7
  %1797 = bitcast i32 %1796 to float, !dbg !7
  %1798 = select i1 %.not, float %1786, float %1797, !dbg !7
  %1799 = select i1 %.not, float %1797, float %1787, !dbg !7
  %1800 = select i1 %.not36, float %1798, float %1792, !dbg !7
  %1801 = bitcast float %1800 to i32, !dbg !7
  %1802 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1801), !dbg !7
  %1803 = bitcast i32 %1802 to float, !dbg !7
  %1804 = select i1 %.not36, float %1792, float %1803, !dbg !7
  %1805 = select i1 %.not36, float %1803, float %1798, !dbg !7
  %1806 = select i1 %.not36, float %1799, float %1793, !dbg !7
  %1807 = bitcast float %1806 to i32, !dbg !7
  %1808 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1807), !dbg !7
  %1809 = bitcast i32 %1808 to float, !dbg !7
  %1810 = select i1 %.not36, float %1793, float %1809, !dbg !7
  %1811 = select i1 %.not36, float %1809, float %1799, !dbg !7
  %1812 = extractelement <8 x float> %1702, i64 4, !dbg !7
  %1813 = extractelement <8 x float> %1702, i64 5, !dbg !7
  %1814 = extractelement <8 x float> %1702, i64 6, !dbg !7
  %1815 = extractelement <8 x float> %1702, i64 7, !dbg !7
  %1816 = select i1 %.not, float %1813, float %1812, !dbg !7
  %1817 = bitcast float %1816 to i32, !dbg !7
  %1818 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1817), !dbg !7
  %1819 = bitcast i32 %1818 to float, !dbg !7
  %1820 = select i1 %.not, float %1812, float %1819, !dbg !7
  %1821 = select i1 %.not, float %1819, float %1813, !dbg !7
  %1822 = select i1 %.not, float %1815, float %1814, !dbg !7
  %1823 = bitcast float %1822 to i32, !dbg !7
  %1824 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1823), !dbg !7
  %1825 = bitcast i32 %1824 to float, !dbg !7
  %1826 = select i1 %.not, float %1814, float %1825, !dbg !7
  %1827 = select i1 %.not, float %1825, float %1815, !dbg !7
  %1828 = select i1 %.not36, float %1826, float %1820, !dbg !7
  %1829 = bitcast float %1828 to i32, !dbg !7
  %1830 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1829), !dbg !7
  %1831 = bitcast i32 %1830 to float, !dbg !7
  %1832 = select i1 %.not36, float %1820, float %1831, !dbg !7
  %1833 = select i1 %.not36, float %1831, float %1826, !dbg !7
  %1834 = select i1 %.not36, float %1827, float %1821, !dbg !7
  %1835 = bitcast float %1834 to i32, !dbg !7
  %1836 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1835), !dbg !7
  %1837 = bitcast i32 %1836 to float, !dbg !7
  %1838 = select i1 %.not36, float %1821, float %1837, !dbg !7
  %1839 = select i1 %.not36, float %1837, float %1827, !dbg !7
  %1840 = extractelement <8 x float> %1731, i64 4, !dbg !7
  %1841 = extractelement <8 x float> %1731, i64 5, !dbg !7
  %1842 = extractelement <8 x float> %1731, i64 6, !dbg !7
  %1843 = extractelement <8 x float> %1731, i64 7, !dbg !7
  %1844 = select i1 %.not, float %1841, float %1840, !dbg !7
  %1845 = bitcast float %1844 to i32, !dbg !7
  %1846 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1845), !dbg !7
  %1847 = bitcast i32 %1846 to float, !dbg !7
  %1848 = select i1 %.not, float %1840, float %1847, !dbg !7
  %1849 = select i1 %.not, float %1847, float %1841, !dbg !7
  %1850 = select i1 %.not, float %1843, float %1842, !dbg !7
  %1851 = bitcast float %1850 to i32, !dbg !7
  %1852 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1851), !dbg !7
  %1853 = bitcast i32 %1852 to float, !dbg !7
  %1854 = select i1 %.not, float %1842, float %1853, !dbg !7
  %1855 = select i1 %.not, float %1853, float %1843, !dbg !7
  %1856 = select i1 %.not36, float %1854, float %1848, !dbg !7
  %1857 = bitcast float %1856 to i32, !dbg !7
  %1858 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1857), !dbg !7
  %1859 = bitcast i32 %1858 to float, !dbg !7
  %1860 = select i1 %.not36, float %1848, float %1859, !dbg !7
  %1861 = select i1 %.not36, float %1859, float %1854, !dbg !7
  %1862 = select i1 %.not36, float %1855, float %1849, !dbg !7
  %1863 = bitcast float %1862 to i32, !dbg !7
  %1864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1863), !dbg !7
  %1865 = bitcast i32 %1864 to float, !dbg !7
  %1866 = select i1 %.not36, float %1849, float %1865, !dbg !7
  %1867 = select i1 %.not36, float %1865, float %1855, !dbg !7
  %1868 = extractelement <8 x float> %1760, i64 4, !dbg !7
  %1869 = extractelement <8 x float> %1760, i64 5, !dbg !7
  %1870 = extractelement <8 x float> %1760, i64 6, !dbg !7
  %1871 = extractelement <8 x float> %1760, i64 7, !dbg !7
  %1872 = select i1 %.not, float %1869, float %1868, !dbg !7
  %1873 = bitcast float %1872 to i32, !dbg !7
  %1874 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1873), !dbg !7
  %1875 = bitcast i32 %1874 to float, !dbg !7
  %1876 = select i1 %.not, float %1868, float %1875, !dbg !7
  %1877 = select i1 %.not, float %1875, float %1869, !dbg !7
  %1878 = select i1 %.not, float %1871, float %1870, !dbg !7
  %1879 = bitcast float %1878 to i32, !dbg !7
  %1880 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1879), !dbg !7
  %1881 = bitcast i32 %1880 to float, !dbg !7
  %1882 = select i1 %.not, float %1870, float %1881, !dbg !7
  %1883 = select i1 %.not, float %1881, float %1871, !dbg !7
  %1884 = select i1 %.not36, float %1882, float %1876, !dbg !7
  %1885 = bitcast float %1884 to i32, !dbg !7
  %1886 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1885), !dbg !7
  %1887 = select i1 %.not36, float %1883, float %1877, !dbg !7
  %1888 = bitcast float %1887 to i32, !dbg !7
  %1889 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1888), !dbg !7
  %1890 = bitcast i32 %1886 to float, !dbg !7
  %1891 = select i1 %.not36, float %1876, float %1890, !dbg !7
  %1892 = insertelement <4 x float> poison, float %1891, i64 0, !dbg !19
  %1893 = bitcast i32 %1889 to float, !dbg !7
  %1894 = select i1 %.not36, float %1877, float %1893, !dbg !7
  %1895 = insertelement <4 x float> %1892, float %1894, i64 1, !dbg !19
  %1896 = select i1 %.not36, float %1890, float %1882, !dbg !7
  %1897 = insertelement <4 x float> %1895, float %1896, i64 2, !dbg !19
  %1898 = select i1 %.not36, float %1893, float %1883, !dbg !7
  %1899 = insertelement <4 x float> %1897, float %1898, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1783, !dbg !19
  %normalized = fmul <4 x float> %1899, %row.reciprocal, !dbg !19
  %1900 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1901 = extractelement <4 x half> %1900, i64 3, !dbg !20
  %1902 = extractelement <4 x half> %1900, i64 2, !dbg !20
  %1903 = extractelement <4 x half> %1900, i64 1, !dbg !20
  %1904 = extractelement <4 x half> %1900, i64 0, !dbg !20
  %1905 = bitcast i32 %1779 to float, !dbg !7
  %1906 = select i1 %.not36, float %1769, float %1905, !dbg !7
  %1907 = insertelement <4 x float> poison, float %1906, i64 0, !dbg !19
  %1908 = bitcast i32 %1782 to float, !dbg !7
  %1909 = select i1 %.not36, float %1770, float %1908, !dbg !7
  %1910 = insertelement <4 x float> %1907, float %1909, i64 1, !dbg !19
  %1911 = select i1 %.not36, float %1905, float %1775, !dbg !7
  %1912 = insertelement <4 x float> %1910, float %1911, i64 2, !dbg !19
  %1913 = select i1 %.not36, float %1908, float %1776, !dbg !7
  %1914 = insertelement <4 x float> %1912, float %1913, i64 3, !dbg !19
  %normalized104 = fmul <4 x float> %1914, %row.reciprocal, !dbg !19
  %1915 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %1916 = extractelement <4 x half> %1915, i64 3, !dbg !20
  %1917 = extractelement <4 x half> %1915, i64 2, !dbg !20
  %1918 = extractelement <4 x half> %1915, i64 1, !dbg !20
  %1919 = extractelement <4 x half> %1915, i64 0, !dbg !20
  %1920 = bitcast i32 %1669 to float, !dbg !7
  %1921 = select i1 %.not36, float %1659, float %1920, !dbg !7
  %1922 = insertelement <4 x float> poison, float %1921, i64 0, !dbg !19
  %1923 = bitcast i32 %1672 to float, !dbg !7
  %1924 = select i1 %.not36, float %1660, float %1923, !dbg !7
  %1925 = insertelement <4 x float> %1922, float %1924, i64 1, !dbg !19
  %1926 = select i1 %.not36, float %1920, float %1665, !dbg !7
  %1927 = insertelement <4 x float> %1925, float %1926, i64 2, !dbg !19
  %1928 = select i1 %.not36, float %1923, float %1666, !dbg !7
  %1929 = insertelement <4 x float> %1927, float %1928, i64 3, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1566, !dbg !19
  %normalized106 = fmul <4 x float> %1929, %row.reciprocal105, !dbg !19
  %1930 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %1931 = extractelement <4 x half> %1930, i64 3, !dbg !20
  %1932 = extractelement <4 x half> %1930, i64 2, !dbg !20
  %1933 = extractelement <4 x half> %1930, i64 1, !dbg !20
  %1934 = extractelement <4 x half> %1930, i64 0, !dbg !20
  %1935 = bitcast i32 %1562 to float, !dbg !7
  %1936 = select i1 %.not36, float %1552, float %1935, !dbg !7
  %1937 = insertelement <4 x float> poison, float %1936, i64 0, !dbg !19
  %1938 = bitcast i32 %1565 to float, !dbg !7
  %1939 = select i1 %.not36, float %1553, float %1938, !dbg !7
  %1940 = insertelement <4 x float> %1937, float %1939, i64 1, !dbg !19
  %1941 = select i1 %.not36, float %1935, float %1558, !dbg !7
  %1942 = insertelement <4 x float> %1940, float %1941, i64 2, !dbg !19
  %1943 = select i1 %.not36, float %1938, float %1559, !dbg !7
  %1944 = insertelement <4 x float> %1942, float %1943, i64 3, !dbg !19
  %normalized107 = fmul <4 x float> %1944, %row.reciprocal105, !dbg !19
  %1945 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %1946 = extractelement <4 x half> %1945, i64 3, !dbg !20
  %1947 = extractelement <4 x half> %1945, i64 2, !dbg !20
  %1948 = extractelement <4 x half> %1945, i64 1, !dbg !20
  %1949 = extractelement <4 x half> %1945, i64 0, !dbg !20
  %1950 = add nsw i64 %16, %22, !dbg !21
  %1951 = shl i64 %1950, 19, !dbg !21
  %1952 = and i64 %1951, 16252928, !dbg !21
  %address.thread1100 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1101 = zext nneg i32 %address.thread1100 to i64, !dbg !8
  %address.local1102 = add i64 %25, %address.local1101, !dbg !21
  %1953 = lshr i64 %address.local1101, 2, !dbg !21
  %address.local1104 = and i64 %1953, 48, !dbg !21
  %address.local1105 = add i64 %address.local1102, %address.local1104, !dbg !21
  %address.local1106 = and i64 %address.local1101, 240, !dbg !21
  %address.local1107 = sub i64 %address.local1105, %address.local1106, !dbg !21
  %address.local1108 = add i64 %address.local1107, %26, !dbg !21
  %address.local1109 = shl nuw nsw i64 %address.local1108, 7, !dbg !21
  %address.local1110 = add nuw nsw i64 %1952, %address.local1109, !dbg !21
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1112 = lshr i64 %address.local1101, 2, !dbg !11
  %address.local1113 = and i64 %address.local1112, 12, !dbg !11
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1113, !dbg !21
  store half %1949, ptr addrspace(1) %address.local1114, align 2, !dbg !21
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1116 = or disjoint i64 %address.local1113, 1, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1116, !dbg !21
  store half %1948, ptr addrspace(1) %address.local1117, align 2, !dbg !21
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1119 = or disjoint i64 %address.local1113, 2, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1118, i64 %address.local1119, !dbg !21
  store half %1947, ptr addrspace(1) %address.local1120, align 2, !dbg !21
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1122 = or disjoint i64 %address.local1113, 3, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1122, !dbg !21
  store half %1946, ptr addrspace(1) %address.local1123, align 2, !dbg !21
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1125 = or disjoint i64 %address.local1113, 16, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1124, i64 %address.local1125, !dbg !21
  store half %1934, ptr addrspace(1) %address.local1126, align 2, !dbg !21
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1128 = or disjoint i64 %address.local1113, 17, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1128, !dbg !21
  store half %1933, ptr addrspace(1) %address.local1129, align 2, !dbg !21
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1131 = or disjoint i64 %address.local1113, 18, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1130, i64 %address.local1131, !dbg !21
  store half %1932, ptr addrspace(1) %address.local1132, align 2, !dbg !21
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1134 = or disjoint i64 %address.local1113, 19, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1134, !dbg !21
  store half %1931, ptr addrspace(1) %address.local1135, align 2, !dbg !21
  %address.local1136 = shl i64 %address.local1108, 7, !dbg !21
  %address.local1137 = add i64 %address.local1136, 8192, !dbg !21
  %address.local1138 = add nuw nsw i64 %1952, %address.local1137, !dbg !21
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1113, !dbg !21
  store half %1919, ptr addrspace(1) %address.local1140, align 2, !dbg !21
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1116, !dbg !21
  store half %1918, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1119, !dbg !21
  store half %1917, ptr addrspace(1) %address.local1144, align 2, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1122, !dbg !21
  store half %1916, ptr addrspace(1) %address.local1146, align 2, !dbg !21
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1125, !dbg !21
  store half %1904, ptr addrspace(1) %address.local1148, align 2, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1128, !dbg !21
  store half %1903, ptr addrspace(1) %address.local1150, align 2, !dbg !21
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1131, !dbg !21
  store half %1902, ptr addrspace(1) %address.local1152, align 2, !dbg !21
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1134, !dbg !21
  store half %1901, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  %1954 = insertelement <4 x float> poison, float %1860, i64 0, !dbg !19
  %1955 = insertelement <4 x float> %1954, float %1866, i64 1, !dbg !19
  %1956 = insertelement <4 x float> %1955, float %1861, i64 2, !dbg !19
  %1957 = insertelement <4 x float> %1956, float %1867, i64 3, !dbg !19
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1783, !dbg !19
  %normalized109 = fmul <4 x float> %1957, %row.reciprocal108, !dbg !19
  %1958 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %1959 = extractelement <4 x half> %1958, i64 3, !dbg !20
  %1960 = extractelement <4 x half> %1958, i64 2, !dbg !20
  %1961 = extractelement <4 x half> %1958, i64 1, !dbg !20
  %1962 = extractelement <4 x half> %1958, i64 0, !dbg !20
  %1963 = insertelement <4 x float> poison, float %1752, i64 0, !dbg !19
  %1964 = insertelement <4 x float> %1963, float %1758, i64 1, !dbg !19
  %1965 = insertelement <4 x float> %1964, float %1753, i64 2, !dbg !19
  %1966 = insertelement <4 x float> %1965, float %1759, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %1966, %row.reciprocal108, !dbg !19
  %1967 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %1968 = extractelement <4 x half> %1967, i64 3, !dbg !20
  %1969 = extractelement <4 x half> %1967, i64 2, !dbg !20
  %1970 = extractelement <4 x half> %1967, i64 1, !dbg !20
  %1971 = extractelement <4 x half> %1967, i64 0, !dbg !20
  %1972 = insertelement <4 x float> poison, float %1643, i64 0, !dbg !19
  %1973 = insertelement <4 x float> %1972, float %1649, i64 1, !dbg !19
  %1974 = insertelement <4 x float> %1973, float %1644, i64 2, !dbg !19
  %1975 = insertelement <4 x float> %1974, float %1650, i64 3, !dbg !19
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1566, !dbg !19
  %normalized112 = fmul <4 x float> %1975, %row.reciprocal111, !dbg !19
  %1976 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %1977 = extractelement <4 x half> %1976, i64 3, !dbg !20
  %1978 = extractelement <4 x half> %1976, i64 2, !dbg !20
  %1979 = extractelement <4 x half> %1976, i64 1, !dbg !20
  %1980 = extractelement <4 x half> %1976, i64 0, !dbg !20
  %1981 = insertelement <4 x float> poison, float %1535, i64 0, !dbg !19
  %1982 = insertelement <4 x float> %1981, float %1541, i64 1, !dbg !19
  %1983 = insertelement <4 x float> %1982, float %1536, i64 2, !dbg !19
  %1984 = insertelement <4 x float> %1983, float %1542, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %1984, %row.reciprocal111, !dbg !19
  %1985 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %1986 = extractelement <4 x half> %1985, i64 3, !dbg !20
  %1987 = extractelement <4 x half> %1985, i64 2, !dbg !20
  %1988 = extractelement <4 x half> %1985, i64 1, !dbg !20
  %1989 = extractelement <4 x half> %1985, i64 0, !dbg !20
  %1990 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1155 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1156 = zext nneg i32 %address.thread1155 to i64, !dbg !8
  %address.local1157 = add i64 %25, %address.local1156, !dbg !21
  %1991 = lshr i64 %address.local1156, 2, !dbg !21
  %address.local1159 = and i64 %1991, 48, !dbg !21
  %address.local1160 = add i64 %address.local1157, %address.local1159, !dbg !21
  %address.local1161 = and i64 %address.local1156, 240, !dbg !21
  %address.local1162 = sub i64 %address.local1160, %address.local1161, !dbg !21
  %address.local1163 = add i64 %address.local1162, %26, !dbg !21
  %address.local1164 = shl nuw nsw i64 %address.local1163, 7, !dbg !21
  %address.local1165 = add nuw nsw i64 %1990, %address.local1164, !dbg !21
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1167 = lshr i64 %address.local1156, 2, !dbg !11
  %address.local1168 = and i64 %address.local1167, 12, !dbg !11
  %address.local1169 = or disjoint i64 %address.local1168, 32, !dbg !21
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1166, i64 %address.local1169, !dbg !21
  store half %1989, ptr addrspace(1) %address.local1170, align 2, !dbg !21
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1172 = or disjoint i64 %address.local1168, 33, !dbg !21
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1172, !dbg !21
  store half %1988, ptr addrspace(1) %address.local1173, align 2, !dbg !21
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1175 = or disjoint i64 %address.local1168, 34, !dbg !21
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1174, i64 %address.local1175, !dbg !21
  store half %1987, ptr addrspace(1) %address.local1176, align 2, !dbg !21
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1178 = or disjoint i64 %address.local1168, 35, !dbg !21
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1178, !dbg !21
  store half %1986, ptr addrspace(1) %address.local1179, align 2, !dbg !21
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1181 = or disjoint i64 %address.local1168, 48, !dbg !21
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1180, i64 %address.local1181, !dbg !21
  store half %1980, ptr addrspace(1) %address.local1182, align 2, !dbg !21
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1184 = or disjoint i64 %address.local1168, 49, !dbg !21
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1184, !dbg !21
  store half %1979, ptr addrspace(1) %address.local1185, align 2, !dbg !21
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1187 = or disjoint i64 %address.local1168, 50, !dbg !21
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1186, i64 %address.local1187, !dbg !21
  store half %1978, ptr addrspace(1) %address.local1188, align 2, !dbg !21
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1190 = or i64 %address.local1167, 51, !dbg !21
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1190, !dbg !21
  store half %1977, ptr addrspace(1) %address.local1191, align 2, !dbg !21
  %address.local1192 = shl i64 %address.local1163, 7, !dbg !21
  %address.local1193 = add i64 %address.local1192, 8192, !dbg !21
  %address.local1194 = add nuw nsw i64 %1990, %address.local1193, !dbg !21
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1169, !dbg !21
  store half %1971, ptr addrspace(1) %address.local1196, align 2, !dbg !21
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1197, i64 %address.local1172, !dbg !21
  store half %1970, ptr addrspace(1) %address.local1198, align 2, !dbg !21
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1175, !dbg !21
  store half %1969, ptr addrspace(1) %address.local1200, align 2, !dbg !21
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1201, i64 %address.local1178, !dbg !21
  store half %1968, ptr addrspace(1) %address.local1202, align 2, !dbg !21
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1203, i64 %address.local1181, !dbg !21
  store half %1962, ptr addrspace(1) %address.local1204, align 2, !dbg !21
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1184, !dbg !21
  store half %1961, ptr addrspace(1) %address.local1206, align 2, !dbg !21
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1207, i64 %address.local1187, !dbg !21
  store half %1960, ptr addrspace(1) %address.local1208, align 2, !dbg !21
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1209, i64 %address.local1190, !dbg !21
  store half %1959, ptr addrspace(1) %address.local1210, align 2, !dbg !21
  %1992 = insertelement <4 x float> poison, float %1832, i64 0, !dbg !19
  %1993 = insertelement <4 x float> %1992, float %1838, i64 1, !dbg !19
  %1994 = insertelement <4 x float> %1993, float %1833, i64 2, !dbg !19
  %1995 = insertelement <4 x float> %1994, float %1839, i64 3, !dbg !19
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1783, !dbg !19
  %normalized115 = fmul <4 x float> %1995, %row.reciprocal114, !dbg !19
  %1996 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %1997 = extractelement <4 x half> %1996, i64 3, !dbg !20
  %1998 = extractelement <4 x half> %1996, i64 2, !dbg !20
  %1999 = extractelement <4 x half> %1996, i64 1, !dbg !20
  %2000 = extractelement <4 x half> %1996, i64 0, !dbg !20
  %2001 = insertelement <4 x float> poison, float %1723, i64 0, !dbg !19
  %2002 = insertelement <4 x float> %2001, float %1729, i64 1, !dbg !19
  %2003 = insertelement <4 x float> %2002, float %1724, i64 2, !dbg !19
  %2004 = insertelement <4 x float> %2003, float %1730, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %2004, %row.reciprocal114, !dbg !19
  %2005 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %2006 = extractelement <4 x half> %2005, i64 3, !dbg !20
  %2007 = extractelement <4 x half> %2005, i64 2, !dbg !20
  %2008 = extractelement <4 x half> %2005, i64 1, !dbg !20
  %2009 = extractelement <4 x half> %2005, i64 0, !dbg !20
  %2010 = insertelement <4 x float> poison, float %1615, i64 0, !dbg !19
  %2011 = insertelement <4 x float> %2010, float %1621, i64 1, !dbg !19
  %2012 = insertelement <4 x float> %2011, float %1616, i64 2, !dbg !19
  %2013 = insertelement <4 x float> %2012, float %1622, i64 3, !dbg !19
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1566, !dbg !19
  %normalized118 = fmul <4 x float> %2013, %row.reciprocal117, !dbg !19
  %2014 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %2015 = extractelement <4 x half> %2014, i64 3, !dbg !20
  %2016 = extractelement <4 x half> %2014, i64 2, !dbg !20
  %2017 = extractelement <4 x half> %2014, i64 1, !dbg !20
  %2018 = extractelement <4 x half> %2014, i64 0, !dbg !20
  %2019 = insertelement <4 x float> poison, float %1506, i64 0, !dbg !19
  %2020 = insertelement <4 x float> %2019, float %1512, i64 1, !dbg !19
  %2021 = insertelement <4 x float> %2020, float %1507, i64 2, !dbg !19
  %2022 = insertelement <4 x float> %2021, float %1513, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %2022, %row.reciprocal117, !dbg !19
  %2023 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %2024 = extractelement <4 x half> %2023, i64 3, !dbg !20
  %2025 = extractelement <4 x half> %2023, i64 2, !dbg !20
  %2026 = extractelement <4 x half> %2023, i64 1, !dbg !20
  %2027 = extractelement <4 x half> %2023, i64 0, !dbg !20
  %2028 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1211 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1212 = zext nneg i32 %address.thread1211 to i64, !dbg !8
  %address.local1213 = add i64 %25, %address.local1212, !dbg !21
  %2029 = lshr i64 %address.local1212, 2, !dbg !21
  %address.local1215 = and i64 %2029, 48, !dbg !21
  %address.local1216 = add i64 %address.local1213, %address.local1215, !dbg !21
  %address.local1217 = and i64 %address.local1212, 240, !dbg !21
  %address.local1218 = sub i64 %address.local1216, %address.local1217, !dbg !21
  %address.local1219 = add i64 %address.local1218, %26, !dbg !21
  %address.local1220 = shl nuw nsw i64 %address.local1219, 7, !dbg !21
  %address.local1221 = add nuw nsw i64 %2028, %address.local1220, !dbg !21
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1223 = lshr i64 %address.local1212, 2, !dbg !11
  %address.local1224 = and i64 %address.local1223, 12, !dbg !11
  %address.local1225 = or disjoint i64 %address.local1224, 64, !dbg !21
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1222, i64 %address.local1225, !dbg !21
  store half %2027, ptr addrspace(1) %address.local1226, align 2, !dbg !21
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1228 = or disjoint i64 %address.local1224, 65, !dbg !21
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1227, i64 %address.local1228, !dbg !21
  store half %2026, ptr addrspace(1) %address.local1229, align 2, !dbg !21
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1231 = or disjoint i64 %address.local1224, 66, !dbg !21
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1230, i64 %address.local1231, !dbg !21
  store half %2025, ptr addrspace(1) %address.local1232, align 2, !dbg !21
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1234 = or disjoint i64 %address.local1224, 67, !dbg !21
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1233, i64 %address.local1234, !dbg !21
  store half %2024, ptr addrspace(1) %address.local1235, align 2, !dbg !21
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1237 = or disjoint i64 %address.local1224, 80, !dbg !21
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1236, i64 %address.local1237, !dbg !21
  store half %2018, ptr addrspace(1) %address.local1238, align 2, !dbg !21
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1240 = or disjoint i64 %address.local1224, 81, !dbg !21
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1239, i64 %address.local1240, !dbg !21
  store half %2017, ptr addrspace(1) %address.local1241, align 2, !dbg !21
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1243 = or disjoint i64 %address.local1224, 82, !dbg !21
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1242, i64 %address.local1243, !dbg !21
  store half %2016, ptr addrspace(1) %address.local1244, align 2, !dbg !21
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1246 = or disjoint i64 %address.local1224, 83, !dbg !21
  %address.local1247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1245, i64 %address.local1246, !dbg !21
  store half %2015, ptr addrspace(1) %address.local1247, align 2, !dbg !21
  %address.local1248 = shl i64 %address.local1219, 7, !dbg !21
  %address.local1249 = add i64 %address.local1248, 8192, !dbg !21
  %address.local1250 = add nuw nsw i64 %2028, %address.local1249, !dbg !21
  %address.local1251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1251, i64 %address.local1225, !dbg !21
  store half %2009, ptr addrspace(1) %address.local1252, align 2, !dbg !21
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1253, i64 %address.local1228, !dbg !21
  store half %2008, ptr addrspace(1) %address.local1254, align 2, !dbg !21
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1255, i64 %address.local1231, !dbg !21
  store half %2007, ptr addrspace(1) %address.local1256, align 2, !dbg !21
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1234, !dbg !21
  store half %2006, ptr addrspace(1) %address.local1258, align 2, !dbg !21
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1259, i64 %address.local1237, !dbg !21
  store half %2000, ptr addrspace(1) %address.local1260, align 2, !dbg !21
  %address.local1261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1261, i64 %address.local1240, !dbg !21
  store half %1999, ptr addrspace(1) %address.local1262, align 2, !dbg !21
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1243, !dbg !21
  store half %1998, ptr addrspace(1) %address.local1264, align 2, !dbg !21
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1265, i64 %address.local1246, !dbg !21
  store half %1997, ptr addrspace(1) %address.local1266, align 2, !dbg !21
  %2030 = insertelement <4 x float> poison, float %1804, i64 0, !dbg !19
  %2031 = insertelement <4 x float> %2030, float %1810, i64 1, !dbg !19
  %2032 = insertelement <4 x float> %2031, float %1805, i64 2, !dbg !19
  %2033 = insertelement <4 x float> %2032, float %1811, i64 3, !dbg !19
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1783, !dbg !19
  %normalized121 = fmul <4 x float> %2033, %row.reciprocal120, !dbg !19
  %2034 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %2035 = extractelement <4 x half> %2034, i64 3, !dbg !20
  %2036 = extractelement <4 x half> %2034, i64 2, !dbg !20
  %2037 = extractelement <4 x half> %2034, i64 1, !dbg !20
  %2038 = extractelement <4 x half> %2034, i64 0, !dbg !20
  %2039 = insertelement <4 x float> poison, float %1694, i64 0, !dbg !19
  %2040 = insertelement <4 x float> %2039, float %1700, i64 1, !dbg !19
  %2041 = insertelement <4 x float> %2040, float %1695, i64 2, !dbg !19
  %2042 = insertelement <4 x float> %2041, float %1701, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %2042, %row.reciprocal120, !dbg !19
  %2043 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %2044 = extractelement <4 x half> %2043, i64 3, !dbg !20
  %2045 = extractelement <4 x half> %2043, i64 2, !dbg !20
  %2046 = extractelement <4 x half> %2043, i64 1, !dbg !20
  %2047 = extractelement <4 x half> %2043, i64 0, !dbg !20
  %2048 = insertelement <4 x float> poison, float %1587, i64 0, !dbg !19
  %2049 = insertelement <4 x float> %2048, float %1593, i64 1, !dbg !19
  %2050 = insertelement <4 x float> %2049, float %1588, i64 2, !dbg !19
  %2051 = insertelement <4 x float> %2050, float %1594, i64 3, !dbg !19
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1566, !dbg !19
  %normalized124 = fmul <4 x float> %2051, %row.reciprocal123, !dbg !19
  %2052 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %2053 = extractelement <4 x half> %2052, i64 3, !dbg !20
  %2054 = extractelement <4 x half> %2052, i64 2, !dbg !20
  %2055 = extractelement <4 x half> %2052, i64 1, !dbg !20
  %2056 = extractelement <4 x half> %2052, i64 0, !dbg !20
  %2057 = insertelement <4 x float> poison, float %1477, i64 0, !dbg !19
  %2058 = insertelement <4 x float> %2057, float %1483, i64 1, !dbg !19
  %2059 = insertelement <4 x float> %2058, float %1478, i64 2, !dbg !19
  %2060 = insertelement <4 x float> %2059, float %1484, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %2060, %row.reciprocal123, !dbg !19
  %2061 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %2062 = extractelement <4 x half> %2061, i64 3, !dbg !20
  %2063 = extractelement <4 x half> %2061, i64 2, !dbg !20
  %2064 = extractelement <4 x half> %2061, i64 1, !dbg !20
  %2065 = extractelement <4 x half> %2061, i64 0, !dbg !20
  %2066 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1267 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1268 = zext nneg i32 %address.thread1267 to i64, !dbg !8
  %address.local1269 = add i64 %25, %address.local1268, !dbg !21
  %2067 = lshr i64 %address.local1268, 2, !dbg !21
  %address.local1271 = and i64 %2067, 48, !dbg !21
  %address.local1272 = add i64 %address.local1269, %address.local1271, !dbg !21
  %address.local1273 = and i64 %address.local1268, 240, !dbg !21
  %address.local1274 = sub i64 %address.local1272, %address.local1273, !dbg !21
  %address.local1275 = add i64 %address.local1274, %26, !dbg !21
  %address.local1276 = shl nuw nsw i64 %address.local1275, 7, !dbg !21
  %address.local1277 = add nuw nsw i64 %2066, %address.local1276, !dbg !21
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1279 = lshr i64 %address.local1268, 2, !dbg !11
  %address.local1280 = and i64 %address.local1279, 12, !dbg !11
  %address.local1281 = or disjoint i64 %address.local1280, 96, !dbg !21
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1278, i64 %address.local1281, !dbg !21
  store half %2065, ptr addrspace(1) %address.local1282, align 2, !dbg !21
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1284 = or disjoint i64 %address.local1280, 97, !dbg !21
  %address.local1285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1283, i64 %address.local1284, !dbg !21
  store half %2064, ptr addrspace(1) %address.local1285, align 2, !dbg !21
  %address.local1286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1287 = or disjoint i64 %address.local1280, 98, !dbg !21
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1286, i64 %address.local1287, !dbg !21
  store half %2063, ptr addrspace(1) %address.local1288, align 2, !dbg !21
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1290 = or disjoint i64 %address.local1280, 99, !dbg !21
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1289, i64 %address.local1290, !dbg !21
  store half %2062, ptr addrspace(1) %address.local1291, align 2, !dbg !21
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1293 = or disjoint i64 %address.local1280, 112, !dbg !21
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1292, i64 %address.local1293, !dbg !21
  store half %2056, ptr addrspace(1) %address.local1294, align 2, !dbg !21
  %address.local1295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1296 = or disjoint i64 %address.local1280, 113, !dbg !21
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1295, i64 %address.local1296, !dbg !21
  store half %2055, ptr addrspace(1) %address.local1297, align 2, !dbg !21
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1299 = or disjoint i64 %address.local1280, 114, !dbg !21
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1298, i64 %address.local1299, !dbg !21
  store half %2054, ptr addrspace(1) %address.local1300, align 2, !dbg !21
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1302 = or i64 %address.local1279, 115, !dbg !21
  %address.local1303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1301, i64 %address.local1302, !dbg !21
  store half %2053, ptr addrspace(1) %address.local1303, align 2, !dbg !21
  %address.local1304 = shl i64 %address.local1275, 7, !dbg !21
  %address.local1305 = add i64 %address.local1304, 8192, !dbg !21
  %address.local1306 = add nuw nsw i64 %2066, %address.local1305, !dbg !21
  %address.local1307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1307, i64 %address.local1281, !dbg !21
  store half %2047, ptr addrspace(1) %address.local1308, align 2, !dbg !21
  %address.local1309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1309, i64 %address.local1284, !dbg !21
  store half %2046, ptr addrspace(1) %address.local1310, align 2, !dbg !21
  %address.local1311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1311, i64 %address.local1287, !dbg !21
  store half %2045, ptr addrspace(1) %address.local1312, align 2, !dbg !21
  %address.local1313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1313, i64 %address.local1290, !dbg !21
  store half %2044, ptr addrspace(1) %address.local1314, align 2, !dbg !21
  %address.local1315 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1315, i64 %address.local1293, !dbg !21
  store half %2038, ptr addrspace(1) %address.local1316, align 2, !dbg !21
  %address.local1317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1317, i64 %address.local1296, !dbg !21
  store half %2037, ptr addrspace(1) %address.local1318, align 2, !dbg !21
  %address.local1319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1319, i64 %address.local1299, !dbg !21
  store half %2036, ptr addrspace(1) %address.local1320, align 2, !dbg !21
  %address.local1321 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1322 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1321, i64 %address.local1302, !dbg !21
  store half %2035, ptr addrspace(1) %address.local1322, align 2, !dbg !21
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-152102-801zvuny/grid-000002")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 70, column: 22, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 77, column: 19, scope: !4)
!14 = !DILocation(line: 102, column: 23, scope: !4)
!15 = !DILocation(line: 0, scope: !4)
!16 = !DILocation(line: 71, column: 22, scope: !4)
!17 = !DILocation(line: 73, column: 26, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!23 = !DILocation(line: 79, column: 24, scope: !4)
!24 = !DILocation(line: 80, column: 23, scope: !4)
!25 = !DILocation(line: 81, column: 26, scope: !4)
!26 = !DILocation(line: 88, column: 21, scope: !4)
!27 = !DILocation(line: 89, column: 20, scope: !4)
!28 = !DILocation(line: 90, column: 22, scope: !4)
!29 = !DILocation(line: 91, column: 23, scope: !4)
!30 = !DILocation(line: 92, column: 20, scope: !4)
!31 = !DILocation(line: 93, column: 22, scope: !4)
!32 = !DILocation(line: 94, column: 22, scope: !4)
!33 = !DILocation(line: 95, column: 20, scope: !4)
!34 = !DILocation(line: 96, column: 16, scope: !4)
!35 = !DILocation(line: 103, column: 22, scope: !4)
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
