; ModuleID = 'generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/layout_3_3/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-152102-801zvuny/grid-000002/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = shl nsw i64 %8, 5, !dbg !7
  %10 = add nsw i64 %9, %6, !dbg !7
  %11 = lshr i64 %10, 5, !dbg !7
  %12 = and i64 %11, 576460752303423484, !dbg !7
  %13 = and i64 %6, 3, !dbg !7
  %14 = or disjoint i64 %12, %13, !dbg !7
  %15 = sub nsw i64 31, %14, !dbg !7
  %16 = lshr i64 %10, 2, !dbg !7
  %17 = icmp ugt i64 %10, 1023, !dbg !8
  %18 = add nsw i64 %14, -32, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 32, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %22, %16, !dbg !8
  %24 = shl i64 %15, 7, !dbg !8
  %25 = shl i64 %22, 12, !dbg !8
  %26 = sub i64 %24, %25, !dbg !8
  %27 = shl i64 %23, 19, !dbg !8
  %28 = and i64 %27, 16252928, !dbg !8
  %29 = shl nsw i64 %26, 7, !dbg !8
  %30 = add i64 %28, %29, !dbg !8
  %31 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local505 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local506 = and i64 %address.local505, 1536, !dbg !8
  %address.local507 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local506, !dbg !8
  %address.local510 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local511 = and i64 %address.local510, 16256, !dbg !8
  %address.local512 = lshr i64 %address.local, 4, !dbg !8
  %address.local513 = and i64 %address.local512, 8, !dbg !8
  %address.local514 = or disjoint i64 %address.local511, %address.local513, !dbg !8
  %33 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1781 = or disjoint i64 %address.local514, %30, !dbg !8
  %.idx1781.tr = trunc i64 %.idx1781 to i32, !dbg !8
  %dma.buffer.offset1778 = shl i32 %.idx1781.tr, 1, !dbg !8
  %34 = insertelement <2 x i64> poison, i64 %33, i64 0, !dbg !8
  %35 = bitcast <2 x i64> %34 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1779 = shufflevector <4 x i32> %35, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local516 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local524 = lshr i64 %address.local516, 4, !dbg !8
  %address.local525 = and i64 %address.local524, 24, !dbg !8
  %address.local526 = or disjoint i64 %address.local525, %address.local511, !dbg !8
  %.idx1782 = or disjoint i64 %address.local526, %30, !dbg !8
  %.idx1782.tr = trunc i64 %.idx1782 to i32, !dbg !8
  %dma.buffer.offset1775 = shl i32 %.idx1782.tr, 1, !dbg !8
  %36 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %37 = sext i32 %36 to i64, !dbg !7
  %address.local528 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local506, !dbg !8
  %38 = or disjoint i64 %30, 32, !dbg !8
  %39 = or disjoint i64 %address.local514, %38, !dbg !8
  %.tr = trunc i64 %39 to i32, !dbg !8
  %dma.buffer.offset1772 = shl i32 %.tr, 1, !dbg !8
  %address.local532 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %40 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local507, i64 0, !dbg !8
  %41 = insertelement <4 x ptr addrspace(3)> %40, ptr addrspace(3) %address.local519, i64 1, !dbg !8
  %42 = insertelement <4 x ptr addrspace(3)> %41, ptr addrspace(3) %address.local528, i64 2, !dbg !8
  %43 = insertelement <4 x ptr addrspace(3)> %42, ptr addrspace(3) %address.local532, i64 3, !dbg !8
  %44 = ptrtoint <4 x ptr addrspace(3)> %43 to <4 x i64>, !dbg !8
  %45 = trunc <4 x i64> %44 to <4 x i32>, !dbg !8
  %46 = extractelement <4 x i32> %45, i64 0, !dbg !8
  %47 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %dma.buffer.lds1780 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %47), !dbg !8
  %48 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1780, i32 %dma.buffer.offset1778, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %49 = extractelement <4 x i32> %45, i64 1, !dbg !8
  %50 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %49), !dbg !8
  %dma.buffer.lds1777 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %51 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1777, i32 %dma.buffer.offset1775, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %52 = extractelement <4 x i32> %45, i64 2, !dbg !8
  %53 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %52), !dbg !8
  %dma.buffer.lds1774 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %54 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1774, i32 %dma.buffer.offset1772, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %55 = extractelement <4 x i32> %45, i64 3, !dbg !8
  %56 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %55), !dbg !8
  %57 = or disjoint i64 %address.local526, %38, !dbg !8
  %.tr1783 = trunc i64 %57 to i32, !dbg !8
  %dma.buffer.offset1769 = shl i32 %.tr1783, 1, !dbg !8
  %dma.buffer.lds1771 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %56), !dbg !8
  %58 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1771, i32 %dma.buffer.offset1769, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %address.local536 = lshr i32 %address.thread, 2, !dbg !10
  %address.local537 = and i32 %address.local536, 4, !dbg !10
  %address.local538 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local539 = and i32 %address.local538, 384, !dbg !10
  %address.local540 = or disjoint i32 %address.local537, %address.local539, !dbg !10
  %address.local541 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local542 = and i32 %address.local541, 120, !dbg !10
  %address.local543 = or disjoint i32 %address.local540, %address.local542, !dbg !10
  %address.local544 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local545 = and i32 %address.local544, 1024, !dbg !10
  %address.local546 = or disjoint i32 %address.local543, %address.local545, !dbg !10
  %address.local547 = zext nneg i32 %address.local546 to i64, !dbg !10
  %address.local548 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local547, !dbg !10
  %59 = load <4 x half>, ptr addrspace(3) %address.local548, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local548, i64 4096, !dbg !10
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %61 = fmul <4 x half> %59, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local548, i64 1024, !dbg !10
  %62 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !10
  %63 = fmul <4 x half> %60, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local548, i64 5120, !dbg !10
  %64 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !10
  %65 = fmul <4 x half> %62, splat (half 1.275630e-01), !dbg !10
  %66 = fmul <4 x half> %64, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %67 = or disjoint i64 %30, 64, !dbg !8
  %68 = or disjoint i64 %address.local514, %67, !dbg !8
  %.tr1784 = trunc i64 %68 to i32, !dbg !8
  %dma.buffer.offset1766 = shl i32 %.tr1784, 1, !dbg !8
  %69 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1780, i32 %dma.buffer.offset1766, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %70 = or disjoint i64 %address.local526, %67, !dbg !8
  %.tr1785 = trunc i64 %70 to i32, !dbg !8
  %dma.buffer.offset1763 = shl i32 %.tr1785, 1, !dbg !8
  %71 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1777, i32 %dma.buffer.offset1763, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local548, i64 8192, !dbg !10
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !10
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local548, i64 12288, !dbg !10
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !10
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local548, i64 9216, !dbg !10
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !10
  %75 = fmul <4 x half> %72, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local548, i64 13312, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !10
  %77 = fmul <4 x half> %73, splat (half 1.275630e-01), !dbg !10
  %78 = fmul <4 x half> %74, splat (half 1.275630e-01), !dbg !10
  %79 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %80 = or disjoint i64 %30, 96, !dbg !8
  %81 = or disjoint i64 %address.local514, %80, !dbg !8
  %.tr1786 = trunc i64 %81 to i32, !dbg !8
  %dma.buffer.offset1760 = shl i32 %.tr1786, 1, !dbg !8
  %82 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1774, i32 %dma.buffer.offset1760, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %83 = or disjoint i64 %address.local526, %80, !dbg !8
  %.tr1787 = trunc i64 %83 to i32, !dbg !8
  %dma.buffer.offset1757 = shl i32 %.tr1787, 1, !dbg !8
  %84 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1771, i32 %dma.buffer.offset1757, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %85 = load <4 x half>, ptr addrspace(3) %address.local548, align 8, !dbg !10
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %87 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !10
  %88 = fmul <4 x half> %85, splat (half 1.275630e-01), !dbg !10
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !10
  %90 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !10
  %91 = fmul <4 x half> %87, splat (half 1.275630e-01), !dbg !10
  %92 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !10
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !10
  %95 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !10
  %96 = fmul <4 x half> %93, splat (half 1.275630e-01), !dbg !10
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !10
  %98 = fmul <4 x half> %94, splat (half 1.275630e-01), !dbg !10
  %99 = fmul <4 x half> %95, splat (half 1.275630e-01), !dbg !10
  %100 = fmul <4 x half> %97, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %101 = or disjoint i64 %24, 127, !dbg !12
  %102 = add i64 %24, 128, !dbg !12
  %103 = icmp uge i64 %102, %101, !dbg !12
  %104 = sub nsw i64 32, %14, !dbg !13
  %105 = shl i64 %16, 7, !dbg !14
  %106 = and i64 %105, 3968, !dbg !14
  %107 = shl nsw i64 %37, 12, !dbg !14
  %108 = or disjoint i64 %106, %107, !dbg !14
  %109 = shl nuw nsw i64 %32, 2, !dbg !15
  %110 = and i64 %109, 12, !dbg !15
  %111 = and i32 %31, 1, !dbg !15
  %narrow = add nuw nsw i32 %111, 1, !dbg !15
  %112 = zext nneg i32 %narrow to i64, !dbg !15
  %113 = or disjoint i64 %110, %112, !dbg !15
  %114 = lshr i64 %113, 1, !dbg !15
  %115 = icmp ult i64 %10, 1024, !dbg !11
  br i1 %115, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !11

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre17 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre18 = add nuw nsw i32 %.pre17, 64, !dbg !11
  %.pre19 = xor i32 %.pre17, 16, !dbg !11
  %.pre21 = and i32 %.pre18, 192, !dbg !11
  %.pre23 = xor i32 %.pre17, 32, !dbg !11
  br label %._crit_edge, !dbg !11

.lr.ph:                                           ; preds = %4
  %116 = ptrtoint ptr addrspace(1) %1 to i64
  %117 = insertelement <2 x i64> poison, i64 %116, i64 0
  %118 = bitcast <2 x i64> %117 to <4 x i32>
  %dma.buffer.rsrc1755 = shufflevector <4 x i32> %118, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %119 = ptrtoint ptr addrspace(1) %2 to i64
  %120 = insertelement <2 x i64> poison, i64 %119, i64 0
  %121 = bitcast <2 x i64> %120 to <4 x i32>
  %dma.buffer.rsrc1731 = shufflevector <4 x i32> %121, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %122 = lshr i64 %32, 2
  %123 = and i64 %122, 48
  %124 = and i32 %31, 240
  %index.compact = zext nneg i32 %124 to i64
  %125 = sub nsw i64 %32, %index.compact
  %126 = add nsw i64 %125, %123
  %127 = add nsw i64 %126, %24
  %128 = and i64 %32, 48
  %129 = add i64 %127, 1
  %130 = lshr i64 %129, 32
  %compare.high = trunc nuw i64 %130 to i32
  %compare.low.base = trunc i64 %129 to i32
  %131 = add i64 %127, 65
  %132 = lshr i64 %131, 32
  %compare.high1447 = trunc nuw i64 %132 to i32
  %compare.low.base1451 = trunc i64 %131 to i32
  %133 = insertelement <4 x i32> poison, i32 %compare.low.base1451, i64 0, !dbg !16
  %134 = shufflevector <4 x i32> %133, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !16
  %135 = insertelement <4 x i32> poison, i32 %compare.low.base, i64 0, !dbg !16
  %136 = shufflevector <4 x i32> %135, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !16
  br label %137, !dbg !11

137:                                              ; preds = %.lr.ph, %592
  %138 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1291, %592 ]
  %139 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1287, %592 ]
  %140 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1283, %592 ]
  %141 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1279, %592 ]
  %142 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %960, %592 ]
  %143 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1277, %592 ]
  %144 = phi i64 [ 0, %.lr.ph ], [ %1292, %592 ]
  %145 = shl nuw nsw i64 %144, 7, !dbg !11
  %146 = lshr i64 %144, 12, !dbg !14
  %147 = add nuw nsw i64 %146, %16, !dbg !14
  %148 = lshr i64 %144, 5, !dbg !14
  %149 = and i64 %148, 127, !dbg !14
  %150 = shl i64 %147, 19, !dbg !14
  %151 = and i64 %150, 16252928, !dbg !14
  %152 = shl i64 %148, 19, !dbg !14
  %153 = shl nuw nsw i64 %144, 14, !dbg !14
  %154 = sub i64 %153, %152, !dbg !14
  %address.thread599 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !14, !range !9, !frisk.register_lifetimes !5
  %address.local600 = zext nneg i32 %address.thread599 to i64, !dbg !8
  %address.local601 = shl nuw nsw i64 %address.local600, 3, !dbg !8
  %address.local602 = and i64 %address.local601, 1536, !dbg !8
  %address.local603 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local602, !dbg !8
  %address.local606 = lshr i64 %address.local600, 4, !dbg !14
  %address.local607 = and i64 %address.local606, 8, !dbg !14
  %address.local608 = and i64 %address.local600, 123, !dbg !14
  %address.local609 = lshr i64 %address.local600, 2, !dbg !14
  %address.local610 = and i64 %address.local609, 4, !dbg !14
  %address.local611 = add nuw nsw i64 %address.local610, %address.local600, !dbg !14
  %address.local612 = and i64 %address.local611, 4, !dbg !14
  %address.local613 = or disjoint i64 %address.local612, %address.local608, !dbg !14
  %address.local614 = shl nuw nsw i64 %address.local613, 7, !dbg !14
  %address.local615 = or disjoint i64 %address.local614, %address.local607, !dbg !14
  %.idx1790 = or disjoint i64 %151, %149, !dbg !14
  %155 = add i64 %.idx1790, %154, !dbg !14
  %156 = add i64 %address.local615, %155, !dbg !14
  %.tr1791 = trunc i64 %156 to i32, !dbg !14
  %dma.buffer.offset1754 = shl i32 %.tr1791, 1, !dbg !14
  %address.local617 = or disjoint i64 %address.local600, 256, !dbg !8
  %address.local618 = shl nuw nsw i64 %address.local617, 3, !dbg !8
  %address.local619 = and i64 %address.local618, 3584, !dbg !8
  %address.local620 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local619, !dbg !8
  %address.local623 = lshr i64 %address.local617, 4, !dbg !14
  %address.local624 = and i64 %address.local623, 24, !dbg !14
  %address.local632 = or disjoint i64 %address.local614, %address.local624, !dbg !14
  %157 = add i64 %address.local632, %155, !dbg !14
  %.tr1795 = trunc i64 %157 to i32, !dbg !14
  %dma.buffer.offset1751 = shl i32 %.tr1795, 1, !dbg !14
  %158 = add nuw nsw i64 %144, 1024, !dbg !14
  %159 = lshr i64 %158, 12, !dbg !14
  %160 = add nuw nsw i64 %159, %16, !dbg !14
  %161 = add nsw i64 %148, %108, !dbg !14
  %162 = shl i64 %160, 19, !dbg !14
  %163 = and i64 %162, 16252928, !dbg !14
  %address.local634 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local602, !dbg !8
  %164 = add i64 %154, %148, !dbg !14
  %.idx1799 = add i64 %164, %108, !dbg !14
  %165 = add i64 %.idx1799, %163, !dbg !14
  %address.local638 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local619, !dbg !8
  %166 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local603, i64 0, !dbg !8
  %167 = insertelement <4 x ptr addrspace(3)> %166, ptr addrspace(3) %address.local620, i64 1, !dbg !8
  %168 = insertelement <4 x ptr addrspace(3)> %167, ptr addrspace(3) %address.local634, i64 2, !dbg !8
  %169 = insertelement <4 x ptr addrspace(3)> %168, ptr addrspace(3) %address.local638, i64 3, !dbg !8
  %170 = ptrtoint <4 x ptr addrspace(3)> %169 to <4 x i64>, !dbg !8
  %171 = trunc <4 x i64> %170 to <4 x i32>, !dbg !8
  %172 = extractelement <4 x i32> %171, i64 0, !dbg !14
  %173 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %172), !dbg !14
  %dma.buffer.lds1756 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %173), !dbg !14
  %174 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1756, i32 %dma.buffer.offset1754, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %175 = extractelement <4 x i32> %171, i64 1, !dbg !14
  %176 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %175), !dbg !14
  %dma.buffer.lds1753 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %176), !dbg !14
  %177 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1753, i32 %dma.buffer.offset1751, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %178 = extractelement <4 x i32> %171, i64 2, !dbg !14
  %179 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %178), !dbg !14
  %dma.buffer.lds1750 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %179), !dbg !14
  %180 = extractelement <4 x i32> %171, i64 3, !dbg !14
  %address.local646 = shl nuw nsw i64 %address.local600, 1, !dbg !15
  %address.local647 = and i64 %address.local646, 24, !dbg !15
  %address.local649 = shl i64 %address.local600, 62, !dbg !15
  %address.local650 = ashr exact i64 %address.local649, 57, !dbg !15
  %address.local651 = and i64 %address.local650, -64, !dbg !15
  %address.local654 = and i64 %address.local609, 12, !dbg !10
  %181 = and i64 %address.local600, 32, !dbg !10
  %.not1806 = icmp eq i64 %181, 0, !dbg !10
  %address.local658 = select i1 %.not1806, i64 0, i64 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local600, 320, !dbg !15
  %182 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !15
  %.idx1807 = mul nsw i64 %address.local609, -1280, !dbg !15
  %183 = getelementptr i8, ptr addrspace(3) %182, i64 %.idx1807, !dbg !15
  %184 = getelementptr [2 x i8], ptr addrspace(3) %183, i64 %address.local647, !dbg !15
  %185 = getelementptr [2 x i8], ptr addrspace(3) %184, i64 %address.local651, !dbg !15
  %186 = getelementptr [2 x i8], ptr addrspace(3) %185, i64 %address.local654, !dbg !15
  %address.local660 = getelementptr [2 x i8], ptr addrspace(3) %186, i64 %address.local658, !dbg !15
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !15
  %address.local671 = shl nuw nsw i32 %address.thread599, 1, !dbg !15
  %address.local672 = and i32 %address.local671, 24, !dbg !15
  %address.local673 = shl nuw nsw i32 %address.thread599, 8, !dbg !15
  %address.local674 = and i32 %address.local673, 768, !dbg !15
  %address.local675 = or disjoint i32 %address.local672, %address.local674, !dbg !15
  %address.local676 = shl nuw nsw i32 %address.thread599, 5, !dbg !15
  %address.local677 = and i32 %address.local676, 96, !dbg !15
  %address.local678 = or disjoint i32 %address.local675, %address.local677, !dbg !15
  %address.local679 = zext nneg i32 %address.local678 to i64, !dbg !15
  %address.local680 = lshr i32 %address.thread599, 1, !dbg !15
  %address.local681 = and i32 %address.local680, 1, !dbg !15
  %address.local682 = zext nneg i32 %address.local681 to i64, !dbg !15
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !15
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !15
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !15
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !15
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !15
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !15
  %187 = add nuw nsw i64 %144, 2048, !dbg !14
  %188 = lshr i64 %187, 12, !dbg !14
  %189 = add nuw nsw i64 %188, %16, !dbg !14
  %190 = insertelement <2 x i64> poison, i64 %161, i64 0, !dbg !14
  %191 = shufflevector <2 x i64> %190, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !14
  %192 = icmp slt <2 x i64> %191, <i64 -32, i64 -64>, !dbg !14
  %193 = extractelement <2 x i1> %192, i64 0, !dbg !14
  %194 = sext i1 %193 to i64, !dbg !14
  %195 = add nsw <2 x i64> %191, <i64 32, i64 64>, !dbg !14
  %196 = sub nsw <2 x i64> <i64 -33, i64 -65>, %191, !dbg !14
  %197 = select <2 x i1> %192, <2 x i64> %196, <2 x i64> %195, !dbg !14
  %198 = sdiv <2 x i64> %197, splat (i64 128), !dbg !14
  %199 = extractelement <2 x i64> %198, i64 0, !dbg !14
  %200 = xor i64 %199, %194, !dbg !14
  %201 = shl nsw i64 %200, 7, !dbg !14
  %202 = sub i64 %165, %201, !dbg !14
  %203 = add i64 %address.local615, %202, !dbg !14
  %.tr1800 = trunc i64 %203 to i32, !dbg !14
  %204 = shl i32 %.tr1800, 1, !dbg !14
  %dma.buffer.offset1748 = add i32 %204, 64, !dbg !14
  %205 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1750, i32 %dma.buffer.offset1748, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %206 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %180), !dbg !14
  %207 = add i64 %address.local632, %202, !dbg !14
  %.tr1805 = trunc i64 %207 to i32, !dbg !14
  %208 = shl i32 %.tr1805, 1, !dbg !14
  %dma.buffer.offset1745 = add i32 %208, 64, !dbg !14
  %dma.buffer.lds1747 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %206), !dbg !14
  %209 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1747, i32 %dma.buffer.offset1745, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %210 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !15
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %210, <4 x float> zeroinitializer) #8, !dbg !15
  %212 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !15
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %114) #8, !dbg !15, !frisk.register_lifetimes !5
  %address.local683 = add nsw i64 %address.base, %address.local682, !dbg !15
  %address.local684 = shl i64 %address.local683, 6, !dbg !15
  %address.local685 = sub i64 %address.local679, %address.local684, !dbg !15
  %213 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local685, !dbg !15
  %214 = getelementptr [2 x i8], ptr addrspace(3) %213, i64 %address.local654, !dbg !15
  %215 = getelementptr [2 x i8], ptr addrspace(3) %214, i64 %address.local658, !dbg !15
  %address.local689 = getelementptr i8, ptr addrspace(3) %215, i64 64, !dbg !15
  %216 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !15
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %212, <4 x float> %211) #8, !dbg !15
  %218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %216, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %215, i64 4160, !dbg !15
  %219 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !15
  %220 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !15
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %219, <4 x float> %218) #8, !dbg !15
  %222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %220, <4 x float> zeroinitializer) #8, !dbg !15
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !15
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %215, i64 192, !dbg !15
  %224 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !15
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %223, <4 x float> %222) #8, !dbg !15
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %215, i64 4288, !dbg !15
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !15
  %228 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !15
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %227, <4 x float> %226) #8, !dbg !15
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %228, <4 x float> zeroinitializer) #8, !dbg !15
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !15
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %215, i64 1088, !dbg !15
  %232 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !15
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %231, <4 x float> %230) #8, !dbg !15
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %232, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %215, i64 5184, !dbg !15
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !15
  %236 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !15
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %235, <4 x float> %234) #8, !dbg !15
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !15
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !15
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %215, i64 1216, !dbg !15
  %240 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !15
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %239, <4 x float> %238) #8, !dbg !15
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %240, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %215, i64 5312, !dbg !15
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !15
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %243, <4 x float> %242) #8, !dbg !15
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %210, <4 x float> zeroinitializer) #8, !dbg !15
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %212, <4 x float> %245) #8, !dbg !15
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %216, <4 x float> zeroinitializer) #8, !dbg !15
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %219, <4 x float> %247) #8, !dbg !15
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %220, <4 x float> zeroinitializer) #8, !dbg !15
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %223, <4 x float> %249) #8, !dbg !15
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !15
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %227, <4 x float> %251) #8, !dbg !15
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %228, <4 x float> zeroinitializer) #8, !dbg !15
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %231, <4 x float> %253) #8, !dbg !15
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %232, <4 x float> zeroinitializer) #8, !dbg !15
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %235, <4 x float> %255) #8, !dbg !15
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !15
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %239, <4 x float> %257) #8, !dbg !15
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %240, <4 x float> zeroinitializer) #8, !dbg !15
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %243, <4 x float> %259) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %261 = extractelement <2 x i1> %192, i64 1, !dbg !14
  %262 = sext i1 %261 to i64, !dbg !14
  %263 = extractelement <2 x i64> %198, i64 1, !dbg !14
  %264 = xor i64 %263, %262, !dbg !14
  %265 = shl i64 %189, 19, !dbg !14
  %266 = and i64 %265, 16252928, !dbg !14
  %267 = add i64 %.idx1799, %266, !dbg !14
  %268 = shl nsw i64 %264, 7, !dbg !14
  %269 = sub i64 %267, %268, !dbg !14
  %270 = add i64 %address.local615, %269, !dbg !14
  %.tr1813 = trunc i64 %270 to i32, !dbg !14
  %271 = shl i32 %.tr1813, 1, !dbg !14
  %dma.buffer.offset1742 = add i32 %271, 128, !dbg !14
  %272 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1756, i32 %dma.buffer.offset1742, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %273 = add i64 %address.local632, %269, !dbg !14
  %.tr1819 = trunc i64 %273 to i32, !dbg !14
  %274 = shl i32 %.tr1819, 1, !dbg !14
  %dma.buffer.offset1739 = add i32 %274, 128, !dbg !14
  %275 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1753, i32 %dma.buffer.offset1739, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !15
  %276 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !15
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %276, <4 x float> %217) #8, !dbg !15
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !15
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !15
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %215, i64 8256, !dbg !15
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !15
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %278, <4 x float> %277) #8, !dbg !15
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %279, <4 x float> %221) #8, !dbg !15
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %215, i64 12352, !dbg !15
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !15
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !15
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !15
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %282, <4 x float> %281) #8, !dbg !15
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %283, <4 x float> %225) #8, !dbg !15
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !15
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !15
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %215, i64 8384, !dbg !15
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !15
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %286, <4 x float> %285) #8, !dbg !15
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %287, <4 x float> %229) #8, !dbg !15
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %215, i64 12480, !dbg !15
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !15
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !15
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !15
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %290, <4 x float> %289) #8, !dbg !15
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %291, <4 x float> %233) #8, !dbg !15
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !15
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !15
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %215, i64 9280, !dbg !15
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !15
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %294, <4 x float> %293) #8, !dbg !15
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %295, <4 x float> %237) #8, !dbg !15
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %215, i64 13376, !dbg !15
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !15
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !15
  %299 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !15
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %298, <4 x float> %297) #8, !dbg !15
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %299, <4 x float> %241) #8, !dbg !15
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !15
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !15
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %215, i64 9408, !dbg !15
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !15
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %302, <4 x float> %301) #8, !dbg !15
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %303, <4 x float> %244) #8, !dbg !15
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %215, i64 13504, !dbg !15
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !15
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %306, <4 x float> %305) #8, !dbg !15
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %276, <4 x float> %246) #8, !dbg !15
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %278, <4 x float> %308) #8, !dbg !15
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %279, <4 x float> %248) #8, !dbg !15
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %282, <4 x float> %310) #8, !dbg !15
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %283, <4 x float> %250) #8, !dbg !15
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %286, <4 x float> %312) #8, !dbg !15
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %287, <4 x float> %252) #8, !dbg !15
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %290, <4 x float> %314) #8, !dbg !15
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %291, <4 x float> %254) #8, !dbg !15
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %294, <4 x float> %316) #8, !dbg !15
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %295, <4 x float> %256) #8, !dbg !15
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %298, <4 x float> %318) #8, !dbg !15
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %299, <4 x float> %258) #8, !dbg !15
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %302, <4 x float> %320) #8, !dbg !15
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %303, <4 x float> %260) #8, !dbg !15
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %306, <4 x float> %322) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %324 = add nuw nsw i64 %144, 3072, !dbg !14
  %325 = lshr i64 %324, 12, !dbg !14
  %326 = add nuw nsw i64 %325, %16, !dbg !14
  %327 = add nsw i64 %161, 96, !dbg !14
  %328 = icmp slt i64 %161, -96, !dbg !14
  %329 = sub nsw i64 -97, %161, !dbg !14
  %330 = select i1 %328, i64 %329, i64 %327, !dbg !14
  %331 = sdiv i64 %330, 128, !dbg !14
  %332 = sext i1 %328 to i64, !dbg !14
  %333 = xor i64 %331, %332, !dbg !14
  %334 = shl i64 %326, 19, !dbg !14
  %335 = and i64 %334, 16252928, !dbg !14
  %336 = add i64 %.idx1799, %335, !dbg !14
  %337 = shl nsw i64 %333, 7, !dbg !14
  %338 = sub i64 %336, %337, !dbg !14
  %339 = add i64 %address.local615, %338, !dbg !14
  %.tr1825 = trunc i64 %339 to i32, !dbg !14
  %340 = shl i32 %.tr1825, 1, !dbg !14
  %dma.buffer.offset1736 = add i32 %340, 192, !dbg !14
  %341 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1750, i32 %dma.buffer.offset1736, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %342 = add i64 %address.local632, %338, !dbg !14
  %.tr1831 = trunc i64 %342 to i32, !dbg !14
  %343 = shl i32 %.tr1831, 1, !dbg !14
  %dma.buffer.offset1733 = add i32 %343, 192, !dbg !14
  %344 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1747, i32 %dma.buffer.offset1733, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %345 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !15
  %346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %345, <4 x float> %280) #8, !dbg !15
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !15
  %348 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !15
  %349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %347, <4 x float> %346) #8, !dbg !15
  %350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %348, <4 x float> %284) #8, !dbg !15
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !15
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !15
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %351, <4 x float> %350) #8, !dbg !15
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %352, <4 x float> %288) #8, !dbg !15
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !15
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !15
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %355, <4 x float> %354) #8, !dbg !15
  %358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %356, <4 x float> %292) #8, !dbg !15
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !15
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !15
  %361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %359, <4 x float> %358) #8, !dbg !15
  %362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %360, <4 x float> %296) #8, !dbg !15
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !15
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !15
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %363, <4 x float> %362) #8, !dbg !15
  %366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %364, <4 x float> %300) #8, !dbg !15
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !15
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !15
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %367, <4 x float> %366) #8, !dbg !15
  %370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %368, <4 x float> %304) #8, !dbg !15
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !15
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !15
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %371, <4 x float> %370) #8, !dbg !15
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %372, <4 x float> %307) #8, !dbg !15
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !15
  %376 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %375, <4 x float> %374) #8, !dbg !15
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %345, <4 x float> %309) #8, !dbg !15
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %347, <4 x float> %377) #8, !dbg !15
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %348, <4 x float> %311) #8, !dbg !15
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %351, <4 x float> %379) #8, !dbg !15
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %352, <4 x float> %313) #8, !dbg !15
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %355, <4 x float> %381) #8, !dbg !15
  %383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %356, <4 x float> %315) #8, !dbg !15
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %359, <4 x float> %383) #8, !dbg !15
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %360, <4 x float> %317) #8, !dbg !15
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %363, <4 x float> %385) #8, !dbg !15
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %364, <4 x float> %319) #8, !dbg !15
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %367, <4 x float> %387) #8, !dbg !15
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %368, <4 x float> %321) #8, !dbg !15
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %371, <4 x float> %389) #8, !dbg !15
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %372, <4 x float> %323) #8, !dbg !15
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %375, <4 x float> %391) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %393 = add nuw nsw i64 %148, %16, !dbg !17
  %394 = shl i64 %393, 19, !dbg !17
  %395 = and i64 %394, 16252928, !dbg !17
  %address.local1018 = shl nuw nsw i64 %address.local600, 6, !dbg !17
  %address.local1019 = and i64 %address.local1018, 16128, !dbg !17
  %address.local1020 = shl nuw nsw i64 %address.local600, 4, !dbg !17
  %address.local1021 = and i64 %address.local1020, 48, !dbg !17
  %.idx18321835 = add i64 %395, %154, !dbg !17
  %396 = shl i64 %.idx18321835, 1, !dbg !17
  %397 = or disjoint i64 %396, %address.local1019, !dbg !17
  %398 = or disjoint i64 %397, %address.local1021, !dbg !17
  %dma.buffer.offset1730 = trunc i64 %398 to i32, !dbg !17
  %399 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1756, i32 %dma.buffer.offset1730, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %address.local1024 = shl nuw nsw i64 %address.local617, 6, !dbg !17
  %address.local1025 = and i64 %address.local1024, 32512, !dbg !17
  %400 = or disjoint i64 %396, %address.local1025, !dbg !17
  %401 = or disjoint i64 %400, %address.local1021, !dbg !17
  %dma.buffer.offset1727 = trunc i64 %401 to i32, !dbg !17
  %402 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1753, i32 %dma.buffer.offset1727, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %403 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !15
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %403, <4 x float> %349) #8, !dbg !15
  %405 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !15
  %406 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !15
  %407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %405, <4 x float> %404) #8, !dbg !15
  %408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %406, <4 x float> %353) #8, !dbg !15
  %409 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !15
  %410 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !15
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %409, <4 x float> %408) #8, !dbg !15
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %410, <4 x float> %357) #8, !dbg !15
  %413 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !15
  %414 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !15
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %413, <4 x float> %412) #8, !dbg !15
  %416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %414, <4 x float> %361) #8, !dbg !15
  %417 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !15
  %418 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !15
  %419 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %417, <4 x float> %416) #8, !dbg !15
  %420 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %418, <4 x float> %365) #8, !dbg !15
  %421 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !15
  %422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !15
  %423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %421, <4 x float> %420) #8, !dbg !15
  %424 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %422, <4 x float> %369) #8, !dbg !15
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !15
  %426 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !15
  %427 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %425, <4 x float> %424) #8, !dbg !15
  %428 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %426, <4 x float> %373) #8, !dbg !15
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !15
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !15
  %431 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %429, <4 x float> %428) #8, !dbg !15
  %432 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %430, <4 x float> %376) #8, !dbg !15
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !15
  %434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %433, <4 x float> %432) #8, !dbg !15
  %435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %403, <4 x float> %378) #8, !dbg !15
  %436 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %405, <4 x float> %435) #8, !dbg !15
  %437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %406, <4 x float> %380) #8, !dbg !15
  %438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %409, <4 x float> %437) #8, !dbg !15
  %439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %410, <4 x float> %382) #8, !dbg !15
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %413, <4 x float> %439) #8, !dbg !15
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %414, <4 x float> %384) #8, !dbg !15
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %417, <4 x float> %441) #8, !dbg !15
  %443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %418, <4 x float> %386) #8, !dbg !15
  %444 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %421, <4 x float> %443) #8, !dbg !15
  %445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %422, <4 x float> %388) #8, !dbg !15
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %425, <4 x float> %445) #8, !dbg !15
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %426, <4 x float> %390) #8, !dbg !15
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %429, <4 x float> %447) #8, !dbg !15
  %449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %430, <4 x float> %392) #8, !dbg !15
  %450 = or disjoint i64 %145, 127, !dbg !12
  %451 = icmp uge i64 %24, %450, !dbg !12
  %452 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %433, <4 x float> %449) #8, !dbg !15
  %453 = and i1 %103, %451, !dbg !12
  br i1 %453, label %592, label %454, !dbg !12

454:                                              ; preds = %137
  %455 = or disjoint i64 %145, %128, !dbg !12
  %456 = lshr i64 %144, 25, !dbg !16
  %compare.high1323 = trunc nuw i64 %456 to i32, !dbg !16
  %compare.low.base1324 = trunc i64 %455 to i32, !dbg !16
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 1) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1327 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 2) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1331 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 3) #8, !dbg !16, !frisk.register_lifetimes !5
  %457 = insertelement <4 x i32> poison, i32 %compare.low.base1324, i64 0, !dbg !16
  %458 = insertelement <4 x i32> %457, i32 %compare.local.base, i64 1, !dbg !16
  %459 = insertelement <4 x i32> %458, i32 %compare.local.base1327, i64 2, !dbg !16
  %460 = insertelement <4 x i32> %459, i32 %compare.local.base1331, i64 3, !dbg !16
  %461 = or <4 x i32> %460, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %462 = icmp ugt <4 x i32> %136, %461, !dbg !16
  %compare.local.base1335 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 4) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1339 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 5) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1343 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 6) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1347 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 7) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1351 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 8) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1355 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 9) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1359 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 10) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1363 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 11) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1367 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 12) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1371 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 13) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1375 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 14) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1379 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 15) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1383 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 16) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1387 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 17) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1391 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 18) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1395 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 19) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1399 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 20) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1403 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 21) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1407 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 22) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1411 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 23) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1415 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 24) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1419 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 25) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1423 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 26) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1427 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 27) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1431 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 28) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1435 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 29) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1439 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 30) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1443 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 31) #8, !dbg !16, !frisk.register_lifetimes !5
  %463 = insertelement <4 x i32> poison, i32 %compare.local.base1415, i64 0, !dbg !16
  %464 = insertelement <4 x i32> %463, i32 %compare.local.base1419, i64 1, !dbg !16
  %465 = insertelement <4 x i32> %464, i32 %compare.local.base1423, i64 2, !dbg !16
  %466 = insertelement <4 x i32> %465, i32 %compare.local.base1427, i64 3, !dbg !16
  %467 = or <4 x i32> %466, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %468 = icmp ult <4 x i32> %467, %136, !dbg !16
  %469 = insertelement <4 x i32> poison, i32 %compare.local.base1399, i64 0, !dbg !16
  %470 = insertelement <4 x i32> %469, i32 %compare.local.base1403, i64 1, !dbg !16
  %471 = insertelement <4 x i32> %470, i32 %compare.local.base1407, i64 2, !dbg !16
  %472 = insertelement <4 x i32> %471, i32 %compare.local.base1411, i64 3, !dbg !16
  %473 = or <4 x i32> %472, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %474 = icmp ult <4 x i32> %473, %136, !dbg !16
  %475 = insertelement <4 x i32> poison, i32 %compare.local.base1383, i64 0, !dbg !16
  %476 = insertelement <4 x i32> %475, i32 %compare.local.base1387, i64 1, !dbg !16
  %477 = insertelement <4 x i32> %476, i32 %compare.local.base1391, i64 2, !dbg !16
  %478 = insertelement <4 x i32> %477, i32 %compare.local.base1395, i64 3, !dbg !16
  %479 = or <4 x i32> %478, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %480 = icmp ult <4 x i32> %479, %136, !dbg !16
  %481 = insertelement <4 x i32> poison, i32 %compare.local.base1367, i64 0, !dbg !16
  %482 = insertelement <4 x i32> %481, i32 %compare.local.base1371, i64 1, !dbg !16
  %483 = insertelement <4 x i32> %482, i32 %compare.local.base1375, i64 2, !dbg !16
  %484 = insertelement <4 x i32> %483, i32 %compare.local.base1379, i64 3, !dbg !16
  %485 = or <4 x i32> %484, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %486 = icmp ult <4 x i32> %485, %136, !dbg !16
  %487 = insertelement <4 x i32> poison, i32 %compare.local.base1351, i64 0, !dbg !16
  %488 = insertelement <4 x i32> %487, i32 %compare.local.base1355, i64 1, !dbg !16
  %489 = insertelement <4 x i32> %488, i32 %compare.local.base1359, i64 2, !dbg !16
  %490 = insertelement <4 x i32> %489, i32 %compare.local.base1363, i64 3, !dbg !16
  %491 = or <4 x i32> %490, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %492 = icmp ult <4 x i32> %491, %136, !dbg !16
  %493 = insertelement <4 x i32> poison, i32 %compare.local.base1335, i64 0, !dbg !16
  %494 = insertelement <4 x i32> %493, i32 %compare.local.base1339, i64 1, !dbg !16
  %495 = insertelement <4 x i32> %494, i32 %compare.local.base1343, i64 2, !dbg !16
  %496 = insertelement <4 x i32> %495, i32 %compare.local.base1347, i64 3, !dbg !16
  %497 = or <4 x i32> %496, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %498 = icmp ult <4 x i32> %497, %136, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1323, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1323, !dbg !16
  %499 = insertelement <4 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !16
  %500 = shufflevector <4 x i1> %499, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %501 = insertelement <4 x i1> poison, i1 %compare.high.order, i64 0, !dbg !16
  %502 = shufflevector <4 x i1> %501, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %503 = select <4 x i1> %500, <4 x i1> %462, <4 x i1> %502, !dbg !16
  %504 = select <4 x i1> %503, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %505 = select <4 x i1> %500, <4 x i1> %498, <4 x i1> %502, !dbg !16
  %506 = select <4 x i1> %505, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %507 = select <4 x i1> %500, <4 x i1> %492, <4 x i1> %502, !dbg !16
  %508 = select <4 x i1> %507, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %509 = select <4 x i1> %500, <4 x i1> %486, <4 x i1> %502, !dbg !16
  %510 = select <4 x i1> %509, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %511 = select <4 x i1> %500, <4 x i1> %480, <4 x i1> %502, !dbg !16
  %512 = select <4 x i1> %511, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %513 = select <4 x i1> %500, <4 x i1> %474, <4 x i1> %502, !dbg !16
  %514 = select <4 x i1> %513, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %515 = select <4 x i1> %500, <4 x i1> %468, <4 x i1> %502, !dbg !16
  %516 = select <4 x i1> %515, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %517 = insertelement <4 x i32> poison, i32 %compare.local.base1431, i64 0, !dbg !16
  %518 = insertelement <4 x i32> %517, i32 %compare.local.base1435, i64 1, !dbg !16
  %519 = insertelement <4 x i32> %518, i32 %compare.local.base1439, i64 2, !dbg !16
  %520 = insertelement <4 x i32> %519, i32 %compare.local.base1443, i64 3, !dbg !16
  %521 = or <4 x i32> %520, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %522 = icmp ult <4 x i32> %521, %136, !dbg !16
  %523 = select <4 x i1> %500, <4 x i1> %522, <4 x i1> %502, !dbg !16
  %524 = select <4 x i1> %523, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %compare.local.base1455 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 32) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1459 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 33) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1463 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 34) #8, !dbg !16, !frisk.register_lifetimes !5
  %525 = insertelement <4 x i32> %457, i32 %compare.local.base1455, i64 1, !dbg !16
  %526 = insertelement <4 x i32> %525, i32 %compare.local.base1459, i64 2, !dbg !16
  %527 = insertelement <4 x i32> %526, i32 %compare.local.base1463, i64 3, !dbg !16
  %528 = or <4 x i32> %527, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %529 = icmp ugt <4 x i32> %134, %528, !dbg !16
  %compare.local.base1467 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 35) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1471 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 36) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1475 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 37) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1479 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 38) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1483 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 39) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1487 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 40) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1491 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 41) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1495 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 42) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1499 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 43) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1503 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 44) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1507 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 45) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1511 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 46) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1515 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 47) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1519 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 48) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1523 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 49) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1527 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 50) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1531 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 51) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1535 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 52) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1539 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 53) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1543 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 54) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1547 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 55) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1551 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 56) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1555 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 57) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1559 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 58) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1563 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 59) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1567 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 60) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1571 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 61) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1575 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 62) #8, !dbg !16, !frisk.register_lifetimes !5
  %530 = insertelement <4 x i32> poison, i32 %compare.local.base1547, i64 0, !dbg !16
  %531 = insertelement <4 x i32> %530, i32 %compare.local.base1551, i64 1, !dbg !16
  %532 = insertelement <4 x i32> %531, i32 %compare.local.base1555, i64 2, !dbg !16
  %533 = insertelement <4 x i32> %532, i32 %compare.local.base1559, i64 3, !dbg !16
  %534 = or <4 x i32> %533, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %535 = icmp ult <4 x i32> %534, %134, !dbg !16
  %536 = insertelement <4 x i32> poison, i32 %compare.local.base1531, i64 0, !dbg !16
  %537 = insertelement <4 x i32> %536, i32 %compare.local.base1535, i64 1, !dbg !16
  %538 = insertelement <4 x i32> %537, i32 %compare.local.base1539, i64 2, !dbg !16
  %539 = insertelement <4 x i32> %538, i32 %compare.local.base1543, i64 3, !dbg !16
  %540 = or <4 x i32> %539, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %541 = icmp ult <4 x i32> %540, %134, !dbg !16
  %542 = insertelement <4 x i32> poison, i32 %compare.local.base1515, i64 0, !dbg !16
  %543 = insertelement <4 x i32> %542, i32 %compare.local.base1519, i64 1, !dbg !16
  %544 = insertelement <4 x i32> %543, i32 %compare.local.base1523, i64 2, !dbg !16
  %545 = insertelement <4 x i32> %544, i32 %compare.local.base1527, i64 3, !dbg !16
  %546 = or <4 x i32> %545, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %547 = icmp ult <4 x i32> %546, %134, !dbg !16
  %548 = insertelement <4 x i32> poison, i32 %compare.local.base1499, i64 0, !dbg !16
  %549 = insertelement <4 x i32> %548, i32 %compare.local.base1503, i64 1, !dbg !16
  %550 = insertelement <4 x i32> %549, i32 %compare.local.base1507, i64 2, !dbg !16
  %551 = insertelement <4 x i32> %550, i32 %compare.local.base1511, i64 3, !dbg !16
  %552 = or <4 x i32> %551, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %553 = icmp ult <4 x i32> %552, %134, !dbg !16
  %554 = insertelement <4 x i32> poison, i32 %compare.local.base1483, i64 0, !dbg !16
  %555 = insertelement <4 x i32> %554, i32 %compare.local.base1487, i64 1, !dbg !16
  %556 = insertelement <4 x i32> %555, i32 %compare.local.base1491, i64 2, !dbg !16
  %557 = insertelement <4 x i32> %556, i32 %compare.local.base1495, i64 3, !dbg !16
  %558 = or <4 x i32> %557, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %559 = icmp ult <4 x i32> %558, %134, !dbg !16
  %560 = insertelement <4 x i32> poison, i32 %compare.local.base1467, i64 0, !dbg !16
  %561 = insertelement <4 x i32> %560, i32 %compare.local.base1471, i64 1, !dbg !16
  %562 = insertelement <4 x i32> %561, i32 %compare.local.base1475, i64 2, !dbg !16
  %563 = insertelement <4 x i32> %562, i32 %compare.local.base1479, i64 3, !dbg !16
  %564 = or <4 x i32> %563, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %565 = icmp ult <4 x i32> %564, %134, !dbg !16
  %compare.high.equal1449 = icmp eq i32 %compare.high1447, %compare.high1323, !dbg !16
  %compare.high.order1450 = icmp ugt i32 %compare.high1447, %compare.high1323, !dbg !16
  %566 = insertelement <4 x i1> poison, i1 %compare.high.equal1449, i64 0, !dbg !16
  %567 = shufflevector <4 x i1> %566, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %568 = insertelement <4 x i1> poison, i1 %compare.high.order1450, i64 0, !dbg !16
  %569 = shufflevector <4 x i1> %568, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %570 = select <4 x i1> %567, <4 x i1> %529, <4 x i1> %569, !dbg !16
  %571 = select <4 x i1> %570, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %572 = select <4 x i1> %567, <4 x i1> %565, <4 x i1> %569, !dbg !16
  %573 = select <4 x i1> %572, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %574 = select <4 x i1> %567, <4 x i1> %559, <4 x i1> %569, !dbg !16
  %575 = select <4 x i1> %574, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %576 = select <4 x i1> %567, <4 x i1> %553, <4 x i1> %569, !dbg !16
  %577 = select <4 x i1> %576, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %578 = select <4 x i1> %567, <4 x i1> %547, <4 x i1> %569, !dbg !16
  %579 = select <4 x i1> %578, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %580 = select <4 x i1> %567, <4 x i1> %541, <4 x i1> %569, !dbg !16
  %581 = select <4 x i1> %580, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %582 = select <4 x i1> %567, <4 x i1> %535, <4 x i1> %569, !dbg !16
  %583 = select <4 x i1> %582, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %584 = insertelement <4 x i32> poison, i32 %compare.local.base1563, i64 0, !dbg !16
  %585 = insertelement <4 x i32> %584, i32 %compare.local.base1567, i64 1, !dbg !16
  %586 = insertelement <4 x i32> %585, i32 %compare.local.base1571, i64 2, !dbg !16
  %587 = insertelement <4 x i32> %586, i32 %compare.local.base1575, i64 3, !dbg !16
  %588 = or <4 x i32> %587, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %589 = icmp ult <4 x i32> %588, %134, !dbg !16
  %590 = select <4 x i1> %567, <4 x i1> %589, <4 x i1> %569, !dbg !16
  %591 = select <4 x i1> %590, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %.masked127 = fadd <4 x float> %407, %504, !dbg !19
  %.masked129 = fadd <4 x float> %411, %506, !dbg !19
  %.masked131 = fadd <4 x float> %415, %508, !dbg !19
  %.masked133 = fadd <4 x float> %419, %510, !dbg !19
  %.masked135 = fadd <4 x float> %423, %512, !dbg !19
  %.masked137 = fadd <4 x float> %427, %514, !dbg !19
  %.masked139 = fadd <4 x float> %431, %516, !dbg !19
  %.masked141 = fadd <4 x float> %434, %524, !dbg !19
  %.masked144 = fadd <4 x float> %436, %571, !dbg !19
  %.masked146 = fadd <4 x float> %438, %573, !dbg !19
  %.masked148 = fadd <4 x float> %440, %575, !dbg !19
  %.masked150 = fadd <4 x float> %442, %577, !dbg !19
  %.masked152 = fadd <4 x float> %444, %579, !dbg !19
  %.masked154 = fadd <4 x float> %446, %581, !dbg !19
  %.masked156 = fadd <4 x float> %448, %583, !dbg !19
  %.masked158 = fadd <4 x float> %452, %591, !dbg !19
  br label %592, !dbg !12

592:                                              ; preds = %454, %137
  %593 = phi <4 x float> [ %.masked127, %454 ], [ %407, %137 ], !dbg !19
  %594 = phi <4 x float> [ %.masked129, %454 ], [ %411, %137 ], !dbg !19
  %595 = phi <4 x float> [ %.masked131, %454 ], [ %415, %137 ], !dbg !19
  %596 = phi <4 x float> [ %.masked133, %454 ], [ %419, %137 ], !dbg !19
  %597 = phi <4 x float> [ %.masked135, %454 ], [ %423, %137 ], !dbg !19
  %598 = phi <4 x float> [ %.masked137, %454 ], [ %427, %137 ], !dbg !19
  %599 = phi <4 x float> [ %.masked139, %454 ], [ %431, %137 ], !dbg !19
  %600 = phi <4 x float> [ %.masked141, %454 ], [ %434, %137 ], !dbg !19
  %601 = phi <4 x float> [ %.masked144, %454 ], [ %436, %137 ], !dbg !19
  %602 = phi <4 x float> [ %.masked146, %454 ], [ %438, %137 ], !dbg !19
  %603 = phi <4 x float> [ %.masked148, %454 ], [ %440, %137 ], !dbg !19
  %604 = phi <4 x float> [ %.masked150, %454 ], [ %442, %137 ], !dbg !19
  %605 = phi <4 x float> [ %.masked152, %454 ], [ %444, %137 ], !dbg !19
  %606 = phi <4 x float> [ %.masked154, %454 ], [ %446, %137 ], !dbg !19
  %607 = phi <4 x float> [ %.masked156, %454 ], [ %448, %137 ], !dbg !19
  %608 = phi <4 x float> [ %.masked158, %454 ], [ %452, %137 ], !dbg !19
  %609 = extractelement <4 x float> %593, i64 0, !dbg !19
  %610 = extractelement <4 x float> %593, i64 1, !dbg !19
  %611 = extractelement <4 x float> %593, i64 2, !dbg !19
  %612 = extractelement <4 x float> %593, i64 3, !dbg !19
  %613 = extractelement <4 x float> %594, i64 0, !dbg !19
  %614 = extractelement <4 x float> %594, i64 1, !dbg !19
  %615 = extractelement <4 x float> %594, i64 2, !dbg !19
  %616 = extractelement <4 x float> %594, i64 3, !dbg !19
  %617 = extractelement <4 x float> %595, i64 0, !dbg !19
  %618 = extractelement <4 x float> %595, i64 1, !dbg !19
  %619 = extractelement <4 x float> %595, i64 2, !dbg !19
  %620 = extractelement <4 x float> %595, i64 3, !dbg !19
  %621 = extractelement <4 x float> %596, i64 0, !dbg !19
  %622 = extractelement <4 x float> %596, i64 1, !dbg !19
  %623 = extractelement <4 x float> %596, i64 2, !dbg !19
  %624 = extractelement <4 x float> %596, i64 3, !dbg !19
  %625 = extractelement <4 x float> %597, i64 0, !dbg !19
  %626 = extractelement <4 x float> %597, i64 1, !dbg !19
  %627 = extractelement <4 x float> %597, i64 2, !dbg !19
  %628 = extractelement <4 x float> %597, i64 3, !dbg !19
  %629 = extractelement <4 x float> %598, i64 0, !dbg !19
  %630 = extractelement <4 x float> %598, i64 1, !dbg !19
  %631 = extractelement <4 x float> %598, i64 2, !dbg !19
  %632 = extractelement <4 x float> %598, i64 3, !dbg !19
  %633 = extractelement <4 x float> %599, i64 0, !dbg !19
  %634 = extractelement <4 x float> %599, i64 1, !dbg !19
  %635 = extractelement <4 x float> %599, i64 2, !dbg !19
  %636 = extractelement <4 x float> %599, i64 3, !dbg !19
  %637 = extractelement <4 x float> %600, i64 0, !dbg !19
  %638 = extractelement <4 x float> %600, i64 1, !dbg !19
  %639 = extractelement <4 x float> %600, i64 2, !dbg !19
  %640 = extractelement <4 x float> %600, i64 3, !dbg !19
  %641 = extractelement <4 x float> %601, i64 0, !dbg !19
  %642 = extractelement <4 x float> %601, i64 1, !dbg !19
  %643 = extractelement <4 x float> %601, i64 2, !dbg !19
  %644 = extractelement <4 x float> %601, i64 3, !dbg !19
  %645 = extractelement <4 x float> %602, i64 0, !dbg !19
  %646 = extractelement <4 x float> %602, i64 1, !dbg !19
  %647 = extractelement <4 x float> %602, i64 2, !dbg !19
  %648 = extractelement <4 x float> %602, i64 3, !dbg !19
  %649 = extractelement <4 x float> %603, i64 0, !dbg !19
  %650 = extractelement <4 x float> %603, i64 1, !dbg !19
  %651 = extractelement <4 x float> %603, i64 2, !dbg !19
  %652 = extractelement <4 x float> %603, i64 3, !dbg !19
  %653 = extractelement <4 x float> %604, i64 0, !dbg !19
  %654 = extractelement <4 x float> %604, i64 1, !dbg !19
  %655 = extractelement <4 x float> %604, i64 2, !dbg !19
  %656 = extractelement <4 x float> %604, i64 3, !dbg !19
  %657 = extractelement <4 x float> %605, i64 0, !dbg !19
  %658 = extractelement <4 x float> %605, i64 1, !dbg !19
  %659 = extractelement <4 x float> %605, i64 2, !dbg !19
  %660 = extractelement <4 x float> %605, i64 3, !dbg !19
  %661 = extractelement <4 x float> %606, i64 0, !dbg !19
  %662 = extractelement <4 x float> %606, i64 1, !dbg !19
  %663 = extractelement <4 x float> %606, i64 2, !dbg !19
  %664 = extractelement <4 x float> %606, i64 3, !dbg !19
  %665 = extractelement <4 x float> %607, i64 0, !dbg !19
  %666 = extractelement <4 x float> %607, i64 1, !dbg !19
  %667 = extractelement <4 x float> %607, i64 2, !dbg !19
  %668 = extractelement <4 x float> %607, i64 3, !dbg !19
  %669 = extractelement <4 x float> %608, i64 0, !dbg !19
  %670 = extractelement <4 x float> %608, i64 1, !dbg !19
  %671 = extractelement <4 x float> %608, i64 2, !dbg !19
  %672 = extractelement <4 x float> %608, i64 3, !dbg !19
  %673 = tail call float @llvm.maxnum.f32(float %609, float -inf), !dbg !20, !frisk.local_reduction !5
  %local.reduce251 = tail call float @llvm.maxnum.f32(float %673, float %613), !dbg !20, !frisk.local_reduction !5
  %local.reduce252 = tail call float @llvm.maxnum.f32(float %610, float %614), !dbg !20, !frisk.local_reduction !5
  %local.reduce253 = tail call float @llvm.maxnum.f32(float %611, float %615), !dbg !20, !frisk.local_reduction !5
  %local.reduce254 = tail call float @llvm.maxnum.f32(float %612, float %616), !dbg !20, !frisk.local_reduction !5
  %local.reduce255 = tail call float @llvm.maxnum.f32(float %local.reduce251, float %617), !dbg !20, !frisk.local_reduction !5
  %local.reduce256 = tail call float @llvm.maxnum.f32(float %local.reduce252, float %618), !dbg !20, !frisk.local_reduction !5
  %local.reduce257 = tail call float @llvm.maxnum.f32(float %local.reduce253, float %619), !dbg !20, !frisk.local_reduction !5
  %local.reduce258 = tail call float @llvm.maxnum.f32(float %local.reduce254, float %620), !dbg !20, !frisk.local_reduction !5
  %local.reduce259 = tail call float @llvm.maxnum.f32(float %local.reduce255, float %621), !dbg !20, !frisk.local_reduction !5
  %local.reduce260 = tail call float @llvm.maxnum.f32(float %local.reduce256, float %622), !dbg !20, !frisk.local_reduction !5
  %local.reduce261 = tail call float @llvm.maxnum.f32(float %local.reduce257, float %623), !dbg !20, !frisk.local_reduction !5
  %local.reduce262 = tail call float @llvm.maxnum.f32(float %local.reduce258, float %624), !dbg !20, !frisk.local_reduction !5
  %local.reduce263 = tail call float @llvm.maxnum.f32(float %local.reduce259, float %625), !dbg !20, !frisk.local_reduction !5
  %local.reduce264 = tail call float @llvm.maxnum.f32(float %local.reduce260, float %626), !dbg !20, !frisk.local_reduction !5
  %local.reduce265 = tail call float @llvm.maxnum.f32(float %local.reduce261, float %627), !dbg !20, !frisk.local_reduction !5
  %local.reduce266 = tail call float @llvm.maxnum.f32(float %local.reduce262, float %628), !dbg !20, !frisk.local_reduction !5
  %local.reduce267 = tail call float @llvm.maxnum.f32(float %local.reduce263, float %629), !dbg !20, !frisk.local_reduction !5
  %local.reduce268 = tail call float @llvm.maxnum.f32(float %local.reduce264, float %630), !dbg !20, !frisk.local_reduction !5
  %local.reduce269 = tail call float @llvm.maxnum.f32(float %local.reduce265, float %631), !dbg !20, !frisk.local_reduction !5
  %local.reduce270 = tail call float @llvm.maxnum.f32(float %local.reduce266, float %632), !dbg !20, !frisk.local_reduction !5
  %local.reduce271 = tail call float @llvm.maxnum.f32(float %local.reduce267, float %633), !dbg !20, !frisk.local_reduction !5
  %local.reduce272 = tail call float @llvm.maxnum.f32(float %local.reduce268, float %634), !dbg !20, !frisk.local_reduction !5
  %local.reduce273 = tail call float @llvm.maxnum.f32(float %local.reduce269, float %635), !dbg !20, !frisk.local_reduction !5
  %local.reduce274 = tail call float @llvm.maxnum.f32(float %local.reduce270, float %636), !dbg !20, !frisk.local_reduction !5
  %local.reduce275 = tail call float @llvm.maxnum.f32(float %local.reduce271, float %637), !dbg !20, !frisk.local_reduction !5
  %local.reduce276 = tail call float @llvm.maxnum.f32(float %local.reduce272, float %638), !dbg !20, !frisk.local_reduction !5
  %local.reduce277 = tail call float @llvm.maxnum.f32(float %local.reduce273, float %639), !dbg !20, !frisk.local_reduction !5
  %local.reduce278 = tail call float @llvm.maxnum.f32(float %local.reduce274, float %640), !dbg !20, !frisk.local_reduction !5
  %local.reduce279 = tail call float @llvm.maxnum.f32(float %local.reduce275, float %local.reduce276), !dbg !20, !frisk.local_reduction !5
  %local.reduce280 = tail call float @llvm.maxnum.f32(float %local.reduce277, float %local.reduce278), !dbg !20, !frisk.local_reduction !5
  %674 = tail call float @llvm.maxnum.f32(float %local.reduce279, float %local.reduce280), !dbg !20, !frisk.local_reduction !5
  %675 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %676 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %675), !dbg !20
  %677 = add nuw nsw i32 %676, 64, !dbg !20
  %678 = xor i32 %676, 16, !dbg !20
  %679 = and i32 %677, 192, !dbg !20
  %680 = icmp samesign ult i32 %678, %679, !dbg !20
  %681 = select i1 %680, i32 %678, i32 %676, !dbg !20
  %682 = shl nuw nsw i32 %681, 2, !dbg !20
  %683 = bitcast float %674 to i32, !dbg !20
  %684 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %682, i32 %683), !dbg !20
  %685 = bitcast i32 %684 to float, !dbg !20
  %686 = tail call float @llvm.maxnum.f32(float %674, float %685), !dbg !20
  %687 = xor i32 %676, 32, !dbg !20
  %688 = icmp samesign ult i32 %687, %679, !dbg !20
  %689 = select i1 %688, i32 %687, i32 %676, !dbg !20
  %690 = shl nuw nsw i32 %689, 2, !dbg !20
  %691 = bitcast float %686 to i32, !dbg !20
  %692 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %690, i32 %691), !dbg !20
  %693 = bitcast i32 %692 to float, !dbg !20
  %694 = tail call float @llvm.maxnum.f32(float %686, float %693), !dbg !20
  %695 = extractvalue [2 x <1 x float>] %141, 0, !dbg !20
  %696 = insertelement <1 x float> poison, float %694, i64 0, !dbg !20
  %697 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %696, <1 x float> %695), !dbg !20
  %698 = tail call float @llvm.maxnum.f32(float %641, float -inf), !dbg !20, !frisk.local_reduction !5
  %local.reduce220 = tail call float @llvm.maxnum.f32(float %698, float %645), !dbg !20, !frisk.local_reduction !5
  %local.reduce221 = tail call float @llvm.maxnum.f32(float %642, float %646), !dbg !20, !frisk.local_reduction !5
  %local.reduce222 = tail call float @llvm.maxnum.f32(float %643, float %647), !dbg !20, !frisk.local_reduction !5
  %local.reduce223 = tail call float @llvm.maxnum.f32(float %644, float %648), !dbg !20, !frisk.local_reduction !5
  %local.reduce224 = tail call float @llvm.maxnum.f32(float %local.reduce220, float %649), !dbg !20, !frisk.local_reduction !5
  %local.reduce225 = tail call float @llvm.maxnum.f32(float %local.reduce221, float %650), !dbg !20, !frisk.local_reduction !5
  %local.reduce226 = tail call float @llvm.maxnum.f32(float %local.reduce222, float %651), !dbg !20, !frisk.local_reduction !5
  %local.reduce227 = tail call float @llvm.maxnum.f32(float %local.reduce223, float %652), !dbg !20, !frisk.local_reduction !5
  %local.reduce228 = tail call float @llvm.maxnum.f32(float %local.reduce224, float %653), !dbg !20, !frisk.local_reduction !5
  %local.reduce229 = tail call float @llvm.maxnum.f32(float %local.reduce225, float %654), !dbg !20, !frisk.local_reduction !5
  %local.reduce230 = tail call float @llvm.maxnum.f32(float %local.reduce226, float %655), !dbg !20, !frisk.local_reduction !5
  %local.reduce231 = tail call float @llvm.maxnum.f32(float %local.reduce227, float %656), !dbg !20, !frisk.local_reduction !5
  %local.reduce232 = tail call float @llvm.maxnum.f32(float %local.reduce228, float %657), !dbg !20, !frisk.local_reduction !5
  %local.reduce233 = tail call float @llvm.maxnum.f32(float %local.reduce229, float %658), !dbg !20, !frisk.local_reduction !5
  %local.reduce234 = tail call float @llvm.maxnum.f32(float %local.reduce230, float %659), !dbg !20, !frisk.local_reduction !5
  %local.reduce235 = tail call float @llvm.maxnum.f32(float %local.reduce231, float %660), !dbg !20, !frisk.local_reduction !5
  %local.reduce236 = tail call float @llvm.maxnum.f32(float %local.reduce232, float %661), !dbg !20, !frisk.local_reduction !5
  %local.reduce237 = tail call float @llvm.maxnum.f32(float %local.reduce233, float %662), !dbg !20, !frisk.local_reduction !5
  %local.reduce238 = tail call float @llvm.maxnum.f32(float %local.reduce234, float %663), !dbg !20, !frisk.local_reduction !5
  %local.reduce239 = tail call float @llvm.maxnum.f32(float %local.reduce235, float %664), !dbg !20, !frisk.local_reduction !5
  %local.reduce240 = tail call float @llvm.maxnum.f32(float %local.reduce236, float %665), !dbg !20, !frisk.local_reduction !5
  %local.reduce241 = tail call float @llvm.maxnum.f32(float %local.reduce237, float %666), !dbg !20, !frisk.local_reduction !5
  %local.reduce242 = tail call float @llvm.maxnum.f32(float %local.reduce238, float %667), !dbg !20, !frisk.local_reduction !5
  %local.reduce243 = tail call float @llvm.maxnum.f32(float %local.reduce239, float %668), !dbg !20, !frisk.local_reduction !5
  %local.reduce244 = tail call float @llvm.maxnum.f32(float %local.reduce240, float %669), !dbg !20, !frisk.local_reduction !5
  %local.reduce245 = tail call float @llvm.maxnum.f32(float %local.reduce241, float %670), !dbg !20, !frisk.local_reduction !5
  %local.reduce246 = tail call float @llvm.maxnum.f32(float %local.reduce242, float %671), !dbg !20, !frisk.local_reduction !5
  %local.reduce247 = tail call float @llvm.maxnum.f32(float %local.reduce243, float %672), !dbg !20, !frisk.local_reduction !5
  %local.reduce248 = tail call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !20, !frisk.local_reduction !5
  %local.reduce249 = tail call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !20, !frisk.local_reduction !5
  %699 = tail call float @llvm.maxnum.f32(float %local.reduce248, float %local.reduce249), !dbg !20, !frisk.local_reduction !5
  %700 = bitcast float %699 to i32, !dbg !20
  %701 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %682, i32 %700), !dbg !20
  %702 = bitcast i32 %701 to float, !dbg !20
  %703 = tail call float @llvm.maxnum.f32(float %699, float %702), !dbg !20
  %704 = bitcast float %703 to i32, !dbg !20
  %705 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %690, i32 %704), !dbg !20
  %706 = bitcast i32 %705 to float, !dbg !20
  %707 = tail call float @llvm.maxnum.f32(float %703, float %706), !dbg !20
  %708 = extractvalue [2 x <1 x float>] %141, 1, !dbg !20
  %709 = insertelement <1 x float> poison, float %707, i64 0, !dbg !20
  %710 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %709, <1 x float> %708), !dbg !20
  %711 = shufflevector <1 x float> %697, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %712 = fsub <4 x float> %593, %711, !dbg !21
  %713 = fsub <4 x float> %594, %711, !dbg !21
  %714 = fsub <4 x float> %595, %711, !dbg !21
  %715 = fsub <4 x float> %596, %711, !dbg !21
  %716 = fsub <4 x float> %597, %711, !dbg !21
  %717 = fsub <4 x float> %598, %711, !dbg !21
  %718 = fsub <4 x float> %599, %711, !dbg !21
  %719 = fsub <4 x float> %600, %711, !dbg !21
  %720 = shufflevector <1 x float> %710, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %721 = fsub <4 x float> %601, %720, !dbg !21
  %722 = fsub <4 x float> %602, %720, !dbg !21
  %723 = fsub <4 x float> %603, %720, !dbg !21
  %724 = fsub <4 x float> %604, %720, !dbg !21
  %725 = fsub <4 x float> %605, %720, !dbg !21
  %726 = fsub <4 x float> %606, %720, !dbg !21
  %727 = fsub <4 x float> %607, %720, !dbg !21
  %728 = fsub <4 x float> %608, %720, !dbg !21
  %729 = fsub <1 x float> %695, %697, !dbg !22
  %730 = extractelement <1 x float> %729, i64 0, !dbg !23
  %731 = tail call float @llvm.amdgcn.exp2.f32(float %730), !dbg !23
  %732 = insertelement <4 x float> poison, float %731, i64 0, !dbg !24
  %733 = shufflevector <4 x float> %732, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %734 = extractvalue [2 x <1 x float>] %142, 0, !dbg !25
  %735 = extractelement <1 x float> %734, i64 0, !dbg !25
  %736 = fmul float %735, %731, !dbg !25
  %737 = fsub <1 x float> %708, %710, !dbg !22
  %738 = extractelement <1 x float> %737, i64 0, !dbg !23
  %739 = tail call float @llvm.amdgcn.exp2.f32(float %738), !dbg !23
  %740 = insertelement <4 x float> poison, float %739, i64 0, !dbg !24
  %741 = shufflevector <4 x float> %740, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %742 = extractvalue [2 x <1 x float>] %142, 1, !dbg !25
  %743 = extractelement <1 x float> %742, i64 0, !dbg !25
  %744 = fmul float %743, %739, !dbg !25
  %745 = extractelement <4 x float> %712, i64 0, !dbg !26
  %746 = tail call float @llvm.amdgcn.exp2.f32(float %745), !dbg !26
  %747 = insertelement <4 x float> poison, float %746, i64 0, !dbg !27
  %748 = extractelement <4 x float> %712, i64 1, !dbg !26
  %749 = tail call float @llvm.amdgcn.exp2.f32(float %748), !dbg !26
  %750 = insertelement <4 x float> %747, float %749, i64 1, !dbg !27
  %751 = extractelement <4 x float> %712, i64 2, !dbg !26
  %752 = tail call float @llvm.amdgcn.exp2.f32(float %751), !dbg !26
  %753 = insertelement <4 x float> %750, float %752, i64 2, !dbg !27
  %754 = extractelement <4 x float> %712, i64 3, !dbg !26
  %755 = tail call float @llvm.amdgcn.exp2.f32(float %754), !dbg !26
  %756 = insertelement <4 x float> %753, float %755, i64 3, !dbg !27
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %756) #8, !dbg !27, !frisk.register_lifetimes !5
  %757 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %757) #8, !dbg !26, !frisk.register_lifetimes !5
  %758 = extractelement <4 x float> %713, i64 0, !dbg !26
  %759 = tail call float @llvm.amdgcn.exp2.f32(float %758), !dbg !26
  %760 = insertelement <4 x float> poison, float %759, i64 0, !dbg !27
  %local.reduce189 = fadd float %746, %759, !dbg !28, !frisk.local_reduction !5
  %761 = extractelement <4 x float> %713, i64 1, !dbg !26
  %762 = tail call float @llvm.amdgcn.exp2.f32(float %761), !dbg !26
  %763 = insertelement <4 x float> %760, float %762, i64 1, !dbg !27
  %local.reduce190 = fadd float %749, %762, !dbg !28, !frisk.local_reduction !5
  %764 = extractelement <4 x float> %713, i64 2, !dbg !26
  %765 = tail call float @llvm.amdgcn.exp2.f32(float %764), !dbg !26
  %766 = insertelement <4 x float> %763, float %765, i64 2, !dbg !27
  %local.reduce191 = fadd float %752, %765, !dbg !28, !frisk.local_reduction !5
  %767 = extractelement <4 x float> %713, i64 3, !dbg !26
  %768 = tail call float @llvm.amdgcn.exp2.f32(float %767), !dbg !26
  %769 = insertelement <4 x float> %766, float %768, i64 3, !dbg !27
  %math.result475 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %769) #8, !dbg !27, !frisk.register_lifetimes !5
  %770 = fptrunc <4 x float> %math.result475 to <4 x half>, !dbg !27
  %math.packed476 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %770) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce192 = fadd float %755, %768, !dbg !28, !frisk.local_reduction !5
  %771 = extractelement <4 x float> %714, i64 0, !dbg !26
  %772 = tail call float @llvm.amdgcn.exp2.f32(float %771), !dbg !26
  %773 = insertelement <4 x float> poison, float %772, i64 0, !dbg !27
  %local.reduce193 = fadd float %772, %local.reduce189, !dbg !28, !frisk.local_reduction !5
  %774 = extractelement <4 x float> %714, i64 1, !dbg !26
  %775 = tail call float @llvm.amdgcn.exp2.f32(float %774), !dbg !26
  %776 = insertelement <4 x float> %773, float %775, i64 1, !dbg !27
  %local.reduce194 = fadd float %775, %local.reduce190, !dbg !28, !frisk.local_reduction !5
  %777 = extractelement <4 x float> %714, i64 2, !dbg !26
  %778 = tail call float @llvm.amdgcn.exp2.f32(float %777), !dbg !26
  %779 = insertelement <4 x float> %776, float %778, i64 2, !dbg !27
  %local.reduce195 = fadd float %778, %local.reduce191, !dbg !28, !frisk.local_reduction !5
  %780 = extractelement <4 x float> %714, i64 3, !dbg !26
  %781 = tail call float @llvm.amdgcn.exp2.f32(float %780), !dbg !26
  %782 = insertelement <4 x float> %779, float %781, i64 3, !dbg !27
  %math.result477 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %782) #8, !dbg !27, !frisk.register_lifetimes !5
  %783 = fptrunc <4 x float> %math.result477 to <4 x half>, !dbg !27
  %math.packed478 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %783) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce196 = fadd float %781, %local.reduce192, !dbg !28, !frisk.local_reduction !5
  %784 = extractelement <4 x float> %715, i64 0, !dbg !26
  %785 = tail call float @llvm.amdgcn.exp2.f32(float %784), !dbg !26
  %786 = insertelement <4 x float> poison, float %785, i64 0, !dbg !27
  %local.reduce197 = fadd float %785, %local.reduce193, !dbg !28, !frisk.local_reduction !5
  %787 = extractelement <4 x float> %715, i64 1, !dbg !26
  %788 = tail call float @llvm.amdgcn.exp2.f32(float %787), !dbg !26
  %789 = insertelement <4 x float> %786, float %788, i64 1, !dbg !27
  %local.reduce198 = fadd float %788, %local.reduce194, !dbg !28, !frisk.local_reduction !5
  %790 = extractelement <4 x float> %715, i64 2, !dbg !26
  %791 = tail call float @llvm.amdgcn.exp2.f32(float %790), !dbg !26
  %792 = insertelement <4 x float> %789, float %791, i64 2, !dbg !27
  %local.reduce199 = fadd float %791, %local.reduce195, !dbg !28, !frisk.local_reduction !5
  %793 = extractelement <4 x float> %715, i64 3, !dbg !26
  %794 = tail call float @llvm.amdgcn.exp2.f32(float %793), !dbg !26
  %795 = insertelement <4 x float> %792, float %794, i64 3, !dbg !27
  %math.result479 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %795) #8, !dbg !27, !frisk.register_lifetimes !5
  %796 = fptrunc <4 x float> %math.result479 to <4 x half>, !dbg !27
  %math.packed480 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %796) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce200 = fadd float %794, %local.reduce196, !dbg !28, !frisk.local_reduction !5
  %797 = extractelement <4 x float> %716, i64 0, !dbg !26
  %798 = tail call float @llvm.amdgcn.exp2.f32(float %797), !dbg !26
  %799 = insertelement <4 x float> poison, float %798, i64 0, !dbg !27
  %local.reduce201 = fadd float %798, %local.reduce197, !dbg !28, !frisk.local_reduction !5
  %800 = extractelement <4 x float> %716, i64 1, !dbg !26
  %801 = tail call float @llvm.amdgcn.exp2.f32(float %800), !dbg !26
  %802 = insertelement <4 x float> %799, float %801, i64 1, !dbg !27
  %local.reduce202 = fadd float %801, %local.reduce198, !dbg !28, !frisk.local_reduction !5
  %803 = extractelement <4 x float> %716, i64 2, !dbg !26
  %804 = tail call float @llvm.amdgcn.exp2.f32(float %803), !dbg !26
  %805 = insertelement <4 x float> %802, float %804, i64 2, !dbg !27
  %local.reduce203 = fadd float %804, %local.reduce199, !dbg !28, !frisk.local_reduction !5
  %806 = extractelement <4 x float> %716, i64 3, !dbg !26
  %807 = tail call float @llvm.amdgcn.exp2.f32(float %806), !dbg !26
  %808 = insertelement <4 x float> %805, float %807, i64 3, !dbg !27
  %math.result481 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %808) #8, !dbg !27, !frisk.register_lifetimes !5
  %809 = fptrunc <4 x float> %math.result481 to <4 x half>, !dbg !27
  %math.packed482 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %809) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce204 = fadd float %807, %local.reduce200, !dbg !28, !frisk.local_reduction !5
  %810 = extractelement <4 x float> %717, i64 0, !dbg !26
  %811 = tail call float @llvm.amdgcn.exp2.f32(float %810), !dbg !26
  %812 = insertelement <4 x float> poison, float %811, i64 0, !dbg !27
  %local.reduce205 = fadd float %811, %local.reduce201, !dbg !28, !frisk.local_reduction !5
  %813 = extractelement <4 x float> %717, i64 1, !dbg !26
  %814 = tail call float @llvm.amdgcn.exp2.f32(float %813), !dbg !26
  %815 = insertelement <4 x float> %812, float %814, i64 1, !dbg !27
  %local.reduce206 = fadd float %814, %local.reduce202, !dbg !28, !frisk.local_reduction !5
  %816 = extractelement <4 x float> %717, i64 2, !dbg !26
  %817 = tail call float @llvm.amdgcn.exp2.f32(float %816), !dbg !26
  %818 = insertelement <4 x float> %815, float %817, i64 2, !dbg !27
  %local.reduce207 = fadd float %817, %local.reduce203, !dbg !28, !frisk.local_reduction !5
  %819 = extractelement <4 x float> %717, i64 3, !dbg !26
  %820 = tail call float @llvm.amdgcn.exp2.f32(float %819), !dbg !26
  %821 = insertelement <4 x float> %818, float %820, i64 3, !dbg !27
  %math.result483 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %821) #8, !dbg !27, !frisk.register_lifetimes !5
  %822 = fptrunc <4 x float> %math.result483 to <4 x half>, !dbg !27
  %math.packed484 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %822) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce208 = fadd float %820, %local.reduce204, !dbg !28, !frisk.local_reduction !5
  %823 = extractelement <4 x float> %718, i64 0, !dbg !26
  %824 = tail call float @llvm.amdgcn.exp2.f32(float %823), !dbg !26
  %825 = insertelement <4 x float> poison, float %824, i64 0, !dbg !27
  %local.reduce209 = fadd float %824, %local.reduce205, !dbg !28, !frisk.local_reduction !5
  %826 = extractelement <4 x float> %718, i64 1, !dbg !26
  %827 = tail call float @llvm.amdgcn.exp2.f32(float %826), !dbg !26
  %828 = insertelement <4 x float> %825, float %827, i64 1, !dbg !27
  %local.reduce210 = fadd float %827, %local.reduce206, !dbg !28, !frisk.local_reduction !5
  %829 = extractelement <4 x float> %718, i64 2, !dbg !26
  %830 = tail call float @llvm.amdgcn.exp2.f32(float %829), !dbg !26
  %831 = insertelement <4 x float> %828, float %830, i64 2, !dbg !27
  %local.reduce211 = fadd float %830, %local.reduce207, !dbg !28, !frisk.local_reduction !5
  %832 = extractelement <4 x float> %718, i64 3, !dbg !26
  %833 = tail call float @llvm.amdgcn.exp2.f32(float %832), !dbg !26
  %834 = insertelement <4 x float> %831, float %833, i64 3, !dbg !27
  %math.result485 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %834) #8, !dbg !27, !frisk.register_lifetimes !5
  %835 = fptrunc <4 x float> %math.result485 to <4 x half>, !dbg !27
  %math.packed486 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %835) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce212 = fadd float %833, %local.reduce208, !dbg !28, !frisk.local_reduction !5
  %836 = extractelement <4 x float> %719, i64 0, !dbg !26
  %837 = tail call float @llvm.amdgcn.exp2.f32(float %836), !dbg !26
  %838 = insertelement <4 x float> poison, float %837, i64 0, !dbg !27
  %local.reduce213 = fadd float %837, %local.reduce209, !dbg !28, !frisk.local_reduction !5
  %839 = extractelement <4 x float> %719, i64 1, !dbg !26
  %840 = tail call float @llvm.amdgcn.exp2.f32(float %839), !dbg !26
  %841 = insertelement <4 x float> %838, float %840, i64 1, !dbg !27
  %local.reduce214 = fadd float %840, %local.reduce210, !dbg !28, !frisk.local_reduction !5
  %local.reduce217 = fadd float %local.reduce213, %local.reduce214, !dbg !28, !frisk.local_reduction !5
  %842 = extractelement <4 x float> %719, i64 2, !dbg !26
  %843 = tail call float @llvm.amdgcn.exp2.f32(float %842), !dbg !26
  %844 = insertelement <4 x float> %841, float %843, i64 2, !dbg !27
  %local.reduce215 = fadd float %843, %local.reduce211, !dbg !28, !frisk.local_reduction !5
  %845 = extractelement <4 x float> %719, i64 3, !dbg !26
  %846 = tail call float @llvm.amdgcn.exp2.f32(float %845), !dbg !26
  %847 = insertelement <4 x float> %844, float %846, i64 3, !dbg !27
  %math.result487 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %847) #8, !dbg !27, !frisk.register_lifetimes !5
  %848 = fptrunc <4 x float> %math.result487 to <4 x half>, !dbg !27
  %math.packed488 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %848) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce216 = fadd float %846, %local.reduce212, !dbg !28, !frisk.local_reduction !5
  %local.reduce218 = fadd float %local.reduce215, %local.reduce216, !dbg !28, !frisk.local_reduction !5
  %849 = fadd float %local.reduce217, %local.reduce218, !dbg !28, !frisk.local_reduction !5
  %850 = fadd float %736, %849, !dbg !28
  %851 = insertelement <1 x float> poison, float %850, i64 0, !dbg !11
  %852 = insertvalue [2 x <1 x float>] poison, <1 x float> %851, 0, !dbg !11
  %853 = extractelement <4 x float> %721, i64 0, !dbg !26
  %854 = tail call float @llvm.amdgcn.exp2.f32(float %853), !dbg !26
  %855 = insertelement <4 x float> poison, float %854, i64 0, !dbg !27
  %856 = extractelement <4 x float> %721, i64 1, !dbg !26
  %857 = tail call float @llvm.amdgcn.exp2.f32(float %856), !dbg !26
  %858 = insertelement <4 x float> %855, float %857, i64 1, !dbg !27
  %859 = extractelement <4 x float> %721, i64 2, !dbg !26
  %860 = tail call float @llvm.amdgcn.exp2.f32(float %859), !dbg !26
  %861 = insertelement <4 x float> %858, float %860, i64 2, !dbg !27
  %862 = extractelement <4 x float> %721, i64 3, !dbg !26
  %863 = tail call float @llvm.amdgcn.exp2.f32(float %862), !dbg !26
  %864 = insertelement <4 x float> %861, float %863, i64 3, !dbg !27
  %math.result489 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %864) #8, !dbg !27, !frisk.register_lifetimes !5
  %865 = fptrunc <4 x float> %math.result489 to <4 x half>, !dbg !27
  %math.packed490 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %865) #8, !dbg !26, !frisk.register_lifetimes !5
  %866 = extractelement <4 x float> %722, i64 0, !dbg !26
  %867 = tail call float @llvm.amdgcn.exp2.f32(float %866), !dbg !26
  %868 = insertelement <4 x float> poison, float %867, i64 0, !dbg !27
  %local.reduce = fadd float %854, %867, !dbg !28, !frisk.local_reduction !5
  %869 = extractelement <4 x float> %722, i64 1, !dbg !26
  %870 = tail call float @llvm.amdgcn.exp2.f32(float %869), !dbg !26
  %871 = insertelement <4 x float> %868, float %870, i64 1, !dbg !27
  %local.reduce159 = fadd float %857, %870, !dbg !28, !frisk.local_reduction !5
  %872 = extractelement <4 x float> %722, i64 2, !dbg !26
  %873 = tail call float @llvm.amdgcn.exp2.f32(float %872), !dbg !26
  %874 = insertelement <4 x float> %871, float %873, i64 2, !dbg !27
  %local.reduce160 = fadd float %860, %873, !dbg !28, !frisk.local_reduction !5
  %875 = extractelement <4 x float> %722, i64 3, !dbg !26
  %876 = tail call float @llvm.amdgcn.exp2.f32(float %875), !dbg !26
  %877 = insertelement <4 x float> %874, float %876, i64 3, !dbg !27
  %math.result491 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %877) #8, !dbg !27, !frisk.register_lifetimes !5
  %878 = fptrunc <4 x float> %math.result491 to <4 x half>, !dbg !27
  %math.packed492 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %878) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce161 = fadd float %863, %876, !dbg !28, !frisk.local_reduction !5
  %879 = extractelement <4 x float> %723, i64 0, !dbg !26
  %880 = tail call float @llvm.amdgcn.exp2.f32(float %879), !dbg !26
  %881 = insertelement <4 x float> poison, float %880, i64 0, !dbg !27
  %local.reduce162 = fadd float %880, %local.reduce, !dbg !28, !frisk.local_reduction !5
  %882 = extractelement <4 x float> %723, i64 1, !dbg !26
  %883 = tail call float @llvm.amdgcn.exp2.f32(float %882), !dbg !26
  %884 = insertelement <4 x float> %881, float %883, i64 1, !dbg !27
  %local.reduce163 = fadd float %883, %local.reduce159, !dbg !28, !frisk.local_reduction !5
  %885 = extractelement <4 x float> %723, i64 2, !dbg !26
  %886 = tail call float @llvm.amdgcn.exp2.f32(float %885), !dbg !26
  %887 = insertelement <4 x float> %884, float %886, i64 2, !dbg !27
  %local.reduce164 = fadd float %886, %local.reduce160, !dbg !28, !frisk.local_reduction !5
  %888 = extractelement <4 x float> %723, i64 3, !dbg !26
  %889 = tail call float @llvm.amdgcn.exp2.f32(float %888), !dbg !26
  %890 = insertelement <4 x float> %887, float %889, i64 3, !dbg !27
  %math.result493 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %890) #8, !dbg !27, !frisk.register_lifetimes !5
  %891 = fptrunc <4 x float> %math.result493 to <4 x half>, !dbg !27
  %math.packed494 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %891) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce165 = fadd float %889, %local.reduce161, !dbg !28, !frisk.local_reduction !5
  %892 = extractelement <4 x float> %724, i64 0, !dbg !26
  %893 = tail call float @llvm.amdgcn.exp2.f32(float %892), !dbg !26
  %894 = insertelement <4 x float> poison, float %893, i64 0, !dbg !27
  %local.reduce166 = fadd float %893, %local.reduce162, !dbg !28, !frisk.local_reduction !5
  %895 = extractelement <4 x float> %724, i64 1, !dbg !26
  %896 = tail call float @llvm.amdgcn.exp2.f32(float %895), !dbg !26
  %897 = insertelement <4 x float> %894, float %896, i64 1, !dbg !27
  %local.reduce167 = fadd float %896, %local.reduce163, !dbg !28, !frisk.local_reduction !5
  %898 = extractelement <4 x float> %724, i64 2, !dbg !26
  %899 = tail call float @llvm.amdgcn.exp2.f32(float %898), !dbg !26
  %900 = insertelement <4 x float> %897, float %899, i64 2, !dbg !27
  %local.reduce168 = fadd float %899, %local.reduce164, !dbg !28, !frisk.local_reduction !5
  %901 = extractelement <4 x float> %724, i64 3, !dbg !26
  %902 = tail call float @llvm.amdgcn.exp2.f32(float %901), !dbg !26
  %903 = insertelement <4 x float> %900, float %902, i64 3, !dbg !27
  %math.result495 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %903) #8, !dbg !27, !frisk.register_lifetimes !5
  %904 = fptrunc <4 x float> %math.result495 to <4 x half>, !dbg !27
  %math.packed496 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %904) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce169 = fadd float %902, %local.reduce165, !dbg !28, !frisk.local_reduction !5
  %905 = extractelement <4 x float> %725, i64 0, !dbg !26
  %906 = tail call float @llvm.amdgcn.exp2.f32(float %905), !dbg !26
  %907 = insertelement <4 x float> poison, float %906, i64 0, !dbg !27
  %local.reduce170 = fadd float %906, %local.reduce166, !dbg !28, !frisk.local_reduction !5
  %908 = extractelement <4 x float> %725, i64 1, !dbg !26
  %909 = tail call float @llvm.amdgcn.exp2.f32(float %908), !dbg !26
  %910 = insertelement <4 x float> %907, float %909, i64 1, !dbg !27
  %local.reduce171 = fadd float %909, %local.reduce167, !dbg !28, !frisk.local_reduction !5
  %911 = extractelement <4 x float> %725, i64 2, !dbg !26
  %912 = tail call float @llvm.amdgcn.exp2.f32(float %911), !dbg !26
  %913 = insertelement <4 x float> %910, float %912, i64 2, !dbg !27
  %local.reduce172 = fadd float %912, %local.reduce168, !dbg !28, !frisk.local_reduction !5
  %914 = extractelement <4 x float> %725, i64 3, !dbg !26
  %915 = tail call float @llvm.amdgcn.exp2.f32(float %914), !dbg !26
  %916 = insertelement <4 x float> %913, float %915, i64 3, !dbg !27
  %math.result497 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %916) #8, !dbg !27, !frisk.register_lifetimes !5
  %917 = fptrunc <4 x float> %math.result497 to <4 x half>, !dbg !27
  %math.packed498 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %917) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce173 = fadd float %915, %local.reduce169, !dbg !28, !frisk.local_reduction !5
  %918 = extractelement <4 x float> %726, i64 0, !dbg !26
  %919 = tail call float @llvm.amdgcn.exp2.f32(float %918), !dbg !26
  %920 = insertelement <4 x float> poison, float %919, i64 0, !dbg !27
  %local.reduce174 = fadd float %919, %local.reduce170, !dbg !28, !frisk.local_reduction !5
  %921 = extractelement <4 x float> %726, i64 1, !dbg !26
  %922 = tail call float @llvm.amdgcn.exp2.f32(float %921), !dbg !26
  %923 = insertelement <4 x float> %920, float %922, i64 1, !dbg !27
  %local.reduce175 = fadd float %922, %local.reduce171, !dbg !28, !frisk.local_reduction !5
  %924 = extractelement <4 x float> %726, i64 2, !dbg !26
  %925 = tail call float @llvm.amdgcn.exp2.f32(float %924), !dbg !26
  %926 = insertelement <4 x float> %923, float %925, i64 2, !dbg !27
  %local.reduce176 = fadd float %925, %local.reduce172, !dbg !28, !frisk.local_reduction !5
  %927 = extractelement <4 x float> %726, i64 3, !dbg !26
  %928 = tail call float @llvm.amdgcn.exp2.f32(float %927), !dbg !26
  %929 = insertelement <4 x float> %926, float %928, i64 3, !dbg !27
  %math.result499 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %929) #8, !dbg !27, !frisk.register_lifetimes !5
  %930 = fptrunc <4 x float> %math.result499 to <4 x half>, !dbg !27
  %math.packed500 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %930) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce177 = fadd float %928, %local.reduce173, !dbg !28, !frisk.local_reduction !5
  %931 = extractelement <4 x float> %727, i64 0, !dbg !26
  %932 = tail call float @llvm.amdgcn.exp2.f32(float %931), !dbg !26
  %933 = insertelement <4 x float> poison, float %932, i64 0, !dbg !27
  %local.reduce178 = fadd float %932, %local.reduce174, !dbg !28, !frisk.local_reduction !5
  %934 = extractelement <4 x float> %727, i64 1, !dbg !26
  %935 = tail call float @llvm.amdgcn.exp2.f32(float %934), !dbg !26
  %936 = insertelement <4 x float> %933, float %935, i64 1, !dbg !27
  %local.reduce179 = fadd float %935, %local.reduce175, !dbg !28, !frisk.local_reduction !5
  %937 = extractelement <4 x float> %727, i64 2, !dbg !26
  %938 = tail call float @llvm.amdgcn.exp2.f32(float %937), !dbg !26
  %939 = insertelement <4 x float> %936, float %938, i64 2, !dbg !27
  %local.reduce180 = fadd float %938, %local.reduce176, !dbg !28, !frisk.local_reduction !5
  %940 = extractelement <4 x float> %727, i64 3, !dbg !26
  %941 = tail call float @llvm.amdgcn.exp2.f32(float %940), !dbg !26
  %942 = insertelement <4 x float> %939, float %941, i64 3, !dbg !27
  %math.result501 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %942) #8, !dbg !27, !frisk.register_lifetimes !5
  %943 = fptrunc <4 x float> %math.result501 to <4 x half>, !dbg !27
  %math.packed502 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %943) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce181 = fadd float %941, %local.reduce177, !dbg !28, !frisk.local_reduction !5
  %944 = extractelement <4 x float> %728, i64 0, !dbg !26
  %945 = tail call float @llvm.amdgcn.exp2.f32(float %944), !dbg !26
  %946 = insertelement <4 x float> poison, float %945, i64 0, !dbg !27
  %local.reduce182 = fadd float %945, %local.reduce178, !dbg !28, !frisk.local_reduction !5
  %947 = extractelement <4 x float> %728, i64 1, !dbg !26
  %948 = tail call float @llvm.amdgcn.exp2.f32(float %947), !dbg !26
  %949 = insertelement <4 x float> %946, float %948, i64 1, !dbg !27
  %local.reduce183 = fadd float %948, %local.reduce179, !dbg !28, !frisk.local_reduction !5
  %local.reduce186 = fadd float %local.reduce182, %local.reduce183, !dbg !28, !frisk.local_reduction !5
  %950 = extractelement <4 x float> %728, i64 2, !dbg !26
  %951 = tail call float @llvm.amdgcn.exp2.f32(float %950), !dbg !26
  %952 = insertelement <4 x float> %949, float %951, i64 2, !dbg !27
  %local.reduce184 = fadd float %951, %local.reduce180, !dbg !28, !frisk.local_reduction !5
  %953 = extractelement <4 x float> %728, i64 3, !dbg !26
  %954 = tail call float @llvm.amdgcn.exp2.f32(float %953), !dbg !26
  %955 = insertelement <4 x float> %952, float %954, i64 3, !dbg !27
  %math.result503 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %955) #8, !dbg !27, !frisk.register_lifetimes !5
  %956 = fptrunc <4 x float> %math.result503 to <4 x half>, !dbg !27
  %math.packed504 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %956) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce185 = fadd float %954, %local.reduce181, !dbg !28, !frisk.local_reduction !5
  %local.reduce187 = fadd float %local.reduce184, %local.reduce185, !dbg !28, !frisk.local_reduction !5
  %957 = fadd float %local.reduce186, %local.reduce187, !dbg !28, !frisk.local_reduction !5
  %958 = fadd float %744, %957, !dbg !28
  %959 = insertelement <1 x float> poison, float %958, i64 0, !dbg !11
  %960 = insertvalue [2 x <1 x float>] %852, <1 x float> %959, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread1030 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !29, !range !9, !frisk.register_lifetimes !5
  %address.local1031 = shl nuw nsw i32 %address.thread1030, 3, !dbg !29
  %address.local1032 = and i32 %address.local1031, 120, !dbg !29
  %address.local1033 = shl nuw nsw i32 %address.thread1030, 5, !dbg !29
  %address.local1034 = and i32 %address.local1033, 1536, !dbg !29
  %address.local1035 = or disjoint i32 %address.local1032, %address.local1034, !dbg !29
  %address.local1036 = zext nneg i32 %address.local1035 to i64, !dbg !29
  %address.local1037 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1036, !dbg !29
  %961 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #10, !dbg !29
  %962 = bitcast <4 x i32> %961 to <8 x half>, !dbg !29
  %address.local1038 = zext nneg i32 %address.thread1030 to i64, !dbg !8
  %address.local1039 = and i64 %address.local1038, 60, !dbg !29
  %address.local1040 = add nuw nsw i64 %address.local1039, %address.local1038, !dbg !29
  %address.local1042.idx = shl nuw nsw i64 %address.local1040, 4, !dbg !29
  %address.local1042 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local1042.idx, !dbg !29
  %address.local1043 = lshr i64 %address.local1038, 4, !dbg !29
  %address.local1044 = and i64 %address.local1043, 3, !dbg !29
  %address.local1046.idx = mul nuw nsw i64 %address.local1044, 768, !dbg !29
  %address.local1046 = getelementptr i8, ptr addrspace(3) %address.local1042, i64 %address.local1046.idx, !dbg !29
  %address.local1047 = lshr i64 %address.local1038, 2, !dbg !15
  %address.local1049.idx = mul nsw i64 %address.local1047, -64, !dbg !29
  %address.local1049 = getelementptr i8, ptr addrspace(3) %address.local1046, i64 %address.local1049.idx, !dbg !29
  %address.local1050 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 256, !dbg !29
  %963 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #10, !dbg !29
  %964 = bitcast <4 x i32> %963 to <8 x half>, !dbg !29
  %address.local1054 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 512, !dbg !29
  %965 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #10, !dbg !29
  %966 = bitcast <4 x i32> %965 to <8 x half>, !dbg !29
  %address.local1058 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 768, !dbg !29
  %967 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #10, !dbg !29
  %968 = bitcast <4 x i32> %967 to <8 x half>, !dbg !29
  %address.local1060 = getelementptr i8, ptr addrspace(3) %address.local1037, i64 4096, !dbg !29
  %969 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #10, !dbg !29
  %970 = bitcast <4 x i32> %969 to <8 x half>, !dbg !29
  %address.local1064 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 4352, !dbg !29
  %971 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #10, !dbg !29
  %972 = bitcast <4 x i32> %971 to <8 x half>, !dbg !29
  %address.local1068 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 4608, !dbg !29
  %973 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #10, !dbg !29
  %974 = bitcast <4 x i32> %973 to <8 x half>, !dbg !29
  %address.local1072 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 4864, !dbg !29
  %975 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #10, !dbg !29
  %976 = bitcast <4 x i32> %975 to <8 x half>, !dbg !29
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local1073 = shl nuw nsw i64 %address.local1038, 3, !dbg !8
  %address.local1074 = and i64 %address.local1073, 1536, !dbg !8
  %address.local1075 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1074, !dbg !8
  %address.local1076 = ptrtoint ptr addrspace(3) %address.local1075 to i64, !dbg !8
  %address.local1077 = trunc i64 %address.local1076 to i32, !dbg !8
  %977 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1077), !dbg !17
  %address.local1078 = shl nuw nsw i64 %address.local1038, 5, !dbg !17
  %address.local1079 = and i64 %address.local1078, 8064, !dbg !17
  %address.local1081 = and i64 %address.local1073, 24, !dbg !17
  %address.local1082 = or disjoint i64 %address.local1079, %address.local1081, !dbg !17
  %978 = or disjoint i64 %.idx18321835, 32, !dbg !17
  %979 = or disjoint i64 %address.local1082, %978, !dbg !17
  %.tr1841 = trunc i64 %979 to i32, !dbg !17
  %dma.buffer.offset1724 = shl i32 %.tr1841, 1, !dbg !17
  %dma.buffer.lds1726 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %977), !dbg !17
  %980 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1724, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %address.local1084 = or disjoint i64 %address.local1038, 256, !dbg !8
  %address.local1085 = shl nuw nsw i64 %address.local1084, 3, !dbg !8
  %address.local1086 = and i64 %address.local1085, 3584, !dbg !8
  %address.local1087 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1086, !dbg !8
  %address.local1088 = ptrtoint ptr addrspace(3) %address.local1087 to i64, !dbg !8
  %address.local1089 = trunc i64 %address.local1088 to i32, !dbg !8
  %981 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1089), !dbg !17
  %address.local1090 = shl nuw nsw i64 %address.local1084, 5, !dbg !17
  %address.local1091 = and i64 %address.local1090, 16256, !dbg !17
  %address.local1094 = or disjoint i64 %address.local1091, %address.local1081, !dbg !17
  %982 = or disjoint i64 %address.local1094, %978, !dbg !17
  %.tr1843 = trunc i64 %982 to i32, !dbg !17
  %dma.buffer.offset1721 = shl i32 %.tr1843, 1, !dbg !17
  %dma.buffer.lds1723 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %981), !dbg !17
  %983 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1721, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %984 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %985 = extractvalue [2 x <8 x float>] %143, 0, !dbg !24
  %986 = shufflevector <8 x float> %985, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %987 = fmul <4 x float> %986, %733, !dbg !24
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %984, <4 x float> %987) #8, !dbg !29
  %989 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %989, <4 x float> %988) #8, !dbg !29
  %991 = shufflevector <8 x half> %966, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %991, <4 x float> %990) #8, !dbg !29
  %993 = shufflevector <8 x half> %968, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %993, <4 x float> %992) #8, !dbg !29
  %995 = shufflevector <8 x half> %970, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %995, <4 x float> %994) #8, !dbg !29
  %997 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %997, <4 x float> %996) #8, !dbg !29
  %999 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %999, <4 x float> %998) #8, !dbg !29
  %1001 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1001, <4 x float> %1000) #8, !dbg !29
  %1003 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1004 = shufflevector <8 x float> %985, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1005 = fmul <4 x float> %1004, %733, !dbg !24
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1003, <4 x float> %1005) #8, !dbg !29
  %1007 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1007, <4 x float> %1006) #8, !dbg !29
  %1009 = shufflevector <8 x half> %966, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1009, <4 x float> %1008) #8, !dbg !29
  %1011 = shufflevector <8 x half> %968, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1011, <4 x float> %1010) #8, !dbg !29
  %1013 = shufflevector <8 x half> %970, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1013, <4 x float> %1012) #8, !dbg !29
  %1015 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1016 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1015, <4 x float> %1014) #8, !dbg !29
  %1017 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1018 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1017, <4 x float> %1016) #8, !dbg !29
  %1019 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1019, <4 x float> %1018) #8, !dbg !29
  %1021 = extractvalue [2 x <8 x float>] %143, 1, !dbg !24
  %1022 = shufflevector <8 x float> %1021, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1023 = fmul <4 x float> %1022, %741, !dbg !24
  %1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %984, <4 x float> %1023) #8, !dbg !29
  %1025 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %989, <4 x float> %1024) #8, !dbg !29
  %1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %991, <4 x float> %1025) #8, !dbg !29
  %1027 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %993, <4 x float> %1026) #8, !dbg !29
  %1028 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %995, <4 x float> %1027) #8, !dbg !29
  %1029 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %997, <4 x float> %1028) #8, !dbg !29
  %1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %999, <4 x float> %1029) #8, !dbg !29
  %1031 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1001, <4 x float> %1030) #8, !dbg !29
  %1032 = shufflevector <8 x float> %1021, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1033 = fmul <4 x float> %1032, %741, !dbg !24
  %1034 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1003, <4 x float> %1033) #8, !dbg !29
  %1035 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1007, <4 x float> %1034) #8, !dbg !29
  %1036 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1009, <4 x float> %1035) #8, !dbg !29
  %1037 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1011, <4 x float> %1036) #8, !dbg !29
  %1038 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1013, <4 x float> %1037) #8, !dbg !29
  %1039 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1015, <4 x float> %1038) #8, !dbg !29
  %1040 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1017, <4 x float> %1039) #8, !dbg !29
  %1041 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1019, <4 x float> %1040) #8, !dbg !29
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1042 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #10, !dbg !29
  %1043 = bitcast <4 x i32> %1042 to <8 x half>, !dbg !29
  %1044 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #10, !dbg !29
  %1045 = bitcast <4 x i32> %1044 to <8 x half>, !dbg !29
  %1046 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #10, !dbg !29
  %1047 = bitcast <4 x i32> %1046 to <8 x half>, !dbg !29
  %1048 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #10, !dbg !29
  %1049 = bitcast <4 x i32> %1048 to <8 x half>, !dbg !29
  %1050 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #10, !dbg !29
  %1051 = bitcast <4 x i32> %1050 to <8 x half>, !dbg !29
  %1052 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #10, !dbg !29
  %1053 = bitcast <4 x i32> %1052 to <8 x half>, !dbg !29
  %1054 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #10, !dbg !29
  %1055 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !29
  %1056 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #10, !dbg !29
  %1057 = bitcast <4 x i32> %1056 to <8 x half>, !dbg !29
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1058 = or disjoint i64 %.idx18321835, 64, !dbg !17
  %1059 = or disjoint i64 %address.local1082, %1058, !dbg !17
  %.tr1845 = trunc i64 %1059 to i32, !dbg !17
  %dma.buffer.offset1718 = shl i32 %.tr1845, 1, !dbg !17
  %1060 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1718, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %1061 = or disjoint i64 %address.local1094, %1058, !dbg !17
  %.tr1847 = trunc i64 %1061 to i32, !dbg !17
  %dma.buffer.offset1715 = shl i32 %.tr1847, 1, !dbg !17
  %1062 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1715, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %1063 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1064 = extractvalue [2 x <8 x float>] %140, 0, !dbg !24
  %1065 = shufflevector <8 x float> %1064, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1066 = fmul <4 x float> %1065, %733, !dbg !24
  %1067 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1063, <4 x float> %1066) #8, !dbg !29
  %1068 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1069 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1068, <4 x float> %1067) #8, !dbg !29
  %1070 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1071 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1070, <4 x float> %1069) #8, !dbg !29
  %1072 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1073 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1072, <4 x float> %1071) #8, !dbg !29
  %1074 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1075 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1074, <4 x float> %1073) #8, !dbg !29
  %1076 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1077 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1076, <4 x float> %1075) #8, !dbg !29
  %1078 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1079 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1078, <4 x float> %1077) #8, !dbg !29
  %1080 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1081 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1080, <4 x float> %1079) #8, !dbg !29
  %1082 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1083 = shufflevector <8 x float> %1064, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1084 = fmul <4 x float> %1083, %733, !dbg !24
  %1085 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1082, <4 x float> %1084) #8, !dbg !29
  %1086 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1087 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1086, <4 x float> %1085) #8, !dbg !29
  %1088 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1089 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1088, <4 x float> %1087) #8, !dbg !29
  %1090 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1091 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1090, <4 x float> %1089) #8, !dbg !29
  %1092 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1093 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1092, <4 x float> %1091) #8, !dbg !29
  %1094 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1095 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1094, <4 x float> %1093) #8, !dbg !29
  %1096 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1097 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1096, <4 x float> %1095) #8, !dbg !29
  %1098 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1099 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1098, <4 x float> %1097) #8, !dbg !29
  %1100 = extractvalue [2 x <8 x float>] %140, 1, !dbg !24
  %1101 = shufflevector <8 x float> %1100, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1102 = fmul <4 x float> %1101, %741, !dbg !24
  %1103 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1063, <4 x float> %1102) #8, !dbg !29
  %1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1068, <4 x float> %1103) #8, !dbg !29
  %1105 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1070, <4 x float> %1104) #8, !dbg !29
  %1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1072, <4 x float> %1105) #8, !dbg !29
  %1107 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1074, <4 x float> %1106) #8, !dbg !29
  %1108 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1076, <4 x float> %1107) #8, !dbg !29
  %1109 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1078, <4 x float> %1108) #8, !dbg !29
  %1110 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1080, <4 x float> %1109) #8, !dbg !29
  %1111 = shufflevector <8 x float> %1100, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1112 = fmul <4 x float> %1111, %741, !dbg !24
  %1113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1082, <4 x float> %1112) #8, !dbg !29
  %1114 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1086, <4 x float> %1113) #8, !dbg !29
  %1115 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1088, <4 x float> %1114) #8, !dbg !29
  %1116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1090, <4 x float> %1115) #8, !dbg !29
  %1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1092, <4 x float> %1116) #8, !dbg !29
  %1118 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1094, <4 x float> %1117) #8, !dbg !29
  %1119 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1096, <4 x float> %1118) #8, !dbg !29
  %1120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1098, <4 x float> %1119) #8, !dbg !29
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1121 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #10, !dbg !29
  %1122 = bitcast <4 x i32> %1121 to <8 x half>, !dbg !29
  %1123 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #10, !dbg !29
  %1124 = bitcast <4 x i32> %1123 to <8 x half>, !dbg !29
  %1125 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #10, !dbg !29
  %1126 = bitcast <4 x i32> %1125 to <8 x half>, !dbg !29
  %1127 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #10, !dbg !29
  %1128 = bitcast <4 x i32> %1127 to <8 x half>, !dbg !29
  %1129 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #10, !dbg !29
  %1130 = bitcast <4 x i32> %1129 to <8 x half>, !dbg !29
  %1131 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #10, !dbg !29
  %1132 = bitcast <4 x i32> %1131 to <8 x half>, !dbg !29
  %1133 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #10, !dbg !29
  %1134 = bitcast <4 x i32> %1133 to <8 x half>, !dbg !29
  %1135 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #10, !dbg !29
  %1136 = bitcast <4 x i32> %1135 to <8 x half>, !dbg !29
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1137 = or disjoint i64 %.idx18321835, 96, !dbg !17
  %1138 = or disjoint i64 %address.local1082, %1137, !dbg !17
  %.tr1849 = trunc i64 %1138 to i32, !dbg !17
  %dma.buffer.offset1712 = shl i32 %.tr1849, 1, !dbg !17
  %1139 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1712, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %1140 = or disjoint i64 %address.local1094, %1137, !dbg !17
  %.tr1851 = trunc i64 %1140 to i32, !dbg !17
  %dma.buffer.offset = shl i32 %.tr1851, 1, !dbg !17
  %1141 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %1142 = shufflevector <8 x half> %1122, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1143 = extractvalue [2 x <8 x float>] %139, 0, !dbg !24
  %1144 = shufflevector <8 x float> %1143, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1145 = fmul <4 x float> %1144, %733, !dbg !24
  %1146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1142, <4 x float> %1145) #8, !dbg !29
  %1147 = shufflevector <8 x half> %1124, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1147, <4 x float> %1146) #8, !dbg !29
  %1149 = shufflevector <8 x half> %1126, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1149, <4 x float> %1148) #8, !dbg !29
  %1151 = shufflevector <8 x half> %1128, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1151, <4 x float> %1150) #8, !dbg !29
  %1153 = shufflevector <8 x half> %1130, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1153, <4 x float> %1152) #8, !dbg !29
  %1155 = shufflevector <8 x half> %1132, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1155, <4 x float> %1154) #8, !dbg !29
  %1157 = shufflevector <8 x half> %1134, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1157, <4 x float> %1156) #8, !dbg !29
  %1159 = shufflevector <8 x half> %1136, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1159, <4 x float> %1158) #8, !dbg !29
  %1161 = shufflevector <8 x half> %1122, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1162 = shufflevector <8 x float> %1143, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1163 = fmul <4 x float> %1162, %733, !dbg !24
  %1164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1161, <4 x float> %1163) #8, !dbg !29
  %1165 = shufflevector <8 x half> %1124, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1165, <4 x float> %1164) #8, !dbg !29
  %1167 = shufflevector <8 x half> %1126, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1167, <4 x float> %1166) #8, !dbg !29
  %1169 = shufflevector <8 x half> %1128, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1169, <4 x float> %1168) #8, !dbg !29
  %1171 = shufflevector <8 x half> %1130, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1171, <4 x float> %1170) #8, !dbg !29
  %1173 = shufflevector <8 x half> %1132, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1173, <4 x float> %1172) #8, !dbg !29
  %1175 = shufflevector <8 x half> %1134, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1175, <4 x float> %1174) #8, !dbg !29
  %1177 = shufflevector <8 x half> %1136, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1177, <4 x float> %1176) #8, !dbg !29
  %1179 = extractvalue [2 x <8 x float>] %139, 1, !dbg !24
  %1180 = shufflevector <8 x float> %1179, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1181 = fmul <4 x float> %1180, %741, !dbg !24
  %1182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1142, <4 x float> %1181) #8, !dbg !29
  %1183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1147, <4 x float> %1182) #8, !dbg !29
  %1184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1149, <4 x float> %1183) #8, !dbg !29
  %1185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1151, <4 x float> %1184) #8, !dbg !29
  %1186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1153, <4 x float> %1185) #8, !dbg !29
  %1187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1155, <4 x float> %1186) #8, !dbg !29
  %1188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1157, <4 x float> %1187) #8, !dbg !29
  %1189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1159, <4 x float> %1188) #8, !dbg !29
  %1190 = shufflevector <8 x float> %1179, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1191 = fmul <4 x float> %1190, %741, !dbg !24
  %1192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1161, <4 x float> %1191) #8, !dbg !29
  %1193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1165, <4 x float> %1192) #8, !dbg !29
  %1194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1167, <4 x float> %1193) #8, !dbg !29
  %1195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1169, <4 x float> %1194) #8, !dbg !29
  %1196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1171, <4 x float> %1195) #8, !dbg !29
  %1197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1173, <4 x float> %1196) #8, !dbg !29
  %1198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1175, <4 x float> %1197) #8, !dbg !29
  %1199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1177, <4 x float> %1198) #8, !dbg !29
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1200 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #10, !dbg !29
  %1201 = bitcast <4 x i32> %1200 to <8 x half>, !dbg !29
  %1202 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #10, !dbg !29
  %1203 = bitcast <4 x i32> %1202 to <8 x half>, !dbg !29
  %1204 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #10, !dbg !29
  %1205 = bitcast <4 x i32> %1204 to <8 x half>, !dbg !29
  %1206 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #10, !dbg !29
  %1207 = bitcast <4 x i32> %1206 to <8 x half>, !dbg !29
  %1208 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #10, !dbg !29
  %1209 = bitcast <4 x i32> %1208 to <8 x half>, !dbg !29
  %1210 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #10, !dbg !29
  %1211 = bitcast <4 x i32> %1210 to <8 x half>, !dbg !29
  %1212 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #10, !dbg !29
  %1213 = bitcast <4 x i32> %1212 to <8 x half>, !dbg !29
  %1214 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #10, !dbg !29
  %1215 = bitcast <4 x i32> %1214 to <8 x half>, !dbg !29
  %1216 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1217 = extractvalue [2 x <8 x float>] %138, 0, !dbg !24
  %1218 = shufflevector <8 x float> %1217, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1219 = fmul <4 x float> %1218, %733, !dbg !24
  %1220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1216, <4 x float> %1219) #8, !dbg !29
  %1221 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1221, <4 x float> %1220) #8, !dbg !29
  %1223 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1223, <4 x float> %1222) #8, !dbg !29
  %1225 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1225, <4 x float> %1224) #8, !dbg !29
  %1227 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1227, <4 x float> %1226) #8, !dbg !29
  %1229 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1229, <4 x float> %1228) #8, !dbg !29
  %1231 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1231, <4 x float> %1230) #8, !dbg !29
  %1233 = shufflevector <8 x half> %1215, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1233, <4 x float> %1232) #8, !dbg !29
  %1235 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1236 = shufflevector <8 x float> %1217, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1237 = fmul <4 x float> %1236, %733, !dbg !24
  %1238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1235, <4 x float> %1237) #8, !dbg !29
  %1239 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1239, <4 x float> %1238) #8, !dbg !29
  %1241 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1241, <4 x float> %1240) #8, !dbg !29
  %1243 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1243, <4 x float> %1242) #8, !dbg !29
  %1245 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1245, <4 x float> %1244) #8, !dbg !29
  %1247 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1247, <4 x float> %1246) #8, !dbg !29
  %1249 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1249, <4 x float> %1248) #8, !dbg !29
  %1251 = shufflevector <8 x half> %1215, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1251, <4 x float> %1250) #8, !dbg !29
  %1253 = extractvalue [2 x <8 x float>] %138, 1, !dbg !24
  %1254 = shufflevector <8 x float> %1253, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1255 = fmul <4 x float> %1254, %741, !dbg !24
  %1256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1216, <4 x float> %1255) #8, !dbg !29
  %1257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1221, <4 x float> %1256) #8, !dbg !29
  %1258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1223, <4 x float> %1257) #8, !dbg !29
  %1259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1225, <4 x float> %1258) #8, !dbg !29
  %1260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1227, <4 x float> %1259) #8, !dbg !29
  %1261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1229, <4 x float> %1260) #8, !dbg !29
  %1262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1231, <4 x float> %1261) #8, !dbg !29
  %1263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1233, <4 x float> %1262) #8, !dbg !29
  %1264 = shufflevector <8 x float> %1253, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1265 = fmul <4 x float> %1264, %741, !dbg !24
  %1266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1235, <4 x float> %1265) #8, !dbg !29
  %1267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1239, <4 x float> %1266) #8, !dbg !29
  %1268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1241, <4 x float> %1267) #8, !dbg !29
  %1269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1243, <4 x float> %1268) #8, !dbg !29
  %1270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1245, <4 x float> %1269) #8, !dbg !29
  %1271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1247, <4 x float> %1270) #8, !dbg !29
  %1272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1249, <4 x float> %1271) #8, !dbg !29
  %1273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1251, <4 x float> %1272) #8, !dbg !29
  %1274 = shufflevector <4 x float> %1002, <4 x float> %1020, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1275 = insertvalue [2 x <8 x float>] poison, <8 x float> %1274, 0, !dbg !11
  %1276 = shufflevector <4 x float> %1031, <4 x float> %1041, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1277 = insertvalue [2 x <8 x float>] %1275, <8 x float> %1276, 1, !dbg !11
  %1278 = insertvalue [2 x <1 x float>] poison, <1 x float> %697, 0, !dbg !11
  %1279 = insertvalue [2 x <1 x float>] %1278, <1 x float> %710, 1, !dbg !11
  %1280 = shufflevector <4 x float> %1081, <4 x float> %1099, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1281 = insertvalue [2 x <8 x float>] poison, <8 x float> %1280, 0, !dbg !11
  %1282 = shufflevector <4 x float> %1110, <4 x float> %1120, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1283 = insertvalue [2 x <8 x float>] %1281, <8 x float> %1282, 1, !dbg !11
  %1284 = shufflevector <4 x float> %1160, <4 x float> %1178, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1285 = insertvalue [2 x <8 x float>] poison, <8 x float> %1284, 0, !dbg !11
  %1286 = shufflevector <4 x float> %1189, <4 x float> %1199, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1287 = insertvalue [2 x <8 x float>] %1285, <8 x float> %1286, 1, !dbg !11
  %1288 = shufflevector <4 x float> %1234, <4 x float> %1252, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1289 = insertvalue [2 x <8 x float>] poison, <8 x float> %1288, 0, !dbg !11
  %1290 = shufflevector <4 x float> %1263, <4 x float> %1273, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1291 = insertvalue [2 x <8 x float>] %1289, <8 x float> %1290, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1292 = add nuw nsw i64 %144, 1, !dbg !11
  %1293 = icmp slt i64 %1292, %104, !dbg !11
  br i1 %1293, label %137, label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %592, %.._crit_edge_crit_edge
  %.pre-phi24 = phi i32 [ %.pre23, %.._crit_edge_crit_edge ], [ %687, %592 ], !dbg !11
  %.pre-phi22 = phi i32 [ %.pre21, %.._crit_edge_crit_edge ], [ %679, %592 ], !dbg !11
  %.pre-phi20 = phi i32 [ %.pre19, %.._crit_edge_crit_edge ], [ %678, %592 ], !dbg !11
  %lane.sum.lane287.pre-phi = phi i32 [ %.pre17, %.._crit_edge_crit_edge ], [ %676, %592 ], !dbg !11
  %.lcssa6 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1277, %592 ], !dbg !11
  %.lcssa5 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %960, %592 ], !dbg !11
  %.lcssa4 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1283, %592 ], !dbg !11
  %.lcssa3 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1287, %592 ], !dbg !11
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1291, %592 ], !dbg !11
  %lane.sum.field284 = extractvalue [2 x <1 x float>] %.lcssa5, 1, !dbg !11
  %lane.sum.local285 = extractelement <1 x float> %lane.sum.field284, i64 0, !dbg !11
  %1294 = shl nuw nsw i32 %lane.sum.lane287.pre-phi, 2, !dbg !11
  %1295 = xor i32 %1294, 64, !dbg !11
  %1296 = bitcast float %lane.sum.local285 to i32, !dbg !11
  %lane.sum.shuffle288 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1295, i32 %1296), !dbg !11
  %1297 = bitcast i32 %lane.sum.shuffle288 to float, !dbg !11
  %lane.sum.final289 = fadd float %lane.sum.local285, %1297, !dbg !11
  %1298 = xor i32 %1294, 128, !dbg !11
  %1299 = bitcast float %lane.sum.final289 to i32, !dbg !11
  %lane.sum.shuffle290 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1298, i32 %1299), !dbg !11
  %1300 = bitcast i32 %lane.sum.shuffle290 to float, !dbg !11
  %lane.sum.final291 = fadd float %lane.sum.final289, %1300, !dbg !11
  %1301 = insertelement <1 x float> poison, float %lane.sum.final291, i64 0, !dbg !11
  %lane.sum.field = extractvalue [2 x <1 x float>] %.lcssa5, 0, !dbg !11
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !11
  %1302 = bitcast float %lane.sum.local to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1295, i32 %1302), !dbg !11
  %1303 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %lane.sum.local, %1303, !dbg !11
  %1304 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle282 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1298, i32 %1304), !dbg !11
  %1305 = bitcast i32 %lane.sum.shuffle282 to float, !dbg !11
  %lane.sum.final283 = fadd float %lane.sum.final, %1305, !dbg !11
  %1306 = insertelement <1 x float> poison, float %lane.sum.final283, i64 0, !dbg !11
  %1307 = and i32 %31, 16, !dbg !11
  %.not = icmp eq i32 %1307, 0, !dbg !11
  %1308 = and i32 %31, 32, !dbg !11
  %.not36 = icmp eq i32 %1308, 0, !dbg !11
  %1309 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !11
  %1310 = extractelement <8 x float> %1309, i64 0, !dbg !11
  %1311 = extractelement <8 x float> %1309, i64 1, !dbg !11
  %1312 = extractelement <8 x float> %1309, i64 2, !dbg !11
  %1313 = extractelement <8 x float> %1309, i64 3, !dbg !11
  %1314 = select i1 %.not, float %1311, float %1310, !dbg !11
  %1315 = icmp samesign ult i32 %.pre-phi20, %.pre-phi22, !dbg !11
  %1316 = select i1 %1315, i32 %.pre-phi20, i32 %lane.sum.lane287.pre-phi, !dbg !11
  %1317 = shl nuw nsw i32 %1316, 2, !dbg !11
  %1318 = bitcast float %1314 to i32, !dbg !11
  %1319 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1318), !dbg !11
  %1320 = bitcast i32 %1319 to float, !dbg !11
  %1321 = select i1 %.not, float %1310, float %1320, !dbg !11
  %1322 = select i1 %.not, float %1320, float %1311, !dbg !11
  %1323 = select i1 %.not, float %1313, float %1312, !dbg !11
  %1324 = bitcast float %1323 to i32, !dbg !11
  %1325 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1324), !dbg !11
  %1326 = bitcast i32 %1325 to float, !dbg !11
  %1327 = select i1 %.not, float %1312, float %1326, !dbg !11
  %1328 = select i1 %.not, float %1326, float %1313, !dbg !11
  %1329 = select i1 %.not36, float %1327, float %1321, !dbg !11
  %1330 = icmp samesign ult i32 %.pre-phi24, %.pre-phi22, !dbg !11
  %1331 = select i1 %1330, i32 %.pre-phi24, i32 %lane.sum.lane287.pre-phi, !dbg !11
  %1332 = shl nuw nsw i32 %1331, 2, !dbg !11
  %1333 = bitcast float %1329 to i32, !dbg !11
  %1334 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1333), !dbg !11
  %1335 = bitcast i32 %1334 to float, !dbg !11
  %1336 = select i1 %.not36, float %1328, float %1322, !dbg !11
  %1337 = bitcast float %1336 to i32, !dbg !11
  %1338 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1337), !dbg !11
  %1339 = bitcast i32 %1338 to float, !dbg !11
  %1340 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1341 = shufflevector <4 x i1> %1340, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1342 = insertelement <4 x float> poison, float %1321, i64 0, !dbg !11
  %1343 = insertelement <4 x float> %1342, float %1322, i64 1, !dbg !11
  %1344 = insertelement <4 x float> %1343, float %1335, i64 2, !dbg !11
  %1345 = insertelement <4 x float> %1344, float %1339, i64 3, !dbg !11
  %1346 = insertelement <4 x float> poison, float %1335, i64 0, !dbg !11
  %1347 = insertelement <4 x float> %1346, float %1339, i64 1, !dbg !11
  %1348 = insertelement <4 x float> %1347, float %1327, i64 2, !dbg !11
  %1349 = insertelement <4 x float> %1348, float %1328, i64 3, !dbg !11
  %1350 = select <4 x i1> %1341, <4 x float> %1345, <4 x float> %1349, !dbg !11
  %1351 = extractvalue [2 x <8 x float>] %.lcssa3, 0, !dbg !11
  %1352 = extractelement <8 x float> %1351, i64 0, !dbg !11
  %1353 = extractelement <8 x float> %1351, i64 1, !dbg !11
  %1354 = extractelement <8 x float> %1351, i64 2, !dbg !11
  %1355 = extractelement <8 x float> %1351, i64 3, !dbg !11
  %1356 = select i1 %.not, float %1353, float %1352, !dbg !11
  %1357 = bitcast float %1356 to i32, !dbg !11
  %1358 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1357), !dbg !11
  %1359 = bitcast i32 %1358 to float, !dbg !11
  %1360 = select i1 %.not, float %1352, float %1359, !dbg !11
  %1361 = select i1 %.not, float %1359, float %1353, !dbg !11
  %1362 = select i1 %.not, float %1355, float %1354, !dbg !11
  %1363 = bitcast float %1362 to i32, !dbg !11
  %1364 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1363), !dbg !11
  %1365 = bitcast i32 %1364 to float, !dbg !11
  %1366 = select i1 %.not, float %1354, float %1365, !dbg !11
  %1367 = select i1 %.not, float %1365, float %1355, !dbg !11
  %1368 = select i1 %.not36, float %1366, float %1360, !dbg !11
  %1369 = bitcast float %1368 to i32, !dbg !11
  %1370 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1369), !dbg !11
  %1371 = bitcast i32 %1370 to float, !dbg !11
  %1372 = select i1 %.not36, float %1367, float %1361, !dbg !11
  %1373 = bitcast float %1372 to i32, !dbg !11
  %1374 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1373), !dbg !11
  %1375 = bitcast i32 %1374 to float, !dbg !11
  %1376 = insertelement <4 x float> poison, float %1360, i64 0, !dbg !11
  %1377 = insertelement <4 x float> %1376, float %1361, i64 1, !dbg !11
  %1378 = insertelement <4 x float> %1377, float %1371, i64 2, !dbg !11
  %1379 = insertelement <4 x float> %1378, float %1375, i64 3, !dbg !11
  %1380 = insertelement <4 x float> poison, float %1371, i64 0, !dbg !11
  %1381 = insertelement <4 x float> %1380, float %1375, i64 1, !dbg !11
  %1382 = insertelement <4 x float> %1381, float %1366, i64 2, !dbg !11
  %1383 = insertelement <4 x float> %1382, float %1367, i64 3, !dbg !11
  %1384 = select <4 x i1> %1341, <4 x float> %1379, <4 x float> %1383, !dbg !11
  %1385 = extractvalue [2 x <8 x float>] %.lcssa4, 0, !dbg !11
  %1386 = extractelement <8 x float> %1385, i64 0, !dbg !11
  %1387 = extractelement <8 x float> %1385, i64 1, !dbg !11
  %1388 = extractelement <8 x float> %1385, i64 2, !dbg !11
  %1389 = extractelement <8 x float> %1385, i64 3, !dbg !11
  %1390 = select i1 %.not, float %1387, float %1386, !dbg !11
  %1391 = bitcast float %1390 to i32, !dbg !11
  %1392 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1391), !dbg !11
  %1393 = bitcast i32 %1392 to float, !dbg !11
  %1394 = select i1 %.not, float %1386, float %1393, !dbg !11
  %1395 = select i1 %.not, float %1393, float %1387, !dbg !11
  %1396 = select i1 %.not, float %1389, float %1388, !dbg !11
  %1397 = bitcast float %1396 to i32, !dbg !11
  %1398 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1397), !dbg !11
  %1399 = bitcast i32 %1398 to float, !dbg !11
  %1400 = select i1 %.not, float %1388, float %1399, !dbg !11
  %1401 = select i1 %.not, float %1399, float %1389, !dbg !11
  %1402 = select i1 %.not36, float %1400, float %1394, !dbg !11
  %1403 = bitcast float %1402 to i32, !dbg !11
  %1404 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1403), !dbg !11
  %1405 = bitcast i32 %1404 to float, !dbg !11
  %1406 = select i1 %.not36, float %1401, float %1395, !dbg !11
  %1407 = bitcast float %1406 to i32, !dbg !11
  %1408 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1407), !dbg !11
  %1409 = bitcast i32 %1408 to float, !dbg !11
  %1410 = insertelement <4 x float> poison, float %1394, i64 0, !dbg !11
  %1411 = insertelement <4 x float> %1410, float %1395, i64 1, !dbg !11
  %1412 = insertelement <4 x float> %1411, float %1405, i64 2, !dbg !11
  %1413 = insertelement <4 x float> %1412, float %1409, i64 3, !dbg !11
  %1414 = insertelement <4 x float> poison, float %1405, i64 0, !dbg !11
  %1415 = insertelement <4 x float> %1414, float %1409, i64 1, !dbg !11
  %1416 = insertelement <4 x float> %1415, float %1400, i64 2, !dbg !11
  %1417 = insertelement <4 x float> %1416, float %1401, i64 3, !dbg !11
  %1418 = select <4 x i1> %1341, <4 x float> %1413, <4 x float> %1417, !dbg !11
  %1419 = extractvalue [2 x <8 x float>] %.lcssa6, 0, !dbg !11
  %1420 = extractelement <8 x float> %1419, i64 0, !dbg !11
  %1421 = extractelement <8 x float> %1419, i64 1, !dbg !11
  %1422 = extractelement <8 x float> %1419, i64 2, !dbg !11
  %1423 = extractelement <8 x float> %1419, i64 3, !dbg !11
  %1424 = select i1 %.not, float %1421, float %1420, !dbg !11
  %1425 = bitcast float %1424 to i32, !dbg !11
  %1426 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1425), !dbg !11
  %1427 = bitcast i32 %1426 to float, !dbg !11
  %1428 = select i1 %.not, float %1420, float %1427, !dbg !11
  %1429 = select i1 %.not, float %1427, float %1421, !dbg !11
  %1430 = select i1 %.not, float %1423, float %1422, !dbg !11
  %1431 = bitcast float %1430 to i32, !dbg !11
  %1432 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1431), !dbg !11
  %1433 = bitcast i32 %1432 to float, !dbg !11
  %1434 = select i1 %.not, float %1422, float %1433, !dbg !11
  %1435 = select i1 %.not, float %1433, float %1423, !dbg !11
  %1436 = select i1 %.not36, float %1434, float %1428, !dbg !11
  %1437 = bitcast float %1436 to i32, !dbg !11
  %1438 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1437), !dbg !11
  %1439 = select i1 %.not36, float %1435, float %1429, !dbg !11
  %1440 = bitcast float %1439 to i32, !dbg !11
  %1441 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1440), !dbg !11
  %1442 = shufflevector <1 x float> %1306, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %1443 = extractelement <8 x float> %1309, i64 4, !dbg !11
  %1444 = extractelement <8 x float> %1309, i64 5, !dbg !11
  %1445 = extractelement <8 x float> %1309, i64 6, !dbg !11
  %1446 = extractelement <8 x float> %1309, i64 7, !dbg !11
  %1447 = select i1 %.not, float %1444, float %1443, !dbg !11
  %1448 = bitcast float %1447 to i32, !dbg !11
  %1449 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1448), !dbg !11
  %1450 = bitcast i32 %1449 to float, !dbg !11
  %1451 = select i1 %.not, float %1443, float %1450, !dbg !11
  %1452 = select i1 %.not, float %1450, float %1444, !dbg !11
  %1453 = select i1 %.not, float %1446, float %1445, !dbg !11
  %1454 = bitcast float %1453 to i32, !dbg !11
  %1455 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1454), !dbg !11
  %1456 = bitcast i32 %1455 to float, !dbg !11
  %1457 = select i1 %.not, float %1445, float %1456, !dbg !11
  %1458 = select i1 %.not, float %1456, float %1446, !dbg !11
  %1459 = select i1 %.not36, float %1457, float %1451, !dbg !11
  %1460 = bitcast float %1459 to i32, !dbg !11
  %1461 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1460), !dbg !11
  %1462 = bitcast i32 %1461 to float, !dbg !11
  %1463 = select i1 %.not36, float %1458, float %1452, !dbg !11
  %1464 = bitcast float %1463 to i32, !dbg !11
  %1465 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1464), !dbg !11
  %1466 = bitcast i32 %1465 to float, !dbg !11
  %1467 = insertelement <4 x float> poison, float %1451, i64 0, !dbg !11
  %1468 = insertelement <4 x float> %1467, float %1452, i64 1, !dbg !11
  %1469 = insertelement <4 x float> %1468, float %1462, i64 2, !dbg !11
  %1470 = insertelement <4 x float> %1469, float %1466, i64 3, !dbg !11
  %1471 = insertelement <4 x float> poison, float %1462, i64 0, !dbg !11
  %1472 = insertelement <4 x float> %1471, float %1466, i64 1, !dbg !11
  %1473 = insertelement <4 x float> %1472, float %1457, i64 2, !dbg !11
  %1474 = insertelement <4 x float> %1473, float %1458, i64 3, !dbg !11
  %1475 = select <4 x i1> %1341, <4 x float> %1470, <4 x float> %1474, !dbg !11
  %1476 = extractelement <8 x float> %1351, i64 4, !dbg !11
  %1477 = extractelement <8 x float> %1351, i64 5, !dbg !11
  %1478 = extractelement <8 x float> %1351, i64 6, !dbg !11
  %1479 = extractelement <8 x float> %1351, i64 7, !dbg !11
  %1480 = select i1 %.not, float %1477, float %1476, !dbg !11
  %1481 = bitcast float %1480 to i32, !dbg !11
  %1482 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1481), !dbg !11
  %1483 = bitcast i32 %1482 to float, !dbg !11
  %1484 = select i1 %.not, float %1476, float %1483, !dbg !11
  %1485 = select i1 %.not, float %1483, float %1477, !dbg !11
  %1486 = select i1 %.not, float %1479, float %1478, !dbg !11
  %1487 = bitcast float %1486 to i32, !dbg !11
  %1488 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1487), !dbg !11
  %1489 = bitcast i32 %1488 to float, !dbg !11
  %1490 = select i1 %.not, float %1478, float %1489, !dbg !11
  %1491 = select i1 %.not, float %1489, float %1479, !dbg !11
  %1492 = select i1 %.not36, float %1490, float %1484, !dbg !11
  %1493 = bitcast float %1492 to i32, !dbg !11
  %1494 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1493), !dbg !11
  %1495 = bitcast i32 %1494 to float, !dbg !11
  %1496 = select i1 %.not36, float %1491, float %1485, !dbg !11
  %1497 = bitcast float %1496 to i32, !dbg !11
  %1498 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1497), !dbg !11
  %1499 = bitcast i32 %1498 to float, !dbg !11
  %1500 = insertelement <4 x float> poison, float %1484, i64 0, !dbg !11
  %1501 = insertelement <4 x float> %1500, float %1485, i64 1, !dbg !11
  %1502 = insertelement <4 x float> %1501, float %1495, i64 2, !dbg !11
  %1503 = insertelement <4 x float> %1502, float %1499, i64 3, !dbg !11
  %1504 = insertelement <4 x float> poison, float %1495, i64 0, !dbg !11
  %1505 = insertelement <4 x float> %1504, float %1499, i64 1, !dbg !11
  %1506 = insertelement <4 x float> %1505, float %1490, i64 2, !dbg !11
  %1507 = insertelement <4 x float> %1506, float %1491, i64 3, !dbg !11
  %1508 = select <4 x i1> %1341, <4 x float> %1503, <4 x float> %1507, !dbg !11
  %1509 = extractelement <8 x float> %1385, i64 4, !dbg !11
  %1510 = extractelement <8 x float> %1385, i64 5, !dbg !11
  %1511 = extractelement <8 x float> %1385, i64 6, !dbg !11
  %1512 = extractelement <8 x float> %1385, i64 7, !dbg !11
  %1513 = select i1 %.not, float %1510, float %1509, !dbg !11
  %1514 = bitcast float %1513 to i32, !dbg !11
  %1515 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1514), !dbg !11
  %1516 = bitcast i32 %1515 to float, !dbg !11
  %1517 = select i1 %.not, float %1509, float %1516, !dbg !11
  %1518 = select i1 %.not, float %1516, float %1510, !dbg !11
  %1519 = select i1 %.not, float %1512, float %1511, !dbg !11
  %1520 = bitcast float %1519 to i32, !dbg !11
  %1521 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1520), !dbg !11
  %1522 = bitcast i32 %1521 to float, !dbg !11
  %1523 = select i1 %.not, float %1511, float %1522, !dbg !11
  %1524 = select i1 %.not, float %1522, float %1512, !dbg !11
  %1525 = select i1 %.not36, float %1523, float %1517, !dbg !11
  %1526 = bitcast float %1525 to i32, !dbg !11
  %1527 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1526), !dbg !11
  %1528 = bitcast i32 %1527 to float, !dbg !11
  %1529 = select i1 %.not36, float %1524, float %1518, !dbg !11
  %1530 = bitcast float %1529 to i32, !dbg !11
  %1531 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1530), !dbg !11
  %1532 = bitcast i32 %1531 to float, !dbg !11
  %1533 = insertelement <4 x float> poison, float %1517, i64 0, !dbg !11
  %1534 = insertelement <4 x float> %1533, float %1518, i64 1, !dbg !11
  %1535 = insertelement <4 x float> %1534, float %1528, i64 2, !dbg !11
  %1536 = insertelement <4 x float> %1535, float %1532, i64 3, !dbg !11
  %1537 = insertelement <4 x float> poison, float %1528, i64 0, !dbg !11
  %1538 = insertelement <4 x float> %1537, float %1532, i64 1, !dbg !11
  %1539 = insertelement <4 x float> %1538, float %1523, i64 2, !dbg !11
  %1540 = insertelement <4 x float> %1539, float %1524, i64 3, !dbg !11
  %1541 = select <4 x i1> %1341, <4 x float> %1536, <4 x float> %1540, !dbg !11
  %1542 = extractelement <8 x float> %1419, i64 4, !dbg !11
  %1543 = extractelement <8 x float> %1419, i64 5, !dbg !11
  %1544 = extractelement <8 x float> %1419, i64 6, !dbg !11
  %1545 = extractelement <8 x float> %1419, i64 7, !dbg !11
  %1546 = select i1 %.not, float %1543, float %1542, !dbg !11
  %1547 = bitcast float %1546 to i32, !dbg !11
  %1548 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1547), !dbg !11
  %1549 = bitcast i32 %1548 to float, !dbg !11
  %1550 = select i1 %.not, float %1542, float %1549, !dbg !11
  %1551 = select i1 %.not, float %1549, float %1543, !dbg !11
  %1552 = select i1 %.not, float %1545, float %1544, !dbg !11
  %1553 = bitcast float %1552 to i32, !dbg !11
  %1554 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1553), !dbg !11
  %1555 = bitcast i32 %1554 to float, !dbg !11
  %1556 = select i1 %.not, float %1544, float %1555, !dbg !11
  %1557 = select i1 %.not, float %1555, float %1545, !dbg !11
  %1558 = select i1 %.not36, float %1556, float %1550, !dbg !11
  %1559 = bitcast float %1558 to i32, !dbg !11
  %1560 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1559), !dbg !11
  %1561 = select i1 %.not36, float %1557, float %1551, !dbg !11
  %1562 = bitcast float %1561 to i32, !dbg !11
  %1563 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1562), !dbg !11
  %1564 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !11
  %1565 = extractelement <8 x float> %1564, i64 0, !dbg !11
  %1566 = extractelement <8 x float> %1564, i64 1, !dbg !11
  %1567 = extractelement <8 x float> %1564, i64 2, !dbg !11
  %1568 = extractelement <8 x float> %1564, i64 3, !dbg !11
  %1569 = select i1 %.not, float %1566, float %1565, !dbg !11
  %1570 = bitcast float %1569 to i32, !dbg !11
  %1571 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1570), !dbg !11
  %1572 = bitcast i32 %1571 to float, !dbg !11
  %1573 = select i1 %.not, float %1565, float %1572, !dbg !11
  %1574 = select i1 %.not, float %1572, float %1566, !dbg !11
  %1575 = select i1 %.not, float %1568, float %1567, !dbg !11
  %1576 = bitcast float %1575 to i32, !dbg !11
  %1577 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1576), !dbg !11
  %1578 = bitcast i32 %1577 to float, !dbg !11
  %1579 = select i1 %.not, float %1567, float %1578, !dbg !11
  %1580 = select i1 %.not, float %1578, float %1568, !dbg !11
  %1581 = select i1 %.not36, float %1579, float %1573, !dbg !11
  %1582 = bitcast float %1581 to i32, !dbg !11
  %1583 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1582), !dbg !11
  %1584 = bitcast i32 %1583 to float, !dbg !11
  %1585 = select i1 %.not36, float %1580, float %1574, !dbg !11
  %1586 = bitcast float %1585 to i32, !dbg !11
  %1587 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1586), !dbg !11
  %1588 = bitcast i32 %1587 to float, !dbg !11
  %1589 = insertelement <4 x float> poison, float %1573, i64 0, !dbg !11
  %1590 = insertelement <4 x float> %1589, float %1574, i64 1, !dbg !11
  %1591 = insertelement <4 x float> %1590, float %1584, i64 2, !dbg !11
  %1592 = insertelement <4 x float> %1591, float %1588, i64 3, !dbg !11
  %1593 = insertelement <4 x float> poison, float %1584, i64 0, !dbg !11
  %1594 = insertelement <4 x float> %1593, float %1588, i64 1, !dbg !11
  %1595 = insertelement <4 x float> %1594, float %1579, i64 2, !dbg !11
  %1596 = insertelement <4 x float> %1595, float %1580, i64 3, !dbg !11
  %1597 = select <4 x i1> %1341, <4 x float> %1592, <4 x float> %1596, !dbg !11
  %1598 = extractvalue [2 x <8 x float>] %.lcssa3, 1, !dbg !11
  %1599 = extractelement <8 x float> %1598, i64 0, !dbg !11
  %1600 = extractelement <8 x float> %1598, i64 1, !dbg !11
  %1601 = extractelement <8 x float> %1598, i64 2, !dbg !11
  %1602 = extractelement <8 x float> %1598, i64 3, !dbg !11
  %1603 = select i1 %.not, float %1600, float %1599, !dbg !11
  %1604 = bitcast float %1603 to i32, !dbg !11
  %1605 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1604), !dbg !11
  %1606 = bitcast i32 %1605 to float, !dbg !11
  %1607 = select i1 %.not, float %1599, float %1606, !dbg !11
  %1608 = select i1 %.not, float %1606, float %1600, !dbg !11
  %1609 = select i1 %.not, float %1602, float %1601, !dbg !11
  %1610 = bitcast float %1609 to i32, !dbg !11
  %1611 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1610), !dbg !11
  %1612 = bitcast i32 %1611 to float, !dbg !11
  %1613 = select i1 %.not, float %1601, float %1612, !dbg !11
  %1614 = select i1 %.not, float %1612, float %1602, !dbg !11
  %1615 = select i1 %.not36, float %1613, float %1607, !dbg !11
  %1616 = bitcast float %1615 to i32, !dbg !11
  %1617 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1616), !dbg !11
  %1618 = bitcast i32 %1617 to float, !dbg !11
  %1619 = select i1 %.not36, float %1614, float %1608, !dbg !11
  %1620 = bitcast float %1619 to i32, !dbg !11
  %1621 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1620), !dbg !11
  %1622 = bitcast i32 %1621 to float, !dbg !11
  %1623 = insertelement <4 x float> poison, float %1607, i64 0, !dbg !11
  %1624 = insertelement <4 x float> %1623, float %1608, i64 1, !dbg !11
  %1625 = insertelement <4 x float> %1624, float %1618, i64 2, !dbg !11
  %1626 = insertelement <4 x float> %1625, float %1622, i64 3, !dbg !11
  %1627 = insertelement <4 x float> poison, float %1618, i64 0, !dbg !11
  %1628 = insertelement <4 x float> %1627, float %1622, i64 1, !dbg !11
  %1629 = insertelement <4 x float> %1628, float %1613, i64 2, !dbg !11
  %1630 = insertelement <4 x float> %1629, float %1614, i64 3, !dbg !11
  %1631 = select <4 x i1> %1341, <4 x float> %1626, <4 x float> %1630, !dbg !11
  %1632 = extractvalue [2 x <8 x float>] %.lcssa4, 1, !dbg !11
  %1633 = extractelement <8 x float> %1632, i64 0, !dbg !11
  %1634 = extractelement <8 x float> %1632, i64 1, !dbg !11
  %1635 = extractelement <8 x float> %1632, i64 2, !dbg !11
  %1636 = extractelement <8 x float> %1632, i64 3, !dbg !11
  %1637 = select i1 %.not, float %1634, float %1633, !dbg !11
  %1638 = bitcast float %1637 to i32, !dbg !11
  %1639 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1638), !dbg !11
  %1640 = bitcast i32 %1639 to float, !dbg !11
  %1641 = select i1 %.not, float %1633, float %1640, !dbg !11
  %1642 = select i1 %.not, float %1640, float %1634, !dbg !11
  %1643 = select i1 %.not, float %1636, float %1635, !dbg !11
  %1644 = bitcast float %1643 to i32, !dbg !11
  %1645 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1644), !dbg !11
  %1646 = bitcast i32 %1645 to float, !dbg !11
  %1647 = select i1 %.not, float %1635, float %1646, !dbg !11
  %1648 = select i1 %.not, float %1646, float %1636, !dbg !11
  %1649 = select i1 %.not36, float %1647, float %1641, !dbg !11
  %1650 = bitcast float %1649 to i32, !dbg !11
  %1651 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1650), !dbg !11
  %1652 = bitcast i32 %1651 to float, !dbg !11
  %1653 = select i1 %.not36, float %1648, float %1642, !dbg !11
  %1654 = bitcast float %1653 to i32, !dbg !11
  %1655 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1654), !dbg !11
  %1656 = bitcast i32 %1655 to float, !dbg !11
  %1657 = insertelement <4 x float> poison, float %1641, i64 0, !dbg !11
  %1658 = insertelement <4 x float> %1657, float %1642, i64 1, !dbg !11
  %1659 = insertelement <4 x float> %1658, float %1652, i64 2, !dbg !11
  %1660 = insertelement <4 x float> %1659, float %1656, i64 3, !dbg !11
  %1661 = insertelement <4 x float> poison, float %1652, i64 0, !dbg !11
  %1662 = insertelement <4 x float> %1661, float %1656, i64 1, !dbg !11
  %1663 = insertelement <4 x float> %1662, float %1647, i64 2, !dbg !11
  %1664 = insertelement <4 x float> %1663, float %1648, i64 3, !dbg !11
  %1665 = select <4 x i1> %1341, <4 x float> %1660, <4 x float> %1664, !dbg !11
  %1666 = extractvalue [2 x <8 x float>] %.lcssa6, 1, !dbg !11
  %1667 = extractelement <8 x float> %1666, i64 0, !dbg !11
  %1668 = extractelement <8 x float> %1666, i64 1, !dbg !11
  %1669 = extractelement <8 x float> %1666, i64 2, !dbg !11
  %1670 = extractelement <8 x float> %1666, i64 3, !dbg !11
  %1671 = select i1 %.not, float %1668, float %1667, !dbg !11
  %1672 = bitcast float %1671 to i32, !dbg !11
  %1673 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1672), !dbg !11
  %1674 = bitcast i32 %1673 to float, !dbg !11
  %1675 = select i1 %.not, float %1667, float %1674, !dbg !11
  %1676 = select i1 %.not, float %1674, float %1668, !dbg !11
  %1677 = select i1 %.not, float %1670, float %1669, !dbg !11
  %1678 = bitcast float %1677 to i32, !dbg !11
  %1679 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1678), !dbg !11
  %1680 = bitcast i32 %1679 to float, !dbg !11
  %1681 = select i1 %.not, float %1669, float %1680, !dbg !11
  %1682 = select i1 %.not, float %1680, float %1670, !dbg !11
  %1683 = select i1 %.not36, float %1681, float %1675, !dbg !11
  %1684 = bitcast float %1683 to i32, !dbg !11
  %1685 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1684), !dbg !11
  %1686 = select i1 %.not36, float %1682, float %1676, !dbg !11
  %1687 = bitcast float %1686 to i32, !dbg !11
  %1688 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1687), !dbg !11
  %1689 = shufflevector <1 x float> %1301, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %1690 = extractelement <8 x float> %1564, i64 4, !dbg !11
  %1691 = extractelement <8 x float> %1564, i64 5, !dbg !11
  %1692 = extractelement <8 x float> %1564, i64 6, !dbg !11
  %1693 = extractelement <8 x float> %1564, i64 7, !dbg !11
  %1694 = select i1 %.not, float %1691, float %1690, !dbg !11
  %1695 = bitcast float %1694 to i32, !dbg !11
  %1696 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1695), !dbg !11
  %1697 = bitcast i32 %1696 to float, !dbg !11
  %1698 = select i1 %.not, float %1690, float %1697, !dbg !11
  %1699 = select i1 %.not, float %1697, float %1691, !dbg !11
  %1700 = select i1 %.not, float %1693, float %1692, !dbg !11
  %1701 = bitcast float %1700 to i32, !dbg !11
  %1702 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1701), !dbg !11
  %1703 = bitcast i32 %1702 to float, !dbg !11
  %1704 = select i1 %.not, float %1692, float %1703, !dbg !11
  %1705 = select i1 %.not, float %1703, float %1693, !dbg !11
  %1706 = select i1 %.not36, float %1704, float %1698, !dbg !11
  %1707 = bitcast float %1706 to i32, !dbg !11
  %1708 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1707), !dbg !11
  %1709 = bitcast i32 %1708 to float, !dbg !11
  %1710 = select i1 %.not36, float %1705, float %1699, !dbg !11
  %1711 = bitcast float %1710 to i32, !dbg !11
  %1712 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1711), !dbg !11
  %1713 = bitcast i32 %1712 to float, !dbg !11
  %1714 = insertelement <4 x float> poison, float %1698, i64 0, !dbg !11
  %1715 = insertelement <4 x float> %1714, float %1699, i64 1, !dbg !11
  %1716 = insertelement <4 x float> %1715, float %1709, i64 2, !dbg !11
  %1717 = insertelement <4 x float> %1716, float %1713, i64 3, !dbg !11
  %1718 = insertelement <4 x float> poison, float %1709, i64 0, !dbg !11
  %1719 = insertelement <4 x float> %1718, float %1713, i64 1, !dbg !11
  %1720 = insertelement <4 x float> %1719, float %1704, i64 2, !dbg !11
  %1721 = insertelement <4 x float> %1720, float %1705, i64 3, !dbg !11
  %1722 = select <4 x i1> %1341, <4 x float> %1717, <4 x float> %1721, !dbg !11
  %1723 = extractelement <8 x float> %1598, i64 4, !dbg !11
  %1724 = extractelement <8 x float> %1598, i64 5, !dbg !11
  %1725 = extractelement <8 x float> %1598, i64 6, !dbg !11
  %1726 = extractelement <8 x float> %1598, i64 7, !dbg !11
  %1727 = select i1 %.not, float %1724, float %1723, !dbg !11
  %1728 = bitcast float %1727 to i32, !dbg !11
  %1729 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1728), !dbg !11
  %1730 = bitcast i32 %1729 to float, !dbg !11
  %1731 = select i1 %.not, float %1723, float %1730, !dbg !11
  %1732 = select i1 %.not, float %1730, float %1724, !dbg !11
  %1733 = select i1 %.not, float %1726, float %1725, !dbg !11
  %1734 = bitcast float %1733 to i32, !dbg !11
  %1735 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1734), !dbg !11
  %1736 = bitcast i32 %1735 to float, !dbg !11
  %1737 = select i1 %.not, float %1725, float %1736, !dbg !11
  %1738 = select i1 %.not, float %1736, float %1726, !dbg !11
  %1739 = select i1 %.not36, float %1737, float %1731, !dbg !11
  %1740 = bitcast float %1739 to i32, !dbg !11
  %1741 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1740), !dbg !11
  %1742 = bitcast i32 %1741 to float, !dbg !11
  %1743 = select i1 %.not36, float %1738, float %1732, !dbg !11
  %1744 = bitcast float %1743 to i32, !dbg !11
  %1745 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1744), !dbg !11
  %1746 = bitcast i32 %1745 to float, !dbg !11
  %1747 = insertelement <4 x float> poison, float %1731, i64 0, !dbg !11
  %1748 = insertelement <4 x float> %1747, float %1732, i64 1, !dbg !11
  %1749 = insertelement <4 x float> %1748, float %1742, i64 2, !dbg !11
  %1750 = insertelement <4 x float> %1749, float %1746, i64 3, !dbg !11
  %1751 = insertelement <4 x float> poison, float %1742, i64 0, !dbg !11
  %1752 = insertelement <4 x float> %1751, float %1746, i64 1, !dbg !11
  %1753 = insertelement <4 x float> %1752, float %1737, i64 2, !dbg !11
  %1754 = insertelement <4 x float> %1753, float %1738, i64 3, !dbg !11
  %1755 = select <4 x i1> %1341, <4 x float> %1750, <4 x float> %1754, !dbg !11
  %1756 = extractelement <8 x float> %1632, i64 4, !dbg !11
  %1757 = extractelement <8 x float> %1632, i64 5, !dbg !11
  %1758 = extractelement <8 x float> %1632, i64 6, !dbg !11
  %1759 = extractelement <8 x float> %1632, i64 7, !dbg !11
  %1760 = select i1 %.not, float %1757, float %1756, !dbg !11
  %1761 = bitcast float %1760 to i32, !dbg !11
  %1762 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1761), !dbg !11
  %1763 = bitcast i32 %1762 to float, !dbg !11
  %1764 = select i1 %.not, float %1756, float %1763, !dbg !11
  %1765 = select i1 %.not, float %1763, float %1757, !dbg !11
  %1766 = select i1 %.not, float %1759, float %1758, !dbg !11
  %1767 = bitcast float %1766 to i32, !dbg !11
  %1768 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1767), !dbg !11
  %1769 = bitcast i32 %1768 to float, !dbg !11
  %1770 = select i1 %.not, float %1758, float %1769, !dbg !11
  %1771 = select i1 %.not, float %1769, float %1759, !dbg !11
  %1772 = select i1 %.not36, float %1770, float %1764, !dbg !11
  %1773 = bitcast float %1772 to i32, !dbg !11
  %1774 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1773), !dbg !11
  %1775 = bitcast i32 %1774 to float, !dbg !11
  %1776 = select i1 %.not36, float %1771, float %1765, !dbg !11
  %1777 = bitcast float %1776 to i32, !dbg !11
  %1778 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1777), !dbg !11
  %1779 = bitcast i32 %1778 to float, !dbg !11
  %1780 = insertelement <4 x float> poison, float %1764, i64 0, !dbg !11
  %1781 = insertelement <4 x float> %1780, float %1765, i64 1, !dbg !11
  %1782 = insertelement <4 x float> %1781, float %1775, i64 2, !dbg !11
  %1783 = insertelement <4 x float> %1782, float %1779, i64 3, !dbg !11
  %1784 = insertelement <4 x float> poison, float %1775, i64 0, !dbg !11
  %1785 = insertelement <4 x float> %1784, float %1779, i64 1, !dbg !11
  %1786 = insertelement <4 x float> %1785, float %1770, i64 2, !dbg !11
  %1787 = insertelement <4 x float> %1786, float %1771, i64 3, !dbg !11
  %1788 = select <4 x i1> %1341, <4 x float> %1783, <4 x float> %1787, !dbg !11
  %1789 = extractelement <8 x float> %1666, i64 4, !dbg !11
  %1790 = extractelement <8 x float> %1666, i64 5, !dbg !11
  %1791 = extractelement <8 x float> %1666, i64 6, !dbg !11
  %1792 = extractelement <8 x float> %1666, i64 7, !dbg !11
  %1793 = select i1 %.not, float %1790, float %1789, !dbg !11
  %1794 = bitcast float %1793 to i32, !dbg !11
  %1795 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1794), !dbg !11
  %1796 = bitcast i32 %1795 to float, !dbg !11
  %1797 = select i1 %.not, float %1789, float %1796, !dbg !11
  %1798 = select i1 %.not, float %1796, float %1790, !dbg !11
  %1799 = select i1 %.not, float %1792, float %1791, !dbg !11
  %1800 = bitcast float %1799 to i32, !dbg !11
  %1801 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1317, i32 %1800), !dbg !11
  %1802 = bitcast i32 %1801 to float, !dbg !11
  %1803 = select i1 %.not, float %1791, float %1802, !dbg !11
  %1804 = select i1 %.not, float %1802, float %1792, !dbg !11
  %1805 = select i1 %.not36, float %1803, float %1797, !dbg !11
  %1806 = bitcast float %1805 to i32, !dbg !11
  %1807 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1806), !dbg !11
  %1808 = select i1 %.not36, float %1804, float %1798, !dbg !11
  %1809 = bitcast float %1808 to i32, !dbg !11
  %1810 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1332, i32 %1809), !dbg !11
  %1811 = bitcast i32 %1807 to float, !dbg !11
  %1812 = bitcast i32 %1810 to float, !dbg !11
  %1813 = insertelement <4 x float> poison, float %1797, i64 0, !dbg !11
  %1814 = insertelement <4 x float> %1813, float %1798, i64 1, !dbg !11
  %1815 = insertelement <4 x float> %1814, float %1811, i64 2, !dbg !11
  %1816 = insertelement <4 x float> %1815, float %1812, i64 3, !dbg !11
  %1817 = insertelement <4 x float> poison, float %1811, i64 0, !dbg !11
  %1818 = insertelement <4 x float> %1817, float %1812, i64 1, !dbg !11
  %1819 = insertelement <4 x float> %1818, float %1803, i64 2, !dbg !11
  %1820 = insertelement <4 x float> %1819, float %1804, i64 3, !dbg !11
  %1821 = select <4 x i1> %1341, <4 x float> %1816, <4 x float> %1820, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1689, !dbg !30
  %normalized = fmul <4 x float> %row.reciprocal, %1821, !dbg !30
  %1822 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !31
  %1823 = extractelement <4 x half> %1822, i64 3, !dbg !31
  %1824 = extractelement <4 x half> %1822, i64 2, !dbg !31
  %1825 = extractelement <4 x half> %1822, i64 1, !dbg !31
  %1826 = extractelement <4 x half> %1822, i64 0, !dbg !31
  %1827 = bitcast i32 %1685 to float, !dbg !11
  %1828 = bitcast i32 %1688 to float, !dbg !11
  %1829 = insertelement <4 x float> poison, float %1675, i64 0, !dbg !11
  %1830 = insertelement <4 x float> %1829, float %1676, i64 1, !dbg !11
  %1831 = insertelement <4 x float> %1830, float %1827, i64 2, !dbg !11
  %1832 = insertelement <4 x float> %1831, float %1828, i64 3, !dbg !11
  %1833 = insertelement <4 x float> poison, float %1827, i64 0, !dbg !11
  %1834 = insertelement <4 x float> %1833, float %1828, i64 1, !dbg !11
  %1835 = insertelement <4 x float> %1834, float %1681, i64 2, !dbg !11
  %1836 = insertelement <4 x float> %1835, float %1682, i64 3, !dbg !11
  %1837 = select <4 x i1> %1341, <4 x float> %1832, <4 x float> %1836, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %1837, !dbg !30
  %1838 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !31
  %1839 = extractelement <4 x half> %1838, i64 3, !dbg !31
  %1840 = extractelement <4 x half> %1838, i64 2, !dbg !31
  %1841 = extractelement <4 x half> %1838, i64 1, !dbg !31
  %1842 = extractelement <4 x half> %1838, i64 0, !dbg !31
  %1843 = bitcast i32 %1560 to float, !dbg !11
  %1844 = bitcast i32 %1563 to float, !dbg !11
  %1845 = insertelement <4 x float> poison, float %1550, i64 0, !dbg !11
  %1846 = insertelement <4 x float> %1845, float %1551, i64 1, !dbg !11
  %1847 = insertelement <4 x float> %1846, float %1843, i64 2, !dbg !11
  %1848 = insertelement <4 x float> %1847, float %1844, i64 3, !dbg !11
  %1849 = insertelement <4 x float> poison, float %1843, i64 0, !dbg !11
  %1850 = insertelement <4 x float> %1849, float %1844, i64 1, !dbg !11
  %1851 = insertelement <4 x float> %1850, float %1556, i64 2, !dbg !11
  %1852 = insertelement <4 x float> %1851, float %1557, i64 3, !dbg !11
  %1853 = select <4 x i1> %1341, <4 x float> %1848, <4 x float> %1852, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1442, !dbg !30
  %normalized106 = fmul <4 x float> %row.reciprocal105, %1853, !dbg !30
  %1854 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !31
  %1855 = extractelement <4 x half> %1854, i64 3, !dbg !31
  %1856 = extractelement <4 x half> %1854, i64 2, !dbg !31
  %1857 = extractelement <4 x half> %1854, i64 1, !dbg !31
  %1858 = extractelement <4 x half> %1854, i64 0, !dbg !31
  %1859 = bitcast i32 %1438 to float, !dbg !11
  %1860 = bitcast i32 %1441 to float, !dbg !11
  %1861 = insertelement <4 x float> poison, float %1428, i64 0, !dbg !11
  %1862 = insertelement <4 x float> %1861, float %1429, i64 1, !dbg !11
  %1863 = insertelement <4 x float> %1862, float %1859, i64 2, !dbg !11
  %1864 = insertelement <4 x float> %1863, float %1860, i64 3, !dbg !11
  %1865 = insertelement <4 x float> poison, float %1859, i64 0, !dbg !11
  %1866 = insertelement <4 x float> %1865, float %1860, i64 1, !dbg !11
  %1867 = insertelement <4 x float> %1866, float %1434, i64 2, !dbg !11
  %1868 = insertelement <4 x float> %1867, float %1435, i64 3, !dbg !11
  %1869 = select <4 x i1> %1341, <4 x float> %1864, <4 x float> %1868, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %1869, !dbg !30
  %1870 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !31
  %1871 = extractelement <4 x half> %1870, i64 3, !dbg !31
  %1872 = extractelement <4 x half> %1870, i64 2, !dbg !31
  %1873 = extractelement <4 x half> %1870, i64 1, !dbg !31
  %1874 = extractelement <4 x half> %1870, i64 0, !dbg !31
  %address.thread1100 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !32, !range !9, !frisk.register_lifetimes !5
  %address.local1101 = zext nneg i32 %address.thread1100 to i64, !dbg !8
  %1875 = lshr i64 %address.local1101, 2, !dbg !32
  %address.local1104 = and i64 %1875, 48, !dbg !32
  %address.local1106 = and i64 %address.local1101, 240, !dbg !32
  %address.local1105 = add i64 %26, %address.local1101, !dbg !32
  %address.local1107 = sub i64 %address.local1105, %address.local1106, !dbg !32
  %address.local1108 = add i64 %address.local1107, %address.local1104, !dbg !32
  %1876 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28, !dbg !32
  %address.local1111.idx = shl nuw nsw i64 %address.local1108, 8, !dbg !32
  %address.local1111 = getelementptr inbounds nuw i8, ptr addrspace(1) %1876, i64 %address.local1111.idx, !dbg !32
  %address.local1113 = and i64 %1875, 12, !dbg !10
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1113, !dbg !32
  store half %1874, ptr addrspace(1) %address.local1114, align 2, !dbg !32
  %address.local1116 = or disjoint i64 %address.local1113, 1, !dbg !32
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1116, !dbg !32
  store half %1873, ptr addrspace(1) %address.local1117, align 2, !dbg !32
  %address.local1119 = or disjoint i64 %address.local1113, 2, !dbg !32
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1119, !dbg !32
  store half %1872, ptr addrspace(1) %address.local1120, align 2, !dbg !32
  %address.local1122 = or disjoint i64 %address.local1113, 3, !dbg !32
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1122, !dbg !32
  store half %1871, ptr addrspace(1) %address.local1123, align 2, !dbg !32
  %address.local1125 = or disjoint i64 %address.local1113, 16, !dbg !32
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1125, !dbg !32
  store half %1858, ptr addrspace(1) %address.local1126, align 2, !dbg !32
  %address.local1128 = or disjoint i64 %address.local1113, 17, !dbg !32
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1128, !dbg !32
  store half %1857, ptr addrspace(1) %address.local1129, align 2, !dbg !32
  %address.local1131 = or disjoint i64 %address.local1113, 18, !dbg !32
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1131, !dbg !32
  store half %1856, ptr addrspace(1) %address.local1132, align 2, !dbg !32
  %address.local1134 = or disjoint i64 %address.local1113, 19, !dbg !32
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1134, !dbg !32
  store half %1855, ptr addrspace(1) %address.local1135, align 2, !dbg !32
  %address.local1139 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1111, i64 16384, !dbg !32
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1113, !dbg !32
  store half %1842, ptr addrspace(1) %address.local1140, align 2, !dbg !32
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1116, !dbg !32
  store half %1841, ptr addrspace(1) %address.local1142, align 2, !dbg !32
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1119, !dbg !32
  store half %1840, ptr addrspace(1) %address.local1144, align 2, !dbg !32
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1122, !dbg !32
  store half %1839, ptr addrspace(1) %address.local1146, align 2, !dbg !32
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1125, !dbg !32
  store half %1826, ptr addrspace(1) %address.local1148, align 2, !dbg !32
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1128, !dbg !32
  store half %1825, ptr addrspace(1) %address.local1150, align 2, !dbg !32
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1131, !dbg !32
  store half %1824, ptr addrspace(1) %address.local1152, align 2, !dbg !32
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1134, !dbg !32
  store half %1823, ptr addrspace(1) %address.local1154, align 2, !dbg !32
  %normalized109 = fmul <4 x float> %row.reciprocal, %1788, !dbg !30
  %1877 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !31
  %1878 = extractelement <4 x half> %1877, i64 3, !dbg !31
  %1879 = extractelement <4 x half> %1877, i64 2, !dbg !31
  %1880 = extractelement <4 x half> %1877, i64 1, !dbg !31
  %1881 = extractelement <4 x half> %1877, i64 0, !dbg !31
  %normalized110 = fmul <4 x float> %row.reciprocal, %1665, !dbg !30
  %1882 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !31
  %1883 = extractelement <4 x half> %1882, i64 3, !dbg !31
  %1884 = extractelement <4 x half> %1882, i64 2, !dbg !31
  %1885 = extractelement <4 x half> %1882, i64 1, !dbg !31
  %1886 = extractelement <4 x half> %1882, i64 0, !dbg !31
  %normalized112 = fmul <4 x float> %row.reciprocal105, %1541, !dbg !30
  %1887 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !31
  %1888 = extractelement <4 x half> %1887, i64 3, !dbg !31
  %1889 = extractelement <4 x half> %1887, i64 2, !dbg !31
  %1890 = extractelement <4 x half> %1887, i64 1, !dbg !31
  %1891 = extractelement <4 x half> %1887, i64 0, !dbg !31
  %normalized113 = fmul <4 x float> %row.reciprocal105, %1418, !dbg !30
  %1892 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !31
  %1893 = extractelement <4 x half> %1892, i64 3, !dbg !31
  %1894 = extractelement <4 x half> %1892, i64 2, !dbg !31
  %1895 = extractelement <4 x half> %1892, i64 1, !dbg !31
  %1896 = extractelement <4 x half> %1892, i64 0, !dbg !31
  %address.local1169 = or disjoint i64 %address.local1113, 32, !dbg !32
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1169, !dbg !32
  store half %1896, ptr addrspace(1) %address.local1170, align 2, !dbg !32
  %address.local1172 = or disjoint i64 %address.local1113, 33, !dbg !32
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1172, !dbg !32
  store half %1895, ptr addrspace(1) %address.local1173, align 2, !dbg !32
  %address.local1175 = or disjoint i64 %address.local1113, 34, !dbg !32
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1175, !dbg !32
  store half %1894, ptr addrspace(1) %address.local1176, align 2, !dbg !32
  %address.local1178 = or disjoint i64 %address.local1113, 35, !dbg !32
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1178, !dbg !32
  store half %1893, ptr addrspace(1) %address.local1179, align 2, !dbg !32
  %address.local1181 = or disjoint i64 %address.local1113, 48, !dbg !32
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1181, !dbg !32
  store half %1891, ptr addrspace(1) %address.local1182, align 2, !dbg !32
  %address.local1184 = or disjoint i64 %address.local1113, 49, !dbg !32
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1184, !dbg !32
  store half %1890, ptr addrspace(1) %address.local1185, align 2, !dbg !32
  %address.local1187 = or disjoint i64 %address.local1113, 50, !dbg !32
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1187, !dbg !32
  store half %1889, ptr addrspace(1) %address.local1188, align 2, !dbg !32
  %address.local1190 = or i64 %1875, 51, !dbg !32
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1190, !dbg !32
  store half %1888, ptr addrspace(1) %address.local1191, align 2, !dbg !32
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1169, !dbg !32
  store half %1886, ptr addrspace(1) %address.local1196, align 2, !dbg !32
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1172, !dbg !32
  store half %1885, ptr addrspace(1) %address.local1198, align 2, !dbg !32
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1175, !dbg !32
  store half %1884, ptr addrspace(1) %address.local1200, align 2, !dbg !32
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1178, !dbg !32
  store half %1883, ptr addrspace(1) %address.local1202, align 2, !dbg !32
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1181, !dbg !32
  store half %1881, ptr addrspace(1) %address.local1204, align 2, !dbg !32
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1184, !dbg !32
  store half %1880, ptr addrspace(1) %address.local1206, align 2, !dbg !32
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1187, !dbg !32
  store half %1879, ptr addrspace(1) %address.local1208, align 2, !dbg !32
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1190, !dbg !32
  store half %1878, ptr addrspace(1) %address.local1210, align 2, !dbg !32
  %normalized115 = fmul <4 x float> %row.reciprocal, %1755, !dbg !30
  %1897 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !31
  %1898 = extractelement <4 x half> %1897, i64 3, !dbg !31
  %1899 = extractelement <4 x half> %1897, i64 2, !dbg !31
  %1900 = extractelement <4 x half> %1897, i64 1, !dbg !31
  %1901 = extractelement <4 x half> %1897, i64 0, !dbg !31
  %normalized116 = fmul <4 x float> %row.reciprocal, %1631, !dbg !30
  %1902 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !31
  %1903 = extractelement <4 x half> %1902, i64 3, !dbg !31
  %1904 = extractelement <4 x half> %1902, i64 2, !dbg !31
  %1905 = extractelement <4 x half> %1902, i64 1, !dbg !31
  %1906 = extractelement <4 x half> %1902, i64 0, !dbg !31
  %normalized118 = fmul <4 x float> %row.reciprocal105, %1508, !dbg !30
  %1907 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !31
  %1908 = extractelement <4 x half> %1907, i64 3, !dbg !31
  %1909 = extractelement <4 x half> %1907, i64 2, !dbg !31
  %1910 = extractelement <4 x half> %1907, i64 1, !dbg !31
  %1911 = extractelement <4 x half> %1907, i64 0, !dbg !31
  %normalized119 = fmul <4 x float> %row.reciprocal105, %1384, !dbg !30
  %1912 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !31
  %1913 = extractelement <4 x half> %1912, i64 3, !dbg !31
  %1914 = extractelement <4 x half> %1912, i64 2, !dbg !31
  %1915 = extractelement <4 x half> %1912, i64 1, !dbg !31
  %1916 = extractelement <4 x half> %1912, i64 0, !dbg !31
  %address.local1225 = or disjoint i64 %address.local1113, 64, !dbg !32
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1225, !dbg !32
  store half %1916, ptr addrspace(1) %address.local1226, align 2, !dbg !32
  %address.local1228 = or disjoint i64 %address.local1113, 65, !dbg !32
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1228, !dbg !32
  store half %1915, ptr addrspace(1) %address.local1229, align 2, !dbg !32
  %address.local1231 = or disjoint i64 %address.local1113, 66, !dbg !32
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1231, !dbg !32
  store half %1914, ptr addrspace(1) %address.local1232, align 2, !dbg !32
  %address.local1234 = or disjoint i64 %address.local1113, 67, !dbg !32
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1234, !dbg !32
  store half %1913, ptr addrspace(1) %address.local1235, align 2, !dbg !32
  %address.local1237 = or disjoint i64 %address.local1113, 80, !dbg !32
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1237, !dbg !32
  store half %1911, ptr addrspace(1) %address.local1238, align 2, !dbg !32
  %address.local1240 = or disjoint i64 %address.local1113, 81, !dbg !32
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1240, !dbg !32
  store half %1910, ptr addrspace(1) %address.local1241, align 2, !dbg !32
  %address.local1243 = or disjoint i64 %address.local1113, 82, !dbg !32
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1243, !dbg !32
  store half %1909, ptr addrspace(1) %address.local1244, align 2, !dbg !32
  %address.local1246 = or disjoint i64 %address.local1113, 83, !dbg !32
  %address.local1247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1246, !dbg !32
  store half %1908, ptr addrspace(1) %address.local1247, align 2, !dbg !32
  %address.local1252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1225, !dbg !32
  store half %1906, ptr addrspace(1) %address.local1252, align 2, !dbg !32
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1228, !dbg !32
  store half %1905, ptr addrspace(1) %address.local1254, align 2, !dbg !32
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1231, !dbg !32
  store half %1904, ptr addrspace(1) %address.local1256, align 2, !dbg !32
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1234, !dbg !32
  store half %1903, ptr addrspace(1) %address.local1258, align 2, !dbg !32
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1237, !dbg !32
  store half %1901, ptr addrspace(1) %address.local1260, align 2, !dbg !32
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1240, !dbg !32
  store half %1900, ptr addrspace(1) %address.local1262, align 2, !dbg !32
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1243, !dbg !32
  store half %1899, ptr addrspace(1) %address.local1264, align 2, !dbg !32
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1246, !dbg !32
  store half %1898, ptr addrspace(1) %address.local1266, align 2, !dbg !32
  %normalized121 = fmul <4 x float> %row.reciprocal, %1722, !dbg !30
  %1917 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !31
  %1918 = extractelement <4 x half> %1917, i64 3, !dbg !31
  %1919 = extractelement <4 x half> %1917, i64 2, !dbg !31
  %1920 = extractelement <4 x half> %1917, i64 1, !dbg !31
  %1921 = extractelement <4 x half> %1917, i64 0, !dbg !31
  %normalized122 = fmul <4 x float> %row.reciprocal, %1597, !dbg !30
  %1922 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !31
  %1923 = extractelement <4 x half> %1922, i64 3, !dbg !31
  %1924 = extractelement <4 x half> %1922, i64 2, !dbg !31
  %1925 = extractelement <4 x half> %1922, i64 1, !dbg !31
  %1926 = extractelement <4 x half> %1922, i64 0, !dbg !31
  %normalized124 = fmul <4 x float> %row.reciprocal105, %1475, !dbg !30
  %1927 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !31
  %1928 = extractelement <4 x half> %1927, i64 3, !dbg !31
  %1929 = extractelement <4 x half> %1927, i64 2, !dbg !31
  %1930 = extractelement <4 x half> %1927, i64 1, !dbg !31
  %1931 = extractelement <4 x half> %1927, i64 0, !dbg !31
  %normalized125 = fmul <4 x float> %row.reciprocal105, %1350, !dbg !30
  %1932 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !31
  %1933 = extractelement <4 x half> %1932, i64 3, !dbg !31
  %1934 = extractelement <4 x half> %1932, i64 2, !dbg !31
  %1935 = extractelement <4 x half> %1932, i64 1, !dbg !31
  %1936 = extractelement <4 x half> %1932, i64 0, !dbg !31
  %address.local1281 = or disjoint i64 %address.local1113, 96, !dbg !32
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1281, !dbg !32
  store half %1936, ptr addrspace(1) %address.local1282, align 2, !dbg !32
  %address.local1284 = or disjoint i64 %address.local1113, 97, !dbg !32
  %address.local1285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1284, !dbg !32
  store half %1935, ptr addrspace(1) %address.local1285, align 2, !dbg !32
  %address.local1287 = or disjoint i64 %address.local1113, 98, !dbg !32
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1287, !dbg !32
  store half %1934, ptr addrspace(1) %address.local1288, align 2, !dbg !32
  %address.local1290 = or disjoint i64 %address.local1113, 99, !dbg !32
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1290, !dbg !32
  store half %1933, ptr addrspace(1) %address.local1291, align 2, !dbg !32
  %address.local1293 = or disjoint i64 %address.local1113, 112, !dbg !32
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1293, !dbg !32
  store half %1931, ptr addrspace(1) %address.local1294, align 2, !dbg !32
  %address.local1296 = or disjoint i64 %address.local1113, 113, !dbg !32
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1296, !dbg !32
  store half %1930, ptr addrspace(1) %address.local1297, align 2, !dbg !32
  %address.local1299 = or disjoint i64 %address.local1113, 114, !dbg !32
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1299, !dbg !32
  store half %1929, ptr addrspace(1) %address.local1300, align 2, !dbg !32
  %address.local1302 = or i64 %1875, 115, !dbg !32
  %address.local1303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1302, !dbg !32
  store half %1928, ptr addrspace(1) %address.local1303, align 2, !dbg !32
  %address.local1308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1281, !dbg !32
  store half %1926, ptr addrspace(1) %address.local1308, align 2, !dbg !32
  %address.local1310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1284, !dbg !32
  store half %1925, ptr addrspace(1) %address.local1310, align 2, !dbg !32
  %address.local1312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1287, !dbg !32
  store half %1924, ptr addrspace(1) %address.local1312, align 2, !dbg !32
  %address.local1314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1290, !dbg !32
  store half %1923, ptr addrspace(1) %address.local1314, align 2, !dbg !32
  %address.local1316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1293, !dbg !32
  store half %1921, ptr addrspace(1) %address.local1316, align 2, !dbg !32
  %address.local1318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1296, !dbg !32
  store half %1920, ptr addrspace(1) %address.local1318, align 2, !dbg !32
  %address.local1320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1299, !dbg !32
  store half %1919, ptr addrspace(1) %address.local1320, align 2, !dbg !32
  %address.local1322 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1302, !dbg !32
  store half %1918, ptr addrspace(1) %address.local1322, align 2, !dbg !32
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-152102-801zvuny/grid-000002")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 23, column: 3, scope: !3)
!8 = !DILocation(line: 70, column: 22, scope: !3)
!9 = !{i32 0, i32 256}
!10 = !DILocation(line: 72, column: 24, scope: !3)
!11 = !DILocation(line: 60, column: 24, scope: !3)
!12 = !DILocation(line: 77, column: 19, scope: !3)
!13 = !DILocation(line: 0, scope: !3)
!14 = !DILocation(line: 71, column: 22, scope: !3)
!15 = !DILocation(line: 73, column: 26, scope: !3)
!16 = !DILocation(line: 80, column: 23, scope: !3)
!17 = !DILocation(line: 101, column: 20, scope: !3)
!18 = !DILocation(line: 81, column: 26, scope: !3)
!19 = !DILocation(line: 88, column: 21, scope: !3)
!20 = !DILocation(line: 89, column: 20, scope: !3)
!21 = !DILocation(line: 90, column: 22, scope: !3)
!22 = !DILocation(line: 91, column: 23, scope: !3)
!23 = !DILocation(line: 92, column: 20, scope: !3)
!24 = !DILocation(line: 103, column: 22, scope: !3)
!25 = !DILocation(line: 93, column: 22, scope: !3)
!26 = !DILocation(line: 94, column: 22, scope: !3)
!27 = !DILocation(line: 96, column: 16, scope: !3)
!28 = !DILocation(line: 95, column: 20, scope: !3)
!29 = !DILocation(line: 102, column: 23, scope: !3)
!30 = !DILocation(line: 107, column: 23, scope: !3)
!31 = !DILocation(line: 108, column: 19, scope: !3)
!32 = !DILocation(line: 109, column: 9, scope: !3)
