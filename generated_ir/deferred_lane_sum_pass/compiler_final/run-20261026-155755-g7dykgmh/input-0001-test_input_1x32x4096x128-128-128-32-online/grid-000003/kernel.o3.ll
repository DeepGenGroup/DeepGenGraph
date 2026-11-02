; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0001-test_input_1x32x4096x128-128-128-32-online/grid-000003/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0001-test_input_1x32x4096x128-128-128-32-online/grid-000003/input.mlir"

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
  %29 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %.idx = shl i64 %26, 8, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local505 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local506 = and i64 %address.local505, 1536, !dbg !8
  %address.local507 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local506, !dbg !8
  %address.local510 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local511 = and i64 %address.local510, 16256, !dbg !8
  %address.local512 = lshr i64 %address.local, 4, !dbg !8
  %address.local513 = and i64 %address.local512, 8, !dbg !8
  %address.local514 = or disjoint i64 %address.local511, %address.local513, !dbg !8
  %30 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %31 = or disjoint i64 %address.local514, %28, !dbg !8
  %32 = shl nuw nsw i64 %31, 1, !dbg !8
  %33 = add i64 %32, %.idx, !dbg !8
  %dma.buffer.offset1778 = trunc i64 %33 to i32, !dbg !8
  %34 = insertelement <2 x i64> poison, i64 %30, i64 0, !dbg !8
  %35 = bitcast <2 x i64> %34 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1779 = shufflevector <4 x i32> %35, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local516 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local524 = lshr i64 %address.local516, 4, !dbg !8
  %address.local525 = and i64 %address.local524, 24, !dbg !8
  %address.local526 = or disjoint i64 %address.local525, %address.local511, !dbg !8
  %36 = or disjoint i64 %address.local526, %28, !dbg !8
  %37 = shl nuw nsw i64 %36, 1, !dbg !8
  %38 = add i64 %37, %.idx, !dbg !8
  %dma.buffer.offset1775 = trunc i64 %38 to i32, !dbg !8
  %39 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %40 = sext i32 %39 to i64, !dbg !7
  %address.local528 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local506, !dbg !8
  %.idx3 = shl nuw nsw i64 %28, 1, !dbg !8
  %41 = add i64 %.idx3, %.idx, !dbg !8
  %address.local531.idx = shl nuw nsw i64 %address.local514, 1, !dbg !8
  %42 = or disjoint i64 %address.local531.idx, %41, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %dma.buffer.offset1772 = or disjoint i32 %43, 64, !dbg !8
  %address.local532 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %44 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local507, i64 0, !dbg !8
  %45 = insertelement <4 x ptr addrspace(3)> %44, ptr addrspace(3) %address.local519, i64 1, !dbg !8
  %46 = insertelement <4 x ptr addrspace(3)> %45, ptr addrspace(3) %address.local528, i64 2, !dbg !8
  %47 = insertelement <4 x ptr addrspace(3)> %46, ptr addrspace(3) %address.local532, i64 3, !dbg !8
  %48 = ptrtoint <4 x ptr addrspace(3)> %47 to <4 x i64>, !dbg !8
  %49 = trunc <4 x i64> %48 to <4 x i32>, !dbg !8
  %50 = extractelement <4 x i32> %49, i64 0, !dbg !8
  %51 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %dma.buffer.lds1780 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %51), !dbg !8
  %52 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1780, i32 %dma.buffer.offset1778, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %53 = extractelement <4 x i32> %49, i64 1, !dbg !8
  %54 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %dma.buffer.lds1777 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %54), !dbg !8
  %55 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1777, i32 %dma.buffer.offset1775, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %56 = extractelement <4 x i32> %49, i64 2, !dbg !8
  %57 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %56), !dbg !8
  %dma.buffer.lds1774 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %57), !dbg !8
  %58 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1774, i32 %dma.buffer.offset1772, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %59 = extractelement <4 x i32> %49, i64 3, !dbg !8
  %60 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %59), !dbg !8
  %address.local535.idx = shl nuw nsw i64 %address.local526, 1, !dbg !8
  %61 = or disjoint i64 %address.local535.idx, %41, !dbg !8
  %62 = trunc i64 %61 to i32, !dbg !8
  %dma.buffer.offset1769 = or disjoint i32 %62, 64, !dbg !8
  %dma.buffer.lds1771 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1771, i32 %dma.buffer.offset1769, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
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
  %64 = load <4 x half>, ptr addrspace(3) %address.local548, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local548, i64 4096, !dbg !10
  %65 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %66 = fmul <4 x half> %64, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local548, i64 1024, !dbg !10
  %67 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !10
  %68 = fmul <4 x half> %65, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local548, i64 5120, !dbg !10
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !10
  %70 = fmul <4 x half> %67, splat (half 1.275630e-01), !dbg !10
  %71 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1766 = or disjoint i32 %43, 128, !dbg !8
  %72 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1780, i32 %dma.buffer.offset1766, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %dma.buffer.offset1763 = or disjoint i32 %62, 128, !dbg !8
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1777, i32 %dma.buffer.offset1763, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local548, i64 8192, !dbg !10
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !10
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local548, i64 12288, !dbg !10
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !10
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local548, i64 9216, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !10
  %77 = fmul <4 x half> %74, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local548, i64 13312, !dbg !10
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !10
  %79 = fmul <4 x half> %75, splat (half 1.275630e-01), !dbg !10
  %80 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !10
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1760 = or disjoint i32 %43, 192, !dbg !8
  %82 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1774, i32 %dma.buffer.offset1760, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %dma.buffer.offset1757 = or disjoint i32 %62, 192, !dbg !8
  %83 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1771, i32 %dma.buffer.offset1757, <4 x i32> %dma.buffer.rsrc1779) #9, !dbg !8
  %84 = load <4 x half>, ptr addrspace(3) %address.local548, align 8, !dbg !10
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !10
  %87 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !10
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !10
  %89 = fmul <4 x half> %85, splat (half 1.275630e-01), !dbg !10
  %90 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !10
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !10
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !10
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !10
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !10
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !10
  %97 = fmul <4 x half> %93, splat (half 1.275630e-01), !dbg !10
  %98 = fmul <4 x half> %94, splat (half 1.275630e-01), !dbg !10
  %99 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %100 = or disjoint i64 %24, 127, !dbg !12
  %101 = add i64 %24, 128, !dbg !12
  %102 = icmp uge i64 %101, %100, !dbg !12
  %103 = sub nsw i64 32, %14, !dbg !13
  %104 = shl i64 %16, 7, !dbg !14
  %105 = and i64 %104, 3968, !dbg !14
  %106 = shl nsw i64 %40, 12, !dbg !14
  %107 = or disjoint i64 %105, %106, !dbg !14
  %108 = shl nuw nsw i32 %29, 2, !dbg !15
  %109 = and i32 %108, 12, !dbg !15
  %110 = and i32 %29, 1, !dbg !15
  %111 = or disjoint i32 %109, %110, !dbg !15
  %112 = trunc nuw nsw i32 %111 to i16, !dbg !15
  %.lhs.trunc = add nuw nsw i16 %112, 1, !dbg !15
  %113 = lshr i16 %.lhs.trunc, 1, !dbg !15
  %.sext = zext nneg i16 %113 to i64, !dbg !15
  %114 = icmp ult i64 %10, 1024, !dbg !11
  br i1 %114, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !11

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre105 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre106 = add nuw nsw i32 %.pre105, 64, !dbg !11
  %.pre107 = xor i32 %.pre105, 16, !dbg !11
  %.pre109 = and i32 %.pre106, 192, !dbg !11
  %.pre111 = xor i32 %.pre105, 32, !dbg !11
  br label %._crit_edge, !dbg !11

.lr.ph:                                           ; preds = %4
  %115 = zext nneg i32 %29 to i64, !dbg !8
  %116 = ptrtoint ptr addrspace(1) %1 to i64
  %117 = insertelement <2 x i64> poison, i64 %116, i64 0
  %118 = bitcast <2 x i64> %117 to <4 x i32>
  %dma.buffer.rsrc1755 = shufflevector <4 x i32> %118, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %119 = ptrtoint ptr addrspace(1) %2 to i64
  %120 = insertelement <2 x i64> poison, i64 %119, i64 0
  %121 = bitcast <2 x i64> %120 to <4 x i32>
  %dma.buffer.rsrc1731 = shufflevector <4 x i32> %121, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %122 = lshr i64 %115, 2
  %123 = and i64 %122, 48
  %124 = and i32 %29, 240
  %index.compact = zext nneg i32 %124 to i64
  %125 = sub nsw i64 %115, %index.compact
  %126 = add nsw i64 %125, %123
  %127 = add nsw i64 %126, %24
  %128 = and i64 %115, 48
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

137:                                              ; preds = %.lr.ph, %591
  %138 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1288, %591 ]
  %139 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1284, %591 ]
  %140 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1280, %591 ]
  %141 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1276, %591 ]
  %142 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %959, %591 ]
  %143 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1274, %591 ]
  %144 = phi i64 [ 0, %.lr.ph ], [ %1289, %591 ]
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
  %address.thread599 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !14, !range !9, !frisk.register_lifetimes !5
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
  %.idx912 = or disjoint i64 %151, %149, !dbg !14
  %155 = add i64 %.idx912, %154, !dbg !14
  %156 = add i64 %address.local615, %155, !dbg !14
  %.tr = trunc i64 %156 to i32, !dbg !14
  %dma.buffer.offset1754 = shl i32 %.tr, 1, !dbg !14
  %address.local617 = or disjoint i64 %address.local600, 256, !dbg !8
  %address.local618 = shl nuw nsw i64 %address.local617, 3, !dbg !8
  %address.local619 = and i64 %address.local618, 3584, !dbg !8
  %address.local620 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local619, !dbg !8
  %address.local623 = lshr i64 %address.local617, 4, !dbg !14
  %address.local624 = and i64 %address.local623, 24, !dbg !14
  %address.local632 = or disjoint i64 %address.local614, %address.local624, !dbg !14
  %157 = add i64 %address.local632, %155, !dbg !14
  %.tr17 = trunc i64 %157 to i32, !dbg !14
  %dma.buffer.offset1751 = shl i32 %.tr17, 1, !dbg !14
  %158 = add nuw nsw i64 %144, 1024, !dbg !14
  %159 = lshr i64 %158, 12, !dbg !14
  %160 = add nuw nsw i64 %159, %16, !dbg !14
  %161 = add nsw i64 %148, %107, !dbg !14
  %162 = shl i64 %160, 19, !dbg !14
  %163 = and i64 %162, 16252928, !dbg !14
  %address.local634 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local602, !dbg !8
  %164 = add i64 %154, %148, !dbg !14
  %.idx1822 = add i64 %164, %107, !dbg !14
  %165 = add i64 %.idx1822, %163, !dbg !14
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
  %.not30 = icmp eq i64 %181, 0, !dbg !10
  %address.local658 = select i1 %.not30, i64 0, i64 1016, !dbg !10
  %.idx31 = mul nuw nsw i64 %address.local600, 320, !dbg !15
  %182 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx31, !dbg !15
  %.idx32 = mul nsw i64 %address.local609, -1280, !dbg !15
  %183 = getelementptr i8, ptr addrspace(3) %182, i64 %.idx32, !dbg !15
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
  %.tr23 = trunc i64 %203 to i32, !dbg !14
  %204 = shl i32 %.tr23, 1, !dbg !14
  %dma.buffer.offset1748 = add i32 %204, 64, !dbg !14
  %205 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1750, i32 %dma.buffer.offset1748, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %206 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %180), !dbg !14
  %207 = add i64 %address.local632, %202, !dbg !14
  %.tr29 = trunc i64 %207 to i32, !dbg !14
  %208 = shl i32 %.tr29, 1, !dbg !14
  %dma.buffer.offset1745 = add i32 %208, 64, !dbg !14
  %dma.buffer.lds1747 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %206), !dbg !14
  %209 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1747, i32 %dma.buffer.offset1745, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %210 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !15
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %210, <4 x float> zeroinitializer) #8, !dbg !15
  %212 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !15
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %.sext) #8, !dbg !15, !frisk.register_lifetimes !5
  %address.local683 = add nsw i64 %address.base, %address.local682, !dbg !15
  %address.local684 = shl i64 %address.local683, 6, !dbg !15
  %address.local685 = sub i64 %address.local679, %address.local684, !dbg !15
  %213 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local685, !dbg !15
  %214 = getelementptr [2 x i8], ptr addrspace(3) %213, i64 %address.local654, !dbg !15
  %215 = getelementptr [2 x i8], ptr addrspace(3) %214, i64 %address.local658, !dbg !15
  %address.local689 = getelementptr i8, ptr addrspace(3) %215, i64 64, !dbg !15
  %216 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !15
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %212, <4 x float> %211) #8, !dbg !15
  %218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %216, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %215, i64 4160, !dbg !15
  %219 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !15
  %220 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !15
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %219, <4 x float> %218) #8, !dbg !15
  %222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %220, <4 x float> zeroinitializer) #8, !dbg !15
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !15
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %215, i64 192, !dbg !15
  %224 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !15
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %223, <4 x float> %222) #8, !dbg !15
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %215, i64 4288, !dbg !15
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !15
  %228 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !15
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %227, <4 x float> %226) #8, !dbg !15
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %228, <4 x float> zeroinitializer) #8, !dbg !15
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !15
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %215, i64 1088, !dbg !15
  %232 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !15
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %231, <4 x float> %230) #8, !dbg !15
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %232, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %215, i64 5184, !dbg !15
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !15
  %236 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !15
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %235, <4 x float> %234) #8, !dbg !15
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !15
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !15
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %215, i64 1216, !dbg !15
  %240 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !15
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %239, <4 x float> %238) #8, !dbg !15
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %240, <4 x float> zeroinitializer) #8, !dbg !15
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %215, i64 5312, !dbg !15
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !15
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %243, <4 x float> %242) #8, !dbg !15
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %210, <4 x float> zeroinitializer) #8, !dbg !15
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %212, <4 x float> %245) #8, !dbg !15
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %216, <4 x float> zeroinitializer) #8, !dbg !15
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %219, <4 x float> %247) #8, !dbg !15
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %220, <4 x float> zeroinitializer) #8, !dbg !15
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %223, <4 x float> %249) #8, !dbg !15
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !15
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %227, <4 x float> %251) #8, !dbg !15
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %228, <4 x float> zeroinitializer) #8, !dbg !15
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %231, <4 x float> %253) #8, !dbg !15
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %232, <4 x float> zeroinitializer) #8, !dbg !15
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %235, <4 x float> %255) #8, !dbg !15
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !15
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %239, <4 x float> %257) #8, !dbg !15
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %240, <4 x float> zeroinitializer) #8, !dbg !15
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %243, <4 x float> %259) #8, !dbg !15
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
  %267 = add i64 %.idx1822, %266, !dbg !14
  %268 = shl nsw i64 %264, 7, !dbg !14
  %269 = sub i64 %267, %268, !dbg !14
  %270 = add i64 %address.local615, %269, !dbg !14
  %.tr38 = trunc i64 %270 to i32, !dbg !14
  %271 = shl i32 %.tr38, 1, !dbg !14
  %dma.buffer.offset1742 = add i32 %271, 128, !dbg !14
  %272 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1756, i32 %dma.buffer.offset1742, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %273 = add i64 %address.local632, %269, !dbg !14
  %.tr44 = trunc i64 %273 to i32, !dbg !14
  %274 = shl i32 %.tr44, 1, !dbg !14
  %dma.buffer.offset1739 = add i32 %274, 128, !dbg !14
  %275 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1753, i32 %dma.buffer.offset1739, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !15
  %276 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !15
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %276, <4 x float> %217) #8, !dbg !15
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !15
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !15
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %215, i64 8256, !dbg !15
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !15
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %278, <4 x float> %277) #8, !dbg !15
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %279, <4 x float> %221) #8, !dbg !15
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %215, i64 12352, !dbg !15
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !15
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !15
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !15
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %282, <4 x float> %281) #8, !dbg !15
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %283, <4 x float> %225) #8, !dbg !15
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !15
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !15
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %215, i64 8384, !dbg !15
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !15
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %286, <4 x float> %285) #8, !dbg !15
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %287, <4 x float> %229) #8, !dbg !15
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %215, i64 12480, !dbg !15
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !15
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !15
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !15
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %290, <4 x float> %289) #8, !dbg !15
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %291, <4 x float> %233) #8, !dbg !15
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !15
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !15
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %215, i64 9280, !dbg !15
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !15
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %294, <4 x float> %293) #8, !dbg !15
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %295, <4 x float> %237) #8, !dbg !15
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %215, i64 13376, !dbg !15
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !15
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !15
  %299 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !15
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %298, <4 x float> %297) #8, !dbg !15
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %299, <4 x float> %241) #8, !dbg !15
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !15
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !15
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %215, i64 9408, !dbg !15
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !15
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %302, <4 x float> %301) #8, !dbg !15
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %303, <4 x float> %244) #8, !dbg !15
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %215, i64 13504, !dbg !15
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !15
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %306, <4 x float> %305) #8, !dbg !15
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %276, <4 x float> %246) #8, !dbg !15
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %278, <4 x float> %308) #8, !dbg !15
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %279, <4 x float> %248) #8, !dbg !15
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %282, <4 x float> %310) #8, !dbg !15
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %283, <4 x float> %250) #8, !dbg !15
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %286, <4 x float> %312) #8, !dbg !15
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %287, <4 x float> %252) #8, !dbg !15
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %290, <4 x float> %314) #8, !dbg !15
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %291, <4 x float> %254) #8, !dbg !15
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %294, <4 x float> %316) #8, !dbg !15
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %295, <4 x float> %256) #8, !dbg !15
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %298, <4 x float> %318) #8, !dbg !15
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %299, <4 x float> %258) #8, !dbg !15
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %302, <4 x float> %320) #8, !dbg !15
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %303, <4 x float> %260) #8, !dbg !15
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %306, <4 x float> %322) #8, !dbg !15
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
  %336 = add i64 %.idx1822, %335, !dbg !14
  %337 = shl nsw i64 %333, 7, !dbg !14
  %338 = sub i64 %336, %337, !dbg !14
  %339 = add i64 %address.local615, %338, !dbg !14
  %.tr50 = trunc i64 %339 to i32, !dbg !14
  %340 = shl i32 %.tr50, 1, !dbg !14
  %dma.buffer.offset1736 = add i32 %340, 192, !dbg !14
  %341 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1750, i32 %dma.buffer.offset1736, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %342 = add i64 %address.local632, %338, !dbg !14
  %.tr56 = trunc i64 %342 to i32, !dbg !14
  %343 = shl i32 %.tr56, 1, !dbg !14
  %dma.buffer.offset1733 = add i32 %343, 192, !dbg !14
  %344 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1747, i32 %dma.buffer.offset1733, <4 x i32> %dma.buffer.rsrc1755) #9, !dbg !14
  %345 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !15
  %346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %345, <4 x float> %280) #8, !dbg !15
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !15
  %348 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !15
  %349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %347, <4 x float> %346) #8, !dbg !15
  %350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %348, <4 x float> %284) #8, !dbg !15
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !15
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !15
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %351, <4 x float> %350) #8, !dbg !15
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %352, <4 x float> %288) #8, !dbg !15
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !15
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !15
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %355, <4 x float> %354) #8, !dbg !15
  %358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %356, <4 x float> %292) #8, !dbg !15
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !15
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !15
  %361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %359, <4 x float> %358) #8, !dbg !15
  %362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %360, <4 x float> %296) #8, !dbg !15
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !15
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !15
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %363, <4 x float> %362) #8, !dbg !15
  %366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %364, <4 x float> %300) #8, !dbg !15
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !15
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !15
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %367, <4 x float> %366) #8, !dbg !15
  %370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %368, <4 x float> %304) #8, !dbg !15
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !15
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !15
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %371, <4 x float> %370) #8, !dbg !15
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %372, <4 x float> %307) #8, !dbg !15
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !15
  %376 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %375, <4 x float> %374) #8, !dbg !15
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %345, <4 x float> %309) #8, !dbg !15
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %347, <4 x float> %377) #8, !dbg !15
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %348, <4 x float> %311) #8, !dbg !15
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %351, <4 x float> %379) #8, !dbg !15
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %352, <4 x float> %313) #8, !dbg !15
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %355, <4 x float> %381) #8, !dbg !15
  %383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %356, <4 x float> %315) #8, !dbg !15
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %359, <4 x float> %383) #8, !dbg !15
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %360, <4 x float> %317) #8, !dbg !15
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %363, <4 x float> %385) #8, !dbg !15
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %364, <4 x float> %319) #8, !dbg !15
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %367, <4 x float> %387) #8, !dbg !15
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %368, <4 x float> %321) #8, !dbg !15
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %371, <4 x float> %389) #8, !dbg !15
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %372, <4 x float> %323) #8, !dbg !15
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %375, <4 x float> %391) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %393 = add nuw nsw i64 %148, %16, !dbg !17
  %394 = shl i64 %393, 19, !dbg !17
  %395 = and i64 %394, 16252928, !dbg !17
  %address.local1018 = shl nuw nsw i64 %address.local600, 6, !dbg !17
  %address.local1019 = and i64 %address.local1018, 16128, !dbg !17
  %address.local1021 = and i64 %address.local601, 24, !dbg !17
  %.idx5760 = add i64 %395, %154, !dbg !17
  %396 = or disjoint i64 %address.local1021, %.idx5760, !dbg !17
  %397 = shl i64 %396, 1, !dbg !17
  %398 = or disjoint i64 %397, %address.local1019, !dbg !17
  %dma.buffer.offset1730 = trunc i64 %398 to i32, !dbg !17
  %399 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1756, i32 %dma.buffer.offset1730, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %address.local1024 = shl nuw nsw i64 %address.local617, 6, !dbg !17
  %address.local1025 = and i64 %address.local1024, 32512, !dbg !17
  %400 = or disjoint i64 %397, %address.local1025, !dbg !17
  %dma.buffer.offset1727 = trunc i64 %400 to i32, !dbg !17
  %401 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1753, i32 %dma.buffer.offset1727, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %402 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !15
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %402, <4 x float> %349) #8, !dbg !15
  %404 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !15
  %405 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !15
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %404, <4 x float> %403) #8, !dbg !15
  %407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %405, <4 x float> %353) #8, !dbg !15
  %408 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !15
  %409 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !15
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %408, <4 x float> %407) #8, !dbg !15
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %409, <4 x float> %357) #8, !dbg !15
  %412 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !15
  %413 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !15
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %412, <4 x float> %411) #8, !dbg !15
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %413, <4 x float> %361) #8, !dbg !15
  %416 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !15
  %417 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !15
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %416, <4 x float> %415) #8, !dbg !15
  %419 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %417, <4 x float> %365) #8, !dbg !15
  %420 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !15
  %421 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !15
  %422 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %420, <4 x float> %419) #8, !dbg !15
  %423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %421, <4 x float> %369) #8, !dbg !15
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !15
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !15
  %426 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %424, <4 x float> %423) #8, !dbg !15
  %427 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %425, <4 x float> %373) #8, !dbg !15
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !15
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !15
  %430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %428, <4 x float> %427) #8, !dbg !15
  %431 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %429, <4 x float> %376) #8, !dbg !15
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !15
  %433 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %432, <4 x float> %431) #8, !dbg !15
  %434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %402, <4 x float> %378) #8, !dbg !15
  %435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %404, <4 x float> %434) #8, !dbg !15
  %436 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %405, <4 x float> %380) #8, !dbg !15
  %437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %408, <4 x float> %436) #8, !dbg !15
  %438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %409, <4 x float> %382) #8, !dbg !15
  %439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %412, <4 x float> %438) #8, !dbg !15
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %413, <4 x float> %384) #8, !dbg !15
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %416, <4 x float> %440) #8, !dbg !15
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %417, <4 x float> %386) #8, !dbg !15
  %443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %420, <4 x float> %442) #8, !dbg !15
  %444 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %421, <4 x float> %388) #8, !dbg !15
  %445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %424, <4 x float> %444) #8, !dbg !15
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %425, <4 x float> %390) #8, !dbg !15
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %428, <4 x float> %446) #8, !dbg !15
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %429, <4 x float> %392) #8, !dbg !15
  %449 = or disjoint i64 %145, 127, !dbg !12
  %450 = icmp uge i64 %24, %449, !dbg !12
  %451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %432, <4 x float> %448) #8, !dbg !15
  %452 = and i1 %102, %450, !dbg !12
  br i1 %452, label %591, label %453, !dbg !12

453:                                              ; preds = %137
  %454 = or disjoint i64 %145, %128, !dbg !12
  %455 = lshr i64 %144, 25, !dbg !16
  %compare.high1323 = trunc nuw i64 %455 to i32, !dbg !16
  %compare.low.base1324 = trunc i64 %454 to i32, !dbg !16
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 1) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1327 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 2) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1331 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 3) #8, !dbg !16, !frisk.register_lifetimes !5
  %456 = insertelement <4 x i32> poison, i32 %compare.low.base1324, i64 0, !dbg !16
  %457 = insertelement <4 x i32> %456, i32 %compare.local.base, i64 1, !dbg !16
  %458 = insertelement <4 x i32> %457, i32 %compare.local.base1327, i64 2, !dbg !16
  %459 = insertelement <4 x i32> %458, i32 %compare.local.base1331, i64 3, !dbg !16
  %460 = or <4 x i32> %459, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %461 = icmp ugt <4 x i32> %136, %460, !dbg !16
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
  %462 = insertelement <4 x i32> poison, i32 %compare.local.base1415, i64 0, !dbg !16
  %463 = insertelement <4 x i32> %462, i32 %compare.local.base1419, i64 1, !dbg !16
  %464 = insertelement <4 x i32> %463, i32 %compare.local.base1423, i64 2, !dbg !16
  %465 = insertelement <4 x i32> %464, i32 %compare.local.base1427, i64 3, !dbg !16
  %466 = or <4 x i32> %465, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %467 = icmp ult <4 x i32> %466, %136, !dbg !16
  %468 = insertelement <4 x i32> poison, i32 %compare.local.base1399, i64 0, !dbg !16
  %469 = insertelement <4 x i32> %468, i32 %compare.local.base1403, i64 1, !dbg !16
  %470 = insertelement <4 x i32> %469, i32 %compare.local.base1407, i64 2, !dbg !16
  %471 = insertelement <4 x i32> %470, i32 %compare.local.base1411, i64 3, !dbg !16
  %472 = or <4 x i32> %471, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %473 = icmp ult <4 x i32> %472, %136, !dbg !16
  %474 = insertelement <4 x i32> poison, i32 %compare.local.base1383, i64 0, !dbg !16
  %475 = insertelement <4 x i32> %474, i32 %compare.local.base1387, i64 1, !dbg !16
  %476 = insertelement <4 x i32> %475, i32 %compare.local.base1391, i64 2, !dbg !16
  %477 = insertelement <4 x i32> %476, i32 %compare.local.base1395, i64 3, !dbg !16
  %478 = or <4 x i32> %477, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %479 = icmp ult <4 x i32> %478, %136, !dbg !16
  %480 = insertelement <4 x i32> poison, i32 %compare.local.base1367, i64 0, !dbg !16
  %481 = insertelement <4 x i32> %480, i32 %compare.local.base1371, i64 1, !dbg !16
  %482 = insertelement <4 x i32> %481, i32 %compare.local.base1375, i64 2, !dbg !16
  %483 = insertelement <4 x i32> %482, i32 %compare.local.base1379, i64 3, !dbg !16
  %484 = or <4 x i32> %483, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %485 = icmp ult <4 x i32> %484, %136, !dbg !16
  %486 = insertelement <4 x i32> poison, i32 %compare.local.base1351, i64 0, !dbg !16
  %487 = insertelement <4 x i32> %486, i32 %compare.local.base1355, i64 1, !dbg !16
  %488 = insertelement <4 x i32> %487, i32 %compare.local.base1359, i64 2, !dbg !16
  %489 = insertelement <4 x i32> %488, i32 %compare.local.base1363, i64 3, !dbg !16
  %490 = or <4 x i32> %489, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %491 = icmp ult <4 x i32> %490, %136, !dbg !16
  %492 = insertelement <4 x i32> poison, i32 %compare.local.base1335, i64 0, !dbg !16
  %493 = insertelement <4 x i32> %492, i32 %compare.local.base1339, i64 1, !dbg !16
  %494 = insertelement <4 x i32> %493, i32 %compare.local.base1343, i64 2, !dbg !16
  %495 = insertelement <4 x i32> %494, i32 %compare.local.base1347, i64 3, !dbg !16
  %496 = or <4 x i32> %495, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %497 = icmp ult <4 x i32> %496, %136, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1323, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1323, !dbg !16
  %498 = insertelement <4 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !16
  %499 = shufflevector <4 x i1> %498, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %500 = insertelement <4 x i1> poison, i1 %compare.high.order, i64 0, !dbg !16
  %501 = shufflevector <4 x i1> %500, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %502 = select <4 x i1> %499, <4 x i1> %461, <4 x i1> %501, !dbg !16
  %503 = select <4 x i1> %502, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %504 = select <4 x i1> %499, <4 x i1> %497, <4 x i1> %501, !dbg !16
  %505 = select <4 x i1> %504, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %506 = select <4 x i1> %499, <4 x i1> %491, <4 x i1> %501, !dbg !16
  %507 = select <4 x i1> %506, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %508 = select <4 x i1> %499, <4 x i1> %485, <4 x i1> %501, !dbg !16
  %509 = select <4 x i1> %508, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %510 = select <4 x i1> %499, <4 x i1> %479, <4 x i1> %501, !dbg !16
  %511 = select <4 x i1> %510, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %512 = select <4 x i1> %499, <4 x i1> %473, <4 x i1> %501, !dbg !16
  %513 = select <4 x i1> %512, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %514 = select <4 x i1> %499, <4 x i1> %467, <4 x i1> %501, !dbg !16
  %515 = select <4 x i1> %514, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %516 = insertelement <4 x i32> poison, i32 %compare.local.base1431, i64 0, !dbg !16
  %517 = insertelement <4 x i32> %516, i32 %compare.local.base1435, i64 1, !dbg !16
  %518 = insertelement <4 x i32> %517, i32 %compare.local.base1439, i64 2, !dbg !16
  %519 = insertelement <4 x i32> %518, i32 %compare.local.base1443, i64 3, !dbg !16
  %520 = or <4 x i32> %519, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %521 = icmp ult <4 x i32> %520, %136, !dbg !16
  %522 = select <4 x i1> %499, <4 x i1> %521, <4 x i1> %501, !dbg !16
  %523 = select <4 x i1> %522, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %compare.local.base1455 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 32) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1459 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 33) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1463 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 34) #8, !dbg !16, !frisk.register_lifetimes !5
  %524 = insertelement <4 x i32> %456, i32 %compare.local.base1455, i64 1, !dbg !16
  %525 = insertelement <4 x i32> %524, i32 %compare.local.base1459, i64 2, !dbg !16
  %526 = insertelement <4 x i32> %525, i32 %compare.local.base1463, i64 3, !dbg !16
  %527 = or <4 x i32> %526, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %528 = icmp ugt <4 x i32> %134, %527, !dbg !16
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
  %529 = insertelement <4 x i32> poison, i32 %compare.local.base1547, i64 0, !dbg !16
  %530 = insertelement <4 x i32> %529, i32 %compare.local.base1551, i64 1, !dbg !16
  %531 = insertelement <4 x i32> %530, i32 %compare.local.base1555, i64 2, !dbg !16
  %532 = insertelement <4 x i32> %531, i32 %compare.local.base1559, i64 3, !dbg !16
  %533 = or <4 x i32> %532, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %534 = icmp ult <4 x i32> %533, %134, !dbg !16
  %535 = insertelement <4 x i32> poison, i32 %compare.local.base1531, i64 0, !dbg !16
  %536 = insertelement <4 x i32> %535, i32 %compare.local.base1535, i64 1, !dbg !16
  %537 = insertelement <4 x i32> %536, i32 %compare.local.base1539, i64 2, !dbg !16
  %538 = insertelement <4 x i32> %537, i32 %compare.local.base1543, i64 3, !dbg !16
  %539 = or <4 x i32> %538, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %540 = icmp ult <4 x i32> %539, %134, !dbg !16
  %541 = insertelement <4 x i32> poison, i32 %compare.local.base1515, i64 0, !dbg !16
  %542 = insertelement <4 x i32> %541, i32 %compare.local.base1519, i64 1, !dbg !16
  %543 = insertelement <4 x i32> %542, i32 %compare.local.base1523, i64 2, !dbg !16
  %544 = insertelement <4 x i32> %543, i32 %compare.local.base1527, i64 3, !dbg !16
  %545 = or <4 x i32> %544, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %546 = icmp ult <4 x i32> %545, %134, !dbg !16
  %547 = insertelement <4 x i32> poison, i32 %compare.local.base1499, i64 0, !dbg !16
  %548 = insertelement <4 x i32> %547, i32 %compare.local.base1503, i64 1, !dbg !16
  %549 = insertelement <4 x i32> %548, i32 %compare.local.base1507, i64 2, !dbg !16
  %550 = insertelement <4 x i32> %549, i32 %compare.local.base1511, i64 3, !dbg !16
  %551 = or <4 x i32> %550, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %552 = icmp ult <4 x i32> %551, %134, !dbg !16
  %553 = insertelement <4 x i32> poison, i32 %compare.local.base1483, i64 0, !dbg !16
  %554 = insertelement <4 x i32> %553, i32 %compare.local.base1487, i64 1, !dbg !16
  %555 = insertelement <4 x i32> %554, i32 %compare.local.base1491, i64 2, !dbg !16
  %556 = insertelement <4 x i32> %555, i32 %compare.local.base1495, i64 3, !dbg !16
  %557 = or <4 x i32> %556, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %558 = icmp ult <4 x i32> %557, %134, !dbg !16
  %559 = insertelement <4 x i32> poison, i32 %compare.local.base1467, i64 0, !dbg !16
  %560 = insertelement <4 x i32> %559, i32 %compare.local.base1471, i64 1, !dbg !16
  %561 = insertelement <4 x i32> %560, i32 %compare.local.base1475, i64 2, !dbg !16
  %562 = insertelement <4 x i32> %561, i32 %compare.local.base1479, i64 3, !dbg !16
  %563 = or <4 x i32> %562, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %564 = icmp ult <4 x i32> %563, %134, !dbg !16
  %compare.high.equal1449 = icmp eq i32 %compare.high1447, %compare.high1323, !dbg !16
  %compare.high.order1450 = icmp ugt i32 %compare.high1447, %compare.high1323, !dbg !16
  %565 = insertelement <4 x i1> poison, i1 %compare.high.equal1449, i64 0, !dbg !16
  %566 = shufflevector <4 x i1> %565, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %567 = insertelement <4 x i1> poison, i1 %compare.high.order1450, i64 0, !dbg !16
  %568 = shufflevector <4 x i1> %567, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %569 = select <4 x i1> %566, <4 x i1> %528, <4 x i1> %568, !dbg !16
  %570 = select <4 x i1> %569, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %571 = select <4 x i1> %566, <4 x i1> %564, <4 x i1> %568, !dbg !16
  %572 = select <4 x i1> %571, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %573 = select <4 x i1> %566, <4 x i1> %558, <4 x i1> %568, !dbg !16
  %574 = select <4 x i1> %573, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %575 = select <4 x i1> %566, <4 x i1> %552, <4 x i1> %568, !dbg !16
  %576 = select <4 x i1> %575, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %577 = select <4 x i1> %566, <4 x i1> %546, <4 x i1> %568, !dbg !16
  %578 = select <4 x i1> %577, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %579 = select <4 x i1> %566, <4 x i1> %540, <4 x i1> %568, !dbg !16
  %580 = select <4 x i1> %579, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %581 = select <4 x i1> %566, <4 x i1> %534, <4 x i1> %568, !dbg !16
  %582 = select <4 x i1> %581, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %583 = insertelement <4 x i32> poison, i32 %compare.local.base1563, i64 0, !dbg !16
  %584 = insertelement <4 x i32> %583, i32 %compare.local.base1567, i64 1, !dbg !16
  %585 = insertelement <4 x i32> %584, i32 %compare.local.base1571, i64 2, !dbg !16
  %586 = insertelement <4 x i32> %585, i32 %compare.local.base1575, i64 3, !dbg !16
  %587 = or <4 x i32> %586, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %588 = icmp ult <4 x i32> %587, %134, !dbg !16
  %589 = select <4 x i1> %566, <4 x i1> %588, <4 x i1> %568, !dbg !16
  %590 = select <4 x i1> %589, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !18
  %.masked127 = fadd <4 x float> %406, %503, !dbg !19
  %.masked129 = fadd <4 x float> %410, %505, !dbg !19
  %.masked131 = fadd <4 x float> %414, %507, !dbg !19
  %.masked133 = fadd <4 x float> %418, %509, !dbg !19
  %.masked135 = fadd <4 x float> %422, %511, !dbg !19
  %.masked137 = fadd <4 x float> %426, %513, !dbg !19
  %.masked139 = fadd <4 x float> %430, %515, !dbg !19
  %.masked141 = fadd <4 x float> %433, %523, !dbg !19
  %.masked144 = fadd <4 x float> %435, %570, !dbg !19
  %.masked146 = fadd <4 x float> %437, %572, !dbg !19
  %.masked148 = fadd <4 x float> %439, %574, !dbg !19
  %.masked150 = fadd <4 x float> %441, %576, !dbg !19
  %.masked152 = fadd <4 x float> %443, %578, !dbg !19
  %.masked154 = fadd <4 x float> %445, %580, !dbg !19
  %.masked156 = fadd <4 x float> %447, %582, !dbg !19
  %.masked158 = fadd <4 x float> %451, %590, !dbg !19
  br label %591, !dbg !12

591:                                              ; preds = %453, %137
  %592 = phi <4 x float> [ %.masked127, %453 ], [ %406, %137 ], !dbg !19
  %593 = phi <4 x float> [ %.masked129, %453 ], [ %410, %137 ], !dbg !19
  %594 = phi <4 x float> [ %.masked131, %453 ], [ %414, %137 ], !dbg !19
  %595 = phi <4 x float> [ %.masked133, %453 ], [ %418, %137 ], !dbg !19
  %596 = phi <4 x float> [ %.masked135, %453 ], [ %422, %137 ], !dbg !19
  %597 = phi <4 x float> [ %.masked137, %453 ], [ %426, %137 ], !dbg !19
  %598 = phi <4 x float> [ %.masked139, %453 ], [ %430, %137 ], !dbg !19
  %599 = phi <4 x float> [ %.masked141, %453 ], [ %433, %137 ], !dbg !19
  %600 = phi <4 x float> [ %.masked144, %453 ], [ %435, %137 ], !dbg !19
  %601 = phi <4 x float> [ %.masked146, %453 ], [ %437, %137 ], !dbg !19
  %602 = phi <4 x float> [ %.masked148, %453 ], [ %439, %137 ], !dbg !19
  %603 = phi <4 x float> [ %.masked150, %453 ], [ %441, %137 ], !dbg !19
  %604 = phi <4 x float> [ %.masked152, %453 ], [ %443, %137 ], !dbg !19
  %605 = phi <4 x float> [ %.masked154, %453 ], [ %445, %137 ], !dbg !19
  %606 = phi <4 x float> [ %.masked156, %453 ], [ %447, %137 ], !dbg !19
  %607 = phi <4 x float> [ %.masked158, %453 ], [ %451, %137 ], !dbg !19
  %608 = extractelement <4 x float> %592, i64 0, !dbg !19
  %609 = extractelement <4 x float> %592, i64 1, !dbg !19
  %610 = extractelement <4 x float> %592, i64 2, !dbg !19
  %611 = extractelement <4 x float> %592, i64 3, !dbg !19
  %612 = extractelement <4 x float> %593, i64 0, !dbg !19
  %613 = extractelement <4 x float> %593, i64 1, !dbg !19
  %614 = extractelement <4 x float> %593, i64 2, !dbg !19
  %615 = extractelement <4 x float> %593, i64 3, !dbg !19
  %616 = extractelement <4 x float> %594, i64 0, !dbg !19
  %617 = extractelement <4 x float> %594, i64 1, !dbg !19
  %618 = extractelement <4 x float> %594, i64 2, !dbg !19
  %619 = extractelement <4 x float> %594, i64 3, !dbg !19
  %620 = extractelement <4 x float> %595, i64 0, !dbg !19
  %621 = extractelement <4 x float> %595, i64 1, !dbg !19
  %622 = extractelement <4 x float> %595, i64 2, !dbg !19
  %623 = extractelement <4 x float> %595, i64 3, !dbg !19
  %624 = extractelement <4 x float> %596, i64 0, !dbg !19
  %625 = extractelement <4 x float> %596, i64 1, !dbg !19
  %626 = extractelement <4 x float> %596, i64 2, !dbg !19
  %627 = extractelement <4 x float> %596, i64 3, !dbg !19
  %628 = extractelement <4 x float> %597, i64 0, !dbg !19
  %629 = extractelement <4 x float> %597, i64 1, !dbg !19
  %630 = extractelement <4 x float> %597, i64 2, !dbg !19
  %631 = extractelement <4 x float> %597, i64 3, !dbg !19
  %632 = extractelement <4 x float> %598, i64 0, !dbg !19
  %633 = extractelement <4 x float> %598, i64 1, !dbg !19
  %634 = extractelement <4 x float> %598, i64 2, !dbg !19
  %635 = extractelement <4 x float> %598, i64 3, !dbg !19
  %636 = extractelement <4 x float> %599, i64 0, !dbg !19
  %637 = extractelement <4 x float> %599, i64 1, !dbg !19
  %638 = extractelement <4 x float> %599, i64 2, !dbg !19
  %639 = extractelement <4 x float> %599, i64 3, !dbg !19
  %640 = extractelement <4 x float> %600, i64 0, !dbg !19
  %641 = extractelement <4 x float> %600, i64 1, !dbg !19
  %642 = extractelement <4 x float> %600, i64 2, !dbg !19
  %643 = extractelement <4 x float> %600, i64 3, !dbg !19
  %644 = extractelement <4 x float> %601, i64 0, !dbg !19
  %645 = extractelement <4 x float> %601, i64 1, !dbg !19
  %646 = extractelement <4 x float> %601, i64 2, !dbg !19
  %647 = extractelement <4 x float> %601, i64 3, !dbg !19
  %648 = extractelement <4 x float> %602, i64 0, !dbg !19
  %649 = extractelement <4 x float> %602, i64 1, !dbg !19
  %650 = extractelement <4 x float> %602, i64 2, !dbg !19
  %651 = extractelement <4 x float> %602, i64 3, !dbg !19
  %652 = extractelement <4 x float> %603, i64 0, !dbg !19
  %653 = extractelement <4 x float> %603, i64 1, !dbg !19
  %654 = extractelement <4 x float> %603, i64 2, !dbg !19
  %655 = extractelement <4 x float> %603, i64 3, !dbg !19
  %656 = extractelement <4 x float> %604, i64 0, !dbg !19
  %657 = extractelement <4 x float> %604, i64 1, !dbg !19
  %658 = extractelement <4 x float> %604, i64 2, !dbg !19
  %659 = extractelement <4 x float> %604, i64 3, !dbg !19
  %660 = extractelement <4 x float> %605, i64 0, !dbg !19
  %661 = extractelement <4 x float> %605, i64 1, !dbg !19
  %662 = extractelement <4 x float> %605, i64 2, !dbg !19
  %663 = extractelement <4 x float> %605, i64 3, !dbg !19
  %664 = extractelement <4 x float> %606, i64 0, !dbg !19
  %665 = extractelement <4 x float> %606, i64 1, !dbg !19
  %666 = extractelement <4 x float> %606, i64 2, !dbg !19
  %667 = extractelement <4 x float> %606, i64 3, !dbg !19
  %668 = extractelement <4 x float> %607, i64 0, !dbg !19
  %669 = extractelement <4 x float> %607, i64 1, !dbg !19
  %670 = extractelement <4 x float> %607, i64 2, !dbg !19
  %671 = extractelement <4 x float> %607, i64 3, !dbg !19
  %672 = tail call float @llvm.maxnum.f32(float %608, float -inf), !dbg !20, !frisk.local_reduction !5
  %local.reduce251 = tail call float @llvm.maxnum.f32(float %672, float %612), !dbg !20, !frisk.local_reduction !5
  %local.reduce252 = tail call float @llvm.maxnum.f32(float %609, float %613), !dbg !20, !frisk.local_reduction !5
  %local.reduce253 = tail call float @llvm.maxnum.f32(float %610, float %614), !dbg !20, !frisk.local_reduction !5
  %local.reduce254 = tail call float @llvm.maxnum.f32(float %611, float %615), !dbg !20, !frisk.local_reduction !5
  %local.reduce255 = tail call float @llvm.maxnum.f32(float %local.reduce251, float %616), !dbg !20, !frisk.local_reduction !5
  %local.reduce256 = tail call float @llvm.maxnum.f32(float %local.reduce252, float %617), !dbg !20, !frisk.local_reduction !5
  %local.reduce257 = tail call float @llvm.maxnum.f32(float %local.reduce253, float %618), !dbg !20, !frisk.local_reduction !5
  %local.reduce258 = tail call float @llvm.maxnum.f32(float %local.reduce254, float %619), !dbg !20, !frisk.local_reduction !5
  %local.reduce259 = tail call float @llvm.maxnum.f32(float %local.reduce255, float %620), !dbg !20, !frisk.local_reduction !5
  %local.reduce260 = tail call float @llvm.maxnum.f32(float %local.reduce256, float %621), !dbg !20, !frisk.local_reduction !5
  %local.reduce261 = tail call float @llvm.maxnum.f32(float %local.reduce257, float %622), !dbg !20, !frisk.local_reduction !5
  %local.reduce262 = tail call float @llvm.maxnum.f32(float %local.reduce258, float %623), !dbg !20, !frisk.local_reduction !5
  %local.reduce263 = tail call float @llvm.maxnum.f32(float %local.reduce259, float %624), !dbg !20, !frisk.local_reduction !5
  %local.reduce264 = tail call float @llvm.maxnum.f32(float %local.reduce260, float %625), !dbg !20, !frisk.local_reduction !5
  %local.reduce265 = tail call float @llvm.maxnum.f32(float %local.reduce261, float %626), !dbg !20, !frisk.local_reduction !5
  %local.reduce266 = tail call float @llvm.maxnum.f32(float %local.reduce262, float %627), !dbg !20, !frisk.local_reduction !5
  %local.reduce267 = tail call float @llvm.maxnum.f32(float %local.reduce263, float %628), !dbg !20, !frisk.local_reduction !5
  %local.reduce268 = tail call float @llvm.maxnum.f32(float %local.reduce264, float %629), !dbg !20, !frisk.local_reduction !5
  %local.reduce269 = tail call float @llvm.maxnum.f32(float %local.reduce265, float %630), !dbg !20, !frisk.local_reduction !5
  %local.reduce270 = tail call float @llvm.maxnum.f32(float %local.reduce266, float %631), !dbg !20, !frisk.local_reduction !5
  %local.reduce271 = tail call float @llvm.maxnum.f32(float %local.reduce267, float %632), !dbg !20, !frisk.local_reduction !5
  %local.reduce272 = tail call float @llvm.maxnum.f32(float %local.reduce268, float %633), !dbg !20, !frisk.local_reduction !5
  %local.reduce273 = tail call float @llvm.maxnum.f32(float %local.reduce269, float %634), !dbg !20, !frisk.local_reduction !5
  %local.reduce274 = tail call float @llvm.maxnum.f32(float %local.reduce270, float %635), !dbg !20, !frisk.local_reduction !5
  %local.reduce275 = tail call float @llvm.maxnum.f32(float %local.reduce271, float %636), !dbg !20, !frisk.local_reduction !5
  %local.reduce276 = tail call float @llvm.maxnum.f32(float %local.reduce272, float %637), !dbg !20, !frisk.local_reduction !5
  %local.reduce277 = tail call float @llvm.maxnum.f32(float %local.reduce273, float %638), !dbg !20, !frisk.local_reduction !5
  %local.reduce278 = tail call float @llvm.maxnum.f32(float %local.reduce274, float %639), !dbg !20, !frisk.local_reduction !5
  %local.reduce279 = tail call float @llvm.maxnum.f32(float %local.reduce275, float %local.reduce276), !dbg !20, !frisk.local_reduction !5
  %local.reduce280 = tail call float @llvm.maxnum.f32(float %local.reduce277, float %local.reduce278), !dbg !20, !frisk.local_reduction !5
  %673 = tail call float @llvm.maxnum.f32(float %local.reduce279, float %local.reduce280), !dbg !20, !frisk.local_reduction !5
  %674 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %675 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %674), !dbg !20
  %676 = add nuw nsw i32 %675, 64, !dbg !20
  %677 = xor i32 %675, 16, !dbg !20
  %678 = and i32 %676, 192, !dbg !20
  %679 = icmp samesign ult i32 %677, %678, !dbg !20
  %680 = select i1 %679, i32 %677, i32 %675, !dbg !20
  %681 = shl nuw nsw i32 %680, 2, !dbg !20
  %682 = bitcast float %673 to i32, !dbg !20
  %683 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %682), !dbg !20
  %684 = bitcast i32 %683 to float, !dbg !20
  %685 = tail call float @llvm.maxnum.f32(float %673, float %684), !dbg !20
  %686 = xor i32 %675, 32, !dbg !20
  %687 = icmp samesign ult i32 %686, %678, !dbg !20
  %688 = select i1 %687, i32 %686, i32 %675, !dbg !20
  %689 = shl nuw nsw i32 %688, 2, !dbg !20
  %690 = bitcast float %685 to i32, !dbg !20
  %691 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %689, i32 %690), !dbg !20
  %692 = bitcast i32 %691 to float, !dbg !20
  %693 = tail call float @llvm.maxnum.f32(float %685, float %692), !dbg !20
  %694 = extractvalue [2 x <1 x float>] %141, 0, !dbg !20
  %695 = insertelement <1 x float> poison, float %693, i64 0, !dbg !20
  %696 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %695, <1 x float> %694), !dbg !20
  %697 = tail call float @llvm.maxnum.f32(float %640, float -inf), !dbg !20, !frisk.local_reduction !5
  %local.reduce220 = tail call float @llvm.maxnum.f32(float %697, float %644), !dbg !20, !frisk.local_reduction !5
  %local.reduce221 = tail call float @llvm.maxnum.f32(float %641, float %645), !dbg !20, !frisk.local_reduction !5
  %local.reduce222 = tail call float @llvm.maxnum.f32(float %642, float %646), !dbg !20, !frisk.local_reduction !5
  %local.reduce223 = tail call float @llvm.maxnum.f32(float %643, float %647), !dbg !20, !frisk.local_reduction !5
  %local.reduce224 = tail call float @llvm.maxnum.f32(float %local.reduce220, float %648), !dbg !20, !frisk.local_reduction !5
  %local.reduce225 = tail call float @llvm.maxnum.f32(float %local.reduce221, float %649), !dbg !20, !frisk.local_reduction !5
  %local.reduce226 = tail call float @llvm.maxnum.f32(float %local.reduce222, float %650), !dbg !20, !frisk.local_reduction !5
  %local.reduce227 = tail call float @llvm.maxnum.f32(float %local.reduce223, float %651), !dbg !20, !frisk.local_reduction !5
  %local.reduce228 = tail call float @llvm.maxnum.f32(float %local.reduce224, float %652), !dbg !20, !frisk.local_reduction !5
  %local.reduce229 = tail call float @llvm.maxnum.f32(float %local.reduce225, float %653), !dbg !20, !frisk.local_reduction !5
  %local.reduce230 = tail call float @llvm.maxnum.f32(float %local.reduce226, float %654), !dbg !20, !frisk.local_reduction !5
  %local.reduce231 = tail call float @llvm.maxnum.f32(float %local.reduce227, float %655), !dbg !20, !frisk.local_reduction !5
  %local.reduce232 = tail call float @llvm.maxnum.f32(float %local.reduce228, float %656), !dbg !20, !frisk.local_reduction !5
  %local.reduce233 = tail call float @llvm.maxnum.f32(float %local.reduce229, float %657), !dbg !20, !frisk.local_reduction !5
  %local.reduce234 = tail call float @llvm.maxnum.f32(float %local.reduce230, float %658), !dbg !20, !frisk.local_reduction !5
  %local.reduce235 = tail call float @llvm.maxnum.f32(float %local.reduce231, float %659), !dbg !20, !frisk.local_reduction !5
  %local.reduce236 = tail call float @llvm.maxnum.f32(float %local.reduce232, float %660), !dbg !20, !frisk.local_reduction !5
  %local.reduce237 = tail call float @llvm.maxnum.f32(float %local.reduce233, float %661), !dbg !20, !frisk.local_reduction !5
  %local.reduce238 = tail call float @llvm.maxnum.f32(float %local.reduce234, float %662), !dbg !20, !frisk.local_reduction !5
  %local.reduce239 = tail call float @llvm.maxnum.f32(float %local.reduce235, float %663), !dbg !20, !frisk.local_reduction !5
  %local.reduce240 = tail call float @llvm.maxnum.f32(float %local.reduce236, float %664), !dbg !20, !frisk.local_reduction !5
  %local.reduce241 = tail call float @llvm.maxnum.f32(float %local.reduce237, float %665), !dbg !20, !frisk.local_reduction !5
  %local.reduce242 = tail call float @llvm.maxnum.f32(float %local.reduce238, float %666), !dbg !20, !frisk.local_reduction !5
  %local.reduce243 = tail call float @llvm.maxnum.f32(float %local.reduce239, float %667), !dbg !20, !frisk.local_reduction !5
  %local.reduce244 = tail call float @llvm.maxnum.f32(float %local.reduce240, float %668), !dbg !20, !frisk.local_reduction !5
  %local.reduce245 = tail call float @llvm.maxnum.f32(float %local.reduce241, float %669), !dbg !20, !frisk.local_reduction !5
  %local.reduce246 = tail call float @llvm.maxnum.f32(float %local.reduce242, float %670), !dbg !20, !frisk.local_reduction !5
  %local.reduce247 = tail call float @llvm.maxnum.f32(float %local.reduce243, float %671), !dbg !20, !frisk.local_reduction !5
  %local.reduce248 = tail call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !20, !frisk.local_reduction !5
  %local.reduce249 = tail call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !20, !frisk.local_reduction !5
  %698 = tail call float @llvm.maxnum.f32(float %local.reduce248, float %local.reduce249), !dbg !20, !frisk.local_reduction !5
  %699 = bitcast float %698 to i32, !dbg !20
  %700 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %699), !dbg !20
  %701 = bitcast i32 %700 to float, !dbg !20
  %702 = tail call float @llvm.maxnum.f32(float %698, float %701), !dbg !20
  %703 = bitcast float %702 to i32, !dbg !20
  %704 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %689, i32 %703), !dbg !20
  %705 = bitcast i32 %704 to float, !dbg !20
  %706 = tail call float @llvm.maxnum.f32(float %702, float %705), !dbg !20
  %707 = extractvalue [2 x <1 x float>] %141, 1, !dbg !20
  %708 = insertelement <1 x float> poison, float %706, i64 0, !dbg !20
  %709 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %708, <1 x float> %707), !dbg !20
  %710 = shufflevector <1 x float> %696, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %711 = fsub <4 x float> %592, %710, !dbg !21
  %712 = fsub <4 x float> %593, %710, !dbg !21
  %713 = fsub <4 x float> %594, %710, !dbg !21
  %714 = fsub <4 x float> %595, %710, !dbg !21
  %715 = fsub <4 x float> %596, %710, !dbg !21
  %716 = fsub <4 x float> %597, %710, !dbg !21
  %717 = fsub <4 x float> %598, %710, !dbg !21
  %718 = fsub <4 x float> %599, %710, !dbg !21
  %719 = shufflevector <1 x float> %709, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %720 = fsub <4 x float> %600, %719, !dbg !21
  %721 = fsub <4 x float> %601, %719, !dbg !21
  %722 = fsub <4 x float> %602, %719, !dbg !21
  %723 = fsub <4 x float> %603, %719, !dbg !21
  %724 = fsub <4 x float> %604, %719, !dbg !21
  %725 = fsub <4 x float> %605, %719, !dbg !21
  %726 = fsub <4 x float> %606, %719, !dbg !21
  %727 = fsub <4 x float> %607, %719, !dbg !21
  %728 = fsub <1 x float> %694, %696, !dbg !22
  %729 = extractelement <1 x float> %728, i64 0, !dbg !23
  %730 = tail call float @llvm.amdgcn.exp2.f32(float %729), !dbg !23
  %731 = insertelement <4 x float> poison, float %730, i64 0, !dbg !24
  %732 = shufflevector <4 x float> %731, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %733 = extractvalue [2 x <1 x float>] %142, 0, !dbg !25
  %734 = extractelement <1 x float> %733, i64 0, !dbg !25
  %735 = fmul float %734, %730, !dbg !25
  %736 = fsub <1 x float> %707, %709, !dbg !22
  %737 = extractelement <1 x float> %736, i64 0, !dbg !23
  %738 = tail call float @llvm.amdgcn.exp2.f32(float %737), !dbg !23
  %739 = insertelement <4 x float> poison, float %738, i64 0, !dbg !24
  %740 = shufflevector <4 x float> %739, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %741 = extractvalue [2 x <1 x float>] %142, 1, !dbg !25
  %742 = extractelement <1 x float> %741, i64 0, !dbg !25
  %743 = fmul float %742, %738, !dbg !25
  %744 = extractelement <4 x float> %711, i64 0, !dbg !26
  %745 = tail call float @llvm.amdgcn.exp2.f32(float %744), !dbg !26
  %746 = insertelement <4 x float> poison, float %745, i64 0, !dbg !27
  %747 = extractelement <4 x float> %711, i64 1, !dbg !26
  %748 = tail call float @llvm.amdgcn.exp2.f32(float %747), !dbg !26
  %749 = insertelement <4 x float> %746, float %748, i64 1, !dbg !27
  %750 = extractelement <4 x float> %711, i64 2, !dbg !26
  %751 = tail call float @llvm.amdgcn.exp2.f32(float %750), !dbg !26
  %752 = insertelement <4 x float> %749, float %751, i64 2, !dbg !27
  %753 = extractelement <4 x float> %711, i64 3, !dbg !26
  %754 = tail call float @llvm.amdgcn.exp2.f32(float %753), !dbg !26
  %755 = insertelement <4 x float> %752, float %754, i64 3, !dbg !27
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %755) #8, !dbg !27, !frisk.register_lifetimes !5
  %756 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %756) #8, !dbg !26, !frisk.register_lifetimes !5
  %757 = extractelement <4 x float> %712, i64 0, !dbg !26
  %758 = tail call float @llvm.amdgcn.exp2.f32(float %757), !dbg !26
  %759 = insertelement <4 x float> poison, float %758, i64 0, !dbg !27
  %local.reduce189 = fadd float %745, %758, !dbg !28, !frisk.local_reduction !5
  %760 = extractelement <4 x float> %712, i64 1, !dbg !26
  %761 = tail call float @llvm.amdgcn.exp2.f32(float %760), !dbg !26
  %762 = insertelement <4 x float> %759, float %761, i64 1, !dbg !27
  %local.reduce190 = fadd float %748, %761, !dbg !28, !frisk.local_reduction !5
  %763 = extractelement <4 x float> %712, i64 2, !dbg !26
  %764 = tail call float @llvm.amdgcn.exp2.f32(float %763), !dbg !26
  %765 = insertelement <4 x float> %762, float %764, i64 2, !dbg !27
  %local.reduce191 = fadd float %751, %764, !dbg !28, !frisk.local_reduction !5
  %766 = extractelement <4 x float> %712, i64 3, !dbg !26
  %767 = tail call float @llvm.amdgcn.exp2.f32(float %766), !dbg !26
  %768 = insertelement <4 x float> %765, float %767, i64 3, !dbg !27
  %math.result475 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %768) #8, !dbg !27, !frisk.register_lifetimes !5
  %769 = fptrunc <4 x float> %math.result475 to <4 x half>, !dbg !27
  %math.packed476 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %769) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce192 = fadd float %754, %767, !dbg !28, !frisk.local_reduction !5
  %770 = extractelement <4 x float> %713, i64 0, !dbg !26
  %771 = tail call float @llvm.amdgcn.exp2.f32(float %770), !dbg !26
  %772 = insertelement <4 x float> poison, float %771, i64 0, !dbg !27
  %local.reduce193 = fadd float %771, %local.reduce189, !dbg !28, !frisk.local_reduction !5
  %773 = extractelement <4 x float> %713, i64 1, !dbg !26
  %774 = tail call float @llvm.amdgcn.exp2.f32(float %773), !dbg !26
  %775 = insertelement <4 x float> %772, float %774, i64 1, !dbg !27
  %local.reduce194 = fadd float %774, %local.reduce190, !dbg !28, !frisk.local_reduction !5
  %776 = extractelement <4 x float> %713, i64 2, !dbg !26
  %777 = tail call float @llvm.amdgcn.exp2.f32(float %776), !dbg !26
  %778 = insertelement <4 x float> %775, float %777, i64 2, !dbg !27
  %local.reduce195 = fadd float %777, %local.reduce191, !dbg !28, !frisk.local_reduction !5
  %779 = extractelement <4 x float> %713, i64 3, !dbg !26
  %780 = tail call float @llvm.amdgcn.exp2.f32(float %779), !dbg !26
  %781 = insertelement <4 x float> %778, float %780, i64 3, !dbg !27
  %math.result477 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %781) #8, !dbg !27, !frisk.register_lifetimes !5
  %782 = fptrunc <4 x float> %math.result477 to <4 x half>, !dbg !27
  %math.packed478 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %782) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce196 = fadd float %780, %local.reduce192, !dbg !28, !frisk.local_reduction !5
  %783 = extractelement <4 x float> %714, i64 0, !dbg !26
  %784 = tail call float @llvm.amdgcn.exp2.f32(float %783), !dbg !26
  %785 = insertelement <4 x float> poison, float %784, i64 0, !dbg !27
  %local.reduce197 = fadd float %784, %local.reduce193, !dbg !28, !frisk.local_reduction !5
  %786 = extractelement <4 x float> %714, i64 1, !dbg !26
  %787 = tail call float @llvm.amdgcn.exp2.f32(float %786), !dbg !26
  %788 = insertelement <4 x float> %785, float %787, i64 1, !dbg !27
  %local.reduce198 = fadd float %787, %local.reduce194, !dbg !28, !frisk.local_reduction !5
  %789 = extractelement <4 x float> %714, i64 2, !dbg !26
  %790 = tail call float @llvm.amdgcn.exp2.f32(float %789), !dbg !26
  %791 = insertelement <4 x float> %788, float %790, i64 2, !dbg !27
  %local.reduce199 = fadd float %790, %local.reduce195, !dbg !28, !frisk.local_reduction !5
  %792 = extractelement <4 x float> %714, i64 3, !dbg !26
  %793 = tail call float @llvm.amdgcn.exp2.f32(float %792), !dbg !26
  %794 = insertelement <4 x float> %791, float %793, i64 3, !dbg !27
  %math.result479 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %794) #8, !dbg !27, !frisk.register_lifetimes !5
  %795 = fptrunc <4 x float> %math.result479 to <4 x half>, !dbg !27
  %math.packed480 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %795) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce200 = fadd float %793, %local.reduce196, !dbg !28, !frisk.local_reduction !5
  %796 = extractelement <4 x float> %715, i64 0, !dbg !26
  %797 = tail call float @llvm.amdgcn.exp2.f32(float %796), !dbg !26
  %798 = insertelement <4 x float> poison, float %797, i64 0, !dbg !27
  %local.reduce201 = fadd float %797, %local.reduce197, !dbg !28, !frisk.local_reduction !5
  %799 = extractelement <4 x float> %715, i64 1, !dbg !26
  %800 = tail call float @llvm.amdgcn.exp2.f32(float %799), !dbg !26
  %801 = insertelement <4 x float> %798, float %800, i64 1, !dbg !27
  %local.reduce202 = fadd float %800, %local.reduce198, !dbg !28, !frisk.local_reduction !5
  %802 = extractelement <4 x float> %715, i64 2, !dbg !26
  %803 = tail call float @llvm.amdgcn.exp2.f32(float %802), !dbg !26
  %804 = insertelement <4 x float> %801, float %803, i64 2, !dbg !27
  %local.reduce203 = fadd float %803, %local.reduce199, !dbg !28, !frisk.local_reduction !5
  %805 = extractelement <4 x float> %715, i64 3, !dbg !26
  %806 = tail call float @llvm.amdgcn.exp2.f32(float %805), !dbg !26
  %807 = insertelement <4 x float> %804, float %806, i64 3, !dbg !27
  %math.result481 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %807) #8, !dbg !27, !frisk.register_lifetimes !5
  %808 = fptrunc <4 x float> %math.result481 to <4 x half>, !dbg !27
  %math.packed482 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %808) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce204 = fadd float %806, %local.reduce200, !dbg !28, !frisk.local_reduction !5
  %809 = extractelement <4 x float> %716, i64 0, !dbg !26
  %810 = tail call float @llvm.amdgcn.exp2.f32(float %809), !dbg !26
  %811 = insertelement <4 x float> poison, float %810, i64 0, !dbg !27
  %local.reduce205 = fadd float %810, %local.reduce201, !dbg !28, !frisk.local_reduction !5
  %812 = extractelement <4 x float> %716, i64 1, !dbg !26
  %813 = tail call float @llvm.amdgcn.exp2.f32(float %812), !dbg !26
  %814 = insertelement <4 x float> %811, float %813, i64 1, !dbg !27
  %local.reduce206 = fadd float %813, %local.reduce202, !dbg !28, !frisk.local_reduction !5
  %815 = extractelement <4 x float> %716, i64 2, !dbg !26
  %816 = tail call float @llvm.amdgcn.exp2.f32(float %815), !dbg !26
  %817 = insertelement <4 x float> %814, float %816, i64 2, !dbg !27
  %local.reduce207 = fadd float %816, %local.reduce203, !dbg !28, !frisk.local_reduction !5
  %818 = extractelement <4 x float> %716, i64 3, !dbg !26
  %819 = tail call float @llvm.amdgcn.exp2.f32(float %818), !dbg !26
  %820 = insertelement <4 x float> %817, float %819, i64 3, !dbg !27
  %math.result483 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %820) #8, !dbg !27, !frisk.register_lifetimes !5
  %821 = fptrunc <4 x float> %math.result483 to <4 x half>, !dbg !27
  %math.packed484 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %821) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce208 = fadd float %819, %local.reduce204, !dbg !28, !frisk.local_reduction !5
  %822 = extractelement <4 x float> %717, i64 0, !dbg !26
  %823 = tail call float @llvm.amdgcn.exp2.f32(float %822), !dbg !26
  %824 = insertelement <4 x float> poison, float %823, i64 0, !dbg !27
  %local.reduce209 = fadd float %823, %local.reduce205, !dbg !28, !frisk.local_reduction !5
  %825 = extractelement <4 x float> %717, i64 1, !dbg !26
  %826 = tail call float @llvm.amdgcn.exp2.f32(float %825), !dbg !26
  %827 = insertelement <4 x float> %824, float %826, i64 1, !dbg !27
  %local.reduce210 = fadd float %826, %local.reduce206, !dbg !28, !frisk.local_reduction !5
  %828 = extractelement <4 x float> %717, i64 2, !dbg !26
  %829 = tail call float @llvm.amdgcn.exp2.f32(float %828), !dbg !26
  %830 = insertelement <4 x float> %827, float %829, i64 2, !dbg !27
  %local.reduce211 = fadd float %829, %local.reduce207, !dbg !28, !frisk.local_reduction !5
  %831 = extractelement <4 x float> %717, i64 3, !dbg !26
  %832 = tail call float @llvm.amdgcn.exp2.f32(float %831), !dbg !26
  %833 = insertelement <4 x float> %830, float %832, i64 3, !dbg !27
  %math.result485 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %833) #8, !dbg !27, !frisk.register_lifetimes !5
  %834 = fptrunc <4 x float> %math.result485 to <4 x half>, !dbg !27
  %math.packed486 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %834) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce212 = fadd float %832, %local.reduce208, !dbg !28, !frisk.local_reduction !5
  %835 = extractelement <4 x float> %718, i64 0, !dbg !26
  %836 = tail call float @llvm.amdgcn.exp2.f32(float %835), !dbg !26
  %837 = insertelement <4 x float> poison, float %836, i64 0, !dbg !27
  %local.reduce213 = fadd float %836, %local.reduce209, !dbg !28, !frisk.local_reduction !5
  %838 = extractelement <4 x float> %718, i64 1, !dbg !26
  %839 = tail call float @llvm.amdgcn.exp2.f32(float %838), !dbg !26
  %840 = insertelement <4 x float> %837, float %839, i64 1, !dbg !27
  %local.reduce214 = fadd float %839, %local.reduce210, !dbg !28, !frisk.local_reduction !5
  %local.reduce217 = fadd float %local.reduce213, %local.reduce214, !dbg !28, !frisk.local_reduction !5
  %841 = extractelement <4 x float> %718, i64 2, !dbg !26
  %842 = tail call float @llvm.amdgcn.exp2.f32(float %841), !dbg !26
  %843 = insertelement <4 x float> %840, float %842, i64 2, !dbg !27
  %local.reduce215 = fadd float %842, %local.reduce211, !dbg !28, !frisk.local_reduction !5
  %844 = extractelement <4 x float> %718, i64 3, !dbg !26
  %845 = tail call float @llvm.amdgcn.exp2.f32(float %844), !dbg !26
  %846 = insertelement <4 x float> %843, float %845, i64 3, !dbg !27
  %math.result487 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %846) #8, !dbg !27, !frisk.register_lifetimes !5
  %847 = fptrunc <4 x float> %math.result487 to <4 x half>, !dbg !27
  %math.packed488 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %847) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce216 = fadd float %845, %local.reduce212, !dbg !28, !frisk.local_reduction !5
  %local.reduce218 = fadd float %local.reduce215, %local.reduce216, !dbg !28, !frisk.local_reduction !5
  %848 = fadd float %local.reduce217, %local.reduce218, !dbg !28, !frisk.local_reduction !5
  %849 = fadd float %735, %848, !dbg !28
  %850 = insertelement <1 x float> poison, float %849, i64 0, !dbg !11
  %851 = insertvalue [2 x <1 x float>] poison, <1 x float> %850, 0, !dbg !11
  %852 = extractelement <4 x float> %720, i64 0, !dbg !26
  %853 = tail call float @llvm.amdgcn.exp2.f32(float %852), !dbg !26
  %854 = insertelement <4 x float> poison, float %853, i64 0, !dbg !27
  %855 = extractelement <4 x float> %720, i64 1, !dbg !26
  %856 = tail call float @llvm.amdgcn.exp2.f32(float %855), !dbg !26
  %857 = insertelement <4 x float> %854, float %856, i64 1, !dbg !27
  %858 = extractelement <4 x float> %720, i64 2, !dbg !26
  %859 = tail call float @llvm.amdgcn.exp2.f32(float %858), !dbg !26
  %860 = insertelement <4 x float> %857, float %859, i64 2, !dbg !27
  %861 = extractelement <4 x float> %720, i64 3, !dbg !26
  %862 = tail call float @llvm.amdgcn.exp2.f32(float %861), !dbg !26
  %863 = insertelement <4 x float> %860, float %862, i64 3, !dbg !27
  %math.result489 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %863) #8, !dbg !27, !frisk.register_lifetimes !5
  %864 = fptrunc <4 x float> %math.result489 to <4 x half>, !dbg !27
  %math.packed490 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %864) #8, !dbg !26, !frisk.register_lifetimes !5
  %865 = extractelement <4 x float> %721, i64 0, !dbg !26
  %866 = tail call float @llvm.amdgcn.exp2.f32(float %865), !dbg !26
  %867 = insertelement <4 x float> poison, float %866, i64 0, !dbg !27
  %local.reduce = fadd float %853, %866, !dbg !28, !frisk.local_reduction !5
  %868 = extractelement <4 x float> %721, i64 1, !dbg !26
  %869 = tail call float @llvm.amdgcn.exp2.f32(float %868), !dbg !26
  %870 = insertelement <4 x float> %867, float %869, i64 1, !dbg !27
  %local.reduce159 = fadd float %856, %869, !dbg !28, !frisk.local_reduction !5
  %871 = extractelement <4 x float> %721, i64 2, !dbg !26
  %872 = tail call float @llvm.amdgcn.exp2.f32(float %871), !dbg !26
  %873 = insertelement <4 x float> %870, float %872, i64 2, !dbg !27
  %local.reduce160 = fadd float %859, %872, !dbg !28, !frisk.local_reduction !5
  %874 = extractelement <4 x float> %721, i64 3, !dbg !26
  %875 = tail call float @llvm.amdgcn.exp2.f32(float %874), !dbg !26
  %876 = insertelement <4 x float> %873, float %875, i64 3, !dbg !27
  %math.result491 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %876) #8, !dbg !27, !frisk.register_lifetimes !5
  %877 = fptrunc <4 x float> %math.result491 to <4 x half>, !dbg !27
  %math.packed492 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %877) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce161 = fadd float %862, %875, !dbg !28, !frisk.local_reduction !5
  %878 = extractelement <4 x float> %722, i64 0, !dbg !26
  %879 = tail call float @llvm.amdgcn.exp2.f32(float %878), !dbg !26
  %880 = insertelement <4 x float> poison, float %879, i64 0, !dbg !27
  %local.reduce162 = fadd float %879, %local.reduce, !dbg !28, !frisk.local_reduction !5
  %881 = extractelement <4 x float> %722, i64 1, !dbg !26
  %882 = tail call float @llvm.amdgcn.exp2.f32(float %881), !dbg !26
  %883 = insertelement <4 x float> %880, float %882, i64 1, !dbg !27
  %local.reduce163 = fadd float %882, %local.reduce159, !dbg !28, !frisk.local_reduction !5
  %884 = extractelement <4 x float> %722, i64 2, !dbg !26
  %885 = tail call float @llvm.amdgcn.exp2.f32(float %884), !dbg !26
  %886 = insertelement <4 x float> %883, float %885, i64 2, !dbg !27
  %local.reduce164 = fadd float %885, %local.reduce160, !dbg !28, !frisk.local_reduction !5
  %887 = extractelement <4 x float> %722, i64 3, !dbg !26
  %888 = tail call float @llvm.amdgcn.exp2.f32(float %887), !dbg !26
  %889 = insertelement <4 x float> %886, float %888, i64 3, !dbg !27
  %math.result493 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %889) #8, !dbg !27, !frisk.register_lifetimes !5
  %890 = fptrunc <4 x float> %math.result493 to <4 x half>, !dbg !27
  %math.packed494 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %890) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce165 = fadd float %888, %local.reduce161, !dbg !28, !frisk.local_reduction !5
  %891 = extractelement <4 x float> %723, i64 0, !dbg !26
  %892 = tail call float @llvm.amdgcn.exp2.f32(float %891), !dbg !26
  %893 = insertelement <4 x float> poison, float %892, i64 0, !dbg !27
  %local.reduce166 = fadd float %892, %local.reduce162, !dbg !28, !frisk.local_reduction !5
  %894 = extractelement <4 x float> %723, i64 1, !dbg !26
  %895 = tail call float @llvm.amdgcn.exp2.f32(float %894), !dbg !26
  %896 = insertelement <4 x float> %893, float %895, i64 1, !dbg !27
  %local.reduce167 = fadd float %895, %local.reduce163, !dbg !28, !frisk.local_reduction !5
  %897 = extractelement <4 x float> %723, i64 2, !dbg !26
  %898 = tail call float @llvm.amdgcn.exp2.f32(float %897), !dbg !26
  %899 = insertelement <4 x float> %896, float %898, i64 2, !dbg !27
  %local.reduce168 = fadd float %898, %local.reduce164, !dbg !28, !frisk.local_reduction !5
  %900 = extractelement <4 x float> %723, i64 3, !dbg !26
  %901 = tail call float @llvm.amdgcn.exp2.f32(float %900), !dbg !26
  %902 = insertelement <4 x float> %899, float %901, i64 3, !dbg !27
  %math.result495 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %902) #8, !dbg !27, !frisk.register_lifetimes !5
  %903 = fptrunc <4 x float> %math.result495 to <4 x half>, !dbg !27
  %math.packed496 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %903) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce169 = fadd float %901, %local.reduce165, !dbg !28, !frisk.local_reduction !5
  %904 = extractelement <4 x float> %724, i64 0, !dbg !26
  %905 = tail call float @llvm.amdgcn.exp2.f32(float %904), !dbg !26
  %906 = insertelement <4 x float> poison, float %905, i64 0, !dbg !27
  %local.reduce170 = fadd float %905, %local.reduce166, !dbg !28, !frisk.local_reduction !5
  %907 = extractelement <4 x float> %724, i64 1, !dbg !26
  %908 = tail call float @llvm.amdgcn.exp2.f32(float %907), !dbg !26
  %909 = insertelement <4 x float> %906, float %908, i64 1, !dbg !27
  %local.reduce171 = fadd float %908, %local.reduce167, !dbg !28, !frisk.local_reduction !5
  %910 = extractelement <4 x float> %724, i64 2, !dbg !26
  %911 = tail call float @llvm.amdgcn.exp2.f32(float %910), !dbg !26
  %912 = insertelement <4 x float> %909, float %911, i64 2, !dbg !27
  %local.reduce172 = fadd float %911, %local.reduce168, !dbg !28, !frisk.local_reduction !5
  %913 = extractelement <4 x float> %724, i64 3, !dbg !26
  %914 = tail call float @llvm.amdgcn.exp2.f32(float %913), !dbg !26
  %915 = insertelement <4 x float> %912, float %914, i64 3, !dbg !27
  %math.result497 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %915) #8, !dbg !27, !frisk.register_lifetimes !5
  %916 = fptrunc <4 x float> %math.result497 to <4 x half>, !dbg !27
  %math.packed498 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %916) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce173 = fadd float %914, %local.reduce169, !dbg !28, !frisk.local_reduction !5
  %917 = extractelement <4 x float> %725, i64 0, !dbg !26
  %918 = tail call float @llvm.amdgcn.exp2.f32(float %917), !dbg !26
  %919 = insertelement <4 x float> poison, float %918, i64 0, !dbg !27
  %local.reduce174 = fadd float %918, %local.reduce170, !dbg !28, !frisk.local_reduction !5
  %920 = extractelement <4 x float> %725, i64 1, !dbg !26
  %921 = tail call float @llvm.amdgcn.exp2.f32(float %920), !dbg !26
  %922 = insertelement <4 x float> %919, float %921, i64 1, !dbg !27
  %local.reduce175 = fadd float %921, %local.reduce171, !dbg !28, !frisk.local_reduction !5
  %923 = extractelement <4 x float> %725, i64 2, !dbg !26
  %924 = tail call float @llvm.amdgcn.exp2.f32(float %923), !dbg !26
  %925 = insertelement <4 x float> %922, float %924, i64 2, !dbg !27
  %local.reduce176 = fadd float %924, %local.reduce172, !dbg !28, !frisk.local_reduction !5
  %926 = extractelement <4 x float> %725, i64 3, !dbg !26
  %927 = tail call float @llvm.amdgcn.exp2.f32(float %926), !dbg !26
  %928 = insertelement <4 x float> %925, float %927, i64 3, !dbg !27
  %math.result499 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %928) #8, !dbg !27, !frisk.register_lifetimes !5
  %929 = fptrunc <4 x float> %math.result499 to <4 x half>, !dbg !27
  %math.packed500 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %929) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce177 = fadd float %927, %local.reduce173, !dbg !28, !frisk.local_reduction !5
  %930 = extractelement <4 x float> %726, i64 0, !dbg !26
  %931 = tail call float @llvm.amdgcn.exp2.f32(float %930), !dbg !26
  %932 = insertelement <4 x float> poison, float %931, i64 0, !dbg !27
  %local.reduce178 = fadd float %931, %local.reduce174, !dbg !28, !frisk.local_reduction !5
  %933 = extractelement <4 x float> %726, i64 1, !dbg !26
  %934 = tail call float @llvm.amdgcn.exp2.f32(float %933), !dbg !26
  %935 = insertelement <4 x float> %932, float %934, i64 1, !dbg !27
  %local.reduce179 = fadd float %934, %local.reduce175, !dbg !28, !frisk.local_reduction !5
  %936 = extractelement <4 x float> %726, i64 2, !dbg !26
  %937 = tail call float @llvm.amdgcn.exp2.f32(float %936), !dbg !26
  %938 = insertelement <4 x float> %935, float %937, i64 2, !dbg !27
  %local.reduce180 = fadd float %937, %local.reduce176, !dbg !28, !frisk.local_reduction !5
  %939 = extractelement <4 x float> %726, i64 3, !dbg !26
  %940 = tail call float @llvm.amdgcn.exp2.f32(float %939), !dbg !26
  %941 = insertelement <4 x float> %938, float %940, i64 3, !dbg !27
  %math.result501 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %941) #8, !dbg !27, !frisk.register_lifetimes !5
  %942 = fptrunc <4 x float> %math.result501 to <4 x half>, !dbg !27
  %math.packed502 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %942) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce181 = fadd float %940, %local.reduce177, !dbg !28, !frisk.local_reduction !5
  %943 = extractelement <4 x float> %727, i64 0, !dbg !26
  %944 = tail call float @llvm.amdgcn.exp2.f32(float %943), !dbg !26
  %945 = insertelement <4 x float> poison, float %944, i64 0, !dbg !27
  %local.reduce182 = fadd float %944, %local.reduce178, !dbg !28, !frisk.local_reduction !5
  %946 = extractelement <4 x float> %727, i64 1, !dbg !26
  %947 = tail call float @llvm.amdgcn.exp2.f32(float %946), !dbg !26
  %948 = insertelement <4 x float> %945, float %947, i64 1, !dbg !27
  %local.reduce183 = fadd float %947, %local.reduce179, !dbg !28, !frisk.local_reduction !5
  %local.reduce186 = fadd float %local.reduce182, %local.reduce183, !dbg !28, !frisk.local_reduction !5
  %949 = extractelement <4 x float> %727, i64 2, !dbg !26
  %950 = tail call float @llvm.amdgcn.exp2.f32(float %949), !dbg !26
  %951 = insertelement <4 x float> %948, float %950, i64 2, !dbg !27
  %local.reduce184 = fadd float %950, %local.reduce180, !dbg !28, !frisk.local_reduction !5
  %952 = extractelement <4 x float> %727, i64 3, !dbg !26
  %953 = tail call float @llvm.amdgcn.exp2.f32(float %952), !dbg !26
  %954 = insertelement <4 x float> %951, float %953, i64 3, !dbg !27
  %math.result503 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %954) #8, !dbg !27, !frisk.register_lifetimes !5
  %955 = fptrunc <4 x float> %math.result503 to <4 x half>, !dbg !27
  %math.packed504 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %955) #8, !dbg !28, !frisk.register_lifetimes !5
  %local.reduce185 = fadd float %953, %local.reduce181, !dbg !28, !frisk.local_reduction !5
  %local.reduce187 = fadd float %local.reduce184, %local.reduce185, !dbg !28, !frisk.local_reduction !5
  %956 = fadd float %local.reduce186, %local.reduce187, !dbg !28, !frisk.local_reduction !5
  %957 = fadd float %743, %956, !dbg !28
  %958 = insertelement <1 x float> poison, float %957, i64 0, !dbg !11
  %959 = insertvalue [2 x <1 x float>] %851, <1 x float> %958, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread1030 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !29, !range !9, !frisk.register_lifetimes !5
  %address.local1031 = shl nuw nsw i32 %address.thread1030, 3, !dbg !29
  %address.local1032 = and i32 %address.local1031, 120, !dbg !29
  %address.local1033 = shl nuw nsw i32 %address.thread1030, 5, !dbg !29
  %address.local1034 = and i32 %address.local1033, 1536, !dbg !29
  %address.local1035 = or disjoint i32 %address.local1032, %address.local1034, !dbg !29
  %address.local1036 = zext nneg i32 %address.local1035 to i64, !dbg !29
  %address.local1037 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1036, !dbg !29
  %960 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #10, !dbg !29
  %961 = bitcast <4 x i32> %960 to <8 x half>, !dbg !29
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
  %962 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #10, !dbg !29
  %963 = bitcast <4 x i32> %962 to <8 x half>, !dbg !29
  %address.local1054 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 512, !dbg !29
  %964 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #10, !dbg !29
  %965 = bitcast <4 x i32> %964 to <8 x half>, !dbg !29
  %address.local1058 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 768, !dbg !29
  %966 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #10, !dbg !29
  %967 = bitcast <4 x i32> %966 to <8 x half>, !dbg !29
  %address.local1060 = getelementptr i8, ptr addrspace(3) %address.local1037, i64 4096, !dbg !29
  %968 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #10, !dbg !29
  %969 = bitcast <4 x i32> %968 to <8 x half>, !dbg !29
  %address.local1064 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 4352, !dbg !29
  %970 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #10, !dbg !29
  %971 = bitcast <4 x i32> %970 to <8 x half>, !dbg !29
  %address.local1068 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 4608, !dbg !29
  %972 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #10, !dbg !29
  %973 = bitcast <4 x i32> %972 to <8 x half>, !dbg !29
  %address.local1072 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 4864, !dbg !29
  %974 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #10, !dbg !29
  %975 = bitcast <4 x i32> %974 to <8 x half>, !dbg !29
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local1073 = shl nuw nsw i64 %address.local1038, 3, !dbg !8
  %address.local1074 = and i64 %address.local1073, 1536, !dbg !8
  %address.local1075 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1074, !dbg !8
  %address.local1076 = ptrtoint ptr addrspace(3) %address.local1075 to i64, !dbg !8
  %address.local1077 = trunc i64 %address.local1076 to i32, !dbg !8
  %976 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1077), !dbg !17
  %address.local1078 = shl nuw nsw i64 %address.local1038, 5, !dbg !17
  %address.local1079 = and i64 %address.local1078, 8064, !dbg !17
  %address.local1081 = and i64 %address.local1073, 24, !dbg !17
  %address.local1082 = or disjoint i64 %address.local1079, %address.local1081, !dbg !17
  %977 = or disjoint i64 %.idx5760, %address.local1082, !dbg !17
  %978 = trunc i64 %977 to i32, !dbg !17
  %.tr68 = shl i32 %978, 1, !dbg !17
  %dma.buffer.offset1724 = or i32 %.tr68, 64, !dbg !17
  %dma.buffer.lds1726 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %976), !dbg !17
  %979 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1724, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %address.local1084 = or disjoint i64 %address.local1038, 256, !dbg !8
  %address.local1085 = shl nuw nsw i64 %address.local1084, 3, !dbg !8
  %address.local1086 = and i64 %address.local1085, 3584, !dbg !8
  %address.local1087 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1086, !dbg !8
  %address.local1088 = ptrtoint ptr addrspace(3) %address.local1087 to i64, !dbg !8
  %address.local1089 = trunc i64 %address.local1088 to i32, !dbg !8
  %980 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1089), !dbg !17
  %address.local1090 = shl nuw nsw i64 %address.local1084, 5, !dbg !17
  %address.local1091 = and i64 %address.local1090, 16256, !dbg !17
  %address.local1094 = or disjoint i64 %address.local1091, %address.local1081, !dbg !17
  %981 = or disjoint i64 %.idx5760, %address.local1094, !dbg !17
  %982 = trunc i64 %981 to i32, !dbg !17
  %.tr72 = shl i32 %982, 1, !dbg !17
  %dma.buffer.offset1721 = or i32 %.tr72, 64, !dbg !17
  %dma.buffer.lds1723 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %980), !dbg !17
  %983 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1721, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %984 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %985 = extractvalue [2 x <8 x float>] %143, 0, !dbg !24
  %986 = shufflevector <8 x float> %985, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %987 = fmul <4 x float> %986, %732, !dbg !24
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %984, <4 x float> %987) #8, !dbg !29
  %989 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %989, <4 x float> %988) #8, !dbg !29
  %991 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %991, <4 x float> %990) #8, !dbg !29
  %993 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %993, <4 x float> %992) #8, !dbg !29
  %995 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %995, <4 x float> %994) #8, !dbg !29
  %997 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %997, <4 x float> %996) #8, !dbg !29
  %999 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %999, <4 x float> %998) #8, !dbg !29
  %1001 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1001, <4 x float> %1000) #8, !dbg !29
  %1003 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1004 = shufflevector <8 x float> %985, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1005 = fmul <4 x float> %1004, %732, !dbg !24
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1003, <4 x float> %1005) #8, !dbg !29
  %1007 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1007, <4 x float> %1006) #8, !dbg !29
  %1009 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1009, <4 x float> %1008) #8, !dbg !29
  %1011 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1011, <4 x float> %1010) #8, !dbg !29
  %1013 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1013, <4 x float> %1012) #8, !dbg !29
  %1015 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1016 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1015, <4 x float> %1014) #8, !dbg !29
  %1017 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1018 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1017, <4 x float> %1016) #8, !dbg !29
  %1019 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1019, <4 x float> %1018) #8, !dbg !29
  %1021 = extractvalue [2 x <8 x float>] %143, 1, !dbg !24
  %1022 = shufflevector <8 x float> %1021, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1023 = fmul <4 x float> %1022, %740, !dbg !24
  %1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %984, <4 x float> %1023) #8, !dbg !29
  %1025 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %989, <4 x float> %1024) #8, !dbg !29
  %1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %991, <4 x float> %1025) #8, !dbg !29
  %1027 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %993, <4 x float> %1026) #8, !dbg !29
  %1028 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %995, <4 x float> %1027) #8, !dbg !29
  %1029 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %997, <4 x float> %1028) #8, !dbg !29
  %1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %999, <4 x float> %1029) #8, !dbg !29
  %1031 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1001, <4 x float> %1030) #8, !dbg !29
  %1032 = shufflevector <8 x float> %1021, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1033 = fmul <4 x float> %1032, %740, !dbg !24
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
  %dma.buffer.offset1718 = or i32 %.tr68, 128, !dbg !17
  %1058 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1718, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %dma.buffer.offset1715 = or i32 %.tr72, 128, !dbg !17
  %1059 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1715, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %1060 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1061 = extractvalue [2 x <8 x float>] %140, 0, !dbg !24
  %1062 = shufflevector <8 x float> %1061, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1063 = fmul <4 x float> %1062, %732, !dbg !24
  %1064 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1060, <4 x float> %1063) #8, !dbg !29
  %1065 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1066 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1065, <4 x float> %1064) #8, !dbg !29
  %1067 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1068 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1067, <4 x float> %1066) #8, !dbg !29
  %1069 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1070 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1069, <4 x float> %1068) #8, !dbg !29
  %1071 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1072 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1071, <4 x float> %1070) #8, !dbg !29
  %1073 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1074 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1073, <4 x float> %1072) #8, !dbg !29
  %1075 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1075, <4 x float> %1074) #8, !dbg !29
  %1077 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1077, <4 x float> %1076) #8, !dbg !29
  %1079 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1080 = shufflevector <8 x float> %1061, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1081 = fmul <4 x float> %1080, %732, !dbg !24
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1079, <4 x float> %1081) #8, !dbg !29
  %1083 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1083, <4 x float> %1082) #8, !dbg !29
  %1085 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1085, <4 x float> %1084) #8, !dbg !29
  %1087 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1087, <4 x float> %1086) #8, !dbg !29
  %1089 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1089, <4 x float> %1088) #8, !dbg !29
  %1091 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1091, <4 x float> %1090) #8, !dbg !29
  %1093 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1093, <4 x float> %1092) #8, !dbg !29
  %1095 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1095, <4 x float> %1094) #8, !dbg !29
  %1097 = extractvalue [2 x <8 x float>] %140, 1, !dbg !24
  %1098 = shufflevector <8 x float> %1097, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1099 = fmul <4 x float> %1098, %740, !dbg !24
  %1100 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1060, <4 x float> %1099) #8, !dbg !29
  %1101 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1065, <4 x float> %1100) #8, !dbg !29
  %1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1067, <4 x float> %1101) #8, !dbg !29
  %1103 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1069, <4 x float> %1102) #8, !dbg !29
  %1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1071, <4 x float> %1103) #8, !dbg !29
  %1105 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1073, <4 x float> %1104) #8, !dbg !29
  %1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1075, <4 x float> %1105) #8, !dbg !29
  %1107 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1077, <4 x float> %1106) #8, !dbg !29
  %1108 = shufflevector <8 x float> %1097, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1109 = fmul <4 x float> %1108, %740, !dbg !24
  %1110 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1079, <4 x float> %1109) #8, !dbg !29
  %1111 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1083, <4 x float> %1110) #8, !dbg !29
  %1112 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1085, <4 x float> %1111) #8, !dbg !29
  %1113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1087, <4 x float> %1112) #8, !dbg !29
  %1114 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1089, <4 x float> %1113) #8, !dbg !29
  %1115 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1091, <4 x float> %1114) #8, !dbg !29
  %1116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1093, <4 x float> %1115) #8, !dbg !29
  %1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1095, <4 x float> %1116) #8, !dbg !29
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1118 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #10, !dbg !29
  %1119 = bitcast <4 x i32> %1118 to <8 x half>, !dbg !29
  %1120 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #10, !dbg !29
  %1121 = bitcast <4 x i32> %1120 to <8 x half>, !dbg !29
  %1122 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #10, !dbg !29
  %1123 = bitcast <4 x i32> %1122 to <8 x half>, !dbg !29
  %1124 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #10, !dbg !29
  %1125 = bitcast <4 x i32> %1124 to <8 x half>, !dbg !29
  %1126 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #10, !dbg !29
  %1127 = bitcast <4 x i32> %1126 to <8 x half>, !dbg !29
  %1128 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #10, !dbg !29
  %1129 = bitcast <4 x i32> %1128 to <8 x half>, !dbg !29
  %1130 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #10, !dbg !29
  %1131 = bitcast <4 x i32> %1130 to <8 x half>, !dbg !29
  %1132 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #10, !dbg !29
  %1133 = bitcast <4 x i32> %1132 to <8 x half>, !dbg !29
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1134 = or disjoint i64 %.idx5760, 96, !dbg !17
  %1135 = or disjoint i64 %address.local1082, %1134, !dbg !17
  %.tr84 = trunc i64 %1135 to i32, !dbg !17
  %dma.buffer.offset1712 = shl i32 %.tr84, 1, !dbg !17
  %1136 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1712, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %1137 = or disjoint i64 %address.local1094, %1134, !dbg !17
  %.tr88 = trunc i64 %1137 to i32, !dbg !17
  %dma.buffer.offset = shl i32 %.tr88, 1, !dbg !17
  %1138 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1731) #9, !dbg !17
  %1139 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1140 = extractvalue [2 x <8 x float>] %139, 0, !dbg !24
  %1141 = shufflevector <8 x float> %1140, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1142 = fmul <4 x float> %1141, %732, !dbg !24
  %1143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1139, <4 x float> %1142) #8, !dbg !29
  %1144 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1145 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1144, <4 x float> %1143) #8, !dbg !29
  %1146 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1146, <4 x float> %1145) #8, !dbg !29
  %1148 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1148, <4 x float> %1147) #8, !dbg !29
  %1150 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1150, <4 x float> %1149) #8, !dbg !29
  %1152 = shufflevector <8 x half> %1129, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1152, <4 x float> %1151) #8, !dbg !29
  %1154 = shufflevector <8 x half> %1131, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1154, <4 x float> %1153) #8, !dbg !29
  %1156 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1156, <4 x float> %1155) #8, !dbg !29
  %1158 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1159 = shufflevector <8 x float> %1140, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1160 = fmul <4 x float> %1159, %732, !dbg !24
  %1161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1158, <4 x float> %1160) #8, !dbg !29
  %1162 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1162, <4 x float> %1161) #8, !dbg !29
  %1164 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1164, <4 x float> %1163) #8, !dbg !29
  %1166 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1166, <4 x float> %1165) #8, !dbg !29
  %1168 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1168, <4 x float> %1167) #8, !dbg !29
  %1170 = shufflevector <8 x half> %1129, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1170, <4 x float> %1169) #8, !dbg !29
  %1172 = shufflevector <8 x half> %1131, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1172, <4 x float> %1171) #8, !dbg !29
  %1174 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1174, <4 x float> %1173) #8, !dbg !29
  %1176 = extractvalue [2 x <8 x float>] %139, 1, !dbg !24
  %1177 = shufflevector <8 x float> %1176, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1178 = fmul <4 x float> %1177, %740, !dbg !24
  %1179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1139, <4 x float> %1178) #8, !dbg !29
  %1180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1144, <4 x float> %1179) #8, !dbg !29
  %1181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1146, <4 x float> %1180) #8, !dbg !29
  %1182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1148, <4 x float> %1181) #8, !dbg !29
  %1183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1150, <4 x float> %1182) #8, !dbg !29
  %1184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1152, <4 x float> %1183) #8, !dbg !29
  %1185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1154, <4 x float> %1184) #8, !dbg !29
  %1186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1156, <4 x float> %1185) #8, !dbg !29
  %1187 = shufflevector <8 x float> %1176, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1188 = fmul <4 x float> %1187, %740, !dbg !24
  %1189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1158, <4 x float> %1188) #8, !dbg !29
  %1190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1162, <4 x float> %1189) #8, !dbg !29
  %1191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1164, <4 x float> %1190) #8, !dbg !29
  %1192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1166, <4 x float> %1191) #8, !dbg !29
  %1193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1168, <4 x float> %1192) #8, !dbg !29
  %1194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1170, <4 x float> %1193) #8, !dbg !29
  %1195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1172, <4 x float> %1194) #8, !dbg !29
  %1196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1174, <4 x float> %1195) #8, !dbg !29
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1197 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #10, !dbg !29
  %1198 = bitcast <4 x i32> %1197 to <8 x half>, !dbg !29
  %1199 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #10, !dbg !29
  %1200 = bitcast <4 x i32> %1199 to <8 x half>, !dbg !29
  %1201 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #10, !dbg !29
  %1202 = bitcast <4 x i32> %1201 to <8 x half>, !dbg !29
  %1203 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #10, !dbg !29
  %1204 = bitcast <4 x i32> %1203 to <8 x half>, !dbg !29
  %1205 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #10, !dbg !29
  %1206 = bitcast <4 x i32> %1205 to <8 x half>, !dbg !29
  %1207 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #10, !dbg !29
  %1208 = bitcast <4 x i32> %1207 to <8 x half>, !dbg !29
  %1209 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #10, !dbg !29
  %1210 = bitcast <4 x i32> %1209 to <8 x half>, !dbg !29
  %1211 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #10, !dbg !29
  %1212 = bitcast <4 x i32> %1211 to <8 x half>, !dbg !29
  %1213 = shufflevector <8 x half> %1198, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1214 = extractvalue [2 x <8 x float>] %138, 0, !dbg !24
  %1215 = shufflevector <8 x float> %1214, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1216 = fmul <4 x float> %1215, %732, !dbg !24
  %1217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1213, <4 x float> %1216) #8, !dbg !29
  %1218 = shufflevector <8 x half> %1200, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1218, <4 x float> %1217) #8, !dbg !29
  %1220 = shufflevector <8 x half> %1202, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1220, <4 x float> %1219) #8, !dbg !29
  %1222 = shufflevector <8 x half> %1204, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1222, <4 x float> %1221) #8, !dbg !29
  %1224 = shufflevector <8 x half> %1206, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1224, <4 x float> %1223) #8, !dbg !29
  %1226 = shufflevector <8 x half> %1208, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1226, <4 x float> %1225) #8, !dbg !29
  %1228 = shufflevector <8 x half> %1210, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1228, <4 x float> %1227) #8, !dbg !29
  %1230 = shufflevector <8 x half> %1212, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1230, <4 x float> %1229) #8, !dbg !29
  %1232 = shufflevector <8 x half> %1198, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1233 = shufflevector <8 x float> %1214, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1234 = fmul <4 x float> %1233, %732, !dbg !24
  %1235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1232, <4 x float> %1234) #8, !dbg !29
  %1236 = shufflevector <8 x half> %1200, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1236, <4 x float> %1235) #8, !dbg !29
  %1238 = shufflevector <8 x half> %1202, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1238, <4 x float> %1237) #8, !dbg !29
  %1240 = shufflevector <8 x half> %1204, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1240, <4 x float> %1239) #8, !dbg !29
  %1242 = shufflevector <8 x half> %1206, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1242, <4 x float> %1241) #8, !dbg !29
  %1244 = shufflevector <8 x half> %1208, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1244, <4 x float> %1243) #8, !dbg !29
  %1246 = shufflevector <8 x half> %1210, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1246, <4 x float> %1245) #8, !dbg !29
  %1248 = shufflevector <8 x half> %1212, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1248, <4 x float> %1247) #8, !dbg !29
  %1250 = extractvalue [2 x <8 x float>] %138, 1, !dbg !24
  %1251 = shufflevector <8 x float> %1250, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %1252 = fmul <4 x float> %1251, %740, !dbg !24
  %1253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1213, <4 x float> %1252) #8, !dbg !29
  %1254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1218, <4 x float> %1253) #8, !dbg !29
  %1255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1220, <4 x float> %1254) #8, !dbg !29
  %1256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1222, <4 x float> %1255) #8, !dbg !29
  %1257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1224, <4 x float> %1256) #8, !dbg !29
  %1258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1226, <4 x float> %1257) #8, !dbg !29
  %1259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1228, <4 x float> %1258) #8, !dbg !29
  %1260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1230, <4 x float> %1259) #8, !dbg !29
  %1261 = shufflevector <8 x float> %1250, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %1262 = fmul <4 x float> %1261, %740, !dbg !24
  %1263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1232, <4 x float> %1262) #8, !dbg !29
  %1264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1236, <4 x float> %1263) #8, !dbg !29
  %1265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1238, <4 x float> %1264) #8, !dbg !29
  %1266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1240, <4 x float> %1265) #8, !dbg !29
  %1267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1242, <4 x float> %1266) #8, !dbg !29
  %1268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1244, <4 x float> %1267) #8, !dbg !29
  %1269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1246, <4 x float> %1268) #8, !dbg !29
  %1270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1248, <4 x float> %1269) #8, !dbg !29
  %1271 = shufflevector <4 x float> %1002, <4 x float> %1020, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1272 = insertvalue [2 x <8 x float>] poison, <8 x float> %1271, 0, !dbg !11
  %1273 = shufflevector <4 x float> %1031, <4 x float> %1041, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1274 = insertvalue [2 x <8 x float>] %1272, <8 x float> %1273, 1, !dbg !11
  %1275 = insertvalue [2 x <1 x float>] poison, <1 x float> %696, 0, !dbg !11
  %1276 = insertvalue [2 x <1 x float>] %1275, <1 x float> %709, 1, !dbg !11
  %1277 = shufflevector <4 x float> %1078, <4 x float> %1096, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1278 = insertvalue [2 x <8 x float>] poison, <8 x float> %1277, 0, !dbg !11
  %1279 = shufflevector <4 x float> %1107, <4 x float> %1117, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1280 = insertvalue [2 x <8 x float>] %1278, <8 x float> %1279, 1, !dbg !11
  %1281 = shufflevector <4 x float> %1157, <4 x float> %1175, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1282 = insertvalue [2 x <8 x float>] poison, <8 x float> %1281, 0, !dbg !11
  %1283 = shufflevector <4 x float> %1186, <4 x float> %1196, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1284 = insertvalue [2 x <8 x float>] %1282, <8 x float> %1283, 1, !dbg !11
  %1285 = shufflevector <4 x float> %1231, <4 x float> %1249, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1286 = insertvalue [2 x <8 x float>] poison, <8 x float> %1285, 0, !dbg !11
  %1287 = shufflevector <4 x float> %1260, <4 x float> %1270, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1288 = insertvalue [2 x <8 x float>] %1286, <8 x float> %1287, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1289 = add nuw nsw i64 %144, 1, !dbg !11
  %1290 = icmp slt i64 %1289, %103, !dbg !11
  br i1 %1290, label %137, label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %591, %.._crit_edge_crit_edge
  %.pre-phi112 = phi i32 [ %.pre111, %.._crit_edge_crit_edge ], [ %686, %591 ], !dbg !11
  %.pre-phi110 = phi i32 [ %.pre109, %.._crit_edge_crit_edge ], [ %678, %591 ], !dbg !11
  %.pre-phi108 = phi i32 [ %.pre107, %.._crit_edge_crit_edge ], [ %677, %591 ], !dbg !11
  %lane.sum.lane287.pre-phi = phi i32 [ %.pre105, %.._crit_edge_crit_edge ], [ %675, %591 ], !dbg !11
  %.lcssa94 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1274, %591 ], !dbg !11
  %.lcssa93 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %959, %591 ], !dbg !11
  %.lcssa92 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1280, %591 ], !dbg !11
  %.lcssa91 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1284, %591 ], !dbg !11
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1288, %591 ], !dbg !11
  %lane.sum.field284 = extractvalue [2 x <1 x float>] %.lcssa93, 1, !dbg !11
  %lane.sum.local285 = extractelement <1 x float> %lane.sum.field284, i64 0, !dbg !11
  %1291 = shl nuw nsw i32 %lane.sum.lane287.pre-phi, 2, !dbg !11
  %1292 = xor i32 %1291, 64, !dbg !11
  %1293 = bitcast float %lane.sum.local285 to i32, !dbg !11
  %lane.sum.shuffle288 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1292, i32 %1293), !dbg !11
  %1294 = bitcast i32 %lane.sum.shuffle288 to float, !dbg !11
  %lane.sum.final289 = fadd float %lane.sum.local285, %1294, !dbg !11
  %1295 = xor i32 %1291, 128, !dbg !11
  %1296 = bitcast float %lane.sum.final289 to i32, !dbg !11
  %lane.sum.shuffle290 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1295, i32 %1296), !dbg !11
  %1297 = bitcast i32 %lane.sum.shuffle290 to float, !dbg !11
  %lane.sum.final291 = fadd float %lane.sum.final289, %1297, !dbg !11
  %1298 = insertelement <1 x float> poison, float %lane.sum.final291, i64 0, !dbg !11
  %lane.sum.field = extractvalue [2 x <1 x float>] %.lcssa93, 0, !dbg !11
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !11
  %1299 = bitcast float %lane.sum.local to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1292, i32 %1299), !dbg !11
  %1300 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %lane.sum.local, %1300, !dbg !11
  %1301 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle282 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1295, i32 %1301), !dbg !11
  %1302 = bitcast i32 %lane.sum.shuffle282 to float, !dbg !11
  %lane.sum.final283 = fadd float %lane.sum.final, %1302, !dbg !11
  %1303 = insertelement <1 x float> poison, float %lane.sum.final283, i64 0, !dbg !11
  %1304 = and i32 %29, 16, !dbg !11
  %.not = icmp eq i32 %1304, 0, !dbg !11
  %1305 = and i32 %29, 32, !dbg !11
  %.not36 = icmp eq i32 %1305, 0, !dbg !11
  %1306 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !11
  %1307 = extractelement <8 x float> %1306, i64 0, !dbg !11
  %1308 = extractelement <8 x float> %1306, i64 1, !dbg !11
  %1309 = extractelement <8 x float> %1306, i64 2, !dbg !11
  %1310 = extractelement <8 x float> %1306, i64 3, !dbg !11
  %1311 = select i1 %.not, float %1308, float %1307, !dbg !11
  %1312 = icmp samesign ult i32 %.pre-phi108, %.pre-phi110, !dbg !11
  %1313 = select i1 %1312, i32 %.pre-phi108, i32 %lane.sum.lane287.pre-phi, !dbg !11
  %1314 = shl nuw nsw i32 %1313, 2, !dbg !11
  %1315 = bitcast float %1311 to i32, !dbg !11
  %1316 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1315), !dbg !11
  %1317 = bitcast i32 %1316 to float, !dbg !11
  %1318 = select i1 %.not, float %1307, float %1317, !dbg !11
  %1319 = select i1 %.not, float %1317, float %1308, !dbg !11
  %1320 = select i1 %.not, float %1310, float %1309, !dbg !11
  %1321 = bitcast float %1320 to i32, !dbg !11
  %1322 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1321), !dbg !11
  %1323 = bitcast i32 %1322 to float, !dbg !11
  %1324 = select i1 %.not, float %1309, float %1323, !dbg !11
  %1325 = select i1 %.not, float %1323, float %1310, !dbg !11
  %1326 = select i1 %.not36, float %1324, float %1318, !dbg !11
  %1327 = icmp samesign ult i32 %.pre-phi112, %.pre-phi110, !dbg !11
  %1328 = select i1 %1327, i32 %.pre-phi112, i32 %lane.sum.lane287.pre-phi, !dbg !11
  %1329 = shl nuw nsw i32 %1328, 2, !dbg !11
  %1330 = bitcast float %1326 to i32, !dbg !11
  %1331 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1330), !dbg !11
  %1332 = bitcast i32 %1331 to float, !dbg !11
  %1333 = select i1 %.not36, float %1325, float %1319, !dbg !11
  %1334 = bitcast float %1333 to i32, !dbg !11
  %1335 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1334), !dbg !11
  %1336 = bitcast i32 %1335 to float, !dbg !11
  %1337 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1338 = shufflevector <4 x i1> %1337, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1339 = insertelement <4 x float> poison, float %1318, i64 0, !dbg !11
  %1340 = insertelement <4 x float> %1339, float %1319, i64 1, !dbg !11
  %1341 = insertelement <4 x float> %1340, float %1332, i64 2, !dbg !11
  %1342 = insertelement <4 x float> %1341, float %1336, i64 3, !dbg !11
  %1343 = insertelement <4 x float> poison, float %1332, i64 0, !dbg !11
  %1344 = insertelement <4 x float> %1343, float %1336, i64 1, !dbg !11
  %1345 = insertelement <4 x float> %1344, float %1324, i64 2, !dbg !11
  %1346 = insertelement <4 x float> %1345, float %1325, i64 3, !dbg !11
  %1347 = select <4 x i1> %1338, <4 x float> %1342, <4 x float> %1346, !dbg !11
  %1348 = extractvalue [2 x <8 x float>] %.lcssa91, 0, !dbg !11
  %1349 = extractelement <8 x float> %1348, i64 0, !dbg !11
  %1350 = extractelement <8 x float> %1348, i64 1, !dbg !11
  %1351 = extractelement <8 x float> %1348, i64 2, !dbg !11
  %1352 = extractelement <8 x float> %1348, i64 3, !dbg !11
  %1353 = select i1 %.not, float %1350, float %1349, !dbg !11
  %1354 = bitcast float %1353 to i32, !dbg !11
  %1355 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1354), !dbg !11
  %1356 = bitcast i32 %1355 to float, !dbg !11
  %1357 = select i1 %.not, float %1349, float %1356, !dbg !11
  %1358 = select i1 %.not, float %1356, float %1350, !dbg !11
  %1359 = select i1 %.not, float %1352, float %1351, !dbg !11
  %1360 = bitcast float %1359 to i32, !dbg !11
  %1361 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1360), !dbg !11
  %1362 = bitcast i32 %1361 to float, !dbg !11
  %1363 = select i1 %.not, float %1351, float %1362, !dbg !11
  %1364 = select i1 %.not, float %1362, float %1352, !dbg !11
  %1365 = select i1 %.not36, float %1363, float %1357, !dbg !11
  %1366 = bitcast float %1365 to i32, !dbg !11
  %1367 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1366), !dbg !11
  %1368 = bitcast i32 %1367 to float, !dbg !11
  %1369 = select i1 %.not36, float %1364, float %1358, !dbg !11
  %1370 = bitcast float %1369 to i32, !dbg !11
  %1371 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1370), !dbg !11
  %1372 = bitcast i32 %1371 to float, !dbg !11
  %1373 = insertelement <4 x float> poison, float %1357, i64 0, !dbg !11
  %1374 = insertelement <4 x float> %1373, float %1358, i64 1, !dbg !11
  %1375 = insertelement <4 x float> %1374, float %1368, i64 2, !dbg !11
  %1376 = insertelement <4 x float> %1375, float %1372, i64 3, !dbg !11
  %1377 = insertelement <4 x float> poison, float %1368, i64 0, !dbg !11
  %1378 = insertelement <4 x float> %1377, float %1372, i64 1, !dbg !11
  %1379 = insertelement <4 x float> %1378, float %1363, i64 2, !dbg !11
  %1380 = insertelement <4 x float> %1379, float %1364, i64 3, !dbg !11
  %1381 = select <4 x i1> %1338, <4 x float> %1376, <4 x float> %1380, !dbg !11
  %1382 = extractvalue [2 x <8 x float>] %.lcssa92, 0, !dbg !11
  %1383 = extractelement <8 x float> %1382, i64 0, !dbg !11
  %1384 = extractelement <8 x float> %1382, i64 1, !dbg !11
  %1385 = extractelement <8 x float> %1382, i64 2, !dbg !11
  %1386 = extractelement <8 x float> %1382, i64 3, !dbg !11
  %1387 = select i1 %.not, float %1384, float %1383, !dbg !11
  %1388 = bitcast float %1387 to i32, !dbg !11
  %1389 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1388), !dbg !11
  %1390 = bitcast i32 %1389 to float, !dbg !11
  %1391 = select i1 %.not, float %1383, float %1390, !dbg !11
  %1392 = select i1 %.not, float %1390, float %1384, !dbg !11
  %1393 = select i1 %.not, float %1386, float %1385, !dbg !11
  %1394 = bitcast float %1393 to i32, !dbg !11
  %1395 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1394), !dbg !11
  %1396 = bitcast i32 %1395 to float, !dbg !11
  %1397 = select i1 %.not, float %1385, float %1396, !dbg !11
  %1398 = select i1 %.not, float %1396, float %1386, !dbg !11
  %1399 = select i1 %.not36, float %1397, float %1391, !dbg !11
  %1400 = bitcast float %1399 to i32, !dbg !11
  %1401 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1400), !dbg !11
  %1402 = bitcast i32 %1401 to float, !dbg !11
  %1403 = select i1 %.not36, float %1398, float %1392, !dbg !11
  %1404 = bitcast float %1403 to i32, !dbg !11
  %1405 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1404), !dbg !11
  %1406 = bitcast i32 %1405 to float, !dbg !11
  %1407 = insertelement <4 x float> poison, float %1391, i64 0, !dbg !11
  %1408 = insertelement <4 x float> %1407, float %1392, i64 1, !dbg !11
  %1409 = insertelement <4 x float> %1408, float %1402, i64 2, !dbg !11
  %1410 = insertelement <4 x float> %1409, float %1406, i64 3, !dbg !11
  %1411 = insertelement <4 x float> poison, float %1402, i64 0, !dbg !11
  %1412 = insertelement <4 x float> %1411, float %1406, i64 1, !dbg !11
  %1413 = insertelement <4 x float> %1412, float %1397, i64 2, !dbg !11
  %1414 = insertelement <4 x float> %1413, float %1398, i64 3, !dbg !11
  %1415 = select <4 x i1> %1338, <4 x float> %1410, <4 x float> %1414, !dbg !11
  %1416 = extractvalue [2 x <8 x float>] %.lcssa94, 0, !dbg !11
  %1417 = extractelement <8 x float> %1416, i64 0, !dbg !11
  %1418 = extractelement <8 x float> %1416, i64 1, !dbg !11
  %1419 = extractelement <8 x float> %1416, i64 2, !dbg !11
  %1420 = extractelement <8 x float> %1416, i64 3, !dbg !11
  %1421 = select i1 %.not, float %1418, float %1417, !dbg !11
  %1422 = bitcast float %1421 to i32, !dbg !11
  %1423 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1422), !dbg !11
  %1424 = bitcast i32 %1423 to float, !dbg !11
  %1425 = select i1 %.not, float %1417, float %1424, !dbg !11
  %1426 = select i1 %.not, float %1424, float %1418, !dbg !11
  %1427 = select i1 %.not, float %1420, float %1419, !dbg !11
  %1428 = bitcast float %1427 to i32, !dbg !11
  %1429 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1428), !dbg !11
  %1430 = bitcast i32 %1429 to float, !dbg !11
  %1431 = select i1 %.not, float %1419, float %1430, !dbg !11
  %1432 = select i1 %.not, float %1430, float %1420, !dbg !11
  %1433 = select i1 %.not36, float %1431, float %1425, !dbg !11
  %1434 = bitcast float %1433 to i32, !dbg !11
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1434), !dbg !11
  %1436 = select i1 %.not36, float %1432, float %1426, !dbg !11
  %1437 = bitcast float %1436 to i32, !dbg !11
  %1438 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1437), !dbg !11
  %1439 = shufflevector <1 x float> %1303, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %1440 = extractelement <8 x float> %1306, i64 4, !dbg !11
  %1441 = extractelement <8 x float> %1306, i64 5, !dbg !11
  %1442 = extractelement <8 x float> %1306, i64 6, !dbg !11
  %1443 = extractelement <8 x float> %1306, i64 7, !dbg !11
  %1444 = select i1 %.not, float %1441, float %1440, !dbg !11
  %1445 = bitcast float %1444 to i32, !dbg !11
  %1446 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1445), !dbg !11
  %1447 = bitcast i32 %1446 to float, !dbg !11
  %1448 = select i1 %.not, float %1440, float %1447, !dbg !11
  %1449 = select i1 %.not, float %1447, float %1441, !dbg !11
  %1450 = select i1 %.not, float %1443, float %1442, !dbg !11
  %1451 = bitcast float %1450 to i32, !dbg !11
  %1452 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1451), !dbg !11
  %1453 = bitcast i32 %1452 to float, !dbg !11
  %1454 = select i1 %.not, float %1442, float %1453, !dbg !11
  %1455 = select i1 %.not, float %1453, float %1443, !dbg !11
  %1456 = select i1 %.not36, float %1454, float %1448, !dbg !11
  %1457 = bitcast float %1456 to i32, !dbg !11
  %1458 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1457), !dbg !11
  %1459 = bitcast i32 %1458 to float, !dbg !11
  %1460 = select i1 %.not36, float %1455, float %1449, !dbg !11
  %1461 = bitcast float %1460 to i32, !dbg !11
  %1462 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1461), !dbg !11
  %1463 = bitcast i32 %1462 to float, !dbg !11
  %1464 = insertelement <4 x float> poison, float %1448, i64 0, !dbg !11
  %1465 = insertelement <4 x float> %1464, float %1449, i64 1, !dbg !11
  %1466 = insertelement <4 x float> %1465, float %1459, i64 2, !dbg !11
  %1467 = insertelement <4 x float> %1466, float %1463, i64 3, !dbg !11
  %1468 = insertelement <4 x float> poison, float %1459, i64 0, !dbg !11
  %1469 = insertelement <4 x float> %1468, float %1463, i64 1, !dbg !11
  %1470 = insertelement <4 x float> %1469, float %1454, i64 2, !dbg !11
  %1471 = insertelement <4 x float> %1470, float %1455, i64 3, !dbg !11
  %1472 = select <4 x i1> %1338, <4 x float> %1467, <4 x float> %1471, !dbg !11
  %1473 = extractelement <8 x float> %1348, i64 4, !dbg !11
  %1474 = extractelement <8 x float> %1348, i64 5, !dbg !11
  %1475 = extractelement <8 x float> %1348, i64 6, !dbg !11
  %1476 = extractelement <8 x float> %1348, i64 7, !dbg !11
  %1477 = select i1 %.not, float %1474, float %1473, !dbg !11
  %1478 = bitcast float %1477 to i32, !dbg !11
  %1479 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1478), !dbg !11
  %1480 = bitcast i32 %1479 to float, !dbg !11
  %1481 = select i1 %.not, float %1473, float %1480, !dbg !11
  %1482 = select i1 %.not, float %1480, float %1474, !dbg !11
  %1483 = select i1 %.not, float %1476, float %1475, !dbg !11
  %1484 = bitcast float %1483 to i32, !dbg !11
  %1485 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1484), !dbg !11
  %1486 = bitcast i32 %1485 to float, !dbg !11
  %1487 = select i1 %.not, float %1475, float %1486, !dbg !11
  %1488 = select i1 %.not, float %1486, float %1476, !dbg !11
  %1489 = select i1 %.not36, float %1487, float %1481, !dbg !11
  %1490 = bitcast float %1489 to i32, !dbg !11
  %1491 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1490), !dbg !11
  %1492 = bitcast i32 %1491 to float, !dbg !11
  %1493 = select i1 %.not36, float %1488, float %1482, !dbg !11
  %1494 = bitcast float %1493 to i32, !dbg !11
  %1495 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1494), !dbg !11
  %1496 = bitcast i32 %1495 to float, !dbg !11
  %1497 = insertelement <4 x float> poison, float %1481, i64 0, !dbg !11
  %1498 = insertelement <4 x float> %1497, float %1482, i64 1, !dbg !11
  %1499 = insertelement <4 x float> %1498, float %1492, i64 2, !dbg !11
  %1500 = insertelement <4 x float> %1499, float %1496, i64 3, !dbg !11
  %1501 = insertelement <4 x float> poison, float %1492, i64 0, !dbg !11
  %1502 = insertelement <4 x float> %1501, float %1496, i64 1, !dbg !11
  %1503 = insertelement <4 x float> %1502, float %1487, i64 2, !dbg !11
  %1504 = insertelement <4 x float> %1503, float %1488, i64 3, !dbg !11
  %1505 = select <4 x i1> %1338, <4 x float> %1500, <4 x float> %1504, !dbg !11
  %1506 = extractelement <8 x float> %1382, i64 4, !dbg !11
  %1507 = extractelement <8 x float> %1382, i64 5, !dbg !11
  %1508 = extractelement <8 x float> %1382, i64 6, !dbg !11
  %1509 = extractelement <8 x float> %1382, i64 7, !dbg !11
  %1510 = select i1 %.not, float %1507, float %1506, !dbg !11
  %1511 = bitcast float %1510 to i32, !dbg !11
  %1512 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1511), !dbg !11
  %1513 = bitcast i32 %1512 to float, !dbg !11
  %1514 = select i1 %.not, float %1506, float %1513, !dbg !11
  %1515 = select i1 %.not, float %1513, float %1507, !dbg !11
  %1516 = select i1 %.not, float %1509, float %1508, !dbg !11
  %1517 = bitcast float %1516 to i32, !dbg !11
  %1518 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1517), !dbg !11
  %1519 = bitcast i32 %1518 to float, !dbg !11
  %1520 = select i1 %.not, float %1508, float %1519, !dbg !11
  %1521 = select i1 %.not, float %1519, float %1509, !dbg !11
  %1522 = select i1 %.not36, float %1520, float %1514, !dbg !11
  %1523 = bitcast float %1522 to i32, !dbg !11
  %1524 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1523), !dbg !11
  %1525 = bitcast i32 %1524 to float, !dbg !11
  %1526 = select i1 %.not36, float %1521, float %1515, !dbg !11
  %1527 = bitcast float %1526 to i32, !dbg !11
  %1528 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1527), !dbg !11
  %1529 = bitcast i32 %1528 to float, !dbg !11
  %1530 = insertelement <4 x float> poison, float %1514, i64 0, !dbg !11
  %1531 = insertelement <4 x float> %1530, float %1515, i64 1, !dbg !11
  %1532 = insertelement <4 x float> %1531, float %1525, i64 2, !dbg !11
  %1533 = insertelement <4 x float> %1532, float %1529, i64 3, !dbg !11
  %1534 = insertelement <4 x float> poison, float %1525, i64 0, !dbg !11
  %1535 = insertelement <4 x float> %1534, float %1529, i64 1, !dbg !11
  %1536 = insertelement <4 x float> %1535, float %1520, i64 2, !dbg !11
  %1537 = insertelement <4 x float> %1536, float %1521, i64 3, !dbg !11
  %1538 = select <4 x i1> %1338, <4 x float> %1533, <4 x float> %1537, !dbg !11
  %1539 = extractelement <8 x float> %1416, i64 4, !dbg !11
  %1540 = extractelement <8 x float> %1416, i64 5, !dbg !11
  %1541 = extractelement <8 x float> %1416, i64 6, !dbg !11
  %1542 = extractelement <8 x float> %1416, i64 7, !dbg !11
  %1543 = select i1 %.not, float %1540, float %1539, !dbg !11
  %1544 = bitcast float %1543 to i32, !dbg !11
  %1545 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1544), !dbg !11
  %1546 = bitcast i32 %1545 to float, !dbg !11
  %1547 = select i1 %.not, float %1539, float %1546, !dbg !11
  %1548 = select i1 %.not, float %1546, float %1540, !dbg !11
  %1549 = select i1 %.not, float %1542, float %1541, !dbg !11
  %1550 = bitcast float %1549 to i32, !dbg !11
  %1551 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1550), !dbg !11
  %1552 = bitcast i32 %1551 to float, !dbg !11
  %1553 = select i1 %.not, float %1541, float %1552, !dbg !11
  %1554 = select i1 %.not, float %1552, float %1542, !dbg !11
  %1555 = select i1 %.not36, float %1553, float %1547, !dbg !11
  %1556 = bitcast float %1555 to i32, !dbg !11
  %1557 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1556), !dbg !11
  %1558 = select i1 %.not36, float %1554, float %1548, !dbg !11
  %1559 = bitcast float %1558 to i32, !dbg !11
  %1560 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1559), !dbg !11
  %1561 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !11
  %1562 = extractelement <8 x float> %1561, i64 0, !dbg !11
  %1563 = extractelement <8 x float> %1561, i64 1, !dbg !11
  %1564 = extractelement <8 x float> %1561, i64 2, !dbg !11
  %1565 = extractelement <8 x float> %1561, i64 3, !dbg !11
  %1566 = select i1 %.not, float %1563, float %1562, !dbg !11
  %1567 = bitcast float %1566 to i32, !dbg !11
  %1568 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1567), !dbg !11
  %1569 = bitcast i32 %1568 to float, !dbg !11
  %1570 = select i1 %.not, float %1562, float %1569, !dbg !11
  %1571 = select i1 %.not, float %1569, float %1563, !dbg !11
  %1572 = select i1 %.not, float %1565, float %1564, !dbg !11
  %1573 = bitcast float %1572 to i32, !dbg !11
  %1574 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1573), !dbg !11
  %1575 = bitcast i32 %1574 to float, !dbg !11
  %1576 = select i1 %.not, float %1564, float %1575, !dbg !11
  %1577 = select i1 %.not, float %1575, float %1565, !dbg !11
  %1578 = select i1 %.not36, float %1576, float %1570, !dbg !11
  %1579 = bitcast float %1578 to i32, !dbg !11
  %1580 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1579), !dbg !11
  %1581 = bitcast i32 %1580 to float, !dbg !11
  %1582 = select i1 %.not36, float %1577, float %1571, !dbg !11
  %1583 = bitcast float %1582 to i32, !dbg !11
  %1584 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1583), !dbg !11
  %1585 = bitcast i32 %1584 to float, !dbg !11
  %1586 = insertelement <4 x float> poison, float %1570, i64 0, !dbg !11
  %1587 = insertelement <4 x float> %1586, float %1571, i64 1, !dbg !11
  %1588 = insertelement <4 x float> %1587, float %1581, i64 2, !dbg !11
  %1589 = insertelement <4 x float> %1588, float %1585, i64 3, !dbg !11
  %1590 = insertelement <4 x float> poison, float %1581, i64 0, !dbg !11
  %1591 = insertelement <4 x float> %1590, float %1585, i64 1, !dbg !11
  %1592 = insertelement <4 x float> %1591, float %1576, i64 2, !dbg !11
  %1593 = insertelement <4 x float> %1592, float %1577, i64 3, !dbg !11
  %1594 = select <4 x i1> %1338, <4 x float> %1589, <4 x float> %1593, !dbg !11
  %1595 = extractvalue [2 x <8 x float>] %.lcssa91, 1, !dbg !11
  %1596 = extractelement <8 x float> %1595, i64 0, !dbg !11
  %1597 = extractelement <8 x float> %1595, i64 1, !dbg !11
  %1598 = extractelement <8 x float> %1595, i64 2, !dbg !11
  %1599 = extractelement <8 x float> %1595, i64 3, !dbg !11
  %1600 = select i1 %.not, float %1597, float %1596, !dbg !11
  %1601 = bitcast float %1600 to i32, !dbg !11
  %1602 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1601), !dbg !11
  %1603 = bitcast i32 %1602 to float, !dbg !11
  %1604 = select i1 %.not, float %1596, float %1603, !dbg !11
  %1605 = select i1 %.not, float %1603, float %1597, !dbg !11
  %1606 = select i1 %.not, float %1599, float %1598, !dbg !11
  %1607 = bitcast float %1606 to i32, !dbg !11
  %1608 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1607), !dbg !11
  %1609 = bitcast i32 %1608 to float, !dbg !11
  %1610 = select i1 %.not, float %1598, float %1609, !dbg !11
  %1611 = select i1 %.not, float %1609, float %1599, !dbg !11
  %1612 = select i1 %.not36, float %1610, float %1604, !dbg !11
  %1613 = bitcast float %1612 to i32, !dbg !11
  %1614 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1613), !dbg !11
  %1615 = bitcast i32 %1614 to float, !dbg !11
  %1616 = select i1 %.not36, float %1611, float %1605, !dbg !11
  %1617 = bitcast float %1616 to i32, !dbg !11
  %1618 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1617), !dbg !11
  %1619 = bitcast i32 %1618 to float, !dbg !11
  %1620 = insertelement <4 x float> poison, float %1604, i64 0, !dbg !11
  %1621 = insertelement <4 x float> %1620, float %1605, i64 1, !dbg !11
  %1622 = insertelement <4 x float> %1621, float %1615, i64 2, !dbg !11
  %1623 = insertelement <4 x float> %1622, float %1619, i64 3, !dbg !11
  %1624 = insertelement <4 x float> poison, float %1615, i64 0, !dbg !11
  %1625 = insertelement <4 x float> %1624, float %1619, i64 1, !dbg !11
  %1626 = insertelement <4 x float> %1625, float %1610, i64 2, !dbg !11
  %1627 = insertelement <4 x float> %1626, float %1611, i64 3, !dbg !11
  %1628 = select <4 x i1> %1338, <4 x float> %1623, <4 x float> %1627, !dbg !11
  %1629 = extractvalue [2 x <8 x float>] %.lcssa92, 1, !dbg !11
  %1630 = extractelement <8 x float> %1629, i64 0, !dbg !11
  %1631 = extractelement <8 x float> %1629, i64 1, !dbg !11
  %1632 = extractelement <8 x float> %1629, i64 2, !dbg !11
  %1633 = extractelement <8 x float> %1629, i64 3, !dbg !11
  %1634 = select i1 %.not, float %1631, float %1630, !dbg !11
  %1635 = bitcast float %1634 to i32, !dbg !11
  %1636 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1635), !dbg !11
  %1637 = bitcast i32 %1636 to float, !dbg !11
  %1638 = select i1 %.not, float %1630, float %1637, !dbg !11
  %1639 = select i1 %.not, float %1637, float %1631, !dbg !11
  %1640 = select i1 %.not, float %1633, float %1632, !dbg !11
  %1641 = bitcast float %1640 to i32, !dbg !11
  %1642 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1641), !dbg !11
  %1643 = bitcast i32 %1642 to float, !dbg !11
  %1644 = select i1 %.not, float %1632, float %1643, !dbg !11
  %1645 = select i1 %.not, float %1643, float %1633, !dbg !11
  %1646 = select i1 %.not36, float %1644, float %1638, !dbg !11
  %1647 = bitcast float %1646 to i32, !dbg !11
  %1648 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1647), !dbg !11
  %1649 = bitcast i32 %1648 to float, !dbg !11
  %1650 = select i1 %.not36, float %1645, float %1639, !dbg !11
  %1651 = bitcast float %1650 to i32, !dbg !11
  %1652 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1651), !dbg !11
  %1653 = bitcast i32 %1652 to float, !dbg !11
  %1654 = insertelement <4 x float> poison, float %1638, i64 0, !dbg !11
  %1655 = insertelement <4 x float> %1654, float %1639, i64 1, !dbg !11
  %1656 = insertelement <4 x float> %1655, float %1649, i64 2, !dbg !11
  %1657 = insertelement <4 x float> %1656, float %1653, i64 3, !dbg !11
  %1658 = insertelement <4 x float> poison, float %1649, i64 0, !dbg !11
  %1659 = insertelement <4 x float> %1658, float %1653, i64 1, !dbg !11
  %1660 = insertelement <4 x float> %1659, float %1644, i64 2, !dbg !11
  %1661 = insertelement <4 x float> %1660, float %1645, i64 3, !dbg !11
  %1662 = select <4 x i1> %1338, <4 x float> %1657, <4 x float> %1661, !dbg !11
  %1663 = extractvalue [2 x <8 x float>] %.lcssa94, 1, !dbg !11
  %1664 = extractelement <8 x float> %1663, i64 0, !dbg !11
  %1665 = extractelement <8 x float> %1663, i64 1, !dbg !11
  %1666 = extractelement <8 x float> %1663, i64 2, !dbg !11
  %1667 = extractelement <8 x float> %1663, i64 3, !dbg !11
  %1668 = select i1 %.not, float %1665, float %1664, !dbg !11
  %1669 = bitcast float %1668 to i32, !dbg !11
  %1670 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1669), !dbg !11
  %1671 = bitcast i32 %1670 to float, !dbg !11
  %1672 = select i1 %.not, float %1664, float %1671, !dbg !11
  %1673 = select i1 %.not, float %1671, float %1665, !dbg !11
  %1674 = select i1 %.not, float %1667, float %1666, !dbg !11
  %1675 = bitcast float %1674 to i32, !dbg !11
  %1676 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1675), !dbg !11
  %1677 = bitcast i32 %1676 to float, !dbg !11
  %1678 = select i1 %.not, float %1666, float %1677, !dbg !11
  %1679 = select i1 %.not, float %1677, float %1667, !dbg !11
  %1680 = select i1 %.not36, float %1678, float %1672, !dbg !11
  %1681 = bitcast float %1680 to i32, !dbg !11
  %1682 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1681), !dbg !11
  %1683 = select i1 %.not36, float %1679, float %1673, !dbg !11
  %1684 = bitcast float %1683 to i32, !dbg !11
  %1685 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1684), !dbg !11
  %1686 = shufflevector <1 x float> %1298, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %1687 = extractelement <8 x float> %1561, i64 4, !dbg !11
  %1688 = extractelement <8 x float> %1561, i64 5, !dbg !11
  %1689 = extractelement <8 x float> %1561, i64 6, !dbg !11
  %1690 = extractelement <8 x float> %1561, i64 7, !dbg !11
  %1691 = select i1 %.not, float %1688, float %1687, !dbg !11
  %1692 = bitcast float %1691 to i32, !dbg !11
  %1693 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1692), !dbg !11
  %1694 = bitcast i32 %1693 to float, !dbg !11
  %1695 = select i1 %.not, float %1687, float %1694, !dbg !11
  %1696 = select i1 %.not, float %1694, float %1688, !dbg !11
  %1697 = select i1 %.not, float %1690, float %1689, !dbg !11
  %1698 = bitcast float %1697 to i32, !dbg !11
  %1699 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1698), !dbg !11
  %1700 = bitcast i32 %1699 to float, !dbg !11
  %1701 = select i1 %.not, float %1689, float %1700, !dbg !11
  %1702 = select i1 %.not, float %1700, float %1690, !dbg !11
  %1703 = select i1 %.not36, float %1701, float %1695, !dbg !11
  %1704 = bitcast float %1703 to i32, !dbg !11
  %1705 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1704), !dbg !11
  %1706 = bitcast i32 %1705 to float, !dbg !11
  %1707 = select i1 %.not36, float %1702, float %1696, !dbg !11
  %1708 = bitcast float %1707 to i32, !dbg !11
  %1709 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1708), !dbg !11
  %1710 = bitcast i32 %1709 to float, !dbg !11
  %1711 = insertelement <4 x float> poison, float %1695, i64 0, !dbg !11
  %1712 = insertelement <4 x float> %1711, float %1696, i64 1, !dbg !11
  %1713 = insertelement <4 x float> %1712, float %1706, i64 2, !dbg !11
  %1714 = insertelement <4 x float> %1713, float %1710, i64 3, !dbg !11
  %1715 = insertelement <4 x float> poison, float %1706, i64 0, !dbg !11
  %1716 = insertelement <4 x float> %1715, float %1710, i64 1, !dbg !11
  %1717 = insertelement <4 x float> %1716, float %1701, i64 2, !dbg !11
  %1718 = insertelement <4 x float> %1717, float %1702, i64 3, !dbg !11
  %1719 = select <4 x i1> %1338, <4 x float> %1714, <4 x float> %1718, !dbg !11
  %1720 = extractelement <8 x float> %1595, i64 4, !dbg !11
  %1721 = extractelement <8 x float> %1595, i64 5, !dbg !11
  %1722 = extractelement <8 x float> %1595, i64 6, !dbg !11
  %1723 = extractelement <8 x float> %1595, i64 7, !dbg !11
  %1724 = select i1 %.not, float %1721, float %1720, !dbg !11
  %1725 = bitcast float %1724 to i32, !dbg !11
  %1726 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1725), !dbg !11
  %1727 = bitcast i32 %1726 to float, !dbg !11
  %1728 = select i1 %.not, float %1720, float %1727, !dbg !11
  %1729 = select i1 %.not, float %1727, float %1721, !dbg !11
  %1730 = select i1 %.not, float %1723, float %1722, !dbg !11
  %1731 = bitcast float %1730 to i32, !dbg !11
  %1732 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1731), !dbg !11
  %1733 = bitcast i32 %1732 to float, !dbg !11
  %1734 = select i1 %.not, float %1722, float %1733, !dbg !11
  %1735 = select i1 %.not, float %1733, float %1723, !dbg !11
  %1736 = select i1 %.not36, float %1734, float %1728, !dbg !11
  %1737 = bitcast float %1736 to i32, !dbg !11
  %1738 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1737), !dbg !11
  %1739 = bitcast i32 %1738 to float, !dbg !11
  %1740 = select i1 %.not36, float %1735, float %1729, !dbg !11
  %1741 = bitcast float %1740 to i32, !dbg !11
  %1742 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1741), !dbg !11
  %1743 = bitcast i32 %1742 to float, !dbg !11
  %1744 = insertelement <4 x float> poison, float %1728, i64 0, !dbg !11
  %1745 = insertelement <4 x float> %1744, float %1729, i64 1, !dbg !11
  %1746 = insertelement <4 x float> %1745, float %1739, i64 2, !dbg !11
  %1747 = insertelement <4 x float> %1746, float %1743, i64 3, !dbg !11
  %1748 = insertelement <4 x float> poison, float %1739, i64 0, !dbg !11
  %1749 = insertelement <4 x float> %1748, float %1743, i64 1, !dbg !11
  %1750 = insertelement <4 x float> %1749, float %1734, i64 2, !dbg !11
  %1751 = insertelement <4 x float> %1750, float %1735, i64 3, !dbg !11
  %1752 = select <4 x i1> %1338, <4 x float> %1747, <4 x float> %1751, !dbg !11
  %1753 = extractelement <8 x float> %1629, i64 4, !dbg !11
  %1754 = extractelement <8 x float> %1629, i64 5, !dbg !11
  %1755 = extractelement <8 x float> %1629, i64 6, !dbg !11
  %1756 = extractelement <8 x float> %1629, i64 7, !dbg !11
  %1757 = select i1 %.not, float %1754, float %1753, !dbg !11
  %1758 = bitcast float %1757 to i32, !dbg !11
  %1759 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1758), !dbg !11
  %1760 = bitcast i32 %1759 to float, !dbg !11
  %1761 = select i1 %.not, float %1753, float %1760, !dbg !11
  %1762 = select i1 %.not, float %1760, float %1754, !dbg !11
  %1763 = select i1 %.not, float %1756, float %1755, !dbg !11
  %1764 = bitcast float %1763 to i32, !dbg !11
  %1765 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1764), !dbg !11
  %1766 = bitcast i32 %1765 to float, !dbg !11
  %1767 = select i1 %.not, float %1755, float %1766, !dbg !11
  %1768 = select i1 %.not, float %1766, float %1756, !dbg !11
  %1769 = select i1 %.not36, float %1767, float %1761, !dbg !11
  %1770 = bitcast float %1769 to i32, !dbg !11
  %1771 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1770), !dbg !11
  %1772 = bitcast i32 %1771 to float, !dbg !11
  %1773 = select i1 %.not36, float %1768, float %1762, !dbg !11
  %1774 = bitcast float %1773 to i32, !dbg !11
  %1775 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1774), !dbg !11
  %1776 = bitcast i32 %1775 to float, !dbg !11
  %1777 = insertelement <4 x float> poison, float %1761, i64 0, !dbg !11
  %1778 = insertelement <4 x float> %1777, float %1762, i64 1, !dbg !11
  %1779 = insertelement <4 x float> %1778, float %1772, i64 2, !dbg !11
  %1780 = insertelement <4 x float> %1779, float %1776, i64 3, !dbg !11
  %1781 = insertelement <4 x float> poison, float %1772, i64 0, !dbg !11
  %1782 = insertelement <4 x float> %1781, float %1776, i64 1, !dbg !11
  %1783 = insertelement <4 x float> %1782, float %1767, i64 2, !dbg !11
  %1784 = insertelement <4 x float> %1783, float %1768, i64 3, !dbg !11
  %1785 = select <4 x i1> %1338, <4 x float> %1780, <4 x float> %1784, !dbg !11
  %1786 = extractelement <8 x float> %1663, i64 4, !dbg !11
  %1787 = extractelement <8 x float> %1663, i64 5, !dbg !11
  %1788 = extractelement <8 x float> %1663, i64 6, !dbg !11
  %1789 = extractelement <8 x float> %1663, i64 7, !dbg !11
  %1790 = select i1 %.not, float %1787, float %1786, !dbg !11
  %1791 = bitcast float %1790 to i32, !dbg !11
  %1792 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1791), !dbg !11
  %1793 = bitcast i32 %1792 to float, !dbg !11
  %1794 = select i1 %.not, float %1786, float %1793, !dbg !11
  %1795 = select i1 %.not, float %1793, float %1787, !dbg !11
  %1796 = select i1 %.not, float %1789, float %1788, !dbg !11
  %1797 = bitcast float %1796 to i32, !dbg !11
  %1798 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1314, i32 %1797), !dbg !11
  %1799 = bitcast i32 %1798 to float, !dbg !11
  %1800 = select i1 %.not, float %1788, float %1799, !dbg !11
  %1801 = select i1 %.not, float %1799, float %1789, !dbg !11
  %1802 = select i1 %.not36, float %1800, float %1794, !dbg !11
  %1803 = bitcast float %1802 to i32, !dbg !11
  %1804 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1803), !dbg !11
  %1805 = select i1 %.not36, float %1801, float %1795, !dbg !11
  %1806 = bitcast float %1805 to i32, !dbg !11
  %1807 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1329, i32 %1806), !dbg !11
  %1808 = bitcast i32 %1804 to float, !dbg !11
  %1809 = bitcast i32 %1807 to float, !dbg !11
  %1810 = insertelement <4 x float> poison, float %1794, i64 0, !dbg !11
  %1811 = insertelement <4 x float> %1810, float %1795, i64 1, !dbg !11
  %1812 = insertelement <4 x float> %1811, float %1808, i64 2, !dbg !11
  %1813 = insertelement <4 x float> %1812, float %1809, i64 3, !dbg !11
  %1814 = insertelement <4 x float> poison, float %1808, i64 0, !dbg !11
  %1815 = insertelement <4 x float> %1814, float %1809, i64 1, !dbg !11
  %1816 = insertelement <4 x float> %1815, float %1800, i64 2, !dbg !11
  %1817 = insertelement <4 x float> %1816, float %1801, i64 3, !dbg !11
  %1818 = select <4 x i1> %1338, <4 x float> %1813, <4 x float> %1817, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1686, !dbg !30
  %normalized = fmul <4 x float> %row.reciprocal, %1818, !dbg !30
  %1819 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !31
  %1820 = extractelement <4 x half> %1819, i64 3, !dbg !31
  %1821 = extractelement <4 x half> %1819, i64 2, !dbg !31
  %1822 = extractelement <4 x half> %1819, i64 1, !dbg !31
  %1823 = extractelement <4 x half> %1819, i64 0, !dbg !31
  %1824 = bitcast i32 %1682 to float, !dbg !11
  %1825 = bitcast i32 %1685 to float, !dbg !11
  %1826 = insertelement <4 x float> poison, float %1672, i64 0, !dbg !11
  %1827 = insertelement <4 x float> %1826, float %1673, i64 1, !dbg !11
  %1828 = insertelement <4 x float> %1827, float %1824, i64 2, !dbg !11
  %1829 = insertelement <4 x float> %1828, float %1825, i64 3, !dbg !11
  %1830 = insertelement <4 x float> poison, float %1824, i64 0, !dbg !11
  %1831 = insertelement <4 x float> %1830, float %1825, i64 1, !dbg !11
  %1832 = insertelement <4 x float> %1831, float %1678, i64 2, !dbg !11
  %1833 = insertelement <4 x float> %1832, float %1679, i64 3, !dbg !11
  %1834 = select <4 x i1> %1338, <4 x float> %1829, <4 x float> %1833, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %1834, !dbg !30
  %1835 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !31
  %1836 = extractelement <4 x half> %1835, i64 3, !dbg !31
  %1837 = extractelement <4 x half> %1835, i64 2, !dbg !31
  %1838 = extractelement <4 x half> %1835, i64 1, !dbg !31
  %1839 = extractelement <4 x half> %1835, i64 0, !dbg !31
  %1840 = bitcast i32 %1557 to float, !dbg !11
  %1841 = bitcast i32 %1560 to float, !dbg !11
  %1842 = insertelement <4 x float> poison, float %1547, i64 0, !dbg !11
  %1843 = insertelement <4 x float> %1842, float %1548, i64 1, !dbg !11
  %1844 = insertelement <4 x float> %1843, float %1840, i64 2, !dbg !11
  %1845 = insertelement <4 x float> %1844, float %1841, i64 3, !dbg !11
  %1846 = insertelement <4 x float> poison, float %1840, i64 0, !dbg !11
  %1847 = insertelement <4 x float> %1846, float %1841, i64 1, !dbg !11
  %1848 = insertelement <4 x float> %1847, float %1553, i64 2, !dbg !11
  %1849 = insertelement <4 x float> %1848, float %1554, i64 3, !dbg !11
  %1850 = select <4 x i1> %1338, <4 x float> %1845, <4 x float> %1849, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1439, !dbg !30
  %normalized106 = fmul <4 x float> %row.reciprocal105, %1850, !dbg !30
  %1851 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !31
  %1852 = extractelement <4 x half> %1851, i64 3, !dbg !31
  %1853 = extractelement <4 x half> %1851, i64 2, !dbg !31
  %1854 = extractelement <4 x half> %1851, i64 1, !dbg !31
  %1855 = extractelement <4 x half> %1851, i64 0, !dbg !31
  %1856 = bitcast i32 %1435 to float, !dbg !11
  %1857 = bitcast i32 %1438 to float, !dbg !11
  %1858 = insertelement <4 x float> poison, float %1425, i64 0, !dbg !11
  %1859 = insertelement <4 x float> %1858, float %1426, i64 1, !dbg !11
  %1860 = insertelement <4 x float> %1859, float %1856, i64 2, !dbg !11
  %1861 = insertelement <4 x float> %1860, float %1857, i64 3, !dbg !11
  %1862 = insertelement <4 x float> poison, float %1856, i64 0, !dbg !11
  %1863 = insertelement <4 x float> %1862, float %1857, i64 1, !dbg !11
  %1864 = insertelement <4 x float> %1863, float %1431, i64 2, !dbg !11
  %1865 = insertelement <4 x float> %1864, float %1432, i64 3, !dbg !11
  %1866 = select <4 x i1> %1338, <4 x float> %1861, <4 x float> %1865, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %1866, !dbg !30
  %1867 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !31
  %1868 = extractelement <4 x half> %1867, i64 3, !dbg !31
  %1869 = extractelement <4 x half> %1867, i64 2, !dbg !31
  %1870 = extractelement <4 x half> %1867, i64 1, !dbg !31
  %1871 = extractelement <4 x half> %1867, i64 0, !dbg !31
  %address.thread1100 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !32, !range !9, !frisk.register_lifetimes !5
  %address.local1101 = zext nneg i32 %address.thread1100 to i64, !dbg !8
  %1872 = lshr i64 %address.local1101, 2, !dbg !32
  %address.local1104 = and i64 %1872, 48, !dbg !32
  %address.local1106 = and i64 %address.local1101, 240, !dbg !32
  %address.local1105 = add i64 %26, %address.local1101, !dbg !32
  %address.local1107 = sub i64 %address.local1105, %address.local1106, !dbg !32
  %address.local1108 = add i64 %address.local1107, %address.local1104, !dbg !32
  %1873 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28, !dbg !32
  %address.local1111.idx = shl nuw nsw i64 %address.local1108, 8, !dbg !32
  %address.local1111 = getelementptr inbounds nuw i8, ptr addrspace(1) %1873, i64 %address.local1111.idx, !dbg !32
  %address.local1113 = and i64 %1872, 12, !dbg !10
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1113, !dbg !32
  store half %1871, ptr addrspace(1) %address.local1114, align 2, !dbg !32
  %address.local1116 = or disjoint i64 %address.local1113, 1, !dbg !32
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1116, !dbg !32
  store half %1870, ptr addrspace(1) %address.local1117, align 2, !dbg !32
  %address.local1119 = or disjoint i64 %address.local1113, 2, !dbg !32
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1119, !dbg !32
  store half %1869, ptr addrspace(1) %address.local1120, align 2, !dbg !32
  %address.local1122 = or disjoint i64 %address.local1113, 3, !dbg !32
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1122, !dbg !32
  store half %1868, ptr addrspace(1) %address.local1123, align 2, !dbg !32
  %address.local1125 = or disjoint i64 %address.local1113, 16, !dbg !32
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1125, !dbg !32
  store half %1855, ptr addrspace(1) %address.local1126, align 2, !dbg !32
  %address.local1128 = or disjoint i64 %address.local1113, 17, !dbg !32
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1128, !dbg !32
  store half %1854, ptr addrspace(1) %address.local1129, align 2, !dbg !32
  %address.local1131 = or disjoint i64 %address.local1113, 18, !dbg !32
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1131, !dbg !32
  store half %1853, ptr addrspace(1) %address.local1132, align 2, !dbg !32
  %address.local1134 = or disjoint i64 %address.local1113, 19, !dbg !32
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1134, !dbg !32
  store half %1852, ptr addrspace(1) %address.local1135, align 2, !dbg !32
  %address.local1139 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1111, i64 16384, !dbg !32
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1113, !dbg !32
  store half %1839, ptr addrspace(1) %address.local1140, align 2, !dbg !32
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1116, !dbg !32
  store half %1838, ptr addrspace(1) %address.local1142, align 2, !dbg !32
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1119, !dbg !32
  store half %1837, ptr addrspace(1) %address.local1144, align 2, !dbg !32
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1122, !dbg !32
  store half %1836, ptr addrspace(1) %address.local1146, align 2, !dbg !32
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1125, !dbg !32
  store half %1823, ptr addrspace(1) %address.local1148, align 2, !dbg !32
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1128, !dbg !32
  store half %1822, ptr addrspace(1) %address.local1150, align 2, !dbg !32
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1131, !dbg !32
  store half %1821, ptr addrspace(1) %address.local1152, align 2, !dbg !32
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1134, !dbg !32
  store half %1820, ptr addrspace(1) %address.local1154, align 2, !dbg !32
  %normalized109 = fmul <4 x float> %row.reciprocal, %1785, !dbg !30
  %1874 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !31
  %1875 = extractelement <4 x half> %1874, i64 3, !dbg !31
  %1876 = extractelement <4 x half> %1874, i64 2, !dbg !31
  %1877 = extractelement <4 x half> %1874, i64 1, !dbg !31
  %1878 = extractelement <4 x half> %1874, i64 0, !dbg !31
  %normalized110 = fmul <4 x float> %row.reciprocal, %1662, !dbg !30
  %1879 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !31
  %1880 = extractelement <4 x half> %1879, i64 3, !dbg !31
  %1881 = extractelement <4 x half> %1879, i64 2, !dbg !31
  %1882 = extractelement <4 x half> %1879, i64 1, !dbg !31
  %1883 = extractelement <4 x half> %1879, i64 0, !dbg !31
  %normalized112 = fmul <4 x float> %row.reciprocal105, %1538, !dbg !30
  %1884 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !31
  %1885 = extractelement <4 x half> %1884, i64 3, !dbg !31
  %1886 = extractelement <4 x half> %1884, i64 2, !dbg !31
  %1887 = extractelement <4 x half> %1884, i64 1, !dbg !31
  %1888 = extractelement <4 x half> %1884, i64 0, !dbg !31
  %normalized113 = fmul <4 x float> %row.reciprocal105, %1415, !dbg !30
  %1889 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !31
  %1890 = extractelement <4 x half> %1889, i64 3, !dbg !31
  %1891 = extractelement <4 x half> %1889, i64 2, !dbg !31
  %1892 = extractelement <4 x half> %1889, i64 1, !dbg !31
  %1893 = extractelement <4 x half> %1889, i64 0, !dbg !31
  %address.local1169 = or disjoint i64 %address.local1113, 32, !dbg !32
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1169, !dbg !32
  store half %1893, ptr addrspace(1) %address.local1170, align 2, !dbg !32
  %address.local1172 = or disjoint i64 %address.local1113, 33, !dbg !32
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1172, !dbg !32
  store half %1892, ptr addrspace(1) %address.local1173, align 2, !dbg !32
  %address.local1175 = or disjoint i64 %address.local1113, 34, !dbg !32
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1175, !dbg !32
  store half %1891, ptr addrspace(1) %address.local1176, align 2, !dbg !32
  %address.local1178 = or disjoint i64 %address.local1113, 35, !dbg !32
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1178, !dbg !32
  store half %1890, ptr addrspace(1) %address.local1179, align 2, !dbg !32
  %address.local1181 = or disjoint i64 %address.local1113, 48, !dbg !32
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1181, !dbg !32
  store half %1888, ptr addrspace(1) %address.local1182, align 2, !dbg !32
  %address.local1184 = or disjoint i64 %address.local1113, 49, !dbg !32
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1184, !dbg !32
  store half %1887, ptr addrspace(1) %address.local1185, align 2, !dbg !32
  %address.local1187 = or disjoint i64 %address.local1113, 50, !dbg !32
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1187, !dbg !32
  store half %1886, ptr addrspace(1) %address.local1188, align 2, !dbg !32
  %address.local1190 = or i64 %1872, 51, !dbg !32
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1190, !dbg !32
  store half %1885, ptr addrspace(1) %address.local1191, align 2, !dbg !32
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1169, !dbg !32
  store half %1883, ptr addrspace(1) %address.local1196, align 2, !dbg !32
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1172, !dbg !32
  store half %1882, ptr addrspace(1) %address.local1198, align 2, !dbg !32
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1175, !dbg !32
  store half %1881, ptr addrspace(1) %address.local1200, align 2, !dbg !32
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1178, !dbg !32
  store half %1880, ptr addrspace(1) %address.local1202, align 2, !dbg !32
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1181, !dbg !32
  store half %1878, ptr addrspace(1) %address.local1204, align 2, !dbg !32
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1184, !dbg !32
  store half %1877, ptr addrspace(1) %address.local1206, align 2, !dbg !32
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1187, !dbg !32
  store half %1876, ptr addrspace(1) %address.local1208, align 2, !dbg !32
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1190, !dbg !32
  store half %1875, ptr addrspace(1) %address.local1210, align 2, !dbg !32
  %normalized115 = fmul <4 x float> %row.reciprocal, %1752, !dbg !30
  %1894 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !31
  %1895 = extractelement <4 x half> %1894, i64 3, !dbg !31
  %1896 = extractelement <4 x half> %1894, i64 2, !dbg !31
  %1897 = extractelement <4 x half> %1894, i64 1, !dbg !31
  %1898 = extractelement <4 x half> %1894, i64 0, !dbg !31
  %normalized116 = fmul <4 x float> %row.reciprocal, %1628, !dbg !30
  %1899 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !31
  %1900 = extractelement <4 x half> %1899, i64 3, !dbg !31
  %1901 = extractelement <4 x half> %1899, i64 2, !dbg !31
  %1902 = extractelement <4 x half> %1899, i64 1, !dbg !31
  %1903 = extractelement <4 x half> %1899, i64 0, !dbg !31
  %normalized118 = fmul <4 x float> %row.reciprocal105, %1505, !dbg !30
  %1904 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !31
  %1905 = extractelement <4 x half> %1904, i64 3, !dbg !31
  %1906 = extractelement <4 x half> %1904, i64 2, !dbg !31
  %1907 = extractelement <4 x half> %1904, i64 1, !dbg !31
  %1908 = extractelement <4 x half> %1904, i64 0, !dbg !31
  %normalized119 = fmul <4 x float> %row.reciprocal105, %1381, !dbg !30
  %1909 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !31
  %1910 = extractelement <4 x half> %1909, i64 3, !dbg !31
  %1911 = extractelement <4 x half> %1909, i64 2, !dbg !31
  %1912 = extractelement <4 x half> %1909, i64 1, !dbg !31
  %1913 = extractelement <4 x half> %1909, i64 0, !dbg !31
  %address.local1225 = or disjoint i64 %address.local1113, 64, !dbg !32
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1225, !dbg !32
  store half %1913, ptr addrspace(1) %address.local1226, align 2, !dbg !32
  %address.local1228 = or disjoint i64 %address.local1113, 65, !dbg !32
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1228, !dbg !32
  store half %1912, ptr addrspace(1) %address.local1229, align 2, !dbg !32
  %address.local1231 = or disjoint i64 %address.local1113, 66, !dbg !32
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1231, !dbg !32
  store half %1911, ptr addrspace(1) %address.local1232, align 2, !dbg !32
  %address.local1234 = or disjoint i64 %address.local1113, 67, !dbg !32
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1234, !dbg !32
  store half %1910, ptr addrspace(1) %address.local1235, align 2, !dbg !32
  %address.local1237 = or disjoint i64 %address.local1113, 80, !dbg !32
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1237, !dbg !32
  store half %1908, ptr addrspace(1) %address.local1238, align 2, !dbg !32
  %address.local1240 = or disjoint i64 %address.local1113, 81, !dbg !32
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1240, !dbg !32
  store half %1907, ptr addrspace(1) %address.local1241, align 2, !dbg !32
  %address.local1243 = or disjoint i64 %address.local1113, 82, !dbg !32
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1243, !dbg !32
  store half %1906, ptr addrspace(1) %address.local1244, align 2, !dbg !32
  %address.local1246 = or disjoint i64 %address.local1113, 83, !dbg !32
  %address.local1247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1246, !dbg !32
  store half %1905, ptr addrspace(1) %address.local1247, align 2, !dbg !32
  %address.local1252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1225, !dbg !32
  store half %1903, ptr addrspace(1) %address.local1252, align 2, !dbg !32
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1228, !dbg !32
  store half %1902, ptr addrspace(1) %address.local1254, align 2, !dbg !32
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1231, !dbg !32
  store half %1901, ptr addrspace(1) %address.local1256, align 2, !dbg !32
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1234, !dbg !32
  store half %1900, ptr addrspace(1) %address.local1258, align 2, !dbg !32
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1237, !dbg !32
  store half %1898, ptr addrspace(1) %address.local1260, align 2, !dbg !32
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1240, !dbg !32
  store half %1897, ptr addrspace(1) %address.local1262, align 2, !dbg !32
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1243, !dbg !32
  store half %1896, ptr addrspace(1) %address.local1264, align 2, !dbg !32
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1246, !dbg !32
  store half %1895, ptr addrspace(1) %address.local1266, align 2, !dbg !32
  %normalized121 = fmul <4 x float> %row.reciprocal, %1719, !dbg !30
  %1914 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !31
  %1915 = extractelement <4 x half> %1914, i64 3, !dbg !31
  %1916 = extractelement <4 x half> %1914, i64 2, !dbg !31
  %1917 = extractelement <4 x half> %1914, i64 1, !dbg !31
  %1918 = extractelement <4 x half> %1914, i64 0, !dbg !31
  %normalized122 = fmul <4 x float> %row.reciprocal, %1594, !dbg !30
  %1919 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !31
  %1920 = extractelement <4 x half> %1919, i64 3, !dbg !31
  %1921 = extractelement <4 x half> %1919, i64 2, !dbg !31
  %1922 = extractelement <4 x half> %1919, i64 1, !dbg !31
  %1923 = extractelement <4 x half> %1919, i64 0, !dbg !31
  %normalized124 = fmul <4 x float> %row.reciprocal105, %1472, !dbg !30
  %1924 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !31
  %1925 = extractelement <4 x half> %1924, i64 3, !dbg !31
  %1926 = extractelement <4 x half> %1924, i64 2, !dbg !31
  %1927 = extractelement <4 x half> %1924, i64 1, !dbg !31
  %1928 = extractelement <4 x half> %1924, i64 0, !dbg !31
  %normalized125 = fmul <4 x float> %row.reciprocal105, %1347, !dbg !30
  %1929 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !31
  %1930 = extractelement <4 x half> %1929, i64 3, !dbg !31
  %1931 = extractelement <4 x half> %1929, i64 2, !dbg !31
  %1932 = extractelement <4 x half> %1929, i64 1, !dbg !31
  %1933 = extractelement <4 x half> %1929, i64 0, !dbg !31
  %address.local1281 = or disjoint i64 %address.local1113, 96, !dbg !32
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1281, !dbg !32
  store half %1933, ptr addrspace(1) %address.local1282, align 2, !dbg !32
  %address.local1284 = or disjoint i64 %address.local1113, 97, !dbg !32
  %address.local1285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1284, !dbg !32
  store half %1932, ptr addrspace(1) %address.local1285, align 2, !dbg !32
  %address.local1287 = or disjoint i64 %address.local1113, 98, !dbg !32
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1287, !dbg !32
  store half %1931, ptr addrspace(1) %address.local1288, align 2, !dbg !32
  %address.local1290 = or disjoint i64 %address.local1113, 99, !dbg !32
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1290, !dbg !32
  store half %1930, ptr addrspace(1) %address.local1291, align 2, !dbg !32
  %address.local1293 = or disjoint i64 %address.local1113, 112, !dbg !32
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1293, !dbg !32
  store half %1928, ptr addrspace(1) %address.local1294, align 2, !dbg !32
  %address.local1296 = or disjoint i64 %address.local1113, 113, !dbg !32
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1296, !dbg !32
  store half %1927, ptr addrspace(1) %address.local1297, align 2, !dbg !32
  %address.local1299 = or disjoint i64 %address.local1113, 114, !dbg !32
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1299, !dbg !32
  store half %1926, ptr addrspace(1) %address.local1300, align 2, !dbg !32
  %address.local1302 = or i64 %1872, 115, !dbg !32
  %address.local1303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1302, !dbg !32
  store half %1925, ptr addrspace(1) %address.local1303, align 2, !dbg !32
  %address.local1308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1281, !dbg !32
  store half %1923, ptr addrspace(1) %address.local1308, align 2, !dbg !32
  %address.local1310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1284, !dbg !32
  store half %1922, ptr addrspace(1) %address.local1310, align 2, !dbg !32
  %address.local1312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1287, !dbg !32
  store half %1921, ptr addrspace(1) %address.local1312, align 2, !dbg !32
  %address.local1314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1290, !dbg !32
  store half %1920, ptr addrspace(1) %address.local1314, align 2, !dbg !32
  %address.local1316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1293, !dbg !32
  store half %1918, ptr addrspace(1) %address.local1316, align 2, !dbg !32
  %address.local1318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1296, !dbg !32
  store half %1917, ptr addrspace(1) %address.local1318, align 2, !dbg !32
  %address.local1320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1299, !dbg !32
  store half %1916, ptr addrspace(1) %address.local1320, align 2, !dbg !32
  %address.local1322 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1302, !dbg !32
  store half %1915, ptr addrspace(1) %address.local1322, align 2, !dbg !32
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
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0001-test_input_1x32x4096x128-128-128-32-online/grid-000003")
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
