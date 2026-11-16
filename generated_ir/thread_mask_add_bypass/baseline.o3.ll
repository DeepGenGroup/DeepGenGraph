; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_mask_add_bypass/baseline.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest/causal.input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %5 = shl nuw nsw i32 %kdma.tid, 3, !dbg !7
  %kdma.intra = and i32 %5, 8, !dbg !7
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !7
  %6 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %7 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %6), !dbg !7
  %8 = add nuw nsw i32 %7, 64, !dbg !7
  %9 = xor i32 %7, 16, !dbg !7
  %10 = and i32 %8, 192, !dbg !7
  %11 = icmp samesign ult i32 %9, %10, !dbg !7
  %12 = select i1 %11, i32 %9, i32 %7, !dbg !7
  %13 = shl nuw nsw i32 %12, 2, !dbg !7
  %14 = xor i32 %7, 32, !dbg !7
  %15 = icmp samesign ult i32 %14, %10, !dbg !7
  %16 = select i1 %15, i32 %14, i32 %7, !dbg !7
  %17 = shl nuw nsw i32 %16, 2, !dbg !7
  %18 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !8
  %19 = sext i32 %18 to i64, !dbg !8
  %20 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !8
  %21 = sext i32 %20 to i64, !dbg !8
  %22 = shl nsw i64 %21, 5, !dbg !8
  %23 = add nsw i64 %22, %19, !dbg !8
  %24 = lshr i64 %23, 5, !dbg !8
  %25 = and i64 %24, 576460752303423484, !dbg !8
  %26 = and i64 %19, 3, !dbg !8
  %27 = or disjoint i64 %25, %26, !dbg !8
  %28 = sub nsw i64 31, %27, !dbg !8
  %29 = lshr i64 %23, 2, !dbg !8
  %30 = and i64 %29, 31, !dbg !8
  %31 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !8
  %32 = sext i32 %31 to i64, !dbg !8
  %33 = icmp ugt i64 %23, 1023, !dbg !9
  %34 = add nsw i64 %27, -32, !dbg !9
  %35 = select i1 %33, i64 %34, i64 %28, !dbg !9
  %36 = sdiv i64 %35, 32, !dbg !9
  %37 = sext i1 %33 to i64, !dbg !9
  %38 = xor i64 %36, %37, !dbg !9
  %39 = add nsw i64 %38, %29, !dbg !9
  %40 = and i64 %39, 31, !dbg !9
  %41 = shl i64 %28, 7, !dbg !9
  %42 = shl i64 %38, 12, !dbg !9
  %43 = sub i64 %41, %42, !dbg !9
  %44 = shl nsw i64 %43, 7, !dbg !9
  %45 = zext nneg i32 %kdma.tid to i64, !dbg !9
  %.idx = shl nuw nsw i64 %40, 20, !dbg !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !9, !range !10, !frisk.register_lifetimes !5
  %46 = shl nuw nsw i32 %address.thread, 3, !dbg !9
  %47 = and i32 %46, 1536, !dbg !9
  %address.local466 = zext nneg i32 %47 to i64, !dbg !9
  %address.local467 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local466, !dbg !9
  %address.local470 = lshr i32 %address.thread, 4, !dbg !9
  %address.local471 = and i32 %address.local470, 8, !dbg !9
  %address.local472 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local473 = and i32 %address.local472, 16256, !dbg !9
  %address.local474 = or disjoint i32 %address.local471, %address.local473, !dbg !9
  %48 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1444 = shl i64 %43, 8, !dbg !9
  %49 = add i64 %.idx, %.idx1444, !dbg !9
  %50 = shl nuw nsw i32 %address.local474, 1, !dbg !9
  %51 = trunc i64 %49 to i32, !dbg !9
  %dma.buffer.offset1442 = or disjoint i32 %50, %51, !dbg !9
  %52 = insertelement <2 x i64> poison, i64 %48, i64 0, !dbg !9
  %53 = bitcast <2 x i64> %52 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1443 = shufflevector <4 x i32> %53, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local479 = or disjoint i64 %address.local466, 2048, !dbg !9
  %address.local480 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local479, !dbg !9
  %address.local488 = or disjoint i32 %address.local474, 16, !dbg !9
  %54 = shl nuw nsw i32 %address.local488, 1, !dbg !9
  %dma.buffer.offset1440 = or disjoint i32 %54, %51, !dbg !9
  %55 = shl nuw nsw i64 %40, 19, !dbg !9
  %56 = add i64 %55, %44, !dbg !9
  %address.local491 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local466, !dbg !9
  %57 = trunc i64 %56 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local474, %57, !dbg !9
  %58 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1438 = or disjoint i32 %58, 64, !dbg !9
  %address.local495 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local479, !dbg !9
  %59 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local467, i64 0, !dbg !9
  %60 = insertelement <4 x ptr addrspace(3)> %59, ptr addrspace(3) %address.local480, i64 1, !dbg !9
  %61 = insertelement <4 x ptr addrspace(3)> %60, ptr addrspace(3) %address.local491, i64 2, !dbg !9
  %62 = insertelement <4 x ptr addrspace(3)> %61, ptr addrspace(3) %address.local495, i64 3, !dbg !9
  %63 = ptrtoint <4 x ptr addrspace(3)> %62 to <4 x i64>, !dbg !9
  %64 = trunc <4 x i64> %63 to <4 x i32>, !dbg !9
  %65 = extractelement <4 x i32> %64, i64 0, !dbg !9
  %66 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %65), !dbg !9
  %67 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1442, <4 x i32> %dma.buffer.rsrc1443) #9, !dbg !9
  %68 = extractelement <4 x i32> %64, i64 1, !dbg !9
  %69 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %68), !dbg !9
  %70 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset1440, <4 x i32> %dma.buffer.rsrc1443) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %71 = extractelement <4 x i32> %64, i64 2, !dbg !9
  %72 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %71), !dbg !9
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %dma.buffer.offset1438, <4 x i32> %dma.buffer.rsrc1443) #9, !dbg !9
  %74 = extractelement <4 x i32> %64, i64 3, !dbg !9
  %75 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %74), !dbg !9
  %.tr1448 = or disjoint i32 %address.local488, %57, !dbg !9
  %76 = shl i32 %.tr1448, 1, !dbg !9
  %dma.buffer.offset1436 = or disjoint i32 %76, 64, !dbg !9
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %dma.buffer.offset1436, <4 x i32> %dma.buffer.rsrc1443) #9, !dbg !9
  %address.local499 = lshr i32 %address.thread, 2, !dbg !11
  %address.local500 = and i32 %address.local499, 4, !dbg !11
  %address.local501 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local502 = and i32 %address.local501, 384, !dbg !11
  %address.local503 = or disjoint i32 %address.local500, %address.local502, !dbg !11
  %address.local505 = and i32 %46, 120, !dbg !11
  %address.local506 = or disjoint i32 %address.local503, %address.local505, !dbg !11
  %address.local507 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local508 = and i32 %address.local507, 1024, !dbg !11
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !11
  %address.local510 = zext nneg i32 %address.local509 to i64, !dbg !11
  %address.local511 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local510, !dbg !11
  %78 = load <4 x half>, ptr addrspace(3) %address.local511, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local511, i64 4096, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %80 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1265 = getelementptr i8, ptr addrspace(3) %address.local511, i64 1024, !dbg !11
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1265, align 8, !dbg !11
  %82 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1266 = getelementptr i8, ptr addrspace(3) %address.local511, i64 5120, !dbg !11
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1266, align 8, !dbg !11
  %84 = fmul <4 x half> %81, splat (half 1.275630e-01), !dbg !11
  %85 = fmul <4 x half> %83, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1434 = or disjoint i32 %58, 128, !dbg !9
  %86 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1434, <4 x i32> %dma.buffer.rsrc1443) #9, !dbg !9
  %dma.buffer.offset1432 = or disjoint i32 %76, 128, !dbg !9
  %87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset1432, <4 x i32> %dma.buffer.rsrc1443) #9, !dbg !9
  %lds.const.gep1267 = getelementptr i8, ptr addrspace(3) %address.local511, i64 8192, !dbg !11
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1267, align 8, !dbg !11
  %lds.const.gep1268 = getelementptr i8, ptr addrspace(3) %address.local511, i64 12288, !dbg !11
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !11
  %lds.const.gep1269 = getelementptr i8, ptr addrspace(3) %address.local511, i64 9216, !dbg !11
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !11
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1270 = getelementptr i8, ptr addrspace(3) %address.local511, i64 13312, !dbg !11
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !11
  %93 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !11
  %94 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !11
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1430 = or disjoint i32 %58, 192, !dbg !9
  %96 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %dma.buffer.offset1430, <4 x i32> %dma.buffer.rsrc1443) #9, !dbg !9
  %dma.buffer.offset1428 = or disjoint i32 %76, 192, !dbg !9
  %97 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %dma.buffer.offset1428, <4 x i32> %dma.buffer.rsrc1443) #9, !dbg !9
  %98 = load <4 x half>, ptr addrspace(3) %address.local511, align 8, !dbg !11
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1265, align 8, !dbg !11
  %101 = fmul <4 x half> %98, splat (half 1.275630e-01), !dbg !11
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep1266, align 8, !dbg !11
  %103 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !11
  %104 = fmul <4 x half> %100, splat (half 1.275630e-01), !dbg !11
  %105 = fmul <4 x half> %102, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %106 = load <4 x half>, ptr addrspace(3) %lds.const.gep1267, align 8, !dbg !11
  %107 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !11
  %108 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !11
  %109 = fmul <4 x half> %106, splat (half 1.275630e-01), !dbg !11
  %110 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !11
  %111 = fmul <4 x half> %107, splat (half 1.275630e-01), !dbg !11
  %112 = fmul <4 x half> %108, splat (half 1.275630e-01), !dbg !11
  %113 = fmul <4 x half> %110, splat (half 1.275630e-01), !dbg !11
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %114 = sub nsw i64 32, %27, !dbg !13
  %115 = shl nuw nsw i64 %30, 7, !dbg !14
  %116 = shl nsw i64 %32, 12, !dbg !14
  %117 = or disjoint i64 %115, %116, !dbg !14
  %118 = shl nuw nsw i64 %45, 2, !dbg !15
  %119 = and i64 %118, 3084, !dbg !15
  %120 = and i32 %kdma.tid, 1, !dbg !15
  %narrow = add nuw nsw i32 %120, 1, !dbg !15
  %121 = zext nneg i32 %narrow to i64, !dbg !15
  %122 = or disjoint i64 %119, %121, !dbg !15
  %123 = lshr i64 %122, 1, !dbg !15
  %124 = and i64 %45, 48, !dbg !16
  %125 = icmp ult i64 %23, 1024, !dbg !12
  br i1 %125, label %.lr.ph, label %._crit_edge, !dbg !12

.lr.ph:                                           ; preds = %4
  %126 = and i64 %45, 783, !dbg !16
  %127 = lshr i64 %45, 2, !dbg !16
  %128 = and i64 %127, 48, !dbg !16
  %129 = or disjoint i64 %126, %128, !dbg !16
  %130 = add nuw nsw i64 %129, %41, !dbg !16
  %131 = trunc i64 %130 to i32, !dbg !16
  %132 = add i32 %131, 65, !dbg !16
  %133 = shl nsw i64 %32, 17, !dbg !14
  %134 = shl nuw nsw i64 %30, 12, !dbg !14
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !7
  %135 = and i32 %kdma.lane, 31, !dbg !7
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !7
  %kdma.owner = or disjoint i32 %135, %kdma.wave, !dbg !7
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !7
  %136 = ptrtoint ptr addrspace(1) %1 to i64
  %137 = insertelement <2 x i64> poison, i64 %136, i64 0
  %138 = bitcast <2 x i64> %137 to <4 x i32>
  %dma.buffer.rsrc1427 = shufflevector <4 x i32> %138, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %kdma.source = zext nneg i32 %kdma.owner to i64
  %kdma.source1594 = lshr i64 %kdma.source, 4
  %kdma.source1595 = shl nuw nsw i64 %kdma.source1594, 9
  %kdma.source1596 = shl nuw nsw i64 %kdma.source, 7
  %kdma.source1597 = add nuw nsw i64 %kdma.source1595, %kdma.source1596
  %kdma.source1598 = lshr i64 %kdma.source, 5
  %kdma.source1599 = lshr i64 %kdma.source, 2
  %kdma.source1600 = and i64 %kdma.source1599, 1
  %kdma.source1601 = and i64 %kdma.source1600, %kdma.source1594
  %kdma.source1602 = or disjoint i64 %kdma.source1601, %kdma.source1598
  %139 = shl nuw nsw i64 %kdma.source1602, 10
  %kdma.source1604 = sub nsw i64 %kdma.source1597, %139
  %kdma.source1605 = and i64 %kdma.source1604, 16256
  %kdma.source1606 = lshr i32 %kdma.tid, 4
  %kdma.source1607 = and i32 %kdma.source1606, 8
  %kdma.source1608 = zext nneg i32 %kdma.source1607 to i64
  %kdma.source1609 = or disjoint i64 %kdma.source1605, %kdma.source1608
  %kdma.source1613 = zext nneg i32 %kdma.owner.hi to i64
  %kdma.source1614 = lshr i64 %kdma.source1613, 4
  %kdma.source1615 = shl nuw nsw i64 %kdma.source1614, 9
  %kdma.source1616 = shl nuw nsw i64 %kdma.source1613, 7
  %kdma.source1617 = add nuw nsw i64 %kdma.source1615, %kdma.source1616
  %kdma.source1618 = lshr i64 %kdma.source1613, 5
  %kdma.source1619 = lshr i64 %kdma.source1613, 2
  %kdma.source1620 = and i64 %kdma.source1619, 1
  %kdma.source1621 = and i64 %kdma.source1620, %kdma.source1614
  %kdma.source1622 = add nuw nsw i64 %kdma.source1621, %kdma.source1618
  %140 = shl nuw nsw i64 %kdma.source1622, 10
  %kdma.source1624 = sub nsw i64 %kdma.source1617, %140
  %kdma.source1625 = and i64 %kdma.source1624, 16256
  %kdma.source1629 = or disjoint i64 %kdma.source1625, %kdma.source1608
  %kdma.lds1634 = add i32 %66, 512
  %kdma.source1636 = or disjoint i64 %kdma.source, 256
  %kdma.source1637 = lshr i64 %kdma.source1636, 4
  %kdma.source1638 = shl nuw nsw i64 %kdma.source1637, 9
  %kdma.source1640 = add nuw nsw i64 %kdma.source1638, %kdma.source1596
  %kdma.source1641 = lshr i64 %kdma.source1636, 5
  %kdma.source1644 = and i64 %kdma.source1637, %kdma.source1600
  %kdma.source1645 = or disjoint i64 %kdma.source1644, %kdma.source1641
  %141 = shl nuw nsw i64 %kdma.source1645, 10
  %kdma.source1647 = sub nsw i64 %kdma.source1640, %141
  %kdma.source1648 = and i64 %kdma.source1647, 16256
  %kdma.source1651 = or disjoint i32 %kdma.source1607, 16
  %kdma.source1652 = zext nneg i32 %kdma.source1651 to i64
  %kdma.source1653 = or disjoint i64 %kdma.source1648, %kdma.source1652
  %kdma.source1660 = or disjoint i64 %kdma.source1613, 256
  %kdma.source1661 = lshr i64 %kdma.source1660, 4
  %kdma.source1662 = shl nuw nsw i64 %kdma.source1661, 9
  %kdma.source1664 = add nuw nsw i64 %kdma.source1662, %kdma.source1616
  %kdma.source1665 = lshr i64 %kdma.source1660, 5
  %kdma.source1668 = and i64 %kdma.source1661, %kdma.source1620
  %kdma.source1669 = add nuw nsw i64 %kdma.source1668, %kdma.source1665
  %142 = shl nuw nsw i64 %kdma.source1669, 10
  %kdma.source1671 = sub nsw i64 %kdma.source1664, %142
  %kdma.source1672 = and i64 %kdma.source1671, 16256
  %kdma.source1677 = or disjoint i64 %kdma.source1672, %kdma.source1652
  %kdma.lds1682 = add i32 %69, 512
  %143 = or disjoint i64 %134, %133
  %kdma.source1703 = or disjoint i32 %kdma.intra, 64
  %kdma.lds1728 = add i32 %72, 512
  %kdma.lds1778 = add i32 %75, 512
  %kdma.source1799 = or disjoint i32 %kdma.intra, 128
  %kdma.source1895 = or disjoint i32 %kdma.intra, 192
  %144 = ptrtoint ptr addrspace(1) %2 to i64
  %145 = insertelement <2 x i64> poison, i64 %144, i64 0
  %146 = bitcast <2 x i64> %145 to <4 x i32>
  %dma.buffer.rsrc1411 = shufflevector <4 x i32> %146, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %vdb.dma.base = add i32 %66, 8192
  %vdb.dma.base1544 = add i32 %69, 8192
  %147 = insertelement <2 x i64> poison, i64 %143, i64 1, !dbg !14
  %148 = insertelement <2 x i64> poison, i64 %117, i64 0, !dbg !14
  %149 = insertelement <32 x i32> poison, i32 %132, i64 0, !dbg !16
  %150 = shufflevector <32 x i32> %149, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  %151 = insertelement <32 x i32> poison, i32 %131, i64 0, !dbg !16
  %invariant.op = sub i32 72, %kdma.intra, !dbg !12
  %invariant.op56 = sub i32 72, %kdma.intra, !dbg !12
  %invariant.op57 = sub i32 136, %kdma.intra, !dbg !12
  %invariant.op58 = sub i32 136, %kdma.intra, !dbg !12
  %invariant.op59 = sub i32 200, %kdma.intra, !dbg !12
  %invariant.op60 = sub i32 200, %kdma.intra, !dbg !12
  %152 = insertelement <32 x i32> poison, i32 %131, i64 8
  br label %153, !dbg !12

153:                                              ; preds = %.lr.ph, %153
  %154 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1285, %153 ]
  %155 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1281, %153 ]
  %156 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1277, %153 ]
  %157 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1273, %153 ]
  %158 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %1037, %153 ]
  %159 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1271, %153 ]
  %160 = phi i64 [ 0, %.lr.ph ], [ %1286, %153 ]
  %161 = lshr i64 %160, 12, !dbg !14
  %162 = add nuw nsw i64 %161, %29, !dbg !14
  %163 = shl nuw nsw i64 %160, 7, !dbg !14
  %164 = shl i64 %162, 19, !dbg !14
  %165 = and i64 %164, 16252928, !dbg !14
  %address.thread562 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local563 = zext nneg i32 %address.thread562 to i64, !dbg !9
  %166 = insertelement <2 x i64> %148, i64 %160, i64 1, !dbg !14
  %address.local606 = shl nuw nsw i64 %address.local563, 1, !dbg !15
  %address.local607 = and i64 %address.local606, 24, !dbg !15
  %address.local609 = shl i64 %address.local563, 62, !dbg !15
  %address.local610 = ashr exact i64 %address.local609, 57, !dbg !15
  %address.local611 = and i64 %address.local610, -64, !dbg !15
  %address.local616 = and i64 %address.local563, 63, !dbg !11
  %address.local617 = lshr i64 %address.local616, 5, !dbg !11
  %address.local618 = mul nuw nsw i64 %address.local617, 1016, !dbg !11
  %.idx1479 = mul nuw nsw i64 %address.local563, 320, !dbg !15
  %167 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1479, !dbg !15
  %.idx2097 = shl nuw nsw i64 %address.local607, 1, !dbg !15
  %address.local620.idx = mul nuw nsw i64 %address.local617, 2032, !dbg !15
  %address.local631 = shl nuw nsw i32 %address.thread562, 1, !dbg !15
  %address.local632 = and i32 %address.local631, 24, !dbg !15
  %address.local633 = shl nuw nsw i32 %address.thread562, 8, !dbg !15
  %address.local634 = and i32 %address.local633, 768, !dbg !15
  %address.local635 = or disjoint i32 %address.local632, %address.local634, !dbg !15
  %address.local636 = shl nuw nsw i32 %address.thread562, 5, !dbg !15
  %address.local637 = and i32 %address.local636, 96, !dbg !15
  %address.local638 = or disjoint i32 %address.local635, %address.local637, !dbg !15
  %address.local639 = zext nneg i32 %address.local638 to i64, !dbg !15
  %address.local640 = lshr i32 %address.thread562, 1, !dbg !15
  %address.local641 = and i32 %address.local640, 1, !dbg !15
  %address.local642 = zext nneg i32 %address.local641 to i64, !dbg !15
  %168 = add nuw nsw i64 %address.local620.idx, 64, !dbg !15
  %169 = insertelement <2 x i64> poison, i64 %address.local563, i64 0, !dbg !14
  %170 = insertelement <2 x i64> %169, i64 %160, i64 1, !dbg !14
  %171 = lshr <2 x i64> %170, <i64 2, i64 5>, !dbg !14
  %172 = extractelement <2 x i64> %171, i64 1, !dbg !14
  %173 = and i64 %172, 127, !dbg !14
  %.idx14571460 = or disjoint i64 %165, %173, !dbg !14
  %174 = shl nuw nsw <2 x i64> %170, <i64 3, i64 14>, !dbg !9
  %175 = shl <2 x i64> %171, <i64 5, i64 19>, !dbg !17
  %176 = sub <2 x i64> %174, %175, !dbg !17
  %177 = extractelement <2 x i64> %176, i64 1, !dbg !14
  %178 = add i64 %.idx14571460, %177, !dbg !14
  %kdma.source1610 = add i64 %178, %kdma.source1609, !dbg !14
  %kdma.source1611 = trunc i64 %kdma.source1610 to i32, !dbg !14
  %kdma.source1612 = shl i32 %kdma.source1611, 1, !dbg !14
  %kdma.offset = add i32 %kdma.source1612, %kdma.intra, !dbg !14
  %179 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1630 = add i64 %178, %kdma.source1629, !dbg !14
  %kdma.source1631 = trunc i64 %kdma.source1630 to i32, !dbg !14
  %kdma.source1632 = shl i32 %kdma.source1631, 1, !dbg !14
  %kdma.offset1633 = add i32 %kdma.source1632, %kdma.intra.hi, !dbg !14
  %180 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1654 = add i64 %178, %kdma.source1653, !dbg !14
  %kdma.source1655 = trunc i64 %kdma.source1654 to i32, !dbg !14
  %kdma.source1656 = shl i32 %kdma.source1655, 1, !dbg !14
  %kdma.offset1657 = add i32 %kdma.source1656, %kdma.intra, !dbg !14
  %181 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %kdma.offset1657, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1678 = add i64 %178, %kdma.source1677, !dbg !14
  %kdma.source1679 = trunc i64 %kdma.source1678 to i32, !dbg !14
  %kdma.source1680 = shl i32 %kdma.source1679, 1, !dbg !14
  %kdma.offset1681 = add i32 %kdma.source1680, %kdma.intra.hi, !dbg !14
  %182 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1682, i32 %kdma.offset1681, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx14671471 = add i64 %177, %172, !dbg !14
  %183 = add i64 %.idx14671471, %117, !dbg !14
  %184 = shufflevector <2 x i64> %147, <2 x i64> %171, <2 x i32> <i32 3, i32 1>, !dbg !14
  %185 = add nsw <2 x i64> %184, %166, !dbg !14
  %186 = extractelement <2 x i64> %185, i64 1, !dbg !14
  %187 = extractelement <2 x i64> %185, i64 0, !dbg !14
  %188 = add nuw nsw i64 %186, 1024, !dbg !14
  %189 = add nsw <2 x i64> %185, <i64 32, i64 1024>, !dbg !14
  %.lobit41 = ashr i64 %188, 63, !dbg !14
  %190 = icmp slt <2 x i64> %185, <i64 -32, i64 -1024>, !dbg !14
  %191 = icmp slt i64 %187, -32, !dbg !14
  %192 = sub nsw <2 x i64> <i64 -33, i64 -1025>, %185, !dbg !14
  %193 = select <2 x i1> %190, <2 x i64> %192, <2 x i64> %189, !dbg !14
  %194 = sdiv <2 x i64> %193, <i64 128, i64 4096>, !dbg !14
  %195 = sext i1 %191 to i64, !dbg !14
  %196 = insertelement <2 x i64> poison, i64 %195, i64 0, !dbg !14
  %197 = insertelement <2 x i64> %196, i64 %.lobit41, i64 1, !dbg !14
  %198 = xor <2 x i64> %194, %197, !dbg !14
  %199 = shl <2 x i64> %198, <i64 7, i64 19>, !dbg !14
  %200 = extractelement <2 x i64> %199, i64 0, !dbg !14
  %201 = sub i64 %183, %200, !dbg !14
  %202 = extractelement <2 x i64> %199, i64 1, !dbg !14
  %203 = and i64 %202, 16252928, !dbg !14
  %204 = add i64 %201, %203, !dbg !14
  %kdma.source1700 = add i64 %204, %kdma.source1609, !dbg !14
  %kdma.source1701 = trunc i64 %kdma.source1700 to i32, !dbg !14
  %kdma.source1702 = shl i32 %kdma.source1701, 1, !dbg !14
  %kdma.offset1704 = add i32 %kdma.source1703, %kdma.source1702, !dbg !14
  %205 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %kdma.offset1704, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1723 = add i64 %204, %kdma.source1629, !dbg !14
  %kdma.source1724 = trunc i64 %kdma.source1723 to i32, !dbg !14
  %kdma.source1725 = shl i32 %kdma.source1724, 1, !dbg !14
  %kdma.offset1727.reass.reass = add i32 %kdma.source1725, %invariant.op
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1728, i32 %kdma.offset1727.reass.reass, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1748 = add i64 %204, %kdma.source1653, !dbg !14
  %kdma.source1749 = trunc i64 %kdma.source1748 to i32, !dbg !14
  %kdma.source1750 = shl i32 %kdma.source1749, 1, !dbg !14
  %kdma.offset1752 = add i32 %kdma.source1703, %kdma.source1750, !dbg !14
  %207 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %kdma.offset1752, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1773 = add i64 %204, %kdma.source1677, !dbg !14
  %kdma.source1774 = trunc i64 %kdma.source1773 to i32, !dbg !14
  %kdma.source1775 = shl i32 %kdma.source1774, 1, !dbg !14
  %kdma.offset1777.reass.reass = add i32 %kdma.source1775, %invariant.op56
  %208 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1778, i32 %kdma.offset1777.reass.reass, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %209 = extractelement <2 x i64> %171, i64 0, !dbg !18
  %address.local614 = and i64 %209, 12, !dbg !11
  %.idx1480 = mul nsw i64 %209, -1280, !dbg !15
  %210 = getelementptr i8, ptr addrspace(3) %167, i64 %.idx1480, !dbg !15
  %211 = getelementptr [2 x i8], ptr addrspace(3) %210, i64 %address.local607, !dbg !15
  %212 = getelementptr [2 x i8], ptr addrspace(3) %211, i64 %address.local611, !dbg !15
  %213 = getelementptr [2 x i8], ptr addrspace(3) %212, i64 %address.local614, !dbg !15
  %address.local620 = getelementptr [2 x i8], ptr addrspace(3) %213, i64 %address.local618, !dbg !15
  %214 = add nsw i64 %.idx1480, %.idx1479, !dbg !15
  %215 = or disjoint i64 %214, %.idx2097, !dbg !15
  %216 = or disjoint i64 %address.local611, %address.local614, !dbg !15
  %217 = shl nsw i64 %216, 1, !dbg !15
  %218 = add nsw i64 %215, %address.local620.idx, !dbg !15
  %219 = add nsw i64 %218, %217, !dbg !15
  %220 = lshr i64 %219, 6, !dbg !15
  %221 = and i64 %220, 8, !dbg !15
  %222 = xor i64 %221, %219, !dbg !15
  %223 = sub i64 %222, %219, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local620, i64 %223, !dbg !15
  %224 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %226 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %123) #8, !dbg !15, !frisk.register_lifetimes !5
  %address.local643 = add nsw i64 %address.base, %address.local642, !dbg !15
  %address.local644 = shl i64 %address.local643, 6, !dbg !15
  %address.local645 = sub i64 %address.local639, %address.local644, !dbg !15
  %227 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local645, !dbg !15
  %228 = getelementptr [2 x i8], ptr addrspace(3) %227, i64 %address.local614, !dbg !15
  %229 = getelementptr [2 x i8], ptr addrspace(3) %228, i64 %address.local618, !dbg !15
  %address.local649 = getelementptr i8, ptr addrspace(3) %229, i64 64, !dbg !15
  %.idx21002103 = add i64 %address.local645, %address.local614, !dbg !15
  %230 = shl i64 %.idx21002103, 1, !dbg !15
  %231 = add i64 %168, %230, !dbg !15
  %232 = lshr i64 %231, 6, !dbg !15
  %233 = and i64 %232, 8, !dbg !15
  %234 = xor i64 %233, %231, !dbg !15
  %235 = sub i64 %234, %231, !dbg !15
  %kdma.swizzled1971 = getelementptr i8, ptr addrspace(3) %address.local649, i64 %235, !dbg !15
  %236 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1971, align 8, !dbg !15, !frisk.k_lds_pair !19
  %kdma.gep1975 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 128, !dbg !15
  %237 = load <4 x half>, ptr addrspace(3) %kdma.gep1975, align 8, !dbg !15, !frisk.k_lds_pair !19
  %kdma.gep1972 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 4096, !dbg !15
  %238 = load <4 x half>, ptr addrspace(3) %kdma.gep1972, align 8, !dbg !15, !frisk.k_lds_pair !20
  %kdma.gep1976 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 4224, !dbg !15
  %239 = load <4 x half>, ptr addrspace(3) %kdma.gep1976, align 8, !dbg !15, !frisk.k_lds_pair !20
  %kdma.gep1973 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %240 = load <4 x half>, ptr addrspace(3) %kdma.gep1973, align 8, !dbg !15, !frisk.k_lds_pair !21
  %kdma.gep1977 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %241 = load <4 x half>, ptr addrspace(3) %kdma.gep1977, align 8, !dbg !15, !frisk.k_lds_pair !21
  %kdma.gep1974 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %242 = load <4 x half>, ptr addrspace(3) %kdma.gep1974, align 8, !dbg !15, !frisk.k_lds_pair !22
  %kdma.gep1978 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %243 = load <4 x half>, ptr addrspace(3) %kdma.gep1978, align 8, !dbg !15, !frisk.k_lds_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %226, <4 x float> %225) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1979 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 1024, !dbg !15
  %245 = load <4 x half>, ptr addrspace(3) %kdma.gep1979, align 8, !dbg !15, !frisk.k_lds_pair !23
  %kdma.gep1983 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 1152, !dbg !15
  %246 = load <4 x half>, ptr addrspace(3) %kdma.gep1983, align 8, !dbg !15, !frisk.k_lds_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1980 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 5120, !dbg !15
  %248 = load <4 x half>, ptr addrspace(3) %kdma.gep1980, align 8, !dbg !15, !frisk.k_lds_pair !24
  %kdma.gep1984 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 5248, !dbg !15
  %249 = load <4 x half>, ptr addrspace(3) %kdma.gep1984, align 8, !dbg !15, !frisk.k_lds_pair !24
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %238, <4 x float> %247) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1981 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %251 = load <4 x half>, ptr addrspace(3) %kdma.gep1981, align 8, !dbg !15, !frisk.k_lds_pair !25
  %kdma.gep1982 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %252 = load <4 x half>, ptr addrspace(3) %kdma.gep1982, align 8, !dbg !15, !frisk.k_lds_pair !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %240, <4 x float> zeroinitializer) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %242, <4 x float> %253) #8, !dbg !15
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %237, <4 x float> zeroinitializer) #8, !dbg !15
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %239, <4 x float> %255) #8, !dbg !15
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %241, <4 x float> zeroinitializer) #8, !dbg !15
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %243, <4 x float> %257) #8, !dbg !15
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %245, <4 x float> zeroinitializer) #8, !dbg !15
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %248, <4 x float> %259) #8, !dbg !15
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %251, <4 x float> zeroinitializer) #8, !dbg !15
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %252, <4 x float> %261) #8, !dbg !15
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %246, <4 x float> zeroinitializer) #8, !dbg !15
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %249, <4 x float> %263) #8, !dbg !15
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !15
  %266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %226, <4 x float> %265) #8, !dbg !15
  %267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !15
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %238, <4 x float> %267) #8, !dbg !15
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %240, <4 x float> zeroinitializer) #8, !dbg !15
  %270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %242, <4 x float> %269) #8, !dbg !15
  %271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %237, <4 x float> zeroinitializer) #8, !dbg !15
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %239, <4 x float> %271) #8, !dbg !15
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %241, <4 x float> zeroinitializer) #8, !dbg !15
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %243, <4 x float> %273) #8, !dbg !15
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %245, <4 x float> zeroinitializer) #8, !dbg !15
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %248, <4 x float> %275) #8, !dbg !15
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %251, <4 x float> zeroinitializer) #8, !dbg !15
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %252, <4 x float> %277) #8, !dbg !15
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %246, <4 x float> zeroinitializer) #8, !dbg !15
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %249, <4 x float> %279) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %281 = add nuw nsw i64 %186, 2048, !dbg !14
  %282 = add nsw <2 x i64> %185, <i64 64, i64 2048>, !dbg !14
  %.lobit42 = ashr i64 %281, 63, !dbg !14
  %283 = icmp slt <2 x i64> %185, <i64 -64, i64 -2048>, !dbg !14
  %284 = icmp slt i64 %187, -64, !dbg !14
  %285 = sub nsw <2 x i64> <i64 -65, i64 -2049>, %185, !dbg !14
  %286 = select <2 x i1> %283, <2 x i64> %285, <2 x i64> %282, !dbg !14
  %287 = sdiv <2 x i64> %286, <i64 128, i64 4096>, !dbg !14
  %288 = sext i1 %284 to i64, !dbg !14
  %289 = insertelement <2 x i64> poison, i64 %288, i64 0, !dbg !14
  %290 = insertelement <2 x i64> %289, i64 %.lobit42, i64 1, !dbg !14
  %291 = xor <2 x i64> %287, %290, !dbg !14
  %292 = shl <2 x i64> %291, <i64 7, i64 19>, !dbg !14
  %293 = extractelement <2 x i64> %292, i64 0, !dbg !14
  %294 = sub i64 %183, %293, !dbg !14
  %295 = extractelement <2 x i64> %292, i64 1, !dbg !14
  %296 = and i64 %295, 16252928, !dbg !14
  %297 = add i64 %294, %296, !dbg !14
  %kdma.source1796 = add i64 %297, %kdma.source1609, !dbg !14
  %kdma.source1797 = trunc i64 %kdma.source1796 to i32, !dbg !14
  %kdma.source1798 = shl i32 %kdma.source1797, 1, !dbg !14
  %kdma.offset1800 = add i32 %kdma.source1799, %kdma.source1798, !dbg !14
  %298 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset1800, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1819 = add i64 %297, %kdma.source1629, !dbg !14
  %kdma.source1820 = trunc i64 %kdma.source1819 to i32, !dbg !14
  %kdma.source1821 = shl i32 %kdma.source1820, 1, !dbg !14
  %kdma.offset1823.reass.reass = add i32 %kdma.source1821, %invariant.op57
  %299 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1823.reass.reass, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1844 = add i64 %297, %kdma.source1653, !dbg !14
  %kdma.source1845 = trunc i64 %kdma.source1844 to i32, !dbg !14
  %kdma.source1846 = shl i32 %kdma.source1845, 1, !dbg !14
  %kdma.offset1848 = add i32 %kdma.source1799, %kdma.source1846, !dbg !14
  %300 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %kdma.offset1848, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1869 = add i64 %297, %kdma.source1677, !dbg !14
  %kdma.source1870 = trunc i64 %kdma.source1869 to i32, !dbg !14
  %kdma.source1871 = shl i32 %kdma.source1870, 1, !dbg !14
  %kdma.offset1873.reass.reass = add i32 %kdma.source1871, %invariant.op58
  %301 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1682, i32 %kdma.offset1873.reass.reass, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.gep2001 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep2001, align 8, !dbg !15, !frisk.k_lds_pair !26
  %kdma.gep2005 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep2005, align 8, !dbg !15, !frisk.k_lds_pair !26
  %kdma.gep2002 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep2002, align 8, !dbg !15, !frisk.k_lds_pair !27
  %kdma.gep2006 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep2006, align 8, !dbg !15, !frisk.k_lds_pair !27
  %kdma.gep2003 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 8192, !dbg !15
  %306 = load <4 x half>, ptr addrspace(3) %kdma.gep2003, align 8, !dbg !15, !frisk.k_lds_pair !28
  %kdma.gep2007 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 8320, !dbg !15
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep2007, align 8, !dbg !15, !frisk.k_lds_pair !28
  %kdma.gep2004 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 12288, !dbg !15
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep2004, align 8, !dbg !15, !frisk.k_lds_pair !29
  %kdma.gep2008 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 12416, !dbg !15
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep2008, align 8, !dbg !15, !frisk.k_lds_pair !29
  %kdma.gep2009 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep2009, align 8, !dbg !15, !frisk.k_lds_pair !30
  %kdma.gep2013 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep2013, align 8, !dbg !15, !frisk.k_lds_pair !30
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %302, <4 x float> %244) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep2010 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep2010, align 8, !dbg !15, !frisk.k_lds_pair !31
  %kdma.gep2014 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep2014, align 8, !dbg !15, !frisk.k_lds_pair !31
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %304, <4 x float> %312) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep2011 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 9216, !dbg !15
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep2011, align 8, !dbg !15, !frisk.k_lds_pair !32
  %kdma.gep2015 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 9344, !dbg !15
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep2015, align 8, !dbg !15, !frisk.k_lds_pair !32
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %306, <4 x float> %250) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep2012 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 13312, !dbg !15
  %319 = load <4 x half>, ptr addrspace(3) %kdma.gep2012, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep2016 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 13440, !dbg !15
  %320 = load <4 x half>, ptr addrspace(3) %kdma.gep2016, align 8, !dbg !15, !frisk.k_lds_pair !33
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %308, <4 x float> %318) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %303, <4 x float> %254) #8, !dbg !15
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %305, <4 x float> %322) #8, !dbg !15
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %307, <4 x float> %256) #8, !dbg !15
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %309, <4 x float> %324) #8, !dbg !15
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %310, <4 x float> %258) #8, !dbg !15
  %327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %313, <4 x float> %326) #8, !dbg !15
  %328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %316, <4 x float> %260) #8, !dbg !15
  %329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %319, <4 x float> %328) #8, !dbg !15
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %311, <4 x float> %262) #8, !dbg !15
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %314, <4 x float> %330) #8, !dbg !15
  %332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %317, <4 x float> %264) #8, !dbg !15
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %320, <4 x float> %332) #8, !dbg !15
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %302, <4 x float> %266) #8, !dbg !15
  %335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %304, <4 x float> %334) #8, !dbg !15
  %336 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %306, <4 x float> %268) #8, !dbg !15
  %337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %308, <4 x float> %336) #8, !dbg !15
  %338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %303, <4 x float> %270) #8, !dbg !15
  %339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %305, <4 x float> %338) #8, !dbg !15
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %307, <4 x float> %272) #8, !dbg !15
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %309, <4 x float> %340) #8, !dbg !15
  %342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %310, <4 x float> %274) #8, !dbg !15
  %343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %313, <4 x float> %342) #8, !dbg !15
  %344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %316, <4 x float> %276) #8, !dbg !15
  %345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %319, <4 x float> %344) #8, !dbg !15
  %346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %311, <4 x float> %278) #8, !dbg !15
  %347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %314, <4 x float> %346) #8, !dbg !15
  %348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %317, <4 x float> %280) #8, !dbg !15
  %349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %320, <4 x float> %348) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %350 = add nuw nsw i64 %186, 3072, !dbg !14
  %351 = add nsw <2 x i64> %185, <i64 96, i64 3072>, !dbg !14
  %.lobit43 = ashr i64 %350, 63, !dbg !14
  %352 = icmp slt <2 x i64> %185, <i64 -96, i64 -3072>, !dbg !14
  %353 = icmp slt i64 %187, -96, !dbg !14
  %354 = sub nsw <2 x i64> <i64 -97, i64 -3073>, %185, !dbg !14
  %355 = select <2 x i1> %352, <2 x i64> %354, <2 x i64> %351, !dbg !14
  %356 = sdiv <2 x i64> %355, <i64 128, i64 4096>, !dbg !14
  %357 = sext i1 %353 to i64, !dbg !14
  %358 = insertelement <2 x i64> poison, i64 %357, i64 0, !dbg !14
  %359 = insertelement <2 x i64> %358, i64 %.lobit43, i64 1, !dbg !14
  %360 = xor <2 x i64> %356, %359, !dbg !14
  %361 = shl <2 x i64> %360, <i64 7, i64 19>, !dbg !14
  %362 = extractelement <2 x i64> %361, i64 0, !dbg !14
  %363 = sub i64 %183, %362, !dbg !14
  %364 = extractelement <2 x i64> %361, i64 1, !dbg !14
  %365 = and i64 %364, 16252928, !dbg !14
  %366 = add i64 %363, %365, !dbg !14
  %kdma.source1892 = add i64 %366, %kdma.source1609, !dbg !14
  %kdma.source1893 = trunc i64 %kdma.source1892 to i32, !dbg !14
  %kdma.source1894 = shl i32 %kdma.source1893, 1, !dbg !14
  %kdma.offset1896 = add i32 %kdma.source1895, %kdma.source1894, !dbg !14
  %367 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %kdma.offset1896, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1915 = add i64 %366, %kdma.source1629, !dbg !14
  %kdma.source1916 = trunc i64 %kdma.source1915 to i32, !dbg !14
  %kdma.source1917 = shl i32 %kdma.source1916, 1, !dbg !14
  %kdma.offset1919.reass.reass = add i32 %kdma.source1917, %invariant.op59
  %368 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1728, i32 %kdma.offset1919.reass.reass, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1940 = add i64 %366, %kdma.source1653, !dbg !14
  %kdma.source1941 = trunc i64 %kdma.source1940 to i32, !dbg !14
  %kdma.source1942 = shl i32 %kdma.source1941, 1, !dbg !14
  %kdma.offset1944 = add i32 %kdma.source1895, %kdma.source1942, !dbg !14
  %369 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %kdma.offset1944, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %kdma.source1965 = add i64 %366, %kdma.source1677, !dbg !14
  %kdma.source1966 = trunc i64 %kdma.source1965 to i32, !dbg !14
  %kdma.source1967 = shl i32 %kdma.source1966, 1, !dbg !14
  %kdma.offset1969.reass.reass = add i32 %kdma.source1967, %invariant.op60
  %370 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1778, i32 %kdma.offset1969.reass.reass, <4 x i32> %dma.buffer.rsrc1427) #9, !dbg !14
  %371 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !34
  %372 = load <4 x half>, ptr addrspace(3) %kdma.gep1973, align 8, !dbg !15, !frisk.k_lds_pair !34
  %373 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !35
  %374 = load <4 x half>, ptr addrspace(3) %kdma.gep1974, align 8, !dbg !15, !frisk.k_lds_pair !35
  %375 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1971, align 8, !dbg !15, !frisk.k_lds_pair !36
  %376 = load <4 x half>, ptr addrspace(3) %kdma.gep1975, align 8, !dbg !15, !frisk.k_lds_pair !36
  %377 = load <4 x half>, ptr addrspace(3) %kdma.gep1972, align 8, !dbg !15, !frisk.k_lds_pair !37
  %378 = load <4 x half>, ptr addrspace(3) %kdma.gep1976, align 8, !dbg !15, !frisk.k_lds_pair !37
  %379 = load <4 x half>, ptr addrspace(3) %kdma.gep1977, align 8, !dbg !15, !frisk.k_lds_pair !38
  %380 = load <4 x half>, ptr addrspace(3) %kdma.gep1981, align 8, !dbg !15, !frisk.k_lds_pair !38
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %371, <4 x float> %315) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %382 = load <4 x half>, ptr addrspace(3) %kdma.gep1978, align 8, !dbg !15, !frisk.k_lds_pair !39
  %383 = load <4 x half>, ptr addrspace(3) %kdma.gep1982, align 8, !dbg !15, !frisk.k_lds_pair !39
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %373, <4 x float> %381) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %385 = load <4 x half>, ptr addrspace(3) %kdma.gep1979, align 8, !dbg !15, !frisk.k_lds_pair !40
  %386 = load <4 x half>, ptr addrspace(3) %kdma.gep1983, align 8, !dbg !15, !frisk.k_lds_pair !40
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %375, <4 x float> %321) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %388 = load <4 x half>, ptr addrspace(3) %kdma.gep1980, align 8, !dbg !15, !frisk.k_lds_pair !41
  %389 = load <4 x half>, ptr addrspace(3) %kdma.gep1984, align 8, !dbg !15, !frisk.k_lds_pair !41
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %377, <4 x float> %387) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %372, <4 x float> %323) #8, !dbg !15
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %374, <4 x float> %391) #8, !dbg !15
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %376, <4 x float> %325) #8, !dbg !15
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %378, <4 x float> %393) #8, !dbg !15
  %395 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %379, <4 x float> %327) #8, !dbg !15
  %396 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %382, <4 x float> %395) #8, !dbg !15
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %385, <4 x float> %329) #8, !dbg !15
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %388, <4 x float> %397) #8, !dbg !15
  %399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %380, <4 x float> %331) #8, !dbg !15
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %383, <4 x float> %399) #8, !dbg !15
  %401 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %386, <4 x float> %333) #8, !dbg !15
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %389, <4 x float> %401) #8, !dbg !15
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %371, <4 x float> %335) #8, !dbg !15
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %373, <4 x float> %403) #8, !dbg !15
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %375, <4 x float> %337) #8, !dbg !15
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %377, <4 x float> %405) #8, !dbg !15
  %407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %372, <4 x float> %339) #8, !dbg !15
  %408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %374, <4 x float> %407) #8, !dbg !15
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %376, <4 x float> %341) #8, !dbg !15
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %378, <4 x float> %409) #8, !dbg !15
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %379, <4 x float> %343) #8, !dbg !15
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %382, <4 x float> %411) #8, !dbg !15
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %385, <4 x float> %345) #8, !dbg !15
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %388, <4 x float> %413) #8, !dbg !15
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %380, <4 x float> %347) #8, !dbg !15
  %416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %383, <4 x float> %415) #8, !dbg !15
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %386, <4 x float> %349) #8, !dbg !15
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %389, <4 x float> %417) #8, !dbg !15
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %419 = add nuw nsw i64 %172, %29, !dbg !17
  %420 = insertelement <2 x i64> %170, i64 %419, i64 1, !dbg !17
  %421 = shl <2 x i64> %420, <i64 5, i64 19>, !dbg !17
  %422 = and <2 x i64> %421, <i64 8064, i64 16252928>, !dbg !17
  %423 = add <2 x i64> %176, %422, !dbg !17
  %424 = extractelement <2 x i64> %423, i64 0, !dbg !17
  %425 = extractelement <2 x i64> %423, i64 1, !dbg !17
  %426 = add i64 %424, %425, !dbg !17
  %.tr1508 = trunc i64 %426 to i32, !dbg !17
  %dma.buffer.offset1410 = shl i32 %.tr1508, 1, !dbg !17
  %427 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1410, <4 x i32> %dma.buffer.rsrc1411) #9, !dbg !17
  %address.local985 = shl nuw nsw i32 %address.thread562, 3, !dbg !17
  %address.local986 = and i32 %address.local985, 24, !dbg !17
  %address.local988 = and i32 %address.local636, 8064, !dbg !17
  %428 = trunc i64 %425 to i32, !dbg !17
  %address.local989 = or disjoint i32 %address.local988, %428, !dbg !17
  %address.local990 = or disjoint i32 %address.local989, %address.local986, !dbg !17
  %.tr1512 = shl i32 %address.local990, 1, !dbg !17
  %dma.buffer.offset1408 = or i32 %.tr1512, 16384, !dbg !17
  %429 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset1408, <4 x i32> %dma.buffer.rsrc1411) #9, !dbg !17
  %430 = load <4 x half>, ptr addrspace(3) %kdma.gep2001, align 8, !dbg !15, !frisk.k_tail_pair !19
  %431 = load <4 x half>, ptr addrspace(3) %kdma.gep2005, align 8, !dbg !15, !frisk.k_tail_pair !19
  %432 = load <4 x half>, ptr addrspace(3) %kdma.gep2002, align 8, !dbg !15, !frisk.k_tail_pair !20
  %433 = load <4 x half>, ptr addrspace(3) %kdma.gep2006, align 8, !dbg !15, !frisk.k_tail_pair !20
  %434 = load <4 x half>, ptr addrspace(3) %kdma.gep2003, align 8, !dbg !15, !frisk.k_tail_pair !21
  %435 = load <4 x half>, ptr addrspace(3) %kdma.gep2007, align 8, !dbg !15, !frisk.k_tail_pair !21
  %436 = load <4 x half>, ptr addrspace(3) %kdma.gep2004, align 8, !dbg !15, !frisk.k_tail_pair !22
  %437 = load <4 x half>, ptr addrspace(3) %kdma.gep2008, align 8, !dbg !15, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %430, <4 x float> %384) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %439 = or disjoint i64 %163, %124, !dbg !16
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %432, <4 x float> %438) #8, !dbg !15
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %434, <4 x float> %390) #8, !dbg !15
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %436, <4 x float> %441) #8, !dbg !15
  %443 = load <4 x half>, ptr addrspace(3) %kdma.gep2009, align 8, !dbg !15, !frisk.k_tail_pair !23
  %444 = load <4 x half>, ptr addrspace(3) %kdma.gep2013, align 8, !dbg !15, !frisk.k_tail_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %431, <4 x float> %392) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %446 = load <4 x half>, ptr addrspace(3) %kdma.gep2010, align 8, !dbg !15, !frisk.k_tail_pair !24
  %447 = load <4 x half>, ptr addrspace(3) %kdma.gep2014, align 8, !dbg !15, !frisk.k_tail_pair !24
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %433, <4 x float> %445) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %449 = load <4 x half>, ptr addrspace(3) %kdma.gep2011, align 8, !dbg !15, !frisk.k_tail_pair !25
  %450 = load <4 x half>, ptr addrspace(3) %kdma.gep2015, align 8, !dbg !15, !frisk.k_tail_pair !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %435, <4 x float> %394) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %452 = load <4 x half>, ptr addrspace(3) %kdma.gep2012, align 8, !dbg !15, !frisk.k_tail_pair !26
  %453 = load <4 x half>, ptr addrspace(3) %kdma.gep2016, align 8, !dbg !15, !frisk.k_tail_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %437, <4 x float> %451) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %455 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %443, <4 x float> %396) #8, !dbg !15
  %456 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %446, <4 x float> %455) #8, !dbg !15
  %457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %449, <4 x float> %398) #8, !dbg !15
  %458 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %452, <4 x float> %457) #8, !dbg !15
  %459 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %444, <4 x float> %400) #8, !dbg !15
  %460 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %447, <4 x float> %459) #8, !dbg !15
  %461 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %450, <4 x float> %402) #8, !dbg !15
  %462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %453, <4 x float> %461) #8, !dbg !15
  %463 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %430, <4 x float> %404) #8, !dbg !15
  %464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %432, <4 x float> %463) #8, !dbg !15
  %465 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %434, <4 x float> %406) #8, !dbg !15
  %466 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %436, <4 x float> %465) #8, !dbg !15
  %467 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %431, <4 x float> %408) #8, !dbg !15
  %468 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %433, <4 x float> %467) #8, !dbg !15
  %469 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %435, <4 x float> %410) #8, !dbg !15
  %470 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %437, <4 x float> %469) #8, !dbg !15
  %471 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %443, <4 x float> %412) #8, !dbg !15
  %472 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %446, <4 x float> %471) #8, !dbg !15
  %473 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %449, <4 x float> %414) #8, !dbg !15
  %474 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %452, <4 x float> %473) #8, !dbg !15
  %475 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %444, <4 x float> %416) #8, !dbg !15
  %476 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %447, <4 x float> %475) #8, !dbg !15
  %477 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %450, <4 x float> %418) #8, !dbg !15
  %478 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %453, <4 x float> %477) #8, !dbg !15
  %479 = trunc i64 %439 to i32, !dbg !16
  %480 = insertelement <32 x i32> poison, i32 %479, i64 0, !dbg !16
  %481 = shufflevector <32 x i32> %480, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  %482 = or disjoint <32 x i32> %481, <i32 0, i32 4, i32 8, i32 12, i32 64, i32 68, i32 72, i32 76, i32 1, i32 5, i32 9, i32 13, i32 65, i32 69, i32 73, i32 77, i32 2, i32 6, i32 10, i32 14, i32 66, i32 70, i32 74, i32 78, i32 3, i32 7, i32 11, i32 15, i32 67, i32 71, i32 75, i32 79>, !dbg !16
  %483 = icmp ugt <32 x i32> %150, %482, !dbg !16
  %484 = insertelement <4 x i32> poison, i32 %479, i64 0, !dbg !16
  %485 = shufflevector <4 x i32> %484, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !16
  %486 = or disjoint <4 x i32> %485, <i32 65, i32 69, i32 73, i32 77>, !dbg !16
  %487 = insertelement <2 x i32> poison, i32 %479, i64 0, !dbg !16
  %488 = shufflevector <2 x i32> %487, <2 x i32> poison, <2 x i32> zeroinitializer, !dbg !16
  %489 = or disjoint <2 x i32> %488, <i32 9, i32 13>, !dbg !16
  %490 = or disjoint i32 %479, 5, !dbg !16
  %491 = or disjoint <4 x i32> %485, <i32 64, i32 68, i32 72, i32 76>, !dbg !16
  %492 = or disjoint <2 x i32> %488, <i32 8, i32 12>, !dbg !16
  %493 = or disjoint i32 %479, 4, !dbg !16
  %494 = insertelement <32 x i32> %151, i32 %479, i64 8, !dbg !16
  %495 = shufflevector <32 x i32> %494, <32 x i32> poison, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %496 = insertelement <32 x i32> %152, i32 %479, i64 0, !dbg !16
  %497 = insertelement <32 x i32> %496, i32 %493, i64 1, !dbg !16
  %498 = insertelement <32 x i32> %497, i32 %490, i64 9, !dbg !16
  %499 = shufflevector <32 x i32> %498, <32 x i32> %482, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %500 = shufflevector <4 x i32> %491, <4 x i32> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %501 = shufflevector <32 x i32> %499, <32 x i32> %500, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 32, i32 33, i32 34, i32 35, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !16
  %502 = shufflevector <4 x i32> %486, <4 x i32> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %503 = shufflevector <32 x i32> %501, <32 x i32> %502, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 poison, i32 poison, i32 32, i32 33, i32 34, i32 35, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !16
  %504 = shufflevector <2 x i32> %492, <2 x i32> poison, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %505 = shufflevector <32 x i32> %503, <32 x i32> %504, <32 x i32> <i32 0, i32 1, i32 32, i32 33, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 poison, i32 poison, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !16
  %506 = shufflevector <2 x i32> %489, <2 x i32> poison, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %507 = shufflevector <32 x i32> %505, <32 x i32> %506, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 32, i32 33, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !16
  %508 = icmp ult <32 x i32> %495, %507, !dbg !16
  %509 = shufflevector <4 x float> %440, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %510 = shufflevector <32 x float> <float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float poison, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf>, <32 x float> %509, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !16
  %511 = shufflevector <4 x float> %440, <4 x float> %442, <16 x i32> <i32 2, i32 6, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 3, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %512 = shufflevector <4 x float> %448, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %513 = shufflevector <16 x float> %511, <16 x float> %512, <16 x i32> <i32 0, i32 1, i32 18, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %514 = shufflevector <4 x float> %454, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %515 = shufflevector <16 x float> %513, <16 x float> %514, <16 x i32> <i32 0, i32 1, i32 2, i32 18, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %516 = shufflevector <4 x float> %456, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %517 = shufflevector <16 x float> %515, <16 x float> %516, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 18, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %518 = shufflevector <4 x float> %458, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %519 = shufflevector <16 x float> %517, <16 x float> %518, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 18, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %520 = shufflevector <4 x float> %460, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %521 = shufflevector <16 x float> %519, <16 x float> %520, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 18, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %522 = shufflevector <4 x float> %462, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %523 = shufflevector <16 x float> %521, <16 x float> %522, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 18, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %524 = shufflevector <4 x float> %448, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %525 = shufflevector <16 x float> %523, <16 x float> %524, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 19, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %526 = shufflevector <4 x float> %454, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %527 = shufflevector <16 x float> %525, <16 x float> %526, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %528 = shufflevector <4 x float> %456, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %529 = shufflevector <16 x float> %527, <16 x float> %528, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 19, i32 poison, i32 poison, i32 poison>, !dbg !16
  %530 = shufflevector <4 x float> %458, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %531 = shufflevector <16 x float> %529, <16 x float> %530, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 19, i32 poison, i32 poison>, !dbg !16
  %532 = shufflevector <4 x float> %460, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %533 = shufflevector <16 x float> %531, <16 x float> %532, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 19, i32 poison>, !dbg !16
  %534 = shufflevector <4 x float> %462, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %535 = shufflevector <4 x float> %440, <4 x float> %442, <8 x i32> <i32 0, i32 4, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %536 = shufflevector <4 x float> %448, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %537 = shufflevector <8 x float> %535, <8 x float> %536, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %538 = shufflevector <4 x float> %454, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %539 = shufflevector <8 x float> %537, <8 x float> %538, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %540 = shufflevector <4 x float> %456, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %541 = shufflevector <8 x float> %539, <8 x float> %540, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>, !dbg !16
  %542 = shufflevector <4 x float> %458, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %543 = shufflevector <8 x float> %541, <8 x float> %542, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>, !dbg !16
  %544 = shufflevector <4 x float> %460, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %545 = shufflevector <8 x float> %543, <8 x float> %544, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>, !dbg !16
  %546 = shufflevector <4 x float> %462, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %547 = shufflevector <4 x float> %456, <4 x float> %458, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !16
  %548 = shufflevector <4 x float> %547, <4 x float> %460, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !16
  %549 = shufflevector <4 x float> %442, <4 x float> %448, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 5, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %550 = insertelement <32 x float> %549, float -inf, i64 8, !dbg !16
  %551 = shufflevector <4 x float> %454, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %552 = shufflevector <32 x float> %550, <32 x float> %551, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %553 = shufflevector <16 x float> %533, <16 x float> %534, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 19, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %554 = shufflevector <32 x float> %552, <32 x float> %553, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 poison, i32 poison, i32 poison, i32 poison, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47>, !dbg !16
  %555 = shufflevector <8 x float> %545, <8 x float> %546, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %556 = shufflevector <32 x float> %555, <32 x float> %554, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 poison, i32 poison, i32 poison, i32 poison, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %557 = shufflevector <4 x float> %548, <4 x float> %462, <32 x i32> <i32 0, i32 1, i32 2, i32 5, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %558 = shufflevector <32 x float> %556, <32 x float> %557, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !16
  %559 = select <32 x i1> %508, <32 x float> %510, <32 x float> %558, !dbg !16
  %560 = extractelement <32 x float> %559, i64 0, !dbg !7
  %561 = tail call float @llvm.maxnum.f32(float %560, float -inf), !dbg !7, !frisk.local_reduction !5
  %562 = extractelement <32 x float> %559, i64 1, !dbg !7
  %local.reduce200 = tail call float @llvm.maxnum.f32(float %561, float %562), !dbg !7, !frisk.local_reduction !5
  %563 = extractelement <32 x float> %559, i64 8, !dbg !7
  %564 = extractelement <32 x float> %559, i64 9, !dbg !7
  %local.reduce201 = tail call float @llvm.maxnum.f32(float %563, float %564), !dbg !7, !frisk.local_reduction !5
  %565 = extractelement <32 x float> %559, i64 16, !dbg !7
  %566 = extractelement <32 x float> %559, i64 17, !dbg !7
  %local.reduce202 = tail call float @llvm.maxnum.f32(float %565, float %566), !dbg !7, !frisk.local_reduction !5
  %567 = extractelement <32 x float> %559, i64 24, !dbg !7
  %568 = extractelement <32 x float> %559, i64 25, !dbg !7
  %local.reduce203 = tail call float @llvm.maxnum.f32(float %567, float %568), !dbg !7, !frisk.local_reduction !5
  %569 = extractelement <32 x float> %559, i64 2, !dbg !7
  %local.reduce204 = tail call float @llvm.maxnum.f32(float %local.reduce200, float %569), !dbg !7, !frisk.local_reduction !5
  %570 = extractelement <32 x float> %559, i64 10, !dbg !7
  %local.reduce205 = tail call float @llvm.maxnum.f32(float %local.reduce201, float %570), !dbg !7, !frisk.local_reduction !5
  %571 = extractelement <32 x float> %559, i64 18, !dbg !7
  %local.reduce206 = tail call float @llvm.maxnum.f32(float %local.reduce202, float %571), !dbg !7, !frisk.local_reduction !5
  %572 = extractelement <32 x float> %559, i64 26, !dbg !7
  %local.reduce207 = tail call float @llvm.maxnum.f32(float %local.reduce203, float %572), !dbg !7, !frisk.local_reduction !5
  %573 = extractelement <32 x float> %559, i64 3, !dbg !7
  %local.reduce208 = tail call float @llvm.maxnum.f32(float %local.reduce204, float %573), !dbg !7, !frisk.local_reduction !5
  %574 = extractelement <32 x float> %559, i64 11, !dbg !7
  %local.reduce209 = tail call float @llvm.maxnum.f32(float %local.reduce205, float %574), !dbg !7, !frisk.local_reduction !5
  %575 = extractelement <32 x float> %559, i64 19, !dbg !7
  %local.reduce210 = tail call float @llvm.maxnum.f32(float %local.reduce206, float %575), !dbg !7, !frisk.local_reduction !5
  %576 = extractelement <32 x float> %559, i64 27, !dbg !7
  %local.reduce211 = tail call float @llvm.maxnum.f32(float %local.reduce207, float %576), !dbg !7, !frisk.local_reduction !5
  %577 = extractelement <32 x float> %559, i64 4, !dbg !7
  %local.reduce212 = tail call float @llvm.maxnum.f32(float %local.reduce208, float %577), !dbg !7, !frisk.local_reduction !5
  %578 = extractelement <32 x float> %559, i64 12, !dbg !7
  %local.reduce213 = tail call float @llvm.maxnum.f32(float %local.reduce209, float %578), !dbg !7, !frisk.local_reduction !5
  %579 = extractelement <32 x float> %559, i64 20, !dbg !7
  %local.reduce214 = tail call float @llvm.maxnum.f32(float %local.reduce210, float %579), !dbg !7, !frisk.local_reduction !5
  %580 = extractelement <32 x float> %559, i64 28, !dbg !7
  %local.reduce215 = tail call float @llvm.maxnum.f32(float %local.reduce211, float %580), !dbg !7, !frisk.local_reduction !5
  %581 = extractelement <32 x float> %559, i64 5, !dbg !7
  %local.reduce216 = tail call float @llvm.maxnum.f32(float %local.reduce212, float %581), !dbg !7, !frisk.local_reduction !5
  %582 = extractelement <32 x float> %559, i64 13, !dbg !7
  %local.reduce217 = tail call float @llvm.maxnum.f32(float %local.reduce213, float %582), !dbg !7, !frisk.local_reduction !5
  %583 = extractelement <32 x float> %559, i64 21, !dbg !7
  %local.reduce218 = tail call float @llvm.maxnum.f32(float %local.reduce214, float %583), !dbg !7, !frisk.local_reduction !5
  %584 = extractelement <32 x float> %559, i64 29, !dbg !7
  %local.reduce219 = tail call float @llvm.maxnum.f32(float %local.reduce215, float %584), !dbg !7, !frisk.local_reduction !5
  %585 = extractelement <32 x float> %559, i64 6, !dbg !7
  %local.reduce220 = tail call float @llvm.maxnum.f32(float %local.reduce216, float %585), !dbg !7, !frisk.local_reduction !5
  %586 = extractelement <32 x float> %559, i64 14, !dbg !7
  %local.reduce221 = tail call float @llvm.maxnum.f32(float %local.reduce217, float %586), !dbg !7, !frisk.local_reduction !5
  %587 = extractelement <32 x float> %559, i64 22, !dbg !7
  %local.reduce222 = tail call float @llvm.maxnum.f32(float %local.reduce218, float %587), !dbg !7, !frisk.local_reduction !5
  %588 = extractelement <32 x float> %559, i64 30, !dbg !7
  %local.reduce223 = tail call float @llvm.maxnum.f32(float %local.reduce219, float %588), !dbg !7, !frisk.local_reduction !5
  %589 = extractelement <32 x float> %559, i64 7, !dbg !7
  %local.reduce224 = tail call float @llvm.maxnum.f32(float %local.reduce220, float %589), !dbg !7, !frisk.local_reduction !5
  %590 = extractelement <32 x float> %559, i64 15, !dbg !7
  %local.reduce225 = tail call float @llvm.maxnum.f32(float %local.reduce221, float %590), !dbg !7, !frisk.local_reduction !5
  %591 = extractelement <32 x float> %559, i64 23, !dbg !7
  %local.reduce226 = tail call float @llvm.maxnum.f32(float %local.reduce222, float %591), !dbg !7, !frisk.local_reduction !5
  %592 = extractelement <32 x float> %559, i64 31, !dbg !7
  %local.reduce227 = tail call float @llvm.maxnum.f32(float %local.reduce223, float %592), !dbg !7, !frisk.local_reduction !5
  %local.reduce228 = tail call float @llvm.maxnum.f32(float %local.reduce224, float %local.reduce225), !dbg !7, !frisk.local_reduction !5
  %local.reduce229 = tail call float @llvm.maxnum.f32(float %local.reduce226, float %local.reduce227), !dbg !7, !frisk.local_reduction !5
  %593 = tail call float @llvm.maxnum.f32(float %local.reduce228, float %local.reduce229), !dbg !7, !frisk.local_reduction !5
  %594 = shufflevector <4 x float> %464, <4 x float> %466, <32 x i32> <i32 0, i32 4, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 5, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 2, i32 6, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 3, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %595 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %596 = shufflevector <32 x float> %594, <32 x float> %595, <32 x i32> <i32 0, i32 1, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %597 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %598 = shufflevector <32 x float> %596, <32 x float> %597, <32 x i32> <i32 0, i32 1, i32 2, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %599 = shufflevector <4 x float> %472, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %600 = shufflevector <32 x float> %598, <32 x float> %599, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 32, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %601 = shufflevector <4 x float> %474, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %602 = shufflevector <32 x float> %600, <32 x float> %601, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 32, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %603 = shufflevector <4 x float> %476, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %604 = shufflevector <32 x float> %602, <32 x float> %603, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 32, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %605 = shufflevector <4 x float> %478, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %606 = shufflevector <32 x float> %604, <32 x float> %605, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 32, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %607 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %608 = shufflevector <32 x float> %606, <32 x float> %607, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %609 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %610 = shufflevector <32 x float> %608, <32 x float> %609, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %611 = shufflevector <4 x float> %472, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %612 = shufflevector <32 x float> %610, <32 x float> %611, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 33, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %613 = shufflevector <4 x float> %474, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %614 = shufflevector <32 x float> %612, <32 x float> %613, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 33, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %615 = shufflevector <4 x float> %476, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %616 = shufflevector <32 x float> %614, <32 x float> %615, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 33, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %617 = shufflevector <4 x float> %478, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %618 = shufflevector <32 x float> %616, <32 x float> %617, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 33, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %619 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %620 = shufflevector <32 x float> %618, <32 x float> %619, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 34, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %621 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %622 = shufflevector <32 x float> %620, <32 x float> %621, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 34, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %623 = shufflevector <4 x float> %472, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %624 = shufflevector <32 x float> %622, <32 x float> %623, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 34, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %625 = shufflevector <4 x float> %474, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %626 = shufflevector <32 x float> %624, <32 x float> %625, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 34, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %627 = shufflevector <4 x float> %476, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %628 = shufflevector <32 x float> %626, <32 x float> %627, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 34, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %629 = shufflevector <4 x float> %478, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %630 = shufflevector <32 x float> %628, <32 x float> %629, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 34, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %631 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %632 = shufflevector <32 x float> %630, <32 x float> %631, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %633 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %634 = shufflevector <32 x float> %632, <32 x float> %633, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %635 = shufflevector <4 x float> %472, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %636 = shufflevector <32 x float> %634, <32 x float> %635, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 35, i32 poison, i32 poison, i32 poison>, !dbg !16
  %637 = shufflevector <4 x float> %474, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %638 = shufflevector <32 x float> %636, <32 x float> %637, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 35, i32 poison, i32 poison>, !dbg !16
  %639 = shufflevector <4 x float> %476, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %640 = shufflevector <32 x float> %638, <32 x float> %639, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 35, i32 poison>, !dbg !16
  %641 = shufflevector <4 x float> %478, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %642 = shufflevector <32 x float> %640, <32 x float> %641, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 35>, !dbg !16
  %643 = select <32 x i1> %483, <32 x float> %642, <32 x float> splat (float -inf), !dbg !16
  %644 = bitcast float %593 to i32, !dbg !7
  %645 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %644), !dbg !7
  %646 = bitcast i32 %645 to float, !dbg !7
  %647 = tail call float @llvm.maxnum.f32(float %593, float %646), !dbg !7
  %648 = bitcast float %647 to i32, !dbg !7
  %649 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %648), !dbg !7
  %650 = bitcast i32 %649 to float, !dbg !7
  %651 = tail call float @llvm.maxnum.f32(float %647, float %650), !dbg !7
  %652 = extractelement <32 x float> %643, i64 0, !dbg !7
  %653 = tail call float @llvm.maxnum.f32(float %652, float -inf), !dbg !7, !frisk.local_reduction !5
  %654 = extractelement <32 x float> %643, i64 1, !dbg !7
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %653, float %654), !dbg !7, !frisk.local_reduction !5
  %655 = extractelement <32 x float> %643, i64 8, !dbg !7
  %656 = extractelement <32 x float> %643, i64 9, !dbg !7
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %655, float %656), !dbg !7, !frisk.local_reduction !5
  %657 = extractelement <32 x float> %643, i64 16, !dbg !7
  %658 = extractelement <32 x float> %643, i64 17, !dbg !7
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %657, float %658), !dbg !7, !frisk.local_reduction !5
  %659 = extractelement <32 x float> %643, i64 24, !dbg !7
  %660 = extractelement <32 x float> %643, i64 25, !dbg !7
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %659, float %660), !dbg !7, !frisk.local_reduction !5
  %661 = extractelement <32 x float> %643, i64 2, !dbg !7
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %661), !dbg !7, !frisk.local_reduction !5
  %662 = extractelement <32 x float> %643, i64 10, !dbg !7
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %662), !dbg !7, !frisk.local_reduction !5
  %663 = extractelement <32 x float> %643, i64 18, !dbg !7
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %663), !dbg !7, !frisk.local_reduction !5
  %664 = extractelement <32 x float> %643, i64 26, !dbg !7
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %664), !dbg !7, !frisk.local_reduction !5
  %665 = extractelement <32 x float> %643, i64 3, !dbg !7
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %665), !dbg !7, !frisk.local_reduction !5
  %666 = extractelement <32 x float> %643, i64 11, !dbg !7
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %666), !dbg !7, !frisk.local_reduction !5
  %667 = extractelement <32 x float> %643, i64 19, !dbg !7
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %667), !dbg !7, !frisk.local_reduction !5
  %668 = extractelement <32 x float> %643, i64 27, !dbg !7
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce176, float %668), !dbg !7, !frisk.local_reduction !5
  %669 = extractelement <32 x float> %643, i64 4, !dbg !7
  %local.reduce181 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %669), !dbg !7, !frisk.local_reduction !5
  %670 = extractelement <32 x float> %643, i64 12, !dbg !7
  %local.reduce182 = tail call float @llvm.maxnum.f32(float %local.reduce178, float %670), !dbg !7, !frisk.local_reduction !5
  %671 = extractelement <32 x float> %643, i64 20, !dbg !7
  %local.reduce183 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %671), !dbg !7, !frisk.local_reduction !5
  %672 = extractelement <32 x float> %643, i64 28, !dbg !7
  %local.reduce184 = tail call float @llvm.maxnum.f32(float %local.reduce180, float %672), !dbg !7, !frisk.local_reduction !5
  %673 = extractelement <32 x float> %643, i64 5, !dbg !7
  %local.reduce185 = tail call float @llvm.maxnum.f32(float %local.reduce181, float %673), !dbg !7, !frisk.local_reduction !5
  %674 = extractelement <32 x float> %643, i64 13, !dbg !7
  %local.reduce186 = tail call float @llvm.maxnum.f32(float %local.reduce182, float %674), !dbg !7, !frisk.local_reduction !5
  %675 = extractelement <32 x float> %643, i64 21, !dbg !7
  %local.reduce187 = tail call float @llvm.maxnum.f32(float %local.reduce183, float %675), !dbg !7, !frisk.local_reduction !5
  %676 = extractelement <32 x float> %643, i64 29, !dbg !7
  %local.reduce188 = tail call float @llvm.maxnum.f32(float %local.reduce184, float %676), !dbg !7, !frisk.local_reduction !5
  %677 = extractelement <32 x float> %643, i64 6, !dbg !7
  %local.reduce189 = tail call float @llvm.maxnum.f32(float %local.reduce185, float %677), !dbg !7, !frisk.local_reduction !5
  %678 = extractelement <32 x float> %643, i64 14, !dbg !7
  %local.reduce190 = tail call float @llvm.maxnum.f32(float %local.reduce186, float %678), !dbg !7, !frisk.local_reduction !5
  %679 = extractelement <32 x float> %643, i64 22, !dbg !7
  %local.reduce191 = tail call float @llvm.maxnum.f32(float %local.reduce187, float %679), !dbg !7, !frisk.local_reduction !5
  %680 = extractelement <32 x float> %643, i64 30, !dbg !7
  %local.reduce192 = tail call float @llvm.maxnum.f32(float %local.reduce188, float %680), !dbg !7, !frisk.local_reduction !5
  %681 = extractelement <32 x float> %643, i64 7, !dbg !7
  %local.reduce193 = tail call float @llvm.maxnum.f32(float %local.reduce189, float %681), !dbg !7, !frisk.local_reduction !5
  %682 = extractelement <32 x float> %643, i64 15, !dbg !7
  %local.reduce194 = tail call float @llvm.maxnum.f32(float %local.reduce190, float %682), !dbg !7, !frisk.local_reduction !5
  %683 = extractelement <32 x float> %643, i64 23, !dbg !7
  %local.reduce195 = tail call float @llvm.maxnum.f32(float %local.reduce191, float %683), !dbg !7, !frisk.local_reduction !5
  %684 = extractelement <32 x float> %643, i64 31, !dbg !7
  %local.reduce196 = tail call float @llvm.maxnum.f32(float %local.reduce192, float %684), !dbg !7, !frisk.local_reduction !5
  %local.reduce197 = tail call float @llvm.maxnum.f32(float %local.reduce193, float %local.reduce194), !dbg !7, !frisk.local_reduction !5
  %local.reduce198 = tail call float @llvm.maxnum.f32(float %local.reduce195, float %local.reduce196), !dbg !7, !frisk.local_reduction !5
  %685 = tail call float @llvm.maxnum.f32(float %local.reduce197, float %local.reduce198), !dbg !7, !frisk.local_reduction !5
  %686 = bitcast float %685 to i32, !dbg !7
  %687 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %686), !dbg !7
  %688 = bitcast i32 %687 to float, !dbg !7
  %689 = tail call float @llvm.maxnum.f32(float %685, float %688), !dbg !7
  %690 = bitcast float %689 to i32, !dbg !7
  %691 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %690), !dbg !7
  %692 = bitcast i32 %691 to float, !dbg !7
  %693 = tail call float @llvm.maxnum.f32(float %689, float %692), !dbg !7
  %694 = extractvalue [2 x <1 x float>] %157, 0, !dbg !7
  %695 = insertelement <1 x float> poison, float %651, i64 0, !dbg !7
  %696 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %695, <1 x float> %694), !dbg !7
  %697 = extractvalue [2 x <1 x float>] %157, 1, !dbg !7
  %698 = insertelement <1 x float> poison, float %693, i64 0, !dbg !7
  %699 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %698, <1 x float> %697), !dbg !7
  %700 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 0, i32 8, i32 16, i32 24>, !dbg !42
  %701 = shufflevector <1 x float> %696, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %702 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 1, i32 9, i32 17, i32 25>, !dbg !42
  %703 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 2, i32 10, i32 18, i32 26>, !dbg !42
  %704 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 3, i32 11, i32 19, i32 27>, !dbg !42
  %705 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 4, i32 12, i32 20, i32 28>, !dbg !42
  %706 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 5, i32 13, i32 21, i32 29>, !dbg !42
  %707 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 6, i32 14, i32 22, i32 30>, !dbg !42
  %708 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 7, i32 15, i32 23, i32 31>, !dbg !42
  %709 = shufflevector <32 x float> %643, <32 x float> poison, <4 x i32> <i32 0, i32 8, i32 16, i32 24>, !dbg !42
  %710 = shufflevector <1 x float> %699, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %711 = shufflevector <32 x float> %643, <32 x float> poison, <4 x i32> <i32 1, i32 9, i32 17, i32 25>, !dbg !42
  %712 = shufflevector <32 x float> %643, <32 x float> poison, <4 x i32> <i32 2, i32 10, i32 18, i32 26>, !dbg !42
  %713 = shufflevector <32 x float> %643, <32 x float> poison, <4 x i32> <i32 3, i32 11, i32 19, i32 27>, !dbg !42
  %714 = shufflevector <32 x float> %643, <32 x float> poison, <4 x i32> <i32 4, i32 12, i32 20, i32 28>, !dbg !42
  %715 = shufflevector <32 x float> %643, <32 x float> poison, <4 x i32> <i32 5, i32 13, i32 21, i32 29>, !dbg !42
  %716 = shufflevector <32 x float> %643, <32 x float> poison, <4 x i32> <i32 6, i32 14, i32 22, i32 30>, !dbg !42
  %717 = shufflevector <32 x float> %643, <32 x float> poison, <4 x i32> <i32 7, i32 15, i32 23, i32 31>, !dbg !42
  %718 = fsub <1 x float> %694, %696, !dbg !43
  %719 = extractelement <1 x float> %718, i64 0, !dbg !44
  %720 = tail call float @llvm.amdgcn.exp2.f32(float %719), !dbg !44
  %721 = insertelement <4 x float> poison, float %720, i64 0, !dbg !45
  %722 = shufflevector <4 x float> %721, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !45
  %723 = extractvalue [2 x <1 x float>] %158, 0, !dbg !46
  %724 = extractelement <1 x float> %723, i64 0, !dbg !46
  %725 = fmul float %724, %720, !dbg !46
  %726 = fsub <1 x float> %697, %699, !dbg !43
  %727 = extractelement <1 x float> %726, i64 0, !dbg !44
  %728 = tail call float @llvm.amdgcn.exp2.f32(float %727), !dbg !44
  %729 = insertelement <4 x float> poison, float %728, i64 0, !dbg !45
  %730 = shufflevector <4 x float> %729, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !45
  %731 = extractvalue [2 x <1 x float>] %158, 1, !dbg !46
  %732 = extractelement <1 x float> %731, i64 0, !dbg !46
  %733 = fmul float %732, %728, !dbg !46
  %734 = extractvalue [2 x <8 x float>] %159, 0, !dbg !45
  %735 = shufflevector <8 x float> %734, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %736 = fmul <4 x float> %735, %722, !dbg !45
  %737 = shufflevector <8 x float> %734, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %738 = fmul <4 x float> %737, %722, !dbg !45
  %739 = extractvalue [2 x <8 x float>] %159, 1, !dbg !45
  %740 = shufflevector <8 x float> %739, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %741 = fmul <4 x float> %740, %730, !dbg !45
  %742 = shufflevector <8 x float> %739, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %743 = fmul <4 x float> %742, %730, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1528 = or disjoint i64 %425, 32, !dbg !17
  %vdb.address1529 = add i64 %vdb.address1528, %424, !dbg !17
  %vdb.address1530 = trunc i64 %vdb.address1529 to i32, !dbg !17
  %vdb.address1531 = shl i32 %vdb.address1530, 1, !dbg !17
  %vdb.address1539 = or i32 %.tr1512, 16448, !dbg !17
  %744 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1531, <4 x i32> %dma.buffer.rsrc1411) #9, !dbg !17
  %745 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1544, i32 %vdb.address1539, <4 x i32> %dma.buffer.rsrc1411) #9, !dbg !17
  %address.local994 = and i32 %address.local985, 120, !dbg !18
  %address.local996 = and i32 %address.local636, 1536, !dbg !18
  %address.local997 = or disjoint i32 %address.local994, %address.local996, !dbg !18
  %address.local998 = zext nneg i32 %address.local997 to i64, !dbg !18
  %address.local999 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local998, !dbg !18
  %746 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local999, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %747 = bitcast <4 x i32> %746 to <8 x half>, !dbg !18
  %address.local1000 = and i64 %address.local563, 60, !dbg !18
  %address.local1001 = add nuw nsw i64 %address.local1000, %address.local563, !dbg !18
  %address.local1003.idx = shl nuw nsw i64 %address.local1001, 4, !dbg !18
  %address.local1003 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local1003.idx, !dbg !18
  %address.local1004 = lshr i64 %address.local616, 4, !dbg !18
  %address.local1006.idx = mul nuw nsw i64 %address.local1004, 768, !dbg !18
  %address.local1006 = getelementptr i8, ptr addrspace(3) %address.local1003, i64 %address.local1006.idx, !dbg !18
  %address.local1008.idx = mul nsw i64 %209, -64, !dbg !18
  %address.local1008 = getelementptr i8, ptr addrspace(3) %address.local1006, i64 %address.local1008.idx, !dbg !18
  %address.local1009 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 256, !dbg !18
  %748 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1009, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %749 = bitcast <4 x i32> %748 to <8 x half>, !dbg !18
  %address.local1013 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 512, !dbg !18
  %750 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %751 = bitcast <4 x i32> %750 to <8 x half>, !dbg !18
  %address.local1017 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 768, !dbg !18
  %752 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %753 = bitcast <4 x i32> %752 to <8 x half>, !dbg !18
  %address.local1019 = getelementptr i8, ptr addrspace(3) %address.local999, i64 4096, !dbg !18
  %754 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1019, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %755 = bitcast <4 x i32> %754 to <8 x half>, !dbg !18
  %address.local1023 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 4352, !dbg !18
  %756 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %757 = bitcast <4 x i32> %756 to <8 x half>, !dbg !18
  %address.local1027 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 4608, !dbg !18
  %758 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %759 = bitcast <4 x i32> %758 to <8 x half>, !dbg !18
  %address.local1031 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 4864, !dbg !18
  %760 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %761 = shufflevector <8 x half> %747, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %762 = fsub <4 x float> %700, %701, !dbg !42
  %763 = extractelement <4 x float> %762, i64 0, !dbg !48
  %764 = tail call float @llvm.amdgcn.exp2.f32(float %763), !dbg !48
  %765 = insertelement <4 x float> poison, float %764, i64 0, !dbg !49
  %766 = extractelement <4 x float> %762, i64 1, !dbg !48
  %767 = tail call float @llvm.amdgcn.exp2.f32(float %766), !dbg !48
  %768 = insertelement <4 x float> %765, float %767, i64 1, !dbg !49
  %769 = extractelement <4 x float> %762, i64 2, !dbg !48
  %770 = tail call float @llvm.amdgcn.exp2.f32(float %769), !dbg !48
  %771 = insertelement <4 x float> %768, float %770, i64 2, !dbg !49
  %772 = extractelement <4 x float> %762, i64 3, !dbg !48
  %773 = tail call float @llvm.amdgcn.exp2.f32(float %772), !dbg !48
  %774 = insertelement <4 x float> %771, float %773, i64 3, !dbg !49
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %774) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !49
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !49
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !49
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !49
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !49
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !49
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !49
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !49
  %775 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !49
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %775) #8, !dbg !48, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %776 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %761, <4 x float> %736) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %777 = shufflevector <8 x half> %747, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %778 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %777, <4 x float> %738) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %779 = fsub <4 x float> %709, %710, !dbg !42
  %780 = extractelement <4 x float> %779, i64 0, !dbg !48
  %781 = tail call float @llvm.amdgcn.exp2.f32(float %780), !dbg !48
  %782 = insertelement <4 x float> poison, float %781, i64 0, !dbg !49
  %783 = extractelement <4 x float> %779, i64 1, !dbg !48
  %784 = tail call float @llvm.amdgcn.exp2.f32(float %783), !dbg !48
  %785 = insertelement <4 x float> %782, float %784, i64 1, !dbg !49
  %786 = extractelement <4 x float> %779, i64 2, !dbg !48
  %787 = tail call float @llvm.amdgcn.exp2.f32(float %786), !dbg !48
  %788 = insertelement <4 x float> %785, float %787, i64 2, !dbg !49
  %789 = extractelement <4 x float> %779, i64 3, !dbg !48
  %790 = tail call float @llvm.amdgcn.exp2.f32(float %789), !dbg !48
  %791 = insertelement <4 x float> %788, float %790, i64 3, !dbg !49
  %math.result449 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %791) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02104 = extractelement <4 x float> %math.result449, i64 0, !dbg !49
  %.cvtpack.f12105 = extractelement <4 x float> %math.result449, i64 1, !dbg !49
  %.cvtpack.f22106 = extractelement <4 x float> %math.result449, i64 2, !dbg !49
  %.cvtpack.f32107 = extractelement <4 x float> %math.result449, i64 3, !dbg !49
  %.cvtpack.p02108 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02104, float %.cvtpack.f12105) #12, !dbg !49
  %.cvtpack.p12109 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22106, float %.cvtpack.f32107) #12, !dbg !49
  %.cvtpack.v02110 = insertelement <2 x i32> poison, i32 %.cvtpack.p02108, i64 0, !dbg !49
  %.cvtpack.v12111 = insertelement <2 x i32> %.cvtpack.v02110, i32 %.cvtpack.p12109, i64 1, !dbg !49
  %792 = bitcast <2 x i32> %.cvtpack.v12111 to <4 x half>, !dbg !49
  %math.packed450 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %792) #8, !dbg !48, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %793 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %761, <4 x float> %741) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %794 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %777, <4 x float> %743) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %795 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %796 = fsub <4 x float> %702, %701, !dbg !42
  %797 = extractelement <4 x float> %796, i64 0, !dbg !48
  %798 = tail call float @llvm.amdgcn.exp2.f32(float %797), !dbg !48
  %local.reduce138 = fadd float %764, %798, !dbg !50, !frisk.local_reduction !5
  %799 = insertelement <4 x float> poison, float %798, i64 0, !dbg !49
  %800 = extractelement <4 x float> %796, i64 1, !dbg !48
  %801 = tail call float @llvm.amdgcn.exp2.f32(float %800), !dbg !48
  %local.reduce139 = fadd float %767, %801, !dbg !50, !frisk.local_reduction !5
  %802 = insertelement <4 x float> %799, float %801, i64 1, !dbg !49
  %803 = extractelement <4 x float> %796, i64 2, !dbg !48
  %804 = tail call float @llvm.amdgcn.exp2.f32(float %803), !dbg !48
  %local.reduce140 = fadd float %770, %804, !dbg !50, !frisk.local_reduction !5
  %805 = insertelement <4 x float> %802, float %804, i64 2, !dbg !49
  %806 = extractelement <4 x float> %796, i64 3, !dbg !48
  %807 = tail call float @llvm.amdgcn.exp2.f32(float %806), !dbg !48
  %local.reduce141 = fadd float %773, %807, !dbg !50, !frisk.local_reduction !5
  %808 = insertelement <4 x float> %805, float %807, i64 3, !dbg !49
  %math.result435 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %808) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02112 = extractelement <4 x float> %math.result435, i64 0, !dbg !49
  %.cvtpack.f12113 = extractelement <4 x float> %math.result435, i64 1, !dbg !49
  %.cvtpack.f22114 = extractelement <4 x float> %math.result435, i64 2, !dbg !49
  %.cvtpack.f32115 = extractelement <4 x float> %math.result435, i64 3, !dbg !49
  %.cvtpack.p02116 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02112, float %.cvtpack.f12113) #12, !dbg !49
  %.cvtpack.p12117 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22114, float %.cvtpack.f32115) #12, !dbg !49
  %.cvtpack.v02118 = insertelement <2 x i32> poison, i32 %.cvtpack.p02116, i64 0, !dbg !49
  %.cvtpack.v12119 = insertelement <2 x i32> %.cvtpack.v02118, i32 %.cvtpack.p12117, i64 1, !dbg !49
  %809 = bitcast <2 x i32> %.cvtpack.v12119 to <4 x half>, !dbg !49
  %math.packed436 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %809) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %810 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %795, <4 x float> %776) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %811 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %812 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %811, <4 x float> %778) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %813 = fsub <4 x float> %711, %710, !dbg !42
  %814 = extractelement <4 x float> %813, i64 0, !dbg !48
  %815 = tail call float @llvm.amdgcn.exp2.f32(float %814), !dbg !48
  %local.reduce = fadd float %781, %815, !dbg !50, !frisk.local_reduction !5
  %816 = insertelement <4 x float> poison, float %815, i64 0, !dbg !49
  %817 = extractelement <4 x float> %813, i64 1, !dbg !48
  %818 = tail call float @llvm.amdgcn.exp2.f32(float %817), !dbg !48
  %local.reduce108 = fadd float %784, %818, !dbg !50, !frisk.local_reduction !5
  %819 = insertelement <4 x float> %816, float %818, i64 1, !dbg !49
  %820 = extractelement <4 x float> %813, i64 2, !dbg !48
  %821 = tail call float @llvm.amdgcn.exp2.f32(float %820), !dbg !48
  %local.reduce109 = fadd float %787, %821, !dbg !50, !frisk.local_reduction !5
  %822 = insertelement <4 x float> %819, float %821, i64 2, !dbg !49
  %823 = extractelement <4 x float> %813, i64 3, !dbg !48
  %824 = tail call float @llvm.amdgcn.exp2.f32(float %823), !dbg !48
  %local.reduce110 = fadd float %790, %824, !dbg !50, !frisk.local_reduction !5
  %825 = insertelement <4 x float> %822, float %824, i64 3, !dbg !49
  %math.result451 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %825) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02120 = extractelement <4 x float> %math.result451, i64 0, !dbg !49
  %.cvtpack.f12121 = extractelement <4 x float> %math.result451, i64 1, !dbg !49
  %.cvtpack.f22122 = extractelement <4 x float> %math.result451, i64 2, !dbg !49
  %.cvtpack.f32123 = extractelement <4 x float> %math.result451, i64 3, !dbg !49
  %.cvtpack.p02124 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02120, float %.cvtpack.f12121) #12, !dbg !49
  %.cvtpack.p12125 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22122, float %.cvtpack.f32123) #12, !dbg !49
  %.cvtpack.v02126 = insertelement <2 x i32> poison, i32 %.cvtpack.p02124, i64 0, !dbg !49
  %.cvtpack.v12127 = insertelement <2 x i32> %.cvtpack.v02126, i32 %.cvtpack.p12125, i64 1, !dbg !49
  %826 = bitcast <2 x i32> %.cvtpack.v12127 to <4 x half>, !dbg !49
  %math.packed452 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %826) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %827 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %795, <4 x float> %793) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %828 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %811, <4 x float> %794) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %829 = shufflevector <8 x half> %751, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %830 = fsub <4 x float> %703, %701, !dbg !42
  %831 = extractelement <4 x float> %830, i64 0, !dbg !48
  %832 = tail call float @llvm.amdgcn.exp2.f32(float %831), !dbg !48
  %local.reduce142 = fadd float %832, %local.reduce138, !dbg !50, !frisk.local_reduction !5
  %833 = insertelement <4 x float> poison, float %832, i64 0, !dbg !49
  %834 = extractelement <4 x float> %830, i64 1, !dbg !48
  %835 = tail call float @llvm.amdgcn.exp2.f32(float %834), !dbg !48
  %local.reduce143 = fadd float %835, %local.reduce139, !dbg !50, !frisk.local_reduction !5
  %836 = insertelement <4 x float> %833, float %835, i64 1, !dbg !49
  %837 = extractelement <4 x float> %830, i64 2, !dbg !48
  %838 = tail call float @llvm.amdgcn.exp2.f32(float %837), !dbg !48
  %local.reduce144 = fadd float %838, %local.reduce140, !dbg !50, !frisk.local_reduction !5
  %839 = insertelement <4 x float> %836, float %838, i64 2, !dbg !49
  %840 = extractelement <4 x float> %830, i64 3, !dbg !48
  %841 = tail call float @llvm.amdgcn.exp2.f32(float %840), !dbg !48
  %local.reduce145 = fadd float %841, %local.reduce141, !dbg !50, !frisk.local_reduction !5
  %842 = insertelement <4 x float> %839, float %841, i64 3, !dbg !49
  %math.result437 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %842) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02128 = extractelement <4 x float> %math.result437, i64 0, !dbg !49
  %.cvtpack.f12129 = extractelement <4 x float> %math.result437, i64 1, !dbg !49
  %.cvtpack.f22130 = extractelement <4 x float> %math.result437, i64 2, !dbg !49
  %.cvtpack.f32131 = extractelement <4 x float> %math.result437, i64 3, !dbg !49
  %.cvtpack.p02132 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02128, float %.cvtpack.f12129) #12, !dbg !49
  %.cvtpack.p12133 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22130, float %.cvtpack.f32131) #12, !dbg !49
  %.cvtpack.v02134 = insertelement <2 x i32> poison, i32 %.cvtpack.p02132, i64 0, !dbg !49
  %.cvtpack.v12135 = insertelement <2 x i32> %.cvtpack.v02134, i32 %.cvtpack.p12133, i64 1, !dbg !49
  %843 = bitcast <2 x i32> %.cvtpack.v12135 to <4 x half>, !dbg !49
  %math.packed438 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %843) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %829, <4 x float> %810) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %845 = shufflevector <8 x half> %751, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %845, <4 x float> %812) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %847 = fsub <4 x float> %712, %710, !dbg !42
  %848 = extractelement <4 x float> %847, i64 0, !dbg !48
  %849 = tail call float @llvm.amdgcn.exp2.f32(float %848), !dbg !48
  %local.reduce111 = fadd float %849, %local.reduce, !dbg !50, !frisk.local_reduction !5
  %850 = insertelement <4 x float> poison, float %849, i64 0, !dbg !49
  %851 = extractelement <4 x float> %847, i64 1, !dbg !48
  %852 = tail call float @llvm.amdgcn.exp2.f32(float %851), !dbg !48
  %local.reduce112 = fadd float %852, %local.reduce108, !dbg !50, !frisk.local_reduction !5
  %853 = insertelement <4 x float> %850, float %852, i64 1, !dbg !49
  %854 = extractelement <4 x float> %847, i64 2, !dbg !48
  %855 = tail call float @llvm.amdgcn.exp2.f32(float %854), !dbg !48
  %local.reduce113 = fadd float %855, %local.reduce109, !dbg !50, !frisk.local_reduction !5
  %856 = insertelement <4 x float> %853, float %855, i64 2, !dbg !49
  %857 = extractelement <4 x float> %847, i64 3, !dbg !48
  %858 = tail call float @llvm.amdgcn.exp2.f32(float %857), !dbg !48
  %local.reduce114 = fadd float %858, %local.reduce110, !dbg !50, !frisk.local_reduction !5
  %859 = insertelement <4 x float> %856, float %858, i64 3, !dbg !49
  %math.result453 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %859) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02136 = extractelement <4 x float> %math.result453, i64 0, !dbg !49
  %.cvtpack.f12137 = extractelement <4 x float> %math.result453, i64 1, !dbg !49
  %.cvtpack.f22138 = extractelement <4 x float> %math.result453, i64 2, !dbg !49
  %.cvtpack.f32139 = extractelement <4 x float> %math.result453, i64 3, !dbg !49
  %.cvtpack.p02140 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02136, float %.cvtpack.f12137) #12, !dbg !49
  %.cvtpack.p12141 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22138, float %.cvtpack.f32139) #12, !dbg !49
  %.cvtpack.v02142 = insertelement <2 x i32> poison, i32 %.cvtpack.p02140, i64 0, !dbg !49
  %.cvtpack.v12143 = insertelement <2 x i32> %.cvtpack.v02142, i32 %.cvtpack.p12141, i64 1, !dbg !49
  %860 = bitcast <2 x i32> %.cvtpack.v12143 to <4 x half>, !dbg !49
  %math.packed454 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %860) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %861 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %829, <4 x float> %827) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %862 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %845, <4 x float> %828) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %863 = shufflevector <8 x half> %753, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %864 = fsub <4 x float> %704, %701, !dbg !42
  %865 = extractelement <4 x float> %864, i64 0, !dbg !48
  %866 = tail call float @llvm.amdgcn.exp2.f32(float %865), !dbg !48
  %local.reduce146 = fadd float %866, %local.reduce142, !dbg !50, !frisk.local_reduction !5
  %867 = insertelement <4 x float> poison, float %866, i64 0, !dbg !49
  %868 = extractelement <4 x float> %864, i64 1, !dbg !48
  %869 = tail call float @llvm.amdgcn.exp2.f32(float %868), !dbg !48
  %local.reduce147 = fadd float %869, %local.reduce143, !dbg !50, !frisk.local_reduction !5
  %870 = insertelement <4 x float> %867, float %869, i64 1, !dbg !49
  %871 = extractelement <4 x float> %864, i64 2, !dbg !48
  %872 = tail call float @llvm.amdgcn.exp2.f32(float %871), !dbg !48
  %local.reduce148 = fadd float %872, %local.reduce144, !dbg !50, !frisk.local_reduction !5
  %873 = insertelement <4 x float> %870, float %872, i64 2, !dbg !49
  %874 = extractelement <4 x float> %864, i64 3, !dbg !48
  %875 = tail call float @llvm.amdgcn.exp2.f32(float %874), !dbg !48
  %local.reduce149 = fadd float %875, %local.reduce145, !dbg !50, !frisk.local_reduction !5
  %876 = insertelement <4 x float> %873, float %875, i64 3, !dbg !49
  %math.result439 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %876) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02144 = extractelement <4 x float> %math.result439, i64 0, !dbg !49
  %.cvtpack.f12145 = extractelement <4 x float> %math.result439, i64 1, !dbg !49
  %.cvtpack.f22146 = extractelement <4 x float> %math.result439, i64 2, !dbg !49
  %.cvtpack.f32147 = extractelement <4 x float> %math.result439, i64 3, !dbg !49
  %.cvtpack.p02148 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02144, float %.cvtpack.f12145) #12, !dbg !49
  %.cvtpack.p12149 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22146, float %.cvtpack.f32147) #12, !dbg !49
  %.cvtpack.v02150 = insertelement <2 x i32> poison, i32 %.cvtpack.p02148, i64 0, !dbg !49
  %.cvtpack.v12151 = insertelement <2 x i32> %.cvtpack.v02150, i32 %.cvtpack.p12149, i64 1, !dbg !49
  %877 = bitcast <2 x i32> %.cvtpack.v12151 to <4 x half>, !dbg !49
  %math.packed440 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %877) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %878 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %863, <4 x float> %844) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %879 = shufflevector <8 x half> %753, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %880 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %879, <4 x float> %846) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %881 = fsub <4 x float> %713, %710, !dbg !42
  %882 = extractelement <4 x float> %881, i64 0, !dbg !48
  %883 = tail call float @llvm.amdgcn.exp2.f32(float %882), !dbg !48
  %local.reduce115 = fadd float %883, %local.reduce111, !dbg !50, !frisk.local_reduction !5
  %884 = insertelement <4 x float> poison, float %883, i64 0, !dbg !49
  %885 = extractelement <4 x float> %881, i64 1, !dbg !48
  %886 = tail call float @llvm.amdgcn.exp2.f32(float %885), !dbg !48
  %local.reduce116 = fadd float %886, %local.reduce112, !dbg !50, !frisk.local_reduction !5
  %887 = insertelement <4 x float> %884, float %886, i64 1, !dbg !49
  %888 = extractelement <4 x float> %881, i64 2, !dbg !48
  %889 = tail call float @llvm.amdgcn.exp2.f32(float %888), !dbg !48
  %local.reduce117 = fadd float %889, %local.reduce113, !dbg !50, !frisk.local_reduction !5
  %890 = insertelement <4 x float> %887, float %889, i64 2, !dbg !49
  %891 = extractelement <4 x float> %881, i64 3, !dbg !48
  %892 = tail call float @llvm.amdgcn.exp2.f32(float %891), !dbg !48
  %local.reduce118 = fadd float %892, %local.reduce114, !dbg !50, !frisk.local_reduction !5
  %893 = insertelement <4 x float> %890, float %892, i64 3, !dbg !49
  %math.result455 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %893) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02152 = extractelement <4 x float> %math.result455, i64 0, !dbg !49
  %.cvtpack.f12153 = extractelement <4 x float> %math.result455, i64 1, !dbg !49
  %.cvtpack.f22154 = extractelement <4 x float> %math.result455, i64 2, !dbg !49
  %.cvtpack.f32155 = extractelement <4 x float> %math.result455, i64 3, !dbg !49
  %.cvtpack.p02156 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02152, float %.cvtpack.f12153) #12, !dbg !49
  %.cvtpack.p12157 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22154, float %.cvtpack.f32155) #12, !dbg !49
  %.cvtpack.v02158 = insertelement <2 x i32> poison, i32 %.cvtpack.p02156, i64 0, !dbg !49
  %.cvtpack.v12159 = insertelement <2 x i32> %.cvtpack.v02158, i32 %.cvtpack.p12157, i64 1, !dbg !49
  %894 = bitcast <2 x i32> %.cvtpack.v12159 to <4 x half>, !dbg !49
  %math.packed456 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %894) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %863, <4 x float> %861) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %879, <4 x float> %862) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %897 = shufflevector <8 x half> %755, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %898 = fsub <4 x float> %705, %701, !dbg !42
  %899 = extractelement <4 x float> %898, i64 0, !dbg !48
  %900 = tail call float @llvm.amdgcn.exp2.f32(float %899), !dbg !48
  %local.reduce150 = fadd float %900, %local.reduce146, !dbg !50, !frisk.local_reduction !5
  %901 = insertelement <4 x float> poison, float %900, i64 0, !dbg !49
  %902 = extractelement <4 x float> %898, i64 1, !dbg !48
  %903 = tail call float @llvm.amdgcn.exp2.f32(float %902), !dbg !48
  %local.reduce151 = fadd float %903, %local.reduce147, !dbg !50, !frisk.local_reduction !5
  %904 = insertelement <4 x float> %901, float %903, i64 1, !dbg !49
  %905 = extractelement <4 x float> %898, i64 2, !dbg !48
  %906 = tail call float @llvm.amdgcn.exp2.f32(float %905), !dbg !48
  %local.reduce152 = fadd float %906, %local.reduce148, !dbg !50, !frisk.local_reduction !5
  %907 = insertelement <4 x float> %904, float %906, i64 2, !dbg !49
  %908 = extractelement <4 x float> %898, i64 3, !dbg !48
  %909 = tail call float @llvm.amdgcn.exp2.f32(float %908), !dbg !48
  %local.reduce153 = fadd float %909, %local.reduce149, !dbg !50, !frisk.local_reduction !5
  %910 = insertelement <4 x float> %907, float %909, i64 3, !dbg !49
  %math.result441 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %910) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02160 = extractelement <4 x float> %math.result441, i64 0, !dbg !49
  %.cvtpack.f12161 = extractelement <4 x float> %math.result441, i64 1, !dbg !49
  %.cvtpack.f22162 = extractelement <4 x float> %math.result441, i64 2, !dbg !49
  %.cvtpack.f32163 = extractelement <4 x float> %math.result441, i64 3, !dbg !49
  %.cvtpack.p02164 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02160, float %.cvtpack.f12161) #12, !dbg !49
  %.cvtpack.p12165 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22162, float %.cvtpack.f32163) #12, !dbg !49
  %.cvtpack.v02166 = insertelement <2 x i32> poison, i32 %.cvtpack.p02164, i64 0, !dbg !49
  %.cvtpack.v12167 = insertelement <2 x i32> %.cvtpack.v02166, i32 %.cvtpack.p12165, i64 1, !dbg !49
  %911 = bitcast <2 x i32> %.cvtpack.v12167 to <4 x half>, !dbg !49
  %math.packed442 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %911) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %897, <4 x float> %878) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %913 = shufflevector <8 x half> %755, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %913, <4 x float> %880) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %915 = fsub <4 x float> %714, %710, !dbg !42
  %916 = extractelement <4 x float> %915, i64 0, !dbg !48
  %917 = tail call float @llvm.amdgcn.exp2.f32(float %916), !dbg !48
  %local.reduce119 = fadd float %917, %local.reduce115, !dbg !50, !frisk.local_reduction !5
  %918 = insertelement <4 x float> poison, float %917, i64 0, !dbg !49
  %919 = extractelement <4 x float> %915, i64 1, !dbg !48
  %920 = tail call float @llvm.amdgcn.exp2.f32(float %919), !dbg !48
  %local.reduce120 = fadd float %920, %local.reduce116, !dbg !50, !frisk.local_reduction !5
  %921 = insertelement <4 x float> %918, float %920, i64 1, !dbg !49
  %922 = extractelement <4 x float> %915, i64 2, !dbg !48
  %923 = tail call float @llvm.amdgcn.exp2.f32(float %922), !dbg !48
  %local.reduce121 = fadd float %923, %local.reduce117, !dbg !50, !frisk.local_reduction !5
  %924 = insertelement <4 x float> %921, float %923, i64 2, !dbg !49
  %925 = extractelement <4 x float> %915, i64 3, !dbg !48
  %926 = tail call float @llvm.amdgcn.exp2.f32(float %925), !dbg !48
  %local.reduce122 = fadd float %926, %local.reduce118, !dbg !50, !frisk.local_reduction !5
  %927 = insertelement <4 x float> %924, float %926, i64 3, !dbg !49
  %math.result457 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %927) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02168 = extractelement <4 x float> %math.result457, i64 0, !dbg !49
  %.cvtpack.f12169 = extractelement <4 x float> %math.result457, i64 1, !dbg !49
  %.cvtpack.f22170 = extractelement <4 x float> %math.result457, i64 2, !dbg !49
  %.cvtpack.f32171 = extractelement <4 x float> %math.result457, i64 3, !dbg !49
  %.cvtpack.p02172 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02168, float %.cvtpack.f12169) #12, !dbg !49
  %.cvtpack.p12173 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22170, float %.cvtpack.f32171) #12, !dbg !49
  %.cvtpack.v02174 = insertelement <2 x i32> poison, i32 %.cvtpack.p02172, i64 0, !dbg !49
  %.cvtpack.v12175 = insertelement <2 x i32> %.cvtpack.v02174, i32 %.cvtpack.p12173, i64 1, !dbg !49
  %928 = bitcast <2 x i32> %.cvtpack.v12175 to <4 x half>, !dbg !49
  %math.packed458 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %928) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %929 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %897, <4 x float> %895) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %913, <4 x float> %896) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %931 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %932 = fsub <4 x float> %706, %701, !dbg !42
  %933 = extractelement <4 x float> %932, i64 0, !dbg !48
  %934 = tail call float @llvm.amdgcn.exp2.f32(float %933), !dbg !48
  %local.reduce154 = fadd float %934, %local.reduce150, !dbg !50, !frisk.local_reduction !5
  %935 = insertelement <4 x float> poison, float %934, i64 0, !dbg !49
  %936 = extractelement <4 x float> %932, i64 1, !dbg !48
  %937 = tail call float @llvm.amdgcn.exp2.f32(float %936), !dbg !48
  %local.reduce155 = fadd float %937, %local.reduce151, !dbg !50, !frisk.local_reduction !5
  %938 = insertelement <4 x float> %935, float %937, i64 1, !dbg !49
  %939 = extractelement <4 x float> %932, i64 2, !dbg !48
  %940 = tail call float @llvm.amdgcn.exp2.f32(float %939), !dbg !48
  %local.reduce156 = fadd float %940, %local.reduce152, !dbg !50, !frisk.local_reduction !5
  %941 = insertelement <4 x float> %938, float %940, i64 2, !dbg !49
  %942 = extractelement <4 x float> %932, i64 3, !dbg !48
  %943 = tail call float @llvm.amdgcn.exp2.f32(float %942), !dbg !48
  %local.reduce157 = fadd float %943, %local.reduce153, !dbg !50, !frisk.local_reduction !5
  %944 = insertelement <4 x float> %941, float %943, i64 3, !dbg !49
  %math.result443 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %944) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02176 = extractelement <4 x float> %math.result443, i64 0, !dbg !49
  %.cvtpack.f12177 = extractelement <4 x float> %math.result443, i64 1, !dbg !49
  %.cvtpack.f22178 = extractelement <4 x float> %math.result443, i64 2, !dbg !49
  %.cvtpack.f32179 = extractelement <4 x float> %math.result443, i64 3, !dbg !49
  %.cvtpack.p02180 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02176, float %.cvtpack.f12177) #12, !dbg !49
  %.cvtpack.p12181 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22178, float %.cvtpack.f32179) #12, !dbg !49
  %.cvtpack.v02182 = insertelement <2 x i32> poison, i32 %.cvtpack.p02180, i64 0, !dbg !49
  %.cvtpack.v12183 = insertelement <2 x i32> %.cvtpack.v02182, i32 %.cvtpack.p12181, i64 1, !dbg !49
  %945 = bitcast <2 x i32> %.cvtpack.v12183 to <4 x half>, !dbg !49
  %math.packed444 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %945) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %946 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %931, <4 x float> %912) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %947 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %948 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %947, <4 x float> %914) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %949 = fsub <4 x float> %715, %710, !dbg !42
  %950 = extractelement <4 x float> %949, i64 0, !dbg !48
  %951 = tail call float @llvm.amdgcn.exp2.f32(float %950), !dbg !48
  %local.reduce123 = fadd float %951, %local.reduce119, !dbg !50, !frisk.local_reduction !5
  %952 = insertelement <4 x float> poison, float %951, i64 0, !dbg !49
  %953 = extractelement <4 x float> %949, i64 1, !dbg !48
  %954 = tail call float @llvm.amdgcn.exp2.f32(float %953), !dbg !48
  %local.reduce124 = fadd float %954, %local.reduce120, !dbg !50, !frisk.local_reduction !5
  %955 = insertelement <4 x float> %952, float %954, i64 1, !dbg !49
  %956 = extractelement <4 x float> %949, i64 2, !dbg !48
  %957 = tail call float @llvm.amdgcn.exp2.f32(float %956), !dbg !48
  %local.reduce125 = fadd float %957, %local.reduce121, !dbg !50, !frisk.local_reduction !5
  %958 = insertelement <4 x float> %955, float %957, i64 2, !dbg !49
  %959 = extractelement <4 x float> %949, i64 3, !dbg !48
  %960 = tail call float @llvm.amdgcn.exp2.f32(float %959), !dbg !48
  %local.reduce126 = fadd float %960, %local.reduce122, !dbg !50, !frisk.local_reduction !5
  %961 = insertelement <4 x float> %958, float %960, i64 3, !dbg !49
  %math.result459 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %961) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02184 = extractelement <4 x float> %math.result459, i64 0, !dbg !49
  %.cvtpack.f12185 = extractelement <4 x float> %math.result459, i64 1, !dbg !49
  %.cvtpack.f22186 = extractelement <4 x float> %math.result459, i64 2, !dbg !49
  %.cvtpack.f32187 = extractelement <4 x float> %math.result459, i64 3, !dbg !49
  %.cvtpack.p02188 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02184, float %.cvtpack.f12185) #12, !dbg !49
  %.cvtpack.p12189 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22186, float %.cvtpack.f32187) #12, !dbg !49
  %.cvtpack.v02190 = insertelement <2 x i32> poison, i32 %.cvtpack.p02188, i64 0, !dbg !49
  %.cvtpack.v12191 = insertelement <2 x i32> %.cvtpack.v02190, i32 %.cvtpack.p12189, i64 1, !dbg !49
  %962 = bitcast <2 x i32> %.cvtpack.v12191 to <4 x half>, !dbg !49
  %math.packed460 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %962) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %931, <4 x float> %929) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %947, <4 x float> %930) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %965 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %966 = fsub <4 x float> %707, %701, !dbg !42
  %967 = extractelement <4 x float> %966, i64 0, !dbg !48
  %968 = tail call float @llvm.amdgcn.exp2.f32(float %967), !dbg !48
  %local.reduce158 = fadd float %968, %local.reduce154, !dbg !50, !frisk.local_reduction !5
  %969 = insertelement <4 x float> poison, float %968, i64 0, !dbg !49
  %970 = extractelement <4 x float> %966, i64 1, !dbg !48
  %971 = tail call float @llvm.amdgcn.exp2.f32(float %970), !dbg !48
  %local.reduce159 = fadd float %971, %local.reduce155, !dbg !50, !frisk.local_reduction !5
  %972 = insertelement <4 x float> %969, float %971, i64 1, !dbg !49
  %973 = extractelement <4 x float> %966, i64 2, !dbg !48
  %974 = tail call float @llvm.amdgcn.exp2.f32(float %973), !dbg !48
  %local.reduce160 = fadd float %974, %local.reduce156, !dbg !50, !frisk.local_reduction !5
  %975 = insertelement <4 x float> %972, float %974, i64 2, !dbg !49
  %976 = extractelement <4 x float> %966, i64 3, !dbg !48
  %977 = tail call float @llvm.amdgcn.exp2.f32(float %976), !dbg !48
  %local.reduce161 = fadd float %977, %local.reduce157, !dbg !50, !frisk.local_reduction !5
  %978 = insertelement <4 x float> %975, float %977, i64 3, !dbg !49
  %math.result445 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %978) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02192 = extractelement <4 x float> %math.result445, i64 0, !dbg !49
  %.cvtpack.f12193 = extractelement <4 x float> %math.result445, i64 1, !dbg !49
  %.cvtpack.f22194 = extractelement <4 x float> %math.result445, i64 2, !dbg !49
  %.cvtpack.f32195 = extractelement <4 x float> %math.result445, i64 3, !dbg !49
  %.cvtpack.p02196 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02192, float %.cvtpack.f12193) #12, !dbg !49
  %.cvtpack.p12197 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22194, float %.cvtpack.f32195) #12, !dbg !49
  %.cvtpack.v02198 = insertelement <2 x i32> poison, i32 %.cvtpack.p02196, i64 0, !dbg !49
  %.cvtpack.v12199 = insertelement <2 x i32> %.cvtpack.v02198, i32 %.cvtpack.p12197, i64 1, !dbg !49
  %979 = bitcast <2 x i32> %.cvtpack.v12199 to <4 x half>, !dbg !49
  %math.packed446 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %979) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %965, <4 x float> %946) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %981 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %981, <4 x float> %948) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %983 = fsub <4 x float> %716, %710, !dbg !42
  %984 = extractelement <4 x float> %983, i64 0, !dbg !48
  %985 = tail call float @llvm.amdgcn.exp2.f32(float %984), !dbg !48
  %local.reduce127 = fadd float %985, %local.reduce123, !dbg !50, !frisk.local_reduction !5
  %986 = insertelement <4 x float> poison, float %985, i64 0, !dbg !49
  %987 = extractelement <4 x float> %983, i64 1, !dbg !48
  %988 = tail call float @llvm.amdgcn.exp2.f32(float %987), !dbg !48
  %local.reduce128 = fadd float %988, %local.reduce124, !dbg !50, !frisk.local_reduction !5
  %989 = insertelement <4 x float> %986, float %988, i64 1, !dbg !49
  %990 = extractelement <4 x float> %983, i64 2, !dbg !48
  %991 = tail call float @llvm.amdgcn.exp2.f32(float %990), !dbg !48
  %local.reduce129 = fadd float %991, %local.reduce125, !dbg !50, !frisk.local_reduction !5
  %992 = insertelement <4 x float> %989, float %991, i64 2, !dbg !49
  %993 = extractelement <4 x float> %983, i64 3, !dbg !48
  %994 = tail call float @llvm.amdgcn.exp2.f32(float %993), !dbg !48
  %local.reduce130 = fadd float %994, %local.reduce126, !dbg !50, !frisk.local_reduction !5
  %995 = insertelement <4 x float> %992, float %994, i64 3, !dbg !49
  %math.result461 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %995) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02200 = extractelement <4 x float> %math.result461, i64 0, !dbg !49
  %.cvtpack.f12201 = extractelement <4 x float> %math.result461, i64 1, !dbg !49
  %.cvtpack.f22202 = extractelement <4 x float> %math.result461, i64 2, !dbg !49
  %.cvtpack.f32203 = extractelement <4 x float> %math.result461, i64 3, !dbg !49
  %.cvtpack.p02204 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02200, float %.cvtpack.f12201) #12, !dbg !49
  %.cvtpack.p12205 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22202, float %.cvtpack.f32203) #12, !dbg !49
  %.cvtpack.v02206 = insertelement <2 x i32> poison, i32 %.cvtpack.p02204, i64 0, !dbg !49
  %.cvtpack.v12207 = insertelement <2 x i32> %.cvtpack.v02206, i32 %.cvtpack.p12205, i64 1, !dbg !49
  %996 = bitcast <2 x i32> %.cvtpack.v12207 to <4 x half>, !dbg !49
  %math.packed462 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %996) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %997 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %965, <4 x float> %963) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %981, <4 x float> %964) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %999 = bitcast <4 x i32> %760 to <8 x half>, !dbg !18
  %1000 = shufflevector <8 x half> %999, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1001 = fsub <4 x float> %708, %701, !dbg !42
  %1002 = extractelement <4 x float> %1001, i64 0, !dbg !48
  %1003 = tail call float @llvm.amdgcn.exp2.f32(float %1002), !dbg !48
  %local.reduce162 = fadd float %1003, %local.reduce158, !dbg !50, !frisk.local_reduction !5
  %1004 = insertelement <4 x float> poison, float %1003, i64 0, !dbg !49
  %1005 = extractelement <4 x float> %1001, i64 1, !dbg !48
  %1006 = tail call float @llvm.amdgcn.exp2.f32(float %1005), !dbg !48
  %local.reduce163 = fadd float %1006, %local.reduce159, !dbg !50, !frisk.local_reduction !5
  %local.reduce166 = fadd float %local.reduce162, %local.reduce163, !dbg !50, !frisk.local_reduction !5
  %1007 = insertelement <4 x float> %1004, float %1006, i64 1, !dbg !49
  %1008 = extractelement <4 x float> %1001, i64 2, !dbg !48
  %1009 = tail call float @llvm.amdgcn.exp2.f32(float %1008), !dbg !48
  %local.reduce164 = fadd float %1009, %local.reduce160, !dbg !50, !frisk.local_reduction !5
  %1010 = insertelement <4 x float> %1007, float %1009, i64 2, !dbg !49
  %1011 = extractelement <4 x float> %1001, i64 3, !dbg !48
  %1012 = tail call float @llvm.amdgcn.exp2.f32(float %1011), !dbg !48
  %local.reduce165 = fadd float %1012, %local.reduce161, !dbg !50, !frisk.local_reduction !5
  %local.reduce167 = fadd float %local.reduce164, %local.reduce165, !dbg !50, !frisk.local_reduction !5
  %1013 = fadd float %local.reduce166, %local.reduce167, !dbg !50, !frisk.local_reduction !5
  %1014 = fadd float %725, %1013, !dbg !50
  %1015 = insertelement <1 x float> poison, float %1014, i64 0, !dbg !12
  %1016 = insertvalue [2 x <1 x float>] poison, <1 x float> %1015, 0, !dbg !12
  %1017 = insertelement <4 x float> %1010, float %1012, i64 3, !dbg !49
  %math.result447 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1017) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02208 = extractelement <4 x float> %math.result447, i64 0, !dbg !49
  %.cvtpack.f12209 = extractelement <4 x float> %math.result447, i64 1, !dbg !49
  %.cvtpack.f22210 = extractelement <4 x float> %math.result447, i64 2, !dbg !49
  %.cvtpack.f32211 = extractelement <4 x float> %math.result447, i64 3, !dbg !49
  %.cvtpack.p02212 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02208, float %.cvtpack.f12209) #12, !dbg !49
  %.cvtpack.p12213 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22210, float %.cvtpack.f32211) #12, !dbg !49
  %.cvtpack.v02214 = insertelement <2 x i32> poison, i32 %.cvtpack.p02212, i64 0, !dbg !49
  %.cvtpack.v12215 = insertelement <2 x i32> %.cvtpack.v02214, i32 %.cvtpack.p12213, i64 1, !dbg !49
  %1018 = bitcast <2 x i32> %.cvtpack.v12215 to <4 x half>, !dbg !49
  %math.packed448 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1018) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %1019 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1000, <4 x float> %980) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1020 = shufflevector <8 x half> %999, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1021 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1020, <4 x float> %982) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1022 = fsub <4 x float> %717, %710, !dbg !42
  %1023 = extractelement <4 x float> %1022, i64 0, !dbg !48
  %1024 = tail call float @llvm.amdgcn.exp2.f32(float %1023), !dbg !48
  %local.reduce131 = fadd float %1024, %local.reduce127, !dbg !50, !frisk.local_reduction !5
  %1025 = insertelement <4 x float> poison, float %1024, i64 0, !dbg !49
  %1026 = extractelement <4 x float> %1022, i64 1, !dbg !48
  %1027 = tail call float @llvm.amdgcn.exp2.f32(float %1026), !dbg !48
  %local.reduce132 = fadd float %1027, %local.reduce128, !dbg !50, !frisk.local_reduction !5
  %local.reduce135 = fadd float %local.reduce131, %local.reduce132, !dbg !50, !frisk.local_reduction !5
  %1028 = insertelement <4 x float> %1025, float %1027, i64 1, !dbg !49
  %1029 = extractelement <4 x float> %1022, i64 2, !dbg !48
  %1030 = tail call float @llvm.amdgcn.exp2.f32(float %1029), !dbg !48
  %local.reduce133 = fadd float %1030, %local.reduce129, !dbg !50, !frisk.local_reduction !5
  %1031 = insertelement <4 x float> %1028, float %1030, i64 2, !dbg !49
  %1032 = extractelement <4 x float> %1022, i64 3, !dbg !48
  %1033 = tail call float @llvm.amdgcn.exp2.f32(float %1032), !dbg !48
  %local.reduce134 = fadd float %1033, %local.reduce130, !dbg !50, !frisk.local_reduction !5
  %local.reduce136 = fadd float %local.reduce133, %local.reduce134, !dbg !50, !frisk.local_reduction !5
  %1034 = fadd float %local.reduce135, %local.reduce136, !dbg !50, !frisk.local_reduction !5
  %1035 = fadd float %733, %1034, !dbg !50
  %1036 = insertelement <1 x float> poison, float %1035, i64 0, !dbg !12
  %1037 = insertvalue [2 x <1 x float>] %1016, <1 x float> %1036, 1, !dbg !12
  %1038 = insertelement <4 x float> %1031, float %1033, i64 3, !dbg !49
  %math.result463 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1038) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02216 = extractelement <4 x float> %math.result463, i64 0, !dbg !49
  %.cvtpack.f12217 = extractelement <4 x float> %math.result463, i64 1, !dbg !49
  %.cvtpack.f22218 = extractelement <4 x float> %math.result463, i64 2, !dbg !49
  %.cvtpack.f32219 = extractelement <4 x float> %math.result463, i64 3, !dbg !49
  %.cvtpack.p02220 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02216, float %.cvtpack.f12217) #12, !dbg !49
  %.cvtpack.p12221 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22218, float %.cvtpack.f32219) #12, !dbg !49
  %.cvtpack.v02222 = insertelement <2 x i32> poison, i32 %.cvtpack.p02220, i64 0, !dbg !49
  %.cvtpack.v12223 = insertelement <2 x i32> %.cvtpack.v02222, i32 %.cvtpack.p12221, i64 1, !dbg !49
  %1039 = bitcast <2 x i32> %.cvtpack.v12223 to <4 x half>, !dbg !49
  %math.packed464 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1039) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %1040 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1000, <4 x float> %997) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1041 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1020, <4 x float> %998) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1042 = extractvalue [2 x <8 x float>] %156, 0, !dbg !45
  %1043 = shufflevector <8 x float> %1042, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1044 = fmul <4 x float> %1043, %722, !dbg !45
  %1045 = shufflevector <8 x float> %1042, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1046 = fmul <4 x float> %1045, %722, !dbg !45
  %1047 = extractvalue [2 x <8 x float>] %156, 1, !dbg !45
  %1048 = shufflevector <8 x float> %1047, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1049 = fmul <4 x float> %1048, %730, !dbg !45
  %1050 = shufflevector <8 x float> %1047, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1051 = fmul <4 x float> %1050, %730, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1552 = or disjoint i64 %425, 64, !dbg !17
  %vdb.address1553 = add i64 %vdb.address1552, %424, !dbg !17
  %vdb.address1554 = trunc i64 %vdb.address1553 to i32, !dbg !17
  %vdb.address1555 = shl i32 %vdb.address1554, 1, !dbg !17
  %vdb.address1563 = or i32 %.tr1512, 16512, !dbg !17
  %1052 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %vdb.address1555, <4 x i32> %dma.buffer.rsrc1411) #9, !dbg !17
  %1053 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %vdb.address1563, <4 x i32> %dma.buffer.rsrc1411) #9, !dbg !17
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local999, i64 8192, !dbg !18
  %1054 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1055 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !18
  %vdb.read1545 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 8448, !dbg !18
  %1056 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1545, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1057 = bitcast <4 x i32> %1056 to <8 x half>, !dbg !18
  %vdb.read1546 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 8704, !dbg !18
  %1058 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1546, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1059 = bitcast <4 x i32> %1058 to <8 x half>, !dbg !18
  %vdb.read1547 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 8960, !dbg !18
  %1060 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1547, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1061 = bitcast <4 x i32> %1060 to <8 x half>, !dbg !18
  %vdb.read1548 = getelementptr i8, ptr addrspace(3) %address.local999, i64 12288, !dbg !18
  %1062 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1548, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !18
  %vdb.read1549 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 12544, !dbg !18
  %1064 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1549, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !18
  %vdb.read1550 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 12800, !dbg !18
  %1066 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1067 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !18
  %vdb.read1551 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 13056, !dbg !18
  %1068 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1069 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1070 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1069, <4 x float> %1044) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1071 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1072 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1071, <4 x float> %1046) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1073 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1069, <4 x float> %1049) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1074 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1071, <4 x float> %1051) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1075 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1075, <4 x float> %1070) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1077 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1077, <4 x float> %1072) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1079 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1075, <4 x float> %1073) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1077, <4 x float> %1074) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1081 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1081, <4 x float> %1076) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1083 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1083, <4 x float> %1078) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1085 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1081, <4 x float> %1079) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1083, <4 x float> %1080) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1087 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1087, <4 x float> %1082) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1089 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1089, <4 x float> %1084) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1091 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1087, <4 x float> %1085) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1089, <4 x float> %1086) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1093 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1093, <4 x float> %1088) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1095 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1095, <4 x float> %1090) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1097 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1093, <4 x float> %1091) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1098 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1095, <4 x float> %1092) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1099 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1100 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1099, <4 x float> %1094) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1101 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1101, <4 x float> %1096) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1103 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1099, <4 x float> %1097) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1101, <4 x float> %1098) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1105 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1105, <4 x float> %1100) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1107 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1108 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1107, <4 x float> %1102) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1109 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1105, <4 x float> %1103) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1110 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1107, <4 x float> %1104) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1111 = bitcast <4 x i32> %1068 to <8 x half>, !dbg !18
  %1112 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1112, <4 x float> %1106) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1114 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1115 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1114, <4 x float> %1108) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1112, <4 x float> %1109) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1114, <4 x float> %1110) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1118 = extractvalue [2 x <8 x float>] %155, 0, !dbg !45
  %1119 = shufflevector <8 x float> %1118, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1120 = fmul <4 x float> %1119, %722, !dbg !45
  %1121 = shufflevector <8 x float> %1118, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1122 = fmul <4 x float> %1121, %722, !dbg !45
  %1123 = extractvalue [2 x <8 x float>] %155, 1, !dbg !45
  %1124 = shufflevector <8 x float> %1123, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1125 = fmul <4 x float> %1124, %730, !dbg !45
  %1126 = shufflevector <8 x float> %1123, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1127 = fmul <4 x float> %1126, %730, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1568 = or disjoint i64 %425, 96, !dbg !17
  %vdb.address1569 = add i64 %vdb.address1568, %424, !dbg !17
  %vdb.address1570 = trunc i64 %vdb.address1569 to i32, !dbg !17
  %vdb.address1571 = shl i32 %vdb.address1570, 1, !dbg !17
  %vdb.address1579 = or i32 %.tr1512, 16576, !dbg !17
  %1128 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1571, <4 x i32> %dma.buffer.rsrc1411) #9, !dbg !17
  %1129 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1544, i32 %vdb.address1579, <4 x i32> %dma.buffer.rsrc1411) #9, !dbg !17
  %1130 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local999, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1131 = bitcast <4 x i32> %1130 to <8 x half>, !dbg !18
  %1132 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1009, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1133 = bitcast <4 x i32> %1132 to <8 x half>, !dbg !18
  %1134 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1135 = bitcast <4 x i32> %1134 to <8 x half>, !dbg !18
  %1136 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1137 = bitcast <4 x i32> %1136 to <8 x half>, !dbg !18
  %1138 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1019, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1139 = bitcast <4 x i32> %1138 to <8 x half>, !dbg !18
  %1140 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1141 = bitcast <4 x i32> %1140 to <8 x half>, !dbg !18
  %1142 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1143 = bitcast <4 x i32> %1142 to <8 x half>, !dbg !18
  %1144 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1145 = shufflevector <8 x half> %1131, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1145, <4 x float> %1120) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1147 = shufflevector <8 x half> %1131, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1147, <4 x float> %1122) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1145, <4 x float> %1125) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1147, <4 x float> %1127) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1151 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1151, <4 x float> %1146) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1153 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1153, <4 x float> %1148) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1151, <4 x float> %1149) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1153, <4 x float> %1150) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1157 = shufflevector <8 x half> %1135, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1157, <4 x float> %1152) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1159 = shufflevector <8 x half> %1135, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1159, <4 x float> %1154) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1157, <4 x float> %1155) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1159, <4 x float> %1156) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1163 = shufflevector <8 x half> %1137, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1163, <4 x float> %1158) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1165 = shufflevector <8 x half> %1137, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1165, <4 x float> %1160) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1163, <4 x float> %1161) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1165, <4 x float> %1162) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1169 = shufflevector <8 x half> %1139, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1169, <4 x float> %1164) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1171 = shufflevector <8 x half> %1139, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1171, <4 x float> %1166) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1169, <4 x float> %1167) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1171, <4 x float> %1168) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1175 = shufflevector <8 x half> %1141, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1175, <4 x float> %1170) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1177 = shufflevector <8 x half> %1141, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1177, <4 x float> %1172) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1175, <4 x float> %1173) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1177, <4 x float> %1174) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1181 = shufflevector <8 x half> %1143, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1181, <4 x float> %1176) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1183 = shufflevector <8 x half> %1143, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1183, <4 x float> %1178) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1181, <4 x float> %1179) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1183, <4 x float> %1180) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1187 = bitcast <4 x i32> %1144 to <8 x half>, !dbg !18
  %1188 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1188, <4 x float> %1182) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1190 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1190, <4 x float> %1184) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1188, <4 x float> %1185) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1190, <4 x float> %1186) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1194 = extractvalue [2 x <8 x float>] %154, 0, !dbg !45
  %1195 = shufflevector <8 x float> %1194, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1196 = fmul <4 x float> %1195, %722, !dbg !45
  %1197 = shufflevector <8 x float> %1194, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1198 = fmul <4 x float> %1197, %722, !dbg !45
  %1199 = extractvalue [2 x <8 x float>] %154, 1, !dbg !45
  %1200 = shufflevector <8 x float> %1199, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1201 = fmul <4 x float> %1200, %730, !dbg !45
  %1202 = shufflevector <8 x float> %1199, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1203 = fmul <4 x float> %1202, %730, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1204 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1205 = bitcast <4 x i32> %1204 to <8 x half>, !dbg !18
  %1206 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1545, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1207 = bitcast <4 x i32> %1206 to <8 x half>, !dbg !18
  %1208 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1546, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1209 = bitcast <4 x i32> %1208 to <8 x half>, !dbg !18
  %1210 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1547, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1211 = bitcast <4 x i32> %1210 to <8 x half>, !dbg !18
  %1212 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1548, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1213 = bitcast <4 x i32> %1212 to <8 x half>, !dbg !18
  %1214 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1549, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1215 = bitcast <4 x i32> %1214 to <8 x half>, !dbg !18
  %1216 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1217 = bitcast <4 x i32> %1216 to <8 x half>, !dbg !18
  %1218 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1219 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1219, <4 x float> %1196) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1221 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1221, <4 x float> %1198) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1219, <4 x float> %1201) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1221, <4 x float> %1203) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1225 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1225, <4 x float> %1220) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1227 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1227, <4 x float> %1222) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1225, <4 x float> %1223) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1227, <4 x float> %1224) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1231 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1231, <4 x float> %1226) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1233 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1233, <4 x float> %1228) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1231, <4 x float> %1229) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1233, <4 x float> %1230) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1237 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1237, <4 x float> %1232) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1239 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1239, <4 x float> %1234) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1237, <4 x float> %1235) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1239, <4 x float> %1236) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1243 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1243, <4 x float> %1238) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1245 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1245, <4 x float> %1240) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1243, <4 x float> %1241) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1245, <4 x float> %1242) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1249 = shufflevector <8 x half> %1215, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1249, <4 x float> %1244) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1251 = shufflevector <8 x half> %1215, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1251, <4 x float> %1246) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1249, <4 x float> %1247) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1251, <4 x float> %1248) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1255 = shufflevector <8 x half> %1217, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1255, <4 x float> %1250) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1257 = shufflevector <8 x half> %1217, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1257, <4 x float> %1252) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1255, <4 x float> %1253) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1257, <4 x float> %1254) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1261 = bitcast <4 x i32> %1218 to <8 x half>, !dbg !18
  %1262 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1262, <4 x float> %1256) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1264 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1264, <4 x float> %1258) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1262, <4 x float> %1259) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1264, <4 x float> %1260) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %1268 = shufflevector <4 x float> %1019, <4 x float> %1021, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1269 = insertvalue [2 x <8 x float>] poison, <8 x float> %1268, 0, !dbg !12
  %1270 = shufflevector <4 x float> %1040, <4 x float> %1041, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1271 = insertvalue [2 x <8 x float>] %1269, <8 x float> %1270, 1, !dbg !12
  %1272 = insertvalue [2 x <1 x float>] poison, <1 x float> %696, 0, !dbg !12
  %1273 = insertvalue [2 x <1 x float>] %1272, <1 x float> %699, 1, !dbg !12
  %1274 = shufflevector <4 x float> %1113, <4 x float> %1115, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1275 = insertvalue [2 x <8 x float>] poison, <8 x float> %1274, 0, !dbg !12
  %1276 = shufflevector <4 x float> %1116, <4 x float> %1117, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1277 = insertvalue [2 x <8 x float>] %1275, <8 x float> %1276, 1, !dbg !12
  %1278 = shufflevector <4 x float> %1189, <4 x float> %1191, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1279 = insertvalue [2 x <8 x float>] poison, <8 x float> %1278, 0, !dbg !12
  %1280 = shufflevector <4 x float> %1192, <4 x float> %1193, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1281 = insertvalue [2 x <8 x float>] %1279, <8 x float> %1280, 1, !dbg !12
  %1282 = shufflevector <4 x float> %1263, <4 x float> %1265, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1283 = insertvalue [2 x <8 x float>] poison, <8 x float> %1282, 0, !dbg !12
  %1284 = shufflevector <4 x float> %1266, <4 x float> %1267, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1285 = insertvalue [2 x <8 x float>] %1283, <8 x float> %1284, 1, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %1286 = add nuw nsw i64 %160, 1, !dbg !12
  %1287 = icmp slt i64 %1286, %114, !dbg !12
  br i1 %1287, label %153, label %._crit_edge, !dbg !12

._crit_edge:                                      ; preds = %153, %4
  %.lcssa7 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1271, %153 ], !dbg !12
  %.lcssa6 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %1037, %153 ], !dbg !12
  %.lcssa5 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1277, %153 ], !dbg !12
  %.lcssa4 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1281, %153 ], !dbg !12
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1285, %153 ], !dbg !12
  %lane.sum.field233 = extractvalue [2 x <1 x float>] %.lcssa6, 1, !dbg !12
  %lane.sum.local234 = extractelement <1 x float> %lane.sum.field233, i64 0, !dbg !12
  %1288 = shl nuw nsw i32 %7, 2, !dbg !12
  %1289 = xor i32 %1288, 64, !dbg !12
  %1290 = bitcast float %lane.sum.local234 to i32, !dbg !12
  %lane.sum.shuffle237 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1289, i32 %1290), !dbg !12
  %1291 = bitcast i32 %lane.sum.shuffle237 to float, !dbg !12
  %lane.sum.final238 = fadd float %lane.sum.local234, %1291, !dbg !12
  %1292 = xor i32 %1288, 128, !dbg !12
  %1293 = bitcast float %lane.sum.final238 to i32, !dbg !12
  %lane.sum.shuffle239 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1292, i32 %1293), !dbg !12
  %1294 = bitcast i32 %lane.sum.shuffle239 to float, !dbg !12
  %lane.sum.final240 = fadd float %lane.sum.final238, %1294, !dbg !12
  %1295 = insertelement <1 x float> poison, float %lane.sum.final240, i64 0, !dbg !12
  %lane.sum.field = extractvalue [2 x <1 x float>] %.lcssa6, 0, !dbg !12
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !12
  %1296 = bitcast float %lane.sum.local to i32, !dbg !12
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1289, i32 %1296), !dbg !12
  %1297 = bitcast i32 %lane.sum.shuffle to float, !dbg !12
  %lane.sum.final = fadd float %lane.sum.local, %1297, !dbg !12
  %1298 = bitcast float %lane.sum.final to i32, !dbg !12
  %lane.sum.shuffle231 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1292, i32 %1298), !dbg !12
  %1299 = bitcast i32 %lane.sum.shuffle231 to float, !dbg !12
  %lane.sum.final232 = fadd float %lane.sum.final, %1299, !dbg !12
  %1300 = insertelement <1 x float> poison, float %lane.sum.final232, i64 0, !dbg !12
  %1301 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %1301, 0, !dbg !12
  %1302 = and i32 %kdma.tid, 32, !dbg !12
  %.not40 = icmp eq i32 %1302, 0, !dbg !12
  %1303 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !12
  %1304 = extractelement <8 x float> %1303, i64 0, !dbg !12
  %1305 = extractelement <8 x float> %1303, i64 1, !dbg !12
  %1306 = extractelement <8 x float> %1303, i64 2, !dbg !12
  %1307 = extractelement <8 x float> %1303, i64 3, !dbg !12
  %1308 = select i1 %.not, float %1305, float %1304, !dbg !12
  %1309 = bitcast float %1308 to i32, !dbg !12
  %1310 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1309), !dbg !12
  %1311 = bitcast i32 %1310 to float, !dbg !12
  %1312 = select i1 %.not, float %1304, float %1311, !dbg !12
  %1313 = select i1 %.not, float %1311, float %1305, !dbg !12
  %1314 = select i1 %.not, float %1307, float %1306, !dbg !12
  %1315 = bitcast float %1314 to i32, !dbg !12
  %1316 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1315), !dbg !12
  %1317 = bitcast i32 %1316 to float, !dbg !12
  %1318 = select i1 %.not, float %1306, float %1317, !dbg !12
  %1319 = select i1 %.not, float %1317, float %1307, !dbg !12
  %1320 = select i1 %.not40, float %1318, float %1312, !dbg !12
  %1321 = bitcast float %1320 to i32, !dbg !12
  %1322 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1321), !dbg !12
  %1323 = bitcast i32 %1322 to float, !dbg !12
  %1324 = select i1 %.not40, float %1319, float %1313, !dbg !12
  %1325 = bitcast float %1324 to i32, !dbg !12
  %1326 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1325), !dbg !12
  %1327 = bitcast i32 %1326 to float, !dbg !12
  %1328 = insertelement <4 x i1> poison, i1 %.not40, i64 0, !dbg !12
  %1329 = shufflevector <4 x i1> %1328, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %1330 = insertelement <4 x float> poison, float %1312, i64 0, !dbg !12
  %1331 = insertelement <4 x float> %1330, float %1313, i64 1, !dbg !12
  %1332 = insertelement <4 x float> %1331, float %1323, i64 2, !dbg !12
  %1333 = insertelement <4 x float> %1332, float %1327, i64 3, !dbg !12
  %1334 = insertelement <4 x float> poison, float %1323, i64 0, !dbg !12
  %1335 = insertelement <4 x float> %1334, float %1327, i64 1, !dbg !12
  %1336 = insertelement <4 x float> %1335, float %1318, i64 2, !dbg !12
  %1337 = insertelement <4 x float> %1336, float %1319, i64 3, !dbg !12
  %1338 = select <4 x i1> %1329, <4 x float> %1333, <4 x float> %1337, !dbg !12
  %1339 = extractelement <8 x float> %1303, i64 4, !dbg !12
  %1340 = extractelement <8 x float> %1303, i64 5, !dbg !12
  %1341 = extractelement <8 x float> %1303, i64 6, !dbg !12
  %1342 = extractelement <8 x float> %1303, i64 7, !dbg !12
  %1343 = select i1 %.not, float %1340, float %1339, !dbg !12
  %1344 = bitcast float %1343 to i32, !dbg !12
  %1345 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1344), !dbg !12
  %1346 = bitcast i32 %1345 to float, !dbg !12
  %1347 = select i1 %.not, float %1339, float %1346, !dbg !12
  %1348 = select i1 %.not, float %1346, float %1340, !dbg !12
  %1349 = select i1 %.not, float %1342, float %1341, !dbg !12
  %1350 = bitcast float %1349 to i32, !dbg !12
  %1351 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1350), !dbg !12
  %1352 = bitcast i32 %1351 to float, !dbg !12
  %1353 = select i1 %.not, float %1341, float %1352, !dbg !12
  %1354 = select i1 %.not, float %1352, float %1342, !dbg !12
  %1355 = select i1 %.not40, float %1353, float %1347, !dbg !12
  %1356 = bitcast float %1355 to i32, !dbg !12
  %1357 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1356), !dbg !12
  %1358 = bitcast i32 %1357 to float, !dbg !12
  %1359 = select i1 %.not40, float %1354, float %1348, !dbg !12
  %1360 = bitcast float %1359 to i32, !dbg !12
  %1361 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1360), !dbg !12
  %1362 = bitcast i32 %1361 to float, !dbg !12
  %1363 = insertelement <4 x float> poison, float %1347, i64 0, !dbg !12
  %1364 = insertelement <4 x float> %1363, float %1348, i64 1, !dbg !12
  %1365 = insertelement <4 x float> %1364, float %1358, i64 2, !dbg !12
  %1366 = insertelement <4 x float> %1365, float %1362, i64 3, !dbg !12
  %1367 = insertelement <4 x float> poison, float %1358, i64 0, !dbg !12
  %1368 = insertelement <4 x float> %1367, float %1362, i64 1, !dbg !12
  %1369 = insertelement <4 x float> %1368, float %1353, i64 2, !dbg !12
  %1370 = insertelement <4 x float> %1369, float %1354, i64 3, !dbg !12
  %1371 = select <4 x i1> %1329, <4 x float> %1366, <4 x float> %1370, !dbg !12
  %1372 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !12
  %1373 = extractelement <8 x float> %1372, i64 0, !dbg !12
  %1374 = extractelement <8 x float> %1372, i64 1, !dbg !12
  %1375 = extractelement <8 x float> %1372, i64 2, !dbg !12
  %1376 = extractelement <8 x float> %1372, i64 3, !dbg !12
  %1377 = select i1 %.not, float %1374, float %1373, !dbg !12
  %1378 = bitcast float %1377 to i32, !dbg !12
  %1379 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1378), !dbg !12
  %1380 = bitcast i32 %1379 to float, !dbg !12
  %1381 = select i1 %.not, float %1373, float %1380, !dbg !12
  %1382 = select i1 %.not, float %1380, float %1374, !dbg !12
  %1383 = select i1 %.not, float %1376, float %1375, !dbg !12
  %1384 = bitcast float %1383 to i32, !dbg !12
  %1385 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1384), !dbg !12
  %1386 = bitcast i32 %1385 to float, !dbg !12
  %1387 = select i1 %.not, float %1375, float %1386, !dbg !12
  %1388 = select i1 %.not, float %1386, float %1376, !dbg !12
  %1389 = select i1 %.not40, float %1387, float %1381, !dbg !12
  %1390 = bitcast float %1389 to i32, !dbg !12
  %1391 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1390), !dbg !12
  %1392 = bitcast i32 %1391 to float, !dbg !12
  %1393 = select i1 %.not40, float %1388, float %1382, !dbg !12
  %1394 = bitcast float %1393 to i32, !dbg !12
  %1395 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1394), !dbg !12
  %1396 = bitcast i32 %1395 to float, !dbg !12
  %1397 = insertelement <4 x float> poison, float %1381, i64 0, !dbg !12
  %1398 = insertelement <4 x float> %1397, float %1382, i64 1, !dbg !12
  %1399 = insertelement <4 x float> %1398, float %1392, i64 2, !dbg !12
  %1400 = insertelement <4 x float> %1399, float %1396, i64 3, !dbg !12
  %1401 = insertelement <4 x float> poison, float %1392, i64 0, !dbg !12
  %1402 = insertelement <4 x float> %1401, float %1396, i64 1, !dbg !12
  %1403 = insertelement <4 x float> %1402, float %1387, i64 2, !dbg !12
  %1404 = insertelement <4 x float> %1403, float %1388, i64 3, !dbg !12
  %1405 = select <4 x i1> %1329, <4 x float> %1400, <4 x float> %1404, !dbg !12
  %1406 = extractelement <8 x float> %1372, i64 4, !dbg !12
  %1407 = extractelement <8 x float> %1372, i64 5, !dbg !12
  %1408 = extractelement <8 x float> %1372, i64 6, !dbg !12
  %1409 = extractelement <8 x float> %1372, i64 7, !dbg !12
  %1410 = select i1 %.not, float %1407, float %1406, !dbg !12
  %1411 = bitcast float %1410 to i32, !dbg !12
  %1412 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1411), !dbg !12
  %1413 = bitcast i32 %1412 to float, !dbg !12
  %1414 = select i1 %.not, float %1406, float %1413, !dbg !12
  %1415 = select i1 %.not, float %1413, float %1407, !dbg !12
  %1416 = select i1 %.not, float %1409, float %1408, !dbg !12
  %1417 = bitcast float %1416 to i32, !dbg !12
  %1418 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1417), !dbg !12
  %1419 = bitcast i32 %1418 to float, !dbg !12
  %1420 = select i1 %.not, float %1408, float %1419, !dbg !12
  %1421 = select i1 %.not, float %1419, float %1409, !dbg !12
  %1422 = select i1 %.not40, float %1420, float %1414, !dbg !12
  %1423 = bitcast float %1422 to i32, !dbg !12
  %1424 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1423), !dbg !12
  %1425 = bitcast i32 %1424 to float, !dbg !12
  %1426 = select i1 %.not40, float %1421, float %1415, !dbg !12
  %1427 = bitcast float %1426 to i32, !dbg !12
  %1428 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1427), !dbg !12
  %1429 = bitcast i32 %1428 to float, !dbg !12
  %1430 = insertelement <4 x float> poison, float %1414, i64 0, !dbg !12
  %1431 = insertelement <4 x float> %1430, float %1415, i64 1, !dbg !12
  %1432 = insertelement <4 x float> %1431, float %1425, i64 2, !dbg !12
  %1433 = insertelement <4 x float> %1432, float %1429, i64 3, !dbg !12
  %1434 = insertelement <4 x float> poison, float %1425, i64 0, !dbg !12
  %1435 = insertelement <4 x float> %1434, float %1429, i64 1, !dbg !12
  %1436 = insertelement <4 x float> %1435, float %1420, i64 2, !dbg !12
  %1437 = insertelement <4 x float> %1436, float %1421, i64 3, !dbg !12
  %1438 = select <4 x i1> %1329, <4 x float> %1433, <4 x float> %1437, !dbg !12
  %1439 = extractvalue [2 x <8 x float>] %.lcssa4, 0, !dbg !12
  %1440 = extractelement <8 x float> %1439, i64 0, !dbg !12
  %1441 = extractelement <8 x float> %1439, i64 1, !dbg !12
  %1442 = extractelement <8 x float> %1439, i64 2, !dbg !12
  %1443 = extractelement <8 x float> %1439, i64 3, !dbg !12
  %1444 = select i1 %.not, float %1441, float %1440, !dbg !12
  %1445 = bitcast float %1444 to i32, !dbg !12
  %1446 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1445), !dbg !12
  %1447 = bitcast i32 %1446 to float, !dbg !12
  %1448 = select i1 %.not, float %1440, float %1447, !dbg !12
  %1449 = select i1 %.not, float %1447, float %1441, !dbg !12
  %1450 = select i1 %.not, float %1443, float %1442, !dbg !12
  %1451 = bitcast float %1450 to i32, !dbg !12
  %1452 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1451), !dbg !12
  %1453 = bitcast i32 %1452 to float, !dbg !12
  %1454 = select i1 %.not, float %1442, float %1453, !dbg !12
  %1455 = select i1 %.not, float %1453, float %1443, !dbg !12
  %1456 = select i1 %.not40, float %1454, float %1448, !dbg !12
  %1457 = bitcast float %1456 to i32, !dbg !12
  %1458 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1457), !dbg !12
  %1459 = bitcast i32 %1458 to float, !dbg !12
  %1460 = select i1 %.not40, float %1455, float %1449, !dbg !12
  %1461 = bitcast float %1460 to i32, !dbg !12
  %1462 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1461), !dbg !12
  %1463 = bitcast i32 %1462 to float, !dbg !12
  %1464 = insertelement <4 x float> poison, float %1448, i64 0, !dbg !12
  %1465 = insertelement <4 x float> %1464, float %1449, i64 1, !dbg !12
  %1466 = insertelement <4 x float> %1465, float %1459, i64 2, !dbg !12
  %1467 = insertelement <4 x float> %1466, float %1463, i64 3, !dbg !12
  %1468 = insertelement <4 x float> poison, float %1459, i64 0, !dbg !12
  %1469 = insertelement <4 x float> %1468, float %1463, i64 1, !dbg !12
  %1470 = insertelement <4 x float> %1469, float %1454, i64 2, !dbg !12
  %1471 = insertelement <4 x float> %1470, float %1455, i64 3, !dbg !12
  %1472 = select <4 x i1> %1329, <4 x float> %1467, <4 x float> %1471, !dbg !12
  %1473 = extractelement <8 x float> %1439, i64 4, !dbg !12
  %1474 = extractelement <8 x float> %1439, i64 5, !dbg !12
  %1475 = extractelement <8 x float> %1439, i64 6, !dbg !12
  %1476 = extractelement <8 x float> %1439, i64 7, !dbg !12
  %1477 = select i1 %.not, float %1474, float %1473, !dbg !12
  %1478 = bitcast float %1477 to i32, !dbg !12
  %1479 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1478), !dbg !12
  %1480 = bitcast i32 %1479 to float, !dbg !12
  %1481 = select i1 %.not, float %1473, float %1480, !dbg !12
  %1482 = select i1 %.not, float %1480, float %1474, !dbg !12
  %1483 = select i1 %.not, float %1476, float %1475, !dbg !12
  %1484 = bitcast float %1483 to i32, !dbg !12
  %1485 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1484), !dbg !12
  %1486 = bitcast i32 %1485 to float, !dbg !12
  %1487 = select i1 %.not, float %1475, float %1486, !dbg !12
  %1488 = select i1 %.not, float %1486, float %1476, !dbg !12
  %1489 = select i1 %.not40, float %1487, float %1481, !dbg !12
  %1490 = bitcast float %1489 to i32, !dbg !12
  %1491 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1490), !dbg !12
  %1492 = bitcast i32 %1491 to float, !dbg !12
  %1493 = select i1 %.not40, float %1488, float %1482, !dbg !12
  %1494 = bitcast float %1493 to i32, !dbg !12
  %1495 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1494), !dbg !12
  %1496 = bitcast i32 %1495 to float, !dbg !12
  %1497 = insertelement <4 x float> poison, float %1481, i64 0, !dbg !12
  %1498 = insertelement <4 x float> %1497, float %1482, i64 1, !dbg !12
  %1499 = insertelement <4 x float> %1498, float %1492, i64 2, !dbg !12
  %1500 = insertelement <4 x float> %1499, float %1496, i64 3, !dbg !12
  %1501 = insertelement <4 x float> poison, float %1492, i64 0, !dbg !12
  %1502 = insertelement <4 x float> %1501, float %1496, i64 1, !dbg !12
  %1503 = insertelement <4 x float> %1502, float %1487, i64 2, !dbg !12
  %1504 = insertelement <4 x float> %1503, float %1488, i64 3, !dbg !12
  %1505 = select <4 x i1> %1329, <4 x float> %1500, <4 x float> %1504, !dbg !12
  %1506 = extractvalue [2 x <8 x float>] %.lcssa4, 1, !dbg !12
  %1507 = extractelement <8 x float> %1506, i64 0, !dbg !12
  %1508 = extractelement <8 x float> %1506, i64 1, !dbg !12
  %1509 = extractelement <8 x float> %1506, i64 2, !dbg !12
  %1510 = extractelement <8 x float> %1506, i64 3, !dbg !12
  %1511 = select i1 %.not, float %1508, float %1507, !dbg !12
  %1512 = bitcast float %1511 to i32, !dbg !12
  %1513 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1512), !dbg !12
  %1514 = bitcast i32 %1513 to float, !dbg !12
  %1515 = select i1 %.not, float %1507, float %1514, !dbg !12
  %1516 = select i1 %.not, float %1514, float %1508, !dbg !12
  %1517 = select i1 %.not, float %1510, float %1509, !dbg !12
  %1518 = bitcast float %1517 to i32, !dbg !12
  %1519 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1518), !dbg !12
  %1520 = bitcast i32 %1519 to float, !dbg !12
  %1521 = select i1 %.not, float %1509, float %1520, !dbg !12
  %1522 = select i1 %.not, float %1520, float %1510, !dbg !12
  %1523 = select i1 %.not40, float %1521, float %1515, !dbg !12
  %1524 = bitcast float %1523 to i32, !dbg !12
  %1525 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1524), !dbg !12
  %1526 = bitcast i32 %1525 to float, !dbg !12
  %1527 = select i1 %.not40, float %1522, float %1516, !dbg !12
  %1528 = bitcast float %1527 to i32, !dbg !12
  %1529 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1528), !dbg !12
  %1530 = bitcast i32 %1529 to float, !dbg !12
  %1531 = insertelement <4 x float> poison, float %1515, i64 0, !dbg !12
  %1532 = insertelement <4 x float> %1531, float %1516, i64 1, !dbg !12
  %1533 = insertelement <4 x float> %1532, float %1526, i64 2, !dbg !12
  %1534 = insertelement <4 x float> %1533, float %1530, i64 3, !dbg !12
  %1535 = insertelement <4 x float> poison, float %1526, i64 0, !dbg !12
  %1536 = insertelement <4 x float> %1535, float %1530, i64 1, !dbg !12
  %1537 = insertelement <4 x float> %1536, float %1521, i64 2, !dbg !12
  %1538 = insertelement <4 x float> %1537, float %1522, i64 3, !dbg !12
  %1539 = select <4 x i1> %1329, <4 x float> %1534, <4 x float> %1538, !dbg !12
  %1540 = extractelement <8 x float> %1506, i64 4, !dbg !12
  %1541 = extractelement <8 x float> %1506, i64 5, !dbg !12
  %1542 = extractelement <8 x float> %1506, i64 6, !dbg !12
  %1543 = extractelement <8 x float> %1506, i64 7, !dbg !12
  %1544 = select i1 %.not, float %1541, float %1540, !dbg !12
  %1545 = bitcast float %1544 to i32, !dbg !12
  %1546 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1545), !dbg !12
  %1547 = bitcast i32 %1546 to float, !dbg !12
  %1548 = select i1 %.not, float %1540, float %1547, !dbg !12
  %1549 = select i1 %.not, float %1547, float %1541, !dbg !12
  %1550 = select i1 %.not, float %1543, float %1542, !dbg !12
  %1551 = bitcast float %1550 to i32, !dbg !12
  %1552 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1551), !dbg !12
  %1553 = bitcast i32 %1552 to float, !dbg !12
  %1554 = select i1 %.not, float %1542, float %1553, !dbg !12
  %1555 = select i1 %.not, float %1553, float %1543, !dbg !12
  %1556 = select i1 %.not40, float %1554, float %1548, !dbg !12
  %1557 = bitcast float %1556 to i32, !dbg !12
  %1558 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1557), !dbg !12
  %1559 = bitcast i32 %1558 to float, !dbg !12
  %1560 = select i1 %.not40, float %1555, float %1549, !dbg !12
  %1561 = bitcast float %1560 to i32, !dbg !12
  %1562 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1561), !dbg !12
  %1563 = bitcast i32 %1562 to float, !dbg !12
  %1564 = insertelement <4 x float> poison, float %1548, i64 0, !dbg !12
  %1565 = insertelement <4 x float> %1564, float %1549, i64 1, !dbg !12
  %1566 = insertelement <4 x float> %1565, float %1559, i64 2, !dbg !12
  %1567 = insertelement <4 x float> %1566, float %1563, i64 3, !dbg !12
  %1568 = insertelement <4 x float> poison, float %1559, i64 0, !dbg !12
  %1569 = insertelement <4 x float> %1568, float %1563, i64 1, !dbg !12
  %1570 = insertelement <4 x float> %1569, float %1554, i64 2, !dbg !12
  %1571 = insertelement <4 x float> %1570, float %1555, i64 3, !dbg !12
  %1572 = select <4 x i1> %1329, <4 x float> %1567, <4 x float> %1571, !dbg !12
  %1573 = extractvalue [2 x <8 x float>] %.lcssa5, 0, !dbg !12
  %1574 = extractelement <8 x float> %1573, i64 0, !dbg !12
  %1575 = extractelement <8 x float> %1573, i64 1, !dbg !12
  %1576 = extractelement <8 x float> %1573, i64 2, !dbg !12
  %1577 = extractelement <8 x float> %1573, i64 3, !dbg !12
  %1578 = select i1 %.not, float %1575, float %1574, !dbg !12
  %1579 = bitcast float %1578 to i32, !dbg !12
  %1580 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1579), !dbg !12
  %1581 = bitcast i32 %1580 to float, !dbg !12
  %1582 = select i1 %.not, float %1574, float %1581, !dbg !12
  %1583 = select i1 %.not, float %1581, float %1575, !dbg !12
  %1584 = select i1 %.not, float %1577, float %1576, !dbg !12
  %1585 = bitcast float %1584 to i32, !dbg !12
  %1586 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1585), !dbg !12
  %1587 = bitcast i32 %1586 to float, !dbg !12
  %1588 = select i1 %.not, float %1576, float %1587, !dbg !12
  %1589 = select i1 %.not, float %1587, float %1577, !dbg !12
  %1590 = select i1 %.not40, float %1588, float %1582, !dbg !12
  %1591 = bitcast float %1590 to i32, !dbg !12
  %1592 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1591), !dbg !12
  %1593 = bitcast i32 %1592 to float, !dbg !12
  %1594 = select i1 %.not40, float %1589, float %1583, !dbg !12
  %1595 = bitcast float %1594 to i32, !dbg !12
  %1596 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1595), !dbg !12
  %1597 = bitcast i32 %1596 to float, !dbg !12
  %1598 = insertelement <4 x float> poison, float %1582, i64 0, !dbg !12
  %1599 = insertelement <4 x float> %1598, float %1583, i64 1, !dbg !12
  %1600 = insertelement <4 x float> %1599, float %1593, i64 2, !dbg !12
  %1601 = insertelement <4 x float> %1600, float %1597, i64 3, !dbg !12
  %1602 = insertelement <4 x float> poison, float %1593, i64 0, !dbg !12
  %1603 = insertelement <4 x float> %1602, float %1597, i64 1, !dbg !12
  %1604 = insertelement <4 x float> %1603, float %1588, i64 2, !dbg !12
  %1605 = insertelement <4 x float> %1604, float %1589, i64 3, !dbg !12
  %1606 = select <4 x i1> %1329, <4 x float> %1601, <4 x float> %1605, !dbg !12
  %1607 = extractelement <8 x float> %1573, i64 4, !dbg !12
  %1608 = extractelement <8 x float> %1573, i64 5, !dbg !12
  %1609 = extractelement <8 x float> %1573, i64 6, !dbg !12
  %1610 = extractelement <8 x float> %1573, i64 7, !dbg !12
  %1611 = select i1 %.not, float %1608, float %1607, !dbg !12
  %1612 = bitcast float %1611 to i32, !dbg !12
  %1613 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1612), !dbg !12
  %1614 = bitcast i32 %1613 to float, !dbg !12
  %1615 = select i1 %.not, float %1607, float %1614, !dbg !12
  %1616 = select i1 %.not, float %1614, float %1608, !dbg !12
  %1617 = select i1 %.not, float %1610, float %1609, !dbg !12
  %1618 = bitcast float %1617 to i32, !dbg !12
  %1619 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1618), !dbg !12
  %1620 = bitcast i32 %1619 to float, !dbg !12
  %1621 = select i1 %.not, float %1609, float %1620, !dbg !12
  %1622 = select i1 %.not, float %1620, float %1610, !dbg !12
  %1623 = select i1 %.not40, float %1621, float %1615, !dbg !12
  %1624 = bitcast float %1623 to i32, !dbg !12
  %1625 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1624), !dbg !12
  %1626 = bitcast i32 %1625 to float, !dbg !12
  %1627 = select i1 %.not40, float %1622, float %1616, !dbg !12
  %1628 = bitcast float %1627 to i32, !dbg !12
  %1629 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1628), !dbg !12
  %1630 = bitcast i32 %1629 to float, !dbg !12
  %1631 = insertelement <4 x float> poison, float %1615, i64 0, !dbg !12
  %1632 = insertelement <4 x float> %1631, float %1616, i64 1, !dbg !12
  %1633 = insertelement <4 x float> %1632, float %1626, i64 2, !dbg !12
  %1634 = insertelement <4 x float> %1633, float %1630, i64 3, !dbg !12
  %1635 = insertelement <4 x float> poison, float %1626, i64 0, !dbg !12
  %1636 = insertelement <4 x float> %1635, float %1630, i64 1, !dbg !12
  %1637 = insertelement <4 x float> %1636, float %1621, i64 2, !dbg !12
  %1638 = insertelement <4 x float> %1637, float %1622, i64 3, !dbg !12
  %1639 = select <4 x i1> %1329, <4 x float> %1634, <4 x float> %1638, !dbg !12
  %1640 = extractvalue [2 x <8 x float>] %.lcssa5, 1, !dbg !12
  %1641 = extractelement <8 x float> %1640, i64 0, !dbg !12
  %1642 = extractelement <8 x float> %1640, i64 1, !dbg !12
  %1643 = extractelement <8 x float> %1640, i64 2, !dbg !12
  %1644 = extractelement <8 x float> %1640, i64 3, !dbg !12
  %1645 = select i1 %.not, float %1642, float %1641, !dbg !12
  %1646 = bitcast float %1645 to i32, !dbg !12
  %1647 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1646), !dbg !12
  %1648 = bitcast i32 %1647 to float, !dbg !12
  %1649 = select i1 %.not, float %1641, float %1648, !dbg !12
  %1650 = select i1 %.not, float %1648, float %1642, !dbg !12
  %1651 = select i1 %.not, float %1644, float %1643, !dbg !12
  %1652 = bitcast float %1651 to i32, !dbg !12
  %1653 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1652), !dbg !12
  %1654 = bitcast i32 %1653 to float, !dbg !12
  %1655 = select i1 %.not, float %1643, float %1654, !dbg !12
  %1656 = select i1 %.not, float %1654, float %1644, !dbg !12
  %1657 = select i1 %.not40, float %1655, float %1649, !dbg !12
  %1658 = bitcast float %1657 to i32, !dbg !12
  %1659 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1658), !dbg !12
  %1660 = bitcast i32 %1659 to float, !dbg !12
  %1661 = select i1 %.not40, float %1656, float %1650, !dbg !12
  %1662 = bitcast float %1661 to i32, !dbg !12
  %1663 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1662), !dbg !12
  %1664 = bitcast i32 %1663 to float, !dbg !12
  %1665 = insertelement <4 x float> poison, float %1649, i64 0, !dbg !12
  %1666 = insertelement <4 x float> %1665, float %1650, i64 1, !dbg !12
  %1667 = insertelement <4 x float> %1666, float %1660, i64 2, !dbg !12
  %1668 = insertelement <4 x float> %1667, float %1664, i64 3, !dbg !12
  %1669 = insertelement <4 x float> poison, float %1660, i64 0, !dbg !12
  %1670 = insertelement <4 x float> %1669, float %1664, i64 1, !dbg !12
  %1671 = insertelement <4 x float> %1670, float %1655, i64 2, !dbg !12
  %1672 = insertelement <4 x float> %1671, float %1656, i64 3, !dbg !12
  %1673 = select <4 x i1> %1329, <4 x float> %1668, <4 x float> %1672, !dbg !12
  %1674 = extractelement <8 x float> %1640, i64 4, !dbg !12
  %1675 = extractelement <8 x float> %1640, i64 5, !dbg !12
  %1676 = extractelement <8 x float> %1640, i64 6, !dbg !12
  %1677 = extractelement <8 x float> %1640, i64 7, !dbg !12
  %1678 = select i1 %.not, float %1675, float %1674, !dbg !12
  %1679 = bitcast float %1678 to i32, !dbg !12
  %1680 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1679), !dbg !12
  %1681 = bitcast i32 %1680 to float, !dbg !12
  %1682 = select i1 %.not, float %1674, float %1681, !dbg !12
  %1683 = select i1 %.not, float %1681, float %1675, !dbg !12
  %1684 = select i1 %.not, float %1677, float %1676, !dbg !12
  %1685 = bitcast float %1684 to i32, !dbg !12
  %1686 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1685), !dbg !12
  %1687 = bitcast i32 %1686 to float, !dbg !12
  %1688 = select i1 %.not, float %1676, float %1687, !dbg !12
  %1689 = select i1 %.not, float %1687, float %1677, !dbg !12
  %1690 = select i1 %.not40, float %1688, float %1682, !dbg !12
  %1691 = bitcast float %1690 to i32, !dbg !12
  %1692 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1691), !dbg !12
  %1693 = bitcast i32 %1692 to float, !dbg !12
  %1694 = select i1 %.not40, float %1689, float %1683, !dbg !12
  %1695 = bitcast float %1694 to i32, !dbg !12
  %1696 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1695), !dbg !12
  %1697 = bitcast i32 %1696 to float, !dbg !12
  %1698 = insertelement <4 x float> poison, float %1682, i64 0, !dbg !12
  %1699 = insertelement <4 x float> %1698, float %1683, i64 1, !dbg !12
  %1700 = insertelement <4 x float> %1699, float %1693, i64 2, !dbg !12
  %1701 = insertelement <4 x float> %1700, float %1697, i64 3, !dbg !12
  %1702 = insertelement <4 x float> poison, float %1693, i64 0, !dbg !12
  %1703 = insertelement <4 x float> %1702, float %1697, i64 1, !dbg !12
  %1704 = insertelement <4 x float> %1703, float %1688, i64 2, !dbg !12
  %1705 = insertelement <4 x float> %1704, float %1689, i64 3, !dbg !12
  %1706 = select <4 x i1> %1329, <4 x float> %1701, <4 x float> %1705, !dbg !12
  %1707 = extractvalue [2 x <8 x float>] %.lcssa7, 0, !dbg !12
  %1708 = extractelement <8 x float> %1707, i64 0, !dbg !12
  %1709 = extractelement <8 x float> %1707, i64 1, !dbg !12
  %1710 = extractelement <8 x float> %1707, i64 2, !dbg !12
  %1711 = extractelement <8 x float> %1707, i64 3, !dbg !12
  %1712 = select i1 %.not, float %1709, float %1708, !dbg !12
  %1713 = bitcast float %1712 to i32, !dbg !12
  %1714 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1713), !dbg !12
  %1715 = bitcast i32 %1714 to float, !dbg !12
  %1716 = select i1 %.not, float %1708, float %1715, !dbg !12
  %1717 = select i1 %.not, float %1715, float %1709, !dbg !12
  %1718 = select i1 %.not, float %1711, float %1710, !dbg !12
  %1719 = bitcast float %1718 to i32, !dbg !12
  %1720 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1719), !dbg !12
  %1721 = bitcast i32 %1720 to float, !dbg !12
  %1722 = select i1 %.not, float %1710, float %1721, !dbg !12
  %1723 = select i1 %.not, float %1721, float %1711, !dbg !12
  %1724 = select i1 %.not40, float %1722, float %1716, !dbg !12
  %1725 = bitcast float %1724 to i32, !dbg !12
  %1726 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1725), !dbg !12
  %1727 = select i1 %.not40, float %1723, float %1717, !dbg !12
  %1728 = bitcast float %1727 to i32, !dbg !12
  %1729 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1728), !dbg !12
  %1730 = extractelement <8 x float> %1707, i64 4, !dbg !12
  %1731 = extractelement <8 x float> %1707, i64 5, !dbg !12
  %1732 = extractelement <8 x float> %1707, i64 6, !dbg !12
  %1733 = extractelement <8 x float> %1707, i64 7, !dbg !12
  %1734 = select i1 %.not, float %1731, float %1730, !dbg !12
  %1735 = bitcast float %1734 to i32, !dbg !12
  %1736 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1735), !dbg !12
  %1737 = bitcast i32 %1736 to float, !dbg !12
  %1738 = select i1 %.not, float %1730, float %1737, !dbg !12
  %1739 = select i1 %.not, float %1737, float %1731, !dbg !12
  %1740 = select i1 %.not, float %1733, float %1732, !dbg !12
  %1741 = bitcast float %1740 to i32, !dbg !12
  %1742 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1741), !dbg !12
  %1743 = bitcast i32 %1742 to float, !dbg !12
  %1744 = select i1 %.not, float %1732, float %1743, !dbg !12
  %1745 = select i1 %.not, float %1743, float %1733, !dbg !12
  %1746 = select i1 %.not40, float %1744, float %1738, !dbg !12
  %1747 = bitcast float %1746 to i32, !dbg !12
  %1748 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1747), !dbg !12
  %1749 = select i1 %.not40, float %1745, float %1739, !dbg !12
  %1750 = bitcast float %1749 to i32, !dbg !12
  %1751 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1750), !dbg !12
  %1752 = extractvalue [2 x <8 x float>] %.lcssa7, 1, !dbg !12
  %1753 = extractelement <8 x float> %1752, i64 0, !dbg !12
  %1754 = extractelement <8 x float> %1752, i64 1, !dbg !12
  %1755 = extractelement <8 x float> %1752, i64 2, !dbg !12
  %1756 = extractelement <8 x float> %1752, i64 3, !dbg !12
  %1757 = select i1 %.not, float %1754, float %1753, !dbg !12
  %1758 = bitcast float %1757 to i32, !dbg !12
  %1759 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1758), !dbg !12
  %1760 = bitcast i32 %1759 to float, !dbg !12
  %1761 = select i1 %.not, float %1753, float %1760, !dbg !12
  %1762 = select i1 %.not, float %1760, float %1754, !dbg !12
  %1763 = select i1 %.not, float %1756, float %1755, !dbg !12
  %1764 = bitcast float %1763 to i32, !dbg !12
  %1765 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1764), !dbg !12
  %1766 = bitcast i32 %1765 to float, !dbg !12
  %1767 = select i1 %.not, float %1755, float %1766, !dbg !12
  %1768 = select i1 %.not, float %1766, float %1756, !dbg !12
  %1769 = select i1 %.not40, float %1767, float %1761, !dbg !12
  %1770 = bitcast float %1769 to i32, !dbg !12
  %1771 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1770), !dbg !12
  %1772 = select i1 %.not40, float %1768, float %1762, !dbg !12
  %1773 = bitcast float %1772 to i32, !dbg !12
  %1774 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1773), !dbg !12
  %1775 = extractelement <8 x float> %1752, i64 4, !dbg !12
  %1776 = extractelement <8 x float> %1752, i64 5, !dbg !12
  %1777 = extractelement <8 x float> %1752, i64 6, !dbg !12
  %1778 = extractelement <8 x float> %1752, i64 7, !dbg !12
  %1779 = select i1 %.not, float %1776, float %1775, !dbg !12
  %1780 = bitcast float %1779 to i32, !dbg !12
  %1781 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1780), !dbg !12
  %1782 = bitcast i32 %1781 to float, !dbg !12
  %1783 = select i1 %.not, float %1775, float %1782, !dbg !12
  %1784 = select i1 %.not, float %1782, float %1776, !dbg !12
  %1785 = select i1 %.not, float %1778, float %1777, !dbg !12
  %1786 = bitcast float %1785 to i32, !dbg !12
  %1787 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1786), !dbg !12
  %1788 = bitcast i32 %1787 to float, !dbg !12
  %1789 = select i1 %.not, float %1777, float %1788, !dbg !12
  %1790 = select i1 %.not, float %1788, float %1778, !dbg !12
  %1791 = select i1 %.not40, float %1789, float %1783, !dbg !12
  %1792 = bitcast float %1791 to i32, !dbg !12
  %1793 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1792), !dbg !12
  %1794 = select i1 %.not40, float %1790, float %1784, !dbg !12
  %1795 = bitcast float %1794 to i32, !dbg !12
  %1796 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1795), !dbg !12
  %1797 = shufflevector <1 x float> %1300, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !51
  %1798 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1797, !dbg !51
  %1799 = shufflevector <1 x float> %1295, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !51
  %1800 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1799, !dbg !51
  %1801 = bitcast i32 %1793 to float, !dbg !12
  %1802 = bitcast i32 %1796 to float, !dbg !12
  %1803 = insertelement <4 x float> poison, float %1783, i64 0, !dbg !12
  %1804 = insertelement <4 x float> %1803, float %1784, i64 1, !dbg !12
  %1805 = insertelement <4 x float> %1804, float %1801, i64 2, !dbg !12
  %1806 = insertelement <4 x float> %1805, float %1802, i64 3, !dbg !12
  %1807 = insertelement <4 x float> poison, float %1801, i64 0, !dbg !12
  %1808 = insertelement <4 x float> %1807, float %1802, i64 1, !dbg !12
  %1809 = insertelement <4 x float> %1808, float %1789, i64 2, !dbg !12
  %1810 = insertelement <4 x float> %1809, float %1790, i64 3, !dbg !12
  %1811 = select <4 x i1> %1329, <4 x float> %1806, <4 x float> %1810, !dbg !12
  %1812 = fmul <4 x float> %1800, %1811, !dbg !51
  %1813 = fptrunc <4 x float> %1812 to <4 x half>, !dbg !52
  %1814 = extractelement <4 x half> %1813, i64 3, !dbg !52
  %1815 = extractelement <4 x half> %1813, i64 2, !dbg !52
  %1816 = extractelement <4 x half> %1813, i64 1, !dbg !52
  %1817 = extractelement <4 x half> %1813, i64 0, !dbg !52
  %1818 = bitcast i32 %1771 to float, !dbg !12
  %1819 = bitcast i32 %1774 to float, !dbg !12
  %1820 = insertelement <4 x float> poison, float %1761, i64 0, !dbg !12
  %1821 = insertelement <4 x float> %1820, float %1762, i64 1, !dbg !12
  %1822 = insertelement <4 x float> %1821, float %1818, i64 2, !dbg !12
  %1823 = insertelement <4 x float> %1822, float %1819, i64 3, !dbg !12
  %1824 = insertelement <4 x float> poison, float %1818, i64 0, !dbg !12
  %1825 = insertelement <4 x float> %1824, float %1819, i64 1, !dbg !12
  %1826 = insertelement <4 x float> %1825, float %1767, i64 2, !dbg !12
  %1827 = insertelement <4 x float> %1826, float %1768, i64 3, !dbg !12
  %1828 = select <4 x i1> %1329, <4 x float> %1823, <4 x float> %1827, !dbg !12
  %1829 = fmul <4 x float> %1800, %1828, !dbg !51
  %1830 = fptrunc <4 x float> %1829 to <4 x half>, !dbg !52
  %1831 = extractelement <4 x half> %1830, i64 3, !dbg !52
  %1832 = extractelement <4 x half> %1830, i64 2, !dbg !52
  %1833 = extractelement <4 x half> %1830, i64 1, !dbg !52
  %1834 = extractelement <4 x half> %1830, i64 0, !dbg !52
  %1835 = bitcast i32 %1748 to float, !dbg !12
  %1836 = bitcast i32 %1751 to float, !dbg !12
  %1837 = insertelement <4 x float> poison, float %1738, i64 0, !dbg !12
  %1838 = insertelement <4 x float> %1837, float %1739, i64 1, !dbg !12
  %1839 = insertelement <4 x float> %1838, float %1835, i64 2, !dbg !12
  %1840 = insertelement <4 x float> %1839, float %1836, i64 3, !dbg !12
  %1841 = insertelement <4 x float> poison, float %1835, i64 0, !dbg !12
  %1842 = insertelement <4 x float> %1841, float %1836, i64 1, !dbg !12
  %1843 = insertelement <4 x float> %1842, float %1744, i64 2, !dbg !12
  %1844 = insertelement <4 x float> %1843, float %1745, i64 3, !dbg !12
  %1845 = select <4 x i1> %1329, <4 x float> %1840, <4 x float> %1844, !dbg !12
  %1846 = fmul <4 x float> %1798, %1845, !dbg !51
  %1847 = fptrunc <4 x float> %1846 to <4 x half>, !dbg !52
  %1848 = extractelement <4 x half> %1847, i64 3, !dbg !52
  %1849 = extractelement <4 x half> %1847, i64 2, !dbg !52
  %1850 = extractelement <4 x half> %1847, i64 1, !dbg !52
  %1851 = extractelement <4 x half> %1847, i64 0, !dbg !52
  %1852 = bitcast i32 %1726 to float, !dbg !12
  %1853 = bitcast i32 %1729 to float, !dbg !12
  %1854 = insertelement <4 x float> poison, float %1716, i64 0, !dbg !12
  %1855 = insertelement <4 x float> %1854, float %1717, i64 1, !dbg !12
  %1856 = insertelement <4 x float> %1855, float %1852, i64 2, !dbg !12
  %1857 = insertelement <4 x float> %1856, float %1853, i64 3, !dbg !12
  %1858 = insertelement <4 x float> poison, float %1852, i64 0, !dbg !12
  %1859 = insertelement <4 x float> %1858, float %1853, i64 1, !dbg !12
  %1860 = insertelement <4 x float> %1859, float %1722, i64 2, !dbg !12
  %1861 = insertelement <4 x float> %1860, float %1723, i64 3, !dbg !12
  %1862 = select <4 x i1> %1329, <4 x float> %1857, <4 x float> %1861, !dbg !12
  %1863 = fmul <4 x float> %1798, %1862, !dbg !51
  %1864 = fptrunc <4 x float> %1863 to <4 x half>, !dbg !52
  %1865 = extractelement <4 x half> %1864, i64 3, !dbg !52
  %1866 = extractelement <4 x half> %1864, i64 2, !dbg !52
  %1867 = extractelement <4 x half> %1864, i64 1, !dbg !52
  %1868 = extractelement <4 x half> %1864, i64 0, !dbg !52
  %address.thread1038 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !53, !range !10, !frisk.register_lifetimes !5
  %address.local1039 = zext nneg i32 %address.thread1038 to i64, !dbg !9
  %address.local1041 = lshr i64 %address.local1039, 2, !dbg !16
  %address.local1042 = and i64 %address.local1041, 48, !dbg !16
  %address.local1044 = and i64 %address.local1039, 240, !dbg !53
  %address.local1043 = add i64 %43, %address.local1039, !dbg !53
  %address.local1046 = sub i64 %address.local1043, %address.local1044, !dbg !53
  %address.local1047 = add i64 %address.local1046, %address.local1042, !dbg !53
  %1869 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %55, !dbg !53
  %address.local1050.idx = shl nuw nsw i64 %address.local1047, 8, !dbg !53
  %address.local1050 = getelementptr inbounds nuw i8, ptr addrspace(1) %1869, i64 %address.local1050.idx, !dbg !53
  %address.local1052 = and i64 %address.local1041, 12, !dbg !11
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1052, !dbg !53
  store half %1868, ptr addrspace(1) %address.local1053, align 2, !dbg !53
  %address.local1055 = or disjoint i64 %address.local1052, 1, !dbg !53
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1055, !dbg !53
  store half %1867, ptr addrspace(1) %address.local1056, align 2, !dbg !53
  %address.local1058 = or disjoint i64 %address.local1052, 2, !dbg !53
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1058, !dbg !53
  store half %1866, ptr addrspace(1) %address.local1059, align 2, !dbg !53
  %address.local1061 = or disjoint i64 %address.local1052, 3, !dbg !53
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1061, !dbg !53
  store half %1865, ptr addrspace(1) %address.local1062, align 2, !dbg !53
  %address.local1064 = or disjoint i64 %address.local1052, 16, !dbg !53
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1064, !dbg !53
  store half %1851, ptr addrspace(1) %address.local1065, align 2, !dbg !53
  %address.local1067 = or disjoint i64 %address.local1052, 17, !dbg !53
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1067, !dbg !53
  store half %1850, ptr addrspace(1) %address.local1068, align 2, !dbg !53
  %address.local1070 = or disjoint i64 %address.local1052, 18, !dbg !53
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1070, !dbg !53
  store half %1849, ptr addrspace(1) %address.local1071, align 2, !dbg !53
  %address.local1073 = or disjoint i64 %address.local1052, 19, !dbg !53
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1073, !dbg !53
  store half %1848, ptr addrspace(1) %address.local1074, align 2, !dbg !53
  %address.local1078 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1050, i64 16384, !dbg !53
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1052, !dbg !53
  store half %1834, ptr addrspace(1) %address.local1079, align 2, !dbg !53
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1055, !dbg !53
  store half %1833, ptr addrspace(1) %address.local1081, align 2, !dbg !53
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1058, !dbg !53
  store half %1832, ptr addrspace(1) %address.local1083, align 2, !dbg !53
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1061, !dbg !53
  store half %1831, ptr addrspace(1) %address.local1085, align 2, !dbg !53
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1064, !dbg !53
  store half %1817, ptr addrspace(1) %address.local1087, align 2, !dbg !53
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1067, !dbg !53
  store half %1816, ptr addrspace(1) %address.local1089, align 2, !dbg !53
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1070, !dbg !53
  store half %1815, ptr addrspace(1) %address.local1091, align 2, !dbg !53
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1073, !dbg !53
  store half %1814, ptr addrspace(1) %address.local1093, align 2, !dbg !53
  %1870 = fmul <4 x float> %1800, %1706, !dbg !51
  %1871 = fptrunc <4 x float> %1870 to <4 x half>, !dbg !52
  %1872 = extractelement <4 x half> %1871, i64 3, !dbg !52
  %1873 = extractelement <4 x half> %1871, i64 2, !dbg !52
  %1874 = extractelement <4 x half> %1871, i64 1, !dbg !52
  %1875 = extractelement <4 x half> %1871, i64 0, !dbg !52
  %1876 = fmul <4 x float> %1800, %1673, !dbg !51
  %1877 = fptrunc <4 x float> %1876 to <4 x half>, !dbg !52
  %1878 = extractelement <4 x half> %1877, i64 3, !dbg !52
  %1879 = extractelement <4 x half> %1877, i64 2, !dbg !52
  %1880 = extractelement <4 x half> %1877, i64 1, !dbg !52
  %1881 = extractelement <4 x half> %1877, i64 0, !dbg !52
  %1882 = fmul <4 x float> %1798, %1639, !dbg !51
  %1883 = fptrunc <4 x float> %1882 to <4 x half>, !dbg !52
  %1884 = extractelement <4 x half> %1883, i64 3, !dbg !52
  %1885 = extractelement <4 x half> %1883, i64 2, !dbg !52
  %1886 = extractelement <4 x half> %1883, i64 1, !dbg !52
  %1887 = extractelement <4 x half> %1883, i64 0, !dbg !52
  %1888 = fmul <4 x float> %1798, %1606, !dbg !51
  %1889 = fptrunc <4 x float> %1888 to <4 x half>, !dbg !52
  %1890 = extractelement <4 x half> %1889, i64 3, !dbg !52
  %1891 = extractelement <4 x half> %1889, i64 2, !dbg !52
  %1892 = extractelement <4 x half> %1889, i64 1, !dbg !52
  %1893 = extractelement <4 x half> %1889, i64 0, !dbg !52
  %address.local1109 = or disjoint i64 %address.local1052, 32, !dbg !53
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1109, !dbg !53
  store half %1893, ptr addrspace(1) %address.local1110, align 2, !dbg !53
  %address.local1112 = or disjoint i64 %address.local1052, 33, !dbg !53
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1112, !dbg !53
  store half %1892, ptr addrspace(1) %address.local1113, align 2, !dbg !53
  %address.local1115 = or disjoint i64 %address.local1052, 34, !dbg !53
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1115, !dbg !53
  store half %1891, ptr addrspace(1) %address.local1116, align 2, !dbg !53
  %address.local1118 = or disjoint i64 %address.local1052, 35, !dbg !53
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1118, !dbg !53
  store half %1890, ptr addrspace(1) %address.local1119, align 2, !dbg !53
  %address.local1121 = or disjoint i64 %address.local1052, 48, !dbg !53
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1121, !dbg !53
  store half %1887, ptr addrspace(1) %address.local1122, align 2, !dbg !53
  %address.local1124 = or disjoint i64 %address.local1052, 49, !dbg !53
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1124, !dbg !53
  store half %1886, ptr addrspace(1) %address.local1125, align 2, !dbg !53
  %address.local1127 = or disjoint i64 %address.local1052, 50, !dbg !53
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1127, !dbg !53
  store half %1885, ptr addrspace(1) %address.local1128, align 2, !dbg !53
  %address.local1130 = or i64 %address.local1041, 51, !dbg !53
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1130, !dbg !53
  store half %1884, ptr addrspace(1) %address.local1131, align 2, !dbg !53
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1109, !dbg !53
  store half %1881, ptr addrspace(1) %address.local1136, align 2, !dbg !53
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1112, !dbg !53
  store half %1880, ptr addrspace(1) %address.local1138, align 2, !dbg !53
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1115, !dbg !53
  store half %1879, ptr addrspace(1) %address.local1140, align 2, !dbg !53
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1118, !dbg !53
  store half %1878, ptr addrspace(1) %address.local1142, align 2, !dbg !53
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1121, !dbg !53
  store half %1875, ptr addrspace(1) %address.local1144, align 2, !dbg !53
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1124, !dbg !53
  store half %1874, ptr addrspace(1) %address.local1146, align 2, !dbg !53
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1127, !dbg !53
  store half %1873, ptr addrspace(1) %address.local1148, align 2, !dbg !53
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1130, !dbg !53
  store half %1872, ptr addrspace(1) %address.local1150, align 2, !dbg !53
  %1894 = fmul <4 x float> %1800, %1572, !dbg !51
  %1895 = fptrunc <4 x float> %1894 to <4 x half>, !dbg !52
  %1896 = extractelement <4 x half> %1895, i64 3, !dbg !52
  %1897 = extractelement <4 x half> %1895, i64 2, !dbg !52
  %1898 = extractelement <4 x half> %1895, i64 1, !dbg !52
  %1899 = extractelement <4 x half> %1895, i64 0, !dbg !52
  %1900 = fmul <4 x float> %1800, %1539, !dbg !51
  %1901 = fptrunc <4 x float> %1900 to <4 x half>, !dbg !52
  %1902 = extractelement <4 x half> %1901, i64 3, !dbg !52
  %1903 = extractelement <4 x half> %1901, i64 2, !dbg !52
  %1904 = extractelement <4 x half> %1901, i64 1, !dbg !52
  %1905 = extractelement <4 x half> %1901, i64 0, !dbg !52
  %1906 = fmul <4 x float> %1798, %1505, !dbg !51
  %1907 = fptrunc <4 x float> %1906 to <4 x half>, !dbg !52
  %1908 = extractelement <4 x half> %1907, i64 3, !dbg !52
  %1909 = extractelement <4 x half> %1907, i64 2, !dbg !52
  %1910 = extractelement <4 x half> %1907, i64 1, !dbg !52
  %1911 = extractelement <4 x half> %1907, i64 0, !dbg !52
  %1912 = fmul <4 x float> %1798, %1472, !dbg !51
  %1913 = fptrunc <4 x float> %1912 to <4 x half>, !dbg !52
  %1914 = extractelement <4 x half> %1913, i64 3, !dbg !52
  %1915 = extractelement <4 x half> %1913, i64 2, !dbg !52
  %1916 = extractelement <4 x half> %1913, i64 1, !dbg !52
  %1917 = extractelement <4 x half> %1913, i64 0, !dbg !52
  %address.local1166 = or disjoint i64 %address.local1052, 64, !dbg !53
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1166, !dbg !53
  store half %1917, ptr addrspace(1) %address.local1167, align 2, !dbg !53
  %address.local1169 = or disjoint i64 %address.local1052, 65, !dbg !53
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1169, !dbg !53
  store half %1916, ptr addrspace(1) %address.local1170, align 2, !dbg !53
  %address.local1172 = or disjoint i64 %address.local1052, 66, !dbg !53
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1172, !dbg !53
  store half %1915, ptr addrspace(1) %address.local1173, align 2, !dbg !53
  %address.local1175 = or disjoint i64 %address.local1052, 67, !dbg !53
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1175, !dbg !53
  store half %1914, ptr addrspace(1) %address.local1176, align 2, !dbg !53
  %address.local1178 = or disjoint i64 %address.local1052, 80, !dbg !53
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1178, !dbg !53
  store half %1911, ptr addrspace(1) %address.local1179, align 2, !dbg !53
  %address.local1181 = or disjoint i64 %address.local1052, 81, !dbg !53
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1181, !dbg !53
  store half %1910, ptr addrspace(1) %address.local1182, align 2, !dbg !53
  %address.local1184 = or disjoint i64 %address.local1052, 82, !dbg !53
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1184, !dbg !53
  store half %1909, ptr addrspace(1) %address.local1185, align 2, !dbg !53
  %address.local1187 = or disjoint i64 %address.local1052, 83, !dbg !53
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1187, !dbg !53
  store half %1908, ptr addrspace(1) %address.local1188, align 2, !dbg !53
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1166, !dbg !53
  store half %1905, ptr addrspace(1) %address.local1193, align 2, !dbg !53
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1169, !dbg !53
  store half %1904, ptr addrspace(1) %address.local1195, align 2, !dbg !53
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1172, !dbg !53
  store half %1903, ptr addrspace(1) %address.local1197, align 2, !dbg !53
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1175, !dbg !53
  store half %1902, ptr addrspace(1) %address.local1199, align 2, !dbg !53
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1178, !dbg !53
  store half %1899, ptr addrspace(1) %address.local1201, align 2, !dbg !53
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1181, !dbg !53
  store half %1898, ptr addrspace(1) %address.local1203, align 2, !dbg !53
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1184, !dbg !53
  store half %1897, ptr addrspace(1) %address.local1205, align 2, !dbg !53
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1187, !dbg !53
  store half %1896, ptr addrspace(1) %address.local1207, align 2, !dbg !53
  %1918 = fmul <4 x float> %1800, %1438, !dbg !51
  %1919 = fptrunc <4 x float> %1918 to <4 x half>, !dbg !52
  %1920 = extractelement <4 x half> %1919, i64 3, !dbg !52
  %1921 = extractelement <4 x half> %1919, i64 2, !dbg !52
  %1922 = extractelement <4 x half> %1919, i64 1, !dbg !52
  %1923 = extractelement <4 x half> %1919, i64 0, !dbg !52
  %1924 = fmul <4 x float> %1800, %1405, !dbg !51
  %1925 = fptrunc <4 x float> %1924 to <4 x half>, !dbg !52
  %1926 = extractelement <4 x half> %1925, i64 3, !dbg !52
  %1927 = extractelement <4 x half> %1925, i64 2, !dbg !52
  %1928 = extractelement <4 x half> %1925, i64 1, !dbg !52
  %1929 = extractelement <4 x half> %1925, i64 0, !dbg !52
  %1930 = fmul <4 x float> %1798, %1371, !dbg !51
  %1931 = fptrunc <4 x float> %1930 to <4 x half>, !dbg !52
  %1932 = extractelement <4 x half> %1931, i64 3, !dbg !52
  %1933 = extractelement <4 x half> %1931, i64 2, !dbg !52
  %1934 = extractelement <4 x half> %1931, i64 1, !dbg !52
  %1935 = extractelement <4 x half> %1931, i64 0, !dbg !52
  %1936 = fmul <4 x float> %1798, %1338, !dbg !51
  %1937 = fptrunc <4 x float> %1936 to <4 x half>, !dbg !52
  %1938 = extractelement <4 x half> %1937, i64 3, !dbg !52
  %1939 = extractelement <4 x half> %1937, i64 2, !dbg !52
  %1940 = extractelement <4 x half> %1937, i64 1, !dbg !52
  %1941 = extractelement <4 x half> %1937, i64 0, !dbg !52
  %address.local1223 = or disjoint i64 %address.local1052, 96, !dbg !53
  %address.local1224 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1223, !dbg !53
  store half %1941, ptr addrspace(1) %address.local1224, align 2, !dbg !53
  %address.local1226 = or disjoint i64 %address.local1052, 97, !dbg !53
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1226, !dbg !53
  store half %1940, ptr addrspace(1) %address.local1227, align 2, !dbg !53
  %address.local1229 = or disjoint i64 %address.local1052, 98, !dbg !53
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1229, !dbg !53
  store half %1939, ptr addrspace(1) %address.local1230, align 2, !dbg !53
  %address.local1232 = or disjoint i64 %address.local1052, 99, !dbg !53
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1232, !dbg !53
  store half %1938, ptr addrspace(1) %address.local1233, align 2, !dbg !53
  %address.local1235 = or disjoint i64 %address.local1052, 112, !dbg !53
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1235, !dbg !53
  store half %1935, ptr addrspace(1) %address.local1236, align 2, !dbg !53
  %address.local1238 = or disjoint i64 %address.local1052, 113, !dbg !53
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1238, !dbg !53
  store half %1934, ptr addrspace(1) %address.local1239, align 2, !dbg !53
  %address.local1241 = or disjoint i64 %address.local1052, 114, !dbg !53
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1241, !dbg !53
  store half %1933, ptr addrspace(1) %address.local1242, align 2, !dbg !53
  %address.local1244 = or i64 %address.local1041, 115, !dbg !53
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1244, !dbg !53
  store half %1932, ptr addrspace(1) %address.local1245, align 2, !dbg !53
  %address.local1250 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1223, !dbg !53
  store half %1929, ptr addrspace(1) %address.local1250, align 2, !dbg !53
  %address.local1252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1226, !dbg !53
  store half %1928, ptr addrspace(1) %address.local1252, align 2, !dbg !53
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1229, !dbg !53
  store half %1927, ptr addrspace(1) %address.local1254, align 2, !dbg !53
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1232, !dbg !53
  store half %1926, ptr addrspace(1) %address.local1256, align 2, !dbg !53
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1235, !dbg !53
  store half %1923, ptr addrspace(1) %address.local1258, align 2, !dbg !53
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1238, !dbg !53
  store half %1922, ptr addrspace(1) %address.local1260, align 2, !dbg !53
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1241, !dbg !53
  store half %1921, ptr addrspace(1) %address.local1262, align 2, !dbg !53
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1244, !dbg !53
  store half %1920, ptr addrspace(1) %address.local1264, align 2, !dbg !53
  ret void, !dbg !8
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }
attributes #12 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "causal.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 89, column: 20, scope: !3)
!8 = !DILocation(line: 23, column: 3, scope: !3)
!9 = !DILocation(line: 70, column: 22, scope: !3)
!10 = !{i32 0, i32 256}
!11 = !DILocation(line: 72, column: 24, scope: !3)
!12 = !DILocation(line: 60, column: 24, scope: !3)
!13 = !DILocation(line: 0, scope: !3)
!14 = !DILocation(line: 71, column: 22, scope: !3)
!15 = !DILocation(line: 74, column: 23, scope: !3)
!16 = !DILocation(line: 77, column: 19, scope: !3)
!17 = !DILocation(line: 101, column: 20, scope: !3)
!18 = !DILocation(line: 104, column: 20, scope: !3)
!19 = !{i32 0}
!20 = !{i32 1}
!21 = !{i32 2}
!22 = !{i32 3}
!23 = !{i32 4}
!24 = !{i32 5}
!25 = !{i32 6}
!26 = !{i32 7}
!27 = !{i32 8}
!28 = !{i32 9}
!29 = !{i32 10}
!30 = !{i32 11}
!31 = !{i32 12}
!32 = !{i32 13}
!33 = !{i32 14}
!34 = !{i32 15}
!35 = !{i32 16}
!36 = !{i32 17}
!37 = !{i32 18}
!38 = !{i32 19}
!39 = !{i32 20}
!40 = !{i32 21}
!41 = !{i32 22}
!42 = !DILocation(line: 90, column: 22, scope: !3)
!43 = !DILocation(line: 91, column: 23, scope: !3)
!44 = !DILocation(line: 92, column: 20, scope: !3)
!45 = !DILocation(line: 103, column: 22, scope: !3)
!46 = !DILocation(line: 93, column: 22, scope: !3)
!47 = !{i64 8192}
!48 = !DILocation(line: 94, column: 22, scope: !3)
!49 = !DILocation(line: 96, column: 16, scope: !3)
!50 = !DILocation(line: 95, column: 20, scope: !3)
!51 = !DILocation(line: 107, column: 23, scope: !3)
!52 = !DILocation(line: 108, column: 19, scope: !3)
!53 = !DILocation(line: 109, column: 9, scope: !3)
