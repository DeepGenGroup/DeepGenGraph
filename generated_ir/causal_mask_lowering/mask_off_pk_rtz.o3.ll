; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/causal_mask_lowering/mask_off_pk_rtz.ll'
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
  %6 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %7 = sext i32 %6 to i64, !dbg !7
  %8 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %9 = sext i32 %8 to i64, !dbg !7
  %10 = shl nsw i64 %9, 5, !dbg !7
  %11 = add nsw i64 %10, %7, !dbg !7
  %12 = lshr i64 %11, 5, !dbg !7
  %13 = and i64 %12, 576460752303423484, !dbg !7
  %14 = and i64 %7, 3, !dbg !7
  %15 = or disjoint i64 %13, %14, !dbg !7
  %16 = sub nsw i64 31, %15, !dbg !7
  %17 = lshr i64 %11, 2, !dbg !7
  %18 = and i64 %17, 31, !dbg !7
  %19 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %20 = sext i32 %19 to i64, !dbg !7
  %21 = icmp ugt i64 %11, 1023, !dbg !8
  %22 = add nsw i64 %15, -32, !dbg !8
  %23 = select i1 %21, i64 %22, i64 %16, !dbg !8
  %24 = sdiv i64 %23, 32, !dbg !8
  %25 = sext i1 %21 to i64, !dbg !8
  %26 = xor i64 %24, %25, !dbg !8
  %27 = add nsw i64 %26, %17, !dbg !8
  %28 = and i64 %27, 31, !dbg !8
  %29 = shl i64 %16, 7, !dbg !8
  %30 = shl i64 %26, 12, !dbg !8
  %31 = sub i64 %29, %30, !dbg !8
  %32 = shl nsw i64 %31, 7, !dbg !8
  %33 = zext nneg i32 %kdma.tid to i64, !dbg !8
  %.idx = shl nuw nsw i64 %28, 20, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local471 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local472 = and i64 %address.local471, 1536, !dbg !8
  %address.local473 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local472, !dbg !8
  %address.local476 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local477 = and i64 %address.local476, 16256, !dbg !8
  %address.local478 = lshr i64 %address.local, 4, !dbg !8
  %address.local479 = and i64 %address.local478, 8, !dbg !8
  %address.local480 = or disjoint i64 %address.local477, %address.local479, !dbg !8
  %34 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1703 = shl i64 %31, 8, !dbg !8
  %35 = add i64 %.idx, %.idx1703, !dbg !8
  %address.local481.idx = shl nuw nsw i64 %address.local480, 1, !dbg !8
  %36 = or disjoint i64 %address.local481.idx, %35, !dbg !8
  %dma.buffer.offset1701 = trunc i64 %36 to i32, !dbg !8
  %37 = insertelement <2 x i64> poison, i64 %34, i64 0, !dbg !8
  %38 = bitcast <2 x i64> %37 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1702 = shufflevector <4 x i32> %38, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local482 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local483 = shl nuw nsw i64 %address.local482, 3, !dbg !8
  %address.local484 = and i64 %address.local483, 3584, !dbg !8
  %address.local485 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local484, !dbg !8
  %address.local490 = lshr i64 %address.local482, 4, !dbg !8
  %address.local491 = and i64 %address.local490, 24, !dbg !8
  %address.local492 = or disjoint i64 %address.local491, %address.local477, !dbg !8
  %address.local493.idx = shl nuw nsw i64 %address.local492, 1, !dbg !8
  %39 = or disjoint i64 %address.local493.idx, %35, !dbg !8
  %dma.buffer.offset1699 = trunc i64 %39 to i32, !dbg !8
  %40 = shl nuw nsw i64 %28, 19, !dbg !8
  %41 = add i64 %40, %32, !dbg !8
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local472, !dbg !8
  %42 = or disjoint i64 %41, 32, !dbg !8
  %43 = or disjoint i64 %address.local480, %42, !dbg !8
  %.tr = trunc i64 %43 to i32, !dbg !8
  %dma.buffer.offset1697 = shl i32 %.tr, 1, !dbg !8
  %address.local498 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local484, !dbg !8
  %44 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local473, i64 0, !dbg !8
  %45 = insertelement <4 x ptr addrspace(3)> %44, ptr addrspace(3) %address.local485, i64 1, !dbg !8
  %46 = insertelement <4 x ptr addrspace(3)> %45, ptr addrspace(3) %address.local494, i64 2, !dbg !8
  %47 = insertelement <4 x ptr addrspace(3)> %46, ptr addrspace(3) %address.local498, i64 3, !dbg !8
  %48 = ptrtoint <4 x ptr addrspace(3)> %47 to <4 x i64>, !dbg !8
  %49 = trunc <4 x i64> %48 to <4 x i32>, !dbg !8
  %50 = extractelement <4 x i32> %49, i64 0, !dbg !8
  %51 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %52 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1701, <4 x i32> %dma.buffer.rsrc1702) #9, !dbg !8
  %53 = extractelement <4 x i32> %49, i64 1, !dbg !8
  %54 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %55 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %54, i32 %dma.buffer.offset1699, <4 x i32> %dma.buffer.rsrc1702) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %56 = extractelement <4 x i32> %49, i64 2, !dbg !8
  %57 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %56), !dbg !8
  %58 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1697, <4 x i32> %dma.buffer.rsrc1702) #9, !dbg !8
  %59 = extractelement <4 x i32> %49, i64 3, !dbg !8
  %60 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %59), !dbg !8
  %61 = or disjoint i64 %address.local492, %42, !dbg !8
  %.tr1707 = trunc i64 %61 to i32, !dbg !8
  %dma.buffer.offset1695 = shl i32 %.tr1707, 1, !dbg !8
  %62 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %60, i32 %dma.buffer.offset1695, <4 x i32> %dma.buffer.rsrc1702) #9, !dbg !8
  %address.local502 = lshr i32 %address.thread, 2, !dbg !10
  %address.local503 = and i32 %address.local502, 4, !dbg !10
  %address.local504 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local505 = and i32 %address.local504, 384, !dbg !10
  %address.local506 = or disjoint i32 %address.local503, %address.local505, !dbg !10
  %address.local507 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local508 = and i32 %address.local507, 120, !dbg !10
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !10
  %address.local510 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local511 = and i32 %address.local510, 1024, !dbg !10
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !10
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !10
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local513, !dbg !10
  %63 = load <4 x half>, ptr addrspace(3) %address.local514, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local514, i64 4096, !dbg !10
  %64 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %65 = fmul <4 x half> %63, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local514, i64 1024, !dbg !10
  %66 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !10
  %67 = fmul <4 x half> %64, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local514, i64 5120, !dbg !10
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !10
  %69 = fmul <4 x half> %66, splat (half 1.275630e-01), !dbg !10
  %70 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %71 = or disjoint i64 %41, 64, !dbg !8
  %72 = or disjoint i64 %address.local480, %71, !dbg !8
  %.tr1709 = trunc i64 %72 to i32, !dbg !8
  %dma.buffer.offset1693 = shl i32 %.tr1709, 1, !dbg !8
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1693, <4 x i32> %dma.buffer.rsrc1702) #9, !dbg !8
  %74 = or disjoint i64 %address.local492, %71, !dbg !8
  %.tr1711 = trunc i64 %74 to i32, !dbg !8
  %dma.buffer.offset1691 = shl i32 %.tr1711, 1, !dbg !8
  %75 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %54, i32 %dma.buffer.offset1691, <4 x i32> %dma.buffer.rsrc1702) #9, !dbg !8
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local514, i64 8192, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !10
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local514, i64 12288, !dbg !10
  %77 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !10
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local514, i64 9216, !dbg !10
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !10
  %79 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local514, i64 13312, !dbg !10
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !10
  %81 = fmul <4 x half> %77, splat (half 1.275630e-01), !dbg !10
  %82 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !10
  %83 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %84 = or disjoint i64 %41, 96, !dbg !8
  %85 = or disjoint i64 %address.local480, %84, !dbg !8
  %.tr1713 = trunc i64 %85 to i32, !dbg !8
  %dma.buffer.offset1689 = shl i32 %.tr1713, 1, !dbg !8
  %86 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1689, <4 x i32> %dma.buffer.rsrc1702) #9, !dbg !8
  %87 = or disjoint i64 %address.local492, %84, !dbg !8
  %.tr1715 = trunc i64 %87 to i32, !dbg !8
  %dma.buffer.offset1687 = shl i32 %.tr1715, 1, !dbg !8
  %88 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %60, i32 %dma.buffer.offset1687, <4 x i32> %dma.buffer.rsrc1702) #9, !dbg !8
  %89 = load <4 x half>, ptr addrspace(3) %address.local514, align 8, !dbg !10
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !10
  %92 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !10
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !10
  %94 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !10
  %95 = fmul <4 x half> %91, splat (half 1.275630e-01), !dbg !10
  %96 = fmul <4 x half> %93, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !10
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !10
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !10
  %100 = fmul <4 x half> %97, splat (half 1.275630e-01), !dbg !10
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !10
  %102 = fmul <4 x half> %98, splat (half 1.275630e-01), !dbg !10
  %103 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !10
  %104 = fmul <4 x half> %101, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %105 = sub nsw i64 32, %15, !dbg !12
  %106 = shl nuw nsw i64 %18, 7, !dbg !13
  %107 = shl nsw i64 %20, 12, !dbg !13
  %108 = or disjoint i64 %106, %107, !dbg !13
  %109 = shl nuw nsw i64 %33, 2, !dbg !14
  %110 = and i64 %109, 3084, !dbg !14
  %111 = and i32 %kdma.tid, 1, !dbg !14
  %narrow = add nuw nsw i32 %111, 1, !dbg !14
  %112 = zext nneg i32 %narrow to i64, !dbg !14
  %113 = or disjoint i64 %110, %112, !dbg !14
  %114 = lshr i64 %113, 1, !dbg !14
  %115 = and i64 %33, 48, !dbg !15
  %116 = icmp ult i64 %11, 1024, !dbg !11
  br i1 %116, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !11

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre22 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre23 = add nuw nsw i32 %.pre22, 64, !dbg !11
  %.pre24 = xor i32 %.pre22, 16, !dbg !11
  %.pre26 = and i32 %.pre23, 192, !dbg !11
  %.pre28 = xor i32 %.pre22, 32, !dbg !11
  br label %._crit_edge, !dbg !11

.lr.ph:                                           ; preds = %4
  %117 = and i64 %33, 783, !dbg !15
  %118 = lshr i64 %33, 2, !dbg !15
  %119 = and i64 %118, 48, !dbg !15
  %120 = or disjoint i64 %117, %119, !dbg !15
  %121 = add nuw nsw i64 %120, %29, !dbg !15
  %122 = add i64 %121, 65, !dbg !16
  %123 = add nuw nsw i64 %121, 1, !dbg !16
  %124 = shl nsw i64 %20, 17, !dbg !13
  %125 = shl nuw nsw i64 %18, 12, !dbg !13
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !7
  %126 = and i32 %kdma.lane, 31, !dbg !7
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !7
  %kdma.owner = or disjoint i32 %126, %kdma.wave, !dbg !7
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !7
  %127 = ptrtoint ptr addrspace(1) %1 to i64
  %128 = insertelement <2 x i64> poison, i64 %127, i64 0
  %129 = bitcast <2 x i64> %128 to <4 x i32>
  %dma.buffer.rsrc1686 = shufflevector <4 x i32> %129, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %kdma.source = zext nneg i32 %kdma.owner to i64
  %kdma.source1859 = lshr i64 %kdma.source, 4
  %kdma.source1860 = and i64 %kdma.source1859, 8
  %kdma.source1861 = and i64 %kdma.source, 91
  %kdma.source1862 = lshr i64 %kdma.source, 2
  %kdma.source1863 = and i64 %kdma.source1862, 4
  %kdma.source1864 = add nuw nsw i64 %kdma.source1863, %kdma.source
  %kdma.source1865 = and i64 %kdma.source1864, 4
  %kdma.source1866 = or disjoint i64 %kdma.source1865, %kdma.source1861
  %kdma.source1867 = shl nuw nsw i64 %kdma.source1866, 7
  %kdma.source1868 = or disjoint i64 %kdma.source1867, %kdma.source1860
  %kdma.source1872 = zext nneg i32 %kdma.owner.hi to i64
  %kdma.source1873 = lshr i64 %kdma.source1872, 4
  %kdma.source1874 = and i64 %kdma.source1873, 8
  %kdma.source1875 = and i64 %kdma.source1872, 123
  %kdma.source1876 = lshr i64 %kdma.source1872, 2
  %kdma.source1877 = and i64 %kdma.source1876, 4
  %kdma.source1878 = add nuw nsw i64 %kdma.source1877, %kdma.source1872
  %kdma.source1879 = and i64 %kdma.source1878, 4
  %kdma.source1880 = or disjoint i64 %kdma.source1879, %kdma.source1875
  %kdma.source1881 = shl nuw nsw i64 %kdma.source1880, 7
  %kdma.source1882 = or disjoint i64 %kdma.source1881, %kdma.source1874
  %kdma.source1899 = or disjoint i64 %kdma.source1868, 16
  %kdma.source1916 = or disjoint i64 %kdma.source1882, 16
  %130 = or disjoint i64 %125, %124
  %kdma.source1936 = or disjoint i32 %kdma.intra, 64
  %kdma.source2006 = or disjoint i32 %kdma.intra, 128
  %kdma.source2076 = or disjoint i32 %kdma.intra, 192
  %131 = ptrtoint ptr addrspace(1) %2 to i64
  %132 = insertelement <2 x i64> poison, i64 %131, i64 0
  %133 = bitcast <2 x i64> %132 to <4 x i32>
  %dma.buffer.rsrc1670 = shufflevector <4 x i32> %133, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %134 = lshr i64 %123, 32
  %compare.high = trunc nuw i64 %134 to i32
  %compare.low.base = trunc i64 %123 to i32
  %135 = lshr i64 %122, 32
  %compare.high1392 = trunc nuw i64 %135 to i32
  %compare.low.base1396 = trunc i64 %122 to i32
  %136 = insertelement <2 x i64> poison, i64 %108, i64 0, !dbg !13
  %137 = insertelement <4 x i32> poison, i32 %compare.low.base1396, i64 0, !dbg !17
  %138 = shufflevector <4 x i32> %137, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !17
  %139 = insertelement <4 x i32> poison, i32 %compare.low.base, i64 0, !dbg !17
  %140 = shufflevector <4 x i32> %139, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !17
  %141 = insertelement <2 x i64> poison, i64 %130, i64 1
  %invariant.op = sub i32 72, %kdma.intra, !dbg !11
  %invariant.op76 = sub i32 72, %kdma.intra, !dbg !11
  %invariant.op77 = sub i32 136, %kdma.intra, !dbg !11
  %invariant.op78 = sub i32 136, %kdma.intra, !dbg !11
  %invariant.op79 = sub i32 200, %kdma.intra, !dbg !11
  %invariant.op80 = sub i32 200, %kdma.intra, !dbg !11
  br label %142, !dbg !11

142:                                              ; preds = %.lr.ph, %142
  %143 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1296, %142 ]
  %144 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1292, %142 ]
  %145 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1288, %142 ]
  %146 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1284, %142 ]
  %147 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %1048, %142 ]
  %148 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1282, %142 ]
  %149 = phi i64 [ 0, %.lr.ph ], [ %1297, %142 ]
  %150 = lshr i64 %149, 12, !dbg !13
  %151 = add nuw nsw i64 %150, %17, !dbg !13
  %152 = lshr i64 %149, 5, !dbg !13
  %153 = and i64 %152, 127, !dbg !13
  %154 = shl nuw nsw i64 %149, 7, !dbg !13
  %155 = shl i64 %151, 19, !dbg !13
  %156 = and i64 %155, 16252928, !dbg !13
  %157 = shl i64 %152, 19, !dbg !13
  %158 = shl nuw nsw i64 %149, 14, !dbg !13
  %159 = sub i64 %158, %157, !dbg !13
  %address.thread565 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %address.local566 = zext nneg i32 %address.thread565 to i64, !dbg !8
  %address.local567 = shl nuw nsw i64 %address.local566, 3, !dbg !8
  %address.local568 = and i64 %address.local567, 1536, !dbg !8
  %address.local569 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local568, !dbg !8
  %.idx17161719 = or disjoint i64 %156, %153, !dbg !13
  %160 = add i64 %.idx17161719, %159, !dbg !13
  %kdma.source1869 = add nuw nsw i64 %160, %kdma.source1868, !dbg !13
  %kdma.source1870 = trunc i64 %kdma.source1869 to i32, !dbg !13
  %kdma.source1871 = shl i32 %kdma.source1870, 1, !dbg !13
  %kdma.offset = add i32 %kdma.source1871, %kdma.intra, !dbg !13
  %kdma.source1883 = add i64 %160, %kdma.source1882, !dbg !13
  %kdma.source1884 = trunc i64 %kdma.source1883 to i32, !dbg !13
  %kdma.source1885 = shl i32 %kdma.source1884, 1, !dbg !13
  %kdma.offset1886 = add i32 %kdma.source1885, %kdma.intra.hi, !dbg !13
  %address.local583 = or disjoint i64 %address.local566, 256, !dbg !8
  %address.local584 = shl nuw nsw i64 %address.local583, 3, !dbg !8
  %address.local585 = and i64 %address.local584, 3584, !dbg !8
  %address.local586 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local585, !dbg !8
  %kdma.source1900 = add nuw nsw i64 %160, %kdma.source1899, !dbg !13
  %kdma.source1901 = trunc i64 %kdma.source1900 to i32, !dbg !13
  %kdma.source1902 = shl i32 %kdma.source1901, 1, !dbg !13
  %kdma.offset1903 = add i32 %kdma.source1902, %kdma.intra, !dbg !13
  %kdma.source1917 = add i64 %160, %kdma.source1916, !dbg !13
  %kdma.source1918 = trunc i64 %kdma.source1917 to i32, !dbg !13
  %kdma.source1919 = shl i32 %kdma.source1918, 1, !dbg !13
  %kdma.offset1920 = add i32 %kdma.source1919, %kdma.intra.hi, !dbg !13
  %address.local600 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local568, !dbg !8
  %.idx17261730 = add i64 %159, %152, !dbg !13
  %161 = add i64 %.idx17261730, %108, !dbg !13
  %162 = insertelement <2 x i64> %141, i64 %152, i64 0, !dbg !13
  %163 = insertelement <2 x i64> %136, i64 %149, i64 1, !dbg !13
  %164 = add nsw <2 x i64> %162, %163, !dbg !13
  %165 = extractelement <2 x i64> %164, i64 1, !dbg !13
  %166 = extractelement <2 x i64> %164, i64 0, !dbg !13
  %167 = add nuw nsw i64 %165, 1024, !dbg !13
  %168 = add nsw <2 x i64> %164, <i64 32, i64 1024>, !dbg !13
  %.lobit37 = ashr i64 %167, 63, !dbg !13
  %169 = icmp slt <2 x i64> %164, <i64 -32, i64 -1024>, !dbg !13
  %170 = icmp slt i64 %166, -32, !dbg !13
  %171 = sub nsw <2 x i64> <i64 -33, i64 -1025>, %164, !dbg !13
  %172 = select <2 x i1> %169, <2 x i64> %171, <2 x i64> %168, !dbg !13
  %173 = sext i1 %170 to i64, !dbg !13
  %174 = insertelement <2 x i64> poison, i64 %173, i64 0, !dbg !13
  %175 = insertelement <2 x i64> %174, i64 %.lobit37, i64 1, !dbg !13
  %address.local604 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local585, !dbg !8
  %176 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local569, i64 0, !dbg !8
  %177 = insertelement <4 x ptr addrspace(3)> %176, ptr addrspace(3) %address.local586, i64 1, !dbg !8
  %178 = insertelement <4 x ptr addrspace(3)> %177, ptr addrspace(3) %address.local600, i64 2, !dbg !8
  %179 = insertelement <4 x ptr addrspace(3)> %178, ptr addrspace(3) %address.local604, i64 3, !dbg !8
  %180 = ptrtoint <4 x ptr addrspace(3)> %179 to <4 x i64>, !dbg !8
  %181 = trunc <4 x i64> %180 to <4 x i32>, !dbg !8
  %182 = extractelement <4 x i32> %181, i64 0, !dbg !13
  %183 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %182), !dbg !13
  %184 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %183, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %kdma.lds1887 = add i32 %183, 512, !dbg !13
  %185 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1887, i32 %kdma.offset1886, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %186 = extractelement <4 x i32> %181, i64 1, !dbg !13
  %187 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %186), !dbg !13
  %188 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %187, i32 %kdma.offset1903, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %kdma.lds1921 = add i32 %187, 512, !dbg !13
  %189 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1921, i32 %kdma.offset1920, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %190 = extractelement <4 x i32> %181, i64 2, !dbg !13
  %191 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %190), !dbg !13
  %192 = sdiv <2 x i64> %172, <i64 128, i64 4096>, !dbg !13
  %193 = xor <2 x i64> %192, %175, !dbg !13
  %194 = shl <2 x i64> %193, <i64 7, i64 19>, !dbg !13
  %195 = extractelement <2 x i64> %194, i64 0, !dbg !13
  %196 = sub i64 %161, %195, !dbg !13
  %197 = extractelement <2 x i64> %194, i64 1, !dbg !13
  %198 = and i64 %197, 16252928, !dbg !13
  %199 = add i64 %196, %198, !dbg !13
  %kdma.source1933 = add i64 %199, %kdma.source1868, !dbg !13
  %kdma.source1934 = trunc i64 %kdma.source1933 to i32, !dbg !13
  %kdma.source1935 = shl i32 %kdma.source1934, 1, !dbg !13
  %kdma.offset1937 = add i32 %kdma.source1936, %kdma.source1935, !dbg !13
  %200 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %191, i32 %kdma.offset1937, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %kdma.source1950 = add i64 %199, %kdma.source1882, !dbg !13
  %kdma.source1951 = trunc i64 %kdma.source1950 to i32, !dbg !13
  %kdma.source1952 = shl i32 %kdma.source1951, 1, !dbg !13
  %kdma.offset1954.reass.reass = add i32 %kdma.source1952, %invariant.op
  %kdma.lds1955 = add i32 %191, 512, !dbg !13
  %201 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1955, i32 %kdma.offset1954.reass.reass, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %202 = extractelement <4 x i32> %181, i64 3, !dbg !13
  %203 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %202), !dbg !13
  %kdma.source1968 = add i64 %199, %kdma.source1899, !dbg !13
  %kdma.source1969 = trunc i64 %kdma.source1968 to i32, !dbg !13
  %kdma.source1970 = shl i32 %kdma.source1969, 1, !dbg !13
  %kdma.offset1972 = add i32 %kdma.source1936, %kdma.source1970, !dbg !13
  %204 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %203, i32 %kdma.offset1972, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %kdma.source1986 = add i64 %199, %kdma.source1916, !dbg !13
  %kdma.source1987 = trunc i64 %kdma.source1986 to i32, !dbg !13
  %kdma.source1988 = shl i32 %kdma.source1987, 1, !dbg !13
  %kdma.offset1990.reass.reass = add i32 %kdma.source1988, %invariant.op76
  %kdma.lds1991 = add i32 %203, 512, !dbg !13
  %205 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1991, i32 %kdma.offset1990.reass.reass, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %address.local609 = lshr i64 %address.local566, 2, !dbg !14
  %address.local612 = shl nuw nsw i64 %address.local566, 1, !dbg !14
  %address.local613 = and i64 %address.local612, 24, !dbg !14
  %address.local615 = shl i64 %address.local566, 62, !dbg !14
  %address.local616 = ashr exact i64 %address.local615, 57, !dbg !14
  %address.local617 = and i64 %address.local616, -64, !dbg !14
  %address.local620 = and i64 %address.local609, 12, !dbg !10
  %address.local622 = and i64 %address.local566, 63, !dbg !10
  %address.local623 = lshr i64 %address.local622, 5, !dbg !10
  %address.local624 = mul nuw nsw i64 %address.local623, 1016, !dbg !10
  %.idx1738 = mul nuw nsw i64 %address.local566, 320, !dbg !14
  %206 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1738, !dbg !14
  %.idx1739 = mul nsw i64 %address.local609, -1280, !dbg !14
  %207 = getelementptr i8, ptr addrspace(3) %206, i64 %.idx1739, !dbg !14
  %208 = getelementptr [2 x i8], ptr addrspace(3) %207, i64 %address.local613, !dbg !14
  %209 = getelementptr [2 x i8], ptr addrspace(3) %208, i64 %address.local617, !dbg !14
  %210 = getelementptr [2 x i8], ptr addrspace(3) %209, i64 %address.local620, !dbg !14
  %address.local626 = getelementptr [2 x i8], ptr addrspace(3) %210, i64 %address.local624, !dbg !14
  %211 = add nsw i64 %.idx1739, %.idx1738, !dbg !14
  %.idx2258 = shl nuw nsw i64 %address.local613, 1, !dbg !14
  %212 = or disjoint i64 %211, %.idx2258, !dbg !14
  %213 = or disjoint i64 %address.local617, %address.local620, !dbg !14
  %214 = shl nsw i64 %213, 1, !dbg !14
  %address.local626.idx = mul nuw nsw i64 %address.local623, 2032, !dbg !14
  %215 = add nsw i64 %212, %address.local626.idx, !dbg !14
  %216 = add nsw i64 %215, %214, !dbg !14
  %217 = lshr i64 %216, 6, !dbg !14
  %218 = and i64 %217, 8, !dbg !14
  %219 = xor i64 %218, %216, !dbg !14
  %220 = sub i64 %219, %216, !dbg !14
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local626, i64 %220, !dbg !14
  %221 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14
  %222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %221, <4 x float> zeroinitializer) #8, !dbg !14
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %223 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14
  %address.local637 = shl nuw nsw i32 %address.thread565, 1, !dbg !14
  %address.local638 = and i32 %address.local637, 24, !dbg !14
  %address.local639 = shl nuw nsw i32 %address.thread565, 8, !dbg !14
  %address.local640 = and i32 %address.local639, 768, !dbg !14
  %address.local641 = or disjoint i32 %address.local638, %address.local640, !dbg !14
  %address.local642 = shl nuw nsw i32 %address.thread565, 5, !dbg !14
  %address.local643 = and i32 %address.local642, 96, !dbg !14
  %address.local644 = or disjoint i32 %address.local641, %address.local643, !dbg !14
  %address.local645 = zext nneg i32 %address.local644 to i64, !dbg !14
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %114) #8, !dbg !14, !frisk.register_lifetimes !5
  %address.local646 = lshr i32 %address.thread565, 1, !dbg !14
  %address.local647 = and i32 %address.local646, 1, !dbg !14
  %address.local648 = zext nneg i32 %address.local647 to i64, !dbg !14
  %address.local649 = add nsw i64 %address.base, %address.local648, !dbg !14
  %address.local650 = shl i64 %address.local649, 6, !dbg !14
  %address.local651 = sub i64 %address.local645, %address.local650, !dbg !14
  %224 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local651, !dbg !14
  %225 = getelementptr [2 x i8], ptr addrspace(3) %224, i64 %address.local620, !dbg !14
  %226 = getelementptr [2 x i8], ptr addrspace(3) %225, i64 %address.local624, !dbg !14
  %address.local655 = getelementptr i8, ptr addrspace(3) %226, i64 64, !dbg !14
  %.idx22612264 = add i64 %address.local651, %address.local620, !dbg !14
  %227 = shl i64 %.idx22612264, 1, !dbg !14
  %228 = add nuw nsw i64 %address.local626.idx, 64, !dbg !14
  %229 = add i64 %228, %227, !dbg !14
  %230 = lshr i64 %229, 6, !dbg !14
  %231 = and i64 %230, 8, !dbg !14
  %232 = xor i64 %231, %229, !dbg !14
  %233 = sub i64 %232, %229, !dbg !14
  %kdma.swizzled2132 = getelementptr i8, ptr addrspace(3) %address.local655, i64 %233, !dbg !14
  %234 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2132, align 8, !dbg !14, !frisk.k_lds_pair !18
  %kdma.gep2136 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 128, !dbg !14
  %235 = load <4 x half>, ptr addrspace(3) %kdma.gep2136, align 8, !dbg !14, !frisk.k_lds_pair !18
  %kdma.gep2133 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 4096, !dbg !14
  %236 = load <4 x half>, ptr addrspace(3) %kdma.gep2133, align 8, !dbg !14, !frisk.k_lds_pair !19
  %kdma.gep2137 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 4224, !dbg !14
  %237 = load <4 x half>, ptr addrspace(3) %kdma.gep2137, align 8, !dbg !14, !frisk.k_lds_pair !19
  %kdma.gep2134 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %238 = load <4 x half>, ptr addrspace(3) %kdma.gep2134, align 8, !dbg !14, !frisk.k_lds_pair !20
  %kdma.gep2138 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %239 = load <4 x half>, ptr addrspace(3) %kdma.gep2138, align 8, !dbg !14, !frisk.k_lds_pair !20
  %kdma.gep2135 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %240 = load <4 x half>, ptr addrspace(3) %kdma.gep2135, align 8, !dbg !14, !frisk.k_lds_pair !21
  %kdma.gep2139 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %241 = load <4 x half>, ptr addrspace(3) %kdma.gep2139, align 8, !dbg !14, !frisk.k_lds_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %223, <4 x float> %222) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep2140 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 1024, !dbg !14
  %243 = load <4 x half>, ptr addrspace(3) %kdma.gep2140, align 8, !dbg !14, !frisk.k_lds_pair !22
  %kdma.gep2144 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 1152, !dbg !14
  %244 = load <4 x half>, ptr addrspace(3) %kdma.gep2144, align 8, !dbg !14, !frisk.k_lds_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %234, <4 x float> zeroinitializer) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep2141 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 5120, !dbg !14
  %246 = load <4 x half>, ptr addrspace(3) %kdma.gep2141, align 8, !dbg !14, !frisk.k_lds_pair !23
  %kdma.gep2145 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 5248, !dbg !14
  %247 = load <4 x half>, ptr addrspace(3) %kdma.gep2145, align 8, !dbg !14, !frisk.k_lds_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %236, <4 x float> %245) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep2142 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %249 = load <4 x half>, ptr addrspace(3) %kdma.gep2142, align 8, !dbg !14, !frisk.k_lds_pair !24
  %kdma.gep2143 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %250 = load <4 x half>, ptr addrspace(3) %kdma.gep2143, align 8, !dbg !14, !frisk.k_lds_pair !24
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %238, <4 x float> zeroinitializer) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %240, <4 x float> %251) #8, !dbg !14
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %235, <4 x float> zeroinitializer) #8, !dbg !14
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %237, <4 x float> %253) #8, !dbg !14
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %239, <4 x float> zeroinitializer) #8, !dbg !14
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %241, <4 x float> %255) #8, !dbg !14
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %243, <4 x float> zeroinitializer) #8, !dbg !14
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %246, <4 x float> %257) #8, !dbg !14
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %249, <4 x float> zeroinitializer) #8, !dbg !14
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %250, <4 x float> %259) #8, !dbg !14
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %244, <4 x float> zeroinitializer) #8, !dbg !14
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %247, <4 x float> %261) #8, !dbg !14
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %221, <4 x float> zeroinitializer) #8, !dbg !14
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %223, <4 x float> %263) #8, !dbg !14
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %234, <4 x float> zeroinitializer) #8, !dbg !14
  %266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %236, <4 x float> %265) #8, !dbg !14
  %267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %238, <4 x float> zeroinitializer) #8, !dbg !14
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %240, <4 x float> %267) #8, !dbg !14
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %235, <4 x float> zeroinitializer) #8, !dbg !14
  %270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %237, <4 x float> %269) #8, !dbg !14
  %271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %239, <4 x float> zeroinitializer) #8, !dbg !14
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %241, <4 x float> %271) #8, !dbg !14
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %243, <4 x float> zeroinitializer) #8, !dbg !14
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %246, <4 x float> %273) #8, !dbg !14
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %249, <4 x float> zeroinitializer) #8, !dbg !14
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %250, <4 x float> %275) #8, !dbg !14
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %244, <4 x float> zeroinitializer) #8, !dbg !14
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %247, <4 x float> %277) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %279 = add nuw nsw i64 %165, 2048, !dbg !13
  %280 = add nsw <2 x i64> %164, <i64 64, i64 2048>, !dbg !13
  %.lobit38 = ashr i64 %279, 63, !dbg !13
  %281 = icmp slt <2 x i64> %164, <i64 -64, i64 -2048>, !dbg !13
  %282 = icmp slt i64 %166, -64, !dbg !13
  %283 = sub nsw <2 x i64> <i64 -65, i64 -2049>, %164, !dbg !13
  %284 = select <2 x i1> %281, <2 x i64> %283, <2 x i64> %280, !dbg !13
  %285 = sdiv <2 x i64> %284, <i64 128, i64 4096>, !dbg !13
  %286 = sext i1 %282 to i64, !dbg !13
  %287 = insertelement <2 x i64> poison, i64 %286, i64 0, !dbg !13
  %288 = insertelement <2 x i64> %287, i64 %.lobit38, i64 1, !dbg !13
  %289 = xor <2 x i64> %285, %288, !dbg !13
  %290 = shl <2 x i64> %289, <i64 7, i64 19>, !dbg !13
  %291 = extractelement <2 x i64> %290, i64 0, !dbg !13
  %292 = sub i64 %161, %291, !dbg !13
  %293 = extractelement <2 x i64> %290, i64 1, !dbg !13
  %294 = and i64 %293, 16252928, !dbg !13
  %295 = add i64 %292, %294, !dbg !13
  %kdma.source2003 = add i64 %295, %kdma.source1868, !dbg !13
  %kdma.source2004 = trunc i64 %kdma.source2003 to i32, !dbg !13
  %kdma.source2005 = shl i32 %kdma.source2004, 1, !dbg !13
  %kdma.offset2007 = add i32 %kdma.source2006, %kdma.source2005, !dbg !13
  %296 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %183, i32 %kdma.offset2007, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %kdma.source2020 = add i64 %295, %kdma.source1882, !dbg !13
  %kdma.source2021 = trunc i64 %kdma.source2020 to i32, !dbg !13
  %kdma.source2022 = shl i32 %kdma.source2021, 1, !dbg !13
  %kdma.offset2024.reass.reass = add i32 %kdma.source2022, %invariant.op77
  %297 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1887, i32 %kdma.offset2024.reass.reass, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %kdma.source2038 = add i64 %295, %kdma.source1899, !dbg !13
  %kdma.source2039 = trunc i64 %kdma.source2038 to i32, !dbg !13
  %kdma.source2040 = shl i32 %kdma.source2039, 1, !dbg !13
  %kdma.offset2042 = add i32 %kdma.source2006, %kdma.source2040, !dbg !13
  %298 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %187, i32 %kdma.offset2042, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %kdma.source2056 = add i64 %295, %kdma.source1916, !dbg !13
  %kdma.source2057 = trunc i64 %kdma.source2056 to i32, !dbg !13
  %kdma.source2058 = shl i32 %kdma.source2057, 1, !dbg !13
  %kdma.offset2060.reass.reass = add i32 %kdma.source2058, %invariant.op78
  %299 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1921, i32 %kdma.offset2060.reass.reass, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %kdma.gep2162 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %300 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !14, !frisk.k_lds_pair !25
  %kdma.gep2166 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %301 = load <4 x half>, ptr addrspace(3) %kdma.gep2166, align 8, !dbg !14, !frisk.k_lds_pair !25
  %kdma.gep2163 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !14, !frisk.k_lds_pair !26
  %kdma.gep2167 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep2167, align 8, !dbg !14, !frisk.k_lds_pair !26
  %kdma.gep2164 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 8192, !dbg !14
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !14, !frisk.k_lds_pair !27
  %kdma.gep2168 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 8320, !dbg !14
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep2168, align 8, !dbg !14, !frisk.k_lds_pair !27
  %kdma.gep2165 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 12288, !dbg !14
  %306 = load <4 x half>, ptr addrspace(3) %kdma.gep2165, align 8, !dbg !14, !frisk.k_lds_pair !28
  %kdma.gep2169 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 12416, !dbg !14
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep2169, align 8, !dbg !14, !frisk.k_lds_pair !28
  %kdma.gep2170 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep2170, align 8, !dbg !14, !frisk.k_lds_pair !29
  %kdma.gep2174 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep2174, align 8, !dbg !14, !frisk.k_lds_pair !29
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %300, <4 x float> %242) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep2171 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep2171, align 8, !dbg !14, !frisk.k_lds_pair !30
  %kdma.gep2175 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep2175, align 8, !dbg !14, !frisk.k_lds_pair !30
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %302, <4 x float> %310) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep2172 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 9216, !dbg !14
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep2172, align 8, !dbg !14, !frisk.k_lds_pair !31
  %kdma.gep2176 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 9344, !dbg !14
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep2176, align 8, !dbg !14, !frisk.k_lds_pair !31
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %304, <4 x float> %248) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep2173 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 13312, !dbg !14
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep2173, align 8, !dbg !14, !frisk.k_lds_pair !32
  %kdma.gep2177 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 13440, !dbg !14
  %318 = load <4 x half>, ptr addrspace(3) %kdma.gep2177, align 8, !dbg !14, !frisk.k_lds_pair !32
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %306, <4 x float> %316) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %301, <4 x float> %252) #8, !dbg !14
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %303, <4 x float> %320) #8, !dbg !14
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %305, <4 x float> %254) #8, !dbg !14
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %307, <4 x float> %322) #8, !dbg !14
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %308, <4 x float> %256) #8, !dbg !14
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %311, <4 x float> %324) #8, !dbg !14
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %314, <4 x float> %258) #8, !dbg !14
  %327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %317, <4 x float> %326) #8, !dbg !14
  %328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %309, <4 x float> %260) #8, !dbg !14
  %329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %312, <4 x float> %328) #8, !dbg !14
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %315, <4 x float> %262) #8, !dbg !14
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %318, <4 x float> %330) #8, !dbg !14
  %332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %300, <4 x float> %264) #8, !dbg !14
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %302, <4 x float> %332) #8, !dbg !14
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %304, <4 x float> %266) #8, !dbg !14
  %335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %306, <4 x float> %334) #8, !dbg !14
  %336 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %301, <4 x float> %268) #8, !dbg !14
  %337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %303, <4 x float> %336) #8, !dbg !14
  %338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %305, <4 x float> %270) #8, !dbg !14
  %339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %307, <4 x float> %338) #8, !dbg !14
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %308, <4 x float> %272) #8, !dbg !14
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %311, <4 x float> %340) #8, !dbg !14
  %342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %314, <4 x float> %274) #8, !dbg !14
  %343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %317, <4 x float> %342) #8, !dbg !14
  %344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %309, <4 x float> %276) #8, !dbg !14
  %345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %312, <4 x float> %344) #8, !dbg !14
  %346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %315, <4 x float> %278) #8, !dbg !14
  %347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %318, <4 x float> %346) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %348 = add nuw nsw i64 %165, 3072, !dbg !13
  %349 = add nsw <2 x i64> %164, <i64 96, i64 3072>, !dbg !13
  %.lobit39 = ashr i64 %348, 63, !dbg !13
  %350 = icmp slt <2 x i64> %164, <i64 -96, i64 -3072>, !dbg !13
  %351 = icmp slt i64 %166, -96, !dbg !13
  %352 = sub nsw <2 x i64> <i64 -97, i64 -3073>, %164, !dbg !13
  %353 = select <2 x i1> %350, <2 x i64> %352, <2 x i64> %349, !dbg !13
  %354 = sdiv <2 x i64> %353, <i64 128, i64 4096>, !dbg !13
  %355 = sext i1 %351 to i64, !dbg !13
  %356 = insertelement <2 x i64> poison, i64 %355, i64 0, !dbg !13
  %357 = insertelement <2 x i64> %356, i64 %.lobit39, i64 1, !dbg !13
  %358 = xor <2 x i64> %354, %357, !dbg !13
  %359 = shl <2 x i64> %358, <i64 7, i64 19>, !dbg !13
  %360 = extractelement <2 x i64> %359, i64 0, !dbg !13
  %361 = sub i64 %161, %360, !dbg !13
  %362 = extractelement <2 x i64> %359, i64 1, !dbg !13
  %363 = and i64 %362, 16252928, !dbg !13
  %364 = add i64 %361, %363, !dbg !13
  %kdma.source2073 = add i64 %364, %kdma.source1868, !dbg !13
  %kdma.source2074 = trunc i64 %kdma.source2073 to i32, !dbg !13
  %kdma.source2075 = shl i32 %kdma.source2074, 1, !dbg !13
  %kdma.offset2077 = add i32 %kdma.source2076, %kdma.source2075, !dbg !13
  %365 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %191, i32 %kdma.offset2077, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %kdma.source2090 = add i64 %364, %kdma.source1882, !dbg !13
  %kdma.source2091 = trunc i64 %kdma.source2090 to i32, !dbg !13
  %kdma.source2092 = shl i32 %kdma.source2091, 1, !dbg !13
  %kdma.offset2094.reass.reass = add i32 %kdma.source2092, %invariant.op79
  %366 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1955, i32 %kdma.offset2094.reass.reass, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %kdma.source2108 = add i64 %364, %kdma.source1899, !dbg !13
  %kdma.source2109 = trunc i64 %kdma.source2108 to i32, !dbg !13
  %kdma.source2110 = shl i32 %kdma.source2109, 1, !dbg !13
  %kdma.offset2112 = add i32 %kdma.source2076, %kdma.source2110, !dbg !13
  %367 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %203, i32 %kdma.offset2112, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %kdma.source2126 = add i64 %364, %kdma.source1916, !dbg !13
  %kdma.source2127 = trunc i64 %kdma.source2126 to i32, !dbg !13
  %kdma.source2128 = shl i32 %kdma.source2127, 1, !dbg !13
  %kdma.offset2130.reass.reass = add i32 %kdma.source2128, %invariant.op80
  %368 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1991, i32 %kdma.offset2130.reass.reass, <4 x i32> %dma.buffer.rsrc1686) #9, !dbg !13
  %369 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14, !frisk.k_lds_pair !33
  %370 = load <4 x half>, ptr addrspace(3) %kdma.gep2134, align 8, !dbg !14, !frisk.k_lds_pair !33
  %371 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14, !frisk.k_lds_pair !34
  %372 = load <4 x half>, ptr addrspace(3) %kdma.gep2135, align 8, !dbg !14, !frisk.k_lds_pair !34
  %373 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2132, align 8, !dbg !14, !frisk.k_lds_pair !35
  %374 = load <4 x half>, ptr addrspace(3) %kdma.gep2136, align 8, !dbg !14, !frisk.k_lds_pair !35
  %375 = load <4 x half>, ptr addrspace(3) %kdma.gep2133, align 8, !dbg !14, !frisk.k_lds_pair !36
  %376 = load <4 x half>, ptr addrspace(3) %kdma.gep2137, align 8, !dbg !14, !frisk.k_lds_pair !36
  %377 = load <4 x half>, ptr addrspace(3) %kdma.gep2138, align 8, !dbg !14, !frisk.k_lds_pair !37
  %378 = load <4 x half>, ptr addrspace(3) %kdma.gep2142, align 8, !dbg !14, !frisk.k_lds_pair !37
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %369, <4 x float> %313) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %380 = load <4 x half>, ptr addrspace(3) %kdma.gep2139, align 8, !dbg !14, !frisk.k_lds_pair !38
  %381 = load <4 x half>, ptr addrspace(3) %kdma.gep2143, align 8, !dbg !14, !frisk.k_lds_pair !38
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %371, <4 x float> %379) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %383 = load <4 x half>, ptr addrspace(3) %kdma.gep2140, align 8, !dbg !14, !frisk.k_lds_pair !39
  %384 = load <4 x half>, ptr addrspace(3) %kdma.gep2144, align 8, !dbg !14, !frisk.k_lds_pair !39
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %373, <4 x float> %319) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %386 = load <4 x half>, ptr addrspace(3) %kdma.gep2141, align 8, !dbg !14, !frisk.k_lds_pair !40
  %387 = load <4 x half>, ptr addrspace(3) %kdma.gep2145, align 8, !dbg !14, !frisk.k_lds_pair !40
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %375, <4 x float> %385) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %370, <4 x float> %321) #8, !dbg !14
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %372, <4 x float> %389) #8, !dbg !14
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %374, <4 x float> %323) #8, !dbg !14
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %376, <4 x float> %391) #8, !dbg !14
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %377, <4 x float> %325) #8, !dbg !14
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %380, <4 x float> %393) #8, !dbg !14
  %395 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %383, <4 x float> %327) #8, !dbg !14
  %396 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %386, <4 x float> %395) #8, !dbg !14
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %378, <4 x float> %329) #8, !dbg !14
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %381, <4 x float> %397) #8, !dbg !14
  %399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %384, <4 x float> %331) #8, !dbg !14
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %387, <4 x float> %399) #8, !dbg !14
  %401 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %369, <4 x float> %333) #8, !dbg !14
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %371, <4 x float> %401) #8, !dbg !14
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %373, <4 x float> %335) #8, !dbg !14
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %375, <4 x float> %403) #8, !dbg !14
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %370, <4 x float> %337) #8, !dbg !14
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %372, <4 x float> %405) #8, !dbg !14
  %407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %374, <4 x float> %339) #8, !dbg !14
  %408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %376, <4 x float> %407) #8, !dbg !14
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %377, <4 x float> %341) #8, !dbg !14
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %380, <4 x float> %409) #8, !dbg !14
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %383, <4 x float> %343) #8, !dbg !14
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %386, <4 x float> %411) #8, !dbg !14
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %378, <4 x float> %345) #8, !dbg !14
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %381, <4 x float> %413) #8, !dbg !14
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %384, <4 x float> %347) #8, !dbg !14
  %416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %387, <4 x float> %415) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %417 = add nuw nsw i64 %152, %17, !dbg !41
  %418 = shl i64 %417, 19, !dbg !41
  %419 = and i64 %418, 16252928, !dbg !41
  %address.local984 = shl nuw nsw i64 %address.local566, 5, !dbg !41
  %address.local985 = and i64 %address.local984, 8064, !dbg !41
  %address.local987 = and i64 %address.local567, 24, !dbg !41
  %address.local988 = or disjoint i64 %address.local985, %address.local987, !dbg !41
  %.idx17641766 = add i64 %419, %159, !dbg !41
  %420 = or disjoint i64 %address.local988, %.idx17641766, !dbg !41
  %.tr1767 = trunc i64 %420 to i32, !dbg !41
  %dma.buffer.offset1669 = shl i32 %.tr1767, 1, !dbg !41
  %421 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %183, i32 %dma.buffer.offset1669, <4 x i32> %dma.buffer.rsrc1670) #9, !dbg !41
  %address.local990 = shl nuw nsw i64 %address.local583, 5, !dbg !41
  %address.local991 = and i64 %address.local990, 16256, !dbg !41
  %address.local994 = or disjoint i64 %address.local991, %address.local987, !dbg !41
  %422 = or disjoint i64 %address.local994, %.idx17641766, !dbg !41
  %.tr1771 = trunc i64 %422 to i32, !dbg !41
  %dma.buffer.offset1667 = shl i32 %.tr1771, 1, !dbg !41
  %423 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %187, i32 %dma.buffer.offset1667, <4 x i32> %dma.buffer.rsrc1670) #9, !dbg !41
  %424 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !14
  %425 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %424, <4 x float> %382) #8, !dbg !14
  %426 = or disjoint i64 %154, %115, !dbg !15
  %427 = lshr i64 %149, 25, !dbg !17
  %compare.high1268 = trunc nuw i64 %427 to i32, !dbg !17
  %compare.low.base1269 = trunc i64 %426 to i32, !dbg !17
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 1) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1272 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 2) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1276 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 3) #8, !dbg !17, !frisk.register_lifetimes !5
  %428 = insertelement <4 x i32> poison, i32 %compare.low.base1269, i64 0, !dbg !17
  %429 = insertelement <4 x i32> %428, i32 %compare.local.base, i64 1, !dbg !17
  %430 = insertelement <4 x i32> %429, i32 %compare.local.base1272, i64 2, !dbg !17
  %431 = insertelement <4 x i32> %430, i32 %compare.local.base1276, i64 3, !dbg !17
  %432 = or <4 x i32> %431, <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %433 = icmp ugt <4 x i32> %140, %432, !dbg !17
  %compare.local.base1280 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 4) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1284 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 5) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1288 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 6) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1292 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 7) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1296 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 8) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1300 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 9) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1304 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 10) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1308 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 11) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1312 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 12) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1316 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 13) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1320 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 14) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1324 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 15) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1328 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 16) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1332 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 17) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1336 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 18) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1340 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 19) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1344 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 20) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1348 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 21) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1352 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 22) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1356 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 23) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1360 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 24) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1364 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 25) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1368 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 26) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1372 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 27) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1376 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 28) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1380 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 29) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1384 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 30) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1388 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 31) #8, !dbg !17, !frisk.register_lifetimes !5
  %434 = insertelement <4 x i32> poison, i32 %compare.local.base1360, i64 0, !dbg !17
  %435 = insertelement <4 x i32> %434, i32 %compare.local.base1364, i64 1, !dbg !17
  %436 = insertelement <4 x i32> %435, i32 %compare.local.base1368, i64 2, !dbg !17
  %437 = insertelement <4 x i32> %436, i32 %compare.local.base1372, i64 3, !dbg !17
  %438 = or <4 x i32> %437, <i32 72, i32 73, i32 74, i32 75>, !dbg !17
  %439 = icmp ult <4 x i32> %438, %140, !dbg !17
  %440 = insertelement <4 x i32> poison, i32 %compare.local.base1344, i64 0, !dbg !17
  %441 = insertelement <4 x i32> %440, i32 %compare.local.base1348, i64 1, !dbg !17
  %442 = insertelement <4 x i32> %441, i32 %compare.local.base1352, i64 2, !dbg !17
  %443 = insertelement <4 x i32> %442, i32 %compare.local.base1356, i64 3, !dbg !17
  %444 = or <4 x i32> %443, <i32 68, i32 69, i32 70, i32 71>, !dbg !17
  %445 = icmp ult <4 x i32> %444, %140, !dbg !17
  %446 = insertelement <4 x i32> poison, i32 %compare.local.base1328, i64 0, !dbg !17
  %447 = insertelement <4 x i32> %446, i32 %compare.local.base1332, i64 1, !dbg !17
  %448 = insertelement <4 x i32> %447, i32 %compare.local.base1336, i64 2, !dbg !17
  %449 = insertelement <4 x i32> %448, i32 %compare.local.base1340, i64 3, !dbg !17
  %450 = or <4 x i32> %449, <i32 64, i32 65, i32 66, i32 67>, !dbg !17
  %451 = icmp ult <4 x i32> %450, %140, !dbg !17
  %452 = insertelement <4 x i32> poison, i32 %compare.local.base1312, i64 0, !dbg !17
  %453 = insertelement <4 x i32> %452, i32 %compare.local.base1316, i64 1, !dbg !17
  %454 = insertelement <4 x i32> %453, i32 %compare.local.base1320, i64 2, !dbg !17
  %455 = insertelement <4 x i32> %454, i32 %compare.local.base1324, i64 3, !dbg !17
  %456 = or <4 x i32> %455, <i32 12, i32 13, i32 14, i32 15>, !dbg !17
  %457 = icmp ult <4 x i32> %456, %140, !dbg !17
  %458 = insertelement <4 x i32> poison, i32 %compare.local.base1296, i64 0, !dbg !17
  %459 = insertelement <4 x i32> %458, i32 %compare.local.base1300, i64 1, !dbg !17
  %460 = insertelement <4 x i32> %459, i32 %compare.local.base1304, i64 2, !dbg !17
  %461 = insertelement <4 x i32> %460, i32 %compare.local.base1308, i64 3, !dbg !17
  %462 = or <4 x i32> %461, <i32 8, i32 9, i32 10, i32 11>, !dbg !17
  %463 = icmp ult <4 x i32> %462, %140, !dbg !17
  %464 = insertelement <4 x i32> poison, i32 %compare.local.base1280, i64 0, !dbg !17
  %465 = insertelement <4 x i32> %464, i32 %compare.local.base1284, i64 1, !dbg !17
  %466 = insertelement <4 x i32> %465, i32 %compare.local.base1288, i64 2, !dbg !17
  %467 = insertelement <4 x i32> %466, i32 %compare.local.base1292, i64 3, !dbg !17
  %468 = or <4 x i32> %467, <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %469 = icmp ult <4 x i32> %468, %140, !dbg !17
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1268, !dbg !17
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1268, !dbg !17
  %470 = insertelement <4 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !17
  %471 = shufflevector <4 x i1> %470, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !17
  %472 = insertelement <4 x i1> poison, i1 %compare.high.order, i64 0, !dbg !17
  %473 = shufflevector <4 x i1> %472, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !17
  %474 = select <4 x i1> %471, <4 x i1> %433, <4 x i1> %473, !dbg !17
  %475 = select <4 x i1> %474, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %476 = select <4 x i1> %471, <4 x i1> %469, <4 x i1> %473, !dbg !17
  %477 = select <4 x i1> %476, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %478 = select <4 x i1> %471, <4 x i1> %463, <4 x i1> %473, !dbg !17
  %479 = select <4 x i1> %478, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %480 = select <4 x i1> %471, <4 x i1> %457, <4 x i1> %473, !dbg !17
  %481 = select <4 x i1> %480, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %482 = select <4 x i1> %471, <4 x i1> %451, <4 x i1> %473, !dbg !17
  %483 = select <4 x i1> %482, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %484 = select <4 x i1> %471, <4 x i1> %445, <4 x i1> %473, !dbg !17
  %485 = select <4 x i1> %484, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %486 = select <4 x i1> %471, <4 x i1> %439, <4 x i1> %473, !dbg !17
  %487 = select <4 x i1> %486, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %488 = insertelement <4 x i32> poison, i32 %compare.local.base1376, i64 0, !dbg !17
  %489 = insertelement <4 x i32> %488, i32 %compare.local.base1380, i64 1, !dbg !17
  %490 = insertelement <4 x i32> %489, i32 %compare.local.base1384, i64 2, !dbg !17
  %491 = insertelement <4 x i32> %490, i32 %compare.local.base1388, i64 3, !dbg !17
  %492 = or <4 x i32> %491, <i32 76, i32 77, i32 78, i32 79>, !dbg !17
  %493 = icmp ult <4 x i32> %492, %140, !dbg !17
  %494 = select <4 x i1> %471, <4 x i1> %493, <4 x i1> %473, !dbg !17
  %495 = select <4 x i1> %494, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %compare.local.base1400 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 32) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1404 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 33) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1408 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 34) #8, !dbg !17, !frisk.register_lifetimes !5
  %496 = insertelement <4 x i32> %428, i32 %compare.local.base1400, i64 1, !dbg !17
  %497 = insertelement <4 x i32> %496, i32 %compare.local.base1404, i64 2, !dbg !17
  %498 = insertelement <4 x i32> %497, i32 %compare.local.base1408, i64 3, !dbg !17
  %499 = or <4 x i32> %498, <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %500 = icmp ugt <4 x i32> %138, %499, !dbg !17
  %compare.local.base1412 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 35) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1416 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 36) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1420 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 37) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1424 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 38) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1428 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 39) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1432 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 40) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1436 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 41) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1440 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 42) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1444 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 43) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1448 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 44) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1452 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 45) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1456 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 46) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1460 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 47) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1464 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 48) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1468 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 49) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1472 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 50) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1476 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 51) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1480 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 52) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1484 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 53) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1488 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 54) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1492 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 55) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1496 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 56) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1500 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 57) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1504 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 58) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1508 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 59) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1512 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 60) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1516 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 61) #8, !dbg !17, !frisk.register_lifetimes !5
  %compare.local.base1520 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 62) #8, !dbg !17, !frisk.register_lifetimes !5
  %501 = insertelement <4 x i32> poison, i32 %compare.local.base1492, i64 0, !dbg !17
  %502 = insertelement <4 x i32> %501, i32 %compare.local.base1496, i64 1, !dbg !17
  %503 = insertelement <4 x i32> %502, i32 %compare.local.base1500, i64 2, !dbg !17
  %504 = insertelement <4 x i32> %503, i32 %compare.local.base1504, i64 3, !dbg !17
  %505 = or <4 x i32> %504, <i32 72, i32 73, i32 74, i32 75>, !dbg !17
  %506 = icmp ult <4 x i32> %505, %138, !dbg !17
  %507 = insertelement <4 x i32> poison, i32 %compare.local.base1476, i64 0, !dbg !17
  %508 = insertelement <4 x i32> %507, i32 %compare.local.base1480, i64 1, !dbg !17
  %509 = insertelement <4 x i32> %508, i32 %compare.local.base1484, i64 2, !dbg !17
  %510 = insertelement <4 x i32> %509, i32 %compare.local.base1488, i64 3, !dbg !17
  %511 = or <4 x i32> %510, <i32 68, i32 69, i32 70, i32 71>, !dbg !17
  %512 = icmp ult <4 x i32> %511, %138, !dbg !17
  %513 = insertelement <4 x i32> poison, i32 %compare.local.base1460, i64 0, !dbg !17
  %514 = insertelement <4 x i32> %513, i32 %compare.local.base1464, i64 1, !dbg !17
  %515 = insertelement <4 x i32> %514, i32 %compare.local.base1468, i64 2, !dbg !17
  %516 = insertelement <4 x i32> %515, i32 %compare.local.base1472, i64 3, !dbg !17
  %517 = or <4 x i32> %516, <i32 64, i32 65, i32 66, i32 67>, !dbg !17
  %518 = icmp ult <4 x i32> %517, %138, !dbg !17
  %519 = insertelement <4 x i32> poison, i32 %compare.local.base1444, i64 0, !dbg !17
  %520 = insertelement <4 x i32> %519, i32 %compare.local.base1448, i64 1, !dbg !17
  %521 = insertelement <4 x i32> %520, i32 %compare.local.base1452, i64 2, !dbg !17
  %522 = insertelement <4 x i32> %521, i32 %compare.local.base1456, i64 3, !dbg !17
  %523 = or <4 x i32> %522, <i32 12, i32 13, i32 14, i32 15>, !dbg !17
  %524 = icmp ult <4 x i32> %523, %138, !dbg !17
  %525 = insertelement <4 x i32> poison, i32 %compare.local.base1428, i64 0, !dbg !17
  %526 = insertelement <4 x i32> %525, i32 %compare.local.base1432, i64 1, !dbg !17
  %527 = insertelement <4 x i32> %526, i32 %compare.local.base1436, i64 2, !dbg !17
  %528 = insertelement <4 x i32> %527, i32 %compare.local.base1440, i64 3, !dbg !17
  %529 = or <4 x i32> %528, <i32 8, i32 9, i32 10, i32 11>, !dbg !17
  %530 = icmp ult <4 x i32> %529, %138, !dbg !17
  %531 = insertelement <4 x i32> poison, i32 %compare.local.base1412, i64 0, !dbg !17
  %532 = insertelement <4 x i32> %531, i32 %compare.local.base1416, i64 1, !dbg !17
  %533 = insertelement <4 x i32> %532, i32 %compare.local.base1420, i64 2, !dbg !17
  %534 = insertelement <4 x i32> %533, i32 %compare.local.base1424, i64 3, !dbg !17
  %535 = or <4 x i32> %534, <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %536 = icmp ult <4 x i32> %535, %138, !dbg !17
  %compare.high.equal1394 = icmp eq i32 %compare.high1392, %compare.high1268, !dbg !17
  %compare.high.order1395 = icmp ugt i32 %compare.high1392, %compare.high1268, !dbg !17
  %537 = insertelement <4 x i1> poison, i1 %compare.high.equal1394, i64 0, !dbg !17
  %538 = shufflevector <4 x i1> %537, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !17
  %539 = insertelement <4 x i1> poison, i1 %compare.high.order1395, i64 0, !dbg !17
  %540 = shufflevector <4 x i1> %539, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !17
  %541 = select <4 x i1> %538, <4 x i1> %500, <4 x i1> %540, !dbg !17
  %542 = select <4 x i1> %541, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %543 = select <4 x i1> %538, <4 x i1> %536, <4 x i1> %540, !dbg !17
  %544 = select <4 x i1> %543, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %545 = select <4 x i1> %538, <4 x i1> %530, <4 x i1> %540, !dbg !17
  %546 = select <4 x i1> %545, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %547 = select <4 x i1> %538, <4 x i1> %524, <4 x i1> %540, !dbg !17
  %548 = select <4 x i1> %547, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %549 = select <4 x i1> %538, <4 x i1> %518, <4 x i1> %540, !dbg !17
  %550 = select <4 x i1> %549, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %551 = select <4 x i1> %538, <4 x i1> %512, <4 x i1> %540, !dbg !17
  %552 = select <4 x i1> %551, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %553 = select <4 x i1> %538, <4 x i1> %506, <4 x i1> %540, !dbg !17
  %554 = select <4 x i1> %553, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %555 = insertelement <4 x i32> poison, i32 %compare.local.base1508, i64 0, !dbg !17
  %556 = insertelement <4 x i32> %555, i32 %compare.local.base1512, i64 1, !dbg !17
  %557 = insertelement <4 x i32> %556, i32 %compare.local.base1516, i64 2, !dbg !17
  %558 = insertelement <4 x i32> %557, i32 %compare.local.base1520, i64 3, !dbg !17
  %559 = or <4 x i32> %558, <i32 76, i32 77, i32 78, i32 79>, !dbg !17
  %560 = icmp ult <4 x i32> %559, %138, !dbg !17
  %561 = select <4 x i1> %538, <4 x i1> %560, <4 x i1> %540, !dbg !17
  %562 = select <4 x i1> %561, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !42
  %563 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !14, !frisk.k_tail_pair !18
  %564 = load <4 x half>, ptr addrspace(3) %kdma.gep2167, align 8, !dbg !14, !frisk.k_tail_pair !18
  %565 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !14, !frisk.k_tail_pair !19
  %566 = load <4 x half>, ptr addrspace(3) %kdma.gep2168, align 8, !dbg !14, !frisk.k_tail_pair !19
  %567 = load <4 x half>, ptr addrspace(3) %kdma.gep2165, align 8, !dbg !14, !frisk.k_tail_pair !20
  %568 = load <4 x half>, ptr addrspace(3) %kdma.gep2169, align 8, !dbg !14, !frisk.k_tail_pair !20
  %569 = load <4 x half>, ptr addrspace(3) %kdma.gep2166, align 8, !dbg !14, !frisk.k_tail_pair !21
  %570 = load <4 x half>, ptr addrspace(3) %kdma.gep2170, align 8, !dbg !14, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %571 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %563, <4 x float> %425) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %572 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %565, <4 x float> %388) #8, !dbg !14
  %573 = fadd <4 x float> %475, %571, !dbg !43
  %574 = extractelement <4 x float> %573, i64 0, !dbg !43
  %575 = tail call float @llvm.maxnum.f32(float %574, float -inf), !dbg !44, !frisk.local_reduction !5
  %576 = extractelement <4 x float> %573, i64 1, !dbg !43
  %577 = extractelement <4 x float> %573, i64 2, !dbg !43
  %578 = extractelement <4 x float> %573, i64 3, !dbg !43
  %579 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %567, <4 x float> %572) #8, !dbg !14
  %580 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %569, <4 x float> %390) #8, !dbg !14
  %581 = fadd <4 x float> %477, %579, !dbg !43
  %582 = extractelement <4 x float> %581, i64 0, !dbg !43
  %local.reduce218 = tail call float @llvm.maxnum.f32(float %575, float %582), !dbg !44, !frisk.local_reduction !5
  %583 = extractelement <4 x float> %581, i64 1, !dbg !43
  %local.reduce219 = tail call float @llvm.maxnum.f32(float %576, float %583), !dbg !44, !frisk.local_reduction !5
  %584 = extractelement <4 x float> %581, i64 2, !dbg !43
  %local.reduce220 = tail call float @llvm.maxnum.f32(float %577, float %584), !dbg !44, !frisk.local_reduction !5
  %585 = extractelement <4 x float> %581, i64 3, !dbg !43
  %local.reduce221 = tail call float @llvm.maxnum.f32(float %578, float %585), !dbg !44, !frisk.local_reduction !5
  %586 = load <4 x half>, ptr addrspace(3) %kdma.gep2171, align 8, !dbg !14, !frisk.k_tail_pair !22
  %587 = load <4 x half>, ptr addrspace(3) %kdma.gep2175, align 8, !dbg !14, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %564, <4 x float> %580) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %589 = load <4 x half>, ptr addrspace(3) %kdma.gep2172, align 8, !dbg !14, !frisk.k_tail_pair !23
  %590 = load <4 x half>, ptr addrspace(3) %kdma.gep2176, align 8, !dbg !14, !frisk.k_tail_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %591 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %566, <4 x float> %392) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.k_tail_prefetch !5
  %592 = fadd <4 x float> %479, %588, !dbg !43
  %593 = extractelement <4 x float> %592, i64 0, !dbg !43
  %local.reduce222 = tail call float @llvm.maxnum.f32(float %local.reduce218, float %593), !dbg !44, !frisk.local_reduction !5
  %594 = extractelement <4 x float> %592, i64 1, !dbg !43
  %local.reduce223 = tail call float @llvm.maxnum.f32(float %local.reduce219, float %594), !dbg !44, !frisk.local_reduction !5
  %595 = extractelement <4 x float> %592, i64 2, !dbg !43
  %local.reduce224 = tail call float @llvm.maxnum.f32(float %local.reduce220, float %595), !dbg !44, !frisk.local_reduction !5
  %596 = extractelement <4 x float> %592, i64 3, !dbg !43
  %local.reduce225 = tail call float @llvm.maxnum.f32(float %local.reduce221, float %596), !dbg !44, !frisk.local_reduction !5
  %597 = load <4 x half>, ptr addrspace(3) %kdma.gep2173, align 8, !dbg !14, !frisk.k_tail_pair !24
  %598 = load <4 x half>, ptr addrspace(3) %kdma.gep2177, align 8, !dbg !14, !frisk.k_tail_pair !24
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %599 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %568, <4 x float> %591) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %600 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %570, <4 x float> %394) #8, !dbg !14
  %601 = fadd <4 x float> %481, %599, !dbg !43
  %602 = extractelement <4 x float> %601, i64 0, !dbg !43
  %local.reduce226 = tail call float @llvm.maxnum.f32(float %local.reduce222, float %602), !dbg !44, !frisk.local_reduction !5
  %603 = extractelement <4 x float> %601, i64 1, !dbg !43
  %local.reduce227 = tail call float @llvm.maxnum.f32(float %local.reduce223, float %603), !dbg !44, !frisk.local_reduction !5
  %604 = extractelement <4 x float> %601, i64 2, !dbg !43
  %local.reduce228 = tail call float @llvm.maxnum.f32(float %local.reduce224, float %604), !dbg !44, !frisk.local_reduction !5
  %605 = extractelement <4 x float> %601, i64 3, !dbg !43
  %local.reduce229 = tail call float @llvm.maxnum.f32(float %local.reduce225, float %605), !dbg !44, !frisk.local_reduction !5
  %606 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %586, <4 x float> %600) #8, !dbg !14
  %607 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %589, <4 x float> %396) #8, !dbg !14
  %608 = fadd <4 x float> %483, %606, !dbg !43
  %609 = extractelement <4 x float> %608, i64 0, !dbg !43
  %local.reduce230 = tail call float @llvm.maxnum.f32(float %local.reduce226, float %609), !dbg !44, !frisk.local_reduction !5
  %610 = extractelement <4 x float> %608, i64 1, !dbg !43
  %local.reduce231 = tail call float @llvm.maxnum.f32(float %local.reduce227, float %610), !dbg !44, !frisk.local_reduction !5
  %611 = extractelement <4 x float> %608, i64 2, !dbg !43
  %local.reduce232 = tail call float @llvm.maxnum.f32(float %local.reduce228, float %611), !dbg !44, !frisk.local_reduction !5
  %612 = extractelement <4 x float> %608, i64 3, !dbg !43
  %local.reduce233 = tail call float @llvm.maxnum.f32(float %local.reduce229, float %612), !dbg !44, !frisk.local_reduction !5
  %613 = load <4 x half>, ptr addrspace(3) %kdma.gep2174, align 8, !dbg !14
  %614 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %597, <4 x float> %607) #8, !dbg !14
  %615 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %613, <4 x float> %398) #8, !dbg !14
  %616 = fadd <4 x float> %485, %614, !dbg !43
  %617 = extractelement <4 x float> %616, i64 0, !dbg !43
  %local.reduce234 = tail call float @llvm.maxnum.f32(float %local.reduce230, float %617), !dbg !44, !frisk.local_reduction !5
  %618 = extractelement <4 x float> %616, i64 1, !dbg !43
  %local.reduce235 = tail call float @llvm.maxnum.f32(float %local.reduce231, float %618), !dbg !44, !frisk.local_reduction !5
  %619 = extractelement <4 x float> %616, i64 2, !dbg !43
  %local.reduce236 = tail call float @llvm.maxnum.f32(float %local.reduce232, float %619), !dbg !44, !frisk.local_reduction !5
  %620 = extractelement <4 x float> %616, i64 3, !dbg !43
  %local.reduce237 = tail call float @llvm.maxnum.f32(float %local.reduce233, float %620), !dbg !44, !frisk.local_reduction !5
  %621 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %587, <4 x float> %615) #8, !dbg !14
  %622 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %590, <4 x float> %400) #8, !dbg !14
  %623 = fadd <4 x float> %487, %621, !dbg !43
  %624 = extractelement <4 x float> %623, i64 0, !dbg !43
  %local.reduce238 = tail call float @llvm.maxnum.f32(float %local.reduce234, float %624), !dbg !44, !frisk.local_reduction !5
  %625 = extractelement <4 x float> %623, i64 1, !dbg !43
  %local.reduce239 = tail call float @llvm.maxnum.f32(float %local.reduce235, float %625), !dbg !44, !frisk.local_reduction !5
  %626 = extractelement <4 x float> %623, i64 2, !dbg !43
  %local.reduce240 = tail call float @llvm.maxnum.f32(float %local.reduce236, float %626), !dbg !44, !frisk.local_reduction !5
  %627 = extractelement <4 x float> %623, i64 3, !dbg !43
  %local.reduce241 = tail call float @llvm.maxnum.f32(float %local.reduce237, float %627), !dbg !44, !frisk.local_reduction !5
  %628 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %598, <4 x float> %622) #8, !dbg !14
  %629 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %424, <4 x float> %402) #8, !dbg !14
  %630 = fadd <4 x float> %495, %628, !dbg !43
  %631 = extractelement <4 x float> %630, i64 0, !dbg !43
  %local.reduce242 = tail call float @llvm.maxnum.f32(float %local.reduce238, float %631), !dbg !44, !frisk.local_reduction !5
  %632 = extractelement <4 x float> %630, i64 1, !dbg !43
  %local.reduce243 = tail call float @llvm.maxnum.f32(float %local.reduce239, float %632), !dbg !44, !frisk.local_reduction !5
  %633 = extractelement <4 x float> %630, i64 2, !dbg !43
  %local.reduce244 = tail call float @llvm.maxnum.f32(float %local.reduce240, float %633), !dbg !44, !frisk.local_reduction !5
  %634 = extractelement <4 x float> %630, i64 3, !dbg !43
  %local.reduce245 = tail call float @llvm.maxnum.f32(float %local.reduce241, float %634), !dbg !44, !frisk.local_reduction !5
  %local.reduce246 = tail call float @llvm.maxnum.f32(float %local.reduce242, float %local.reduce243), !dbg !44, !frisk.local_reduction !5
  %local.reduce247 = tail call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !44, !frisk.local_reduction !5
  %635 = tail call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !44, !frisk.local_reduction !5
  %636 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %563, <4 x float> %629) #8, !dbg !14
  %637 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %565, <4 x float> %404) #8, !dbg !14
  %638 = fadd <4 x float> %542, %636, !dbg !43
  %639 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %567, <4 x float> %637) #8, !dbg !14
  %640 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %569, <4 x float> %406) #8, !dbg !14
  %641 = fadd <4 x float> %544, %639, !dbg !43
  %642 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %564, <4 x float> %640) #8, !dbg !14
  %643 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %566, <4 x float> %408) #8, !dbg !14
  %644 = fadd <4 x float> %546, %642, !dbg !43
  %645 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %568, <4 x float> %643) #8, !dbg !14
  %646 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %570, <4 x float> %410) #8, !dbg !14
  %647 = fadd <4 x float> %548, %645, !dbg !43
  %648 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %586, <4 x float> %646) #8, !dbg !14
  %649 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %589, <4 x float> %412) #8, !dbg !14
  %650 = fadd <4 x float> %550, %648, !dbg !43
  %651 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %597, <4 x float> %649) #8, !dbg !14
  %652 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %613, <4 x float> %414) #8, !dbg !14
  %653 = fadd <4 x float> %552, %651, !dbg !43
  %654 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %587, <4 x float> %652) #8, !dbg !14
  %655 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %590, <4 x float> %416) #8, !dbg !14
  %656 = fadd <4 x float> %554, %654, !dbg !43
  %657 = extractelement <4 x float> %638, i64 0, !dbg !43
  %658 = extractelement <4 x float> %638, i64 1, !dbg !43
  %659 = extractelement <4 x float> %638, i64 2, !dbg !43
  %660 = extractelement <4 x float> %638, i64 3, !dbg !43
  %661 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %598, <4 x float> %655) #8, !dbg !14
  %662 = extractelement <4 x float> %641, i64 0, !dbg !43
  %663 = extractelement <4 x float> %641, i64 1, !dbg !43
  %664 = extractelement <4 x float> %641, i64 2, !dbg !43
  %665 = extractelement <4 x float> %641, i64 3, !dbg !43
  %666 = extractelement <4 x float> %644, i64 0, !dbg !43
  %667 = extractelement <4 x float> %644, i64 1, !dbg !43
  %668 = extractelement <4 x float> %644, i64 2, !dbg !43
  %669 = extractelement <4 x float> %644, i64 3, !dbg !43
  %670 = fadd <4 x float> %562, %661, !dbg !43
  %671 = extractelement <4 x float> %647, i64 0, !dbg !43
  %672 = extractelement <4 x float> %647, i64 1, !dbg !43
  %673 = extractelement <4 x float> %647, i64 2, !dbg !43
  %674 = extractelement <4 x float> %647, i64 3, !dbg !43
  %675 = extractelement <4 x float> %650, i64 0, !dbg !43
  %676 = extractelement <4 x float> %650, i64 1, !dbg !43
  %677 = extractelement <4 x float> %650, i64 2, !dbg !43
  %678 = extractelement <4 x float> %650, i64 3, !dbg !43
  %679 = extractelement <4 x float> %653, i64 0, !dbg !43
  %680 = extractelement <4 x float> %653, i64 1, !dbg !43
  %681 = extractelement <4 x float> %653, i64 2, !dbg !43
  %682 = extractelement <4 x float> %653, i64 3, !dbg !43
  %683 = extractelement <4 x float> %656, i64 0, !dbg !43
  %684 = extractelement <4 x float> %656, i64 1, !dbg !43
  %685 = extractelement <4 x float> %656, i64 2, !dbg !43
  %686 = extractelement <4 x float> %656, i64 3, !dbg !43
  %687 = extractelement <4 x float> %670, i64 0, !dbg !43
  %688 = extractelement <4 x float> %670, i64 1, !dbg !43
  %689 = extractelement <4 x float> %670, i64 2, !dbg !43
  %690 = extractelement <4 x float> %670, i64 3, !dbg !43
  %691 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !44
  %692 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %691), !dbg !44
  %693 = add nuw nsw i32 %692, 64, !dbg !44
  %694 = xor i32 %692, 16, !dbg !44
  %695 = and i32 %693, 192, !dbg !44
  %696 = icmp samesign ult i32 %694, %695, !dbg !44
  %697 = select i1 %696, i32 %694, i32 %692, !dbg !44
  %698 = shl nuw nsw i32 %697, 2, !dbg !44
  %699 = bitcast float %635 to i32, !dbg !44
  %700 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %698, i32 %699), !dbg !44
  %701 = bitcast i32 %700 to float, !dbg !44
  %702 = tail call float @llvm.maxnum.f32(float %635, float %701), !dbg !44
  %703 = xor i32 %692, 32, !dbg !44
  %704 = icmp samesign ult i32 %703, %695, !dbg !44
  %705 = select i1 %704, i32 %703, i32 %692, !dbg !44
  %706 = shl nuw nsw i32 %705, 2, !dbg !44
  %707 = bitcast float %702 to i32, !dbg !44
  %708 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %706, i32 %707), !dbg !44
  %709 = bitcast i32 %708 to float, !dbg !44
  %710 = tail call float @llvm.maxnum.f32(float %702, float %709), !dbg !44
  %711 = tail call float @llvm.maxnum.f32(float %657, float -inf), !dbg !44, !frisk.local_reduction !5
  %local.reduce187 = tail call float @llvm.maxnum.f32(float %711, float %662), !dbg !44, !frisk.local_reduction !5
  %local.reduce188 = tail call float @llvm.maxnum.f32(float %658, float %663), !dbg !44, !frisk.local_reduction !5
  %local.reduce189 = tail call float @llvm.maxnum.f32(float %659, float %664), !dbg !44, !frisk.local_reduction !5
  %local.reduce190 = tail call float @llvm.maxnum.f32(float %660, float %665), !dbg !44, !frisk.local_reduction !5
  %local.reduce191 = tail call float @llvm.maxnum.f32(float %local.reduce187, float %666), !dbg !44, !frisk.local_reduction !5
  %local.reduce192 = tail call float @llvm.maxnum.f32(float %local.reduce188, float %667), !dbg !44, !frisk.local_reduction !5
  %local.reduce193 = tail call float @llvm.maxnum.f32(float %local.reduce189, float %668), !dbg !44, !frisk.local_reduction !5
  %local.reduce194 = tail call float @llvm.maxnum.f32(float %local.reduce190, float %669), !dbg !44, !frisk.local_reduction !5
  %local.reduce195 = tail call float @llvm.maxnum.f32(float %local.reduce191, float %671), !dbg !44, !frisk.local_reduction !5
  %local.reduce196 = tail call float @llvm.maxnum.f32(float %local.reduce192, float %672), !dbg !44, !frisk.local_reduction !5
  %local.reduce197 = tail call float @llvm.maxnum.f32(float %local.reduce193, float %673), !dbg !44, !frisk.local_reduction !5
  %local.reduce198 = tail call float @llvm.maxnum.f32(float %local.reduce194, float %674), !dbg !44, !frisk.local_reduction !5
  %local.reduce199 = tail call float @llvm.maxnum.f32(float %local.reduce195, float %675), !dbg !44, !frisk.local_reduction !5
  %local.reduce200 = tail call float @llvm.maxnum.f32(float %local.reduce196, float %676), !dbg !44, !frisk.local_reduction !5
  %local.reduce201 = tail call float @llvm.maxnum.f32(float %local.reduce197, float %677), !dbg !44, !frisk.local_reduction !5
  %local.reduce202 = tail call float @llvm.maxnum.f32(float %local.reduce198, float %678), !dbg !44, !frisk.local_reduction !5
  %local.reduce203 = tail call float @llvm.maxnum.f32(float %local.reduce199, float %679), !dbg !44, !frisk.local_reduction !5
  %local.reduce204 = tail call float @llvm.maxnum.f32(float %local.reduce200, float %680), !dbg !44, !frisk.local_reduction !5
  %local.reduce205 = tail call float @llvm.maxnum.f32(float %local.reduce201, float %681), !dbg !44, !frisk.local_reduction !5
  %local.reduce206 = tail call float @llvm.maxnum.f32(float %local.reduce202, float %682), !dbg !44, !frisk.local_reduction !5
  %local.reduce207 = tail call float @llvm.maxnum.f32(float %local.reduce203, float %683), !dbg !44, !frisk.local_reduction !5
  %local.reduce208 = tail call float @llvm.maxnum.f32(float %local.reduce204, float %684), !dbg !44, !frisk.local_reduction !5
  %local.reduce209 = tail call float @llvm.maxnum.f32(float %local.reduce205, float %685), !dbg !44, !frisk.local_reduction !5
  %local.reduce210 = tail call float @llvm.maxnum.f32(float %local.reduce206, float %686), !dbg !44, !frisk.local_reduction !5
  %local.reduce211 = tail call float @llvm.maxnum.f32(float %local.reduce207, float %687), !dbg !44, !frisk.local_reduction !5
  %local.reduce212 = tail call float @llvm.maxnum.f32(float %local.reduce208, float %688), !dbg !44, !frisk.local_reduction !5
  %local.reduce213 = tail call float @llvm.maxnum.f32(float %local.reduce209, float %689), !dbg !44, !frisk.local_reduction !5
  %local.reduce214 = tail call float @llvm.maxnum.f32(float %local.reduce210, float %690), !dbg !44, !frisk.local_reduction !5
  %local.reduce215 = tail call float @llvm.maxnum.f32(float %local.reduce211, float %local.reduce212), !dbg !44, !frisk.local_reduction !5
  %local.reduce216 = tail call float @llvm.maxnum.f32(float %local.reduce213, float %local.reduce214), !dbg !44, !frisk.local_reduction !5
  %712 = tail call float @llvm.maxnum.f32(float %local.reduce215, float %local.reduce216), !dbg !44, !frisk.local_reduction !5
  %713 = bitcast float %712 to i32, !dbg !44
  %714 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %698, i32 %713), !dbg !44
  %715 = bitcast i32 %714 to float, !dbg !44
  %716 = tail call float @llvm.maxnum.f32(float %712, float %715), !dbg !44
  %717 = bitcast float %716 to i32, !dbg !44
  %718 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %706, i32 %717), !dbg !44
  %719 = bitcast i32 %718 to float, !dbg !44
  %720 = tail call float @llvm.maxnum.f32(float %716, float %719), !dbg !44
  %721 = extractvalue [2 x <1 x float>] %146, 0, !dbg !44
  %722 = insertelement <1 x float> poison, float %710, i64 0, !dbg !44
  %723 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %722, <1 x float> %721), !dbg !44
  %724 = extractvalue [2 x <1 x float>] %146, 1, !dbg !44
  %725 = insertelement <1 x float> poison, float %720, i64 0, !dbg !44
  %726 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %725, <1 x float> %724), !dbg !44
  %727 = shufflevector <1 x float> %723, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !45
  %728 = shufflevector <1 x float> %726, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !45
  %729 = fsub <1 x float> %721, %723, !dbg !46
  %730 = extractelement <1 x float> %729, i64 0, !dbg !47
  %731 = tail call float @llvm.amdgcn.exp2.f32(float %730), !dbg !47
  %732 = insertelement <4 x float> poison, float %731, i64 0, !dbg !48
  %733 = shufflevector <4 x float> %732, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !48
  %734 = extractvalue [2 x <1 x float>] %147, 0, !dbg !49
  %735 = extractelement <1 x float> %734, i64 0, !dbg !49
  %736 = fmul float %735, %731, !dbg !49
  %737 = fsub <1 x float> %724, %726, !dbg !46
  %738 = extractelement <1 x float> %737, i64 0, !dbg !47
  %739 = tail call float @llvm.amdgcn.exp2.f32(float %738), !dbg !47
  %740 = insertelement <4 x float> poison, float %739, i64 0, !dbg !48
  %741 = shufflevector <4 x float> %740, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !48
  %742 = extractvalue [2 x <1 x float>] %147, 1, !dbg !49
  %743 = extractelement <1 x float> %742, i64 0, !dbg !49
  %744 = fmul float %743, %739, !dbg !49
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  tail call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %vdb.address1788 = or disjoint i64 %.idx17641766, 32, !dbg !41
  %vdb.address1789 = or disjoint i64 %address.local988, %vdb.address1788, !dbg !41
  %vdb.address1790 = trunc i64 %vdb.address1789 to i32, !dbg !41
  %vdb.address1791 = shl i32 %vdb.address1790, 1, !dbg !41
  %vdb.address1798 = or disjoint i64 %address.local994, %vdb.address1788, !dbg !41
  %vdb.address1799 = trunc i64 %vdb.address1798 to i32, !dbg !41
  %vdb.address1800 = shl i32 %vdb.address1799, 1, !dbg !41
  %vdb.dma.base = add i32 %183, 8192, !dbg !50
  %745 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1791, <4 x i32> %dma.buffer.rsrc1670) #9, !dbg !41
  %vdb.dma.base1805 = add i32 %187, 8192, !dbg !50
  %746 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1805, i32 %vdb.address1800, <4 x i32> %dma.buffer.rsrc1670) #9, !dbg !41
  %address.local996 = shl nuw nsw i32 %address.thread565, 3, !dbg !50
  %address.local997 = and i32 %address.local996, 120, !dbg !50
  %address.local999 = and i32 %address.local642, 1536, !dbg !50
  %address.local1000 = or disjoint i32 %address.local997, %address.local999, !dbg !50
  %address.local1001 = zext nneg i32 %address.local1000 to i64, !dbg !50
  %address.local1002 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1001, !dbg !50
  %747 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %748 = bitcast <4 x i32> %747 to <8 x half>, !dbg !50
  %address.local1003 = and i64 %address.local566, 60, !dbg !50
  %address.local1004 = add nuw nsw i64 %address.local1003, %address.local566, !dbg !50
  %address.local1006.idx = shl nuw nsw i64 %address.local1004, 4, !dbg !50
  %address.local1006 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local1006.idx, !dbg !50
  %address.local1007 = lshr i64 %address.local622, 4, !dbg !50
  %address.local1009.idx = mul nuw nsw i64 %address.local1007, 768, !dbg !50
  %address.local1009 = getelementptr i8, ptr addrspace(3) %address.local1006, i64 %address.local1009.idx, !dbg !50
  %address.local1011.idx = mul nsw i64 %address.local609, -64, !dbg !50
  %address.local1011 = getelementptr i8, ptr addrspace(3) %address.local1009, i64 %address.local1011.idx, !dbg !50
  %address.local1012 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 256, !dbg !50
  %749 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %750 = bitcast <4 x i32> %749 to <8 x half>, !dbg !50
  %address.local1016 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 512, !dbg !50
  %751 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %752 = bitcast <4 x i32> %751 to <8 x half>, !dbg !50
  %address.local1020 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 768, !dbg !50
  %753 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %754 = bitcast <4 x i32> %753 to <8 x half>, !dbg !50
  %address.local1022 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 4096, !dbg !50
  %755 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %756 = bitcast <4 x i32> %755 to <8 x half>, !dbg !50
  %address.local1026 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4352, !dbg !50
  %757 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %758 = bitcast <4 x i32> %757 to <8 x half>, !dbg !50
  %address.local1030 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4608, !dbg !50
  %759 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %760 = bitcast <4 x i32> %759 to <8 x half>, !dbg !50
  %address.local1034 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4864, !dbg !50
  %761 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %762 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  %763 = extractvalue [2 x <8 x float>] %148, 0, !dbg !48
  %764 = shufflevector <8 x float> %763, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %765 = fmul <4 x float> %764, %733, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %766 = fsub <4 x float> %573, %727, !dbg !45
  %767 = extractelement <4 x float> %766, i64 0, !dbg !52
  %768 = tail call float @llvm.amdgcn.exp2.f32(float %767), !dbg !52
  %769 = insertelement <4 x float> poison, float %768, i64 0, !dbg !53
  %770 = extractelement <4 x float> %766, i64 1, !dbg !52
  %771 = tail call float @llvm.amdgcn.exp2.f32(float %770), !dbg !52
  %772 = insertelement <4 x float> %769, float %771, i64 1, !dbg !53
  %773 = extractelement <4 x float> %766, i64 2, !dbg !52
  %774 = tail call float @llvm.amdgcn.exp2.f32(float %773), !dbg !52
  %775 = insertelement <4 x float> %772, float %774, i64 2, !dbg !53
  %776 = extractelement <4 x float> %766, i64 3, !dbg !52
  %777 = tail call float @llvm.amdgcn.exp2.f32(float %776), !dbg !52
  %778 = insertelement <4 x float> %775, float %777, i64 3, !dbg !53
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %778) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !53
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !53
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !53
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !53
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !53
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !53
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !53
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !53
  %779 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !53
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %779) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %780 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %762, <4 x float> %765) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %781 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  %782 = shufflevector <8 x float> %763, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %783 = fmul <4 x float> %782, %733, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %784 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %781, <4 x float> %783) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %785 = extractvalue [2 x <8 x float>] %148, 1, !dbg !48
  %786 = shufflevector <8 x float> %785, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %787 = fmul <4 x float> %786, %741, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %788 = fsub <4 x float> %638, %728, !dbg !45
  %789 = extractelement <4 x float> %788, i64 0, !dbg !52
  %790 = tail call float @llvm.amdgcn.exp2.f32(float %789), !dbg !52
  %791 = insertelement <4 x float> poison, float %790, i64 0, !dbg !53
  %792 = extractelement <4 x float> %788, i64 1, !dbg !52
  %793 = tail call float @llvm.amdgcn.exp2.f32(float %792), !dbg !52
  %794 = insertelement <4 x float> %791, float %793, i64 1, !dbg !53
  %795 = extractelement <4 x float> %788, i64 2, !dbg !52
  %796 = tail call float @llvm.amdgcn.exp2.f32(float %795), !dbg !52
  %797 = insertelement <4 x float> %794, float %796, i64 2, !dbg !53
  %798 = extractelement <4 x float> %788, i64 3, !dbg !52
  %799 = tail call float @llvm.amdgcn.exp2.f32(float %798), !dbg !52
  %800 = insertelement <4 x float> %797, float %799, i64 3, !dbg !53
  %math.result455 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %800) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02265 = extractelement <4 x float> %math.result455, i64 0, !dbg !53
  %.cvtpack.f12266 = extractelement <4 x float> %math.result455, i64 1, !dbg !53
  %.cvtpack.f22267 = extractelement <4 x float> %math.result455, i64 2, !dbg !53
  %.cvtpack.f32268 = extractelement <4 x float> %math.result455, i64 3, !dbg !53
  %.cvtpack.p02269 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02265, float %.cvtpack.f12266) #12, !dbg !53
  %.cvtpack.p12270 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22267, float %.cvtpack.f32268) #12, !dbg !53
  %.cvtpack.v02271 = insertelement <2 x i32> poison, i32 %.cvtpack.p02269, i64 0, !dbg !53
  %.cvtpack.v12272 = insertelement <2 x i32> %.cvtpack.v02271, i32 %.cvtpack.p12270, i64 1, !dbg !53
  %801 = bitcast <2 x i32> %.cvtpack.v12272 to <4 x half>, !dbg !53
  %math.packed456 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %801) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %802 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %762, <4 x float> %787) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %803 = shufflevector <8 x float> %785, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %804 = fmul <4 x float> %803, %741, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %805 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %781, <4 x float> %804) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %806 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %807 = fsub <4 x float> %581, %727, !dbg !45
  %808 = extractelement <4 x float> %807, i64 0, !dbg !52
  %809 = tail call float @llvm.amdgcn.exp2.f32(float %808), !dbg !52
  %local.reduce156 = fadd float %768, %809, !dbg !54, !frisk.local_reduction !5
  %810 = insertelement <4 x float> poison, float %809, i64 0, !dbg !53
  %811 = extractelement <4 x float> %807, i64 1, !dbg !52
  %812 = tail call float @llvm.amdgcn.exp2.f32(float %811), !dbg !52
  %local.reduce157 = fadd float %771, %812, !dbg !54, !frisk.local_reduction !5
  %813 = insertelement <4 x float> %810, float %812, i64 1, !dbg !53
  %814 = extractelement <4 x float> %807, i64 2, !dbg !52
  %815 = tail call float @llvm.amdgcn.exp2.f32(float %814), !dbg !52
  %local.reduce158 = fadd float %774, %815, !dbg !54, !frisk.local_reduction !5
  %816 = insertelement <4 x float> %813, float %815, i64 2, !dbg !53
  %817 = extractelement <4 x float> %807, i64 3, !dbg !52
  %818 = tail call float @llvm.amdgcn.exp2.f32(float %817), !dbg !52
  %local.reduce159 = fadd float %777, %818, !dbg !54, !frisk.local_reduction !5
  %819 = insertelement <4 x float> %816, float %818, i64 3, !dbg !53
  %math.result441 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %819) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02273 = extractelement <4 x float> %math.result441, i64 0, !dbg !53
  %.cvtpack.f12274 = extractelement <4 x float> %math.result441, i64 1, !dbg !53
  %.cvtpack.f22275 = extractelement <4 x float> %math.result441, i64 2, !dbg !53
  %.cvtpack.f32276 = extractelement <4 x float> %math.result441, i64 3, !dbg !53
  %.cvtpack.p02277 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02273, float %.cvtpack.f12274) #12, !dbg !53
  %.cvtpack.p12278 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22275, float %.cvtpack.f32276) #12, !dbg !53
  %.cvtpack.v02279 = insertelement <2 x i32> poison, i32 %.cvtpack.p02277, i64 0, !dbg !53
  %.cvtpack.v12280 = insertelement <2 x i32> %.cvtpack.v02279, i32 %.cvtpack.p12278, i64 1, !dbg !53
  %820 = bitcast <2 x i32> %.cvtpack.v12280 to <4 x half>, !dbg !53
  %math.packed442 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %820) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %821 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %806, <4 x float> %780) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %822 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %823 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %822, <4 x float> %784) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %824 = fsub <4 x float> %641, %728, !dbg !45
  %825 = extractelement <4 x float> %824, i64 0, !dbg !52
  %826 = tail call float @llvm.amdgcn.exp2.f32(float %825), !dbg !52
  %local.reduce = fadd float %790, %826, !dbg !54, !frisk.local_reduction !5
  %827 = insertelement <4 x float> poison, float %826, i64 0, !dbg !53
  %828 = extractelement <4 x float> %824, i64 1, !dbg !52
  %829 = tail call float @llvm.amdgcn.exp2.f32(float %828), !dbg !52
  %local.reduce126 = fadd float %793, %829, !dbg !54, !frisk.local_reduction !5
  %830 = insertelement <4 x float> %827, float %829, i64 1, !dbg !53
  %831 = extractelement <4 x float> %824, i64 2, !dbg !52
  %832 = tail call float @llvm.amdgcn.exp2.f32(float %831), !dbg !52
  %local.reduce127 = fadd float %796, %832, !dbg !54, !frisk.local_reduction !5
  %833 = insertelement <4 x float> %830, float %832, i64 2, !dbg !53
  %834 = extractelement <4 x float> %824, i64 3, !dbg !52
  %835 = tail call float @llvm.amdgcn.exp2.f32(float %834), !dbg !52
  %local.reduce128 = fadd float %799, %835, !dbg !54, !frisk.local_reduction !5
  %836 = insertelement <4 x float> %833, float %835, i64 3, !dbg !53
  %math.result457 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %836) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02281 = extractelement <4 x float> %math.result457, i64 0, !dbg !53
  %.cvtpack.f12282 = extractelement <4 x float> %math.result457, i64 1, !dbg !53
  %.cvtpack.f22283 = extractelement <4 x float> %math.result457, i64 2, !dbg !53
  %.cvtpack.f32284 = extractelement <4 x float> %math.result457, i64 3, !dbg !53
  %.cvtpack.p02285 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02281, float %.cvtpack.f12282) #12, !dbg !53
  %.cvtpack.p12286 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22283, float %.cvtpack.f32284) #12, !dbg !53
  %.cvtpack.v02287 = insertelement <2 x i32> poison, i32 %.cvtpack.p02285, i64 0, !dbg !53
  %.cvtpack.v12288 = insertelement <2 x i32> %.cvtpack.v02287, i32 %.cvtpack.p12286, i64 1, !dbg !53
  %837 = bitcast <2 x i32> %.cvtpack.v12288 to <4 x half>, !dbg !53
  %math.packed458 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %837) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %806, <4 x float> %802) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %839 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %822, <4 x float> %805) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %840 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %841 = fsub <4 x float> %592, %727, !dbg !45
  %842 = extractelement <4 x float> %841, i64 0, !dbg !52
  %843 = tail call float @llvm.amdgcn.exp2.f32(float %842), !dbg !52
  %local.reduce160 = fadd float %843, %local.reduce156, !dbg !54, !frisk.local_reduction !5
  %844 = insertelement <4 x float> poison, float %843, i64 0, !dbg !53
  %845 = extractelement <4 x float> %841, i64 1, !dbg !52
  %846 = tail call float @llvm.amdgcn.exp2.f32(float %845), !dbg !52
  %local.reduce161 = fadd float %846, %local.reduce157, !dbg !54, !frisk.local_reduction !5
  %847 = insertelement <4 x float> %844, float %846, i64 1, !dbg !53
  %848 = extractelement <4 x float> %841, i64 2, !dbg !52
  %849 = tail call float @llvm.amdgcn.exp2.f32(float %848), !dbg !52
  %local.reduce162 = fadd float %849, %local.reduce158, !dbg !54, !frisk.local_reduction !5
  %850 = insertelement <4 x float> %847, float %849, i64 2, !dbg !53
  %851 = extractelement <4 x float> %841, i64 3, !dbg !52
  %852 = tail call float @llvm.amdgcn.exp2.f32(float %851), !dbg !52
  %local.reduce163 = fadd float %852, %local.reduce159, !dbg !54, !frisk.local_reduction !5
  %853 = insertelement <4 x float> %850, float %852, i64 3, !dbg !53
  %math.result443 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %853) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02289 = extractelement <4 x float> %math.result443, i64 0, !dbg !53
  %.cvtpack.f12290 = extractelement <4 x float> %math.result443, i64 1, !dbg !53
  %.cvtpack.f22291 = extractelement <4 x float> %math.result443, i64 2, !dbg !53
  %.cvtpack.f32292 = extractelement <4 x float> %math.result443, i64 3, !dbg !53
  %.cvtpack.p02293 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02289, float %.cvtpack.f12290) #12, !dbg !53
  %.cvtpack.p12294 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22291, float %.cvtpack.f32292) #12, !dbg !53
  %.cvtpack.v02295 = insertelement <2 x i32> poison, i32 %.cvtpack.p02293, i64 0, !dbg !53
  %.cvtpack.v12296 = insertelement <2 x i32> %.cvtpack.v02295, i32 %.cvtpack.p12294, i64 1, !dbg !53
  %854 = bitcast <2 x i32> %.cvtpack.v12296 to <4 x half>, !dbg !53
  %math.packed444 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %854) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %855 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %840, <4 x float> %821) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %856 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %857 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %856, <4 x float> %823) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %858 = fsub <4 x float> %644, %728, !dbg !45
  %859 = extractelement <4 x float> %858, i64 0, !dbg !52
  %860 = tail call float @llvm.amdgcn.exp2.f32(float %859), !dbg !52
  %local.reduce129 = fadd float %860, %local.reduce, !dbg !54, !frisk.local_reduction !5
  %861 = insertelement <4 x float> poison, float %860, i64 0, !dbg !53
  %862 = extractelement <4 x float> %858, i64 1, !dbg !52
  %863 = tail call float @llvm.amdgcn.exp2.f32(float %862), !dbg !52
  %local.reduce130 = fadd float %863, %local.reduce126, !dbg !54, !frisk.local_reduction !5
  %864 = insertelement <4 x float> %861, float %863, i64 1, !dbg !53
  %865 = extractelement <4 x float> %858, i64 2, !dbg !52
  %866 = tail call float @llvm.amdgcn.exp2.f32(float %865), !dbg !52
  %local.reduce131 = fadd float %866, %local.reduce127, !dbg !54, !frisk.local_reduction !5
  %867 = insertelement <4 x float> %864, float %866, i64 2, !dbg !53
  %868 = extractelement <4 x float> %858, i64 3, !dbg !52
  %869 = tail call float @llvm.amdgcn.exp2.f32(float %868), !dbg !52
  %local.reduce132 = fadd float %869, %local.reduce128, !dbg !54, !frisk.local_reduction !5
  %870 = insertelement <4 x float> %867, float %869, i64 3, !dbg !53
  %math.result459 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %870) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02297 = extractelement <4 x float> %math.result459, i64 0, !dbg !53
  %.cvtpack.f12298 = extractelement <4 x float> %math.result459, i64 1, !dbg !53
  %.cvtpack.f22299 = extractelement <4 x float> %math.result459, i64 2, !dbg !53
  %.cvtpack.f32300 = extractelement <4 x float> %math.result459, i64 3, !dbg !53
  %.cvtpack.p02301 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02297, float %.cvtpack.f12298) #12, !dbg !53
  %.cvtpack.p12302 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22299, float %.cvtpack.f32300) #12, !dbg !53
  %.cvtpack.v02303 = insertelement <2 x i32> poison, i32 %.cvtpack.p02301, i64 0, !dbg !53
  %.cvtpack.v12304 = insertelement <2 x i32> %.cvtpack.v02303, i32 %.cvtpack.p12302, i64 1, !dbg !53
  %871 = bitcast <2 x i32> %.cvtpack.v12304 to <4 x half>, !dbg !53
  %math.packed460 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %871) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %872 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %840, <4 x float> %838) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %856, <4 x float> %839) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %874 = shufflevector <8 x half> %754, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %875 = fsub <4 x float> %601, %727, !dbg !45
  %876 = extractelement <4 x float> %875, i64 0, !dbg !52
  %877 = tail call float @llvm.amdgcn.exp2.f32(float %876), !dbg !52
  %local.reduce164 = fadd float %877, %local.reduce160, !dbg !54, !frisk.local_reduction !5
  %878 = insertelement <4 x float> poison, float %877, i64 0, !dbg !53
  %879 = extractelement <4 x float> %875, i64 1, !dbg !52
  %880 = tail call float @llvm.amdgcn.exp2.f32(float %879), !dbg !52
  %local.reduce165 = fadd float %880, %local.reduce161, !dbg !54, !frisk.local_reduction !5
  %881 = insertelement <4 x float> %878, float %880, i64 1, !dbg !53
  %882 = extractelement <4 x float> %875, i64 2, !dbg !52
  %883 = tail call float @llvm.amdgcn.exp2.f32(float %882), !dbg !52
  %local.reduce166 = fadd float %883, %local.reduce162, !dbg !54, !frisk.local_reduction !5
  %884 = insertelement <4 x float> %881, float %883, i64 2, !dbg !53
  %885 = extractelement <4 x float> %875, i64 3, !dbg !52
  %886 = tail call float @llvm.amdgcn.exp2.f32(float %885), !dbg !52
  %local.reduce167 = fadd float %886, %local.reduce163, !dbg !54, !frisk.local_reduction !5
  %887 = insertelement <4 x float> %884, float %886, i64 3, !dbg !53
  %math.result445 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %887) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02305 = extractelement <4 x float> %math.result445, i64 0, !dbg !53
  %.cvtpack.f12306 = extractelement <4 x float> %math.result445, i64 1, !dbg !53
  %.cvtpack.f22307 = extractelement <4 x float> %math.result445, i64 2, !dbg !53
  %.cvtpack.f32308 = extractelement <4 x float> %math.result445, i64 3, !dbg !53
  %.cvtpack.p02309 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02305, float %.cvtpack.f12306) #12, !dbg !53
  %.cvtpack.p12310 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22307, float %.cvtpack.f32308) #12, !dbg !53
  %.cvtpack.v02311 = insertelement <2 x i32> poison, i32 %.cvtpack.p02309, i64 0, !dbg !53
  %.cvtpack.v12312 = insertelement <2 x i32> %.cvtpack.v02311, i32 %.cvtpack.p12310, i64 1, !dbg !53
  %888 = bitcast <2 x i32> %.cvtpack.v12312 to <4 x half>, !dbg !53
  %math.packed446 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %888) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %889 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %874, <4 x float> %855) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %890 = shufflevector <8 x half> %754, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %891 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %890, <4 x float> %857) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %892 = fsub <4 x float> %647, %728, !dbg !45
  %893 = extractelement <4 x float> %892, i64 0, !dbg !52
  %894 = tail call float @llvm.amdgcn.exp2.f32(float %893), !dbg !52
  %local.reduce133 = fadd float %894, %local.reduce129, !dbg !54, !frisk.local_reduction !5
  %895 = insertelement <4 x float> poison, float %894, i64 0, !dbg !53
  %896 = extractelement <4 x float> %892, i64 1, !dbg !52
  %897 = tail call float @llvm.amdgcn.exp2.f32(float %896), !dbg !52
  %local.reduce134 = fadd float %897, %local.reduce130, !dbg !54, !frisk.local_reduction !5
  %898 = insertelement <4 x float> %895, float %897, i64 1, !dbg !53
  %899 = extractelement <4 x float> %892, i64 2, !dbg !52
  %900 = tail call float @llvm.amdgcn.exp2.f32(float %899), !dbg !52
  %local.reduce135 = fadd float %900, %local.reduce131, !dbg !54, !frisk.local_reduction !5
  %901 = insertelement <4 x float> %898, float %900, i64 2, !dbg !53
  %902 = extractelement <4 x float> %892, i64 3, !dbg !52
  %903 = tail call float @llvm.amdgcn.exp2.f32(float %902), !dbg !52
  %local.reduce136 = fadd float %903, %local.reduce132, !dbg !54, !frisk.local_reduction !5
  %904 = insertelement <4 x float> %901, float %903, i64 3, !dbg !53
  %math.result461 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %904) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02313 = extractelement <4 x float> %math.result461, i64 0, !dbg !53
  %.cvtpack.f12314 = extractelement <4 x float> %math.result461, i64 1, !dbg !53
  %.cvtpack.f22315 = extractelement <4 x float> %math.result461, i64 2, !dbg !53
  %.cvtpack.f32316 = extractelement <4 x float> %math.result461, i64 3, !dbg !53
  %.cvtpack.p02317 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02313, float %.cvtpack.f12314) #12, !dbg !53
  %.cvtpack.p12318 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22315, float %.cvtpack.f32316) #12, !dbg !53
  %.cvtpack.v02319 = insertelement <2 x i32> poison, i32 %.cvtpack.p02317, i64 0, !dbg !53
  %.cvtpack.v12320 = insertelement <2 x i32> %.cvtpack.v02319, i32 %.cvtpack.p12318, i64 1, !dbg !53
  %905 = bitcast <2 x i32> %.cvtpack.v12320 to <4 x half>, !dbg !53
  %math.packed462 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %905) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %874, <4 x float> %872) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %907 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %890, <4 x float> %873) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %908 = shufflevector <8 x half> %756, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %909 = fsub <4 x float> %608, %727, !dbg !45
  %910 = extractelement <4 x float> %909, i64 0, !dbg !52
  %911 = tail call float @llvm.amdgcn.exp2.f32(float %910), !dbg !52
  %local.reduce168 = fadd float %911, %local.reduce164, !dbg !54, !frisk.local_reduction !5
  %912 = insertelement <4 x float> poison, float %911, i64 0, !dbg !53
  %913 = extractelement <4 x float> %909, i64 1, !dbg !52
  %914 = tail call float @llvm.amdgcn.exp2.f32(float %913), !dbg !52
  %local.reduce169 = fadd float %914, %local.reduce165, !dbg !54, !frisk.local_reduction !5
  %915 = insertelement <4 x float> %912, float %914, i64 1, !dbg !53
  %916 = extractelement <4 x float> %909, i64 2, !dbg !52
  %917 = tail call float @llvm.amdgcn.exp2.f32(float %916), !dbg !52
  %local.reduce170 = fadd float %917, %local.reduce166, !dbg !54, !frisk.local_reduction !5
  %918 = insertelement <4 x float> %915, float %917, i64 2, !dbg !53
  %919 = extractelement <4 x float> %909, i64 3, !dbg !52
  %920 = tail call float @llvm.amdgcn.exp2.f32(float %919), !dbg !52
  %local.reduce171 = fadd float %920, %local.reduce167, !dbg !54, !frisk.local_reduction !5
  %921 = insertelement <4 x float> %918, float %920, i64 3, !dbg !53
  %math.result447 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %921) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02321 = extractelement <4 x float> %math.result447, i64 0, !dbg !53
  %.cvtpack.f12322 = extractelement <4 x float> %math.result447, i64 1, !dbg !53
  %.cvtpack.f22323 = extractelement <4 x float> %math.result447, i64 2, !dbg !53
  %.cvtpack.f32324 = extractelement <4 x float> %math.result447, i64 3, !dbg !53
  %.cvtpack.p02325 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02321, float %.cvtpack.f12322) #12, !dbg !53
  %.cvtpack.p12326 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22323, float %.cvtpack.f32324) #12, !dbg !53
  %.cvtpack.v02327 = insertelement <2 x i32> poison, i32 %.cvtpack.p02325, i64 0, !dbg !53
  %.cvtpack.v12328 = insertelement <2 x i32> %.cvtpack.v02327, i32 %.cvtpack.p12326, i64 1, !dbg !53
  %922 = bitcast <2 x i32> %.cvtpack.v12328 to <4 x half>, !dbg !53
  %math.packed448 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %922) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %923 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %908, <4 x float> %889) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %924 = shufflevector <8 x half> %756, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %925 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %924, <4 x float> %891) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %926 = fsub <4 x float> %650, %728, !dbg !45
  %927 = extractelement <4 x float> %926, i64 0, !dbg !52
  %928 = tail call float @llvm.amdgcn.exp2.f32(float %927), !dbg !52
  %local.reduce137 = fadd float %928, %local.reduce133, !dbg !54, !frisk.local_reduction !5
  %929 = insertelement <4 x float> poison, float %928, i64 0, !dbg !53
  %930 = extractelement <4 x float> %926, i64 1, !dbg !52
  %931 = tail call float @llvm.amdgcn.exp2.f32(float %930), !dbg !52
  %local.reduce138 = fadd float %931, %local.reduce134, !dbg !54, !frisk.local_reduction !5
  %932 = insertelement <4 x float> %929, float %931, i64 1, !dbg !53
  %933 = extractelement <4 x float> %926, i64 2, !dbg !52
  %934 = tail call float @llvm.amdgcn.exp2.f32(float %933), !dbg !52
  %local.reduce139 = fadd float %934, %local.reduce135, !dbg !54, !frisk.local_reduction !5
  %935 = insertelement <4 x float> %932, float %934, i64 2, !dbg !53
  %936 = extractelement <4 x float> %926, i64 3, !dbg !52
  %937 = tail call float @llvm.amdgcn.exp2.f32(float %936), !dbg !52
  %local.reduce140 = fadd float %937, %local.reduce136, !dbg !54, !frisk.local_reduction !5
  %938 = insertelement <4 x float> %935, float %937, i64 3, !dbg !53
  %math.result463 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %938) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02329 = extractelement <4 x float> %math.result463, i64 0, !dbg !53
  %.cvtpack.f12330 = extractelement <4 x float> %math.result463, i64 1, !dbg !53
  %.cvtpack.f22331 = extractelement <4 x float> %math.result463, i64 2, !dbg !53
  %.cvtpack.f32332 = extractelement <4 x float> %math.result463, i64 3, !dbg !53
  %.cvtpack.p02333 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02329, float %.cvtpack.f12330) #12, !dbg !53
  %.cvtpack.p12334 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22331, float %.cvtpack.f32332) #12, !dbg !53
  %.cvtpack.v02335 = insertelement <2 x i32> poison, i32 %.cvtpack.p02333, i64 0, !dbg !53
  %.cvtpack.v12336 = insertelement <2 x i32> %.cvtpack.v02335, i32 %.cvtpack.p12334, i64 1, !dbg !53
  %939 = bitcast <2 x i32> %.cvtpack.v12336 to <4 x half>, !dbg !53
  %math.packed464 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %939) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %940 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %908, <4 x float> %906) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %941 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %924, <4 x float> %907) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %942 = shufflevector <8 x half> %758, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %943 = fsub <4 x float> %616, %727, !dbg !45
  %944 = extractelement <4 x float> %943, i64 0, !dbg !52
  %945 = tail call float @llvm.amdgcn.exp2.f32(float %944), !dbg !52
  %local.reduce172 = fadd float %945, %local.reduce168, !dbg !54, !frisk.local_reduction !5
  %946 = insertelement <4 x float> poison, float %945, i64 0, !dbg !53
  %947 = extractelement <4 x float> %943, i64 1, !dbg !52
  %948 = tail call float @llvm.amdgcn.exp2.f32(float %947), !dbg !52
  %local.reduce173 = fadd float %948, %local.reduce169, !dbg !54, !frisk.local_reduction !5
  %949 = insertelement <4 x float> %946, float %948, i64 1, !dbg !53
  %950 = extractelement <4 x float> %943, i64 2, !dbg !52
  %951 = tail call float @llvm.amdgcn.exp2.f32(float %950), !dbg !52
  %local.reduce174 = fadd float %951, %local.reduce170, !dbg !54, !frisk.local_reduction !5
  %952 = insertelement <4 x float> %949, float %951, i64 2, !dbg !53
  %953 = extractelement <4 x float> %943, i64 3, !dbg !52
  %954 = tail call float @llvm.amdgcn.exp2.f32(float %953), !dbg !52
  %local.reduce175 = fadd float %954, %local.reduce171, !dbg !54, !frisk.local_reduction !5
  %955 = insertelement <4 x float> %952, float %954, i64 3, !dbg !53
  %math.result449 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %955) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02337 = extractelement <4 x float> %math.result449, i64 0, !dbg !53
  %.cvtpack.f12338 = extractelement <4 x float> %math.result449, i64 1, !dbg !53
  %.cvtpack.f22339 = extractelement <4 x float> %math.result449, i64 2, !dbg !53
  %.cvtpack.f32340 = extractelement <4 x float> %math.result449, i64 3, !dbg !53
  %.cvtpack.p02341 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02337, float %.cvtpack.f12338) #12, !dbg !53
  %.cvtpack.p12342 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22339, float %.cvtpack.f32340) #12, !dbg !53
  %.cvtpack.v02343 = insertelement <2 x i32> poison, i32 %.cvtpack.p02341, i64 0, !dbg !53
  %.cvtpack.v12344 = insertelement <2 x i32> %.cvtpack.v02343, i32 %.cvtpack.p12342, i64 1, !dbg !53
  %956 = bitcast <2 x i32> %.cvtpack.v12344 to <4 x half>, !dbg !53
  %math.packed450 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %956) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %942, <4 x float> %923) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %958 = shufflevector <8 x half> %758, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %958, <4 x float> %925) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %960 = fsub <4 x float> %653, %728, !dbg !45
  %961 = extractelement <4 x float> %960, i64 0, !dbg !52
  %962 = tail call float @llvm.amdgcn.exp2.f32(float %961), !dbg !52
  %local.reduce141 = fadd float %962, %local.reduce137, !dbg !54, !frisk.local_reduction !5
  %963 = insertelement <4 x float> poison, float %962, i64 0, !dbg !53
  %964 = extractelement <4 x float> %960, i64 1, !dbg !52
  %965 = tail call float @llvm.amdgcn.exp2.f32(float %964), !dbg !52
  %local.reduce142 = fadd float %965, %local.reduce138, !dbg !54, !frisk.local_reduction !5
  %966 = insertelement <4 x float> %963, float %965, i64 1, !dbg !53
  %967 = extractelement <4 x float> %960, i64 2, !dbg !52
  %968 = tail call float @llvm.amdgcn.exp2.f32(float %967), !dbg !52
  %local.reduce143 = fadd float %968, %local.reduce139, !dbg !54, !frisk.local_reduction !5
  %969 = insertelement <4 x float> %966, float %968, i64 2, !dbg !53
  %970 = extractelement <4 x float> %960, i64 3, !dbg !52
  %971 = tail call float @llvm.amdgcn.exp2.f32(float %970), !dbg !52
  %local.reduce144 = fadd float %971, %local.reduce140, !dbg !54, !frisk.local_reduction !5
  %972 = insertelement <4 x float> %969, float %971, i64 3, !dbg !53
  %math.result465 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %972) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02345 = extractelement <4 x float> %math.result465, i64 0, !dbg !53
  %.cvtpack.f12346 = extractelement <4 x float> %math.result465, i64 1, !dbg !53
  %.cvtpack.f22347 = extractelement <4 x float> %math.result465, i64 2, !dbg !53
  %.cvtpack.f32348 = extractelement <4 x float> %math.result465, i64 3, !dbg !53
  %.cvtpack.p02349 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02345, float %.cvtpack.f12346) #12, !dbg !53
  %.cvtpack.p12350 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22347, float %.cvtpack.f32348) #12, !dbg !53
  %.cvtpack.v02351 = insertelement <2 x i32> poison, i32 %.cvtpack.p02349, i64 0, !dbg !53
  %.cvtpack.v12352 = insertelement <2 x i32> %.cvtpack.v02351, i32 %.cvtpack.p12350, i64 1, !dbg !53
  %973 = bitcast <2 x i32> %.cvtpack.v12352 to <4 x half>, !dbg !53
  %math.packed466 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %973) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %974 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %942, <4 x float> %940) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %958, <4 x float> %941) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %976 = shufflevector <8 x half> %760, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %977 = fsub <4 x float> %623, %727, !dbg !45
  %978 = extractelement <4 x float> %977, i64 0, !dbg !52
  %979 = tail call float @llvm.amdgcn.exp2.f32(float %978), !dbg !52
  %local.reduce176 = fadd float %979, %local.reduce172, !dbg !54, !frisk.local_reduction !5
  %980 = insertelement <4 x float> poison, float %979, i64 0, !dbg !53
  %981 = extractelement <4 x float> %977, i64 1, !dbg !52
  %982 = tail call float @llvm.amdgcn.exp2.f32(float %981), !dbg !52
  %local.reduce177 = fadd float %982, %local.reduce173, !dbg !54, !frisk.local_reduction !5
  %983 = insertelement <4 x float> %980, float %982, i64 1, !dbg !53
  %984 = extractelement <4 x float> %977, i64 2, !dbg !52
  %985 = tail call float @llvm.amdgcn.exp2.f32(float %984), !dbg !52
  %local.reduce178 = fadd float %985, %local.reduce174, !dbg !54, !frisk.local_reduction !5
  %986 = insertelement <4 x float> %983, float %985, i64 2, !dbg !53
  %987 = extractelement <4 x float> %977, i64 3, !dbg !52
  %988 = tail call float @llvm.amdgcn.exp2.f32(float %987), !dbg !52
  %local.reduce179 = fadd float %988, %local.reduce175, !dbg !54, !frisk.local_reduction !5
  %989 = insertelement <4 x float> %986, float %988, i64 3, !dbg !53
  %math.result451 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %989) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02353 = extractelement <4 x float> %math.result451, i64 0, !dbg !53
  %.cvtpack.f12354 = extractelement <4 x float> %math.result451, i64 1, !dbg !53
  %.cvtpack.f22355 = extractelement <4 x float> %math.result451, i64 2, !dbg !53
  %.cvtpack.f32356 = extractelement <4 x float> %math.result451, i64 3, !dbg !53
  %.cvtpack.p02357 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02353, float %.cvtpack.f12354) #12, !dbg !53
  %.cvtpack.p12358 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22355, float %.cvtpack.f32356) #12, !dbg !53
  %.cvtpack.v02359 = insertelement <2 x i32> poison, i32 %.cvtpack.p02357, i64 0, !dbg !53
  %.cvtpack.v12360 = insertelement <2 x i32> %.cvtpack.v02359, i32 %.cvtpack.p12358, i64 1, !dbg !53
  %990 = bitcast <2 x i32> %.cvtpack.v12360 to <4 x half>, !dbg !53
  %math.packed452 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %990) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %976, <4 x float> %957) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %992 = shufflevector <8 x half> %760, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %992, <4 x float> %959) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %994 = fsub <4 x float> %656, %728, !dbg !45
  %995 = extractelement <4 x float> %994, i64 0, !dbg !52
  %996 = tail call float @llvm.amdgcn.exp2.f32(float %995), !dbg !52
  %local.reduce145 = fadd float %996, %local.reduce141, !dbg !54, !frisk.local_reduction !5
  %997 = insertelement <4 x float> poison, float %996, i64 0, !dbg !53
  %998 = extractelement <4 x float> %994, i64 1, !dbg !52
  %999 = tail call float @llvm.amdgcn.exp2.f32(float %998), !dbg !52
  %local.reduce146 = fadd float %999, %local.reduce142, !dbg !54, !frisk.local_reduction !5
  %1000 = insertelement <4 x float> %997, float %999, i64 1, !dbg !53
  %1001 = extractelement <4 x float> %994, i64 2, !dbg !52
  %1002 = tail call float @llvm.amdgcn.exp2.f32(float %1001), !dbg !52
  %local.reduce147 = fadd float %1002, %local.reduce143, !dbg !54, !frisk.local_reduction !5
  %1003 = insertelement <4 x float> %1000, float %1002, i64 2, !dbg !53
  %1004 = extractelement <4 x float> %994, i64 3, !dbg !52
  %1005 = tail call float @llvm.amdgcn.exp2.f32(float %1004), !dbg !52
  %local.reduce148 = fadd float %1005, %local.reduce144, !dbg !54, !frisk.local_reduction !5
  %1006 = insertelement <4 x float> %1003, float %1005, i64 3, !dbg !53
  %math.result467 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1006) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02361 = extractelement <4 x float> %math.result467, i64 0, !dbg !53
  %.cvtpack.f12362 = extractelement <4 x float> %math.result467, i64 1, !dbg !53
  %.cvtpack.f22363 = extractelement <4 x float> %math.result467, i64 2, !dbg !53
  %.cvtpack.f32364 = extractelement <4 x float> %math.result467, i64 3, !dbg !53
  %.cvtpack.p02365 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02361, float %.cvtpack.f12362) #12, !dbg !53
  %.cvtpack.p12366 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22363, float %.cvtpack.f32364) #12, !dbg !53
  %.cvtpack.v02367 = insertelement <2 x i32> poison, i32 %.cvtpack.p02365, i64 0, !dbg !53
  %.cvtpack.v12368 = insertelement <2 x i32> %.cvtpack.v02367, i32 %.cvtpack.p12366, i64 1, !dbg !53
  %1007 = bitcast <2 x i32> %.cvtpack.v12368 to <4 x half>, !dbg !53
  %math.packed468 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1007) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %976, <4 x float> %974) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1009 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %992, <4 x float> %975) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1010 = bitcast <4 x i32> %761 to <8 x half>, !dbg !50
  %1011 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1012 = fsub <4 x float> %630, %727, !dbg !45
  %1013 = extractelement <4 x float> %1012, i64 0, !dbg !52
  %1014 = tail call float @llvm.amdgcn.exp2.f32(float %1013), !dbg !52
  %local.reduce180 = fadd float %1014, %local.reduce176, !dbg !54, !frisk.local_reduction !5
  %1015 = insertelement <4 x float> poison, float %1014, i64 0, !dbg !53
  %1016 = extractelement <4 x float> %1012, i64 1, !dbg !52
  %1017 = tail call float @llvm.amdgcn.exp2.f32(float %1016), !dbg !52
  %local.reduce181 = fadd float %1017, %local.reduce177, !dbg !54, !frisk.local_reduction !5
  %local.reduce184 = fadd float %local.reduce180, %local.reduce181, !dbg !54, !frisk.local_reduction !5
  %1018 = insertelement <4 x float> %1015, float %1017, i64 1, !dbg !53
  %1019 = extractelement <4 x float> %1012, i64 2, !dbg !52
  %1020 = tail call float @llvm.amdgcn.exp2.f32(float %1019), !dbg !52
  %local.reduce182 = fadd float %1020, %local.reduce178, !dbg !54, !frisk.local_reduction !5
  %1021 = insertelement <4 x float> %1018, float %1020, i64 2, !dbg !53
  %1022 = extractelement <4 x float> %1012, i64 3, !dbg !52
  %1023 = tail call float @llvm.amdgcn.exp2.f32(float %1022), !dbg !52
  %local.reduce183 = fadd float %1023, %local.reduce179, !dbg !54, !frisk.local_reduction !5
  %local.reduce185 = fadd float %local.reduce182, %local.reduce183, !dbg !54, !frisk.local_reduction !5
  %1024 = fadd float %local.reduce184, %local.reduce185, !dbg !54, !frisk.local_reduction !5
  %1025 = fadd float %736, %1024, !dbg !54
  %1026 = insertelement <1 x float> poison, float %1025, i64 0, !dbg !11
  %1027 = insertvalue [2 x <1 x float>] poison, <1 x float> %1026, 0, !dbg !11
  %1028 = insertelement <4 x float> %1021, float %1023, i64 3, !dbg !53
  %math.result453 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1028) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02369 = extractelement <4 x float> %math.result453, i64 0, !dbg !53
  %.cvtpack.f12370 = extractelement <4 x float> %math.result453, i64 1, !dbg !53
  %.cvtpack.f22371 = extractelement <4 x float> %math.result453, i64 2, !dbg !53
  %.cvtpack.f32372 = extractelement <4 x float> %math.result453, i64 3, !dbg !53
  %.cvtpack.p02373 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02369, float %.cvtpack.f12370) #12, !dbg !53
  %.cvtpack.p12374 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22371, float %.cvtpack.f32372) #12, !dbg !53
  %.cvtpack.v02375 = insertelement <2 x i32> poison, i32 %.cvtpack.p02373, i64 0, !dbg !53
  %.cvtpack.v12376 = insertelement <2 x i32> %.cvtpack.v02375, i32 %.cvtpack.p12374, i64 1, !dbg !53
  %1029 = bitcast <2 x i32> %.cvtpack.v12376 to <4 x half>, !dbg !53
  %math.packed454 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1029) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1011, <4 x float> %991) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1031 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1032 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1031, <4 x float> %993) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1033 = fsub <4 x float> %670, %728, !dbg !45
  %1034 = extractelement <4 x float> %1033, i64 0, !dbg !52
  %1035 = tail call float @llvm.amdgcn.exp2.f32(float %1034), !dbg !52
  %local.reduce149 = fadd float %1035, %local.reduce145, !dbg !54, !frisk.local_reduction !5
  %1036 = insertelement <4 x float> poison, float %1035, i64 0, !dbg !53
  %1037 = extractelement <4 x float> %1033, i64 1, !dbg !52
  %1038 = tail call float @llvm.amdgcn.exp2.f32(float %1037), !dbg !52
  %local.reduce150 = fadd float %1038, %local.reduce146, !dbg !54, !frisk.local_reduction !5
  %local.reduce153 = fadd float %local.reduce149, %local.reduce150, !dbg !54, !frisk.local_reduction !5
  %1039 = insertelement <4 x float> %1036, float %1038, i64 1, !dbg !53
  %1040 = extractelement <4 x float> %1033, i64 2, !dbg !52
  %1041 = tail call float @llvm.amdgcn.exp2.f32(float %1040), !dbg !52
  %local.reduce151 = fadd float %1041, %local.reduce147, !dbg !54, !frisk.local_reduction !5
  %1042 = insertelement <4 x float> %1039, float %1041, i64 2, !dbg !53
  %1043 = extractelement <4 x float> %1033, i64 3, !dbg !52
  %1044 = tail call float @llvm.amdgcn.exp2.f32(float %1043), !dbg !52
  %local.reduce152 = fadd float %1044, %local.reduce148, !dbg !54, !frisk.local_reduction !5
  %local.reduce154 = fadd float %local.reduce151, %local.reduce152, !dbg !54, !frisk.local_reduction !5
  %1045 = fadd float %local.reduce153, %local.reduce154, !dbg !54, !frisk.local_reduction !5
  %1046 = fadd float %744, %1045, !dbg !54
  %1047 = insertelement <1 x float> poison, float %1046, i64 0, !dbg !11
  %1048 = insertvalue [2 x <1 x float>] %1027, <1 x float> %1047, 1, !dbg !11
  %1049 = insertelement <4 x float> %1042, float %1044, i64 3, !dbg !53
  %math.result469 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1049) #8, !dbg !53, !frisk.register_lifetimes !5
  %.cvtpack.f02377 = extractelement <4 x float> %math.result469, i64 0, !dbg !53
  %.cvtpack.f12378 = extractelement <4 x float> %math.result469, i64 1, !dbg !53
  %.cvtpack.f22379 = extractelement <4 x float> %math.result469, i64 2, !dbg !53
  %.cvtpack.f32380 = extractelement <4 x float> %math.result469, i64 3, !dbg !53
  %.cvtpack.p02381 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02377, float %.cvtpack.f12378) #12, !dbg !53
  %.cvtpack.p12382 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22379, float %.cvtpack.f32380) #12, !dbg !53
  %.cvtpack.v02383 = insertelement <2 x i32> poison, i32 %.cvtpack.p02381, i64 0, !dbg !53
  %.cvtpack.v12384 = insertelement <2 x i32> %.cvtpack.v02383, i32 %.cvtpack.p12382, i64 1, !dbg !53
  %1050 = bitcast <2 x i32> %.cvtpack.v12384 to <4 x half>, !dbg !53
  %math.packed470 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1050) #8, !dbg !54, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50, !frisk.softmax_fragments !5
  %1051 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1011, <4 x float> %1008) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1052 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1031, <4 x float> %1009) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  tail call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %vdb.address1814 = or disjoint i64 %.idx17641766, 64, !dbg !41
  %vdb.address1815 = or disjoint i64 %address.local988, %vdb.address1814, !dbg !41
  %vdb.address1816 = trunc i64 %vdb.address1815 to i32, !dbg !41
  %vdb.address1817 = shl i32 %vdb.address1816, 1, !dbg !41
  %vdb.address1824 = or disjoint i64 %address.local994, %vdb.address1814, !dbg !41
  %vdb.address1825 = trunc i64 %vdb.address1824 to i32, !dbg !41
  %vdb.address1826 = shl i32 %vdb.address1825, 1, !dbg !41
  %1053 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %183, i32 %vdb.address1817, <4 x i32> %dma.buffer.rsrc1670) #9, !dbg !41
  %1054 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %187, i32 %vdb.address1826, <4 x i32> %dma.buffer.rsrc1670) #9, !dbg !41
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local1002, i64 8192, !dbg !50
  %1055 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1056 = bitcast <4 x i32> %1055 to <8 x half>, !dbg !50
  %vdb.read1806 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8448, !dbg !50
  %1057 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1806, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1058 = bitcast <4 x i32> %1057 to <8 x half>, !dbg !50
  %vdb.read1807 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8704, !dbg !50
  %1059 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1807, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1060 = bitcast <4 x i32> %1059 to <8 x half>, !dbg !50
  %vdb.read1808 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8960, !dbg !50
  %1061 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1808, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1062 = bitcast <4 x i32> %1061 to <8 x half>, !dbg !50
  %vdb.read1809 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 12288, !dbg !50
  %1063 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1809, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1064 = bitcast <4 x i32> %1063 to <8 x half>, !dbg !50
  %vdb.read1810 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12544, !dbg !50
  %1065 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1810, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1066 = bitcast <4 x i32> %1065 to <8 x half>, !dbg !50
  %vdb.read1811 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12800, !dbg !50
  %1067 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1811, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1068 = bitcast <4 x i32> %1067 to <8 x half>, !dbg !50
  %vdb.read1812 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 13056, !dbg !50
  %1069 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1812, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1070 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  %1071 = extractvalue [2 x <8 x float>] %145, 0, !dbg !48
  %1072 = shufflevector <8 x float> %1071, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %1073 = fmul <4 x float> %1072, %733, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1074 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1070, <4 x float> %1073) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1075 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  %1076 = shufflevector <8 x float> %1071, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %1077 = fmul <4 x float> %1076, %733, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1075, <4 x float> %1077) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1079 = extractvalue [2 x <8 x float>] %145, 1, !dbg !48
  %1080 = shufflevector <8 x float> %1079, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %1081 = fmul <4 x float> %1080, %741, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1070, <4 x float> %1081) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1083 = shufflevector <8 x float> %1079, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %1084 = fmul <4 x float> %1083, %741, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1085 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1075, <4 x float> %1084) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1086 = shufflevector <8 x half> %1058, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1087 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1086, <4 x float> %1074) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1088 = shufflevector <8 x half> %1058, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1089 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1088, <4 x float> %1078) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1086, <4 x float> %1082) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1091 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1088, <4 x float> %1085) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1092 = shufflevector <8 x half> %1060, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1093 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1092, <4 x float> %1087) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1094 = shufflevector <8 x half> %1060, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1095 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1094, <4 x float> %1089) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1092, <4 x float> %1090) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1097 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1094, <4 x float> %1091) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1098 = shufflevector <8 x half> %1062, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1099 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1098, <4 x float> %1093) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1100 = shufflevector <8 x half> %1062, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1101 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1100, <4 x float> %1095) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1098, <4 x float> %1096) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1103 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1100, <4 x float> %1097) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1104 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1105 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1104, <4 x float> %1099) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1106 = shufflevector <8 x half> %1064, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1107 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1106, <4 x float> %1101) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1108 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1104, <4 x float> %1102) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1109 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1106, <4 x float> %1103) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1110 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1111 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1110, <4 x float> %1105) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1112 = shufflevector <8 x half> %1066, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1112, <4 x float> %1107) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1114 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1110, <4 x float> %1108) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1115 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1112, <4 x float> %1109) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1116 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1116, <4 x float> %1111) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1118 = shufflevector <8 x half> %1068, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1119 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1118, <4 x float> %1113) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1116, <4 x float> %1114) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1121 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1118, <4 x float> %1115) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1122 = bitcast <4 x i32> %1069 to <8 x half>, !dbg !50
  %1123 = shufflevector <8 x half> %1122, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1124 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1123, <4 x float> %1117) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1125 = shufflevector <8 x half> %1122, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1126 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1125, <4 x float> %1119) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1127 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1123, <4 x float> %1120) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1128 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1125, <4 x float> %1121) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  tail call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %vdb.address1832 = or disjoint i64 %.idx17641766, 96, !dbg !41
  %vdb.address1833 = or disjoint i64 %address.local988, %vdb.address1832, !dbg !41
  %vdb.address1834 = trunc i64 %vdb.address1833 to i32, !dbg !41
  %vdb.address1835 = shl i32 %vdb.address1834, 1, !dbg !41
  %vdb.address1842 = or disjoint i64 %address.local994, %vdb.address1832, !dbg !41
  %vdb.address1843 = trunc i64 %vdb.address1842 to i32, !dbg !41
  %vdb.address1844 = shl i32 %vdb.address1843, 1, !dbg !41
  %1129 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1835, <4 x i32> %dma.buffer.rsrc1670) #9, !dbg !41
  %1130 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1805, i32 %vdb.address1844, <4 x i32> %dma.buffer.rsrc1670) #9, !dbg !41
  %1131 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1132 = bitcast <4 x i32> %1131 to <8 x half>, !dbg !50
  %1133 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1134 = bitcast <4 x i32> %1133 to <8 x half>, !dbg !50
  %1135 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1136 = bitcast <4 x i32> %1135 to <8 x half>, !dbg !50
  %1137 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1138 = bitcast <4 x i32> %1137 to <8 x half>, !dbg !50
  %1139 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1140 = bitcast <4 x i32> %1139 to <8 x half>, !dbg !50
  %1141 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1142 = bitcast <4 x i32> %1141 to <8 x half>, !dbg !50
  %1143 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1144 = bitcast <4 x i32> %1143 to <8 x half>, !dbg !50
  %1145 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1146 = shufflevector <8 x half> %1132, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  %1147 = extractvalue [2 x <8 x float>] %144, 0, !dbg !48
  %1148 = shufflevector <8 x float> %1147, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %1149 = fmul <4 x float> %1148, %733, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1146, <4 x float> %1149) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1151 = shufflevector <8 x half> %1132, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  %1152 = shufflevector <8 x float> %1147, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %1153 = fmul <4 x float> %1152, %733, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1151, <4 x float> %1153) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1155 = extractvalue [2 x <8 x float>] %144, 1, !dbg !48
  %1156 = shufflevector <8 x float> %1155, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %1157 = fmul <4 x float> %1156, %741, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1146, <4 x float> %1157) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1159 = shufflevector <8 x float> %1155, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %1160 = fmul <4 x float> %1159, %741, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1151, <4 x float> %1160) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1162 = shufflevector <8 x half> %1134, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1162, <4 x float> %1150) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1164 = shufflevector <8 x half> %1134, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1164, <4 x float> %1154) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1162, <4 x float> %1158) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1164, <4 x float> %1161) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1168 = shufflevector <8 x half> %1136, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1168, <4 x float> %1163) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1170 = shufflevector <8 x half> %1136, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1170, <4 x float> %1165) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1168, <4 x float> %1166) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1170, <4 x float> %1167) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1174 = shufflevector <8 x half> %1138, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1174, <4 x float> %1169) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1176 = shufflevector <8 x half> %1138, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1176, <4 x float> %1171) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1174, <4 x float> %1172) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1176, <4 x float> %1173) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1180 = shufflevector <8 x half> %1140, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1180, <4 x float> %1175) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1182 = shufflevector <8 x half> %1140, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1182, <4 x float> %1177) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1180, <4 x float> %1178) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1182, <4 x float> %1179) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1186 = shufflevector <8 x half> %1142, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1186, <4 x float> %1181) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1188 = shufflevector <8 x half> %1142, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1188, <4 x float> %1183) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1186, <4 x float> %1184) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1188, <4 x float> %1185) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1192 = shufflevector <8 x half> %1144, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1192, <4 x float> %1187) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1194 = shufflevector <8 x half> %1144, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1194, <4 x float> %1189) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1192, <4 x float> %1190) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1194, <4 x float> %1191) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1198 = bitcast <4 x i32> %1145 to <8 x half>, !dbg !50
  %1199 = shufflevector <8 x half> %1198, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1199, <4 x float> %1193) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1201 = shufflevector <8 x half> %1198, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1201, <4 x float> %1195) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1199, <4 x float> %1196) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1201, <4 x float> %1197) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  tail call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %1205 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1206 = bitcast <4 x i32> %1205 to <8 x half>, !dbg !50
  %1207 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1806, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1208 = bitcast <4 x i32> %1207 to <8 x half>, !dbg !50
  %1209 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1807, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1210 = bitcast <4 x i32> %1209 to <8 x half>, !dbg !50
  %1211 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1808, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1212 = bitcast <4 x i32> %1211 to <8 x half>, !dbg !50
  %1213 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1809, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1214 = bitcast <4 x i32> %1213 to <8 x half>, !dbg !50
  %1215 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1810, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1216 = bitcast <4 x i32> %1215 to <8 x half>, !dbg !50
  %1217 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1811, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1218 = bitcast <4 x i32> %1217 to <8 x half>, !dbg !50
  %1219 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1812, i32 0) #11, !dbg !50, !frisk.v_double_buffer !51
  %1220 = shufflevector <8 x half> %1206, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  %1221 = extractvalue [2 x <8 x float>] %143, 0, !dbg !48
  %1222 = shufflevector <8 x float> %1221, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %1223 = fmul <4 x float> %1222, %733, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1220, <4 x float> %1223) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1225 = shufflevector <8 x half> %1206, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  %1226 = shufflevector <8 x float> %1221, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %1227 = fmul <4 x float> %1226, %733, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1225, <4 x float> %1227) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1229 = extractvalue [2 x <8 x float>] %143, 1, !dbg !48
  %1230 = shufflevector <8 x float> %1229, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %1231 = fmul <4 x float> %1230, %741, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1220, <4 x float> %1231) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1233 = shufflevector <8 x float> %1229, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %1234 = fmul <4 x float> %1233, %741, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1225, <4 x float> %1234) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1236 = shufflevector <8 x half> %1208, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1236, <4 x float> %1224) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1238 = shufflevector <8 x half> %1208, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1238, <4 x float> %1228) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1236, <4 x float> %1232) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1238, <4 x float> %1235) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1242 = shufflevector <8 x half> %1210, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1242, <4 x float> %1237) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1244 = shufflevector <8 x half> %1210, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1244, <4 x float> %1239) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1242, <4 x float> %1240) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1244, <4 x float> %1241) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1248 = shufflevector <8 x half> %1212, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1248, <4 x float> %1243) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1250 = shufflevector <8 x half> %1212, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1250, <4 x float> %1245) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1248, <4 x float> %1246) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1250, <4 x float> %1247) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1254 = shufflevector <8 x half> %1214, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1254, <4 x float> %1249) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1256 = shufflevector <8 x half> %1214, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1256, <4 x float> %1251) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1254, <4 x float> %1252) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1256, <4 x float> %1253) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1260 = shufflevector <8 x half> %1216, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1260, <4 x float> %1255) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1262 = shufflevector <8 x half> %1216, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1262, <4 x float> %1257) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1260, <4 x float> %1258) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1262, <4 x float> %1259) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1266 = shufflevector <8 x half> %1218, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1266, <4 x float> %1261) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1268 = shufflevector <8 x half> %1218, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1268, <4 x float> %1263) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1266, <4 x float> %1264) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1268, <4 x float> %1265) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1272 = bitcast <4 x i32> %1219 to <8 x half>, !dbg !50
  %1273 = shufflevector <8 x half> %1272, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1273, <4 x float> %1267) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1275 = shufflevector <8 x half> %1272, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1275, <4 x float> %1269) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1273, <4 x float> %1270) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !50
  %1278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1275, <4 x float> %1271) #8, !dbg !50, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !11
  %1279 = shufflevector <4 x float> %1030, <4 x float> %1032, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1280 = insertvalue [2 x <8 x float>] poison, <8 x float> %1279, 0, !dbg !11
  %1281 = shufflevector <4 x float> %1051, <4 x float> %1052, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1282 = insertvalue [2 x <8 x float>] %1280, <8 x float> %1281, 1, !dbg !11
  %1283 = insertvalue [2 x <1 x float>] poison, <1 x float> %723, 0, !dbg !11
  %1284 = insertvalue [2 x <1 x float>] %1283, <1 x float> %726, 1, !dbg !11
  %1285 = shufflevector <4 x float> %1124, <4 x float> %1126, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1286 = insertvalue [2 x <8 x float>] poison, <8 x float> %1285, 0, !dbg !11
  %1287 = shufflevector <4 x float> %1127, <4 x float> %1128, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1288 = insertvalue [2 x <8 x float>] %1286, <8 x float> %1287, 1, !dbg !11
  %1289 = shufflevector <4 x float> %1200, <4 x float> %1202, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1290 = insertvalue [2 x <8 x float>] poison, <8 x float> %1289, 0, !dbg !11
  %1291 = shufflevector <4 x float> %1203, <4 x float> %1204, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1292 = insertvalue [2 x <8 x float>] %1290, <8 x float> %1291, 1, !dbg !11
  %1293 = shufflevector <4 x float> %1274, <4 x float> %1276, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1294 = insertvalue [2 x <8 x float>] poison, <8 x float> %1293, 0, !dbg !11
  %1295 = shufflevector <4 x float> %1277, <4 x float> %1278, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1296 = insertvalue [2 x <8 x float>] %1294, <8 x float> %1295, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1297 = add nuw nsw i64 %149, 1, !dbg !11
  %1298 = icmp slt i64 %1297, %105, !dbg !11
  br i1 %1298, label %142, label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %142, %.._crit_edge_crit_edge
  %.pre-phi29 = phi i32 [ %.pre28, %.._crit_edge_crit_edge ], [ %703, %142 ], !dbg !11
  %.pre-phi27 = phi i32 [ %.pre26, %.._crit_edge_crit_edge ], [ %695, %142 ], !dbg !11
  %.pre-phi25 = phi i32 [ %.pre24, %.._crit_edge_crit_edge ], [ %694, %142 ], !dbg !11
  %lane.sum.lane254.pre-phi = phi i32 [ %.pre22, %.._crit_edge_crit_edge ], [ %692, %142 ], !dbg !11
  %.lcssa7 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1282, %142 ], !dbg !11
  %.lcssa6 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1048, %142 ], !dbg !11
  %.lcssa5 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1288, %142 ], !dbg !11
  %.lcssa4 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1292, %142 ], !dbg !11
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1296, %142 ], !dbg !11
  %lane.sum.field251 = extractvalue [2 x <1 x float>] %.lcssa6, 1, !dbg !11
  %lane.sum.local252 = extractelement <1 x float> %lane.sum.field251, i64 0, !dbg !11
  %1299 = shl nuw nsw i32 %lane.sum.lane254.pre-phi, 2, !dbg !11
  %1300 = xor i32 %1299, 64, !dbg !11
  %1301 = bitcast float %lane.sum.local252 to i32, !dbg !11
  %lane.sum.shuffle255 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1301), !dbg !11
  %1302 = bitcast i32 %lane.sum.shuffle255 to float, !dbg !11
  %lane.sum.final256 = fadd float %lane.sum.local252, %1302, !dbg !11
  %1303 = xor i32 %1299, 128, !dbg !11
  %1304 = bitcast float %lane.sum.final256 to i32, !dbg !11
  %lane.sum.shuffle257 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1304), !dbg !11
  %1305 = bitcast i32 %lane.sum.shuffle257 to float, !dbg !11
  %lane.sum.final258 = fadd float %lane.sum.final256, %1305, !dbg !11
  %1306 = insertelement <1 x float> poison, float %lane.sum.final258, i64 0, !dbg !11
  %lane.sum.field = extractvalue [2 x <1 x float>] %.lcssa6, 0, !dbg !11
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !11
  %1307 = bitcast float %lane.sum.local to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1307), !dbg !11
  %1308 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %lane.sum.local, %1308, !dbg !11
  %1309 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle249 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1309), !dbg !11
  %1310 = bitcast i32 %lane.sum.shuffle249 to float, !dbg !11
  %lane.sum.final250 = fadd float %lane.sum.final, %1310, !dbg !11
  %1311 = insertelement <1 x float> poison, float %lane.sum.final250, i64 0, !dbg !11
  %1312 = and i32 %kdma.tid, 16, !dbg !11
  %.not = icmp eq i32 %1312, 0, !dbg !11
  %1313 = and i32 %kdma.tid, 32, !dbg !11
  %.not36 = icmp eq i32 %1313, 0, !dbg !11
  %1314 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !11
  %1315 = extractelement <8 x float> %1314, i64 0, !dbg !11
  %1316 = extractelement <8 x float> %1314, i64 1, !dbg !11
  %1317 = extractelement <8 x float> %1314, i64 2, !dbg !11
  %1318 = extractelement <8 x float> %1314, i64 3, !dbg !11
  %1319 = select i1 %.not, float %1316, float %1315, !dbg !11
  %1320 = icmp samesign ult i32 %.pre-phi25, %.pre-phi27, !dbg !11
  %1321 = select i1 %1320, i32 %.pre-phi25, i32 %lane.sum.lane254.pre-phi, !dbg !11
  %1322 = shl nuw nsw i32 %1321, 2, !dbg !11
  %1323 = bitcast float %1319 to i32, !dbg !11
  %1324 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1323), !dbg !11
  %1325 = bitcast i32 %1324 to float, !dbg !11
  %1326 = select i1 %.not, float %1315, float %1325, !dbg !11
  %1327 = select i1 %.not, float %1325, float %1316, !dbg !11
  %1328 = select i1 %.not, float %1318, float %1317, !dbg !11
  %1329 = bitcast float %1328 to i32, !dbg !11
  %1330 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1329), !dbg !11
  %1331 = bitcast i32 %1330 to float, !dbg !11
  %1332 = select i1 %.not, float %1317, float %1331, !dbg !11
  %1333 = select i1 %.not, float %1331, float %1318, !dbg !11
  %1334 = select i1 %.not36, float %1332, float %1326, !dbg !11
  %1335 = icmp samesign ult i32 %.pre-phi29, %.pre-phi27, !dbg !11
  %1336 = select i1 %1335, i32 %.pre-phi29, i32 %lane.sum.lane254.pre-phi, !dbg !11
  %1337 = shl nuw nsw i32 %1336, 2, !dbg !11
  %1338 = bitcast float %1334 to i32, !dbg !11
  %1339 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1338), !dbg !11
  %1340 = bitcast i32 %1339 to float, !dbg !11
  %1341 = select i1 %.not36, float %1333, float %1327, !dbg !11
  %1342 = bitcast float %1341 to i32, !dbg !11
  %1343 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1342), !dbg !11
  %1344 = bitcast i32 %1343 to float, !dbg !11
  %1345 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1346 = shufflevector <4 x i1> %1345, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1347 = insertelement <4 x float> poison, float %1326, i64 0, !dbg !11
  %1348 = insertelement <4 x float> %1347, float %1327, i64 1, !dbg !11
  %1349 = insertelement <4 x float> %1348, float %1340, i64 2, !dbg !11
  %1350 = insertelement <4 x float> %1349, float %1344, i64 3, !dbg !11
  %1351 = insertelement <4 x float> poison, float %1340, i64 0, !dbg !11
  %1352 = insertelement <4 x float> %1351, float %1344, i64 1, !dbg !11
  %1353 = insertelement <4 x float> %1352, float %1332, i64 2, !dbg !11
  %1354 = insertelement <4 x float> %1353, float %1333, i64 3, !dbg !11
  %1355 = select <4 x i1> %1346, <4 x float> %1350, <4 x float> %1354, !dbg !11
  %1356 = extractelement <8 x float> %1314, i64 4, !dbg !11
  %1357 = extractelement <8 x float> %1314, i64 5, !dbg !11
  %1358 = extractelement <8 x float> %1314, i64 6, !dbg !11
  %1359 = extractelement <8 x float> %1314, i64 7, !dbg !11
  %1360 = select i1 %.not, float %1357, float %1356, !dbg !11
  %1361 = bitcast float %1360 to i32, !dbg !11
  %1362 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1361), !dbg !11
  %1363 = bitcast i32 %1362 to float, !dbg !11
  %1364 = select i1 %.not, float %1356, float %1363, !dbg !11
  %1365 = select i1 %.not, float %1363, float %1357, !dbg !11
  %1366 = select i1 %.not, float %1359, float %1358, !dbg !11
  %1367 = bitcast float %1366 to i32, !dbg !11
  %1368 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1367), !dbg !11
  %1369 = bitcast i32 %1368 to float, !dbg !11
  %1370 = select i1 %.not, float %1358, float %1369, !dbg !11
  %1371 = select i1 %.not, float %1369, float %1359, !dbg !11
  %1372 = select i1 %.not36, float %1370, float %1364, !dbg !11
  %1373 = bitcast float %1372 to i32, !dbg !11
  %1374 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1373), !dbg !11
  %1375 = bitcast i32 %1374 to float, !dbg !11
  %1376 = select i1 %.not36, float %1371, float %1365, !dbg !11
  %1377 = bitcast float %1376 to i32, !dbg !11
  %1378 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1377), !dbg !11
  %1379 = bitcast i32 %1378 to float, !dbg !11
  %1380 = insertelement <4 x float> poison, float %1364, i64 0, !dbg !11
  %1381 = insertelement <4 x float> %1380, float %1365, i64 1, !dbg !11
  %1382 = insertelement <4 x float> %1381, float %1375, i64 2, !dbg !11
  %1383 = insertelement <4 x float> %1382, float %1379, i64 3, !dbg !11
  %1384 = insertelement <4 x float> poison, float %1375, i64 0, !dbg !11
  %1385 = insertelement <4 x float> %1384, float %1379, i64 1, !dbg !11
  %1386 = insertelement <4 x float> %1385, float %1370, i64 2, !dbg !11
  %1387 = insertelement <4 x float> %1386, float %1371, i64 3, !dbg !11
  %1388 = select <4 x i1> %1346, <4 x float> %1383, <4 x float> %1387, !dbg !11
  %1389 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !11
  %1390 = extractelement <8 x float> %1389, i64 0, !dbg !11
  %1391 = extractelement <8 x float> %1389, i64 1, !dbg !11
  %1392 = extractelement <8 x float> %1389, i64 2, !dbg !11
  %1393 = extractelement <8 x float> %1389, i64 3, !dbg !11
  %1394 = select i1 %.not, float %1391, float %1390, !dbg !11
  %1395 = bitcast float %1394 to i32, !dbg !11
  %1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1395), !dbg !11
  %1397 = bitcast i32 %1396 to float, !dbg !11
  %1398 = select i1 %.not, float %1390, float %1397, !dbg !11
  %1399 = select i1 %.not, float %1397, float %1391, !dbg !11
  %1400 = select i1 %.not, float %1393, float %1392, !dbg !11
  %1401 = bitcast float %1400 to i32, !dbg !11
  %1402 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1401), !dbg !11
  %1403 = bitcast i32 %1402 to float, !dbg !11
  %1404 = select i1 %.not, float %1392, float %1403, !dbg !11
  %1405 = select i1 %.not, float %1403, float %1393, !dbg !11
  %1406 = select i1 %.not36, float %1404, float %1398, !dbg !11
  %1407 = bitcast float %1406 to i32, !dbg !11
  %1408 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1407), !dbg !11
  %1409 = bitcast i32 %1408 to float, !dbg !11
  %1410 = select i1 %.not36, float %1405, float %1399, !dbg !11
  %1411 = bitcast float %1410 to i32, !dbg !11
  %1412 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1411), !dbg !11
  %1413 = bitcast i32 %1412 to float, !dbg !11
  %1414 = insertelement <4 x float> poison, float %1398, i64 0, !dbg !11
  %1415 = insertelement <4 x float> %1414, float %1399, i64 1, !dbg !11
  %1416 = insertelement <4 x float> %1415, float %1409, i64 2, !dbg !11
  %1417 = insertelement <4 x float> %1416, float %1413, i64 3, !dbg !11
  %1418 = insertelement <4 x float> poison, float %1409, i64 0, !dbg !11
  %1419 = insertelement <4 x float> %1418, float %1413, i64 1, !dbg !11
  %1420 = insertelement <4 x float> %1419, float %1404, i64 2, !dbg !11
  %1421 = insertelement <4 x float> %1420, float %1405, i64 3, !dbg !11
  %1422 = select <4 x i1> %1346, <4 x float> %1417, <4 x float> %1421, !dbg !11
  %1423 = extractelement <8 x float> %1389, i64 4, !dbg !11
  %1424 = extractelement <8 x float> %1389, i64 5, !dbg !11
  %1425 = extractelement <8 x float> %1389, i64 6, !dbg !11
  %1426 = extractelement <8 x float> %1389, i64 7, !dbg !11
  %1427 = select i1 %.not, float %1424, float %1423, !dbg !11
  %1428 = bitcast float %1427 to i32, !dbg !11
  %1429 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1428), !dbg !11
  %1430 = bitcast i32 %1429 to float, !dbg !11
  %1431 = select i1 %.not, float %1423, float %1430, !dbg !11
  %1432 = select i1 %.not, float %1430, float %1424, !dbg !11
  %1433 = select i1 %.not, float %1426, float %1425, !dbg !11
  %1434 = bitcast float %1433 to i32, !dbg !11
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1434), !dbg !11
  %1436 = bitcast i32 %1435 to float, !dbg !11
  %1437 = select i1 %.not, float %1425, float %1436, !dbg !11
  %1438 = select i1 %.not, float %1436, float %1426, !dbg !11
  %1439 = select i1 %.not36, float %1437, float %1431, !dbg !11
  %1440 = bitcast float %1439 to i32, !dbg !11
  %1441 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1440), !dbg !11
  %1442 = bitcast i32 %1441 to float, !dbg !11
  %1443 = select i1 %.not36, float %1438, float %1432, !dbg !11
  %1444 = bitcast float %1443 to i32, !dbg !11
  %1445 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1444), !dbg !11
  %1446 = bitcast i32 %1445 to float, !dbg !11
  %1447 = insertelement <4 x float> poison, float %1431, i64 0, !dbg !11
  %1448 = insertelement <4 x float> %1447, float %1432, i64 1, !dbg !11
  %1449 = insertelement <4 x float> %1448, float %1442, i64 2, !dbg !11
  %1450 = insertelement <4 x float> %1449, float %1446, i64 3, !dbg !11
  %1451 = insertelement <4 x float> poison, float %1442, i64 0, !dbg !11
  %1452 = insertelement <4 x float> %1451, float %1446, i64 1, !dbg !11
  %1453 = insertelement <4 x float> %1452, float %1437, i64 2, !dbg !11
  %1454 = insertelement <4 x float> %1453, float %1438, i64 3, !dbg !11
  %1455 = select <4 x i1> %1346, <4 x float> %1450, <4 x float> %1454, !dbg !11
  %1456 = extractvalue [2 x <8 x float>] %.lcssa4, 0, !dbg !11
  %1457 = extractelement <8 x float> %1456, i64 0, !dbg !11
  %1458 = extractelement <8 x float> %1456, i64 1, !dbg !11
  %1459 = extractelement <8 x float> %1456, i64 2, !dbg !11
  %1460 = extractelement <8 x float> %1456, i64 3, !dbg !11
  %1461 = select i1 %.not, float %1458, float %1457, !dbg !11
  %1462 = bitcast float %1461 to i32, !dbg !11
  %1463 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1462), !dbg !11
  %1464 = bitcast i32 %1463 to float, !dbg !11
  %1465 = select i1 %.not, float %1457, float %1464, !dbg !11
  %1466 = select i1 %.not, float %1464, float %1458, !dbg !11
  %1467 = select i1 %.not, float %1460, float %1459, !dbg !11
  %1468 = bitcast float %1467 to i32, !dbg !11
  %1469 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1468), !dbg !11
  %1470 = bitcast i32 %1469 to float, !dbg !11
  %1471 = select i1 %.not, float %1459, float %1470, !dbg !11
  %1472 = select i1 %.not, float %1470, float %1460, !dbg !11
  %1473 = select i1 %.not36, float %1471, float %1465, !dbg !11
  %1474 = bitcast float %1473 to i32, !dbg !11
  %1475 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1474), !dbg !11
  %1476 = bitcast i32 %1475 to float, !dbg !11
  %1477 = select i1 %.not36, float %1472, float %1466, !dbg !11
  %1478 = bitcast float %1477 to i32, !dbg !11
  %1479 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1478), !dbg !11
  %1480 = bitcast i32 %1479 to float, !dbg !11
  %1481 = insertelement <4 x float> poison, float %1465, i64 0, !dbg !11
  %1482 = insertelement <4 x float> %1481, float %1466, i64 1, !dbg !11
  %1483 = insertelement <4 x float> %1482, float %1476, i64 2, !dbg !11
  %1484 = insertelement <4 x float> %1483, float %1480, i64 3, !dbg !11
  %1485 = insertelement <4 x float> poison, float %1476, i64 0, !dbg !11
  %1486 = insertelement <4 x float> %1485, float %1480, i64 1, !dbg !11
  %1487 = insertelement <4 x float> %1486, float %1471, i64 2, !dbg !11
  %1488 = insertelement <4 x float> %1487, float %1472, i64 3, !dbg !11
  %1489 = select <4 x i1> %1346, <4 x float> %1484, <4 x float> %1488, !dbg !11
  %1490 = extractelement <8 x float> %1456, i64 4, !dbg !11
  %1491 = extractelement <8 x float> %1456, i64 5, !dbg !11
  %1492 = extractelement <8 x float> %1456, i64 6, !dbg !11
  %1493 = extractelement <8 x float> %1456, i64 7, !dbg !11
  %1494 = select i1 %.not, float %1491, float %1490, !dbg !11
  %1495 = bitcast float %1494 to i32, !dbg !11
  %1496 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1495), !dbg !11
  %1497 = bitcast i32 %1496 to float, !dbg !11
  %1498 = select i1 %.not, float %1490, float %1497, !dbg !11
  %1499 = select i1 %.not, float %1497, float %1491, !dbg !11
  %1500 = select i1 %.not, float %1493, float %1492, !dbg !11
  %1501 = bitcast float %1500 to i32, !dbg !11
  %1502 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1501), !dbg !11
  %1503 = bitcast i32 %1502 to float, !dbg !11
  %1504 = select i1 %.not, float %1492, float %1503, !dbg !11
  %1505 = select i1 %.not, float %1503, float %1493, !dbg !11
  %1506 = select i1 %.not36, float %1504, float %1498, !dbg !11
  %1507 = bitcast float %1506 to i32, !dbg !11
  %1508 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1507), !dbg !11
  %1509 = bitcast i32 %1508 to float, !dbg !11
  %1510 = select i1 %.not36, float %1505, float %1499, !dbg !11
  %1511 = bitcast float %1510 to i32, !dbg !11
  %1512 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1511), !dbg !11
  %1513 = bitcast i32 %1512 to float, !dbg !11
  %1514 = insertelement <4 x float> poison, float %1498, i64 0, !dbg !11
  %1515 = insertelement <4 x float> %1514, float %1499, i64 1, !dbg !11
  %1516 = insertelement <4 x float> %1515, float %1509, i64 2, !dbg !11
  %1517 = insertelement <4 x float> %1516, float %1513, i64 3, !dbg !11
  %1518 = insertelement <4 x float> poison, float %1509, i64 0, !dbg !11
  %1519 = insertelement <4 x float> %1518, float %1513, i64 1, !dbg !11
  %1520 = insertelement <4 x float> %1519, float %1504, i64 2, !dbg !11
  %1521 = insertelement <4 x float> %1520, float %1505, i64 3, !dbg !11
  %1522 = select <4 x i1> %1346, <4 x float> %1517, <4 x float> %1521, !dbg !11
  %1523 = extractvalue [2 x <8 x float>] %.lcssa4, 1, !dbg !11
  %1524 = extractelement <8 x float> %1523, i64 0, !dbg !11
  %1525 = extractelement <8 x float> %1523, i64 1, !dbg !11
  %1526 = extractelement <8 x float> %1523, i64 2, !dbg !11
  %1527 = extractelement <8 x float> %1523, i64 3, !dbg !11
  %1528 = select i1 %.not, float %1525, float %1524, !dbg !11
  %1529 = bitcast float %1528 to i32, !dbg !11
  %1530 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1529), !dbg !11
  %1531 = bitcast i32 %1530 to float, !dbg !11
  %1532 = select i1 %.not, float %1524, float %1531, !dbg !11
  %1533 = select i1 %.not, float %1531, float %1525, !dbg !11
  %1534 = select i1 %.not, float %1527, float %1526, !dbg !11
  %1535 = bitcast float %1534 to i32, !dbg !11
  %1536 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1535), !dbg !11
  %1537 = bitcast i32 %1536 to float, !dbg !11
  %1538 = select i1 %.not, float %1526, float %1537, !dbg !11
  %1539 = select i1 %.not, float %1537, float %1527, !dbg !11
  %1540 = select i1 %.not36, float %1538, float %1532, !dbg !11
  %1541 = bitcast float %1540 to i32, !dbg !11
  %1542 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1541), !dbg !11
  %1543 = bitcast i32 %1542 to float, !dbg !11
  %1544 = select i1 %.not36, float %1539, float %1533, !dbg !11
  %1545 = bitcast float %1544 to i32, !dbg !11
  %1546 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1545), !dbg !11
  %1547 = bitcast i32 %1546 to float, !dbg !11
  %1548 = insertelement <4 x float> poison, float %1532, i64 0, !dbg !11
  %1549 = insertelement <4 x float> %1548, float %1533, i64 1, !dbg !11
  %1550 = insertelement <4 x float> %1549, float %1543, i64 2, !dbg !11
  %1551 = insertelement <4 x float> %1550, float %1547, i64 3, !dbg !11
  %1552 = insertelement <4 x float> poison, float %1543, i64 0, !dbg !11
  %1553 = insertelement <4 x float> %1552, float %1547, i64 1, !dbg !11
  %1554 = insertelement <4 x float> %1553, float %1538, i64 2, !dbg !11
  %1555 = insertelement <4 x float> %1554, float %1539, i64 3, !dbg !11
  %1556 = select <4 x i1> %1346, <4 x float> %1551, <4 x float> %1555, !dbg !11
  %1557 = extractelement <8 x float> %1523, i64 4, !dbg !11
  %1558 = extractelement <8 x float> %1523, i64 5, !dbg !11
  %1559 = extractelement <8 x float> %1523, i64 6, !dbg !11
  %1560 = extractelement <8 x float> %1523, i64 7, !dbg !11
  %1561 = select i1 %.not, float %1558, float %1557, !dbg !11
  %1562 = bitcast float %1561 to i32, !dbg !11
  %1563 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1562), !dbg !11
  %1564 = bitcast i32 %1563 to float, !dbg !11
  %1565 = select i1 %.not, float %1557, float %1564, !dbg !11
  %1566 = select i1 %.not, float %1564, float %1558, !dbg !11
  %1567 = select i1 %.not, float %1560, float %1559, !dbg !11
  %1568 = bitcast float %1567 to i32, !dbg !11
  %1569 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1568), !dbg !11
  %1570 = bitcast i32 %1569 to float, !dbg !11
  %1571 = select i1 %.not, float %1559, float %1570, !dbg !11
  %1572 = select i1 %.not, float %1570, float %1560, !dbg !11
  %1573 = select i1 %.not36, float %1571, float %1565, !dbg !11
  %1574 = bitcast float %1573 to i32, !dbg !11
  %1575 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1574), !dbg !11
  %1576 = bitcast i32 %1575 to float, !dbg !11
  %1577 = select i1 %.not36, float %1572, float %1566, !dbg !11
  %1578 = bitcast float %1577 to i32, !dbg !11
  %1579 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1578), !dbg !11
  %1580 = bitcast i32 %1579 to float, !dbg !11
  %1581 = insertelement <4 x float> poison, float %1565, i64 0, !dbg !11
  %1582 = insertelement <4 x float> %1581, float %1566, i64 1, !dbg !11
  %1583 = insertelement <4 x float> %1582, float %1576, i64 2, !dbg !11
  %1584 = insertelement <4 x float> %1583, float %1580, i64 3, !dbg !11
  %1585 = insertelement <4 x float> poison, float %1576, i64 0, !dbg !11
  %1586 = insertelement <4 x float> %1585, float %1580, i64 1, !dbg !11
  %1587 = insertelement <4 x float> %1586, float %1571, i64 2, !dbg !11
  %1588 = insertelement <4 x float> %1587, float %1572, i64 3, !dbg !11
  %1589 = select <4 x i1> %1346, <4 x float> %1584, <4 x float> %1588, !dbg !11
  %1590 = extractvalue [2 x <8 x float>] %.lcssa5, 0, !dbg !11
  %1591 = extractelement <8 x float> %1590, i64 0, !dbg !11
  %1592 = extractelement <8 x float> %1590, i64 1, !dbg !11
  %1593 = extractelement <8 x float> %1590, i64 2, !dbg !11
  %1594 = extractelement <8 x float> %1590, i64 3, !dbg !11
  %1595 = select i1 %.not, float %1592, float %1591, !dbg !11
  %1596 = bitcast float %1595 to i32, !dbg !11
  %1597 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1596), !dbg !11
  %1598 = bitcast i32 %1597 to float, !dbg !11
  %1599 = select i1 %.not, float %1591, float %1598, !dbg !11
  %1600 = select i1 %.not, float %1598, float %1592, !dbg !11
  %1601 = select i1 %.not, float %1594, float %1593, !dbg !11
  %1602 = bitcast float %1601 to i32, !dbg !11
  %1603 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1602), !dbg !11
  %1604 = bitcast i32 %1603 to float, !dbg !11
  %1605 = select i1 %.not, float %1593, float %1604, !dbg !11
  %1606 = select i1 %.not, float %1604, float %1594, !dbg !11
  %1607 = select i1 %.not36, float %1605, float %1599, !dbg !11
  %1608 = bitcast float %1607 to i32, !dbg !11
  %1609 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1608), !dbg !11
  %1610 = bitcast i32 %1609 to float, !dbg !11
  %1611 = select i1 %.not36, float %1606, float %1600, !dbg !11
  %1612 = bitcast float %1611 to i32, !dbg !11
  %1613 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1612), !dbg !11
  %1614 = bitcast i32 %1613 to float, !dbg !11
  %1615 = insertelement <4 x float> poison, float %1599, i64 0, !dbg !11
  %1616 = insertelement <4 x float> %1615, float %1600, i64 1, !dbg !11
  %1617 = insertelement <4 x float> %1616, float %1610, i64 2, !dbg !11
  %1618 = insertelement <4 x float> %1617, float %1614, i64 3, !dbg !11
  %1619 = insertelement <4 x float> poison, float %1610, i64 0, !dbg !11
  %1620 = insertelement <4 x float> %1619, float %1614, i64 1, !dbg !11
  %1621 = insertelement <4 x float> %1620, float %1605, i64 2, !dbg !11
  %1622 = insertelement <4 x float> %1621, float %1606, i64 3, !dbg !11
  %1623 = select <4 x i1> %1346, <4 x float> %1618, <4 x float> %1622, !dbg !11
  %1624 = extractelement <8 x float> %1590, i64 4, !dbg !11
  %1625 = extractelement <8 x float> %1590, i64 5, !dbg !11
  %1626 = extractelement <8 x float> %1590, i64 6, !dbg !11
  %1627 = extractelement <8 x float> %1590, i64 7, !dbg !11
  %1628 = select i1 %.not, float %1625, float %1624, !dbg !11
  %1629 = bitcast float %1628 to i32, !dbg !11
  %1630 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1629), !dbg !11
  %1631 = bitcast i32 %1630 to float, !dbg !11
  %1632 = select i1 %.not, float %1624, float %1631, !dbg !11
  %1633 = select i1 %.not, float %1631, float %1625, !dbg !11
  %1634 = select i1 %.not, float %1627, float %1626, !dbg !11
  %1635 = bitcast float %1634 to i32, !dbg !11
  %1636 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1635), !dbg !11
  %1637 = bitcast i32 %1636 to float, !dbg !11
  %1638 = select i1 %.not, float %1626, float %1637, !dbg !11
  %1639 = select i1 %.not, float %1637, float %1627, !dbg !11
  %1640 = select i1 %.not36, float %1638, float %1632, !dbg !11
  %1641 = bitcast float %1640 to i32, !dbg !11
  %1642 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1641), !dbg !11
  %1643 = bitcast i32 %1642 to float, !dbg !11
  %1644 = select i1 %.not36, float %1639, float %1633, !dbg !11
  %1645 = bitcast float %1644 to i32, !dbg !11
  %1646 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1645), !dbg !11
  %1647 = bitcast i32 %1646 to float, !dbg !11
  %1648 = insertelement <4 x float> poison, float %1632, i64 0, !dbg !11
  %1649 = insertelement <4 x float> %1648, float %1633, i64 1, !dbg !11
  %1650 = insertelement <4 x float> %1649, float %1643, i64 2, !dbg !11
  %1651 = insertelement <4 x float> %1650, float %1647, i64 3, !dbg !11
  %1652 = insertelement <4 x float> poison, float %1643, i64 0, !dbg !11
  %1653 = insertelement <4 x float> %1652, float %1647, i64 1, !dbg !11
  %1654 = insertelement <4 x float> %1653, float %1638, i64 2, !dbg !11
  %1655 = insertelement <4 x float> %1654, float %1639, i64 3, !dbg !11
  %1656 = select <4 x i1> %1346, <4 x float> %1651, <4 x float> %1655, !dbg !11
  %1657 = extractvalue [2 x <8 x float>] %.lcssa5, 1, !dbg !11
  %1658 = extractelement <8 x float> %1657, i64 0, !dbg !11
  %1659 = extractelement <8 x float> %1657, i64 1, !dbg !11
  %1660 = extractelement <8 x float> %1657, i64 2, !dbg !11
  %1661 = extractelement <8 x float> %1657, i64 3, !dbg !11
  %1662 = select i1 %.not, float %1659, float %1658, !dbg !11
  %1663 = bitcast float %1662 to i32, !dbg !11
  %1664 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1663), !dbg !11
  %1665 = bitcast i32 %1664 to float, !dbg !11
  %1666 = select i1 %.not, float %1658, float %1665, !dbg !11
  %1667 = select i1 %.not, float %1665, float %1659, !dbg !11
  %1668 = select i1 %.not, float %1661, float %1660, !dbg !11
  %1669 = bitcast float %1668 to i32, !dbg !11
  %1670 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1669), !dbg !11
  %1671 = bitcast i32 %1670 to float, !dbg !11
  %1672 = select i1 %.not, float %1660, float %1671, !dbg !11
  %1673 = select i1 %.not, float %1671, float %1661, !dbg !11
  %1674 = select i1 %.not36, float %1672, float %1666, !dbg !11
  %1675 = bitcast float %1674 to i32, !dbg !11
  %1676 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1675), !dbg !11
  %1677 = bitcast i32 %1676 to float, !dbg !11
  %1678 = select i1 %.not36, float %1673, float %1667, !dbg !11
  %1679 = bitcast float %1678 to i32, !dbg !11
  %1680 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1679), !dbg !11
  %1681 = bitcast i32 %1680 to float, !dbg !11
  %1682 = insertelement <4 x float> poison, float %1666, i64 0, !dbg !11
  %1683 = insertelement <4 x float> %1682, float %1667, i64 1, !dbg !11
  %1684 = insertelement <4 x float> %1683, float %1677, i64 2, !dbg !11
  %1685 = insertelement <4 x float> %1684, float %1681, i64 3, !dbg !11
  %1686 = insertelement <4 x float> poison, float %1677, i64 0, !dbg !11
  %1687 = insertelement <4 x float> %1686, float %1681, i64 1, !dbg !11
  %1688 = insertelement <4 x float> %1687, float %1672, i64 2, !dbg !11
  %1689 = insertelement <4 x float> %1688, float %1673, i64 3, !dbg !11
  %1690 = select <4 x i1> %1346, <4 x float> %1685, <4 x float> %1689, !dbg !11
  %1691 = extractelement <8 x float> %1657, i64 4, !dbg !11
  %1692 = extractelement <8 x float> %1657, i64 5, !dbg !11
  %1693 = extractelement <8 x float> %1657, i64 6, !dbg !11
  %1694 = extractelement <8 x float> %1657, i64 7, !dbg !11
  %1695 = select i1 %.not, float %1692, float %1691, !dbg !11
  %1696 = bitcast float %1695 to i32, !dbg !11
  %1697 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1696), !dbg !11
  %1698 = bitcast i32 %1697 to float, !dbg !11
  %1699 = select i1 %.not, float %1691, float %1698, !dbg !11
  %1700 = select i1 %.not, float %1698, float %1692, !dbg !11
  %1701 = select i1 %.not, float %1694, float %1693, !dbg !11
  %1702 = bitcast float %1701 to i32, !dbg !11
  %1703 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1702), !dbg !11
  %1704 = bitcast i32 %1703 to float, !dbg !11
  %1705 = select i1 %.not, float %1693, float %1704, !dbg !11
  %1706 = select i1 %.not, float %1704, float %1694, !dbg !11
  %1707 = select i1 %.not36, float %1705, float %1699, !dbg !11
  %1708 = bitcast float %1707 to i32, !dbg !11
  %1709 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1708), !dbg !11
  %1710 = bitcast i32 %1709 to float, !dbg !11
  %1711 = select i1 %.not36, float %1706, float %1700, !dbg !11
  %1712 = bitcast float %1711 to i32, !dbg !11
  %1713 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1712), !dbg !11
  %1714 = bitcast i32 %1713 to float, !dbg !11
  %1715 = insertelement <4 x float> poison, float %1699, i64 0, !dbg !11
  %1716 = insertelement <4 x float> %1715, float %1700, i64 1, !dbg !11
  %1717 = insertelement <4 x float> %1716, float %1710, i64 2, !dbg !11
  %1718 = insertelement <4 x float> %1717, float %1714, i64 3, !dbg !11
  %1719 = insertelement <4 x float> poison, float %1710, i64 0, !dbg !11
  %1720 = insertelement <4 x float> %1719, float %1714, i64 1, !dbg !11
  %1721 = insertelement <4 x float> %1720, float %1705, i64 2, !dbg !11
  %1722 = insertelement <4 x float> %1721, float %1706, i64 3, !dbg !11
  %1723 = select <4 x i1> %1346, <4 x float> %1718, <4 x float> %1722, !dbg !11
  %1724 = extractvalue [2 x <8 x float>] %.lcssa7, 0, !dbg !11
  %1725 = extractelement <8 x float> %1724, i64 0, !dbg !11
  %1726 = extractelement <8 x float> %1724, i64 1, !dbg !11
  %1727 = extractelement <8 x float> %1724, i64 2, !dbg !11
  %1728 = extractelement <8 x float> %1724, i64 3, !dbg !11
  %1729 = select i1 %.not, float %1726, float %1725, !dbg !11
  %1730 = bitcast float %1729 to i32, !dbg !11
  %1731 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1730), !dbg !11
  %1732 = bitcast i32 %1731 to float, !dbg !11
  %1733 = select i1 %.not, float %1725, float %1732, !dbg !11
  %1734 = select i1 %.not, float %1732, float %1726, !dbg !11
  %1735 = select i1 %.not, float %1728, float %1727, !dbg !11
  %1736 = bitcast float %1735 to i32, !dbg !11
  %1737 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1736), !dbg !11
  %1738 = bitcast i32 %1737 to float, !dbg !11
  %1739 = select i1 %.not, float %1727, float %1738, !dbg !11
  %1740 = select i1 %.not, float %1738, float %1728, !dbg !11
  %1741 = select i1 %.not36, float %1739, float %1733, !dbg !11
  %1742 = bitcast float %1741 to i32, !dbg !11
  %1743 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1742), !dbg !11
  %1744 = select i1 %.not36, float %1740, float %1734, !dbg !11
  %1745 = bitcast float %1744 to i32, !dbg !11
  %1746 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1745), !dbg !11
  %1747 = extractelement <8 x float> %1724, i64 4, !dbg !11
  %1748 = extractelement <8 x float> %1724, i64 5, !dbg !11
  %1749 = extractelement <8 x float> %1724, i64 6, !dbg !11
  %1750 = extractelement <8 x float> %1724, i64 7, !dbg !11
  %1751 = select i1 %.not, float %1748, float %1747, !dbg !11
  %1752 = bitcast float %1751 to i32, !dbg !11
  %1753 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1752), !dbg !11
  %1754 = bitcast i32 %1753 to float, !dbg !11
  %1755 = select i1 %.not, float %1747, float %1754, !dbg !11
  %1756 = select i1 %.not, float %1754, float %1748, !dbg !11
  %1757 = select i1 %.not, float %1750, float %1749, !dbg !11
  %1758 = bitcast float %1757 to i32, !dbg !11
  %1759 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1758), !dbg !11
  %1760 = bitcast i32 %1759 to float, !dbg !11
  %1761 = select i1 %.not, float %1749, float %1760, !dbg !11
  %1762 = select i1 %.not, float %1760, float %1750, !dbg !11
  %1763 = select i1 %.not36, float %1761, float %1755, !dbg !11
  %1764 = bitcast float %1763 to i32, !dbg !11
  %1765 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1764), !dbg !11
  %1766 = select i1 %.not36, float %1762, float %1756, !dbg !11
  %1767 = bitcast float %1766 to i32, !dbg !11
  %1768 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1767), !dbg !11
  %1769 = extractvalue [2 x <8 x float>] %.lcssa7, 1, !dbg !11
  %1770 = extractelement <8 x float> %1769, i64 0, !dbg !11
  %1771 = extractelement <8 x float> %1769, i64 1, !dbg !11
  %1772 = extractelement <8 x float> %1769, i64 2, !dbg !11
  %1773 = extractelement <8 x float> %1769, i64 3, !dbg !11
  %1774 = select i1 %.not, float %1771, float %1770, !dbg !11
  %1775 = bitcast float %1774 to i32, !dbg !11
  %1776 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1775), !dbg !11
  %1777 = bitcast i32 %1776 to float, !dbg !11
  %1778 = select i1 %.not, float %1770, float %1777, !dbg !11
  %1779 = select i1 %.not, float %1777, float %1771, !dbg !11
  %1780 = select i1 %.not, float %1773, float %1772, !dbg !11
  %1781 = bitcast float %1780 to i32, !dbg !11
  %1782 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1781), !dbg !11
  %1783 = bitcast i32 %1782 to float, !dbg !11
  %1784 = select i1 %.not, float %1772, float %1783, !dbg !11
  %1785 = select i1 %.not, float %1783, float %1773, !dbg !11
  %1786 = select i1 %.not36, float %1784, float %1778, !dbg !11
  %1787 = bitcast float %1786 to i32, !dbg !11
  %1788 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1787), !dbg !11
  %1789 = select i1 %.not36, float %1785, float %1779, !dbg !11
  %1790 = bitcast float %1789 to i32, !dbg !11
  %1791 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1790), !dbg !11
  %1792 = extractelement <8 x float> %1769, i64 4, !dbg !11
  %1793 = extractelement <8 x float> %1769, i64 5, !dbg !11
  %1794 = extractelement <8 x float> %1769, i64 6, !dbg !11
  %1795 = extractelement <8 x float> %1769, i64 7, !dbg !11
  %1796 = select i1 %.not, float %1793, float %1792, !dbg !11
  %1797 = bitcast float %1796 to i32, !dbg !11
  %1798 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1797), !dbg !11
  %1799 = bitcast i32 %1798 to float, !dbg !11
  %1800 = select i1 %.not, float %1792, float %1799, !dbg !11
  %1801 = select i1 %.not, float %1799, float %1793, !dbg !11
  %1802 = select i1 %.not, float %1795, float %1794, !dbg !11
  %1803 = bitcast float %1802 to i32, !dbg !11
  %1804 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1322, i32 %1803), !dbg !11
  %1805 = bitcast i32 %1804 to float, !dbg !11
  %1806 = select i1 %.not, float %1794, float %1805, !dbg !11
  %1807 = select i1 %.not, float %1805, float %1795, !dbg !11
  %1808 = select i1 %.not36, float %1806, float %1800, !dbg !11
  %1809 = bitcast float %1808 to i32, !dbg !11
  %1810 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1809), !dbg !11
  %1811 = select i1 %.not36, float %1807, float %1801, !dbg !11
  %1812 = bitcast float %1811 to i32, !dbg !11
  %1813 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1337, i32 %1812), !dbg !11
  %1814 = shufflevector <1 x float> %1311, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !55
  %1815 = shufflevector <1 x float> %1306, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !55
  %1816 = bitcast i32 %1810 to float, !dbg !11
  %1817 = bitcast i32 %1813 to float, !dbg !11
  %1818 = insertelement <4 x float> poison, float %1800, i64 0, !dbg !11
  %1819 = insertelement <4 x float> %1818, float %1801, i64 1, !dbg !11
  %1820 = insertelement <4 x float> %1819, float %1816, i64 2, !dbg !11
  %1821 = insertelement <4 x float> %1820, float %1817, i64 3, !dbg !11
  %1822 = insertelement <4 x float> poison, float %1816, i64 0, !dbg !11
  %1823 = insertelement <4 x float> %1822, float %1817, i64 1, !dbg !11
  %1824 = insertelement <4 x float> %1823, float %1806, i64 2, !dbg !11
  %1825 = insertelement <4 x float> %1824, float %1807, i64 3, !dbg !11
  %1826 = select <4 x i1> %1346, <4 x float> %1821, <4 x float> %1825, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1815, !dbg !55
  %normalized = fmul <4 x float> %row.reciprocal, %1826, !dbg !55
  %1827 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !56
  %1828 = extractelement <4 x half> %1827, i64 3, !dbg !56
  %1829 = extractelement <4 x half> %1827, i64 2, !dbg !56
  %1830 = extractelement <4 x half> %1827, i64 1, !dbg !56
  %1831 = extractelement <4 x half> %1827, i64 0, !dbg !56
  %1832 = bitcast i32 %1788 to float, !dbg !11
  %1833 = bitcast i32 %1791 to float, !dbg !11
  %1834 = insertelement <4 x float> poison, float %1778, i64 0, !dbg !11
  %1835 = insertelement <4 x float> %1834, float %1779, i64 1, !dbg !11
  %1836 = insertelement <4 x float> %1835, float %1832, i64 2, !dbg !11
  %1837 = insertelement <4 x float> %1836, float %1833, i64 3, !dbg !11
  %1838 = insertelement <4 x float> poison, float %1832, i64 0, !dbg !11
  %1839 = insertelement <4 x float> %1838, float %1833, i64 1, !dbg !11
  %1840 = insertelement <4 x float> %1839, float %1784, i64 2, !dbg !11
  %1841 = insertelement <4 x float> %1840, float %1785, i64 3, !dbg !11
  %1842 = select <4 x i1> %1346, <4 x float> %1837, <4 x float> %1841, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %1842, !dbg !55
  %1843 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !56
  %1844 = extractelement <4 x half> %1843, i64 3, !dbg !56
  %1845 = extractelement <4 x half> %1843, i64 2, !dbg !56
  %1846 = extractelement <4 x half> %1843, i64 1, !dbg !56
  %1847 = extractelement <4 x half> %1843, i64 0, !dbg !56
  %1848 = bitcast i32 %1765 to float, !dbg !11
  %1849 = bitcast i32 %1768 to float, !dbg !11
  %1850 = insertelement <4 x float> poison, float %1755, i64 0, !dbg !11
  %1851 = insertelement <4 x float> %1850, float %1756, i64 1, !dbg !11
  %1852 = insertelement <4 x float> %1851, float %1848, i64 2, !dbg !11
  %1853 = insertelement <4 x float> %1852, float %1849, i64 3, !dbg !11
  %1854 = insertelement <4 x float> poison, float %1848, i64 0, !dbg !11
  %1855 = insertelement <4 x float> %1854, float %1849, i64 1, !dbg !11
  %1856 = insertelement <4 x float> %1855, float %1761, i64 2, !dbg !11
  %1857 = insertelement <4 x float> %1856, float %1762, i64 3, !dbg !11
  %1858 = select <4 x i1> %1346, <4 x float> %1853, <4 x float> %1857, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1814, !dbg !55
  %normalized106 = fmul <4 x float> %row.reciprocal105, %1858, !dbg !55
  %1859 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !56
  %1860 = extractelement <4 x half> %1859, i64 3, !dbg !56
  %1861 = extractelement <4 x half> %1859, i64 2, !dbg !56
  %1862 = extractelement <4 x half> %1859, i64 1, !dbg !56
  %1863 = extractelement <4 x half> %1859, i64 0, !dbg !56
  %1864 = bitcast i32 %1743 to float, !dbg !11
  %1865 = bitcast i32 %1746 to float, !dbg !11
  %1866 = insertelement <4 x float> poison, float %1733, i64 0, !dbg !11
  %1867 = insertelement <4 x float> %1866, float %1734, i64 1, !dbg !11
  %1868 = insertelement <4 x float> %1867, float %1864, i64 2, !dbg !11
  %1869 = insertelement <4 x float> %1868, float %1865, i64 3, !dbg !11
  %1870 = insertelement <4 x float> poison, float %1864, i64 0, !dbg !11
  %1871 = insertelement <4 x float> %1870, float %1865, i64 1, !dbg !11
  %1872 = insertelement <4 x float> %1871, float %1739, i64 2, !dbg !11
  %1873 = insertelement <4 x float> %1872, float %1740, i64 3, !dbg !11
  %1874 = select <4 x i1> %1346, <4 x float> %1869, <4 x float> %1873, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %1874, !dbg !55
  %1875 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !56
  %1876 = extractelement <4 x half> %1875, i64 3, !dbg !56
  %1877 = extractelement <4 x half> %1875, i64 2, !dbg !56
  %1878 = extractelement <4 x half> %1875, i64 1, !dbg !56
  %1879 = extractelement <4 x half> %1875, i64 0, !dbg !56
  %address.thread1041 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !57, !range !9, !frisk.register_lifetimes !5
  %address.local1042 = zext nneg i32 %address.thread1041 to i64, !dbg !8
  %address.local1044 = lshr i64 %address.local1042, 2, !dbg !15
  %address.local1045 = and i64 %address.local1044, 48, !dbg !15
  %address.local1047 = and i64 %address.local1042, 240, !dbg !57
  %address.local1046 = add i64 %31, %address.local1042, !dbg !57
  %address.local1049 = sub i64 %address.local1046, %address.local1047, !dbg !57
  %address.local1050 = add i64 %address.local1049, %address.local1045, !dbg !57
  %1880 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %40, !dbg !57
  %address.local1053.idx = shl nuw nsw i64 %address.local1050, 8, !dbg !57
  %address.local1053 = getelementptr inbounds nuw i8, ptr addrspace(1) %1880, i64 %address.local1053.idx, !dbg !57
  %address.local1055 = and i64 %address.local1044, 12, !dbg !10
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1055, !dbg !57
  store half %1879, ptr addrspace(1) %address.local1056, align 2, !dbg !57
  %address.local1058 = or disjoint i64 %address.local1055, 1, !dbg !57
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1058, !dbg !57
  store half %1878, ptr addrspace(1) %address.local1059, align 2, !dbg !57
  %address.local1061 = or disjoint i64 %address.local1055, 2, !dbg !57
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1061, !dbg !57
  store half %1877, ptr addrspace(1) %address.local1062, align 2, !dbg !57
  %address.local1064 = or disjoint i64 %address.local1055, 3, !dbg !57
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1064, !dbg !57
  store half %1876, ptr addrspace(1) %address.local1065, align 2, !dbg !57
  %address.local1067 = or disjoint i64 %address.local1055, 16, !dbg !57
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1067, !dbg !57
  store half %1863, ptr addrspace(1) %address.local1068, align 2, !dbg !57
  %address.local1070 = or disjoint i64 %address.local1055, 17, !dbg !57
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1070, !dbg !57
  store half %1862, ptr addrspace(1) %address.local1071, align 2, !dbg !57
  %address.local1073 = or disjoint i64 %address.local1055, 18, !dbg !57
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1073, !dbg !57
  store half %1861, ptr addrspace(1) %address.local1074, align 2, !dbg !57
  %address.local1076 = or disjoint i64 %address.local1055, 19, !dbg !57
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1076, !dbg !57
  store half %1860, ptr addrspace(1) %address.local1077, align 2, !dbg !57
  %address.local1081 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1053, i64 16384, !dbg !57
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1055, !dbg !57
  store half %1847, ptr addrspace(1) %address.local1082, align 2, !dbg !57
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1058, !dbg !57
  store half %1846, ptr addrspace(1) %address.local1084, align 2, !dbg !57
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !57
  store half %1845, ptr addrspace(1) %address.local1086, align 2, !dbg !57
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1064, !dbg !57
  store half %1844, ptr addrspace(1) %address.local1088, align 2, !dbg !57
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1067, !dbg !57
  store half %1831, ptr addrspace(1) %address.local1090, align 2, !dbg !57
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1070, !dbg !57
  store half %1830, ptr addrspace(1) %address.local1092, align 2, !dbg !57
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1073, !dbg !57
  store half %1829, ptr addrspace(1) %address.local1094, align 2, !dbg !57
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1076, !dbg !57
  store half %1828, ptr addrspace(1) %address.local1096, align 2, !dbg !57
  %normalized109 = fmul <4 x float> %row.reciprocal, %1723, !dbg !55
  %1881 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !56
  %1882 = extractelement <4 x half> %1881, i64 3, !dbg !56
  %1883 = extractelement <4 x half> %1881, i64 2, !dbg !56
  %1884 = extractelement <4 x half> %1881, i64 1, !dbg !56
  %1885 = extractelement <4 x half> %1881, i64 0, !dbg !56
  %normalized110 = fmul <4 x float> %row.reciprocal, %1690, !dbg !55
  %1886 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !56
  %1887 = extractelement <4 x half> %1886, i64 3, !dbg !56
  %1888 = extractelement <4 x half> %1886, i64 2, !dbg !56
  %1889 = extractelement <4 x half> %1886, i64 1, !dbg !56
  %1890 = extractelement <4 x half> %1886, i64 0, !dbg !56
  %normalized112 = fmul <4 x float> %row.reciprocal105, %1656, !dbg !55
  %1891 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !56
  %1892 = extractelement <4 x half> %1891, i64 3, !dbg !56
  %1893 = extractelement <4 x half> %1891, i64 2, !dbg !56
  %1894 = extractelement <4 x half> %1891, i64 1, !dbg !56
  %1895 = extractelement <4 x half> %1891, i64 0, !dbg !56
  %normalized113 = fmul <4 x float> %row.reciprocal105, %1623, !dbg !55
  %1896 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !56
  %1897 = extractelement <4 x half> %1896, i64 3, !dbg !56
  %1898 = extractelement <4 x half> %1896, i64 2, !dbg !56
  %1899 = extractelement <4 x half> %1896, i64 1, !dbg !56
  %1900 = extractelement <4 x half> %1896, i64 0, !dbg !56
  %address.local1112 = or disjoint i64 %address.local1055, 32, !dbg !57
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1112, !dbg !57
  store half %1900, ptr addrspace(1) %address.local1113, align 2, !dbg !57
  %address.local1115 = or disjoint i64 %address.local1055, 33, !dbg !57
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1115, !dbg !57
  store half %1899, ptr addrspace(1) %address.local1116, align 2, !dbg !57
  %address.local1118 = or disjoint i64 %address.local1055, 34, !dbg !57
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1118, !dbg !57
  store half %1898, ptr addrspace(1) %address.local1119, align 2, !dbg !57
  %address.local1121 = or disjoint i64 %address.local1055, 35, !dbg !57
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1121, !dbg !57
  store half %1897, ptr addrspace(1) %address.local1122, align 2, !dbg !57
  %address.local1124 = or disjoint i64 %address.local1055, 48, !dbg !57
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1124, !dbg !57
  store half %1895, ptr addrspace(1) %address.local1125, align 2, !dbg !57
  %address.local1127 = or disjoint i64 %address.local1055, 49, !dbg !57
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1127, !dbg !57
  store half %1894, ptr addrspace(1) %address.local1128, align 2, !dbg !57
  %address.local1130 = or disjoint i64 %address.local1055, 50, !dbg !57
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1130, !dbg !57
  store half %1893, ptr addrspace(1) %address.local1131, align 2, !dbg !57
  %address.local1133 = or i64 %address.local1044, 51, !dbg !57
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1133, !dbg !57
  store half %1892, ptr addrspace(1) %address.local1134, align 2, !dbg !57
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1112, !dbg !57
  store half %1890, ptr addrspace(1) %address.local1139, align 2, !dbg !57
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1115, !dbg !57
  store half %1889, ptr addrspace(1) %address.local1141, align 2, !dbg !57
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1118, !dbg !57
  store half %1888, ptr addrspace(1) %address.local1143, align 2, !dbg !57
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1121, !dbg !57
  store half %1887, ptr addrspace(1) %address.local1145, align 2, !dbg !57
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1124, !dbg !57
  store half %1885, ptr addrspace(1) %address.local1147, align 2, !dbg !57
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1127, !dbg !57
  store half %1884, ptr addrspace(1) %address.local1149, align 2, !dbg !57
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1130, !dbg !57
  store half %1883, ptr addrspace(1) %address.local1151, align 2, !dbg !57
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1133, !dbg !57
  store half %1882, ptr addrspace(1) %address.local1153, align 2, !dbg !57
  %normalized115 = fmul <4 x float> %row.reciprocal, %1589, !dbg !55
  %1901 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !56
  %1902 = extractelement <4 x half> %1901, i64 3, !dbg !56
  %1903 = extractelement <4 x half> %1901, i64 2, !dbg !56
  %1904 = extractelement <4 x half> %1901, i64 1, !dbg !56
  %1905 = extractelement <4 x half> %1901, i64 0, !dbg !56
  %normalized116 = fmul <4 x float> %row.reciprocal, %1556, !dbg !55
  %1906 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !56
  %1907 = extractelement <4 x half> %1906, i64 3, !dbg !56
  %1908 = extractelement <4 x half> %1906, i64 2, !dbg !56
  %1909 = extractelement <4 x half> %1906, i64 1, !dbg !56
  %1910 = extractelement <4 x half> %1906, i64 0, !dbg !56
  %normalized118 = fmul <4 x float> %row.reciprocal105, %1522, !dbg !55
  %1911 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !56
  %1912 = extractelement <4 x half> %1911, i64 3, !dbg !56
  %1913 = extractelement <4 x half> %1911, i64 2, !dbg !56
  %1914 = extractelement <4 x half> %1911, i64 1, !dbg !56
  %1915 = extractelement <4 x half> %1911, i64 0, !dbg !56
  %normalized119 = fmul <4 x float> %row.reciprocal105, %1489, !dbg !55
  %1916 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !56
  %1917 = extractelement <4 x half> %1916, i64 3, !dbg !56
  %1918 = extractelement <4 x half> %1916, i64 2, !dbg !56
  %1919 = extractelement <4 x half> %1916, i64 1, !dbg !56
  %1920 = extractelement <4 x half> %1916, i64 0, !dbg !56
  %address.local1169 = or disjoint i64 %address.local1055, 64, !dbg !57
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1169, !dbg !57
  store half %1920, ptr addrspace(1) %address.local1170, align 2, !dbg !57
  %address.local1172 = or disjoint i64 %address.local1055, 65, !dbg !57
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1172, !dbg !57
  store half %1919, ptr addrspace(1) %address.local1173, align 2, !dbg !57
  %address.local1175 = or disjoint i64 %address.local1055, 66, !dbg !57
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1175, !dbg !57
  store half %1918, ptr addrspace(1) %address.local1176, align 2, !dbg !57
  %address.local1178 = or disjoint i64 %address.local1055, 67, !dbg !57
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1178, !dbg !57
  store half %1917, ptr addrspace(1) %address.local1179, align 2, !dbg !57
  %address.local1181 = or disjoint i64 %address.local1055, 80, !dbg !57
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1181, !dbg !57
  store half %1915, ptr addrspace(1) %address.local1182, align 2, !dbg !57
  %address.local1184 = or disjoint i64 %address.local1055, 81, !dbg !57
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1184, !dbg !57
  store half %1914, ptr addrspace(1) %address.local1185, align 2, !dbg !57
  %address.local1187 = or disjoint i64 %address.local1055, 82, !dbg !57
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1187, !dbg !57
  store half %1913, ptr addrspace(1) %address.local1188, align 2, !dbg !57
  %address.local1190 = or disjoint i64 %address.local1055, 83, !dbg !57
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1190, !dbg !57
  store half %1912, ptr addrspace(1) %address.local1191, align 2, !dbg !57
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1169, !dbg !57
  store half %1910, ptr addrspace(1) %address.local1196, align 2, !dbg !57
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1172, !dbg !57
  store half %1909, ptr addrspace(1) %address.local1198, align 2, !dbg !57
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1175, !dbg !57
  store half %1908, ptr addrspace(1) %address.local1200, align 2, !dbg !57
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1178, !dbg !57
  store half %1907, ptr addrspace(1) %address.local1202, align 2, !dbg !57
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1181, !dbg !57
  store half %1905, ptr addrspace(1) %address.local1204, align 2, !dbg !57
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1184, !dbg !57
  store half %1904, ptr addrspace(1) %address.local1206, align 2, !dbg !57
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1187, !dbg !57
  store half %1903, ptr addrspace(1) %address.local1208, align 2, !dbg !57
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1190, !dbg !57
  store half %1902, ptr addrspace(1) %address.local1210, align 2, !dbg !57
  %normalized121 = fmul <4 x float> %row.reciprocal, %1455, !dbg !55
  %1921 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !56
  %1922 = extractelement <4 x half> %1921, i64 3, !dbg !56
  %1923 = extractelement <4 x half> %1921, i64 2, !dbg !56
  %1924 = extractelement <4 x half> %1921, i64 1, !dbg !56
  %1925 = extractelement <4 x half> %1921, i64 0, !dbg !56
  %normalized122 = fmul <4 x float> %row.reciprocal, %1422, !dbg !55
  %1926 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !56
  %1927 = extractelement <4 x half> %1926, i64 3, !dbg !56
  %1928 = extractelement <4 x half> %1926, i64 2, !dbg !56
  %1929 = extractelement <4 x half> %1926, i64 1, !dbg !56
  %1930 = extractelement <4 x half> %1926, i64 0, !dbg !56
  %normalized124 = fmul <4 x float> %row.reciprocal105, %1388, !dbg !55
  %1931 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !56
  %1932 = extractelement <4 x half> %1931, i64 3, !dbg !56
  %1933 = extractelement <4 x half> %1931, i64 2, !dbg !56
  %1934 = extractelement <4 x half> %1931, i64 1, !dbg !56
  %1935 = extractelement <4 x half> %1931, i64 0, !dbg !56
  %normalized125 = fmul <4 x float> %row.reciprocal105, %1355, !dbg !55
  %1936 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !56
  %1937 = extractelement <4 x half> %1936, i64 3, !dbg !56
  %1938 = extractelement <4 x half> %1936, i64 2, !dbg !56
  %1939 = extractelement <4 x half> %1936, i64 1, !dbg !56
  %1940 = extractelement <4 x half> %1936, i64 0, !dbg !56
  %address.local1226 = or disjoint i64 %address.local1055, 96, !dbg !57
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1226, !dbg !57
  store half %1940, ptr addrspace(1) %address.local1227, align 2, !dbg !57
  %address.local1229 = or disjoint i64 %address.local1055, 97, !dbg !57
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1229, !dbg !57
  store half %1939, ptr addrspace(1) %address.local1230, align 2, !dbg !57
  %address.local1232 = or disjoint i64 %address.local1055, 98, !dbg !57
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1232, !dbg !57
  store half %1938, ptr addrspace(1) %address.local1233, align 2, !dbg !57
  %address.local1235 = or disjoint i64 %address.local1055, 99, !dbg !57
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1235, !dbg !57
  store half %1937, ptr addrspace(1) %address.local1236, align 2, !dbg !57
  %address.local1238 = or disjoint i64 %address.local1055, 112, !dbg !57
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1238, !dbg !57
  store half %1935, ptr addrspace(1) %address.local1239, align 2, !dbg !57
  %address.local1241 = or disjoint i64 %address.local1055, 113, !dbg !57
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1241, !dbg !57
  store half %1934, ptr addrspace(1) %address.local1242, align 2, !dbg !57
  %address.local1244 = or disjoint i64 %address.local1055, 114, !dbg !57
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1244, !dbg !57
  store half %1933, ptr addrspace(1) %address.local1245, align 2, !dbg !57
  %address.local1247 = or i64 %address.local1044, 115, !dbg !57
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1247, !dbg !57
  store half %1932, ptr addrspace(1) %address.local1248, align 2, !dbg !57
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1226, !dbg !57
  store half %1930, ptr addrspace(1) %address.local1253, align 2, !dbg !57
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1229, !dbg !57
  store half %1929, ptr addrspace(1) %address.local1255, align 2, !dbg !57
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1232, !dbg !57
  store half %1928, ptr addrspace(1) %address.local1257, align 2, !dbg !57
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1235, !dbg !57
  store half %1927, ptr addrspace(1) %address.local1259, align 2, !dbg !57
  %address.local1261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1238, !dbg !57
  store half %1925, ptr addrspace(1) %address.local1261, align 2, !dbg !57
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1241, !dbg !57
  store half %1924, ptr addrspace(1) %address.local1263, align 2, !dbg !57
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1244, !dbg !57
  store half %1923, ptr addrspace(1) %address.local1265, align 2, !dbg !57
  %address.local1267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1247, !dbg !57
  store half %1922, ptr addrspace(1) %address.local1267, align 2, !dbg !57
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

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

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #4

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
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
!7 = !DILocation(line: 23, column: 3, scope: !3)
!8 = !DILocation(line: 70, column: 22, scope: !3)
!9 = !{i32 0, i32 256}
!10 = !DILocation(line: 72, column: 24, scope: !3)
!11 = !DILocation(line: 60, column: 24, scope: !3)
!12 = !DILocation(line: 0, scope: !3)
!13 = !DILocation(line: 71, column: 22, scope: !3)
!14 = !DILocation(line: 73, column: 26, scope: !3)
!15 = !DILocation(line: 77, column: 19, scope: !3)
!16 = !DILocation(line: 79, column: 24, scope: !3)
!17 = !DILocation(line: 80, column: 23, scope: !3)
!18 = !{i32 0}
!19 = !{i32 1}
!20 = !{i32 2}
!21 = !{i32 3}
!22 = !{i32 4}
!23 = !{i32 5}
!24 = !{i32 6}
!25 = !{i32 7}
!26 = !{i32 8}
!27 = !{i32 9}
!28 = !{i32 10}
!29 = !{i32 11}
!30 = !{i32 12}
!31 = !{i32 13}
!32 = !{i32 14}
!33 = !{i32 15}
!34 = !{i32 16}
!35 = !{i32 17}
!36 = !{i32 18}
!37 = !{i32 19}
!38 = !{i32 20}
!39 = !{i32 21}
!40 = !{i32 22}
!41 = !DILocation(line: 101, column: 20, scope: !3)
!42 = !DILocation(line: 81, column: 26, scope: !3)
!43 = !DILocation(line: 88, column: 21, scope: !3)
!44 = !DILocation(line: 89, column: 20, scope: !3)
!45 = !DILocation(line: 90, column: 22, scope: !3)
!46 = !DILocation(line: 91, column: 23, scope: !3)
!47 = !DILocation(line: 92, column: 20, scope: !3)
!48 = !DILocation(line: 103, column: 22, scope: !3)
!49 = !DILocation(line: 93, column: 22, scope: !3)
!50 = !DILocation(line: 102, column: 23, scope: !3)
!51 = !{i64 8192}
!52 = !DILocation(line: 94, column: 22, scope: !3)
!53 = !DILocation(line: 96, column: 16, scope: !3)
!54 = !DILocation(line: 95, column: 20, scope: !3)
!55 = !DILocation(line: 107, column: 23, scope: !3)
!56 = !DILocation(line: 108, column: 19, scope: !3)
!57 = !DILocation(line: 109, column: 9, scope: !3)
