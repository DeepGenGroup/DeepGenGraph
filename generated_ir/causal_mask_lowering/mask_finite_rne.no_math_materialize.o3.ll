; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/causal_mask_lowering/mask_finite_pk_rtz.ll'
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
  %.idx1447 = shl i64 %31, 8, !dbg !8
  %35 = add i64 %.idx, %.idx1447, !dbg !8
  %address.local481.idx = shl nuw nsw i64 %address.local480, 1, !dbg !8
  %36 = or disjoint i64 %address.local481.idx, %35, !dbg !8
  %dma.buffer.offset1445 = trunc i64 %36 to i32, !dbg !8
  %37 = insertelement <2 x i64> poison, i64 %34, i64 0, !dbg !8
  %38 = bitcast <2 x i64> %37 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1446 = shufflevector <4 x i32> %38, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local482 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local483 = shl nuw nsw i64 %address.local482, 3, !dbg !8
  %address.local484 = and i64 %address.local483, 3584, !dbg !8
  %address.local485 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local484, !dbg !8
  %address.local490 = lshr i64 %address.local482, 4, !dbg !8
  %address.local491 = and i64 %address.local490, 24, !dbg !8
  %address.local492 = or disjoint i64 %address.local491, %address.local477, !dbg !8
  %address.local493.idx = shl nuw nsw i64 %address.local492, 1, !dbg !8
  %39 = or disjoint i64 %address.local493.idx, %35, !dbg !8
  %dma.buffer.offset1443 = trunc i64 %39 to i32, !dbg !8
  %40 = shl nuw nsw i64 %28, 19, !dbg !8
  %41 = add i64 %40, %32, !dbg !8
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local472, !dbg !8
  %42 = or disjoint i64 %41, 32, !dbg !8
  %43 = or disjoint i64 %address.local480, %42, !dbg !8
  %.tr = trunc i64 %43 to i32, !dbg !8
  %dma.buffer.offset1441 = shl i32 %.tr, 1, !dbg !8
  %address.local498 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local484, !dbg !8
  %44 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local473, i64 0, !dbg !8
  %45 = insertelement <4 x ptr addrspace(3)> %44, ptr addrspace(3) %address.local485, i64 1, !dbg !8
  %46 = insertelement <4 x ptr addrspace(3)> %45, ptr addrspace(3) %address.local494, i64 2, !dbg !8
  %47 = insertelement <4 x ptr addrspace(3)> %46, ptr addrspace(3) %address.local498, i64 3, !dbg !8
  %48 = ptrtoint <4 x ptr addrspace(3)> %47 to <4 x i64>, !dbg !8
  %49 = trunc <4 x i64> %48 to <4 x i32>, !dbg !8
  %50 = extractelement <4 x i32> %49, i64 0, !dbg !8
  %51 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %52 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1445, <4 x i32> %dma.buffer.rsrc1446) #9, !dbg !8
  %53 = extractelement <4 x i32> %49, i64 1, !dbg !8
  %54 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %55 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %54, i32 %dma.buffer.offset1443, <4 x i32> %dma.buffer.rsrc1446) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %56 = extractelement <4 x i32> %49, i64 2, !dbg !8
  %57 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %56), !dbg !8
  %58 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1441, <4 x i32> %dma.buffer.rsrc1446) #9, !dbg !8
  %59 = extractelement <4 x i32> %49, i64 3, !dbg !8
  %60 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %59), !dbg !8
  %61 = or disjoint i64 %address.local492, %42, !dbg !8
  %.tr1451 = trunc i64 %61 to i32, !dbg !8
  %dma.buffer.offset1439 = shl i32 %.tr1451, 1, !dbg !8
  %62 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %60, i32 %dma.buffer.offset1439, <4 x i32> %dma.buffer.rsrc1446) #9, !dbg !8
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
  %lds.const.gep1268 = getelementptr i8, ptr addrspace(3) %address.local514, i64 1024, !dbg !10
  %66 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !10
  %67 = fmul <4 x half> %64, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1269 = getelementptr i8, ptr addrspace(3) %address.local514, i64 5120, !dbg !10
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !10
  %69 = fmul <4 x half> %66, splat (half 1.275630e-01), !dbg !10
  %70 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %71 = or disjoint i64 %41, 64, !dbg !8
  %72 = or disjoint i64 %address.local480, %71, !dbg !8
  %.tr1453 = trunc i64 %72 to i32, !dbg !8
  %dma.buffer.offset1437 = shl i32 %.tr1453, 1, !dbg !8
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1437, <4 x i32> %dma.buffer.rsrc1446) #9, !dbg !8
  %74 = or disjoint i64 %address.local492, %71, !dbg !8
  %.tr1455 = trunc i64 %74 to i32, !dbg !8
  %dma.buffer.offset1435 = shl i32 %.tr1455, 1, !dbg !8
  %75 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %54, i32 %dma.buffer.offset1435, <4 x i32> %dma.buffer.rsrc1446) #9, !dbg !8
  %lds.const.gep1270 = getelementptr i8, ptr addrspace(3) %address.local514, i64 8192, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !10
  %lds.const.gep1271 = getelementptr i8, ptr addrspace(3) %address.local514, i64 12288, !dbg !10
  %77 = load <4 x half>, ptr addrspace(3) %lds.const.gep1271, align 8, !dbg !10
  %lds.const.gep1272 = getelementptr i8, ptr addrspace(3) %address.local514, i64 9216, !dbg !10
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1272, align 8, !dbg !10
  %79 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1273 = getelementptr i8, ptr addrspace(3) %address.local514, i64 13312, !dbg !10
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1273, align 8, !dbg !10
  %81 = fmul <4 x half> %77, splat (half 1.275630e-01), !dbg !10
  %82 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !10
  %83 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %84 = or disjoint i64 %41, 96, !dbg !8
  %85 = or disjoint i64 %address.local480, %84, !dbg !8
  %.tr1457 = trunc i64 %85 to i32, !dbg !8
  %dma.buffer.offset1433 = shl i32 %.tr1457, 1, !dbg !8
  %86 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1433, <4 x i32> %dma.buffer.rsrc1446) #9, !dbg !8
  %87 = or disjoint i64 %address.local492, %84, !dbg !8
  %.tr1459 = trunc i64 %87 to i32, !dbg !8
  %dma.buffer.offset1431 = shl i32 %.tr1459, 1, !dbg !8
  %88 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %60, i32 %dma.buffer.offset1431, <4 x i32> %dma.buffer.rsrc1446) #9, !dbg !8
  %89 = load <4 x half>, ptr addrspace(3) %address.local514, align 8, !dbg !10
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !10
  %92 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !10
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !10
  %94 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !10
  %95 = fmul <4 x half> %91, splat (half 1.275630e-01), !dbg !10
  %96 = fmul <4 x half> %93, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !10
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1271, align 8, !dbg !10
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1272, align 8, !dbg !10
  %100 = fmul <4 x half> %97, splat (half 1.275630e-01), !dbg !10
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1273, align 8, !dbg !10
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
  %122 = trunc i64 %121 to i32, !dbg !15
  %123 = add i32 %122, 65, !dbg !15
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
  %dma.buffer.rsrc1430 = shufflevector <4 x i32> %129, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %kdma.source = zext nneg i32 %kdma.owner to i64
  %kdma.source1603 = lshr i64 %kdma.source, 4
  %kdma.source1604 = and i64 %kdma.source1603, 8
  %kdma.source1605 = and i64 %kdma.source, 91
  %kdma.source1606 = lshr i64 %kdma.source, 2
  %kdma.source1607 = and i64 %kdma.source1606, 4
  %kdma.source1608 = add nuw nsw i64 %kdma.source1607, %kdma.source
  %kdma.source1609 = and i64 %kdma.source1608, 4
  %kdma.source1610 = or disjoint i64 %kdma.source1609, %kdma.source1605
  %kdma.source1611 = shl nuw nsw i64 %kdma.source1610, 7
  %kdma.source1612 = or disjoint i64 %kdma.source1611, %kdma.source1604
  %kdma.source1616 = zext nneg i32 %kdma.owner.hi to i64
  %kdma.source1617 = lshr i64 %kdma.source1616, 4
  %kdma.source1618 = and i64 %kdma.source1617, 8
  %kdma.source1619 = and i64 %kdma.source1616, 123
  %kdma.source1620 = lshr i64 %kdma.source1616, 2
  %kdma.source1621 = and i64 %kdma.source1620, 4
  %kdma.source1622 = add nuw nsw i64 %kdma.source1621, %kdma.source1616
  %kdma.source1623 = and i64 %kdma.source1622, 4
  %kdma.source1624 = or disjoint i64 %kdma.source1623, %kdma.source1619
  %kdma.source1625 = shl nuw nsw i64 %kdma.source1624, 7
  %kdma.source1626 = or disjoint i64 %kdma.source1625, %kdma.source1618
  %kdma.source1643 = or disjoint i64 %kdma.source1612, 16
  %kdma.source1660 = or disjoint i64 %kdma.source1626, 16
  %130 = or disjoint i64 %125, %124
  %kdma.source1680 = or disjoint i32 %kdma.intra, 64
  %kdma.source1750 = or disjoint i32 %kdma.intra, 128
  %kdma.source1820 = or disjoint i32 %kdma.intra, 192
  %131 = ptrtoint ptr addrspace(1) %2 to i64
  %132 = insertelement <2 x i64> poison, i64 %131, i64 0
  %133 = bitcast <2 x i64> %132 to <4 x i32>
  %dma.buffer.rsrc1414 = shufflevector <4 x i32> %133, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %134 = insertelement <2 x i64> poison, i64 %108, i64 0, !dbg !13
  %135 = insertelement <32 x i32> poison, i32 %123, i64 0, !dbg !15
  %136 = shufflevector <32 x i32> %135, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !15
  %137 = insertelement <32 x i32> poison, i32 %122, i64 0, !dbg !15
  %138 = insertelement <2 x i64> poison, i64 %130, i64 1
  %invariant.op = sub i32 72, %kdma.intra, !dbg !11
  %invariant.op76 = sub i32 72, %kdma.intra, !dbg !11
  %invariant.op77 = sub i32 136, %kdma.intra, !dbg !11
  %invariant.op78 = sub i32 136, %kdma.intra, !dbg !11
  %invariant.op79 = sub i32 200, %kdma.intra, !dbg !11
  %invariant.op80 = sub i32 200, %kdma.intra, !dbg !11
  %139 = insertelement <32 x i32> poison, i32 %122, i64 8
  br label %140, !dbg !11

140:                                              ; preds = %.lr.ph, %140
  %141 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1289, %140 ]
  %142 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1285, %140 ]
  %143 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1281, %140 ]
  %144 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1277, %140 ]
  %145 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %1041, %140 ]
  %146 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1275, %140 ]
  %147 = phi i64 [ 0, %.lr.ph ], [ %1290, %140 ]
  %148 = lshr i64 %147, 12, !dbg !13
  %149 = add nuw nsw i64 %148, %17, !dbg !13
  %150 = lshr i64 %147, 5, !dbg !13
  %151 = and i64 %150, 127, !dbg !13
  %152 = shl nuw nsw i64 %147, 7, !dbg !13
  %153 = shl i64 %149, 19, !dbg !13
  %154 = and i64 %153, 16252928, !dbg !13
  %155 = shl i64 %150, 19, !dbg !13
  %156 = shl nuw nsw i64 %147, 14, !dbg !13
  %157 = sub i64 %156, %155, !dbg !13
  %address.thread565 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %address.local566 = zext nneg i32 %address.thread565 to i64, !dbg !8
  %address.local567 = shl nuw nsw i64 %address.local566, 3, !dbg !8
  %address.local568 = and i64 %address.local567, 1536, !dbg !8
  %address.local569 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local568, !dbg !8
  %.idx14601463 = or disjoint i64 %154, %151, !dbg !13
  %158 = add i64 %.idx14601463, %157, !dbg !13
  %kdma.source1613 = add nuw nsw i64 %158, %kdma.source1612, !dbg !13
  %kdma.source1614 = trunc i64 %kdma.source1613 to i32, !dbg !13
  %kdma.source1615 = shl i32 %kdma.source1614, 1, !dbg !13
  %kdma.offset = add i32 %kdma.source1615, %kdma.intra, !dbg !13
  %kdma.source1627 = add i64 %158, %kdma.source1626, !dbg !13
  %kdma.source1628 = trunc i64 %kdma.source1627 to i32, !dbg !13
  %kdma.source1629 = shl i32 %kdma.source1628, 1, !dbg !13
  %kdma.offset1630 = add i32 %kdma.source1629, %kdma.intra.hi, !dbg !13
  %address.local583 = or disjoint i64 %address.local566, 256, !dbg !8
  %address.local584 = shl nuw nsw i64 %address.local583, 3, !dbg !8
  %address.local585 = and i64 %address.local584, 3584, !dbg !8
  %address.local586 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local585, !dbg !8
  %kdma.source1644 = add nuw nsw i64 %158, %kdma.source1643, !dbg !13
  %kdma.source1645 = trunc i64 %kdma.source1644 to i32, !dbg !13
  %kdma.source1646 = shl i32 %kdma.source1645, 1, !dbg !13
  %kdma.offset1647 = add i32 %kdma.source1646, %kdma.intra, !dbg !13
  %kdma.source1661 = add i64 %158, %kdma.source1660, !dbg !13
  %kdma.source1662 = trunc i64 %kdma.source1661 to i32, !dbg !13
  %kdma.source1663 = shl i32 %kdma.source1662, 1, !dbg !13
  %kdma.offset1664 = add i32 %kdma.source1663, %kdma.intra.hi, !dbg !13
  %address.local600 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local568, !dbg !8
  %.idx14701474 = add i64 %157, %150, !dbg !13
  %159 = add i64 %.idx14701474, %108, !dbg !13
  %160 = insertelement <2 x i64> %138, i64 %150, i64 0, !dbg !13
  %161 = insertelement <2 x i64> %134, i64 %147, i64 1, !dbg !13
  %162 = add nsw <2 x i64> %160, %161, !dbg !13
  %163 = extractelement <2 x i64> %162, i64 1, !dbg !13
  %164 = extractelement <2 x i64> %162, i64 0, !dbg !13
  %165 = add nuw nsw i64 %163, 1024, !dbg !13
  %166 = add nsw <2 x i64> %162, <i64 32, i64 1024>, !dbg !13
  %.lobit37 = ashr i64 %165, 63, !dbg !13
  %167 = icmp slt <2 x i64> %162, <i64 -32, i64 -1024>, !dbg !13
  %168 = icmp slt i64 %164, -32, !dbg !13
  %169 = sub nsw <2 x i64> <i64 -33, i64 -1025>, %162, !dbg !13
  %170 = select <2 x i1> %167, <2 x i64> %169, <2 x i64> %166, !dbg !13
  %171 = sext i1 %168 to i64, !dbg !13
  %172 = insertelement <2 x i64> poison, i64 %171, i64 0, !dbg !13
  %173 = insertelement <2 x i64> %172, i64 %.lobit37, i64 1, !dbg !13
  %address.local604 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local585, !dbg !8
  %174 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local569, i64 0, !dbg !8
  %175 = insertelement <4 x ptr addrspace(3)> %174, ptr addrspace(3) %address.local586, i64 1, !dbg !8
  %176 = insertelement <4 x ptr addrspace(3)> %175, ptr addrspace(3) %address.local600, i64 2, !dbg !8
  %177 = insertelement <4 x ptr addrspace(3)> %176, ptr addrspace(3) %address.local604, i64 3, !dbg !8
  %178 = ptrtoint <4 x ptr addrspace(3)> %177 to <4 x i64>, !dbg !8
  %179 = trunc <4 x i64> %178 to <4 x i32>, !dbg !8
  %180 = extractelement <4 x i32> %179, i64 0, !dbg !13
  %181 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %180), !dbg !13
  %182 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %181, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %kdma.lds1631 = add i32 %181, 512, !dbg !13
  %183 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1631, i32 %kdma.offset1630, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %184 = extractelement <4 x i32> %179, i64 1, !dbg !13
  %185 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %184), !dbg !13
  %186 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %185, i32 %kdma.offset1647, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %kdma.lds1665 = add i32 %185, 512, !dbg !13
  %187 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1665, i32 %kdma.offset1664, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %188 = extractelement <4 x i32> %179, i64 2, !dbg !13
  %189 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %188), !dbg !13
  %190 = sdiv <2 x i64> %170, <i64 128, i64 4096>, !dbg !13
  %191 = xor <2 x i64> %190, %173, !dbg !13
  %192 = shl <2 x i64> %191, <i64 7, i64 19>, !dbg !13
  %193 = extractelement <2 x i64> %192, i64 0, !dbg !13
  %194 = sub i64 %159, %193, !dbg !13
  %195 = extractelement <2 x i64> %192, i64 1, !dbg !13
  %196 = and i64 %195, 16252928, !dbg !13
  %197 = add i64 %194, %196, !dbg !13
  %kdma.source1677 = add i64 %197, %kdma.source1612, !dbg !13
  %kdma.source1678 = trunc i64 %kdma.source1677 to i32, !dbg !13
  %kdma.source1679 = shl i32 %kdma.source1678, 1, !dbg !13
  %kdma.offset1681 = add i32 %kdma.source1680, %kdma.source1679, !dbg !13
  %198 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %189, i32 %kdma.offset1681, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %kdma.source1694 = add i64 %197, %kdma.source1626, !dbg !13
  %kdma.source1695 = trunc i64 %kdma.source1694 to i32, !dbg !13
  %kdma.source1696 = shl i32 %kdma.source1695, 1, !dbg !13
  %kdma.offset1698.reass.reass = add i32 %kdma.source1696, %invariant.op
  %kdma.lds1699 = add i32 %189, 512, !dbg !13
  %199 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1699, i32 %kdma.offset1698.reass.reass, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %200 = extractelement <4 x i32> %179, i64 3, !dbg !13
  %201 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %200), !dbg !13
  %kdma.source1712 = add i64 %197, %kdma.source1643, !dbg !13
  %kdma.source1713 = trunc i64 %kdma.source1712 to i32, !dbg !13
  %kdma.source1714 = shl i32 %kdma.source1713, 1, !dbg !13
  %kdma.offset1716 = add i32 %kdma.source1680, %kdma.source1714, !dbg !13
  %202 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %201, i32 %kdma.offset1716, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %kdma.source1730 = add i64 %197, %kdma.source1660, !dbg !13
  %kdma.source1731 = trunc i64 %kdma.source1730 to i32, !dbg !13
  %kdma.source1732 = shl i32 %kdma.source1731, 1, !dbg !13
  %kdma.offset1734.reass.reass = add i32 %kdma.source1732, %invariant.op76
  %kdma.lds1735 = add i32 %201, 512, !dbg !13
  %203 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1735, i32 %kdma.offset1734.reass.reass, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
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
  %.idx1482 = mul nuw nsw i64 %address.local566, 320, !dbg !14
  %204 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1482, !dbg !14
  %.idx1483 = mul nsw i64 %address.local609, -1280, !dbg !14
  %205 = getelementptr i8, ptr addrspace(3) %204, i64 %.idx1483, !dbg !14
  %206 = getelementptr [2 x i8], ptr addrspace(3) %205, i64 %address.local613, !dbg !14
  %207 = getelementptr [2 x i8], ptr addrspace(3) %206, i64 %address.local617, !dbg !14
  %208 = getelementptr [2 x i8], ptr addrspace(3) %207, i64 %address.local620, !dbg !14
  %address.local626 = getelementptr [2 x i8], ptr addrspace(3) %208, i64 %address.local624, !dbg !14
  %209 = add nsw i64 %.idx1483, %.idx1482, !dbg !14
  %.idx2002 = shl nuw nsw i64 %address.local613, 1, !dbg !14
  %210 = or disjoint i64 %209, %.idx2002, !dbg !14
  %211 = or disjoint i64 %address.local617, %address.local620, !dbg !14
  %212 = shl nsw i64 %211, 1, !dbg !14
  %address.local626.idx = mul nuw nsw i64 %address.local623, 2032, !dbg !14
  %213 = add nsw i64 %210, %address.local626.idx, !dbg !14
  %214 = add nsw i64 %213, %212, !dbg !14
  %215 = lshr i64 %214, 6, !dbg !14
  %216 = and i64 %215, 8, !dbg !14
  %217 = xor i64 %216, %214, !dbg !14
  %218 = sub i64 %217, %214, !dbg !14
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local626, i64 %218, !dbg !14
  %219 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %219, <4 x float> zeroinitializer) #8, !dbg !14
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14
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
  %222 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local651, !dbg !14
  %223 = getelementptr [2 x i8], ptr addrspace(3) %222, i64 %address.local620, !dbg !14
  %224 = getelementptr [2 x i8], ptr addrspace(3) %223, i64 %address.local624, !dbg !14
  %address.local655 = getelementptr i8, ptr addrspace(3) %224, i64 64, !dbg !14
  %.idx20052008 = add i64 %address.local651, %address.local620, !dbg !14
  %225 = shl i64 %.idx20052008, 1, !dbg !14
  %226 = add nuw nsw i64 %address.local626.idx, 64, !dbg !14
  %227 = add i64 %226, %225, !dbg !14
  %228 = lshr i64 %227, 6, !dbg !14
  %229 = and i64 %228, 8, !dbg !14
  %230 = xor i64 %229, %227, !dbg !14
  %231 = sub i64 %230, %227, !dbg !14
  %kdma.swizzled1876 = getelementptr i8, ptr addrspace(3) %address.local655, i64 %231, !dbg !14
  %232 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1876, align 8, !dbg !14, !frisk.k_lds_pair !16
  %kdma.gep1880 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 128, !dbg !14
  %233 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !14, !frisk.k_lds_pair !16
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 4096, !dbg !14
  %234 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !14, !frisk.k_lds_pair !17
  %kdma.gep1881 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 4224, !dbg !14
  %235 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !14, !frisk.k_lds_pair !17
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %236 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !14, !frisk.k_lds_pair !18
  %kdma.gep1882 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %237 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !14, !frisk.k_lds_pair !18
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %238 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !14, !frisk.k_lds_pair !19
  %kdma.gep1883 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %239 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !14, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %221, <4 x float> %220) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep1884 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 1024, !dbg !14
  %241 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !14, !frisk.k_lds_pair !20
  %kdma.gep1888 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 1152, !dbg !14
  %242 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !14, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %232, <4 x float> zeroinitializer) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep1885 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 5120, !dbg !14
  %244 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !14, !frisk.k_lds_pair !21
  %kdma.gep1889 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 5248, !dbg !14
  %245 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !14, !frisk.k_lds_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %234, <4 x float> %243) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep1886 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %247 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !14, !frisk.k_lds_pair !22
  %kdma.gep1887 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %248 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !14, !frisk.k_lds_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %238, <4 x float> %249) #8, !dbg !14
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %233, <4 x float> zeroinitializer) #8, !dbg !14
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %235, <4 x float> %251) #8, !dbg !14
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %237, <4 x float> zeroinitializer) #8, !dbg !14
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %239, <4 x float> %253) #8, !dbg !14
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %241, <4 x float> zeroinitializer) #8, !dbg !14
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %244, <4 x float> %255) #8, !dbg !14
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %247, <4 x float> zeroinitializer) #8, !dbg !14
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %248, <4 x float> %257) #8, !dbg !14
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %242, <4 x float> zeroinitializer) #8, !dbg !14
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %245, <4 x float> %259) #8, !dbg !14
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %219, <4 x float> zeroinitializer) #8, !dbg !14
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %221, <4 x float> %261) #8, !dbg !14
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %232, <4 x float> zeroinitializer) #8, !dbg !14
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %234, <4 x float> %263) #8, !dbg !14
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !14
  %266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %238, <4 x float> %265) #8, !dbg !14
  %267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %233, <4 x float> zeroinitializer) #8, !dbg !14
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %235, <4 x float> %267) #8, !dbg !14
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %237, <4 x float> zeroinitializer) #8, !dbg !14
  %270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %239, <4 x float> %269) #8, !dbg !14
  %271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %241, <4 x float> zeroinitializer) #8, !dbg !14
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %244, <4 x float> %271) #8, !dbg !14
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %247, <4 x float> zeroinitializer) #8, !dbg !14
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %248, <4 x float> %273) #8, !dbg !14
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %242, <4 x float> zeroinitializer) #8, !dbg !14
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %245, <4 x float> %275) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %277 = add nuw nsw i64 %163, 2048, !dbg !13
  %278 = add nsw <2 x i64> %162, <i64 64, i64 2048>, !dbg !13
  %.lobit38 = ashr i64 %277, 63, !dbg !13
  %279 = icmp slt <2 x i64> %162, <i64 -64, i64 -2048>, !dbg !13
  %280 = icmp slt i64 %164, -64, !dbg !13
  %281 = sub nsw <2 x i64> <i64 -65, i64 -2049>, %162, !dbg !13
  %282 = select <2 x i1> %279, <2 x i64> %281, <2 x i64> %278, !dbg !13
  %283 = sdiv <2 x i64> %282, <i64 128, i64 4096>, !dbg !13
  %284 = sext i1 %280 to i64, !dbg !13
  %285 = insertelement <2 x i64> poison, i64 %284, i64 0, !dbg !13
  %286 = insertelement <2 x i64> %285, i64 %.lobit38, i64 1, !dbg !13
  %287 = xor <2 x i64> %283, %286, !dbg !13
  %288 = shl <2 x i64> %287, <i64 7, i64 19>, !dbg !13
  %289 = extractelement <2 x i64> %288, i64 0, !dbg !13
  %290 = sub i64 %159, %289, !dbg !13
  %291 = extractelement <2 x i64> %288, i64 1, !dbg !13
  %292 = and i64 %291, 16252928, !dbg !13
  %293 = add i64 %290, %292, !dbg !13
  %kdma.source1747 = add i64 %293, %kdma.source1612, !dbg !13
  %kdma.source1748 = trunc i64 %kdma.source1747 to i32, !dbg !13
  %kdma.source1749 = shl i32 %kdma.source1748, 1, !dbg !13
  %kdma.offset1751 = add i32 %kdma.source1750, %kdma.source1749, !dbg !13
  %294 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %181, i32 %kdma.offset1751, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %kdma.source1764 = add i64 %293, %kdma.source1626, !dbg !13
  %kdma.source1765 = trunc i64 %kdma.source1764 to i32, !dbg !13
  %kdma.source1766 = shl i32 %kdma.source1765, 1, !dbg !13
  %kdma.offset1768.reass.reass = add i32 %kdma.source1766, %invariant.op77
  %295 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1631, i32 %kdma.offset1768.reass.reass, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %kdma.source1782 = add i64 %293, %kdma.source1643, !dbg !13
  %kdma.source1783 = trunc i64 %kdma.source1782 to i32, !dbg !13
  %kdma.source1784 = shl i32 %kdma.source1783, 1, !dbg !13
  %kdma.offset1786 = add i32 %kdma.source1750, %kdma.source1784, !dbg !13
  %296 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %185, i32 %kdma.offset1786, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %kdma.source1800 = add i64 %293, %kdma.source1660, !dbg !13
  %kdma.source1801 = trunc i64 %kdma.source1800 to i32, !dbg !13
  %kdma.source1802 = shl i32 %kdma.source1801, 1, !dbg !13
  %kdma.offset1804.reass.reass = add i32 %kdma.source1802, %invariant.op78
  %297 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1665, i32 %kdma.offset1804.reass.reass, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %kdma.gep1906 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %298 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !14, !frisk.k_lds_pair !23
  %kdma.gep1910 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %299 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !14, !frisk.k_lds_pair !23
  %kdma.gep1907 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %300 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !14, !frisk.k_lds_pair !24
  %kdma.gep1911 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %301 = load <4 x half>, ptr addrspace(3) %kdma.gep1911, align 8, !dbg !14, !frisk.k_lds_pair !24
  %kdma.gep1908 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 8192, !dbg !14
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !14, !frisk.k_lds_pair !25
  %kdma.gep1912 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 8320, !dbg !14
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep1912, align 8, !dbg !14, !frisk.k_lds_pair !25
  %kdma.gep1909 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 12288, !dbg !14
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !14, !frisk.k_lds_pair !26
  %kdma.gep1913 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 12416, !dbg !14
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep1913, align 8, !dbg !14, !frisk.k_lds_pair !26
  %kdma.gep1914 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %306 = load <4 x half>, ptr addrspace(3) %kdma.gep1914, align 8, !dbg !14, !frisk.k_lds_pair !27
  %kdma.gep1918 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1918, align 8, !dbg !14, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %298, <4 x float> %240) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep1915 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep1915, align 8, !dbg !14, !frisk.k_lds_pair !28
  %kdma.gep1919 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1919, align 8, !dbg !14, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %300, <4 x float> %308) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep1916 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 9216, !dbg !14
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1916, align 8, !dbg !14, !frisk.k_lds_pair !29
  %kdma.gep1920 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 9344, !dbg !14
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1920, align 8, !dbg !14, !frisk.k_lds_pair !29
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %302, <4 x float> %246) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep1917 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 13312, !dbg !14
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1917, align 8, !dbg !14, !frisk.k_lds_pair !30
  %kdma.gep1921 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 13440, !dbg !14
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1921, align 8, !dbg !14, !frisk.k_lds_pair !30
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %304, <4 x float> %314) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %299, <4 x float> %250) #8, !dbg !14
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %301, <4 x float> %318) #8, !dbg !14
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %303, <4 x float> %252) #8, !dbg !14
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %305, <4 x float> %320) #8, !dbg !14
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %306, <4 x float> %254) #8, !dbg !14
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %309, <4 x float> %322) #8, !dbg !14
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %312, <4 x float> %256) #8, !dbg !14
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %315, <4 x float> %324) #8, !dbg !14
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %307, <4 x float> %258) #8, !dbg !14
  %327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %310, <4 x float> %326) #8, !dbg !14
  %328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %313, <4 x float> %260) #8, !dbg !14
  %329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %316, <4 x float> %328) #8, !dbg !14
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %298, <4 x float> %262) #8, !dbg !14
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %300, <4 x float> %330) #8, !dbg !14
  %332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %302, <4 x float> %264) #8, !dbg !14
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %304, <4 x float> %332) #8, !dbg !14
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %299, <4 x float> %266) #8, !dbg !14
  %335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %301, <4 x float> %334) #8, !dbg !14
  %336 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %303, <4 x float> %268) #8, !dbg !14
  %337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %305, <4 x float> %336) #8, !dbg !14
  %338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %306, <4 x float> %270) #8, !dbg !14
  %339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %309, <4 x float> %338) #8, !dbg !14
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %312, <4 x float> %272) #8, !dbg !14
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %315, <4 x float> %340) #8, !dbg !14
  %342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %307, <4 x float> %274) #8, !dbg !14
  %343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %310, <4 x float> %342) #8, !dbg !14
  %344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %313, <4 x float> %276) #8, !dbg !14
  %345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %316, <4 x float> %344) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %346 = add nuw nsw i64 %163, 3072, !dbg !13
  %347 = add nsw <2 x i64> %162, <i64 96, i64 3072>, !dbg !13
  %.lobit39 = ashr i64 %346, 63, !dbg !13
  %348 = icmp slt <2 x i64> %162, <i64 -96, i64 -3072>, !dbg !13
  %349 = icmp slt i64 %164, -96, !dbg !13
  %350 = sub nsw <2 x i64> <i64 -97, i64 -3073>, %162, !dbg !13
  %351 = select <2 x i1> %348, <2 x i64> %350, <2 x i64> %347, !dbg !13
  %352 = sdiv <2 x i64> %351, <i64 128, i64 4096>, !dbg !13
  %353 = sext i1 %349 to i64, !dbg !13
  %354 = insertelement <2 x i64> poison, i64 %353, i64 0, !dbg !13
  %355 = insertelement <2 x i64> %354, i64 %.lobit39, i64 1, !dbg !13
  %356 = xor <2 x i64> %352, %355, !dbg !13
  %357 = shl <2 x i64> %356, <i64 7, i64 19>, !dbg !13
  %358 = extractelement <2 x i64> %357, i64 0, !dbg !13
  %359 = sub i64 %159, %358, !dbg !13
  %360 = extractelement <2 x i64> %357, i64 1, !dbg !13
  %361 = and i64 %360, 16252928, !dbg !13
  %362 = add i64 %359, %361, !dbg !13
  %kdma.source1817 = add i64 %362, %kdma.source1612, !dbg !13
  %kdma.source1818 = trunc i64 %kdma.source1817 to i32, !dbg !13
  %kdma.source1819 = shl i32 %kdma.source1818, 1, !dbg !13
  %kdma.offset1821 = add i32 %kdma.source1820, %kdma.source1819, !dbg !13
  %363 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %189, i32 %kdma.offset1821, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %kdma.source1834 = add i64 %362, %kdma.source1626, !dbg !13
  %kdma.source1835 = trunc i64 %kdma.source1834 to i32, !dbg !13
  %kdma.source1836 = shl i32 %kdma.source1835, 1, !dbg !13
  %kdma.offset1838.reass.reass = add i32 %kdma.source1836, %invariant.op79
  %364 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1699, i32 %kdma.offset1838.reass.reass, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %kdma.source1852 = add i64 %362, %kdma.source1643, !dbg !13
  %kdma.source1853 = trunc i64 %kdma.source1852 to i32, !dbg !13
  %kdma.source1854 = shl i32 %kdma.source1853, 1, !dbg !13
  %kdma.offset1856 = add i32 %kdma.source1820, %kdma.source1854, !dbg !13
  %365 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %201, i32 %kdma.offset1856, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %kdma.source1870 = add i64 %362, %kdma.source1660, !dbg !13
  %kdma.source1871 = trunc i64 %kdma.source1870 to i32, !dbg !13
  %kdma.source1872 = shl i32 %kdma.source1871, 1, !dbg !13
  %kdma.offset1874.reass.reass = add i32 %kdma.source1872, %invariant.op80
  %366 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1735, i32 %kdma.offset1874.reass.reass, <4 x i32> %dma.buffer.rsrc1430) #9, !dbg !13
  %367 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14, !frisk.k_lds_pair !31
  %368 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !14, !frisk.k_lds_pair !31
  %369 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14, !frisk.k_lds_pair !32
  %370 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !14, !frisk.k_lds_pair !32
  %371 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1876, align 8, !dbg !14, !frisk.k_lds_pair !33
  %372 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !14, !frisk.k_lds_pair !33
  %373 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !14, !frisk.k_lds_pair !34
  %374 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !14, !frisk.k_lds_pair !34
  %375 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !14, !frisk.k_lds_pair !35
  %376 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !14, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %367, <4 x float> %311) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %378 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !14, !frisk.k_lds_pair !36
  %379 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !14, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %369, <4 x float> %377) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %381 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !14, !frisk.k_lds_pair !37
  %382 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !14, !frisk.k_lds_pair !37
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %371, <4 x float> %317) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %384 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !14, !frisk.k_lds_pair !38
  %385 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !14, !frisk.k_lds_pair !38
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %373, <4 x float> %383) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %368, <4 x float> %319) #8, !dbg !14
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %370, <4 x float> %387) #8, !dbg !14
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %372, <4 x float> %321) #8, !dbg !14
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %374, <4 x float> %389) #8, !dbg !14
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %375, <4 x float> %323) #8, !dbg !14
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %378, <4 x float> %391) #8, !dbg !14
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %381, <4 x float> %325) #8, !dbg !14
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %384, <4 x float> %393) #8, !dbg !14
  %395 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %376, <4 x float> %327) #8, !dbg !14
  %396 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %379, <4 x float> %395) #8, !dbg !14
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %382, <4 x float> %329) #8, !dbg !14
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %385, <4 x float> %397) #8, !dbg !14
  %399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %367, <4 x float> %331) #8, !dbg !14
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %369, <4 x float> %399) #8, !dbg !14
  %401 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %371, <4 x float> %333) #8, !dbg !14
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %373, <4 x float> %401) #8, !dbg !14
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %368, <4 x float> %335) #8, !dbg !14
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %370, <4 x float> %403) #8, !dbg !14
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %372, <4 x float> %337) #8, !dbg !14
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %374, <4 x float> %405) #8, !dbg !14
  %407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %375, <4 x float> %339) #8, !dbg !14
  %408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %378, <4 x float> %407) #8, !dbg !14
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %381, <4 x float> %341) #8, !dbg !14
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %384, <4 x float> %409) #8, !dbg !14
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %376, <4 x float> %343) #8, !dbg !14
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %379, <4 x float> %411) #8, !dbg !14
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %382, <4 x float> %345) #8, !dbg !14
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %385, <4 x float> %413) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %415 = add nuw nsw i64 %150, %17, !dbg !39
  %416 = shl i64 %415, 19, !dbg !39
  %417 = and i64 %416, 16252928, !dbg !39
  %address.local984 = shl nuw nsw i64 %address.local566, 5, !dbg !39
  %address.local985 = and i64 %address.local984, 8064, !dbg !39
  %address.local987 = and i64 %address.local567, 24, !dbg !39
  %address.local988 = or disjoint i64 %address.local985, %address.local987, !dbg !39
  %.idx15081510 = add i64 %417, %157, !dbg !39
  %418 = or disjoint i64 %address.local988, %.idx15081510, !dbg !39
  %.tr1511 = trunc i64 %418 to i32, !dbg !39
  %dma.buffer.offset1413 = shl i32 %.tr1511, 1, !dbg !39
  %419 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %181, i32 %dma.buffer.offset1413, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %address.local990 = shl nuw nsw i64 %address.local583, 5, !dbg !39
  %address.local991 = and i64 %address.local990, 16256, !dbg !39
  %address.local994 = or disjoint i64 %address.local991, %address.local987, !dbg !39
  %420 = or disjoint i64 %address.local994, %.idx15081510, !dbg !39
  %.tr1515 = trunc i64 %420 to i32, !dbg !39
  %dma.buffer.offset1411 = shl i32 %.tr1515, 1, !dbg !39
  %421 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %185, i32 %dma.buffer.offset1411, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %422 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !14, !frisk.k_tail_pair !16
  %423 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !14, !frisk.k_tail_pair !16
  %424 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !14, !frisk.k_tail_pair !17
  %425 = load <4 x half>, ptr addrspace(3) %kdma.gep1911, align 8, !dbg !14, !frisk.k_tail_pair !17
  %426 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !14, !frisk.k_tail_pair !18
  %427 = load <4 x half>, ptr addrspace(3) %kdma.gep1912, align 8, !dbg !14, !frisk.k_tail_pair !18
  %428 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !14, !frisk.k_tail_pair !19
  %429 = load <4 x half>, ptr addrspace(3) %kdma.gep1913, align 8, !dbg !14, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %422, <4 x float> %380) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %431 = or disjoint i64 %152, %115, !dbg !15
  %432 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %424, <4 x float> %430) #8, !dbg !14
  %433 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %426, <4 x float> %386) #8, !dbg !14
  %434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %428, <4 x float> %433) #8, !dbg !14
  %435 = load <4 x half>, ptr addrspace(3) %kdma.gep1914, align 8, !dbg !14, !frisk.k_tail_pair !20
  %436 = load <4 x half>, ptr addrspace(3) %kdma.gep1918, align 8, !dbg !14, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %423, <4 x float> %388) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !40, !frisk.k_tail_prefetch !5
  %438 = load <4 x half>, ptr addrspace(3) %kdma.gep1915, align 8, !dbg !14, !frisk.k_tail_pair !21
  %439 = load <4 x half>, ptr addrspace(3) %kdma.gep1919, align 8, !dbg !14, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %425, <4 x float> %437) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %441 = load <4 x half>, ptr addrspace(3) %kdma.gep1916, align 8, !dbg !14, !frisk.k_tail_pair !22
  %442 = load <4 x half>, ptr addrspace(3) %kdma.gep1920, align 8, !dbg !14, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %427, <4 x float> %390) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !40, !frisk.k_tail_prefetch !5
  %444 = load <4 x half>, ptr addrspace(3) %kdma.gep1917, align 8, !dbg !14, !frisk.k_tail_pair !23
  %445 = load <4 x half>, ptr addrspace(3) %kdma.gep1921, align 8, !dbg !14, !frisk.k_tail_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %429, <4 x float> %443) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %435, <4 x float> %392) #8, !dbg !14
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %438, <4 x float> %447) #8, !dbg !14
  %449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %441, <4 x float> %394) #8, !dbg !14
  %450 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %444, <4 x float> %449) #8, !dbg !14
  %451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %436, <4 x float> %396) #8, !dbg !14
  %452 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %439, <4 x float> %451) #8, !dbg !14
  %453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %442, <4 x float> %398) #8, !dbg !14
  %454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %445, <4 x float> %453) #8, !dbg !14
  %455 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %422, <4 x float> %400) #8, !dbg !14
  %456 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %424, <4 x float> %455) #8, !dbg !14
  %457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %426, <4 x float> %402) #8, !dbg !14
  %458 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %428, <4 x float> %457) #8, !dbg !14
  %459 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %423, <4 x float> %404) #8, !dbg !14
  %460 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %425, <4 x float> %459) #8, !dbg !14
  %461 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %427, <4 x float> %406) #8, !dbg !14
  %462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %429, <4 x float> %461) #8, !dbg !14
  %463 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %435, <4 x float> %408) #8, !dbg !14
  %464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %438, <4 x float> %463) #8, !dbg !14
  %465 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %441, <4 x float> %410) #8, !dbg !14
  %466 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %444, <4 x float> %465) #8, !dbg !14
  %467 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %436, <4 x float> %412) #8, !dbg !14
  %468 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %439, <4 x float> %467) #8, !dbg !14
  %469 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %442, <4 x float> %414) #8, !dbg !14
  %470 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %445, <4 x float> %469) #8, !dbg !14
  %471 = trunc i64 %431 to i32, !dbg !15
  %472 = insertelement <32 x i32> poison, i32 %471, i64 0, !dbg !15
  %473 = shufflevector <32 x i32> %472, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !15
  %474 = or disjoint <32 x i32> %473, <i32 0, i32 4, i32 8, i32 12, i32 64, i32 68, i32 72, i32 76, i32 1, i32 5, i32 9, i32 13, i32 65, i32 69, i32 73, i32 77, i32 2, i32 6, i32 10, i32 14, i32 66, i32 70, i32 74, i32 78, i32 3, i32 7, i32 11, i32 15, i32 67, i32 71, i32 75, i32 79>, !dbg !15
  %475 = icmp ugt <32 x i32> %136, %474, !dbg !15
  %476 = insertelement <4 x i32> poison, i32 %471, i64 0, !dbg !15
  %477 = shufflevector <4 x i32> %476, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !15
  %478 = or disjoint <4 x i32> %477, <i32 65, i32 69, i32 73, i32 77>, !dbg !15
  %479 = insertelement <2 x i32> poison, i32 %471, i64 0, !dbg !15
  %480 = shufflevector <2 x i32> %479, <2 x i32> poison, <2 x i32> zeroinitializer, !dbg !15
  %481 = or disjoint <2 x i32> %480, <i32 9, i32 13>, !dbg !15
  %482 = or disjoint i32 %471, 5, !dbg !15
  %483 = or disjoint <4 x i32> %477, <i32 64, i32 68, i32 72, i32 76>, !dbg !15
  %484 = or disjoint <2 x i32> %480, <i32 8, i32 12>, !dbg !15
  %485 = or disjoint i32 %471, 4, !dbg !15
  %486 = insertelement <32 x i32> %137, i32 %471, i64 8, !dbg !15
  %487 = shufflevector <32 x i32> %486, <32 x i32> poison, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !15
  %488 = insertelement <32 x i32> %139, i32 %471, i64 0, !dbg !15
  %489 = insertelement <32 x i32> %488, i32 %485, i64 1, !dbg !15
  %490 = insertelement <32 x i32> %489, i32 %482, i64 9, !dbg !15
  %491 = shufflevector <32 x i32> %490, <32 x i32> %474, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !15
  %492 = shufflevector <4 x i32> %483, <4 x i32> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %493 = shufflevector <32 x i32> %491, <32 x i32> %492, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 32, i32 33, i32 34, i32 35, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !15
  %494 = shufflevector <4 x i32> %478, <4 x i32> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %495 = shufflevector <32 x i32> %493, <32 x i32> %494, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 poison, i32 poison, i32 32, i32 33, i32 34, i32 35, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !15
  %496 = shufflevector <2 x i32> %484, <2 x i32> poison, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %497 = shufflevector <32 x i32> %495, <32 x i32> %496, <32 x i32> <i32 0, i32 1, i32 32, i32 33, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 poison, i32 poison, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !15
  %498 = shufflevector <2 x i32> %481, <2 x i32> poison, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %499 = shufflevector <32 x i32> %497, <32 x i32> %498, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 32, i32 33, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !15
  %500 = icmp ult <32 x i32> %487, %499, !dbg !15
  %501 = shufflevector <4 x float> %432, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %502 = shufflevector <32 x float> <float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float poison, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf, float -inf>, <32 x float> %501, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !15
  %503 = shufflevector <4 x float> %432, <4 x float> %434, <16 x i32> <i32 2, i32 6, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 3, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %504 = shufflevector <4 x float> %440, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %505 = shufflevector <16 x float> %503, <16 x float> %504, <16 x i32> <i32 0, i32 1, i32 18, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %506 = shufflevector <4 x float> %446, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %507 = shufflevector <16 x float> %505, <16 x float> %506, <16 x i32> <i32 0, i32 1, i32 2, i32 18, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %508 = shufflevector <4 x float> %448, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %509 = shufflevector <16 x float> %507, <16 x float> %508, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 18, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %510 = shufflevector <4 x float> %450, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %511 = shufflevector <16 x float> %509, <16 x float> %510, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 18, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %512 = shufflevector <4 x float> %452, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %513 = shufflevector <16 x float> %511, <16 x float> %512, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 18, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %514 = shufflevector <4 x float> %454, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %515 = shufflevector <16 x float> %513, <16 x float> %514, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 18, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %516 = shufflevector <4 x float> %440, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %517 = shufflevector <16 x float> %515, <16 x float> %516, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 19, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %518 = shufflevector <4 x float> %446, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %519 = shufflevector <16 x float> %517, <16 x float> %518, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %520 = shufflevector <4 x float> %448, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %521 = shufflevector <16 x float> %519, <16 x float> %520, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 19, i32 poison, i32 poison, i32 poison>, !dbg !15
  %522 = shufflevector <4 x float> %450, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %523 = shufflevector <16 x float> %521, <16 x float> %522, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 19, i32 poison, i32 poison>, !dbg !15
  %524 = shufflevector <4 x float> %452, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %525 = shufflevector <16 x float> %523, <16 x float> %524, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 19, i32 poison>, !dbg !15
  %526 = shufflevector <4 x float> %454, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %527 = shufflevector <4 x float> %432, <4 x float> %434, <8 x i32> <i32 0, i32 4, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %528 = shufflevector <4 x float> %440, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %529 = shufflevector <8 x float> %527, <8 x float> %528, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %530 = shufflevector <4 x float> %446, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %531 = shufflevector <8 x float> %529, <8 x float> %530, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %532 = shufflevector <4 x float> %448, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %533 = shufflevector <8 x float> %531, <8 x float> %532, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>, !dbg !15
  %534 = shufflevector <4 x float> %450, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %535 = shufflevector <8 x float> %533, <8 x float> %534, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>, !dbg !15
  %536 = shufflevector <4 x float> %452, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %537 = shufflevector <8 x float> %535, <8 x float> %536, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>, !dbg !15
  %538 = shufflevector <4 x float> %454, <4 x float> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %539 = shufflevector <4 x float> %448, <4 x float> %450, <4 x i32> <i32 1, i32 5, i32 poison, i32 poison>, !dbg !15
  %540 = shufflevector <4 x float> %539, <4 x float> %452, <4 x i32> <i32 0, i32 1, i32 5, i32 poison>, !dbg !15
  %541 = shufflevector <4 x float> %434, <4 x float> %440, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 5, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %542 = insertelement <32 x float> %541, float -inf, i64 8, !dbg !15
  %543 = shufflevector <4 x float> %446, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %544 = shufflevector <32 x float> %542, <32 x float> %543, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %545 = shufflevector <16 x float> %525, <16 x float> %526, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 19, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %546 = shufflevector <32 x float> %544, <32 x float> %545, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 poison, i32 poison, i32 poison, i32 poison, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47>, !dbg !15
  %547 = shufflevector <8 x float> %537, <8 x float> %538, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %548 = shufflevector <32 x float> %547, <32 x float> %546, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 poison, i32 poison, i32 poison, i32 poison, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !15
  %549 = shufflevector <4 x float> %540, <4 x float> %454, <32 x i32> <i32 0, i32 1, i32 2, i32 5, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %550 = shufflevector <32 x float> %548, <32 x float> %549, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !15
  %551 = select <32 x i1> %500, <32 x float> %502, <32 x float> %550, !dbg !15
  %552 = extractelement <32 x float> %551, i64 0, !dbg !41
  %553 = tail call float @llvm.maxnum.f32(float %552, float -inf), !dbg !41, !frisk.local_reduction !5
  %554 = extractelement <32 x float> %551, i64 1, !dbg !41
  %local.reduce218 = tail call float @llvm.maxnum.f32(float %553, float %554), !dbg !41, !frisk.local_reduction !5
  %555 = extractelement <32 x float> %551, i64 8, !dbg !41
  %556 = extractelement <32 x float> %551, i64 9, !dbg !41
  %local.reduce219 = tail call float @llvm.maxnum.f32(float %555, float %556), !dbg !41, !frisk.local_reduction !5
  %557 = extractelement <32 x float> %551, i64 16, !dbg !41
  %558 = extractelement <32 x float> %551, i64 17, !dbg !41
  %local.reduce220 = tail call float @llvm.maxnum.f32(float %557, float %558), !dbg !41, !frisk.local_reduction !5
  %559 = extractelement <32 x float> %551, i64 24, !dbg !41
  %560 = extractelement <32 x float> %551, i64 25, !dbg !41
  %local.reduce221 = tail call float @llvm.maxnum.f32(float %559, float %560), !dbg !41, !frisk.local_reduction !5
  %561 = extractelement <32 x float> %551, i64 2, !dbg !41
  %local.reduce222 = tail call float @llvm.maxnum.f32(float %local.reduce218, float %561), !dbg !41, !frisk.local_reduction !5
  %562 = extractelement <32 x float> %551, i64 10, !dbg !41
  %local.reduce223 = tail call float @llvm.maxnum.f32(float %local.reduce219, float %562), !dbg !41, !frisk.local_reduction !5
  %563 = extractelement <32 x float> %551, i64 18, !dbg !41
  %local.reduce224 = tail call float @llvm.maxnum.f32(float %local.reduce220, float %563), !dbg !41, !frisk.local_reduction !5
  %564 = extractelement <32 x float> %551, i64 26, !dbg !41
  %local.reduce225 = tail call float @llvm.maxnum.f32(float %local.reduce221, float %564), !dbg !41, !frisk.local_reduction !5
  %565 = extractelement <32 x float> %551, i64 3, !dbg !41
  %local.reduce226 = tail call float @llvm.maxnum.f32(float %local.reduce222, float %565), !dbg !41, !frisk.local_reduction !5
  %566 = extractelement <32 x float> %551, i64 11, !dbg !41
  %local.reduce227 = tail call float @llvm.maxnum.f32(float %local.reduce223, float %566), !dbg !41, !frisk.local_reduction !5
  %567 = extractelement <32 x float> %551, i64 19, !dbg !41
  %local.reduce228 = tail call float @llvm.maxnum.f32(float %local.reduce224, float %567), !dbg !41, !frisk.local_reduction !5
  %568 = extractelement <32 x float> %551, i64 27, !dbg !41
  %local.reduce229 = tail call float @llvm.maxnum.f32(float %local.reduce225, float %568), !dbg !41, !frisk.local_reduction !5
  %569 = extractelement <32 x float> %551, i64 4, !dbg !41
  %local.reduce230 = tail call float @llvm.maxnum.f32(float %local.reduce226, float %569), !dbg !41, !frisk.local_reduction !5
  %570 = extractelement <32 x float> %551, i64 12, !dbg !41
  %local.reduce231 = tail call float @llvm.maxnum.f32(float %local.reduce227, float %570), !dbg !41, !frisk.local_reduction !5
  %571 = extractelement <32 x float> %551, i64 20, !dbg !41
  %local.reduce232 = tail call float @llvm.maxnum.f32(float %local.reduce228, float %571), !dbg !41, !frisk.local_reduction !5
  %572 = extractelement <32 x float> %551, i64 28, !dbg !41
  %local.reduce233 = tail call float @llvm.maxnum.f32(float %local.reduce229, float %572), !dbg !41, !frisk.local_reduction !5
  %573 = extractelement <32 x float> %551, i64 5, !dbg !41
  %local.reduce234 = tail call float @llvm.maxnum.f32(float %local.reduce230, float %573), !dbg !41, !frisk.local_reduction !5
  %574 = extractelement <32 x float> %551, i64 13, !dbg !41
  %local.reduce235 = tail call float @llvm.maxnum.f32(float %local.reduce231, float %574), !dbg !41, !frisk.local_reduction !5
  %575 = extractelement <32 x float> %551, i64 21, !dbg !41
  %local.reduce236 = tail call float @llvm.maxnum.f32(float %local.reduce232, float %575), !dbg !41, !frisk.local_reduction !5
  %576 = extractelement <32 x float> %551, i64 29, !dbg !41
  %local.reduce237 = tail call float @llvm.maxnum.f32(float %local.reduce233, float %576), !dbg !41, !frisk.local_reduction !5
  %577 = extractelement <32 x float> %551, i64 6, !dbg !41
  %local.reduce238 = tail call float @llvm.maxnum.f32(float %local.reduce234, float %577), !dbg !41, !frisk.local_reduction !5
  %578 = extractelement <32 x float> %551, i64 14, !dbg !41
  %local.reduce239 = tail call float @llvm.maxnum.f32(float %local.reduce235, float %578), !dbg !41, !frisk.local_reduction !5
  %579 = extractelement <32 x float> %551, i64 22, !dbg !41
  %local.reduce240 = tail call float @llvm.maxnum.f32(float %local.reduce236, float %579), !dbg !41, !frisk.local_reduction !5
  %580 = extractelement <32 x float> %551, i64 30, !dbg !41
  %local.reduce241 = tail call float @llvm.maxnum.f32(float %local.reduce237, float %580), !dbg !41, !frisk.local_reduction !5
  %581 = extractelement <32 x float> %551, i64 7, !dbg !41
  %local.reduce242 = tail call float @llvm.maxnum.f32(float %local.reduce238, float %581), !dbg !41, !frisk.local_reduction !5
  %582 = extractelement <32 x float> %551, i64 15, !dbg !41
  %local.reduce243 = tail call float @llvm.maxnum.f32(float %local.reduce239, float %582), !dbg !41, !frisk.local_reduction !5
  %583 = extractelement <32 x float> %551, i64 23, !dbg !41
  %local.reduce244 = tail call float @llvm.maxnum.f32(float %local.reduce240, float %583), !dbg !41, !frisk.local_reduction !5
  %584 = extractelement <32 x float> %551, i64 31, !dbg !41
  %local.reduce245 = tail call float @llvm.maxnum.f32(float %local.reduce241, float %584), !dbg !41, !frisk.local_reduction !5
  %local.reduce246 = tail call float @llvm.maxnum.f32(float %local.reduce242, float %local.reduce243), !dbg !41, !frisk.local_reduction !5
  %local.reduce247 = tail call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !41, !frisk.local_reduction !5
  %585 = tail call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !41, !frisk.local_reduction !5
  %586 = shufflevector <4 x float> %456, <4 x float> %458, <32 x i32> <i32 0, i32 4, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 5, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 2, i32 6, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 3, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %587 = shufflevector <4 x float> %460, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %588 = shufflevector <32 x float> %586, <32 x float> %587, <32 x i32> <i32 0, i32 1, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %589 = shufflevector <4 x float> %462, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %590 = shufflevector <32 x float> %588, <32 x float> %589, <32 x i32> <i32 0, i32 1, i32 2, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %591 = shufflevector <4 x float> %464, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %592 = shufflevector <32 x float> %590, <32 x float> %591, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 32, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %593 = shufflevector <4 x float> %466, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %594 = shufflevector <32 x float> %592, <32 x float> %593, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 32, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %595 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %596 = shufflevector <32 x float> %594, <32 x float> %595, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 32, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %597 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %598 = shufflevector <32 x float> %596, <32 x float> %597, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 32, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %599 = shufflevector <4 x float> %460, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %600 = shufflevector <32 x float> %598, <32 x float> %599, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %601 = shufflevector <4 x float> %462, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %602 = shufflevector <32 x float> %600, <32 x float> %601, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %603 = shufflevector <4 x float> %464, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %604 = shufflevector <32 x float> %602, <32 x float> %603, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 33, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %605 = shufflevector <4 x float> %466, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %606 = shufflevector <32 x float> %604, <32 x float> %605, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 33, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %607 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %608 = shufflevector <32 x float> %606, <32 x float> %607, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 33, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %609 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %610 = shufflevector <32 x float> %608, <32 x float> %609, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 33, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %611 = shufflevector <4 x float> %460, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %612 = shufflevector <32 x float> %610, <32 x float> %611, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 34, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %613 = shufflevector <4 x float> %462, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %614 = shufflevector <32 x float> %612, <32 x float> %613, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 34, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %615 = shufflevector <4 x float> %464, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %616 = shufflevector <32 x float> %614, <32 x float> %615, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 34, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %617 = shufflevector <4 x float> %466, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %618 = shufflevector <32 x float> %616, <32 x float> %617, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 34, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %619 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %620 = shufflevector <32 x float> %618, <32 x float> %619, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 34, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %621 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %622 = shufflevector <32 x float> %620, <32 x float> %621, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 34, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %623 = shufflevector <4 x float> %460, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %624 = shufflevector <32 x float> %622, <32 x float> %623, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %625 = shufflevector <4 x float> %462, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %626 = shufflevector <32 x float> %624, <32 x float> %625, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %627 = shufflevector <4 x float> %464, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %628 = shufflevector <32 x float> %626, <32 x float> %627, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 35, i32 poison, i32 poison, i32 poison>, !dbg !15
  %629 = shufflevector <4 x float> %466, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %630 = shufflevector <32 x float> %628, <32 x float> %629, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 35, i32 poison, i32 poison>, !dbg !15
  %631 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %632 = shufflevector <32 x float> %630, <32 x float> %631, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 35, i32 poison>, !dbg !15
  %633 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %634 = shufflevector <32 x float> %632, <32 x float> %633, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 35>, !dbg !15
  %635 = select <32 x i1> %475, <32 x float> %634, <32 x float> splat (float -inf), !dbg !15
  %636 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !41
  %637 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %636), !dbg !41
  %638 = add nuw nsw i32 %637, 64, !dbg !41
  %639 = xor i32 %637, 16, !dbg !41
  %640 = and i32 %638, 192, !dbg !41
  %641 = icmp samesign ult i32 %639, %640, !dbg !41
  %642 = select i1 %641, i32 %639, i32 %637, !dbg !41
  %643 = shl nuw nsw i32 %642, 2, !dbg !41
  %644 = bitcast float %585 to i32, !dbg !41
  %645 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %643, i32 %644), !dbg !41
  %646 = bitcast i32 %645 to float, !dbg !41
  %647 = tail call float @llvm.maxnum.f32(float %585, float %646), !dbg !41
  %648 = xor i32 %637, 32, !dbg !41
  %649 = icmp samesign ult i32 %648, %640, !dbg !41
  %650 = select i1 %649, i32 %648, i32 %637, !dbg !41
  %651 = shl nuw nsw i32 %650, 2, !dbg !41
  %652 = bitcast float %647 to i32, !dbg !41
  %653 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %651, i32 %652), !dbg !41
  %654 = bitcast i32 %653 to float, !dbg !41
  %655 = tail call float @llvm.maxnum.f32(float %647, float %654), !dbg !41
  %656 = extractelement <32 x float> %635, i64 0, !dbg !41
  %657 = tail call float @llvm.maxnum.f32(float %656, float -inf), !dbg !41, !frisk.local_reduction !5
  %658 = extractelement <32 x float> %635, i64 1, !dbg !41
  %local.reduce187 = tail call float @llvm.maxnum.f32(float %657, float %658), !dbg !41, !frisk.local_reduction !5
  %659 = extractelement <32 x float> %635, i64 8, !dbg !41
  %660 = extractelement <32 x float> %635, i64 9, !dbg !41
  %local.reduce188 = tail call float @llvm.maxnum.f32(float %659, float %660), !dbg !41, !frisk.local_reduction !5
  %661 = extractelement <32 x float> %635, i64 16, !dbg !41
  %662 = extractelement <32 x float> %635, i64 17, !dbg !41
  %local.reduce189 = tail call float @llvm.maxnum.f32(float %661, float %662), !dbg !41, !frisk.local_reduction !5
  %663 = extractelement <32 x float> %635, i64 24, !dbg !41
  %664 = extractelement <32 x float> %635, i64 25, !dbg !41
  %local.reduce190 = tail call float @llvm.maxnum.f32(float %663, float %664), !dbg !41, !frisk.local_reduction !5
  %665 = extractelement <32 x float> %635, i64 2, !dbg !41
  %local.reduce191 = tail call float @llvm.maxnum.f32(float %local.reduce187, float %665), !dbg !41, !frisk.local_reduction !5
  %666 = extractelement <32 x float> %635, i64 10, !dbg !41
  %local.reduce192 = tail call float @llvm.maxnum.f32(float %local.reduce188, float %666), !dbg !41, !frisk.local_reduction !5
  %667 = extractelement <32 x float> %635, i64 18, !dbg !41
  %local.reduce193 = tail call float @llvm.maxnum.f32(float %local.reduce189, float %667), !dbg !41, !frisk.local_reduction !5
  %668 = extractelement <32 x float> %635, i64 26, !dbg !41
  %local.reduce194 = tail call float @llvm.maxnum.f32(float %local.reduce190, float %668), !dbg !41, !frisk.local_reduction !5
  %669 = extractelement <32 x float> %635, i64 3, !dbg !41
  %local.reduce195 = tail call float @llvm.maxnum.f32(float %local.reduce191, float %669), !dbg !41, !frisk.local_reduction !5
  %670 = extractelement <32 x float> %635, i64 11, !dbg !41
  %local.reduce196 = tail call float @llvm.maxnum.f32(float %local.reduce192, float %670), !dbg !41, !frisk.local_reduction !5
  %671 = extractelement <32 x float> %635, i64 19, !dbg !41
  %local.reduce197 = tail call float @llvm.maxnum.f32(float %local.reduce193, float %671), !dbg !41, !frisk.local_reduction !5
  %672 = extractelement <32 x float> %635, i64 27, !dbg !41
  %local.reduce198 = tail call float @llvm.maxnum.f32(float %local.reduce194, float %672), !dbg !41, !frisk.local_reduction !5
  %673 = extractelement <32 x float> %635, i64 4, !dbg !41
  %local.reduce199 = tail call float @llvm.maxnum.f32(float %local.reduce195, float %673), !dbg !41, !frisk.local_reduction !5
  %674 = extractelement <32 x float> %635, i64 12, !dbg !41
  %local.reduce200 = tail call float @llvm.maxnum.f32(float %local.reduce196, float %674), !dbg !41, !frisk.local_reduction !5
  %675 = extractelement <32 x float> %635, i64 20, !dbg !41
  %local.reduce201 = tail call float @llvm.maxnum.f32(float %local.reduce197, float %675), !dbg !41, !frisk.local_reduction !5
  %676 = extractelement <32 x float> %635, i64 28, !dbg !41
  %local.reduce202 = tail call float @llvm.maxnum.f32(float %local.reduce198, float %676), !dbg !41, !frisk.local_reduction !5
  %677 = extractelement <32 x float> %635, i64 5, !dbg !41
  %local.reduce203 = tail call float @llvm.maxnum.f32(float %local.reduce199, float %677), !dbg !41, !frisk.local_reduction !5
  %678 = extractelement <32 x float> %635, i64 13, !dbg !41
  %local.reduce204 = tail call float @llvm.maxnum.f32(float %local.reduce200, float %678), !dbg !41, !frisk.local_reduction !5
  %679 = extractelement <32 x float> %635, i64 21, !dbg !41
  %local.reduce205 = tail call float @llvm.maxnum.f32(float %local.reduce201, float %679), !dbg !41, !frisk.local_reduction !5
  %680 = extractelement <32 x float> %635, i64 29, !dbg !41
  %local.reduce206 = tail call float @llvm.maxnum.f32(float %local.reduce202, float %680), !dbg !41, !frisk.local_reduction !5
  %681 = extractelement <32 x float> %635, i64 6, !dbg !41
  %local.reduce207 = tail call float @llvm.maxnum.f32(float %local.reduce203, float %681), !dbg !41, !frisk.local_reduction !5
  %682 = extractelement <32 x float> %635, i64 14, !dbg !41
  %local.reduce208 = tail call float @llvm.maxnum.f32(float %local.reduce204, float %682), !dbg !41, !frisk.local_reduction !5
  %683 = extractelement <32 x float> %635, i64 22, !dbg !41
  %local.reduce209 = tail call float @llvm.maxnum.f32(float %local.reduce205, float %683), !dbg !41, !frisk.local_reduction !5
  %684 = extractelement <32 x float> %635, i64 30, !dbg !41
  %local.reduce210 = tail call float @llvm.maxnum.f32(float %local.reduce206, float %684), !dbg !41, !frisk.local_reduction !5
  %685 = extractelement <32 x float> %635, i64 7, !dbg !41
  %local.reduce211 = tail call float @llvm.maxnum.f32(float %local.reduce207, float %685), !dbg !41, !frisk.local_reduction !5
  %686 = extractelement <32 x float> %635, i64 15, !dbg !41
  %local.reduce212 = tail call float @llvm.maxnum.f32(float %local.reduce208, float %686), !dbg !41, !frisk.local_reduction !5
  %687 = extractelement <32 x float> %635, i64 23, !dbg !41
  %local.reduce213 = tail call float @llvm.maxnum.f32(float %local.reduce209, float %687), !dbg !41, !frisk.local_reduction !5
  %688 = extractelement <32 x float> %635, i64 31, !dbg !41
  %local.reduce214 = tail call float @llvm.maxnum.f32(float %local.reduce210, float %688), !dbg !41, !frisk.local_reduction !5
  %local.reduce215 = tail call float @llvm.maxnum.f32(float %local.reduce211, float %local.reduce212), !dbg !41, !frisk.local_reduction !5
  %local.reduce216 = tail call float @llvm.maxnum.f32(float %local.reduce213, float %local.reduce214), !dbg !41, !frisk.local_reduction !5
  %689 = tail call float @llvm.maxnum.f32(float %local.reduce215, float %local.reduce216), !dbg !41, !frisk.local_reduction !5
  %690 = bitcast float %689 to i32, !dbg !41
  %691 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %643, i32 %690), !dbg !41
  %692 = bitcast i32 %691 to float, !dbg !41
  %693 = tail call float @llvm.maxnum.f32(float %689, float %692), !dbg !41
  %694 = bitcast float %693 to i32, !dbg !41
  %695 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %651, i32 %694), !dbg !41
  %696 = bitcast i32 %695 to float, !dbg !41
  %697 = tail call float @llvm.maxnum.f32(float %693, float %696), !dbg !41
  %698 = extractvalue [2 x <1 x float>] %144, 0, !dbg !41
  %699 = insertelement <1 x float> poison, float %655, i64 0, !dbg !41
  %700 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %699, <1 x float> %698), !dbg !41
  %701 = extractvalue [2 x <1 x float>] %144, 1, !dbg !41
  %702 = insertelement <1 x float> poison, float %697, i64 0, !dbg !41
  %703 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %702, <1 x float> %701), !dbg !41
  %704 = shufflevector <32 x float> %551, <32 x float> poison, <4 x i32> <i32 0, i32 8, i32 16, i32 24>, !dbg !42
  %705 = shufflevector <1 x float> %700, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %706 = shufflevector <32 x float> %551, <32 x float> poison, <4 x i32> <i32 1, i32 9, i32 17, i32 25>, !dbg !42
  %707 = shufflevector <32 x float> %551, <32 x float> poison, <4 x i32> <i32 2, i32 10, i32 18, i32 26>, !dbg !42
  %708 = shufflevector <32 x float> %551, <32 x float> poison, <4 x i32> <i32 3, i32 11, i32 19, i32 27>, !dbg !42
  %709 = shufflevector <32 x float> %551, <32 x float> poison, <4 x i32> <i32 4, i32 12, i32 20, i32 28>, !dbg !42
  %710 = shufflevector <32 x float> %551, <32 x float> poison, <4 x i32> <i32 5, i32 13, i32 21, i32 29>, !dbg !42
  %711 = shufflevector <32 x float> %551, <32 x float> poison, <4 x i32> <i32 6, i32 14, i32 22, i32 30>, !dbg !42
  %712 = shufflevector <32 x float> %551, <32 x float> poison, <4 x i32> <i32 7, i32 15, i32 23, i32 31>, !dbg !42
  %713 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 0, i32 8, i32 16, i32 24>, !dbg !42
  %714 = shufflevector <1 x float> %703, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %715 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 1, i32 9, i32 17, i32 25>, !dbg !42
  %716 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 2, i32 10, i32 18, i32 26>, !dbg !42
  %717 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 3, i32 11, i32 19, i32 27>, !dbg !42
  %718 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 4, i32 12, i32 20, i32 28>, !dbg !42
  %719 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 5, i32 13, i32 21, i32 29>, !dbg !42
  %720 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 6, i32 14, i32 22, i32 30>, !dbg !42
  %721 = shufflevector <32 x float> %635, <32 x float> poison, <4 x i32> <i32 7, i32 15, i32 23, i32 31>, !dbg !42
  %722 = fsub <1 x float> %698, %700, !dbg !43
  %723 = extractelement <1 x float> %722, i64 0, !dbg !44
  %724 = tail call float @llvm.amdgcn.exp2.f32(float %723), !dbg !44
  %725 = insertelement <4 x float> poison, float %724, i64 0, !dbg !45
  %726 = shufflevector <4 x float> %725, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !45
  %727 = extractvalue [2 x <1 x float>] %145, 0, !dbg !46
  %728 = extractelement <1 x float> %727, i64 0, !dbg !46
  %729 = fmul float %728, %724, !dbg !46
  %730 = fsub <1 x float> %701, %703, !dbg !43
  %731 = extractelement <1 x float> %730, i64 0, !dbg !44
  %732 = tail call float @llvm.amdgcn.exp2.f32(float %731), !dbg !44
  %733 = insertelement <4 x float> poison, float %732, i64 0, !dbg !45
  %734 = shufflevector <4 x float> %733, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !45
  %735 = extractvalue [2 x <1 x float>] %145, 1, !dbg !46
  %736 = extractelement <1 x float> %735, i64 0, !dbg !46
  %737 = fmul float %736, %732, !dbg !46
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %vdb.address1532 = or disjoint i64 %.idx15081510, 32, !dbg !39
  %vdb.address1533 = or disjoint i64 %address.local988, %vdb.address1532, !dbg !39
  %vdb.address1534 = trunc i64 %vdb.address1533 to i32, !dbg !39
  %vdb.address1535 = shl i32 %vdb.address1534, 1, !dbg !39
  %vdb.address1542 = or disjoint i64 %address.local994, %vdb.address1532, !dbg !39
  %vdb.address1543 = trunc i64 %vdb.address1542 to i32, !dbg !39
  %vdb.address1544 = shl i32 %vdb.address1543, 1, !dbg !39
  %vdb.dma.base = add i32 %181, 8192, !dbg !47
  %738 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1535, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %vdb.dma.base1549 = add i32 %185, 8192, !dbg !47
  %739 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1549, i32 %vdb.address1544, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %address.local996 = shl nuw nsw i32 %address.thread565, 3, !dbg !47
  %address.local997 = and i32 %address.local996, 120, !dbg !47
  %address.local999 = and i32 %address.local642, 1536, !dbg !47
  %address.local1000 = or disjoint i32 %address.local997, %address.local999, !dbg !47
  %address.local1001 = zext nneg i32 %address.local1000 to i64, !dbg !47
  %address.local1002 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1001, !dbg !47
  %740 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %741 = bitcast <4 x i32> %740 to <8 x half>, !dbg !47
  %address.local1003 = and i64 %address.local566, 60, !dbg !47
  %address.local1004 = add nuw nsw i64 %address.local1003, %address.local566, !dbg !47
  %address.local1006.idx = shl nuw nsw i64 %address.local1004, 4, !dbg !47
  %address.local1006 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local1006.idx, !dbg !47
  %address.local1007 = lshr i64 %address.local622, 4, !dbg !47
  %address.local1009.idx = mul nuw nsw i64 %address.local1007, 768, !dbg !47
  %address.local1009 = getelementptr i8, ptr addrspace(3) %address.local1006, i64 %address.local1009.idx, !dbg !47
  %address.local1011.idx = mul nsw i64 %address.local609, -64, !dbg !47
  %address.local1011 = getelementptr i8, ptr addrspace(3) %address.local1009, i64 %address.local1011.idx, !dbg !47
  %address.local1012 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 256, !dbg !47
  %742 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %743 = bitcast <4 x i32> %742 to <8 x half>, !dbg !47
  %address.local1016 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 512, !dbg !47
  %744 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %745 = bitcast <4 x i32> %744 to <8 x half>, !dbg !47
  %address.local1020 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 768, !dbg !47
  %746 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %747 = bitcast <4 x i32> %746 to <8 x half>, !dbg !47
  %address.local1022 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 4096, !dbg !47
  %748 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %749 = bitcast <4 x i32> %748 to <8 x half>, !dbg !47
  %address.local1026 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4352, !dbg !47
  %750 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %751 = bitcast <4 x i32> %750 to <8 x half>, !dbg !47
  %address.local1030 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4608, !dbg !47
  %752 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %753 = bitcast <4 x i32> %752 to <8 x half>, !dbg !47
  %address.local1034 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4864, !dbg !47
  %754 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %755 = shufflevector <8 x half> %741, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %756 = extractvalue [2 x <8 x float>] %146, 0, !dbg !45
  %757 = shufflevector <8 x float> %756, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %758 = fmul <4 x float> %757, %726, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %759 = fsub <4 x float> %704, %705, !dbg !42
  %760 = extractelement <4 x float> %759, i64 0, !dbg !49
  %761 = tail call float @llvm.amdgcn.exp2.f32(float %760), !dbg !49
  %762 = insertelement <4 x float> poison, float %761, i64 0, !dbg !50
  %763 = extractelement <4 x float> %759, i64 1, !dbg !49
  %764 = tail call float @llvm.amdgcn.exp2.f32(float %763), !dbg !49
  %765 = insertelement <4 x float> %762, float %764, i64 1, !dbg !50
  %766 = extractelement <4 x float> %759, i64 2, !dbg !49
  %767 = tail call float @llvm.amdgcn.exp2.f32(float %766), !dbg !49
  %768 = insertelement <4 x float> %765, float %767, i64 2, !dbg !50
  %769 = extractelement <4 x float> %759, i64 3, !dbg !49
  %770 = tail call float @llvm.amdgcn.exp2.f32(float %769), !dbg !49
  %771 = insertelement <4 x float> %768, float %770, i64 3, !dbg !50
  %772 = fptrunc <4 x float> %771 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %773 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %772, <4 x half> %755, <4 x float> %758) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %774 = shufflevector <8 x half> %741, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %775 = shufflevector <8 x float> %756, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %776 = fmul <4 x float> %775, %726, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %777 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %772, <4 x half> %774, <4 x float> %776) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %778 = extractvalue [2 x <8 x float>] %146, 1, !dbg !45
  %779 = shufflevector <8 x float> %778, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %780 = fmul <4 x float> %779, %734, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %781 = fsub <4 x float> %713, %714, !dbg !42
  %782 = extractelement <4 x float> %781, i64 0, !dbg !49
  %783 = tail call float @llvm.amdgcn.exp2.f32(float %782), !dbg !49
  %784 = insertelement <4 x float> poison, float %783, i64 0, !dbg !50
  %785 = extractelement <4 x float> %781, i64 1, !dbg !49
  %786 = tail call float @llvm.amdgcn.exp2.f32(float %785), !dbg !49
  %787 = insertelement <4 x float> %784, float %786, i64 1, !dbg !50
  %788 = extractelement <4 x float> %781, i64 2, !dbg !49
  %789 = tail call float @llvm.amdgcn.exp2.f32(float %788), !dbg !49
  %790 = insertelement <4 x float> %787, float %789, i64 2, !dbg !50
  %791 = extractelement <4 x float> %781, i64 3, !dbg !49
  %792 = tail call float @llvm.amdgcn.exp2.f32(float %791), !dbg !49
  %793 = insertelement <4 x float> %790, float %792, i64 3, !dbg !50
  %794 = fptrunc <4 x float> %793 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %795 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %794, <4 x half> %755, <4 x float> %780) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %796 = shufflevector <8 x float> %778, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %797 = fmul <4 x float> %796, %734, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %798 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %794, <4 x half> %774, <4 x float> %797) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %799 = shufflevector <8 x half> %743, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %800 = fsub <4 x float> %706, %705, !dbg !42
  %801 = extractelement <4 x float> %800, i64 0, !dbg !49
  %802 = tail call float @llvm.amdgcn.exp2.f32(float %801), !dbg !49
  %local.reduce156 = fadd float %761, %802, !dbg !51, !frisk.local_reduction !5
  %803 = insertelement <4 x float> poison, float %802, i64 0, !dbg !50
  %804 = extractelement <4 x float> %800, i64 1, !dbg !49
  %805 = tail call float @llvm.amdgcn.exp2.f32(float %804), !dbg !49
  %local.reduce157 = fadd float %764, %805, !dbg !51, !frisk.local_reduction !5
  %806 = insertelement <4 x float> %803, float %805, i64 1, !dbg !50
  %807 = extractelement <4 x float> %800, i64 2, !dbg !49
  %808 = tail call float @llvm.amdgcn.exp2.f32(float %807), !dbg !49
  %local.reduce158 = fadd float %767, %808, !dbg !51, !frisk.local_reduction !5
  %809 = insertelement <4 x float> %806, float %808, i64 2, !dbg !50
  %810 = extractelement <4 x float> %800, i64 3, !dbg !49
  %811 = tail call float @llvm.amdgcn.exp2.f32(float %810), !dbg !49
  %local.reduce159 = fadd float %770, %811, !dbg !51, !frisk.local_reduction !5
  %812 = insertelement <4 x float> %809, float %811, i64 3, !dbg !50
  %813 = fptrunc <4 x float> %812 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %814 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %813, <4 x half> %799, <4 x float> %773) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %815 = shufflevector <8 x half> %743, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %813, <4 x half> %815, <4 x float> %777) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %817 = fsub <4 x float> %715, %714, !dbg !42
  %818 = extractelement <4 x float> %817, i64 0, !dbg !49
  %819 = tail call float @llvm.amdgcn.exp2.f32(float %818), !dbg !49
  %local.reduce = fadd float %783, %819, !dbg !51, !frisk.local_reduction !5
  %820 = insertelement <4 x float> poison, float %819, i64 0, !dbg !50
  %821 = extractelement <4 x float> %817, i64 1, !dbg !49
  %822 = tail call float @llvm.amdgcn.exp2.f32(float %821), !dbg !49
  %local.reduce126 = fadd float %786, %822, !dbg !51, !frisk.local_reduction !5
  %823 = insertelement <4 x float> %820, float %822, i64 1, !dbg !50
  %824 = extractelement <4 x float> %817, i64 2, !dbg !49
  %825 = tail call float @llvm.amdgcn.exp2.f32(float %824), !dbg !49
  %local.reduce127 = fadd float %789, %825, !dbg !51, !frisk.local_reduction !5
  %826 = insertelement <4 x float> %823, float %825, i64 2, !dbg !50
  %827 = extractelement <4 x float> %817, i64 3, !dbg !49
  %828 = tail call float @llvm.amdgcn.exp2.f32(float %827), !dbg !49
  %local.reduce128 = fadd float %792, %828, !dbg !51, !frisk.local_reduction !5
  %829 = insertelement <4 x float> %826, float %828, i64 3, !dbg !50
  %830 = fptrunc <4 x float> %829 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %831 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %830, <4 x half> %799, <4 x float> %795) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %830, <4 x half> %815, <4 x float> %798) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %833 = shufflevector <8 x half> %745, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %834 = fsub <4 x float> %707, %705, !dbg !42
  %835 = extractelement <4 x float> %834, i64 0, !dbg !49
  %836 = tail call float @llvm.amdgcn.exp2.f32(float %835), !dbg !49
  %local.reduce160 = fadd float %836, %local.reduce156, !dbg !51, !frisk.local_reduction !5
  %837 = insertelement <4 x float> poison, float %836, i64 0, !dbg !50
  %838 = extractelement <4 x float> %834, i64 1, !dbg !49
  %839 = tail call float @llvm.amdgcn.exp2.f32(float %838), !dbg !49
  %local.reduce161 = fadd float %839, %local.reduce157, !dbg !51, !frisk.local_reduction !5
  %840 = insertelement <4 x float> %837, float %839, i64 1, !dbg !50
  %841 = extractelement <4 x float> %834, i64 2, !dbg !49
  %842 = tail call float @llvm.amdgcn.exp2.f32(float %841), !dbg !49
  %local.reduce162 = fadd float %842, %local.reduce158, !dbg !51, !frisk.local_reduction !5
  %843 = insertelement <4 x float> %840, float %842, i64 2, !dbg !50
  %844 = extractelement <4 x float> %834, i64 3, !dbg !49
  %845 = tail call float @llvm.amdgcn.exp2.f32(float %844), !dbg !49
  %local.reduce163 = fadd float %845, %local.reduce159, !dbg !51, !frisk.local_reduction !5
  %846 = insertelement <4 x float> %843, float %845, i64 3, !dbg !50
  %847 = fptrunc <4 x float> %846 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %847, <4 x half> %833, <4 x float> %814) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %849 = shufflevector <8 x half> %745, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %847, <4 x half> %849, <4 x float> %816) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %851 = fsub <4 x float> %716, %714, !dbg !42
  %852 = extractelement <4 x float> %851, i64 0, !dbg !49
  %853 = tail call float @llvm.amdgcn.exp2.f32(float %852), !dbg !49
  %local.reduce129 = fadd float %853, %local.reduce, !dbg !51, !frisk.local_reduction !5
  %854 = insertelement <4 x float> poison, float %853, i64 0, !dbg !50
  %855 = extractelement <4 x float> %851, i64 1, !dbg !49
  %856 = tail call float @llvm.amdgcn.exp2.f32(float %855), !dbg !49
  %local.reduce130 = fadd float %856, %local.reduce126, !dbg !51, !frisk.local_reduction !5
  %857 = insertelement <4 x float> %854, float %856, i64 1, !dbg !50
  %858 = extractelement <4 x float> %851, i64 2, !dbg !49
  %859 = tail call float @llvm.amdgcn.exp2.f32(float %858), !dbg !49
  %local.reduce131 = fadd float %859, %local.reduce127, !dbg !51, !frisk.local_reduction !5
  %860 = insertelement <4 x float> %857, float %859, i64 2, !dbg !50
  %861 = extractelement <4 x float> %851, i64 3, !dbg !49
  %862 = tail call float @llvm.amdgcn.exp2.f32(float %861), !dbg !49
  %local.reduce132 = fadd float %862, %local.reduce128, !dbg !51, !frisk.local_reduction !5
  %863 = insertelement <4 x float> %860, float %862, i64 3, !dbg !50
  %864 = fptrunc <4 x float> %863 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %864, <4 x half> %833, <4 x float> %831) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %866 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %864, <4 x half> %849, <4 x float> %832) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %867 = shufflevector <8 x half> %747, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %868 = fsub <4 x float> %708, %705, !dbg !42
  %869 = extractelement <4 x float> %868, i64 0, !dbg !49
  %870 = tail call float @llvm.amdgcn.exp2.f32(float %869), !dbg !49
  %local.reduce164 = fadd float %870, %local.reduce160, !dbg !51, !frisk.local_reduction !5
  %871 = insertelement <4 x float> poison, float %870, i64 0, !dbg !50
  %872 = extractelement <4 x float> %868, i64 1, !dbg !49
  %873 = tail call float @llvm.amdgcn.exp2.f32(float %872), !dbg !49
  %local.reduce165 = fadd float %873, %local.reduce161, !dbg !51, !frisk.local_reduction !5
  %874 = insertelement <4 x float> %871, float %873, i64 1, !dbg !50
  %875 = extractelement <4 x float> %868, i64 2, !dbg !49
  %876 = tail call float @llvm.amdgcn.exp2.f32(float %875), !dbg !49
  %local.reduce166 = fadd float %876, %local.reduce162, !dbg !51, !frisk.local_reduction !5
  %877 = insertelement <4 x float> %874, float %876, i64 2, !dbg !50
  %878 = extractelement <4 x float> %868, i64 3, !dbg !49
  %879 = tail call float @llvm.amdgcn.exp2.f32(float %878), !dbg !49
  %local.reduce167 = fadd float %879, %local.reduce163, !dbg !51, !frisk.local_reduction !5
  %880 = insertelement <4 x float> %877, float %879, i64 3, !dbg !50
  %881 = fptrunc <4 x float> %880 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %882 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %881, <4 x half> %867, <4 x float> %848) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %883 = shufflevector <8 x half> %747, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %881, <4 x half> %883, <4 x float> %850) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %885 = fsub <4 x float> %717, %714, !dbg !42
  %886 = extractelement <4 x float> %885, i64 0, !dbg !49
  %887 = tail call float @llvm.amdgcn.exp2.f32(float %886), !dbg !49
  %local.reduce133 = fadd float %887, %local.reduce129, !dbg !51, !frisk.local_reduction !5
  %888 = insertelement <4 x float> poison, float %887, i64 0, !dbg !50
  %889 = extractelement <4 x float> %885, i64 1, !dbg !49
  %890 = tail call float @llvm.amdgcn.exp2.f32(float %889), !dbg !49
  %local.reduce134 = fadd float %890, %local.reduce130, !dbg !51, !frisk.local_reduction !5
  %891 = insertelement <4 x float> %888, float %890, i64 1, !dbg !50
  %892 = extractelement <4 x float> %885, i64 2, !dbg !49
  %893 = tail call float @llvm.amdgcn.exp2.f32(float %892), !dbg !49
  %local.reduce135 = fadd float %893, %local.reduce131, !dbg !51, !frisk.local_reduction !5
  %894 = insertelement <4 x float> %891, float %893, i64 2, !dbg !50
  %895 = extractelement <4 x float> %885, i64 3, !dbg !49
  %896 = tail call float @llvm.amdgcn.exp2.f32(float %895), !dbg !49
  %local.reduce136 = fadd float %896, %local.reduce132, !dbg !51, !frisk.local_reduction !5
  %897 = insertelement <4 x float> %894, float %896, i64 3, !dbg !50
  %898 = fptrunc <4 x float> %897 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %898, <4 x half> %867, <4 x float> %865) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %898, <4 x half> %883, <4 x float> %866) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %901 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %902 = fsub <4 x float> %709, %705, !dbg !42
  %903 = extractelement <4 x float> %902, i64 0, !dbg !49
  %904 = tail call float @llvm.amdgcn.exp2.f32(float %903), !dbg !49
  %local.reduce168 = fadd float %904, %local.reduce164, !dbg !51, !frisk.local_reduction !5
  %905 = insertelement <4 x float> poison, float %904, i64 0, !dbg !50
  %906 = extractelement <4 x float> %902, i64 1, !dbg !49
  %907 = tail call float @llvm.amdgcn.exp2.f32(float %906), !dbg !49
  %local.reduce169 = fadd float %907, %local.reduce165, !dbg !51, !frisk.local_reduction !5
  %908 = insertelement <4 x float> %905, float %907, i64 1, !dbg !50
  %909 = extractelement <4 x float> %902, i64 2, !dbg !49
  %910 = tail call float @llvm.amdgcn.exp2.f32(float %909), !dbg !49
  %local.reduce170 = fadd float %910, %local.reduce166, !dbg !51, !frisk.local_reduction !5
  %911 = insertelement <4 x float> %908, float %910, i64 2, !dbg !50
  %912 = extractelement <4 x float> %902, i64 3, !dbg !49
  %913 = tail call float @llvm.amdgcn.exp2.f32(float %912), !dbg !49
  %local.reduce171 = fadd float %913, %local.reduce167, !dbg !51, !frisk.local_reduction !5
  %914 = insertelement <4 x float> %911, float %913, i64 3, !dbg !50
  %915 = fptrunc <4 x float> %914 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %915, <4 x half> %901, <4 x float> %882) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %917 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %915, <4 x half> %917, <4 x float> %884) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %919 = fsub <4 x float> %718, %714, !dbg !42
  %920 = extractelement <4 x float> %919, i64 0, !dbg !49
  %921 = tail call float @llvm.amdgcn.exp2.f32(float %920), !dbg !49
  %local.reduce137 = fadd float %921, %local.reduce133, !dbg !51, !frisk.local_reduction !5
  %922 = insertelement <4 x float> poison, float %921, i64 0, !dbg !50
  %923 = extractelement <4 x float> %919, i64 1, !dbg !49
  %924 = tail call float @llvm.amdgcn.exp2.f32(float %923), !dbg !49
  %local.reduce138 = fadd float %924, %local.reduce134, !dbg !51, !frisk.local_reduction !5
  %925 = insertelement <4 x float> %922, float %924, i64 1, !dbg !50
  %926 = extractelement <4 x float> %919, i64 2, !dbg !49
  %927 = tail call float @llvm.amdgcn.exp2.f32(float %926), !dbg !49
  %local.reduce139 = fadd float %927, %local.reduce135, !dbg !51, !frisk.local_reduction !5
  %928 = insertelement <4 x float> %925, float %927, i64 2, !dbg !50
  %929 = extractelement <4 x float> %919, i64 3, !dbg !49
  %930 = tail call float @llvm.amdgcn.exp2.f32(float %929), !dbg !49
  %local.reduce140 = fadd float %930, %local.reduce136, !dbg !51, !frisk.local_reduction !5
  %931 = insertelement <4 x float> %928, float %930, i64 3, !dbg !50
  %932 = fptrunc <4 x float> %931 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %933 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %932, <4 x half> %901, <4 x float> %899) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %934 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %932, <4 x half> %917, <4 x float> %900) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %935 = shufflevector <8 x half> %751, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %936 = fsub <4 x float> %710, %705, !dbg !42
  %937 = extractelement <4 x float> %936, i64 0, !dbg !49
  %938 = tail call float @llvm.amdgcn.exp2.f32(float %937), !dbg !49
  %local.reduce172 = fadd float %938, %local.reduce168, !dbg !51, !frisk.local_reduction !5
  %939 = insertelement <4 x float> poison, float %938, i64 0, !dbg !50
  %940 = extractelement <4 x float> %936, i64 1, !dbg !49
  %941 = tail call float @llvm.amdgcn.exp2.f32(float %940), !dbg !49
  %local.reduce173 = fadd float %941, %local.reduce169, !dbg !51, !frisk.local_reduction !5
  %942 = insertelement <4 x float> %939, float %941, i64 1, !dbg !50
  %943 = extractelement <4 x float> %936, i64 2, !dbg !49
  %944 = tail call float @llvm.amdgcn.exp2.f32(float %943), !dbg !49
  %local.reduce174 = fadd float %944, %local.reduce170, !dbg !51, !frisk.local_reduction !5
  %945 = insertelement <4 x float> %942, float %944, i64 2, !dbg !50
  %946 = extractelement <4 x float> %936, i64 3, !dbg !49
  %947 = tail call float @llvm.amdgcn.exp2.f32(float %946), !dbg !49
  %local.reduce175 = fadd float %947, %local.reduce171, !dbg !51, !frisk.local_reduction !5
  %948 = insertelement <4 x float> %945, float %947, i64 3, !dbg !50
  %949 = fptrunc <4 x float> %948 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %950 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %949, <4 x half> %935, <4 x float> %916) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %951 = shufflevector <8 x half> %751, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %949, <4 x half> %951, <4 x float> %918) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %953 = fsub <4 x float> %719, %714, !dbg !42
  %954 = extractelement <4 x float> %953, i64 0, !dbg !49
  %955 = tail call float @llvm.amdgcn.exp2.f32(float %954), !dbg !49
  %local.reduce141 = fadd float %955, %local.reduce137, !dbg !51, !frisk.local_reduction !5
  %956 = insertelement <4 x float> poison, float %955, i64 0, !dbg !50
  %957 = extractelement <4 x float> %953, i64 1, !dbg !49
  %958 = tail call float @llvm.amdgcn.exp2.f32(float %957), !dbg !49
  %local.reduce142 = fadd float %958, %local.reduce138, !dbg !51, !frisk.local_reduction !5
  %959 = insertelement <4 x float> %956, float %958, i64 1, !dbg !50
  %960 = extractelement <4 x float> %953, i64 2, !dbg !49
  %961 = tail call float @llvm.amdgcn.exp2.f32(float %960), !dbg !49
  %local.reduce143 = fadd float %961, %local.reduce139, !dbg !51, !frisk.local_reduction !5
  %962 = insertelement <4 x float> %959, float %961, i64 2, !dbg !50
  %963 = extractelement <4 x float> %953, i64 3, !dbg !49
  %964 = tail call float @llvm.amdgcn.exp2.f32(float %963), !dbg !49
  %local.reduce144 = fadd float %964, %local.reduce140, !dbg !51, !frisk.local_reduction !5
  %965 = insertelement <4 x float> %962, float %964, i64 3, !dbg !50
  %966 = fptrunc <4 x float> %965 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %966, <4 x half> %935, <4 x float> %933) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %966, <4 x half> %951, <4 x float> %934) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %969 = shufflevector <8 x half> %753, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %970 = fsub <4 x float> %711, %705, !dbg !42
  %971 = extractelement <4 x float> %970, i64 0, !dbg !49
  %972 = tail call float @llvm.amdgcn.exp2.f32(float %971), !dbg !49
  %local.reduce176 = fadd float %972, %local.reduce172, !dbg !51, !frisk.local_reduction !5
  %973 = insertelement <4 x float> poison, float %972, i64 0, !dbg !50
  %974 = extractelement <4 x float> %970, i64 1, !dbg !49
  %975 = tail call float @llvm.amdgcn.exp2.f32(float %974), !dbg !49
  %local.reduce177 = fadd float %975, %local.reduce173, !dbg !51, !frisk.local_reduction !5
  %976 = insertelement <4 x float> %973, float %975, i64 1, !dbg !50
  %977 = extractelement <4 x float> %970, i64 2, !dbg !49
  %978 = tail call float @llvm.amdgcn.exp2.f32(float %977), !dbg !49
  %local.reduce178 = fadd float %978, %local.reduce174, !dbg !51, !frisk.local_reduction !5
  %979 = insertelement <4 x float> %976, float %978, i64 2, !dbg !50
  %980 = extractelement <4 x float> %970, i64 3, !dbg !49
  %981 = tail call float @llvm.amdgcn.exp2.f32(float %980), !dbg !49
  %local.reduce179 = fadd float %981, %local.reduce175, !dbg !51, !frisk.local_reduction !5
  %982 = insertelement <4 x float> %979, float %981, i64 3, !dbg !50
  %983 = fptrunc <4 x float> %982 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %983, <4 x half> %969, <4 x float> %950) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %985 = shufflevector <8 x half> %753, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %983, <4 x half> %985, <4 x float> %952) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %987 = fsub <4 x float> %720, %714, !dbg !42
  %988 = extractelement <4 x float> %987, i64 0, !dbg !49
  %989 = tail call float @llvm.amdgcn.exp2.f32(float %988), !dbg !49
  %local.reduce145 = fadd float %989, %local.reduce141, !dbg !51, !frisk.local_reduction !5
  %990 = insertelement <4 x float> poison, float %989, i64 0, !dbg !50
  %991 = extractelement <4 x float> %987, i64 1, !dbg !49
  %992 = tail call float @llvm.amdgcn.exp2.f32(float %991), !dbg !49
  %local.reduce146 = fadd float %992, %local.reduce142, !dbg !51, !frisk.local_reduction !5
  %993 = insertelement <4 x float> %990, float %992, i64 1, !dbg !50
  %994 = extractelement <4 x float> %987, i64 2, !dbg !49
  %995 = tail call float @llvm.amdgcn.exp2.f32(float %994), !dbg !49
  %local.reduce147 = fadd float %995, %local.reduce143, !dbg !51, !frisk.local_reduction !5
  %996 = insertelement <4 x float> %993, float %995, i64 2, !dbg !50
  %997 = extractelement <4 x float> %987, i64 3, !dbg !49
  %998 = tail call float @llvm.amdgcn.exp2.f32(float %997), !dbg !49
  %local.reduce148 = fadd float %998, %local.reduce144, !dbg !51, !frisk.local_reduction !5
  %999 = insertelement <4 x float> %996, float %998, i64 3, !dbg !50
  %1000 = fptrunc <4 x float> %999 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1001 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1000, <4 x half> %969, <4 x float> %967) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1000, <4 x half> %985, <4 x float> %968) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1003 = bitcast <4 x i32> %754 to <8 x half>, !dbg !47
  %1004 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1005 = fsub <4 x float> %712, %705, !dbg !42
  %1006 = extractelement <4 x float> %1005, i64 0, !dbg !49
  %1007 = tail call float @llvm.amdgcn.exp2.f32(float %1006), !dbg !49
  %local.reduce180 = fadd float %1007, %local.reduce176, !dbg !51, !frisk.local_reduction !5
  %1008 = insertelement <4 x float> poison, float %1007, i64 0, !dbg !50
  %1009 = extractelement <4 x float> %1005, i64 1, !dbg !49
  %1010 = tail call float @llvm.amdgcn.exp2.f32(float %1009), !dbg !49
  %local.reduce181 = fadd float %1010, %local.reduce177, !dbg !51, !frisk.local_reduction !5
  %local.reduce184 = fadd float %local.reduce180, %local.reduce181, !dbg !51, !frisk.local_reduction !5
  %1011 = insertelement <4 x float> %1008, float %1010, i64 1, !dbg !50
  %1012 = extractelement <4 x float> %1005, i64 2, !dbg !49
  %1013 = tail call float @llvm.amdgcn.exp2.f32(float %1012), !dbg !49
  %local.reduce182 = fadd float %1013, %local.reduce178, !dbg !51, !frisk.local_reduction !5
  %1014 = insertelement <4 x float> %1011, float %1013, i64 2, !dbg !50
  %1015 = extractelement <4 x float> %1005, i64 3, !dbg !49
  %1016 = tail call float @llvm.amdgcn.exp2.f32(float %1015), !dbg !49
  %local.reduce183 = fadd float %1016, %local.reduce179, !dbg !51, !frisk.local_reduction !5
  %local.reduce185 = fadd float %local.reduce182, %local.reduce183, !dbg !51, !frisk.local_reduction !5
  %1017 = fadd float %local.reduce184, %local.reduce185, !dbg !51, !frisk.local_reduction !5
  %1018 = fadd float %729, %1017, !dbg !51
  %1019 = insertelement <1 x float> poison, float %1018, i64 0, !dbg !11
  %1020 = insertvalue [2 x <1 x float>] poison, <1 x float> %1019, 0, !dbg !11
  %1021 = insertelement <4 x float> %1014, float %1016, i64 3, !dbg !50
  %1022 = fptrunc <4 x float> %1021 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1023 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1022, <4 x half> %1004, <4 x float> %984) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1024 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1025 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1022, <4 x half> %1024, <4 x float> %986) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1026 = fsub <4 x float> %721, %714, !dbg !42
  %1027 = extractelement <4 x float> %1026, i64 0, !dbg !49
  %1028 = tail call float @llvm.amdgcn.exp2.f32(float %1027), !dbg !49
  %local.reduce149 = fadd float %1028, %local.reduce145, !dbg !51, !frisk.local_reduction !5
  %1029 = insertelement <4 x float> poison, float %1028, i64 0, !dbg !50
  %1030 = extractelement <4 x float> %1026, i64 1, !dbg !49
  %1031 = tail call float @llvm.amdgcn.exp2.f32(float %1030), !dbg !49
  %local.reduce150 = fadd float %1031, %local.reduce146, !dbg !51, !frisk.local_reduction !5
  %local.reduce153 = fadd float %local.reduce149, %local.reduce150, !dbg !51, !frisk.local_reduction !5
  %1032 = insertelement <4 x float> %1029, float %1031, i64 1, !dbg !50
  %1033 = extractelement <4 x float> %1026, i64 2, !dbg !49
  %1034 = tail call float @llvm.amdgcn.exp2.f32(float %1033), !dbg !49
  %local.reduce151 = fadd float %1034, %local.reduce147, !dbg !51, !frisk.local_reduction !5
  %1035 = insertelement <4 x float> %1032, float %1034, i64 2, !dbg !50
  %1036 = extractelement <4 x float> %1026, i64 3, !dbg !49
  %1037 = tail call float @llvm.amdgcn.exp2.f32(float %1036), !dbg !49
  %local.reduce152 = fadd float %1037, %local.reduce148, !dbg !51, !frisk.local_reduction !5
  %local.reduce154 = fadd float %local.reduce151, %local.reduce152, !dbg !51, !frisk.local_reduction !5
  %1038 = fadd float %local.reduce153, %local.reduce154, !dbg !51, !frisk.local_reduction !5
  %1039 = fadd float %737, %1038, !dbg !51
  %1040 = insertelement <1 x float> poison, float %1039, i64 0, !dbg !11
  %1041 = insertvalue [2 x <1 x float>] %1020, <1 x float> %1040, 1, !dbg !11
  %1042 = insertelement <4 x float> %1035, float %1037, i64 3, !dbg !50
  %1043 = fptrunc <4 x float> %1042 to <4 x half>, !dbg !50
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1044 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1043, <4 x half> %1004, <4 x float> %1001) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1045 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1043, <4 x half> %1024, <4 x float> %1002) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %vdb.address1558 = or disjoint i64 %.idx15081510, 64, !dbg !39
  %vdb.address1559 = or disjoint i64 %address.local988, %vdb.address1558, !dbg !39
  %vdb.address1560 = trunc i64 %vdb.address1559 to i32, !dbg !39
  %vdb.address1561 = shl i32 %vdb.address1560, 1, !dbg !39
  %vdb.address1568 = or disjoint i64 %address.local994, %vdb.address1558, !dbg !39
  %vdb.address1569 = trunc i64 %vdb.address1568 to i32, !dbg !39
  %vdb.address1570 = shl i32 %vdb.address1569, 1, !dbg !39
  %1046 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %181, i32 %vdb.address1561, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %1047 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %185, i32 %vdb.address1570, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local1002, i64 8192, !dbg !47
  %1048 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1049 = bitcast <4 x i32> %1048 to <8 x half>, !dbg !47
  %vdb.read1550 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8448, !dbg !47
  %1050 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1051 = bitcast <4 x i32> %1050 to <8 x half>, !dbg !47
  %vdb.read1551 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8704, !dbg !47
  %1052 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1053 = bitcast <4 x i32> %1052 to <8 x half>, !dbg !47
  %vdb.read1552 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8960, !dbg !47
  %1054 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1552, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1055 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !47
  %vdb.read1553 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 12288, !dbg !47
  %1056 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1553, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1057 = bitcast <4 x i32> %1056 to <8 x half>, !dbg !47
  %vdb.read1554 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12544, !dbg !47
  %1058 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1554, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1059 = bitcast <4 x i32> %1058 to <8 x half>, !dbg !47
  %vdb.read1555 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12800, !dbg !47
  %1060 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1555, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1061 = bitcast <4 x i32> %1060 to <8 x half>, !dbg !47
  %vdb.read1556 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 13056, !dbg !47
  %1062 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1556, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1063 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1064 = extractvalue [2 x <8 x float>] %143, 0, !dbg !45
  %1065 = shufflevector <8 x float> %1064, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1066 = fmul <4 x float> %1065, %726, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1067 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %772, <4 x half> %1063, <4 x float> %1066) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1068 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1069 = shufflevector <8 x float> %1064, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1070 = fmul <4 x float> %1069, %726, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1071 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %772, <4 x half> %1068, <4 x float> %1070) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1072 = extractvalue [2 x <8 x float>] %143, 1, !dbg !45
  %1073 = shufflevector <8 x float> %1072, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1074 = fmul <4 x float> %1073, %734, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1075 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %794, <4 x half> %1063, <4 x float> %1074) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1076 = shufflevector <8 x float> %1072, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1077 = fmul <4 x float> %1076, %734, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %794, <4 x half> %1068, <4 x float> %1077) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1079 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %813, <4 x half> %1079, <4 x float> %1067) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1081 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %813, <4 x half> %1081, <4 x float> %1071) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1083 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %830, <4 x half> %1079, <4 x float> %1075) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %830, <4 x half> %1081, <4 x float> %1078) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1085 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %847, <4 x half> %1085, <4 x float> %1080) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1087 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %847, <4 x half> %1087, <4 x float> %1082) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1089 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %864, <4 x half> %1085, <4 x float> %1083) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %864, <4 x half> %1087, <4 x float> %1084) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1091 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %881, <4 x half> %1091, <4 x float> %1086) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1093 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %881, <4 x half> %1093, <4 x float> %1088) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1095 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %898, <4 x half> %1091, <4 x float> %1089) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %898, <4 x half> %1093, <4 x float> %1090) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1097 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1098 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %915, <4 x half> %1097, <4 x float> %1092) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1099 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1100 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %915, <4 x half> %1099, <4 x float> %1094) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1101 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %932, <4 x half> %1097, <4 x float> %1095) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %932, <4 x half> %1099, <4 x float> %1096) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1103 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %949, <4 x half> %1103, <4 x float> %1098) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1105 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %949, <4 x half> %1105, <4 x float> %1100) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1107 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %966, <4 x half> %1103, <4 x float> %1101) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1108 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %966, <4 x half> %1105, <4 x float> %1102) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1109 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1110 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %983, <4 x half> %1109, <4 x float> %1104) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1111 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1112 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %983, <4 x half> %1111, <4 x float> %1106) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1000, <4 x half> %1109, <4 x float> %1107) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1114 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1000, <4 x half> %1111, <4 x float> %1108) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1115 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !47
  %1116 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1022, <4 x half> %1116, <4 x float> %1110) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1118 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1119 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1022, <4 x half> %1118, <4 x float> %1112) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1043, <4 x half> %1116, <4 x float> %1113) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1121 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1043, <4 x half> %1118, <4 x float> %1114) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %vdb.address1576 = or disjoint i64 %.idx15081510, 96, !dbg !39
  %vdb.address1577 = or disjoint i64 %address.local988, %vdb.address1576, !dbg !39
  %vdb.address1578 = trunc i64 %vdb.address1577 to i32, !dbg !39
  %vdb.address1579 = shl i32 %vdb.address1578, 1, !dbg !39
  %vdb.address1586 = or disjoint i64 %address.local994, %vdb.address1576, !dbg !39
  %vdb.address1587 = trunc i64 %vdb.address1586 to i32, !dbg !39
  %vdb.address1588 = shl i32 %vdb.address1587, 1, !dbg !39
  %1122 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1579, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %1123 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1549, i32 %vdb.address1588, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %1124 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1125 = bitcast <4 x i32> %1124 to <8 x half>, !dbg !47
  %1126 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1127 = bitcast <4 x i32> %1126 to <8 x half>, !dbg !47
  %1128 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1129 = bitcast <4 x i32> %1128 to <8 x half>, !dbg !47
  %1130 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1131 = bitcast <4 x i32> %1130 to <8 x half>, !dbg !47
  %1132 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1133 = bitcast <4 x i32> %1132 to <8 x half>, !dbg !47
  %1134 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1135 = bitcast <4 x i32> %1134 to <8 x half>, !dbg !47
  %1136 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1137 = bitcast <4 x i32> %1136 to <8 x half>, !dbg !47
  %1138 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1139 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1140 = extractvalue [2 x <8 x float>] %142, 0, !dbg !45
  %1141 = shufflevector <8 x float> %1140, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1142 = fmul <4 x float> %1141, %726, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %772, <4 x half> %1139, <4 x float> %1142) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1144 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1145 = shufflevector <8 x float> %1140, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1146 = fmul <4 x float> %1145, %726, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %772, <4 x half> %1144, <4 x float> %1146) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1148 = extractvalue [2 x <8 x float>] %142, 1, !dbg !45
  %1149 = shufflevector <8 x float> %1148, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1150 = fmul <4 x float> %1149, %734, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %794, <4 x half> %1139, <4 x float> %1150) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1152 = shufflevector <8 x float> %1148, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1153 = fmul <4 x float> %1152, %734, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %794, <4 x half> %1144, <4 x float> %1153) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1155 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %813, <4 x half> %1155, <4 x float> %1143) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1157 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %813, <4 x half> %1157, <4 x float> %1147) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %830, <4 x half> %1155, <4 x float> %1151) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %830, <4 x half> %1157, <4 x float> %1154) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1161 = shufflevector <8 x half> %1129, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %847, <4 x half> %1161, <4 x float> %1156) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1163 = shufflevector <8 x half> %1129, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %847, <4 x half> %1163, <4 x float> %1158) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %864, <4 x half> %1161, <4 x float> %1159) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %864, <4 x half> %1163, <4 x float> %1160) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1167 = shufflevector <8 x half> %1131, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %881, <4 x half> %1167, <4 x float> %1162) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1169 = shufflevector <8 x half> %1131, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %881, <4 x half> %1169, <4 x float> %1164) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %898, <4 x half> %1167, <4 x float> %1165) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %898, <4 x half> %1169, <4 x float> %1166) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1173 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %915, <4 x half> %1173, <4 x float> %1168) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1175 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %915, <4 x half> %1175, <4 x float> %1170) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %932, <4 x half> %1173, <4 x float> %1171) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %932, <4 x half> %1175, <4 x float> %1172) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1179 = shufflevector <8 x half> %1135, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %949, <4 x half> %1179, <4 x float> %1174) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1181 = shufflevector <8 x half> %1135, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %949, <4 x half> %1181, <4 x float> %1176) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %966, <4 x half> %1179, <4 x float> %1177) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %966, <4 x half> %1181, <4 x float> %1178) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1185 = shufflevector <8 x half> %1137, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %983, <4 x half> %1185, <4 x float> %1180) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1187 = shufflevector <8 x half> %1137, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %983, <4 x half> %1187, <4 x float> %1182) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1000, <4 x half> %1185, <4 x float> %1183) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1000, <4 x half> %1187, <4 x float> %1184) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1191 = bitcast <4 x i32> %1138 to <8 x half>, !dbg !47
  %1192 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1022, <4 x half> %1192, <4 x float> %1186) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1194 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1022, <4 x half> %1194, <4 x float> %1188) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1043, <4 x half> %1192, <4 x float> %1189) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1043, <4 x half> %1194, <4 x float> %1190) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %1198 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1199 = bitcast <4 x i32> %1198 to <8 x half>, !dbg !47
  %1200 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1201 = bitcast <4 x i32> %1200 to <8 x half>, !dbg !47
  %1202 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1203 = bitcast <4 x i32> %1202 to <8 x half>, !dbg !47
  %1204 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1552, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1205 = bitcast <4 x i32> %1204 to <8 x half>, !dbg !47
  %1206 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1553, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1207 = bitcast <4 x i32> %1206 to <8 x half>, !dbg !47
  %1208 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1554, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1209 = bitcast <4 x i32> %1208 to <8 x half>, !dbg !47
  %1210 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1555, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1211 = bitcast <4 x i32> %1210 to <8 x half>, !dbg !47
  %1212 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1556, i32 0) #11, !dbg !47, !frisk.v_double_buffer !48
  %1213 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1214 = extractvalue [2 x <8 x float>] %141, 0, !dbg !45
  %1215 = shufflevector <8 x float> %1214, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1216 = fmul <4 x float> %1215, %726, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %772, <4 x half> %1213, <4 x float> %1216) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1218 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1219 = shufflevector <8 x float> %1214, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1220 = fmul <4 x float> %1219, %726, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %772, <4 x half> %1218, <4 x float> %1220) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1222 = extractvalue [2 x <8 x float>] %141, 1, !dbg !45
  %1223 = shufflevector <8 x float> %1222, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1224 = fmul <4 x float> %1223, %734, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %794, <4 x half> %1213, <4 x float> %1224) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1226 = shufflevector <8 x float> %1222, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1227 = fmul <4 x float> %1226, %734, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %794, <4 x half> %1218, <4 x float> %1227) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1229 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %813, <4 x half> %1229, <4 x float> %1217) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1231 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %813, <4 x half> %1231, <4 x float> %1221) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %830, <4 x half> %1229, <4 x float> %1225) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %830, <4 x half> %1231, <4 x float> %1228) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1235 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %847, <4 x half> %1235, <4 x float> %1230) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1237 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %847, <4 x half> %1237, <4 x float> %1232) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %864, <4 x half> %1235, <4 x float> %1233) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %864, <4 x half> %1237, <4 x float> %1234) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1241 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %881, <4 x half> %1241, <4 x float> %1236) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1243 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %881, <4 x half> %1243, <4 x float> %1238) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %898, <4 x half> %1241, <4 x float> %1239) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %898, <4 x half> %1243, <4 x float> %1240) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1247 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %915, <4 x half> %1247, <4 x float> %1242) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1249 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %915, <4 x half> %1249, <4 x float> %1244) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %932, <4 x half> %1247, <4 x float> %1245) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %932, <4 x half> %1249, <4 x float> %1246) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1253 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %949, <4 x half> %1253, <4 x float> %1248) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1255 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %949, <4 x half> %1255, <4 x float> %1250) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %966, <4 x half> %1253, <4 x float> %1251) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %966, <4 x half> %1255, <4 x float> %1252) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1259 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %983, <4 x half> %1259, <4 x float> %1254) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1261 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %983, <4 x half> %1261, <4 x float> %1256) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1000, <4 x half> %1259, <4 x float> %1257) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1000, <4 x half> %1261, <4 x float> %1258) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1265 = bitcast <4 x i32> %1212 to <8 x half>, !dbg !47
  %1266 = shufflevector <8 x half> %1265, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1022, <4 x half> %1266, <4 x float> %1260) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1268 = shufflevector <8 x half> %1265, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1022, <4 x half> %1268, <4 x float> %1262) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1043, <4 x half> %1266, <4 x float> %1263) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1043, <4 x half> %1268, <4 x float> %1264) #8, !dbg !47, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !11
  %1272 = shufflevector <4 x float> %1023, <4 x float> %1025, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1273 = insertvalue [2 x <8 x float>] poison, <8 x float> %1272, 0, !dbg !11
  %1274 = shufflevector <4 x float> %1044, <4 x float> %1045, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1275 = insertvalue [2 x <8 x float>] %1273, <8 x float> %1274, 1, !dbg !11
  %1276 = insertvalue [2 x <1 x float>] poison, <1 x float> %700, 0, !dbg !11
  %1277 = insertvalue [2 x <1 x float>] %1276, <1 x float> %703, 1, !dbg !11
  %1278 = shufflevector <4 x float> %1117, <4 x float> %1119, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1279 = insertvalue [2 x <8 x float>] poison, <8 x float> %1278, 0, !dbg !11
  %1280 = shufflevector <4 x float> %1120, <4 x float> %1121, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1281 = insertvalue [2 x <8 x float>] %1279, <8 x float> %1280, 1, !dbg !11
  %1282 = shufflevector <4 x float> %1193, <4 x float> %1195, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1283 = insertvalue [2 x <8 x float>] poison, <8 x float> %1282, 0, !dbg !11
  %1284 = shufflevector <4 x float> %1196, <4 x float> %1197, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1285 = insertvalue [2 x <8 x float>] %1283, <8 x float> %1284, 1, !dbg !11
  %1286 = shufflevector <4 x float> %1267, <4 x float> %1269, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1287 = insertvalue [2 x <8 x float>] poison, <8 x float> %1286, 0, !dbg !11
  %1288 = shufflevector <4 x float> %1270, <4 x float> %1271, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1289 = insertvalue [2 x <8 x float>] %1287, <8 x float> %1288, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1290 = add nuw nsw i64 %147, 1, !dbg !11
  %1291 = icmp slt i64 %1290, %105, !dbg !11
  br i1 %1291, label %140, label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %140, %.._crit_edge_crit_edge
  %.pre-phi29 = phi i32 [ %.pre28, %.._crit_edge_crit_edge ], [ %648, %140 ], !dbg !11
  %.pre-phi27 = phi i32 [ %.pre26, %.._crit_edge_crit_edge ], [ %640, %140 ], !dbg !11
  %.pre-phi25 = phi i32 [ %.pre24, %.._crit_edge_crit_edge ], [ %639, %140 ], !dbg !11
  %lane.sum.lane254.pre-phi = phi i32 [ %.pre22, %.._crit_edge_crit_edge ], [ %637, %140 ], !dbg !11
  %.lcssa7 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1275, %140 ], !dbg !11
  %.lcssa6 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1041, %140 ], !dbg !11
  %.lcssa5 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1281, %140 ], !dbg !11
  %.lcssa4 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1285, %140 ], !dbg !11
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1289, %140 ], !dbg !11
  %lane.sum.field251 = extractvalue [2 x <1 x float>] %.lcssa6, 1, !dbg !11
  %lane.sum.local252 = extractelement <1 x float> %lane.sum.field251, i64 0, !dbg !11
  %1292 = shl nuw nsw i32 %lane.sum.lane254.pre-phi, 2, !dbg !11
  %1293 = xor i32 %1292, 64, !dbg !11
  %1294 = bitcast float %lane.sum.local252 to i32, !dbg !11
  %lane.sum.shuffle255 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1293, i32 %1294), !dbg !11
  %1295 = bitcast i32 %lane.sum.shuffle255 to float, !dbg !11
  %lane.sum.final256 = fadd float %lane.sum.local252, %1295, !dbg !11
  %1296 = xor i32 %1292, 128, !dbg !11
  %1297 = bitcast float %lane.sum.final256 to i32, !dbg !11
  %lane.sum.shuffle257 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1297), !dbg !11
  %1298 = bitcast i32 %lane.sum.shuffle257 to float, !dbg !11
  %lane.sum.final258 = fadd float %lane.sum.final256, %1298, !dbg !11
  %1299 = insertelement <1 x float> poison, float %lane.sum.final258, i64 0, !dbg !11
  %lane.sum.field = extractvalue [2 x <1 x float>] %.lcssa6, 0, !dbg !11
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !11
  %1300 = bitcast float %lane.sum.local to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1293, i32 %1300), !dbg !11
  %1301 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %lane.sum.local, %1301, !dbg !11
  %1302 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle249 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1302), !dbg !11
  %1303 = bitcast i32 %lane.sum.shuffle249 to float, !dbg !11
  %lane.sum.final250 = fadd float %lane.sum.final, %1303, !dbg !11
  %1304 = insertelement <1 x float> poison, float %lane.sum.final250, i64 0, !dbg !11
  %1305 = and i32 %kdma.tid, 16, !dbg !11
  %.not = icmp eq i32 %1305, 0, !dbg !11
  %1306 = and i32 %kdma.tid, 32, !dbg !11
  %.not36 = icmp eq i32 %1306, 0, !dbg !11
  %1307 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !11
  %1308 = extractelement <8 x float> %1307, i64 0, !dbg !11
  %1309 = extractelement <8 x float> %1307, i64 1, !dbg !11
  %1310 = extractelement <8 x float> %1307, i64 2, !dbg !11
  %1311 = extractelement <8 x float> %1307, i64 3, !dbg !11
  %1312 = select i1 %.not, float %1309, float %1308, !dbg !11
  %1313 = icmp samesign ult i32 %.pre-phi25, %.pre-phi27, !dbg !11
  %1314 = select i1 %1313, i32 %.pre-phi25, i32 %lane.sum.lane254.pre-phi, !dbg !11
  %1315 = shl nuw nsw i32 %1314, 2, !dbg !11
  %1316 = bitcast float %1312 to i32, !dbg !11
  %1317 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1316), !dbg !11
  %1318 = bitcast i32 %1317 to float, !dbg !11
  %1319 = select i1 %.not, float %1308, float %1318, !dbg !11
  %1320 = select i1 %.not, float %1318, float %1309, !dbg !11
  %1321 = select i1 %.not, float %1311, float %1310, !dbg !11
  %1322 = bitcast float %1321 to i32, !dbg !11
  %1323 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1322), !dbg !11
  %1324 = bitcast i32 %1323 to float, !dbg !11
  %1325 = select i1 %.not, float %1310, float %1324, !dbg !11
  %1326 = select i1 %.not, float %1324, float %1311, !dbg !11
  %1327 = select i1 %.not36, float %1325, float %1319, !dbg !11
  %1328 = icmp samesign ult i32 %.pre-phi29, %.pre-phi27, !dbg !11
  %1329 = select i1 %1328, i32 %.pre-phi29, i32 %lane.sum.lane254.pre-phi, !dbg !11
  %1330 = shl nuw nsw i32 %1329, 2, !dbg !11
  %1331 = bitcast float %1327 to i32, !dbg !11
  %1332 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1331), !dbg !11
  %1333 = bitcast i32 %1332 to float, !dbg !11
  %1334 = select i1 %.not36, float %1326, float %1320, !dbg !11
  %1335 = bitcast float %1334 to i32, !dbg !11
  %1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1335), !dbg !11
  %1337 = bitcast i32 %1336 to float, !dbg !11
  %1338 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1339 = shufflevector <4 x i1> %1338, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1340 = insertelement <4 x float> poison, float %1319, i64 0, !dbg !11
  %1341 = insertelement <4 x float> %1340, float %1320, i64 1, !dbg !11
  %1342 = insertelement <4 x float> %1341, float %1333, i64 2, !dbg !11
  %1343 = insertelement <4 x float> %1342, float %1337, i64 3, !dbg !11
  %1344 = insertelement <4 x float> poison, float %1333, i64 0, !dbg !11
  %1345 = insertelement <4 x float> %1344, float %1337, i64 1, !dbg !11
  %1346 = insertelement <4 x float> %1345, float %1325, i64 2, !dbg !11
  %1347 = insertelement <4 x float> %1346, float %1326, i64 3, !dbg !11
  %1348 = select <4 x i1> %1339, <4 x float> %1343, <4 x float> %1347, !dbg !11
  %1349 = extractelement <8 x float> %1307, i64 4, !dbg !11
  %1350 = extractelement <8 x float> %1307, i64 5, !dbg !11
  %1351 = extractelement <8 x float> %1307, i64 6, !dbg !11
  %1352 = extractelement <8 x float> %1307, i64 7, !dbg !11
  %1353 = select i1 %.not, float %1350, float %1349, !dbg !11
  %1354 = bitcast float %1353 to i32, !dbg !11
  %1355 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1354), !dbg !11
  %1356 = bitcast i32 %1355 to float, !dbg !11
  %1357 = select i1 %.not, float %1349, float %1356, !dbg !11
  %1358 = select i1 %.not, float %1356, float %1350, !dbg !11
  %1359 = select i1 %.not, float %1352, float %1351, !dbg !11
  %1360 = bitcast float %1359 to i32, !dbg !11
  %1361 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1360), !dbg !11
  %1362 = bitcast i32 %1361 to float, !dbg !11
  %1363 = select i1 %.not, float %1351, float %1362, !dbg !11
  %1364 = select i1 %.not, float %1362, float %1352, !dbg !11
  %1365 = select i1 %.not36, float %1363, float %1357, !dbg !11
  %1366 = bitcast float %1365 to i32, !dbg !11
  %1367 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1366), !dbg !11
  %1368 = bitcast i32 %1367 to float, !dbg !11
  %1369 = select i1 %.not36, float %1364, float %1358, !dbg !11
  %1370 = bitcast float %1369 to i32, !dbg !11
  %1371 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1370), !dbg !11
  %1372 = bitcast i32 %1371 to float, !dbg !11
  %1373 = insertelement <4 x float> poison, float %1357, i64 0, !dbg !11
  %1374 = insertelement <4 x float> %1373, float %1358, i64 1, !dbg !11
  %1375 = insertelement <4 x float> %1374, float %1368, i64 2, !dbg !11
  %1376 = insertelement <4 x float> %1375, float %1372, i64 3, !dbg !11
  %1377 = insertelement <4 x float> poison, float %1368, i64 0, !dbg !11
  %1378 = insertelement <4 x float> %1377, float %1372, i64 1, !dbg !11
  %1379 = insertelement <4 x float> %1378, float %1363, i64 2, !dbg !11
  %1380 = insertelement <4 x float> %1379, float %1364, i64 3, !dbg !11
  %1381 = select <4 x i1> %1339, <4 x float> %1376, <4 x float> %1380, !dbg !11
  %1382 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !11
  %1383 = extractelement <8 x float> %1382, i64 0, !dbg !11
  %1384 = extractelement <8 x float> %1382, i64 1, !dbg !11
  %1385 = extractelement <8 x float> %1382, i64 2, !dbg !11
  %1386 = extractelement <8 x float> %1382, i64 3, !dbg !11
  %1387 = select i1 %.not, float %1384, float %1383, !dbg !11
  %1388 = bitcast float %1387 to i32, !dbg !11
  %1389 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1388), !dbg !11
  %1390 = bitcast i32 %1389 to float, !dbg !11
  %1391 = select i1 %.not, float %1383, float %1390, !dbg !11
  %1392 = select i1 %.not, float %1390, float %1384, !dbg !11
  %1393 = select i1 %.not, float %1386, float %1385, !dbg !11
  %1394 = bitcast float %1393 to i32, !dbg !11
  %1395 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1394), !dbg !11
  %1396 = bitcast i32 %1395 to float, !dbg !11
  %1397 = select i1 %.not, float %1385, float %1396, !dbg !11
  %1398 = select i1 %.not, float %1396, float %1386, !dbg !11
  %1399 = select i1 %.not36, float %1397, float %1391, !dbg !11
  %1400 = bitcast float %1399 to i32, !dbg !11
  %1401 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1400), !dbg !11
  %1402 = bitcast i32 %1401 to float, !dbg !11
  %1403 = select i1 %.not36, float %1398, float %1392, !dbg !11
  %1404 = bitcast float %1403 to i32, !dbg !11
  %1405 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1404), !dbg !11
  %1406 = bitcast i32 %1405 to float, !dbg !11
  %1407 = insertelement <4 x float> poison, float %1391, i64 0, !dbg !11
  %1408 = insertelement <4 x float> %1407, float %1392, i64 1, !dbg !11
  %1409 = insertelement <4 x float> %1408, float %1402, i64 2, !dbg !11
  %1410 = insertelement <4 x float> %1409, float %1406, i64 3, !dbg !11
  %1411 = insertelement <4 x float> poison, float %1402, i64 0, !dbg !11
  %1412 = insertelement <4 x float> %1411, float %1406, i64 1, !dbg !11
  %1413 = insertelement <4 x float> %1412, float %1397, i64 2, !dbg !11
  %1414 = insertelement <4 x float> %1413, float %1398, i64 3, !dbg !11
  %1415 = select <4 x i1> %1339, <4 x float> %1410, <4 x float> %1414, !dbg !11
  %1416 = extractelement <8 x float> %1382, i64 4, !dbg !11
  %1417 = extractelement <8 x float> %1382, i64 5, !dbg !11
  %1418 = extractelement <8 x float> %1382, i64 6, !dbg !11
  %1419 = extractelement <8 x float> %1382, i64 7, !dbg !11
  %1420 = select i1 %.not, float %1417, float %1416, !dbg !11
  %1421 = bitcast float %1420 to i32, !dbg !11
  %1422 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1421), !dbg !11
  %1423 = bitcast i32 %1422 to float, !dbg !11
  %1424 = select i1 %.not, float %1416, float %1423, !dbg !11
  %1425 = select i1 %.not, float %1423, float %1417, !dbg !11
  %1426 = select i1 %.not, float %1419, float %1418, !dbg !11
  %1427 = bitcast float %1426 to i32, !dbg !11
  %1428 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1427), !dbg !11
  %1429 = bitcast i32 %1428 to float, !dbg !11
  %1430 = select i1 %.not, float %1418, float %1429, !dbg !11
  %1431 = select i1 %.not, float %1429, float %1419, !dbg !11
  %1432 = select i1 %.not36, float %1430, float %1424, !dbg !11
  %1433 = bitcast float %1432 to i32, !dbg !11
  %1434 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1433), !dbg !11
  %1435 = bitcast i32 %1434 to float, !dbg !11
  %1436 = select i1 %.not36, float %1431, float %1425, !dbg !11
  %1437 = bitcast float %1436 to i32, !dbg !11
  %1438 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1437), !dbg !11
  %1439 = bitcast i32 %1438 to float, !dbg !11
  %1440 = insertelement <4 x float> poison, float %1424, i64 0, !dbg !11
  %1441 = insertelement <4 x float> %1440, float %1425, i64 1, !dbg !11
  %1442 = insertelement <4 x float> %1441, float %1435, i64 2, !dbg !11
  %1443 = insertelement <4 x float> %1442, float %1439, i64 3, !dbg !11
  %1444 = insertelement <4 x float> poison, float %1435, i64 0, !dbg !11
  %1445 = insertelement <4 x float> %1444, float %1439, i64 1, !dbg !11
  %1446 = insertelement <4 x float> %1445, float %1430, i64 2, !dbg !11
  %1447 = insertelement <4 x float> %1446, float %1431, i64 3, !dbg !11
  %1448 = select <4 x i1> %1339, <4 x float> %1443, <4 x float> %1447, !dbg !11
  %1449 = extractvalue [2 x <8 x float>] %.lcssa4, 0, !dbg !11
  %1450 = extractelement <8 x float> %1449, i64 0, !dbg !11
  %1451 = extractelement <8 x float> %1449, i64 1, !dbg !11
  %1452 = extractelement <8 x float> %1449, i64 2, !dbg !11
  %1453 = extractelement <8 x float> %1449, i64 3, !dbg !11
  %1454 = select i1 %.not, float %1451, float %1450, !dbg !11
  %1455 = bitcast float %1454 to i32, !dbg !11
  %1456 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1455), !dbg !11
  %1457 = bitcast i32 %1456 to float, !dbg !11
  %1458 = select i1 %.not, float %1450, float %1457, !dbg !11
  %1459 = select i1 %.not, float %1457, float %1451, !dbg !11
  %1460 = select i1 %.not, float %1453, float %1452, !dbg !11
  %1461 = bitcast float %1460 to i32, !dbg !11
  %1462 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1461), !dbg !11
  %1463 = bitcast i32 %1462 to float, !dbg !11
  %1464 = select i1 %.not, float %1452, float %1463, !dbg !11
  %1465 = select i1 %.not, float %1463, float %1453, !dbg !11
  %1466 = select i1 %.not36, float %1464, float %1458, !dbg !11
  %1467 = bitcast float %1466 to i32, !dbg !11
  %1468 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1467), !dbg !11
  %1469 = bitcast i32 %1468 to float, !dbg !11
  %1470 = select i1 %.not36, float %1465, float %1459, !dbg !11
  %1471 = bitcast float %1470 to i32, !dbg !11
  %1472 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1471), !dbg !11
  %1473 = bitcast i32 %1472 to float, !dbg !11
  %1474 = insertelement <4 x float> poison, float %1458, i64 0, !dbg !11
  %1475 = insertelement <4 x float> %1474, float %1459, i64 1, !dbg !11
  %1476 = insertelement <4 x float> %1475, float %1469, i64 2, !dbg !11
  %1477 = insertelement <4 x float> %1476, float %1473, i64 3, !dbg !11
  %1478 = insertelement <4 x float> poison, float %1469, i64 0, !dbg !11
  %1479 = insertelement <4 x float> %1478, float %1473, i64 1, !dbg !11
  %1480 = insertelement <4 x float> %1479, float %1464, i64 2, !dbg !11
  %1481 = insertelement <4 x float> %1480, float %1465, i64 3, !dbg !11
  %1482 = select <4 x i1> %1339, <4 x float> %1477, <4 x float> %1481, !dbg !11
  %1483 = extractelement <8 x float> %1449, i64 4, !dbg !11
  %1484 = extractelement <8 x float> %1449, i64 5, !dbg !11
  %1485 = extractelement <8 x float> %1449, i64 6, !dbg !11
  %1486 = extractelement <8 x float> %1449, i64 7, !dbg !11
  %1487 = select i1 %.not, float %1484, float %1483, !dbg !11
  %1488 = bitcast float %1487 to i32, !dbg !11
  %1489 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1488), !dbg !11
  %1490 = bitcast i32 %1489 to float, !dbg !11
  %1491 = select i1 %.not, float %1483, float %1490, !dbg !11
  %1492 = select i1 %.not, float %1490, float %1484, !dbg !11
  %1493 = select i1 %.not, float %1486, float %1485, !dbg !11
  %1494 = bitcast float %1493 to i32, !dbg !11
  %1495 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1494), !dbg !11
  %1496 = bitcast i32 %1495 to float, !dbg !11
  %1497 = select i1 %.not, float %1485, float %1496, !dbg !11
  %1498 = select i1 %.not, float %1496, float %1486, !dbg !11
  %1499 = select i1 %.not36, float %1497, float %1491, !dbg !11
  %1500 = bitcast float %1499 to i32, !dbg !11
  %1501 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1500), !dbg !11
  %1502 = bitcast i32 %1501 to float, !dbg !11
  %1503 = select i1 %.not36, float %1498, float %1492, !dbg !11
  %1504 = bitcast float %1503 to i32, !dbg !11
  %1505 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1504), !dbg !11
  %1506 = bitcast i32 %1505 to float, !dbg !11
  %1507 = insertelement <4 x float> poison, float %1491, i64 0, !dbg !11
  %1508 = insertelement <4 x float> %1507, float %1492, i64 1, !dbg !11
  %1509 = insertelement <4 x float> %1508, float %1502, i64 2, !dbg !11
  %1510 = insertelement <4 x float> %1509, float %1506, i64 3, !dbg !11
  %1511 = insertelement <4 x float> poison, float %1502, i64 0, !dbg !11
  %1512 = insertelement <4 x float> %1511, float %1506, i64 1, !dbg !11
  %1513 = insertelement <4 x float> %1512, float %1497, i64 2, !dbg !11
  %1514 = insertelement <4 x float> %1513, float %1498, i64 3, !dbg !11
  %1515 = select <4 x i1> %1339, <4 x float> %1510, <4 x float> %1514, !dbg !11
  %1516 = extractvalue [2 x <8 x float>] %.lcssa4, 1, !dbg !11
  %1517 = extractelement <8 x float> %1516, i64 0, !dbg !11
  %1518 = extractelement <8 x float> %1516, i64 1, !dbg !11
  %1519 = extractelement <8 x float> %1516, i64 2, !dbg !11
  %1520 = extractelement <8 x float> %1516, i64 3, !dbg !11
  %1521 = select i1 %.not, float %1518, float %1517, !dbg !11
  %1522 = bitcast float %1521 to i32, !dbg !11
  %1523 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1522), !dbg !11
  %1524 = bitcast i32 %1523 to float, !dbg !11
  %1525 = select i1 %.not, float %1517, float %1524, !dbg !11
  %1526 = select i1 %.not, float %1524, float %1518, !dbg !11
  %1527 = select i1 %.not, float %1520, float %1519, !dbg !11
  %1528 = bitcast float %1527 to i32, !dbg !11
  %1529 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1528), !dbg !11
  %1530 = bitcast i32 %1529 to float, !dbg !11
  %1531 = select i1 %.not, float %1519, float %1530, !dbg !11
  %1532 = select i1 %.not, float %1530, float %1520, !dbg !11
  %1533 = select i1 %.not36, float %1531, float %1525, !dbg !11
  %1534 = bitcast float %1533 to i32, !dbg !11
  %1535 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1534), !dbg !11
  %1536 = bitcast i32 %1535 to float, !dbg !11
  %1537 = select i1 %.not36, float %1532, float %1526, !dbg !11
  %1538 = bitcast float %1537 to i32, !dbg !11
  %1539 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1538), !dbg !11
  %1540 = bitcast i32 %1539 to float, !dbg !11
  %1541 = insertelement <4 x float> poison, float %1525, i64 0, !dbg !11
  %1542 = insertelement <4 x float> %1541, float %1526, i64 1, !dbg !11
  %1543 = insertelement <4 x float> %1542, float %1536, i64 2, !dbg !11
  %1544 = insertelement <4 x float> %1543, float %1540, i64 3, !dbg !11
  %1545 = insertelement <4 x float> poison, float %1536, i64 0, !dbg !11
  %1546 = insertelement <4 x float> %1545, float %1540, i64 1, !dbg !11
  %1547 = insertelement <4 x float> %1546, float %1531, i64 2, !dbg !11
  %1548 = insertelement <4 x float> %1547, float %1532, i64 3, !dbg !11
  %1549 = select <4 x i1> %1339, <4 x float> %1544, <4 x float> %1548, !dbg !11
  %1550 = extractelement <8 x float> %1516, i64 4, !dbg !11
  %1551 = extractelement <8 x float> %1516, i64 5, !dbg !11
  %1552 = extractelement <8 x float> %1516, i64 6, !dbg !11
  %1553 = extractelement <8 x float> %1516, i64 7, !dbg !11
  %1554 = select i1 %.not, float %1551, float %1550, !dbg !11
  %1555 = bitcast float %1554 to i32, !dbg !11
  %1556 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1555), !dbg !11
  %1557 = bitcast i32 %1556 to float, !dbg !11
  %1558 = select i1 %.not, float %1550, float %1557, !dbg !11
  %1559 = select i1 %.not, float %1557, float %1551, !dbg !11
  %1560 = select i1 %.not, float %1553, float %1552, !dbg !11
  %1561 = bitcast float %1560 to i32, !dbg !11
  %1562 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1561), !dbg !11
  %1563 = bitcast i32 %1562 to float, !dbg !11
  %1564 = select i1 %.not, float %1552, float %1563, !dbg !11
  %1565 = select i1 %.not, float %1563, float %1553, !dbg !11
  %1566 = select i1 %.not36, float %1564, float %1558, !dbg !11
  %1567 = bitcast float %1566 to i32, !dbg !11
  %1568 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1567), !dbg !11
  %1569 = bitcast i32 %1568 to float, !dbg !11
  %1570 = select i1 %.not36, float %1565, float %1559, !dbg !11
  %1571 = bitcast float %1570 to i32, !dbg !11
  %1572 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1571), !dbg !11
  %1573 = bitcast i32 %1572 to float, !dbg !11
  %1574 = insertelement <4 x float> poison, float %1558, i64 0, !dbg !11
  %1575 = insertelement <4 x float> %1574, float %1559, i64 1, !dbg !11
  %1576 = insertelement <4 x float> %1575, float %1569, i64 2, !dbg !11
  %1577 = insertelement <4 x float> %1576, float %1573, i64 3, !dbg !11
  %1578 = insertelement <4 x float> poison, float %1569, i64 0, !dbg !11
  %1579 = insertelement <4 x float> %1578, float %1573, i64 1, !dbg !11
  %1580 = insertelement <4 x float> %1579, float %1564, i64 2, !dbg !11
  %1581 = insertelement <4 x float> %1580, float %1565, i64 3, !dbg !11
  %1582 = select <4 x i1> %1339, <4 x float> %1577, <4 x float> %1581, !dbg !11
  %1583 = extractvalue [2 x <8 x float>] %.lcssa5, 0, !dbg !11
  %1584 = extractelement <8 x float> %1583, i64 0, !dbg !11
  %1585 = extractelement <8 x float> %1583, i64 1, !dbg !11
  %1586 = extractelement <8 x float> %1583, i64 2, !dbg !11
  %1587 = extractelement <8 x float> %1583, i64 3, !dbg !11
  %1588 = select i1 %.not, float %1585, float %1584, !dbg !11
  %1589 = bitcast float %1588 to i32, !dbg !11
  %1590 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1589), !dbg !11
  %1591 = bitcast i32 %1590 to float, !dbg !11
  %1592 = select i1 %.not, float %1584, float %1591, !dbg !11
  %1593 = select i1 %.not, float %1591, float %1585, !dbg !11
  %1594 = select i1 %.not, float %1587, float %1586, !dbg !11
  %1595 = bitcast float %1594 to i32, !dbg !11
  %1596 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1595), !dbg !11
  %1597 = bitcast i32 %1596 to float, !dbg !11
  %1598 = select i1 %.not, float %1586, float %1597, !dbg !11
  %1599 = select i1 %.not, float %1597, float %1587, !dbg !11
  %1600 = select i1 %.not36, float %1598, float %1592, !dbg !11
  %1601 = bitcast float %1600 to i32, !dbg !11
  %1602 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1601), !dbg !11
  %1603 = bitcast i32 %1602 to float, !dbg !11
  %1604 = select i1 %.not36, float %1599, float %1593, !dbg !11
  %1605 = bitcast float %1604 to i32, !dbg !11
  %1606 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1605), !dbg !11
  %1607 = bitcast i32 %1606 to float, !dbg !11
  %1608 = insertelement <4 x float> poison, float %1592, i64 0, !dbg !11
  %1609 = insertelement <4 x float> %1608, float %1593, i64 1, !dbg !11
  %1610 = insertelement <4 x float> %1609, float %1603, i64 2, !dbg !11
  %1611 = insertelement <4 x float> %1610, float %1607, i64 3, !dbg !11
  %1612 = insertelement <4 x float> poison, float %1603, i64 0, !dbg !11
  %1613 = insertelement <4 x float> %1612, float %1607, i64 1, !dbg !11
  %1614 = insertelement <4 x float> %1613, float %1598, i64 2, !dbg !11
  %1615 = insertelement <4 x float> %1614, float %1599, i64 3, !dbg !11
  %1616 = select <4 x i1> %1339, <4 x float> %1611, <4 x float> %1615, !dbg !11
  %1617 = extractelement <8 x float> %1583, i64 4, !dbg !11
  %1618 = extractelement <8 x float> %1583, i64 5, !dbg !11
  %1619 = extractelement <8 x float> %1583, i64 6, !dbg !11
  %1620 = extractelement <8 x float> %1583, i64 7, !dbg !11
  %1621 = select i1 %.not, float %1618, float %1617, !dbg !11
  %1622 = bitcast float %1621 to i32, !dbg !11
  %1623 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1622), !dbg !11
  %1624 = bitcast i32 %1623 to float, !dbg !11
  %1625 = select i1 %.not, float %1617, float %1624, !dbg !11
  %1626 = select i1 %.not, float %1624, float %1618, !dbg !11
  %1627 = select i1 %.not, float %1620, float %1619, !dbg !11
  %1628 = bitcast float %1627 to i32, !dbg !11
  %1629 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1628), !dbg !11
  %1630 = bitcast i32 %1629 to float, !dbg !11
  %1631 = select i1 %.not, float %1619, float %1630, !dbg !11
  %1632 = select i1 %.not, float %1630, float %1620, !dbg !11
  %1633 = select i1 %.not36, float %1631, float %1625, !dbg !11
  %1634 = bitcast float %1633 to i32, !dbg !11
  %1635 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1634), !dbg !11
  %1636 = bitcast i32 %1635 to float, !dbg !11
  %1637 = select i1 %.not36, float %1632, float %1626, !dbg !11
  %1638 = bitcast float %1637 to i32, !dbg !11
  %1639 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1638), !dbg !11
  %1640 = bitcast i32 %1639 to float, !dbg !11
  %1641 = insertelement <4 x float> poison, float %1625, i64 0, !dbg !11
  %1642 = insertelement <4 x float> %1641, float %1626, i64 1, !dbg !11
  %1643 = insertelement <4 x float> %1642, float %1636, i64 2, !dbg !11
  %1644 = insertelement <4 x float> %1643, float %1640, i64 3, !dbg !11
  %1645 = insertelement <4 x float> poison, float %1636, i64 0, !dbg !11
  %1646 = insertelement <4 x float> %1645, float %1640, i64 1, !dbg !11
  %1647 = insertelement <4 x float> %1646, float %1631, i64 2, !dbg !11
  %1648 = insertelement <4 x float> %1647, float %1632, i64 3, !dbg !11
  %1649 = select <4 x i1> %1339, <4 x float> %1644, <4 x float> %1648, !dbg !11
  %1650 = extractvalue [2 x <8 x float>] %.lcssa5, 1, !dbg !11
  %1651 = extractelement <8 x float> %1650, i64 0, !dbg !11
  %1652 = extractelement <8 x float> %1650, i64 1, !dbg !11
  %1653 = extractelement <8 x float> %1650, i64 2, !dbg !11
  %1654 = extractelement <8 x float> %1650, i64 3, !dbg !11
  %1655 = select i1 %.not, float %1652, float %1651, !dbg !11
  %1656 = bitcast float %1655 to i32, !dbg !11
  %1657 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1656), !dbg !11
  %1658 = bitcast i32 %1657 to float, !dbg !11
  %1659 = select i1 %.not, float %1651, float %1658, !dbg !11
  %1660 = select i1 %.not, float %1658, float %1652, !dbg !11
  %1661 = select i1 %.not, float %1654, float %1653, !dbg !11
  %1662 = bitcast float %1661 to i32, !dbg !11
  %1663 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1662), !dbg !11
  %1664 = bitcast i32 %1663 to float, !dbg !11
  %1665 = select i1 %.not, float %1653, float %1664, !dbg !11
  %1666 = select i1 %.not, float %1664, float %1654, !dbg !11
  %1667 = select i1 %.not36, float %1665, float %1659, !dbg !11
  %1668 = bitcast float %1667 to i32, !dbg !11
  %1669 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1668), !dbg !11
  %1670 = bitcast i32 %1669 to float, !dbg !11
  %1671 = select i1 %.not36, float %1666, float %1660, !dbg !11
  %1672 = bitcast float %1671 to i32, !dbg !11
  %1673 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1672), !dbg !11
  %1674 = bitcast i32 %1673 to float, !dbg !11
  %1675 = insertelement <4 x float> poison, float %1659, i64 0, !dbg !11
  %1676 = insertelement <4 x float> %1675, float %1660, i64 1, !dbg !11
  %1677 = insertelement <4 x float> %1676, float %1670, i64 2, !dbg !11
  %1678 = insertelement <4 x float> %1677, float %1674, i64 3, !dbg !11
  %1679 = insertelement <4 x float> poison, float %1670, i64 0, !dbg !11
  %1680 = insertelement <4 x float> %1679, float %1674, i64 1, !dbg !11
  %1681 = insertelement <4 x float> %1680, float %1665, i64 2, !dbg !11
  %1682 = insertelement <4 x float> %1681, float %1666, i64 3, !dbg !11
  %1683 = select <4 x i1> %1339, <4 x float> %1678, <4 x float> %1682, !dbg !11
  %1684 = extractelement <8 x float> %1650, i64 4, !dbg !11
  %1685 = extractelement <8 x float> %1650, i64 5, !dbg !11
  %1686 = extractelement <8 x float> %1650, i64 6, !dbg !11
  %1687 = extractelement <8 x float> %1650, i64 7, !dbg !11
  %1688 = select i1 %.not, float %1685, float %1684, !dbg !11
  %1689 = bitcast float %1688 to i32, !dbg !11
  %1690 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1689), !dbg !11
  %1691 = bitcast i32 %1690 to float, !dbg !11
  %1692 = select i1 %.not, float %1684, float %1691, !dbg !11
  %1693 = select i1 %.not, float %1691, float %1685, !dbg !11
  %1694 = select i1 %.not, float %1687, float %1686, !dbg !11
  %1695 = bitcast float %1694 to i32, !dbg !11
  %1696 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1695), !dbg !11
  %1697 = bitcast i32 %1696 to float, !dbg !11
  %1698 = select i1 %.not, float %1686, float %1697, !dbg !11
  %1699 = select i1 %.not, float %1697, float %1687, !dbg !11
  %1700 = select i1 %.not36, float %1698, float %1692, !dbg !11
  %1701 = bitcast float %1700 to i32, !dbg !11
  %1702 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1701), !dbg !11
  %1703 = bitcast i32 %1702 to float, !dbg !11
  %1704 = select i1 %.not36, float %1699, float %1693, !dbg !11
  %1705 = bitcast float %1704 to i32, !dbg !11
  %1706 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1705), !dbg !11
  %1707 = bitcast i32 %1706 to float, !dbg !11
  %1708 = insertelement <4 x float> poison, float %1692, i64 0, !dbg !11
  %1709 = insertelement <4 x float> %1708, float %1693, i64 1, !dbg !11
  %1710 = insertelement <4 x float> %1709, float %1703, i64 2, !dbg !11
  %1711 = insertelement <4 x float> %1710, float %1707, i64 3, !dbg !11
  %1712 = insertelement <4 x float> poison, float %1703, i64 0, !dbg !11
  %1713 = insertelement <4 x float> %1712, float %1707, i64 1, !dbg !11
  %1714 = insertelement <4 x float> %1713, float %1698, i64 2, !dbg !11
  %1715 = insertelement <4 x float> %1714, float %1699, i64 3, !dbg !11
  %1716 = select <4 x i1> %1339, <4 x float> %1711, <4 x float> %1715, !dbg !11
  %1717 = extractvalue [2 x <8 x float>] %.lcssa7, 0, !dbg !11
  %1718 = extractelement <8 x float> %1717, i64 0, !dbg !11
  %1719 = extractelement <8 x float> %1717, i64 1, !dbg !11
  %1720 = extractelement <8 x float> %1717, i64 2, !dbg !11
  %1721 = extractelement <8 x float> %1717, i64 3, !dbg !11
  %1722 = select i1 %.not, float %1719, float %1718, !dbg !11
  %1723 = bitcast float %1722 to i32, !dbg !11
  %1724 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1723), !dbg !11
  %1725 = bitcast i32 %1724 to float, !dbg !11
  %1726 = select i1 %.not, float %1718, float %1725, !dbg !11
  %1727 = select i1 %.not, float %1725, float %1719, !dbg !11
  %1728 = select i1 %.not, float %1721, float %1720, !dbg !11
  %1729 = bitcast float %1728 to i32, !dbg !11
  %1730 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1729), !dbg !11
  %1731 = bitcast i32 %1730 to float, !dbg !11
  %1732 = select i1 %.not, float %1720, float %1731, !dbg !11
  %1733 = select i1 %.not, float %1731, float %1721, !dbg !11
  %1734 = select i1 %.not36, float %1732, float %1726, !dbg !11
  %1735 = bitcast float %1734 to i32, !dbg !11
  %1736 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1735), !dbg !11
  %1737 = select i1 %.not36, float %1733, float %1727, !dbg !11
  %1738 = bitcast float %1737 to i32, !dbg !11
  %1739 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1738), !dbg !11
  %1740 = extractelement <8 x float> %1717, i64 4, !dbg !11
  %1741 = extractelement <8 x float> %1717, i64 5, !dbg !11
  %1742 = extractelement <8 x float> %1717, i64 6, !dbg !11
  %1743 = extractelement <8 x float> %1717, i64 7, !dbg !11
  %1744 = select i1 %.not, float %1741, float %1740, !dbg !11
  %1745 = bitcast float %1744 to i32, !dbg !11
  %1746 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1745), !dbg !11
  %1747 = bitcast i32 %1746 to float, !dbg !11
  %1748 = select i1 %.not, float %1740, float %1747, !dbg !11
  %1749 = select i1 %.not, float %1747, float %1741, !dbg !11
  %1750 = select i1 %.not, float %1743, float %1742, !dbg !11
  %1751 = bitcast float %1750 to i32, !dbg !11
  %1752 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1751), !dbg !11
  %1753 = bitcast i32 %1752 to float, !dbg !11
  %1754 = select i1 %.not, float %1742, float %1753, !dbg !11
  %1755 = select i1 %.not, float %1753, float %1743, !dbg !11
  %1756 = select i1 %.not36, float %1754, float %1748, !dbg !11
  %1757 = bitcast float %1756 to i32, !dbg !11
  %1758 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1757), !dbg !11
  %1759 = select i1 %.not36, float %1755, float %1749, !dbg !11
  %1760 = bitcast float %1759 to i32, !dbg !11
  %1761 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1760), !dbg !11
  %1762 = extractvalue [2 x <8 x float>] %.lcssa7, 1, !dbg !11
  %1763 = extractelement <8 x float> %1762, i64 0, !dbg !11
  %1764 = extractelement <8 x float> %1762, i64 1, !dbg !11
  %1765 = extractelement <8 x float> %1762, i64 2, !dbg !11
  %1766 = extractelement <8 x float> %1762, i64 3, !dbg !11
  %1767 = select i1 %.not, float %1764, float %1763, !dbg !11
  %1768 = bitcast float %1767 to i32, !dbg !11
  %1769 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1768), !dbg !11
  %1770 = bitcast i32 %1769 to float, !dbg !11
  %1771 = select i1 %.not, float %1763, float %1770, !dbg !11
  %1772 = select i1 %.not, float %1770, float %1764, !dbg !11
  %1773 = select i1 %.not, float %1766, float %1765, !dbg !11
  %1774 = bitcast float %1773 to i32, !dbg !11
  %1775 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1774), !dbg !11
  %1776 = bitcast i32 %1775 to float, !dbg !11
  %1777 = select i1 %.not, float %1765, float %1776, !dbg !11
  %1778 = select i1 %.not, float %1776, float %1766, !dbg !11
  %1779 = select i1 %.not36, float %1777, float %1771, !dbg !11
  %1780 = bitcast float %1779 to i32, !dbg !11
  %1781 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1780), !dbg !11
  %1782 = select i1 %.not36, float %1778, float %1772, !dbg !11
  %1783 = bitcast float %1782 to i32, !dbg !11
  %1784 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1783), !dbg !11
  %1785 = extractelement <8 x float> %1762, i64 4, !dbg !11
  %1786 = extractelement <8 x float> %1762, i64 5, !dbg !11
  %1787 = extractelement <8 x float> %1762, i64 6, !dbg !11
  %1788 = extractelement <8 x float> %1762, i64 7, !dbg !11
  %1789 = select i1 %.not, float %1786, float %1785, !dbg !11
  %1790 = bitcast float %1789 to i32, !dbg !11
  %1791 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1790), !dbg !11
  %1792 = bitcast i32 %1791 to float, !dbg !11
  %1793 = select i1 %.not, float %1785, float %1792, !dbg !11
  %1794 = select i1 %.not, float %1792, float %1786, !dbg !11
  %1795 = select i1 %.not, float %1788, float %1787, !dbg !11
  %1796 = bitcast float %1795 to i32, !dbg !11
  %1797 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1315, i32 %1796), !dbg !11
  %1798 = bitcast i32 %1797 to float, !dbg !11
  %1799 = select i1 %.not, float %1787, float %1798, !dbg !11
  %1800 = select i1 %.not, float %1798, float %1788, !dbg !11
  %1801 = select i1 %.not36, float %1799, float %1793, !dbg !11
  %1802 = bitcast float %1801 to i32, !dbg !11
  %1803 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1802), !dbg !11
  %1804 = select i1 %.not36, float %1800, float %1794, !dbg !11
  %1805 = bitcast float %1804 to i32, !dbg !11
  %1806 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1330, i32 %1805), !dbg !11
  %1807 = shufflevector <1 x float> %1304, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !52
  %1808 = shufflevector <1 x float> %1299, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !52
  %1809 = bitcast i32 %1803 to float, !dbg !11
  %1810 = bitcast i32 %1806 to float, !dbg !11
  %1811 = insertelement <4 x float> poison, float %1793, i64 0, !dbg !11
  %1812 = insertelement <4 x float> %1811, float %1794, i64 1, !dbg !11
  %1813 = insertelement <4 x float> %1812, float %1809, i64 2, !dbg !11
  %1814 = insertelement <4 x float> %1813, float %1810, i64 3, !dbg !11
  %1815 = insertelement <4 x float> poison, float %1809, i64 0, !dbg !11
  %1816 = insertelement <4 x float> %1815, float %1810, i64 1, !dbg !11
  %1817 = insertelement <4 x float> %1816, float %1799, i64 2, !dbg !11
  %1818 = insertelement <4 x float> %1817, float %1800, i64 3, !dbg !11
  %1819 = select <4 x i1> %1339, <4 x float> %1814, <4 x float> %1818, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1808, !dbg !52
  %normalized = fmul <4 x float> %row.reciprocal, %1819, !dbg !52
  %1820 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !53
  %1821 = extractelement <4 x half> %1820, i64 3, !dbg !53
  %1822 = extractelement <4 x half> %1820, i64 2, !dbg !53
  %1823 = extractelement <4 x half> %1820, i64 1, !dbg !53
  %1824 = extractelement <4 x half> %1820, i64 0, !dbg !53
  %1825 = bitcast i32 %1781 to float, !dbg !11
  %1826 = bitcast i32 %1784 to float, !dbg !11
  %1827 = insertelement <4 x float> poison, float %1771, i64 0, !dbg !11
  %1828 = insertelement <4 x float> %1827, float %1772, i64 1, !dbg !11
  %1829 = insertelement <4 x float> %1828, float %1825, i64 2, !dbg !11
  %1830 = insertelement <4 x float> %1829, float %1826, i64 3, !dbg !11
  %1831 = insertelement <4 x float> poison, float %1825, i64 0, !dbg !11
  %1832 = insertelement <4 x float> %1831, float %1826, i64 1, !dbg !11
  %1833 = insertelement <4 x float> %1832, float %1777, i64 2, !dbg !11
  %1834 = insertelement <4 x float> %1833, float %1778, i64 3, !dbg !11
  %1835 = select <4 x i1> %1339, <4 x float> %1830, <4 x float> %1834, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %1835, !dbg !52
  %1836 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !53
  %1837 = extractelement <4 x half> %1836, i64 3, !dbg !53
  %1838 = extractelement <4 x half> %1836, i64 2, !dbg !53
  %1839 = extractelement <4 x half> %1836, i64 1, !dbg !53
  %1840 = extractelement <4 x half> %1836, i64 0, !dbg !53
  %1841 = bitcast i32 %1758 to float, !dbg !11
  %1842 = bitcast i32 %1761 to float, !dbg !11
  %1843 = insertelement <4 x float> poison, float %1748, i64 0, !dbg !11
  %1844 = insertelement <4 x float> %1843, float %1749, i64 1, !dbg !11
  %1845 = insertelement <4 x float> %1844, float %1841, i64 2, !dbg !11
  %1846 = insertelement <4 x float> %1845, float %1842, i64 3, !dbg !11
  %1847 = insertelement <4 x float> poison, float %1841, i64 0, !dbg !11
  %1848 = insertelement <4 x float> %1847, float %1842, i64 1, !dbg !11
  %1849 = insertelement <4 x float> %1848, float %1754, i64 2, !dbg !11
  %1850 = insertelement <4 x float> %1849, float %1755, i64 3, !dbg !11
  %1851 = select <4 x i1> %1339, <4 x float> %1846, <4 x float> %1850, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1807, !dbg !52
  %normalized106 = fmul <4 x float> %row.reciprocal105, %1851, !dbg !52
  %1852 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !53
  %1853 = extractelement <4 x half> %1852, i64 3, !dbg !53
  %1854 = extractelement <4 x half> %1852, i64 2, !dbg !53
  %1855 = extractelement <4 x half> %1852, i64 1, !dbg !53
  %1856 = extractelement <4 x half> %1852, i64 0, !dbg !53
  %1857 = bitcast i32 %1736 to float, !dbg !11
  %1858 = bitcast i32 %1739 to float, !dbg !11
  %1859 = insertelement <4 x float> poison, float %1726, i64 0, !dbg !11
  %1860 = insertelement <4 x float> %1859, float %1727, i64 1, !dbg !11
  %1861 = insertelement <4 x float> %1860, float %1857, i64 2, !dbg !11
  %1862 = insertelement <4 x float> %1861, float %1858, i64 3, !dbg !11
  %1863 = insertelement <4 x float> poison, float %1857, i64 0, !dbg !11
  %1864 = insertelement <4 x float> %1863, float %1858, i64 1, !dbg !11
  %1865 = insertelement <4 x float> %1864, float %1732, i64 2, !dbg !11
  %1866 = insertelement <4 x float> %1865, float %1733, i64 3, !dbg !11
  %1867 = select <4 x i1> %1339, <4 x float> %1862, <4 x float> %1866, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %1867, !dbg !52
  %1868 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !53
  %1869 = extractelement <4 x half> %1868, i64 3, !dbg !53
  %1870 = extractelement <4 x half> %1868, i64 2, !dbg !53
  %1871 = extractelement <4 x half> %1868, i64 1, !dbg !53
  %1872 = extractelement <4 x half> %1868, i64 0, !dbg !53
  %address.thread1041 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !54, !range !9, !frisk.register_lifetimes !5
  %address.local1042 = zext nneg i32 %address.thread1041 to i64, !dbg !8
  %address.local1044 = lshr i64 %address.local1042, 2, !dbg !15
  %address.local1045 = and i64 %address.local1044, 48, !dbg !15
  %address.local1047 = and i64 %address.local1042, 240, !dbg !54
  %address.local1046 = add i64 %31, %address.local1042, !dbg !54
  %address.local1049 = sub i64 %address.local1046, %address.local1047, !dbg !54
  %address.local1050 = add i64 %address.local1049, %address.local1045, !dbg !54
  %1873 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %40, !dbg !54
  %address.local1053.idx = shl nuw nsw i64 %address.local1050, 8, !dbg !54
  %address.local1053 = getelementptr inbounds nuw i8, ptr addrspace(1) %1873, i64 %address.local1053.idx, !dbg !54
  %address.local1055 = and i64 %address.local1044, 12, !dbg !10
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1055, !dbg !54
  store half %1872, ptr addrspace(1) %address.local1056, align 2, !dbg !54
  %address.local1058 = or disjoint i64 %address.local1055, 1, !dbg !54
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1058, !dbg !54
  store half %1871, ptr addrspace(1) %address.local1059, align 2, !dbg !54
  %address.local1061 = or disjoint i64 %address.local1055, 2, !dbg !54
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1061, !dbg !54
  store half %1870, ptr addrspace(1) %address.local1062, align 2, !dbg !54
  %address.local1064 = or disjoint i64 %address.local1055, 3, !dbg !54
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1064, !dbg !54
  store half %1869, ptr addrspace(1) %address.local1065, align 2, !dbg !54
  %address.local1067 = or disjoint i64 %address.local1055, 16, !dbg !54
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1067, !dbg !54
  store half %1856, ptr addrspace(1) %address.local1068, align 2, !dbg !54
  %address.local1070 = or disjoint i64 %address.local1055, 17, !dbg !54
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1070, !dbg !54
  store half %1855, ptr addrspace(1) %address.local1071, align 2, !dbg !54
  %address.local1073 = or disjoint i64 %address.local1055, 18, !dbg !54
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1073, !dbg !54
  store half %1854, ptr addrspace(1) %address.local1074, align 2, !dbg !54
  %address.local1076 = or disjoint i64 %address.local1055, 19, !dbg !54
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1076, !dbg !54
  store half %1853, ptr addrspace(1) %address.local1077, align 2, !dbg !54
  %address.local1081 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1053, i64 16384, !dbg !54
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1055, !dbg !54
  store half %1840, ptr addrspace(1) %address.local1082, align 2, !dbg !54
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1058, !dbg !54
  store half %1839, ptr addrspace(1) %address.local1084, align 2, !dbg !54
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !54
  store half %1838, ptr addrspace(1) %address.local1086, align 2, !dbg !54
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1064, !dbg !54
  store half %1837, ptr addrspace(1) %address.local1088, align 2, !dbg !54
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1067, !dbg !54
  store half %1824, ptr addrspace(1) %address.local1090, align 2, !dbg !54
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1070, !dbg !54
  store half %1823, ptr addrspace(1) %address.local1092, align 2, !dbg !54
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1073, !dbg !54
  store half %1822, ptr addrspace(1) %address.local1094, align 2, !dbg !54
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1076, !dbg !54
  store half %1821, ptr addrspace(1) %address.local1096, align 2, !dbg !54
  %normalized109 = fmul <4 x float> %row.reciprocal, %1716, !dbg !52
  %1874 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !53
  %1875 = extractelement <4 x half> %1874, i64 3, !dbg !53
  %1876 = extractelement <4 x half> %1874, i64 2, !dbg !53
  %1877 = extractelement <4 x half> %1874, i64 1, !dbg !53
  %1878 = extractelement <4 x half> %1874, i64 0, !dbg !53
  %normalized110 = fmul <4 x float> %row.reciprocal, %1683, !dbg !52
  %1879 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !53
  %1880 = extractelement <4 x half> %1879, i64 3, !dbg !53
  %1881 = extractelement <4 x half> %1879, i64 2, !dbg !53
  %1882 = extractelement <4 x half> %1879, i64 1, !dbg !53
  %1883 = extractelement <4 x half> %1879, i64 0, !dbg !53
  %normalized112 = fmul <4 x float> %row.reciprocal105, %1649, !dbg !52
  %1884 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !53
  %1885 = extractelement <4 x half> %1884, i64 3, !dbg !53
  %1886 = extractelement <4 x half> %1884, i64 2, !dbg !53
  %1887 = extractelement <4 x half> %1884, i64 1, !dbg !53
  %1888 = extractelement <4 x half> %1884, i64 0, !dbg !53
  %normalized113 = fmul <4 x float> %row.reciprocal105, %1616, !dbg !52
  %1889 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !53
  %1890 = extractelement <4 x half> %1889, i64 3, !dbg !53
  %1891 = extractelement <4 x half> %1889, i64 2, !dbg !53
  %1892 = extractelement <4 x half> %1889, i64 1, !dbg !53
  %1893 = extractelement <4 x half> %1889, i64 0, !dbg !53
  %address.local1112 = or disjoint i64 %address.local1055, 32, !dbg !54
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1112, !dbg !54
  store half %1893, ptr addrspace(1) %address.local1113, align 2, !dbg !54
  %address.local1115 = or disjoint i64 %address.local1055, 33, !dbg !54
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1115, !dbg !54
  store half %1892, ptr addrspace(1) %address.local1116, align 2, !dbg !54
  %address.local1118 = or disjoint i64 %address.local1055, 34, !dbg !54
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1118, !dbg !54
  store half %1891, ptr addrspace(1) %address.local1119, align 2, !dbg !54
  %address.local1121 = or disjoint i64 %address.local1055, 35, !dbg !54
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1121, !dbg !54
  store half %1890, ptr addrspace(1) %address.local1122, align 2, !dbg !54
  %address.local1124 = or disjoint i64 %address.local1055, 48, !dbg !54
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1124, !dbg !54
  store half %1888, ptr addrspace(1) %address.local1125, align 2, !dbg !54
  %address.local1127 = or disjoint i64 %address.local1055, 49, !dbg !54
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1127, !dbg !54
  store half %1887, ptr addrspace(1) %address.local1128, align 2, !dbg !54
  %address.local1130 = or disjoint i64 %address.local1055, 50, !dbg !54
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1130, !dbg !54
  store half %1886, ptr addrspace(1) %address.local1131, align 2, !dbg !54
  %address.local1133 = or i64 %address.local1044, 51, !dbg !54
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1133, !dbg !54
  store half %1885, ptr addrspace(1) %address.local1134, align 2, !dbg !54
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1112, !dbg !54
  store half %1883, ptr addrspace(1) %address.local1139, align 2, !dbg !54
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1115, !dbg !54
  store half %1882, ptr addrspace(1) %address.local1141, align 2, !dbg !54
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1118, !dbg !54
  store half %1881, ptr addrspace(1) %address.local1143, align 2, !dbg !54
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1121, !dbg !54
  store half %1880, ptr addrspace(1) %address.local1145, align 2, !dbg !54
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1124, !dbg !54
  store half %1878, ptr addrspace(1) %address.local1147, align 2, !dbg !54
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1127, !dbg !54
  store half %1877, ptr addrspace(1) %address.local1149, align 2, !dbg !54
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1130, !dbg !54
  store half %1876, ptr addrspace(1) %address.local1151, align 2, !dbg !54
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1133, !dbg !54
  store half %1875, ptr addrspace(1) %address.local1153, align 2, !dbg !54
  %normalized115 = fmul <4 x float> %row.reciprocal, %1582, !dbg !52
  %1894 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !53
  %1895 = extractelement <4 x half> %1894, i64 3, !dbg !53
  %1896 = extractelement <4 x half> %1894, i64 2, !dbg !53
  %1897 = extractelement <4 x half> %1894, i64 1, !dbg !53
  %1898 = extractelement <4 x half> %1894, i64 0, !dbg !53
  %normalized116 = fmul <4 x float> %row.reciprocal, %1549, !dbg !52
  %1899 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !53
  %1900 = extractelement <4 x half> %1899, i64 3, !dbg !53
  %1901 = extractelement <4 x half> %1899, i64 2, !dbg !53
  %1902 = extractelement <4 x half> %1899, i64 1, !dbg !53
  %1903 = extractelement <4 x half> %1899, i64 0, !dbg !53
  %normalized118 = fmul <4 x float> %row.reciprocal105, %1515, !dbg !52
  %1904 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !53
  %1905 = extractelement <4 x half> %1904, i64 3, !dbg !53
  %1906 = extractelement <4 x half> %1904, i64 2, !dbg !53
  %1907 = extractelement <4 x half> %1904, i64 1, !dbg !53
  %1908 = extractelement <4 x half> %1904, i64 0, !dbg !53
  %normalized119 = fmul <4 x float> %row.reciprocal105, %1482, !dbg !52
  %1909 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !53
  %1910 = extractelement <4 x half> %1909, i64 3, !dbg !53
  %1911 = extractelement <4 x half> %1909, i64 2, !dbg !53
  %1912 = extractelement <4 x half> %1909, i64 1, !dbg !53
  %1913 = extractelement <4 x half> %1909, i64 0, !dbg !53
  %address.local1169 = or disjoint i64 %address.local1055, 64, !dbg !54
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1169, !dbg !54
  store half %1913, ptr addrspace(1) %address.local1170, align 2, !dbg !54
  %address.local1172 = or disjoint i64 %address.local1055, 65, !dbg !54
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1172, !dbg !54
  store half %1912, ptr addrspace(1) %address.local1173, align 2, !dbg !54
  %address.local1175 = or disjoint i64 %address.local1055, 66, !dbg !54
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1175, !dbg !54
  store half %1911, ptr addrspace(1) %address.local1176, align 2, !dbg !54
  %address.local1178 = or disjoint i64 %address.local1055, 67, !dbg !54
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1178, !dbg !54
  store half %1910, ptr addrspace(1) %address.local1179, align 2, !dbg !54
  %address.local1181 = or disjoint i64 %address.local1055, 80, !dbg !54
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1181, !dbg !54
  store half %1908, ptr addrspace(1) %address.local1182, align 2, !dbg !54
  %address.local1184 = or disjoint i64 %address.local1055, 81, !dbg !54
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1184, !dbg !54
  store half %1907, ptr addrspace(1) %address.local1185, align 2, !dbg !54
  %address.local1187 = or disjoint i64 %address.local1055, 82, !dbg !54
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1187, !dbg !54
  store half %1906, ptr addrspace(1) %address.local1188, align 2, !dbg !54
  %address.local1190 = or disjoint i64 %address.local1055, 83, !dbg !54
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1190, !dbg !54
  store half %1905, ptr addrspace(1) %address.local1191, align 2, !dbg !54
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1169, !dbg !54
  store half %1903, ptr addrspace(1) %address.local1196, align 2, !dbg !54
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1172, !dbg !54
  store half %1902, ptr addrspace(1) %address.local1198, align 2, !dbg !54
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1175, !dbg !54
  store half %1901, ptr addrspace(1) %address.local1200, align 2, !dbg !54
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1178, !dbg !54
  store half %1900, ptr addrspace(1) %address.local1202, align 2, !dbg !54
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1181, !dbg !54
  store half %1898, ptr addrspace(1) %address.local1204, align 2, !dbg !54
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1184, !dbg !54
  store half %1897, ptr addrspace(1) %address.local1206, align 2, !dbg !54
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1187, !dbg !54
  store half %1896, ptr addrspace(1) %address.local1208, align 2, !dbg !54
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1190, !dbg !54
  store half %1895, ptr addrspace(1) %address.local1210, align 2, !dbg !54
  %normalized121 = fmul <4 x float> %row.reciprocal, %1448, !dbg !52
  %1914 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !53
  %1915 = extractelement <4 x half> %1914, i64 3, !dbg !53
  %1916 = extractelement <4 x half> %1914, i64 2, !dbg !53
  %1917 = extractelement <4 x half> %1914, i64 1, !dbg !53
  %1918 = extractelement <4 x half> %1914, i64 0, !dbg !53
  %normalized122 = fmul <4 x float> %row.reciprocal, %1415, !dbg !52
  %1919 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !53
  %1920 = extractelement <4 x half> %1919, i64 3, !dbg !53
  %1921 = extractelement <4 x half> %1919, i64 2, !dbg !53
  %1922 = extractelement <4 x half> %1919, i64 1, !dbg !53
  %1923 = extractelement <4 x half> %1919, i64 0, !dbg !53
  %normalized124 = fmul <4 x float> %row.reciprocal105, %1381, !dbg !52
  %1924 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !53
  %1925 = extractelement <4 x half> %1924, i64 3, !dbg !53
  %1926 = extractelement <4 x half> %1924, i64 2, !dbg !53
  %1927 = extractelement <4 x half> %1924, i64 1, !dbg !53
  %1928 = extractelement <4 x half> %1924, i64 0, !dbg !53
  %normalized125 = fmul <4 x float> %row.reciprocal105, %1348, !dbg !52
  %1929 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !53
  %1930 = extractelement <4 x half> %1929, i64 3, !dbg !53
  %1931 = extractelement <4 x half> %1929, i64 2, !dbg !53
  %1932 = extractelement <4 x half> %1929, i64 1, !dbg !53
  %1933 = extractelement <4 x half> %1929, i64 0, !dbg !53
  %address.local1226 = or disjoint i64 %address.local1055, 96, !dbg !54
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1226, !dbg !54
  store half %1933, ptr addrspace(1) %address.local1227, align 2, !dbg !54
  %address.local1229 = or disjoint i64 %address.local1055, 97, !dbg !54
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1229, !dbg !54
  store half %1932, ptr addrspace(1) %address.local1230, align 2, !dbg !54
  %address.local1232 = or disjoint i64 %address.local1055, 98, !dbg !54
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1232, !dbg !54
  store half %1931, ptr addrspace(1) %address.local1233, align 2, !dbg !54
  %address.local1235 = or disjoint i64 %address.local1055, 99, !dbg !54
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1235, !dbg !54
  store half %1930, ptr addrspace(1) %address.local1236, align 2, !dbg !54
  %address.local1238 = or disjoint i64 %address.local1055, 112, !dbg !54
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1238, !dbg !54
  store half %1928, ptr addrspace(1) %address.local1239, align 2, !dbg !54
  %address.local1241 = or disjoint i64 %address.local1055, 113, !dbg !54
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1241, !dbg !54
  store half %1927, ptr addrspace(1) %address.local1242, align 2, !dbg !54
  %address.local1244 = or disjoint i64 %address.local1055, 114, !dbg !54
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1244, !dbg !54
  store half %1926, ptr addrspace(1) %address.local1245, align 2, !dbg !54
  %address.local1247 = or i64 %address.local1044, 115, !dbg !54
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1247, !dbg !54
  store half %1925, ptr addrspace(1) %address.local1248, align 2, !dbg !54
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1226, !dbg !54
  store half %1923, ptr addrspace(1) %address.local1253, align 2, !dbg !54
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1229, !dbg !54
  store half %1922, ptr addrspace(1) %address.local1255, align 2, !dbg !54
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1232, !dbg !54
  store half %1921, ptr addrspace(1) %address.local1257, align 2, !dbg !54
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1235, !dbg !54
  store half %1920, ptr addrspace(1) %address.local1259, align 2, !dbg !54
  %address.local1261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1238, !dbg !54
  store half %1918, ptr addrspace(1) %address.local1261, align 2, !dbg !54
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1241, !dbg !54
  store half %1917, ptr addrspace(1) %address.local1263, align 2, !dbg !54
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1244, !dbg !54
  store half %1916, ptr addrspace(1) %address.local1265, align 2, !dbg !54
  %address.local1267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1247, !dbg !54
  store half %1915, ptr addrspace(1) %address.local1267, align 2, !dbg !54
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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
!16 = !{i32 0}
!17 = !{i32 1}
!18 = !{i32 2}
!19 = !{i32 3}
!20 = !{i32 4}
!21 = !{i32 5}
!22 = !{i32 6}
!23 = !{i32 7}
!24 = !{i32 8}
!25 = !{i32 9}
!26 = !{i32 10}
!27 = !{i32 11}
!28 = !{i32 12}
!29 = !{i32 13}
!30 = !{i32 14}
!31 = !{i32 15}
!32 = !{i32 16}
!33 = !{i32 17}
!34 = !{i32 18}
!35 = !{i32 19}
!36 = !{i32 20}
!37 = !{i32 21}
!38 = !{i32 22}
!39 = !DILocation(line: 101, column: 20, scope: !3)
!40 = !DILocation(line: 74, column: 23, scope: !3)
!41 = !DILocation(line: 89, column: 20, scope: !3)
!42 = !DILocation(line: 90, column: 22, scope: !3)
!43 = !DILocation(line: 91, column: 23, scope: !3)
!44 = !DILocation(line: 92, column: 20, scope: !3)
!45 = !DILocation(line: 103, column: 22, scope: !3)
!46 = !DILocation(line: 93, column: 22, scope: !3)
!47 = !DILocation(line: 102, column: 23, scope: !3)
!48 = !{i64 8192}
!49 = !DILocation(line: 94, column: 22, scope: !3)
!50 = !DILocation(line: 96, column: 16, scope: !3)
!51 = !DILocation(line: 95, column: 20, scope: !3)
!52 = !DILocation(line: 107, column: 23, scope: !3)
!53 = !DILocation(line: 108, column: 19, scope: !3)
!54 = !DILocation(line: 109, column: 9, scope: !3)
