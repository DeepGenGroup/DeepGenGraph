; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/causal_mask_lowering/mask_i32_pk_rtz.ll'
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
  %115 = lshr i64 %33, 2, !dbg !15
  %116 = and i64 %115, 48, !dbg !15
  %117 = and i64 %33, 783, !dbg !15
  %118 = or disjoint i64 %117, %116, !dbg !15
  %119 = add i64 %118, %29, !dbg !15
  %120 = and i64 %33, 48, !dbg !15
  %121 = trunc i64 %119 to i32, !dbg !15
  %122 = icmp ult i64 %11, 1024, !dbg !11
  br i1 %122, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !11

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %.pre22 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !11
  %.pre23 = add nuw nsw i32 %.pre22, 64, !dbg !11
  %.pre24 = xor i32 %.pre22, 16, !dbg !11
  %.pre26 = and i32 %.pre23, 192, !dbg !11
  %.pre28 = xor i32 %.pre22, 32, !dbg !11
  br label %._crit_edge, !dbg !11

.lr.ph:                                           ; preds = %4
  %123 = add i32 %121, 65, !dbg !15
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
  %135 = insertelement <4 x i32> poison, i32 %123, i64 0, !dbg !15
  %136 = shufflevector <4 x i32> %135, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !15
  %137 = insertelement <4 x i32> poison, i32 %121, i64 0, !dbg !15
  %138 = shufflevector <4 x i32> %137, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !15
  %139 = insertelement <2 x i64> poison, i64 %130, i64 1
  %invariant.op = sub i32 72, %kdma.intra, !dbg !11
  %invariant.op76 = sub i32 72, %kdma.intra, !dbg !11
  %invariant.op77 = sub i32 136, %kdma.intra, !dbg !11
  %invariant.op78 = sub i32 136, %kdma.intra, !dbg !11
  %invariant.op79 = sub i32 200, %kdma.intra, !dbg !11
  %invariant.op80 = sub i32 200, %kdma.intra, !dbg !11
  br label %140, !dbg !11

140:                                              ; preds = %.lr.ph, %140
  %141 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1205, %140 ]
  %142 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1201, %140 ]
  %143 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1197, %140 ]
  %144 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1193, %140 ]
  %145 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %957, %140 ]
  %146 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1191, %140 ]
  %147 = phi i64 [ 0, %.lr.ph ], [ %1206, %140 ]
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
  %160 = insertelement <2 x i64> %139, i64 %150, i64 0, !dbg !13
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
  %431 = or disjoint i64 %152, %120, !dbg !15
  %432 = trunc i64 %431 to i32, !dbg !15
  %433 = insertelement <4 x i32> poison, i32 %432, i64 0, !dbg !15
  %434 = shufflevector <4 x i32> %433, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !15
  %435 = or disjoint <4 x i32> %434, <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %436 = insertelement <4 x i32> %137, i32 %432, i64 1, !dbg !15
  %437 = shufflevector <4 x i32> %436, <4 x i32> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 0>, !dbg !15
  %438 = insertelement <4 x i32> %435, i32 %432, i64 0, !dbg !15
  %439 = insertelement <4 x i32> %438, i32 %121, i64 1, !dbg !15
  %440 = icmp ult <4 x i32> %437, %439, !dbg !15
  %441 = select <4 x i1> %440, <4 x float> <float -inf, float 0.000000e+00, float -inf, float -inf>, <4 x float> <float 0.000000e+00, float -inf, float 0.000000e+00, float 0.000000e+00>, !dbg !40
  %442 = icmp ugt <4 x i32> %136, %435, !dbg !15
  %443 = select <4 x i1> %442, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !40
  %444 = or disjoint <4 x i32> %434, <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %445 = icmp ugt <4 x i32> %444, %138, !dbg !15
  %446 = select <4 x i1> %445, <4 x float> splat (float -inf), <4 x float> zeroinitializer, !dbg !40
  %447 = icmp ugt <4 x i32> %136, %444, !dbg !15
  %448 = select <4 x i1> %447, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !40
  %449 = or disjoint <4 x i32> %434, <i32 8, i32 9, i32 10, i32 11>, !dbg !15
  %450 = icmp ugt <4 x i32> %449, %138, !dbg !15
  %451 = select <4 x i1> %450, <4 x float> splat (float -inf), <4 x float> zeroinitializer, !dbg !40
  %452 = icmp ugt <4 x i32> %136, %449, !dbg !15
  %453 = select <4 x i1> %452, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !40
  %454 = or disjoint <4 x i32> %434, <i32 12, i32 13, i32 14, i32 15>, !dbg !15
  %455 = icmp ugt <4 x i32> %454, %138, !dbg !15
  %456 = select <4 x i1> %455, <4 x float> splat (float -inf), <4 x float> zeroinitializer, !dbg !40
  %457 = icmp ugt <4 x i32> %136, %454, !dbg !15
  %458 = select <4 x i1> %457, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !40
  %459 = or disjoint <4 x i32> %434, <i32 64, i32 65, i32 66, i32 67>, !dbg !15
  %460 = icmp ugt <4 x i32> %459, %138, !dbg !15
  %461 = select <4 x i1> %460, <4 x float> splat (float -inf), <4 x float> zeroinitializer, !dbg !40
  %462 = icmp ugt <4 x i32> %136, %459, !dbg !15
  %463 = select <4 x i1> %462, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !40
  %464 = or disjoint <4 x i32> %434, <i32 68, i32 69, i32 70, i32 71>, !dbg !15
  %465 = icmp ugt <4 x i32> %464, %138, !dbg !15
  %466 = select <4 x i1> %465, <4 x float> splat (float -inf), <4 x float> zeroinitializer, !dbg !40
  %467 = icmp ugt <4 x i32> %136, %464, !dbg !15
  %468 = select <4 x i1> %467, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !40
  %469 = or disjoint <4 x i32> %434, <i32 72, i32 73, i32 74, i32 75>, !dbg !15
  %470 = icmp ugt <4 x i32> %469, %138, !dbg !15
  %471 = select <4 x i1> %470, <4 x float> splat (float -inf), <4 x float> zeroinitializer, !dbg !40
  %472 = icmp ugt <4 x i32> %136, %469, !dbg !15
  %473 = select <4 x i1> %472, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !40
  %474 = or disjoint <4 x i32> %434, <i32 76, i32 77, i32 78, i32 79>, !dbg !15
  %475 = icmp ugt <4 x i32> %474, %138, !dbg !15
  %476 = select <4 x i1> %475, <4 x float> splat (float -inf), <4 x float> zeroinitializer, !dbg !40
  %477 = icmp ugt <4 x i32> %136, %474, !dbg !15
  %478 = select <4 x i1> %477, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !40
  %479 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %424, <4 x float> %430) #8, !dbg !14
  %480 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %426, <4 x float> %386) #8, !dbg !14
  %481 = fadd <4 x float> %441, %479, !dbg !41
  %482 = extractelement <4 x float> %481, i64 0, !dbg !41
  %483 = tail call float @llvm.maxnum.f32(float %482, float -inf), !dbg !42, !frisk.local_reduction !5
  %484 = extractelement <4 x float> %481, i64 1, !dbg !41
  %485 = extractelement <4 x float> %481, i64 2, !dbg !41
  %486 = extractelement <4 x float> %481, i64 3, !dbg !41
  %487 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %428, <4 x float> %480) #8, !dbg !14
  %488 = load <4 x half>, ptr addrspace(3) %kdma.gep1914, align 8, !dbg !14, !frisk.k_tail_pair !20
  %489 = load <4 x half>, ptr addrspace(3) %kdma.gep1918, align 8, !dbg !14, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %490 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %423, <4 x float> %388) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41, !frisk.k_tail_prefetch !5
  %491 = fadd <4 x float> %446, %487, !dbg !41
  %492 = extractelement <4 x float> %491, i64 0, !dbg !41
  %local.reduce218 = tail call float @llvm.maxnum.f32(float %483, float %492), !dbg !42, !frisk.local_reduction !5
  %493 = extractelement <4 x float> %491, i64 1, !dbg !41
  %local.reduce219 = tail call float @llvm.maxnum.f32(float %484, float %493), !dbg !42, !frisk.local_reduction !5
  %494 = extractelement <4 x float> %491, i64 2, !dbg !41
  %local.reduce220 = tail call float @llvm.maxnum.f32(float %485, float %494), !dbg !42, !frisk.local_reduction !5
  %495 = extractelement <4 x float> %491, i64 3, !dbg !41
  %local.reduce221 = tail call float @llvm.maxnum.f32(float %486, float %495), !dbg !42, !frisk.local_reduction !5
  %496 = load <4 x half>, ptr addrspace(3) %kdma.gep1915, align 8, !dbg !14, !frisk.k_tail_pair !21
  %497 = load <4 x half>, ptr addrspace(3) %kdma.gep1919, align 8, !dbg !14, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %498 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %425, <4 x float> %490) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %499 = load <4 x half>, ptr addrspace(3) %kdma.gep1916, align 8, !dbg !14, !frisk.k_tail_pair !22
  %500 = load <4 x half>, ptr addrspace(3) %kdma.gep1920, align 8, !dbg !14, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %501 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %427, <4 x float> %390) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41, !frisk.k_tail_prefetch !5
  %502 = fadd <4 x float> %451, %498, !dbg !41
  %503 = extractelement <4 x float> %502, i64 0, !dbg !41
  %local.reduce222 = tail call float @llvm.maxnum.f32(float %local.reduce218, float %503), !dbg !42, !frisk.local_reduction !5
  %504 = extractelement <4 x float> %502, i64 1, !dbg !41
  %local.reduce223 = tail call float @llvm.maxnum.f32(float %local.reduce219, float %504), !dbg !42, !frisk.local_reduction !5
  %505 = extractelement <4 x float> %502, i64 2, !dbg !41
  %local.reduce224 = tail call float @llvm.maxnum.f32(float %local.reduce220, float %505), !dbg !42, !frisk.local_reduction !5
  %506 = extractelement <4 x float> %502, i64 3, !dbg !41
  %local.reduce225 = tail call float @llvm.maxnum.f32(float %local.reduce221, float %506), !dbg !42, !frisk.local_reduction !5
  %507 = load <4 x half>, ptr addrspace(3) %kdma.gep1917, align 8, !dbg !14, !frisk.k_tail_pair !23
  %508 = load <4 x half>, ptr addrspace(3) %kdma.gep1921, align 8, !dbg !14, !frisk.k_tail_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %509 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %429, <4 x float> %501) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %510 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %488, <4 x float> %392) #8, !dbg !14
  %511 = fadd <4 x float> %456, %509, !dbg !41
  %512 = extractelement <4 x float> %511, i64 0, !dbg !41
  %local.reduce226 = tail call float @llvm.maxnum.f32(float %local.reduce222, float %512), !dbg !42, !frisk.local_reduction !5
  %513 = extractelement <4 x float> %511, i64 1, !dbg !41
  %local.reduce227 = tail call float @llvm.maxnum.f32(float %local.reduce223, float %513), !dbg !42, !frisk.local_reduction !5
  %514 = extractelement <4 x float> %511, i64 2, !dbg !41
  %local.reduce228 = tail call float @llvm.maxnum.f32(float %local.reduce224, float %514), !dbg !42, !frisk.local_reduction !5
  %515 = extractelement <4 x float> %511, i64 3, !dbg !41
  %local.reduce229 = tail call float @llvm.maxnum.f32(float %local.reduce225, float %515), !dbg !42, !frisk.local_reduction !5
  %516 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %496, <4 x float> %510) #8, !dbg !14
  %517 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %499, <4 x float> %394) #8, !dbg !14
  %518 = fadd <4 x float> %461, %516, !dbg !41
  %519 = extractelement <4 x float> %518, i64 0, !dbg !41
  %local.reduce230 = tail call float @llvm.maxnum.f32(float %local.reduce226, float %519), !dbg !42, !frisk.local_reduction !5
  %520 = extractelement <4 x float> %518, i64 1, !dbg !41
  %local.reduce231 = tail call float @llvm.maxnum.f32(float %local.reduce227, float %520), !dbg !42, !frisk.local_reduction !5
  %521 = extractelement <4 x float> %518, i64 2, !dbg !41
  %local.reduce232 = tail call float @llvm.maxnum.f32(float %local.reduce228, float %521), !dbg !42, !frisk.local_reduction !5
  %522 = extractelement <4 x float> %518, i64 3, !dbg !41
  %local.reduce233 = tail call float @llvm.maxnum.f32(float %local.reduce229, float %522), !dbg !42, !frisk.local_reduction !5
  %523 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %507, <4 x float> %517) #8, !dbg !14
  %524 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %489, <4 x float> %396) #8, !dbg !14
  %525 = fadd <4 x float> %466, %523, !dbg !41
  %526 = extractelement <4 x float> %525, i64 0, !dbg !41
  %local.reduce234 = tail call float @llvm.maxnum.f32(float %local.reduce230, float %526), !dbg !42, !frisk.local_reduction !5
  %527 = extractelement <4 x float> %525, i64 1, !dbg !41
  %local.reduce235 = tail call float @llvm.maxnum.f32(float %local.reduce231, float %527), !dbg !42, !frisk.local_reduction !5
  %528 = extractelement <4 x float> %525, i64 2, !dbg !41
  %local.reduce236 = tail call float @llvm.maxnum.f32(float %local.reduce232, float %528), !dbg !42, !frisk.local_reduction !5
  %529 = extractelement <4 x float> %525, i64 3, !dbg !41
  %local.reduce237 = tail call float @llvm.maxnum.f32(float %local.reduce233, float %529), !dbg !42, !frisk.local_reduction !5
  %530 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %497, <4 x float> %524) #8, !dbg !14
  %531 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %500, <4 x float> %398) #8, !dbg !14
  %532 = fadd <4 x float> %471, %530, !dbg !41
  %533 = extractelement <4 x float> %532, i64 0, !dbg !41
  %local.reduce238 = tail call float @llvm.maxnum.f32(float %local.reduce234, float %533), !dbg !42, !frisk.local_reduction !5
  %534 = extractelement <4 x float> %532, i64 1, !dbg !41
  %local.reduce239 = tail call float @llvm.maxnum.f32(float %local.reduce235, float %534), !dbg !42, !frisk.local_reduction !5
  %535 = extractelement <4 x float> %532, i64 2, !dbg !41
  %local.reduce240 = tail call float @llvm.maxnum.f32(float %local.reduce236, float %535), !dbg !42, !frisk.local_reduction !5
  %536 = extractelement <4 x float> %532, i64 3, !dbg !41
  %local.reduce241 = tail call float @llvm.maxnum.f32(float %local.reduce237, float %536), !dbg !42, !frisk.local_reduction !5
  %537 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %508, <4 x float> %531) #8, !dbg !14
  %538 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %422, <4 x float> %400) #8, !dbg !14
  %539 = fadd <4 x float> %476, %537, !dbg !41
  %540 = extractelement <4 x float> %539, i64 0, !dbg !41
  %local.reduce242 = tail call float @llvm.maxnum.f32(float %local.reduce238, float %540), !dbg !42, !frisk.local_reduction !5
  %541 = extractelement <4 x float> %539, i64 1, !dbg !41
  %local.reduce243 = tail call float @llvm.maxnum.f32(float %local.reduce239, float %541), !dbg !42, !frisk.local_reduction !5
  %542 = extractelement <4 x float> %539, i64 2, !dbg !41
  %local.reduce244 = tail call float @llvm.maxnum.f32(float %local.reduce240, float %542), !dbg !42, !frisk.local_reduction !5
  %543 = extractelement <4 x float> %539, i64 3, !dbg !41
  %local.reduce245 = tail call float @llvm.maxnum.f32(float %local.reduce241, float %543), !dbg !42, !frisk.local_reduction !5
  %local.reduce246 = tail call float @llvm.maxnum.f32(float %local.reduce242, float %local.reduce243), !dbg !42, !frisk.local_reduction !5
  %local.reduce247 = tail call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !42, !frisk.local_reduction !5
  %544 = tail call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !42, !frisk.local_reduction !5
  %545 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %424, <4 x float> %538) #8, !dbg !14
  %546 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %426, <4 x float> %402) #8, !dbg !14
  %547 = fadd <4 x float> %443, %545, !dbg !41
  %548 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %428, <4 x float> %546) #8, !dbg !14
  %549 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %423, <4 x float> %404) #8, !dbg !14
  %550 = fadd <4 x float> %448, %548, !dbg !41
  %551 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %425, <4 x float> %549) #8, !dbg !14
  %552 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %427, <4 x float> %406) #8, !dbg !14
  %553 = fadd <4 x float> %453, %551, !dbg !41
  %554 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %429, <4 x float> %552) #8, !dbg !14
  %555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %488, <4 x float> %408) #8, !dbg !14
  %556 = fadd <4 x float> %458, %554, !dbg !41
  %557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %496, <4 x float> %555) #8, !dbg !14
  %558 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %499, <4 x float> %410) #8, !dbg !14
  %559 = fadd <4 x float> %463, %557, !dbg !41
  %560 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %507, <4 x float> %558) #8, !dbg !14
  %561 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %489, <4 x float> %412) #8, !dbg !14
  %562 = fadd <4 x float> %468, %560, !dbg !41
  %563 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %497, <4 x float> %561) #8, !dbg !14
  %564 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %500, <4 x float> %414) #8, !dbg !14
  %565 = fadd <4 x float> %473, %563, !dbg !41
  %566 = extractelement <4 x float> %547, i64 0, !dbg !41
  %567 = extractelement <4 x float> %547, i64 1, !dbg !41
  %568 = extractelement <4 x float> %547, i64 2, !dbg !41
  %569 = extractelement <4 x float> %547, i64 3, !dbg !41
  %570 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %508, <4 x float> %564) #8, !dbg !14
  %571 = extractelement <4 x float> %550, i64 0, !dbg !41
  %572 = extractelement <4 x float> %550, i64 1, !dbg !41
  %573 = extractelement <4 x float> %550, i64 2, !dbg !41
  %574 = extractelement <4 x float> %550, i64 3, !dbg !41
  %575 = extractelement <4 x float> %553, i64 0, !dbg !41
  %576 = extractelement <4 x float> %553, i64 1, !dbg !41
  %577 = extractelement <4 x float> %553, i64 2, !dbg !41
  %578 = extractelement <4 x float> %553, i64 3, !dbg !41
  %579 = fadd <4 x float> %478, %570, !dbg !41
  %580 = extractelement <4 x float> %556, i64 0, !dbg !41
  %581 = extractelement <4 x float> %556, i64 1, !dbg !41
  %582 = extractelement <4 x float> %556, i64 2, !dbg !41
  %583 = extractelement <4 x float> %556, i64 3, !dbg !41
  %584 = extractelement <4 x float> %559, i64 0, !dbg !41
  %585 = extractelement <4 x float> %559, i64 1, !dbg !41
  %586 = extractelement <4 x float> %559, i64 2, !dbg !41
  %587 = extractelement <4 x float> %559, i64 3, !dbg !41
  %588 = extractelement <4 x float> %562, i64 0, !dbg !41
  %589 = extractelement <4 x float> %562, i64 1, !dbg !41
  %590 = extractelement <4 x float> %562, i64 2, !dbg !41
  %591 = extractelement <4 x float> %562, i64 3, !dbg !41
  %592 = extractelement <4 x float> %565, i64 0, !dbg !41
  %593 = extractelement <4 x float> %565, i64 1, !dbg !41
  %594 = extractelement <4 x float> %565, i64 2, !dbg !41
  %595 = extractelement <4 x float> %565, i64 3, !dbg !41
  %596 = extractelement <4 x float> %579, i64 0, !dbg !41
  %597 = extractelement <4 x float> %579, i64 1, !dbg !41
  %598 = extractelement <4 x float> %579, i64 2, !dbg !41
  %599 = extractelement <4 x float> %579, i64 3, !dbg !41
  %600 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !42
  %601 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %600), !dbg !42
  %602 = add nuw nsw i32 %601, 64, !dbg !42
  %603 = xor i32 %601, 16, !dbg !42
  %604 = and i32 %602, 192, !dbg !42
  %605 = icmp samesign ult i32 %603, %604, !dbg !42
  %606 = select i1 %605, i32 %603, i32 %601, !dbg !42
  %607 = shl nuw nsw i32 %606, 2, !dbg !42
  %608 = bitcast float %544 to i32, !dbg !42
  %609 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %607, i32 %608), !dbg !42
  %610 = bitcast i32 %609 to float, !dbg !42
  %611 = tail call float @llvm.maxnum.f32(float %544, float %610), !dbg !42
  %612 = xor i32 %601, 32, !dbg !42
  %613 = icmp samesign ult i32 %612, %604, !dbg !42
  %614 = select i1 %613, i32 %612, i32 %601, !dbg !42
  %615 = shl nuw nsw i32 %614, 2, !dbg !42
  %616 = bitcast float %611 to i32, !dbg !42
  %617 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %615, i32 %616), !dbg !42
  %618 = bitcast i32 %617 to float, !dbg !42
  %619 = tail call float @llvm.maxnum.f32(float %611, float %618), !dbg !42
  %620 = tail call float @llvm.maxnum.f32(float %566, float -inf), !dbg !42, !frisk.local_reduction !5
  %local.reduce187 = tail call float @llvm.maxnum.f32(float %620, float %571), !dbg !42, !frisk.local_reduction !5
  %local.reduce188 = tail call float @llvm.maxnum.f32(float %567, float %572), !dbg !42, !frisk.local_reduction !5
  %local.reduce189 = tail call float @llvm.maxnum.f32(float %568, float %573), !dbg !42, !frisk.local_reduction !5
  %local.reduce190 = tail call float @llvm.maxnum.f32(float %569, float %574), !dbg !42, !frisk.local_reduction !5
  %local.reduce191 = tail call float @llvm.maxnum.f32(float %local.reduce187, float %575), !dbg !42, !frisk.local_reduction !5
  %local.reduce192 = tail call float @llvm.maxnum.f32(float %local.reduce188, float %576), !dbg !42, !frisk.local_reduction !5
  %local.reduce193 = tail call float @llvm.maxnum.f32(float %local.reduce189, float %577), !dbg !42, !frisk.local_reduction !5
  %local.reduce194 = tail call float @llvm.maxnum.f32(float %local.reduce190, float %578), !dbg !42, !frisk.local_reduction !5
  %local.reduce195 = tail call float @llvm.maxnum.f32(float %local.reduce191, float %580), !dbg !42, !frisk.local_reduction !5
  %local.reduce196 = tail call float @llvm.maxnum.f32(float %local.reduce192, float %581), !dbg !42, !frisk.local_reduction !5
  %local.reduce197 = tail call float @llvm.maxnum.f32(float %local.reduce193, float %582), !dbg !42, !frisk.local_reduction !5
  %local.reduce198 = tail call float @llvm.maxnum.f32(float %local.reduce194, float %583), !dbg !42, !frisk.local_reduction !5
  %local.reduce199 = tail call float @llvm.maxnum.f32(float %local.reduce195, float %584), !dbg !42, !frisk.local_reduction !5
  %local.reduce200 = tail call float @llvm.maxnum.f32(float %local.reduce196, float %585), !dbg !42, !frisk.local_reduction !5
  %local.reduce201 = tail call float @llvm.maxnum.f32(float %local.reduce197, float %586), !dbg !42, !frisk.local_reduction !5
  %local.reduce202 = tail call float @llvm.maxnum.f32(float %local.reduce198, float %587), !dbg !42, !frisk.local_reduction !5
  %local.reduce203 = tail call float @llvm.maxnum.f32(float %local.reduce199, float %588), !dbg !42, !frisk.local_reduction !5
  %local.reduce204 = tail call float @llvm.maxnum.f32(float %local.reduce200, float %589), !dbg !42, !frisk.local_reduction !5
  %local.reduce205 = tail call float @llvm.maxnum.f32(float %local.reduce201, float %590), !dbg !42, !frisk.local_reduction !5
  %local.reduce206 = tail call float @llvm.maxnum.f32(float %local.reduce202, float %591), !dbg !42, !frisk.local_reduction !5
  %local.reduce207 = tail call float @llvm.maxnum.f32(float %local.reduce203, float %592), !dbg !42, !frisk.local_reduction !5
  %local.reduce208 = tail call float @llvm.maxnum.f32(float %local.reduce204, float %593), !dbg !42, !frisk.local_reduction !5
  %local.reduce209 = tail call float @llvm.maxnum.f32(float %local.reduce205, float %594), !dbg !42, !frisk.local_reduction !5
  %local.reduce210 = tail call float @llvm.maxnum.f32(float %local.reduce206, float %595), !dbg !42, !frisk.local_reduction !5
  %local.reduce211 = tail call float @llvm.maxnum.f32(float %local.reduce207, float %596), !dbg !42, !frisk.local_reduction !5
  %local.reduce212 = tail call float @llvm.maxnum.f32(float %local.reduce208, float %597), !dbg !42, !frisk.local_reduction !5
  %local.reduce213 = tail call float @llvm.maxnum.f32(float %local.reduce209, float %598), !dbg !42, !frisk.local_reduction !5
  %local.reduce214 = tail call float @llvm.maxnum.f32(float %local.reduce210, float %599), !dbg !42, !frisk.local_reduction !5
  %local.reduce215 = tail call float @llvm.maxnum.f32(float %local.reduce211, float %local.reduce212), !dbg !42, !frisk.local_reduction !5
  %local.reduce216 = tail call float @llvm.maxnum.f32(float %local.reduce213, float %local.reduce214), !dbg !42, !frisk.local_reduction !5
  %621 = tail call float @llvm.maxnum.f32(float %local.reduce215, float %local.reduce216), !dbg !42, !frisk.local_reduction !5
  %622 = bitcast float %621 to i32, !dbg !42
  %623 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %607, i32 %622), !dbg !42
  %624 = bitcast i32 %623 to float, !dbg !42
  %625 = tail call float @llvm.maxnum.f32(float %621, float %624), !dbg !42
  %626 = bitcast float %625 to i32, !dbg !42
  %627 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %615, i32 %626), !dbg !42
  %628 = bitcast i32 %627 to float, !dbg !42
  %629 = tail call float @llvm.maxnum.f32(float %625, float %628), !dbg !42
  %630 = extractvalue [2 x <1 x float>] %144, 0, !dbg !42
  %631 = insertelement <1 x float> poison, float %619, i64 0, !dbg !42
  %632 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %631, <1 x float> %630), !dbg !42
  %633 = extractvalue [2 x <1 x float>] %144, 1, !dbg !42
  %634 = insertelement <1 x float> poison, float %629, i64 0, !dbg !42
  %635 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %634, <1 x float> %633), !dbg !42
  %636 = shufflevector <1 x float> %632, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %637 = shufflevector <1 x float> %635, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %638 = fsub <1 x float> %630, %632, !dbg !44
  %639 = extractelement <1 x float> %638, i64 0, !dbg !45
  %640 = tail call float @llvm.amdgcn.exp2.f32(float %639), !dbg !45
  %641 = insertelement <4 x float> poison, float %640, i64 0, !dbg !46
  %642 = shufflevector <4 x float> %641, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %643 = extractvalue [2 x <1 x float>] %145, 0, !dbg !47
  %644 = extractelement <1 x float> %643, i64 0, !dbg !47
  %645 = fmul float %644, %640, !dbg !47
  %646 = fsub <1 x float> %633, %635, !dbg !44
  %647 = extractelement <1 x float> %646, i64 0, !dbg !45
  %648 = tail call float @llvm.amdgcn.exp2.f32(float %647), !dbg !45
  %649 = insertelement <4 x float> poison, float %648, i64 0, !dbg !46
  %650 = shufflevector <4 x float> %649, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %651 = extractvalue [2 x <1 x float>] %145, 1, !dbg !47
  %652 = extractelement <1 x float> %651, i64 0, !dbg !47
  %653 = fmul float %652, %648, !dbg !47
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
  %vdb.dma.base = add i32 %181, 8192, !dbg !48
  %654 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1535, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %vdb.dma.base1549 = add i32 %185, 8192, !dbg !48
  %655 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1549, i32 %vdb.address1544, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %address.local996 = shl nuw nsw i32 %address.thread565, 3, !dbg !48
  %address.local997 = and i32 %address.local996, 120, !dbg !48
  %address.local999 = and i32 %address.local642, 1536, !dbg !48
  %address.local1000 = or disjoint i32 %address.local997, %address.local999, !dbg !48
  %address.local1001 = zext nneg i32 %address.local1000 to i64, !dbg !48
  %address.local1002 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1001, !dbg !48
  %656 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %657 = bitcast <4 x i32> %656 to <8 x half>, !dbg !48
  %address.local1003 = and i64 %address.local566, 60, !dbg !48
  %address.local1004 = add nuw nsw i64 %address.local1003, %address.local566, !dbg !48
  %address.local1006.idx = shl nuw nsw i64 %address.local1004, 4, !dbg !48
  %address.local1006 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local1006.idx, !dbg !48
  %address.local1007 = lshr i64 %address.local622, 4, !dbg !48
  %address.local1009.idx = mul nuw nsw i64 %address.local1007, 768, !dbg !48
  %address.local1009 = getelementptr i8, ptr addrspace(3) %address.local1006, i64 %address.local1009.idx, !dbg !48
  %address.local1011.idx = mul nsw i64 %address.local609, -64, !dbg !48
  %address.local1011 = getelementptr i8, ptr addrspace(3) %address.local1009, i64 %address.local1011.idx, !dbg !48
  %address.local1012 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 256, !dbg !48
  %658 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %659 = bitcast <4 x i32> %658 to <8 x half>, !dbg !48
  %address.local1016 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 512, !dbg !48
  %660 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %661 = bitcast <4 x i32> %660 to <8 x half>, !dbg !48
  %address.local1020 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 768, !dbg !48
  %662 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %663 = bitcast <4 x i32> %662 to <8 x half>, !dbg !48
  %address.local1022 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 4096, !dbg !48
  %664 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %665 = bitcast <4 x i32> %664 to <8 x half>, !dbg !48
  %address.local1026 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4352, !dbg !48
  %666 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %667 = bitcast <4 x i32> %666 to <8 x half>, !dbg !48
  %address.local1030 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4608, !dbg !48
  %668 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %669 = bitcast <4 x i32> %668 to <8 x half>, !dbg !48
  %address.local1034 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4864, !dbg !48
  %670 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %671 = shufflevector <8 x half> %657, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %672 = extractvalue [2 x <8 x float>] %146, 0, !dbg !46
  %673 = shufflevector <8 x float> %672, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %674 = fmul <4 x float> %673, %642, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %675 = fsub <4 x float> %481, %636, !dbg !43
  %676 = extractelement <4 x float> %675, i64 0, !dbg !50
  %677 = tail call float @llvm.amdgcn.exp2.f32(float %676), !dbg !50
  %678 = insertelement <4 x float> poison, float %677, i64 0, !dbg !51
  %679 = extractelement <4 x float> %675, i64 1, !dbg !50
  %680 = tail call float @llvm.amdgcn.exp2.f32(float %679), !dbg !50
  %681 = insertelement <4 x float> %678, float %680, i64 1, !dbg !51
  %682 = extractelement <4 x float> %675, i64 2, !dbg !50
  %683 = tail call float @llvm.amdgcn.exp2.f32(float %682), !dbg !50
  %684 = insertelement <4 x float> %681, float %683, i64 2, !dbg !51
  %685 = extractelement <4 x float> %675, i64 3, !dbg !50
  %686 = tail call float @llvm.amdgcn.exp2.f32(float %685), !dbg !50
  %687 = insertelement <4 x float> %684, float %686, i64 3, !dbg !51
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %687) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !51
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !51
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !51
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !51
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !51
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !51
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !51
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !51
  %688 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !51
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %688) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %689 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %671, <4 x float> %674) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %690 = shufflevector <8 x half> %657, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %691 = shufflevector <8 x float> %672, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %692 = fmul <4 x float> %691, %642, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %693 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %690, <4 x float> %692) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %694 = extractvalue [2 x <8 x float>] %146, 1, !dbg !46
  %695 = shufflevector <8 x float> %694, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %696 = fmul <4 x float> %695, %650, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %697 = fsub <4 x float> %547, %637, !dbg !43
  %698 = extractelement <4 x float> %697, i64 0, !dbg !50
  %699 = tail call float @llvm.amdgcn.exp2.f32(float %698), !dbg !50
  %700 = insertelement <4 x float> poison, float %699, i64 0, !dbg !51
  %701 = extractelement <4 x float> %697, i64 1, !dbg !50
  %702 = tail call float @llvm.amdgcn.exp2.f32(float %701), !dbg !50
  %703 = insertelement <4 x float> %700, float %702, i64 1, !dbg !51
  %704 = extractelement <4 x float> %697, i64 2, !dbg !50
  %705 = tail call float @llvm.amdgcn.exp2.f32(float %704), !dbg !50
  %706 = insertelement <4 x float> %703, float %705, i64 2, !dbg !51
  %707 = extractelement <4 x float> %697, i64 3, !dbg !50
  %708 = tail call float @llvm.amdgcn.exp2.f32(float %707), !dbg !50
  %709 = insertelement <4 x float> %706, float %708, i64 3, !dbg !51
  %math.result455 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %709) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02009 = extractelement <4 x float> %math.result455, i64 0, !dbg !51
  %.cvtpack.f12010 = extractelement <4 x float> %math.result455, i64 1, !dbg !51
  %.cvtpack.f22011 = extractelement <4 x float> %math.result455, i64 2, !dbg !51
  %.cvtpack.f32012 = extractelement <4 x float> %math.result455, i64 3, !dbg !51
  %.cvtpack.p02013 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02009, float %.cvtpack.f12010) #12, !dbg !51
  %.cvtpack.p12014 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22011, float %.cvtpack.f32012) #12, !dbg !51
  %.cvtpack.v02015 = insertelement <2 x i32> poison, i32 %.cvtpack.p02013, i64 0, !dbg !51
  %.cvtpack.v12016 = insertelement <2 x i32> %.cvtpack.v02015, i32 %.cvtpack.p12014, i64 1, !dbg !51
  %710 = bitcast <2 x i32> %.cvtpack.v12016 to <4 x half>, !dbg !51
  %math.packed456 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %710) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %711 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %671, <4 x float> %696) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %712 = shufflevector <8 x float> %694, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %713 = fmul <4 x float> %712, %650, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %714 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %690, <4 x float> %713) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %715 = shufflevector <8 x half> %659, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %716 = fsub <4 x float> %491, %636, !dbg !43
  %717 = extractelement <4 x float> %716, i64 0, !dbg !50
  %718 = tail call float @llvm.amdgcn.exp2.f32(float %717), !dbg !50
  %local.reduce156 = fadd float %677, %718, !dbg !52, !frisk.local_reduction !5
  %719 = insertelement <4 x float> poison, float %718, i64 0, !dbg !51
  %720 = extractelement <4 x float> %716, i64 1, !dbg !50
  %721 = tail call float @llvm.amdgcn.exp2.f32(float %720), !dbg !50
  %local.reduce157 = fadd float %680, %721, !dbg !52, !frisk.local_reduction !5
  %722 = insertelement <4 x float> %719, float %721, i64 1, !dbg !51
  %723 = extractelement <4 x float> %716, i64 2, !dbg !50
  %724 = tail call float @llvm.amdgcn.exp2.f32(float %723), !dbg !50
  %local.reduce158 = fadd float %683, %724, !dbg !52, !frisk.local_reduction !5
  %725 = insertelement <4 x float> %722, float %724, i64 2, !dbg !51
  %726 = extractelement <4 x float> %716, i64 3, !dbg !50
  %727 = tail call float @llvm.amdgcn.exp2.f32(float %726), !dbg !50
  %local.reduce159 = fadd float %686, %727, !dbg !52, !frisk.local_reduction !5
  %728 = insertelement <4 x float> %725, float %727, i64 3, !dbg !51
  %math.result441 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %728) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02017 = extractelement <4 x float> %math.result441, i64 0, !dbg !51
  %.cvtpack.f12018 = extractelement <4 x float> %math.result441, i64 1, !dbg !51
  %.cvtpack.f22019 = extractelement <4 x float> %math.result441, i64 2, !dbg !51
  %.cvtpack.f32020 = extractelement <4 x float> %math.result441, i64 3, !dbg !51
  %.cvtpack.p02021 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02017, float %.cvtpack.f12018) #12, !dbg !51
  %.cvtpack.p12022 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22019, float %.cvtpack.f32020) #12, !dbg !51
  %.cvtpack.v02023 = insertelement <2 x i32> poison, i32 %.cvtpack.p02021, i64 0, !dbg !51
  %.cvtpack.v12024 = insertelement <2 x i32> %.cvtpack.v02023, i32 %.cvtpack.p12022, i64 1, !dbg !51
  %729 = bitcast <2 x i32> %.cvtpack.v12024 to <4 x half>, !dbg !51
  %math.packed442 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %729) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %730 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %715, <4 x float> %689) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %731 = shufflevector <8 x half> %659, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %732 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %731, <4 x float> %693) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %733 = fsub <4 x float> %550, %637, !dbg !43
  %734 = extractelement <4 x float> %733, i64 0, !dbg !50
  %735 = tail call float @llvm.amdgcn.exp2.f32(float %734), !dbg !50
  %local.reduce = fadd float %699, %735, !dbg !52, !frisk.local_reduction !5
  %736 = insertelement <4 x float> poison, float %735, i64 0, !dbg !51
  %737 = extractelement <4 x float> %733, i64 1, !dbg !50
  %738 = tail call float @llvm.amdgcn.exp2.f32(float %737), !dbg !50
  %local.reduce126 = fadd float %702, %738, !dbg !52, !frisk.local_reduction !5
  %739 = insertelement <4 x float> %736, float %738, i64 1, !dbg !51
  %740 = extractelement <4 x float> %733, i64 2, !dbg !50
  %741 = tail call float @llvm.amdgcn.exp2.f32(float %740), !dbg !50
  %local.reduce127 = fadd float %705, %741, !dbg !52, !frisk.local_reduction !5
  %742 = insertelement <4 x float> %739, float %741, i64 2, !dbg !51
  %743 = extractelement <4 x float> %733, i64 3, !dbg !50
  %744 = tail call float @llvm.amdgcn.exp2.f32(float %743), !dbg !50
  %local.reduce128 = fadd float %708, %744, !dbg !52, !frisk.local_reduction !5
  %745 = insertelement <4 x float> %742, float %744, i64 3, !dbg !51
  %math.result457 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %745) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02025 = extractelement <4 x float> %math.result457, i64 0, !dbg !51
  %.cvtpack.f12026 = extractelement <4 x float> %math.result457, i64 1, !dbg !51
  %.cvtpack.f22027 = extractelement <4 x float> %math.result457, i64 2, !dbg !51
  %.cvtpack.f32028 = extractelement <4 x float> %math.result457, i64 3, !dbg !51
  %.cvtpack.p02029 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02025, float %.cvtpack.f12026) #12, !dbg !51
  %.cvtpack.p12030 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22027, float %.cvtpack.f32028) #12, !dbg !51
  %.cvtpack.v02031 = insertelement <2 x i32> poison, i32 %.cvtpack.p02029, i64 0, !dbg !51
  %.cvtpack.v12032 = insertelement <2 x i32> %.cvtpack.v02031, i32 %.cvtpack.p12030, i64 1, !dbg !51
  %746 = bitcast <2 x i32> %.cvtpack.v12032 to <4 x half>, !dbg !51
  %math.packed458 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %746) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %747 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %715, <4 x float> %711) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %748 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %731, <4 x float> %714) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %749 = shufflevector <8 x half> %661, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %750 = fsub <4 x float> %502, %636, !dbg !43
  %751 = extractelement <4 x float> %750, i64 0, !dbg !50
  %752 = tail call float @llvm.amdgcn.exp2.f32(float %751), !dbg !50
  %local.reduce160 = fadd float %752, %local.reduce156, !dbg !52, !frisk.local_reduction !5
  %753 = insertelement <4 x float> poison, float %752, i64 0, !dbg !51
  %754 = extractelement <4 x float> %750, i64 1, !dbg !50
  %755 = tail call float @llvm.amdgcn.exp2.f32(float %754), !dbg !50
  %local.reduce161 = fadd float %755, %local.reduce157, !dbg !52, !frisk.local_reduction !5
  %756 = insertelement <4 x float> %753, float %755, i64 1, !dbg !51
  %757 = extractelement <4 x float> %750, i64 2, !dbg !50
  %758 = tail call float @llvm.amdgcn.exp2.f32(float %757), !dbg !50
  %local.reduce162 = fadd float %758, %local.reduce158, !dbg !52, !frisk.local_reduction !5
  %759 = insertelement <4 x float> %756, float %758, i64 2, !dbg !51
  %760 = extractelement <4 x float> %750, i64 3, !dbg !50
  %761 = tail call float @llvm.amdgcn.exp2.f32(float %760), !dbg !50
  %local.reduce163 = fadd float %761, %local.reduce159, !dbg !52, !frisk.local_reduction !5
  %762 = insertelement <4 x float> %759, float %761, i64 3, !dbg !51
  %math.result443 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %762) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02033 = extractelement <4 x float> %math.result443, i64 0, !dbg !51
  %.cvtpack.f12034 = extractelement <4 x float> %math.result443, i64 1, !dbg !51
  %.cvtpack.f22035 = extractelement <4 x float> %math.result443, i64 2, !dbg !51
  %.cvtpack.f32036 = extractelement <4 x float> %math.result443, i64 3, !dbg !51
  %.cvtpack.p02037 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02033, float %.cvtpack.f12034) #12, !dbg !51
  %.cvtpack.p12038 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22035, float %.cvtpack.f32036) #12, !dbg !51
  %.cvtpack.v02039 = insertelement <2 x i32> poison, i32 %.cvtpack.p02037, i64 0, !dbg !51
  %.cvtpack.v12040 = insertelement <2 x i32> %.cvtpack.v02039, i32 %.cvtpack.p12038, i64 1, !dbg !51
  %763 = bitcast <2 x i32> %.cvtpack.v12040 to <4 x half>, !dbg !51
  %math.packed444 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %763) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %764 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %749, <4 x float> %730) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %765 = shufflevector <8 x half> %661, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %766 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %765, <4 x float> %732) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %767 = fsub <4 x float> %553, %637, !dbg !43
  %768 = extractelement <4 x float> %767, i64 0, !dbg !50
  %769 = tail call float @llvm.amdgcn.exp2.f32(float %768), !dbg !50
  %local.reduce129 = fadd float %769, %local.reduce, !dbg !52, !frisk.local_reduction !5
  %770 = insertelement <4 x float> poison, float %769, i64 0, !dbg !51
  %771 = extractelement <4 x float> %767, i64 1, !dbg !50
  %772 = tail call float @llvm.amdgcn.exp2.f32(float %771), !dbg !50
  %local.reduce130 = fadd float %772, %local.reduce126, !dbg !52, !frisk.local_reduction !5
  %773 = insertelement <4 x float> %770, float %772, i64 1, !dbg !51
  %774 = extractelement <4 x float> %767, i64 2, !dbg !50
  %775 = tail call float @llvm.amdgcn.exp2.f32(float %774), !dbg !50
  %local.reduce131 = fadd float %775, %local.reduce127, !dbg !52, !frisk.local_reduction !5
  %776 = insertelement <4 x float> %773, float %775, i64 2, !dbg !51
  %777 = extractelement <4 x float> %767, i64 3, !dbg !50
  %778 = tail call float @llvm.amdgcn.exp2.f32(float %777), !dbg !50
  %local.reduce132 = fadd float %778, %local.reduce128, !dbg !52, !frisk.local_reduction !5
  %779 = insertelement <4 x float> %776, float %778, i64 3, !dbg !51
  %math.result459 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %779) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02041 = extractelement <4 x float> %math.result459, i64 0, !dbg !51
  %.cvtpack.f12042 = extractelement <4 x float> %math.result459, i64 1, !dbg !51
  %.cvtpack.f22043 = extractelement <4 x float> %math.result459, i64 2, !dbg !51
  %.cvtpack.f32044 = extractelement <4 x float> %math.result459, i64 3, !dbg !51
  %.cvtpack.p02045 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02041, float %.cvtpack.f12042) #12, !dbg !51
  %.cvtpack.p12046 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22043, float %.cvtpack.f32044) #12, !dbg !51
  %.cvtpack.v02047 = insertelement <2 x i32> poison, i32 %.cvtpack.p02045, i64 0, !dbg !51
  %.cvtpack.v12048 = insertelement <2 x i32> %.cvtpack.v02047, i32 %.cvtpack.p12046, i64 1, !dbg !51
  %780 = bitcast <2 x i32> %.cvtpack.v12048 to <4 x half>, !dbg !51
  %math.packed460 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %780) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %781 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %749, <4 x float> %747) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %782 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %765, <4 x float> %748) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %783 = shufflevector <8 x half> %663, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %784 = fsub <4 x float> %511, %636, !dbg !43
  %785 = extractelement <4 x float> %784, i64 0, !dbg !50
  %786 = tail call float @llvm.amdgcn.exp2.f32(float %785), !dbg !50
  %local.reduce164 = fadd float %786, %local.reduce160, !dbg !52, !frisk.local_reduction !5
  %787 = insertelement <4 x float> poison, float %786, i64 0, !dbg !51
  %788 = extractelement <4 x float> %784, i64 1, !dbg !50
  %789 = tail call float @llvm.amdgcn.exp2.f32(float %788), !dbg !50
  %local.reduce165 = fadd float %789, %local.reduce161, !dbg !52, !frisk.local_reduction !5
  %790 = insertelement <4 x float> %787, float %789, i64 1, !dbg !51
  %791 = extractelement <4 x float> %784, i64 2, !dbg !50
  %792 = tail call float @llvm.amdgcn.exp2.f32(float %791), !dbg !50
  %local.reduce166 = fadd float %792, %local.reduce162, !dbg !52, !frisk.local_reduction !5
  %793 = insertelement <4 x float> %790, float %792, i64 2, !dbg !51
  %794 = extractelement <4 x float> %784, i64 3, !dbg !50
  %795 = tail call float @llvm.amdgcn.exp2.f32(float %794), !dbg !50
  %local.reduce167 = fadd float %795, %local.reduce163, !dbg !52, !frisk.local_reduction !5
  %796 = insertelement <4 x float> %793, float %795, i64 3, !dbg !51
  %math.result445 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %796) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02049 = extractelement <4 x float> %math.result445, i64 0, !dbg !51
  %.cvtpack.f12050 = extractelement <4 x float> %math.result445, i64 1, !dbg !51
  %.cvtpack.f22051 = extractelement <4 x float> %math.result445, i64 2, !dbg !51
  %.cvtpack.f32052 = extractelement <4 x float> %math.result445, i64 3, !dbg !51
  %.cvtpack.p02053 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02049, float %.cvtpack.f12050) #12, !dbg !51
  %.cvtpack.p12054 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22051, float %.cvtpack.f32052) #12, !dbg !51
  %.cvtpack.v02055 = insertelement <2 x i32> poison, i32 %.cvtpack.p02053, i64 0, !dbg !51
  %.cvtpack.v12056 = insertelement <2 x i32> %.cvtpack.v02055, i32 %.cvtpack.p12054, i64 1, !dbg !51
  %797 = bitcast <2 x i32> %.cvtpack.v12056 to <4 x half>, !dbg !51
  %math.packed446 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %797) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %798 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %783, <4 x float> %764) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %799 = shufflevector <8 x half> %663, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %800 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %799, <4 x float> %766) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %801 = fsub <4 x float> %556, %637, !dbg !43
  %802 = extractelement <4 x float> %801, i64 0, !dbg !50
  %803 = tail call float @llvm.amdgcn.exp2.f32(float %802), !dbg !50
  %local.reduce133 = fadd float %803, %local.reduce129, !dbg !52, !frisk.local_reduction !5
  %804 = insertelement <4 x float> poison, float %803, i64 0, !dbg !51
  %805 = extractelement <4 x float> %801, i64 1, !dbg !50
  %806 = tail call float @llvm.amdgcn.exp2.f32(float %805), !dbg !50
  %local.reduce134 = fadd float %806, %local.reduce130, !dbg !52, !frisk.local_reduction !5
  %807 = insertelement <4 x float> %804, float %806, i64 1, !dbg !51
  %808 = extractelement <4 x float> %801, i64 2, !dbg !50
  %809 = tail call float @llvm.amdgcn.exp2.f32(float %808), !dbg !50
  %local.reduce135 = fadd float %809, %local.reduce131, !dbg !52, !frisk.local_reduction !5
  %810 = insertelement <4 x float> %807, float %809, i64 2, !dbg !51
  %811 = extractelement <4 x float> %801, i64 3, !dbg !50
  %812 = tail call float @llvm.amdgcn.exp2.f32(float %811), !dbg !50
  %local.reduce136 = fadd float %812, %local.reduce132, !dbg !52, !frisk.local_reduction !5
  %813 = insertelement <4 x float> %810, float %812, i64 3, !dbg !51
  %math.result461 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %813) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02057 = extractelement <4 x float> %math.result461, i64 0, !dbg !51
  %.cvtpack.f12058 = extractelement <4 x float> %math.result461, i64 1, !dbg !51
  %.cvtpack.f22059 = extractelement <4 x float> %math.result461, i64 2, !dbg !51
  %.cvtpack.f32060 = extractelement <4 x float> %math.result461, i64 3, !dbg !51
  %.cvtpack.p02061 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02057, float %.cvtpack.f12058) #12, !dbg !51
  %.cvtpack.p12062 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22059, float %.cvtpack.f32060) #12, !dbg !51
  %.cvtpack.v02063 = insertelement <2 x i32> poison, i32 %.cvtpack.p02061, i64 0, !dbg !51
  %.cvtpack.v12064 = insertelement <2 x i32> %.cvtpack.v02063, i32 %.cvtpack.p12062, i64 1, !dbg !51
  %814 = bitcast <2 x i32> %.cvtpack.v12064 to <4 x half>, !dbg !51
  %math.packed462 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %814) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %815 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %783, <4 x float> %781) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %799, <4 x float> %782) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %817 = shufflevector <8 x half> %665, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %818 = fsub <4 x float> %518, %636, !dbg !43
  %819 = extractelement <4 x float> %818, i64 0, !dbg !50
  %820 = tail call float @llvm.amdgcn.exp2.f32(float %819), !dbg !50
  %local.reduce168 = fadd float %820, %local.reduce164, !dbg !52, !frisk.local_reduction !5
  %821 = insertelement <4 x float> poison, float %820, i64 0, !dbg !51
  %822 = extractelement <4 x float> %818, i64 1, !dbg !50
  %823 = tail call float @llvm.amdgcn.exp2.f32(float %822), !dbg !50
  %local.reduce169 = fadd float %823, %local.reduce165, !dbg !52, !frisk.local_reduction !5
  %824 = insertelement <4 x float> %821, float %823, i64 1, !dbg !51
  %825 = extractelement <4 x float> %818, i64 2, !dbg !50
  %826 = tail call float @llvm.amdgcn.exp2.f32(float %825), !dbg !50
  %local.reduce170 = fadd float %826, %local.reduce166, !dbg !52, !frisk.local_reduction !5
  %827 = insertelement <4 x float> %824, float %826, i64 2, !dbg !51
  %828 = extractelement <4 x float> %818, i64 3, !dbg !50
  %829 = tail call float @llvm.amdgcn.exp2.f32(float %828), !dbg !50
  %local.reduce171 = fadd float %829, %local.reduce167, !dbg !52, !frisk.local_reduction !5
  %830 = insertelement <4 x float> %827, float %829, i64 3, !dbg !51
  %math.result447 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %830) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02065 = extractelement <4 x float> %math.result447, i64 0, !dbg !51
  %.cvtpack.f12066 = extractelement <4 x float> %math.result447, i64 1, !dbg !51
  %.cvtpack.f22067 = extractelement <4 x float> %math.result447, i64 2, !dbg !51
  %.cvtpack.f32068 = extractelement <4 x float> %math.result447, i64 3, !dbg !51
  %.cvtpack.p02069 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02065, float %.cvtpack.f12066) #12, !dbg !51
  %.cvtpack.p12070 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22067, float %.cvtpack.f32068) #12, !dbg !51
  %.cvtpack.v02071 = insertelement <2 x i32> poison, i32 %.cvtpack.p02069, i64 0, !dbg !51
  %.cvtpack.v12072 = insertelement <2 x i32> %.cvtpack.v02071, i32 %.cvtpack.p12070, i64 1, !dbg !51
  %831 = bitcast <2 x i32> %.cvtpack.v12072 to <4 x half>, !dbg !51
  %math.packed448 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %831) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %817, <4 x float> %798) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %833 = shufflevector <8 x half> %665, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %833, <4 x float> %800) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %835 = fsub <4 x float> %559, %637, !dbg !43
  %836 = extractelement <4 x float> %835, i64 0, !dbg !50
  %837 = tail call float @llvm.amdgcn.exp2.f32(float %836), !dbg !50
  %local.reduce137 = fadd float %837, %local.reduce133, !dbg !52, !frisk.local_reduction !5
  %838 = insertelement <4 x float> poison, float %837, i64 0, !dbg !51
  %839 = extractelement <4 x float> %835, i64 1, !dbg !50
  %840 = tail call float @llvm.amdgcn.exp2.f32(float %839), !dbg !50
  %local.reduce138 = fadd float %840, %local.reduce134, !dbg !52, !frisk.local_reduction !5
  %841 = insertelement <4 x float> %838, float %840, i64 1, !dbg !51
  %842 = extractelement <4 x float> %835, i64 2, !dbg !50
  %843 = tail call float @llvm.amdgcn.exp2.f32(float %842), !dbg !50
  %local.reduce139 = fadd float %843, %local.reduce135, !dbg !52, !frisk.local_reduction !5
  %844 = insertelement <4 x float> %841, float %843, i64 2, !dbg !51
  %845 = extractelement <4 x float> %835, i64 3, !dbg !50
  %846 = tail call float @llvm.amdgcn.exp2.f32(float %845), !dbg !50
  %local.reduce140 = fadd float %846, %local.reduce136, !dbg !52, !frisk.local_reduction !5
  %847 = insertelement <4 x float> %844, float %846, i64 3, !dbg !51
  %math.result463 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %847) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02073 = extractelement <4 x float> %math.result463, i64 0, !dbg !51
  %.cvtpack.f12074 = extractelement <4 x float> %math.result463, i64 1, !dbg !51
  %.cvtpack.f22075 = extractelement <4 x float> %math.result463, i64 2, !dbg !51
  %.cvtpack.f32076 = extractelement <4 x float> %math.result463, i64 3, !dbg !51
  %.cvtpack.p02077 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02073, float %.cvtpack.f12074) #12, !dbg !51
  %.cvtpack.p12078 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22075, float %.cvtpack.f32076) #12, !dbg !51
  %.cvtpack.v02079 = insertelement <2 x i32> poison, i32 %.cvtpack.p02077, i64 0, !dbg !51
  %.cvtpack.v12080 = insertelement <2 x i32> %.cvtpack.v02079, i32 %.cvtpack.p12078, i64 1, !dbg !51
  %848 = bitcast <2 x i32> %.cvtpack.v12080 to <4 x half>, !dbg !51
  %math.packed464 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %848) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %849 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %817, <4 x float> %815) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %833, <4 x float> %816) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %851 = shufflevector <8 x half> %667, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %852 = fsub <4 x float> %525, %636, !dbg !43
  %853 = extractelement <4 x float> %852, i64 0, !dbg !50
  %854 = tail call float @llvm.amdgcn.exp2.f32(float %853), !dbg !50
  %local.reduce172 = fadd float %854, %local.reduce168, !dbg !52, !frisk.local_reduction !5
  %855 = insertelement <4 x float> poison, float %854, i64 0, !dbg !51
  %856 = extractelement <4 x float> %852, i64 1, !dbg !50
  %857 = tail call float @llvm.amdgcn.exp2.f32(float %856), !dbg !50
  %local.reduce173 = fadd float %857, %local.reduce169, !dbg !52, !frisk.local_reduction !5
  %858 = insertelement <4 x float> %855, float %857, i64 1, !dbg !51
  %859 = extractelement <4 x float> %852, i64 2, !dbg !50
  %860 = tail call float @llvm.amdgcn.exp2.f32(float %859), !dbg !50
  %local.reduce174 = fadd float %860, %local.reduce170, !dbg !52, !frisk.local_reduction !5
  %861 = insertelement <4 x float> %858, float %860, i64 2, !dbg !51
  %862 = extractelement <4 x float> %852, i64 3, !dbg !50
  %863 = tail call float @llvm.amdgcn.exp2.f32(float %862), !dbg !50
  %local.reduce175 = fadd float %863, %local.reduce171, !dbg !52, !frisk.local_reduction !5
  %864 = insertelement <4 x float> %861, float %863, i64 3, !dbg !51
  %math.result449 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %864) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02081 = extractelement <4 x float> %math.result449, i64 0, !dbg !51
  %.cvtpack.f12082 = extractelement <4 x float> %math.result449, i64 1, !dbg !51
  %.cvtpack.f22083 = extractelement <4 x float> %math.result449, i64 2, !dbg !51
  %.cvtpack.f32084 = extractelement <4 x float> %math.result449, i64 3, !dbg !51
  %.cvtpack.p02085 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02081, float %.cvtpack.f12082) #12, !dbg !51
  %.cvtpack.p12086 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22083, float %.cvtpack.f32084) #12, !dbg !51
  %.cvtpack.v02087 = insertelement <2 x i32> poison, i32 %.cvtpack.p02085, i64 0, !dbg !51
  %.cvtpack.v12088 = insertelement <2 x i32> %.cvtpack.v02087, i32 %.cvtpack.p12086, i64 1, !dbg !51
  %865 = bitcast <2 x i32> %.cvtpack.v12088 to <4 x half>, !dbg !51
  %math.packed450 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %865) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %866 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %851, <4 x float> %832) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %867 = shufflevector <8 x half> %667, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %868 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %867, <4 x float> %834) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %869 = fsub <4 x float> %562, %637, !dbg !43
  %870 = extractelement <4 x float> %869, i64 0, !dbg !50
  %871 = tail call float @llvm.amdgcn.exp2.f32(float %870), !dbg !50
  %local.reduce141 = fadd float %871, %local.reduce137, !dbg !52, !frisk.local_reduction !5
  %872 = insertelement <4 x float> poison, float %871, i64 0, !dbg !51
  %873 = extractelement <4 x float> %869, i64 1, !dbg !50
  %874 = tail call float @llvm.amdgcn.exp2.f32(float %873), !dbg !50
  %local.reduce142 = fadd float %874, %local.reduce138, !dbg !52, !frisk.local_reduction !5
  %875 = insertelement <4 x float> %872, float %874, i64 1, !dbg !51
  %876 = extractelement <4 x float> %869, i64 2, !dbg !50
  %877 = tail call float @llvm.amdgcn.exp2.f32(float %876), !dbg !50
  %local.reduce143 = fadd float %877, %local.reduce139, !dbg !52, !frisk.local_reduction !5
  %878 = insertelement <4 x float> %875, float %877, i64 2, !dbg !51
  %879 = extractelement <4 x float> %869, i64 3, !dbg !50
  %880 = tail call float @llvm.amdgcn.exp2.f32(float %879), !dbg !50
  %local.reduce144 = fadd float %880, %local.reduce140, !dbg !52, !frisk.local_reduction !5
  %881 = insertelement <4 x float> %878, float %880, i64 3, !dbg !51
  %math.result465 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %881) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02089 = extractelement <4 x float> %math.result465, i64 0, !dbg !51
  %.cvtpack.f12090 = extractelement <4 x float> %math.result465, i64 1, !dbg !51
  %.cvtpack.f22091 = extractelement <4 x float> %math.result465, i64 2, !dbg !51
  %.cvtpack.f32092 = extractelement <4 x float> %math.result465, i64 3, !dbg !51
  %.cvtpack.p02093 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02089, float %.cvtpack.f12090) #12, !dbg !51
  %.cvtpack.p12094 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22091, float %.cvtpack.f32092) #12, !dbg !51
  %.cvtpack.v02095 = insertelement <2 x i32> poison, i32 %.cvtpack.p02093, i64 0, !dbg !51
  %.cvtpack.v12096 = insertelement <2 x i32> %.cvtpack.v02095, i32 %.cvtpack.p12094, i64 1, !dbg !51
  %882 = bitcast <2 x i32> %.cvtpack.v12096 to <4 x half>, !dbg !51
  %math.packed466 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %882) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %851, <4 x float> %849) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %867, <4 x float> %850) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %885 = shufflevector <8 x half> %669, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %886 = fsub <4 x float> %532, %636, !dbg !43
  %887 = extractelement <4 x float> %886, i64 0, !dbg !50
  %888 = tail call float @llvm.amdgcn.exp2.f32(float %887), !dbg !50
  %local.reduce176 = fadd float %888, %local.reduce172, !dbg !52, !frisk.local_reduction !5
  %889 = insertelement <4 x float> poison, float %888, i64 0, !dbg !51
  %890 = extractelement <4 x float> %886, i64 1, !dbg !50
  %891 = tail call float @llvm.amdgcn.exp2.f32(float %890), !dbg !50
  %local.reduce177 = fadd float %891, %local.reduce173, !dbg !52, !frisk.local_reduction !5
  %892 = insertelement <4 x float> %889, float %891, i64 1, !dbg !51
  %893 = extractelement <4 x float> %886, i64 2, !dbg !50
  %894 = tail call float @llvm.amdgcn.exp2.f32(float %893), !dbg !50
  %local.reduce178 = fadd float %894, %local.reduce174, !dbg !52, !frisk.local_reduction !5
  %895 = insertelement <4 x float> %892, float %894, i64 2, !dbg !51
  %896 = extractelement <4 x float> %886, i64 3, !dbg !50
  %897 = tail call float @llvm.amdgcn.exp2.f32(float %896), !dbg !50
  %local.reduce179 = fadd float %897, %local.reduce175, !dbg !52, !frisk.local_reduction !5
  %898 = insertelement <4 x float> %895, float %897, i64 3, !dbg !51
  %math.result451 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %898) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02097 = extractelement <4 x float> %math.result451, i64 0, !dbg !51
  %.cvtpack.f12098 = extractelement <4 x float> %math.result451, i64 1, !dbg !51
  %.cvtpack.f22099 = extractelement <4 x float> %math.result451, i64 2, !dbg !51
  %.cvtpack.f32100 = extractelement <4 x float> %math.result451, i64 3, !dbg !51
  %.cvtpack.p02101 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02097, float %.cvtpack.f12098) #12, !dbg !51
  %.cvtpack.p12102 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22099, float %.cvtpack.f32100) #12, !dbg !51
  %.cvtpack.v02103 = insertelement <2 x i32> poison, i32 %.cvtpack.p02101, i64 0, !dbg !51
  %.cvtpack.v12104 = insertelement <2 x i32> %.cvtpack.v02103, i32 %.cvtpack.p12102, i64 1, !dbg !51
  %899 = bitcast <2 x i32> %.cvtpack.v12104 to <4 x half>, !dbg !51
  %math.packed452 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %899) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %885, <4 x float> %866) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %901 = shufflevector <8 x half> %669, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %901, <4 x float> %868) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %903 = fsub <4 x float> %565, %637, !dbg !43
  %904 = extractelement <4 x float> %903, i64 0, !dbg !50
  %905 = tail call float @llvm.amdgcn.exp2.f32(float %904), !dbg !50
  %local.reduce145 = fadd float %905, %local.reduce141, !dbg !52, !frisk.local_reduction !5
  %906 = insertelement <4 x float> poison, float %905, i64 0, !dbg !51
  %907 = extractelement <4 x float> %903, i64 1, !dbg !50
  %908 = tail call float @llvm.amdgcn.exp2.f32(float %907), !dbg !50
  %local.reduce146 = fadd float %908, %local.reduce142, !dbg !52, !frisk.local_reduction !5
  %909 = insertelement <4 x float> %906, float %908, i64 1, !dbg !51
  %910 = extractelement <4 x float> %903, i64 2, !dbg !50
  %911 = tail call float @llvm.amdgcn.exp2.f32(float %910), !dbg !50
  %local.reduce147 = fadd float %911, %local.reduce143, !dbg !52, !frisk.local_reduction !5
  %912 = insertelement <4 x float> %909, float %911, i64 2, !dbg !51
  %913 = extractelement <4 x float> %903, i64 3, !dbg !50
  %914 = tail call float @llvm.amdgcn.exp2.f32(float %913), !dbg !50
  %local.reduce148 = fadd float %914, %local.reduce144, !dbg !52, !frisk.local_reduction !5
  %915 = insertelement <4 x float> %912, float %914, i64 3, !dbg !51
  %math.result467 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %915) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02105 = extractelement <4 x float> %math.result467, i64 0, !dbg !51
  %.cvtpack.f12106 = extractelement <4 x float> %math.result467, i64 1, !dbg !51
  %.cvtpack.f22107 = extractelement <4 x float> %math.result467, i64 2, !dbg !51
  %.cvtpack.f32108 = extractelement <4 x float> %math.result467, i64 3, !dbg !51
  %.cvtpack.p02109 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02105, float %.cvtpack.f12106) #12, !dbg !51
  %.cvtpack.p12110 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22107, float %.cvtpack.f32108) #12, !dbg !51
  %.cvtpack.v02111 = insertelement <2 x i32> poison, i32 %.cvtpack.p02109, i64 0, !dbg !51
  %.cvtpack.v12112 = insertelement <2 x i32> %.cvtpack.v02111, i32 %.cvtpack.p12110, i64 1, !dbg !51
  %916 = bitcast <2 x i32> %.cvtpack.v12112 to <4 x half>, !dbg !51
  %math.packed468 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %916) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %885, <4 x float> %883) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %901, <4 x float> %884) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %919 = bitcast <4 x i32> %670 to <8 x half>, !dbg !48
  %920 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %921 = fsub <4 x float> %539, %636, !dbg !43
  %922 = extractelement <4 x float> %921, i64 0, !dbg !50
  %923 = tail call float @llvm.amdgcn.exp2.f32(float %922), !dbg !50
  %local.reduce180 = fadd float %923, %local.reduce176, !dbg !52, !frisk.local_reduction !5
  %924 = insertelement <4 x float> poison, float %923, i64 0, !dbg !51
  %925 = extractelement <4 x float> %921, i64 1, !dbg !50
  %926 = tail call float @llvm.amdgcn.exp2.f32(float %925), !dbg !50
  %local.reduce181 = fadd float %926, %local.reduce177, !dbg !52, !frisk.local_reduction !5
  %local.reduce184 = fadd float %local.reduce180, %local.reduce181, !dbg !52, !frisk.local_reduction !5
  %927 = insertelement <4 x float> %924, float %926, i64 1, !dbg !51
  %928 = extractelement <4 x float> %921, i64 2, !dbg !50
  %929 = tail call float @llvm.amdgcn.exp2.f32(float %928), !dbg !50
  %local.reduce182 = fadd float %929, %local.reduce178, !dbg !52, !frisk.local_reduction !5
  %930 = insertelement <4 x float> %927, float %929, i64 2, !dbg !51
  %931 = extractelement <4 x float> %921, i64 3, !dbg !50
  %932 = tail call float @llvm.amdgcn.exp2.f32(float %931), !dbg !50
  %local.reduce183 = fadd float %932, %local.reduce179, !dbg !52, !frisk.local_reduction !5
  %local.reduce185 = fadd float %local.reduce182, %local.reduce183, !dbg !52, !frisk.local_reduction !5
  %933 = fadd float %local.reduce184, %local.reduce185, !dbg !52, !frisk.local_reduction !5
  %934 = fadd float %645, %933, !dbg !52
  %935 = insertelement <1 x float> poison, float %934, i64 0, !dbg !11
  %936 = insertvalue [2 x <1 x float>] poison, <1 x float> %935, 0, !dbg !11
  %937 = insertelement <4 x float> %930, float %932, i64 3, !dbg !51
  %math.result453 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %937) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02113 = extractelement <4 x float> %math.result453, i64 0, !dbg !51
  %.cvtpack.f12114 = extractelement <4 x float> %math.result453, i64 1, !dbg !51
  %.cvtpack.f22115 = extractelement <4 x float> %math.result453, i64 2, !dbg !51
  %.cvtpack.f32116 = extractelement <4 x float> %math.result453, i64 3, !dbg !51
  %.cvtpack.p02117 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02113, float %.cvtpack.f12114) #12, !dbg !51
  %.cvtpack.p12118 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22115, float %.cvtpack.f32116) #12, !dbg !51
  %.cvtpack.v02119 = insertelement <2 x i32> poison, i32 %.cvtpack.p02117, i64 0, !dbg !51
  %.cvtpack.v12120 = insertelement <2 x i32> %.cvtpack.v02119, i32 %.cvtpack.p12118, i64 1, !dbg !51
  %938 = bitcast <2 x i32> %.cvtpack.v12120 to <4 x half>, !dbg !51
  %math.packed454 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %938) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %939 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %920, <4 x float> %900) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %940 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %941 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %940, <4 x float> %902) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %942 = fsub <4 x float> %579, %637, !dbg !43
  %943 = extractelement <4 x float> %942, i64 0, !dbg !50
  %944 = tail call float @llvm.amdgcn.exp2.f32(float %943), !dbg !50
  %local.reduce149 = fadd float %944, %local.reduce145, !dbg !52, !frisk.local_reduction !5
  %945 = insertelement <4 x float> poison, float %944, i64 0, !dbg !51
  %946 = extractelement <4 x float> %942, i64 1, !dbg !50
  %947 = tail call float @llvm.amdgcn.exp2.f32(float %946), !dbg !50
  %local.reduce150 = fadd float %947, %local.reduce146, !dbg !52, !frisk.local_reduction !5
  %local.reduce153 = fadd float %local.reduce149, %local.reduce150, !dbg !52, !frisk.local_reduction !5
  %948 = insertelement <4 x float> %945, float %947, i64 1, !dbg !51
  %949 = extractelement <4 x float> %942, i64 2, !dbg !50
  %950 = tail call float @llvm.amdgcn.exp2.f32(float %949), !dbg !50
  %local.reduce151 = fadd float %950, %local.reduce147, !dbg !52, !frisk.local_reduction !5
  %951 = insertelement <4 x float> %948, float %950, i64 2, !dbg !51
  %952 = extractelement <4 x float> %942, i64 3, !dbg !50
  %953 = tail call float @llvm.amdgcn.exp2.f32(float %952), !dbg !50
  %local.reduce152 = fadd float %953, %local.reduce148, !dbg !52, !frisk.local_reduction !5
  %local.reduce154 = fadd float %local.reduce151, %local.reduce152, !dbg !52, !frisk.local_reduction !5
  %954 = fadd float %local.reduce153, %local.reduce154, !dbg !52, !frisk.local_reduction !5
  %955 = fadd float %653, %954, !dbg !52
  %956 = insertelement <1 x float> poison, float %955, i64 0, !dbg !11
  %957 = insertvalue [2 x <1 x float>] %936, <1 x float> %956, 1, !dbg !11
  %958 = insertelement <4 x float> %951, float %953, i64 3, !dbg !51
  %math.result469 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %958) #8, !dbg !51, !frisk.register_lifetimes !5
  %.cvtpack.f02121 = extractelement <4 x float> %math.result469, i64 0, !dbg !51
  %.cvtpack.f12122 = extractelement <4 x float> %math.result469, i64 1, !dbg !51
  %.cvtpack.f22123 = extractelement <4 x float> %math.result469, i64 2, !dbg !51
  %.cvtpack.f32124 = extractelement <4 x float> %math.result469, i64 3, !dbg !51
  %.cvtpack.p02125 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02121, float %.cvtpack.f12122) #12, !dbg !51
  %.cvtpack.p12126 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22123, float %.cvtpack.f32124) #12, !dbg !51
  %.cvtpack.v02127 = insertelement <2 x i32> poison, i32 %.cvtpack.p02125, i64 0, !dbg !51
  %.cvtpack.v12128 = insertelement <2 x i32> %.cvtpack.v02127, i32 %.cvtpack.p12126, i64 1, !dbg !51
  %959 = bitcast <2 x i32> %.cvtpack.v12128 to <4 x half>, !dbg !51
  %math.packed470 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %959) #8, !dbg !52, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48, !frisk.softmax_fragments !5
  %960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %920, <4 x float> %917) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %940, <4 x float> %918) #8, !dbg !48, !frisk.pv_interleave !20
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
  %962 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %181, i32 %vdb.address1561, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %963 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %185, i32 %vdb.address1570, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local1002, i64 8192, !dbg !48
  %964 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %965 = bitcast <4 x i32> %964 to <8 x half>, !dbg !48
  %vdb.read1550 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8448, !dbg !48
  %966 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %967 = bitcast <4 x i32> %966 to <8 x half>, !dbg !48
  %vdb.read1551 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8704, !dbg !48
  %968 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %969 = bitcast <4 x i32> %968 to <8 x half>, !dbg !48
  %vdb.read1552 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8960, !dbg !48
  %970 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1552, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %971 = bitcast <4 x i32> %970 to <8 x half>, !dbg !48
  %vdb.read1553 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 12288, !dbg !48
  %972 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1553, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %973 = bitcast <4 x i32> %972 to <8 x half>, !dbg !48
  %vdb.read1554 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12544, !dbg !48
  %974 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1554, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %975 = bitcast <4 x i32> %974 to <8 x half>, !dbg !48
  %vdb.read1555 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12800, !dbg !48
  %976 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1555, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %977 = bitcast <4 x i32> %976 to <8 x half>, !dbg !48
  %vdb.read1556 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 13056, !dbg !48
  %978 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1556, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %979 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %980 = extractvalue [2 x <8 x float>] %143, 0, !dbg !46
  %981 = shufflevector <8 x float> %980, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %982 = fmul <4 x float> %981, %642, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %983 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %979, <4 x float> %982) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %984 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %985 = shufflevector <8 x float> %980, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %986 = fmul <4 x float> %985, %642, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %984, <4 x float> %986) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %988 = extractvalue [2 x <8 x float>] %143, 1, !dbg !46
  %989 = shufflevector <8 x float> %988, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %990 = fmul <4 x float> %989, %650, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %979, <4 x float> %990) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %992 = shufflevector <8 x float> %988, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %993 = fmul <4 x float> %992, %650, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %984, <4 x float> %993) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %995 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %995, <4 x float> %983) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %997 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %997, <4 x float> %987) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %999 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %995, <4 x float> %991) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %997, <4 x float> %994) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1001 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1001, <4 x float> %996) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1003 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1003, <4 x float> %998) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1005 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1001, <4 x float> %999) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1003, <4 x float> %1000) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1007 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1007, <4 x float> %1002) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1009 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1009, <4 x float> %1004) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1011 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1007, <4 x float> %1005) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1009, <4 x float> %1006) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1013 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1013, <4 x float> %1008) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1015 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1016 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1015, <4 x float> %1010) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1017 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1013, <4 x float> %1011) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1018 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1015, <4 x float> %1012) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1019 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1019, <4 x float> %1014) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1021 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1022 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1021, <4 x float> %1016) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1023 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1019, <4 x float> %1017) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1021, <4 x float> %1018) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1025 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1025, <4 x float> %1020) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1027 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1028 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1027, <4 x float> %1022) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1029 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1025, <4 x float> %1023) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1027, <4 x float> %1024) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1031 = bitcast <4 x i32> %978 to <8 x half>, !dbg !48
  %1032 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1033 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1032, <4 x float> %1026) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1034 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1035 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1034, <4 x float> %1028) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1036 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1032, <4 x float> %1029) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1037 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1034, <4 x float> %1030) #8, !dbg !48, !frisk.pv_interleave !20
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
  %1038 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1579, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %1039 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1549, i32 %vdb.address1588, <4 x i32> %dma.buffer.rsrc1414) #9, !dbg !39
  %1040 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1041 = bitcast <4 x i32> %1040 to <8 x half>, !dbg !48
  %1042 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1043 = bitcast <4 x i32> %1042 to <8 x half>, !dbg !48
  %1044 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1045 = bitcast <4 x i32> %1044 to <8 x half>, !dbg !48
  %1046 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1047 = bitcast <4 x i32> %1046 to <8 x half>, !dbg !48
  %1048 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1049 = bitcast <4 x i32> %1048 to <8 x half>, !dbg !48
  %1050 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1051 = bitcast <4 x i32> %1050 to <8 x half>, !dbg !48
  %1052 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1053 = bitcast <4 x i32> %1052 to <8 x half>, !dbg !48
  %1054 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1055 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %1056 = extractvalue [2 x <8 x float>] %142, 0, !dbg !46
  %1057 = shufflevector <8 x float> %1056, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %1058 = fmul <4 x float> %1057, %642, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1059 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1055, <4 x float> %1058) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1060 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %1061 = shufflevector <8 x float> %1056, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %1062 = fmul <4 x float> %1061, %642, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1063 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1060, <4 x float> %1062) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1064 = extractvalue [2 x <8 x float>] %142, 1, !dbg !46
  %1065 = shufflevector <8 x float> %1064, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %1066 = fmul <4 x float> %1065, %650, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1067 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1055, <4 x float> %1066) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1068 = shufflevector <8 x float> %1064, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %1069 = fmul <4 x float> %1068, %650, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1070 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1060, <4 x float> %1069) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1071 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1072 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1071, <4 x float> %1059) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1073 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1074 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1073, <4 x float> %1063) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1075 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1071, <4 x float> %1067) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1073, <4 x float> %1070) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1077 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1077, <4 x float> %1072) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1079 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1079, <4 x float> %1074) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1081 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1077, <4 x float> %1075) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1079, <4 x float> %1076) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1083 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1083, <4 x float> %1078) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1085 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1085, <4 x float> %1080) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1087 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1083, <4 x float> %1081) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1085, <4 x float> %1082) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1089 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1089, <4 x float> %1084) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1091 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1091, <4 x float> %1086) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1093 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1089, <4 x float> %1087) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1091, <4 x float> %1088) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1095 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1095, <4 x float> %1090) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1097 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1098 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1097, <4 x float> %1092) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1099 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1095, <4 x float> %1093) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1100 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1097, <4 x float> %1094) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1101 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1101, <4 x float> %1096) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1103 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1103, <4 x float> %1098) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1105 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1101, <4 x float> %1099) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1103, <4 x float> %1100) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1107 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !48
  %1108 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1109 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1108, <4 x float> %1102) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1110 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1111 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1110, <4 x float> %1104) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1112 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1108, <4 x float> %1105) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1110, <4 x float> %1106) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %1114 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1115 = bitcast <4 x i32> %1114 to <8 x half>, !dbg !48
  %1116 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1117 = bitcast <4 x i32> %1116 to <8 x half>, !dbg !48
  %1118 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1119 = bitcast <4 x i32> %1118 to <8 x half>, !dbg !48
  %1120 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1552, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1121 = bitcast <4 x i32> %1120 to <8 x half>, !dbg !48
  %1122 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1553, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1123 = bitcast <4 x i32> %1122 to <8 x half>, !dbg !48
  %1124 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1554, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1125 = bitcast <4 x i32> %1124 to <8 x half>, !dbg !48
  %1126 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1555, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1127 = bitcast <4 x i32> %1126 to <8 x half>, !dbg !48
  %1128 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1556, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %1129 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %1130 = extractvalue [2 x <8 x float>] %141, 0, !dbg !46
  %1131 = shufflevector <8 x float> %1130, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %1132 = fmul <4 x float> %1131, %642, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1129, <4 x float> %1132) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1134 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %1135 = shufflevector <8 x float> %1130, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %1136 = fmul <4 x float> %1135, %642, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1134, <4 x float> %1136) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1138 = extractvalue [2 x <8 x float>] %141, 1, !dbg !46
  %1139 = shufflevector <8 x float> %1138, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %1140 = fmul <4 x float> %1139, %650, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1129, <4 x float> %1140) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1142 = shufflevector <8 x float> %1138, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %1143 = fmul <4 x float> %1142, %650, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1144 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1134, <4 x float> %1143) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1145 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1145, <4 x float> %1133) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1147 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1147, <4 x float> %1137) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1145, <4 x float> %1141) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1147, <4 x float> %1144) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1151 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1151, <4 x float> %1146) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1153 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1153, <4 x float> %1148) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1151, <4 x float> %1149) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1153, <4 x float> %1150) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1157 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1157, <4 x float> %1152) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1159 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1159, <4 x float> %1154) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1157, <4 x float> %1155) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1159, <4 x float> %1156) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1163 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1163, <4 x float> %1158) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1165 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1165, <4 x float> %1160) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1163, <4 x float> %1161) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1165, <4 x float> %1162) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1169 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1169, <4 x float> %1164) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1171 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1171, <4 x float> %1166) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1169, <4 x float> %1167) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1171, <4 x float> %1168) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1175 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1175, <4 x float> %1170) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1177 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1177, <4 x float> %1172) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1175, <4 x float> %1173) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1177, <4 x float> %1174) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1181 = bitcast <4 x i32> %1128 to <8 x half>, !dbg !48
  %1182 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1182, <4 x float> %1176) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1184 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1184, <4 x float> %1178) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1182, <4 x float> %1179) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1184, <4 x float> %1180) #8, !dbg !48, !frisk.pv_interleave !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !11
  %1188 = shufflevector <4 x float> %939, <4 x float> %941, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1189 = insertvalue [2 x <8 x float>] poison, <8 x float> %1188, 0, !dbg !11
  %1190 = shufflevector <4 x float> %960, <4 x float> %961, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1191 = insertvalue [2 x <8 x float>] %1189, <8 x float> %1190, 1, !dbg !11
  %1192 = insertvalue [2 x <1 x float>] poison, <1 x float> %632, 0, !dbg !11
  %1193 = insertvalue [2 x <1 x float>] %1192, <1 x float> %635, 1, !dbg !11
  %1194 = shufflevector <4 x float> %1033, <4 x float> %1035, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1195 = insertvalue [2 x <8 x float>] poison, <8 x float> %1194, 0, !dbg !11
  %1196 = shufflevector <4 x float> %1036, <4 x float> %1037, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1197 = insertvalue [2 x <8 x float>] %1195, <8 x float> %1196, 1, !dbg !11
  %1198 = shufflevector <4 x float> %1109, <4 x float> %1111, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1199 = insertvalue [2 x <8 x float>] poison, <8 x float> %1198, 0, !dbg !11
  %1200 = shufflevector <4 x float> %1112, <4 x float> %1113, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1201 = insertvalue [2 x <8 x float>] %1199, <8 x float> %1200, 1, !dbg !11
  %1202 = shufflevector <4 x float> %1183, <4 x float> %1185, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1203 = insertvalue [2 x <8 x float>] poison, <8 x float> %1202, 0, !dbg !11
  %1204 = shufflevector <4 x float> %1186, <4 x float> %1187, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1205 = insertvalue [2 x <8 x float>] %1203, <8 x float> %1204, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1206 = add nuw nsw i64 %147, 1, !dbg !11
  %1207 = icmp slt i64 %1206, %105, !dbg !11
  br i1 %1207, label %140, label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %140, %.._crit_edge_crit_edge
  %.pre-phi29 = phi i32 [ %.pre28, %.._crit_edge_crit_edge ], [ %612, %140 ], !dbg !11
  %.pre-phi27 = phi i32 [ %.pre26, %.._crit_edge_crit_edge ], [ %604, %140 ], !dbg !11
  %.pre-phi25 = phi i32 [ %.pre24, %.._crit_edge_crit_edge ], [ %603, %140 ], !dbg !11
  %lane.sum.lane254.pre-phi = phi i32 [ %.pre22, %.._crit_edge_crit_edge ], [ %601, %140 ], !dbg !11
  %.lcssa7 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1191, %140 ], !dbg !11
  %.lcssa6 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %957, %140 ], !dbg !11
  %.lcssa5 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1197, %140 ], !dbg !11
  %.lcssa4 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1201, %140 ], !dbg !11
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1205, %140 ], !dbg !11
  %lane.sum.field251 = extractvalue [2 x <1 x float>] %.lcssa6, 1, !dbg !11
  %lane.sum.local252 = extractelement <1 x float> %lane.sum.field251, i64 0, !dbg !11
  %1208 = shl nuw nsw i32 %lane.sum.lane254.pre-phi, 2, !dbg !11
  %1209 = xor i32 %1208, 64, !dbg !11
  %1210 = bitcast float %lane.sum.local252 to i32, !dbg !11
  %lane.sum.shuffle255 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1209, i32 %1210), !dbg !11
  %1211 = bitcast i32 %lane.sum.shuffle255 to float, !dbg !11
  %lane.sum.final256 = fadd float %lane.sum.local252, %1211, !dbg !11
  %1212 = xor i32 %1208, 128, !dbg !11
  %1213 = bitcast float %lane.sum.final256 to i32, !dbg !11
  %lane.sum.shuffle257 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1213), !dbg !11
  %1214 = bitcast i32 %lane.sum.shuffle257 to float, !dbg !11
  %lane.sum.final258 = fadd float %lane.sum.final256, %1214, !dbg !11
  %1215 = insertelement <1 x float> poison, float %lane.sum.final258, i64 0, !dbg !11
  %lane.sum.field = extractvalue [2 x <1 x float>] %.lcssa6, 0, !dbg !11
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !11
  %1216 = bitcast float %lane.sum.local to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1209, i32 %1216), !dbg !11
  %1217 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %lane.sum.local, %1217, !dbg !11
  %1218 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle249 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1212, i32 %1218), !dbg !11
  %1219 = bitcast i32 %lane.sum.shuffle249 to float, !dbg !11
  %lane.sum.final250 = fadd float %lane.sum.final, %1219, !dbg !11
  %1220 = insertelement <1 x float> poison, float %lane.sum.final250, i64 0, !dbg !11
  %1221 = and i32 %kdma.tid, 16, !dbg !11
  %.not = icmp eq i32 %1221, 0, !dbg !11
  %1222 = and i32 %kdma.tid, 32, !dbg !11
  %.not36 = icmp eq i32 %1222, 0, !dbg !11
  %1223 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !11
  %1224 = extractelement <8 x float> %1223, i64 0, !dbg !11
  %1225 = extractelement <8 x float> %1223, i64 1, !dbg !11
  %1226 = extractelement <8 x float> %1223, i64 2, !dbg !11
  %1227 = extractelement <8 x float> %1223, i64 3, !dbg !11
  %1228 = select i1 %.not, float %1225, float %1224, !dbg !11
  %1229 = icmp samesign ult i32 %.pre-phi25, %.pre-phi27, !dbg !11
  %1230 = select i1 %1229, i32 %.pre-phi25, i32 %lane.sum.lane254.pre-phi, !dbg !11
  %1231 = shl nuw nsw i32 %1230, 2, !dbg !11
  %1232 = bitcast float %1228 to i32, !dbg !11
  %1233 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1232), !dbg !11
  %1234 = bitcast i32 %1233 to float, !dbg !11
  %1235 = select i1 %.not, float %1224, float %1234, !dbg !11
  %1236 = select i1 %.not, float %1234, float %1225, !dbg !11
  %1237 = select i1 %.not, float %1227, float %1226, !dbg !11
  %1238 = bitcast float %1237 to i32, !dbg !11
  %1239 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1238), !dbg !11
  %1240 = bitcast i32 %1239 to float, !dbg !11
  %1241 = select i1 %.not, float %1226, float %1240, !dbg !11
  %1242 = select i1 %.not, float %1240, float %1227, !dbg !11
  %1243 = select i1 %.not36, float %1241, float %1235, !dbg !11
  %1244 = icmp samesign ult i32 %.pre-phi29, %.pre-phi27, !dbg !11
  %1245 = select i1 %1244, i32 %.pre-phi29, i32 %lane.sum.lane254.pre-phi, !dbg !11
  %1246 = shl nuw nsw i32 %1245, 2, !dbg !11
  %1247 = bitcast float %1243 to i32, !dbg !11
  %1248 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1247), !dbg !11
  %1249 = bitcast i32 %1248 to float, !dbg !11
  %1250 = select i1 %.not36, float %1242, float %1236, !dbg !11
  %1251 = bitcast float %1250 to i32, !dbg !11
  %1252 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1251), !dbg !11
  %1253 = bitcast i32 %1252 to float, !dbg !11
  %1254 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1255 = shufflevector <4 x i1> %1254, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1256 = insertelement <4 x float> poison, float %1235, i64 0, !dbg !11
  %1257 = insertelement <4 x float> %1256, float %1236, i64 1, !dbg !11
  %1258 = insertelement <4 x float> %1257, float %1249, i64 2, !dbg !11
  %1259 = insertelement <4 x float> %1258, float %1253, i64 3, !dbg !11
  %1260 = insertelement <4 x float> poison, float %1249, i64 0, !dbg !11
  %1261 = insertelement <4 x float> %1260, float %1253, i64 1, !dbg !11
  %1262 = insertelement <4 x float> %1261, float %1241, i64 2, !dbg !11
  %1263 = insertelement <4 x float> %1262, float %1242, i64 3, !dbg !11
  %1264 = select <4 x i1> %1255, <4 x float> %1259, <4 x float> %1263, !dbg !11
  %1265 = extractelement <8 x float> %1223, i64 4, !dbg !11
  %1266 = extractelement <8 x float> %1223, i64 5, !dbg !11
  %1267 = extractelement <8 x float> %1223, i64 6, !dbg !11
  %1268 = extractelement <8 x float> %1223, i64 7, !dbg !11
  %1269 = select i1 %.not, float %1266, float %1265, !dbg !11
  %1270 = bitcast float %1269 to i32, !dbg !11
  %1271 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1270), !dbg !11
  %1272 = bitcast i32 %1271 to float, !dbg !11
  %1273 = select i1 %.not, float %1265, float %1272, !dbg !11
  %1274 = select i1 %.not, float %1272, float %1266, !dbg !11
  %1275 = select i1 %.not, float %1268, float %1267, !dbg !11
  %1276 = bitcast float %1275 to i32, !dbg !11
  %1277 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1276), !dbg !11
  %1278 = bitcast i32 %1277 to float, !dbg !11
  %1279 = select i1 %.not, float %1267, float %1278, !dbg !11
  %1280 = select i1 %.not, float %1278, float %1268, !dbg !11
  %1281 = select i1 %.not36, float %1279, float %1273, !dbg !11
  %1282 = bitcast float %1281 to i32, !dbg !11
  %1283 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1282), !dbg !11
  %1284 = bitcast i32 %1283 to float, !dbg !11
  %1285 = select i1 %.not36, float %1280, float %1274, !dbg !11
  %1286 = bitcast float %1285 to i32, !dbg !11
  %1287 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1286), !dbg !11
  %1288 = bitcast i32 %1287 to float, !dbg !11
  %1289 = insertelement <4 x float> poison, float %1273, i64 0, !dbg !11
  %1290 = insertelement <4 x float> %1289, float %1274, i64 1, !dbg !11
  %1291 = insertelement <4 x float> %1290, float %1284, i64 2, !dbg !11
  %1292 = insertelement <4 x float> %1291, float %1288, i64 3, !dbg !11
  %1293 = insertelement <4 x float> poison, float %1284, i64 0, !dbg !11
  %1294 = insertelement <4 x float> %1293, float %1288, i64 1, !dbg !11
  %1295 = insertelement <4 x float> %1294, float %1279, i64 2, !dbg !11
  %1296 = insertelement <4 x float> %1295, float %1280, i64 3, !dbg !11
  %1297 = select <4 x i1> %1255, <4 x float> %1292, <4 x float> %1296, !dbg !11
  %1298 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !11
  %1299 = extractelement <8 x float> %1298, i64 0, !dbg !11
  %1300 = extractelement <8 x float> %1298, i64 1, !dbg !11
  %1301 = extractelement <8 x float> %1298, i64 2, !dbg !11
  %1302 = extractelement <8 x float> %1298, i64 3, !dbg !11
  %1303 = select i1 %.not, float %1300, float %1299, !dbg !11
  %1304 = bitcast float %1303 to i32, !dbg !11
  %1305 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1304), !dbg !11
  %1306 = bitcast i32 %1305 to float, !dbg !11
  %1307 = select i1 %.not, float %1299, float %1306, !dbg !11
  %1308 = select i1 %.not, float %1306, float %1300, !dbg !11
  %1309 = select i1 %.not, float %1302, float %1301, !dbg !11
  %1310 = bitcast float %1309 to i32, !dbg !11
  %1311 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1310), !dbg !11
  %1312 = bitcast i32 %1311 to float, !dbg !11
  %1313 = select i1 %.not, float %1301, float %1312, !dbg !11
  %1314 = select i1 %.not, float %1312, float %1302, !dbg !11
  %1315 = select i1 %.not36, float %1313, float %1307, !dbg !11
  %1316 = bitcast float %1315 to i32, !dbg !11
  %1317 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1316), !dbg !11
  %1318 = bitcast i32 %1317 to float, !dbg !11
  %1319 = select i1 %.not36, float %1314, float %1308, !dbg !11
  %1320 = bitcast float %1319 to i32, !dbg !11
  %1321 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1320), !dbg !11
  %1322 = bitcast i32 %1321 to float, !dbg !11
  %1323 = insertelement <4 x float> poison, float %1307, i64 0, !dbg !11
  %1324 = insertelement <4 x float> %1323, float %1308, i64 1, !dbg !11
  %1325 = insertelement <4 x float> %1324, float %1318, i64 2, !dbg !11
  %1326 = insertelement <4 x float> %1325, float %1322, i64 3, !dbg !11
  %1327 = insertelement <4 x float> poison, float %1318, i64 0, !dbg !11
  %1328 = insertelement <4 x float> %1327, float %1322, i64 1, !dbg !11
  %1329 = insertelement <4 x float> %1328, float %1313, i64 2, !dbg !11
  %1330 = insertelement <4 x float> %1329, float %1314, i64 3, !dbg !11
  %1331 = select <4 x i1> %1255, <4 x float> %1326, <4 x float> %1330, !dbg !11
  %1332 = extractelement <8 x float> %1298, i64 4, !dbg !11
  %1333 = extractelement <8 x float> %1298, i64 5, !dbg !11
  %1334 = extractelement <8 x float> %1298, i64 6, !dbg !11
  %1335 = extractelement <8 x float> %1298, i64 7, !dbg !11
  %1336 = select i1 %.not, float %1333, float %1332, !dbg !11
  %1337 = bitcast float %1336 to i32, !dbg !11
  %1338 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1337), !dbg !11
  %1339 = bitcast i32 %1338 to float, !dbg !11
  %1340 = select i1 %.not, float %1332, float %1339, !dbg !11
  %1341 = select i1 %.not, float %1339, float %1333, !dbg !11
  %1342 = select i1 %.not, float %1335, float %1334, !dbg !11
  %1343 = bitcast float %1342 to i32, !dbg !11
  %1344 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1343), !dbg !11
  %1345 = bitcast i32 %1344 to float, !dbg !11
  %1346 = select i1 %.not, float %1334, float %1345, !dbg !11
  %1347 = select i1 %.not, float %1345, float %1335, !dbg !11
  %1348 = select i1 %.not36, float %1346, float %1340, !dbg !11
  %1349 = bitcast float %1348 to i32, !dbg !11
  %1350 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1349), !dbg !11
  %1351 = bitcast i32 %1350 to float, !dbg !11
  %1352 = select i1 %.not36, float %1347, float %1341, !dbg !11
  %1353 = bitcast float %1352 to i32, !dbg !11
  %1354 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1353), !dbg !11
  %1355 = bitcast i32 %1354 to float, !dbg !11
  %1356 = insertelement <4 x float> poison, float %1340, i64 0, !dbg !11
  %1357 = insertelement <4 x float> %1356, float %1341, i64 1, !dbg !11
  %1358 = insertelement <4 x float> %1357, float %1351, i64 2, !dbg !11
  %1359 = insertelement <4 x float> %1358, float %1355, i64 3, !dbg !11
  %1360 = insertelement <4 x float> poison, float %1351, i64 0, !dbg !11
  %1361 = insertelement <4 x float> %1360, float %1355, i64 1, !dbg !11
  %1362 = insertelement <4 x float> %1361, float %1346, i64 2, !dbg !11
  %1363 = insertelement <4 x float> %1362, float %1347, i64 3, !dbg !11
  %1364 = select <4 x i1> %1255, <4 x float> %1359, <4 x float> %1363, !dbg !11
  %1365 = extractvalue [2 x <8 x float>] %.lcssa4, 0, !dbg !11
  %1366 = extractelement <8 x float> %1365, i64 0, !dbg !11
  %1367 = extractelement <8 x float> %1365, i64 1, !dbg !11
  %1368 = extractelement <8 x float> %1365, i64 2, !dbg !11
  %1369 = extractelement <8 x float> %1365, i64 3, !dbg !11
  %1370 = select i1 %.not, float %1367, float %1366, !dbg !11
  %1371 = bitcast float %1370 to i32, !dbg !11
  %1372 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1371), !dbg !11
  %1373 = bitcast i32 %1372 to float, !dbg !11
  %1374 = select i1 %.not, float %1366, float %1373, !dbg !11
  %1375 = select i1 %.not, float %1373, float %1367, !dbg !11
  %1376 = select i1 %.not, float %1369, float %1368, !dbg !11
  %1377 = bitcast float %1376 to i32, !dbg !11
  %1378 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1377), !dbg !11
  %1379 = bitcast i32 %1378 to float, !dbg !11
  %1380 = select i1 %.not, float %1368, float %1379, !dbg !11
  %1381 = select i1 %.not, float %1379, float %1369, !dbg !11
  %1382 = select i1 %.not36, float %1380, float %1374, !dbg !11
  %1383 = bitcast float %1382 to i32, !dbg !11
  %1384 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1383), !dbg !11
  %1385 = bitcast i32 %1384 to float, !dbg !11
  %1386 = select i1 %.not36, float %1381, float %1375, !dbg !11
  %1387 = bitcast float %1386 to i32, !dbg !11
  %1388 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1387), !dbg !11
  %1389 = bitcast i32 %1388 to float, !dbg !11
  %1390 = insertelement <4 x float> poison, float %1374, i64 0, !dbg !11
  %1391 = insertelement <4 x float> %1390, float %1375, i64 1, !dbg !11
  %1392 = insertelement <4 x float> %1391, float %1385, i64 2, !dbg !11
  %1393 = insertelement <4 x float> %1392, float %1389, i64 3, !dbg !11
  %1394 = insertelement <4 x float> poison, float %1385, i64 0, !dbg !11
  %1395 = insertelement <4 x float> %1394, float %1389, i64 1, !dbg !11
  %1396 = insertelement <4 x float> %1395, float %1380, i64 2, !dbg !11
  %1397 = insertelement <4 x float> %1396, float %1381, i64 3, !dbg !11
  %1398 = select <4 x i1> %1255, <4 x float> %1393, <4 x float> %1397, !dbg !11
  %1399 = extractelement <8 x float> %1365, i64 4, !dbg !11
  %1400 = extractelement <8 x float> %1365, i64 5, !dbg !11
  %1401 = extractelement <8 x float> %1365, i64 6, !dbg !11
  %1402 = extractelement <8 x float> %1365, i64 7, !dbg !11
  %1403 = select i1 %.not, float %1400, float %1399, !dbg !11
  %1404 = bitcast float %1403 to i32, !dbg !11
  %1405 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1404), !dbg !11
  %1406 = bitcast i32 %1405 to float, !dbg !11
  %1407 = select i1 %.not, float %1399, float %1406, !dbg !11
  %1408 = select i1 %.not, float %1406, float %1400, !dbg !11
  %1409 = select i1 %.not, float %1402, float %1401, !dbg !11
  %1410 = bitcast float %1409 to i32, !dbg !11
  %1411 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1410), !dbg !11
  %1412 = bitcast i32 %1411 to float, !dbg !11
  %1413 = select i1 %.not, float %1401, float %1412, !dbg !11
  %1414 = select i1 %.not, float %1412, float %1402, !dbg !11
  %1415 = select i1 %.not36, float %1413, float %1407, !dbg !11
  %1416 = bitcast float %1415 to i32, !dbg !11
  %1417 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1416), !dbg !11
  %1418 = bitcast i32 %1417 to float, !dbg !11
  %1419 = select i1 %.not36, float %1414, float %1408, !dbg !11
  %1420 = bitcast float %1419 to i32, !dbg !11
  %1421 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1420), !dbg !11
  %1422 = bitcast i32 %1421 to float, !dbg !11
  %1423 = insertelement <4 x float> poison, float %1407, i64 0, !dbg !11
  %1424 = insertelement <4 x float> %1423, float %1408, i64 1, !dbg !11
  %1425 = insertelement <4 x float> %1424, float %1418, i64 2, !dbg !11
  %1426 = insertelement <4 x float> %1425, float %1422, i64 3, !dbg !11
  %1427 = insertelement <4 x float> poison, float %1418, i64 0, !dbg !11
  %1428 = insertelement <4 x float> %1427, float %1422, i64 1, !dbg !11
  %1429 = insertelement <4 x float> %1428, float %1413, i64 2, !dbg !11
  %1430 = insertelement <4 x float> %1429, float %1414, i64 3, !dbg !11
  %1431 = select <4 x i1> %1255, <4 x float> %1426, <4 x float> %1430, !dbg !11
  %1432 = extractvalue [2 x <8 x float>] %.lcssa4, 1, !dbg !11
  %1433 = extractelement <8 x float> %1432, i64 0, !dbg !11
  %1434 = extractelement <8 x float> %1432, i64 1, !dbg !11
  %1435 = extractelement <8 x float> %1432, i64 2, !dbg !11
  %1436 = extractelement <8 x float> %1432, i64 3, !dbg !11
  %1437 = select i1 %.not, float %1434, float %1433, !dbg !11
  %1438 = bitcast float %1437 to i32, !dbg !11
  %1439 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1438), !dbg !11
  %1440 = bitcast i32 %1439 to float, !dbg !11
  %1441 = select i1 %.not, float %1433, float %1440, !dbg !11
  %1442 = select i1 %.not, float %1440, float %1434, !dbg !11
  %1443 = select i1 %.not, float %1436, float %1435, !dbg !11
  %1444 = bitcast float %1443 to i32, !dbg !11
  %1445 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1444), !dbg !11
  %1446 = bitcast i32 %1445 to float, !dbg !11
  %1447 = select i1 %.not, float %1435, float %1446, !dbg !11
  %1448 = select i1 %.not, float %1446, float %1436, !dbg !11
  %1449 = select i1 %.not36, float %1447, float %1441, !dbg !11
  %1450 = bitcast float %1449 to i32, !dbg !11
  %1451 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1450), !dbg !11
  %1452 = bitcast i32 %1451 to float, !dbg !11
  %1453 = select i1 %.not36, float %1448, float %1442, !dbg !11
  %1454 = bitcast float %1453 to i32, !dbg !11
  %1455 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1454), !dbg !11
  %1456 = bitcast i32 %1455 to float, !dbg !11
  %1457 = insertelement <4 x float> poison, float %1441, i64 0, !dbg !11
  %1458 = insertelement <4 x float> %1457, float %1442, i64 1, !dbg !11
  %1459 = insertelement <4 x float> %1458, float %1452, i64 2, !dbg !11
  %1460 = insertelement <4 x float> %1459, float %1456, i64 3, !dbg !11
  %1461 = insertelement <4 x float> poison, float %1452, i64 0, !dbg !11
  %1462 = insertelement <4 x float> %1461, float %1456, i64 1, !dbg !11
  %1463 = insertelement <4 x float> %1462, float %1447, i64 2, !dbg !11
  %1464 = insertelement <4 x float> %1463, float %1448, i64 3, !dbg !11
  %1465 = select <4 x i1> %1255, <4 x float> %1460, <4 x float> %1464, !dbg !11
  %1466 = extractelement <8 x float> %1432, i64 4, !dbg !11
  %1467 = extractelement <8 x float> %1432, i64 5, !dbg !11
  %1468 = extractelement <8 x float> %1432, i64 6, !dbg !11
  %1469 = extractelement <8 x float> %1432, i64 7, !dbg !11
  %1470 = select i1 %.not, float %1467, float %1466, !dbg !11
  %1471 = bitcast float %1470 to i32, !dbg !11
  %1472 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1471), !dbg !11
  %1473 = bitcast i32 %1472 to float, !dbg !11
  %1474 = select i1 %.not, float %1466, float %1473, !dbg !11
  %1475 = select i1 %.not, float %1473, float %1467, !dbg !11
  %1476 = select i1 %.not, float %1469, float %1468, !dbg !11
  %1477 = bitcast float %1476 to i32, !dbg !11
  %1478 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1477), !dbg !11
  %1479 = bitcast i32 %1478 to float, !dbg !11
  %1480 = select i1 %.not, float %1468, float %1479, !dbg !11
  %1481 = select i1 %.not, float %1479, float %1469, !dbg !11
  %1482 = select i1 %.not36, float %1480, float %1474, !dbg !11
  %1483 = bitcast float %1482 to i32, !dbg !11
  %1484 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1483), !dbg !11
  %1485 = bitcast i32 %1484 to float, !dbg !11
  %1486 = select i1 %.not36, float %1481, float %1475, !dbg !11
  %1487 = bitcast float %1486 to i32, !dbg !11
  %1488 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1487), !dbg !11
  %1489 = bitcast i32 %1488 to float, !dbg !11
  %1490 = insertelement <4 x float> poison, float %1474, i64 0, !dbg !11
  %1491 = insertelement <4 x float> %1490, float %1475, i64 1, !dbg !11
  %1492 = insertelement <4 x float> %1491, float %1485, i64 2, !dbg !11
  %1493 = insertelement <4 x float> %1492, float %1489, i64 3, !dbg !11
  %1494 = insertelement <4 x float> poison, float %1485, i64 0, !dbg !11
  %1495 = insertelement <4 x float> %1494, float %1489, i64 1, !dbg !11
  %1496 = insertelement <4 x float> %1495, float %1480, i64 2, !dbg !11
  %1497 = insertelement <4 x float> %1496, float %1481, i64 3, !dbg !11
  %1498 = select <4 x i1> %1255, <4 x float> %1493, <4 x float> %1497, !dbg !11
  %1499 = extractvalue [2 x <8 x float>] %.lcssa5, 0, !dbg !11
  %1500 = extractelement <8 x float> %1499, i64 0, !dbg !11
  %1501 = extractelement <8 x float> %1499, i64 1, !dbg !11
  %1502 = extractelement <8 x float> %1499, i64 2, !dbg !11
  %1503 = extractelement <8 x float> %1499, i64 3, !dbg !11
  %1504 = select i1 %.not, float %1501, float %1500, !dbg !11
  %1505 = bitcast float %1504 to i32, !dbg !11
  %1506 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1505), !dbg !11
  %1507 = bitcast i32 %1506 to float, !dbg !11
  %1508 = select i1 %.not, float %1500, float %1507, !dbg !11
  %1509 = select i1 %.not, float %1507, float %1501, !dbg !11
  %1510 = select i1 %.not, float %1503, float %1502, !dbg !11
  %1511 = bitcast float %1510 to i32, !dbg !11
  %1512 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1511), !dbg !11
  %1513 = bitcast i32 %1512 to float, !dbg !11
  %1514 = select i1 %.not, float %1502, float %1513, !dbg !11
  %1515 = select i1 %.not, float %1513, float %1503, !dbg !11
  %1516 = select i1 %.not36, float %1514, float %1508, !dbg !11
  %1517 = bitcast float %1516 to i32, !dbg !11
  %1518 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1517), !dbg !11
  %1519 = bitcast i32 %1518 to float, !dbg !11
  %1520 = select i1 %.not36, float %1515, float %1509, !dbg !11
  %1521 = bitcast float %1520 to i32, !dbg !11
  %1522 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1521), !dbg !11
  %1523 = bitcast i32 %1522 to float, !dbg !11
  %1524 = insertelement <4 x float> poison, float %1508, i64 0, !dbg !11
  %1525 = insertelement <4 x float> %1524, float %1509, i64 1, !dbg !11
  %1526 = insertelement <4 x float> %1525, float %1519, i64 2, !dbg !11
  %1527 = insertelement <4 x float> %1526, float %1523, i64 3, !dbg !11
  %1528 = insertelement <4 x float> poison, float %1519, i64 0, !dbg !11
  %1529 = insertelement <4 x float> %1528, float %1523, i64 1, !dbg !11
  %1530 = insertelement <4 x float> %1529, float %1514, i64 2, !dbg !11
  %1531 = insertelement <4 x float> %1530, float %1515, i64 3, !dbg !11
  %1532 = select <4 x i1> %1255, <4 x float> %1527, <4 x float> %1531, !dbg !11
  %1533 = extractelement <8 x float> %1499, i64 4, !dbg !11
  %1534 = extractelement <8 x float> %1499, i64 5, !dbg !11
  %1535 = extractelement <8 x float> %1499, i64 6, !dbg !11
  %1536 = extractelement <8 x float> %1499, i64 7, !dbg !11
  %1537 = select i1 %.not, float %1534, float %1533, !dbg !11
  %1538 = bitcast float %1537 to i32, !dbg !11
  %1539 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1538), !dbg !11
  %1540 = bitcast i32 %1539 to float, !dbg !11
  %1541 = select i1 %.not, float %1533, float %1540, !dbg !11
  %1542 = select i1 %.not, float %1540, float %1534, !dbg !11
  %1543 = select i1 %.not, float %1536, float %1535, !dbg !11
  %1544 = bitcast float %1543 to i32, !dbg !11
  %1545 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1544), !dbg !11
  %1546 = bitcast i32 %1545 to float, !dbg !11
  %1547 = select i1 %.not, float %1535, float %1546, !dbg !11
  %1548 = select i1 %.not, float %1546, float %1536, !dbg !11
  %1549 = select i1 %.not36, float %1547, float %1541, !dbg !11
  %1550 = bitcast float %1549 to i32, !dbg !11
  %1551 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1550), !dbg !11
  %1552 = bitcast i32 %1551 to float, !dbg !11
  %1553 = select i1 %.not36, float %1548, float %1542, !dbg !11
  %1554 = bitcast float %1553 to i32, !dbg !11
  %1555 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1554), !dbg !11
  %1556 = bitcast i32 %1555 to float, !dbg !11
  %1557 = insertelement <4 x float> poison, float %1541, i64 0, !dbg !11
  %1558 = insertelement <4 x float> %1557, float %1542, i64 1, !dbg !11
  %1559 = insertelement <4 x float> %1558, float %1552, i64 2, !dbg !11
  %1560 = insertelement <4 x float> %1559, float %1556, i64 3, !dbg !11
  %1561 = insertelement <4 x float> poison, float %1552, i64 0, !dbg !11
  %1562 = insertelement <4 x float> %1561, float %1556, i64 1, !dbg !11
  %1563 = insertelement <4 x float> %1562, float %1547, i64 2, !dbg !11
  %1564 = insertelement <4 x float> %1563, float %1548, i64 3, !dbg !11
  %1565 = select <4 x i1> %1255, <4 x float> %1560, <4 x float> %1564, !dbg !11
  %1566 = extractvalue [2 x <8 x float>] %.lcssa5, 1, !dbg !11
  %1567 = extractelement <8 x float> %1566, i64 0, !dbg !11
  %1568 = extractelement <8 x float> %1566, i64 1, !dbg !11
  %1569 = extractelement <8 x float> %1566, i64 2, !dbg !11
  %1570 = extractelement <8 x float> %1566, i64 3, !dbg !11
  %1571 = select i1 %.not, float %1568, float %1567, !dbg !11
  %1572 = bitcast float %1571 to i32, !dbg !11
  %1573 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1572), !dbg !11
  %1574 = bitcast i32 %1573 to float, !dbg !11
  %1575 = select i1 %.not, float %1567, float %1574, !dbg !11
  %1576 = select i1 %.not, float %1574, float %1568, !dbg !11
  %1577 = select i1 %.not, float %1570, float %1569, !dbg !11
  %1578 = bitcast float %1577 to i32, !dbg !11
  %1579 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1578), !dbg !11
  %1580 = bitcast i32 %1579 to float, !dbg !11
  %1581 = select i1 %.not, float %1569, float %1580, !dbg !11
  %1582 = select i1 %.not, float %1580, float %1570, !dbg !11
  %1583 = select i1 %.not36, float %1581, float %1575, !dbg !11
  %1584 = bitcast float %1583 to i32, !dbg !11
  %1585 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1584), !dbg !11
  %1586 = bitcast i32 %1585 to float, !dbg !11
  %1587 = select i1 %.not36, float %1582, float %1576, !dbg !11
  %1588 = bitcast float %1587 to i32, !dbg !11
  %1589 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1588), !dbg !11
  %1590 = bitcast i32 %1589 to float, !dbg !11
  %1591 = insertelement <4 x float> poison, float %1575, i64 0, !dbg !11
  %1592 = insertelement <4 x float> %1591, float %1576, i64 1, !dbg !11
  %1593 = insertelement <4 x float> %1592, float %1586, i64 2, !dbg !11
  %1594 = insertelement <4 x float> %1593, float %1590, i64 3, !dbg !11
  %1595 = insertelement <4 x float> poison, float %1586, i64 0, !dbg !11
  %1596 = insertelement <4 x float> %1595, float %1590, i64 1, !dbg !11
  %1597 = insertelement <4 x float> %1596, float %1581, i64 2, !dbg !11
  %1598 = insertelement <4 x float> %1597, float %1582, i64 3, !dbg !11
  %1599 = select <4 x i1> %1255, <4 x float> %1594, <4 x float> %1598, !dbg !11
  %1600 = extractelement <8 x float> %1566, i64 4, !dbg !11
  %1601 = extractelement <8 x float> %1566, i64 5, !dbg !11
  %1602 = extractelement <8 x float> %1566, i64 6, !dbg !11
  %1603 = extractelement <8 x float> %1566, i64 7, !dbg !11
  %1604 = select i1 %.not, float %1601, float %1600, !dbg !11
  %1605 = bitcast float %1604 to i32, !dbg !11
  %1606 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1605), !dbg !11
  %1607 = bitcast i32 %1606 to float, !dbg !11
  %1608 = select i1 %.not, float %1600, float %1607, !dbg !11
  %1609 = select i1 %.not, float %1607, float %1601, !dbg !11
  %1610 = select i1 %.not, float %1603, float %1602, !dbg !11
  %1611 = bitcast float %1610 to i32, !dbg !11
  %1612 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1611), !dbg !11
  %1613 = bitcast i32 %1612 to float, !dbg !11
  %1614 = select i1 %.not, float %1602, float %1613, !dbg !11
  %1615 = select i1 %.not, float %1613, float %1603, !dbg !11
  %1616 = select i1 %.not36, float %1614, float %1608, !dbg !11
  %1617 = bitcast float %1616 to i32, !dbg !11
  %1618 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1617), !dbg !11
  %1619 = bitcast i32 %1618 to float, !dbg !11
  %1620 = select i1 %.not36, float %1615, float %1609, !dbg !11
  %1621 = bitcast float %1620 to i32, !dbg !11
  %1622 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1621), !dbg !11
  %1623 = bitcast i32 %1622 to float, !dbg !11
  %1624 = insertelement <4 x float> poison, float %1608, i64 0, !dbg !11
  %1625 = insertelement <4 x float> %1624, float %1609, i64 1, !dbg !11
  %1626 = insertelement <4 x float> %1625, float %1619, i64 2, !dbg !11
  %1627 = insertelement <4 x float> %1626, float %1623, i64 3, !dbg !11
  %1628 = insertelement <4 x float> poison, float %1619, i64 0, !dbg !11
  %1629 = insertelement <4 x float> %1628, float %1623, i64 1, !dbg !11
  %1630 = insertelement <4 x float> %1629, float %1614, i64 2, !dbg !11
  %1631 = insertelement <4 x float> %1630, float %1615, i64 3, !dbg !11
  %1632 = select <4 x i1> %1255, <4 x float> %1627, <4 x float> %1631, !dbg !11
  %1633 = extractvalue [2 x <8 x float>] %.lcssa7, 0, !dbg !11
  %1634 = extractelement <8 x float> %1633, i64 0, !dbg !11
  %1635 = extractelement <8 x float> %1633, i64 1, !dbg !11
  %1636 = extractelement <8 x float> %1633, i64 2, !dbg !11
  %1637 = extractelement <8 x float> %1633, i64 3, !dbg !11
  %1638 = select i1 %.not, float %1635, float %1634, !dbg !11
  %1639 = bitcast float %1638 to i32, !dbg !11
  %1640 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1639), !dbg !11
  %1641 = bitcast i32 %1640 to float, !dbg !11
  %1642 = select i1 %.not, float %1634, float %1641, !dbg !11
  %1643 = select i1 %.not, float %1641, float %1635, !dbg !11
  %1644 = select i1 %.not, float %1637, float %1636, !dbg !11
  %1645 = bitcast float %1644 to i32, !dbg !11
  %1646 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1645), !dbg !11
  %1647 = bitcast i32 %1646 to float, !dbg !11
  %1648 = select i1 %.not, float %1636, float %1647, !dbg !11
  %1649 = select i1 %.not, float %1647, float %1637, !dbg !11
  %1650 = select i1 %.not36, float %1648, float %1642, !dbg !11
  %1651 = bitcast float %1650 to i32, !dbg !11
  %1652 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1651), !dbg !11
  %1653 = select i1 %.not36, float %1649, float %1643, !dbg !11
  %1654 = bitcast float %1653 to i32, !dbg !11
  %1655 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1654), !dbg !11
  %1656 = extractelement <8 x float> %1633, i64 4, !dbg !11
  %1657 = extractelement <8 x float> %1633, i64 5, !dbg !11
  %1658 = extractelement <8 x float> %1633, i64 6, !dbg !11
  %1659 = extractelement <8 x float> %1633, i64 7, !dbg !11
  %1660 = select i1 %.not, float %1657, float %1656, !dbg !11
  %1661 = bitcast float %1660 to i32, !dbg !11
  %1662 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1661), !dbg !11
  %1663 = bitcast i32 %1662 to float, !dbg !11
  %1664 = select i1 %.not, float %1656, float %1663, !dbg !11
  %1665 = select i1 %.not, float %1663, float %1657, !dbg !11
  %1666 = select i1 %.not, float %1659, float %1658, !dbg !11
  %1667 = bitcast float %1666 to i32, !dbg !11
  %1668 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1667), !dbg !11
  %1669 = bitcast i32 %1668 to float, !dbg !11
  %1670 = select i1 %.not, float %1658, float %1669, !dbg !11
  %1671 = select i1 %.not, float %1669, float %1659, !dbg !11
  %1672 = select i1 %.not36, float %1670, float %1664, !dbg !11
  %1673 = bitcast float %1672 to i32, !dbg !11
  %1674 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1673), !dbg !11
  %1675 = select i1 %.not36, float %1671, float %1665, !dbg !11
  %1676 = bitcast float %1675 to i32, !dbg !11
  %1677 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1676), !dbg !11
  %1678 = extractvalue [2 x <8 x float>] %.lcssa7, 1, !dbg !11
  %1679 = extractelement <8 x float> %1678, i64 0, !dbg !11
  %1680 = extractelement <8 x float> %1678, i64 1, !dbg !11
  %1681 = extractelement <8 x float> %1678, i64 2, !dbg !11
  %1682 = extractelement <8 x float> %1678, i64 3, !dbg !11
  %1683 = select i1 %.not, float %1680, float %1679, !dbg !11
  %1684 = bitcast float %1683 to i32, !dbg !11
  %1685 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1684), !dbg !11
  %1686 = bitcast i32 %1685 to float, !dbg !11
  %1687 = select i1 %.not, float %1679, float %1686, !dbg !11
  %1688 = select i1 %.not, float %1686, float %1680, !dbg !11
  %1689 = select i1 %.not, float %1682, float %1681, !dbg !11
  %1690 = bitcast float %1689 to i32, !dbg !11
  %1691 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1690), !dbg !11
  %1692 = bitcast i32 %1691 to float, !dbg !11
  %1693 = select i1 %.not, float %1681, float %1692, !dbg !11
  %1694 = select i1 %.not, float %1692, float %1682, !dbg !11
  %1695 = select i1 %.not36, float %1693, float %1687, !dbg !11
  %1696 = bitcast float %1695 to i32, !dbg !11
  %1697 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1696), !dbg !11
  %1698 = select i1 %.not36, float %1694, float %1688, !dbg !11
  %1699 = bitcast float %1698 to i32, !dbg !11
  %1700 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1699), !dbg !11
  %1701 = extractelement <8 x float> %1678, i64 4, !dbg !11
  %1702 = extractelement <8 x float> %1678, i64 5, !dbg !11
  %1703 = extractelement <8 x float> %1678, i64 6, !dbg !11
  %1704 = extractelement <8 x float> %1678, i64 7, !dbg !11
  %1705 = select i1 %.not, float %1702, float %1701, !dbg !11
  %1706 = bitcast float %1705 to i32, !dbg !11
  %1707 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1706), !dbg !11
  %1708 = bitcast i32 %1707 to float, !dbg !11
  %1709 = select i1 %.not, float %1701, float %1708, !dbg !11
  %1710 = select i1 %.not, float %1708, float %1702, !dbg !11
  %1711 = select i1 %.not, float %1704, float %1703, !dbg !11
  %1712 = bitcast float %1711 to i32, !dbg !11
  %1713 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1231, i32 %1712), !dbg !11
  %1714 = bitcast i32 %1713 to float, !dbg !11
  %1715 = select i1 %.not, float %1703, float %1714, !dbg !11
  %1716 = select i1 %.not, float %1714, float %1704, !dbg !11
  %1717 = select i1 %.not36, float %1715, float %1709, !dbg !11
  %1718 = bitcast float %1717 to i32, !dbg !11
  %1719 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1718), !dbg !11
  %1720 = select i1 %.not36, float %1716, float %1710, !dbg !11
  %1721 = bitcast float %1720 to i32, !dbg !11
  %1722 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1246, i32 %1721), !dbg !11
  %1723 = shufflevector <1 x float> %1220, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !53
  %1724 = shufflevector <1 x float> %1215, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !53
  %1725 = bitcast i32 %1719 to float, !dbg !11
  %1726 = bitcast i32 %1722 to float, !dbg !11
  %1727 = insertelement <4 x float> poison, float %1709, i64 0, !dbg !11
  %1728 = insertelement <4 x float> %1727, float %1710, i64 1, !dbg !11
  %1729 = insertelement <4 x float> %1728, float %1725, i64 2, !dbg !11
  %1730 = insertelement <4 x float> %1729, float %1726, i64 3, !dbg !11
  %1731 = insertelement <4 x float> poison, float %1725, i64 0, !dbg !11
  %1732 = insertelement <4 x float> %1731, float %1726, i64 1, !dbg !11
  %1733 = insertelement <4 x float> %1732, float %1715, i64 2, !dbg !11
  %1734 = insertelement <4 x float> %1733, float %1716, i64 3, !dbg !11
  %1735 = select <4 x i1> %1255, <4 x float> %1730, <4 x float> %1734, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1724, !dbg !53
  %normalized = fmul <4 x float> %row.reciprocal, %1735, !dbg !53
  %1736 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !54
  %1737 = extractelement <4 x half> %1736, i64 3, !dbg !54
  %1738 = extractelement <4 x half> %1736, i64 2, !dbg !54
  %1739 = extractelement <4 x half> %1736, i64 1, !dbg !54
  %1740 = extractelement <4 x half> %1736, i64 0, !dbg !54
  %1741 = bitcast i32 %1697 to float, !dbg !11
  %1742 = bitcast i32 %1700 to float, !dbg !11
  %1743 = insertelement <4 x float> poison, float %1687, i64 0, !dbg !11
  %1744 = insertelement <4 x float> %1743, float %1688, i64 1, !dbg !11
  %1745 = insertelement <4 x float> %1744, float %1741, i64 2, !dbg !11
  %1746 = insertelement <4 x float> %1745, float %1742, i64 3, !dbg !11
  %1747 = insertelement <4 x float> poison, float %1741, i64 0, !dbg !11
  %1748 = insertelement <4 x float> %1747, float %1742, i64 1, !dbg !11
  %1749 = insertelement <4 x float> %1748, float %1693, i64 2, !dbg !11
  %1750 = insertelement <4 x float> %1749, float %1694, i64 3, !dbg !11
  %1751 = select <4 x i1> %1255, <4 x float> %1746, <4 x float> %1750, !dbg !11
  %normalized104 = fmul <4 x float> %row.reciprocal, %1751, !dbg !53
  %1752 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !54
  %1753 = extractelement <4 x half> %1752, i64 3, !dbg !54
  %1754 = extractelement <4 x half> %1752, i64 2, !dbg !54
  %1755 = extractelement <4 x half> %1752, i64 1, !dbg !54
  %1756 = extractelement <4 x half> %1752, i64 0, !dbg !54
  %1757 = bitcast i32 %1674 to float, !dbg !11
  %1758 = bitcast i32 %1677 to float, !dbg !11
  %1759 = insertelement <4 x float> poison, float %1664, i64 0, !dbg !11
  %1760 = insertelement <4 x float> %1759, float %1665, i64 1, !dbg !11
  %1761 = insertelement <4 x float> %1760, float %1757, i64 2, !dbg !11
  %1762 = insertelement <4 x float> %1761, float %1758, i64 3, !dbg !11
  %1763 = insertelement <4 x float> poison, float %1757, i64 0, !dbg !11
  %1764 = insertelement <4 x float> %1763, float %1758, i64 1, !dbg !11
  %1765 = insertelement <4 x float> %1764, float %1670, i64 2, !dbg !11
  %1766 = insertelement <4 x float> %1765, float %1671, i64 3, !dbg !11
  %1767 = select <4 x i1> %1255, <4 x float> %1762, <4 x float> %1766, !dbg !11
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1723, !dbg !53
  %normalized106 = fmul <4 x float> %row.reciprocal105, %1767, !dbg !53
  %1768 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !54
  %1769 = extractelement <4 x half> %1768, i64 3, !dbg !54
  %1770 = extractelement <4 x half> %1768, i64 2, !dbg !54
  %1771 = extractelement <4 x half> %1768, i64 1, !dbg !54
  %1772 = extractelement <4 x half> %1768, i64 0, !dbg !54
  %1773 = bitcast i32 %1652 to float, !dbg !11
  %1774 = bitcast i32 %1655 to float, !dbg !11
  %1775 = insertelement <4 x float> poison, float %1642, i64 0, !dbg !11
  %1776 = insertelement <4 x float> %1775, float %1643, i64 1, !dbg !11
  %1777 = insertelement <4 x float> %1776, float %1773, i64 2, !dbg !11
  %1778 = insertelement <4 x float> %1777, float %1774, i64 3, !dbg !11
  %1779 = insertelement <4 x float> poison, float %1773, i64 0, !dbg !11
  %1780 = insertelement <4 x float> %1779, float %1774, i64 1, !dbg !11
  %1781 = insertelement <4 x float> %1780, float %1648, i64 2, !dbg !11
  %1782 = insertelement <4 x float> %1781, float %1649, i64 3, !dbg !11
  %1783 = select <4 x i1> %1255, <4 x float> %1778, <4 x float> %1782, !dbg !11
  %normalized107 = fmul <4 x float> %row.reciprocal105, %1783, !dbg !53
  %1784 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !54
  %1785 = extractelement <4 x half> %1784, i64 3, !dbg !54
  %1786 = extractelement <4 x half> %1784, i64 2, !dbg !54
  %1787 = extractelement <4 x half> %1784, i64 1, !dbg !54
  %1788 = extractelement <4 x half> %1784, i64 0, !dbg !54
  %address.thread1041 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !55, !range !9, !frisk.register_lifetimes !5
  %address.local1042 = zext nneg i32 %address.thread1041 to i64, !dbg !8
  %address.local1044 = lshr i64 %address.local1042, 2, !dbg !15
  %address.local1045 = and i64 %address.local1044, 48, !dbg !15
  %address.local1047 = and i64 %address.local1042, 240, !dbg !55
  %address.local1046 = add i64 %31, %address.local1042, !dbg !55
  %address.local1049 = sub i64 %address.local1046, %address.local1047, !dbg !55
  %address.local1050 = add i64 %address.local1049, %address.local1045, !dbg !55
  %1789 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %40, !dbg !55
  %address.local1053.idx = shl nuw nsw i64 %address.local1050, 8, !dbg !55
  %address.local1053 = getelementptr inbounds nuw i8, ptr addrspace(1) %1789, i64 %address.local1053.idx, !dbg !55
  %address.local1055 = and i64 %address.local1044, 12, !dbg !10
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1055, !dbg !55
  store half %1788, ptr addrspace(1) %address.local1056, align 2, !dbg !55
  %address.local1058 = or disjoint i64 %address.local1055, 1, !dbg !55
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1058, !dbg !55
  store half %1787, ptr addrspace(1) %address.local1059, align 2, !dbg !55
  %address.local1061 = or disjoint i64 %address.local1055, 2, !dbg !55
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1061, !dbg !55
  store half %1786, ptr addrspace(1) %address.local1062, align 2, !dbg !55
  %address.local1064 = or disjoint i64 %address.local1055, 3, !dbg !55
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1064, !dbg !55
  store half %1785, ptr addrspace(1) %address.local1065, align 2, !dbg !55
  %address.local1067 = or disjoint i64 %address.local1055, 16, !dbg !55
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1067, !dbg !55
  store half %1772, ptr addrspace(1) %address.local1068, align 2, !dbg !55
  %address.local1070 = or disjoint i64 %address.local1055, 17, !dbg !55
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1070, !dbg !55
  store half %1771, ptr addrspace(1) %address.local1071, align 2, !dbg !55
  %address.local1073 = or disjoint i64 %address.local1055, 18, !dbg !55
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1073, !dbg !55
  store half %1770, ptr addrspace(1) %address.local1074, align 2, !dbg !55
  %address.local1076 = or disjoint i64 %address.local1055, 19, !dbg !55
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1076, !dbg !55
  store half %1769, ptr addrspace(1) %address.local1077, align 2, !dbg !55
  %address.local1081 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1053, i64 16384, !dbg !55
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1055, !dbg !55
  store half %1756, ptr addrspace(1) %address.local1082, align 2, !dbg !55
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1058, !dbg !55
  store half %1755, ptr addrspace(1) %address.local1084, align 2, !dbg !55
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !55
  store half %1754, ptr addrspace(1) %address.local1086, align 2, !dbg !55
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1064, !dbg !55
  store half %1753, ptr addrspace(1) %address.local1088, align 2, !dbg !55
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1067, !dbg !55
  store half %1740, ptr addrspace(1) %address.local1090, align 2, !dbg !55
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1070, !dbg !55
  store half %1739, ptr addrspace(1) %address.local1092, align 2, !dbg !55
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1073, !dbg !55
  store half %1738, ptr addrspace(1) %address.local1094, align 2, !dbg !55
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1076, !dbg !55
  store half %1737, ptr addrspace(1) %address.local1096, align 2, !dbg !55
  %normalized109 = fmul <4 x float> %row.reciprocal, %1632, !dbg !53
  %1790 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !54
  %1791 = extractelement <4 x half> %1790, i64 3, !dbg !54
  %1792 = extractelement <4 x half> %1790, i64 2, !dbg !54
  %1793 = extractelement <4 x half> %1790, i64 1, !dbg !54
  %1794 = extractelement <4 x half> %1790, i64 0, !dbg !54
  %normalized110 = fmul <4 x float> %row.reciprocal, %1599, !dbg !53
  %1795 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !54
  %1796 = extractelement <4 x half> %1795, i64 3, !dbg !54
  %1797 = extractelement <4 x half> %1795, i64 2, !dbg !54
  %1798 = extractelement <4 x half> %1795, i64 1, !dbg !54
  %1799 = extractelement <4 x half> %1795, i64 0, !dbg !54
  %normalized112 = fmul <4 x float> %row.reciprocal105, %1565, !dbg !53
  %1800 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !54
  %1801 = extractelement <4 x half> %1800, i64 3, !dbg !54
  %1802 = extractelement <4 x half> %1800, i64 2, !dbg !54
  %1803 = extractelement <4 x half> %1800, i64 1, !dbg !54
  %1804 = extractelement <4 x half> %1800, i64 0, !dbg !54
  %normalized113 = fmul <4 x float> %row.reciprocal105, %1532, !dbg !53
  %1805 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !54
  %1806 = extractelement <4 x half> %1805, i64 3, !dbg !54
  %1807 = extractelement <4 x half> %1805, i64 2, !dbg !54
  %1808 = extractelement <4 x half> %1805, i64 1, !dbg !54
  %1809 = extractelement <4 x half> %1805, i64 0, !dbg !54
  %address.local1112 = or disjoint i64 %address.local1055, 32, !dbg !55
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1112, !dbg !55
  store half %1809, ptr addrspace(1) %address.local1113, align 2, !dbg !55
  %address.local1115 = or disjoint i64 %address.local1055, 33, !dbg !55
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1115, !dbg !55
  store half %1808, ptr addrspace(1) %address.local1116, align 2, !dbg !55
  %address.local1118 = or disjoint i64 %address.local1055, 34, !dbg !55
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1118, !dbg !55
  store half %1807, ptr addrspace(1) %address.local1119, align 2, !dbg !55
  %address.local1121 = or disjoint i64 %address.local1055, 35, !dbg !55
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1121, !dbg !55
  store half %1806, ptr addrspace(1) %address.local1122, align 2, !dbg !55
  %address.local1124 = or disjoint i64 %address.local1055, 48, !dbg !55
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1124, !dbg !55
  store half %1804, ptr addrspace(1) %address.local1125, align 2, !dbg !55
  %address.local1127 = or disjoint i64 %address.local1055, 49, !dbg !55
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1127, !dbg !55
  store half %1803, ptr addrspace(1) %address.local1128, align 2, !dbg !55
  %address.local1130 = or disjoint i64 %address.local1055, 50, !dbg !55
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1130, !dbg !55
  store half %1802, ptr addrspace(1) %address.local1131, align 2, !dbg !55
  %address.local1133 = or i64 %address.local1044, 51, !dbg !55
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1133, !dbg !55
  store half %1801, ptr addrspace(1) %address.local1134, align 2, !dbg !55
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1112, !dbg !55
  store half %1799, ptr addrspace(1) %address.local1139, align 2, !dbg !55
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1115, !dbg !55
  store half %1798, ptr addrspace(1) %address.local1141, align 2, !dbg !55
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1118, !dbg !55
  store half %1797, ptr addrspace(1) %address.local1143, align 2, !dbg !55
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1121, !dbg !55
  store half %1796, ptr addrspace(1) %address.local1145, align 2, !dbg !55
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1124, !dbg !55
  store half %1794, ptr addrspace(1) %address.local1147, align 2, !dbg !55
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1127, !dbg !55
  store half %1793, ptr addrspace(1) %address.local1149, align 2, !dbg !55
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1130, !dbg !55
  store half %1792, ptr addrspace(1) %address.local1151, align 2, !dbg !55
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1133, !dbg !55
  store half %1791, ptr addrspace(1) %address.local1153, align 2, !dbg !55
  %normalized115 = fmul <4 x float> %row.reciprocal, %1498, !dbg !53
  %1810 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !54
  %1811 = extractelement <4 x half> %1810, i64 3, !dbg !54
  %1812 = extractelement <4 x half> %1810, i64 2, !dbg !54
  %1813 = extractelement <4 x half> %1810, i64 1, !dbg !54
  %1814 = extractelement <4 x half> %1810, i64 0, !dbg !54
  %normalized116 = fmul <4 x float> %row.reciprocal, %1465, !dbg !53
  %1815 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !54
  %1816 = extractelement <4 x half> %1815, i64 3, !dbg !54
  %1817 = extractelement <4 x half> %1815, i64 2, !dbg !54
  %1818 = extractelement <4 x half> %1815, i64 1, !dbg !54
  %1819 = extractelement <4 x half> %1815, i64 0, !dbg !54
  %normalized118 = fmul <4 x float> %row.reciprocal105, %1431, !dbg !53
  %1820 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !54
  %1821 = extractelement <4 x half> %1820, i64 3, !dbg !54
  %1822 = extractelement <4 x half> %1820, i64 2, !dbg !54
  %1823 = extractelement <4 x half> %1820, i64 1, !dbg !54
  %1824 = extractelement <4 x half> %1820, i64 0, !dbg !54
  %normalized119 = fmul <4 x float> %row.reciprocal105, %1398, !dbg !53
  %1825 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !54
  %1826 = extractelement <4 x half> %1825, i64 3, !dbg !54
  %1827 = extractelement <4 x half> %1825, i64 2, !dbg !54
  %1828 = extractelement <4 x half> %1825, i64 1, !dbg !54
  %1829 = extractelement <4 x half> %1825, i64 0, !dbg !54
  %address.local1169 = or disjoint i64 %address.local1055, 64, !dbg !55
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1169, !dbg !55
  store half %1829, ptr addrspace(1) %address.local1170, align 2, !dbg !55
  %address.local1172 = or disjoint i64 %address.local1055, 65, !dbg !55
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1172, !dbg !55
  store half %1828, ptr addrspace(1) %address.local1173, align 2, !dbg !55
  %address.local1175 = or disjoint i64 %address.local1055, 66, !dbg !55
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1175, !dbg !55
  store half %1827, ptr addrspace(1) %address.local1176, align 2, !dbg !55
  %address.local1178 = or disjoint i64 %address.local1055, 67, !dbg !55
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1178, !dbg !55
  store half %1826, ptr addrspace(1) %address.local1179, align 2, !dbg !55
  %address.local1181 = or disjoint i64 %address.local1055, 80, !dbg !55
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1181, !dbg !55
  store half %1824, ptr addrspace(1) %address.local1182, align 2, !dbg !55
  %address.local1184 = or disjoint i64 %address.local1055, 81, !dbg !55
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1184, !dbg !55
  store half %1823, ptr addrspace(1) %address.local1185, align 2, !dbg !55
  %address.local1187 = or disjoint i64 %address.local1055, 82, !dbg !55
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1187, !dbg !55
  store half %1822, ptr addrspace(1) %address.local1188, align 2, !dbg !55
  %address.local1190 = or disjoint i64 %address.local1055, 83, !dbg !55
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1190, !dbg !55
  store half %1821, ptr addrspace(1) %address.local1191, align 2, !dbg !55
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1169, !dbg !55
  store half %1819, ptr addrspace(1) %address.local1196, align 2, !dbg !55
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1172, !dbg !55
  store half %1818, ptr addrspace(1) %address.local1198, align 2, !dbg !55
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1175, !dbg !55
  store half %1817, ptr addrspace(1) %address.local1200, align 2, !dbg !55
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1178, !dbg !55
  store half %1816, ptr addrspace(1) %address.local1202, align 2, !dbg !55
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1181, !dbg !55
  store half %1814, ptr addrspace(1) %address.local1204, align 2, !dbg !55
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1184, !dbg !55
  store half %1813, ptr addrspace(1) %address.local1206, align 2, !dbg !55
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1187, !dbg !55
  store half %1812, ptr addrspace(1) %address.local1208, align 2, !dbg !55
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1190, !dbg !55
  store half %1811, ptr addrspace(1) %address.local1210, align 2, !dbg !55
  %normalized121 = fmul <4 x float> %row.reciprocal, %1364, !dbg !53
  %1830 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !54
  %1831 = extractelement <4 x half> %1830, i64 3, !dbg !54
  %1832 = extractelement <4 x half> %1830, i64 2, !dbg !54
  %1833 = extractelement <4 x half> %1830, i64 1, !dbg !54
  %1834 = extractelement <4 x half> %1830, i64 0, !dbg !54
  %normalized122 = fmul <4 x float> %row.reciprocal, %1331, !dbg !53
  %1835 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !54
  %1836 = extractelement <4 x half> %1835, i64 3, !dbg !54
  %1837 = extractelement <4 x half> %1835, i64 2, !dbg !54
  %1838 = extractelement <4 x half> %1835, i64 1, !dbg !54
  %1839 = extractelement <4 x half> %1835, i64 0, !dbg !54
  %normalized124 = fmul <4 x float> %row.reciprocal105, %1297, !dbg !53
  %1840 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !54
  %1841 = extractelement <4 x half> %1840, i64 3, !dbg !54
  %1842 = extractelement <4 x half> %1840, i64 2, !dbg !54
  %1843 = extractelement <4 x half> %1840, i64 1, !dbg !54
  %1844 = extractelement <4 x half> %1840, i64 0, !dbg !54
  %normalized125 = fmul <4 x float> %row.reciprocal105, %1264, !dbg !53
  %1845 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !54
  %1846 = extractelement <4 x half> %1845, i64 3, !dbg !54
  %1847 = extractelement <4 x half> %1845, i64 2, !dbg !54
  %1848 = extractelement <4 x half> %1845, i64 1, !dbg !54
  %1849 = extractelement <4 x half> %1845, i64 0, !dbg !54
  %address.local1226 = or disjoint i64 %address.local1055, 96, !dbg !55
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1226, !dbg !55
  store half %1849, ptr addrspace(1) %address.local1227, align 2, !dbg !55
  %address.local1229 = or disjoint i64 %address.local1055, 97, !dbg !55
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1229, !dbg !55
  store half %1848, ptr addrspace(1) %address.local1230, align 2, !dbg !55
  %address.local1232 = or disjoint i64 %address.local1055, 98, !dbg !55
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1232, !dbg !55
  store half %1847, ptr addrspace(1) %address.local1233, align 2, !dbg !55
  %address.local1235 = or disjoint i64 %address.local1055, 99, !dbg !55
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1235, !dbg !55
  store half %1846, ptr addrspace(1) %address.local1236, align 2, !dbg !55
  %address.local1238 = or disjoint i64 %address.local1055, 112, !dbg !55
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1238, !dbg !55
  store half %1844, ptr addrspace(1) %address.local1239, align 2, !dbg !55
  %address.local1241 = or disjoint i64 %address.local1055, 113, !dbg !55
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1241, !dbg !55
  store half %1843, ptr addrspace(1) %address.local1242, align 2, !dbg !55
  %address.local1244 = or disjoint i64 %address.local1055, 114, !dbg !55
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1244, !dbg !55
  store half %1842, ptr addrspace(1) %address.local1245, align 2, !dbg !55
  %address.local1247 = or i64 %address.local1044, 115, !dbg !55
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1247, !dbg !55
  store half %1841, ptr addrspace(1) %address.local1248, align 2, !dbg !55
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1226, !dbg !55
  store half %1839, ptr addrspace(1) %address.local1253, align 2, !dbg !55
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1229, !dbg !55
  store half %1838, ptr addrspace(1) %address.local1255, align 2, !dbg !55
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1232, !dbg !55
  store half %1837, ptr addrspace(1) %address.local1257, align 2, !dbg !55
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1235, !dbg !55
  store half %1836, ptr addrspace(1) %address.local1259, align 2, !dbg !55
  %address.local1261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1238, !dbg !55
  store half %1834, ptr addrspace(1) %address.local1261, align 2, !dbg !55
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1241, !dbg !55
  store half %1833, ptr addrspace(1) %address.local1263, align 2, !dbg !55
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1244, !dbg !55
  store half %1832, ptr addrspace(1) %address.local1265, align 2, !dbg !55
  %address.local1267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1247, !dbg !55
  store half %1831, ptr addrspace(1) %address.local1267, align 2, !dbg !55
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
!40 = !DILocation(line: 81, column: 26, scope: !3)
!41 = !DILocation(line: 88, column: 21, scope: !3)
!42 = !DILocation(line: 89, column: 20, scope: !3)
!43 = !DILocation(line: 90, column: 22, scope: !3)
!44 = !DILocation(line: 91, column: 23, scope: !3)
!45 = !DILocation(line: 92, column: 20, scope: !3)
!46 = !DILocation(line: 103, column: 22, scope: !3)
!47 = !DILocation(line: 93, column: 22, scope: !3)
!48 = !DILocation(line: 102, column: 23, scope: !3)
!49 = !{i64 8192}
!50 = !DILocation(line: 94, column: 22, scope: !3)
!51 = !DILocation(line: 96, column: 16, scope: !3)
!52 = !DILocation(line: 95, column: 20, scope: !3)
!53 = !DILocation(line: 107, column: 23, scope: !3)
!54 = !DILocation(line: 108, column: 19, scope: !3)
!55 = !DILocation(line: 109, column: 9, scope: !3)
