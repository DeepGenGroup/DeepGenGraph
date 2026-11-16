; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer/causal.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer/causal.input.mlir"

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
  %address.local346 = zext nneg i32 %47 to i64, !dbg !9
  %address.local347 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local346, !dbg !9
  %address.local350 = lshr i32 %address.thread, 4, !dbg !9
  %address.local351 = and i32 %address.local350, 8, !dbg !9
  %address.local352 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local353 = and i32 %address.local352, 16256, !dbg !9
  %address.local354 = or disjoint i32 %address.local351, %address.local353, !dbg !9
  %48 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1356 = shl i64 %43, 8, !dbg !9
  %49 = add i64 %.idx, %.idx1356, !dbg !9
  %50 = shl nuw nsw i32 %address.local354, 1, !dbg !9
  %51 = trunc i64 %49 to i32, !dbg !9
  %dma.buffer.offset1354 = or disjoint i32 %50, %51, !dbg !9
  %52 = insertelement <2 x i64> poison, i64 %48, i64 0, !dbg !9
  %53 = bitcast <2 x i64> %52 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1355 = shufflevector <4 x i32> %53, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local359 = or disjoint i64 %address.local346, 2048, !dbg !9
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local359, !dbg !9
  %address.local368 = or disjoint i32 %address.local354, 16, !dbg !9
  %54 = shl nuw nsw i32 %address.local368, 1, !dbg !9
  %dma.buffer.offset1352 = or disjoint i32 %54, %51, !dbg !9
  %55 = shl nuw nsw i64 %40, 19, !dbg !9
  %56 = add i64 %55, %44, !dbg !9
  %address.local371 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local346, !dbg !9
  %57 = trunc i64 %56 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local354, %57, !dbg !9
  %58 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1350 = or disjoint i32 %58, 64, !dbg !9
  %address.local375 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local359, !dbg !9
  %59 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local347, i64 0, !dbg !9
  %60 = insertelement <4 x ptr addrspace(3)> %59, ptr addrspace(3) %address.local360, i64 1, !dbg !9
  %61 = insertelement <4 x ptr addrspace(3)> %60, ptr addrspace(3) %address.local371, i64 2, !dbg !9
  %62 = insertelement <4 x ptr addrspace(3)> %61, ptr addrspace(3) %address.local375, i64 3, !dbg !9
  %63 = ptrtoint <4 x ptr addrspace(3)> %62 to <4 x i64>, !dbg !9
  %64 = trunc <4 x i64> %63 to <4 x i32>, !dbg !9
  %65 = extractelement <4 x i32> %64, i64 0, !dbg !9
  %66 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %65), !dbg !9
  %67 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1354, <4 x i32> %dma.buffer.rsrc1355) #9, !dbg !9
  %68 = extractelement <4 x i32> %64, i64 1, !dbg !9
  %69 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %68), !dbg !9
  %70 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset1352, <4 x i32> %dma.buffer.rsrc1355) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %71 = extractelement <4 x i32> %64, i64 2, !dbg !9
  %72 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %71), !dbg !9
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %dma.buffer.offset1350, <4 x i32> %dma.buffer.rsrc1355) #9, !dbg !9
  %74 = extractelement <4 x i32> %64, i64 3, !dbg !9
  %75 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %74), !dbg !9
  %.tr1360 = or disjoint i32 %address.local368, %57, !dbg !9
  %76 = shl i32 %.tr1360, 1, !dbg !9
  %dma.buffer.offset1348 = or disjoint i32 %76, 64, !dbg !9
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1355) #9, !dbg !9
  %address.local379 = lshr i32 %address.thread, 2, !dbg !11
  %address.local380 = and i32 %address.local379, 4, !dbg !11
  %address.local381 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local382 = and i32 %address.local381, 384, !dbg !11
  %address.local383 = or disjoint i32 %address.local380, %address.local382, !dbg !11
  %address.local385 = and i32 %46, 120, !dbg !11
  %address.local386 = or disjoint i32 %address.local383, %address.local385, !dbg !11
  %address.local387 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local388 = and i32 %address.local387, 1024, !dbg !11
  %address.local389 = or disjoint i32 %address.local386, %address.local388, !dbg !11
  %address.local390 = zext nneg i32 %address.local389 to i64, !dbg !11
  %address.local391 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local390, !dbg !11
  %78 = load <4 x half>, ptr addrspace(3) %address.local391, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local391, i64 4096, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %80 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1177 = getelementptr i8, ptr addrspace(3) %address.local391, i64 1024, !dbg !11
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1177, align 8, !dbg !11
  %82 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1178 = getelementptr i8, ptr addrspace(3) %address.local391, i64 5120, !dbg !11
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1178, align 8, !dbg !11
  %84 = fmul <4 x half> %81, splat (half 1.275630e-01), !dbg !11
  %85 = fmul <4 x half> %83, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1346 = or disjoint i32 %58, 128, !dbg !9
  %86 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1346, <4 x i32> %dma.buffer.rsrc1355) #9, !dbg !9
  %dma.buffer.offset1344 = or disjoint i32 %76, 128, !dbg !9
  %87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset1344, <4 x i32> %dma.buffer.rsrc1355) #9, !dbg !9
  %lds.const.gep1179 = getelementptr i8, ptr addrspace(3) %address.local391, i64 8192, !dbg !11
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1179, align 8, !dbg !11
  %lds.const.gep1180 = getelementptr i8, ptr addrspace(3) %address.local391, i64 12288, !dbg !11
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep1180, align 8, !dbg !11
  %lds.const.gep1181 = getelementptr i8, ptr addrspace(3) %address.local391, i64 9216, !dbg !11
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1181, align 8, !dbg !11
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1182 = getelementptr i8, ptr addrspace(3) %address.local391, i64 13312, !dbg !11
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1182, align 8, !dbg !11
  %93 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !11
  %94 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !11
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1342 = or disjoint i32 %58, 192, !dbg !9
  %96 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %dma.buffer.offset1342, <4 x i32> %dma.buffer.rsrc1355) #9, !dbg !9
  %dma.buffer.offset1340 = or disjoint i32 %76, 192, !dbg !9
  %97 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %dma.buffer.offset1340, <4 x i32> %dma.buffer.rsrc1355) #9, !dbg !9
  %98 = load <4 x half>, ptr addrspace(3) %address.local391, align 8, !dbg !11
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1177, align 8, !dbg !11
  %101 = fmul <4 x half> %98, splat (half 1.275630e-01), !dbg !11
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep1178, align 8, !dbg !11
  %103 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !11
  %104 = fmul <4 x half> %100, splat (half 1.275630e-01), !dbg !11
  %105 = fmul <4 x half> %102, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %106 = load <4 x half>, ptr addrspace(3) %lds.const.gep1179, align 8, !dbg !11
  %107 = load <4 x half>, ptr addrspace(3) %lds.const.gep1180, align 8, !dbg !11
  %108 = load <4 x half>, ptr addrspace(3) %lds.const.gep1181, align 8, !dbg !11
  %109 = fmul <4 x half> %106, splat (half 1.275630e-01), !dbg !11
  %110 = load <4 x half>, ptr addrspace(3) %lds.const.gep1182, align 8, !dbg !11
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
  %dma.buffer.rsrc1339 = shufflevector <4 x i32> %138, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %kdma.source = zext nneg i32 %kdma.owner to i64
  %kdma.source1437 = lshr i64 %kdma.source, 4
  %kdma.source1438 = shl nuw nsw i64 %kdma.source1437, 9
  %kdma.source1439 = shl nuw nsw i64 %kdma.source, 7
  %kdma.source1440 = add nuw nsw i64 %kdma.source1438, %kdma.source1439
  %kdma.source1441 = lshr i64 %kdma.source, 5
  %kdma.source1442 = lshr i64 %kdma.source, 2
  %kdma.source1443 = and i64 %kdma.source1442, 1
  %kdma.source1444 = and i64 %kdma.source1443, %kdma.source1437
  %kdma.source1445 = or disjoint i64 %kdma.source1444, %kdma.source1441
  %139 = shl nuw nsw i64 %kdma.source1445, 10
  %kdma.source1447 = sub nsw i64 %kdma.source1440, %139
  %kdma.source1448 = and i64 %kdma.source1447, 16256
  %kdma.source1449 = lshr i32 %kdma.tid, 4
  %kdma.source1450 = and i32 %kdma.source1449, 8
  %kdma.source1451 = zext nneg i32 %kdma.source1450 to i64
  %kdma.source1452 = or disjoint i64 %kdma.source1448, %kdma.source1451
  %kdma.source1456 = zext nneg i32 %kdma.owner.hi to i64
  %kdma.source1457 = lshr i64 %kdma.source1456, 4
  %kdma.source1458 = shl nuw nsw i64 %kdma.source1457, 9
  %kdma.source1459 = shl nuw nsw i64 %kdma.source1456, 7
  %kdma.source1460 = add nuw nsw i64 %kdma.source1458, %kdma.source1459
  %kdma.source1461 = lshr i64 %kdma.source1456, 5
  %kdma.source1462 = lshr i64 %kdma.source1456, 2
  %kdma.source1463 = and i64 %kdma.source1462, 1
  %kdma.source1464 = and i64 %kdma.source1463, %kdma.source1457
  %kdma.source1465 = add nuw nsw i64 %kdma.source1464, %kdma.source1461
  %140 = shl nuw nsw i64 %kdma.source1465, 10
  %kdma.source1467 = sub nsw i64 %kdma.source1460, %140
  %kdma.source1468 = and i64 %kdma.source1467, 16256
  %kdma.source1472 = or disjoint i64 %kdma.source1468, %kdma.source1451
  %kdma.lds1477 = add i32 %66, 512
  %kdma.source1479 = or disjoint i64 %kdma.source, 256
  %kdma.source1480 = lshr i64 %kdma.source1479, 4
  %kdma.source1481 = shl nuw nsw i64 %kdma.source1480, 9
  %kdma.source1483 = add nuw nsw i64 %kdma.source1481, %kdma.source1439
  %kdma.source1484 = lshr i64 %kdma.source1479, 5
  %kdma.source1487 = and i64 %kdma.source1480, %kdma.source1443
  %kdma.source1488 = or disjoint i64 %kdma.source1487, %kdma.source1484
  %141 = shl nuw nsw i64 %kdma.source1488, 10
  %kdma.source1490 = sub nsw i64 %kdma.source1483, %141
  %kdma.source1491 = and i64 %kdma.source1490, 16256
  %kdma.source1494 = or disjoint i32 %kdma.source1450, 16
  %kdma.source1495 = zext nneg i32 %kdma.source1494 to i64
  %kdma.source1496 = or disjoint i64 %kdma.source1491, %kdma.source1495
  %kdma.source1503 = or disjoint i64 %kdma.source1456, 256
  %kdma.source1504 = lshr i64 %kdma.source1503, 4
  %kdma.source1505 = shl nuw nsw i64 %kdma.source1504, 9
  %kdma.source1507 = add nuw nsw i64 %kdma.source1505, %kdma.source1459
  %kdma.source1508 = lshr i64 %kdma.source1503, 5
  %kdma.source1511 = and i64 %kdma.source1504, %kdma.source1463
  %kdma.source1512 = add nuw nsw i64 %kdma.source1511, %kdma.source1508
  %142 = shl nuw nsw i64 %kdma.source1512, 10
  %kdma.source1514 = sub nsw i64 %kdma.source1507, %142
  %kdma.source1515 = and i64 %kdma.source1514, 16256
  %kdma.source1520 = or disjoint i64 %kdma.source1515, %kdma.source1495
  %kdma.lds1525 = add i32 %69, 512
  %143 = or disjoint i64 %134, %133
  %kdma.source1546 = or disjoint i32 %kdma.intra, 64
  %kdma.lds1571 = add i32 %72, 512
  %kdma.lds1621 = add i32 %75, 512
  %kdma.source1642 = or disjoint i32 %kdma.intra, 128
  %kdma.source1738 = or disjoint i32 %kdma.intra, 192
  %144 = ptrtoint ptr addrspace(1) %2 to i64
  %145 = insertelement <2 x i64> poison, i64 %144, i64 0
  %146 = bitcast <2 x i64> %145 to <4 x i32>
  %dma.buffer.rsrc1323 = shufflevector <4 x i32> %146, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %147 = insertelement <2 x i64> poison, i64 %143, i64 1, !dbg !14
  %148 = insertelement <2 x i64> poison, i64 %117, i64 0, !dbg !14
  %149 = insertelement <32 x i32> poison, i32 %132, i64 0, !dbg !16
  %150 = shufflevector <32 x i32> %149, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !16
  %151 = insertelement <32 x i32> poison, i32 %131, i64 0, !dbg !16
  %invariant.op = sub i32 72, %kdma.intra, !dbg !12
  %invariant.op60 = sub i32 72, %kdma.intra, !dbg !12
  %invariant.op61 = sub i32 136, %kdma.intra, !dbg !12
  %invariant.op62 = sub i32 136, %kdma.intra, !dbg !12
  %invariant.op63 = sub i32 200, %kdma.intra, !dbg !12
  %invariant.op64 = sub i32 200, %kdma.intra, !dbg !12
  %152 = insertelement <32 x i32> poison, i32 %131, i64 8
  br label %153, !dbg !12

153:                                              ; preds = %.lr.ph, %153
  %154 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1411, %153 ]
  %155 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1407, %153 ]
  %156 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1403, %153 ]
  %157 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1399, %153 ]
  %158 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %1159, %153 ]
  %159 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1397, %153 ]
  %160 = phi i64 [ 0, %.lr.ph ], [ %1412, %153 ]
  %161 = lshr i64 %160, 12, !dbg !14
  %162 = add nuw nsw i64 %161, %29, !dbg !14
  %163 = shl nuw nsw i64 %160, 7, !dbg !14
  %164 = shl i64 %162, 19, !dbg !14
  %165 = and i64 %164, 16252928, !dbg !14
  %address.thread442 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local443 = zext nneg i32 %address.thread442 to i64, !dbg !9
  %166 = insertelement <2 x i64> %148, i64 %160, i64 1, !dbg !14
  %address.local486 = shl nuw nsw i64 %address.local443, 1, !dbg !15
  %address.local487 = and i64 %address.local486, 24, !dbg !15
  %address.local489 = shl i64 %address.local443, 62, !dbg !15
  %address.local490 = ashr exact i64 %address.local489, 57, !dbg !15
  %address.local491 = and i64 %address.local490, -64, !dbg !15
  %address.local496 = and i64 %address.local443, 32, !dbg !11
  %address.local497 = icmp eq i64 %address.local496, 0, !dbg !11
  %address.local498 = select i1 %address.local497, i64 0, i64 1016, !dbg !11
  %.idx1391 = mul nuw nsw i64 %address.local443, 320, !dbg !15
  %167 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1391, !dbg !15
  %.idx1940 = shl nuw nsw i64 %address.local487, 1, !dbg !15
  %address.local511 = shl nuw nsw i32 %address.thread442, 1, !dbg !15
  %address.local512 = and i32 %address.local511, 24, !dbg !15
  %address.local513 = shl nuw nsw i32 %address.thread442, 8, !dbg !15
  %address.local514 = and i32 %address.local513, 768, !dbg !15
  %address.local515 = or disjoint i32 %address.local512, %address.local514, !dbg !15
  %address.local516 = shl nuw nsw i32 %address.thread442, 5, !dbg !15
  %address.local517 = and i32 %address.local516, 96, !dbg !15
  %address.local518 = or disjoint i32 %address.local515, %address.local517, !dbg !15
  %address.local519 = zext nneg i32 %address.local518 to i64, !dbg !15
  %address.local520 = lshr i32 %address.thread442, 1, !dbg !15
  %address.local521 = and i32 %address.local520, 1, !dbg !15
  %address.local522 = zext nneg i32 %address.local521 to i64, !dbg !15
  %168 = insertelement <2 x i64> poison, i64 %address.local443, i64 0, !dbg !14
  %169 = insertelement <2 x i64> %168, i64 %160, i64 1, !dbg !14
  %170 = lshr <2 x i64> %169, <i64 2, i64 5>, !dbg !14
  %171 = extractelement <2 x i64> %170, i64 1, !dbg !14
  %172 = and i64 %171, 127, !dbg !14
  %.idx13691372 = or disjoint i64 %165, %172, !dbg !14
  %173 = shl nuw nsw <2 x i64> %169, <i64 3, i64 14>, !dbg !9
  %174 = shl <2 x i64> %170, <i64 5, i64 19>, !dbg !17
  %175 = sub <2 x i64> %173, %174, !dbg !17
  %176 = extractelement <2 x i64> %175, i64 1, !dbg !14
  %177 = add i64 %.idx13691372, %176, !dbg !14
  %kdma.source1453 = add i64 %177, %kdma.source1452, !dbg !14
  %kdma.source1454 = trunc i64 %kdma.source1453 to i32, !dbg !14
  %kdma.source1455 = shl i32 %kdma.source1454, 1, !dbg !14
  %kdma.offset = add i32 %kdma.source1455, %kdma.intra, !dbg !14
  %178 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1473 = add i64 %177, %kdma.source1472, !dbg !14
  %kdma.source1474 = trunc i64 %kdma.source1473 to i32, !dbg !14
  %kdma.source1475 = shl i32 %kdma.source1474, 1, !dbg !14
  %kdma.offset1476 = add i32 %kdma.source1475, %kdma.intra.hi, !dbg !14
  %179 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1477, i32 %kdma.offset1476, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1497 = add i64 %177, %kdma.source1496, !dbg !14
  %kdma.source1498 = trunc i64 %kdma.source1497 to i32, !dbg !14
  %kdma.source1499 = shl i32 %kdma.source1498, 1, !dbg !14
  %kdma.offset1500 = add i32 %kdma.source1499, %kdma.intra, !dbg !14
  %180 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %kdma.offset1500, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1521 = add i64 %177, %kdma.source1520, !dbg !14
  %kdma.source1522 = trunc i64 %kdma.source1521 to i32, !dbg !14
  %kdma.source1523 = shl i32 %kdma.source1522, 1, !dbg !14
  %kdma.offset1524 = add i32 %kdma.source1523, %kdma.intra.hi, !dbg !14
  %181 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1525, i32 %kdma.offset1524, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx13791383 = add i64 %176, %171, !dbg !14
  %182 = add i64 %.idx13791383, %117, !dbg !14
  %183 = shufflevector <2 x i64> %147, <2 x i64> %170, <2 x i32> <i32 3, i32 1>, !dbg !14
  %184 = add nsw <2 x i64> %183, %166, !dbg !14
  %185 = extractelement <2 x i64> %184, i64 1, !dbg !14
  %186 = extractelement <2 x i64> %184, i64 0, !dbg !14
  %187 = add nuw nsw i64 %185, 1024, !dbg !14
  %188 = add nsw <2 x i64> %184, <i64 32, i64 1024>, !dbg !14
  %.lobit41 = ashr i64 %187, 63, !dbg !14
  %189 = icmp slt <2 x i64> %184, <i64 -32, i64 -1024>, !dbg !14
  %190 = icmp slt i64 %186, -32, !dbg !14
  %191 = sub nsw <2 x i64> <i64 -33, i64 -1025>, %184, !dbg !14
  %192 = select <2 x i1> %189, <2 x i64> %191, <2 x i64> %188, !dbg !14
  %193 = sdiv <2 x i64> %192, <i64 128, i64 4096>, !dbg !14
  %194 = sext i1 %190 to i64, !dbg !14
  %195 = insertelement <2 x i64> poison, i64 %194, i64 0, !dbg !14
  %196 = insertelement <2 x i64> %195, i64 %.lobit41, i64 1, !dbg !14
  %197 = xor <2 x i64> %193, %196, !dbg !14
  %198 = shl <2 x i64> %197, <i64 7, i64 19>, !dbg !14
  %199 = extractelement <2 x i64> %198, i64 0, !dbg !14
  %200 = sub i64 %182, %199, !dbg !14
  %201 = extractelement <2 x i64> %198, i64 1, !dbg !14
  %202 = and i64 %201, 16252928, !dbg !14
  %203 = add i64 %200, %202, !dbg !14
  %kdma.source1543 = add i64 %203, %kdma.source1452, !dbg !14
  %kdma.source1544 = trunc i64 %kdma.source1543 to i32, !dbg !14
  %kdma.source1545 = shl i32 %kdma.source1544, 1, !dbg !14
  %kdma.offset1547 = add i32 %kdma.source1546, %kdma.source1545, !dbg !14
  %204 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %kdma.offset1547, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1566 = add i64 %203, %kdma.source1472, !dbg !14
  %kdma.source1567 = trunc i64 %kdma.source1566 to i32, !dbg !14
  %kdma.source1568 = shl i32 %kdma.source1567, 1, !dbg !14
  %kdma.offset1570.reass.reass = add i32 %kdma.source1568, %invariant.op
  %205 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1571, i32 %kdma.offset1570.reass.reass, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1591 = add i64 %203, %kdma.source1496, !dbg !14
  %kdma.source1592 = trunc i64 %kdma.source1591 to i32, !dbg !14
  %kdma.source1593 = shl i32 %kdma.source1592, 1, !dbg !14
  %kdma.offset1595 = add i32 %kdma.source1546, %kdma.source1593, !dbg !14
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %kdma.offset1595, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1616 = add i64 %203, %kdma.source1520, !dbg !14
  %kdma.source1617 = trunc i64 %kdma.source1616 to i32, !dbg !14
  %kdma.source1618 = shl i32 %kdma.source1617, 1, !dbg !14
  %kdma.offset1620.reass.reass = add i32 %kdma.source1618, %invariant.op60
  %207 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1621, i32 %kdma.offset1620.reass.reass, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %208 = extractelement <2 x i64> %170, i64 0, !dbg !15
  %address.local494 = and i64 %208, 12, !dbg !11
  %.idx1392 = mul nsw i64 %208, -1280, !dbg !15
  %209 = getelementptr i8, ptr addrspace(3) %167, i64 %.idx1392, !dbg !15
  %210 = getelementptr [2 x i8], ptr addrspace(3) %209, i64 %address.local487, !dbg !15
  %211 = getelementptr [2 x i8], ptr addrspace(3) %210, i64 %address.local491, !dbg !15
  %212 = getelementptr [2 x i8], ptr addrspace(3) %211, i64 %address.local494, !dbg !15
  %address.local500 = getelementptr [2 x i8], ptr addrspace(3) %212, i64 %address.local498, !dbg !15
  %213 = add nsw i64 %.idx1392, %.idx1391, !dbg !15
  %214 = or disjoint i64 %213, %.idx1940, !dbg !15
  %215 = or disjoint i64 %address.local491, %address.local494, !dbg !15
  %216 = add nsw i64 %215, %address.local498, !dbg !15
  %217 = shl nsw i64 %216, 1, !dbg !15
  %218 = add nsw i64 %214, %217, !dbg !15
  %219 = lshr i64 %218, 6, !dbg !15
  %220 = and i64 %219, 8, !dbg !15
  %221 = xor i64 %220, %218, !dbg !15
  %222 = sub i64 %221, %218, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local500, i64 %222, !dbg !15
  %223 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %123) #8, !dbg !15, !frisk.register_lifetimes !5
  %address.local523 = add nsw i64 %address.base, %address.local522, !dbg !15
  %address.local524 = shl i64 %address.local523, 6, !dbg !15
  %address.local525 = sub i64 %address.local519, %address.local524, !dbg !15
  %226 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local525, !dbg !15
  %227 = getelementptr [2 x i8], ptr addrspace(3) %226, i64 %address.local494, !dbg !15
  %228 = getelementptr [2 x i8], ptr addrspace(3) %227, i64 %address.local498, !dbg !15
  %address.local529 = getelementptr i8, ptr addrspace(3) %228, i64 64, !dbg !15
  %.idx19431946 = add nuw nsw i64 %address.local498, %address.local494, !dbg !15
  %229 = add i64 %.idx19431946, %address.local525, !dbg !15
  %230 = shl i64 %229, 1, !dbg !15
  %231 = add i64 %230, 64, !dbg !15
  %232 = lshr i64 %231, 6, !dbg !15
  %233 = and i64 %232, 8, !dbg !15
  %234 = xor i64 %233, %231, !dbg !15
  %235 = sub i64 %234, %231, !dbg !15
  %kdma.swizzled1814 = getelementptr i8, ptr addrspace(3) %address.local529, i64 %235, !dbg !15
  %236 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1814, align 8, !dbg !15, !frisk.k_lds_pair !18
  %kdma.gep1818 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 128, !dbg !15
  %237 = load <4 x half>, ptr addrspace(3) %kdma.gep1818, align 8, !dbg !15, !frisk.k_lds_pair !18
  %kdma.gep1815 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 4096, !dbg !15
  %238 = load <4 x half>, ptr addrspace(3) %kdma.gep1815, align 8, !dbg !15, !frisk.k_lds_pair !19
  %kdma.gep1819 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 4224, !dbg !15
  %239 = load <4 x half>, ptr addrspace(3) %kdma.gep1819, align 8, !dbg !15, !frisk.k_lds_pair !19
  %kdma.gep1816 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %240 = load <4 x half>, ptr addrspace(3) %kdma.gep1816, align 8, !dbg !15, !frisk.k_lds_pair !20
  %kdma.gep1820 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %241 = load <4 x half>, ptr addrspace(3) %kdma.gep1820, align 8, !dbg !15, !frisk.k_lds_pair !20
  %kdma.gep1817 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %242 = load <4 x half>, ptr addrspace(3) %kdma.gep1817, align 8, !dbg !15, !frisk.k_lds_pair !21
  %kdma.gep1821 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %243 = load <4 x half>, ptr addrspace(3) %kdma.gep1821, align 8, !dbg !15, !frisk.k_lds_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %225, <4 x float> %224) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1822 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 1024, !dbg !15
  %245 = load <4 x half>, ptr addrspace(3) %kdma.gep1822, align 8, !dbg !15, !frisk.k_lds_pair !22
  %kdma.gep1826 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 1152, !dbg !15
  %246 = load <4 x half>, ptr addrspace(3) %kdma.gep1826, align 8, !dbg !15, !frisk.k_lds_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1823 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 5120, !dbg !15
  %248 = load <4 x half>, ptr addrspace(3) %kdma.gep1823, align 8, !dbg !15, !frisk.k_lds_pair !23
  %kdma.gep1827 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 5248, !dbg !15
  %249 = load <4 x half>, ptr addrspace(3) %kdma.gep1827, align 8, !dbg !15, !frisk.k_lds_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %238, <4 x float> %247) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1824 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %251 = load <4 x half>, ptr addrspace(3) %kdma.gep1824, align 8, !dbg !15, !frisk.k_lds_pair !24
  %kdma.gep1825 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %252 = load <4 x half>, ptr addrspace(3) %kdma.gep1825, align 8, !dbg !15, !frisk.k_lds_pair !24
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
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !15
  %266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %225, <4 x float> %265) #8, !dbg !15
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
  %281 = add nuw nsw i64 %185, 2048, !dbg !14
  %282 = add nsw <2 x i64> %184, <i64 64, i64 2048>, !dbg !14
  %.lobit42 = ashr i64 %281, 63, !dbg !14
  %283 = icmp slt <2 x i64> %184, <i64 -64, i64 -2048>, !dbg !14
  %284 = icmp slt i64 %186, -64, !dbg !14
  %285 = sub nsw <2 x i64> <i64 -65, i64 -2049>, %184, !dbg !14
  %286 = select <2 x i1> %283, <2 x i64> %285, <2 x i64> %282, !dbg !14
  %287 = sdiv <2 x i64> %286, <i64 128, i64 4096>, !dbg !14
  %288 = sext i1 %284 to i64, !dbg !14
  %289 = insertelement <2 x i64> poison, i64 %288, i64 0, !dbg !14
  %290 = insertelement <2 x i64> %289, i64 %.lobit42, i64 1, !dbg !14
  %291 = xor <2 x i64> %287, %290, !dbg !14
  %292 = shl <2 x i64> %291, <i64 7, i64 19>, !dbg !14
  %293 = extractelement <2 x i64> %292, i64 0, !dbg !14
  %294 = sub i64 %182, %293, !dbg !14
  %295 = extractelement <2 x i64> %292, i64 1, !dbg !14
  %296 = and i64 %295, 16252928, !dbg !14
  %297 = add i64 %294, %296, !dbg !14
  %kdma.source1639 = add i64 %297, %kdma.source1452, !dbg !14
  %kdma.source1640 = trunc i64 %kdma.source1639 to i32, !dbg !14
  %kdma.source1641 = shl i32 %kdma.source1640, 1, !dbg !14
  %kdma.offset1643 = add i32 %kdma.source1642, %kdma.source1641, !dbg !14
  %298 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset1643, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1662 = add i64 %297, %kdma.source1472, !dbg !14
  %kdma.source1663 = trunc i64 %kdma.source1662 to i32, !dbg !14
  %kdma.source1664 = shl i32 %kdma.source1663, 1, !dbg !14
  %kdma.offset1666.reass.reass = add i32 %kdma.source1664, %invariant.op61
  %299 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1477, i32 %kdma.offset1666.reass.reass, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1687 = add i64 %297, %kdma.source1496, !dbg !14
  %kdma.source1688 = trunc i64 %kdma.source1687 to i32, !dbg !14
  %kdma.source1689 = shl i32 %kdma.source1688, 1, !dbg !14
  %kdma.offset1691 = add i32 %kdma.source1642, %kdma.source1689, !dbg !14
  %300 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %kdma.offset1691, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1712 = add i64 %297, %kdma.source1520, !dbg !14
  %kdma.source1713 = trunc i64 %kdma.source1712 to i32, !dbg !14
  %kdma.source1714 = shl i32 %kdma.source1713, 1, !dbg !14
  %kdma.offset1716.reass.reass = add i32 %kdma.source1714, %invariant.op62
  %301 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1525, i32 %kdma.offset1716.reass.reass, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.gep1844 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !15, !frisk.k_lds_pair !25
  %kdma.gep1848 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !15, !frisk.k_lds_pair !25
  %kdma.gep1845 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !15, !frisk.k_lds_pair !26
  %kdma.gep1849 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !15, !frisk.k_lds_pair !26
  %kdma.gep1846 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 8192, !dbg !15
  %306 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !15, !frisk.k_lds_pair !27
  %kdma.gep1850 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 8320, !dbg !15
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !15, !frisk.k_lds_pair !27
  %kdma.gep1847 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 12288, !dbg !15
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !15, !frisk.k_lds_pair !28
  %kdma.gep1851 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 12416, !dbg !15
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !15, !frisk.k_lds_pair !28
  %kdma.gep1852 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !15, !frisk.k_lds_pair !29
  %kdma.gep1856 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !15, !frisk.k_lds_pair !29
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %302, <4 x float> %244) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1853 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !15, !frisk.k_lds_pair !30
  %kdma.gep1857 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !15, !frisk.k_lds_pair !30
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %304, <4 x float> %312) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1854 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 9216, !dbg !15
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !15, !frisk.k_lds_pair !31
  %kdma.gep1858 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 9344, !dbg !15
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep1858, align 8, !dbg !15, !frisk.k_lds_pair !31
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %306, <4 x float> %250) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1855 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 13312, !dbg !15
  %319 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !15, !frisk.k_lds_pair !32
  %kdma.gep1859 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 13440, !dbg !15
  %320 = load <4 x half>, ptr addrspace(3) %kdma.gep1859, align 8, !dbg !15, !frisk.k_lds_pair !32
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
  %350 = add nuw nsw i64 %185, 3072, !dbg !14
  %351 = add nsw <2 x i64> %184, <i64 96, i64 3072>, !dbg !14
  %.lobit43 = ashr i64 %350, 63, !dbg !14
  %352 = icmp slt <2 x i64> %184, <i64 -96, i64 -3072>, !dbg !14
  %353 = icmp slt i64 %186, -96, !dbg !14
  %354 = sub nsw <2 x i64> <i64 -97, i64 -3073>, %184, !dbg !14
  %355 = select <2 x i1> %352, <2 x i64> %354, <2 x i64> %351, !dbg !14
  %356 = sdiv <2 x i64> %355, <i64 128, i64 4096>, !dbg !14
  %357 = sext i1 %353 to i64, !dbg !14
  %358 = insertelement <2 x i64> poison, i64 %357, i64 0, !dbg !14
  %359 = insertelement <2 x i64> %358, i64 %.lobit43, i64 1, !dbg !14
  %360 = xor <2 x i64> %356, %359, !dbg !14
  %361 = shl <2 x i64> %360, <i64 7, i64 19>, !dbg !14
  %362 = extractelement <2 x i64> %361, i64 0, !dbg !14
  %363 = sub i64 %182, %362, !dbg !14
  %364 = extractelement <2 x i64> %361, i64 1, !dbg !14
  %365 = and i64 %364, 16252928, !dbg !14
  %366 = add i64 %363, %365, !dbg !14
  %kdma.source1735 = add i64 %366, %kdma.source1452, !dbg !14
  %kdma.source1736 = trunc i64 %kdma.source1735 to i32, !dbg !14
  %kdma.source1737 = shl i32 %kdma.source1736, 1, !dbg !14
  %kdma.offset1739 = add i32 %kdma.source1738, %kdma.source1737, !dbg !14
  %367 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %kdma.offset1739, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1758 = add i64 %366, %kdma.source1472, !dbg !14
  %kdma.source1759 = trunc i64 %kdma.source1758 to i32, !dbg !14
  %kdma.source1760 = shl i32 %kdma.source1759, 1, !dbg !14
  %kdma.offset1762.reass.reass = add i32 %kdma.source1760, %invariant.op63
  %368 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1571, i32 %kdma.offset1762.reass.reass, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1783 = add i64 %366, %kdma.source1496, !dbg !14
  %kdma.source1784 = trunc i64 %kdma.source1783 to i32, !dbg !14
  %kdma.source1785 = shl i32 %kdma.source1784, 1, !dbg !14
  %kdma.offset1787 = add i32 %kdma.source1738, %kdma.source1785, !dbg !14
  %369 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %kdma.offset1787, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %kdma.source1808 = add i64 %366, %kdma.source1520, !dbg !14
  %kdma.source1809 = trunc i64 %kdma.source1808 to i32, !dbg !14
  %kdma.source1810 = shl i32 %kdma.source1809, 1, !dbg !14
  %kdma.offset1812.reass.reass = add i32 %kdma.source1810, %invariant.op64
  %370 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1621, i32 %kdma.offset1812.reass.reass, <4 x i32> %dma.buffer.rsrc1339) #9, !dbg !14
  %371 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !33
  %372 = load <4 x half>, ptr addrspace(3) %kdma.gep1816, align 8, !dbg !15, !frisk.k_lds_pair !33
  %373 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !34
  %374 = load <4 x half>, ptr addrspace(3) %kdma.gep1817, align 8, !dbg !15, !frisk.k_lds_pair !34
  %375 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1814, align 8, !dbg !15, !frisk.k_lds_pair !35
  %376 = load <4 x half>, ptr addrspace(3) %kdma.gep1818, align 8, !dbg !15, !frisk.k_lds_pair !35
  %377 = load <4 x half>, ptr addrspace(3) %kdma.gep1815, align 8, !dbg !15, !frisk.k_lds_pair !36
  %378 = load <4 x half>, ptr addrspace(3) %kdma.gep1819, align 8, !dbg !15, !frisk.k_lds_pair !36
  %379 = load <4 x half>, ptr addrspace(3) %kdma.gep1820, align 8, !dbg !15, !frisk.k_lds_pair !37
  %380 = load <4 x half>, ptr addrspace(3) %kdma.gep1824, align 8, !dbg !15, !frisk.k_lds_pair !37
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %371, <4 x float> %315) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %382 = load <4 x half>, ptr addrspace(3) %kdma.gep1821, align 8, !dbg !15, !frisk.k_lds_pair !38
  %383 = load <4 x half>, ptr addrspace(3) %kdma.gep1825, align 8, !dbg !15, !frisk.k_lds_pair !38
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %373, <4 x float> %381) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %385 = load <4 x half>, ptr addrspace(3) %kdma.gep1822, align 8, !dbg !15, !frisk.k_lds_pair !39
  %386 = load <4 x half>, ptr addrspace(3) %kdma.gep1826, align 8, !dbg !15, !frisk.k_lds_pair !39
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %375, <4 x float> %321) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %388 = load <4 x half>, ptr addrspace(3) %kdma.gep1823, align 8, !dbg !15, !frisk.k_lds_pair !40
  %389 = load <4 x half>, ptr addrspace(3) %kdma.gep1827, align 8, !dbg !15, !frisk.k_lds_pair !40
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
  %419 = add nuw nsw i64 %171, %29, !dbg !17
  %420 = insertelement <2 x i64> %169, i64 %419, i64 1, !dbg !17
  %421 = shl <2 x i64> %420, <i64 5, i64 19>, !dbg !17
  %422 = and <2 x i64> %421, <i64 8064, i64 16252928>, !dbg !17
  %423 = add <2 x i64> %175, %422, !dbg !17
  %424 = extractelement <2 x i64> %423, i64 0, !dbg !17
  %425 = extractelement <2 x i64> %423, i64 1, !dbg !17
  %426 = add i64 %424, %425, !dbg !17
  %.tr1420 = trunc i64 %426 to i32, !dbg !17
  %dma.buffer.offset1322 = shl i32 %.tr1420, 1, !dbg !17
  %427 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1322, <4 x i32> %dma.buffer.rsrc1323) #9, !dbg !17
  %address.local865 = shl nuw nsw i32 %address.thread442, 3, !dbg !17
  %address.local866 = and i32 %address.local865, 24, !dbg !17
  %address.local868 = and i32 %address.local516, 8064, !dbg !17
  %428 = trunc i64 %425 to i32, !dbg !17
  %address.local869 = or disjoint i32 %address.local868, %428, !dbg !17
  %address.local870 = or disjoint i32 %address.local869, %address.local866, !dbg !17
  %.tr1424 = shl i32 %address.local870, 1, !dbg !17
  %dma.buffer.offset1320 = or i32 %.tr1424, 16384, !dbg !17
  %429 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset1320, <4 x i32> %dma.buffer.rsrc1323) #9, !dbg !17
  %430 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !15, !frisk.k_tail_pair !18
  %431 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !15, !frisk.k_tail_pair !18
  %432 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !15, !frisk.k_tail_pair !19
  %433 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !15, !frisk.k_tail_pair !19
  %434 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !15, !frisk.k_tail_pair !20
  %435 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !15, !frisk.k_tail_pair !20
  %436 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !15, !frisk.k_tail_pair !21
  %437 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !15, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %430, <4 x float> %384) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %439 = or disjoint i64 %163, %124, !dbg !16
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %432, <4 x float> %438) #8, !dbg !15
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %434, <4 x float> %390) #8, !dbg !15
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %436, <4 x float> %441) #8, !dbg !15
  %443 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !15, !frisk.k_tail_pair !22
  %444 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !15, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %431, <4 x float> %392) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %446 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !15, !frisk.k_tail_pair !23
  %447 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !15, !frisk.k_tail_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %433, <4 x float> %445) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %449 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !15, !frisk.k_tail_pair !24
  %450 = load <4 x half>, ptr addrspace(3) %kdma.gep1858, align 8, !dbg !15, !frisk.k_tail_pair !24
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %435, <4 x float> %394) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %452 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !15, !frisk.k_tail_pair !25
  %453 = load <4 x half>, ptr addrspace(3) %kdma.gep1859, align 8, !dbg !15, !frisk.k_tail_pair !25
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
  %561 = tail call float @llvm.maxnum.f32(float %560, float -inf), !dbg !7
  %562 = extractelement <32 x float> %559, i64 1, !dbg !7
  %563 = tail call float @llvm.maxnum.f32(float %561, float %562), !dbg !7
  %564 = extractelement <32 x float> %559, i64 8, !dbg !7
  %565 = extractelement <32 x float> %559, i64 9, !dbg !7
  %566 = tail call float @llvm.maxnum.f32(float %564, float %565), !dbg !7
  %567 = extractelement <32 x float> %559, i64 16, !dbg !7
  %568 = extractelement <32 x float> %559, i64 17, !dbg !7
  %569 = tail call float @llvm.maxnum.f32(float %567, float %568), !dbg !7
  %570 = extractelement <32 x float> %559, i64 24, !dbg !7
  %571 = extractelement <32 x float> %559, i64 25, !dbg !7
  %572 = tail call float @llvm.maxnum.f32(float %570, float %571), !dbg !7
  %573 = extractelement <32 x float> %559, i64 2, !dbg !7
  %574 = tail call float @llvm.maxnum.f32(float %563, float %573), !dbg !7
  %575 = extractelement <32 x float> %559, i64 10, !dbg !7
  %576 = tail call float @llvm.maxnum.f32(float %566, float %575), !dbg !7
  %577 = extractelement <32 x float> %559, i64 18, !dbg !7
  %578 = tail call float @llvm.maxnum.f32(float %569, float %577), !dbg !7
  %579 = extractelement <32 x float> %559, i64 26, !dbg !7
  %580 = tail call float @llvm.maxnum.f32(float %572, float %579), !dbg !7
  %581 = extractelement <32 x float> %559, i64 3, !dbg !7
  %582 = tail call float @llvm.maxnum.f32(float %574, float %581), !dbg !7
  %583 = extractelement <32 x float> %559, i64 11, !dbg !7
  %584 = tail call float @llvm.maxnum.f32(float %576, float %583), !dbg !7
  %585 = extractelement <32 x float> %559, i64 19, !dbg !7
  %586 = tail call float @llvm.maxnum.f32(float %578, float %585), !dbg !7
  %587 = extractelement <32 x float> %559, i64 27, !dbg !7
  %588 = tail call float @llvm.maxnum.f32(float %580, float %587), !dbg !7
  %589 = extractelement <32 x float> %559, i64 4, !dbg !7
  %590 = tail call float @llvm.maxnum.f32(float %582, float %589), !dbg !7
  %591 = extractelement <32 x float> %559, i64 12, !dbg !7
  %592 = tail call float @llvm.maxnum.f32(float %584, float %591), !dbg !7
  %593 = extractelement <32 x float> %559, i64 20, !dbg !7
  %594 = tail call float @llvm.maxnum.f32(float %586, float %593), !dbg !7
  %595 = extractelement <32 x float> %559, i64 28, !dbg !7
  %596 = tail call float @llvm.maxnum.f32(float %588, float %595), !dbg !7
  %597 = extractelement <32 x float> %559, i64 5, !dbg !7
  %598 = tail call float @llvm.maxnum.f32(float %590, float %597), !dbg !7
  %599 = extractelement <32 x float> %559, i64 13, !dbg !7
  %600 = tail call float @llvm.maxnum.f32(float %592, float %599), !dbg !7
  %601 = extractelement <32 x float> %559, i64 21, !dbg !7
  %602 = tail call float @llvm.maxnum.f32(float %594, float %601), !dbg !7
  %603 = extractelement <32 x float> %559, i64 29, !dbg !7
  %604 = tail call float @llvm.maxnum.f32(float %596, float %603), !dbg !7
  %605 = extractelement <32 x float> %559, i64 6, !dbg !7
  %606 = tail call float @llvm.maxnum.f32(float %598, float %605), !dbg !7
  %607 = extractelement <32 x float> %559, i64 14, !dbg !7
  %608 = tail call float @llvm.maxnum.f32(float %600, float %607), !dbg !7
  %609 = extractelement <32 x float> %559, i64 22, !dbg !7
  %610 = tail call float @llvm.maxnum.f32(float %602, float %609), !dbg !7
  %611 = extractelement <32 x float> %559, i64 30, !dbg !7
  %612 = tail call float @llvm.maxnum.f32(float %604, float %611), !dbg !7
  %613 = extractelement <32 x float> %559, i64 7, !dbg !7
  %614 = tail call float @llvm.maxnum.f32(float %606, float %613), !dbg !7
  %615 = extractelement <32 x float> %559, i64 15, !dbg !7
  %616 = tail call float @llvm.maxnum.f32(float %608, float %615), !dbg !7
  %617 = extractelement <32 x float> %559, i64 23, !dbg !7
  %618 = tail call float @llvm.maxnum.f32(float %610, float %617), !dbg !7
  %619 = extractelement <32 x float> %559, i64 31, !dbg !7
  %620 = tail call float @llvm.maxnum.f32(float %612, float %619), !dbg !7
  %621 = tail call float @llvm.maxnum.f32(float %614, float %616), !dbg !7
  %622 = tail call float @llvm.maxnum.f32(float %618, float %620), !dbg !7
  %623 = tail call float @llvm.maxnum.f32(float %621, float %622), !dbg !7
  %624 = shufflevector <4 x float> %464, <4 x float> %466, <32 x i32> <i32 0, i32 4, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 5, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 2, i32 6, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 3, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %625 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %626 = shufflevector <32 x float> %624, <32 x float> %625, <32 x i32> <i32 0, i32 1, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %627 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %628 = shufflevector <32 x float> %626, <32 x float> %627, <32 x i32> <i32 0, i32 1, i32 2, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %629 = shufflevector <4 x float> %472, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %630 = shufflevector <32 x float> %628, <32 x float> %629, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 32, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %631 = shufflevector <4 x float> %474, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %632 = shufflevector <32 x float> %630, <32 x float> %631, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 32, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %633 = shufflevector <4 x float> %476, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %634 = shufflevector <32 x float> %632, <32 x float> %633, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 32, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %635 = shufflevector <4 x float> %478, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %636 = shufflevector <32 x float> %634, <32 x float> %635, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 32, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %637 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %638 = shufflevector <32 x float> %636, <32 x float> %637, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %639 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %640 = shufflevector <32 x float> %638, <32 x float> %639, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %641 = shufflevector <4 x float> %472, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %642 = shufflevector <32 x float> %640, <32 x float> %641, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 33, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %643 = shufflevector <4 x float> %474, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %644 = shufflevector <32 x float> %642, <32 x float> %643, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 33, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %645 = shufflevector <4 x float> %476, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %646 = shufflevector <32 x float> %644, <32 x float> %645, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 33, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %647 = shufflevector <4 x float> %478, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %648 = shufflevector <32 x float> %646, <32 x float> %647, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 33, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %649 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %650 = shufflevector <32 x float> %648, <32 x float> %649, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 34, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %651 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %652 = shufflevector <32 x float> %650, <32 x float> %651, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 34, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %653 = shufflevector <4 x float> %472, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %654 = shufflevector <32 x float> %652, <32 x float> %653, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 34, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %655 = shufflevector <4 x float> %474, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %656 = shufflevector <32 x float> %654, <32 x float> %655, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 34, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %657 = shufflevector <4 x float> %476, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %658 = shufflevector <32 x float> %656, <32 x float> %657, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 34, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %659 = shufflevector <4 x float> %478, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %660 = shufflevector <32 x float> %658, <32 x float> %659, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 34, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %661 = shufflevector <4 x float> %468, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %662 = shufflevector <32 x float> %660, <32 x float> %661, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %663 = shufflevector <4 x float> %470, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %664 = shufflevector <32 x float> %662, <32 x float> %663, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %665 = shufflevector <4 x float> %472, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %666 = shufflevector <32 x float> %664, <32 x float> %665, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 35, i32 poison, i32 poison, i32 poison>, !dbg !16
  %667 = shufflevector <4 x float> %474, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %668 = shufflevector <32 x float> %666, <32 x float> %667, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 35, i32 poison, i32 poison>, !dbg !16
  %669 = shufflevector <4 x float> %476, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %670 = shufflevector <32 x float> %668, <32 x float> %669, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 35, i32 poison>, !dbg !16
  %671 = shufflevector <4 x float> %478, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %672 = shufflevector <32 x float> %670, <32 x float> %671, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 35>, !dbg !16
  %673 = select <32 x i1> %483, <32 x float> %672, <32 x float> splat (float -inf), !dbg !16
  %674 = bitcast float %623 to i32, !dbg !7
  %675 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %674), !dbg !7
  %676 = bitcast i32 %675 to float, !dbg !7
  %677 = tail call float @llvm.maxnum.f32(float %623, float %676), !dbg !7
  %678 = bitcast float %677 to i32, !dbg !7
  %679 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %678), !dbg !7
  %680 = bitcast i32 %679 to float, !dbg !7
  %681 = tail call float @llvm.maxnum.f32(float %677, float %680), !dbg !7
  %682 = extractelement <32 x float> %673, i64 0, !dbg !7
  %683 = tail call float @llvm.maxnum.f32(float %682, float -inf), !dbg !7
  %684 = extractelement <32 x float> %673, i64 1, !dbg !7
  %685 = tail call float @llvm.maxnum.f32(float %683, float %684), !dbg !7
  %686 = extractelement <32 x float> %673, i64 8, !dbg !7
  %687 = extractelement <32 x float> %673, i64 9, !dbg !7
  %688 = tail call float @llvm.maxnum.f32(float %686, float %687), !dbg !7
  %689 = extractelement <32 x float> %673, i64 16, !dbg !7
  %690 = extractelement <32 x float> %673, i64 17, !dbg !7
  %691 = tail call float @llvm.maxnum.f32(float %689, float %690), !dbg !7
  %692 = extractelement <32 x float> %673, i64 24, !dbg !7
  %693 = extractelement <32 x float> %673, i64 25, !dbg !7
  %694 = tail call float @llvm.maxnum.f32(float %692, float %693), !dbg !7
  %695 = extractelement <32 x float> %673, i64 2, !dbg !7
  %696 = tail call float @llvm.maxnum.f32(float %685, float %695), !dbg !7
  %697 = extractelement <32 x float> %673, i64 10, !dbg !7
  %698 = tail call float @llvm.maxnum.f32(float %688, float %697), !dbg !7
  %699 = extractelement <32 x float> %673, i64 18, !dbg !7
  %700 = tail call float @llvm.maxnum.f32(float %691, float %699), !dbg !7
  %701 = extractelement <32 x float> %673, i64 26, !dbg !7
  %702 = tail call float @llvm.maxnum.f32(float %694, float %701), !dbg !7
  %703 = extractelement <32 x float> %673, i64 3, !dbg !7
  %704 = tail call float @llvm.maxnum.f32(float %696, float %703), !dbg !7
  %705 = extractelement <32 x float> %673, i64 11, !dbg !7
  %706 = tail call float @llvm.maxnum.f32(float %698, float %705), !dbg !7
  %707 = extractelement <32 x float> %673, i64 19, !dbg !7
  %708 = tail call float @llvm.maxnum.f32(float %700, float %707), !dbg !7
  %709 = extractelement <32 x float> %673, i64 27, !dbg !7
  %710 = tail call float @llvm.maxnum.f32(float %702, float %709), !dbg !7
  %711 = extractelement <32 x float> %673, i64 4, !dbg !7
  %712 = tail call float @llvm.maxnum.f32(float %704, float %711), !dbg !7
  %713 = extractelement <32 x float> %673, i64 12, !dbg !7
  %714 = tail call float @llvm.maxnum.f32(float %706, float %713), !dbg !7
  %715 = extractelement <32 x float> %673, i64 20, !dbg !7
  %716 = tail call float @llvm.maxnum.f32(float %708, float %715), !dbg !7
  %717 = extractelement <32 x float> %673, i64 28, !dbg !7
  %718 = tail call float @llvm.maxnum.f32(float %710, float %717), !dbg !7
  %719 = extractelement <32 x float> %673, i64 5, !dbg !7
  %720 = tail call float @llvm.maxnum.f32(float %712, float %719), !dbg !7
  %721 = extractelement <32 x float> %673, i64 13, !dbg !7
  %722 = tail call float @llvm.maxnum.f32(float %714, float %721), !dbg !7
  %723 = extractelement <32 x float> %673, i64 21, !dbg !7
  %724 = tail call float @llvm.maxnum.f32(float %716, float %723), !dbg !7
  %725 = extractelement <32 x float> %673, i64 29, !dbg !7
  %726 = tail call float @llvm.maxnum.f32(float %718, float %725), !dbg !7
  %727 = extractelement <32 x float> %673, i64 6, !dbg !7
  %728 = tail call float @llvm.maxnum.f32(float %720, float %727), !dbg !7
  %729 = extractelement <32 x float> %673, i64 14, !dbg !7
  %730 = tail call float @llvm.maxnum.f32(float %722, float %729), !dbg !7
  %731 = extractelement <32 x float> %673, i64 22, !dbg !7
  %732 = tail call float @llvm.maxnum.f32(float %724, float %731), !dbg !7
  %733 = extractelement <32 x float> %673, i64 30, !dbg !7
  %734 = tail call float @llvm.maxnum.f32(float %726, float %733), !dbg !7
  %735 = extractelement <32 x float> %673, i64 7, !dbg !7
  %736 = tail call float @llvm.maxnum.f32(float %728, float %735), !dbg !7
  %737 = extractelement <32 x float> %673, i64 15, !dbg !7
  %738 = tail call float @llvm.maxnum.f32(float %730, float %737), !dbg !7
  %739 = extractelement <32 x float> %673, i64 23, !dbg !7
  %740 = tail call float @llvm.maxnum.f32(float %732, float %739), !dbg !7
  %741 = extractelement <32 x float> %673, i64 31, !dbg !7
  %742 = tail call float @llvm.maxnum.f32(float %734, float %741), !dbg !7
  %743 = tail call float @llvm.maxnum.f32(float %736, float %738), !dbg !7
  %744 = tail call float @llvm.maxnum.f32(float %740, float %742), !dbg !7
  %745 = tail call float @llvm.maxnum.f32(float %743, float %744), !dbg !7
  %746 = bitcast float %745 to i32, !dbg !7
  %747 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %746), !dbg !7
  %748 = bitcast i32 %747 to float, !dbg !7
  %749 = tail call float @llvm.maxnum.f32(float %745, float %748), !dbg !7
  %750 = bitcast float %749 to i32, !dbg !7
  %751 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %750), !dbg !7
  %752 = bitcast i32 %751 to float, !dbg !7
  %753 = tail call float @llvm.maxnum.f32(float %749, float %752), !dbg !7
  %754 = extractvalue [2 x <1 x float>] %157, 0, !dbg !7
  %755 = insertelement <1 x float> poison, float %681, i64 0, !dbg !7
  %756 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %755, <1 x float> %754), !dbg !7
  %757 = extractvalue [2 x <1 x float>] %157, 1, !dbg !7
  %758 = insertelement <1 x float> poison, float %753, i64 0, !dbg !7
  %759 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %758, <1 x float> %757), !dbg !7
  %760 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 0, i32 8, i32 16, i32 24>, !dbg !41
  %761 = shufflevector <1 x float> %756, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !41
  %762 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 1, i32 9, i32 17, i32 25>, !dbg !41
  %763 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 2, i32 10, i32 18, i32 26>, !dbg !41
  %764 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 3, i32 11, i32 19, i32 27>, !dbg !41
  %765 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 4, i32 12, i32 20, i32 28>, !dbg !41
  %766 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 5, i32 13, i32 21, i32 29>, !dbg !41
  %767 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 6, i32 14, i32 22, i32 30>, !dbg !41
  %768 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 7, i32 15, i32 23, i32 31>, !dbg !41
  %769 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 0, i32 8, i32 16, i32 24>, !dbg !41
  %770 = shufflevector <1 x float> %759, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !41
  %771 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 1, i32 9, i32 17, i32 25>, !dbg !41
  %772 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 2, i32 10, i32 18, i32 26>, !dbg !41
  %773 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 3, i32 11, i32 19, i32 27>, !dbg !41
  %774 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 4, i32 12, i32 20, i32 28>, !dbg !41
  %775 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 5, i32 13, i32 21, i32 29>, !dbg !41
  %776 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 6, i32 14, i32 22, i32 30>, !dbg !41
  %777 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 7, i32 15, i32 23, i32 31>, !dbg !41
  %778 = fsub <1 x float> %754, %756, !dbg !42
  %779 = extractelement <1 x float> %778, i64 0, !dbg !43
  %780 = tail call float @llvm.amdgcn.exp2.f32(float %779), !dbg !43
  %781 = insertelement <4 x float> poison, float %780, i64 0, !dbg !44
  %782 = shufflevector <4 x float> %781, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !44
  %783 = extractvalue [2 x <1 x float>] %158, 0, !dbg !45
  %784 = extractelement <1 x float> %783, i64 0, !dbg !45
  %785 = fmul float %784, %780, !dbg !45
  %786 = fsub <1 x float> %757, %759, !dbg !42
  %787 = extractelement <1 x float> %786, i64 0, !dbg !43
  %788 = tail call float @llvm.amdgcn.exp2.f32(float %787), !dbg !43
  %789 = insertelement <4 x float> poison, float %788, i64 0, !dbg !44
  %790 = shufflevector <4 x float> %789, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !44
  %791 = extractvalue [2 x <1 x float>] %158, 1, !dbg !45
  %792 = extractelement <1 x float> %791, i64 0, !dbg !45
  %793 = fmul float %792, %788, !dbg !45
  %794 = extractvalue [2 x <8 x float>] %159, 0, !dbg !44
  %795 = shufflevector <8 x float> %794, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %796 = fmul <4 x float> %795, %782, !dbg !44
  %797 = shufflevector <8 x float> %794, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %798 = fmul <4 x float> %797, %782, !dbg !44
  %799 = extractvalue [2 x <8 x float>] %159, 1, !dbg !44
  %800 = shufflevector <8 x float> %799, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %801 = fmul <4 x float> %800, %790, !dbg !44
  %802 = shufflevector <8 x float> %799, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %803 = fmul <4 x float> %802, %790, !dbg !44
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %804 = or disjoint i64 %425, 32, !dbg !17
  %805 = add i64 %804, %424, !dbg !17
  %.tr1426 = trunc i64 %805 to i32, !dbg !17
  %dma.buffer.offset1318 = shl i32 %.tr1426, 1, !dbg !17
  %806 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %dma.buffer.offset1318, <4 x i32> %dma.buffer.rsrc1323) #9, !dbg !17
  %dma.buffer.offset1316 = or i32 %.tr1424, 16448, !dbg !17
  %807 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %dma.buffer.offset1316, <4 x i32> %dma.buffer.rsrc1323) #9, !dbg !17
  %address.local876 = and i32 %address.local865, 120, !dbg !46
  %address.local878 = and i32 %address.local516, 1536, !dbg !46
  %address.local879 = or disjoint i32 %address.local876, %address.local878, !dbg !46
  %address.local880 = zext nneg i32 %address.local879 to i64, !dbg !46
  %address.local881 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local880, !dbg !46
  %808 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local881, i32 0) #11, !dbg !46
  %809 = bitcast <4 x i32> %808 to <8 x half>, !dbg !46
  %address.local887 = or disjoint i32 %address.local879, 128, !dbg !46
  %address.local888 = zext nneg i32 %address.local887 to i64, !dbg !46
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local888, !dbg !46
  %810 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local889, i32 0) #11, !dbg !46
  %811 = bitcast <4 x i32> %810 to <8 x half>, !dbg !46
  %address.local895 = or disjoint i32 %address.local879, 256, !dbg !46
  %address.local896 = zext nneg i32 %address.local895 to i64, !dbg !46
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local896, !dbg !46
  %812 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local897, i32 0) #11, !dbg !46
  %813 = bitcast <4 x i32> %812 to <8 x half>, !dbg !46
  %address.local903 = or disjoint i32 %address.local879, 384, !dbg !46
  %address.local904 = zext nneg i32 %address.local903 to i64, !dbg !46
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local904, !dbg !46
  %814 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local905, i32 0) #11, !dbg !46
  %815 = bitcast <4 x i32> %814 to <8 x half>, !dbg !46
  %address.local911 = or disjoint i32 %address.local879, 2048, !dbg !46
  %address.local912 = zext nneg i32 %address.local911 to i64, !dbg !46
  %address.local913 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local912, !dbg !46
  %816 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local913, i32 0) #11, !dbg !46
  %817 = bitcast <4 x i32> %816 to <8 x half>, !dbg !46
  %address.local919 = or disjoint i32 %address.local879, 2176, !dbg !46
  %address.local920 = zext nneg i32 %address.local919 to i64, !dbg !46
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !46
  %818 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local921, i32 0) #11, !dbg !46
  %819 = bitcast <4 x i32> %818 to <8 x half>, !dbg !46
  %address.local927 = or disjoint i32 %address.local879, 2304, !dbg !46
  %address.local928 = zext nneg i32 %address.local927 to i64, !dbg !46
  %address.local929 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local928, !dbg !46
  %820 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #11, !dbg !46
  %821 = bitcast <4 x i32> %820 to <8 x half>, !dbg !46
  %address.local935 = or disjoint i32 %address.local879, 2432, !dbg !46
  %address.local936 = zext nneg i32 %address.local935 to i64, !dbg !46
  %address.local937 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local936, !dbg !46
  %822 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #11, !dbg !46
  %823 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %824 = fsub <4 x float> %760, %761, !dbg !41
  %825 = extractelement <4 x float> %824, i64 0, !dbg !47
  %826 = tail call float @llvm.amdgcn.exp2.f32(float %825), !dbg !47
  %827 = insertelement <4 x float> poison, float %826, i64 0, !dbg !48
  %828 = extractelement <4 x float> %824, i64 1, !dbg !47
  %829 = tail call float @llvm.amdgcn.exp2.f32(float %828), !dbg !47
  %830 = insertelement <4 x float> %827, float %829, i64 1, !dbg !48
  %831 = extractelement <4 x float> %824, i64 2, !dbg !47
  %832 = tail call float @llvm.amdgcn.exp2.f32(float %831), !dbg !47
  %833 = insertelement <4 x float> %830, float %832, i64 2, !dbg !48
  %834 = extractelement <4 x float> %824, i64 3, !dbg !47
  %835 = tail call float @llvm.amdgcn.exp2.f32(float %834), !dbg !47
  %836 = insertelement <4 x float> %833, float %835, i64 3, !dbg !48
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %836) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !48
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !48
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !48
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !48
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !48
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !48
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !48
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !48
  %837 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !48
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %837) #8, !dbg !47, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %823, <4 x float> %796) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %839 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %839, <4 x float> %798) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %841 = fsub <4 x float> %769, %770, !dbg !41
  %842 = extractelement <4 x float> %841, i64 0, !dbg !47
  %843 = tail call float @llvm.amdgcn.exp2.f32(float %842), !dbg !47
  %844 = insertelement <4 x float> poison, float %843, i64 0, !dbg !48
  %845 = extractelement <4 x float> %841, i64 1, !dbg !47
  %846 = tail call float @llvm.amdgcn.exp2.f32(float %845), !dbg !47
  %847 = insertelement <4 x float> %844, float %846, i64 1, !dbg !48
  %848 = extractelement <4 x float> %841, i64 2, !dbg !47
  %849 = tail call float @llvm.amdgcn.exp2.f32(float %848), !dbg !47
  %850 = insertelement <4 x float> %847, float %849, i64 2, !dbg !48
  %851 = extractelement <4 x float> %841, i64 3, !dbg !47
  %852 = tail call float @llvm.amdgcn.exp2.f32(float %851), !dbg !47
  %853 = insertelement <4 x float> %850, float %852, i64 3, !dbg !48
  %math.result329 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %853) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f01947 = extractelement <4 x float> %math.result329, i64 0, !dbg !48
  %.cvtpack.f11948 = extractelement <4 x float> %math.result329, i64 1, !dbg !48
  %.cvtpack.f21949 = extractelement <4 x float> %math.result329, i64 2, !dbg !48
  %.cvtpack.f31950 = extractelement <4 x float> %math.result329, i64 3, !dbg !48
  %.cvtpack.p01951 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01947, float %.cvtpack.f11948) #12, !dbg !48
  %.cvtpack.p11952 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21949, float %.cvtpack.f31950) #12, !dbg !48
  %.cvtpack.v01953 = insertelement <2 x i32> poison, i32 %.cvtpack.p01951, i64 0, !dbg !48
  %.cvtpack.v11954 = insertelement <2 x i32> %.cvtpack.v01953, i32 %.cvtpack.p11952, i64 1, !dbg !48
  %854 = bitcast <2 x i32> %.cvtpack.v11954 to <4 x half>, !dbg !48
  %math.packed330 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %854) #8, !dbg !47, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %855 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %823, <4 x float> %801) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %839, <4 x float> %803) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %857 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %858 = fsub <4 x float> %762, %761, !dbg !41
  %859 = extractelement <4 x float> %858, i64 0, !dbg !47
  %860 = tail call float @llvm.amdgcn.exp2.f32(float %859), !dbg !47
  %861 = fadd float %826, %860, !dbg !49
  %862 = insertelement <4 x float> poison, float %860, i64 0, !dbg !48
  %863 = extractelement <4 x float> %858, i64 1, !dbg !47
  %864 = tail call float @llvm.amdgcn.exp2.f32(float %863), !dbg !47
  %865 = fadd float %829, %864, !dbg !49
  %866 = insertelement <4 x float> %862, float %864, i64 1, !dbg !48
  %867 = extractelement <4 x float> %858, i64 2, !dbg !47
  %868 = tail call float @llvm.amdgcn.exp2.f32(float %867), !dbg !47
  %869 = fadd float %832, %868, !dbg !49
  %870 = insertelement <4 x float> %866, float %868, i64 2, !dbg !48
  %871 = extractelement <4 x float> %858, i64 3, !dbg !47
  %872 = tail call float @llvm.amdgcn.exp2.f32(float %871), !dbg !47
  %873 = fadd float %835, %872, !dbg !49
  %874 = insertelement <4 x float> %870, float %872, i64 3, !dbg !48
  %math.result315 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %874) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f01955 = extractelement <4 x float> %math.result315, i64 0, !dbg !48
  %.cvtpack.f11956 = extractelement <4 x float> %math.result315, i64 1, !dbg !48
  %.cvtpack.f21957 = extractelement <4 x float> %math.result315, i64 2, !dbg !48
  %.cvtpack.f31958 = extractelement <4 x float> %math.result315, i64 3, !dbg !48
  %.cvtpack.p01959 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01955, float %.cvtpack.f11956) #12, !dbg !48
  %.cvtpack.p11960 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21957, float %.cvtpack.f31958) #12, !dbg !48
  %.cvtpack.v01961 = insertelement <2 x i32> poison, i32 %.cvtpack.p01959, i64 0, !dbg !48
  %.cvtpack.v11962 = insertelement <2 x i32> %.cvtpack.v01961, i32 %.cvtpack.p11960, i64 1, !dbg !48
  %875 = bitcast <2 x i32> %.cvtpack.v11962 to <4 x half>, !dbg !48
  %math.packed316 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %875) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %876 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %857, <4 x float> %838) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %877 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %878 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %877, <4 x float> %840) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %879 = fsub <4 x float> %771, %770, !dbg !41
  %880 = extractelement <4 x float> %879, i64 0, !dbg !47
  %881 = tail call float @llvm.amdgcn.exp2.f32(float %880), !dbg !47
  %882 = fadd float %843, %881, !dbg !49
  %883 = insertelement <4 x float> poison, float %881, i64 0, !dbg !48
  %884 = extractelement <4 x float> %879, i64 1, !dbg !47
  %885 = tail call float @llvm.amdgcn.exp2.f32(float %884), !dbg !47
  %886 = fadd float %846, %885, !dbg !49
  %887 = insertelement <4 x float> %883, float %885, i64 1, !dbg !48
  %888 = extractelement <4 x float> %879, i64 2, !dbg !47
  %889 = tail call float @llvm.amdgcn.exp2.f32(float %888), !dbg !47
  %890 = fadd float %849, %889, !dbg !49
  %891 = insertelement <4 x float> %887, float %889, i64 2, !dbg !48
  %892 = extractelement <4 x float> %879, i64 3, !dbg !47
  %893 = tail call float @llvm.amdgcn.exp2.f32(float %892), !dbg !47
  %894 = fadd float %852, %893, !dbg !49
  %895 = insertelement <4 x float> %891, float %893, i64 3, !dbg !48
  %math.result331 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %895) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f01963 = extractelement <4 x float> %math.result331, i64 0, !dbg !48
  %.cvtpack.f11964 = extractelement <4 x float> %math.result331, i64 1, !dbg !48
  %.cvtpack.f21965 = extractelement <4 x float> %math.result331, i64 2, !dbg !48
  %.cvtpack.f31966 = extractelement <4 x float> %math.result331, i64 3, !dbg !48
  %.cvtpack.p01967 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01963, float %.cvtpack.f11964) #12, !dbg !48
  %.cvtpack.p11968 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21965, float %.cvtpack.f31966) #12, !dbg !48
  %.cvtpack.v01969 = insertelement <2 x i32> poison, i32 %.cvtpack.p01967, i64 0, !dbg !48
  %.cvtpack.v11970 = insertelement <2 x i32> %.cvtpack.v01969, i32 %.cvtpack.p11968, i64 1, !dbg !48
  %896 = bitcast <2 x i32> %.cvtpack.v11970 to <4 x half>, !dbg !48
  %math.packed332 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %896) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %897 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %857, <4 x float> %855) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %898 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %877, <4 x float> %856) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %899 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %900 = fsub <4 x float> %763, %761, !dbg !41
  %901 = extractelement <4 x float> %900, i64 0, !dbg !47
  %902 = tail call float @llvm.amdgcn.exp2.f32(float %901), !dbg !47
  %903 = fadd float %902, %861, !dbg !49
  %904 = insertelement <4 x float> poison, float %902, i64 0, !dbg !48
  %905 = extractelement <4 x float> %900, i64 1, !dbg !47
  %906 = tail call float @llvm.amdgcn.exp2.f32(float %905), !dbg !47
  %907 = fadd float %906, %865, !dbg !49
  %908 = insertelement <4 x float> %904, float %906, i64 1, !dbg !48
  %909 = extractelement <4 x float> %900, i64 2, !dbg !47
  %910 = tail call float @llvm.amdgcn.exp2.f32(float %909), !dbg !47
  %911 = fadd float %910, %869, !dbg !49
  %912 = insertelement <4 x float> %908, float %910, i64 2, !dbg !48
  %913 = extractelement <4 x float> %900, i64 3, !dbg !47
  %914 = tail call float @llvm.amdgcn.exp2.f32(float %913), !dbg !47
  %915 = fadd float %914, %873, !dbg !49
  %916 = insertelement <4 x float> %912, float %914, i64 3, !dbg !48
  %math.result317 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %916) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f01971 = extractelement <4 x float> %math.result317, i64 0, !dbg !48
  %.cvtpack.f11972 = extractelement <4 x float> %math.result317, i64 1, !dbg !48
  %.cvtpack.f21973 = extractelement <4 x float> %math.result317, i64 2, !dbg !48
  %.cvtpack.f31974 = extractelement <4 x float> %math.result317, i64 3, !dbg !48
  %.cvtpack.p01975 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01971, float %.cvtpack.f11972) #12, !dbg !48
  %.cvtpack.p11976 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21973, float %.cvtpack.f31974) #12, !dbg !48
  %.cvtpack.v01977 = insertelement <2 x i32> poison, i32 %.cvtpack.p01975, i64 0, !dbg !48
  %.cvtpack.v11978 = insertelement <2 x i32> %.cvtpack.v01977, i32 %.cvtpack.p11976, i64 1, !dbg !48
  %917 = bitcast <2 x i32> %.cvtpack.v11978 to <4 x half>, !dbg !48
  %math.packed318 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %917) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %899, <4 x float> %876) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %919 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %919, <4 x float> %878) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %921 = fsub <4 x float> %772, %770, !dbg !41
  %922 = extractelement <4 x float> %921, i64 0, !dbg !47
  %923 = tail call float @llvm.amdgcn.exp2.f32(float %922), !dbg !47
  %924 = fadd float %923, %882, !dbg !49
  %925 = insertelement <4 x float> poison, float %923, i64 0, !dbg !48
  %926 = extractelement <4 x float> %921, i64 1, !dbg !47
  %927 = tail call float @llvm.amdgcn.exp2.f32(float %926), !dbg !47
  %928 = fadd float %927, %886, !dbg !49
  %929 = insertelement <4 x float> %925, float %927, i64 1, !dbg !48
  %930 = extractelement <4 x float> %921, i64 2, !dbg !47
  %931 = tail call float @llvm.amdgcn.exp2.f32(float %930), !dbg !47
  %932 = fadd float %931, %890, !dbg !49
  %933 = insertelement <4 x float> %929, float %931, i64 2, !dbg !48
  %934 = extractelement <4 x float> %921, i64 3, !dbg !47
  %935 = tail call float @llvm.amdgcn.exp2.f32(float %934), !dbg !47
  %936 = fadd float %935, %894, !dbg !49
  %937 = insertelement <4 x float> %933, float %935, i64 3, !dbg !48
  %math.result333 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %937) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f01979 = extractelement <4 x float> %math.result333, i64 0, !dbg !48
  %.cvtpack.f11980 = extractelement <4 x float> %math.result333, i64 1, !dbg !48
  %.cvtpack.f21981 = extractelement <4 x float> %math.result333, i64 2, !dbg !48
  %.cvtpack.f31982 = extractelement <4 x float> %math.result333, i64 3, !dbg !48
  %.cvtpack.p01983 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01979, float %.cvtpack.f11980) #12, !dbg !48
  %.cvtpack.p11984 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21981, float %.cvtpack.f31982) #12, !dbg !48
  %.cvtpack.v01985 = insertelement <2 x i32> poison, i32 %.cvtpack.p01983, i64 0, !dbg !48
  %.cvtpack.v11986 = insertelement <2 x i32> %.cvtpack.v01985, i32 %.cvtpack.p11984, i64 1, !dbg !48
  %938 = bitcast <2 x i32> %.cvtpack.v11986 to <4 x half>, !dbg !48
  %math.packed334 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %938) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %939 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %899, <4 x float> %897) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %940 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %919, <4 x float> %898) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %941 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %942 = fsub <4 x float> %764, %761, !dbg !41
  %943 = extractelement <4 x float> %942, i64 0, !dbg !47
  %944 = tail call float @llvm.amdgcn.exp2.f32(float %943), !dbg !47
  %945 = fadd float %944, %903, !dbg !49
  %946 = insertelement <4 x float> poison, float %944, i64 0, !dbg !48
  %947 = extractelement <4 x float> %942, i64 1, !dbg !47
  %948 = tail call float @llvm.amdgcn.exp2.f32(float %947), !dbg !47
  %949 = fadd float %948, %907, !dbg !49
  %950 = insertelement <4 x float> %946, float %948, i64 1, !dbg !48
  %951 = extractelement <4 x float> %942, i64 2, !dbg !47
  %952 = tail call float @llvm.amdgcn.exp2.f32(float %951), !dbg !47
  %953 = fadd float %952, %911, !dbg !49
  %954 = insertelement <4 x float> %950, float %952, i64 2, !dbg !48
  %955 = extractelement <4 x float> %942, i64 3, !dbg !47
  %956 = tail call float @llvm.amdgcn.exp2.f32(float %955), !dbg !47
  %957 = fadd float %956, %915, !dbg !49
  %958 = insertelement <4 x float> %954, float %956, i64 3, !dbg !48
  %math.result319 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %958) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f01987 = extractelement <4 x float> %math.result319, i64 0, !dbg !48
  %.cvtpack.f11988 = extractelement <4 x float> %math.result319, i64 1, !dbg !48
  %.cvtpack.f21989 = extractelement <4 x float> %math.result319, i64 2, !dbg !48
  %.cvtpack.f31990 = extractelement <4 x float> %math.result319, i64 3, !dbg !48
  %.cvtpack.p01991 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01987, float %.cvtpack.f11988) #12, !dbg !48
  %.cvtpack.p11992 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21989, float %.cvtpack.f31990) #12, !dbg !48
  %.cvtpack.v01993 = insertelement <2 x i32> poison, i32 %.cvtpack.p01991, i64 0, !dbg !48
  %.cvtpack.v11994 = insertelement <2 x i32> %.cvtpack.v01993, i32 %.cvtpack.p11992, i64 1, !dbg !48
  %959 = bitcast <2 x i32> %.cvtpack.v11994 to <4 x half>, !dbg !48
  %math.packed320 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %959) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %941, <4 x float> %918) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %961 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %961, <4 x float> %920) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %963 = fsub <4 x float> %773, %770, !dbg !41
  %964 = extractelement <4 x float> %963, i64 0, !dbg !47
  %965 = tail call float @llvm.amdgcn.exp2.f32(float %964), !dbg !47
  %966 = fadd float %965, %924, !dbg !49
  %967 = insertelement <4 x float> poison, float %965, i64 0, !dbg !48
  %968 = extractelement <4 x float> %963, i64 1, !dbg !47
  %969 = tail call float @llvm.amdgcn.exp2.f32(float %968), !dbg !47
  %970 = fadd float %969, %928, !dbg !49
  %971 = insertelement <4 x float> %967, float %969, i64 1, !dbg !48
  %972 = extractelement <4 x float> %963, i64 2, !dbg !47
  %973 = tail call float @llvm.amdgcn.exp2.f32(float %972), !dbg !47
  %974 = fadd float %973, %932, !dbg !49
  %975 = insertelement <4 x float> %971, float %973, i64 2, !dbg !48
  %976 = extractelement <4 x float> %963, i64 3, !dbg !47
  %977 = tail call float @llvm.amdgcn.exp2.f32(float %976), !dbg !47
  %978 = fadd float %977, %936, !dbg !49
  %979 = insertelement <4 x float> %975, float %977, i64 3, !dbg !48
  %math.result335 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %979) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f01995 = extractelement <4 x float> %math.result335, i64 0, !dbg !48
  %.cvtpack.f11996 = extractelement <4 x float> %math.result335, i64 1, !dbg !48
  %.cvtpack.f21997 = extractelement <4 x float> %math.result335, i64 2, !dbg !48
  %.cvtpack.f31998 = extractelement <4 x float> %math.result335, i64 3, !dbg !48
  %.cvtpack.p01999 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01995, float %.cvtpack.f11996) #12, !dbg !48
  %.cvtpack.p12000 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21997, float %.cvtpack.f31998) #12, !dbg !48
  %.cvtpack.v02001 = insertelement <2 x i32> poison, i32 %.cvtpack.p01999, i64 0, !dbg !48
  %.cvtpack.v12002 = insertelement <2 x i32> %.cvtpack.v02001, i32 %.cvtpack.p12000, i64 1, !dbg !48
  %980 = bitcast <2 x i32> %.cvtpack.v12002 to <4 x half>, !dbg !48
  %math.packed336 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %980) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %941, <4 x float> %939) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %961, <4 x float> %940) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %983 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %984 = fsub <4 x float> %765, %761, !dbg !41
  %985 = extractelement <4 x float> %984, i64 0, !dbg !47
  %986 = tail call float @llvm.amdgcn.exp2.f32(float %985), !dbg !47
  %987 = fadd float %986, %945, !dbg !49
  %988 = insertelement <4 x float> poison, float %986, i64 0, !dbg !48
  %989 = extractelement <4 x float> %984, i64 1, !dbg !47
  %990 = tail call float @llvm.amdgcn.exp2.f32(float %989), !dbg !47
  %991 = fadd float %990, %949, !dbg !49
  %992 = insertelement <4 x float> %988, float %990, i64 1, !dbg !48
  %993 = extractelement <4 x float> %984, i64 2, !dbg !47
  %994 = tail call float @llvm.amdgcn.exp2.f32(float %993), !dbg !47
  %995 = fadd float %994, %953, !dbg !49
  %996 = insertelement <4 x float> %992, float %994, i64 2, !dbg !48
  %997 = extractelement <4 x float> %984, i64 3, !dbg !47
  %998 = tail call float @llvm.amdgcn.exp2.f32(float %997), !dbg !47
  %999 = fadd float %998, %957, !dbg !49
  %1000 = insertelement <4 x float> %996, float %998, i64 3, !dbg !48
  %math.result321 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1000) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f02003 = extractelement <4 x float> %math.result321, i64 0, !dbg !48
  %.cvtpack.f12004 = extractelement <4 x float> %math.result321, i64 1, !dbg !48
  %.cvtpack.f22005 = extractelement <4 x float> %math.result321, i64 2, !dbg !48
  %.cvtpack.f32006 = extractelement <4 x float> %math.result321, i64 3, !dbg !48
  %.cvtpack.p02007 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02003, float %.cvtpack.f12004) #12, !dbg !48
  %.cvtpack.p12008 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22005, float %.cvtpack.f32006) #12, !dbg !48
  %.cvtpack.v02009 = insertelement <2 x i32> poison, i32 %.cvtpack.p02007, i64 0, !dbg !48
  %.cvtpack.v12010 = insertelement <2 x i32> %.cvtpack.v02009, i32 %.cvtpack.p12008, i64 1, !dbg !48
  %1001 = bitcast <2 x i32> %.cvtpack.v12010 to <4 x half>, !dbg !48
  %math.packed322 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1001) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %983, <4 x float> %960) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1003 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1003, <4 x float> %962) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1005 = fsub <4 x float> %774, %770, !dbg !41
  %1006 = extractelement <4 x float> %1005, i64 0, !dbg !47
  %1007 = tail call float @llvm.amdgcn.exp2.f32(float %1006), !dbg !47
  %1008 = fadd float %1007, %966, !dbg !49
  %1009 = insertelement <4 x float> poison, float %1007, i64 0, !dbg !48
  %1010 = extractelement <4 x float> %1005, i64 1, !dbg !47
  %1011 = tail call float @llvm.amdgcn.exp2.f32(float %1010), !dbg !47
  %1012 = fadd float %1011, %970, !dbg !49
  %1013 = insertelement <4 x float> %1009, float %1011, i64 1, !dbg !48
  %1014 = extractelement <4 x float> %1005, i64 2, !dbg !47
  %1015 = tail call float @llvm.amdgcn.exp2.f32(float %1014), !dbg !47
  %1016 = fadd float %1015, %974, !dbg !49
  %1017 = insertelement <4 x float> %1013, float %1015, i64 2, !dbg !48
  %1018 = extractelement <4 x float> %1005, i64 3, !dbg !47
  %1019 = tail call float @llvm.amdgcn.exp2.f32(float %1018), !dbg !47
  %1020 = fadd float %1019, %978, !dbg !49
  %1021 = insertelement <4 x float> %1017, float %1019, i64 3, !dbg !48
  %math.result337 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1021) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f02011 = extractelement <4 x float> %math.result337, i64 0, !dbg !48
  %.cvtpack.f12012 = extractelement <4 x float> %math.result337, i64 1, !dbg !48
  %.cvtpack.f22013 = extractelement <4 x float> %math.result337, i64 2, !dbg !48
  %.cvtpack.f32014 = extractelement <4 x float> %math.result337, i64 3, !dbg !48
  %.cvtpack.p02015 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02011, float %.cvtpack.f12012) #12, !dbg !48
  %.cvtpack.p12016 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22013, float %.cvtpack.f32014) #12, !dbg !48
  %.cvtpack.v02017 = insertelement <2 x i32> poison, i32 %.cvtpack.p02015, i64 0, !dbg !48
  %.cvtpack.v12018 = insertelement <2 x i32> %.cvtpack.v02017, i32 %.cvtpack.p12016, i64 1, !dbg !48
  %1022 = bitcast <2 x i32> %.cvtpack.v12018 to <4 x half>, !dbg !48
  %math.packed338 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1022) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %1023 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %983, <4 x float> %981) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1003, <4 x float> %982) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1025 = shufflevector <8 x half> %819, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1026 = fsub <4 x float> %766, %761, !dbg !41
  %1027 = extractelement <4 x float> %1026, i64 0, !dbg !47
  %1028 = tail call float @llvm.amdgcn.exp2.f32(float %1027), !dbg !47
  %1029 = fadd float %1028, %987, !dbg !49
  %1030 = insertelement <4 x float> poison, float %1028, i64 0, !dbg !48
  %1031 = extractelement <4 x float> %1026, i64 1, !dbg !47
  %1032 = tail call float @llvm.amdgcn.exp2.f32(float %1031), !dbg !47
  %1033 = fadd float %1032, %991, !dbg !49
  %1034 = insertelement <4 x float> %1030, float %1032, i64 1, !dbg !48
  %1035 = extractelement <4 x float> %1026, i64 2, !dbg !47
  %1036 = tail call float @llvm.amdgcn.exp2.f32(float %1035), !dbg !47
  %1037 = fadd float %1036, %995, !dbg !49
  %1038 = insertelement <4 x float> %1034, float %1036, i64 2, !dbg !48
  %1039 = extractelement <4 x float> %1026, i64 3, !dbg !47
  %1040 = tail call float @llvm.amdgcn.exp2.f32(float %1039), !dbg !47
  %1041 = fadd float %1040, %999, !dbg !49
  %1042 = insertelement <4 x float> %1038, float %1040, i64 3, !dbg !48
  %math.result323 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1042) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f02019 = extractelement <4 x float> %math.result323, i64 0, !dbg !48
  %.cvtpack.f12020 = extractelement <4 x float> %math.result323, i64 1, !dbg !48
  %.cvtpack.f22021 = extractelement <4 x float> %math.result323, i64 2, !dbg !48
  %.cvtpack.f32022 = extractelement <4 x float> %math.result323, i64 3, !dbg !48
  %.cvtpack.p02023 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02019, float %.cvtpack.f12020) #12, !dbg !48
  %.cvtpack.p12024 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22021, float %.cvtpack.f32022) #12, !dbg !48
  %.cvtpack.v02025 = insertelement <2 x i32> poison, i32 %.cvtpack.p02023, i64 0, !dbg !48
  %.cvtpack.v12026 = insertelement <2 x i32> %.cvtpack.v02025, i32 %.cvtpack.p12024, i64 1, !dbg !48
  %1043 = bitcast <2 x i32> %.cvtpack.v12026 to <4 x half>, !dbg !48
  %math.packed324 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1043) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %1044 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1025, <4 x float> %1002) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1045 = shufflevector <8 x half> %819, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1046 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1045, <4 x float> %1004) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1047 = fsub <4 x float> %775, %770, !dbg !41
  %1048 = extractelement <4 x float> %1047, i64 0, !dbg !47
  %1049 = tail call float @llvm.amdgcn.exp2.f32(float %1048), !dbg !47
  %1050 = fadd float %1049, %1008, !dbg !49
  %1051 = insertelement <4 x float> poison, float %1049, i64 0, !dbg !48
  %1052 = extractelement <4 x float> %1047, i64 1, !dbg !47
  %1053 = tail call float @llvm.amdgcn.exp2.f32(float %1052), !dbg !47
  %1054 = fadd float %1053, %1012, !dbg !49
  %1055 = insertelement <4 x float> %1051, float %1053, i64 1, !dbg !48
  %1056 = extractelement <4 x float> %1047, i64 2, !dbg !47
  %1057 = tail call float @llvm.amdgcn.exp2.f32(float %1056), !dbg !47
  %1058 = fadd float %1057, %1016, !dbg !49
  %1059 = insertelement <4 x float> %1055, float %1057, i64 2, !dbg !48
  %1060 = extractelement <4 x float> %1047, i64 3, !dbg !47
  %1061 = tail call float @llvm.amdgcn.exp2.f32(float %1060), !dbg !47
  %1062 = fadd float %1061, %1020, !dbg !49
  %1063 = insertelement <4 x float> %1059, float %1061, i64 3, !dbg !48
  %math.result339 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1063) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f02027 = extractelement <4 x float> %math.result339, i64 0, !dbg !48
  %.cvtpack.f12028 = extractelement <4 x float> %math.result339, i64 1, !dbg !48
  %.cvtpack.f22029 = extractelement <4 x float> %math.result339, i64 2, !dbg !48
  %.cvtpack.f32030 = extractelement <4 x float> %math.result339, i64 3, !dbg !48
  %.cvtpack.p02031 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02027, float %.cvtpack.f12028) #12, !dbg !48
  %.cvtpack.p12032 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22029, float %.cvtpack.f32030) #12, !dbg !48
  %.cvtpack.v02033 = insertelement <2 x i32> poison, i32 %.cvtpack.p02031, i64 0, !dbg !48
  %.cvtpack.v12034 = insertelement <2 x i32> %.cvtpack.v02033, i32 %.cvtpack.p12032, i64 1, !dbg !48
  %1064 = bitcast <2 x i32> %.cvtpack.v12034 to <4 x half>, !dbg !48
  %math.packed340 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1064) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %1065 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1025, <4 x float> %1023) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1066 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1045, <4 x float> %1024) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1067 = shufflevector <8 x half> %821, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1068 = fsub <4 x float> %767, %761, !dbg !41
  %1069 = extractelement <4 x float> %1068, i64 0, !dbg !47
  %1070 = tail call float @llvm.amdgcn.exp2.f32(float %1069), !dbg !47
  %1071 = fadd float %1070, %1029, !dbg !49
  %1072 = insertelement <4 x float> poison, float %1070, i64 0, !dbg !48
  %1073 = extractelement <4 x float> %1068, i64 1, !dbg !47
  %1074 = tail call float @llvm.amdgcn.exp2.f32(float %1073), !dbg !47
  %1075 = fadd float %1074, %1033, !dbg !49
  %1076 = insertelement <4 x float> %1072, float %1074, i64 1, !dbg !48
  %1077 = extractelement <4 x float> %1068, i64 2, !dbg !47
  %1078 = tail call float @llvm.amdgcn.exp2.f32(float %1077), !dbg !47
  %1079 = fadd float %1078, %1037, !dbg !49
  %1080 = insertelement <4 x float> %1076, float %1078, i64 2, !dbg !48
  %1081 = extractelement <4 x float> %1068, i64 3, !dbg !47
  %1082 = tail call float @llvm.amdgcn.exp2.f32(float %1081), !dbg !47
  %1083 = fadd float %1082, %1041, !dbg !49
  %1084 = insertelement <4 x float> %1080, float %1082, i64 3, !dbg !48
  %math.result325 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1084) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f02035 = extractelement <4 x float> %math.result325, i64 0, !dbg !48
  %.cvtpack.f12036 = extractelement <4 x float> %math.result325, i64 1, !dbg !48
  %.cvtpack.f22037 = extractelement <4 x float> %math.result325, i64 2, !dbg !48
  %.cvtpack.f32038 = extractelement <4 x float> %math.result325, i64 3, !dbg !48
  %.cvtpack.p02039 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02035, float %.cvtpack.f12036) #12, !dbg !48
  %.cvtpack.p12040 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22037, float %.cvtpack.f32038) #12, !dbg !48
  %.cvtpack.v02041 = insertelement <2 x i32> poison, i32 %.cvtpack.p02039, i64 0, !dbg !48
  %.cvtpack.v12042 = insertelement <2 x i32> %.cvtpack.v02041, i32 %.cvtpack.p12040, i64 1, !dbg !48
  %1085 = bitcast <2 x i32> %.cvtpack.v12042 to <4 x half>, !dbg !48
  %math.packed326 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1085) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1067, <4 x float> %1044) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1087 = shufflevector <8 x half> %821, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1087, <4 x float> %1046) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1089 = fsub <4 x float> %776, %770, !dbg !41
  %1090 = extractelement <4 x float> %1089, i64 0, !dbg !47
  %1091 = tail call float @llvm.amdgcn.exp2.f32(float %1090), !dbg !47
  %1092 = fadd float %1091, %1050, !dbg !49
  %1093 = insertelement <4 x float> poison, float %1091, i64 0, !dbg !48
  %1094 = extractelement <4 x float> %1089, i64 1, !dbg !47
  %1095 = tail call float @llvm.amdgcn.exp2.f32(float %1094), !dbg !47
  %1096 = fadd float %1095, %1054, !dbg !49
  %1097 = insertelement <4 x float> %1093, float %1095, i64 1, !dbg !48
  %1098 = extractelement <4 x float> %1089, i64 2, !dbg !47
  %1099 = tail call float @llvm.amdgcn.exp2.f32(float %1098), !dbg !47
  %1100 = fadd float %1099, %1058, !dbg !49
  %1101 = insertelement <4 x float> %1097, float %1099, i64 2, !dbg !48
  %1102 = extractelement <4 x float> %1089, i64 3, !dbg !47
  %1103 = tail call float @llvm.amdgcn.exp2.f32(float %1102), !dbg !47
  %1104 = fadd float %1103, %1062, !dbg !49
  %1105 = insertelement <4 x float> %1101, float %1103, i64 3, !dbg !48
  %math.result341 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1105) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f02043 = extractelement <4 x float> %math.result341, i64 0, !dbg !48
  %.cvtpack.f12044 = extractelement <4 x float> %math.result341, i64 1, !dbg !48
  %.cvtpack.f22045 = extractelement <4 x float> %math.result341, i64 2, !dbg !48
  %.cvtpack.f32046 = extractelement <4 x float> %math.result341, i64 3, !dbg !48
  %.cvtpack.p02047 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02043, float %.cvtpack.f12044) #12, !dbg !48
  %.cvtpack.p12048 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22045, float %.cvtpack.f32046) #12, !dbg !48
  %.cvtpack.v02049 = insertelement <2 x i32> poison, i32 %.cvtpack.p02047, i64 0, !dbg !48
  %.cvtpack.v12050 = insertelement <2 x i32> %.cvtpack.v02049, i32 %.cvtpack.p12048, i64 1, !dbg !48
  %1106 = bitcast <2 x i32> %.cvtpack.v12050 to <4 x half>, !dbg !48
  %math.packed342 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1106) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %1107 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1067, <4 x float> %1065) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1108 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1087, <4 x float> %1066) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1109 = bitcast <4 x i32> %822 to <8 x half>, !dbg !46
  %1110 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1111 = fsub <4 x float> %768, %761, !dbg !41
  %1112 = extractelement <4 x float> %1111, i64 0, !dbg !47
  %1113 = tail call float @llvm.amdgcn.exp2.f32(float %1112), !dbg !47
  %1114 = fadd float %1113, %1071, !dbg !49
  %1115 = insertelement <4 x float> poison, float %1113, i64 0, !dbg !48
  %1116 = extractelement <4 x float> %1111, i64 1, !dbg !47
  %1117 = tail call float @llvm.amdgcn.exp2.f32(float %1116), !dbg !47
  %1118 = fadd float %1117, %1075, !dbg !49
  %1119 = fadd float %1114, %1118, !dbg !49
  %1120 = insertelement <4 x float> %1115, float %1117, i64 1, !dbg !48
  %1121 = extractelement <4 x float> %1111, i64 2, !dbg !47
  %1122 = tail call float @llvm.amdgcn.exp2.f32(float %1121), !dbg !47
  %1123 = fadd float %1122, %1079, !dbg !49
  %1124 = insertelement <4 x float> %1120, float %1122, i64 2, !dbg !48
  %1125 = extractelement <4 x float> %1111, i64 3, !dbg !47
  %1126 = tail call float @llvm.amdgcn.exp2.f32(float %1125), !dbg !47
  %1127 = fadd float %1126, %1083, !dbg !49
  %1128 = fadd float %1123, %1127, !dbg !49
  %1129 = fadd float %1119, %1128, !dbg !49
  %1130 = fadd float %785, %1129, !dbg !49
  %1131 = insertelement <1 x float> poison, float %1130, i64 0, !dbg !12
  %1132 = insertvalue [2 x <1 x float>] poison, <1 x float> %1131, 0, !dbg !12
  %1133 = insertelement <4 x float> %1124, float %1126, i64 3, !dbg !48
  %math.result327 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1133) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f02051 = extractelement <4 x float> %math.result327, i64 0, !dbg !48
  %.cvtpack.f12052 = extractelement <4 x float> %math.result327, i64 1, !dbg !48
  %.cvtpack.f22053 = extractelement <4 x float> %math.result327, i64 2, !dbg !48
  %.cvtpack.f32054 = extractelement <4 x float> %math.result327, i64 3, !dbg !48
  %.cvtpack.p02055 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02051, float %.cvtpack.f12052) #12, !dbg !48
  %.cvtpack.p12056 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22053, float %.cvtpack.f32054) #12, !dbg !48
  %.cvtpack.v02057 = insertelement <2 x i32> poison, i32 %.cvtpack.p02055, i64 0, !dbg !48
  %.cvtpack.v12058 = insertelement <2 x i32> %.cvtpack.v02057, i32 %.cvtpack.p12056, i64 1, !dbg !48
  %1134 = bitcast <2 x i32> %.cvtpack.v12058 to <4 x half>, !dbg !48
  %math.packed328 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1134) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %1135 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1110, <4 x float> %1086) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1136 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1136, <4 x float> %1088) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1138 = fsub <4 x float> %777, %770, !dbg !41
  %1139 = extractelement <4 x float> %1138, i64 0, !dbg !47
  %1140 = tail call float @llvm.amdgcn.exp2.f32(float %1139), !dbg !47
  %1141 = fadd float %1140, %1092, !dbg !49
  %1142 = insertelement <4 x float> poison, float %1140, i64 0, !dbg !48
  %1143 = extractelement <4 x float> %1138, i64 1, !dbg !47
  %1144 = tail call float @llvm.amdgcn.exp2.f32(float %1143), !dbg !47
  %1145 = fadd float %1144, %1096, !dbg !49
  %1146 = fadd float %1141, %1145, !dbg !49
  %1147 = insertelement <4 x float> %1142, float %1144, i64 1, !dbg !48
  %1148 = extractelement <4 x float> %1138, i64 2, !dbg !47
  %1149 = tail call float @llvm.amdgcn.exp2.f32(float %1148), !dbg !47
  %1150 = fadd float %1149, %1100, !dbg !49
  %1151 = insertelement <4 x float> %1147, float %1149, i64 2, !dbg !48
  %1152 = extractelement <4 x float> %1138, i64 3, !dbg !47
  %1153 = tail call float @llvm.amdgcn.exp2.f32(float %1152), !dbg !47
  %1154 = fadd float %1153, %1104, !dbg !49
  %1155 = fadd float %1150, %1154, !dbg !49
  %1156 = fadd float %1146, %1155, !dbg !49
  %1157 = fadd float %793, %1156, !dbg !49
  %1158 = insertelement <1 x float> poison, float %1157, i64 0, !dbg !12
  %1159 = insertvalue [2 x <1 x float>] %1132, <1 x float> %1158, 1, !dbg !12
  %1160 = insertelement <4 x float> %1151, float %1153, i64 3, !dbg !48
  %math.result343 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1160) #8, !dbg !48, !frisk.register_lifetimes !5
  %.cvtpack.f02059 = extractelement <4 x float> %math.result343, i64 0, !dbg !48
  %.cvtpack.f12060 = extractelement <4 x float> %math.result343, i64 1, !dbg !48
  %.cvtpack.f22061 = extractelement <4 x float> %math.result343, i64 2, !dbg !48
  %.cvtpack.f32062 = extractelement <4 x float> %math.result343, i64 3, !dbg !48
  %.cvtpack.p02063 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02059, float %.cvtpack.f12060) #12, !dbg !48
  %.cvtpack.p12064 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22061, float %.cvtpack.f32062) #12, !dbg !48
  %.cvtpack.v02065 = insertelement <2 x i32> poison, i32 %.cvtpack.p02063, i64 0, !dbg !48
  %.cvtpack.v12066 = insertelement <2 x i32> %.cvtpack.v02065, i32 %.cvtpack.p12064, i64 1, !dbg !48
  %1161 = bitcast <2 x i32> %.cvtpack.v12066 to <4 x half>, !dbg !48
  %math.packed344 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1161) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1110, <4 x float> %1107) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1136, <4 x float> %1108) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %1164 = extractvalue [2 x <8 x float>] %156, 0, !dbg !44
  %1165 = shufflevector <8 x float> %1164, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1166 = fmul <4 x float> %1165, %782, !dbg !44
  %1167 = shufflevector <8 x float> %1164, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1168 = fmul <4 x float> %1167, %782, !dbg !44
  %1169 = extractvalue [2 x <8 x float>] %156, 1, !dbg !44
  %1170 = shufflevector <8 x float> %1169, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1171 = fmul <4 x float> %1170, %790, !dbg !44
  %1172 = shufflevector <8 x float> %1169, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1173 = fmul <4 x float> %1172, %790, !dbg !44
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1174 = or disjoint i64 %425, 64, !dbg !17
  %1175 = add i64 %1174, %424, !dbg !17
  %.tr1430 = trunc i64 %1175 to i32, !dbg !17
  %dma.buffer.offset1314 = shl i32 %.tr1430, 1, !dbg !17
  %1176 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1314, <4 x i32> %dma.buffer.rsrc1323) #9, !dbg !17
  %dma.buffer.offset1312 = or i32 %.tr1424, 16512, !dbg !17
  %1177 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset1312, <4 x i32> %dma.buffer.rsrc1323) #9, !dbg !17
  %address.local940 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local880, !dbg !46
  %1178 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local940, i32 0) #11, !dbg !46
  %1179 = bitcast <4 x i32> %1178 to <8 x half>, !dbg !46
  %address.local941 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local888, !dbg !46
  %1180 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local941, i32 0) #11, !dbg !46
  %1181 = bitcast <4 x i32> %1180 to <8 x half>, !dbg !46
  %address.local942 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local896, !dbg !46
  %1182 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #11, !dbg !46
  %1183 = bitcast <4 x i32> %1182 to <8 x half>, !dbg !46
  %address.local943 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local904, !dbg !46
  %1184 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #11, !dbg !46
  %1185 = bitcast <4 x i32> %1184 to <8 x half>, !dbg !46
  %address.local944 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local912, !dbg !46
  %1186 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local944, i32 0) #11, !dbg !46
  %1187 = bitcast <4 x i32> %1186 to <8 x half>, !dbg !46
  %address.local945 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local920, !dbg !46
  %1188 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !46
  %1189 = bitcast <4 x i32> %1188 to <8 x half>, !dbg !46
  %address.local946 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local928, !dbg !46
  %1190 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #11, !dbg !46
  %1191 = bitcast <4 x i32> %1190 to <8 x half>, !dbg !46
  %address.local947 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local936, !dbg !46
  %1192 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #11, !dbg !46
  %1193 = shufflevector <8 x half> %1179, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1193, <4 x float> %1166) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1195 = shufflevector <8 x half> %1179, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1195, <4 x float> %1168) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1193, <4 x float> %1171) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1195, <4 x float> %1173) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1199 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1199, <4 x float> %1194) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1201 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1201, <4 x float> %1196) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1199, <4 x float> %1197) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1201, <4 x float> %1198) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1205 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1205, <4 x float> %1200) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1207 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1207, <4 x float> %1202) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1205, <4 x float> %1203) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1207, <4 x float> %1204) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1211 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1211, <4 x float> %1206) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1213 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1213, <4 x float> %1208) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1211, <4 x float> %1209) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1213, <4 x float> %1210) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1217 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1217, <4 x float> %1212) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1219 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1219, <4 x float> %1214) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1217, <4 x float> %1215) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1219, <4 x float> %1216) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1223 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1223, <4 x float> %1218) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1225 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1225, <4 x float> %1220) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1223, <4 x float> %1221) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1225, <4 x float> %1222) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1229 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1229, <4 x float> %1224) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1231 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1231, <4 x float> %1226) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1229, <4 x float> %1227) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1231, <4 x float> %1228) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1235 = bitcast <4 x i32> %1192 to <8 x half>, !dbg !46
  %1236 = shufflevector <8 x half> %1235, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1236, <4 x float> %1230) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1238 = shufflevector <8 x half> %1235, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1238, <4 x float> %1232) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1236, <4 x float> %1233) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1238, <4 x float> %1234) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %1242 = extractvalue [2 x <8 x float>] %155, 0, !dbg !44
  %1243 = shufflevector <8 x float> %1242, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1244 = fmul <4 x float> %1243, %782, !dbg !44
  %1245 = shufflevector <8 x float> %1242, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1246 = fmul <4 x float> %1245, %782, !dbg !44
  %1247 = extractvalue [2 x <8 x float>] %155, 1, !dbg !44
  %1248 = shufflevector <8 x float> %1247, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1249 = fmul <4 x float> %1248, %790, !dbg !44
  %1250 = shufflevector <8 x float> %1247, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1251 = fmul <4 x float> %1250, %790, !dbg !44
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1252 = or disjoint i64 %425, 96, !dbg !17
  %1253 = add i64 %1252, %424, !dbg !17
  %.tr1434 = trunc i64 %1253 to i32, !dbg !17
  %dma.buffer.offset1310 = shl i32 %.tr1434, 1, !dbg !17
  %1254 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %dma.buffer.offset1310, <4 x i32> %dma.buffer.rsrc1323) #9, !dbg !17
  %dma.buffer.offset = or i32 %.tr1424, 16576, !dbg !17
  %1255 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1323) #9, !dbg !17
  %1256 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local881, i32 0) #11, !dbg !46
  %1257 = bitcast <4 x i32> %1256 to <8 x half>, !dbg !46
  %1258 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local889, i32 0) #11, !dbg !46
  %1259 = bitcast <4 x i32> %1258 to <8 x half>, !dbg !46
  %1260 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local897, i32 0) #11, !dbg !46
  %1261 = bitcast <4 x i32> %1260 to <8 x half>, !dbg !46
  %1262 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local905, i32 0) #11, !dbg !46
  %1263 = bitcast <4 x i32> %1262 to <8 x half>, !dbg !46
  %1264 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local913, i32 0) #11, !dbg !46
  %1265 = bitcast <4 x i32> %1264 to <8 x half>, !dbg !46
  %1266 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local921, i32 0) #11, !dbg !46
  %1267 = bitcast <4 x i32> %1266 to <8 x half>, !dbg !46
  %1268 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #11, !dbg !46
  %1269 = bitcast <4 x i32> %1268 to <8 x half>, !dbg !46
  %1270 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #11, !dbg !46
  %1271 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1271, <4 x float> %1244) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1273 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1273, <4 x float> %1246) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1271, <4 x float> %1249) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1273, <4 x float> %1251) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1277 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1277, <4 x float> %1272) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1279 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1279, <4 x float> %1274) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1277, <4 x float> %1275) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1279, <4 x float> %1276) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1283 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1283, <4 x float> %1278) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1285 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1285, <4 x float> %1280) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1283, <4 x float> %1281) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1285, <4 x float> %1282) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1289 = shufflevector <8 x half> %1263, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1289, <4 x float> %1284) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1291 = shufflevector <8 x half> %1263, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1291, <4 x float> %1286) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1289, <4 x float> %1287) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1291, <4 x float> %1288) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1295 = shufflevector <8 x half> %1265, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1295, <4 x float> %1290) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1297 = shufflevector <8 x half> %1265, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1297, <4 x float> %1292) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1295, <4 x float> %1293) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1297, <4 x float> %1294) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1301 = shufflevector <8 x half> %1267, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1301, <4 x float> %1296) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1303 = shufflevector <8 x half> %1267, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1303, <4 x float> %1298) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1301, <4 x float> %1299) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1303, <4 x float> %1300) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1307 = shufflevector <8 x half> %1269, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1307, <4 x float> %1302) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1309 = shufflevector <8 x half> %1269, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1309, <4 x float> %1304) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1307, <4 x float> %1305) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1309, <4 x float> %1306) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1313 = bitcast <4 x i32> %1270 to <8 x half>, !dbg !46
  %1314 = shufflevector <8 x half> %1313, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1314, <4 x float> %1308) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1316 = shufflevector <8 x half> %1313, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1316, <4 x float> %1310) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1314, <4 x float> %1311) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1316, <4 x float> %1312) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %1320 = extractvalue [2 x <8 x float>] %154, 0, !dbg !44
  %1321 = shufflevector <8 x float> %1320, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1322 = fmul <4 x float> %1321, %782, !dbg !44
  %1323 = shufflevector <8 x float> %1320, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1324 = fmul <4 x float> %1323, %782, !dbg !44
  %1325 = extractvalue [2 x <8 x float>] %154, 1, !dbg !44
  %1326 = shufflevector <8 x float> %1325, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1327 = fmul <4 x float> %1326, %790, !dbg !44
  %1328 = shufflevector <8 x float> %1325, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1329 = fmul <4 x float> %1328, %790, !dbg !44
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1330 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local940, i32 0) #11, !dbg !46
  %1331 = bitcast <4 x i32> %1330 to <8 x half>, !dbg !46
  %1332 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local941, i32 0) #11, !dbg !46
  %1333 = bitcast <4 x i32> %1332 to <8 x half>, !dbg !46
  %1334 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #11, !dbg !46
  %1335 = bitcast <4 x i32> %1334 to <8 x half>, !dbg !46
  %1336 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #11, !dbg !46
  %1337 = bitcast <4 x i32> %1336 to <8 x half>, !dbg !46
  %1338 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local944, i32 0) #11, !dbg !46
  %1339 = bitcast <4 x i32> %1338 to <8 x half>, !dbg !46
  %1340 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !46
  %1341 = bitcast <4 x i32> %1340 to <8 x half>, !dbg !46
  %1342 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #11, !dbg !46
  %1343 = bitcast <4 x i32> %1342 to <8 x half>, !dbg !46
  %1344 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #11, !dbg !46
  %1345 = shufflevector <8 x half> %1331, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1345, <4 x float> %1322) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1347 = shufflevector <8 x half> %1331, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1347, <4 x float> %1324) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1345, <4 x float> %1327) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1347, <4 x float> %1329) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1351 = shufflevector <8 x half> %1333, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1351, <4 x float> %1346) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1353 = shufflevector <8 x half> %1333, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1353, <4 x float> %1348) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1351, <4 x float> %1349) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1353, <4 x float> %1350) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1357 = shufflevector <8 x half> %1335, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1357, <4 x float> %1352) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1359 = shufflevector <8 x half> %1335, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1359, <4 x float> %1354) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1357, <4 x float> %1355) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1359, <4 x float> %1356) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1363 = shufflevector <8 x half> %1337, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1363, <4 x float> %1358) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1365 = shufflevector <8 x half> %1337, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1365, <4 x float> %1360) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1363, <4 x float> %1361) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1365, <4 x float> %1362) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1369 = shufflevector <8 x half> %1339, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1369, <4 x float> %1364) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1371 = shufflevector <8 x half> %1339, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1371, <4 x float> %1366) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1369, <4 x float> %1367) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1371, <4 x float> %1368) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1375 = shufflevector <8 x half> %1341, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1376 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1375, <4 x float> %1370) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1377 = shufflevector <8 x half> %1341, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1377, <4 x float> %1372) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1375, <4 x float> %1373) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1377, <4 x float> %1374) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1381 = shufflevector <8 x half> %1343, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1381, <4 x float> %1376) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1383 = shufflevector <8 x half> %1343, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1383, <4 x float> %1378) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1381, <4 x float> %1379) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1383, <4 x float> %1380) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1387 = bitcast <4 x i32> %1344 to <8 x half>, !dbg !46
  %1388 = shufflevector <8 x half> %1387, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1388, <4 x float> %1382) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1390 = shufflevector <8 x half> %1387, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1390, <4 x float> %1384) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1388, <4 x float> %1385) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %1393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1390, <4 x float> %1386) #8, !dbg !46, !frisk.pv_interleave !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %1394 = shufflevector <4 x float> %1135, <4 x float> %1137, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1395 = insertvalue [2 x <8 x float>] poison, <8 x float> %1394, 0, !dbg !12
  %1396 = shufflevector <4 x float> %1162, <4 x float> %1163, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1397 = insertvalue [2 x <8 x float>] %1395, <8 x float> %1396, 1, !dbg !12
  %1398 = insertvalue [2 x <1 x float>] poison, <1 x float> %756, 0, !dbg !12
  %1399 = insertvalue [2 x <1 x float>] %1398, <1 x float> %759, 1, !dbg !12
  %1400 = shufflevector <4 x float> %1237, <4 x float> %1239, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1401 = insertvalue [2 x <8 x float>] poison, <8 x float> %1400, 0, !dbg !12
  %1402 = shufflevector <4 x float> %1240, <4 x float> %1241, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1403 = insertvalue [2 x <8 x float>] %1401, <8 x float> %1402, 1, !dbg !12
  %1404 = shufflevector <4 x float> %1315, <4 x float> %1317, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1405 = insertvalue [2 x <8 x float>] poison, <8 x float> %1404, 0, !dbg !12
  %1406 = shufflevector <4 x float> %1318, <4 x float> %1319, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1407 = insertvalue [2 x <8 x float>] %1405, <8 x float> %1406, 1, !dbg !12
  %1408 = shufflevector <4 x float> %1389, <4 x float> %1391, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1409 = insertvalue [2 x <8 x float>] poison, <8 x float> %1408, 0, !dbg !12
  %1410 = shufflevector <4 x float> %1392, <4 x float> %1393, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1411 = insertvalue [2 x <8 x float>] %1409, <8 x float> %1410, 1, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %1412 = add nuw nsw i64 %160, 1, !dbg !12
  %1413 = icmp slt i64 %1412, %114, !dbg !12
  br i1 %1413, label %153, label %._crit_edge, !dbg !12

._crit_edge:                                      ; preds = %153, %4
  %.lcssa7 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1397, %153 ], !dbg !12
  %.lcssa6 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %1159, %153 ], !dbg !12
  %.lcssa5 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1403, %153 ], !dbg !12
  %.lcssa4 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1407, %153 ], !dbg !12
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1411, %153 ], !dbg !12
  %1414 = extractvalue [2 x <1 x float>] %.lcssa6, 1, !dbg !12
  %bc = bitcast <1 x float> %1414 to <1 x i32>, !dbg !12
  %1415 = extractelement <1 x i32> %bc, i64 0, !dbg !12
  %1416 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1415), !dbg !12
  %1417 = extractvalue [2 x <1 x float>] %.lcssa6, 0, !dbg !12
  %bc47 = bitcast <1 x float> %1417 to <1 x i32>, !dbg !12
  %1418 = extractelement <1 x i32> %bc47, i64 0, !dbg !12
  %1419 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1418), !dbg !12
  %1420 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %1420, 0, !dbg !12
  %1421 = and i32 %kdma.tid, 32, !dbg !12
  %.not40 = icmp eq i32 %1421, 0, !dbg !12
  %1422 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !12
  %1423 = extractelement <8 x float> %1422, i64 0, !dbg !12
  %1424 = extractelement <8 x float> %1422, i64 1, !dbg !12
  %1425 = extractelement <8 x float> %1422, i64 2, !dbg !12
  %1426 = extractelement <8 x float> %1422, i64 3, !dbg !12
  %1427 = select i1 %.not, float %1424, float %1423, !dbg !12
  %1428 = bitcast float %1427 to i32, !dbg !12
  %1429 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1428), !dbg !12
  %1430 = bitcast i32 %1429 to float, !dbg !12
  %1431 = select i1 %.not, float %1423, float %1430, !dbg !12
  %1432 = select i1 %.not, float %1430, float %1424, !dbg !12
  %1433 = select i1 %.not, float %1426, float %1425, !dbg !12
  %1434 = bitcast float %1433 to i32, !dbg !12
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1434), !dbg !12
  %1436 = bitcast i32 %1435 to float, !dbg !12
  %1437 = select i1 %.not, float %1425, float %1436, !dbg !12
  %1438 = select i1 %.not, float %1436, float %1426, !dbg !12
  %1439 = select i1 %.not40, float %1437, float %1431, !dbg !12
  %1440 = bitcast float %1439 to i32, !dbg !12
  %1441 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1440), !dbg !12
  %1442 = bitcast i32 %1441 to float, !dbg !12
  %1443 = select i1 %.not40, float %1438, float %1432, !dbg !12
  %1444 = bitcast float %1443 to i32, !dbg !12
  %1445 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1444), !dbg !12
  %1446 = bitcast i32 %1445 to float, !dbg !12
  %1447 = insertelement <4 x i1> poison, i1 %.not40, i64 0, !dbg !12
  %1448 = shufflevector <4 x i1> %1447, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %1449 = insertelement <4 x float> poison, float %1431, i64 0, !dbg !12
  %1450 = insertelement <4 x float> %1449, float %1432, i64 1, !dbg !12
  %1451 = insertelement <4 x float> %1450, float %1442, i64 2, !dbg !12
  %1452 = insertelement <4 x float> %1451, float %1446, i64 3, !dbg !12
  %1453 = insertelement <4 x float> poison, float %1442, i64 0, !dbg !12
  %1454 = insertelement <4 x float> %1453, float %1446, i64 1, !dbg !12
  %1455 = insertelement <4 x float> %1454, float %1437, i64 2, !dbg !12
  %1456 = insertelement <4 x float> %1455, float %1438, i64 3, !dbg !12
  %1457 = select <4 x i1> %1448, <4 x float> %1452, <4 x float> %1456, !dbg !12
  %1458 = extractelement <8 x float> %1422, i64 4, !dbg !12
  %1459 = extractelement <8 x float> %1422, i64 5, !dbg !12
  %1460 = extractelement <8 x float> %1422, i64 6, !dbg !12
  %1461 = extractelement <8 x float> %1422, i64 7, !dbg !12
  %1462 = select i1 %.not, float %1459, float %1458, !dbg !12
  %1463 = bitcast float %1462 to i32, !dbg !12
  %1464 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1463), !dbg !12
  %1465 = bitcast i32 %1464 to float, !dbg !12
  %1466 = select i1 %.not, float %1458, float %1465, !dbg !12
  %1467 = select i1 %.not, float %1465, float %1459, !dbg !12
  %1468 = select i1 %.not, float %1461, float %1460, !dbg !12
  %1469 = bitcast float %1468 to i32, !dbg !12
  %1470 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1469), !dbg !12
  %1471 = bitcast i32 %1470 to float, !dbg !12
  %1472 = select i1 %.not, float %1460, float %1471, !dbg !12
  %1473 = select i1 %.not, float %1471, float %1461, !dbg !12
  %1474 = select i1 %.not40, float %1472, float %1466, !dbg !12
  %1475 = bitcast float %1474 to i32, !dbg !12
  %1476 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1475), !dbg !12
  %1477 = bitcast i32 %1476 to float, !dbg !12
  %1478 = select i1 %.not40, float %1473, float %1467, !dbg !12
  %1479 = bitcast float %1478 to i32, !dbg !12
  %1480 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1479), !dbg !12
  %1481 = bitcast i32 %1480 to float, !dbg !12
  %1482 = insertelement <4 x float> poison, float %1466, i64 0, !dbg !12
  %1483 = insertelement <4 x float> %1482, float %1467, i64 1, !dbg !12
  %1484 = insertelement <4 x float> %1483, float %1477, i64 2, !dbg !12
  %1485 = insertelement <4 x float> %1484, float %1481, i64 3, !dbg !12
  %1486 = insertelement <4 x float> poison, float %1477, i64 0, !dbg !12
  %1487 = insertelement <4 x float> %1486, float %1481, i64 1, !dbg !12
  %1488 = insertelement <4 x float> %1487, float %1472, i64 2, !dbg !12
  %1489 = insertelement <4 x float> %1488, float %1473, i64 3, !dbg !12
  %1490 = select <4 x i1> %1448, <4 x float> %1485, <4 x float> %1489, !dbg !12
  %1491 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !12
  %1492 = extractelement <8 x float> %1491, i64 0, !dbg !12
  %1493 = extractelement <8 x float> %1491, i64 1, !dbg !12
  %1494 = extractelement <8 x float> %1491, i64 2, !dbg !12
  %1495 = extractelement <8 x float> %1491, i64 3, !dbg !12
  %1496 = select i1 %.not, float %1493, float %1492, !dbg !12
  %1497 = bitcast float %1496 to i32, !dbg !12
  %1498 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1497), !dbg !12
  %1499 = bitcast i32 %1498 to float, !dbg !12
  %1500 = select i1 %.not, float %1492, float %1499, !dbg !12
  %1501 = select i1 %.not, float %1499, float %1493, !dbg !12
  %1502 = select i1 %.not, float %1495, float %1494, !dbg !12
  %1503 = bitcast float %1502 to i32, !dbg !12
  %1504 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1503), !dbg !12
  %1505 = bitcast i32 %1504 to float, !dbg !12
  %1506 = select i1 %.not, float %1494, float %1505, !dbg !12
  %1507 = select i1 %.not, float %1505, float %1495, !dbg !12
  %1508 = select i1 %.not40, float %1506, float %1500, !dbg !12
  %1509 = bitcast float %1508 to i32, !dbg !12
  %1510 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1509), !dbg !12
  %1511 = bitcast i32 %1510 to float, !dbg !12
  %1512 = select i1 %.not40, float %1507, float %1501, !dbg !12
  %1513 = bitcast float %1512 to i32, !dbg !12
  %1514 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1513), !dbg !12
  %1515 = bitcast i32 %1514 to float, !dbg !12
  %1516 = insertelement <4 x float> poison, float %1500, i64 0, !dbg !12
  %1517 = insertelement <4 x float> %1516, float %1501, i64 1, !dbg !12
  %1518 = insertelement <4 x float> %1517, float %1511, i64 2, !dbg !12
  %1519 = insertelement <4 x float> %1518, float %1515, i64 3, !dbg !12
  %1520 = insertelement <4 x float> poison, float %1511, i64 0, !dbg !12
  %1521 = insertelement <4 x float> %1520, float %1515, i64 1, !dbg !12
  %1522 = insertelement <4 x float> %1521, float %1506, i64 2, !dbg !12
  %1523 = insertelement <4 x float> %1522, float %1507, i64 3, !dbg !12
  %1524 = select <4 x i1> %1448, <4 x float> %1519, <4 x float> %1523, !dbg !12
  %1525 = extractelement <8 x float> %1491, i64 4, !dbg !12
  %1526 = extractelement <8 x float> %1491, i64 5, !dbg !12
  %1527 = extractelement <8 x float> %1491, i64 6, !dbg !12
  %1528 = extractelement <8 x float> %1491, i64 7, !dbg !12
  %1529 = select i1 %.not, float %1526, float %1525, !dbg !12
  %1530 = bitcast float %1529 to i32, !dbg !12
  %1531 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1530), !dbg !12
  %1532 = bitcast i32 %1531 to float, !dbg !12
  %1533 = select i1 %.not, float %1525, float %1532, !dbg !12
  %1534 = select i1 %.not, float %1532, float %1526, !dbg !12
  %1535 = select i1 %.not, float %1528, float %1527, !dbg !12
  %1536 = bitcast float %1535 to i32, !dbg !12
  %1537 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1536), !dbg !12
  %1538 = bitcast i32 %1537 to float, !dbg !12
  %1539 = select i1 %.not, float %1527, float %1538, !dbg !12
  %1540 = select i1 %.not, float %1538, float %1528, !dbg !12
  %1541 = select i1 %.not40, float %1539, float %1533, !dbg !12
  %1542 = bitcast float %1541 to i32, !dbg !12
  %1543 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1542), !dbg !12
  %1544 = bitcast i32 %1543 to float, !dbg !12
  %1545 = select i1 %.not40, float %1540, float %1534, !dbg !12
  %1546 = bitcast float %1545 to i32, !dbg !12
  %1547 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1546), !dbg !12
  %1548 = bitcast i32 %1547 to float, !dbg !12
  %1549 = insertelement <4 x float> poison, float %1533, i64 0, !dbg !12
  %1550 = insertelement <4 x float> %1549, float %1534, i64 1, !dbg !12
  %1551 = insertelement <4 x float> %1550, float %1544, i64 2, !dbg !12
  %1552 = insertelement <4 x float> %1551, float %1548, i64 3, !dbg !12
  %1553 = insertelement <4 x float> poison, float %1544, i64 0, !dbg !12
  %1554 = insertelement <4 x float> %1553, float %1548, i64 1, !dbg !12
  %1555 = insertelement <4 x float> %1554, float %1539, i64 2, !dbg !12
  %1556 = insertelement <4 x float> %1555, float %1540, i64 3, !dbg !12
  %1557 = select <4 x i1> %1448, <4 x float> %1552, <4 x float> %1556, !dbg !12
  %1558 = extractvalue [2 x <8 x float>] %.lcssa4, 0, !dbg !12
  %1559 = extractelement <8 x float> %1558, i64 0, !dbg !12
  %1560 = extractelement <8 x float> %1558, i64 1, !dbg !12
  %1561 = extractelement <8 x float> %1558, i64 2, !dbg !12
  %1562 = extractelement <8 x float> %1558, i64 3, !dbg !12
  %1563 = select i1 %.not, float %1560, float %1559, !dbg !12
  %1564 = bitcast float %1563 to i32, !dbg !12
  %1565 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1564), !dbg !12
  %1566 = bitcast i32 %1565 to float, !dbg !12
  %1567 = select i1 %.not, float %1559, float %1566, !dbg !12
  %1568 = select i1 %.not, float %1566, float %1560, !dbg !12
  %1569 = select i1 %.not, float %1562, float %1561, !dbg !12
  %1570 = bitcast float %1569 to i32, !dbg !12
  %1571 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1570), !dbg !12
  %1572 = bitcast i32 %1571 to float, !dbg !12
  %1573 = select i1 %.not, float %1561, float %1572, !dbg !12
  %1574 = select i1 %.not, float %1572, float %1562, !dbg !12
  %1575 = select i1 %.not40, float %1573, float %1567, !dbg !12
  %1576 = bitcast float %1575 to i32, !dbg !12
  %1577 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1576), !dbg !12
  %1578 = bitcast i32 %1577 to float, !dbg !12
  %1579 = select i1 %.not40, float %1574, float %1568, !dbg !12
  %1580 = bitcast float %1579 to i32, !dbg !12
  %1581 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1580), !dbg !12
  %1582 = bitcast i32 %1581 to float, !dbg !12
  %1583 = insertelement <4 x float> poison, float %1567, i64 0, !dbg !12
  %1584 = insertelement <4 x float> %1583, float %1568, i64 1, !dbg !12
  %1585 = insertelement <4 x float> %1584, float %1578, i64 2, !dbg !12
  %1586 = insertelement <4 x float> %1585, float %1582, i64 3, !dbg !12
  %1587 = insertelement <4 x float> poison, float %1578, i64 0, !dbg !12
  %1588 = insertelement <4 x float> %1587, float %1582, i64 1, !dbg !12
  %1589 = insertelement <4 x float> %1588, float %1573, i64 2, !dbg !12
  %1590 = insertelement <4 x float> %1589, float %1574, i64 3, !dbg !12
  %1591 = select <4 x i1> %1448, <4 x float> %1586, <4 x float> %1590, !dbg !12
  %1592 = extractelement <8 x float> %1558, i64 4, !dbg !12
  %1593 = extractelement <8 x float> %1558, i64 5, !dbg !12
  %1594 = extractelement <8 x float> %1558, i64 6, !dbg !12
  %1595 = extractelement <8 x float> %1558, i64 7, !dbg !12
  %1596 = select i1 %.not, float %1593, float %1592, !dbg !12
  %1597 = bitcast float %1596 to i32, !dbg !12
  %1598 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1597), !dbg !12
  %1599 = bitcast i32 %1598 to float, !dbg !12
  %1600 = select i1 %.not, float %1592, float %1599, !dbg !12
  %1601 = select i1 %.not, float %1599, float %1593, !dbg !12
  %1602 = select i1 %.not, float %1595, float %1594, !dbg !12
  %1603 = bitcast float %1602 to i32, !dbg !12
  %1604 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1603), !dbg !12
  %1605 = bitcast i32 %1604 to float, !dbg !12
  %1606 = select i1 %.not, float %1594, float %1605, !dbg !12
  %1607 = select i1 %.not, float %1605, float %1595, !dbg !12
  %1608 = select i1 %.not40, float %1606, float %1600, !dbg !12
  %1609 = bitcast float %1608 to i32, !dbg !12
  %1610 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1609), !dbg !12
  %1611 = bitcast i32 %1610 to float, !dbg !12
  %1612 = select i1 %.not40, float %1607, float %1601, !dbg !12
  %1613 = bitcast float %1612 to i32, !dbg !12
  %1614 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1613), !dbg !12
  %1615 = bitcast i32 %1614 to float, !dbg !12
  %1616 = insertelement <4 x float> poison, float %1600, i64 0, !dbg !12
  %1617 = insertelement <4 x float> %1616, float %1601, i64 1, !dbg !12
  %1618 = insertelement <4 x float> %1617, float %1611, i64 2, !dbg !12
  %1619 = insertelement <4 x float> %1618, float %1615, i64 3, !dbg !12
  %1620 = insertelement <4 x float> poison, float %1611, i64 0, !dbg !12
  %1621 = insertelement <4 x float> %1620, float %1615, i64 1, !dbg !12
  %1622 = insertelement <4 x float> %1621, float %1606, i64 2, !dbg !12
  %1623 = insertelement <4 x float> %1622, float %1607, i64 3, !dbg !12
  %1624 = select <4 x i1> %1448, <4 x float> %1619, <4 x float> %1623, !dbg !12
  %1625 = extractvalue [2 x <8 x float>] %.lcssa4, 1, !dbg !12
  %1626 = extractelement <8 x float> %1625, i64 0, !dbg !12
  %1627 = extractelement <8 x float> %1625, i64 1, !dbg !12
  %1628 = extractelement <8 x float> %1625, i64 2, !dbg !12
  %1629 = extractelement <8 x float> %1625, i64 3, !dbg !12
  %1630 = select i1 %.not, float %1627, float %1626, !dbg !12
  %1631 = bitcast float %1630 to i32, !dbg !12
  %1632 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1631), !dbg !12
  %1633 = bitcast i32 %1632 to float, !dbg !12
  %1634 = select i1 %.not, float %1626, float %1633, !dbg !12
  %1635 = select i1 %.not, float %1633, float %1627, !dbg !12
  %1636 = select i1 %.not, float %1629, float %1628, !dbg !12
  %1637 = bitcast float %1636 to i32, !dbg !12
  %1638 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1637), !dbg !12
  %1639 = bitcast i32 %1638 to float, !dbg !12
  %1640 = select i1 %.not, float %1628, float %1639, !dbg !12
  %1641 = select i1 %.not, float %1639, float %1629, !dbg !12
  %1642 = select i1 %.not40, float %1640, float %1634, !dbg !12
  %1643 = bitcast float %1642 to i32, !dbg !12
  %1644 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1643), !dbg !12
  %1645 = bitcast i32 %1644 to float, !dbg !12
  %1646 = select i1 %.not40, float %1641, float %1635, !dbg !12
  %1647 = bitcast float %1646 to i32, !dbg !12
  %1648 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1647), !dbg !12
  %1649 = bitcast i32 %1648 to float, !dbg !12
  %1650 = insertelement <4 x float> poison, float %1634, i64 0, !dbg !12
  %1651 = insertelement <4 x float> %1650, float %1635, i64 1, !dbg !12
  %1652 = insertelement <4 x float> %1651, float %1645, i64 2, !dbg !12
  %1653 = insertelement <4 x float> %1652, float %1649, i64 3, !dbg !12
  %1654 = insertelement <4 x float> poison, float %1645, i64 0, !dbg !12
  %1655 = insertelement <4 x float> %1654, float %1649, i64 1, !dbg !12
  %1656 = insertelement <4 x float> %1655, float %1640, i64 2, !dbg !12
  %1657 = insertelement <4 x float> %1656, float %1641, i64 3, !dbg !12
  %1658 = select <4 x i1> %1448, <4 x float> %1653, <4 x float> %1657, !dbg !12
  %1659 = extractelement <8 x float> %1625, i64 4, !dbg !12
  %1660 = extractelement <8 x float> %1625, i64 5, !dbg !12
  %1661 = extractelement <8 x float> %1625, i64 6, !dbg !12
  %1662 = extractelement <8 x float> %1625, i64 7, !dbg !12
  %1663 = select i1 %.not, float %1660, float %1659, !dbg !12
  %1664 = bitcast float %1663 to i32, !dbg !12
  %1665 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1664), !dbg !12
  %1666 = bitcast i32 %1665 to float, !dbg !12
  %1667 = select i1 %.not, float %1659, float %1666, !dbg !12
  %1668 = select i1 %.not, float %1666, float %1660, !dbg !12
  %1669 = select i1 %.not, float %1662, float %1661, !dbg !12
  %1670 = bitcast float %1669 to i32, !dbg !12
  %1671 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1670), !dbg !12
  %1672 = bitcast i32 %1671 to float, !dbg !12
  %1673 = select i1 %.not, float %1661, float %1672, !dbg !12
  %1674 = select i1 %.not, float %1672, float %1662, !dbg !12
  %1675 = select i1 %.not40, float %1673, float %1667, !dbg !12
  %1676 = bitcast float %1675 to i32, !dbg !12
  %1677 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1676), !dbg !12
  %1678 = bitcast i32 %1677 to float, !dbg !12
  %1679 = select i1 %.not40, float %1674, float %1668, !dbg !12
  %1680 = bitcast float %1679 to i32, !dbg !12
  %1681 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1680), !dbg !12
  %1682 = bitcast i32 %1681 to float, !dbg !12
  %1683 = insertelement <4 x float> poison, float %1667, i64 0, !dbg !12
  %1684 = insertelement <4 x float> %1683, float %1668, i64 1, !dbg !12
  %1685 = insertelement <4 x float> %1684, float %1678, i64 2, !dbg !12
  %1686 = insertelement <4 x float> %1685, float %1682, i64 3, !dbg !12
  %1687 = insertelement <4 x float> poison, float %1678, i64 0, !dbg !12
  %1688 = insertelement <4 x float> %1687, float %1682, i64 1, !dbg !12
  %1689 = insertelement <4 x float> %1688, float %1673, i64 2, !dbg !12
  %1690 = insertelement <4 x float> %1689, float %1674, i64 3, !dbg !12
  %1691 = select <4 x i1> %1448, <4 x float> %1686, <4 x float> %1690, !dbg !12
  %1692 = extractvalue [2 x <8 x float>] %.lcssa5, 0, !dbg !12
  %1693 = extractelement <8 x float> %1692, i64 0, !dbg !12
  %1694 = extractelement <8 x float> %1692, i64 1, !dbg !12
  %1695 = extractelement <8 x float> %1692, i64 2, !dbg !12
  %1696 = extractelement <8 x float> %1692, i64 3, !dbg !12
  %1697 = select i1 %.not, float %1694, float %1693, !dbg !12
  %1698 = bitcast float %1697 to i32, !dbg !12
  %1699 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1698), !dbg !12
  %1700 = bitcast i32 %1699 to float, !dbg !12
  %1701 = select i1 %.not, float %1693, float %1700, !dbg !12
  %1702 = select i1 %.not, float %1700, float %1694, !dbg !12
  %1703 = select i1 %.not, float %1696, float %1695, !dbg !12
  %1704 = bitcast float %1703 to i32, !dbg !12
  %1705 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1704), !dbg !12
  %1706 = bitcast i32 %1705 to float, !dbg !12
  %1707 = select i1 %.not, float %1695, float %1706, !dbg !12
  %1708 = select i1 %.not, float %1706, float %1696, !dbg !12
  %1709 = select i1 %.not40, float %1707, float %1701, !dbg !12
  %1710 = bitcast float %1709 to i32, !dbg !12
  %1711 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1710), !dbg !12
  %1712 = bitcast i32 %1711 to float, !dbg !12
  %1713 = select i1 %.not40, float %1708, float %1702, !dbg !12
  %1714 = bitcast float %1713 to i32, !dbg !12
  %1715 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1714), !dbg !12
  %1716 = bitcast i32 %1715 to float, !dbg !12
  %1717 = insertelement <4 x float> poison, float %1701, i64 0, !dbg !12
  %1718 = insertelement <4 x float> %1717, float %1702, i64 1, !dbg !12
  %1719 = insertelement <4 x float> %1718, float %1712, i64 2, !dbg !12
  %1720 = insertelement <4 x float> %1719, float %1716, i64 3, !dbg !12
  %1721 = insertelement <4 x float> poison, float %1712, i64 0, !dbg !12
  %1722 = insertelement <4 x float> %1721, float %1716, i64 1, !dbg !12
  %1723 = insertelement <4 x float> %1722, float %1707, i64 2, !dbg !12
  %1724 = insertelement <4 x float> %1723, float %1708, i64 3, !dbg !12
  %1725 = select <4 x i1> %1448, <4 x float> %1720, <4 x float> %1724, !dbg !12
  %1726 = extractelement <8 x float> %1692, i64 4, !dbg !12
  %1727 = extractelement <8 x float> %1692, i64 5, !dbg !12
  %1728 = extractelement <8 x float> %1692, i64 6, !dbg !12
  %1729 = extractelement <8 x float> %1692, i64 7, !dbg !12
  %1730 = select i1 %.not, float %1727, float %1726, !dbg !12
  %1731 = bitcast float %1730 to i32, !dbg !12
  %1732 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1731), !dbg !12
  %1733 = bitcast i32 %1732 to float, !dbg !12
  %1734 = select i1 %.not, float %1726, float %1733, !dbg !12
  %1735 = select i1 %.not, float %1733, float %1727, !dbg !12
  %1736 = select i1 %.not, float %1729, float %1728, !dbg !12
  %1737 = bitcast float %1736 to i32, !dbg !12
  %1738 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1737), !dbg !12
  %1739 = bitcast i32 %1738 to float, !dbg !12
  %1740 = select i1 %.not, float %1728, float %1739, !dbg !12
  %1741 = select i1 %.not, float %1739, float %1729, !dbg !12
  %1742 = select i1 %.not40, float %1740, float %1734, !dbg !12
  %1743 = bitcast float %1742 to i32, !dbg !12
  %1744 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1743), !dbg !12
  %1745 = bitcast i32 %1744 to float, !dbg !12
  %1746 = select i1 %.not40, float %1741, float %1735, !dbg !12
  %1747 = bitcast float %1746 to i32, !dbg !12
  %1748 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1747), !dbg !12
  %1749 = bitcast i32 %1748 to float, !dbg !12
  %1750 = insertelement <4 x float> poison, float %1734, i64 0, !dbg !12
  %1751 = insertelement <4 x float> %1750, float %1735, i64 1, !dbg !12
  %1752 = insertelement <4 x float> %1751, float %1745, i64 2, !dbg !12
  %1753 = insertelement <4 x float> %1752, float %1749, i64 3, !dbg !12
  %1754 = insertelement <4 x float> poison, float %1745, i64 0, !dbg !12
  %1755 = insertelement <4 x float> %1754, float %1749, i64 1, !dbg !12
  %1756 = insertelement <4 x float> %1755, float %1740, i64 2, !dbg !12
  %1757 = insertelement <4 x float> %1756, float %1741, i64 3, !dbg !12
  %1758 = select <4 x i1> %1448, <4 x float> %1753, <4 x float> %1757, !dbg !12
  %1759 = extractvalue [2 x <8 x float>] %.lcssa5, 1, !dbg !12
  %1760 = extractelement <8 x float> %1759, i64 0, !dbg !12
  %1761 = extractelement <8 x float> %1759, i64 1, !dbg !12
  %1762 = extractelement <8 x float> %1759, i64 2, !dbg !12
  %1763 = extractelement <8 x float> %1759, i64 3, !dbg !12
  %1764 = select i1 %.not, float %1761, float %1760, !dbg !12
  %1765 = bitcast float %1764 to i32, !dbg !12
  %1766 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1765), !dbg !12
  %1767 = bitcast i32 %1766 to float, !dbg !12
  %1768 = select i1 %.not, float %1760, float %1767, !dbg !12
  %1769 = select i1 %.not, float %1767, float %1761, !dbg !12
  %1770 = select i1 %.not, float %1763, float %1762, !dbg !12
  %1771 = bitcast float %1770 to i32, !dbg !12
  %1772 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1771), !dbg !12
  %1773 = bitcast i32 %1772 to float, !dbg !12
  %1774 = select i1 %.not, float %1762, float %1773, !dbg !12
  %1775 = select i1 %.not, float %1773, float %1763, !dbg !12
  %1776 = select i1 %.not40, float %1774, float %1768, !dbg !12
  %1777 = bitcast float %1776 to i32, !dbg !12
  %1778 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1777), !dbg !12
  %1779 = bitcast i32 %1778 to float, !dbg !12
  %1780 = select i1 %.not40, float %1775, float %1769, !dbg !12
  %1781 = bitcast float %1780 to i32, !dbg !12
  %1782 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1781), !dbg !12
  %1783 = bitcast i32 %1782 to float, !dbg !12
  %1784 = insertelement <4 x float> poison, float %1768, i64 0, !dbg !12
  %1785 = insertelement <4 x float> %1784, float %1769, i64 1, !dbg !12
  %1786 = insertelement <4 x float> %1785, float %1779, i64 2, !dbg !12
  %1787 = insertelement <4 x float> %1786, float %1783, i64 3, !dbg !12
  %1788 = insertelement <4 x float> poison, float %1779, i64 0, !dbg !12
  %1789 = insertelement <4 x float> %1788, float %1783, i64 1, !dbg !12
  %1790 = insertelement <4 x float> %1789, float %1774, i64 2, !dbg !12
  %1791 = insertelement <4 x float> %1790, float %1775, i64 3, !dbg !12
  %1792 = select <4 x i1> %1448, <4 x float> %1787, <4 x float> %1791, !dbg !12
  %1793 = extractelement <8 x float> %1759, i64 4, !dbg !12
  %1794 = extractelement <8 x float> %1759, i64 5, !dbg !12
  %1795 = extractelement <8 x float> %1759, i64 6, !dbg !12
  %1796 = extractelement <8 x float> %1759, i64 7, !dbg !12
  %1797 = select i1 %.not, float %1794, float %1793, !dbg !12
  %1798 = bitcast float %1797 to i32, !dbg !12
  %1799 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1798), !dbg !12
  %1800 = bitcast i32 %1799 to float, !dbg !12
  %1801 = select i1 %.not, float %1793, float %1800, !dbg !12
  %1802 = select i1 %.not, float %1800, float %1794, !dbg !12
  %1803 = select i1 %.not, float %1796, float %1795, !dbg !12
  %1804 = bitcast float %1803 to i32, !dbg !12
  %1805 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1804), !dbg !12
  %1806 = bitcast i32 %1805 to float, !dbg !12
  %1807 = select i1 %.not, float %1795, float %1806, !dbg !12
  %1808 = select i1 %.not, float %1806, float %1796, !dbg !12
  %1809 = select i1 %.not40, float %1807, float %1801, !dbg !12
  %1810 = bitcast float %1809 to i32, !dbg !12
  %1811 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1810), !dbg !12
  %1812 = bitcast i32 %1811 to float, !dbg !12
  %1813 = select i1 %.not40, float %1808, float %1802, !dbg !12
  %1814 = bitcast float %1813 to i32, !dbg !12
  %1815 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1814), !dbg !12
  %1816 = bitcast i32 %1815 to float, !dbg !12
  %1817 = insertelement <4 x float> poison, float %1801, i64 0, !dbg !12
  %1818 = insertelement <4 x float> %1817, float %1802, i64 1, !dbg !12
  %1819 = insertelement <4 x float> %1818, float %1812, i64 2, !dbg !12
  %1820 = insertelement <4 x float> %1819, float %1816, i64 3, !dbg !12
  %1821 = insertelement <4 x float> poison, float %1812, i64 0, !dbg !12
  %1822 = insertelement <4 x float> %1821, float %1816, i64 1, !dbg !12
  %1823 = insertelement <4 x float> %1822, float %1807, i64 2, !dbg !12
  %1824 = insertelement <4 x float> %1823, float %1808, i64 3, !dbg !12
  %1825 = select <4 x i1> %1448, <4 x float> %1820, <4 x float> %1824, !dbg !12
  %1826 = extractvalue [2 x <8 x float>] %.lcssa7, 0, !dbg !12
  %1827 = extractelement <8 x float> %1826, i64 0, !dbg !12
  %1828 = extractelement <8 x float> %1826, i64 1, !dbg !12
  %1829 = extractelement <8 x float> %1826, i64 2, !dbg !12
  %1830 = extractelement <8 x float> %1826, i64 3, !dbg !12
  %1831 = select i1 %.not, float %1828, float %1827, !dbg !12
  %1832 = bitcast float %1831 to i32, !dbg !12
  %1833 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1832), !dbg !12
  %1834 = bitcast i32 %1833 to float, !dbg !12
  %1835 = select i1 %.not, float %1827, float %1834, !dbg !12
  %1836 = select i1 %.not, float %1834, float %1828, !dbg !12
  %1837 = select i1 %.not, float %1830, float %1829, !dbg !12
  %1838 = bitcast float %1837 to i32, !dbg !12
  %1839 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1838), !dbg !12
  %1840 = bitcast i32 %1839 to float, !dbg !12
  %1841 = select i1 %.not, float %1829, float %1840, !dbg !12
  %1842 = select i1 %.not, float %1840, float %1830, !dbg !12
  %1843 = select i1 %.not40, float %1841, float %1835, !dbg !12
  %1844 = bitcast float %1843 to i32, !dbg !12
  %1845 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1844), !dbg !12
  %1846 = select i1 %.not40, float %1842, float %1836, !dbg !12
  %1847 = bitcast float %1846 to i32, !dbg !12
  %1848 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1847), !dbg !12
  %1849 = extractelement <8 x float> %1826, i64 4, !dbg !12
  %1850 = extractelement <8 x float> %1826, i64 5, !dbg !12
  %1851 = extractelement <8 x float> %1826, i64 6, !dbg !12
  %1852 = extractelement <8 x float> %1826, i64 7, !dbg !12
  %1853 = select i1 %.not, float %1850, float %1849, !dbg !12
  %1854 = bitcast float %1853 to i32, !dbg !12
  %1855 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1854), !dbg !12
  %1856 = bitcast i32 %1855 to float, !dbg !12
  %1857 = select i1 %.not, float %1849, float %1856, !dbg !12
  %1858 = select i1 %.not, float %1856, float %1850, !dbg !12
  %1859 = select i1 %.not, float %1852, float %1851, !dbg !12
  %1860 = bitcast float %1859 to i32, !dbg !12
  %1861 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1860), !dbg !12
  %1862 = bitcast i32 %1861 to float, !dbg !12
  %1863 = select i1 %.not, float %1851, float %1862, !dbg !12
  %1864 = select i1 %.not, float %1862, float %1852, !dbg !12
  %1865 = select i1 %.not40, float %1863, float %1857, !dbg !12
  %1866 = bitcast float %1865 to i32, !dbg !12
  %1867 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1866), !dbg !12
  %1868 = select i1 %.not40, float %1864, float %1858, !dbg !12
  %1869 = bitcast float %1868 to i32, !dbg !12
  %1870 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1869), !dbg !12
  %1871 = extractvalue [2 x <8 x float>] %.lcssa7, 1, !dbg !12
  %1872 = extractelement <8 x float> %1871, i64 0, !dbg !12
  %1873 = extractelement <8 x float> %1871, i64 1, !dbg !12
  %1874 = extractelement <8 x float> %1871, i64 2, !dbg !12
  %1875 = extractelement <8 x float> %1871, i64 3, !dbg !12
  %1876 = select i1 %.not, float %1873, float %1872, !dbg !12
  %1877 = bitcast float %1876 to i32, !dbg !12
  %1878 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1877), !dbg !12
  %1879 = bitcast i32 %1878 to float, !dbg !12
  %1880 = select i1 %.not, float %1872, float %1879, !dbg !12
  %1881 = select i1 %.not, float %1879, float %1873, !dbg !12
  %1882 = select i1 %.not, float %1875, float %1874, !dbg !12
  %1883 = bitcast float %1882 to i32, !dbg !12
  %1884 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1883), !dbg !12
  %1885 = bitcast i32 %1884 to float, !dbg !12
  %1886 = select i1 %.not, float %1874, float %1885, !dbg !12
  %1887 = select i1 %.not, float %1885, float %1875, !dbg !12
  %1888 = select i1 %.not40, float %1886, float %1880, !dbg !12
  %1889 = bitcast float %1888 to i32, !dbg !12
  %1890 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1889), !dbg !12
  %1891 = select i1 %.not40, float %1887, float %1881, !dbg !12
  %1892 = bitcast float %1891 to i32, !dbg !12
  %1893 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1892), !dbg !12
  %1894 = extractelement <8 x float> %1871, i64 4, !dbg !12
  %1895 = extractelement <8 x float> %1871, i64 5, !dbg !12
  %1896 = extractelement <8 x float> %1871, i64 6, !dbg !12
  %1897 = extractelement <8 x float> %1871, i64 7, !dbg !12
  %1898 = select i1 %.not, float %1895, float %1894, !dbg !12
  %1899 = bitcast float %1898 to i32, !dbg !12
  %1900 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1899), !dbg !12
  %1901 = bitcast i32 %1900 to float, !dbg !12
  %1902 = select i1 %.not, float %1894, float %1901, !dbg !12
  %1903 = select i1 %.not, float %1901, float %1895, !dbg !12
  %1904 = select i1 %.not, float %1897, float %1896, !dbg !12
  %1905 = bitcast float %1904 to i32, !dbg !12
  %1906 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1905), !dbg !12
  %1907 = bitcast i32 %1906 to float, !dbg !12
  %1908 = select i1 %.not, float %1896, float %1907, !dbg !12
  %1909 = select i1 %.not, float %1907, float %1897, !dbg !12
  %1910 = select i1 %.not40, float %1908, float %1902, !dbg !12
  %1911 = bitcast float %1910 to i32, !dbg !12
  %1912 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1911), !dbg !12
  %1913 = select i1 %.not40, float %1909, float %1903, !dbg !12
  %1914 = bitcast float %1913 to i32, !dbg !12
  %1915 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1914), !dbg !12
  %1916 = insertelement <2 x i32> poison, i32 %1419, i64 0, !dbg !12
  %1917 = insertelement <2 x i32> %1916, i32 %1416, i64 1, !dbg !12
  %1918 = bitcast <2 x i32> %1917 to <2 x float>, !dbg !12
  %1919 = shufflevector <1 x float> %1417, <1 x float> %1414, <2 x i32> <i32 0, i32 1>, !dbg !12
  %1920 = fadd <2 x float> %1919, %1918, !dbg !12
  %bc48 = bitcast <2 x float> %1920 to <2 x i32>, !dbg !12
  %1921 = extractelement <2 x i32> %bc48, i64 1, !dbg !12
  %1922 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1921), !dbg !12
  %bc49 = bitcast <2 x float> %1920 to <2 x i32>, !dbg !12
  %1923 = extractelement <2 x i32> %bc49, i64 0, !dbg !12
  %1924 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1923), !dbg !12
  %1925 = insertelement <2 x i32> poison, i32 %1924, i64 0, !dbg !12
  %1926 = insertelement <2 x i32> %1925, i32 %1922, i64 1, !dbg !12
  %1927 = bitcast <2 x i32> %1926 to <2 x float>, !dbg !12
  %1928 = fadd <2 x float> %1920, %1927, !dbg !12
  %1929 = fdiv arcp <2 x float> splat (float 1.000000e+00), %1928, !dbg !50
  %1930 = shufflevector <2 x float> %1929, <2 x float> poison, <4 x i32> zeroinitializer, !dbg !50
  %1931 = shufflevector <2 x float> %1929, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, !dbg !50
  %1932 = bitcast i32 %1912 to float, !dbg !12
  %1933 = bitcast i32 %1915 to float, !dbg !12
  %1934 = insertelement <4 x float> poison, float %1902, i64 0, !dbg !12
  %1935 = insertelement <4 x float> %1934, float %1903, i64 1, !dbg !12
  %1936 = insertelement <4 x float> %1935, float %1932, i64 2, !dbg !12
  %1937 = insertelement <4 x float> %1936, float %1933, i64 3, !dbg !12
  %1938 = insertelement <4 x float> poison, float %1932, i64 0, !dbg !12
  %1939 = insertelement <4 x float> %1938, float %1933, i64 1, !dbg !12
  %1940 = insertelement <4 x float> %1939, float %1908, i64 2, !dbg !12
  %1941 = insertelement <4 x float> %1940, float %1909, i64 3, !dbg !12
  %1942 = select <4 x i1> %1448, <4 x float> %1937, <4 x float> %1941, !dbg !12
  %1943 = fmul <4 x float> %1931, %1942, !dbg !50
  %1944 = fptrunc <4 x float> %1943 to <4 x half>, !dbg !51
  %1945 = extractelement <4 x half> %1944, i64 3, !dbg !51
  %1946 = extractelement <4 x half> %1944, i64 2, !dbg !51
  %1947 = extractelement <4 x half> %1944, i64 1, !dbg !51
  %1948 = extractelement <4 x half> %1944, i64 0, !dbg !51
  %1949 = bitcast i32 %1890 to float, !dbg !12
  %1950 = bitcast i32 %1893 to float, !dbg !12
  %1951 = insertelement <4 x float> poison, float %1880, i64 0, !dbg !12
  %1952 = insertelement <4 x float> %1951, float %1881, i64 1, !dbg !12
  %1953 = insertelement <4 x float> %1952, float %1949, i64 2, !dbg !12
  %1954 = insertelement <4 x float> %1953, float %1950, i64 3, !dbg !12
  %1955 = insertelement <4 x float> poison, float %1949, i64 0, !dbg !12
  %1956 = insertelement <4 x float> %1955, float %1950, i64 1, !dbg !12
  %1957 = insertelement <4 x float> %1956, float %1886, i64 2, !dbg !12
  %1958 = insertelement <4 x float> %1957, float %1887, i64 3, !dbg !12
  %1959 = select <4 x i1> %1448, <4 x float> %1954, <4 x float> %1958, !dbg !12
  %1960 = fmul <4 x float> %1931, %1959, !dbg !50
  %1961 = fptrunc <4 x float> %1960 to <4 x half>, !dbg !51
  %1962 = extractelement <4 x half> %1961, i64 3, !dbg !51
  %1963 = extractelement <4 x half> %1961, i64 2, !dbg !51
  %1964 = extractelement <4 x half> %1961, i64 1, !dbg !51
  %1965 = extractelement <4 x half> %1961, i64 0, !dbg !51
  %1966 = bitcast i32 %1867 to float, !dbg !12
  %1967 = bitcast i32 %1870 to float, !dbg !12
  %1968 = insertelement <4 x float> poison, float %1857, i64 0, !dbg !12
  %1969 = insertelement <4 x float> %1968, float %1858, i64 1, !dbg !12
  %1970 = insertelement <4 x float> %1969, float %1966, i64 2, !dbg !12
  %1971 = insertelement <4 x float> %1970, float %1967, i64 3, !dbg !12
  %1972 = insertelement <4 x float> poison, float %1966, i64 0, !dbg !12
  %1973 = insertelement <4 x float> %1972, float %1967, i64 1, !dbg !12
  %1974 = insertelement <4 x float> %1973, float %1863, i64 2, !dbg !12
  %1975 = insertelement <4 x float> %1974, float %1864, i64 3, !dbg !12
  %1976 = select <4 x i1> %1448, <4 x float> %1971, <4 x float> %1975, !dbg !12
  %1977 = fmul <4 x float> %1930, %1976, !dbg !50
  %1978 = fptrunc <4 x float> %1977 to <4 x half>, !dbg !51
  %1979 = extractelement <4 x half> %1978, i64 3, !dbg !51
  %1980 = extractelement <4 x half> %1978, i64 2, !dbg !51
  %1981 = extractelement <4 x half> %1978, i64 1, !dbg !51
  %1982 = extractelement <4 x half> %1978, i64 0, !dbg !51
  %1983 = bitcast i32 %1845 to float, !dbg !12
  %1984 = bitcast i32 %1848 to float, !dbg !12
  %1985 = insertelement <4 x float> poison, float %1835, i64 0, !dbg !12
  %1986 = insertelement <4 x float> %1985, float %1836, i64 1, !dbg !12
  %1987 = insertelement <4 x float> %1986, float %1983, i64 2, !dbg !12
  %1988 = insertelement <4 x float> %1987, float %1984, i64 3, !dbg !12
  %1989 = insertelement <4 x float> poison, float %1983, i64 0, !dbg !12
  %1990 = insertelement <4 x float> %1989, float %1984, i64 1, !dbg !12
  %1991 = insertelement <4 x float> %1990, float %1841, i64 2, !dbg !12
  %1992 = insertelement <4 x float> %1991, float %1842, i64 3, !dbg !12
  %1993 = select <4 x i1> %1448, <4 x float> %1988, <4 x float> %1992, !dbg !12
  %1994 = fmul <4 x float> %1930, %1993, !dbg !50
  %1995 = fptrunc <4 x float> %1994 to <4 x half>, !dbg !51
  %1996 = extractelement <4 x half> %1995, i64 3, !dbg !51
  %1997 = extractelement <4 x half> %1995, i64 2, !dbg !51
  %1998 = extractelement <4 x half> %1995, i64 1, !dbg !51
  %1999 = extractelement <4 x half> %1995, i64 0, !dbg !51
  %address.thread950 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !52, !range !10, !frisk.register_lifetimes !5
  %address.local951 = zext nneg i32 %address.thread950 to i64, !dbg !9
  %address.local953 = lshr i64 %address.local951, 2, !dbg !16
  %address.local954 = and i64 %address.local953, 48, !dbg !16
  %address.local956 = and i64 %address.local951, 240, !dbg !52
  %address.local955 = add i64 %43, %address.local951, !dbg !52
  %address.local958 = sub i64 %address.local955, %address.local956, !dbg !52
  %address.local959 = add i64 %address.local958, %address.local954, !dbg !52
  %2000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %55, !dbg !52
  %address.local962.idx = shl nuw nsw i64 %address.local959, 8, !dbg !52
  %address.local962 = getelementptr inbounds nuw i8, ptr addrspace(1) %2000, i64 %address.local962.idx, !dbg !52
  %address.local964 = and i64 %address.local953, 12, !dbg !11
  %address.local965 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local964, !dbg !52
  store half %1999, ptr addrspace(1) %address.local965, align 2, !dbg !52
  %address.local967 = or disjoint i64 %address.local964, 1, !dbg !52
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local967, !dbg !52
  store half %1998, ptr addrspace(1) %address.local968, align 2, !dbg !52
  %address.local970 = or disjoint i64 %address.local964, 2, !dbg !52
  %address.local971 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local970, !dbg !52
  store half %1997, ptr addrspace(1) %address.local971, align 2, !dbg !52
  %address.local973 = or disjoint i64 %address.local964, 3, !dbg !52
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local973, !dbg !52
  store half %1996, ptr addrspace(1) %address.local974, align 2, !dbg !52
  %address.local976 = or disjoint i64 %address.local964, 16, !dbg !52
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local976, !dbg !52
  store half %1982, ptr addrspace(1) %address.local977, align 2, !dbg !52
  %address.local979 = or disjoint i64 %address.local964, 17, !dbg !52
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local979, !dbg !52
  store half %1981, ptr addrspace(1) %address.local980, align 2, !dbg !52
  %address.local982 = or disjoint i64 %address.local964, 18, !dbg !52
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local982, !dbg !52
  store half %1980, ptr addrspace(1) %address.local983, align 2, !dbg !52
  %address.local985 = or disjoint i64 %address.local964, 19, !dbg !52
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local985, !dbg !52
  store half %1979, ptr addrspace(1) %address.local986, align 2, !dbg !52
  %address.local990 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local962, i64 16384, !dbg !52
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local964, !dbg !52
  store half %1965, ptr addrspace(1) %address.local991, align 2, !dbg !52
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local967, !dbg !52
  store half %1964, ptr addrspace(1) %address.local993, align 2, !dbg !52
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local970, !dbg !52
  store half %1963, ptr addrspace(1) %address.local995, align 2, !dbg !52
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local973, !dbg !52
  store half %1962, ptr addrspace(1) %address.local997, align 2, !dbg !52
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local976, !dbg !52
  store half %1948, ptr addrspace(1) %address.local999, align 2, !dbg !52
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local979, !dbg !52
  store half %1947, ptr addrspace(1) %address.local1001, align 2, !dbg !52
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local982, !dbg !52
  store half %1946, ptr addrspace(1) %address.local1003, align 2, !dbg !52
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local985, !dbg !52
  store half %1945, ptr addrspace(1) %address.local1005, align 2, !dbg !52
  %2001 = fmul <4 x float> %1931, %1825, !dbg !50
  %2002 = fptrunc <4 x float> %2001 to <4 x half>, !dbg !51
  %2003 = extractelement <4 x half> %2002, i64 3, !dbg !51
  %2004 = extractelement <4 x half> %2002, i64 2, !dbg !51
  %2005 = extractelement <4 x half> %2002, i64 1, !dbg !51
  %2006 = extractelement <4 x half> %2002, i64 0, !dbg !51
  %2007 = fmul <4 x float> %1931, %1792, !dbg !50
  %2008 = fptrunc <4 x float> %2007 to <4 x half>, !dbg !51
  %2009 = extractelement <4 x half> %2008, i64 3, !dbg !51
  %2010 = extractelement <4 x half> %2008, i64 2, !dbg !51
  %2011 = extractelement <4 x half> %2008, i64 1, !dbg !51
  %2012 = extractelement <4 x half> %2008, i64 0, !dbg !51
  %2013 = fmul <4 x float> %1930, %1758, !dbg !50
  %2014 = fptrunc <4 x float> %2013 to <4 x half>, !dbg !51
  %2015 = extractelement <4 x half> %2014, i64 3, !dbg !51
  %2016 = extractelement <4 x half> %2014, i64 2, !dbg !51
  %2017 = extractelement <4 x half> %2014, i64 1, !dbg !51
  %2018 = extractelement <4 x half> %2014, i64 0, !dbg !51
  %2019 = fmul <4 x float> %1930, %1725, !dbg !50
  %2020 = fptrunc <4 x float> %2019 to <4 x half>, !dbg !51
  %2021 = extractelement <4 x half> %2020, i64 3, !dbg !51
  %2022 = extractelement <4 x half> %2020, i64 2, !dbg !51
  %2023 = extractelement <4 x half> %2020, i64 1, !dbg !51
  %2024 = extractelement <4 x half> %2020, i64 0, !dbg !51
  %address.local1021 = or disjoint i64 %address.local964, 32, !dbg !52
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1021, !dbg !52
  store half %2024, ptr addrspace(1) %address.local1022, align 2, !dbg !52
  %address.local1024 = or disjoint i64 %address.local964, 33, !dbg !52
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1024, !dbg !52
  store half %2023, ptr addrspace(1) %address.local1025, align 2, !dbg !52
  %address.local1027 = or disjoint i64 %address.local964, 34, !dbg !52
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1027, !dbg !52
  store half %2022, ptr addrspace(1) %address.local1028, align 2, !dbg !52
  %address.local1030 = or disjoint i64 %address.local964, 35, !dbg !52
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1030, !dbg !52
  store half %2021, ptr addrspace(1) %address.local1031, align 2, !dbg !52
  %address.local1033 = or disjoint i64 %address.local964, 48, !dbg !52
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1033, !dbg !52
  store half %2018, ptr addrspace(1) %address.local1034, align 2, !dbg !52
  %address.local1036 = or disjoint i64 %address.local964, 49, !dbg !52
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1036, !dbg !52
  store half %2017, ptr addrspace(1) %address.local1037, align 2, !dbg !52
  %address.local1039 = or disjoint i64 %address.local964, 50, !dbg !52
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1039, !dbg !52
  store half %2016, ptr addrspace(1) %address.local1040, align 2, !dbg !52
  %address.local1042 = or i64 %address.local953, 51, !dbg !52
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1042, !dbg !52
  store half %2015, ptr addrspace(1) %address.local1043, align 2, !dbg !52
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1021, !dbg !52
  store half %2012, ptr addrspace(1) %address.local1048, align 2, !dbg !52
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1024, !dbg !52
  store half %2011, ptr addrspace(1) %address.local1050, align 2, !dbg !52
  %address.local1052 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1027, !dbg !52
  store half %2010, ptr addrspace(1) %address.local1052, align 2, !dbg !52
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1030, !dbg !52
  store half %2009, ptr addrspace(1) %address.local1054, align 2, !dbg !52
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1033, !dbg !52
  store half %2006, ptr addrspace(1) %address.local1056, align 2, !dbg !52
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1036, !dbg !52
  store half %2005, ptr addrspace(1) %address.local1058, align 2, !dbg !52
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1039, !dbg !52
  store half %2004, ptr addrspace(1) %address.local1060, align 2, !dbg !52
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1042, !dbg !52
  store half %2003, ptr addrspace(1) %address.local1062, align 2, !dbg !52
  %2025 = fmul <4 x float> %1931, %1691, !dbg !50
  %2026 = fptrunc <4 x float> %2025 to <4 x half>, !dbg !51
  %2027 = extractelement <4 x half> %2026, i64 3, !dbg !51
  %2028 = extractelement <4 x half> %2026, i64 2, !dbg !51
  %2029 = extractelement <4 x half> %2026, i64 1, !dbg !51
  %2030 = extractelement <4 x half> %2026, i64 0, !dbg !51
  %2031 = fmul <4 x float> %1931, %1658, !dbg !50
  %2032 = fptrunc <4 x float> %2031 to <4 x half>, !dbg !51
  %2033 = extractelement <4 x half> %2032, i64 3, !dbg !51
  %2034 = extractelement <4 x half> %2032, i64 2, !dbg !51
  %2035 = extractelement <4 x half> %2032, i64 1, !dbg !51
  %2036 = extractelement <4 x half> %2032, i64 0, !dbg !51
  %2037 = fmul <4 x float> %1930, %1624, !dbg !50
  %2038 = fptrunc <4 x float> %2037 to <4 x half>, !dbg !51
  %2039 = extractelement <4 x half> %2038, i64 3, !dbg !51
  %2040 = extractelement <4 x half> %2038, i64 2, !dbg !51
  %2041 = extractelement <4 x half> %2038, i64 1, !dbg !51
  %2042 = extractelement <4 x half> %2038, i64 0, !dbg !51
  %2043 = fmul <4 x float> %1930, %1591, !dbg !50
  %2044 = fptrunc <4 x float> %2043 to <4 x half>, !dbg !51
  %2045 = extractelement <4 x half> %2044, i64 3, !dbg !51
  %2046 = extractelement <4 x half> %2044, i64 2, !dbg !51
  %2047 = extractelement <4 x half> %2044, i64 1, !dbg !51
  %2048 = extractelement <4 x half> %2044, i64 0, !dbg !51
  %address.local1078 = or disjoint i64 %address.local964, 64, !dbg !52
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1078, !dbg !52
  store half %2048, ptr addrspace(1) %address.local1079, align 2, !dbg !52
  %address.local1081 = or disjoint i64 %address.local964, 65, !dbg !52
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1081, !dbg !52
  store half %2047, ptr addrspace(1) %address.local1082, align 2, !dbg !52
  %address.local1084 = or disjoint i64 %address.local964, 66, !dbg !52
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1084, !dbg !52
  store half %2046, ptr addrspace(1) %address.local1085, align 2, !dbg !52
  %address.local1087 = or disjoint i64 %address.local964, 67, !dbg !52
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1087, !dbg !52
  store half %2045, ptr addrspace(1) %address.local1088, align 2, !dbg !52
  %address.local1090 = or disjoint i64 %address.local964, 80, !dbg !52
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1090, !dbg !52
  store half %2042, ptr addrspace(1) %address.local1091, align 2, !dbg !52
  %address.local1093 = or disjoint i64 %address.local964, 81, !dbg !52
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1093, !dbg !52
  store half %2041, ptr addrspace(1) %address.local1094, align 2, !dbg !52
  %address.local1096 = or disjoint i64 %address.local964, 82, !dbg !52
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1096, !dbg !52
  store half %2040, ptr addrspace(1) %address.local1097, align 2, !dbg !52
  %address.local1099 = or disjoint i64 %address.local964, 83, !dbg !52
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1099, !dbg !52
  store half %2039, ptr addrspace(1) %address.local1100, align 2, !dbg !52
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1078, !dbg !52
  store half %2036, ptr addrspace(1) %address.local1105, align 2, !dbg !52
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1081, !dbg !52
  store half %2035, ptr addrspace(1) %address.local1107, align 2, !dbg !52
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1084, !dbg !52
  store half %2034, ptr addrspace(1) %address.local1109, align 2, !dbg !52
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1087, !dbg !52
  store half %2033, ptr addrspace(1) %address.local1111, align 2, !dbg !52
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1090, !dbg !52
  store half %2030, ptr addrspace(1) %address.local1113, align 2, !dbg !52
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1093, !dbg !52
  store half %2029, ptr addrspace(1) %address.local1115, align 2, !dbg !52
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1096, !dbg !52
  store half %2028, ptr addrspace(1) %address.local1117, align 2, !dbg !52
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1099, !dbg !52
  store half %2027, ptr addrspace(1) %address.local1119, align 2, !dbg !52
  %2049 = fmul <4 x float> %1931, %1557, !dbg !50
  %2050 = fptrunc <4 x float> %2049 to <4 x half>, !dbg !51
  %2051 = extractelement <4 x half> %2050, i64 3, !dbg !51
  %2052 = extractelement <4 x half> %2050, i64 2, !dbg !51
  %2053 = extractelement <4 x half> %2050, i64 1, !dbg !51
  %2054 = extractelement <4 x half> %2050, i64 0, !dbg !51
  %2055 = fmul <4 x float> %1931, %1524, !dbg !50
  %2056 = fptrunc <4 x float> %2055 to <4 x half>, !dbg !51
  %2057 = extractelement <4 x half> %2056, i64 3, !dbg !51
  %2058 = extractelement <4 x half> %2056, i64 2, !dbg !51
  %2059 = extractelement <4 x half> %2056, i64 1, !dbg !51
  %2060 = extractelement <4 x half> %2056, i64 0, !dbg !51
  %2061 = fmul <4 x float> %1930, %1490, !dbg !50
  %2062 = fptrunc <4 x float> %2061 to <4 x half>, !dbg !51
  %2063 = extractelement <4 x half> %2062, i64 3, !dbg !51
  %2064 = extractelement <4 x half> %2062, i64 2, !dbg !51
  %2065 = extractelement <4 x half> %2062, i64 1, !dbg !51
  %2066 = extractelement <4 x half> %2062, i64 0, !dbg !51
  %2067 = fmul <4 x float> %1930, %1457, !dbg !50
  %2068 = fptrunc <4 x float> %2067 to <4 x half>, !dbg !51
  %2069 = extractelement <4 x half> %2068, i64 3, !dbg !51
  %2070 = extractelement <4 x half> %2068, i64 2, !dbg !51
  %2071 = extractelement <4 x half> %2068, i64 1, !dbg !51
  %2072 = extractelement <4 x half> %2068, i64 0, !dbg !51
  %address.local1135 = or disjoint i64 %address.local964, 96, !dbg !52
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1135, !dbg !52
  store half %2072, ptr addrspace(1) %address.local1136, align 2, !dbg !52
  %address.local1138 = or disjoint i64 %address.local964, 97, !dbg !52
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1138, !dbg !52
  store half %2071, ptr addrspace(1) %address.local1139, align 2, !dbg !52
  %address.local1141 = or disjoint i64 %address.local964, 98, !dbg !52
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1141, !dbg !52
  store half %2070, ptr addrspace(1) %address.local1142, align 2, !dbg !52
  %address.local1144 = or disjoint i64 %address.local964, 99, !dbg !52
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1144, !dbg !52
  store half %2069, ptr addrspace(1) %address.local1145, align 2, !dbg !52
  %address.local1147 = or disjoint i64 %address.local964, 112, !dbg !52
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1147, !dbg !52
  store half %2066, ptr addrspace(1) %address.local1148, align 2, !dbg !52
  %address.local1150 = or disjoint i64 %address.local964, 113, !dbg !52
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1150, !dbg !52
  store half %2065, ptr addrspace(1) %address.local1151, align 2, !dbg !52
  %address.local1153 = or disjoint i64 %address.local964, 114, !dbg !52
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1153, !dbg !52
  store half %2064, ptr addrspace(1) %address.local1154, align 2, !dbg !52
  %address.local1156 = or i64 %address.local953, 115, !dbg !52
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1156, !dbg !52
  store half %2063, ptr addrspace(1) %address.local1157, align 2, !dbg !52
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1135, !dbg !52
  store half %2060, ptr addrspace(1) %address.local1162, align 2, !dbg !52
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1138, !dbg !52
  store half %2059, ptr addrspace(1) %address.local1164, align 2, !dbg !52
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1141, !dbg !52
  store half %2058, ptr addrspace(1) %address.local1166, align 2, !dbg !52
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1144, !dbg !52
  store half %2057, ptr addrspace(1) %address.local1168, align 2, !dbg !52
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1147, !dbg !52
  store half %2054, ptr addrspace(1) %address.local1170, align 2, !dbg !52
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1150, !dbg !52
  store half %2053, ptr addrspace(1) %address.local1172, align 2, !dbg !52
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1153, !dbg !52
  store half %2052, ptr addrspace(1) %address.local1174, align 2, !dbg !52
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1156, !dbg !52
  store half %2051, ptr addrspace(1) %address.local1176, align 2, !dbg !52
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
!1 = !DIFile(filename: "causal.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer")
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
!41 = !DILocation(line: 90, column: 22, scope: !3)
!42 = !DILocation(line: 91, column: 23, scope: !3)
!43 = !DILocation(line: 92, column: 20, scope: !3)
!44 = !DILocation(line: 103, column: 22, scope: !3)
!45 = !DILocation(line: 93, column: 22, scope: !3)
!46 = !DILocation(line: 104, column: 20, scope: !3)
!47 = !DILocation(line: 94, column: 22, scope: !3)
!48 = !DILocation(line: 96, column: 16, scope: !3)
!49 = !DILocation(line: 95, column: 20, scope: !3)
!50 = !DILocation(line: 107, column: 23, scope: !3)
!51 = !DILocation(line: 108, column: 19, scope: !3)
!52 = !DILocation(line: 109, column: 9, scope: !3)
