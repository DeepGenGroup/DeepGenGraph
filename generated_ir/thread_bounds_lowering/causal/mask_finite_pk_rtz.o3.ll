; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_bounds_lowering/causal/mask_finite_pk_rtz.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_bounds_lowering/causal/input.mlir"

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
  %address.local332 = zext nneg i32 %47 to i64, !dbg !9
  %address.local333 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !9
  %address.local336 = lshr i32 %address.thread, 4, !dbg !9
  %address.local337 = and i32 %address.local336, 8, !dbg !9
  %address.local338 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local339 = and i32 %address.local338, 16256, !dbg !9
  %address.local340 = or disjoint i32 %address.local337, %address.local339, !dbg !9
  %48 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1310 = shl i64 %43, 8, !dbg !9
  %49 = add i64 %.idx, %.idx1310, !dbg !9
  %50 = shl nuw nsw i32 %address.local340, 1, !dbg !9
  %51 = trunc i64 %49 to i32, !dbg !9
  %dma.buffer.offset1308 = or disjoint i32 %50, %51, !dbg !9
  %52 = insertelement <2 x i64> poison, i64 %48, i64 0, !dbg !9
  %53 = bitcast <2 x i64> %52 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1309 = shufflevector <4 x i32> %53, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local345 = or disjoint i64 %address.local332, 2048, !dbg !9
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local345, !dbg !9
  %address.local354 = or disjoint i32 %address.local340, 16, !dbg !9
  %54 = shl nuw nsw i32 %address.local354, 1, !dbg !9
  %dma.buffer.offset1306 = or disjoint i32 %54, %51, !dbg !9
  %55 = shl nuw nsw i64 %40, 19, !dbg !9
  %56 = add i64 %55, %44, !dbg !9
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local332, !dbg !9
  %57 = trunc i64 %56 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local340, %57, !dbg !9
  %58 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1304 = or disjoint i32 %58, 64, !dbg !9
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local345, !dbg !9
  %59 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local333, i64 0, !dbg !9
  %60 = insertelement <4 x ptr addrspace(3)> %59, ptr addrspace(3) %address.local346, i64 1, !dbg !9
  %61 = insertelement <4 x ptr addrspace(3)> %60, ptr addrspace(3) %address.local357, i64 2, !dbg !9
  %62 = insertelement <4 x ptr addrspace(3)> %61, ptr addrspace(3) %address.local361, i64 3, !dbg !9
  %63 = ptrtoint <4 x ptr addrspace(3)> %62 to <4 x i64>, !dbg !9
  %64 = trunc <4 x i64> %63 to <4 x i32>, !dbg !9
  %65 = extractelement <4 x i32> %64, i64 0, !dbg !9
  %66 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %65), !dbg !9
  %67 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1308, <4 x i32> %dma.buffer.rsrc1309) #9, !dbg !9
  %68 = extractelement <4 x i32> %64, i64 1, !dbg !9
  %69 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %68), !dbg !9
  %70 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset1306, <4 x i32> %dma.buffer.rsrc1309) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %71 = extractelement <4 x i32> %64, i64 2, !dbg !9
  %72 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %71), !dbg !9
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %dma.buffer.offset1304, <4 x i32> %dma.buffer.rsrc1309) #9, !dbg !9
  %74 = extractelement <4 x i32> %64, i64 3, !dbg !9
  %75 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %74), !dbg !9
  %.tr1314 = or disjoint i32 %address.local354, %57, !dbg !9
  %76 = shl i32 %.tr1314, 1, !dbg !9
  %dma.buffer.offset1302 = or disjoint i32 %76, 64, !dbg !9
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %dma.buffer.offset1302, <4 x i32> %dma.buffer.rsrc1309) #9, !dbg !9
  %address.local365 = lshr i32 %address.thread, 2, !dbg !11
  %address.local366 = and i32 %address.local365, 4, !dbg !11
  %address.local367 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local368 = and i32 %address.local367, 384, !dbg !11
  %address.local369 = or disjoint i32 %address.local366, %address.local368, !dbg !11
  %address.local371 = and i32 %46, 120, !dbg !11
  %address.local372 = or disjoint i32 %address.local369, %address.local371, !dbg !11
  %address.local373 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local374 = and i32 %address.local373, 1024, !dbg !11
  %address.local375 = or disjoint i32 %address.local372, %address.local374, !dbg !11
  %address.local376 = zext nneg i32 %address.local375 to i64, !dbg !11
  %address.local377 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local376, !dbg !11
  %78 = load <4 x half>, ptr addrspace(3) %address.local377, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local377, i64 4096, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %80 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1131 = getelementptr i8, ptr addrspace(3) %address.local377, i64 1024, !dbg !11
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1131, align 8, !dbg !11
  %82 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1132 = getelementptr i8, ptr addrspace(3) %address.local377, i64 5120, !dbg !11
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1132, align 8, !dbg !11
  %84 = fmul <4 x half> %81, splat (half 1.275630e-01), !dbg !11
  %85 = fmul <4 x half> %83, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1300 = or disjoint i32 %58, 128, !dbg !9
  %86 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1300, <4 x i32> %dma.buffer.rsrc1309) #9, !dbg !9
  %dma.buffer.offset1298 = or disjoint i32 %76, 128, !dbg !9
  %87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset1298, <4 x i32> %dma.buffer.rsrc1309) #9, !dbg !9
  %lds.const.gep1133 = getelementptr i8, ptr addrspace(3) %address.local377, i64 8192, !dbg !11
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1133, align 8, !dbg !11
  %lds.const.gep1134 = getelementptr i8, ptr addrspace(3) %address.local377, i64 12288, !dbg !11
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep1134, align 8, !dbg !11
  %lds.const.gep1135 = getelementptr i8, ptr addrspace(3) %address.local377, i64 9216, !dbg !11
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1135, align 8, !dbg !11
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1136 = getelementptr i8, ptr addrspace(3) %address.local377, i64 13312, !dbg !11
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1136, align 8, !dbg !11
  %93 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !11
  %94 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !11
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1296 = or disjoint i32 %58, 192, !dbg !9
  %96 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %dma.buffer.offset1296, <4 x i32> %dma.buffer.rsrc1309) #9, !dbg !9
  %dma.buffer.offset1294 = or disjoint i32 %76, 192, !dbg !9
  %97 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %dma.buffer.offset1294, <4 x i32> %dma.buffer.rsrc1309) #9, !dbg !9
  %98 = load <4 x half>, ptr addrspace(3) %address.local377, align 8, !dbg !11
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1131, align 8, !dbg !11
  %101 = fmul <4 x half> %98, splat (half 1.275630e-01), !dbg !11
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep1132, align 8, !dbg !11
  %103 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !11
  %104 = fmul <4 x half> %100, splat (half 1.275630e-01), !dbg !11
  %105 = fmul <4 x half> %102, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %106 = load <4 x half>, ptr addrspace(3) %lds.const.gep1133, align 8, !dbg !11
  %107 = load <4 x half>, ptr addrspace(3) %lds.const.gep1134, align 8, !dbg !11
  %108 = load <4 x half>, ptr addrspace(3) %lds.const.gep1135, align 8, !dbg !11
  %109 = fmul <4 x half> %106, splat (half 1.275630e-01), !dbg !11
  %110 = load <4 x half>, ptr addrspace(3) %lds.const.gep1136, align 8, !dbg !11
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
  %dma.buffer.rsrc1293 = shufflevector <4 x i32> %138, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %kdma.source = zext nneg i32 %kdma.owner to i64
  %kdma.source1460 = lshr i64 %kdma.source, 4
  %kdma.source1461 = shl nuw nsw i64 %kdma.source1460, 9
  %kdma.source1462 = shl nuw nsw i64 %kdma.source, 7
  %kdma.source1463 = add nuw nsw i64 %kdma.source1461, %kdma.source1462
  %kdma.source1464 = lshr i64 %kdma.source, 5
  %kdma.source1465 = lshr i64 %kdma.source, 2
  %kdma.source1466 = and i64 %kdma.source1465, 1
  %kdma.source1467 = and i64 %kdma.source1466, %kdma.source1460
  %kdma.source1468 = or disjoint i64 %kdma.source1467, %kdma.source1464
  %139 = shl nuw nsw i64 %kdma.source1468, 10
  %kdma.source1470 = sub nsw i64 %kdma.source1463, %139
  %kdma.source1471 = and i64 %kdma.source1470, 16256
  %kdma.source1472 = lshr i32 %kdma.tid, 4
  %kdma.source1473 = and i32 %kdma.source1472, 8
  %kdma.source1474 = zext nneg i32 %kdma.source1473 to i64
  %kdma.source1475 = or disjoint i64 %kdma.source1471, %kdma.source1474
  %kdma.source1479 = zext nneg i32 %kdma.owner.hi to i64
  %kdma.source1480 = lshr i64 %kdma.source1479, 4
  %kdma.source1481 = shl nuw nsw i64 %kdma.source1480, 9
  %kdma.source1482 = shl nuw nsw i64 %kdma.source1479, 7
  %kdma.source1483 = add nuw nsw i64 %kdma.source1481, %kdma.source1482
  %kdma.source1484 = lshr i64 %kdma.source1479, 5
  %kdma.source1485 = lshr i64 %kdma.source1479, 2
  %kdma.source1486 = and i64 %kdma.source1485, 1
  %kdma.source1487 = and i64 %kdma.source1486, %kdma.source1480
  %kdma.source1488 = add nuw nsw i64 %kdma.source1487, %kdma.source1484
  %140 = shl nuw nsw i64 %kdma.source1488, 10
  %kdma.source1490 = sub nsw i64 %kdma.source1483, %140
  %kdma.source1491 = and i64 %kdma.source1490, 16256
  %kdma.source1495 = or disjoint i64 %kdma.source1491, %kdma.source1474
  %kdma.lds1500 = add i32 %66, 512
  %kdma.source1502 = or disjoint i64 %kdma.source, 256
  %kdma.source1503 = lshr i64 %kdma.source1502, 4
  %kdma.source1504 = shl nuw nsw i64 %kdma.source1503, 9
  %kdma.source1506 = add nuw nsw i64 %kdma.source1504, %kdma.source1462
  %kdma.source1507 = lshr i64 %kdma.source1502, 5
  %kdma.source1510 = and i64 %kdma.source1503, %kdma.source1466
  %kdma.source1511 = or disjoint i64 %kdma.source1510, %kdma.source1507
  %141 = shl nuw nsw i64 %kdma.source1511, 10
  %kdma.source1513 = sub nsw i64 %kdma.source1506, %141
  %kdma.source1514 = and i64 %kdma.source1513, 16256
  %kdma.source1517 = or disjoint i32 %kdma.source1473, 16
  %kdma.source1518 = zext nneg i32 %kdma.source1517 to i64
  %kdma.source1519 = or disjoint i64 %kdma.source1514, %kdma.source1518
  %kdma.source1526 = or disjoint i64 %kdma.source1479, 256
  %kdma.source1527 = lshr i64 %kdma.source1526, 4
  %kdma.source1528 = shl nuw nsw i64 %kdma.source1527, 9
  %kdma.source1530 = add nuw nsw i64 %kdma.source1528, %kdma.source1482
  %kdma.source1531 = lshr i64 %kdma.source1526, 5
  %kdma.source1534 = and i64 %kdma.source1527, %kdma.source1486
  %kdma.source1535 = add nuw nsw i64 %kdma.source1534, %kdma.source1531
  %142 = shl nuw nsw i64 %kdma.source1535, 10
  %kdma.source1537 = sub nsw i64 %kdma.source1530, %142
  %kdma.source1538 = and i64 %kdma.source1537, 16256
  %kdma.source1543 = or disjoint i64 %kdma.source1538, %kdma.source1518
  %kdma.lds1548 = add i32 %69, 512
  %143 = or disjoint i64 %134, %133
  %kdma.source1569 = or disjoint i32 %kdma.intra, 64
  %kdma.lds1594 = add i32 %72, 512
  %kdma.lds1644 = add i32 %75, 512
  %kdma.source1665 = or disjoint i32 %kdma.intra, 128
  %kdma.source1761 = or disjoint i32 %kdma.intra, 192
  %144 = ptrtoint ptr addrspace(1) %2 to i64
  %145 = insertelement <2 x i64> poison, i64 %144, i64 0
  %146 = bitcast <2 x i64> %145 to <4 x i32>
  %dma.buffer.rsrc1277 = shufflevector <4 x i32> %146, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %vdb.dma.base = add i32 %66, 8192
  %vdb.dma.base1410 = add i32 %69, 8192
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
  %154 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1405, %153 ]
  %155 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1401, %153 ]
  %156 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1397, %153 ]
  %157 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1393, %153 ]
  %158 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %1157, %153 ]
  %159 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1391, %153 ]
  %160 = phi i64 [ 0, %.lr.ph ], [ %1406, %153 ]
  %161 = lshr i64 %160, 12, !dbg !14
  %162 = add nuw nsw i64 %161, %29, !dbg !14
  %163 = shl nuw nsw i64 %160, 7, !dbg !14
  %164 = shl i64 %162, 19, !dbg !14
  %165 = and i64 %164, 16252928, !dbg !14
  %address.thread428 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local429 = zext nneg i32 %address.thread428 to i64, !dbg !9
  %166 = insertelement <2 x i64> %148, i64 %160, i64 1, !dbg !14
  %address.local472 = shl nuw nsw i64 %address.local429, 1, !dbg !15
  %address.local473 = and i64 %address.local472, 24, !dbg !15
  %address.local475 = shl i64 %address.local429, 62, !dbg !15
  %address.local476 = ashr exact i64 %address.local475, 57, !dbg !15
  %address.local477 = and i64 %address.local476, -64, !dbg !15
  %address.local482 = and i64 %address.local429, 63, !dbg !11
  %address.local483 = lshr i64 %address.local482, 5, !dbg !11
  %address.local484 = mul nuw nsw i64 %address.local483, 1016, !dbg !11
  %.idx1345 = mul nuw nsw i64 %address.local429, 320, !dbg !15
  %167 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1345, !dbg !15
  %.idx1963 = shl nuw nsw i64 %address.local473, 1, !dbg !15
  %address.local486.idx = mul nuw nsw i64 %address.local483, 2032, !dbg !15
  %address.local497 = shl nuw nsw i32 %address.thread428, 1, !dbg !15
  %address.local498 = and i32 %address.local497, 24, !dbg !15
  %address.local499 = shl nuw nsw i32 %address.thread428, 8, !dbg !15
  %address.local500 = and i32 %address.local499, 768, !dbg !15
  %address.local501 = or disjoint i32 %address.local498, %address.local500, !dbg !15
  %address.local502 = shl nuw nsw i32 %address.thread428, 5, !dbg !15
  %address.local503 = and i32 %address.local502, 96, !dbg !15
  %address.local504 = or disjoint i32 %address.local501, %address.local503, !dbg !15
  %address.local505 = zext nneg i32 %address.local504 to i64, !dbg !15
  %address.local506 = lshr i32 %address.thread428, 1, !dbg !15
  %address.local507 = and i32 %address.local506, 1, !dbg !15
  %address.local508 = zext nneg i32 %address.local507 to i64, !dbg !15
  %168 = add nuw nsw i64 %address.local486.idx, 64, !dbg !15
  %169 = insertelement <2 x i64> poison, i64 %address.local429, i64 0, !dbg !14
  %170 = insertelement <2 x i64> %169, i64 %160, i64 1, !dbg !14
  %171 = lshr <2 x i64> %170, <i64 2, i64 5>, !dbg !14
  %172 = extractelement <2 x i64> %171, i64 1, !dbg !14
  %173 = and i64 %172, 127, !dbg !14
  %.idx13231326 = or disjoint i64 %165, %173, !dbg !14
  %174 = shl nuw nsw <2 x i64> %170, <i64 3, i64 14>, !dbg !9
  %175 = shl <2 x i64> %171, <i64 5, i64 19>, !dbg !17
  %176 = sub <2 x i64> %174, %175, !dbg !17
  %177 = extractelement <2 x i64> %176, i64 1, !dbg !14
  %178 = add i64 %.idx13231326, %177, !dbg !14
  %kdma.source1476 = add i64 %178, %kdma.source1475, !dbg !14
  %kdma.source1477 = trunc i64 %kdma.source1476 to i32, !dbg !14
  %kdma.source1478 = shl i32 %kdma.source1477, 1, !dbg !14
  %kdma.offset = add i32 %kdma.source1478, %kdma.intra, !dbg !14
  %179 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1496 = add i64 %178, %kdma.source1495, !dbg !14
  %kdma.source1497 = trunc i64 %kdma.source1496 to i32, !dbg !14
  %kdma.source1498 = shl i32 %kdma.source1497, 1, !dbg !14
  %kdma.offset1499 = add i32 %kdma.source1498, %kdma.intra.hi, !dbg !14
  %180 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1500, i32 %kdma.offset1499, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1520 = add i64 %178, %kdma.source1519, !dbg !14
  %kdma.source1521 = trunc i64 %kdma.source1520 to i32, !dbg !14
  %kdma.source1522 = shl i32 %kdma.source1521, 1, !dbg !14
  %kdma.offset1523 = add i32 %kdma.source1522, %kdma.intra, !dbg !14
  %181 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %kdma.offset1523, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1544 = add i64 %178, %kdma.source1543, !dbg !14
  %kdma.source1545 = trunc i64 %kdma.source1544 to i32, !dbg !14
  %kdma.source1546 = shl i32 %kdma.source1545, 1, !dbg !14
  %kdma.offset1547 = add i32 %kdma.source1546, %kdma.intra.hi, !dbg !14
  %182 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1548, i32 %kdma.offset1547, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx13331337 = add i64 %177, %172, !dbg !14
  %183 = add i64 %.idx13331337, %117, !dbg !14
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
  %kdma.source1566 = add i64 %204, %kdma.source1475, !dbg !14
  %kdma.source1567 = trunc i64 %kdma.source1566 to i32, !dbg !14
  %kdma.source1568 = shl i32 %kdma.source1567, 1, !dbg !14
  %kdma.offset1570 = add i32 %kdma.source1569, %kdma.source1568, !dbg !14
  %205 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %kdma.offset1570, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1589 = add i64 %204, %kdma.source1495, !dbg !14
  %kdma.source1590 = trunc i64 %kdma.source1589 to i32, !dbg !14
  %kdma.source1591 = shl i32 %kdma.source1590, 1, !dbg !14
  %kdma.offset1593.reass.reass = add i32 %kdma.source1591, %invariant.op
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1594, i32 %kdma.offset1593.reass.reass, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1614 = add i64 %204, %kdma.source1519, !dbg !14
  %kdma.source1615 = trunc i64 %kdma.source1614 to i32, !dbg !14
  %kdma.source1616 = shl i32 %kdma.source1615, 1, !dbg !14
  %kdma.offset1618 = add i32 %kdma.source1569, %kdma.source1616, !dbg !14
  %207 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %kdma.offset1618, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1639 = add i64 %204, %kdma.source1543, !dbg !14
  %kdma.source1640 = trunc i64 %kdma.source1639 to i32, !dbg !14
  %kdma.source1641 = shl i32 %kdma.source1640, 1, !dbg !14
  %kdma.offset1643.reass.reass = add i32 %kdma.source1641, %invariant.op60
  %208 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1644, i32 %kdma.offset1643.reass.reass, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %209 = extractelement <2 x i64> %171, i64 0, !dbg !18
  %address.local480 = and i64 %209, 12, !dbg !11
  %.idx1346 = mul nsw i64 %209, -1280, !dbg !15
  %210 = getelementptr i8, ptr addrspace(3) %167, i64 %.idx1346, !dbg !15
  %211 = getelementptr [2 x i8], ptr addrspace(3) %210, i64 %address.local473, !dbg !15
  %212 = getelementptr [2 x i8], ptr addrspace(3) %211, i64 %address.local477, !dbg !15
  %213 = getelementptr [2 x i8], ptr addrspace(3) %212, i64 %address.local480, !dbg !15
  %address.local486 = getelementptr [2 x i8], ptr addrspace(3) %213, i64 %address.local484, !dbg !15
  %214 = add nsw i64 %.idx1346, %.idx1345, !dbg !15
  %215 = or disjoint i64 %214, %.idx1963, !dbg !15
  %216 = or disjoint i64 %address.local477, %address.local480, !dbg !15
  %217 = shl nsw i64 %216, 1, !dbg !15
  %218 = add nsw i64 %215, %address.local486.idx, !dbg !15
  %219 = add nsw i64 %218, %217, !dbg !15
  %220 = lshr i64 %219, 6, !dbg !15
  %221 = and i64 %220, 8, !dbg !15
  %222 = xor i64 %221, %219, !dbg !15
  %223 = sub i64 %222, %219, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local486, i64 %223, !dbg !15
  %224 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %226 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %123) #8, !dbg !15, !frisk.register_lifetimes !5
  %address.local509 = add nsw i64 %address.base, %address.local508, !dbg !15
  %address.local510 = shl i64 %address.local509, 6, !dbg !15
  %address.local511 = sub i64 %address.local505, %address.local510, !dbg !15
  %227 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local511, !dbg !15
  %228 = getelementptr [2 x i8], ptr addrspace(3) %227, i64 %address.local480, !dbg !15
  %229 = getelementptr [2 x i8], ptr addrspace(3) %228, i64 %address.local484, !dbg !15
  %address.local515 = getelementptr i8, ptr addrspace(3) %229, i64 64, !dbg !15
  %.idx19661969 = add i64 %address.local511, %address.local480, !dbg !15
  %230 = shl i64 %.idx19661969, 1, !dbg !15
  %231 = add i64 %168, %230, !dbg !15
  %232 = lshr i64 %231, 6, !dbg !15
  %233 = and i64 %232, 8, !dbg !15
  %234 = xor i64 %233, %231, !dbg !15
  %235 = sub i64 %234, %231, !dbg !15
  %kdma.swizzled1837 = getelementptr i8, ptr addrspace(3) %address.local515, i64 %235, !dbg !15
  %236 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1837, align 8, !dbg !15, !frisk.k_lds_pair !19
  %kdma.gep1841 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 128, !dbg !15
  %237 = load <4 x half>, ptr addrspace(3) %kdma.gep1841, align 8, !dbg !15, !frisk.k_lds_pair !19
  %kdma.gep1838 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 4096, !dbg !15
  %238 = load <4 x half>, ptr addrspace(3) %kdma.gep1838, align 8, !dbg !15, !frisk.k_lds_pair !20
  %kdma.gep1842 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 4224, !dbg !15
  %239 = load <4 x half>, ptr addrspace(3) %kdma.gep1842, align 8, !dbg !15, !frisk.k_lds_pair !20
  %kdma.gep1839 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %240 = load <4 x half>, ptr addrspace(3) %kdma.gep1839, align 8, !dbg !15, !frisk.k_lds_pair !21
  %kdma.gep1843 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %241 = load <4 x half>, ptr addrspace(3) %kdma.gep1843, align 8, !dbg !15, !frisk.k_lds_pair !21
  %kdma.gep1840 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %242 = load <4 x half>, ptr addrspace(3) %kdma.gep1840, align 8, !dbg !15, !frisk.k_lds_pair !22
  %kdma.gep1844 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %243 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !15, !frisk.k_lds_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %226, <4 x float> %225) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1845 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 1024, !dbg !15
  %245 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !15, !frisk.k_lds_pair !23
  %kdma.gep1849 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 1152, !dbg !15
  %246 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !15, !frisk.k_lds_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %236, <4 x float> zeroinitializer) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1846 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 5120, !dbg !15
  %248 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !15, !frisk.k_lds_pair !24
  %kdma.gep1850 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 5248, !dbg !15
  %249 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !15, !frisk.k_lds_pair !24
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %238, <4 x float> %247) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1847 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %251 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !15, !frisk.k_lds_pair !25
  %kdma.gep1848 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %252 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !15, !frisk.k_lds_pair !25
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
  %kdma.source1662 = add i64 %297, %kdma.source1475, !dbg !14
  %kdma.source1663 = trunc i64 %kdma.source1662 to i32, !dbg !14
  %kdma.source1664 = shl i32 %kdma.source1663, 1, !dbg !14
  %kdma.offset1666 = add i32 %kdma.source1665, %kdma.source1664, !dbg !14
  %298 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset1666, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1685 = add i64 %297, %kdma.source1495, !dbg !14
  %kdma.source1686 = trunc i64 %kdma.source1685 to i32, !dbg !14
  %kdma.source1687 = shl i32 %kdma.source1686, 1, !dbg !14
  %kdma.offset1689.reass.reass = add i32 %kdma.source1687, %invariant.op61
  %299 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1500, i32 %kdma.offset1689.reass.reass, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1710 = add i64 %297, %kdma.source1519, !dbg !14
  %kdma.source1711 = trunc i64 %kdma.source1710 to i32, !dbg !14
  %kdma.source1712 = shl i32 %kdma.source1711, 1, !dbg !14
  %kdma.offset1714 = add i32 %kdma.source1665, %kdma.source1712, !dbg !14
  %300 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %kdma.offset1714, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1735 = add i64 %297, %kdma.source1543, !dbg !14
  %kdma.source1736 = trunc i64 %kdma.source1735 to i32, !dbg !14
  %kdma.source1737 = shl i32 %kdma.source1736, 1, !dbg !14
  %kdma.offset1739.reass.reass = add i32 %kdma.source1737, %invariant.op62
  %301 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1548, i32 %kdma.offset1739.reass.reass, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.gep1867 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !15, !frisk.k_lds_pair !26
  %kdma.gep1871 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !15, !frisk.k_lds_pair !26
  %kdma.gep1868 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !15, !frisk.k_lds_pair !27
  %kdma.gep1872 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !15, !frisk.k_lds_pair !27
  %kdma.gep1869 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 8192, !dbg !15
  %306 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !15, !frisk.k_lds_pair !28
  %kdma.gep1873 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 8320, !dbg !15
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !15, !frisk.k_lds_pair !28
  %kdma.gep1870 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 12288, !dbg !15
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !15, !frisk.k_lds_pair !29
  %kdma.gep1874 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 12416, !dbg !15
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !15, !frisk.k_lds_pair !29
  %kdma.gep1875 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !15, !frisk.k_lds_pair !30
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !15, !frisk.k_lds_pair !30
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %302, <4 x float> %244) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1876 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !15, !frisk.k_lds_pair !31
  %kdma.gep1880 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !15, !frisk.k_lds_pair !31
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %304, <4 x float> %312) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 9216, !dbg !15
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !15, !frisk.k_lds_pair !32
  %kdma.gep1881 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 9344, !dbg !15
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !15, !frisk.k_lds_pair !32
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %306, <4 x float> %250) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 13312, !dbg !15
  %319 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1882 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 13440, !dbg !15
  %320 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !15, !frisk.k_lds_pair !33
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
  %kdma.source1758 = add i64 %366, %kdma.source1475, !dbg !14
  %kdma.source1759 = trunc i64 %kdma.source1758 to i32, !dbg !14
  %kdma.source1760 = shl i32 %kdma.source1759, 1, !dbg !14
  %kdma.offset1762 = add i32 %kdma.source1761, %kdma.source1760, !dbg !14
  %367 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %kdma.offset1762, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1781 = add i64 %366, %kdma.source1495, !dbg !14
  %kdma.source1782 = trunc i64 %kdma.source1781 to i32, !dbg !14
  %kdma.source1783 = shl i32 %kdma.source1782, 1, !dbg !14
  %kdma.offset1785.reass.reass = add i32 %kdma.source1783, %invariant.op63
  %368 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1594, i32 %kdma.offset1785.reass.reass, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1806 = add i64 %366, %kdma.source1519, !dbg !14
  %kdma.source1807 = trunc i64 %kdma.source1806 to i32, !dbg !14
  %kdma.source1808 = shl i32 %kdma.source1807, 1, !dbg !14
  %kdma.offset1810 = add i32 %kdma.source1761, %kdma.source1808, !dbg !14
  %369 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %75, i32 %kdma.offset1810, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %kdma.source1831 = add i64 %366, %kdma.source1543, !dbg !14
  %kdma.source1832 = trunc i64 %kdma.source1831 to i32, !dbg !14
  %kdma.source1833 = shl i32 %kdma.source1832, 1, !dbg !14
  %kdma.offset1835.reass.reass = add i32 %kdma.source1833, %invariant.op64
  %370 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1644, i32 %kdma.offset1835.reass.reass, <4 x i32> %dma.buffer.rsrc1293) #9, !dbg !14
  %371 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !34
  %372 = load <4 x half>, ptr addrspace(3) %kdma.gep1839, align 8, !dbg !15, !frisk.k_lds_pair !34
  %373 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !35
  %374 = load <4 x half>, ptr addrspace(3) %kdma.gep1840, align 8, !dbg !15, !frisk.k_lds_pair !35
  %375 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1837, align 8, !dbg !15, !frisk.k_lds_pair !36
  %376 = load <4 x half>, ptr addrspace(3) %kdma.gep1841, align 8, !dbg !15, !frisk.k_lds_pair !36
  %377 = load <4 x half>, ptr addrspace(3) %kdma.gep1838, align 8, !dbg !15, !frisk.k_lds_pair !37
  %378 = load <4 x half>, ptr addrspace(3) %kdma.gep1842, align 8, !dbg !15, !frisk.k_lds_pair !37
  %379 = load <4 x half>, ptr addrspace(3) %kdma.gep1843, align 8, !dbg !15, !frisk.k_lds_pair !38
  %380 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !15, !frisk.k_lds_pair !38
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %371, <4 x float> %315) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %382 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !15, !frisk.k_lds_pair !39
  %383 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !15, !frisk.k_lds_pair !39
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %373, <4 x float> %381) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %385 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !15, !frisk.k_lds_pair !40
  %386 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !15, !frisk.k_lds_pair !40
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %375, <4 x float> %321) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_lds_prefetch !5
  %388 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !15, !frisk.k_lds_pair !41
  %389 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !15, !frisk.k_lds_pair !41
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
  %.tr1374 = trunc i64 %426 to i32, !dbg !17
  %dma.buffer.offset1276 = shl i32 %.tr1374, 1, !dbg !17
  %427 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1276, <4 x i32> %dma.buffer.rsrc1277) #9, !dbg !17
  %address.local851 = shl nuw nsw i32 %address.thread428, 3, !dbg !17
  %address.local852 = and i32 %address.local851, 24, !dbg !17
  %address.local854 = and i32 %address.local502, 8064, !dbg !17
  %428 = trunc i64 %425 to i32, !dbg !17
  %address.local855 = or disjoint i32 %address.local854, %428, !dbg !17
  %address.local856 = or disjoint i32 %address.local855, %address.local852, !dbg !17
  %.tr1378 = shl i32 %address.local856, 1, !dbg !17
  %dma.buffer.offset1274 = or i32 %.tr1378, 16384, !dbg !17
  %429 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset1274, <4 x i32> %dma.buffer.rsrc1277) #9, !dbg !17
  %430 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !15, !frisk.k_tail_pair !19
  %431 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !15, !frisk.k_tail_pair !19
  %432 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !15, !frisk.k_tail_pair !20
  %433 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !15, !frisk.k_tail_pair !20
  %434 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !15, !frisk.k_tail_pair !21
  %435 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !15, !frisk.k_tail_pair !21
  %436 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !15, !frisk.k_tail_pair !22
  %437 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !15, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %438 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %430, <4 x float> %384) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %439 = or disjoint i64 %163, %124, !dbg !16
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %432, <4 x float> %438) #8, !dbg !15
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %434, <4 x float> %390) #8, !dbg !15
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %436, <4 x float> %441) #8, !dbg !15
  %443 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !15, !frisk.k_tail_pair !23
  %444 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !15, !frisk.k_tail_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %431, <4 x float> %392) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %446 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !15, !frisk.k_tail_pair !24
  %447 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !15, !frisk.k_tail_pair !24
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %433, <4 x float> %445) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %449 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !15, !frisk.k_tail_pair !25
  %450 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !15, !frisk.k_tail_pair !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %435, <4 x float> %394) #8, !dbg !15
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %452 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !15, !frisk.k_tail_pair !26
  %453 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !15, !frisk.k_tail_pair !26
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
  %760 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 0, i32 8, i32 16, i32 24>, !dbg !42
  %761 = shufflevector <1 x float> %756, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %762 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 1, i32 9, i32 17, i32 25>, !dbg !42
  %763 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 2, i32 10, i32 18, i32 26>, !dbg !42
  %764 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 3, i32 11, i32 19, i32 27>, !dbg !42
  %765 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 4, i32 12, i32 20, i32 28>, !dbg !42
  %766 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 5, i32 13, i32 21, i32 29>, !dbg !42
  %767 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 6, i32 14, i32 22, i32 30>, !dbg !42
  %768 = shufflevector <32 x float> %559, <32 x float> poison, <4 x i32> <i32 7, i32 15, i32 23, i32 31>, !dbg !42
  %769 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 0, i32 8, i32 16, i32 24>, !dbg !42
  %770 = shufflevector <1 x float> %759, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %771 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 1, i32 9, i32 17, i32 25>, !dbg !42
  %772 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 2, i32 10, i32 18, i32 26>, !dbg !42
  %773 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 3, i32 11, i32 19, i32 27>, !dbg !42
  %774 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 4, i32 12, i32 20, i32 28>, !dbg !42
  %775 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 5, i32 13, i32 21, i32 29>, !dbg !42
  %776 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 6, i32 14, i32 22, i32 30>, !dbg !42
  %777 = shufflevector <32 x float> %673, <32 x float> poison, <4 x i32> <i32 7, i32 15, i32 23, i32 31>, !dbg !42
  %778 = fsub <1 x float> %754, %756, !dbg !43
  %779 = extractelement <1 x float> %778, i64 0, !dbg !44
  %780 = tail call float @llvm.amdgcn.exp2.f32(float %779), !dbg !44
  %781 = insertelement <4 x float> poison, float %780, i64 0, !dbg !45
  %782 = shufflevector <4 x float> %781, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !45
  %783 = extractvalue [2 x <1 x float>] %158, 0, !dbg !46
  %784 = extractelement <1 x float> %783, i64 0, !dbg !46
  %785 = fmul float %784, %780, !dbg !46
  %786 = fsub <1 x float> %757, %759, !dbg !43
  %787 = extractelement <1 x float> %786, i64 0, !dbg !44
  %788 = tail call float @llvm.amdgcn.exp2.f32(float %787), !dbg !44
  %789 = insertelement <4 x float> poison, float %788, i64 0, !dbg !45
  %790 = shufflevector <4 x float> %789, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !45
  %791 = extractvalue [2 x <1 x float>] %158, 1, !dbg !46
  %792 = extractelement <1 x float> %791, i64 0, !dbg !46
  %793 = fmul float %792, %788, !dbg !46
  %794 = extractvalue [2 x <8 x float>] %159, 0, !dbg !45
  %795 = shufflevector <8 x float> %794, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %796 = fmul <4 x float> %795, %782, !dbg !45
  %797 = shufflevector <8 x float> %794, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %798 = fmul <4 x float> %797, %782, !dbg !45
  %799 = extractvalue [2 x <8 x float>] %159, 1, !dbg !45
  %800 = shufflevector <8 x float> %799, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %801 = fmul <4 x float> %800, %790, !dbg !45
  %802 = shufflevector <8 x float> %799, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %803 = fmul <4 x float> %802, %790, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1394 = or disjoint i64 %425, 32, !dbg !17
  %vdb.address1395 = add i64 %vdb.address1394, %424, !dbg !17
  %vdb.address1396 = trunc i64 %vdb.address1395 to i32, !dbg !17
  %vdb.address1397 = shl i32 %vdb.address1396, 1, !dbg !17
  %vdb.address1405 = or i32 %.tr1378, 16448, !dbg !17
  %804 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1397, <4 x i32> %dma.buffer.rsrc1277) #9, !dbg !17
  %805 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1410, i32 %vdb.address1405, <4 x i32> %dma.buffer.rsrc1277) #9, !dbg !17
  %address.local860 = and i32 %address.local851, 120, !dbg !18
  %address.local862 = and i32 %address.local502, 1536, !dbg !18
  %address.local863 = or disjoint i32 %address.local860, %address.local862, !dbg !18
  %address.local864 = zext nneg i32 %address.local863 to i64, !dbg !18
  %address.local865 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local864, !dbg !18
  %806 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local865, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %807 = bitcast <4 x i32> %806 to <8 x half>, !dbg !18
  %address.local866 = and i64 %address.local429, 60, !dbg !18
  %address.local867 = add nuw nsw i64 %address.local866, %address.local429, !dbg !18
  %address.local869.idx = shl nuw nsw i64 %address.local867, 4, !dbg !18
  %address.local869 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local869.idx, !dbg !18
  %address.local870 = lshr i64 %address.local482, 4, !dbg !18
  %address.local872.idx = mul nuw nsw i64 %address.local870, 768, !dbg !18
  %address.local872 = getelementptr i8, ptr addrspace(3) %address.local869, i64 %address.local872.idx, !dbg !18
  %address.local874.idx = mul nsw i64 %209, -64, !dbg !18
  %address.local874 = getelementptr i8, ptr addrspace(3) %address.local872, i64 %address.local874.idx, !dbg !18
  %address.local875 = getelementptr i8, ptr addrspace(3) %address.local874, i64 256, !dbg !18
  %808 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local875, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %809 = bitcast <4 x i32> %808 to <8 x half>, !dbg !18
  %address.local879 = getelementptr i8, ptr addrspace(3) %address.local874, i64 512, !dbg !18
  %810 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local879, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %811 = bitcast <4 x i32> %810 to <8 x half>, !dbg !18
  %address.local883 = getelementptr i8, ptr addrspace(3) %address.local874, i64 768, !dbg !18
  %812 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local883, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %813 = bitcast <4 x i32> %812 to <8 x half>, !dbg !18
  %address.local885 = getelementptr i8, ptr addrspace(3) %address.local865, i64 4096, !dbg !18
  %814 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local885, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %815 = bitcast <4 x i32> %814 to <8 x half>, !dbg !18
  %address.local889 = getelementptr i8, ptr addrspace(3) %address.local874, i64 4352, !dbg !18
  %816 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local889, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %817 = bitcast <4 x i32> %816 to <8 x half>, !dbg !18
  %address.local893 = getelementptr i8, ptr addrspace(3) %address.local874, i64 4608, !dbg !18
  %818 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local893, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %819 = bitcast <4 x i32> %818 to <8 x half>, !dbg !18
  %address.local897 = getelementptr i8, ptr addrspace(3) %address.local874, i64 4864, !dbg !18
  %820 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local897, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %821 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %822 = fsub <4 x float> %760, %761, !dbg !42
  %823 = extractelement <4 x float> %822, i64 0, !dbg !48
  %824 = tail call float @llvm.amdgcn.exp2.f32(float %823), !dbg !48
  %825 = insertelement <4 x float> poison, float %824, i64 0, !dbg !49
  %826 = extractelement <4 x float> %822, i64 1, !dbg !48
  %827 = tail call float @llvm.amdgcn.exp2.f32(float %826), !dbg !48
  %828 = insertelement <4 x float> %825, float %827, i64 1, !dbg !49
  %829 = extractelement <4 x float> %822, i64 2, !dbg !48
  %830 = tail call float @llvm.amdgcn.exp2.f32(float %829), !dbg !48
  %831 = insertelement <4 x float> %828, float %830, i64 2, !dbg !49
  %832 = extractelement <4 x float> %822, i64 3, !dbg !48
  %833 = tail call float @llvm.amdgcn.exp2.f32(float %832), !dbg !48
  %834 = insertelement <4 x float> %831, float %833, i64 3, !dbg !49
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %834) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !49
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !49
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !49
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !49
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !49
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !49
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !49
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !49
  %835 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !49
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %835) #8, !dbg !48, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %821, <4 x float> %796) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %837 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %837, <4 x float> %798) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %839 = fsub <4 x float> %769, %770, !dbg !42
  %840 = extractelement <4 x float> %839, i64 0, !dbg !48
  %841 = tail call float @llvm.amdgcn.exp2.f32(float %840), !dbg !48
  %842 = insertelement <4 x float> poison, float %841, i64 0, !dbg !49
  %843 = extractelement <4 x float> %839, i64 1, !dbg !48
  %844 = tail call float @llvm.amdgcn.exp2.f32(float %843), !dbg !48
  %845 = insertelement <4 x float> %842, float %844, i64 1, !dbg !49
  %846 = extractelement <4 x float> %839, i64 2, !dbg !48
  %847 = tail call float @llvm.amdgcn.exp2.f32(float %846), !dbg !48
  %848 = insertelement <4 x float> %845, float %847, i64 2, !dbg !49
  %849 = extractelement <4 x float> %839, i64 3, !dbg !48
  %850 = tail call float @llvm.amdgcn.exp2.f32(float %849), !dbg !48
  %851 = insertelement <4 x float> %848, float %850, i64 3, !dbg !49
  %math.result315 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %851) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f01970 = extractelement <4 x float> %math.result315, i64 0, !dbg !49
  %.cvtpack.f11971 = extractelement <4 x float> %math.result315, i64 1, !dbg !49
  %.cvtpack.f21972 = extractelement <4 x float> %math.result315, i64 2, !dbg !49
  %.cvtpack.f31973 = extractelement <4 x float> %math.result315, i64 3, !dbg !49
  %.cvtpack.p01974 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01970, float %.cvtpack.f11971) #12, !dbg !49
  %.cvtpack.p11975 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21972, float %.cvtpack.f31973) #12, !dbg !49
  %.cvtpack.v01976 = insertelement <2 x i32> poison, i32 %.cvtpack.p01974, i64 0, !dbg !49
  %.cvtpack.v11977 = insertelement <2 x i32> %.cvtpack.v01976, i32 %.cvtpack.p11975, i64 1, !dbg !49
  %852 = bitcast <2 x i32> %.cvtpack.v11977 to <4 x half>, !dbg !49
  %math.packed316 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %852) #8, !dbg !48, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %853 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %821, <4 x float> %801) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %837, <4 x float> %803) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %855 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %856 = fsub <4 x float> %762, %761, !dbg !42
  %857 = extractelement <4 x float> %856, i64 0, !dbg !48
  %858 = tail call float @llvm.amdgcn.exp2.f32(float %857), !dbg !48
  %859 = fadd float %824, %858, !dbg !50
  %860 = insertelement <4 x float> poison, float %858, i64 0, !dbg !49
  %861 = extractelement <4 x float> %856, i64 1, !dbg !48
  %862 = tail call float @llvm.amdgcn.exp2.f32(float %861), !dbg !48
  %863 = fadd float %827, %862, !dbg !50
  %864 = insertelement <4 x float> %860, float %862, i64 1, !dbg !49
  %865 = extractelement <4 x float> %856, i64 2, !dbg !48
  %866 = tail call float @llvm.amdgcn.exp2.f32(float %865), !dbg !48
  %867 = fadd float %830, %866, !dbg !50
  %868 = insertelement <4 x float> %864, float %866, i64 2, !dbg !49
  %869 = extractelement <4 x float> %856, i64 3, !dbg !48
  %870 = tail call float @llvm.amdgcn.exp2.f32(float %869), !dbg !48
  %871 = fadd float %833, %870, !dbg !50
  %872 = insertelement <4 x float> %868, float %870, i64 3, !dbg !49
  %math.result301 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %872) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f01978 = extractelement <4 x float> %math.result301, i64 0, !dbg !49
  %.cvtpack.f11979 = extractelement <4 x float> %math.result301, i64 1, !dbg !49
  %.cvtpack.f21980 = extractelement <4 x float> %math.result301, i64 2, !dbg !49
  %.cvtpack.f31981 = extractelement <4 x float> %math.result301, i64 3, !dbg !49
  %.cvtpack.p01982 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01978, float %.cvtpack.f11979) #12, !dbg !49
  %.cvtpack.p11983 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21980, float %.cvtpack.f31981) #12, !dbg !49
  %.cvtpack.v01984 = insertelement <2 x i32> poison, i32 %.cvtpack.p01982, i64 0, !dbg !49
  %.cvtpack.v11985 = insertelement <2 x i32> %.cvtpack.v01984, i32 %.cvtpack.p11983, i64 1, !dbg !49
  %873 = bitcast <2 x i32> %.cvtpack.v11985 to <4 x half>, !dbg !49
  %math.packed302 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %873) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %874 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %855, <4 x float> %836) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %875 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %876 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %875, <4 x float> %838) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %877 = fsub <4 x float> %771, %770, !dbg !42
  %878 = extractelement <4 x float> %877, i64 0, !dbg !48
  %879 = tail call float @llvm.amdgcn.exp2.f32(float %878), !dbg !48
  %880 = fadd float %841, %879, !dbg !50
  %881 = insertelement <4 x float> poison, float %879, i64 0, !dbg !49
  %882 = extractelement <4 x float> %877, i64 1, !dbg !48
  %883 = tail call float @llvm.amdgcn.exp2.f32(float %882), !dbg !48
  %884 = fadd float %844, %883, !dbg !50
  %885 = insertelement <4 x float> %881, float %883, i64 1, !dbg !49
  %886 = extractelement <4 x float> %877, i64 2, !dbg !48
  %887 = tail call float @llvm.amdgcn.exp2.f32(float %886), !dbg !48
  %888 = fadd float %847, %887, !dbg !50
  %889 = insertelement <4 x float> %885, float %887, i64 2, !dbg !49
  %890 = extractelement <4 x float> %877, i64 3, !dbg !48
  %891 = tail call float @llvm.amdgcn.exp2.f32(float %890), !dbg !48
  %892 = fadd float %850, %891, !dbg !50
  %893 = insertelement <4 x float> %889, float %891, i64 3, !dbg !49
  %math.result317 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %893) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f01986 = extractelement <4 x float> %math.result317, i64 0, !dbg !49
  %.cvtpack.f11987 = extractelement <4 x float> %math.result317, i64 1, !dbg !49
  %.cvtpack.f21988 = extractelement <4 x float> %math.result317, i64 2, !dbg !49
  %.cvtpack.f31989 = extractelement <4 x float> %math.result317, i64 3, !dbg !49
  %.cvtpack.p01990 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01986, float %.cvtpack.f11987) #12, !dbg !49
  %.cvtpack.p11991 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21988, float %.cvtpack.f31989) #12, !dbg !49
  %.cvtpack.v01992 = insertelement <2 x i32> poison, i32 %.cvtpack.p01990, i64 0, !dbg !49
  %.cvtpack.v11993 = insertelement <2 x i32> %.cvtpack.v01992, i32 %.cvtpack.p11991, i64 1, !dbg !49
  %894 = bitcast <2 x i32> %.cvtpack.v11993 to <4 x half>, !dbg !49
  %math.packed318 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %894) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %855, <4 x float> %853) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %875, <4 x float> %854) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %897 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %898 = fsub <4 x float> %763, %761, !dbg !42
  %899 = extractelement <4 x float> %898, i64 0, !dbg !48
  %900 = tail call float @llvm.amdgcn.exp2.f32(float %899), !dbg !48
  %901 = fadd float %900, %859, !dbg !50
  %902 = insertelement <4 x float> poison, float %900, i64 0, !dbg !49
  %903 = extractelement <4 x float> %898, i64 1, !dbg !48
  %904 = tail call float @llvm.amdgcn.exp2.f32(float %903), !dbg !48
  %905 = fadd float %904, %863, !dbg !50
  %906 = insertelement <4 x float> %902, float %904, i64 1, !dbg !49
  %907 = extractelement <4 x float> %898, i64 2, !dbg !48
  %908 = tail call float @llvm.amdgcn.exp2.f32(float %907), !dbg !48
  %909 = fadd float %908, %867, !dbg !50
  %910 = insertelement <4 x float> %906, float %908, i64 2, !dbg !49
  %911 = extractelement <4 x float> %898, i64 3, !dbg !48
  %912 = tail call float @llvm.amdgcn.exp2.f32(float %911), !dbg !48
  %913 = fadd float %912, %871, !dbg !50
  %914 = insertelement <4 x float> %910, float %912, i64 3, !dbg !49
  %math.result303 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %914) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f01994 = extractelement <4 x float> %math.result303, i64 0, !dbg !49
  %.cvtpack.f11995 = extractelement <4 x float> %math.result303, i64 1, !dbg !49
  %.cvtpack.f21996 = extractelement <4 x float> %math.result303, i64 2, !dbg !49
  %.cvtpack.f31997 = extractelement <4 x float> %math.result303, i64 3, !dbg !49
  %.cvtpack.p01998 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01994, float %.cvtpack.f11995) #12, !dbg !49
  %.cvtpack.p11999 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21996, float %.cvtpack.f31997) #12, !dbg !49
  %.cvtpack.v02000 = insertelement <2 x i32> poison, i32 %.cvtpack.p01998, i64 0, !dbg !49
  %.cvtpack.v12001 = insertelement <2 x i32> %.cvtpack.v02000, i32 %.cvtpack.p11999, i64 1, !dbg !49
  %915 = bitcast <2 x i32> %.cvtpack.v12001 to <4 x half>, !dbg !49
  %math.packed304 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %915) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %897, <4 x float> %874) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %917 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %917, <4 x float> %876) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %919 = fsub <4 x float> %772, %770, !dbg !42
  %920 = extractelement <4 x float> %919, i64 0, !dbg !48
  %921 = tail call float @llvm.amdgcn.exp2.f32(float %920), !dbg !48
  %922 = fadd float %921, %880, !dbg !50
  %923 = insertelement <4 x float> poison, float %921, i64 0, !dbg !49
  %924 = extractelement <4 x float> %919, i64 1, !dbg !48
  %925 = tail call float @llvm.amdgcn.exp2.f32(float %924), !dbg !48
  %926 = fadd float %925, %884, !dbg !50
  %927 = insertelement <4 x float> %923, float %925, i64 1, !dbg !49
  %928 = extractelement <4 x float> %919, i64 2, !dbg !48
  %929 = tail call float @llvm.amdgcn.exp2.f32(float %928), !dbg !48
  %930 = fadd float %929, %888, !dbg !50
  %931 = insertelement <4 x float> %927, float %929, i64 2, !dbg !49
  %932 = extractelement <4 x float> %919, i64 3, !dbg !48
  %933 = tail call float @llvm.amdgcn.exp2.f32(float %932), !dbg !48
  %934 = fadd float %933, %892, !dbg !50
  %935 = insertelement <4 x float> %931, float %933, i64 3, !dbg !49
  %math.result319 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %935) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02002 = extractelement <4 x float> %math.result319, i64 0, !dbg !49
  %.cvtpack.f12003 = extractelement <4 x float> %math.result319, i64 1, !dbg !49
  %.cvtpack.f22004 = extractelement <4 x float> %math.result319, i64 2, !dbg !49
  %.cvtpack.f32005 = extractelement <4 x float> %math.result319, i64 3, !dbg !49
  %.cvtpack.p02006 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02002, float %.cvtpack.f12003) #12, !dbg !49
  %.cvtpack.p12007 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22004, float %.cvtpack.f32005) #12, !dbg !49
  %.cvtpack.v02008 = insertelement <2 x i32> poison, i32 %.cvtpack.p02006, i64 0, !dbg !49
  %.cvtpack.v12009 = insertelement <2 x i32> %.cvtpack.v02008, i32 %.cvtpack.p12007, i64 1, !dbg !49
  %936 = bitcast <2 x i32> %.cvtpack.v12009 to <4 x half>, !dbg !49
  %math.packed320 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %936) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %937 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %897, <4 x float> %895) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %938 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %917, <4 x float> %896) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %939 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %940 = fsub <4 x float> %764, %761, !dbg !42
  %941 = extractelement <4 x float> %940, i64 0, !dbg !48
  %942 = tail call float @llvm.amdgcn.exp2.f32(float %941), !dbg !48
  %943 = fadd float %942, %901, !dbg !50
  %944 = insertelement <4 x float> poison, float %942, i64 0, !dbg !49
  %945 = extractelement <4 x float> %940, i64 1, !dbg !48
  %946 = tail call float @llvm.amdgcn.exp2.f32(float %945), !dbg !48
  %947 = fadd float %946, %905, !dbg !50
  %948 = insertelement <4 x float> %944, float %946, i64 1, !dbg !49
  %949 = extractelement <4 x float> %940, i64 2, !dbg !48
  %950 = tail call float @llvm.amdgcn.exp2.f32(float %949), !dbg !48
  %951 = fadd float %950, %909, !dbg !50
  %952 = insertelement <4 x float> %948, float %950, i64 2, !dbg !49
  %953 = extractelement <4 x float> %940, i64 3, !dbg !48
  %954 = tail call float @llvm.amdgcn.exp2.f32(float %953), !dbg !48
  %955 = fadd float %954, %913, !dbg !50
  %956 = insertelement <4 x float> %952, float %954, i64 3, !dbg !49
  %math.result305 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %956) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02010 = extractelement <4 x float> %math.result305, i64 0, !dbg !49
  %.cvtpack.f12011 = extractelement <4 x float> %math.result305, i64 1, !dbg !49
  %.cvtpack.f22012 = extractelement <4 x float> %math.result305, i64 2, !dbg !49
  %.cvtpack.f32013 = extractelement <4 x float> %math.result305, i64 3, !dbg !49
  %.cvtpack.p02014 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02010, float %.cvtpack.f12011) #12, !dbg !49
  %.cvtpack.p12015 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22012, float %.cvtpack.f32013) #12, !dbg !49
  %.cvtpack.v02016 = insertelement <2 x i32> poison, i32 %.cvtpack.p02014, i64 0, !dbg !49
  %.cvtpack.v12017 = insertelement <2 x i32> %.cvtpack.v02016, i32 %.cvtpack.p12015, i64 1, !dbg !49
  %957 = bitcast <2 x i32> %.cvtpack.v12017 to <4 x half>, !dbg !49
  %math.packed306 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %957) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %958 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %939, <4 x float> %916) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %959 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %959, <4 x float> %918) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %961 = fsub <4 x float> %773, %770, !dbg !42
  %962 = extractelement <4 x float> %961, i64 0, !dbg !48
  %963 = tail call float @llvm.amdgcn.exp2.f32(float %962), !dbg !48
  %964 = fadd float %963, %922, !dbg !50
  %965 = insertelement <4 x float> poison, float %963, i64 0, !dbg !49
  %966 = extractelement <4 x float> %961, i64 1, !dbg !48
  %967 = tail call float @llvm.amdgcn.exp2.f32(float %966), !dbg !48
  %968 = fadd float %967, %926, !dbg !50
  %969 = insertelement <4 x float> %965, float %967, i64 1, !dbg !49
  %970 = extractelement <4 x float> %961, i64 2, !dbg !48
  %971 = tail call float @llvm.amdgcn.exp2.f32(float %970), !dbg !48
  %972 = fadd float %971, %930, !dbg !50
  %973 = insertelement <4 x float> %969, float %971, i64 2, !dbg !49
  %974 = extractelement <4 x float> %961, i64 3, !dbg !48
  %975 = tail call float @llvm.amdgcn.exp2.f32(float %974), !dbg !48
  %976 = fadd float %975, %934, !dbg !50
  %977 = insertelement <4 x float> %973, float %975, i64 3, !dbg !49
  %math.result321 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %977) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02018 = extractelement <4 x float> %math.result321, i64 0, !dbg !49
  %.cvtpack.f12019 = extractelement <4 x float> %math.result321, i64 1, !dbg !49
  %.cvtpack.f22020 = extractelement <4 x float> %math.result321, i64 2, !dbg !49
  %.cvtpack.f32021 = extractelement <4 x float> %math.result321, i64 3, !dbg !49
  %.cvtpack.p02022 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02018, float %.cvtpack.f12019) #12, !dbg !49
  %.cvtpack.p12023 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22020, float %.cvtpack.f32021) #12, !dbg !49
  %.cvtpack.v02024 = insertelement <2 x i32> poison, i32 %.cvtpack.p02022, i64 0, !dbg !49
  %.cvtpack.v12025 = insertelement <2 x i32> %.cvtpack.v02024, i32 %.cvtpack.p12023, i64 1, !dbg !49
  %978 = bitcast <2 x i32> %.cvtpack.v12025 to <4 x half>, !dbg !49
  %math.packed322 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %978) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %979 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %939, <4 x float> %937) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %959, <4 x float> %938) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %981 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %982 = fsub <4 x float> %765, %761, !dbg !42
  %983 = extractelement <4 x float> %982, i64 0, !dbg !48
  %984 = tail call float @llvm.amdgcn.exp2.f32(float %983), !dbg !48
  %985 = fadd float %984, %943, !dbg !50
  %986 = insertelement <4 x float> poison, float %984, i64 0, !dbg !49
  %987 = extractelement <4 x float> %982, i64 1, !dbg !48
  %988 = tail call float @llvm.amdgcn.exp2.f32(float %987), !dbg !48
  %989 = fadd float %988, %947, !dbg !50
  %990 = insertelement <4 x float> %986, float %988, i64 1, !dbg !49
  %991 = extractelement <4 x float> %982, i64 2, !dbg !48
  %992 = tail call float @llvm.amdgcn.exp2.f32(float %991), !dbg !48
  %993 = fadd float %992, %951, !dbg !50
  %994 = insertelement <4 x float> %990, float %992, i64 2, !dbg !49
  %995 = extractelement <4 x float> %982, i64 3, !dbg !48
  %996 = tail call float @llvm.amdgcn.exp2.f32(float %995), !dbg !48
  %997 = fadd float %996, %955, !dbg !50
  %998 = insertelement <4 x float> %994, float %996, i64 3, !dbg !49
  %math.result307 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %998) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02026 = extractelement <4 x float> %math.result307, i64 0, !dbg !49
  %.cvtpack.f12027 = extractelement <4 x float> %math.result307, i64 1, !dbg !49
  %.cvtpack.f22028 = extractelement <4 x float> %math.result307, i64 2, !dbg !49
  %.cvtpack.f32029 = extractelement <4 x float> %math.result307, i64 3, !dbg !49
  %.cvtpack.p02030 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02026, float %.cvtpack.f12027) #12, !dbg !49
  %.cvtpack.p12031 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22028, float %.cvtpack.f32029) #12, !dbg !49
  %.cvtpack.v02032 = insertelement <2 x i32> poison, i32 %.cvtpack.p02030, i64 0, !dbg !49
  %.cvtpack.v12033 = insertelement <2 x i32> %.cvtpack.v02032, i32 %.cvtpack.p12031, i64 1, !dbg !49
  %999 = bitcast <2 x i32> %.cvtpack.v12033 to <4 x half>, !dbg !49
  %math.packed308 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %999) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %981, <4 x float> %958) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1001 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1001, <4 x float> %960) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1003 = fsub <4 x float> %774, %770, !dbg !42
  %1004 = extractelement <4 x float> %1003, i64 0, !dbg !48
  %1005 = tail call float @llvm.amdgcn.exp2.f32(float %1004), !dbg !48
  %1006 = fadd float %1005, %964, !dbg !50
  %1007 = insertelement <4 x float> poison, float %1005, i64 0, !dbg !49
  %1008 = extractelement <4 x float> %1003, i64 1, !dbg !48
  %1009 = tail call float @llvm.amdgcn.exp2.f32(float %1008), !dbg !48
  %1010 = fadd float %1009, %968, !dbg !50
  %1011 = insertelement <4 x float> %1007, float %1009, i64 1, !dbg !49
  %1012 = extractelement <4 x float> %1003, i64 2, !dbg !48
  %1013 = tail call float @llvm.amdgcn.exp2.f32(float %1012), !dbg !48
  %1014 = fadd float %1013, %972, !dbg !50
  %1015 = insertelement <4 x float> %1011, float %1013, i64 2, !dbg !49
  %1016 = extractelement <4 x float> %1003, i64 3, !dbg !48
  %1017 = tail call float @llvm.amdgcn.exp2.f32(float %1016), !dbg !48
  %1018 = fadd float %1017, %976, !dbg !50
  %1019 = insertelement <4 x float> %1015, float %1017, i64 3, !dbg !49
  %math.result323 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1019) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02034 = extractelement <4 x float> %math.result323, i64 0, !dbg !49
  %.cvtpack.f12035 = extractelement <4 x float> %math.result323, i64 1, !dbg !49
  %.cvtpack.f22036 = extractelement <4 x float> %math.result323, i64 2, !dbg !49
  %.cvtpack.f32037 = extractelement <4 x float> %math.result323, i64 3, !dbg !49
  %.cvtpack.p02038 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02034, float %.cvtpack.f12035) #12, !dbg !49
  %.cvtpack.p12039 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22036, float %.cvtpack.f32037) #12, !dbg !49
  %.cvtpack.v02040 = insertelement <2 x i32> poison, i32 %.cvtpack.p02038, i64 0, !dbg !49
  %.cvtpack.v12041 = insertelement <2 x i32> %.cvtpack.v02040, i32 %.cvtpack.p12039, i64 1, !dbg !49
  %1020 = bitcast <2 x i32> %.cvtpack.v12041 to <4 x half>, !dbg !49
  %math.packed324 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1020) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %1021 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %981, <4 x float> %979) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1022 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1001, <4 x float> %980) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1023 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1024 = fsub <4 x float> %766, %761, !dbg !42
  %1025 = extractelement <4 x float> %1024, i64 0, !dbg !48
  %1026 = tail call float @llvm.amdgcn.exp2.f32(float %1025), !dbg !48
  %1027 = fadd float %1026, %985, !dbg !50
  %1028 = insertelement <4 x float> poison, float %1026, i64 0, !dbg !49
  %1029 = extractelement <4 x float> %1024, i64 1, !dbg !48
  %1030 = tail call float @llvm.amdgcn.exp2.f32(float %1029), !dbg !48
  %1031 = fadd float %1030, %989, !dbg !50
  %1032 = insertelement <4 x float> %1028, float %1030, i64 1, !dbg !49
  %1033 = extractelement <4 x float> %1024, i64 2, !dbg !48
  %1034 = tail call float @llvm.amdgcn.exp2.f32(float %1033), !dbg !48
  %1035 = fadd float %1034, %993, !dbg !50
  %1036 = insertelement <4 x float> %1032, float %1034, i64 2, !dbg !49
  %1037 = extractelement <4 x float> %1024, i64 3, !dbg !48
  %1038 = tail call float @llvm.amdgcn.exp2.f32(float %1037), !dbg !48
  %1039 = fadd float %1038, %997, !dbg !50
  %1040 = insertelement <4 x float> %1036, float %1038, i64 3, !dbg !49
  %math.result309 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1040) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02042 = extractelement <4 x float> %math.result309, i64 0, !dbg !49
  %.cvtpack.f12043 = extractelement <4 x float> %math.result309, i64 1, !dbg !49
  %.cvtpack.f22044 = extractelement <4 x float> %math.result309, i64 2, !dbg !49
  %.cvtpack.f32045 = extractelement <4 x float> %math.result309, i64 3, !dbg !49
  %.cvtpack.p02046 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02042, float %.cvtpack.f12043) #12, !dbg !49
  %.cvtpack.p12047 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22044, float %.cvtpack.f32045) #12, !dbg !49
  %.cvtpack.v02048 = insertelement <2 x i32> poison, i32 %.cvtpack.p02046, i64 0, !dbg !49
  %.cvtpack.v12049 = insertelement <2 x i32> %.cvtpack.v02048, i32 %.cvtpack.p12047, i64 1, !dbg !49
  %1041 = bitcast <2 x i32> %.cvtpack.v12049 to <4 x half>, !dbg !49
  %math.packed310 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1041) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %1042 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1023, <4 x float> %1000) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1043 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1044 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1043, <4 x float> %1002) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1045 = fsub <4 x float> %775, %770, !dbg !42
  %1046 = extractelement <4 x float> %1045, i64 0, !dbg !48
  %1047 = tail call float @llvm.amdgcn.exp2.f32(float %1046), !dbg !48
  %1048 = fadd float %1047, %1006, !dbg !50
  %1049 = insertelement <4 x float> poison, float %1047, i64 0, !dbg !49
  %1050 = extractelement <4 x float> %1045, i64 1, !dbg !48
  %1051 = tail call float @llvm.amdgcn.exp2.f32(float %1050), !dbg !48
  %1052 = fadd float %1051, %1010, !dbg !50
  %1053 = insertelement <4 x float> %1049, float %1051, i64 1, !dbg !49
  %1054 = extractelement <4 x float> %1045, i64 2, !dbg !48
  %1055 = tail call float @llvm.amdgcn.exp2.f32(float %1054), !dbg !48
  %1056 = fadd float %1055, %1014, !dbg !50
  %1057 = insertelement <4 x float> %1053, float %1055, i64 2, !dbg !49
  %1058 = extractelement <4 x float> %1045, i64 3, !dbg !48
  %1059 = tail call float @llvm.amdgcn.exp2.f32(float %1058), !dbg !48
  %1060 = fadd float %1059, %1018, !dbg !50
  %1061 = insertelement <4 x float> %1057, float %1059, i64 3, !dbg !49
  %math.result325 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1061) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02050 = extractelement <4 x float> %math.result325, i64 0, !dbg !49
  %.cvtpack.f12051 = extractelement <4 x float> %math.result325, i64 1, !dbg !49
  %.cvtpack.f22052 = extractelement <4 x float> %math.result325, i64 2, !dbg !49
  %.cvtpack.f32053 = extractelement <4 x float> %math.result325, i64 3, !dbg !49
  %.cvtpack.p02054 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02050, float %.cvtpack.f12051) #12, !dbg !49
  %.cvtpack.p12055 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22052, float %.cvtpack.f32053) #12, !dbg !49
  %.cvtpack.v02056 = insertelement <2 x i32> poison, i32 %.cvtpack.p02054, i64 0, !dbg !49
  %.cvtpack.v12057 = insertelement <2 x i32> %.cvtpack.v02056, i32 %.cvtpack.p12055, i64 1, !dbg !49
  %1062 = bitcast <2 x i32> %.cvtpack.v12057 to <4 x half>, !dbg !49
  %math.packed326 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1062) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %1063 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1023, <4 x float> %1021) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1064 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1043, <4 x float> %1022) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1065 = shufflevector <8 x half> %819, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1066 = fsub <4 x float> %767, %761, !dbg !42
  %1067 = extractelement <4 x float> %1066, i64 0, !dbg !48
  %1068 = tail call float @llvm.amdgcn.exp2.f32(float %1067), !dbg !48
  %1069 = fadd float %1068, %1027, !dbg !50
  %1070 = insertelement <4 x float> poison, float %1068, i64 0, !dbg !49
  %1071 = extractelement <4 x float> %1066, i64 1, !dbg !48
  %1072 = tail call float @llvm.amdgcn.exp2.f32(float %1071), !dbg !48
  %1073 = fadd float %1072, %1031, !dbg !50
  %1074 = insertelement <4 x float> %1070, float %1072, i64 1, !dbg !49
  %1075 = extractelement <4 x float> %1066, i64 2, !dbg !48
  %1076 = tail call float @llvm.amdgcn.exp2.f32(float %1075), !dbg !48
  %1077 = fadd float %1076, %1035, !dbg !50
  %1078 = insertelement <4 x float> %1074, float %1076, i64 2, !dbg !49
  %1079 = extractelement <4 x float> %1066, i64 3, !dbg !48
  %1080 = tail call float @llvm.amdgcn.exp2.f32(float %1079), !dbg !48
  %1081 = fadd float %1080, %1039, !dbg !50
  %1082 = insertelement <4 x float> %1078, float %1080, i64 3, !dbg !49
  %math.result311 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1082) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02058 = extractelement <4 x float> %math.result311, i64 0, !dbg !49
  %.cvtpack.f12059 = extractelement <4 x float> %math.result311, i64 1, !dbg !49
  %.cvtpack.f22060 = extractelement <4 x float> %math.result311, i64 2, !dbg !49
  %.cvtpack.f32061 = extractelement <4 x float> %math.result311, i64 3, !dbg !49
  %.cvtpack.p02062 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02058, float %.cvtpack.f12059) #12, !dbg !49
  %.cvtpack.p12063 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22060, float %.cvtpack.f32061) #12, !dbg !49
  %.cvtpack.v02064 = insertelement <2 x i32> poison, i32 %.cvtpack.p02062, i64 0, !dbg !49
  %.cvtpack.v12065 = insertelement <2 x i32> %.cvtpack.v02064, i32 %.cvtpack.p12063, i64 1, !dbg !49
  %1083 = bitcast <2 x i32> %.cvtpack.v12065 to <4 x half>, !dbg !49
  %math.packed312 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1083) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1065, <4 x float> %1042) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1085 = shufflevector <8 x half> %819, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1085, <4 x float> %1044) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1087 = fsub <4 x float> %776, %770, !dbg !42
  %1088 = extractelement <4 x float> %1087, i64 0, !dbg !48
  %1089 = tail call float @llvm.amdgcn.exp2.f32(float %1088), !dbg !48
  %1090 = fadd float %1089, %1048, !dbg !50
  %1091 = insertelement <4 x float> poison, float %1089, i64 0, !dbg !49
  %1092 = extractelement <4 x float> %1087, i64 1, !dbg !48
  %1093 = tail call float @llvm.amdgcn.exp2.f32(float %1092), !dbg !48
  %1094 = fadd float %1093, %1052, !dbg !50
  %1095 = insertelement <4 x float> %1091, float %1093, i64 1, !dbg !49
  %1096 = extractelement <4 x float> %1087, i64 2, !dbg !48
  %1097 = tail call float @llvm.amdgcn.exp2.f32(float %1096), !dbg !48
  %1098 = fadd float %1097, %1056, !dbg !50
  %1099 = insertelement <4 x float> %1095, float %1097, i64 2, !dbg !49
  %1100 = extractelement <4 x float> %1087, i64 3, !dbg !48
  %1101 = tail call float @llvm.amdgcn.exp2.f32(float %1100), !dbg !48
  %1102 = fadd float %1101, %1060, !dbg !50
  %1103 = insertelement <4 x float> %1099, float %1101, i64 3, !dbg !49
  %math.result327 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1103) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02066 = extractelement <4 x float> %math.result327, i64 0, !dbg !49
  %.cvtpack.f12067 = extractelement <4 x float> %math.result327, i64 1, !dbg !49
  %.cvtpack.f22068 = extractelement <4 x float> %math.result327, i64 2, !dbg !49
  %.cvtpack.f32069 = extractelement <4 x float> %math.result327, i64 3, !dbg !49
  %.cvtpack.p02070 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02066, float %.cvtpack.f12067) #12, !dbg !49
  %.cvtpack.p12071 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22068, float %.cvtpack.f32069) #12, !dbg !49
  %.cvtpack.v02072 = insertelement <2 x i32> poison, i32 %.cvtpack.p02070, i64 0, !dbg !49
  %.cvtpack.v12073 = insertelement <2 x i32> %.cvtpack.v02072, i32 %.cvtpack.p12071, i64 1, !dbg !49
  %1104 = bitcast <2 x i32> %.cvtpack.v12073 to <4 x half>, !dbg !49
  %math.packed328 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1104) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %1105 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1065, <4 x float> %1063) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1085, <4 x float> %1064) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1107 = bitcast <4 x i32> %820 to <8 x half>, !dbg !18
  %1108 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1109 = fsub <4 x float> %768, %761, !dbg !42
  %1110 = extractelement <4 x float> %1109, i64 0, !dbg !48
  %1111 = tail call float @llvm.amdgcn.exp2.f32(float %1110), !dbg !48
  %1112 = fadd float %1111, %1069, !dbg !50
  %1113 = insertelement <4 x float> poison, float %1111, i64 0, !dbg !49
  %1114 = extractelement <4 x float> %1109, i64 1, !dbg !48
  %1115 = tail call float @llvm.amdgcn.exp2.f32(float %1114), !dbg !48
  %1116 = fadd float %1115, %1073, !dbg !50
  %1117 = fadd float %1112, %1116, !dbg !50
  %1118 = insertelement <4 x float> %1113, float %1115, i64 1, !dbg !49
  %1119 = extractelement <4 x float> %1109, i64 2, !dbg !48
  %1120 = tail call float @llvm.amdgcn.exp2.f32(float %1119), !dbg !48
  %1121 = fadd float %1120, %1077, !dbg !50
  %1122 = insertelement <4 x float> %1118, float %1120, i64 2, !dbg !49
  %1123 = extractelement <4 x float> %1109, i64 3, !dbg !48
  %1124 = tail call float @llvm.amdgcn.exp2.f32(float %1123), !dbg !48
  %1125 = fadd float %1124, %1081, !dbg !50
  %1126 = fadd float %1121, %1125, !dbg !50
  %1127 = fadd float %1117, %1126, !dbg !50
  %1128 = fadd float %785, %1127, !dbg !50
  %1129 = insertelement <1 x float> poison, float %1128, i64 0, !dbg !12
  %1130 = insertvalue [2 x <1 x float>] poison, <1 x float> %1129, 0, !dbg !12
  %1131 = insertelement <4 x float> %1122, float %1124, i64 3, !dbg !49
  %math.result313 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1131) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02074 = extractelement <4 x float> %math.result313, i64 0, !dbg !49
  %.cvtpack.f12075 = extractelement <4 x float> %math.result313, i64 1, !dbg !49
  %.cvtpack.f22076 = extractelement <4 x float> %math.result313, i64 2, !dbg !49
  %.cvtpack.f32077 = extractelement <4 x float> %math.result313, i64 3, !dbg !49
  %.cvtpack.p02078 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02074, float %.cvtpack.f12075) #12, !dbg !49
  %.cvtpack.p12079 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22076, float %.cvtpack.f32077) #12, !dbg !49
  %.cvtpack.v02080 = insertelement <2 x i32> poison, i32 %.cvtpack.p02078, i64 0, !dbg !49
  %.cvtpack.v12081 = insertelement <2 x i32> %.cvtpack.v02080, i32 %.cvtpack.p12079, i64 1, !dbg !49
  %1132 = bitcast <2 x i32> %.cvtpack.v12081 to <4 x half>, !dbg !49
  %math.packed314 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1132) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %1133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1108, <4 x float> %1084) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1134 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1135 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1134, <4 x float> %1086) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1136 = fsub <4 x float> %777, %770, !dbg !42
  %1137 = extractelement <4 x float> %1136, i64 0, !dbg !48
  %1138 = tail call float @llvm.amdgcn.exp2.f32(float %1137), !dbg !48
  %1139 = fadd float %1138, %1090, !dbg !50
  %1140 = insertelement <4 x float> poison, float %1138, i64 0, !dbg !49
  %1141 = extractelement <4 x float> %1136, i64 1, !dbg !48
  %1142 = tail call float @llvm.amdgcn.exp2.f32(float %1141), !dbg !48
  %1143 = fadd float %1142, %1094, !dbg !50
  %1144 = fadd float %1139, %1143, !dbg !50
  %1145 = insertelement <4 x float> %1140, float %1142, i64 1, !dbg !49
  %1146 = extractelement <4 x float> %1136, i64 2, !dbg !48
  %1147 = tail call float @llvm.amdgcn.exp2.f32(float %1146), !dbg !48
  %1148 = fadd float %1147, %1098, !dbg !50
  %1149 = insertelement <4 x float> %1145, float %1147, i64 2, !dbg !49
  %1150 = extractelement <4 x float> %1136, i64 3, !dbg !48
  %1151 = tail call float @llvm.amdgcn.exp2.f32(float %1150), !dbg !48
  %1152 = fadd float %1151, %1102, !dbg !50
  %1153 = fadd float %1148, %1152, !dbg !50
  %1154 = fadd float %1144, %1153, !dbg !50
  %1155 = fadd float %793, %1154, !dbg !50
  %1156 = insertelement <1 x float> poison, float %1155, i64 0, !dbg !12
  %1157 = insertvalue [2 x <1 x float>] %1130, <1 x float> %1156, 1, !dbg !12
  %1158 = insertelement <4 x float> %1149, float %1151, i64 3, !dbg !49
  %math.result329 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1158) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f02082 = extractelement <4 x float> %math.result329, i64 0, !dbg !49
  %.cvtpack.f12083 = extractelement <4 x float> %math.result329, i64 1, !dbg !49
  %.cvtpack.f22084 = extractelement <4 x float> %math.result329, i64 2, !dbg !49
  %.cvtpack.f32085 = extractelement <4 x float> %math.result329, i64 3, !dbg !49
  %.cvtpack.p02086 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02082, float %.cvtpack.f12083) #12, !dbg !49
  %.cvtpack.p12087 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22084, float %.cvtpack.f32085) #12, !dbg !49
  %.cvtpack.v02088 = insertelement <2 x i32> poison, i32 %.cvtpack.p02086, i64 0, !dbg !49
  %.cvtpack.v12089 = insertelement <2 x i32> %.cvtpack.v02088, i32 %.cvtpack.p12087, i64 1, !dbg !49
  %1159 = bitcast <2 x i32> %.cvtpack.v12089 to <4 x half>, !dbg !49
  %math.packed330 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1159) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18, !frisk.softmax_fragments !5
  %1160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1108, <4 x float> %1105) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1134, <4 x float> %1106) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1162 = extractvalue [2 x <8 x float>] %156, 0, !dbg !45
  %1163 = shufflevector <8 x float> %1162, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1164 = fmul <4 x float> %1163, %782, !dbg !45
  %1165 = shufflevector <8 x float> %1162, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1166 = fmul <4 x float> %1165, %782, !dbg !45
  %1167 = extractvalue [2 x <8 x float>] %156, 1, !dbg !45
  %1168 = shufflevector <8 x float> %1167, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1169 = fmul <4 x float> %1168, %790, !dbg !45
  %1170 = shufflevector <8 x float> %1167, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1171 = fmul <4 x float> %1170, %790, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1418 = or disjoint i64 %425, 64, !dbg !17
  %vdb.address1419 = add i64 %vdb.address1418, %424, !dbg !17
  %vdb.address1420 = trunc i64 %vdb.address1419 to i32, !dbg !17
  %vdb.address1421 = shl i32 %vdb.address1420, 1, !dbg !17
  %vdb.address1429 = or i32 %.tr1378, 16512, !dbg !17
  %1172 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %vdb.address1421, <4 x i32> %dma.buffer.rsrc1277) #9, !dbg !17
  %1173 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %vdb.address1429, <4 x i32> %dma.buffer.rsrc1277) #9, !dbg !17
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local865, i64 8192, !dbg !18
  %1174 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1175 = bitcast <4 x i32> %1174 to <8 x half>, !dbg !18
  %vdb.read1411 = getelementptr i8, ptr addrspace(3) %address.local874, i64 8448, !dbg !18
  %1176 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1411, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1177 = bitcast <4 x i32> %1176 to <8 x half>, !dbg !18
  %vdb.read1412 = getelementptr i8, ptr addrspace(3) %address.local874, i64 8704, !dbg !18
  %1178 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1412, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1179 = bitcast <4 x i32> %1178 to <8 x half>, !dbg !18
  %vdb.read1413 = getelementptr i8, ptr addrspace(3) %address.local874, i64 8960, !dbg !18
  %1180 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1413, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1181 = bitcast <4 x i32> %1180 to <8 x half>, !dbg !18
  %vdb.read1414 = getelementptr i8, ptr addrspace(3) %address.local865, i64 12288, !dbg !18
  %1182 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1414, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1183 = bitcast <4 x i32> %1182 to <8 x half>, !dbg !18
  %vdb.read1415 = getelementptr i8, ptr addrspace(3) %address.local874, i64 12544, !dbg !18
  %1184 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1415, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1185 = bitcast <4 x i32> %1184 to <8 x half>, !dbg !18
  %vdb.read1416 = getelementptr i8, ptr addrspace(3) %address.local874, i64 12800, !dbg !18
  %1186 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1416, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1187 = bitcast <4 x i32> %1186 to <8 x half>, !dbg !18
  %vdb.read1417 = getelementptr i8, ptr addrspace(3) %address.local874, i64 13056, !dbg !18
  %1188 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1417, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1189 = shufflevector <8 x half> %1175, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1189, <4 x float> %1164) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1191 = shufflevector <8 x half> %1175, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1191, <4 x float> %1166) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1189, <4 x float> %1169) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1191, <4 x float> %1171) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1195 = shufflevector <8 x half> %1177, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1195, <4 x float> %1190) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1197 = shufflevector <8 x half> %1177, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1197, <4 x float> %1192) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1195, <4 x float> %1193) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1197, <4 x float> %1194) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1201 = shufflevector <8 x half> %1179, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1201, <4 x float> %1196) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1203 = shufflevector <8 x half> %1179, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1203, <4 x float> %1198) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1205 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1201, <4 x float> %1199) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1203, <4 x float> %1200) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1207 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1207, <4 x float> %1202) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1209 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1209, <4 x float> %1204) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1207, <4 x float> %1205) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1209, <4 x float> %1206) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1213 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1213, <4 x float> %1208) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1215 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1215, <4 x float> %1210) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1213, <4 x float> %1211) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1215, <4 x float> %1212) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1219 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1219, <4 x float> %1214) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1221 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1221, <4 x float> %1216) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1219, <4 x float> %1217) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1221, <4 x float> %1218) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1225 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1225, <4 x float> %1220) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1227 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1227, <4 x float> %1222) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1225, <4 x float> %1223) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1227, <4 x float> %1224) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1231 = bitcast <4 x i32> %1188 to <8 x half>, !dbg !18
  %1232 = shufflevector <8 x half> %1231, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1232, <4 x float> %1226) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1234 = shufflevector <8 x half> %1231, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1234, <4 x float> %1228) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1232, <4 x float> %1229) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1234, <4 x float> %1230) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1238 = extractvalue [2 x <8 x float>] %155, 0, !dbg !45
  %1239 = shufflevector <8 x float> %1238, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1240 = fmul <4 x float> %1239, %782, !dbg !45
  %1241 = shufflevector <8 x float> %1238, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1242 = fmul <4 x float> %1241, %782, !dbg !45
  %1243 = extractvalue [2 x <8 x float>] %155, 1, !dbg !45
  %1244 = shufflevector <8 x float> %1243, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1245 = fmul <4 x float> %1244, %790, !dbg !45
  %1246 = shufflevector <8 x float> %1243, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1247 = fmul <4 x float> %1246, %790, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1434 = or disjoint i64 %425, 96, !dbg !17
  %vdb.address1435 = add i64 %vdb.address1434, %424, !dbg !17
  %vdb.address1436 = trunc i64 %vdb.address1435 to i32, !dbg !17
  %vdb.address1437 = shl i32 %vdb.address1436, 1, !dbg !17
  %vdb.address1445 = or i32 %.tr1378, 16576, !dbg !17
  %1248 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1437, <4 x i32> %dma.buffer.rsrc1277) #9, !dbg !17
  %1249 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1410, i32 %vdb.address1445, <4 x i32> %dma.buffer.rsrc1277) #9, !dbg !17
  %1250 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local865, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1251 = bitcast <4 x i32> %1250 to <8 x half>, !dbg !18
  %1252 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local875, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1253 = bitcast <4 x i32> %1252 to <8 x half>, !dbg !18
  %1254 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local879, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1255 = bitcast <4 x i32> %1254 to <8 x half>, !dbg !18
  %1256 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local883, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1257 = bitcast <4 x i32> %1256 to <8 x half>, !dbg !18
  %1258 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local885, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1259 = bitcast <4 x i32> %1258 to <8 x half>, !dbg !18
  %1260 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local889, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1261 = bitcast <4 x i32> %1260 to <8 x half>, !dbg !18
  %1262 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local893, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1263 = bitcast <4 x i32> %1262 to <8 x half>, !dbg !18
  %1264 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local897, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1265 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1265, <4 x float> %1240) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1267 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1267, <4 x float> %1242) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1265, <4 x float> %1245) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1267, <4 x float> %1247) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1271 = shufflevector <8 x half> %1253, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1271, <4 x float> %1266) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1273 = shufflevector <8 x half> %1253, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1273, <4 x float> %1268) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1271, <4 x float> %1269) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1273, <4 x float> %1270) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1277 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1277, <4 x float> %1272) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1279 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1279, <4 x float> %1274) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1277, <4 x float> %1275) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1279, <4 x float> %1276) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1283 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1283, <4 x float> %1278) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1285 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1285, <4 x float> %1280) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1283, <4 x float> %1281) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1285, <4 x float> %1282) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1289 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1289, <4 x float> %1284) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1291 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1291, <4 x float> %1286) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1289, <4 x float> %1287) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1291, <4 x float> %1288) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1295 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1295, <4 x float> %1290) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1297 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1297, <4 x float> %1292) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1295, <4 x float> %1293) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1297, <4 x float> %1294) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1301 = shufflevector <8 x half> %1263, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1301, <4 x float> %1296) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1303 = shufflevector <8 x half> %1263, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1303, <4 x float> %1298) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1301, <4 x float> %1299) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1303, <4 x float> %1300) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1307 = bitcast <4 x i32> %1264 to <8 x half>, !dbg !18
  %1308 = shufflevector <8 x half> %1307, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1308, <4 x float> %1302) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1310 = shufflevector <8 x half> %1307, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1310, <4 x float> %1304) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1308, <4 x float> %1305) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1310, <4 x float> %1306) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1314 = extractvalue [2 x <8 x float>] %154, 0, !dbg !45
  %1315 = shufflevector <8 x float> %1314, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1316 = fmul <4 x float> %1315, %782, !dbg !45
  %1317 = shufflevector <8 x float> %1314, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1318 = fmul <4 x float> %1317, %782, !dbg !45
  %1319 = extractvalue [2 x <8 x float>] %154, 1, !dbg !45
  %1320 = shufflevector <8 x float> %1319, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1321 = fmul <4 x float> %1320, %790, !dbg !45
  %1322 = shufflevector <8 x float> %1319, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1323 = fmul <4 x float> %1322, %790, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1324 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1325 = bitcast <4 x i32> %1324 to <8 x half>, !dbg !18
  %1326 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1411, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1327 = bitcast <4 x i32> %1326 to <8 x half>, !dbg !18
  %1328 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1412, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1329 = bitcast <4 x i32> %1328 to <8 x half>, !dbg !18
  %1330 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1413, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1331 = bitcast <4 x i32> %1330 to <8 x half>, !dbg !18
  %1332 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1414, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1333 = bitcast <4 x i32> %1332 to <8 x half>, !dbg !18
  %1334 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1415, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1335 = bitcast <4 x i32> %1334 to <8 x half>, !dbg !18
  %1336 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1416, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1337 = bitcast <4 x i32> %1336 to <8 x half>, !dbg !18
  %1338 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1417, i32 0) #11, !dbg !18, !frisk.v_double_buffer !47
  %1339 = shufflevector <8 x half> %1325, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1339, <4 x float> %1316) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1341 = shufflevector <8 x half> %1325, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1341, <4 x float> %1318) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1339, <4 x float> %1321) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1341, <4 x float> %1323) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1345 = shufflevector <8 x half> %1327, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1345, <4 x float> %1340) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1347 = shufflevector <8 x half> %1327, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1347, <4 x float> %1342) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1345, <4 x float> %1343) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1347, <4 x float> %1344) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1351 = shufflevector <8 x half> %1329, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1351, <4 x float> %1346) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1353 = shufflevector <8 x half> %1329, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1353, <4 x float> %1348) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1351, <4 x float> %1349) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1353, <4 x float> %1350) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1357 = shufflevector <8 x half> %1331, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1357, <4 x float> %1352) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1359 = shufflevector <8 x half> %1331, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1359, <4 x float> %1354) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1357, <4 x float> %1355) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1359, <4 x float> %1356) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1363 = shufflevector <8 x half> %1333, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1363, <4 x float> %1358) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1365 = shufflevector <8 x half> %1333, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1365, <4 x float> %1360) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1363, <4 x float> %1361) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1365, <4 x float> %1362) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1369 = shufflevector <8 x half> %1335, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1369, <4 x float> %1364) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1371 = shufflevector <8 x half> %1335, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1371, <4 x float> %1366) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1369, <4 x float> %1367) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1371, <4 x float> %1368) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1375 = shufflevector <8 x half> %1337, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1376 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1375, <4 x float> %1370) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1377 = shufflevector <8 x half> %1337, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1377, <4 x float> %1372) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1375, <4 x float> %1373) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1377, <4 x float> %1374) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1381 = bitcast <4 x i32> %1338 to <8 x half>, !dbg !18
  %1382 = shufflevector <8 x half> %1381, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1382, <4 x float> %1376) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1384 = shufflevector <8 x half> %1381, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1384, <4 x float> %1378) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1382, <4 x float> %1379) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !18
  %1387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1384, <4 x float> %1380) #8, !dbg !18, !frisk.pv_interleave !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %1388 = shufflevector <4 x float> %1133, <4 x float> %1135, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1389 = insertvalue [2 x <8 x float>] poison, <8 x float> %1388, 0, !dbg !12
  %1390 = shufflevector <4 x float> %1160, <4 x float> %1161, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1391 = insertvalue [2 x <8 x float>] %1389, <8 x float> %1390, 1, !dbg !12
  %1392 = insertvalue [2 x <1 x float>] poison, <1 x float> %756, 0, !dbg !12
  %1393 = insertvalue [2 x <1 x float>] %1392, <1 x float> %759, 1, !dbg !12
  %1394 = shufflevector <4 x float> %1233, <4 x float> %1235, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1395 = insertvalue [2 x <8 x float>] poison, <8 x float> %1394, 0, !dbg !12
  %1396 = shufflevector <4 x float> %1236, <4 x float> %1237, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1397 = insertvalue [2 x <8 x float>] %1395, <8 x float> %1396, 1, !dbg !12
  %1398 = shufflevector <4 x float> %1309, <4 x float> %1311, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1399 = insertvalue [2 x <8 x float>] poison, <8 x float> %1398, 0, !dbg !12
  %1400 = shufflevector <4 x float> %1312, <4 x float> %1313, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1401 = insertvalue [2 x <8 x float>] %1399, <8 x float> %1400, 1, !dbg !12
  %1402 = shufflevector <4 x float> %1383, <4 x float> %1385, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1403 = insertvalue [2 x <8 x float>] poison, <8 x float> %1402, 0, !dbg !12
  %1404 = shufflevector <4 x float> %1386, <4 x float> %1387, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1405 = insertvalue [2 x <8 x float>] %1403, <8 x float> %1404, 1, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %1406 = add nuw nsw i64 %160, 1, !dbg !12
  %1407 = icmp slt i64 %1406, %114, !dbg !12
  br i1 %1407, label %153, label %._crit_edge, !dbg !12

._crit_edge:                                      ; preds = %153, %4
  %.lcssa7 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1391, %153 ], !dbg !12
  %.lcssa6 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %1157, %153 ], !dbg !12
  %.lcssa5 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1397, %153 ], !dbg !12
  %.lcssa4 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1401, %153 ], !dbg !12
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1405, %153 ], !dbg !12
  %1408 = extractvalue [2 x <1 x float>] %.lcssa6, 1, !dbg !12
  %bc = bitcast <1 x float> %1408 to <1 x i32>, !dbg !12
  %1409 = extractelement <1 x i32> %bc, i64 0, !dbg !12
  %1410 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1409), !dbg !12
  %1411 = extractvalue [2 x <1 x float>] %.lcssa6, 0, !dbg !12
  %bc47 = bitcast <1 x float> %1411 to <1 x i32>, !dbg !12
  %1412 = extractelement <1 x i32> %bc47, i64 0, !dbg !12
  %1413 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1412), !dbg !12
  %1414 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %1414, 0, !dbg !12
  %1415 = and i32 %kdma.tid, 32, !dbg !12
  %.not40 = icmp eq i32 %1415, 0, !dbg !12
  %1416 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !12
  %1417 = extractelement <8 x float> %1416, i64 0, !dbg !12
  %1418 = extractelement <8 x float> %1416, i64 1, !dbg !12
  %1419 = extractelement <8 x float> %1416, i64 2, !dbg !12
  %1420 = extractelement <8 x float> %1416, i64 3, !dbg !12
  %1421 = select i1 %.not, float %1418, float %1417, !dbg !12
  %1422 = bitcast float %1421 to i32, !dbg !12
  %1423 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1422), !dbg !12
  %1424 = bitcast i32 %1423 to float, !dbg !12
  %1425 = select i1 %.not, float %1417, float %1424, !dbg !12
  %1426 = select i1 %.not, float %1424, float %1418, !dbg !12
  %1427 = select i1 %.not, float %1420, float %1419, !dbg !12
  %1428 = bitcast float %1427 to i32, !dbg !12
  %1429 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1428), !dbg !12
  %1430 = bitcast i32 %1429 to float, !dbg !12
  %1431 = select i1 %.not, float %1419, float %1430, !dbg !12
  %1432 = select i1 %.not, float %1430, float %1420, !dbg !12
  %1433 = select i1 %.not40, float %1431, float %1425, !dbg !12
  %1434 = bitcast float %1433 to i32, !dbg !12
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1434), !dbg !12
  %1436 = bitcast i32 %1435 to float, !dbg !12
  %1437 = select i1 %.not40, float %1432, float %1426, !dbg !12
  %1438 = bitcast float %1437 to i32, !dbg !12
  %1439 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1438), !dbg !12
  %1440 = bitcast i32 %1439 to float, !dbg !12
  %1441 = insertelement <4 x i1> poison, i1 %.not40, i64 0, !dbg !12
  %1442 = shufflevector <4 x i1> %1441, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %1443 = insertelement <4 x float> poison, float %1425, i64 0, !dbg !12
  %1444 = insertelement <4 x float> %1443, float %1426, i64 1, !dbg !12
  %1445 = insertelement <4 x float> %1444, float %1436, i64 2, !dbg !12
  %1446 = insertelement <4 x float> %1445, float %1440, i64 3, !dbg !12
  %1447 = insertelement <4 x float> poison, float %1436, i64 0, !dbg !12
  %1448 = insertelement <4 x float> %1447, float %1440, i64 1, !dbg !12
  %1449 = insertelement <4 x float> %1448, float %1431, i64 2, !dbg !12
  %1450 = insertelement <4 x float> %1449, float %1432, i64 3, !dbg !12
  %1451 = select <4 x i1> %1442, <4 x float> %1446, <4 x float> %1450, !dbg !12
  %1452 = extractelement <8 x float> %1416, i64 4, !dbg !12
  %1453 = extractelement <8 x float> %1416, i64 5, !dbg !12
  %1454 = extractelement <8 x float> %1416, i64 6, !dbg !12
  %1455 = extractelement <8 x float> %1416, i64 7, !dbg !12
  %1456 = select i1 %.not, float %1453, float %1452, !dbg !12
  %1457 = bitcast float %1456 to i32, !dbg !12
  %1458 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1457), !dbg !12
  %1459 = bitcast i32 %1458 to float, !dbg !12
  %1460 = select i1 %.not, float %1452, float %1459, !dbg !12
  %1461 = select i1 %.not, float %1459, float %1453, !dbg !12
  %1462 = select i1 %.not, float %1455, float %1454, !dbg !12
  %1463 = bitcast float %1462 to i32, !dbg !12
  %1464 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1463), !dbg !12
  %1465 = bitcast i32 %1464 to float, !dbg !12
  %1466 = select i1 %.not, float %1454, float %1465, !dbg !12
  %1467 = select i1 %.not, float %1465, float %1455, !dbg !12
  %1468 = select i1 %.not40, float %1466, float %1460, !dbg !12
  %1469 = bitcast float %1468 to i32, !dbg !12
  %1470 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1469), !dbg !12
  %1471 = bitcast i32 %1470 to float, !dbg !12
  %1472 = select i1 %.not40, float %1467, float %1461, !dbg !12
  %1473 = bitcast float %1472 to i32, !dbg !12
  %1474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1473), !dbg !12
  %1475 = bitcast i32 %1474 to float, !dbg !12
  %1476 = insertelement <4 x float> poison, float %1460, i64 0, !dbg !12
  %1477 = insertelement <4 x float> %1476, float %1461, i64 1, !dbg !12
  %1478 = insertelement <4 x float> %1477, float %1471, i64 2, !dbg !12
  %1479 = insertelement <4 x float> %1478, float %1475, i64 3, !dbg !12
  %1480 = insertelement <4 x float> poison, float %1471, i64 0, !dbg !12
  %1481 = insertelement <4 x float> %1480, float %1475, i64 1, !dbg !12
  %1482 = insertelement <4 x float> %1481, float %1466, i64 2, !dbg !12
  %1483 = insertelement <4 x float> %1482, float %1467, i64 3, !dbg !12
  %1484 = select <4 x i1> %1442, <4 x float> %1479, <4 x float> %1483, !dbg !12
  %1485 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !12
  %1486 = extractelement <8 x float> %1485, i64 0, !dbg !12
  %1487 = extractelement <8 x float> %1485, i64 1, !dbg !12
  %1488 = extractelement <8 x float> %1485, i64 2, !dbg !12
  %1489 = extractelement <8 x float> %1485, i64 3, !dbg !12
  %1490 = select i1 %.not, float %1487, float %1486, !dbg !12
  %1491 = bitcast float %1490 to i32, !dbg !12
  %1492 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1491), !dbg !12
  %1493 = bitcast i32 %1492 to float, !dbg !12
  %1494 = select i1 %.not, float %1486, float %1493, !dbg !12
  %1495 = select i1 %.not, float %1493, float %1487, !dbg !12
  %1496 = select i1 %.not, float %1489, float %1488, !dbg !12
  %1497 = bitcast float %1496 to i32, !dbg !12
  %1498 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1497), !dbg !12
  %1499 = bitcast i32 %1498 to float, !dbg !12
  %1500 = select i1 %.not, float %1488, float %1499, !dbg !12
  %1501 = select i1 %.not, float %1499, float %1489, !dbg !12
  %1502 = select i1 %.not40, float %1500, float %1494, !dbg !12
  %1503 = bitcast float %1502 to i32, !dbg !12
  %1504 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1503), !dbg !12
  %1505 = bitcast i32 %1504 to float, !dbg !12
  %1506 = select i1 %.not40, float %1501, float %1495, !dbg !12
  %1507 = bitcast float %1506 to i32, !dbg !12
  %1508 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1507), !dbg !12
  %1509 = bitcast i32 %1508 to float, !dbg !12
  %1510 = insertelement <4 x float> poison, float %1494, i64 0, !dbg !12
  %1511 = insertelement <4 x float> %1510, float %1495, i64 1, !dbg !12
  %1512 = insertelement <4 x float> %1511, float %1505, i64 2, !dbg !12
  %1513 = insertelement <4 x float> %1512, float %1509, i64 3, !dbg !12
  %1514 = insertelement <4 x float> poison, float %1505, i64 0, !dbg !12
  %1515 = insertelement <4 x float> %1514, float %1509, i64 1, !dbg !12
  %1516 = insertelement <4 x float> %1515, float %1500, i64 2, !dbg !12
  %1517 = insertelement <4 x float> %1516, float %1501, i64 3, !dbg !12
  %1518 = select <4 x i1> %1442, <4 x float> %1513, <4 x float> %1517, !dbg !12
  %1519 = extractelement <8 x float> %1485, i64 4, !dbg !12
  %1520 = extractelement <8 x float> %1485, i64 5, !dbg !12
  %1521 = extractelement <8 x float> %1485, i64 6, !dbg !12
  %1522 = extractelement <8 x float> %1485, i64 7, !dbg !12
  %1523 = select i1 %.not, float %1520, float %1519, !dbg !12
  %1524 = bitcast float %1523 to i32, !dbg !12
  %1525 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1524), !dbg !12
  %1526 = bitcast i32 %1525 to float, !dbg !12
  %1527 = select i1 %.not, float %1519, float %1526, !dbg !12
  %1528 = select i1 %.not, float %1526, float %1520, !dbg !12
  %1529 = select i1 %.not, float %1522, float %1521, !dbg !12
  %1530 = bitcast float %1529 to i32, !dbg !12
  %1531 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1530), !dbg !12
  %1532 = bitcast i32 %1531 to float, !dbg !12
  %1533 = select i1 %.not, float %1521, float %1532, !dbg !12
  %1534 = select i1 %.not, float %1532, float %1522, !dbg !12
  %1535 = select i1 %.not40, float %1533, float %1527, !dbg !12
  %1536 = bitcast float %1535 to i32, !dbg !12
  %1537 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1536), !dbg !12
  %1538 = bitcast i32 %1537 to float, !dbg !12
  %1539 = select i1 %.not40, float %1534, float %1528, !dbg !12
  %1540 = bitcast float %1539 to i32, !dbg !12
  %1541 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1540), !dbg !12
  %1542 = bitcast i32 %1541 to float, !dbg !12
  %1543 = insertelement <4 x float> poison, float %1527, i64 0, !dbg !12
  %1544 = insertelement <4 x float> %1543, float %1528, i64 1, !dbg !12
  %1545 = insertelement <4 x float> %1544, float %1538, i64 2, !dbg !12
  %1546 = insertelement <4 x float> %1545, float %1542, i64 3, !dbg !12
  %1547 = insertelement <4 x float> poison, float %1538, i64 0, !dbg !12
  %1548 = insertelement <4 x float> %1547, float %1542, i64 1, !dbg !12
  %1549 = insertelement <4 x float> %1548, float %1533, i64 2, !dbg !12
  %1550 = insertelement <4 x float> %1549, float %1534, i64 3, !dbg !12
  %1551 = select <4 x i1> %1442, <4 x float> %1546, <4 x float> %1550, !dbg !12
  %1552 = extractvalue [2 x <8 x float>] %.lcssa4, 0, !dbg !12
  %1553 = extractelement <8 x float> %1552, i64 0, !dbg !12
  %1554 = extractelement <8 x float> %1552, i64 1, !dbg !12
  %1555 = extractelement <8 x float> %1552, i64 2, !dbg !12
  %1556 = extractelement <8 x float> %1552, i64 3, !dbg !12
  %1557 = select i1 %.not, float %1554, float %1553, !dbg !12
  %1558 = bitcast float %1557 to i32, !dbg !12
  %1559 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1558), !dbg !12
  %1560 = bitcast i32 %1559 to float, !dbg !12
  %1561 = select i1 %.not, float %1553, float %1560, !dbg !12
  %1562 = select i1 %.not, float %1560, float %1554, !dbg !12
  %1563 = select i1 %.not, float %1556, float %1555, !dbg !12
  %1564 = bitcast float %1563 to i32, !dbg !12
  %1565 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1564), !dbg !12
  %1566 = bitcast i32 %1565 to float, !dbg !12
  %1567 = select i1 %.not, float %1555, float %1566, !dbg !12
  %1568 = select i1 %.not, float %1566, float %1556, !dbg !12
  %1569 = select i1 %.not40, float %1567, float %1561, !dbg !12
  %1570 = bitcast float %1569 to i32, !dbg !12
  %1571 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1570), !dbg !12
  %1572 = bitcast i32 %1571 to float, !dbg !12
  %1573 = select i1 %.not40, float %1568, float %1562, !dbg !12
  %1574 = bitcast float %1573 to i32, !dbg !12
  %1575 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1574), !dbg !12
  %1576 = bitcast i32 %1575 to float, !dbg !12
  %1577 = insertelement <4 x float> poison, float %1561, i64 0, !dbg !12
  %1578 = insertelement <4 x float> %1577, float %1562, i64 1, !dbg !12
  %1579 = insertelement <4 x float> %1578, float %1572, i64 2, !dbg !12
  %1580 = insertelement <4 x float> %1579, float %1576, i64 3, !dbg !12
  %1581 = insertelement <4 x float> poison, float %1572, i64 0, !dbg !12
  %1582 = insertelement <4 x float> %1581, float %1576, i64 1, !dbg !12
  %1583 = insertelement <4 x float> %1582, float %1567, i64 2, !dbg !12
  %1584 = insertelement <4 x float> %1583, float %1568, i64 3, !dbg !12
  %1585 = select <4 x i1> %1442, <4 x float> %1580, <4 x float> %1584, !dbg !12
  %1586 = extractelement <8 x float> %1552, i64 4, !dbg !12
  %1587 = extractelement <8 x float> %1552, i64 5, !dbg !12
  %1588 = extractelement <8 x float> %1552, i64 6, !dbg !12
  %1589 = extractelement <8 x float> %1552, i64 7, !dbg !12
  %1590 = select i1 %.not, float %1587, float %1586, !dbg !12
  %1591 = bitcast float %1590 to i32, !dbg !12
  %1592 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1591), !dbg !12
  %1593 = bitcast i32 %1592 to float, !dbg !12
  %1594 = select i1 %.not, float %1586, float %1593, !dbg !12
  %1595 = select i1 %.not, float %1593, float %1587, !dbg !12
  %1596 = select i1 %.not, float %1589, float %1588, !dbg !12
  %1597 = bitcast float %1596 to i32, !dbg !12
  %1598 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1597), !dbg !12
  %1599 = bitcast i32 %1598 to float, !dbg !12
  %1600 = select i1 %.not, float %1588, float %1599, !dbg !12
  %1601 = select i1 %.not, float %1599, float %1589, !dbg !12
  %1602 = select i1 %.not40, float %1600, float %1594, !dbg !12
  %1603 = bitcast float %1602 to i32, !dbg !12
  %1604 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1603), !dbg !12
  %1605 = bitcast i32 %1604 to float, !dbg !12
  %1606 = select i1 %.not40, float %1601, float %1595, !dbg !12
  %1607 = bitcast float %1606 to i32, !dbg !12
  %1608 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1607), !dbg !12
  %1609 = bitcast i32 %1608 to float, !dbg !12
  %1610 = insertelement <4 x float> poison, float %1594, i64 0, !dbg !12
  %1611 = insertelement <4 x float> %1610, float %1595, i64 1, !dbg !12
  %1612 = insertelement <4 x float> %1611, float %1605, i64 2, !dbg !12
  %1613 = insertelement <4 x float> %1612, float %1609, i64 3, !dbg !12
  %1614 = insertelement <4 x float> poison, float %1605, i64 0, !dbg !12
  %1615 = insertelement <4 x float> %1614, float %1609, i64 1, !dbg !12
  %1616 = insertelement <4 x float> %1615, float %1600, i64 2, !dbg !12
  %1617 = insertelement <4 x float> %1616, float %1601, i64 3, !dbg !12
  %1618 = select <4 x i1> %1442, <4 x float> %1613, <4 x float> %1617, !dbg !12
  %1619 = extractvalue [2 x <8 x float>] %.lcssa4, 1, !dbg !12
  %1620 = extractelement <8 x float> %1619, i64 0, !dbg !12
  %1621 = extractelement <8 x float> %1619, i64 1, !dbg !12
  %1622 = extractelement <8 x float> %1619, i64 2, !dbg !12
  %1623 = extractelement <8 x float> %1619, i64 3, !dbg !12
  %1624 = select i1 %.not, float %1621, float %1620, !dbg !12
  %1625 = bitcast float %1624 to i32, !dbg !12
  %1626 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1625), !dbg !12
  %1627 = bitcast i32 %1626 to float, !dbg !12
  %1628 = select i1 %.not, float %1620, float %1627, !dbg !12
  %1629 = select i1 %.not, float %1627, float %1621, !dbg !12
  %1630 = select i1 %.not, float %1623, float %1622, !dbg !12
  %1631 = bitcast float %1630 to i32, !dbg !12
  %1632 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1631), !dbg !12
  %1633 = bitcast i32 %1632 to float, !dbg !12
  %1634 = select i1 %.not, float %1622, float %1633, !dbg !12
  %1635 = select i1 %.not, float %1633, float %1623, !dbg !12
  %1636 = select i1 %.not40, float %1634, float %1628, !dbg !12
  %1637 = bitcast float %1636 to i32, !dbg !12
  %1638 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1637), !dbg !12
  %1639 = bitcast i32 %1638 to float, !dbg !12
  %1640 = select i1 %.not40, float %1635, float %1629, !dbg !12
  %1641 = bitcast float %1640 to i32, !dbg !12
  %1642 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1641), !dbg !12
  %1643 = bitcast i32 %1642 to float, !dbg !12
  %1644 = insertelement <4 x float> poison, float %1628, i64 0, !dbg !12
  %1645 = insertelement <4 x float> %1644, float %1629, i64 1, !dbg !12
  %1646 = insertelement <4 x float> %1645, float %1639, i64 2, !dbg !12
  %1647 = insertelement <4 x float> %1646, float %1643, i64 3, !dbg !12
  %1648 = insertelement <4 x float> poison, float %1639, i64 0, !dbg !12
  %1649 = insertelement <4 x float> %1648, float %1643, i64 1, !dbg !12
  %1650 = insertelement <4 x float> %1649, float %1634, i64 2, !dbg !12
  %1651 = insertelement <4 x float> %1650, float %1635, i64 3, !dbg !12
  %1652 = select <4 x i1> %1442, <4 x float> %1647, <4 x float> %1651, !dbg !12
  %1653 = extractelement <8 x float> %1619, i64 4, !dbg !12
  %1654 = extractelement <8 x float> %1619, i64 5, !dbg !12
  %1655 = extractelement <8 x float> %1619, i64 6, !dbg !12
  %1656 = extractelement <8 x float> %1619, i64 7, !dbg !12
  %1657 = select i1 %.not, float %1654, float %1653, !dbg !12
  %1658 = bitcast float %1657 to i32, !dbg !12
  %1659 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1658), !dbg !12
  %1660 = bitcast i32 %1659 to float, !dbg !12
  %1661 = select i1 %.not, float %1653, float %1660, !dbg !12
  %1662 = select i1 %.not, float %1660, float %1654, !dbg !12
  %1663 = select i1 %.not, float %1656, float %1655, !dbg !12
  %1664 = bitcast float %1663 to i32, !dbg !12
  %1665 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1664), !dbg !12
  %1666 = bitcast i32 %1665 to float, !dbg !12
  %1667 = select i1 %.not, float %1655, float %1666, !dbg !12
  %1668 = select i1 %.not, float %1666, float %1656, !dbg !12
  %1669 = select i1 %.not40, float %1667, float %1661, !dbg !12
  %1670 = bitcast float %1669 to i32, !dbg !12
  %1671 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1670), !dbg !12
  %1672 = bitcast i32 %1671 to float, !dbg !12
  %1673 = select i1 %.not40, float %1668, float %1662, !dbg !12
  %1674 = bitcast float %1673 to i32, !dbg !12
  %1675 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1674), !dbg !12
  %1676 = bitcast i32 %1675 to float, !dbg !12
  %1677 = insertelement <4 x float> poison, float %1661, i64 0, !dbg !12
  %1678 = insertelement <4 x float> %1677, float %1662, i64 1, !dbg !12
  %1679 = insertelement <4 x float> %1678, float %1672, i64 2, !dbg !12
  %1680 = insertelement <4 x float> %1679, float %1676, i64 3, !dbg !12
  %1681 = insertelement <4 x float> poison, float %1672, i64 0, !dbg !12
  %1682 = insertelement <4 x float> %1681, float %1676, i64 1, !dbg !12
  %1683 = insertelement <4 x float> %1682, float %1667, i64 2, !dbg !12
  %1684 = insertelement <4 x float> %1683, float %1668, i64 3, !dbg !12
  %1685 = select <4 x i1> %1442, <4 x float> %1680, <4 x float> %1684, !dbg !12
  %1686 = extractvalue [2 x <8 x float>] %.lcssa5, 0, !dbg !12
  %1687 = extractelement <8 x float> %1686, i64 0, !dbg !12
  %1688 = extractelement <8 x float> %1686, i64 1, !dbg !12
  %1689 = extractelement <8 x float> %1686, i64 2, !dbg !12
  %1690 = extractelement <8 x float> %1686, i64 3, !dbg !12
  %1691 = select i1 %.not, float %1688, float %1687, !dbg !12
  %1692 = bitcast float %1691 to i32, !dbg !12
  %1693 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1692), !dbg !12
  %1694 = bitcast i32 %1693 to float, !dbg !12
  %1695 = select i1 %.not, float %1687, float %1694, !dbg !12
  %1696 = select i1 %.not, float %1694, float %1688, !dbg !12
  %1697 = select i1 %.not, float %1690, float %1689, !dbg !12
  %1698 = bitcast float %1697 to i32, !dbg !12
  %1699 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1698), !dbg !12
  %1700 = bitcast i32 %1699 to float, !dbg !12
  %1701 = select i1 %.not, float %1689, float %1700, !dbg !12
  %1702 = select i1 %.not, float %1700, float %1690, !dbg !12
  %1703 = select i1 %.not40, float %1701, float %1695, !dbg !12
  %1704 = bitcast float %1703 to i32, !dbg !12
  %1705 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1704), !dbg !12
  %1706 = bitcast i32 %1705 to float, !dbg !12
  %1707 = select i1 %.not40, float %1702, float %1696, !dbg !12
  %1708 = bitcast float %1707 to i32, !dbg !12
  %1709 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1708), !dbg !12
  %1710 = bitcast i32 %1709 to float, !dbg !12
  %1711 = insertelement <4 x float> poison, float %1695, i64 0, !dbg !12
  %1712 = insertelement <4 x float> %1711, float %1696, i64 1, !dbg !12
  %1713 = insertelement <4 x float> %1712, float %1706, i64 2, !dbg !12
  %1714 = insertelement <4 x float> %1713, float %1710, i64 3, !dbg !12
  %1715 = insertelement <4 x float> poison, float %1706, i64 0, !dbg !12
  %1716 = insertelement <4 x float> %1715, float %1710, i64 1, !dbg !12
  %1717 = insertelement <4 x float> %1716, float %1701, i64 2, !dbg !12
  %1718 = insertelement <4 x float> %1717, float %1702, i64 3, !dbg !12
  %1719 = select <4 x i1> %1442, <4 x float> %1714, <4 x float> %1718, !dbg !12
  %1720 = extractelement <8 x float> %1686, i64 4, !dbg !12
  %1721 = extractelement <8 x float> %1686, i64 5, !dbg !12
  %1722 = extractelement <8 x float> %1686, i64 6, !dbg !12
  %1723 = extractelement <8 x float> %1686, i64 7, !dbg !12
  %1724 = select i1 %.not, float %1721, float %1720, !dbg !12
  %1725 = bitcast float %1724 to i32, !dbg !12
  %1726 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1725), !dbg !12
  %1727 = bitcast i32 %1726 to float, !dbg !12
  %1728 = select i1 %.not, float %1720, float %1727, !dbg !12
  %1729 = select i1 %.not, float %1727, float %1721, !dbg !12
  %1730 = select i1 %.not, float %1723, float %1722, !dbg !12
  %1731 = bitcast float %1730 to i32, !dbg !12
  %1732 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1731), !dbg !12
  %1733 = bitcast i32 %1732 to float, !dbg !12
  %1734 = select i1 %.not, float %1722, float %1733, !dbg !12
  %1735 = select i1 %.not, float %1733, float %1723, !dbg !12
  %1736 = select i1 %.not40, float %1734, float %1728, !dbg !12
  %1737 = bitcast float %1736 to i32, !dbg !12
  %1738 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1737), !dbg !12
  %1739 = bitcast i32 %1738 to float, !dbg !12
  %1740 = select i1 %.not40, float %1735, float %1729, !dbg !12
  %1741 = bitcast float %1740 to i32, !dbg !12
  %1742 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1741), !dbg !12
  %1743 = bitcast i32 %1742 to float, !dbg !12
  %1744 = insertelement <4 x float> poison, float %1728, i64 0, !dbg !12
  %1745 = insertelement <4 x float> %1744, float %1729, i64 1, !dbg !12
  %1746 = insertelement <4 x float> %1745, float %1739, i64 2, !dbg !12
  %1747 = insertelement <4 x float> %1746, float %1743, i64 3, !dbg !12
  %1748 = insertelement <4 x float> poison, float %1739, i64 0, !dbg !12
  %1749 = insertelement <4 x float> %1748, float %1743, i64 1, !dbg !12
  %1750 = insertelement <4 x float> %1749, float %1734, i64 2, !dbg !12
  %1751 = insertelement <4 x float> %1750, float %1735, i64 3, !dbg !12
  %1752 = select <4 x i1> %1442, <4 x float> %1747, <4 x float> %1751, !dbg !12
  %1753 = extractvalue [2 x <8 x float>] %.lcssa5, 1, !dbg !12
  %1754 = extractelement <8 x float> %1753, i64 0, !dbg !12
  %1755 = extractelement <8 x float> %1753, i64 1, !dbg !12
  %1756 = extractelement <8 x float> %1753, i64 2, !dbg !12
  %1757 = extractelement <8 x float> %1753, i64 3, !dbg !12
  %1758 = select i1 %.not, float %1755, float %1754, !dbg !12
  %1759 = bitcast float %1758 to i32, !dbg !12
  %1760 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1759), !dbg !12
  %1761 = bitcast i32 %1760 to float, !dbg !12
  %1762 = select i1 %.not, float %1754, float %1761, !dbg !12
  %1763 = select i1 %.not, float %1761, float %1755, !dbg !12
  %1764 = select i1 %.not, float %1757, float %1756, !dbg !12
  %1765 = bitcast float %1764 to i32, !dbg !12
  %1766 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1765), !dbg !12
  %1767 = bitcast i32 %1766 to float, !dbg !12
  %1768 = select i1 %.not, float %1756, float %1767, !dbg !12
  %1769 = select i1 %.not, float %1767, float %1757, !dbg !12
  %1770 = select i1 %.not40, float %1768, float %1762, !dbg !12
  %1771 = bitcast float %1770 to i32, !dbg !12
  %1772 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1771), !dbg !12
  %1773 = bitcast i32 %1772 to float, !dbg !12
  %1774 = select i1 %.not40, float %1769, float %1763, !dbg !12
  %1775 = bitcast float %1774 to i32, !dbg !12
  %1776 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1775), !dbg !12
  %1777 = bitcast i32 %1776 to float, !dbg !12
  %1778 = insertelement <4 x float> poison, float %1762, i64 0, !dbg !12
  %1779 = insertelement <4 x float> %1778, float %1763, i64 1, !dbg !12
  %1780 = insertelement <4 x float> %1779, float %1773, i64 2, !dbg !12
  %1781 = insertelement <4 x float> %1780, float %1777, i64 3, !dbg !12
  %1782 = insertelement <4 x float> poison, float %1773, i64 0, !dbg !12
  %1783 = insertelement <4 x float> %1782, float %1777, i64 1, !dbg !12
  %1784 = insertelement <4 x float> %1783, float %1768, i64 2, !dbg !12
  %1785 = insertelement <4 x float> %1784, float %1769, i64 3, !dbg !12
  %1786 = select <4 x i1> %1442, <4 x float> %1781, <4 x float> %1785, !dbg !12
  %1787 = extractelement <8 x float> %1753, i64 4, !dbg !12
  %1788 = extractelement <8 x float> %1753, i64 5, !dbg !12
  %1789 = extractelement <8 x float> %1753, i64 6, !dbg !12
  %1790 = extractelement <8 x float> %1753, i64 7, !dbg !12
  %1791 = select i1 %.not, float %1788, float %1787, !dbg !12
  %1792 = bitcast float %1791 to i32, !dbg !12
  %1793 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1792), !dbg !12
  %1794 = bitcast i32 %1793 to float, !dbg !12
  %1795 = select i1 %.not, float %1787, float %1794, !dbg !12
  %1796 = select i1 %.not, float %1794, float %1788, !dbg !12
  %1797 = select i1 %.not, float %1790, float %1789, !dbg !12
  %1798 = bitcast float %1797 to i32, !dbg !12
  %1799 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1798), !dbg !12
  %1800 = bitcast i32 %1799 to float, !dbg !12
  %1801 = select i1 %.not, float %1789, float %1800, !dbg !12
  %1802 = select i1 %.not, float %1800, float %1790, !dbg !12
  %1803 = select i1 %.not40, float %1801, float %1795, !dbg !12
  %1804 = bitcast float %1803 to i32, !dbg !12
  %1805 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1804), !dbg !12
  %1806 = bitcast i32 %1805 to float, !dbg !12
  %1807 = select i1 %.not40, float %1802, float %1796, !dbg !12
  %1808 = bitcast float %1807 to i32, !dbg !12
  %1809 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1808), !dbg !12
  %1810 = bitcast i32 %1809 to float, !dbg !12
  %1811 = insertelement <4 x float> poison, float %1795, i64 0, !dbg !12
  %1812 = insertelement <4 x float> %1811, float %1796, i64 1, !dbg !12
  %1813 = insertelement <4 x float> %1812, float %1806, i64 2, !dbg !12
  %1814 = insertelement <4 x float> %1813, float %1810, i64 3, !dbg !12
  %1815 = insertelement <4 x float> poison, float %1806, i64 0, !dbg !12
  %1816 = insertelement <4 x float> %1815, float %1810, i64 1, !dbg !12
  %1817 = insertelement <4 x float> %1816, float %1801, i64 2, !dbg !12
  %1818 = insertelement <4 x float> %1817, float %1802, i64 3, !dbg !12
  %1819 = select <4 x i1> %1442, <4 x float> %1814, <4 x float> %1818, !dbg !12
  %1820 = extractvalue [2 x <8 x float>] %.lcssa7, 0, !dbg !12
  %1821 = extractelement <8 x float> %1820, i64 0, !dbg !12
  %1822 = extractelement <8 x float> %1820, i64 1, !dbg !12
  %1823 = extractelement <8 x float> %1820, i64 2, !dbg !12
  %1824 = extractelement <8 x float> %1820, i64 3, !dbg !12
  %1825 = select i1 %.not, float %1822, float %1821, !dbg !12
  %1826 = bitcast float %1825 to i32, !dbg !12
  %1827 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1826), !dbg !12
  %1828 = bitcast i32 %1827 to float, !dbg !12
  %1829 = select i1 %.not, float %1821, float %1828, !dbg !12
  %1830 = select i1 %.not, float %1828, float %1822, !dbg !12
  %1831 = select i1 %.not, float %1824, float %1823, !dbg !12
  %1832 = bitcast float %1831 to i32, !dbg !12
  %1833 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1832), !dbg !12
  %1834 = bitcast i32 %1833 to float, !dbg !12
  %1835 = select i1 %.not, float %1823, float %1834, !dbg !12
  %1836 = select i1 %.not, float %1834, float %1824, !dbg !12
  %1837 = select i1 %.not40, float %1835, float %1829, !dbg !12
  %1838 = bitcast float %1837 to i32, !dbg !12
  %1839 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1838), !dbg !12
  %1840 = select i1 %.not40, float %1836, float %1830, !dbg !12
  %1841 = bitcast float %1840 to i32, !dbg !12
  %1842 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1841), !dbg !12
  %1843 = extractelement <8 x float> %1820, i64 4, !dbg !12
  %1844 = extractelement <8 x float> %1820, i64 5, !dbg !12
  %1845 = extractelement <8 x float> %1820, i64 6, !dbg !12
  %1846 = extractelement <8 x float> %1820, i64 7, !dbg !12
  %1847 = select i1 %.not, float %1844, float %1843, !dbg !12
  %1848 = bitcast float %1847 to i32, !dbg !12
  %1849 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1848), !dbg !12
  %1850 = bitcast i32 %1849 to float, !dbg !12
  %1851 = select i1 %.not, float %1843, float %1850, !dbg !12
  %1852 = select i1 %.not, float %1850, float %1844, !dbg !12
  %1853 = select i1 %.not, float %1846, float %1845, !dbg !12
  %1854 = bitcast float %1853 to i32, !dbg !12
  %1855 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1854), !dbg !12
  %1856 = bitcast i32 %1855 to float, !dbg !12
  %1857 = select i1 %.not, float %1845, float %1856, !dbg !12
  %1858 = select i1 %.not, float %1856, float %1846, !dbg !12
  %1859 = select i1 %.not40, float %1857, float %1851, !dbg !12
  %1860 = bitcast float %1859 to i32, !dbg !12
  %1861 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1860), !dbg !12
  %1862 = select i1 %.not40, float %1858, float %1852, !dbg !12
  %1863 = bitcast float %1862 to i32, !dbg !12
  %1864 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1863), !dbg !12
  %1865 = extractvalue [2 x <8 x float>] %.lcssa7, 1, !dbg !12
  %1866 = extractelement <8 x float> %1865, i64 0, !dbg !12
  %1867 = extractelement <8 x float> %1865, i64 1, !dbg !12
  %1868 = extractelement <8 x float> %1865, i64 2, !dbg !12
  %1869 = extractelement <8 x float> %1865, i64 3, !dbg !12
  %1870 = select i1 %.not, float %1867, float %1866, !dbg !12
  %1871 = bitcast float %1870 to i32, !dbg !12
  %1872 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1871), !dbg !12
  %1873 = bitcast i32 %1872 to float, !dbg !12
  %1874 = select i1 %.not, float %1866, float %1873, !dbg !12
  %1875 = select i1 %.not, float %1873, float %1867, !dbg !12
  %1876 = select i1 %.not, float %1869, float %1868, !dbg !12
  %1877 = bitcast float %1876 to i32, !dbg !12
  %1878 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1877), !dbg !12
  %1879 = bitcast i32 %1878 to float, !dbg !12
  %1880 = select i1 %.not, float %1868, float %1879, !dbg !12
  %1881 = select i1 %.not, float %1879, float %1869, !dbg !12
  %1882 = select i1 %.not40, float %1880, float %1874, !dbg !12
  %1883 = bitcast float %1882 to i32, !dbg !12
  %1884 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1883), !dbg !12
  %1885 = select i1 %.not40, float %1881, float %1875, !dbg !12
  %1886 = bitcast float %1885 to i32, !dbg !12
  %1887 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1886), !dbg !12
  %1888 = extractelement <8 x float> %1865, i64 4, !dbg !12
  %1889 = extractelement <8 x float> %1865, i64 5, !dbg !12
  %1890 = extractelement <8 x float> %1865, i64 6, !dbg !12
  %1891 = extractelement <8 x float> %1865, i64 7, !dbg !12
  %1892 = select i1 %.not, float %1889, float %1888, !dbg !12
  %1893 = bitcast float %1892 to i32, !dbg !12
  %1894 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1893), !dbg !12
  %1895 = bitcast i32 %1894 to float, !dbg !12
  %1896 = select i1 %.not, float %1888, float %1895, !dbg !12
  %1897 = select i1 %.not, float %1895, float %1889, !dbg !12
  %1898 = select i1 %.not, float %1891, float %1890, !dbg !12
  %1899 = bitcast float %1898 to i32, !dbg !12
  %1900 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %1899), !dbg !12
  %1901 = bitcast i32 %1900 to float, !dbg !12
  %1902 = select i1 %.not, float %1890, float %1901, !dbg !12
  %1903 = select i1 %.not, float %1901, float %1891, !dbg !12
  %1904 = select i1 %.not40, float %1902, float %1896, !dbg !12
  %1905 = bitcast float %1904 to i32, !dbg !12
  %1906 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1905), !dbg !12
  %1907 = select i1 %.not40, float %1903, float %1897, !dbg !12
  %1908 = bitcast float %1907 to i32, !dbg !12
  %1909 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1908), !dbg !12
  %1910 = insertelement <2 x i32> poison, i32 %1413, i64 0, !dbg !12
  %1911 = insertelement <2 x i32> %1910, i32 %1410, i64 1, !dbg !12
  %1912 = bitcast <2 x i32> %1911 to <2 x float>, !dbg !12
  %1913 = shufflevector <1 x float> %1411, <1 x float> %1408, <2 x i32> <i32 0, i32 1>, !dbg !12
  %1914 = fadd <2 x float> %1913, %1912, !dbg !12
  %bc48 = bitcast <2 x float> %1914 to <2 x i32>, !dbg !12
  %1915 = extractelement <2 x i32> %bc48, i64 1, !dbg !12
  %1916 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1915), !dbg !12
  %bc49 = bitcast <2 x float> %1914 to <2 x i32>, !dbg !12
  %1917 = extractelement <2 x i32> %bc49, i64 0, !dbg !12
  %1918 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %1917), !dbg !12
  %1919 = insertelement <2 x i32> poison, i32 %1918, i64 0, !dbg !12
  %1920 = insertelement <2 x i32> %1919, i32 %1916, i64 1, !dbg !12
  %1921 = bitcast <2 x i32> %1920 to <2 x float>, !dbg !12
  %1922 = fadd <2 x float> %1914, %1921, !dbg !12
  %1923 = fdiv arcp <2 x float> splat (float 1.000000e+00), %1922, !dbg !51
  %1924 = shufflevector <2 x float> %1923, <2 x float> poison, <4 x i32> zeroinitializer, !dbg !51
  %1925 = shufflevector <2 x float> %1923, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, !dbg !51
  %1926 = bitcast i32 %1906 to float, !dbg !12
  %1927 = bitcast i32 %1909 to float, !dbg !12
  %1928 = insertelement <4 x float> poison, float %1896, i64 0, !dbg !12
  %1929 = insertelement <4 x float> %1928, float %1897, i64 1, !dbg !12
  %1930 = insertelement <4 x float> %1929, float %1926, i64 2, !dbg !12
  %1931 = insertelement <4 x float> %1930, float %1927, i64 3, !dbg !12
  %1932 = insertelement <4 x float> poison, float %1926, i64 0, !dbg !12
  %1933 = insertelement <4 x float> %1932, float %1927, i64 1, !dbg !12
  %1934 = insertelement <4 x float> %1933, float %1902, i64 2, !dbg !12
  %1935 = insertelement <4 x float> %1934, float %1903, i64 3, !dbg !12
  %1936 = select <4 x i1> %1442, <4 x float> %1931, <4 x float> %1935, !dbg !12
  %1937 = fmul <4 x float> %1925, %1936, !dbg !51
  %1938 = fptrunc <4 x float> %1937 to <4 x half>, !dbg !52
  %1939 = extractelement <4 x half> %1938, i64 3, !dbg !52
  %1940 = extractelement <4 x half> %1938, i64 2, !dbg !52
  %1941 = extractelement <4 x half> %1938, i64 1, !dbg !52
  %1942 = extractelement <4 x half> %1938, i64 0, !dbg !52
  %1943 = bitcast i32 %1884 to float, !dbg !12
  %1944 = bitcast i32 %1887 to float, !dbg !12
  %1945 = insertelement <4 x float> poison, float %1874, i64 0, !dbg !12
  %1946 = insertelement <4 x float> %1945, float %1875, i64 1, !dbg !12
  %1947 = insertelement <4 x float> %1946, float %1943, i64 2, !dbg !12
  %1948 = insertelement <4 x float> %1947, float %1944, i64 3, !dbg !12
  %1949 = insertelement <4 x float> poison, float %1943, i64 0, !dbg !12
  %1950 = insertelement <4 x float> %1949, float %1944, i64 1, !dbg !12
  %1951 = insertelement <4 x float> %1950, float %1880, i64 2, !dbg !12
  %1952 = insertelement <4 x float> %1951, float %1881, i64 3, !dbg !12
  %1953 = select <4 x i1> %1442, <4 x float> %1948, <4 x float> %1952, !dbg !12
  %1954 = fmul <4 x float> %1925, %1953, !dbg !51
  %1955 = fptrunc <4 x float> %1954 to <4 x half>, !dbg !52
  %1956 = extractelement <4 x half> %1955, i64 3, !dbg !52
  %1957 = extractelement <4 x half> %1955, i64 2, !dbg !52
  %1958 = extractelement <4 x half> %1955, i64 1, !dbg !52
  %1959 = extractelement <4 x half> %1955, i64 0, !dbg !52
  %1960 = bitcast i32 %1861 to float, !dbg !12
  %1961 = bitcast i32 %1864 to float, !dbg !12
  %1962 = insertelement <4 x float> poison, float %1851, i64 0, !dbg !12
  %1963 = insertelement <4 x float> %1962, float %1852, i64 1, !dbg !12
  %1964 = insertelement <4 x float> %1963, float %1960, i64 2, !dbg !12
  %1965 = insertelement <4 x float> %1964, float %1961, i64 3, !dbg !12
  %1966 = insertelement <4 x float> poison, float %1960, i64 0, !dbg !12
  %1967 = insertelement <4 x float> %1966, float %1961, i64 1, !dbg !12
  %1968 = insertelement <4 x float> %1967, float %1857, i64 2, !dbg !12
  %1969 = insertelement <4 x float> %1968, float %1858, i64 3, !dbg !12
  %1970 = select <4 x i1> %1442, <4 x float> %1965, <4 x float> %1969, !dbg !12
  %1971 = fmul <4 x float> %1924, %1970, !dbg !51
  %1972 = fptrunc <4 x float> %1971 to <4 x half>, !dbg !52
  %1973 = extractelement <4 x half> %1972, i64 3, !dbg !52
  %1974 = extractelement <4 x half> %1972, i64 2, !dbg !52
  %1975 = extractelement <4 x half> %1972, i64 1, !dbg !52
  %1976 = extractelement <4 x half> %1972, i64 0, !dbg !52
  %1977 = bitcast i32 %1839 to float, !dbg !12
  %1978 = bitcast i32 %1842 to float, !dbg !12
  %1979 = insertelement <4 x float> poison, float %1829, i64 0, !dbg !12
  %1980 = insertelement <4 x float> %1979, float %1830, i64 1, !dbg !12
  %1981 = insertelement <4 x float> %1980, float %1977, i64 2, !dbg !12
  %1982 = insertelement <4 x float> %1981, float %1978, i64 3, !dbg !12
  %1983 = insertelement <4 x float> poison, float %1977, i64 0, !dbg !12
  %1984 = insertelement <4 x float> %1983, float %1978, i64 1, !dbg !12
  %1985 = insertelement <4 x float> %1984, float %1835, i64 2, !dbg !12
  %1986 = insertelement <4 x float> %1985, float %1836, i64 3, !dbg !12
  %1987 = select <4 x i1> %1442, <4 x float> %1982, <4 x float> %1986, !dbg !12
  %1988 = fmul <4 x float> %1924, %1987, !dbg !51
  %1989 = fptrunc <4 x float> %1988 to <4 x half>, !dbg !52
  %1990 = extractelement <4 x half> %1989, i64 3, !dbg !52
  %1991 = extractelement <4 x half> %1989, i64 2, !dbg !52
  %1992 = extractelement <4 x half> %1989, i64 1, !dbg !52
  %1993 = extractelement <4 x half> %1989, i64 0, !dbg !52
  %address.thread904 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !53, !range !10, !frisk.register_lifetimes !5
  %address.local905 = zext nneg i32 %address.thread904 to i64, !dbg !9
  %address.local907 = lshr i64 %address.local905, 2, !dbg !16
  %address.local908 = and i64 %address.local907, 48, !dbg !16
  %address.local910 = and i64 %address.local905, 240, !dbg !53
  %address.local909 = add i64 %43, %address.local905, !dbg !53
  %address.local912 = sub i64 %address.local909, %address.local910, !dbg !53
  %address.local913 = add i64 %address.local912, %address.local908, !dbg !53
  %1994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %55, !dbg !53
  %address.local916.idx = shl nuw nsw i64 %address.local913, 8, !dbg !53
  %address.local916 = getelementptr inbounds nuw i8, ptr addrspace(1) %1994, i64 %address.local916.idx, !dbg !53
  %address.local918 = and i64 %address.local907, 12, !dbg !11
  %address.local919 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local918, !dbg !53
  store half %1993, ptr addrspace(1) %address.local919, align 2, !dbg !53
  %address.local921 = or disjoint i64 %address.local918, 1, !dbg !53
  %address.local922 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local921, !dbg !53
  store half %1992, ptr addrspace(1) %address.local922, align 2, !dbg !53
  %address.local924 = or disjoint i64 %address.local918, 2, !dbg !53
  %address.local925 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local924, !dbg !53
  store half %1991, ptr addrspace(1) %address.local925, align 2, !dbg !53
  %address.local927 = or disjoint i64 %address.local918, 3, !dbg !53
  %address.local928 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local927, !dbg !53
  store half %1990, ptr addrspace(1) %address.local928, align 2, !dbg !53
  %address.local930 = or disjoint i64 %address.local918, 16, !dbg !53
  %address.local931 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local930, !dbg !53
  store half %1976, ptr addrspace(1) %address.local931, align 2, !dbg !53
  %address.local933 = or disjoint i64 %address.local918, 17, !dbg !53
  %address.local934 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local933, !dbg !53
  store half %1975, ptr addrspace(1) %address.local934, align 2, !dbg !53
  %address.local936 = or disjoint i64 %address.local918, 18, !dbg !53
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local936, !dbg !53
  store half %1974, ptr addrspace(1) %address.local937, align 2, !dbg !53
  %address.local939 = or disjoint i64 %address.local918, 19, !dbg !53
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local939, !dbg !53
  store half %1973, ptr addrspace(1) %address.local940, align 2, !dbg !53
  %address.local944 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local916, i64 16384, !dbg !53
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local918, !dbg !53
  store half %1959, ptr addrspace(1) %address.local945, align 2, !dbg !53
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local921, !dbg !53
  store half %1958, ptr addrspace(1) %address.local947, align 2, !dbg !53
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local924, !dbg !53
  store half %1957, ptr addrspace(1) %address.local949, align 2, !dbg !53
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local927, !dbg !53
  store half %1956, ptr addrspace(1) %address.local951, align 2, !dbg !53
  %address.local953 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local930, !dbg !53
  store half %1942, ptr addrspace(1) %address.local953, align 2, !dbg !53
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local933, !dbg !53
  store half %1941, ptr addrspace(1) %address.local955, align 2, !dbg !53
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local936, !dbg !53
  store half %1940, ptr addrspace(1) %address.local957, align 2, !dbg !53
  %address.local959 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local939, !dbg !53
  store half %1939, ptr addrspace(1) %address.local959, align 2, !dbg !53
  %1995 = fmul <4 x float> %1925, %1819, !dbg !51
  %1996 = fptrunc <4 x float> %1995 to <4 x half>, !dbg !52
  %1997 = extractelement <4 x half> %1996, i64 3, !dbg !52
  %1998 = extractelement <4 x half> %1996, i64 2, !dbg !52
  %1999 = extractelement <4 x half> %1996, i64 1, !dbg !52
  %2000 = extractelement <4 x half> %1996, i64 0, !dbg !52
  %2001 = fmul <4 x float> %1925, %1786, !dbg !51
  %2002 = fptrunc <4 x float> %2001 to <4 x half>, !dbg !52
  %2003 = extractelement <4 x half> %2002, i64 3, !dbg !52
  %2004 = extractelement <4 x half> %2002, i64 2, !dbg !52
  %2005 = extractelement <4 x half> %2002, i64 1, !dbg !52
  %2006 = extractelement <4 x half> %2002, i64 0, !dbg !52
  %2007 = fmul <4 x float> %1924, %1752, !dbg !51
  %2008 = fptrunc <4 x float> %2007 to <4 x half>, !dbg !52
  %2009 = extractelement <4 x half> %2008, i64 3, !dbg !52
  %2010 = extractelement <4 x half> %2008, i64 2, !dbg !52
  %2011 = extractelement <4 x half> %2008, i64 1, !dbg !52
  %2012 = extractelement <4 x half> %2008, i64 0, !dbg !52
  %2013 = fmul <4 x float> %1924, %1719, !dbg !51
  %2014 = fptrunc <4 x float> %2013 to <4 x half>, !dbg !52
  %2015 = extractelement <4 x half> %2014, i64 3, !dbg !52
  %2016 = extractelement <4 x half> %2014, i64 2, !dbg !52
  %2017 = extractelement <4 x half> %2014, i64 1, !dbg !52
  %2018 = extractelement <4 x half> %2014, i64 0, !dbg !52
  %address.local975 = or disjoint i64 %address.local918, 32, !dbg !53
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local975, !dbg !53
  store half %2018, ptr addrspace(1) %address.local976, align 2, !dbg !53
  %address.local978 = or disjoint i64 %address.local918, 33, !dbg !53
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local978, !dbg !53
  store half %2017, ptr addrspace(1) %address.local979, align 2, !dbg !53
  %address.local981 = or disjoint i64 %address.local918, 34, !dbg !53
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local981, !dbg !53
  store half %2016, ptr addrspace(1) %address.local982, align 2, !dbg !53
  %address.local984 = or disjoint i64 %address.local918, 35, !dbg !53
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local984, !dbg !53
  store half %2015, ptr addrspace(1) %address.local985, align 2, !dbg !53
  %address.local987 = or disjoint i64 %address.local918, 48, !dbg !53
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local987, !dbg !53
  store half %2012, ptr addrspace(1) %address.local988, align 2, !dbg !53
  %address.local990 = or disjoint i64 %address.local918, 49, !dbg !53
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local990, !dbg !53
  store half %2011, ptr addrspace(1) %address.local991, align 2, !dbg !53
  %address.local993 = or disjoint i64 %address.local918, 50, !dbg !53
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local993, !dbg !53
  store half %2010, ptr addrspace(1) %address.local994, align 2, !dbg !53
  %address.local996 = or i64 %address.local907, 51, !dbg !53
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local996, !dbg !53
  store half %2009, ptr addrspace(1) %address.local997, align 2, !dbg !53
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local975, !dbg !53
  store half %2006, ptr addrspace(1) %address.local1002, align 2, !dbg !53
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local978, !dbg !53
  store half %2005, ptr addrspace(1) %address.local1004, align 2, !dbg !53
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local981, !dbg !53
  store half %2004, ptr addrspace(1) %address.local1006, align 2, !dbg !53
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local984, !dbg !53
  store half %2003, ptr addrspace(1) %address.local1008, align 2, !dbg !53
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local987, !dbg !53
  store half %2000, ptr addrspace(1) %address.local1010, align 2, !dbg !53
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local990, !dbg !53
  store half %1999, ptr addrspace(1) %address.local1012, align 2, !dbg !53
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local993, !dbg !53
  store half %1998, ptr addrspace(1) %address.local1014, align 2, !dbg !53
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local996, !dbg !53
  store half %1997, ptr addrspace(1) %address.local1016, align 2, !dbg !53
  %2019 = fmul <4 x float> %1925, %1685, !dbg !51
  %2020 = fptrunc <4 x float> %2019 to <4 x half>, !dbg !52
  %2021 = extractelement <4 x half> %2020, i64 3, !dbg !52
  %2022 = extractelement <4 x half> %2020, i64 2, !dbg !52
  %2023 = extractelement <4 x half> %2020, i64 1, !dbg !52
  %2024 = extractelement <4 x half> %2020, i64 0, !dbg !52
  %2025 = fmul <4 x float> %1925, %1652, !dbg !51
  %2026 = fptrunc <4 x float> %2025 to <4 x half>, !dbg !52
  %2027 = extractelement <4 x half> %2026, i64 3, !dbg !52
  %2028 = extractelement <4 x half> %2026, i64 2, !dbg !52
  %2029 = extractelement <4 x half> %2026, i64 1, !dbg !52
  %2030 = extractelement <4 x half> %2026, i64 0, !dbg !52
  %2031 = fmul <4 x float> %1924, %1618, !dbg !51
  %2032 = fptrunc <4 x float> %2031 to <4 x half>, !dbg !52
  %2033 = extractelement <4 x half> %2032, i64 3, !dbg !52
  %2034 = extractelement <4 x half> %2032, i64 2, !dbg !52
  %2035 = extractelement <4 x half> %2032, i64 1, !dbg !52
  %2036 = extractelement <4 x half> %2032, i64 0, !dbg !52
  %2037 = fmul <4 x float> %1924, %1585, !dbg !51
  %2038 = fptrunc <4 x float> %2037 to <4 x half>, !dbg !52
  %2039 = extractelement <4 x half> %2038, i64 3, !dbg !52
  %2040 = extractelement <4 x half> %2038, i64 2, !dbg !52
  %2041 = extractelement <4 x half> %2038, i64 1, !dbg !52
  %2042 = extractelement <4 x half> %2038, i64 0, !dbg !52
  %address.local1032 = or disjoint i64 %address.local918, 64, !dbg !53
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1032, !dbg !53
  store half %2042, ptr addrspace(1) %address.local1033, align 2, !dbg !53
  %address.local1035 = or disjoint i64 %address.local918, 65, !dbg !53
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1035, !dbg !53
  store half %2041, ptr addrspace(1) %address.local1036, align 2, !dbg !53
  %address.local1038 = or disjoint i64 %address.local918, 66, !dbg !53
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1038, !dbg !53
  store half %2040, ptr addrspace(1) %address.local1039, align 2, !dbg !53
  %address.local1041 = or disjoint i64 %address.local918, 67, !dbg !53
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1041, !dbg !53
  store half %2039, ptr addrspace(1) %address.local1042, align 2, !dbg !53
  %address.local1044 = or disjoint i64 %address.local918, 80, !dbg !53
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1044, !dbg !53
  store half %2036, ptr addrspace(1) %address.local1045, align 2, !dbg !53
  %address.local1047 = or disjoint i64 %address.local918, 81, !dbg !53
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1047, !dbg !53
  store half %2035, ptr addrspace(1) %address.local1048, align 2, !dbg !53
  %address.local1050 = or disjoint i64 %address.local918, 82, !dbg !53
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1050, !dbg !53
  store half %2034, ptr addrspace(1) %address.local1051, align 2, !dbg !53
  %address.local1053 = or disjoint i64 %address.local918, 83, !dbg !53
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1053, !dbg !53
  store half %2033, ptr addrspace(1) %address.local1054, align 2, !dbg !53
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1032, !dbg !53
  store half %2030, ptr addrspace(1) %address.local1059, align 2, !dbg !53
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1035, !dbg !53
  store half %2029, ptr addrspace(1) %address.local1061, align 2, !dbg !53
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1038, !dbg !53
  store half %2028, ptr addrspace(1) %address.local1063, align 2, !dbg !53
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1041, !dbg !53
  store half %2027, ptr addrspace(1) %address.local1065, align 2, !dbg !53
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1044, !dbg !53
  store half %2024, ptr addrspace(1) %address.local1067, align 2, !dbg !53
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1047, !dbg !53
  store half %2023, ptr addrspace(1) %address.local1069, align 2, !dbg !53
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1050, !dbg !53
  store half %2022, ptr addrspace(1) %address.local1071, align 2, !dbg !53
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1053, !dbg !53
  store half %2021, ptr addrspace(1) %address.local1073, align 2, !dbg !53
  %2043 = fmul <4 x float> %1925, %1551, !dbg !51
  %2044 = fptrunc <4 x float> %2043 to <4 x half>, !dbg !52
  %2045 = extractelement <4 x half> %2044, i64 3, !dbg !52
  %2046 = extractelement <4 x half> %2044, i64 2, !dbg !52
  %2047 = extractelement <4 x half> %2044, i64 1, !dbg !52
  %2048 = extractelement <4 x half> %2044, i64 0, !dbg !52
  %2049 = fmul <4 x float> %1925, %1518, !dbg !51
  %2050 = fptrunc <4 x float> %2049 to <4 x half>, !dbg !52
  %2051 = extractelement <4 x half> %2050, i64 3, !dbg !52
  %2052 = extractelement <4 x half> %2050, i64 2, !dbg !52
  %2053 = extractelement <4 x half> %2050, i64 1, !dbg !52
  %2054 = extractelement <4 x half> %2050, i64 0, !dbg !52
  %2055 = fmul <4 x float> %1924, %1484, !dbg !51
  %2056 = fptrunc <4 x float> %2055 to <4 x half>, !dbg !52
  %2057 = extractelement <4 x half> %2056, i64 3, !dbg !52
  %2058 = extractelement <4 x half> %2056, i64 2, !dbg !52
  %2059 = extractelement <4 x half> %2056, i64 1, !dbg !52
  %2060 = extractelement <4 x half> %2056, i64 0, !dbg !52
  %2061 = fmul <4 x float> %1924, %1451, !dbg !51
  %2062 = fptrunc <4 x float> %2061 to <4 x half>, !dbg !52
  %2063 = extractelement <4 x half> %2062, i64 3, !dbg !52
  %2064 = extractelement <4 x half> %2062, i64 2, !dbg !52
  %2065 = extractelement <4 x half> %2062, i64 1, !dbg !52
  %2066 = extractelement <4 x half> %2062, i64 0, !dbg !52
  %address.local1089 = or disjoint i64 %address.local918, 96, !dbg !53
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1089, !dbg !53
  store half %2066, ptr addrspace(1) %address.local1090, align 2, !dbg !53
  %address.local1092 = or disjoint i64 %address.local918, 97, !dbg !53
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1092, !dbg !53
  store half %2065, ptr addrspace(1) %address.local1093, align 2, !dbg !53
  %address.local1095 = or disjoint i64 %address.local918, 98, !dbg !53
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1095, !dbg !53
  store half %2064, ptr addrspace(1) %address.local1096, align 2, !dbg !53
  %address.local1098 = or disjoint i64 %address.local918, 99, !dbg !53
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1098, !dbg !53
  store half %2063, ptr addrspace(1) %address.local1099, align 2, !dbg !53
  %address.local1101 = or disjoint i64 %address.local918, 112, !dbg !53
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1101, !dbg !53
  store half %2060, ptr addrspace(1) %address.local1102, align 2, !dbg !53
  %address.local1104 = or disjoint i64 %address.local918, 113, !dbg !53
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1104, !dbg !53
  store half %2059, ptr addrspace(1) %address.local1105, align 2, !dbg !53
  %address.local1107 = or disjoint i64 %address.local918, 114, !dbg !53
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1107, !dbg !53
  store half %2058, ptr addrspace(1) %address.local1108, align 2, !dbg !53
  %address.local1110 = or i64 %address.local907, 115, !dbg !53
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1110, !dbg !53
  store half %2057, ptr addrspace(1) %address.local1111, align 2, !dbg !53
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1089, !dbg !53
  store half %2054, ptr addrspace(1) %address.local1116, align 2, !dbg !53
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1092, !dbg !53
  store half %2053, ptr addrspace(1) %address.local1118, align 2, !dbg !53
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1095, !dbg !53
  store half %2052, ptr addrspace(1) %address.local1120, align 2, !dbg !53
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1098, !dbg !53
  store half %2051, ptr addrspace(1) %address.local1122, align 2, !dbg !53
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1101, !dbg !53
  store half %2048, ptr addrspace(1) %address.local1124, align 2, !dbg !53
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1104, !dbg !53
  store half %2047, ptr addrspace(1) %address.local1126, align 2, !dbg !53
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1107, !dbg !53
  store half %2046, ptr addrspace(1) %address.local1128, align 2, !dbg !53
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1110, !dbg !53
  store half %2045, ptr addrspace(1) %address.local1130, align 2, !dbg !53
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
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_bounds_lowering/causal")
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
