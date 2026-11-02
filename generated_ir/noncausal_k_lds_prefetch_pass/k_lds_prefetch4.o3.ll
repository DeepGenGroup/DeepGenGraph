; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_lds_prefetch_pass/k_lds_prefetch4.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000003/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !7
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !7
  %5 = and i32 %kdma.lane, 31, !dbg !7
  %kdma.owner = or disjoint i32 %5, %kdma.wave, !dbg !7
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !7
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !7
  %kdma.intra = and i32 %6, 8, !dbg !7
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %10 = sext i32 %9 to i64, !dbg !7
  %11 = shl nsw i64 %10, 5, !dbg !7
  %12 = add nsw i64 %11, %8, !dbg !7
  %13 = lshr i64 %12, 5, !dbg !7
  %14 = and i64 %13, 576460752303423456, !dbg !7
  %15 = and i64 %8, 31, !dbg !7
  %16 = or disjoint i64 %14, %15, !dbg !7
  %17 = lshr i64 %12, 10, !dbg !8
  %18 = add nuw nsw i64 %13, %17, !dbg !8
  %19 = shl nuw nsw i64 %16, 7, !dbg !8
  %20 = shl i64 %17, 12, !dbg !8
  %21 = sub i64 %19, %20, !dbg !8
  %22 = shl i64 %18, 19, !dbg !8
  %23 = and i64 %22, 16252928, !dbg !8
  %24 = shl nsw i64 %21, 7, !dbg !8
  %25 = add i64 %24, %23, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %26 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %address.local413, %25, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %27 = insertelement <2 x i64> poison, i64 %26, i64 0, !dbg !8
  %28 = bitcast <2 x i64> %27 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %28, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local424, %address.local410, !dbg !8
  %.idx1400 = or disjoint i64 %address.local425, %25, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %29 = or disjoint i64 %25, 32, !dbg !8
  %30 = or disjoint i64 %address.local413, %29, !dbg !8
  %.tr = trunc i64 %30 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %31 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local406, i64 0, !dbg !8
  %32 = insertelement <4 x ptr addrspace(3)> %31, ptr addrspace(3) %address.local418, i64 1, !dbg !8
  %33 = insertelement <4 x ptr addrspace(3)> %32, ptr addrspace(3) %address.local427, i64 2, !dbg !8
  %34 = insertelement <4 x ptr addrspace(3)> %33, ptr addrspace(3) %address.local431, i64 3, !dbg !8
  %35 = ptrtoint <4 x ptr addrspace(3)> %34 to <4 x i64>, !dbg !8
  %36 = trunc <4 x i64> %35 to <4 x i32>, !dbg !8
  %37 = extractelement <4 x i32> %36, i64 0, !dbg !8
  %38 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %37), !dbg !8
  %dma.buffer.lds1398 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38), !dbg !8
  %39 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %40 = extractelement <4 x i32> %36, i64 1, !dbg !8
  %41 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %40), !dbg !8
  %dma.buffer.lds1395 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %41), !dbg !8
  %42 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %43 = extractelement <4 x i32> %36, i64 2, !dbg !8
  %44 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %43), !dbg !8
  %dma.buffer.lds1392 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %44), !dbg !8
  %45 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %46 = extractelement <4 x i32> %36, i64 3, !dbg !8
  %47 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %48 = or disjoint i64 %address.local425, %29, !dbg !8
  %.tr1401 = trunc i64 %48 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %dma.buffer.lds1389 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %47), !dbg !8
  %49 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %address.local435 = lshr i32 %address.thread, 2, !dbg !10
  %address.local436 = and i32 %address.local435, 4, !dbg !10
  %address.local437 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local438 = and i32 %address.local437, 384, !dbg !10
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !10
  %address.local440 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local441 = and i32 %address.local440, 120, !dbg !10
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !10
  %address.local443 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local444 = and i32 %address.local443, 1024, !dbg !10
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !10
  %address.local446 = zext nneg i32 %address.local445 to i64, !dbg !10
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local446, !dbg !10
  %50 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %51 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %52 = fmul <4 x half> %50, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %53 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %54 = fmul <4 x half> %51, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %55 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %56 = fmul <4 x half> %53, splat (half 1.275630e-01), !dbg !10
  %57 = fmul <4 x half> %55, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %58 = or disjoint i64 %25, 64, !dbg !8
  %59 = or disjoint i64 %address.local413, %58, !dbg !8
  %.tr1402 = trunc i64 %59 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %60 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %61 = or disjoint i64 %address.local425, %58, !dbg !8
  %.tr1403 = trunc i64 %61 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %62 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %64 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %65 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %66 = fmul <4 x half> %63, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %67 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %68 = fmul <4 x half> %64, splat (half 1.275630e-01), !dbg !10
  %69 = fmul <4 x half> %65, splat (half 1.275630e-01), !dbg !10
  %70 = fmul <4 x half> %67, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %71 = or disjoint i64 %25, 96, !dbg !8
  %72 = or disjoint i64 %address.local413, %71, !dbg !8
  %.tr1404 = trunc i64 %72 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %74 = or disjoint i64 %address.local425, %71, !dbg !8
  %.tr1405 = trunc i64 %74 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %75 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %76 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %77 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %79 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !10
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %81 = fmul <4 x half> %77, splat (half 1.275630e-01), !dbg !10
  %82 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !10
  %83 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %87 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !10
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %89 = fmul <4 x half> %85, splat (half 1.275630e-01), !dbg !10
  %90 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !10
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %92 = shl nuw nsw i32 %kdma.tid, 2, !dbg !12
  %93 = and i32 %92, 3084, !dbg !12
  %94 = and i32 %kdma.tid, 1, !dbg !12
  %95 = add nuw nsw i32 %94, 1, !dbg !12
  %narrow = or disjoint i32 %95, %93, !dbg !12
  %96 = lshr i32 %narrow, 1, !dbg !12
  %97 = zext nneg i32 %96 to i64, !dbg !12
  %98 = shl i64 %13, 19
  %99 = and i64 %98, 16252928
  %100 = ptrtoint ptr addrspace(1) %1 to i64
  %101 = insertelement <2 x i64> poison, i64 %100, i64 0
  %102 = bitcast <2 x i64> %101 to <4 x i32>
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %102, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %kdma.source = zext nneg i32 %kdma.owner to i64
  %kdma.source1506 = lshr i64 %kdma.source, 4
  %kdma.source1507 = and i64 %kdma.source1506, 8
  %kdma.source1508 = and i64 %kdma.source, 91
  %kdma.source1509 = lshr i64 %kdma.source, 2
  %kdma.source1510 = and i64 %kdma.source1509, 4
  %kdma.source1511 = add nuw nsw i64 %kdma.source1510, %kdma.source
  %kdma.source1512 = and i64 %kdma.source1511, 4
  %kdma.source1513 = or disjoint i64 %kdma.source1512, %kdma.source1508
  %kdma.source1514 = shl nuw nsw i64 %kdma.source1513, 7
  %kdma.source1515 = or disjoint i64 %kdma.source1514, %kdma.source1507
  %kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64
  %kdma.source1520 = lshr i64 %kdma.source1519, 4
  %kdma.source1521 = and i64 %kdma.source1520, 8
  %kdma.source1522 = and i64 %kdma.source1519, 123
  %kdma.source1523 = lshr i64 %kdma.source1519, 2
  %kdma.source1524 = and i64 %kdma.source1523, 4
  %kdma.source1525 = add nuw nsw i64 %kdma.source1524, %kdma.source1519
  %kdma.source1526 = and i64 %kdma.source1525, 4
  %kdma.source1527 = or disjoint i64 %kdma.source1526, %kdma.source1522
  %kdma.source1528 = shl nuw nsw i64 %kdma.source1527, 7
  %kdma.source1529 = or disjoint i64 %kdma.source1528, %kdma.source1521
  %kdma.source1546 = or disjoint i64 %kdma.source1515, 16
  %kdma.source1563 = or disjoint i64 %kdma.source1529, 16
  %103 = ptrtoint ptr addrspace(1) %2 to i64
  %104 = insertelement <2 x i64> poison, i64 %103, i64 0
  %105 = bitcast <2 x i64> %104 to <4 x i32>
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %105, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  br label %106, !dbg !11

106:                                              ; preds = %4, %106
  %107 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1039, %106 ]
  %108 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1035, %106 ]
  %109 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1031, %106 ]
  %110 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %1027, %106 ]
  %111 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %719, %106 ]
  %112 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1025, %106 ]
  %113 = phi i64 [ 0, %4 ], [ %1040, %106 ]
  %114 = shl nuw nsw i64 %113, 14, !dbg !13
  %115 = add nuw nsw i64 %114, %99, !dbg !13
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !8
  %address.local501 = and i64 %address.local500, 1536, !dbg !8
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !8
  %kdma.source1516 = or disjoint i64 %115, %kdma.source1515, !dbg !13
  %kdma.source1517 = trunc nuw nsw i64 %kdma.source1516 to i32, !dbg !13
  %kdma.source1518 = shl nuw nsw i32 %kdma.source1517, 1, !dbg !13
  %kdma.offset = or disjoint i32 %kdma.source1518, %kdma.intra, !dbg !13
  %kdma.source1530 = or disjoint i64 %115, %kdma.source1529, !dbg !13
  %kdma.source1531 = trunc nuw nsw i64 %kdma.source1530 to i32, !dbg !13
  %kdma.source1532 = shl nuw nsw i32 %kdma.source1531, 1, !dbg !13
  %kdma.offset1533 = or disjoint i32 %kdma.source1532, %kdma.intra.hi, !dbg !13
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %kdma.source1547 = or disjoint i64 %115, %kdma.source1546, !dbg !13
  %kdma.source1548 = trunc nuw nsw i64 %kdma.source1547 to i32, !dbg !13
  %kdma.source1549 = shl nuw nsw i32 %kdma.source1548, 1, !dbg !13
  %kdma.offset1550 = or disjoint i32 %kdma.source1549, %kdma.intra, !dbg !13
  %kdma.source1564 = or disjoint i64 %115, %kdma.source1563, !dbg !13
  %kdma.source1565 = trunc nuw nsw i64 %kdma.source1564 to i32, !dbg !13
  %kdma.source1566 = shl nuw nsw i32 %kdma.source1565, 1, !dbg !13
  %kdma.offset1567 = or disjoint i32 %kdma.source1566, %kdma.intra.hi, !dbg !13
  %116 = or disjoint i64 %115, 32, !dbg !13
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %kdma.source1580 = or disjoint i64 %116, %kdma.source1515, !dbg !13
  %kdma.source1581 = trunc nuw nsw i64 %kdma.source1580 to i32, !dbg !13
  %kdma.source1582 = shl nuw nsw i32 %kdma.source1581, 1, !dbg !13
  %kdma.offset1583 = or disjoint i32 %kdma.source1582, %kdma.intra, !dbg !13
  %kdma.source1596 = or disjoint i64 %116, %kdma.source1529, !dbg !13
  %kdma.source1597 = trunc nuw nsw i64 %kdma.source1596 to i32, !dbg !13
  %kdma.source1598 = shl nuw nsw i32 %kdma.source1597, 1, !dbg !13
  %kdma.offset1599 = or disjoint i32 %kdma.source1598, %kdma.intra.hi, !dbg !13
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %117 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local502, i64 0, !dbg !8
  %118 = insertelement <4 x ptr addrspace(3)> %117, ptr addrspace(3) %address.local519, i64 1, !dbg !8
  %119 = insertelement <4 x ptr addrspace(3)> %118, ptr addrspace(3) %address.local533, i64 2, !dbg !8
  %120 = insertelement <4 x ptr addrspace(3)> %119, ptr addrspace(3) %address.local537, i64 3, !dbg !8
  %121 = ptrtoint <4 x ptr addrspace(3)> %120 to <4 x i64>, !dbg !8
  %122 = trunc <4 x i64> %121 to <4 x i32>, !dbg !8
  %123 = extractelement <4 x i32> %122, i64 0, !dbg !13
  %124 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %123), !dbg !13
  %dma.buffer.lds1374 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %124), !dbg !13
  %125 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.lds1534 = add i32 %dma.buffer.lds1374, 512, !dbg !13
  %126 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1533, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %127 = extractelement <4 x i32> %122, i64 1, !dbg !13
  %128 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %127), !dbg !13
  %dma.buffer.lds1371 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %128), !dbg !13
  %129 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1550, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.lds1568 = add i32 %dma.buffer.lds1371, 512, !dbg !13
  %130 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1567, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %131 = extractelement <4 x i32> %122, i64 2, !dbg !13
  %132 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %131), !dbg !13
  %dma.buffer.lds1368 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !13
  %133 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1583, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.lds1600 = add i32 %dma.buffer.lds1368, 512, !dbg !13
  %134 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1599, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %135 = extractelement <4 x i32> %122, i64 3, !dbg !13
  %136 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %135), !dbg !13
  %dma.buffer.lds1365 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %136), !dbg !13
  %kdma.source1613 = or disjoint i64 %116, %kdma.source1546, !dbg !13
  %kdma.source1614 = trunc nuw nsw i64 %kdma.source1613 to i32, !dbg !13
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !13
  %kdma.offset1616 = or disjoint i32 %kdma.source1615, %kdma.intra, !dbg !13
  %137 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1616, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1630 = or disjoint i64 %116, %kdma.source1563, !dbg !13
  %kdma.source1631 = trunc nuw nsw i64 %kdma.source1630 to i32, !dbg !13
  %kdma.source1632 = shl nuw nsw i32 %kdma.source1631, 1, !dbg !13
  %kdma.offset1633 = or disjoint i32 %kdma.source1632, %kdma.intra.hi, !dbg !13
  %kdma.lds1634 = add i32 %dma.buffer.lds1365, 512, !dbg !13
  %138 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %address.local542 = lshr i64 %address.local499, 2, !dbg !12
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !12
  %address.local546 = and i64 %address.local545, 24, !dbg !12
  %address.local548 = shl i64 %address.local499, 62, !dbg !12
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !12
  %address.local550 = and i64 %address.local549, -64, !dbg !12
  %address.local553 = and i64 %address.local542, 12, !dbg !10
  %address.local555 = and i64 %address.local499, 63, !dbg !10
  %address.local556 = lshr i64 %address.local555, 5, !dbg !10
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !12
  %139 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !12
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !12
  %140 = getelementptr i8, ptr addrspace(3) %139, i64 %.idx1410, !dbg !12
  %141 = getelementptr [2 x i8], ptr addrspace(3) %140, i64 %address.local546, !dbg !12
  %142 = getelementptr [2 x i8], ptr addrspace(3) %141, i64 %address.local550, !dbg !12
  %143 = getelementptr [2 x i8], ptr addrspace(3) %142, i64 %address.local553, !dbg !12
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %143, i64 %address.local557, !dbg !12
  %144 = add nsw i64 %.idx1410, %.idx, !dbg !12
  %.idx1 = shl nuw nsw i64 %address.local546, 1, !dbg !12
  %145 = or disjoint i64 %144, %.idx1, !dbg !12
  %146 = or disjoint i64 %address.local550, %address.local553, !dbg !12
  %147 = shl nsw i64 %146, 1, !dbg !12
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032, !dbg !12
  %148 = add nsw i64 %145, %address.local559.idx, !dbg !12
  %149 = add nsw i64 %148, %147, !dbg !12
  %150 = lshr i64 %149, 6, !dbg !12
  %151 = and i64 %150, 8, !dbg !12
  %152 = xor i64 %151, %149, !dbg !12
  %153 = sub i64 %152, %149, !dbg !12
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %153, !dbg !12
  %154 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !12
  %155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %154, <4 x float> zeroinitializer) #8, !dbg !12
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !12
  %156 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !12
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !12
  %address.local571 = and i32 %address.local570, 24, !dbg !12
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !12
  %address.local573 = and i32 %address.local572, 768, !dbg !12
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !12
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !12
  %address.local576 = and i32 %address.local575, 96, !dbg !12
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !12
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !12
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %97) #8, !dbg !12, !frisk.register_lifetimes !5
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !12
  %address.local580 = and i32 %address.local579, 1, !dbg !12
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !12
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !12
  %address.local583 = shl i64 %address.local582, 6, !dbg !12
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !12
  %157 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !12
  %158 = getelementptr [2 x i8], ptr addrspace(3) %157, i64 %address.local553, !dbg !12
  %159 = getelementptr [2 x i8], ptr addrspace(3) %158, i64 %address.local557, !dbg !12
  %address.local588 = getelementptr i8, ptr addrspace(3) %159, i64 64, !dbg !12
  %.idx47 = add i64 %address.local584, %address.local553, !dbg !12
  %160 = shl i64 %.idx47, 1, !dbg !12
  %161 = add nuw nsw i64 %address.local559.idx, 64, !dbg !12
  %162 = add i64 %161, %160, !dbg !12
  %163 = lshr i64 %162, 6, !dbg !12
  %164 = and i64 %163, 8, !dbg !12
  %165 = xor i64 %164, %162, !dbg !12
  %166 = sub i64 %165, %162, !dbg !12
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %166, !dbg !12
  %167 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !12, !frisk.k_lds_pair !14
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !12
  %168 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !12, !frisk.k_lds_pair !14
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !12
  %169 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !12, !frisk.k_lds_pair !15
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !12
  %170 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !12, !frisk.k_lds_pair !15
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !12
  %171 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !12, !frisk.k_lds_pair !16
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !12
  %172 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !12, !frisk.k_lds_pair !16
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !12
  %173 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !12, !frisk.k_lds_pair !17
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !12
  %174 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !12, !frisk.k_lds_pair !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %156, <4 x float> %155) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %167, <4 x float> zeroinitializer) #8, !dbg !12
  %177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %169, <4 x float> %176) #8, !dbg !12
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %171, <4 x float> zeroinitializer) #8, !dbg !12
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %173, <4 x float> %178) #8, !dbg !12
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !12
  %180 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !12, !frisk.k_lds_pair !18
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !12
  %181 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !12, !frisk.k_lds_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %168, <4 x float> zeroinitializer) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !12
  %183 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !12, !frisk.k_lds_pair !19
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !12
  %184 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !12, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %170, <4 x float> %182) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !12
  %186 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !12, !frisk.k_lds_pair !20
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !12
  %187 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !12, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %172, <4 x float> zeroinitializer) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %174, <4 x float> %188) #8, !dbg !12
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %180, <4 x float> zeroinitializer) #8, !dbg !12
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %183, <4 x float> %190) #8, !dbg !12
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %186, <4 x float> zeroinitializer) #8, !dbg !12
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %187, <4 x float> %192) #8, !dbg !12
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %181, <4 x float> zeroinitializer) #8, !dbg !12
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %184, <4 x float> %194) #8, !dbg !12
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %154, <4 x float> zeroinitializer) #8, !dbg !12
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %156, <4 x float> %196) #8, !dbg !12
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %167, <4 x float> zeroinitializer) #8, !dbg !12
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %169, <4 x float> %198) #8, !dbg !12
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %171, <4 x float> zeroinitializer) #8, !dbg !12
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %173, <4 x float> %200) #8, !dbg !12
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %168, <4 x float> zeroinitializer) #8, !dbg !12
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %170, <4 x float> %202) #8, !dbg !12
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %172, <4 x float> zeroinitializer) #8, !dbg !12
  %205 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %174, <4 x float> %204) #8, !dbg !12
  %206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %180, <4 x float> zeroinitializer) #8, !dbg !12
  %207 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %183, <4 x float> %206) #8, !dbg !12
  %208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %186, <4 x float> zeroinitializer) #8, !dbg !12
  %209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %187, <4 x float> %208) #8, !dbg !12
  %210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %181, <4 x float> zeroinitializer) #8, !dbg !12
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %184, <4 x float> %210) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %212 = or disjoint i64 %115, 64, !dbg !13
  %kdma.source1646 = or disjoint i64 %212, %kdma.source1515, !dbg !13
  %kdma.source1647 = trunc nuw nsw i64 %kdma.source1646 to i32, !dbg !13
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !13
  %kdma.offset1649 = or disjoint i32 %kdma.source1648, %kdma.intra, !dbg !13
  %213 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset1649, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1662 = or disjoint i64 %212, %kdma.source1529, !dbg !13
  %kdma.source1663 = trunc nuw nsw i64 %kdma.source1662 to i32, !dbg !13
  %kdma.source1664 = shl nuw nsw i32 %kdma.source1663, 1, !dbg !13
  %kdma.offset1665 = or disjoint i32 %kdma.source1664, %kdma.intra.hi, !dbg !13
  %214 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1665, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1679 = or disjoint i64 %212, %kdma.source1546, !dbg !13
  %kdma.source1680 = trunc nuw nsw i64 %kdma.source1679 to i32, !dbg !13
  %kdma.source1681 = shl nuw nsw i32 %kdma.source1680, 1, !dbg !13
  %kdma.offset1682 = or disjoint i32 %kdma.source1681, %kdma.intra, !dbg !13
  %215 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1682, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1696 = or disjoint i64 %212, %kdma.source1563, !dbg !13
  %kdma.source1697 = trunc nuw nsw i64 %kdma.source1696 to i32, !dbg !13
  %kdma.source1698 = shl nuw nsw i32 %kdma.source1697, 1, !dbg !13
  %kdma.offset1699 = or disjoint i32 %kdma.source1698, %kdma.intra.hi, !dbg !13
  %216 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1699, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !12
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !12, !frisk.k_lds_pair !21
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !12
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !12, !frisk.k_lds_pair !21
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !12
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !12, !frisk.k_lds_pair !22
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !12
  %220 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !12, !frisk.k_lds_pair !22
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !12
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !12, !frisk.k_lds_pair !23
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !12
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !12, !frisk.k_lds_pair !23
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !12
  %223 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !12, !frisk.k_lds_pair !24
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !12
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !12, !frisk.k_lds_pair !24
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %217, <4 x float> %175) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %219, <4 x float> %225) #8, !dbg !12
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %221, <4 x float> %177) #8, !dbg !12
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %223, <4 x float> %227) #8, !dbg !12
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !12
  %229 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !12, !frisk.k_lds_pair !25
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !12
  %230 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !12, !frisk.k_lds_pair !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %218, <4 x float> %179) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !12
  %232 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !12, !frisk.k_lds_pair !26
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !12
  %233 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !12, !frisk.k_lds_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %220, <4 x float> %231) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !12
  %235 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !12, !frisk.k_lds_pair !27
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !12
  %236 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !12, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %222, <4 x float> %185) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !12
  %238 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !12, !frisk.k_lds_pair !28
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !12
  %239 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !12, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %224, <4 x float> %237) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %229, <4 x float> %189) #8, !dbg !12
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %232, <4 x float> %241) #8, !dbg !12
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %235, <4 x float> %191) #8, !dbg !12
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %238, <4 x float> %243) #8, !dbg !12
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %230, <4 x float> %193) #8, !dbg !12
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %233, <4 x float> %245) #8, !dbg !12
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %236, <4 x float> %195) #8, !dbg !12
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %239, <4 x float> %247) #8, !dbg !12
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %217, <4 x float> %197) #8, !dbg !12
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %219, <4 x float> %249) #8, !dbg !12
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %221, <4 x float> %199) #8, !dbg !12
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %223, <4 x float> %251) #8, !dbg !12
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %218, <4 x float> %201) #8, !dbg !12
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %220, <4 x float> %253) #8, !dbg !12
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %222, <4 x float> %203) #8, !dbg !12
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %224, <4 x float> %255) #8, !dbg !12
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %229, <4 x float> %205) #8, !dbg !12
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %232, <4 x float> %257) #8, !dbg !12
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %235, <4 x float> %207) #8, !dbg !12
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %238, <4 x float> %259) #8, !dbg !12
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %230, <4 x float> %209) #8, !dbg !12
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %233, <4 x float> %261) #8, !dbg !12
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %236, <4 x float> %211) #8, !dbg !12
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %239, <4 x float> %263) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %265 = or disjoint i64 %115, 96, !dbg !13
  %kdma.source1712 = or disjoint i64 %265, %kdma.source1515, !dbg !13
  %kdma.source1713 = trunc nuw nsw i64 %kdma.source1712 to i32, !dbg !13
  %kdma.source1714 = shl nuw nsw i32 %kdma.source1713, 1, !dbg !13
  %kdma.offset1715 = or disjoint i32 %kdma.source1714, %kdma.intra, !dbg !13
  %266 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1715, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1728 = or disjoint i64 %265, %kdma.source1529, !dbg !13
  %kdma.source1729 = trunc nuw nsw i64 %kdma.source1728 to i32, !dbg !13
  %kdma.source1730 = shl nuw nsw i32 %kdma.source1729, 1, !dbg !13
  %kdma.offset1731 = or disjoint i32 %kdma.source1730, %kdma.intra.hi, !dbg !13
  %267 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1731, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1745 = or disjoint i64 %265, %kdma.source1546, !dbg !13
  %kdma.source1746 = trunc nuw nsw i64 %kdma.source1745 to i32, !dbg !13
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !13
  %kdma.offset1748 = or disjoint i32 %kdma.source1747, %kdma.intra, !dbg !13
  %268 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1748, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1762 = or disjoint i64 %265, %kdma.source1563, !dbg !13
  %kdma.source1763 = trunc nuw nsw i64 %kdma.source1762 to i32, !dbg !13
  %kdma.source1764 = shl nuw nsw i32 %kdma.source1763, 1, !dbg !13
  %kdma.offset1765 = or disjoint i32 %kdma.source1764, %kdma.intra.hi, !dbg !13
  %269 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1765, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %270 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !12, !frisk.k_lds_pair !29
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !12, !frisk.k_lds_pair !29
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !12, !frisk.k_lds_pair !30
  %273 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !12, !frisk.k_lds_pair !30
  %274 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !12, !frisk.k_lds_pair !31
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !12, !frisk.k_lds_pair !31
  %276 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !12, !frisk.k_lds_pair !32
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !12, !frisk.k_lds_pair !32
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %270, <4 x float> %226) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %272, <4 x float> %278) #8, !dbg !12
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %274, <4 x float> %228) #8, !dbg !12
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %276, <4 x float> %280) #8, !dbg !12
  %282 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !12, !frisk.k_lds_pair !33
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !12, !frisk.k_lds_pair !33
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %271, <4 x float> %234) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %285 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !12, !frisk.k_lds_pair !34
  %286 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !12, !frisk.k_lds_pair !34
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %273, <4 x float> %284) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %288 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !12, !frisk.k_lds_pair !35
  %289 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !12, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %275, <4 x float> %240) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %291 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !12, !frisk.k_lds_pair !36
  %292 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !12, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %277, <4 x float> %290) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %282, <4 x float> %242) #8, !dbg !12
  %295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %285, <4 x float> %294) #8, !dbg !12
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %288, <4 x float> %244) #8, !dbg !12
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %291, <4 x float> %296) #8, !dbg !12
  %298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %283, <4 x float> %246) #8, !dbg !12
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %286, <4 x float> %298) #8, !dbg !12
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %289, <4 x float> %248) #8, !dbg !12
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %292, <4 x float> %300) #8, !dbg !12
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %270, <4 x float> %250) #8, !dbg !12
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %272, <4 x float> %302) #8, !dbg !12
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %274, <4 x float> %252) #8, !dbg !12
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %276, <4 x float> %304) #8, !dbg !12
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %271, <4 x float> %254) #8, !dbg !12
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %273, <4 x float> %306) #8, !dbg !12
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %275, <4 x float> %256) #8, !dbg !12
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %277, <4 x float> %308) #8, !dbg !12
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %282, <4 x float> %258) #8, !dbg !12
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %285, <4 x float> %310) #8, !dbg !12
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %288, <4 x float> %260) #8, !dbg !12
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %291, <4 x float> %312) #8, !dbg !12
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %283, <4 x float> %262) #8, !dbg !12
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %286, <4 x float> %314) #8, !dbg !12
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %289, <4 x float> %264) #8, !dbg !12
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %292, <4 x float> %316) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !37
  %address.local918 = and i64 %address.local917, 8064, !dbg !37
  %address.local920 = and i64 %address.local500, 24, !dbg !37
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !37
  %.idx14191420 = or disjoint i64 %address.local921, %115, !dbg !37
  %.idx14191420.tr = trunc nuw nsw i64 %.idx14191420 to i32, !dbg !37
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !37
  %318 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !37
  %address.local924 = and i64 %address.local923, 16256, !dbg !37
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !37
  %.idx14211422 = or disjoint i64 %address.local927, %115, !dbg !37
  %.idx14211422.tr = trunc nuw nsw i64 %.idx14211422 to i32, !dbg !37
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !37
  %319 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %320 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !12
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %320, <4 x float> %279) #8, !dbg !12
  %322 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !12
  %323 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !12
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %322, <4 x float> %321) #8, !dbg !12
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %323, <4 x float> %281) #8, !dbg !12
  %326 = extractelement <4 x float> %324, i64 0, !dbg !38
  %327 = tail call float @llvm.maxnum.f32(float %326, float -inf), !dbg !39, !frisk.local_reduction !5
  %328 = extractelement <4 x float> %324, i64 1, !dbg !38
  %329 = extractelement <4 x float> %324, i64 2, !dbg !38
  %330 = extractelement <4 x float> %324, i64 3, !dbg !38
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !12
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !12
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %331, <4 x float> %325) #8, !dbg !12
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %332, <4 x float> %287) #8, !dbg !12
  %335 = extractelement <4 x float> %333, i64 0, !dbg !38
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %327, float %335), !dbg !39, !frisk.local_reduction !5
  %336 = extractelement <4 x float> %333, i64 1, !dbg !38
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %328, float %336), !dbg !39, !frisk.local_reduction !5
  %337 = extractelement <4 x float> %333, i64 2, !dbg !38
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %329, float %337), !dbg !39, !frisk.local_reduction !5
  %338 = extractelement <4 x float> %333, i64 3, !dbg !38
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %330, float %338), !dbg !39, !frisk.local_reduction !5
  %339 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !12
  %340 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !12
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %339, <4 x float> %334) #8, !dbg !12
  %342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %340, <4 x float> %293) #8, !dbg !12
  %343 = extractelement <4 x float> %341, i64 0, !dbg !38
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %343), !dbg !39, !frisk.local_reduction !5
  %344 = extractelement <4 x float> %341, i64 1, !dbg !38
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %344), !dbg !39, !frisk.local_reduction !5
  %345 = extractelement <4 x float> %341, i64 2, !dbg !38
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %345), !dbg !39, !frisk.local_reduction !5
  %346 = extractelement <4 x float> %341, i64 3, !dbg !38
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %346), !dbg !39, !frisk.local_reduction !5
  %347 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !12
  %348 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !12
  %349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %347, <4 x float> %342) #8, !dbg !12
  %350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %348, <4 x float> %295) #8, !dbg !12
  %351 = extractelement <4 x float> %349, i64 0, !dbg !38
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %351), !dbg !39, !frisk.local_reduction !5
  %352 = extractelement <4 x float> %349, i64 1, !dbg !38
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %352), !dbg !39, !frisk.local_reduction !5
  %353 = extractelement <4 x float> %349, i64 2, !dbg !38
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %353), !dbg !39, !frisk.local_reduction !5
  %354 = extractelement <4 x float> %349, i64 3, !dbg !38
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %354), !dbg !39, !frisk.local_reduction !5
  %355 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !12
  %356 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !12
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %355, <4 x float> %350) #8, !dbg !12
  %358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %356, <4 x float> %297) #8, !dbg !12
  %359 = extractelement <4 x float> %357, i64 0, !dbg !38
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %359), !dbg !39, !frisk.local_reduction !5
  %360 = extractelement <4 x float> %357, i64 1, !dbg !38
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %360), !dbg !39, !frisk.local_reduction !5
  %361 = extractelement <4 x float> %357, i64 2, !dbg !38
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %361), !dbg !39, !frisk.local_reduction !5
  %362 = extractelement <4 x float> %357, i64 3, !dbg !38
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %362), !dbg !39, !frisk.local_reduction !5
  %363 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !12
  %364 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !12
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %363, <4 x float> %358) #8, !dbg !12
  %366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %364, <4 x float> %299) #8, !dbg !12
  %367 = extractelement <4 x float> %365, i64 0, !dbg !38
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %367), !dbg !39, !frisk.local_reduction !5
  %368 = extractelement <4 x float> %365, i64 1, !dbg !38
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %368), !dbg !39, !frisk.local_reduction !5
  %369 = extractelement <4 x float> %365, i64 2, !dbg !38
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %369), !dbg !39, !frisk.local_reduction !5
  %370 = extractelement <4 x float> %365, i64 3, !dbg !38
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %370), !dbg !39, !frisk.local_reduction !5
  %371 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !12
  %372 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !12
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %371, <4 x float> %366) #8, !dbg !12
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %372, <4 x float> %301) #8, !dbg !12
  %375 = extractelement <4 x float> %373, i64 0, !dbg !38
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %375), !dbg !39, !frisk.local_reduction !5
  %376 = extractelement <4 x float> %373, i64 1, !dbg !38
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %376), !dbg !39, !frisk.local_reduction !5
  %377 = extractelement <4 x float> %373, i64 2, !dbg !38
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %377), !dbg !39, !frisk.local_reduction !5
  %378 = extractelement <4 x float> %373, i64 3, !dbg !38
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %378), !dbg !39, !frisk.local_reduction !5
  %379 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !12
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %379, <4 x float> %374) #8, !dbg !12
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %320, <4 x float> %303) #8, !dbg !12
  %382 = extractelement <4 x float> %380, i64 0, !dbg !38
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %382), !dbg !39, !frisk.local_reduction !5
  %383 = extractelement <4 x float> %380, i64 1, !dbg !38
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %383), !dbg !39, !frisk.local_reduction !5
  %384 = extractelement <4 x float> %380, i64 2, !dbg !38
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %384), !dbg !39, !frisk.local_reduction !5
  %385 = extractelement <4 x float> %380, i64 3, !dbg !38
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %385), !dbg !39, !frisk.local_reduction !5
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !39, !frisk.local_reduction !5
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !39, !frisk.local_reduction !5
  %386 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !39, !frisk.local_reduction !5
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %322, <4 x float> %381) #8, !dbg !12
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %323, <4 x float> %305) #8, !dbg !12
  %389 = extractelement <4 x float> %387, i64 0, !dbg !38
  %390 = extractelement <4 x float> %387, i64 1, !dbg !38
  %391 = extractelement <4 x float> %387, i64 2, !dbg !38
  %392 = extractelement <4 x float> %387, i64 3, !dbg !38
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %331, <4 x float> %388) #8, !dbg !12
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %332, <4 x float> %307) #8, !dbg !12
  %395 = extractelement <4 x float> %393, i64 0, !dbg !38
  %396 = extractelement <4 x float> %393, i64 1, !dbg !38
  %397 = extractelement <4 x float> %393, i64 2, !dbg !38
  %398 = extractelement <4 x float> %393, i64 3, !dbg !38
  %399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %339, <4 x float> %394) #8, !dbg !12
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %340, <4 x float> %309) #8, !dbg !12
  %401 = extractelement <4 x float> %399, i64 0, !dbg !38
  %402 = extractelement <4 x float> %399, i64 1, !dbg !38
  %403 = extractelement <4 x float> %399, i64 2, !dbg !38
  %404 = extractelement <4 x float> %399, i64 3, !dbg !38
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %347, <4 x float> %400) #8, !dbg !12
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %348, <4 x float> %311) #8, !dbg !12
  %407 = extractelement <4 x float> %405, i64 0, !dbg !38
  %408 = extractelement <4 x float> %405, i64 1, !dbg !38
  %409 = extractelement <4 x float> %405, i64 2, !dbg !38
  %410 = extractelement <4 x float> %405, i64 3, !dbg !38
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %355, <4 x float> %406) #8, !dbg !12
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %356, <4 x float> %313) #8, !dbg !12
  %413 = extractelement <4 x float> %411, i64 0, !dbg !38
  %414 = extractelement <4 x float> %411, i64 1, !dbg !38
  %415 = extractelement <4 x float> %411, i64 2, !dbg !38
  %416 = extractelement <4 x float> %411, i64 3, !dbg !38
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %363, <4 x float> %412) #8, !dbg !12
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %364, <4 x float> %315) #8, !dbg !12
  %419 = extractelement <4 x float> %417, i64 0, !dbg !38
  %420 = extractelement <4 x float> %417, i64 1, !dbg !38
  %421 = extractelement <4 x float> %417, i64 2, !dbg !38
  %422 = extractelement <4 x float> %417, i64 3, !dbg !38
  %423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %371, <4 x float> %418) #8, !dbg !12
  %424 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %372, <4 x float> %317) #8, !dbg !12
  %425 = extractelement <4 x float> %423, i64 0, !dbg !38
  %426 = extractelement <4 x float> %423, i64 1, !dbg !38
  %427 = extractelement <4 x float> %423, i64 2, !dbg !38
  %428 = extractelement <4 x float> %423, i64 3, !dbg !38
  %429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %379, <4 x float> %424) #8, !dbg !12
  %430 = extractelement <4 x float> %429, i64 0, !dbg !38
  %431 = extractelement <4 x float> %429, i64 1, !dbg !38
  %432 = extractelement <4 x float> %429, i64 2, !dbg !38
  %433 = extractelement <4 x float> %429, i64 3, !dbg !38
  %434 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !39
  %435 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %434), !dbg !39
  %436 = add nuw nsw i32 %435, 64, !dbg !39
  %437 = xor i32 %435, 16, !dbg !39
  %438 = and i32 %436, 192, !dbg !39
  %439 = icmp samesign ult i32 %437, %438, !dbg !39
  %440 = select i1 %439, i32 %437, i32 %435, !dbg !39
  %441 = shl nuw nsw i32 %440, 2, !dbg !39
  %442 = bitcast float %386 to i32, !dbg !39
  %443 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %442), !dbg !39
  %444 = bitcast i32 %443 to float, !dbg !39
  %445 = tail call float @llvm.maxnum.f32(float %386, float %444), !dbg !39
  %446 = xor i32 %435, 32, !dbg !39
  %447 = icmp samesign ult i32 %446, %438, !dbg !39
  %448 = select i1 %447, i32 %446, i32 %435, !dbg !39
  %449 = shl nuw nsw i32 %448, 2, !dbg !39
  %450 = bitcast float %445 to i32, !dbg !39
  %451 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %450), !dbg !39
  %452 = bitcast i32 %451 to float, !dbg !39
  %453 = tail call float @llvm.maxnum.f32(float %445, float %452), !dbg !39
  %454 = extractvalue [2 x <1 x float>] %110, 0, !dbg !39
  %455 = insertelement <1 x float> poison, float %453, i64 0, !dbg !39
  %456 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %455, <1 x float> %454), !dbg !39
  %457 = tail call float @llvm.maxnum.f32(float %389, float -inf), !dbg !39, !frisk.local_reduction !5
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %457, float %395), !dbg !39, !frisk.local_reduction !5
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %390, float %396), !dbg !39, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %391, float %397), !dbg !39, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %392, float %398), !dbg !39, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %401), !dbg !39, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %402), !dbg !39, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %403), !dbg !39, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %404), !dbg !39, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %407), !dbg !39, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %408), !dbg !39, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %409), !dbg !39, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %410), !dbg !39, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %413), !dbg !39, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %414), !dbg !39, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %415), !dbg !39, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %416), !dbg !39, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %419), !dbg !39, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %420), !dbg !39, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %421), !dbg !39, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %422), !dbg !39, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %425), !dbg !39, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %426), !dbg !39, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %427), !dbg !39, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %428), !dbg !39, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %430), !dbg !39, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %431), !dbg !39, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %432), !dbg !39, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %433), !dbg !39, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !39, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !39, !frisk.local_reduction !5
  %458 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !39, !frisk.local_reduction !5
  %459 = bitcast float %458 to i32, !dbg !39
  %460 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %459), !dbg !39
  %461 = bitcast i32 %460 to float, !dbg !39
  %462 = tail call float @llvm.maxnum.f32(float %458, float %461), !dbg !39
  %463 = bitcast float %462 to i32, !dbg !39
  %464 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %463), !dbg !39
  %465 = bitcast i32 %464 to float, !dbg !39
  %466 = tail call float @llvm.maxnum.f32(float %462, float %465), !dbg !39
  %467 = extractvalue [2 x <1 x float>] %110, 1, !dbg !39
  %468 = insertelement <1 x float> poison, float %466, i64 0, !dbg !39
  %469 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %468, <1 x float> %467), !dbg !39
  %470 = shufflevector <1 x float> %456, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %471 = fsub <4 x float> %324, %470, !dbg !40
  %472 = fsub <4 x float> %333, %470, !dbg !40
  %473 = fsub <4 x float> %341, %470, !dbg !40
  %474 = fsub <4 x float> %349, %470, !dbg !40
  %475 = fsub <4 x float> %357, %470, !dbg !40
  %476 = fsub <4 x float> %365, %470, !dbg !40
  %477 = fsub <4 x float> %373, %470, !dbg !40
  %478 = fsub <4 x float> %380, %470, !dbg !40
  %479 = shufflevector <1 x float> %469, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %480 = fsub <4 x float> %387, %479, !dbg !40
  %481 = fsub <4 x float> %393, %479, !dbg !40
  %482 = fsub <4 x float> %399, %479, !dbg !40
  %483 = fsub <4 x float> %405, %479, !dbg !40
  %484 = fsub <4 x float> %411, %479, !dbg !40
  %485 = fsub <4 x float> %417, %479, !dbg !40
  %486 = fsub <4 x float> %423, %479, !dbg !40
  %487 = fsub <4 x float> %429, %479, !dbg !40
  %488 = fsub <1 x float> %454, %456, !dbg !41
  %489 = extractelement <1 x float> %488, i64 0, !dbg !42
  %490 = tail call float @llvm.amdgcn.exp2.f32(float %489), !dbg !42
  %491 = insertelement <4 x float> poison, float %490, i64 0, !dbg !43
  %492 = shufflevector <4 x float> %491, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %493 = extractvalue [2 x <1 x float>] %111, 0, !dbg !44
  %494 = extractelement <1 x float> %493, i64 0, !dbg !44
  %495 = fmul float %494, %490, !dbg !44
  %496 = fsub <1 x float> %467, %469, !dbg !41
  %497 = extractelement <1 x float> %496, i64 0, !dbg !42
  %498 = tail call float @llvm.amdgcn.exp2.f32(float %497), !dbg !42
  %499 = insertelement <4 x float> poison, float %498, i64 0, !dbg !43
  %500 = shufflevector <4 x float> %499, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %501 = extractvalue [2 x <1 x float>] %111, 1, !dbg !44
  %502 = extractelement <1 x float> %501, i64 0, !dbg !44
  %503 = fmul float %502, %498, !dbg !44
  %504 = extractelement <4 x float> %471, i64 0, !dbg !45
  %505 = tail call float @llvm.amdgcn.exp2.f32(float %504), !dbg !45
  %506 = insertelement <4 x float> poison, float %505, i64 0, !dbg !46
  %507 = extractelement <4 x float> %471, i64 1, !dbg !45
  %508 = tail call float @llvm.amdgcn.exp2.f32(float %507), !dbg !45
  %509 = insertelement <4 x float> %506, float %508, i64 1, !dbg !46
  %510 = extractelement <4 x float> %471, i64 2, !dbg !45
  %511 = tail call float @llvm.amdgcn.exp2.f32(float %510), !dbg !45
  %512 = insertelement <4 x float> %509, float %511, i64 2, !dbg !46
  %513 = extractelement <4 x float> %471, i64 3, !dbg !45
  %514 = tail call float @llvm.amdgcn.exp2.f32(float %513), !dbg !45
  %515 = insertelement <4 x float> %512, float %514, i64 3, !dbg !46
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %515) #8, !dbg !46, !frisk.register_lifetimes !5
  %516 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !46
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %516) #8, !dbg !45, !frisk.register_lifetimes !5
  %517 = extractelement <4 x float> %472, i64 0, !dbg !45
  %518 = tail call float @llvm.amdgcn.exp2.f32(float %517), !dbg !45
  %519 = insertelement <4 x float> poison, float %518, i64 0, !dbg !46
  %local.reduce89 = fadd float %505, %518, !dbg !47, !frisk.local_reduction !5
  %520 = extractelement <4 x float> %472, i64 1, !dbg !45
  %521 = tail call float @llvm.amdgcn.exp2.f32(float %520), !dbg !45
  %522 = insertelement <4 x float> %519, float %521, i64 1, !dbg !46
  %local.reduce90 = fadd float %508, %521, !dbg !47, !frisk.local_reduction !5
  %523 = extractelement <4 x float> %472, i64 2, !dbg !45
  %524 = tail call float @llvm.amdgcn.exp2.f32(float %523), !dbg !45
  %525 = insertelement <4 x float> %522, float %524, i64 2, !dbg !46
  %local.reduce91 = fadd float %511, %524, !dbg !47, !frisk.local_reduction !5
  %526 = extractelement <4 x float> %472, i64 3, !dbg !45
  %527 = tail call float @llvm.amdgcn.exp2.f32(float %526), !dbg !45
  %528 = insertelement <4 x float> %525, float %527, i64 3, !dbg !46
  %math.result374 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %528) #8, !dbg !46, !frisk.register_lifetimes !5
  %529 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !46
  %math.packed375 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %529) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce92 = fadd float %514, %527, !dbg !47, !frisk.local_reduction !5
  %530 = extractelement <4 x float> %473, i64 0, !dbg !45
  %531 = tail call float @llvm.amdgcn.exp2.f32(float %530), !dbg !45
  %532 = insertelement <4 x float> poison, float %531, i64 0, !dbg !46
  %local.reduce93 = fadd float %531, %local.reduce89, !dbg !47, !frisk.local_reduction !5
  %533 = extractelement <4 x float> %473, i64 1, !dbg !45
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %533), !dbg !45
  %535 = insertelement <4 x float> %532, float %534, i64 1, !dbg !46
  %local.reduce94 = fadd float %534, %local.reduce90, !dbg !47, !frisk.local_reduction !5
  %536 = extractelement <4 x float> %473, i64 2, !dbg !45
  %537 = tail call float @llvm.amdgcn.exp2.f32(float %536), !dbg !45
  %538 = insertelement <4 x float> %535, float %537, i64 2, !dbg !46
  %local.reduce95 = fadd float %537, %local.reduce91, !dbg !47, !frisk.local_reduction !5
  %539 = extractelement <4 x float> %473, i64 3, !dbg !45
  %540 = tail call float @llvm.amdgcn.exp2.f32(float %539), !dbg !45
  %541 = insertelement <4 x float> %538, float %540, i64 3, !dbg !46
  %math.result376 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %541) #8, !dbg !46, !frisk.register_lifetimes !5
  %542 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !46
  %math.packed377 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %542) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce96 = fadd float %540, %local.reduce92, !dbg !47, !frisk.local_reduction !5
  %543 = extractelement <4 x float> %474, i64 0, !dbg !45
  %544 = tail call float @llvm.amdgcn.exp2.f32(float %543), !dbg !45
  %545 = insertelement <4 x float> poison, float %544, i64 0, !dbg !46
  %local.reduce97 = fadd float %544, %local.reduce93, !dbg !47, !frisk.local_reduction !5
  %546 = extractelement <4 x float> %474, i64 1, !dbg !45
  %547 = tail call float @llvm.amdgcn.exp2.f32(float %546), !dbg !45
  %548 = insertelement <4 x float> %545, float %547, i64 1, !dbg !46
  %local.reduce98 = fadd float %547, %local.reduce94, !dbg !47, !frisk.local_reduction !5
  %549 = extractelement <4 x float> %474, i64 2, !dbg !45
  %550 = tail call float @llvm.amdgcn.exp2.f32(float %549), !dbg !45
  %551 = insertelement <4 x float> %548, float %550, i64 2, !dbg !46
  %local.reduce99 = fadd float %550, %local.reduce95, !dbg !47, !frisk.local_reduction !5
  %552 = extractelement <4 x float> %474, i64 3, !dbg !45
  %553 = tail call float @llvm.amdgcn.exp2.f32(float %552), !dbg !45
  %554 = insertelement <4 x float> %551, float %553, i64 3, !dbg !46
  %math.result378 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %554) #8, !dbg !46, !frisk.register_lifetimes !5
  %555 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !46
  %math.packed379 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %555) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce100 = fadd float %553, %local.reduce96, !dbg !47, !frisk.local_reduction !5
  %556 = extractelement <4 x float> %475, i64 0, !dbg !45
  %557 = tail call float @llvm.amdgcn.exp2.f32(float %556), !dbg !45
  %558 = insertelement <4 x float> poison, float %557, i64 0, !dbg !46
  %local.reduce101 = fadd float %557, %local.reduce97, !dbg !47, !frisk.local_reduction !5
  %559 = extractelement <4 x float> %475, i64 1, !dbg !45
  %560 = tail call float @llvm.amdgcn.exp2.f32(float %559), !dbg !45
  %561 = insertelement <4 x float> %558, float %560, i64 1, !dbg !46
  %local.reduce102 = fadd float %560, %local.reduce98, !dbg !47, !frisk.local_reduction !5
  %562 = extractelement <4 x float> %475, i64 2, !dbg !45
  %563 = tail call float @llvm.amdgcn.exp2.f32(float %562), !dbg !45
  %564 = insertelement <4 x float> %561, float %563, i64 2, !dbg !46
  %local.reduce103 = fadd float %563, %local.reduce99, !dbg !47, !frisk.local_reduction !5
  %565 = extractelement <4 x float> %475, i64 3, !dbg !45
  %566 = tail call float @llvm.amdgcn.exp2.f32(float %565), !dbg !45
  %567 = insertelement <4 x float> %564, float %566, i64 3, !dbg !46
  %math.result380 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %567) #8, !dbg !46, !frisk.register_lifetimes !5
  %568 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !46
  %math.packed381 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %568) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce104 = fadd float %566, %local.reduce100, !dbg !47, !frisk.local_reduction !5
  %569 = extractelement <4 x float> %476, i64 0, !dbg !45
  %570 = tail call float @llvm.amdgcn.exp2.f32(float %569), !dbg !45
  %571 = insertelement <4 x float> poison, float %570, i64 0, !dbg !46
  %local.reduce105 = fadd float %570, %local.reduce101, !dbg !47, !frisk.local_reduction !5
  %572 = extractelement <4 x float> %476, i64 1, !dbg !45
  %573 = tail call float @llvm.amdgcn.exp2.f32(float %572), !dbg !45
  %574 = insertelement <4 x float> %571, float %573, i64 1, !dbg !46
  %local.reduce106 = fadd float %573, %local.reduce102, !dbg !47, !frisk.local_reduction !5
  %575 = extractelement <4 x float> %476, i64 2, !dbg !45
  %576 = tail call float @llvm.amdgcn.exp2.f32(float %575), !dbg !45
  %577 = insertelement <4 x float> %574, float %576, i64 2, !dbg !46
  %local.reduce107 = fadd float %576, %local.reduce103, !dbg !47, !frisk.local_reduction !5
  %578 = extractelement <4 x float> %476, i64 3, !dbg !45
  %579 = tail call float @llvm.amdgcn.exp2.f32(float %578), !dbg !45
  %580 = insertelement <4 x float> %577, float %579, i64 3, !dbg !46
  %math.result382 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %580) #8, !dbg !46, !frisk.register_lifetimes !5
  %581 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !46
  %math.packed383 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %581) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce108 = fadd float %579, %local.reduce104, !dbg !47, !frisk.local_reduction !5
  %582 = extractelement <4 x float> %477, i64 0, !dbg !45
  %583 = tail call float @llvm.amdgcn.exp2.f32(float %582), !dbg !45
  %584 = insertelement <4 x float> poison, float %583, i64 0, !dbg !46
  %local.reduce109 = fadd float %583, %local.reduce105, !dbg !47, !frisk.local_reduction !5
  %585 = extractelement <4 x float> %477, i64 1, !dbg !45
  %586 = tail call float @llvm.amdgcn.exp2.f32(float %585), !dbg !45
  %587 = insertelement <4 x float> %584, float %586, i64 1, !dbg !46
  %local.reduce110 = fadd float %586, %local.reduce106, !dbg !47, !frisk.local_reduction !5
  %588 = extractelement <4 x float> %477, i64 2, !dbg !45
  %589 = tail call float @llvm.amdgcn.exp2.f32(float %588), !dbg !45
  %590 = insertelement <4 x float> %587, float %589, i64 2, !dbg !46
  %local.reduce111 = fadd float %589, %local.reduce107, !dbg !47, !frisk.local_reduction !5
  %591 = extractelement <4 x float> %477, i64 3, !dbg !45
  %592 = tail call float @llvm.amdgcn.exp2.f32(float %591), !dbg !45
  %593 = insertelement <4 x float> %590, float %592, i64 3, !dbg !46
  %math.result384 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %593) #8, !dbg !46, !frisk.register_lifetimes !5
  %594 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !46
  %math.packed385 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %594) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce112 = fadd float %592, %local.reduce108, !dbg !47, !frisk.local_reduction !5
  %595 = extractelement <4 x float> %478, i64 0, !dbg !45
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %595), !dbg !45
  %597 = insertelement <4 x float> poison, float %596, i64 0, !dbg !46
  %local.reduce113 = fadd float %596, %local.reduce109, !dbg !47, !frisk.local_reduction !5
  %598 = extractelement <4 x float> %478, i64 1, !dbg !45
  %599 = tail call float @llvm.amdgcn.exp2.f32(float %598), !dbg !45
  %600 = insertelement <4 x float> %597, float %599, i64 1, !dbg !46
  %local.reduce114 = fadd float %599, %local.reduce110, !dbg !47, !frisk.local_reduction !5
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !47, !frisk.local_reduction !5
  %601 = extractelement <4 x float> %478, i64 2, !dbg !45
  %602 = tail call float @llvm.amdgcn.exp2.f32(float %601), !dbg !45
  %603 = insertelement <4 x float> %600, float %602, i64 2, !dbg !46
  %local.reduce115 = fadd float %602, %local.reduce111, !dbg !47, !frisk.local_reduction !5
  %604 = extractelement <4 x float> %478, i64 3, !dbg !45
  %605 = tail call float @llvm.amdgcn.exp2.f32(float %604), !dbg !45
  %606 = insertelement <4 x float> %603, float %605, i64 3, !dbg !46
  %math.result386 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %606) #8, !dbg !46, !frisk.register_lifetimes !5
  %607 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !46
  %math.packed387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %607) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce116 = fadd float %605, %local.reduce112, !dbg !47, !frisk.local_reduction !5
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !47, !frisk.local_reduction !5
  %608 = fadd float %local.reduce117, %local.reduce118, !dbg !47, !frisk.local_reduction !5
  %609 = fadd float %495, %608, !dbg !47
  %610 = insertelement <1 x float> poison, float %609, i64 0, !dbg !11
  %611 = insertvalue [2 x <1 x float>] poison, <1 x float> %610, 0, !dbg !11
  %612 = extractelement <4 x float> %480, i64 0, !dbg !45
  %613 = tail call float @llvm.amdgcn.exp2.f32(float %612), !dbg !45
  %614 = insertelement <4 x float> poison, float %613, i64 0, !dbg !46
  %615 = extractelement <4 x float> %480, i64 1, !dbg !45
  %616 = tail call float @llvm.amdgcn.exp2.f32(float %615), !dbg !45
  %617 = insertelement <4 x float> %614, float %616, i64 1, !dbg !46
  %618 = extractelement <4 x float> %480, i64 2, !dbg !45
  %619 = tail call float @llvm.amdgcn.exp2.f32(float %618), !dbg !45
  %620 = insertelement <4 x float> %617, float %619, i64 2, !dbg !46
  %621 = extractelement <4 x float> %480, i64 3, !dbg !45
  %622 = tail call float @llvm.amdgcn.exp2.f32(float %621), !dbg !45
  %623 = insertelement <4 x float> %620, float %622, i64 3, !dbg !46
  %math.result388 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %623) #8, !dbg !46, !frisk.register_lifetimes !5
  %624 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !46
  %math.packed389 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %624) #8, !dbg !45, !frisk.register_lifetimes !5
  %625 = extractelement <4 x float> %481, i64 0, !dbg !45
  %626 = tail call float @llvm.amdgcn.exp2.f32(float %625), !dbg !45
  %627 = insertelement <4 x float> poison, float %626, i64 0, !dbg !46
  %local.reduce = fadd float %613, %626, !dbg !47, !frisk.local_reduction !5
  %628 = extractelement <4 x float> %481, i64 1, !dbg !45
  %629 = tail call float @llvm.amdgcn.exp2.f32(float %628), !dbg !45
  %630 = insertelement <4 x float> %627, float %629, i64 1, !dbg !46
  %local.reduce59 = fadd float %616, %629, !dbg !47, !frisk.local_reduction !5
  %631 = extractelement <4 x float> %481, i64 2, !dbg !45
  %632 = tail call float @llvm.amdgcn.exp2.f32(float %631), !dbg !45
  %633 = insertelement <4 x float> %630, float %632, i64 2, !dbg !46
  %local.reduce60 = fadd float %619, %632, !dbg !47, !frisk.local_reduction !5
  %634 = extractelement <4 x float> %481, i64 3, !dbg !45
  %635 = tail call float @llvm.amdgcn.exp2.f32(float %634), !dbg !45
  %636 = insertelement <4 x float> %633, float %635, i64 3, !dbg !46
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %636) #8, !dbg !46, !frisk.register_lifetimes !5
  %637 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !46
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %637) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce61 = fadd float %622, %635, !dbg !47, !frisk.local_reduction !5
  %638 = extractelement <4 x float> %482, i64 0, !dbg !45
  %639 = tail call float @llvm.amdgcn.exp2.f32(float %638), !dbg !45
  %640 = insertelement <4 x float> poison, float %639, i64 0, !dbg !46
  %local.reduce62 = fadd float %639, %local.reduce, !dbg !47, !frisk.local_reduction !5
  %641 = extractelement <4 x float> %482, i64 1, !dbg !45
  %642 = tail call float @llvm.amdgcn.exp2.f32(float %641), !dbg !45
  %643 = insertelement <4 x float> %640, float %642, i64 1, !dbg !46
  %local.reduce63 = fadd float %642, %local.reduce59, !dbg !47, !frisk.local_reduction !5
  %644 = extractelement <4 x float> %482, i64 2, !dbg !45
  %645 = tail call float @llvm.amdgcn.exp2.f32(float %644), !dbg !45
  %646 = insertelement <4 x float> %643, float %645, i64 2, !dbg !46
  %local.reduce64 = fadd float %645, %local.reduce60, !dbg !47, !frisk.local_reduction !5
  %647 = extractelement <4 x float> %482, i64 3, !dbg !45
  %648 = tail call float @llvm.amdgcn.exp2.f32(float %647), !dbg !45
  %649 = insertelement <4 x float> %646, float %648, i64 3, !dbg !46
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %649) #8, !dbg !46, !frisk.register_lifetimes !5
  %650 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !46
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %650) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce65 = fadd float %648, %local.reduce61, !dbg !47, !frisk.local_reduction !5
  %651 = extractelement <4 x float> %483, i64 0, !dbg !45
  %652 = tail call float @llvm.amdgcn.exp2.f32(float %651), !dbg !45
  %653 = insertelement <4 x float> poison, float %652, i64 0, !dbg !46
  %local.reduce66 = fadd float %652, %local.reduce62, !dbg !47, !frisk.local_reduction !5
  %654 = extractelement <4 x float> %483, i64 1, !dbg !45
  %655 = tail call float @llvm.amdgcn.exp2.f32(float %654), !dbg !45
  %656 = insertelement <4 x float> %653, float %655, i64 1, !dbg !46
  %local.reduce67 = fadd float %655, %local.reduce63, !dbg !47, !frisk.local_reduction !5
  %657 = extractelement <4 x float> %483, i64 2, !dbg !45
  %658 = tail call float @llvm.amdgcn.exp2.f32(float %657), !dbg !45
  %659 = insertelement <4 x float> %656, float %658, i64 2, !dbg !46
  %local.reduce68 = fadd float %658, %local.reduce64, !dbg !47, !frisk.local_reduction !5
  %660 = extractelement <4 x float> %483, i64 3, !dbg !45
  %661 = tail call float @llvm.amdgcn.exp2.f32(float %660), !dbg !45
  %662 = insertelement <4 x float> %659, float %661, i64 3, !dbg !46
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %662) #8, !dbg !46, !frisk.register_lifetimes !5
  %663 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !46
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %663) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce69 = fadd float %661, %local.reduce65, !dbg !47, !frisk.local_reduction !5
  %664 = extractelement <4 x float> %484, i64 0, !dbg !45
  %665 = tail call float @llvm.amdgcn.exp2.f32(float %664), !dbg !45
  %666 = insertelement <4 x float> poison, float %665, i64 0, !dbg !46
  %local.reduce70 = fadd float %665, %local.reduce66, !dbg !47, !frisk.local_reduction !5
  %667 = extractelement <4 x float> %484, i64 1, !dbg !45
  %668 = tail call float @llvm.amdgcn.exp2.f32(float %667), !dbg !45
  %669 = insertelement <4 x float> %666, float %668, i64 1, !dbg !46
  %local.reduce71 = fadd float %668, %local.reduce67, !dbg !47, !frisk.local_reduction !5
  %670 = extractelement <4 x float> %484, i64 2, !dbg !45
  %671 = tail call float @llvm.amdgcn.exp2.f32(float %670), !dbg !45
  %672 = insertelement <4 x float> %669, float %671, i64 2, !dbg !46
  %local.reduce72 = fadd float %671, %local.reduce68, !dbg !47, !frisk.local_reduction !5
  %673 = extractelement <4 x float> %484, i64 3, !dbg !45
  %674 = tail call float @llvm.amdgcn.exp2.f32(float %673), !dbg !45
  %675 = insertelement <4 x float> %672, float %674, i64 3, !dbg !46
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %675) #8, !dbg !46, !frisk.register_lifetimes !5
  %676 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !46
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %676) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce73 = fadd float %674, %local.reduce69, !dbg !47, !frisk.local_reduction !5
  %677 = extractelement <4 x float> %485, i64 0, !dbg !45
  %678 = tail call float @llvm.amdgcn.exp2.f32(float %677), !dbg !45
  %679 = insertelement <4 x float> poison, float %678, i64 0, !dbg !46
  %local.reduce74 = fadd float %678, %local.reduce70, !dbg !47, !frisk.local_reduction !5
  %680 = extractelement <4 x float> %485, i64 1, !dbg !45
  %681 = tail call float @llvm.amdgcn.exp2.f32(float %680), !dbg !45
  %682 = insertelement <4 x float> %679, float %681, i64 1, !dbg !46
  %local.reduce75 = fadd float %681, %local.reduce71, !dbg !47, !frisk.local_reduction !5
  %683 = extractelement <4 x float> %485, i64 2, !dbg !45
  %684 = tail call float @llvm.amdgcn.exp2.f32(float %683), !dbg !45
  %685 = insertelement <4 x float> %682, float %684, i64 2, !dbg !46
  %local.reduce76 = fadd float %684, %local.reduce72, !dbg !47, !frisk.local_reduction !5
  %686 = extractelement <4 x float> %485, i64 3, !dbg !45
  %687 = tail call float @llvm.amdgcn.exp2.f32(float %686), !dbg !45
  %688 = insertelement <4 x float> %685, float %687, i64 3, !dbg !46
  %math.result398 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %688) #8, !dbg !46, !frisk.register_lifetimes !5
  %689 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !46
  %math.packed399 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %689) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce77 = fadd float %687, %local.reduce73, !dbg !47, !frisk.local_reduction !5
  %690 = extractelement <4 x float> %486, i64 0, !dbg !45
  %691 = tail call float @llvm.amdgcn.exp2.f32(float %690), !dbg !45
  %692 = insertelement <4 x float> poison, float %691, i64 0, !dbg !46
  %local.reduce78 = fadd float %691, %local.reduce74, !dbg !47, !frisk.local_reduction !5
  %693 = extractelement <4 x float> %486, i64 1, !dbg !45
  %694 = tail call float @llvm.amdgcn.exp2.f32(float %693), !dbg !45
  %695 = insertelement <4 x float> %692, float %694, i64 1, !dbg !46
  %local.reduce79 = fadd float %694, %local.reduce75, !dbg !47, !frisk.local_reduction !5
  %696 = extractelement <4 x float> %486, i64 2, !dbg !45
  %697 = tail call float @llvm.amdgcn.exp2.f32(float %696), !dbg !45
  %698 = insertelement <4 x float> %695, float %697, i64 2, !dbg !46
  %local.reduce80 = fadd float %697, %local.reduce76, !dbg !47, !frisk.local_reduction !5
  %699 = extractelement <4 x float> %486, i64 3, !dbg !45
  %700 = tail call float @llvm.amdgcn.exp2.f32(float %699), !dbg !45
  %701 = insertelement <4 x float> %698, float %700, i64 3, !dbg !46
  %math.result400 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %701) #8, !dbg !46, !frisk.register_lifetimes !5
  %702 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !46
  %math.packed401 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %702) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce81 = fadd float %700, %local.reduce77, !dbg !47, !frisk.local_reduction !5
  %703 = extractelement <4 x float> %487, i64 0, !dbg !45
  %704 = tail call float @llvm.amdgcn.exp2.f32(float %703), !dbg !45
  %705 = insertelement <4 x float> poison, float %704, i64 0, !dbg !46
  %local.reduce82 = fadd float %704, %local.reduce78, !dbg !47, !frisk.local_reduction !5
  %706 = extractelement <4 x float> %487, i64 1, !dbg !45
  %707 = tail call float @llvm.amdgcn.exp2.f32(float %706), !dbg !45
  %708 = insertelement <4 x float> %705, float %707, i64 1, !dbg !46
  %local.reduce83 = fadd float %707, %local.reduce79, !dbg !47, !frisk.local_reduction !5
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !47, !frisk.local_reduction !5
  %709 = extractelement <4 x float> %487, i64 2, !dbg !45
  %710 = tail call float @llvm.amdgcn.exp2.f32(float %709), !dbg !45
  %711 = insertelement <4 x float> %708, float %710, i64 2, !dbg !46
  %local.reduce84 = fadd float %710, %local.reduce80, !dbg !47, !frisk.local_reduction !5
  %712 = extractelement <4 x float> %487, i64 3, !dbg !45
  %713 = tail call float @llvm.amdgcn.exp2.f32(float %712), !dbg !45
  %714 = insertelement <4 x float> %711, float %713, i64 3, !dbg !46
  %math.result402 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %714) #8, !dbg !46, !frisk.register_lifetimes !5
  %715 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !46
  %math.packed403 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %715) #8, !dbg !47, !frisk.register_lifetimes !5
  %local.reduce85 = fadd float %713, %local.reduce81, !dbg !47, !frisk.local_reduction !5
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !47, !frisk.local_reduction !5
  %716 = fadd float %local.reduce86, %local.reduce87, !dbg !47, !frisk.local_reduction !5
  %717 = fadd float %503, %716, !dbg !47
  %718 = insertelement <1 x float> poison, float %717, i64 0, !dbg !11
  %719 = insertvalue [2 x <1 x float>] %611, <1 x float> %718, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1436 = or disjoint i64 %address.local921, %116, !dbg !37
  %vdb.address1437 = trunc nuw nsw i64 %vdb.address1436 to i32, !dbg !37
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !37
  %vdb.address1445 = or disjoint i64 %address.local927, %116, !dbg !37
  %vdb.address1446 = trunc nuw nsw i64 %vdb.address1445 to i32, !dbg !37
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !37
  %vdb.dma.base = add i32 %dma.buffer.lds1374, 8192, !dbg !48
  %720 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %vdb.dma.base1452 = add i32 %dma.buffer.lds1371, 8192, !dbg !48
  %721 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !48
  %address.local930 = and i32 %address.local929, 120, !dbg !48
  %address.local932 = and i32 %address.local575, 1536, !dbg !48
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !48
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !48
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !48
  %722 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %723 = bitcast <4 x i32> %722 to <8 x half>, !dbg !48
  %address.local936 = and i64 %address.local499, 60, !dbg !48
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !48
  %address.local939.idx = shl nuw nsw i64 %address.local937, 4, !dbg !48
  %address.local939 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local939.idx, !dbg !48
  %address.local940 = lshr i64 %address.local555, 4, !dbg !48
  %address.local942.idx = mul nuw nsw i64 %address.local940, 768, !dbg !48
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local942.idx, !dbg !48
  %address.local944.idx = mul nsw i64 %address.local542, -64, !dbg !48
  %address.local944 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local944.idx, !dbg !48
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !48
  %724 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %725 = bitcast <4 x i32> %724 to <8 x half>, !dbg !48
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !48
  %726 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %727 = bitcast <4 x i32> %726 to <8 x half>, !dbg !48
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !48
  %728 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %729 = bitcast <4 x i32> %728 to <8 x half>, !dbg !48
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !48
  %730 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %731 = bitcast <4 x i32> %730 to <8 x half>, !dbg !48
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !48
  %732 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %733 = bitcast <4 x i32> %732 to <8 x half>, !dbg !48
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !48
  %734 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %735 = bitcast <4 x i32> %734 to <8 x half>, !dbg !48
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !48
  %736 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %737 = shufflevector <8 x half> %723, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %738 = extractvalue [2 x <8 x float>] %112, 0, !dbg !43
  %739 = shufflevector <8 x float> %738, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %740 = fmul <4 x float> %739, %492, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %741 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %737, <4 x float> %740) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %742 = shufflevector <8 x half> %723, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %743 = shufflevector <8 x float> %738, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %744 = fmul <4 x float> %743, %492, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %745 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %742, <4 x float> %744) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %746 = extractvalue [2 x <8 x float>] %112, 1, !dbg !43
  %747 = shufflevector <8 x float> %746, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %748 = fmul <4 x float> %747, %500, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %749 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %737, <4 x float> %748) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %750 = shufflevector <8 x float> %746, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %751 = fmul <4 x float> %750, %500, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %752 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %742, <4 x float> %751) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %753 = shufflevector <8 x half> %725, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %754 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %753, <4 x float> %741) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %755 = shufflevector <8 x half> %725, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %756 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %755, <4 x float> %745) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %757 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %753, <4 x float> %749) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %758 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %755, <4 x float> %752) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %759 = shufflevector <8 x half> %727, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %760 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %759, <4 x float> %754) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %761 = shufflevector <8 x half> %727, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %762 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %761, <4 x float> %756) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %763 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %759, <4 x float> %757) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %764 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %761, <4 x float> %758) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %765 = shufflevector <8 x half> %729, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %766 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %765, <4 x float> %760) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %767 = shufflevector <8 x half> %729, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %768 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %767, <4 x float> %762) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %769 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %765, <4 x float> %763) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %770 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %767, <4 x float> %764) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %771 = shufflevector <8 x half> %731, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %772 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %771, <4 x float> %766) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %773 = shufflevector <8 x half> %731, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %774 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %773, <4 x float> %768) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %775 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %771, <4 x float> %769) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %776 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %773, <4 x float> %770) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %777 = shufflevector <8 x half> %733, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %778 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %777, <4 x float> %772) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %779 = shufflevector <8 x half> %733, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %780 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %779, <4 x float> %774) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %781 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %777, <4 x float> %775) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %782 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %779, <4 x float> %776) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %783 = shufflevector <8 x half> %735, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %784 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %783, <4 x float> %778) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %785 = shufflevector <8 x half> %735, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %786 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %785, <4 x float> %780) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %787 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %783, <4 x float> %781) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %788 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %785, <4 x float> %782) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %789 = bitcast <4 x i32> %736 to <8 x half>, !dbg !48
  %790 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %791 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %790, <4 x float> %784) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %792 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %793 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %792, <4 x float> %786) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %794 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %790, <4 x float> %787) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %795 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %792, <4 x float> %788) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1462 = or disjoint i64 %address.local921, %212, !dbg !37
  %vdb.address1463 = trunc nuw nsw i64 %vdb.address1462 to i32, !dbg !37
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !37
  %vdb.address1471 = or disjoint i64 %address.local927, %212, !dbg !37
  %vdb.address1472 = trunc nuw nsw i64 %vdb.address1471 to i32, !dbg !37
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !37
  %796 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %797 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %vdb.address1473, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !48
  %798 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %799 = bitcast <4 x i32> %798 to <8 x half>, !dbg !48
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8448, !dbg !48
  %800 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %801 = bitcast <4 x i32> %800 to <8 x half>, !dbg !48
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8704, !dbg !48
  %802 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %803 = bitcast <4 x i32> %802 to <8 x half>, !dbg !48
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8960, !dbg !48
  %804 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %805 = bitcast <4 x i32> %804 to <8 x half>, !dbg !48
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local935, i64 12288, !dbg !48
  %806 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %807 = bitcast <4 x i32> %806 to <8 x half>, !dbg !48
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12544, !dbg !48
  %808 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %809 = bitcast <4 x i32> %808 to <8 x half>, !dbg !48
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12800, !dbg !48
  %810 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %811 = bitcast <4 x i32> %810 to <8 x half>, !dbg !48
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local944, i64 13056, !dbg !48
  %812 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %813 = shufflevector <8 x half> %799, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %814 = extractvalue [2 x <8 x float>] %109, 0, !dbg !43
  %815 = shufflevector <8 x float> %814, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %816 = fmul <4 x float> %815, %492, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %817 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %813, <4 x float> %816) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %818 = shufflevector <8 x half> %799, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %819 = shufflevector <8 x float> %814, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %820 = fmul <4 x float> %819, %492, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %821 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %818, <4 x float> %820) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %822 = extractvalue [2 x <8 x float>] %109, 1, !dbg !43
  %823 = shufflevector <8 x float> %822, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %824 = fmul <4 x float> %823, %500, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %825 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %813, <4 x float> %824) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %826 = shufflevector <8 x float> %822, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %827 = fmul <4 x float> %826, %500, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %828 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %818, <4 x float> %827) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %829 = shufflevector <8 x half> %801, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %829, <4 x float> %817) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %831 = shufflevector <8 x half> %801, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %831, <4 x float> %821) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %833 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %829, <4 x float> %825) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %831, <4 x float> %828) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %835 = shufflevector <8 x half> %803, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %835, <4 x float> %830) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %837 = shufflevector <8 x half> %803, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %837, <4 x float> %832) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %839 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %835, <4 x float> %833) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %837, <4 x float> %834) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %841 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %841, <4 x float> %836) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %843 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %843, <4 x float> %838) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %841, <4 x float> %839) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %843, <4 x float> %840) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %847 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %847, <4 x float> %842) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %849 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %849, <4 x float> %844) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %847, <4 x float> %845) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %849, <4 x float> %846) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %853 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %853, <4 x float> %848) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %855 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %855, <4 x float> %850) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %857 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %853, <4 x float> %851) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %858 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %855, <4 x float> %852) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %859 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %860 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %859, <4 x float> %854) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %861 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %862 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %861, <4 x float> %856) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %859, <4 x float> %857) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %864 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %861, <4 x float> %858) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %865 = bitcast <4 x i32> %812 to <8 x half>, !dbg !48
  %866 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %866, <4 x float> %860) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %868 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %868, <4 x float> %862) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %870 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %866, <4 x float> %863) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %868, <4 x float> %864) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1480 = or disjoint i64 %address.local921, %265, !dbg !37
  %vdb.address1481 = trunc nuw nsw i64 %vdb.address1480 to i32, !dbg !37
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !37
  %vdb.address1489 = or disjoint i64 %address.local927, %265, !dbg !37
  %vdb.address1490 = trunc nuw nsw i64 %vdb.address1489 to i32, !dbg !37
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !37
  %872 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1482, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %873 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1491, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %874 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %875 = bitcast <4 x i32> %874 to <8 x half>, !dbg !48
  %876 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %877 = bitcast <4 x i32> %876 to <8 x half>, !dbg !48
  %878 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %879 = bitcast <4 x i32> %878 to <8 x half>, !dbg !48
  %880 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %881 = bitcast <4 x i32> %880 to <8 x half>, !dbg !48
  %882 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %883 = bitcast <4 x i32> %882 to <8 x half>, !dbg !48
  %884 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !48
  %886 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !48
  %888 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %889 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %890 = extractvalue [2 x <8 x float>] %108, 0, !dbg !43
  %891 = shufflevector <8 x float> %890, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %892 = fmul <4 x float> %891, %492, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %889, <4 x float> %892) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %894 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %895 = shufflevector <8 x float> %890, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %896 = fmul <4 x float> %895, %492, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %897 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %894, <4 x float> %896) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %898 = extractvalue [2 x <8 x float>] %108, 1, !dbg !43
  %899 = shufflevector <8 x float> %898, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %900 = fmul <4 x float> %899, %500, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %901 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %889, <4 x float> %900) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %902 = shufflevector <8 x float> %898, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %903 = fmul <4 x float> %902, %500, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %904 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %894, <4 x float> %903) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %905 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %905, <4 x float> %893) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %907 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %908 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %907, <4 x float> %897) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %909 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %905, <4 x float> %901) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %910 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %907, <4 x float> %904) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %911 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %911, <4 x float> %906) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %913 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %913, <4 x float> %908) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %915 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %911, <4 x float> %909) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %913, <4 x float> %910) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %917 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %917, <4 x float> %912) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %919 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %919, <4 x float> %914) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %921 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %917, <4 x float> %915) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %919, <4 x float> %916) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %923 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %923, <4 x float> %918) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %925 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %925, <4 x float> %920) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %927 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %923, <4 x float> %921) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %925, <4 x float> %922) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %929 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %929, <4 x float> %924) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %931 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %932 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %931, <4 x float> %926) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %933 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %929, <4 x float> %927) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %934 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %931, <4 x float> %928) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %935 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %935, <4 x float> %930) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %937 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %938 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %937, <4 x float> %932) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %939 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %935, <4 x float> %933) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %940 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %937, <4 x float> %934) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %941 = bitcast <4 x i32> %888 to <8 x half>, !dbg !48
  %942 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %943 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %942, <4 x float> %936) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %944 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %945 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %944, <4 x float> %938) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %946 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %942, <4 x float> %939) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %947 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %944, <4 x float> %940) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %948 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %949 = bitcast <4 x i32> %948 to <8 x half>, !dbg !48
  %950 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %951 = bitcast <4 x i32> %950 to <8 x half>, !dbg !48
  %952 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %953 = bitcast <4 x i32> %952 to <8 x half>, !dbg !48
  %954 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %955 = bitcast <4 x i32> %954 to <8 x half>, !dbg !48
  %956 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %957 = bitcast <4 x i32> %956 to <8 x half>, !dbg !48
  %958 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %959 = bitcast <4 x i32> %958 to <8 x half>, !dbg !48
  %960 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %961 = bitcast <4 x i32> %960 to <8 x half>, !dbg !48
  %962 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #11, !dbg !48, !frisk.v_double_buffer !49
  %963 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  %964 = extractvalue [2 x <8 x float>] %107, 0, !dbg !43
  %965 = shufflevector <8 x float> %964, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %966 = fmul <4 x float> %965, %492, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %963, <4 x float> %966) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %968 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  %969 = shufflevector <8 x float> %964, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %970 = fmul <4 x float> %969, %492, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %971 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %968, <4 x float> %970) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %972 = extractvalue [2 x <8 x float>] %107, 1, !dbg !43
  %973 = shufflevector <8 x float> %972, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %974 = fmul <4 x float> %973, %500, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %963, <4 x float> %974) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %976 = shufflevector <8 x float> %972, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %977 = fmul <4 x float> %976, %500, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %978 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %968, <4 x float> %977) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %979 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %979, <4 x float> %967) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %981 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %981, <4 x float> %971) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %983 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %979, <4 x float> %975) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %981, <4 x float> %978) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %985 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %985, <4 x float> %980) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %987 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %987, <4 x float> %982) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %989 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %985, <4 x float> %983) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %987, <4 x float> %984) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %991 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %991, <4 x float> %986) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %993 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %993, <4 x float> %988) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %995 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %991, <4 x float> %989) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %993, <4 x float> %990) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %997 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %997, <4 x float> %992) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %999 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %999, <4 x float> %994) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1001 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %997, <4 x float> %995) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %999, <4 x float> %996) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1003 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1003, <4 x float> %998) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1005 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1005, <4 x float> %1000) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1007 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1003, <4 x float> %1001) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1005, <4 x float> %1002) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1009 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1009, <4 x float> %1004) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1011 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1011, <4 x float> %1006) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1013 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1009, <4 x float> %1007) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1011, <4 x float> %1008) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1015 = bitcast <4 x i32> %962 to <8 x half>, !dbg !48
  %1016 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1017 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1016, <4 x float> %1010) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1018 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1019 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1018, <4 x float> %1012) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1016, <4 x float> %1013) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !48
  %1021 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1018, <4 x float> %1014) #8, !dbg !48, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !11
  %1022 = shufflevector <4 x float> %791, <4 x float> %793, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1023 = insertvalue [2 x <8 x float>] poison, <8 x float> %1022, 0, !dbg !11
  %1024 = shufflevector <4 x float> %794, <4 x float> %795, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1025 = insertvalue [2 x <8 x float>] %1023, <8 x float> %1024, 1, !dbg !11
  %1026 = insertvalue [2 x <1 x float>] poison, <1 x float> %456, 0, !dbg !11
  %1027 = insertvalue [2 x <1 x float>] %1026, <1 x float> %469, 1, !dbg !11
  %1028 = shufflevector <4 x float> %867, <4 x float> %869, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1029 = insertvalue [2 x <8 x float>] poison, <8 x float> %1028, 0, !dbg !11
  %1030 = shufflevector <4 x float> %870, <4 x float> %871, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1031 = insertvalue [2 x <8 x float>] %1029, <8 x float> %1030, 1, !dbg !11
  %1032 = shufflevector <4 x float> %943, <4 x float> %945, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1033 = insertvalue [2 x <8 x float>] poison, <8 x float> %1032, 0, !dbg !11
  %1034 = shufflevector <4 x float> %946, <4 x float> %947, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1035 = insertvalue [2 x <8 x float>] %1033, <8 x float> %1034, 1, !dbg !11
  %1036 = shufflevector <4 x float> %1017, <4 x float> %1019, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1037 = insertvalue [2 x <8 x float>] poison, <8 x float> %1036, 0, !dbg !11
  %1038 = shufflevector <4 x float> %1020, <4 x float> %1021, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1039 = insertvalue [2 x <8 x float>] %1037, <8 x float> %1038, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1040 = add nuw nsw i64 %113, 1, !dbg !11
  %1041 = icmp samesign ult i64 %113, 31, !dbg !11
  br i1 %1041, label %106, label %1042, !dbg !11

1042:                                             ; preds = %106
  %1043 = shl nuw nsw i32 %435, 2, !dbg !11
  %1044 = xor i32 %1043, 64, !dbg !11
  %1045 = bitcast float %717 to i32, !dbg !11
  %lane.sum.shuffle188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1044, i32 %1045), !dbg !11
  %1046 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !11
  %lane.sum.final189 = fadd float %717, %1046, !dbg !11
  %1047 = xor i32 %1043, 128, !dbg !11
  %1048 = bitcast float %lane.sum.final189 to i32, !dbg !11
  %lane.sum.shuffle190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1047, i32 %1048), !dbg !11
  %1049 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !11
  %lane.sum.final191 = fadd float %lane.sum.final189, %1049, !dbg !11
  %1050 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !11
  %1051 = bitcast float %609 to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1044, i32 %1051), !dbg !11
  %1052 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %609, %1052, !dbg !11
  %1053 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1047, i32 %1053), !dbg !11
  %1054 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !11
  %lane.sum.final183 = fadd float %lane.sum.final, %1054, !dbg !11
  %1055 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !11
  %1056 = and i32 %kdma.tid, 16, !dbg !11
  %.not = icmp eq i32 %1056, 0, !dbg !11
  %1057 = and i32 %kdma.tid, 32, !dbg !11
  %.not36 = icmp eq i32 %1057, 0, !dbg !11
  %1058 = extractelement <4 x float> %1017, i64 0, !dbg !11
  %1059 = extractelement <4 x float> %1017, i64 1, !dbg !11
  %1060 = extractelement <4 x float> %1017, i64 2, !dbg !11
  %1061 = extractelement <4 x float> %1017, i64 3, !dbg !11
  %1062 = select i1 %.not, float %1059, float %1058, !dbg !11
  %1063 = bitcast float %1062 to i32, !dbg !11
  %1064 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1063), !dbg !11
  %1065 = bitcast i32 %1064 to float, !dbg !11
  %1066 = select i1 %.not, float %1058, float %1065, !dbg !11
  %1067 = select i1 %.not, float %1065, float %1059, !dbg !11
  %1068 = select i1 %.not, float %1061, float %1060, !dbg !11
  %1069 = bitcast float %1068 to i32, !dbg !11
  %1070 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1069), !dbg !11
  %1071 = bitcast i32 %1070 to float, !dbg !11
  %1072 = select i1 %.not, float %1060, float %1071, !dbg !11
  %1073 = select i1 %.not, float %1071, float %1061, !dbg !11
  %1074 = select i1 %.not36, float %1072, float %1066, !dbg !11
  %1075 = bitcast float %1074 to i32, !dbg !11
  %1076 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1075), !dbg !11
  %1077 = bitcast i32 %1076 to float, !dbg !11
  %1078 = select i1 %.not36, float %1073, float %1067, !dbg !11
  %1079 = bitcast float %1078 to i32, !dbg !11
  %1080 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1079), !dbg !11
  %1081 = bitcast i32 %1080 to float, !dbg !11
  %1082 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1083 = shufflevector <4 x i1> %1082, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1084 = insertelement <4 x float> poison, float %1066, i64 0, !dbg !11
  %1085 = insertelement <4 x float> %1084, float %1067, i64 1, !dbg !11
  %1086 = insertelement <4 x float> %1085, float %1077, i64 2, !dbg !11
  %1087 = insertelement <4 x float> %1086, float %1081, i64 3, !dbg !11
  %1088 = insertelement <4 x float> poison, float %1077, i64 0, !dbg !11
  %1089 = insertelement <4 x float> %1088, float %1081, i64 1, !dbg !11
  %1090 = insertelement <4 x float> %1089, float %1072, i64 2, !dbg !11
  %1091 = insertelement <4 x float> %1090, float %1073, i64 3, !dbg !11
  %1092 = select <4 x i1> %1083, <4 x float> %1087, <4 x float> %1091, !dbg !11
  %1093 = extractelement <4 x float> %943, i64 0, !dbg !11
  %1094 = extractelement <4 x float> %943, i64 1, !dbg !11
  %1095 = extractelement <4 x float> %943, i64 2, !dbg !11
  %1096 = extractelement <4 x float> %943, i64 3, !dbg !11
  %1097 = select i1 %.not, float %1094, float %1093, !dbg !11
  %1098 = bitcast float %1097 to i32, !dbg !11
  %1099 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1098), !dbg !11
  %1100 = bitcast i32 %1099 to float, !dbg !11
  %1101 = select i1 %.not, float %1093, float %1100, !dbg !11
  %1102 = select i1 %.not, float %1100, float %1094, !dbg !11
  %1103 = select i1 %.not, float %1096, float %1095, !dbg !11
  %1104 = bitcast float %1103 to i32, !dbg !11
  %1105 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1104), !dbg !11
  %1106 = bitcast i32 %1105 to float, !dbg !11
  %1107 = select i1 %.not, float %1095, float %1106, !dbg !11
  %1108 = select i1 %.not, float %1106, float %1096, !dbg !11
  %1109 = select i1 %.not36, float %1107, float %1101, !dbg !11
  %1110 = bitcast float %1109 to i32, !dbg !11
  %1111 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1110), !dbg !11
  %1112 = bitcast i32 %1111 to float, !dbg !11
  %1113 = select i1 %.not36, float %1108, float %1102, !dbg !11
  %1114 = bitcast float %1113 to i32, !dbg !11
  %1115 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1114), !dbg !11
  %1116 = bitcast i32 %1115 to float, !dbg !11
  %1117 = insertelement <4 x float> poison, float %1101, i64 0, !dbg !11
  %1118 = insertelement <4 x float> %1117, float %1102, i64 1, !dbg !11
  %1119 = insertelement <4 x float> %1118, float %1112, i64 2, !dbg !11
  %1120 = insertelement <4 x float> %1119, float %1116, i64 3, !dbg !11
  %1121 = insertelement <4 x float> poison, float %1112, i64 0, !dbg !11
  %1122 = insertelement <4 x float> %1121, float %1116, i64 1, !dbg !11
  %1123 = insertelement <4 x float> %1122, float %1107, i64 2, !dbg !11
  %1124 = insertelement <4 x float> %1123, float %1108, i64 3, !dbg !11
  %1125 = select <4 x i1> %1083, <4 x float> %1120, <4 x float> %1124, !dbg !11
  %1126 = extractelement <4 x float> %867, i64 0, !dbg !11
  %1127 = extractelement <4 x float> %867, i64 1, !dbg !11
  %1128 = extractelement <4 x float> %867, i64 2, !dbg !11
  %1129 = extractelement <4 x float> %867, i64 3, !dbg !11
  %1130 = select i1 %.not, float %1127, float %1126, !dbg !11
  %1131 = bitcast float %1130 to i32, !dbg !11
  %1132 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1131), !dbg !11
  %1133 = bitcast i32 %1132 to float, !dbg !11
  %1134 = select i1 %.not, float %1126, float %1133, !dbg !11
  %1135 = select i1 %.not, float %1133, float %1127, !dbg !11
  %1136 = select i1 %.not, float %1129, float %1128, !dbg !11
  %1137 = bitcast float %1136 to i32, !dbg !11
  %1138 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1137), !dbg !11
  %1139 = bitcast i32 %1138 to float, !dbg !11
  %1140 = select i1 %.not, float %1128, float %1139, !dbg !11
  %1141 = select i1 %.not, float %1139, float %1129, !dbg !11
  %1142 = select i1 %.not36, float %1140, float %1134, !dbg !11
  %1143 = bitcast float %1142 to i32, !dbg !11
  %1144 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1143), !dbg !11
  %1145 = bitcast i32 %1144 to float, !dbg !11
  %1146 = select i1 %.not36, float %1141, float %1135, !dbg !11
  %1147 = bitcast float %1146 to i32, !dbg !11
  %1148 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1147), !dbg !11
  %1149 = bitcast i32 %1148 to float, !dbg !11
  %1150 = insertelement <4 x float> poison, float %1134, i64 0, !dbg !11
  %1151 = insertelement <4 x float> %1150, float %1135, i64 1, !dbg !11
  %1152 = insertelement <4 x float> %1151, float %1145, i64 2, !dbg !11
  %1153 = insertelement <4 x float> %1152, float %1149, i64 3, !dbg !11
  %1154 = insertelement <4 x float> poison, float %1145, i64 0, !dbg !11
  %1155 = insertelement <4 x float> %1154, float %1149, i64 1, !dbg !11
  %1156 = insertelement <4 x float> %1155, float %1140, i64 2, !dbg !11
  %1157 = insertelement <4 x float> %1156, float %1141, i64 3, !dbg !11
  %1158 = select <4 x i1> %1083, <4 x float> %1153, <4 x float> %1157, !dbg !11
  %1159 = extractelement <4 x float> %791, i64 0, !dbg !11
  %1160 = extractelement <4 x float> %791, i64 1, !dbg !11
  %1161 = extractelement <4 x float> %791, i64 2, !dbg !11
  %1162 = extractelement <4 x float> %791, i64 3, !dbg !11
  %1163 = select i1 %.not, float %1160, float %1159, !dbg !11
  %1164 = bitcast float %1163 to i32, !dbg !11
  %1165 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1164), !dbg !11
  %1166 = bitcast i32 %1165 to float, !dbg !11
  %1167 = select i1 %.not, float %1159, float %1166, !dbg !11
  %1168 = select i1 %.not, float %1166, float %1160, !dbg !11
  %1169 = select i1 %.not, float %1162, float %1161, !dbg !11
  %1170 = bitcast float %1169 to i32, !dbg !11
  %1171 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1170), !dbg !11
  %1172 = bitcast i32 %1171 to float, !dbg !11
  %1173 = select i1 %.not, float %1161, float %1172, !dbg !11
  %1174 = select i1 %.not, float %1172, float %1162, !dbg !11
  %1175 = select i1 %.not36, float %1173, float %1167, !dbg !11
  %1176 = bitcast float %1175 to i32, !dbg !11
  %1177 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1176), !dbg !11
  %1178 = select i1 %.not36, float %1174, float %1168, !dbg !11
  %1179 = bitcast float %1178 to i32, !dbg !11
  %1180 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1179), !dbg !11
  %1181 = shufflevector <1 x float> %1055, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !50
  %1182 = extractelement <4 x float> %1019, i64 0, !dbg !11
  %1183 = extractelement <4 x float> %1019, i64 1, !dbg !11
  %1184 = extractelement <4 x float> %1019, i64 2, !dbg !11
  %1185 = extractelement <4 x float> %1019, i64 3, !dbg !11
  %1186 = select i1 %.not, float %1183, float %1182, !dbg !11
  %1187 = bitcast float %1186 to i32, !dbg !11
  %1188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1187), !dbg !11
  %1189 = bitcast i32 %1188 to float, !dbg !11
  %1190 = select i1 %.not, float %1182, float %1189, !dbg !11
  %1191 = select i1 %.not, float %1189, float %1183, !dbg !11
  %1192 = select i1 %.not, float %1185, float %1184, !dbg !11
  %1193 = bitcast float %1192 to i32, !dbg !11
  %1194 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1193), !dbg !11
  %1195 = bitcast i32 %1194 to float, !dbg !11
  %1196 = select i1 %.not, float %1184, float %1195, !dbg !11
  %1197 = select i1 %.not, float %1195, float %1185, !dbg !11
  %1198 = select i1 %.not36, float %1196, float %1190, !dbg !11
  %1199 = bitcast float %1198 to i32, !dbg !11
  %1200 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1199), !dbg !11
  %1201 = bitcast i32 %1200 to float, !dbg !11
  %1202 = select i1 %.not36, float %1197, float %1191, !dbg !11
  %1203 = bitcast float %1202 to i32, !dbg !11
  %1204 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1203), !dbg !11
  %1205 = bitcast i32 %1204 to float, !dbg !11
  %1206 = insertelement <4 x float> poison, float %1190, i64 0, !dbg !11
  %1207 = insertelement <4 x float> %1206, float %1191, i64 1, !dbg !11
  %1208 = insertelement <4 x float> %1207, float %1201, i64 2, !dbg !11
  %1209 = insertelement <4 x float> %1208, float %1205, i64 3, !dbg !11
  %1210 = insertelement <4 x float> poison, float %1201, i64 0, !dbg !11
  %1211 = insertelement <4 x float> %1210, float %1205, i64 1, !dbg !11
  %1212 = insertelement <4 x float> %1211, float %1196, i64 2, !dbg !11
  %1213 = insertelement <4 x float> %1212, float %1197, i64 3, !dbg !11
  %1214 = select <4 x i1> %1083, <4 x float> %1209, <4 x float> %1213, !dbg !11
  %1215 = extractelement <4 x float> %945, i64 0, !dbg !11
  %1216 = extractelement <4 x float> %945, i64 1, !dbg !11
  %1217 = extractelement <4 x float> %945, i64 2, !dbg !11
  %1218 = extractelement <4 x float> %945, i64 3, !dbg !11
  %1219 = select i1 %.not, float %1216, float %1215, !dbg !11
  %1220 = bitcast float %1219 to i32, !dbg !11
  %1221 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1220), !dbg !11
  %1222 = bitcast i32 %1221 to float, !dbg !11
  %1223 = select i1 %.not, float %1215, float %1222, !dbg !11
  %1224 = select i1 %.not, float %1222, float %1216, !dbg !11
  %1225 = select i1 %.not, float %1218, float %1217, !dbg !11
  %1226 = bitcast float %1225 to i32, !dbg !11
  %1227 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1226), !dbg !11
  %1228 = bitcast i32 %1227 to float, !dbg !11
  %1229 = select i1 %.not, float %1217, float %1228, !dbg !11
  %1230 = select i1 %.not, float %1228, float %1218, !dbg !11
  %1231 = select i1 %.not36, float %1229, float %1223, !dbg !11
  %1232 = bitcast float %1231 to i32, !dbg !11
  %1233 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1232), !dbg !11
  %1234 = bitcast i32 %1233 to float, !dbg !11
  %1235 = select i1 %.not36, float %1230, float %1224, !dbg !11
  %1236 = bitcast float %1235 to i32, !dbg !11
  %1237 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1236), !dbg !11
  %1238 = bitcast i32 %1237 to float, !dbg !11
  %1239 = insertelement <4 x float> poison, float %1223, i64 0, !dbg !11
  %1240 = insertelement <4 x float> %1239, float %1224, i64 1, !dbg !11
  %1241 = insertelement <4 x float> %1240, float %1234, i64 2, !dbg !11
  %1242 = insertelement <4 x float> %1241, float %1238, i64 3, !dbg !11
  %1243 = insertelement <4 x float> poison, float %1234, i64 0, !dbg !11
  %1244 = insertelement <4 x float> %1243, float %1238, i64 1, !dbg !11
  %1245 = insertelement <4 x float> %1244, float %1229, i64 2, !dbg !11
  %1246 = insertelement <4 x float> %1245, float %1230, i64 3, !dbg !11
  %1247 = select <4 x i1> %1083, <4 x float> %1242, <4 x float> %1246, !dbg !11
  %1248 = extractelement <4 x float> %869, i64 0, !dbg !11
  %1249 = extractelement <4 x float> %869, i64 1, !dbg !11
  %1250 = extractelement <4 x float> %869, i64 2, !dbg !11
  %1251 = extractelement <4 x float> %869, i64 3, !dbg !11
  %1252 = select i1 %.not, float %1249, float %1248, !dbg !11
  %1253 = bitcast float %1252 to i32, !dbg !11
  %1254 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1253), !dbg !11
  %1255 = bitcast i32 %1254 to float, !dbg !11
  %1256 = select i1 %.not, float %1248, float %1255, !dbg !11
  %1257 = select i1 %.not, float %1255, float %1249, !dbg !11
  %1258 = select i1 %.not, float %1251, float %1250, !dbg !11
  %1259 = bitcast float %1258 to i32, !dbg !11
  %1260 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1259), !dbg !11
  %1261 = bitcast i32 %1260 to float, !dbg !11
  %1262 = select i1 %.not, float %1250, float %1261, !dbg !11
  %1263 = select i1 %.not, float %1261, float %1251, !dbg !11
  %1264 = select i1 %.not36, float %1262, float %1256, !dbg !11
  %1265 = bitcast float %1264 to i32, !dbg !11
  %1266 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1265), !dbg !11
  %1267 = bitcast i32 %1266 to float, !dbg !11
  %1268 = select i1 %.not36, float %1263, float %1257, !dbg !11
  %1269 = bitcast float %1268 to i32, !dbg !11
  %1270 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1269), !dbg !11
  %1271 = bitcast i32 %1270 to float, !dbg !11
  %1272 = insertelement <4 x float> poison, float %1256, i64 0, !dbg !11
  %1273 = insertelement <4 x float> %1272, float %1257, i64 1, !dbg !11
  %1274 = insertelement <4 x float> %1273, float %1267, i64 2, !dbg !11
  %1275 = insertelement <4 x float> %1274, float %1271, i64 3, !dbg !11
  %1276 = insertelement <4 x float> poison, float %1267, i64 0, !dbg !11
  %1277 = insertelement <4 x float> %1276, float %1271, i64 1, !dbg !11
  %1278 = insertelement <4 x float> %1277, float %1262, i64 2, !dbg !11
  %1279 = insertelement <4 x float> %1278, float %1263, i64 3, !dbg !11
  %1280 = select <4 x i1> %1083, <4 x float> %1275, <4 x float> %1279, !dbg !11
  %1281 = extractelement <4 x float> %793, i64 0, !dbg !11
  %1282 = extractelement <4 x float> %793, i64 1, !dbg !11
  %1283 = extractelement <4 x float> %793, i64 2, !dbg !11
  %1284 = extractelement <4 x float> %793, i64 3, !dbg !11
  %1285 = select i1 %.not, float %1282, float %1281, !dbg !11
  %1286 = bitcast float %1285 to i32, !dbg !11
  %1287 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1286), !dbg !11
  %1288 = bitcast i32 %1287 to float, !dbg !11
  %1289 = select i1 %.not, float %1281, float %1288, !dbg !11
  %1290 = select i1 %.not, float %1288, float %1282, !dbg !11
  %1291 = select i1 %.not, float %1284, float %1283, !dbg !11
  %1292 = bitcast float %1291 to i32, !dbg !11
  %1293 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1292), !dbg !11
  %1294 = bitcast i32 %1293 to float, !dbg !11
  %1295 = select i1 %.not, float %1283, float %1294, !dbg !11
  %1296 = select i1 %.not, float %1294, float %1284, !dbg !11
  %1297 = select i1 %.not36, float %1295, float %1289, !dbg !11
  %1298 = bitcast float %1297 to i32, !dbg !11
  %1299 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1298), !dbg !11
  %1300 = select i1 %.not36, float %1296, float %1290, !dbg !11
  %1301 = bitcast float %1300 to i32, !dbg !11
  %1302 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1301), !dbg !11
  %1303 = extractelement <4 x float> %1020, i64 0, !dbg !11
  %1304 = extractelement <4 x float> %1020, i64 1, !dbg !11
  %1305 = extractelement <4 x float> %1020, i64 2, !dbg !11
  %1306 = extractelement <4 x float> %1020, i64 3, !dbg !11
  %1307 = select i1 %.not, float %1304, float %1303, !dbg !11
  %1308 = bitcast float %1307 to i32, !dbg !11
  %1309 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1308), !dbg !11
  %1310 = bitcast i32 %1309 to float, !dbg !11
  %1311 = select i1 %.not, float %1303, float %1310, !dbg !11
  %1312 = select i1 %.not, float %1310, float %1304, !dbg !11
  %1313 = select i1 %.not, float %1306, float %1305, !dbg !11
  %1314 = bitcast float %1313 to i32, !dbg !11
  %1315 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1314), !dbg !11
  %1316 = bitcast i32 %1315 to float, !dbg !11
  %1317 = select i1 %.not, float %1305, float %1316, !dbg !11
  %1318 = select i1 %.not, float %1316, float %1306, !dbg !11
  %1319 = select i1 %.not36, float %1317, float %1311, !dbg !11
  %1320 = bitcast float %1319 to i32, !dbg !11
  %1321 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1320), !dbg !11
  %1322 = bitcast i32 %1321 to float, !dbg !11
  %1323 = select i1 %.not36, float %1318, float %1312, !dbg !11
  %1324 = bitcast float %1323 to i32, !dbg !11
  %1325 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1324), !dbg !11
  %1326 = bitcast i32 %1325 to float, !dbg !11
  %1327 = insertelement <4 x float> poison, float %1311, i64 0, !dbg !11
  %1328 = insertelement <4 x float> %1327, float %1312, i64 1, !dbg !11
  %1329 = insertelement <4 x float> %1328, float %1322, i64 2, !dbg !11
  %1330 = insertelement <4 x float> %1329, float %1326, i64 3, !dbg !11
  %1331 = insertelement <4 x float> poison, float %1322, i64 0, !dbg !11
  %1332 = insertelement <4 x float> %1331, float %1326, i64 1, !dbg !11
  %1333 = insertelement <4 x float> %1332, float %1317, i64 2, !dbg !11
  %1334 = insertelement <4 x float> %1333, float %1318, i64 3, !dbg !11
  %1335 = select <4 x i1> %1083, <4 x float> %1330, <4 x float> %1334, !dbg !11
  %1336 = extractelement <4 x float> %946, i64 0, !dbg !11
  %1337 = extractelement <4 x float> %946, i64 1, !dbg !11
  %1338 = extractelement <4 x float> %946, i64 2, !dbg !11
  %1339 = extractelement <4 x float> %946, i64 3, !dbg !11
  %1340 = select i1 %.not, float %1337, float %1336, !dbg !11
  %1341 = bitcast float %1340 to i32, !dbg !11
  %1342 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1341), !dbg !11
  %1343 = bitcast i32 %1342 to float, !dbg !11
  %1344 = select i1 %.not, float %1336, float %1343, !dbg !11
  %1345 = select i1 %.not, float %1343, float %1337, !dbg !11
  %1346 = select i1 %.not, float %1339, float %1338, !dbg !11
  %1347 = bitcast float %1346 to i32, !dbg !11
  %1348 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1347), !dbg !11
  %1349 = bitcast i32 %1348 to float, !dbg !11
  %1350 = select i1 %.not, float %1338, float %1349, !dbg !11
  %1351 = select i1 %.not, float %1349, float %1339, !dbg !11
  %1352 = select i1 %.not36, float %1350, float %1344, !dbg !11
  %1353 = bitcast float %1352 to i32, !dbg !11
  %1354 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1353), !dbg !11
  %1355 = bitcast i32 %1354 to float, !dbg !11
  %1356 = select i1 %.not36, float %1351, float %1345, !dbg !11
  %1357 = bitcast float %1356 to i32, !dbg !11
  %1358 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1357), !dbg !11
  %1359 = bitcast i32 %1358 to float, !dbg !11
  %1360 = insertelement <4 x float> poison, float %1344, i64 0, !dbg !11
  %1361 = insertelement <4 x float> %1360, float %1345, i64 1, !dbg !11
  %1362 = insertelement <4 x float> %1361, float %1355, i64 2, !dbg !11
  %1363 = insertelement <4 x float> %1362, float %1359, i64 3, !dbg !11
  %1364 = insertelement <4 x float> poison, float %1355, i64 0, !dbg !11
  %1365 = insertelement <4 x float> %1364, float %1359, i64 1, !dbg !11
  %1366 = insertelement <4 x float> %1365, float %1350, i64 2, !dbg !11
  %1367 = insertelement <4 x float> %1366, float %1351, i64 3, !dbg !11
  %1368 = select <4 x i1> %1083, <4 x float> %1363, <4 x float> %1367, !dbg !11
  %1369 = extractelement <4 x float> %870, i64 0, !dbg !11
  %1370 = extractelement <4 x float> %870, i64 1, !dbg !11
  %1371 = extractelement <4 x float> %870, i64 2, !dbg !11
  %1372 = extractelement <4 x float> %870, i64 3, !dbg !11
  %1373 = select i1 %.not, float %1370, float %1369, !dbg !11
  %1374 = bitcast float %1373 to i32, !dbg !11
  %1375 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1374), !dbg !11
  %1376 = bitcast i32 %1375 to float, !dbg !11
  %1377 = select i1 %.not, float %1369, float %1376, !dbg !11
  %1378 = select i1 %.not, float %1376, float %1370, !dbg !11
  %1379 = select i1 %.not, float %1372, float %1371, !dbg !11
  %1380 = bitcast float %1379 to i32, !dbg !11
  %1381 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1380), !dbg !11
  %1382 = bitcast i32 %1381 to float, !dbg !11
  %1383 = select i1 %.not, float %1371, float %1382, !dbg !11
  %1384 = select i1 %.not, float %1382, float %1372, !dbg !11
  %1385 = select i1 %.not36, float %1383, float %1377, !dbg !11
  %1386 = bitcast float %1385 to i32, !dbg !11
  %1387 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1386), !dbg !11
  %1388 = bitcast i32 %1387 to float, !dbg !11
  %1389 = select i1 %.not36, float %1384, float %1378, !dbg !11
  %1390 = bitcast float %1389 to i32, !dbg !11
  %1391 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1390), !dbg !11
  %1392 = bitcast i32 %1391 to float, !dbg !11
  %1393 = insertelement <4 x float> poison, float %1377, i64 0, !dbg !11
  %1394 = insertelement <4 x float> %1393, float %1378, i64 1, !dbg !11
  %1395 = insertelement <4 x float> %1394, float %1388, i64 2, !dbg !11
  %1396 = insertelement <4 x float> %1395, float %1392, i64 3, !dbg !11
  %1397 = insertelement <4 x float> poison, float %1388, i64 0, !dbg !11
  %1398 = insertelement <4 x float> %1397, float %1392, i64 1, !dbg !11
  %1399 = insertelement <4 x float> %1398, float %1383, i64 2, !dbg !11
  %1400 = insertelement <4 x float> %1399, float %1384, i64 3, !dbg !11
  %1401 = select <4 x i1> %1083, <4 x float> %1396, <4 x float> %1400, !dbg !11
  %1402 = extractelement <4 x float> %794, i64 0, !dbg !11
  %1403 = extractelement <4 x float> %794, i64 1, !dbg !11
  %1404 = extractelement <4 x float> %794, i64 2, !dbg !11
  %1405 = extractelement <4 x float> %794, i64 3, !dbg !11
  %1406 = select i1 %.not, float %1403, float %1402, !dbg !11
  %1407 = bitcast float %1406 to i32, !dbg !11
  %1408 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1407), !dbg !11
  %1409 = bitcast i32 %1408 to float, !dbg !11
  %1410 = select i1 %.not, float %1402, float %1409, !dbg !11
  %1411 = select i1 %.not, float %1409, float %1403, !dbg !11
  %1412 = select i1 %.not, float %1405, float %1404, !dbg !11
  %1413 = bitcast float %1412 to i32, !dbg !11
  %1414 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1413), !dbg !11
  %1415 = bitcast i32 %1414 to float, !dbg !11
  %1416 = select i1 %.not, float %1404, float %1415, !dbg !11
  %1417 = select i1 %.not, float %1415, float %1405, !dbg !11
  %1418 = select i1 %.not36, float %1416, float %1410, !dbg !11
  %1419 = bitcast float %1418 to i32, !dbg !11
  %1420 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1419), !dbg !11
  %1421 = select i1 %.not36, float %1417, float %1411, !dbg !11
  %1422 = bitcast float %1421 to i32, !dbg !11
  %1423 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1422), !dbg !11
  %1424 = shufflevector <1 x float> %1050, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !50
  %1425 = extractelement <4 x float> %1021, i64 0, !dbg !11
  %1426 = extractelement <4 x float> %1021, i64 1, !dbg !11
  %1427 = extractelement <4 x float> %1021, i64 2, !dbg !11
  %1428 = extractelement <4 x float> %1021, i64 3, !dbg !11
  %1429 = select i1 %.not, float %1426, float %1425, !dbg !11
  %1430 = bitcast float %1429 to i32, !dbg !11
  %1431 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1430), !dbg !11
  %1432 = bitcast i32 %1431 to float, !dbg !11
  %1433 = select i1 %.not, float %1425, float %1432, !dbg !11
  %1434 = select i1 %.not, float %1432, float %1426, !dbg !11
  %1435 = select i1 %.not, float %1428, float %1427, !dbg !11
  %1436 = bitcast float %1435 to i32, !dbg !11
  %1437 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1436), !dbg !11
  %1438 = bitcast i32 %1437 to float, !dbg !11
  %1439 = select i1 %.not, float %1427, float %1438, !dbg !11
  %1440 = select i1 %.not, float %1438, float %1428, !dbg !11
  %1441 = select i1 %.not36, float %1439, float %1433, !dbg !11
  %1442 = bitcast float %1441 to i32, !dbg !11
  %1443 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1442), !dbg !11
  %1444 = bitcast i32 %1443 to float, !dbg !11
  %1445 = select i1 %.not36, float %1440, float %1434, !dbg !11
  %1446 = bitcast float %1445 to i32, !dbg !11
  %1447 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1446), !dbg !11
  %1448 = bitcast i32 %1447 to float, !dbg !11
  %1449 = insertelement <4 x float> poison, float %1433, i64 0, !dbg !11
  %1450 = insertelement <4 x float> %1449, float %1434, i64 1, !dbg !11
  %1451 = insertelement <4 x float> %1450, float %1444, i64 2, !dbg !11
  %1452 = insertelement <4 x float> %1451, float %1448, i64 3, !dbg !11
  %1453 = insertelement <4 x float> poison, float %1444, i64 0, !dbg !11
  %1454 = insertelement <4 x float> %1453, float %1448, i64 1, !dbg !11
  %1455 = insertelement <4 x float> %1454, float %1439, i64 2, !dbg !11
  %1456 = insertelement <4 x float> %1455, float %1440, i64 3, !dbg !11
  %1457 = select <4 x i1> %1083, <4 x float> %1452, <4 x float> %1456, !dbg !11
  %1458 = extractelement <4 x float> %947, i64 0, !dbg !11
  %1459 = extractelement <4 x float> %947, i64 1, !dbg !11
  %1460 = extractelement <4 x float> %947, i64 2, !dbg !11
  %1461 = extractelement <4 x float> %947, i64 3, !dbg !11
  %1462 = select i1 %.not, float %1459, float %1458, !dbg !11
  %1463 = bitcast float %1462 to i32, !dbg !11
  %1464 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1463), !dbg !11
  %1465 = bitcast i32 %1464 to float, !dbg !11
  %1466 = select i1 %.not, float %1458, float %1465, !dbg !11
  %1467 = select i1 %.not, float %1465, float %1459, !dbg !11
  %1468 = select i1 %.not, float %1461, float %1460, !dbg !11
  %1469 = bitcast float %1468 to i32, !dbg !11
  %1470 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1469), !dbg !11
  %1471 = bitcast i32 %1470 to float, !dbg !11
  %1472 = select i1 %.not, float %1460, float %1471, !dbg !11
  %1473 = select i1 %.not, float %1471, float %1461, !dbg !11
  %1474 = select i1 %.not36, float %1472, float %1466, !dbg !11
  %1475 = bitcast float %1474 to i32, !dbg !11
  %1476 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1475), !dbg !11
  %1477 = bitcast i32 %1476 to float, !dbg !11
  %1478 = select i1 %.not36, float %1473, float %1467, !dbg !11
  %1479 = bitcast float %1478 to i32, !dbg !11
  %1480 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1479), !dbg !11
  %1481 = bitcast i32 %1480 to float, !dbg !11
  %1482 = insertelement <4 x float> poison, float %1466, i64 0, !dbg !11
  %1483 = insertelement <4 x float> %1482, float %1467, i64 1, !dbg !11
  %1484 = insertelement <4 x float> %1483, float %1477, i64 2, !dbg !11
  %1485 = insertelement <4 x float> %1484, float %1481, i64 3, !dbg !11
  %1486 = insertelement <4 x float> poison, float %1477, i64 0, !dbg !11
  %1487 = insertelement <4 x float> %1486, float %1481, i64 1, !dbg !11
  %1488 = insertelement <4 x float> %1487, float %1472, i64 2, !dbg !11
  %1489 = insertelement <4 x float> %1488, float %1473, i64 3, !dbg !11
  %1490 = select <4 x i1> %1083, <4 x float> %1485, <4 x float> %1489, !dbg !11
  %1491 = extractelement <4 x float> %871, i64 0, !dbg !11
  %1492 = extractelement <4 x float> %871, i64 1, !dbg !11
  %1493 = extractelement <4 x float> %871, i64 2, !dbg !11
  %1494 = extractelement <4 x float> %871, i64 3, !dbg !11
  %1495 = select i1 %.not, float %1492, float %1491, !dbg !11
  %1496 = bitcast float %1495 to i32, !dbg !11
  %1497 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1496), !dbg !11
  %1498 = bitcast i32 %1497 to float, !dbg !11
  %1499 = select i1 %.not, float %1491, float %1498, !dbg !11
  %1500 = select i1 %.not, float %1498, float %1492, !dbg !11
  %1501 = select i1 %.not, float %1494, float %1493, !dbg !11
  %1502 = bitcast float %1501 to i32, !dbg !11
  %1503 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1502), !dbg !11
  %1504 = bitcast i32 %1503 to float, !dbg !11
  %1505 = select i1 %.not, float %1493, float %1504, !dbg !11
  %1506 = select i1 %.not, float %1504, float %1494, !dbg !11
  %1507 = select i1 %.not36, float %1505, float %1499, !dbg !11
  %1508 = bitcast float %1507 to i32, !dbg !11
  %1509 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1508), !dbg !11
  %1510 = bitcast i32 %1509 to float, !dbg !11
  %1511 = select i1 %.not36, float %1506, float %1500, !dbg !11
  %1512 = bitcast float %1511 to i32, !dbg !11
  %1513 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1512), !dbg !11
  %1514 = bitcast i32 %1513 to float, !dbg !11
  %1515 = insertelement <4 x float> poison, float %1499, i64 0, !dbg !11
  %1516 = insertelement <4 x float> %1515, float %1500, i64 1, !dbg !11
  %1517 = insertelement <4 x float> %1516, float %1510, i64 2, !dbg !11
  %1518 = insertelement <4 x float> %1517, float %1514, i64 3, !dbg !11
  %1519 = insertelement <4 x float> poison, float %1510, i64 0, !dbg !11
  %1520 = insertelement <4 x float> %1519, float %1514, i64 1, !dbg !11
  %1521 = insertelement <4 x float> %1520, float %1505, i64 2, !dbg !11
  %1522 = insertelement <4 x float> %1521, float %1506, i64 3, !dbg !11
  %1523 = select <4 x i1> %1083, <4 x float> %1518, <4 x float> %1522, !dbg !11
  %1524 = extractelement <4 x float> %795, i64 0, !dbg !11
  %1525 = extractelement <4 x float> %795, i64 1, !dbg !11
  %1526 = extractelement <4 x float> %795, i64 2, !dbg !11
  %1527 = extractelement <4 x float> %795, i64 3, !dbg !11
  %1528 = select i1 %.not, float %1525, float %1524, !dbg !11
  %1529 = bitcast float %1528 to i32, !dbg !11
  %1530 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1529), !dbg !11
  %1531 = bitcast i32 %1530 to float, !dbg !11
  %1532 = select i1 %.not, float %1524, float %1531, !dbg !11
  %1533 = select i1 %.not, float %1531, float %1525, !dbg !11
  %1534 = select i1 %.not, float %1527, float %1526, !dbg !11
  %1535 = bitcast float %1534 to i32, !dbg !11
  %1536 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %441, i32 %1535), !dbg !11
  %1537 = bitcast i32 %1536 to float, !dbg !11
  %1538 = select i1 %.not, float %1526, float %1537, !dbg !11
  %1539 = select i1 %.not, float %1537, float %1527, !dbg !11
  %1540 = select i1 %.not36, float %1538, float %1532, !dbg !11
  %1541 = bitcast float %1540 to i32, !dbg !11
  %1542 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1541), !dbg !11
  %1543 = select i1 %.not36, float %1539, float %1533, !dbg !11
  %1544 = bitcast float %1543 to i32, !dbg !11
  %1545 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %449, i32 %1544), !dbg !11
  %1546 = bitcast i32 %1542 to float, !dbg !11
  %1547 = bitcast i32 %1545 to float, !dbg !11
  %1548 = insertelement <4 x float> poison, float %1532, i64 0, !dbg !11
  %1549 = insertelement <4 x float> %1548, float %1533, i64 1, !dbg !11
  %1550 = insertelement <4 x float> %1549, float %1546, i64 2, !dbg !11
  %1551 = insertelement <4 x float> %1550, float %1547, i64 3, !dbg !11
  %1552 = insertelement <4 x float> poison, float %1546, i64 0, !dbg !11
  %1553 = insertelement <4 x float> %1552, float %1547, i64 1, !dbg !11
  %1554 = insertelement <4 x float> %1553, float %1538, i64 2, !dbg !11
  %1555 = insertelement <4 x float> %1554, float %1539, i64 3, !dbg !11
  %1556 = select <4 x i1> %1083, <4 x float> %1551, <4 x float> %1555, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1424, !dbg !50
  %normalized = fmul <4 x float> %row.reciprocal, %1556, !dbg !50
  %1557 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !51
  %1558 = extractelement <4 x half> %1557, i64 3, !dbg !51
  %1559 = extractelement <4 x half> %1557, i64 2, !dbg !51
  %1560 = extractelement <4 x half> %1557, i64 1, !dbg !51
  %1561 = extractelement <4 x half> %1557, i64 0, !dbg !51
  %1562 = bitcast i32 %1420 to float, !dbg !11
  %1563 = bitcast i32 %1423 to float, !dbg !11
  %1564 = insertelement <4 x float> poison, float %1410, i64 0, !dbg !11
  %1565 = insertelement <4 x float> %1564, float %1411, i64 1, !dbg !11
  %1566 = insertelement <4 x float> %1565, float %1562, i64 2, !dbg !11
  %1567 = insertelement <4 x float> %1566, float %1563, i64 3, !dbg !11
  %1568 = insertelement <4 x float> poison, float %1562, i64 0, !dbg !11
  %1569 = insertelement <4 x float> %1568, float %1563, i64 1, !dbg !11
  %1570 = insertelement <4 x float> %1569, float %1416, i64 2, !dbg !11
  %1571 = insertelement <4 x float> %1570, float %1417, i64 3, !dbg !11
  %1572 = select <4 x i1> %1083, <4 x float> %1567, <4 x float> %1571, !dbg !11
  %normalized37 = fmul <4 x float> %row.reciprocal, %1572, !dbg !50
  %1573 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !51
  %1574 = extractelement <4 x half> %1573, i64 3, !dbg !51
  %1575 = extractelement <4 x half> %1573, i64 2, !dbg !51
  %1576 = extractelement <4 x half> %1573, i64 1, !dbg !51
  %1577 = extractelement <4 x half> %1573, i64 0, !dbg !51
  %1578 = bitcast i32 %1299 to float, !dbg !11
  %1579 = bitcast i32 %1302 to float, !dbg !11
  %1580 = insertelement <4 x float> poison, float %1289, i64 0, !dbg !11
  %1581 = insertelement <4 x float> %1580, float %1290, i64 1, !dbg !11
  %1582 = insertelement <4 x float> %1581, float %1578, i64 2, !dbg !11
  %1583 = insertelement <4 x float> %1582, float %1579, i64 3, !dbg !11
  %1584 = insertelement <4 x float> poison, float %1578, i64 0, !dbg !11
  %1585 = insertelement <4 x float> %1584, float %1579, i64 1, !dbg !11
  %1586 = insertelement <4 x float> %1585, float %1295, i64 2, !dbg !11
  %1587 = insertelement <4 x float> %1586, float %1296, i64 3, !dbg !11
  %1588 = select <4 x i1> %1083, <4 x float> %1583, <4 x float> %1587, !dbg !11
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1181, !dbg !50
  %normalized39 = fmul <4 x float> %row.reciprocal38, %1588, !dbg !50
  %1589 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !51
  %1590 = extractelement <4 x half> %1589, i64 3, !dbg !51
  %1591 = extractelement <4 x half> %1589, i64 2, !dbg !51
  %1592 = extractelement <4 x half> %1589, i64 1, !dbg !51
  %1593 = extractelement <4 x half> %1589, i64 0, !dbg !51
  %1594 = bitcast i32 %1177 to float, !dbg !11
  %1595 = bitcast i32 %1180 to float, !dbg !11
  %1596 = insertelement <4 x float> poison, float %1167, i64 0, !dbg !11
  %1597 = insertelement <4 x float> %1596, float %1168, i64 1, !dbg !11
  %1598 = insertelement <4 x float> %1597, float %1594, i64 2, !dbg !11
  %1599 = insertelement <4 x float> %1598, float %1595, i64 3, !dbg !11
  %1600 = insertelement <4 x float> poison, float %1594, i64 0, !dbg !11
  %1601 = insertelement <4 x float> %1600, float %1595, i64 1, !dbg !11
  %1602 = insertelement <4 x float> %1601, float %1173, i64 2, !dbg !11
  %1603 = insertelement <4 x float> %1602, float %1174, i64 3, !dbg !11
  %1604 = select <4 x i1> %1083, <4 x float> %1599, <4 x float> %1603, !dbg !11
  %normalized40 = fmul <4 x float> %row.reciprocal38, %1604, !dbg !50
  %1605 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !51
  %1606 = extractelement <4 x half> %1605, i64 3, !dbg !51
  %1607 = extractelement <4 x half> %1605, i64 2, !dbg !51
  %1608 = extractelement <4 x half> %1605, i64 1, !dbg !51
  %1609 = extractelement <4 x half> %1605, i64 0, !dbg !51
  %address.thread974 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !52, !range !9, !frisk.register_lifetimes !5
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %1610 = lshr i64 %address.local975, 2, !dbg !52
  %address.local978 = and i64 %1610, 48, !dbg !52
  %address.local980 = and i64 %address.local975, 240, !dbg !52
  %address.local979 = add i64 %21, %address.local975, !dbg !52
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !52
  %address.local982 = add i64 %address.local981, %address.local978, !dbg !52
  %1611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %23, !dbg !52
  %address.local985.idx = shl nuw nsw i64 %address.local982, 8, !dbg !52
  %address.local985 = getelementptr inbounds nuw i8, ptr addrspace(1) %1611, i64 %address.local985.idx, !dbg !52
  %address.local987 = and i64 %1610, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !52
  store half %1609, ptr addrspace(1) %address.local988, align 2, !dbg !52
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !52
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !52
  store half %1608, ptr addrspace(1) %address.local991, align 2, !dbg !52
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !52
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !52
  store half %1607, ptr addrspace(1) %address.local994, align 2, !dbg !52
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !52
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !52
  store half %1606, ptr addrspace(1) %address.local997, align 2, !dbg !52
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !52
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !52
  store half %1593, ptr addrspace(1) %address.local1000, align 2, !dbg !52
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !52
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !52
  store half %1592, ptr addrspace(1) %address.local1003, align 2, !dbg !52
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !52
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !52
  store half %1591, ptr addrspace(1) %address.local1006, align 2, !dbg !52
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !52
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !52
  store half %1590, ptr addrspace(1) %address.local1009, align 2, !dbg !52
  %address.local1013 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local985, i64 16384, !dbg !52
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !52
  store half %1577, ptr addrspace(1) %address.local1014, align 2, !dbg !52
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !52
  store half %1576, ptr addrspace(1) %address.local1016, align 2, !dbg !52
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !52
  store half %1575, ptr addrspace(1) %address.local1018, align 2, !dbg !52
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !52
  store half %1574, ptr addrspace(1) %address.local1020, align 2, !dbg !52
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !52
  store half %1561, ptr addrspace(1) %address.local1022, align 2, !dbg !52
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !52
  store half %1560, ptr addrspace(1) %address.local1024, align 2, !dbg !52
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !52
  store half %1559, ptr addrspace(1) %address.local1026, align 2, !dbg !52
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !52
  store half %1558, ptr addrspace(1) %address.local1028, align 2, !dbg !52
  %normalized42 = fmul <4 x float> %row.reciprocal, %1523, !dbg !50
  %1612 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !51
  %1613 = extractelement <4 x half> %1612, i64 3, !dbg !51
  %1614 = extractelement <4 x half> %1612, i64 2, !dbg !51
  %1615 = extractelement <4 x half> %1612, i64 1, !dbg !51
  %1616 = extractelement <4 x half> %1612, i64 0, !dbg !51
  %normalized43 = fmul <4 x float> %row.reciprocal, %1401, !dbg !50
  %1617 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !51
  %1618 = extractelement <4 x half> %1617, i64 3, !dbg !51
  %1619 = extractelement <4 x half> %1617, i64 2, !dbg !51
  %1620 = extractelement <4 x half> %1617, i64 1, !dbg !51
  %1621 = extractelement <4 x half> %1617, i64 0, !dbg !51
  %normalized45 = fmul <4 x float> %row.reciprocal38, %1280, !dbg !50
  %1622 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !51
  %1623 = extractelement <4 x half> %1622, i64 3, !dbg !51
  %1624 = extractelement <4 x half> %1622, i64 2, !dbg !51
  %1625 = extractelement <4 x half> %1622, i64 1, !dbg !51
  %1626 = extractelement <4 x half> %1622, i64 0, !dbg !51
  %normalized46 = fmul <4 x float> %row.reciprocal38, %1158, !dbg !50
  %1627 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !51
  %1628 = extractelement <4 x half> %1627, i64 3, !dbg !51
  %1629 = extractelement <4 x half> %1627, i64 2, !dbg !51
  %1630 = extractelement <4 x half> %1627, i64 1, !dbg !51
  %1631 = extractelement <4 x half> %1627, i64 0, !dbg !51
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !52
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !52
  store half %1631, ptr addrspace(1) %address.local1044, align 2, !dbg !52
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !52
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !52
  store half %1630, ptr addrspace(1) %address.local1047, align 2, !dbg !52
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !52
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !52
  store half %1629, ptr addrspace(1) %address.local1050, align 2, !dbg !52
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !52
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !52
  store half %1628, ptr addrspace(1) %address.local1053, align 2, !dbg !52
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !52
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !52
  store half %1626, ptr addrspace(1) %address.local1056, align 2, !dbg !52
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !52
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !52
  store half %1625, ptr addrspace(1) %address.local1059, align 2, !dbg !52
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !52
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !52
  store half %1624, ptr addrspace(1) %address.local1062, align 2, !dbg !52
  %address.local1064 = or i64 %1610, 51, !dbg !52
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !52
  store half %1623, ptr addrspace(1) %address.local1065, align 2, !dbg !52
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !52
  store half %1621, ptr addrspace(1) %address.local1070, align 2, !dbg !52
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !52
  store half %1620, ptr addrspace(1) %address.local1072, align 2, !dbg !52
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !52
  store half %1619, ptr addrspace(1) %address.local1074, align 2, !dbg !52
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !52
  store half %1618, ptr addrspace(1) %address.local1076, align 2, !dbg !52
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !52
  store half %1616, ptr addrspace(1) %address.local1078, align 2, !dbg !52
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !52
  store half %1615, ptr addrspace(1) %address.local1080, align 2, !dbg !52
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !52
  store half %1614, ptr addrspace(1) %address.local1082, align 2, !dbg !52
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !52
  store half %1613, ptr addrspace(1) %address.local1084, align 2, !dbg !52
  %normalized48 = fmul <4 x float> %row.reciprocal, %1490, !dbg !50
  %1632 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !51
  %1633 = extractelement <4 x half> %1632, i64 3, !dbg !51
  %1634 = extractelement <4 x half> %1632, i64 2, !dbg !51
  %1635 = extractelement <4 x half> %1632, i64 1, !dbg !51
  %1636 = extractelement <4 x half> %1632, i64 0, !dbg !51
  %normalized49 = fmul <4 x float> %row.reciprocal, %1368, !dbg !50
  %1637 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !51
  %1638 = extractelement <4 x half> %1637, i64 3, !dbg !51
  %1639 = extractelement <4 x half> %1637, i64 2, !dbg !51
  %1640 = extractelement <4 x half> %1637, i64 1, !dbg !51
  %1641 = extractelement <4 x half> %1637, i64 0, !dbg !51
  %normalized51 = fmul <4 x float> %row.reciprocal38, %1247, !dbg !50
  %1642 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !51
  %1643 = extractelement <4 x half> %1642, i64 3, !dbg !51
  %1644 = extractelement <4 x half> %1642, i64 2, !dbg !51
  %1645 = extractelement <4 x half> %1642, i64 1, !dbg !51
  %1646 = extractelement <4 x half> %1642, i64 0, !dbg !51
  %normalized52 = fmul <4 x float> %row.reciprocal38, %1125, !dbg !50
  %1647 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !51
  %1648 = extractelement <4 x half> %1647, i64 3, !dbg !51
  %1649 = extractelement <4 x half> %1647, i64 2, !dbg !51
  %1650 = extractelement <4 x half> %1647, i64 1, !dbg !51
  %1651 = extractelement <4 x half> %1647, i64 0, !dbg !51
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !52
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !52
  store half %1651, ptr addrspace(1) %address.local1100, align 2, !dbg !52
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !52
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !52
  store half %1650, ptr addrspace(1) %address.local1103, align 2, !dbg !52
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !52
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !52
  store half %1649, ptr addrspace(1) %address.local1106, align 2, !dbg !52
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !52
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !52
  store half %1648, ptr addrspace(1) %address.local1109, align 2, !dbg !52
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !52
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !52
  store half %1646, ptr addrspace(1) %address.local1112, align 2, !dbg !52
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !52
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !52
  store half %1645, ptr addrspace(1) %address.local1115, align 2, !dbg !52
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !52
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !52
  store half %1644, ptr addrspace(1) %address.local1118, align 2, !dbg !52
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !52
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !52
  store half %1643, ptr addrspace(1) %address.local1121, align 2, !dbg !52
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !52
  store half %1641, ptr addrspace(1) %address.local1126, align 2, !dbg !52
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !52
  store half %1640, ptr addrspace(1) %address.local1128, align 2, !dbg !52
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !52
  store half %1639, ptr addrspace(1) %address.local1130, align 2, !dbg !52
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !52
  store half %1638, ptr addrspace(1) %address.local1132, align 2, !dbg !52
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !52
  store half %1636, ptr addrspace(1) %address.local1134, align 2, !dbg !52
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !52
  store half %1635, ptr addrspace(1) %address.local1136, align 2, !dbg !52
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !52
  store half %1634, ptr addrspace(1) %address.local1138, align 2, !dbg !52
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !52
  store half %1633, ptr addrspace(1) %address.local1140, align 2, !dbg !52
  %normalized54 = fmul <4 x float> %row.reciprocal, %1457, !dbg !50
  %1652 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !51
  %1653 = extractelement <4 x half> %1652, i64 3, !dbg !51
  %1654 = extractelement <4 x half> %1652, i64 2, !dbg !51
  %1655 = extractelement <4 x half> %1652, i64 1, !dbg !51
  %1656 = extractelement <4 x half> %1652, i64 0, !dbg !51
  %normalized55 = fmul <4 x float> %row.reciprocal, %1335, !dbg !50
  %1657 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !51
  %1658 = extractelement <4 x half> %1657, i64 3, !dbg !51
  %1659 = extractelement <4 x half> %1657, i64 2, !dbg !51
  %1660 = extractelement <4 x half> %1657, i64 1, !dbg !51
  %1661 = extractelement <4 x half> %1657, i64 0, !dbg !51
  %normalized57 = fmul <4 x float> %row.reciprocal38, %1214, !dbg !50
  %1662 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !51
  %1663 = extractelement <4 x half> %1662, i64 3, !dbg !51
  %1664 = extractelement <4 x half> %1662, i64 2, !dbg !51
  %1665 = extractelement <4 x half> %1662, i64 1, !dbg !51
  %1666 = extractelement <4 x half> %1662, i64 0, !dbg !51
  %normalized58 = fmul <4 x float> %row.reciprocal38, %1092, !dbg !50
  %1667 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !51
  %1668 = extractelement <4 x half> %1667, i64 3, !dbg !51
  %1669 = extractelement <4 x half> %1667, i64 2, !dbg !51
  %1670 = extractelement <4 x half> %1667, i64 1, !dbg !51
  %1671 = extractelement <4 x half> %1667, i64 0, !dbg !51
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !52
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !52
  store half %1671, ptr addrspace(1) %address.local1156, align 2, !dbg !52
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !52
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !52
  store half %1670, ptr addrspace(1) %address.local1159, align 2, !dbg !52
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !52
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !52
  store half %1669, ptr addrspace(1) %address.local1162, align 2, !dbg !52
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !52
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !52
  store half %1668, ptr addrspace(1) %address.local1165, align 2, !dbg !52
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !52
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !52
  store half %1666, ptr addrspace(1) %address.local1168, align 2, !dbg !52
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !52
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !52
  store half %1665, ptr addrspace(1) %address.local1171, align 2, !dbg !52
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !52
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !52
  store half %1664, ptr addrspace(1) %address.local1174, align 2, !dbg !52
  %address.local1176 = or i64 %1610, 115, !dbg !52
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !52
  store half %1663, ptr addrspace(1) %address.local1177, align 2, !dbg !52
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !52
  store half %1661, ptr addrspace(1) %address.local1182, align 2, !dbg !52
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !52
  store half %1660, ptr addrspace(1) %address.local1184, align 2, !dbg !52
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !52
  store half %1659, ptr addrspace(1) %address.local1186, align 2, !dbg !52
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !52
  store half %1658, ptr addrspace(1) %address.local1188, align 2, !dbg !52
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !52
  store half %1656, ptr addrspace(1) %address.local1190, align 2, !dbg !52
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !52
  store half %1655, ptr addrspace(1) %address.local1192, align 2, !dbg !52
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !52
  store half %1654, ptr addrspace(1) %address.local1194, align 2, !dbg !52
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !52
  store half %1653, ptr addrspace(1) %address.local1196, align 2, !dbg !52
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000003")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 23, column: 3, scope: !3)
!8 = !DILocation(line: 67, column: 22, scope: !3)
!9 = !{i32 0, i32 256}
!10 = !DILocation(line: 69, column: 24, scope: !3)
!11 = !DILocation(line: 57, column: 24, scope: !3)
!12 = !DILocation(line: 70, column: 26, scope: !3)
!13 = !DILocation(line: 68, column: 22, scope: !3)
!14 = !{i32 0}
!15 = !{i32 1}
!16 = !{i32 2}
!17 = !{i32 3}
!18 = !{i32 4}
!19 = !{i32 5}
!20 = !{i32 6}
!21 = !{i32 7}
!22 = !{i32 8}
!23 = !{i32 9}
!24 = !{i32 10}
!25 = !{i32 11}
!26 = !{i32 12}
!27 = !{i32 13}
!28 = !{i32 14}
!29 = !{i32 15}
!30 = !{i32 16}
!31 = !{i32 17}
!32 = !{i32 18}
!33 = !{i32 19}
!34 = !{i32 20}
!35 = !{i32 21}
!36 = !{i32 22}
!37 = !DILocation(line: 86, column: 20, scope: !3)
!38 = !DILocation(line: 71, column: 23, scope: !3)
!39 = !DILocation(line: 74, column: 20, scope: !3)
!40 = !DILocation(line: 75, column: 22, scope: !3)
!41 = !DILocation(line: 76, column: 23, scope: !3)
!42 = !DILocation(line: 77, column: 20, scope: !3)
!43 = !DILocation(line: 88, column: 22, scope: !3)
!44 = !DILocation(line: 78, column: 22, scope: !3)
!45 = !DILocation(line: 79, column: 22, scope: !3)
!46 = !DILocation(line: 81, column: 16, scope: !3)
!47 = !DILocation(line: 80, column: 20, scope: !3)
!48 = !DILocation(line: 87, column: 23, scope: !3)
!49 = !{i64 8192}
!50 = !DILocation(line: 92, column: 23, scope: !3)
!51 = !DILocation(line: 93, column: 19, scope: !3)
!52 = !DILocation(line: 94, column: 9, scope: !3)
