; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_cross_bn_k0_pass/cross_bn_k0.ll'
source_filename = "generated_ir/noncausal_k_lds_prefetch_pass/mytest_final/run-20261030-171621-98fko2hz/online-vdb-k64-lds2/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %arg, ptr addrspace(1) %arg1, ptr addrspace(1) %arg2, ptr addrspace(1) nofree writeonly captures(none) %arg3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
bb:
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !7
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !7
  %i = and i32 %kdma.lane, 31, !dbg !7
  %kdma.owner = or disjoint i32 %i, %kdma.wave, !dbg !7
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !7
  %i4 = shl nuw nsw i32 %kdma.tid, 3, !dbg !7
  %kdma.intra = and i32 %i4, 8, !dbg !7
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !7
  %i5 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %i6 = sext i32 %i5 to i64, !dbg !7
  %i7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %i8 = sext i32 %i7 to i64, !dbg !7
  %i9 = shl nsw i64 %i8, 5, !dbg !7
  %i10 = add nsw i64 %i9, %i6, !dbg !7
  %i11 = lshr i64 %i10, 5, !dbg !7
  %i12 = and i64 %i11, 576460752303423456, !dbg !7
  %i13 = and i64 %i6, 31, !dbg !7
  %i14 = or disjoint i64 %i12, %i13, !dbg !7
  %i15 = lshr i64 %i10, 10, !dbg !8
  %i16 = add nuw nsw i64 %i11, %i15, !dbg !8
  %i17 = shl nuw nsw i64 %i14, 7, !dbg !8
  %0 = shl i64 %i15, 12, !dbg !8
  %i19 = sub i64 %i17, %0, !dbg !8
  %i20 = shl i64 %i16, 19, !dbg !8
  %i21 = and i64 %i20, 16252928, !dbg !8
  %i22 = shl nsw i64 %i19, 7, !dbg !8
  %i23 = add i64 %i22, %i21, !dbg !8
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
  %i25 = ptrtoint ptr addrspace(1) %arg to i64, !dbg !8
  %.idx1399 = or disjoint i64 %address.local413, %i23, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %i26 = insertelement <2 x i64> poison, i64 %i25, i64 0, !dbg !8
  %i27 = bitcast <2 x i64> %i26 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %i27, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local424, %address.local410, !dbg !8
  %.idx1400 = or disjoint i64 %address.local425, %i23, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %i32 = or disjoint i64 %i23, 32, !dbg !8
  %i33 = or disjoint i64 %address.local413, %i32, !dbg !8
  %.tr = trunc i64 %i33 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %1 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local406, i64 0, !dbg !8
  %2 = insertelement <4 x ptr addrspace(3)> %1, ptr addrspace(3) %address.local418, i64 1, !dbg !8
  %3 = insertelement <4 x ptr addrspace(3)> %2, ptr addrspace(3) %address.local427, i64 2, !dbg !8
  %4 = insertelement <4 x ptr addrspace(3)> %3, ptr addrspace(3) %address.local431, i64 3, !dbg !8
  %5 = ptrtoint <4 x ptr addrspace(3)> %4 to <4 x i64>, !dbg !8
  %6 = trunc <4 x i64> %5 to <4 x i32>, !dbg !8
  %7 = extractelement <4 x i32> %6, i64 0, !dbg !8
  %i24 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %7), !dbg !8
  %dma.buffer.lds1398 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i24), !dbg !8
  %i28 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %8 = extractelement <4 x i32> %6, i64 1, !dbg !8
  %i29 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %8), !dbg !8
  %dma.buffer.lds1395 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i29), !dbg !8
  %i30 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %9 = extractelement <4 x i32> %6, i64 2, !dbg !8
  %i31 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %9), !dbg !8
  %dma.buffer.lds1392 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i31), !dbg !8
  %i34 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %10 = extractelement <4 x i32> %6, i64 3, !dbg !8
  %i35 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %10), !dbg !8
  %i36 = or disjoint i64 %address.local425, %i32, !dbg !8
  %.tr1401 = trunc i64 %i36 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %dma.buffer.lds1389 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i35), !dbg !8
  %i37 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
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
  %i38 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %i39 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %i40 = fmul <4 x half> %i38, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %i41 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %i42 = fmul <4 x half> %i39, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %i43 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %i44 = fmul <4 x half> %i41, splat (half 1.275630e-01), !dbg !10
  %i45 = fmul <4 x half> %i43, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %i46 = or disjoint i64 %i23, 64, !dbg !8
  %i47 = or disjoint i64 %address.local413, %i46, !dbg !8
  %.tr1402 = trunc i64 %i47 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %i48 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %i49 = or disjoint i64 %address.local425, %i46, !dbg !8
  %.tr1403 = trunc i64 %i49 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %i50 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %i51 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %i52 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %i53 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %i54 = fmul <4 x half> %i51, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %i55 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %i56 = fmul <4 x half> %i52, splat (half 1.275630e-01), !dbg !10
  %i57 = fmul <4 x half> %i53, splat (half 1.275630e-01), !dbg !10
  %i58 = fmul <4 x half> %i55, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %i59 = or disjoint i64 %i23, 96, !dbg !8
  %i60 = or disjoint i64 %address.local413, %i59, !dbg !8
  %.tr1404 = trunc i64 %i60 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %i61 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %i62 = or disjoint i64 %address.local425, %i59, !dbg !8
  %.tr1405 = trunc i64 %i62 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %i63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %i64 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %i65 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %i66 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %i67 = fmul <4 x half> %i64, splat (half 1.275630e-01), !dbg !10
  %i68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %i69 = fmul <4 x half> %i65, splat (half 1.275630e-01), !dbg !10
  %i70 = fmul <4 x half> %i66, splat (half 1.275630e-01), !dbg !10
  %i71 = fmul <4 x half> %i68, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %i72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %i73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %i74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %i75 = fmul <4 x half> %i72, splat (half 1.275630e-01), !dbg !10
  %i76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %i77 = fmul <4 x half> %i73, splat (half 1.275630e-01), !dbg !10
  %i78 = fmul <4 x half> %i74, splat (half 1.275630e-01), !dbg !10
  %i79 = fmul <4 x half> %i76, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %i80 = shl nuw nsw i32 %kdma.tid, 2, !dbg !12
  %i81 = and i32 %i80, 3084, !dbg !12
  %i82 = and i32 %kdma.tid, 1, !dbg !12
  %i83 = add nuw nsw i32 %i82, 1, !dbg !12
  %narrow = or disjoint i32 %i83, %i81, !dbg !12
  %i84 = lshr i32 %narrow, 1, !dbg !12
  %i85 = zext nneg i32 %i84 to i64, !dbg !12
  %k0.prime.i96 = shl i64 %i11, 19, !dbg !13
  %k0.prime.i97 = and i64 %k0.prime.i96, 16252928, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1506 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1507 = and i64 %k0.prime.kdma.source1506, 8, !dbg !13
  %k0.prime.kdma.source1508 = and i64 %k0.prime.kdma.source, 91, !dbg !13
  %k0.prime.kdma.source1509 = lshr i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1510 = and i64 %k0.prime.kdma.source1509, 4, !dbg !13
  %k0.prime.kdma.source1511 = add nuw nsw i64 %k0.prime.kdma.source1510, %k0.prime.kdma.source, !dbg !13
  %k0.prime.kdma.source1512 = and i64 %k0.prime.kdma.source1511, 4, !dbg !13
  %k0.prime.kdma.source1513 = or disjoint i64 %k0.prime.kdma.source1512, %k0.prime.kdma.source1508, !dbg !13
  %k0.prime.kdma.source1514 = shl nuw nsw i64 %k0.prime.kdma.source1513, 7, !dbg !13
  %k0.prime.kdma.source1515 = or disjoint i64 %k0.prime.kdma.source1514, %k0.prime.kdma.source1507, !dbg !13
  %k0.prime.kdma.source1516 = or disjoint i64 %k0.prime.kdma.source1515, %k0.prime.i97, !dbg !13
  %k0.prime.kdma.source1517 = trunc nuw nsw i64 %k0.prime.kdma.source1516 to i32, !dbg !13
  %k0.prime.kdma.source1518 = shl nuw nsw i32 %k0.prime.kdma.source1517, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1518, %kdma.intra, !dbg !13
  %k0.prime.i101 = ptrtoint ptr addrspace(1) %arg1 to i64, !dbg !13
  %k0.prime.i102 = insertelement <2 x i64> poison, i64 %k0.prime.i101, i64 0, !dbg !13
  %k0.prime.i103 = bitcast <2 x i64> %k0.prime.i102 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1373 = shufflevector <4 x i32> %k0.prime.i103, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.i104 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1534 = add i32 %dma.buffer.lds1398, 512, !dbg !13
  %k0.prime.kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1520 = lshr i64 %k0.prime.kdma.source1519, 4, !dbg !13
  %k0.prime.kdma.source1521 = and i64 %k0.prime.kdma.source1520, 8, !dbg !13
  %k0.prime.kdma.source1522 = and i64 %k0.prime.kdma.source1519, 123, !dbg !13
  %k0.prime.kdma.source1523 = lshr i64 %k0.prime.kdma.source1519, 2, !dbg !13
  %k0.prime.kdma.source1524 = and i64 %k0.prime.kdma.source1523, 4, !dbg !13
  %k0.prime.kdma.source1525 = add nuw nsw i64 %k0.prime.kdma.source1524, %k0.prime.kdma.source1519, !dbg !13
  %k0.prime.kdma.source1526 = and i64 %k0.prime.kdma.source1525, 4, !dbg !13
  %k0.prime.kdma.source1527 = or disjoint i64 %k0.prime.kdma.source1526, %k0.prime.kdma.source1522, !dbg !13
  %k0.prime.kdma.source1528 = shl nuw nsw i64 %k0.prime.kdma.source1527, 7, !dbg !13
  %k0.prime.kdma.source1529 = or disjoint i64 %k0.prime.kdma.source1528, %k0.prime.kdma.source1521, !dbg !13
  %k0.prime.kdma.source1530 = or disjoint i64 %k0.prime.kdma.source1529, %k0.prime.i97, !dbg !13
  %k0.prime.kdma.source1531 = trunc nuw nsw i64 %k0.prime.kdma.source1530 to i32, !dbg !13
  %k0.prime.kdma.source1532 = shl nuw nsw i32 %k0.prime.kdma.source1531, 1, !dbg !13
  %k0.prime.kdma.offset1533 = or disjoint i32 %k0.prime.kdma.source1532, %kdma.intra.hi, !dbg !13
  %k0.prime.i105 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1534, i32 %k0.prime.kdma.offset1533, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1546 = or disjoint i64 %k0.prime.kdma.source1515, 16, !dbg !13
  %k0.prime.kdma.source1547 = or disjoint i64 %k0.prime.kdma.source1546, %k0.prime.i97, !dbg !13
  %k0.prime.kdma.source1548 = trunc nuw nsw i64 %k0.prime.kdma.source1547 to i32, !dbg !13
  %k0.prime.kdma.source1549 = shl nuw nsw i32 %k0.prime.kdma.source1548, 1, !dbg !13
  %k0.prime.kdma.offset1550 = or disjoint i32 %k0.prime.kdma.source1549, %kdma.intra, !dbg !13
  %k0.prime.i107 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %k0.prime.kdma.offset1550, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1568 = add i32 %dma.buffer.lds1395, 512, !dbg !13
  %k0.prime.kdma.source1563 = or disjoint i64 %k0.prime.kdma.source1529, 16, !dbg !13
  %k0.prime.kdma.source1564 = or disjoint i64 %k0.prime.kdma.source1563, %k0.prime.i97, !dbg !13
  %k0.prime.kdma.source1565 = trunc nuw nsw i64 %k0.prime.kdma.source1564 to i32, !dbg !13
  %k0.prime.kdma.source1566 = shl nuw nsw i32 %k0.prime.kdma.source1565, 1, !dbg !13
  %k0.prime.kdma.offset1567 = or disjoint i32 %k0.prime.kdma.source1566, %kdma.intra.hi, !dbg !13
  %k0.prime.i108 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1568, i32 %k0.prime.kdma.offset1567, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %i295 = ptrtoint ptr addrspace(1) %arg2 to i64
  %i296 = insertelement <2 x i64> poison, i64 %i295, i64 0
  %i297 = bitcast <2 x i64> %i296 to <4 x i32>
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %i297, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  br label %bb95, !dbg !11

bb95:                                             ; preds = %bb, %k0.join
  %i9316 = phi [2 x <8 x float>] [ zeroinitializer, %bb ], [ %i1019, %k0.join ]
  %i9215 = phi [2 x <8 x float>] [ zeroinitializer, %bb ], [ %i1015, %k0.join ]
  %i9114 = phi [2 x <8 x float>] [ zeroinitializer, %bb ], [ %i1011, %k0.join ]
  %i9013 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %bb ], [ %i1007, %k0.join ]
  %i8912 = phi [2 x <1 x float>] [ zeroinitializer, %bb ], [ %i699, %k0.join ]
  %i8811 = phi [2 x <8 x float>] [ zeroinitializer, %bb ], [ %i1005, %k0.join ]
  %i8710 = phi i64 [ 0, %bb ], [ %i1020, %k0.join ]
  %i98 = shl nuw nsw i64 %i8710, 14, !dbg !13
  %i99 = add nuw nsw i64 %i98, %k0.prime.i97, !dbg !13
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !8
  %address.local501 = and i64 %address.local500, 1536, !dbg !8
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !8
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %i109 = or disjoint i64 %i99, 32, !dbg !13
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %kdma.source1580 = or disjoint i64 %i109, %k0.prime.kdma.source1515, !dbg !13
  %kdma.source1581 = trunc nuw nsw i64 %kdma.source1580 to i32, !dbg !13
  %kdma.source1582 = shl nuw nsw i32 %kdma.source1581, 1, !dbg !13
  %kdma.offset1583 = or disjoint i32 %kdma.source1582, %kdma.intra, !dbg !13
  %kdma.source1596 = or disjoint i64 %i109, %k0.prime.kdma.source1529, !dbg !13
  %kdma.source1597 = trunc nuw nsw i64 %kdma.source1596 to i32, !dbg !13
  %kdma.source1598 = shl nuw nsw i32 %kdma.source1597, 1, !dbg !13
  %kdma.offset1599 = or disjoint i32 %kdma.source1598, %kdma.intra.hi, !dbg !13
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %11 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local502, i64 0, !dbg !8
  %12 = insertelement <4 x ptr addrspace(3)> %11, ptr addrspace(3) %address.local519, i64 1, !dbg !8
  %13 = insertelement <4 x ptr addrspace(3)> %12, ptr addrspace(3) %address.local533, i64 2, !dbg !8
  %14 = insertelement <4 x ptr addrspace(3)> %13, ptr addrspace(3) %address.local537, i64 3, !dbg !8
  %15 = ptrtoint <4 x ptr addrspace(3)> %14 to <4 x i64>, !dbg !8
  %16 = trunc <4 x i64> %15 to <4 x i32>, !dbg !8
  %17 = extractelement <4 x i32> %16, i64 0, !dbg !13
  %i100 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %17), !dbg !13
  %dma.buffer.lds1374 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i100), !dbg !13
  %kdma.lds1534 = add i32 %dma.buffer.lds1374, 512, !dbg !13
  %18 = extractelement <4 x i32> %16, i64 1, !dbg !13
  %i106 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %18), !dbg !13
  %dma.buffer.lds1371 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i106), !dbg !13
  %kdma.lds1568 = add i32 %dma.buffer.lds1371, 512, !dbg !13
  %19 = extractelement <4 x i32> %16, i64 2, !dbg !13
  %i110 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %19), !dbg !13
  %dma.buffer.lds1368 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i110), !dbg !13
  %i111 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1583, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.lds1600 = add i32 %dma.buffer.lds1368, 512, !dbg !13
  %i112 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1599, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %20 = extractelement <4 x i32> %16, i64 3, !dbg !13
  %i113 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %20), !dbg !13
  %dma.buffer.lds1365 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i113), !dbg !13
  %kdma.source1613 = or disjoint i64 %i109, %k0.prime.kdma.source1546, !dbg !13
  %kdma.source1614 = trunc nuw nsw i64 %kdma.source1613 to i32, !dbg !13
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !13
  %kdma.offset1616 = or disjoint i32 %kdma.source1615, %kdma.intra, !dbg !13
  %i114 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1616, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1630 = or disjoint i64 %i109, %k0.prime.kdma.source1563, !dbg !13
  %kdma.source1631 = trunc nuw nsw i64 %kdma.source1630 to i32, !dbg !13
  %kdma.source1632 = shl nuw nsw i32 %kdma.source1631, 1, !dbg !13
  %kdma.offset1633 = or disjoint i32 %kdma.source1632, %kdma.intra.hi, !dbg !13
  %kdma.lds1634 = add i32 %dma.buffer.lds1365, 512, !dbg !13
  %i115 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
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
  %i116 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !12
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !12
  %i117 = getelementptr i8, ptr addrspace(3) %i116, i64 %.idx1410, !dbg !12
  %i118 = getelementptr [2 x i8], ptr addrspace(3) %i117, i64 %address.local546, !dbg !12
  %i119 = getelementptr [2 x i8], ptr addrspace(3) %i118, i64 %address.local550, !dbg !12
  %i120 = getelementptr [2 x i8], ptr addrspace(3) %i119, i64 %address.local553, !dbg !12
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %i120, i64 %address.local557, !dbg !12
  %i121 = add nsw i64 %.idx1410, %.idx, !dbg !12
  %.idx1893 = shl nuw nsw i64 %address.local546, 1, !dbg !12
  %i122 = or disjoint i64 %i121, %.idx1893, !dbg !12
  %i123 = or disjoint i64 %address.local550, %address.local553, !dbg !12
  %i124 = shl nsw i64 %i123, 1, !dbg !12
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032, !dbg !12
  %i125 = add nsw i64 %i122, %address.local559.idx, !dbg !12
  %i126 = add nsw i64 %i125, %i124, !dbg !12
  %i127 = lshr i64 %i126, 6, !dbg !12
  %i128 = and i64 %i127, 8, !dbg !12
  %i129 = xor i64 %i128, %i126, !dbg !12
  %i130 = sub i64 %i129, %i126, !dbg !12
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %i130, !dbg !12
  %i131 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !12
  %i132 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i131, <4 x float> zeroinitializer) #8, !dbg !12
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !12
  %i133 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !12
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !12
  %address.local571 = and i32 %address.local570, 24, !dbg !12
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !12
  %address.local573 = and i32 %address.local572, 768, !dbg !12
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !12
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !12
  %address.local576 = and i32 %address.local575, 96, !dbg !12
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !12
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !12
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %i85) #8, !dbg !12, !frisk.register_lifetimes !5
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !12
  %address.local580 = and i32 %address.local579, 1, !dbg !12
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !12
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !12
  %address.local583 = shl i64 %address.local582, 6, !dbg !12
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !12
  %i134 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !12
  %i135 = getelementptr [2 x i8], ptr addrspace(3) %i134, i64 %address.local553, !dbg !12
  %i136 = getelementptr [2 x i8], ptr addrspace(3) %i135, i64 %address.local557, !dbg !12
  %address.local588 = getelementptr i8, ptr addrspace(3) %i136, i64 64, !dbg !12
  %.idx18961899 = add i64 %address.local584, %address.local553, !dbg !12
  %i137 = shl i64 %.idx18961899, 1, !dbg !12
  %i138 = add nuw nsw i64 %address.local559.idx, 64, !dbg !12
  %i139 = add i64 %i138, %i137, !dbg !12
  %i140 = lshr i64 %i139, 6, !dbg !12
  %i141 = and i64 %i140, 8, !dbg !12
  %i142 = xor i64 %i141, %i139, !dbg !12
  %i143 = sub i64 %i142, %i139, !dbg !12
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %i143, !dbg !12
  %i144 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !12, !frisk.k_lds_pair !14
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !12
  %i145 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !12, !frisk.k_lds_pair !14
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !12
  %i146 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !12, !frisk.k_lds_pair !15
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !12
  %i147 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !12, !frisk.k_lds_pair !15
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !12
  %i148 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !12, !frisk.k_lds_pair !16
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !12
  %i149 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !12, !frisk.k_lds_pair !16
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !12
  %i150 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !12, !frisk.k_lds_pair !17
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !12
  %i151 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !12, !frisk.k_lds_pair !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i133, <4 x float> %i132) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !12
  %i153 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !12, !frisk.k_lds_pair !18
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !12
  %i154 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !12, !frisk.k_lds_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i144, <4 x float> zeroinitializer) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !12
  %i156 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !12, !frisk.k_lds_pair !19
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !12
  %i157 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !12, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i146, <4 x float> %i155) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !12
  %i159 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !12, !frisk.k_lds_pair !20
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !12
  %i160 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !12, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i148, <4 x float> zeroinitializer) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i150, <4 x float> %i161) #8, !dbg !12
  %i163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i145, <4 x float> zeroinitializer) #8, !dbg !12
  %i164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i147, <4 x float> %i163) #8, !dbg !12
  %i165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i149, <4 x float> zeroinitializer) #8, !dbg !12
  %i166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i151, <4 x float> %i165) #8, !dbg !12
  %i167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i153, <4 x float> zeroinitializer) #8, !dbg !12
  %i168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i156, <4 x float> %i167) #8, !dbg !12
  %i169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i159, <4 x float> zeroinitializer) #8, !dbg !12
  %i170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i160, <4 x float> %i169) #8, !dbg !12
  %i171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i154, <4 x float> zeroinitializer) #8, !dbg !12
  %i172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i157, <4 x float> %i171) #8, !dbg !12
  %i173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i131, <4 x float> zeroinitializer) #8, !dbg !12
  %i174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i133, <4 x float> %i173) #8, !dbg !12
  %i175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i144, <4 x float> zeroinitializer) #8, !dbg !12
  %i176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i146, <4 x float> %i175) #8, !dbg !12
  %i177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i148, <4 x float> zeroinitializer) #8, !dbg !12
  %i178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i150, <4 x float> %i177) #8, !dbg !12
  %i179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i145, <4 x float> zeroinitializer) #8, !dbg !12
  %i180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i147, <4 x float> %i179) #8, !dbg !12
  %i181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i149, <4 x float> zeroinitializer) #8, !dbg !12
  %i182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i151, <4 x float> %i181) #8, !dbg !12
  %i183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i153, <4 x float> zeroinitializer) #8, !dbg !12
  %i184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i156, <4 x float> %i183) #8, !dbg !12
  %i185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i159, <4 x float> zeroinitializer) #8, !dbg !12
  %i186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i160, <4 x float> %i185) #8, !dbg !12
  %i187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i154, <4 x float> zeroinitializer) #8, !dbg !12
  %i188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i157, <4 x float> %i187) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %i189 = or disjoint i64 %i99, 64, !dbg !13
  %kdma.source1646 = or disjoint i64 %i189, %k0.prime.kdma.source1515, !dbg !13
  %kdma.source1647 = trunc nuw nsw i64 %kdma.source1646 to i32, !dbg !13
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !13
  %kdma.offset1649 = or disjoint i32 %kdma.source1648, %kdma.intra, !dbg !13
  %i190 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset1649, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1662 = or disjoint i64 %i189, %k0.prime.kdma.source1529, !dbg !13
  %kdma.source1663 = trunc nuw nsw i64 %kdma.source1662 to i32, !dbg !13
  %kdma.source1664 = shl nuw nsw i32 %kdma.source1663, 1, !dbg !13
  %kdma.offset1665 = or disjoint i32 %kdma.source1664, %kdma.intra.hi, !dbg !13
  %i191 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1665, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1679 = or disjoint i64 %i189, %k0.prime.kdma.source1546, !dbg !13
  %kdma.source1680 = trunc nuw nsw i64 %kdma.source1679 to i32, !dbg !13
  %kdma.source1681 = shl nuw nsw i32 %kdma.source1680, 1, !dbg !13
  %kdma.offset1682 = or disjoint i32 %kdma.source1681, %kdma.intra, !dbg !13
  %i192 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1682, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1696 = or disjoint i64 %i189, %k0.prime.kdma.source1563, !dbg !13
  %kdma.source1697 = trunc nuw nsw i64 %kdma.source1696 to i32, !dbg !13
  %kdma.source1698 = shl nuw nsw i32 %kdma.source1697, 1, !dbg !13
  %kdma.offset1699 = or disjoint i32 %kdma.source1698, %kdma.intra.hi, !dbg !13
  %i193 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1699, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !12
  %i194 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !12, !frisk.k_lds_pair !21
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !12
  %i195 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !12, !frisk.k_lds_pair !21
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !12
  %i196 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !12, !frisk.k_lds_pair !22
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !12
  %i197 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !12, !frisk.k_lds_pair !22
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !12
  %i198 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !12, !frisk.k_lds_pair !23
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !12
  %i199 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !12, !frisk.k_lds_pair !23
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !12
  %i200 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !12, !frisk.k_lds_pair !24
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !12
  %i201 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !12, !frisk.k_lds_pair !24
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !12
  %i202 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !12, !frisk.k_lds_pair !25
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !12
  %i203 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !12, !frisk.k_lds_pair !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i194, <4 x float> %i152) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !12
  %i205 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !12, !frisk.k_lds_pair !26
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !12
  %i206 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !12, !frisk.k_lds_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i207 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i196, <4 x float> %i204) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !12
  %i208 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !12, !frisk.k_lds_pair !27
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !12
  %i209 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !12, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i198, <4 x float> %i158) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !12
  %i211 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !12, !frisk.k_lds_pair !28
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !12
  %i212 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !12, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i200, <4 x float> %i210) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i195, <4 x float> %i162) #8, !dbg !12
  %i215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i197, <4 x float> %i214) #8, !dbg !12
  %i216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i199, <4 x float> %i164) #8, !dbg !12
  %i217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i201, <4 x float> %i216) #8, !dbg !12
  %i218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i202, <4 x float> %i166) #8, !dbg !12
  %i219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i205, <4 x float> %i218) #8, !dbg !12
  %i220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i208, <4 x float> %i168) #8, !dbg !12
  %i221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i211, <4 x float> %i220) #8, !dbg !12
  %i222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i203, <4 x float> %i170) #8, !dbg !12
  %i223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i206, <4 x float> %i222) #8, !dbg !12
  %i224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i209, <4 x float> %i172) #8, !dbg !12
  %i225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i212, <4 x float> %i224) #8, !dbg !12
  %i226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i194, <4 x float> %i174) #8, !dbg !12
  %i227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i196, <4 x float> %i226) #8, !dbg !12
  %i228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i198, <4 x float> %i176) #8, !dbg !12
  %i229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i200, <4 x float> %i228) #8, !dbg !12
  %i230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i195, <4 x float> %i178) #8, !dbg !12
  %i231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i197, <4 x float> %i230) #8, !dbg !12
  %i232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i199, <4 x float> %i180) #8, !dbg !12
  %i233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i201, <4 x float> %i232) #8, !dbg !12
  %i234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i202, <4 x float> %i182) #8, !dbg !12
  %i235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i205, <4 x float> %i234) #8, !dbg !12
  %i236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i208, <4 x float> %i184) #8, !dbg !12
  %i237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i211, <4 x float> %i236) #8, !dbg !12
  %i238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i203, <4 x float> %i186) #8, !dbg !12
  %i239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i206, <4 x float> %i238) #8, !dbg !12
  %i240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i209, <4 x float> %i188) #8, !dbg !12
  %i241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i212, <4 x float> %i240) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %i242 = or disjoint i64 %i99, 96, !dbg !13
  %kdma.source1712 = or disjoint i64 %i242, %k0.prime.kdma.source1515, !dbg !13
  %kdma.source1713 = trunc nuw nsw i64 %kdma.source1712 to i32, !dbg !13
  %kdma.source1714 = shl nuw nsw i32 %kdma.source1713, 1, !dbg !13
  %kdma.offset1715 = or disjoint i32 %kdma.source1714, %kdma.intra, !dbg !13
  %i243 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1715, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1728 = or disjoint i64 %i242, %k0.prime.kdma.source1529, !dbg !13
  %kdma.source1729 = trunc nuw nsw i64 %kdma.source1728 to i32, !dbg !13
  %kdma.source1730 = shl nuw nsw i32 %kdma.source1729, 1, !dbg !13
  %kdma.offset1731 = or disjoint i32 %kdma.source1730, %kdma.intra.hi, !dbg !13
  %i244 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1731, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1745 = or disjoint i64 %i242, %k0.prime.kdma.source1546, !dbg !13
  %kdma.source1746 = trunc nuw nsw i64 %kdma.source1745 to i32, !dbg !13
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !13
  %kdma.offset1748 = or disjoint i32 %kdma.source1747, %kdma.intra, !dbg !13
  %i245 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1748, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1762 = or disjoint i64 %i242, %k0.prime.kdma.source1563, !dbg !13
  %kdma.source1763 = trunc nuw nsw i64 %kdma.source1762 to i32, !dbg !13
  %kdma.source1764 = shl nuw nsw i32 %kdma.source1763, 1, !dbg !13
  %kdma.offset1765 = or disjoint i32 %kdma.source1764, %kdma.intra.hi, !dbg !13
  %i246 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1765, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %i247 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !12, !frisk.k_lds_pair !29
  %i248 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !12, !frisk.k_lds_pair !29
  %i249 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !12, !frisk.k_lds_pair !30
  %i250 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !12, !frisk.k_lds_pair !30
  %i251 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !12, !frisk.k_lds_pair !31
  %i252 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !12, !frisk.k_lds_pair !31
  %i253 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !12, !frisk.k_lds_pair !32
  %i254 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !12, !frisk.k_lds_pair !32
  %i255 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !12, !frisk.k_lds_pair !33
  %i256 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !12, !frisk.k_lds_pair !33
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i247, <4 x float> %i207) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i258 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !12, !frisk.k_lds_pair !34
  %i259 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !12, !frisk.k_lds_pair !34
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i249, <4 x float> %i257) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i261 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !12, !frisk.k_lds_pair !35
  %i262 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !12, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i251, <4 x float> %i213) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i264 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !12, !frisk.k_lds_pair !36
  %i265 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !12, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i253, <4 x float> %i263) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %i267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i248, <4 x float> %i215) #8, !dbg !12
  %i268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i250, <4 x float> %i267) #8, !dbg !12
  %i269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i252, <4 x float> %i217) #8, !dbg !12
  %i270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i254, <4 x float> %i269) #8, !dbg !12
  %i271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i255, <4 x float> %i219) #8, !dbg !12
  %i272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i258, <4 x float> %i271) #8, !dbg !12
  %i273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i261, <4 x float> %i221) #8, !dbg !12
  %i274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i264, <4 x float> %i273) #8, !dbg !12
  %i275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i256, <4 x float> %i223) #8, !dbg !12
  %i276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i259, <4 x float> %i275) #8, !dbg !12
  %i277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i262, <4 x float> %i225) #8, !dbg !12
  %i278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i265, <4 x float> %i277) #8, !dbg !12
  %i279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i247, <4 x float> %i227) #8, !dbg !12
  %i280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i249, <4 x float> %i279) #8, !dbg !12
  %i281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i251, <4 x float> %i229) #8, !dbg !12
  %i282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i253, <4 x float> %i281) #8, !dbg !12
  %i283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i248, <4 x float> %i231) #8, !dbg !12
  %i284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i250, <4 x float> %i283) #8, !dbg !12
  %i285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i252, <4 x float> %i233) #8, !dbg !12
  %i286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i254, <4 x float> %i285) #8, !dbg !12
  %i287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i255, <4 x float> %i235) #8, !dbg !12
  %i288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i258, <4 x float> %i287) #8, !dbg !12
  %i289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i261, <4 x float> %i237) #8, !dbg !12
  %i290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i264, <4 x float> %i289) #8, !dbg !12
  %i291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i256, <4 x float> %i239) #8, !dbg !12
  %i292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i259, <4 x float> %i291) #8, !dbg !12
  %i293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i262, <4 x float> %i241) #8, !dbg !12
  %i294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i265, <4 x float> %i293) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !37
  %address.local918 = and i64 %address.local917, 8064, !dbg !37
  %address.local920 = and i64 %address.local500, 24, !dbg !37
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !37
  %.idx14191420 = or disjoint i64 %address.local921, %i99, !dbg !37
  %.idx14191420.tr = trunc nuw nsw i64 %.idx14191420 to i32, !dbg !37
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !37
  %i298 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !37
  %address.local924 = and i64 %address.local923, 16256, !dbg !37
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !37
  %.idx14211422 = or disjoint i64 %address.local927, %i99, !dbg !37
  %.idx14211422.tr = trunc nuw nsw i64 %.idx14211422 to i32, !dbg !37
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !37
  %i299 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %i300 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !12, !frisk.k_tail_pair !14
  %i312 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !12, !frisk.k_tail_pair !14
  %i302 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !12, !frisk.k_tail_pair !15
  %i319 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !12, !frisk.k_tail_pair !15
  %i303 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !12, !frisk.k_tail_pair !16
  %i320 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !12, !frisk.k_tail_pair !16
  %i311 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !12, !frisk.k_tail_pair !17
  %i327 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !12, !frisk.k_tail_pair !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %i301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i300, <4 x float> %i260) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %i304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i302, <4 x float> %i301) #8, !dbg !12
  %i305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i303, <4 x float> %i266) #8, !dbg !12
  %i306 = extractelement <4 x float> %i304, i64 0, !dbg !38
  %i307 = tail call float @llvm.maxnum.f32(float %i306, float -inf), !dbg !39, !frisk.local_reduction !5
  %i308 = extractelement <4 x float> %i304, i64 1, !dbg !38
  %i309 = extractelement <4 x float> %i304, i64 2, !dbg !38
  %i310 = extractelement <4 x float> %i304, i64 3, !dbg !38
  %i313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i311, <4 x float> %i305) #8, !dbg !12
  %i328 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !12, !frisk.k_tail_pair !18
  %i344 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !12, !frisk.k_tail_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %i314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i312, <4 x float> %i268) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !38, !frisk.k_tail_prefetch !5
  %i315 = extractelement <4 x float> %i313, i64 0, !dbg !38
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %i307, float %i315), !dbg !39, !frisk.local_reduction !5
  %i316 = extractelement <4 x float> %i313, i64 1, !dbg !38
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %i308, float %i316), !dbg !39, !frisk.local_reduction !5
  %i317 = extractelement <4 x float> %i313, i64 2, !dbg !38
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %i309, float %i317), !dbg !39, !frisk.local_reduction !5
  %i318 = extractelement <4 x float> %i313, i64 3, !dbg !38
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %i310, float %i318), !dbg !39, !frisk.local_reduction !5
  %i335 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !12, !frisk.k_tail_pair !19
  %i351 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !12, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %i321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i319, <4 x float> %i314) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %i336 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !12, !frisk.k_tail_pair !20
  %i352 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !12, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %i322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i320, <4 x float> %i270) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !38, !frisk.k_tail_prefetch !5
  %i323 = extractelement <4 x float> %i321, i64 0, !dbg !38
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %i323), !dbg !39, !frisk.local_reduction !5
  %i324 = extractelement <4 x float> %i321, i64 1, !dbg !38
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %i324), !dbg !39, !frisk.local_reduction !5
  %i325 = extractelement <4 x float> %i321, i64 2, !dbg !38
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %i325), !dbg !39, !frisk.local_reduction !5
  %i326 = extractelement <4 x float> %i321, i64 3, !dbg !38
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %i326), !dbg !39, !frisk.local_reduction !5
  %i343 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !12, !frisk.k_tail_pair !21
  %i359 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !12, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %i329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i327, <4 x float> %i322) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %i330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i328, <4 x float> %i272) #8, !dbg !12
  %i331 = extractelement <4 x float> %i329, i64 0, !dbg !38
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %i331), !dbg !39, !frisk.local_reduction !5
  %i332 = extractelement <4 x float> %i329, i64 1, !dbg !38
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %i332), !dbg !39, !frisk.local_reduction !5
  %i333 = extractelement <4 x float> %i329, i64 2, !dbg !38
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %i333), !dbg !39, !frisk.local_reduction !5
  %i334 = extractelement <4 x float> %i329, i64 3, !dbg !38
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %i334), !dbg !39, !frisk.local_reduction !5
  %i337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i335, <4 x float> %i330) #8, !dbg !12
  %i338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i336, <4 x float> %i274) #8, !dbg !12
  %i339 = extractelement <4 x float> %i337, i64 0, !dbg !38
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %i339), !dbg !39, !frisk.local_reduction !5
  %i340 = extractelement <4 x float> %i337, i64 1, !dbg !38
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %i340), !dbg !39, !frisk.local_reduction !5
  %i341 = extractelement <4 x float> %i337, i64 2, !dbg !38
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %i341), !dbg !39, !frisk.local_reduction !5
  %i342 = extractelement <4 x float> %i337, i64 3, !dbg !38
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %i342), !dbg !39, !frisk.local_reduction !5
  %i345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i343, <4 x float> %i338) #8, !dbg !12
  %i346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i344, <4 x float> %i276) #8, !dbg !12
  %i347 = extractelement <4 x float> %i345, i64 0, !dbg !38
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %i347), !dbg !39, !frisk.local_reduction !5
  %i348 = extractelement <4 x float> %i345, i64 1, !dbg !38
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %i348), !dbg !39, !frisk.local_reduction !5
  %i349 = extractelement <4 x float> %i345, i64 2, !dbg !38
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %i349), !dbg !39, !frisk.local_reduction !5
  %i350 = extractelement <4 x float> %i345, i64 3, !dbg !38
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %i350), !dbg !39, !frisk.local_reduction !5
  %i353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i351, <4 x float> %i346) #8, !dbg !12
  %i354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i352, <4 x float> %i278) #8, !dbg !12
  %i355 = extractelement <4 x float> %i353, i64 0, !dbg !38
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %i355), !dbg !39, !frisk.local_reduction !5
  %i356 = extractelement <4 x float> %i353, i64 1, !dbg !38
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %i356), !dbg !39, !frisk.local_reduction !5
  %i357 = extractelement <4 x float> %i353, i64 2, !dbg !38
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %i357), !dbg !39, !frisk.local_reduction !5
  %i358 = extractelement <4 x float> %i353, i64 3, !dbg !38
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %i358), !dbg !39, !frisk.local_reduction !5
  %i360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i359, <4 x float> %i354) #8, !dbg !12
  %i361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i300, <4 x float> %i280) #8, !dbg !12
  %i362 = extractelement <4 x float> %i360, i64 0, !dbg !38
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %i362), !dbg !39, !frisk.local_reduction !5
  %i363 = extractelement <4 x float> %i360, i64 1, !dbg !38
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %i363), !dbg !39, !frisk.local_reduction !5
  %i364 = extractelement <4 x float> %i360, i64 2, !dbg !38
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %i364), !dbg !39, !frisk.local_reduction !5
  %i365 = extractelement <4 x float> %i360, i64 3, !dbg !38
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %i365), !dbg !39, !frisk.local_reduction !5
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !39, !frisk.local_reduction !5
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !39, !frisk.local_reduction !5
  %i366 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !39, !frisk.local_reduction !5
  %i367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i302, <4 x float> %i361) #8, !dbg !12
  %i368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i303, <4 x float> %i282) #8, !dbg !12
  %i369 = extractelement <4 x float> %i367, i64 0, !dbg !38
  %i370 = extractelement <4 x float> %i367, i64 1, !dbg !38
  %i371 = extractelement <4 x float> %i367, i64 2, !dbg !38
  %i372 = extractelement <4 x float> %i367, i64 3, !dbg !38
  %i373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i311, <4 x float> %i368) #8, !dbg !12
  %i374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i312, <4 x float> %i284) #8, !dbg !12
  %i375 = extractelement <4 x float> %i373, i64 0, !dbg !38
  %i376 = extractelement <4 x float> %i373, i64 1, !dbg !38
  %i377 = extractelement <4 x float> %i373, i64 2, !dbg !38
  %i378 = extractelement <4 x float> %i373, i64 3, !dbg !38
  %i379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i319, <4 x float> %i374) #8, !dbg !12
  %i380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i320, <4 x float> %i286) #8, !dbg !12
  %i381 = extractelement <4 x float> %i379, i64 0, !dbg !38
  %i382 = extractelement <4 x float> %i379, i64 1, !dbg !38
  %i383 = extractelement <4 x float> %i379, i64 2, !dbg !38
  %i384 = extractelement <4 x float> %i379, i64 3, !dbg !38
  %i385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i327, <4 x float> %i380) #8, !dbg !12
  %i386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i328, <4 x float> %i288) #8, !dbg !12
  %i387 = extractelement <4 x float> %i385, i64 0, !dbg !38
  %i388 = extractelement <4 x float> %i385, i64 1, !dbg !38
  %i389 = extractelement <4 x float> %i385, i64 2, !dbg !38
  %i390 = extractelement <4 x float> %i385, i64 3, !dbg !38
  %i391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i335, <4 x float> %i386) #8, !dbg !12
  %i392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i336, <4 x float> %i290) #8, !dbg !12
  %i393 = extractelement <4 x float> %i391, i64 0, !dbg !38
  %i394 = extractelement <4 x float> %i391, i64 1, !dbg !38
  %i395 = extractelement <4 x float> %i391, i64 2, !dbg !38
  %i396 = extractelement <4 x float> %i391, i64 3, !dbg !38
  %i397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i343, <4 x float> %i392) #8, !dbg !12
  %i398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i344, <4 x float> %i292) #8, !dbg !12
  %i399 = extractelement <4 x float> %i397, i64 0, !dbg !38
  %i400 = extractelement <4 x float> %i397, i64 1, !dbg !38
  %i401 = extractelement <4 x float> %i397, i64 2, !dbg !38
  %i402 = extractelement <4 x float> %i397, i64 3, !dbg !38
  %i403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i351, <4 x float> %i398) #8, !dbg !12
  %i404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i352, <4 x float> %i294) #8, !dbg !12
  %i405 = extractelement <4 x float> %i403, i64 0, !dbg !38
  %i406 = extractelement <4 x float> %i403, i64 1, !dbg !38
  %i407 = extractelement <4 x float> %i403, i64 2, !dbg !38
  %i408 = extractelement <4 x float> %i403, i64 3, !dbg !38
  %i409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i359, <4 x float> %i404) #8, !dbg !12
  %i410 = extractelement <4 x float> %i409, i64 0, !dbg !38
  %i411 = extractelement <4 x float> %i409, i64 1, !dbg !38
  %i412 = extractelement <4 x float> %i409, i64 2, !dbg !38
  %i413 = extractelement <4 x float> %i409, i64 3, !dbg !38
  %i414 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !39
  %i415 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %i414), !dbg !39
  %i416 = add nuw nsw i32 %i415, 64, !dbg !39
  %i417 = xor i32 %i415, 16, !dbg !39
  %i418 = and i32 %i416, 192, !dbg !39
  %i419 = icmp samesign ult i32 %i417, %i418, !dbg !39
  %i420 = select i1 %i419, i32 %i417, i32 %i415, !dbg !39
  %i421 = shl nuw nsw i32 %i420, 2, !dbg !39
  %i422 = bitcast float %i366 to i32, !dbg !39
  %i423 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i422), !dbg !39
  %i424 = bitcast i32 %i423 to float, !dbg !39
  %i425 = tail call float @llvm.maxnum.f32(float %i366, float %i424), !dbg !39
  %i426 = xor i32 %i415, 32, !dbg !39
  %i427 = icmp samesign ult i32 %i426, %i418, !dbg !39
  %i428 = select i1 %i427, i32 %i426, i32 %i415, !dbg !39
  %i429 = shl nuw nsw i32 %i428, 2, !dbg !39
  %i430 = bitcast float %i425 to i32, !dbg !39
  %i431 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i430), !dbg !39
  %i432 = bitcast i32 %i431 to float, !dbg !39
  %i433 = tail call float @llvm.maxnum.f32(float %i425, float %i432), !dbg !39
  %i434 = extractvalue [2 x <1 x float>] %i9013, 0, !dbg !39
  %i435 = insertelement <1 x float> poison, float %i433, i64 0, !dbg !39
  %i436 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %i435, <1 x float> %i434), !dbg !39
  %i437 = tail call float @llvm.maxnum.f32(float %i369, float -inf), !dbg !39, !frisk.local_reduction !5
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %i437, float %i375), !dbg !39, !frisk.local_reduction !5
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %i370, float %i376), !dbg !39, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %i371, float %i377), !dbg !39, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %i372, float %i378), !dbg !39, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %i381), !dbg !39, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %i382), !dbg !39, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %i383), !dbg !39, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %i384), !dbg !39, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %i387), !dbg !39, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %i388), !dbg !39, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %i389), !dbg !39, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %i390), !dbg !39, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %i393), !dbg !39, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %i394), !dbg !39, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %i395), !dbg !39, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %i396), !dbg !39, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %i399), !dbg !39, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %i400), !dbg !39, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %i401), !dbg !39, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %i402), !dbg !39, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %i405), !dbg !39, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %i406), !dbg !39, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %i407), !dbg !39, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %i408), !dbg !39, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %i410), !dbg !39, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %i411), !dbg !39, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %i412), !dbg !39, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %i413), !dbg !39, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !39, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !39, !frisk.local_reduction !5
  %i438 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !39, !frisk.local_reduction !5
  %i439 = bitcast float %i438 to i32, !dbg !39
  %i440 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i439), !dbg !39
  %i441 = bitcast i32 %i440 to float, !dbg !39
  %i442 = tail call float @llvm.maxnum.f32(float %i438, float %i441), !dbg !39
  %i443 = bitcast float %i442 to i32, !dbg !39
  %i444 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i443), !dbg !39
  %i445 = bitcast i32 %i444 to float, !dbg !39
  %i446 = tail call float @llvm.maxnum.f32(float %i442, float %i445), !dbg !39
  %i447 = extractvalue [2 x <1 x float>] %i9013, 1, !dbg !39
  %i448 = insertelement <1 x float> poison, float %i446, i64 0, !dbg !39
  %i449 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %i448, <1 x float> %i447), !dbg !39
  %i450 = shufflevector <1 x float> %i436, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %i459 = shufflevector <1 x float> %i449, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %i468 = fsub <1 x float> %i434, %i436, !dbg !41
  %i469 = extractelement <1 x float> %i468, i64 0, !dbg !42
  %i470 = tail call float @llvm.amdgcn.exp2.f32(float %i469), !dbg !42
  %i471 = insertelement <4 x float> poison, float %i470, i64 0, !dbg !43
  %i472 = shufflevector <4 x float> %i471, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %i473 = extractvalue [2 x <1 x float>] %i8912, 0, !dbg !44
  %i474 = extractelement <1 x float> %i473, i64 0, !dbg !44
  %i475 = fmul float %i474, %i470, !dbg !44
  %i476 = fsub <1 x float> %i447, %i449, !dbg !41
  %i477 = extractelement <1 x float> %i476, i64 0, !dbg !42
  %i478 = tail call float @llvm.amdgcn.exp2.f32(float %i477), !dbg !42
  %i479 = insertelement <4 x float> poison, float %i478, i64 0, !dbg !43
  %i480 = shufflevector <4 x float> %i479, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %i481 = extractvalue [2 x <1 x float>] %i8912, 1, !dbg !44
  %i482 = extractelement <1 x float> %i481, i64 0, !dbg !44
  %i483 = fmul float %i482, %i478, !dbg !44
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1436 = or disjoint i64 %address.local921, %i109, !dbg !37
  %vdb.address1437 = trunc nuw nsw i64 %vdb.address1436 to i32, !dbg !37
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !37
  %vdb.address1445 = or disjoint i64 %address.local927, %i109, !dbg !37
  %vdb.address1446 = trunc nuw nsw i64 %vdb.address1445 to i32, !dbg !37
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !37
  %vdb.dma.base = add i32 %dma.buffer.lds1374, 8192, !dbg !45
  %i700 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %vdb.dma.base1452 = add i32 %dma.buffer.lds1371, 8192, !dbg !45
  %i701 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !45
  %address.local930 = and i32 %address.local929, 120, !dbg !45
  %address.local932 = and i32 %address.local575, 1536, !dbg !45
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !45
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !45
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !45
  %i702 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i703 = bitcast <4 x i32> %i702 to <8 x half>, !dbg !45
  %address.local936 = and i64 %address.local499, 60, !dbg !45
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !45
  %address.local939.idx = shl nuw nsw i64 %address.local937, 4, !dbg !45
  %address.local939 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local939.idx, !dbg !45
  %address.local940 = lshr i64 %address.local555, 4, !dbg !45
  %address.local942.idx = mul nuw nsw i64 %address.local940, 768, !dbg !45
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local942.idx, !dbg !45
  %address.local944.idx = mul nsw i64 %address.local542, -64, !dbg !45
  %address.local944 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local944.idx, !dbg !45
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !45
  %i704 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i705 = bitcast <4 x i32> %i704 to <8 x half>, !dbg !45
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !45
  %i706 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i707 = bitcast <4 x i32> %i706 to <8 x half>, !dbg !45
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !45
  %i708 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i709 = bitcast <4 x i32> %i708 to <8 x half>, !dbg !45
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !45
  %i710 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i711 = bitcast <4 x i32> %i710 to <8 x half>, !dbg !45
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !45
  %i712 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i713 = bitcast <4 x i32> %i712 to <8 x half>, !dbg !45
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !45
  %i714 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i715 = bitcast <4 x i32> %i714 to <8 x half>, !dbg !45
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !45
  %i716 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i717 = shufflevector <8 x half> %i703, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %i718 = extractvalue [2 x <8 x float>] %i8811, 0, !dbg !43
  %i719 = shufflevector <8 x float> %i718, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %i720 = fmul <4 x float> %i719, %i472, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i451 = fsub <4 x float> %i304, %i450, !dbg !40
  %i484 = extractelement <4 x float> %i451, i64 0, !dbg !47
  %i485 = tail call float @llvm.amdgcn.exp2.f32(float %i484), !dbg !47
  %i486 = insertelement <4 x float> poison, float %i485, i64 0, !dbg !48
  %i487 = extractelement <4 x float> %i451, i64 1, !dbg !47
  %i488 = tail call float @llvm.amdgcn.exp2.f32(float %i487), !dbg !47
  %i489 = insertelement <4 x float> %i486, float %i488, i64 1, !dbg !48
  %i490 = extractelement <4 x float> %i451, i64 2, !dbg !47
  %i491 = tail call float @llvm.amdgcn.exp2.f32(float %i490), !dbg !47
  %i492 = insertelement <4 x float> %i489, float %i491, i64 2, !dbg !48
  %i493 = extractelement <4 x float> %i451, i64 3, !dbg !47
  %i494 = tail call float @llvm.amdgcn.exp2.f32(float %i493), !dbg !47
  %i495 = insertelement <4 x float> %i492, float %i494, i64 3, !dbg !48
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i495) #8, !dbg !48, !frisk.register_lifetimes !5
  %i496 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !48
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i496) #8, !dbg !47, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i721 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i717, <4 x float> %i720) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i722 = shufflevector <8 x half> %i703, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %i723 = shufflevector <8 x float> %i718, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %i724 = fmul <4 x float> %i723, %i472, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i725 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i722, <4 x float> %i724) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %i726 = extractvalue [2 x <8 x float>] %i8811, 1, !dbg !43
  %i727 = shufflevector <8 x float> %i726, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %i728 = fmul <4 x float> %i727, %i480, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i460 = fsub <4 x float> %i367, %i459, !dbg !40
  %i592 = extractelement <4 x float> %i460, i64 0, !dbg !47
  %i593 = tail call float @llvm.amdgcn.exp2.f32(float %i592), !dbg !47
  %i594 = insertelement <4 x float> poison, float %i593, i64 0, !dbg !48
  %i595 = extractelement <4 x float> %i460, i64 1, !dbg !47
  %i596 = tail call float @llvm.amdgcn.exp2.f32(float %i595), !dbg !47
  %i597 = insertelement <4 x float> %i594, float %i596, i64 1, !dbg !48
  %i598 = extractelement <4 x float> %i460, i64 2, !dbg !47
  %i599 = tail call float @llvm.amdgcn.exp2.f32(float %i598), !dbg !47
  %i600 = insertelement <4 x float> %i597, float %i599, i64 2, !dbg !48
  %i601 = extractelement <4 x float> %i460, i64 3, !dbg !47
  %i602 = tail call float @llvm.amdgcn.exp2.f32(float %i601), !dbg !47
  %i603 = insertelement <4 x float> %i600, float %i602, i64 3, !dbg !48
  %math.result388 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i603) #8, !dbg !48, !frisk.register_lifetimes !5
  %i604 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !48
  %math.packed389 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i604) #8, !dbg !47, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i729 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i717, <4 x float> %i728) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %i730 = shufflevector <8 x float> %i726, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %i731 = fmul <4 x float> %i730, %i480, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i732 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i722, <4 x float> %i731) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i733 = shufflevector <8 x half> %i705, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i452 = fsub <4 x float> %i313, %i450, !dbg !40
  %i497 = extractelement <4 x float> %i452, i64 0, !dbg !47
  %i498 = tail call float @llvm.amdgcn.exp2.f32(float %i497), !dbg !47
  %local.reduce89 = fadd float %i485, %i498, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %local.reduce89) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i499 = insertelement <4 x float> poison, float %i498, i64 0, !dbg !48
  %i500 = extractelement <4 x float> %i452, i64 1, !dbg !47
  %i501 = tail call float @llvm.amdgcn.exp2.f32(float %i500), !dbg !47
  %local.reduce90 = fadd float %i488, %i501, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1 = tail call float asm sideeffect "", "=v,0"(float %local.reduce90) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i502 = insertelement <4 x float> %i499, float %i501, i64 1, !dbg !48
  %i503 = extractelement <4 x float> %i452, i64 2, !dbg !47
  %i504 = tail call float @llvm.amdgcn.exp2.f32(float %i503), !dbg !47
  %local.reduce91 = fadd float %i491, %i504, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor2 = tail call float asm sideeffect "", "=v,0"(float %local.reduce91) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i505 = insertelement <4 x float> %i502, float %i504, i64 2, !dbg !48
  %i506 = extractelement <4 x float> %i452, i64 3, !dbg !47
  %i507 = tail call float @llvm.amdgcn.exp2.f32(float %i506), !dbg !47
  %local.reduce92 = fadd float %i494, %i507, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor3 = tail call float asm sideeffect "", "=v,0"(float %local.reduce92) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i508 = insertelement <4 x float> %i505, float %i507, i64 3, !dbg !48
  %math.result374 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i508) #8, !dbg !48, !frisk.register_lifetimes !5
  %i509 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !48
  %math.packed375 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i509) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i734 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i733, <4 x float> %i721) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i735 = shufflevector <8 x half> %i705, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i736 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i735, <4 x float> %i725) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i461 = fsub <4 x float> %i373, %i459, !dbg !40
  %i605 = extractelement <4 x float> %i461, i64 0, !dbg !47
  %i606 = tail call float @llvm.amdgcn.exp2.f32(float %i605), !dbg !47
  %local.reduce = fadd float %i593, %i606, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor4 = tail call float asm sideeffect "", "=v,0"(float %local.reduce) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i607 = insertelement <4 x float> poison, float %i606, i64 0, !dbg !48
  %i608 = extractelement <4 x float> %i461, i64 1, !dbg !47
  %i609 = tail call float @llvm.amdgcn.exp2.f32(float %i608), !dbg !47
  %local.reduce59 = fadd float %i596, %i609, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor5 = tail call float asm sideeffect "", "=v,0"(float %local.reduce59) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i610 = insertelement <4 x float> %i607, float %i609, i64 1, !dbg !48
  %i611 = extractelement <4 x float> %i461, i64 2, !dbg !47
  %i612 = tail call float @llvm.amdgcn.exp2.f32(float %i611), !dbg !47
  %local.reduce60 = fadd float %i599, %i612, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor6 = tail call float asm sideeffect "", "=v,0"(float %local.reduce60) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i613 = insertelement <4 x float> %i610, float %i612, i64 2, !dbg !48
  %i614 = extractelement <4 x float> %i461, i64 3, !dbg !47
  %i615 = tail call float @llvm.amdgcn.exp2.f32(float %i614), !dbg !47
  %local.reduce61 = fadd float %i602, %i615, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor7 = tail call float asm sideeffect "", "=v,0"(float %local.reduce61) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i616 = insertelement <4 x float> %i613, float %i615, i64 3, !dbg !48
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i616) #8, !dbg !48, !frisk.register_lifetimes !5
  %i617 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !48
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i617) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i737 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i733, <4 x float> %i729) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i738 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i735, <4 x float> %i732) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i739 = shufflevector <8 x half> %i707, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i453 = fsub <4 x float> %i321, %i450, !dbg !40
  %i510 = extractelement <4 x float> %i453, i64 0, !dbg !47
  %i511 = tail call float @llvm.amdgcn.exp2.f32(float %i510), !dbg !47
  %local.reduce93 = fadd float %i511, %k0.sum.anchor, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor8 = tail call float asm sideeffect "", "=v,0"(float %local.reduce93) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i512 = insertelement <4 x float> poison, float %i511, i64 0, !dbg !48
  %i513 = extractelement <4 x float> %i453, i64 1, !dbg !47
  %i514 = tail call float @llvm.amdgcn.exp2.f32(float %i513), !dbg !47
  %local.reduce94 = fadd float %i514, %k0.sum.anchor1, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor9 = tail call float asm sideeffect "", "=v,0"(float %local.reduce94) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i515 = insertelement <4 x float> %i512, float %i514, i64 1, !dbg !48
  %i516 = extractelement <4 x float> %i453, i64 2, !dbg !47
  %i517 = tail call float @llvm.amdgcn.exp2.f32(float %i516), !dbg !47
  %local.reduce95 = fadd float %i517, %k0.sum.anchor2, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor10 = tail call float asm sideeffect "", "=v,0"(float %local.reduce95) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i518 = insertelement <4 x float> %i515, float %i517, i64 2, !dbg !48
  %i519 = extractelement <4 x float> %i453, i64 3, !dbg !47
  %i520 = tail call float @llvm.amdgcn.exp2.f32(float %i519), !dbg !47
  %local.reduce96 = fadd float %i520, %k0.sum.anchor3, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor11 = tail call float asm sideeffect "", "=v,0"(float %local.reduce96) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i521 = insertelement <4 x float> %i518, float %i520, i64 3, !dbg !48
  %math.result376 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i521) #8, !dbg !48, !frisk.register_lifetimes !5
  %i522 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !48
  %math.packed377 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i522) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i740 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i739, <4 x float> %i734) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i741 = shufflevector <8 x half> %i707, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i742 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i741, <4 x float> %i736) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i462 = fsub <4 x float> %i379, %i459, !dbg !40
  %i618 = extractelement <4 x float> %i462, i64 0, !dbg !47
  %i619 = tail call float @llvm.amdgcn.exp2.f32(float %i618), !dbg !47
  %local.reduce62 = fadd float %i619, %k0.sum.anchor4, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor12 = tail call float asm sideeffect "", "=v,0"(float %local.reduce62) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i620 = insertelement <4 x float> poison, float %i619, i64 0, !dbg !48
  %i621 = extractelement <4 x float> %i462, i64 1, !dbg !47
  %i622 = tail call float @llvm.amdgcn.exp2.f32(float %i621), !dbg !47
  %local.reduce63 = fadd float %i622, %k0.sum.anchor5, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor13 = tail call float asm sideeffect "", "=v,0"(float %local.reduce63) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i623 = insertelement <4 x float> %i620, float %i622, i64 1, !dbg !48
  %i624 = extractelement <4 x float> %i462, i64 2, !dbg !47
  %i625 = tail call float @llvm.amdgcn.exp2.f32(float %i624), !dbg !47
  %local.reduce64 = fadd float %i625, %k0.sum.anchor6, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor14 = tail call float asm sideeffect "", "=v,0"(float %local.reduce64) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i626 = insertelement <4 x float> %i623, float %i625, i64 2, !dbg !48
  %i627 = extractelement <4 x float> %i462, i64 3, !dbg !47
  %i628 = tail call float @llvm.amdgcn.exp2.f32(float %i627), !dbg !47
  %local.reduce65 = fadd float %i628, %k0.sum.anchor7, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor15 = tail call float asm sideeffect "", "=v,0"(float %local.reduce65) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i629 = insertelement <4 x float> %i626, float %i628, i64 3, !dbg !48
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i629) #8, !dbg !48, !frisk.register_lifetimes !5
  %i630 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !48
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i630) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i739, <4 x float> %i737) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i744 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i741, <4 x float> %i738) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i745 = shufflevector <8 x half> %i709, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i454 = fsub <4 x float> %i329, %i450, !dbg !40
  %i523 = extractelement <4 x float> %i454, i64 0, !dbg !47
  %i524 = tail call float @llvm.amdgcn.exp2.f32(float %i523), !dbg !47
  %local.reduce97 = fadd float %i524, %k0.sum.anchor8, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor16 = tail call float asm sideeffect "", "=v,0"(float %local.reduce97) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i525 = insertelement <4 x float> poison, float %i524, i64 0, !dbg !48
  %i526 = extractelement <4 x float> %i454, i64 1, !dbg !47
  %i527 = tail call float @llvm.amdgcn.exp2.f32(float %i526), !dbg !47
  %local.reduce98 = fadd float %i527, %k0.sum.anchor9, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor17 = tail call float asm sideeffect "", "=v,0"(float %local.reduce98) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i528 = insertelement <4 x float> %i525, float %i527, i64 1, !dbg !48
  %i529 = extractelement <4 x float> %i454, i64 2, !dbg !47
  %i530 = tail call float @llvm.amdgcn.exp2.f32(float %i529), !dbg !47
  %local.reduce99 = fadd float %i530, %k0.sum.anchor10, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor18 = tail call float asm sideeffect "", "=v,0"(float %local.reduce99) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i531 = insertelement <4 x float> %i528, float %i530, i64 2, !dbg !48
  %i532 = extractelement <4 x float> %i454, i64 3, !dbg !47
  %i533 = tail call float @llvm.amdgcn.exp2.f32(float %i532), !dbg !47
  %local.reduce100 = fadd float %i533, %k0.sum.anchor11, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor19 = tail call float asm sideeffect "", "=v,0"(float %local.reduce100) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i534 = insertelement <4 x float> %i531, float %i533, i64 3, !dbg !48
  %math.result378 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i534) #8, !dbg !48, !frisk.register_lifetimes !5
  %i535 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !48
  %math.packed379 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i535) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i746 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i745, <4 x float> %i740) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i747 = shufflevector <8 x half> %i709, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i748 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i747, <4 x float> %i742) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i463 = fsub <4 x float> %i385, %i459, !dbg !40
  %i631 = extractelement <4 x float> %i463, i64 0, !dbg !47
  %i632 = tail call float @llvm.amdgcn.exp2.f32(float %i631), !dbg !47
  %local.reduce66 = fadd float %i632, %k0.sum.anchor12, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor20 = tail call float asm sideeffect "", "=v,0"(float %local.reduce66) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i633 = insertelement <4 x float> poison, float %i632, i64 0, !dbg !48
  %i634 = extractelement <4 x float> %i463, i64 1, !dbg !47
  %i635 = tail call float @llvm.amdgcn.exp2.f32(float %i634), !dbg !47
  %local.reduce67 = fadd float %i635, %k0.sum.anchor13, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor21 = tail call float asm sideeffect "", "=v,0"(float %local.reduce67) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i636 = insertelement <4 x float> %i633, float %i635, i64 1, !dbg !48
  %i637 = extractelement <4 x float> %i463, i64 2, !dbg !47
  %i638 = tail call float @llvm.amdgcn.exp2.f32(float %i637), !dbg !47
  %local.reduce68 = fadd float %i638, %k0.sum.anchor14, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor22 = tail call float asm sideeffect "", "=v,0"(float %local.reduce68) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i639 = insertelement <4 x float> %i636, float %i638, i64 2, !dbg !48
  %i640 = extractelement <4 x float> %i463, i64 3, !dbg !47
  %i641 = tail call float @llvm.amdgcn.exp2.f32(float %i640), !dbg !47
  %local.reduce69 = fadd float %i641, %k0.sum.anchor15, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor23 = tail call float asm sideeffect "", "=v,0"(float %local.reduce69) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i642 = insertelement <4 x float> %i639, float %i641, i64 3, !dbg !48
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i642) #8, !dbg !48, !frisk.register_lifetimes !5
  %i643 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !48
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i643) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i749 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i745, <4 x float> %i743) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i750 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i747, <4 x float> %i744) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i751 = shufflevector <8 x half> %i711, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i455 = fsub <4 x float> %i337, %i450, !dbg !40
  %i536 = extractelement <4 x float> %i455, i64 0, !dbg !47
  %i537 = tail call float @llvm.amdgcn.exp2.f32(float %i536), !dbg !47
  %local.reduce101 = fadd float %i537, %k0.sum.anchor16, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor24 = tail call float asm sideeffect "", "=v,0"(float %local.reduce101) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i538 = insertelement <4 x float> poison, float %i537, i64 0, !dbg !48
  %i539 = extractelement <4 x float> %i455, i64 1, !dbg !47
  %i540 = tail call float @llvm.amdgcn.exp2.f32(float %i539), !dbg !47
  %local.reduce102 = fadd float %i540, %k0.sum.anchor17, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor25 = tail call float asm sideeffect "", "=v,0"(float %local.reduce102) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i541 = insertelement <4 x float> %i538, float %i540, i64 1, !dbg !48
  %i542 = extractelement <4 x float> %i455, i64 2, !dbg !47
  %i543 = tail call float @llvm.amdgcn.exp2.f32(float %i542), !dbg !47
  %local.reduce103 = fadd float %i543, %k0.sum.anchor18, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor26 = tail call float asm sideeffect "", "=v,0"(float %local.reduce103) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i544 = insertelement <4 x float> %i541, float %i543, i64 2, !dbg !48
  %i545 = extractelement <4 x float> %i455, i64 3, !dbg !47
  %i546 = tail call float @llvm.amdgcn.exp2.f32(float %i545), !dbg !47
  %local.reduce104 = fadd float %i546, %k0.sum.anchor19, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor27 = tail call float asm sideeffect "", "=v,0"(float %local.reduce104) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i547 = insertelement <4 x float> %i544, float %i546, i64 3, !dbg !48
  %math.result380 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i547) #8, !dbg !48, !frisk.register_lifetimes !5
  %i548 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !48
  %math.packed381 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i548) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i752 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i751, <4 x float> %i746) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i753 = shufflevector <8 x half> %i711, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i754 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i753, <4 x float> %i748) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i464 = fsub <4 x float> %i391, %i459, !dbg !40
  %i644 = extractelement <4 x float> %i464, i64 0, !dbg !47
  %i645 = tail call float @llvm.amdgcn.exp2.f32(float %i644), !dbg !47
  %local.reduce70 = fadd float %i645, %k0.sum.anchor20, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor28 = tail call float asm sideeffect "", "=v,0"(float %local.reduce70) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i646 = insertelement <4 x float> poison, float %i645, i64 0, !dbg !48
  %i647 = extractelement <4 x float> %i464, i64 1, !dbg !47
  %i648 = tail call float @llvm.amdgcn.exp2.f32(float %i647), !dbg !47
  %local.reduce71 = fadd float %i648, %k0.sum.anchor21, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor29 = tail call float asm sideeffect "", "=v,0"(float %local.reduce71) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i649 = insertelement <4 x float> %i646, float %i648, i64 1, !dbg !48
  %i650 = extractelement <4 x float> %i464, i64 2, !dbg !47
  %i651 = tail call float @llvm.amdgcn.exp2.f32(float %i650), !dbg !47
  %local.reduce72 = fadd float %i651, %k0.sum.anchor22, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor30 = tail call float asm sideeffect "", "=v,0"(float %local.reduce72) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i652 = insertelement <4 x float> %i649, float %i651, i64 2, !dbg !48
  %i653 = extractelement <4 x float> %i464, i64 3, !dbg !47
  %i654 = tail call float @llvm.amdgcn.exp2.f32(float %i653), !dbg !47
  %local.reduce73 = fadd float %i654, %k0.sum.anchor23, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor31 = tail call float asm sideeffect "", "=v,0"(float %local.reduce73) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i655 = insertelement <4 x float> %i652, float %i654, i64 3, !dbg !48
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i655) #8, !dbg !48, !frisk.register_lifetimes !5
  %i656 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !48
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i656) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i755 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i751, <4 x float> %i749) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i756 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i753, <4 x float> %i750) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i757 = shufflevector <8 x half> %i713, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i456 = fsub <4 x float> %i345, %i450, !dbg !40
  %i549 = extractelement <4 x float> %i456, i64 0, !dbg !47
  %i550 = tail call float @llvm.amdgcn.exp2.f32(float %i549), !dbg !47
  %local.reduce105 = fadd float %i550, %k0.sum.anchor24, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor32 = tail call float asm sideeffect "", "=v,0"(float %local.reduce105) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i551 = insertelement <4 x float> poison, float %i550, i64 0, !dbg !48
  %i552 = extractelement <4 x float> %i456, i64 1, !dbg !47
  %i553 = tail call float @llvm.amdgcn.exp2.f32(float %i552), !dbg !47
  %local.reduce106 = fadd float %i553, %k0.sum.anchor25, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor33 = tail call float asm sideeffect "", "=v,0"(float %local.reduce106) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i554 = insertelement <4 x float> %i551, float %i553, i64 1, !dbg !48
  %i555 = extractelement <4 x float> %i456, i64 2, !dbg !47
  %i556 = tail call float @llvm.amdgcn.exp2.f32(float %i555), !dbg !47
  %local.reduce107 = fadd float %i556, %k0.sum.anchor26, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor34 = tail call float asm sideeffect "", "=v,0"(float %local.reduce107) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i557 = insertelement <4 x float> %i554, float %i556, i64 2, !dbg !48
  %i558 = extractelement <4 x float> %i456, i64 3, !dbg !47
  %i559 = tail call float @llvm.amdgcn.exp2.f32(float %i558), !dbg !47
  %local.reduce108 = fadd float %i559, %k0.sum.anchor27, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor35 = tail call float asm sideeffect "", "=v,0"(float %local.reduce108) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i560 = insertelement <4 x float> %i557, float %i559, i64 3, !dbg !48
  %math.result382 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i560) #8, !dbg !48, !frisk.register_lifetimes !5
  %i561 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !48
  %math.packed383 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i561) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i758 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i757, <4 x float> %i752) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i759 = shufflevector <8 x half> %i713, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i760 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i759, <4 x float> %i754) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i465 = fsub <4 x float> %i397, %i459, !dbg !40
  %i657 = extractelement <4 x float> %i465, i64 0, !dbg !47
  %i658 = tail call float @llvm.amdgcn.exp2.f32(float %i657), !dbg !47
  %local.reduce74 = fadd float %i658, %k0.sum.anchor28, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor36 = tail call float asm sideeffect "", "=v,0"(float %local.reduce74) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i659 = insertelement <4 x float> poison, float %i658, i64 0, !dbg !48
  %i660 = extractelement <4 x float> %i465, i64 1, !dbg !47
  %i661 = tail call float @llvm.amdgcn.exp2.f32(float %i660), !dbg !47
  %local.reduce75 = fadd float %i661, %k0.sum.anchor29, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor37 = tail call float asm sideeffect "", "=v,0"(float %local.reduce75) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i662 = insertelement <4 x float> %i659, float %i661, i64 1, !dbg !48
  %i663 = extractelement <4 x float> %i465, i64 2, !dbg !47
  %i664 = tail call float @llvm.amdgcn.exp2.f32(float %i663), !dbg !47
  %local.reduce76 = fadd float %i664, %k0.sum.anchor30, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor38 = tail call float asm sideeffect "", "=v,0"(float %local.reduce76) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i665 = insertelement <4 x float> %i662, float %i664, i64 2, !dbg !48
  %i666 = extractelement <4 x float> %i465, i64 3, !dbg !47
  %i667 = tail call float @llvm.amdgcn.exp2.f32(float %i666), !dbg !47
  %local.reduce77 = fadd float %i667, %k0.sum.anchor31, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor39 = tail call float asm sideeffect "", "=v,0"(float %local.reduce77) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i668 = insertelement <4 x float> %i665, float %i667, i64 3, !dbg !48
  %math.result398 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i668) #8, !dbg !48, !frisk.register_lifetimes !5
  %i669 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !48
  %math.packed399 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i669) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i761 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i757, <4 x float> %i755) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i762 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i759, <4 x float> %i756) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i763 = shufflevector <8 x half> %i715, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i457 = fsub <4 x float> %i353, %i450, !dbg !40
  %i562 = extractelement <4 x float> %i457, i64 0, !dbg !47
  %i563 = tail call float @llvm.amdgcn.exp2.f32(float %i562), !dbg !47
  %local.reduce109 = fadd float %i563, %k0.sum.anchor32, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor40 = tail call float asm sideeffect "", "=v,0"(float %local.reduce109) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i564 = insertelement <4 x float> poison, float %i563, i64 0, !dbg !48
  %i565 = extractelement <4 x float> %i457, i64 1, !dbg !47
  %i566 = tail call float @llvm.amdgcn.exp2.f32(float %i565), !dbg !47
  %local.reduce110 = fadd float %i566, %k0.sum.anchor33, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor41 = tail call float asm sideeffect "", "=v,0"(float %local.reduce110) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i567 = insertelement <4 x float> %i564, float %i566, i64 1, !dbg !48
  %i568 = extractelement <4 x float> %i457, i64 2, !dbg !47
  %i569 = tail call float @llvm.amdgcn.exp2.f32(float %i568), !dbg !47
  %local.reduce111 = fadd float %i569, %k0.sum.anchor34, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor42 = tail call float asm sideeffect "", "=v,0"(float %local.reduce111) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i570 = insertelement <4 x float> %i567, float %i569, i64 2, !dbg !48
  %i571 = extractelement <4 x float> %i457, i64 3, !dbg !47
  %i572 = tail call float @llvm.amdgcn.exp2.f32(float %i571), !dbg !47
  %local.reduce112 = fadd float %i572, %k0.sum.anchor35, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor43 = tail call float asm sideeffect "", "=v,0"(float %local.reduce112) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i573 = insertelement <4 x float> %i570, float %i572, i64 3, !dbg !48
  %math.result384 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i573) #8, !dbg !48, !frisk.register_lifetimes !5
  %i574 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !48
  %math.packed385 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i574) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i764 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i763, <4 x float> %i758) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i765 = shufflevector <8 x half> %i715, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i766 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i765, <4 x float> %i760) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i466 = fsub <4 x float> %i403, %i459, !dbg !40
  %i670 = extractelement <4 x float> %i466, i64 0, !dbg !47
  %i671 = tail call float @llvm.amdgcn.exp2.f32(float %i670), !dbg !47
  %local.reduce78 = fadd float %i671, %k0.sum.anchor36, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor44 = tail call float asm sideeffect "", "=v,0"(float %local.reduce78) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i672 = insertelement <4 x float> poison, float %i671, i64 0, !dbg !48
  %i673 = extractelement <4 x float> %i466, i64 1, !dbg !47
  %i674 = tail call float @llvm.amdgcn.exp2.f32(float %i673), !dbg !47
  %local.reduce79 = fadd float %i674, %k0.sum.anchor37, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor45 = tail call float asm sideeffect "", "=v,0"(float %local.reduce79) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i675 = insertelement <4 x float> %i672, float %i674, i64 1, !dbg !48
  %i676 = extractelement <4 x float> %i466, i64 2, !dbg !47
  %i677 = tail call float @llvm.amdgcn.exp2.f32(float %i676), !dbg !47
  %local.reduce80 = fadd float %i677, %k0.sum.anchor38, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor46 = tail call float asm sideeffect "", "=v,0"(float %local.reduce80) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i678 = insertelement <4 x float> %i675, float %i677, i64 2, !dbg !48
  %i679 = extractelement <4 x float> %i466, i64 3, !dbg !47
  %i680 = tail call float @llvm.amdgcn.exp2.f32(float %i679), !dbg !47
  %local.reduce81 = fadd float %i680, %k0.sum.anchor39, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor47 = tail call float asm sideeffect "", "=v,0"(float %local.reduce81) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i681 = insertelement <4 x float> %i678, float %i680, i64 3, !dbg !48
  %math.result400 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i681) #8, !dbg !48, !frisk.register_lifetimes !5
  %i682 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !48
  %math.packed401 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i682) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i767 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i763, <4 x float> %i761) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i768 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i765, <4 x float> %i762) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i769 = bitcast <4 x i32> %i716 to <8 x half>, !dbg !45
  %i770 = shufflevector <8 x half> %i769, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i458 = fsub <4 x float> %i360, %i450, !dbg !40
  %i575 = extractelement <4 x float> %i458, i64 0, !dbg !47
  %i576 = tail call float @llvm.amdgcn.exp2.f32(float %i575), !dbg !47
  %local.reduce113 = fadd float %i576, %k0.sum.anchor40, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor48 = tail call float asm sideeffect "", "=v,0"(float %local.reduce113) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i577 = insertelement <4 x float> poison, float %i576, i64 0, !dbg !48
  %i578 = extractelement <4 x float> %i458, i64 1, !dbg !47
  %i579 = tail call float @llvm.amdgcn.exp2.f32(float %i578), !dbg !47
  %local.reduce114 = fadd float %i579, %k0.sum.anchor41, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor49 = tail call float asm sideeffect "", "=v,0"(float %local.reduce114) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce117 = fadd float %k0.sum.anchor48, %k0.sum.anchor49, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor50 = tail call float asm sideeffect "", "=v,0"(float %local.reduce117) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i580 = insertelement <4 x float> %i577, float %i579, i64 1, !dbg !48
  %i581 = extractelement <4 x float> %i458, i64 2, !dbg !47
  %i582 = tail call float @llvm.amdgcn.exp2.f32(float %i581), !dbg !47
  %local.reduce115 = fadd float %i582, %k0.sum.anchor42, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor51 = tail call float asm sideeffect "", "=v,0"(float %local.reduce115) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i583 = insertelement <4 x float> %i580, float %i582, i64 2, !dbg !48
  %i584 = extractelement <4 x float> %i458, i64 3, !dbg !47
  %i585 = tail call float @llvm.amdgcn.exp2.f32(float %i584), !dbg !47
  %local.reduce116 = fadd float %i585, %k0.sum.anchor43, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor52 = tail call float asm sideeffect "", "=v,0"(float %local.reduce116) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce118 = fadd float %k0.sum.anchor51, %k0.sum.anchor52, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor53 = tail call float asm sideeffect "", "=v,0"(float %local.reduce118) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i588 = fadd float %k0.sum.anchor50, %k0.sum.anchor53, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor54 = tail call float asm sideeffect "", "=v,0"(float %i588) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i589 = fadd float %i475, %k0.sum.anchor54, !dbg !49
  %k0.sum.anchor55 = tail call float asm sideeffect "", "=v,0"(float %i589) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i590 = insertelement <1 x float> poison, float %k0.sum.anchor55, i64 0, !dbg !11
  %i591 = insertvalue [2 x <1 x float>] poison, <1 x float> %i590, 0, !dbg !11
  %i586 = insertelement <4 x float> %i583, float %i585, i64 3, !dbg !48
  %math.result386 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i586) #8, !dbg !48, !frisk.register_lifetimes !5
  %i587 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !48
  %math.packed387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i587) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i771 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i770, <4 x float> %i764) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i772 = shufflevector <8 x half> %i769, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i773 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i772, <4 x float> %i766) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i467 = fsub <4 x float> %i409, %i459, !dbg !40
  %i683 = extractelement <4 x float> %i467, i64 0, !dbg !47
  %i684 = tail call float @llvm.amdgcn.exp2.f32(float %i683), !dbg !47
  %local.reduce82 = fadd float %i684, %k0.sum.anchor44, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor56 = tail call float asm sideeffect "", "=v,0"(float %local.reduce82) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i685 = insertelement <4 x float> poison, float %i684, i64 0, !dbg !48
  %i686 = extractelement <4 x float> %i467, i64 1, !dbg !47
  %i687 = tail call float @llvm.amdgcn.exp2.f32(float %i686), !dbg !47
  %local.reduce83 = fadd float %i687, %k0.sum.anchor45, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor57 = tail call float asm sideeffect "", "=v,0"(float %local.reduce83) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce86 = fadd float %k0.sum.anchor56, %k0.sum.anchor57, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor58 = tail call float asm sideeffect "", "=v,0"(float %local.reduce86) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i688 = insertelement <4 x float> %i685, float %i687, i64 1, !dbg !48
  %i689 = extractelement <4 x float> %i467, i64 2, !dbg !47
  %i690 = tail call float @llvm.amdgcn.exp2.f32(float %i689), !dbg !47
  %local.reduce84 = fadd float %i690, %k0.sum.anchor46, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor59 = tail call float asm sideeffect "", "=v,0"(float %local.reduce84) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i691 = insertelement <4 x float> %i688, float %i690, i64 2, !dbg !48
  %i692 = extractelement <4 x float> %i467, i64 3, !dbg !47
  %i693 = tail call float @llvm.amdgcn.exp2.f32(float %i692), !dbg !47
  %local.reduce85 = fadd float %i693, %k0.sum.anchor47, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor60 = tail call float asm sideeffect "", "=v,0"(float %local.reduce85) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce87 = fadd float %k0.sum.anchor59, %k0.sum.anchor60, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor61 = tail call float asm sideeffect "", "=v,0"(float %local.reduce87) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i696 = fadd float %k0.sum.anchor58, %k0.sum.anchor61, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor62 = tail call float asm sideeffect "", "=v,0"(float %i696) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i697 = fadd float %i483, %k0.sum.anchor62, !dbg !49
  %k0.sum.anchor63 = tail call float asm sideeffect "", "=v,0"(float %i697) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %i698 = insertelement <1 x float> poison, float %k0.sum.anchor63, i64 0, !dbg !11
  %i699 = insertvalue [2 x <1 x float>] %i591, <1 x float> %i698, 1, !dbg !11
  %i694 = insertelement <4 x float> %i691, float %i693, i64 3, !dbg !48
  %math.result402 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i694) #8, !dbg !48, !frisk.register_lifetimes !5
  %i695 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !48
  %math.packed403 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i695) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %i774 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i770, <4 x float> %i767) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i775 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i772, <4 x float> %i768) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1462 = or disjoint i64 %address.local921, %i189, !dbg !37
  %vdb.address1463 = trunc nuw nsw i64 %vdb.address1462 to i32, !dbg !37
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !37
  %vdb.address1471 = or disjoint i64 %address.local927, %i189, !dbg !37
  %vdb.address1472 = trunc nuw nsw i64 %vdb.address1471 to i32, !dbg !37
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !37
  %i776 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %i777 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %vdb.address1473, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !45
  %i778 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i779 = bitcast <4 x i32> %i778 to <8 x half>, !dbg !45
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8448, !dbg !45
  %i780 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i781 = bitcast <4 x i32> %i780 to <8 x half>, !dbg !45
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8704, !dbg !45
  %i782 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i783 = bitcast <4 x i32> %i782 to <8 x half>, !dbg !45
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8960, !dbg !45
  %i784 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i785 = bitcast <4 x i32> %i784 to <8 x half>, !dbg !45
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local935, i64 12288, !dbg !45
  %i786 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i787 = bitcast <4 x i32> %i786 to <8 x half>, !dbg !45
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12544, !dbg !45
  %i788 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i789 = bitcast <4 x i32> %i788 to <8 x half>, !dbg !45
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12800, !dbg !45
  %i790 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i791 = bitcast <4 x i32> %i790 to <8 x half>, !dbg !45
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local944, i64 13056, !dbg !45
  %i792 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i793 = shufflevector <8 x half> %i779, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %i794 = extractvalue [2 x <8 x float>] %i9114, 0, !dbg !43
  %i795 = shufflevector <8 x float> %i794, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %i796 = fmul <4 x float> %i795, %i472, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i797 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i793, <4 x float> %i796) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i798 = shufflevector <8 x half> %i779, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %i799 = shufflevector <8 x float> %i794, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %i800 = fmul <4 x float> %i799, %i472, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i801 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i798, <4 x float> %i800) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %i802 = extractvalue [2 x <8 x float>] %i9114, 1, !dbg !43
  %i803 = shufflevector <8 x float> %i802, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %i804 = fmul <4 x float> %i803, %i480, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i805 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i793, <4 x float> %i804) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %i806 = shufflevector <8 x float> %i802, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %i807 = fmul <4 x float> %i806, %i480, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i808 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i798, <4 x float> %i807) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i809 = shufflevector <8 x half> %i781, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i810 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i809, <4 x float> %i797) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i811 = shufflevector <8 x half> %i781, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i812 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i811, <4 x float> %i801) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i809, <4 x float> %i805) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i814 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i811, <4 x float> %i808) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i815 = shufflevector <8 x half> %i783, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i815, <4 x float> %i810) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i817 = shufflevector <8 x half> %i783, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i818 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i817, <4 x float> %i812) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i819 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i815, <4 x float> %i813) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i820 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i817, <4 x float> %i814) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i821 = shufflevector <8 x half> %i785, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i822 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i821, <4 x float> %i816) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i823 = shufflevector <8 x half> %i785, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i824 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i823, <4 x float> %i818) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i825 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i821, <4 x float> %i819) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i826 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i823, <4 x float> %i820) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i827 = shufflevector <8 x half> %i787, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i828 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i827, <4 x float> %i822) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i829 = shufflevector <8 x half> %i787, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i829, <4 x float> %i824) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i831 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i827, <4 x float> %i825) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i829, <4 x float> %i826) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i833 = shufflevector <8 x half> %i789, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i833, <4 x float> %i828) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i835 = shufflevector <8 x half> %i789, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i835, <4 x float> %i830) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i837 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i833, <4 x float> %i831) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i835, <4 x float> %i832) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i839 = shufflevector <8 x half> %i791, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i839, <4 x float> %i834) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i841 = shufflevector <8 x half> %i791, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i841, <4 x float> %i836) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i843 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i839, <4 x float> %i837) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i841, <4 x float> %i838) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i845 = bitcast <4 x i32> %i792 to <8 x half>, !dbg !45
  %i846 = shufflevector <8 x half> %i845, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i846, <4 x float> %i840) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i848 = shufflevector <8 x half> %i845, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i849 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i848, <4 x float> %i842) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i846, <4 x float> %i843) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i848, <4 x float> %i844) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1480 = or disjoint i64 %address.local921, %i242, !dbg !37
  %vdb.address1481 = trunc nuw nsw i64 %vdb.address1480 to i32, !dbg !37
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !37
  %vdb.address1489 = or disjoint i64 %address.local927, %i242, !dbg !37
  %vdb.address1490 = trunc nuw nsw i64 %vdb.address1489 to i32, !dbg !37
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !37
  %i852 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1482, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %i853 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1491, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %i854 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i855 = bitcast <4 x i32> %i854 to <8 x half>, !dbg !45
  %i856 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i857 = bitcast <4 x i32> %i856 to <8 x half>, !dbg !45
  %i858 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i859 = bitcast <4 x i32> %i858 to <8 x half>, !dbg !45
  %i860 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i861 = bitcast <4 x i32> %i860 to <8 x half>, !dbg !45
  %i862 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i863 = bitcast <4 x i32> %i862 to <8 x half>, !dbg !45
  %i864 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i865 = bitcast <4 x i32> %i864 to <8 x half>, !dbg !45
  %i866 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i867 = bitcast <4 x i32> %i866 to <8 x half>, !dbg !45
  %i868 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i869 = shufflevector <8 x half> %i855, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %i870 = extractvalue [2 x <8 x float>] %i9215, 0, !dbg !43
  %i871 = shufflevector <8 x float> %i870, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %i872 = fmul <4 x float> %i871, %i472, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i869, <4 x float> %i872) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i874 = shufflevector <8 x half> %i855, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %i875 = shufflevector <8 x float> %i870, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %i876 = fmul <4 x float> %i875, %i472, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i874, <4 x float> %i876) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %i878 = extractvalue [2 x <8 x float>] %i9215, 1, !dbg !43
  %i879 = shufflevector <8 x float> %i878, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %i880 = fmul <4 x float> %i879, %i480, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i869, <4 x float> %i880) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %i882 = shufflevector <8 x float> %i878, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %i883 = fmul <4 x float> %i882, %i480, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i874, <4 x float> %i883) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i885 = shufflevector <8 x half> %i857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i885, <4 x float> %i873) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i887 = shufflevector <8 x half> %i857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i888 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i887, <4 x float> %i877) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i889 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i885, <4 x float> %i881) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i890 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i887, <4 x float> %i884) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i891 = shufflevector <8 x half> %i859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i892 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i891, <4 x float> %i886) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i893 = shufflevector <8 x half> %i859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i893, <4 x float> %i888) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i891, <4 x float> %i889) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i893, <4 x float> %i890) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i897 = shufflevector <8 x half> %i861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i898 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i897, <4 x float> %i892) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i899 = shufflevector <8 x half> %i861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i899, <4 x float> %i894) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i901 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i897, <4 x float> %i895) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i899, <4 x float> %i896) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i903 = shufflevector <8 x half> %i863, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i904 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i903, <4 x float> %i898) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i905 = shufflevector <8 x half> %i863, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i905, <4 x float> %i900) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i907 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i903, <4 x float> %i901) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i908 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i905, <4 x float> %i902) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i909 = shufflevector <8 x half> %i865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i910 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i909, <4 x float> %i904) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i911 = shufflevector <8 x half> %i865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i911, <4 x float> %i906) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i909, <4 x float> %i907) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i911, <4 x float> %i908) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i915 = shufflevector <8 x half> %i867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i915, <4 x float> %i910) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i917 = shufflevector <8 x half> %i867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i917, <4 x float> %i912) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i919 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i915, <4 x float> %i913) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i917, <4 x float> %i914) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i921 = bitcast <4 x i32> %i868 to <8 x half>, !dbg !45
  %i922 = shufflevector <8 x half> %i921, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i923 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i922, <4 x float> %i916) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i924 = shufflevector <8 x half> %i921, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i925 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i924, <4 x float> %i918) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i922, <4 x float> %i919) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i927 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i924, <4 x float> %i920) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %i1020 = add nuw nsw i64 %i8710, 1, !dbg !11
  %k0.has.next.not = icmp eq i64 %i8710, 31, !dbg !45
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !45

k0.issue:                                         ; preds = %bb95
  %k0.next.i98 = shl nuw nsw i64 %i1020, 14, !dbg !13
  %k0.next.i99 = add nuw nsw i64 %k0.next.i98, %k0.prime.i97, !dbg !13
  %k0.next.kdma.source1516 = or disjoint i64 %k0.next.i99, %k0.prime.kdma.source1515, !dbg !13
  %k0.next.kdma.source1517 = trunc nuw nsw i64 %k0.next.kdma.source1516 to i32, !dbg !13
  %k0.next.kdma.source1518 = shl nuw nsw i32 %k0.next.kdma.source1517, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1518, %kdma.intra, !dbg !13
  %k0.next.i104 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1530 = or disjoint i64 %k0.next.i99, %k0.prime.kdma.source1529, !dbg !13
  %k0.next.kdma.source1531 = trunc nuw nsw i64 %k0.next.kdma.source1530 to i32, !dbg !13
  %k0.next.kdma.source1532 = shl nuw nsw i32 %k0.next.kdma.source1531, 1, !dbg !13
  %k0.next.kdma.offset1533 = or disjoint i32 %k0.next.kdma.source1532, %kdma.intra.hi, !dbg !13
  %k0.next.i105 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %k0.next.kdma.offset1533, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1547 = or disjoint i64 %k0.next.i99, %k0.prime.kdma.source1546, !dbg !13
  %k0.next.kdma.source1548 = trunc nuw nsw i64 %k0.next.kdma.source1547 to i32, !dbg !13
  %k0.next.kdma.source1549 = shl nuw nsw i32 %k0.next.kdma.source1548, 1, !dbg !13
  %k0.next.kdma.offset1550 = or disjoint i32 %k0.next.kdma.source1549, %kdma.intra, !dbg !13
  %k0.next.i107 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %k0.next.kdma.offset1550, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1564 = or disjoint i64 %k0.next.i99, %k0.prime.kdma.source1563, !dbg !13
  %k0.next.kdma.source1565 = trunc nuw nsw i64 %k0.next.kdma.source1564 to i32, !dbg !13
  %k0.next.kdma.source1566 = shl nuw nsw i32 %k0.next.kdma.source1565, 1, !dbg !13
  %k0.next.kdma.offset1567 = or disjoint i32 %k0.next.kdma.source1566, %kdma.intra.hi, !dbg !13
  %k0.next.i108 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %k0.next.kdma.offset1567, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %bb95
  %i928 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i929 = bitcast <4 x i32> %i928 to <8 x half>, !dbg !45
  %i930 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i931 = bitcast <4 x i32> %i930 to <8 x half>, !dbg !45
  %i932 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i933 = bitcast <4 x i32> %i932 to <8 x half>, !dbg !45
  %i934 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i935 = bitcast <4 x i32> %i934 to <8 x half>, !dbg !45
  %i936 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i937 = bitcast <4 x i32> %i936 to <8 x half>, !dbg !45
  %i938 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i939 = bitcast <4 x i32> %i938 to <8 x half>, !dbg !45
  %i940 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i941 = bitcast <4 x i32> %i940 to <8 x half>, !dbg !45
  %i942 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %i943 = shufflevector <8 x half> %i929, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %i944 = extractvalue [2 x <8 x float>] %i9316, 0, !dbg !43
  %i945 = shufflevector <8 x float> %i944, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %i946 = fmul <4 x float> %i945, %i472, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i947 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i943, <4 x float> %i946) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i948 = shufflevector <8 x half> %i929, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %i949 = shufflevector <8 x float> %i944, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %i950 = fmul <4 x float> %i949, %i472, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i951 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i948, <4 x float> %i950) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %i952 = extractvalue [2 x <8 x float>] %i9316, 1, !dbg !43
  %i953 = shufflevector <8 x float> %i952, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %i954 = fmul <4 x float> %i953, %i480, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i943, <4 x float> %i954) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %i956 = shufflevector <8 x float> %i952, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %i957 = fmul <4 x float> %i956, %i480, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i958 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i948, <4 x float> %i957) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i959 = shufflevector <8 x half> %i931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i959, <4 x float> %i947) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i961 = shufflevector <8 x half> %i931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i961, <4 x float> %i951) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i959, <4 x float> %i955) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i961, <4 x float> %i958) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i965 = shufflevector <8 x half> %i933, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i966 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i965, <4 x float> %i960) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i967 = shufflevector <8 x half> %i933, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i967, <4 x float> %i962) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i965, <4 x float> %i963) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i967, <4 x float> %i964) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i971 = shufflevector <8 x half> %i935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i972 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i971, <4 x float> %i966) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i973 = shufflevector <8 x half> %i935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i974 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i973, <4 x float> %i968) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i971, <4 x float> %i969) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i976 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i973, <4 x float> %i970) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i977 = shufflevector <8 x half> %i937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i978 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i977, <4 x float> %i972) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i979 = shufflevector <8 x half> %i937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i979, <4 x float> %i974) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i977, <4 x float> %i975) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i979, <4 x float> %i976) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i983 = shufflevector <8 x half> %i939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i983, <4 x float> %i978) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i985 = shufflevector <8 x half> %i939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i985, <4 x float> %i980) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i983, <4 x float> %i981) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i985, <4 x float> %i982) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i989 = shufflevector <8 x half> %i941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i989, <4 x float> %i984) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i991 = shufflevector <8 x half> %i941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i991, <4 x float> %i986) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i989, <4 x float> %i987) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i991, <4 x float> %i988) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i995 = bitcast <4 x i32> %i942 to <8 x half>, !dbg !45
  %i996 = shufflevector <8 x half> %i995, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i997 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i996, <4 x float> %i990) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i998 = shufflevector <8 x half> %i995, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i999 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i998, <4 x float> %i992) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i996, <4 x float> %i993) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %i1001 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i998, <4 x float> %i994) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !11
  %i1002 = shufflevector <4 x float> %i771, <4 x float> %i773, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %i1003 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1002, 0, !dbg !11
  %i1004 = shufflevector <4 x float> %i774, <4 x float> %i775, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %i1005 = insertvalue [2 x <8 x float>] %i1003, <8 x float> %i1004, 1, !dbg !11
  %i1006 = insertvalue [2 x <1 x float>] poison, <1 x float> %i436, 0, !dbg !11
  %i1007 = insertvalue [2 x <1 x float>] %i1006, <1 x float> %i449, 1, !dbg !11
  %i1008 = shufflevector <4 x float> %i847, <4 x float> %i849, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %i1009 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1008, 0, !dbg !11
  %i1010 = shufflevector <4 x float> %i850, <4 x float> %i851, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %i1011 = insertvalue [2 x <8 x float>] %i1009, <8 x float> %i1010, 1, !dbg !11
  %i1012 = shufflevector <4 x float> %i923, <4 x float> %i925, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %i1013 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1012, 0, !dbg !11
  %i1014 = shufflevector <4 x float> %i926, <4 x float> %i927, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %i1015 = insertvalue [2 x <8 x float>] %i1013, <8 x float> %i1014, 1, !dbg !11
  %i1016 = shufflevector <4 x float> %i997, <4 x float> %i999, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %i1017 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1016, 0, !dbg !11
  %i1018 = shufflevector <4 x float> %i1000, <4 x float> %i1001, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %i1019 = insertvalue [2 x <8 x float>] %i1017, <8 x float> %i1018, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %i94 = icmp samesign ult i64 %i8710, 31, !dbg !11
  br i1 %i94, label %bb95, label %bb1021, !dbg !11

bb1021:                                           ; preds = %k0.join
  %i1022 = shl nuw nsw i32 %i415, 2, !dbg !11
  %i1023 = xor i32 %i1022, 64, !dbg !11
  %i1024 = bitcast float %k0.sum.anchor63 to i32, !dbg !11
  %lane.sum.shuffle188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i1023, i32 %i1024), !dbg !11
  %i1025 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !11
  %lane.sum.final189 = fadd float %k0.sum.anchor63, %i1025, !dbg !11
  %i1026 = xor i32 %i1022, 128, !dbg !11
  %i1027 = bitcast float %lane.sum.final189 to i32, !dbg !11
  %lane.sum.shuffle190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i1026, i32 %i1027), !dbg !11
  %i1028 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !11
  %lane.sum.final191 = fadd float %lane.sum.final189, %i1028, !dbg !11
  %i1029 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !11
  %i1030 = bitcast float %k0.sum.anchor55 to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i1023, i32 %i1030), !dbg !11
  %i1031 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %k0.sum.anchor55, %i1031, !dbg !11
  %i1032 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i1026, i32 %i1032), !dbg !11
  %i1033 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !11
  %lane.sum.final183 = fadd float %lane.sum.final, %i1033, !dbg !11
  %i1034 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !11
  %i1035 = and i32 %kdma.tid, 16, !dbg !11
  %.not = icmp eq i32 %i1035, 0, !dbg !11
  %i1036 = and i32 %kdma.tid, 32, !dbg !11
  %.not36 = icmp eq i32 %i1036, 0, !dbg !11
  %i1038 = extractelement <4 x float> %i997, i64 0, !dbg !11
  %i1039 = extractelement <4 x float> %i997, i64 1, !dbg !11
  %i1040 = extractelement <4 x float> %i997, i64 2, !dbg !11
  %i1041 = extractelement <4 x float> %i997, i64 3, !dbg !11
  %i1042 = select i1 %.not, float %i1039, float %i1038, !dbg !11
  %i1049 = bitcast float %i1042 to i32, !dbg !11
  %i1050 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1049), !dbg !11
  %i1051 = bitcast i32 %i1050 to float, !dbg !11
  %i1052 = select i1 %.not, float %i1038, float %i1051, !dbg !11
  %i1053 = select i1 %.not, float %i1051, float %i1039, !dbg !11
  %i1054 = select i1 %.not, float %i1041, float %i1040, !dbg !11
  %i1055 = bitcast float %i1054 to i32, !dbg !11
  %i1056 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1055), !dbg !11
  %i1057 = bitcast i32 %i1056 to float, !dbg !11
  %i1058 = select i1 %.not, float %i1040, float %i1057, !dbg !11
  %i1059 = select i1 %.not, float %i1057, float %i1041, !dbg !11
  %i1060 = select i1 %.not36, float %i1058, float %i1052, !dbg !11
  %i1065 = bitcast float %i1060 to i32, !dbg !11
  %i1066 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1065), !dbg !11
  %i1067 = bitcast i32 %i1066 to float, !dbg !11
  %i1070 = select i1 %.not36, float %i1059, float %i1053, !dbg !11
  %i1071 = bitcast float %i1070 to i32, !dbg !11
  %i1072 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1071), !dbg !11
  %i1073 = bitcast i32 %i1072 to float, !dbg !11
  %21 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %22 = shufflevector <4 x i1> %21, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %23 = insertelement <4 x float> poison, float %i1052, i64 0, !dbg !11
  %24 = insertelement <4 x float> %23, float %i1053, i64 1, !dbg !11
  %25 = insertelement <4 x float> %24, float %i1067, i64 2, !dbg !11
  %26 = insertelement <4 x float> %25, float %i1073, i64 3, !dbg !11
  %27 = insertelement <4 x float> poison, float %i1067, i64 0, !dbg !11
  %28 = insertelement <4 x float> %27, float %i1073, i64 1, !dbg !11
  %29 = insertelement <4 x float> %28, float %i1058, i64 2, !dbg !11
  %30 = insertelement <4 x float> %29, float %i1059, i64 3, !dbg !11
  %31 = select <4 x i1> %22, <4 x float> %26, <4 x float> %30, !dbg !11
  %i1077 = extractelement <4 x float> %i923, i64 0, !dbg !11
  %i1078 = extractelement <4 x float> %i923, i64 1, !dbg !11
  %i1079 = extractelement <4 x float> %i923, i64 2, !dbg !11
  %i1080 = extractelement <4 x float> %i923, i64 3, !dbg !11
  %i1081 = select i1 %.not, float %i1078, float %i1077, !dbg !11
  %i1082 = bitcast float %i1081 to i32, !dbg !11
  %i1083 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1082), !dbg !11
  %i1084 = bitcast i32 %i1083 to float, !dbg !11
  %i1085 = select i1 %.not, float %i1077, float %i1084, !dbg !11
  %i1086 = select i1 %.not, float %i1084, float %i1078, !dbg !11
  %i1087 = select i1 %.not, float %i1080, float %i1079, !dbg !11
  %i1088 = bitcast float %i1087 to i32, !dbg !11
  %i1089 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1088), !dbg !11
  %i1090 = bitcast i32 %i1089 to float, !dbg !11
  %i1091 = select i1 %.not, float %i1079, float %i1090, !dbg !11
  %i1092 = select i1 %.not, float %i1090, float %i1080, !dbg !11
  %i1093 = select i1 %.not36, float %i1091, float %i1085, !dbg !11
  %i1094 = bitcast float %i1093 to i32, !dbg !11
  %i1095 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1094), !dbg !11
  %i1096 = bitcast i32 %i1095 to float, !dbg !11
  %i1099 = select i1 %.not36, float %i1092, float %i1086, !dbg !11
  %i1100 = bitcast float %i1099 to i32, !dbg !11
  %i1101 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1100), !dbg !11
  %i1102 = bitcast i32 %i1101 to float, !dbg !11
  %32 = insertelement <4 x float> poison, float %i1085, i64 0, !dbg !11
  %33 = insertelement <4 x float> %32, float %i1086, i64 1, !dbg !11
  %34 = insertelement <4 x float> %33, float %i1096, i64 2, !dbg !11
  %35 = insertelement <4 x float> %34, float %i1102, i64 3, !dbg !11
  %36 = insertelement <4 x float> poison, float %i1096, i64 0, !dbg !11
  %37 = insertelement <4 x float> %36, float %i1102, i64 1, !dbg !11
  %38 = insertelement <4 x float> %37, float %i1091, i64 2, !dbg !11
  %39 = insertelement <4 x float> %38, float %i1092, i64 3, !dbg !11
  %40 = select <4 x i1> %22, <4 x float> %35, <4 x float> %39, !dbg !11
  %i1106 = extractelement <4 x float> %i847, i64 0, !dbg !11
  %i1107 = extractelement <4 x float> %i847, i64 1, !dbg !11
  %i1108 = extractelement <4 x float> %i847, i64 2, !dbg !11
  %i1109 = extractelement <4 x float> %i847, i64 3, !dbg !11
  %i1110 = select i1 %.not, float %i1107, float %i1106, !dbg !11
  %i1111 = bitcast float %i1110 to i32, !dbg !11
  %i1112 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1111), !dbg !11
  %i1113 = bitcast i32 %i1112 to float, !dbg !11
  %i1114 = select i1 %.not, float %i1106, float %i1113, !dbg !11
  %i1115 = select i1 %.not, float %i1113, float %i1107, !dbg !11
  %i1116 = select i1 %.not, float %i1109, float %i1108, !dbg !11
  %i1117 = bitcast float %i1116 to i32, !dbg !11
  %i1118 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1117), !dbg !11
  %i1119 = bitcast i32 %i1118 to float, !dbg !11
  %i1120 = select i1 %.not, float %i1108, float %i1119, !dbg !11
  %i1121 = select i1 %.not, float %i1119, float %i1109, !dbg !11
  %i1122 = select i1 %.not36, float %i1120, float %i1114, !dbg !11
  %i1123 = bitcast float %i1122 to i32, !dbg !11
  %i1124 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1123), !dbg !11
  %i1125 = bitcast i32 %i1124 to float, !dbg !11
  %i1128 = select i1 %.not36, float %i1121, float %i1115, !dbg !11
  %i1129 = bitcast float %i1128 to i32, !dbg !11
  %i1130 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1129), !dbg !11
  %i1131 = bitcast i32 %i1130 to float, !dbg !11
  %41 = insertelement <4 x float> poison, float %i1114, i64 0, !dbg !11
  %42 = insertelement <4 x float> %41, float %i1115, i64 1, !dbg !11
  %43 = insertelement <4 x float> %42, float %i1125, i64 2, !dbg !11
  %44 = insertelement <4 x float> %43, float %i1131, i64 3, !dbg !11
  %45 = insertelement <4 x float> poison, float %i1125, i64 0, !dbg !11
  %46 = insertelement <4 x float> %45, float %i1131, i64 1, !dbg !11
  %47 = insertelement <4 x float> %46, float %i1120, i64 2, !dbg !11
  %48 = insertelement <4 x float> %47, float %i1121, i64 3, !dbg !11
  %49 = select <4 x i1> %22, <4 x float> %44, <4 x float> %48, !dbg !11
  %i1135 = extractelement <4 x float> %i771, i64 0, !dbg !11
  %i1136 = extractelement <4 x float> %i771, i64 1, !dbg !11
  %i1137 = extractelement <4 x float> %i771, i64 2, !dbg !11
  %i1138 = extractelement <4 x float> %i771, i64 3, !dbg !11
  %i1139 = select i1 %.not, float %i1136, float %i1135, !dbg !11
  %i1140 = bitcast float %i1139 to i32, !dbg !11
  %i1141 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1140), !dbg !11
  %i1142 = bitcast i32 %i1141 to float, !dbg !11
  %i1143 = select i1 %.not, float %i1135, float %i1142, !dbg !11
  %i1144 = select i1 %.not, float %i1142, float %i1136, !dbg !11
  %i1145 = select i1 %.not, float %i1138, float %i1137, !dbg !11
  %i1146 = bitcast float %i1145 to i32, !dbg !11
  %i1147 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1146), !dbg !11
  %i1148 = bitcast i32 %i1147 to float, !dbg !11
  %i1149 = select i1 %.not, float %i1137, float %i1148, !dbg !11
  %i1150 = select i1 %.not, float %i1148, float %i1138, !dbg !11
  %i1151 = select i1 %.not36, float %i1149, float %i1143, !dbg !11
  %i1152 = bitcast float %i1151 to i32, !dbg !11
  %i1153 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1152), !dbg !11
  %i1154 = select i1 %.not36, float %i1150, float %i1144, !dbg !11
  %i1155 = bitcast float %i1154 to i32, !dbg !11
  %i1156 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1155), !dbg !11
  %i1157 = shufflevector <1 x float> %i1034, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !50
  %i1158 = extractelement <4 x float> %i999, i64 0, !dbg !11
  %i1159 = extractelement <4 x float> %i999, i64 1, !dbg !11
  %i1160 = extractelement <4 x float> %i999, i64 2, !dbg !11
  %i1161 = extractelement <4 x float> %i999, i64 3, !dbg !11
  %i1162 = select i1 %.not, float %i1159, float %i1158, !dbg !11
  %i1163 = bitcast float %i1162 to i32, !dbg !11
  %i1164 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1163), !dbg !11
  %i1165 = bitcast i32 %i1164 to float, !dbg !11
  %i1166 = select i1 %.not, float %i1158, float %i1165, !dbg !11
  %i1167 = select i1 %.not, float %i1165, float %i1159, !dbg !11
  %i1168 = select i1 %.not, float %i1161, float %i1160, !dbg !11
  %i1169 = bitcast float %i1168 to i32, !dbg !11
  %i1170 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1169), !dbg !11
  %i1171 = bitcast i32 %i1170 to float, !dbg !11
  %i1172 = select i1 %.not, float %i1160, float %i1171, !dbg !11
  %i1173 = select i1 %.not, float %i1171, float %i1161, !dbg !11
  %i1174 = select i1 %.not36, float %i1172, float %i1166, !dbg !11
  %i1175 = bitcast float %i1174 to i32, !dbg !11
  %i1176 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1175), !dbg !11
  %i1177 = bitcast i32 %i1176 to float, !dbg !11
  %i1180 = select i1 %.not36, float %i1173, float %i1167, !dbg !11
  %i1181 = bitcast float %i1180 to i32, !dbg !11
  %i1182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1181), !dbg !11
  %i1183 = bitcast i32 %i1182 to float, !dbg !11
  %50 = insertelement <4 x float> poison, float %i1166, i64 0, !dbg !11
  %51 = insertelement <4 x float> %50, float %i1167, i64 1, !dbg !11
  %52 = insertelement <4 x float> %51, float %i1177, i64 2, !dbg !11
  %53 = insertelement <4 x float> %52, float %i1183, i64 3, !dbg !11
  %54 = insertelement <4 x float> poison, float %i1177, i64 0, !dbg !11
  %55 = insertelement <4 x float> %54, float %i1183, i64 1, !dbg !11
  %56 = insertelement <4 x float> %55, float %i1172, i64 2, !dbg !11
  %57 = insertelement <4 x float> %56, float %i1173, i64 3, !dbg !11
  %58 = select <4 x i1> %22, <4 x float> %53, <4 x float> %57, !dbg !11
  %i1186 = extractelement <4 x float> %i925, i64 0, !dbg !11
  %i1187 = extractelement <4 x float> %i925, i64 1, !dbg !11
  %i1188 = extractelement <4 x float> %i925, i64 2, !dbg !11
  %i1189 = extractelement <4 x float> %i925, i64 3, !dbg !11
  %i1190 = select i1 %.not, float %i1187, float %i1186, !dbg !11
  %i1191 = bitcast float %i1190 to i32, !dbg !11
  %i1192 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1191), !dbg !11
  %i1193 = bitcast i32 %i1192 to float, !dbg !11
  %i1194 = select i1 %.not, float %i1186, float %i1193, !dbg !11
  %i1195 = select i1 %.not, float %i1193, float %i1187, !dbg !11
  %i1196 = select i1 %.not, float %i1189, float %i1188, !dbg !11
  %i1197 = bitcast float %i1196 to i32, !dbg !11
  %i1198 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1197), !dbg !11
  %i1199 = bitcast i32 %i1198 to float, !dbg !11
  %i1200 = select i1 %.not, float %i1188, float %i1199, !dbg !11
  %i1201 = select i1 %.not, float %i1199, float %i1189, !dbg !11
  %i1202 = select i1 %.not36, float %i1200, float %i1194, !dbg !11
  %i1203 = bitcast float %i1202 to i32, !dbg !11
  %i1204 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1203), !dbg !11
  %i1205 = bitcast i32 %i1204 to float, !dbg !11
  %i1208 = select i1 %.not36, float %i1201, float %i1195, !dbg !11
  %i1209 = bitcast float %i1208 to i32, !dbg !11
  %i1210 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1209), !dbg !11
  %i1211 = bitcast i32 %i1210 to float, !dbg !11
  %59 = insertelement <4 x float> poison, float %i1194, i64 0, !dbg !11
  %60 = insertelement <4 x float> %59, float %i1195, i64 1, !dbg !11
  %61 = insertelement <4 x float> %60, float %i1205, i64 2, !dbg !11
  %62 = insertelement <4 x float> %61, float %i1211, i64 3, !dbg !11
  %63 = insertelement <4 x float> poison, float %i1205, i64 0, !dbg !11
  %64 = insertelement <4 x float> %63, float %i1211, i64 1, !dbg !11
  %65 = insertelement <4 x float> %64, float %i1200, i64 2, !dbg !11
  %66 = insertelement <4 x float> %65, float %i1201, i64 3, !dbg !11
  %67 = select <4 x i1> %22, <4 x float> %62, <4 x float> %66, !dbg !11
  %i1214 = extractelement <4 x float> %i849, i64 0, !dbg !11
  %i1215 = extractelement <4 x float> %i849, i64 1, !dbg !11
  %i1216 = extractelement <4 x float> %i849, i64 2, !dbg !11
  %i1217 = extractelement <4 x float> %i849, i64 3, !dbg !11
  %i1218 = select i1 %.not, float %i1215, float %i1214, !dbg !11
  %i1219 = bitcast float %i1218 to i32, !dbg !11
  %i1220 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1219), !dbg !11
  %i1221 = bitcast i32 %i1220 to float, !dbg !11
  %i1222 = select i1 %.not, float %i1214, float %i1221, !dbg !11
  %i1223 = select i1 %.not, float %i1221, float %i1215, !dbg !11
  %i1224 = select i1 %.not, float %i1217, float %i1216, !dbg !11
  %i1225 = bitcast float %i1224 to i32, !dbg !11
  %i1226 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1225), !dbg !11
  %i1227 = bitcast i32 %i1226 to float, !dbg !11
  %i1228 = select i1 %.not, float %i1216, float %i1227, !dbg !11
  %i1229 = select i1 %.not, float %i1227, float %i1217, !dbg !11
  %i1230 = select i1 %.not36, float %i1228, float %i1222, !dbg !11
  %i1231 = bitcast float %i1230 to i32, !dbg !11
  %i1232 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1231), !dbg !11
  %i1233 = bitcast i32 %i1232 to float, !dbg !11
  %i1236 = select i1 %.not36, float %i1229, float %i1223, !dbg !11
  %i1237 = bitcast float %i1236 to i32, !dbg !11
  %i1238 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1237), !dbg !11
  %i1239 = bitcast i32 %i1238 to float, !dbg !11
  %68 = insertelement <4 x float> poison, float %i1222, i64 0, !dbg !11
  %69 = insertelement <4 x float> %68, float %i1223, i64 1, !dbg !11
  %70 = insertelement <4 x float> %69, float %i1233, i64 2, !dbg !11
  %71 = insertelement <4 x float> %70, float %i1239, i64 3, !dbg !11
  %72 = insertelement <4 x float> poison, float %i1233, i64 0, !dbg !11
  %73 = insertelement <4 x float> %72, float %i1239, i64 1, !dbg !11
  %74 = insertelement <4 x float> %73, float %i1228, i64 2, !dbg !11
  %75 = insertelement <4 x float> %74, float %i1229, i64 3, !dbg !11
  %76 = select <4 x i1> %22, <4 x float> %71, <4 x float> %75, !dbg !11
  %i1242 = extractelement <4 x float> %i773, i64 0, !dbg !11
  %i1243 = extractelement <4 x float> %i773, i64 1, !dbg !11
  %i1244 = extractelement <4 x float> %i773, i64 2, !dbg !11
  %i1245 = extractelement <4 x float> %i773, i64 3, !dbg !11
  %i1246 = select i1 %.not, float %i1243, float %i1242, !dbg !11
  %i1247 = bitcast float %i1246 to i32, !dbg !11
  %i1248 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1247), !dbg !11
  %i1249 = bitcast i32 %i1248 to float, !dbg !11
  %i1250 = select i1 %.not, float %i1242, float %i1249, !dbg !11
  %i1251 = select i1 %.not, float %i1249, float %i1243, !dbg !11
  %i1252 = select i1 %.not, float %i1245, float %i1244, !dbg !11
  %i1253 = bitcast float %i1252 to i32, !dbg !11
  %i1254 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1253), !dbg !11
  %i1255 = bitcast i32 %i1254 to float, !dbg !11
  %i1256 = select i1 %.not, float %i1244, float %i1255, !dbg !11
  %i1257 = select i1 %.not, float %i1255, float %i1245, !dbg !11
  %i1258 = select i1 %.not36, float %i1256, float %i1250, !dbg !11
  %i1259 = bitcast float %i1258 to i32, !dbg !11
  %i1260 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1259), !dbg !11
  %i1261 = select i1 %.not36, float %i1257, float %i1251, !dbg !11
  %i1262 = bitcast float %i1261 to i32, !dbg !11
  %i1263 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1262), !dbg !11
  %i1265 = extractelement <4 x float> %i1000, i64 0, !dbg !11
  %i1266 = extractelement <4 x float> %i1000, i64 1, !dbg !11
  %i1267 = extractelement <4 x float> %i1000, i64 2, !dbg !11
  %i1268 = extractelement <4 x float> %i1000, i64 3, !dbg !11
  %i1269 = select i1 %.not, float %i1266, float %i1265, !dbg !11
  %i1270 = bitcast float %i1269 to i32, !dbg !11
  %i1271 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1270), !dbg !11
  %i1272 = bitcast i32 %i1271 to float, !dbg !11
  %i1273 = select i1 %.not, float %i1265, float %i1272, !dbg !11
  %i1274 = select i1 %.not, float %i1272, float %i1266, !dbg !11
  %i1275 = select i1 %.not, float %i1268, float %i1267, !dbg !11
  %i1276 = bitcast float %i1275 to i32, !dbg !11
  %i1277 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1276), !dbg !11
  %i1278 = bitcast i32 %i1277 to float, !dbg !11
  %i1279 = select i1 %.not, float %i1267, float %i1278, !dbg !11
  %i1280 = select i1 %.not, float %i1278, float %i1268, !dbg !11
  %i1281 = select i1 %.not36, float %i1279, float %i1273, !dbg !11
  %i1282 = bitcast float %i1281 to i32, !dbg !11
  %i1283 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1282), !dbg !11
  %i1284 = bitcast i32 %i1283 to float, !dbg !11
  %i1287 = select i1 %.not36, float %i1280, float %i1274, !dbg !11
  %i1288 = bitcast float %i1287 to i32, !dbg !11
  %i1289 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1288), !dbg !11
  %i1290 = bitcast i32 %i1289 to float, !dbg !11
  %77 = insertelement <4 x float> poison, float %i1273, i64 0, !dbg !11
  %78 = insertelement <4 x float> %77, float %i1274, i64 1, !dbg !11
  %79 = insertelement <4 x float> %78, float %i1284, i64 2, !dbg !11
  %80 = insertelement <4 x float> %79, float %i1290, i64 3, !dbg !11
  %81 = insertelement <4 x float> poison, float %i1284, i64 0, !dbg !11
  %82 = insertelement <4 x float> %81, float %i1290, i64 1, !dbg !11
  %83 = insertelement <4 x float> %82, float %i1279, i64 2, !dbg !11
  %84 = insertelement <4 x float> %83, float %i1280, i64 3, !dbg !11
  %85 = select <4 x i1> %22, <4 x float> %80, <4 x float> %84, !dbg !11
  %i1294 = extractelement <4 x float> %i926, i64 0, !dbg !11
  %i1295 = extractelement <4 x float> %i926, i64 1, !dbg !11
  %i1296 = extractelement <4 x float> %i926, i64 2, !dbg !11
  %i1297 = extractelement <4 x float> %i926, i64 3, !dbg !11
  %i1298 = select i1 %.not, float %i1295, float %i1294, !dbg !11
  %i1299 = bitcast float %i1298 to i32, !dbg !11
  %i1300 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1299), !dbg !11
  %i1301 = bitcast i32 %i1300 to float, !dbg !11
  %i1302 = select i1 %.not, float %i1294, float %i1301, !dbg !11
  %i1303 = select i1 %.not, float %i1301, float %i1295, !dbg !11
  %i1304 = select i1 %.not, float %i1297, float %i1296, !dbg !11
  %i1305 = bitcast float %i1304 to i32, !dbg !11
  %i1306 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1305), !dbg !11
  %i1307 = bitcast i32 %i1306 to float, !dbg !11
  %i1308 = select i1 %.not, float %i1296, float %i1307, !dbg !11
  %i1309 = select i1 %.not, float %i1307, float %i1297, !dbg !11
  %i1310 = select i1 %.not36, float %i1308, float %i1302, !dbg !11
  %i1311 = bitcast float %i1310 to i32, !dbg !11
  %i1312 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1311), !dbg !11
  %i1313 = bitcast i32 %i1312 to float, !dbg !11
  %i1316 = select i1 %.not36, float %i1309, float %i1303, !dbg !11
  %i1317 = bitcast float %i1316 to i32, !dbg !11
  %i1318 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1317), !dbg !11
  %i1319 = bitcast i32 %i1318 to float, !dbg !11
  %86 = insertelement <4 x float> poison, float %i1302, i64 0, !dbg !11
  %87 = insertelement <4 x float> %86, float %i1303, i64 1, !dbg !11
  %88 = insertelement <4 x float> %87, float %i1313, i64 2, !dbg !11
  %89 = insertelement <4 x float> %88, float %i1319, i64 3, !dbg !11
  %90 = insertelement <4 x float> poison, float %i1313, i64 0, !dbg !11
  %91 = insertelement <4 x float> %90, float %i1319, i64 1, !dbg !11
  %92 = insertelement <4 x float> %91, float %i1308, i64 2, !dbg !11
  %93 = insertelement <4 x float> %92, float %i1309, i64 3, !dbg !11
  %94 = select <4 x i1> %22, <4 x float> %89, <4 x float> %93, !dbg !11
  %i1323 = extractelement <4 x float> %i850, i64 0, !dbg !11
  %i1324 = extractelement <4 x float> %i850, i64 1, !dbg !11
  %i1325 = extractelement <4 x float> %i850, i64 2, !dbg !11
  %i1326 = extractelement <4 x float> %i850, i64 3, !dbg !11
  %i1327 = select i1 %.not, float %i1324, float %i1323, !dbg !11
  %i1328 = bitcast float %i1327 to i32, !dbg !11
  %i1329 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1328), !dbg !11
  %i1330 = bitcast i32 %i1329 to float, !dbg !11
  %i1331 = select i1 %.not, float %i1323, float %i1330, !dbg !11
  %i1332 = select i1 %.not, float %i1330, float %i1324, !dbg !11
  %i1333 = select i1 %.not, float %i1326, float %i1325, !dbg !11
  %i1334 = bitcast float %i1333 to i32, !dbg !11
  %i1335 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1334), !dbg !11
  %i1336 = bitcast i32 %i1335 to float, !dbg !11
  %i1337 = select i1 %.not, float %i1325, float %i1336, !dbg !11
  %i1338 = select i1 %.not, float %i1336, float %i1326, !dbg !11
  %i1339 = select i1 %.not36, float %i1337, float %i1331, !dbg !11
  %i1340 = bitcast float %i1339 to i32, !dbg !11
  %i1341 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1340), !dbg !11
  %i1342 = bitcast i32 %i1341 to float, !dbg !11
  %i1345 = select i1 %.not36, float %i1338, float %i1332, !dbg !11
  %i1346 = bitcast float %i1345 to i32, !dbg !11
  %i1347 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1346), !dbg !11
  %i1348 = bitcast i32 %i1347 to float, !dbg !11
  %95 = insertelement <4 x float> poison, float %i1331, i64 0, !dbg !11
  %96 = insertelement <4 x float> %95, float %i1332, i64 1, !dbg !11
  %97 = insertelement <4 x float> %96, float %i1342, i64 2, !dbg !11
  %98 = insertelement <4 x float> %97, float %i1348, i64 3, !dbg !11
  %99 = insertelement <4 x float> poison, float %i1342, i64 0, !dbg !11
  %100 = insertelement <4 x float> %99, float %i1348, i64 1, !dbg !11
  %101 = insertelement <4 x float> %100, float %i1337, i64 2, !dbg !11
  %102 = insertelement <4 x float> %101, float %i1338, i64 3, !dbg !11
  %103 = select <4 x i1> %22, <4 x float> %98, <4 x float> %102, !dbg !11
  %i1352 = extractelement <4 x float> %i774, i64 0, !dbg !11
  %i1353 = extractelement <4 x float> %i774, i64 1, !dbg !11
  %i1354 = extractelement <4 x float> %i774, i64 2, !dbg !11
  %i1355 = extractelement <4 x float> %i774, i64 3, !dbg !11
  %i1356 = select i1 %.not, float %i1353, float %i1352, !dbg !11
  %i1357 = bitcast float %i1356 to i32, !dbg !11
  %i1358 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1357), !dbg !11
  %i1359 = bitcast i32 %i1358 to float, !dbg !11
  %i1360 = select i1 %.not, float %i1352, float %i1359, !dbg !11
  %i1361 = select i1 %.not, float %i1359, float %i1353, !dbg !11
  %i1362 = select i1 %.not, float %i1355, float %i1354, !dbg !11
  %i1363 = bitcast float %i1362 to i32, !dbg !11
  %i1364 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1363), !dbg !11
  %i1365 = bitcast i32 %i1364 to float, !dbg !11
  %i1366 = select i1 %.not, float %i1354, float %i1365, !dbg !11
  %i1367 = select i1 %.not, float %i1365, float %i1355, !dbg !11
  %i1368 = select i1 %.not36, float %i1366, float %i1360, !dbg !11
  %i1369 = bitcast float %i1368 to i32, !dbg !11
  %i1370 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1369), !dbg !11
  %i1371 = select i1 %.not36, float %i1367, float %i1361, !dbg !11
  %i1372 = bitcast float %i1371 to i32, !dbg !11
  %i1373 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1372), !dbg !11
  %i1374 = shufflevector <1 x float> %i1029, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !50
  %i1375 = extractelement <4 x float> %i1001, i64 0, !dbg !11
  %i1376 = extractelement <4 x float> %i1001, i64 1, !dbg !11
  %i1377 = extractelement <4 x float> %i1001, i64 2, !dbg !11
  %i1378 = extractelement <4 x float> %i1001, i64 3, !dbg !11
  %i1379 = select i1 %.not, float %i1376, float %i1375, !dbg !11
  %i1380 = bitcast float %i1379 to i32, !dbg !11
  %i1381 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1380), !dbg !11
  %i1382 = bitcast i32 %i1381 to float, !dbg !11
  %i1383 = select i1 %.not, float %i1375, float %i1382, !dbg !11
  %i1384 = select i1 %.not, float %i1382, float %i1376, !dbg !11
  %i1385 = select i1 %.not, float %i1378, float %i1377, !dbg !11
  %i1386 = bitcast float %i1385 to i32, !dbg !11
  %i1387 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1386), !dbg !11
  %i1388 = bitcast i32 %i1387 to float, !dbg !11
  %i1389 = select i1 %.not, float %i1377, float %i1388, !dbg !11
  %i1390 = select i1 %.not, float %i1388, float %i1378, !dbg !11
  %i1391 = select i1 %.not36, float %i1389, float %i1383, !dbg !11
  %i1392 = bitcast float %i1391 to i32, !dbg !11
  %i1393 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1392), !dbg !11
  %i1394 = bitcast i32 %i1393 to float, !dbg !11
  %i1397 = select i1 %.not36, float %i1390, float %i1384, !dbg !11
  %i1398 = bitcast float %i1397 to i32, !dbg !11
  %i1399 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1398), !dbg !11
  %i1400 = bitcast i32 %i1399 to float, !dbg !11
  %104 = insertelement <4 x float> poison, float %i1383, i64 0, !dbg !11
  %105 = insertelement <4 x float> %104, float %i1384, i64 1, !dbg !11
  %106 = insertelement <4 x float> %105, float %i1394, i64 2, !dbg !11
  %107 = insertelement <4 x float> %106, float %i1400, i64 3, !dbg !11
  %108 = insertelement <4 x float> poison, float %i1394, i64 0, !dbg !11
  %109 = insertelement <4 x float> %108, float %i1400, i64 1, !dbg !11
  %110 = insertelement <4 x float> %109, float %i1389, i64 2, !dbg !11
  %111 = insertelement <4 x float> %110, float %i1390, i64 3, !dbg !11
  %112 = select <4 x i1> %22, <4 x float> %107, <4 x float> %111, !dbg !11
  %i1403 = extractelement <4 x float> %i927, i64 0, !dbg !11
  %i1404 = extractelement <4 x float> %i927, i64 1, !dbg !11
  %i1405 = extractelement <4 x float> %i927, i64 2, !dbg !11
  %i1406 = extractelement <4 x float> %i927, i64 3, !dbg !11
  %i1407 = select i1 %.not, float %i1404, float %i1403, !dbg !11
  %i1408 = bitcast float %i1407 to i32, !dbg !11
  %i1409 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1408), !dbg !11
  %i1410 = bitcast i32 %i1409 to float, !dbg !11
  %i1411 = select i1 %.not, float %i1403, float %i1410, !dbg !11
  %i1412 = select i1 %.not, float %i1410, float %i1404, !dbg !11
  %i1413 = select i1 %.not, float %i1406, float %i1405, !dbg !11
  %i1414 = bitcast float %i1413 to i32, !dbg !11
  %i1415 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1414), !dbg !11
  %i1416 = bitcast i32 %i1415 to float, !dbg !11
  %i1417 = select i1 %.not, float %i1405, float %i1416, !dbg !11
  %i1418 = select i1 %.not, float %i1416, float %i1406, !dbg !11
  %i1419 = select i1 %.not36, float %i1417, float %i1411, !dbg !11
  %i1420 = bitcast float %i1419 to i32, !dbg !11
  %i1421 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1420), !dbg !11
  %i1422 = bitcast i32 %i1421 to float, !dbg !11
  %i1425 = select i1 %.not36, float %i1418, float %i1412, !dbg !11
  %i1426 = bitcast float %i1425 to i32, !dbg !11
  %i1427 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1426), !dbg !11
  %i1428 = bitcast i32 %i1427 to float, !dbg !11
  %113 = insertelement <4 x float> poison, float %i1411, i64 0, !dbg !11
  %114 = insertelement <4 x float> %113, float %i1412, i64 1, !dbg !11
  %115 = insertelement <4 x float> %114, float %i1422, i64 2, !dbg !11
  %116 = insertelement <4 x float> %115, float %i1428, i64 3, !dbg !11
  %117 = insertelement <4 x float> poison, float %i1422, i64 0, !dbg !11
  %118 = insertelement <4 x float> %117, float %i1428, i64 1, !dbg !11
  %119 = insertelement <4 x float> %118, float %i1417, i64 2, !dbg !11
  %120 = insertelement <4 x float> %119, float %i1418, i64 3, !dbg !11
  %121 = select <4 x i1> %22, <4 x float> %116, <4 x float> %120, !dbg !11
  %i1431 = extractelement <4 x float> %i851, i64 0, !dbg !11
  %i1432 = extractelement <4 x float> %i851, i64 1, !dbg !11
  %i1433 = extractelement <4 x float> %i851, i64 2, !dbg !11
  %i1434 = extractelement <4 x float> %i851, i64 3, !dbg !11
  %i1435 = select i1 %.not, float %i1432, float %i1431, !dbg !11
  %i1436 = bitcast float %i1435 to i32, !dbg !11
  %i1437 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1436), !dbg !11
  %i1438 = bitcast i32 %i1437 to float, !dbg !11
  %i1439 = select i1 %.not, float %i1431, float %i1438, !dbg !11
  %i1440 = select i1 %.not, float %i1438, float %i1432, !dbg !11
  %i1441 = select i1 %.not, float %i1434, float %i1433, !dbg !11
  %i1442 = bitcast float %i1441 to i32, !dbg !11
  %i1443 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1442), !dbg !11
  %i1444 = bitcast i32 %i1443 to float, !dbg !11
  %i1445 = select i1 %.not, float %i1433, float %i1444, !dbg !11
  %i1446 = select i1 %.not, float %i1444, float %i1434, !dbg !11
  %i1447 = select i1 %.not36, float %i1445, float %i1439, !dbg !11
  %i1448 = bitcast float %i1447 to i32, !dbg !11
  %i1449 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1448), !dbg !11
  %i1450 = bitcast i32 %i1449 to float, !dbg !11
  %i1453 = select i1 %.not36, float %i1446, float %i1440, !dbg !11
  %i1454 = bitcast float %i1453 to i32, !dbg !11
  %i1455 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1454), !dbg !11
  %i1456 = bitcast i32 %i1455 to float, !dbg !11
  %122 = insertelement <4 x float> poison, float %i1439, i64 0, !dbg !11
  %123 = insertelement <4 x float> %122, float %i1440, i64 1, !dbg !11
  %124 = insertelement <4 x float> %123, float %i1450, i64 2, !dbg !11
  %125 = insertelement <4 x float> %124, float %i1456, i64 3, !dbg !11
  %126 = insertelement <4 x float> poison, float %i1450, i64 0, !dbg !11
  %127 = insertelement <4 x float> %126, float %i1456, i64 1, !dbg !11
  %128 = insertelement <4 x float> %127, float %i1445, i64 2, !dbg !11
  %129 = insertelement <4 x float> %128, float %i1446, i64 3, !dbg !11
  %130 = select <4 x i1> %22, <4 x float> %125, <4 x float> %129, !dbg !11
  %i1459 = extractelement <4 x float> %i775, i64 0, !dbg !11
  %i1460 = extractelement <4 x float> %i775, i64 1, !dbg !11
  %i1461 = extractelement <4 x float> %i775, i64 2, !dbg !11
  %i1462 = extractelement <4 x float> %i775, i64 3, !dbg !11
  %i1463 = select i1 %.not, float %i1460, float %i1459, !dbg !11
  %i1464 = bitcast float %i1463 to i32, !dbg !11
  %i1465 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1464), !dbg !11
  %i1466 = bitcast i32 %i1465 to float, !dbg !11
  %i1467 = select i1 %.not, float %i1459, float %i1466, !dbg !11
  %i1468 = select i1 %.not, float %i1466, float %i1460, !dbg !11
  %i1469 = select i1 %.not, float %i1462, float %i1461, !dbg !11
  %i1470 = bitcast float %i1469 to i32, !dbg !11
  %i1471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i1470), !dbg !11
  %i1472 = bitcast i32 %i1471 to float, !dbg !11
  %i1473 = select i1 %.not, float %i1461, float %i1472, !dbg !11
  %i1474 = select i1 %.not, float %i1472, float %i1462, !dbg !11
  %i1475 = select i1 %.not36, float %i1473, float %i1467, !dbg !11
  %i1476 = bitcast float %i1475 to i32, !dbg !11
  %i1477 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1476), !dbg !11
  %i1478 = select i1 %.not36, float %i1474, float %i1468, !dbg !11
  %i1479 = bitcast float %i1478 to i32, !dbg !11
  %i1480 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i1479), !dbg !11
  %i1481 = bitcast i32 %i1477 to float, !dbg !11
  %i1484 = bitcast i32 %i1480 to float, !dbg !11
  %131 = insertelement <4 x float> poison, float %i1467, i64 0, !dbg !11
  %132 = insertelement <4 x float> %131, float %i1468, i64 1, !dbg !11
  %133 = insertelement <4 x float> %132, float %i1481, i64 2, !dbg !11
  %134 = insertelement <4 x float> %133, float %i1484, i64 3, !dbg !11
  %135 = insertelement <4 x float> poison, float %i1481, i64 0, !dbg !11
  %136 = insertelement <4 x float> %135, float %i1484, i64 1, !dbg !11
  %137 = insertelement <4 x float> %136, float %i1473, i64 2, !dbg !11
  %138 = insertelement <4 x float> %137, float %i1474, i64 3, !dbg !11
  %139 = select <4 x i1> %22, <4 x float> %134, <4 x float> %138, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %i1374, !dbg !50
  %normalized = fmul <4 x float> %row.reciprocal, %139, !dbg !50
  %i1491 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !51
  %i1492 = extractelement <4 x half> %i1491, i64 3, !dbg !51
  %i1493 = extractelement <4 x half> %i1491, i64 2, !dbg !51
  %i1494 = extractelement <4 x half> %i1491, i64 1, !dbg !51
  %i1495 = extractelement <4 x half> %i1491, i64 0, !dbg !51
  %i1496 = bitcast i32 %i1370 to float, !dbg !11
  %i1499 = bitcast i32 %i1373 to float, !dbg !11
  %140 = insertelement <4 x float> poison, float %i1360, i64 0, !dbg !11
  %141 = insertelement <4 x float> %140, float %i1361, i64 1, !dbg !11
  %142 = insertelement <4 x float> %141, float %i1496, i64 2, !dbg !11
  %143 = insertelement <4 x float> %142, float %i1499, i64 3, !dbg !11
  %144 = insertelement <4 x float> poison, float %i1496, i64 0, !dbg !11
  %145 = insertelement <4 x float> %144, float %i1499, i64 1, !dbg !11
  %146 = insertelement <4 x float> %145, float %i1366, i64 2, !dbg !11
  %147 = insertelement <4 x float> %146, float %i1367, i64 3, !dbg !11
  %148 = select <4 x i1> %22, <4 x float> %143, <4 x float> %147, !dbg !11
  %normalized37 = fmul <4 x float> %row.reciprocal, %148, !dbg !50
  %i1506 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !51
  %i1507 = extractelement <4 x half> %i1506, i64 3, !dbg !51
  %i1508 = extractelement <4 x half> %i1506, i64 2, !dbg !51
  %i1509 = extractelement <4 x half> %i1506, i64 1, !dbg !51
  %i1510 = extractelement <4 x half> %i1506, i64 0, !dbg !51
  %i1511 = bitcast i32 %i1260 to float, !dbg !11
  %i1514 = bitcast i32 %i1263 to float, !dbg !11
  %149 = insertelement <4 x float> poison, float %i1250, i64 0, !dbg !11
  %150 = insertelement <4 x float> %149, float %i1251, i64 1, !dbg !11
  %151 = insertelement <4 x float> %150, float %i1511, i64 2, !dbg !11
  %152 = insertelement <4 x float> %151, float %i1514, i64 3, !dbg !11
  %153 = insertelement <4 x float> poison, float %i1511, i64 0, !dbg !11
  %154 = insertelement <4 x float> %153, float %i1514, i64 1, !dbg !11
  %155 = insertelement <4 x float> %154, float %i1256, i64 2, !dbg !11
  %156 = insertelement <4 x float> %155, float %i1257, i64 3, !dbg !11
  %157 = select <4 x i1> %22, <4 x float> %152, <4 x float> %156, !dbg !11
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %i1157, !dbg !50
  %normalized39 = fmul <4 x float> %row.reciprocal38, %157, !dbg !50
  %i1521 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !51
  %i1522 = extractelement <4 x half> %i1521, i64 3, !dbg !51
  %i1523 = extractelement <4 x half> %i1521, i64 2, !dbg !51
  %i1524 = extractelement <4 x half> %i1521, i64 1, !dbg !51
  %i1525 = extractelement <4 x half> %i1521, i64 0, !dbg !51
  %i1526 = bitcast i32 %i1153 to float, !dbg !11
  %i1529 = bitcast i32 %i1156 to float, !dbg !11
  %158 = insertelement <4 x float> poison, float %i1143, i64 0, !dbg !11
  %159 = insertelement <4 x float> %158, float %i1144, i64 1, !dbg !11
  %160 = insertelement <4 x float> %159, float %i1526, i64 2, !dbg !11
  %161 = insertelement <4 x float> %160, float %i1529, i64 3, !dbg !11
  %162 = insertelement <4 x float> poison, float %i1526, i64 0, !dbg !11
  %163 = insertelement <4 x float> %162, float %i1529, i64 1, !dbg !11
  %164 = insertelement <4 x float> %163, float %i1149, i64 2, !dbg !11
  %165 = insertelement <4 x float> %164, float %i1150, i64 3, !dbg !11
  %166 = select <4 x i1> %22, <4 x float> %161, <4 x float> %165, !dbg !11
  %normalized40 = fmul <4 x float> %row.reciprocal38, %166, !dbg !50
  %i1536 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !51
  %i1537 = extractelement <4 x half> %i1536, i64 3, !dbg !51
  %i1538 = extractelement <4 x half> %i1536, i64 2, !dbg !51
  %i1539 = extractelement <4 x half> %i1536, i64 1, !dbg !51
  %i1540 = extractelement <4 x half> %i1536, i64 0, !dbg !51
  %address.thread974 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !52, !range !9, !frisk.register_lifetimes !5
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %i1541 = lshr i64 %address.local975, 2, !dbg !52
  %address.local978 = and i64 %i1541, 48, !dbg !52
  %address.local980 = and i64 %address.local975, 240, !dbg !52
  %address.local979 = add i64 %i19, %address.local975, !dbg !52
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !52
  %address.local982 = add i64 %address.local981, %address.local978, !dbg !52
  %i1542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %arg3, i64 %i21, !dbg !52
  %address.local985.idx = shl nuw nsw i64 %address.local982, 8, !dbg !52
  %address.local985 = getelementptr inbounds nuw i8, ptr addrspace(1) %i1542, i64 %address.local985.idx, !dbg !52
  %address.local987 = and i64 %i1541, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !52
  store half %i1540, ptr addrspace(1) %address.local988, align 2, !dbg !52
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !52
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !52
  store half %i1539, ptr addrspace(1) %address.local991, align 2, !dbg !52
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !52
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !52
  store half %i1538, ptr addrspace(1) %address.local994, align 2, !dbg !52
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !52
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !52
  store half %i1537, ptr addrspace(1) %address.local997, align 2, !dbg !52
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !52
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !52
  store half %i1525, ptr addrspace(1) %address.local1000, align 2, !dbg !52
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !52
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !52
  store half %i1524, ptr addrspace(1) %address.local1003, align 2, !dbg !52
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !52
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !52
  store half %i1523, ptr addrspace(1) %address.local1006, align 2, !dbg !52
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !52
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !52
  store half %i1522, ptr addrspace(1) %address.local1009, align 2, !dbg !52
  %address.local1013 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local985, i64 16384, !dbg !52
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !52
  store half %i1510, ptr addrspace(1) %address.local1014, align 2, !dbg !52
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !52
  store half %i1509, ptr addrspace(1) %address.local1016, align 2, !dbg !52
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !52
  store half %i1508, ptr addrspace(1) %address.local1018, align 2, !dbg !52
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !52
  store half %i1507, ptr addrspace(1) %address.local1020, align 2, !dbg !52
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !52
  store half %i1495, ptr addrspace(1) %address.local1022, align 2, !dbg !52
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !52
  store half %i1494, ptr addrspace(1) %address.local1024, align 2, !dbg !52
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !52
  store half %i1493, ptr addrspace(1) %address.local1026, align 2, !dbg !52
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !52
  store half %i1492, ptr addrspace(1) %address.local1028, align 2, !dbg !52
  %normalized42 = fmul <4 x float> %row.reciprocal, %130, !dbg !50
  %i1547 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !51
  %i1548 = extractelement <4 x half> %i1547, i64 3, !dbg !51
  %i1549 = extractelement <4 x half> %i1547, i64 2, !dbg !51
  %i1550 = extractelement <4 x half> %i1547, i64 1, !dbg !51
  %i1551 = extractelement <4 x half> %i1547, i64 0, !dbg !51
  %normalized43 = fmul <4 x float> %row.reciprocal, %103, !dbg !50
  %i1556 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !51
  %i1557 = extractelement <4 x half> %i1556, i64 3, !dbg !51
  %i1558 = extractelement <4 x half> %i1556, i64 2, !dbg !51
  %i1559 = extractelement <4 x half> %i1556, i64 1, !dbg !51
  %i1560 = extractelement <4 x half> %i1556, i64 0, !dbg !51
  %normalized45 = fmul <4 x float> %row.reciprocal38, %76, !dbg !50
  %i1565 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !51
  %i1566 = extractelement <4 x half> %i1565, i64 3, !dbg !51
  %i1567 = extractelement <4 x half> %i1565, i64 2, !dbg !51
  %i1568 = extractelement <4 x half> %i1565, i64 1, !dbg !51
  %i1569 = extractelement <4 x half> %i1565, i64 0, !dbg !51
  %normalized46 = fmul <4 x float> %row.reciprocal38, %49, !dbg !50
  %i1574 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !51
  %i1575 = extractelement <4 x half> %i1574, i64 3, !dbg !51
  %i1576 = extractelement <4 x half> %i1574, i64 2, !dbg !51
  %i1577 = extractelement <4 x half> %i1574, i64 1, !dbg !51
  %i1578 = extractelement <4 x half> %i1574, i64 0, !dbg !51
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !52
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !52
  store half %i1578, ptr addrspace(1) %address.local1044, align 2, !dbg !52
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !52
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !52
  store half %i1577, ptr addrspace(1) %address.local1047, align 2, !dbg !52
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !52
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !52
  store half %i1576, ptr addrspace(1) %address.local1050, align 2, !dbg !52
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !52
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !52
  store half %i1575, ptr addrspace(1) %address.local1053, align 2, !dbg !52
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !52
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !52
  store half %i1569, ptr addrspace(1) %address.local1056, align 2, !dbg !52
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !52
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !52
  store half %i1568, ptr addrspace(1) %address.local1059, align 2, !dbg !52
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !52
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !52
  store half %i1567, ptr addrspace(1) %address.local1062, align 2, !dbg !52
  %address.local1064 = or i64 %i1541, 51, !dbg !52
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !52
  store half %i1566, ptr addrspace(1) %address.local1065, align 2, !dbg !52
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !52
  store half %i1560, ptr addrspace(1) %address.local1070, align 2, !dbg !52
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !52
  store half %i1559, ptr addrspace(1) %address.local1072, align 2, !dbg !52
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !52
  store half %i1558, ptr addrspace(1) %address.local1074, align 2, !dbg !52
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !52
  store half %i1557, ptr addrspace(1) %address.local1076, align 2, !dbg !52
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !52
  store half %i1551, ptr addrspace(1) %address.local1078, align 2, !dbg !52
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !52
  store half %i1550, ptr addrspace(1) %address.local1080, align 2, !dbg !52
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !52
  store half %i1549, ptr addrspace(1) %address.local1082, align 2, !dbg !52
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !52
  store half %i1548, ptr addrspace(1) %address.local1084, align 2, !dbg !52
  %normalized48 = fmul <4 x float> %row.reciprocal, %121, !dbg !50
  %i1583 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !51
  %i1584 = extractelement <4 x half> %i1583, i64 3, !dbg !51
  %i1585 = extractelement <4 x half> %i1583, i64 2, !dbg !51
  %i1586 = extractelement <4 x half> %i1583, i64 1, !dbg !51
  %i1587 = extractelement <4 x half> %i1583, i64 0, !dbg !51
  %normalized49 = fmul <4 x float> %row.reciprocal, %94, !dbg !50
  %i1592 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !51
  %i1593 = extractelement <4 x half> %i1592, i64 3, !dbg !51
  %i1594 = extractelement <4 x half> %i1592, i64 2, !dbg !51
  %i1595 = extractelement <4 x half> %i1592, i64 1, !dbg !51
  %i1596 = extractelement <4 x half> %i1592, i64 0, !dbg !51
  %normalized51 = fmul <4 x float> %row.reciprocal38, %67, !dbg !50
  %i1601 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !51
  %i1602 = extractelement <4 x half> %i1601, i64 3, !dbg !51
  %i1603 = extractelement <4 x half> %i1601, i64 2, !dbg !51
  %i1604 = extractelement <4 x half> %i1601, i64 1, !dbg !51
  %i1605 = extractelement <4 x half> %i1601, i64 0, !dbg !51
  %normalized52 = fmul <4 x float> %row.reciprocal38, %40, !dbg !50
  %i1610 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !51
  %i1611 = extractelement <4 x half> %i1610, i64 3, !dbg !51
  %i1612 = extractelement <4 x half> %i1610, i64 2, !dbg !51
  %i1613 = extractelement <4 x half> %i1610, i64 1, !dbg !51
  %i1614 = extractelement <4 x half> %i1610, i64 0, !dbg !51
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !52
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !52
  store half %i1614, ptr addrspace(1) %address.local1100, align 2, !dbg !52
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !52
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !52
  store half %i1613, ptr addrspace(1) %address.local1103, align 2, !dbg !52
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !52
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !52
  store half %i1612, ptr addrspace(1) %address.local1106, align 2, !dbg !52
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !52
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !52
  store half %i1611, ptr addrspace(1) %address.local1109, align 2, !dbg !52
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !52
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !52
  store half %i1605, ptr addrspace(1) %address.local1112, align 2, !dbg !52
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !52
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !52
  store half %i1604, ptr addrspace(1) %address.local1115, align 2, !dbg !52
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !52
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !52
  store half %i1603, ptr addrspace(1) %address.local1118, align 2, !dbg !52
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !52
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !52
  store half %i1602, ptr addrspace(1) %address.local1121, align 2, !dbg !52
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !52
  store half %i1596, ptr addrspace(1) %address.local1126, align 2, !dbg !52
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !52
  store half %i1595, ptr addrspace(1) %address.local1128, align 2, !dbg !52
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !52
  store half %i1594, ptr addrspace(1) %address.local1130, align 2, !dbg !52
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !52
  store half %i1593, ptr addrspace(1) %address.local1132, align 2, !dbg !52
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !52
  store half %i1587, ptr addrspace(1) %address.local1134, align 2, !dbg !52
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !52
  store half %i1586, ptr addrspace(1) %address.local1136, align 2, !dbg !52
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !52
  store half %i1585, ptr addrspace(1) %address.local1138, align 2, !dbg !52
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !52
  store half %i1584, ptr addrspace(1) %address.local1140, align 2, !dbg !52
  %normalized54 = fmul <4 x float> %row.reciprocal, %112, !dbg !50
  %i1619 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !51
  %i1620 = extractelement <4 x half> %i1619, i64 3, !dbg !51
  %i1621 = extractelement <4 x half> %i1619, i64 2, !dbg !51
  %i1622 = extractelement <4 x half> %i1619, i64 1, !dbg !51
  %i1623 = extractelement <4 x half> %i1619, i64 0, !dbg !51
  %normalized55 = fmul <4 x float> %row.reciprocal, %85, !dbg !50
  %i1628 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !51
  %i1629 = extractelement <4 x half> %i1628, i64 3, !dbg !51
  %i1630 = extractelement <4 x half> %i1628, i64 2, !dbg !51
  %i1631 = extractelement <4 x half> %i1628, i64 1, !dbg !51
  %i1632 = extractelement <4 x half> %i1628, i64 0, !dbg !51
  %normalized57 = fmul <4 x float> %row.reciprocal38, %58, !dbg !50
  %i1637 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !51
  %i1638 = extractelement <4 x half> %i1637, i64 3, !dbg !51
  %i1639 = extractelement <4 x half> %i1637, i64 2, !dbg !51
  %i1640 = extractelement <4 x half> %i1637, i64 1, !dbg !51
  %i1641 = extractelement <4 x half> %i1637, i64 0, !dbg !51
  %normalized58 = fmul <4 x float> %row.reciprocal38, %31, !dbg !50
  %i1646 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !51
  %i1647 = extractelement <4 x half> %i1646, i64 3, !dbg !51
  %i1648 = extractelement <4 x half> %i1646, i64 2, !dbg !51
  %i1649 = extractelement <4 x half> %i1646, i64 1, !dbg !51
  %i1650 = extractelement <4 x half> %i1646, i64 0, !dbg !51
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !52
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !52
  store half %i1650, ptr addrspace(1) %address.local1156, align 2, !dbg !52
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !52
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !52
  store half %i1649, ptr addrspace(1) %address.local1159, align 2, !dbg !52
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !52
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !52
  store half %i1648, ptr addrspace(1) %address.local1162, align 2, !dbg !52
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !52
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !52
  store half %i1647, ptr addrspace(1) %address.local1165, align 2, !dbg !52
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !52
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !52
  store half %i1641, ptr addrspace(1) %address.local1168, align 2, !dbg !52
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !52
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !52
  store half %i1640, ptr addrspace(1) %address.local1171, align 2, !dbg !52
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !52
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !52
  store half %i1639, ptr addrspace(1) %address.local1174, align 2, !dbg !52
  %address.local1176 = or i64 %i1541, 115, !dbg !52
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !52
  store half %i1638, ptr addrspace(1) %address.local1177, align 2, !dbg !52
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !52
  store half %i1632, ptr addrspace(1) %address.local1182, align 2, !dbg !52
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !52
  store half %i1631, ptr addrspace(1) %address.local1184, align 2, !dbg !52
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !52
  store half %i1630, ptr addrspace(1) %address.local1186, align 2, !dbg !52
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !52
  store half %i1629, ptr addrspace(1) %address.local1188, align 2, !dbg !52
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !52
  store half %i1623, ptr addrspace(1) %address.local1190, align 2, !dbg !52
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !52
  store half %i1622, ptr addrspace(1) %address.local1192, align 2, !dbg !52
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !52
  store half %i1621, ptr addrspace(1) %address.local1194, align 2, !dbg !52
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !52
  store half %i1620, ptr addrspace(1) %address.local1196, align 2, !dbg !52
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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
!1 = !DIFile(filename: "input.mlir", directory: "generated_ir/noncausal_k_lds_prefetch_pass/mytest_final/run-20261030-171621-98fko2hz/online-vdb-k64-lds2")
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
!45 = !DILocation(line: 87, column: 23, scope: !3)
!46 = !{i64 8192}
!47 = !DILocation(line: 79, column: 22, scope: !3)
!48 = !DILocation(line: 81, column: 16, scope: !3)
!49 = !DILocation(line: 80, column: 20, scope: !3)
!50 = !DILocation(line: 92, column: 23, scope: !3)
!51 = !DILocation(line: 93, column: 19, scope: !3)
!52 = !DILocation(line: 94, column: 9, scope: !3)
