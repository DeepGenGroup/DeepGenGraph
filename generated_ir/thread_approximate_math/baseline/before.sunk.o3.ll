; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/baseline/kernel.o3.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/baseline/input.mlir"

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
  %i5 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %i6 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %i5), !dbg !7
  %i7 = add nuw nsw i32 %i6, 64, !dbg !7
  %i8 = xor i32 %i6, 16, !dbg !7
  %i9 = and i32 %i7, 192, !dbg !7
  %i10 = icmp samesign ult i32 %i8, %i9, !dbg !7
  %i11 = select i1 %i10, i32 %i8, i32 %i6, !dbg !7
  %i12 = shl nuw nsw i32 %i11, 2, !dbg !7
  %i13 = xor i32 %i6, 32, !dbg !7
  %i14 = icmp samesign ult i32 %i13, %i9, !dbg !7
  %i15 = select i1 %i14, i32 %i13, i32 %i6, !dbg !7
  %i16 = shl nuw nsw i32 %i15, 2, !dbg !7
  %i17 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !8
  %i18 = sext i32 %i17 to i64, !dbg !8
  %i19 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !8
  %i20 = sext i32 %i19 to i64, !dbg !8
  %i21 = shl nsw i64 %i20, 5, !dbg !8
  %i22 = add nsw i64 %i21, %i18, !dbg !8
  %i23 = lshr i64 %i22, 5, !dbg !8
  %i24 = and i64 %i23, 576460752303423456, !dbg !8
  %i25 = and i64 %i18, 31, !dbg !8
  %i26 = or disjoint i64 %i24, %i25, !dbg !8
  %i27 = lshr i64 %i22, 10, !dbg !9
  %i28 = add nuw nsw i64 %i23, %i27, !dbg !9
  %i29 = and i64 %i28, 31, !dbg !9
  %i30 = shl nuw nsw i64 %i26, 7, !dbg !9
  %i31 = shl i64 %i27, 12, !dbg !9
  %i32 = sub i64 %i30, %i31, !dbg !9
  %i33 = shl nsw i64 %i32, 7, !dbg !9
  %.idx = shl nuw nsw i64 %i29, 20, !dbg !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !9, !range !10, !frisk.register_lifetimes !5
  %i34 = shl nuw nsw i32 %address.thread, 3, !dbg !9
  %i35 = and i32 %i34, 1536, !dbg !9
  %address.local421 = zext nneg i32 %i35 to i64, !dbg !9
  %address.local422 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local421, !dbg !9
  %address.local425 = lshr i32 %address.thread, 4, !dbg !9
  %address.local426 = and i32 %address.local425, 8, !dbg !9
  %address.local427 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local428 = and i32 %address.local427, 16256, !dbg !9
  %address.local429 = or disjoint i32 %address.local426, %address.local428, !dbg !9
  %i36 = ptrtoint ptr addrspace(1) %arg to i64, !dbg !9
  %.idx1395 = shl i64 %i32, 8, !dbg !9
  %i37 = add i64 %.idx1395, %.idx, !dbg !9
  %i38 = shl nuw nsw i32 %address.local429, 1, !dbg !9
  %i39 = trunc i64 %i37 to i32, !dbg !9
  %dma.buffer.offset1393 = or disjoint i32 %i38, %i39, !dbg !9
  %i40 = insertelement <2 x i64> poison, i64 %i36, i64 0, !dbg !9
  %i41 = bitcast <2 x i64> %i40 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1394 = shufflevector <4 x i32> %i41, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local434 = or disjoint i64 %address.local421, 2048, !dbg !9
  %address.local435 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local434, !dbg !9
  %address.local443 = or disjoint i32 %address.local429, 16, !dbg !9
  %i42 = shl nuw nsw i32 %address.local443, 1, !dbg !9
  %dma.buffer.offset1391 = or disjoint i32 %i42, %i39, !dbg !9
  %i43 = shl nuw nsw i64 %i29, 19, !dbg !9
  %i44 = add i64 %i33, %i43, !dbg !9
  %address.local446 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local421, !dbg !9
  %i45 = trunc i64 %i44 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local429, %i45, !dbg !9
  %i46 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1389 = or disjoint i32 %i46, 64, !dbg !9
  %address.local450 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local434, !dbg !9
  %i47 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local422, i64 0, !dbg !9
  %i48 = insertelement <4 x ptr addrspace(3)> %i47, ptr addrspace(3) %address.local435, i64 1, !dbg !9
  %i49 = insertelement <4 x ptr addrspace(3)> %i48, ptr addrspace(3) %address.local446, i64 2, !dbg !9
  %i50 = insertelement <4 x ptr addrspace(3)> %i49, ptr addrspace(3) %address.local450, i64 3, !dbg !9
  %i51 = ptrtoint <4 x ptr addrspace(3)> %i50 to <4 x i64>, !dbg !9
  %i52 = trunc <4 x i64> %i51 to <4 x i32>, !dbg !9
  %i53 = extractelement <4 x i32> %i52, i64 0, !dbg !9
  %i54 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i53), !dbg !9
  %i55 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i54, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1394) #9, !dbg !9
  %i56 = extractelement <4 x i32> %i52, i64 1, !dbg !9
  %i57 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i56), !dbg !9
  %i58 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i57, i32 %dma.buffer.offset1391, <4 x i32> %dma.buffer.rsrc1394) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %i59 = extractelement <4 x i32> %i52, i64 2, !dbg !9
  %i60 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i59), !dbg !9
  %i61 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i60, i32 %dma.buffer.offset1389, <4 x i32> %dma.buffer.rsrc1394) #9, !dbg !9
  %i62 = extractelement <4 x i32> %i52, i64 3, !dbg !9
  %i63 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i62), !dbg !9
  %.tr1399 = or disjoint i32 %address.local443, %i45, !dbg !9
  %i64 = shl i32 %.tr1399, 1, !dbg !9
  %dma.buffer.offset1387 = or disjoint i32 %i64, 64, !dbg !9
  %i65 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i63, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1394) #9, !dbg !9
  %address.local454 = lshr i32 %address.thread, 2, !dbg !11
  %address.local455 = and i32 %address.local454, 4, !dbg !11
  %address.local456 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local457 = and i32 %address.local456, 384, !dbg !11
  %address.local458 = or disjoint i32 %address.local455, %address.local457, !dbg !11
  %address.local460 = and i32 %i34, 120, !dbg !11
  %address.local461 = or disjoint i32 %address.local458, %address.local460, !dbg !11
  %address.local462 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local463 = and i32 %address.local462, 1024, !dbg !11
  %address.local464 = or disjoint i32 %address.local461, %address.local463, !dbg !11
  %address.local465 = zext nneg i32 %address.local464 to i64, !dbg !11
  %address.local466 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local465, !dbg !11
  %i66 = load <4 x half>, ptr addrspace(3) %address.local466, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local466, i64 4096, !dbg !11
  %i67 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %i68 = fmul <4 x half> %i66, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local466, i64 1024, !dbg !11
  %i69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !11
  %i70 = fmul <4 x half> %i67, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local466, i64 5120, !dbg !11
  %i71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !11
  %i72 = fmul <4 x half> %i69, splat (half 1.275630e-01), !dbg !11
  %i73 = fmul <4 x half> %i71, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1385 = or disjoint i32 %i46, 128, !dbg !9
  %i74 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i54, i32 %dma.buffer.offset1385, <4 x i32> %dma.buffer.rsrc1394) #9, !dbg !9
  %dma.buffer.offset1383 = or disjoint i32 %i64, 128, !dbg !9
  %i75 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i57, i32 %dma.buffer.offset1383, <4 x i32> %dma.buffer.rsrc1394) #9, !dbg !9
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %address.local466, i64 8192, !dbg !11
  %i76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !11
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %address.local466, i64 12288, !dbg !11
  %i77 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !11
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local466, i64 9216, !dbg !11
  %i78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !11
  %i79 = fmul <4 x half> %i76, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1221 = getelementptr i8, ptr addrspace(3) %address.local466, i64 13312, !dbg !11
  %i80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !11
  %i81 = fmul <4 x half> %i77, splat (half 1.275630e-01), !dbg !11
  %i82 = fmul <4 x half> %i78, splat (half 1.275630e-01), !dbg !11
  %i83 = fmul <4 x half> %i80, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1381 = or disjoint i32 %i46, 192, !dbg !9
  %i84 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i60, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1394) #9, !dbg !9
  %dma.buffer.offset1379 = or disjoint i32 %i64, 192, !dbg !9
  %i85 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i63, i32 %dma.buffer.offset1379, <4 x i32> %dma.buffer.rsrc1394) #9, !dbg !9
  %i86 = load <4 x half>, ptr addrspace(3) %address.local466, align 8, !dbg !11
  %i87 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %i88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !11
  %i89 = fmul <4 x half> %i86, splat (half 1.275630e-01), !dbg !11
  %i90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !11
  %i91 = fmul <4 x half> %i87, splat (half 1.275630e-01), !dbg !11
  %i92 = fmul <4 x half> %i88, splat (half 1.275630e-01), !dbg !11
  %i93 = fmul <4 x half> %i90, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %i94 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !11
  %i95 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !11
  %i96 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !11
  %i97 = fmul <4 x half> %i94, splat (half 1.275630e-01), !dbg !11
  %i98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !11
  %i99 = fmul <4 x half> %i95, splat (half 1.275630e-01), !dbg !11
  %i100 = fmul <4 x half> %i96, splat (half 1.275630e-01), !dbg !11
  %i101 = fmul <4 x half> %i98, splat (half 1.275630e-01), !dbg !11
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %i102 = shl nuw nsw i32 %kdma.tid, 2, !dbg !13
  %i103 = and i32 %i102, 3084, !dbg !13
  %i104 = and i32 %kdma.tid, 1, !dbg !13
  %i105 = add nuw nsw i32 %i104, 1, !dbg !13
  %narrow = or disjoint i32 %i105, %i103, !dbg !13
  %i106 = lshr i32 %narrow, 1, !dbg !13
  %i107 = zext nneg i32 %i106 to i64, !dbg !13
  %k0.prime. = shl i64 %i23, 19, !dbg !14
  %k0.prime.2062 = and i64 %k0.prime., 16252928, !dbg !14
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %k0.prime.kdma.source1501 = lshr i64 %k0.prime.kdma.source, 4, !dbg !14
  %k0.prime.kdma.source1502 = shl nuw nsw i64 %k0.prime.kdma.source1501, 9, !dbg !14
  %k0.prime.kdma.source1503 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !14
  %k0.prime.kdma.source1504 = add nuw nsw i64 %k0.prime.kdma.source1502, %k0.prime.kdma.source1503, !dbg !14
  %k0.prime.kdma.source1505 = lshr i64 %k0.prime.kdma.source, 5, !dbg !14
  %k0.prime.kdma.source1506 = lshr i64 %k0.prime.kdma.source, 2, !dbg !14
  %k0.prime.kdma.source1507 = and i64 %k0.prime.kdma.source1506, 1, !dbg !14
  %k0.prime.kdma.source1508 = and i64 %k0.prime.kdma.source1507, %k0.prime.kdma.source1501, !dbg !14
  %k0.prime.kdma.source1509 = or disjoint i64 %k0.prime.kdma.source1508, %k0.prime.kdma.source1505, !dbg !14
  %i108 = shl nuw nsw i64 %k0.prime.kdma.source1509, 10, !dbg !14
  %k0.prime.kdma.source1511 = sub nsw i64 %k0.prime.kdma.source1504, %i108, !dbg !14
  %k0.prime.kdma.source1512 = and i64 %k0.prime.kdma.source1511, 16256, !dbg !14
  %k0.prime.kdma.source1513 = lshr i32 %kdma.tid, 4, !dbg !14
  %k0.prime.kdma.source1514 = and i32 %k0.prime.kdma.source1513, 8, !dbg !14
  %k0.prime.kdma.source1515 = zext nneg i32 %k0.prime.kdma.source1514 to i64, !dbg !14
  %k0.prime.kdma.source1516 = or disjoint i64 %k0.prime.kdma.source1512, %k0.prime.kdma.source1515, !dbg !14
  %k0.prime.kdma.source1517 = or disjoint i64 %k0.prime.kdma.source1516, %k0.prime.2062, !dbg !14
  %k0.prime.kdma.source1518 = trunc nuw nsw i64 %k0.prime.kdma.source1517 to i32, !dbg !14
  %k0.prime.kdma.source1519 = shl nuw nsw i32 %k0.prime.kdma.source1518, 1, !dbg !14
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1519, %kdma.intra, !dbg !14
  %k0.prime.2065 = ptrtoint ptr addrspace(1) %arg1 to i64, !dbg !14
  %k0.prime.2066 = insertelement <2 x i64> poison, i64 %k0.prime.2065, i64 0, !dbg !14
  %k0.prime.2067 = bitcast <2 x i64> %k0.prime.2066 to <4 x i32>, !dbg !14
  %k0.prime.dma.buffer.rsrc1378 = shufflevector <4 x i32> %k0.prime.2067, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %k0.prime.2068 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i54, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1541 = add i32 %i54, 512, !dbg !14
  %k0.prime.kdma.source1520 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %k0.prime.kdma.source1521 = lshr i64 %k0.prime.kdma.source1520, 4, !dbg !14
  %k0.prime.kdma.source1522 = shl nuw nsw i64 %k0.prime.kdma.source1521, 9, !dbg !14
  %k0.prime.kdma.source1523 = shl nuw nsw i64 %k0.prime.kdma.source1520, 7, !dbg !14
  %k0.prime.kdma.source1524 = add nuw nsw i64 %k0.prime.kdma.source1522, %k0.prime.kdma.source1523, !dbg !14
  %k0.prime.kdma.source1525 = lshr i64 %k0.prime.kdma.source1520, 5, !dbg !14
  %k0.prime.kdma.source1526 = lshr i64 %k0.prime.kdma.source1520, 2, !dbg !14
  %k0.prime.kdma.source1527 = and i64 %k0.prime.kdma.source1526, 1, !dbg !14
  %k0.prime.kdma.source1528 = and i64 %k0.prime.kdma.source1527, %k0.prime.kdma.source1521, !dbg !14
  %k0.prime.kdma.source1529 = add nuw nsw i64 %k0.prime.kdma.source1528, %k0.prime.kdma.source1525, !dbg !14
  %i109 = shl nuw nsw i64 %k0.prime.kdma.source1529, 10, !dbg !14
  %k0.prime.kdma.source1531 = sub nsw i64 %k0.prime.kdma.source1524, %i109, !dbg !14
  %k0.prime.kdma.source1532 = and i64 %k0.prime.kdma.source1531, 16256, !dbg !14
  %k0.prime.kdma.source1536 = or disjoint i64 %k0.prime.kdma.source1532, %k0.prime.kdma.source1515, !dbg !14
  %k0.prime.kdma.source1537 = or disjoint i64 %k0.prime.kdma.source1536, %k0.prime.2062, !dbg !14
  %k0.prime.kdma.source1538 = trunc nuw nsw i64 %k0.prime.kdma.source1537 to i32, !dbg !14
  %k0.prime.kdma.source1539 = shl nuw nsw i32 %k0.prime.kdma.source1538, 1, !dbg !14
  %k0.prime.kdma.offset1540 = or disjoint i32 %k0.prime.kdma.source1539, %kdma.intra.hi, !dbg !14
  %k0.prime.2069 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1541, i32 %k0.prime.kdma.offset1540, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1543 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !14
  %k0.prime.kdma.source1544 = lshr i64 %k0.prime.kdma.source1543, 4, !dbg !14
  %k0.prime.kdma.source1545 = shl nuw nsw i64 %k0.prime.kdma.source1544, 9, !dbg !14
  %k0.prime.kdma.source1547 = add nuw nsw i64 %k0.prime.kdma.source1545, %k0.prime.kdma.source1503, !dbg !14
  %k0.prime.kdma.source1548 = lshr i64 %k0.prime.kdma.source1543, 5, !dbg !14
  %k0.prime.kdma.source1551 = and i64 %k0.prime.kdma.source1544, %k0.prime.kdma.source1507, !dbg !14
  %k0.prime.kdma.source1552 = or disjoint i64 %k0.prime.kdma.source1551, %k0.prime.kdma.source1548, !dbg !14
  %i110 = shl nuw nsw i64 %k0.prime.kdma.source1552, 10, !dbg !14
  %k0.prime.kdma.source1554 = sub nsw i64 %k0.prime.kdma.source1547, %i110, !dbg !14
  %k0.prime.kdma.source1555 = and i64 %k0.prime.kdma.source1554, 16256, !dbg !14
  %k0.prime.kdma.source1558 = or disjoint i32 %k0.prime.kdma.source1514, 16, !dbg !14
  %k0.prime.kdma.source1559 = zext nneg i32 %k0.prime.kdma.source1558 to i64, !dbg !14
  %k0.prime.kdma.source1560 = or disjoint i64 %k0.prime.kdma.source1555, %k0.prime.kdma.source1559, !dbg !14
  %k0.prime.kdma.source1561 = or disjoint i64 %k0.prime.kdma.source1560, %k0.prime.2062, !dbg !14
  %k0.prime.kdma.source1562 = trunc nuw nsw i64 %k0.prime.kdma.source1561 to i32, !dbg !14
  %k0.prime.kdma.source1563 = shl nuw nsw i32 %k0.prime.kdma.source1562, 1, !dbg !14
  %k0.prime.kdma.offset1564 = add nuw nsw i32 %k0.prime.kdma.source1563, %kdma.intra, !dbg !14
  %k0.prime.2070 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i57, i32 %k0.prime.kdma.offset1564, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1589 = add i32 %i57, 512, !dbg !14
  %k0.prime.kdma.source1567 = or disjoint i64 %k0.prime.kdma.source1520, 256, !dbg !14
  %k0.prime.kdma.source1568 = lshr i64 %k0.prime.kdma.source1567, 4, !dbg !14
  %k0.prime.kdma.source1569 = shl nuw nsw i64 %k0.prime.kdma.source1568, 9, !dbg !14
  %k0.prime.kdma.source1571 = add nuw nsw i64 %k0.prime.kdma.source1569, %k0.prime.kdma.source1523, !dbg !14
  %k0.prime.kdma.source1572 = lshr i64 %k0.prime.kdma.source1567, 5, !dbg !14
  %k0.prime.kdma.source1575 = and i64 %k0.prime.kdma.source1568, %k0.prime.kdma.source1527, !dbg !14
  %k0.prime.kdma.source1576 = add nuw nsw i64 %k0.prime.kdma.source1575, %k0.prime.kdma.source1572, !dbg !14
  %i111 = shl nuw nsw i64 %k0.prime.kdma.source1576, 10, !dbg !14
  %k0.prime.kdma.source1578 = sub nsw i64 %k0.prime.kdma.source1571, %i111, !dbg !14
  %k0.prime.kdma.source1579 = and i64 %k0.prime.kdma.source1578, 16256, !dbg !14
  %k0.prime.kdma.source1584 = or disjoint i64 %k0.prime.kdma.source1579, %k0.prime.kdma.source1559, !dbg !14
  %k0.prime.kdma.source1585 = or disjoint i64 %k0.prime.kdma.source1584, %k0.prime.2062, !dbg !14
  %k0.prime.kdma.source1586 = trunc nuw nsw i64 %k0.prime.kdma.source1585 to i32, !dbg !14
  %k0.prime.kdma.source1587 = shl nuw nsw i32 %k0.prime.kdma.source1586, 1, !dbg !14
  %k0.prime.kdma.offset1588 = add nuw nsw i32 %k0.prime.kdma.source1587, %kdma.intra.hi, !dbg !14
  %k0.prime.2071 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1589, i32 %k0.prime.kdma.offset1588, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %kdma.lds1633 = add i32 %i60, 512
  %kdma.lds1681 = add i32 %i63, 512
  %i112 = ptrtoint ptr addrspace(1) %arg2 to i64
  %i113 = insertelement <2 x i64> poison, i64 %i112, i64 0
  %i114 = bitcast <2 x i64> %i113 to <4 x i32>
  %dma.buffer.rsrc1362 = shufflevector <4 x i32> %i114, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %vdb.dma.base = add i32 %i54, 8192
  %vdb.dma.base1455 = add i32 %i57, 8192
  br label %bb115, !dbg !12

bb115:                                            ; preds = %k0.join, %bb
  %i116 = phi [2 x <8 x float>] [ zeroinitializer, %bb ], [ %i902, %k0.join ]
  %i117 = phi [2 x <8 x float>] [ zeroinitializer, %bb ], [ %i898, %k0.join ]
  %i118 = phi [2 x <8 x float>] [ zeroinitializer, %bb ], [ %i894, %k0.join ]
  %i119 = phi [2 x <1 x float>] [ zeroinitializer, %bb ], [ %lazy.base.join, %k0.join ]
  %i120 = phi [2 x <1 x float>] [ zeroinitializer, %bb ], [ %i667, %k0.join ]
  %i121 = phi [2 x <8 x float>] [ zeroinitializer, %bb ], [ %i890, %k0.join ]
  %i122 = phi i64 [ 0, %bb ], [ %i822, %k0.join ]
  %i123 = shl nuw nsw i64 %i122, 14, !dbg !14
  %i124 = add nuw nsw i64 %i123, %k0.prime.2062, !dbg !14
  %address.thread517 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local518 = zext nneg i32 %address.thread517 to i64, !dbg !9
  %address.local524 = lshr i64 %address.local518, 2, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %i125 = or disjoint i64 %i124, 32, !dbg !14
  %kdma.source1607 = or disjoint i64 %i125, %k0.prime.kdma.source1516, !dbg !14
  %kdma.source1608 = trunc nuw nsw i64 %kdma.source1607 to i32, !dbg !14
  %kdma.source1609 = shl nuw nsw i32 %kdma.source1608, 1, !dbg !14
  %kdma.offset1610 = or disjoint i32 %kdma.source1609, %kdma.intra, !dbg !14
  %i126 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i60, i32 %kdma.offset1610, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %kdma.source1629 = or disjoint i64 %i125, %k0.prime.kdma.source1536, !dbg !14
  %kdma.source1630 = trunc nuw nsw i64 %kdma.source1629 to i32, !dbg !14
  %kdma.source1631 = shl nuw nsw i32 %kdma.source1630, 1, !dbg !14
  %kdma.offset1632 = or disjoint i32 %kdma.source1631, %kdma.intra.hi, !dbg !14
  %i127 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1633, i32 %kdma.offset1632, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %kdma.source1653 = or disjoint i64 %i125, %k0.prime.kdma.source1560, !dbg !14
  %kdma.source1654 = trunc nuw nsw i64 %kdma.source1653 to i32, !dbg !14
  %kdma.source1655 = shl nuw nsw i32 %kdma.source1654, 1, !dbg !14
  %kdma.offset1656 = add nuw nsw i32 %kdma.source1655, %kdma.intra, !dbg !14
  %i128 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i63, i32 %kdma.offset1656, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %kdma.source1677 = or disjoint i64 %i125, %k0.prime.kdma.source1584, !dbg !14
  %kdma.source1678 = trunc nuw nsw i64 %kdma.source1677 to i32, !dbg !14
  %kdma.source1679 = shl nuw nsw i32 %kdma.source1678, 1, !dbg !14
  %kdma.offset1680 = add nuw nsw i32 %kdma.source1679, %kdma.intra.hi, !dbg !14
  %i129 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1681, i32 %kdma.offset1680, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %address.local561 = shl nuw nsw i64 %address.local518, 1, !dbg !13
  %address.local562 = and i64 %address.local561, 24, !dbg !13
  %address.local564 = shl i64 %address.local518, 62, !dbg !13
  %address.local565 = ashr exact i64 %address.local564, 57, !dbg !13
  %address.local566 = and i64 %address.local565, -64, !dbg !13
  %address.local569 = and i64 %address.local524, 12, !dbg !11
  %address.local571 = and i64 %address.local518, 63, !dbg !11
  %address.local572 = lshr i64 %address.local571, 5, !dbg !11
  %address.local573 = mul nuw nsw i64 %address.local572, 1016, !dbg !11
  %.idx1416 = mul nuw nsw i64 %address.local518, 320, !dbg !13
  %i130 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1416, !dbg !13
  %.idx1417 = mul nsw i64 %address.local524, -1280, !dbg !13
  %i131 = getelementptr i8, ptr addrspace(3) %i130, i64 %.idx1417, !dbg !13
  %i132 = getelementptr [2 x i8], ptr addrspace(3) %i131, i64 %address.local562, !dbg !13
  %i133 = getelementptr [2 x i8], ptr addrspace(3) %i132, i64 %address.local566, !dbg !13
  %i134 = getelementptr [2 x i8], ptr addrspace(3) %i133, i64 %address.local569, !dbg !13
  %address.local575 = getelementptr [2 x i8], ptr addrspace(3) %i134, i64 %address.local573, !dbg !13
  %i135 = add nsw i64 %.idx1417, %.idx1416, !dbg !13
  %.idx1992 = shl nuw nsw i64 %address.local562, 1, !dbg !13
  %i136 = or disjoint i64 %i135, %.idx1992, !dbg !13
  %i137 = or disjoint i64 %address.local566, %address.local569, !dbg !13
  %i138 = shl nsw i64 %i137, 1, !dbg !13
  %address.local575.idx = mul nuw nsw i64 %address.local572, 2032, !dbg !13
  %i139 = add nsw i64 %i136, %address.local575.idx, !dbg !13
  %i140 = add nsw i64 %i139, %i138, !dbg !13
  %i141 = lshr i64 %i140, 6, !dbg !13
  %i142 = and i64 %i141, 8, !dbg !13
  %i143 = xor i64 %i142, %i140, !dbg !13
  %i144 = sub i64 %i143, %i140, !dbg !13
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local575, i64 %i144, !dbg !13
  %i145 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13
  %lazy.base = extractvalue [2 x <1 x float>] %i119, 0, !dbg !13
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %lazy.base2077 = extractvalue [2 x <1 x float>] %i119, 1, !dbg !13
  %lazy.seed2078 = shufflevector <1 x float> %lazy.base2077, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %i146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i68, <4 x half> %i145, <4 x float> %lazy.seed) #8, !dbg !13
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !13
  %i147 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13
  %address.local586 = shl nuw nsw i32 %address.thread517, 1, !dbg !13
  %address.local587 = and i32 %address.local586, 24, !dbg !13
  %address.local588 = shl nuw nsw i32 %address.thread517, 8, !dbg !13
  %address.local589 = and i32 %address.local588, 768, !dbg !13
  %address.local590 = or disjoint i32 %address.local587, %address.local589, !dbg !13
  %address.local591 = shl nuw nsw i32 %address.thread517, 5, !dbg !13
  %address.local592 = and i32 %address.local591, 96, !dbg !13
  %address.local593 = or disjoint i32 %address.local590, %address.local592, !dbg !13
  %address.local594 = zext nneg i32 %address.local593 to i64, !dbg !13
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %i107) #8, !dbg !13, !frisk.register_lifetimes !5
  %address.local595 = lshr i32 %address.thread517, 1, !dbg !13
  %address.local596 = and i32 %address.local595, 1, !dbg !13
  %address.local597 = zext nneg i32 %address.local596 to i64, !dbg !13
  %address.local598 = add nsw i64 %address.base, %address.local597, !dbg !13
  %address.local599 = shl i64 %address.local598, 6, !dbg !13
  %address.local600 = sub i64 %address.local594, %address.local599, !dbg !13
  %i148 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local600, !dbg !13
  %i149 = getelementptr [2 x i8], ptr addrspace(3) %i148, i64 %address.local569, !dbg !13
  %i150 = getelementptr [2 x i8], ptr addrspace(3) %i149, i64 %address.local573, !dbg !13
  %address.local604 = getelementptr i8, ptr addrspace(3) %i150, i64 64, !dbg !13
  %.idx19951998 = add i64 %address.local600, %address.local569, !dbg !13
  %i151 = shl i64 %.idx19951998, 1, !dbg !13
  %i152 = add nuw nsw i64 %address.local575.idx, 64, !dbg !13
  %i153 = add i64 %i152, %i151, !dbg !13
  %i154 = lshr i64 %i153, 6, !dbg !13
  %i155 = and i64 %i154, 8, !dbg !13
  %i156 = xor i64 %i155, %i153, !dbg !13
  %i157 = sub i64 %i156, %i153, !dbg !13
  %kdma.swizzled1866 = getelementptr i8, ptr addrspace(3) %address.local604, i64 %i157, !dbg !13
  %i158 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1866, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1870 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 128, !dbg !13
  %i159 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1867 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 4096, !dbg !13
  %i160 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1871 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 4224, !dbg !13
  %i161 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1868 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !13
  %i162 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1872 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !13
  %i163 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1869 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !13
  %i164 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !13, !frisk.k_lds_pair !18
  %kdma.gep1873 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !13
  %i165 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !13, !frisk.k_lds_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i147, <4 x float> %i146) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1874 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 1024, !dbg !13
  %i167 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !13, !frisk.k_lds_pair !19
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 1152, !dbg !13
  %i168 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !13, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i68, <4 x half> %i158, <4 x float> %lazy.seed) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1875 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 5120, !dbg !13
  %i170 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !13, !frisk.k_lds_pair !20
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 5248, !dbg !13
  %i171 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !13, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i160, <4 x float> %i169) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1876 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !13
  %i173 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !13
  %i174 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !13, !frisk.k_lds_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i68, <4 x half> %i162, <4 x float> %lazy.seed) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i164, <4 x float> %i175) #8, !dbg !13
  %i177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i68, <4 x half> %i159, <4 x float> %lazy.seed) #8, !dbg !13
  %i178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i161, <4 x float> %i177) #8, !dbg !13
  %i179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i68, <4 x half> %i163, <4 x float> %lazy.seed) #8, !dbg !13
  %i180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i165, <4 x float> %i179) #8, !dbg !13
  %i181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i68, <4 x half> %i167, <4 x float> %lazy.seed) #8, !dbg !13
  %i182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i170, <4 x float> %i181) #8, !dbg !13
  %i183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i68, <4 x half> %i173, <4 x float> %lazy.seed) #8, !dbg !13
  %i184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i174, <4 x float> %i183) #8, !dbg !13
  %i185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i68, <4 x half> %i168, <4 x float> %lazy.seed) #8, !dbg !13
  %i186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i171, <4 x float> %i185) #8, !dbg !13
  %i187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i145, <4 x float> %lazy.seed2078) #8, !dbg !13
  %i188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i147, <4 x float> %i187) #8, !dbg !13
  %i189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i158, <4 x float> %lazy.seed2078) #8, !dbg !13
  %i190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i160, <4 x float> %i189) #8, !dbg !13
  %i191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i162, <4 x float> %lazy.seed2078) #8, !dbg !13
  %i192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i164, <4 x float> %i191) #8, !dbg !13
  %i193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i159, <4 x float> %lazy.seed2078) #8, !dbg !13
  %i194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i161, <4 x float> %i193) #8, !dbg !13
  %i195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i163, <4 x float> %lazy.seed2078) #8, !dbg !13
  %i196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i165, <4 x float> %i195) #8, !dbg !13
  %i197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i167, <4 x float> %lazy.seed2078) #8, !dbg !13
  %i198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i170, <4 x float> %i197) #8, !dbg !13
  %i199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i173, <4 x float> %lazy.seed2078) #8, !dbg !13
  %i200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i174, <4 x float> %i199) #8, !dbg !13
  %i201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i168, <4 x float> %lazy.seed2078) #8, !dbg !13
  %i202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i171, <4 x float> %i201) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %i203 = or disjoint i64 %i124, 64, !dbg !14
  %kdma.source1699 = or disjoint i64 %i203, %k0.prime.kdma.source1516, !dbg !14
  %kdma.source1700 = trunc nuw nsw i64 %kdma.source1699 to i32, !dbg !14
  %kdma.source1701 = shl nuw nsw i32 %kdma.source1700, 1, !dbg !14
  %kdma.offset1702 = or disjoint i32 %kdma.source1701, %kdma.intra, !dbg !14
  %i204 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i54, i32 %kdma.offset1702, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %kdma.source1721 = or disjoint i64 %i203, %k0.prime.kdma.source1536, !dbg !14
  %kdma.source1722 = trunc nuw nsw i64 %kdma.source1721 to i32, !dbg !14
  %kdma.source1723 = shl nuw nsw i32 %kdma.source1722, 1, !dbg !14
  %kdma.offset1724 = or disjoint i32 %kdma.source1723, %kdma.intra.hi, !dbg !14
  %i205 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1541, i32 %kdma.offset1724, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %kdma.source1745 = or disjoint i64 %i203, %k0.prime.kdma.source1560, !dbg !14
  %kdma.source1746 = trunc nuw nsw i64 %kdma.source1745 to i32, !dbg !14
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !14
  %kdma.offset1748 = add nuw nsw i32 %kdma.source1747, %kdma.intra, !dbg !14
  %i206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i57, i32 %kdma.offset1748, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %kdma.source1769 = or disjoint i64 %i203, %k0.prime.kdma.source1584, !dbg !14
  %kdma.source1770 = trunc nuw nsw i64 %kdma.source1769 to i32, !dbg !14
  %kdma.source1771 = shl nuw nsw i32 %kdma.source1770, 1, !dbg !14
  %kdma.offset1772 = add nuw nsw i32 %kdma.source1771, %kdma.intra.hi, !dbg !14
  %i207 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1589, i32 %kdma.offset1772, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %kdma.gep1896 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !13
  %i208 = load <4 x half>, ptr addrspace(3) %kdma.gep1896, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1900 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !13
  %i209 = load <4 x half>, ptr addrspace(3) %kdma.gep1900, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1897 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !13
  %i210 = load <4 x half>, ptr addrspace(3) %kdma.gep1897, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1901 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !13
  %i211 = load <4 x half>, ptr addrspace(3) %kdma.gep1901, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1898 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 8192, !dbg !13
  %i212 = load <4 x half>, ptr addrspace(3) %kdma.gep1898, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1902 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 8320, !dbg !13
  %i213 = load <4 x half>, ptr addrspace(3) %kdma.gep1902, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1899 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 12288, !dbg !13
  %i214 = load <4 x half>, ptr addrspace(3) %kdma.gep1899, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1903 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 12416, !dbg !13
  %i215 = load <4 x half>, ptr addrspace(3) %kdma.gep1903, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1904 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !13
  %i216 = load <4 x half>, ptr addrspace(3) %kdma.gep1904, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep1908 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !13
  %i217 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !13, !frisk.k_lds_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i208, <4 x float> %i166) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1905 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !13
  %i219 = load <4 x half>, ptr addrspace(3) %kdma.gep1905, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep1909 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !13
  %i220 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !13, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i210, <4 x float> %i218) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1906 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 9216, !dbg !13
  %i222 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !13, !frisk.k_lds_pair !28
  %kdma.gep1910 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 9344, !dbg !13
  %i223 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !13, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i212, <4 x float> %i172) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1907 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 13312, !dbg !13
  %i225 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !13, !frisk.k_lds_pair !29
  %kdma.gep1911 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 13440, !dbg !13
  %i226 = load <4 x half>, ptr addrspace(3) %kdma.gep1911, align 8, !dbg !13, !frisk.k_lds_pair !29
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i214, <4 x float> %i224) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i209, <4 x float> %i176) #8, !dbg !13
  %i229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i211, <4 x float> %i228) #8, !dbg !13
  %i230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i213, <4 x float> %i178) #8, !dbg !13
  %i231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i215, <4 x float> %i230) #8, !dbg !13
  %i232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i216, <4 x float> %i180) #8, !dbg !13
  %i233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i219, <4 x float> %i232) #8, !dbg !13
  %i234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i222, <4 x float> %i182) #8, !dbg !13
  %i235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i225, <4 x float> %i234) #8, !dbg !13
  %i236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i217, <4 x float> %i184) #8, !dbg !13
  %i237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i220, <4 x float> %i236) #8, !dbg !13
  %i238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i223, <4 x float> %i186) #8, !dbg !13
  %i239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i226, <4 x float> %i238) #8, !dbg !13
  %i240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i208, <4 x float> %i188) #8, !dbg !13
  %i241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i210, <4 x float> %i240) #8, !dbg !13
  %i242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i212, <4 x float> %i190) #8, !dbg !13
  %i243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i214, <4 x float> %i242) #8, !dbg !13
  %i244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i209, <4 x float> %i192) #8, !dbg !13
  %i245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i211, <4 x float> %i244) #8, !dbg !13
  %i246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i213, <4 x float> %i194) #8, !dbg !13
  %i247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i215, <4 x float> %i246) #8, !dbg !13
  %i248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i216, <4 x float> %i196) #8, !dbg !13
  %i249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i219, <4 x float> %i248) #8, !dbg !13
  %i250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i222, <4 x float> %i198) #8, !dbg !13
  %i251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i225, <4 x float> %i250) #8, !dbg !13
  %i252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i217, <4 x float> %i200) #8, !dbg !13
  %i253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i220, <4 x float> %i252) #8, !dbg !13
  %i254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i223, <4 x float> %i202) #8, !dbg !13
  %i255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i226, <4 x float> %i254) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %i256 = or disjoint i64 %i124, 96, !dbg !14
  %kdma.source1791 = or disjoint i64 %i256, %k0.prime.kdma.source1516, !dbg !14
  %kdma.source1792 = trunc nuw nsw i64 %kdma.source1791 to i32, !dbg !14
  %kdma.source1793 = shl nuw nsw i32 %kdma.source1792, 1, !dbg !14
  %kdma.offset1794 = or disjoint i32 %kdma.source1793, %kdma.intra, !dbg !14
  %i257 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i60, i32 %kdma.offset1794, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %kdma.source1813 = or disjoint i64 %i256, %k0.prime.kdma.source1536, !dbg !14
  %kdma.source1814 = trunc nuw nsw i64 %kdma.source1813 to i32, !dbg !14
  %kdma.source1815 = shl nuw nsw i32 %kdma.source1814, 1, !dbg !14
  %kdma.offset1816 = or disjoint i32 %kdma.source1815, %kdma.intra.hi, !dbg !14
  %i258 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1633, i32 %kdma.offset1816, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %kdma.source1837 = or disjoint i64 %i256, %k0.prime.kdma.source1560, !dbg !14
  %kdma.source1838 = trunc nuw nsw i64 %kdma.source1837 to i32, !dbg !14
  %kdma.source1839 = shl nuw nsw i32 %kdma.source1838, 1, !dbg !14
  %kdma.offset1840 = add nuw nsw i32 %kdma.source1839, %kdma.intra, !dbg !14
  %i259 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i63, i32 %kdma.offset1840, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %kdma.source1861 = or disjoint i64 %i256, %k0.prime.kdma.source1584, !dbg !14
  %kdma.source1862 = trunc nuw nsw i64 %kdma.source1861 to i32, !dbg !14
  %kdma.source1863 = shl nuw nsw i32 %kdma.source1862, 1, !dbg !14
  %kdma.offset1864 = add nuw nsw i32 %kdma.source1863, %kdma.intra.hi, !dbg !14
  %i260 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1681, i32 %kdma.offset1864, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14
  %i261 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13, !frisk.k_lds_pair !30
  %i262 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !13, !frisk.k_lds_pair !30
  %i263 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13, !frisk.k_lds_pair !31
  %i264 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !13, !frisk.k_lds_pair !31
  %i265 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1866, align 8, !dbg !13, !frisk.k_lds_pair !32
  %i266 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !13, !frisk.k_lds_pair !32
  %i267 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !13, !frisk.k_lds_pair !33
  %i268 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !13, !frisk.k_lds_pair !33
  %i269 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !13, !frisk.k_lds_pair !34
  %i270 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !13, !frisk.k_lds_pair !34
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i261, <4 x float> %i221) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i272 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !13, !frisk.k_lds_pair !35
  %i273 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !13, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i263, <4 x float> %i271) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i275 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !13, !frisk.k_lds_pair !36
  %i276 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !13, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i265, <4 x float> %i227) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i278 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !13, !frisk.k_lds_pair !37
  %i279 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !13, !frisk.k_lds_pair !37
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i267, <4 x float> %i277) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %i281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i262, <4 x float> %i229) #8, !dbg !13
  %i282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i264, <4 x float> %i281) #8, !dbg !13
  %i283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i266, <4 x float> %i231) #8, !dbg !13
  %i284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i268, <4 x float> %i283) #8, !dbg !13
  %i285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i269, <4 x float> %i233) #8, !dbg !13
  %i286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i272, <4 x float> %i285) #8, !dbg !13
  %i287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i275, <4 x float> %i235) #8, !dbg !13
  %i288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i278, <4 x float> %i287) #8, !dbg !13
  %i289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i270, <4 x float> %i237) #8, !dbg !13
  %i290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i273, <4 x float> %i289) #8, !dbg !13
  %i291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i276, <4 x float> %i239) #8, !dbg !13
  %i292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i279, <4 x float> %i291) #8, !dbg !13
  %i293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i92, <4 x half> %i261, <4 x float> %i241) #8, !dbg !13
  %i294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i93, <4 x half> %i263, <4 x float> %i293) #8, !dbg !13
  %i295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i92, <4 x half> %i265, <4 x float> %i243) #8, !dbg !13
  %i296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i93, <4 x half> %i267, <4 x float> %i295) #8, !dbg !13
  %i297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i92, <4 x half> %i262, <4 x float> %i245) #8, !dbg !13
  %i298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i93, <4 x half> %i264, <4 x float> %i297) #8, !dbg !13
  %i299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i92, <4 x half> %i266, <4 x float> %i247) #8, !dbg !13
  %i300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i93, <4 x half> %i268, <4 x float> %i299) #8, !dbg !13
  %i301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i92, <4 x half> %i269, <4 x float> %i249) #8, !dbg !13
  %i302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i93, <4 x half> %i272, <4 x float> %i301) #8, !dbg !13
  %i303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i92, <4 x half> %i275, <4 x float> %i251) #8, !dbg !13
  %i304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i93, <4 x half> %i278, <4 x float> %i303) #8, !dbg !13
  %i305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i92, <4 x half> %i270, <4 x float> %i253) #8, !dbg !13
  %i306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i93, <4 x half> %i273, <4 x float> %i305) #8, !dbg !13
  %i307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i92, <4 x half> %i276, <4 x float> %i255) #8, !dbg !13
  %i308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i93, <4 x half> %i279, <4 x float> %i307) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local933 = shl nuw nsw i64 %address.local518, 5, !dbg !38
  %address.local934 = and i64 %address.local933, 8064, !dbg !38
  %address.local935 = shl nuw nsw i64 %address.local518, 3, !dbg !9
  %address.local936 = shl nuw nsw i64 %address.local524, 5, !dbg !38
  %address.local937 = sub nsw i64 %address.local935, %address.local936, !dbg !38
  %address.local938 = add nsw i64 %address.local937, %address.local934, !dbg !38
  %.idx14261427 = add nsw i64 %address.local938, %i124, !dbg !38
  %.idx14261427.tr = trunc nsw i64 %.idx14261427 to i32, !dbg !38
  %dma.buffer.offset1361 = shl nsw i32 %.idx14261427.tr, 1, !dbg !38
  %i309 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i54, i32 %dma.buffer.offset1361, <4 x i32> %dma.buffer.rsrc1362) #9, !dbg !38
  %address.local940 = shl nuw nsw i32 %address.thread517, 3, !dbg !38
  %address.local941 = and i32 %address.local940, 24, !dbg !38
  %address.local943 = and i32 %address.local591, 8064, !dbg !38
  %address.local944 = or disjoint i32 %address.local943, %address.local941, !dbg !38
  %address.local945 = or disjoint i32 %address.local944, 8192, !dbg !38
  %i310 = trunc nuw nsw i64 %i124 to i32, !dbg !38
  %.idx14281429.tr = or disjoint i32 %address.local945, %i310, !dbg !38
  %dma.buffer.offset1359 = shl nuw nsw i32 %.idx14281429.tr, 1, !dbg !38
  %i311 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i57, i32 %dma.buffer.offset1359, <4 x i32> %dma.buffer.rsrc1362) #9, !dbg !38
  %i312 = load <4 x half>, ptr addrspace(3) %kdma.gep1896, align 8, !dbg !13, !frisk.k_tail_pair !15
  %i313 = load <4 x half>, ptr addrspace(3) %kdma.gep1900, align 8, !dbg !13, !frisk.k_tail_pair !15
  %i314 = load <4 x half>, ptr addrspace(3) %kdma.gep1897, align 8, !dbg !13, !frisk.k_tail_pair !16
  %i315 = load <4 x half>, ptr addrspace(3) %kdma.gep1901, align 8, !dbg !13, !frisk.k_tail_pair !16
  %i316 = load <4 x half>, ptr addrspace(3) %kdma.gep1898, align 8, !dbg !13, !frisk.k_tail_pair !17
  %i317 = load <4 x half>, ptr addrspace(3) %kdma.gep1902, align 8, !dbg !13, !frisk.k_tail_pair !17
  %i318 = load <4 x half>, ptr addrspace(3) %kdma.gep1899, align 8, !dbg !13, !frisk.k_tail_pair !18
  %i319 = load <4 x half>, ptr addrspace(3) %kdma.gep1903, align 8, !dbg !13, !frisk.k_tail_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %i320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i97, <4 x half> %i312, <4 x float> %i274) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %i321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i99, <4 x half> %i314, <4 x float> %i320) #8, !dbg !13
  %i322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i97, <4 x half> %i316, <4 x float> %i280) #8, !dbg !13
  %i323 = extractelement <4 x float> %i321, i64 0, !dbg !13
  %i324 = tail call float @llvm.maxnum.f32(float %i323, float -inf), !dbg !7, !frisk.local_reduction !5
  %i325 = extractelement <4 x float> %i321, i64 1, !dbg !13
  %i326 = extractelement <4 x float> %i321, i64 2, !dbg !13
  %i327 = extractelement <4 x float> %i321, i64 3, !dbg !13
  %i328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i99, <4 x half> %i318, <4 x float> %i322) #8, !dbg !13
  %i329 = load <4 x half>, ptr addrspace(3) %kdma.gep1904, align 8, !dbg !13, !frisk.k_tail_pair !19
  %i330 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !13, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %i331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i97, <4 x half> %i313, <4 x float> %i282) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %i332 = extractelement <4 x float> %i328, i64 0, !dbg !13
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %i324, float %i332), !dbg !7, !frisk.local_reduction !5
  %i333 = extractelement <4 x float> %i328, i64 1, !dbg !13
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %i325, float %i333), !dbg !7, !frisk.local_reduction !5
  %i334 = extractelement <4 x float> %i328, i64 2, !dbg !13
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %i326, float %i334), !dbg !7, !frisk.local_reduction !5
  %i335 = extractelement <4 x float> %i328, i64 3, !dbg !13
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %i327, float %i335), !dbg !7, !frisk.local_reduction !5
  %i336 = load <4 x half>, ptr addrspace(3) %kdma.gep1905, align 8, !dbg !13, !frisk.k_tail_pair !20
  %i337 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !13, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %i338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i99, <4 x half> %i315, <4 x float> %i331) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %i339 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !13, !frisk.k_tail_pair !21
  %i340 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !13, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %i341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i97, <4 x half> %i317, <4 x float> %i284) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %i342 = extractelement <4 x float> %i338, i64 0, !dbg !13
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %i342), !dbg !7, !frisk.local_reduction !5
  %i343 = extractelement <4 x float> %i338, i64 1, !dbg !13
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %i343), !dbg !7, !frisk.local_reduction !5
  %i344 = extractelement <4 x float> %i338, i64 2, !dbg !13
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %i344), !dbg !7, !frisk.local_reduction !5
  %i345 = extractelement <4 x float> %i338, i64 3, !dbg !13
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %i345), !dbg !7, !frisk.local_reduction !5
  %i346 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !13, !frisk.k_tail_pair !22
  %i347 = load <4 x half>, ptr addrspace(3) %kdma.gep1911, align 8, !dbg !13, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %i348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i99, <4 x half> %i319, <4 x float> %i341) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %i349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i97, <4 x half> %i329, <4 x float> %i286) #8, !dbg !13
  %i350 = extractelement <4 x float> %i348, i64 0, !dbg !13
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %i350), !dbg !7, !frisk.local_reduction !5
  %i351 = extractelement <4 x float> %i348, i64 1, !dbg !13
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %i351), !dbg !7, !frisk.local_reduction !5
  %i352 = extractelement <4 x float> %i348, i64 2, !dbg !13
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %i352), !dbg !7, !frisk.local_reduction !5
  %i353 = extractelement <4 x float> %i348, i64 3, !dbg !13
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %i353), !dbg !7, !frisk.local_reduction !5
  %i354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i99, <4 x half> %i336, <4 x float> %i349) #8, !dbg !13
  %i355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i97, <4 x half> %i339, <4 x float> %i288) #8, !dbg !13
  %i356 = extractelement <4 x float> %i354, i64 0, !dbg !13
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %i356), !dbg !7, !frisk.local_reduction !5
  %i357 = extractelement <4 x float> %i354, i64 1, !dbg !13
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %i357), !dbg !7, !frisk.local_reduction !5
  %i358 = extractelement <4 x float> %i354, i64 2, !dbg !13
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %i358), !dbg !7, !frisk.local_reduction !5
  %i359 = extractelement <4 x float> %i354, i64 3, !dbg !13
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %i359), !dbg !7, !frisk.local_reduction !5
  %i360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i99, <4 x half> %i346, <4 x float> %i355) #8, !dbg !13
  %i361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i97, <4 x half> %i330, <4 x float> %i290) #8, !dbg !13
  %i362 = extractelement <4 x float> %i360, i64 0, !dbg !13
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %i362), !dbg !7, !frisk.local_reduction !5
  %i363 = extractelement <4 x float> %i360, i64 1, !dbg !13
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %i363), !dbg !7, !frisk.local_reduction !5
  %i364 = extractelement <4 x float> %i360, i64 2, !dbg !13
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %i364), !dbg !7, !frisk.local_reduction !5
  %i365 = extractelement <4 x float> %i360, i64 3, !dbg !13
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %i365), !dbg !7, !frisk.local_reduction !5
  %i366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i99, <4 x half> %i337, <4 x float> %i361) #8, !dbg !13
  %i367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i97, <4 x half> %i340, <4 x float> %i292) #8, !dbg !13
  %i368 = extractelement <4 x float> %i366, i64 0, !dbg !13
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %i368), !dbg !7, !frisk.local_reduction !5
  %i369 = extractelement <4 x float> %i366, i64 1, !dbg !13
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %i369), !dbg !7, !frisk.local_reduction !5
  %i370 = extractelement <4 x float> %i366, i64 2, !dbg !13
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %i370), !dbg !7, !frisk.local_reduction !5
  %i371 = extractelement <4 x float> %i366, i64 3, !dbg !13
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %i371), !dbg !7, !frisk.local_reduction !5
  %i372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i99, <4 x half> %i347, <4 x float> %i367) #8, !dbg !13
  %i373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i100, <4 x half> %i312, <4 x float> %i294) #8, !dbg !13
  %i374 = extractelement <4 x float> %i372, i64 0, !dbg !13
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %i374), !dbg !7, !frisk.local_reduction !5
  %i375 = extractelement <4 x float> %i372, i64 1, !dbg !13
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce176, float %i375), !dbg !7, !frisk.local_reduction !5
  %i376 = extractelement <4 x float> %i372, i64 2, !dbg !13
  %local.reduce181 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %i376), !dbg !7, !frisk.local_reduction !5
  %i377 = extractelement <4 x float> %i372, i64 3, !dbg !13
  %local.reduce182 = tail call float @llvm.maxnum.f32(float %local.reduce178, float %i377), !dbg !7, !frisk.local_reduction !5
  %local.reduce183 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !7, !frisk.local_reduction !5
  %local.reduce184 = tail call float @llvm.maxnum.f32(float %local.reduce181, float %local.reduce182), !dbg !7, !frisk.local_reduction !5
  %i378 = tail call float @llvm.maxnum.f32(float %local.reduce183, float %local.reduce184), !dbg !7, !frisk.local_reduction !5
  %i379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i101, <4 x half> %i314, <4 x float> %i373) #8, !dbg !13
  %i380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i100, <4 x half> %i316, <4 x float> %i296) #8, !dbg !13
  %i381 = extractelement <4 x float> %i379, i64 0, !dbg !13
  %i382 = extractelement <4 x float> %i379, i64 1, !dbg !13
  %i383 = extractelement <4 x float> %i379, i64 2, !dbg !13
  %i384 = extractelement <4 x float> %i379, i64 3, !dbg !13
  %i385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i101, <4 x half> %i318, <4 x float> %i380) #8, !dbg !13
  %i386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i100, <4 x half> %i313, <4 x float> %i298) #8, !dbg !13
  %i387 = extractelement <4 x float> %i385, i64 0, !dbg !13
  %i388 = extractelement <4 x float> %i385, i64 1, !dbg !13
  %i389 = extractelement <4 x float> %i385, i64 2, !dbg !13
  %i390 = extractelement <4 x float> %i385, i64 3, !dbg !13
  %i391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i101, <4 x half> %i315, <4 x float> %i386) #8, !dbg !13
  %i392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i100, <4 x half> %i317, <4 x float> %i300) #8, !dbg !13
  %i393 = extractelement <4 x float> %i391, i64 0, !dbg !13
  %i394 = extractelement <4 x float> %i391, i64 1, !dbg !13
  %i395 = extractelement <4 x float> %i391, i64 2, !dbg !13
  %i396 = extractelement <4 x float> %i391, i64 3, !dbg !13
  %i397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i101, <4 x half> %i319, <4 x float> %i392) #8, !dbg !13
  %i398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i100, <4 x half> %i329, <4 x float> %i302) #8, !dbg !13
  %i399 = extractelement <4 x float> %i397, i64 0, !dbg !13
  %i400 = extractelement <4 x float> %i397, i64 1, !dbg !13
  %i401 = extractelement <4 x float> %i397, i64 2, !dbg !13
  %i402 = extractelement <4 x float> %i397, i64 3, !dbg !13
  %i403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i101, <4 x half> %i336, <4 x float> %i398) #8, !dbg !13
  %i404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i100, <4 x half> %i339, <4 x float> %i304) #8, !dbg !13
  %i405 = extractelement <4 x float> %i403, i64 0, !dbg !13
  %i406 = extractelement <4 x float> %i403, i64 1, !dbg !13
  %i407 = extractelement <4 x float> %i403, i64 2, !dbg !13
  %i408 = extractelement <4 x float> %i403, i64 3, !dbg !13
  %i409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i101, <4 x half> %i346, <4 x float> %i404) #8, !dbg !13
  %i410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i100, <4 x half> %i330, <4 x float> %i306) #8, !dbg !13
  %i411 = extractelement <4 x float> %i409, i64 0, !dbg !13
  %i412 = extractelement <4 x float> %i409, i64 1, !dbg !13
  %i413 = extractelement <4 x float> %i409, i64 2, !dbg !13
  %i414 = extractelement <4 x float> %i409, i64 3, !dbg !13
  %i415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i101, <4 x half> %i337, <4 x float> %i410) #8, !dbg !13
  %i416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i100, <4 x half> %i340, <4 x float> %i308) #8, !dbg !13
  %i417 = extractelement <4 x float> %i415, i64 0, !dbg !13
  %i418 = extractelement <4 x float> %i415, i64 1, !dbg !13
  %i419 = extractelement <4 x float> %i415, i64 2, !dbg !13
  %i420 = extractelement <4 x float> %i415, i64 3, !dbg !13
  %i421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i101, <4 x half> %i347, <4 x float> %i416) #8, !dbg !13
  %i422 = extractelement <4 x float> %i421, i64 0, !dbg !13
  %i423 = extractelement <4 x float> %i421, i64 1, !dbg !13
  %i424 = extractelement <4 x float> %i421, i64 2, !dbg !13
  %i425 = extractelement <4 x float> %i421, i64 3, !dbg !13
  %i426 = bitcast float %i378 to i32, !dbg !7
  %i427 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i426), !dbg !7
  %i428 = bitcast i32 %i427 to float, !dbg !7
  %i429 = tail call float @llvm.maxnum.f32(float %i378, float %i428), !dbg !7
  %i430 = bitcast float %i429 to i32, !dbg !7
  %i431 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i430), !dbg !7
  %i432 = bitcast i32 %i431 to float, !dbg !7
  %i433 = tail call float @llvm.maxnum.f32(float %i429, float %i432), !dbg !7
  %i434 = tail call float @llvm.maxnum.f32(float %i381, float -inf), !dbg !7, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %i434, float %i387), !dbg !7, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %i382, float %i388), !dbg !7, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %i383, float %i389), !dbg !7, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %i384, float %i390), !dbg !7, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %i393), !dbg !7, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %i394), !dbg !7, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %i395), !dbg !7, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %i396), !dbg !7, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %i399), !dbg !7, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %i400), !dbg !7, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %i401), !dbg !7, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %i402), !dbg !7, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %i405), !dbg !7, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %i406), !dbg !7, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %i407), !dbg !7, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %i408), !dbg !7, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %i411), !dbg !7, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %i412), !dbg !7, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %i413), !dbg !7, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %i414), !dbg !7, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %i417), !dbg !7, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %i418), !dbg !7, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %i419), !dbg !7, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %i420), !dbg !7, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %i422), !dbg !7, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce145, float %i423), !dbg !7, !frisk.local_reduction !5
  %local.reduce150 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %i424), !dbg !7, !frisk.local_reduction !5
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %local.reduce147, float %i425), !dbg !7, !frisk.local_reduction !5
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !7, !frisk.local_reduction !5
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %local.reduce150, float %local.reduce151), !dbg !7, !frisk.local_reduction !5
  %i435 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %local.reduce153), !dbg !7, !frisk.local_reduction !5
  %i436 = bitcast float %i435 to i32, !dbg !7
  %i437 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i436), !dbg !7
  %i438 = bitcast i32 %i437 to float, !dbg !7
  %i439 = tail call float @llvm.maxnum.f32(float %i435, float %i438), !dbg !7
  %i440 = bitcast float %i439 to i32, !dbg !7
  %i441 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i440), !dbg !7
  %i442 = bitcast i32 %i441 to float, !dbg !7
  %i443 = tail call float @llvm.maxnum.f32(float %i439, float %i442), !dbg !7
  %lazy.peak = tail call float @llvm.maxnum.f32(float %i433, float %i443), !dbg !39
  %lazy.over = fcmp ogt float %lazy.peak, 8.000000e+00, !dbg !39
  %lazy.mask = tail call i64 @llvm.amdgcn.ballot.i64(i1 %lazy.over), !dbg !39
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !39
  %lazy.first = icmp eq i64 %i122, 0, !dbg !39
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !39
  br i1 %lazy.take, label %lazy.slow, label %lazy.join, !dbg !39

lazy.slow:                                        ; preds = %bb115
  %lazy.positive = tail call float @llvm.maxnum.f32(float %i433, float 0.000000e+00), !dbg !39
  %lazy.delta = select i1 %lazy.first, float %i433, float %lazy.positive, !dbg !39
  %lazy.negated = fsub float 0.000000e+00, %lazy.delta, !dbg !39
  %lazy.exp = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated), !dbg !39
  %lazy.alpha = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp, !dbg !39
  %lazy.positive2079 = tail call float @llvm.maxnum.f32(float %i443, float 0.000000e+00), !dbg !39
  %lazy.delta2080 = select i1 %lazy.first, float %i443, float %lazy.positive2079, !dbg !39
  %lazy.negated2081 = fsub float 0.000000e+00, %lazy.delta2080, !dbg !39
  %lazy.exp2082 = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated2081), !dbg !39
  %lazy.alpha2083 = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp2082, !dbg !39
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted = fsub <4 x float> %i321, %lazy.delta.v, !dbg !39
  %lazy.shifted2086 = fsub <4 x float> %i328, %lazy.delta.v, !dbg !39
  %lazy.shifted2089 = fsub <4 x float> %i338, %lazy.delta.v, !dbg !39
  %lazy.shifted2092 = fsub <4 x float> %i348, %lazy.delta.v, !dbg !39
  %lazy.shifted2095 = fsub <4 x float> %i354, %lazy.delta.v, !dbg !39
  %lazy.shifted2098 = fsub <4 x float> %i360, %lazy.delta.v, !dbg !39
  %lazy.shifted2101 = fsub <4 x float> %i366, %lazy.delta.v, !dbg !39
  %lazy.shifted2104 = fsub <4 x float> %i372, %lazy.delta.v, !dbg !39
  %lazy.delta.v.lane2105 = insertelement <1 x float> poison, float %lazy.delta2080, i64 0, !dbg !39
  %lazy.delta.v2106 = shufflevector <1 x float> %lazy.delta.v.lane2105, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2107 = fsub <4 x float> %i379, %lazy.delta.v2106, !dbg !39
  %lazy.shifted2110 = fsub <4 x float> %i385, %lazy.delta.v2106, !dbg !39
  %lazy.shifted2113 = fsub <4 x float> %i391, %lazy.delta.v2106, !dbg !39
  %lazy.shifted2116 = fsub <4 x float> %i397, %lazy.delta.v2106, !dbg !39
  %lazy.shifted2119 = fsub <4 x float> %i403, %lazy.delta.v2106, !dbg !39
  %lazy.shifted2122 = fsub <4 x float> %i409, %lazy.delta.v2106, !dbg !39
  %lazy.shifted2125 = fsub <4 x float> %i415, %lazy.delta.v2106, !dbg !39
  %lazy.shifted2128 = fsub <4 x float> %i421, %lazy.delta.v2106, !dbg !39
  %lazy.state = extractvalue [2 x <1 x float>] %i120, 0, !dbg !39
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !39
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !39
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !39
  %lazy.state2129 = extractvalue [2 x <1 x float>] %i120, 1, !dbg !39
  %lazy.alpha.v.lane2130 = insertelement <1 x float> poison, float %lazy.alpha2083, i64 0, !dbg !39
  %lazy.scaled2131 = fmul <1 x float> %lazy.state2129, %lazy.alpha.v.lane2130, !dbg !39
  %lazy.state.next2132 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled2131, 1, !dbg !39
  %lazy.state2133 = extractvalue [2 x <8 x float>] %i116, 0, !dbg !39
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2135 = fmul <8 x float> %lazy.state2133, %lazy.alpha.v, !dbg !39
  %lazy.state.next2136 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2135, 0, !dbg !39
  %lazy.state2137 = extractvalue [2 x <8 x float>] %i116, 1, !dbg !39
  %lazy.alpha.v2139 = shufflevector <1 x float> %lazy.alpha.v.lane2130, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2140 = fmul <8 x float> %lazy.state2137, %lazy.alpha.v2139, !dbg !39
  %lazy.state.next2141 = insertvalue [2 x <8 x float>] %lazy.state.next2136, <8 x float> %lazy.scaled2140, 1, !dbg !39
  %lazy.state2142 = extractvalue [2 x <8 x float>] %i117, 0, !dbg !39
  %lazy.scaled2145 = fmul <8 x float> %lazy.state2142, %lazy.alpha.v, !dbg !39
  %lazy.state.next2146 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2145, 0, !dbg !39
  %lazy.state2147 = extractvalue [2 x <8 x float>] %i117, 1, !dbg !39
  %lazy.scaled2150 = fmul <8 x float> %lazy.state2147, %lazy.alpha.v2139, !dbg !39
  %lazy.state.next2151 = insertvalue [2 x <8 x float>] %lazy.state.next2146, <8 x float> %lazy.scaled2150, 1, !dbg !39
  %lazy.state2152 = extractvalue [2 x <8 x float>] %i118, 0, !dbg !39
  %lazy.scaled2155 = fmul <8 x float> %lazy.state2152, %lazy.alpha.v, !dbg !39
  %lazy.state.next2156 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2155, 0, !dbg !39
  %lazy.state2157 = extractvalue [2 x <8 x float>] %i118, 1, !dbg !39
  %lazy.scaled2160 = fmul <8 x float> %lazy.state2157, %lazy.alpha.v2139, !dbg !39
  %lazy.state.next2161 = insertvalue [2 x <8 x float>] %lazy.state.next2156, <8 x float> %lazy.scaled2160, 1, !dbg !39
  %lazy.state2162 = extractvalue [2 x <8 x float>] %i121, 0, !dbg !39
  %lazy.scaled2165 = fmul <8 x float> %lazy.state2162, %lazy.alpha.v, !dbg !39
  %lazy.state.next2166 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2165, 0, !dbg !39
  %lazy.state2167 = extractvalue [2 x <8 x float>] %i121, 1, !dbg !39
  %lazy.scaled2170 = fmul <8 x float> %lazy.state2167, %lazy.alpha.v2139, !dbg !39
  %lazy.state.next2171 = insertvalue [2 x <8 x float>] %lazy.state.next2166, <8 x float> %lazy.scaled2170, 1, !dbg !39
  %lazy.base.next = fsub <1 x float> %lazy.base, %lazy.delta.v.lane, !dbg !39
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !39
  %lazy.base.next2174 = fsub <1 x float> %lazy.base2077, %lazy.delta.v.lane2105, !dbg !39
  %lazy.base.state2175 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2174, 1, !dbg !39
  %.pre = extractelement <4 x float> %lazy.shifted, i64 0, !dbg !40
  %.pre25 = extractelement <4 x float> %lazy.shifted, i64 1, !dbg !40
  %.pre27 = extractelement <4 x float> %lazy.shifted, i64 2, !dbg !40
  %.pre29 = extractelement <4 x float> %lazy.shifted, i64 3, !dbg !40
  %.pre31 = extractelement <4 x float> %lazy.shifted2107, i64 0, !dbg !40
  %.pre33 = extractelement <4 x float> %lazy.shifted2107, i64 1, !dbg !40
  %.pre35 = extractelement <4 x float> %lazy.shifted2107, i64 2, !dbg !40
  %.pre37 = extractelement <4 x float> %lazy.shifted2107, i64 3, !dbg !40
  %.pre39 = extractelement <4 x float> %lazy.shifted2086, i64 0, !dbg !40
  %.pre41 = extractelement <4 x float> %lazy.shifted2086, i64 1, !dbg !40
  %.pre43 = extractelement <4 x float> %lazy.shifted2086, i64 2, !dbg !40
  %.pre45 = extractelement <4 x float> %lazy.shifted2086, i64 3, !dbg !40
  %.pre47 = extractelement <4 x float> %lazy.shifted2110, i64 0, !dbg !40
  %.pre49 = extractelement <4 x float> %lazy.shifted2110, i64 1, !dbg !40
  %.pre51 = extractelement <4 x float> %lazy.shifted2110, i64 2, !dbg !40
  %.pre53 = extractelement <4 x float> %lazy.shifted2110, i64 3, !dbg !40
  %.pre55 = extractelement <4 x float> %lazy.shifted2089, i64 0, !dbg !40
  %.pre57 = extractelement <4 x float> %lazy.shifted2089, i64 1, !dbg !40
  %.pre59 = extractelement <4 x float> %lazy.shifted2089, i64 2, !dbg !40
  %.pre61 = extractelement <4 x float> %lazy.shifted2089, i64 3, !dbg !40
  %.pre63 = extractelement <4 x float> %lazy.shifted2113, i64 0, !dbg !40
  %.pre65 = extractelement <4 x float> %lazy.shifted2113, i64 1, !dbg !40
  %.pre67 = extractelement <4 x float> %lazy.shifted2113, i64 2, !dbg !40
  %.pre69 = extractelement <4 x float> %lazy.shifted2113, i64 3, !dbg !40
  %.pre71 = extractelement <4 x float> %lazy.shifted2092, i64 0, !dbg !40
  %.pre73 = extractelement <4 x float> %lazy.shifted2092, i64 1, !dbg !40
  %.pre75 = extractelement <4 x float> %lazy.shifted2092, i64 2, !dbg !40
  %.pre77 = extractelement <4 x float> %lazy.shifted2092, i64 3, !dbg !40
  %.pre79 = extractelement <4 x float> %lazy.shifted2116, i64 0, !dbg !40
  %.pre81 = extractelement <4 x float> %lazy.shifted2116, i64 1, !dbg !40
  %.pre83 = extractelement <4 x float> %lazy.shifted2116, i64 2, !dbg !40
  %.pre85 = extractelement <4 x float> %lazy.shifted2116, i64 3, !dbg !40
  %.pre87 = extractelement <4 x float> %lazy.shifted2095, i64 0, !dbg !40
  %.pre89 = extractelement <4 x float> %lazy.shifted2095, i64 1, !dbg !40
  %.pre91 = extractelement <4 x float> %lazy.shifted2095, i64 2, !dbg !40
  %.pre93 = extractelement <4 x float> %lazy.shifted2095, i64 3, !dbg !40
  %.pre95 = extractelement <4 x float> %lazy.shifted2119, i64 0, !dbg !40
  %.pre97 = extractelement <4 x float> %lazy.shifted2119, i64 1, !dbg !40
  %.pre99 = extractelement <4 x float> %lazy.shifted2119, i64 2, !dbg !40
  %.pre101 = extractelement <4 x float> %lazy.shifted2119, i64 3, !dbg !40
  %.pre103 = extractelement <4 x float> %lazy.shifted2098, i64 0, !dbg !40
  %.pre105 = extractelement <4 x float> %lazy.shifted2098, i64 1, !dbg !40
  %.pre107 = extractelement <4 x float> %lazy.shifted2098, i64 2, !dbg !40
  %.pre109 = extractelement <4 x float> %lazy.shifted2098, i64 3, !dbg !40
  %.pre111 = extractelement <4 x float> %lazy.shifted2122, i64 0, !dbg !40
  %.pre113 = extractelement <4 x float> %lazy.shifted2122, i64 1, !dbg !40
  %.pre115 = extractelement <4 x float> %lazy.shifted2122, i64 2, !dbg !40
  %.pre117 = extractelement <4 x float> %lazy.shifted2122, i64 3, !dbg !40
  %.pre119 = extractelement <4 x float> %lazy.shifted2101, i64 0, !dbg !40
  %.pre121 = extractelement <4 x float> %lazy.shifted2101, i64 1, !dbg !40
  %.pre123 = extractelement <4 x float> %lazy.shifted2101, i64 2, !dbg !40
  %.pre125 = extractelement <4 x float> %lazy.shifted2101, i64 3, !dbg !40
  %.pre127 = extractelement <4 x float> %lazy.shifted2125, i64 0, !dbg !40
  %.pre129 = extractelement <4 x float> %lazy.shifted2125, i64 1, !dbg !40
  %.pre131 = extractelement <4 x float> %lazy.shifted2125, i64 2, !dbg !40
  %.pre133 = extractelement <4 x float> %lazy.shifted2125, i64 3, !dbg !40
  %.pre135 = extractelement <4 x float> %lazy.shifted2104, i64 0, !dbg !40
  %.pre137 = extractelement <4 x float> %lazy.shifted2104, i64 1, !dbg !40
  %.pre139 = extractelement <4 x float> %lazy.shifted2104, i64 2, !dbg !40
  %.pre141 = extractelement <4 x float> %lazy.shifted2104, i64 3, !dbg !40
  %.pre143 = extractelement <4 x float> %lazy.shifted2128, i64 0, !dbg !40
  %.pre145 = extractelement <4 x float> %lazy.shifted2128, i64 1, !dbg !40
  %.pre147 = extractelement <4 x float> %lazy.shifted2128, i64 2, !dbg !40
  %.pre149 = extractelement <4 x float> %lazy.shifted2128, i64 3, !dbg !40
  br label %lazy.join, !dbg !39

lazy.join:                                        ; preds = %lazy.slow, %bb115
  %.pre-phi150 = phi float [ %.pre149, %lazy.slow ], [ %i425, %bb115 ], !dbg !40
  %.pre-phi148 = phi float [ %.pre147, %lazy.slow ], [ %i424, %bb115 ], !dbg !40
  %.pre-phi146 = phi float [ %.pre145, %lazy.slow ], [ %i423, %bb115 ], !dbg !40
  %.pre-phi144 = phi float [ %.pre143, %lazy.slow ], [ %i422, %bb115 ], !dbg !40
  %.pre-phi142 = phi float [ %.pre141, %lazy.slow ], [ %i377, %bb115 ], !dbg !40
  %.pre-phi140 = phi float [ %.pre139, %lazy.slow ], [ %i376, %bb115 ], !dbg !40
  %.pre-phi138 = phi float [ %.pre137, %lazy.slow ], [ %i375, %bb115 ], !dbg !40
  %.pre-phi136 = phi float [ %.pre135, %lazy.slow ], [ %i374, %bb115 ], !dbg !40
  %.pre-phi134 = phi float [ %.pre133, %lazy.slow ], [ %i420, %bb115 ], !dbg !40
  %.pre-phi132 = phi float [ %.pre131, %lazy.slow ], [ %i419, %bb115 ], !dbg !40
  %.pre-phi130 = phi float [ %.pre129, %lazy.slow ], [ %i418, %bb115 ], !dbg !40
  %.pre-phi128 = phi float [ %.pre127, %lazy.slow ], [ %i417, %bb115 ], !dbg !40
  %.pre-phi126 = phi float [ %.pre125, %lazy.slow ], [ %i371, %bb115 ], !dbg !40
  %.pre-phi124 = phi float [ %.pre123, %lazy.slow ], [ %i370, %bb115 ], !dbg !40
  %.pre-phi122 = phi float [ %.pre121, %lazy.slow ], [ %i369, %bb115 ], !dbg !40
  %.pre-phi120 = phi float [ %.pre119, %lazy.slow ], [ %i368, %bb115 ], !dbg !40
  %.pre-phi118 = phi float [ %.pre117, %lazy.slow ], [ %i414, %bb115 ], !dbg !40
  %.pre-phi116 = phi float [ %.pre115, %lazy.slow ], [ %i413, %bb115 ], !dbg !40
  %.pre-phi114 = phi float [ %.pre113, %lazy.slow ], [ %i412, %bb115 ], !dbg !40
  %.pre-phi112 = phi float [ %.pre111, %lazy.slow ], [ %i411, %bb115 ], !dbg !40
  %.pre-phi110 = phi float [ %.pre109, %lazy.slow ], [ %i365, %bb115 ], !dbg !40
  %.pre-phi108 = phi float [ %.pre107, %lazy.slow ], [ %i364, %bb115 ], !dbg !40
  %.pre-phi106 = phi float [ %.pre105, %lazy.slow ], [ %i363, %bb115 ], !dbg !40
  %.pre-phi104 = phi float [ %.pre103, %lazy.slow ], [ %i362, %bb115 ], !dbg !40
  %.pre-phi102 = phi float [ %.pre101, %lazy.slow ], [ %i408, %bb115 ], !dbg !40
  %.pre-phi100 = phi float [ %.pre99, %lazy.slow ], [ %i407, %bb115 ], !dbg !40
  %.pre-phi98 = phi float [ %.pre97, %lazy.slow ], [ %i406, %bb115 ], !dbg !40
  %.pre-phi96 = phi float [ %.pre95, %lazy.slow ], [ %i405, %bb115 ], !dbg !40
  %.pre-phi94 = phi float [ %.pre93, %lazy.slow ], [ %i359, %bb115 ], !dbg !40
  %.pre-phi92 = phi float [ %.pre91, %lazy.slow ], [ %i358, %bb115 ], !dbg !40
  %.pre-phi90 = phi float [ %.pre89, %lazy.slow ], [ %i357, %bb115 ], !dbg !40
  %.pre-phi88 = phi float [ %.pre87, %lazy.slow ], [ %i356, %bb115 ], !dbg !40
  %.pre-phi86 = phi float [ %.pre85, %lazy.slow ], [ %i402, %bb115 ], !dbg !40
  %.pre-phi84 = phi float [ %.pre83, %lazy.slow ], [ %i401, %bb115 ], !dbg !40
  %.pre-phi82 = phi float [ %.pre81, %lazy.slow ], [ %i400, %bb115 ], !dbg !40
  %.pre-phi80 = phi float [ %.pre79, %lazy.slow ], [ %i399, %bb115 ], !dbg !40
  %.pre-phi78 = phi float [ %.pre77, %lazy.slow ], [ %i353, %bb115 ], !dbg !40
  %.pre-phi76 = phi float [ %.pre75, %lazy.slow ], [ %i352, %bb115 ], !dbg !40
  %.pre-phi74 = phi float [ %.pre73, %lazy.slow ], [ %i351, %bb115 ], !dbg !40
  %.pre-phi72 = phi float [ %.pre71, %lazy.slow ], [ %i350, %bb115 ], !dbg !40
  %.pre-phi70 = phi float [ %.pre69, %lazy.slow ], [ %i396, %bb115 ], !dbg !40
  %.pre-phi68 = phi float [ %.pre67, %lazy.slow ], [ %i395, %bb115 ], !dbg !40
  %.pre-phi66 = phi float [ %.pre65, %lazy.slow ], [ %i394, %bb115 ], !dbg !40
  %.pre-phi64 = phi float [ %.pre63, %lazy.slow ], [ %i393, %bb115 ], !dbg !40
  %.pre-phi62 = phi float [ %.pre61, %lazy.slow ], [ %i345, %bb115 ], !dbg !40
  %.pre-phi60 = phi float [ %.pre59, %lazy.slow ], [ %i344, %bb115 ], !dbg !40
  %.pre-phi58 = phi float [ %.pre57, %lazy.slow ], [ %i343, %bb115 ], !dbg !40
  %.pre-phi56 = phi float [ %.pre55, %lazy.slow ], [ %i342, %bb115 ], !dbg !40
  %.pre-phi54 = phi float [ %.pre53, %lazy.slow ], [ %i390, %bb115 ], !dbg !40
  %.pre-phi52 = phi float [ %.pre51, %lazy.slow ], [ %i389, %bb115 ], !dbg !40
  %.pre-phi50 = phi float [ %.pre49, %lazy.slow ], [ %i388, %bb115 ], !dbg !40
  %.pre-phi48 = phi float [ %.pre47, %lazy.slow ], [ %i387, %bb115 ], !dbg !40
  %.pre-phi46 = phi float [ %.pre45, %lazy.slow ], [ %i335, %bb115 ], !dbg !40
  %.pre-phi44 = phi float [ %.pre43, %lazy.slow ], [ %i334, %bb115 ], !dbg !40
  %.pre-phi42 = phi float [ %.pre41, %lazy.slow ], [ %i333, %bb115 ], !dbg !40
  %.pre-phi40 = phi float [ %.pre39, %lazy.slow ], [ %i332, %bb115 ], !dbg !40
  %.pre-phi38 = phi float [ %.pre37, %lazy.slow ], [ %i384, %bb115 ], !dbg !40
  %.pre-phi36 = phi float [ %.pre35, %lazy.slow ], [ %i383, %bb115 ], !dbg !40
  %.pre-phi34 = phi float [ %.pre33, %lazy.slow ], [ %i382, %bb115 ], !dbg !40
  %.pre-phi32 = phi float [ %.pre31, %lazy.slow ], [ %i381, %bb115 ], !dbg !40
  %.pre-phi30 = phi float [ %.pre29, %lazy.slow ], [ %i327, %bb115 ], !dbg !40
  %.pre-phi28 = phi float [ %.pre27, %lazy.slow ], [ %i326, %bb115 ], !dbg !40
  %.pre-phi26 = phi float [ %.pre25, %lazy.slow ], [ %i325, %bb115 ], !dbg !40
  %.pre-phi = phi float [ %.pre, %lazy.slow ], [ %i323, %bb115 ], !dbg !40
  %lazy.state2191 = phi [2 x <1 x float>] [ %lazy.state.next2132, %lazy.slow ], [ %i120, %bb115 ], !dbg !39
  %lazy.state2192 = phi [2 x <8 x float>] [ %lazy.state.next2141, %lazy.slow ], [ %i116, %bb115 ], !dbg !39
  %lazy.state2193 = phi [2 x <8 x float>] [ %lazy.state.next2151, %lazy.slow ], [ %i117, %bb115 ], !dbg !39
  %lazy.state2194 = phi [2 x <8 x float>] [ %lazy.state.next2161, %lazy.slow ], [ %i118, %bb115 ], !dbg !39
  %lazy.state2195 = phi [2 x <8 x float>] [ %lazy.state.next2171, %lazy.slow ], [ %i121, %bb115 ], !dbg !39
  %lazy.base.join = phi [2 x <1 x float>] [ %lazy.base.state2175, %lazy.slow ], [ %i119, %bb115 ], !dbg !39
  %i444 = extractvalue [2 x <1 x float>] %lazy.state2191, 0, !dbg !39
  %i445 = extractelement <1 x float> %i444, i64 0, !dbg !39
  %i446 = extractvalue [2 x <1 x float>] %lazy.state2191, 1, !dbg !39
  %i447 = extractelement <1 x float> %i446, i64 0, !dbg !39
  %i448 = extractvalue [2 x <8 x float>] %lazy.state2195, 0, !dbg !41
  %i449 = shufflevector <8 x float> %i448, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %i450 = shufflevector <8 x float> %i448, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %i451 = extractvalue [2 x <8 x float>] %lazy.state2195, 1, !dbg !41
  %i452 = shufflevector <8 x float> %i451, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %i453 = shufflevector <8 x float> %i451, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address = add nsw i64 %address.local938, %i125, !dbg !38
  %vdb.address1442 = trunc nsw i64 %vdb.address to i32, !dbg !38
  %vdb.address1443 = shl nsw i32 %vdb.address1442, 1, !dbg !38
  %vdb.address1448 = trunc nuw nsw i64 %i125 to i32, !dbg !38
  %vdb.address1449 = or disjoint i32 %address.local945, %vdb.address1448, !dbg !38
  %vdb.address1450 = shl nuw nsw i32 %vdb.address1449, 1, !dbg !38
  %i454 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1443, <4 x i32> %dma.buffer.rsrc1362) #9, !dbg !38
  %i455 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1455, i32 %vdb.address1450, <4 x i32> %dma.buffer.rsrc1362) #9, !dbg !38
  %address.local949 = and i32 %address.local940, 120, !dbg !42
  %address.local951 = and i32 %address.local591, 1536, !dbg !42
  %address.local952 = or disjoint i32 %address.local949, %address.local951, !dbg !42
  %address.local953 = zext nneg i32 %address.local952 to i64, !dbg !42
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local953, !dbg !42
  %i456 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local954, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i457 = bitcast <4 x i32> %i456 to <8 x half>, !dbg !42
  %address.local955 = and i64 %address.local518, 60, !dbg !42
  %address.local956 = add nuw nsw i64 %address.local955, %address.local518, !dbg !42
  %address.local958.idx = shl nuw nsw i64 %address.local956, 4, !dbg !42
  %address.local958 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local958.idx, !dbg !42
  %address.local959 = lshr i64 %address.local571, 4, !dbg !42
  %address.local961.idx = mul nuw nsw i64 %address.local959, 768, !dbg !42
  %address.local961 = getelementptr i8, ptr addrspace(3) %address.local958, i64 %address.local961.idx, !dbg !42
  %address.local963.idx = mul nsw i64 %address.local524, -64, !dbg !42
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local961, i64 %address.local963.idx, !dbg !42
  %address.local964 = getelementptr i8, ptr addrspace(3) %address.local963, i64 256, !dbg !42
  %i458 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i459 = bitcast <4 x i32> %i458 to <8 x half>, !dbg !42
  %address.local968 = getelementptr i8, ptr addrspace(3) %address.local963, i64 512, !dbg !42
  %i460 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local968, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i461 = bitcast <4 x i32> %i460 to <8 x half>, !dbg !42
  %address.local972 = getelementptr i8, ptr addrspace(3) %address.local963, i64 768, !dbg !42
  %i462 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local972, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i463 = bitcast <4 x i32> %i462 to <8 x half>, !dbg !42
  %address.local974 = getelementptr i8, ptr addrspace(3) %address.local954, i64 4096, !dbg !42
  %i464 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local974, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i465 = bitcast <4 x i32> %i464 to <8 x half>, !dbg !42
  %address.local978 = getelementptr i8, ptr addrspace(3) %address.local963, i64 4352, !dbg !42
  %i466 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local978, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i467 = bitcast <4 x i32> %i466 to <8 x half>, !dbg !42
  %address.local982 = getelementptr i8, ptr addrspace(3) %address.local963, i64 4608, !dbg !42
  %i468 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local982, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i469 = bitcast <4 x i32> %i468 to <8 x half>, !dbg !42
  %address.local986 = getelementptr i8, ptr addrspace(3) %address.local963, i64 4864, !dbg !42
  %i470 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local986, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i471 = shufflevector <8 x half> %i457, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i472 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi), !dbg !40
  %i473 = insertelement <4 x float> poison, float %i472, i64 0, !dbg !44
  %i474 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi26), !dbg !40
  %i475 = insertelement <4 x float> %i473, float %i474, i64 1, !dbg !44
  %i476 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi28), !dbg !40
  %i477 = insertelement <4 x float> %i475, float %i476, i64 2, !dbg !44
  %i478 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi30), !dbg !40
  %i479 = insertelement <4 x float> %i477, float %i478, i64 3, !dbg !44
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i479) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !44
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !44
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !44
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !44
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !44
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !44
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !44
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !44
  %i480 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !44
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i480) #8, !dbg !40, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i481 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i471, <4 x float> %i449) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i482 = shufflevector <8 x half> %i457, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i483 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i482, <4 x float> %i450) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i484 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi32), !dbg !40
  %i485 = insertelement <4 x float> poison, float %i484, i64 0, !dbg !44
  %i486 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi34), !dbg !40
  %i487 = insertelement <4 x float> %i485, float %i486, i64 1, !dbg !44
  %i488 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi36), !dbg !40
  %i489 = insertelement <4 x float> %i487, float %i488, i64 2, !dbg !44
  %i490 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi38), !dbg !40
  %i491 = insertelement <4 x float> %i489, float %i490, i64 3, !dbg !44
  %math.result404 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i491) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02196 = extractelement <4 x float> %math.result404, i64 0, !dbg !44
  %.cvtpack.f12197 = extractelement <4 x float> %math.result404, i64 1, !dbg !44
  %.cvtpack.f22198 = extractelement <4 x float> %math.result404, i64 2, !dbg !44
  %.cvtpack.f32199 = extractelement <4 x float> %math.result404, i64 3, !dbg !44
  %.cvtpack.p02200 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02196, float %.cvtpack.f12197) #12, !dbg !44
  %.cvtpack.p12201 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22198, float %.cvtpack.f32199) #12, !dbg !44
  %.cvtpack.v02202 = insertelement <2 x i32> poison, i32 %.cvtpack.p02200, i64 0, !dbg !44
  %.cvtpack.v12203 = insertelement <2 x i32> %.cvtpack.v02202, i32 %.cvtpack.p12201, i64 1, !dbg !44
  %i492 = bitcast <2 x i32> %.cvtpack.v12203 to <4 x half>, !dbg !44
  %math.packed405 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i492) #8, !dbg !40, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i493 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %i471, <4 x float> %i452) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i494 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %i482, <4 x float> %i453) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i495 = shufflevector <8 x half> %i459, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i496 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi40), !dbg !40
  %local.reduce93 = fadd float %i472, %i496, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %local.reduce93) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i497 = insertelement <4 x float> poison, float %i496, i64 0, !dbg !44
  %i498 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi42), !dbg !40
  %local.reduce94 = fadd float %i474, %i498, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1999 = tail call float asm sideeffect "", "=v,0"(float %local.reduce94) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i499 = insertelement <4 x float> %i497, float %i498, i64 1, !dbg !44
  %i500 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi44), !dbg !40
  %local.reduce95 = fadd float %i476, %i500, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2000 = tail call float asm sideeffect "", "=v,0"(float %local.reduce95) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i501 = insertelement <4 x float> %i499, float %i500, i64 2, !dbg !44
  %i502 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi46), !dbg !40
  %local.reduce96 = fadd float %i478, %i502, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2001 = tail call float asm sideeffect "", "=v,0"(float %local.reduce96) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i503 = insertelement <4 x float> %i501, float %i502, i64 3, !dbg !44
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i503) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02204 = extractelement <4 x float> %math.result390, i64 0, !dbg !44
  %.cvtpack.f12205 = extractelement <4 x float> %math.result390, i64 1, !dbg !44
  %.cvtpack.f22206 = extractelement <4 x float> %math.result390, i64 2, !dbg !44
  %.cvtpack.f32207 = extractelement <4 x float> %math.result390, i64 3, !dbg !44
  %.cvtpack.p02208 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02204, float %.cvtpack.f12205) #12, !dbg !44
  %.cvtpack.p12209 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22206, float %.cvtpack.f32207) #12, !dbg !44
  %.cvtpack.v02210 = insertelement <2 x i32> poison, i32 %.cvtpack.p02208, i64 0, !dbg !44
  %.cvtpack.v12211 = insertelement <2 x i32> %.cvtpack.v02210, i32 %.cvtpack.p12209, i64 1, !dbg !44
  %i504 = bitcast <2 x i32> %.cvtpack.v12211 to <4 x half>, !dbg !44
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i504) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i505 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i495, <4 x float> %i481) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i506 = shufflevector <8 x half> %i459, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i507 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i506, <4 x float> %i483) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i508 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi48), !dbg !40
  %local.reduce = fadd float %i484, %i508, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2002 = tail call float asm sideeffect "", "=v,0"(float %local.reduce) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i509 = insertelement <4 x float> poison, float %i508, i64 0, !dbg !44
  %i510 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi50), !dbg !40
  %local.reduce63 = fadd float %i486, %i510, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2003 = tail call float asm sideeffect "", "=v,0"(float %local.reduce63) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i511 = insertelement <4 x float> %i509, float %i510, i64 1, !dbg !44
  %i512 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi52), !dbg !40
  %local.reduce64 = fadd float %i488, %i512, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2004 = tail call float asm sideeffect "", "=v,0"(float %local.reduce64) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i513 = insertelement <4 x float> %i511, float %i512, i64 2, !dbg !44
  %i514 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi54), !dbg !40
  %local.reduce65 = fadd float %i490, %i514, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2005 = tail call float asm sideeffect "", "=v,0"(float %local.reduce65) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i515 = insertelement <4 x float> %i513, float %i514, i64 3, !dbg !44
  %math.result406 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i515) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02212 = extractelement <4 x float> %math.result406, i64 0, !dbg !44
  %.cvtpack.f12213 = extractelement <4 x float> %math.result406, i64 1, !dbg !44
  %.cvtpack.f22214 = extractelement <4 x float> %math.result406, i64 2, !dbg !44
  %.cvtpack.f32215 = extractelement <4 x float> %math.result406, i64 3, !dbg !44
  %.cvtpack.p02216 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02212, float %.cvtpack.f12213) #12, !dbg !44
  %.cvtpack.p12217 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22214, float %.cvtpack.f32215) #12, !dbg !44
  %.cvtpack.v02218 = insertelement <2 x i32> poison, i32 %.cvtpack.p02216, i64 0, !dbg !44
  %.cvtpack.v12219 = insertelement <2 x i32> %.cvtpack.v02218, i32 %.cvtpack.p12217, i64 1, !dbg !44
  %i516 = bitcast <2 x i32> %.cvtpack.v12219 to <4 x half>, !dbg !44
  %math.packed407 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i516) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i517 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %i495, <4 x float> %i493) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i518 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %i506, <4 x float> %i494) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i519 = shufflevector <8 x half> %i461, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i520 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi56), !dbg !40
  %local.reduce97 = fadd float %i520, %k0.sum.anchor, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2006 = tail call float asm sideeffect "", "=v,0"(float %local.reduce97) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i521 = insertelement <4 x float> poison, float %i520, i64 0, !dbg !44
  %i522 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi58), !dbg !40
  %local.reduce98 = fadd float %i522, %k0.sum.anchor1999, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2007 = tail call float asm sideeffect "", "=v,0"(float %local.reduce98) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i523 = insertelement <4 x float> %i521, float %i522, i64 1, !dbg !44
  %i524 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi60), !dbg !40
  %local.reduce99 = fadd float %i524, %k0.sum.anchor2000, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2008 = tail call float asm sideeffect "", "=v,0"(float %local.reduce99) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i525 = insertelement <4 x float> %i523, float %i524, i64 2, !dbg !44
  %i526 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi62), !dbg !40
  %local.reduce100 = fadd float %i526, %k0.sum.anchor2001, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2009 = tail call float asm sideeffect "", "=v,0"(float %local.reduce100) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i527 = insertelement <4 x float> %i525, float %i526, i64 3, !dbg !44
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i527) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02220 = extractelement <4 x float> %math.result392, i64 0, !dbg !44
  %.cvtpack.f12221 = extractelement <4 x float> %math.result392, i64 1, !dbg !44
  %.cvtpack.f22222 = extractelement <4 x float> %math.result392, i64 2, !dbg !44
  %.cvtpack.f32223 = extractelement <4 x float> %math.result392, i64 3, !dbg !44
  %.cvtpack.p02224 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02220, float %.cvtpack.f12221) #12, !dbg !44
  %.cvtpack.p12225 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22222, float %.cvtpack.f32223) #12, !dbg !44
  %.cvtpack.v02226 = insertelement <2 x i32> poison, i32 %.cvtpack.p02224, i64 0, !dbg !44
  %.cvtpack.v12227 = insertelement <2 x i32> %.cvtpack.v02226, i32 %.cvtpack.p12225, i64 1, !dbg !44
  %i528 = bitcast <2 x i32> %.cvtpack.v12227 to <4 x half>, !dbg !44
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i528) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i529 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i519, <4 x float> %i505) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i530 = shufflevector <8 x half> %i461, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i531 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i530, <4 x float> %i507) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i532 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi64), !dbg !40
  %local.reduce66 = fadd float %i532, %k0.sum.anchor2002, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2010 = tail call float asm sideeffect "", "=v,0"(float %local.reduce66) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i533 = insertelement <4 x float> poison, float %i532, i64 0, !dbg !44
  %i534 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi66), !dbg !40
  %local.reduce67 = fadd float %i534, %k0.sum.anchor2003, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2011 = tail call float asm sideeffect "", "=v,0"(float %local.reduce67) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i535 = insertelement <4 x float> %i533, float %i534, i64 1, !dbg !44
  %i536 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi68), !dbg !40
  %local.reduce68 = fadd float %i536, %k0.sum.anchor2004, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2012 = tail call float asm sideeffect "", "=v,0"(float %local.reduce68) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i537 = insertelement <4 x float> %i535, float %i536, i64 2, !dbg !44
  %i538 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi70), !dbg !40
  %local.reduce69 = fadd float %i538, %k0.sum.anchor2005, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2013 = tail call float asm sideeffect "", "=v,0"(float %local.reduce69) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i539 = insertelement <4 x float> %i537, float %i538, i64 3, !dbg !44
  %math.result408 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i539) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02228 = extractelement <4 x float> %math.result408, i64 0, !dbg !44
  %.cvtpack.f12229 = extractelement <4 x float> %math.result408, i64 1, !dbg !44
  %.cvtpack.f22230 = extractelement <4 x float> %math.result408, i64 2, !dbg !44
  %.cvtpack.f32231 = extractelement <4 x float> %math.result408, i64 3, !dbg !44
  %.cvtpack.p02232 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02228, float %.cvtpack.f12229) #12, !dbg !44
  %.cvtpack.p12233 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22230, float %.cvtpack.f32231) #12, !dbg !44
  %.cvtpack.v02234 = insertelement <2 x i32> poison, i32 %.cvtpack.p02232, i64 0, !dbg !44
  %.cvtpack.v12235 = insertelement <2 x i32> %.cvtpack.v02234, i32 %.cvtpack.p12233, i64 1, !dbg !44
  %i540 = bitcast <2 x i32> %.cvtpack.v12235 to <4 x half>, !dbg !44
  %math.packed409 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i540) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %i519, <4 x float> %i517) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i542 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %i530, <4 x float> %i518) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i543 = shufflevector <8 x half> %i463, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i544 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi72), !dbg !40
  %local.reduce101 = fadd float %i544, %k0.sum.anchor2006, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2014 = tail call float asm sideeffect "", "=v,0"(float %local.reduce101) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i545 = insertelement <4 x float> poison, float %i544, i64 0, !dbg !44
  %i546 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi74), !dbg !40
  %local.reduce102 = fadd float %i546, %k0.sum.anchor2007, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2015 = tail call float asm sideeffect "", "=v,0"(float %local.reduce102) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i547 = insertelement <4 x float> %i545, float %i546, i64 1, !dbg !44
  %i548 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi76), !dbg !40
  %local.reduce103 = fadd float %i548, %k0.sum.anchor2008, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2016 = tail call float asm sideeffect "", "=v,0"(float %local.reduce103) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i549 = insertelement <4 x float> %i547, float %i548, i64 2, !dbg !44
  %i550 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi78), !dbg !40
  %local.reduce104 = fadd float %i550, %k0.sum.anchor2009, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2017 = tail call float asm sideeffect "", "=v,0"(float %local.reduce104) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i551 = insertelement <4 x float> %i549, float %i550, i64 3, !dbg !44
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i551) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02236 = extractelement <4 x float> %math.result394, i64 0, !dbg !44
  %.cvtpack.f12237 = extractelement <4 x float> %math.result394, i64 1, !dbg !44
  %.cvtpack.f22238 = extractelement <4 x float> %math.result394, i64 2, !dbg !44
  %.cvtpack.f32239 = extractelement <4 x float> %math.result394, i64 3, !dbg !44
  %.cvtpack.p02240 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02236, float %.cvtpack.f12237) #12, !dbg !44
  %.cvtpack.p12241 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22238, float %.cvtpack.f32239) #12, !dbg !44
  %.cvtpack.v02242 = insertelement <2 x i32> poison, i32 %.cvtpack.p02240, i64 0, !dbg !44
  %.cvtpack.v12243 = insertelement <2 x i32> %.cvtpack.v02242, i32 %.cvtpack.p12241, i64 1, !dbg !44
  %i552 = bitcast <2 x i32> %.cvtpack.v12243 to <4 x half>, !dbg !44
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i552) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i543, <4 x float> %i529) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i554 = shufflevector <8 x half> %i463, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i554, <4 x float> %i531) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i556 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi80), !dbg !40
  %local.reduce70 = fadd float %i556, %k0.sum.anchor2010, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2018 = tail call float asm sideeffect "", "=v,0"(float %local.reduce70) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i557 = insertelement <4 x float> poison, float %i556, i64 0, !dbg !44
  %i558 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi82), !dbg !40
  %local.reduce71 = fadd float %i558, %k0.sum.anchor2011, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2019 = tail call float asm sideeffect "", "=v,0"(float %local.reduce71) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i559 = insertelement <4 x float> %i557, float %i558, i64 1, !dbg !44
  %i560 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi84), !dbg !40
  %local.reduce72 = fadd float %i560, %k0.sum.anchor2012, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2020 = tail call float asm sideeffect "", "=v,0"(float %local.reduce72) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i561 = insertelement <4 x float> %i559, float %i560, i64 2, !dbg !44
  %i562 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi86), !dbg !40
  %local.reduce73 = fadd float %i562, %k0.sum.anchor2013, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2021 = tail call float asm sideeffect "", "=v,0"(float %local.reduce73) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i563 = insertelement <4 x float> %i561, float %i562, i64 3, !dbg !44
  %math.result410 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i563) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02244 = extractelement <4 x float> %math.result410, i64 0, !dbg !44
  %.cvtpack.f12245 = extractelement <4 x float> %math.result410, i64 1, !dbg !44
  %.cvtpack.f22246 = extractelement <4 x float> %math.result410, i64 2, !dbg !44
  %.cvtpack.f32247 = extractelement <4 x float> %math.result410, i64 3, !dbg !44
  %.cvtpack.p02248 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02244, float %.cvtpack.f12245) #12, !dbg !44
  %.cvtpack.p12249 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22246, float %.cvtpack.f32247) #12, !dbg !44
  %.cvtpack.v02250 = insertelement <2 x i32> poison, i32 %.cvtpack.p02248, i64 0, !dbg !44
  %.cvtpack.v12251 = insertelement <2 x i32> %.cvtpack.v02250, i32 %.cvtpack.p12249, i64 1, !dbg !44
  %i564 = bitcast <2 x i32> %.cvtpack.v12251 to <4 x half>, !dbg !44
  %math.packed411 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i564) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i565 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %i543, <4 x float> %i541) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i566 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %i554, <4 x float> %i542) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i567 = shufflevector <8 x half> %i465, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i568 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi88), !dbg !40
  %local.reduce105 = fadd float %i568, %k0.sum.anchor2014, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2022 = tail call float asm sideeffect "", "=v,0"(float %local.reduce105) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i569 = insertelement <4 x float> poison, float %i568, i64 0, !dbg !44
  %i570 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi90), !dbg !40
  %local.reduce106 = fadd float %i570, %k0.sum.anchor2015, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2023 = tail call float asm sideeffect "", "=v,0"(float %local.reduce106) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i571 = insertelement <4 x float> %i569, float %i570, i64 1, !dbg !44
  %i572 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi92), !dbg !40
  %local.reduce107 = fadd float %i572, %k0.sum.anchor2016, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2024 = tail call float asm sideeffect "", "=v,0"(float %local.reduce107) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i573 = insertelement <4 x float> %i571, float %i572, i64 2, !dbg !44
  %i574 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi94), !dbg !40
  %local.reduce108 = fadd float %i574, %k0.sum.anchor2017, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2025 = tail call float asm sideeffect "", "=v,0"(float %local.reduce108) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i575 = insertelement <4 x float> %i573, float %i574, i64 3, !dbg !44
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i575) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02252 = extractelement <4 x float> %math.result396, i64 0, !dbg !44
  %.cvtpack.f12253 = extractelement <4 x float> %math.result396, i64 1, !dbg !44
  %.cvtpack.f22254 = extractelement <4 x float> %math.result396, i64 2, !dbg !44
  %.cvtpack.f32255 = extractelement <4 x float> %math.result396, i64 3, !dbg !44
  %.cvtpack.p02256 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02252, float %.cvtpack.f12253) #12, !dbg !44
  %.cvtpack.p12257 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22254, float %.cvtpack.f32255) #12, !dbg !44
  %.cvtpack.v02258 = insertelement <2 x i32> poison, i32 %.cvtpack.p02256, i64 0, !dbg !44
  %.cvtpack.v12259 = insertelement <2 x i32> %.cvtpack.v02258, i32 %.cvtpack.p12257, i64 1, !dbg !44
  %i576 = bitcast <2 x i32> %.cvtpack.v12259 to <4 x half>, !dbg !44
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i576) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i577 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i567, <4 x float> %i553) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i578 = shufflevector <8 x half> %i465, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i579 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i578, <4 x float> %i555) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i580 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi96), !dbg !40
  %local.reduce74 = fadd float %i580, %k0.sum.anchor2018, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2026 = tail call float asm sideeffect "", "=v,0"(float %local.reduce74) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i581 = insertelement <4 x float> poison, float %i580, i64 0, !dbg !44
  %i582 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi98), !dbg !40
  %local.reduce75 = fadd float %i582, %k0.sum.anchor2019, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2027 = tail call float asm sideeffect "", "=v,0"(float %local.reduce75) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i583 = insertelement <4 x float> %i581, float %i582, i64 1, !dbg !44
  %i584 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi100), !dbg !40
  %local.reduce76 = fadd float %i584, %k0.sum.anchor2020, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2028 = tail call float asm sideeffect "", "=v,0"(float %local.reduce76) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i585 = insertelement <4 x float> %i583, float %i584, i64 2, !dbg !44
  %i586 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi102), !dbg !40
  %local.reduce77 = fadd float %i586, %k0.sum.anchor2021, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2029 = tail call float asm sideeffect "", "=v,0"(float %local.reduce77) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i587 = insertelement <4 x float> %i585, float %i586, i64 3, !dbg !44
  %math.result412 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i587) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02260 = extractelement <4 x float> %math.result412, i64 0, !dbg !44
  %.cvtpack.f12261 = extractelement <4 x float> %math.result412, i64 1, !dbg !44
  %.cvtpack.f22262 = extractelement <4 x float> %math.result412, i64 2, !dbg !44
  %.cvtpack.f32263 = extractelement <4 x float> %math.result412, i64 3, !dbg !44
  %.cvtpack.p02264 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02260, float %.cvtpack.f12261) #12, !dbg !44
  %.cvtpack.p12265 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22262, float %.cvtpack.f32263) #12, !dbg !44
  %.cvtpack.v02266 = insertelement <2 x i32> poison, i32 %.cvtpack.p02264, i64 0, !dbg !44
  %.cvtpack.v12267 = insertelement <2 x i32> %.cvtpack.v02266, i32 %.cvtpack.p12265, i64 1, !dbg !44
  %i588 = bitcast <2 x i32> %.cvtpack.v12267 to <4 x half>, !dbg !44
  %math.packed413 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i588) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i589 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %i567, <4 x float> %i565) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %i578, <4 x float> %i566) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i591 = shufflevector <8 x half> %i467, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i592 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi104), !dbg !40
  %local.reduce109 = fadd float %i592, %k0.sum.anchor2022, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2030 = tail call float asm sideeffect "", "=v,0"(float %local.reduce109) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i593 = insertelement <4 x float> poison, float %i592, i64 0, !dbg !44
  %i594 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi106), !dbg !40
  %local.reduce110 = fadd float %i594, %k0.sum.anchor2023, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2031 = tail call float asm sideeffect "", "=v,0"(float %local.reduce110) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i595 = insertelement <4 x float> %i593, float %i594, i64 1, !dbg !44
  %i596 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi108), !dbg !40
  %local.reduce111 = fadd float %i596, %k0.sum.anchor2024, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2032 = tail call float asm sideeffect "", "=v,0"(float %local.reduce111) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i597 = insertelement <4 x float> %i595, float %i596, i64 2, !dbg !44
  %i598 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi110), !dbg !40
  %local.reduce112 = fadd float %i598, %k0.sum.anchor2025, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2033 = tail call float asm sideeffect "", "=v,0"(float %local.reduce112) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i599 = insertelement <4 x float> %i597, float %i598, i64 3, !dbg !44
  %math.result398 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i599) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02268 = extractelement <4 x float> %math.result398, i64 0, !dbg !44
  %.cvtpack.f12269 = extractelement <4 x float> %math.result398, i64 1, !dbg !44
  %.cvtpack.f22270 = extractelement <4 x float> %math.result398, i64 2, !dbg !44
  %.cvtpack.f32271 = extractelement <4 x float> %math.result398, i64 3, !dbg !44
  %.cvtpack.p02272 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02268, float %.cvtpack.f12269) #12, !dbg !44
  %.cvtpack.p12273 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22270, float %.cvtpack.f32271) #12, !dbg !44
  %.cvtpack.v02274 = insertelement <2 x i32> poison, i32 %.cvtpack.p02272, i64 0, !dbg !44
  %.cvtpack.v12275 = insertelement <2 x i32> %.cvtpack.v02274, i32 %.cvtpack.p12273, i64 1, !dbg !44
  %i600 = bitcast <2 x i32> %.cvtpack.v12275 to <4 x half>, !dbg !44
  %math.packed399 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i600) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i601 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i591, <4 x float> %i577) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i602 = shufflevector <8 x half> %i467, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i603 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i602, <4 x float> %i579) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i604 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi112), !dbg !40
  %local.reduce78 = fadd float %i604, %k0.sum.anchor2026, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2034 = tail call float asm sideeffect "", "=v,0"(float %local.reduce78) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i605 = insertelement <4 x float> poison, float %i604, i64 0, !dbg !44
  %i606 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi114), !dbg !40
  %local.reduce79 = fadd float %i606, %k0.sum.anchor2027, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2035 = tail call float asm sideeffect "", "=v,0"(float %local.reduce79) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i607 = insertelement <4 x float> %i605, float %i606, i64 1, !dbg !44
  %i608 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi116), !dbg !40
  %local.reduce80 = fadd float %i608, %k0.sum.anchor2028, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2036 = tail call float asm sideeffect "", "=v,0"(float %local.reduce80) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i609 = insertelement <4 x float> %i607, float %i608, i64 2, !dbg !44
  %i610 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi118), !dbg !40
  %local.reduce81 = fadd float %i610, %k0.sum.anchor2029, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2037 = tail call float asm sideeffect "", "=v,0"(float %local.reduce81) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i611 = insertelement <4 x float> %i609, float %i610, i64 3, !dbg !44
  %math.result414 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i611) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02276 = extractelement <4 x float> %math.result414, i64 0, !dbg !44
  %.cvtpack.f12277 = extractelement <4 x float> %math.result414, i64 1, !dbg !44
  %.cvtpack.f22278 = extractelement <4 x float> %math.result414, i64 2, !dbg !44
  %.cvtpack.f32279 = extractelement <4 x float> %math.result414, i64 3, !dbg !44
  %.cvtpack.p02280 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02276, float %.cvtpack.f12277) #12, !dbg !44
  %.cvtpack.p12281 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22278, float %.cvtpack.f32279) #12, !dbg !44
  %.cvtpack.v02282 = insertelement <2 x i32> poison, i32 %.cvtpack.p02280, i64 0, !dbg !44
  %.cvtpack.v12283 = insertelement <2 x i32> %.cvtpack.v02282, i32 %.cvtpack.p12281, i64 1, !dbg !44
  %i612 = bitcast <2 x i32> %.cvtpack.v12283 to <4 x half>, !dbg !44
  %math.packed415 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i612) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i613 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %i591, <4 x float> %i589) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i614 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %i602, <4 x float> %i590) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i615 = shufflevector <8 x half> %i469, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i616 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi120), !dbg !40
  %local.reduce113 = fadd float %i616, %k0.sum.anchor2030, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2038 = tail call float asm sideeffect "", "=v,0"(float %local.reduce113) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i617 = insertelement <4 x float> poison, float %i616, i64 0, !dbg !44
  %i618 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi122), !dbg !40
  %local.reduce114 = fadd float %i618, %k0.sum.anchor2031, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2039 = tail call float asm sideeffect "", "=v,0"(float %local.reduce114) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i619 = insertelement <4 x float> %i617, float %i618, i64 1, !dbg !44
  %i620 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi124), !dbg !40
  %local.reduce115 = fadd float %i620, %k0.sum.anchor2032, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2040 = tail call float asm sideeffect "", "=v,0"(float %local.reduce115) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i621 = insertelement <4 x float> %i619, float %i620, i64 2, !dbg !44
  %i622 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi126), !dbg !40
  %local.reduce116 = fadd float %i622, %k0.sum.anchor2033, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2041 = tail call float asm sideeffect "", "=v,0"(float %local.reduce116) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i623 = insertelement <4 x float> %i621, float %i622, i64 3, !dbg !44
  %math.result400 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i623) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02284 = extractelement <4 x float> %math.result400, i64 0, !dbg !44
  %.cvtpack.f12285 = extractelement <4 x float> %math.result400, i64 1, !dbg !44
  %.cvtpack.f22286 = extractelement <4 x float> %math.result400, i64 2, !dbg !44
  %.cvtpack.f32287 = extractelement <4 x float> %math.result400, i64 3, !dbg !44
  %.cvtpack.p02288 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02284, float %.cvtpack.f12285) #12, !dbg !44
  %.cvtpack.p12289 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22286, float %.cvtpack.f32287) #12, !dbg !44
  %.cvtpack.v02290 = insertelement <2 x i32> poison, i32 %.cvtpack.p02288, i64 0, !dbg !44
  %.cvtpack.v12291 = insertelement <2 x i32> %.cvtpack.v02290, i32 %.cvtpack.p12289, i64 1, !dbg !44
  %i624 = bitcast <2 x i32> %.cvtpack.v12291 to <4 x half>, !dbg !44
  %math.packed401 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i624) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i625 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i615, <4 x float> %i601) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i626 = shufflevector <8 x half> %i469, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i627 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i626, <4 x float> %i603) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i628 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi128), !dbg !40
  %local.reduce82 = fadd float %i628, %k0.sum.anchor2034, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2042 = tail call float asm sideeffect "", "=v,0"(float %local.reduce82) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i629 = insertelement <4 x float> poison, float %i628, i64 0, !dbg !44
  %i630 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi130), !dbg !40
  %local.reduce83 = fadd float %i630, %k0.sum.anchor2035, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2043 = tail call float asm sideeffect "", "=v,0"(float %local.reduce83) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i631 = insertelement <4 x float> %i629, float %i630, i64 1, !dbg !44
  %i632 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi132), !dbg !40
  %local.reduce84 = fadd float %i632, %k0.sum.anchor2036, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2044 = tail call float asm sideeffect "", "=v,0"(float %local.reduce84) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i633 = insertelement <4 x float> %i631, float %i632, i64 2, !dbg !44
  %i634 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi134), !dbg !40
  %local.reduce85 = fadd float %i634, %k0.sum.anchor2037, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2045 = tail call float asm sideeffect "", "=v,0"(float %local.reduce85) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i635 = insertelement <4 x float> %i633, float %i634, i64 3, !dbg !44
  %math.result416 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i635) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02292 = extractelement <4 x float> %math.result416, i64 0, !dbg !44
  %.cvtpack.f12293 = extractelement <4 x float> %math.result416, i64 1, !dbg !44
  %.cvtpack.f22294 = extractelement <4 x float> %math.result416, i64 2, !dbg !44
  %.cvtpack.f32295 = extractelement <4 x float> %math.result416, i64 3, !dbg !44
  %.cvtpack.p02296 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02292, float %.cvtpack.f12293) #12, !dbg !44
  %.cvtpack.p12297 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22294, float %.cvtpack.f32295) #12, !dbg !44
  %.cvtpack.v02298 = insertelement <2 x i32> poison, i32 %.cvtpack.p02296, i64 0, !dbg !44
  %.cvtpack.v12299 = insertelement <2 x i32> %.cvtpack.v02298, i32 %.cvtpack.p12297, i64 1, !dbg !44
  %i636 = bitcast <2 x i32> %.cvtpack.v12299 to <4 x half>, !dbg !44
  %math.packed417 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i636) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i637 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %i615, <4 x float> %i613) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i638 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %i626, <4 x float> %i614) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i639 = bitcast <4 x i32> %i470 to <8 x half>, !dbg !42
  %i640 = shufflevector <8 x half> %i639, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i641 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi136), !dbg !40
  %local.reduce117 = fadd float %i641, %k0.sum.anchor2038, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2046 = tail call float asm sideeffect "", "=v,0"(float %local.reduce117) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i642 = insertelement <4 x float> poison, float %i641, i64 0, !dbg !44
  %i643 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi138), !dbg !40
  %local.reduce118 = fadd float %i643, %k0.sum.anchor2039, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2047 = tail call float asm sideeffect "", "=v,0"(float %local.reduce118) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce121 = fadd float %k0.sum.anchor2046, %k0.sum.anchor2047, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2048 = tail call float asm sideeffect "", "=v,0"(float %local.reduce121) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i644 = insertelement <4 x float> %i642, float %i643, i64 1, !dbg !44
  %i645 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi140), !dbg !40
  %local.reduce119 = fadd float %i645, %k0.sum.anchor2040, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2049 = tail call float asm sideeffect "", "=v,0"(float %local.reduce119) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i646 = insertelement <4 x float> %i644, float %i645, i64 2, !dbg !44
  %i647 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi142), !dbg !40
  %local.reduce120 = fadd float %i647, %k0.sum.anchor2041, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2050 = tail call float asm sideeffect "", "=v,0"(float %local.reduce120) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce122 = fadd float %k0.sum.anchor2049, %k0.sum.anchor2050, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2051 = tail call float asm sideeffect "", "=v,0"(float %local.reduce122) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i648 = fadd float %k0.sum.anchor2048, %k0.sum.anchor2051, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2052 = tail call float asm sideeffect "", "=v,0"(float %i648) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i649 = fadd float %i445, %k0.sum.anchor2052, !dbg !45
  %k0.sum.anchor2053 = tail call float asm sideeffect "", "=v,0"(float %i649) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i650 = insertelement <1 x float> poison, float %k0.sum.anchor2053, i64 0, !dbg !12
  %i651 = insertvalue [2 x <1 x float>] poison, <1 x float> %i650, 0, !dbg !12
  %i652 = insertelement <4 x float> %i646, float %i647, i64 3, !dbg !44
  %math.result402 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i652) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02300 = extractelement <4 x float> %math.result402, i64 0, !dbg !44
  %.cvtpack.f12301 = extractelement <4 x float> %math.result402, i64 1, !dbg !44
  %.cvtpack.f22302 = extractelement <4 x float> %math.result402, i64 2, !dbg !44
  %.cvtpack.f32303 = extractelement <4 x float> %math.result402, i64 3, !dbg !44
  %.cvtpack.p02304 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02300, float %.cvtpack.f12301) #12, !dbg !44
  %.cvtpack.p12305 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22302, float %.cvtpack.f32303) #12, !dbg !44
  %.cvtpack.v02306 = insertelement <2 x i32> poison, i32 %.cvtpack.p02304, i64 0, !dbg !44
  %.cvtpack.v12307 = insertelement <2 x i32> %.cvtpack.v02306, i32 %.cvtpack.p12305, i64 1, !dbg !44
  %i653 = bitcast <2 x i32> %.cvtpack.v12307 to <4 x half>, !dbg !44
  %math.packed403 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i653) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i654 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i640, <4 x float> %i625) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i655 = shufflevector <8 x half> %i639, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i656 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i655, <4 x float> %i627) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i657 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi144), !dbg !40
  %local.reduce86 = fadd float %i657, %k0.sum.anchor2042, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2054 = tail call float asm sideeffect "", "=v,0"(float %local.reduce86) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i658 = insertelement <4 x float> poison, float %i657, i64 0, !dbg !44
  %i659 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi146), !dbg !40
  %local.reduce87 = fadd float %i659, %k0.sum.anchor2043, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2055 = tail call float asm sideeffect "", "=v,0"(float %local.reduce87) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce90 = fadd float %k0.sum.anchor2054, %k0.sum.anchor2055, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2056 = tail call float asm sideeffect "", "=v,0"(float %local.reduce90) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i660 = insertelement <4 x float> %i658, float %i659, i64 1, !dbg !44
  %i661 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi148), !dbg !40
  %local.reduce88 = fadd float %i661, %k0.sum.anchor2044, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2057 = tail call float asm sideeffect "", "=v,0"(float %local.reduce88) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i662 = insertelement <4 x float> %i660, float %i661, i64 2, !dbg !44
  %i663 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi150), !dbg !40
  %local.reduce89 = fadd float %i663, %k0.sum.anchor2045, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2058 = tail call float asm sideeffect "", "=v,0"(float %local.reduce89) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce91 = fadd float %k0.sum.anchor2057, %k0.sum.anchor2058, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2059 = tail call float asm sideeffect "", "=v,0"(float %local.reduce91) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i664 = fadd float %k0.sum.anchor2056, %k0.sum.anchor2059, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2060 = tail call float asm sideeffect "", "=v,0"(float %i664) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i665 = fadd float %i447, %k0.sum.anchor2060, !dbg !45
  %k0.sum.anchor2061 = tail call float asm sideeffect "", "=v,0"(float %i665) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i666 = insertelement <1 x float> poison, float %k0.sum.anchor2061, i64 0, !dbg !12
  %i667 = insertvalue [2 x <1 x float>] %i651, <1 x float> %i666, 1, !dbg !12
  %i668 = insertelement <4 x float> %i662, float %i663, i64 3, !dbg !44
  %math.result418 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i668) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02308 = extractelement <4 x float> %math.result418, i64 0, !dbg !44
  %.cvtpack.f12309 = extractelement <4 x float> %math.result418, i64 1, !dbg !44
  %.cvtpack.f22310 = extractelement <4 x float> %math.result418, i64 2, !dbg !44
  %.cvtpack.f32311 = extractelement <4 x float> %math.result418, i64 3, !dbg !44
  %.cvtpack.p02312 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02308, float %.cvtpack.f12309) #12, !dbg !44
  %.cvtpack.p12313 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22310, float %.cvtpack.f32311) #12, !dbg !44
  %.cvtpack.v02314 = insertelement <2 x i32> poison, i32 %.cvtpack.p02312, i64 0, !dbg !44
  %.cvtpack.v12315 = insertelement <2 x i32> %.cvtpack.v02314, i32 %.cvtpack.p12313, i64 1, !dbg !44
  %i669 = bitcast <2 x i32> %.cvtpack.v12315 to <4 x half>, !dbg !44
  %math.packed419 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i669) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %i670 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %i640, <4 x float> %i637) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i671 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %i655, <4 x float> %i638) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %i672 = extractvalue [2 x <8 x float>] %lazy.state2194, 0, !dbg !41
  %i673 = shufflevector <8 x float> %i672, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %i674 = shufflevector <8 x float> %i672, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %i675 = extractvalue [2 x <8 x float>] %lazy.state2194, 1, !dbg !41
  %i676 = shufflevector <8 x float> %i675, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %i677 = shufflevector <8 x float> %i675, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1463 = add nsw i64 %address.local938, %i203, !dbg !38
  %vdb.address1464 = trunc nsw i64 %vdb.address1463 to i32, !dbg !38
  %vdb.address1465 = shl nsw i32 %vdb.address1464, 1, !dbg !38
  %vdb.address1470 = trunc nuw nsw i64 %i203 to i32, !dbg !38
  %vdb.address1471 = or disjoint i32 %address.local945, %vdb.address1470, !dbg !38
  %vdb.address1472 = shl nuw nsw i32 %vdb.address1471, 1, !dbg !38
  %i678 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i54, i32 %vdb.address1465, <4 x i32> %dma.buffer.rsrc1362) #9, !dbg !38
  %i679 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i57, i32 %vdb.address1472, <4 x i32> %dma.buffer.rsrc1362) #9, !dbg !38
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local954, i64 8192, !dbg !42
  %i680 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i681 = bitcast <4 x i32> %i680 to <8 x half>, !dbg !42
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8448, !dbg !42
  %i682 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i683 = bitcast <4 x i32> %i682 to <8 x half>, !dbg !42
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8704, !dbg !42
  %i684 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i685 = bitcast <4 x i32> %i684 to <8 x half>, !dbg !42
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8960, !dbg !42
  %i686 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i687 = bitcast <4 x i32> %i686 to <8 x half>, !dbg !42
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local954, i64 12288, !dbg !42
  %i688 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i689 = bitcast <4 x i32> %i688 to <8 x half>, !dbg !42
  %vdb.read1460 = getelementptr i8, ptr addrspace(3) %address.local963, i64 12544, !dbg !42
  %i690 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1460, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i691 = bitcast <4 x i32> %i690 to <8 x half>, !dbg !42
  %vdb.read1461 = getelementptr i8, ptr addrspace(3) %address.local963, i64 12800, !dbg !42
  %i692 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1461, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i693 = bitcast <4 x i32> %i692 to <8 x half>, !dbg !42
  %vdb.read1462 = getelementptr i8, ptr addrspace(3) %address.local963, i64 13056, !dbg !42
  %i694 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1462, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i695 = shufflevector <8 x half> %i681, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i696 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i695, <4 x float> %i673) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i697 = shufflevector <8 x half> %i681, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i698 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i697, <4 x float> %i674) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i699 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %i695, <4 x float> %i676) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i700 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %i697, <4 x float> %i677) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i701 = shufflevector <8 x half> %i683, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i702 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i701, <4 x float> %i696) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i703 = shufflevector <8 x half> %i683, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i704 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i703, <4 x float> %i698) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i705 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %i701, <4 x float> %i699) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i706 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %i703, <4 x float> %i700) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i707 = shufflevector <8 x half> %i685, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i708 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i707, <4 x float> %i702) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i709 = shufflevector <8 x half> %i685, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i710 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i709, <4 x float> %i704) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i711 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %i707, <4 x float> %i705) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i712 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %i709, <4 x float> %i706) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i713 = shufflevector <8 x half> %i687, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i714 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i713, <4 x float> %i708) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i715 = shufflevector <8 x half> %i687, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i716 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i715, <4 x float> %i710) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i717 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %i713, <4 x float> %i711) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i718 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %i715, <4 x float> %i712) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i719 = shufflevector <8 x half> %i689, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i720 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i719, <4 x float> %i714) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i721 = shufflevector <8 x half> %i689, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i722 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i721, <4 x float> %i716) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i723 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %i719, <4 x float> %i717) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i724 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %i721, <4 x float> %i718) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i725 = shufflevector <8 x half> %i691, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i726 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i725, <4 x float> %i720) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i727 = shufflevector <8 x half> %i691, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i728 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i727, <4 x float> %i722) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i729 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %i725, <4 x float> %i723) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i730 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %i727, <4 x float> %i724) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i731 = shufflevector <8 x half> %i693, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i732 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i731, <4 x float> %i726) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i733 = shufflevector <8 x half> %i693, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i734 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i733, <4 x float> %i728) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i735 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %i731, <4 x float> %i729) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i736 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %i733, <4 x float> %i730) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i737 = bitcast <4 x i32> %i694 to <8 x half>, !dbg !42
  %i738 = shufflevector <8 x half> %i737, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i739 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i738, <4 x float> %i732) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i740 = shufflevector <8 x half> %i737, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i741 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i740, <4 x float> %i734) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i742 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %i738, <4 x float> %i735) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %i740, <4 x float> %i736) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %i744 = extractvalue [2 x <8 x float>] %lazy.state2193, 0, !dbg !41
  %i745 = shufflevector <8 x float> %i744, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %i746 = shufflevector <8 x float> %i744, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %i747 = extractvalue [2 x <8 x float>] %lazy.state2193, 1, !dbg !41
  %i748 = shufflevector <8 x float> %i747, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %i749 = shufflevector <8 x float> %i747, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1477 = add nsw i64 %address.local938, %i256, !dbg !38
  %vdb.address1478 = trunc nsw i64 %vdb.address1477 to i32, !dbg !38
  %vdb.address1479 = shl nsw i32 %vdb.address1478, 1, !dbg !38
  %vdb.address1484 = trunc nuw nsw i64 %i256 to i32, !dbg !38
  %vdb.address1485 = or disjoint i32 %address.local945, %vdb.address1484, !dbg !38
  %vdb.address1486 = shl nuw nsw i32 %vdb.address1485, 1, !dbg !38
  %i750 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1479, <4 x i32> %dma.buffer.rsrc1362) #9, !dbg !38
  %i751 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1455, i32 %vdb.address1486, <4 x i32> %dma.buffer.rsrc1362) #9, !dbg !38
  %i752 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local954, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i753 = bitcast <4 x i32> %i752 to <8 x half>, !dbg !42
  %i754 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i755 = bitcast <4 x i32> %i754 to <8 x half>, !dbg !42
  %i756 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local968, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i757 = bitcast <4 x i32> %i756 to <8 x half>, !dbg !42
  %i758 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local972, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i759 = bitcast <4 x i32> %i758 to <8 x half>, !dbg !42
  %i760 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local974, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i761 = bitcast <4 x i32> %i760 to <8 x half>, !dbg !42
  %i762 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local978, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i763 = bitcast <4 x i32> %i762 to <8 x half>, !dbg !42
  %i764 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local982, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i765 = bitcast <4 x i32> %i764 to <8 x half>, !dbg !42
  %i766 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local986, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i767 = shufflevector <8 x half> %i753, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i768 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i767, <4 x float> %i745) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i769 = shufflevector <8 x half> %i753, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i770 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i769, <4 x float> %i746) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i771 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %i767, <4 x float> %i748) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i772 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %i769, <4 x float> %i749) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i773 = shufflevector <8 x half> %i755, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i774 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i773, <4 x float> %i768) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i775 = shufflevector <8 x half> %i755, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i776 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i775, <4 x float> %i770) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i777 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %i773, <4 x float> %i771) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i778 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %i775, <4 x float> %i772) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i779 = shufflevector <8 x half> %i757, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i780 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i779, <4 x float> %i774) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i781 = shufflevector <8 x half> %i757, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i782 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i781, <4 x float> %i776) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i783 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %i779, <4 x float> %i777) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i784 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %i781, <4 x float> %i778) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i785 = shufflevector <8 x half> %i759, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i786 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i785, <4 x float> %i780) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i787 = shufflevector <8 x half> %i759, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i788 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i787, <4 x float> %i782) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i789 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %i785, <4 x float> %i783) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i790 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %i787, <4 x float> %i784) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i791 = shufflevector <8 x half> %i761, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i792 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i791, <4 x float> %i786) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i793 = shufflevector <8 x half> %i761, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i794 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i793, <4 x float> %i788) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i795 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %i791, <4 x float> %i789) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i796 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %i793, <4 x float> %i790) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i797 = shufflevector <8 x half> %i763, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i798 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i797, <4 x float> %i792) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i799 = shufflevector <8 x half> %i763, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i800 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i799, <4 x float> %i794) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i801 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %i797, <4 x float> %i795) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i802 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %i799, <4 x float> %i796) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i803 = shufflevector <8 x half> %i765, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i804 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i803, <4 x float> %i798) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i805 = shufflevector <8 x half> %i765, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i806 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i805, <4 x float> %i800) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i807 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %i803, <4 x float> %i801) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i808 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %i805, <4 x float> %i802) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i809 = bitcast <4 x i32> %i766 to <8 x half>, !dbg !42
  %i810 = shufflevector <8 x half> %i809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i811 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i810, <4 x float> %i804) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i812 = shufflevector <8 x half> %i809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i812, <4 x float> %i806) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i814 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %i810, <4 x float> %i807) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i815 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %i812, <4 x float> %i808) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %i816 = extractvalue [2 x <8 x float>] %lazy.state2192, 0, !dbg !41
  %i817 = shufflevector <8 x float> %i816, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %i818 = shufflevector <8 x float> %i816, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %i819 = extractvalue [2 x <8 x float>] %lazy.state2192, 1, !dbg !41
  %i820 = shufflevector <8 x float> %i819, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %i821 = shufflevector <8 x float> %i819, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %i822 = add nuw nsw i64 %i122, 1, !dbg !12
  %k0.has.next.not = icmp eq i64 %i122, 31, !dbg !42
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !42

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %i822, 14, !dbg !14
  %k0.next.2072 = add nuw nsw i64 %k0.next., %k0.prime.2062, !dbg !14
  %k0.next.kdma.source1517 = or disjoint i64 %k0.next.2072, %k0.prime.kdma.source1516, !dbg !14
  %k0.next.kdma.source1518 = trunc nuw nsw i64 %k0.next.kdma.source1517 to i32, !dbg !14
  %k0.next.kdma.source1519 = shl nuw nsw i32 %k0.next.kdma.source1518, 1, !dbg !14
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1519, %kdma.intra, !dbg !14
  %k0.next.2073 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i54, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1537 = or disjoint i64 %k0.next.2072, %k0.prime.kdma.source1536, !dbg !14
  %k0.next.kdma.source1538 = trunc nuw nsw i64 %k0.next.kdma.source1537 to i32, !dbg !14
  %k0.next.kdma.source1539 = shl nuw nsw i32 %k0.next.kdma.source1538, 1, !dbg !14
  %k0.next.kdma.offset1540 = or disjoint i32 %k0.next.kdma.source1539, %kdma.intra.hi, !dbg !14
  %k0.next.2074 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1541, i32 %k0.next.kdma.offset1540, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1561 = or disjoint i64 %k0.next.2072, %k0.prime.kdma.source1560, !dbg !14
  %k0.next.kdma.source1562 = trunc nuw nsw i64 %k0.next.kdma.source1561 to i32, !dbg !14
  %k0.next.kdma.source1563 = shl nuw nsw i32 %k0.next.kdma.source1562, 1, !dbg !14
  %k0.next.kdma.offset1564 = add nuw nsw i32 %k0.next.kdma.source1563, %kdma.intra, !dbg !14
  %k0.next.2075 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i57, i32 %k0.next.kdma.offset1564, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1585 = or disjoint i64 %k0.next.2072, %k0.prime.kdma.source1584, !dbg !14
  %k0.next.kdma.source1586 = trunc nuw nsw i64 %k0.next.kdma.source1585 to i32, !dbg !14
  %k0.next.kdma.source1587 = shl nuw nsw i32 %k0.next.kdma.source1586, 1, !dbg !14
  %k0.next.kdma.offset1588 = add nuw nsw i32 %k0.next.kdma.source1587, %kdma.intra.hi, !dbg !14
  %k0.next.2076 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1589, i32 %k0.next.kdma.offset1588, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #9, !dbg !14, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %lazy.join
  %i823 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i824 = bitcast <4 x i32> %i823 to <8 x half>, !dbg !42
  %i825 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i826 = bitcast <4 x i32> %i825 to <8 x half>, !dbg !42
  %i827 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i828 = bitcast <4 x i32> %i827 to <8 x half>, !dbg !42
  %i829 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i830 = bitcast <4 x i32> %i829 to <8 x half>, !dbg !42
  %i831 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i832 = bitcast <4 x i32> %i831 to <8 x half>, !dbg !42
  %i833 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1460, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i834 = bitcast <4 x i32> %i833 to <8 x half>, !dbg !42
  %i835 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1461, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i836 = bitcast <4 x i32> %i835 to <8 x half>, !dbg !42
  %i837 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1462, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %i838 = shufflevector <8 x half> %i824, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i839 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i838, <4 x float> %i817) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i840 = shufflevector <8 x half> %i824, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i841 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i840, <4 x float> %i818) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %i838, <4 x float> %i820) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i843 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %i840, <4 x float> %i821) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i844 = shufflevector <8 x half> %i826, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i844, <4 x float> %i839) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i846 = shufflevector <8 x half> %i826, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i846, <4 x float> %i841) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %i844, <4 x float> %i842) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i849 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %i846, <4 x float> %i843) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i850 = shufflevector <8 x half> %i828, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i850, <4 x float> %i845) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i852 = shufflevector <8 x half> %i828, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i853 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i852, <4 x float> %i847) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %i850, <4 x float> %i848) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i855 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %i852, <4 x float> %i849) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i856 = shufflevector <8 x half> %i830, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i857 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i856, <4 x float> %i851) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i858 = shufflevector <8 x half> %i830, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i859 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i858, <4 x float> %i853) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i860 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %i856, <4 x float> %i854) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i861 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %i858, <4 x float> %i855) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i862 = shufflevector <8 x half> %i832, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i862, <4 x float> %i857) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i864 = shufflevector <8 x half> %i832, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i864, <4 x float> %i859) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i866 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %i862, <4 x float> %i860) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %i864, <4 x float> %i861) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i868 = shufflevector <8 x half> %i834, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i868, <4 x float> %i863) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i870 = shufflevector <8 x half> %i834, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i870, <4 x float> %i865) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i872 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %i868, <4 x float> %i866) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %i870, <4 x float> %i867) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i874 = shufflevector <8 x half> %i836, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i874, <4 x float> %i869) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i876 = shufflevector <8 x half> %i836, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i876, <4 x float> %i871) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i878 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %i874, <4 x float> %i872) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %i876, <4 x float> %i873) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i880 = bitcast <4 x i32> %i837 to <8 x half>, !dbg !42
  %i881 = shufflevector <8 x half> %i880, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i882 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i881, <4 x float> %i875) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i883 = shufflevector <8 x half> %i880, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i883, <4 x float> %i877) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %i881, <4 x float> %i878) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %i886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %i883, <4 x float> %i879) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %i887 = shufflevector <4 x float> %i654, <4 x float> %i656, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i888 = insertvalue [2 x <8 x float>] poison, <8 x float> %i887, 0, !dbg !12
  %i889 = shufflevector <4 x float> %i670, <4 x float> %i671, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i890 = insertvalue [2 x <8 x float>] %i888, <8 x float> %i889, 1, !dbg !12
  %i891 = shufflevector <4 x float> %i739, <4 x float> %i741, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i892 = insertvalue [2 x <8 x float>] poison, <8 x float> %i891, 0, !dbg !12
  %i893 = shufflevector <4 x float> %i742, <4 x float> %i743, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i894 = insertvalue [2 x <8 x float>] %i892, <8 x float> %i893, 1, !dbg !12
  %i895 = shufflevector <4 x float> %i811, <4 x float> %i813, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i896 = insertvalue [2 x <8 x float>] poison, <8 x float> %i895, 0, !dbg !12
  %i897 = shufflevector <4 x float> %i814, <4 x float> %i815, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i898 = insertvalue [2 x <8 x float>] %i896, <8 x float> %i897, 1, !dbg !12
  %i899 = shufflevector <4 x float> %i882, <4 x float> %i884, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i900 = insertvalue [2 x <8 x float>] poison, <8 x float> %i899, 0, !dbg !12
  %i901 = shufflevector <4 x float> %i885, <4 x float> %i886, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i902 = insertvalue [2 x <8 x float>] %i900, <8 x float> %i901, 1, !dbg !12
  %i903 = icmp samesign ult i64 %i122, 31, !dbg !12
  br i1 %i903, label %bb115, label %bb904, !dbg !12

bb904:                                            ; preds = %k0.join
  %i905 = shl nuw nsw i32 %i6, 2, !dbg !12
  %i906 = xor i32 %i905, 64, !dbg !12
  %i907 = bitcast float %k0.sum.anchor2061 to i32, !dbg !12
  %lane.sum.shuffle192 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i906, i32 %i907), !dbg !12
  %i908 = bitcast i32 %lane.sum.shuffle192 to float, !dbg !12
  %lane.sum.final193 = fadd float %k0.sum.anchor2061, %i908, !dbg !12
  %i909 = xor i32 %i905, 128, !dbg !12
  %i910 = bitcast float %lane.sum.final193 to i32, !dbg !12
  %lane.sum.shuffle194 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i909, i32 %i910), !dbg !12
  %i911 = bitcast i32 %lane.sum.shuffle194 to float, !dbg !12
  %lane.sum.final195 = fadd float %lane.sum.final193, %i911, !dbg !12
  %i912 = insertelement <1 x float> poison, float %lane.sum.final195, i64 0, !dbg !12
  %i913 = bitcast float %k0.sum.anchor2053 to i32, !dbg !12
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i906, i32 %i913), !dbg !12
  %i914 = bitcast i32 %lane.sum.shuffle to float, !dbg !12
  %lane.sum.final = fadd float %k0.sum.anchor2053, %i914, !dbg !12
  %i915 = bitcast float %lane.sum.final to i32, !dbg !12
  %lane.sum.shuffle186 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i909, i32 %i915), !dbg !12
  %i916 = bitcast i32 %lane.sum.shuffle186 to float, !dbg !12
  %lane.sum.final187 = fadd float %lane.sum.final, %i916, !dbg !12
  %i917 = insertelement <1 x float> poison, float %lane.sum.final187, i64 0, !dbg !12
  %i918 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %i918, 0, !dbg !12
  %i919 = and i32 %kdma.tid, 32, !dbg !12
  %.not40 = icmp eq i32 %i919, 0, !dbg !12
  %i920 = extractelement <4 x float> %i882, i64 0, !dbg !12
  %i921 = extractelement <4 x float> %i882, i64 1, !dbg !12
  %i922 = extractelement <4 x float> %i882, i64 2, !dbg !12
  %i923 = extractelement <4 x float> %i882, i64 3, !dbg !12
  %i924 = select i1 %.not, float %i921, float %i920, !dbg !12
  %i925 = bitcast float %i924 to i32, !dbg !12
  %i926 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i925), !dbg !12
  %i927 = bitcast i32 %i926 to float, !dbg !12
  %i928 = select i1 %.not, float %i920, float %i927, !dbg !12
  %i929 = select i1 %.not, float %i927, float %i921, !dbg !12
  %i930 = select i1 %.not, float %i923, float %i922, !dbg !12
  %i931 = bitcast float %i930 to i32, !dbg !12
  %i932 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i931), !dbg !12
  %i933 = bitcast i32 %i932 to float, !dbg !12
  %i934 = select i1 %.not, float %i922, float %i933, !dbg !12
  %i935 = select i1 %.not, float %i933, float %i923, !dbg !12
  %i936 = select i1 %.not40, float %i934, float %i928, !dbg !12
  %i937 = bitcast float %i936 to i32, !dbg !12
  %i938 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i937), !dbg !12
  %i939 = bitcast i32 %i938 to float, !dbg !12
  %i940 = select i1 %.not40, float %i935, float %i929, !dbg !12
  %i941 = bitcast float %i940 to i32, !dbg !12
  %i942 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i941), !dbg !12
  %i943 = bitcast i32 %i942 to float, !dbg !12
  %i944 = insertelement <4 x i1> poison, i1 %.not40, i64 0, !dbg !12
  %i945 = shufflevector <4 x i1> %i944, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %i946 = insertelement <4 x float> poison, float %i928, i64 0, !dbg !12
  %i947 = insertelement <4 x float> %i946, float %i929, i64 1, !dbg !12
  %i948 = insertelement <4 x float> %i947, float %i939, i64 2, !dbg !12
  %i949 = insertelement <4 x float> %i948, float %i943, i64 3, !dbg !12
  %i950 = insertelement <4 x float> poison, float %i939, i64 0, !dbg !12
  %i951 = insertelement <4 x float> %i950, float %i943, i64 1, !dbg !12
  %i952 = insertelement <4 x float> %i951, float %i934, i64 2, !dbg !12
  %i953 = insertelement <4 x float> %i952, float %i935, i64 3, !dbg !12
  %i954 = select <4 x i1> %i945, <4 x float> %i949, <4 x float> %i953, !dbg !12
  %i955 = extractelement <4 x float> %i884, i64 0, !dbg !12
  %i956 = extractelement <4 x float> %i884, i64 1, !dbg !12
  %i957 = extractelement <4 x float> %i884, i64 2, !dbg !12
  %i958 = extractelement <4 x float> %i884, i64 3, !dbg !12
  %i959 = select i1 %.not, float %i956, float %i955, !dbg !12
  %i960 = bitcast float %i959 to i32, !dbg !12
  %i961 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i960), !dbg !12
  %i962 = bitcast i32 %i961 to float, !dbg !12
  %i963 = select i1 %.not, float %i955, float %i962, !dbg !12
  %i964 = select i1 %.not, float %i962, float %i956, !dbg !12
  %i965 = select i1 %.not, float %i958, float %i957, !dbg !12
  %i966 = bitcast float %i965 to i32, !dbg !12
  %i967 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i966), !dbg !12
  %i968 = bitcast i32 %i967 to float, !dbg !12
  %i969 = select i1 %.not, float %i957, float %i968, !dbg !12
  %i970 = select i1 %.not, float %i968, float %i958, !dbg !12
  %i971 = select i1 %.not40, float %i969, float %i963, !dbg !12
  %i972 = bitcast float %i971 to i32, !dbg !12
  %i973 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i972), !dbg !12
  %i974 = bitcast i32 %i973 to float, !dbg !12
  %i975 = select i1 %.not40, float %i970, float %i964, !dbg !12
  %i976 = bitcast float %i975 to i32, !dbg !12
  %i977 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i976), !dbg !12
  %i978 = bitcast i32 %i977 to float, !dbg !12
  %i979 = insertelement <4 x float> poison, float %i963, i64 0, !dbg !12
  %i980 = insertelement <4 x float> %i979, float %i964, i64 1, !dbg !12
  %i981 = insertelement <4 x float> %i980, float %i974, i64 2, !dbg !12
  %i982 = insertelement <4 x float> %i981, float %i978, i64 3, !dbg !12
  %i983 = insertelement <4 x float> poison, float %i974, i64 0, !dbg !12
  %i984 = insertelement <4 x float> %i983, float %i978, i64 1, !dbg !12
  %i985 = insertelement <4 x float> %i984, float %i969, i64 2, !dbg !12
  %i986 = insertelement <4 x float> %i985, float %i970, i64 3, !dbg !12
  %i987 = select <4 x i1> %i945, <4 x float> %i982, <4 x float> %i986, !dbg !12
  %i988 = extractelement <4 x float> %i885, i64 0, !dbg !12
  %i989 = extractelement <4 x float> %i885, i64 1, !dbg !12
  %i990 = extractelement <4 x float> %i885, i64 2, !dbg !12
  %i991 = extractelement <4 x float> %i885, i64 3, !dbg !12
  %i992 = select i1 %.not, float %i989, float %i988, !dbg !12
  %i993 = bitcast float %i992 to i32, !dbg !12
  %i994 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i993), !dbg !12
  %i995 = bitcast i32 %i994 to float, !dbg !12
  %i996 = select i1 %.not, float %i988, float %i995, !dbg !12
  %i997 = select i1 %.not, float %i995, float %i989, !dbg !12
  %i998 = select i1 %.not, float %i991, float %i990, !dbg !12
  %i999 = bitcast float %i998 to i32, !dbg !12
  %i1000 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i999), !dbg !12
  %i1001 = bitcast i32 %i1000 to float, !dbg !12
  %i1002 = select i1 %.not, float %i990, float %i1001, !dbg !12
  %i1003 = select i1 %.not, float %i1001, float %i991, !dbg !12
  %i1004 = select i1 %.not40, float %i1002, float %i996, !dbg !12
  %i1005 = bitcast float %i1004 to i32, !dbg !12
  %i1006 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1005), !dbg !12
  %i1007 = bitcast i32 %i1006 to float, !dbg !12
  %i1008 = select i1 %.not40, float %i1003, float %i997, !dbg !12
  %i1009 = bitcast float %i1008 to i32, !dbg !12
  %i1010 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1009), !dbg !12
  %i1011 = bitcast i32 %i1010 to float, !dbg !12
  %i1012 = insertelement <4 x float> poison, float %i996, i64 0, !dbg !12
  %i1013 = insertelement <4 x float> %i1012, float %i997, i64 1, !dbg !12
  %i1014 = insertelement <4 x float> %i1013, float %i1007, i64 2, !dbg !12
  %i1015 = insertelement <4 x float> %i1014, float %i1011, i64 3, !dbg !12
  %i1016 = insertelement <4 x float> poison, float %i1007, i64 0, !dbg !12
  %i1017 = insertelement <4 x float> %i1016, float %i1011, i64 1, !dbg !12
  %i1018 = insertelement <4 x float> %i1017, float %i1002, i64 2, !dbg !12
  %i1019 = insertelement <4 x float> %i1018, float %i1003, i64 3, !dbg !12
  %i1020 = select <4 x i1> %i945, <4 x float> %i1015, <4 x float> %i1019, !dbg !12
  %i1021 = extractelement <4 x float> %i886, i64 0, !dbg !12
  %i1022 = extractelement <4 x float> %i886, i64 1, !dbg !12
  %i1023 = extractelement <4 x float> %i886, i64 2, !dbg !12
  %i1024 = extractelement <4 x float> %i886, i64 3, !dbg !12
  %i1025 = select i1 %.not, float %i1022, float %i1021, !dbg !12
  %i1026 = bitcast float %i1025 to i32, !dbg !12
  %i1027 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1026), !dbg !12
  %i1028 = bitcast i32 %i1027 to float, !dbg !12
  %i1029 = select i1 %.not, float %i1021, float %i1028, !dbg !12
  %i1030 = select i1 %.not, float %i1028, float %i1022, !dbg !12
  %i1031 = select i1 %.not, float %i1024, float %i1023, !dbg !12
  %i1032 = bitcast float %i1031 to i32, !dbg !12
  %i1033 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1032), !dbg !12
  %i1034 = bitcast i32 %i1033 to float, !dbg !12
  %i1035 = select i1 %.not, float %i1023, float %i1034, !dbg !12
  %i1036 = select i1 %.not, float %i1034, float %i1024, !dbg !12
  %i1037 = select i1 %.not40, float %i1035, float %i1029, !dbg !12
  %i1038 = bitcast float %i1037 to i32, !dbg !12
  %i1039 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1038), !dbg !12
  %i1040 = bitcast i32 %i1039 to float, !dbg !12
  %i1041 = select i1 %.not40, float %i1036, float %i1030, !dbg !12
  %i1042 = bitcast float %i1041 to i32, !dbg !12
  %i1043 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1042), !dbg !12
  %i1044 = bitcast i32 %i1043 to float, !dbg !12
  %i1045 = insertelement <4 x float> poison, float %i1029, i64 0, !dbg !12
  %i1046 = insertelement <4 x float> %i1045, float %i1030, i64 1, !dbg !12
  %i1047 = insertelement <4 x float> %i1046, float %i1040, i64 2, !dbg !12
  %i1048 = insertelement <4 x float> %i1047, float %i1044, i64 3, !dbg !12
  %i1049 = insertelement <4 x float> poison, float %i1040, i64 0, !dbg !12
  %i1050 = insertelement <4 x float> %i1049, float %i1044, i64 1, !dbg !12
  %i1051 = insertelement <4 x float> %i1050, float %i1035, i64 2, !dbg !12
  %i1052 = insertelement <4 x float> %i1051, float %i1036, i64 3, !dbg !12
  %i1053 = select <4 x i1> %i945, <4 x float> %i1048, <4 x float> %i1052, !dbg !12
  %i1054 = extractelement <4 x float> %i811, i64 0, !dbg !12
  %i1055 = extractelement <4 x float> %i811, i64 1, !dbg !12
  %i1056 = extractelement <4 x float> %i811, i64 2, !dbg !12
  %i1057 = extractelement <4 x float> %i811, i64 3, !dbg !12
  %i1058 = select i1 %.not, float %i1055, float %i1054, !dbg !12
  %i1059 = bitcast float %i1058 to i32, !dbg !12
  %i1060 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1059), !dbg !12
  %i1061 = bitcast i32 %i1060 to float, !dbg !12
  %i1062 = select i1 %.not, float %i1054, float %i1061, !dbg !12
  %i1063 = select i1 %.not, float %i1061, float %i1055, !dbg !12
  %i1064 = select i1 %.not, float %i1057, float %i1056, !dbg !12
  %i1065 = bitcast float %i1064 to i32, !dbg !12
  %i1066 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1065), !dbg !12
  %i1067 = bitcast i32 %i1066 to float, !dbg !12
  %i1068 = select i1 %.not, float %i1056, float %i1067, !dbg !12
  %i1069 = select i1 %.not, float %i1067, float %i1057, !dbg !12
  %i1070 = select i1 %.not40, float %i1068, float %i1062, !dbg !12
  %i1071 = bitcast float %i1070 to i32, !dbg !12
  %i1072 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1071), !dbg !12
  %i1073 = bitcast i32 %i1072 to float, !dbg !12
  %i1074 = select i1 %.not40, float %i1069, float %i1063, !dbg !12
  %i1075 = bitcast float %i1074 to i32, !dbg !12
  %i1076 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1075), !dbg !12
  %i1077 = bitcast i32 %i1076 to float, !dbg !12
  %i1078 = insertelement <4 x float> poison, float %i1062, i64 0, !dbg !12
  %i1079 = insertelement <4 x float> %i1078, float %i1063, i64 1, !dbg !12
  %i1080 = insertelement <4 x float> %i1079, float %i1073, i64 2, !dbg !12
  %i1081 = insertelement <4 x float> %i1080, float %i1077, i64 3, !dbg !12
  %i1082 = insertelement <4 x float> poison, float %i1073, i64 0, !dbg !12
  %i1083 = insertelement <4 x float> %i1082, float %i1077, i64 1, !dbg !12
  %i1084 = insertelement <4 x float> %i1083, float %i1068, i64 2, !dbg !12
  %i1085 = insertelement <4 x float> %i1084, float %i1069, i64 3, !dbg !12
  %i1086 = select <4 x i1> %i945, <4 x float> %i1081, <4 x float> %i1085, !dbg !12
  %i1087 = extractelement <4 x float> %i813, i64 0, !dbg !12
  %i1088 = extractelement <4 x float> %i813, i64 1, !dbg !12
  %i1089 = extractelement <4 x float> %i813, i64 2, !dbg !12
  %i1090 = extractelement <4 x float> %i813, i64 3, !dbg !12
  %i1091 = select i1 %.not, float %i1088, float %i1087, !dbg !12
  %i1092 = bitcast float %i1091 to i32, !dbg !12
  %i1093 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1092), !dbg !12
  %i1094 = bitcast i32 %i1093 to float, !dbg !12
  %i1095 = select i1 %.not, float %i1087, float %i1094, !dbg !12
  %i1096 = select i1 %.not, float %i1094, float %i1088, !dbg !12
  %i1097 = select i1 %.not, float %i1090, float %i1089, !dbg !12
  %i1098 = bitcast float %i1097 to i32, !dbg !12
  %i1099 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1098), !dbg !12
  %i1100 = bitcast i32 %i1099 to float, !dbg !12
  %i1101 = select i1 %.not, float %i1089, float %i1100, !dbg !12
  %i1102 = select i1 %.not, float %i1100, float %i1090, !dbg !12
  %i1103 = select i1 %.not40, float %i1101, float %i1095, !dbg !12
  %i1104 = bitcast float %i1103 to i32, !dbg !12
  %i1105 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1104), !dbg !12
  %i1106 = bitcast i32 %i1105 to float, !dbg !12
  %i1107 = select i1 %.not40, float %i1102, float %i1096, !dbg !12
  %i1108 = bitcast float %i1107 to i32, !dbg !12
  %i1109 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1108), !dbg !12
  %i1110 = bitcast i32 %i1109 to float, !dbg !12
  %i1111 = insertelement <4 x float> poison, float %i1095, i64 0, !dbg !12
  %i1112 = insertelement <4 x float> %i1111, float %i1096, i64 1, !dbg !12
  %i1113 = insertelement <4 x float> %i1112, float %i1106, i64 2, !dbg !12
  %i1114 = insertelement <4 x float> %i1113, float %i1110, i64 3, !dbg !12
  %i1115 = insertelement <4 x float> poison, float %i1106, i64 0, !dbg !12
  %i1116 = insertelement <4 x float> %i1115, float %i1110, i64 1, !dbg !12
  %i1117 = insertelement <4 x float> %i1116, float %i1101, i64 2, !dbg !12
  %i1118 = insertelement <4 x float> %i1117, float %i1102, i64 3, !dbg !12
  %i1119 = select <4 x i1> %i945, <4 x float> %i1114, <4 x float> %i1118, !dbg !12
  %i1120 = extractelement <4 x float> %i814, i64 0, !dbg !12
  %i1121 = extractelement <4 x float> %i814, i64 1, !dbg !12
  %i1122 = extractelement <4 x float> %i814, i64 2, !dbg !12
  %i1123 = extractelement <4 x float> %i814, i64 3, !dbg !12
  %i1124 = select i1 %.not, float %i1121, float %i1120, !dbg !12
  %i1125 = bitcast float %i1124 to i32, !dbg !12
  %i1126 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1125), !dbg !12
  %i1127 = bitcast i32 %i1126 to float, !dbg !12
  %i1128 = select i1 %.not, float %i1120, float %i1127, !dbg !12
  %i1129 = select i1 %.not, float %i1127, float %i1121, !dbg !12
  %i1130 = select i1 %.not, float %i1123, float %i1122, !dbg !12
  %i1131 = bitcast float %i1130 to i32, !dbg !12
  %i1132 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1131), !dbg !12
  %i1133 = bitcast i32 %i1132 to float, !dbg !12
  %i1134 = select i1 %.not, float %i1122, float %i1133, !dbg !12
  %i1135 = select i1 %.not, float %i1133, float %i1123, !dbg !12
  %i1136 = select i1 %.not40, float %i1134, float %i1128, !dbg !12
  %i1137 = bitcast float %i1136 to i32, !dbg !12
  %i1138 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1137), !dbg !12
  %i1139 = bitcast i32 %i1138 to float, !dbg !12
  %i1140 = select i1 %.not40, float %i1135, float %i1129, !dbg !12
  %i1141 = bitcast float %i1140 to i32, !dbg !12
  %i1142 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1141), !dbg !12
  %i1143 = bitcast i32 %i1142 to float, !dbg !12
  %i1144 = insertelement <4 x float> poison, float %i1128, i64 0, !dbg !12
  %i1145 = insertelement <4 x float> %i1144, float %i1129, i64 1, !dbg !12
  %i1146 = insertelement <4 x float> %i1145, float %i1139, i64 2, !dbg !12
  %i1147 = insertelement <4 x float> %i1146, float %i1143, i64 3, !dbg !12
  %i1148 = insertelement <4 x float> poison, float %i1139, i64 0, !dbg !12
  %i1149 = insertelement <4 x float> %i1148, float %i1143, i64 1, !dbg !12
  %i1150 = insertelement <4 x float> %i1149, float %i1134, i64 2, !dbg !12
  %i1151 = insertelement <4 x float> %i1150, float %i1135, i64 3, !dbg !12
  %i1152 = select <4 x i1> %i945, <4 x float> %i1147, <4 x float> %i1151, !dbg !12
  %i1153 = extractelement <4 x float> %i815, i64 0, !dbg !12
  %i1154 = extractelement <4 x float> %i815, i64 1, !dbg !12
  %i1155 = extractelement <4 x float> %i815, i64 2, !dbg !12
  %i1156 = extractelement <4 x float> %i815, i64 3, !dbg !12
  %i1157 = select i1 %.not, float %i1154, float %i1153, !dbg !12
  %i1158 = bitcast float %i1157 to i32, !dbg !12
  %i1159 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1158), !dbg !12
  %i1160 = bitcast i32 %i1159 to float, !dbg !12
  %i1161 = select i1 %.not, float %i1153, float %i1160, !dbg !12
  %i1162 = select i1 %.not, float %i1160, float %i1154, !dbg !12
  %i1163 = select i1 %.not, float %i1156, float %i1155, !dbg !12
  %i1164 = bitcast float %i1163 to i32, !dbg !12
  %i1165 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1164), !dbg !12
  %i1166 = bitcast i32 %i1165 to float, !dbg !12
  %i1167 = select i1 %.not, float %i1155, float %i1166, !dbg !12
  %i1168 = select i1 %.not, float %i1166, float %i1156, !dbg !12
  %i1169 = select i1 %.not40, float %i1167, float %i1161, !dbg !12
  %i1170 = bitcast float %i1169 to i32, !dbg !12
  %i1171 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1170), !dbg !12
  %i1172 = bitcast i32 %i1171 to float, !dbg !12
  %i1173 = select i1 %.not40, float %i1168, float %i1162, !dbg !12
  %i1174 = bitcast float %i1173 to i32, !dbg !12
  %i1175 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1174), !dbg !12
  %i1176 = bitcast i32 %i1175 to float, !dbg !12
  %i1177 = insertelement <4 x float> poison, float %i1161, i64 0, !dbg !12
  %i1178 = insertelement <4 x float> %i1177, float %i1162, i64 1, !dbg !12
  %i1179 = insertelement <4 x float> %i1178, float %i1172, i64 2, !dbg !12
  %i1180 = insertelement <4 x float> %i1179, float %i1176, i64 3, !dbg !12
  %i1181 = insertelement <4 x float> poison, float %i1172, i64 0, !dbg !12
  %i1182 = insertelement <4 x float> %i1181, float %i1176, i64 1, !dbg !12
  %i1183 = insertelement <4 x float> %i1182, float %i1167, i64 2, !dbg !12
  %i1184 = insertelement <4 x float> %i1183, float %i1168, i64 3, !dbg !12
  %i1185 = select <4 x i1> %i945, <4 x float> %i1180, <4 x float> %i1184, !dbg !12
  %i1186 = extractelement <4 x float> %i739, i64 0, !dbg !12
  %i1187 = extractelement <4 x float> %i739, i64 1, !dbg !12
  %i1188 = extractelement <4 x float> %i739, i64 2, !dbg !12
  %i1189 = extractelement <4 x float> %i739, i64 3, !dbg !12
  %i1190 = select i1 %.not, float %i1187, float %i1186, !dbg !12
  %i1191 = bitcast float %i1190 to i32, !dbg !12
  %i1192 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1191), !dbg !12
  %i1193 = bitcast i32 %i1192 to float, !dbg !12
  %i1194 = select i1 %.not, float %i1186, float %i1193, !dbg !12
  %i1195 = select i1 %.not, float %i1193, float %i1187, !dbg !12
  %i1196 = select i1 %.not, float %i1189, float %i1188, !dbg !12
  %i1197 = bitcast float %i1196 to i32, !dbg !12
  %i1198 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1197), !dbg !12
  %i1199 = bitcast i32 %i1198 to float, !dbg !12
  %i1200 = select i1 %.not, float %i1188, float %i1199, !dbg !12
  %i1201 = select i1 %.not, float %i1199, float %i1189, !dbg !12
  %i1202 = select i1 %.not40, float %i1200, float %i1194, !dbg !12
  %i1203 = bitcast float %i1202 to i32, !dbg !12
  %i1204 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1203), !dbg !12
  %i1205 = bitcast i32 %i1204 to float, !dbg !12
  %i1206 = select i1 %.not40, float %i1201, float %i1195, !dbg !12
  %i1207 = bitcast float %i1206 to i32, !dbg !12
  %i1208 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1207), !dbg !12
  %i1209 = bitcast i32 %i1208 to float, !dbg !12
  %i1210 = insertelement <4 x float> poison, float %i1194, i64 0, !dbg !12
  %i1211 = insertelement <4 x float> %i1210, float %i1195, i64 1, !dbg !12
  %i1212 = insertelement <4 x float> %i1211, float %i1205, i64 2, !dbg !12
  %i1213 = insertelement <4 x float> %i1212, float %i1209, i64 3, !dbg !12
  %i1214 = insertelement <4 x float> poison, float %i1205, i64 0, !dbg !12
  %i1215 = insertelement <4 x float> %i1214, float %i1209, i64 1, !dbg !12
  %i1216 = insertelement <4 x float> %i1215, float %i1200, i64 2, !dbg !12
  %i1217 = insertelement <4 x float> %i1216, float %i1201, i64 3, !dbg !12
  %i1218 = select <4 x i1> %i945, <4 x float> %i1213, <4 x float> %i1217, !dbg !12
  %i1219 = extractelement <4 x float> %i741, i64 0, !dbg !12
  %i1220 = extractelement <4 x float> %i741, i64 1, !dbg !12
  %i1221 = extractelement <4 x float> %i741, i64 2, !dbg !12
  %i1222 = extractelement <4 x float> %i741, i64 3, !dbg !12
  %i1223 = select i1 %.not, float %i1220, float %i1219, !dbg !12
  %i1224 = bitcast float %i1223 to i32, !dbg !12
  %i1225 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1224), !dbg !12
  %i1226 = bitcast i32 %i1225 to float, !dbg !12
  %i1227 = select i1 %.not, float %i1219, float %i1226, !dbg !12
  %i1228 = select i1 %.not, float %i1226, float %i1220, !dbg !12
  %i1229 = select i1 %.not, float %i1222, float %i1221, !dbg !12
  %i1230 = bitcast float %i1229 to i32, !dbg !12
  %i1231 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1230), !dbg !12
  %i1232 = bitcast i32 %i1231 to float, !dbg !12
  %i1233 = select i1 %.not, float %i1221, float %i1232, !dbg !12
  %i1234 = select i1 %.not, float %i1232, float %i1222, !dbg !12
  %i1235 = select i1 %.not40, float %i1233, float %i1227, !dbg !12
  %i1236 = bitcast float %i1235 to i32, !dbg !12
  %i1237 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1236), !dbg !12
  %i1238 = bitcast i32 %i1237 to float, !dbg !12
  %i1239 = select i1 %.not40, float %i1234, float %i1228, !dbg !12
  %i1240 = bitcast float %i1239 to i32, !dbg !12
  %i1241 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1240), !dbg !12
  %i1242 = bitcast i32 %i1241 to float, !dbg !12
  %i1243 = insertelement <4 x float> poison, float %i1227, i64 0, !dbg !12
  %i1244 = insertelement <4 x float> %i1243, float %i1228, i64 1, !dbg !12
  %i1245 = insertelement <4 x float> %i1244, float %i1238, i64 2, !dbg !12
  %i1246 = insertelement <4 x float> %i1245, float %i1242, i64 3, !dbg !12
  %i1247 = insertelement <4 x float> poison, float %i1238, i64 0, !dbg !12
  %i1248 = insertelement <4 x float> %i1247, float %i1242, i64 1, !dbg !12
  %i1249 = insertelement <4 x float> %i1248, float %i1233, i64 2, !dbg !12
  %i1250 = insertelement <4 x float> %i1249, float %i1234, i64 3, !dbg !12
  %i1251 = select <4 x i1> %i945, <4 x float> %i1246, <4 x float> %i1250, !dbg !12
  %i1252 = extractelement <4 x float> %i742, i64 0, !dbg !12
  %i1253 = extractelement <4 x float> %i742, i64 1, !dbg !12
  %i1254 = extractelement <4 x float> %i742, i64 2, !dbg !12
  %i1255 = extractelement <4 x float> %i742, i64 3, !dbg !12
  %i1256 = select i1 %.not, float %i1253, float %i1252, !dbg !12
  %i1257 = bitcast float %i1256 to i32, !dbg !12
  %i1258 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1257), !dbg !12
  %i1259 = bitcast i32 %i1258 to float, !dbg !12
  %i1260 = select i1 %.not, float %i1252, float %i1259, !dbg !12
  %i1261 = select i1 %.not, float %i1259, float %i1253, !dbg !12
  %i1262 = select i1 %.not, float %i1255, float %i1254, !dbg !12
  %i1263 = bitcast float %i1262 to i32, !dbg !12
  %i1264 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1263), !dbg !12
  %i1265 = bitcast i32 %i1264 to float, !dbg !12
  %i1266 = select i1 %.not, float %i1254, float %i1265, !dbg !12
  %i1267 = select i1 %.not, float %i1265, float %i1255, !dbg !12
  %i1268 = select i1 %.not40, float %i1266, float %i1260, !dbg !12
  %i1269 = bitcast float %i1268 to i32, !dbg !12
  %i1270 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1269), !dbg !12
  %i1271 = bitcast i32 %i1270 to float, !dbg !12
  %i1272 = select i1 %.not40, float %i1267, float %i1261, !dbg !12
  %i1273 = bitcast float %i1272 to i32, !dbg !12
  %i1274 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1273), !dbg !12
  %i1275 = bitcast i32 %i1274 to float, !dbg !12
  %i1276 = insertelement <4 x float> poison, float %i1260, i64 0, !dbg !12
  %i1277 = insertelement <4 x float> %i1276, float %i1261, i64 1, !dbg !12
  %i1278 = insertelement <4 x float> %i1277, float %i1271, i64 2, !dbg !12
  %i1279 = insertelement <4 x float> %i1278, float %i1275, i64 3, !dbg !12
  %i1280 = insertelement <4 x float> poison, float %i1271, i64 0, !dbg !12
  %i1281 = insertelement <4 x float> %i1280, float %i1275, i64 1, !dbg !12
  %i1282 = insertelement <4 x float> %i1281, float %i1266, i64 2, !dbg !12
  %i1283 = insertelement <4 x float> %i1282, float %i1267, i64 3, !dbg !12
  %i1284 = select <4 x i1> %i945, <4 x float> %i1279, <4 x float> %i1283, !dbg !12
  %i1285 = extractelement <4 x float> %i743, i64 0, !dbg !12
  %i1286 = extractelement <4 x float> %i743, i64 1, !dbg !12
  %i1287 = extractelement <4 x float> %i743, i64 2, !dbg !12
  %i1288 = extractelement <4 x float> %i743, i64 3, !dbg !12
  %i1289 = select i1 %.not, float %i1286, float %i1285, !dbg !12
  %i1290 = bitcast float %i1289 to i32, !dbg !12
  %i1291 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1290), !dbg !12
  %i1292 = bitcast i32 %i1291 to float, !dbg !12
  %i1293 = select i1 %.not, float %i1285, float %i1292, !dbg !12
  %i1294 = select i1 %.not, float %i1292, float %i1286, !dbg !12
  %i1295 = select i1 %.not, float %i1288, float %i1287, !dbg !12
  %i1296 = bitcast float %i1295 to i32, !dbg !12
  %i1297 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1296), !dbg !12
  %i1298 = bitcast i32 %i1297 to float, !dbg !12
  %i1299 = select i1 %.not, float %i1287, float %i1298, !dbg !12
  %i1300 = select i1 %.not, float %i1298, float %i1288, !dbg !12
  %i1301 = select i1 %.not40, float %i1299, float %i1293, !dbg !12
  %i1302 = bitcast float %i1301 to i32, !dbg !12
  %i1303 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1302), !dbg !12
  %i1304 = bitcast i32 %i1303 to float, !dbg !12
  %i1305 = select i1 %.not40, float %i1300, float %i1294, !dbg !12
  %i1306 = bitcast float %i1305 to i32, !dbg !12
  %i1307 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1306), !dbg !12
  %i1308 = bitcast i32 %i1307 to float, !dbg !12
  %i1309 = insertelement <4 x float> poison, float %i1293, i64 0, !dbg !12
  %i1310 = insertelement <4 x float> %i1309, float %i1294, i64 1, !dbg !12
  %i1311 = insertelement <4 x float> %i1310, float %i1304, i64 2, !dbg !12
  %i1312 = insertelement <4 x float> %i1311, float %i1308, i64 3, !dbg !12
  %i1313 = insertelement <4 x float> poison, float %i1304, i64 0, !dbg !12
  %i1314 = insertelement <4 x float> %i1313, float %i1308, i64 1, !dbg !12
  %i1315 = insertelement <4 x float> %i1314, float %i1299, i64 2, !dbg !12
  %i1316 = insertelement <4 x float> %i1315, float %i1300, i64 3, !dbg !12
  %i1317 = select <4 x i1> %i945, <4 x float> %i1312, <4 x float> %i1316, !dbg !12
  %i1318 = extractelement <4 x float> %i654, i64 0, !dbg !12
  %i1319 = extractelement <4 x float> %i654, i64 1, !dbg !12
  %i1320 = extractelement <4 x float> %i654, i64 2, !dbg !12
  %i1321 = extractelement <4 x float> %i654, i64 3, !dbg !12
  %i1322 = select i1 %.not, float %i1319, float %i1318, !dbg !12
  %i1323 = bitcast float %i1322 to i32, !dbg !12
  %i1324 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1323), !dbg !12
  %i1325 = bitcast i32 %i1324 to float, !dbg !12
  %i1326 = select i1 %.not, float %i1318, float %i1325, !dbg !12
  %i1327 = select i1 %.not, float %i1325, float %i1319, !dbg !12
  %i1328 = select i1 %.not, float %i1321, float %i1320, !dbg !12
  %i1329 = bitcast float %i1328 to i32, !dbg !12
  %i1330 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1329), !dbg !12
  %i1331 = bitcast i32 %i1330 to float, !dbg !12
  %i1332 = select i1 %.not, float %i1320, float %i1331, !dbg !12
  %i1333 = select i1 %.not, float %i1331, float %i1321, !dbg !12
  %i1334 = select i1 %.not40, float %i1332, float %i1326, !dbg !12
  %i1335 = bitcast float %i1334 to i32, !dbg !12
  %i1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1335), !dbg !12
  %i1337 = select i1 %.not40, float %i1333, float %i1327, !dbg !12
  %i1338 = bitcast float %i1337 to i32, !dbg !12
  %i1339 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1338), !dbg !12
  %i1340 = extractelement <4 x float> %i656, i64 0, !dbg !12
  %i1341 = extractelement <4 x float> %i656, i64 1, !dbg !12
  %i1342 = extractelement <4 x float> %i656, i64 2, !dbg !12
  %i1343 = extractelement <4 x float> %i656, i64 3, !dbg !12
  %i1344 = select i1 %.not, float %i1341, float %i1340, !dbg !12
  %i1345 = bitcast float %i1344 to i32, !dbg !12
  %i1346 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1345), !dbg !12
  %i1347 = bitcast i32 %i1346 to float, !dbg !12
  %i1348 = select i1 %.not, float %i1340, float %i1347, !dbg !12
  %i1349 = select i1 %.not, float %i1347, float %i1341, !dbg !12
  %i1350 = select i1 %.not, float %i1343, float %i1342, !dbg !12
  %i1351 = bitcast float %i1350 to i32, !dbg !12
  %i1352 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1351), !dbg !12
  %i1353 = bitcast i32 %i1352 to float, !dbg !12
  %i1354 = select i1 %.not, float %i1342, float %i1353, !dbg !12
  %i1355 = select i1 %.not, float %i1353, float %i1343, !dbg !12
  %i1356 = select i1 %.not40, float %i1354, float %i1348, !dbg !12
  %i1357 = bitcast float %i1356 to i32, !dbg !12
  %i1358 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1357), !dbg !12
  %i1359 = select i1 %.not40, float %i1355, float %i1349, !dbg !12
  %i1360 = bitcast float %i1359 to i32, !dbg !12
  %i1361 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1360), !dbg !12
  %i1362 = extractelement <4 x float> %i670, i64 0, !dbg !12
  %i1363 = extractelement <4 x float> %i670, i64 1, !dbg !12
  %i1364 = extractelement <4 x float> %i670, i64 2, !dbg !12
  %i1365 = extractelement <4 x float> %i670, i64 3, !dbg !12
  %i1366 = select i1 %.not, float %i1363, float %i1362, !dbg !12
  %i1367 = bitcast float %i1366 to i32, !dbg !12
  %i1368 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1367), !dbg !12
  %i1369 = bitcast i32 %i1368 to float, !dbg !12
  %i1370 = select i1 %.not, float %i1362, float %i1369, !dbg !12
  %i1371 = select i1 %.not, float %i1369, float %i1363, !dbg !12
  %i1372 = select i1 %.not, float %i1365, float %i1364, !dbg !12
  %i1373 = bitcast float %i1372 to i32, !dbg !12
  %i1374 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1373), !dbg !12
  %i1375 = bitcast i32 %i1374 to float, !dbg !12
  %i1376 = select i1 %.not, float %i1364, float %i1375, !dbg !12
  %i1377 = select i1 %.not, float %i1375, float %i1365, !dbg !12
  %i1378 = select i1 %.not40, float %i1376, float %i1370, !dbg !12
  %i1379 = bitcast float %i1378 to i32, !dbg !12
  %i1380 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1379), !dbg !12
  %i1381 = select i1 %.not40, float %i1377, float %i1371, !dbg !12
  %i1382 = bitcast float %i1381 to i32, !dbg !12
  %i1383 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1382), !dbg !12
  %i1384 = extractelement <4 x float> %i671, i64 0, !dbg !12
  %i1385 = extractelement <4 x float> %i671, i64 1, !dbg !12
  %i1386 = extractelement <4 x float> %i671, i64 2, !dbg !12
  %i1387 = extractelement <4 x float> %i671, i64 3, !dbg !12
  %i1388 = select i1 %.not, float %i1385, float %i1384, !dbg !12
  %i1389 = bitcast float %i1388 to i32, !dbg !12
  %i1390 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1389), !dbg !12
  %i1391 = bitcast i32 %i1390 to float, !dbg !12
  %i1392 = select i1 %.not, float %i1384, float %i1391, !dbg !12
  %i1393 = select i1 %.not, float %i1391, float %i1385, !dbg !12
  %i1394 = select i1 %.not, float %i1387, float %i1386, !dbg !12
  %i1395 = bitcast float %i1394 to i32, !dbg !12
  %i1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1395), !dbg !12
  %i1397 = bitcast i32 %i1396 to float, !dbg !12
  %i1398 = select i1 %.not, float %i1386, float %i1397, !dbg !12
  %i1399 = select i1 %.not, float %i1397, float %i1387, !dbg !12
  %i1400 = select i1 %.not40, float %i1398, float %i1392, !dbg !12
  %i1401 = bitcast float %i1400 to i32, !dbg !12
  %i1402 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1401), !dbg !12
  %i1403 = select i1 %.not40, float %i1399, float %i1393, !dbg !12
  %i1404 = bitcast float %i1403 to i32, !dbg !12
  %i1405 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1404), !dbg !12
  %i1406 = shufflevector <1 x float> %i917, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %i1407 = shufflevector <1 x float> %i912, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %i1408 = bitcast i32 %i1402 to float, !dbg !12
  %i1409 = bitcast i32 %i1405 to float, !dbg !12
  %i1410 = insertelement <4 x float> poison, float %i1392, i64 0, !dbg !12
  %i1411 = insertelement <4 x float> %i1410, float %i1393, i64 1, !dbg !12
  %i1412 = insertelement <4 x float> %i1411, float %i1408, i64 2, !dbg !12
  %i1413 = insertelement <4 x float> %i1412, float %i1409, i64 3, !dbg !12
  %i1414 = insertelement <4 x float> poison, float %i1408, i64 0, !dbg !12
  %i1415 = insertelement <4 x float> %i1414, float %i1409, i64 1, !dbg !12
  %i1416 = insertelement <4 x float> %i1415, float %i1398, i64 2, !dbg !12
  %i1417 = insertelement <4 x float> %i1416, float %i1399, i64 3, !dbg !12
  %i1418 = select <4 x i1> %i945, <4 x float> %i1413, <4 x float> %i1417, !dbg !12
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %i1407, !dbg !46
  %normalized = fmul <4 x float> %row.reciprocal, %i1418, !dbg !46
  %i1419 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !47
  %i1420 = extractelement <4 x half> %i1419, i64 3, !dbg !47
  %i1421 = extractelement <4 x half> %i1419, i64 2, !dbg !47
  %i1422 = extractelement <4 x half> %i1419, i64 1, !dbg !47
  %i1423 = extractelement <4 x half> %i1419, i64 0, !dbg !47
  %i1424 = bitcast i32 %i1380 to float, !dbg !12
  %i1425 = bitcast i32 %i1383 to float, !dbg !12
  %i1426 = insertelement <4 x float> poison, float %i1370, i64 0, !dbg !12
  %i1427 = insertelement <4 x float> %i1426, float %i1371, i64 1, !dbg !12
  %i1428 = insertelement <4 x float> %i1427, float %i1424, i64 2, !dbg !12
  %i1429 = insertelement <4 x float> %i1428, float %i1425, i64 3, !dbg !12
  %i1430 = insertelement <4 x float> poison, float %i1424, i64 0, !dbg !12
  %i1431 = insertelement <4 x float> %i1430, float %i1425, i64 1, !dbg !12
  %i1432 = insertelement <4 x float> %i1431, float %i1376, i64 2, !dbg !12
  %i1433 = insertelement <4 x float> %i1432, float %i1377, i64 3, !dbg !12
  %i1434 = select <4 x i1> %i945, <4 x float> %i1429, <4 x float> %i1433, !dbg !12
  %normalized41 = fmul <4 x float> %row.reciprocal, %i1434, !dbg !46
  %i1435 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !47
  %i1436 = extractelement <4 x half> %i1435, i64 3, !dbg !47
  %i1437 = extractelement <4 x half> %i1435, i64 2, !dbg !47
  %i1438 = extractelement <4 x half> %i1435, i64 1, !dbg !47
  %i1439 = extractelement <4 x half> %i1435, i64 0, !dbg !47
  %i1440 = bitcast i32 %i1358 to float, !dbg !12
  %i1441 = bitcast i32 %i1361 to float, !dbg !12
  %i1442 = insertelement <4 x float> poison, float %i1348, i64 0, !dbg !12
  %i1443 = insertelement <4 x float> %i1442, float %i1349, i64 1, !dbg !12
  %i1444 = insertelement <4 x float> %i1443, float %i1440, i64 2, !dbg !12
  %i1445 = insertelement <4 x float> %i1444, float %i1441, i64 3, !dbg !12
  %i1446 = insertelement <4 x float> poison, float %i1440, i64 0, !dbg !12
  %i1447 = insertelement <4 x float> %i1446, float %i1441, i64 1, !dbg !12
  %i1448 = insertelement <4 x float> %i1447, float %i1354, i64 2, !dbg !12
  %i1449 = insertelement <4 x float> %i1448, float %i1355, i64 3, !dbg !12
  %i1450 = select <4 x i1> %i945, <4 x float> %i1445, <4 x float> %i1449, !dbg !12
  %row.reciprocal42 = fdiv arcp <4 x float> splat (float 1.000000e+00), %i1406, !dbg !46
  %normalized43 = fmul <4 x float> %row.reciprocal42, %i1450, !dbg !46
  %i1451 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !47
  %i1452 = extractelement <4 x half> %i1451, i64 3, !dbg !47
  %i1453 = extractelement <4 x half> %i1451, i64 2, !dbg !47
  %i1454 = extractelement <4 x half> %i1451, i64 1, !dbg !47
  %i1455 = extractelement <4 x half> %i1451, i64 0, !dbg !47
  %i1456 = bitcast i32 %i1336 to float, !dbg !12
  %i1457 = bitcast i32 %i1339 to float, !dbg !12
  %i1458 = insertelement <4 x float> poison, float %i1326, i64 0, !dbg !12
  %i1459 = insertelement <4 x float> %i1458, float %i1327, i64 1, !dbg !12
  %i1460 = insertelement <4 x float> %i1459, float %i1456, i64 2, !dbg !12
  %i1461 = insertelement <4 x float> %i1460, float %i1457, i64 3, !dbg !12
  %i1462 = insertelement <4 x float> poison, float %i1456, i64 0, !dbg !12
  %i1463 = insertelement <4 x float> %i1462, float %i1457, i64 1, !dbg !12
  %i1464 = insertelement <4 x float> %i1463, float %i1332, i64 2, !dbg !12
  %i1465 = insertelement <4 x float> %i1464, float %i1333, i64 3, !dbg !12
  %i1466 = select <4 x i1> %i945, <4 x float> %i1461, <4 x float> %i1465, !dbg !12
  %normalized44 = fmul <4 x float> %row.reciprocal42, %i1466, !dbg !46
  %i1467 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !47
  %i1468 = extractelement <4 x half> %i1467, i64 3, !dbg !47
  %i1469 = extractelement <4 x half> %i1467, i64 2, !dbg !47
  %i1470 = extractelement <4 x half> %i1467, i64 1, !dbg !47
  %i1471 = extractelement <4 x half> %i1467, i64 0, !dbg !47
  %address.thread993 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !48, !range !10, !frisk.register_lifetimes !5
  %address.local994 = zext nneg i32 %address.thread993 to i64, !dbg !9
  %i1472 = lshr i64 %address.local994, 2, !dbg !48
  %address.local997 = and i64 %i1472, 48, !dbg !48
  %address.local999 = and i64 %address.local994, 240, !dbg !48
  %address.local998 = add i64 %i32, %address.local994, !dbg !48
  %address.local1000 = sub i64 %address.local998, %address.local999, !dbg !48
  %address.local1001 = add i64 %address.local1000, %address.local997, !dbg !48
  %i1473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %arg3, i64 %i43, !dbg !48
  %address.local1004.idx = shl nuw nsw i64 %address.local1001, 8, !dbg !48
  %address.local1004 = getelementptr inbounds nuw i8, ptr addrspace(1) %i1473, i64 %address.local1004.idx, !dbg !48
  %address.local1006 = and i64 %i1472, 12, !dbg !11
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1006, !dbg !48
  store half %i1471, ptr addrspace(1) %address.local1007, align 2, !dbg !48
  %address.local1009 = or disjoint i64 %address.local1006, 1, !dbg !48
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1009, !dbg !48
  store half %i1470, ptr addrspace(1) %address.local1010, align 2, !dbg !48
  %address.local1012 = or disjoint i64 %address.local1006, 2, !dbg !48
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1012, !dbg !48
  store half %i1469, ptr addrspace(1) %address.local1013, align 2, !dbg !48
  %address.local1015 = or disjoint i64 %address.local1006, 3, !dbg !48
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1015, !dbg !48
  store half %i1468, ptr addrspace(1) %address.local1016, align 2, !dbg !48
  %address.local1018 = or disjoint i64 %address.local1006, 16, !dbg !48
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1018, !dbg !48
  store half %i1455, ptr addrspace(1) %address.local1019, align 2, !dbg !48
  %address.local1021 = or disjoint i64 %address.local1006, 17, !dbg !48
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1021, !dbg !48
  store half %i1454, ptr addrspace(1) %address.local1022, align 2, !dbg !48
  %address.local1024 = or disjoint i64 %address.local1006, 18, !dbg !48
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1024, !dbg !48
  store half %i1453, ptr addrspace(1) %address.local1025, align 2, !dbg !48
  %address.local1027 = or disjoint i64 %address.local1006, 19, !dbg !48
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1027, !dbg !48
  store half %i1452, ptr addrspace(1) %address.local1028, align 2, !dbg !48
  %address.local1032 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1004, i64 16384, !dbg !48
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1006, !dbg !48
  store half %i1439, ptr addrspace(1) %address.local1033, align 2, !dbg !48
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1009, !dbg !48
  store half %i1438, ptr addrspace(1) %address.local1035, align 2, !dbg !48
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1012, !dbg !48
  store half %i1437, ptr addrspace(1) %address.local1037, align 2, !dbg !48
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1015, !dbg !48
  store half %i1436, ptr addrspace(1) %address.local1039, align 2, !dbg !48
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1018, !dbg !48
  store half %i1423, ptr addrspace(1) %address.local1041, align 2, !dbg !48
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1021, !dbg !48
  store half %i1422, ptr addrspace(1) %address.local1043, align 2, !dbg !48
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1024, !dbg !48
  store half %i1421, ptr addrspace(1) %address.local1045, align 2, !dbg !48
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1027, !dbg !48
  store half %i1420, ptr addrspace(1) %address.local1047, align 2, !dbg !48
  %normalized46 = fmul <4 x float> %row.reciprocal, %i1317, !dbg !46
  %i1474 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !47
  %i1475 = extractelement <4 x half> %i1474, i64 3, !dbg !47
  %i1476 = extractelement <4 x half> %i1474, i64 2, !dbg !47
  %i1477 = extractelement <4 x half> %i1474, i64 1, !dbg !47
  %i1478 = extractelement <4 x half> %i1474, i64 0, !dbg !47
  %normalized47 = fmul <4 x float> %row.reciprocal, %i1284, !dbg !46
  %i1479 = fptrunc <4 x float> %normalized47 to <4 x half>, !dbg !47
  %i1480 = extractelement <4 x half> %i1479, i64 3, !dbg !47
  %i1481 = extractelement <4 x half> %i1479, i64 2, !dbg !47
  %i1482 = extractelement <4 x half> %i1479, i64 1, !dbg !47
  %i1483 = extractelement <4 x half> %i1479, i64 0, !dbg !47
  %normalized49 = fmul <4 x float> %row.reciprocal42, %i1251, !dbg !46
  %i1484 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !47
  %i1485 = extractelement <4 x half> %i1484, i64 3, !dbg !47
  %i1486 = extractelement <4 x half> %i1484, i64 2, !dbg !47
  %i1487 = extractelement <4 x half> %i1484, i64 1, !dbg !47
  %i1488 = extractelement <4 x half> %i1484, i64 0, !dbg !47
  %normalized50 = fmul <4 x float> %row.reciprocal42, %i1218, !dbg !46
  %i1489 = fptrunc <4 x float> %normalized50 to <4 x half>, !dbg !47
  %i1490 = extractelement <4 x half> %i1489, i64 3, !dbg !47
  %i1491 = extractelement <4 x half> %i1489, i64 2, !dbg !47
  %i1492 = extractelement <4 x half> %i1489, i64 1, !dbg !47
  %i1493 = extractelement <4 x half> %i1489, i64 0, !dbg !47
  %address.local1062 = or disjoint i64 %address.local1006, 32, !dbg !48
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1062, !dbg !48
  store half %i1493, ptr addrspace(1) %address.local1063, align 2, !dbg !48
  %address.local1065 = or disjoint i64 %address.local1006, 33, !dbg !48
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1065, !dbg !48
  store half %i1492, ptr addrspace(1) %address.local1066, align 2, !dbg !48
  %address.local1068 = or disjoint i64 %address.local1006, 34, !dbg !48
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1068, !dbg !48
  store half %i1491, ptr addrspace(1) %address.local1069, align 2, !dbg !48
  %address.local1071 = or disjoint i64 %address.local1006, 35, !dbg !48
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1071, !dbg !48
  store half %i1490, ptr addrspace(1) %address.local1072, align 2, !dbg !48
  %address.local1074 = or disjoint i64 %address.local1006, 48, !dbg !48
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1074, !dbg !48
  store half %i1488, ptr addrspace(1) %address.local1075, align 2, !dbg !48
  %address.local1077 = or disjoint i64 %address.local1006, 49, !dbg !48
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1077, !dbg !48
  store half %i1487, ptr addrspace(1) %address.local1078, align 2, !dbg !48
  %address.local1080 = or disjoint i64 %address.local1006, 50, !dbg !48
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1080, !dbg !48
  store half %i1486, ptr addrspace(1) %address.local1081, align 2, !dbg !48
  %address.local1083 = or i64 %i1472, 51, !dbg !48
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1083, !dbg !48
  store half %i1485, ptr addrspace(1) %address.local1084, align 2, !dbg !48
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1062, !dbg !48
  store half %i1483, ptr addrspace(1) %address.local1089, align 2, !dbg !48
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1065, !dbg !48
  store half %i1482, ptr addrspace(1) %address.local1091, align 2, !dbg !48
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1068, !dbg !48
  store half %i1481, ptr addrspace(1) %address.local1093, align 2, !dbg !48
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1071, !dbg !48
  store half %i1480, ptr addrspace(1) %address.local1095, align 2, !dbg !48
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1074, !dbg !48
  store half %i1478, ptr addrspace(1) %address.local1097, align 2, !dbg !48
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1077, !dbg !48
  store half %i1477, ptr addrspace(1) %address.local1099, align 2, !dbg !48
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1080, !dbg !48
  store half %i1476, ptr addrspace(1) %address.local1101, align 2, !dbg !48
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1083, !dbg !48
  store half %i1475, ptr addrspace(1) %address.local1103, align 2, !dbg !48
  %normalized52 = fmul <4 x float> %row.reciprocal, %i1185, !dbg !46
  %i1494 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !47
  %i1495 = extractelement <4 x half> %i1494, i64 3, !dbg !47
  %i1496 = extractelement <4 x half> %i1494, i64 2, !dbg !47
  %i1497 = extractelement <4 x half> %i1494, i64 1, !dbg !47
  %i1498 = extractelement <4 x half> %i1494, i64 0, !dbg !47
  %normalized53 = fmul <4 x float> %row.reciprocal, %i1152, !dbg !46
  %i1499 = fptrunc <4 x float> %normalized53 to <4 x half>, !dbg !47
  %i1500 = extractelement <4 x half> %i1499, i64 3, !dbg !47
  %i1501 = extractelement <4 x half> %i1499, i64 2, !dbg !47
  %i1502 = extractelement <4 x half> %i1499, i64 1, !dbg !47
  %i1503 = extractelement <4 x half> %i1499, i64 0, !dbg !47
  %normalized55 = fmul <4 x float> %row.reciprocal42, %i1119, !dbg !46
  %i1504 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !47
  %i1505 = extractelement <4 x half> %i1504, i64 3, !dbg !47
  %i1506 = extractelement <4 x half> %i1504, i64 2, !dbg !47
  %i1507 = extractelement <4 x half> %i1504, i64 1, !dbg !47
  %i1508 = extractelement <4 x half> %i1504, i64 0, !dbg !47
  %normalized56 = fmul <4 x float> %row.reciprocal42, %i1086, !dbg !46
  %i1509 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !47
  %i1510 = extractelement <4 x half> %i1509, i64 3, !dbg !47
  %i1511 = extractelement <4 x half> %i1509, i64 2, !dbg !47
  %i1512 = extractelement <4 x half> %i1509, i64 1, !dbg !47
  %i1513 = extractelement <4 x half> %i1509, i64 0, !dbg !47
  %address.local1118 = or disjoint i64 %address.local1006, 64, !dbg !48
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1118, !dbg !48
  store half %i1513, ptr addrspace(1) %address.local1119, align 2, !dbg !48
  %address.local1121 = or disjoint i64 %address.local1006, 65, !dbg !48
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1121, !dbg !48
  store half %i1512, ptr addrspace(1) %address.local1122, align 2, !dbg !48
  %address.local1124 = or disjoint i64 %address.local1006, 66, !dbg !48
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1124, !dbg !48
  store half %i1511, ptr addrspace(1) %address.local1125, align 2, !dbg !48
  %address.local1127 = or disjoint i64 %address.local1006, 67, !dbg !48
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1127, !dbg !48
  store half %i1510, ptr addrspace(1) %address.local1128, align 2, !dbg !48
  %address.local1130 = or disjoint i64 %address.local1006, 80, !dbg !48
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1130, !dbg !48
  store half %i1508, ptr addrspace(1) %address.local1131, align 2, !dbg !48
  %address.local1133 = or disjoint i64 %address.local1006, 81, !dbg !48
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1133, !dbg !48
  store half %i1507, ptr addrspace(1) %address.local1134, align 2, !dbg !48
  %address.local1136 = or disjoint i64 %address.local1006, 82, !dbg !48
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1136, !dbg !48
  store half %i1506, ptr addrspace(1) %address.local1137, align 2, !dbg !48
  %address.local1139 = or disjoint i64 %address.local1006, 83, !dbg !48
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1139, !dbg !48
  store half %i1505, ptr addrspace(1) %address.local1140, align 2, !dbg !48
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1118, !dbg !48
  store half %i1503, ptr addrspace(1) %address.local1145, align 2, !dbg !48
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1121, !dbg !48
  store half %i1502, ptr addrspace(1) %address.local1147, align 2, !dbg !48
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1124, !dbg !48
  store half %i1501, ptr addrspace(1) %address.local1149, align 2, !dbg !48
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1127, !dbg !48
  store half %i1500, ptr addrspace(1) %address.local1151, align 2, !dbg !48
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1130, !dbg !48
  store half %i1498, ptr addrspace(1) %address.local1153, align 2, !dbg !48
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1133, !dbg !48
  store half %i1497, ptr addrspace(1) %address.local1155, align 2, !dbg !48
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1136, !dbg !48
  store half %i1496, ptr addrspace(1) %address.local1157, align 2, !dbg !48
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1139, !dbg !48
  store half %i1495, ptr addrspace(1) %address.local1159, align 2, !dbg !48
  %normalized58 = fmul <4 x float> %row.reciprocal, %i1053, !dbg !46
  %i1514 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !47
  %i1515 = extractelement <4 x half> %i1514, i64 3, !dbg !47
  %i1516 = extractelement <4 x half> %i1514, i64 2, !dbg !47
  %i1517 = extractelement <4 x half> %i1514, i64 1, !dbg !47
  %i1518 = extractelement <4 x half> %i1514, i64 0, !dbg !47
  %normalized59 = fmul <4 x float> %row.reciprocal, %i1020, !dbg !46
  %i1519 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !47
  %i1520 = extractelement <4 x half> %i1519, i64 3, !dbg !47
  %i1521 = extractelement <4 x half> %i1519, i64 2, !dbg !47
  %i1522 = extractelement <4 x half> %i1519, i64 1, !dbg !47
  %i1523 = extractelement <4 x half> %i1519, i64 0, !dbg !47
  %normalized61 = fmul <4 x float> %row.reciprocal42, %i987, !dbg !46
  %i1524 = fptrunc <4 x float> %normalized61 to <4 x half>, !dbg !47
  %i1525 = extractelement <4 x half> %i1524, i64 3, !dbg !47
  %i1526 = extractelement <4 x half> %i1524, i64 2, !dbg !47
  %i1527 = extractelement <4 x half> %i1524, i64 1, !dbg !47
  %i1528 = extractelement <4 x half> %i1524, i64 0, !dbg !47
  %normalized62 = fmul <4 x float> %row.reciprocal42, %i954, !dbg !46
  %i1529 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !47
  %i1530 = extractelement <4 x half> %i1529, i64 3, !dbg !47
  %i1531 = extractelement <4 x half> %i1529, i64 2, !dbg !47
  %i1532 = extractelement <4 x half> %i1529, i64 1, !dbg !47
  %i1533 = extractelement <4 x half> %i1529, i64 0, !dbg !47
  %address.local1174 = or disjoint i64 %address.local1006, 96, !dbg !48
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1174, !dbg !48
  store half %i1533, ptr addrspace(1) %address.local1175, align 2, !dbg !48
  %address.local1177 = or disjoint i64 %address.local1006, 97, !dbg !48
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1177, !dbg !48
  store half %i1532, ptr addrspace(1) %address.local1178, align 2, !dbg !48
  %address.local1180 = or disjoint i64 %address.local1006, 98, !dbg !48
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1180, !dbg !48
  store half %i1531, ptr addrspace(1) %address.local1181, align 2, !dbg !48
  %address.local1183 = or disjoint i64 %address.local1006, 99, !dbg !48
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1183, !dbg !48
  store half %i1530, ptr addrspace(1) %address.local1184, align 2, !dbg !48
  %address.local1186 = or disjoint i64 %address.local1006, 112, !dbg !48
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1186, !dbg !48
  store half %i1528, ptr addrspace(1) %address.local1187, align 2, !dbg !48
  %address.local1189 = or disjoint i64 %address.local1006, 113, !dbg !48
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1189, !dbg !48
  store half %i1527, ptr addrspace(1) %address.local1190, align 2, !dbg !48
  %address.local1192 = or disjoint i64 %address.local1006, 114, !dbg !48
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1192, !dbg !48
  store half %i1526, ptr addrspace(1) %address.local1193, align 2, !dbg !48
  %address.local1195 = or i64 %i1472, 115, !dbg !48
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1195, !dbg !48
  store half %i1525, ptr addrspace(1) %address.local1196, align 2, !dbg !48
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1174, !dbg !48
  store half %i1523, ptr addrspace(1) %address.local1201, align 2, !dbg !48
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1177, !dbg !48
  store half %i1522, ptr addrspace(1) %address.local1203, align 2, !dbg !48
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1180, !dbg !48
  store half %i1521, ptr addrspace(1) %address.local1205, align 2, !dbg !48
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1183, !dbg !48
  store half %i1520, ptr addrspace(1) %address.local1207, align 2, !dbg !48
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1186, !dbg !48
  store half %i1518, ptr addrspace(1) %address.local1209, align 2, !dbg !48
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1189, !dbg !48
  store half %i1517, ptr addrspace(1) %address.local1211, align 2, !dbg !48
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1192, !dbg !48
  store half %i1516, ptr addrspace(1) %address.local1213, align 2, !dbg !48
  %address.local1215 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1195, !dbg !48
  store half %i1515, ptr addrspace(1) %address.local1215, align 2, !dbg !48
  ret void, !dbg !8
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.ballot.i64(i1) #4

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.lazy_softmax_base"="v1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }
attributes #12 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/baseline")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 74, column: 20, scope: !3)
!8 = !DILocation(line: 23, column: 3, scope: !3)
!9 = !DILocation(line: 67, column: 22, scope: !3)
!10 = !{i32 0, i32 256}
!11 = !DILocation(line: 69, column: 24, scope: !3)
!12 = !DILocation(line: 57, column: 24, scope: !3)
!13 = !DILocation(line: 71, column: 23, scope: !3)
!14 = !DILocation(line: 68, column: 22, scope: !3)
!15 = !{i32 0}
!16 = !{i32 1}
!17 = !{i32 2}
!18 = !{i32 3}
!19 = !{i32 4}
!20 = !{i32 5}
!21 = !{i32 6}
!22 = !{i32 7}
!23 = !{i32 8}
!24 = !{i32 9}
!25 = !{i32 10}
!26 = !{i32 11}
!27 = !{i32 12}
!28 = !{i32 13}
!29 = !{i32 14}
!30 = !{i32 15}
!31 = !{i32 16}
!32 = !{i32 17}
!33 = !{i32 18}
!34 = !{i32 19}
!35 = !{i32 20}
!36 = !{i32 21}
!37 = !{i32 22}
!38 = !DILocation(line: 86, column: 20, scope: !3)
!39 = !DILocation(line: 78, column: 22, scope: !3)
!40 = !DILocation(line: 79, column: 22, scope: !3)
!41 = !DILocation(line: 88, column: 22, scope: !3)
!42 = !DILocation(line: 89, column: 20, scope: !3)
!43 = !{i64 8192}
!44 = !DILocation(line: 81, column: 16, scope: !3)
!45 = !DILocation(line: 80, column: 20, scope: !3)
!46 = !DILocation(line: 92, column: 23, scope: !3)
!47 = !DILocation(line: 93, column: 19, scope: !3)
!48 = !DILocation(line: 94, column: 9, scope: !3)
