; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %arg, ptr addrspace(1) %arg1, ptr addrspace(1) %arg2, ptr addrspace(1) %arg3) #1 !dbg !3 !reqd_work_group_size !6 {
bb:
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !7
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !7
  %i = and i32 %kdma.lane, 31, !dbg !7
  %kdma.owner = or disjoint i32 %kdma.wave, %i, !dbg !7
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !7
  %i4 = shl nuw nsw i32 %kdma.tid, 3, !dbg !7
  %kdma.intra = and i32 %i4, 8, !dbg !7
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !7
  %i5 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %i6 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %i5), !dbg !7
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
  %i17 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !8
  %i18 = sext i32 %i17 to i64, !dbg !8
  %i19 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !8
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
  %i30 = shl nsw i64 %i26, 7, !dbg !9
  %i31 = mul nsw i64 %i27, -4096, !dbg !9
  %i32 = add i64 %i30, %i31, !dbg !9
  %i33 = shl nsw i64 %i32, 7, !dbg !9
  %.idx = shl nuw nsw i64 %i29, 20, !dbg !9
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !9, !range !10, !frisk.register_lifetimes !5
  %i34 = shl nuw nsw i32 %address.thread, 3, !dbg !9
  %i35 = and i32 %i34, 1536, !dbg !9
  %address.local399 = zext nneg i32 %i35 to i64, !dbg !9
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local399, !dbg !9
  %address.local401 = ptrtoint ptr addrspace(3) %address.local400 to i64, !dbg !9
  %address.local402 = trunc i64 %address.local401 to i32, !dbg !9
  %i36 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local402), !dbg !9
  %address.local403 = lshr i32 %address.thread, 4, !dbg !9
  %address.local404 = and i32 %address.local403, 8, !dbg !9
  %address.local405 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local406 = and i32 %address.local405, 16256, !dbg !9
  %address.local407 = or disjoint i32 %address.local404, %address.local406, !dbg !9
  %i37 = ptrtoint ptr addrspace(1) %arg to i64, !dbg !9
  %.idx1373 = shl i64 %i32, 8, !dbg !9
  %i38 = add i64 %.idx, %.idx1373, !dbg !9
  %i39 = shl nuw nsw i32 %address.local407, 1, !dbg !9
  %i40 = trunc i64 %i38 to i32, !dbg !9
  %dma.buffer.offset1371 = or disjoint i32 %i39, %i40, !dbg !9
  %i41 = insertelement <2 x i64> poison, i64 %i37, i64 0, !dbg !9
  %i42 = bitcast <2 x i64> %i41 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1372 = shufflevector <4 x i32> %i42, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %i43 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i36, i32 %dma.buffer.offset1371, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %address.local412 = or disjoint i64 %address.local399, 2048, !dbg !9
  %address.local413 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local412, !dbg !9
  %address.local414 = ptrtoint ptr addrspace(3) %address.local413 to i64, !dbg !9
  %address.local415 = trunc i64 %address.local414 to i32, !dbg !9
  %i44 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local415), !dbg !9
  %address.local421 = or disjoint i32 %address.local407, 16, !dbg !9
  %i45 = shl nuw nsw i32 %address.local421, 1, !dbg !9
  %dma.buffer.offset1369 = or disjoint i32 %i45, %i40, !dbg !9
  %i46 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i44, i32 %dma.buffer.offset1369, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %i47 = shl nuw nsw i64 %i29, 19, !dbg !9
  %i48 = add i64 %i47, %i33, !dbg !9
  %address.local424 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local399, !dbg !9
  %address.local425 = ptrtoint ptr addrspace(3) %address.local424 to i64, !dbg !9
  %address.local426 = trunc i64 %address.local425 to i32, !dbg !9
  %i49 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local426), !dbg !9
  %i50 = trunc i64 %i48 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local407, %i50, !dbg !9
  %i51 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1367 = or disjoint i32 %i51, 64, !dbg !9
  %i52 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i49, i32 %dma.buffer.offset1367, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %address.local428 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local412, !dbg !9
  %address.local429 = ptrtoint ptr addrspace(3) %address.local428 to i64, !dbg !9
  %address.local430 = trunc i64 %address.local429 to i32, !dbg !9
  %i53 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local430), !dbg !9
  %.tr1377 = or disjoint i32 %address.local421, %i50, !dbg !9
  %i54 = shl i32 %.tr1377, 1, !dbg !9
  %dma.buffer.offset1365 = or disjoint i32 %i54, 64, !dbg !9
  %i55 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i53, i32 %dma.buffer.offset1365, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %address.local432 = lshr i32 %address.thread, 2, !dbg !11
  %address.local433 = and i32 %address.local432, 4, !dbg !11
  %address.local434 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local435 = and i32 %address.local434, 384, !dbg !11
  %address.local436 = or disjoint i32 %address.local433, %address.local435, !dbg !11
  %address.local438 = and i32 %i34, 120, !dbg !11
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !11
  %address.local440 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local441 = and i32 %address.local440, 1024, !dbg !11
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !11
  %address.local443 = zext nneg i32 %address.local442 to i64, !dbg !11
  %address.local444 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local443, !dbg !11
  %i56 = load <4 x half>, ptr addrspace(3) %address.local444, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local444, i64 4096, !dbg !11
  %i57 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %i58 = fmul <4 x half> %i56, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1194 = getelementptr i8, ptr addrspace(3) %address.local444, i64 1024, !dbg !11
  %i59 = load <4 x half>, ptr addrspace(3) %lds.const.gep1194, align 8, !dbg !11
  %i60 = fmul <4 x half> %i57, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1195 = getelementptr i8, ptr addrspace(3) %address.local444, i64 5120, !dbg !11
  %i61 = load <4 x half>, ptr addrspace(3) %lds.const.gep1195, align 8, !dbg !11
  %i62 = fmul <4 x half> %i59, splat (half 1.275630e-01), !dbg !11
  %i63 = fmul <4 x half> %i61, splat (half 1.275630e-01), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1363 = or disjoint i32 %i51, 128, !dbg !9
  %i64 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i36, i32 %dma.buffer.offset1363, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %dma.buffer.offset1361 = or disjoint i32 %i54, 128, !dbg !9
  %i65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i44, i32 %dma.buffer.offset1361, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %lds.const.gep1196 = getelementptr i8, ptr addrspace(3) %address.local444, i64 8192, !dbg !11
  %i66 = load <4 x half>, ptr addrspace(3) %lds.const.gep1196, align 8, !dbg !11
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local444, i64 12288, !dbg !11
  %i67 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !11
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local444, i64 9216, !dbg !11
  %i68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !11
  %i69 = fmul <4 x half> %i66, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local444, i64 13312, !dbg !11
  %i70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !11
  %i71 = fmul <4 x half> %i67, splat (half 1.275630e-01), !dbg !11
  %i72 = fmul <4 x half> %i68, splat (half 1.275630e-01), !dbg !11
  %i73 = fmul <4 x half> %i70, splat (half 1.275630e-01), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1359 = or disjoint i32 %i51, 192, !dbg !9
  %i74 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i49, i32 %dma.buffer.offset1359, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %dma.buffer.offset1357 = or disjoint i32 %i54, 192, !dbg !9
  %i75 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i53, i32 %dma.buffer.offset1357, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %i76 = load <4 x half>, ptr addrspace(3) %address.local444, align 8, !dbg !11
  %i77 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %i78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1194, align 8, !dbg !11
  %i79 = fmul <4 x half> %i76, splat (half 1.275630e-01), !dbg !11
  %i80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1195, align 8, !dbg !11
  %i81 = fmul <4 x half> %i77, splat (half 1.275630e-01), !dbg !11
  %i82 = fmul <4 x half> %i78, splat (half 1.275630e-01), !dbg !11
  %i83 = fmul <4 x half> %i80, splat (half 1.275630e-01), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %i84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1196, align 8, !dbg !11
  %i85 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !11
  %i86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !11
  %i87 = fmul <4 x half> %i84, splat (half 1.275630e-01), !dbg !11
  %i88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !11
  %i89 = fmul <4 x half> %i85, splat (half 1.275630e-01), !dbg !11
  %i90 = fmul <4 x half> %i86, splat (half 1.275630e-01), !dbg !11
  %i91 = fmul <4 x half> %i88, splat (half 1.275630e-01), !dbg !11
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %i92 = shl nuw nsw i32 %kdma.tid, 2, !dbg !13
  %i93 = and i32 %i92, 3084, !dbg !13
  %i94 = and i32 %kdma.tid, 1, !dbg !13
  %i95 = or disjoint i32 %i93, %i94, !dbg !13
  %narrow = add nuw nsw i32 %i95, 1, !dbg !13
  %i96 = lshr i32 %narrow, 1, !dbg !13
  %i97 = zext nneg i32 %i96 to i64, !dbg !13
  %k0.prime. = shl i64 %i23, 19, !dbg !14
  %k0.prime.2040 = and i64 %k0.prime., 16252928, !dbg !14
  %k0.prime.2041 = shl nuw nsw i64 0, 14, !dbg !14
  %k0.prime.2042 = or disjoint i64 %k0.prime.2040, %k0.prime.2041, !dbg !14
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %k0.prime.kdma.source1479 = lshr i64 %k0.prime.kdma.source, 4, !dbg !14
  %k0.prime.kdma.source1480 = shl nuw nsw i64 %k0.prime.kdma.source1479, 9, !dbg !14
  %k0.prime.kdma.source1481 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !14
  %k0.prime.kdma.source1482 = add nuw nsw i64 %k0.prime.kdma.source1480, %k0.prime.kdma.source1481, !dbg !14
  %k0.prime.kdma.source1483 = lshr i64 %k0.prime.kdma.source, 5, !dbg !14
  %k0.prime.kdma.source1484 = lshr i64 %k0.prime.kdma.source, 2, !dbg !14
  %k0.prime.kdma.source1485 = and i64 %k0.prime.kdma.source1484, 1, !dbg !14
  %k0.prime.kdma.source1486 = and i64 %k0.prime.kdma.source1479, %k0.prime.kdma.source1485, !dbg !14
  %k0.prime.kdma.source1487 = or disjoint i64 %k0.prime.kdma.source1483, %k0.prime.kdma.source1486, !dbg !14
  %k0.prime.kdma.source1488 = shl nuw nsw i64 %k0.prime.kdma.source1487, 10, !dbg !14
  %k0.prime.kdma.source1489 = sub nsw i64 %k0.prime.kdma.source1482, %k0.prime.kdma.source1488, !dbg !14
  %k0.prime.kdma.source1490 = and i64 %k0.prime.kdma.source1489, 16256, !dbg !14
  %k0.prime.kdma.source1491 = lshr i32 %kdma.tid, 4, !dbg !14
  %k0.prime.kdma.source1492 = and i32 %k0.prime.kdma.source1491, 8, !dbg !14
  %k0.prime.kdma.source1493 = zext nneg i32 %k0.prime.kdma.source1492 to i64, !dbg !14
  %k0.prime.kdma.source1494 = or disjoint i64 %k0.prime.kdma.source1490, %k0.prime.kdma.source1493, !dbg !14
  %k0.prime.kdma.source1495 = or disjoint i64 %k0.prime.2042, %k0.prime.kdma.source1494, !dbg !14
  %k0.prime.kdma.source1496 = trunc nuw i64 %k0.prime.kdma.source1495 to i32, !dbg !14
  %k0.prime.kdma.source1497 = shl nuw nsw i32 %k0.prime.kdma.source1496, 1, !dbg !14
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1497, %kdma.intra, !dbg !14
  %k0.prime.2043 = ptrtoint ptr addrspace(1) %arg1 to i64, !dbg !14
  %k0.prime.2044 = insertelement <2 x i64> poison, i64 %k0.prime.2043, i64 0, !dbg !14
  %k0.prime.2045 = bitcast <2 x i64> %k0.prime.2044 to <4 x i32>, !dbg !14
  %k0.prime.dma.buffer.rsrc1356 = shufflevector <4 x i32> %k0.prime.2045, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %k0.prime.2046 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i36, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #10, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1519 = add i32 %i36, 512, !dbg !14
  %k0.prime.kdma.source1498 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %k0.prime.kdma.source1499 = lshr i64 %k0.prime.kdma.source1498, 4, !dbg !14
  %k0.prime.kdma.source1500 = shl nuw nsw i64 %k0.prime.kdma.source1499, 9, !dbg !14
  %k0.prime.kdma.source1501 = shl nuw nsw i64 %k0.prime.kdma.source1498, 7, !dbg !14
  %k0.prime.kdma.source1502 = add nuw nsw i64 %k0.prime.kdma.source1500, %k0.prime.kdma.source1501, !dbg !14
  %k0.prime.kdma.source1503 = lshr i64 %k0.prime.kdma.source1498, 5, !dbg !14
  %k0.prime.kdma.source1504 = lshr i64 %k0.prime.kdma.source1498, 2, !dbg !14
  %k0.prime.kdma.source1505 = and i64 %k0.prime.kdma.source1504, 1, !dbg !14
  %k0.prime.kdma.source1506 = and i64 %k0.prime.kdma.source1499, %k0.prime.kdma.source1505, !dbg !14
  %k0.prime.kdma.source1507 = add nuw nsw i64 %k0.prime.kdma.source1503, %k0.prime.kdma.source1506, !dbg !14
  %k0.prime.kdma.source1508 = shl nuw nsw i64 %k0.prime.kdma.source1507, 10, !dbg !14
  %k0.prime.kdma.source1509 = sub nsw i64 %k0.prime.kdma.source1502, %k0.prime.kdma.source1508, !dbg !14
  %k0.prime.kdma.source1510 = and i64 %k0.prime.kdma.source1509, 16256, !dbg !14
  %k0.prime.kdma.source1514 = or disjoint i64 %k0.prime.kdma.source1510, %k0.prime.kdma.source1493, !dbg !14
  %k0.prime.kdma.source1515 = or disjoint i64 %k0.prime.2042, %k0.prime.kdma.source1514, !dbg !14
  %k0.prime.kdma.source1516 = trunc nuw i64 %k0.prime.kdma.source1515 to i32, !dbg !14
  %k0.prime.kdma.source1517 = shl nuw nsw i32 %k0.prime.kdma.source1516, 1, !dbg !14
  %k0.prime.kdma.offset1518 = or disjoint i32 %k0.prime.kdma.source1517, %kdma.intra.hi, !dbg !14
  %k0.prime.2047 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1519, i32 %k0.prime.kdma.offset1518, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #10, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1521 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !14
  %k0.prime.kdma.source1522 = lshr i64 %k0.prime.kdma.source1521, 4, !dbg !14
  %k0.prime.kdma.source1523 = shl nuw nsw i64 %k0.prime.kdma.source1522, 9, !dbg !14
  %k0.prime.kdma.source1525 = add nuw nsw i64 %k0.prime.kdma.source1523, %k0.prime.kdma.source1481, !dbg !14
  %k0.prime.kdma.source1526 = lshr i64 %k0.prime.kdma.source1521, 5, !dbg !14
  %k0.prime.kdma.source1529 = and i64 %k0.prime.kdma.source1522, %k0.prime.kdma.source1485, !dbg !14
  %k0.prime.kdma.source1530 = or disjoint i64 %k0.prime.kdma.source1526, %k0.prime.kdma.source1529, !dbg !14
  %k0.prime.kdma.source1531 = shl nuw nsw i64 %k0.prime.kdma.source1530, 10, !dbg !14
  %k0.prime.kdma.source1532 = sub nsw i64 %k0.prime.kdma.source1525, %k0.prime.kdma.source1531, !dbg !14
  %k0.prime.kdma.source1533 = and i64 %k0.prime.kdma.source1532, 16256, !dbg !14
  %k0.prime.kdma.source1536 = or disjoint i32 %k0.prime.kdma.source1492, 16, !dbg !14
  %k0.prime.kdma.source1537 = zext nneg i32 %k0.prime.kdma.source1536 to i64, !dbg !14
  %k0.prime.kdma.source1538 = or disjoint i64 %k0.prime.kdma.source1533, %k0.prime.kdma.source1537, !dbg !14
  %k0.prime.kdma.source1539 = or disjoint i64 %k0.prime.2042, %k0.prime.kdma.source1538, !dbg !14
  %k0.prime.kdma.source1540 = trunc nuw i64 %k0.prime.kdma.source1539 to i32, !dbg !14
  %k0.prime.kdma.source1541 = shl nuw nsw i32 %k0.prime.kdma.source1540, 1, !dbg !14
  %k0.prime.kdma.offset1542 = add nuw i32 %k0.prime.kdma.source1541, %kdma.intra, !dbg !14
  %k0.prime.2048 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i44, i32 %k0.prime.kdma.offset1542, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #10, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1567 = add i32 %i44, 512, !dbg !14
  %k0.prime.kdma.source1545 = or disjoint i64 %k0.prime.kdma.source1498, 256, !dbg !14
  %k0.prime.kdma.source1546 = lshr i64 %k0.prime.kdma.source1545, 4, !dbg !14
  %k0.prime.kdma.source1547 = shl nuw nsw i64 %k0.prime.kdma.source1546, 9, !dbg !14
  %k0.prime.kdma.source1549 = add nuw nsw i64 %k0.prime.kdma.source1547, %k0.prime.kdma.source1501, !dbg !14
  %k0.prime.kdma.source1550 = lshr i64 %k0.prime.kdma.source1545, 5, !dbg !14
  %k0.prime.kdma.source1553 = and i64 %k0.prime.kdma.source1546, %k0.prime.kdma.source1505, !dbg !14
  %k0.prime.kdma.source1554 = add nuw nsw i64 %k0.prime.kdma.source1550, %k0.prime.kdma.source1553, !dbg !14
  %k0.prime.kdma.source1555 = shl nuw nsw i64 %k0.prime.kdma.source1554, 10, !dbg !14
  %k0.prime.kdma.source1556 = sub nsw i64 %k0.prime.kdma.source1549, %k0.prime.kdma.source1555, !dbg !14
  %k0.prime.kdma.source1557 = and i64 %k0.prime.kdma.source1556, 16256, !dbg !14
  %k0.prime.kdma.source1562 = or disjoint i64 %k0.prime.kdma.source1557, %k0.prime.kdma.source1537, !dbg !14
  %k0.prime.kdma.source1563 = or disjoint i64 %k0.prime.2042, %k0.prime.kdma.source1562, !dbg !14
  %k0.prime.kdma.source1564 = trunc nuw i64 %k0.prime.kdma.source1563 to i32, !dbg !14
  %k0.prime.kdma.source1565 = shl nuw nsw i32 %k0.prime.kdma.source1564, 1, !dbg !14
  %k0.prime.kdma.offset1566 = add nuw i32 %k0.prime.kdma.source1565, %kdma.intra.hi, !dbg !14
  %k0.prime.2049 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1567, i32 %k0.prime.kdma.offset1566, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #10, !dbg !14, !frisk.cross_bn_k0 !5
  br label %bb98, !dbg !12

bb98:                                             ; preds = %k0.join, %bb
  %i99 = phi i64 [ %i879, %k0.join ], [ 0, %bb ], !dbg !12
  %i100 = phi [2 x <8 x float>] [ %i947, %k0.join ], [ zeroinitializer, %bb ], !dbg !12
  %i101 = phi [2 x <1 x float>] [ %i724, %k0.join ], [ zeroinitializer, %bb ], !dbg !12
  %i102 = phi [2 x <1 x float>] [ %lazy.base.join, %k0.join ], [ zeroinitializer, %bb ], !dbg !12
  %i103 = phi [2 x <8 x float>] [ %i951, %k0.join ], [ zeroinitializer, %bb ], !dbg !12
  %i104 = phi [2 x <8 x float>] [ %i955, %k0.join ], [ zeroinitializer, %bb ], !dbg !12
  %i105 = phi [2 x <8 x float>] [ %i959, %k0.join ], [ zeroinitializer, %bb ], !dbg !12
  %i106 = icmp samesign ult i64 %i99, 32, !dbg !12
  br i1 %i106, label %bb107, label %bb960, !dbg !12

bb107:                                            ; preds = %bb98
  %i108 = shl i64 %i23, 19, !dbg !14
  %i109 = and i64 %i108, 16252928, !dbg !14
  %i110 = shl nuw nsw i64 %i99, 14, !dbg !14
  %i111 = or disjoint i64 %i109, %i110, !dbg !14
  %address.thread495 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local496 = zext nneg i32 %address.thread495 to i64, !dbg !9
  %address.local502 = lshr i64 %address.local496, 2, !dbg !14
  %i112 = ptrtoint ptr addrspace(1) %arg1 to i64, !dbg !14
  %i113 = insertelement <2 x i64> poison, i64 %i112, i64 0, !dbg !14
  %i114 = bitcast <2 x i64> %i113 to <4 x i32>, !dbg !14
  %dma.buffer.rsrc1356 = shufflevector <4 x i32> %i114, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %kdma.source1479 = lshr i64 %kdma.source, 4, !dbg !14
  %kdma.source1480 = shl nuw nsw i64 %kdma.source1479, 9, !dbg !14
  %kdma.source1481 = shl nuw nsw i64 %kdma.source, 7, !dbg !14
  %kdma.source1482 = add nuw nsw i64 %kdma.source1480, %kdma.source1481, !dbg !14
  %kdma.source1483 = lshr i64 %kdma.source, 5, !dbg !14
  %kdma.source1484 = lshr i64 %kdma.source, 2, !dbg !14
  %kdma.source1485 = and i64 %kdma.source1484, 1, !dbg !14
  %kdma.source1486 = and i64 %kdma.source1479, %kdma.source1485, !dbg !14
  %kdma.source1487 = or disjoint i64 %kdma.source1483, %kdma.source1486, !dbg !14
  %kdma.source1488 = shl nuw nsw i64 %kdma.source1487, 10, !dbg !14
  %kdma.source1489 = sub nsw i64 %kdma.source1482, %kdma.source1488, !dbg !14
  %kdma.source1490 = and i64 %kdma.source1489, 16256, !dbg !14
  %kdma.source1491 = lshr i32 %kdma.tid, 4, !dbg !14
  %kdma.source1492 = and i32 %kdma.source1491, 8, !dbg !14
  %kdma.source1493 = zext nneg i32 %kdma.source1492 to i64, !dbg !14
  %kdma.source1494 = or disjoint i64 %kdma.source1490, %kdma.source1493, !dbg !14
  %kdma.source1495 = or disjoint i64 %i111, %kdma.source1494, !dbg !14
  %kdma.source1496 = trunc nuw i64 %kdma.source1495 to i32, !dbg !14
  %kdma.source1497 = shl nuw nsw i32 %kdma.source1496, 1, !dbg !14
  %kdma.offset = or disjoint i32 %kdma.source1497, %kdma.intra, !dbg !14
  %kdma.source1498 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %kdma.source1499 = lshr i64 %kdma.source1498, 4, !dbg !14
  %kdma.source1500 = shl nuw nsw i64 %kdma.source1499, 9, !dbg !14
  %kdma.source1501 = shl nuw nsw i64 %kdma.source1498, 7, !dbg !14
  %kdma.source1502 = add nuw nsw i64 %kdma.source1500, %kdma.source1501, !dbg !14
  %kdma.source1503 = lshr i64 %kdma.source1498, 5, !dbg !14
  %kdma.source1504 = lshr i64 %kdma.source1498, 2, !dbg !14
  %kdma.source1505 = and i64 %kdma.source1504, 1, !dbg !14
  %kdma.source1506 = and i64 %kdma.source1499, %kdma.source1505, !dbg !14
  %kdma.source1507 = add nuw nsw i64 %kdma.source1503, %kdma.source1506, !dbg !14
  %kdma.source1508 = shl nuw nsw i64 %kdma.source1507, 10, !dbg !14
  %kdma.source1509 = sub nsw i64 %kdma.source1502, %kdma.source1508, !dbg !14
  %kdma.source1510 = and i64 %kdma.source1509, 16256, !dbg !14
  %kdma.source1514 = or disjoint i64 %kdma.source1510, %kdma.source1493, !dbg !14
  %kdma.source1515 = or disjoint i64 %i111, %kdma.source1514, !dbg !14
  %kdma.source1516 = trunc nuw i64 %kdma.source1515 to i32, !dbg !14
  %kdma.source1517 = shl nuw nsw i32 %kdma.source1516, 1, !dbg !14
  %kdma.offset1518 = or disjoint i32 %kdma.source1517, %kdma.intra.hi, !dbg !14
  %kdma.lds1519 = add i32 %i36, 512, !dbg !14
  %kdma.source1521 = or disjoint i64 %kdma.source, 256, !dbg !14
  %kdma.source1522 = lshr i64 %kdma.source1521, 4, !dbg !14
  %kdma.source1523 = shl nuw nsw i64 %kdma.source1522, 9, !dbg !14
  %kdma.source1525 = add nuw nsw i64 %kdma.source1523, %kdma.source1481, !dbg !14
  %kdma.source1526 = lshr i64 %kdma.source1521, 5, !dbg !14
  %kdma.source1529 = and i64 %kdma.source1522, %kdma.source1485, !dbg !14
  %kdma.source1530 = or disjoint i64 %kdma.source1526, %kdma.source1529, !dbg !14
  %kdma.source1531 = shl nuw nsw i64 %kdma.source1530, 10, !dbg !14
  %kdma.source1532 = sub nsw i64 %kdma.source1525, %kdma.source1531, !dbg !14
  %kdma.source1533 = and i64 %kdma.source1532, 16256, !dbg !14
  %kdma.source1536 = or disjoint i32 %kdma.source1492, 16, !dbg !14
  %kdma.source1537 = zext nneg i32 %kdma.source1536 to i64, !dbg !14
  %kdma.source1538 = or disjoint i64 %kdma.source1533, %kdma.source1537, !dbg !14
  %kdma.source1539 = or disjoint i64 %i111, %kdma.source1538, !dbg !14
  %kdma.source1540 = trunc nuw i64 %kdma.source1539 to i32, !dbg !14
  %kdma.source1541 = shl nuw nsw i32 %kdma.source1540, 1, !dbg !14
  %kdma.offset1542 = add nuw i32 %kdma.source1541, %kdma.intra, !dbg !14
  %kdma.source1545 = or disjoint i64 %kdma.source1498, 256, !dbg !14
  %kdma.source1546 = lshr i64 %kdma.source1545, 4, !dbg !14
  %kdma.source1547 = shl nuw nsw i64 %kdma.source1546, 9, !dbg !14
  %kdma.source1549 = add nuw nsw i64 %kdma.source1547, %kdma.source1501, !dbg !14
  %kdma.source1550 = lshr i64 %kdma.source1545, 5, !dbg !14
  %kdma.source1553 = and i64 %kdma.source1546, %kdma.source1505, !dbg !14
  %kdma.source1554 = add nuw nsw i64 %kdma.source1550, %kdma.source1553, !dbg !14
  %kdma.source1555 = shl nuw nsw i64 %kdma.source1554, 10, !dbg !14
  %kdma.source1556 = sub nsw i64 %kdma.source1549, %kdma.source1555, !dbg !14
  %kdma.source1557 = and i64 %kdma.source1556, 16256, !dbg !14
  %kdma.source1562 = or disjoint i64 %kdma.source1557, %kdma.source1537, !dbg !14
  %kdma.source1563 = or disjoint i64 %i111, %kdma.source1562, !dbg !14
  %kdma.source1564 = trunc nuw i64 %kdma.source1563 to i32, !dbg !14
  %kdma.source1565 = shl nuw nsw i32 %kdma.source1564, 1, !dbg !14
  %kdma.offset1566 = add nuw i32 %kdma.source1565, %kdma.intra.hi, !dbg !14
  %kdma.lds1567 = add i32 %i44, 512, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %i115 = or disjoint i64 %i111, 32, !dbg !14
  %kdma.source1585 = or disjoint i64 %i115, %kdma.source1494, !dbg !14
  %kdma.source1586 = trunc nuw i64 %kdma.source1585 to i32, !dbg !14
  %kdma.source1587 = shl nuw nsw i32 %kdma.source1586, 1, !dbg !14
  %kdma.offset1588 = or disjoint i32 %kdma.source1587, %kdma.intra, !dbg !14
  %i116 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i49, i32 %kdma.offset1588, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %kdma.source1607 = or disjoint i64 %i115, %kdma.source1514, !dbg !14
  %kdma.source1608 = trunc nuw i64 %kdma.source1607 to i32, !dbg !14
  %kdma.source1609 = shl nuw nsw i32 %kdma.source1608, 1, !dbg !14
  %kdma.offset1610 = or disjoint i32 %kdma.source1609, %kdma.intra.hi, !dbg !14
  %kdma.lds1611 = add i32 %i49, 512, !dbg !14
  %i117 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1611, i32 %kdma.offset1610, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %kdma.source1631 = or disjoint i64 %i115, %kdma.source1538, !dbg !14
  %kdma.source1632 = trunc nuw i64 %kdma.source1631 to i32, !dbg !14
  %kdma.source1633 = shl nuw nsw i32 %kdma.source1632, 1, !dbg !14
  %kdma.offset1634 = add nuw i32 %kdma.source1633, %kdma.intra, !dbg !14
  %i118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i53, i32 %kdma.offset1634, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %kdma.source1655 = or disjoint i64 %i115, %kdma.source1562, !dbg !14
  %kdma.source1656 = trunc nuw i64 %kdma.source1655 to i32, !dbg !14
  %kdma.source1657 = shl nuw nsw i32 %kdma.source1656, 1, !dbg !14
  %kdma.offset1658 = add nuw i32 %kdma.source1657, %kdma.intra.hi, !dbg !14
  %kdma.lds1659 = add i32 %i53, 512, !dbg !14
  %i119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1659, i32 %kdma.offset1658, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %address.local539 = shl nuw nsw i64 %address.local496, 1, !dbg !13
  %address.local540 = and i64 %address.local539, 24, !dbg !13
  %address.local542 = shl i64 %address.local496, 62, !dbg !13
  %address.local543 = ashr exact i64 %address.local542, 57, !dbg !13
  %address.local544 = and i64 %address.local543, -64, !dbg !13
  %address.local547 = and i64 %address.local502, 12, !dbg !11
  %address.local549 = and i64 %address.local496, 63, !dbg !11
  %address.local550 = lshr i64 %address.local549, 5, !dbg !11
  %address.local551 = mul nuw nsw i64 %address.local550, 1016, !dbg !11
  %.idx1394 = mul nuw nsw i64 %address.local496, 320, !dbg !13
  %i120 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1394, !dbg !13
  %.idx1395 = mul nsw i64 %address.local502, -1280, !dbg !13
  %i121 = getelementptr i8, ptr addrspace(3) %i120, i64 %.idx1395, !dbg !13
  %i122 = getelementptr [2 x i8], ptr addrspace(3) %i121, i64 %address.local540, !dbg !13
  %i123 = getelementptr [2 x i8], ptr addrspace(3) %i122, i64 %address.local544, !dbg !13
  %i124 = getelementptr [2 x i8], ptr addrspace(3) %i123, i64 %address.local547, !dbg !13
  %address.local553 = getelementptr [2 x i8], ptr addrspace(3) %i124, i64 %address.local551, !dbg !13
  %i125 = add nsw i64 %.idx1394, %.idx1395, !dbg !13
  %.idx1970 = shl nuw nsw i64 %address.local540, 1, !dbg !13
  %i126 = or disjoint i64 %i125, %.idx1970, !dbg !13
  %i127 = or disjoint i64 %address.local544, %address.local547, !dbg !13
  %i128 = shl nsw i64 %i127, 1, !dbg !13
  %i129 = add nsw i64 %i128, %i126, !dbg !13
  %address.local553.idx = mul nuw nsw i64 %address.local550, 2032, !dbg !13
  %i130 = add nsw i64 %i129, %address.local553.idx, !dbg !13
  %i131 = lshr i64 %i130, 6, !dbg !13
  %i132 = and i64 %i131, 8, !dbg !13
  %i133 = xor i64 %i130, %i132, !dbg !13
  %i134 = sub i64 %i133, %i130, !dbg !13
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local553, i64 %i134, !dbg !13
  %i135 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13
  %lazy.base = extractvalue [2 x <1 x float>] %i102, 0, !dbg !13
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %lazy.base2055 = extractvalue [2 x <1 x float>] %i102, 1, !dbg !13
  %lazy.seed2056 = shufflevector <1 x float> %lazy.base2055, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %i136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i135, <4 x float> %lazy.seed) #9, !dbg !13
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !13
  %i137 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13
  %address.local564 = shl nuw nsw i32 %address.thread495, 1, !dbg !13
  %address.local565 = and i32 %address.local564, 24, !dbg !13
  %address.local566 = shl nuw nsw i32 %address.thread495, 8, !dbg !13
  %address.local567 = and i32 %address.local566, 768, !dbg !13
  %address.local568 = or disjoint i32 %address.local565, %address.local567, !dbg !13
  %address.local569 = shl nuw nsw i32 %address.thread495, 5, !dbg !13
  %address.local570 = and i32 %address.local569, 96, !dbg !13
  %address.local571 = or disjoint i32 %address.local568, %address.local570, !dbg !13
  %address.local572 = zext nneg i32 %address.local571 to i64, !dbg !13
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %i97) #9, !dbg !13, !frisk.register_lifetimes !5
  %address.local573 = lshr i32 %address.thread495, 1, !dbg !13
  %address.local574 = and i32 %address.local573, 1, !dbg !13
  %address.local575 = zext nneg i32 %address.local574 to i64, !dbg !13
  %address.local576 = add nsw i64 %address.base, %address.local575, !dbg !13
  %address.local577 = shl i64 %address.local576, 6, !dbg !13
  %address.local578 = sub i64 %address.local572, %address.local577, !dbg !13
  %i138 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local578, !dbg !13
  %i139 = getelementptr [2 x i8], ptr addrspace(3) %i138, i64 %address.local547, !dbg !13
  %i140 = getelementptr [2 x i8], ptr addrspace(3) %i139, i64 %address.local551, !dbg !13
  %address.local582 = getelementptr i8, ptr addrspace(3) %i140, i64 64, !dbg !13
  %.idx19731976 = add i64 %address.local578, %address.local547, !dbg !13
  %i141 = shl i64 %.idx19731976, 1, !dbg !13
  %i142 = add i64 %i141, %address.local553.idx, !dbg !13
  %i143 = add i64 %i142, 64, !dbg !13
  %i144 = lshr i64 %i143, 6, !dbg !13
  %i145 = and i64 %i144, 8, !dbg !13
  %i146 = xor i64 %i143, %i145, !dbg !13
  %i147 = sub i64 %i146, %i143, !dbg !13
  %kdma.swizzled1844 = getelementptr i8, ptr addrspace(3) %address.local582, i64 %i147, !dbg !13
  %i148 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1844, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1848 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 128, !dbg !13
  %i149 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1845 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 4096, !dbg !13
  %i150 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1849 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 4224, !dbg !13
  %i151 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1846 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !13
  %i152 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1850 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !13
  %i153 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1847 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !13
  %i154 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !13, !frisk.k_lds_pair !18
  %kdma.gep1851 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !13
  %i155 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !13, !frisk.k_lds_pair !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i60, <4 x half> %i137, <4 x float> %i136) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1852 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 1024, !dbg !13
  %i157 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !13, !frisk.k_lds_pair !19
  %kdma.gep1856 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 1152, !dbg !13
  %i158 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !13, !frisk.k_lds_pair !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i148, <4 x float> %lazy.seed) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1853 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 5120, !dbg !13
  %i160 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !13, !frisk.k_lds_pair !20
  %kdma.gep1857 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 5248, !dbg !13
  %i161 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !13, !frisk.k_lds_pair !20
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i60, <4 x half> %i150, <4 x float> %i159) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1854 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !13
  %i163 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep1855 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !13
  %i164 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !13, !frisk.k_lds_pair !21
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i152, <4 x float> %lazy.seed) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i60, <4 x half> %i154, <4 x float> %i165) #9, !dbg !13
  %i167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i149, <4 x float> %lazy.seed) #9, !dbg !13
  %i168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i60, <4 x half> %i151, <4 x float> %i167) #9, !dbg !13
  %i169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i153, <4 x float> %lazy.seed) #9, !dbg !13
  %i170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i60, <4 x half> %i155, <4 x float> %i169) #9, !dbg !13
  %i171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i157, <4 x float> %lazy.seed) #9, !dbg !13
  %i172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i60, <4 x half> %i160, <4 x float> %i171) #9, !dbg !13
  %i173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i163, <4 x float> %lazy.seed) #9, !dbg !13
  %i174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i60, <4 x half> %i164, <4 x float> %i173) #9, !dbg !13
  %i175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i158, <4 x float> %lazy.seed) #9, !dbg !13
  %i176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i60, <4 x half> %i161, <4 x float> %i175) #9, !dbg !13
  %i177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i62, <4 x half> %i135, <4 x float> %lazy.seed2056) #9, !dbg !13
  %i178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i63, <4 x half> %i137, <4 x float> %i177) #9, !dbg !13
  %i179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i62, <4 x half> %i148, <4 x float> %lazy.seed2056) #9, !dbg !13
  %i180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i63, <4 x half> %i150, <4 x float> %i179) #9, !dbg !13
  %i181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i62, <4 x half> %i152, <4 x float> %lazy.seed2056) #9, !dbg !13
  %i182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i63, <4 x half> %i154, <4 x float> %i181) #9, !dbg !13
  %i183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i62, <4 x half> %i149, <4 x float> %lazy.seed2056) #9, !dbg !13
  %i184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i63, <4 x half> %i151, <4 x float> %i183) #9, !dbg !13
  %i185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i62, <4 x half> %i153, <4 x float> %lazy.seed2056) #9, !dbg !13
  %i186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i63, <4 x half> %i155, <4 x float> %i185) #9, !dbg !13
  %i187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i62, <4 x half> %i157, <4 x float> %lazy.seed2056) #9, !dbg !13
  %i188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i63, <4 x half> %i160, <4 x float> %i187) #9, !dbg !13
  %i189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i62, <4 x half> %i163, <4 x float> %lazy.seed2056) #9, !dbg !13
  %i190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i63, <4 x half> %i164, <4 x float> %i189) #9, !dbg !13
  %i191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i62, <4 x half> %i158, <4 x float> %lazy.seed2056) #9, !dbg !13
  %i192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i63, <4 x half> %i161, <4 x float> %i191) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %i193 = or disjoint i64 %i111, 64, !dbg !14
  %kdma.source1677 = or disjoint i64 %i193, %kdma.source1494, !dbg !14
  %kdma.source1678 = trunc nuw i64 %kdma.source1677 to i32, !dbg !14
  %kdma.source1679 = shl nuw nsw i32 %kdma.source1678, 1, !dbg !14
  %kdma.offset1680 = or disjoint i32 %kdma.source1679, %kdma.intra, !dbg !14
  %i194 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i36, i32 %kdma.offset1680, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %kdma.source1699 = or disjoint i64 %i193, %kdma.source1514, !dbg !14
  %kdma.source1700 = trunc nuw i64 %kdma.source1699 to i32, !dbg !14
  %kdma.source1701 = shl nuw nsw i32 %kdma.source1700, 1, !dbg !14
  %kdma.offset1702 = or disjoint i32 %kdma.source1701, %kdma.intra.hi, !dbg !14
  %i195 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1519, i32 %kdma.offset1702, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %kdma.source1723 = or disjoint i64 %i193, %kdma.source1538, !dbg !14
  %kdma.source1724 = trunc nuw i64 %kdma.source1723 to i32, !dbg !14
  %kdma.source1725 = shl nuw nsw i32 %kdma.source1724, 1, !dbg !14
  %kdma.offset1726 = add nuw i32 %kdma.source1725, %kdma.intra, !dbg !14
  %i196 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i44, i32 %kdma.offset1726, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %kdma.source1747 = or disjoint i64 %i193, %kdma.source1562, !dbg !14
  %kdma.source1748 = trunc nuw i64 %kdma.source1747 to i32, !dbg !14
  %kdma.source1749 = shl nuw nsw i32 %kdma.source1748, 1, !dbg !14
  %kdma.offset1750 = add nuw i32 %kdma.source1749, %kdma.intra.hi, !dbg !14
  %i197 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1567, i32 %kdma.offset1750, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %kdma.gep1874 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !13
  %i198 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !13
  %i199 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1875 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !13
  %i200 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !13
  %i201 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1876 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 8192, !dbg !13
  %i202 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1880 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 8320, !dbg !13
  %i203 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 12288, !dbg !13
  %i204 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1881 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 12416, !dbg !13
  %i205 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1882 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !13
  %i206 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep1886 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !13
  %i207 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !13, !frisk.k_lds_pair !26
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i198, <4 x float> %i156) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1883 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !13
  %i209 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep1887 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !13
  %i210 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !13, !frisk.k_lds_pair !27
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i200, <4 x float> %i208) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1884 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 9216, !dbg !13
  %i212 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !13, !frisk.k_lds_pair !28
  %kdma.gep1888 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 9344, !dbg !13
  %i213 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !13, !frisk.k_lds_pair !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i202, <4 x float> %i162) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1885 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 13312, !dbg !13
  %i215 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !13, !frisk.k_lds_pair !29
  %kdma.gep1889 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 13440, !dbg !13
  %i216 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !13, !frisk.k_lds_pair !29
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i204, <4 x float> %i214) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i199, <4 x float> %i166) #9, !dbg !13
  %i219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i201, <4 x float> %i218) #9, !dbg !13
  %i220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i203, <4 x float> %i168) #9, !dbg !13
  %i221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i205, <4 x float> %i220) #9, !dbg !13
  %i222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i206, <4 x float> %i170) #9, !dbg !13
  %i223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i209, <4 x float> %i222) #9, !dbg !13
  %i224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i212, <4 x float> %i172) #9, !dbg !13
  %i225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i215, <4 x float> %i224) #9, !dbg !13
  %i226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i207, <4 x float> %i174) #9, !dbg !13
  %i227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i210, <4 x float> %i226) #9, !dbg !13
  %i228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i213, <4 x float> %i176) #9, !dbg !13
  %i229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i216, <4 x float> %i228) #9, !dbg !13
  %i230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i198, <4 x float> %i178) #9, !dbg !13
  %i231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i200, <4 x float> %i230) #9, !dbg !13
  %i232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i202, <4 x float> %i180) #9, !dbg !13
  %i233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i204, <4 x float> %i232) #9, !dbg !13
  %i234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i199, <4 x float> %i182) #9, !dbg !13
  %i235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i201, <4 x float> %i234) #9, !dbg !13
  %i236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i203, <4 x float> %i184) #9, !dbg !13
  %i237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i205, <4 x float> %i236) #9, !dbg !13
  %i238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i206, <4 x float> %i186) #9, !dbg !13
  %i239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i209, <4 x float> %i238) #9, !dbg !13
  %i240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i212, <4 x float> %i188) #9, !dbg !13
  %i241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i215, <4 x float> %i240) #9, !dbg !13
  %i242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i207, <4 x float> %i190) #9, !dbg !13
  %i243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i210, <4 x float> %i242) #9, !dbg !13
  %i244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i72, <4 x half> %i213, <4 x float> %i192) #9, !dbg !13
  %i245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i73, <4 x half> %i216, <4 x float> %i244) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %i246 = or disjoint i64 %i111, 96, !dbg !14
  %kdma.source1769 = or disjoint i64 %i246, %kdma.source1494, !dbg !14
  %kdma.source1770 = trunc nuw i64 %kdma.source1769 to i32, !dbg !14
  %kdma.source1771 = shl nuw nsw i32 %kdma.source1770, 1, !dbg !14
  %kdma.offset1772 = or disjoint i32 %kdma.source1771, %kdma.intra, !dbg !14
  %i247 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i49, i32 %kdma.offset1772, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %kdma.source1791 = or disjoint i64 %i246, %kdma.source1514, !dbg !14
  %kdma.source1792 = trunc nuw i64 %kdma.source1791 to i32, !dbg !14
  %kdma.source1793 = shl nuw nsw i32 %kdma.source1792, 1, !dbg !14
  %kdma.offset1794 = or disjoint i32 %kdma.source1793, %kdma.intra.hi, !dbg !14
  %i248 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1611, i32 %kdma.offset1794, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %kdma.source1815 = or disjoint i64 %i246, %kdma.source1538, !dbg !14
  %kdma.source1816 = trunc nuw i64 %kdma.source1815 to i32, !dbg !14
  %kdma.source1817 = shl nuw nsw i32 %kdma.source1816, 1, !dbg !14
  %kdma.offset1818 = add nuw i32 %kdma.source1817, %kdma.intra, !dbg !14
  %i249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i53, i32 %kdma.offset1818, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %kdma.source1839 = or disjoint i64 %i246, %kdma.source1562, !dbg !14
  %kdma.source1840 = trunc nuw i64 %kdma.source1839 to i32, !dbg !14
  %kdma.source1841 = shl nuw nsw i32 %kdma.source1840, 1, !dbg !14
  %kdma.offset1842 = add nuw i32 %kdma.source1841, %kdma.intra.hi, !dbg !14
  %i250 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1659, i32 %kdma.offset1842, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14
  %i251 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13, !frisk.k_lds_pair !30
  %i252 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !13, !frisk.k_lds_pair !30
  %i253 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13, !frisk.k_lds_pair !31
  %i254 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !13, !frisk.k_lds_pair !31
  %i255 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1844, align 8, !dbg !13, !frisk.k_lds_pair !32
  %i256 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !13, !frisk.k_lds_pair !32
  %i257 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !13, !frisk.k_lds_pair !33
  %i258 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !13, !frisk.k_lds_pair !33
  %i259 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !13, !frisk.k_lds_pair !34
  %i260 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !13, !frisk.k_lds_pair !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i251, <4 x float> %i211) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i262 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !13, !frisk.k_lds_pair !35
  %i263 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !13, !frisk.k_lds_pair !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i253, <4 x float> %i261) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i265 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !13, !frisk.k_lds_pair !36
  %i266 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !13, !frisk.k_lds_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i255, <4 x float> %i217) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i268 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !13, !frisk.k_lds_pair !37
  %i269 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !13, !frisk.k_lds_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i257, <4 x float> %i267) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_lds_prefetch !5
  %i271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i252, <4 x float> %i219) #9, !dbg !13
  %i272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i254, <4 x float> %i271) #9, !dbg !13
  %i273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i256, <4 x float> %i221) #9, !dbg !13
  %i274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i258, <4 x float> %i273) #9, !dbg !13
  %i275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i259, <4 x float> %i223) #9, !dbg !13
  %i276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i262, <4 x float> %i275) #9, !dbg !13
  %i277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i265, <4 x float> %i225) #9, !dbg !13
  %i278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i268, <4 x float> %i277) #9, !dbg !13
  %i279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i260, <4 x float> %i227) #9, !dbg !13
  %i280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i263, <4 x float> %i279) #9, !dbg !13
  %i281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i266, <4 x float> %i229) #9, !dbg !13
  %i282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i81, <4 x half> %i269, <4 x float> %i281) #9, !dbg !13
  %i283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i251, <4 x float> %i231) #9, !dbg !13
  %i284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i253, <4 x float> %i283) #9, !dbg !13
  %i285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i255, <4 x float> %i233) #9, !dbg !13
  %i286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i257, <4 x float> %i285) #9, !dbg !13
  %i287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i252, <4 x float> %i235) #9, !dbg !13
  %i288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i254, <4 x float> %i287) #9, !dbg !13
  %i289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i256, <4 x float> %i237) #9, !dbg !13
  %i290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i258, <4 x float> %i289) #9, !dbg !13
  %i291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i259, <4 x float> %i239) #9, !dbg !13
  %i292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i262, <4 x float> %i291) #9, !dbg !13
  %i293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i265, <4 x float> %i241) #9, !dbg !13
  %i294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i268, <4 x float> %i293) #9, !dbg !13
  %i295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i260, <4 x float> %i243) #9, !dbg !13
  %i296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i263, <4 x float> %i295) #9, !dbg !13
  %i297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i82, <4 x half> %i266, <4 x float> %i245) #9, !dbg !13
  %i298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i83, <4 x half> %i269, <4 x float> %i297) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local911 = shl nuw nsw i64 %address.local496, 5, !dbg !38
  %address.local912 = and i64 %address.local911, 8064, !dbg !38
  %address.local913 = shl nuw nsw i64 %address.local496, 3, !dbg !9
  %address.local914 = shl nuw nsw i64 %address.local502, 5, !dbg !38
  %address.local915 = sub nsw i64 %address.local913, %address.local914, !dbg !38
  %address.local916 = add nsw i64 %address.local912, %address.local915, !dbg !38
  %i299 = ptrtoint ptr addrspace(1) %arg2 to i64, !dbg !38
  %.idx14041405 = add nsw i64 %i111, %address.local916, !dbg !38
  %.idx14041405.tr = trunc i64 %.idx14041405 to i32, !dbg !38
  %dma.buffer.offset1339 = shl i32 %.idx14041405.tr, 1, !dbg !38
  %i300 = insertelement <2 x i64> poison, i64 %i299, i64 0, !dbg !38
  %i301 = bitcast <2 x i64> %i300 to <4 x i32>, !dbg !38
  %dma.buffer.rsrc1340 = shufflevector <4 x i32> %i301, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !38
  %i302 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i36, i32 %dma.buffer.offset1339, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !38
  %address.local918 = shl nuw nsw i32 %address.thread495, 3, !dbg !38
  %address.local919 = and i32 %address.local918, 24, !dbg !38
  %address.local921 = and i32 %address.local569, 8064, !dbg !38
  %address.local922 = or disjoint i32 %address.local919, %address.local921, !dbg !38
  %address.local923 = or disjoint i32 %address.local922, 8192, !dbg !38
  %i303 = trunc nuw i64 %i111 to i32, !dbg !38
  %.idx14061407.tr = or disjoint i32 %address.local923, %i303, !dbg !38
  %dma.buffer.offset1337 = shl nuw nsw i32 %.idx14061407.tr, 1, !dbg !38
  %i304 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i44, i32 %dma.buffer.offset1337, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !38
  %i305 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !13, !frisk.k_tail_pair !15
  %i306 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !13, !frisk.k_tail_pair !15
  %i307 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !13, !frisk.k_tail_pair !16
  %i308 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !13, !frisk.k_tail_pair !16
  %i309 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !13, !frisk.k_tail_pair !17
  %i310 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !13, !frisk.k_tail_pair !17
  %i311 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !13, !frisk.k_tail_pair !18
  %i312 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !13, !frisk.k_tail_pair !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_tail_prefetch !5
  %i313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i87, <4 x half> %i305, <4 x float> %i264) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_tail_prefetch !5
  %i314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i307, <4 x float> %i313) #9, !dbg !13
  %i315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i87, <4 x half> %i309, <4 x float> %i270) #9, !dbg !13
  %i316 = extractelement <4 x float> %i314, i64 0, !dbg !13
  %i317 = call float @llvm.maxnum.f32(float %i316, float -inf), !dbg !7, !frisk.local_reduction !5
  %i318 = extractelement <4 x float> %i314, i64 1, !dbg !13
  %i319 = extractelement <4 x float> %i314, i64 2, !dbg !13
  %i320 = extractelement <4 x float> %i314, i64 3, !dbg !13
  %i321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i311, <4 x float> %i315) #9, !dbg !13
  %i322 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !13, !frisk.k_tail_pair !19
  %i323 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !13, !frisk.k_tail_pair !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_tail_prefetch !5
  %i324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i87, <4 x half> %i306, <4 x float> %i272) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_tail_prefetch !5
  %i325 = extractelement <4 x float> %i321, i64 0, !dbg !13
  %local.reduce133 = call float @llvm.maxnum.f32(float %i317, float %i325), !dbg !7, !frisk.local_reduction !5
  %i326 = extractelement <4 x float> %i321, i64 1, !dbg !13
  %local.reduce134 = call float @llvm.maxnum.f32(float %i318, float %i326), !dbg !7, !frisk.local_reduction !5
  %i327 = extractelement <4 x float> %i321, i64 2, !dbg !13
  %local.reduce135 = call float @llvm.maxnum.f32(float %i319, float %i327), !dbg !7, !frisk.local_reduction !5
  %i328 = extractelement <4 x float> %i321, i64 3, !dbg !13
  %local.reduce136 = call float @llvm.maxnum.f32(float %i320, float %i328), !dbg !7, !frisk.local_reduction !5
  %i329 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !13, !frisk.k_tail_pair !20
  %i330 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !13, !frisk.k_tail_pair !20
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_tail_prefetch !5
  %i331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i308, <4 x float> %i324) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_tail_prefetch !5
  %i332 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !13, !frisk.k_tail_pair !21
  %i333 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !13, !frisk.k_tail_pair !21
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_tail_prefetch !5
  %i334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i87, <4 x half> %i310, <4 x float> %i274) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_tail_prefetch !5
  %i335 = extractelement <4 x float> %i331, i64 0, !dbg !13
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %i335), !dbg !7, !frisk.local_reduction !5
  %i336 = extractelement <4 x float> %i331, i64 1, !dbg !13
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %i336), !dbg !7, !frisk.local_reduction !5
  %i337 = extractelement <4 x float> %i331, i64 2, !dbg !13
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %i337), !dbg !7, !frisk.local_reduction !5
  %i338 = extractelement <4 x float> %i331, i64 3, !dbg !13
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %i338), !dbg !7, !frisk.local_reduction !5
  %i339 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !13, !frisk.k_tail_pair !22
  %i340 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !13, !frisk.k_tail_pair !22
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_tail_prefetch !5
  %i341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i312, <4 x float> %i334) #9, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.k_tail_prefetch !5
  %i342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i87, <4 x half> %i322, <4 x float> %i276) #9, !dbg !13
  %i343 = extractelement <4 x float> %i341, i64 0, !dbg !13
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %i343), !dbg !7, !frisk.local_reduction !5
  %i344 = extractelement <4 x float> %i341, i64 1, !dbg !13
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %i344), !dbg !7, !frisk.local_reduction !5
  %i345 = extractelement <4 x float> %i341, i64 2, !dbg !13
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %i345), !dbg !7, !frisk.local_reduction !5
  %i346 = extractelement <4 x float> %i341, i64 3, !dbg !13
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %i346), !dbg !7, !frisk.local_reduction !5
  %i347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i329, <4 x float> %i342) #9, !dbg !13
  %i348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i87, <4 x half> %i332, <4 x float> %i278) #9, !dbg !13
  %i349 = extractelement <4 x float> %i347, i64 0, !dbg !13
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %i349), !dbg !7, !frisk.local_reduction !5
  %i350 = extractelement <4 x float> %i347, i64 1, !dbg !13
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %i350), !dbg !7, !frisk.local_reduction !5
  %i351 = extractelement <4 x float> %i347, i64 2, !dbg !13
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %i351), !dbg !7, !frisk.local_reduction !5
  %i352 = extractelement <4 x float> %i347, i64 3, !dbg !13
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %i352), !dbg !7, !frisk.local_reduction !5
  %i353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i339, <4 x float> %i348) #9, !dbg !13
  %i354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i87, <4 x half> %i323, <4 x float> %i280) #9, !dbg !13
  %i355 = extractelement <4 x float> %i353, i64 0, !dbg !13
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce145, float %i355), !dbg !7, !frisk.local_reduction !5
  %i356 = extractelement <4 x float> %i353, i64 1, !dbg !13
  %local.reduce150 = call float @llvm.maxnum.f32(float %local.reduce146, float %i356), !dbg !7, !frisk.local_reduction !5
  %i357 = extractelement <4 x float> %i353, i64 2, !dbg !13
  %local.reduce151 = call float @llvm.maxnum.f32(float %local.reduce147, float %i357), !dbg !7, !frisk.local_reduction !5
  %i358 = extractelement <4 x float> %i353, i64 3, !dbg !13
  %local.reduce152 = call float @llvm.maxnum.f32(float %local.reduce148, float %i358), !dbg !7, !frisk.local_reduction !5
  %i359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i330, <4 x float> %i354) #9, !dbg !13
  %i360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i87, <4 x half> %i333, <4 x float> %i282) #9, !dbg !13
  %i361 = extractelement <4 x float> %i359, i64 0, !dbg !13
  %local.reduce153 = call float @llvm.maxnum.f32(float %local.reduce149, float %i361), !dbg !7, !frisk.local_reduction !5
  %i362 = extractelement <4 x float> %i359, i64 1, !dbg !13
  %local.reduce154 = call float @llvm.maxnum.f32(float %local.reduce150, float %i362), !dbg !7, !frisk.local_reduction !5
  %i363 = extractelement <4 x float> %i359, i64 2, !dbg !13
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %i363), !dbg !7, !frisk.local_reduction !5
  %i364 = extractelement <4 x float> %i359, i64 3, !dbg !13
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %i364), !dbg !7, !frisk.local_reduction !5
  %i365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i89, <4 x half> %i340, <4 x float> %i360) #9, !dbg !13
  %i366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i90, <4 x half> %i305, <4 x float> %i284) #9, !dbg !13
  %i367 = extractelement <4 x float> %i365, i64 0, !dbg !13
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %i367), !dbg !7, !frisk.local_reduction !5
  %i368 = extractelement <4 x float> %i365, i64 1, !dbg !13
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %i368), !dbg !7, !frisk.local_reduction !5
  %i369 = extractelement <4 x float> %i365, i64 2, !dbg !13
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %i369), !dbg !7, !frisk.local_reduction !5
  %i370 = extractelement <4 x float> %i365, i64 3, !dbg !13
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %i370), !dbg !7, !frisk.local_reduction !5
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %local.reduce158), !dbg !7, !frisk.local_reduction !5
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce159, float %local.reduce160), !dbg !7, !frisk.local_reduction !5
  %i371 = call float @llvm.maxnum.f32(float %local.reduce161, float %local.reduce162), !dbg !7, !frisk.local_reduction !5
  %i372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i307, <4 x float> %i366) #9, !dbg !13
  %i373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i90, <4 x half> %i309, <4 x float> %i286) #9, !dbg !13
  %i374 = extractelement <4 x float> %i372, i64 0, !dbg !13
  %i375 = extractelement <4 x float> %i372, i64 1, !dbg !13
  %i376 = extractelement <4 x float> %i372, i64 2, !dbg !13
  %i377 = extractelement <4 x float> %i372, i64 3, !dbg !13
  %i378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i311, <4 x float> %i373) #9, !dbg !13
  %i379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i90, <4 x half> %i306, <4 x float> %i288) #9, !dbg !13
  %i380 = extractelement <4 x float> %i378, i64 0, !dbg !13
  %i381 = extractelement <4 x float> %i378, i64 1, !dbg !13
  %i382 = extractelement <4 x float> %i378, i64 2, !dbg !13
  %i383 = extractelement <4 x float> %i378, i64 3, !dbg !13
  %i384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i308, <4 x float> %i379) #9, !dbg !13
  %i385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i90, <4 x half> %i310, <4 x float> %i290) #9, !dbg !13
  %i386 = extractelement <4 x float> %i384, i64 0, !dbg !13
  %i387 = extractelement <4 x float> %i384, i64 1, !dbg !13
  %i388 = extractelement <4 x float> %i384, i64 2, !dbg !13
  %i389 = extractelement <4 x float> %i384, i64 3, !dbg !13
  %i390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i312, <4 x float> %i385) #9, !dbg !13
  %i391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i90, <4 x half> %i322, <4 x float> %i292) #9, !dbg !13
  %i392 = extractelement <4 x float> %i390, i64 0, !dbg !13
  %i393 = extractelement <4 x float> %i390, i64 1, !dbg !13
  %i394 = extractelement <4 x float> %i390, i64 2, !dbg !13
  %i395 = extractelement <4 x float> %i390, i64 3, !dbg !13
  %i396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i329, <4 x float> %i391) #9, !dbg !13
  %i397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i90, <4 x half> %i332, <4 x float> %i294) #9, !dbg !13
  %i398 = extractelement <4 x float> %i396, i64 0, !dbg !13
  %i399 = extractelement <4 x float> %i396, i64 1, !dbg !13
  %i400 = extractelement <4 x float> %i396, i64 2, !dbg !13
  %i401 = extractelement <4 x float> %i396, i64 3, !dbg !13
  %i402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i339, <4 x float> %i397) #9, !dbg !13
  %i403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i90, <4 x half> %i323, <4 x float> %i296) #9, !dbg !13
  %i404 = extractelement <4 x float> %i402, i64 0, !dbg !13
  %i405 = extractelement <4 x float> %i402, i64 1, !dbg !13
  %i406 = extractelement <4 x float> %i402, i64 2, !dbg !13
  %i407 = extractelement <4 x float> %i402, i64 3, !dbg !13
  %i408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i330, <4 x float> %i403) #9, !dbg !13
  %i409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i90, <4 x half> %i333, <4 x float> %i298) #9, !dbg !13
  %i410 = extractelement <4 x float> %i408, i64 0, !dbg !13
  %i411 = extractelement <4 x float> %i408, i64 1, !dbg !13
  %i412 = extractelement <4 x float> %i408, i64 2, !dbg !13
  %i413 = extractelement <4 x float> %i408, i64 3, !dbg !13
  %i414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i91, <4 x half> %i340, <4 x float> %i409) #9, !dbg !13
  %i415 = extractelement <4 x float> %i414, i64 0, !dbg !13
  %i416 = extractelement <4 x float> %i414, i64 1, !dbg !13
  %i417 = extractelement <4 x float> %i414, i64 2, !dbg !13
  %i418 = extractelement <4 x float> %i414, i64 3, !dbg !13
  %i419 = bitcast float %i371 to i32, !dbg !7
  %i420 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i419), !dbg !7
  %i421 = bitcast i32 %i420 to float, !dbg !7
  %i422 = call float @llvm.maxnum.f32(float %i371, float %i421), !dbg !7
  %i423 = bitcast float %i422 to i32, !dbg !7
  %i424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i423), !dbg !7
  %i425 = bitcast i32 %i424 to float, !dbg !7
  %i426 = call float @llvm.maxnum.f32(float %i422, float %i425), !dbg !7
  %i427 = call float @llvm.maxnum.f32(float %i374, float -inf), !dbg !7, !frisk.local_reduction !5
  %local.reduce102 = call float @llvm.maxnum.f32(float %i427, float %i380), !dbg !7, !frisk.local_reduction !5
  %local.reduce103 = call float @llvm.maxnum.f32(float %i375, float %i381), !dbg !7, !frisk.local_reduction !5
  %local.reduce104 = call float @llvm.maxnum.f32(float %i376, float %i382), !dbg !7, !frisk.local_reduction !5
  %local.reduce105 = call float @llvm.maxnum.f32(float %i377, float %i383), !dbg !7, !frisk.local_reduction !5
  %local.reduce106 = call float @llvm.maxnum.f32(float %local.reduce102, float %i386), !dbg !7, !frisk.local_reduction !5
  %local.reduce107 = call float @llvm.maxnum.f32(float %local.reduce103, float %i387), !dbg !7, !frisk.local_reduction !5
  %local.reduce108 = call float @llvm.maxnum.f32(float %local.reduce104, float %i388), !dbg !7, !frisk.local_reduction !5
  %local.reduce109 = call float @llvm.maxnum.f32(float %local.reduce105, float %i389), !dbg !7, !frisk.local_reduction !5
  %local.reduce110 = call float @llvm.maxnum.f32(float %local.reduce106, float %i392), !dbg !7, !frisk.local_reduction !5
  %local.reduce111 = call float @llvm.maxnum.f32(float %local.reduce107, float %i393), !dbg !7, !frisk.local_reduction !5
  %local.reduce112 = call float @llvm.maxnum.f32(float %local.reduce108, float %i394), !dbg !7, !frisk.local_reduction !5
  %local.reduce113 = call float @llvm.maxnum.f32(float %local.reduce109, float %i395), !dbg !7, !frisk.local_reduction !5
  %local.reduce114 = call float @llvm.maxnum.f32(float %local.reduce110, float %i398), !dbg !7, !frisk.local_reduction !5
  %local.reduce115 = call float @llvm.maxnum.f32(float %local.reduce111, float %i399), !dbg !7, !frisk.local_reduction !5
  %local.reduce116 = call float @llvm.maxnum.f32(float %local.reduce112, float %i400), !dbg !7, !frisk.local_reduction !5
  %local.reduce117 = call float @llvm.maxnum.f32(float %local.reduce113, float %i401), !dbg !7, !frisk.local_reduction !5
  %local.reduce118 = call float @llvm.maxnum.f32(float %local.reduce114, float %i404), !dbg !7, !frisk.local_reduction !5
  %local.reduce119 = call float @llvm.maxnum.f32(float %local.reduce115, float %i405), !dbg !7, !frisk.local_reduction !5
  %local.reduce120 = call float @llvm.maxnum.f32(float %local.reduce116, float %i406), !dbg !7, !frisk.local_reduction !5
  %local.reduce121 = call float @llvm.maxnum.f32(float %local.reduce117, float %i407), !dbg !7, !frisk.local_reduction !5
  %local.reduce122 = call float @llvm.maxnum.f32(float %local.reduce118, float %i410), !dbg !7, !frisk.local_reduction !5
  %local.reduce123 = call float @llvm.maxnum.f32(float %local.reduce119, float %i411), !dbg !7, !frisk.local_reduction !5
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %i412), !dbg !7, !frisk.local_reduction !5
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %i413), !dbg !7, !frisk.local_reduction !5
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %i415), !dbg !7, !frisk.local_reduction !5
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %i416), !dbg !7, !frisk.local_reduction !5
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %i417), !dbg !7, !frisk.local_reduction !5
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %i418), !dbg !7, !frisk.local_reduction !5
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %local.reduce127), !dbg !7, !frisk.local_reduction !5
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce128, float %local.reduce129), !dbg !7, !frisk.local_reduction !5
  %i428 = call float @llvm.maxnum.f32(float %local.reduce130, float %local.reduce131), !dbg !7, !frisk.local_reduction !5
  %i429 = bitcast float %i428 to i32, !dbg !7
  %i430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i429), !dbg !7
  %i431 = bitcast i32 %i430 to float, !dbg !7
  %i432 = call float @llvm.maxnum.f32(float %i428, float %i431), !dbg !7
  %i433 = bitcast float %i432 to i32, !dbg !7
  %i434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i433), !dbg !7
  %i435 = bitcast i32 %i434 to float, !dbg !7
  %i436 = call float @llvm.maxnum.f32(float %i432, float %i435), !dbg !7
  %lazy.peak = call float @llvm.maxnum.f32(float %i426, float %i436), !dbg !39
  %lazy.over = fcmp ogt float %lazy.peak, 8.000000e+00, !dbg !39
  %lazy.mask = call i64 @llvm.amdgcn.ballot.i64(i1 %lazy.over), !dbg !39
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !39
  %lazy.first = icmp eq i64 %i99, 0, !dbg !39
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !39
  br i1 %lazy.take, label %lazy.slow, label %lazy.join, !dbg !39

lazy.slow:                                        ; preds = %bb107
  %lazy.positive = call float @llvm.maxnum.f32(float %i426, float 0.000000e+00), !dbg !39
  %lazy.delta = select i1 %lazy.first, float %i426, float %lazy.positive, !dbg !39
  %lazy.negated = fsub float 0.000000e+00, %lazy.delta, !dbg !39
  %lazy.exp = call float @llvm.amdgcn.exp2.f32(float %lazy.negated), !dbg !39
  %lazy.alpha = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp, !dbg !39
  %lazy.positive2057 = call float @llvm.maxnum.f32(float %i436, float 0.000000e+00), !dbg !39
  %lazy.delta2058 = select i1 %lazy.first, float %i436, float %lazy.positive2057, !dbg !39
  %lazy.negated2059 = fsub float 0.000000e+00, %lazy.delta2058, !dbg !39
  %lazy.exp2060 = call float @llvm.amdgcn.exp2.f32(float %lazy.negated2059), !dbg !39
  %lazy.alpha2061 = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp2060, !dbg !39
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted = fsub <4 x float> %i314, %lazy.delta.v, !dbg !39
  %lazy.delta.v.lane2062 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v2063 = shufflevector <1 x float> %lazy.delta.v.lane2062, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2064 = fsub <4 x float> %i321, %lazy.delta.v2063, !dbg !39
  %lazy.delta.v.lane2065 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v2066 = shufflevector <1 x float> %lazy.delta.v.lane2065, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2067 = fsub <4 x float> %i331, %lazy.delta.v2066, !dbg !39
  %lazy.delta.v.lane2068 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v2069 = shufflevector <1 x float> %lazy.delta.v.lane2068, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2070 = fsub <4 x float> %i341, %lazy.delta.v2069, !dbg !39
  %lazy.delta.v.lane2071 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v2072 = shufflevector <1 x float> %lazy.delta.v.lane2071, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2073 = fsub <4 x float> %i347, %lazy.delta.v2072, !dbg !39
  %lazy.delta.v.lane2074 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v2075 = shufflevector <1 x float> %lazy.delta.v.lane2074, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2076 = fsub <4 x float> %i353, %lazy.delta.v2075, !dbg !39
  %lazy.delta.v.lane2077 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v2078 = shufflevector <1 x float> %lazy.delta.v.lane2077, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2079 = fsub <4 x float> %i359, %lazy.delta.v2078, !dbg !39
  %lazy.delta.v.lane2080 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v2081 = shufflevector <1 x float> %lazy.delta.v.lane2080, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2082 = fsub <4 x float> %i365, %lazy.delta.v2081, !dbg !39
  %lazy.delta.v.lane2083 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !39
  %lazy.delta.v2084 = shufflevector <1 x float> %lazy.delta.v.lane2083, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2085 = fsub <4 x float> %i372, %lazy.delta.v2084, !dbg !39
  %lazy.delta.v.lane2086 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !39
  %lazy.delta.v2087 = shufflevector <1 x float> %lazy.delta.v.lane2086, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2088 = fsub <4 x float> %i378, %lazy.delta.v2087, !dbg !39
  %lazy.delta.v.lane2089 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !39
  %lazy.delta.v2090 = shufflevector <1 x float> %lazy.delta.v.lane2089, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2091 = fsub <4 x float> %i384, %lazy.delta.v2090, !dbg !39
  %lazy.delta.v.lane2092 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !39
  %lazy.delta.v2093 = shufflevector <1 x float> %lazy.delta.v.lane2092, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2094 = fsub <4 x float> %i390, %lazy.delta.v2093, !dbg !39
  %lazy.delta.v.lane2095 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !39
  %lazy.delta.v2096 = shufflevector <1 x float> %lazy.delta.v.lane2095, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2097 = fsub <4 x float> %i396, %lazy.delta.v2096, !dbg !39
  %lazy.delta.v.lane2098 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !39
  %lazy.delta.v2099 = shufflevector <1 x float> %lazy.delta.v.lane2098, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2100 = fsub <4 x float> %i402, %lazy.delta.v2099, !dbg !39
  %lazy.delta.v.lane2101 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !39
  %lazy.delta.v2102 = shufflevector <1 x float> %lazy.delta.v.lane2101, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2103 = fsub <4 x float> %i408, %lazy.delta.v2102, !dbg !39
  %lazy.delta.v.lane2104 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !39
  %lazy.delta.v2105 = shufflevector <1 x float> %lazy.delta.v.lane2104, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2106 = fsub <4 x float> %i414, %lazy.delta.v2105, !dbg !39
  %lazy.state = extractvalue [2 x <1 x float>] %i101, 0, !dbg !39
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !39
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !39
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !39
  %lazy.state2107 = extractvalue [2 x <1 x float>] %i101, 1, !dbg !39
  %lazy.alpha.v.lane2108 = insertelement <1 x float> poison, float %lazy.alpha2061, i64 0, !dbg !39
  %lazy.scaled2109 = fmul <1 x float> %lazy.state2107, %lazy.alpha.v.lane2108, !dbg !39
  %lazy.state.next2110 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled2109, 1, !dbg !39
  %lazy.state2111 = extractvalue [2 x <8 x float>] %i105, 0, !dbg !39
  %lazy.alpha.v.lane2112 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !39
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane2112, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2113 = fmul <8 x float> %lazy.state2111, %lazy.alpha.v, !dbg !39
  %lazy.state.next2114 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2113, 0, !dbg !39
  %lazy.state2115 = extractvalue [2 x <8 x float>] %i105, 1, !dbg !39
  %lazy.alpha.v.lane2116 = insertelement <1 x float> poison, float %lazy.alpha2061, i64 0, !dbg !39
  %lazy.alpha.v2117 = shufflevector <1 x float> %lazy.alpha.v.lane2116, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2118 = fmul <8 x float> %lazy.state2115, %lazy.alpha.v2117, !dbg !39
  %lazy.state.next2119 = insertvalue [2 x <8 x float>] %lazy.state.next2114, <8 x float> %lazy.scaled2118, 1, !dbg !39
  %lazy.state2120 = extractvalue [2 x <8 x float>] %i104, 0, !dbg !39
  %lazy.alpha.v.lane2121 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !39
  %lazy.alpha.v2122 = shufflevector <1 x float> %lazy.alpha.v.lane2121, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2123 = fmul <8 x float> %lazy.state2120, %lazy.alpha.v2122, !dbg !39
  %lazy.state.next2124 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2123, 0, !dbg !39
  %lazy.state2125 = extractvalue [2 x <8 x float>] %i104, 1, !dbg !39
  %lazy.alpha.v.lane2126 = insertelement <1 x float> poison, float %lazy.alpha2061, i64 0, !dbg !39
  %lazy.alpha.v2127 = shufflevector <1 x float> %lazy.alpha.v.lane2126, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2128 = fmul <8 x float> %lazy.state2125, %lazy.alpha.v2127, !dbg !39
  %lazy.state.next2129 = insertvalue [2 x <8 x float>] %lazy.state.next2124, <8 x float> %lazy.scaled2128, 1, !dbg !39
  %lazy.state2130 = extractvalue [2 x <8 x float>] %i103, 0, !dbg !39
  %lazy.alpha.v.lane2131 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !39
  %lazy.alpha.v2132 = shufflevector <1 x float> %lazy.alpha.v.lane2131, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2133 = fmul <8 x float> %lazy.state2130, %lazy.alpha.v2132, !dbg !39
  %lazy.state.next2134 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2133, 0, !dbg !39
  %lazy.state2135 = extractvalue [2 x <8 x float>] %i103, 1, !dbg !39
  %lazy.alpha.v.lane2136 = insertelement <1 x float> poison, float %lazy.alpha2061, i64 0, !dbg !39
  %lazy.alpha.v2137 = shufflevector <1 x float> %lazy.alpha.v.lane2136, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2138 = fmul <8 x float> %lazy.state2135, %lazy.alpha.v2137, !dbg !39
  %lazy.state.next2139 = insertvalue [2 x <8 x float>] %lazy.state.next2134, <8 x float> %lazy.scaled2138, 1, !dbg !39
  %lazy.state2140 = extractvalue [2 x <8 x float>] %i100, 0, !dbg !39
  %lazy.alpha.v.lane2141 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !39
  %lazy.alpha.v2142 = shufflevector <1 x float> %lazy.alpha.v.lane2141, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2143 = fmul <8 x float> %lazy.state2140, %lazy.alpha.v2142, !dbg !39
  %lazy.state.next2144 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2143, 0, !dbg !39
  %lazy.state2145 = extractvalue [2 x <8 x float>] %i100, 1, !dbg !39
  %lazy.alpha.v.lane2146 = insertelement <1 x float> poison, float %lazy.alpha2061, i64 0, !dbg !39
  %lazy.alpha.v2147 = shufflevector <1 x float> %lazy.alpha.v.lane2146, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2148 = fmul <8 x float> %lazy.state2145, %lazy.alpha.v2147, !dbg !39
  %lazy.state.next2149 = insertvalue [2 x <8 x float>] %lazy.state.next2144, <8 x float> %lazy.scaled2148, 1, !dbg !39
  %lazy.base.old = extractvalue [2 x <1 x float>] %i102, 0, !dbg !39
  %lazy.delta.b.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.base.next = fsub <1 x float> %lazy.base.old, %lazy.delta.b.lane, !dbg !39
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !39
  %lazy.base.old2150 = extractvalue [2 x <1 x float>] %i102, 1, !dbg !39
  %lazy.delta.b.lane2151 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !39
  %lazy.base.next2152 = fsub <1 x float> %lazy.base.old2150, %lazy.delta.b.lane2151, !dbg !39
  %lazy.base.state2153 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2152, 1, !dbg !39
  br label %lazy.join, !dbg !39

lazy.join:                                        ; preds = %lazy.slow, %bb107
  %lazy.score = phi <4 x float> [ %i314, %bb107 ], [ %lazy.shifted, %lazy.slow ], !dbg !39
  %lazy.score2154 = phi <4 x float> [ %i321, %bb107 ], [ %lazy.shifted2064, %lazy.slow ], !dbg !39
  %lazy.score2155 = phi <4 x float> [ %i331, %bb107 ], [ %lazy.shifted2067, %lazy.slow ], !dbg !39
  %lazy.score2156 = phi <4 x float> [ %i341, %bb107 ], [ %lazy.shifted2070, %lazy.slow ], !dbg !39
  %lazy.score2157 = phi <4 x float> [ %i347, %bb107 ], [ %lazy.shifted2073, %lazy.slow ], !dbg !39
  %lazy.score2158 = phi <4 x float> [ %i353, %bb107 ], [ %lazy.shifted2076, %lazy.slow ], !dbg !39
  %lazy.score2159 = phi <4 x float> [ %i359, %bb107 ], [ %lazy.shifted2079, %lazy.slow ], !dbg !39
  %lazy.score2160 = phi <4 x float> [ %i365, %bb107 ], [ %lazy.shifted2082, %lazy.slow ], !dbg !39
  %lazy.score2161 = phi <4 x float> [ %i372, %bb107 ], [ %lazy.shifted2085, %lazy.slow ], !dbg !39
  %lazy.score2162 = phi <4 x float> [ %i378, %bb107 ], [ %lazy.shifted2088, %lazy.slow ], !dbg !39
  %lazy.score2163 = phi <4 x float> [ %i384, %bb107 ], [ %lazy.shifted2091, %lazy.slow ], !dbg !39
  %lazy.score2164 = phi <4 x float> [ %i390, %bb107 ], [ %lazy.shifted2094, %lazy.slow ], !dbg !39
  %lazy.score2165 = phi <4 x float> [ %i396, %bb107 ], [ %lazy.shifted2097, %lazy.slow ], !dbg !39
  %lazy.score2166 = phi <4 x float> [ %i402, %bb107 ], [ %lazy.shifted2100, %lazy.slow ], !dbg !39
  %lazy.score2167 = phi <4 x float> [ %i408, %bb107 ], [ %lazy.shifted2103, %lazy.slow ], !dbg !39
  %lazy.score2168 = phi <4 x float> [ %i414, %bb107 ], [ %lazy.shifted2106, %lazy.slow ], !dbg !39
  %lazy.state2169 = phi [2 x <1 x float>] [ %i101, %bb107 ], [ %lazy.state.next2110, %lazy.slow ], !dbg !39
  %lazy.state2170 = phi [2 x <8 x float>] [ %i105, %bb107 ], [ %lazy.state.next2119, %lazy.slow ], !dbg !39
  %lazy.state2171 = phi [2 x <8 x float>] [ %i104, %bb107 ], [ %lazy.state.next2129, %lazy.slow ], !dbg !39
  %lazy.state2172 = phi [2 x <8 x float>] [ %i103, %bb107 ], [ %lazy.state.next2139, %lazy.slow ], !dbg !39
  %lazy.state2173 = phi [2 x <8 x float>] [ %i100, %bb107 ], [ %lazy.state.next2149, %lazy.slow ], !dbg !39
  %lazy.base.join = phi [2 x <1 x float>] [ %i102, %bb107 ], [ %lazy.base.state2153, %lazy.slow ], !dbg !39
  %i437 = extractvalue [2 x <1 x float>] %lazy.state2169, 0, !dbg !39
  %i438 = extractelement <1 x float> %i437, i64 0, !dbg !39
  %i439 = extractvalue [2 x <1 x float>] %lazy.state2169, 1, !dbg !39
  %i440 = extractelement <1 x float> %i439, i64 0, !dbg !39
  %i441 = extractvalue [2 x <8 x float>] %lazy.state2173, 0, !dbg !40
  %i442 = shufflevector <8 x float> %i441, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !40
  %i443 = shufflevector <8 x float> %i441, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %i444 = extractvalue [2 x <8 x float>] %lazy.state2173, 1, !dbg !40
  %i445 = shufflevector <8 x float> %i444, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !40
  %i446 = shufflevector <8 x float> %i444, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !40
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address = add nsw i64 %i115, %address.local916, !dbg !38
  %vdb.address1420 = trunc i64 %vdb.address to i32, !dbg !38
  %vdb.address1421 = shl i32 %vdb.address1420, 1, !dbg !38
  %vdb.address1426 = trunc nuw i64 %i115 to i32, !dbg !38
  %vdb.address1427 = or disjoint i32 %address.local923, %vdb.address1426, !dbg !38
  %vdb.address1428 = shl nuw nsw i32 %vdb.address1427, 1, !dbg !38
  %vdb.dma.base = add i32 %i36, 8192, !dbg !41
  %i447 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1421, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !38
  %vdb.dma.base1433 = add i32 %i44, 8192, !dbg !41
  %i448 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1433, i32 %vdb.address1428, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !38
  %address.local927 = and i32 %address.local918, 120, !dbg !41
  %address.local929 = and i32 %address.local569, 1536, !dbg !41
  %address.local930 = or disjoint i32 %address.local927, %address.local929, !dbg !41
  %address.local931 = zext nneg i32 %address.local930 to i64, !dbg !41
  %address.local932 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local931, !dbg !41
  %i449 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local932, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i450 = bitcast <4 x i32> %i449 to <8 x half>, !dbg !41
  %address.local933 = and i64 %address.local496, 60, !dbg !41
  %address.local934 = add nuw nsw i64 %address.local933, %address.local496, !dbg !41
  %address.local936.idx = shl nuw nsw i64 %address.local934, 4, !dbg !41
  %address.local936 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local936.idx, !dbg !41
  %address.local937 = lshr i64 %address.local549, 4, !dbg !41
  %address.local939.idx = mul nuw nsw i64 %address.local937, 768, !dbg !41
  %address.local939 = getelementptr i8, ptr addrspace(3) %address.local936, i64 %address.local939.idx, !dbg !41
  %address.local941.idx = mul nsw i64 %address.local502, -64, !dbg !41
  %address.local941 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local941.idx, !dbg !41
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local941, i64 256, !dbg !41
  %i451 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i452 = bitcast <4 x i32> %i451 to <8 x half>, !dbg !41
  %address.local946 = getelementptr i8, ptr addrspace(3) %address.local941, i64 512, !dbg !41
  %i453 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i454 = bitcast <4 x i32> %i453 to <8 x half>, !dbg !41
  %address.local950 = getelementptr i8, ptr addrspace(3) %address.local941, i64 768, !dbg !41
  %i455 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local950, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i456 = bitcast <4 x i32> %i455 to <8 x half>, !dbg !41
  %address.local952 = getelementptr i8, ptr addrspace(3) %address.local932, i64 4096, !dbg !41
  %i457 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i458 = bitcast <4 x i32> %i457 to <8 x half>, !dbg !41
  %address.local956 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4352, !dbg !41
  %i459 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i460 = bitcast <4 x i32> %i459 to <8 x half>, !dbg !41
  %address.local960 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4608, !dbg !41
  %i461 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local960, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i462 = bitcast <4 x i32> %i461 to <8 x half>, !dbg !41
  %address.local964 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4864, !dbg !41
  %i463 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i464 = shufflevector <8 x half> %i450, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i465 = extractelement <4 x float> %lazy.score, i64 0, !dbg !43
  %i466 = call float @llvm.amdgcn.exp2.f32(float %i465), !dbg !43
  %i467 = insertelement <4 x float> poison, float %i466, i64 0, !dbg !44
  %i468 = extractelement <4 x float> %lazy.score, i64 1, !dbg !43
  %i469 = call float @llvm.amdgcn.exp2.f32(float %i468), !dbg !43
  %i470 = insertelement <4 x float> %i467, float %i469, i64 1, !dbg !44
  %i471 = extractelement <4 x float> %lazy.score, i64 2, !dbg !43
  %i472 = call float @llvm.amdgcn.exp2.f32(float %i471), !dbg !43
  %i473 = insertelement <4 x float> %i470, float %i472, i64 2, !dbg !44
  %i474 = extractelement <4 x float> %lazy.score, i64 3, !dbg !43
  %i475 = call float @llvm.amdgcn.exp2.f32(float %i474), !dbg !43
  %i476 = insertelement <4 x float> %i473, float %i475, i64 3, !dbg !44
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i476) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !44
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !44
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !44
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !44
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !44
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !44
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !44
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !44
  %i477 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !44
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i477) #9, !dbg !43, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i464, <4 x float> %i442) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i479 = shufflevector <8 x half> %i450, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i479, <4 x float> %i443) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i481 = extractelement <4 x float> %lazy.score2161, i64 0, !dbg !43
  %i482 = call float @llvm.amdgcn.exp2.f32(float %i481), !dbg !43
  %i483 = insertelement <4 x float> poison, float %i482, i64 0, !dbg !44
  %i484 = extractelement <4 x float> %lazy.score2161, i64 1, !dbg !43
  %i485 = call float @llvm.amdgcn.exp2.f32(float %i484), !dbg !43
  %i486 = insertelement <4 x float> %i483, float %i485, i64 1, !dbg !44
  %i487 = extractelement <4 x float> %lazy.score2161, i64 2, !dbg !43
  %i488 = call float @llvm.amdgcn.exp2.f32(float %i487), !dbg !43
  %i489 = insertelement <4 x float> %i486, float %i488, i64 2, !dbg !44
  %i490 = extractelement <4 x float> %lazy.score2161, i64 3, !dbg !43
  %i491 = call float @llvm.amdgcn.exp2.f32(float %i490), !dbg !43
  %i492 = insertelement <4 x float> %i489, float %i491, i64 3, !dbg !44
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i492) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02174 = extractelement <4 x float> %math.result382, i32 0, !dbg !44
  %.cvtpack.f12175 = extractelement <4 x float> %math.result382, i32 1, !dbg !44
  %.cvtpack.f22176 = extractelement <4 x float> %math.result382, i32 2, !dbg !44
  %.cvtpack.f32177 = extractelement <4 x float> %math.result382, i32 3, !dbg !44
  %.cvtpack.p02178 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02174, float %.cvtpack.f12175), !dbg !44
  %.cvtpack.p12179 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22176, float %.cvtpack.f32177), !dbg !44
  %.cvtpack.v02180 = insertelement <2 x i32> poison, i32 %.cvtpack.p02178, i32 0, !dbg !44
  %.cvtpack.v12181 = insertelement <2 x i32> %.cvtpack.v02180, i32 %.cvtpack.p12179, i32 1, !dbg !44
  %i493 = bitcast <2 x i32> %.cvtpack.v12181 to <4 x half>, !dbg !44
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i493) #9, !dbg !43, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i464, <4 x float> %i445) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i479, <4 x float> %i446) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i496 = shufflevector <8 x half> %i452, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i497 = extractelement <4 x float> %lazy.score2154, i64 0, !dbg !43
  %i498 = call float @llvm.amdgcn.exp2.f32(float %i497), !dbg !43
  %local.reduce71 = fadd float %i466, %i498, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce71) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i499 = insertelement <4 x float> poison, float %i498, i64 0, !dbg !44
  %i500 = extractelement <4 x float> %lazy.score2154, i64 1, !dbg !43
  %i501 = call float @llvm.amdgcn.exp2.f32(float %i500), !dbg !43
  %local.reduce72 = fadd float %i469, %i501, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1977 = call float asm sideeffect "", "=v,0"(float %local.reduce72) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i502 = insertelement <4 x float> %i499, float %i501, i64 1, !dbg !44
  %i503 = extractelement <4 x float> %lazy.score2154, i64 2, !dbg !43
  %i504 = call float @llvm.amdgcn.exp2.f32(float %i503), !dbg !43
  %local.reduce73 = fadd float %i472, %i504, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1978 = call float asm sideeffect "", "=v,0"(float %local.reduce73) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i505 = insertelement <4 x float> %i502, float %i504, i64 2, !dbg !44
  %i506 = extractelement <4 x float> %lazy.score2154, i64 3, !dbg !43
  %i507 = call float @llvm.amdgcn.exp2.f32(float %i506), !dbg !43
  %local.reduce74 = fadd float %i475, %i507, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1979 = call float asm sideeffect "", "=v,0"(float %local.reduce74) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i508 = insertelement <4 x float> %i505, float %i507, i64 3, !dbg !44
  %math.result368 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i508) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02182 = extractelement <4 x float> %math.result368, i32 0, !dbg !44
  %.cvtpack.f12183 = extractelement <4 x float> %math.result368, i32 1, !dbg !44
  %.cvtpack.f22184 = extractelement <4 x float> %math.result368, i32 2, !dbg !44
  %.cvtpack.f32185 = extractelement <4 x float> %math.result368, i32 3, !dbg !44
  %.cvtpack.p02186 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02182, float %.cvtpack.f12183), !dbg !44
  %.cvtpack.p12187 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22184, float %.cvtpack.f32185), !dbg !44
  %.cvtpack.v02188 = insertelement <2 x i32> poison, i32 %.cvtpack.p02186, i32 0, !dbg !44
  %.cvtpack.v12189 = insertelement <2 x i32> %.cvtpack.v02188, i32 %.cvtpack.p12187, i32 1, !dbg !44
  %i509 = bitcast <2 x i32> %.cvtpack.v12189 to <4 x half>, !dbg !44
  %math.packed369 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i509) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %i496, <4 x float> %i478) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i511 = shufflevector <8 x half> %i452, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %i511, <4 x float> %i480) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i513 = extractelement <4 x float> %lazy.score2162, i64 0, !dbg !43
  %i514 = call float @llvm.amdgcn.exp2.f32(float %i513), !dbg !43
  %local.reduce = fadd float %i482, %i514, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1980 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i515 = insertelement <4 x float> poison, float %i514, i64 0, !dbg !44
  %i516 = extractelement <4 x float> %lazy.score2162, i64 1, !dbg !43
  %i517 = call float @llvm.amdgcn.exp2.f32(float %i516), !dbg !43
  %local.reduce41 = fadd float %i485, %i517, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1981 = call float asm sideeffect "", "=v,0"(float %local.reduce41) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i518 = insertelement <4 x float> %i515, float %i517, i64 1, !dbg !44
  %i519 = extractelement <4 x float> %lazy.score2162, i64 2, !dbg !43
  %i520 = call float @llvm.amdgcn.exp2.f32(float %i519), !dbg !43
  %local.reduce42 = fadd float %i488, %i520, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1982 = call float asm sideeffect "", "=v,0"(float %local.reduce42) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i521 = insertelement <4 x float> %i518, float %i520, i64 2, !dbg !44
  %i522 = extractelement <4 x float> %lazy.score2162, i64 3, !dbg !43
  %i523 = call float @llvm.amdgcn.exp2.f32(float %i522), !dbg !43
  %local.reduce43 = fadd float %i491, %i523, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1983 = call float asm sideeffect "", "=v,0"(float %local.reduce43) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i524 = insertelement <4 x float> %i521, float %i523, i64 3, !dbg !44
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i524) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02190 = extractelement <4 x float> %math.result384, i32 0, !dbg !44
  %.cvtpack.f12191 = extractelement <4 x float> %math.result384, i32 1, !dbg !44
  %.cvtpack.f22192 = extractelement <4 x float> %math.result384, i32 2, !dbg !44
  %.cvtpack.f32193 = extractelement <4 x float> %math.result384, i32 3, !dbg !44
  %.cvtpack.p02194 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02190, float %.cvtpack.f12191), !dbg !44
  %.cvtpack.p12195 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22192, float %.cvtpack.f32193), !dbg !44
  %.cvtpack.v02196 = insertelement <2 x i32> poison, i32 %.cvtpack.p02194, i32 0, !dbg !44
  %.cvtpack.v12197 = insertelement <2 x i32> %.cvtpack.v02196, i32 %.cvtpack.p12195, i32 1, !dbg !44
  %i525 = bitcast <2 x i32> %.cvtpack.v12197 to <4 x half>, !dbg !44
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i525) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i496, <4 x float> %i494) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i511, <4 x float> %i495) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i528 = shufflevector <8 x half> %i454, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i529 = extractelement <4 x float> %lazy.score2155, i64 0, !dbg !43
  %i530 = call float @llvm.amdgcn.exp2.f32(float %i529), !dbg !43
  %local.reduce75 = fadd float %k0.sum.anchor, %i530, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1984 = call float asm sideeffect "", "=v,0"(float %local.reduce75) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i531 = insertelement <4 x float> poison, float %i530, i64 0, !dbg !44
  %i532 = extractelement <4 x float> %lazy.score2155, i64 1, !dbg !43
  %i533 = call float @llvm.amdgcn.exp2.f32(float %i532), !dbg !43
  %local.reduce76 = fadd float %k0.sum.anchor1977, %i533, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1985 = call float asm sideeffect "", "=v,0"(float %local.reduce76) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i534 = insertelement <4 x float> %i531, float %i533, i64 1, !dbg !44
  %i535 = extractelement <4 x float> %lazy.score2155, i64 2, !dbg !43
  %i536 = call float @llvm.amdgcn.exp2.f32(float %i535), !dbg !43
  %local.reduce77 = fadd float %k0.sum.anchor1978, %i536, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1986 = call float asm sideeffect "", "=v,0"(float %local.reduce77) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i537 = insertelement <4 x float> %i534, float %i536, i64 2, !dbg !44
  %i538 = extractelement <4 x float> %lazy.score2155, i64 3, !dbg !43
  %i539 = call float @llvm.amdgcn.exp2.f32(float %i538), !dbg !43
  %local.reduce78 = fadd float %k0.sum.anchor1979, %i539, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1987 = call float asm sideeffect "", "=v,0"(float %local.reduce78) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i540 = insertelement <4 x float> %i537, float %i539, i64 3, !dbg !44
  %math.result370 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i540) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02198 = extractelement <4 x float> %math.result370, i32 0, !dbg !44
  %.cvtpack.f12199 = extractelement <4 x float> %math.result370, i32 1, !dbg !44
  %.cvtpack.f22200 = extractelement <4 x float> %math.result370, i32 2, !dbg !44
  %.cvtpack.f32201 = extractelement <4 x float> %math.result370, i32 3, !dbg !44
  %.cvtpack.p02202 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02198, float %.cvtpack.f12199), !dbg !44
  %.cvtpack.p12203 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22200, float %.cvtpack.f32201), !dbg !44
  %.cvtpack.v02204 = insertelement <2 x i32> poison, i32 %.cvtpack.p02202, i32 0, !dbg !44
  %.cvtpack.v12205 = insertelement <2 x i32> %.cvtpack.v02204, i32 %.cvtpack.p12203, i32 1, !dbg !44
  %i541 = bitcast <2 x i32> %.cvtpack.v12205 to <4 x half>, !dbg !44
  %math.packed371 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i541) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %i528, <4 x float> %i510) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i543 = shufflevector <8 x half> %i454, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %i543, <4 x float> %i512) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i545 = extractelement <4 x float> %lazy.score2163, i64 0, !dbg !43
  %i546 = call float @llvm.amdgcn.exp2.f32(float %i545), !dbg !43
  %local.reduce44 = fadd float %k0.sum.anchor1980, %i546, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1988 = call float asm sideeffect "", "=v,0"(float %local.reduce44) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i547 = insertelement <4 x float> poison, float %i546, i64 0, !dbg !44
  %i548 = extractelement <4 x float> %lazy.score2163, i64 1, !dbg !43
  %i549 = call float @llvm.amdgcn.exp2.f32(float %i548), !dbg !43
  %local.reduce45 = fadd float %k0.sum.anchor1981, %i549, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1989 = call float asm sideeffect "", "=v,0"(float %local.reduce45) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i550 = insertelement <4 x float> %i547, float %i549, i64 1, !dbg !44
  %i551 = extractelement <4 x float> %lazy.score2163, i64 2, !dbg !43
  %i552 = call float @llvm.amdgcn.exp2.f32(float %i551), !dbg !43
  %local.reduce46 = fadd float %k0.sum.anchor1982, %i552, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1990 = call float asm sideeffect "", "=v,0"(float %local.reduce46) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i553 = insertelement <4 x float> %i550, float %i552, i64 2, !dbg !44
  %i554 = extractelement <4 x float> %lazy.score2163, i64 3, !dbg !43
  %i555 = call float @llvm.amdgcn.exp2.f32(float %i554), !dbg !43
  %local.reduce47 = fadd float %k0.sum.anchor1983, %i555, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1991 = call float asm sideeffect "", "=v,0"(float %local.reduce47) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i556 = insertelement <4 x float> %i553, float %i555, i64 3, !dbg !44
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i556) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02206 = extractelement <4 x float> %math.result386, i32 0, !dbg !44
  %.cvtpack.f12207 = extractelement <4 x float> %math.result386, i32 1, !dbg !44
  %.cvtpack.f22208 = extractelement <4 x float> %math.result386, i32 2, !dbg !44
  %.cvtpack.f32209 = extractelement <4 x float> %math.result386, i32 3, !dbg !44
  %.cvtpack.p02210 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02206, float %.cvtpack.f12207), !dbg !44
  %.cvtpack.p12211 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22208, float %.cvtpack.f32209), !dbg !44
  %.cvtpack.v02212 = insertelement <2 x i32> poison, i32 %.cvtpack.p02210, i32 0, !dbg !44
  %.cvtpack.v12213 = insertelement <2 x i32> %.cvtpack.v02212, i32 %.cvtpack.p12211, i32 1, !dbg !44
  %i557 = bitcast <2 x i32> %.cvtpack.v12213 to <4 x half>, !dbg !44
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i557) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i528, <4 x float> %i526) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i543, <4 x float> %i527) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i560 = shufflevector <8 x half> %i456, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i561 = extractelement <4 x float> %lazy.score2156, i64 0, !dbg !43
  %i562 = call float @llvm.amdgcn.exp2.f32(float %i561), !dbg !43
  %local.reduce79 = fadd float %k0.sum.anchor1984, %i562, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1992 = call float asm sideeffect "", "=v,0"(float %local.reduce79) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i563 = insertelement <4 x float> poison, float %i562, i64 0, !dbg !44
  %i564 = extractelement <4 x float> %lazy.score2156, i64 1, !dbg !43
  %i565 = call float @llvm.amdgcn.exp2.f32(float %i564), !dbg !43
  %local.reduce80 = fadd float %k0.sum.anchor1985, %i565, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1993 = call float asm sideeffect "", "=v,0"(float %local.reduce80) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i566 = insertelement <4 x float> %i563, float %i565, i64 1, !dbg !44
  %i567 = extractelement <4 x float> %lazy.score2156, i64 2, !dbg !43
  %i568 = call float @llvm.amdgcn.exp2.f32(float %i567), !dbg !43
  %local.reduce81 = fadd float %k0.sum.anchor1986, %i568, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1994 = call float asm sideeffect "", "=v,0"(float %local.reduce81) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i569 = insertelement <4 x float> %i566, float %i568, i64 2, !dbg !44
  %i570 = extractelement <4 x float> %lazy.score2156, i64 3, !dbg !43
  %i571 = call float @llvm.amdgcn.exp2.f32(float %i570), !dbg !43
  %local.reduce82 = fadd float %k0.sum.anchor1987, %i571, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1995 = call float asm sideeffect "", "=v,0"(float %local.reduce82) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i572 = insertelement <4 x float> %i569, float %i571, i64 3, !dbg !44
  %math.result372 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i572) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02214 = extractelement <4 x float> %math.result372, i32 0, !dbg !44
  %.cvtpack.f12215 = extractelement <4 x float> %math.result372, i32 1, !dbg !44
  %.cvtpack.f22216 = extractelement <4 x float> %math.result372, i32 2, !dbg !44
  %.cvtpack.f32217 = extractelement <4 x float> %math.result372, i32 3, !dbg !44
  %.cvtpack.p02218 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02214, float %.cvtpack.f12215), !dbg !44
  %.cvtpack.p12219 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22216, float %.cvtpack.f32217), !dbg !44
  %.cvtpack.v02220 = insertelement <2 x i32> poison, i32 %.cvtpack.p02218, i32 0, !dbg !44
  %.cvtpack.v12221 = insertelement <2 x i32> %.cvtpack.v02220, i32 %.cvtpack.p12219, i32 1, !dbg !44
  %i573 = bitcast <2 x i32> %.cvtpack.v12221 to <4 x half>, !dbg !44
  %math.packed373 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i573) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %i560, <4 x float> %i542) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i575 = shufflevector <8 x half> %i456, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %i575, <4 x float> %i544) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i577 = extractelement <4 x float> %lazy.score2164, i64 0, !dbg !43
  %i578 = call float @llvm.amdgcn.exp2.f32(float %i577), !dbg !43
  %local.reduce48 = fadd float %k0.sum.anchor1988, %i578, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1996 = call float asm sideeffect "", "=v,0"(float %local.reduce48) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i579 = insertelement <4 x float> poison, float %i578, i64 0, !dbg !44
  %i580 = extractelement <4 x float> %lazy.score2164, i64 1, !dbg !43
  %i581 = call float @llvm.amdgcn.exp2.f32(float %i580), !dbg !43
  %local.reduce49 = fadd float %k0.sum.anchor1989, %i581, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1997 = call float asm sideeffect "", "=v,0"(float %local.reduce49) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i582 = insertelement <4 x float> %i579, float %i581, i64 1, !dbg !44
  %i583 = extractelement <4 x float> %lazy.score2164, i64 2, !dbg !43
  %i584 = call float @llvm.amdgcn.exp2.f32(float %i583), !dbg !43
  %local.reduce50 = fadd float %k0.sum.anchor1990, %i584, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1998 = call float asm sideeffect "", "=v,0"(float %local.reduce50) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i585 = insertelement <4 x float> %i582, float %i584, i64 2, !dbg !44
  %i586 = extractelement <4 x float> %lazy.score2164, i64 3, !dbg !43
  %i587 = call float @llvm.amdgcn.exp2.f32(float %i586), !dbg !43
  %local.reduce51 = fadd float %k0.sum.anchor1991, %i587, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1999 = call float asm sideeffect "", "=v,0"(float %local.reduce51) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i588 = insertelement <4 x float> %i585, float %i587, i64 3, !dbg !44
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i588) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02222 = extractelement <4 x float> %math.result388, i32 0, !dbg !44
  %.cvtpack.f12223 = extractelement <4 x float> %math.result388, i32 1, !dbg !44
  %.cvtpack.f22224 = extractelement <4 x float> %math.result388, i32 2, !dbg !44
  %.cvtpack.f32225 = extractelement <4 x float> %math.result388, i32 3, !dbg !44
  %.cvtpack.p02226 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02222, float %.cvtpack.f12223), !dbg !44
  %.cvtpack.p12227 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22224, float %.cvtpack.f32225), !dbg !44
  %.cvtpack.v02228 = insertelement <2 x i32> poison, i32 %.cvtpack.p02226, i32 0, !dbg !44
  %.cvtpack.v12229 = insertelement <2 x i32> %.cvtpack.v02228, i32 %.cvtpack.p12227, i32 1, !dbg !44
  %i589 = bitcast <2 x i32> %.cvtpack.v12229 to <4 x half>, !dbg !44
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i589) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i560, <4 x float> %i558) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i591 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i575, <4 x float> %i559) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i592 = shufflevector <8 x half> %i458, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i593 = extractelement <4 x float> %lazy.score2157, i64 0, !dbg !43
  %i594 = call float @llvm.amdgcn.exp2.f32(float %i593), !dbg !43
  %local.reduce83 = fadd float %k0.sum.anchor1992, %i594, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2000 = call float asm sideeffect "", "=v,0"(float %local.reduce83) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i595 = insertelement <4 x float> poison, float %i594, i64 0, !dbg !44
  %i596 = extractelement <4 x float> %lazy.score2157, i64 1, !dbg !43
  %i597 = call float @llvm.amdgcn.exp2.f32(float %i596), !dbg !43
  %local.reduce84 = fadd float %k0.sum.anchor1993, %i597, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2001 = call float asm sideeffect "", "=v,0"(float %local.reduce84) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i598 = insertelement <4 x float> %i595, float %i597, i64 1, !dbg !44
  %i599 = extractelement <4 x float> %lazy.score2157, i64 2, !dbg !43
  %i600 = call float @llvm.amdgcn.exp2.f32(float %i599), !dbg !43
  %local.reduce85 = fadd float %k0.sum.anchor1994, %i600, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2002 = call float asm sideeffect "", "=v,0"(float %local.reduce85) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i601 = insertelement <4 x float> %i598, float %i600, i64 2, !dbg !44
  %i602 = extractelement <4 x float> %lazy.score2157, i64 3, !dbg !43
  %i603 = call float @llvm.amdgcn.exp2.f32(float %i602), !dbg !43
  %local.reduce86 = fadd float %k0.sum.anchor1995, %i603, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2003 = call float asm sideeffect "", "=v,0"(float %local.reduce86) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i604 = insertelement <4 x float> %i601, float %i603, i64 3, !dbg !44
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i604) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02230 = extractelement <4 x float> %math.result374, i32 0, !dbg !44
  %.cvtpack.f12231 = extractelement <4 x float> %math.result374, i32 1, !dbg !44
  %.cvtpack.f22232 = extractelement <4 x float> %math.result374, i32 2, !dbg !44
  %.cvtpack.f32233 = extractelement <4 x float> %math.result374, i32 3, !dbg !44
  %.cvtpack.p02234 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02230, float %.cvtpack.f12231), !dbg !44
  %.cvtpack.p12235 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22232, float %.cvtpack.f32233), !dbg !44
  %.cvtpack.v02236 = insertelement <2 x i32> poison, i32 %.cvtpack.p02234, i32 0, !dbg !44
  %.cvtpack.v12237 = insertelement <2 x i32> %.cvtpack.v02236, i32 %.cvtpack.p12235, i32 1, !dbg !44
  %i605 = bitcast <2 x i32> %.cvtpack.v12237 to <4 x half>, !dbg !44
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i605) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i606 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i592, <4 x float> %i574) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i607 = shufflevector <8 x half> %i458, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i607, <4 x float> %i576) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i609 = extractelement <4 x float> %lazy.score2165, i64 0, !dbg !43
  %i610 = call float @llvm.amdgcn.exp2.f32(float %i609), !dbg !43
  %local.reduce52 = fadd float %k0.sum.anchor1996, %i610, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2004 = call float asm sideeffect "", "=v,0"(float %local.reduce52) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i611 = insertelement <4 x float> poison, float %i610, i64 0, !dbg !44
  %i612 = extractelement <4 x float> %lazy.score2165, i64 1, !dbg !43
  %i613 = call float @llvm.amdgcn.exp2.f32(float %i612), !dbg !43
  %local.reduce53 = fadd float %k0.sum.anchor1997, %i613, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2005 = call float asm sideeffect "", "=v,0"(float %local.reduce53) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i614 = insertelement <4 x float> %i611, float %i613, i64 1, !dbg !44
  %i615 = extractelement <4 x float> %lazy.score2165, i64 2, !dbg !43
  %i616 = call float @llvm.amdgcn.exp2.f32(float %i615), !dbg !43
  %local.reduce54 = fadd float %k0.sum.anchor1998, %i616, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2006 = call float asm sideeffect "", "=v,0"(float %local.reduce54) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i617 = insertelement <4 x float> %i614, float %i616, i64 2, !dbg !44
  %i618 = extractelement <4 x float> %lazy.score2165, i64 3, !dbg !43
  %i619 = call float @llvm.amdgcn.exp2.f32(float %i618), !dbg !43
  %local.reduce55 = fadd float %k0.sum.anchor1999, %i619, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2007 = call float asm sideeffect "", "=v,0"(float %local.reduce55) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i620 = insertelement <4 x float> %i617, float %i619, i64 3, !dbg !44
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i620) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02238 = extractelement <4 x float> %math.result390, i32 0, !dbg !44
  %.cvtpack.f12239 = extractelement <4 x float> %math.result390, i32 1, !dbg !44
  %.cvtpack.f22240 = extractelement <4 x float> %math.result390, i32 2, !dbg !44
  %.cvtpack.f32241 = extractelement <4 x float> %math.result390, i32 3, !dbg !44
  %.cvtpack.p02242 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02238, float %.cvtpack.f12239), !dbg !44
  %.cvtpack.p12243 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22240, float %.cvtpack.f32241), !dbg !44
  %.cvtpack.v02244 = insertelement <2 x i32> poison, i32 %.cvtpack.p02242, i32 0, !dbg !44
  %.cvtpack.v12245 = insertelement <2 x i32> %.cvtpack.v02244, i32 %.cvtpack.p12243, i32 1, !dbg !44
  %i621 = bitcast <2 x i32> %.cvtpack.v12245 to <4 x half>, !dbg !44
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i621) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i592, <4 x float> %i590) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i623 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i607, <4 x float> %i591) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i624 = shufflevector <8 x half> %i460, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i625 = extractelement <4 x float> %lazy.score2158, i64 0, !dbg !43
  %i626 = call float @llvm.amdgcn.exp2.f32(float %i625), !dbg !43
  %local.reduce87 = fadd float %k0.sum.anchor2000, %i626, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2008 = call float asm sideeffect "", "=v,0"(float %local.reduce87) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i627 = insertelement <4 x float> poison, float %i626, i64 0, !dbg !44
  %i628 = extractelement <4 x float> %lazy.score2158, i64 1, !dbg !43
  %i629 = call float @llvm.amdgcn.exp2.f32(float %i628), !dbg !43
  %local.reduce88 = fadd float %k0.sum.anchor2001, %i629, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2009 = call float asm sideeffect "", "=v,0"(float %local.reduce88) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i630 = insertelement <4 x float> %i627, float %i629, i64 1, !dbg !44
  %i631 = extractelement <4 x float> %lazy.score2158, i64 2, !dbg !43
  %i632 = call float @llvm.amdgcn.exp2.f32(float %i631), !dbg !43
  %local.reduce89 = fadd float %k0.sum.anchor2002, %i632, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2010 = call float asm sideeffect "", "=v,0"(float %local.reduce89) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i633 = insertelement <4 x float> %i630, float %i632, i64 2, !dbg !44
  %i634 = extractelement <4 x float> %lazy.score2158, i64 3, !dbg !43
  %i635 = call float @llvm.amdgcn.exp2.f32(float %i634), !dbg !43
  %local.reduce90 = fadd float %k0.sum.anchor2003, %i635, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2011 = call float asm sideeffect "", "=v,0"(float %local.reduce90) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i636 = insertelement <4 x float> %i633, float %i635, i64 3, !dbg !44
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i636) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02246 = extractelement <4 x float> %math.result376, i32 0, !dbg !44
  %.cvtpack.f12247 = extractelement <4 x float> %math.result376, i32 1, !dbg !44
  %.cvtpack.f22248 = extractelement <4 x float> %math.result376, i32 2, !dbg !44
  %.cvtpack.f32249 = extractelement <4 x float> %math.result376, i32 3, !dbg !44
  %.cvtpack.p02250 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02246, float %.cvtpack.f12247), !dbg !44
  %.cvtpack.p12251 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22248, float %.cvtpack.f32249), !dbg !44
  %.cvtpack.v02252 = insertelement <2 x i32> poison, i32 %.cvtpack.p02250, i32 0, !dbg !44
  %.cvtpack.v12253 = insertelement <2 x i32> %.cvtpack.v02252, i32 %.cvtpack.p12251, i32 1, !dbg !44
  %i637 = bitcast <2 x i32> %.cvtpack.v12253 to <4 x half>, !dbg !44
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i637) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i624, <4 x float> %i606) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i639 = shufflevector <8 x half> %i460, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i639, <4 x float> %i608) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i641 = extractelement <4 x float> %lazy.score2166, i64 0, !dbg !43
  %i642 = call float @llvm.amdgcn.exp2.f32(float %i641), !dbg !43
  %local.reduce56 = fadd float %k0.sum.anchor2004, %i642, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2012 = call float asm sideeffect "", "=v,0"(float %local.reduce56) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i643 = insertelement <4 x float> poison, float %i642, i64 0, !dbg !44
  %i644 = extractelement <4 x float> %lazy.score2166, i64 1, !dbg !43
  %i645 = call float @llvm.amdgcn.exp2.f32(float %i644), !dbg !43
  %local.reduce57 = fadd float %k0.sum.anchor2005, %i645, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2013 = call float asm sideeffect "", "=v,0"(float %local.reduce57) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i646 = insertelement <4 x float> %i643, float %i645, i64 1, !dbg !44
  %i647 = extractelement <4 x float> %lazy.score2166, i64 2, !dbg !43
  %i648 = call float @llvm.amdgcn.exp2.f32(float %i647), !dbg !43
  %local.reduce58 = fadd float %k0.sum.anchor2006, %i648, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2014 = call float asm sideeffect "", "=v,0"(float %local.reduce58) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i649 = insertelement <4 x float> %i646, float %i648, i64 2, !dbg !44
  %i650 = extractelement <4 x float> %lazy.score2166, i64 3, !dbg !43
  %i651 = call float @llvm.amdgcn.exp2.f32(float %i650), !dbg !43
  %local.reduce59 = fadd float %k0.sum.anchor2007, %i651, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2015 = call float asm sideeffect "", "=v,0"(float %local.reduce59) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i652 = insertelement <4 x float> %i649, float %i651, i64 3, !dbg !44
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i652) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02254 = extractelement <4 x float> %math.result392, i32 0, !dbg !44
  %.cvtpack.f12255 = extractelement <4 x float> %math.result392, i32 1, !dbg !44
  %.cvtpack.f22256 = extractelement <4 x float> %math.result392, i32 2, !dbg !44
  %.cvtpack.f32257 = extractelement <4 x float> %math.result392, i32 3, !dbg !44
  %.cvtpack.p02258 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02254, float %.cvtpack.f12255), !dbg !44
  %.cvtpack.p12259 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22256, float %.cvtpack.f32257), !dbg !44
  %.cvtpack.v02260 = insertelement <2 x i32> poison, i32 %.cvtpack.p02258, i32 0, !dbg !44
  %.cvtpack.v12261 = insertelement <2 x i32> %.cvtpack.v02260, i32 %.cvtpack.p12259, i32 1, !dbg !44
  %i653 = bitcast <2 x i32> %.cvtpack.v12261 to <4 x half>, !dbg !44
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i653) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i624, <4 x float> %i622) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i639, <4 x float> %i623) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i656 = shufflevector <8 x half> %i462, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i657 = extractelement <4 x float> %lazy.score2159, i64 0, !dbg !43
  %i658 = call float @llvm.amdgcn.exp2.f32(float %i657), !dbg !43
  %local.reduce91 = fadd float %k0.sum.anchor2008, %i658, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2016 = call float asm sideeffect "", "=v,0"(float %local.reduce91) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i659 = insertelement <4 x float> poison, float %i658, i64 0, !dbg !44
  %i660 = extractelement <4 x float> %lazy.score2159, i64 1, !dbg !43
  %i661 = call float @llvm.amdgcn.exp2.f32(float %i660), !dbg !43
  %local.reduce92 = fadd float %k0.sum.anchor2009, %i661, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2017 = call float asm sideeffect "", "=v,0"(float %local.reduce92) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i662 = insertelement <4 x float> %i659, float %i661, i64 1, !dbg !44
  %i663 = extractelement <4 x float> %lazy.score2159, i64 2, !dbg !43
  %i664 = call float @llvm.amdgcn.exp2.f32(float %i663), !dbg !43
  %local.reduce93 = fadd float %k0.sum.anchor2010, %i664, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2018 = call float asm sideeffect "", "=v,0"(float %local.reduce93) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i665 = insertelement <4 x float> %i662, float %i664, i64 2, !dbg !44
  %i666 = extractelement <4 x float> %lazy.score2159, i64 3, !dbg !43
  %i667 = call float @llvm.amdgcn.exp2.f32(float %i666), !dbg !43
  %local.reduce94 = fadd float %k0.sum.anchor2011, %i667, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2019 = call float asm sideeffect "", "=v,0"(float %local.reduce94) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i668 = insertelement <4 x float> %i665, float %i667, i64 3, !dbg !44
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i668) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02262 = extractelement <4 x float> %math.result378, i32 0, !dbg !44
  %.cvtpack.f12263 = extractelement <4 x float> %math.result378, i32 1, !dbg !44
  %.cvtpack.f22264 = extractelement <4 x float> %math.result378, i32 2, !dbg !44
  %.cvtpack.f32265 = extractelement <4 x float> %math.result378, i32 3, !dbg !44
  %.cvtpack.p02266 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02262, float %.cvtpack.f12263), !dbg !44
  %.cvtpack.p12267 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22264, float %.cvtpack.f32265), !dbg !44
  %.cvtpack.v02268 = insertelement <2 x i32> poison, i32 %.cvtpack.p02266, i32 0, !dbg !44
  %.cvtpack.v12269 = insertelement <2 x i32> %.cvtpack.v02268, i32 %.cvtpack.p12267, i32 1, !dbg !44
  %i669 = bitcast <2 x i32> %.cvtpack.v12269 to <4 x half>, !dbg !44
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i669) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i656, <4 x float> %i638) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i671 = shufflevector <8 x half> %i462, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i671, <4 x float> %i640) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i673 = extractelement <4 x float> %lazy.score2167, i64 0, !dbg !43
  %i674 = call float @llvm.amdgcn.exp2.f32(float %i673), !dbg !43
  %local.reduce60 = fadd float %k0.sum.anchor2012, %i674, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2020 = call float asm sideeffect "", "=v,0"(float %local.reduce60) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i675 = insertelement <4 x float> poison, float %i674, i64 0, !dbg !44
  %i676 = extractelement <4 x float> %lazy.score2167, i64 1, !dbg !43
  %i677 = call float @llvm.amdgcn.exp2.f32(float %i676), !dbg !43
  %local.reduce61 = fadd float %k0.sum.anchor2013, %i677, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2021 = call float asm sideeffect "", "=v,0"(float %local.reduce61) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i678 = insertelement <4 x float> %i675, float %i677, i64 1, !dbg !44
  %i679 = extractelement <4 x float> %lazy.score2167, i64 2, !dbg !43
  %i680 = call float @llvm.amdgcn.exp2.f32(float %i679), !dbg !43
  %local.reduce62 = fadd float %k0.sum.anchor2014, %i680, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2022 = call float asm sideeffect "", "=v,0"(float %local.reduce62) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i681 = insertelement <4 x float> %i678, float %i680, i64 2, !dbg !44
  %i682 = extractelement <4 x float> %lazy.score2167, i64 3, !dbg !43
  %i683 = call float @llvm.amdgcn.exp2.f32(float %i682), !dbg !43
  %local.reduce63 = fadd float %k0.sum.anchor2015, %i683, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2023 = call float asm sideeffect "", "=v,0"(float %local.reduce63) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i684 = insertelement <4 x float> %i681, float %i683, i64 3, !dbg !44
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i684) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02270 = extractelement <4 x float> %math.result394, i32 0, !dbg !44
  %.cvtpack.f12271 = extractelement <4 x float> %math.result394, i32 1, !dbg !44
  %.cvtpack.f22272 = extractelement <4 x float> %math.result394, i32 2, !dbg !44
  %.cvtpack.f32273 = extractelement <4 x float> %math.result394, i32 3, !dbg !44
  %.cvtpack.p02274 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02270, float %.cvtpack.f12271), !dbg !44
  %.cvtpack.p12275 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22272, float %.cvtpack.f32273), !dbg !44
  %.cvtpack.v02276 = insertelement <2 x i32> poison, i32 %.cvtpack.p02274, i32 0, !dbg !44
  %.cvtpack.v12277 = insertelement <2 x i32> %.cvtpack.v02276, i32 %.cvtpack.p12275, i32 1, !dbg !44
  %i685 = bitcast <2 x i32> %.cvtpack.v12277 to <4 x half>, !dbg !44
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i685) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i656, <4 x float> %i654) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i671, <4 x float> %i655) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i688 = bitcast <4 x i32> %i463 to <8 x half>, !dbg !41
  %i689 = shufflevector <8 x half> %i688, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i690 = extractelement <4 x float> %lazy.score2160, i64 0, !dbg !43
  %i691 = call float @llvm.amdgcn.exp2.f32(float %i690), !dbg !43
  %local.reduce95 = fadd float %k0.sum.anchor2016, %i691, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2024 = call float asm sideeffect "", "=v,0"(float %local.reduce95) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i692 = insertelement <4 x float> poison, float %i691, i64 0, !dbg !44
  %i693 = extractelement <4 x float> %lazy.score2160, i64 1, !dbg !43
  %i694 = call float @llvm.amdgcn.exp2.f32(float %i693), !dbg !43
  %local.reduce96 = fadd float %k0.sum.anchor2017, %i694, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2025 = call float asm sideeffect "", "=v,0"(float %local.reduce96) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce99 = fadd float %k0.sum.anchor2024, %k0.sum.anchor2025, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2026 = call float asm sideeffect "", "=v,0"(float %local.reduce99) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i695 = insertelement <4 x float> %i692, float %i694, i64 1, !dbg !44
  %i696 = extractelement <4 x float> %lazy.score2160, i64 2, !dbg !43
  %i697 = call float @llvm.amdgcn.exp2.f32(float %i696), !dbg !43
  %local.reduce97 = fadd float %k0.sum.anchor2018, %i697, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2027 = call float asm sideeffect "", "=v,0"(float %local.reduce97) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i698 = insertelement <4 x float> %i695, float %i697, i64 2, !dbg !44
  %i699 = extractelement <4 x float> %lazy.score2160, i64 3, !dbg !43
  %i700 = call float @llvm.amdgcn.exp2.f32(float %i699), !dbg !43
  %local.reduce98 = fadd float %k0.sum.anchor2019, %i700, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2028 = call float asm sideeffect "", "=v,0"(float %local.reduce98) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce100 = fadd float %k0.sum.anchor2027, %k0.sum.anchor2028, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2029 = call float asm sideeffect "", "=v,0"(float %local.reduce100) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i701 = fadd float %k0.sum.anchor2026, %k0.sum.anchor2029, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2030 = call float asm sideeffect "", "=v,0"(float %i701) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i702 = fadd float %k0.sum.anchor2030, %i438, !dbg !45
  %k0.sum.anchor2031 = call float asm sideeffect "", "=v,0"(float %i702) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i703 = insertelement <1 x float> poison, float %k0.sum.anchor2031, i64 0, !dbg !12
  %i704 = insertvalue [2 x <1 x float>] poison, <1 x float> %i703, 0, !dbg !12
  %i705 = insertelement <4 x float> %i698, float %i700, i64 3, !dbg !44
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i705) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02278 = extractelement <4 x float> %math.result380, i32 0, !dbg !44
  %.cvtpack.f12279 = extractelement <4 x float> %math.result380, i32 1, !dbg !44
  %.cvtpack.f22280 = extractelement <4 x float> %math.result380, i32 2, !dbg !44
  %.cvtpack.f32281 = extractelement <4 x float> %math.result380, i32 3, !dbg !44
  %.cvtpack.p02282 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02278, float %.cvtpack.f12279), !dbg !44
  %.cvtpack.p12283 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22280, float %.cvtpack.f32281), !dbg !44
  %.cvtpack.v02284 = insertelement <2 x i32> poison, i32 %.cvtpack.p02282, i32 0, !dbg !44
  %.cvtpack.v12285 = insertelement <2 x i32> %.cvtpack.v02284, i32 %.cvtpack.p12283, i32 1, !dbg !44
  %i706 = bitcast <2 x i32> %.cvtpack.v12285 to <4 x half>, !dbg !44
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i706) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i707 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i689, <4 x float> %i670) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i708 = shufflevector <8 x half> %i688, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i708, <4 x float> %i672) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i710 = extractelement <4 x float> %lazy.score2168, i64 0, !dbg !43
  %i711 = call float @llvm.amdgcn.exp2.f32(float %i710), !dbg !43
  %local.reduce64 = fadd float %k0.sum.anchor2020, %i711, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2032 = call float asm sideeffect "", "=v,0"(float %local.reduce64) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i712 = insertelement <4 x float> poison, float %i711, i64 0, !dbg !44
  %i713 = extractelement <4 x float> %lazy.score2168, i64 1, !dbg !43
  %i714 = call float @llvm.amdgcn.exp2.f32(float %i713), !dbg !43
  %local.reduce65 = fadd float %k0.sum.anchor2021, %i714, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2033 = call float asm sideeffect "", "=v,0"(float %local.reduce65) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce68 = fadd float %k0.sum.anchor2032, %k0.sum.anchor2033, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2034 = call float asm sideeffect "", "=v,0"(float %local.reduce68) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i715 = insertelement <4 x float> %i712, float %i714, i64 1, !dbg !44
  %i716 = extractelement <4 x float> %lazy.score2168, i64 2, !dbg !43
  %i717 = call float @llvm.amdgcn.exp2.f32(float %i716), !dbg !43
  %local.reduce66 = fadd float %k0.sum.anchor2022, %i717, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2035 = call float asm sideeffect "", "=v,0"(float %local.reduce66) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i718 = insertelement <4 x float> %i715, float %i717, i64 2, !dbg !44
  %i719 = extractelement <4 x float> %lazy.score2168, i64 3, !dbg !43
  %i720 = call float @llvm.amdgcn.exp2.f32(float %i719), !dbg !43
  %local.reduce67 = fadd float %k0.sum.anchor2023, %i720, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2036 = call float asm sideeffect "", "=v,0"(float %local.reduce67) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce69 = fadd float %k0.sum.anchor2035, %k0.sum.anchor2036, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2037 = call float asm sideeffect "", "=v,0"(float %local.reduce69) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i721 = fadd float %k0.sum.anchor2034, %k0.sum.anchor2037, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2038 = call float asm sideeffect "", "=v,0"(float %i721) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i722 = fadd float %k0.sum.anchor2038, %i440, !dbg !45
  %k0.sum.anchor2039 = call float asm sideeffect "", "=v,0"(float %i722) #9, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %i723 = insertelement <1 x float> poison, float %k0.sum.anchor2039, i64 0, !dbg !12
  %i724 = insertvalue [2 x <1 x float>] %i704, <1 x float> %i723, 1, !dbg !12
  %i725 = insertelement <4 x float> %i718, float %i720, i64 3, !dbg !44
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i725) #9, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02286 = extractelement <4 x float> %math.result396, i32 0, !dbg !44
  %.cvtpack.f12287 = extractelement <4 x float> %math.result396, i32 1, !dbg !44
  %.cvtpack.f22288 = extractelement <4 x float> %math.result396, i32 2, !dbg !44
  %.cvtpack.f32289 = extractelement <4 x float> %math.result396, i32 3, !dbg !44
  %.cvtpack.p02290 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02286, float %.cvtpack.f12287), !dbg !44
  %.cvtpack.p12291 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22288, float %.cvtpack.f32289), !dbg !44
  %.cvtpack.v02292 = insertelement <2 x i32> poison, i32 %.cvtpack.p02290, i32 0, !dbg !44
  %.cvtpack.v12293 = insertelement <2 x i32> %.cvtpack.v02292, i32 %.cvtpack.p12291, i32 1, !dbg !44
  %i726 = bitcast <2 x i32> %.cvtpack.v12293 to <4 x half>, !dbg !44
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i726) #9, !dbg !45, !frisk.register_lifetimes !5
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41, !frisk.softmax_fragments !5
  %i727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i689, <4 x float> %i686) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i728 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i708, <4 x float> %i687) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40
  %i729 = extractvalue [2 x <8 x float>] %lazy.state2172, 0, !dbg !40
  %i730 = shufflevector <8 x float> %i729, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !40
  %i731 = shufflevector <8 x float> %i729, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %i732 = extractvalue [2 x <8 x float>] %lazy.state2172, 1, !dbg !40
  %i733 = shufflevector <8 x float> %i732, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !40
  %i734 = shufflevector <8 x float> %i732, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !40
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1441 = add nsw i64 %i193, %address.local916, !dbg !38
  %vdb.address1442 = trunc i64 %vdb.address1441 to i32, !dbg !38
  %vdb.address1443 = shl i32 %vdb.address1442, 1, !dbg !38
  %vdb.address1448 = trunc nuw i64 %i193 to i32, !dbg !38
  %vdb.address1449 = or disjoint i32 %address.local923, %vdb.address1448, !dbg !38
  %vdb.address1450 = shl nuw nsw i32 %vdb.address1449, 1, !dbg !38
  %i735 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i36, i32 %vdb.address1443, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !38
  %i736 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i44, i32 %vdb.address1450, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !38
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local932, i64 8192, !dbg !41
  %i737 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i738 = bitcast <4 x i32> %i737 to <8 x half>, !dbg !41
  %vdb.read1434 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8448, !dbg !41
  %i739 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1434, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i740 = bitcast <4 x i32> %i739 to <8 x half>, !dbg !41
  %vdb.read1435 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8704, !dbg !41
  %i741 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1435, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i742 = bitcast <4 x i32> %i741 to <8 x half>, !dbg !41
  %vdb.read1436 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8960, !dbg !41
  %i743 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1436, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i744 = bitcast <4 x i32> %i743 to <8 x half>, !dbg !41
  %vdb.read1437 = getelementptr i8, ptr addrspace(3) %address.local932, i64 12288, !dbg !41
  %i745 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1437, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i746 = bitcast <4 x i32> %i745 to <8 x half>, !dbg !41
  %vdb.read1438 = getelementptr i8, ptr addrspace(3) %address.local941, i64 12544, !dbg !41
  %i747 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1438, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i748 = bitcast <4 x i32> %i747 to <8 x half>, !dbg !41
  %vdb.read1439 = getelementptr i8, ptr addrspace(3) %address.local941, i64 12800, !dbg !41
  %i749 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1439, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i750 = bitcast <4 x i32> %i749 to <8 x half>, !dbg !41
  %vdb.read1440 = getelementptr i8, ptr addrspace(3) %address.local941, i64 13056, !dbg !41
  %i751 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1440, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i752 = shufflevector <8 x half> %i738, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i753 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i752, <4 x float> %i730) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i754 = shufflevector <8 x half> %i738, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i754, <4 x float> %i731) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i752, <4 x float> %i733) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i754, <4 x float> %i734) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i758 = shufflevector <8 x half> %i740, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %i758, <4 x float> %i753) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i760 = shufflevector <8 x half> %i740, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i761 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %i760, <4 x float> %i755) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i762 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i758, <4 x float> %i756) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i763 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i760, <4 x float> %i757) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i764 = shufflevector <8 x half> %i742, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i765 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %i764, <4 x float> %i759) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i766 = shufflevector <8 x half> %i742, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i767 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %i766, <4 x float> %i761) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i768 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i764, <4 x float> %i762) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i769 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i766, <4 x float> %i763) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i770 = shufflevector <8 x half> %i744, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %i770, <4 x float> %i765) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i772 = shufflevector <8 x half> %i744, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %i772, <4 x float> %i767) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i774 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i770, <4 x float> %i768) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i775 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i772, <4 x float> %i769) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i776 = shufflevector <8 x half> %i746, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i777 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i776, <4 x float> %i771) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i778 = shufflevector <8 x half> %i746, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i779 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i778, <4 x float> %i773) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i780 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i776, <4 x float> %i774) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i781 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i778, <4 x float> %i775) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i782 = shufflevector <8 x half> %i748, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i783 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i782, <4 x float> %i777) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i784 = shufflevector <8 x half> %i748, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i785 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i784, <4 x float> %i779) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i786 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i782, <4 x float> %i780) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i787 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i784, <4 x float> %i781) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i788 = shufflevector <8 x half> %i750, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i789 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i788, <4 x float> %i783) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i790 = shufflevector <8 x half> %i750, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i791 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i790, <4 x float> %i785) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i792 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i788, <4 x float> %i786) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i793 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i790, <4 x float> %i787) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i794 = bitcast <4 x i32> %i751 to <8 x half>, !dbg !41
  %i795 = shufflevector <8 x half> %i794, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i796 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i795, <4 x float> %i789) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i797 = shufflevector <8 x half> %i794, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i798 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i797, <4 x float> %i791) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i799 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i795, <4 x float> %i792) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i800 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i797, <4 x float> %i793) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40
  %i801 = extractvalue [2 x <8 x float>] %lazy.state2171, 0, !dbg !40
  %i802 = shufflevector <8 x float> %i801, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !40
  %i803 = shufflevector <8 x float> %i801, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %i804 = extractvalue [2 x <8 x float>] %lazy.state2171, 1, !dbg !40
  %i805 = shufflevector <8 x float> %i804, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !40
  %i806 = shufflevector <8 x float> %i804, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !40
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1455 = add nsw i64 %i246, %address.local916, !dbg !38
  %vdb.address1456 = trunc i64 %vdb.address1455 to i32, !dbg !38
  %vdb.address1457 = shl i32 %vdb.address1456, 1, !dbg !38
  %vdb.address1462 = trunc nuw i64 %i246 to i32, !dbg !38
  %vdb.address1463 = or disjoint i32 %address.local923, %vdb.address1462, !dbg !38
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !38
  %i807 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1457, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !38
  %i808 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1433, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !38
  %i809 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local932, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i810 = bitcast <4 x i32> %i809 to <8 x half>, !dbg !41
  %i811 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i812 = bitcast <4 x i32> %i811 to <8 x half>, !dbg !41
  %i813 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i814 = bitcast <4 x i32> %i813 to <8 x half>, !dbg !41
  %i815 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local950, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i816 = bitcast <4 x i32> %i815 to <8 x half>, !dbg !41
  %i817 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i818 = bitcast <4 x i32> %i817 to <8 x half>, !dbg !41
  %i819 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i820 = bitcast <4 x i32> %i819 to <8 x half>, !dbg !41
  %i821 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local960, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i822 = bitcast <4 x i32> %i821 to <8 x half>, !dbg !41
  %i823 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i824 = shufflevector <8 x half> %i810, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i824, <4 x float> %i802) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i826 = shufflevector <8 x half> %i810, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i826, <4 x float> %i803) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i824, <4 x float> %i805) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i826, <4 x float> %i806) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i830 = shufflevector <8 x half> %i812, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %i830, <4 x float> %i825) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i832 = shufflevector <8 x half> %i812, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %i832, <4 x float> %i827) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i834 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i830, <4 x float> %i828) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i832, <4 x float> %i829) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i836 = shufflevector <8 x half> %i814, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %i836, <4 x float> %i831) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i838 = shufflevector <8 x half> %i814, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %i838, <4 x float> %i833) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i836, <4 x float> %i834) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i838, <4 x float> %i835) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i842 = shufflevector <8 x half> %i816, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %i842, <4 x float> %i837) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i844 = shufflevector <8 x half> %i816, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %i844, <4 x float> %i839) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i842, <4 x float> %i840) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i844, <4 x float> %i841) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i848 = shufflevector <8 x half> %i818, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i848, <4 x float> %i843) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i850 = shufflevector <8 x half> %i818, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i850, <4 x float> %i845) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i848, <4 x float> %i846) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i850, <4 x float> %i847) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i854 = shufflevector <8 x half> %i820, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i854, <4 x float> %i849) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i856 = shufflevector <8 x half> %i820, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i856, <4 x float> %i851) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i854, <4 x float> %i852) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i859 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i856, <4 x float> %i853) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i860 = shufflevector <8 x half> %i822, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i860, <4 x float> %i855) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i862 = shufflevector <8 x half> %i822, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i863 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i862, <4 x float> %i857) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i860, <4 x float> %i858) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i862, <4 x float> %i859) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i866 = bitcast <4 x i32> %i823 to <8 x half>, !dbg !41
  %i867 = shufflevector <8 x half> %i866, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i867, <4 x float> %i861) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i869 = shufflevector <8 x half> %i866, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i869, <4 x float> %i863) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i871 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i867, <4 x float> %i864) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i869, <4 x float> %i865) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40
  %i873 = extractvalue [2 x <8 x float>] %lazy.state2170, 0, !dbg !40
  %i874 = shufflevector <8 x float> %i873, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !40
  %i875 = shufflevector <8 x float> %i873, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %i876 = extractvalue [2 x <8 x float>] %lazy.state2170, 1, !dbg !40
  %i877 = shufflevector <8 x float> %i876, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !40
  %i878 = shufflevector <8 x float> %i876, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !40
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %i879 = add nuw nsw i64 %i99, 1, !dbg !12
  %k0.has.next = icmp ult i64 %i879, 32, !dbg !41
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !41

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %i879, 14, !dbg !14
  %k0.next.2050 = or disjoint i64 %i109, %k0.next., !dbg !14
  %k0.next.kdma.source1495 = or disjoint i64 %k0.next.2050, %kdma.source1494, !dbg !14
  %k0.next.kdma.source1496 = trunc nuw i64 %k0.next.kdma.source1495 to i32, !dbg !14
  %k0.next.kdma.source1497 = shl nuw nsw i32 %k0.next.kdma.source1496, 1, !dbg !14
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1497, %kdma.intra, !dbg !14
  %k0.next.2051 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i36, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1515 = or disjoint i64 %k0.next.2050, %kdma.source1514, !dbg !14
  %k0.next.kdma.source1516 = trunc nuw i64 %k0.next.kdma.source1515 to i32, !dbg !14
  %k0.next.kdma.source1517 = shl nuw nsw i32 %k0.next.kdma.source1516, 1, !dbg !14
  %k0.next.kdma.offset1518 = or disjoint i32 %k0.next.kdma.source1517, %kdma.intra.hi, !dbg !14
  %k0.next.2052 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1519, i32 %k0.next.kdma.offset1518, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1539 = or disjoint i64 %k0.next.2050, %kdma.source1538, !dbg !14
  %k0.next.kdma.source1540 = trunc nuw i64 %k0.next.kdma.source1539 to i32, !dbg !14
  %k0.next.kdma.source1541 = shl nuw nsw i32 %k0.next.kdma.source1540, 1, !dbg !14
  %k0.next.kdma.offset1542 = add nuw i32 %k0.next.kdma.source1541, %kdma.intra, !dbg !14
  %k0.next.2053 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %i44, i32 %k0.next.kdma.offset1542, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1563 = or disjoint i64 %k0.next.2050, %kdma.source1562, !dbg !14
  %k0.next.kdma.source1564 = trunc nuw i64 %k0.next.kdma.source1563 to i32, !dbg !14
  %k0.next.kdma.source1565 = shl nuw nsw i32 %k0.next.kdma.source1564, 1, !dbg !14
  %k0.next.kdma.offset1566 = add nuw i32 %k0.next.kdma.source1565, %kdma.intra.hi, !dbg !14
  %k0.next.2054 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1567, i32 %k0.next.kdma.offset1566, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !14, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %lazy.join
  %i880 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i881 = bitcast <4 x i32> %i880 to <8 x half>, !dbg !41
  %i882 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1434, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i883 = bitcast <4 x i32> %i882 to <8 x half>, !dbg !41
  %i884 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1435, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i885 = bitcast <4 x i32> %i884 to <8 x half>, !dbg !41
  %i886 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1436, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i887 = bitcast <4 x i32> %i886 to <8 x half>, !dbg !41
  %i888 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1437, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i889 = bitcast <4 x i32> %i888 to <8 x half>, !dbg !41
  %i890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1438, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i891 = bitcast <4 x i32> %i890 to <8 x half>, !dbg !41
  %i892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1439, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i893 = bitcast <4 x i32> %i892 to <8 x half>, !dbg !41
  %i894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1440, i32 0) #0, !dbg !41, !frisk.v_double_buffer !42
  %i895 = shufflevector <8 x half> %i881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i895, <4 x float> %i874) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i897 = shufflevector <8 x half> %i881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i897, <4 x float> %i875) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i895, <4 x float> %i877) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i897, <4 x float> %i878) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i901 = shufflevector <8 x half> %i883, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %i901, <4 x float> %i896) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i903 = shufflevector <8 x half> %i883, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %i903, <4 x float> %i898) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i901, <4 x float> %i899) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i903, <4 x float> %i900) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i907 = shufflevector <8 x half> %i885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %i907, <4 x float> %i902) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i909 = shufflevector <8 x half> %i885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %i909, <4 x float> %i904) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i907, <4 x float> %i905) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i909, <4 x float> %i906) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i913 = shufflevector <8 x half> %i887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %i913, <4 x float> %i908) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i915 = shufflevector <8 x half> %i887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %i915, <4 x float> %i910) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i913, <4 x float> %i911) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i915, <4 x float> %i912) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i919 = shufflevector <8 x half> %i889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i919, <4 x float> %i914) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i921 = shufflevector <8 x half> %i889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i921, <4 x float> %i916) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i919, <4 x float> %i917) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i921, <4 x float> %i918) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i925 = shufflevector <8 x half> %i891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i925, <4 x float> %i920) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i927 = shufflevector <8 x half> %i891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i927, <4 x float> %i922) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i925, <4 x float> %i923) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i927, <4 x float> %i924) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i931 = shufflevector <8 x half> %i893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i931, <4 x float> %i926) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i933 = shufflevector <8 x half> %i893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i933, <4 x float> %i928) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i935 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i931, <4 x float> %i929) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i933, <4 x float> %i930) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i937 = bitcast <4 x i32> %i894 to <8 x half>, !dbg !41
  %i938 = shufflevector <8 x half> %i937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i938, <4 x float> %i932) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i940 = shufflevector <8 x half> %i937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i940, <4 x float> %i934) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i938, <4 x float> %i935) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  %i943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i940, <4 x float> %i936) #9, !dbg !41, !frisk.pv_interleave !19
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !12
  %i944 = shufflevector <4 x float> %i707, <4 x float> %i709, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i945 = insertvalue [2 x <8 x float>] poison, <8 x float> %i944, 0, !dbg !12
  %i946 = shufflevector <4 x float> %i727, <4 x float> %i728, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i947 = insertvalue [2 x <8 x float>] %i945, <8 x float> %i946, 1, !dbg !12
  %i948 = shufflevector <4 x float> %i796, <4 x float> %i798, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i949 = insertvalue [2 x <8 x float>] poison, <8 x float> %i948, 0, !dbg !12
  %i950 = shufflevector <4 x float> %i799, <4 x float> %i800, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i951 = insertvalue [2 x <8 x float>] %i949, <8 x float> %i950, 1, !dbg !12
  %i952 = shufflevector <4 x float> %i868, <4 x float> %i870, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i953 = insertvalue [2 x <8 x float>] poison, <8 x float> %i952, 0, !dbg !12
  %i954 = shufflevector <4 x float> %i871, <4 x float> %i872, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i955 = insertvalue [2 x <8 x float>] %i953, <8 x float> %i954, 1, !dbg !12
  %i956 = shufflevector <4 x float> %i939, <4 x float> %i941, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i957 = insertvalue [2 x <8 x float>] poison, <8 x float> %i956, 0, !dbg !12
  %i958 = shufflevector <4 x float> %i942, <4 x float> %i943, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %i959 = insertvalue [2 x <8 x float>] %i957, <8 x float> %i958, 1, !dbg !12
  br label %bb98, !dbg !12

bb960:                                            ; preds = %bb98
  %lane.sum.field166 = extractvalue [2 x <1 x float>] %i101, 1, !dbg !12
  %lane.sum.local167 = extractelement <1 x float> %lane.sum.field166, i64 0, !dbg !12
  %i961 = shl nuw nsw i32 %i6, 2, !dbg !12
  %i962 = xor i32 %i961, 64, !dbg !12
  %i963 = bitcast float %lane.sum.local167 to i32, !dbg !12
  %lane.sum.shuffle170 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i962, i32 %i963), !dbg !12
  %i964 = bitcast i32 %lane.sum.shuffle170 to float, !dbg !12
  %lane.sum.final171 = fadd float %lane.sum.local167, %i964, !dbg !12
  %i965 = xor i32 %i961, 128, !dbg !12
  %i966 = bitcast float %lane.sum.final171 to i32, !dbg !12
  %lane.sum.shuffle172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i965, i32 %i966), !dbg !12
  %i967 = bitcast i32 %lane.sum.shuffle172 to float, !dbg !12
  %lane.sum.final173 = fadd float %lane.sum.final171, %i967, !dbg !12
  %i968 = insertelement <1 x float> poison, float %lane.sum.final173, i64 0, !dbg !12
  %lane.sum.field = extractvalue [2 x <1 x float>] %i101, 0, !dbg !12
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !12
  %i969 = bitcast float %lane.sum.local to i32, !dbg !12
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %i962, i32 %i969), !dbg !12
  %i970 = bitcast i32 %lane.sum.shuffle to float, !dbg !12
  %lane.sum.final = fadd float %lane.sum.local, %i970, !dbg !12
  %i971 = bitcast float %lane.sum.final to i32, !dbg !12
  %lane.sum.shuffle164 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i965, i32 %i971), !dbg !12
  %i972 = bitcast i32 %lane.sum.shuffle164 to float, !dbg !12
  %lane.sum.final165 = fadd float %lane.sum.final, %i972, !dbg !12
  %i973 = insertelement <1 x float> poison, float %lane.sum.final165, i64 0, !dbg !12
  %i974 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %i974, 0, !dbg !12
  %i975 = and i32 %kdma.tid, 32, !dbg !12
  %.not40 = icmp eq i32 %i975, 0, !dbg !12
  %i976 = extractvalue [2 x <8 x float>] %i105, 0, !dbg !12
  %i977 = extractelement <8 x float> %i976, i64 0, !dbg !12
  %i978 = extractelement <8 x float> %i976, i64 1, !dbg !12
  %i979 = extractelement <8 x float> %i976, i64 2, !dbg !12
  %i980 = extractelement <8 x float> %i976, i64 3, !dbg !12
  %i981 = select i1 %.not, float %i978, float %i977, !dbg !12
  %i982 = bitcast float %i981 to i32, !dbg !12
  %i983 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i982), !dbg !12
  %i984 = bitcast i32 %i983 to float, !dbg !12
  %i985 = select i1 %.not, float %i977, float %i984, !dbg !12
  %i986 = select i1 %.not, float %i984, float %i978, !dbg !12
  %i987 = select i1 %.not, float %i980, float %i979, !dbg !12
  %i988 = bitcast float %i987 to i32, !dbg !12
  %i989 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i988), !dbg !12
  %i990 = bitcast i32 %i989 to float, !dbg !12
  %i991 = select i1 %.not, float %i979, float %i990, !dbg !12
  %i992 = select i1 %.not, float %i990, float %i980, !dbg !12
  %i993 = select i1 %.not40, float %i991, float %i985, !dbg !12
  %i994 = bitcast float %i993 to i32, !dbg !12
  %i995 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i994), !dbg !12
  %i996 = bitcast i32 %i995 to float, !dbg !12
  %i997 = select i1 %.not40, float %i985, float %i996, !dbg !12
  %i998 = select i1 %.not40, float %i996, float %i991, !dbg !12
  %i999 = select i1 %.not40, float %i992, float %i986, !dbg !12
  %i1000 = bitcast float %i999 to i32, !dbg !12
  %i1001 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1000), !dbg !12
  %i1002 = bitcast i32 %i1001 to float, !dbg !12
  %i1003 = select i1 %.not40, float %i986, float %i1002, !dbg !12
  %i1004 = select i1 %.not40, float %i1002, float %i992, !dbg !12
  %i1005 = extractelement <8 x float> %i976, i64 4, !dbg !12
  %i1006 = extractelement <8 x float> %i976, i64 5, !dbg !12
  %i1007 = extractelement <8 x float> %i976, i64 6, !dbg !12
  %i1008 = extractelement <8 x float> %i976, i64 7, !dbg !12
  %i1009 = select i1 %.not, float %i1006, float %i1005, !dbg !12
  %i1010 = bitcast float %i1009 to i32, !dbg !12
  %i1011 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1010), !dbg !12
  %i1012 = bitcast i32 %i1011 to float, !dbg !12
  %i1013 = select i1 %.not, float %i1005, float %i1012, !dbg !12
  %i1014 = select i1 %.not, float %i1012, float %i1006, !dbg !12
  %i1015 = select i1 %.not, float %i1008, float %i1007, !dbg !12
  %i1016 = bitcast float %i1015 to i32, !dbg !12
  %i1017 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1016), !dbg !12
  %i1018 = bitcast i32 %i1017 to float, !dbg !12
  %i1019 = select i1 %.not, float %i1007, float %i1018, !dbg !12
  %i1020 = select i1 %.not, float %i1018, float %i1008, !dbg !12
  %i1021 = select i1 %.not40, float %i1019, float %i1013, !dbg !12
  %i1022 = bitcast float %i1021 to i32, !dbg !12
  %i1023 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1022), !dbg !12
  %i1024 = bitcast i32 %i1023 to float, !dbg !12
  %i1025 = select i1 %.not40, float %i1013, float %i1024, !dbg !12
  %i1026 = select i1 %.not40, float %i1024, float %i1019, !dbg !12
  %i1027 = select i1 %.not40, float %i1020, float %i1014, !dbg !12
  %i1028 = bitcast float %i1027 to i32, !dbg !12
  %i1029 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1028), !dbg !12
  %i1030 = bitcast i32 %i1029 to float, !dbg !12
  %i1031 = select i1 %.not40, float %i1014, float %i1030, !dbg !12
  %i1032 = select i1 %.not40, float %i1030, float %i1020, !dbg !12
  %i1033 = extractvalue [2 x <8 x float>] %i105, 1, !dbg !12
  %i1034 = extractelement <8 x float> %i1033, i64 0, !dbg !12
  %i1035 = extractelement <8 x float> %i1033, i64 1, !dbg !12
  %i1036 = extractelement <8 x float> %i1033, i64 2, !dbg !12
  %i1037 = extractelement <8 x float> %i1033, i64 3, !dbg !12
  %i1038 = select i1 %.not, float %i1035, float %i1034, !dbg !12
  %i1039 = bitcast float %i1038 to i32, !dbg !12
  %i1040 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1039), !dbg !12
  %i1041 = bitcast i32 %i1040 to float, !dbg !12
  %i1042 = select i1 %.not, float %i1034, float %i1041, !dbg !12
  %i1043 = select i1 %.not, float %i1041, float %i1035, !dbg !12
  %i1044 = select i1 %.not, float %i1037, float %i1036, !dbg !12
  %i1045 = bitcast float %i1044 to i32, !dbg !12
  %i1046 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1045), !dbg !12
  %i1047 = bitcast i32 %i1046 to float, !dbg !12
  %i1048 = select i1 %.not, float %i1036, float %i1047, !dbg !12
  %i1049 = select i1 %.not, float %i1047, float %i1037, !dbg !12
  %i1050 = select i1 %.not40, float %i1048, float %i1042, !dbg !12
  %i1051 = bitcast float %i1050 to i32, !dbg !12
  %i1052 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1051), !dbg !12
  %i1053 = bitcast i32 %i1052 to float, !dbg !12
  %i1054 = select i1 %.not40, float %i1042, float %i1053, !dbg !12
  %i1055 = select i1 %.not40, float %i1053, float %i1048, !dbg !12
  %i1056 = select i1 %.not40, float %i1049, float %i1043, !dbg !12
  %i1057 = bitcast float %i1056 to i32, !dbg !12
  %i1058 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1057), !dbg !12
  %i1059 = bitcast i32 %i1058 to float, !dbg !12
  %i1060 = select i1 %.not40, float %i1043, float %i1059, !dbg !12
  %i1061 = select i1 %.not40, float %i1059, float %i1049, !dbg !12
  %i1062 = extractelement <8 x float> %i1033, i64 4, !dbg !12
  %i1063 = extractelement <8 x float> %i1033, i64 5, !dbg !12
  %i1064 = extractelement <8 x float> %i1033, i64 6, !dbg !12
  %i1065 = extractelement <8 x float> %i1033, i64 7, !dbg !12
  %i1066 = select i1 %.not, float %i1063, float %i1062, !dbg !12
  %i1067 = bitcast float %i1066 to i32, !dbg !12
  %i1068 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1067), !dbg !12
  %i1069 = bitcast i32 %i1068 to float, !dbg !12
  %i1070 = select i1 %.not, float %i1062, float %i1069, !dbg !12
  %i1071 = select i1 %.not, float %i1069, float %i1063, !dbg !12
  %i1072 = select i1 %.not, float %i1065, float %i1064, !dbg !12
  %i1073 = bitcast float %i1072 to i32, !dbg !12
  %i1074 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1073), !dbg !12
  %i1075 = bitcast i32 %i1074 to float, !dbg !12
  %i1076 = select i1 %.not, float %i1064, float %i1075, !dbg !12
  %i1077 = select i1 %.not, float %i1075, float %i1065, !dbg !12
  %i1078 = select i1 %.not40, float %i1076, float %i1070, !dbg !12
  %i1079 = bitcast float %i1078 to i32, !dbg !12
  %i1080 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1079), !dbg !12
  %i1081 = bitcast i32 %i1080 to float, !dbg !12
  %i1082 = select i1 %.not40, float %i1070, float %i1081, !dbg !12
  %i1083 = select i1 %.not40, float %i1081, float %i1076, !dbg !12
  %i1084 = select i1 %.not40, float %i1077, float %i1071, !dbg !12
  %i1085 = bitcast float %i1084 to i32, !dbg !12
  %i1086 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1085), !dbg !12
  %i1087 = bitcast i32 %i1086 to float, !dbg !12
  %i1088 = select i1 %.not40, float %i1071, float %i1087, !dbg !12
  %i1089 = select i1 %.not40, float %i1087, float %i1077, !dbg !12
  %i1090 = extractvalue [2 x <8 x float>] %i104, 0, !dbg !12
  %i1091 = extractelement <8 x float> %i1090, i64 0, !dbg !12
  %i1092 = extractelement <8 x float> %i1090, i64 1, !dbg !12
  %i1093 = extractelement <8 x float> %i1090, i64 2, !dbg !12
  %i1094 = extractelement <8 x float> %i1090, i64 3, !dbg !12
  %i1095 = select i1 %.not, float %i1092, float %i1091, !dbg !12
  %i1096 = bitcast float %i1095 to i32, !dbg !12
  %i1097 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1096), !dbg !12
  %i1098 = bitcast i32 %i1097 to float, !dbg !12
  %i1099 = select i1 %.not, float %i1091, float %i1098, !dbg !12
  %i1100 = select i1 %.not, float %i1098, float %i1092, !dbg !12
  %i1101 = select i1 %.not, float %i1094, float %i1093, !dbg !12
  %i1102 = bitcast float %i1101 to i32, !dbg !12
  %i1103 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1102), !dbg !12
  %i1104 = bitcast i32 %i1103 to float, !dbg !12
  %i1105 = select i1 %.not, float %i1093, float %i1104, !dbg !12
  %i1106 = select i1 %.not, float %i1104, float %i1094, !dbg !12
  %i1107 = select i1 %.not40, float %i1105, float %i1099, !dbg !12
  %i1108 = bitcast float %i1107 to i32, !dbg !12
  %i1109 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1108), !dbg !12
  %i1110 = bitcast i32 %i1109 to float, !dbg !12
  %i1111 = select i1 %.not40, float %i1099, float %i1110, !dbg !12
  %i1112 = select i1 %.not40, float %i1110, float %i1105, !dbg !12
  %i1113 = select i1 %.not40, float %i1106, float %i1100, !dbg !12
  %i1114 = bitcast float %i1113 to i32, !dbg !12
  %i1115 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1114), !dbg !12
  %i1116 = bitcast i32 %i1115 to float, !dbg !12
  %i1117 = select i1 %.not40, float %i1100, float %i1116, !dbg !12
  %i1118 = select i1 %.not40, float %i1116, float %i1106, !dbg !12
  %i1119 = extractelement <8 x float> %i1090, i64 4, !dbg !12
  %i1120 = extractelement <8 x float> %i1090, i64 5, !dbg !12
  %i1121 = extractelement <8 x float> %i1090, i64 6, !dbg !12
  %i1122 = extractelement <8 x float> %i1090, i64 7, !dbg !12
  %i1123 = select i1 %.not, float %i1120, float %i1119, !dbg !12
  %i1124 = bitcast float %i1123 to i32, !dbg !12
  %i1125 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1124), !dbg !12
  %i1126 = bitcast i32 %i1125 to float, !dbg !12
  %i1127 = select i1 %.not, float %i1119, float %i1126, !dbg !12
  %i1128 = select i1 %.not, float %i1126, float %i1120, !dbg !12
  %i1129 = select i1 %.not, float %i1122, float %i1121, !dbg !12
  %i1130 = bitcast float %i1129 to i32, !dbg !12
  %i1131 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1130), !dbg !12
  %i1132 = bitcast i32 %i1131 to float, !dbg !12
  %i1133 = select i1 %.not, float %i1121, float %i1132, !dbg !12
  %i1134 = select i1 %.not, float %i1132, float %i1122, !dbg !12
  %i1135 = select i1 %.not40, float %i1133, float %i1127, !dbg !12
  %i1136 = bitcast float %i1135 to i32, !dbg !12
  %i1137 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1136), !dbg !12
  %i1138 = bitcast i32 %i1137 to float, !dbg !12
  %i1139 = select i1 %.not40, float %i1127, float %i1138, !dbg !12
  %i1140 = select i1 %.not40, float %i1138, float %i1133, !dbg !12
  %i1141 = select i1 %.not40, float %i1134, float %i1128, !dbg !12
  %i1142 = bitcast float %i1141 to i32, !dbg !12
  %i1143 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1142), !dbg !12
  %i1144 = bitcast i32 %i1143 to float, !dbg !12
  %i1145 = select i1 %.not40, float %i1128, float %i1144, !dbg !12
  %i1146 = select i1 %.not40, float %i1144, float %i1134, !dbg !12
  %i1147 = extractvalue [2 x <8 x float>] %i104, 1, !dbg !12
  %i1148 = extractelement <8 x float> %i1147, i64 0, !dbg !12
  %i1149 = extractelement <8 x float> %i1147, i64 1, !dbg !12
  %i1150 = extractelement <8 x float> %i1147, i64 2, !dbg !12
  %i1151 = extractelement <8 x float> %i1147, i64 3, !dbg !12
  %i1152 = select i1 %.not, float %i1149, float %i1148, !dbg !12
  %i1153 = bitcast float %i1152 to i32, !dbg !12
  %i1154 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1153), !dbg !12
  %i1155 = bitcast i32 %i1154 to float, !dbg !12
  %i1156 = select i1 %.not, float %i1148, float %i1155, !dbg !12
  %i1157 = select i1 %.not, float %i1155, float %i1149, !dbg !12
  %i1158 = select i1 %.not, float %i1151, float %i1150, !dbg !12
  %i1159 = bitcast float %i1158 to i32, !dbg !12
  %i1160 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1159), !dbg !12
  %i1161 = bitcast i32 %i1160 to float, !dbg !12
  %i1162 = select i1 %.not, float %i1150, float %i1161, !dbg !12
  %i1163 = select i1 %.not, float %i1161, float %i1151, !dbg !12
  %i1164 = select i1 %.not40, float %i1162, float %i1156, !dbg !12
  %i1165 = bitcast float %i1164 to i32, !dbg !12
  %i1166 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1165), !dbg !12
  %i1167 = bitcast i32 %i1166 to float, !dbg !12
  %i1168 = select i1 %.not40, float %i1156, float %i1167, !dbg !12
  %i1169 = select i1 %.not40, float %i1167, float %i1162, !dbg !12
  %i1170 = select i1 %.not40, float %i1163, float %i1157, !dbg !12
  %i1171 = bitcast float %i1170 to i32, !dbg !12
  %i1172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1171), !dbg !12
  %i1173 = bitcast i32 %i1172 to float, !dbg !12
  %i1174 = select i1 %.not40, float %i1157, float %i1173, !dbg !12
  %i1175 = select i1 %.not40, float %i1173, float %i1163, !dbg !12
  %i1176 = extractelement <8 x float> %i1147, i64 4, !dbg !12
  %i1177 = extractelement <8 x float> %i1147, i64 5, !dbg !12
  %i1178 = extractelement <8 x float> %i1147, i64 6, !dbg !12
  %i1179 = extractelement <8 x float> %i1147, i64 7, !dbg !12
  %i1180 = select i1 %.not, float %i1177, float %i1176, !dbg !12
  %i1181 = bitcast float %i1180 to i32, !dbg !12
  %i1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1181), !dbg !12
  %i1183 = bitcast i32 %i1182 to float, !dbg !12
  %i1184 = select i1 %.not, float %i1176, float %i1183, !dbg !12
  %i1185 = select i1 %.not, float %i1183, float %i1177, !dbg !12
  %i1186 = select i1 %.not, float %i1179, float %i1178, !dbg !12
  %i1187 = bitcast float %i1186 to i32, !dbg !12
  %i1188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1187), !dbg !12
  %i1189 = bitcast i32 %i1188 to float, !dbg !12
  %i1190 = select i1 %.not, float %i1178, float %i1189, !dbg !12
  %i1191 = select i1 %.not, float %i1189, float %i1179, !dbg !12
  %i1192 = select i1 %.not40, float %i1190, float %i1184, !dbg !12
  %i1193 = bitcast float %i1192 to i32, !dbg !12
  %i1194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1193), !dbg !12
  %i1195 = bitcast i32 %i1194 to float, !dbg !12
  %i1196 = select i1 %.not40, float %i1184, float %i1195, !dbg !12
  %i1197 = select i1 %.not40, float %i1195, float %i1190, !dbg !12
  %i1198 = select i1 %.not40, float %i1191, float %i1185, !dbg !12
  %i1199 = bitcast float %i1198 to i32, !dbg !12
  %i1200 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1199), !dbg !12
  %i1201 = bitcast i32 %i1200 to float, !dbg !12
  %i1202 = select i1 %.not40, float %i1185, float %i1201, !dbg !12
  %i1203 = select i1 %.not40, float %i1201, float %i1191, !dbg !12
  %i1204 = extractvalue [2 x <8 x float>] %i103, 0, !dbg !12
  %i1205 = extractelement <8 x float> %i1204, i64 0, !dbg !12
  %i1206 = extractelement <8 x float> %i1204, i64 1, !dbg !12
  %i1207 = extractelement <8 x float> %i1204, i64 2, !dbg !12
  %i1208 = extractelement <8 x float> %i1204, i64 3, !dbg !12
  %i1209 = select i1 %.not, float %i1206, float %i1205, !dbg !12
  %i1210 = bitcast float %i1209 to i32, !dbg !12
  %i1211 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1210), !dbg !12
  %i1212 = bitcast i32 %i1211 to float, !dbg !12
  %i1213 = select i1 %.not, float %i1205, float %i1212, !dbg !12
  %i1214 = select i1 %.not, float %i1212, float %i1206, !dbg !12
  %i1215 = select i1 %.not, float %i1208, float %i1207, !dbg !12
  %i1216 = bitcast float %i1215 to i32, !dbg !12
  %i1217 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1216), !dbg !12
  %i1218 = bitcast i32 %i1217 to float, !dbg !12
  %i1219 = select i1 %.not, float %i1207, float %i1218, !dbg !12
  %i1220 = select i1 %.not, float %i1218, float %i1208, !dbg !12
  %i1221 = select i1 %.not40, float %i1219, float %i1213, !dbg !12
  %i1222 = bitcast float %i1221 to i32, !dbg !12
  %i1223 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1222), !dbg !12
  %i1224 = bitcast i32 %i1223 to float, !dbg !12
  %i1225 = select i1 %.not40, float %i1213, float %i1224, !dbg !12
  %i1226 = select i1 %.not40, float %i1224, float %i1219, !dbg !12
  %i1227 = select i1 %.not40, float %i1220, float %i1214, !dbg !12
  %i1228 = bitcast float %i1227 to i32, !dbg !12
  %i1229 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1228), !dbg !12
  %i1230 = bitcast i32 %i1229 to float, !dbg !12
  %i1231 = select i1 %.not40, float %i1214, float %i1230, !dbg !12
  %i1232 = select i1 %.not40, float %i1230, float %i1220, !dbg !12
  %i1233 = extractelement <8 x float> %i1204, i64 4, !dbg !12
  %i1234 = extractelement <8 x float> %i1204, i64 5, !dbg !12
  %i1235 = extractelement <8 x float> %i1204, i64 6, !dbg !12
  %i1236 = extractelement <8 x float> %i1204, i64 7, !dbg !12
  %i1237 = select i1 %.not, float %i1234, float %i1233, !dbg !12
  %i1238 = bitcast float %i1237 to i32, !dbg !12
  %i1239 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1238), !dbg !12
  %i1240 = bitcast i32 %i1239 to float, !dbg !12
  %i1241 = select i1 %.not, float %i1233, float %i1240, !dbg !12
  %i1242 = select i1 %.not, float %i1240, float %i1234, !dbg !12
  %i1243 = select i1 %.not, float %i1236, float %i1235, !dbg !12
  %i1244 = bitcast float %i1243 to i32, !dbg !12
  %i1245 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1244), !dbg !12
  %i1246 = bitcast i32 %i1245 to float, !dbg !12
  %i1247 = select i1 %.not, float %i1235, float %i1246, !dbg !12
  %i1248 = select i1 %.not, float %i1246, float %i1236, !dbg !12
  %i1249 = select i1 %.not40, float %i1247, float %i1241, !dbg !12
  %i1250 = bitcast float %i1249 to i32, !dbg !12
  %i1251 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1250), !dbg !12
  %i1252 = bitcast i32 %i1251 to float, !dbg !12
  %i1253 = select i1 %.not40, float %i1241, float %i1252, !dbg !12
  %i1254 = select i1 %.not40, float %i1252, float %i1247, !dbg !12
  %i1255 = select i1 %.not40, float %i1248, float %i1242, !dbg !12
  %i1256 = bitcast float %i1255 to i32, !dbg !12
  %i1257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1256), !dbg !12
  %i1258 = bitcast i32 %i1257 to float, !dbg !12
  %i1259 = select i1 %.not40, float %i1242, float %i1258, !dbg !12
  %i1260 = select i1 %.not40, float %i1258, float %i1248, !dbg !12
  %i1261 = extractvalue [2 x <8 x float>] %i103, 1, !dbg !12
  %i1262 = extractelement <8 x float> %i1261, i64 0, !dbg !12
  %i1263 = extractelement <8 x float> %i1261, i64 1, !dbg !12
  %i1264 = extractelement <8 x float> %i1261, i64 2, !dbg !12
  %i1265 = extractelement <8 x float> %i1261, i64 3, !dbg !12
  %i1266 = select i1 %.not, float %i1263, float %i1262, !dbg !12
  %i1267 = bitcast float %i1266 to i32, !dbg !12
  %i1268 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1267), !dbg !12
  %i1269 = bitcast i32 %i1268 to float, !dbg !12
  %i1270 = select i1 %.not, float %i1262, float %i1269, !dbg !12
  %i1271 = select i1 %.not, float %i1269, float %i1263, !dbg !12
  %i1272 = select i1 %.not, float %i1265, float %i1264, !dbg !12
  %i1273 = bitcast float %i1272 to i32, !dbg !12
  %i1274 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1273), !dbg !12
  %i1275 = bitcast i32 %i1274 to float, !dbg !12
  %i1276 = select i1 %.not, float %i1264, float %i1275, !dbg !12
  %i1277 = select i1 %.not, float %i1275, float %i1265, !dbg !12
  %i1278 = select i1 %.not40, float %i1276, float %i1270, !dbg !12
  %i1279 = bitcast float %i1278 to i32, !dbg !12
  %i1280 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1279), !dbg !12
  %i1281 = bitcast i32 %i1280 to float, !dbg !12
  %i1282 = select i1 %.not40, float %i1270, float %i1281, !dbg !12
  %i1283 = select i1 %.not40, float %i1281, float %i1276, !dbg !12
  %i1284 = select i1 %.not40, float %i1277, float %i1271, !dbg !12
  %i1285 = bitcast float %i1284 to i32, !dbg !12
  %i1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1285), !dbg !12
  %i1287 = bitcast i32 %i1286 to float, !dbg !12
  %i1288 = select i1 %.not40, float %i1271, float %i1287, !dbg !12
  %i1289 = select i1 %.not40, float %i1287, float %i1277, !dbg !12
  %i1290 = extractelement <8 x float> %i1261, i64 4, !dbg !12
  %i1291 = extractelement <8 x float> %i1261, i64 5, !dbg !12
  %i1292 = extractelement <8 x float> %i1261, i64 6, !dbg !12
  %i1293 = extractelement <8 x float> %i1261, i64 7, !dbg !12
  %i1294 = select i1 %.not, float %i1291, float %i1290, !dbg !12
  %i1295 = bitcast float %i1294 to i32, !dbg !12
  %i1296 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1295), !dbg !12
  %i1297 = bitcast i32 %i1296 to float, !dbg !12
  %i1298 = select i1 %.not, float %i1290, float %i1297, !dbg !12
  %i1299 = select i1 %.not, float %i1297, float %i1291, !dbg !12
  %i1300 = select i1 %.not, float %i1293, float %i1292, !dbg !12
  %i1301 = bitcast float %i1300 to i32, !dbg !12
  %i1302 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1301), !dbg !12
  %i1303 = bitcast i32 %i1302 to float, !dbg !12
  %i1304 = select i1 %.not, float %i1292, float %i1303, !dbg !12
  %i1305 = select i1 %.not, float %i1303, float %i1293, !dbg !12
  %i1306 = select i1 %.not40, float %i1304, float %i1298, !dbg !12
  %i1307 = bitcast float %i1306 to i32, !dbg !12
  %i1308 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1307), !dbg !12
  %i1309 = bitcast i32 %i1308 to float, !dbg !12
  %i1310 = select i1 %.not40, float %i1298, float %i1309, !dbg !12
  %i1311 = select i1 %.not40, float %i1309, float %i1304, !dbg !12
  %i1312 = select i1 %.not40, float %i1305, float %i1299, !dbg !12
  %i1313 = bitcast float %i1312 to i32, !dbg !12
  %i1314 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1313), !dbg !12
  %i1315 = bitcast i32 %i1314 to float, !dbg !12
  %i1316 = select i1 %.not40, float %i1299, float %i1315, !dbg !12
  %i1317 = select i1 %.not40, float %i1315, float %i1305, !dbg !12
  %i1318 = extractvalue [2 x <8 x float>] %i100, 0, !dbg !12
  %i1319 = extractelement <8 x float> %i1318, i64 0, !dbg !12
  %i1320 = extractelement <8 x float> %i1318, i64 1, !dbg !12
  %i1321 = extractelement <8 x float> %i1318, i64 2, !dbg !12
  %i1322 = extractelement <8 x float> %i1318, i64 3, !dbg !12
  %i1323 = select i1 %.not, float %i1320, float %i1319, !dbg !12
  %i1324 = bitcast float %i1323 to i32, !dbg !12
  %i1325 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1324), !dbg !12
  %i1326 = bitcast i32 %i1325 to float, !dbg !12
  %i1327 = select i1 %.not, float %i1319, float %i1326, !dbg !12
  %i1328 = select i1 %.not, float %i1326, float %i1320, !dbg !12
  %i1329 = select i1 %.not, float %i1322, float %i1321, !dbg !12
  %i1330 = bitcast float %i1329 to i32, !dbg !12
  %i1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1330), !dbg !12
  %i1332 = bitcast i32 %i1331 to float, !dbg !12
  %i1333 = select i1 %.not, float %i1321, float %i1332, !dbg !12
  %i1334 = select i1 %.not, float %i1332, float %i1322, !dbg !12
  %i1335 = select i1 %.not40, float %i1333, float %i1327, !dbg !12
  %i1336 = bitcast float %i1335 to i32, !dbg !12
  %i1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1336), !dbg !12
  %i1338 = select i1 %.not40, float %i1334, float %i1328, !dbg !12
  %i1339 = bitcast float %i1338 to i32, !dbg !12
  %i1340 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1339), !dbg !12
  %i1341 = extractelement <8 x float> %i1318, i64 4, !dbg !12
  %i1342 = extractelement <8 x float> %i1318, i64 5, !dbg !12
  %i1343 = extractelement <8 x float> %i1318, i64 6, !dbg !12
  %i1344 = extractelement <8 x float> %i1318, i64 7, !dbg !12
  %i1345 = select i1 %.not, float %i1342, float %i1341, !dbg !12
  %i1346 = bitcast float %i1345 to i32, !dbg !12
  %i1347 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1346), !dbg !12
  %i1348 = bitcast i32 %i1347 to float, !dbg !12
  %i1349 = select i1 %.not, float %i1341, float %i1348, !dbg !12
  %i1350 = select i1 %.not, float %i1348, float %i1342, !dbg !12
  %i1351 = select i1 %.not, float %i1344, float %i1343, !dbg !12
  %i1352 = bitcast float %i1351 to i32, !dbg !12
  %i1353 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1352), !dbg !12
  %i1354 = bitcast i32 %i1353 to float, !dbg !12
  %i1355 = select i1 %.not, float %i1343, float %i1354, !dbg !12
  %i1356 = select i1 %.not, float %i1354, float %i1344, !dbg !12
  %i1357 = select i1 %.not40, float %i1355, float %i1349, !dbg !12
  %i1358 = bitcast float %i1357 to i32, !dbg !12
  %i1359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1358), !dbg !12
  %i1360 = select i1 %.not40, float %i1356, float %i1350, !dbg !12
  %i1361 = bitcast float %i1360 to i32, !dbg !12
  %i1362 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1361), !dbg !12
  %i1363 = extractvalue [2 x <8 x float>] %i100, 1, !dbg !12
  %i1364 = extractelement <8 x float> %i1363, i64 0, !dbg !12
  %i1365 = extractelement <8 x float> %i1363, i64 1, !dbg !12
  %i1366 = extractelement <8 x float> %i1363, i64 2, !dbg !12
  %i1367 = extractelement <8 x float> %i1363, i64 3, !dbg !12
  %i1368 = select i1 %.not, float %i1365, float %i1364, !dbg !12
  %i1369 = bitcast float %i1368 to i32, !dbg !12
  %i1370 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1369), !dbg !12
  %i1371 = bitcast i32 %i1370 to float, !dbg !12
  %i1372 = select i1 %.not, float %i1364, float %i1371, !dbg !12
  %i1373 = select i1 %.not, float %i1371, float %i1365, !dbg !12
  %i1374 = select i1 %.not, float %i1367, float %i1366, !dbg !12
  %i1375 = bitcast float %i1374 to i32, !dbg !12
  %i1376 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1375), !dbg !12
  %i1377 = bitcast i32 %i1376 to float, !dbg !12
  %i1378 = select i1 %.not, float %i1366, float %i1377, !dbg !12
  %i1379 = select i1 %.not, float %i1377, float %i1367, !dbg !12
  %i1380 = select i1 %.not40, float %i1378, float %i1372, !dbg !12
  %i1381 = bitcast float %i1380 to i32, !dbg !12
  %i1382 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1381), !dbg !12
  %i1383 = select i1 %.not40, float %i1379, float %i1373, !dbg !12
  %i1384 = bitcast float %i1383 to i32, !dbg !12
  %i1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1384), !dbg !12
  %i1386 = extractelement <8 x float> %i1363, i64 4, !dbg !12
  %i1387 = extractelement <8 x float> %i1363, i64 5, !dbg !12
  %i1388 = extractelement <8 x float> %i1363, i64 6, !dbg !12
  %i1389 = extractelement <8 x float> %i1363, i64 7, !dbg !12
  %i1390 = select i1 %.not, float %i1387, float %i1386, !dbg !12
  %i1391 = bitcast float %i1390 to i32, !dbg !12
  %i1392 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1391), !dbg !12
  %i1393 = bitcast i32 %i1392 to float, !dbg !12
  %i1394 = select i1 %.not, float %i1386, float %i1393, !dbg !12
  %i1395 = select i1 %.not, float %i1393, float %i1387, !dbg !12
  %i1396 = select i1 %.not, float %i1389, float %i1388, !dbg !12
  %i1397 = bitcast float %i1396 to i32, !dbg !12
  %i1398 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i12, i32 %i1397), !dbg !12
  %i1399 = bitcast i32 %i1398 to float, !dbg !12
  %i1400 = select i1 %.not, float %i1388, float %i1399, !dbg !12
  %i1401 = select i1 %.not, float %i1399, float %i1389, !dbg !12
  %i1402 = select i1 %.not40, float %i1400, float %i1394, !dbg !12
  %i1403 = bitcast float %i1402 to i32, !dbg !12
  %i1404 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1403), !dbg !12
  %i1405 = select i1 %.not40, float %i1401, float %i1395, !dbg !12
  %i1406 = bitcast float %i1405 to i32, !dbg !12
  %i1407 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i16, i32 %i1406), !dbg !12
  %i1408 = shufflevector <1 x float> %i973, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %i1410 = shufflevector <1 x float> %i968, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %i1412 = bitcast i32 %i1404 to float, !dbg !12
  %i1413 = select i1 %.not40, float %i1394, float %i1412, !dbg !12
  %i1414 = insertelement <4 x float> poison, float %i1413, i64 0, !dbg !46
  %i1415 = bitcast i32 %i1407 to float, !dbg !12
  %i1416 = select i1 %.not40, float %i1395, float %i1415, !dbg !12
  %i1417 = insertelement <4 x float> %i1414, float %i1416, i64 1, !dbg !46
  %i1418 = select i1 %.not40, float %i1412, float %i1400, !dbg !12
  %i1419 = insertelement <4 x float> %i1417, float %i1418, i64 2, !dbg !46
  %i1420 = select i1 %.not40, float %i1415, float %i1401, !dbg !12
  %i1421 = insertelement <4 x float> %i1419, float %i1420, i64 3, !dbg !46
  %i1411 = fdiv arcp <4 x float> splat (float 1.000000e+00), %i1410, !dbg !46
  %i1422 = fmul <4 x float> %i1421, %i1411, !dbg !46
  %i1423 = fptrunc <4 x float> %i1422 to <4 x half>, !dbg !47
  %i1424 = extractelement <4 x half> %i1423, i64 3, !dbg !47
  %i1425 = extractelement <4 x half> %i1423, i64 2, !dbg !47
  %i1426 = extractelement <4 x half> %i1423, i64 1, !dbg !47
  %i1427 = extractelement <4 x half> %i1423, i64 0, !dbg !47
  %i1428 = bitcast i32 %i1382 to float, !dbg !12
  %i1429 = select i1 %.not40, float %i1372, float %i1428, !dbg !12
  %i1430 = insertelement <4 x float> poison, float %i1429, i64 0, !dbg !46
  %i1431 = bitcast i32 %i1385 to float, !dbg !12
  %i1432 = select i1 %.not40, float %i1373, float %i1431, !dbg !12
  %i1433 = insertelement <4 x float> %i1430, float %i1432, i64 1, !dbg !46
  %i1434 = select i1 %.not40, float %i1428, float %i1378, !dbg !12
  %i1435 = insertelement <4 x float> %i1433, float %i1434, i64 2, !dbg !46
  %i1436 = select i1 %.not40, float %i1431, float %i1379, !dbg !12
  %i1437 = insertelement <4 x float> %i1435, float %i1436, i64 3, !dbg !46
  %i1438 = fmul <4 x float> %i1437, %i1411, !dbg !46
  %i1439 = fptrunc <4 x float> %i1438 to <4 x half>, !dbg !47
  %i1440 = extractelement <4 x half> %i1439, i64 3, !dbg !47
  %i1441 = extractelement <4 x half> %i1439, i64 2, !dbg !47
  %i1442 = extractelement <4 x half> %i1439, i64 1, !dbg !47
  %i1443 = extractelement <4 x half> %i1439, i64 0, !dbg !47
  %i1444 = bitcast i32 %i1359 to float, !dbg !12
  %i1445 = select i1 %.not40, float %i1349, float %i1444, !dbg !12
  %i1446 = insertelement <4 x float> poison, float %i1445, i64 0, !dbg !46
  %i1447 = bitcast i32 %i1362 to float, !dbg !12
  %i1448 = select i1 %.not40, float %i1350, float %i1447, !dbg !12
  %i1449 = insertelement <4 x float> %i1446, float %i1448, i64 1, !dbg !46
  %i1450 = select i1 %.not40, float %i1444, float %i1355, !dbg !12
  %i1451 = insertelement <4 x float> %i1449, float %i1450, i64 2, !dbg !46
  %i1452 = select i1 %.not40, float %i1447, float %i1356, !dbg !12
  %i1453 = insertelement <4 x float> %i1451, float %i1452, i64 3, !dbg !46
  %i1409 = fdiv arcp <4 x float> splat (float 1.000000e+00), %i1408, !dbg !46
  %i1454 = fmul <4 x float> %i1453, %i1409, !dbg !46
  %i1455 = fptrunc <4 x float> %i1454 to <4 x half>, !dbg !47
  %i1456 = extractelement <4 x half> %i1455, i64 3, !dbg !47
  %i1457 = extractelement <4 x half> %i1455, i64 2, !dbg !47
  %i1458 = extractelement <4 x half> %i1455, i64 1, !dbg !47
  %i1459 = extractelement <4 x half> %i1455, i64 0, !dbg !47
  %i1460 = bitcast i32 %i1337 to float, !dbg !12
  %i1461 = select i1 %.not40, float %i1327, float %i1460, !dbg !12
  %i1462 = insertelement <4 x float> poison, float %i1461, i64 0, !dbg !46
  %i1463 = bitcast i32 %i1340 to float, !dbg !12
  %i1464 = select i1 %.not40, float %i1328, float %i1463, !dbg !12
  %i1465 = insertelement <4 x float> %i1462, float %i1464, i64 1, !dbg !46
  %i1466 = select i1 %.not40, float %i1460, float %i1333, !dbg !12
  %i1467 = insertelement <4 x float> %i1465, float %i1466, i64 2, !dbg !46
  %i1468 = select i1 %.not40, float %i1463, float %i1334, !dbg !12
  %i1469 = insertelement <4 x float> %i1467, float %i1468, i64 3, !dbg !46
  %i1470 = fmul <4 x float> %i1469, %i1409, !dbg !46
  %i1471 = fptrunc <4 x float> %i1470 to <4 x half>, !dbg !47
  %i1472 = extractelement <4 x half> %i1471, i64 3, !dbg !47
  %i1473 = extractelement <4 x half> %i1471, i64 2, !dbg !47
  %i1474 = extractelement <4 x half> %i1471, i64 1, !dbg !47
  %i1475 = extractelement <4 x half> %i1471, i64 0, !dbg !47
  %address.thread971 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !48, !range !10, !frisk.register_lifetimes !5
  %address.local972 = zext nneg i32 %address.thread971 to i64, !dbg !9
  %address.local973 = add nuw i64 %i30, %address.local972, !dbg !48
  %i1476 = lshr i64 %address.local972, 2, !dbg !48
  %address.local975 = and i64 %i1476, 48, !dbg !48
  %address.local976 = add i64 %address.local973, %address.local975, !dbg !48
  %address.local977 = and i64 %address.local972, 240, !dbg !48
  %address.local978 = sub i64 %address.local976, %address.local977, !dbg !48
  %address.local979 = add i64 %address.local978, %i31, !dbg !48
  %address.local980 = shl nuw nsw i64 %address.local979, 7, !dbg !48
  %i1477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %arg3, i64 %i47, !dbg !48
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %i1477, i64 %address.local980, !dbg !48
  %address.local984 = and i64 %i1476, 12, !dbg !11
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local984, !dbg !48
  store half %i1475, ptr addrspace(1) %address.local985, align 2, !dbg !48
  %address.local987 = or disjoint i64 %address.local984, 1, !dbg !48
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local987, !dbg !48
  store half %i1474, ptr addrspace(1) %address.local988, align 2, !dbg !48
  %address.local990 = or disjoint i64 %address.local984, 2, !dbg !48
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local990, !dbg !48
  store half %i1473, ptr addrspace(1) %address.local991, align 2, !dbg !48
  %address.local993 = or disjoint i64 %address.local984, 3, !dbg !48
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local993, !dbg !48
  store half %i1472, ptr addrspace(1) %address.local994, align 2, !dbg !48
  %address.local996 = or disjoint i64 %address.local984, 16, !dbg !48
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local996, !dbg !48
  store half %i1459, ptr addrspace(1) %address.local997, align 2, !dbg !48
  %address.local999 = or disjoint i64 %address.local984, 17, !dbg !48
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local999, !dbg !48
  store half %i1458, ptr addrspace(1) %address.local1000, align 2, !dbg !48
  %address.local1002 = or disjoint i64 %address.local984, 18, !dbg !48
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1002, !dbg !48
  store half %i1457, ptr addrspace(1) %address.local1003, align 2, !dbg !48
  %address.local1005 = or disjoint i64 %address.local984, 19, !dbg !48
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1005, !dbg !48
  store half %i1456, ptr addrspace(1) %address.local1006, align 2, !dbg !48
  %address.local1010 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local982, i64 16384, !dbg !48
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local984, !dbg !48
  store half %i1443, ptr addrspace(1) %address.local1011, align 2, !dbg !48
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local987, !dbg !48
  store half %i1442, ptr addrspace(1) %address.local1013, align 2, !dbg !48
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local990, !dbg !48
  store half %i1441, ptr addrspace(1) %address.local1015, align 2, !dbg !48
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local993, !dbg !48
  store half %i1440, ptr addrspace(1) %address.local1017, align 2, !dbg !48
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local996, !dbg !48
  store half %i1427, ptr addrspace(1) %address.local1019, align 2, !dbg !48
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local999, !dbg !48
  store half %i1426, ptr addrspace(1) %address.local1021, align 2, !dbg !48
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1002, !dbg !48
  store half %i1425, ptr addrspace(1) %address.local1023, align 2, !dbg !48
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1005, !dbg !48
  store half %i1424, ptr addrspace(1) %address.local1025, align 2, !dbg !48
  %i1478 = insertelement <4 x float> poison, float %i1310, i64 0, !dbg !46
  %i1479 = insertelement <4 x float> %i1478, float %i1316, i64 1, !dbg !46
  %i1480 = insertelement <4 x float> %i1479, float %i1311, i64 2, !dbg !46
  %i1481 = insertelement <4 x float> %i1480, float %i1317, i64 3, !dbg !46
  %i1482 = fmul <4 x float> %i1481, %i1411, !dbg !46
  %i1483 = fptrunc <4 x float> %i1482 to <4 x half>, !dbg !47
  %i1484 = extractelement <4 x half> %i1483, i64 3, !dbg !47
  %i1485 = extractelement <4 x half> %i1483, i64 2, !dbg !47
  %i1486 = extractelement <4 x half> %i1483, i64 1, !dbg !47
  %i1487 = extractelement <4 x half> %i1483, i64 0, !dbg !47
  %i1488 = insertelement <4 x float> poison, float %i1282, i64 0, !dbg !46
  %i1489 = insertelement <4 x float> %i1488, float %i1288, i64 1, !dbg !46
  %i1490 = insertelement <4 x float> %i1489, float %i1283, i64 2, !dbg !46
  %i1491 = insertelement <4 x float> %i1490, float %i1289, i64 3, !dbg !46
  %i1492 = fmul <4 x float> %i1491, %i1411, !dbg !46
  %i1493 = fptrunc <4 x float> %i1492 to <4 x half>, !dbg !47
  %i1494 = extractelement <4 x half> %i1493, i64 3, !dbg !47
  %i1495 = extractelement <4 x half> %i1493, i64 2, !dbg !47
  %i1496 = extractelement <4 x half> %i1493, i64 1, !dbg !47
  %i1497 = extractelement <4 x half> %i1493, i64 0, !dbg !47
  %i1498 = insertelement <4 x float> poison, float %i1253, i64 0, !dbg !46
  %i1499 = insertelement <4 x float> %i1498, float %i1259, i64 1, !dbg !46
  %i1500 = insertelement <4 x float> %i1499, float %i1254, i64 2, !dbg !46
  %i1501 = insertelement <4 x float> %i1500, float %i1260, i64 3, !dbg !46
  %i1502 = fmul <4 x float> %i1501, %i1409, !dbg !46
  %i1503 = fptrunc <4 x float> %i1502 to <4 x half>, !dbg !47
  %i1504 = extractelement <4 x half> %i1503, i64 3, !dbg !47
  %i1505 = extractelement <4 x half> %i1503, i64 2, !dbg !47
  %i1506 = extractelement <4 x half> %i1503, i64 1, !dbg !47
  %i1507 = extractelement <4 x half> %i1503, i64 0, !dbg !47
  %i1508 = insertelement <4 x float> poison, float %i1225, i64 0, !dbg !46
  %i1509 = insertelement <4 x float> %i1508, float %i1231, i64 1, !dbg !46
  %i1510 = insertelement <4 x float> %i1509, float %i1226, i64 2, !dbg !46
  %i1511 = insertelement <4 x float> %i1510, float %i1232, i64 3, !dbg !46
  %i1512 = fmul <4 x float> %i1511, %i1409, !dbg !46
  %i1513 = fptrunc <4 x float> %i1512 to <4 x half>, !dbg !47
  %i1514 = extractelement <4 x half> %i1513, i64 3, !dbg !47
  %i1515 = extractelement <4 x half> %i1513, i64 2, !dbg !47
  %i1516 = extractelement <4 x half> %i1513, i64 1, !dbg !47
  %i1517 = extractelement <4 x half> %i1513, i64 0, !dbg !47
  %address.local1040 = or disjoint i64 %address.local984, 32, !dbg !48
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1040, !dbg !48
  store half %i1517, ptr addrspace(1) %address.local1041, align 2, !dbg !48
  %address.local1043 = or disjoint i64 %address.local984, 33, !dbg !48
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1043, !dbg !48
  store half %i1516, ptr addrspace(1) %address.local1044, align 2, !dbg !48
  %address.local1046 = or disjoint i64 %address.local984, 34, !dbg !48
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1046, !dbg !48
  store half %i1515, ptr addrspace(1) %address.local1047, align 2, !dbg !48
  %address.local1049 = or disjoint i64 %address.local984, 35, !dbg !48
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1049, !dbg !48
  store half %i1514, ptr addrspace(1) %address.local1050, align 2, !dbg !48
  %address.local1052 = or disjoint i64 %address.local984, 48, !dbg !48
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1052, !dbg !48
  store half %i1507, ptr addrspace(1) %address.local1053, align 2, !dbg !48
  %address.local1055 = or disjoint i64 %address.local984, 49, !dbg !48
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1055, !dbg !48
  store half %i1506, ptr addrspace(1) %address.local1056, align 2, !dbg !48
  %address.local1058 = or disjoint i64 %address.local984, 50, !dbg !48
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1058, !dbg !48
  store half %i1505, ptr addrspace(1) %address.local1059, align 2, !dbg !48
  %address.local1061 = or i64 %i1476, 51, !dbg !48
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1061, !dbg !48
  store half %i1504, ptr addrspace(1) %address.local1062, align 2, !dbg !48
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1040, !dbg !48
  store half %i1497, ptr addrspace(1) %address.local1067, align 2, !dbg !48
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1043, !dbg !48
  store half %i1496, ptr addrspace(1) %address.local1069, align 2, !dbg !48
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1046, !dbg !48
  store half %i1495, ptr addrspace(1) %address.local1071, align 2, !dbg !48
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1049, !dbg !48
  store half %i1494, ptr addrspace(1) %address.local1073, align 2, !dbg !48
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1052, !dbg !48
  store half %i1487, ptr addrspace(1) %address.local1075, align 2, !dbg !48
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1055, !dbg !48
  store half %i1486, ptr addrspace(1) %address.local1077, align 2, !dbg !48
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1058, !dbg !48
  store half %i1485, ptr addrspace(1) %address.local1079, align 2, !dbg !48
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1061, !dbg !48
  store half %i1484, ptr addrspace(1) %address.local1081, align 2, !dbg !48
  %i1518 = insertelement <4 x float> poison, float %i1196, i64 0, !dbg !46
  %i1519 = insertelement <4 x float> %i1518, float %i1202, i64 1, !dbg !46
  %i1520 = insertelement <4 x float> %i1519, float %i1197, i64 2, !dbg !46
  %i1521 = insertelement <4 x float> %i1520, float %i1203, i64 3, !dbg !46
  %i1522 = fmul <4 x float> %i1521, %i1411, !dbg !46
  %i1523 = fptrunc <4 x float> %i1522 to <4 x half>, !dbg !47
  %i1524 = extractelement <4 x half> %i1523, i64 3, !dbg !47
  %i1525 = extractelement <4 x half> %i1523, i64 2, !dbg !47
  %i1526 = extractelement <4 x half> %i1523, i64 1, !dbg !47
  %i1527 = extractelement <4 x half> %i1523, i64 0, !dbg !47
  %i1528 = insertelement <4 x float> poison, float %i1168, i64 0, !dbg !46
  %i1529 = insertelement <4 x float> %i1528, float %i1174, i64 1, !dbg !46
  %i1530 = insertelement <4 x float> %i1529, float %i1169, i64 2, !dbg !46
  %i1531 = insertelement <4 x float> %i1530, float %i1175, i64 3, !dbg !46
  %i1532 = fmul <4 x float> %i1531, %i1411, !dbg !46
  %i1533 = fptrunc <4 x float> %i1532 to <4 x half>, !dbg !47
  %i1534 = extractelement <4 x half> %i1533, i64 3, !dbg !47
  %i1535 = extractelement <4 x half> %i1533, i64 2, !dbg !47
  %i1536 = extractelement <4 x half> %i1533, i64 1, !dbg !47
  %i1537 = extractelement <4 x half> %i1533, i64 0, !dbg !47
  %i1538 = insertelement <4 x float> poison, float %i1139, i64 0, !dbg !46
  %i1539 = insertelement <4 x float> %i1538, float %i1145, i64 1, !dbg !46
  %i1540 = insertelement <4 x float> %i1539, float %i1140, i64 2, !dbg !46
  %i1541 = insertelement <4 x float> %i1540, float %i1146, i64 3, !dbg !46
  %i1542 = fmul <4 x float> %i1541, %i1409, !dbg !46
  %i1543 = fptrunc <4 x float> %i1542 to <4 x half>, !dbg !47
  %i1544 = extractelement <4 x half> %i1543, i64 3, !dbg !47
  %i1545 = extractelement <4 x half> %i1543, i64 2, !dbg !47
  %i1546 = extractelement <4 x half> %i1543, i64 1, !dbg !47
  %i1547 = extractelement <4 x half> %i1543, i64 0, !dbg !47
  %i1548 = insertelement <4 x float> poison, float %i1111, i64 0, !dbg !46
  %i1549 = insertelement <4 x float> %i1548, float %i1117, i64 1, !dbg !46
  %i1550 = insertelement <4 x float> %i1549, float %i1112, i64 2, !dbg !46
  %i1551 = insertelement <4 x float> %i1550, float %i1118, i64 3, !dbg !46
  %i1552 = fmul <4 x float> %i1551, %i1409, !dbg !46
  %i1553 = fptrunc <4 x float> %i1552 to <4 x half>, !dbg !47
  %i1554 = extractelement <4 x half> %i1553, i64 3, !dbg !47
  %i1555 = extractelement <4 x half> %i1553, i64 2, !dbg !47
  %i1556 = extractelement <4 x half> %i1553, i64 1, !dbg !47
  %i1557 = extractelement <4 x half> %i1553, i64 0, !dbg !47
  %address.local1096 = or disjoint i64 %address.local984, 64, !dbg !48
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1096, !dbg !48
  store half %i1557, ptr addrspace(1) %address.local1097, align 2, !dbg !48
  %address.local1099 = or disjoint i64 %address.local984, 65, !dbg !48
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1099, !dbg !48
  store half %i1556, ptr addrspace(1) %address.local1100, align 2, !dbg !48
  %address.local1102 = or disjoint i64 %address.local984, 66, !dbg !48
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1102, !dbg !48
  store half %i1555, ptr addrspace(1) %address.local1103, align 2, !dbg !48
  %address.local1105 = or disjoint i64 %address.local984, 67, !dbg !48
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1105, !dbg !48
  store half %i1554, ptr addrspace(1) %address.local1106, align 2, !dbg !48
  %address.local1108 = or disjoint i64 %address.local984, 80, !dbg !48
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1108, !dbg !48
  store half %i1547, ptr addrspace(1) %address.local1109, align 2, !dbg !48
  %address.local1111 = or disjoint i64 %address.local984, 81, !dbg !48
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1111, !dbg !48
  store half %i1546, ptr addrspace(1) %address.local1112, align 2, !dbg !48
  %address.local1114 = or disjoint i64 %address.local984, 82, !dbg !48
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1114, !dbg !48
  store half %i1545, ptr addrspace(1) %address.local1115, align 2, !dbg !48
  %address.local1117 = or disjoint i64 %address.local984, 83, !dbg !48
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1117, !dbg !48
  store half %i1544, ptr addrspace(1) %address.local1118, align 2, !dbg !48
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1096, !dbg !48
  store half %i1537, ptr addrspace(1) %address.local1123, align 2, !dbg !48
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1099, !dbg !48
  store half %i1536, ptr addrspace(1) %address.local1125, align 2, !dbg !48
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1102, !dbg !48
  store half %i1535, ptr addrspace(1) %address.local1127, align 2, !dbg !48
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1105, !dbg !48
  store half %i1534, ptr addrspace(1) %address.local1129, align 2, !dbg !48
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1108, !dbg !48
  store half %i1527, ptr addrspace(1) %address.local1131, align 2, !dbg !48
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1111, !dbg !48
  store half %i1526, ptr addrspace(1) %address.local1133, align 2, !dbg !48
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1114, !dbg !48
  store half %i1525, ptr addrspace(1) %address.local1135, align 2, !dbg !48
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1117, !dbg !48
  store half %i1524, ptr addrspace(1) %address.local1137, align 2, !dbg !48
  %i1558 = insertelement <4 x float> poison, float %i1082, i64 0, !dbg !46
  %i1559 = insertelement <4 x float> %i1558, float %i1088, i64 1, !dbg !46
  %i1560 = insertelement <4 x float> %i1559, float %i1083, i64 2, !dbg !46
  %i1561 = insertelement <4 x float> %i1560, float %i1089, i64 3, !dbg !46
  %i1562 = fmul <4 x float> %i1561, %i1411, !dbg !46
  %i1563 = fptrunc <4 x float> %i1562 to <4 x half>, !dbg !47
  %i1564 = extractelement <4 x half> %i1563, i64 3, !dbg !47
  %i1565 = extractelement <4 x half> %i1563, i64 2, !dbg !47
  %i1566 = extractelement <4 x half> %i1563, i64 1, !dbg !47
  %i1567 = extractelement <4 x half> %i1563, i64 0, !dbg !47
  %i1568 = insertelement <4 x float> poison, float %i1054, i64 0, !dbg !46
  %i1569 = insertelement <4 x float> %i1568, float %i1060, i64 1, !dbg !46
  %i1570 = insertelement <4 x float> %i1569, float %i1055, i64 2, !dbg !46
  %i1571 = insertelement <4 x float> %i1570, float %i1061, i64 3, !dbg !46
  %i1572 = fmul <4 x float> %i1571, %i1411, !dbg !46
  %i1573 = fptrunc <4 x float> %i1572 to <4 x half>, !dbg !47
  %i1574 = extractelement <4 x half> %i1573, i64 3, !dbg !47
  %i1575 = extractelement <4 x half> %i1573, i64 2, !dbg !47
  %i1576 = extractelement <4 x half> %i1573, i64 1, !dbg !47
  %i1577 = extractelement <4 x half> %i1573, i64 0, !dbg !47
  %i1578 = insertelement <4 x float> poison, float %i1025, i64 0, !dbg !46
  %i1579 = insertelement <4 x float> %i1578, float %i1031, i64 1, !dbg !46
  %i1580 = insertelement <4 x float> %i1579, float %i1026, i64 2, !dbg !46
  %i1581 = insertelement <4 x float> %i1580, float %i1032, i64 3, !dbg !46
  %i1582 = fmul <4 x float> %i1581, %i1409, !dbg !46
  %i1583 = fptrunc <4 x float> %i1582 to <4 x half>, !dbg !47
  %i1584 = extractelement <4 x half> %i1583, i64 3, !dbg !47
  %i1585 = extractelement <4 x half> %i1583, i64 2, !dbg !47
  %i1586 = extractelement <4 x half> %i1583, i64 1, !dbg !47
  %i1587 = extractelement <4 x half> %i1583, i64 0, !dbg !47
  %i1588 = insertelement <4 x float> poison, float %i997, i64 0, !dbg !46
  %i1589 = insertelement <4 x float> %i1588, float %i1003, i64 1, !dbg !46
  %i1590 = insertelement <4 x float> %i1589, float %i998, i64 2, !dbg !46
  %i1591 = insertelement <4 x float> %i1590, float %i1004, i64 3, !dbg !46
  %i1592 = fmul <4 x float> %i1591, %i1409, !dbg !46
  %i1593 = fptrunc <4 x float> %i1592 to <4 x half>, !dbg !47
  %i1594 = extractelement <4 x half> %i1593, i64 3, !dbg !47
  %i1595 = extractelement <4 x half> %i1593, i64 2, !dbg !47
  %i1596 = extractelement <4 x half> %i1593, i64 1, !dbg !47
  %i1597 = extractelement <4 x half> %i1593, i64 0, !dbg !47
  %address.local1152 = or disjoint i64 %address.local984, 96, !dbg !48
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1152, !dbg !48
  store half %i1597, ptr addrspace(1) %address.local1153, align 2, !dbg !48
  %address.local1155 = or disjoint i64 %address.local984, 97, !dbg !48
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1155, !dbg !48
  store half %i1596, ptr addrspace(1) %address.local1156, align 2, !dbg !48
  %address.local1158 = or disjoint i64 %address.local984, 98, !dbg !48
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1158, !dbg !48
  store half %i1595, ptr addrspace(1) %address.local1159, align 2, !dbg !48
  %address.local1161 = or disjoint i64 %address.local984, 99, !dbg !48
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1161, !dbg !48
  store half %i1594, ptr addrspace(1) %address.local1162, align 2, !dbg !48
  %address.local1164 = or disjoint i64 %address.local984, 112, !dbg !48
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1164, !dbg !48
  store half %i1587, ptr addrspace(1) %address.local1165, align 2, !dbg !48
  %address.local1167 = or disjoint i64 %address.local984, 113, !dbg !48
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1167, !dbg !48
  store half %i1586, ptr addrspace(1) %address.local1168, align 2, !dbg !48
  %address.local1170 = or disjoint i64 %address.local984, 114, !dbg !48
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1170, !dbg !48
  store half %i1585, ptr addrspace(1) %address.local1171, align 2, !dbg !48
  %address.local1173 = or i64 %i1476, 115, !dbg !48
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1173, !dbg !48
  store half %i1584, ptr addrspace(1) %address.local1174, align 2, !dbg !48
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1152, !dbg !48
  store half %i1577, ptr addrspace(1) %address.local1179, align 2, !dbg !48
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1155, !dbg !48
  store half %i1576, ptr addrspace(1) %address.local1181, align 2, !dbg !48
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1158, !dbg !48
  store half %i1575, ptr addrspace(1) %address.local1183, align 2, !dbg !48
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1161, !dbg !48
  store half %i1574, ptr addrspace(1) %address.local1185, align 2, !dbg !48
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1164, !dbg !48
  store half %i1567, ptr addrspace(1) %address.local1187, align 2, !dbg !48
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1167, !dbg !48
  store half %i1566, ptr addrspace(1) %address.local1189, align 2, !dbg !48
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1170, !dbg !48
  store half %i1565, ptr addrspace(1) %address.local1191, align 2, !dbg !48
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1173, !dbg !48
  store half %i1564, ptr addrspace(1) %address.local1193, align 2, !dbg !48
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
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #5

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #6

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.ballot.i64(i1) #4

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.lazy_softmax_base"="v1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math")
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
!40 = !DILocation(line: 88, column: 22, scope: !3)
!41 = !DILocation(line: 89, column: 20, scope: !3)
!42 = !{i64 8192}
!43 = !DILocation(line: 79, column: 22, scope: !3)
!44 = !DILocation(line: 81, column: 16, scope: !3)
!45 = !DILocation(line: 80, column: 20, scope: !3)
!46 = !DILocation(line: 92, column: 23, scope: !3)
!47 = !DILocation(line: 93, column: 19, scope: !3)
!48 = !DILocation(line: 94, column: 9, scope: !3)
