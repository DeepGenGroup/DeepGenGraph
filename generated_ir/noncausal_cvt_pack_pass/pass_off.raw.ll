; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_cvt_pack/lazy_t8.ll'
source_filename = "generated_ir/noncausal_k_lds_prefetch_pass/mytest_final/run-20261030-171621-98fko2hz/online-vdb-k64-lds2/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %arg, ptr addrspace(1) %arg1, ptr addrspace(1) %arg2, ptr addrspace(1) %arg3) #1 !dbg !54 !reqd_work_group_size !57 {
bb:
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !58
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !58
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !58
  %i = and i32 %kdma.lane, 31, !dbg !58
  %kdma.owner = or disjoint i32 %kdma.wave, %i, !dbg !58
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !58
  %i4 = shl nuw nsw i32 %kdma.tid, 3, !dbg !58
  %kdma.intra = and i32 %i4, 8, !dbg !58
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !58
  %i5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !58
  %i6 = sext i32 %i5 to i64, !dbg !58
  %i7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !58
  %i8 = sext i32 %i7 to i64, !dbg !58
  %i9 = shl nsw i64 %i8, 5, !dbg !58
  %i10 = add nsw i64 %i9, %i6, !dbg !58
  %i11 = lshr i64 %i10, 5, !dbg !58
  %i12 = and i64 %i11, 576460752303423456, !dbg !58
  %i13 = and i64 %i6, 31, !dbg !58
  %i14 = or disjoint i64 %i12, %i13, !dbg !58
  %i15 = lshr i64 %i10, 10, !dbg !59
  %i16 = add nuw nsw i64 %i11, %i15, !dbg !59
  %i17 = shl nsw i64 %i14, 7, !dbg !59
  %i18 = mul nsw i64 %i15, -4096, !dbg !59
  %i19 = add i64 %i17, %i18, !dbg !59
  %i20 = shl i64 %i16, 19, !dbg !59
  %i21 = and i64 %i20, 16252928, !dbg !59
  %i22 = shl nsw i64 %i19, 7, !dbg !59
  %i23 = add i64 %i21, %i22, !dbg !59
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !59, !range !60, !frisk.register_lifetimes !56
  %address.local = zext nneg i32 %address.thread to i64, !dbg !59
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !59
  %address.local405 = and i64 %address.local404, 1536, !dbg !59
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !59
  %address.local407 = ptrtoint ptr addrspace(3) %address.local406 to i64, !dbg !59
  %address.local408 = trunc i64 %address.local407 to i32, !dbg !59
  %i24 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !59
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !59
  %address.local410 = and i64 %address.local409, 16256, !dbg !59
  %address.local411 = lshr i64 %address.local, 4, !dbg !59
  %address.local412 = and i64 %address.local411, 8, !dbg !59
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !59
  %i25 = ptrtoint ptr addrspace(1) %arg to i64, !dbg !59
  %.idx1399 = or disjoint i64 %i23, %address.local413, !dbg !59
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !59
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !59
  %i26 = insertelement <2 x i64> poison, i64 %i25, i64 0, !dbg !59
  %i27 = bitcast <2 x i64> %i26 to <4 x i32>, !dbg !59
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %i27, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !59
  %dma.buffer.lds1398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i24), !dbg !59
  %i28 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !59
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !59
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !59
  %address.local417 = and i64 %address.local416, 3584, !dbg !59
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !59
  %address.local419 = ptrtoint ptr addrspace(3) %address.local418 to i64, !dbg !59
  %address.local420 = trunc i64 %address.local419 to i32, !dbg !59
  %i29 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !59
  %address.local423 = lshr i64 %address.local415, 4, !dbg !59
  %address.local424 = and i64 %address.local423, 24, !dbg !59
  %address.local425 = or disjoint i64 %address.local410, %address.local424, !dbg !59
  %.idx1400 = or disjoint i64 %i23, %address.local425, !dbg !59
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !59
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !59
  %dma.buffer.lds1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i29), !dbg !59
  %i30 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !59
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !59
  fence syncscope("workgroup") release, !dbg !59
  call void @llvm.amdgcn.s.barrier(), !dbg !59
  fence syncscope("workgroup") acquire, !dbg !59
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !59
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !59
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !59
  %i31 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !59
  %i32 = or disjoint i64 %i23, 32, !dbg !59
  %i33 = or disjoint i64 %address.local413, %i32, !dbg !59
  %.tr = trunc i64 %i33 to i32, !dbg !59
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !59
  %dma.buffer.lds1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i31), !dbg !59
  %i34 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !59
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !59
  %address.local432 = ptrtoint ptr addrspace(3) %address.local431 to i64, !dbg !59
  %address.local433 = trunc i64 %address.local432 to i32, !dbg !59
  %i35 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !59
  %i36 = or disjoint i64 %address.local425, %i32, !dbg !59
  %.tr1401 = trunc i64 %i36 to i32, !dbg !59
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !59
  %dma.buffer.lds1389 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i35), !dbg !59
  %i37 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !59
  %address.local435 = lshr i32 %address.thread, 2, !dbg !61
  %address.local436 = and i32 %address.local435, 4, !dbg !61
  %address.local437 = shl nuw nsw i32 %address.thread, 1, !dbg !61
  %address.local438 = and i32 %address.local437, 384, !dbg !61
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !61
  %address.local440 = shl nuw nsw i32 %address.thread, 3, !dbg !61
  %address.local441 = and i32 %address.local440, 120, !dbg !61
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !61
  %address.local443 = shl nuw nsw i32 %address.thread, 5, !dbg !61
  %address.local444 = and i32 %address.local443, 1024, !dbg !61
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !61
  %address.local446 = zext nneg i32 %address.local445 to i64, !dbg !61
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local446, !dbg !61
  %i38 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !61
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !61
  %i39 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !61
  %i40 = fmul <4 x half> %i38, splat (half 1.275630e-01), !dbg !61
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !61
  %i41 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !61
  %i42 = fmul <4 x half> %i39, splat (half 1.275630e-01), !dbg !61
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !61
  %i43 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !61
  %i44 = fmul <4 x half> %i41, splat (half 1.275630e-01), !dbg !61
  %i45 = fmul <4 x half> %i43, splat (half 1.275630e-01), !dbg !61
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !59
  fence syncscope("workgroup") release, !dbg !59
  call void @llvm.amdgcn.s.barrier(), !dbg !59
  fence syncscope("workgroup") acquire, !dbg !59
  %i46 = or disjoint i64 %i23, 64, !dbg !59
  %i47 = or disjoint i64 %address.local413, %i46, !dbg !59
  %.tr1402 = trunc i64 %i47 to i32, !dbg !59
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !59
  %i48 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !59
  %i49 = or disjoint i64 %address.local425, %i46, !dbg !59
  %.tr1403 = trunc i64 %i49 to i32, !dbg !59
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !59
  %i50 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !59
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !61
  %i51 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !61
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !61
  %i52 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !61
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !61
  %i53 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !61
  %i54 = fmul <4 x half> %i51, splat (half 1.275630e-01), !dbg !61
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !61
  %i55 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !61
  %i56 = fmul <4 x half> %i52, splat (half 1.275630e-01), !dbg !61
  %i57 = fmul <4 x half> %i53, splat (half 1.275630e-01), !dbg !61
  %i58 = fmul <4 x half> %i55, splat (half 1.275630e-01), !dbg !61
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !59
  fence syncscope("workgroup") release, !dbg !59
  call void @llvm.amdgcn.s.barrier(), !dbg !59
  fence syncscope("workgroup") acquire, !dbg !59
  %i59 = or disjoint i64 %i23, 96, !dbg !59
  %i60 = or disjoint i64 %address.local413, %i59, !dbg !59
  %.tr1404 = trunc i64 %i60 to i32, !dbg !59
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !59
  %i61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !59
  %i62 = or disjoint i64 %address.local425, %i59, !dbg !59
  %.tr1405 = trunc i64 %i62 to i32, !dbg !59
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !59
  %i63 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !59
  %i64 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !61
  %i65 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !61
  %i66 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !61
  %i67 = fmul <4 x half> %i64, splat (half 1.275630e-01), !dbg !61
  %i68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !61
  %i69 = fmul <4 x half> %i65, splat (half 1.275630e-01), !dbg !61
  %i70 = fmul <4 x half> %i66, splat (half 1.275630e-01), !dbg !61
  %i71 = fmul <4 x half> %i68, splat (half 1.275630e-01), !dbg !61
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !59
  fence syncscope("workgroup") release, !dbg !59
  call void @llvm.amdgcn.s.barrier(), !dbg !59
  fence syncscope("workgroup") acquire, !dbg !59
  %i72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !61
  %i73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !61
  %i74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !61
  %i75 = fmul <4 x half> %i72, splat (half 1.275630e-01), !dbg !61
  %i76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !61
  %i77 = fmul <4 x half> %i73, splat (half 1.275630e-01), !dbg !61
  %i78 = fmul <4 x half> %i74, splat (half 1.275630e-01), !dbg !61
  %i79 = fmul <4 x half> %i76, splat (half 1.275630e-01), !dbg !61
  fence syncscope("workgroup") release, !dbg !62
  call void @llvm.amdgcn.s.barrier(), !dbg !62
  fence syncscope("workgroup") acquire, !dbg !62
  %i80 = shl nuw nsw i32 %kdma.tid, 2, !dbg !63
  %i81 = and i32 %i80, 3084, !dbg !63
  %i82 = and i32 %kdma.tid, 1, !dbg !63
  %i83 = or disjoint i32 %i81, %i82, !dbg !63
  %narrow = add nuw nsw i32 %i83, 1, !dbg !63
  %i84 = lshr i32 %narrow, 1, !dbg !63
  %i85 = zext nneg i32 %i84 to i64, !dbg !63
  %k0.prime.address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !64, !range !60, !frisk.register_lifetimes !56
  %k0.prime.address.local499 = zext nneg i32 %k0.prime.address.thread498 to i64, !dbg !59
  %k0.prime.address.local500 = shl nuw nsw i64 %k0.prime.address.local499, 3, !dbg !59
  %k0.prime.address.local501 = and i64 %k0.prime.address.local500, 1536, !dbg !59
  %k0.prime.address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local501, !dbg !59
  %k0.prime.address.local503 = ptrtoint ptr addrspace(3) %k0.prime.address.local502 to i64, !dbg !59
  %k0.prime.address.local504 = trunc i64 %k0.prime.address.local503 to i32, !dbg !59
  %k0.prime.i100 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local504), !dbg !64
  %k0.prime.dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.i100), !dbg !64
  %k0.prime.i96 = shl i64 %i11, 19, !dbg !64
  %k0.prime.i97 = and i64 %k0.prime.i96, 16252928, !dbg !64
  %k0.prime.i98 = shl nuw nsw i64 0, 14, !dbg !64
  %k0.prime.i99 = or disjoint i64 %k0.prime.i97, %k0.prime.i98, !dbg !64
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !64
  %k0.prime.kdma.source1506 = lshr i64 %k0.prime.kdma.source, 4, !dbg !64
  %k0.prime.kdma.source1507 = and i64 %k0.prime.kdma.source1506, 8, !dbg !64
  %k0.prime.kdma.source1508 = and i64 %k0.prime.kdma.source, 91, !dbg !64
  %k0.prime.kdma.source1509 = lshr i64 %k0.prime.kdma.source, 2, !dbg !64
  %k0.prime.kdma.source1510 = and i64 %k0.prime.kdma.source1509, 4, !dbg !64
  %k0.prime.kdma.source1511 = add nuw nsw i64 %k0.prime.kdma.source1510, %k0.prime.kdma.source, !dbg !64
  %k0.prime.kdma.source1512 = and i64 %k0.prime.kdma.source1511, 4, !dbg !64
  %k0.prime.kdma.source1513 = or disjoint i64 %k0.prime.kdma.source1508, %k0.prime.kdma.source1512, !dbg !64
  %k0.prime.kdma.source1514 = shl nuw nsw i64 %k0.prime.kdma.source1513, 7, !dbg !64
  %k0.prime.kdma.source1515 = or disjoint i64 %k0.prime.kdma.source1507, %k0.prime.kdma.source1514, !dbg !64
  %k0.prime.kdma.source1516 = or disjoint i64 %k0.prime.i99, %k0.prime.kdma.source1515, !dbg !64
  %k0.prime.kdma.source1517 = trunc nuw i64 %k0.prime.kdma.source1516 to i32, !dbg !64
  %k0.prime.kdma.source1518 = shl nuw nsw i32 %k0.prime.kdma.source1517, 1, !dbg !64
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1518, %kdma.intra, !dbg !64
  %k0.prime.i101 = ptrtoint ptr addrspace(1) %arg1 to i64, !dbg !64
  %k0.prime.i102 = insertelement <2 x i64> poison, i64 %k0.prime.i101, i64 0, !dbg !64
  %k0.prime.i103 = bitcast <2 x i64> %k0.prime.i102 to <4 x i32>, !dbg !64
  %k0.prime.dma.buffer.rsrc1373 = shufflevector <4 x i32> %k0.prime.i103, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !64
  %k0.prime.i104 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1374, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #10, !dbg !64
  %k0.prime.kdma.lds1534 = add i32 %k0.prime.dma.buffer.lds1374, 512, !dbg !64
  %k0.prime.kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !64
  %k0.prime.kdma.source1520 = lshr i64 %k0.prime.kdma.source1519, 4, !dbg !64
  %k0.prime.kdma.source1521 = and i64 %k0.prime.kdma.source1520, 8, !dbg !64
  %k0.prime.kdma.source1522 = and i64 %k0.prime.kdma.source1519, 123, !dbg !64
  %k0.prime.kdma.source1523 = lshr i64 %k0.prime.kdma.source1519, 2, !dbg !64
  %k0.prime.kdma.source1524 = and i64 %k0.prime.kdma.source1523, 4, !dbg !64
  %k0.prime.kdma.source1525 = add nuw nsw i64 %k0.prime.kdma.source1524, %k0.prime.kdma.source1519, !dbg !64
  %k0.prime.kdma.source1526 = and i64 %k0.prime.kdma.source1525, 4, !dbg !64
  %k0.prime.kdma.source1527 = or disjoint i64 %k0.prime.kdma.source1522, %k0.prime.kdma.source1526, !dbg !64
  %k0.prime.kdma.source1528 = shl nuw nsw i64 %k0.prime.kdma.source1527, 7, !dbg !64
  %k0.prime.kdma.source1529 = or disjoint i64 %k0.prime.kdma.source1521, %k0.prime.kdma.source1528, !dbg !64
  %k0.prime.kdma.source1530 = or disjoint i64 %k0.prime.i99, %k0.prime.kdma.source1529, !dbg !64
  %k0.prime.kdma.source1531 = trunc nuw i64 %k0.prime.kdma.source1530 to i32, !dbg !64
  %k0.prime.kdma.source1532 = shl nuw nsw i32 %k0.prime.kdma.source1531, 1, !dbg !64
  %k0.prime.kdma.offset1533 = or disjoint i32 %k0.prime.kdma.source1532, %kdma.intra.hi, !dbg !64
  %k0.prime.i105 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1534, i32 %k0.prime.kdma.offset1533, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #10, !dbg !64
  %k0.prime.address.local516 = or disjoint i64 %k0.prime.address.local499, 256, !dbg !59
  %k0.prime.address.local517 = shl nuw nsw i64 %k0.prime.address.local516, 3, !dbg !59
  %k0.prime.address.local518 = and i64 %k0.prime.address.local517, 3584, !dbg !59
  %k0.prime.address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local518, !dbg !59
  %k0.prime.address.local520 = ptrtoint ptr addrspace(3) %k0.prime.address.local519 to i64, !dbg !59
  %k0.prime.address.local521 = trunc i64 %k0.prime.address.local520 to i32, !dbg !59
  %k0.prime.i106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local521), !dbg !64
  %k0.prime.dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.i106), !dbg !64
  %k0.prime.kdma.source1546 = or disjoint i64 %k0.prime.kdma.source1515, 16, !dbg !64
  %k0.prime.kdma.source1547 = or disjoint i64 %k0.prime.i99, %k0.prime.kdma.source1546, !dbg !64
  %k0.prime.kdma.source1548 = trunc nuw i64 %k0.prime.kdma.source1547 to i32, !dbg !64
  %k0.prime.kdma.source1549 = shl nuw nsw i32 %k0.prime.kdma.source1548, 1, !dbg !64
  %k0.prime.kdma.offset1550 = or disjoint i32 %k0.prime.kdma.source1549, %kdma.intra, !dbg !64
  %k0.prime.i107 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1371, i32 %k0.prime.kdma.offset1550, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #10, !dbg !64
  %k0.prime.kdma.lds1568 = add i32 %k0.prime.dma.buffer.lds1371, 512, !dbg !64
  %k0.prime.kdma.source1563 = or disjoint i64 %k0.prime.kdma.source1529, 16, !dbg !64
  %k0.prime.kdma.source1564 = or disjoint i64 %k0.prime.i99, %k0.prime.kdma.source1563, !dbg !64
  %k0.prime.kdma.source1565 = trunc nuw i64 %k0.prime.kdma.source1564 to i32, !dbg !64
  %k0.prime.kdma.source1566 = shl nuw nsw i32 %k0.prime.kdma.source1565, 1, !dbg !64
  %k0.prime.kdma.offset1567 = or disjoint i32 %k0.prime.kdma.source1566, %kdma.intra.hi, !dbg !64
  %k0.prime.i108 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1568, i32 %k0.prime.kdma.offset1567, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #10, !dbg !64
  br label %bb86, !dbg !62

bb86:                                             ; preds = %k0.join, %bb
  %i87 = phi i64 [ %i1020, %k0.join ], [ 0, %bb ], !dbg !62
  %i88 = phi [2 x <8 x float>] [ %i1005, %k0.join ], [ zeroinitializer, %bb ], !dbg !62
  %i89 = phi [2 x <1 x float>] [ %i699, %k0.join ], [ zeroinitializer, %bb ], !dbg !62
  %i90 = phi [2 x <1 x float>] [ %lazy.N, %k0.join ], [ zeroinitializer, %bb ], !dbg !62
  %i91 = phi [2 x <8 x float>] [ %i1011, %k0.join ], [ zeroinitializer, %bb ], !dbg !62
  %i92 = phi [2 x <8 x float>] [ %i1015, %k0.join ], [ zeroinitializer, %bb ], !dbg !62
  %i93 = phi [2 x <8 x float>] [ %i1019, %k0.join ], [ zeroinitializer, %bb ], !dbg !62
  %i94 = icmp samesign ult i64 %i87, 32, !dbg !62
  br i1 %i94, label %bb95, label %bb1021, !dbg !62

bb95:                                             ; preds = %bb86
  %i96 = shl i64 %i11, 19, !dbg !64
  %i97 = and i64 %i96, 16252928, !dbg !64
  %i98 = shl nuw nsw i64 %i87, 14, !dbg !64
  %i99 = or disjoint i64 %i97, %i98, !dbg !64
  %address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !64, !range !60, !frisk.register_lifetimes !56
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !59
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !59
  %address.local501 = and i64 %address.local500, 1536, !dbg !59
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !59
  %address.local503 = ptrtoint ptr addrspace(3) %address.local502 to i64, !dbg !59
  %address.local504 = trunc i64 %address.local503 to i32, !dbg !59
  %i100 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !64
  %i101 = ptrtoint ptr addrspace(1) %arg1 to i64, !dbg !64
  %i102 = insertelement <2 x i64> poison, i64 %i101, i64 0, !dbg !64
  %i103 = bitcast <2 x i64> %i102 to <4 x i32>, !dbg !64
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %i103, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !64
  %dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i100), !dbg !64
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !64
  %kdma.source1506 = lshr i64 %kdma.source, 4, !dbg !64
  %kdma.source1507 = and i64 %kdma.source1506, 8, !dbg !64
  %kdma.source1508 = and i64 %kdma.source, 91, !dbg !64
  %kdma.source1509 = lshr i64 %kdma.source, 2, !dbg !64
  %kdma.source1510 = and i64 %kdma.source1509, 4, !dbg !64
  %kdma.source1511 = add nuw nsw i64 %kdma.source1510, %kdma.source, !dbg !64
  %kdma.source1512 = and i64 %kdma.source1511, 4, !dbg !64
  %kdma.source1513 = or disjoint i64 %kdma.source1508, %kdma.source1512, !dbg !64
  %kdma.source1514 = shl nuw nsw i64 %kdma.source1513, 7, !dbg !64
  %kdma.source1515 = or disjoint i64 %kdma.source1507, %kdma.source1514, !dbg !64
  %kdma.source1516 = or disjoint i64 %i99, %kdma.source1515, !dbg !64
  %kdma.source1517 = trunc nuw i64 %kdma.source1516 to i32, !dbg !64
  %kdma.source1518 = shl nuw nsw i32 %kdma.source1517, 1, !dbg !64
  %kdma.offset = or disjoint i32 %kdma.source1518, %kdma.intra, !dbg !64
  %kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !64
  %kdma.source1520 = lshr i64 %kdma.source1519, 4, !dbg !64
  %kdma.source1521 = and i64 %kdma.source1520, 8, !dbg !64
  %kdma.source1522 = and i64 %kdma.source1519, 123, !dbg !64
  %kdma.source1523 = lshr i64 %kdma.source1519, 2, !dbg !64
  %kdma.source1524 = and i64 %kdma.source1523, 4, !dbg !64
  %kdma.source1525 = add nuw nsw i64 %kdma.source1524, %kdma.source1519, !dbg !64
  %kdma.source1526 = and i64 %kdma.source1525, 4, !dbg !64
  %kdma.source1527 = or disjoint i64 %kdma.source1522, %kdma.source1526, !dbg !64
  %kdma.source1528 = shl nuw nsw i64 %kdma.source1527, 7, !dbg !64
  %kdma.source1529 = or disjoint i64 %kdma.source1521, %kdma.source1528, !dbg !64
  %kdma.source1530 = or disjoint i64 %i99, %kdma.source1529, !dbg !64
  %kdma.source1531 = trunc nuw i64 %kdma.source1530 to i32, !dbg !64
  %kdma.source1532 = shl nuw nsw i32 %kdma.source1531, 1, !dbg !64
  %kdma.offset1533 = or disjoint i32 %kdma.source1532, %kdma.intra.hi, !dbg !64
  %kdma.lds1534 = add i32 %dma.buffer.lds1374, 512, !dbg !64
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !59
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !59
  %address.local518 = and i64 %address.local517, 3584, !dbg !59
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !59
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !59
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !59
  %i106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !64
  %dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i106), !dbg !64
  %kdma.source1546 = or disjoint i64 %kdma.source1515, 16, !dbg !64
  %kdma.source1547 = or disjoint i64 %i99, %kdma.source1546, !dbg !64
  %kdma.source1548 = trunc nuw i64 %kdma.source1547 to i32, !dbg !64
  %kdma.source1549 = shl nuw nsw i32 %kdma.source1548, 1, !dbg !64
  %kdma.offset1550 = or disjoint i32 %kdma.source1549, %kdma.intra, !dbg !64
  %kdma.source1563 = or disjoint i64 %kdma.source1529, 16, !dbg !64
  %kdma.source1564 = or disjoint i64 %i99, %kdma.source1563, !dbg !64
  %kdma.source1565 = trunc nuw i64 %kdma.source1564 to i32, !dbg !64
  %kdma.source1566 = shl nuw nsw i32 %kdma.source1565, 1, !dbg !64
  %kdma.offset1567 = or disjoint i32 %kdma.source1566, %kdma.intra.hi, !dbg !64
  %kdma.lds1568 = add i32 %dma.buffer.lds1371, 512, !dbg !64
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !64
  fence syncscope("workgroup") release, !dbg !64
  call void @llvm.amdgcn.s.barrier(), !dbg !64
  fence syncscope("workgroup") acquire, !dbg !64
  %i109 = or disjoint i64 %i99, 32, !dbg !64
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !59
  %address.local534 = ptrtoint ptr addrspace(3) %address.local533 to i64, !dbg !59
  %address.local535 = trunc i64 %address.local534 to i32, !dbg !59
  %i110 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !64
  %dma.buffer.lds1368 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i110), !dbg !64
  %kdma.source1580 = or disjoint i64 %i109, %kdma.source1515, !dbg !64
  %kdma.source1581 = trunc nuw i64 %kdma.source1580 to i32, !dbg !64
  %kdma.source1582 = shl nuw nsw i32 %kdma.source1581, 1, !dbg !64
  %kdma.offset1583 = or disjoint i32 %kdma.source1582, %kdma.intra, !dbg !64
  %i111 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1583, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %kdma.source1596 = or disjoint i64 %i109, %kdma.source1529, !dbg !64
  %kdma.source1597 = trunc nuw i64 %kdma.source1596 to i32, !dbg !64
  %kdma.source1598 = shl nuw nsw i32 %kdma.source1597, 1, !dbg !64
  %kdma.offset1599 = or disjoint i32 %kdma.source1598, %kdma.intra.hi, !dbg !64
  %kdma.lds1600 = add i32 %dma.buffer.lds1368, 512, !dbg !64
  %i112 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1599, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !59
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !59
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !59
  %i113 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !64
  %dma.buffer.lds1365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i113), !dbg !64
  %kdma.source1613 = or disjoint i64 %i109, %kdma.source1546, !dbg !64
  %kdma.source1614 = trunc nuw i64 %kdma.source1613 to i32, !dbg !64
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !64
  %kdma.offset1616 = or disjoint i32 %kdma.source1615, %kdma.intra, !dbg !64
  %i114 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1616, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %kdma.source1630 = or disjoint i64 %i109, %kdma.source1563, !dbg !64
  %kdma.source1631 = trunc nuw i64 %kdma.source1630 to i32, !dbg !64
  %kdma.source1632 = shl nuw nsw i32 %kdma.source1631, 1, !dbg !64
  %kdma.offset1633 = or disjoint i32 %kdma.source1632, %kdma.intra.hi, !dbg !64
  %kdma.lds1634 = add i32 %dma.buffer.lds1365, 512, !dbg !64
  %i115 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %address.local542 = lshr i64 %address.local499, 2, !dbg !63
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !63
  %address.local546 = and i64 %address.local545, 24, !dbg !63
  %address.local548 = shl i64 %address.local499, 62, !dbg !63
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !63
  %address.local550 = and i64 %address.local549, -64, !dbg !63
  %address.local553 = and i64 %address.local542, 12, !dbg !61
  %address.local555 = and i64 %address.local499, 63, !dbg !61
  %address.local556 = lshr i64 %address.local555, 5, !dbg !61
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !61
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !63
  %i116 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !63
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !63
  %i117 = getelementptr i8, ptr addrspace(3) %i116, i64 %.idx1410, !dbg !63
  %i118 = getelementptr [2 x i8], ptr addrspace(3) %i117, i64 %address.local546, !dbg !63
  %i119 = getelementptr [2 x i8], ptr addrspace(3) %i118, i64 %address.local550, !dbg !63
  %i120 = getelementptr [2 x i8], ptr addrspace(3) %i119, i64 %address.local553, !dbg !63
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %i120, i64 %address.local557, !dbg !63
  %i121 = add nsw i64 %.idx, %.idx1410, !dbg !63
  %.idx1893 = shl nuw nsw i64 %address.local546, 1, !dbg !63
  %i122 = or disjoint i64 %i121, %.idx1893, !dbg !63
  %i123 = or disjoint i64 %address.local550, %address.local553, !dbg !63
  %i124 = shl nsw i64 %i123, 1, !dbg !63
  %i125 = add nsw i64 %i124, %i122, !dbg !63
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032, !dbg !63
  %i126 = add nsw i64 %i125, %address.local559.idx, !dbg !63
  %i127 = lshr i64 %i126, 6, !dbg !63
  %i128 = and i64 %i127, 8, !dbg !63
  %i129 = xor i64 %i126, %i128, !dbg !63
  %i130 = sub i64 %i129, %i126, !dbg !63
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %i130, !dbg !63
  %i131 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !63
  %lazy.n0 = extractvalue [2 x <1 x float>] %i90, 0, !dbg !92
  %lazy.seed0 = shufflevector <1 x float> %lazy.n0, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !92
  %lazy.n1 = extractvalue [2 x <1 x float>] %i90, 1, !dbg !92
  %lazy.seed1 = shufflevector <1 x float> %lazy.n1, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !92
  %i132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i131, <4 x float> %lazy.seed0) #9, !dbg !63
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !63
  %i133 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !63
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !63
  %address.local571 = and i32 %address.local570, 24, !dbg !63
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !63
  %address.local573 = and i32 %address.local572, 768, !dbg !63
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !63
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !63
  %address.local576 = and i32 %address.local575, 96, !dbg !63
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !63
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !63
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %i85) #9, !dbg !63, !frisk.register_lifetimes !56
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !63
  %address.local580 = and i32 %address.local579, 1, !dbg !63
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !63
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !63
  %address.local583 = shl i64 %address.local582, 6, !dbg !63
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !63
  %i134 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !63
  %i135 = getelementptr [2 x i8], ptr addrspace(3) %i134, i64 %address.local553, !dbg !63
  %i136 = getelementptr [2 x i8], ptr addrspace(3) %i135, i64 %address.local557, !dbg !63
  %address.local588 = getelementptr i8, ptr addrspace(3) %i136, i64 64, !dbg !63
  %.idx18961899 = add i64 %address.local584, %address.local553, !dbg !63
  %i137 = shl i64 %.idx18961899, 1, !dbg !63
  %i138 = add i64 %i137, %address.local559.idx, !dbg !63
  %i139 = add i64 %i138, 64, !dbg !63
  %i140 = lshr i64 %i139, 6, !dbg !63
  %i141 = and i64 %i140, 8, !dbg !63
  %i142 = xor i64 %i139, %i141, !dbg !63
  %i143 = sub i64 %i142, %i139, !dbg !63
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %i143, !dbg !63
  %i144 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !63, !frisk.k_lds_pair !65
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !63
  %i145 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !63, !frisk.k_lds_pair !65
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !63
  %i146 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !63, !frisk.k_lds_pair !66
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !63
  %i147 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !63, !frisk.k_lds_pair !66
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !63
  %i148 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !63, !frisk.k_lds_pair !67
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !63
  %i149 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !63, !frisk.k_lds_pair !67
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !63
  %i150 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !63, !frisk.k_lds_pair !68
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !63
  %i151 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !63, !frisk.k_lds_pair !68
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i133, <4 x float> %i132) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !63
  %i153 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !63, !frisk.k_lds_pair !69
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !63
  %i154 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !63, !frisk.k_lds_pair !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i144, <4 x float> %lazy.seed0) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !63
  %i156 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !63, !frisk.k_lds_pair !70
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !63
  %i157 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !63, !frisk.k_lds_pair !70
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i146, <4 x float> %i155) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !63
  %i159 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !63, !frisk.k_lds_pair !71
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !63
  %i160 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !63, !frisk.k_lds_pair !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i148, <4 x float> %lazy.seed0) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i150, <4 x float> %i161) #9, !dbg !63
  %i163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i145, <4 x float> %lazy.seed0) #9, !dbg !63
  %i164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i147, <4 x float> %i163) #9, !dbg !63
  %i165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i149, <4 x float> %lazy.seed0) #9, !dbg !63
  %i166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i151, <4 x float> %i165) #9, !dbg !63
  %i167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i153, <4 x float> %lazy.seed0) #9, !dbg !63
  %i168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i156, <4 x float> %i167) #9, !dbg !63
  %i169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i159, <4 x float> %lazy.seed0) #9, !dbg !63
  %i170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i160, <4 x float> %i169) #9, !dbg !63
  %i171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i154, <4 x float> %lazy.seed0) #9, !dbg !63
  %i172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i157, <4 x float> %i171) #9, !dbg !63
  %i173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i131, <4 x float> %lazy.seed1) #9, !dbg !63
  %i174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i133, <4 x float> %i173) #9, !dbg !63
  %i175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i144, <4 x float> %lazy.seed1) #9, !dbg !63
  %i176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i146, <4 x float> %i175) #9, !dbg !63
  %i177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i148, <4 x float> %lazy.seed1) #9, !dbg !63
  %i178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i150, <4 x float> %i177) #9, !dbg !63
  %i179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i145, <4 x float> %lazy.seed1) #9, !dbg !63
  %i180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i147, <4 x float> %i179) #9, !dbg !63
  %i181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i149, <4 x float> %lazy.seed1) #9, !dbg !63
  %i182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i151, <4 x float> %i181) #9, !dbg !63
  %i183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i153, <4 x float> %lazy.seed1) #9, !dbg !63
  %i184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i156, <4 x float> %i183) #9, !dbg !63
  %i185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i159, <4 x float> %lazy.seed1) #9, !dbg !63
  %i186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i160, <4 x float> %i185) #9, !dbg !63
  %i187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i154, <4 x float> %lazy.seed1) #9, !dbg !63
  %i188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i157, <4 x float> %i187) #9, !dbg !63
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !64
  fence syncscope("workgroup") release, !dbg !64
  call void @llvm.amdgcn.s.barrier(), !dbg !64
  fence syncscope("workgroup") acquire, !dbg !64
  %i189 = or disjoint i64 %i99, 64, !dbg !64
  %kdma.source1646 = or disjoint i64 %i189, %kdma.source1515, !dbg !64
  %kdma.source1647 = trunc nuw i64 %kdma.source1646 to i32, !dbg !64
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !64
  %kdma.offset1649 = or disjoint i32 %kdma.source1648, %kdma.intra, !dbg !64
  %i190 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset1649, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %kdma.source1662 = or disjoint i64 %i189, %kdma.source1529, !dbg !64
  %kdma.source1663 = trunc nuw i64 %kdma.source1662 to i32, !dbg !64
  %kdma.source1664 = shl nuw nsw i32 %kdma.source1663, 1, !dbg !64
  %kdma.offset1665 = or disjoint i32 %kdma.source1664, %kdma.intra.hi, !dbg !64
  %i191 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1665, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %kdma.source1679 = or disjoint i64 %i189, %kdma.source1546, !dbg !64
  %kdma.source1680 = trunc nuw i64 %kdma.source1679 to i32, !dbg !64
  %kdma.source1681 = shl nuw nsw i32 %kdma.source1680, 1, !dbg !64
  %kdma.offset1682 = or disjoint i32 %kdma.source1681, %kdma.intra, !dbg !64
  %i192 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1682, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %kdma.source1696 = or disjoint i64 %i189, %kdma.source1563, !dbg !64
  %kdma.source1697 = trunc nuw i64 %kdma.source1696 to i32, !dbg !64
  %kdma.source1698 = shl nuw nsw i32 %kdma.source1697, 1, !dbg !64
  %kdma.offset1699 = or disjoint i32 %kdma.source1698, %kdma.intra.hi, !dbg !64
  %i193 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1699, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !63
  %i194 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !63, !frisk.k_lds_pair !72
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !63
  %i195 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !63, !frisk.k_lds_pair !72
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !63
  %i196 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !63, !frisk.k_lds_pair !73
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !63
  %i197 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !63, !frisk.k_lds_pair !73
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !63
  %i198 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !63, !frisk.k_lds_pair !74
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !63
  %i199 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !63, !frisk.k_lds_pair !74
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !63
  %i200 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !63, !frisk.k_lds_pair !75
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !63
  %i201 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !63, !frisk.k_lds_pair !75
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !63
  %i202 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !63, !frisk.k_lds_pair !76
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !63
  %i203 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !63, !frisk.k_lds_pair !76
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i194, <4 x float> %i152) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !63
  %i205 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !63, !frisk.k_lds_pair !77
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !63
  %i206 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !63, !frisk.k_lds_pair !77
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i196, <4 x float> %i204) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !63
  %i208 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !63, !frisk.k_lds_pair !78
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !63
  %i209 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !63, !frisk.k_lds_pair !78
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i198, <4 x float> %i158) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !63
  %i211 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !63, !frisk.k_lds_pair !79
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !63
  %i212 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !63, !frisk.k_lds_pair !79
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i200, <4 x float> %i210) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i195, <4 x float> %i162) #9, !dbg !63
  %i215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i197, <4 x float> %i214) #9, !dbg !63
  %i216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i199, <4 x float> %i164) #9, !dbg !63
  %i217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i201, <4 x float> %i216) #9, !dbg !63
  %i218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i202, <4 x float> %i166) #9, !dbg !63
  %i219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i205, <4 x float> %i218) #9, !dbg !63
  %i220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i208, <4 x float> %i168) #9, !dbg !63
  %i221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i211, <4 x float> %i220) #9, !dbg !63
  %i222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i203, <4 x float> %i170) #9, !dbg !63
  %i223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i206, <4 x float> %i222) #9, !dbg !63
  %i224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i209, <4 x float> %i172) #9, !dbg !63
  %i225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i212, <4 x float> %i224) #9, !dbg !63
  %i226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i194, <4 x float> %i174) #9, !dbg !63
  %i227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i196, <4 x float> %i226) #9, !dbg !63
  %i228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i198, <4 x float> %i176) #9, !dbg !63
  %i229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i200, <4 x float> %i228) #9, !dbg !63
  %i230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i195, <4 x float> %i178) #9, !dbg !63
  %i231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i197, <4 x float> %i230) #9, !dbg !63
  %i232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i199, <4 x float> %i180) #9, !dbg !63
  %i233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i201, <4 x float> %i232) #9, !dbg !63
  %i234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i202, <4 x float> %i182) #9, !dbg !63
  %i235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i205, <4 x float> %i234) #9, !dbg !63
  %i236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i208, <4 x float> %i184) #9, !dbg !63
  %i237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i211, <4 x float> %i236) #9, !dbg !63
  %i238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i203, <4 x float> %i186) #9, !dbg !63
  %i239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i206, <4 x float> %i238) #9, !dbg !63
  %i240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i209, <4 x float> %i188) #9, !dbg !63
  %i241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i212, <4 x float> %i240) #9, !dbg !63
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !64
  fence syncscope("workgroup") release, !dbg !64
  call void @llvm.amdgcn.s.barrier(), !dbg !64
  fence syncscope("workgroup") acquire, !dbg !64
  %i242 = or disjoint i64 %i99, 96, !dbg !64
  %kdma.source1712 = or disjoint i64 %i242, %kdma.source1515, !dbg !64
  %kdma.source1713 = trunc nuw i64 %kdma.source1712 to i32, !dbg !64
  %kdma.source1714 = shl nuw nsw i32 %kdma.source1713, 1, !dbg !64
  %kdma.offset1715 = or disjoint i32 %kdma.source1714, %kdma.intra, !dbg !64
  %i243 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1715, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %kdma.source1728 = or disjoint i64 %i242, %kdma.source1529, !dbg !64
  %kdma.source1729 = trunc nuw i64 %kdma.source1728 to i32, !dbg !64
  %kdma.source1730 = shl nuw nsw i32 %kdma.source1729, 1, !dbg !64
  %kdma.offset1731 = or disjoint i32 %kdma.source1730, %kdma.intra.hi, !dbg !64
  %i244 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1731, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %kdma.source1745 = or disjoint i64 %i242, %kdma.source1546, !dbg !64
  %kdma.source1746 = trunc nuw i64 %kdma.source1745 to i32, !dbg !64
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !64
  %kdma.offset1748 = or disjoint i32 %kdma.source1747, %kdma.intra, !dbg !64
  %i245 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1748, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %kdma.source1762 = or disjoint i64 %i242, %kdma.source1563, !dbg !64
  %kdma.source1763 = trunc nuw i64 %kdma.source1762 to i32, !dbg !64
  %kdma.source1764 = shl nuw nsw i32 %kdma.source1763, 1, !dbg !64
  %kdma.offset1765 = or disjoint i32 %kdma.source1764, %kdma.intra.hi, !dbg !64
  %i246 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1765, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %i247 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !63, !frisk.k_lds_pair !80
  %i248 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !63, !frisk.k_lds_pair !80
  %i249 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !63, !frisk.k_lds_pair !81
  %i250 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !63, !frisk.k_lds_pair !81
  %i251 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !63, !frisk.k_lds_pair !82
  %i252 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !63, !frisk.k_lds_pair !82
  %i253 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !63, !frisk.k_lds_pair !83
  %i254 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !63, !frisk.k_lds_pair !83
  %i255 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !63, !frisk.k_lds_pair !84
  %i256 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !63, !frisk.k_lds_pair !84
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i247, <4 x float> %i207) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i258 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !63, !frisk.k_lds_pair !85
  %i259 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !63, !frisk.k_lds_pair !85
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i249, <4 x float> %i257) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i261 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !63, !frisk.k_lds_pair !86
  %i262 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !63, !frisk.k_lds_pair !86
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i251, <4 x float> %i213) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i264 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !63, !frisk.k_lds_pair !87
  %i265 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !63, !frisk.k_lds_pair !87
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i253, <4 x float> %i263) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_lds_prefetch !56
  %i267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i248, <4 x float> %i215) #9, !dbg !63
  %i268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i250, <4 x float> %i267) #9, !dbg !63
  %i269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i252, <4 x float> %i217) #9, !dbg !63
  %i270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i254, <4 x float> %i269) #9, !dbg !63
  %i271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i255, <4 x float> %i219) #9, !dbg !63
  %i272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i258, <4 x float> %i271) #9, !dbg !63
  %i273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i261, <4 x float> %i221) #9, !dbg !63
  %i274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i264, <4 x float> %i273) #9, !dbg !63
  %i275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i256, <4 x float> %i223) #9, !dbg !63
  %i276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i259, <4 x float> %i275) #9, !dbg !63
  %i277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i262, <4 x float> %i225) #9, !dbg !63
  %i278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i265, <4 x float> %i277) #9, !dbg !63
  %i279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i247, <4 x float> %i227) #9, !dbg !63
  %i280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i249, <4 x float> %i279) #9, !dbg !63
  %i281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i251, <4 x float> %i229) #9, !dbg !63
  %i282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i253, <4 x float> %i281) #9, !dbg !63
  %i283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i248, <4 x float> %i231) #9, !dbg !63
  %i284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i250, <4 x float> %i283) #9, !dbg !63
  %i285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i252, <4 x float> %i233) #9, !dbg !63
  %i286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i254, <4 x float> %i285) #9, !dbg !63
  %i287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i255, <4 x float> %i235) #9, !dbg !63
  %i288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i258, <4 x float> %i287) #9, !dbg !63
  %i289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i261, <4 x float> %i237) #9, !dbg !63
  %i290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i264, <4 x float> %i289) #9, !dbg !63
  %i291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i256, <4 x float> %i239) #9, !dbg !63
  %i292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i259, <4 x float> %i291) #9, !dbg !63
  %i293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i262, <4 x float> %i241) #9, !dbg !63
  %i294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i265, <4 x float> %i293) #9, !dbg !63
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !64
  fence syncscope("workgroup") release, !dbg !64
  call void @llvm.amdgcn.s.barrier(), !dbg !64
  fence syncscope("workgroup") acquire, !dbg !64
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !88
  %address.local918 = and i64 %address.local917, 8064, !dbg !88
  %address.local920 = and i64 %address.local500, 24, !dbg !88
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !88
  %i295 = ptrtoint ptr addrspace(1) %arg2 to i64, !dbg !88
  %.idx14191420 = or disjoint i64 %i99, %address.local921, !dbg !88
  %.idx14191420.tr = trunc nuw i64 %.idx14191420 to i32, !dbg !88
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !88
  %i296 = insertelement <2 x i64> poison, i64 %i295, i64 0, !dbg !88
  %i297 = bitcast <2 x i64> %i296 to <4 x i32>, !dbg !88
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %i297, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !88
  %i298 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !88
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !88
  %address.local924 = and i64 %address.local923, 16256, !dbg !88
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !88
  %.idx14211422 = or disjoint i64 %i99, %address.local927, !dbg !88
  %.idx14211422.tr = trunc nuw i64 %.idx14211422 to i32, !dbg !88
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !88
  %i299 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !88
  %i300 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !63, !frisk.k_tail_pair !65
  %i312 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !63, !frisk.k_tail_pair !65
  %i302 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !63, !frisk.k_tail_pair !66
  %i319 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !63, !frisk.k_tail_pair !66
  %i303 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !63, !frisk.k_tail_pair !67
  %i320 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !63, !frisk.k_tail_pair !67
  %i311 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !63, !frisk.k_tail_pair !68
  %i327 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !63, !frisk.k_tail_pair !68
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_tail_prefetch !56
  %i301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i300, <4 x float> %i260) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_tail_prefetch !56
  %i304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i302, <4 x float> %i301) #9, !dbg !63
  %i305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i303, <4 x float> %i266) #9, !dbg !63
  %i306 = extractelement <4 x float> %i304, i64 0, !dbg !89
  %i307 = call float @llvm.maxnum.f32(float %i306, float -inf), !dbg !90, !frisk.local_reduction !56
  %i308 = extractelement <4 x float> %i304, i64 1, !dbg !89
  %i309 = extractelement <4 x float> %i304, i64 2, !dbg !89
  %i310 = extractelement <4 x float> %i304, i64 3, !dbg !89
  %i313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i311, <4 x float> %i305) #9, !dbg !63
  %i328 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !63, !frisk.k_tail_pair !69
  %i344 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !63, !frisk.k_tail_pair !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_tail_prefetch !56
  %i314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i312, <4 x float> %i268) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !89, !frisk.k_tail_prefetch !56
  %i315 = extractelement <4 x float> %i313, i64 0, !dbg !89
  %local.reduce151 = call float @llvm.maxnum.f32(float %i307, float %i315), !dbg !90, !frisk.local_reduction !56
  %i316 = extractelement <4 x float> %i313, i64 1, !dbg !89
  %local.reduce152 = call float @llvm.maxnum.f32(float %i308, float %i316), !dbg !90, !frisk.local_reduction !56
  %i317 = extractelement <4 x float> %i313, i64 2, !dbg !89
  %local.reduce153 = call float @llvm.maxnum.f32(float %i309, float %i317), !dbg !90, !frisk.local_reduction !56
  %i318 = extractelement <4 x float> %i313, i64 3, !dbg !89
  %local.reduce154 = call float @llvm.maxnum.f32(float %i310, float %i318), !dbg !90, !frisk.local_reduction !56
  %i335 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !63, !frisk.k_tail_pair !70
  %i351 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !63, !frisk.k_tail_pair !70
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_tail_prefetch !56
  %i321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i319, <4 x float> %i314) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_tail_prefetch !56
  %i336 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !63, !frisk.k_tail_pair !71
  %i352 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !63, !frisk.k_tail_pair !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_tail_prefetch !56
  %i322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i320, <4 x float> %i270) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !89, !frisk.k_tail_prefetch !56
  %i323 = extractelement <4 x float> %i321, i64 0, !dbg !89
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %i323), !dbg !90, !frisk.local_reduction !56
  %i324 = extractelement <4 x float> %i321, i64 1, !dbg !89
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %i324), !dbg !90, !frisk.local_reduction !56
  %i325 = extractelement <4 x float> %i321, i64 2, !dbg !89
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %i325), !dbg !90, !frisk.local_reduction !56
  %i326 = extractelement <4 x float> %i321, i64 3, !dbg !89
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %i326), !dbg !90, !frisk.local_reduction !56
  %i343 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !63, !frisk.k_tail_pair !72
  %i359 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !63, !frisk.k_tail_pair !72
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_tail_prefetch !56
  %i329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i327, <4 x float> %i322) #9, !dbg !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !63, !frisk.k_tail_prefetch !56
  %i330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i328, <4 x float> %i272) #9, !dbg !63
  %i331 = extractelement <4 x float> %i329, i64 0, !dbg !89
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %i331), !dbg !90, !frisk.local_reduction !56
  %i332 = extractelement <4 x float> %i329, i64 1, !dbg !89
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %i332), !dbg !90, !frisk.local_reduction !56
  %i333 = extractelement <4 x float> %i329, i64 2, !dbg !89
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %i333), !dbg !90, !frisk.local_reduction !56
  %i334 = extractelement <4 x float> %i329, i64 3, !dbg !89
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %i334), !dbg !90, !frisk.local_reduction !56
  %i337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i335, <4 x float> %i330) #9, !dbg !63
  %i338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i336, <4 x float> %i274) #9, !dbg !63
  %i339 = extractelement <4 x float> %i337, i64 0, !dbg !89
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %i339), !dbg !90, !frisk.local_reduction !56
  %i340 = extractelement <4 x float> %i337, i64 1, !dbg !89
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %i340), !dbg !90, !frisk.local_reduction !56
  %i341 = extractelement <4 x float> %i337, i64 2, !dbg !89
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %i341), !dbg !90, !frisk.local_reduction !56
  %i342 = extractelement <4 x float> %i337, i64 3, !dbg !89
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %i342), !dbg !90, !frisk.local_reduction !56
  %i345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i343, <4 x float> %i338) #9, !dbg !63
  %i346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i344, <4 x float> %i276) #9, !dbg !63
  %i347 = extractelement <4 x float> %i345, i64 0, !dbg !89
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %i347), !dbg !90, !frisk.local_reduction !56
  %i348 = extractelement <4 x float> %i345, i64 1, !dbg !89
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %i348), !dbg !90, !frisk.local_reduction !56
  %i349 = extractelement <4 x float> %i345, i64 2, !dbg !89
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %i349), !dbg !90, !frisk.local_reduction !56
  %i350 = extractelement <4 x float> %i345, i64 3, !dbg !89
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %i350), !dbg !90, !frisk.local_reduction !56
  %i353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i351, <4 x float> %i346) #9, !dbg !63
  %i354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i352, <4 x float> %i278) #9, !dbg !63
  %i355 = extractelement <4 x float> %i353, i64 0, !dbg !89
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %i355), !dbg !90, !frisk.local_reduction !56
  %i356 = extractelement <4 x float> %i353, i64 1, !dbg !89
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %i356), !dbg !90, !frisk.local_reduction !56
  %i357 = extractelement <4 x float> %i353, i64 2, !dbg !89
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %i357), !dbg !90, !frisk.local_reduction !56
  %i358 = extractelement <4 x float> %i353, i64 3, !dbg !89
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %i358), !dbg !90, !frisk.local_reduction !56
  %i360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i359, <4 x float> %i354) #9, !dbg !63
  %i361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i300, <4 x float> %i280) #9, !dbg !63
  %i362 = extractelement <4 x float> %i360, i64 0, !dbg !89
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %i362), !dbg !90, !frisk.local_reduction !56
  %i363 = extractelement <4 x float> %i360, i64 1, !dbg !89
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %i363), !dbg !90, !frisk.local_reduction !56
  %i364 = extractelement <4 x float> %i360, i64 2, !dbg !89
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %i364), !dbg !90, !frisk.local_reduction !56
  %i365 = extractelement <4 x float> %i360, i64 3, !dbg !89
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %i365), !dbg !90, !frisk.local_reduction !56
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !90, !frisk.local_reduction !56
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !90, !frisk.local_reduction !56
  %i366 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !90, !frisk.local_reduction !56
  %i367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i302, <4 x float> %i361) #9, !dbg !63
  %i368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i303, <4 x float> %i282) #9, !dbg !63
  %i369 = extractelement <4 x float> %i367, i64 0, !dbg !89
  %i370 = extractelement <4 x float> %i367, i64 1, !dbg !89
  %i371 = extractelement <4 x float> %i367, i64 2, !dbg !89
  %i372 = extractelement <4 x float> %i367, i64 3, !dbg !89
  %i373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i311, <4 x float> %i368) #9, !dbg !63
  %i374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i312, <4 x float> %i284) #9, !dbg !63
  %i375 = extractelement <4 x float> %i373, i64 0, !dbg !89
  %i376 = extractelement <4 x float> %i373, i64 1, !dbg !89
  %i377 = extractelement <4 x float> %i373, i64 2, !dbg !89
  %i378 = extractelement <4 x float> %i373, i64 3, !dbg !89
  %i379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i319, <4 x float> %i374) #9, !dbg !63
  %i380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i320, <4 x float> %i286) #9, !dbg !63
  %i381 = extractelement <4 x float> %i379, i64 0, !dbg !89
  %i382 = extractelement <4 x float> %i379, i64 1, !dbg !89
  %i383 = extractelement <4 x float> %i379, i64 2, !dbg !89
  %i384 = extractelement <4 x float> %i379, i64 3, !dbg !89
  %i385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i327, <4 x float> %i380) #9, !dbg !63
  %i386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i328, <4 x float> %i288) #9, !dbg !63
  %i387 = extractelement <4 x float> %i385, i64 0, !dbg !89
  %i388 = extractelement <4 x float> %i385, i64 1, !dbg !89
  %i389 = extractelement <4 x float> %i385, i64 2, !dbg !89
  %i390 = extractelement <4 x float> %i385, i64 3, !dbg !89
  %i391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i335, <4 x float> %i386) #9, !dbg !63
  %i392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i336, <4 x float> %i290) #9, !dbg !63
  %i393 = extractelement <4 x float> %i391, i64 0, !dbg !89
  %i394 = extractelement <4 x float> %i391, i64 1, !dbg !89
  %i395 = extractelement <4 x float> %i391, i64 2, !dbg !89
  %i396 = extractelement <4 x float> %i391, i64 3, !dbg !89
  %i397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i343, <4 x float> %i392) #9, !dbg !63
  %i398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i344, <4 x float> %i292) #9, !dbg !63
  %i399 = extractelement <4 x float> %i397, i64 0, !dbg !89
  %i400 = extractelement <4 x float> %i397, i64 1, !dbg !89
  %i401 = extractelement <4 x float> %i397, i64 2, !dbg !89
  %i402 = extractelement <4 x float> %i397, i64 3, !dbg !89
  %i403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i351, <4 x float> %i398) #9, !dbg !63
  %i404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i352, <4 x float> %i294) #9, !dbg !63
  %i405 = extractelement <4 x float> %i403, i64 0, !dbg !89
  %i406 = extractelement <4 x float> %i403, i64 1, !dbg !89
  %i407 = extractelement <4 x float> %i403, i64 2, !dbg !89
  %i408 = extractelement <4 x float> %i403, i64 3, !dbg !89
  %i409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i359, <4 x float> %i404) #9, !dbg !63
  %i410 = extractelement <4 x float> %i409, i64 0, !dbg !89
  %i411 = extractelement <4 x float> %i409, i64 1, !dbg !89
  %i412 = extractelement <4 x float> %i409, i64 2, !dbg !89
  %i413 = extractelement <4 x float> %i409, i64 3, !dbg !89
  %i414 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !90
  %i415 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %i414), !dbg !90
  %i416 = add nuw nsw i32 %i415, 64, !dbg !90
  %i417 = xor i32 %i415, 16, !dbg !90
  %i418 = and i32 %i416, 192, !dbg !90
  %i419 = icmp samesign ult i32 %i417, %i418, !dbg !90
  %i420 = select i1 %i419, i32 %i417, i32 %i415, !dbg !90
  %i421 = shl nuw nsw i32 %i420, 2, !dbg !90
  %i422 = bitcast float %i366 to i32, !dbg !90
  %i423 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i422), !dbg !90
  %i424 = bitcast i32 %i423 to float, !dbg !90
  %i425 = call float @llvm.maxnum.f32(float %i366, float %i424), !dbg !90
  %i426 = xor i32 %i415, 32, !dbg !90
  %i427 = icmp samesign ult i32 %i426, %i418, !dbg !90
  %i428 = select i1 %i427, i32 %i426, i32 %i415, !dbg !90
  %i429 = shl nuw nsw i32 %i428, 2, !dbg !90
  %i430 = bitcast float %i425 to i32, !dbg !90
  %i431 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i430), !dbg !90
  %i432 = bitcast i32 %i431 to float, !dbg !90
  %i433 = call float @llvm.maxnum.f32(float %i425, float %i432), !dbg !90
  %i437 = call float @llvm.maxnum.f32(float %i369, float -inf), !dbg !90, !frisk.local_reduction !56
  %local.reduce120 = call float @llvm.maxnum.f32(float %i437, float %i375), !dbg !90, !frisk.local_reduction !56
  %local.reduce121 = call float @llvm.maxnum.f32(float %i370, float %i376), !dbg !90, !frisk.local_reduction !56
  %local.reduce122 = call float @llvm.maxnum.f32(float %i371, float %i377), !dbg !90, !frisk.local_reduction !56
  %local.reduce123 = call float @llvm.maxnum.f32(float %i372, float %i378), !dbg !90, !frisk.local_reduction !56
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %i381), !dbg !90, !frisk.local_reduction !56
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %i382), !dbg !90, !frisk.local_reduction !56
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %i383), !dbg !90, !frisk.local_reduction !56
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %i384), !dbg !90, !frisk.local_reduction !56
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %i387), !dbg !90, !frisk.local_reduction !56
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %i388), !dbg !90, !frisk.local_reduction !56
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %i389), !dbg !90, !frisk.local_reduction !56
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %i390), !dbg !90, !frisk.local_reduction !56
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %i393), !dbg !90, !frisk.local_reduction !56
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %i394), !dbg !90, !frisk.local_reduction !56
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %i395), !dbg !90, !frisk.local_reduction !56
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %i396), !dbg !90, !frisk.local_reduction !56
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %i399), !dbg !90, !frisk.local_reduction !56
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %i400), !dbg !90, !frisk.local_reduction !56
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %i401), !dbg !90, !frisk.local_reduction !56
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %i402), !dbg !90, !frisk.local_reduction !56
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %i405), !dbg !90, !frisk.local_reduction !56
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %i406), !dbg !90, !frisk.local_reduction !56
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %i407), !dbg !90, !frisk.local_reduction !56
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %i408), !dbg !90, !frisk.local_reduction !56
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %i410), !dbg !90, !frisk.local_reduction !56
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %i411), !dbg !90, !frisk.local_reduction !56
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %i412), !dbg !90, !frisk.local_reduction !56
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %i413), !dbg !90, !frisk.local_reduction !56
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !90, !frisk.local_reduction !56
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !90, !frisk.local_reduction !56
  %i438 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !90, !frisk.local_reduction !56
  %i439 = bitcast float %i438 to i32, !dbg !90
  %i440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i439), !dbg !90
  %i441 = bitcast i32 %i440 to float, !dbg !90
  %i442 = call float @llvm.maxnum.f32(float %i438, float %i441), !dbg !90
  %i443 = bitcast float %i442 to i32, !dbg !90
  %i444 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i443), !dbg !90
  %i445 = bitcast i32 %i444 to float, !dbg !90
  %i446 = call float @llvm.maxnum.f32(float %i442, float %i445), !dbg !90
  %lazy.peak = call float @llvm.maxnum.f32(float %i433, float %i446), !dbg !92
  %0 = fcmp ogt float %lazy.peak, 8.000000e+00, !dbg !92
  %lazy.mask = call i64 @llvm.amdgcn.ballot.i64(i1 %0), !dbg !92
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !92
  %lazy.first = icmp eq i64 %i87, 0, !dbg !92
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !92
  br i1 %lazy.take, label %lazy.slow, label %lazy.join

lazy.slow:                                        ; preds = %bb95
  %lazy.pos0 = call float @llvm.maxnum.f32(float %i433, float 0.000000e+00), !dbg !92
  %lazy.d0 = select i1 %lazy.first, float %i433, float %lazy.pos0, !dbg !92
  %lazy.neg0 = fsub float 0.000000e+00, %lazy.d0, !dbg !92
  %lazy.exp0 = call float @llvm.amdgcn.exp2.f32(float %lazy.neg0), !dbg !92
  %lazy.alpha0 = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp0, !dbg !92
  %lazy.d0.b = insertelement <1 x float> poison, float %lazy.d0, i64 0, !dbg !92
  %lazy.d0.v4 = shufflevector <1 x float> %lazy.d0.b, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !92
  %lazy.alpha0.v1 = insertelement <1 x float> poison, float %lazy.alpha0, i64 0, !dbg !92
  %lazy.alpha0.v8 = shufflevector <1 x float> %lazy.alpha0.v1, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !92
  %lazy.Lo0 = extractvalue [2 x <1 x float>] %i89, 0, !dbg !92
  %lazy.Ln0 = fmul <1 x float> %lazy.Lo0, %lazy.alpha0.v1, !dbg !92
  %lazy.Ns0 = extractelement <1 x float> %lazy.n0, i64 0, !dbg !92
  %lazy.Nd0 = fsub float %lazy.Ns0, %lazy.d0, !dbg !92
  %lazy.Nn0 = insertelement <1 x float> poison, float %lazy.Nd0, i64 0, !dbg !92
  %lazy.pos1 = call float @llvm.maxnum.f32(float %i446, float 0.000000e+00), !dbg !92
  %lazy.d1 = select i1 %lazy.first, float %i446, float %lazy.pos1, !dbg !92
  %lazy.neg1 = fsub float 0.000000e+00, %lazy.d1, !dbg !92
  %lazy.exp1 = call float @llvm.amdgcn.exp2.f32(float %lazy.neg1), !dbg !92
  %lazy.alpha1 = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp1, !dbg !92
  %lazy.d1.b = insertelement <1 x float> poison, float %lazy.d1, i64 0, !dbg !92
  %lazy.d1.v4 = shufflevector <1 x float> %lazy.d1.b, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !92
  %lazy.alpha1.v1 = insertelement <1 x float> poison, float %lazy.alpha1, i64 0, !dbg !92
  %lazy.alpha1.v8 = shufflevector <1 x float> %lazy.alpha1.v1, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !92
  %lazy.Lo1 = extractvalue [2 x <1 x float>] %i89, 1, !dbg !92
  %lazy.Ln1 = fmul <1 x float> %lazy.Lo1, %lazy.alpha1.v1, !dbg !92
  %lazy.Ns1 = extractelement <1 x float> %lazy.n1, i64 0, !dbg !92
  %lazy.Nd1 = fsub float %lazy.Ns1, %lazy.d1, !dbg !92
  %lazy.Nn1 = insertelement <1 x float> poison, float %lazy.Nd1, i64 0, !dbg !92
  %lazy.LA = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.Ln0, 0, !dbg !92
  %lazy.LB = insertvalue [2 x <1 x float>] %lazy.LA, <1 x float> %lazy.Ln1, 1, !dbg !92
  %lazy.NA = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.Nn0, 0, !dbg !92
  %lazy.NB = insertvalue [2 x <1 x float>] %lazy.NA, <1 x float> %lazy.Nn1, 1, !dbg !92
  %lazy.Oo.i88.0 = extractvalue [2 x <8 x float>] %i88, 0, !dbg !92
  %lazy.On.i88.0 = fmul <8 x float> %lazy.Oo.i88.0, %lazy.alpha0.v8, !dbg !92
  %lazy.Oo.i88.1 = extractvalue [2 x <8 x float>] %i88, 1, !dbg !92
  %lazy.On.i88.1 = fmul <8 x float> %lazy.Oo.i88.1, %lazy.alpha1.v8, !dbg !92
  %lazy.OA.i88 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.On.i88.0, 0, !dbg !92
  %lazy.OB.i88 = insertvalue [2 x <8 x float>] %lazy.OA.i88, <8 x float> %lazy.On.i88.1, 1, !dbg !92
  %lazy.Oo.i91.0 = extractvalue [2 x <8 x float>] %i91, 0, !dbg !92
  %lazy.On.i91.0 = fmul <8 x float> %lazy.Oo.i91.0, %lazy.alpha0.v8, !dbg !92
  %lazy.Oo.i91.1 = extractvalue [2 x <8 x float>] %i91, 1, !dbg !92
  %lazy.On.i91.1 = fmul <8 x float> %lazy.Oo.i91.1, %lazy.alpha1.v8, !dbg !92
  %lazy.OA.i91 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.On.i91.0, 0, !dbg !92
  %lazy.OB.i91 = insertvalue [2 x <8 x float>] %lazy.OA.i91, <8 x float> %lazy.On.i91.1, 1, !dbg !92
  %lazy.Oo.i92.0 = extractvalue [2 x <8 x float>] %i92, 0, !dbg !92
  %lazy.On.i92.0 = fmul <8 x float> %lazy.Oo.i92.0, %lazy.alpha0.v8, !dbg !92
  %lazy.Oo.i92.1 = extractvalue [2 x <8 x float>] %i92, 1, !dbg !92
  %lazy.On.i92.1 = fmul <8 x float> %lazy.Oo.i92.1, %lazy.alpha1.v8, !dbg !92
  %lazy.OA.i92 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.On.i92.0, 0, !dbg !92
  %lazy.OB.i92 = insertvalue [2 x <8 x float>] %lazy.OA.i92, <8 x float> %lazy.On.i92.1, 1, !dbg !92
  %lazy.Oo.i93.0 = extractvalue [2 x <8 x float>] %i93, 0, !dbg !92
  %lazy.On.i93.0 = fmul <8 x float> %lazy.Oo.i93.0, %lazy.alpha0.v8, !dbg !92
  %lazy.Oo.i93.1 = extractvalue [2 x <8 x float>] %i93, 1, !dbg !92
  %lazy.On.i93.1 = fmul <8 x float> %lazy.Oo.i93.1, %lazy.alpha1.v8, !dbg !92
  %lazy.OA.i93 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.On.i93.0, 0, !dbg !92
  %lazy.OB.i93 = insertvalue [2 x <8 x float>] %lazy.OA.i93, <8 x float> %lazy.On.i93.1, 1, !dbg !92
  %lazy.SS.i304 = fsub <4 x float> %i304, %lazy.d0.v4, !dbg !92
  %lazy.SS.i367 = fsub <4 x float> %i367, %lazy.d1.v4, !dbg !92
  %lazy.SS.i313 = fsub <4 x float> %i313, %lazy.d0.v4, !dbg !92
  %lazy.SS.i373 = fsub <4 x float> %i373, %lazy.d1.v4, !dbg !92
  %lazy.SS.i321 = fsub <4 x float> %i321, %lazy.d0.v4, !dbg !92
  %lazy.SS.i379 = fsub <4 x float> %i379, %lazy.d1.v4, !dbg !92
  %lazy.SS.i329 = fsub <4 x float> %i329, %lazy.d0.v4, !dbg !92
  %lazy.SS.i385 = fsub <4 x float> %i385, %lazy.d1.v4, !dbg !92
  %lazy.SS.i337 = fsub <4 x float> %i337, %lazy.d0.v4, !dbg !92
  %lazy.SS.i391 = fsub <4 x float> %i391, %lazy.d1.v4, !dbg !92
  %lazy.SS.i345 = fsub <4 x float> %i345, %lazy.d0.v4, !dbg !92
  %lazy.SS.i397 = fsub <4 x float> %i397, %lazy.d1.v4, !dbg !92
  %lazy.SS.i353 = fsub <4 x float> %i353, %lazy.d0.v4, !dbg !92
  %lazy.SS.i403 = fsub <4 x float> %i403, %lazy.d1.v4, !dbg !92
  %lazy.SS.i360 = fsub <4 x float> %i360, %lazy.d0.v4, !dbg !92
  %lazy.SS.i409 = fsub <4 x float> %i409, %lazy.d1.v4, !dbg !92
  br label %lazy.join

lazy.join:                                        ; preds = %lazy.slow, %bb95
  %lazy.S.i304 = phi <4 x float> [ %i304, %bb95 ], [ %lazy.SS.i304, %lazy.slow ], !dbg !92
  %lazy.S.i367 = phi <4 x float> [ %i367, %bb95 ], [ %lazy.SS.i367, %lazy.slow ], !dbg !92
  %lazy.S.i313 = phi <4 x float> [ %i313, %bb95 ], [ %lazy.SS.i313, %lazy.slow ], !dbg !92
  %lazy.S.i373 = phi <4 x float> [ %i373, %bb95 ], [ %lazy.SS.i373, %lazy.slow ], !dbg !92
  %lazy.S.i321 = phi <4 x float> [ %i321, %bb95 ], [ %lazy.SS.i321, %lazy.slow ], !dbg !92
  %lazy.S.i379 = phi <4 x float> [ %i379, %bb95 ], [ %lazy.SS.i379, %lazy.slow ], !dbg !92
  %lazy.S.i329 = phi <4 x float> [ %i329, %bb95 ], [ %lazy.SS.i329, %lazy.slow ], !dbg !92
  %lazy.S.i385 = phi <4 x float> [ %i385, %bb95 ], [ %lazy.SS.i385, %lazy.slow ], !dbg !92
  %lazy.S.i337 = phi <4 x float> [ %i337, %bb95 ], [ %lazy.SS.i337, %lazy.slow ], !dbg !92
  %lazy.S.i391 = phi <4 x float> [ %i391, %bb95 ], [ %lazy.SS.i391, %lazy.slow ], !dbg !92
  %lazy.S.i345 = phi <4 x float> [ %i345, %bb95 ], [ %lazy.SS.i345, %lazy.slow ], !dbg !92
  %lazy.S.i397 = phi <4 x float> [ %i397, %bb95 ], [ %lazy.SS.i397, %lazy.slow ], !dbg !92
  %lazy.S.i353 = phi <4 x float> [ %i353, %bb95 ], [ %lazy.SS.i353, %lazy.slow ], !dbg !92
  %lazy.S.i403 = phi <4 x float> [ %i403, %bb95 ], [ %lazy.SS.i403, %lazy.slow ], !dbg !92
  %lazy.S.i360 = phi <4 x float> [ %i360, %bb95 ], [ %lazy.SS.i360, %lazy.slow ], !dbg !92
  %lazy.S.i409 = phi <4 x float> [ %i409, %bb95 ], [ %lazy.SS.i409, %lazy.slow ], !dbg !92
  %lazy.O.i88 = phi [2 x <8 x float>] [ %i88, %bb95 ], [ %lazy.OB.i88, %lazy.slow ], !dbg !92
  %lazy.O.i91 = phi [2 x <8 x float>] [ %i91, %bb95 ], [ %lazy.OB.i91, %lazy.slow ], !dbg !92
  %lazy.O.i92 = phi [2 x <8 x float>] [ %i92, %bb95 ], [ %lazy.OB.i92, %lazy.slow ], !dbg !92
  %lazy.O.i93 = phi [2 x <8 x float>] [ %i93, %bb95 ], [ %lazy.OB.i93, %lazy.slow ], !dbg !92
  %lazy.L = phi [2 x <1 x float>] [ %i89, %bb95 ], [ %lazy.LB, %lazy.slow ], !dbg !92
  %lazy.N = phi [2 x <1 x float>] [ %i90, %bb95 ], [ %lazy.NB, %lazy.slow ], !dbg !92
  %i473 = extractvalue [2 x <1 x float>] %lazy.L, 0, !dbg !95
  %i474 = extractelement <1 x float> %i473, i64 0, !dbg !95
  %i481 = extractvalue [2 x <1 x float>] %lazy.L, 1, !dbg !95
  %i482 = extractelement <1 x float> %i481, i64 0, !dbg !95
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !88
  fence syncscope("workgroup") release, !dbg !88
  call void @llvm.amdgcn.s.barrier(), !dbg !88
  fence syncscope("workgroup") acquire, !dbg !88
  %vdb.address1436 = or disjoint i64 %i109, %address.local921, !dbg !88
  %vdb.address1437 = trunc nuw i64 %vdb.address1436 to i32, !dbg !88
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !88
  %vdb.address1445 = or disjoint i64 %i109, %address.local927, !dbg !88
  %vdb.address1446 = trunc nuw i64 %vdb.address1445 to i32, !dbg !88
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !88
  %vdb.dma.base = add i32 %dma.buffer.lds1374, 8192, !dbg !99
  %i700 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !88
  %vdb.dma.base1452 = add i32 %dma.buffer.lds1371, 8192, !dbg !99
  %i701 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !88
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !99
  %address.local930 = and i32 %address.local929, 120, !dbg !99
  %address.local932 = and i32 %address.local575, 1536, !dbg !99
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !99
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !99
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !99
  %i702 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i703 = bitcast <4 x i32> %i702 to <8 x half>, !dbg !99
  %address.local936 = and i64 %address.local499, 60, !dbg !99
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !99
  %address.local939.idx = shl nuw nsw i64 %address.local937, 4, !dbg !99
  %address.local939 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local939.idx, !dbg !99
  %address.local940 = lshr i64 %address.local555, 4, !dbg !99
  %address.local942.idx = mul nuw nsw i64 %address.local940, 768, !dbg !99
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local942.idx, !dbg !99
  %address.local944.idx = mul nsw i64 %address.local542, -64, !dbg !99
  %address.local944 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local944.idx, !dbg !99
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !99
  %i704 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i705 = bitcast <4 x i32> %i704 to <8 x half>, !dbg !99
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !99
  %i706 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i707 = bitcast <4 x i32> %i706 to <8 x half>, !dbg !99
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !99
  %i708 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i709 = bitcast <4 x i32> %i708 to <8 x half>, !dbg !99
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !99
  %i710 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i711 = bitcast <4 x i32> %i710 to <8 x half>, !dbg !99
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !99
  %i712 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i713 = bitcast <4 x i32> %i712 to <8 x half>, !dbg !99
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !99
  %i714 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i715 = bitcast <4 x i32> %i714 to <8 x half>, !dbg !99
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !99
  %i716 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i717 = shufflevector <8 x half> %i703, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  %i718 = extractvalue [2 x <8 x float>] %lazy.O.i88, 0, !dbg !94
  %i719 = shufflevector <8 x float> %i718, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i484 = extractelement <4 x float> %lazy.S.i304, i64 0, !dbg !96
  %i485 = call float @llvm.amdgcn.exp2.f32(float %i484), !dbg !96
  %i486 = insertelement <4 x float> poison, float %i485, i64 0, !dbg !97
  %i487 = extractelement <4 x float> %lazy.S.i304, i64 1, !dbg !96
  %i488 = call float @llvm.amdgcn.exp2.f32(float %i487), !dbg !96
  %i489 = insertelement <4 x float> %i486, float %i488, i64 1, !dbg !97
  %i490 = extractelement <4 x float> %lazy.S.i304, i64 2, !dbg !96
  %i491 = call float @llvm.amdgcn.exp2.f32(float %i490), !dbg !96
  %i492 = insertelement <4 x float> %i489, float %i491, i64 2, !dbg !97
  %i493 = extractelement <4 x float> %lazy.S.i304, i64 3, !dbg !96
  %i494 = call float @llvm.amdgcn.exp2.f32(float %i493), !dbg !96
  %i495 = insertelement <4 x float> %i492, float %i494, i64 3, !dbg !97
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i495) #9, !dbg !97, !frisk.register_lifetimes !56
  %i496 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !97
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i496) #9, !dbg !96, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i717, <4 x float> %i719) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i722 = shufflevector <8 x half> %i703, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  %i723 = shufflevector <8 x float> %i718, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i722, <4 x float> %i723) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !94
  %i726 = extractvalue [2 x <8 x float>] %lazy.O.i88, 1, !dbg !94
  %i727 = shufflevector <8 x float> %i726, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i592 = extractelement <4 x float> %lazy.S.i367, i64 0, !dbg !96
  %i593 = call float @llvm.amdgcn.exp2.f32(float %i592), !dbg !96
  %i594 = insertelement <4 x float> poison, float %i593, i64 0, !dbg !97
  %i595 = extractelement <4 x float> %lazy.S.i367, i64 1, !dbg !96
  %i596 = call float @llvm.amdgcn.exp2.f32(float %i595), !dbg !96
  %i597 = insertelement <4 x float> %i594, float %i596, i64 1, !dbg !97
  %i598 = extractelement <4 x float> %lazy.S.i367, i64 2, !dbg !96
  %i599 = call float @llvm.amdgcn.exp2.f32(float %i598), !dbg !96
  %i600 = insertelement <4 x float> %i597, float %i599, i64 2, !dbg !97
  %i601 = extractelement <4 x float> %lazy.S.i367, i64 3, !dbg !96
  %i602 = call float @llvm.amdgcn.exp2.f32(float %i601), !dbg !96
  %i603 = insertelement <4 x float> %i600, float %i602, i64 3, !dbg !97
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i603) #9, !dbg !97, !frisk.register_lifetimes !56
  %i604 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !97
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i604) #9, !dbg !96, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i717, <4 x float> %i727) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !94
  %i730 = shufflevector <8 x float> %i726, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i732 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i722, <4 x float> %i730) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i733 = shufflevector <8 x half> %i705, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i497 = extractelement <4 x float> %lazy.S.i313, i64 0, !dbg !96
  %i498 = call float @llvm.amdgcn.exp2.f32(float %i497), !dbg !96
  %local.reduce89.before.anchor = fadd float %i485, %i498, !dbg !98, !frisk.local_reduction !56
  %local.reduce89 = call float asm sideeffect "", "=v,0"(float %local.reduce89.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i499 = insertelement <4 x float> poison, float %i498, i64 0, !dbg !97
  %i500 = extractelement <4 x float> %lazy.S.i313, i64 1, !dbg !96
  %i501 = call float @llvm.amdgcn.exp2.f32(float %i500), !dbg !96
  %local.reduce90.before.anchor = fadd float %i488, %i501, !dbg !98, !frisk.local_reduction !56
  %local.reduce90 = call float asm sideeffect "", "=v,0"(float %local.reduce90.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i502 = insertelement <4 x float> %i499, float %i501, i64 1, !dbg !97
  %i503 = extractelement <4 x float> %lazy.S.i313, i64 2, !dbg !96
  %i504 = call float @llvm.amdgcn.exp2.f32(float %i503), !dbg !96
  %local.reduce91.before.anchor = fadd float %i491, %i504, !dbg !98, !frisk.local_reduction !56
  %local.reduce91 = call float asm sideeffect "", "=v,0"(float %local.reduce91.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i505 = insertelement <4 x float> %i502, float %i504, i64 2, !dbg !97
  %i506 = extractelement <4 x float> %lazy.S.i313, i64 3, !dbg !96
  %i507 = call float @llvm.amdgcn.exp2.f32(float %i506), !dbg !96
  %local.reduce92.before.anchor = fadd float %i494, %i507, !dbg !98, !frisk.local_reduction !56
  %local.reduce92 = call float asm sideeffect "", "=v,0"(float %local.reduce92.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i508 = insertelement <4 x float> %i505, float %i507, i64 3, !dbg !97
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i508) #9, !dbg !97, !frisk.register_lifetimes !56
  %i509 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !97
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i509) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i734 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i733, <4 x float> %i721) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i735 = shufflevector <8 x half> %i705, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i735, <4 x float> %i725) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i605 = extractelement <4 x float> %lazy.S.i373, i64 0, !dbg !96
  %i606 = call float @llvm.amdgcn.exp2.f32(float %i605), !dbg !96
  %local.reduce.before.anchor = fadd float %i593, %i606, !dbg !98, !frisk.local_reduction !56
  %local.reduce = call float asm sideeffect "", "=v,0"(float %local.reduce.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i607 = insertelement <4 x float> poison, float %i606, i64 0, !dbg !97
  %i608 = extractelement <4 x float> %lazy.S.i373, i64 1, !dbg !96
  %i609 = call float @llvm.amdgcn.exp2.f32(float %i608), !dbg !96
  %local.reduce59.before.anchor = fadd float %i596, %i609, !dbg !98, !frisk.local_reduction !56
  %local.reduce59 = call float asm sideeffect "", "=v,0"(float %local.reduce59.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i610 = insertelement <4 x float> %i607, float %i609, i64 1, !dbg !97
  %i611 = extractelement <4 x float> %lazy.S.i373, i64 2, !dbg !96
  %i612 = call float @llvm.amdgcn.exp2.f32(float %i611), !dbg !96
  %local.reduce60.before.anchor = fadd float %i599, %i612, !dbg !98, !frisk.local_reduction !56
  %local.reduce60 = call float asm sideeffect "", "=v,0"(float %local.reduce60.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i613 = insertelement <4 x float> %i610, float %i612, i64 2, !dbg !97
  %i614 = extractelement <4 x float> %lazy.S.i373, i64 3, !dbg !96
  %i615 = call float @llvm.amdgcn.exp2.f32(float %i614), !dbg !96
  %local.reduce61.before.anchor = fadd float %i602, %i615, !dbg !98, !frisk.local_reduction !56
  %local.reduce61 = call float asm sideeffect "", "=v,0"(float %local.reduce61.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i616 = insertelement <4 x float> %i613, float %i615, i64 3, !dbg !97
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i616) #9, !dbg !97, !frisk.register_lifetimes !56
  %i617 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !97
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i617) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i733, <4 x float> %i729) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i738 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i735, <4 x float> %i732) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i739 = shufflevector <8 x half> %i707, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i510 = extractelement <4 x float> %lazy.S.i321, i64 0, !dbg !96
  %i511 = call float @llvm.amdgcn.exp2.f32(float %i510), !dbg !96
  %local.reduce93.before.anchor = fadd float %local.reduce89, %i511, !dbg !98, !frisk.local_reduction !56
  %local.reduce93 = call float asm sideeffect "", "=v,0"(float %local.reduce93.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i512 = insertelement <4 x float> poison, float %i511, i64 0, !dbg !97
  %i513 = extractelement <4 x float> %lazy.S.i321, i64 1, !dbg !96
  %i514 = call float @llvm.amdgcn.exp2.f32(float %i513), !dbg !96
  %local.reduce94.before.anchor = fadd float %local.reduce90, %i514, !dbg !98, !frisk.local_reduction !56
  %local.reduce94 = call float asm sideeffect "", "=v,0"(float %local.reduce94.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i515 = insertelement <4 x float> %i512, float %i514, i64 1, !dbg !97
  %i516 = extractelement <4 x float> %lazy.S.i321, i64 2, !dbg !96
  %i517 = call float @llvm.amdgcn.exp2.f32(float %i516), !dbg !96
  %local.reduce95.before.anchor = fadd float %local.reduce91, %i517, !dbg !98, !frisk.local_reduction !56
  %local.reduce95 = call float asm sideeffect "", "=v,0"(float %local.reduce95.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i518 = insertelement <4 x float> %i515, float %i517, i64 2, !dbg !97
  %i519 = extractelement <4 x float> %lazy.S.i321, i64 3, !dbg !96
  %i520 = call float @llvm.amdgcn.exp2.f32(float %i519), !dbg !96
  %local.reduce96.before.anchor = fadd float %local.reduce92, %i520, !dbg !98, !frisk.local_reduction !56
  %local.reduce96 = call float asm sideeffect "", "=v,0"(float %local.reduce96.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i521 = insertelement <4 x float> %i518, float %i520, i64 3, !dbg !97
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i521) #9, !dbg !97, !frisk.register_lifetimes !56
  %i522 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !97
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i522) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i740 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i739, <4 x float> %i734) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i741 = shufflevector <8 x half> %i707, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i742 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i741, <4 x float> %i736) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i618 = extractelement <4 x float> %lazy.S.i379, i64 0, !dbg !96
  %i619 = call float @llvm.amdgcn.exp2.f32(float %i618), !dbg !96
  %local.reduce62.before.anchor = fadd float %local.reduce, %i619, !dbg !98, !frisk.local_reduction !56
  %local.reduce62 = call float asm sideeffect "", "=v,0"(float %local.reduce62.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i620 = insertelement <4 x float> poison, float %i619, i64 0, !dbg !97
  %i621 = extractelement <4 x float> %lazy.S.i379, i64 1, !dbg !96
  %i622 = call float @llvm.amdgcn.exp2.f32(float %i621), !dbg !96
  %local.reduce63.before.anchor = fadd float %local.reduce59, %i622, !dbg !98, !frisk.local_reduction !56
  %local.reduce63 = call float asm sideeffect "", "=v,0"(float %local.reduce63.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i623 = insertelement <4 x float> %i620, float %i622, i64 1, !dbg !97
  %i624 = extractelement <4 x float> %lazy.S.i379, i64 2, !dbg !96
  %i625 = call float @llvm.amdgcn.exp2.f32(float %i624), !dbg !96
  %local.reduce64.before.anchor = fadd float %local.reduce60, %i625, !dbg !98, !frisk.local_reduction !56
  %local.reduce64 = call float asm sideeffect "", "=v,0"(float %local.reduce64.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i626 = insertelement <4 x float> %i623, float %i625, i64 2, !dbg !97
  %i627 = extractelement <4 x float> %lazy.S.i379, i64 3, !dbg !96
  %i628 = call float @llvm.amdgcn.exp2.f32(float %i627), !dbg !96
  %local.reduce65.before.anchor = fadd float %local.reduce61, %i628, !dbg !98, !frisk.local_reduction !56
  %local.reduce65 = call float asm sideeffect "", "=v,0"(float %local.reduce65.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i629 = insertelement <4 x float> %i626, float %i628, i64 3, !dbg !97
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i629) #9, !dbg !97, !frisk.register_lifetimes !56
  %i630 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !97
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i630) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i743 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i739, <4 x float> %i737) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i744 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i741, <4 x float> %i738) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i745 = shufflevector <8 x half> %i709, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i523 = extractelement <4 x float> %lazy.S.i329, i64 0, !dbg !96
  %i524 = call float @llvm.amdgcn.exp2.f32(float %i523), !dbg !96
  %local.reduce97.before.anchor = fadd float %local.reduce93, %i524, !dbg !98, !frisk.local_reduction !56
  %local.reduce97 = call float asm sideeffect "", "=v,0"(float %local.reduce97.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i525 = insertelement <4 x float> poison, float %i524, i64 0, !dbg !97
  %i526 = extractelement <4 x float> %lazy.S.i329, i64 1, !dbg !96
  %i527 = call float @llvm.amdgcn.exp2.f32(float %i526), !dbg !96
  %local.reduce98.before.anchor = fadd float %local.reduce94, %i527, !dbg !98, !frisk.local_reduction !56
  %local.reduce98 = call float asm sideeffect "", "=v,0"(float %local.reduce98.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i528 = insertelement <4 x float> %i525, float %i527, i64 1, !dbg !97
  %i529 = extractelement <4 x float> %lazy.S.i329, i64 2, !dbg !96
  %i530 = call float @llvm.amdgcn.exp2.f32(float %i529), !dbg !96
  %local.reduce99.before.anchor = fadd float %local.reduce95, %i530, !dbg !98, !frisk.local_reduction !56
  %local.reduce99 = call float asm sideeffect "", "=v,0"(float %local.reduce99.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i531 = insertelement <4 x float> %i528, float %i530, i64 2, !dbg !97
  %i532 = extractelement <4 x float> %lazy.S.i329, i64 3, !dbg !96
  %i533 = call float @llvm.amdgcn.exp2.f32(float %i532), !dbg !96
  %local.reduce100.before.anchor = fadd float %local.reduce96, %i533, !dbg !98, !frisk.local_reduction !56
  %local.reduce100 = call float asm sideeffect "", "=v,0"(float %local.reduce100.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i534 = insertelement <4 x float> %i531, float %i533, i64 3, !dbg !97
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i534) #9, !dbg !97, !frisk.register_lifetimes !56
  %i535 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !97
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i535) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i746 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i745, <4 x float> %i740) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i747 = shufflevector <8 x half> %i709, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i748 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i747, <4 x float> %i742) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i631 = extractelement <4 x float> %lazy.S.i385, i64 0, !dbg !96
  %i632 = call float @llvm.amdgcn.exp2.f32(float %i631), !dbg !96
  %local.reduce66.before.anchor = fadd float %local.reduce62, %i632, !dbg !98, !frisk.local_reduction !56
  %local.reduce66 = call float asm sideeffect "", "=v,0"(float %local.reduce66.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i633 = insertelement <4 x float> poison, float %i632, i64 0, !dbg !97
  %i634 = extractelement <4 x float> %lazy.S.i385, i64 1, !dbg !96
  %i635 = call float @llvm.amdgcn.exp2.f32(float %i634), !dbg !96
  %local.reduce67.before.anchor = fadd float %local.reduce63, %i635, !dbg !98, !frisk.local_reduction !56
  %local.reduce67 = call float asm sideeffect "", "=v,0"(float %local.reduce67.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i636 = insertelement <4 x float> %i633, float %i635, i64 1, !dbg !97
  %i637 = extractelement <4 x float> %lazy.S.i385, i64 2, !dbg !96
  %i638 = call float @llvm.amdgcn.exp2.f32(float %i637), !dbg !96
  %local.reduce68.before.anchor = fadd float %local.reduce64, %i638, !dbg !98, !frisk.local_reduction !56
  %local.reduce68 = call float asm sideeffect "", "=v,0"(float %local.reduce68.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i639 = insertelement <4 x float> %i636, float %i638, i64 2, !dbg !97
  %i640 = extractelement <4 x float> %lazy.S.i385, i64 3, !dbg !96
  %i641 = call float @llvm.amdgcn.exp2.f32(float %i640), !dbg !96
  %local.reduce69.before.anchor = fadd float %local.reduce65, %i641, !dbg !98, !frisk.local_reduction !56
  %local.reduce69 = call float asm sideeffect "", "=v,0"(float %local.reduce69.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i642 = insertelement <4 x float> %i639, float %i641, i64 3, !dbg !97
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i642) #9, !dbg !97, !frisk.register_lifetimes !56
  %i643 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !97
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i643) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i749 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i745, <4 x float> %i743) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i747, <4 x float> %i744) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i751 = shufflevector <8 x half> %i711, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i536 = extractelement <4 x float> %lazy.S.i337, i64 0, !dbg !96
  %i537 = call float @llvm.amdgcn.exp2.f32(float %i536), !dbg !96
  %local.reduce101.before.anchor = fadd float %local.reduce97, %i537, !dbg !98, !frisk.local_reduction !56
  %local.reduce101 = call float asm sideeffect "", "=v,0"(float %local.reduce101.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i538 = insertelement <4 x float> poison, float %i537, i64 0, !dbg !97
  %i539 = extractelement <4 x float> %lazy.S.i337, i64 1, !dbg !96
  %i540 = call float @llvm.amdgcn.exp2.f32(float %i539), !dbg !96
  %local.reduce102.before.anchor = fadd float %local.reduce98, %i540, !dbg !98, !frisk.local_reduction !56
  %local.reduce102 = call float asm sideeffect "", "=v,0"(float %local.reduce102.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i541 = insertelement <4 x float> %i538, float %i540, i64 1, !dbg !97
  %i542 = extractelement <4 x float> %lazy.S.i337, i64 2, !dbg !96
  %i543 = call float @llvm.amdgcn.exp2.f32(float %i542), !dbg !96
  %local.reduce103.before.anchor = fadd float %local.reduce99, %i543, !dbg !98, !frisk.local_reduction !56
  %local.reduce103 = call float asm sideeffect "", "=v,0"(float %local.reduce103.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i544 = insertelement <4 x float> %i541, float %i543, i64 2, !dbg !97
  %i545 = extractelement <4 x float> %lazy.S.i337, i64 3, !dbg !96
  %i546 = call float @llvm.amdgcn.exp2.f32(float %i545), !dbg !96
  %local.reduce104.before.anchor = fadd float %local.reduce100, %i546, !dbg !98, !frisk.local_reduction !56
  %local.reduce104 = call float asm sideeffect "", "=v,0"(float %local.reduce104.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i547 = insertelement <4 x float> %i544, float %i546, i64 3, !dbg !97
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i547) #9, !dbg !97, !frisk.register_lifetimes !56
  %i548 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !97
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i548) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i752 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i751, <4 x float> %i746) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i753 = shufflevector <8 x half> %i711, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i754 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i753, <4 x float> %i748) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i644 = extractelement <4 x float> %lazy.S.i391, i64 0, !dbg !96
  %i645 = call float @llvm.amdgcn.exp2.f32(float %i644), !dbg !96
  %local.reduce70.before.anchor = fadd float %local.reduce66, %i645, !dbg !98, !frisk.local_reduction !56
  %local.reduce70 = call float asm sideeffect "", "=v,0"(float %local.reduce70.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i646 = insertelement <4 x float> poison, float %i645, i64 0, !dbg !97
  %i647 = extractelement <4 x float> %lazy.S.i391, i64 1, !dbg !96
  %i648 = call float @llvm.amdgcn.exp2.f32(float %i647), !dbg !96
  %local.reduce71.before.anchor = fadd float %local.reduce67, %i648, !dbg !98, !frisk.local_reduction !56
  %local.reduce71 = call float asm sideeffect "", "=v,0"(float %local.reduce71.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i649 = insertelement <4 x float> %i646, float %i648, i64 1, !dbg !97
  %i650 = extractelement <4 x float> %lazy.S.i391, i64 2, !dbg !96
  %i651 = call float @llvm.amdgcn.exp2.f32(float %i650), !dbg !96
  %local.reduce72.before.anchor = fadd float %local.reduce68, %i651, !dbg !98, !frisk.local_reduction !56
  %local.reduce72 = call float asm sideeffect "", "=v,0"(float %local.reduce72.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i652 = insertelement <4 x float> %i649, float %i651, i64 2, !dbg !97
  %i653 = extractelement <4 x float> %lazy.S.i391, i64 3, !dbg !96
  %i654 = call float @llvm.amdgcn.exp2.f32(float %i653), !dbg !96
  %local.reduce73.before.anchor = fadd float %local.reduce69, %i654, !dbg !98, !frisk.local_reduction !56
  %local.reduce73 = call float asm sideeffect "", "=v,0"(float %local.reduce73.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i655 = insertelement <4 x float> %i652, float %i654, i64 3, !dbg !97
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i655) #9, !dbg !97, !frisk.register_lifetimes !56
  %i656 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !97
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i656) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i751, <4 x float> %i749) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i753, <4 x float> %i750) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i757 = shufflevector <8 x half> %i713, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i549 = extractelement <4 x float> %lazy.S.i345, i64 0, !dbg !96
  %i550 = call float @llvm.amdgcn.exp2.f32(float %i549), !dbg !96
  %local.reduce105.before.anchor = fadd float %local.reduce101, %i550, !dbg !98, !frisk.local_reduction !56
  %local.reduce105 = call float asm sideeffect "", "=v,0"(float %local.reduce105.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i551 = insertelement <4 x float> poison, float %i550, i64 0, !dbg !97
  %i552 = extractelement <4 x float> %lazy.S.i345, i64 1, !dbg !96
  %i553 = call float @llvm.amdgcn.exp2.f32(float %i552), !dbg !96
  %local.reduce106.before.anchor = fadd float %local.reduce102, %i553, !dbg !98, !frisk.local_reduction !56
  %local.reduce106 = call float asm sideeffect "", "=v,0"(float %local.reduce106.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i554 = insertelement <4 x float> %i551, float %i553, i64 1, !dbg !97
  %i555 = extractelement <4 x float> %lazy.S.i345, i64 2, !dbg !96
  %i556 = call float @llvm.amdgcn.exp2.f32(float %i555), !dbg !96
  %local.reduce107.before.anchor = fadd float %local.reduce103, %i556, !dbg !98, !frisk.local_reduction !56
  %local.reduce107 = call float asm sideeffect "", "=v,0"(float %local.reduce107.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i557 = insertelement <4 x float> %i554, float %i556, i64 2, !dbg !97
  %i558 = extractelement <4 x float> %lazy.S.i345, i64 3, !dbg !96
  %i559 = call float @llvm.amdgcn.exp2.f32(float %i558), !dbg !96
  %local.reduce108.before.anchor = fadd float %local.reduce104, %i559, !dbg !98, !frisk.local_reduction !56
  %local.reduce108 = call float asm sideeffect "", "=v,0"(float %local.reduce108.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i560 = insertelement <4 x float> %i557, float %i559, i64 3, !dbg !97
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i560) #9, !dbg !97, !frisk.register_lifetimes !56
  %i561 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !97
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i561) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i757, <4 x float> %i752) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i759 = shufflevector <8 x half> %i713, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i759, <4 x float> %i754) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i657 = extractelement <4 x float> %lazy.S.i397, i64 0, !dbg !96
  %i658 = call float @llvm.amdgcn.exp2.f32(float %i657), !dbg !96
  %local.reduce74.before.anchor = fadd float %local.reduce70, %i658, !dbg !98, !frisk.local_reduction !56
  %local.reduce74 = call float asm sideeffect "", "=v,0"(float %local.reduce74.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i659 = insertelement <4 x float> poison, float %i658, i64 0, !dbg !97
  %i660 = extractelement <4 x float> %lazy.S.i397, i64 1, !dbg !96
  %i661 = call float @llvm.amdgcn.exp2.f32(float %i660), !dbg !96
  %local.reduce75.before.anchor = fadd float %local.reduce71, %i661, !dbg !98, !frisk.local_reduction !56
  %local.reduce75 = call float asm sideeffect "", "=v,0"(float %local.reduce75.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i662 = insertelement <4 x float> %i659, float %i661, i64 1, !dbg !97
  %i663 = extractelement <4 x float> %lazy.S.i397, i64 2, !dbg !96
  %i664 = call float @llvm.amdgcn.exp2.f32(float %i663), !dbg !96
  %local.reduce76.before.anchor = fadd float %local.reduce72, %i664, !dbg !98, !frisk.local_reduction !56
  %local.reduce76 = call float asm sideeffect "", "=v,0"(float %local.reduce76.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i665 = insertelement <4 x float> %i662, float %i664, i64 2, !dbg !97
  %i666 = extractelement <4 x float> %lazy.S.i397, i64 3, !dbg !96
  %i667 = call float @llvm.amdgcn.exp2.f32(float %i666), !dbg !96
  %local.reduce77.before.anchor = fadd float %local.reduce73, %i667, !dbg !98, !frisk.local_reduction !56
  %local.reduce77 = call float asm sideeffect "", "=v,0"(float %local.reduce77.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i668 = insertelement <4 x float> %i665, float %i667, i64 3, !dbg !97
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i668) #9, !dbg !97, !frisk.register_lifetimes !56
  %i669 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !97
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i669) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i761 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i757, <4 x float> %i755) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i762 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i759, <4 x float> %i756) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i763 = shufflevector <8 x half> %i715, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i562 = extractelement <4 x float> %lazy.S.i353, i64 0, !dbg !96
  %i563 = call float @llvm.amdgcn.exp2.f32(float %i562), !dbg !96
  %local.reduce109.before.anchor = fadd float %local.reduce105, %i563, !dbg !98, !frisk.local_reduction !56
  %local.reduce109 = call float asm sideeffect "", "=v,0"(float %local.reduce109.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i564 = insertelement <4 x float> poison, float %i563, i64 0, !dbg !97
  %i565 = extractelement <4 x float> %lazy.S.i353, i64 1, !dbg !96
  %i566 = call float @llvm.amdgcn.exp2.f32(float %i565), !dbg !96
  %local.reduce110.before.anchor = fadd float %local.reduce106, %i566, !dbg !98, !frisk.local_reduction !56
  %local.reduce110 = call float asm sideeffect "", "=v,0"(float %local.reduce110.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i567 = insertelement <4 x float> %i564, float %i566, i64 1, !dbg !97
  %i568 = extractelement <4 x float> %lazy.S.i353, i64 2, !dbg !96
  %i569 = call float @llvm.amdgcn.exp2.f32(float %i568), !dbg !96
  %local.reduce111.before.anchor = fadd float %local.reduce107, %i569, !dbg !98, !frisk.local_reduction !56
  %local.reduce111 = call float asm sideeffect "", "=v,0"(float %local.reduce111.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i570 = insertelement <4 x float> %i567, float %i569, i64 2, !dbg !97
  %i571 = extractelement <4 x float> %lazy.S.i353, i64 3, !dbg !96
  %i572 = call float @llvm.amdgcn.exp2.f32(float %i571), !dbg !96
  %local.reduce112.before.anchor = fadd float %local.reduce108, %i572, !dbg !98, !frisk.local_reduction !56
  %local.reduce112 = call float asm sideeffect "", "=v,0"(float %local.reduce112.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i573 = insertelement <4 x float> %i570, float %i572, i64 3, !dbg !97
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i573) #9, !dbg !97, !frisk.register_lifetimes !56
  %i574 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !97
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i574) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i764 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i763, <4 x float> %i758) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i765 = shufflevector <8 x half> %i715, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i766 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i765, <4 x float> %i760) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i670 = extractelement <4 x float> %lazy.S.i403, i64 0, !dbg !96
  %i671 = call float @llvm.amdgcn.exp2.f32(float %i670), !dbg !96
  %local.reduce78.before.anchor = fadd float %local.reduce74, %i671, !dbg !98, !frisk.local_reduction !56
  %local.reduce78 = call float asm sideeffect "", "=v,0"(float %local.reduce78.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i672 = insertelement <4 x float> poison, float %i671, i64 0, !dbg !97
  %i673 = extractelement <4 x float> %lazy.S.i403, i64 1, !dbg !96
  %i674 = call float @llvm.amdgcn.exp2.f32(float %i673), !dbg !96
  %local.reduce79.before.anchor = fadd float %local.reduce75, %i674, !dbg !98, !frisk.local_reduction !56
  %local.reduce79 = call float asm sideeffect "", "=v,0"(float %local.reduce79.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i675 = insertelement <4 x float> %i672, float %i674, i64 1, !dbg !97
  %i676 = extractelement <4 x float> %lazy.S.i403, i64 2, !dbg !96
  %i677 = call float @llvm.amdgcn.exp2.f32(float %i676), !dbg !96
  %local.reduce80.before.anchor = fadd float %local.reduce76, %i677, !dbg !98, !frisk.local_reduction !56
  %local.reduce80 = call float asm sideeffect "", "=v,0"(float %local.reduce80.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i678 = insertelement <4 x float> %i675, float %i677, i64 2, !dbg !97
  %i679 = extractelement <4 x float> %lazy.S.i403, i64 3, !dbg !96
  %i680 = call float @llvm.amdgcn.exp2.f32(float %i679), !dbg !96
  %local.reduce81.before.anchor = fadd float %local.reduce77, %i680, !dbg !98, !frisk.local_reduction !56
  %local.reduce81 = call float asm sideeffect "", "=v,0"(float %local.reduce81.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i681 = insertelement <4 x float> %i678, float %i680, i64 3, !dbg !97
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i681) #9, !dbg !97, !frisk.register_lifetimes !56
  %i682 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !97
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i682) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i767 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i763, <4 x float> %i761) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i768 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i765, <4 x float> %i762) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i769 = bitcast <4 x i32> %i716 to <8 x half>, !dbg !99
  %i770 = shufflevector <8 x half> %i769, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i575 = extractelement <4 x float> %lazy.S.i360, i64 0, !dbg !96
  %i576 = call float @llvm.amdgcn.exp2.f32(float %i575), !dbg !96
  %local.reduce113.before.anchor = fadd float %local.reduce109, %i576, !dbg !98, !frisk.local_reduction !56
  %local.reduce113 = call float asm sideeffect "", "=v,0"(float %local.reduce113.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i577 = insertelement <4 x float> poison, float %i576, i64 0, !dbg !97
  %i578 = extractelement <4 x float> %lazy.S.i360, i64 1, !dbg !96
  %i579 = call float @llvm.amdgcn.exp2.f32(float %i578), !dbg !96
  %local.reduce114.before.anchor = fadd float %local.reduce110, %i579, !dbg !98, !frisk.local_reduction !56
  %local.reduce114 = call float asm sideeffect "", "=v,0"(float %local.reduce114.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %local.reduce117.before.anchor = fadd float %local.reduce113, %local.reduce114, !dbg !98, !frisk.local_reduction !56
  %local.reduce117 = call float asm sideeffect "", "=v,0"(float %local.reduce117.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i580 = insertelement <4 x float> %i577, float %i579, i64 1, !dbg !97
  %i581 = extractelement <4 x float> %lazy.S.i360, i64 2, !dbg !96
  %i582 = call float @llvm.amdgcn.exp2.f32(float %i581), !dbg !96
  %local.reduce115.before.anchor = fadd float %local.reduce111, %i582, !dbg !98, !frisk.local_reduction !56
  %local.reduce115 = call float asm sideeffect "", "=v,0"(float %local.reduce115.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i583 = insertelement <4 x float> %i580, float %i582, i64 2, !dbg !97
  %i584 = extractelement <4 x float> %lazy.S.i360, i64 3, !dbg !96
  %i585 = call float @llvm.amdgcn.exp2.f32(float %i584), !dbg !96
  %local.reduce116.before.anchor = fadd float %local.reduce112, %i585, !dbg !98, !frisk.local_reduction !56
  %local.reduce116 = call float asm sideeffect "", "=v,0"(float %local.reduce116.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %local.reduce118.before.anchor = fadd float %local.reduce115, %local.reduce116, !dbg !98, !frisk.local_reduction !56
  %local.reduce118 = call float asm sideeffect "", "=v,0"(float %local.reduce118.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i588.before.anchor = fadd float %local.reduce117, %local.reduce118, !dbg !98, !frisk.local_reduction !56
  %i588 = call float asm sideeffect "", "=v,0"(float %i588.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i589.before.anchor = fadd float %i588, %i474, !dbg !98
  %i589 = call float asm sideeffect "", "=v,0"(float %i589.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i590 = insertelement <1 x float> poison, float %i589, i64 0, !dbg !62
  %i591 = insertvalue [2 x <1 x float>] poison, <1 x float> %i590, 0, !dbg !62
  %i586 = insertelement <4 x float> %i583, float %i585, i64 3, !dbg !97
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i586) #9, !dbg !97, !frisk.register_lifetimes !56
  %i587 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !97
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i587) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i770, <4 x float> %i764) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i772 = shufflevector <8 x half> %i769, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i772, <4 x float> %i766) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i683 = extractelement <4 x float> %lazy.S.i409, i64 0, !dbg !96
  %i684 = call float @llvm.amdgcn.exp2.f32(float %i683), !dbg !96
  %local.reduce82.before.anchor = fadd float %local.reduce78, %i684, !dbg !98, !frisk.local_reduction !56
  %local.reduce82 = call float asm sideeffect "", "=v,0"(float %local.reduce82.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i685 = insertelement <4 x float> poison, float %i684, i64 0, !dbg !97
  %i686 = extractelement <4 x float> %lazy.S.i409, i64 1, !dbg !96
  %i687 = call float @llvm.amdgcn.exp2.f32(float %i686), !dbg !96
  %local.reduce83.before.anchor = fadd float %local.reduce79, %i687, !dbg !98, !frisk.local_reduction !56
  %local.reduce83 = call float asm sideeffect "", "=v,0"(float %local.reduce83.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %local.reduce86.before.anchor = fadd float %local.reduce82, %local.reduce83, !dbg !98, !frisk.local_reduction !56
  %local.reduce86 = call float asm sideeffect "", "=v,0"(float %local.reduce86.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i688 = insertelement <4 x float> %i685, float %i687, i64 1, !dbg !97
  %i689 = extractelement <4 x float> %lazy.S.i409, i64 2, !dbg !96
  %i690 = call float @llvm.amdgcn.exp2.f32(float %i689), !dbg !96
  %local.reduce84.before.anchor = fadd float %local.reduce80, %i690, !dbg !98, !frisk.local_reduction !56
  %local.reduce84 = call float asm sideeffect "", "=v,0"(float %local.reduce84.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i691 = insertelement <4 x float> %i688, float %i690, i64 2, !dbg !97
  %i692 = extractelement <4 x float> %lazy.S.i409, i64 3, !dbg !96
  %i693 = call float @llvm.amdgcn.exp2.f32(float %i692), !dbg !96
  %local.reduce85.before.anchor = fadd float %local.reduce81, %i693, !dbg !98, !frisk.local_reduction !56
  %local.reduce85 = call float asm sideeffect "", "=v,0"(float %local.reduce85.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %local.reduce87.before.anchor = fadd float %local.reduce84, %local.reduce85, !dbg !98, !frisk.local_reduction !56
  %local.reduce87 = call float asm sideeffect "", "=v,0"(float %local.reduce87.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i696.before.anchor = fadd float %local.reduce86, %local.reduce87, !dbg !98, !frisk.local_reduction !56
  %i696 = call float asm sideeffect "", "=v,0"(float %i696.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i697.before.anchor = fadd float %i696, %i482, !dbg !98
  %i697 = call float asm sideeffect "", "=v,0"(float %i697.before.anchor) #9, !dbg !94, !frisk.register_lifetimes !56
  %i698 = insertelement <1 x float> poison, float %i697, i64 0, !dbg !62
  %i699 = insertvalue [2 x <1 x float>] %i591, <1 x float> %i698, 1, !dbg !62
  %i694 = insertelement <4 x float> %i691, float %i693, i64 3, !dbg !97
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i694) #9, !dbg !97, !frisk.register_lifetimes !56
  %i695 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !97
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i695) #9, !dbg !98, !frisk.register_lifetimes !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99, !frisk.softmax_fragments !56
  %i774 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i770, <4 x float> %i767) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i775 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i772, <4 x float> %i768) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !88
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !88
  fence syncscope("workgroup") release, !dbg !88
  call void @llvm.amdgcn.s.barrier(), !dbg !88
  fence syncscope("workgroup") acquire, !dbg !88
  %vdb.address1462 = or disjoint i64 %i189, %address.local921, !dbg !88
  %vdb.address1463 = trunc nuw i64 %vdb.address1462 to i32, !dbg !88
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !88
  %vdb.address1471 = or disjoint i64 %i189, %address.local927, !dbg !88
  %vdb.address1472 = trunc nuw i64 %vdb.address1471 to i32, !dbg !88
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !88
  %i776 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !88
  %i777 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %vdb.address1473, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !88
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !99
  %i778 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i779 = bitcast <4 x i32> %i778 to <8 x half>, !dbg !99
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8448, !dbg !99
  %i780 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i781 = bitcast <4 x i32> %i780 to <8 x half>, !dbg !99
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8704, !dbg !99
  %i782 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i783 = bitcast <4 x i32> %i782 to <8 x half>, !dbg !99
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8960, !dbg !99
  %i784 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i785 = bitcast <4 x i32> %i784 to <8 x half>, !dbg !99
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local935, i64 12288, !dbg !99
  %i786 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i787 = bitcast <4 x i32> %i786 to <8 x half>, !dbg !99
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12544, !dbg !99
  %i788 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i789 = bitcast <4 x i32> %i788 to <8 x half>, !dbg !99
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12800, !dbg !99
  %i790 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i791 = bitcast <4 x i32> %i790 to <8 x half>, !dbg !99
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local944, i64 13056, !dbg !99
  %i792 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i793 = shufflevector <8 x half> %i779, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  %i794 = extractvalue [2 x <8 x float>] %lazy.O.i91, 0, !dbg !94
  %i795 = shufflevector <8 x float> %i794, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i793, <4 x float> %i795) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i798 = shufflevector <8 x half> %i779, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  %i799 = shufflevector <8 x float> %i794, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i801 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i798, <4 x float> %i799) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !94
  %i802 = extractvalue [2 x <8 x float>] %lazy.O.i91, 1, !dbg !94
  %i803 = shufflevector <8 x float> %i802, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i805 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i793, <4 x float> %i803) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !94
  %i806 = shufflevector <8 x float> %i802, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i808 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i798, <4 x float> %i806) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i809 = shufflevector <8 x half> %i781, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i809, <4 x float> %i797) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i811 = shufflevector <8 x half> %i781, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i811, <4 x float> %i801) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i813 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i809, <4 x float> %i805) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i811, <4 x float> %i808) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i815 = shufflevector <8 x half> %i783, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i815, <4 x float> %i810) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i817 = shufflevector <8 x half> %i783, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i818 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i817, <4 x float> %i812) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i819 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i815, <4 x float> %i813) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i820 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i817, <4 x float> %i814) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i821 = shufflevector <8 x half> %i785, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i821, <4 x float> %i816) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i823 = shufflevector <8 x half> %i785, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i823, <4 x float> %i818) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i821, <4 x float> %i819) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i823, <4 x float> %i820) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i827 = shufflevector <8 x half> %i787, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i827, <4 x float> %i822) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i829 = shufflevector <8 x half> %i787, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i830 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i829, <4 x float> %i824) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i827, <4 x float> %i825) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i832 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i829, <4 x float> %i826) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i833 = shufflevector <8 x half> %i789, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i834 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i833, <4 x float> %i828) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i835 = shufflevector <8 x half> %i789, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i836 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i835, <4 x float> %i830) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i833, <4 x float> %i831) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i835, <4 x float> %i832) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i839 = shufflevector <8 x half> %i791, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i839, <4 x float> %i834) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i841 = shufflevector <8 x half> %i791, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i841, <4 x float> %i836) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i839, <4 x float> %i837) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i841, <4 x float> %i838) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i845 = bitcast <4 x i32> %i792 to <8 x half>, !dbg !99
  %i846 = shufflevector <8 x half> %i845, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i846, <4 x float> %i840) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i848 = shufflevector <8 x half> %i845, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i848, <4 x float> %i842) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i846, <4 x float> %i843) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i848, <4 x float> %i844) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !88
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !88
  fence syncscope("workgroup") release, !dbg !88
  call void @llvm.amdgcn.s.barrier(), !dbg !88
  fence syncscope("workgroup") acquire, !dbg !88
  %vdb.address1480 = or disjoint i64 %i242, %address.local921, !dbg !88
  %vdb.address1481 = trunc nuw i64 %vdb.address1480 to i32, !dbg !88
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !88
  %vdb.address1489 = or disjoint i64 %i242, %address.local927, !dbg !88
  %vdb.address1490 = trunc nuw i64 %vdb.address1489 to i32, !dbg !88
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !88
  %i852 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1482, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !88
  %i853 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1491, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !88
  %i854 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i855 = bitcast <4 x i32> %i854 to <8 x half>, !dbg !99
  %i856 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i857 = bitcast <4 x i32> %i856 to <8 x half>, !dbg !99
  %i858 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i859 = bitcast <4 x i32> %i858 to <8 x half>, !dbg !99
  %i860 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i861 = bitcast <4 x i32> %i860 to <8 x half>, !dbg !99
  %i862 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i863 = bitcast <4 x i32> %i862 to <8 x half>, !dbg !99
  %i864 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i865 = bitcast <4 x i32> %i864 to <8 x half>, !dbg !99
  %i866 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i867 = bitcast <4 x i32> %i866 to <8 x half>, !dbg !99
  %i868 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i869 = shufflevector <8 x half> %i855, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  %i870 = extractvalue [2 x <8 x float>] %lazy.O.i92, 0, !dbg !94
  %i871 = shufflevector <8 x float> %i870, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i869, <4 x float> %i871) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i874 = shufflevector <8 x half> %i855, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  %i875 = shufflevector <8 x float> %i870, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i874, <4 x float> %i875) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !94
  %i878 = extractvalue [2 x <8 x float>] %lazy.O.i92, 1, !dbg !94
  %i879 = shufflevector <8 x float> %i878, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i869, <4 x float> %i879) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !94
  %i882 = shufflevector <8 x float> %i878, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i874, <4 x float> %i882) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i885 = shufflevector <8 x half> %i857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i885, <4 x float> %i873) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i887 = shufflevector <8 x half> %i857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i887, <4 x float> %i877) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i889 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i885, <4 x float> %i881) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i887, <4 x float> %i884) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i891 = shufflevector <8 x half> %i859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i891, <4 x float> %i886) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i893 = shufflevector <8 x half> %i859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i893, <4 x float> %i888) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i891, <4 x float> %i889) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i893, <4 x float> %i890) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i897 = shufflevector <8 x half> %i861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i897, <4 x float> %i892) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i899 = shufflevector <8 x half> %i861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i899, <4 x float> %i894) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i897, <4 x float> %i895) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i899, <4 x float> %i896) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i903 = shufflevector <8 x half> %i863, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i903, <4 x float> %i898) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i905 = shufflevector <8 x half> %i863, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i905, <4 x float> %i900) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i903, <4 x float> %i901) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i905, <4 x float> %i902) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i909 = shufflevector <8 x half> %i865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i909, <4 x float> %i904) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i911 = shufflevector <8 x half> %i865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i911, <4 x float> %i906) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i909, <4 x float> %i907) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i911, <4 x float> %i908) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i915 = shufflevector <8 x half> %i867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i915, <4 x float> %i910) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i917 = shufflevector <8 x half> %i867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i917, <4 x float> %i912) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i915, <4 x float> %i913) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i917, <4 x float> %i914) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i921 = bitcast <4 x i32> %i868 to <8 x half>, !dbg !99
  %i922 = shufflevector <8 x half> %i921, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i922, <4 x float> %i916) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i924 = shufflevector <8 x half> %i921, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i924, <4 x float> %i918) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i922, <4 x float> %i919) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i924, <4 x float> %i920) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !88
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !88
  fence syncscope("workgroup") release, !dbg !88
  call void @llvm.amdgcn.s.barrier(), !dbg !88
  fence syncscope("workgroup") acquire, !dbg !88
  %i1020 = add nuw nsw i64 %i87, 1, !dbg !62
  %k0.has.next = icmp ult i64 %i1020, 32
  br i1 %k0.has.next, label %k0.issue, label %k0.join

k0.issue:                                         ; preds = %lazy.join
  %k0.next.i98 = shl nuw nsw i64 %i1020, 14, !dbg !64
  %k0.next.i99 = or disjoint i64 %i97, %k0.next.i98, !dbg !64
  %k0.next.kdma.source1516 = or disjoint i64 %k0.next.i99, %kdma.source1515, !dbg !64
  %k0.next.kdma.source1517 = trunc nuw i64 %k0.next.kdma.source1516 to i32, !dbg !64
  %k0.next.kdma.source1518 = shl nuw nsw i32 %k0.next.kdma.source1517, 1, !dbg !64
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1518, %kdma.intra, !dbg !64
  %k0.next.i104 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %k0.next.kdma.source1530 = or disjoint i64 %k0.next.i99, %kdma.source1529, !dbg !64
  %k0.next.kdma.source1531 = trunc nuw i64 %k0.next.kdma.source1530 to i32, !dbg !64
  %k0.next.kdma.source1532 = shl nuw nsw i32 %k0.next.kdma.source1531, 1, !dbg !64
  %k0.next.kdma.offset1533 = or disjoint i32 %k0.next.kdma.source1532, %kdma.intra.hi, !dbg !64
  %k0.next.i105 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %k0.next.kdma.offset1533, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %k0.next.kdma.source1547 = or disjoint i64 %k0.next.i99, %kdma.source1546, !dbg !64
  %k0.next.kdma.source1548 = trunc nuw i64 %k0.next.kdma.source1547 to i32, !dbg !64
  %k0.next.kdma.source1549 = shl nuw nsw i32 %k0.next.kdma.source1548, 1, !dbg !64
  %k0.next.kdma.offset1550 = or disjoint i32 %k0.next.kdma.source1549, %kdma.intra, !dbg !64
  %k0.next.i107 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %k0.next.kdma.offset1550, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  %k0.next.kdma.source1564 = or disjoint i64 %k0.next.i99, %kdma.source1563, !dbg !64
  %k0.next.kdma.source1565 = trunc nuw i64 %k0.next.kdma.source1564 to i32, !dbg !64
  %k0.next.kdma.source1566 = shl nuw nsw i32 %k0.next.kdma.source1565, 1, !dbg !64
  %k0.next.kdma.offset1567 = or disjoint i32 %k0.next.kdma.source1566, %kdma.intra.hi, !dbg !64
  %k0.next.i108 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %k0.next.kdma.offset1567, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !64
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %lazy.join
  %i928 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i929 = bitcast <4 x i32> %i928 to <8 x half>, !dbg !99
  %i930 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i931 = bitcast <4 x i32> %i930 to <8 x half>, !dbg !99
  %i932 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i933 = bitcast <4 x i32> %i932 to <8 x half>, !dbg !99
  %i934 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i935 = bitcast <4 x i32> %i934 to <8 x half>, !dbg !99
  %i936 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i937 = bitcast <4 x i32> %i936 to <8 x half>, !dbg !99
  %i938 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i939 = bitcast <4 x i32> %i938 to <8 x half>, !dbg !99
  %i940 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i941 = bitcast <4 x i32> %i940 to <8 x half>, !dbg !99
  %i942 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !99, !frisk.v_double_buffer !100
  %i943 = shufflevector <8 x half> %i929, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  %i944 = extractvalue [2 x <8 x float>] %lazy.O.i93, 0, !dbg !94
  %i945 = shufflevector <8 x float> %i944, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i943, <4 x float> %i945) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i948 = shufflevector <8 x half> %i929, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  %i949 = shufflevector <8 x float> %i944, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i951 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i948, <4 x float> %i949) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !94
  %i952 = extractvalue [2 x <8 x float>] %lazy.O.i93, 1, !dbg !94
  %i953 = shufflevector <8 x float> %i952, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i943, <4 x float> %i953) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !94
  %i956 = shufflevector <8 x float> %i952, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !94
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i948, <4 x float> %i956) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i959 = shufflevector <8 x half> %i931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i959, <4 x float> %i947) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i961 = shufflevector <8 x half> %i931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i961, <4 x float> %i951) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i959, <4 x float> %i955) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i961, <4 x float> %i958) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i965 = shufflevector <8 x half> %i933, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i965, <4 x float> %i960) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i967 = shufflevector <8 x half> %i933, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i967, <4 x float> %i962) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i965, <4 x float> %i963) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i967, <4 x float> %i964) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i971 = shufflevector <8 x half> %i935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i971, <4 x float> %i966) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i973 = shufflevector <8 x half> %i935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i973, <4 x float> %i968) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i971, <4 x float> %i969) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i973, <4 x float> %i970) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i977 = shufflevector <8 x half> %i937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i977, <4 x float> %i972) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i979 = shufflevector <8 x half> %i937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i979, <4 x float> %i974) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i977, <4 x float> %i975) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i979, <4 x float> %i976) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i983 = shufflevector <8 x half> %i939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i983, <4 x float> %i978) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i985 = shufflevector <8 x half> %i939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i985, <4 x float> %i980) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i983, <4 x float> %i981) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i985, <4 x float> %i982) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i989 = shufflevector <8 x half> %i941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i989, <4 x float> %i984) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i991 = shufflevector <8 x half> %i941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i991, <4 x float> %i986) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i989, <4 x float> %i987) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i991, <4 x float> %i988) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i995 = bitcast <4 x i32> %i942 to <8 x half>, !dbg !99
  %i996 = shufflevector <8 x half> %i995, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i996, <4 x float> %i990) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i998 = shufflevector <8 x half> %i995, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i998, <4 x float> %i992) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i1000 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i996, <4 x float> %i993) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !99
  %i1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i998, <4 x float> %i994) #9, !dbg !99, !frisk.pv_interleave !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !62
  %i1002 = shufflevector <4 x float> %i771, <4 x float> %i773, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %i1003 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1002, 0, !dbg !62
  %i1004 = shufflevector <4 x float> %i774, <4 x float> %i775, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %i1005 = insertvalue [2 x <8 x float>] %i1003, <8 x float> %i1004, 1, !dbg !62
  %i1008 = shufflevector <4 x float> %i847, <4 x float> %i849, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %i1009 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1008, 0, !dbg !62
  %i1010 = shufflevector <4 x float> %i850, <4 x float> %i851, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %i1011 = insertvalue [2 x <8 x float>] %i1009, <8 x float> %i1010, 1, !dbg !62
  %i1012 = shufflevector <4 x float> %i923, <4 x float> %i925, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %i1013 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1012, 0, !dbg !62
  %i1014 = shufflevector <4 x float> %i926, <4 x float> %i927, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %i1015 = insertvalue [2 x <8 x float>] %i1013, <8 x float> %i1014, 1, !dbg !62
  %i1016 = shufflevector <4 x float> %i997, <4 x float> %i999, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %i1017 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1016, 0, !dbg !62
  %i1018 = shufflevector <4 x float> %i1000, <4 x float> %i1001, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %i1019 = insertvalue [2 x <8 x float>] %i1017, <8 x float> %i1018, 1, !dbg !62
  fence syncscope("workgroup") release, !dbg !62
  call void @llvm.amdgcn.s.barrier(), !dbg !62
  fence syncscope("workgroup") acquire, !dbg !62
  br label %bb86, !dbg !62

bb1021:                                           ; preds = %bb86
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %i89, 1, !dbg !62
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !62
  %lane.sum.lane.lo186 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !62
  %lane.sum.lane187 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo186), !dbg !62
  %i1022 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !62
  %i1023 = xor i32 %i1022, 64, !dbg !62
  %i1024 = bitcast float %lane.sum.local185 to i32, !dbg !62
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1023, i32 %i1024), !dbg !62
  %i1025 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !62
  %lane.sum.final189 = fadd float %lane.sum.local185, %i1025, !dbg !62
  %i1026 = xor i32 %i1022, 128, !dbg !62
  %i1027 = bitcast float %lane.sum.final189 to i32, !dbg !62
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1026, i32 %i1027), !dbg !62
  %i1028 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !62
  %lane.sum.final191 = fadd float %lane.sum.final189, %i1028, !dbg !62
  %i1029 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !62
  %lane.sum.field = extractvalue [2 x <1 x float>] %i89, 0, !dbg !62
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !62
  %i1030 = bitcast float %lane.sum.local to i32, !dbg !62
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1023, i32 %i1030), !dbg !62
  %i1031 = bitcast i32 %lane.sum.shuffle to float, !dbg !62
  %lane.sum.final = fadd float %lane.sum.local, %i1031, !dbg !62
  %i1032 = bitcast float %lane.sum.final to i32, !dbg !62
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1026, i32 %i1032), !dbg !62
  %i1033 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !62
  %lane.sum.final183 = fadd float %lane.sum.final, %i1033, !dbg !62
  %i1034 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !62
  %i1035 = and i32 %kdma.tid, 16, !dbg !62
  %.not = icmp eq i32 %i1035, 0, !dbg !62
  %i1036 = and i32 %kdma.tid, 32, !dbg !62
  %.not36 = icmp eq i32 %i1036, 0, !dbg !62
  %i1037 = extractvalue [2 x <8 x float>] %i93, 0, !dbg !62
  %i1038 = extractelement <8 x float> %i1037, i64 0, !dbg !62
  %i1039 = extractelement <8 x float> %i1037, i64 1, !dbg !62
  %i1040 = extractelement <8 x float> %i1037, i64 2, !dbg !62
  %i1041 = extractelement <8 x float> %i1037, i64 3, !dbg !62
  %i1042 = select i1 %.not, float %i1039, float %i1038, !dbg !62
  %i1043 = add nuw nsw i32 %lane.sum.lane187, 64, !dbg !62
  %i1044 = xor i32 %lane.sum.lane187, 16, !dbg !62
  %i1045 = and i32 %i1043, 192, !dbg !62
  %i1046 = icmp samesign ult i32 %i1044, %i1045, !dbg !62
  %i1047 = select i1 %i1046, i32 %i1044, i32 %lane.sum.lane187, !dbg !62
  %i1048 = shl nuw nsw i32 %i1047, 2, !dbg !62
  %i1049 = bitcast float %i1042 to i32, !dbg !62
  %i1050 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1049), !dbg !62
  %i1051 = bitcast i32 %i1050 to float, !dbg !62
  %i1052 = select i1 %.not, float %i1038, float %i1051, !dbg !62
  %i1053 = select i1 %.not, float %i1051, float %i1039, !dbg !62
  %i1054 = select i1 %.not, float %i1041, float %i1040, !dbg !62
  %i1055 = bitcast float %i1054 to i32, !dbg !62
  %i1056 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1055), !dbg !62
  %i1057 = bitcast i32 %i1056 to float, !dbg !62
  %i1058 = select i1 %.not, float %i1040, float %i1057, !dbg !62
  %i1059 = select i1 %.not, float %i1057, float %i1041, !dbg !62
  %i1060 = select i1 %.not36, float %i1058, float %i1052, !dbg !62
  %i1061 = xor i32 %lane.sum.lane187, 32, !dbg !62
  %i1062 = icmp samesign ult i32 %i1061, %i1045, !dbg !62
  %i1063 = select i1 %i1062, i32 %i1061, i32 %lane.sum.lane187, !dbg !62
  %i1064 = shl nuw nsw i32 %i1063, 2, !dbg !62
  %i1065 = bitcast float %i1060 to i32, !dbg !62
  %i1066 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1065), !dbg !62
  %i1067 = bitcast i32 %i1066 to float, !dbg !62
  %i1068 = select i1 %.not36, float %i1052, float %i1067, !dbg !62
  %i1069 = select i1 %.not36, float %i1067, float %i1058, !dbg !62
  %i1070 = select i1 %.not36, float %i1059, float %i1053, !dbg !62
  %i1071 = bitcast float %i1070 to i32, !dbg !62
  %i1072 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1071), !dbg !62
  %i1073 = bitcast i32 %i1072 to float, !dbg !62
  %i1074 = select i1 %.not36, float %i1053, float %i1073, !dbg !62
  %i1075 = select i1 %.not36, float %i1073, float %i1059, !dbg !62
  %i1076 = extractvalue [2 x <8 x float>] %i92, 0, !dbg !62
  %i1077 = extractelement <8 x float> %i1076, i64 0, !dbg !62
  %i1078 = extractelement <8 x float> %i1076, i64 1, !dbg !62
  %i1079 = extractelement <8 x float> %i1076, i64 2, !dbg !62
  %i1080 = extractelement <8 x float> %i1076, i64 3, !dbg !62
  %i1081 = select i1 %.not, float %i1078, float %i1077, !dbg !62
  %i1082 = bitcast float %i1081 to i32, !dbg !62
  %i1083 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1082), !dbg !62
  %i1084 = bitcast i32 %i1083 to float, !dbg !62
  %i1085 = select i1 %.not, float %i1077, float %i1084, !dbg !62
  %i1086 = select i1 %.not, float %i1084, float %i1078, !dbg !62
  %i1087 = select i1 %.not, float %i1080, float %i1079, !dbg !62
  %i1088 = bitcast float %i1087 to i32, !dbg !62
  %i1089 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1088), !dbg !62
  %i1090 = bitcast i32 %i1089 to float, !dbg !62
  %i1091 = select i1 %.not, float %i1079, float %i1090, !dbg !62
  %i1092 = select i1 %.not, float %i1090, float %i1080, !dbg !62
  %i1093 = select i1 %.not36, float %i1091, float %i1085, !dbg !62
  %i1094 = bitcast float %i1093 to i32, !dbg !62
  %i1095 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1094), !dbg !62
  %i1096 = bitcast i32 %i1095 to float, !dbg !62
  %i1097 = select i1 %.not36, float %i1085, float %i1096, !dbg !62
  %i1098 = select i1 %.not36, float %i1096, float %i1091, !dbg !62
  %i1099 = select i1 %.not36, float %i1092, float %i1086, !dbg !62
  %i1100 = bitcast float %i1099 to i32, !dbg !62
  %i1101 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1100), !dbg !62
  %i1102 = bitcast i32 %i1101 to float, !dbg !62
  %i1103 = select i1 %.not36, float %i1086, float %i1102, !dbg !62
  %i1104 = select i1 %.not36, float %i1102, float %i1092, !dbg !62
  %i1105 = extractvalue [2 x <8 x float>] %i91, 0, !dbg !62
  %i1106 = extractelement <8 x float> %i1105, i64 0, !dbg !62
  %i1107 = extractelement <8 x float> %i1105, i64 1, !dbg !62
  %i1108 = extractelement <8 x float> %i1105, i64 2, !dbg !62
  %i1109 = extractelement <8 x float> %i1105, i64 3, !dbg !62
  %i1110 = select i1 %.not, float %i1107, float %i1106, !dbg !62
  %i1111 = bitcast float %i1110 to i32, !dbg !62
  %i1112 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1111), !dbg !62
  %i1113 = bitcast i32 %i1112 to float, !dbg !62
  %i1114 = select i1 %.not, float %i1106, float %i1113, !dbg !62
  %i1115 = select i1 %.not, float %i1113, float %i1107, !dbg !62
  %i1116 = select i1 %.not, float %i1109, float %i1108, !dbg !62
  %i1117 = bitcast float %i1116 to i32, !dbg !62
  %i1118 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1117), !dbg !62
  %i1119 = bitcast i32 %i1118 to float, !dbg !62
  %i1120 = select i1 %.not, float %i1108, float %i1119, !dbg !62
  %i1121 = select i1 %.not, float %i1119, float %i1109, !dbg !62
  %i1122 = select i1 %.not36, float %i1120, float %i1114, !dbg !62
  %i1123 = bitcast float %i1122 to i32, !dbg !62
  %i1124 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1123), !dbg !62
  %i1125 = bitcast i32 %i1124 to float, !dbg !62
  %i1126 = select i1 %.not36, float %i1114, float %i1125, !dbg !62
  %i1127 = select i1 %.not36, float %i1125, float %i1120, !dbg !62
  %i1128 = select i1 %.not36, float %i1121, float %i1115, !dbg !62
  %i1129 = bitcast float %i1128 to i32, !dbg !62
  %i1130 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1129), !dbg !62
  %i1131 = bitcast i32 %i1130 to float, !dbg !62
  %i1132 = select i1 %.not36, float %i1115, float %i1131, !dbg !62
  %i1133 = select i1 %.not36, float %i1131, float %i1121, !dbg !62
  %i1134 = extractvalue [2 x <8 x float>] %i88, 0, !dbg !62
  %i1135 = extractelement <8 x float> %i1134, i64 0, !dbg !62
  %i1136 = extractelement <8 x float> %i1134, i64 1, !dbg !62
  %i1137 = extractelement <8 x float> %i1134, i64 2, !dbg !62
  %i1138 = extractelement <8 x float> %i1134, i64 3, !dbg !62
  %i1139 = select i1 %.not, float %i1136, float %i1135, !dbg !62
  %i1140 = bitcast float %i1139 to i32, !dbg !62
  %i1141 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1140), !dbg !62
  %i1142 = bitcast i32 %i1141 to float, !dbg !62
  %i1143 = select i1 %.not, float %i1135, float %i1142, !dbg !62
  %i1144 = select i1 %.not, float %i1142, float %i1136, !dbg !62
  %i1145 = select i1 %.not, float %i1138, float %i1137, !dbg !62
  %i1146 = bitcast float %i1145 to i32, !dbg !62
  %i1147 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1146), !dbg !62
  %i1148 = bitcast i32 %i1147 to float, !dbg !62
  %i1149 = select i1 %.not, float %i1137, float %i1148, !dbg !62
  %i1150 = select i1 %.not, float %i1148, float %i1138, !dbg !62
  %i1151 = select i1 %.not36, float %i1149, float %i1143, !dbg !62
  %i1152 = bitcast float %i1151 to i32, !dbg !62
  %i1153 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1152), !dbg !62
  %i1154 = select i1 %.not36, float %i1150, float %i1144, !dbg !62
  %i1155 = bitcast float %i1154 to i32, !dbg !62
  %i1156 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1155), !dbg !62
  %i1157 = shufflevector <1 x float> %i1034, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !101
  %i1158 = extractelement <8 x float> %i1037, i64 4, !dbg !62
  %i1159 = extractelement <8 x float> %i1037, i64 5, !dbg !62
  %i1160 = extractelement <8 x float> %i1037, i64 6, !dbg !62
  %i1161 = extractelement <8 x float> %i1037, i64 7, !dbg !62
  %i1162 = select i1 %.not, float %i1159, float %i1158, !dbg !62
  %i1163 = bitcast float %i1162 to i32, !dbg !62
  %i1164 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1163), !dbg !62
  %i1165 = bitcast i32 %i1164 to float, !dbg !62
  %i1166 = select i1 %.not, float %i1158, float %i1165, !dbg !62
  %i1167 = select i1 %.not, float %i1165, float %i1159, !dbg !62
  %i1168 = select i1 %.not, float %i1161, float %i1160, !dbg !62
  %i1169 = bitcast float %i1168 to i32, !dbg !62
  %i1170 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1169), !dbg !62
  %i1171 = bitcast i32 %i1170 to float, !dbg !62
  %i1172 = select i1 %.not, float %i1160, float %i1171, !dbg !62
  %i1173 = select i1 %.not, float %i1171, float %i1161, !dbg !62
  %i1174 = select i1 %.not36, float %i1172, float %i1166, !dbg !62
  %i1175 = bitcast float %i1174 to i32, !dbg !62
  %i1176 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1175), !dbg !62
  %i1177 = bitcast i32 %i1176 to float, !dbg !62
  %i1178 = select i1 %.not36, float %i1166, float %i1177, !dbg !62
  %i1179 = select i1 %.not36, float %i1177, float %i1172, !dbg !62
  %i1180 = select i1 %.not36, float %i1173, float %i1167, !dbg !62
  %i1181 = bitcast float %i1180 to i32, !dbg !62
  %i1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1181), !dbg !62
  %i1183 = bitcast i32 %i1182 to float, !dbg !62
  %i1184 = select i1 %.not36, float %i1167, float %i1183, !dbg !62
  %i1185 = select i1 %.not36, float %i1183, float %i1173, !dbg !62
  %i1186 = extractelement <8 x float> %i1076, i64 4, !dbg !62
  %i1187 = extractelement <8 x float> %i1076, i64 5, !dbg !62
  %i1188 = extractelement <8 x float> %i1076, i64 6, !dbg !62
  %i1189 = extractelement <8 x float> %i1076, i64 7, !dbg !62
  %i1190 = select i1 %.not, float %i1187, float %i1186, !dbg !62
  %i1191 = bitcast float %i1190 to i32, !dbg !62
  %i1192 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1191), !dbg !62
  %i1193 = bitcast i32 %i1192 to float, !dbg !62
  %i1194 = select i1 %.not, float %i1186, float %i1193, !dbg !62
  %i1195 = select i1 %.not, float %i1193, float %i1187, !dbg !62
  %i1196 = select i1 %.not, float %i1189, float %i1188, !dbg !62
  %i1197 = bitcast float %i1196 to i32, !dbg !62
  %i1198 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1197), !dbg !62
  %i1199 = bitcast i32 %i1198 to float, !dbg !62
  %i1200 = select i1 %.not, float %i1188, float %i1199, !dbg !62
  %i1201 = select i1 %.not, float %i1199, float %i1189, !dbg !62
  %i1202 = select i1 %.not36, float %i1200, float %i1194, !dbg !62
  %i1203 = bitcast float %i1202 to i32, !dbg !62
  %i1204 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1203), !dbg !62
  %i1205 = bitcast i32 %i1204 to float, !dbg !62
  %i1206 = select i1 %.not36, float %i1194, float %i1205, !dbg !62
  %i1207 = select i1 %.not36, float %i1205, float %i1200, !dbg !62
  %i1208 = select i1 %.not36, float %i1201, float %i1195, !dbg !62
  %i1209 = bitcast float %i1208 to i32, !dbg !62
  %i1210 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1209), !dbg !62
  %i1211 = bitcast i32 %i1210 to float, !dbg !62
  %i1212 = select i1 %.not36, float %i1195, float %i1211, !dbg !62
  %i1213 = select i1 %.not36, float %i1211, float %i1201, !dbg !62
  %i1214 = extractelement <8 x float> %i1105, i64 4, !dbg !62
  %i1215 = extractelement <8 x float> %i1105, i64 5, !dbg !62
  %i1216 = extractelement <8 x float> %i1105, i64 6, !dbg !62
  %i1217 = extractelement <8 x float> %i1105, i64 7, !dbg !62
  %i1218 = select i1 %.not, float %i1215, float %i1214, !dbg !62
  %i1219 = bitcast float %i1218 to i32, !dbg !62
  %i1220 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1219), !dbg !62
  %i1221 = bitcast i32 %i1220 to float, !dbg !62
  %i1222 = select i1 %.not, float %i1214, float %i1221, !dbg !62
  %i1223 = select i1 %.not, float %i1221, float %i1215, !dbg !62
  %i1224 = select i1 %.not, float %i1217, float %i1216, !dbg !62
  %i1225 = bitcast float %i1224 to i32, !dbg !62
  %i1226 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1225), !dbg !62
  %i1227 = bitcast i32 %i1226 to float, !dbg !62
  %i1228 = select i1 %.not, float %i1216, float %i1227, !dbg !62
  %i1229 = select i1 %.not, float %i1227, float %i1217, !dbg !62
  %i1230 = select i1 %.not36, float %i1228, float %i1222, !dbg !62
  %i1231 = bitcast float %i1230 to i32, !dbg !62
  %i1232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1231), !dbg !62
  %i1233 = bitcast i32 %i1232 to float, !dbg !62
  %i1234 = select i1 %.not36, float %i1222, float %i1233, !dbg !62
  %i1235 = select i1 %.not36, float %i1233, float %i1228, !dbg !62
  %i1236 = select i1 %.not36, float %i1229, float %i1223, !dbg !62
  %i1237 = bitcast float %i1236 to i32, !dbg !62
  %i1238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1237), !dbg !62
  %i1239 = bitcast i32 %i1238 to float, !dbg !62
  %i1240 = select i1 %.not36, float %i1223, float %i1239, !dbg !62
  %i1241 = select i1 %.not36, float %i1239, float %i1229, !dbg !62
  %i1242 = extractelement <8 x float> %i1134, i64 4, !dbg !62
  %i1243 = extractelement <8 x float> %i1134, i64 5, !dbg !62
  %i1244 = extractelement <8 x float> %i1134, i64 6, !dbg !62
  %i1245 = extractelement <8 x float> %i1134, i64 7, !dbg !62
  %i1246 = select i1 %.not, float %i1243, float %i1242, !dbg !62
  %i1247 = bitcast float %i1246 to i32, !dbg !62
  %i1248 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1247), !dbg !62
  %i1249 = bitcast i32 %i1248 to float, !dbg !62
  %i1250 = select i1 %.not, float %i1242, float %i1249, !dbg !62
  %i1251 = select i1 %.not, float %i1249, float %i1243, !dbg !62
  %i1252 = select i1 %.not, float %i1245, float %i1244, !dbg !62
  %i1253 = bitcast float %i1252 to i32, !dbg !62
  %i1254 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1253), !dbg !62
  %i1255 = bitcast i32 %i1254 to float, !dbg !62
  %i1256 = select i1 %.not, float %i1244, float %i1255, !dbg !62
  %i1257 = select i1 %.not, float %i1255, float %i1245, !dbg !62
  %i1258 = select i1 %.not36, float %i1256, float %i1250, !dbg !62
  %i1259 = bitcast float %i1258 to i32, !dbg !62
  %i1260 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1259), !dbg !62
  %i1261 = select i1 %.not36, float %i1257, float %i1251, !dbg !62
  %i1262 = bitcast float %i1261 to i32, !dbg !62
  %i1263 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1262), !dbg !62
  %i1264 = extractvalue [2 x <8 x float>] %i93, 1, !dbg !62
  %i1265 = extractelement <8 x float> %i1264, i64 0, !dbg !62
  %i1266 = extractelement <8 x float> %i1264, i64 1, !dbg !62
  %i1267 = extractelement <8 x float> %i1264, i64 2, !dbg !62
  %i1268 = extractelement <8 x float> %i1264, i64 3, !dbg !62
  %i1269 = select i1 %.not, float %i1266, float %i1265, !dbg !62
  %i1270 = bitcast float %i1269 to i32, !dbg !62
  %i1271 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1270), !dbg !62
  %i1272 = bitcast i32 %i1271 to float, !dbg !62
  %i1273 = select i1 %.not, float %i1265, float %i1272, !dbg !62
  %i1274 = select i1 %.not, float %i1272, float %i1266, !dbg !62
  %i1275 = select i1 %.not, float %i1268, float %i1267, !dbg !62
  %i1276 = bitcast float %i1275 to i32, !dbg !62
  %i1277 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1276), !dbg !62
  %i1278 = bitcast i32 %i1277 to float, !dbg !62
  %i1279 = select i1 %.not, float %i1267, float %i1278, !dbg !62
  %i1280 = select i1 %.not, float %i1278, float %i1268, !dbg !62
  %i1281 = select i1 %.not36, float %i1279, float %i1273, !dbg !62
  %i1282 = bitcast float %i1281 to i32, !dbg !62
  %i1283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1282), !dbg !62
  %i1284 = bitcast i32 %i1283 to float, !dbg !62
  %i1285 = select i1 %.not36, float %i1273, float %i1284, !dbg !62
  %i1286 = select i1 %.not36, float %i1284, float %i1279, !dbg !62
  %i1287 = select i1 %.not36, float %i1280, float %i1274, !dbg !62
  %i1288 = bitcast float %i1287 to i32, !dbg !62
  %i1289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1288), !dbg !62
  %i1290 = bitcast i32 %i1289 to float, !dbg !62
  %i1291 = select i1 %.not36, float %i1274, float %i1290, !dbg !62
  %i1292 = select i1 %.not36, float %i1290, float %i1280, !dbg !62
  %i1293 = extractvalue [2 x <8 x float>] %i92, 1, !dbg !62
  %i1294 = extractelement <8 x float> %i1293, i64 0, !dbg !62
  %i1295 = extractelement <8 x float> %i1293, i64 1, !dbg !62
  %i1296 = extractelement <8 x float> %i1293, i64 2, !dbg !62
  %i1297 = extractelement <8 x float> %i1293, i64 3, !dbg !62
  %i1298 = select i1 %.not, float %i1295, float %i1294, !dbg !62
  %i1299 = bitcast float %i1298 to i32, !dbg !62
  %i1300 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1299), !dbg !62
  %i1301 = bitcast i32 %i1300 to float, !dbg !62
  %i1302 = select i1 %.not, float %i1294, float %i1301, !dbg !62
  %i1303 = select i1 %.not, float %i1301, float %i1295, !dbg !62
  %i1304 = select i1 %.not, float %i1297, float %i1296, !dbg !62
  %i1305 = bitcast float %i1304 to i32, !dbg !62
  %i1306 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1305), !dbg !62
  %i1307 = bitcast i32 %i1306 to float, !dbg !62
  %i1308 = select i1 %.not, float %i1296, float %i1307, !dbg !62
  %i1309 = select i1 %.not, float %i1307, float %i1297, !dbg !62
  %i1310 = select i1 %.not36, float %i1308, float %i1302, !dbg !62
  %i1311 = bitcast float %i1310 to i32, !dbg !62
  %i1312 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1311), !dbg !62
  %i1313 = bitcast i32 %i1312 to float, !dbg !62
  %i1314 = select i1 %.not36, float %i1302, float %i1313, !dbg !62
  %i1315 = select i1 %.not36, float %i1313, float %i1308, !dbg !62
  %i1316 = select i1 %.not36, float %i1309, float %i1303, !dbg !62
  %i1317 = bitcast float %i1316 to i32, !dbg !62
  %i1318 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1317), !dbg !62
  %i1319 = bitcast i32 %i1318 to float, !dbg !62
  %i1320 = select i1 %.not36, float %i1303, float %i1319, !dbg !62
  %i1321 = select i1 %.not36, float %i1319, float %i1309, !dbg !62
  %i1322 = extractvalue [2 x <8 x float>] %i91, 1, !dbg !62
  %i1323 = extractelement <8 x float> %i1322, i64 0, !dbg !62
  %i1324 = extractelement <8 x float> %i1322, i64 1, !dbg !62
  %i1325 = extractelement <8 x float> %i1322, i64 2, !dbg !62
  %i1326 = extractelement <8 x float> %i1322, i64 3, !dbg !62
  %i1327 = select i1 %.not, float %i1324, float %i1323, !dbg !62
  %i1328 = bitcast float %i1327 to i32, !dbg !62
  %i1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1328), !dbg !62
  %i1330 = bitcast i32 %i1329 to float, !dbg !62
  %i1331 = select i1 %.not, float %i1323, float %i1330, !dbg !62
  %i1332 = select i1 %.not, float %i1330, float %i1324, !dbg !62
  %i1333 = select i1 %.not, float %i1326, float %i1325, !dbg !62
  %i1334 = bitcast float %i1333 to i32, !dbg !62
  %i1335 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1334), !dbg !62
  %i1336 = bitcast i32 %i1335 to float, !dbg !62
  %i1337 = select i1 %.not, float %i1325, float %i1336, !dbg !62
  %i1338 = select i1 %.not, float %i1336, float %i1326, !dbg !62
  %i1339 = select i1 %.not36, float %i1337, float %i1331, !dbg !62
  %i1340 = bitcast float %i1339 to i32, !dbg !62
  %i1341 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1340), !dbg !62
  %i1342 = bitcast i32 %i1341 to float, !dbg !62
  %i1343 = select i1 %.not36, float %i1331, float %i1342, !dbg !62
  %i1344 = select i1 %.not36, float %i1342, float %i1337, !dbg !62
  %i1345 = select i1 %.not36, float %i1338, float %i1332, !dbg !62
  %i1346 = bitcast float %i1345 to i32, !dbg !62
  %i1347 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1346), !dbg !62
  %i1348 = bitcast i32 %i1347 to float, !dbg !62
  %i1349 = select i1 %.not36, float %i1332, float %i1348, !dbg !62
  %i1350 = select i1 %.not36, float %i1348, float %i1338, !dbg !62
  %i1351 = extractvalue [2 x <8 x float>] %i88, 1, !dbg !62
  %i1352 = extractelement <8 x float> %i1351, i64 0, !dbg !62
  %i1353 = extractelement <8 x float> %i1351, i64 1, !dbg !62
  %i1354 = extractelement <8 x float> %i1351, i64 2, !dbg !62
  %i1355 = extractelement <8 x float> %i1351, i64 3, !dbg !62
  %i1356 = select i1 %.not, float %i1353, float %i1352, !dbg !62
  %i1357 = bitcast float %i1356 to i32, !dbg !62
  %i1358 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1357), !dbg !62
  %i1359 = bitcast i32 %i1358 to float, !dbg !62
  %i1360 = select i1 %.not, float %i1352, float %i1359, !dbg !62
  %i1361 = select i1 %.not, float %i1359, float %i1353, !dbg !62
  %i1362 = select i1 %.not, float %i1355, float %i1354, !dbg !62
  %i1363 = bitcast float %i1362 to i32, !dbg !62
  %i1364 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1363), !dbg !62
  %i1365 = bitcast i32 %i1364 to float, !dbg !62
  %i1366 = select i1 %.not, float %i1354, float %i1365, !dbg !62
  %i1367 = select i1 %.not, float %i1365, float %i1355, !dbg !62
  %i1368 = select i1 %.not36, float %i1366, float %i1360, !dbg !62
  %i1369 = bitcast float %i1368 to i32, !dbg !62
  %i1370 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1369), !dbg !62
  %i1371 = select i1 %.not36, float %i1367, float %i1361, !dbg !62
  %i1372 = bitcast float %i1371 to i32, !dbg !62
  %i1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1372), !dbg !62
  %i1374 = shufflevector <1 x float> %i1029, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !101
  %i1375 = extractelement <8 x float> %i1264, i64 4, !dbg !62
  %i1376 = extractelement <8 x float> %i1264, i64 5, !dbg !62
  %i1377 = extractelement <8 x float> %i1264, i64 6, !dbg !62
  %i1378 = extractelement <8 x float> %i1264, i64 7, !dbg !62
  %i1379 = select i1 %.not, float %i1376, float %i1375, !dbg !62
  %i1380 = bitcast float %i1379 to i32, !dbg !62
  %i1381 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1380), !dbg !62
  %i1382 = bitcast i32 %i1381 to float, !dbg !62
  %i1383 = select i1 %.not, float %i1375, float %i1382, !dbg !62
  %i1384 = select i1 %.not, float %i1382, float %i1376, !dbg !62
  %i1385 = select i1 %.not, float %i1378, float %i1377, !dbg !62
  %i1386 = bitcast float %i1385 to i32, !dbg !62
  %i1387 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1386), !dbg !62
  %i1388 = bitcast i32 %i1387 to float, !dbg !62
  %i1389 = select i1 %.not, float %i1377, float %i1388, !dbg !62
  %i1390 = select i1 %.not, float %i1388, float %i1378, !dbg !62
  %i1391 = select i1 %.not36, float %i1389, float %i1383, !dbg !62
  %i1392 = bitcast float %i1391 to i32, !dbg !62
  %i1393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1392), !dbg !62
  %i1394 = bitcast i32 %i1393 to float, !dbg !62
  %i1395 = select i1 %.not36, float %i1383, float %i1394, !dbg !62
  %i1396 = select i1 %.not36, float %i1394, float %i1389, !dbg !62
  %i1397 = select i1 %.not36, float %i1390, float %i1384, !dbg !62
  %i1398 = bitcast float %i1397 to i32, !dbg !62
  %i1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1398), !dbg !62
  %i1400 = bitcast i32 %i1399 to float, !dbg !62
  %i1401 = select i1 %.not36, float %i1384, float %i1400, !dbg !62
  %i1402 = select i1 %.not36, float %i1400, float %i1390, !dbg !62
  %i1403 = extractelement <8 x float> %i1293, i64 4, !dbg !62
  %i1404 = extractelement <8 x float> %i1293, i64 5, !dbg !62
  %i1405 = extractelement <8 x float> %i1293, i64 6, !dbg !62
  %i1406 = extractelement <8 x float> %i1293, i64 7, !dbg !62
  %i1407 = select i1 %.not, float %i1404, float %i1403, !dbg !62
  %i1408 = bitcast float %i1407 to i32, !dbg !62
  %i1409 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1408), !dbg !62
  %i1410 = bitcast i32 %i1409 to float, !dbg !62
  %i1411 = select i1 %.not, float %i1403, float %i1410, !dbg !62
  %i1412 = select i1 %.not, float %i1410, float %i1404, !dbg !62
  %i1413 = select i1 %.not, float %i1406, float %i1405, !dbg !62
  %i1414 = bitcast float %i1413 to i32, !dbg !62
  %i1415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1414), !dbg !62
  %i1416 = bitcast i32 %i1415 to float, !dbg !62
  %i1417 = select i1 %.not, float %i1405, float %i1416, !dbg !62
  %i1418 = select i1 %.not, float %i1416, float %i1406, !dbg !62
  %i1419 = select i1 %.not36, float %i1417, float %i1411, !dbg !62
  %i1420 = bitcast float %i1419 to i32, !dbg !62
  %i1421 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1420), !dbg !62
  %i1422 = bitcast i32 %i1421 to float, !dbg !62
  %i1423 = select i1 %.not36, float %i1411, float %i1422, !dbg !62
  %i1424 = select i1 %.not36, float %i1422, float %i1417, !dbg !62
  %i1425 = select i1 %.not36, float %i1418, float %i1412, !dbg !62
  %i1426 = bitcast float %i1425 to i32, !dbg !62
  %i1427 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1426), !dbg !62
  %i1428 = bitcast i32 %i1427 to float, !dbg !62
  %i1429 = select i1 %.not36, float %i1412, float %i1428, !dbg !62
  %i1430 = select i1 %.not36, float %i1428, float %i1418, !dbg !62
  %i1431 = extractelement <8 x float> %i1322, i64 4, !dbg !62
  %i1432 = extractelement <8 x float> %i1322, i64 5, !dbg !62
  %i1433 = extractelement <8 x float> %i1322, i64 6, !dbg !62
  %i1434 = extractelement <8 x float> %i1322, i64 7, !dbg !62
  %i1435 = select i1 %.not, float %i1432, float %i1431, !dbg !62
  %i1436 = bitcast float %i1435 to i32, !dbg !62
  %i1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1436), !dbg !62
  %i1438 = bitcast i32 %i1437 to float, !dbg !62
  %i1439 = select i1 %.not, float %i1431, float %i1438, !dbg !62
  %i1440 = select i1 %.not, float %i1438, float %i1432, !dbg !62
  %i1441 = select i1 %.not, float %i1434, float %i1433, !dbg !62
  %i1442 = bitcast float %i1441 to i32, !dbg !62
  %i1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1442), !dbg !62
  %i1444 = bitcast i32 %i1443 to float, !dbg !62
  %i1445 = select i1 %.not, float %i1433, float %i1444, !dbg !62
  %i1446 = select i1 %.not, float %i1444, float %i1434, !dbg !62
  %i1447 = select i1 %.not36, float %i1445, float %i1439, !dbg !62
  %i1448 = bitcast float %i1447 to i32, !dbg !62
  %i1449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1448), !dbg !62
  %i1450 = bitcast i32 %i1449 to float, !dbg !62
  %i1451 = select i1 %.not36, float %i1439, float %i1450, !dbg !62
  %i1452 = select i1 %.not36, float %i1450, float %i1445, !dbg !62
  %i1453 = select i1 %.not36, float %i1446, float %i1440, !dbg !62
  %i1454 = bitcast float %i1453 to i32, !dbg !62
  %i1455 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1454), !dbg !62
  %i1456 = bitcast i32 %i1455 to float, !dbg !62
  %i1457 = select i1 %.not36, float %i1440, float %i1456, !dbg !62
  %i1458 = select i1 %.not36, float %i1456, float %i1446, !dbg !62
  %i1459 = extractelement <8 x float> %i1351, i64 4, !dbg !62
  %i1460 = extractelement <8 x float> %i1351, i64 5, !dbg !62
  %i1461 = extractelement <8 x float> %i1351, i64 6, !dbg !62
  %i1462 = extractelement <8 x float> %i1351, i64 7, !dbg !62
  %i1463 = select i1 %.not, float %i1460, float %i1459, !dbg !62
  %i1464 = bitcast float %i1463 to i32, !dbg !62
  %i1465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1464), !dbg !62
  %i1466 = bitcast i32 %i1465 to float, !dbg !62
  %i1467 = select i1 %.not, float %i1459, float %i1466, !dbg !62
  %i1468 = select i1 %.not, float %i1466, float %i1460, !dbg !62
  %i1469 = select i1 %.not, float %i1462, float %i1461, !dbg !62
  %i1470 = bitcast float %i1469 to i32, !dbg !62
  %i1471 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1470), !dbg !62
  %i1472 = bitcast i32 %i1471 to float, !dbg !62
  %i1473 = select i1 %.not, float %i1461, float %i1472, !dbg !62
  %i1474 = select i1 %.not, float %i1472, float %i1462, !dbg !62
  %i1475 = select i1 %.not36, float %i1473, float %i1467, !dbg !62
  %i1476 = bitcast float %i1475 to i32, !dbg !62
  %i1477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1476), !dbg !62
  %i1478 = select i1 %.not36, float %i1474, float %i1468, !dbg !62
  %i1479 = bitcast float %i1478 to i32, !dbg !62
  %i1480 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1479), !dbg !62
  %i1481 = bitcast i32 %i1477 to float, !dbg !62
  %i1482 = select i1 %.not36, float %i1467, float %i1481, !dbg !62
  %i1483 = insertelement <4 x float> poison, float %i1482, i64 0, !dbg !101
  %i1484 = bitcast i32 %i1480 to float, !dbg !62
  %i1485 = select i1 %.not36, float %i1468, float %i1484, !dbg !62
  %i1486 = insertelement <4 x float> %i1483, float %i1485, i64 1, !dbg !101
  %i1487 = select i1 %.not36, float %i1481, float %i1473, !dbg !62
  %i1488 = insertelement <4 x float> %i1486, float %i1487, i64 2, !dbg !101
  %i1489 = select i1 %.not36, float %i1484, float %i1474, !dbg !62
  %i1490 = insertelement <4 x float> %i1488, float %i1489, i64 3, !dbg !101
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %i1374, !dbg !101
  %normalized = fmul <4 x float> %i1490, %row.reciprocal, !dbg !101
  %i1491 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !102
  %i1492 = extractelement <4 x half> %i1491, i64 3, !dbg !102
  %i1493 = extractelement <4 x half> %i1491, i64 2, !dbg !102
  %i1494 = extractelement <4 x half> %i1491, i64 1, !dbg !102
  %i1495 = extractelement <4 x half> %i1491, i64 0, !dbg !102
  %i1496 = bitcast i32 %i1370 to float, !dbg !62
  %i1497 = select i1 %.not36, float %i1360, float %i1496, !dbg !62
  %i1498 = insertelement <4 x float> poison, float %i1497, i64 0, !dbg !101
  %i1499 = bitcast i32 %i1373 to float, !dbg !62
  %i1500 = select i1 %.not36, float %i1361, float %i1499, !dbg !62
  %i1501 = insertelement <4 x float> %i1498, float %i1500, i64 1, !dbg !101
  %i1502 = select i1 %.not36, float %i1496, float %i1366, !dbg !62
  %i1503 = insertelement <4 x float> %i1501, float %i1502, i64 2, !dbg !101
  %i1504 = select i1 %.not36, float %i1499, float %i1367, !dbg !62
  %i1505 = insertelement <4 x float> %i1503, float %i1504, i64 3, !dbg !101
  %normalized37 = fmul <4 x float> %i1505, %row.reciprocal, !dbg !101
  %i1506 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !102
  %i1507 = extractelement <4 x half> %i1506, i64 3, !dbg !102
  %i1508 = extractelement <4 x half> %i1506, i64 2, !dbg !102
  %i1509 = extractelement <4 x half> %i1506, i64 1, !dbg !102
  %i1510 = extractelement <4 x half> %i1506, i64 0, !dbg !102
  %i1511 = bitcast i32 %i1260 to float, !dbg !62
  %i1512 = select i1 %.not36, float %i1250, float %i1511, !dbg !62
  %i1513 = insertelement <4 x float> poison, float %i1512, i64 0, !dbg !101
  %i1514 = bitcast i32 %i1263 to float, !dbg !62
  %i1515 = select i1 %.not36, float %i1251, float %i1514, !dbg !62
  %i1516 = insertelement <4 x float> %i1513, float %i1515, i64 1, !dbg !101
  %i1517 = select i1 %.not36, float %i1511, float %i1256, !dbg !62
  %i1518 = insertelement <4 x float> %i1516, float %i1517, i64 2, !dbg !101
  %i1519 = select i1 %.not36, float %i1514, float %i1257, !dbg !62
  %i1520 = insertelement <4 x float> %i1518, float %i1519, i64 3, !dbg !101
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %i1157, !dbg !101
  %normalized39 = fmul <4 x float> %i1520, %row.reciprocal38, !dbg !101
  %i1521 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !102
  %i1522 = extractelement <4 x half> %i1521, i64 3, !dbg !102
  %i1523 = extractelement <4 x half> %i1521, i64 2, !dbg !102
  %i1524 = extractelement <4 x half> %i1521, i64 1, !dbg !102
  %i1525 = extractelement <4 x half> %i1521, i64 0, !dbg !102
  %i1526 = bitcast i32 %i1153 to float, !dbg !62
  %i1527 = select i1 %.not36, float %i1143, float %i1526, !dbg !62
  %i1528 = insertelement <4 x float> poison, float %i1527, i64 0, !dbg !101
  %i1529 = bitcast i32 %i1156 to float, !dbg !62
  %i1530 = select i1 %.not36, float %i1144, float %i1529, !dbg !62
  %i1531 = insertelement <4 x float> %i1528, float %i1530, i64 1, !dbg !101
  %i1532 = select i1 %.not36, float %i1526, float %i1149, !dbg !62
  %i1533 = insertelement <4 x float> %i1531, float %i1532, i64 2, !dbg !101
  %i1534 = select i1 %.not36, float %i1529, float %i1150, !dbg !62
  %i1535 = insertelement <4 x float> %i1533, float %i1534, i64 3, !dbg !101
  %normalized40 = fmul <4 x float> %i1535, %row.reciprocal38, !dbg !101
  %i1536 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !102
  %i1537 = extractelement <4 x half> %i1536, i64 3, !dbg !102
  %i1538 = extractelement <4 x half> %i1536, i64 2, !dbg !102
  %i1539 = extractelement <4 x half> %i1536, i64 1, !dbg !102
  %i1540 = extractelement <4 x half> %i1536, i64 0, !dbg !102
  %address.thread974 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !103, !range !60, !frisk.register_lifetimes !56
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !59
  %address.local976 = add nuw i64 %i17, %address.local975, !dbg !103
  %i1541 = lshr i64 %address.local975, 2, !dbg !103
  %address.local978 = and i64 %i1541, 48, !dbg !103
  %address.local979 = add i64 %address.local976, %address.local978, !dbg !103
  %address.local980 = and i64 %address.local975, 240, !dbg !103
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !103
  %address.local982 = add i64 %address.local981, %i18, !dbg !103
  %address.local983 = shl nuw nsw i64 %address.local982, 7, !dbg !103
  %i1542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %arg3, i64 %i21, !dbg !103
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %i1542, i64 %address.local983, !dbg !103
  %address.local987 = and i64 %i1541, 12, !dbg !61
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !103
  store half %i1540, ptr addrspace(1) %address.local988, align 2, !dbg !103
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !103
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !103
  store half %i1539, ptr addrspace(1) %address.local991, align 2, !dbg !103
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !103
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !103
  store half %i1538, ptr addrspace(1) %address.local994, align 2, !dbg !103
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !103
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !103
  store half %i1537, ptr addrspace(1) %address.local997, align 2, !dbg !103
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !103
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !103
  store half %i1525, ptr addrspace(1) %address.local1000, align 2, !dbg !103
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !103
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !103
  store half %i1524, ptr addrspace(1) %address.local1003, align 2, !dbg !103
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !103
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !103
  store half %i1523, ptr addrspace(1) %address.local1006, align 2, !dbg !103
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !103
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !103
  store half %i1522, ptr addrspace(1) %address.local1009, align 2, !dbg !103
  %address.local1011 = add nuw i64 %address.local983, 8192, !dbg !103
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %i1542, i64 %address.local1011, !dbg !103
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !103
  store half %i1510, ptr addrspace(1) %address.local1014, align 2, !dbg !103
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !103
  store half %i1509, ptr addrspace(1) %address.local1016, align 2, !dbg !103
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !103
  store half %i1508, ptr addrspace(1) %address.local1018, align 2, !dbg !103
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !103
  store half %i1507, ptr addrspace(1) %address.local1020, align 2, !dbg !103
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !103
  store half %i1495, ptr addrspace(1) %address.local1022, align 2, !dbg !103
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !103
  store half %i1494, ptr addrspace(1) %address.local1024, align 2, !dbg !103
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !103
  store half %i1493, ptr addrspace(1) %address.local1026, align 2, !dbg !103
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !103
  store half %i1492, ptr addrspace(1) %address.local1028, align 2, !dbg !103
  %i1543 = insertelement <4 x float> poison, float %i1451, i64 0, !dbg !101
  %i1544 = insertelement <4 x float> %i1543, float %i1457, i64 1, !dbg !101
  %i1545 = insertelement <4 x float> %i1544, float %i1452, i64 2, !dbg !101
  %i1546 = insertelement <4 x float> %i1545, float %i1458, i64 3, !dbg !101
  %normalized42 = fmul <4 x float> %i1546, %row.reciprocal, !dbg !101
  %i1547 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !102
  %i1548 = extractelement <4 x half> %i1547, i64 3, !dbg !102
  %i1549 = extractelement <4 x half> %i1547, i64 2, !dbg !102
  %i1550 = extractelement <4 x half> %i1547, i64 1, !dbg !102
  %i1551 = extractelement <4 x half> %i1547, i64 0, !dbg !102
  %i1552 = insertelement <4 x float> poison, float %i1343, i64 0, !dbg !101
  %i1553 = insertelement <4 x float> %i1552, float %i1349, i64 1, !dbg !101
  %i1554 = insertelement <4 x float> %i1553, float %i1344, i64 2, !dbg !101
  %i1555 = insertelement <4 x float> %i1554, float %i1350, i64 3, !dbg !101
  %normalized43 = fmul <4 x float> %i1555, %row.reciprocal, !dbg !101
  %i1556 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !102
  %i1557 = extractelement <4 x half> %i1556, i64 3, !dbg !102
  %i1558 = extractelement <4 x half> %i1556, i64 2, !dbg !102
  %i1559 = extractelement <4 x half> %i1556, i64 1, !dbg !102
  %i1560 = extractelement <4 x half> %i1556, i64 0, !dbg !102
  %i1561 = insertelement <4 x float> poison, float %i1234, i64 0, !dbg !101
  %i1562 = insertelement <4 x float> %i1561, float %i1240, i64 1, !dbg !101
  %i1563 = insertelement <4 x float> %i1562, float %i1235, i64 2, !dbg !101
  %i1564 = insertelement <4 x float> %i1563, float %i1241, i64 3, !dbg !101
  %normalized45 = fmul <4 x float> %i1564, %row.reciprocal38, !dbg !101
  %i1565 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !102
  %i1566 = extractelement <4 x half> %i1565, i64 3, !dbg !102
  %i1567 = extractelement <4 x half> %i1565, i64 2, !dbg !102
  %i1568 = extractelement <4 x half> %i1565, i64 1, !dbg !102
  %i1569 = extractelement <4 x half> %i1565, i64 0, !dbg !102
  %i1570 = insertelement <4 x float> poison, float %i1126, i64 0, !dbg !101
  %i1571 = insertelement <4 x float> %i1570, float %i1132, i64 1, !dbg !101
  %i1572 = insertelement <4 x float> %i1571, float %i1127, i64 2, !dbg !101
  %i1573 = insertelement <4 x float> %i1572, float %i1133, i64 3, !dbg !101
  %normalized46 = fmul <4 x float> %i1573, %row.reciprocal38, !dbg !101
  %i1574 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !102
  %i1575 = extractelement <4 x half> %i1574, i64 3, !dbg !102
  %i1576 = extractelement <4 x half> %i1574, i64 2, !dbg !102
  %i1577 = extractelement <4 x half> %i1574, i64 1, !dbg !102
  %i1578 = extractelement <4 x half> %i1574, i64 0, !dbg !102
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !103
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !103
  store half %i1578, ptr addrspace(1) %address.local1044, align 2, !dbg !103
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !103
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !103
  store half %i1577, ptr addrspace(1) %address.local1047, align 2, !dbg !103
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !103
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !103
  store half %i1576, ptr addrspace(1) %address.local1050, align 2, !dbg !103
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !103
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !103
  store half %i1575, ptr addrspace(1) %address.local1053, align 2, !dbg !103
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !103
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !103
  store half %i1569, ptr addrspace(1) %address.local1056, align 2, !dbg !103
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !103
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !103
  store half %i1568, ptr addrspace(1) %address.local1059, align 2, !dbg !103
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !103
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !103
  store half %i1567, ptr addrspace(1) %address.local1062, align 2, !dbg !103
  %address.local1064 = or i64 %i1541, 51, !dbg !103
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !103
  store half %i1566, ptr addrspace(1) %address.local1065, align 2, !dbg !103
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !103
  store half %i1560, ptr addrspace(1) %address.local1070, align 2, !dbg !103
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !103
  store half %i1559, ptr addrspace(1) %address.local1072, align 2, !dbg !103
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !103
  store half %i1558, ptr addrspace(1) %address.local1074, align 2, !dbg !103
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !103
  store half %i1557, ptr addrspace(1) %address.local1076, align 2, !dbg !103
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !103
  store half %i1551, ptr addrspace(1) %address.local1078, align 2, !dbg !103
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !103
  store half %i1550, ptr addrspace(1) %address.local1080, align 2, !dbg !103
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !103
  store half %i1549, ptr addrspace(1) %address.local1082, align 2, !dbg !103
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !103
  store half %i1548, ptr addrspace(1) %address.local1084, align 2, !dbg !103
  %i1579 = insertelement <4 x float> poison, float %i1423, i64 0, !dbg !101
  %i1580 = insertelement <4 x float> %i1579, float %i1429, i64 1, !dbg !101
  %i1581 = insertelement <4 x float> %i1580, float %i1424, i64 2, !dbg !101
  %i1582 = insertelement <4 x float> %i1581, float %i1430, i64 3, !dbg !101
  %normalized48 = fmul <4 x float> %i1582, %row.reciprocal, !dbg !101
  %i1583 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !102
  %i1584 = extractelement <4 x half> %i1583, i64 3, !dbg !102
  %i1585 = extractelement <4 x half> %i1583, i64 2, !dbg !102
  %i1586 = extractelement <4 x half> %i1583, i64 1, !dbg !102
  %i1587 = extractelement <4 x half> %i1583, i64 0, !dbg !102
  %i1588 = insertelement <4 x float> poison, float %i1314, i64 0, !dbg !101
  %i1589 = insertelement <4 x float> %i1588, float %i1320, i64 1, !dbg !101
  %i1590 = insertelement <4 x float> %i1589, float %i1315, i64 2, !dbg !101
  %i1591 = insertelement <4 x float> %i1590, float %i1321, i64 3, !dbg !101
  %normalized49 = fmul <4 x float> %i1591, %row.reciprocal, !dbg !101
  %i1592 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !102
  %i1593 = extractelement <4 x half> %i1592, i64 3, !dbg !102
  %i1594 = extractelement <4 x half> %i1592, i64 2, !dbg !102
  %i1595 = extractelement <4 x half> %i1592, i64 1, !dbg !102
  %i1596 = extractelement <4 x half> %i1592, i64 0, !dbg !102
  %i1597 = insertelement <4 x float> poison, float %i1206, i64 0, !dbg !101
  %i1598 = insertelement <4 x float> %i1597, float %i1212, i64 1, !dbg !101
  %i1599 = insertelement <4 x float> %i1598, float %i1207, i64 2, !dbg !101
  %i1600 = insertelement <4 x float> %i1599, float %i1213, i64 3, !dbg !101
  %normalized51 = fmul <4 x float> %i1600, %row.reciprocal38, !dbg !101
  %i1601 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !102
  %i1602 = extractelement <4 x half> %i1601, i64 3, !dbg !102
  %i1603 = extractelement <4 x half> %i1601, i64 2, !dbg !102
  %i1604 = extractelement <4 x half> %i1601, i64 1, !dbg !102
  %i1605 = extractelement <4 x half> %i1601, i64 0, !dbg !102
  %i1606 = insertelement <4 x float> poison, float %i1097, i64 0, !dbg !101
  %i1607 = insertelement <4 x float> %i1606, float %i1103, i64 1, !dbg !101
  %i1608 = insertelement <4 x float> %i1607, float %i1098, i64 2, !dbg !101
  %i1609 = insertelement <4 x float> %i1608, float %i1104, i64 3, !dbg !101
  %normalized52 = fmul <4 x float> %i1609, %row.reciprocal38, !dbg !101
  %i1610 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !102
  %i1611 = extractelement <4 x half> %i1610, i64 3, !dbg !102
  %i1612 = extractelement <4 x half> %i1610, i64 2, !dbg !102
  %i1613 = extractelement <4 x half> %i1610, i64 1, !dbg !102
  %i1614 = extractelement <4 x half> %i1610, i64 0, !dbg !102
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !103
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !103
  store half %i1614, ptr addrspace(1) %address.local1100, align 2, !dbg !103
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !103
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !103
  store half %i1613, ptr addrspace(1) %address.local1103, align 2, !dbg !103
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !103
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !103
  store half %i1612, ptr addrspace(1) %address.local1106, align 2, !dbg !103
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !103
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !103
  store half %i1611, ptr addrspace(1) %address.local1109, align 2, !dbg !103
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !103
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !103
  store half %i1605, ptr addrspace(1) %address.local1112, align 2, !dbg !103
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !103
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !103
  store half %i1604, ptr addrspace(1) %address.local1115, align 2, !dbg !103
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !103
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !103
  store half %i1603, ptr addrspace(1) %address.local1118, align 2, !dbg !103
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !103
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !103
  store half %i1602, ptr addrspace(1) %address.local1121, align 2, !dbg !103
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !103
  store half %i1596, ptr addrspace(1) %address.local1126, align 2, !dbg !103
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !103
  store half %i1595, ptr addrspace(1) %address.local1128, align 2, !dbg !103
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !103
  store half %i1594, ptr addrspace(1) %address.local1130, align 2, !dbg !103
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !103
  store half %i1593, ptr addrspace(1) %address.local1132, align 2, !dbg !103
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !103
  store half %i1587, ptr addrspace(1) %address.local1134, align 2, !dbg !103
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !103
  store half %i1586, ptr addrspace(1) %address.local1136, align 2, !dbg !103
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !103
  store half %i1585, ptr addrspace(1) %address.local1138, align 2, !dbg !103
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !103
  store half %i1584, ptr addrspace(1) %address.local1140, align 2, !dbg !103
  %i1615 = insertelement <4 x float> poison, float %i1395, i64 0, !dbg !101
  %i1616 = insertelement <4 x float> %i1615, float %i1401, i64 1, !dbg !101
  %i1617 = insertelement <4 x float> %i1616, float %i1396, i64 2, !dbg !101
  %i1618 = insertelement <4 x float> %i1617, float %i1402, i64 3, !dbg !101
  %normalized54 = fmul <4 x float> %i1618, %row.reciprocal, !dbg !101
  %i1619 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !102
  %i1620 = extractelement <4 x half> %i1619, i64 3, !dbg !102
  %i1621 = extractelement <4 x half> %i1619, i64 2, !dbg !102
  %i1622 = extractelement <4 x half> %i1619, i64 1, !dbg !102
  %i1623 = extractelement <4 x half> %i1619, i64 0, !dbg !102
  %i1624 = insertelement <4 x float> poison, float %i1285, i64 0, !dbg !101
  %i1625 = insertelement <4 x float> %i1624, float %i1291, i64 1, !dbg !101
  %i1626 = insertelement <4 x float> %i1625, float %i1286, i64 2, !dbg !101
  %i1627 = insertelement <4 x float> %i1626, float %i1292, i64 3, !dbg !101
  %normalized55 = fmul <4 x float> %i1627, %row.reciprocal, !dbg !101
  %i1628 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !102
  %i1629 = extractelement <4 x half> %i1628, i64 3, !dbg !102
  %i1630 = extractelement <4 x half> %i1628, i64 2, !dbg !102
  %i1631 = extractelement <4 x half> %i1628, i64 1, !dbg !102
  %i1632 = extractelement <4 x half> %i1628, i64 0, !dbg !102
  %i1633 = insertelement <4 x float> poison, float %i1178, i64 0, !dbg !101
  %i1634 = insertelement <4 x float> %i1633, float %i1184, i64 1, !dbg !101
  %i1635 = insertelement <4 x float> %i1634, float %i1179, i64 2, !dbg !101
  %i1636 = insertelement <4 x float> %i1635, float %i1185, i64 3, !dbg !101
  %normalized57 = fmul <4 x float> %i1636, %row.reciprocal38, !dbg !101
  %i1637 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !102
  %i1638 = extractelement <4 x half> %i1637, i64 3, !dbg !102
  %i1639 = extractelement <4 x half> %i1637, i64 2, !dbg !102
  %i1640 = extractelement <4 x half> %i1637, i64 1, !dbg !102
  %i1641 = extractelement <4 x half> %i1637, i64 0, !dbg !102
  %i1642 = insertelement <4 x float> poison, float %i1068, i64 0, !dbg !101
  %i1643 = insertelement <4 x float> %i1642, float %i1074, i64 1, !dbg !101
  %i1644 = insertelement <4 x float> %i1643, float %i1069, i64 2, !dbg !101
  %i1645 = insertelement <4 x float> %i1644, float %i1075, i64 3, !dbg !101
  %normalized58 = fmul <4 x float> %i1645, %row.reciprocal38, !dbg !101
  %i1646 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !102
  %i1647 = extractelement <4 x half> %i1646, i64 3, !dbg !102
  %i1648 = extractelement <4 x half> %i1646, i64 2, !dbg !102
  %i1649 = extractelement <4 x half> %i1646, i64 1, !dbg !102
  %i1650 = extractelement <4 x half> %i1646, i64 0, !dbg !102
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !103
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !103
  store half %i1650, ptr addrspace(1) %address.local1156, align 2, !dbg !103
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !103
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !103
  store half %i1649, ptr addrspace(1) %address.local1159, align 2, !dbg !103
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !103
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !103
  store half %i1648, ptr addrspace(1) %address.local1162, align 2, !dbg !103
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !103
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !103
  store half %i1647, ptr addrspace(1) %address.local1165, align 2, !dbg !103
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !103
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !103
  store half %i1641, ptr addrspace(1) %address.local1168, align 2, !dbg !103
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !103
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !103
  store half %i1640, ptr addrspace(1) %address.local1171, align 2, !dbg !103
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !103
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !103
  store half %i1639, ptr addrspace(1) %address.local1174, align 2, !dbg !103
  %address.local1176 = or i64 %i1541, 115, !dbg !103
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !103
  store half %i1638, ptr addrspace(1) %address.local1177, align 2, !dbg !103
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !103
  store half %i1632, ptr addrspace(1) %address.local1182, align 2, !dbg !103
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !103
  store half %i1631, ptr addrspace(1) %address.local1184, align 2, !dbg !103
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !103
  store half %i1630, ptr addrspace(1) %address.local1186, align 2, !dbg !103
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !103
  store half %i1629, ptr addrspace(1) %address.local1188, align 2, !dbg !103
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !103
  store half %i1623, ptr addrspace(1) %address.local1190, align 2, !dbg !103
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !103
  store half %i1622, ptr addrspace(1) %address.local1192, align 2, !dbg !103
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !103
  store half %i1621, ptr addrspace(1) %address.local1194, align 2, !dbg !103
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !103
  store half %i1620, ptr addrspace(1) %address.local1196, align 2, !dbg !103
  ret void, !dbg !58
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.ballot.i64(i1) #3

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!50}
!llvm.module.flags = !{!52}

!50 = distinct !DICompileUnit(language: DW_LANG_C, file: !51, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!51 = !DIFile(filename: "input.mlir", directory: "generated_ir/noncausal_k_lds_prefetch_pass/mytest_final/run-20261030-171621-98fko2hz/online-vdb-k64-lds2")
!52 = !{i32 2, !"Debug Info Version", i32 3}
!54 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !51, file: !51, line: 23, type: !55, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !50)
!55 = !DISubroutineType(types: !56)
!56 = !{}
!57 = !{i32 256, i32 1, i32 1}
!58 = !DILocation(line: 23, column: 3, scope: !54)
!59 = !DILocation(line: 67, column: 22, scope: !54)
!60 = !{i32 0, i32 256}
!61 = !DILocation(line: 69, column: 24, scope: !54)
!62 = !DILocation(line: 57, column: 24, scope: !54)
!63 = !DILocation(line: 70, column: 26, scope: !54)
!64 = !DILocation(line: 68, column: 22, scope: !54)
!65 = !{i32 0}
!66 = !{i32 1}
!67 = !{i32 2}
!68 = !{i32 3}
!69 = !{i32 4}
!70 = !{i32 5}
!71 = !{i32 6}
!72 = !{i32 7}
!73 = !{i32 8}
!74 = !{i32 9}
!75 = !{i32 10}
!76 = !{i32 11}
!77 = !{i32 12}
!78 = !{i32 13}
!79 = !{i32 14}
!80 = !{i32 15}
!81 = !{i32 16}
!82 = !{i32 17}
!83 = !{i32 18}
!84 = !{i32 19}
!85 = !{i32 20}
!86 = !{i32 21}
!87 = !{i32 22}
!88 = !DILocation(line: 86, column: 20, scope: !54)
!89 = !DILocation(line: 71, column: 23, scope: !54)
!90 = !DILocation(line: 74, column: 20, scope: !54)
!92 = !DILocation(line: 76, column: 23, scope: !54)
!94 = !DILocation(line: 88, column: 22, scope: !54)
!95 = !DILocation(line: 78, column: 22, scope: !54)
!96 = !DILocation(line: 79, column: 22, scope: !54)
!97 = !DILocation(line: 81, column: 16, scope: !54)
!98 = !DILocation(line: 80, column: 20, scope: !54)
!99 = !DILocation(line: 87, column: 23, scope: !54)
!100 = !{i64 8192}
!101 = !DILocation(line: 92, column: 23, scope: !54)
!102 = !DILocation(line: 93, column: 19, scope: !54)
!103 = !DILocation(line: 94, column: 9, scope: !54)
