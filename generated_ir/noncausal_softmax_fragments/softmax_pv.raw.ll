; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_softmax_fragments/control.named.ll'
source_filename = "generated_ir/noncausal_k_lds_prefetch_pass/mytest_final/run-20261030-171621-98fko2hz/online-vdb-k64-lds2/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %arg, ptr addrspace(1) %arg1, ptr addrspace(1) %arg2, ptr addrspace(1) %arg3) #1 !dbg !56 !reqd_work_group_size !59 {
bb:
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !60
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !60
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !60
  %i = and i32 %kdma.lane, 31, !dbg !60
  %kdma.owner = or disjoint i32 %kdma.wave, %i, !dbg !60
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !60
  %i4 = shl nuw nsw i32 %kdma.tid, 3, !dbg !60
  %kdma.intra = and i32 %i4, 8, !dbg !60
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !60
  %i5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !60
  %i6 = sext i32 %i5 to i64, !dbg !60
  %i7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !60
  %i8 = sext i32 %i7 to i64, !dbg !60
  %i9 = shl nsw i64 %i8, 5, !dbg !60
  %i10 = add nsw i64 %i9, %i6, !dbg !60
  %i11 = lshr i64 %i10, 5, !dbg !60
  %i12 = and i64 %i11, 576460752303423456, !dbg !60
  %i13 = and i64 %i6, 31, !dbg !60
  %i14 = or disjoint i64 %i12, %i13, !dbg !60
  %i15 = lshr i64 %i10, 10, !dbg !61
  %i16 = add nuw nsw i64 %i11, %i15, !dbg !61
  %i17 = shl nsw i64 %i14, 7, !dbg !61
  %i18 = mul nsw i64 %i15, -4096, !dbg !61
  %i19 = add i64 %i17, %i18, !dbg !61
  %i20 = shl i64 %i16, 19, !dbg !61
  %i21 = and i64 %i20, 16252928, !dbg !61
  %i22 = shl nsw i64 %i19, 7, !dbg !61
  %i23 = add i64 %i21, %i22, !dbg !61
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !61, !range !62, !frisk.register_lifetimes !58
  %address.local = zext nneg i32 %address.thread to i64, !dbg !61
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !61
  %address.local405 = and i64 %address.local404, 1536, !dbg !61
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !61
  %address.local407 = ptrtoint ptr addrspace(3) %address.local406 to i64, !dbg !61
  %address.local408 = trunc i64 %address.local407 to i32, !dbg !61
  %i24 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !61
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !61
  %address.local410 = and i64 %address.local409, 16256, !dbg !61
  %address.local411 = lshr i64 %address.local, 4, !dbg !61
  %address.local412 = and i64 %address.local411, 8, !dbg !61
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !61
  %i25 = ptrtoint ptr addrspace(1) %arg to i64, !dbg !61
  %.idx1399 = or disjoint i64 %i23, %address.local413, !dbg !61
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !61
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !61
  %i26 = insertelement <2 x i64> poison, i64 %i25, i64 0, !dbg !61
  %i27 = bitcast <2 x i64> %i26 to <4 x i32>, !dbg !61
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %i27, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !61
  %dma.buffer.lds1398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i24), !dbg !61
  %i28 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !61
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !61
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !61
  %address.local417 = and i64 %address.local416, 3584, !dbg !61
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !61
  %address.local419 = ptrtoint ptr addrspace(3) %address.local418 to i64, !dbg !61
  %address.local420 = trunc i64 %address.local419 to i32, !dbg !61
  %i29 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !61
  %address.local423 = lshr i64 %address.local415, 4, !dbg !61
  %address.local424 = and i64 %address.local423, 24, !dbg !61
  %address.local425 = or disjoint i64 %address.local410, %address.local424, !dbg !61
  %.idx1400 = or disjoint i64 %i23, %address.local425, !dbg !61
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !61
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !61
  %dma.buffer.lds1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i29), !dbg !61
  %i30 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !61
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !61
  fence syncscope("workgroup") release, !dbg !61
  call void @llvm.amdgcn.s.barrier(), !dbg !61
  fence syncscope("workgroup") acquire, !dbg !61
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !61
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !61
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !61
  %i31 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !61
  %i32 = or disjoint i64 %i23, 32, !dbg !61
  %i33 = or disjoint i64 %address.local413, %i32, !dbg !61
  %.tr = trunc i64 %i33 to i32, !dbg !61
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !61
  %dma.buffer.lds1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i31), !dbg !61
  %i34 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !61
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !61
  %address.local432 = ptrtoint ptr addrspace(3) %address.local431 to i64, !dbg !61
  %address.local433 = trunc i64 %address.local432 to i32, !dbg !61
  %i35 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !61
  %i36 = or disjoint i64 %address.local425, %i32, !dbg !61
  %.tr1401 = trunc i64 %i36 to i32, !dbg !61
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !61
  %dma.buffer.lds1389 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i35), !dbg !61
  %i37 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !61
  %address.local435 = lshr i32 %address.thread, 2, !dbg !63
  %address.local436 = and i32 %address.local435, 4, !dbg !63
  %address.local437 = shl nuw nsw i32 %address.thread, 1, !dbg !63
  %address.local438 = and i32 %address.local437, 384, !dbg !63
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !63
  %address.local440 = shl nuw nsw i32 %address.thread, 3, !dbg !63
  %address.local441 = and i32 %address.local440, 120, !dbg !63
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !63
  %address.local443 = shl nuw nsw i32 %address.thread, 5, !dbg !63
  %address.local444 = and i32 %address.local443, 1024, !dbg !63
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !63
  %address.local446 = zext nneg i32 %address.local445 to i64, !dbg !63
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local446, !dbg !63
  %i38 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !63
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !63
  %i39 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !63
  %i40 = fmul <4 x half> %i38, splat (half 1.275630e-01), !dbg !63
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !63
  %i41 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !63
  %i42 = fmul <4 x half> %i39, splat (half 1.275630e-01), !dbg !63
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !63
  %i43 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !63
  %i44 = fmul <4 x half> %i41, splat (half 1.275630e-01), !dbg !63
  %i45 = fmul <4 x half> %i43, splat (half 1.275630e-01), !dbg !63
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !61
  fence syncscope("workgroup") release, !dbg !61
  call void @llvm.amdgcn.s.barrier(), !dbg !61
  fence syncscope("workgroup") acquire, !dbg !61
  %i46 = or disjoint i64 %i23, 64, !dbg !61
  %i47 = or disjoint i64 %address.local413, %i46, !dbg !61
  %.tr1402 = trunc i64 %i47 to i32, !dbg !61
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !61
  %i48 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !61
  %i49 = or disjoint i64 %address.local425, %i46, !dbg !61
  %.tr1403 = trunc i64 %i49 to i32, !dbg !61
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !61
  %i50 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !61
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !63
  %i51 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !63
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !63
  %i52 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !63
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !63
  %i53 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !63
  %i54 = fmul <4 x half> %i51, splat (half 1.275630e-01), !dbg !63
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !63
  %i55 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !63
  %i56 = fmul <4 x half> %i52, splat (half 1.275630e-01), !dbg !63
  %i57 = fmul <4 x half> %i53, splat (half 1.275630e-01), !dbg !63
  %i58 = fmul <4 x half> %i55, splat (half 1.275630e-01), !dbg !63
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !61
  fence syncscope("workgroup") release, !dbg !61
  call void @llvm.amdgcn.s.barrier(), !dbg !61
  fence syncscope("workgroup") acquire, !dbg !61
  %i59 = or disjoint i64 %i23, 96, !dbg !61
  %i60 = or disjoint i64 %address.local413, %i59, !dbg !61
  %.tr1404 = trunc i64 %i60 to i32, !dbg !61
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !61
  %i61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !61
  %i62 = or disjoint i64 %address.local425, %i59, !dbg !61
  %.tr1405 = trunc i64 %i62 to i32, !dbg !61
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !61
  %i63 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !61
  %i64 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !63
  %i65 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !63
  %i66 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !63
  %i67 = fmul <4 x half> %i64, splat (half 1.275630e-01), !dbg !63
  %i68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !63
  %i69 = fmul <4 x half> %i65, splat (half 1.275630e-01), !dbg !63
  %i70 = fmul <4 x half> %i66, splat (half 1.275630e-01), !dbg !63
  %i71 = fmul <4 x half> %i68, splat (half 1.275630e-01), !dbg !63
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !61
  fence syncscope("workgroup") release, !dbg !61
  call void @llvm.amdgcn.s.barrier(), !dbg !61
  fence syncscope("workgroup") acquire, !dbg !61
  %i72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !63
  %i73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !63
  %i74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !63
  %i75 = fmul <4 x half> %i72, splat (half 1.275630e-01), !dbg !63
  %i76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !63
  %i77 = fmul <4 x half> %i73, splat (half 1.275630e-01), !dbg !63
  %i78 = fmul <4 x half> %i74, splat (half 1.275630e-01), !dbg !63
  %i79 = fmul <4 x half> %i76, splat (half 1.275630e-01), !dbg !63
  fence syncscope("workgroup") release, !dbg !64
  call void @llvm.amdgcn.s.barrier(), !dbg !64
  fence syncscope("workgroup") acquire, !dbg !64
  %i80 = shl nuw nsw i32 %kdma.tid, 2, !dbg !65
  %i81 = and i32 %i80, 3084, !dbg !65
  %i82 = and i32 %kdma.tid, 1, !dbg !65
  %i83 = or disjoint i32 %i81, %i82, !dbg !65
  %narrow = add nuw nsw i32 %i83, 1, !dbg !65
  %i84 = lshr i32 %narrow, 1, !dbg !65
  %i85 = zext nneg i32 %i84 to i64, !dbg !65
  br label %bb86, !dbg !64

bb86:                                             ; preds = %bb95, %bb
  %i87 = phi i64 [ %i1020, %bb95 ], [ 0, %bb ], !dbg !64
  %i88 = phi [2 x <8 x float>] [ %i1005, %bb95 ], [ zeroinitializer, %bb ], !dbg !64
  %i89 = phi [2 x <1 x float>] [ %i699, %bb95 ], [ zeroinitializer, %bb ], !dbg !64
  %i90 = phi [2 x <1 x float>] [ %i1007, %bb95 ], [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %bb ], !dbg !64
  %i91 = phi [2 x <8 x float>] [ %i1011, %bb95 ], [ zeroinitializer, %bb ], !dbg !64
  %i92 = phi [2 x <8 x float>] [ %i1015, %bb95 ], [ zeroinitializer, %bb ], !dbg !64
  %i93 = phi [2 x <8 x float>] [ %i1019, %bb95 ], [ zeroinitializer, %bb ], !dbg !64
  %i94 = icmp samesign ult i64 %i87, 32, !dbg !64
  br i1 %i94, label %bb95, label %bb1021, !dbg !64

bb95:                                             ; preds = %bb86
  %i96 = shl i64 %i11, 19, !dbg !66
  %i97 = and i64 %i96, 16252928, !dbg !66
  %i98 = shl nuw nsw i64 %i87, 14, !dbg !66
  %i99 = or disjoint i64 %i97, %i98, !dbg !66
  %address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !66, !range !62, !frisk.register_lifetimes !58
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !61
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !61
  %address.local501 = and i64 %address.local500, 1536, !dbg !61
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !61
  %address.local503 = ptrtoint ptr addrspace(3) %address.local502 to i64, !dbg !61
  %address.local504 = trunc i64 %address.local503 to i32, !dbg !61
  %i100 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !66
  %i101 = ptrtoint ptr addrspace(1) %arg1 to i64, !dbg !66
  %i102 = insertelement <2 x i64> poison, i64 %i101, i64 0, !dbg !66
  %i103 = bitcast <2 x i64> %i102 to <4 x i32>, !dbg !66
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %i103, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !66
  %dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i100), !dbg !66
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !66
  %kdma.source1506 = lshr i64 %kdma.source, 4, !dbg !66
  %kdma.source1507 = and i64 %kdma.source1506, 8, !dbg !66
  %kdma.source1508 = and i64 %kdma.source, 91, !dbg !66
  %kdma.source1509 = lshr i64 %kdma.source, 2, !dbg !66
  %kdma.source1510 = and i64 %kdma.source1509, 4, !dbg !66
  %kdma.source1511 = add nuw nsw i64 %kdma.source1510, %kdma.source, !dbg !66
  %kdma.source1512 = and i64 %kdma.source1511, 4, !dbg !66
  %kdma.source1513 = or disjoint i64 %kdma.source1508, %kdma.source1512, !dbg !66
  %kdma.source1514 = shl nuw nsw i64 %kdma.source1513, 7, !dbg !66
  %kdma.source1515 = or disjoint i64 %kdma.source1507, %kdma.source1514, !dbg !66
  %kdma.source1516 = or disjoint i64 %i99, %kdma.source1515, !dbg !66
  %kdma.source1517 = trunc nuw i64 %kdma.source1516 to i32, !dbg !66
  %kdma.source1518 = shl nuw nsw i32 %kdma.source1517, 1, !dbg !66
  %kdma.offset = or disjoint i32 %kdma.source1518, %kdma.intra, !dbg !66
  %i104 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !66
  %kdma.source1520 = lshr i64 %kdma.source1519, 4, !dbg !66
  %kdma.source1521 = and i64 %kdma.source1520, 8, !dbg !66
  %kdma.source1522 = and i64 %kdma.source1519, 123, !dbg !66
  %kdma.source1523 = lshr i64 %kdma.source1519, 2, !dbg !66
  %kdma.source1524 = and i64 %kdma.source1523, 4, !dbg !66
  %kdma.source1525 = add nuw nsw i64 %kdma.source1524, %kdma.source1519, !dbg !66
  %kdma.source1526 = and i64 %kdma.source1525, 4, !dbg !66
  %kdma.source1527 = or disjoint i64 %kdma.source1522, %kdma.source1526, !dbg !66
  %kdma.source1528 = shl nuw nsw i64 %kdma.source1527, 7, !dbg !66
  %kdma.source1529 = or disjoint i64 %kdma.source1521, %kdma.source1528, !dbg !66
  %kdma.source1530 = or disjoint i64 %i99, %kdma.source1529, !dbg !66
  %kdma.source1531 = trunc nuw i64 %kdma.source1530 to i32, !dbg !66
  %kdma.source1532 = shl nuw nsw i32 %kdma.source1531, 1, !dbg !66
  %kdma.offset1533 = or disjoint i32 %kdma.source1532, %kdma.intra.hi, !dbg !66
  %kdma.lds1534 = add i32 %dma.buffer.lds1374, 512, !dbg !66
  %i105 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1533, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !61
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !61
  %address.local518 = and i64 %address.local517, 3584, !dbg !61
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !61
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !61
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !61
  %i106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !66
  %dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i106), !dbg !66
  %kdma.source1546 = or disjoint i64 %kdma.source1515, 16, !dbg !66
  %kdma.source1547 = or disjoint i64 %i99, %kdma.source1546, !dbg !66
  %kdma.source1548 = trunc nuw i64 %kdma.source1547 to i32, !dbg !66
  %kdma.source1549 = shl nuw nsw i32 %kdma.source1548, 1, !dbg !66
  %kdma.offset1550 = or disjoint i32 %kdma.source1549, %kdma.intra, !dbg !66
  %i107 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1550, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %kdma.source1563 = or disjoint i64 %kdma.source1529, 16, !dbg !66
  %kdma.source1564 = or disjoint i64 %i99, %kdma.source1563, !dbg !66
  %kdma.source1565 = trunc nuw i64 %kdma.source1564 to i32, !dbg !66
  %kdma.source1566 = shl nuw nsw i32 %kdma.source1565, 1, !dbg !66
  %kdma.offset1567 = or disjoint i32 %kdma.source1566, %kdma.intra.hi, !dbg !66
  %kdma.lds1568 = add i32 %dma.buffer.lds1371, 512, !dbg !66
  %i108 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1567, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !66
  fence syncscope("workgroup") release, !dbg !66
  call void @llvm.amdgcn.s.barrier(), !dbg !66
  fence syncscope("workgroup") acquire, !dbg !66
  %i109 = or disjoint i64 %i99, 32, !dbg !66
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !61
  %address.local534 = ptrtoint ptr addrspace(3) %address.local533 to i64, !dbg !61
  %address.local535 = trunc i64 %address.local534 to i32, !dbg !61
  %i110 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !66
  %dma.buffer.lds1368 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i110), !dbg !66
  %kdma.source1580 = or disjoint i64 %i109, %kdma.source1515, !dbg !66
  %kdma.source1581 = trunc nuw i64 %kdma.source1580 to i32, !dbg !66
  %kdma.source1582 = shl nuw nsw i32 %kdma.source1581, 1, !dbg !66
  %kdma.offset1583 = or disjoint i32 %kdma.source1582, %kdma.intra, !dbg !66
  %i111 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1583, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %kdma.source1596 = or disjoint i64 %i109, %kdma.source1529, !dbg !66
  %kdma.source1597 = trunc nuw i64 %kdma.source1596 to i32, !dbg !66
  %kdma.source1598 = shl nuw nsw i32 %kdma.source1597, 1, !dbg !66
  %kdma.offset1599 = or disjoint i32 %kdma.source1598, %kdma.intra.hi, !dbg !66
  %kdma.lds1600 = add i32 %dma.buffer.lds1368, 512, !dbg !66
  %i112 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1599, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !61
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !61
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !61
  %i113 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !66
  %dma.buffer.lds1365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i113), !dbg !66
  %kdma.source1613 = or disjoint i64 %i109, %kdma.source1546, !dbg !66
  %kdma.source1614 = trunc nuw i64 %kdma.source1613 to i32, !dbg !66
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !66
  %kdma.offset1616 = or disjoint i32 %kdma.source1615, %kdma.intra, !dbg !66
  %i114 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1616, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %kdma.source1630 = or disjoint i64 %i109, %kdma.source1563, !dbg !66
  %kdma.source1631 = trunc nuw i64 %kdma.source1630 to i32, !dbg !66
  %kdma.source1632 = shl nuw nsw i32 %kdma.source1631, 1, !dbg !66
  %kdma.offset1633 = or disjoint i32 %kdma.source1632, %kdma.intra.hi, !dbg !66
  %kdma.lds1634 = add i32 %dma.buffer.lds1365, 512, !dbg !66
  %i115 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %address.local542 = lshr i64 %address.local499, 2, !dbg !65
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !65
  %address.local546 = and i64 %address.local545, 24, !dbg !65
  %address.local548 = shl i64 %address.local499, 62, !dbg !65
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !65
  %address.local550 = and i64 %address.local549, -64, !dbg !65
  %address.local553 = and i64 %address.local542, 12, !dbg !63
  %address.local555 = and i64 %address.local499, 63, !dbg !63
  %address.local556 = lshr i64 %address.local555, 5, !dbg !63
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !63
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !65
  %i116 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !65
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !65
  %i117 = getelementptr i8, ptr addrspace(3) %i116, i64 %.idx1410, !dbg !65
  %i118 = getelementptr [2 x i8], ptr addrspace(3) %i117, i64 %address.local546, !dbg !65
  %i119 = getelementptr [2 x i8], ptr addrspace(3) %i118, i64 %address.local550, !dbg !65
  %i120 = getelementptr [2 x i8], ptr addrspace(3) %i119, i64 %address.local553, !dbg !65
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %i120, i64 %address.local557, !dbg !65
  %i121 = add nsw i64 %.idx, %.idx1410, !dbg !65
  %.idx1893 = shl nuw nsw i64 %address.local546, 1, !dbg !65
  %i122 = or disjoint i64 %i121, %.idx1893, !dbg !65
  %i123 = or disjoint i64 %address.local550, %address.local553, !dbg !65
  %i124 = shl nsw i64 %i123, 1, !dbg !65
  %i125 = add nsw i64 %i124, %i122, !dbg !65
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032, !dbg !65
  %i126 = add nsw i64 %i125, %address.local559.idx, !dbg !65
  %i127 = lshr i64 %i126, 6, !dbg !65
  %i128 = and i64 %i127, 8, !dbg !65
  %i129 = xor i64 %i126, %i128, !dbg !65
  %i130 = sub i64 %i129, %i126, !dbg !65
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %i130, !dbg !65
  %i131 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !65
  %i132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i131, <4 x float> zeroinitializer) #9, !dbg !65
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !65
  %i133 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !65
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !65
  %address.local571 = and i32 %address.local570, 24, !dbg !65
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !65
  %address.local573 = and i32 %address.local572, 768, !dbg !65
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !65
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !65
  %address.local576 = and i32 %address.local575, 96, !dbg !65
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !65
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !65
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %i85) #9, !dbg !65, !frisk.register_lifetimes !58
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !65
  %address.local580 = and i32 %address.local579, 1, !dbg !65
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !65
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !65
  %address.local583 = shl i64 %address.local582, 6, !dbg !65
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !65
  %i134 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !65
  %i135 = getelementptr [2 x i8], ptr addrspace(3) %i134, i64 %address.local553, !dbg !65
  %i136 = getelementptr [2 x i8], ptr addrspace(3) %i135, i64 %address.local557, !dbg !65
  %address.local588 = getelementptr i8, ptr addrspace(3) %i136, i64 64, !dbg !65
  %.idx18961899 = add i64 %address.local584, %address.local553, !dbg !65
  %i137 = shl i64 %.idx18961899, 1, !dbg !65
  %i138 = add i64 %i137, %address.local559.idx, !dbg !65
  %i139 = add i64 %i138, 64, !dbg !65
  %i140 = lshr i64 %i139, 6, !dbg !65
  %i141 = and i64 %i140, 8, !dbg !65
  %i142 = xor i64 %i139, %i141, !dbg !65
  %i143 = sub i64 %i142, %i139, !dbg !65
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %i143, !dbg !65
  %i144 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !65, !frisk.k_lds_pair !67
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !65
  %i145 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !65, !frisk.k_lds_pair !67
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !65
  %i146 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !65, !frisk.k_lds_pair !68
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !65
  %i147 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !65, !frisk.k_lds_pair !68
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !65
  %i148 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !65, !frisk.k_lds_pair !69
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !65
  %i149 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !65, !frisk.k_lds_pair !69
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !65
  %i150 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !65, !frisk.k_lds_pair !70
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !65
  %i151 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !65, !frisk.k_lds_pair !70
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i133, <4 x float> %i132) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !65
  %i153 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !65, !frisk.k_lds_pair !71
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !65
  %i154 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !65, !frisk.k_lds_pair !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i144, <4 x float> zeroinitializer) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !65
  %i156 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !65, !frisk.k_lds_pair !72
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !65
  %i157 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !65, !frisk.k_lds_pair !72
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i146, <4 x float> %i155) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !65
  %i159 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !65, !frisk.k_lds_pair !73
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !65
  %i160 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !65, !frisk.k_lds_pair !73
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i148, <4 x float> zeroinitializer) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i150, <4 x float> %i161) #9, !dbg !65
  %i163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i145, <4 x float> zeroinitializer) #9, !dbg !65
  %i164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i147, <4 x float> %i163) #9, !dbg !65
  %i165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i149, <4 x float> zeroinitializer) #9, !dbg !65
  %i166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i151, <4 x float> %i165) #9, !dbg !65
  %i167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i153, <4 x float> zeroinitializer) #9, !dbg !65
  %i168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i156, <4 x float> %i167) #9, !dbg !65
  %i169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i159, <4 x float> zeroinitializer) #9, !dbg !65
  %i170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i160, <4 x float> %i169) #9, !dbg !65
  %i171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i40, <4 x half> %i154, <4 x float> zeroinitializer) #9, !dbg !65
  %i172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i42, <4 x half> %i157, <4 x float> %i171) #9, !dbg !65
  %i173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i131, <4 x float> zeroinitializer) #9, !dbg !65
  %i174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i133, <4 x float> %i173) #9, !dbg !65
  %i175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i144, <4 x float> zeroinitializer) #9, !dbg !65
  %i176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i146, <4 x float> %i175) #9, !dbg !65
  %i177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i148, <4 x float> zeroinitializer) #9, !dbg !65
  %i178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i150, <4 x float> %i177) #9, !dbg !65
  %i179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i145, <4 x float> zeroinitializer) #9, !dbg !65
  %i180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i147, <4 x float> %i179) #9, !dbg !65
  %i181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i149, <4 x float> zeroinitializer) #9, !dbg !65
  %i182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i151, <4 x float> %i181) #9, !dbg !65
  %i183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i153, <4 x float> zeroinitializer) #9, !dbg !65
  %i184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i156, <4 x float> %i183) #9, !dbg !65
  %i185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i159, <4 x float> zeroinitializer) #9, !dbg !65
  %i186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i160, <4 x float> %i185) #9, !dbg !65
  %i187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i44, <4 x half> %i154, <4 x float> zeroinitializer) #9, !dbg !65
  %i188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i45, <4 x half> %i157, <4 x float> %i187) #9, !dbg !65
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !66
  fence syncscope("workgroup") release, !dbg !66
  call void @llvm.amdgcn.s.barrier(), !dbg !66
  fence syncscope("workgroup") acquire, !dbg !66
  %i189 = or disjoint i64 %i99, 64, !dbg !66
  %kdma.source1646 = or disjoint i64 %i189, %kdma.source1515, !dbg !66
  %kdma.source1647 = trunc nuw i64 %kdma.source1646 to i32, !dbg !66
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !66
  %kdma.offset1649 = or disjoint i32 %kdma.source1648, %kdma.intra, !dbg !66
  %i190 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset1649, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %kdma.source1662 = or disjoint i64 %i189, %kdma.source1529, !dbg !66
  %kdma.source1663 = trunc nuw i64 %kdma.source1662 to i32, !dbg !66
  %kdma.source1664 = shl nuw nsw i32 %kdma.source1663, 1, !dbg !66
  %kdma.offset1665 = or disjoint i32 %kdma.source1664, %kdma.intra.hi, !dbg !66
  %i191 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1665, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %kdma.source1679 = or disjoint i64 %i189, %kdma.source1546, !dbg !66
  %kdma.source1680 = trunc nuw i64 %kdma.source1679 to i32, !dbg !66
  %kdma.source1681 = shl nuw nsw i32 %kdma.source1680, 1, !dbg !66
  %kdma.offset1682 = or disjoint i32 %kdma.source1681, %kdma.intra, !dbg !66
  %i192 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1682, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %kdma.source1696 = or disjoint i64 %i189, %kdma.source1563, !dbg !66
  %kdma.source1697 = trunc nuw i64 %kdma.source1696 to i32, !dbg !66
  %kdma.source1698 = shl nuw nsw i32 %kdma.source1697, 1, !dbg !66
  %kdma.offset1699 = or disjoint i32 %kdma.source1698, %kdma.intra.hi, !dbg !66
  %i193 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1699, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !65
  %i194 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !65, !frisk.k_lds_pair !74
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !65
  %i195 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !65, !frisk.k_lds_pair !74
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !65
  %i196 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !65, !frisk.k_lds_pair !75
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !65
  %i197 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !65, !frisk.k_lds_pair !75
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !65
  %i198 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !65, !frisk.k_lds_pair !76
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !65
  %i199 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !65, !frisk.k_lds_pair !76
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !65
  %i200 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !65, !frisk.k_lds_pair !77
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !65
  %i201 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !65, !frisk.k_lds_pair !77
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !65
  %i202 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !65, !frisk.k_lds_pair !78
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !65
  %i203 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !65, !frisk.k_lds_pair !78
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i194, <4 x float> %i152) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !65
  %i205 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !65, !frisk.k_lds_pair !79
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !65
  %i206 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !65, !frisk.k_lds_pair !79
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i196, <4 x float> %i204) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !65
  %i208 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !65, !frisk.k_lds_pair !80
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !65
  %i209 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !65, !frisk.k_lds_pair !80
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i198, <4 x float> %i158) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !65
  %i211 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !65, !frisk.k_lds_pair !81
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !65
  %i212 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !65, !frisk.k_lds_pair !81
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i200, <4 x float> %i210) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i195, <4 x float> %i162) #9, !dbg !65
  %i215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i197, <4 x float> %i214) #9, !dbg !65
  %i216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i199, <4 x float> %i164) #9, !dbg !65
  %i217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i201, <4 x float> %i216) #9, !dbg !65
  %i218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i202, <4 x float> %i166) #9, !dbg !65
  %i219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i205, <4 x float> %i218) #9, !dbg !65
  %i220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i208, <4 x float> %i168) #9, !dbg !65
  %i221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i211, <4 x float> %i220) #9, !dbg !65
  %i222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i203, <4 x float> %i170) #9, !dbg !65
  %i223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i206, <4 x float> %i222) #9, !dbg !65
  %i224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i54, <4 x half> %i209, <4 x float> %i172) #9, !dbg !65
  %i225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i56, <4 x half> %i212, <4 x float> %i224) #9, !dbg !65
  %i226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i194, <4 x float> %i174) #9, !dbg !65
  %i227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i196, <4 x float> %i226) #9, !dbg !65
  %i228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i198, <4 x float> %i176) #9, !dbg !65
  %i229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i200, <4 x float> %i228) #9, !dbg !65
  %i230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i195, <4 x float> %i178) #9, !dbg !65
  %i231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i197, <4 x float> %i230) #9, !dbg !65
  %i232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i199, <4 x float> %i180) #9, !dbg !65
  %i233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i201, <4 x float> %i232) #9, !dbg !65
  %i234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i202, <4 x float> %i182) #9, !dbg !65
  %i235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i205, <4 x float> %i234) #9, !dbg !65
  %i236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i208, <4 x float> %i184) #9, !dbg !65
  %i237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i211, <4 x float> %i236) #9, !dbg !65
  %i238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i203, <4 x float> %i186) #9, !dbg !65
  %i239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i206, <4 x float> %i238) #9, !dbg !65
  %i240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i57, <4 x half> %i209, <4 x float> %i188) #9, !dbg !65
  %i241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i58, <4 x half> %i212, <4 x float> %i240) #9, !dbg !65
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !66
  fence syncscope("workgroup") release, !dbg !66
  call void @llvm.amdgcn.s.barrier(), !dbg !66
  fence syncscope("workgroup") acquire, !dbg !66
  %i242 = or disjoint i64 %i99, 96, !dbg !66
  %kdma.source1712 = or disjoint i64 %i242, %kdma.source1515, !dbg !66
  %kdma.source1713 = trunc nuw i64 %kdma.source1712 to i32, !dbg !66
  %kdma.source1714 = shl nuw nsw i32 %kdma.source1713, 1, !dbg !66
  %kdma.offset1715 = or disjoint i32 %kdma.source1714, %kdma.intra, !dbg !66
  %i243 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1715, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %kdma.source1728 = or disjoint i64 %i242, %kdma.source1529, !dbg !66
  %kdma.source1729 = trunc nuw i64 %kdma.source1728 to i32, !dbg !66
  %kdma.source1730 = shl nuw nsw i32 %kdma.source1729, 1, !dbg !66
  %kdma.offset1731 = or disjoint i32 %kdma.source1730, %kdma.intra.hi, !dbg !66
  %i244 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1731, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %kdma.source1745 = or disjoint i64 %i242, %kdma.source1546, !dbg !66
  %kdma.source1746 = trunc nuw i64 %kdma.source1745 to i32, !dbg !66
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !66
  %kdma.offset1748 = or disjoint i32 %kdma.source1747, %kdma.intra, !dbg !66
  %i245 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1748, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %kdma.source1762 = or disjoint i64 %i242, %kdma.source1563, !dbg !66
  %kdma.source1763 = trunc nuw i64 %kdma.source1762 to i32, !dbg !66
  %kdma.source1764 = shl nuw nsw i32 %kdma.source1763, 1, !dbg !66
  %kdma.offset1765 = or disjoint i32 %kdma.source1764, %kdma.intra.hi, !dbg !66
  %i246 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1765, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !66
  %i247 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !65, !frisk.k_lds_pair !82
  %i248 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !65, !frisk.k_lds_pair !82
  %i249 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !65, !frisk.k_lds_pair !83
  %i250 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !65, !frisk.k_lds_pair !83
  %i251 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !65, !frisk.k_lds_pair !84
  %i252 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !65, !frisk.k_lds_pair !84
  %i253 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !65, !frisk.k_lds_pair !85
  %i254 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !65, !frisk.k_lds_pair !85
  %i255 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !65, !frisk.k_lds_pair !86
  %i256 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !65, !frisk.k_lds_pair !86
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i247, <4 x float> %i207) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i258 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !65, !frisk.k_lds_pair !87
  %i259 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !65, !frisk.k_lds_pair !87
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i249, <4 x float> %i257) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i261 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !65, !frisk.k_lds_pair !88
  %i262 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !65, !frisk.k_lds_pair !88
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i251, <4 x float> %i213) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i264 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !65, !frisk.k_lds_pair !89
  %i265 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !65, !frisk.k_lds_pair !89
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i253, <4 x float> %i263) #9, !dbg !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !65, !frisk.k_lds_prefetch !58
  %i267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i248, <4 x float> %i215) #9, !dbg !65
  %i268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i250, <4 x float> %i267) #9, !dbg !65
  %i269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i252, <4 x float> %i217) #9, !dbg !65
  %i270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i254, <4 x float> %i269) #9, !dbg !65
  %i271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i255, <4 x float> %i219) #9, !dbg !65
  %i272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i258, <4 x float> %i271) #9, !dbg !65
  %i273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i261, <4 x float> %i221) #9, !dbg !65
  %i274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i264, <4 x float> %i273) #9, !dbg !65
  %i275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i256, <4 x float> %i223) #9, !dbg !65
  %i276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i259, <4 x float> %i275) #9, !dbg !65
  %i277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i67, <4 x half> %i262, <4 x float> %i225) #9, !dbg !65
  %i278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i69, <4 x half> %i265, <4 x float> %i277) #9, !dbg !65
  %i279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i247, <4 x float> %i227) #9, !dbg !65
  %i280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i249, <4 x float> %i279) #9, !dbg !65
  %i281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i251, <4 x float> %i229) #9, !dbg !65
  %i282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i253, <4 x float> %i281) #9, !dbg !65
  %i283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i248, <4 x float> %i231) #9, !dbg !65
  %i284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i250, <4 x float> %i283) #9, !dbg !65
  %i285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i252, <4 x float> %i233) #9, !dbg !65
  %i286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i254, <4 x float> %i285) #9, !dbg !65
  %i287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i255, <4 x float> %i235) #9, !dbg !65
  %i288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i258, <4 x float> %i287) #9, !dbg !65
  %i289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i261, <4 x float> %i237) #9, !dbg !65
  %i290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i264, <4 x float> %i289) #9, !dbg !65
  %i291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i256, <4 x float> %i239) #9, !dbg !65
  %i292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i259, <4 x float> %i291) #9, !dbg !65
  %i293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i70, <4 x half> %i262, <4 x float> %i241) #9, !dbg !65
  %i294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i71, <4 x half> %i265, <4 x float> %i293) #9, !dbg !65
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !66
  fence syncscope("workgroup") release, !dbg !66
  call void @llvm.amdgcn.s.barrier(), !dbg !66
  fence syncscope("workgroup") acquire, !dbg !66
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !90
  %address.local918 = and i64 %address.local917, 8064, !dbg !90
  %address.local920 = and i64 %address.local500, 24, !dbg !90
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !90
  %i295 = ptrtoint ptr addrspace(1) %arg2 to i64, !dbg !90
  %.idx14191420 = or disjoint i64 %i99, %address.local921, !dbg !90
  %.idx14191420.tr = trunc nuw i64 %.idx14191420 to i32, !dbg !90
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !90
  %i296 = insertelement <2 x i64> poison, i64 %i295, i64 0, !dbg !90
  %i297 = bitcast <2 x i64> %i296 to <4 x i32>, !dbg !90
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %i297, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !90
  %i298 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !90
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !90
  %address.local924 = and i64 %address.local923, 16256, !dbg !90
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !90
  %.idx14211422 = or disjoint i64 %i99, %address.local927, !dbg !90
  %.idx14211422.tr = trunc nuw i64 %.idx14211422 to i32, !dbg !90
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !90
  %i299 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !90
  %i300 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !65
  %i301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i300, <4 x float> %i260) #9, !dbg !65
  %i302 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !65
  %i303 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !65
  %i304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i302, <4 x float> %i301) #9, !dbg !65
  %i305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i303, <4 x float> %i266) #9, !dbg !65
  %i306 = extractelement <4 x float> %i304, i64 0, !dbg !91
  %i307 = call float @llvm.maxnum.f32(float %i306, float -inf), !dbg !92, !frisk.local_reduction !58
  %i308 = extractelement <4 x float> %i304, i64 1, !dbg !91
  %i309 = extractelement <4 x float> %i304, i64 2, !dbg !91
  %i310 = extractelement <4 x float> %i304, i64 3, !dbg !91
  %i311 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !65
  %i312 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !65
  %i313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i311, <4 x float> %i305) #9, !dbg !65
  %i314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i312, <4 x float> %i268) #9, !dbg !65
  %i315 = extractelement <4 x float> %i313, i64 0, !dbg !91
  %local.reduce151 = call float @llvm.maxnum.f32(float %i307, float %i315), !dbg !92, !frisk.local_reduction !58
  %i316 = extractelement <4 x float> %i313, i64 1, !dbg !91
  %local.reduce152 = call float @llvm.maxnum.f32(float %i308, float %i316), !dbg !92, !frisk.local_reduction !58
  %i317 = extractelement <4 x float> %i313, i64 2, !dbg !91
  %local.reduce153 = call float @llvm.maxnum.f32(float %i309, float %i317), !dbg !92, !frisk.local_reduction !58
  %i318 = extractelement <4 x float> %i313, i64 3, !dbg !91
  %local.reduce154 = call float @llvm.maxnum.f32(float %i310, float %i318), !dbg !92, !frisk.local_reduction !58
  %i319 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !65
  %i320 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !65
  %i321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i319, <4 x float> %i314) #9, !dbg !65
  %i322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i320, <4 x float> %i270) #9, !dbg !65
  %i323 = extractelement <4 x float> %i321, i64 0, !dbg !91
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %i323), !dbg !92, !frisk.local_reduction !58
  %i324 = extractelement <4 x float> %i321, i64 1, !dbg !91
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %i324), !dbg !92, !frisk.local_reduction !58
  %i325 = extractelement <4 x float> %i321, i64 2, !dbg !91
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %i325), !dbg !92, !frisk.local_reduction !58
  %i326 = extractelement <4 x float> %i321, i64 3, !dbg !91
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %i326), !dbg !92, !frisk.local_reduction !58
  %i327 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !65
  %i328 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !65
  %i329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i327, <4 x float> %i322) #9, !dbg !65
  %i330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i328, <4 x float> %i272) #9, !dbg !65
  %i331 = extractelement <4 x float> %i329, i64 0, !dbg !91
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %i331), !dbg !92, !frisk.local_reduction !58
  %i332 = extractelement <4 x float> %i329, i64 1, !dbg !91
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %i332), !dbg !92, !frisk.local_reduction !58
  %i333 = extractelement <4 x float> %i329, i64 2, !dbg !91
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %i333), !dbg !92, !frisk.local_reduction !58
  %i334 = extractelement <4 x float> %i329, i64 3, !dbg !91
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %i334), !dbg !92, !frisk.local_reduction !58
  %i335 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !65
  %i336 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !65
  %i337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i335, <4 x float> %i330) #9, !dbg !65
  %i338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i336, <4 x float> %i274) #9, !dbg !65
  %i339 = extractelement <4 x float> %i337, i64 0, !dbg !91
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %i339), !dbg !92, !frisk.local_reduction !58
  %i340 = extractelement <4 x float> %i337, i64 1, !dbg !91
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %i340), !dbg !92, !frisk.local_reduction !58
  %i341 = extractelement <4 x float> %i337, i64 2, !dbg !91
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %i341), !dbg !92, !frisk.local_reduction !58
  %i342 = extractelement <4 x float> %i337, i64 3, !dbg !91
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %i342), !dbg !92, !frisk.local_reduction !58
  %i343 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !65
  %i344 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !65
  %i345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i343, <4 x float> %i338) #9, !dbg !65
  %i346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i344, <4 x float> %i276) #9, !dbg !65
  %i347 = extractelement <4 x float> %i345, i64 0, !dbg !91
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %i347), !dbg !92, !frisk.local_reduction !58
  %i348 = extractelement <4 x float> %i345, i64 1, !dbg !91
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %i348), !dbg !92, !frisk.local_reduction !58
  %i349 = extractelement <4 x float> %i345, i64 2, !dbg !91
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %i349), !dbg !92, !frisk.local_reduction !58
  %i350 = extractelement <4 x float> %i345, i64 3, !dbg !91
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %i350), !dbg !92, !frisk.local_reduction !58
  %i351 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !65
  %i352 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !65
  %i353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i351, <4 x float> %i346) #9, !dbg !65
  %i354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i75, <4 x half> %i352, <4 x float> %i278) #9, !dbg !65
  %i355 = extractelement <4 x float> %i353, i64 0, !dbg !91
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %i355), !dbg !92, !frisk.local_reduction !58
  %i356 = extractelement <4 x float> %i353, i64 1, !dbg !91
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %i356), !dbg !92, !frisk.local_reduction !58
  %i357 = extractelement <4 x float> %i353, i64 2, !dbg !91
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %i357), !dbg !92, !frisk.local_reduction !58
  %i358 = extractelement <4 x float> %i353, i64 3, !dbg !91
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %i358), !dbg !92, !frisk.local_reduction !58
  %i359 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !65
  %i360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i77, <4 x half> %i359, <4 x float> %i354) #9, !dbg !65
  %i361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i300, <4 x float> %i280) #9, !dbg !65
  %i362 = extractelement <4 x float> %i360, i64 0, !dbg !91
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %i362), !dbg !92, !frisk.local_reduction !58
  %i363 = extractelement <4 x float> %i360, i64 1, !dbg !91
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %i363), !dbg !92, !frisk.local_reduction !58
  %i364 = extractelement <4 x float> %i360, i64 2, !dbg !91
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %i364), !dbg !92, !frisk.local_reduction !58
  %i365 = extractelement <4 x float> %i360, i64 3, !dbg !91
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %i365), !dbg !92, !frisk.local_reduction !58
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !92, !frisk.local_reduction !58
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !92, !frisk.local_reduction !58
  %i366 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !92, !frisk.local_reduction !58
  %i367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i302, <4 x float> %i361) #9, !dbg !65
  %i368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i303, <4 x float> %i282) #9, !dbg !65
  %i369 = extractelement <4 x float> %i367, i64 0, !dbg !91
  %i370 = extractelement <4 x float> %i367, i64 1, !dbg !91
  %i371 = extractelement <4 x float> %i367, i64 2, !dbg !91
  %i372 = extractelement <4 x float> %i367, i64 3, !dbg !91
  %i373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i311, <4 x float> %i368) #9, !dbg !65
  %i374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i312, <4 x float> %i284) #9, !dbg !65
  %i375 = extractelement <4 x float> %i373, i64 0, !dbg !91
  %i376 = extractelement <4 x float> %i373, i64 1, !dbg !91
  %i377 = extractelement <4 x float> %i373, i64 2, !dbg !91
  %i378 = extractelement <4 x float> %i373, i64 3, !dbg !91
  %i379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i319, <4 x float> %i374) #9, !dbg !65
  %i380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i320, <4 x float> %i286) #9, !dbg !65
  %i381 = extractelement <4 x float> %i379, i64 0, !dbg !91
  %i382 = extractelement <4 x float> %i379, i64 1, !dbg !91
  %i383 = extractelement <4 x float> %i379, i64 2, !dbg !91
  %i384 = extractelement <4 x float> %i379, i64 3, !dbg !91
  %i385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i327, <4 x float> %i380) #9, !dbg !65
  %i386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i328, <4 x float> %i288) #9, !dbg !65
  %i387 = extractelement <4 x float> %i385, i64 0, !dbg !91
  %i388 = extractelement <4 x float> %i385, i64 1, !dbg !91
  %i389 = extractelement <4 x float> %i385, i64 2, !dbg !91
  %i390 = extractelement <4 x float> %i385, i64 3, !dbg !91
  %i391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i335, <4 x float> %i386) #9, !dbg !65
  %i392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i336, <4 x float> %i290) #9, !dbg !65
  %i393 = extractelement <4 x float> %i391, i64 0, !dbg !91
  %i394 = extractelement <4 x float> %i391, i64 1, !dbg !91
  %i395 = extractelement <4 x float> %i391, i64 2, !dbg !91
  %i396 = extractelement <4 x float> %i391, i64 3, !dbg !91
  %i397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i343, <4 x float> %i392) #9, !dbg !65
  %i398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i344, <4 x float> %i292) #9, !dbg !65
  %i399 = extractelement <4 x float> %i397, i64 0, !dbg !91
  %i400 = extractelement <4 x float> %i397, i64 1, !dbg !91
  %i401 = extractelement <4 x float> %i397, i64 2, !dbg !91
  %i402 = extractelement <4 x float> %i397, i64 3, !dbg !91
  %i403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i351, <4 x float> %i398) #9, !dbg !65
  %i404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i78, <4 x half> %i352, <4 x float> %i294) #9, !dbg !65
  %i405 = extractelement <4 x float> %i403, i64 0, !dbg !91
  %i406 = extractelement <4 x float> %i403, i64 1, !dbg !91
  %i407 = extractelement <4 x float> %i403, i64 2, !dbg !91
  %i408 = extractelement <4 x float> %i403, i64 3, !dbg !91
  %i409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %i79, <4 x half> %i359, <4 x float> %i404) #9, !dbg !65
  %i410 = extractelement <4 x float> %i409, i64 0, !dbg !91
  %i411 = extractelement <4 x float> %i409, i64 1, !dbg !91
  %i412 = extractelement <4 x float> %i409, i64 2, !dbg !91
  %i413 = extractelement <4 x float> %i409, i64 3, !dbg !91
  %i414 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !92
  %i415 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %i414), !dbg !92
  %i416 = add nuw nsw i32 %i415, 64, !dbg !92
  %i417 = xor i32 %i415, 16, !dbg !92
  %i418 = and i32 %i416, 192, !dbg !92
  %i419 = icmp samesign ult i32 %i417, %i418, !dbg !92
  %i420 = select i1 %i419, i32 %i417, i32 %i415, !dbg !92
  %i421 = shl nuw nsw i32 %i420, 2, !dbg !92
  %i422 = bitcast float %i366 to i32, !dbg !92
  %i423 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i422), !dbg !92
  %i424 = bitcast i32 %i423 to float, !dbg !92
  %i425 = call float @llvm.maxnum.f32(float %i366, float %i424), !dbg !92
  %i426 = xor i32 %i415, 32, !dbg !92
  %i427 = icmp samesign ult i32 %i426, %i418, !dbg !92
  %i428 = select i1 %i427, i32 %i426, i32 %i415, !dbg !92
  %i429 = shl nuw nsw i32 %i428, 2, !dbg !92
  %i430 = bitcast float %i425 to i32, !dbg !92
  %i431 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i430), !dbg !92
  %i432 = bitcast i32 %i431 to float, !dbg !92
  %i433 = call float @llvm.maxnum.f32(float %i425, float %i432), !dbg !92
  %i434 = extractvalue [2 x <1 x float>] %i90, 0, !dbg !92
  %i435 = insertelement <1 x float> poison, float %i433, i64 0, !dbg !92
  %i436 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %i435, <1 x float> %i434), !dbg !92
  %i437 = call float @llvm.maxnum.f32(float %i369, float -inf), !dbg !92, !frisk.local_reduction !58
  %local.reduce120 = call float @llvm.maxnum.f32(float %i437, float %i375), !dbg !92, !frisk.local_reduction !58
  %local.reduce121 = call float @llvm.maxnum.f32(float %i370, float %i376), !dbg !92, !frisk.local_reduction !58
  %local.reduce122 = call float @llvm.maxnum.f32(float %i371, float %i377), !dbg !92, !frisk.local_reduction !58
  %local.reduce123 = call float @llvm.maxnum.f32(float %i372, float %i378), !dbg !92, !frisk.local_reduction !58
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %i381), !dbg !92, !frisk.local_reduction !58
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %i382), !dbg !92, !frisk.local_reduction !58
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %i383), !dbg !92, !frisk.local_reduction !58
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %i384), !dbg !92, !frisk.local_reduction !58
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %i387), !dbg !92, !frisk.local_reduction !58
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %i388), !dbg !92, !frisk.local_reduction !58
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %i389), !dbg !92, !frisk.local_reduction !58
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %i390), !dbg !92, !frisk.local_reduction !58
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %i393), !dbg !92, !frisk.local_reduction !58
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %i394), !dbg !92, !frisk.local_reduction !58
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %i395), !dbg !92, !frisk.local_reduction !58
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %i396), !dbg !92, !frisk.local_reduction !58
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %i399), !dbg !92, !frisk.local_reduction !58
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %i400), !dbg !92, !frisk.local_reduction !58
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %i401), !dbg !92, !frisk.local_reduction !58
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %i402), !dbg !92, !frisk.local_reduction !58
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %i405), !dbg !92, !frisk.local_reduction !58
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %i406), !dbg !92, !frisk.local_reduction !58
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %i407), !dbg !92, !frisk.local_reduction !58
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %i408), !dbg !92, !frisk.local_reduction !58
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %i410), !dbg !92, !frisk.local_reduction !58
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %i411), !dbg !92, !frisk.local_reduction !58
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %i412), !dbg !92, !frisk.local_reduction !58
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %i413), !dbg !92, !frisk.local_reduction !58
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !92, !frisk.local_reduction !58
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !92, !frisk.local_reduction !58
  %i438 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !92, !frisk.local_reduction !58
  %i439 = bitcast float %i438 to i32, !dbg !92
  %i440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i421, i32 %i439), !dbg !92
  %i441 = bitcast i32 %i440 to float, !dbg !92
  %i442 = call float @llvm.maxnum.f32(float %i438, float %i441), !dbg !92
  %i443 = bitcast float %i442 to i32, !dbg !92
  %i444 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i429, i32 %i443), !dbg !92
  %i445 = bitcast i32 %i444 to float, !dbg !92
  %i446 = call float @llvm.maxnum.f32(float %i442, float %i445), !dbg !92
  %i447 = extractvalue [2 x <1 x float>] %i90, 1, !dbg !92
  %i448 = insertelement <1 x float> poison, float %i446, i64 0, !dbg !92
  %i449 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %i448, <1 x float> %i447), !dbg !92
  %i450 = shufflevector <1 x float> %i436, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !93
  %i459 = shufflevector <1 x float> %i449, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !93
  %i468 = fsub <1 x float> %i434, %i436, !dbg !94
  %i469 = extractelement <1 x float> %i468, i64 0, !dbg !95
  %i470 = call float @llvm.amdgcn.exp2.f32(float %i469), !dbg !95
  %i471 = insertelement <4 x float> poison, float %i470, i64 0, !dbg !96
  %i472 = shufflevector <4 x float> %i471, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !96
  %i473 = extractvalue [2 x <1 x float>] %i89, 0, !dbg !97
  %i474 = extractelement <1 x float> %i473, i64 0, !dbg !97
  %i475 = fmul float %i474, %i470, !dbg !97
  %i476 = fsub <1 x float> %i447, %i449, !dbg !94
  %i477 = extractelement <1 x float> %i476, i64 0, !dbg !95
  %i478 = call float @llvm.amdgcn.exp2.f32(float %i477), !dbg !95
  %i479 = insertelement <4 x float> poison, float %i478, i64 0, !dbg !96
  %i480 = shufflevector <4 x float> %i479, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !96
  %i481 = extractvalue [2 x <1 x float>] %i89, 1, !dbg !97
  %i482 = extractelement <1 x float> %i481, i64 0, !dbg !97
  %i483 = fmul float %i482, %i478, !dbg !97
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !90
  fence syncscope("workgroup") release, !dbg !90
  call void @llvm.amdgcn.s.barrier(), !dbg !90
  fence syncscope("workgroup") acquire, !dbg !90
  %vdb.address1436 = or disjoint i64 %i109, %address.local921, !dbg !90
  %vdb.address1437 = trunc nuw i64 %vdb.address1436 to i32, !dbg !90
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !90
  %vdb.address1445 = or disjoint i64 %i109, %address.local927, !dbg !90
  %vdb.address1446 = trunc nuw i64 %vdb.address1445 to i32, !dbg !90
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !90
  %vdb.dma.base = add i32 %dma.buffer.lds1374, 8192, !dbg !101
  %i700 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !90
  %vdb.dma.base1452 = add i32 %dma.buffer.lds1371, 8192, !dbg !101
  %i701 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !90
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !101
  %address.local930 = and i32 %address.local929, 120, !dbg !101
  %address.local932 = and i32 %address.local575, 1536, !dbg !101
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !101
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !101
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !101
  %i702 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i703 = bitcast <4 x i32> %i702 to <8 x half>, !dbg !101
  %address.local936 = and i64 %address.local499, 60, !dbg !101
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !101
  %address.local939.idx = shl nuw nsw i64 %address.local937, 4, !dbg !101
  %address.local939 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local939.idx, !dbg !101
  %address.local940 = lshr i64 %address.local555, 4, !dbg !101
  %address.local942.idx = mul nuw nsw i64 %address.local940, 768, !dbg !101
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local942.idx, !dbg !101
  %address.local944.idx = mul nsw i64 %address.local542, -64, !dbg !101
  %address.local944 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local944.idx, !dbg !101
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !101
  %i704 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i705 = bitcast <4 x i32> %i704 to <8 x half>, !dbg !101
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !101
  %i706 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i707 = bitcast <4 x i32> %i706 to <8 x half>, !dbg !101
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !101
  %i708 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i709 = bitcast <4 x i32> %i708 to <8 x half>, !dbg !101
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !101
  %i710 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i711 = bitcast <4 x i32> %i710 to <8 x half>, !dbg !101
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !101
  %i712 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i713 = bitcast <4 x i32> %i712 to <8 x half>, !dbg !101
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !101
  %i714 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i715 = bitcast <4 x i32> %i714 to <8 x half>, !dbg !101
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !101
  %i716 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i717 = shufflevector <8 x half> %i703, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  %i718 = extractvalue [2 x <8 x float>] %i88, 0, !dbg !96
  %i719 = shufflevector <8 x float> %i718, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !96
  %i720 = fmul <4 x float> %i719, %i472, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i451 = fsub <4 x float> %i304, %i450, !dbg !93
  %i484 = extractelement <4 x float> %i451, i64 0, !dbg !98
  %i485 = call float @llvm.amdgcn.exp2.f32(float %i484), !dbg !98
  %i486 = insertelement <4 x float> poison, float %i485, i64 0, !dbg !99
  %i487 = extractelement <4 x float> %i451, i64 1, !dbg !98
  %i488 = call float @llvm.amdgcn.exp2.f32(float %i487), !dbg !98
  %i489 = insertelement <4 x float> %i486, float %i488, i64 1, !dbg !99
  %i490 = extractelement <4 x float> %i451, i64 2, !dbg !98
  %i491 = call float @llvm.amdgcn.exp2.f32(float %i490), !dbg !98
  %i492 = insertelement <4 x float> %i489, float %i491, i64 2, !dbg !99
  %i493 = extractelement <4 x float> %i451, i64 3, !dbg !98
  %i494 = call float @llvm.amdgcn.exp2.f32(float %i493), !dbg !98
  %i495 = insertelement <4 x float> %i492, float %i494, i64 3, !dbg !99
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i495) #9, !dbg !99, !frisk.register_lifetimes !58
  %i496 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !99
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i496) #9, !dbg !98, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i717, <4 x float> %i720) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i722 = shufflevector <8 x half> %i703, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  %i723 = shufflevector <8 x float> %i718, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !96
  %i724 = fmul <4 x float> %i723, %i472, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i722, <4 x float> %i724) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !96
  %i726 = extractvalue [2 x <8 x float>] %i88, 1, !dbg !96
  %i727 = shufflevector <8 x float> %i726, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !96
  %i728 = fmul <4 x float> %i727, %i480, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i460 = fsub <4 x float> %i367, %i459, !dbg !93
  %i592 = extractelement <4 x float> %i460, i64 0, !dbg !98
  %i593 = call float @llvm.amdgcn.exp2.f32(float %i592), !dbg !98
  %i594 = insertelement <4 x float> poison, float %i593, i64 0, !dbg !99
  %i595 = extractelement <4 x float> %i460, i64 1, !dbg !98
  %i596 = call float @llvm.amdgcn.exp2.f32(float %i595), !dbg !98
  %i597 = insertelement <4 x float> %i594, float %i596, i64 1, !dbg !99
  %i598 = extractelement <4 x float> %i460, i64 2, !dbg !98
  %i599 = call float @llvm.amdgcn.exp2.f32(float %i598), !dbg !98
  %i600 = insertelement <4 x float> %i597, float %i599, i64 2, !dbg !99
  %i601 = extractelement <4 x float> %i460, i64 3, !dbg !98
  %i602 = call float @llvm.amdgcn.exp2.f32(float %i601), !dbg !98
  %i603 = insertelement <4 x float> %i600, float %i602, i64 3, !dbg !99
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i603) #9, !dbg !99, !frisk.register_lifetimes !58
  %i604 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !99
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i604) #9, !dbg !98, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i717, <4 x float> %i728) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !96
  %i730 = shufflevector <8 x float> %i726, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !96
  %i731 = fmul <4 x float> %i730, %i480, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i732 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i722, <4 x float> %i731) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i733 = shufflevector <8 x half> %i705, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i452 = fsub <4 x float> %i313, %i450, !dbg !93
  %i497 = extractelement <4 x float> %i452, i64 0, !dbg !98
  %i498 = call float @llvm.amdgcn.exp2.f32(float %i497), !dbg !98
  %local.reduce89 = fadd float %i485, %i498, !dbg !100, !frisk.local_reduction !58
  %i499 = insertelement <4 x float> poison, float %i498, i64 0, !dbg !99
  %i500 = extractelement <4 x float> %i452, i64 1, !dbg !98
  %i501 = call float @llvm.amdgcn.exp2.f32(float %i500), !dbg !98
  %local.reduce90 = fadd float %i488, %i501, !dbg !100, !frisk.local_reduction !58
  %i502 = insertelement <4 x float> %i499, float %i501, i64 1, !dbg !99
  %i503 = extractelement <4 x float> %i452, i64 2, !dbg !98
  %i504 = call float @llvm.amdgcn.exp2.f32(float %i503), !dbg !98
  %local.reduce91 = fadd float %i491, %i504, !dbg !100, !frisk.local_reduction !58
  %i505 = insertelement <4 x float> %i502, float %i504, i64 2, !dbg !99
  %i506 = extractelement <4 x float> %i452, i64 3, !dbg !98
  %i507 = call float @llvm.amdgcn.exp2.f32(float %i506), !dbg !98
  %local.reduce92 = fadd float %i494, %i507, !dbg !100, !frisk.local_reduction !58
  %i508 = insertelement <4 x float> %i505, float %i507, i64 3, !dbg !99
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i508) #9, !dbg !99, !frisk.register_lifetimes !58
  %i509 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !99
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i509) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i734 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i733, <4 x float> %i721) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i735 = shufflevector <8 x half> %i705, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i735, <4 x float> %i725) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i461 = fsub <4 x float> %i373, %i459, !dbg !93
  %i605 = extractelement <4 x float> %i461, i64 0, !dbg !98
  %i606 = call float @llvm.amdgcn.exp2.f32(float %i605), !dbg !98
  %local.reduce = fadd float %i593, %i606, !dbg !100, !frisk.local_reduction !58
  %i607 = insertelement <4 x float> poison, float %i606, i64 0, !dbg !99
  %i608 = extractelement <4 x float> %i461, i64 1, !dbg !98
  %i609 = call float @llvm.amdgcn.exp2.f32(float %i608), !dbg !98
  %local.reduce59 = fadd float %i596, %i609, !dbg !100, !frisk.local_reduction !58
  %i610 = insertelement <4 x float> %i607, float %i609, i64 1, !dbg !99
  %i611 = extractelement <4 x float> %i461, i64 2, !dbg !98
  %i612 = call float @llvm.amdgcn.exp2.f32(float %i611), !dbg !98
  %local.reduce60 = fadd float %i599, %i612, !dbg !100, !frisk.local_reduction !58
  %i613 = insertelement <4 x float> %i610, float %i612, i64 2, !dbg !99
  %i614 = extractelement <4 x float> %i461, i64 3, !dbg !98
  %i615 = call float @llvm.amdgcn.exp2.f32(float %i614), !dbg !98
  %local.reduce61 = fadd float %i602, %i615, !dbg !100, !frisk.local_reduction !58
  %i616 = insertelement <4 x float> %i613, float %i615, i64 3, !dbg !99
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i616) #9, !dbg !99, !frisk.register_lifetimes !58
  %i617 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !99
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i617) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i733, <4 x float> %i729) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i738 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i735, <4 x float> %i732) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i739 = shufflevector <8 x half> %i707, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i453 = fsub <4 x float> %i321, %i450, !dbg !93
  %i510 = extractelement <4 x float> %i453, i64 0, !dbg !98
  %i511 = call float @llvm.amdgcn.exp2.f32(float %i510), !dbg !98
  %local.reduce93 = fadd float %local.reduce89, %i511, !dbg !100, !frisk.local_reduction !58
  %i512 = insertelement <4 x float> poison, float %i511, i64 0, !dbg !99
  %i513 = extractelement <4 x float> %i453, i64 1, !dbg !98
  %i514 = call float @llvm.amdgcn.exp2.f32(float %i513), !dbg !98
  %local.reduce94 = fadd float %local.reduce90, %i514, !dbg !100, !frisk.local_reduction !58
  %i515 = insertelement <4 x float> %i512, float %i514, i64 1, !dbg !99
  %i516 = extractelement <4 x float> %i453, i64 2, !dbg !98
  %i517 = call float @llvm.amdgcn.exp2.f32(float %i516), !dbg !98
  %local.reduce95 = fadd float %local.reduce91, %i517, !dbg !100, !frisk.local_reduction !58
  %i518 = insertelement <4 x float> %i515, float %i517, i64 2, !dbg !99
  %i519 = extractelement <4 x float> %i453, i64 3, !dbg !98
  %i520 = call float @llvm.amdgcn.exp2.f32(float %i519), !dbg !98
  %local.reduce96 = fadd float %local.reduce92, %i520, !dbg !100, !frisk.local_reduction !58
  %i521 = insertelement <4 x float> %i518, float %i520, i64 3, !dbg !99
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i521) #9, !dbg !99, !frisk.register_lifetimes !58
  %i522 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !99
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i522) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i740 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i739, <4 x float> %i734) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i741 = shufflevector <8 x half> %i707, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i742 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i741, <4 x float> %i736) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i462 = fsub <4 x float> %i379, %i459, !dbg !93
  %i618 = extractelement <4 x float> %i462, i64 0, !dbg !98
  %i619 = call float @llvm.amdgcn.exp2.f32(float %i618), !dbg !98
  %local.reduce62 = fadd float %local.reduce, %i619, !dbg !100, !frisk.local_reduction !58
  %i620 = insertelement <4 x float> poison, float %i619, i64 0, !dbg !99
  %i621 = extractelement <4 x float> %i462, i64 1, !dbg !98
  %i622 = call float @llvm.amdgcn.exp2.f32(float %i621), !dbg !98
  %local.reduce63 = fadd float %local.reduce59, %i622, !dbg !100, !frisk.local_reduction !58
  %i623 = insertelement <4 x float> %i620, float %i622, i64 1, !dbg !99
  %i624 = extractelement <4 x float> %i462, i64 2, !dbg !98
  %i625 = call float @llvm.amdgcn.exp2.f32(float %i624), !dbg !98
  %local.reduce64 = fadd float %local.reduce60, %i625, !dbg !100, !frisk.local_reduction !58
  %i626 = insertelement <4 x float> %i623, float %i625, i64 2, !dbg !99
  %i627 = extractelement <4 x float> %i462, i64 3, !dbg !98
  %i628 = call float @llvm.amdgcn.exp2.f32(float %i627), !dbg !98
  %local.reduce65 = fadd float %local.reduce61, %i628, !dbg !100, !frisk.local_reduction !58
  %i629 = insertelement <4 x float> %i626, float %i628, i64 3, !dbg !99
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i629) #9, !dbg !99, !frisk.register_lifetimes !58
  %i630 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !99
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i630) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i743 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i739, <4 x float> %i737) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i744 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i741, <4 x float> %i738) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i745 = shufflevector <8 x half> %i709, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i454 = fsub <4 x float> %i329, %i450, !dbg !93
  %i523 = extractelement <4 x float> %i454, i64 0, !dbg !98
  %i524 = call float @llvm.amdgcn.exp2.f32(float %i523), !dbg !98
  %local.reduce97 = fadd float %local.reduce93, %i524, !dbg !100, !frisk.local_reduction !58
  %i525 = insertelement <4 x float> poison, float %i524, i64 0, !dbg !99
  %i526 = extractelement <4 x float> %i454, i64 1, !dbg !98
  %i527 = call float @llvm.amdgcn.exp2.f32(float %i526), !dbg !98
  %local.reduce98 = fadd float %local.reduce94, %i527, !dbg !100, !frisk.local_reduction !58
  %i528 = insertelement <4 x float> %i525, float %i527, i64 1, !dbg !99
  %i529 = extractelement <4 x float> %i454, i64 2, !dbg !98
  %i530 = call float @llvm.amdgcn.exp2.f32(float %i529), !dbg !98
  %local.reduce99 = fadd float %local.reduce95, %i530, !dbg !100, !frisk.local_reduction !58
  %i531 = insertelement <4 x float> %i528, float %i530, i64 2, !dbg !99
  %i532 = extractelement <4 x float> %i454, i64 3, !dbg !98
  %i533 = call float @llvm.amdgcn.exp2.f32(float %i532), !dbg !98
  %local.reduce100 = fadd float %local.reduce96, %i533, !dbg !100, !frisk.local_reduction !58
  %i534 = insertelement <4 x float> %i531, float %i533, i64 3, !dbg !99
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i534) #9, !dbg !99, !frisk.register_lifetimes !58
  %i535 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !99
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i535) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i746 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i745, <4 x float> %i740) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i747 = shufflevector <8 x half> %i709, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i748 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i747, <4 x float> %i742) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i463 = fsub <4 x float> %i385, %i459, !dbg !93
  %i631 = extractelement <4 x float> %i463, i64 0, !dbg !98
  %i632 = call float @llvm.amdgcn.exp2.f32(float %i631), !dbg !98
  %local.reduce66 = fadd float %local.reduce62, %i632, !dbg !100, !frisk.local_reduction !58
  %i633 = insertelement <4 x float> poison, float %i632, i64 0, !dbg !99
  %i634 = extractelement <4 x float> %i463, i64 1, !dbg !98
  %i635 = call float @llvm.amdgcn.exp2.f32(float %i634), !dbg !98
  %local.reduce67 = fadd float %local.reduce63, %i635, !dbg !100, !frisk.local_reduction !58
  %i636 = insertelement <4 x float> %i633, float %i635, i64 1, !dbg !99
  %i637 = extractelement <4 x float> %i463, i64 2, !dbg !98
  %i638 = call float @llvm.amdgcn.exp2.f32(float %i637), !dbg !98
  %local.reduce68 = fadd float %local.reduce64, %i638, !dbg !100, !frisk.local_reduction !58
  %i639 = insertelement <4 x float> %i636, float %i638, i64 2, !dbg !99
  %i640 = extractelement <4 x float> %i463, i64 3, !dbg !98
  %i641 = call float @llvm.amdgcn.exp2.f32(float %i640), !dbg !98
  %local.reduce69 = fadd float %local.reduce65, %i641, !dbg !100, !frisk.local_reduction !58
  %i642 = insertelement <4 x float> %i639, float %i641, i64 3, !dbg !99
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i642) #9, !dbg !99, !frisk.register_lifetimes !58
  %i643 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !99
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i643) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i749 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i745, <4 x float> %i743) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i747, <4 x float> %i744) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i751 = shufflevector <8 x half> %i711, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i455 = fsub <4 x float> %i337, %i450, !dbg !93
  %i536 = extractelement <4 x float> %i455, i64 0, !dbg !98
  %i537 = call float @llvm.amdgcn.exp2.f32(float %i536), !dbg !98
  %local.reduce101 = fadd float %local.reduce97, %i537, !dbg !100, !frisk.local_reduction !58
  %i538 = insertelement <4 x float> poison, float %i537, i64 0, !dbg !99
  %i539 = extractelement <4 x float> %i455, i64 1, !dbg !98
  %i540 = call float @llvm.amdgcn.exp2.f32(float %i539), !dbg !98
  %local.reduce102 = fadd float %local.reduce98, %i540, !dbg !100, !frisk.local_reduction !58
  %i541 = insertelement <4 x float> %i538, float %i540, i64 1, !dbg !99
  %i542 = extractelement <4 x float> %i455, i64 2, !dbg !98
  %i543 = call float @llvm.amdgcn.exp2.f32(float %i542), !dbg !98
  %local.reduce103 = fadd float %local.reduce99, %i543, !dbg !100, !frisk.local_reduction !58
  %i544 = insertelement <4 x float> %i541, float %i543, i64 2, !dbg !99
  %i545 = extractelement <4 x float> %i455, i64 3, !dbg !98
  %i546 = call float @llvm.amdgcn.exp2.f32(float %i545), !dbg !98
  %local.reduce104 = fadd float %local.reduce100, %i546, !dbg !100, !frisk.local_reduction !58
  %i547 = insertelement <4 x float> %i544, float %i546, i64 3, !dbg !99
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i547) #9, !dbg !99, !frisk.register_lifetimes !58
  %i548 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !99
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i548) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i752 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i751, <4 x float> %i746) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i753 = shufflevector <8 x half> %i711, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i754 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i753, <4 x float> %i748) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i464 = fsub <4 x float> %i391, %i459, !dbg !93
  %i644 = extractelement <4 x float> %i464, i64 0, !dbg !98
  %i645 = call float @llvm.amdgcn.exp2.f32(float %i644), !dbg !98
  %local.reduce70 = fadd float %local.reduce66, %i645, !dbg !100, !frisk.local_reduction !58
  %i646 = insertelement <4 x float> poison, float %i645, i64 0, !dbg !99
  %i647 = extractelement <4 x float> %i464, i64 1, !dbg !98
  %i648 = call float @llvm.amdgcn.exp2.f32(float %i647), !dbg !98
  %local.reduce71 = fadd float %local.reduce67, %i648, !dbg !100, !frisk.local_reduction !58
  %i649 = insertelement <4 x float> %i646, float %i648, i64 1, !dbg !99
  %i650 = extractelement <4 x float> %i464, i64 2, !dbg !98
  %i651 = call float @llvm.amdgcn.exp2.f32(float %i650), !dbg !98
  %local.reduce72 = fadd float %local.reduce68, %i651, !dbg !100, !frisk.local_reduction !58
  %i652 = insertelement <4 x float> %i649, float %i651, i64 2, !dbg !99
  %i653 = extractelement <4 x float> %i464, i64 3, !dbg !98
  %i654 = call float @llvm.amdgcn.exp2.f32(float %i653), !dbg !98
  %local.reduce73 = fadd float %local.reduce69, %i654, !dbg !100, !frisk.local_reduction !58
  %i655 = insertelement <4 x float> %i652, float %i654, i64 3, !dbg !99
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i655) #9, !dbg !99, !frisk.register_lifetimes !58
  %i656 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !99
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i656) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i751, <4 x float> %i749) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i753, <4 x float> %i750) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i757 = shufflevector <8 x half> %i713, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i456 = fsub <4 x float> %i345, %i450, !dbg !93
  %i549 = extractelement <4 x float> %i456, i64 0, !dbg !98
  %i550 = call float @llvm.amdgcn.exp2.f32(float %i549), !dbg !98
  %local.reduce105 = fadd float %local.reduce101, %i550, !dbg !100, !frisk.local_reduction !58
  %i551 = insertelement <4 x float> poison, float %i550, i64 0, !dbg !99
  %i552 = extractelement <4 x float> %i456, i64 1, !dbg !98
  %i553 = call float @llvm.amdgcn.exp2.f32(float %i552), !dbg !98
  %local.reduce106 = fadd float %local.reduce102, %i553, !dbg !100, !frisk.local_reduction !58
  %i554 = insertelement <4 x float> %i551, float %i553, i64 1, !dbg !99
  %i555 = extractelement <4 x float> %i456, i64 2, !dbg !98
  %i556 = call float @llvm.amdgcn.exp2.f32(float %i555), !dbg !98
  %local.reduce107 = fadd float %local.reduce103, %i556, !dbg !100, !frisk.local_reduction !58
  %i557 = insertelement <4 x float> %i554, float %i556, i64 2, !dbg !99
  %i558 = extractelement <4 x float> %i456, i64 3, !dbg !98
  %i559 = call float @llvm.amdgcn.exp2.f32(float %i558), !dbg !98
  %local.reduce108 = fadd float %local.reduce104, %i559, !dbg !100, !frisk.local_reduction !58
  %i560 = insertelement <4 x float> %i557, float %i559, i64 3, !dbg !99
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i560) #9, !dbg !99, !frisk.register_lifetimes !58
  %i561 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !99
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i561) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i757, <4 x float> %i752) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i759 = shufflevector <8 x half> %i713, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i759, <4 x float> %i754) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i465 = fsub <4 x float> %i397, %i459, !dbg !93
  %i657 = extractelement <4 x float> %i465, i64 0, !dbg !98
  %i658 = call float @llvm.amdgcn.exp2.f32(float %i657), !dbg !98
  %local.reduce74 = fadd float %local.reduce70, %i658, !dbg !100, !frisk.local_reduction !58
  %i659 = insertelement <4 x float> poison, float %i658, i64 0, !dbg !99
  %i660 = extractelement <4 x float> %i465, i64 1, !dbg !98
  %i661 = call float @llvm.amdgcn.exp2.f32(float %i660), !dbg !98
  %local.reduce75 = fadd float %local.reduce71, %i661, !dbg !100, !frisk.local_reduction !58
  %i662 = insertelement <4 x float> %i659, float %i661, i64 1, !dbg !99
  %i663 = extractelement <4 x float> %i465, i64 2, !dbg !98
  %i664 = call float @llvm.amdgcn.exp2.f32(float %i663), !dbg !98
  %local.reduce76 = fadd float %local.reduce72, %i664, !dbg !100, !frisk.local_reduction !58
  %i665 = insertelement <4 x float> %i662, float %i664, i64 2, !dbg !99
  %i666 = extractelement <4 x float> %i465, i64 3, !dbg !98
  %i667 = call float @llvm.amdgcn.exp2.f32(float %i666), !dbg !98
  %local.reduce77 = fadd float %local.reduce73, %i667, !dbg !100, !frisk.local_reduction !58
  %i668 = insertelement <4 x float> %i665, float %i667, i64 3, !dbg !99
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i668) #9, !dbg !99, !frisk.register_lifetimes !58
  %i669 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !99
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i669) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i761 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i757, <4 x float> %i755) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i762 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i759, <4 x float> %i756) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i763 = shufflevector <8 x half> %i715, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i457 = fsub <4 x float> %i353, %i450, !dbg !93
  %i562 = extractelement <4 x float> %i457, i64 0, !dbg !98
  %i563 = call float @llvm.amdgcn.exp2.f32(float %i562), !dbg !98
  %local.reduce109 = fadd float %local.reduce105, %i563, !dbg !100, !frisk.local_reduction !58
  %i564 = insertelement <4 x float> poison, float %i563, i64 0, !dbg !99
  %i565 = extractelement <4 x float> %i457, i64 1, !dbg !98
  %i566 = call float @llvm.amdgcn.exp2.f32(float %i565), !dbg !98
  %local.reduce110 = fadd float %local.reduce106, %i566, !dbg !100, !frisk.local_reduction !58
  %i567 = insertelement <4 x float> %i564, float %i566, i64 1, !dbg !99
  %i568 = extractelement <4 x float> %i457, i64 2, !dbg !98
  %i569 = call float @llvm.amdgcn.exp2.f32(float %i568), !dbg !98
  %local.reduce111 = fadd float %local.reduce107, %i569, !dbg !100, !frisk.local_reduction !58
  %i570 = insertelement <4 x float> %i567, float %i569, i64 2, !dbg !99
  %i571 = extractelement <4 x float> %i457, i64 3, !dbg !98
  %i572 = call float @llvm.amdgcn.exp2.f32(float %i571), !dbg !98
  %local.reduce112 = fadd float %local.reduce108, %i572, !dbg !100, !frisk.local_reduction !58
  %i573 = insertelement <4 x float> %i570, float %i572, i64 3, !dbg !99
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i573) #9, !dbg !99, !frisk.register_lifetimes !58
  %i574 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !99
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i574) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i764 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i763, <4 x float> %i758) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i765 = shufflevector <8 x half> %i715, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i766 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i765, <4 x float> %i760) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i466 = fsub <4 x float> %i403, %i459, !dbg !93
  %i670 = extractelement <4 x float> %i466, i64 0, !dbg !98
  %i671 = call float @llvm.amdgcn.exp2.f32(float %i670), !dbg !98
  %local.reduce78 = fadd float %local.reduce74, %i671, !dbg !100, !frisk.local_reduction !58
  %i672 = insertelement <4 x float> poison, float %i671, i64 0, !dbg !99
  %i673 = extractelement <4 x float> %i466, i64 1, !dbg !98
  %i674 = call float @llvm.amdgcn.exp2.f32(float %i673), !dbg !98
  %local.reduce79 = fadd float %local.reduce75, %i674, !dbg !100, !frisk.local_reduction !58
  %i675 = insertelement <4 x float> %i672, float %i674, i64 1, !dbg !99
  %i676 = extractelement <4 x float> %i466, i64 2, !dbg !98
  %i677 = call float @llvm.amdgcn.exp2.f32(float %i676), !dbg !98
  %local.reduce80 = fadd float %local.reduce76, %i677, !dbg !100, !frisk.local_reduction !58
  %i678 = insertelement <4 x float> %i675, float %i677, i64 2, !dbg !99
  %i679 = extractelement <4 x float> %i466, i64 3, !dbg !98
  %i680 = call float @llvm.amdgcn.exp2.f32(float %i679), !dbg !98
  %local.reduce81 = fadd float %local.reduce77, %i680, !dbg !100, !frisk.local_reduction !58
  %i681 = insertelement <4 x float> %i678, float %i680, i64 3, !dbg !99
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i681) #9, !dbg !99, !frisk.register_lifetimes !58
  %i682 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !99
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i682) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i767 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i763, <4 x float> %i761) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i768 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i765, <4 x float> %i762) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i769 = bitcast <4 x i32> %i716 to <8 x half>, !dbg !101
  %i770 = shufflevector <8 x half> %i769, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i458 = fsub <4 x float> %i360, %i450, !dbg !93
  %i575 = extractelement <4 x float> %i458, i64 0, !dbg !98
  %i576 = call float @llvm.amdgcn.exp2.f32(float %i575), !dbg !98
  %local.reduce113 = fadd float %local.reduce109, %i576, !dbg !100, !frisk.local_reduction !58
  %i577 = insertelement <4 x float> poison, float %i576, i64 0, !dbg !99
  %i578 = extractelement <4 x float> %i458, i64 1, !dbg !98
  %i579 = call float @llvm.amdgcn.exp2.f32(float %i578), !dbg !98
  %local.reduce114 = fadd float %local.reduce110, %i579, !dbg !100, !frisk.local_reduction !58
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !100, !frisk.local_reduction !58
  %i580 = insertelement <4 x float> %i577, float %i579, i64 1, !dbg !99
  %i581 = extractelement <4 x float> %i458, i64 2, !dbg !98
  %i582 = call float @llvm.amdgcn.exp2.f32(float %i581), !dbg !98
  %local.reduce115 = fadd float %local.reduce111, %i582, !dbg !100, !frisk.local_reduction !58
  %i583 = insertelement <4 x float> %i580, float %i582, i64 2, !dbg !99
  %i584 = extractelement <4 x float> %i458, i64 3, !dbg !98
  %i585 = call float @llvm.amdgcn.exp2.f32(float %i584), !dbg !98
  %local.reduce116 = fadd float %local.reduce112, %i585, !dbg !100, !frisk.local_reduction !58
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !100, !frisk.local_reduction !58
  %i588 = fadd float %local.reduce117, %local.reduce118, !dbg !100, !frisk.local_reduction !58
  %i589 = fadd float %i588, %i475, !dbg !100
  %i590 = insertelement <1 x float> poison, float %i589, i64 0, !dbg !64
  %i591 = insertvalue [2 x <1 x float>] poison, <1 x float> %i590, 0, !dbg !64
  %i586 = insertelement <4 x float> %i583, float %i585, i64 3, !dbg !99
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i586) #9, !dbg !99, !frisk.register_lifetimes !58
  %i587 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !99
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i587) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i770, <4 x float> %i764) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i772 = shufflevector <8 x half> %i769, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i772, <4 x float> %i766) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i467 = fsub <4 x float> %i409, %i459, !dbg !93
  %i683 = extractelement <4 x float> %i467, i64 0, !dbg !98
  %i684 = call float @llvm.amdgcn.exp2.f32(float %i683), !dbg !98
  %local.reduce82 = fadd float %local.reduce78, %i684, !dbg !100, !frisk.local_reduction !58
  %i685 = insertelement <4 x float> poison, float %i684, i64 0, !dbg !99
  %i686 = extractelement <4 x float> %i467, i64 1, !dbg !98
  %i687 = call float @llvm.amdgcn.exp2.f32(float %i686), !dbg !98
  %local.reduce83 = fadd float %local.reduce79, %i687, !dbg !100, !frisk.local_reduction !58
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !100, !frisk.local_reduction !58
  %i688 = insertelement <4 x float> %i685, float %i687, i64 1, !dbg !99
  %i689 = extractelement <4 x float> %i467, i64 2, !dbg !98
  %i690 = call float @llvm.amdgcn.exp2.f32(float %i689), !dbg !98
  %local.reduce84 = fadd float %local.reduce80, %i690, !dbg !100, !frisk.local_reduction !58
  %i691 = insertelement <4 x float> %i688, float %i690, i64 2, !dbg !99
  %i692 = extractelement <4 x float> %i467, i64 3, !dbg !98
  %i693 = call float @llvm.amdgcn.exp2.f32(float %i692), !dbg !98
  %local.reduce85 = fadd float %local.reduce81, %i693, !dbg !100, !frisk.local_reduction !58
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !100, !frisk.local_reduction !58
  %i696 = fadd float %local.reduce86, %local.reduce87, !dbg !100, !frisk.local_reduction !58
  %i697 = fadd float %i696, %i483, !dbg !100
  %i698 = insertelement <1 x float> poison, float %i697, i64 0, !dbg !64
  %i699 = insertvalue [2 x <1 x float>] %i591, <1 x float> %i698, 1, !dbg !64
  %i694 = insertelement <4 x float> %i691, float %i693, i64 3, !dbg !99
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %i694) #9, !dbg !99, !frisk.register_lifetimes !58
  %i695 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !99
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %i695) #9, !dbg !100, !frisk.register_lifetimes !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101, !frisk.softmax_fragments !58
  %i774 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i770, <4 x float> %i767) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i775 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i772, <4 x float> %i768) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !90
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !90
  fence syncscope("workgroup") release, !dbg !90
  call void @llvm.amdgcn.s.barrier(), !dbg !90
  fence syncscope("workgroup") acquire, !dbg !90
  %vdb.address1462 = or disjoint i64 %i189, %address.local921, !dbg !90
  %vdb.address1463 = trunc nuw i64 %vdb.address1462 to i32, !dbg !90
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !90
  %vdb.address1471 = or disjoint i64 %i189, %address.local927, !dbg !90
  %vdb.address1472 = trunc nuw i64 %vdb.address1471 to i32, !dbg !90
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !90
  %i776 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !90
  %i777 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %vdb.address1473, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !90
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !101
  %i778 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i779 = bitcast <4 x i32> %i778 to <8 x half>, !dbg !101
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8448, !dbg !101
  %i780 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i781 = bitcast <4 x i32> %i780 to <8 x half>, !dbg !101
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8704, !dbg !101
  %i782 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i783 = bitcast <4 x i32> %i782 to <8 x half>, !dbg !101
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8960, !dbg !101
  %i784 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i785 = bitcast <4 x i32> %i784 to <8 x half>, !dbg !101
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local935, i64 12288, !dbg !101
  %i786 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i787 = bitcast <4 x i32> %i786 to <8 x half>, !dbg !101
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12544, !dbg !101
  %i788 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i789 = bitcast <4 x i32> %i788 to <8 x half>, !dbg !101
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12800, !dbg !101
  %i790 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i791 = bitcast <4 x i32> %i790 to <8 x half>, !dbg !101
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local944, i64 13056, !dbg !101
  %i792 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i793 = shufflevector <8 x half> %i779, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  %i794 = extractvalue [2 x <8 x float>] %i91, 0, !dbg !96
  %i795 = shufflevector <8 x float> %i794, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !96
  %i796 = fmul <4 x float> %i795, %i472, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i793, <4 x float> %i796) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i798 = shufflevector <8 x half> %i779, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  %i799 = shufflevector <8 x float> %i794, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !96
  %i800 = fmul <4 x float> %i799, %i472, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i801 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i798, <4 x float> %i800) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !96
  %i802 = extractvalue [2 x <8 x float>] %i91, 1, !dbg !96
  %i803 = shufflevector <8 x float> %i802, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !96
  %i804 = fmul <4 x float> %i803, %i480, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i805 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i793, <4 x float> %i804) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !96
  %i806 = shufflevector <8 x float> %i802, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !96
  %i807 = fmul <4 x float> %i806, %i480, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i808 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i798, <4 x float> %i807) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i809 = shufflevector <8 x half> %i781, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i809, <4 x float> %i797) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i811 = shufflevector <8 x half> %i781, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i811, <4 x float> %i801) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i813 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i809, <4 x float> %i805) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i811, <4 x float> %i808) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i815 = shufflevector <8 x half> %i783, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i815, <4 x float> %i810) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i817 = shufflevector <8 x half> %i783, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i818 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i817, <4 x float> %i812) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i819 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i815, <4 x float> %i813) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i820 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i817, <4 x float> %i814) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i821 = shufflevector <8 x half> %i785, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i821, <4 x float> %i816) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i823 = shufflevector <8 x half> %i785, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i823, <4 x float> %i818) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i821, <4 x float> %i819) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i823, <4 x float> %i820) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i827 = shufflevector <8 x half> %i787, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i827, <4 x float> %i822) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i829 = shufflevector <8 x half> %i787, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i830 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i829, <4 x float> %i824) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i827, <4 x float> %i825) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i832 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i829, <4 x float> %i826) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i833 = shufflevector <8 x half> %i789, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i834 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i833, <4 x float> %i828) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i835 = shufflevector <8 x half> %i789, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i836 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i835, <4 x float> %i830) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i833, <4 x float> %i831) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i835, <4 x float> %i832) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i839 = shufflevector <8 x half> %i791, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i839, <4 x float> %i834) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i841 = shufflevector <8 x half> %i791, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i841, <4 x float> %i836) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i839, <4 x float> %i837) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i841, <4 x float> %i838) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i845 = bitcast <4 x i32> %i792 to <8 x half>, !dbg !101
  %i846 = shufflevector <8 x half> %i845, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i846, <4 x float> %i840) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i848 = shufflevector <8 x half> %i845, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i848, <4 x float> %i842) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i846, <4 x float> %i843) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i848, <4 x float> %i844) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !90
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !90
  fence syncscope("workgroup") release, !dbg !90
  call void @llvm.amdgcn.s.barrier(), !dbg !90
  fence syncscope("workgroup") acquire, !dbg !90
  %vdb.address1480 = or disjoint i64 %i242, %address.local921, !dbg !90
  %vdb.address1481 = trunc nuw i64 %vdb.address1480 to i32, !dbg !90
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !90
  %vdb.address1489 = or disjoint i64 %i242, %address.local927, !dbg !90
  %vdb.address1490 = trunc nuw i64 %vdb.address1489 to i32, !dbg !90
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !90
  %i852 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1482, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !90
  %i853 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1491, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !90
  %i854 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i855 = bitcast <4 x i32> %i854 to <8 x half>, !dbg !101
  %i856 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i857 = bitcast <4 x i32> %i856 to <8 x half>, !dbg !101
  %i858 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i859 = bitcast <4 x i32> %i858 to <8 x half>, !dbg !101
  %i860 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i861 = bitcast <4 x i32> %i860 to <8 x half>, !dbg !101
  %i862 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i863 = bitcast <4 x i32> %i862 to <8 x half>, !dbg !101
  %i864 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i865 = bitcast <4 x i32> %i864 to <8 x half>, !dbg !101
  %i866 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i867 = bitcast <4 x i32> %i866 to <8 x half>, !dbg !101
  %i868 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i869 = shufflevector <8 x half> %i855, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  %i870 = extractvalue [2 x <8 x float>] %i92, 0, !dbg !96
  %i871 = shufflevector <8 x float> %i870, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !96
  %i872 = fmul <4 x float> %i871, %i472, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i869, <4 x float> %i872) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i874 = shufflevector <8 x half> %i855, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  %i875 = shufflevector <8 x float> %i870, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !96
  %i876 = fmul <4 x float> %i875, %i472, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i874, <4 x float> %i876) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !96
  %i878 = extractvalue [2 x <8 x float>] %i92, 1, !dbg !96
  %i879 = shufflevector <8 x float> %i878, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !96
  %i880 = fmul <4 x float> %i879, %i480, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i869, <4 x float> %i880) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !96
  %i882 = shufflevector <8 x float> %i878, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !96
  %i883 = fmul <4 x float> %i882, %i480, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i874, <4 x float> %i883) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i885 = shufflevector <8 x half> %i857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i885, <4 x float> %i873) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i887 = shufflevector <8 x half> %i857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i887, <4 x float> %i877) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i889 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i885, <4 x float> %i881) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i887, <4 x float> %i884) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i891 = shufflevector <8 x half> %i859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i891, <4 x float> %i886) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i893 = shufflevector <8 x half> %i859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i893, <4 x float> %i888) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i891, <4 x float> %i889) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i893, <4 x float> %i890) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i897 = shufflevector <8 x half> %i861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i897, <4 x float> %i892) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i899 = shufflevector <8 x half> %i861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i899, <4 x float> %i894) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i897, <4 x float> %i895) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i899, <4 x float> %i896) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i903 = shufflevector <8 x half> %i863, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i903, <4 x float> %i898) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i905 = shufflevector <8 x half> %i863, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i905, <4 x float> %i900) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i903, <4 x float> %i901) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i905, <4 x float> %i902) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i909 = shufflevector <8 x half> %i865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i909, <4 x float> %i904) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i911 = shufflevector <8 x half> %i865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i911, <4 x float> %i906) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i909, <4 x float> %i907) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i911, <4 x float> %i908) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i915 = shufflevector <8 x half> %i867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i915, <4 x float> %i910) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i917 = shufflevector <8 x half> %i867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i917, <4 x float> %i912) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i915, <4 x float> %i913) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i917, <4 x float> %i914) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i921 = bitcast <4 x i32> %i868 to <8 x half>, !dbg !101
  %i922 = shufflevector <8 x half> %i921, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i922, <4 x float> %i916) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i924 = shufflevector <8 x half> %i921, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i924, <4 x float> %i918) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i922, <4 x float> %i919) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i924, <4 x float> %i920) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !90
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !90
  fence syncscope("workgroup") release, !dbg !90
  call void @llvm.amdgcn.s.barrier(), !dbg !90
  fence syncscope("workgroup") acquire, !dbg !90
  %i928 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i929 = bitcast <4 x i32> %i928 to <8 x half>, !dbg !101
  %i930 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i931 = bitcast <4 x i32> %i930 to <8 x half>, !dbg !101
  %i932 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i933 = bitcast <4 x i32> %i932 to <8 x half>, !dbg !101
  %i934 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i935 = bitcast <4 x i32> %i934 to <8 x half>, !dbg !101
  %i936 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i937 = bitcast <4 x i32> %i936 to <8 x half>, !dbg !101
  %i938 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i939 = bitcast <4 x i32> %i938 to <8 x half>, !dbg !101
  %i940 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i941 = bitcast <4 x i32> %i940 to <8 x half>, !dbg !101
  %i942 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !101, !frisk.v_double_buffer !102
  %i943 = shufflevector <8 x half> %i929, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  %i944 = extractvalue [2 x <8 x float>] %i93, 0, !dbg !96
  %i945 = shufflevector <8 x float> %i944, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !96
  %i946 = fmul <4 x float> %i945, %i472, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i943, <4 x float> %i946) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i948 = shufflevector <8 x half> %i929, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  %i949 = shufflevector <8 x float> %i944, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !96
  %i950 = fmul <4 x float> %i949, %i472, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i951 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %i948, <4 x float> %i950) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !96
  %i952 = extractvalue [2 x <8 x float>] %i93, 1, !dbg !96
  %i953 = shufflevector <8 x float> %i952, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !96
  %i954 = fmul <4 x float> %i953, %i480, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i943, <4 x float> %i954) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !96
  %i956 = shufflevector <8 x float> %i952, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !96
  %i957 = fmul <4 x float> %i956, %i480, !dbg !96
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %i948, <4 x float> %i957) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i959 = shufflevector <8 x half> %i931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i959, <4 x float> %i947) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i961 = shufflevector <8 x half> %i931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %i961, <4 x float> %i951) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i959, <4 x float> %i955) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %i961, <4 x float> %i958) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i965 = shufflevector <8 x half> %i933, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i965, <4 x float> %i960) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i967 = shufflevector <8 x half> %i933, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %i967, <4 x float> %i962) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i965, <4 x float> %i963) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %i967, <4 x float> %i964) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i971 = shufflevector <8 x half> %i935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i971, <4 x float> %i966) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i973 = shufflevector <8 x half> %i935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %i973, <4 x float> %i968) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i971, <4 x float> %i969) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %i973, <4 x float> %i970) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i977 = shufflevector <8 x half> %i937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i977, <4 x float> %i972) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i979 = shufflevector <8 x half> %i937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %i979, <4 x float> %i974) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i977, <4 x float> %i975) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %i979, <4 x float> %i976) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i983 = shufflevector <8 x half> %i939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i983, <4 x float> %i978) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i985 = shufflevector <8 x half> %i939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %i985, <4 x float> %i980) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i983, <4 x float> %i981) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %i985, <4 x float> %i982) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i989 = shufflevector <8 x half> %i941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i989, <4 x float> %i984) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i991 = shufflevector <8 x half> %i941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %i991, <4 x float> %i986) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i989, <4 x float> %i987) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %i991, <4 x float> %i988) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i995 = bitcast <4 x i32> %i942 to <8 x half>, !dbg !101
  %i996 = shufflevector <8 x half> %i995, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i996, <4 x float> %i990) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i998 = shufflevector <8 x half> %i995, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %i998, <4 x float> %i992) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i1000 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i996, <4 x float> %i993) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !101
  %i1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %i998, <4 x float> %i994) #9, !dbg !101, !frisk.pv_interleave !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !64
  %i1002 = shufflevector <4 x float> %i771, <4 x float> %i773, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %i1003 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1002, 0, !dbg !64
  %i1004 = shufflevector <4 x float> %i774, <4 x float> %i775, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %i1005 = insertvalue [2 x <8 x float>] %i1003, <8 x float> %i1004, 1, !dbg !64
  %i1006 = insertvalue [2 x <1 x float>] poison, <1 x float> %i436, 0, !dbg !64
  %i1007 = insertvalue [2 x <1 x float>] %i1006, <1 x float> %i449, 1, !dbg !64
  %i1008 = shufflevector <4 x float> %i847, <4 x float> %i849, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %i1009 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1008, 0, !dbg !64
  %i1010 = shufflevector <4 x float> %i850, <4 x float> %i851, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %i1011 = insertvalue [2 x <8 x float>] %i1009, <8 x float> %i1010, 1, !dbg !64
  %i1012 = shufflevector <4 x float> %i923, <4 x float> %i925, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %i1013 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1012, 0, !dbg !64
  %i1014 = shufflevector <4 x float> %i926, <4 x float> %i927, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %i1015 = insertvalue [2 x <8 x float>] %i1013, <8 x float> %i1014, 1, !dbg !64
  %i1016 = shufflevector <4 x float> %i997, <4 x float> %i999, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %i1017 = insertvalue [2 x <8 x float>] poison, <8 x float> %i1016, 0, !dbg !64
  %i1018 = shufflevector <4 x float> %i1000, <4 x float> %i1001, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %i1019 = insertvalue [2 x <8 x float>] %i1017, <8 x float> %i1018, 1, !dbg !64
  fence syncscope("workgroup") release, !dbg !64
  call void @llvm.amdgcn.s.barrier(), !dbg !64
  fence syncscope("workgroup") acquire, !dbg !64
  %i1020 = add nuw nsw i64 %i87, 1, !dbg !64
  br label %bb86, !dbg !64

bb1021:                                           ; preds = %bb86
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %i89, 1, !dbg !64
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !64
  %lane.sum.lane.lo186 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !64
  %lane.sum.lane187 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo186), !dbg !64
  %i1022 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !64
  %i1023 = xor i32 %i1022, 64, !dbg !64
  %i1024 = bitcast float %lane.sum.local185 to i32, !dbg !64
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1023, i32 %i1024), !dbg !64
  %i1025 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !64
  %lane.sum.final189 = fadd float %lane.sum.local185, %i1025, !dbg !64
  %i1026 = xor i32 %i1022, 128, !dbg !64
  %i1027 = bitcast float %lane.sum.final189 to i32, !dbg !64
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1026, i32 %i1027), !dbg !64
  %i1028 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !64
  %lane.sum.final191 = fadd float %lane.sum.final189, %i1028, !dbg !64
  %i1029 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !64
  %lane.sum.field = extractvalue [2 x <1 x float>] %i89, 0, !dbg !64
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !64
  %i1030 = bitcast float %lane.sum.local to i32, !dbg !64
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1023, i32 %i1030), !dbg !64
  %i1031 = bitcast i32 %lane.sum.shuffle to float, !dbg !64
  %lane.sum.final = fadd float %lane.sum.local, %i1031, !dbg !64
  %i1032 = bitcast float %lane.sum.final to i32, !dbg !64
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1026, i32 %i1032), !dbg !64
  %i1033 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !64
  %lane.sum.final183 = fadd float %lane.sum.final, %i1033, !dbg !64
  %i1034 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !64
  %i1035 = and i32 %kdma.tid, 16, !dbg !64
  %.not = icmp eq i32 %i1035, 0, !dbg !64
  %i1036 = and i32 %kdma.tid, 32, !dbg !64
  %.not36 = icmp eq i32 %i1036, 0, !dbg !64
  %i1037 = extractvalue [2 x <8 x float>] %i93, 0, !dbg !64
  %i1038 = extractelement <8 x float> %i1037, i64 0, !dbg !64
  %i1039 = extractelement <8 x float> %i1037, i64 1, !dbg !64
  %i1040 = extractelement <8 x float> %i1037, i64 2, !dbg !64
  %i1041 = extractelement <8 x float> %i1037, i64 3, !dbg !64
  %i1042 = select i1 %.not, float %i1039, float %i1038, !dbg !64
  %i1043 = add nuw nsw i32 %lane.sum.lane187, 64, !dbg !64
  %i1044 = xor i32 %lane.sum.lane187, 16, !dbg !64
  %i1045 = and i32 %i1043, 192, !dbg !64
  %i1046 = icmp samesign ult i32 %i1044, %i1045, !dbg !64
  %i1047 = select i1 %i1046, i32 %i1044, i32 %lane.sum.lane187, !dbg !64
  %i1048 = shl nuw nsw i32 %i1047, 2, !dbg !64
  %i1049 = bitcast float %i1042 to i32, !dbg !64
  %i1050 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1049), !dbg !64
  %i1051 = bitcast i32 %i1050 to float, !dbg !64
  %i1052 = select i1 %.not, float %i1038, float %i1051, !dbg !64
  %i1053 = select i1 %.not, float %i1051, float %i1039, !dbg !64
  %i1054 = select i1 %.not, float %i1041, float %i1040, !dbg !64
  %i1055 = bitcast float %i1054 to i32, !dbg !64
  %i1056 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1055), !dbg !64
  %i1057 = bitcast i32 %i1056 to float, !dbg !64
  %i1058 = select i1 %.not, float %i1040, float %i1057, !dbg !64
  %i1059 = select i1 %.not, float %i1057, float %i1041, !dbg !64
  %i1060 = select i1 %.not36, float %i1058, float %i1052, !dbg !64
  %i1061 = xor i32 %lane.sum.lane187, 32, !dbg !64
  %i1062 = icmp samesign ult i32 %i1061, %i1045, !dbg !64
  %i1063 = select i1 %i1062, i32 %i1061, i32 %lane.sum.lane187, !dbg !64
  %i1064 = shl nuw nsw i32 %i1063, 2, !dbg !64
  %i1065 = bitcast float %i1060 to i32, !dbg !64
  %i1066 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1065), !dbg !64
  %i1067 = bitcast i32 %i1066 to float, !dbg !64
  %i1068 = select i1 %.not36, float %i1052, float %i1067, !dbg !64
  %i1069 = select i1 %.not36, float %i1067, float %i1058, !dbg !64
  %i1070 = select i1 %.not36, float %i1059, float %i1053, !dbg !64
  %i1071 = bitcast float %i1070 to i32, !dbg !64
  %i1072 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1071), !dbg !64
  %i1073 = bitcast i32 %i1072 to float, !dbg !64
  %i1074 = select i1 %.not36, float %i1053, float %i1073, !dbg !64
  %i1075 = select i1 %.not36, float %i1073, float %i1059, !dbg !64
  %i1076 = extractvalue [2 x <8 x float>] %i92, 0, !dbg !64
  %i1077 = extractelement <8 x float> %i1076, i64 0, !dbg !64
  %i1078 = extractelement <8 x float> %i1076, i64 1, !dbg !64
  %i1079 = extractelement <8 x float> %i1076, i64 2, !dbg !64
  %i1080 = extractelement <8 x float> %i1076, i64 3, !dbg !64
  %i1081 = select i1 %.not, float %i1078, float %i1077, !dbg !64
  %i1082 = bitcast float %i1081 to i32, !dbg !64
  %i1083 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1082), !dbg !64
  %i1084 = bitcast i32 %i1083 to float, !dbg !64
  %i1085 = select i1 %.not, float %i1077, float %i1084, !dbg !64
  %i1086 = select i1 %.not, float %i1084, float %i1078, !dbg !64
  %i1087 = select i1 %.not, float %i1080, float %i1079, !dbg !64
  %i1088 = bitcast float %i1087 to i32, !dbg !64
  %i1089 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1088), !dbg !64
  %i1090 = bitcast i32 %i1089 to float, !dbg !64
  %i1091 = select i1 %.not, float %i1079, float %i1090, !dbg !64
  %i1092 = select i1 %.not, float %i1090, float %i1080, !dbg !64
  %i1093 = select i1 %.not36, float %i1091, float %i1085, !dbg !64
  %i1094 = bitcast float %i1093 to i32, !dbg !64
  %i1095 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1094), !dbg !64
  %i1096 = bitcast i32 %i1095 to float, !dbg !64
  %i1097 = select i1 %.not36, float %i1085, float %i1096, !dbg !64
  %i1098 = select i1 %.not36, float %i1096, float %i1091, !dbg !64
  %i1099 = select i1 %.not36, float %i1092, float %i1086, !dbg !64
  %i1100 = bitcast float %i1099 to i32, !dbg !64
  %i1101 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1100), !dbg !64
  %i1102 = bitcast i32 %i1101 to float, !dbg !64
  %i1103 = select i1 %.not36, float %i1086, float %i1102, !dbg !64
  %i1104 = select i1 %.not36, float %i1102, float %i1092, !dbg !64
  %i1105 = extractvalue [2 x <8 x float>] %i91, 0, !dbg !64
  %i1106 = extractelement <8 x float> %i1105, i64 0, !dbg !64
  %i1107 = extractelement <8 x float> %i1105, i64 1, !dbg !64
  %i1108 = extractelement <8 x float> %i1105, i64 2, !dbg !64
  %i1109 = extractelement <8 x float> %i1105, i64 3, !dbg !64
  %i1110 = select i1 %.not, float %i1107, float %i1106, !dbg !64
  %i1111 = bitcast float %i1110 to i32, !dbg !64
  %i1112 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1111), !dbg !64
  %i1113 = bitcast i32 %i1112 to float, !dbg !64
  %i1114 = select i1 %.not, float %i1106, float %i1113, !dbg !64
  %i1115 = select i1 %.not, float %i1113, float %i1107, !dbg !64
  %i1116 = select i1 %.not, float %i1109, float %i1108, !dbg !64
  %i1117 = bitcast float %i1116 to i32, !dbg !64
  %i1118 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1117), !dbg !64
  %i1119 = bitcast i32 %i1118 to float, !dbg !64
  %i1120 = select i1 %.not, float %i1108, float %i1119, !dbg !64
  %i1121 = select i1 %.not, float %i1119, float %i1109, !dbg !64
  %i1122 = select i1 %.not36, float %i1120, float %i1114, !dbg !64
  %i1123 = bitcast float %i1122 to i32, !dbg !64
  %i1124 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1123), !dbg !64
  %i1125 = bitcast i32 %i1124 to float, !dbg !64
  %i1126 = select i1 %.not36, float %i1114, float %i1125, !dbg !64
  %i1127 = select i1 %.not36, float %i1125, float %i1120, !dbg !64
  %i1128 = select i1 %.not36, float %i1121, float %i1115, !dbg !64
  %i1129 = bitcast float %i1128 to i32, !dbg !64
  %i1130 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1129), !dbg !64
  %i1131 = bitcast i32 %i1130 to float, !dbg !64
  %i1132 = select i1 %.not36, float %i1115, float %i1131, !dbg !64
  %i1133 = select i1 %.not36, float %i1131, float %i1121, !dbg !64
  %i1134 = extractvalue [2 x <8 x float>] %i88, 0, !dbg !64
  %i1135 = extractelement <8 x float> %i1134, i64 0, !dbg !64
  %i1136 = extractelement <8 x float> %i1134, i64 1, !dbg !64
  %i1137 = extractelement <8 x float> %i1134, i64 2, !dbg !64
  %i1138 = extractelement <8 x float> %i1134, i64 3, !dbg !64
  %i1139 = select i1 %.not, float %i1136, float %i1135, !dbg !64
  %i1140 = bitcast float %i1139 to i32, !dbg !64
  %i1141 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1140), !dbg !64
  %i1142 = bitcast i32 %i1141 to float, !dbg !64
  %i1143 = select i1 %.not, float %i1135, float %i1142, !dbg !64
  %i1144 = select i1 %.not, float %i1142, float %i1136, !dbg !64
  %i1145 = select i1 %.not, float %i1138, float %i1137, !dbg !64
  %i1146 = bitcast float %i1145 to i32, !dbg !64
  %i1147 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1146), !dbg !64
  %i1148 = bitcast i32 %i1147 to float, !dbg !64
  %i1149 = select i1 %.not, float %i1137, float %i1148, !dbg !64
  %i1150 = select i1 %.not, float %i1148, float %i1138, !dbg !64
  %i1151 = select i1 %.not36, float %i1149, float %i1143, !dbg !64
  %i1152 = bitcast float %i1151 to i32, !dbg !64
  %i1153 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1152), !dbg !64
  %i1154 = select i1 %.not36, float %i1150, float %i1144, !dbg !64
  %i1155 = bitcast float %i1154 to i32, !dbg !64
  %i1156 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1155), !dbg !64
  %i1157 = shufflevector <1 x float> %i1034, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !103
  %i1158 = extractelement <8 x float> %i1037, i64 4, !dbg !64
  %i1159 = extractelement <8 x float> %i1037, i64 5, !dbg !64
  %i1160 = extractelement <8 x float> %i1037, i64 6, !dbg !64
  %i1161 = extractelement <8 x float> %i1037, i64 7, !dbg !64
  %i1162 = select i1 %.not, float %i1159, float %i1158, !dbg !64
  %i1163 = bitcast float %i1162 to i32, !dbg !64
  %i1164 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1163), !dbg !64
  %i1165 = bitcast i32 %i1164 to float, !dbg !64
  %i1166 = select i1 %.not, float %i1158, float %i1165, !dbg !64
  %i1167 = select i1 %.not, float %i1165, float %i1159, !dbg !64
  %i1168 = select i1 %.not, float %i1161, float %i1160, !dbg !64
  %i1169 = bitcast float %i1168 to i32, !dbg !64
  %i1170 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1169), !dbg !64
  %i1171 = bitcast i32 %i1170 to float, !dbg !64
  %i1172 = select i1 %.not, float %i1160, float %i1171, !dbg !64
  %i1173 = select i1 %.not, float %i1171, float %i1161, !dbg !64
  %i1174 = select i1 %.not36, float %i1172, float %i1166, !dbg !64
  %i1175 = bitcast float %i1174 to i32, !dbg !64
  %i1176 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1175), !dbg !64
  %i1177 = bitcast i32 %i1176 to float, !dbg !64
  %i1178 = select i1 %.not36, float %i1166, float %i1177, !dbg !64
  %i1179 = select i1 %.not36, float %i1177, float %i1172, !dbg !64
  %i1180 = select i1 %.not36, float %i1173, float %i1167, !dbg !64
  %i1181 = bitcast float %i1180 to i32, !dbg !64
  %i1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1181), !dbg !64
  %i1183 = bitcast i32 %i1182 to float, !dbg !64
  %i1184 = select i1 %.not36, float %i1167, float %i1183, !dbg !64
  %i1185 = select i1 %.not36, float %i1183, float %i1173, !dbg !64
  %i1186 = extractelement <8 x float> %i1076, i64 4, !dbg !64
  %i1187 = extractelement <8 x float> %i1076, i64 5, !dbg !64
  %i1188 = extractelement <8 x float> %i1076, i64 6, !dbg !64
  %i1189 = extractelement <8 x float> %i1076, i64 7, !dbg !64
  %i1190 = select i1 %.not, float %i1187, float %i1186, !dbg !64
  %i1191 = bitcast float %i1190 to i32, !dbg !64
  %i1192 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1191), !dbg !64
  %i1193 = bitcast i32 %i1192 to float, !dbg !64
  %i1194 = select i1 %.not, float %i1186, float %i1193, !dbg !64
  %i1195 = select i1 %.not, float %i1193, float %i1187, !dbg !64
  %i1196 = select i1 %.not, float %i1189, float %i1188, !dbg !64
  %i1197 = bitcast float %i1196 to i32, !dbg !64
  %i1198 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1197), !dbg !64
  %i1199 = bitcast i32 %i1198 to float, !dbg !64
  %i1200 = select i1 %.not, float %i1188, float %i1199, !dbg !64
  %i1201 = select i1 %.not, float %i1199, float %i1189, !dbg !64
  %i1202 = select i1 %.not36, float %i1200, float %i1194, !dbg !64
  %i1203 = bitcast float %i1202 to i32, !dbg !64
  %i1204 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1203), !dbg !64
  %i1205 = bitcast i32 %i1204 to float, !dbg !64
  %i1206 = select i1 %.not36, float %i1194, float %i1205, !dbg !64
  %i1207 = select i1 %.not36, float %i1205, float %i1200, !dbg !64
  %i1208 = select i1 %.not36, float %i1201, float %i1195, !dbg !64
  %i1209 = bitcast float %i1208 to i32, !dbg !64
  %i1210 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1209), !dbg !64
  %i1211 = bitcast i32 %i1210 to float, !dbg !64
  %i1212 = select i1 %.not36, float %i1195, float %i1211, !dbg !64
  %i1213 = select i1 %.not36, float %i1211, float %i1201, !dbg !64
  %i1214 = extractelement <8 x float> %i1105, i64 4, !dbg !64
  %i1215 = extractelement <8 x float> %i1105, i64 5, !dbg !64
  %i1216 = extractelement <8 x float> %i1105, i64 6, !dbg !64
  %i1217 = extractelement <8 x float> %i1105, i64 7, !dbg !64
  %i1218 = select i1 %.not, float %i1215, float %i1214, !dbg !64
  %i1219 = bitcast float %i1218 to i32, !dbg !64
  %i1220 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1219), !dbg !64
  %i1221 = bitcast i32 %i1220 to float, !dbg !64
  %i1222 = select i1 %.not, float %i1214, float %i1221, !dbg !64
  %i1223 = select i1 %.not, float %i1221, float %i1215, !dbg !64
  %i1224 = select i1 %.not, float %i1217, float %i1216, !dbg !64
  %i1225 = bitcast float %i1224 to i32, !dbg !64
  %i1226 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1225), !dbg !64
  %i1227 = bitcast i32 %i1226 to float, !dbg !64
  %i1228 = select i1 %.not, float %i1216, float %i1227, !dbg !64
  %i1229 = select i1 %.not, float %i1227, float %i1217, !dbg !64
  %i1230 = select i1 %.not36, float %i1228, float %i1222, !dbg !64
  %i1231 = bitcast float %i1230 to i32, !dbg !64
  %i1232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1231), !dbg !64
  %i1233 = bitcast i32 %i1232 to float, !dbg !64
  %i1234 = select i1 %.not36, float %i1222, float %i1233, !dbg !64
  %i1235 = select i1 %.not36, float %i1233, float %i1228, !dbg !64
  %i1236 = select i1 %.not36, float %i1229, float %i1223, !dbg !64
  %i1237 = bitcast float %i1236 to i32, !dbg !64
  %i1238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1237), !dbg !64
  %i1239 = bitcast i32 %i1238 to float, !dbg !64
  %i1240 = select i1 %.not36, float %i1223, float %i1239, !dbg !64
  %i1241 = select i1 %.not36, float %i1239, float %i1229, !dbg !64
  %i1242 = extractelement <8 x float> %i1134, i64 4, !dbg !64
  %i1243 = extractelement <8 x float> %i1134, i64 5, !dbg !64
  %i1244 = extractelement <8 x float> %i1134, i64 6, !dbg !64
  %i1245 = extractelement <8 x float> %i1134, i64 7, !dbg !64
  %i1246 = select i1 %.not, float %i1243, float %i1242, !dbg !64
  %i1247 = bitcast float %i1246 to i32, !dbg !64
  %i1248 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1247), !dbg !64
  %i1249 = bitcast i32 %i1248 to float, !dbg !64
  %i1250 = select i1 %.not, float %i1242, float %i1249, !dbg !64
  %i1251 = select i1 %.not, float %i1249, float %i1243, !dbg !64
  %i1252 = select i1 %.not, float %i1245, float %i1244, !dbg !64
  %i1253 = bitcast float %i1252 to i32, !dbg !64
  %i1254 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1253), !dbg !64
  %i1255 = bitcast i32 %i1254 to float, !dbg !64
  %i1256 = select i1 %.not, float %i1244, float %i1255, !dbg !64
  %i1257 = select i1 %.not, float %i1255, float %i1245, !dbg !64
  %i1258 = select i1 %.not36, float %i1256, float %i1250, !dbg !64
  %i1259 = bitcast float %i1258 to i32, !dbg !64
  %i1260 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1259), !dbg !64
  %i1261 = select i1 %.not36, float %i1257, float %i1251, !dbg !64
  %i1262 = bitcast float %i1261 to i32, !dbg !64
  %i1263 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1262), !dbg !64
  %i1264 = extractvalue [2 x <8 x float>] %i93, 1, !dbg !64
  %i1265 = extractelement <8 x float> %i1264, i64 0, !dbg !64
  %i1266 = extractelement <8 x float> %i1264, i64 1, !dbg !64
  %i1267 = extractelement <8 x float> %i1264, i64 2, !dbg !64
  %i1268 = extractelement <8 x float> %i1264, i64 3, !dbg !64
  %i1269 = select i1 %.not, float %i1266, float %i1265, !dbg !64
  %i1270 = bitcast float %i1269 to i32, !dbg !64
  %i1271 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1270), !dbg !64
  %i1272 = bitcast i32 %i1271 to float, !dbg !64
  %i1273 = select i1 %.not, float %i1265, float %i1272, !dbg !64
  %i1274 = select i1 %.not, float %i1272, float %i1266, !dbg !64
  %i1275 = select i1 %.not, float %i1268, float %i1267, !dbg !64
  %i1276 = bitcast float %i1275 to i32, !dbg !64
  %i1277 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1276), !dbg !64
  %i1278 = bitcast i32 %i1277 to float, !dbg !64
  %i1279 = select i1 %.not, float %i1267, float %i1278, !dbg !64
  %i1280 = select i1 %.not, float %i1278, float %i1268, !dbg !64
  %i1281 = select i1 %.not36, float %i1279, float %i1273, !dbg !64
  %i1282 = bitcast float %i1281 to i32, !dbg !64
  %i1283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1282), !dbg !64
  %i1284 = bitcast i32 %i1283 to float, !dbg !64
  %i1285 = select i1 %.not36, float %i1273, float %i1284, !dbg !64
  %i1286 = select i1 %.not36, float %i1284, float %i1279, !dbg !64
  %i1287 = select i1 %.not36, float %i1280, float %i1274, !dbg !64
  %i1288 = bitcast float %i1287 to i32, !dbg !64
  %i1289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1288), !dbg !64
  %i1290 = bitcast i32 %i1289 to float, !dbg !64
  %i1291 = select i1 %.not36, float %i1274, float %i1290, !dbg !64
  %i1292 = select i1 %.not36, float %i1290, float %i1280, !dbg !64
  %i1293 = extractvalue [2 x <8 x float>] %i92, 1, !dbg !64
  %i1294 = extractelement <8 x float> %i1293, i64 0, !dbg !64
  %i1295 = extractelement <8 x float> %i1293, i64 1, !dbg !64
  %i1296 = extractelement <8 x float> %i1293, i64 2, !dbg !64
  %i1297 = extractelement <8 x float> %i1293, i64 3, !dbg !64
  %i1298 = select i1 %.not, float %i1295, float %i1294, !dbg !64
  %i1299 = bitcast float %i1298 to i32, !dbg !64
  %i1300 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1299), !dbg !64
  %i1301 = bitcast i32 %i1300 to float, !dbg !64
  %i1302 = select i1 %.not, float %i1294, float %i1301, !dbg !64
  %i1303 = select i1 %.not, float %i1301, float %i1295, !dbg !64
  %i1304 = select i1 %.not, float %i1297, float %i1296, !dbg !64
  %i1305 = bitcast float %i1304 to i32, !dbg !64
  %i1306 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1305), !dbg !64
  %i1307 = bitcast i32 %i1306 to float, !dbg !64
  %i1308 = select i1 %.not, float %i1296, float %i1307, !dbg !64
  %i1309 = select i1 %.not, float %i1307, float %i1297, !dbg !64
  %i1310 = select i1 %.not36, float %i1308, float %i1302, !dbg !64
  %i1311 = bitcast float %i1310 to i32, !dbg !64
  %i1312 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1311), !dbg !64
  %i1313 = bitcast i32 %i1312 to float, !dbg !64
  %i1314 = select i1 %.not36, float %i1302, float %i1313, !dbg !64
  %i1315 = select i1 %.not36, float %i1313, float %i1308, !dbg !64
  %i1316 = select i1 %.not36, float %i1309, float %i1303, !dbg !64
  %i1317 = bitcast float %i1316 to i32, !dbg !64
  %i1318 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1317), !dbg !64
  %i1319 = bitcast i32 %i1318 to float, !dbg !64
  %i1320 = select i1 %.not36, float %i1303, float %i1319, !dbg !64
  %i1321 = select i1 %.not36, float %i1319, float %i1309, !dbg !64
  %i1322 = extractvalue [2 x <8 x float>] %i91, 1, !dbg !64
  %i1323 = extractelement <8 x float> %i1322, i64 0, !dbg !64
  %i1324 = extractelement <8 x float> %i1322, i64 1, !dbg !64
  %i1325 = extractelement <8 x float> %i1322, i64 2, !dbg !64
  %i1326 = extractelement <8 x float> %i1322, i64 3, !dbg !64
  %i1327 = select i1 %.not, float %i1324, float %i1323, !dbg !64
  %i1328 = bitcast float %i1327 to i32, !dbg !64
  %i1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1328), !dbg !64
  %i1330 = bitcast i32 %i1329 to float, !dbg !64
  %i1331 = select i1 %.not, float %i1323, float %i1330, !dbg !64
  %i1332 = select i1 %.not, float %i1330, float %i1324, !dbg !64
  %i1333 = select i1 %.not, float %i1326, float %i1325, !dbg !64
  %i1334 = bitcast float %i1333 to i32, !dbg !64
  %i1335 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1334), !dbg !64
  %i1336 = bitcast i32 %i1335 to float, !dbg !64
  %i1337 = select i1 %.not, float %i1325, float %i1336, !dbg !64
  %i1338 = select i1 %.not, float %i1336, float %i1326, !dbg !64
  %i1339 = select i1 %.not36, float %i1337, float %i1331, !dbg !64
  %i1340 = bitcast float %i1339 to i32, !dbg !64
  %i1341 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1340), !dbg !64
  %i1342 = bitcast i32 %i1341 to float, !dbg !64
  %i1343 = select i1 %.not36, float %i1331, float %i1342, !dbg !64
  %i1344 = select i1 %.not36, float %i1342, float %i1337, !dbg !64
  %i1345 = select i1 %.not36, float %i1338, float %i1332, !dbg !64
  %i1346 = bitcast float %i1345 to i32, !dbg !64
  %i1347 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1346), !dbg !64
  %i1348 = bitcast i32 %i1347 to float, !dbg !64
  %i1349 = select i1 %.not36, float %i1332, float %i1348, !dbg !64
  %i1350 = select i1 %.not36, float %i1348, float %i1338, !dbg !64
  %i1351 = extractvalue [2 x <8 x float>] %i88, 1, !dbg !64
  %i1352 = extractelement <8 x float> %i1351, i64 0, !dbg !64
  %i1353 = extractelement <8 x float> %i1351, i64 1, !dbg !64
  %i1354 = extractelement <8 x float> %i1351, i64 2, !dbg !64
  %i1355 = extractelement <8 x float> %i1351, i64 3, !dbg !64
  %i1356 = select i1 %.not, float %i1353, float %i1352, !dbg !64
  %i1357 = bitcast float %i1356 to i32, !dbg !64
  %i1358 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1357), !dbg !64
  %i1359 = bitcast i32 %i1358 to float, !dbg !64
  %i1360 = select i1 %.not, float %i1352, float %i1359, !dbg !64
  %i1361 = select i1 %.not, float %i1359, float %i1353, !dbg !64
  %i1362 = select i1 %.not, float %i1355, float %i1354, !dbg !64
  %i1363 = bitcast float %i1362 to i32, !dbg !64
  %i1364 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1363), !dbg !64
  %i1365 = bitcast i32 %i1364 to float, !dbg !64
  %i1366 = select i1 %.not, float %i1354, float %i1365, !dbg !64
  %i1367 = select i1 %.not, float %i1365, float %i1355, !dbg !64
  %i1368 = select i1 %.not36, float %i1366, float %i1360, !dbg !64
  %i1369 = bitcast float %i1368 to i32, !dbg !64
  %i1370 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1369), !dbg !64
  %i1371 = select i1 %.not36, float %i1367, float %i1361, !dbg !64
  %i1372 = bitcast float %i1371 to i32, !dbg !64
  %i1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1372), !dbg !64
  %i1374 = shufflevector <1 x float> %i1029, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !103
  %i1375 = extractelement <8 x float> %i1264, i64 4, !dbg !64
  %i1376 = extractelement <8 x float> %i1264, i64 5, !dbg !64
  %i1377 = extractelement <8 x float> %i1264, i64 6, !dbg !64
  %i1378 = extractelement <8 x float> %i1264, i64 7, !dbg !64
  %i1379 = select i1 %.not, float %i1376, float %i1375, !dbg !64
  %i1380 = bitcast float %i1379 to i32, !dbg !64
  %i1381 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1380), !dbg !64
  %i1382 = bitcast i32 %i1381 to float, !dbg !64
  %i1383 = select i1 %.not, float %i1375, float %i1382, !dbg !64
  %i1384 = select i1 %.not, float %i1382, float %i1376, !dbg !64
  %i1385 = select i1 %.not, float %i1378, float %i1377, !dbg !64
  %i1386 = bitcast float %i1385 to i32, !dbg !64
  %i1387 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1386), !dbg !64
  %i1388 = bitcast i32 %i1387 to float, !dbg !64
  %i1389 = select i1 %.not, float %i1377, float %i1388, !dbg !64
  %i1390 = select i1 %.not, float %i1388, float %i1378, !dbg !64
  %i1391 = select i1 %.not36, float %i1389, float %i1383, !dbg !64
  %i1392 = bitcast float %i1391 to i32, !dbg !64
  %i1393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1392), !dbg !64
  %i1394 = bitcast i32 %i1393 to float, !dbg !64
  %i1395 = select i1 %.not36, float %i1383, float %i1394, !dbg !64
  %i1396 = select i1 %.not36, float %i1394, float %i1389, !dbg !64
  %i1397 = select i1 %.not36, float %i1390, float %i1384, !dbg !64
  %i1398 = bitcast float %i1397 to i32, !dbg !64
  %i1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1398), !dbg !64
  %i1400 = bitcast i32 %i1399 to float, !dbg !64
  %i1401 = select i1 %.not36, float %i1384, float %i1400, !dbg !64
  %i1402 = select i1 %.not36, float %i1400, float %i1390, !dbg !64
  %i1403 = extractelement <8 x float> %i1293, i64 4, !dbg !64
  %i1404 = extractelement <8 x float> %i1293, i64 5, !dbg !64
  %i1405 = extractelement <8 x float> %i1293, i64 6, !dbg !64
  %i1406 = extractelement <8 x float> %i1293, i64 7, !dbg !64
  %i1407 = select i1 %.not, float %i1404, float %i1403, !dbg !64
  %i1408 = bitcast float %i1407 to i32, !dbg !64
  %i1409 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1408), !dbg !64
  %i1410 = bitcast i32 %i1409 to float, !dbg !64
  %i1411 = select i1 %.not, float %i1403, float %i1410, !dbg !64
  %i1412 = select i1 %.not, float %i1410, float %i1404, !dbg !64
  %i1413 = select i1 %.not, float %i1406, float %i1405, !dbg !64
  %i1414 = bitcast float %i1413 to i32, !dbg !64
  %i1415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1414), !dbg !64
  %i1416 = bitcast i32 %i1415 to float, !dbg !64
  %i1417 = select i1 %.not, float %i1405, float %i1416, !dbg !64
  %i1418 = select i1 %.not, float %i1416, float %i1406, !dbg !64
  %i1419 = select i1 %.not36, float %i1417, float %i1411, !dbg !64
  %i1420 = bitcast float %i1419 to i32, !dbg !64
  %i1421 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1420), !dbg !64
  %i1422 = bitcast i32 %i1421 to float, !dbg !64
  %i1423 = select i1 %.not36, float %i1411, float %i1422, !dbg !64
  %i1424 = select i1 %.not36, float %i1422, float %i1417, !dbg !64
  %i1425 = select i1 %.not36, float %i1418, float %i1412, !dbg !64
  %i1426 = bitcast float %i1425 to i32, !dbg !64
  %i1427 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1426), !dbg !64
  %i1428 = bitcast i32 %i1427 to float, !dbg !64
  %i1429 = select i1 %.not36, float %i1412, float %i1428, !dbg !64
  %i1430 = select i1 %.not36, float %i1428, float %i1418, !dbg !64
  %i1431 = extractelement <8 x float> %i1322, i64 4, !dbg !64
  %i1432 = extractelement <8 x float> %i1322, i64 5, !dbg !64
  %i1433 = extractelement <8 x float> %i1322, i64 6, !dbg !64
  %i1434 = extractelement <8 x float> %i1322, i64 7, !dbg !64
  %i1435 = select i1 %.not, float %i1432, float %i1431, !dbg !64
  %i1436 = bitcast float %i1435 to i32, !dbg !64
  %i1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1436), !dbg !64
  %i1438 = bitcast i32 %i1437 to float, !dbg !64
  %i1439 = select i1 %.not, float %i1431, float %i1438, !dbg !64
  %i1440 = select i1 %.not, float %i1438, float %i1432, !dbg !64
  %i1441 = select i1 %.not, float %i1434, float %i1433, !dbg !64
  %i1442 = bitcast float %i1441 to i32, !dbg !64
  %i1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1442), !dbg !64
  %i1444 = bitcast i32 %i1443 to float, !dbg !64
  %i1445 = select i1 %.not, float %i1433, float %i1444, !dbg !64
  %i1446 = select i1 %.not, float %i1444, float %i1434, !dbg !64
  %i1447 = select i1 %.not36, float %i1445, float %i1439, !dbg !64
  %i1448 = bitcast float %i1447 to i32, !dbg !64
  %i1449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1448), !dbg !64
  %i1450 = bitcast i32 %i1449 to float, !dbg !64
  %i1451 = select i1 %.not36, float %i1439, float %i1450, !dbg !64
  %i1452 = select i1 %.not36, float %i1450, float %i1445, !dbg !64
  %i1453 = select i1 %.not36, float %i1446, float %i1440, !dbg !64
  %i1454 = bitcast float %i1453 to i32, !dbg !64
  %i1455 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1454), !dbg !64
  %i1456 = bitcast i32 %i1455 to float, !dbg !64
  %i1457 = select i1 %.not36, float %i1440, float %i1456, !dbg !64
  %i1458 = select i1 %.not36, float %i1456, float %i1446, !dbg !64
  %i1459 = extractelement <8 x float> %i1351, i64 4, !dbg !64
  %i1460 = extractelement <8 x float> %i1351, i64 5, !dbg !64
  %i1461 = extractelement <8 x float> %i1351, i64 6, !dbg !64
  %i1462 = extractelement <8 x float> %i1351, i64 7, !dbg !64
  %i1463 = select i1 %.not, float %i1460, float %i1459, !dbg !64
  %i1464 = bitcast float %i1463 to i32, !dbg !64
  %i1465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1464), !dbg !64
  %i1466 = bitcast i32 %i1465 to float, !dbg !64
  %i1467 = select i1 %.not, float %i1459, float %i1466, !dbg !64
  %i1468 = select i1 %.not, float %i1466, float %i1460, !dbg !64
  %i1469 = select i1 %.not, float %i1462, float %i1461, !dbg !64
  %i1470 = bitcast float %i1469 to i32, !dbg !64
  %i1471 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1048, i32 %i1470), !dbg !64
  %i1472 = bitcast i32 %i1471 to float, !dbg !64
  %i1473 = select i1 %.not, float %i1461, float %i1472, !dbg !64
  %i1474 = select i1 %.not, float %i1472, float %i1462, !dbg !64
  %i1475 = select i1 %.not36, float %i1473, float %i1467, !dbg !64
  %i1476 = bitcast float %i1475 to i32, !dbg !64
  %i1477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1476), !dbg !64
  %i1478 = select i1 %.not36, float %i1474, float %i1468, !dbg !64
  %i1479 = bitcast float %i1478 to i32, !dbg !64
  %i1480 = call i32 @llvm.amdgcn.ds.bpermute(i32 %i1064, i32 %i1479), !dbg !64
  %i1481 = bitcast i32 %i1477 to float, !dbg !64
  %i1482 = select i1 %.not36, float %i1467, float %i1481, !dbg !64
  %i1483 = insertelement <4 x float> poison, float %i1482, i64 0, !dbg !103
  %i1484 = bitcast i32 %i1480 to float, !dbg !64
  %i1485 = select i1 %.not36, float %i1468, float %i1484, !dbg !64
  %i1486 = insertelement <4 x float> %i1483, float %i1485, i64 1, !dbg !103
  %i1487 = select i1 %.not36, float %i1481, float %i1473, !dbg !64
  %i1488 = insertelement <4 x float> %i1486, float %i1487, i64 2, !dbg !103
  %i1489 = select i1 %.not36, float %i1484, float %i1474, !dbg !64
  %i1490 = insertelement <4 x float> %i1488, float %i1489, i64 3, !dbg !103
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %i1374, !dbg !103
  %normalized = fmul <4 x float> %i1490, %row.reciprocal, !dbg !103
  %i1491 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !104
  %i1492 = extractelement <4 x half> %i1491, i64 3, !dbg !104
  %i1493 = extractelement <4 x half> %i1491, i64 2, !dbg !104
  %i1494 = extractelement <4 x half> %i1491, i64 1, !dbg !104
  %i1495 = extractelement <4 x half> %i1491, i64 0, !dbg !104
  %i1496 = bitcast i32 %i1370 to float, !dbg !64
  %i1497 = select i1 %.not36, float %i1360, float %i1496, !dbg !64
  %i1498 = insertelement <4 x float> poison, float %i1497, i64 0, !dbg !103
  %i1499 = bitcast i32 %i1373 to float, !dbg !64
  %i1500 = select i1 %.not36, float %i1361, float %i1499, !dbg !64
  %i1501 = insertelement <4 x float> %i1498, float %i1500, i64 1, !dbg !103
  %i1502 = select i1 %.not36, float %i1496, float %i1366, !dbg !64
  %i1503 = insertelement <4 x float> %i1501, float %i1502, i64 2, !dbg !103
  %i1504 = select i1 %.not36, float %i1499, float %i1367, !dbg !64
  %i1505 = insertelement <4 x float> %i1503, float %i1504, i64 3, !dbg !103
  %normalized37 = fmul <4 x float> %i1505, %row.reciprocal, !dbg !103
  %i1506 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !104
  %i1507 = extractelement <4 x half> %i1506, i64 3, !dbg !104
  %i1508 = extractelement <4 x half> %i1506, i64 2, !dbg !104
  %i1509 = extractelement <4 x half> %i1506, i64 1, !dbg !104
  %i1510 = extractelement <4 x half> %i1506, i64 0, !dbg !104
  %i1511 = bitcast i32 %i1260 to float, !dbg !64
  %i1512 = select i1 %.not36, float %i1250, float %i1511, !dbg !64
  %i1513 = insertelement <4 x float> poison, float %i1512, i64 0, !dbg !103
  %i1514 = bitcast i32 %i1263 to float, !dbg !64
  %i1515 = select i1 %.not36, float %i1251, float %i1514, !dbg !64
  %i1516 = insertelement <4 x float> %i1513, float %i1515, i64 1, !dbg !103
  %i1517 = select i1 %.not36, float %i1511, float %i1256, !dbg !64
  %i1518 = insertelement <4 x float> %i1516, float %i1517, i64 2, !dbg !103
  %i1519 = select i1 %.not36, float %i1514, float %i1257, !dbg !64
  %i1520 = insertelement <4 x float> %i1518, float %i1519, i64 3, !dbg !103
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %i1157, !dbg !103
  %normalized39 = fmul <4 x float> %i1520, %row.reciprocal38, !dbg !103
  %i1521 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !104
  %i1522 = extractelement <4 x half> %i1521, i64 3, !dbg !104
  %i1523 = extractelement <4 x half> %i1521, i64 2, !dbg !104
  %i1524 = extractelement <4 x half> %i1521, i64 1, !dbg !104
  %i1525 = extractelement <4 x half> %i1521, i64 0, !dbg !104
  %i1526 = bitcast i32 %i1153 to float, !dbg !64
  %i1527 = select i1 %.not36, float %i1143, float %i1526, !dbg !64
  %i1528 = insertelement <4 x float> poison, float %i1527, i64 0, !dbg !103
  %i1529 = bitcast i32 %i1156 to float, !dbg !64
  %i1530 = select i1 %.not36, float %i1144, float %i1529, !dbg !64
  %i1531 = insertelement <4 x float> %i1528, float %i1530, i64 1, !dbg !103
  %i1532 = select i1 %.not36, float %i1526, float %i1149, !dbg !64
  %i1533 = insertelement <4 x float> %i1531, float %i1532, i64 2, !dbg !103
  %i1534 = select i1 %.not36, float %i1529, float %i1150, !dbg !64
  %i1535 = insertelement <4 x float> %i1533, float %i1534, i64 3, !dbg !103
  %normalized40 = fmul <4 x float> %i1535, %row.reciprocal38, !dbg !103
  %i1536 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !104
  %i1537 = extractelement <4 x half> %i1536, i64 3, !dbg !104
  %i1538 = extractelement <4 x half> %i1536, i64 2, !dbg !104
  %i1539 = extractelement <4 x half> %i1536, i64 1, !dbg !104
  %i1540 = extractelement <4 x half> %i1536, i64 0, !dbg !104
  %address.thread974 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !105, !range !62, !frisk.register_lifetimes !58
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !61
  %address.local976 = add nuw i64 %i17, %address.local975, !dbg !105
  %i1541 = lshr i64 %address.local975, 2, !dbg !105
  %address.local978 = and i64 %i1541, 48, !dbg !105
  %address.local979 = add i64 %address.local976, %address.local978, !dbg !105
  %address.local980 = and i64 %address.local975, 240, !dbg !105
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !105
  %address.local982 = add i64 %address.local981, %i18, !dbg !105
  %address.local983 = shl nuw nsw i64 %address.local982, 7, !dbg !105
  %i1542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %arg3, i64 %i21, !dbg !105
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %i1542, i64 %address.local983, !dbg !105
  %address.local987 = and i64 %i1541, 12, !dbg !63
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !105
  store half %i1540, ptr addrspace(1) %address.local988, align 2, !dbg !105
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !105
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !105
  store half %i1539, ptr addrspace(1) %address.local991, align 2, !dbg !105
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !105
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !105
  store half %i1538, ptr addrspace(1) %address.local994, align 2, !dbg !105
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !105
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !105
  store half %i1537, ptr addrspace(1) %address.local997, align 2, !dbg !105
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !105
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !105
  store half %i1525, ptr addrspace(1) %address.local1000, align 2, !dbg !105
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !105
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !105
  store half %i1524, ptr addrspace(1) %address.local1003, align 2, !dbg !105
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !105
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !105
  store half %i1523, ptr addrspace(1) %address.local1006, align 2, !dbg !105
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !105
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !105
  store half %i1522, ptr addrspace(1) %address.local1009, align 2, !dbg !105
  %address.local1011 = add nuw i64 %address.local983, 8192, !dbg !105
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %i1542, i64 %address.local1011, !dbg !105
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !105
  store half %i1510, ptr addrspace(1) %address.local1014, align 2, !dbg !105
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !105
  store half %i1509, ptr addrspace(1) %address.local1016, align 2, !dbg !105
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !105
  store half %i1508, ptr addrspace(1) %address.local1018, align 2, !dbg !105
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !105
  store half %i1507, ptr addrspace(1) %address.local1020, align 2, !dbg !105
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !105
  store half %i1495, ptr addrspace(1) %address.local1022, align 2, !dbg !105
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !105
  store half %i1494, ptr addrspace(1) %address.local1024, align 2, !dbg !105
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !105
  store half %i1493, ptr addrspace(1) %address.local1026, align 2, !dbg !105
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !105
  store half %i1492, ptr addrspace(1) %address.local1028, align 2, !dbg !105
  %i1543 = insertelement <4 x float> poison, float %i1451, i64 0, !dbg !103
  %i1544 = insertelement <4 x float> %i1543, float %i1457, i64 1, !dbg !103
  %i1545 = insertelement <4 x float> %i1544, float %i1452, i64 2, !dbg !103
  %i1546 = insertelement <4 x float> %i1545, float %i1458, i64 3, !dbg !103
  %normalized42 = fmul <4 x float> %i1546, %row.reciprocal, !dbg !103
  %i1547 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !104
  %i1548 = extractelement <4 x half> %i1547, i64 3, !dbg !104
  %i1549 = extractelement <4 x half> %i1547, i64 2, !dbg !104
  %i1550 = extractelement <4 x half> %i1547, i64 1, !dbg !104
  %i1551 = extractelement <4 x half> %i1547, i64 0, !dbg !104
  %i1552 = insertelement <4 x float> poison, float %i1343, i64 0, !dbg !103
  %i1553 = insertelement <4 x float> %i1552, float %i1349, i64 1, !dbg !103
  %i1554 = insertelement <4 x float> %i1553, float %i1344, i64 2, !dbg !103
  %i1555 = insertelement <4 x float> %i1554, float %i1350, i64 3, !dbg !103
  %normalized43 = fmul <4 x float> %i1555, %row.reciprocal, !dbg !103
  %i1556 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !104
  %i1557 = extractelement <4 x half> %i1556, i64 3, !dbg !104
  %i1558 = extractelement <4 x half> %i1556, i64 2, !dbg !104
  %i1559 = extractelement <4 x half> %i1556, i64 1, !dbg !104
  %i1560 = extractelement <4 x half> %i1556, i64 0, !dbg !104
  %i1561 = insertelement <4 x float> poison, float %i1234, i64 0, !dbg !103
  %i1562 = insertelement <4 x float> %i1561, float %i1240, i64 1, !dbg !103
  %i1563 = insertelement <4 x float> %i1562, float %i1235, i64 2, !dbg !103
  %i1564 = insertelement <4 x float> %i1563, float %i1241, i64 3, !dbg !103
  %normalized45 = fmul <4 x float> %i1564, %row.reciprocal38, !dbg !103
  %i1565 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !104
  %i1566 = extractelement <4 x half> %i1565, i64 3, !dbg !104
  %i1567 = extractelement <4 x half> %i1565, i64 2, !dbg !104
  %i1568 = extractelement <4 x half> %i1565, i64 1, !dbg !104
  %i1569 = extractelement <4 x half> %i1565, i64 0, !dbg !104
  %i1570 = insertelement <4 x float> poison, float %i1126, i64 0, !dbg !103
  %i1571 = insertelement <4 x float> %i1570, float %i1132, i64 1, !dbg !103
  %i1572 = insertelement <4 x float> %i1571, float %i1127, i64 2, !dbg !103
  %i1573 = insertelement <4 x float> %i1572, float %i1133, i64 3, !dbg !103
  %normalized46 = fmul <4 x float> %i1573, %row.reciprocal38, !dbg !103
  %i1574 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !104
  %i1575 = extractelement <4 x half> %i1574, i64 3, !dbg !104
  %i1576 = extractelement <4 x half> %i1574, i64 2, !dbg !104
  %i1577 = extractelement <4 x half> %i1574, i64 1, !dbg !104
  %i1578 = extractelement <4 x half> %i1574, i64 0, !dbg !104
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !105
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !105
  store half %i1578, ptr addrspace(1) %address.local1044, align 2, !dbg !105
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !105
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !105
  store half %i1577, ptr addrspace(1) %address.local1047, align 2, !dbg !105
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !105
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !105
  store half %i1576, ptr addrspace(1) %address.local1050, align 2, !dbg !105
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !105
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !105
  store half %i1575, ptr addrspace(1) %address.local1053, align 2, !dbg !105
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !105
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !105
  store half %i1569, ptr addrspace(1) %address.local1056, align 2, !dbg !105
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !105
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !105
  store half %i1568, ptr addrspace(1) %address.local1059, align 2, !dbg !105
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !105
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !105
  store half %i1567, ptr addrspace(1) %address.local1062, align 2, !dbg !105
  %address.local1064 = or i64 %i1541, 51, !dbg !105
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !105
  store half %i1566, ptr addrspace(1) %address.local1065, align 2, !dbg !105
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !105
  store half %i1560, ptr addrspace(1) %address.local1070, align 2, !dbg !105
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !105
  store half %i1559, ptr addrspace(1) %address.local1072, align 2, !dbg !105
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !105
  store half %i1558, ptr addrspace(1) %address.local1074, align 2, !dbg !105
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !105
  store half %i1557, ptr addrspace(1) %address.local1076, align 2, !dbg !105
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !105
  store half %i1551, ptr addrspace(1) %address.local1078, align 2, !dbg !105
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !105
  store half %i1550, ptr addrspace(1) %address.local1080, align 2, !dbg !105
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !105
  store half %i1549, ptr addrspace(1) %address.local1082, align 2, !dbg !105
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !105
  store half %i1548, ptr addrspace(1) %address.local1084, align 2, !dbg !105
  %i1579 = insertelement <4 x float> poison, float %i1423, i64 0, !dbg !103
  %i1580 = insertelement <4 x float> %i1579, float %i1429, i64 1, !dbg !103
  %i1581 = insertelement <4 x float> %i1580, float %i1424, i64 2, !dbg !103
  %i1582 = insertelement <4 x float> %i1581, float %i1430, i64 3, !dbg !103
  %normalized48 = fmul <4 x float> %i1582, %row.reciprocal, !dbg !103
  %i1583 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !104
  %i1584 = extractelement <4 x half> %i1583, i64 3, !dbg !104
  %i1585 = extractelement <4 x half> %i1583, i64 2, !dbg !104
  %i1586 = extractelement <4 x half> %i1583, i64 1, !dbg !104
  %i1587 = extractelement <4 x half> %i1583, i64 0, !dbg !104
  %i1588 = insertelement <4 x float> poison, float %i1314, i64 0, !dbg !103
  %i1589 = insertelement <4 x float> %i1588, float %i1320, i64 1, !dbg !103
  %i1590 = insertelement <4 x float> %i1589, float %i1315, i64 2, !dbg !103
  %i1591 = insertelement <4 x float> %i1590, float %i1321, i64 3, !dbg !103
  %normalized49 = fmul <4 x float> %i1591, %row.reciprocal, !dbg !103
  %i1592 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !104
  %i1593 = extractelement <4 x half> %i1592, i64 3, !dbg !104
  %i1594 = extractelement <4 x half> %i1592, i64 2, !dbg !104
  %i1595 = extractelement <4 x half> %i1592, i64 1, !dbg !104
  %i1596 = extractelement <4 x half> %i1592, i64 0, !dbg !104
  %i1597 = insertelement <4 x float> poison, float %i1206, i64 0, !dbg !103
  %i1598 = insertelement <4 x float> %i1597, float %i1212, i64 1, !dbg !103
  %i1599 = insertelement <4 x float> %i1598, float %i1207, i64 2, !dbg !103
  %i1600 = insertelement <4 x float> %i1599, float %i1213, i64 3, !dbg !103
  %normalized51 = fmul <4 x float> %i1600, %row.reciprocal38, !dbg !103
  %i1601 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !104
  %i1602 = extractelement <4 x half> %i1601, i64 3, !dbg !104
  %i1603 = extractelement <4 x half> %i1601, i64 2, !dbg !104
  %i1604 = extractelement <4 x half> %i1601, i64 1, !dbg !104
  %i1605 = extractelement <4 x half> %i1601, i64 0, !dbg !104
  %i1606 = insertelement <4 x float> poison, float %i1097, i64 0, !dbg !103
  %i1607 = insertelement <4 x float> %i1606, float %i1103, i64 1, !dbg !103
  %i1608 = insertelement <4 x float> %i1607, float %i1098, i64 2, !dbg !103
  %i1609 = insertelement <4 x float> %i1608, float %i1104, i64 3, !dbg !103
  %normalized52 = fmul <4 x float> %i1609, %row.reciprocal38, !dbg !103
  %i1610 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !104
  %i1611 = extractelement <4 x half> %i1610, i64 3, !dbg !104
  %i1612 = extractelement <4 x half> %i1610, i64 2, !dbg !104
  %i1613 = extractelement <4 x half> %i1610, i64 1, !dbg !104
  %i1614 = extractelement <4 x half> %i1610, i64 0, !dbg !104
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !105
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !105
  store half %i1614, ptr addrspace(1) %address.local1100, align 2, !dbg !105
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !105
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !105
  store half %i1613, ptr addrspace(1) %address.local1103, align 2, !dbg !105
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !105
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !105
  store half %i1612, ptr addrspace(1) %address.local1106, align 2, !dbg !105
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !105
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !105
  store half %i1611, ptr addrspace(1) %address.local1109, align 2, !dbg !105
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !105
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !105
  store half %i1605, ptr addrspace(1) %address.local1112, align 2, !dbg !105
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !105
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !105
  store half %i1604, ptr addrspace(1) %address.local1115, align 2, !dbg !105
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !105
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !105
  store half %i1603, ptr addrspace(1) %address.local1118, align 2, !dbg !105
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !105
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !105
  store half %i1602, ptr addrspace(1) %address.local1121, align 2, !dbg !105
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !105
  store half %i1596, ptr addrspace(1) %address.local1126, align 2, !dbg !105
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !105
  store half %i1595, ptr addrspace(1) %address.local1128, align 2, !dbg !105
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !105
  store half %i1594, ptr addrspace(1) %address.local1130, align 2, !dbg !105
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !105
  store half %i1593, ptr addrspace(1) %address.local1132, align 2, !dbg !105
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !105
  store half %i1587, ptr addrspace(1) %address.local1134, align 2, !dbg !105
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !105
  store half %i1586, ptr addrspace(1) %address.local1136, align 2, !dbg !105
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !105
  store half %i1585, ptr addrspace(1) %address.local1138, align 2, !dbg !105
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !105
  store half %i1584, ptr addrspace(1) %address.local1140, align 2, !dbg !105
  %i1615 = insertelement <4 x float> poison, float %i1395, i64 0, !dbg !103
  %i1616 = insertelement <4 x float> %i1615, float %i1401, i64 1, !dbg !103
  %i1617 = insertelement <4 x float> %i1616, float %i1396, i64 2, !dbg !103
  %i1618 = insertelement <4 x float> %i1617, float %i1402, i64 3, !dbg !103
  %normalized54 = fmul <4 x float> %i1618, %row.reciprocal, !dbg !103
  %i1619 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !104
  %i1620 = extractelement <4 x half> %i1619, i64 3, !dbg !104
  %i1621 = extractelement <4 x half> %i1619, i64 2, !dbg !104
  %i1622 = extractelement <4 x half> %i1619, i64 1, !dbg !104
  %i1623 = extractelement <4 x half> %i1619, i64 0, !dbg !104
  %i1624 = insertelement <4 x float> poison, float %i1285, i64 0, !dbg !103
  %i1625 = insertelement <4 x float> %i1624, float %i1291, i64 1, !dbg !103
  %i1626 = insertelement <4 x float> %i1625, float %i1286, i64 2, !dbg !103
  %i1627 = insertelement <4 x float> %i1626, float %i1292, i64 3, !dbg !103
  %normalized55 = fmul <4 x float> %i1627, %row.reciprocal, !dbg !103
  %i1628 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !104
  %i1629 = extractelement <4 x half> %i1628, i64 3, !dbg !104
  %i1630 = extractelement <4 x half> %i1628, i64 2, !dbg !104
  %i1631 = extractelement <4 x half> %i1628, i64 1, !dbg !104
  %i1632 = extractelement <4 x half> %i1628, i64 0, !dbg !104
  %i1633 = insertelement <4 x float> poison, float %i1178, i64 0, !dbg !103
  %i1634 = insertelement <4 x float> %i1633, float %i1184, i64 1, !dbg !103
  %i1635 = insertelement <4 x float> %i1634, float %i1179, i64 2, !dbg !103
  %i1636 = insertelement <4 x float> %i1635, float %i1185, i64 3, !dbg !103
  %normalized57 = fmul <4 x float> %i1636, %row.reciprocal38, !dbg !103
  %i1637 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !104
  %i1638 = extractelement <4 x half> %i1637, i64 3, !dbg !104
  %i1639 = extractelement <4 x half> %i1637, i64 2, !dbg !104
  %i1640 = extractelement <4 x half> %i1637, i64 1, !dbg !104
  %i1641 = extractelement <4 x half> %i1637, i64 0, !dbg !104
  %i1642 = insertelement <4 x float> poison, float %i1068, i64 0, !dbg !103
  %i1643 = insertelement <4 x float> %i1642, float %i1074, i64 1, !dbg !103
  %i1644 = insertelement <4 x float> %i1643, float %i1069, i64 2, !dbg !103
  %i1645 = insertelement <4 x float> %i1644, float %i1075, i64 3, !dbg !103
  %normalized58 = fmul <4 x float> %i1645, %row.reciprocal38, !dbg !103
  %i1646 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !104
  %i1647 = extractelement <4 x half> %i1646, i64 3, !dbg !104
  %i1648 = extractelement <4 x half> %i1646, i64 2, !dbg !104
  %i1649 = extractelement <4 x half> %i1646, i64 1, !dbg !104
  %i1650 = extractelement <4 x half> %i1646, i64 0, !dbg !104
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !105
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !105
  store half %i1650, ptr addrspace(1) %address.local1156, align 2, !dbg !105
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !105
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !105
  store half %i1649, ptr addrspace(1) %address.local1159, align 2, !dbg !105
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !105
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !105
  store half %i1648, ptr addrspace(1) %address.local1162, align 2, !dbg !105
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !105
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !105
  store half %i1647, ptr addrspace(1) %address.local1165, align 2, !dbg !105
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !105
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !105
  store half %i1641, ptr addrspace(1) %address.local1168, align 2, !dbg !105
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !105
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !105
  store half %i1640, ptr addrspace(1) %address.local1171, align 2, !dbg !105
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !105
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !105
  store half %i1639, ptr addrspace(1) %address.local1174, align 2, !dbg !105
  %address.local1176 = or i64 %i1541, 115, !dbg !105
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !105
  store half %i1638, ptr addrspace(1) %address.local1177, align 2, !dbg !105
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !105
  store half %i1632, ptr addrspace(1) %address.local1182, align 2, !dbg !105
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !105
  store half %i1631, ptr addrspace(1) %address.local1184, align 2, !dbg !105
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !105
  store half %i1630, ptr addrspace(1) %address.local1186, align 2, !dbg !105
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !105
  store half %i1629, ptr addrspace(1) %address.local1188, align 2, !dbg !105
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !105
  store half %i1623, ptr addrspace(1) %address.local1190, align 2, !dbg !105
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !105
  store half %i1622, ptr addrspace(1) %address.local1192, align 2, !dbg !105
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !105
  store half %i1621, ptr addrspace(1) %address.local1194, align 2, !dbg !105
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !105
  store half %i1620, ptr addrspace(1) %address.local1196, align 2, !dbg !105
  ret void, !dbg !60
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

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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

!llvm.dbg.cu = !{!52}
!llvm.module.flags = !{!54}

!52 = distinct !DICompileUnit(language: DW_LANG_C, file: !53, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!53 = !DIFile(filename: "input.mlir", directory: "generated_ir/noncausal_k_lds_prefetch_pass/mytest_final/run-20261030-171621-98fko2hz/online-vdb-k64-lds2")
!54 = !{i32 2, !"Debug Info Version", i32 3}
!56 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !53, file: !53, line: 23, type: !57, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !52)
!57 = !DISubroutineType(types: !58)
!58 = !{}
!59 = !{i32 256, i32 1, i32 1}
!60 = !DILocation(line: 23, column: 3, scope: !56)
!61 = !DILocation(line: 67, column: 22, scope: !56)
!62 = !{i32 0, i32 256}
!63 = !DILocation(line: 69, column: 24, scope: !56)
!64 = !DILocation(line: 57, column: 24, scope: !56)
!65 = !DILocation(line: 70, column: 26, scope: !56)
!66 = !DILocation(line: 68, column: 22, scope: !56)
!67 = !{i32 0}
!68 = !{i32 1}
!69 = !{i32 2}
!70 = !{i32 3}
!71 = !{i32 4}
!72 = !{i32 5}
!73 = !{i32 6}
!74 = !{i32 7}
!75 = !{i32 8}
!76 = !{i32 9}
!77 = !{i32 10}
!78 = !{i32 11}
!79 = !{i32 12}
!80 = !{i32 13}
!81 = !{i32 14}
!82 = !{i32 15}
!83 = !{i32 16}
!84 = !{i32 17}
!85 = !{i32 18}
!86 = !{i32 19}
!87 = !{i32 20}
!88 = !{i32 21}
!89 = !{i32 22}
!90 = !DILocation(line: 86, column: 20, scope: !56)
!91 = !DILocation(line: 71, column: 23, scope: !56)
!92 = !DILocation(line: 74, column: 20, scope: !56)
!93 = !DILocation(line: 75, column: 22, scope: !56)
!94 = !DILocation(line: 76, column: 23, scope: !56)
!95 = !DILocation(line: 77, column: 20, scope: !56)
!96 = !DILocation(line: 88, column: 22, scope: !56)
!97 = !DILocation(line: 78, column: 22, scope: !56)
!98 = !DILocation(line: 79, column: 22, scope: !56)
!99 = !DILocation(line: 81, column: 16, scope: !56)
!100 = !DILocation(line: 80, column: 20, scope: !56)
!101 = !DILocation(line: 87, column: 23, scope: !56)
!102 = !{i64 8192}
!103 = !DILocation(line: 92, column: 23, scope: !56)
!104 = !DILocation(line: 93, column: 19, scope: !56)
!105 = !DILocation(line: 94, column: 9, scope: !56)
