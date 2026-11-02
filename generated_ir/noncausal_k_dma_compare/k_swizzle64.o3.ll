; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/k_swizzle64.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_frag_prefetch_forward/control/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !reqd_work_group_size !1 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %kdma.wave = and i32 %kdma.tid, 960
  %kdma.lane = lshr i32 %kdma.tid, 1
  %5 = and i32 %kdma.lane, 31
  %kdma.owner = or disjoint i32 %5, %kdma.wave
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32
  %6 = shl nuw nsw i32 %kdma.tid, 3
  %kdma.intra = and i32 %6, 8
  %kdma.intra.hi = xor i32 %kdma.intra, 8
  %grp.n5 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %grp.n6 = sext i32 %grp.n5 to i64
  %grp.n7 = tail call i32 @llvm.amdgcn.workgroup.id.y()
  %grp.n8 = sext i32 %grp.n7 to i64
  %grp.n9 = shl nsw i64 %grp.n8, 5
  %grp.n10 = add nsw i64 %grp.n9, %grp.n6
  %grp.n11 = lshr i64 %grp.n10, 5
  %grp.n12 = and i64 %grp.n11, 576460752303423456
  %grp.n13 = and i64 %grp.n6, 31
  %grp.n14 = or disjoint i64 %grp.n12, %grp.n13
  %grp.n16 = lshr i64 %grp.n10, 10
  %grp.n17 = add nuw nsw i64 %grp.n11, %grp.n16
  %grp.n19 = shl nuw nsw i64 %grp.n14, 7
  %7 = shl i64 %grp.n16, 12
  %grp.n21 = sub i64 %grp.n19, %7
  %grp.n18 = shl i64 %grp.n17, 19
  %grp.n22 = and i64 %grp.n18, 16252928
  %grp.n23 = shl nsw i64 %grp.n21, 7
  %grp.n24 = add i64 %grp.n23, %grp.n22
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !range !2, !frisk.register_lifetimes !3
  %address.local = zext nneg i32 %address.thread to i64
  %address.local404 = shl nuw nsw i64 %address.local, 3
  %address.local405 = and i64 %address.local404, 1536
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405
  %address.local409 = shl nuw nsw i64 %address.local, 7
  %address.local410 = and i64 %address.local409, 16256
  %address.local411 = lshr i64 %address.local, 4
  %address.local412 = and i64 %address.local411, 8
  %address.local413 = or disjoint i64 %address.local410, %address.local412
  %grp.n28 = ptrtoint ptr addrspace(1) %0 to i64
  %.idx1399 = or disjoint i64 %address.local413, %grp.n24
  %.idx1399.tr = trunc i64 %.idx1399 to i32
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1
  %grp.n29 = insertelement <2 x i64> poison, i64 %grp.n28, i64 0
  %grp.n30 = bitcast <2 x i64> %grp.n29 to <4 x i32>
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %grp.n30, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %address.local415 = or disjoint i64 %address.local, 256
  %address.local416 = shl nuw nsw i64 %address.local415, 3
  %address.local417 = and i64 %address.local416, 3584
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417
  %address.local423 = lshr i64 %address.local415, 4
  %address.local424 = and i64 %address.local423, 24
  %address.local425 = or disjoint i64 %address.local424, %address.local410
  %.idx1400 = or disjoint i64 %address.local425, %grp.n24
  %.idx1400.tr = trunc i64 %.idx1400 to i32
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405
  %grp.n39 = or disjoint i64 %grp.n24, 32
  %grp.n40 = or disjoint i64 %address.local413, %grp.n39
  %.tr = trunc i64 %grp.n40 to i32
  %dma.buffer.offset1390 = shl i32 %.tr, 1
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417
  %8 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local406, i64 0
  %9 = insertelement <4 x ptr addrspace(3)> %8, ptr addrspace(3) %address.local418, i64 1
  %10 = insertelement <4 x ptr addrspace(3)> %9, ptr addrspace(3) %address.local427, i64 2
  %11 = insertelement <4 x ptr addrspace(3)> %10, ptr addrspace(3) %address.local431, i64 3
  %12 = ptrtoint <4 x ptr addrspace(3)> %11 to <4 x i64>
  %13 = trunc <4 x i64> %12 to <4 x i32>
  %14 = extractelement <4 x i32> %13, i64 0
  %grp.n27 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %14)
  %dma.buffer.lds1398 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n27)
  %grp.n31 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #9
  %15 = extractelement <4 x i32> %13, i64 1
  %grp.n32 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %15)
  %dma.buffer.lds1395 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n32)
  %grp.n36 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %16 = extractelement <4 x i32> %13, i64 2
  %grp.n37 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %16)
  %dma.buffer.lds1392 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n37)
  %grp.n43 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #9
  %17 = extractelement <4 x i32> %13, i64 3
  %grp.n44 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %17)
  %grp.n47 = or disjoint i64 %address.local425, %grp.n39
  %.tr1401 = trunc i64 %grp.n47 to i32
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1
  %dma.buffer.lds1389 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n44)
  %grp.n50 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #9
  %address.local435 = lshr i32 %address.thread, 2
  %address.local436 = and i32 %address.local435, 4
  %address.local437 = shl nuw nsw i32 %address.thread, 1
  %address.local438 = and i32 %address.local437, 384
  %address.local439 = or disjoint i32 %address.local436, %address.local438
  %address.local440 = shl nuw nsw i32 %address.thread, 3
  %address.local441 = and i32 %address.local440, 120
  %address.local442 = or disjoint i32 %address.local439, %address.local441
  %address.local443 = shl nuw nsw i32 %address.thread, 5
  %address.local444 = and i32 %address.local443, 1024
  %address.local445 = or disjoint i32 %address.local442, %address.local444
  %address.local446 = zext nneg i32 %address.local445 to i64
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local446
  %grp.n51 = load <4 x half>, ptr addrspace(3) %address.local447, align 8
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096
  %grp.n52 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8
  %grp.n53 = fmul <4 x half> %grp.n51, splat (half 1.275630e-01)
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024
  %grp.n54 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8
  %grp.n55 = fmul <4 x half> %grp.n52, splat (half 1.275630e-01)
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120
  %grp.n56 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8
  %grp.n57 = fmul <4 x half> %grp.n54, splat (half 1.275630e-01)
  %grp.n58 = fmul <4 x half> %grp.n56, splat (half 1.275630e-01)
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %grp.n61 = or disjoint i64 %grp.n24, 64
  %grp.n62 = or disjoint i64 %address.local413, %grp.n61
  %.tr1402 = trunc i64 %grp.n62 to i32
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1
  %grp.n65 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #9
  %grp.n69 = or disjoint i64 %address.local425, %grp.n61
  %.tr1403 = trunc i64 %grp.n69 to i32
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1
  %grp.n72 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #9
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192
  %grp.n73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288
  %grp.n74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216
  %grp.n75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8
  %grp.n76 = fmul <4 x half> %grp.n73, splat (half 1.275630e-01)
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312
  %grp.n77 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8
  %grp.n78 = fmul <4 x half> %grp.n74, splat (half 1.275630e-01)
  %grp.n79 = fmul <4 x half> %grp.n75, splat (half 1.275630e-01)
  %grp.n80 = fmul <4 x half> %grp.n77, splat (half 1.275630e-01)
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %grp.n83 = or disjoint i64 %grp.n24, 96
  %grp.n84 = or disjoint i64 %address.local413, %grp.n83
  %.tr1404 = trunc i64 %grp.n84 to i32
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1
  %grp.n87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #9
  %grp.n91 = or disjoint i64 %address.local425, %grp.n83
  %.tr1405 = trunc i64 %grp.n91 to i32
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1
  %grp.n94 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #9
  %grp.n95 = load <4 x half>, ptr addrspace(3) %address.local447, align 8
  %grp.n96 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8
  %grp.n97 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8
  %grp.n98 = fmul <4 x half> %grp.n95, splat (half 1.275630e-01)
  %grp.n99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8
  %grp.n100 = fmul <4 x half> %grp.n96, splat (half 1.275630e-01)
  %grp.n101 = fmul <4 x half> %grp.n97, splat (half 1.275630e-01)
  %grp.n102 = fmul <4 x half> %grp.n99, splat (half 1.275630e-01)
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %grp.n103 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8
  %grp.n104 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8
  %grp.n105 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8
  %grp.n106 = fmul <4 x half> %grp.n103, splat (half 1.275630e-01)
  %grp.n107 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8
  %grp.n108 = fmul <4 x half> %grp.n104, splat (half 1.275630e-01)
  %grp.n109 = fmul <4 x half> %grp.n105, splat (half 1.275630e-01)
  %grp.n110 = fmul <4 x half> %grp.n107, splat (half 1.275630e-01)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %18 = shl nuw nsw i32 %kdma.tid, 2
  %19 = and i32 %18, 12
  %grp.n115 = and i32 %kdma.tid, 1
  %grp.n1161 = add nuw nsw i32 %grp.n115, 1
  %narrow = or disjoint i32 %grp.n1161, %19
  %20 = lshr i32 %narrow, 1
  %grp.n118 = zext nneg i32 %20 to i64
  %grp.n129 = shl i64 %grp.n11, 19
  %grp.n130 = and i64 %grp.n129, 16252928
  %grp.n134 = ptrtoint ptr addrspace(1) %1 to i64
  %grp.n135 = insertelement <2 x i64> poison, i64 %grp.n134, i64 0
  %grp.n136 = bitcast <2 x i64> %grp.n135 to <4 x i32>
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %grp.n136, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %grp.n377 = ptrtoint ptr addrspace(1) %2 to i64
  %grp.n378 = insertelement <2 x i64> poison, i64 %grp.n377, i64 0
  %grp.n379 = bitcast <2 x i64> %grp.n378 to <4 x i32>
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %grp.n379, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %kdma.source = zext nneg i32 %kdma.owner to i64
  %kdma.source1 = lshr i64 %kdma.source, 2
  %kdma.source2 = and i64 %kdma.source1, 4
  %kdma.source3 = add nuw nsw i64 %kdma.source2, %kdma.source
  %kdma.source4 = and i64 %kdma.source3, 4
  %kdma.source5 = and i64 %kdma.source, 91
  %kdma.source6 = or disjoint i64 %kdma.source4, %kdma.source5
  %kdma.source7 = shl nuw nsw i64 %kdma.source6, 7
  %kdma.source8 = lshr i64 %kdma.source, 4
  %kdma.source9 = and i64 %kdma.source8, 8
  %kdma.source10 = or disjoint i64 %kdma.source7, %kdma.source9
  %kdma.source14 = zext nneg i32 %kdma.owner.hi to i64
  %kdma.source15 = lshr i64 %kdma.source14, 2
  %kdma.source16 = and i64 %kdma.source15, 4
  %kdma.source17 = add nuw nsw i64 %kdma.source16, %kdma.source14
  %kdma.source18 = and i64 %kdma.source17, 4
  %kdma.source19 = and i64 %kdma.source14, 123
  %kdma.source20 = or disjoint i64 %kdma.source18, %kdma.source19
  %kdma.source21 = shl nuw nsw i64 %kdma.source20, 7
  %kdma.source22 = lshr i64 %kdma.source14, 4
  %kdma.source23 = and i64 %kdma.source22, 8
  %kdma.source24 = or disjoint i64 %kdma.source21, %kdma.source23
  %kdma.source41 = or disjoint i64 %kdma.source10, 16
  %kdma.source58 = or disjoint i64 %kdma.source24, 16
  br label %grp.n128

grp.n128:                                         ; preds = %grp.n128, %4
  %grp.n12616 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %grp.n1145, %grp.n128 ]
  %grp.n12515 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %grp.n1141, %grp.n128 ]
  %grp.n12414 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %grp.n1137, %grp.n128 ]
  %grp.n12313 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %grp.n1133, %grp.n128 ]
  %grp.n12212 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %grp.n801, %grp.n128 ]
  %grp.n12111 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %grp.n1131, %grp.n128 ]
  %grp.n12010 = phi i64 [ 0, %4 ], [ %grp.n1146, %grp.n128 ]
  %grp.n131 = shl nuw nsw i64 %grp.n12010, 14
  %grp.n132 = add nuw nsw i64 %grp.n131, %grp.n130
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !range !2, !frisk.register_lifetimes !3
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %address.local500 = shl nuw nsw i64 %address.local499, 3
  %address.local501 = and i64 %address.local500, 1536
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501
  %address.local508 = lshr i64 %address.local499, 2
  %address.local516 = or disjoint i64 %address.local499, 256
  %address.local517 = shl nuw nsw i64 %address.local516, 3
  %address.local518 = and i64 %address.local517, 3584
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518
  %grp.n143 = or disjoint i64 %grp.n132, 32
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518
  %21 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local502, i64 0
  %22 = insertelement <4 x ptr addrspace(3)> %21, ptr addrspace(3) %address.local519, i64 1
  %23 = insertelement <4 x ptr addrspace(3)> %22, ptr addrspace(3) %address.local533, i64 2
  %24 = insertelement <4 x ptr addrspace(3)> %23, ptr addrspace(3) %address.local537, i64 3
  %25 = ptrtoint <4 x ptr addrspace(3)> %24 to <4 x i64>
  %26 = trunc <4 x i64> %25 to <4 x i32>
  %27 = extractelement <4 x i32> %26, i64 0
  %grp.n133 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %27)
  %dma.buffer.lds1374 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n133)
  %kdma.source11 = or disjoint i64 %grp.n132, %kdma.source10
  %kdma.source12 = trunc nuw nsw i64 %kdma.source11 to i32
  %kdma.source13 = shl nuw nsw i32 %kdma.source12, 1
  %kdma.offset = or disjoint i32 %kdma.source13, %kdma.intra
  %28 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1373) #9
  %kdma.source25 = or disjoint i64 %grp.n132, %kdma.source24
  %kdma.source26 = trunc nuw nsw i64 %kdma.source25 to i32
  %kdma.source27 = shl nuw nsw i32 %kdma.source26, 1
  %kdma.offset28 = or disjoint i32 %kdma.source27, %kdma.intra.hi
  %kdma.lds29 = add i32 %dma.buffer.lds1374, 512
  %29 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds29, i32 %kdma.offset28, <4 x i32> %dma.buffer.rsrc1373) #9
  %30 = extractelement <4 x i32> %26, i64 1
  %grp.n138 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %30)
  %dma.buffer.lds1371 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n138)
  %kdma.source42 = or disjoint i64 %grp.n132, %kdma.source41
  %kdma.source43 = trunc nuw nsw i64 %kdma.source42 to i32
  %kdma.source44 = shl nuw nsw i32 %kdma.source43, 1
  %kdma.offset45 = or disjoint i32 %kdma.source44, %kdma.intra
  %31 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset45, <4 x i32> %dma.buffer.rsrc1373) #9
  %kdma.source59 = or disjoint i64 %grp.n132, %kdma.source58
  %kdma.source60 = trunc nuw nsw i64 %kdma.source59 to i32
  %kdma.source61 = shl nuw nsw i32 %kdma.source60, 1
  %kdma.offset62 = or disjoint i32 %kdma.source61, %kdma.intra.hi
  %kdma.lds63 = add i32 %dma.buffer.lds1371, 512
  %32 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds63, i32 %kdma.offset62, <4 x i32> %dma.buffer.rsrc1373) #9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %33 = extractelement <4 x i32> %26, i64 2
  %grp.n144 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %33)
  %dma.buffer.lds1368 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n144)
  %kdma.source75 = or disjoint i64 %grp.n143, %kdma.source10
  %kdma.source76 = trunc nuw nsw i64 %kdma.source75 to i32
  %kdma.source77 = shl nuw nsw i32 %kdma.source76, 1
  %kdma.offset78 = or disjoint i32 %kdma.source77, %kdma.intra
  %34 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset78, <4 x i32> %dma.buffer.rsrc1373) #9
  %kdma.source91 = or disjoint i64 %grp.n143, %kdma.source24
  %kdma.source92 = trunc nuw nsw i64 %kdma.source91 to i32
  %kdma.source93 = shl nuw nsw i32 %kdma.source92, 1
  %kdma.offset94 = or disjoint i32 %kdma.source93, %kdma.intra.hi
  %kdma.lds95 = add i32 %dma.buffer.lds1368, 512
  %35 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds95, i32 %kdma.offset94, <4 x i32> %dma.buffer.rsrc1373) #9
  %36 = extractelement <4 x i32> %26, i64 3
  %grp.n149 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %36)
  %dma.buffer.lds1365 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n149)
  %kdma.source108 = or disjoint i64 %grp.n143, %kdma.source41
  %kdma.source109 = trunc nuw nsw i64 %kdma.source108 to i32
  %kdma.source110 = shl nuw nsw i32 %kdma.source109, 1
  %kdma.offset111 = or disjoint i32 %kdma.source110, %kdma.intra
  %37 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset111, <4 x i32> %dma.buffer.rsrc1373) #9
  %kdma.source125 = or disjoint i64 %grp.n143, %kdma.source58
  %kdma.source126 = trunc nuw nsw i64 %kdma.source125 to i32
  %kdma.source127 = shl nuw nsw i32 %kdma.source126, 1
  %kdma.offset128 = or disjoint i32 %kdma.source127, %kdma.intra.hi
  %kdma.lds129 = add i32 %dma.buffer.lds1365, 512
  %38 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds129, i32 %kdma.offset128, <4 x i32> %dma.buffer.rsrc1373) #9
  %address.local545 = shl nuw nsw i64 %address.local499, 1
  %address.local546 = and i64 %address.local545, 24
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %address.local550 = and i64 %address.local549, -64
  %address.local553 = and i64 %address.local508, 12
  %address.local555 = and i64 %address.local499, 63
  %address.local556 = lshr i64 %address.local555, 5
  %address.local557 = mul nuw nsw i64 %address.local556, 1016
  %.idx = mul nuw nsw i64 %address.local499, 320
  %grp.n154 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx
  %.idx1410 = mul nsw i64 %address.local508, -1280
  %grp.n155 = getelementptr i8, ptr addrspace(3) %grp.n154, i64 %.idx1410
  %grp.n156 = getelementptr [2 x i8], ptr addrspace(3) %grp.n155, i64 %address.local546
  %grp.n157 = getelementptr [2 x i8], ptr addrspace(3) %grp.n156, i64 %address.local550
  %grp.n158 = getelementptr [2 x i8], ptr addrspace(3) %grp.n157, i64 %address.local553
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %grp.n158, i64 %address.local557
  %39 = add nsw i64 %.idx1410, %.idx
  %grp.n156.idx = shl nuw nsw i64 %address.local546, 1
  %40 = or disjoint i64 %39, %grp.n156.idx
  %41 = or disjoint i64 %address.local550, %address.local553
  %42 = shl nsw i64 %41, 1
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032
  %43 = add nsw i64 %40, %address.local559.idx
  %44 = add nsw i64 %43, %42
  %45 = lshr i64 %44, 6
  %46 = and i64 %45, 8
  %47 = xor i64 %46, %44
  %48 = sub i64 %47, %44
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %48
  %grp.n159 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8
  %grp.n160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n159, <4 x float> zeroinitializer) #8
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096
  %grp.n161 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8
  %address.local570 = shl nuw nsw i32 %address.thread498, 1
  %address.local571 = and i32 %address.local570, 24
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local574 = or disjoint i32 %address.local571, %address.local573
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %address.local577 = or disjoint i32 %address.local574, %address.local576
  %address.local578 = zext nneg i32 %address.local577 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %grp.n118) #8, !frisk.register_lifetimes !3
  %address.local579 = lshr i32 %address.thread498, 1
  %address.local580 = and i32 %address.local579, 1
  %address.local581 = zext nneg i32 %address.local580 to i64
  %address.local582 = add nsw i64 %address.base, %address.local581
  %address.local583 = shl i64 %address.local582, 6
  %address.local584 = sub i64 %address.local578, %address.local583
  %grp.n162 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584
  %grp.n163 = getelementptr [2 x i8], ptr addrspace(3) %grp.n162, i64 %address.local553
  %grp.n164 = getelementptr [2 x i8], ptr addrspace(3) %grp.n163, i64 %address.local557
  %address.local588 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 64
  %grp.n162.idx1 = add i64 %address.local584, %address.local553
  %49 = shl i64 %grp.n162.idx1, 1
  %50 = add nuw nsw i64 %address.local559.idx, 64
  %51 = add i64 %50, %49
  %52 = lshr i64 %51, 6
  %53 = and i64 %52, 8
  %54 = xor i64 %53, %51
  %55 = sub i64 %54, %51
  %kdma.swizzled262 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %55
  %grp.n165 = load <4 x half>, ptr addrspace(3) %kdma.swizzled262, align 8
  %kdma.gep263 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 128
  %grp.n173 = load <4 x half>, ptr addrspace(3) %kdma.gep263, align 8
  %kdma.gep264 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 4096
  %grp.n168 = load <4 x half>, ptr addrspace(3) %kdma.gep264, align 8
  %kdma.gep265 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 4224
  %grp.n176 = load <4 x half>, ptr addrspace(3) %kdma.gep265, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n161, <4 x float> %grp.n160) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %kdma.gep266 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128
  %grp.n169 = load <4 x half>, ptr addrspace(3) %kdma.gep266, align 8
  %kdma.gep267 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024
  %grp.n177 = load <4 x half>, ptr addrspace(3) %kdma.gep267, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n165, <4 x float> zeroinitializer) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %kdma.gep268 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224
  %grp.n172 = load <4 x half>, ptr addrspace(3) %kdma.gep268, align 8
  %kdma.gep269 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152
  %grp.n185 = load <4 x half>, ptr addrspace(3) %kdma.gep269, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n168, <4 x float> %grp.n167) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n169, <4 x float> zeroinitializer) #8
  %grp.n174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n172, <4 x float> %grp.n171) #8
  %grp.n175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n173, <4 x float> zeroinitializer) #8
  %kdma.gep270 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120
  %grp.n180 = load <4 x half>, ptr addrspace(3) %kdma.gep270, align 8
  %kdma.gep271 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248
  %grp.n188 = load <4 x half>, ptr addrspace(3) %kdma.gep271, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n176, <4 x float> %grp.n175) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %kdma.gep272 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 1024
  %grp.n181 = load <4 x half>, ptr addrspace(3) %kdma.gep272, align 8
  %kdma.gep273 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 1152
  %grp.n189 = load <4 x half>, ptr addrspace(3) %kdma.gep273, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n177, <4 x float> zeroinitializer) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %kdma.gep274 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 5120
  %grp.n184 = load <4 x half>, ptr addrspace(3) %kdma.gep274, align 8
  %kdma.gep275 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 5248
  %grp.n192 = load <4 x half>, ptr addrspace(3) %kdma.gep275, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n180, <4 x float> %grp.n179) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n181, <4 x float> zeroinitializer) #8
  %grp.n186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n184, <4 x float> %grp.n183) #8
  %grp.n187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n185, <4 x float> zeroinitializer) #8
  %grp.n190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n188, <4 x float> %grp.n187) #8
  %grp.n191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n189, <4 x float> zeroinitializer) #8
  %grp.n194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n192, <4 x float> %grp.n191) #8
  %grp.n195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n159, <4 x float> zeroinitializer) #8
  %grp.n198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n161, <4 x float> %grp.n195) #8
  %grp.n199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n165, <4 x float> zeroinitializer) #8
  %grp.n202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n168, <4 x float> %grp.n199) #8
  %grp.n203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n169, <4 x float> zeroinitializer) #8
  %grp.n206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n172, <4 x float> %grp.n203) #8
  %grp.n207 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n173, <4 x float> zeroinitializer) #8
  %grp.n210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n176, <4 x float> %grp.n207) #8
  %grp.n211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n177, <4 x float> zeroinitializer) #8
  %grp.n214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n180, <4 x float> %grp.n211) #8
  %grp.n215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n181, <4 x float> zeroinitializer) #8
  %grp.n218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n184, <4 x float> %grp.n215) #8
  %grp.n219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n185, <4 x float> zeroinitializer) #8
  %grp.n222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n188, <4 x float> %grp.n219) #8
  %grp.n223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n189, <4 x float> zeroinitializer) #8
  %grp.n225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n192, <4 x float> %grp.n223) #8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %grp.n226 = or disjoint i64 %grp.n132, 64
  %kdma.source141 = or disjoint i64 %grp.n226, %kdma.source10
  %kdma.source142 = trunc nuw nsw i64 %kdma.source141 to i32
  %kdma.source143 = shl nuw nsw i32 %kdma.source142, 1
  %kdma.offset144 = or disjoint i32 %kdma.source143, %kdma.intra
  %56 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset144, <4 x i32> %dma.buffer.rsrc1373) #9
  %kdma.source157 = or disjoint i64 %grp.n226, %kdma.source24
  %kdma.source158 = trunc nuw nsw i64 %kdma.source157 to i32
  %kdma.source159 = shl nuw nsw i32 %kdma.source158, 1
  %kdma.offset160 = or disjoint i32 %kdma.source159, %kdma.intra.hi
  %57 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds29, i32 %kdma.offset160, <4 x i32> %dma.buffer.rsrc1373) #9
  %kdma.source174 = or disjoint i64 %grp.n226, %kdma.source41
  %kdma.source175 = trunc nuw nsw i64 %kdma.source174 to i32
  %kdma.source176 = shl nuw nsw i32 %kdma.source175, 1
  %kdma.offset177 = or disjoint i32 %kdma.source176, %kdma.intra
  %58 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset177, <4 x i32> %dma.buffer.rsrc1373) #9
  %kdma.source191 = or disjoint i64 %grp.n226, %kdma.source58
  %kdma.source192 = trunc nuw nsw i64 %kdma.source191 to i32
  %kdma.source193 = shl nuw nsw i32 %kdma.source192, 1
  %kdma.offset194 = or disjoint i32 %kdma.source193, %kdma.intra.hi
  %59 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds63, i32 %kdma.offset194, <4 x i32> %dma.buffer.rsrc1373) #9
  %kdma.gep276 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192
  %grp.n237 = load <4 x half>, ptr addrspace(3) %kdma.gep276, align 8
  %kdma.gep277 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320
  %grp.n244 = load <4 x half>, ptr addrspace(3) %kdma.gep277, align 8
  %kdma.gep278 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288
  %grp.n239 = load <4 x half>, ptr addrspace(3) %kdma.gep278, align 8
  %kdma.gep279 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416
  %grp.n247 = load <4 x half>, ptr addrspace(3) %kdma.gep279, align 8
  %kdma.gep280 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 8192
  %grp.n240 = load <4 x half>, ptr addrspace(3) %kdma.gep280, align 8
  %kdma.gep281 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 8320
  %grp.n248 = load <4 x half>, ptr addrspace(3) %kdma.gep281, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n237, <4 x float> %grp.n166) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %kdma.gep282 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 12288
  %grp.n243 = load <4 x half>, ptr addrspace(3) %kdma.gep282, align 8
  %kdma.gep283 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 12416
  %grp.n251 = load <4 x half>, ptr addrspace(3) %kdma.gep283, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n239, <4 x float> %grp.n238) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n240, <4 x float> %grp.n170) #8
  %grp.n245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n243, <4 x float> %grp.n242) #8
  %grp.n246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n244, <4 x float> %grp.n174) #8
  %grp.n249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n247, <4 x float> %grp.n246) #8
  %kdma.gep284 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216
  %grp.n252 = load <4 x half>, ptr addrspace(3) %kdma.gep284, align 8
  %kdma.gep285 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344
  %grp.n260 = load <4 x half>, ptr addrspace(3) %kdma.gep285, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n248, <4 x float> %grp.n178) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %kdma.gep286 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312
  %grp.n255 = load <4 x half>, ptr addrspace(3) %kdma.gep286, align 8
  %kdma.gep287 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440
  %grp.n263 = load <4 x half>, ptr addrspace(3) %kdma.gep287, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n251, <4 x float> %grp.n250) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %kdma.gep288 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 9216
  %grp.n256 = load <4 x half>, ptr addrspace(3) %kdma.gep288, align 8
  %kdma.gep289 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 9344
  %grp.n264 = load <4 x half>, ptr addrspace(3) %kdma.gep289, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n252, <4 x float> %grp.n182) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %kdma.gep290 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 13312
  %grp.n259 = load <4 x half>, ptr addrspace(3) %kdma.gep290, align 8
  %kdma.gep291 = getelementptr i8, ptr addrspace(3) %kdma.swizzled262, i64 13440
  %grp.n267 = load <4 x half>, ptr addrspace(3) %kdma.gep291, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n255, <4 x float> %grp.n254) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n256, <4 x float> %grp.n186) #8
  %grp.n261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n259, <4 x float> %grp.n258) #8
  %grp.n262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n260, <4 x float> %grp.n190) #8
  %grp.n265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n263, <4 x float> %grp.n262) #8
  %grp.n266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n264, <4 x float> %grp.n194) #8
  %grp.n269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n267, <4 x float> %grp.n266) #8
  %grp.n270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n237, <4 x float> %grp.n198) #8
  %grp.n273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n239, <4 x float> %grp.n270) #8
  %grp.n274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n240, <4 x float> %grp.n202) #8
  %grp.n277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n243, <4 x float> %grp.n274) #8
  %grp.n278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n244, <4 x float> %grp.n206) #8
  %grp.n281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n247, <4 x float> %grp.n278) #8
  %grp.n282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n248, <4 x float> %grp.n210) #8
  %grp.n285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n251, <4 x float> %grp.n282) #8
  %grp.n286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n252, <4 x float> %grp.n214) #8
  %grp.n289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n255, <4 x float> %grp.n286) #8
  %grp.n290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n256, <4 x float> %grp.n218) #8
  %grp.n293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n259, <4 x float> %grp.n290) #8
  %grp.n294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n260, <4 x float> %grp.n222) #8
  %grp.n297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n263, <4 x float> %grp.n294) #8
  %grp.n298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n264, <4 x float> %grp.n225) #8
  %grp.n300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n267, <4 x float> %grp.n298) #8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %grp.n301 = or disjoint i64 %grp.n132, 96
  %kdma.source207 = or disjoint i64 %grp.n301, %kdma.source10
  %kdma.source208 = trunc nuw nsw i64 %kdma.source207 to i32
  %kdma.source209 = shl nuw nsw i32 %kdma.source208, 1
  %kdma.offset210 = or disjoint i32 %kdma.source209, %kdma.intra
  %60 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset210, <4 x i32> %dma.buffer.rsrc1373) #9
  %kdma.source223 = or disjoint i64 %grp.n301, %kdma.source24
  %kdma.source224 = trunc nuw nsw i64 %kdma.source223 to i32
  %kdma.source225 = shl nuw nsw i32 %kdma.source224, 1
  %kdma.offset226 = or disjoint i32 %kdma.source225, %kdma.intra.hi
  %61 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds95, i32 %kdma.offset226, <4 x i32> %dma.buffer.rsrc1373) #9
  %kdma.source240 = or disjoint i64 %grp.n301, %kdma.source41
  %kdma.source241 = trunc nuw nsw i64 %kdma.source240 to i32
  %kdma.source242 = shl nuw nsw i32 %kdma.source241, 1
  %kdma.offset243 = or disjoint i32 %kdma.source242, %kdma.intra
  %62 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset243, <4 x i32> %dma.buffer.rsrc1373) #9
  %kdma.source257 = or disjoint i64 %grp.n301, %kdma.source58
  %kdma.source258 = trunc nuw nsw i64 %kdma.source257 to i32
  %kdma.source259 = shl nuw nsw i32 %kdma.source258, 1
  %kdma.offset260 = or disjoint i32 %kdma.source259, %kdma.intra.hi
  %63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds129, i32 %kdma.offset260, <4 x i32> %dma.buffer.rsrc1373) #9
  %grp.n312 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8
  %grp.n319 = load <4 x half>, ptr addrspace(3) %kdma.gep266, align 8
  %grp.n314 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8
  %grp.n322 = load <4 x half>, ptr addrspace(3) %kdma.gep268, align 8
  %grp.n315 = load <4 x half>, ptr addrspace(3) %kdma.swizzled262, align 8
  %grp.n323 = load <4 x half>, ptr addrspace(3) %kdma.gep263, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n312, <4 x float> %grp.n241) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n318 = load <4 x half>, ptr addrspace(3) %kdma.gep264, align 8
  %grp.n326 = load <4 x half>, ptr addrspace(3) %kdma.gep265, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n314, <4 x float> %grp.n313) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n315, <4 x float> %grp.n245) #8
  %grp.n320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n318, <4 x float> %grp.n317) #8
  %grp.n321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n319, <4 x float> %grp.n249) #8
  %grp.n324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n322, <4 x float> %grp.n321) #8
  %grp.n327 = load <4 x half>, ptr addrspace(3) %kdma.gep267, align 8
  %grp.n335 = load <4 x half>, ptr addrspace(3) %kdma.gep269, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n323, <4 x float> %grp.n253) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n330 = load <4 x half>, ptr addrspace(3) %kdma.gep270, align 8
  %grp.n338 = load <4 x half>, ptr addrspace(3) %kdma.gep271, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n326, <4 x float> %grp.n325) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n331 = load <4 x half>, ptr addrspace(3) %kdma.gep272, align 8
  %grp.n339 = load <4 x half>, ptr addrspace(3) %kdma.gep273, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n327, <4 x float> %grp.n257) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n334 = load <4 x half>, ptr addrspace(3) %kdma.gep274, align 8
  %grp.n342 = load <4 x half>, ptr addrspace(3) %kdma.gep275, align 8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n330, <4 x float> %grp.n329) #8
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n331, <4 x float> %grp.n261) #8
  %grp.n336 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n334, <4 x float> %grp.n333) #8
  %grp.n337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n335, <4 x float> %grp.n265) #8
  %grp.n340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n338, <4 x float> %grp.n337) #8
  %grp.n341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n339, <4 x float> %grp.n269) #8
  %grp.n344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n342, <4 x float> %grp.n341) #8
  %grp.n345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n312, <4 x float> %grp.n273) #8
  %grp.n348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n314, <4 x float> %grp.n345) #8
  %grp.n349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n315, <4 x float> %grp.n277) #8
  %grp.n352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n318, <4 x float> %grp.n349) #8
  %grp.n353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n319, <4 x float> %grp.n281) #8
  %grp.n356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n322, <4 x float> %grp.n353) #8
  %grp.n357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n323, <4 x float> %grp.n285) #8
  %grp.n360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n326, <4 x float> %grp.n357) #8
  %grp.n361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n327, <4 x float> %grp.n289) #8
  %grp.n364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n330, <4 x float> %grp.n361) #8
  %grp.n365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n331, <4 x float> %grp.n293) #8
  %grp.n368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n334, <4 x float> %grp.n365) #8
  %grp.n369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n335, <4 x float> %grp.n297) #8
  %grp.n372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n338, <4 x float> %grp.n369) #8
  %grp.n373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n339, <4 x float> %grp.n300) #8
  %grp.n375 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n342, <4 x float> %grp.n373) #8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %address.local917 = shl nuw nsw i64 %address.local499, 5
  %address.local918 = and i64 %address.local917, 8064
  %address.local920 = and i64 %address.local500, 24
  %address.local921 = or disjoint i64 %address.local918, %address.local920
  %.idx14191420 = or disjoint i64 %address.local921, %grp.n132
  %.idx14191420.tr = trunc nuw nsw i64 %.idx14191420 to i32
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1
  %grp.n380 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #9
  %address.local923 = shl nuw nsw i64 %address.local516, 5
  %address.local924 = and i64 %address.local923, 16256
  %address.local927 = or disjoint i64 %address.local924, %address.local920
  %.idx14211422 = or disjoint i64 %address.local927, %grp.n132
  %.idx14211422.tr = trunc nuw nsw i64 %.idx14211422 to i32
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1
  %grp.n385 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #9
  %grp.n386 = load <4 x half>, ptr addrspace(3) %kdma.gep276, align 8
  %grp.n387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n386, <4 x float> %grp.n316) #8
  %grp.n388 = load <4 x half>, ptr addrspace(3) %kdma.gep278, align 8
  %grp.n389 = load <4 x half>, ptr addrspace(3) %kdma.gep280, align 8
  %grp.n390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n388, <4 x float> %grp.n387) #8
  %grp.n391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n389, <4 x float> %grp.n320) #8
  %grp.n392 = extractelement <4 x float> %grp.n390, i64 0
  %grp.n393 = tail call float @llvm.maxnum.f32(float %grp.n392, float -inf), !frisk.local_reduction !3
  %grp.n394 = extractelement <4 x float> %grp.n390, i64 1
  %grp.n395 = extractelement <4 x float> %grp.n390, i64 2
  %grp.n396 = extractelement <4 x float> %grp.n390, i64 3
  %grp.n397 = load <4 x half>, ptr addrspace(3) %kdma.gep282, align 8
  %grp.n398 = load <4 x half>, ptr addrspace(3) %kdma.gep277, align 8
  %grp.n399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n397, <4 x float> %grp.n391) #8
  %grp.n400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n398, <4 x float> %grp.n324) #8
  %grp.n401 = extractelement <4 x float> %grp.n399, i64 0
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %grp.n393, float %grp.n401), !frisk.local_reduction !3
  %grp.n402 = extractelement <4 x float> %grp.n399, i64 1
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %grp.n394, float %grp.n402), !frisk.local_reduction !3
  %grp.n403 = extractelement <4 x float> %grp.n399, i64 2
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %grp.n395, float %grp.n403), !frisk.local_reduction !3
  %grp.n404 = extractelement <4 x float> %grp.n399, i64 3
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %grp.n396, float %grp.n404), !frisk.local_reduction !3
  %grp.n405 = load <4 x half>, ptr addrspace(3) %kdma.gep279, align 8
  %grp.n406 = load <4 x half>, ptr addrspace(3) %kdma.gep281, align 8
  %grp.n407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n405, <4 x float> %grp.n400) #8
  %grp.n408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n406, <4 x float> %grp.n328) #8
  %grp.n409 = extractelement <4 x float> %grp.n407, i64 0
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %grp.n409), !frisk.local_reduction !3
  %grp.n410 = extractelement <4 x float> %grp.n407, i64 1
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %grp.n410), !frisk.local_reduction !3
  %grp.n411 = extractelement <4 x float> %grp.n407, i64 2
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %grp.n411), !frisk.local_reduction !3
  %grp.n412 = extractelement <4 x float> %grp.n407, i64 3
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %grp.n412), !frisk.local_reduction !3
  %grp.n413 = load <4 x half>, ptr addrspace(3) %kdma.gep283, align 8
  %grp.n414 = load <4 x half>, ptr addrspace(3) %kdma.gep284, align 8
  %grp.n415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n413, <4 x float> %grp.n408) #8
  %grp.n416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n414, <4 x float> %grp.n332) #8
  %grp.n417 = extractelement <4 x float> %grp.n415, i64 0
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %grp.n417), !frisk.local_reduction !3
  %grp.n418 = extractelement <4 x float> %grp.n415, i64 1
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %grp.n418), !frisk.local_reduction !3
  %grp.n419 = extractelement <4 x float> %grp.n415, i64 2
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %grp.n419), !frisk.local_reduction !3
  %grp.n420 = extractelement <4 x float> %grp.n415, i64 3
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %grp.n420), !frisk.local_reduction !3
  %grp.n421 = load <4 x half>, ptr addrspace(3) %kdma.gep286, align 8
  %grp.n422 = load <4 x half>, ptr addrspace(3) %kdma.gep288, align 8
  %grp.n423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n421, <4 x float> %grp.n416) #8
  %grp.n424 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n422, <4 x float> %grp.n336) #8
  %grp.n425 = extractelement <4 x float> %grp.n423, i64 0
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %grp.n425), !frisk.local_reduction !3
  %grp.n426 = extractelement <4 x float> %grp.n423, i64 1
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %grp.n426), !frisk.local_reduction !3
  %grp.n427 = extractelement <4 x float> %grp.n423, i64 2
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %grp.n427), !frisk.local_reduction !3
  %grp.n428 = extractelement <4 x float> %grp.n423, i64 3
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %grp.n428), !frisk.local_reduction !3
  %grp.n429 = load <4 x half>, ptr addrspace(3) %kdma.gep290, align 8
  %grp.n430 = load <4 x half>, ptr addrspace(3) %kdma.gep285, align 8
  %grp.n431 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n429, <4 x float> %grp.n424) #8
  %grp.n432 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n430, <4 x float> %grp.n340) #8
  %grp.n433 = extractelement <4 x float> %grp.n431, i64 0
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %grp.n433), !frisk.local_reduction !3
  %grp.n434 = extractelement <4 x float> %grp.n431, i64 1
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %grp.n434), !frisk.local_reduction !3
  %grp.n435 = extractelement <4 x float> %grp.n431, i64 2
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %grp.n435), !frisk.local_reduction !3
  %grp.n436 = extractelement <4 x float> %grp.n431, i64 3
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %grp.n436), !frisk.local_reduction !3
  %grp.n437 = load <4 x half>, ptr addrspace(3) %kdma.gep287, align 8
  %grp.n438 = load <4 x half>, ptr addrspace(3) %kdma.gep289, align 8
  %grp.n439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n437, <4 x float> %grp.n432) #8
  %grp.n440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n438, <4 x float> %grp.n344) #8
  %grp.n441 = extractelement <4 x float> %grp.n439, i64 0
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %grp.n441), !frisk.local_reduction !3
  %grp.n442 = extractelement <4 x float> %grp.n439, i64 1
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %grp.n442), !frisk.local_reduction !3
  %grp.n443 = extractelement <4 x float> %grp.n439, i64 2
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %grp.n443), !frisk.local_reduction !3
  %grp.n444 = extractelement <4 x float> %grp.n439, i64 3
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %grp.n444), !frisk.local_reduction !3
  %grp.n445 = load <4 x half>, ptr addrspace(3) %kdma.gep291, align 8
  %grp.n447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n445, <4 x float> %grp.n440) #8
  %grp.n448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n386, <4 x float> %grp.n348) #8
  %grp.n449 = extractelement <4 x float> %grp.n447, i64 0
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %grp.n449), !frisk.local_reduction !3
  %grp.n450 = extractelement <4 x float> %grp.n447, i64 1
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %grp.n450), !frisk.local_reduction !3
  %grp.n451 = extractelement <4 x float> %grp.n447, i64 2
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %grp.n451), !frisk.local_reduction !3
  %grp.n452 = extractelement <4 x float> %grp.n447, i64 3
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %grp.n452), !frisk.local_reduction !3
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !frisk.local_reduction !3
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !frisk.local_reduction !3
  %grp.n453 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !frisk.local_reduction !3
  %grp.n456 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n388, <4 x float> %grp.n448) #8
  %grp.n457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n389, <4 x float> %grp.n352) #8
  %grp.n458 = extractelement <4 x float> %grp.n456, i64 0
  %grp.n459 = extractelement <4 x float> %grp.n456, i64 1
  %grp.n460 = extractelement <4 x float> %grp.n456, i64 2
  %grp.n461 = extractelement <4 x float> %grp.n456, i64 3
  %grp.n464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n397, <4 x float> %grp.n457) #8
  %grp.n465 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n398, <4 x float> %grp.n356) #8
  %grp.n466 = extractelement <4 x float> %grp.n464, i64 0
  %grp.n467 = extractelement <4 x float> %grp.n464, i64 1
  %grp.n468 = extractelement <4 x float> %grp.n464, i64 2
  %grp.n469 = extractelement <4 x float> %grp.n464, i64 3
  %grp.n472 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n405, <4 x float> %grp.n465) #8
  %grp.n473 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n406, <4 x float> %grp.n360) #8
  %grp.n474 = extractelement <4 x float> %grp.n472, i64 0
  %grp.n475 = extractelement <4 x float> %grp.n472, i64 1
  %grp.n476 = extractelement <4 x float> %grp.n472, i64 2
  %grp.n477 = extractelement <4 x float> %grp.n472, i64 3
  %grp.n480 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n413, <4 x float> %grp.n473) #8
  %grp.n481 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n414, <4 x float> %grp.n364) #8
  %grp.n482 = extractelement <4 x float> %grp.n480, i64 0
  %grp.n483 = extractelement <4 x float> %grp.n480, i64 1
  %grp.n484 = extractelement <4 x float> %grp.n480, i64 2
  %grp.n485 = extractelement <4 x float> %grp.n480, i64 3
  %grp.n488 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n421, <4 x float> %grp.n481) #8
  %grp.n489 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n422, <4 x float> %grp.n368) #8
  %grp.n490 = extractelement <4 x float> %grp.n488, i64 0
  %grp.n491 = extractelement <4 x float> %grp.n488, i64 1
  %grp.n492 = extractelement <4 x float> %grp.n488, i64 2
  %grp.n493 = extractelement <4 x float> %grp.n488, i64 3
  %grp.n496 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n429, <4 x float> %grp.n489) #8
  %grp.n497 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n430, <4 x float> %grp.n372) #8
  %grp.n498 = extractelement <4 x float> %grp.n496, i64 0
  %grp.n499 = extractelement <4 x float> %grp.n496, i64 1
  %grp.n500 = extractelement <4 x float> %grp.n496, i64 2
  %grp.n501 = extractelement <4 x float> %grp.n496, i64 3
  %grp.n504 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n437, <4 x float> %grp.n497) #8
  %grp.n505 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n438, <4 x float> %grp.n375) #8
  %grp.n506 = extractelement <4 x float> %grp.n504, i64 0
  %grp.n507 = extractelement <4 x float> %grp.n504, i64 1
  %grp.n508 = extractelement <4 x float> %grp.n504, i64 2
  %grp.n509 = extractelement <4 x float> %grp.n504, i64 3
  %grp.n511 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n445, <4 x float> %grp.n505) #8
  %grp.n512 = extractelement <4 x float> %grp.n511, i64 0
  %grp.n513 = extractelement <4 x float> %grp.n511, i64 1
  %grp.n514 = extractelement <4 x float> %grp.n511, i64 2
  %grp.n515 = extractelement <4 x float> %grp.n511, i64 3
  %grp.n516 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %grp.n517 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %grp.n516)
  %grp.n518 = add nuw nsw i32 %grp.n517, 64
  %grp.n519 = xor i32 %grp.n517, 16
  %grp.n520 = and i32 %grp.n518, 192
  %grp.n521 = icmp samesign ult i32 %grp.n519, %grp.n520
  %grp.n522 = select i1 %grp.n521, i32 %grp.n519, i32 %grp.n517
  %grp.n523 = shl nuw nsw i32 %grp.n522, 2
  %grp.n524 = bitcast float %grp.n453 to i32
  %grp.n525 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n524)
  %grp.n526 = bitcast i32 %grp.n525 to float
  %grp.n527 = tail call float @llvm.maxnum.f32(float %grp.n453, float %grp.n526)
  %grp.n528 = xor i32 %grp.n517, 32
  %grp.n529 = icmp samesign ult i32 %grp.n528, %grp.n520
  %grp.n530 = select i1 %grp.n529, i32 %grp.n528, i32 %grp.n517
  %grp.n531 = shl nuw nsw i32 %grp.n530, 2
  %grp.n532 = bitcast float %grp.n527 to i32
  %grp.n533 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n532)
  %grp.n534 = bitcast i32 %grp.n533 to float
  %grp.n535 = tail call float @llvm.maxnum.f32(float %grp.n527, float %grp.n534)
  %grp.n536 = extractvalue [2 x <1 x float>] %grp.n12313, 0
  %grp.n537 = insertelement <1 x float> poison, float %grp.n535, i64 0
  %grp.n538 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %grp.n537, <1 x float> %grp.n536)
  %grp.n539 = tail call float @llvm.maxnum.f32(float %grp.n458, float -inf), !frisk.local_reduction !3
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %grp.n539, float %grp.n466), !frisk.local_reduction !3
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %grp.n459, float %grp.n467), !frisk.local_reduction !3
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %grp.n460, float %grp.n468), !frisk.local_reduction !3
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %grp.n461, float %grp.n469), !frisk.local_reduction !3
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %grp.n474), !frisk.local_reduction !3
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %grp.n475), !frisk.local_reduction !3
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %grp.n476), !frisk.local_reduction !3
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %grp.n477), !frisk.local_reduction !3
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %grp.n482), !frisk.local_reduction !3
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %grp.n483), !frisk.local_reduction !3
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %grp.n484), !frisk.local_reduction !3
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %grp.n485), !frisk.local_reduction !3
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %grp.n490), !frisk.local_reduction !3
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %grp.n491), !frisk.local_reduction !3
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %grp.n492), !frisk.local_reduction !3
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %grp.n493), !frisk.local_reduction !3
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %grp.n498), !frisk.local_reduction !3
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %grp.n499), !frisk.local_reduction !3
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %grp.n500), !frisk.local_reduction !3
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %grp.n501), !frisk.local_reduction !3
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %grp.n506), !frisk.local_reduction !3
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %grp.n507), !frisk.local_reduction !3
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %grp.n508), !frisk.local_reduction !3
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %grp.n509), !frisk.local_reduction !3
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %grp.n512), !frisk.local_reduction !3
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %grp.n513), !frisk.local_reduction !3
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %grp.n514), !frisk.local_reduction !3
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %grp.n515), !frisk.local_reduction !3
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !frisk.local_reduction !3
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !frisk.local_reduction !3
  %grp.n540 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !frisk.local_reduction !3
  %grp.n541 = bitcast float %grp.n540 to i32
  %grp.n542 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n541)
  %grp.n543 = bitcast i32 %grp.n542 to float
  %grp.n544 = tail call float @llvm.maxnum.f32(float %grp.n540, float %grp.n543)
  %grp.n545 = bitcast float %grp.n544 to i32
  %grp.n546 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n545)
  %grp.n547 = bitcast i32 %grp.n546 to float
  %grp.n548 = tail call float @llvm.maxnum.f32(float %grp.n544, float %grp.n547)
  %grp.n549 = extractvalue [2 x <1 x float>] %grp.n12313, 1
  %grp.n550 = insertelement <1 x float> poison, float %grp.n548, i64 0
  %grp.n551 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %grp.n550, <1 x float> %grp.n549)
  %grp.n552 = shufflevector <1 x float> %grp.n538, <1 x float> poison, <4 x i32> zeroinitializer
  %grp.n553 = fsub <4 x float> %grp.n390, %grp.n552
  %grp.n554 = fsub <4 x float> %grp.n399, %grp.n552
  %grp.n555 = fsub <4 x float> %grp.n407, %grp.n552
  %grp.n556 = fsub <4 x float> %grp.n415, %grp.n552
  %grp.n557 = fsub <4 x float> %grp.n423, %grp.n552
  %grp.n558 = fsub <4 x float> %grp.n431, %grp.n552
  %grp.n559 = fsub <4 x float> %grp.n439, %grp.n552
  %grp.n560 = fsub <4 x float> %grp.n447, %grp.n552
  %grp.n561 = shufflevector <1 x float> %grp.n551, <1 x float> poison, <4 x i32> zeroinitializer
  %grp.n562 = fsub <4 x float> %grp.n456, %grp.n561
  %grp.n563 = fsub <4 x float> %grp.n464, %grp.n561
  %grp.n564 = fsub <4 x float> %grp.n472, %grp.n561
  %grp.n565 = fsub <4 x float> %grp.n480, %grp.n561
  %grp.n566 = fsub <4 x float> %grp.n488, %grp.n561
  %grp.n567 = fsub <4 x float> %grp.n496, %grp.n561
  %grp.n568 = fsub <4 x float> %grp.n504, %grp.n561
  %grp.n569 = fsub <4 x float> %grp.n511, %grp.n561
  %grp.n570 = fsub <1 x float> %grp.n536, %grp.n538
  %grp.n571 = extractelement <1 x float> %grp.n570, i64 0
  %grp.n572 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n571)
  %grp.n573 = insertelement <4 x float> poison, float %grp.n572, i64 0
  %grp.n574 = shufflevector <4 x float> %grp.n573, <4 x float> poison, <4 x i32> zeroinitializer
  %grp.n575 = extractvalue [2 x <1 x float>] %grp.n12212, 0
  %grp.n576 = extractelement <1 x float> %grp.n575, i64 0
  %grp.n577 = fmul float %grp.n576, %grp.n572
  %grp.n578 = fsub <1 x float> %grp.n549, %grp.n551
  %grp.n579 = extractelement <1 x float> %grp.n578, i64 0
  %grp.n580 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n579)
  %grp.n581 = insertelement <4 x float> poison, float %grp.n580, i64 0
  %grp.n582 = shufflevector <4 x float> %grp.n581, <4 x float> poison, <4 x i32> zeroinitializer
  %grp.n583 = extractvalue [2 x <1 x float>] %grp.n12212, 1
  %grp.n584 = extractelement <1 x float> %grp.n583, i64 0
  %grp.n585 = fmul float %grp.n584, %grp.n580
  %grp.n586 = extractelement <4 x float> %grp.n553, i64 0
  %grp.n587 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n586)
  %grp.n588 = insertelement <4 x float> poison, float %grp.n587, i64 0
  %grp.n589 = extractelement <4 x float> %grp.n553, i64 1
  %grp.n590 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n589)
  %grp.n591 = insertelement <4 x float> %grp.n588, float %grp.n590, i64 1
  %grp.n592 = extractelement <4 x float> %grp.n553, i64 2
  %grp.n593 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n592)
  %grp.n594 = insertelement <4 x float> %grp.n591, float %grp.n593, i64 2
  %grp.n595 = extractelement <4 x float> %grp.n553, i64 3
  %grp.n596 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n595)
  %grp.n597 = insertelement <4 x float> %grp.n594, float %grp.n596, i64 3
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n597) #8, !frisk.register_lifetimes !3
  %grp.n598 = fptrunc <4 x float> %math.result to <4 x half>
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n598) #8, !frisk.register_lifetimes !3
  %grp.n599 = extractelement <4 x float> %grp.n554, i64 0
  %grp.n600 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n599)
  %grp.n601 = insertelement <4 x float> poison, float %grp.n600, i64 0
  %local.reduce89 = fadd float %grp.n587, %grp.n600, !frisk.local_reduction !3
  %grp.n602 = extractelement <4 x float> %grp.n554, i64 1
  %grp.n603 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n602)
  %grp.n604 = insertelement <4 x float> %grp.n601, float %grp.n603, i64 1
  %local.reduce90 = fadd float %grp.n590, %grp.n603, !frisk.local_reduction !3
  %grp.n605 = extractelement <4 x float> %grp.n554, i64 2
  %grp.n606 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n605)
  %grp.n607 = insertelement <4 x float> %grp.n604, float %grp.n606, i64 2
  %local.reduce91 = fadd float %grp.n593, %grp.n606, !frisk.local_reduction !3
  %grp.n608 = extractelement <4 x float> %grp.n554, i64 3
  %grp.n609 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n608)
  %grp.n610 = insertelement <4 x float> %grp.n607, float %grp.n609, i64 3
  %math.result374 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n610) #8, !frisk.register_lifetimes !3
  %grp.n611 = fptrunc <4 x float> %math.result374 to <4 x half>
  %math.packed375 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n611) #8, !frisk.register_lifetimes !3
  %local.reduce92 = fadd float %grp.n596, %grp.n609, !frisk.local_reduction !3
  %grp.n612 = extractelement <4 x float> %grp.n555, i64 0
  %grp.n613 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n612)
  %grp.n614 = insertelement <4 x float> poison, float %grp.n613, i64 0
  %local.reduce93 = fadd float %grp.n613, %local.reduce89, !frisk.local_reduction !3
  %grp.n615 = extractelement <4 x float> %grp.n555, i64 1
  %grp.n616 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n615)
  %grp.n617 = insertelement <4 x float> %grp.n614, float %grp.n616, i64 1
  %local.reduce94 = fadd float %grp.n616, %local.reduce90, !frisk.local_reduction !3
  %grp.n618 = extractelement <4 x float> %grp.n555, i64 2
  %grp.n619 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n618)
  %grp.n620 = insertelement <4 x float> %grp.n617, float %grp.n619, i64 2
  %local.reduce95 = fadd float %grp.n619, %local.reduce91, !frisk.local_reduction !3
  %grp.n621 = extractelement <4 x float> %grp.n555, i64 3
  %grp.n622 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n621)
  %grp.n623 = insertelement <4 x float> %grp.n620, float %grp.n622, i64 3
  %math.result376 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n623) #8, !frisk.register_lifetimes !3
  %grp.n624 = fptrunc <4 x float> %math.result376 to <4 x half>
  %math.packed377 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n624) #8, !frisk.register_lifetimes !3
  %local.reduce96 = fadd float %grp.n622, %local.reduce92, !frisk.local_reduction !3
  %grp.n625 = extractelement <4 x float> %grp.n556, i64 0
  %grp.n626 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n625)
  %grp.n627 = insertelement <4 x float> poison, float %grp.n626, i64 0
  %local.reduce97 = fadd float %grp.n626, %local.reduce93, !frisk.local_reduction !3
  %grp.n628 = extractelement <4 x float> %grp.n556, i64 1
  %grp.n629 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n628)
  %grp.n630 = insertelement <4 x float> %grp.n627, float %grp.n629, i64 1
  %local.reduce98 = fadd float %grp.n629, %local.reduce94, !frisk.local_reduction !3
  %grp.n631 = extractelement <4 x float> %grp.n556, i64 2
  %grp.n632 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n631)
  %grp.n633 = insertelement <4 x float> %grp.n630, float %grp.n632, i64 2
  %local.reduce99 = fadd float %grp.n632, %local.reduce95, !frisk.local_reduction !3
  %grp.n634 = extractelement <4 x float> %grp.n556, i64 3
  %grp.n635 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n634)
  %grp.n636 = insertelement <4 x float> %grp.n633, float %grp.n635, i64 3
  %math.result378 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n636) #8, !frisk.register_lifetimes !3
  %grp.n637 = fptrunc <4 x float> %math.result378 to <4 x half>
  %math.packed379 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n637) #8, !frisk.register_lifetimes !3
  %local.reduce100 = fadd float %grp.n635, %local.reduce96, !frisk.local_reduction !3
  %grp.n638 = extractelement <4 x float> %grp.n557, i64 0
  %grp.n639 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n638)
  %grp.n640 = insertelement <4 x float> poison, float %grp.n639, i64 0
  %local.reduce101 = fadd float %grp.n639, %local.reduce97, !frisk.local_reduction !3
  %grp.n641 = extractelement <4 x float> %grp.n557, i64 1
  %grp.n642 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n641)
  %grp.n643 = insertelement <4 x float> %grp.n640, float %grp.n642, i64 1
  %local.reduce102 = fadd float %grp.n642, %local.reduce98, !frisk.local_reduction !3
  %grp.n644 = extractelement <4 x float> %grp.n557, i64 2
  %grp.n645 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n644)
  %grp.n646 = insertelement <4 x float> %grp.n643, float %grp.n645, i64 2
  %local.reduce103 = fadd float %grp.n645, %local.reduce99, !frisk.local_reduction !3
  %grp.n647 = extractelement <4 x float> %grp.n557, i64 3
  %grp.n648 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n647)
  %grp.n649 = insertelement <4 x float> %grp.n646, float %grp.n648, i64 3
  %math.result380 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n649) #8, !frisk.register_lifetimes !3
  %grp.n650 = fptrunc <4 x float> %math.result380 to <4 x half>
  %math.packed381 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n650) #8, !frisk.register_lifetimes !3
  %local.reduce104 = fadd float %grp.n648, %local.reduce100, !frisk.local_reduction !3
  %grp.n651 = extractelement <4 x float> %grp.n558, i64 0
  %grp.n652 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n651)
  %grp.n653 = insertelement <4 x float> poison, float %grp.n652, i64 0
  %local.reduce105 = fadd float %grp.n652, %local.reduce101, !frisk.local_reduction !3
  %grp.n654 = extractelement <4 x float> %grp.n558, i64 1
  %grp.n655 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n654)
  %grp.n656 = insertelement <4 x float> %grp.n653, float %grp.n655, i64 1
  %local.reduce106 = fadd float %grp.n655, %local.reduce102, !frisk.local_reduction !3
  %grp.n657 = extractelement <4 x float> %grp.n558, i64 2
  %grp.n658 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n657)
  %grp.n659 = insertelement <4 x float> %grp.n656, float %grp.n658, i64 2
  %local.reduce107 = fadd float %grp.n658, %local.reduce103, !frisk.local_reduction !3
  %grp.n660 = extractelement <4 x float> %grp.n558, i64 3
  %grp.n661 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n660)
  %grp.n662 = insertelement <4 x float> %grp.n659, float %grp.n661, i64 3
  %math.result382 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n662) #8, !frisk.register_lifetimes !3
  %grp.n663 = fptrunc <4 x float> %math.result382 to <4 x half>
  %math.packed383 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n663) #8, !frisk.register_lifetimes !3
  %local.reduce108 = fadd float %grp.n661, %local.reduce104, !frisk.local_reduction !3
  %grp.n664 = extractelement <4 x float> %grp.n559, i64 0
  %grp.n665 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n664)
  %grp.n666 = insertelement <4 x float> poison, float %grp.n665, i64 0
  %local.reduce109 = fadd float %grp.n665, %local.reduce105, !frisk.local_reduction !3
  %grp.n667 = extractelement <4 x float> %grp.n559, i64 1
  %grp.n668 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n667)
  %grp.n669 = insertelement <4 x float> %grp.n666, float %grp.n668, i64 1
  %local.reduce110 = fadd float %grp.n668, %local.reduce106, !frisk.local_reduction !3
  %grp.n670 = extractelement <4 x float> %grp.n559, i64 2
  %grp.n671 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n670)
  %grp.n672 = insertelement <4 x float> %grp.n669, float %grp.n671, i64 2
  %local.reduce111 = fadd float %grp.n671, %local.reduce107, !frisk.local_reduction !3
  %grp.n673 = extractelement <4 x float> %grp.n559, i64 3
  %grp.n674 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n673)
  %grp.n675 = insertelement <4 x float> %grp.n672, float %grp.n674, i64 3
  %math.result384 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n675) #8, !frisk.register_lifetimes !3
  %grp.n676 = fptrunc <4 x float> %math.result384 to <4 x half>
  %math.packed385 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n676) #8, !frisk.register_lifetimes !3
  %local.reduce112 = fadd float %grp.n674, %local.reduce108, !frisk.local_reduction !3
  %grp.n677 = extractelement <4 x float> %grp.n560, i64 0
  %grp.n678 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n677)
  %grp.n679 = insertelement <4 x float> poison, float %grp.n678, i64 0
  %local.reduce113 = fadd float %grp.n678, %local.reduce109, !frisk.local_reduction !3
  %grp.n680 = extractelement <4 x float> %grp.n560, i64 1
  %grp.n681 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n680)
  %grp.n682 = insertelement <4 x float> %grp.n679, float %grp.n681, i64 1
  %local.reduce114 = fadd float %grp.n681, %local.reduce110, !frisk.local_reduction !3
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !frisk.local_reduction !3
  %grp.n683 = extractelement <4 x float> %grp.n560, i64 2
  %grp.n684 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n683)
  %grp.n685 = insertelement <4 x float> %grp.n682, float %grp.n684, i64 2
  %local.reduce115 = fadd float %grp.n684, %local.reduce111, !frisk.local_reduction !3
  %grp.n686 = extractelement <4 x float> %grp.n560, i64 3
  %grp.n687 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n686)
  %grp.n688 = insertelement <4 x float> %grp.n685, float %grp.n687, i64 3
  %math.result386 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n688) #8, !frisk.register_lifetimes !3
  %grp.n689 = fptrunc <4 x float> %math.result386 to <4 x half>
  %math.packed387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n689) #8, !frisk.register_lifetimes !3
  %local.reduce116 = fadd float %grp.n687, %local.reduce112, !frisk.local_reduction !3
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !frisk.local_reduction !3
  %grp.n690 = fadd float %local.reduce117, %local.reduce118, !frisk.local_reduction !3
  %grp.n691 = fadd float %grp.n577, %grp.n690
  %grp.n692 = insertelement <1 x float> poison, float %grp.n691, i64 0
  %grp.n693 = insertvalue [2 x <1 x float>] poison, <1 x float> %grp.n692, 0
  %grp.n694 = extractelement <4 x float> %grp.n562, i64 0
  %grp.n695 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n694)
  %grp.n696 = insertelement <4 x float> poison, float %grp.n695, i64 0
  %grp.n697 = extractelement <4 x float> %grp.n562, i64 1
  %grp.n698 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n697)
  %grp.n699 = insertelement <4 x float> %grp.n696, float %grp.n698, i64 1
  %grp.n700 = extractelement <4 x float> %grp.n562, i64 2
  %grp.n701 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n700)
  %grp.n702 = insertelement <4 x float> %grp.n699, float %grp.n701, i64 2
  %grp.n703 = extractelement <4 x float> %grp.n562, i64 3
  %grp.n704 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n703)
  %grp.n705 = insertelement <4 x float> %grp.n702, float %grp.n704, i64 3
  %math.result388 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n705) #8, !frisk.register_lifetimes !3
  %grp.n706 = fptrunc <4 x float> %math.result388 to <4 x half>
  %math.packed389 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n706) #8, !frisk.register_lifetimes !3
  %grp.n707 = extractelement <4 x float> %grp.n563, i64 0
  %grp.n708 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n707)
  %grp.n709 = insertelement <4 x float> poison, float %grp.n708, i64 0
  %local.reduce = fadd float %grp.n695, %grp.n708, !frisk.local_reduction !3
  %grp.n710 = extractelement <4 x float> %grp.n563, i64 1
  %grp.n711 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n710)
  %grp.n712 = insertelement <4 x float> %grp.n709, float %grp.n711, i64 1
  %local.reduce59 = fadd float %grp.n698, %grp.n711, !frisk.local_reduction !3
  %grp.n713 = extractelement <4 x float> %grp.n563, i64 2
  %grp.n714 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n713)
  %grp.n715 = insertelement <4 x float> %grp.n712, float %grp.n714, i64 2
  %local.reduce60 = fadd float %grp.n701, %grp.n714, !frisk.local_reduction !3
  %grp.n716 = extractelement <4 x float> %grp.n563, i64 3
  %grp.n717 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n716)
  %grp.n718 = insertelement <4 x float> %grp.n715, float %grp.n717, i64 3
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n718) #8, !frisk.register_lifetimes !3
  %grp.n719 = fptrunc <4 x float> %math.result390 to <4 x half>
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n719) #8, !frisk.register_lifetimes !3
  %local.reduce61 = fadd float %grp.n704, %grp.n717, !frisk.local_reduction !3
  %grp.n720 = extractelement <4 x float> %grp.n564, i64 0
  %grp.n721 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n720)
  %grp.n722 = insertelement <4 x float> poison, float %grp.n721, i64 0
  %local.reduce62 = fadd float %grp.n721, %local.reduce, !frisk.local_reduction !3
  %grp.n723 = extractelement <4 x float> %grp.n564, i64 1
  %grp.n724 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n723)
  %grp.n725 = insertelement <4 x float> %grp.n722, float %grp.n724, i64 1
  %local.reduce63 = fadd float %grp.n724, %local.reduce59, !frisk.local_reduction !3
  %grp.n726 = extractelement <4 x float> %grp.n564, i64 2
  %grp.n727 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n726)
  %grp.n728 = insertelement <4 x float> %grp.n725, float %grp.n727, i64 2
  %local.reduce64 = fadd float %grp.n727, %local.reduce60, !frisk.local_reduction !3
  %grp.n729 = extractelement <4 x float> %grp.n564, i64 3
  %grp.n730 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n729)
  %grp.n731 = insertelement <4 x float> %grp.n728, float %grp.n730, i64 3
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n731) #8, !frisk.register_lifetimes !3
  %grp.n732 = fptrunc <4 x float> %math.result392 to <4 x half>
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n732) #8, !frisk.register_lifetimes !3
  %local.reduce65 = fadd float %grp.n730, %local.reduce61, !frisk.local_reduction !3
  %grp.n733 = extractelement <4 x float> %grp.n565, i64 0
  %grp.n734 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n733)
  %grp.n735 = insertelement <4 x float> poison, float %grp.n734, i64 0
  %local.reduce66 = fadd float %grp.n734, %local.reduce62, !frisk.local_reduction !3
  %grp.n736 = extractelement <4 x float> %grp.n565, i64 1
  %grp.n737 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n736)
  %grp.n738 = insertelement <4 x float> %grp.n735, float %grp.n737, i64 1
  %local.reduce67 = fadd float %grp.n737, %local.reduce63, !frisk.local_reduction !3
  %grp.n739 = extractelement <4 x float> %grp.n565, i64 2
  %grp.n740 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n739)
  %grp.n741 = insertelement <4 x float> %grp.n738, float %grp.n740, i64 2
  %local.reduce68 = fadd float %grp.n740, %local.reduce64, !frisk.local_reduction !3
  %grp.n742 = extractelement <4 x float> %grp.n565, i64 3
  %grp.n743 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n742)
  %grp.n744 = insertelement <4 x float> %grp.n741, float %grp.n743, i64 3
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n744) #8, !frisk.register_lifetimes !3
  %grp.n745 = fptrunc <4 x float> %math.result394 to <4 x half>
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n745) #8, !frisk.register_lifetimes !3
  %local.reduce69 = fadd float %grp.n743, %local.reduce65, !frisk.local_reduction !3
  %grp.n746 = extractelement <4 x float> %grp.n566, i64 0
  %grp.n747 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n746)
  %grp.n748 = insertelement <4 x float> poison, float %grp.n747, i64 0
  %local.reduce70 = fadd float %grp.n747, %local.reduce66, !frisk.local_reduction !3
  %grp.n749 = extractelement <4 x float> %grp.n566, i64 1
  %grp.n750 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n749)
  %grp.n751 = insertelement <4 x float> %grp.n748, float %grp.n750, i64 1
  %local.reduce71 = fadd float %grp.n750, %local.reduce67, !frisk.local_reduction !3
  %grp.n752 = extractelement <4 x float> %grp.n566, i64 2
  %grp.n753 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n752)
  %grp.n754 = insertelement <4 x float> %grp.n751, float %grp.n753, i64 2
  %local.reduce72 = fadd float %grp.n753, %local.reduce68, !frisk.local_reduction !3
  %grp.n755 = extractelement <4 x float> %grp.n566, i64 3
  %grp.n756 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n755)
  %grp.n757 = insertelement <4 x float> %grp.n754, float %grp.n756, i64 3
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n757) #8, !frisk.register_lifetimes !3
  %grp.n758 = fptrunc <4 x float> %math.result396 to <4 x half>
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n758) #8, !frisk.register_lifetimes !3
  %local.reduce73 = fadd float %grp.n756, %local.reduce69, !frisk.local_reduction !3
  %grp.n759 = extractelement <4 x float> %grp.n567, i64 0
  %grp.n760 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n759)
  %grp.n761 = insertelement <4 x float> poison, float %grp.n760, i64 0
  %local.reduce74 = fadd float %grp.n760, %local.reduce70, !frisk.local_reduction !3
  %grp.n762 = extractelement <4 x float> %grp.n567, i64 1
  %grp.n763 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n762)
  %grp.n764 = insertelement <4 x float> %grp.n761, float %grp.n763, i64 1
  %local.reduce75 = fadd float %grp.n763, %local.reduce71, !frisk.local_reduction !3
  %grp.n765 = extractelement <4 x float> %grp.n567, i64 2
  %grp.n766 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n765)
  %grp.n767 = insertelement <4 x float> %grp.n764, float %grp.n766, i64 2
  %local.reduce76 = fadd float %grp.n766, %local.reduce72, !frisk.local_reduction !3
  %grp.n768 = extractelement <4 x float> %grp.n567, i64 3
  %grp.n769 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n768)
  %grp.n770 = insertelement <4 x float> %grp.n767, float %grp.n769, i64 3
  %math.result398 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n770) #8, !frisk.register_lifetimes !3
  %grp.n771 = fptrunc <4 x float> %math.result398 to <4 x half>
  %math.packed399 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n771) #8, !frisk.register_lifetimes !3
  %local.reduce77 = fadd float %grp.n769, %local.reduce73, !frisk.local_reduction !3
  %grp.n772 = extractelement <4 x float> %grp.n568, i64 0
  %grp.n773 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n772)
  %grp.n774 = insertelement <4 x float> poison, float %grp.n773, i64 0
  %local.reduce78 = fadd float %grp.n773, %local.reduce74, !frisk.local_reduction !3
  %grp.n775 = extractelement <4 x float> %grp.n568, i64 1
  %grp.n776 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n775)
  %grp.n777 = insertelement <4 x float> %grp.n774, float %grp.n776, i64 1
  %local.reduce79 = fadd float %grp.n776, %local.reduce75, !frisk.local_reduction !3
  %grp.n778 = extractelement <4 x float> %grp.n568, i64 2
  %grp.n779 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n778)
  %grp.n780 = insertelement <4 x float> %grp.n777, float %grp.n779, i64 2
  %local.reduce80 = fadd float %grp.n779, %local.reduce76, !frisk.local_reduction !3
  %grp.n781 = extractelement <4 x float> %grp.n568, i64 3
  %grp.n782 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n781)
  %grp.n783 = insertelement <4 x float> %grp.n780, float %grp.n782, i64 3
  %math.result400 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n783) #8, !frisk.register_lifetimes !3
  %grp.n784 = fptrunc <4 x float> %math.result400 to <4 x half>
  %math.packed401 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n784) #8, !frisk.register_lifetimes !3
  %local.reduce81 = fadd float %grp.n782, %local.reduce77, !frisk.local_reduction !3
  %grp.n785 = extractelement <4 x float> %grp.n569, i64 0
  %grp.n786 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n785)
  %grp.n787 = insertelement <4 x float> poison, float %grp.n786, i64 0
  %local.reduce82 = fadd float %grp.n786, %local.reduce78, !frisk.local_reduction !3
  %grp.n788 = extractelement <4 x float> %grp.n569, i64 1
  %grp.n789 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n788)
  %grp.n790 = insertelement <4 x float> %grp.n787, float %grp.n789, i64 1
  %local.reduce83 = fadd float %grp.n789, %local.reduce79, !frisk.local_reduction !3
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !frisk.local_reduction !3
  %grp.n791 = extractelement <4 x float> %grp.n569, i64 2
  %grp.n792 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n791)
  %grp.n793 = insertelement <4 x float> %grp.n790, float %grp.n792, i64 2
  %local.reduce84 = fadd float %grp.n792, %local.reduce80, !frisk.local_reduction !3
  %grp.n794 = extractelement <4 x float> %grp.n569, i64 3
  %grp.n795 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n794)
  %grp.n796 = insertelement <4 x float> %grp.n793, float %grp.n795, i64 3
  %math.result402 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n796) #8, !frisk.register_lifetimes !3
  %grp.n797 = fptrunc <4 x float> %math.result402 to <4 x half>
  %math.packed403 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n797) #8, !frisk.register_lifetimes !3
  %local.reduce85 = fadd float %grp.n795, %local.reduce81, !frisk.local_reduction !3
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !frisk.local_reduction !3
  %grp.n798 = fadd float %local.reduce86, %local.reduce87, !frisk.local_reduction !3
  %grp.n799 = fadd float %grp.n585, %grp.n798
  %grp.n800 = insertelement <1 x float> poison, float %grp.n799, i64 0
  %grp.n801 = insertvalue [2 x <1 x float>] %grp.n693, <1 x float> %grp.n800, 1
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %.idx14231424 = or disjoint i64 %address.local921, %grp.n143
  %.idx14231424.tr = trunc nuw nsw i64 %.idx14231424 to i32
  %dma.buffer.offset1342 = shl nuw nsw i32 %.idx14231424.tr, 1
  %grp.n822 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1342, <4 x i32> %dma.buffer.rsrc1349) #9
  %.idx14251426 = or disjoint i64 %address.local927, %grp.n143
  %.idx14251426.tr = trunc nuw nsw i64 %.idx14251426 to i32
  %dma.buffer.offset1339 = shl nuw nsw i32 %.idx14251426.tr, 1
  %grp.n827 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1339, <4 x i32> %dma.buffer.rsrc1349) #9
  %address.local929 = shl nuw nsw i32 %address.thread498, 3
  %address.local930 = and i32 %address.local929, 120
  %address.local932 = and i32 %address.local575, 1536
  %address.local933 = or disjoint i32 %address.local930, %address.local932
  %address.local934 = zext nneg i32 %address.local933 to i64
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934
  %grp.n802 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11
  %address.local936 = and i64 %address.local499, 60
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499
  %address.local939.idx = shl nuw nsw i64 %address.local937, 4
  %address.local939 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local939.idx
  %address.local940 = lshr i64 %address.local555, 4
  %address.local942.idx = mul nuw nsw i64 %address.local940, 768
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local942.idx
  %address.local944.idx = mul nsw i64 %address.local508, -64
  %address.local944 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local944.idx
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256
  %grp.n804 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512
  %grp.n806 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768
  %grp.n808 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096
  %grp.n810 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352
  %grp.n812 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608
  %grp.n814 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864
  %grp.n816 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11
  %grp.n803 = bitcast <4 x i32> %grp.n802 to <8 x half>
  %grp.n828 = shufflevector <8 x half> %grp.n803, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n829 = extractvalue [2 x <8 x float>] %grp.n12111, 0
  %grp.n830 = shufflevector <8 x float> %grp.n829, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n831 = fmul <4 x float> %grp.n830, %grp.n574
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n828, <4 x float> %grp.n831) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n847 = shufflevector <8 x half> %grp.n803, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n848 = shufflevector <8 x float> %grp.n829, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n849 = fmul <4 x float> %grp.n848, %grp.n574
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n847, <4 x float> %grp.n849) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n865 = extractvalue [2 x <8 x float>] %grp.n12111, 1
  %grp.n866 = shufflevector <8 x float> %grp.n865, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n867 = fmul <4 x float> %grp.n866, %grp.n582
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n868 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n828, <4 x float> %grp.n867) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n876 = shufflevector <8 x float> %grp.n865, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n877 = fmul <4 x float> %grp.n876, %grp.n582
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n878 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n847, <4 x float> %grp.n877) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n805 = bitcast <4 x i32> %grp.n804 to <8 x half>
  %grp.n833 = shufflevector <8 x half> %grp.n805, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n833, <4 x float> %grp.n832) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n851 = shufflevector <8 x half> %grp.n805, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n851, <4 x float> %grp.n850) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n833, <4 x float> %grp.n868) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n851, <4 x float> %grp.n878) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n807 = bitcast <4 x i32> %grp.n806 to <8 x half>
  %grp.n835 = shufflevector <8 x half> %grp.n807, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n835, <4 x float> %grp.n834) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n853 = shufflevector <8 x half> %grp.n807, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n853, <4 x float> %grp.n852) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n870 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n835, <4 x float> %grp.n869) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n880 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n853, <4 x float> %grp.n879) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n809 = bitcast <4 x i32> %grp.n808 to <8 x half>
  %grp.n837 = shufflevector <8 x half> %grp.n809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n837, <4 x float> %grp.n836) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n855 = shufflevector <8 x half> %grp.n809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n855, <4 x float> %grp.n854) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n837, <4 x float> %grp.n870) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n855, <4 x float> %grp.n880) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n811 = bitcast <4 x i32> %grp.n810 to <8 x half>
  %grp.n839 = shufflevector <8 x half> %grp.n811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n839, <4 x float> %grp.n838) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n857 = shufflevector <8 x half> %grp.n811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n858 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n857, <4 x float> %grp.n856) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n872 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n839, <4 x float> %grp.n871) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n882 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n857, <4 x float> %grp.n881) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n813 = bitcast <4 x i32> %grp.n812 to <8 x half>
  %grp.n841 = shufflevector <8 x half> %grp.n813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n841, <4 x float> %grp.n840) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n859 = shufflevector <8 x half> %grp.n813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n860 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n859, <4 x float> %grp.n858) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n841, <4 x float> %grp.n872) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n859, <4 x float> %grp.n882) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n815 = bitcast <4 x i32> %grp.n814 to <8 x half>
  %grp.n843 = shufflevector <8 x half> %grp.n815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n843, <4 x float> %grp.n842) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n861 = shufflevector <8 x half> %grp.n815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n862 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n861, <4 x float> %grp.n860) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n874 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n843, <4 x float> %grp.n873) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n861, <4 x float> %grp.n883) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n817 = bitcast <4 x i32> %grp.n816 to <8 x half>
  %grp.n845 = shufflevector <8 x half> %grp.n817, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n845, <4 x float> %grp.n844) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n863 = shufflevector <8 x half> %grp.n817, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n864 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n863, <4 x float> %grp.n862) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n845, <4 x float> %grp.n874) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n863, <4 x float> %grp.n884) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %.idx14271428 = or disjoint i64 %address.local921, %grp.n226
  %.idx14271428.tr = trunc nuw nsw i64 %.idx14271428 to i32
  %dma.buffer.offset1336 = shl nuw nsw i32 %.idx14271428.tr, 1
  %grp.n906 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1336, <4 x i32> %dma.buffer.rsrc1349) #9
  %.idx14291430 = or disjoint i64 %address.local927, %grp.n226
  %.idx14291430.tr = trunc nuw nsw i64 %.idx14291430 to i32
  %dma.buffer.offset1333 = shl nuw nsw i32 %.idx14291430.tr, 1
  %grp.n911 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1333, <4 x i32> %dma.buffer.rsrc1349) #9
  %vdb.read.s1.p0 = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192
  %grp.n886 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p0, i32 0) #11
  %vdb.read.s1.p1 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8448
  %grp.n888 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p1, i32 0) #11
  %vdb.read.s1.p2 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8704
  %grp.n890 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p2, i32 0) #11
  %vdb.read.s1.p3 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8960
  %grp.n892 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p3, i32 0) #11
  %vdb.read.s1.p4 = getelementptr i8, ptr addrspace(3) %address.local935, i64 12288
  %grp.n894 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p4, i32 0) #11
  %vdb.read.s1.p5 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12544
  %grp.n896 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p5, i32 0) #11
  %vdb.read.s1.p6 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12800
  %grp.n898 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p6, i32 0) #11
  %vdb.read.s1.p7 = getelementptr i8, ptr addrspace(3) %address.local944, i64 13056
  %grp.n900 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p7, i32 0) #11
  %grp.n887 = bitcast <4 x i32> %grp.n886 to <8 x half>
  %grp.n912 = shufflevector <8 x half> %grp.n887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n913 = extractvalue [2 x <8 x float>] %grp.n12414, 0
  %grp.n914 = shufflevector <8 x float> %grp.n913, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n915 = fmul <4 x float> %grp.n914, %grp.n574
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n912, <4 x float> %grp.n915) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n931 = shufflevector <8 x half> %grp.n887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n932 = shufflevector <8 x float> %grp.n913, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n933 = fmul <4 x float> %grp.n932, %grp.n574
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n934 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n931, <4 x float> %grp.n933) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n949 = extractvalue [2 x <8 x float>] %grp.n12414, 1
  %grp.n950 = shufflevector <8 x float> %grp.n949, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n951 = fmul <4 x float> %grp.n950, %grp.n582
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n912, <4 x float> %grp.n951) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n960 = shufflevector <8 x float> %grp.n949, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n961 = fmul <4 x float> %grp.n960, %grp.n582
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n931, <4 x float> %grp.n961) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n889 = bitcast <4 x i32> %grp.n888 to <8 x half>
  %grp.n917 = shufflevector <8 x half> %grp.n889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n917, <4 x float> %grp.n916) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n935 = shufflevector <8 x half> %grp.n889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n935, <4 x float> %grp.n934) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n953 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n917, <4 x float> %grp.n952) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n935, <4 x float> %grp.n962) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n891 = bitcast <4 x i32> %grp.n890 to <8 x half>
  %grp.n919 = shufflevector <8 x half> %grp.n891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n919, <4 x float> %grp.n918) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n937 = shufflevector <8 x half> %grp.n891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n938 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n937, <4 x float> %grp.n936) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n919, <4 x float> %grp.n953) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n937, <4 x float> %grp.n963) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n893 = bitcast <4 x i32> %grp.n892 to <8 x half>
  %grp.n921 = shufflevector <8 x half> %grp.n893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n921, <4 x float> %grp.n920) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n939 = shufflevector <8 x half> %grp.n893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n940 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n939, <4 x float> %grp.n938) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n921, <4 x float> %grp.n954) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n965 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n939, <4 x float> %grp.n964) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n895 = bitcast <4 x i32> %grp.n894 to <8 x half>
  %grp.n923 = shufflevector <8 x half> %grp.n895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n923, <4 x float> %grp.n922) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n941 = shufflevector <8 x half> %grp.n895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n942 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n941, <4 x float> %grp.n940) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n923, <4 x float> %grp.n955) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n966 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n941, <4 x float> %grp.n965) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n897 = bitcast <4 x i32> %grp.n896 to <8 x half>
  %grp.n925 = shufflevector <8 x half> %grp.n897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n925, <4 x float> %grp.n924) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n943 = shufflevector <8 x half> %grp.n897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n944 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n943, <4 x float> %grp.n942) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n925, <4 x float> %grp.n956) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n943, <4 x float> %grp.n966) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n899 = bitcast <4 x i32> %grp.n898 to <8 x half>
  %grp.n927 = shufflevector <8 x half> %grp.n899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n927, <4 x float> %grp.n926) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n945 = shufflevector <8 x half> %grp.n899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n946 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n945, <4 x float> %grp.n944) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n958 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n927, <4 x float> %grp.n957) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n945, <4 x float> %grp.n967) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n901 = bitcast <4 x i32> %grp.n900 to <8 x half>
  %grp.n929 = shufflevector <8 x half> %grp.n901, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n929, <4 x float> %grp.n928) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n947 = shufflevector <8 x half> %grp.n901, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n948 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n947, <4 x float> %grp.n946) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n929, <4 x float> %grp.n958) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n947, <4 x float> %grp.n968) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %.idx14311432 = or disjoint i64 %address.local921, %grp.n301
  %.idx14311432.tr = trunc nuw nsw i64 %.idx14311432 to i32
  %dma.buffer.offset1330 = shl nuw nsw i32 %.idx14311432.tr, 1
  %grp.n990 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1330, <4 x i32> %dma.buffer.rsrc1349) #9
  %.idx14331434 = or disjoint i64 %address.local927, %grp.n301
  %.idx14331434.tr = trunc nuw nsw i64 %.idx14331434 to i32
  %dma.buffer.offset = shl nuw nsw i32 %.idx14331434.tr, 1
  %grp.n995 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1349) #9
  %grp.n970 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11
  %grp.n972 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11
  %grp.n974 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11
  %grp.n976 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11
  %grp.n978 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11
  %grp.n980 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11
  %grp.n982 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11
  %grp.n984 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11
  %grp.n971 = bitcast <4 x i32> %grp.n970 to <8 x half>
  %grp.n996 = shufflevector <8 x half> %grp.n971, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n997 = extractvalue [2 x <8 x float>] %grp.n12515, 0
  %grp.n998 = shufflevector <8 x float> %grp.n997, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n999 = fmul <4 x float> %grp.n998, %grp.n574
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n996, <4 x float> %grp.n999) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1015 = shufflevector <8 x half> %grp.n971, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n1016 = shufflevector <8 x float> %grp.n997, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n1017 = fmul <4 x float> %grp.n1016, %grp.n574
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1018 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n1015, <4 x float> %grp.n1017) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1033 = extractvalue [2 x <8 x float>] %grp.n12515, 1
  %grp.n1034 = shufflevector <8 x float> %grp.n1033, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n1035 = fmul <4 x float> %grp.n1034, %grp.n582
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1036 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n996, <4 x float> %grp.n1035) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1044 = shufflevector <8 x float> %grp.n1033, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n1045 = fmul <4 x float> %grp.n1044, %grp.n582
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1046 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n1015, <4 x float> %grp.n1045) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n973 = bitcast <4 x i32> %grp.n972 to <8 x half>
  %grp.n1001 = shufflevector <8 x half> %grp.n973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n1001, <4 x float> %grp.n1000) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1019 = shufflevector <8 x half> %grp.n973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n1019, <4 x float> %grp.n1018) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1037 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n1001, <4 x float> %grp.n1036) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1047 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n1019, <4 x float> %grp.n1046) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n975 = bitcast <4 x i32> %grp.n974 to <8 x half>
  %grp.n1003 = shufflevector <8 x half> %grp.n975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n1003, <4 x float> %grp.n1002) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1021 = shufflevector <8 x half> %grp.n975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1022 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n1021, <4 x float> %grp.n1020) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1038 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n1003, <4 x float> %grp.n1037) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1048 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n1021, <4 x float> %grp.n1047) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n977 = bitcast <4 x i32> %grp.n976 to <8 x half>
  %grp.n1005 = shufflevector <8 x half> %grp.n977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n1005, <4 x float> %grp.n1004) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1023 = shufflevector <8 x half> %grp.n977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n1023, <4 x float> %grp.n1022) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1039 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n1005, <4 x float> %grp.n1038) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1049 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n1023, <4 x float> %grp.n1048) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n979 = bitcast <4 x i32> %grp.n978 to <8 x half>
  %grp.n1007 = shufflevector <8 x half> %grp.n979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n1007, <4 x float> %grp.n1006) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1025 = shufflevector <8 x half> %grp.n979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n1025, <4 x float> %grp.n1024) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1040 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n1007, <4 x float> %grp.n1039) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1050 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n1025, <4 x float> %grp.n1049) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n981 = bitcast <4 x i32> %grp.n980 to <8 x half>
  %grp.n1009 = shufflevector <8 x half> %grp.n981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n1009, <4 x float> %grp.n1008) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1027 = shufflevector <8 x half> %grp.n981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1028 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n1027, <4 x float> %grp.n1026) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1041 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n1009, <4 x float> %grp.n1040) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1051 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n1027, <4 x float> %grp.n1050) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n983 = bitcast <4 x i32> %grp.n982 to <8 x half>
  %grp.n1011 = shufflevector <8 x half> %grp.n983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n1011, <4 x float> %grp.n1010) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1029 = shufflevector <8 x half> %grp.n983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n1029, <4 x float> %grp.n1028) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1042 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n1011, <4 x float> %grp.n1041) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1052 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n1029, <4 x float> %grp.n1051) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n985 = bitcast <4 x i32> %grp.n984 to <8 x half>
  %grp.n1013 = shufflevector <8 x half> %grp.n985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n1013, <4 x float> %grp.n1012) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1031 = shufflevector <8 x half> %grp.n985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1032 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n1031, <4 x float> %grp.n1030) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1043 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n1013, <4 x float> %grp.n1042) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1053 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n1031, <4 x float> %grp.n1052) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %grp.n1054 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p0, i32 0) #11
  %grp.n1056 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p1, i32 0) #11
  %grp.n1058 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p2, i32 0) #11
  %grp.n1060 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p3, i32 0) #11
  %grp.n1062 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p4, i32 0) #11
  %grp.n1064 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p5, i32 0) #11
  %grp.n1066 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p6, i32 0) #11
  %grp.n1068 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read.s1.p7, i32 0) #11
  %grp.n1055 = bitcast <4 x i32> %grp.n1054 to <8 x half>
  %grp.n1070 = shufflevector <8 x half> %grp.n1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n1071 = extractvalue [2 x <8 x float>] %grp.n12616, 0
  %grp.n1072 = shufflevector <8 x float> %grp.n1071, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n1073 = fmul <4 x float> %grp.n1072, %grp.n574
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1074 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n1070, <4 x float> %grp.n1073) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1089 = shufflevector <8 x half> %grp.n1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n1090 = shufflevector <8 x float> %grp.n1071, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n1091 = fmul <4 x float> %grp.n1090, %grp.n574
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n1089, <4 x float> %grp.n1091) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1107 = extractvalue [2 x <8 x float>] %grp.n12616, 1
  %grp.n1108 = shufflevector <8 x float> %grp.n1107, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %grp.n1109 = fmul <4 x float> %grp.n1108, %grp.n582
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1110 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n1070, <4 x float> %grp.n1109) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1118 = shufflevector <8 x float> %grp.n1107, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %grp.n1119 = fmul <4 x float> %grp.n1118, %grp.n582
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n1089, <4 x float> %grp.n1119) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1057 = bitcast <4 x i32> %grp.n1056 to <8 x half>
  %grp.n1075 = shufflevector <8 x half> %grp.n1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n1075, <4 x float> %grp.n1074) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1093 = shufflevector <8 x half> %grp.n1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n1093, <4 x float> %grp.n1092) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1111 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n1075, <4 x float> %grp.n1110) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1121 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n1093, <4 x float> %grp.n1120) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1059 = bitcast <4 x i32> %grp.n1058 to <8 x half>
  %grp.n1077 = shufflevector <8 x half> %grp.n1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n1077, <4 x float> %grp.n1076) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1095 = shufflevector <8 x half> %grp.n1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n1095, <4 x float> %grp.n1094) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1112 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n1077, <4 x float> %grp.n1111) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1122 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n1095, <4 x float> %grp.n1121) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1061 = bitcast <4 x i32> %grp.n1060 to <8 x half>
  %grp.n1079 = shufflevector <8 x half> %grp.n1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n1079, <4 x float> %grp.n1078) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1097 = shufflevector <8 x half> %grp.n1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1098 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n1097, <4 x float> %grp.n1096) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n1079, <4 x float> %grp.n1112) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1123 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n1097, <4 x float> %grp.n1122) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1063 = bitcast <4 x i32> %grp.n1062 to <8 x half>
  %grp.n1081 = shufflevector <8 x half> %grp.n1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n1081, <4 x float> %grp.n1080) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1099 = shufflevector <8 x half> %grp.n1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1100 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n1099, <4 x float> %grp.n1098) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1114 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n1081, <4 x float> %grp.n1113) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1124 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n1099, <4 x float> %grp.n1123) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1065 = bitcast <4 x i32> %grp.n1064 to <8 x half>
  %grp.n1083 = shufflevector <8 x half> %grp.n1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n1083, <4 x float> %grp.n1082) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1101 = shufflevector <8 x half> %grp.n1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n1101, <4 x float> %grp.n1100) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1115 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n1083, <4 x float> %grp.n1114) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1125 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n1101, <4 x float> %grp.n1124) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1067 = bitcast <4 x i32> %grp.n1066 to <8 x half>
  %grp.n1085 = shufflevector <8 x half> %grp.n1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n1085, <4 x float> %grp.n1084) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1103 = shufflevector <8 x half> %grp.n1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n1103, <4 x float> %grp.n1102) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n1085, <4 x float> %grp.n1115) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1126 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n1103, <4 x float> %grp.n1125) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1069 = bitcast <4 x i32> %grp.n1068 to <8 x half>
  %grp.n1087 = shufflevector <8 x half> %grp.n1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n1087, <4 x float> %grp.n1086) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1105 = shufflevector <8 x half> %grp.n1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n1105, <4 x float> %grp.n1104) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n1087, <4 x float> %grp.n1116) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1127 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n1105, <4 x float> %grp.n1126) #8, !frisk.pv_interleave !4
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #12
  %grp.n1128 = shufflevector <4 x float> %grp.n846, <4 x float> %grp.n864, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %grp.n1129 = insertvalue [2 x <8 x float>] poison, <8 x float> %grp.n1128, 0
  %grp.n1130 = shufflevector <4 x float> %grp.n875, <4 x float> %grp.n885, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %grp.n1131 = insertvalue [2 x <8 x float>] %grp.n1129, <8 x float> %grp.n1130, 1
  %grp.n1132 = insertvalue [2 x <1 x float>] poison, <1 x float> %grp.n538, 0
  %grp.n1133 = insertvalue [2 x <1 x float>] %grp.n1132, <1 x float> %grp.n551, 1
  %grp.n1134 = shufflevector <4 x float> %grp.n930, <4 x float> %grp.n948, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %grp.n1135 = insertvalue [2 x <8 x float>] poison, <8 x float> %grp.n1134, 0
  %grp.n1136 = shufflevector <4 x float> %grp.n959, <4 x float> %grp.n969, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %grp.n1137 = insertvalue [2 x <8 x float>] %grp.n1135, <8 x float> %grp.n1136, 1
  %grp.n1138 = shufflevector <4 x float> %grp.n1014, <4 x float> %grp.n1032, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %grp.n1139 = insertvalue [2 x <8 x float>] poison, <8 x float> %grp.n1138, 0
  %grp.n1140 = shufflevector <4 x float> %grp.n1043, <4 x float> %grp.n1053, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %grp.n1141 = insertvalue [2 x <8 x float>] %grp.n1139, <8 x float> %grp.n1140, 1
  %grp.n1142 = shufflevector <4 x float> %grp.n1088, <4 x float> %grp.n1106, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %grp.n1143 = insertvalue [2 x <8 x float>] poison, <8 x float> %grp.n1142, 0
  %grp.n1144 = shufflevector <4 x float> %grp.n1117, <4 x float> %grp.n1127, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %grp.n1145 = insertvalue [2 x <8 x float>] %grp.n1143, <8 x float> %grp.n1144, 1
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %grp.n1146 = add nuw nsw i64 %grp.n12010, 1
  %grp.n127 = icmp samesign ult i64 %grp.n12010, 31
  br i1 %grp.n127, label %grp.n128, label %grp.n1147

grp.n1147:                                        ; preds = %grp.n128
  %grp.n1148 = shl nuw nsw i32 %grp.n517, 2
  %grp.n1149 = xor i32 %grp.n1148, 64
  %grp.n1150 = bitcast float %grp.n799 to i32
  %lane.sum.shuffle188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n1149, i32 %grp.n1150)
  %grp.n1151 = bitcast i32 %lane.sum.shuffle188 to float
  %lane.sum.final189 = fadd float %grp.n799, %grp.n1151
  %grp.n1153 = xor i32 %grp.n1148, 128
  %grp.n1154 = bitcast float %lane.sum.final189 to i32
  %lane.sum.shuffle190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n1153, i32 %grp.n1154)
  %grp.n1155 = bitcast i32 %lane.sum.shuffle190 to float
  %lane.sum.final191 = fadd float %lane.sum.final189, %grp.n1155
  %grp.n1156 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0
  %grp.n1159 = bitcast float %grp.n691 to i32
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n1149, i32 %grp.n1159)
  %grp.n1160 = bitcast i32 %lane.sum.shuffle to float
  %lane.sum.final = fadd float %grp.n691, %grp.n1160
  %grp.n1163 = bitcast float %lane.sum.final to i32
  %lane.sum.shuffle182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n1153, i32 %grp.n1163)
  %grp.n1164 = bitcast i32 %lane.sum.shuffle182 to float
  %lane.sum.final183 = fadd float %lane.sum.final, %grp.n1164
  %grp.n1165 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0
  %grp.n1166 = and i32 %kdma.tid, 16
  %.not = icmp eq i32 %grp.n1166, 0
  %grp.n1167 = and i32 %kdma.tid, 32
  %.not36 = icmp eq i32 %grp.n1167, 0
  %grp.n1169 = extractelement <4 x float> %grp.n1088, i64 0
  %grp.n1170 = extractelement <4 x float> %grp.n1088, i64 1
  %grp.n1171 = extractelement <4 x float> %grp.n1088, i64 2
  %grp.n1172 = extractelement <4 x float> %grp.n1088, i64 3
  %grp.n1173 = select i1 %.not, float %grp.n1170, float %grp.n1169
  %grp.n1182 = bitcast float %grp.n1173 to i32
  %grp.n1183 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1182)
  %grp.n1184 = bitcast i32 %grp.n1183 to float
  %grp.n1185 = select i1 %.not, float %grp.n1169, float %grp.n1184
  %grp.n1186 = select i1 %.not, float %grp.n1184, float %grp.n1170
  %grp.n1187 = select i1 %.not, float %grp.n1172, float %grp.n1171
  %grp.n1188 = bitcast float %grp.n1187 to i32
  %grp.n1189 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1188)
  %grp.n1190 = bitcast i32 %grp.n1189 to float
  %grp.n1191 = select i1 %.not, float %grp.n1171, float %grp.n1190
  %grp.n1192 = select i1 %.not, float %grp.n1190, float %grp.n1172
  %grp.n1193 = select i1 %.not36, float %grp.n1191, float %grp.n1185
  %grp.n1198 = bitcast float %grp.n1193 to i32
  %grp.n1199 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1198)
  %grp.n1200 = bitcast i32 %grp.n1199 to float
  %grp.n1203 = select i1 %.not36, float %grp.n1192, float %grp.n1186
  %grp.n1204 = bitcast float %grp.n1203 to i32
  %grp.n1205 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1204)
  %grp.n1206 = bitcast i32 %grp.n1205 to float
  %64 = insertelement <4 x i1> poison, i1 %.not36, i64 0
  %65 = shufflevector <4 x i1> %64, <4 x i1> poison, <4 x i32> zeroinitializer
  %66 = insertelement <4 x float> poison, float %grp.n1185, i64 0
  %67 = insertelement <4 x float> %66, float %grp.n1186, i64 1
  %68 = insertelement <4 x float> %67, float %grp.n1200, i64 2
  %69 = insertelement <4 x float> %68, float %grp.n1206, i64 3
  %70 = insertelement <4 x float> poison, float %grp.n1200, i64 0
  %71 = insertelement <4 x float> %70, float %grp.n1206, i64 1
  %72 = insertelement <4 x float> %71, float %grp.n1191, i64 2
  %73 = insertelement <4 x float> %72, float %grp.n1192, i64 3
  %74 = select <4 x i1> %65, <4 x float> %69, <4 x float> %73
  %grp.n1210 = extractelement <4 x float> %grp.n1014, i64 0
  %grp.n1211 = extractelement <4 x float> %grp.n1014, i64 1
  %grp.n1212 = extractelement <4 x float> %grp.n1014, i64 2
  %grp.n1213 = extractelement <4 x float> %grp.n1014, i64 3
  %grp.n1214 = select i1 %.not, float %grp.n1211, float %grp.n1210
  %grp.n1215 = bitcast float %grp.n1214 to i32
  %grp.n1216 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1215)
  %grp.n1217 = bitcast i32 %grp.n1216 to float
  %grp.n1218 = select i1 %.not, float %grp.n1210, float %grp.n1217
  %grp.n1219 = select i1 %.not, float %grp.n1217, float %grp.n1211
  %grp.n1220 = select i1 %.not, float %grp.n1213, float %grp.n1212
  %grp.n1221 = bitcast float %grp.n1220 to i32
  %grp.n1222 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1221)
  %grp.n1223 = bitcast i32 %grp.n1222 to float
  %grp.n1224 = select i1 %.not, float %grp.n1212, float %grp.n1223
  %grp.n1225 = select i1 %.not, float %grp.n1223, float %grp.n1213
  %grp.n1226 = select i1 %.not36, float %grp.n1224, float %grp.n1218
  %grp.n1227 = bitcast float %grp.n1226 to i32
  %grp.n1228 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1227)
  %grp.n1229 = bitcast i32 %grp.n1228 to float
  %grp.n1232 = select i1 %.not36, float %grp.n1225, float %grp.n1219
  %grp.n1233 = bitcast float %grp.n1232 to i32
  %grp.n1234 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1233)
  %grp.n1235 = bitcast i32 %grp.n1234 to float
  %75 = insertelement <4 x float> poison, float %grp.n1218, i64 0
  %76 = insertelement <4 x float> %75, float %grp.n1219, i64 1
  %77 = insertelement <4 x float> %76, float %grp.n1229, i64 2
  %78 = insertelement <4 x float> %77, float %grp.n1235, i64 3
  %79 = insertelement <4 x float> poison, float %grp.n1229, i64 0
  %80 = insertelement <4 x float> %79, float %grp.n1235, i64 1
  %81 = insertelement <4 x float> %80, float %grp.n1224, i64 2
  %82 = insertelement <4 x float> %81, float %grp.n1225, i64 3
  %83 = select <4 x i1> %65, <4 x float> %78, <4 x float> %82
  %grp.n1239 = extractelement <4 x float> %grp.n930, i64 0
  %grp.n1240 = extractelement <4 x float> %grp.n930, i64 1
  %grp.n1241 = extractelement <4 x float> %grp.n930, i64 2
  %grp.n1242 = extractelement <4 x float> %grp.n930, i64 3
  %grp.n1243 = select i1 %.not, float %grp.n1240, float %grp.n1239
  %grp.n1244 = bitcast float %grp.n1243 to i32
  %grp.n1245 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1244)
  %grp.n1246 = bitcast i32 %grp.n1245 to float
  %grp.n1247 = select i1 %.not, float %grp.n1239, float %grp.n1246
  %grp.n1248 = select i1 %.not, float %grp.n1246, float %grp.n1240
  %grp.n1249 = select i1 %.not, float %grp.n1242, float %grp.n1241
  %grp.n1250 = bitcast float %grp.n1249 to i32
  %grp.n1251 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1250)
  %grp.n1252 = bitcast i32 %grp.n1251 to float
  %grp.n1253 = select i1 %.not, float %grp.n1241, float %grp.n1252
  %grp.n1254 = select i1 %.not, float %grp.n1252, float %grp.n1242
  %grp.n1255 = select i1 %.not36, float %grp.n1253, float %grp.n1247
  %grp.n1256 = bitcast float %grp.n1255 to i32
  %grp.n1257 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1256)
  %grp.n1258 = bitcast i32 %grp.n1257 to float
  %grp.n1261 = select i1 %.not36, float %grp.n1254, float %grp.n1248
  %grp.n1262 = bitcast float %grp.n1261 to i32
  %grp.n1263 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1262)
  %grp.n1264 = bitcast i32 %grp.n1263 to float
  %84 = insertelement <4 x float> poison, float %grp.n1247, i64 0
  %85 = insertelement <4 x float> %84, float %grp.n1248, i64 1
  %86 = insertelement <4 x float> %85, float %grp.n1258, i64 2
  %87 = insertelement <4 x float> %86, float %grp.n1264, i64 3
  %88 = insertelement <4 x float> poison, float %grp.n1258, i64 0
  %89 = insertelement <4 x float> %88, float %grp.n1264, i64 1
  %90 = insertelement <4 x float> %89, float %grp.n1253, i64 2
  %91 = insertelement <4 x float> %90, float %grp.n1254, i64 3
  %92 = select <4 x i1> %65, <4 x float> %87, <4 x float> %91
  %grp.n1268 = extractelement <4 x float> %grp.n846, i64 0
  %grp.n1269 = extractelement <4 x float> %grp.n846, i64 1
  %grp.n1270 = extractelement <4 x float> %grp.n846, i64 2
  %grp.n1271 = extractelement <4 x float> %grp.n846, i64 3
  %grp.n1272 = select i1 %.not, float %grp.n1269, float %grp.n1268
  %grp.n1273 = bitcast float %grp.n1272 to i32
  %grp.n1274 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1273)
  %grp.n1275 = bitcast i32 %grp.n1274 to float
  %grp.n1276 = select i1 %.not, float %grp.n1268, float %grp.n1275
  %grp.n1277 = select i1 %.not, float %grp.n1275, float %grp.n1269
  %grp.n1278 = select i1 %.not, float %grp.n1271, float %grp.n1270
  %grp.n1279 = bitcast float %grp.n1278 to i32
  %grp.n1280 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1279)
  %grp.n1281 = bitcast i32 %grp.n1280 to float
  %grp.n1282 = select i1 %.not, float %grp.n1270, float %grp.n1281
  %grp.n1283 = select i1 %.not, float %grp.n1281, float %grp.n1271
  %grp.n1284 = select i1 %.not36, float %grp.n1282, float %grp.n1276
  %grp.n1285 = bitcast float %grp.n1284 to i32
  %grp.n1286 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1285)
  %grp.n1287 = select i1 %.not36, float %grp.n1283, float %grp.n1277
  %grp.n1288 = bitcast float %grp.n1287 to i32
  %grp.n1289 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1288)
  %grp.n1290 = shufflevector <1 x float> %grp.n1165, <1 x float> poison, <4 x i32> zeroinitializer
  %grp.n1291 = extractelement <4 x float> %grp.n1106, i64 0
  %grp.n1292 = extractelement <4 x float> %grp.n1106, i64 1
  %grp.n1293 = extractelement <4 x float> %grp.n1106, i64 2
  %grp.n1294 = extractelement <4 x float> %grp.n1106, i64 3
  %grp.n1295 = select i1 %.not, float %grp.n1292, float %grp.n1291
  %grp.n1296 = bitcast float %grp.n1295 to i32
  %grp.n1297 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1296)
  %grp.n1298 = bitcast i32 %grp.n1297 to float
  %grp.n1299 = select i1 %.not, float %grp.n1291, float %grp.n1298
  %grp.n1300 = select i1 %.not, float %grp.n1298, float %grp.n1292
  %grp.n1301 = select i1 %.not, float %grp.n1294, float %grp.n1293
  %grp.n1302 = bitcast float %grp.n1301 to i32
  %grp.n1303 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1302)
  %grp.n1304 = bitcast i32 %grp.n1303 to float
  %grp.n1305 = select i1 %.not, float %grp.n1293, float %grp.n1304
  %grp.n1306 = select i1 %.not, float %grp.n1304, float %grp.n1294
  %grp.n1307 = select i1 %.not36, float %grp.n1305, float %grp.n1299
  %grp.n1308 = bitcast float %grp.n1307 to i32
  %grp.n1309 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1308)
  %grp.n1310 = bitcast i32 %grp.n1309 to float
  %grp.n1313 = select i1 %.not36, float %grp.n1306, float %grp.n1300
  %grp.n1314 = bitcast float %grp.n1313 to i32
  %grp.n1315 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1314)
  %grp.n1316 = bitcast i32 %grp.n1315 to float
  %93 = insertelement <4 x float> poison, float %grp.n1299, i64 0
  %94 = insertelement <4 x float> %93, float %grp.n1300, i64 1
  %95 = insertelement <4 x float> %94, float %grp.n1310, i64 2
  %96 = insertelement <4 x float> %95, float %grp.n1316, i64 3
  %97 = insertelement <4 x float> poison, float %grp.n1310, i64 0
  %98 = insertelement <4 x float> %97, float %grp.n1316, i64 1
  %99 = insertelement <4 x float> %98, float %grp.n1305, i64 2
  %100 = insertelement <4 x float> %99, float %grp.n1306, i64 3
  %101 = select <4 x i1> %65, <4 x float> %96, <4 x float> %100
  %grp.n1319 = extractelement <4 x float> %grp.n1032, i64 0
  %grp.n1320 = extractelement <4 x float> %grp.n1032, i64 1
  %grp.n1321 = extractelement <4 x float> %grp.n1032, i64 2
  %grp.n1322 = extractelement <4 x float> %grp.n1032, i64 3
  %grp.n1323 = select i1 %.not, float %grp.n1320, float %grp.n1319
  %grp.n1324 = bitcast float %grp.n1323 to i32
  %grp.n1325 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1324)
  %grp.n1326 = bitcast i32 %grp.n1325 to float
  %grp.n1327 = select i1 %.not, float %grp.n1319, float %grp.n1326
  %grp.n1328 = select i1 %.not, float %grp.n1326, float %grp.n1320
  %grp.n1329 = select i1 %.not, float %grp.n1322, float %grp.n1321
  %grp.n1330 = bitcast float %grp.n1329 to i32
  %grp.n1331 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1330)
  %grp.n1332 = bitcast i32 %grp.n1331 to float
  %grp.n1333 = select i1 %.not, float %grp.n1321, float %grp.n1332
  %grp.n1334 = select i1 %.not, float %grp.n1332, float %grp.n1322
  %grp.n1335 = select i1 %.not36, float %grp.n1333, float %grp.n1327
  %grp.n1336 = bitcast float %grp.n1335 to i32
  %grp.n1337 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1336)
  %grp.n1338 = bitcast i32 %grp.n1337 to float
  %grp.n1341 = select i1 %.not36, float %grp.n1334, float %grp.n1328
  %grp.n1342 = bitcast float %grp.n1341 to i32
  %grp.n1343 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1342)
  %grp.n1344 = bitcast i32 %grp.n1343 to float
  %102 = insertelement <4 x float> poison, float %grp.n1327, i64 0
  %103 = insertelement <4 x float> %102, float %grp.n1328, i64 1
  %104 = insertelement <4 x float> %103, float %grp.n1338, i64 2
  %105 = insertelement <4 x float> %104, float %grp.n1344, i64 3
  %106 = insertelement <4 x float> poison, float %grp.n1338, i64 0
  %107 = insertelement <4 x float> %106, float %grp.n1344, i64 1
  %108 = insertelement <4 x float> %107, float %grp.n1333, i64 2
  %109 = insertelement <4 x float> %108, float %grp.n1334, i64 3
  %110 = select <4 x i1> %65, <4 x float> %105, <4 x float> %109
  %grp.n1347 = extractelement <4 x float> %grp.n948, i64 0
  %grp.n1348 = extractelement <4 x float> %grp.n948, i64 1
  %grp.n1349 = extractelement <4 x float> %grp.n948, i64 2
  %grp.n1350 = extractelement <4 x float> %grp.n948, i64 3
  %grp.n1351 = select i1 %.not, float %grp.n1348, float %grp.n1347
  %grp.n1352 = bitcast float %grp.n1351 to i32
  %grp.n1353 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1352)
  %grp.n1354 = bitcast i32 %grp.n1353 to float
  %grp.n1355 = select i1 %.not, float %grp.n1347, float %grp.n1354
  %grp.n1356 = select i1 %.not, float %grp.n1354, float %grp.n1348
  %grp.n1357 = select i1 %.not, float %grp.n1350, float %grp.n1349
  %grp.n1358 = bitcast float %grp.n1357 to i32
  %grp.n1359 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1358)
  %grp.n1360 = bitcast i32 %grp.n1359 to float
  %grp.n1361 = select i1 %.not, float %grp.n1349, float %grp.n1360
  %grp.n1362 = select i1 %.not, float %grp.n1360, float %grp.n1350
  %grp.n1363 = select i1 %.not36, float %grp.n1361, float %grp.n1355
  %grp.n1364 = bitcast float %grp.n1363 to i32
  %grp.n1365 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1364)
  %grp.n1366 = bitcast i32 %grp.n1365 to float
  %grp.n1369 = select i1 %.not36, float %grp.n1362, float %grp.n1356
  %grp.n1370 = bitcast float %grp.n1369 to i32
  %grp.n1371 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1370)
  %grp.n1372 = bitcast i32 %grp.n1371 to float
  %111 = insertelement <4 x float> poison, float %grp.n1355, i64 0
  %112 = insertelement <4 x float> %111, float %grp.n1356, i64 1
  %113 = insertelement <4 x float> %112, float %grp.n1366, i64 2
  %114 = insertelement <4 x float> %113, float %grp.n1372, i64 3
  %115 = insertelement <4 x float> poison, float %grp.n1366, i64 0
  %116 = insertelement <4 x float> %115, float %grp.n1372, i64 1
  %117 = insertelement <4 x float> %116, float %grp.n1361, i64 2
  %118 = insertelement <4 x float> %117, float %grp.n1362, i64 3
  %119 = select <4 x i1> %65, <4 x float> %114, <4 x float> %118
  %grp.n1375 = extractelement <4 x float> %grp.n864, i64 0
  %grp.n1376 = extractelement <4 x float> %grp.n864, i64 1
  %grp.n1377 = extractelement <4 x float> %grp.n864, i64 2
  %grp.n1378 = extractelement <4 x float> %grp.n864, i64 3
  %grp.n1379 = select i1 %.not, float %grp.n1376, float %grp.n1375
  %grp.n1380 = bitcast float %grp.n1379 to i32
  %grp.n1381 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1380)
  %grp.n1382 = bitcast i32 %grp.n1381 to float
  %grp.n1383 = select i1 %.not, float %grp.n1375, float %grp.n1382
  %grp.n1384 = select i1 %.not, float %grp.n1382, float %grp.n1376
  %grp.n1385 = select i1 %.not, float %grp.n1378, float %grp.n1377
  %grp.n1386 = bitcast float %grp.n1385 to i32
  %grp.n1387 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1386)
  %grp.n1388 = bitcast i32 %grp.n1387 to float
  %grp.n1389 = select i1 %.not, float %grp.n1377, float %grp.n1388
  %grp.n1390 = select i1 %.not, float %grp.n1388, float %grp.n1378
  %grp.n1391 = select i1 %.not36, float %grp.n1389, float %grp.n1383
  %grp.n1392 = bitcast float %grp.n1391 to i32
  %grp.n1393 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1392)
  %grp.n1394 = select i1 %.not36, float %grp.n1390, float %grp.n1384
  %grp.n1395 = bitcast float %grp.n1394 to i32
  %grp.n1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1395)
  %grp.n1398 = extractelement <4 x float> %grp.n1117, i64 0
  %grp.n1399 = extractelement <4 x float> %grp.n1117, i64 1
  %grp.n1400 = extractelement <4 x float> %grp.n1117, i64 2
  %grp.n1401 = extractelement <4 x float> %grp.n1117, i64 3
  %grp.n1402 = select i1 %.not, float %grp.n1399, float %grp.n1398
  %grp.n1403 = bitcast float %grp.n1402 to i32
  %grp.n1404 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1403)
  %grp.n1405 = bitcast i32 %grp.n1404 to float
  %grp.n1406 = select i1 %.not, float %grp.n1398, float %grp.n1405
  %grp.n1407 = select i1 %.not, float %grp.n1405, float %grp.n1399
  %grp.n1408 = select i1 %.not, float %grp.n1401, float %grp.n1400
  %grp.n1409 = bitcast float %grp.n1408 to i32
  %grp.n1410 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1409)
  %grp.n1411 = bitcast i32 %grp.n1410 to float
  %grp.n1412 = select i1 %.not, float %grp.n1400, float %grp.n1411
  %grp.n1413 = select i1 %.not, float %grp.n1411, float %grp.n1401
  %grp.n1414 = select i1 %.not36, float %grp.n1412, float %grp.n1406
  %grp.n1415 = bitcast float %grp.n1414 to i32
  %grp.n1416 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1415)
  %grp.n1417 = bitcast i32 %grp.n1416 to float
  %grp.n1420 = select i1 %.not36, float %grp.n1413, float %grp.n1407
  %grp.n1421 = bitcast float %grp.n1420 to i32
  %grp.n1422 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1421)
  %grp.n1423 = bitcast i32 %grp.n1422 to float
  %120 = insertelement <4 x float> poison, float %grp.n1406, i64 0
  %121 = insertelement <4 x float> %120, float %grp.n1407, i64 1
  %122 = insertelement <4 x float> %121, float %grp.n1417, i64 2
  %123 = insertelement <4 x float> %122, float %grp.n1423, i64 3
  %124 = insertelement <4 x float> poison, float %grp.n1417, i64 0
  %125 = insertelement <4 x float> %124, float %grp.n1423, i64 1
  %126 = insertelement <4 x float> %125, float %grp.n1412, i64 2
  %127 = insertelement <4 x float> %126, float %grp.n1413, i64 3
  %128 = select <4 x i1> %65, <4 x float> %123, <4 x float> %127
  %grp.n1427 = extractelement <4 x float> %grp.n1043, i64 0
  %grp.n1428 = extractelement <4 x float> %grp.n1043, i64 1
  %grp.n1429 = extractelement <4 x float> %grp.n1043, i64 2
  %grp.n1430 = extractelement <4 x float> %grp.n1043, i64 3
  %grp.n1431 = select i1 %.not, float %grp.n1428, float %grp.n1427
  %grp.n1432 = bitcast float %grp.n1431 to i32
  %grp.n1433 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1432)
  %grp.n1434 = bitcast i32 %grp.n1433 to float
  %grp.n1435 = select i1 %.not, float %grp.n1427, float %grp.n1434
  %grp.n1436 = select i1 %.not, float %grp.n1434, float %grp.n1428
  %grp.n1437 = select i1 %.not, float %grp.n1430, float %grp.n1429
  %grp.n1438 = bitcast float %grp.n1437 to i32
  %grp.n1439 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1438)
  %grp.n1440 = bitcast i32 %grp.n1439 to float
  %grp.n1441 = select i1 %.not, float %grp.n1429, float %grp.n1440
  %grp.n1442 = select i1 %.not, float %grp.n1440, float %grp.n1430
  %grp.n1443 = select i1 %.not36, float %grp.n1441, float %grp.n1435
  %grp.n1444 = bitcast float %grp.n1443 to i32
  %grp.n1445 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1444)
  %grp.n1446 = bitcast i32 %grp.n1445 to float
  %grp.n1449 = select i1 %.not36, float %grp.n1442, float %grp.n1436
  %grp.n1450 = bitcast float %grp.n1449 to i32
  %grp.n1451 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1450)
  %grp.n1452 = bitcast i32 %grp.n1451 to float
  %129 = insertelement <4 x float> poison, float %grp.n1435, i64 0
  %130 = insertelement <4 x float> %129, float %grp.n1436, i64 1
  %131 = insertelement <4 x float> %130, float %grp.n1446, i64 2
  %132 = insertelement <4 x float> %131, float %grp.n1452, i64 3
  %133 = insertelement <4 x float> poison, float %grp.n1446, i64 0
  %134 = insertelement <4 x float> %133, float %grp.n1452, i64 1
  %135 = insertelement <4 x float> %134, float %grp.n1441, i64 2
  %136 = insertelement <4 x float> %135, float %grp.n1442, i64 3
  %137 = select <4 x i1> %65, <4 x float> %132, <4 x float> %136
  %grp.n1456 = extractelement <4 x float> %grp.n959, i64 0
  %grp.n1457 = extractelement <4 x float> %grp.n959, i64 1
  %grp.n1458 = extractelement <4 x float> %grp.n959, i64 2
  %grp.n1459 = extractelement <4 x float> %grp.n959, i64 3
  %grp.n1460 = select i1 %.not, float %grp.n1457, float %grp.n1456
  %grp.n1461 = bitcast float %grp.n1460 to i32
  %grp.n1462 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1461)
  %grp.n1463 = bitcast i32 %grp.n1462 to float
  %grp.n1464 = select i1 %.not, float %grp.n1456, float %grp.n1463
  %grp.n1465 = select i1 %.not, float %grp.n1463, float %grp.n1457
  %grp.n1466 = select i1 %.not, float %grp.n1459, float %grp.n1458
  %grp.n1467 = bitcast float %grp.n1466 to i32
  %grp.n1468 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1467)
  %grp.n1469 = bitcast i32 %grp.n1468 to float
  %grp.n1470 = select i1 %.not, float %grp.n1458, float %grp.n1469
  %grp.n1471 = select i1 %.not, float %grp.n1469, float %grp.n1459
  %grp.n1472 = select i1 %.not36, float %grp.n1470, float %grp.n1464
  %grp.n1473 = bitcast float %grp.n1472 to i32
  %grp.n1474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1473)
  %grp.n1475 = bitcast i32 %grp.n1474 to float
  %grp.n1478 = select i1 %.not36, float %grp.n1471, float %grp.n1465
  %grp.n1479 = bitcast float %grp.n1478 to i32
  %grp.n1480 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1479)
  %grp.n1481 = bitcast i32 %grp.n1480 to float
  %138 = insertelement <4 x float> poison, float %grp.n1464, i64 0
  %139 = insertelement <4 x float> %138, float %grp.n1465, i64 1
  %140 = insertelement <4 x float> %139, float %grp.n1475, i64 2
  %141 = insertelement <4 x float> %140, float %grp.n1481, i64 3
  %142 = insertelement <4 x float> poison, float %grp.n1475, i64 0
  %143 = insertelement <4 x float> %142, float %grp.n1481, i64 1
  %144 = insertelement <4 x float> %143, float %grp.n1470, i64 2
  %145 = insertelement <4 x float> %144, float %grp.n1471, i64 3
  %146 = select <4 x i1> %65, <4 x float> %141, <4 x float> %145
  %grp.n1485 = extractelement <4 x float> %grp.n875, i64 0
  %grp.n1486 = extractelement <4 x float> %grp.n875, i64 1
  %grp.n1487 = extractelement <4 x float> %grp.n875, i64 2
  %grp.n1488 = extractelement <4 x float> %grp.n875, i64 3
  %grp.n1489 = select i1 %.not, float %grp.n1486, float %grp.n1485
  %grp.n1490 = bitcast float %grp.n1489 to i32
  %grp.n1491 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1490)
  %grp.n1492 = bitcast i32 %grp.n1491 to float
  %grp.n1493 = select i1 %.not, float %grp.n1485, float %grp.n1492
  %grp.n1494 = select i1 %.not, float %grp.n1492, float %grp.n1486
  %grp.n1495 = select i1 %.not, float %grp.n1488, float %grp.n1487
  %grp.n1496 = bitcast float %grp.n1495 to i32
  %grp.n1497 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1496)
  %grp.n1498 = bitcast i32 %grp.n1497 to float
  %grp.n1499 = select i1 %.not, float %grp.n1487, float %grp.n1498
  %grp.n1500 = select i1 %.not, float %grp.n1498, float %grp.n1488
  %grp.n1501 = select i1 %.not36, float %grp.n1499, float %grp.n1493
  %grp.n1502 = bitcast float %grp.n1501 to i32
  %grp.n1503 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1502)
  %grp.n1504 = select i1 %.not36, float %grp.n1500, float %grp.n1494
  %grp.n1505 = bitcast float %grp.n1504 to i32
  %grp.n1506 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1505)
  %grp.n1507 = shufflevector <1 x float> %grp.n1156, <1 x float> poison, <4 x i32> zeroinitializer
  %grp.n1508 = extractelement <4 x float> %grp.n1127, i64 0
  %grp.n1509 = extractelement <4 x float> %grp.n1127, i64 1
  %grp.n1510 = extractelement <4 x float> %grp.n1127, i64 2
  %grp.n1511 = extractelement <4 x float> %grp.n1127, i64 3
  %grp.n1512 = select i1 %.not, float %grp.n1509, float %grp.n1508
  %grp.n1513 = bitcast float %grp.n1512 to i32
  %grp.n1514 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1513)
  %grp.n1515 = bitcast i32 %grp.n1514 to float
  %grp.n1516 = select i1 %.not, float %grp.n1508, float %grp.n1515
  %grp.n1517 = select i1 %.not, float %grp.n1515, float %grp.n1509
  %grp.n1518 = select i1 %.not, float %grp.n1511, float %grp.n1510
  %grp.n1519 = bitcast float %grp.n1518 to i32
  %grp.n1520 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1519)
  %grp.n1521 = bitcast i32 %grp.n1520 to float
  %grp.n1522 = select i1 %.not, float %grp.n1510, float %grp.n1521
  %grp.n1523 = select i1 %.not, float %grp.n1521, float %grp.n1511
  %grp.n1524 = select i1 %.not36, float %grp.n1522, float %grp.n1516
  %grp.n1525 = bitcast float %grp.n1524 to i32
  %grp.n1526 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1525)
  %grp.n1527 = bitcast i32 %grp.n1526 to float
  %grp.n1530 = select i1 %.not36, float %grp.n1523, float %grp.n1517
  %grp.n1531 = bitcast float %grp.n1530 to i32
  %grp.n1532 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1531)
  %grp.n1533 = bitcast i32 %grp.n1532 to float
  %147 = insertelement <4 x float> poison, float %grp.n1516, i64 0
  %148 = insertelement <4 x float> %147, float %grp.n1517, i64 1
  %149 = insertelement <4 x float> %148, float %grp.n1527, i64 2
  %150 = insertelement <4 x float> %149, float %grp.n1533, i64 3
  %151 = insertelement <4 x float> poison, float %grp.n1527, i64 0
  %152 = insertelement <4 x float> %151, float %grp.n1533, i64 1
  %153 = insertelement <4 x float> %152, float %grp.n1522, i64 2
  %154 = insertelement <4 x float> %153, float %grp.n1523, i64 3
  %155 = select <4 x i1> %65, <4 x float> %150, <4 x float> %154
  %grp.n1536 = extractelement <4 x float> %grp.n1053, i64 0
  %grp.n1537 = extractelement <4 x float> %grp.n1053, i64 1
  %grp.n1538 = extractelement <4 x float> %grp.n1053, i64 2
  %grp.n1539 = extractelement <4 x float> %grp.n1053, i64 3
  %grp.n1540 = select i1 %.not, float %grp.n1537, float %grp.n1536
  %grp.n1541 = bitcast float %grp.n1540 to i32
  %grp.n1542 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1541)
  %grp.n1543 = bitcast i32 %grp.n1542 to float
  %grp.n1544 = select i1 %.not, float %grp.n1536, float %grp.n1543
  %grp.n1545 = select i1 %.not, float %grp.n1543, float %grp.n1537
  %grp.n1546 = select i1 %.not, float %grp.n1539, float %grp.n1538
  %grp.n1547 = bitcast float %grp.n1546 to i32
  %grp.n1548 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1547)
  %grp.n1549 = bitcast i32 %grp.n1548 to float
  %grp.n1550 = select i1 %.not, float %grp.n1538, float %grp.n1549
  %grp.n1551 = select i1 %.not, float %grp.n1549, float %grp.n1539
  %grp.n1552 = select i1 %.not36, float %grp.n1550, float %grp.n1544
  %grp.n1553 = bitcast float %grp.n1552 to i32
  %grp.n1554 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1553)
  %grp.n1555 = bitcast i32 %grp.n1554 to float
  %grp.n1558 = select i1 %.not36, float %grp.n1551, float %grp.n1545
  %grp.n1559 = bitcast float %grp.n1558 to i32
  %grp.n1560 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1559)
  %grp.n1561 = bitcast i32 %grp.n1560 to float
  %156 = insertelement <4 x float> poison, float %grp.n1544, i64 0
  %157 = insertelement <4 x float> %156, float %grp.n1545, i64 1
  %158 = insertelement <4 x float> %157, float %grp.n1555, i64 2
  %159 = insertelement <4 x float> %158, float %grp.n1561, i64 3
  %160 = insertelement <4 x float> poison, float %grp.n1555, i64 0
  %161 = insertelement <4 x float> %160, float %grp.n1561, i64 1
  %162 = insertelement <4 x float> %161, float %grp.n1550, i64 2
  %163 = insertelement <4 x float> %162, float %grp.n1551, i64 3
  %164 = select <4 x i1> %65, <4 x float> %159, <4 x float> %163
  %grp.n1564 = extractelement <4 x float> %grp.n969, i64 0
  %grp.n1565 = extractelement <4 x float> %grp.n969, i64 1
  %grp.n1566 = extractelement <4 x float> %grp.n969, i64 2
  %grp.n1567 = extractelement <4 x float> %grp.n969, i64 3
  %grp.n1568 = select i1 %.not, float %grp.n1565, float %grp.n1564
  %grp.n1569 = bitcast float %grp.n1568 to i32
  %grp.n1570 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1569)
  %grp.n1571 = bitcast i32 %grp.n1570 to float
  %grp.n1572 = select i1 %.not, float %grp.n1564, float %grp.n1571
  %grp.n1573 = select i1 %.not, float %grp.n1571, float %grp.n1565
  %grp.n1574 = select i1 %.not, float %grp.n1567, float %grp.n1566
  %grp.n1575 = bitcast float %grp.n1574 to i32
  %grp.n1576 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1575)
  %grp.n1577 = bitcast i32 %grp.n1576 to float
  %grp.n1578 = select i1 %.not, float %grp.n1566, float %grp.n1577
  %grp.n1579 = select i1 %.not, float %grp.n1577, float %grp.n1567
  %grp.n1580 = select i1 %.not36, float %grp.n1578, float %grp.n1572
  %grp.n1581 = bitcast float %grp.n1580 to i32
  %grp.n1582 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1581)
  %grp.n1583 = bitcast i32 %grp.n1582 to float
  %grp.n1586 = select i1 %.not36, float %grp.n1579, float %grp.n1573
  %grp.n1587 = bitcast float %grp.n1586 to i32
  %grp.n1588 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1587)
  %grp.n1589 = bitcast i32 %grp.n1588 to float
  %165 = insertelement <4 x float> poison, float %grp.n1572, i64 0
  %166 = insertelement <4 x float> %165, float %grp.n1573, i64 1
  %167 = insertelement <4 x float> %166, float %grp.n1583, i64 2
  %168 = insertelement <4 x float> %167, float %grp.n1589, i64 3
  %169 = insertelement <4 x float> poison, float %grp.n1583, i64 0
  %170 = insertelement <4 x float> %169, float %grp.n1589, i64 1
  %171 = insertelement <4 x float> %170, float %grp.n1578, i64 2
  %172 = insertelement <4 x float> %171, float %grp.n1579, i64 3
  %173 = select <4 x i1> %65, <4 x float> %168, <4 x float> %172
  %grp.n1592 = extractelement <4 x float> %grp.n885, i64 0
  %grp.n1593 = extractelement <4 x float> %grp.n885, i64 1
  %grp.n1594 = extractelement <4 x float> %grp.n885, i64 2
  %grp.n1595 = extractelement <4 x float> %grp.n885, i64 3
  %grp.n1596 = select i1 %.not, float %grp.n1593, float %grp.n1592
  %grp.n1597 = bitcast float %grp.n1596 to i32
  %grp.n1598 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1597)
  %grp.n1599 = bitcast i32 %grp.n1598 to float
  %grp.n1600 = select i1 %.not, float %grp.n1592, float %grp.n1599
  %grp.n1601 = select i1 %.not, float %grp.n1599, float %grp.n1593
  %grp.n1602 = select i1 %.not, float %grp.n1595, float %grp.n1594
  %grp.n1603 = bitcast float %grp.n1602 to i32
  %grp.n1604 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1603)
  %grp.n1605 = bitcast i32 %grp.n1604 to float
  %grp.n1606 = select i1 %.not, float %grp.n1594, float %grp.n1605
  %grp.n1607 = select i1 %.not, float %grp.n1605, float %grp.n1595
  %grp.n1608 = select i1 %.not36, float %grp.n1606, float %grp.n1600
  %grp.n1609 = bitcast float %grp.n1608 to i32
  %grp.n1610 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1609)
  %grp.n1611 = select i1 %.not36, float %grp.n1607, float %grp.n1601
  %grp.n1612 = bitcast float %grp.n1611 to i32
  %grp.n1613 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1612)
  %grp.n1614 = bitcast i32 %grp.n1610 to float
  %grp.n1617 = bitcast i32 %grp.n1613 to float
  %174 = insertelement <4 x float> poison, float %grp.n1600, i64 0
  %175 = insertelement <4 x float> %174, float %grp.n1601, i64 1
  %176 = insertelement <4 x float> %175, float %grp.n1614, i64 2
  %177 = insertelement <4 x float> %176, float %grp.n1617, i64 3
  %178 = insertelement <4 x float> poison, float %grp.n1614, i64 0
  %179 = insertelement <4 x float> %178, float %grp.n1617, i64 1
  %180 = insertelement <4 x float> %179, float %grp.n1606, i64 2
  %181 = insertelement <4 x float> %180, float %grp.n1607, i64 3
  %182 = select <4 x i1> %65, <4 x float> %177, <4 x float> %181
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %grp.n1507
  %normalized = fmul <4 x float> %row.reciprocal, %182
  %grp.n1624 = fptrunc <4 x float> %normalized to <4 x half>
  %grp.n1625 = extractelement <4 x half> %grp.n1624, i64 3
  %grp.n1626 = extractelement <4 x half> %grp.n1624, i64 2
  %grp.n1627 = extractelement <4 x half> %grp.n1624, i64 1
  %grp.n1628 = extractelement <4 x half> %grp.n1624, i64 0
  %grp.n1629 = bitcast i32 %grp.n1503 to float
  %grp.n1632 = bitcast i32 %grp.n1506 to float
  %183 = insertelement <4 x float> poison, float %grp.n1493, i64 0
  %184 = insertelement <4 x float> %183, float %grp.n1494, i64 1
  %185 = insertelement <4 x float> %184, float %grp.n1629, i64 2
  %186 = insertelement <4 x float> %185, float %grp.n1632, i64 3
  %187 = insertelement <4 x float> poison, float %grp.n1629, i64 0
  %188 = insertelement <4 x float> %187, float %grp.n1632, i64 1
  %189 = insertelement <4 x float> %188, float %grp.n1499, i64 2
  %190 = insertelement <4 x float> %189, float %grp.n1500, i64 3
  %191 = select <4 x i1> %65, <4 x float> %186, <4 x float> %190
  %normalized37 = fmul <4 x float> %row.reciprocal, %191
  %grp.n1639 = fptrunc <4 x float> %normalized37 to <4 x half>
  %grp.n1640 = extractelement <4 x half> %grp.n1639, i64 3
  %grp.n1641 = extractelement <4 x half> %grp.n1639, i64 2
  %grp.n1642 = extractelement <4 x half> %grp.n1639, i64 1
  %grp.n1643 = extractelement <4 x half> %grp.n1639, i64 0
  %grp.n1644 = bitcast i32 %grp.n1393 to float
  %grp.n1647 = bitcast i32 %grp.n1396 to float
  %192 = insertelement <4 x float> poison, float %grp.n1383, i64 0
  %193 = insertelement <4 x float> %192, float %grp.n1384, i64 1
  %194 = insertelement <4 x float> %193, float %grp.n1644, i64 2
  %195 = insertelement <4 x float> %194, float %grp.n1647, i64 3
  %196 = insertelement <4 x float> poison, float %grp.n1644, i64 0
  %197 = insertelement <4 x float> %196, float %grp.n1647, i64 1
  %198 = insertelement <4 x float> %197, float %grp.n1389, i64 2
  %199 = insertelement <4 x float> %198, float %grp.n1390, i64 3
  %200 = select <4 x i1> %65, <4 x float> %195, <4 x float> %199
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %grp.n1290
  %normalized39 = fmul <4 x float> %row.reciprocal38, %200
  %grp.n1654 = fptrunc <4 x float> %normalized39 to <4 x half>
  %grp.n1655 = extractelement <4 x half> %grp.n1654, i64 3
  %grp.n1656 = extractelement <4 x half> %grp.n1654, i64 2
  %grp.n1657 = extractelement <4 x half> %grp.n1654, i64 1
  %grp.n1658 = extractelement <4 x half> %grp.n1654, i64 0
  %grp.n1659 = bitcast i32 %grp.n1286 to float
  %grp.n1662 = bitcast i32 %grp.n1289 to float
  %201 = insertelement <4 x float> poison, float %grp.n1276, i64 0
  %202 = insertelement <4 x float> %201, float %grp.n1277, i64 1
  %203 = insertelement <4 x float> %202, float %grp.n1659, i64 2
  %204 = insertelement <4 x float> %203, float %grp.n1662, i64 3
  %205 = insertelement <4 x float> poison, float %grp.n1659, i64 0
  %206 = insertelement <4 x float> %205, float %grp.n1662, i64 1
  %207 = insertelement <4 x float> %206, float %grp.n1282, i64 2
  %208 = insertelement <4 x float> %207, float %grp.n1283, i64 3
  %209 = select <4 x i1> %65, <4 x float> %204, <4 x float> %208
  %normalized40 = fmul <4 x float> %row.reciprocal38, %209
  %grp.n1669 = fptrunc <4 x float> %normalized40 to <4 x half>
  %grp.n1670 = extractelement <4 x half> %grp.n1669, i64 3
  %grp.n1671 = extractelement <4 x half> %grp.n1669, i64 2
  %grp.n1672 = extractelement <4 x half> %grp.n1669, i64 1
  %grp.n1673 = extractelement <4 x half> %grp.n1669, i64 0
  %address.thread974 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !range !2, !frisk.register_lifetimes !3
  %address.local975 = zext nneg i32 %address.thread974 to i64
  %grp.n1677 = lshr i64 %address.local975, 2
  %address.local978 = and i64 %grp.n1677, 48
  %address.local980 = and i64 %address.local975, 240
  %address.local979 = add i64 %grp.n21, %address.local975
  %address.local981 = sub i64 %address.local979, %address.local980
  %address.local982 = add i64 %address.local981, %address.local978
  %210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %grp.n22
  %address.local985.idx = shl nuw nsw i64 %address.local982, 8
  %address.local985 = getelementptr inbounds nuw i8, ptr addrspace(1) %210, i64 %address.local985.idx
  %address.local987 = and i64 %grp.n1677, 12
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987
  store half %grp.n1673, ptr addrspace(1) %address.local988, align 2
  %address.local990 = or disjoint i64 %address.local987, 1
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990
  store half %grp.n1672, ptr addrspace(1) %address.local991, align 2
  %address.local993 = or disjoint i64 %address.local987, 2
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993
  store half %grp.n1671, ptr addrspace(1) %address.local994, align 2
  %address.local996 = or disjoint i64 %address.local987, 3
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996
  store half %grp.n1670, ptr addrspace(1) %address.local997, align 2
  %address.local999 = or disjoint i64 %address.local987, 16
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999
  store half %grp.n1658, ptr addrspace(1) %address.local1000, align 2
  %address.local1002 = or disjoint i64 %address.local987, 17
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002
  store half %grp.n1657, ptr addrspace(1) %address.local1003, align 2
  %address.local1005 = or disjoint i64 %address.local987, 18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005
  store half %grp.n1656, ptr addrspace(1) %address.local1006, align 2
  %address.local1008 = or disjoint i64 %address.local987, 19
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008
  store half %grp.n1655, ptr addrspace(1) %address.local1009, align 2
  %address.local1013 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local985, i64 16384
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987
  store half %grp.n1643, ptr addrspace(1) %address.local1014, align 2
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990
  store half %grp.n1642, ptr addrspace(1) %address.local1016, align 2
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993
  store half %grp.n1641, ptr addrspace(1) %address.local1018, align 2
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996
  store half %grp.n1640, ptr addrspace(1) %address.local1020, align 2
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999
  store half %grp.n1628, ptr addrspace(1) %address.local1022, align 2
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002
  store half %grp.n1627, ptr addrspace(1) %address.local1024, align 2
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005
  store half %grp.n1626, ptr addrspace(1) %address.local1026, align 2
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008
  store half %grp.n1625, ptr addrspace(1) %address.local1028, align 2
  %normalized42 = fmul <4 x float> %row.reciprocal, %173
  %grp.n1682 = fptrunc <4 x float> %normalized42 to <4 x half>
  %grp.n1683 = extractelement <4 x half> %grp.n1682, i64 3
  %grp.n1684 = extractelement <4 x half> %grp.n1682, i64 2
  %grp.n1685 = extractelement <4 x half> %grp.n1682, i64 1
  %grp.n1686 = extractelement <4 x half> %grp.n1682, i64 0
  %normalized43 = fmul <4 x float> %row.reciprocal, %146
  %grp.n1691 = fptrunc <4 x float> %normalized43 to <4 x half>
  %grp.n1692 = extractelement <4 x half> %grp.n1691, i64 3
  %grp.n1693 = extractelement <4 x half> %grp.n1691, i64 2
  %grp.n1694 = extractelement <4 x half> %grp.n1691, i64 1
  %grp.n1695 = extractelement <4 x half> %grp.n1691, i64 0
  %normalized45 = fmul <4 x float> %row.reciprocal38, %119
  %grp.n1700 = fptrunc <4 x float> %normalized45 to <4 x half>
  %grp.n1701 = extractelement <4 x half> %grp.n1700, i64 3
  %grp.n1702 = extractelement <4 x half> %grp.n1700, i64 2
  %grp.n1703 = extractelement <4 x half> %grp.n1700, i64 1
  %grp.n1704 = extractelement <4 x half> %grp.n1700, i64 0
  %normalized46 = fmul <4 x float> %row.reciprocal38, %92
  %grp.n1709 = fptrunc <4 x float> %normalized46 to <4 x half>
  %grp.n1710 = extractelement <4 x half> %grp.n1709, i64 3
  %grp.n1711 = extractelement <4 x half> %grp.n1709, i64 2
  %grp.n1712 = extractelement <4 x half> %grp.n1709, i64 1
  %grp.n1713 = extractelement <4 x half> %grp.n1709, i64 0
  %address.local1043 = or disjoint i64 %address.local987, 32
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043
  store half %grp.n1713, ptr addrspace(1) %address.local1044, align 2
  %address.local1046 = or disjoint i64 %address.local987, 33
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046
  store half %grp.n1712, ptr addrspace(1) %address.local1047, align 2
  %address.local1049 = or disjoint i64 %address.local987, 34
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049
  store half %grp.n1711, ptr addrspace(1) %address.local1050, align 2
  %address.local1052 = or disjoint i64 %address.local987, 35
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052
  store half %grp.n1710, ptr addrspace(1) %address.local1053, align 2
  %address.local1055 = or disjoint i64 %address.local987, 48
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055
  store half %grp.n1704, ptr addrspace(1) %address.local1056, align 2
  %address.local1058 = or disjoint i64 %address.local987, 49
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058
  store half %grp.n1703, ptr addrspace(1) %address.local1059, align 2
  %address.local1061 = or disjoint i64 %address.local987, 50
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061
  store half %grp.n1702, ptr addrspace(1) %address.local1062, align 2
  %address.local1064 = or i64 %grp.n1677, 51
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064
  store half %grp.n1701, ptr addrspace(1) %address.local1065, align 2
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043
  store half %grp.n1695, ptr addrspace(1) %address.local1070, align 2
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046
  store half %grp.n1694, ptr addrspace(1) %address.local1072, align 2
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049
  store half %grp.n1693, ptr addrspace(1) %address.local1074, align 2
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052
  store half %grp.n1692, ptr addrspace(1) %address.local1076, align 2
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055
  store half %grp.n1686, ptr addrspace(1) %address.local1078, align 2
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058
  store half %grp.n1685, ptr addrspace(1) %address.local1080, align 2
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061
  store half %grp.n1684, ptr addrspace(1) %address.local1082, align 2
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064
  store half %grp.n1683, ptr addrspace(1) %address.local1084, align 2
  %normalized48 = fmul <4 x float> %row.reciprocal, %164
  %grp.n1720 = fptrunc <4 x float> %normalized48 to <4 x half>
  %grp.n1721 = extractelement <4 x half> %grp.n1720, i64 3
  %grp.n1722 = extractelement <4 x half> %grp.n1720, i64 2
  %grp.n1723 = extractelement <4 x half> %grp.n1720, i64 1
  %grp.n1724 = extractelement <4 x half> %grp.n1720, i64 0
  %normalized49 = fmul <4 x float> %row.reciprocal, %137
  %grp.n1729 = fptrunc <4 x float> %normalized49 to <4 x half>
  %grp.n1730 = extractelement <4 x half> %grp.n1729, i64 3
  %grp.n1731 = extractelement <4 x half> %grp.n1729, i64 2
  %grp.n1732 = extractelement <4 x half> %grp.n1729, i64 1
  %grp.n1733 = extractelement <4 x half> %grp.n1729, i64 0
  %normalized51 = fmul <4 x float> %row.reciprocal38, %110
  %grp.n1738 = fptrunc <4 x float> %normalized51 to <4 x half>
  %grp.n1739 = extractelement <4 x half> %grp.n1738, i64 3
  %grp.n1740 = extractelement <4 x half> %grp.n1738, i64 2
  %grp.n1741 = extractelement <4 x half> %grp.n1738, i64 1
  %grp.n1742 = extractelement <4 x half> %grp.n1738, i64 0
  %normalized52 = fmul <4 x float> %row.reciprocal38, %83
  %grp.n1747 = fptrunc <4 x float> %normalized52 to <4 x half>
  %grp.n1748 = extractelement <4 x half> %grp.n1747, i64 3
  %grp.n1749 = extractelement <4 x half> %grp.n1747, i64 2
  %grp.n1750 = extractelement <4 x half> %grp.n1747, i64 1
  %grp.n1751 = extractelement <4 x half> %grp.n1747, i64 0
  %address.local1099 = or disjoint i64 %address.local987, 64
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099
  store half %grp.n1751, ptr addrspace(1) %address.local1100, align 2
  %address.local1102 = or disjoint i64 %address.local987, 65
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102
  store half %grp.n1750, ptr addrspace(1) %address.local1103, align 2
  %address.local1105 = or disjoint i64 %address.local987, 66
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105
  store half %grp.n1749, ptr addrspace(1) %address.local1106, align 2
  %address.local1108 = or disjoint i64 %address.local987, 67
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108
  store half %grp.n1748, ptr addrspace(1) %address.local1109, align 2
  %address.local1111 = or disjoint i64 %address.local987, 80
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111
  store half %grp.n1742, ptr addrspace(1) %address.local1112, align 2
  %address.local1114 = or disjoint i64 %address.local987, 81
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114
  store half %grp.n1741, ptr addrspace(1) %address.local1115, align 2
  %address.local1117 = or disjoint i64 %address.local987, 82
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117
  store half %grp.n1740, ptr addrspace(1) %address.local1118, align 2
  %address.local1120 = or disjoint i64 %address.local987, 83
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120
  store half %grp.n1739, ptr addrspace(1) %address.local1121, align 2
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099
  store half %grp.n1733, ptr addrspace(1) %address.local1126, align 2
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102
  store half %grp.n1732, ptr addrspace(1) %address.local1128, align 2
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105
  store half %grp.n1731, ptr addrspace(1) %address.local1130, align 2
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108
  store half %grp.n1730, ptr addrspace(1) %address.local1132, align 2
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111
  store half %grp.n1724, ptr addrspace(1) %address.local1134, align 2
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114
  store half %grp.n1723, ptr addrspace(1) %address.local1136, align 2
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117
  store half %grp.n1722, ptr addrspace(1) %address.local1138, align 2
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120
  store half %grp.n1721, ptr addrspace(1) %address.local1140, align 2
  %normalized54 = fmul <4 x float> %row.reciprocal, %155
  %grp.n1758 = fptrunc <4 x float> %normalized54 to <4 x half>
  %grp.n1759 = extractelement <4 x half> %grp.n1758, i64 3
  %grp.n1760 = extractelement <4 x half> %grp.n1758, i64 2
  %grp.n1761 = extractelement <4 x half> %grp.n1758, i64 1
  %grp.n1762 = extractelement <4 x half> %grp.n1758, i64 0
  %normalized55 = fmul <4 x float> %row.reciprocal, %128
  %grp.n1767 = fptrunc <4 x float> %normalized55 to <4 x half>
  %grp.n1768 = extractelement <4 x half> %grp.n1767, i64 3
  %grp.n1769 = extractelement <4 x half> %grp.n1767, i64 2
  %grp.n1770 = extractelement <4 x half> %grp.n1767, i64 1
  %grp.n1771 = extractelement <4 x half> %grp.n1767, i64 0
  %normalized57 = fmul <4 x float> %row.reciprocal38, %101
  %grp.n1776 = fptrunc <4 x float> %normalized57 to <4 x half>
  %grp.n1777 = extractelement <4 x half> %grp.n1776, i64 3
  %grp.n1778 = extractelement <4 x half> %grp.n1776, i64 2
  %grp.n1779 = extractelement <4 x half> %grp.n1776, i64 1
  %grp.n1780 = extractelement <4 x half> %grp.n1776, i64 0
  %normalized58 = fmul <4 x float> %row.reciprocal38, %74
  %grp.n1785 = fptrunc <4 x float> %normalized58 to <4 x half>
  %grp.n1786 = extractelement <4 x half> %grp.n1785, i64 3
  %grp.n1787 = extractelement <4 x half> %grp.n1785, i64 2
  %grp.n1788 = extractelement <4 x half> %grp.n1785, i64 1
  %grp.n1789 = extractelement <4 x half> %grp.n1785, i64 0
  %address.local1155 = or disjoint i64 %address.local987, 96
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155
  store half %grp.n1789, ptr addrspace(1) %address.local1156, align 2
  %address.local1158 = or disjoint i64 %address.local987, 97
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158
  store half %grp.n1788, ptr addrspace(1) %address.local1159, align 2
  %address.local1161 = or disjoint i64 %address.local987, 98
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161
  store half %grp.n1787, ptr addrspace(1) %address.local1162, align 2
  %address.local1164 = or disjoint i64 %address.local987, 99
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164
  store half %grp.n1786, ptr addrspace(1) %address.local1165, align 2
  %address.local1167 = or disjoint i64 %address.local987, 112
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167
  store half %grp.n1780, ptr addrspace(1) %address.local1168, align 2
  %address.local1170 = or disjoint i64 %address.local987, 113
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170
  store half %grp.n1779, ptr addrspace(1) %address.local1171, align 2
  %address.local1173 = or disjoint i64 %address.local987, 114
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173
  store half %grp.n1778, ptr addrspace(1) %address.local1174, align 2
  %address.local1176 = or i64 %grp.n1677, 115
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176
  store half %grp.n1777, ptr addrspace(1) %address.local1177, align 2
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155
  store half %grp.n1771, ptr addrspace(1) %address.local1182, align 2
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158
  store half %grp.n1770, ptr addrspace(1) %address.local1184, align 2
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161
  store half %grp.n1769, ptr addrspace(1) %address.local1186, align 2
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164
  store half %grp.n1768, ptr addrspace(1) %address.local1188, align 2
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167
  store half %grp.n1762, ptr addrspace(1) %address.local1190, align 2
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170
  store half %grp.n1761, ptr addrspace(1) %address.local1192, align 2
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173
  store half %grp.n1760, ptr addrspace(1) %address.local1194, align 2
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176
  store half %grp.n1759, ptr addrspace(1) %address.local1196, align 2
  ret void
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { convergent nounwind memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }
attributes #12 = { memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 256, i32 1, i32 1}
!2 = !{i32 0, i32 256}
!3 = !{}
!4 = !{i32 4}
