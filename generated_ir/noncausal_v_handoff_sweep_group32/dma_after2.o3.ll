; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_v_handoff_sweep_group32/dma_after2.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_frag_prefetch_forward/control/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %grp.n5 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %grp.n6 = sext i32 %grp.n5 to i64, !dbg !7
  %grp.n7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %grp.n8 = sext i32 %grp.n7 to i64, !dbg !7
  %grp.n9 = shl nsw i64 %grp.n8, 5, !dbg !7
  %grp.n10 = add nsw i64 %grp.n9, %grp.n6, !dbg !7
  %grp.n11 = lshr i64 %grp.n10, 5, !dbg !7
  %grp.n12 = and i64 %grp.n11, 576460752303423456, !dbg !7
  %grp.n13 = and i64 %grp.n6, 31, !dbg !7
  %grp.n14 = or disjoint i64 %grp.n12, %grp.n13, !dbg !7
  %grp.n16 = lshr i64 %grp.n10, 10, !dbg !8
  %grp.n17 = add nuw nsw i64 %grp.n11, %grp.n16, !dbg !8
  %grp.n19 = shl nuw nsw i64 %grp.n14, 7, !dbg !8
  %5 = shl i64 %grp.n16, 12, !dbg !8
  %grp.n21 = sub i64 %grp.n19, %5, !dbg !8
  %grp.n18 = shl i64 %grp.n17, 19, !dbg !8
  %grp.n22 = and i64 %grp.n18, 16252928, !dbg !8
  %grp.n23 = shl nsw i64 %grp.n21, 7, !dbg !8
  %grp.n24 = add i64 %grp.n23, %grp.n22, !dbg !8
  %grp.n25 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %grp.n25) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %grp.n28 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %address.local413, %grp.n24, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %grp.n29 = insertelement <2 x i64> poison, i64 %grp.n28, i64 0, !dbg !8
  %grp.n30 = bitcast <2 x i64> %grp.n29 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %grp.n30, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local424, %address.local410, !dbg !8
  %.idx1400 = or disjoint i64 %address.local425, %grp.n24, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %grp.n39 = or disjoint i64 %grp.n24, 32, !dbg !8
  %grp.n40 = or disjoint i64 %address.local413, %grp.n39, !dbg !8
  %.tr = trunc i64 %grp.n40 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %6 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local406, i64 0, !dbg !8
  %7 = insertelement <4 x ptr addrspace(3)> %6, ptr addrspace(3) %address.local418, i64 1, !dbg !8
  %8 = insertelement <4 x ptr addrspace(3)> %7, ptr addrspace(3) %address.local427, i64 2, !dbg !8
  %9 = insertelement <4 x ptr addrspace(3)> %8, ptr addrspace(3) %address.local431, i64 3, !dbg !8
  %10 = ptrtoint <4 x ptr addrspace(3)> %9 to <4 x i64>, !dbg !8
  %11 = trunc <4 x i64> %10 to <4 x i32>, !dbg !8
  %12 = extractelement <4 x i32> %11, i64 0, !dbg !8
  %grp.n27 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %12), !dbg !8
  %dma.buffer.lds1398 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n27), !dbg !8
  %grp.n31 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %13 = extractelement <4 x i32> %11, i64 1, !dbg !8
  %grp.n32 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %13), !dbg !8
  %dma.buffer.lds1395 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n32), !dbg !8
  %grp.n36 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %14 = extractelement <4 x i32> %11, i64 2, !dbg !8
  %grp.n37 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %14), !dbg !8
  %dma.buffer.lds1392 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n37), !dbg !8
  %grp.n43 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %15 = extractelement <4 x i32> %11, i64 3, !dbg !8
  %grp.n44 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %15), !dbg !8
  %grp.n47 = or disjoint i64 %address.local425, %grp.n39, !dbg !8
  %.tr1401 = trunc i64 %grp.n47 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %dma.buffer.lds1389 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n44), !dbg !8
  %grp.n50 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
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
  %grp.n51 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %grp.n52 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %grp.n53 = fmul <4 x half> %grp.n51, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %grp.n54 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %grp.n55 = fmul <4 x half> %grp.n52, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %grp.n56 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %grp.n57 = fmul <4 x half> %grp.n54, splat (half 1.275630e-01), !dbg !10
  %grp.n58 = fmul <4 x half> %grp.n56, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %grp.n61 = or disjoint i64 %grp.n24, 64, !dbg !8
  %grp.n62 = or disjoint i64 %address.local413, %grp.n61, !dbg !8
  %.tr1402 = trunc i64 %grp.n62 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %grp.n65 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %grp.n69 = or disjoint i64 %address.local425, %grp.n61, !dbg !8
  %.tr1403 = trunc i64 %grp.n69 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %grp.n72 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %grp.n73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %grp.n74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %grp.n75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %grp.n76 = fmul <4 x half> %grp.n73, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %grp.n77 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %grp.n78 = fmul <4 x half> %grp.n74, splat (half 1.275630e-01), !dbg !10
  %grp.n79 = fmul <4 x half> %grp.n75, splat (half 1.275630e-01), !dbg !10
  %grp.n80 = fmul <4 x half> %grp.n77, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %grp.n83 = or disjoint i64 %grp.n24, 96, !dbg !8
  %grp.n84 = or disjoint i64 %address.local413, %grp.n83, !dbg !8
  %.tr1404 = trunc i64 %grp.n84 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %grp.n87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %grp.n91 = or disjoint i64 %address.local425, %grp.n83, !dbg !8
  %.tr1405 = trunc i64 %grp.n91 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %grp.n94 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %grp.n95 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %grp.n96 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %grp.n97 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %grp.n98 = fmul <4 x half> %grp.n95, splat (half 1.275630e-01), !dbg !10
  %grp.n99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %grp.n100 = fmul <4 x half> %grp.n96, splat (half 1.275630e-01), !dbg !10
  %grp.n101 = fmul <4 x half> %grp.n97, splat (half 1.275630e-01), !dbg !10
  %grp.n102 = fmul <4 x half> %grp.n99, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %grp.n103 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %grp.n104 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %grp.n105 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %grp.n106 = fmul <4 x half> %grp.n103, splat (half 1.275630e-01), !dbg !10
  %grp.n107 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %grp.n108 = fmul <4 x half> %grp.n104, splat (half 1.275630e-01), !dbg !10
  %grp.n109 = fmul <4 x half> %grp.n105, splat (half 1.275630e-01), !dbg !10
  %grp.n110 = fmul <4 x half> %grp.n107, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %16 = shl nuw nsw i32 %grp.n25, 2, !dbg !12
  %17 = and i32 %16, 12, !dbg !12
  %grp.n115 = and i32 %grp.n25, 1, !dbg !12
  %grp.n1161 = add nuw nsw i32 %grp.n115, 1, !dbg !12
  %narrow = or disjoint i32 %grp.n1161, %17, !dbg !12
  %18 = lshr i32 %narrow, 1, !dbg !12
  %grp.n118 = zext nneg i32 %18 to i64, !dbg !12
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
  br label %grp.n128, !dbg !11

grp.n128:                                         ; preds = %grp.n128, %4
  %grp.n12616 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %grp.n1145, %grp.n128 ]
  %grp.n12515 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %grp.n1141, %grp.n128 ]
  %grp.n12414 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %grp.n1137, %grp.n128 ]
  %grp.n12313 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %grp.n1133, %grp.n128 ]
  %grp.n12212 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %grp.n801, %grp.n128 ]
  %grp.n12111 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %grp.n1131, %grp.n128 ]
  %grp.n12010 = phi i64 [ 0, %4 ], [ %grp.n1146, %grp.n128 ]
  %grp.n131 = shl nuw nsw i64 %grp.n12010, 14, !dbg !13
  %grp.n132 = add nuw nsw i64 %grp.n131, %grp.n130, !dbg !13
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %grp.n25) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !8
  %address.local501 = and i64 %address.local500, 1536, !dbg !8
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !8
  %address.local505 = lshr i64 %address.local499, 4, !dbg !13
  %address.local506 = and i64 %address.local505, 8, !dbg !13
  %address.local507 = and i64 %address.local499, 123, !dbg !13
  %address.local508 = lshr i64 %address.local499, 2, !dbg !13
  %address.local509 = and i64 %address.local508, 4, !dbg !13
  %address.local510 = add nuw nsw i64 %address.local509, %address.local499, !dbg !13
  %address.local511 = and i64 %address.local510, 4, !dbg !13
  %address.local512 = or disjoint i64 %address.local511, %address.local507, !dbg !13
  %address.local513 = shl nuw nsw i64 %address.local512, 7, !dbg !13
  %address.local514 = or disjoint i64 %address.local513, %address.local506, !dbg !13
  %.idx1406 = or disjoint i64 %address.local514, %grp.n132, !dbg !13
  %.idx1406.tr = trunc nuw nsw i64 %.idx1406 to i32, !dbg !13
  %dma.buffer.offset1372 = shl nuw nsw i32 %.idx1406.tr, 1, !dbg !13
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local522 = lshr i64 %address.local516, 4, !dbg !13
  %address.local523 = and i64 %address.local522, 24, !dbg !13
  %address.local531 = or disjoint i64 %address.local513, %address.local523, !dbg !13
  %.idx1407 = or disjoint i64 %address.local531, %grp.n132, !dbg !13
  %.idx1407.tr = trunc nuw nsw i64 %.idx1407 to i32, !dbg !13
  %dma.buffer.offset1369 = shl nuw nsw i32 %.idx1407.tr, 1, !dbg !13
  %grp.n143 = or disjoint i64 %grp.n132, 32, !dbg !13
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %.idx1408 = or disjoint i64 %address.local514, %grp.n143, !dbg !13
  %.idx1408.tr = trunc nuw nsw i64 %.idx1408 to i32, !dbg !13
  %dma.buffer.offset1366 = shl nuw nsw i32 %.idx1408.tr, 1, !dbg !13
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %19 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local502, i64 0, !dbg !8
  %20 = insertelement <4 x ptr addrspace(3)> %19, ptr addrspace(3) %address.local519, i64 1, !dbg !8
  %21 = insertelement <4 x ptr addrspace(3)> %20, ptr addrspace(3) %address.local533, i64 2, !dbg !8
  %22 = insertelement <4 x ptr addrspace(3)> %21, ptr addrspace(3) %address.local537, i64 3, !dbg !8
  %23 = ptrtoint <4 x ptr addrspace(3)> %22 to <4 x i64>, !dbg !8
  %24 = trunc <4 x i64> %23 to <4 x i32>, !dbg !8
  %25 = extractelement <4 x i32> %24, i64 0, !dbg !13
  %grp.n133 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %25), !dbg !13
  %dma.buffer.lds1374 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n133), !dbg !13
  %grp.n137 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1372, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %26 = extractelement <4 x i32> %24, i64 1, !dbg !13
  %grp.n138 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %26), !dbg !13
  %dma.buffer.lds1371 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n138), !dbg !13
  %grp.n142 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1369, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %27 = extractelement <4 x i32> %24, i64 2, !dbg !13
  %grp.n144 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %27), !dbg !13
  %dma.buffer.lds1368 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n144), !dbg !13
  %grp.n148 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1366, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %28 = extractelement <4 x i32> %24, i64 3, !dbg !13
  %grp.n149 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %28), !dbg !13
  %.idx1409 = or disjoint i64 %address.local531, %grp.n143, !dbg !13
  %.idx1409.tr = trunc nuw nsw i64 %.idx1409 to i32, !dbg !13
  %dma.buffer.offset1363 = shl nuw nsw i32 %.idx1409.tr, 1, !dbg !13
  %dma.buffer.lds1365 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %grp.n149), !dbg !13
  %grp.n153 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1363, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !12
  %address.local546 = and i64 %address.local545, 24, !dbg !12
  %address.local548 = shl i64 %address.local499, 62, !dbg !12
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !12
  %address.local550 = and i64 %address.local549, -64, !dbg !12
  %address.local553 = and i64 %address.local508, 12, !dbg !10
  %address.local555 = and i64 %address.local499, 63, !dbg !10
  %address.local556 = lshr i64 %address.local555, 5, !dbg !10
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !12
  %grp.n154 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !12
  %.idx1410 = mul nsw i64 %address.local508, -1280, !dbg !12
  %grp.n155 = getelementptr i8, ptr addrspace(3) %grp.n154, i64 %.idx1410, !dbg !12
  %grp.n156 = getelementptr [2 x i8], ptr addrspace(3) %grp.n155, i64 %address.local546, !dbg !12
  %grp.n157 = getelementptr [2 x i8], ptr addrspace(3) %grp.n156, i64 %address.local550, !dbg !12
  %grp.n158 = getelementptr [2 x i8], ptr addrspace(3) %grp.n157, i64 %address.local553, !dbg !12
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %grp.n158, i64 %address.local557, !dbg !12
  %grp.n159 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !12
  %grp.n160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n159, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1210 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !12
  %grp.n161 = load <4 x half>, ptr addrspace(3) %lds.const.gep1210, align 8, !dbg !12
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !12
  %address.local571 = and i32 %address.local570, 24, !dbg !12
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !12
  %address.local573 = and i32 %address.local572, 768, !dbg !12
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !12
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !12
  %address.local576 = and i32 %address.local575, 96, !dbg !12
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !12
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !12
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %grp.n118) #8, !dbg !12, !frisk.register_lifetimes !5
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !12
  %address.local580 = and i32 %address.local579, 1, !dbg !12
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !12
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !12
  %address.local583 = shl i64 %address.local582, 6, !dbg !12
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !12
  %grp.n162 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !12
  %grp.n163 = getelementptr [2 x i8], ptr addrspace(3) %grp.n162, i64 %address.local553, !dbg !12
  %grp.n164 = getelementptr [2 x i8], ptr addrspace(3) %grp.n163, i64 %address.local557, !dbg !12
  %address.local588 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 64, !dbg !12
  %grp.n165 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !12
  %lds.const.gep1214 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 192, !dbg !12
  %grp.n173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !12
  %lds.const.gep1211 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 4160, !dbg !12
  %grp.n168 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !12
  %lds.const.gep1215 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 4288, !dbg !12
  %grp.n176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n161, <4 x float> %grp.n160) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %lds.const.gep1212 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !12
  %grp.n169 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !12
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !12
  %grp.n177 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n165, <4 x float> zeroinitializer) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %lds.const.gep1213 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !12
  %grp.n172 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !12
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !12
  %grp.n185 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n168, <4 x float> %grp.n167) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n169, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n172, <4 x float> %grp.n171) #8, !dbg !12
  %grp.n175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n173, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !12
  %grp.n180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %lds.const.gep1221 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !12
  %grp.n188 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n176, <4 x float> %grp.n175) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 1088, !dbg !12
  %grp.n181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %lds.const.gep1222 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 1216, !dbg !12
  %grp.n189 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n177, <4 x float> zeroinitializer) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 5184, !dbg !12
  %grp.n184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %lds.const.gep1223 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 5312, !dbg !12
  %grp.n192 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n180, <4 x float> %grp.n179) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n181, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n184, <4 x float> %grp.n183) #8, !dbg !12
  %grp.n187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n185, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n188, <4 x float> %grp.n187) #8, !dbg !12
  %grp.n191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n53, <4 x half> %grp.n189, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n55, <4 x half> %grp.n192, <4 x float> %grp.n191) #8, !dbg !12
  %grp.n195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n159, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n161, <4 x float> %grp.n195) #8, !dbg !12
  %grp.n199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n165, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n168, <4 x float> %grp.n199) #8, !dbg !12
  %grp.n203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n169, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n172, <4 x float> %grp.n203) #8, !dbg !12
  %grp.n207 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n173, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n176, <4 x float> %grp.n207) #8, !dbg !12
  %grp.n211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n177, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n180, <4 x float> %grp.n211) #8, !dbg !12
  %grp.n215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n181, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n184, <4 x float> %grp.n215) #8, !dbg !12
  %grp.n219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n185, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n188, <4 x float> %grp.n219) #8, !dbg !12
  %grp.n223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n57, <4 x half> %grp.n189, <4 x float> zeroinitializer) #8, !dbg !12
  %grp.n225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n58, <4 x half> %grp.n192, <4 x float> %grp.n223) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %grp.n226 = or disjoint i64 %grp.n132, 64, !dbg !13
  %.idx14111412 = or disjoint i64 %address.local514, %grp.n226, !dbg !13
  %.idx14111412.tr = trunc nuw nsw i64 %.idx14111412 to i32, !dbg !13
  %dma.buffer.offset1360 = shl nuw nsw i32 %.idx14111412.tr, 1, !dbg !13
  %grp.n231 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1360, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %.idx14131414 = or disjoint i64 %address.local531, %grp.n226, !dbg !13
  %.idx14131414.tr = trunc nuw nsw i64 %.idx14131414 to i32, !dbg !13
  %dma.buffer.offset1357 = shl nuw nsw i32 %.idx14131414.tr, 1, !dbg !13
  %grp.n236 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1357, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %lds.const.gep1238 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !12
  %grp.n237 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !12
  %lds.const.gep1242 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !12
  %grp.n244 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !12
  %lds.const.gep1239 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !12
  %grp.n239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !12
  %lds.const.gep1243 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !12
  %grp.n247 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !12
  %lds.const.gep1240 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 8256, !dbg !12
  %grp.n240 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !12
  %lds.const.gep1244 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 8384, !dbg !12
  %grp.n248 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n237, <4 x float> %grp.n166) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %lds.const.gep1241 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 12352, !dbg !12
  %grp.n243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !12
  %lds.const.gep1245 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 12480, !dbg !12
  %grp.n251 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n239, <4 x float> %grp.n238) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n240, <4 x float> %grp.n170) #8, !dbg !12
  %grp.n245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n243, <4 x float> %grp.n242) #8, !dbg !12
  %grp.n246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n244, <4 x float> %grp.n174) #8, !dbg !12
  %grp.n249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n247, <4 x float> %grp.n246) #8, !dbg !12
  %lds.const.gep1246 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !12
  %grp.n252 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !12
  %lds.const.gep1250 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !12
  %grp.n260 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n248, <4 x float> %grp.n178) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %lds.const.gep1247 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !12
  %grp.n255 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !12
  %lds.const.gep1251 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !12
  %grp.n263 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n251, <4 x float> %grp.n250) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %lds.const.gep1248 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 9280, !dbg !12
  %grp.n256 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !12
  %lds.const.gep1252 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 9408, !dbg !12
  %grp.n264 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n252, <4 x float> %grp.n182) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %lds.const.gep1249 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 13376, !dbg !12
  %grp.n259 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !12
  %lds.const.gep1253 = getelementptr i8, ptr addrspace(3) %grp.n164, i64 13504, !dbg !12
  %grp.n267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n255, <4 x float> %grp.n254) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n256, <4 x float> %grp.n186) #8, !dbg !12
  %grp.n261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n259, <4 x float> %grp.n258) #8, !dbg !12
  %grp.n262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n260, <4 x float> %grp.n190) #8, !dbg !12
  %grp.n265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n263, <4 x float> %grp.n262) #8, !dbg !12
  %grp.n266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n76, <4 x half> %grp.n264, <4 x float> %grp.n194) #8, !dbg !12
  %grp.n269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n78, <4 x half> %grp.n267, <4 x float> %grp.n266) #8, !dbg !12
  %grp.n270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n237, <4 x float> %grp.n198) #8, !dbg !12
  %grp.n273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n239, <4 x float> %grp.n270) #8, !dbg !12
  %grp.n274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n240, <4 x float> %grp.n202) #8, !dbg !12
  %grp.n277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n243, <4 x float> %grp.n274) #8, !dbg !12
  %grp.n278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n244, <4 x float> %grp.n206) #8, !dbg !12
  %grp.n281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n247, <4 x float> %grp.n278) #8, !dbg !12
  %grp.n282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n248, <4 x float> %grp.n210) #8, !dbg !12
  %grp.n285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n251, <4 x float> %grp.n282) #8, !dbg !12
  %grp.n286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n252, <4 x float> %grp.n214) #8, !dbg !12
  %grp.n289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n255, <4 x float> %grp.n286) #8, !dbg !12
  %grp.n290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n256, <4 x float> %grp.n218) #8, !dbg !12
  %grp.n293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n259, <4 x float> %grp.n290) #8, !dbg !12
  %grp.n294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n260, <4 x float> %grp.n222) #8, !dbg !12
  %grp.n297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n263, <4 x float> %grp.n294) #8, !dbg !12
  %grp.n298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n79, <4 x half> %grp.n264, <4 x float> %grp.n225) #8, !dbg !12
  %grp.n300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n80, <4 x half> %grp.n267, <4 x float> %grp.n298) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %grp.n301 = or disjoint i64 %grp.n132, 96, !dbg !13
  %.idx14151416 = or disjoint i64 %address.local514, %grp.n301, !dbg !13
  %.idx14151416.tr = trunc nuw nsw i64 %.idx14151416 to i32, !dbg !13
  %dma.buffer.offset1354 = shl nuw nsw i32 %.idx14151416.tr, 1, !dbg !13
  %grp.n306 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1354, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %.idx14171418 = or disjoint i64 %address.local531, %grp.n301, !dbg !13
  %.idx14171418.tr = trunc nuw nsw i64 %.idx14171418 to i32, !dbg !13
  %dma.buffer.offset1351 = shl nuw nsw i32 %.idx14171418.tr, 1, !dbg !13
  %grp.n311 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1351, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %grp.n312 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !12
  %grp.n319 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !12
  %grp.n314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1210, align 8, !dbg !12
  %grp.n322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !12
  %grp.n315 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !12
  %grp.n323 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n312, <4 x float> %grp.n241) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !12
  %grp.n326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n314, <4 x float> %grp.n313) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n315, <4 x float> %grp.n245) #8, !dbg !12
  %grp.n320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n318, <4 x float> %grp.n317) #8, !dbg !12
  %grp.n321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n319, <4 x float> %grp.n249) #8, !dbg !12
  %grp.n324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n322, <4 x float> %grp.n321) #8, !dbg !12
  %grp.n327 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  %grp.n335 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n323, <4 x float> %grp.n253) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %grp.n338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n326, <4 x float> %grp.n325) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n331 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %grp.n339 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n327, <4 x float> %grp.n257) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %grp.n342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n330, <4 x float> %grp.n329) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n331, <4 x float> %grp.n261) #8, !dbg !12
  %grp.n336 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n334, <4 x float> %grp.n333) #8, !dbg !12
  %grp.n337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n335, <4 x float> %grp.n265) #8, !dbg !12
  %grp.n340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n338, <4 x float> %grp.n337) #8, !dbg !12
  %grp.n341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n98, <4 x half> %grp.n339, <4 x float> %grp.n269) #8, !dbg !12
  %grp.n344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n100, <4 x half> %grp.n342, <4 x float> %grp.n341) #8, !dbg !12
  %grp.n345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n312, <4 x float> %grp.n273) #8, !dbg !12
  %grp.n348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n314, <4 x float> %grp.n345) #8, !dbg !12
  %grp.n349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n315, <4 x float> %grp.n277) #8, !dbg !12
  %grp.n352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n318, <4 x float> %grp.n349) #8, !dbg !12
  %grp.n353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n319, <4 x float> %grp.n281) #8, !dbg !12
  %grp.n356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n322, <4 x float> %grp.n353) #8, !dbg !12
  %grp.n357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n323, <4 x float> %grp.n285) #8, !dbg !12
  %grp.n360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n326, <4 x float> %grp.n357) #8, !dbg !12
  %grp.n361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n327, <4 x float> %grp.n289) #8, !dbg !12
  %grp.n364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n330, <4 x float> %grp.n361) #8, !dbg !12
  %grp.n365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n331, <4 x float> %grp.n293) #8, !dbg !12
  %grp.n368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n334, <4 x float> %grp.n365) #8, !dbg !12
  %grp.n369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n335, <4 x float> %grp.n297) #8, !dbg !12
  %grp.n372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n338, <4 x float> %grp.n369) #8, !dbg !12
  %grp.n373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n101, <4 x half> %grp.n339, <4 x float> %grp.n300) #8, !dbg !12
  %grp.n375 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n102, <4 x half> %grp.n342, <4 x float> %grp.n373) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !14
  %address.local918 = and i64 %address.local917, 8064, !dbg !14
  %address.local920 = and i64 %address.local500, 24, !dbg !14
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !14
  %.idx14191420 = or disjoint i64 %address.local921, %grp.n132, !dbg !14
  %.idx14191420.tr = trunc nuw nsw i64 %.idx14191420 to i32, !dbg !14
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !14
  %grp.n380 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !14
  %address.local924 = and i64 %address.local923, 16256, !dbg !14
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !14
  %.idx14211422 = or disjoint i64 %address.local927, %grp.n132, !dbg !14
  %.idx14211422.tr = trunc nuw nsw i64 %.idx14211422 to i32, !dbg !14
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !14
  %grp.n385 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %grp.n386 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !12
  %grp.n387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n386, <4 x float> %grp.n316) #8, !dbg !12
  %grp.n388 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !12
  %grp.n389 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !12
  %grp.n390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n388, <4 x float> %grp.n387) #8, !dbg !12
  %grp.n391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n389, <4 x float> %grp.n320) #8, !dbg !12
  %grp.n392 = extractelement <4 x float> %grp.n390, i64 0, !dbg !15
  %grp.n393 = tail call float @llvm.maxnum.f32(float %grp.n392, float -inf), !dbg !16, !frisk.local_reduction !5
  %grp.n394 = extractelement <4 x float> %grp.n390, i64 1, !dbg !15
  %grp.n395 = extractelement <4 x float> %grp.n390, i64 2, !dbg !15
  %grp.n396 = extractelement <4 x float> %grp.n390, i64 3, !dbg !15
  %grp.n397 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !12
  %grp.n398 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !12
  %grp.n399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n397, <4 x float> %grp.n391) #8, !dbg !12
  %grp.n400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n398, <4 x float> %grp.n324) #8, !dbg !12
  %grp.n401 = extractelement <4 x float> %grp.n399, i64 0, !dbg !15
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %grp.n393, float %grp.n401), !dbg !16, !frisk.local_reduction !5
  %grp.n402 = extractelement <4 x float> %grp.n399, i64 1, !dbg !15
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %grp.n394, float %grp.n402), !dbg !16, !frisk.local_reduction !5
  %grp.n403 = extractelement <4 x float> %grp.n399, i64 2, !dbg !15
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %grp.n395, float %grp.n403), !dbg !16, !frisk.local_reduction !5
  %grp.n404 = extractelement <4 x float> %grp.n399, i64 3, !dbg !15
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %grp.n396, float %grp.n404), !dbg !16, !frisk.local_reduction !5
  %grp.n405 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !12
  %grp.n406 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !12
  %grp.n407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n405, <4 x float> %grp.n400) #8, !dbg !12
  %grp.n408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n406, <4 x float> %grp.n328) #8, !dbg !12
  %grp.n409 = extractelement <4 x float> %grp.n407, i64 0, !dbg !15
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %grp.n409), !dbg !16, !frisk.local_reduction !5
  %grp.n410 = extractelement <4 x float> %grp.n407, i64 1, !dbg !15
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %grp.n410), !dbg !16, !frisk.local_reduction !5
  %grp.n411 = extractelement <4 x float> %grp.n407, i64 2, !dbg !15
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %grp.n411), !dbg !16, !frisk.local_reduction !5
  %grp.n412 = extractelement <4 x float> %grp.n407, i64 3, !dbg !15
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %grp.n412), !dbg !16, !frisk.local_reduction !5
  %grp.n413 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !12
  %grp.n414 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !12
  %grp.n415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n413, <4 x float> %grp.n408) #8, !dbg !12
  %grp.n416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n414, <4 x float> %grp.n332) #8, !dbg !12
  %grp.n417 = extractelement <4 x float> %grp.n415, i64 0, !dbg !15
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %grp.n417), !dbg !16, !frisk.local_reduction !5
  %grp.n418 = extractelement <4 x float> %grp.n415, i64 1, !dbg !15
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %grp.n418), !dbg !16, !frisk.local_reduction !5
  %grp.n419 = extractelement <4 x float> %grp.n415, i64 2, !dbg !15
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %grp.n419), !dbg !16, !frisk.local_reduction !5
  %grp.n420 = extractelement <4 x float> %grp.n415, i64 3, !dbg !15
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %grp.n420), !dbg !16, !frisk.local_reduction !5
  %grp.n421 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !12
  %grp.n422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !12
  %grp.n423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n421, <4 x float> %grp.n416) #8, !dbg !12
  %grp.n424 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n422, <4 x float> %grp.n336) #8, !dbg !12
  %grp.n425 = extractelement <4 x float> %grp.n423, i64 0, !dbg !15
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %grp.n425), !dbg !16, !frisk.local_reduction !5
  %grp.n426 = extractelement <4 x float> %grp.n423, i64 1, !dbg !15
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %grp.n426), !dbg !16, !frisk.local_reduction !5
  %grp.n427 = extractelement <4 x float> %grp.n423, i64 2, !dbg !15
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %grp.n427), !dbg !16, !frisk.local_reduction !5
  %grp.n428 = extractelement <4 x float> %grp.n423, i64 3, !dbg !15
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %grp.n428), !dbg !16, !frisk.local_reduction !5
  %grp.n429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !12
  %grp.n430 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !12
  %grp.n431 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n429, <4 x float> %grp.n424) #8, !dbg !12
  %grp.n432 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n430, <4 x float> %grp.n340) #8, !dbg !12
  %grp.n433 = extractelement <4 x float> %grp.n431, i64 0, !dbg !15
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %grp.n433), !dbg !16, !frisk.local_reduction !5
  %grp.n434 = extractelement <4 x float> %grp.n431, i64 1, !dbg !15
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %grp.n434), !dbg !16, !frisk.local_reduction !5
  %grp.n435 = extractelement <4 x float> %grp.n431, i64 2, !dbg !15
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %grp.n435), !dbg !16, !frisk.local_reduction !5
  %grp.n436 = extractelement <4 x float> %grp.n431, i64 3, !dbg !15
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %grp.n436), !dbg !16, !frisk.local_reduction !5
  %grp.n437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !12
  %grp.n438 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !12
  %grp.n439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n437, <4 x float> %grp.n432) #8, !dbg !12
  %grp.n440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n106, <4 x half> %grp.n438, <4 x float> %grp.n344) #8, !dbg !12
  %grp.n441 = extractelement <4 x float> %grp.n439, i64 0, !dbg !15
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %grp.n441), !dbg !16, !frisk.local_reduction !5
  %grp.n442 = extractelement <4 x float> %grp.n439, i64 1, !dbg !15
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %grp.n442), !dbg !16, !frisk.local_reduction !5
  %grp.n443 = extractelement <4 x float> %grp.n439, i64 2, !dbg !15
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %grp.n443), !dbg !16, !frisk.local_reduction !5
  %grp.n444 = extractelement <4 x float> %grp.n439, i64 3, !dbg !15
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %grp.n444), !dbg !16, !frisk.local_reduction !5
  %grp.n445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !12
  %grp.n447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n108, <4 x half> %grp.n445, <4 x float> %grp.n440) #8, !dbg !12
  %grp.n448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n386, <4 x float> %grp.n348) #8, !dbg !12
  %grp.n449 = extractelement <4 x float> %grp.n447, i64 0, !dbg !15
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %grp.n449), !dbg !16, !frisk.local_reduction !5
  %grp.n450 = extractelement <4 x float> %grp.n447, i64 1, !dbg !15
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %grp.n450), !dbg !16, !frisk.local_reduction !5
  %grp.n451 = extractelement <4 x float> %grp.n447, i64 2, !dbg !15
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %grp.n451), !dbg !16, !frisk.local_reduction !5
  %grp.n452 = extractelement <4 x float> %grp.n447, i64 3, !dbg !15
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %grp.n452), !dbg !16, !frisk.local_reduction !5
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !16, !frisk.local_reduction !5
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !16, !frisk.local_reduction !5
  %grp.n453 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !16, !frisk.local_reduction !5
  %grp.n456 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n388, <4 x float> %grp.n448) #8, !dbg !12
  %grp.n457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n389, <4 x float> %grp.n352) #8, !dbg !12
  %grp.n458 = extractelement <4 x float> %grp.n456, i64 0, !dbg !15
  %grp.n459 = extractelement <4 x float> %grp.n456, i64 1, !dbg !15
  %grp.n460 = extractelement <4 x float> %grp.n456, i64 2, !dbg !15
  %grp.n461 = extractelement <4 x float> %grp.n456, i64 3, !dbg !15
  %grp.n464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n397, <4 x float> %grp.n457) #8, !dbg !12
  %grp.n465 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n398, <4 x float> %grp.n356) #8, !dbg !12
  %grp.n466 = extractelement <4 x float> %grp.n464, i64 0, !dbg !15
  %grp.n467 = extractelement <4 x float> %grp.n464, i64 1, !dbg !15
  %grp.n468 = extractelement <4 x float> %grp.n464, i64 2, !dbg !15
  %grp.n469 = extractelement <4 x float> %grp.n464, i64 3, !dbg !15
  %grp.n472 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n405, <4 x float> %grp.n465) #8, !dbg !12
  %grp.n473 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n406, <4 x float> %grp.n360) #8, !dbg !12
  %grp.n474 = extractelement <4 x float> %grp.n472, i64 0, !dbg !15
  %grp.n475 = extractelement <4 x float> %grp.n472, i64 1, !dbg !15
  %grp.n476 = extractelement <4 x float> %grp.n472, i64 2, !dbg !15
  %grp.n477 = extractelement <4 x float> %grp.n472, i64 3, !dbg !15
  %grp.n480 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n413, <4 x float> %grp.n473) #8, !dbg !12
  %grp.n481 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n414, <4 x float> %grp.n364) #8, !dbg !12
  %grp.n482 = extractelement <4 x float> %grp.n480, i64 0, !dbg !15
  %grp.n483 = extractelement <4 x float> %grp.n480, i64 1, !dbg !15
  %grp.n484 = extractelement <4 x float> %grp.n480, i64 2, !dbg !15
  %grp.n485 = extractelement <4 x float> %grp.n480, i64 3, !dbg !15
  %grp.n488 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n421, <4 x float> %grp.n481) #8, !dbg !12
  %grp.n489 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n422, <4 x float> %grp.n368) #8, !dbg !12
  %grp.n490 = extractelement <4 x float> %grp.n488, i64 0, !dbg !15
  %grp.n491 = extractelement <4 x float> %grp.n488, i64 1, !dbg !15
  %grp.n492 = extractelement <4 x float> %grp.n488, i64 2, !dbg !15
  %grp.n493 = extractelement <4 x float> %grp.n488, i64 3, !dbg !15
  %grp.n496 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n429, <4 x float> %grp.n489) #8, !dbg !12
  %grp.n497 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n430, <4 x float> %grp.n372) #8, !dbg !12
  %grp.n498 = extractelement <4 x float> %grp.n496, i64 0, !dbg !15
  %grp.n499 = extractelement <4 x float> %grp.n496, i64 1, !dbg !15
  %grp.n500 = extractelement <4 x float> %grp.n496, i64 2, !dbg !15
  %grp.n501 = extractelement <4 x float> %grp.n496, i64 3, !dbg !15
  %grp.n504 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n437, <4 x float> %grp.n497) #8, !dbg !12
  %grp.n505 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n109, <4 x half> %grp.n438, <4 x float> %grp.n375) #8, !dbg !12
  %grp.n506 = extractelement <4 x float> %grp.n504, i64 0, !dbg !15
  %grp.n507 = extractelement <4 x float> %grp.n504, i64 1, !dbg !15
  %grp.n508 = extractelement <4 x float> %grp.n504, i64 2, !dbg !15
  %grp.n509 = extractelement <4 x float> %grp.n504, i64 3, !dbg !15
  %grp.n511 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %grp.n110, <4 x half> %grp.n445, <4 x float> %grp.n505) #8, !dbg !12
  %grp.n512 = extractelement <4 x float> %grp.n511, i64 0, !dbg !15
  %grp.n513 = extractelement <4 x float> %grp.n511, i64 1, !dbg !15
  %grp.n514 = extractelement <4 x float> %grp.n511, i64 2, !dbg !15
  %grp.n515 = extractelement <4 x float> %grp.n511, i64 3, !dbg !15
  %grp.n516 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %grp.n517 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %grp.n516), !dbg !16
  %grp.n518 = add nuw nsw i32 %grp.n517, 64, !dbg !16
  %grp.n519 = xor i32 %grp.n517, 16, !dbg !16
  %grp.n520 = and i32 %grp.n518, 192, !dbg !16
  %grp.n521 = icmp samesign ult i32 %grp.n519, %grp.n520, !dbg !16
  %grp.n522 = select i1 %grp.n521, i32 %grp.n519, i32 %grp.n517, !dbg !16
  %grp.n523 = shl nuw nsw i32 %grp.n522, 2, !dbg !16
  %grp.n524 = bitcast float %grp.n453 to i32, !dbg !16
  %grp.n525 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n524), !dbg !16
  %grp.n526 = bitcast i32 %grp.n525 to float, !dbg !16
  %grp.n527 = tail call float @llvm.maxnum.f32(float %grp.n453, float %grp.n526), !dbg !16
  %grp.n528 = xor i32 %grp.n517, 32, !dbg !16
  %grp.n529 = icmp samesign ult i32 %grp.n528, %grp.n520, !dbg !16
  %grp.n530 = select i1 %grp.n529, i32 %grp.n528, i32 %grp.n517, !dbg !16
  %grp.n531 = shl nuw nsw i32 %grp.n530, 2, !dbg !16
  %grp.n532 = bitcast float %grp.n527 to i32, !dbg !16
  %grp.n533 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n532), !dbg !16
  %grp.n534 = bitcast i32 %grp.n533 to float, !dbg !16
  %grp.n535 = tail call float @llvm.maxnum.f32(float %grp.n527, float %grp.n534), !dbg !16
  %grp.n536 = extractvalue [2 x <1 x float>] %grp.n12313, 0, !dbg !16
  %grp.n537 = insertelement <1 x float> poison, float %grp.n535, i64 0, !dbg !16
  %grp.n538 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %grp.n537, <1 x float> %grp.n536), !dbg !16
  %grp.n539 = tail call float @llvm.maxnum.f32(float %grp.n458, float -inf), !dbg !16, !frisk.local_reduction !5
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %grp.n539, float %grp.n466), !dbg !16, !frisk.local_reduction !5
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %grp.n459, float %grp.n467), !dbg !16, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %grp.n460, float %grp.n468), !dbg !16, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %grp.n461, float %grp.n469), !dbg !16, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %grp.n474), !dbg !16, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %grp.n475), !dbg !16, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %grp.n476), !dbg !16, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %grp.n477), !dbg !16, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %grp.n482), !dbg !16, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %grp.n483), !dbg !16, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %grp.n484), !dbg !16, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %grp.n485), !dbg !16, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %grp.n490), !dbg !16, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %grp.n491), !dbg !16, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %grp.n492), !dbg !16, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %grp.n493), !dbg !16, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %grp.n498), !dbg !16, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %grp.n499), !dbg !16, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %grp.n500), !dbg !16, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %grp.n501), !dbg !16, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %grp.n506), !dbg !16, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %grp.n507), !dbg !16, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %grp.n508), !dbg !16, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %grp.n509), !dbg !16, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %grp.n512), !dbg !16, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %grp.n513), !dbg !16, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %grp.n514), !dbg !16, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %grp.n515), !dbg !16, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !16, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !16, !frisk.local_reduction !5
  %grp.n540 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !16, !frisk.local_reduction !5
  %grp.n541 = bitcast float %grp.n540 to i32, !dbg !16
  %grp.n542 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n541), !dbg !16
  %grp.n543 = bitcast i32 %grp.n542 to float, !dbg !16
  %grp.n544 = tail call float @llvm.maxnum.f32(float %grp.n540, float %grp.n543), !dbg !16
  %grp.n545 = bitcast float %grp.n544 to i32, !dbg !16
  %grp.n546 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n545), !dbg !16
  %grp.n547 = bitcast i32 %grp.n546 to float, !dbg !16
  %grp.n548 = tail call float @llvm.maxnum.f32(float %grp.n544, float %grp.n547), !dbg !16
  %grp.n549 = extractvalue [2 x <1 x float>] %grp.n12313, 1, !dbg !16
  %grp.n550 = insertelement <1 x float> poison, float %grp.n548, i64 0, !dbg !16
  %grp.n551 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %grp.n550, <1 x float> %grp.n549), !dbg !16
  %grp.n552 = shufflevector <1 x float> %grp.n538, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %grp.n553 = fsub <4 x float> %grp.n390, %grp.n552, !dbg !17
  %grp.n554 = fsub <4 x float> %grp.n399, %grp.n552, !dbg !17
  %grp.n555 = fsub <4 x float> %grp.n407, %grp.n552, !dbg !17
  %grp.n556 = fsub <4 x float> %grp.n415, %grp.n552, !dbg !17
  %grp.n557 = fsub <4 x float> %grp.n423, %grp.n552, !dbg !17
  %grp.n558 = fsub <4 x float> %grp.n431, %grp.n552, !dbg !17
  %grp.n559 = fsub <4 x float> %grp.n439, %grp.n552, !dbg !17
  %grp.n560 = fsub <4 x float> %grp.n447, %grp.n552, !dbg !17
  %grp.n561 = shufflevector <1 x float> %grp.n551, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %grp.n562 = fsub <4 x float> %grp.n456, %grp.n561, !dbg !17
  %grp.n563 = fsub <4 x float> %grp.n464, %grp.n561, !dbg !17
  %grp.n564 = fsub <4 x float> %grp.n472, %grp.n561, !dbg !17
  %grp.n565 = fsub <4 x float> %grp.n480, %grp.n561, !dbg !17
  %grp.n566 = fsub <4 x float> %grp.n488, %grp.n561, !dbg !17
  %grp.n567 = fsub <4 x float> %grp.n496, %grp.n561, !dbg !17
  %grp.n568 = fsub <4 x float> %grp.n504, %grp.n561, !dbg !17
  %grp.n569 = fsub <4 x float> %grp.n511, %grp.n561, !dbg !17
  %grp.n570 = fsub <1 x float> %grp.n536, %grp.n538, !dbg !18
  %grp.n571 = extractelement <1 x float> %grp.n570, i64 0, !dbg !19
  %grp.n572 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n571), !dbg !19
  %grp.n573 = insertelement <4 x float> poison, float %grp.n572, i64 0, !dbg !20
  %grp.n574 = shufflevector <4 x float> %grp.n573, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %grp.n575 = extractvalue [2 x <1 x float>] %grp.n12212, 0, !dbg !21
  %grp.n576 = extractelement <1 x float> %grp.n575, i64 0, !dbg !21
  %grp.n577 = fmul float %grp.n576, %grp.n572, !dbg !21
  %grp.n578 = fsub <1 x float> %grp.n549, %grp.n551, !dbg !18
  %grp.n579 = extractelement <1 x float> %grp.n578, i64 0, !dbg !19
  %grp.n580 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n579), !dbg !19
  %grp.n581 = insertelement <4 x float> poison, float %grp.n580, i64 0, !dbg !20
  %grp.n582 = shufflevector <4 x float> %grp.n581, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %grp.n583 = extractvalue [2 x <1 x float>] %grp.n12212, 1, !dbg !21
  %grp.n584 = extractelement <1 x float> %grp.n583, i64 0, !dbg !21
  %grp.n585 = fmul float %grp.n584, %grp.n580, !dbg !21
  %grp.n586 = extractelement <4 x float> %grp.n553, i64 0, !dbg !22
  %grp.n587 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n586), !dbg !22
  %grp.n588 = insertelement <4 x float> poison, float %grp.n587, i64 0, !dbg !23
  %grp.n589 = extractelement <4 x float> %grp.n553, i64 1, !dbg !22
  %grp.n590 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n589), !dbg !22
  %grp.n591 = insertelement <4 x float> %grp.n588, float %grp.n590, i64 1, !dbg !23
  %grp.n592 = extractelement <4 x float> %grp.n553, i64 2, !dbg !22
  %grp.n593 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n592), !dbg !22
  %grp.n594 = insertelement <4 x float> %grp.n591, float %grp.n593, i64 2, !dbg !23
  %grp.n595 = extractelement <4 x float> %grp.n553, i64 3, !dbg !22
  %grp.n596 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n595), !dbg !22
  %grp.n597 = insertelement <4 x float> %grp.n594, float %grp.n596, i64 3, !dbg !23
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n597) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n598 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !23
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n598) #8, !dbg !22, !frisk.register_lifetimes !5
  %grp.n599 = extractelement <4 x float> %grp.n554, i64 0, !dbg !22
  %grp.n600 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n599), !dbg !22
  %grp.n601 = insertelement <4 x float> poison, float %grp.n600, i64 0, !dbg !23
  %local.reduce89 = fadd float %grp.n587, %grp.n600, !dbg !24, !frisk.local_reduction !5
  %grp.n602 = extractelement <4 x float> %grp.n554, i64 1, !dbg !22
  %grp.n603 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n602), !dbg !22
  %grp.n604 = insertelement <4 x float> %grp.n601, float %grp.n603, i64 1, !dbg !23
  %local.reduce90 = fadd float %grp.n590, %grp.n603, !dbg !24, !frisk.local_reduction !5
  %grp.n605 = extractelement <4 x float> %grp.n554, i64 2, !dbg !22
  %grp.n606 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n605), !dbg !22
  %grp.n607 = insertelement <4 x float> %grp.n604, float %grp.n606, i64 2, !dbg !23
  %local.reduce91 = fadd float %grp.n593, %grp.n606, !dbg !24, !frisk.local_reduction !5
  %grp.n608 = extractelement <4 x float> %grp.n554, i64 3, !dbg !22
  %grp.n609 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n608), !dbg !22
  %grp.n610 = insertelement <4 x float> %grp.n607, float %grp.n609, i64 3, !dbg !23
  %math.result374 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n610) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n611 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !23
  %math.packed375 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n611) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce92 = fadd float %grp.n596, %grp.n609, !dbg !24, !frisk.local_reduction !5
  %grp.n612 = extractelement <4 x float> %grp.n555, i64 0, !dbg !22
  %grp.n613 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n612), !dbg !22
  %grp.n614 = insertelement <4 x float> poison, float %grp.n613, i64 0, !dbg !23
  %local.reduce93 = fadd float %grp.n613, %local.reduce89, !dbg !24, !frisk.local_reduction !5
  %grp.n615 = extractelement <4 x float> %grp.n555, i64 1, !dbg !22
  %grp.n616 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n615), !dbg !22
  %grp.n617 = insertelement <4 x float> %grp.n614, float %grp.n616, i64 1, !dbg !23
  %local.reduce94 = fadd float %grp.n616, %local.reduce90, !dbg !24, !frisk.local_reduction !5
  %grp.n618 = extractelement <4 x float> %grp.n555, i64 2, !dbg !22
  %grp.n619 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n618), !dbg !22
  %grp.n620 = insertelement <4 x float> %grp.n617, float %grp.n619, i64 2, !dbg !23
  %local.reduce95 = fadd float %grp.n619, %local.reduce91, !dbg !24, !frisk.local_reduction !5
  %grp.n621 = extractelement <4 x float> %grp.n555, i64 3, !dbg !22
  %grp.n622 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n621), !dbg !22
  %grp.n623 = insertelement <4 x float> %grp.n620, float %grp.n622, i64 3, !dbg !23
  %math.result376 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n623) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n624 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !23
  %math.packed377 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n624) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce96 = fadd float %grp.n622, %local.reduce92, !dbg !24, !frisk.local_reduction !5
  %grp.n625 = extractelement <4 x float> %grp.n556, i64 0, !dbg !22
  %grp.n626 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n625), !dbg !22
  %grp.n627 = insertelement <4 x float> poison, float %grp.n626, i64 0, !dbg !23
  %local.reduce97 = fadd float %grp.n626, %local.reduce93, !dbg !24, !frisk.local_reduction !5
  %grp.n628 = extractelement <4 x float> %grp.n556, i64 1, !dbg !22
  %grp.n629 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n628), !dbg !22
  %grp.n630 = insertelement <4 x float> %grp.n627, float %grp.n629, i64 1, !dbg !23
  %local.reduce98 = fadd float %grp.n629, %local.reduce94, !dbg !24, !frisk.local_reduction !5
  %grp.n631 = extractelement <4 x float> %grp.n556, i64 2, !dbg !22
  %grp.n632 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n631), !dbg !22
  %grp.n633 = insertelement <4 x float> %grp.n630, float %grp.n632, i64 2, !dbg !23
  %local.reduce99 = fadd float %grp.n632, %local.reduce95, !dbg !24, !frisk.local_reduction !5
  %grp.n634 = extractelement <4 x float> %grp.n556, i64 3, !dbg !22
  %grp.n635 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n634), !dbg !22
  %grp.n636 = insertelement <4 x float> %grp.n633, float %grp.n635, i64 3, !dbg !23
  %math.result378 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n636) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n637 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !23
  %math.packed379 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n637) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce100 = fadd float %grp.n635, %local.reduce96, !dbg !24, !frisk.local_reduction !5
  %grp.n638 = extractelement <4 x float> %grp.n557, i64 0, !dbg !22
  %grp.n639 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n638), !dbg !22
  %grp.n640 = insertelement <4 x float> poison, float %grp.n639, i64 0, !dbg !23
  %local.reduce101 = fadd float %grp.n639, %local.reduce97, !dbg !24, !frisk.local_reduction !5
  %grp.n641 = extractelement <4 x float> %grp.n557, i64 1, !dbg !22
  %grp.n642 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n641), !dbg !22
  %grp.n643 = insertelement <4 x float> %grp.n640, float %grp.n642, i64 1, !dbg !23
  %local.reduce102 = fadd float %grp.n642, %local.reduce98, !dbg !24, !frisk.local_reduction !5
  %grp.n644 = extractelement <4 x float> %grp.n557, i64 2, !dbg !22
  %grp.n645 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n644), !dbg !22
  %grp.n646 = insertelement <4 x float> %grp.n643, float %grp.n645, i64 2, !dbg !23
  %local.reduce103 = fadd float %grp.n645, %local.reduce99, !dbg !24, !frisk.local_reduction !5
  %grp.n647 = extractelement <4 x float> %grp.n557, i64 3, !dbg !22
  %grp.n648 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n647), !dbg !22
  %grp.n649 = insertelement <4 x float> %grp.n646, float %grp.n648, i64 3, !dbg !23
  %math.result380 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n649) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n650 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !23
  %math.packed381 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n650) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce104 = fadd float %grp.n648, %local.reduce100, !dbg !24, !frisk.local_reduction !5
  %grp.n651 = extractelement <4 x float> %grp.n558, i64 0, !dbg !22
  %grp.n652 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n651), !dbg !22
  %grp.n653 = insertelement <4 x float> poison, float %grp.n652, i64 0, !dbg !23
  %local.reduce105 = fadd float %grp.n652, %local.reduce101, !dbg !24, !frisk.local_reduction !5
  %grp.n654 = extractelement <4 x float> %grp.n558, i64 1, !dbg !22
  %grp.n655 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n654), !dbg !22
  %grp.n656 = insertelement <4 x float> %grp.n653, float %grp.n655, i64 1, !dbg !23
  %local.reduce106 = fadd float %grp.n655, %local.reduce102, !dbg !24, !frisk.local_reduction !5
  %grp.n657 = extractelement <4 x float> %grp.n558, i64 2, !dbg !22
  %grp.n658 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n657), !dbg !22
  %grp.n659 = insertelement <4 x float> %grp.n656, float %grp.n658, i64 2, !dbg !23
  %local.reduce107 = fadd float %grp.n658, %local.reduce103, !dbg !24, !frisk.local_reduction !5
  %grp.n660 = extractelement <4 x float> %grp.n558, i64 3, !dbg !22
  %grp.n661 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n660), !dbg !22
  %grp.n662 = insertelement <4 x float> %grp.n659, float %grp.n661, i64 3, !dbg !23
  %math.result382 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n662) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n663 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !23
  %math.packed383 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n663) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce108 = fadd float %grp.n661, %local.reduce104, !dbg !24, !frisk.local_reduction !5
  %grp.n664 = extractelement <4 x float> %grp.n559, i64 0, !dbg !22
  %grp.n665 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n664), !dbg !22
  %grp.n666 = insertelement <4 x float> poison, float %grp.n665, i64 0, !dbg !23
  %local.reduce109 = fadd float %grp.n665, %local.reduce105, !dbg !24, !frisk.local_reduction !5
  %grp.n667 = extractelement <4 x float> %grp.n559, i64 1, !dbg !22
  %grp.n668 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n667), !dbg !22
  %grp.n669 = insertelement <4 x float> %grp.n666, float %grp.n668, i64 1, !dbg !23
  %local.reduce110 = fadd float %grp.n668, %local.reduce106, !dbg !24, !frisk.local_reduction !5
  %grp.n670 = extractelement <4 x float> %grp.n559, i64 2, !dbg !22
  %grp.n671 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n670), !dbg !22
  %grp.n672 = insertelement <4 x float> %grp.n669, float %grp.n671, i64 2, !dbg !23
  %local.reduce111 = fadd float %grp.n671, %local.reduce107, !dbg !24, !frisk.local_reduction !5
  %grp.n673 = extractelement <4 x float> %grp.n559, i64 3, !dbg !22
  %grp.n674 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n673), !dbg !22
  %grp.n675 = insertelement <4 x float> %grp.n672, float %grp.n674, i64 3, !dbg !23
  %math.result384 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n675) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n676 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !23
  %math.packed385 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n676) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce112 = fadd float %grp.n674, %local.reduce108, !dbg !24, !frisk.local_reduction !5
  %grp.n677 = extractelement <4 x float> %grp.n560, i64 0, !dbg !22
  %grp.n678 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n677), !dbg !22
  %grp.n679 = insertelement <4 x float> poison, float %grp.n678, i64 0, !dbg !23
  %local.reduce113 = fadd float %grp.n678, %local.reduce109, !dbg !24, !frisk.local_reduction !5
  %grp.n680 = extractelement <4 x float> %grp.n560, i64 1, !dbg !22
  %grp.n681 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n680), !dbg !22
  %grp.n682 = insertelement <4 x float> %grp.n679, float %grp.n681, i64 1, !dbg !23
  %local.reduce114 = fadd float %grp.n681, %local.reduce110, !dbg !24, !frisk.local_reduction !5
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !24, !frisk.local_reduction !5
  %grp.n683 = extractelement <4 x float> %grp.n560, i64 2, !dbg !22
  %grp.n684 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n683), !dbg !22
  %grp.n685 = insertelement <4 x float> %grp.n682, float %grp.n684, i64 2, !dbg !23
  %local.reduce115 = fadd float %grp.n684, %local.reduce111, !dbg !24, !frisk.local_reduction !5
  %grp.n686 = extractelement <4 x float> %grp.n560, i64 3, !dbg !22
  %grp.n687 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n686), !dbg !22
  %grp.n688 = insertelement <4 x float> %grp.n685, float %grp.n687, i64 3, !dbg !23
  %math.result386 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n688) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n689 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !23
  %math.packed387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n689) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce116 = fadd float %grp.n687, %local.reduce112, !dbg !24, !frisk.local_reduction !5
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !24, !frisk.local_reduction !5
  %grp.n690 = fadd float %local.reduce117, %local.reduce118, !dbg !24, !frisk.local_reduction !5
  %grp.n691 = fadd float %grp.n577, %grp.n690, !dbg !24
  %grp.n692 = insertelement <1 x float> poison, float %grp.n691, i64 0, !dbg !11
  %grp.n693 = insertvalue [2 x <1 x float>] poison, <1 x float> %grp.n692, 0, !dbg !11
  %grp.n694 = extractelement <4 x float> %grp.n562, i64 0, !dbg !22
  %grp.n695 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n694), !dbg !22
  %grp.n696 = insertelement <4 x float> poison, float %grp.n695, i64 0, !dbg !23
  %grp.n697 = extractelement <4 x float> %grp.n562, i64 1, !dbg !22
  %grp.n698 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n697), !dbg !22
  %grp.n699 = insertelement <4 x float> %grp.n696, float %grp.n698, i64 1, !dbg !23
  %grp.n700 = extractelement <4 x float> %grp.n562, i64 2, !dbg !22
  %grp.n701 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n700), !dbg !22
  %grp.n702 = insertelement <4 x float> %grp.n699, float %grp.n701, i64 2, !dbg !23
  %grp.n703 = extractelement <4 x float> %grp.n562, i64 3, !dbg !22
  %grp.n704 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n703), !dbg !22
  %grp.n705 = insertelement <4 x float> %grp.n702, float %grp.n704, i64 3, !dbg !23
  %math.result388 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n705) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n706 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !23
  %math.packed389 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n706) #8, !dbg !22, !frisk.register_lifetimes !5
  %grp.n707 = extractelement <4 x float> %grp.n563, i64 0, !dbg !22
  %grp.n708 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n707), !dbg !22
  %grp.n709 = insertelement <4 x float> poison, float %grp.n708, i64 0, !dbg !23
  %local.reduce = fadd float %grp.n695, %grp.n708, !dbg !24, !frisk.local_reduction !5
  %grp.n710 = extractelement <4 x float> %grp.n563, i64 1, !dbg !22
  %grp.n711 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n710), !dbg !22
  %grp.n712 = insertelement <4 x float> %grp.n709, float %grp.n711, i64 1, !dbg !23
  %local.reduce59 = fadd float %grp.n698, %grp.n711, !dbg !24, !frisk.local_reduction !5
  %grp.n713 = extractelement <4 x float> %grp.n563, i64 2, !dbg !22
  %grp.n714 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n713), !dbg !22
  %grp.n715 = insertelement <4 x float> %grp.n712, float %grp.n714, i64 2, !dbg !23
  %local.reduce60 = fadd float %grp.n701, %grp.n714, !dbg !24, !frisk.local_reduction !5
  %grp.n716 = extractelement <4 x float> %grp.n563, i64 3, !dbg !22
  %grp.n717 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n716), !dbg !22
  %grp.n718 = insertelement <4 x float> %grp.n715, float %grp.n717, i64 3, !dbg !23
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n718) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n719 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !23
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n719) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce61 = fadd float %grp.n704, %grp.n717, !dbg !24, !frisk.local_reduction !5
  %grp.n720 = extractelement <4 x float> %grp.n564, i64 0, !dbg !22
  %grp.n721 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n720), !dbg !22
  %grp.n722 = insertelement <4 x float> poison, float %grp.n721, i64 0, !dbg !23
  %local.reduce62 = fadd float %grp.n721, %local.reduce, !dbg !24, !frisk.local_reduction !5
  %grp.n723 = extractelement <4 x float> %grp.n564, i64 1, !dbg !22
  %grp.n724 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n723), !dbg !22
  %grp.n725 = insertelement <4 x float> %grp.n722, float %grp.n724, i64 1, !dbg !23
  %local.reduce63 = fadd float %grp.n724, %local.reduce59, !dbg !24, !frisk.local_reduction !5
  %grp.n726 = extractelement <4 x float> %grp.n564, i64 2, !dbg !22
  %grp.n727 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n726), !dbg !22
  %grp.n728 = insertelement <4 x float> %grp.n725, float %grp.n727, i64 2, !dbg !23
  %local.reduce64 = fadd float %grp.n727, %local.reduce60, !dbg !24, !frisk.local_reduction !5
  %grp.n729 = extractelement <4 x float> %grp.n564, i64 3, !dbg !22
  %grp.n730 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n729), !dbg !22
  %grp.n731 = insertelement <4 x float> %grp.n728, float %grp.n730, i64 3, !dbg !23
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n731) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n732 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !23
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n732) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce65 = fadd float %grp.n730, %local.reduce61, !dbg !24, !frisk.local_reduction !5
  %grp.n733 = extractelement <4 x float> %grp.n565, i64 0, !dbg !22
  %grp.n734 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n733), !dbg !22
  %grp.n735 = insertelement <4 x float> poison, float %grp.n734, i64 0, !dbg !23
  %local.reduce66 = fadd float %grp.n734, %local.reduce62, !dbg !24, !frisk.local_reduction !5
  %grp.n736 = extractelement <4 x float> %grp.n565, i64 1, !dbg !22
  %grp.n737 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n736), !dbg !22
  %grp.n738 = insertelement <4 x float> %grp.n735, float %grp.n737, i64 1, !dbg !23
  %local.reduce67 = fadd float %grp.n737, %local.reduce63, !dbg !24, !frisk.local_reduction !5
  %grp.n739 = extractelement <4 x float> %grp.n565, i64 2, !dbg !22
  %grp.n740 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n739), !dbg !22
  %grp.n741 = insertelement <4 x float> %grp.n738, float %grp.n740, i64 2, !dbg !23
  %local.reduce68 = fadd float %grp.n740, %local.reduce64, !dbg !24, !frisk.local_reduction !5
  %grp.n742 = extractelement <4 x float> %grp.n565, i64 3, !dbg !22
  %grp.n743 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n742), !dbg !22
  %grp.n744 = insertelement <4 x float> %grp.n741, float %grp.n743, i64 3, !dbg !23
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n744) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n745 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !23
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n745) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce69 = fadd float %grp.n743, %local.reduce65, !dbg !24, !frisk.local_reduction !5
  %grp.n746 = extractelement <4 x float> %grp.n566, i64 0, !dbg !22
  %grp.n747 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n746), !dbg !22
  %grp.n748 = insertelement <4 x float> poison, float %grp.n747, i64 0, !dbg !23
  %local.reduce70 = fadd float %grp.n747, %local.reduce66, !dbg !24, !frisk.local_reduction !5
  %grp.n749 = extractelement <4 x float> %grp.n566, i64 1, !dbg !22
  %grp.n750 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n749), !dbg !22
  %grp.n751 = insertelement <4 x float> %grp.n748, float %grp.n750, i64 1, !dbg !23
  %local.reduce71 = fadd float %grp.n750, %local.reduce67, !dbg !24, !frisk.local_reduction !5
  %grp.n752 = extractelement <4 x float> %grp.n566, i64 2, !dbg !22
  %grp.n753 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n752), !dbg !22
  %grp.n754 = insertelement <4 x float> %grp.n751, float %grp.n753, i64 2, !dbg !23
  %local.reduce72 = fadd float %grp.n753, %local.reduce68, !dbg !24, !frisk.local_reduction !5
  %grp.n755 = extractelement <4 x float> %grp.n566, i64 3, !dbg !22
  %grp.n756 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n755), !dbg !22
  %grp.n757 = insertelement <4 x float> %grp.n754, float %grp.n756, i64 3, !dbg !23
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n757) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n758 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !23
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n758) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce73 = fadd float %grp.n756, %local.reduce69, !dbg !24, !frisk.local_reduction !5
  %grp.n759 = extractelement <4 x float> %grp.n567, i64 0, !dbg !22
  %grp.n760 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n759), !dbg !22
  %grp.n761 = insertelement <4 x float> poison, float %grp.n760, i64 0, !dbg !23
  %local.reduce74 = fadd float %grp.n760, %local.reduce70, !dbg !24, !frisk.local_reduction !5
  %grp.n762 = extractelement <4 x float> %grp.n567, i64 1, !dbg !22
  %grp.n763 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n762), !dbg !22
  %grp.n764 = insertelement <4 x float> %grp.n761, float %grp.n763, i64 1, !dbg !23
  %local.reduce75 = fadd float %grp.n763, %local.reduce71, !dbg !24, !frisk.local_reduction !5
  %grp.n765 = extractelement <4 x float> %grp.n567, i64 2, !dbg !22
  %grp.n766 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n765), !dbg !22
  %grp.n767 = insertelement <4 x float> %grp.n764, float %grp.n766, i64 2, !dbg !23
  %local.reduce76 = fadd float %grp.n766, %local.reduce72, !dbg !24, !frisk.local_reduction !5
  %grp.n768 = extractelement <4 x float> %grp.n567, i64 3, !dbg !22
  %grp.n769 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n768), !dbg !22
  %grp.n770 = insertelement <4 x float> %grp.n767, float %grp.n769, i64 3, !dbg !23
  %math.result398 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n770) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n771 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !23
  %math.packed399 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n771) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce77 = fadd float %grp.n769, %local.reduce73, !dbg !24, !frisk.local_reduction !5
  %grp.n772 = extractelement <4 x float> %grp.n568, i64 0, !dbg !22
  %grp.n773 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n772), !dbg !22
  %grp.n774 = insertelement <4 x float> poison, float %grp.n773, i64 0, !dbg !23
  %local.reduce78 = fadd float %grp.n773, %local.reduce74, !dbg !24, !frisk.local_reduction !5
  %grp.n775 = extractelement <4 x float> %grp.n568, i64 1, !dbg !22
  %grp.n776 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n775), !dbg !22
  %grp.n777 = insertelement <4 x float> %grp.n774, float %grp.n776, i64 1, !dbg !23
  %local.reduce79 = fadd float %grp.n776, %local.reduce75, !dbg !24, !frisk.local_reduction !5
  %grp.n778 = extractelement <4 x float> %grp.n568, i64 2, !dbg !22
  %grp.n779 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n778), !dbg !22
  %grp.n780 = insertelement <4 x float> %grp.n777, float %grp.n779, i64 2, !dbg !23
  %local.reduce80 = fadd float %grp.n779, %local.reduce76, !dbg !24, !frisk.local_reduction !5
  %grp.n781 = extractelement <4 x float> %grp.n568, i64 3, !dbg !22
  %grp.n782 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n781), !dbg !22
  %grp.n783 = insertelement <4 x float> %grp.n780, float %grp.n782, i64 3, !dbg !23
  %math.result400 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n783) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n784 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !23
  %math.packed401 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n784) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce81 = fadd float %grp.n782, %local.reduce77, !dbg !24, !frisk.local_reduction !5
  %grp.n785 = extractelement <4 x float> %grp.n569, i64 0, !dbg !22
  %grp.n786 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n785), !dbg !22
  %grp.n787 = insertelement <4 x float> poison, float %grp.n786, i64 0, !dbg !23
  %local.reduce82 = fadd float %grp.n786, %local.reduce78, !dbg !24, !frisk.local_reduction !5
  %grp.n788 = extractelement <4 x float> %grp.n569, i64 1, !dbg !22
  %grp.n789 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n788), !dbg !22
  %grp.n790 = insertelement <4 x float> %grp.n787, float %grp.n789, i64 1, !dbg !23
  %local.reduce83 = fadd float %grp.n789, %local.reduce79, !dbg !24, !frisk.local_reduction !5
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !24, !frisk.local_reduction !5
  %grp.n791 = extractelement <4 x float> %grp.n569, i64 2, !dbg !22
  %grp.n792 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n791), !dbg !22
  %grp.n793 = insertelement <4 x float> %grp.n790, float %grp.n792, i64 2, !dbg !23
  %local.reduce84 = fadd float %grp.n792, %local.reduce80, !dbg !24, !frisk.local_reduction !5
  %grp.n794 = extractelement <4 x float> %grp.n569, i64 3, !dbg !22
  %grp.n795 = tail call float @llvm.amdgcn.exp2.f32(float %grp.n794), !dbg !22
  %grp.n796 = insertelement <4 x float> %grp.n793, float %grp.n795, i64 3, !dbg !23
  %math.result402 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %grp.n796) #8, !dbg !23, !frisk.register_lifetimes !5
  %grp.n797 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !23
  %math.packed403 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %grp.n797) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce85 = fadd float %grp.n795, %local.reduce81, !dbg !24, !frisk.local_reduction !5
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !24, !frisk.local_reduction !5
  %grp.n798 = fadd float %local.reduce86, %local.reduce87, !dbg !24, !frisk.local_reduction !5
  %grp.n799 = fadd float %grp.n585, %grp.n798, !dbg !24
  %grp.n800 = insertelement <1 x float> poison, float %grp.n799, i64 0, !dbg !11
  %grp.n801 = insertvalue [2 x <1 x float>] %grp.n693, <1 x float> %grp.n800, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !25
  %address.local930 = and i32 %address.local929, 120, !dbg !25
  %address.local932 = and i32 %address.local575, 1536, !dbg !25
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !25
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !25
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !25
  %grp.n802 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11, !dbg !25
  %address.local936 = and i64 %address.local499, 60, !dbg !25
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !25
  %address.local939.idx = shl nuw nsw i64 %address.local937, 4, !dbg !25
  %address.local939 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local939.idx, !dbg !25
  %address.local940 = lshr i64 %address.local555, 4, !dbg !25
  %address.local942.idx = mul nuw nsw i64 %address.local940, 768, !dbg !25
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local942.idx, !dbg !25
  %address.local944.idx = mul nsw i64 %address.local508, -64, !dbg !25
  %address.local944 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local944.idx, !dbg !25
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !25
  %grp.n804 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !25
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !25
  %grp.n806 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !25
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !25
  %grp.n808 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !25
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !25
  %grp.n810 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11, !dbg !25
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !25
  %grp.n812 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !25
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !25
  %grp.n814 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !25
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !25
  %grp.n816 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !25
  %grp.n803 = bitcast <4 x i32> %grp.n802 to <8 x half>, !dbg !25
  %grp.n828 = shufflevector <8 x half> %grp.n803, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %grp.n829 = extractvalue [2 x <8 x float>] %grp.n12111, 0, !dbg !20
  %grp.n830 = shufflevector <8 x float> %grp.n829, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %grp.n831 = fmul <4 x float> %grp.n830, %grp.n574, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n828, <4 x float> %grp.n831) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n805 = bitcast <4 x i32> %grp.n804 to <8 x half>, !dbg !25
  %grp.n833 = shufflevector <8 x half> %grp.n805, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n833, <4 x float> %grp.n832) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx14231424 = or disjoint i64 %address.local921, %grp.n143, !dbg !14
  %.idx14231424.tr = trunc nuw nsw i64 %.idx14231424 to i32, !dbg !14
  %dma.buffer.offset1342 = shl nuw nsw i32 %.idx14231424.tr, 1, !dbg !14
  %grp.n822 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1342, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %.idx14251426 = or disjoint i64 %address.local927, %grp.n143, !dbg !14
  %.idx14251426.tr = trunc nuw nsw i64 %.idx14251426 to i32, !dbg !14
  %dma.buffer.offset1339 = shl nuw nsw i32 %.idx14251426.tr, 1, !dbg !14
  %grp.n827 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1339, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %grp.n807 = bitcast <4 x i32> %grp.n806 to <8 x half>, !dbg !25
  %grp.n835 = shufflevector <8 x half> %grp.n807, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n835, <4 x float> %grp.n834) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n809 = bitcast <4 x i32> %grp.n808 to <8 x half>, !dbg !25
  %grp.n837 = shufflevector <8 x half> %grp.n809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n837, <4 x float> %grp.n836) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n811 = bitcast <4 x i32> %grp.n810 to <8 x half>, !dbg !25
  %grp.n839 = shufflevector <8 x half> %grp.n811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n839, <4 x float> %grp.n838) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n813 = bitcast <4 x i32> %grp.n812 to <8 x half>, !dbg !25
  %grp.n841 = shufflevector <8 x half> %grp.n813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n841, <4 x float> %grp.n840) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n815 = bitcast <4 x i32> %grp.n814 to <8 x half>, !dbg !25
  %grp.n843 = shufflevector <8 x half> %grp.n815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n843, <4 x float> %grp.n842) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n817 = bitcast <4 x i32> %grp.n816 to <8 x half>, !dbg !25
  %grp.n845 = shufflevector <8 x half> %grp.n817, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n845, <4 x float> %grp.n844) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n847 = shufflevector <8 x half> %grp.n803, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n848 = shufflevector <8 x float> %grp.n829, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %grp.n849 = fmul <4 x float> %grp.n848, %grp.n574, !dbg !20
  %grp.n850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n847, <4 x float> %grp.n849) #8, !dbg !25
  %grp.n851 = shufflevector <8 x half> %grp.n805, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n851, <4 x float> %grp.n850) #8, !dbg !25
  %grp.n853 = shufflevector <8 x half> %grp.n807, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n853, <4 x float> %grp.n852) #8, !dbg !25
  %grp.n855 = shufflevector <8 x half> %grp.n809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n855, <4 x float> %grp.n854) #8, !dbg !25
  %grp.n857 = shufflevector <8 x half> %grp.n811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n858 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n857, <4 x float> %grp.n856) #8, !dbg !25
  %grp.n859 = shufflevector <8 x half> %grp.n813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n860 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n859, <4 x float> %grp.n858) #8, !dbg !25
  %grp.n861 = shufflevector <8 x half> %grp.n815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n862 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n861, <4 x float> %grp.n860) #8, !dbg !25
  %grp.n863 = shufflevector <8 x half> %grp.n817, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n864 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n863, <4 x float> %grp.n862) #8, !dbg !25
  %grp.n865 = extractvalue [2 x <8 x float>] %grp.n12111, 1, !dbg !20
  %grp.n866 = shufflevector <8 x float> %grp.n865, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %grp.n867 = fmul <4 x float> %grp.n866, %grp.n582, !dbg !20
  %grp.n868 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n828, <4 x float> %grp.n867) #8, !dbg !25
  %grp.n869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n833, <4 x float> %grp.n868) #8, !dbg !25
  %grp.n870 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n835, <4 x float> %grp.n869) #8, !dbg !25
  %grp.n871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n837, <4 x float> %grp.n870) #8, !dbg !25
  %grp.n872 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n839, <4 x float> %grp.n871) #8, !dbg !25
  %grp.n873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n841, <4 x float> %grp.n872) #8, !dbg !25
  %grp.n874 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n843, <4 x float> %grp.n873) #8, !dbg !25
  %grp.n875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n845, <4 x float> %grp.n874) #8, !dbg !25
  %grp.n876 = shufflevector <8 x float> %grp.n865, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %grp.n877 = fmul <4 x float> %grp.n876, %grp.n582, !dbg !20
  %grp.n878 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n847, <4 x float> %grp.n877) #8, !dbg !25
  %grp.n879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n851, <4 x float> %grp.n878) #8, !dbg !25
  %grp.n880 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n853, <4 x float> %grp.n879) #8, !dbg !25
  %grp.n881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n855, <4 x float> %grp.n880) #8, !dbg !25
  %grp.n882 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n857, <4 x float> %grp.n881) #8, !dbg !25
  %grp.n883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n859, <4 x float> %grp.n882) #8, !dbg !25
  %grp.n884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n861, <4 x float> %grp.n883) #8, !dbg !25
  %grp.n885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n863, <4 x float> %grp.n884) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %grp.n886 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11, !dbg !25
  %grp.n888 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !25
  %grp.n890 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !25
  %grp.n892 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !25
  %grp.n894 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11, !dbg !25
  %grp.n896 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !25
  %grp.n898 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !25
  %grp.n900 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !25
  %grp.n887 = bitcast <4 x i32> %grp.n886 to <8 x half>, !dbg !25
  %grp.n912 = shufflevector <8 x half> %grp.n887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %grp.n913 = extractvalue [2 x <8 x float>] %grp.n12414, 0, !dbg !20
  %grp.n914 = shufflevector <8 x float> %grp.n913, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %grp.n915 = fmul <4 x float> %grp.n914, %grp.n574, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n912, <4 x float> %grp.n915) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n889 = bitcast <4 x i32> %grp.n888 to <8 x half>, !dbg !25
  %grp.n917 = shufflevector <8 x half> %grp.n889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n917, <4 x float> %grp.n916) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx14271428 = or disjoint i64 %address.local921, %grp.n226, !dbg !14
  %.idx14271428.tr = trunc nuw nsw i64 %.idx14271428 to i32, !dbg !14
  %dma.buffer.offset1336 = shl nuw nsw i32 %.idx14271428.tr, 1, !dbg !14
  %grp.n906 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1336, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %.idx14291430 = or disjoint i64 %address.local927, %grp.n226, !dbg !14
  %.idx14291430.tr = trunc nuw nsw i64 %.idx14291430 to i32, !dbg !14
  %dma.buffer.offset1333 = shl nuw nsw i32 %.idx14291430.tr, 1, !dbg !14
  %grp.n911 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1333, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %grp.n891 = bitcast <4 x i32> %grp.n890 to <8 x half>, !dbg !25
  %grp.n919 = shufflevector <8 x half> %grp.n891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n919, <4 x float> %grp.n918) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n893 = bitcast <4 x i32> %grp.n892 to <8 x half>, !dbg !25
  %grp.n921 = shufflevector <8 x half> %grp.n893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n921, <4 x float> %grp.n920) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n895 = bitcast <4 x i32> %grp.n894 to <8 x half>, !dbg !25
  %grp.n923 = shufflevector <8 x half> %grp.n895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n923, <4 x float> %grp.n922) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n897 = bitcast <4 x i32> %grp.n896 to <8 x half>, !dbg !25
  %grp.n925 = shufflevector <8 x half> %grp.n897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n925, <4 x float> %grp.n924) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n899 = bitcast <4 x i32> %grp.n898 to <8 x half>, !dbg !25
  %grp.n927 = shufflevector <8 x half> %grp.n899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n927, <4 x float> %grp.n926) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n901 = bitcast <4 x i32> %grp.n900 to <8 x half>, !dbg !25
  %grp.n929 = shufflevector <8 x half> %grp.n901, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n929, <4 x float> %grp.n928) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n931 = shufflevector <8 x half> %grp.n887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n932 = shufflevector <8 x float> %grp.n913, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %grp.n933 = fmul <4 x float> %grp.n932, %grp.n574, !dbg !20
  %grp.n934 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n931, <4 x float> %grp.n933) #8, !dbg !25
  %grp.n935 = shufflevector <8 x half> %grp.n889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n935, <4 x float> %grp.n934) #8, !dbg !25
  %grp.n937 = shufflevector <8 x half> %grp.n891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n938 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n937, <4 x float> %grp.n936) #8, !dbg !25
  %grp.n939 = shufflevector <8 x half> %grp.n893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n940 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n939, <4 x float> %grp.n938) #8, !dbg !25
  %grp.n941 = shufflevector <8 x half> %grp.n895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n942 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n941, <4 x float> %grp.n940) #8, !dbg !25
  %grp.n943 = shufflevector <8 x half> %grp.n897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n944 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n943, <4 x float> %grp.n942) #8, !dbg !25
  %grp.n945 = shufflevector <8 x half> %grp.n899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n946 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n945, <4 x float> %grp.n944) #8, !dbg !25
  %grp.n947 = shufflevector <8 x half> %grp.n901, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n948 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n947, <4 x float> %grp.n946) #8, !dbg !25
  %grp.n949 = extractvalue [2 x <8 x float>] %grp.n12414, 1, !dbg !20
  %grp.n950 = shufflevector <8 x float> %grp.n949, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %grp.n951 = fmul <4 x float> %grp.n950, %grp.n582, !dbg !20
  %grp.n952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n912, <4 x float> %grp.n951) #8, !dbg !25
  %grp.n953 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n917, <4 x float> %grp.n952) #8, !dbg !25
  %grp.n954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n919, <4 x float> %grp.n953) #8, !dbg !25
  %grp.n955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n921, <4 x float> %grp.n954) #8, !dbg !25
  %grp.n956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n923, <4 x float> %grp.n955) #8, !dbg !25
  %grp.n957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n925, <4 x float> %grp.n956) #8, !dbg !25
  %grp.n958 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n927, <4 x float> %grp.n957) #8, !dbg !25
  %grp.n959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n929, <4 x float> %grp.n958) #8, !dbg !25
  %grp.n960 = shufflevector <8 x float> %grp.n949, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %grp.n961 = fmul <4 x float> %grp.n960, %grp.n582, !dbg !20
  %grp.n962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n931, <4 x float> %grp.n961) #8, !dbg !25
  %grp.n963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n935, <4 x float> %grp.n962) #8, !dbg !25
  %grp.n964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n937, <4 x float> %grp.n963) #8, !dbg !25
  %grp.n965 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n939, <4 x float> %grp.n964) #8, !dbg !25
  %grp.n966 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n941, <4 x float> %grp.n965) #8, !dbg !25
  %grp.n967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n943, <4 x float> %grp.n966) #8, !dbg !25
  %grp.n968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n945, <4 x float> %grp.n967) #8, !dbg !25
  %grp.n969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n947, <4 x float> %grp.n968) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %grp.n970 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11, !dbg !25
  %grp.n972 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !25
  %grp.n974 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !25
  %grp.n976 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !25
  %grp.n978 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11, !dbg !25
  %grp.n980 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !25
  %grp.n982 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !25
  %grp.n984 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !25
  %grp.n971 = bitcast <4 x i32> %grp.n970 to <8 x half>, !dbg !25
  %grp.n996 = shufflevector <8 x half> %grp.n971, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %grp.n997 = extractvalue [2 x <8 x float>] %grp.n12515, 0, !dbg !20
  %grp.n998 = shufflevector <8 x float> %grp.n997, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %grp.n999 = fmul <4 x float> %grp.n998, %grp.n574, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n996, <4 x float> %grp.n999) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n973 = bitcast <4 x i32> %grp.n972 to <8 x half>, !dbg !25
  %grp.n1001 = shufflevector <8 x half> %grp.n973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n1001, <4 x float> %grp.n1000) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx14311432 = or disjoint i64 %address.local921, %grp.n301, !dbg !14
  %.idx14311432.tr = trunc nuw nsw i64 %.idx14311432 to i32, !dbg !14
  %dma.buffer.offset1330 = shl nuw nsw i32 %.idx14311432.tr, 1, !dbg !14
  %grp.n990 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1330, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %.idx14331434 = or disjoint i64 %address.local927, %grp.n301, !dbg !14
  %.idx14331434.tr = trunc nuw nsw i64 %.idx14331434 to i32, !dbg !14
  %dma.buffer.offset = shl nuw nsw i32 %.idx14331434.tr, 1, !dbg !14
  %grp.n995 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %grp.n975 = bitcast <4 x i32> %grp.n974 to <8 x half>, !dbg !25
  %grp.n1003 = shufflevector <8 x half> %grp.n975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n1003, <4 x float> %grp.n1002) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n977 = bitcast <4 x i32> %grp.n976 to <8 x half>, !dbg !25
  %grp.n1005 = shufflevector <8 x half> %grp.n977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n1005, <4 x float> %grp.n1004) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n979 = bitcast <4 x i32> %grp.n978 to <8 x half>, !dbg !25
  %grp.n1007 = shufflevector <8 x half> %grp.n979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n1007, <4 x float> %grp.n1006) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n981 = bitcast <4 x i32> %grp.n980 to <8 x half>, !dbg !25
  %grp.n1009 = shufflevector <8 x half> %grp.n981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n1009, <4 x float> %grp.n1008) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n983 = bitcast <4 x i32> %grp.n982 to <8 x half>, !dbg !25
  %grp.n1011 = shufflevector <8 x half> %grp.n983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n1011, <4 x float> %grp.n1010) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n985 = bitcast <4 x i32> %grp.n984 to <8 x half>, !dbg !25
  %grp.n1013 = shufflevector <8 x half> %grp.n985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n1013, <4 x float> %grp.n1012) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1015 = shufflevector <8 x half> %grp.n971, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1016 = shufflevector <8 x float> %grp.n997, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %grp.n1017 = fmul <4 x float> %grp.n1016, %grp.n574, !dbg !20
  %grp.n1018 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n1015, <4 x float> %grp.n1017) #8, !dbg !25
  %grp.n1019 = shufflevector <8 x half> %grp.n973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n1019, <4 x float> %grp.n1018) #8, !dbg !25
  %grp.n1021 = shufflevector <8 x half> %grp.n975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1022 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n1021, <4 x float> %grp.n1020) #8, !dbg !25
  %grp.n1023 = shufflevector <8 x half> %grp.n977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n1023, <4 x float> %grp.n1022) #8, !dbg !25
  %grp.n1025 = shufflevector <8 x half> %grp.n979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n1025, <4 x float> %grp.n1024) #8, !dbg !25
  %grp.n1027 = shufflevector <8 x half> %grp.n981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1028 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n1027, <4 x float> %grp.n1026) #8, !dbg !25
  %grp.n1029 = shufflevector <8 x half> %grp.n983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n1029, <4 x float> %grp.n1028) #8, !dbg !25
  %grp.n1031 = shufflevector <8 x half> %grp.n985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1032 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n1031, <4 x float> %grp.n1030) #8, !dbg !25
  %grp.n1033 = extractvalue [2 x <8 x float>] %grp.n12515, 1, !dbg !20
  %grp.n1034 = shufflevector <8 x float> %grp.n1033, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %grp.n1035 = fmul <4 x float> %grp.n1034, %grp.n582, !dbg !20
  %grp.n1036 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n996, <4 x float> %grp.n1035) #8, !dbg !25
  %grp.n1037 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n1001, <4 x float> %grp.n1036) #8, !dbg !25
  %grp.n1038 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n1003, <4 x float> %grp.n1037) #8, !dbg !25
  %grp.n1039 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n1005, <4 x float> %grp.n1038) #8, !dbg !25
  %grp.n1040 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n1007, <4 x float> %grp.n1039) #8, !dbg !25
  %grp.n1041 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n1009, <4 x float> %grp.n1040) #8, !dbg !25
  %grp.n1042 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n1011, <4 x float> %grp.n1041) #8, !dbg !25
  %grp.n1043 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n1013, <4 x float> %grp.n1042) #8, !dbg !25
  %grp.n1044 = shufflevector <8 x float> %grp.n1033, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %grp.n1045 = fmul <4 x float> %grp.n1044, %grp.n582, !dbg !20
  %grp.n1046 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n1015, <4 x float> %grp.n1045) #8, !dbg !25
  %grp.n1047 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n1019, <4 x float> %grp.n1046) #8, !dbg !25
  %grp.n1048 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n1021, <4 x float> %grp.n1047) #8, !dbg !25
  %grp.n1049 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n1023, <4 x float> %grp.n1048) #8, !dbg !25
  %grp.n1050 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n1025, <4 x float> %grp.n1049) #8, !dbg !25
  %grp.n1051 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n1027, <4 x float> %grp.n1050) #8, !dbg !25
  %grp.n1052 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n1029, <4 x float> %grp.n1051) #8, !dbg !25
  %grp.n1053 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n1031, <4 x float> %grp.n1052) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %grp.n1054 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11, !dbg !25
  %grp.n1056 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !25
  %grp.n1058 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !25
  %grp.n1060 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !25
  %grp.n1062 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11, !dbg !25
  %grp.n1064 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !25
  %grp.n1066 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !25
  %grp.n1068 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !25
  %grp.n1055 = bitcast <4 x i32> %grp.n1054 to <8 x half>, !dbg !25
  %grp.n1070 = shufflevector <8 x half> %grp.n1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %grp.n1071 = extractvalue [2 x <8 x float>] %grp.n12616, 0, !dbg !20
  %grp.n1072 = shufflevector <8 x float> %grp.n1071, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %grp.n1073 = fmul <4 x float> %grp.n1072, %grp.n574, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1074 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n1070, <4 x float> %grp.n1073) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1057 = bitcast <4 x i32> %grp.n1056 to <8 x half>, !dbg !25
  %grp.n1075 = shufflevector <8 x half> %grp.n1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n1075, <4 x float> %grp.n1074) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1059 = bitcast <4 x i32> %grp.n1058 to <8 x half>, !dbg !25
  %grp.n1077 = shufflevector <8 x half> %grp.n1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n1077, <4 x float> %grp.n1076) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1061 = bitcast <4 x i32> %grp.n1060 to <8 x half>, !dbg !25
  %grp.n1079 = shufflevector <8 x half> %grp.n1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n1079, <4 x float> %grp.n1078) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1063 = bitcast <4 x i32> %grp.n1062 to <8 x half>, !dbg !25
  %grp.n1081 = shufflevector <8 x half> %grp.n1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n1081, <4 x float> %grp.n1080) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1065 = bitcast <4 x i32> %grp.n1064 to <8 x half>, !dbg !25
  %grp.n1083 = shufflevector <8 x half> %grp.n1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n1083, <4 x float> %grp.n1082) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1067 = bitcast <4 x i32> %grp.n1066 to <8 x half>, !dbg !25
  %grp.n1085 = shufflevector <8 x half> %grp.n1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n1085, <4 x float> %grp.n1084) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1069 = bitcast <4 x i32> %grp.n1068 to <8 x half>, !dbg !25
  %grp.n1087 = shufflevector <8 x half> %grp.n1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n1087, <4 x float> %grp.n1086) #8, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10
  %grp.n1089 = shufflevector <8 x half> %grp.n1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1090 = shufflevector <8 x float> %grp.n1071, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %grp.n1091 = fmul <4 x float> %grp.n1090, %grp.n574, !dbg !20
  %grp.n1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %grp.n1089, <4 x float> %grp.n1091) #8, !dbg !25
  %grp.n1093 = shufflevector <8 x half> %grp.n1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %grp.n1093, <4 x float> %grp.n1092) #8, !dbg !25
  %grp.n1095 = shufflevector <8 x half> %grp.n1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %grp.n1095, <4 x float> %grp.n1094) #8, !dbg !25
  %grp.n1097 = shufflevector <8 x half> %grp.n1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1098 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %grp.n1097, <4 x float> %grp.n1096) #8, !dbg !25
  %grp.n1099 = shufflevector <8 x half> %grp.n1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1100 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %grp.n1099, <4 x float> %grp.n1098) #8, !dbg !25
  %grp.n1101 = shufflevector <8 x half> %grp.n1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %grp.n1101, <4 x float> %grp.n1100) #8, !dbg !25
  %grp.n1103 = shufflevector <8 x half> %grp.n1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %grp.n1103, <4 x float> %grp.n1102) #8, !dbg !25
  %grp.n1105 = shufflevector <8 x half> %grp.n1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %grp.n1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %grp.n1105, <4 x float> %grp.n1104) #8, !dbg !25
  %grp.n1107 = extractvalue [2 x <8 x float>] %grp.n12616, 1, !dbg !20
  %grp.n1108 = shufflevector <8 x float> %grp.n1107, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %grp.n1109 = fmul <4 x float> %grp.n1108, %grp.n582, !dbg !20
  %grp.n1110 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n1070, <4 x float> %grp.n1109) #8, !dbg !25
  %grp.n1111 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n1075, <4 x float> %grp.n1110) #8, !dbg !25
  %grp.n1112 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n1077, <4 x float> %grp.n1111) #8, !dbg !25
  %grp.n1113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n1079, <4 x float> %grp.n1112) #8, !dbg !25
  %grp.n1114 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n1081, <4 x float> %grp.n1113) #8, !dbg !25
  %grp.n1115 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n1083, <4 x float> %grp.n1114) #8, !dbg !25
  %grp.n1116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n1085, <4 x float> %grp.n1115) #8, !dbg !25
  %grp.n1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n1087, <4 x float> %grp.n1116) #8, !dbg !25
  %grp.n1118 = shufflevector <8 x float> %grp.n1107, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %grp.n1119 = fmul <4 x float> %grp.n1118, %grp.n582, !dbg !20
  %grp.n1120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %grp.n1089, <4 x float> %grp.n1119) #8, !dbg !25
  %grp.n1121 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %grp.n1093, <4 x float> %grp.n1120) #8, !dbg !25
  %grp.n1122 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %grp.n1095, <4 x float> %grp.n1121) #8, !dbg !25
  %grp.n1123 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %grp.n1097, <4 x float> %grp.n1122) #8, !dbg !25
  %grp.n1124 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %grp.n1099, <4 x float> %grp.n1123) #8, !dbg !25
  %grp.n1125 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %grp.n1101, <4 x float> %grp.n1124) #8, !dbg !25
  %grp.n1126 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %grp.n1103, <4 x float> %grp.n1125) #8, !dbg !25
  %grp.n1127 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %grp.n1105, <4 x float> %grp.n1126) #8, !dbg !25
  %grp.n1128 = shufflevector <4 x float> %grp.n846, <4 x float> %grp.n864, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %grp.n1129 = insertvalue [2 x <8 x float>] poison, <8 x float> %grp.n1128, 0, !dbg !11
  %grp.n1130 = shufflevector <4 x float> %grp.n875, <4 x float> %grp.n885, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %grp.n1131 = insertvalue [2 x <8 x float>] %grp.n1129, <8 x float> %grp.n1130, 1, !dbg !11
  %grp.n1132 = insertvalue [2 x <1 x float>] poison, <1 x float> %grp.n538, 0, !dbg !11
  %grp.n1133 = insertvalue [2 x <1 x float>] %grp.n1132, <1 x float> %grp.n551, 1, !dbg !11
  %grp.n1134 = shufflevector <4 x float> %grp.n930, <4 x float> %grp.n948, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %grp.n1135 = insertvalue [2 x <8 x float>] poison, <8 x float> %grp.n1134, 0, !dbg !11
  %grp.n1136 = shufflevector <4 x float> %grp.n959, <4 x float> %grp.n969, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %grp.n1137 = insertvalue [2 x <8 x float>] %grp.n1135, <8 x float> %grp.n1136, 1, !dbg !11
  %grp.n1138 = shufflevector <4 x float> %grp.n1014, <4 x float> %grp.n1032, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %grp.n1139 = insertvalue [2 x <8 x float>] poison, <8 x float> %grp.n1138, 0, !dbg !11
  %grp.n1140 = shufflevector <4 x float> %grp.n1043, <4 x float> %grp.n1053, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %grp.n1141 = insertvalue [2 x <8 x float>] %grp.n1139, <8 x float> %grp.n1140, 1, !dbg !11
  %grp.n1142 = shufflevector <4 x float> %grp.n1088, <4 x float> %grp.n1106, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %grp.n1143 = insertvalue [2 x <8 x float>] poison, <8 x float> %grp.n1142, 0, !dbg !11
  %grp.n1144 = shufflevector <4 x float> %grp.n1117, <4 x float> %grp.n1127, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %grp.n1145 = insertvalue [2 x <8 x float>] %grp.n1143, <8 x float> %grp.n1144, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %grp.n1146 = add nuw nsw i64 %grp.n12010, 1, !dbg !11
  %grp.n127 = icmp samesign ult i64 %grp.n12010, 31, !dbg !11
  br i1 %grp.n127, label %grp.n128, label %grp.n1147, !dbg !11

grp.n1147:                                        ; preds = %grp.n128
  %grp.n1148 = shl nuw nsw i32 %grp.n517, 2, !dbg !11
  %grp.n1149 = xor i32 %grp.n1148, 64, !dbg !11
  %grp.n1150 = bitcast float %grp.n799 to i32, !dbg !11
  %lane.sum.shuffle188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n1149, i32 %grp.n1150), !dbg !11
  %grp.n1151 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !11
  %lane.sum.final189 = fadd float %grp.n799, %grp.n1151, !dbg !11
  %grp.n1153 = xor i32 %grp.n1148, 128, !dbg !11
  %grp.n1154 = bitcast float %lane.sum.final189 to i32, !dbg !11
  %lane.sum.shuffle190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n1153, i32 %grp.n1154), !dbg !11
  %grp.n1155 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !11
  %lane.sum.final191 = fadd float %lane.sum.final189, %grp.n1155, !dbg !11
  %grp.n1156 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !11
  %grp.n1159 = bitcast float %grp.n691 to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n1149, i32 %grp.n1159), !dbg !11
  %grp.n1160 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %grp.n691, %grp.n1160, !dbg !11
  %grp.n1163 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n1153, i32 %grp.n1163), !dbg !11
  %grp.n1164 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !11
  %lane.sum.final183 = fadd float %lane.sum.final, %grp.n1164, !dbg !11
  %grp.n1165 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !11
  %grp.n1166 = and i32 %grp.n25, 16, !dbg !11
  %.not = icmp eq i32 %grp.n1166, 0, !dbg !11
  %grp.n1167 = and i32 %grp.n25, 32, !dbg !11
  %.not36 = icmp eq i32 %grp.n1167, 0, !dbg !11
  %grp.n1169 = extractelement <4 x float> %grp.n1088, i64 0, !dbg !11
  %grp.n1170 = extractelement <4 x float> %grp.n1088, i64 1, !dbg !11
  %grp.n1171 = extractelement <4 x float> %grp.n1088, i64 2, !dbg !11
  %grp.n1172 = extractelement <4 x float> %grp.n1088, i64 3, !dbg !11
  %grp.n1173 = select i1 %.not, float %grp.n1170, float %grp.n1169, !dbg !11
  %grp.n1182 = bitcast float %grp.n1173 to i32, !dbg !11
  %grp.n1183 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1182), !dbg !11
  %grp.n1184 = bitcast i32 %grp.n1183 to float, !dbg !11
  %grp.n1185 = select i1 %.not, float %grp.n1169, float %grp.n1184, !dbg !11
  %grp.n1186 = select i1 %.not, float %grp.n1184, float %grp.n1170, !dbg !11
  %grp.n1187 = select i1 %.not, float %grp.n1172, float %grp.n1171, !dbg !11
  %grp.n1188 = bitcast float %grp.n1187 to i32, !dbg !11
  %grp.n1189 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1188), !dbg !11
  %grp.n1190 = bitcast i32 %grp.n1189 to float, !dbg !11
  %grp.n1191 = select i1 %.not, float %grp.n1171, float %grp.n1190, !dbg !11
  %grp.n1192 = select i1 %.not, float %grp.n1190, float %grp.n1172, !dbg !11
  %grp.n1193 = select i1 %.not36, float %grp.n1191, float %grp.n1185, !dbg !11
  %grp.n1198 = bitcast float %grp.n1193 to i32, !dbg !11
  %grp.n1199 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1198), !dbg !11
  %grp.n1200 = bitcast i32 %grp.n1199 to float, !dbg !11
  %grp.n1203 = select i1 %.not36, float %grp.n1192, float %grp.n1186, !dbg !11
  %grp.n1204 = bitcast float %grp.n1203 to i32, !dbg !11
  %grp.n1205 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1204), !dbg !11
  %grp.n1206 = bitcast i32 %grp.n1205 to float, !dbg !11
  %29 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %30 = shufflevector <4 x i1> %29, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %31 = insertelement <4 x float> poison, float %grp.n1185, i64 0, !dbg !11
  %32 = insertelement <4 x float> %31, float %grp.n1186, i64 1, !dbg !11
  %33 = insertelement <4 x float> %32, float %grp.n1200, i64 2, !dbg !11
  %34 = insertelement <4 x float> %33, float %grp.n1206, i64 3, !dbg !11
  %35 = insertelement <4 x float> poison, float %grp.n1200, i64 0, !dbg !11
  %36 = insertelement <4 x float> %35, float %grp.n1206, i64 1, !dbg !11
  %37 = insertelement <4 x float> %36, float %grp.n1191, i64 2, !dbg !11
  %38 = insertelement <4 x float> %37, float %grp.n1192, i64 3, !dbg !11
  %39 = select <4 x i1> %30, <4 x float> %34, <4 x float> %38, !dbg !11
  %grp.n1210 = extractelement <4 x float> %grp.n1014, i64 0, !dbg !11
  %grp.n1211 = extractelement <4 x float> %grp.n1014, i64 1, !dbg !11
  %grp.n1212 = extractelement <4 x float> %grp.n1014, i64 2, !dbg !11
  %grp.n1213 = extractelement <4 x float> %grp.n1014, i64 3, !dbg !11
  %grp.n1214 = select i1 %.not, float %grp.n1211, float %grp.n1210, !dbg !11
  %grp.n1215 = bitcast float %grp.n1214 to i32, !dbg !11
  %grp.n1216 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1215), !dbg !11
  %grp.n1217 = bitcast i32 %grp.n1216 to float, !dbg !11
  %grp.n1218 = select i1 %.not, float %grp.n1210, float %grp.n1217, !dbg !11
  %grp.n1219 = select i1 %.not, float %grp.n1217, float %grp.n1211, !dbg !11
  %grp.n1220 = select i1 %.not, float %grp.n1213, float %grp.n1212, !dbg !11
  %grp.n1221 = bitcast float %grp.n1220 to i32, !dbg !11
  %grp.n1222 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1221), !dbg !11
  %grp.n1223 = bitcast i32 %grp.n1222 to float, !dbg !11
  %grp.n1224 = select i1 %.not, float %grp.n1212, float %grp.n1223, !dbg !11
  %grp.n1225 = select i1 %.not, float %grp.n1223, float %grp.n1213, !dbg !11
  %grp.n1226 = select i1 %.not36, float %grp.n1224, float %grp.n1218, !dbg !11
  %grp.n1227 = bitcast float %grp.n1226 to i32, !dbg !11
  %grp.n1228 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1227), !dbg !11
  %grp.n1229 = bitcast i32 %grp.n1228 to float, !dbg !11
  %grp.n1232 = select i1 %.not36, float %grp.n1225, float %grp.n1219, !dbg !11
  %grp.n1233 = bitcast float %grp.n1232 to i32, !dbg !11
  %grp.n1234 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1233), !dbg !11
  %grp.n1235 = bitcast i32 %grp.n1234 to float, !dbg !11
  %40 = insertelement <4 x float> poison, float %grp.n1218, i64 0, !dbg !11
  %41 = insertelement <4 x float> %40, float %grp.n1219, i64 1, !dbg !11
  %42 = insertelement <4 x float> %41, float %grp.n1229, i64 2, !dbg !11
  %43 = insertelement <4 x float> %42, float %grp.n1235, i64 3, !dbg !11
  %44 = insertelement <4 x float> poison, float %grp.n1229, i64 0, !dbg !11
  %45 = insertelement <4 x float> %44, float %grp.n1235, i64 1, !dbg !11
  %46 = insertelement <4 x float> %45, float %grp.n1224, i64 2, !dbg !11
  %47 = insertelement <4 x float> %46, float %grp.n1225, i64 3, !dbg !11
  %48 = select <4 x i1> %30, <4 x float> %43, <4 x float> %47, !dbg !11
  %grp.n1239 = extractelement <4 x float> %grp.n930, i64 0, !dbg !11
  %grp.n1240 = extractelement <4 x float> %grp.n930, i64 1, !dbg !11
  %grp.n1241 = extractelement <4 x float> %grp.n930, i64 2, !dbg !11
  %grp.n1242 = extractelement <4 x float> %grp.n930, i64 3, !dbg !11
  %grp.n1243 = select i1 %.not, float %grp.n1240, float %grp.n1239, !dbg !11
  %grp.n1244 = bitcast float %grp.n1243 to i32, !dbg !11
  %grp.n1245 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1244), !dbg !11
  %grp.n1246 = bitcast i32 %grp.n1245 to float, !dbg !11
  %grp.n1247 = select i1 %.not, float %grp.n1239, float %grp.n1246, !dbg !11
  %grp.n1248 = select i1 %.not, float %grp.n1246, float %grp.n1240, !dbg !11
  %grp.n1249 = select i1 %.not, float %grp.n1242, float %grp.n1241, !dbg !11
  %grp.n1250 = bitcast float %grp.n1249 to i32, !dbg !11
  %grp.n1251 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1250), !dbg !11
  %grp.n1252 = bitcast i32 %grp.n1251 to float, !dbg !11
  %grp.n1253 = select i1 %.not, float %grp.n1241, float %grp.n1252, !dbg !11
  %grp.n1254 = select i1 %.not, float %grp.n1252, float %grp.n1242, !dbg !11
  %grp.n1255 = select i1 %.not36, float %grp.n1253, float %grp.n1247, !dbg !11
  %grp.n1256 = bitcast float %grp.n1255 to i32, !dbg !11
  %grp.n1257 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1256), !dbg !11
  %grp.n1258 = bitcast i32 %grp.n1257 to float, !dbg !11
  %grp.n1261 = select i1 %.not36, float %grp.n1254, float %grp.n1248, !dbg !11
  %grp.n1262 = bitcast float %grp.n1261 to i32, !dbg !11
  %grp.n1263 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1262), !dbg !11
  %grp.n1264 = bitcast i32 %grp.n1263 to float, !dbg !11
  %49 = insertelement <4 x float> poison, float %grp.n1247, i64 0, !dbg !11
  %50 = insertelement <4 x float> %49, float %grp.n1248, i64 1, !dbg !11
  %51 = insertelement <4 x float> %50, float %grp.n1258, i64 2, !dbg !11
  %52 = insertelement <4 x float> %51, float %grp.n1264, i64 3, !dbg !11
  %53 = insertelement <4 x float> poison, float %grp.n1258, i64 0, !dbg !11
  %54 = insertelement <4 x float> %53, float %grp.n1264, i64 1, !dbg !11
  %55 = insertelement <4 x float> %54, float %grp.n1253, i64 2, !dbg !11
  %56 = insertelement <4 x float> %55, float %grp.n1254, i64 3, !dbg !11
  %57 = select <4 x i1> %30, <4 x float> %52, <4 x float> %56, !dbg !11
  %grp.n1268 = extractelement <4 x float> %grp.n846, i64 0, !dbg !11
  %grp.n1269 = extractelement <4 x float> %grp.n846, i64 1, !dbg !11
  %grp.n1270 = extractelement <4 x float> %grp.n846, i64 2, !dbg !11
  %grp.n1271 = extractelement <4 x float> %grp.n846, i64 3, !dbg !11
  %grp.n1272 = select i1 %.not, float %grp.n1269, float %grp.n1268, !dbg !11
  %grp.n1273 = bitcast float %grp.n1272 to i32, !dbg !11
  %grp.n1274 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1273), !dbg !11
  %grp.n1275 = bitcast i32 %grp.n1274 to float, !dbg !11
  %grp.n1276 = select i1 %.not, float %grp.n1268, float %grp.n1275, !dbg !11
  %grp.n1277 = select i1 %.not, float %grp.n1275, float %grp.n1269, !dbg !11
  %grp.n1278 = select i1 %.not, float %grp.n1271, float %grp.n1270, !dbg !11
  %grp.n1279 = bitcast float %grp.n1278 to i32, !dbg !11
  %grp.n1280 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1279), !dbg !11
  %grp.n1281 = bitcast i32 %grp.n1280 to float, !dbg !11
  %grp.n1282 = select i1 %.not, float %grp.n1270, float %grp.n1281, !dbg !11
  %grp.n1283 = select i1 %.not, float %grp.n1281, float %grp.n1271, !dbg !11
  %grp.n1284 = select i1 %.not36, float %grp.n1282, float %grp.n1276, !dbg !11
  %grp.n1285 = bitcast float %grp.n1284 to i32, !dbg !11
  %grp.n1286 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1285), !dbg !11
  %grp.n1287 = select i1 %.not36, float %grp.n1283, float %grp.n1277, !dbg !11
  %grp.n1288 = bitcast float %grp.n1287 to i32, !dbg !11
  %grp.n1289 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1288), !dbg !11
  %grp.n1290 = shufflevector <1 x float> %grp.n1165, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !26
  %grp.n1291 = extractelement <4 x float> %grp.n1106, i64 0, !dbg !11
  %grp.n1292 = extractelement <4 x float> %grp.n1106, i64 1, !dbg !11
  %grp.n1293 = extractelement <4 x float> %grp.n1106, i64 2, !dbg !11
  %grp.n1294 = extractelement <4 x float> %grp.n1106, i64 3, !dbg !11
  %grp.n1295 = select i1 %.not, float %grp.n1292, float %grp.n1291, !dbg !11
  %grp.n1296 = bitcast float %grp.n1295 to i32, !dbg !11
  %grp.n1297 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1296), !dbg !11
  %grp.n1298 = bitcast i32 %grp.n1297 to float, !dbg !11
  %grp.n1299 = select i1 %.not, float %grp.n1291, float %grp.n1298, !dbg !11
  %grp.n1300 = select i1 %.not, float %grp.n1298, float %grp.n1292, !dbg !11
  %grp.n1301 = select i1 %.not, float %grp.n1294, float %grp.n1293, !dbg !11
  %grp.n1302 = bitcast float %grp.n1301 to i32, !dbg !11
  %grp.n1303 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1302), !dbg !11
  %grp.n1304 = bitcast i32 %grp.n1303 to float, !dbg !11
  %grp.n1305 = select i1 %.not, float %grp.n1293, float %grp.n1304, !dbg !11
  %grp.n1306 = select i1 %.not, float %grp.n1304, float %grp.n1294, !dbg !11
  %grp.n1307 = select i1 %.not36, float %grp.n1305, float %grp.n1299, !dbg !11
  %grp.n1308 = bitcast float %grp.n1307 to i32, !dbg !11
  %grp.n1309 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1308), !dbg !11
  %grp.n1310 = bitcast i32 %grp.n1309 to float, !dbg !11
  %grp.n1313 = select i1 %.not36, float %grp.n1306, float %grp.n1300, !dbg !11
  %grp.n1314 = bitcast float %grp.n1313 to i32, !dbg !11
  %grp.n1315 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1314), !dbg !11
  %grp.n1316 = bitcast i32 %grp.n1315 to float, !dbg !11
  %58 = insertelement <4 x float> poison, float %grp.n1299, i64 0, !dbg !11
  %59 = insertelement <4 x float> %58, float %grp.n1300, i64 1, !dbg !11
  %60 = insertelement <4 x float> %59, float %grp.n1310, i64 2, !dbg !11
  %61 = insertelement <4 x float> %60, float %grp.n1316, i64 3, !dbg !11
  %62 = insertelement <4 x float> poison, float %grp.n1310, i64 0, !dbg !11
  %63 = insertelement <4 x float> %62, float %grp.n1316, i64 1, !dbg !11
  %64 = insertelement <4 x float> %63, float %grp.n1305, i64 2, !dbg !11
  %65 = insertelement <4 x float> %64, float %grp.n1306, i64 3, !dbg !11
  %66 = select <4 x i1> %30, <4 x float> %61, <4 x float> %65, !dbg !11
  %grp.n1319 = extractelement <4 x float> %grp.n1032, i64 0, !dbg !11
  %grp.n1320 = extractelement <4 x float> %grp.n1032, i64 1, !dbg !11
  %grp.n1321 = extractelement <4 x float> %grp.n1032, i64 2, !dbg !11
  %grp.n1322 = extractelement <4 x float> %grp.n1032, i64 3, !dbg !11
  %grp.n1323 = select i1 %.not, float %grp.n1320, float %grp.n1319, !dbg !11
  %grp.n1324 = bitcast float %grp.n1323 to i32, !dbg !11
  %grp.n1325 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1324), !dbg !11
  %grp.n1326 = bitcast i32 %grp.n1325 to float, !dbg !11
  %grp.n1327 = select i1 %.not, float %grp.n1319, float %grp.n1326, !dbg !11
  %grp.n1328 = select i1 %.not, float %grp.n1326, float %grp.n1320, !dbg !11
  %grp.n1329 = select i1 %.not, float %grp.n1322, float %grp.n1321, !dbg !11
  %grp.n1330 = bitcast float %grp.n1329 to i32, !dbg !11
  %grp.n1331 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1330), !dbg !11
  %grp.n1332 = bitcast i32 %grp.n1331 to float, !dbg !11
  %grp.n1333 = select i1 %.not, float %grp.n1321, float %grp.n1332, !dbg !11
  %grp.n1334 = select i1 %.not, float %grp.n1332, float %grp.n1322, !dbg !11
  %grp.n1335 = select i1 %.not36, float %grp.n1333, float %grp.n1327, !dbg !11
  %grp.n1336 = bitcast float %grp.n1335 to i32, !dbg !11
  %grp.n1337 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1336), !dbg !11
  %grp.n1338 = bitcast i32 %grp.n1337 to float, !dbg !11
  %grp.n1341 = select i1 %.not36, float %grp.n1334, float %grp.n1328, !dbg !11
  %grp.n1342 = bitcast float %grp.n1341 to i32, !dbg !11
  %grp.n1343 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1342), !dbg !11
  %grp.n1344 = bitcast i32 %grp.n1343 to float, !dbg !11
  %67 = insertelement <4 x float> poison, float %grp.n1327, i64 0, !dbg !11
  %68 = insertelement <4 x float> %67, float %grp.n1328, i64 1, !dbg !11
  %69 = insertelement <4 x float> %68, float %grp.n1338, i64 2, !dbg !11
  %70 = insertelement <4 x float> %69, float %grp.n1344, i64 3, !dbg !11
  %71 = insertelement <4 x float> poison, float %grp.n1338, i64 0, !dbg !11
  %72 = insertelement <4 x float> %71, float %grp.n1344, i64 1, !dbg !11
  %73 = insertelement <4 x float> %72, float %grp.n1333, i64 2, !dbg !11
  %74 = insertelement <4 x float> %73, float %grp.n1334, i64 3, !dbg !11
  %75 = select <4 x i1> %30, <4 x float> %70, <4 x float> %74, !dbg !11
  %grp.n1347 = extractelement <4 x float> %grp.n948, i64 0, !dbg !11
  %grp.n1348 = extractelement <4 x float> %grp.n948, i64 1, !dbg !11
  %grp.n1349 = extractelement <4 x float> %grp.n948, i64 2, !dbg !11
  %grp.n1350 = extractelement <4 x float> %grp.n948, i64 3, !dbg !11
  %grp.n1351 = select i1 %.not, float %grp.n1348, float %grp.n1347, !dbg !11
  %grp.n1352 = bitcast float %grp.n1351 to i32, !dbg !11
  %grp.n1353 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1352), !dbg !11
  %grp.n1354 = bitcast i32 %grp.n1353 to float, !dbg !11
  %grp.n1355 = select i1 %.not, float %grp.n1347, float %grp.n1354, !dbg !11
  %grp.n1356 = select i1 %.not, float %grp.n1354, float %grp.n1348, !dbg !11
  %grp.n1357 = select i1 %.not, float %grp.n1350, float %grp.n1349, !dbg !11
  %grp.n1358 = bitcast float %grp.n1357 to i32, !dbg !11
  %grp.n1359 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1358), !dbg !11
  %grp.n1360 = bitcast i32 %grp.n1359 to float, !dbg !11
  %grp.n1361 = select i1 %.not, float %grp.n1349, float %grp.n1360, !dbg !11
  %grp.n1362 = select i1 %.not, float %grp.n1360, float %grp.n1350, !dbg !11
  %grp.n1363 = select i1 %.not36, float %grp.n1361, float %grp.n1355, !dbg !11
  %grp.n1364 = bitcast float %grp.n1363 to i32, !dbg !11
  %grp.n1365 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1364), !dbg !11
  %grp.n1366 = bitcast i32 %grp.n1365 to float, !dbg !11
  %grp.n1369 = select i1 %.not36, float %grp.n1362, float %grp.n1356, !dbg !11
  %grp.n1370 = bitcast float %grp.n1369 to i32, !dbg !11
  %grp.n1371 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1370), !dbg !11
  %grp.n1372 = bitcast i32 %grp.n1371 to float, !dbg !11
  %76 = insertelement <4 x float> poison, float %grp.n1355, i64 0, !dbg !11
  %77 = insertelement <4 x float> %76, float %grp.n1356, i64 1, !dbg !11
  %78 = insertelement <4 x float> %77, float %grp.n1366, i64 2, !dbg !11
  %79 = insertelement <4 x float> %78, float %grp.n1372, i64 3, !dbg !11
  %80 = insertelement <4 x float> poison, float %grp.n1366, i64 0, !dbg !11
  %81 = insertelement <4 x float> %80, float %grp.n1372, i64 1, !dbg !11
  %82 = insertelement <4 x float> %81, float %grp.n1361, i64 2, !dbg !11
  %83 = insertelement <4 x float> %82, float %grp.n1362, i64 3, !dbg !11
  %84 = select <4 x i1> %30, <4 x float> %79, <4 x float> %83, !dbg !11
  %grp.n1375 = extractelement <4 x float> %grp.n864, i64 0, !dbg !11
  %grp.n1376 = extractelement <4 x float> %grp.n864, i64 1, !dbg !11
  %grp.n1377 = extractelement <4 x float> %grp.n864, i64 2, !dbg !11
  %grp.n1378 = extractelement <4 x float> %grp.n864, i64 3, !dbg !11
  %grp.n1379 = select i1 %.not, float %grp.n1376, float %grp.n1375, !dbg !11
  %grp.n1380 = bitcast float %grp.n1379 to i32, !dbg !11
  %grp.n1381 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1380), !dbg !11
  %grp.n1382 = bitcast i32 %grp.n1381 to float, !dbg !11
  %grp.n1383 = select i1 %.not, float %grp.n1375, float %grp.n1382, !dbg !11
  %grp.n1384 = select i1 %.not, float %grp.n1382, float %grp.n1376, !dbg !11
  %grp.n1385 = select i1 %.not, float %grp.n1378, float %grp.n1377, !dbg !11
  %grp.n1386 = bitcast float %grp.n1385 to i32, !dbg !11
  %grp.n1387 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1386), !dbg !11
  %grp.n1388 = bitcast i32 %grp.n1387 to float, !dbg !11
  %grp.n1389 = select i1 %.not, float %grp.n1377, float %grp.n1388, !dbg !11
  %grp.n1390 = select i1 %.not, float %grp.n1388, float %grp.n1378, !dbg !11
  %grp.n1391 = select i1 %.not36, float %grp.n1389, float %grp.n1383, !dbg !11
  %grp.n1392 = bitcast float %grp.n1391 to i32, !dbg !11
  %grp.n1393 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1392), !dbg !11
  %grp.n1394 = select i1 %.not36, float %grp.n1390, float %grp.n1384, !dbg !11
  %grp.n1395 = bitcast float %grp.n1394 to i32, !dbg !11
  %grp.n1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1395), !dbg !11
  %grp.n1398 = extractelement <4 x float> %grp.n1117, i64 0, !dbg !11
  %grp.n1399 = extractelement <4 x float> %grp.n1117, i64 1, !dbg !11
  %grp.n1400 = extractelement <4 x float> %grp.n1117, i64 2, !dbg !11
  %grp.n1401 = extractelement <4 x float> %grp.n1117, i64 3, !dbg !11
  %grp.n1402 = select i1 %.not, float %grp.n1399, float %grp.n1398, !dbg !11
  %grp.n1403 = bitcast float %grp.n1402 to i32, !dbg !11
  %grp.n1404 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1403), !dbg !11
  %grp.n1405 = bitcast i32 %grp.n1404 to float, !dbg !11
  %grp.n1406 = select i1 %.not, float %grp.n1398, float %grp.n1405, !dbg !11
  %grp.n1407 = select i1 %.not, float %grp.n1405, float %grp.n1399, !dbg !11
  %grp.n1408 = select i1 %.not, float %grp.n1401, float %grp.n1400, !dbg !11
  %grp.n1409 = bitcast float %grp.n1408 to i32, !dbg !11
  %grp.n1410 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1409), !dbg !11
  %grp.n1411 = bitcast i32 %grp.n1410 to float, !dbg !11
  %grp.n1412 = select i1 %.not, float %grp.n1400, float %grp.n1411, !dbg !11
  %grp.n1413 = select i1 %.not, float %grp.n1411, float %grp.n1401, !dbg !11
  %grp.n1414 = select i1 %.not36, float %grp.n1412, float %grp.n1406, !dbg !11
  %grp.n1415 = bitcast float %grp.n1414 to i32, !dbg !11
  %grp.n1416 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1415), !dbg !11
  %grp.n1417 = bitcast i32 %grp.n1416 to float, !dbg !11
  %grp.n1420 = select i1 %.not36, float %grp.n1413, float %grp.n1407, !dbg !11
  %grp.n1421 = bitcast float %grp.n1420 to i32, !dbg !11
  %grp.n1422 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1421), !dbg !11
  %grp.n1423 = bitcast i32 %grp.n1422 to float, !dbg !11
  %85 = insertelement <4 x float> poison, float %grp.n1406, i64 0, !dbg !11
  %86 = insertelement <4 x float> %85, float %grp.n1407, i64 1, !dbg !11
  %87 = insertelement <4 x float> %86, float %grp.n1417, i64 2, !dbg !11
  %88 = insertelement <4 x float> %87, float %grp.n1423, i64 3, !dbg !11
  %89 = insertelement <4 x float> poison, float %grp.n1417, i64 0, !dbg !11
  %90 = insertelement <4 x float> %89, float %grp.n1423, i64 1, !dbg !11
  %91 = insertelement <4 x float> %90, float %grp.n1412, i64 2, !dbg !11
  %92 = insertelement <4 x float> %91, float %grp.n1413, i64 3, !dbg !11
  %93 = select <4 x i1> %30, <4 x float> %88, <4 x float> %92, !dbg !11
  %grp.n1427 = extractelement <4 x float> %grp.n1043, i64 0, !dbg !11
  %grp.n1428 = extractelement <4 x float> %grp.n1043, i64 1, !dbg !11
  %grp.n1429 = extractelement <4 x float> %grp.n1043, i64 2, !dbg !11
  %grp.n1430 = extractelement <4 x float> %grp.n1043, i64 3, !dbg !11
  %grp.n1431 = select i1 %.not, float %grp.n1428, float %grp.n1427, !dbg !11
  %grp.n1432 = bitcast float %grp.n1431 to i32, !dbg !11
  %grp.n1433 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1432), !dbg !11
  %grp.n1434 = bitcast i32 %grp.n1433 to float, !dbg !11
  %grp.n1435 = select i1 %.not, float %grp.n1427, float %grp.n1434, !dbg !11
  %grp.n1436 = select i1 %.not, float %grp.n1434, float %grp.n1428, !dbg !11
  %grp.n1437 = select i1 %.not, float %grp.n1430, float %grp.n1429, !dbg !11
  %grp.n1438 = bitcast float %grp.n1437 to i32, !dbg !11
  %grp.n1439 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1438), !dbg !11
  %grp.n1440 = bitcast i32 %grp.n1439 to float, !dbg !11
  %grp.n1441 = select i1 %.not, float %grp.n1429, float %grp.n1440, !dbg !11
  %grp.n1442 = select i1 %.not, float %grp.n1440, float %grp.n1430, !dbg !11
  %grp.n1443 = select i1 %.not36, float %grp.n1441, float %grp.n1435, !dbg !11
  %grp.n1444 = bitcast float %grp.n1443 to i32, !dbg !11
  %grp.n1445 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1444), !dbg !11
  %grp.n1446 = bitcast i32 %grp.n1445 to float, !dbg !11
  %grp.n1449 = select i1 %.not36, float %grp.n1442, float %grp.n1436, !dbg !11
  %grp.n1450 = bitcast float %grp.n1449 to i32, !dbg !11
  %grp.n1451 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1450), !dbg !11
  %grp.n1452 = bitcast i32 %grp.n1451 to float, !dbg !11
  %94 = insertelement <4 x float> poison, float %grp.n1435, i64 0, !dbg !11
  %95 = insertelement <4 x float> %94, float %grp.n1436, i64 1, !dbg !11
  %96 = insertelement <4 x float> %95, float %grp.n1446, i64 2, !dbg !11
  %97 = insertelement <4 x float> %96, float %grp.n1452, i64 3, !dbg !11
  %98 = insertelement <4 x float> poison, float %grp.n1446, i64 0, !dbg !11
  %99 = insertelement <4 x float> %98, float %grp.n1452, i64 1, !dbg !11
  %100 = insertelement <4 x float> %99, float %grp.n1441, i64 2, !dbg !11
  %101 = insertelement <4 x float> %100, float %grp.n1442, i64 3, !dbg !11
  %102 = select <4 x i1> %30, <4 x float> %97, <4 x float> %101, !dbg !11
  %grp.n1456 = extractelement <4 x float> %grp.n959, i64 0, !dbg !11
  %grp.n1457 = extractelement <4 x float> %grp.n959, i64 1, !dbg !11
  %grp.n1458 = extractelement <4 x float> %grp.n959, i64 2, !dbg !11
  %grp.n1459 = extractelement <4 x float> %grp.n959, i64 3, !dbg !11
  %grp.n1460 = select i1 %.not, float %grp.n1457, float %grp.n1456, !dbg !11
  %grp.n1461 = bitcast float %grp.n1460 to i32, !dbg !11
  %grp.n1462 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1461), !dbg !11
  %grp.n1463 = bitcast i32 %grp.n1462 to float, !dbg !11
  %grp.n1464 = select i1 %.not, float %grp.n1456, float %grp.n1463, !dbg !11
  %grp.n1465 = select i1 %.not, float %grp.n1463, float %grp.n1457, !dbg !11
  %grp.n1466 = select i1 %.not, float %grp.n1459, float %grp.n1458, !dbg !11
  %grp.n1467 = bitcast float %grp.n1466 to i32, !dbg !11
  %grp.n1468 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1467), !dbg !11
  %grp.n1469 = bitcast i32 %grp.n1468 to float, !dbg !11
  %grp.n1470 = select i1 %.not, float %grp.n1458, float %grp.n1469, !dbg !11
  %grp.n1471 = select i1 %.not, float %grp.n1469, float %grp.n1459, !dbg !11
  %grp.n1472 = select i1 %.not36, float %grp.n1470, float %grp.n1464, !dbg !11
  %grp.n1473 = bitcast float %grp.n1472 to i32, !dbg !11
  %grp.n1474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1473), !dbg !11
  %grp.n1475 = bitcast i32 %grp.n1474 to float, !dbg !11
  %grp.n1478 = select i1 %.not36, float %grp.n1471, float %grp.n1465, !dbg !11
  %grp.n1479 = bitcast float %grp.n1478 to i32, !dbg !11
  %grp.n1480 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1479), !dbg !11
  %grp.n1481 = bitcast i32 %grp.n1480 to float, !dbg !11
  %103 = insertelement <4 x float> poison, float %grp.n1464, i64 0, !dbg !11
  %104 = insertelement <4 x float> %103, float %grp.n1465, i64 1, !dbg !11
  %105 = insertelement <4 x float> %104, float %grp.n1475, i64 2, !dbg !11
  %106 = insertelement <4 x float> %105, float %grp.n1481, i64 3, !dbg !11
  %107 = insertelement <4 x float> poison, float %grp.n1475, i64 0, !dbg !11
  %108 = insertelement <4 x float> %107, float %grp.n1481, i64 1, !dbg !11
  %109 = insertelement <4 x float> %108, float %grp.n1470, i64 2, !dbg !11
  %110 = insertelement <4 x float> %109, float %grp.n1471, i64 3, !dbg !11
  %111 = select <4 x i1> %30, <4 x float> %106, <4 x float> %110, !dbg !11
  %grp.n1485 = extractelement <4 x float> %grp.n875, i64 0, !dbg !11
  %grp.n1486 = extractelement <4 x float> %grp.n875, i64 1, !dbg !11
  %grp.n1487 = extractelement <4 x float> %grp.n875, i64 2, !dbg !11
  %grp.n1488 = extractelement <4 x float> %grp.n875, i64 3, !dbg !11
  %grp.n1489 = select i1 %.not, float %grp.n1486, float %grp.n1485, !dbg !11
  %grp.n1490 = bitcast float %grp.n1489 to i32, !dbg !11
  %grp.n1491 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1490), !dbg !11
  %grp.n1492 = bitcast i32 %grp.n1491 to float, !dbg !11
  %grp.n1493 = select i1 %.not, float %grp.n1485, float %grp.n1492, !dbg !11
  %grp.n1494 = select i1 %.not, float %grp.n1492, float %grp.n1486, !dbg !11
  %grp.n1495 = select i1 %.not, float %grp.n1488, float %grp.n1487, !dbg !11
  %grp.n1496 = bitcast float %grp.n1495 to i32, !dbg !11
  %grp.n1497 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1496), !dbg !11
  %grp.n1498 = bitcast i32 %grp.n1497 to float, !dbg !11
  %grp.n1499 = select i1 %.not, float %grp.n1487, float %grp.n1498, !dbg !11
  %grp.n1500 = select i1 %.not, float %grp.n1498, float %grp.n1488, !dbg !11
  %grp.n1501 = select i1 %.not36, float %grp.n1499, float %grp.n1493, !dbg !11
  %grp.n1502 = bitcast float %grp.n1501 to i32, !dbg !11
  %grp.n1503 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1502), !dbg !11
  %grp.n1504 = select i1 %.not36, float %grp.n1500, float %grp.n1494, !dbg !11
  %grp.n1505 = bitcast float %grp.n1504 to i32, !dbg !11
  %grp.n1506 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1505), !dbg !11
  %grp.n1507 = shufflevector <1 x float> %grp.n1156, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !26
  %grp.n1508 = extractelement <4 x float> %grp.n1127, i64 0, !dbg !11
  %grp.n1509 = extractelement <4 x float> %grp.n1127, i64 1, !dbg !11
  %grp.n1510 = extractelement <4 x float> %grp.n1127, i64 2, !dbg !11
  %grp.n1511 = extractelement <4 x float> %grp.n1127, i64 3, !dbg !11
  %grp.n1512 = select i1 %.not, float %grp.n1509, float %grp.n1508, !dbg !11
  %grp.n1513 = bitcast float %grp.n1512 to i32, !dbg !11
  %grp.n1514 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1513), !dbg !11
  %grp.n1515 = bitcast i32 %grp.n1514 to float, !dbg !11
  %grp.n1516 = select i1 %.not, float %grp.n1508, float %grp.n1515, !dbg !11
  %grp.n1517 = select i1 %.not, float %grp.n1515, float %grp.n1509, !dbg !11
  %grp.n1518 = select i1 %.not, float %grp.n1511, float %grp.n1510, !dbg !11
  %grp.n1519 = bitcast float %grp.n1518 to i32, !dbg !11
  %grp.n1520 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1519), !dbg !11
  %grp.n1521 = bitcast i32 %grp.n1520 to float, !dbg !11
  %grp.n1522 = select i1 %.not, float %grp.n1510, float %grp.n1521, !dbg !11
  %grp.n1523 = select i1 %.not, float %grp.n1521, float %grp.n1511, !dbg !11
  %grp.n1524 = select i1 %.not36, float %grp.n1522, float %grp.n1516, !dbg !11
  %grp.n1525 = bitcast float %grp.n1524 to i32, !dbg !11
  %grp.n1526 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1525), !dbg !11
  %grp.n1527 = bitcast i32 %grp.n1526 to float, !dbg !11
  %grp.n1530 = select i1 %.not36, float %grp.n1523, float %grp.n1517, !dbg !11
  %grp.n1531 = bitcast float %grp.n1530 to i32, !dbg !11
  %grp.n1532 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1531), !dbg !11
  %grp.n1533 = bitcast i32 %grp.n1532 to float, !dbg !11
  %112 = insertelement <4 x float> poison, float %grp.n1516, i64 0, !dbg !11
  %113 = insertelement <4 x float> %112, float %grp.n1517, i64 1, !dbg !11
  %114 = insertelement <4 x float> %113, float %grp.n1527, i64 2, !dbg !11
  %115 = insertelement <4 x float> %114, float %grp.n1533, i64 3, !dbg !11
  %116 = insertelement <4 x float> poison, float %grp.n1527, i64 0, !dbg !11
  %117 = insertelement <4 x float> %116, float %grp.n1533, i64 1, !dbg !11
  %118 = insertelement <4 x float> %117, float %grp.n1522, i64 2, !dbg !11
  %119 = insertelement <4 x float> %118, float %grp.n1523, i64 3, !dbg !11
  %120 = select <4 x i1> %30, <4 x float> %115, <4 x float> %119, !dbg !11
  %grp.n1536 = extractelement <4 x float> %grp.n1053, i64 0, !dbg !11
  %grp.n1537 = extractelement <4 x float> %grp.n1053, i64 1, !dbg !11
  %grp.n1538 = extractelement <4 x float> %grp.n1053, i64 2, !dbg !11
  %grp.n1539 = extractelement <4 x float> %grp.n1053, i64 3, !dbg !11
  %grp.n1540 = select i1 %.not, float %grp.n1537, float %grp.n1536, !dbg !11
  %grp.n1541 = bitcast float %grp.n1540 to i32, !dbg !11
  %grp.n1542 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1541), !dbg !11
  %grp.n1543 = bitcast i32 %grp.n1542 to float, !dbg !11
  %grp.n1544 = select i1 %.not, float %grp.n1536, float %grp.n1543, !dbg !11
  %grp.n1545 = select i1 %.not, float %grp.n1543, float %grp.n1537, !dbg !11
  %grp.n1546 = select i1 %.not, float %grp.n1539, float %grp.n1538, !dbg !11
  %grp.n1547 = bitcast float %grp.n1546 to i32, !dbg !11
  %grp.n1548 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1547), !dbg !11
  %grp.n1549 = bitcast i32 %grp.n1548 to float, !dbg !11
  %grp.n1550 = select i1 %.not, float %grp.n1538, float %grp.n1549, !dbg !11
  %grp.n1551 = select i1 %.not, float %grp.n1549, float %grp.n1539, !dbg !11
  %grp.n1552 = select i1 %.not36, float %grp.n1550, float %grp.n1544, !dbg !11
  %grp.n1553 = bitcast float %grp.n1552 to i32, !dbg !11
  %grp.n1554 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1553), !dbg !11
  %grp.n1555 = bitcast i32 %grp.n1554 to float, !dbg !11
  %grp.n1558 = select i1 %.not36, float %grp.n1551, float %grp.n1545, !dbg !11
  %grp.n1559 = bitcast float %grp.n1558 to i32, !dbg !11
  %grp.n1560 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1559), !dbg !11
  %grp.n1561 = bitcast i32 %grp.n1560 to float, !dbg !11
  %121 = insertelement <4 x float> poison, float %grp.n1544, i64 0, !dbg !11
  %122 = insertelement <4 x float> %121, float %grp.n1545, i64 1, !dbg !11
  %123 = insertelement <4 x float> %122, float %grp.n1555, i64 2, !dbg !11
  %124 = insertelement <4 x float> %123, float %grp.n1561, i64 3, !dbg !11
  %125 = insertelement <4 x float> poison, float %grp.n1555, i64 0, !dbg !11
  %126 = insertelement <4 x float> %125, float %grp.n1561, i64 1, !dbg !11
  %127 = insertelement <4 x float> %126, float %grp.n1550, i64 2, !dbg !11
  %128 = insertelement <4 x float> %127, float %grp.n1551, i64 3, !dbg !11
  %129 = select <4 x i1> %30, <4 x float> %124, <4 x float> %128, !dbg !11
  %grp.n1564 = extractelement <4 x float> %grp.n969, i64 0, !dbg !11
  %grp.n1565 = extractelement <4 x float> %grp.n969, i64 1, !dbg !11
  %grp.n1566 = extractelement <4 x float> %grp.n969, i64 2, !dbg !11
  %grp.n1567 = extractelement <4 x float> %grp.n969, i64 3, !dbg !11
  %grp.n1568 = select i1 %.not, float %grp.n1565, float %grp.n1564, !dbg !11
  %grp.n1569 = bitcast float %grp.n1568 to i32, !dbg !11
  %grp.n1570 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1569), !dbg !11
  %grp.n1571 = bitcast i32 %grp.n1570 to float, !dbg !11
  %grp.n1572 = select i1 %.not, float %grp.n1564, float %grp.n1571, !dbg !11
  %grp.n1573 = select i1 %.not, float %grp.n1571, float %grp.n1565, !dbg !11
  %grp.n1574 = select i1 %.not, float %grp.n1567, float %grp.n1566, !dbg !11
  %grp.n1575 = bitcast float %grp.n1574 to i32, !dbg !11
  %grp.n1576 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1575), !dbg !11
  %grp.n1577 = bitcast i32 %grp.n1576 to float, !dbg !11
  %grp.n1578 = select i1 %.not, float %grp.n1566, float %grp.n1577, !dbg !11
  %grp.n1579 = select i1 %.not, float %grp.n1577, float %grp.n1567, !dbg !11
  %grp.n1580 = select i1 %.not36, float %grp.n1578, float %grp.n1572, !dbg !11
  %grp.n1581 = bitcast float %grp.n1580 to i32, !dbg !11
  %grp.n1582 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1581), !dbg !11
  %grp.n1583 = bitcast i32 %grp.n1582 to float, !dbg !11
  %grp.n1586 = select i1 %.not36, float %grp.n1579, float %grp.n1573, !dbg !11
  %grp.n1587 = bitcast float %grp.n1586 to i32, !dbg !11
  %grp.n1588 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1587), !dbg !11
  %grp.n1589 = bitcast i32 %grp.n1588 to float, !dbg !11
  %130 = insertelement <4 x float> poison, float %grp.n1572, i64 0, !dbg !11
  %131 = insertelement <4 x float> %130, float %grp.n1573, i64 1, !dbg !11
  %132 = insertelement <4 x float> %131, float %grp.n1583, i64 2, !dbg !11
  %133 = insertelement <4 x float> %132, float %grp.n1589, i64 3, !dbg !11
  %134 = insertelement <4 x float> poison, float %grp.n1583, i64 0, !dbg !11
  %135 = insertelement <4 x float> %134, float %grp.n1589, i64 1, !dbg !11
  %136 = insertelement <4 x float> %135, float %grp.n1578, i64 2, !dbg !11
  %137 = insertelement <4 x float> %136, float %grp.n1579, i64 3, !dbg !11
  %138 = select <4 x i1> %30, <4 x float> %133, <4 x float> %137, !dbg !11
  %grp.n1592 = extractelement <4 x float> %grp.n885, i64 0, !dbg !11
  %grp.n1593 = extractelement <4 x float> %grp.n885, i64 1, !dbg !11
  %grp.n1594 = extractelement <4 x float> %grp.n885, i64 2, !dbg !11
  %grp.n1595 = extractelement <4 x float> %grp.n885, i64 3, !dbg !11
  %grp.n1596 = select i1 %.not, float %grp.n1593, float %grp.n1592, !dbg !11
  %grp.n1597 = bitcast float %grp.n1596 to i32, !dbg !11
  %grp.n1598 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1597), !dbg !11
  %grp.n1599 = bitcast i32 %grp.n1598 to float, !dbg !11
  %grp.n1600 = select i1 %.not, float %grp.n1592, float %grp.n1599, !dbg !11
  %grp.n1601 = select i1 %.not, float %grp.n1599, float %grp.n1593, !dbg !11
  %grp.n1602 = select i1 %.not, float %grp.n1595, float %grp.n1594, !dbg !11
  %grp.n1603 = bitcast float %grp.n1602 to i32, !dbg !11
  %grp.n1604 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n523, i32 %grp.n1603), !dbg !11
  %grp.n1605 = bitcast i32 %grp.n1604 to float, !dbg !11
  %grp.n1606 = select i1 %.not, float %grp.n1594, float %grp.n1605, !dbg !11
  %grp.n1607 = select i1 %.not, float %grp.n1605, float %grp.n1595, !dbg !11
  %grp.n1608 = select i1 %.not36, float %grp.n1606, float %grp.n1600, !dbg !11
  %grp.n1609 = bitcast float %grp.n1608 to i32, !dbg !11
  %grp.n1610 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1609), !dbg !11
  %grp.n1611 = select i1 %.not36, float %grp.n1607, float %grp.n1601, !dbg !11
  %grp.n1612 = bitcast float %grp.n1611 to i32, !dbg !11
  %grp.n1613 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %grp.n531, i32 %grp.n1612), !dbg !11
  %grp.n1614 = bitcast i32 %grp.n1610 to float, !dbg !11
  %grp.n1617 = bitcast i32 %grp.n1613 to float, !dbg !11
  %139 = insertelement <4 x float> poison, float %grp.n1600, i64 0, !dbg !11
  %140 = insertelement <4 x float> %139, float %grp.n1601, i64 1, !dbg !11
  %141 = insertelement <4 x float> %140, float %grp.n1614, i64 2, !dbg !11
  %142 = insertelement <4 x float> %141, float %grp.n1617, i64 3, !dbg !11
  %143 = insertelement <4 x float> poison, float %grp.n1614, i64 0, !dbg !11
  %144 = insertelement <4 x float> %143, float %grp.n1617, i64 1, !dbg !11
  %145 = insertelement <4 x float> %144, float %grp.n1606, i64 2, !dbg !11
  %146 = insertelement <4 x float> %145, float %grp.n1607, i64 3, !dbg !11
  %147 = select <4 x i1> %30, <4 x float> %142, <4 x float> %146, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %grp.n1507, !dbg !26
  %normalized = fmul <4 x float> %row.reciprocal, %147, !dbg !26
  %grp.n1624 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !27
  %grp.n1625 = extractelement <4 x half> %grp.n1624, i64 3, !dbg !27
  %grp.n1626 = extractelement <4 x half> %grp.n1624, i64 2, !dbg !27
  %grp.n1627 = extractelement <4 x half> %grp.n1624, i64 1, !dbg !27
  %grp.n1628 = extractelement <4 x half> %grp.n1624, i64 0, !dbg !27
  %grp.n1629 = bitcast i32 %grp.n1503 to float, !dbg !11
  %grp.n1632 = bitcast i32 %grp.n1506 to float, !dbg !11
  %148 = insertelement <4 x float> poison, float %grp.n1493, i64 0, !dbg !11
  %149 = insertelement <4 x float> %148, float %grp.n1494, i64 1, !dbg !11
  %150 = insertelement <4 x float> %149, float %grp.n1629, i64 2, !dbg !11
  %151 = insertelement <4 x float> %150, float %grp.n1632, i64 3, !dbg !11
  %152 = insertelement <4 x float> poison, float %grp.n1629, i64 0, !dbg !11
  %153 = insertelement <4 x float> %152, float %grp.n1632, i64 1, !dbg !11
  %154 = insertelement <4 x float> %153, float %grp.n1499, i64 2, !dbg !11
  %155 = insertelement <4 x float> %154, float %grp.n1500, i64 3, !dbg !11
  %156 = select <4 x i1> %30, <4 x float> %151, <4 x float> %155, !dbg !11
  %normalized37 = fmul <4 x float> %row.reciprocal, %156, !dbg !26
  %grp.n1639 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !27
  %grp.n1640 = extractelement <4 x half> %grp.n1639, i64 3, !dbg !27
  %grp.n1641 = extractelement <4 x half> %grp.n1639, i64 2, !dbg !27
  %grp.n1642 = extractelement <4 x half> %grp.n1639, i64 1, !dbg !27
  %grp.n1643 = extractelement <4 x half> %grp.n1639, i64 0, !dbg !27
  %grp.n1644 = bitcast i32 %grp.n1393 to float, !dbg !11
  %grp.n1647 = bitcast i32 %grp.n1396 to float, !dbg !11
  %157 = insertelement <4 x float> poison, float %grp.n1383, i64 0, !dbg !11
  %158 = insertelement <4 x float> %157, float %grp.n1384, i64 1, !dbg !11
  %159 = insertelement <4 x float> %158, float %grp.n1644, i64 2, !dbg !11
  %160 = insertelement <4 x float> %159, float %grp.n1647, i64 3, !dbg !11
  %161 = insertelement <4 x float> poison, float %grp.n1644, i64 0, !dbg !11
  %162 = insertelement <4 x float> %161, float %grp.n1647, i64 1, !dbg !11
  %163 = insertelement <4 x float> %162, float %grp.n1389, i64 2, !dbg !11
  %164 = insertelement <4 x float> %163, float %grp.n1390, i64 3, !dbg !11
  %165 = select <4 x i1> %30, <4 x float> %160, <4 x float> %164, !dbg !11
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %grp.n1290, !dbg !26
  %normalized39 = fmul <4 x float> %row.reciprocal38, %165, !dbg !26
  %grp.n1654 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !27
  %grp.n1655 = extractelement <4 x half> %grp.n1654, i64 3, !dbg !27
  %grp.n1656 = extractelement <4 x half> %grp.n1654, i64 2, !dbg !27
  %grp.n1657 = extractelement <4 x half> %grp.n1654, i64 1, !dbg !27
  %grp.n1658 = extractelement <4 x half> %grp.n1654, i64 0, !dbg !27
  %grp.n1659 = bitcast i32 %grp.n1286 to float, !dbg !11
  %grp.n1662 = bitcast i32 %grp.n1289 to float, !dbg !11
  %166 = insertelement <4 x float> poison, float %grp.n1276, i64 0, !dbg !11
  %167 = insertelement <4 x float> %166, float %grp.n1277, i64 1, !dbg !11
  %168 = insertelement <4 x float> %167, float %grp.n1659, i64 2, !dbg !11
  %169 = insertelement <4 x float> %168, float %grp.n1662, i64 3, !dbg !11
  %170 = insertelement <4 x float> poison, float %grp.n1659, i64 0, !dbg !11
  %171 = insertelement <4 x float> %170, float %grp.n1662, i64 1, !dbg !11
  %172 = insertelement <4 x float> %171, float %grp.n1282, i64 2, !dbg !11
  %173 = insertelement <4 x float> %172, float %grp.n1283, i64 3, !dbg !11
  %174 = select <4 x i1> %30, <4 x float> %169, <4 x float> %173, !dbg !11
  %normalized40 = fmul <4 x float> %row.reciprocal38, %174, !dbg !26
  %grp.n1669 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !27
  %grp.n1670 = extractelement <4 x half> %grp.n1669, i64 3, !dbg !27
  %grp.n1671 = extractelement <4 x half> %grp.n1669, i64 2, !dbg !27
  %grp.n1672 = extractelement <4 x half> %grp.n1669, i64 1, !dbg !27
  %grp.n1673 = extractelement <4 x half> %grp.n1669, i64 0, !dbg !27
  %address.thread974 = tail call i32 asm sideeffect "", "=v,0"(i32 %grp.n25) #8, !dbg !28, !range !9, !frisk.register_lifetimes !5
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %grp.n1677 = lshr i64 %address.local975, 2, !dbg !28
  %address.local978 = and i64 %grp.n1677, 48, !dbg !28
  %address.local980 = and i64 %address.local975, 240, !dbg !28
  %address.local979 = add i64 %grp.n21, %address.local975, !dbg !28
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !28
  %address.local982 = add i64 %address.local981, %address.local978, !dbg !28
  %175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %grp.n22, !dbg !28
  %address.local985.idx = shl nuw nsw i64 %address.local982, 8, !dbg !28
  %address.local985 = getelementptr inbounds nuw i8, ptr addrspace(1) %175, i64 %address.local985.idx, !dbg !28
  %address.local987 = and i64 %grp.n1677, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !28
  store half %grp.n1673, ptr addrspace(1) %address.local988, align 2, !dbg !28
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !28
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !28
  store half %grp.n1672, ptr addrspace(1) %address.local991, align 2, !dbg !28
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !28
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !28
  store half %grp.n1671, ptr addrspace(1) %address.local994, align 2, !dbg !28
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !28
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !28
  store half %grp.n1670, ptr addrspace(1) %address.local997, align 2, !dbg !28
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !28
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !28
  store half %grp.n1658, ptr addrspace(1) %address.local1000, align 2, !dbg !28
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !28
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !28
  store half %grp.n1657, ptr addrspace(1) %address.local1003, align 2, !dbg !28
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !28
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !28
  store half %grp.n1656, ptr addrspace(1) %address.local1006, align 2, !dbg !28
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !28
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !28
  store half %grp.n1655, ptr addrspace(1) %address.local1009, align 2, !dbg !28
  %address.local1013 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local985, i64 16384, !dbg !28
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !28
  store half %grp.n1643, ptr addrspace(1) %address.local1014, align 2, !dbg !28
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !28
  store half %grp.n1642, ptr addrspace(1) %address.local1016, align 2, !dbg !28
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !28
  store half %grp.n1641, ptr addrspace(1) %address.local1018, align 2, !dbg !28
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !28
  store half %grp.n1640, ptr addrspace(1) %address.local1020, align 2, !dbg !28
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !28
  store half %grp.n1628, ptr addrspace(1) %address.local1022, align 2, !dbg !28
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !28
  store half %grp.n1627, ptr addrspace(1) %address.local1024, align 2, !dbg !28
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !28
  store half %grp.n1626, ptr addrspace(1) %address.local1026, align 2, !dbg !28
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !28
  store half %grp.n1625, ptr addrspace(1) %address.local1028, align 2, !dbg !28
  %normalized42 = fmul <4 x float> %row.reciprocal, %138, !dbg !26
  %grp.n1682 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !27
  %grp.n1683 = extractelement <4 x half> %grp.n1682, i64 3, !dbg !27
  %grp.n1684 = extractelement <4 x half> %grp.n1682, i64 2, !dbg !27
  %grp.n1685 = extractelement <4 x half> %grp.n1682, i64 1, !dbg !27
  %grp.n1686 = extractelement <4 x half> %grp.n1682, i64 0, !dbg !27
  %normalized43 = fmul <4 x float> %row.reciprocal, %111, !dbg !26
  %grp.n1691 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !27
  %grp.n1692 = extractelement <4 x half> %grp.n1691, i64 3, !dbg !27
  %grp.n1693 = extractelement <4 x half> %grp.n1691, i64 2, !dbg !27
  %grp.n1694 = extractelement <4 x half> %grp.n1691, i64 1, !dbg !27
  %grp.n1695 = extractelement <4 x half> %grp.n1691, i64 0, !dbg !27
  %normalized45 = fmul <4 x float> %row.reciprocal38, %84, !dbg !26
  %grp.n1700 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !27
  %grp.n1701 = extractelement <4 x half> %grp.n1700, i64 3, !dbg !27
  %grp.n1702 = extractelement <4 x half> %grp.n1700, i64 2, !dbg !27
  %grp.n1703 = extractelement <4 x half> %grp.n1700, i64 1, !dbg !27
  %grp.n1704 = extractelement <4 x half> %grp.n1700, i64 0, !dbg !27
  %normalized46 = fmul <4 x float> %row.reciprocal38, %57, !dbg !26
  %grp.n1709 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !27
  %grp.n1710 = extractelement <4 x half> %grp.n1709, i64 3, !dbg !27
  %grp.n1711 = extractelement <4 x half> %grp.n1709, i64 2, !dbg !27
  %grp.n1712 = extractelement <4 x half> %grp.n1709, i64 1, !dbg !27
  %grp.n1713 = extractelement <4 x half> %grp.n1709, i64 0, !dbg !27
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !28
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !28
  store half %grp.n1713, ptr addrspace(1) %address.local1044, align 2, !dbg !28
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !28
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !28
  store half %grp.n1712, ptr addrspace(1) %address.local1047, align 2, !dbg !28
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !28
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !28
  store half %grp.n1711, ptr addrspace(1) %address.local1050, align 2, !dbg !28
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !28
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !28
  store half %grp.n1710, ptr addrspace(1) %address.local1053, align 2, !dbg !28
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !28
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !28
  store half %grp.n1704, ptr addrspace(1) %address.local1056, align 2, !dbg !28
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !28
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !28
  store half %grp.n1703, ptr addrspace(1) %address.local1059, align 2, !dbg !28
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !28
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !28
  store half %grp.n1702, ptr addrspace(1) %address.local1062, align 2, !dbg !28
  %address.local1064 = or i64 %grp.n1677, 51, !dbg !28
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !28
  store half %grp.n1701, ptr addrspace(1) %address.local1065, align 2, !dbg !28
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !28
  store half %grp.n1695, ptr addrspace(1) %address.local1070, align 2, !dbg !28
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !28
  store half %grp.n1694, ptr addrspace(1) %address.local1072, align 2, !dbg !28
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !28
  store half %grp.n1693, ptr addrspace(1) %address.local1074, align 2, !dbg !28
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !28
  store half %grp.n1692, ptr addrspace(1) %address.local1076, align 2, !dbg !28
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !28
  store half %grp.n1686, ptr addrspace(1) %address.local1078, align 2, !dbg !28
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !28
  store half %grp.n1685, ptr addrspace(1) %address.local1080, align 2, !dbg !28
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !28
  store half %grp.n1684, ptr addrspace(1) %address.local1082, align 2, !dbg !28
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !28
  store half %grp.n1683, ptr addrspace(1) %address.local1084, align 2, !dbg !28
  %normalized48 = fmul <4 x float> %row.reciprocal, %129, !dbg !26
  %grp.n1720 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !27
  %grp.n1721 = extractelement <4 x half> %grp.n1720, i64 3, !dbg !27
  %grp.n1722 = extractelement <4 x half> %grp.n1720, i64 2, !dbg !27
  %grp.n1723 = extractelement <4 x half> %grp.n1720, i64 1, !dbg !27
  %grp.n1724 = extractelement <4 x half> %grp.n1720, i64 0, !dbg !27
  %normalized49 = fmul <4 x float> %row.reciprocal, %102, !dbg !26
  %grp.n1729 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !27
  %grp.n1730 = extractelement <4 x half> %grp.n1729, i64 3, !dbg !27
  %grp.n1731 = extractelement <4 x half> %grp.n1729, i64 2, !dbg !27
  %grp.n1732 = extractelement <4 x half> %grp.n1729, i64 1, !dbg !27
  %grp.n1733 = extractelement <4 x half> %grp.n1729, i64 0, !dbg !27
  %normalized51 = fmul <4 x float> %row.reciprocal38, %75, !dbg !26
  %grp.n1738 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !27
  %grp.n1739 = extractelement <4 x half> %grp.n1738, i64 3, !dbg !27
  %grp.n1740 = extractelement <4 x half> %grp.n1738, i64 2, !dbg !27
  %grp.n1741 = extractelement <4 x half> %grp.n1738, i64 1, !dbg !27
  %grp.n1742 = extractelement <4 x half> %grp.n1738, i64 0, !dbg !27
  %normalized52 = fmul <4 x float> %row.reciprocal38, %48, !dbg !26
  %grp.n1747 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !27
  %grp.n1748 = extractelement <4 x half> %grp.n1747, i64 3, !dbg !27
  %grp.n1749 = extractelement <4 x half> %grp.n1747, i64 2, !dbg !27
  %grp.n1750 = extractelement <4 x half> %grp.n1747, i64 1, !dbg !27
  %grp.n1751 = extractelement <4 x half> %grp.n1747, i64 0, !dbg !27
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !28
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !28
  store half %grp.n1751, ptr addrspace(1) %address.local1100, align 2, !dbg !28
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !28
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !28
  store half %grp.n1750, ptr addrspace(1) %address.local1103, align 2, !dbg !28
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !28
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !28
  store half %grp.n1749, ptr addrspace(1) %address.local1106, align 2, !dbg !28
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !28
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !28
  store half %grp.n1748, ptr addrspace(1) %address.local1109, align 2, !dbg !28
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !28
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !28
  store half %grp.n1742, ptr addrspace(1) %address.local1112, align 2, !dbg !28
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !28
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !28
  store half %grp.n1741, ptr addrspace(1) %address.local1115, align 2, !dbg !28
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !28
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !28
  store half %grp.n1740, ptr addrspace(1) %address.local1118, align 2, !dbg !28
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !28
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !28
  store half %grp.n1739, ptr addrspace(1) %address.local1121, align 2, !dbg !28
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !28
  store half %grp.n1733, ptr addrspace(1) %address.local1126, align 2, !dbg !28
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !28
  store half %grp.n1732, ptr addrspace(1) %address.local1128, align 2, !dbg !28
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !28
  store half %grp.n1731, ptr addrspace(1) %address.local1130, align 2, !dbg !28
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !28
  store half %grp.n1730, ptr addrspace(1) %address.local1132, align 2, !dbg !28
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !28
  store half %grp.n1724, ptr addrspace(1) %address.local1134, align 2, !dbg !28
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !28
  store half %grp.n1723, ptr addrspace(1) %address.local1136, align 2, !dbg !28
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !28
  store half %grp.n1722, ptr addrspace(1) %address.local1138, align 2, !dbg !28
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !28
  store half %grp.n1721, ptr addrspace(1) %address.local1140, align 2, !dbg !28
  %normalized54 = fmul <4 x float> %row.reciprocal, %120, !dbg !26
  %grp.n1758 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !27
  %grp.n1759 = extractelement <4 x half> %grp.n1758, i64 3, !dbg !27
  %grp.n1760 = extractelement <4 x half> %grp.n1758, i64 2, !dbg !27
  %grp.n1761 = extractelement <4 x half> %grp.n1758, i64 1, !dbg !27
  %grp.n1762 = extractelement <4 x half> %grp.n1758, i64 0, !dbg !27
  %normalized55 = fmul <4 x float> %row.reciprocal, %93, !dbg !26
  %grp.n1767 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !27
  %grp.n1768 = extractelement <4 x half> %grp.n1767, i64 3, !dbg !27
  %grp.n1769 = extractelement <4 x half> %grp.n1767, i64 2, !dbg !27
  %grp.n1770 = extractelement <4 x half> %grp.n1767, i64 1, !dbg !27
  %grp.n1771 = extractelement <4 x half> %grp.n1767, i64 0, !dbg !27
  %normalized57 = fmul <4 x float> %row.reciprocal38, %66, !dbg !26
  %grp.n1776 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !27
  %grp.n1777 = extractelement <4 x half> %grp.n1776, i64 3, !dbg !27
  %grp.n1778 = extractelement <4 x half> %grp.n1776, i64 2, !dbg !27
  %grp.n1779 = extractelement <4 x half> %grp.n1776, i64 1, !dbg !27
  %grp.n1780 = extractelement <4 x half> %grp.n1776, i64 0, !dbg !27
  %normalized58 = fmul <4 x float> %row.reciprocal38, %39, !dbg !26
  %grp.n1785 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !27
  %grp.n1786 = extractelement <4 x half> %grp.n1785, i64 3, !dbg !27
  %grp.n1787 = extractelement <4 x half> %grp.n1785, i64 2, !dbg !27
  %grp.n1788 = extractelement <4 x half> %grp.n1785, i64 1, !dbg !27
  %grp.n1789 = extractelement <4 x half> %grp.n1785, i64 0, !dbg !27
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !28
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !28
  store half %grp.n1789, ptr addrspace(1) %address.local1156, align 2, !dbg !28
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !28
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !28
  store half %grp.n1788, ptr addrspace(1) %address.local1159, align 2, !dbg !28
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !28
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !28
  store half %grp.n1787, ptr addrspace(1) %address.local1162, align 2, !dbg !28
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !28
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !28
  store half %grp.n1786, ptr addrspace(1) %address.local1165, align 2, !dbg !28
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !28
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !28
  store half %grp.n1780, ptr addrspace(1) %address.local1168, align 2, !dbg !28
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !28
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !28
  store half %grp.n1779, ptr addrspace(1) %address.local1171, align 2, !dbg !28
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !28
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !28
  store half %grp.n1778, ptr addrspace(1) %address.local1174, align 2, !dbg !28
  %address.local1176 = or i64 %grp.n1677, 115, !dbg !28
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !28
  store half %grp.n1777, ptr addrspace(1) %address.local1177, align 2, !dbg !28
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !28
  store half %grp.n1771, ptr addrspace(1) %address.local1182, align 2, !dbg !28
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !28
  store half %grp.n1770, ptr addrspace(1) %address.local1184, align 2, !dbg !28
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !28
  store half %grp.n1769, ptr addrspace(1) %address.local1186, align 2, !dbg !28
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !28
  store half %grp.n1768, ptr addrspace(1) %address.local1188, align 2, !dbg !28
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !28
  store half %grp.n1762, ptr addrspace(1) %address.local1190, align 2, !dbg !28
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !28
  store half %grp.n1761, ptr addrspace(1) %address.local1192, align 2, !dbg !28
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !28
  store half %grp.n1760, ptr addrspace(1) %address.local1194, align 2, !dbg !28
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !28
  store half %grp.n1759, ptr addrspace(1) %address.local1196, align 2, !dbg !28
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_frag_prefetch_forward/control")
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
!14 = !DILocation(line: 86, column: 20, scope: !3)
!15 = !DILocation(line: 71, column: 23, scope: !3)
!16 = !DILocation(line: 74, column: 20, scope: !3)
!17 = !DILocation(line: 75, column: 22, scope: !3)
!18 = !DILocation(line: 76, column: 23, scope: !3)
!19 = !DILocation(line: 77, column: 20, scope: !3)
!20 = !DILocation(line: 88, column: 22, scope: !3)
!21 = !DILocation(line: 78, column: 22, scope: !3)
!22 = !DILocation(line: 79, column: 22, scope: !3)
!23 = !DILocation(line: 81, column: 16, scope: !3)
!24 = !DILocation(line: 80, column: 20, scope: !3)
!25 = !DILocation(line: 87, column: 23, scope: !3)
!26 = !DILocation(line: 92, column: 23, scope: !3)
!27 = !DILocation(line: 93, column: 19, scope: !3)
!28 = !DILocation(line: 94, column: 9, scope: !3)
